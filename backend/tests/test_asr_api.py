"""The API speech-recognition engine: windowing, parsing, validation.

Not one real request is made here. The fake endpoint below reproduces what
the real one was measured doing — the `fake_rapidocr` lesson: a stand-in
that is politer than the real server tests our imagination, not the code.
"""

from __future__ import annotations

import base64
import re

import numpy as np
import pytest

from app.models.schemas import ASRSettings, LLMSettings
from app.services import asr
from app.services import asr_api as A

SR = 16_000


# ------------------------------------------------------------- the fake


def cue_lines(seconds: float, step: float = 10.0, text="line", rate: float = 0.0):
    """A plausible transcript: cues every `step` seconds across the window.

    `rate` compresses the clock, the way a drifting model does — 0.08 means
    it reports everything 8% early.
    """
    out = []
    true_time = 1.0
    index = 0
    while true_time < seconds - 1.5:
        said = true_time * (1 - rate)
        out.append(f"[{_clock(said)} --> {_clock(min(said + 3.0, seconds))}] "
                   f"{text} {index + 1}")
        true_time += step
        index += 1
    return "\n".join(out)


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


class FakeAudio:
    """The endpoint, with the quirks the real one was measured to have.

    * replies in `[MM:SS.mmm --> MM:SS.mmm]`, never decimal seconds;
    * `usage.prompt_tokens` carries 25 tokens per second of audio — the
      only evidence that the audio arrived at all;
    * `max_tokens` is ignored and `finish_reason` is always "stop", so a
      truncated reply looks exactly like a complete one.

    `replies` is a script; the last entry sticks once it runs out, so "this
    endpoint is simply broken" is one entry rather than a guess at the call
    count. An Exception in the script is raised. A callable is handed the
    window length and returns the reply text.
    """

    def __init__(self, replies=None, audio_tokens=True, lang="ja"):
        self.replies = list(replies) if replies is not None else None
        self.audio_tokens = audio_tokens
        self.lang = lang
        self.calls = []
        self.seconds = []
        self.parts = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature, **kw):
        content = messages[0]["content"]
        prompt = content[0]["text"]
        part = content[1]
        seconds = float(re.search(r"这段音频长 ([\d.]+) 秒", prompt).group(1))
        self.calls.append(messages)
        self.seconds.append(seconds)
        self.parts.append(part)

        if self.replies is None:
            reply = cue_lines(seconds)
        else:
            reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(seconds)
        if self.lang and not reply.startswith("#lang"):
            reply = f"#lang: {self.lang}\n{reply}"

        class Obj:
            pass

        usage = Obj()
        usage.prompt_tokens = (
            int(A.AUDIO_TOKENS_PER_SECOND * seconds) + 272 if self.audio_tokens else 272
        )
        usage.completion_tokens = 120
        message, choice, resp = Obj(), Obj(), Obj()
        message.content = reply
        choice.message = message
        choice.finish_reason = "stop"  # even when it truncated
        resp.choices = [choice]
        resp.usage = usage
        return resp


def tone(seconds: float, freq=220.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


@pytest.fixture(scope="module")
def wav(tmp_path_factory):
    """A real WAV on disk — decode_audio has to be able to read it."""
    path = tmp_path_factory.mktemp("asr_api") / "audio.wav"
    path.write_bytes(A.encode_samples(tone(600.0), "wav"))
    return path


@pytest.fixture
def run(monkeypatch, wav):
    """Drive asr_api.transcribe with controlled VAD intervals.

    Silero's opinion of a sine wave is not the subject of any test here, and
    pinning the intervals is what makes the window plan predictable.
    """
    def _run(client=None, intervals=None, settings=None, language=None, **kwargs):
        spans = intervals if intervals is not None else [
            (t, t + 8.0) for t in range(0, 600, 10)
        ]
        monkeypatch.setattr(A, "cut_intervals", lambda audio: spans)
        monkeypatch.setattr(asr, "speech_intervals_of", lambda audio, s: spans)
        logged: list[str] = []
        segments, detected = A.transcribe(
            str(wav),
            # wav by default: these tests are not about the encoder, and mp3
            # costs eight times as much per window
            settings or ASRSettings(engine="api", api_window_seconds=300.0,
                                    api_concurrency=1, api_audio_format="wav"),
            LLMSettings(base_url="http://x/v1", api_key="k", model="m"),
            language=language,
            log=logged.append,
            client=client if client is not None else FakeAudio(),
            **kwargs,
        )
        return segments, detected, logged
    return _run


# ------------------------------------------------------- window planning


def test_the_windows_cover_the_whole_timeline_with_no_gaps():
    """The guarantee the whole design rests on: the VAD chooses where the
    seams go, never what gets left out."""
    intervals = [(t, t + 8.0) for t in range(0, 600, 10)]
    windows, forced = A.plan_windows(intervals, 600.0, 300.0)

    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(600.0)
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))
    assert sum(b - a for a, b in windows) == pytest.approx(600.0)
    assert forced == 0


def test_a_cut_lands_in_silence_not_in_the_middle_of_a_line():
    intervals = [(t, t + 8.0) for t in range(0, 600, 10)]
    windows, _ = A.plan_windows(intervals, 600.0, 300.0)
    cut = windows[0][1]
    assert not any(start < cut < end for start, end in intervals)


def test_the_longest_silence_in_the_band_wins():
    """A three-second pause is a scene boundary; a 0.4s one may be inside a
    sentence. Both are candidates, and the long one should win."""
    intervals = [(0.0, 140.0), (140.4, 160.0), (163.0, 600.0)]
    windows, _ = A.plan_windows(intervals, 600.0, 300.0)
    assert windows[0][1] == pytest.approx(161.5)  # the middle of the 3s gap


def test_speech_the_vad_scores_below_its_threshold_is_still_inside_a_window():
    """Dialogue under music scores zero with silero and is transcribed fine
    by this engine — so a stretch the VAD calls silent may never be dropped."""
    intervals = [(0.0, 100.0), (500.0, 600.0)]  # silero hears nothing 100-500
    windows, _ = A.plan_windows(intervals, 600.0, 300.0)

    covered = [t for t in (150.0, 250.0, 380.0, 499.0)
               if any(a <= t < b for a, b in windows)]
    assert covered == [150.0, 250.0, 380.0, 499.0]


def test_a_stretch_with_no_silence_at_all_is_cut_at_the_time_limit():
    windows, forced = A.plan_windows([(0.0, 1200.0)], 1200.0, 300.0)
    assert forced >= 1
    assert max(b - a for a, b in windows) <= A.WINDOW_HARD_MAX
    assert windows[-1][1] == pytest.approx(1200.0)


def test_no_speech_anywhere_still_covers_the_film():
    windows, forced = A.plan_windows([], 900.0, 300.0)
    assert windows[0][0] == 0.0 and windows[-1][1] == pytest.approx(900.0)
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))


def test_a_window_is_never_longer_than_the_hard_maximum():
    windows, _ = A.plan_windows([(0.0, 3000.0)], 3000.0, 420.0)
    assert max(b - a for a, b in windows) <= A.WINDOW_HARD_MAX


# ------------------------------------------------------------- the request


def test_each_window_is_sent_as_an_input_audio_part(run):
    client = FakeAudio()
    run(client, settings=ASRSettings(engine="api", api_window_seconds=300.0,
                                     api_concurrency=1))

    part = client.parts[0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "mp3"
    data = base64.b64decode(part["input_audio"]["data"])
    assert data[:3] == b"ID3"


def test_wav_is_available_for_an_endpoint_that_will_not_take_mp3(run):
    client = FakeAudio()
    run(client)
    assert client.parts[0]["input_audio"]["format"] == "wav"
    assert base64.b64decode(client.parts[0]["input_audio"]["data"])[:4] == b"RIFF"


def test_an_encoded_window_really_holds_that_many_seconds(tmp_path):
    from faster_whisper.audio import decode_audio

    path = tmp_path / "w.mp3"
    path.write_bytes(A.encode_window(tone(120.0), 30.0, 90.0, "mp3"))
    decoded = decode_audio(str(path), sampling_rate=SR)
    assert len(decoded) / SR == pytest.approx(60.0, abs=0.5)


def test_times_are_relative_in_the_request_and_absolute_in_the_result(run):
    """Every window starts its own clock at zero; the offset is added here,
    which is why a model whose clock slips cannot drift across a film."""
    client = FakeAudio(replies=["[00:01.000 --> 00:02.000] 台詞"])
    # speech only at the very start of each window, so one short line is a
    # complete transcript of it rather than a truncated one
    segments, _, _ = run(client, intervals=[(0.0, 2.0), (330.0, 332.0)])

    # windows tile, so each one begins where the previous ended — and every
    # cue came back at 00:01 of its own window
    offsets = [sum(client.seconds[:i]) for i in range(len(client.seconds))]
    assert sorted(s.start for s in segments) == pytest.approx(
        [offset + 1.0 for offset in offsets]
    )
    assert "这段音频长" in client.calls[0][0]["content"][0]["text"]


def test_the_proper_noun_hint_and_the_language_reach_the_prompt(run):
    client = FakeAudio()
    run(client, language="ja",
        settings=ASRSettings(engine="api", api_window_seconds=300.0,
                             api_concurrency=1, initial_prompt="佐藤健一"))
    prompt = client.calls[0][0]["content"][0]["text"]
    assert "佐藤健一" in prompt
    assert "ja" in prompt


# --------------------------------------------------------------- parsing


@pytest.mark.parametrize("line", [
    "[00:03.120 --> 00:05.480] 台詞",
    "  [00:03,120 -> 00:05,480] 台詞",
    "- [0:00:03.120 --> 0:00:05.480] 台詞",
    "* 00:03.12 → 00:05.48 台詞",
    "[3.12 --> 5.48] 台詞",
])
def test_the_reply_parser_tolerates_the_usual_model_noise(line):
    _, cues = A.parse_reply(f"```\n#lang: ja\n{line}\n```")
    assert len(cues) == 1
    assert cues[0][0] == pytest.approx(3.12, abs=0.01)
    assert cues[0][2] == "台詞"


def test_prose_is_not_mistaken_for_a_transcript():
    reply = "很抱歉，我无法转写这段音频里的内容，因为它可能涉及受版权保护的材料。"
    _, cues = A.parse_reply(reply)
    assert cues == []
    assert A.looks_like_prose(reply, cues)
    assert not A.looks_like_prose("NO SPEECH", [])


def test_the_language_tag_is_read_and_not_mistaken_for_a_cue():
    language, cues = A.parse_reply("#lang: JA\n[00:01.000 --> 00:02.000] はい")
    assert language == "ja"
    assert len(cues) == 1


# ------------------------------------------------------------ validation


def test_a_line_past_the_window_end_is_clamped():
    kept, fatal, notes = A.validate([(295.0, 300.6, "a")], 300.0, 60.0)
    assert not fatal
    assert kept[0][1] == pytest.approx(300.0)


def test_a_line_far_past_the_window_end_is_dropped():
    kept, fatal, notes = A.validate(
        [(10.0, 12.0, "a"), (400.0, 402.0, "b")], 300.0, 60.0)
    assert [c[2] for c in kept] == ["a"]
    assert any("之外" in n for n in notes)


def test_times_running_backwards_fail_the_window():
    cues = [(float(200 - i), float(201 - i), f"l{i}") for i in range(10)]
    kept, fatal, _ = A.validate(cues, 300.0, 60.0)
    assert fatal and not kept


def test_a_small_overlap_is_trimmed_rather_than_rejected():
    kept, fatal, _ = A.validate([(1.0, 3.0, "a"), (2.9, 5.0, "b")], 300.0, 60.0)
    assert not fatal
    assert kept[0][1] == pytest.approx(2.9)


def test_a_decode_loop_is_caught_by_the_repeat_run():
    cues = [(float(i), float(i) + 1.0, "同じ台詞") for i in range(10)]
    kept, fatal, _ = A.validate(cues, 300.0, 60.0)
    assert fatal and "陷环" in fatal[0]


def test_a_line_with_nothing_readable_is_dropped():
    kept, _, _ = A.validate([(1.0, 2.0, "---"), (3.0, 4.0, "ん")], 300.0, 60.0)
    assert [c[2] for c in kept] == ["ん"]


def test_an_empty_answer_is_only_a_problem_where_there_is_speech():
    assert A.validate([], 300.0, 60.0)[1]       # silero is sure there was speech
    assert not A.validate([], 300.0, 0.5)[1]    # ...and here it is not


def test_the_audio_token_count_is_what_proves_the_audio_arrived():
    class Usage:
        prompt_tokens = 272  # text only: the relay dropped the audio part

    assert A.audio_reached_the_model(Usage(), 300.0) is False
    Usage.prompt_tokens = 7772
    assert A.audio_reached_the_model(Usage(), 300.0) is True
    assert A.audio_reached_the_model(None, 300.0) is None


# ----------------------------------------------------- the transcribe loop


def test_a_silent_window_costs_one_request_and_no_retry(run):
    client = FakeAudio(replies=["NO SPEECH"])
    segments, _, logged = run(client, intervals=[(0.0, 1.0)])

    assert segments == []
    assert len(client.calls) == 2  # two windows, one call each
    assert any("无人说话" in line for line in logged)


def test_a_disputed_silence_is_believed_not_argued_with(run):
    """Silero says there is a minute of speech and the model says there is
    none. The model wins — measured: pressing it over a stretch of score
    produced fifteen lines of dialogue that was never spoken, while silero
    was the one hearing things. The disagreement is logged, not obeyed."""
    client = FakeAudio(replies=["NO SPEECH"])
    segments, _, logged = run(client, intervals=[(0.0, 300.0)])

    assert segments == []
    assert len(client.calls) == 2  # one per window, no argument
    assert any("采信模型" in line for line in logged)


def test_a_window_that_will_not_verify_is_halved_and_retried(run):
    backwards = "\n".join(
        f"[{_clock(200 - i)} --> {_clock(201 - i)}] l{i}" for i in range(10)
    )
    client = FakeAudio(replies=[backwards, backwards, lambda s: cue_lines(s)])
    _, _, logged = run(client)
    assert any("拆成两段再试" in line for line in logged)


def test_an_endpoint_that_never_answers_fails_the_job(run):
    """One bad stretch is a gap; an endpoint refusing everything is not.

    The stage no longer dies on the first window it cannot get through —
    it names the stretch and carries on — but a run that loses more than
    MAX_MISSING_SHARE of the film has nothing worth delivering, and says
    which ranges are missing rather than just that something went wrong.
    """
    client = FakeAudio(replies=[RuntimeError("connection reset")])
    with pytest.raises(RuntimeError) as caught:
        run(client)
    said = str(caught.value)
    assert "没能转出来" in said
    assert "本地识别引擎" in said
    assert ":" in said          # the missing ranges, named


def test_one_bad_stretch_does_not_lose_the_rest_of_the_film(run):
    """The whole point: ninety good minutes are not thrown away because
    one stretch is unreadable."""
    class OneBadStretch(FakeAudio):
        def create(self, model, messages, temperature, **kw):
            resp = super().create(model, messages, temperature, **kw)
            # the window planner puts a seam at 300s; refuse only what
            # starts inside the last stretch of the film
            if "这段音频长" in messages[0]["content"][0]["text"]:
                seconds = self.seconds[-1]
                if seconds <= 60.0:      # only the smallest splits fail
                    raise RuntimeError("connection reset")
            return resp

    client = OneBadStretch()
    segments, _, logged = run(client)
    assert segments, "the windows that answered should still be here"


def test_three_prose_replies_in_a_row_are_called_a_refusal(run):
    prose = "抱歉，我不能转写这段音频，它似乎包含受保护的内容，请换一段音频试试。"
    client = FakeAudio(replies=[prose])
    with pytest.raises(RuntimeError) as caught:
        run(client)
    assert "拒绝转写" in str(caught.value)


def test_a_server_that_answers_without_any_choices_says_what_it_said(run):
    class NoChoices(FakeAudio):
        def create(self, model, messages, temperature, **kw):
            super().create(model, messages, temperature, **kw)

            class Resp:
                choices = None
                error = {"message": "model not loaded"}

            return Resp()

    client = NoChoices()
    with pytest.raises(RuntimeError) as caught:
        run(client)
    # what the server said is the whole point of the test: an empty choices
    # list with no usage behind it is a server naming its own problem, not
    # the billed-but-empty fault, so the advice it gets is the ordinary one
    assert "model not loaded" in str(caught.value)
    assert "已知故障" not in str(caught.value)


def test_an_endpoint_that_drops_the_audio_is_named_as_such(run):
    client = FakeAudio(audio_tokens=False)
    with pytest.raises(RuntimeError):
        run(client)


def test_the_language_comes_from_whichever_window_answered(run):
    client = FakeAudio(replies=["NO SPEECH", lambda s: cue_lines(s)], lang="ja")
    _, detected, _ = run(client)
    assert detected == "ja"


def test_a_forced_language_is_returned_unchanged(run):
    client = FakeAudio(lang="en")
    _, detected, _ = run(client, language="ja")
    assert detected == "ja"


def test_cancelling_stops_the_run(run):
    with pytest.raises(InterruptedError):
        run(FakeAudio(), should_cancel=lambda: True)


def test_the_engine_says_the_second_pass_does_not_apply(run):
    _, _, logged = run(FakeAudio())
    assert any("不使用二次识别" in line for line in logged)


def test_segments_come_back_sorted_and_without_word_timestamps(run):
    segments, _, _ = run(FakeAudio())
    assert segments == sorted(segments, key=lambda s: s.start)
    assert all(s.words == [] and not s.recovered for s in segments)


def test_coverage_is_reported_without_word_timestamps(run):
    _, _, logged = run(FakeAudio())
    assert any("覆盖率" in line or "偏乐观" in line for line in logged)


def test_the_tokens_it_spent_are_reported(run):
    spent: dict = {}
    run(FakeAudio(), usage=spent)
    assert spent["calls"] >= 2
    assert spent["prompt"] > 0


# ------------------------------------------------------------ diagnostics


def test_a_clock_that_slides_fails_the_stage(run):
    """The interlock. This model does not drift; others are documented to,
    and a subtitle track that is approximately right is worse than none."""
    intervals = [(t, t + 4.0) for t in range(0, 600, 5)]
    client = FakeAudio(replies=[lambda s: cue_lines(s, step=5.0, rate=0.08)])
    short = ASRSettings(engine="api", api_window_seconds=60.0, api_concurrency=1)

    with pytest.raises(RuntimeError) as caught:
        run(client, intervals=intervals, settings=short)
    assert "时间戳与音频持续对不上" in str(caught.value)


def test_hearing_more_than_the_vad_is_not_called_drift(run):
    """The measured case on a real film: the model's last line lands after
    silero's last speech, because it heard dialogue under the music. That is
    the engine's advantage, not a clock problem — a warning there would fire
    on every film with a score."""
    intervals = [(t, t + 4.0) for t in range(0, 600, 5)]

    def past_the_vad(seconds):
        # transcribes right to the end of the window, past the last interval
        return "\n".join(
            f"[{_clock(t)} --> {_clock(min(t + 4.0, seconds))}] line {int(t)}"
            for t in range(1, int(seconds) - 2, 5)
        )

    segments, _, logged = run(FakeAudio(replies=[past_the_vad]),
                              intervals=intervals,
                              settings=ASRSettings(engine="api", api_concurrency=1,
                                                   api_window_seconds=60.0,
                                                   api_audio_format="wav"))
    assert segments
    assert not any("漂移" in line for line in logged)


def test_a_steady_clock_passes_the_interlock(run):
    intervals = [(t, t + 4.0) for t in range(0, 600, 5)]
    client = FakeAudio(replies=[lambda s: cue_lines(s, step=5.0)])
    short = ASRSettings(engine="api", api_window_seconds=60.0, api_concurrency=1)

    segments, _, logged = run(client, intervals=intervals, settings=short)
    assert segments
    assert not any("漂移" in line or "对不上" in line for line in logged)


def test_the_alignment_correlation_is_the_same_check_the_translator_makes():
    aligned = [(10.0, 100.0), (20.0, 210.0), (30.0, 290.0), (40.0, 400.0),
               (50.0, 520.0), (60.0, 600.0)]
    scrambled = [(10.0, 600.0), (20.0, 100.0), (30.0, 520.0), (40.0, 210.0),
                 (50.0, 290.0), (60.0, 400.0)]
    assert A.correlation(aligned) > A.ALIGNMENT_MIN_CORRELATION
    assert A.correlation(scrambled) < A.ALIGNMENT_MIN_CORRELATION
    assert A.correlation([(1.0, 2.0)]) is None


def test_the_split_point_prefers_a_silence_near_the_middle():
    intervals = [(0.0, 140.0), (145.0, 300.0)]
    assert A.split_point(intervals, 0.0, 300.0) == pytest.approx(142.5)
    assert A.split_point([(0.0, 300.0)], 0.0, 300.0) == pytest.approx(150.0)


# -------------------------------------------------------------- dispatch


def test_the_api_engine_never_loads_whisper(monkeypatch, wav):
    monkeypatch.setattr(asr, "_get_model",
                        lambda *a, **k: pytest.fail("whisper loaded"))
    monkeypatch.setattr(A, "cut_intervals", lambda audio: [(0.0, 600.0)])
    monkeypatch.setattr(asr, "speech_intervals_of", lambda audio, s: [(0.0, 600.0)])

    segments, detected = asr.transcribe(
        str(wav), ASRSettings(engine="api", api_concurrency=1),
        llm=LLMSettings(model="m"), client=FakeAudio(),
    )
    assert segments and detected == "ja"


def test_the_api_engine_refuses_to_run_without_llm_settings(wav):
    with pytest.raises(ValueError) as caught:
        asr.transcribe(str(wav), ASRSettings(engine="api"))
    assert "需要先配置 LLM" in str(caught.value)


def test_the_local_engine_still_goes_to_whisper(monkeypatch, wav):
    called = []
    monkeypatch.setattr(asr, "_get_model",
                        lambda *a, **k: called.append(1) or pytest.fail("ok"))
    with pytest.raises(BaseException):
        asr.transcribe(str(wav), ASRSettings(engine="local"))
    assert called


# ------------------------------------------------- the settings-page probe


def probe(monkeypatch, reply, settings_file, prompt_tokens=None, seconds=4.0):
    """Run POST /api/settings/test-asr-api against a scripted endpoint."""
    from app.main import app as fastapi_app
    from tests.conftest import local_client

    class Client:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.sent = None

        def create(self, model, messages, temperature, **kw):
            self.sent = messages
            if isinstance(reply, Exception):
                raise reply

            class Obj:
                pass

            usage = Obj()
            usage.prompt_tokens = (
                prompt_tokens if prompt_tokens is not None
                else int(A.AUDIO_TOKENS_PER_SECOND * seconds) + 40
            )
            msg, choice, resp = Obj(), Obj(), Obj()
            msg.content = reply
            choice.message = msg
            resp.choices = [choice]
            resp.usage = usage
            return resp

    client = Client()
    monkeypatch.setattr("app.services.translator.make_audio_client",
                        lambda *a, **k: client)
    settings_file()
    body = local_client(fastapi_app).post(
        "/api/settings/test-asr-api",
        json={"base_url": "http://x/v1", "model": "m", "audio_model": "listener"},
    ).json()
    return body, client


def test_the_probe_sends_real_audio_and_checks_it_was_heard(monkeypatch,
                                                            settings_file):
    """A text ping proves nothing here either. Measured: this model cannot
    count beeps (it said 5 for two seconds of silence) but never misses the
    direction of a sweep — so that is what the probe asks."""
    monkeypatch.setattr("random.choice", lambda options: True)  # rising
    body, client = probe(monkeypatch, "升高", settings_file)

    assert body["ok"] and body["heard_it"]
    assert body["model"] == "listener"
    assert body["asked"] == "升高"
    part = client.sent[0]["content"][1]
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "mp3"
    assert base64.b64decode(part["input_audio"]["data"])[:3] == b"ID3"


def test_the_probe_reports_a_relay_that_swallowed_the_audio(monkeypatch,
                                                            settings_file):
    """Measured on a real endpoint: strip the audio and the model still
    answers, fluently. Only the token count gives it away."""
    monkeypatch.setattr("random.choice", lambda options: True)
    body, _ = probe(monkeypatch, "升高", settings_file, prompt_tokens=40)
    assert body["ok"] and body["carried_audio"] is False


def test_a_model_that_hears_nothing_is_not_called_working(monkeypatch,
                                                          settings_file):
    monkeypatch.setattr("random.choice", lambda options: True)
    body, _ = probe(monkeypatch, "降低", settings_file)
    assert body["ok"] and not body["heard_it"]


def test_an_answer_naming_both_directions_does_not_count(monkeypatch,
                                                         settings_file):
    monkeypatch.setattr("random.choice", lambda options: True)
    body, _ = probe(monkeypatch, "可能是升高，也可能是降低", settings_file)
    assert not body["heard_it"]


def test_a_dead_audio_endpoint_answers_ok_false_inside_a_200(monkeypatch,
                                                             settings_file):
    body, _ = probe(monkeypatch, RuntimeError("connection refused"), settings_file)
    assert body["ok"] is False
    assert "connection refused" in body["error"]
    assert body["endpoint"] == "http://x/v1"


# ------------------------------------------------- where a cue really starts
#
# This engine has no word-level timestamps, so segmenter._trustworthy_start
# never sees its cues and a start that lands a second early stays there.
# Measured on a Japanese DVD source: where the two engines disagreed about a
# start by more than 1.5s, the audio sided with whisper 21 times to 13.
# snap_start moves the start forward onto the sound — forward only, never
# past MAX_START_MOVE, and never past what the cue can spare.


def silence(seconds: float):
    return np.zeros(int(SR * seconds), dtype="float32")


def quiet(seconds: float, db_under: float, freq=220.0):
    """A tone this many dB below the 0.3-amplitude one `tone()` makes."""
    return tone(seconds, freq=freq, amp=0.3 * (10 ** (-db_under / 20.0)))


def bursts(step=10.0, lead=1.0, length=3.0, count=60):
    """Silence with a burst of sound `lead` seconds into every `step`."""
    out = []
    for _ in range(count):
        out.append(silence(lead))
        out.append(tone(length))
        out.append(silence(step - lead - length))
    return np.concatenate(out)


def test_a_start_in_silence_is_moved_to_just_before_the_sound():
    audio = np.concatenate([silence(1.0), tone(2.0)])
    moved, distance, why = A.snap_start((0.0, 3.0, "x"), audio)
    assert why == "moved"
    assert moved[0] == pytest.approx(1.0 - A.ONSET_LEAD, abs=0.04)
    assert distance == pytest.approx(0.9, abs=0.04)
    assert moved[1] == 3.0 and moved[2] == "x"  # end and text untouched


def test_a_start_already_on_speech_is_left_alone():
    audio = tone(3.0)
    moved, distance, why = A.snap_start((0.0, 3.0, "x"), audio)
    assert (moved, distance, why) == ((0.0, 3.0, "x"), 0.0, "on_speech")


def test_a_soft_first_syllable_is_speech_not_silence():
    """8 dB under the loudest part is a quiet syllable; 12 dB is the line."""
    audio = np.concatenate([quiet(0.5, db_under=8.0), tone(2.5)])
    _, _, why = A.snap_start((0.0, 3.0, "x"), audio)
    assert why == "on_speech"


def test_a_start_is_never_moved_earlier():
    rng = np.random.default_rng(3)
    audio = np.concatenate([silence(1.0), tone(1.0), silence(1.0), tone(1.0)])
    for _ in range(50):
        start = float(rng.uniform(0.0, 3.0))
        end = start + float(rng.uniform(0.6, 1.0))
        moved, distance, _ = A.snap_start((start, end, "x"), audio)
        assert moved[0] >= start
        assert distance >= 0.0


def test_sound_that_starts_past_the_limit_is_refused_not_clamped():
    """Beyond MAX_START_MOVE the cue is misplaced, not late — do not guess."""
    audio = np.concatenate([silence(3.0), tone(7.0)])
    moved, distance, why = A.snap_start((0.0, 10.0, "x"), audio)
    assert why == "beyond_bound"
    assert moved[0] == 0.0 and distance == 0.0


def test_the_cue_keeps_enough_length_to_be_read():
    """Moving must not leave a cue shorter than segmenter.MIN_LINE_DURATION."""
    audio = np.concatenate([silence(1.2), tone(0.4)])
    moved, _, why = A.snap_start((0.0, 1.4, "x"), audio)
    assert why == "moved"
    assert moved[0] == pytest.approx(0.9, abs=1e-6)  # 1.4 - 0.5
    assert moved[1] - moved[0] == pytest.approx(0.5, abs=1e-6)


def test_a_cue_with_nothing_in_it_is_left_alone_and_counted():
    cues, stats = A.snap_starts([(0.0, 2.0, "x")], silence(3.0))
    assert cues == [(0.0, 2.0, "x")]
    assert (stats.silent, stats.moved, stats.total) == (1, [], 1)


def test_one_frame_of_noise_is_not_an_onset():
    """A door click reaches the threshold for a single frame and no more."""
    audio = np.concatenate([
        silence(0.3), tone(A.ONSET_FRAME), silence(0.68 - A.ONSET_FRAME), tone(2.0),
    ])
    moved, _, why = A.snap_start((0.0, 3.0, "x"), audio)
    assert why == "moved"
    assert moved[0] == pytest.approx(1.0 - A.ONSET_LEAD, abs=0.05)


def test_neighbours_stay_in_order_after_snapping():
    audio = np.concatenate([silence(1.0), tone(1.0), silence(1.0), tone(1.0)])
    cues = [(0.0, 2.0, "a"), (2.0, 4.0, "b")]
    moved, stats = A.snap_starts(cues, audio)
    assert stats.total == 2
    assert all(b[0] >= a[1] - 1e-9 or b[0] > a[0] for a, b in zip(moved, moved[1:]))
    assert all(cue[0] < cue[1] for cue in moved)
    assert [m[0] for m in moved] == sorted(m[0] for m in moved)


def test_snapping_reports_what_it_did():
    audio = np.concatenate([silence(1.0), tone(2.0)])
    _, stats = A.snap_starts([(0.0, 3.0, "a"), (1.0, 3.0, "b")], audio)
    assert "1/2 行" in stats.describe()
    assert "中位 +0.90s" in stats.describe()


@pytest.fixture(scope="module")
def bursty_wav(tmp_path_factory):
    """600s where every 10s block is silent for 1s, then sounds for 3s."""
    path = tmp_path_factory.mktemp("asr_api_bursts") / "bursts.wav"
    path.write_bytes(A.encode_samples(bursts(), "wav"))
    return path


def test_a_run_moves_its_starts_onto_the_sound(monkeypatch, bursty_wav):
    """End to end: the model says 0-4s, the sound is at 1-4s, the cue lands
    just before the sound rather than a second into silence."""
    spans = [(t + 1.0, t + 4.0) for t in range(0, 600, 10)]
    monkeypatch.setattr(A, "cut_intervals", lambda audio: spans)
    monkeypatch.setattr(asr, "speech_intervals_of", lambda audio, s: spans)
    # windows pinned to the burst grid so the fake reply, which only knows a
    # window's length, can place its cues where the sound actually is
    monkeypatch.setattr(A, "plan_windows",
                        lambda intervals, duration, target: ([(0.0, 300.0),
                                                              (300.0, 600.0)], 0))

    def reply(seconds):
        return "\n".join(
            f"[{_clock(t)} --> {_clock(t + 4.0)}] line {int(t // 10) + 1}"
            for t in np.arange(0.0, seconds - 4.0, 10.0)
        )

    logged: list[str] = []
    segments, _ = A.transcribe(
        str(bursty_wav),
        ASRSettings(engine="api", api_window_seconds=300.0, api_concurrency=1,
                    api_audio_format="wav"),
        LLMSettings(base_url="http://x/v1", api_key="k", model="m"),
        log=logged.append, client=FakeAudio(replies=[reply]),
    )
    assert segments
    offsets = [s.start % 10.0 for s in segments]
    assert all(abs(o - 0.9) < 0.08 for o in offsets), offsets[:5]
    assert any("起点后移" in line for line in logged)


def test_a_run_on_continuous_speech_moves_nothing(run):
    """The old fixture is a solid tone: every start is already on sound."""
    segments, _, logged = run()
    assert segments
    assert not any("起点后移" in line for line in logged)


def test_the_prompt_asks_for_sentence_punctuation():
    """Two Japanese films came back 72% / 57% open-ended, which flips the
    segmenter onto its "defer every merge to refine" path."""
    prompt = A.build_prompt(300.0)
    assert "句末标点" in prompt
    # the half that matters: a full stop on every line would make
    # open_ended_ratio zero and leave the merger nothing to work with
    assert "只有句子真正结束的那一行以句末标点结尾" in prompt


def test_asking_for_punctuation_does_not_ask_for_note_marks():
    prompt = A.build_prompt(300.0)
    assert "不要加 ♪" in prompt


# ------------------------------------------- when silero is the thing that broke
#
# Every diagnostic in _report compares this engine against silero. On the
# VHS capture all three went off at once — 40.4 chars per speech-second,
# r=+0.61, -266 ms/s — measured against a VAD that had found 3.5% of the
# film. The numbers stay in the debug log; what stops is calling them
# faults of the transcript.


def drifting_records(count=6, window=300.0, fast_ms=200.0):
    """Windows whose transcript ends earlier and earlier: a fast clock."""
    return [
        {"window": (i * window, (i + 1) * window), "attempts": 1, "no_speech": False,
         "speech": window * 0.8, "speech_end": window,
         "last_cue_end": window - window * fast_ms / 1000.0,
         "cues": [(0.0, 1.0, "x")], "reply": ""}
        for i in range(count)
    ]


def lopsided_windows():
    """Wildly different chars-per-speech-second, and no correlation."""
    return [(100.0, 10.0, 0), (100.0, 4000.0, 0), (100.0, 20.0, 0),
            (50.0, 3000.0, 0), (200.0, 30.0, 0), (10.0, 900.0, 0)]


def test_a_blind_vad_silences_the_three_diagnostics_that_rest_on_it():
    logged: list[str] = []
    A._report([], [], lopsided_windows(), drifting_records(), 0, logged.append,
              None, vad_blind=True)
    assert not any("字数/语音秒" in line for line in logged)
    assert not any("相关性" in line for line in logged)
    assert not any("系统性漂移" in line for line in logged)


def test_the_same_run_with_a_working_vad_still_warns():
    logged: list[str] = []
    with pytest.raises(RuntimeError):
        A._report([], [], lopsided_windows(), drifting_records(), 0, logged.append,
                  None, vad_blind=False)
    assert any("字数/语音秒" in line for line in logged)
    assert any("相关性" in line for line in logged)


def test_a_blind_vad_downgrades_the_drift_interlock_to_a_note():
    """Invariant 3 assumes silero's last speech is where speech ends. On a
    blind source that assumption has just been measured false, so the slope
    is reported and not obeyed."""
    logged: list[str] = []
    A._report([], [], [], drifting_records(), 0, logged.append, None,
              vad_blind=True)   # no exception
    drift = [line for line in logged if "时钟漂移" in line]
    assert drift and "不据此中止" in drift[0]


def test_a_working_vad_still_fails_the_job_on_a_fast_clock():
    with pytest.raises(RuntimeError, match="持续对不上"):
        A._report([], [], [], drifting_records(), 0, None, None, vad_blind=False)


def test_the_engine_reports_coverage_before_it_judges_the_timestamps(run):
    """Order matters: whether silero worked has to be said before its
    readings are used to accuse the transcript."""
    _, _, logged = run()
    coverage = next(i for i, line in enumerate(logged) if "识别覆盖率" in line)
    finished = next(i for i, line in enumerate(logged) if "识别完成" in line)
    assert finished < coverage


def test_the_engine_runs_silero_once(monkeypatch, wav):
    calls: list[int] = []
    spans = [(t, t + 8.0) for t in range(0, 600, 10)]
    monkeypatch.setattr(A, "cut_intervals", lambda audio: spans)

    def counted(audio, settings):
        calls.append(1)
        return spans

    monkeypatch.setattr(asr, "speech_intervals_of", counted)
    A.transcribe(str(wav),
                 ASRSettings(engine="api", api_window_seconds=300.0,
                             api_concurrency=1, api_audio_format="wav"),
                 LLMSettings(base_url="http://x/v1", api_key="k", model="m"),
                 client=FakeAudio())
    assert calls == [1]


# ------------------------------------------------- the empty-reply ladder
# Measured on the relay this project is developed against: one 92-second
# clip of a film came back with an empty candidate list ten times out of
# ten as mp3, and was transcribed normally as wav — same seconds, same
# prompt, same model. The usage block showed the audio had arrived and
# been charged for on every one of the refusals, which is what separates
# this from a server that simply has no such model.


class FormatPicky(FakeAudio):
    """Answers one container with no choices at all, the other normally."""

    def __init__(self, refuses="mp3", billed=True, **kw):
        super().__init__(**kw)
        self.refuses = refuses
        self.billed = billed
        self.formats: list[str] = []

    def create(self, model, messages, temperature, **kw):
        resp = super().create(model, messages, temperature, **kw)
        fmt = messages[0]["content"][1]["input_audio"]["format"]
        self.formats.append(fmt)
        if self.refuses in (fmt, "both"):
            resp.choices = []
            if not self.billed:
                resp.usage = None
        return resp


def mp3_settings():
    return ASRSettings(engine="api", api_window_seconds=300.0,
                       api_concurrency=1, api_audio_format="mp3")


def test_an_empty_reply_is_resent_in_the_other_container(run):
    client = FormatPicky(refuses="mp3")
    segments, _, _ = run(client, settings=mp3_settings())
    assert segments, "the wav retry should have carried the window"
    # every window: refused as mp3, then accepted as wav
    assert client.formats == ["mp3", "wav", "mp3", "wav"]


def test_the_empty_reply_is_named_and_the_switch_is_logged(run):
    client = FormatPicky(refuses="mp3")
    _, _, logged = run(client, settings=mp3_settings())
    text = "\n".join(logged)
    assert "空回复" in text
    assert "音频已到服务端并已计费" in text
    assert "换成 wav 重发同一段" in text


def test_a_resend_is_a_resend_and_carries_no_complaint(run):
    """The prompt may not change between the two containers.

    `complaint` is model-facing — it becomes "your last answer was wrong
    because …" — and there was no answer to fault. Changing the prompt
    would also make the retry a different experiment from the one that was
    measured.
    """
    client = FormatPicky(refuses="mp3")
    run(client, settings=mp3_settings())
    first, second = client.calls[0], client.calls[1]
    assert first[0]["content"][0]["text"] == second[0]["content"][0]["text"]
    assert "不符合要求" not in second[0]["content"][0]["text"]


def test_the_container_is_changed_once_per_window_not_repeatedly(run):
    """Both containers refused: switch once, then fall back to splitting."""
    client = FormatPicky(refuses="both")
    with pytest.raises(RuntimeError):
        run(client, settings=mp3_settings())
    # the first window's attempts, before any split: one mp3, one wav, stop
    assert client.formats[:2] == ["mp3", "wav"]
    assert client.formats.count("wav") <= client.formats.count("mp3") + 1


def test_a_billed_empty_reply_is_not_blamed_on_the_model(run):
    client = FormatPicky(refuses="both")
    with pytest.raises(RuntimeError) as caught:
        run(client, settings=mp3_settings())
    said = str(caught.value)
    assert "连续收到空回复" in said
    assert "换音频格式与拆短窗口都没能绕开" in said
    # the advice this replaces pointed at the two things that are working
    assert "换一个支持音频的模型" not in said
    assert "换一个模型" not in said


def test_an_unbilled_empty_reply_keeps_the_ordinary_advice(run):
    """No usage behind it means no proof the clip ever arrived, so this is
    not the known fault and the ordinary advice still applies."""
    client = FormatPicky(refuses="both", billed=False)
    with pytest.raises(RuntimeError) as caught:
        run(client, settings=mp3_settings())
    assert "已知故障" not in str(caught.value)


def test_the_billed_tokens_of_an_empty_reply_are_still_counted(run):
    """They were charged for. A cost report that hides them is wrong."""
    client = FormatPicky(refuses="mp3")
    usage: dict = {}
    run(client, settings=mp3_settings(), usage=usage)
    assert usage["empty"] == 2          # one refused window each
    assert usage["calls"] == 4          # two refusals, two that answered
    assert usage["prompt"] > A.AUDIO_TOKENS_PER_SECOND * 600
