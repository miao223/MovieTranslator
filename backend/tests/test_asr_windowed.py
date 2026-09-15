"""The VAD-off first pass, for sources silero cannot hear at all.

On a VHS capture silero kept 5.2% of the film (296.9s of 5760). The first
pass produced 228 segments; the second pass, looking only where the VAD had
said nothing, came back with 547 — so almost the entire film arrived at the
LLM review as material to be judged, against a "confirmed" transcript that
was 228 lines long. This switch changes where the decode happens rather
than what is allowed to delete: the whole timeline is decoded as VAD-off
windows, and a segment counts as first pass exactly when a VAD-gated run
would have seen it.

Off by default, and the tests below pin both halves of that: the default
path executes byte for byte what it executed before, and the fallback
engages only on a film that is both mostly-rejected and actually loud.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.models.schemas import ASRSettings
from app.services import asr
from tests.test_asr_content import FakeSegment, FakeWord

SR = 16_000


class WindowedFakeModel:
    """Records every transcribe() call, and refuses to be iterated lazily.

    The first-pass generator must NOT be consumed when the fallback
    engages — decoding is lazy in faster-whisper, so not iterating it is
    what makes the skip free. Iterating it here is an error.
    """

    def __init__(self, per_window=None, duration=900.0, after_vad=27.0,
                 first_pass=None):
        self.calls: list[dict] = []
        self.per_window = per_window or (lambda index, samples: [])
        self.duration = duration
        self.after_vad = after_vad
        self.first_pass = first_pass
        self.first_pass_consumed = False

    def _first_pass(self):
        self.first_pass_consumed = True
        if self.first_pass is None:
            raise AssertionError("the VAD-gated first pass was decoded after all")
        yield from self.first_pass

    def detect_language(self, audio=None, **kw):
        return "ja", 0.97, []

    def transcribe(self, audio, **kw):
        self.calls.append({"audio": audio, **kw})
        info = SimpleNamespace(duration=self.duration,
                               duration_after_vad=self.after_vad,
                               language="ja", language_probability=0.97)
        if isinstance(audio, str):          # the whole-file, VAD-gated call
            return self._first_pass(), info
        index = sum(1 for call in self.calls if not isinstance(call["audio"], str)) - 1
        return iter(self.per_window(index, audio)), info


def loud(seconds: float, level: float = 0.2):
    rng = np.random.default_rng(5)
    return (level * rng.standard_normal(int(seconds * SR))).astype("float32")


def quiet(seconds: float):
    """A track with nothing on it but a whisper of hiss — about -74 dBFS."""
    rng = np.random.default_rng(6)
    return (2e-4 * rng.standard_normal(int(seconds * SR))).astype("float32")


@pytest.fixture
def run(monkeypatch):
    """Drive asr.transcribe with a pinned model, audio and silero verdict."""
    def _run(audio, per_window=None, heard=((0.0, 5.0),), duration=900.0,
             after_vad=27.0, first_pass=None, **settings_kwargs):
        model = WindowedFakeModel(per_window, duration, after_vad, first_pass)
        monkeypatch.setattr(asr, "_get_model", lambda *a, **k: model)
        monkeypatch.setattr("faster_whisper.audio.decode_audio",
                            lambda *a, **k: audio)
        monkeypatch.setattr(asr, "speech_intervals_of",
                            lambda a, s: [(0.0, 5.0)])
        monkeypatch.setattr(asr, "_vad_intervals", lambda a, p: list(heard))
        monkeypatch.setattr(asr, "_report_coverage",
                            lambda *a, **k: None)
        logged: list[str] = []
        segments, _ = asr.transcribe("/x/audio.wav", ASRSettings(**settings_kwargs),
                                     log=logged.append)
        return segments, logged, model
    return _run


def window_of(texts):
    """A per-window callback returning the same lines in every window."""
    def _per_window(index, samples):
        return [FakeSegment(i * 3.0, i * 3.0 + 2.0, t) for i, t in enumerate(texts)]
    return _per_window


# ------------------------------------------------------------------ the switch


def test_the_switch_is_off_by_default():
    assert ASRSettings().windowed_first_pass is False


def test_off_means_the_first_pass_runs_exactly_as_before(run):
    """No new statement executes before collect(): the generator is consumed."""
    model = WindowedFakeModel(first_pass=[FakeSegment(0.0, 2.0, "ふつうの台詞")])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(asr, "_get_model", lambda *a, **k: model)
        patch.setattr("faster_whisper.audio.decode_audio",
                      lambda *a, **k: loud(900.0))
        patch.setattr(asr, "_report_coverage", lambda *a, **k: None)
        patch.setattr(asr, "second_pass", lambda *a, **k: a[2])
        out, _ = asr.transcribe("/x/audio.wav", ASRSettings(), log=None)
    assert [s.text for s in out] == ["ふつうの台詞"]
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["vad_filter"] is True and call["condition_on_previous_text"] is True


# ----------------------------------------------------------- when it engages


def test_a_deaf_vad_over_loud_audio_engages_the_fallback(run):
    segments, logged, model = run(loud(900.0), window_of(["台詞A", "台詞B"]),
                                  windowed_first_pass=True)
    assert not model.first_pass_consumed
    windows = [c for c in model.calls if not isinstance(c["audio"], str)]
    assert len(windows) == 3          # ceil(900 / SECOND_PASS_WINDOW)
    assert all(c["vad_filter"] is False for c in windows)
    assert all(c["condition_on_previous_text"] is False for c in windows)
    assert any("分窗识别兜底" in line for line in logged)
    assert len(segments) == 6         # two lines per window


def test_the_timestamps_are_offset_by_their_window(run):
    segments, _, _ = run(loud(900.0), window_of(["一", "二"]),
                         windowed_first_pass=True)
    assert [round(s.start, 2) for s in segments] == [0.0, 3.0, 300.0, 303.0,
                                                     600.0, 603.0]


def test_the_second_pass_is_skipped_because_the_timeline_is_already_covered(run):
    calls = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(asr, "second_pass",
                      lambda *a, **k: calls.append(1) or a[2])
        segments, logged, _ = run(loud(900.0), window_of(["台詞"]),
                                  windowed_first_pass=True)
    assert calls == []
    assert any("跳过二次识别" in line for line in logged)


def test_a_track_with_nothing_on_it_does_not_engage_it(run):
    """Same 3% VAD share, but nothing audible: a wrong track, not a deaf VAD.

    The loudness test is relative, so a track of pure hiss passes it
    trivially — its own hiss is "as loud as speech". The absolute floor is
    what stops a VAD-off whisper being pointed at an empty stream.
    """
    segments, logged, model = run(
        quiet(900.0), window_of(["台詞"]),
        first_pass=[FakeSegment(0.0, 2.0, "常規の台詞")],
        windowed_first_pass=True)
    assert model.first_pass_consumed          # it fell through to the normal path
    assert "常規の台詞" in [s.text for s in segments]
    assert any("未启用分窗识别兜底" in line for line in logged)
    assert any("更像是音轨本身没有内容" in line for line in logged)


def test_a_film_the_vad_can_hear_does_not_engage_it(run):
    _, logged, model = run(loud(900.0), window_of(["台詞"]),
                           after_vad=252.0,  # 28%, the Japanese DVD
                           first_pass=[FakeSegment(0.0, 2.0, "常規の台詞")],
                           windowed_first_pass=True)
    assert any("不需要兜底" in line for line in logged)
    assert model.first_pass_consumed
    assert not any(line.startswith("分窗识别兜底：") for line in logged)


def test_the_threshold_is_the_share_the_job_log_already_prints():
    assert asr.WINDOWED_FIRST_PASS_BELOW == 0.10


# ------------------------------------------------------------ what gets marked


def test_a_line_the_vad_also_heard_counts_as_first_pass(run):
    """The authority boundary: review may only delete what only the
    VAD-off decode could see."""
    def per_window(index, samples):
        if index:
            return []
        inside = FakeSegment(100.0, 112.0, "VAD も聞こえた")
        inside.words = [FakeWord(105.0, 106.0, "VAD も聞こえた")]
        outside = FakeSegment(400.0, 402.0, "関 VAD でしか見えない")
        outside.words = [FakeWord(400.0, 401.0, "関 VAD でしか見えない")]
        return [inside, outside]

    segments, _, _ = run(loud(900.0), per_window, heard=[(100.0, 110.0)],
                         windowed_first_pass=True)
    assert [s.recovered for s in segments] == [False, True]


def test_the_mark_follows_the_words_not_the_stretched_span(run):
    """A segment spanning silence it never transcribed must not be promoted."""
    def per_window(index, samples):
        if index:
            return []
        stretched = FakeSegment(0.0, 280.0, "引き伸ばされた")
        stretched.words = [FakeWord(250.0, 251.0, "引き伸ばされた")]
        return [stretched]

    segments, _, _ = run(loud(900.0), per_window, heard=[(0.0, 10.0)],
                         windowed_first_pass=True)
    assert [s.recovered for s in segments] == [True]


def test_the_gates_are_the_same_ones_the_second_pass_uses(run):
    def per_window(index, samples):
        if index:
            return []
        repeating = FakeSegment(0.0, 2.0, "繰り返し")
        repeating.compression_ratio = 3.0
        unsure = FakeSegment(4.0, 6.0, "自信なし")
        unsure.avg_logprob = -1.5
        musical = FakeSegment(8.0, 10.0, "音楽の下の台詞")
        musical.no_speech_prob = 0.95   # music does this to clean dialogue
        return [repeating, unsure, musical]

    segments, _, _ = run(loud(900.0), per_window, windowed_first_pass=True)
    assert [s.text for s in segments] == ["音楽の下の台詞"]


def test_nothing_readable_is_still_dropped(run):
    def per_window(index, samples):
        return [FakeSegment(0.0, 2.0, "-")] if index == 0 else []

    segments, _, _ = run(loud(900.0), per_window, windowed_first_pass=True)
    assert segments == []


# ------------------------------------------------- the extraction it relies on


def test_decode_windows_is_a_pure_lift_out_of_the_second_pass():
    """second_pass must still send the same kwargs it always sent."""
    import inspect

    source = inspect.getsource(asr._decode_windows)
    for expected in ("vad_filter=False", "condition_on_previous_text=False",
                     "beam_size=settings.beam_size",
                     "word_timestamps=settings.word_timestamps",
                     # build_initial_prompt(settings, None) IS that expression
                     "initial_prompt=build_initial_prompt(settings, exemplar)"):
        assert expected in source
    assert "recovered=True" not in source  # the caller decides
    assert asr.build_initial_prompt(ASRSettings(initial_prompt=" 佐藤 "), None) == "佐藤"
    assert asr.build_initial_prompt(ASRSettings(), None) is None


def test_the_second_pass_still_marks_everything_it_finds_as_recovered():
    model = WindowedFakeModel(lambda index, samples: [
        FakeSegment(0.0, 2.0, "見つかった台詞")])
    first = [asr.Segment(0.0, 1.0, "第一遍", [asr.Word(0.0, 1.0, "第一遍")])]
    out = asr.second_pass(model, loud(400.0), first, ASRSettings(), "ja")
    found = [s for s in out if s.recovered]
    assert found and all(s.recovered for s in found)
    assert any(not s.recovered for s in out)


# ------------------------------------------------------- the punctuation prompt
#
# Whisper writes in the shape of its prompt. Two Japanese films came back
# 72% and 57% open-ended, which is exactly what flips segmenter's
# is_effectively_unpunctuated and defers every sentence merge to refine.
# Off by default, because this one changes decoding.


def test_the_style_prompt_is_off_by_default():
    assert ASRSettings().style_prompt is False


def test_off_sends_the_prompt_expression_it_always_sent():
    settings = ASRSettings(initial_prompt="佐藤健一")
    assert asr.build_initial_prompt(settings, None) == "佐藤健一"
    assert asr.build_initial_prompt(ASRSettings(), None) is None


def test_the_exemplar_goes_first_and_the_names_last():
    """Whisper keeps the TAIL of an over-long prompt, so names must be last."""
    built = asr.build_initial_prompt(ASRSettings(initial_prompt="佐藤健一"),
                                     asr.STYLE_EXEMPLARS["ja"])
    assert built.startswith(asr.STYLE_EXEMPLARS["ja"])
    assert built.endswith("佐藤健一")


def test_a_language_with_no_exemplar_gets_none():
    """A prompt in the wrong language drags the whole transcript with it."""
    assert asr.style_exemplar("cy") is None
    assert asr.build_initial_prompt(ASRSettings(), asr.style_exemplar("cy")) is None


def test_every_exemplar_ends_a_sentence_and_starts_another():
    import re

    for code, sentence in asr.STYLE_EXEMPLARS.items():
        assert re.search(r"[.!?。！？]\s*\S", sentence), code   # a mid-text stop
        assert re.search(r"[.!?。！？]$", sentence), code       # and a final one


def test_a_stated_language_is_looked_up_without_detection():
    calls = []

    class Model:
        def detect_language(self, **kw):
            calls.append(1)
            return "en", 0.99, []

    assert asr._pick_exemplar(Model(), "ja", lambda: None, None) \
        == asr.STYLE_EXEMPLARS["ja"]
    assert calls == []


def test_auto_asks_whisper_first_and_only_for_the_exemplar():
    class Model:
        def detect_language(self, **kw):
            return "ja", 0.97, []

    logged: list[str] = []
    assert asr._pick_exemplar(Model(), None, lambda: None, logged.append) \
        == asr.STYLE_EXEMPLARS["ja"]
    assert any("预检测" in line for line in logged)


def test_the_detector_is_shown_speech_not_the_opening_titles():
    """Asked its default way, the detector sees one 30-second window from
    the very start — on a film, the logo and the score. Measured: a
    Japanese VHS transfer came back 20% sure and an English BD 49% sure,
    so neither arm got an exemplar at all.
    """
    seen = {}

    class Model:
        def detect_language(self, **kw):
            seen.update(kw)
            return "ja", 0.97, []

    asr._pick_exemplar(Model(), None, lambda: None, None)
    assert seen["vad_filter"] is True
    assert seen["language_detection_segments"] > 1


def test_an_unsure_detection_adds_nothing():
    class Model:
        def detect_language(self, **kw):
            return "ja", 0.2, []

    logged: list[str] = []
    assert asr._pick_exemplar(Model(), None, lambda: None, logged.append) is None
    assert any("把握" in line for line in logged)


def test_a_detector_that_raises_is_not_fatal():
    class Model:
        def detect_language(self, **kw):
            raise RuntimeError("no such model")

    logged: list[str] = []
    assert asr._pick_exemplar(Model(), None, lambda: None, logged.append) is None
    assert any("未能进行" in line for line in logged)


@pytest.mark.parametrize("said", [
    "こんにちは、お元気ですか。はい、おかげさまで元気です。",
    "こんにちは、お元気ですか",
    "はい、おかげさまで元気です！",
])
def test_the_exemplar_echoed_back_is_dropped(said):
    assert asr._is_prompt_echo(said, asr.STYLE_EXEMPLARS["ja"])


@pytest.mark.parametrize("said", [
    "こんにちは、田中さん。",       # the film really does greet someone
    "元気ですか、先生",
    "",
])
def test_a_line_that_merely_resembles_the_exemplar_is_kept(said):
    assert not asr._is_prompt_echo(said, asr.STYLE_EXEMPLARS["ja"])


def test_nothing_is_an_echo_when_the_switch_is_off():
    assert not asr._is_prompt_echo(asr.STYLE_EXEMPLARS["ja"], None)


def test_the_first_pass_receives_the_exemplar_and_drops_its_echo(run):
    said = [FakeSegment(0.0, 2.0, asr.STYLE_EXEMPLARS["ja"]),
            FakeSegment(3.0, 5.0, "本当の台詞です。")]
    segments, logged, model = run(loud(900.0), first_pass=said, after_vad=800.0,
                                  style_prompt=True, initial_prompt="佐藤健一",
                                  second_pass=False)
    prompt = model.calls[0]["initial_prompt"]
    assert prompt.startswith(asr.STYLE_EXEMPLARS["ja"]) and prompt.endswith("佐藤健一")
    assert [s.text for s in segments] == ["本当の台詞です。"]
    assert any("示例句完全相同" in line for line in logged)


def test_the_api_engine_never_sees_the_exemplar(monkeypatch):
    """asr_api gets the user's own prompt and nothing else."""
    from app.models.schemas import LLMSettings
    from app.services import asr_api

    seen = {}

    def fake(wav_path, settings, llm, **kw):
        seen["prompt"] = settings.initial_prompt
        return [], "ja"

    monkeypatch.setattr(asr_api, "transcribe", fake)
    asr.transcribe("/x.wav",
                   ASRSettings(engine="api", style_prompt=True,
                               initial_prompt="佐藤健一"),
                   llm=LLMSettings(model="m"))
    assert seen["prompt"] == "佐藤健一"


def test_a_stated_language_says_whether_it_added_anything():
    """The switch being on and an exemplar being added are two different
    facts, and only the second one changes the transcript."""
    class Model:
        def detect_language(self, **kw):
            raise AssertionError("a stated language needs no detection")

    logged: list[str] = []
    asr._pick_exemplar(Model(), "ja", lambda: None, logged.append)
    assert any("加在识别提示词前" in line for line in logged)

    logged.clear()
    asr._pick_exemplar(Model(), "xx", lambda: None, logged.append)
    assert any("本次不加" in line for line in logged)
