"""Whisper output that contains nothing readable must not become subtitles.

The case behind these tests: an 85-minute flv whose first pass returned 577
segments that were every one of them the single character "-". Nothing
downstream objected — the segmenter merged the dashes into cues, the LLM
was billed for translating them, and 224 of the 387 finished cues (577
seconds of screen time) were rows of dashes. Worse, those segments counted
as transcribed, so the second pass treated 1,412 seconds as already covered
and never looked there.
"""

from types import SimpleNamespace

import pytest

from app.models.schemas import ASRSettings
from app.services import asr


class FakeWord:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class FakeSegment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob = -0.15
        self.no_speech_prob = 0.59
        self.compression_ratio = 0.33  # a lone dash compresses beautifully
        self.words = [FakeWord(start, end, text)]


def fake_segments(texts):
    return [FakeSegment(i * 2.0, i * 2.0 + 1.0, t) for i, t in enumerate(texts)]


class FakeModel:
    """Stands in for faster_whisper.WhisperModel.

    *unprimed* is what it returns once conditioning on the previous window
    is switched off — the retry path.
    """

    def __init__(self, segments, unprimed=None):
        self._segments = segments
        self._unprimed = unprimed
        self.conditions: list[bool] = []

    def transcribe(self, *_a, **kw):
        condition = kw.get("condition_on_previous_text", True)
        self.conditions.append(condition)
        info = SimpleNamespace(
            duration=100.0, duration_after_vad=50.0,
            language="ja", language_probability=0.98,
        )
        segs = self._segments
        if not condition and self._unprimed is not None:
            segs = self._unprimed
        return iter(segs), info


@pytest.fixture
def run(monkeypatch):
    def _run(texts, unprimed=None, **settings_kwargs):
        model = FakeModel(
            fake_segments(texts),
            fake_segments(unprimed) if unprimed is not None else None,
        )
        monkeypatch.setattr(asr, "_get_model", lambda *a, **k: model)
        logged: list[str] = []
        # a wav_path that cannot be decoded keeps the second pass and the
        # coverage report out of the way; only the segment loop is under test
        results, _lang = asr.transcribe(
            "/nonexistent/audio.wav",
            ASRSettings(**settings_kwargs),
            log=logged.append,
        )
        return results, logged, model
    return _run


# ---------------------------------------------------------------- has_content

@pytest.mark.parametrize("text", [
    "-", "- -", "- - - - - -", "…", "。。。", "!!", "♪", "♪♪", "___", "  ",
])
def test_symbol_only_output_carries_no_content(text):
    assert not asr.has_content(text)


@pytest.mark.parametrize("text", [
    "これは何だと思う?", "はい", "ん", "Okay.", "3", "【ドアの音】",
    "♪ You're the only one I'll ever love ♪",
])
def test_anything_readable_counts_as_content(text):
    # 「ん」 and 【ドアの音】 are decided by vet.py against the film, not here:
    # this gate only removes what has no characters at all
    assert asr.has_content(text)


# ---------------------------------------------------------------- transcribe

def test_dash_only_segments_never_reach_the_pipeline(run):
    results, _, _ = run(["-", "これは何だと思う?", "-", "はい", "- - -"])
    assert [s.text for s in results] == ["これは何だと思う?", "はい"]


def test_a_stray_dash_is_dropped_without_alarming_the_user(run):
    _, logged, model = run(["-"] + ["セリフ"] * 30)
    joined = "\n".join(logged)
    assert "没有任何文字内容" in joined
    assert "⚠" not in joined
    assert model.conditions == [True]  # nothing to retry


def test_nothing_is_reported_when_every_segment_has_content(run):
    results, logged, _ = run(["セリフ", "はい"])
    assert len(results) == 2
    assert "没有任何文字内容" not in "\n".join(logged)


# ------------------------------------------------------- the decode loop
#
# Whisper primes each window with what it decoded in the previous one, so a
# single meaningless symbol can hold for the rest of the film. The second
# pass already runs its windows unprimed for this reason; when the first
# pass comes back with nothing readable at all, it gets the same treatment.


def test_a_film_transcribed_entirely_as_dashes_is_decoded_again_unprimed(run):
    results, logged, model = run(["-"] * 20, unprimed=["これは何だと思う?", "はい"])
    assert [s.text for s in results] == ["これは何だと思う?", "はい"]
    assert model.conditions == [True, False]
    assert "重试成功" in "\n".join(logged)


def test_the_retry_is_reported_when_it_does_not_help(run):
    results, logged, model = run(["-"] * 20)
    assert results == []
    joined = "\n".join(logged)
    assert model.conditions == [True, False]
    # the actionable part: a wrong audio track or an unusable model,
    # not a film that happens to be quiet
    assert "重试后仍然没有任何文字" in joined


def test_a_film_with_some_dialogue_is_never_decoded_twice(run):
    """The retry costs a whole second decode — only a total loss earns it."""
    results, _, model = run(["-"] * 20 + ["セリフ"])
    assert [s.text for s in results] == ["セリフ"]
    assert model.conditions == [True]


def test_dropped_segments_stay_in_the_debug_log(run, tmp_path):
    """The debug log records what whisper *said*; that is its whole job."""
    from app.core.debuglog import DebugLog

    path = tmp_path / "film.debug.log"
    dbg = DebugLog(path, enabled=True)
    segs = [FakeSegment(0.0, 1.0, "-"), FakeSegment(2.0, 3.0, "セリフ")]

    import unittest.mock as mock
    with mock.patch.object(asr, "_get_model", lambda *a, **k: FakeModel(segs)):
        results, _ = asr.transcribe(
            "/nonexistent/audio.wav", ASRSettings(), debug=dbg
        )

    text = path.read_text(encoding="utf-8")
    assert len(results) == 1
    assert "无文字内容，已丢弃" in text
    assert "无文字内容而丢弃的 segment" in text


# ------------------------------------------------------------ second pass

def test_second_pass_drops_content_free_recoveries(monkeypatch):
    """The same gate on the far noisier side of the pipeline."""
    import numpy as np

    class Recovering:
        def transcribe(self, *_a, **_k):
            segs = [FakeSegment(1.0, 2.0, "-"), FakeSegment(3.0, 4.0, "誰かいる")]
            return iter(segs), SimpleNamespace(duration=300.0)

    audio = np.zeros(16000 * 60, dtype="float32")
    out = asr.second_pass(
        Recovering(), audio, [], ASRSettings(), "ja",
    )
    assert [s.text for s in out] == ["誰かいる"]
    assert all(s.recovered for s in out)
