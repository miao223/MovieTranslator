"""Debug mode must be invisible when off and complete when on."""
import pytest

from pathlib import Path

from app.core.debuglog import DebugLog, debug_path_for, fmt_cue, percentiles


def test_disabled_log_writes_nothing(tmp_path):
    log = DebugLog(tmp_path / "x.debug.log", enabled=False)
    assert not log.enabled
    log.section("s")
    log.kv("k", "v")
    log.block("b", "body")
    log.lines(["a", "b"])
    assert not (tmp_path / "x.debug.log").exists()


def test_enabled_log_records_sections_and_blocks(tmp_path):
    path = tmp_path / "movie.debug.log"
    log = DebugLog(path, enabled=True)
    log.section("语音识别")
    log.kv("语言", "ja")
    log.block("system prompt", "你是字幕转写整理员")
    text = path.read_text(encoding="utf-8")
    assert "语音识别" in text and "ja" in text and "你是字幕转写整理员" in text


def test_a_new_run_starts_a_fresh_file(tmp_path):
    path = tmp_path / "movie.debug.log"
    DebugLog(path, enabled=True).line("first run")
    DebugLog(path, enabled=True).line("second run")
    text = path.read_text(encoding="utf-8")
    assert "first run" not in text and "second run" in text


def test_unwritable_target_falls_back_to_the_work_dir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    missing = Path("/proc/nonexistent-dir/movie.srt")
    assert debug_path_for(missing, workdir) == workdir / "movie.debug.log"


def test_path_sits_next_to_the_subtitle(tmp_path):
    assert debug_path_for(tmp_path / "movie.srt", tmp_path) == tmp_path / "movie.debug.log"


def test_formatting_helpers():
    assert "0:01:01.500" in fmt_cue(1, 61.5, 63.0, "hi")
    assert percentiles([]) == "（无数据）"
    assert "P50=2.00" in percentiles([1.0, 2.0, 3.0])


def test_speech_coverage_finds_intervals_with_no_words():
    """The metric that turns "some lines feel missing" into a number."""
    from app.services.asr import Segment, Word, coverage_report

    intervals = [(0.0, 5.0), (10.0, 14.0), (20.0, 23.0)]
    segments = [
        Segment(0.0, 5.0, "spoken", words=[Word(0.2, 4.8, "spoken")]),
        # nothing at all for 10-14 — a miss
        Segment(20.0, 23.0, "also", words=[Word(20.1, 22.9, "also")]),
    ]
    speech, covered, misses = coverage_report(intervals, segments)
    assert speech == 12.0
    assert covered == pytest.approx(7.4)  # 4.6s + 2.8s of words
    assert [round(m[0]) for m in misses] == [10]


def test_coverage_ignores_intervals_shorter_than_a_second():
    """Sub-second gaps are breaths, not missing dialogue."""
    from app.services.asr import Segment, Word, coverage_report

    intervals = [(0.0, 5.0), (10.0, 10.4)]
    segments = [Segment(0.0, 5.0, "x", words=[Word(0.2, 4.8, "x")])]
    _, _, misses = coverage_report(intervals, segments)
    assert misses == []


def test_coverage_denominator_ignores_the_job_s_own_threshold():
    """Otherwise lowering the threshold inflates the denominator and the
    ratio falls even when the transcription got better — as it did on a
    real run: speech 1044s→1378s, transcribed 893s→953s, ratio 77%→63%."""
    import inspect

    from app.services import asr

    src = inspect.getsource(asr.speech_intervals_of)
    assert "REFERENCE_VAD_THRESHOLD" in src
    assert "settings.vad_threshold" not in src


def test_short_uncovered_runs_are_pauses_not_misses():
    from app.services.asr import Segment, Word, coverage_report

    # one 10s interval, words with a 1s pause and a 3s hole
    intervals = [(0.0, 10.0)]
    segments = [Segment(0.0, 10.0, "x", words=[
        Word(0.0, 3.0, "a"), Word(4.0, 5.0, "b"), Word(8.0, 10.0, "c"),
    ])]
    _, _, misses = coverage_report(intervals, segments)
    assert [(round(s), round(e)) for s, e, _ in misses] == [(5, 8)]


def test_stretches_outside_every_interval_are_found():
    from app.services.asr import _outside_intervals

    gaps = _outside_intervals([(10.0, 20.0), (40.0, 50.0)], 100.0, 5.0)
    assert [(round(s), round(e)) for s, e in gaps] == [(0, 10), (20, 40), (50, 100)]


def test_level_profile_is_one_value_per_second():
    import math

    from app.services.asr import level_profile

    quiet = [0.0] * 16000
    loud = [0.5] * 16000
    levels = level_profile(quiet + loud + quiet)
    assert len(levels) == 3
    assert levels[0] < -100          # digital silence
    assert math.isclose(levels[1], -6.0, abs_tol=0.5)  # 0.5 full scale ≈ -6 dBFS


# ------------------------------------------------------ second ASR pass


def test_blank_regions_are_what_the_first_pass_left_empty():
    from app.services.asr import Segment, _blank_regions

    segs = [Segment(40.0, 50.0, "a"), Segment(55.0, 60.0, "b")]
    # the 5s hole between them is a pause, not a blank; the 40s lead-in and
    # the tail both qualify
    assert _blank_regions(segs, 100.0, 15.0) == [(0.0, 40.0), (60.0, 100.0)]


def test_long_blanks_are_sliced_into_bounded_windows():
    """Unbounded VAD-free decoding is what looped; slices cannot."""
    from app.services.asr import _windows

    assert list(_windows([(0.0, 750.0)], 300.0)) == [
        (0.0, 300.0), (300.0, 600.0), (600.0, 750.0)
    ]


def test_recovered_segments_never_overwrite_the_first_pass():
    from app.services.asr import Segment, _overlaps_any

    existing = [Segment(10.0, 20.0, "already here")]
    assert _overlaps_any(Segment(15.0, 25.0, "x"), existing)
    assert not _overlaps_any(Segment(20.0, 25.0, "x"), existing)
