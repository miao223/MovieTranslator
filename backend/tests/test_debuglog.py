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
