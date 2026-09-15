"""The comparison tool has to be right about formats it does not control.

Every figure the tool prints was hand-computed once, from a script that no
longer exists, against films that cannot go in the repository. What can be
pinned here is the machinery underneath: that both debug-log dialects parse,
that the interval arithmetic is the arithmetic, that pairing does not fall
for a film which says the same word forty times, and that the onset and
arbitration measurements agree with audio built to a known answer.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tools import compare_runs
from tools import runmetrics as M
from tools import runparse

SAMPLE_RATE = 16000


# --------------------------------------------------------------- fixtures

LOCAL_LOG = """=== MovieTranslator 0.17.0 调试日志 ===
生成时间: 2026-09-15 14:13:02


==============================================================================
== 语音识别原始输出（faster-whisper）    [任务开始后 1.5 分钟]
==============================================================================
语言                          : en (99%)
音频时长                        : 120.0s
VAD 后语音时长                   : 61.0s

每个 segment：起止、平均对数概率……

[   10.00 →    12.50] logprob=-0.37 no_speech=0.01 compress=1.74
    Hello there, friend.
    词: 10.00-10.40 Hello  10.40-10.90 there,  11.00-12.50 friend.

[   20.00 →    21.00] logprob=-0.52 no_speech=0.36 compress=1.05
    Second line.
    词: 20.00-20.50 Second  20.50-21.00 line.


==============================================================================
== 二次识别（对 VAD 丢弃的区间关 VAD 重跑）    [任务开始后 3.0 分钟]
==============================================================================
空白区间                        : 1 段，切成 1 个窗口

[   30.00 →    30.78] logprob=-0.52 no_speech=0.36 compress=1.05
    Recovered words.
    词: 30.00-30.46 Recovered  30.52-30.78 words.


==============================================================================
== 二次识别复核（LLM 判断是否属于本片）    [任务开始后 5.0 分钟]
==============================================================================
送审                          : 2 段 / 3s
分块数                         : 1

--- 第 1 块 响应（第 1 次） ---
[R1] 保留
[R2] 丢弃
--- /第 1 块 响应（第 1 次） ---

复核结果：保留 1 段 / 1s，丢弃 1 段 / 2s

逐条判定：
  [R1] 0:00:30.000 保留 | Recovered words.
  [R2] 0:00:40.000 丢弃 视频片尾语 | Thanks for watching.


==============================================================================
== 分句结果（segmenter）    [任务开始后 5.2 分钟]
==============================================================================
断行后条数                       : 3
句子合并后条数                     : 2
未完句结尾占比                     : 50%

合并前（断行直出）：
[   1] 0:00:10.000 → 0:00:11.000 ( 1.00s) | Hello there,
[   2] 0:00:11.000 → 0:00:12.500 ( 1.50s) | friend.
[   3] 0:00:20.000 → 0:00:21.000 ( 1.00s) | Second line.

合并后（送入预处理的内容）：
[   1] 0:00:10.000 → 0:00:12.500 ( 2.50s) | Hello there, friend.
[   2] 0:00:20.000 → 0:00:21.000 ( 1.00s) | Second line


==============================================================================
== 最终字幕    [任务开始后 9.9 分钟]
==============================================================================
条数                          : 2

最终结果（原文 ⇒ 译文）：
[   1] 0:00:10.000 → 0:00:12.500 ( 2.50s) | Hello there, friend.  ⇒  你好呀，朋友。
[   2] 0:00:20.000 → 0:00:21.000 ( 1.00s) | Second line  ⇒  第二行
"""

API_LOG = """=== MovieTranslator 0.19.0 调试日志 ===
生成时间: 2026-09-15 06:43:27


==============================================================================
== LLM 转写 vs silero VAD（只作参考，不改时间戳）    [任务开始后 2.6 分钟]
==============================================================================
窗口                          : 2（硬切 0，无人说话 1，重试 0）
时钟漂移（各段中位）                  : -266.1 ms/s（19 段）


==============================================================================
== 语音识别原始输出（API 引擎）    [任务开始后 2.6 分钟]
==============================================================================

--- [00:00–01:00] 0 行 第 1 次通过 ---

--- 00:00–01:00 ---
NO SPEECH
--- /00:00–01:00 ---

--- [01:00–02:00] 2 行 第 2 次通过 ---

--- 01:00–02:00 ---
#lang: ja
[00:10.500 --> 00:12.000] こんにちは
[00:20.000 --> 00:21.500] さようなら
--- /01:00–02:00 ---


==============================================================================
== 最终字幕    [任务开始后 6.4 分钟]
==============================================================================
条数                          : 2

最终结果（原文 ⇒ 译文）：
[   1] 0:01:10.500 → 0:01:12.000 ( 1.50s) | こんにちは  ⇒  你好
[   2] 0:01:20.000 → 0:01:21.500 ( 1.50s) | さようなら  ⇒  再见
"""

SRT = """1
00:00:10,000 --> 00:00:12,500
Hello there, friend.
你好呀，朋友。

2
00:00:20,000 --> 00:00:21,000
Second line
第二行
"""


@pytest.fixture()
def local_run(tmp_path: Path) -> runparse.Run:
    (tmp_path / "film.debug.log").write_text(LOCAL_LOG, encoding="utf-8")
    return runparse.load_run([tmp_path], "local")


@pytest.fixture()
def api_run(tmp_path: Path) -> runparse.Run:
    (tmp_path / "film.api.debug.log").write_text(API_LOG, encoding="utf-8")
    return runparse.load_run([tmp_path], "api")


# ------------------------------------------------------------------ parsing


def test_a_whisper_log_gives_back_its_segments_and_words(local_run):
    assert local_run.engine == "local"
    assert local_run.version == "0.17.0"
    assert local_run.duration == 120.0
    assert [(s.start, s.end, s.text) for s in local_run.raw] == [
        (10.0, 12.5, "Hello there, friend."),
        (20.0, 21.0, "Second line."),
    ]
    assert local_run.raw[0].logprob == -0.37
    assert [w.text for w in local_run.raw[0].words] == ["Hello", "there,", "friend."]
    assert [s.text for s in local_run.recovered] == ["Recovered words."]


def test_the_merged_block_is_read_not_the_one_before_it(local_run):
    """`句子合并后条数` also contains '合并后'; arming on it loses every cue."""
    assert [c.text for c in local_run.segmented] == ["Hello there, friend.", "Second line"]


def test_the_review_table_is_counted_with_its_reasons(local_run):
    assert local_run.vet["kept"] == 1
    assert local_run.vet["dropped"] == 1
    assert local_run.vet["reasons"] == {"视频片尾语": 1}
    assert local_run.vet["submitted"] == "2 段 / 3s"
    assert local_run.vet["failures"] == []


def test_a_review_that_voided_a_whole_chunk_says_so(tmp_path):
    text = LOCAL_LOG.replace(
        "--- 第 1 块 响应（第 1 次） ---",
        "⚠ 第 1 块第 1 次失败：判定覆盖校验未通过（送审 547 条，收到 545 条有效判定）\n"
        "--- 第 1 块 响应（第 1 次） ---")
    (tmp_path / "film.debug.log").write_text(text, encoding="utf-8")
    run = runparse.load_run([tmp_path])
    assert run.vet["failures"] == [
        "第 1 块第 1 次：判定覆盖校验未通过（送审 547 条，收到 545 条有效判定）"]


def test_api_windows_are_anchored_to_their_own_heading(api_run):
    """Window-relative times plus the MM:SS heading = absolute times."""
    assert api_run.engine == "api"
    assert [(round(s.start, 3), round(s.end, 3), s.text) for s in api_run.raw] == [
        (70.5, 72.0, "こんにちは"),
        (80.0, 81.5, "さようなら"),
    ]
    assert [w["attempts"] for w in api_run.windows] == [1, 2]
    assert api_run.windows[0]["parsed"] == 0  # NO SPEECH
    assert api_run.section("LLM 转写")["时钟漂移（各段中位）"] == "-266.1 ms/s（19 段）"


def test_final_cues_carry_their_translation(local_run):
    assert local_run.final[0].text == "Hello there, friend."
    assert local_run.final[0].translation == "你好呀，朋友。"


def test_an_srt_is_read_as_bilingual(tmp_path):
    (tmp_path / "film.srt").write_text(SRT, encoding="utf-8")
    run = runparse.load_run([tmp_path])
    assert [(c.start, c.end, c.text, c.translation) for c in run.final] == [
        (10.0, 12.5, "Hello there, friend.", "你好呀，朋友。"),
        (20.0, 21.0, "Second line", "第二行"),
    ]


def test_json_beats_the_log_because_it_has_the_exact_floats(tmp_path):
    import json

    (tmp_path / "film.debug.log").write_text(API_LOG, encoding="utf-8")
    (tmp_path / "film.api.translation.json").write_text(json.dumps(
        [{"start": 70.512, "end": 72.0, "text": "こんにちは", "translation": "你好"}]),
        encoding="utf-8")
    run = runparse.load_run([tmp_path])
    assert run.sources["final"] == "translation.json"
    assert run.final[0].start == 70.512


def test_a_run_whose_debug_log_was_lost_still_loads(tmp_path):
    """K9's API-side debug log was overwritten; srt + json must be enough."""
    (tmp_path / "film.api.srt").write_text(SRT, encoding="utf-8")
    run = runparse.load_run([tmp_path])
    assert len(run.final) == 2
    assert run.raw == []
    assert run.engine == "unknown"


# --------------------------------------------------------------- interval math


def test_subtract_leaves_only_what_the_other_side_does_not_have():
    assert M.subtract([(0, 10)], [(2, 4), (6, 7)]) == [(0, 2), (4, 6), (7, 10)]
    assert M.subtract([(0, 10)], [(0, 10)]) == []


def test_intersect_is_symmetric_and_counts_each_second_once():
    a = M.merge([(0, 5), (4, 8)])
    b = M.merge([(3, 6), (7, 12)])
    assert a == [(0.0, 8.0)]
    assert M.intersect(a, b) == pytest.approx(4.0)
    assert M.intersect(b, a) == pytest.approx(4.0)


def test_merge_bridges_only_up_to_the_bridge():
    assert M.merge([(0, 1), (1.2, 2)], bridge=0.3) == [(0.0, 2.0)]
    assert M.merge([(0, 1), (1.2, 2)], bridge=0.1) == [(0.0, 1.0), (1.2, 2.0)]


# ------------------------------------------------------------------ pairing


class Line:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def test_a_film_that_repeats_a_line_forty_times_pairs_by_time():
    """Identical text everywhere: only the tie-break keeps pairs honest.

    Every candidate inside the search window scores a perfect 1.0, so
    without "on a tie take the nearer start" the pairs wander by up to the
    window and the timeline comparison becomes noise.
    """
    said = "I bring joy on Earth"
    left = [Line(t, t + 1, said) for t in range(0, 400, 10)]
    right = [Line(t + 0.4, t + 1.4, said) for t in range(0, 400, 10)]
    pairs = M.pair_by_text(left, right)
    assert len(pairs) == len(left)
    assert all(abs(b.start - a.start - 0.4) < 1e-6 for a, b, _ in pairs)


def test_a_line_of_one_repeated_word_is_left_unpaired():
    """`Hecate, Hecate, Hecate` is one distinct word: unpairable, not mispaired."""
    left = [Line(t, t + 1, "Hecate, Hecate, Hecate") for t in range(0, 100, 10)]
    right = [Line(t + 5, t + 6, "Hecate, Hecate") for t in range(0, 100, 10)]
    assert M.pair_by_text(left, right) == []


def test_japanese_pairs_on_character_bigrams_not_on_words():
    left = [Line(10, 12, "アリス、外してね？")]
    right = [Line(10.5, 12.5, "アリス 外して ね？")]
    assert M.grams("アリス、外してね？") & M.grams("アリス 外して ね？")
    pairs = M.pair_by_text(left, right)
    assert len(pairs) == 1 and pairs[0][2] > 0.8


def test_lines_too_short_to_identify_are_not_paired():
    assert M.pair_by_text([Line(0, 1, "ね")], [Line(0, 1, "ね")]) == []


# -------------------------------------------------------------- the audio


def tone(seconds: float, level: float = 0.2, freq: float = 220.0):
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (level * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def silence(seconds: float):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_onset_is_measured_from_the_cue_start_to_the_first_sound():
    audio = np.concatenate([silence(1.0), tone(1.0)])
    stats = M.onset_delays([Line(0.0, 2.0, "x")], audio)
    assert stats.measured == 1
    assert stats.delays[0] == pytest.approx(1.0, abs=0.04)


def test_a_cue_that_starts_on_speech_measures_zero():
    audio = np.concatenate([tone(2.0), silence(1.0)])
    stats = M.onset_delays([Line(0.0, 2.0, "x")], audio)
    assert stats.delays[0] == pytest.approx(0.0, abs=0.04)


def test_a_cue_with_nothing_in_it_is_reported_not_measured():
    stats = M.onset_delays([Line(0.0, 2.0, "x")], silence(3.0))
    assert stats.silent == 1 and stats.measured == 0


def test_the_reference_is_the_loud_end_so_mostly_silent_cues_still_count():
    """Against the cue's median, room tone is the reference and the lag vanishes."""
    room = (0.0005 * np.random.default_rng(0).standard_normal(
        int(3.0 * SAMPLE_RATE))).astype(np.float32)
    audio = room.copy()
    audio[int(2.0 * SAMPLE_RATE):] += tone(1.0)
    stats = M.onset_delays([Line(0.0, 3.0, "x")], audio)
    assert stats.delays[0] == pytest.approx(2.0, abs=0.06)


def test_arbitration_gives_the_start_that_lands_on_sound():
    audio = np.concatenate([silence(5.0), tone(5.0)])
    early = Line(1.0, 9.0, "same words here")
    late = Line(5.0, 9.0, "same words here")
    verdict = M.arbitrate([(early, late, 1.0)], audio)
    assert (verdict.disputed, verdict.left, verdict.right) == (1, 0, 1)


def test_starts_that_nearly_agree_are_not_put_to_the_question():
    audio = np.concatenate([silence(5.0), tone(5.0)])
    verdict = M.arbitrate([(Line(5.0, 9.0, "x"), Line(5.4, 9.0, "x"), 1.0)], audio)
    assert verdict.disputed == 0


# ------------------------------------------------------------- vad blindness


def test_a_source_silero_cannot_hear_is_flagged_blind():
    picture = M.VadPicture(duration=600.0, intervals=[(10.0, 20.0)],
                           levels=[-20.0] * 600, speech=10.0, speech_db=-20.0,
                           loud=600.0)
    coverage = M.coverage(picture, [Line(0, 300, "x")])
    assert coverage.vad_share < 0.25 and coverage.transcript_in_vad < 0.25
    assert coverage.blind


def test_a_film_that_is_mostly_music_is_not_blind_if_silero_hears_the_dialogue():
    picture = M.VadPicture(duration=600.0, intervals=[(0.0, 100.0)],
                           levels=[-20.0] * 600, speech=100.0, speech_db=-20.0,
                           loud=600.0)
    coverage = M.coverage(picture, [Line(0, 100, "x")])
    assert coverage.vad_share < 0.25          # one signal says blind…
    assert coverage.transcript_in_vad == 1.0  # …the other says silero heard it
    assert not coverage.blind


def test_too_little_loud_audio_to_judge_is_not_blind():
    picture = M.VadPicture(duration=60.0, intervals=[], levels=[-20.0] * 30,
                           speech=0.0, speech_db=-20.0, loud=30.0)
    assert not M.coverage(picture, [Line(0, 30, "x")]).blind


# ------------------------------------------------------------------ hygiene


def test_the_application_never_imports_the_tools():
    """tools -> app is the only direction; app must ship without tools."""
    app = Path(__file__).resolve().parents[1] / "app"
    offenders = [
        path.relative_to(app)
        for path in app.rglob("*.py")
        if "tools" in path.read_text(encoding="utf-8")
        and any(line.strip().startswith(("import tools", "from tools"))
                for line in path.read_text(encoding="utf-8").splitlines())
    ]
    assert offenders == []


def test_open_ended_ratio_matches_the_segmenter_switch():
    from app.services import segmenter

    cues = [Line(0, 1, "Hello."), Line(1, 2, "and then"), Line(2, 3, "more")]
    assert M.open_ended_ratio(cues) == pytest.approx(2 / 3)
    assert segmenter.UNPUNCTUATED_ABOVE == 0.5


def test_order_faults_see_overlap_inversion_and_zero_length():
    cues = [Line(0, 5, "a"), Line(4, 6, "b"), Line(3, 3, "c")]
    assert M.order_faults(cues) == {"inverted": 1, "overlapping": 2, "empty": 1}


def test_a_merged_line_is_not_counted_as_a_misplaced_one():
    """Asking for punctuation makes the segmenter merge, and a merged cue
    legitimately starts earlier than the fragment it pairs with."""
    fragment = Line(10.0, 12.0, "this is the second half")
    merged = Line(5.0, 12.0, "this is the first half and this is the second half")
    pairs = [(fragment, merged, 0.5)]
    assert M.comparable(pairs) == []
    assert M.far_share(pairs) is None


def test_the_same_line_placed_five_seconds_off_is_counted():
    here = Line(10.0, 12.0, "exactly the same words here")
    there = Line(16.0, 18.0, "exactly the same words here")
    assert M.far_share([(here, there, 1.0)]) == 1.0
    assert M.far_share([(here, Line(10.4, 12.4, here.text), 1.0)]) == 0.0


def test_a_perfectly_corrected_start_is_not_judged_by_its_median():
    """Snapping lands every start ONSET_LEAD ahead of the sound, so the
    median floors at that constant — the defect lives in the tail."""
    from app.services import asr_api

    # the shape a real Blu-ray baseline has: most starts already on the
    # sound, a tail that is not
    baseline = M.OnsetStats(delays=[0.0] * 60 + [0.2] * 30 + [0.6] * 10,
                            measured=100)
    corrected = M.OnsetStats(delays=[asr_api.ONSET_LEAD] * 100, measured=100)
    assert corrected.median > baseline.median      # the misleading comparison
    assert corrected.p90 < baseline.p90            # ...and the honest ones
    assert corrected.late_share == 0.0
    assert baseline.late_share == 0.10


def test_an_arms_segments_json_carries_words_and_provenance(tmp_path):
    """The GPU arms dump the post-review transcript; a debug log cannot."""
    import json

    (tmp_path / "run.segments.json").write_text(json.dumps([
        {"start": 1.0, "end": 2.0, "text": "第一遍", "recovered": False,
         "words": [{"start": 1.1, "end": 1.9, "text": "第一遍"}]},
        {"start": 5.0, "end": 6.0, "text": "找回来的", "recovered": True, "words": []},
    ]), encoding="utf-8")
    run = runparse.load_run([tmp_path])
    assert run.post_vet and run.engine == "local"
    assert [s.text for s in run.raw] == ["第一遍"]
    assert [s.text for s in run.recovered] == ["找回来的"]
    assert run.raw[0].words[0].text == "第一遍"


def test_comparing_a_run_with_itself_never_fails():
    """A regression gate that fails on identity is not a gate.

    The punctuation check used to hold every candidate to A2's acceptance
    target, which whisper's Japanese (80% open-ended) cannot meet and which
    nothing on the local route addresses.
    """
    from tools import compare_runs as C

    side = C.Side(
        run=runparse.Run(label="x", engine="local"),
        coverage=M.Coverage(speech=100.0, transcribed=100.0, inside=90.0,
                            share=0.9, transcript_in_vad=0.9, vad_share=0.9),
        onset=M.OnsetStats(delays=[0.1] * 10, measured=10),
        open_ended=0.80, faults={"inverted": 0, "overlapping": 0, "empty": 1},
        characters=1000, finals=100, segmented=100,
    )
    checks = C.verdicts(side, [side], "local", None, 0.0, 0.0)
    assert [c.level for c in checks if c.level == C.FAIL] == [], \
        [(c.name, c.detail) for c in checks if c.level == C.FAIL]


def test_crossing_the_segmenter_switch_upwards_is_a_failure():
    from tools import compare_runs as C

    def side(open_ended):
        return C.Side(
            run=runparse.Run(label="x", engine="local"),
            coverage=M.Coverage(speech=100.0, transcribed=100.0, inside=90.0,
                                share=0.9, transcript_in_vad=0.9, vad_share=0.9),
            onset=M.OnsetStats(delays=[0.1] * 10, measured=10),
            open_ended=open_ended, faults={"inverted": 0, "overlapping": 0, "empty": 0},
            characters=1000, finals=100, segmented=100)

    worse = C.verdicts(side(0.7), [side(0.4)], "local", None, 0.0, 0.0)
    assert any(c.level == C.FAIL and "未完句" in c.name for c in worse)
    better = C.verdicts(side(0.1), [side(0.7)], "api", None, 0.0, 0.0)
    assert all(c.level != C.FAIL for c in better if "未完句" in c.name)


def test_a_baseline_that_is_not_there_stops_the_run(tmp_path):
    """A missing directory must not read as an empty side.

    Measured: three baseline paths that had not been fetched yet made
    every metric read 0 and the verdict table come back all PASS — the
    mistake produced the green light instead of being caught by it.
    """
    with pytest.raises(SystemExit) as caught:
        compare_runs.main([
            "--baseline", str(tmp_path / "not-fetched-yet"),
            "--candidate", str(tmp_path),
            "--audio", str(tmp_path / "a.wav"),
        ])
    assert "不存在" in str(caught.value)
