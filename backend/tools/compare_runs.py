"""Did this change make the subtitles worse? Compare two runs and say so.

    cd backend
    .venv/bin/python -m tools.compare_runs --film 可愛い悪魔 --mode cross
    .venv/bin/python -m tools.compare_runs --film K9.and.Company --mode api \
        --baseline text/K9.and.Company/api --candidate /path/to/new/run

The point of the tool is that "not worse" stops being a judgement call.
Every threshold below has a measured band behind it, printed with the
verdict, and where two runs of the *same* code are given as baselines the
spread between them is the noise floor a candidate is allowed to sit in —
whisper loses a different 15-20% of a film on every run, so equality was
never the right test.

``--mode cross`` compares the two engines and deliberately prints no
verdict: they are not each other's baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.services import asr, segmenter
from tools import runmetrics as M
from tools import runparse

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "text"

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "····"


def segmenter_switch() -> float:
    """The ratio above which the segmenter stops merging sentences itself."""
    return segmenter.UNPUNCTUATED_ABOVE


# ------------------------------------------------------------------ figures


@dataclass
class Side:
    """Everything measured about one run, on one film."""

    run: runparse.Run
    coverage: M.Coverage
    onset: M.OnsetStats
    open_ended: float
    faults: Dict[str, int]
    characters: int
    finals: int
    segmented: int
    lines: List[runparse.Cue] = field(default_factory=list)
    raw_spans: List[M.Span] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.run.label


def measure(run: runparse.Run, picture: M.VadPicture, audio,
            stage: str = "final") -> Side:
    # After the review, the first pass and what survived it are one
    # transcript — that is what asr._report_coverage measures. A debug log
    # cannot give that back: its two sections are from before the review.
    if run.post_vet:
        source = sorted(run.raw + run.recovered, key=lambda s: s.start)
        run.sources["coverage"] = "复核后的全部 segment（词级区间）"
    else:
        source = run.raw or run.segmented or run.final
        run.sources["coverage"] = ("第一遍 segment 区间（不含二次识别）" if run.raw else
                                   "分句后区间（无原始输出，读数略不同）" if run.segmented
                                   else "最终字幕区间")
    # A run that stopped after the segmenter — the shape the GPU arm
    # produces, and what is left when the translation endpoint drops a
    # batch — still has everything the recognition-side metrics need.
    lines = run.segmented if stage == "segmented" else (run.final or run.segmented)
    if stage == "segmented":
        run.sources["final"] = run.sources.get("segmented", "分句后")
    elif not run.final and run.segmented:
        run.sources["final"] = run.sources.get("segmented", "") + "（未跑到翻译，按分句后计量）"
    coverage = M.coverage(picture, source)
    return Side(
        run=run,
        coverage=coverage,
        onset=M.onset_delays(lines, audio),
        open_ended=M.open_ended_ratio(run.segmented or lines),
        faults=M.order_faults(lines),
        characters=M.characters(lines),
        finals=len(lines),
        segmented=len(run.segmented),
        lines=lines,
        raw_spans=M.spans_of(source),
    )


# ------------------------------------------------------------- vad caching


def vad_picture_cached(audio, key: Path, cache_dir: Optional[Path]) -> M.VadPicture:
    """Silero + level profile, remembered per audio file.

    Every run of this tool would otherwise spend a minute re-deriving the
    same intervals from the same wav.
    """
    if cache_dir is None:
        return M.vad_picture(audio)
    stamp = f"{key.resolve()}:{key.stat().st_size}"
    path = cache_dir / (hashlib.sha256(stamp.encode()).hexdigest()[:16] + ".json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return M.VadPicture(
            duration=data["duration"],
            intervals=[tuple(x) for x in data["intervals"]],
            levels=data["levels"], speech=data["speech"],
            speech_db=data["speech_db"], loud=data["loud"],
        )
    picture = M.vad_picture(audio)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "duration": picture.duration, "intervals": picture.intervals,
        "levels": picture.levels, "speech": picture.speech,
        "speech_db": picture.speech_db, "loud": picture.loud,
    }), encoding="utf-8")
    return picture


# ------------------------------------------------------------------ verdict


@dataclass
class Check:
    name: str
    level: str
    detail: str


def _band(values: Sequence[float]) -> str:
    if len(values) < 2:
        return ""
    return f"（基线离散 {min(values):.3g}–{max(values):.3g}）"


def _worse(candidate: float, baselines: Sequence[float], warn: float, fail: float,
           bigger_is_better: bool) -> tuple[str, float]:
    """Level and the signed drop, allowing the baselines' own spread."""
    mean = statistics.fmean(baselines)
    drop = (mean - candidate) if bigger_is_better else (candidate - mean)
    if len(baselines) > 1:
        # inside the noise floor of two runs of the same code = PASS
        low, high = min(baselines), max(baselines)
        if low <= candidate <= high:
            return PASS, drop
    if drop >= fail:
        return FAIL, drop
    if drop >= warn:
        return WARN, drop
    return PASS, drop


def verdicts(candidate: Side, baselines: Sequence[Side], mode: str,
             arbitration: Optional[M.Arbitration],
             exclusive_share: float, far_share_delta: Optional[float]) -> List[Check]:
    api = mode == "api"
    out: List[Check] = []

    # --- coverage, in percentage points --------------------------------
    have = [b.coverage.share for b in baselines if b.coverage.share is not None]
    if candidate.coverage.blind or any(b.coverage.blind for b in baselines):
        out.append(Check("同口径覆盖率", INFO, "VAD 失聪，本片不判定"))
    elif have and candidate.coverage.share is not None:
        level, drop = _worse(candidate.coverage.share * 100, [h * 100 for h in have],
                             2.0 if api else 5.0, 5.0 if api else 10.0, True)
        out.append(Check("同口径覆盖率", level,
                         f"{candidate.coverage.share:.1%} vs {statistics.fmean(have):.1%}"
                         f"（{-drop:+.1f} 点）{_band([h * 100 for h in have])}"))

    # --- how much text came out ----------------------------------------
    chars = [float(b.characters) for b in baselines]
    if chars and statistics.fmean(chars):
        level, drop = _worse(candidate.characters / statistics.fmean(chars) * 100,
                             [c / statistics.fmean(chars) * 100 for c in chars],
                             10.0 if api else 15.0, 20.0 if api else 25.0, True)
        out.append(Check("原文字数", level,
                         f"{candidate.characters} vs {statistics.fmean(chars):.0f}"
                         f"（{-drop:+.1f}%）"))

    counts = [float(b.finals) for b in baselines]
    if counts and statistics.fmean(counts):
        change = (candidate.finals - statistics.fmean(counts)) / statistics.fmean(counts)
        out.append(Check("条数", WARN if abs(change) > 0.30 else INFO,
                         f"{candidate.finals} vs {statistics.fmean(counts):.0f}"
                         f"（{change:+.0%}，已知非确定量）"))

    # --- content only the baseline has ---------------------------------
    warn_at, fail_at = (0.05, 0.10) if api else (0.20, 0.30)
    out.append(Check("基线独有内容占比",
                     FAIL if exclusive_share > fail_at else
                     WARN if exclusive_share > warn_at else PASS,
                     f"{exclusive_share:.1%}（阈值 {warn_at:.0%}/{fail_at:.0%}，按时间不按行）"))

    # --- where the cues start ------------------------------------------
    # Judged on the tail, not the middle: a start is *meant* to land
    # ONSET_LEAD ahead of the sound, so once snapping is on the median
    # cannot fall below it and a perfect run would score worse than a
    # baseline whose median happened to be 0. What the defect actually
    # looks like is starts sitting in silence, and that is the tail.
    p90s = [b.onset.p90 for b in baselines if M.nan_safe(b.onset.p90)]
    lates = [b.onset.late_share for b in baselines if M.nan_safe(b.onset.late_share)]
    if p90s and M.nan_safe(candidate.onset.p90):
        rise90 = candidate.onset.p90 - statistics.fmean(p90s)
        rise_late = (candidate.onset.late_share - statistics.fmean(lates)) if lates else 0.0
        level = (FAIL if rise90 > 0.20 or rise_late > 0.05 else
                 WARN if rise90 > 0.10 or rise_late > 0.02 else PASS)
        out.append(Check("起点到出声（尾部）", level,
                         f"p90 {candidate.onset.p90:.2f}s（{rise90:+.2f}）  "
                         f"落在静音里 {candidate.onset.late_share:.1%}"
                         f"（{rise_late:+.1%}）  "
                         f"中位 {candidate.onset.median:.2f}s（只报告，"
                         f"校正后下限就是 {0.10:.2f}s）"))

    if arbitration is not None and arbitration.disputed:
        allowed = max(3, int(arbitration.disputed * 0.10))
        lead = arbitration.left - arbitration.right   # left = baseline
        out.append(Check("争议起点音频仲裁",
                         FAIL if lead > allowed else PASS,
                         f"基线 {arbitration.left} : {arbitration.right} 候选"
                         f"（平手 {arbitration.tie}，容许基线多赢 {allowed}）"))

    if far_share_delta is not None:
        out.append(Check("起点差 >2s（同句同切法）",
                         FAIL if far_share_delta > 0.10 else
                         WARN if far_share_delta > 0.03 else PASS,
                         f"{far_share_delta:+.1%} 相对基线"))

    # --- punctuation, which decides which segmenter path a film takes ---
    # Relative, not against a target. 0.5 is segmenter's structural switch
    # (is_effectively_unpunctuated): above it every sentence merge is
    # deferred to refine. So crossing it upwards is a failure whatever the
    # numbers, and otherwise the question is only "worse than the baseline".
    # Judging against an absolute target instead fails a run compared with
    # itself, because whisper's Japanese is 80% open-ended and no change in
    # this round addresses that on the local route.
    ratios = [b.open_ended for b in baselines if M.nan_safe(b.open_ended)]
    if ratios and M.nan_safe(candidate.open_ended):
        reference = statistics.fmean(ratios)
        switch = segmenter_switch()
        crossed_up = candidate.open_ended > switch >= reference
        rise = candidate.open_ended - reference
        level = (FAIL if crossed_up or rise > 0.15 else
                 WARN if rise > 0.05 else PASS)
        note = ""
        if reference > switch >= candidate.open_ended:
            note = f"——越过了 {switch:.0%} 那道结构开关，整句合并不再推迟给 refine"
        elif crossed_up:
            note = f"——越过了 {switch:.0%}，整句合并被推迟给 refine"
        out.append(Check("分句后未完句占比", level,
                         f"{candidate.open_ended:.0%} vs 基线 {reference:.0%}"
                         f"（{rise:+.0%}）{note}"))

    broken = sum(candidate.faults.values())
    inherited = min(sum(b.faults.values()) for b in baselines) if baselines else 0
    out.append(Check("cue 重叠/乱序", FAIL if broken > inherited else PASS,
                     ", ".join(f"{k}={v}" for k, v in candidate.faults.items())
                     + (f"（基线已有 {inherited} 处，只判新增）" if inherited else "")))
    return out


# ------------------------------------------------------------------ output


def show_side(side: Side, picture: M.VadPicture) -> None:
    run = side.run
    print(f"  {side.label}  [{run.engine} v{run.version or '?'}]")
    print(f"    第一遍/二次/分句/比较 : {len(run.raw) or 'n/a':>6} / "
          f"{len(run.recovered) or 'n/a':>4} / {side.segmented:>5} / {side.finals:>5}"
          f"   原文 {side.characters} 字  （{run.sources.get('final', '?')}）")
    if run.vet:
        print(f"    复核                 : 送审 {run.vet.get('submitted') or 'n/a'}，"
              f"保留 {run.vet.get('kept')} / 丢弃 {run.vet.get('dropped')}"
              + (f"  ⚠ {'; '.join(run.vet.get('failures') or [])}"
                 if run.vet.get("failures") else ""))
        reasons = run.vet.get("reasons") or {}
        if reasons:
            print("    丢弃理由             : " + ", ".join(
                f"{k} × {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:5]))
    coverage = side.coverage
    share = "n/a" if coverage.share is None else f"{coverage.share:.1%}"
    print(f"    同口径覆盖率         : {share}"
          f"（{run.sources.get('coverage', '?')} {coverage.transcribed:.0f}s "
          f"∩ silero {coverage.inside:.0f}s）"
          + ("   ⚠ VAD 失聪，仅供参考" if coverage.blind else ""))
    print(f"    转写落在 VAD 内      : {coverage.transcript_in_vad:.0%}"
          f"   silero 占响亮音频 {picture.vad_share:.0%}")
    print(f"    起点到出声           : {side.onset.describe()}")
    print(f"    分句后未完句结尾     : "
          + ("n/a" if not M.nan_safe(side.open_ended) else f"{side.open_ended:.0%}"))
    faults = ", ".join(f"{k}={v}" for k, v in side.faults.items())
    print(f"    最终 cue 秩序        : {faults}")
    if run.minutes:
        stages = "  ".join(f"{t.split('（')[0]} {m:.1f}′" for t, m in run.minutes.items()
                           if any(k in t for k in ("语音识别原始", "复核", "分句", "最终")))
        print(f"    阶段用时             : {stages}")
    if run.notes:
        print(f"    备注                 : {'; '.join(run.notes)}")


def show_pairing(left: Side, right: Side, audio, apart: float) -> tuple:
    pairs = M.pair_by_text(left.lines, right.lines)
    deltas = M.start_deltas(pairs)
    print(f"\n  配对 {len(pairs)} 句（{left.label} 共 {left.finals} 条）")
    print(f"    起点差（{right.label} − {left.label}）: {M.describe_deltas(deltas)}")
    if pairs:
        third = max(1, len(pairs) // 3)
        parts = [deltas[:third], deltas[third:2 * third], deltas[2 * third:]]
        print("    三段中位             : " + "  ".join(
            f"{statistics.median(p):+.2f}s" for p in parts if p))
    tight = M.arbitrate(pairs, audio, apart=apart)
    print(f"    争议起点音频仲裁 6dB : {tight.describe(left.label, right.label)}")
    loose = M.arbitrate(pairs, audio, apart=apart, margin_db=3.0)
    print(f"    ……同上 3dB 边际      : {loose.describe(left.label, right.label)}")
    far = M.far_share(pairs)
    comparable = M.comparable(pairs)
    print(f"    同一句、同样切法的配对 : {len(comparable)} 句，其中起点差 >2s 的 "
          + ("n/a" if far is None else f"{far:.1%}")
          + "（只有这些能说明「放错位置」；切法不同的差异归到下一节）")
    return pairs, tight, far


def show_exclusive(left: Side, right: Side, top: int = 6) -> float:
    only_left = M.exclusive(left.lines, right.lines)
    only_right = M.exclusive(right.lines, left.lines)
    left_total = M.total(M.spans_of(left.lines)) or 1.0
    print(f"\n  只有 {left.label} 有（≥1.5s）: {len(only_left)} 处 / "
          f"{M.total(only_left):.0f}s = {M.total(only_left) / left_total:.1%} 的基线时长")
    for start, end in sorted(only_left, key=lambda g: g[0] - g[1])[:top]:
        print(f"    {M.clock(start)}–{M.clock(end)} ({end - start:4.1f}s) | "
              f"{M.said_between(left.lines, start, end)}")
    print(f"  只有 {right.label} 有（≥1.5s）: {len(only_right)} 处 / {M.total(only_right):.0f}s")
    for start, end in sorted(only_right, key=lambda g: g[0] - g[1])[:top]:
        print(f"    {M.clock(start)}–{M.clock(end)} ({end - start:4.1f}s) | "
              f"{M.said_between(right.lines, start, end)}")
    return M.total(only_left) / left_total


# --------------------------------------------------------------------- main


def resolve_film(name: str, mode: str) -> tuple[List[Path], List[Path], Optional[Path]]:
    """Baseline, candidate and audio for ``--film``, out of the archive."""
    home = ARCHIVE / name
    if not home.is_dir():
        raise SystemExit(f"找不到归档目录: {home}")
    audio = home / "audio.wav"
    if mode == "cross":
        return [home / "large-v2"], [home / "api"], audio if audio.exists() else None
    arm = "api" if mode == "api" else "large-v2"
    return [home / arm], [], audio if audio.exists() else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_runs", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--film", help="归档片名，取 text/<片名>/ 下的基线与音频")
    parser.add_argument("--baseline", action="append", default=[], metavar="路径",
                        help="基线目录或文件，可重复；多份基线用来量离散度")
    parser.add_argument("--candidate", action="append", default=[], metavar="路径")
    parser.add_argument("--audio", help="片源或 16kHz wav")
    parser.add_argument("--mode", choices=("api", "local", "cross"), default="cross")
    parser.add_argument("--stage", choices=("final", "segmented"), default="final",
                        help="比哪一层：最终字幕，还是分句后（识别阶段的产物）")
    parser.add_argument("--work", default=None, help="临时目录（提音频、VAD 缓存）")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--apart", type=float, default=1.5,
                        help="起点相差多少秒才算争议（默认 1.5）")
    parser.add_argument("--json", action="store_true", help="机器可读的结果")
    args = parser.parse_args(argv)

    baselines_paths = [Path(p) for p in args.baseline]
    candidate_paths = [Path(p) for p in args.candidate]
    audio_path = Path(args.audio) if args.audio else None
    if args.film:
        found_base, found_candidate, found_audio = resolve_film(args.film, args.mode)
        baselines_paths = baselines_paths or found_base
        candidate_paths = candidate_paths or found_candidate
        audio_path = audio_path or found_audio
    if not baselines_paths:
        parser.error("需要 --baseline 或 --film")

    # A path that is not there must stop the run, not quietly become an
    # empty side. Measured the hard way: three baseline directories were
    # named that had not been fetched yet, every metric read 0, and the
    # table came back all PASS — a green light produced by the mistake it
    # should have caught. This tool exists to be trusted about "not
    # worse", so it may not answer at all rather than answer from nothing.
    missing = [p for p in baselines_paths + candidate_paths if not p.exists()]
    if missing:
        raise SystemExit("这些路径不存在：\n  "
                         + "\n  ".join(str(p) for p in missing))

    work = Path(args.work) if args.work else Path(
        __import__("tempfile").gettempdir()) / "compare_runs"
    if audio_path and audio_path.suffix.lower() not in (".wav",):
        from app.services import audio as audio_service
        work.mkdir(parents=True, exist_ok=True)
        extracted = work / (audio_path.stem + ".16k.wav")
        if not extracted.exists():
            audio_service.extract_audio(str(audio_path), str(extracted))
        audio_path = extracted

    if audio_path is None or not audio_path.exists():
        raise SystemExit("需要 --audio（片源或 wav）：覆盖率、起点与仲裁都要听音频")

    samples = M.load_audio(audio_path)
    picture = vad_picture_cached(samples, audio_path,
                                 None if args.no_cache else work / "vad")

    baselines = [measure(runparse.load_run([p], p.parent.name + "/" + p.name),
                         picture, samples, args.stage) for p in baselines_paths]
    candidates = [measure(runparse.load_run([p], p.parent.name + "/" + p.name),
                          picture, samples, args.stage) for p in candidate_paths]

    title = args.film or baselines[0].label
    print("=" * 78)
    print(f"{title}   模式 {args.mode}   比较层 {args.stage}   "
          f"音频 {picture.duration:.0f}s")
    print(f"silero（阈值 {asr.REFERENCE_VAD_THRESHOLD}）"
          f" 语音 {picture.speech:.0f}s，人声电平中位 {picture.speech_db:.1f} dBFS，"
          f"音量接近人声 {picture.loud:.0f}s，silero 占其中 {picture.vad_share:.0%}")
    print("=" * 78)

    print("\n一、每一侧自己的数字")
    for side in baselines + candidates:
        show_side(side, picture)

    checks: List[Check] = []
    if candidates:
        left, right = baselines[0], candidates[0]
        print("\n二、同一句话，两边的时间轴")
        _, arbitration, far = show_pairing(left, right, samples, args.apart)
        print("\n三、只有一边有的内容")
        exclusive_share = show_exclusive(left, right)

        far_baseline = None
        if len(baselines) > 1:
            print("\n二之二、基线彼此（这就是噪声底）")
            _, _, far_baseline = show_pairing(baselines[0], baselines[1], samples, args.apart)
        far_delta = None if far is None else far - (far_baseline or 0.0)

        if args.mode != "cross":
            print("\n四、判定")
            checks = verdicts(right, baselines, args.mode, arbitration,
                              exclusive_share, far_delta)
            width = max(len(c.name) for c in checks)
            for check in checks:
                print(f"  [{check.level}] {check.name:<{width}}  {check.detail}")
        else:
            print("\n（cross 模式：两条引擎不是彼此的基线，只出表不判定）")

    if args.json:
        print(json.dumps({
            "film": title, "mode": args.mode, "duration": picture.duration,
            "vad_share": picture.vad_share,
            "sides": [{
                "label": s.label, "engine": s.run.engine, "finals": s.finals,
                "characters": s.characters, "coverage": s.coverage.share,
                "transcript_in_vad": s.coverage.transcript_in_vad,
                "vad_blind": s.coverage.blind, "onset_median": s.onset.median,
                "onset_p90": s.onset.p90, "open_ended": s.open_ended,
                "faults": s.faults,
            } for s in baselines + candidates],
            "checks": [{"name": c.name, "level": c.level, "detail": c.detail}
                       for c in checks],
        }, ensure_ascii=False, indent=2))

    return 1 if any(c.level == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
