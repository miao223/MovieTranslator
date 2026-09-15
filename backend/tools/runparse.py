"""Read back what one run left behind: debug log, job log, SRT, JSON.

A run of this pipeline scatters its evidence across four kinds of file and
two engines write them differently, so every comparison so far has been a
throwaway script that parsed one of them one way. This module is that
parsing, once, with the shapes named.

Nothing here computes a verdict — see ``runmetrics`` for the numbers and
``compare_runs`` for the judgement. The split matters because the parsing
is the part that has to be right about a format we do not control.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ------------------------------------------------------------------ shapes


@dataclass
class Cue:
    start: float
    end: float
    text: str
    translation: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    logprob: Optional[float] = None
    no_speech: Optional[float] = None
    compression: Optional[float] = None
    words: List[Word] = field(default_factory=list)


@dataclass
class Run:
    """Everything one run of the pipeline said about itself."""

    label: str = ""
    engine: str = "unknown"          # "local" | "api" | "unknown"
    version: str = ""
    duration: Optional[float] = None  # seconds of audio

    raw: List[Segment] = field(default_factory=list)        # first pass
    recovered: List[Segment] = field(default_factory=list)  # second pass
    segmented: List[Cue] = field(default_factory=list)      # after the segmenter
    final: List[Cue] = field(default_factory=list)          # what shipped

    vet: Dict[str, object] = field(default_factory=dict)
    kv: Dict[str, Dict[str, str]] = field(default_factory=dict)  # section -> key -> value
    minutes: Dict[str, float] = field(default_factory=dict)      # section -> elapsed
    joblog: Dict[str, object] = field(default_factory=dict)
    windows: List[dict] = field(default_factory=list)

    post_vet: bool = False  # the segments are what survived the LLM review
    files: Dict[str, Path] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)  # figure -> where it came from
    notes: List[str] = field(default_factory=list)

    def section(self, contains: str) -> Dict[str, str]:
        """kv pairs of the first section whose title contains *contains*."""
        for title, pairs in self.kv.items():
            if contains in title:
                return pairs
        return {}


# ------------------------------------------------------------------ regexes

_SECTION_RE = re.compile(r"^== (?P<title>.+?)\s{2,}\[任务开始后 (?P<minutes>[\d.]+) 分钟\]\s*$")
_KV_RE = re.compile(r"^(?P<key>\S[^:：]*?)\s*:\s(?P<value>.*)$")
_SEGMENT_RE = re.compile(
    r"^\[\s*(?P<start>[\d.]+)\s*→\s*(?P<end>[\d.]+)\]\s+logprob=(?P<logprob>-?[\d.]+)"
    r"\s+no_speech=(?P<no_speech>[\d.]+)\s+compress=(?P<compress>[\d.]+)"
)
# `152.24-152.46 talk` in English, `21.68-22.20心` in Japanese — CJK words
# are written straight after the timestamp with no space, and a long gap
# between two words is annotated inline as `⟨间隔1.3s⟩`. Requiring the
# space silently dropped every Japanese word span, which in turn made
# asr._word_spans fall back for some segments and not others.
_WORD_RE = re.compile(
    r"(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)\s?"
    r"(?P<text>.*?)(?=\s+\d+\.\d+-\d+\.\d+|$)"
)
_GAP_NOTE_RE = re.compile(r"\s*⟨[^⟩]*⟩\s*$")
_NUMBERED_RE = re.compile(
    r"^\[\s*(?P<index>\d+)\]\s+(?P<start>\d+:\d\d:\d\d\.\d+)\s*→\s*"
    r"(?P<end>\d+:\d\d:\d\d\.\d+)\s*\(\s*(?P<duration>[\d.]+)s\)\s*"
    r"(?P<extra>.*?)\|\s*(?P<text>.*?)\s*$"
)
# 0.19.0 wrote the heading as MM:SS, later builds as H:MM:SS.mmm; both are
# read, and the older one only anchors its window to about a second.
_WINDOW_RE = re.compile(
    r"^--- \[(?P<start>[\d:.]+)–(?P<end>[\d:.]+)\]\s+(?P<lines>\d+) 行"
    r"\s+第 (?P<attempts>\d+) 次通过 ---\s*$"
)
_BLOCK_OPEN_RE = re.compile(r"^--- (?P<title>[\d:.]+–[\d:.]+) ---\s*$")
# The per-line verdict table, which is the one place both the number, the
# verdict and the reason appear together. The LLM's own reply also holds
# `[Rn] 保留` lines, so the trailing `| text` is required — that is what
# tells the table apart from the raw response block quoted above it.
_VERDICT_LINE_RE = re.compile(
    r"^\s*\[R(?P<number>\d+)\]\s+(?P<time>\d+:\d\d:\d\d\.\d+)\s+"
    r"(?P<verdict>保留|丢弃)(?:\s+(?P<reason>\S[^|]*?))?\s*\|\s*(?P<text>.*)$"
)
_VET_SUMMARY_RE = re.compile(
    r"^复核结果：保留 (?P<kept>\d+) 段 / (?P<kept_seconds>[\d.]+)s，"
    r"丢弃 (?P<dropped>\d+) 段 / (?P<dropped_seconds>[\d.]+)s"
)
_VET_FAILURE_RE = re.compile(r"^⚠ 第 (?P<chunk>\d+) 块第 (?P<attempt>\d+) 次失败：(?P<why>.*)$")
_VERSION_RE = re.compile(r"^=== MovieTranslator (?P<version>\S+) (调试日志|任务日志) ===")
_PROGRESS_RE = re.compile(
    r"^\[(?P<clock>\d\d:\d\d:\d\d)\] \[(?P<stage>\w+)\s+(?P<percent>[\d.]+)%\] (?P<text>.*)$"
)


def _clock_seconds(text: str) -> float:
    """'0:12:34.560' or '12:34' -> seconds."""
    parts = text.split(":")
    total = 0.0
    for part in parts:
        total = total * 60.0 + float(part)
    return total


# ------------------------------------------------------------- debug log


def sections(text: str) -> tuple[Dict[str, List[str]], Dict[str, float]]:
    """{title: body lines}, {title: minutes since the job started}."""
    bodies: Dict[str, List[str]] = {}
    minutes: Dict[str, float] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        found = _SECTION_RE.match(line)
        if found:
            current = found.group("title").strip()
            bodies[current] = []
            minutes[current] = float(found.group("minutes"))
            continue
        if current is None:
            continue
        if line and set(line) == {"="}:
            continue
        bodies[current].append(line)
    return bodies, minutes


def parse_kv(lines: Sequence[str]) -> Dict[str, str]:
    """The ``key : value`` header of a section, stopping where payload starts.

    Payload lines are indented, bracketed or fenced; a line of dialogue can
    contain ': ' and would otherwise be read as a key.
    """
    out: Dict[str, str] = {}
    for line in lines:
        if line.startswith(("[", "---", "  ", "\t")):
            break
        found = _KV_RE.match(line)
        if found:
            out[found.group("key").strip()] = found.group("value").strip()
    return out


def parse_whisper_segments(lines: Sequence[str]) -> List[Segment]:
    """Raw faster-whisper segments, with their word timestamps."""
    out: List[Segment] = []
    pending: Optional[Segment] = None
    for line in lines:
        found = _SEGMENT_RE.match(line)
        if found:
            if pending is not None and pending.text:
                out.append(pending)
            pending = Segment(
                start=float(found.group("start")), end=float(found.group("end")),
                text="", logprob=float(found.group("logprob")),
                no_speech=float(found.group("no_speech")),
                compression=float(found.group("compress")),
            )
            continue
        if pending is None:
            continue
        stripped = line.strip()
        if stripped.startswith("词:"):
            pending.words = [
                Word(float(m.group("start")), float(m.group("end")),
                     _GAP_NOTE_RE.sub("", m.group("text")).strip())
                for m in _WORD_RE.finditer(stripped[2:])
            ]
            out.append(pending)
            pending = None
        elif stripped and not pending.text:
            pending.text = stripped
    if pending is not None and pending.text:
        out.append(pending)
    return out


def parse_api_windows(lines: Sequence[str]) -> tuple[List[Segment], List[dict]]:
    """Cues and per-window facts out of the API engine's raw section.

    The replies are parsed by ``asr_api.parse_reply`` itself rather than by
    a second regex here — the tool must read a window exactly the way the
    engine read it. Window starts come from the ``MM:SS`` heading, so every
    absolute time here is good to about a second; ``transcript.json`` is
    preferred wherever it exists.
    """
    from app.services import asr_api

    out: List[Segment] = []
    windows: List[dict] = []
    heading: Optional[dict] = None
    body: Optional[List[str]] = None

    def flush() -> None:
        nonlocal body
        if heading is None or body is None:
            body = None
            return
        _, cues = asr_api.parse_reply("\n".join(body))
        offset = heading["start"]
        for start, end, text in cues:
            out.append(Segment(start=offset + start, end=offset + end, text=text))
        heading["parsed"] = len(cues)
        body = None

    for line in lines:
        found = _WINDOW_RE.match(line)
        if found:
            flush()
            heading = {
                "start": _clock_seconds(found.group("start")),
                "end": _clock_seconds(found.group("end")),
                "claimed": int(found.group("lines")),
                "attempts": int(found.group("attempts")),
            }
            windows.append(heading)
            continue
        if _BLOCK_OPEN_RE.match(line):
            body = []
            continue
        if line.startswith("--- /") and body is not None:
            flush()
            continue
        if body is not None:
            body.append(line)
    flush()
    out.sort(key=lambda s: s.start)
    return out, windows


def parse_numbered_cues(lines: Sequence[str], after: str = "") -> List[Cue]:
    """``[   1] 0:00:30.000 → 0:00:30.780 ( 0.78s) | text  ⇒  translation``.

    *after* names a marker line; only cues below it are read, which is how
    the segmenter section's two blocks (before and after merging) are told
    apart.
    """
    out: List[Cue] = []
    armed = not after
    for line in lines:
        if not armed:
            armed = line.startswith(after)
            continue
        if after and line.startswith(("合并前", "合并后", "最终结果")) and after not in line:
            break
        found = _NUMBERED_RE.match(line)
        if not found:
            continue
        text = found.group("text")
        translation = ""
        if "  ⇒  " in text:
            text, translation = text.split("  ⇒  ", 1)
        out.append(Cue(_clock_seconds(found.group("start")),
                       _clock_seconds(found.group("end")),
                       text.strip(), translation.strip()))
    return out


def parse_vet(lines: Sequence[str]) -> Dict[str, object]:
    """Submitted / kept / dropped, and why, out of the review section."""
    pairs = parse_kv(lines)
    kept = dropped = 0
    reasons: Dict[str, int] = {}
    failures: List[str] = []
    summary = ""
    for line in lines:
        failed = _VET_FAILURE_RE.match(line)
        if failed:
            failures.append(f"第 {failed.group('chunk')} 块第 "
                            f"{failed.group('attempt')} 次：{failed.group('why')}")
            continue
        if _VET_SUMMARY_RE.match(line):
            summary = line.strip()
            continue
        found = _VERDICT_LINE_RE.match(line)
        if not found:
            continue
        if found.group("verdict") == "保留":
            kept += 1
        else:
            dropped += 1
            reason = (found.group("reason") or "（未注明）").strip()
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "submitted": pairs.get("送审", ""),
        "chunks": pairs.get("分块数", ""),
        "kept": kept,
        "dropped": dropped,
        "reasons": reasons,
        "failures": failures,
        "summary": summary,
    }


# ------------------------------------------------------------- other files


def parse_srt(path: Path) -> List[Cue]:
    out: List[Cue] = []
    block: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        if line.strip():
            block.append(line)
            continue
        if len(block) >= 3 and "-->" in block[1]:
            start, end = block[1].split("-->")
            body = [b for b in block[2:] if b.strip()]
            # bilingual output puts the translation on the last line
            text = " ".join(body[:-1]).strip() if len(body) > 1 else body[0].strip()
            translation = body[-1].strip() if len(body) > 1 else ""
            out.append(Cue(
                _clock_seconds(start.strip().replace(",", ".")),
                _clock_seconds(end.strip().replace(",", ".")),
                text, translation,
            ))
        block = []
    return out


def parse_cue_json(path: Path) -> List[Cue]:
    """transcript.json / transcript_refined.json / translation.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Cue] = []
    for item in data:
        out.append(Cue(
            float(item.get("start", 0.0)), float(item.get("end", 0.0)),
            str(item.get("text", "")).strip(),
            str(item.get("translation", "") or "").strip(),
        ))
    return out


def parse_segment_json(path: Path) -> List[Segment]:
    """ASR segments as an arm dumps them: words and provenance intact.

    This is the post-review set — what the pipeline actually hands to the
    segmenter — so a coverage figure taken from it is the same quantity
    asr._report_coverage measures, which a debug log cannot give back
    (its two sections are before the review, not after).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Segment(
            start=float(item["start"]), end=float(item["end"]),
            text=str(item.get("text", "")).strip(),
            words=[Word(float(w["start"]), float(w["end"]), w.get("text", ""))
                   for w in item.get("words", [])],
        )
        for item in data
    ]


def parse_joblog(path: Path) -> Dict[str, object]:
    """Settings snapshot plus the progress lines, with their wall clock."""
    text = path.read_text(encoding="utf-8", errors="replace")
    settings: Dict[str, str] = {}
    progress: List[tuple[str, str, float, str]] = []
    for line in text.splitlines():
        found = _PROGRESS_RE.match(line)
        if found:
            progress.append((found.group("clock"), found.group("stage"),
                             float(found.group("percent")), found.group("text")))
            continue
        pair = _KV_RE.match(line)
        if pair and not line.startswith(("[", " ")):
            settings[pair.group("key").strip()] = pair.group("value").strip()
    version = ""
    first = _VERSION_RE.match(text.splitlines()[0] if text else "")
    if first:
        version = first.group("version")
    return {"settings": settings, "progress": progress, "version": version}


# ---------------------------------------------------------------- loading

# Longest suffix first: `transcript_refined.json` also ends in `.json`, and
# `.api.debug.log` also ends in `.log`.
_KINDS = (
    ("segments.json", "segments"),
    ("transcript_refined.json", "transcript_refined"),
    ("translation.json", "translation"),
    ("transcript.json", "transcript"),
    ("joblog.log", "joblog"),
    ("debug.log", "debug"),
    (".srt", "srt"),
    (".wav", "audio"),
)


def classify(paths: Sequence[Path]) -> Dict[str, Path]:
    """Map each file to the role its name gives it, newest wins."""
    out: Dict[str, Path] = {}
    for path in sorted(paths):
        name = path.name
        for suffix, kind in _KINDS:
            if name.endswith(suffix):
                out[kind] = path
                break
    return out


def collect(target: Path) -> Dict[str, Path]:
    """Files of a run: a directory's contents, or one file on its own."""
    if target.is_dir():
        return classify([p for p in sorted(target.iterdir()) if p.is_file()])
    return classify([target])


def load_run(targets: Sequence[Path], label: str = "") -> Run:
    """Assemble a Run from any mix of directories and files."""
    files: Dict[str, Path] = {}
    for target in targets:
        files.update(collect(Path(target)))

    run = Run(label=label or (Path(targets[0]).name if targets else "?"), files=files)

    if "debug" in files:
        text = files["debug"].read_text(encoding="utf-8", errors="replace")
        first = _VERSION_RE.match(text.splitlines()[0] if text else "")
        run.version = first.group("version") if first else ""
        bodies, minutes = sections(text)
        run.minutes = minutes
        run.kv = {title: parse_kv(body) for title, body in bodies.items()}
        for title, body in bodies.items():
            if "语音识别原始输出" in title:
                if "API 引擎" in title:
                    run.engine = "api"
                    run.raw, run.windows = parse_api_windows(body)
                    run.sources["raw"] = "debug 窗口块（起点 ±1s）"
                else:
                    run.engine = "local"
                    run.raw = parse_whisper_segments(body)
                    run.sources["raw"] = "debug 语音识别原始输出"
                    header = parse_kv(body)
                    if "音频时长" in header:
                        run.duration = float(header["音频时长"].rstrip("s"))
            elif "二次识别（" in title:
                run.recovered = parse_whisper_segments(body)
                run.sources["recovered"] = "debug 二次识别"
            elif "二次识别复核" in title:
                run.vet = parse_vet(body)
            elif "分句结果" in title:
                run.segmented = parse_numbered_cues(body, after="合并后")
                run.sources["segmented"] = "debug 分句结果（合并后）"
            elif "最终字幕" in title:
                run.final = parse_numbered_cues(body, after="最终结果")
                run.sources["final"] = "debug 最终字幕"

    if "segments" in files:
        data = json.loads(files["segments"].read_text(encoding="utf-8"))
        every = parse_segment_json(files["segments"])
        flags = [bool(item.get("recovered")) for item in data]
        run.raw = [s for s, r in zip(every, flags) if not r]
        run.recovered = [s for s, r in zip(every, flags) if r]
        run.post_vet = True
        run.engine = "local"
        run.sources["raw"] = "segments.json（复核后）"

    # JSON beats the log: exact floats, no ±1s from a MM:SS heading.
    if "transcript" in files:
        run.segmented = parse_cue_json(files["transcript"])
        run.sources["segmented"] = "transcript.json"
    if "translation" in files:
        run.final = parse_cue_json(files["translation"])
        run.sources["final"] = "translation.json"
    elif "srt" in files and not run.final:
        run.final = parse_srt(files["srt"])
        run.sources["final"] = "srt"

    if "joblog" in files:
        run.joblog = parse_joblog(files["joblog"])
        run.version = run.version or str(run.joblog.get("version", ""))
        settings = run.joblog.get("settings", {}) or {}
        if run.engine == "unknown":
            engine = str(settings.get("识别引擎", ""))
            run.engine = "api" if "API" in engine else "local" if engine else "unknown"

    if run.engine == "unknown" and run.raw:
        run.engine = "api" if not any(s.words for s in run.raw) else "local"
    if not run.final:
        run.notes.append("没有最终字幕可读")
    return run
