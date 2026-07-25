"""Deep-diagnostics log (设置 → 调试模式), off by default.

The regular job log (``core/joblog.py``) records what the pipeline *did*.
This one records what it *saw and decided*: raw whisper segments with
word-level timestamps and per-segment confidence, every line the segmenter
produced before and after merging, every merge the refine pass proposed
together with the reason a rejected one was rejected, and the full LLM
traffic.

That is the difference between "the subtitles look wrong" and a diagnosis:
the failure mode that motivated this file (single characters ending up as
their own cue and being translated into half a word) is invisible in the
finished SRT — you cannot tell whether the split came from whisper's word
timestamps, from the segmenter's gap rule, or from a merge the refine pass
vetoed. All three are recorded here.

The file is large (a two-hour film runs to a few MB), which is why the
mode is opt-in. It is written next to the subtitle output so a user can
attach it to a bug report in one step.

Nothing secret is written: the API key is recorded as configured or not,
never its value — same rule as the job log.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

MAX_TEXT_BLOCK = 200_000  # a single LLM message longer than this gets clipped


def debug_path_for(subtitle_target: Path, fallback_dir: Path) -> Path:
    """`<video stem>.debug.log` next to the subtitle, or in the work dir."""
    candidate = subtitle_target.with_suffix(".debug.log")
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        with candidate.open("a", encoding="utf-8"):
            pass
        return candidate
    except OSError:
        return fallback_dir / candidate.name


class DebugLog:
    """Append-only diagnostics writer. A disabled instance is a no-op.

    Every method tolerates being called on a disabled or broken log, so
    call sites never need a guard — diagnostics must not be able to fail
    a job.
    """

    def __init__(self, path: Optional[Path] = None, enabled: bool = False):
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self.enabled = bool(enabled and path is not None)
        self.path: Optional[Path] = path if self.enabled else None
        if not self.enabled:
            return
        try:
            assert self.path is not None
            from app.core.joblog import APP_VERSION

            self.path.write_text("", encoding="utf-8")
            self._write(
                f"=== MovieTranslator {APP_VERSION} 调试日志 ===\n"
                f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                f"说明: 本文件用于排查字幕质量问题，可直接发给开发者。\n"
                f"      不含 API key 等敏感信息。\n"
            )
        except OSError:
            self.enabled = False
            self.path = None

    # ---------------------------------------------------------- primitives

    def _write(self, text: str) -> None:
        if not self.enabled or self.path is None:
            return
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            self.enabled = False  # disk full / unplugged: stop trying

    def section(self, title: str) -> None:
        elapsed = time.monotonic() - self._started
        self._write(
            f"\n\n{'=' * 78}\n== {title}"
            f"    [任务开始后 {elapsed / 60:.1f} 分钟]\n{'=' * 78}\n"
        )

    def line(self, text: str = "") -> None:
        self._write(text + "\n")

    def lines(self, items: Iterable[str]) -> None:
        self._write("".join(f"{item}\n" for item in items))

    def kv(self, key: str, value) -> None:
        self._write(f"{key:<28}: {value}\n")

    def block(self, title: str, body: str) -> None:
        """A verbatim payload (an LLM message), fenced so it is greppable."""
        if not self.enabled:
            return
        if len(body) > MAX_TEXT_BLOCK:
            body = body[:MAX_TEXT_BLOCK] + f"\n…（已截断，原长度 {len(body)} 字符）"
        self._write(f"\n--- {title} ---\n{body}\n--- /{title} ---\n")


# --------------------------------------------------------------- helpers
# Formatting shared by the modules that report into the log, kept here so
# the services stay free of presentation code.


def fmt_time(seconds: float) -> str:
    m, s = divmod(max(seconds, 0.0), 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}"


def fmt_cue(index: int, start: float, end: float, text: str, extra: str = "") -> str:
    return (
        f"[{index:>4}] {fmt_time(start)} → {fmt_time(end)} "
        f"({end - start:5.2f}s) {extra}| {text}"
    )


def percentiles(values: Sequence[float], points=(5, 25, 50, 75, 95)) -> str:
    if not values:
        return "（无数据）"
    ordered = sorted(values)
    parts = []
    for p in points:
        idx = min(int(len(ordered) * p / 100), len(ordered) - 1)
        parts.append(f"P{p}={ordered[idx]:.2f}")
    return "  ".join(parts) + f"  (n={len(ordered)})"
