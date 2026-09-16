"""Local cache directory management.

All intermediate artifacts (extracted audio, transcript JSON, translation JSON)
live under the platform cache dir and are wiped on every application startup.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from platformdirs import user_cache_dir

APP_NAME = "MovieTranslator"


def _base_dir() -> Path:
    """Configured work dir, or the platform cache dir when unset.

    Imported lazily to avoid a circular import (config imports APP_NAME
    from this module).
    """
    try:
        from app.core import config

        work_dir = config.load_settings().work_dir.strip()
        if work_dir:
            return Path(work_dir)
    except Exception:
        pass
    return Path(user_cache_dir(APP_NAME))


def cache_root() -> Path:
    root = _base_dir() / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def clear_cache() -> None:
    # only the managed "jobs" subdir is wiped, never the user's folder itself
    root = _base_dir() / "jobs"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    d = cache_root() / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------- checkpoints
#
# A sibling of "jobs", and deliberately outside it: clear_cache wipes jobs
# on every startup, which is what made resuming impossible. These survive,
# so a film interrupted at 90% does not have to buy its audio and its
# transcription a second time.
#
# The directory name is the whole validity check. It hashes the things that
# would make a half-finished result wrong to carry on from — the source
# file, the settings it was started under, and the program version — so a
# key that still matches cannot be stale in any of those ways. Nothing has
# to be remembered or compared later; a mismatch simply lands somewhere
# else and starts fresh.

# Defaults; the user's own limits come from settings (see _limits).
CHECKPOINT_DAYS = 7
# Audio is kept too (it is the expensive part of a restart), and that is
# ~230 MB for a two-hour film, so age alone is not enough of a bound.
MAX_CHECKPOINT_GB = 10


def _limits() -> tuple[int, int]:
    """(days, bytes) from settings, falling back to the defaults."""
    try:
        from app.core import config

        s = config.load_settings()
        return s.checkpoint_days, s.checkpoint_max_gb * 1024 ** 3
    except Exception:  # noqa: BLE001 — pruning must never fail a job
        return CHECKPOINT_DAYS, MAX_CHECKPOINT_GB * 1024 ** 3


def checkpoints_enabled() -> bool:
    days, cap = _limits()
    return days > 0 and cap > 0


def checkpoints_usage() -> dict:
    """What the kept work is costing, for the settings page to show."""
    root = checkpoints_root()
    dirs = [d for d in root.iterdir() if d.is_dir()]
    return {"dir": str(root), "count": len(dirs),
            "bytes": sum(_dir_size(d) for d in dirs)}


def checkpoints_root() -> Path:
    root = _base_dir() / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def checkpoint_dir(key: str) -> Path:
    d = checkpoints_root() / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def prune_checkpoints() -> int:
    """Drop what is too old, then what does not fit. Returns bytes freed."""
    root = checkpoints_root()
    days, cap = _limits()
    if days <= 0 or cap <= 0:
        # resuming turned off: keep nothing at all
        freed = _dir_size(root)
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return freed
    cutoff = time.time() - days * 86400
    freed = 0
    alive: list[tuple[float, int, Path]] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        try:
            touched = max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()),
                          default=d.stat().st_mtime)
            size = _dir_size(d)
        except OSError:
            continue
        if touched < cutoff:
            freed += size
            shutil.rmtree(d, ignore_errors=True)
        else:
            alive.append((touched, size, d))

    total = sum(size for _, size, _ in alive)
    alive.sort()                       # oldest first
    for touched, size, d in alive:
        if total <= cap:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= size
        freed += size
    return freed


def drop_checkpoint(key: str) -> None:
    """Forget one half-finished run — 'start this over' means start over."""
    d = checkpoints_root() / key
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
