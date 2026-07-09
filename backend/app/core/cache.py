"""Local cache directory management.

All intermediate artifacts (extracted audio, transcript JSON, translation JSON)
live under the platform cache dir and are wiped on every application startup.
"""

from __future__ import annotations

import shutil
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
