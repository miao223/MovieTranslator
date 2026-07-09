"""Local cache directory management.

All intermediate artifacts (extracted audio, transcript JSON, translation JSON)
live under the platform cache dir and are wiped on every application startup.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from platformdirs import user_cache_dir

APP_NAME = "MovieTranslator"


def cache_root() -> Path:
    root = Path(user_cache_dir(APP_NAME)) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def clear_cache() -> None:
    root = Path(user_cache_dir(APP_NAME)) / "jobs"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    d = cache_root() / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d
