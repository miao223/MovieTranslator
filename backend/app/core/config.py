"""Persistent settings stored in the platform user config directory.

Unlike the cache, settings (including the LLM API key) survive restarts.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from platformdirs import user_config_dir

from app.core.cache import APP_NAME
from app.models.schemas import AppSettings

_lock = threading.Lock()

# (mtime, size) of the file the cached value was parsed from. The access
# guard consults the settings on every single request, and re-reading plus
# re-validating the JSON that often is pure waste; keying on the stat means
# an edit made by any process (or by save_settings below) still wins.
_cache: tuple[tuple[float, int], AppSettings] | None = None


def settings_path() -> Path:
    d = Path(user_config_dir(APP_NAME))
    d.mkdir(parents=True, exist_ok=True)
    return d / "settings.json"


def load_settings() -> AppSettings:
    global _cache

    path = settings_path()
    try:
        stat = path.stat()
        stamp = (stat.st_mtime, stat.st_size)
    except OSError:
        return AppSettings()  # no settings file yet

    cached = _cache
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        settings = AppSettings.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        # corrupt or outdated settings file: fall back to defaults
        return AppSettings()
    _cache = (stamp, settings)
    return settings


def save_settings(settings: AppSettings) -> None:
    global _cache

    with _lock:
        _cache = None
        settings_path().write_text(
            settings.model_dump_json(indent=2), encoding="utf-8"
        )
