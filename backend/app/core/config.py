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


def settings_path() -> Path:
    d = Path(user_config_dir(APP_NAME))
    d.mkdir(parents=True, exist_ok=True)
    return d / "settings.json"


def load_settings() -> AppSettings:
    path = settings_path()
    if path.exists():
        try:
            return AppSettings.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            # corrupt or outdated settings file: fall back to defaults
            pass
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    with _lock:
        settings_path().write_text(
            settings.model_dump_json(indent=2), encoding="utf-8"
        )
