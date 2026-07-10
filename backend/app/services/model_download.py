"""Pre-download whisper models from the settings page, with progress.

State is kept in-process (one entry per model size); the frontend polls
GET /api/asr/download-status. Downloads honour the model-download proxy
switch via asr.proxy_env().
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from app.models.schemas import NetworkSettings
from app.services.asr import proxy_env

_state: Dict[str, dict] = {}
_state_lock = threading.Lock()

# only the files faster-whisper actually needs (mirrors its download_model)
_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def get_status(model_size: str) -> dict:
    with _state_lock:
        return dict(_state.get(model_size, {"status": "idle", "progress": 0.0}))


def _set_status(model_size: str, **fields) -> None:
    with _state_lock:
        entry = _state.setdefault(model_size, {"status": "idle", "progress": 0.0})
        entry.update(fields)


def _repo_id(model_size: str) -> str:
    from faster_whisper.utils import _MODELS

    from app.services.asr import resolve_model

    resolved = resolve_model(model_size)
    if resolved in _MODELS:
        return _MODELS[resolved]
    if "/" in resolved:  # already a HuggingFace repo id (CT2 format)
        return resolved
    raise ValueError(f"未知模型: {model_size}")


def _make_progress_tqdm(model_size: str):
    """A tqdm subclass that aggregates byte progress across all files."""
    from tqdm import tqdm

    totals_lock = threading.Lock()
    totals: Dict[int, tuple] = {}  # id(bar) -> (n, total)

    def _report():
        with totals_lock:
            done = sum(n for n, _ in totals.values())
            total = sum(t for _, t in totals.values() if t)
        if total:
            _set_status(
                model_size,
                progress=round(done / total * 100, 1),
                downloaded_bytes=done,
                total_bytes=total,
            )

    class ProgressTqdm(tqdm):
        def update(self, n=1):
            result = super().update(n)
            # only byte-unit bars reflect actual file downloads
            if self.unit in ("B", "iB", "bytes") or (self.total or 0) > 10_000:
                with totals_lock:
                    totals[id(self)] = (self.n, self.total or 0)
                _report()
            return result

    return ProgressTqdm


def start_download(model_size: str, network: Optional[NetworkSettings]) -> dict:
    """Kick off a background download; idempotent while one is running."""
    repo = _repo_id(model_size)  # validates the name before touching state
    current = get_status(model_size)
    if current["status"] == "downloading":
        return current
    _set_status(model_size, status="downloading", progress=0.0, error=None)

    def run():
        from huggingface_hub import snapshot_download

        try:
            from app.services.asr import get_model_cache_dir

            with proxy_env(network):
                snapshot_download(
                    repo,
                    allow_patterns=_ALLOW_PATTERNS,
                    tqdm_class=_make_progress_tqdm(model_size),
                    cache_dir=get_model_cache_dir(),
                )
            _set_status(model_size, status="done", progress=100.0)
        except Exception as exc:  # noqa: BLE001 — shown in the UI
            _set_status(model_size, status="failed", error=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return get_status(model_size)
