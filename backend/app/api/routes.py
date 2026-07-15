"""REST + SSE endpoints."""

from __future__ import annotations

import os
import string
import sys
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.core import config
from app.models.schemas import AppSettings, JobRequest, JobStatus, LLMSettings
from app.services.pipeline import manager

router = APIRouter(prefix="/api")

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".ts", ".m2ts", ".mpg", ".mpeg", ".rmvb", ".m4v",
}


# ------------------------------------------------------------------- jobs


@router.post("/jobs", response_model=JobStatus)
def create_job(req: JobRequest) -> JobStatus:
    try:
        job = manager.create(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _remember_dir(Path(req.video_path).parent)
    return job.status


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    try:
        return manager.get(job_id).status
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str):
    try:
        job = manager.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")

    def stream():
        for event in job.subscribe():
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    try:
        job = manager.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status.stage != "done" or not job.srt_path:
        raise HTTPException(status_code=409, detail="任务尚未完成")
    return FileResponse(
        job.srt_path, media_type="text/plain", filename=job.srt_path.name
    )


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        manager.cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True}


# --------------------------------------------------------------- settings


@router.get("/settings", response_model=AppSettings)
def get_settings() -> AppSettings:
    return config.load_settings()


@router.put("/settings", response_model=AppSettings)
def put_settings(settings: AppSettings) -> AppSettings:
    config.save_settings(settings)
    return settings


@router.post("/settings/test-llm")
def test_llm(llm: LLMSettings):
    from app.services.translator import make_openai_client

    try:
        # use the saved network settings so the proxy switch is exercised too
        client = make_openai_client(llm, config.load_settings().network)
        resp = client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": "ping，请回复 pong"}],
            max_tokens=8,
            temperature=0,
        )
        reply = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "reply": reply}
    except Exception as exc:  # noqa: BLE001 — report connectivity errors verbatim
        return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------- prompts


@router.post("/prompts/preview")
def prompts_preview(body: dict):
    """Assemble and return the final system prompt for the given settings.

    Body: { prompts: PromptSettings, target_language?, synopsis?, max_line_chars? }
    """
    from app.models.schemas import PromptSettings
    from app.services.translator import build_system_prompt

    prompts = PromptSettings.model_validate(body.get("prompts", {}))
    return {
        "prompt": build_system_prompt(
            prompts,
            target_language=body.get("target_language", "简体中文"),
            synopsis=body.get("synopsis", ""),
            max_line_chars=int(body.get("max_line_chars", 42)),
        )
    }


# ------------------------------------------------------------------ asr


@router.get("/asr/model-status")
def asr_model_status(model_size: str):
    """Whether the whisper model is already downloaded to the local cache.

    If a local model directory is configured in settings, it wins.
    """
    from app.services.asr import is_local_model_dir, is_model_cached

    model_path = config.load_settings().asr.model_path.strip()
    if model_path:
        ok = is_local_model_dir(model_path)
        return {
            "model_size": model_size,
            "downloaded": ok,
            "source": "local_path",
            "model_path": model_path,
            "valid": ok,
        }
    return {
        "model_size": model_size,
        "downloaded": is_model_cached(model_size),
        "source": "hub_cache",
    }


@router.post("/asr/download")
def asr_download(body: dict):
    """Start downloading a whisper model in the background (idempotent)."""
    from app.services import model_download

    model_size = str(body.get("model_size", "")).strip()
    try:
        return model_download.start_download(
            model_size, config.load_settings().network
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/asr/download-status")
def asr_download_status(model_size: str):
    from app.services import model_download

    return model_download.get_status(model_size)


@router.get("/asr/storage-info")
def asr_storage_info():
    """Where models are actually stored (custom dir or the HF default cache)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    from app.services.asr import get_model_cache_dir

    custom = get_model_cache_dir()
    return {
        "custom_dir": custom or "",
        "effective_dir": custom or str(HF_HUB_CACHE),
        "is_default": custom is None,
    }


@router.get("/asr/cuda-status")
def asr_cuda_status():
    """Whether CUDA is usable by ctranslate2 on this machine."""
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        return {"available": count > 0, "device_count": count}
    except Exception as exc:  # noqa: BLE001 — missing CUDA libs land here
        return {"available": False, "device_count": 0, "error": str(exc)}


# ------------------------------------------------------------ file locate

# directories worth checking when locating a drag-dropped file: browsers
# never reveal local paths, so we match by exact name + size instead
_recent_dirs: "deque[str]" = deque(maxlen=15)


def _remember_dir(path: Path) -> None:
    s = str(path)
    if s in _recent_dirs:
        _recent_dirs.remove(s)
    _recent_dirs.appendleft(s)


@router.post("/fs/locate")
def fs_locate(body: dict):
    """Find the local path of a dropped file by exact name + size match."""
    name = Path(str(body.get("name", ""))).name
    size = int(body.get("size") or 0)
    if not name:
        raise HTTPException(status_code=400, detail="缺少文件名")
    home = Path.home()
    candidates = [Path(d) for d in _recent_dirs] + [
        home / "Desktop", home / "Downloads", home / "Videos",
        home / "Movies", home,
    ]
    seen = set()
    for d in candidates:
        key = str(d)
        if key in seen or not d.is_dir():
            continue
        seen.add(key)
        p = d / name
        try:
            if p.is_file() and (size == 0 or p.stat().st_size == size):
                return {"found": True, "path": str(p)}
        except OSError:
            continue
    return {"found": False}


# ------------------------------------------------------------ file browse


@router.get("/fs/browse")
def fs_browse(path: str = ""):
    """List directories and video files for the server-side file picker."""
    if not path:
        if sys.platform == "win32":
            drives = [
                f"{letter}:\\" for letter in string.ascii_uppercase
                if os.path.exists(f"{letter}:\\")
            ]
            return {"path": "", "parent": None, "dirs": drives, "files": []}
        path = str(Path.home())

    p = Path(path)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"不是有效目录: {path}")
    _remember_dir(p)

    dirs, files = [], []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append(entry.name)
                elif entry.suffix.lower() in VIDEO_EXTS:
                    files.append(
                        {"name": entry.name, "size": entry.stat().st_size}
                    )
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"无权限访问: {path}")

    if p.parent == p:  # filesystem root: C:\ on Windows, / on Linux
        # Windows: "" navigates back to the drive list; Linux: no parent
        parent = "" if sys.platform == "win32" else None
    else:
        parent = str(p.parent)
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files}
