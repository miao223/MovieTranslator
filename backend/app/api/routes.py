"""REST + SSE endpoints."""

from __future__ import annotations

import os
import shutil
import string
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
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


@router.get("/asr/cuda-status")
def asr_cuda_status():
    """Whether CUDA is usable by ctranslate2 on this machine."""
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        return {"available": count > 0, "device_count": count}
    except Exception as exc:  # noqa: BLE001 — missing CUDA libs land here
        return {"available": False, "device_count": 0, "error": str(exc)}


# ---------------------------------------------------------------- upload


@router.post("/upload")
def upload_video(file: UploadFile):
    """Receive a drag-dropped video into the work dir and return its path.

    Browsers cannot reveal the local path of a dropped file, so drag & drop
    uploads a copy into the managed cache (wiped on next startup).
    """
    from app.core.cache import cache_root

    name = Path(file.filename or "video").name
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的视频格式: {name}")
    dest_dir = cache_root() / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f, length=1024 * 1024)
    return {"path": str(dest), "size": dest.stat().st_size}


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

    parent = str(p.parent) if p.parent != p else None
    if sys.platform == "win32" and parent == str(p):
        parent = ""  # back to drive list
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files}
