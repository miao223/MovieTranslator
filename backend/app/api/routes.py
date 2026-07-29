"""REST + SSE endpoints."""

from __future__ import annotations

import ipaddress
import os
import string
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from app.core import config, joblog, server
from app.core.auth import MCP_PREFIX
from app.core.media import VIDEO_EXTS, scan_videos
from app.models.schemas import (
    AppSettings,
    AudioTrack,
    BatchRequest,
    BatchStatus,
    JobRequest,
    JobStatus,
    LLMSettings,
)
from app.services import audio, mcp_server
from app.services.batch import batch_manager
from app.services.pipeline import manager

router = APIRouter(prefix="/api")


@router.get("/version")
def get_version() -> dict:
    """Shown in the page header: which build is actually running.

    The app is distributed as a zip and updated by replacing files, so
    "did the update take?" is a question that comes up on every release.
    """
    return {"version": joblog.APP_VERSION}


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


# ------------------------------------------------------------------ batch


@router.get("/batch/scan")
def batch_scan(path: str, recursive: bool = True, skip_existing: bool = True):
    """Preview which videos a batch would translate."""
    try:
        videos, skipped = scan_videos(path, recursive, skip_existing)
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "videos": [str(v) for v in videos],
        "skipped": [str(s) for s in skipped],
        "total": len(videos),
    }


@router.post("/batch", response_model=BatchStatus)
def create_batch(req: BatchRequest) -> BatchStatus:
    try:
        return batch_manager.create(req)
    except (NotADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/batch/{batch_id}", response_model=BatchStatus)
def get_batch(batch_id: str) -> BatchStatus:
    try:
        return batch_manager.status(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")


@router.post("/batch/{batch_id}/cancel")
def cancel_batch(batch_id: str):
    try:
        batch_manager.cancel(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")
    return {"ok": True}


# --------------------------------------------------------------- settings


# Stand-in for the API key when the settings are read from another machine.
# Sent back unchanged on save, which the handler below reads as "keep the
# stored one" — without that round trip, one save from a LAN browser would
# overwrite the real key with these asterisks.
MASKED = "********"


def _is_local(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    try:
        return ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        return False


@router.get("/settings", response_model=AppSettings)
def get_settings(request: Request) -> AppSettings:
    settings = config.load_settings()
    if _is_local(request):
        return settings
    remote = settings.model_copy(deep=True)
    if remote.llm.api_key:
        remote.llm.api_key = MASKED
    return remote


@router.put("/settings", response_model=AppSettings)
def put_settings(settings: AppSettings, request: Request) -> AppSettings:
    if settings.llm.api_key == MASKED:
        settings.llm.api_key = config.load_settings().llm.api_key
    # a LAN user who just switched the switch on still needs a way in
    server.ensure_token(settings)
    config.save_settings(settings)
    return get_settings(request)


# ----------------------------------------------------------------- server


@router.get("/server/info")
def server_info(request: Request) -> dict:
    """Where the app listens, where it could be reached, and MCP's state.

    `running` is what was actually bound and `configured` is what is saved;
    they differ after a change until the next restart, which is the whole
    reason both are reported.
    """
    settings = config.load_settings()
    srv, local = settings.server, _is_local(request)
    ips = server.lan_ips()
    # only the links need the token stripped out when it is not required;
    # the settings page still shows the stored one, so switching the
    # requirement back on does not mean hunting for it
    suffix = (
        f"/?token={srv.access_token}"
        if srv.require_token and srv.access_token
        else "/"
    )

    return {
        "configured": {
            "lan_access": srv.lan_access,
            "port": srv.port,
            "require_token": srv.require_token,
            "has_token": bool(srv.access_token),
        },
        "running": dict(server.RUNNING) or None,
        "lan_ips": ips,
        "urls": {
            "local": f"http://127.0.0.1:{srv.port}/",
            "lan": [f"http://{ip}:{srv.port}{suffix}" for ip in ips],
            "mcp": [f"http://{ip}:{srv.port}{MCP_PREFIX}" for ip in ips],
            "mcp_local": f"http://127.0.0.1:{srv.port}{MCP_PREFIX}",
        },
        "mcp": {
            "enabled": settings.mcp.enabled,
            "available": mcp_server.AVAILABLE,
            "error": mcp_server.IMPORT_ERROR,
        },
        # never handed to a remote caller: they would have to already know it
        "token": srv.access_token if local else "",
    }


@router.post("/server/token/regenerate")
def regenerate_token(request: Request) -> dict:
    """Issue a new access token, invalidating every device already let in.

    Loopback only — a caller holding the current token must not be able to
    rotate the owner out of their own server.
    """
    if not _is_local(request):
        raise HTTPException(status_code=403, detail="只能在运行本程序的机器上操作")
    settings = config.load_settings()
    settings.server.access_token = server.new_token()
    config.save_settings(settings)
    return {"token": settings.server.access_token}


@router.post("/settings/test-llm")
def test_llm(llm: LLMSettings):
    from app.services.translator import (
        _THINKING_OFF,
        _rejects_thinking,
        make_openai_client,
    )

    try:
        # use the saved network settings so the proxy switch is exercised too
        client = make_openai_client(llm, config.load_settings().network)
        # a question worth a moment's thought, so a model that reasons
        # actually will — "ping" is answered without thinking either way
        messages = [{"role": "user", "content": "3 和 5 哪个大？只回答数字。"}]
        extra = _THINKING_OFF if llm.disable_thinking else None
        accepted = extra is not None
        try:
            resp = client.chat.completions.create(
                model=llm.model, messages=messages, temperature=0,
                **({"extra_body": extra} if extra else {}),
            )
        except Exception as exc:  # noqa: BLE001
            if not (extra and _rejects_thinking(exc)):
                raise
            accepted = False
            resp = client.chat.completions.create(
                model=llm.model, messages=messages, temperature=0,
            )
        message = resp.choices[0].message
        reply = (getattr(message, "content", "") or "").strip()
        return {
            "ok": True,
            "reply": reply,
            "thinking": _describe_thinking(llm, resp, message, accepted),
        }
    except Exception as exc:  # noqa: BLE001 — report connectivity errors verbatim
        return {"ok": False, "error": str(exc)}


def _describe_thinking(llm, resp, message, accepted: bool) -> dict:
    """Did the model actually think, and did the switch take effect?

    Two independent signals, because either can be absent: the provider
    returns its reasoning as `reasoning_content` next to the answer, and
    reports `reasoning_tokens` in the usage block.
    """
    reasoning = (getattr(message, "reasoning_content", "") or "").strip()
    details = getattr(getattr(resp, "usage", None), "completion_tokens_details", None)
    tokens = getattr(details, "reasoning_tokens", 0) or 0
    thought = bool(reasoning) or tokens > 0

    if not llm.disable_thinking:
        state = "开启（模型返回了思考内容）" if thought else "开启（本次回复未产生思考内容）"
        return {"level": "info", "text": f"思考模式：{state}", "reasoning_tokens": tokens}
    if not accepted:
        return {
            "level": "warning",
            "text": "该服务不认识关闭思考的参数，已自动跳过——翻译不受影响，"
                    "但如果这个模型本身会思考，那部分开销无法避免",
            "reasoning_tokens": tokens,
        }
    if thought:
        return {
            "level": "warning",
            "text": f"已发送关闭指令且服务端接受，但模型仍返回了思考内容"
                    f"（{tokens} tokens）——该模型可能不支持关闭",
            "reasoning_tokens": tokens,
        }
    return {
        "level": "success",
        "text": "思考模式已成功关闭（本次回复无思考内容）",
        "reasoning_tokens": 0,
    }


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


# ------------------------------------------------------------------- logs


@router.get("/logs")
def list_logs():
    """Recent job logs, newest first, plus the folder holding them."""
    files = []
    for path in sorted(
        joblog.logs_dir().glob("*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        stat = path.stat()
        files.append({
            "name": path.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })
    return {"dir": str(joblog.logs_dir()), "files": files}


@router.get("/logs/job/{job_id}")
def download_job_log(job_id: str):
    path = joblog.find_log(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="该任务没有日志文件")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@router.get("/logs/file/{name}")
def download_log(name: str):
    # resolve against the listing so no path can escape the log folder
    path = joblog.logs_dir() / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="日志文件不存在")
    return FileResponse(path, media_type="text/plain", filename=path.name)


# ------------------------------------------------------------------ media


@router.get("/media/audio-tracks", response_model=list[AudioTrack])
def media_audio_tracks(path: str) -> list[AudioTrack]:
    """List the audio tracks of a video so the user can pick one."""
    p = Path(path.strip().strip('"').strip("'")).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"文件不存在: {p}")
    try:
        tracks = audio.list_tracks(p)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — unreadable/corrupt container
        raise HTTPException(status_code=400, detail=f"无法读取视频: {exc}") from exc
    return [AudioTrack(**t) for t in tracks]


# ------------------------------------------------------------ file browse


@router.get("/fs/resolve")
def fs_resolve(path: str):
    """Resolve a pasted path (Explorer address / file path, maybe quoted)."""
    raw = path.strip().strip('"').strip("'").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="路径为空")
    p = Path(raw).expanduser()
    if p.is_dir():
        return {"type": "dir", "path": str(p)}
    if p.is_file():
        return {
            "type": "file",
            "path": str(p),
            "is_video": p.suffix.lower() in VIDEO_EXTS,
        }
    return {"type": "missing", "path": raw}


@router.get("/fs/quick-access")
def fs_quick_access():
    """Shortcuts shown in the file browser: drives + common user folders."""
    items = []
    if sys.platform == "win32":
        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if os.path.exists(root):
                items.append({"name": f"{letter}:", "path": root})
    home = Path.home()
    items.append({"name": "主目录", "path": str(home)})
    for label, sub in (
        ("桌面", "Desktop"), ("下载", "Downloads"),
        ("视频", "Videos"), ("影片", "Movies"),
    ):
        d = home / sub
        if d.is_dir():
            items.append({"name": label, "path": str(d)})
    return {"items": items}


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

    if p.parent == p:  # filesystem root: C:\ on Windows, / on Linux
        # Windows: "" navigates back to the drive list; Linux: no parent
        parent = "" if sys.platform == "win32" else None
    else:
        parent = str(p.parent)
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files}
