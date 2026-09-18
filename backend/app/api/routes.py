"""REST + SSE endpoints."""

from __future__ import annotations

import ipaddress
import os
import re
import string
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from app.core import config, joblog, server
from app.core.cache import job_dir
from app.core.auth import MCP_PREFIX
from app.core.media import MEDIA_EXTS, kind_of, scan_media
from app.models.schemas import (
    AppSettings,
    AudioTrack,
    BatchRequest,
    BatchStatus,
    JobRequest,
    JobStatus,
    LLMSettings,
    QueueEntryView,
    QueueView,
    SubtitleTrack,
)
from app.services import audio, jobqueue, mcp_server, mux, series, subsource
from app.services.jobqueue import queue_manager
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
    # Snapshotted here too, not just for the queue: what the settings said
    # when the button was pressed is what this job runs with, whether it
    # starts now or waits behind something else.
    try:
        job = manager.create(req, settings=jobqueue.snapshot())
    except (FileNotFoundError, ValueError) as exc:
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
def job_result(job_id: str, part: str = "translation"):
    """这个任务的字幕。part="original" 取双文件模式的原文那一份。

    缺省值就是改动前的行为，所以既有的下载链接一个字都没变。
    """
    try:
        job = manager.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status.stage != "done" or not job.srt_path:
        raise HTTPException(status_code=409, detail="任务尚未完成")
    if part == "translation":
        path = Path(job.srt_path)
    elif part == "original":
        path = Path(job.status.original_srt_filename or "")
        if not path.name:
            raise HTTPException(status_code=404, detail="这个任务没有单独的原文字幕")
    else:
        raise HTTPException(status_code=400, detail="part 只能是 translation 或 original")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        manager.cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True}


# ------------------------------------------------------------------ batch


@router.get("/batch/scan")
def batch_scan(path: str, recursive: bool = True, skip_existing: bool = True,
               target_language: str = ""):
    """Preview which videos and audio files a batch would translate."""
    try:
        found, skipped, shadowed, with_source = scan_media(
            path, recursive, skip_existing, target_language)
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audio_files = [str(f) for f in found if kind_of(f) == "audio"]
    return {
        "videos": [str(v) for v in found],
        "skipped": [str(s) for s in skipped],
        "total": len(found),
        # a subset of "videos": the confirm dialog says these will only ever
        # produce a subtitle file, however the output switch is set
        "audio": audio_files,
        "audio_count": len(audio_files),
        "shadowed": [str(s) for s in shadowed],
        # files that already carry a subtitle in some OTHER language. Not
        # acted on — reported, so the user can choose to translate from
        # that text instead of from speech, which is faster and more
        # accurate. Which source to use is their call, not ours.
        "with_source": [str(s) for s in with_source],
        "with_source_count": len(with_source),
    }


@router.post("/batch", response_model=BatchStatus)
def create_batch(req: BatchRequest) -> BatchStatus:
    try:
        # One snapshot for the whole batch: editing settings while a season
        # is half done used to change the parameters from that episode on.
        return batch_manager.create(req, settings=jobqueue.snapshot())
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


@router.post("/batch/{batch_id}/glossary/save")
def save_batch_glossary(batch_id: str) -> dict:
    """Keep a series-mode glossary for good, in the settings' own table.

    The table itself only lives as long as the batch. Merging here rather
    than in the browser keeps the dedup rule in one place and avoids
    round-tripping the whole settings object — which, read from another
    machine, has its API key masked.
    """
    try:
        batch = batch_manager.status(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")
    if not batch.glossary.strip():
        raise HTTPException(status_code=400, detail="该批量没有可保存的译名对照表")

    settings = config.load_settings().model_copy(deep=True)
    existing = settings.prompts.glossary
    kept = {source for source, _ in series.parse_terms(existing)}
    fresh = [
        (source, target)
        for source, target in series.parse_terms(batch.glossary)
        if source not in kept
    ]
    if fresh:
        block = series.render_terms(fresh)
        settings.prompts.glossary = (
            f"{existing.rstrip()}\n{block}" if existing.strip() else block
        )
        config.save_settings(settings)
    return {"added": len(fresh), "total": len(kept) + len(fresh)}


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
    if remote.llm.vision_api_key:
        remote.llm.vision_api_key = MASKED
    if remote.llm.audio_api_key:
        remote.llm.audio_api_key = MASKED
    return remote


@router.put("/settings", response_model=AppSettings)
def put_settings(settings: AppSettings, request: Request) -> AppSettings:
    stored = config.load_settings()
    if settings.llm.api_key == MASKED:
        settings.llm.api_key = stored.llm.api_key
    if settings.llm.vision_api_key == MASKED:
        settings.llm.vision_api_key = stored.llm.vision_api_key
    if settings.llm.audio_api_key == MASKED:
        settings.llm.audio_api_key = stored.llm.audio_api_key
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
        reply_text,
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
        reply = reply_text(resp)  # raises with the server's own words
        message = resp.choices[0].message
        return {
            "ok": True,
            "reply": reply,
            "thinking": _describe_thinking(llm, resp, message, accepted),
        }
    except Exception as exc:  # noqa: BLE001 — report connectivity errors verbatim
        return {"ok": False, "error": str(exc)}


# what the test picture says; short, unambiguous, and not a word a model
# could produce by guessing what a test image probably contains
VISION_PROBE = "MOVIE 42"


@router.post("/settings/test-vision")
def test_vision(llm: LLMSettings):
    """Prove the vision endpoint works — by sending it an actual picture.

    A text ping proves nothing here: a text-only model answers it perfectly
    and then fails on the first subtitle. So the request is built exactly
    the way ocr.py builds its own — one PNG data URL, one instruction — and
    the answer is checked against what the picture says.
    """
    from PIL import Image, ImageDraw

    from app.services.ocr import _label_font, _png_data_url
    from app.services.translator import make_vision_client, reply_text

    model = llm.vision_model.strip() or llm.model
    endpoint = llm.vision_base_url.strip() or llm.base_url
    image = Image.new("L", (440, 90), 255)
    ImageDraw.Draw(image).text((20, 28), VISION_PROBE, fill=0, font=_label_font())
    try:
        client = make_vision_client(llm, config.load_settings().network)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "只输出这张图片里的文字，不要任何解释。"},
                {"type": "image_url", "image_url": {"url": _png_data_url(image)}},
            ]}],
            temperature=0,
        )
        reply = reply_text(resp)
    except Exception as exc:  # noqa: BLE001 — report connectivity errors verbatim
        return {"ok": False, "error": str(exc), "model": model, "endpoint": endpoint}
    return {
        "ok": True,
        "reply": reply,
        "model": model,
        "endpoint": endpoint,
        "read_it": _squash(VISION_PROBE) in _squash(reply),
    }


# What the test clip does. Measured on a real endpoint: this model cannot
# count beeps at all — it answered 6 for one beep, 3 for three, and 5 for
# two seconds of pure silence — but it identifies a rising or falling sweep
# every time. So the probe asks the question the model can answer, and the
# direction is drawn at random so a guess cannot keep passing.
PROBE_SECONDS = 4.0
PROBE_QUESTION = ("这段音频是一个单一的纯音。它的音调是一直升高还是一直降低？"
                  "只回答「升高」或「降低」两个词之一，不要任何解释。")


@router.post("/settings/test-asr-api")
def test_asr_api(llm: LLMSettings):
    """Prove the audio endpoint works — by sending it actual audio.

    Two different things can be wrong and they need different answers. The
    model may not listen at all (a text-only model answers this question
    happily and then fails on the first window), or the relay in between
    may drop the audio part and pass the text through — measured on a real
    endpoint, and invisible in the reply, because the model answers
    plausibly either way.

    So two independent signals: the pitch question, which needs ears, and
    the number of prompt tokens the server reports, which is arithmetic —
    audio costs about 25 tokens a second and text alone cannot fake that.

    The clip is built with the engine's own encoder, the same way
    test-vision draws its picture with ocr.py's own helpers.
    """
    import random

    import numpy as np

    from app.services import asr_api
    from app.services.translator import make_audio_client, reply_text

    model = llm.audio_model.strip() or llm.model
    endpoint = llm.audio_base_url.strip() or llm.base_url

    rate = asr_api.SAMPLE_RATE
    rising = random.choice((True, False))
    steps = np.arange(int(rate * PROBE_SECONDS))
    sweep = np.linspace(400.0, 1600.0, len(steps))
    clip = (0.5 * np.sin(2 * np.pi * np.cumsum(
        sweep if rising else sweep[::-1]) / rate)).astype("float32")

    try:
        client = make_audio_client(llm, config.load_settings().network)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROBE_QUESTION},
                asr_api.audio_part(asr_api.encode_samples(clip, "mp3"), "mp3"),
            ]}],
            temperature=0,
        )
        reply = reply_text(resp)
    except Exception as exc:  # noqa: BLE001 — report connectivity errors verbatim
        return {"ok": False, "error": str(exc), "model": model, "endpoint": endpoint}

    said, other = ("升高", "降低") if rising else ("降低", "升高")
    usage = getattr(resp, "usage", None)
    return {
        "ok": True,
        "reply": reply,
        "model": model,
        "endpoint": endpoint,
        "asked": said,
        "heard_it": said in reply and other not in reply,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "expected_tokens": int(asr_api.AUDIO_TOKENS_PER_SECOND * PROBE_SECONDS),
        # None when the server reports no usage at all
        "carried_audio": asr_api.audio_reached_the_model(usage, PROBE_SECONDS),
    }


def _squash(text: str) -> str:
    return "".join(text.split()).lower()


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


@router.get("/storage/usage")
def storage_usage() -> dict:
    """What this program is keeping on disk, so it can be seen and dropped.

    Half-finished work is the only part that grows with use: a two-hour
    film's audio is about 230 MB, and it is kept so an interrupted run does
    not have to buy its audio and transcription again. Both limits are in
    settings, and either at 0 turns the whole thing off.
    """
    from app.core.cache import checkpoints_usage

    settings = config.load_settings()
    kept = checkpoints_usage()
    return {
        "checkpoints": kept,
        "days": settings.checkpoint_days,
        "max_gb": settings.checkpoint_max_gb,
        "enabled": settings.checkpoint_days > 0 and settings.checkpoint_max_gb > 0,
        "logs_dir": str(joblog.logs_dir()),
    }


@router.post("/storage/clear-checkpoints")
def clear_checkpoints() -> dict:
    """Drop every half-finished run. Finished subtitles are untouched."""
    import shutil

    from app.core.cache import checkpoints_root, checkpoints_usage

    before = checkpoints_usage()["bytes"]
    root = checkpoints_root()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return {"freed": before}


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


@router.get("/media/encoders")
def media_encoders() -> dict:
    """Containers, and the video encoders this machine can actually use.

    The dropdown is built from this rather than a fixed list: h264_nvenc is
    registered on a machine with no NVIDIA card at all and only fails when
    the encoder is opened, so offering it everywhere would mean offering an
    option that is certain to fail on most machines.
    """
    return {
        "containers": [
            {"value": "mkv", "label": "MKV（推荐）", "styled_subtitle": True},
            {"value": "mp4", "label": "MP4（兼容性最好）", "styled_subtitle": False},
        ],
        "video": [{"id": mux.COPY, "label": "保持原编码（不重编码）",
                   "family": "", "hardware": False}] + mux.available_encoders(),
        "audio": [
            {"id": "copy", "label": "保持原编码（不重编码）"},
            {"id": "aac", "label": "AAC（兼容性最好）"},
            {"id": "ac3", "label": "AC-3"},
            {"id": "eac3", "label": "E-AC-3"},
            {"id": "flac", "label": "FLAC（无损）"},
            {"id": "libopus", "label": "Opus"},
        ],
        "presets": ["ultrafast", "fast", "medium", "slow", "veryslow"],
    }


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
        raise HTTPException(status_code=400, detail=f"无法读取媒体文件: {exc}") from exc
    return [AudioTrack(**t) for t in tracks]


@router.get("/media/subtitle-tracks", response_model=list[SubtitleTrack])
def media_subtitle_tracks(path: str) -> list[SubtitleTrack]:
    """List the subtitles this video already carries, plus any beside it.

    An empty list is a normal answer — most files have none — so unlike the
    audio probe this never turns "nothing found" into an error.
    """
    p = Path(path.strip().strip('"').strip("'")).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"文件不存在: {p}")
    try:
        tracks = subsource.all_tracks(p)
    except Exception as exc:  # noqa: BLE001 — unreadable/corrupt container
        raise HTTPException(status_code=400, detail=f"无法读取媒体文件: {exc}") from exc
    return [SubtitleTrack(**t) for t in tracks]


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
            "is_media": p.suffix.lower() in MEDIA_EXTS,
            "is_audio": kind_of(p) == "audio",
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
    """List directories, videos and audio files for the file picker."""
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
                elif entry.suffix.lower() in MEDIA_EXTS:
                    files.append({
                        "name": entry.name,
                        "size": entry.stat().st_size,
                        # the picker draws its icon from this rather than
                        # keeping a second copy of the extension table
                        "kind": kind_of(entry),
                    })
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


# ------------------------------------------------------------------ queue


def _entry_view(entry, current_hash: str) -> QueueEntryView:
    """One entry as the UI sees it: no snapshot, plus the live stage.

    The snapshot runs to a couple of kilobytes and this list is polled, so
    only its fingerprint travels here; the text lives behind
    /api/queue/{id}/settings.

    A terminal entry is rendered entirely from what it recorded when it
    finished — never from manager.jobs, which a restart empties. That rule
    is what lets a finished entry still say what it produced tomorrow.
    """
    stage, progress, message, live = entry.status, 0.0, "", False
    if entry.status == "running" and entry.job_id:
        try:
            status = manager.get(entry.job_id).status
            stage, progress, message, live = (
                status.stage, status.progress, status.message, True)
        except KeyError:
            pass
    fingerprint = jobqueue.settings_hash(entry.settings)
    return QueueEntryView(
        id=entry.id, kind=entry.kind, status=entry.status, title=entry.title,
        summary=jobqueue.describe(entry.request),
        created_at=entry.created_at, started_at=entry.started_at,
        finished_at=entry.finished_at, job_id=entry.job_id,
        error=entry.error, note=entry.note,
        settings_hash=fingerprint,
        settings_differs=bool(fingerprint) and fingerprint != current_hash,
        interrupted=entry.interrupted,
        stage=stage, progress=progress, message=message, job_live=live,
        has_log=bool(entry.job_id) and joblog.find_log(entry.job_id) is not None,
        result_srt=entry.result_srt, result_video=entry.result_video,
        result_srt_original=entry.result_srt_original,
        result_in_place=entry.result_in_place,
        group_id=entry.group_id, group_title=entry.group_title,
    )


def _queue_view() -> QueueView:
    store = queue_manager.store
    current = jobqueue.settings_hash(jobqueue.snapshot())
    with store.lock:
        entries = [_entry_view(e, current) for e in store.entries]
        active = next((e.id for e in store.entries if e.status == "running"), "")
        return QueueView(paused=store.paused, worker_alive=queue_manager.alive,
                         active_id=active, settings_hash=current, entries=entries)


@router.get("/queue", response_model=QueueView)
def get_queue() -> QueueView:
    return _queue_view()


@router.post("/queue/jobs")
def enqueue_job(req: JobRequest) -> dict:
    """Add one film, frozen under the settings as they stand right now.

    The body is a JobRequest and nothing else — there is deliberately no
    field through which a client could hand us the settings to freeze. A
    LAN browser's copy has its API keys masked to ********, and freezing
    those would produce a job that cannot authenticate.
    """
    video = Path(req.video_path)
    if not video.is_file():
        raise HTTPException(status_code=400, detail=f"片源文件不存在: {video}")
    try:
        entry = queue_manager.store.add(req, jobqueue.snapshot())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    queue_manager.nudge()
    waiting = [e for e in queue_manager.store.entries if e.status == "queued"]
    position = next((i + 1 for i, e in enumerate(waiting) if e.id == entry.id), 0)
    return {"entry": _entry_view(entry, jobqueue.settings_hash(entry.settings)),
            "position": position}


@router.post("/queue/batch")
def enqueue_batch(req: BatchRequest) -> dict:
    """Add a directory, expanded into one entry per file straight away.

    Expanded now rather than when its turn comes, so the list shows exactly
    what will run and individual files can be reordered or dropped. All of
    them share one snapshot — a season translated under two different sets
    of settings is the problem this feature exists to prevent.
    """
    try:
        videos, skipped, _shadowed, _with_source = scan_media(
            req.directory, req.recursive, req.skip_existing_srt,
            req.target_language)
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not videos:
        raise HTTPException(status_code=400, detail="目录中没有需要翻译的视频或音频文件")

    settings = jobqueue.snapshot()
    group_id = uuid.uuid4().hex[:12]
    series_id = group_id if req.series_mode else ""
    added = []
    try:
        for video in videos:
            added.append(queue_manager.store.add(
                JobRequest(
                    video_path=str(video),
                    audio_language=req.audio_language,
                    text_source=req.text_source,
                    subtitle_language=req.subtitle_language,
                    # a season where one episode ships without a subtitle
                    # should translate that episode, not stop the batch
                    subtitle_fallback_asr=True,
                    source_language=req.source_language,
                    target_language=req.target_language,
                    synopsis=req.synopsis,
                    output_mode=req.output_mode,
                    embed_subtitle=req.embed_subtitle,
                    embed=req.embed,
                    series_id=series_id,
                ),
                settings, group_id=group_id, group_title=req.directory))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    queue_manager.nudge()
    current = jobqueue.settings_hash(settings)
    return {"entries": [_entry_view(e, current) for e in added],
            "count": len(added), "skipped": [str(p) for p in skipped]}


@router.post("/queue/pause")
def pause_queue(body: dict) -> dict:
    queue_manager.store.set_paused(bool(body.get("paused", True)))
    if not queue_manager.store.paused:
        queue_manager.nudge()
    return {"paused": queue_manager.store.paused}


@router.put("/queue/order", response_model=QueueView)
def reorder_queue(body: dict) -> QueueView:
    try:
        queue_manager.store.reorder([str(i) for i in body.get("ids", [])])
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _queue_view()


# Registered before /queue/{entry_id}: routes match in registration order,
# so the other way round "finished" would be read as an entry id.
@router.delete("/queue/finished")
def clear_finished() -> dict:
    return {"removed": queue_manager.store.clear_finished()}


@router.get("/queue/{entry_id}/settings")
def queue_entry_settings(entry_id: str) -> dict:
    """The settings this entry froze, rendered the way the job log renders them.

    Reuses joblog._settings_lines so that what you approve before the run
    and what the downloadable log says during it are the same text — and
    that rendering never prints a key.
    """
    entry = queue_manager.store.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="列队里没有这条任务")
    current = config.load_settings()
    if entry.settings is None:
        return {"lines": joblog._settings_lines(current), "differs": [],
                "same_as_current": True, "unreadable": True}
    lines = joblog._settings_lines(entry.settings)
    live = joblog._settings_lines(current)
    differs = [line for line in lines if line not in live]
    return {"lines": lines, "differs": differs,
            "same_as_current": not differs, "unreadable": False}


@router.post("/queue/{entry_id}/cancel")
def cancel_queue_entry(entry_id: str) -> dict:
    try:
        return {"status": queue_manager.cancel(entry_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="列队里没有这条任务")


@router.post("/queue/{entry_id}/retry")
def retry_queue_entry(entry_id: str, body: dict | None = None) -> dict:
    fresh = bool((body or {}).get("fresh", False))
    try:
        entry = queue_manager.retry(entry_id, fresh=fresh)
    except KeyError:
        raise HTTPException(status_code=404, detail="列队里没有这条任务")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"entry": _entry_view(entry, jobqueue.settings_hash(entry.settings)),
            "used_current_settings": fresh}


@router.get("/queue/{entry_id}/result")
def queue_entry_result(entry_id: str, part: str = "translation"):
    """The subtitle this entry produced.

    Served from the path the entry recorded rather than through the job,
    so it still works after the restart that emptied JobManager.jobs.

    part="original" 取双文件模式的原文那一份，它**永远**是字幕文件：内嵌模式
    下另一个按钮给的是视频，而原文那一份仍然在工作目录里躺着。
    """
    entry = queue_manager.store.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="列队里没有这条任务")
    if part == "original":
        original = Path(entry.result_srt_original or "")
        if not original.name:
            raise HTTPException(status_code=404, detail="这条任务没有单独的原文字幕")
        if not original.is_file():
            raise HTTPException(status_code=404, detail=f"文件已不在原位置：{original}")
        return FileResponse(str(original), filename=original.name)
    if part != "translation":
        raise HTTPException(status_code=400, detail="part 只能是 translation 或 original")
    name = entry.result_video or entry.result_srt
    if not name:
        raise HTTPException(status_code=404, detail="这条任务还没有产物")
    if entry.result_in_place or entry.result_video:
        path = Path(entry.title).parent / name
    else:
        path = Path(job_dir(entry.job_id)) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"文件已不在原位置：{path}")
    return FileResponse(str(path), filename=path.name)


@router.delete("/queue/{entry_id}")
def remove_queue_entry(entry_id: str) -> dict:
    try:
        if not queue_manager.store.remove(entry_id):
            raise HTTPException(status_code=404, detail="列队里没有这条任务")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}
