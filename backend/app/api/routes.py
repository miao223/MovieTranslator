"""REST + SSE endpoints."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import string
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError

from app.core import config, joblog, server
from app.core.cache import job_dir
from app.core.auth import MCP_PREFIX
from app.core.media import SOURCE_EXTS, disc_root_of, kind_of, scan_media
from app.models.schemas import (
    AppSettings,
    AudioTrack,
    BatchRequest,
    BatchStatus,
    DiscAnswer,
    DiscBatchReport,
    DiscBatchRequest,
    DiscOutputMode,
    DiscReport,
    DiscEnqueueRequest,
    DiscRequest,
    DiscSkipped,
    DiscSubtitles,
    EncodeBatchRequest,
    EncodeEnqueueRequest,
    EncodeOptions,
    EncodeRequest,
    EncodeScanRequest,
    JobRequest,
    JobStatus,
    LLMSettings,
    QueueEntryView,
    QueueView,
    SubtitleTrack,
)
from app.services import (
    audio, cpuyield, encode, jobqueue, mcp_server, memguard, mux, series, subsource,
)
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
               target_language: str = "", subtitle_language: str = ""):
    """Preview which videos and audio files a batch would translate."""
    try:
        scan = scan_media(path, recursive, skip_existing, target_language,
                          subtitle_language)
        found, skipped = scan.to_translate, scan.skipped
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
        "shadowed": [str(s) for s in scan.shadowed],
        # 一份现成的字幕顶替掉的视频/音频：更快，但这一批不会产出内嵌视频、
        # 也不会用语音识别。与 shadowed 分开，因为要说的话完全不同。
        "replaced": [str(s) for s in scan.replaced],
        "replaced_count": len(scan.replaced),
        "subtitle_count": sum(1 for f in found if kind_of(f) == "subtitle"),
        # files that already carry a subtitle in some OTHER language. Not
        # acted on — reported, so the user can choose to translate from
        # that text instead of from speech, which is faster and more
        # accurate. Which source to use is their call, not ours.
        "with_source": [str(s) for s in scan.with_source],
        "with_source_count": len(scan.with_source),
        # discs are never translated file by file: their stream folders hold
        # the film, pieces of the film, menus and trailers side by side
        "discs": [str(d) for d in scan.discs],
        "disc_count": len(scan.discs),
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
        # for the 压制 form (encode.capabilities): 10-bit, tunings, and
        # each encoder's own quality scale. Only added keys — the 翻译任务
        # page reads the ones above and nothing else.
        "capabilities": encode.capabilities(),
        "deinterlace": "bwdif" in av_filters(),
    }


def av_filters() -> set:
    import av

    return set(getattr(av.filter, "filters_available", ()) or ())


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


# ------------------------------------------------------------------- disc


def _disc_answer(answer: Optional[DiscAnswer]):
    """The batch mode's per-disc answer as the planner takes it."""
    from app.services.disc.plan import Answer

    if answer is None:
        return Answer()
    return Answer(answer.series, answer.name, answer.episode_start)


@router.get("/disc/scan", response_model=DiscReport)
def disc_scan(path: str, series: Optional[bool] = None,
              episode_start: Optional[int] = None, name: str = "",
              output_mode: Optional[DiscOutputMode] = None,
              output_dir: str = "") -> DiscReport:
    """The 原盘 page's analysis of one disc. Reads only the disc's metadata
    (playlists, clip info, IFOs) — and, for a volume of a box set, its
    sibling volumes', whose episodes it numbers on from — so it answers in
    milliseconds and is simply asked again whenever a switch changes."""
    from app.services.disc import plan as disc_plan

    settings = config.load_settings()
    mode, custom = disc_plan.output_choice(output_mode, output_dir, settings)
    item = disc_plan.plan(
        [(path, disc_plan.Answer(series, name, episode_start))], settings)[0]
    if item.disc is None:
        raise HTTPException(status_code=400, detail=item.error)
    return disc_plan.report_of(item, settings, mode, custom)


@router.post("/disc/batch-scan", response_model=DiscBatchReport)
def disc_batch_scan(req: DiscBatchRequest) -> DiscBatchReport:
    """The batch mode: every disc in a folder, analysed as the single scan
    would analyse it. A disc that cannot be read is listed with why, not
    dropped — the user is looking at that folder and knows it is there."""
    from app.services.disc import plan as disc_plan
    from app.services.disc.binary import DiscError
    from app.services.disc.fs import find_discs

    settings = config.load_settings()
    mode, custom = disc_plan.output_choice(req.output_mode, req.output_dir, settings)
    try:
        found = find_discs(req.path, req.recursive)
    except (DiscError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not found:
        raise HTTPException(status_code=400,
                            detail="这个文件夹里没有找到原盘（BDMV / VIDEO_TS 文件夹或 .iso 镜像）")
    answers = {str(Path(a.path)): a for a in req.discs}
    planned = disc_plan.plan(
        [(str(p), _disc_answer(answers.get(str(p)))) for p in found], settings)
    report = DiscBatchReport(path=req.path, recursive=req.recursive,
                             output_mode=mode, output_dir=custom)
    for item in planned:
        if item.disc is None:
            report.skipped.append(DiscSkipped(path=item.path, reason=item.error))
        else:
            report.discs.append(disc_plan.report_of(item, settings, mode, custom))
    return report


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
            "is_media": p.suffix.lower() in SOURCE_EXTS,
            "is_audio": kind_of(p) == "audio",
            # 字幕文件也能当翻译对象。前端据此隐藏音轨/原文来源、置灰内嵌，
            # 页面不自己抄一份后缀表——它只问这里。
            "is_subtitle": kind_of(p) == "subtitle",
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
def fs_browse(path: str = "", mode: str = ""):
    """List directories, videos and audio files for the file picker.

    mode="disc" is the 原盘 page's picker: it lists .iso images instead of
    media files and names which folders are discs (disc_dirs), so they can
    be picked rather than opened. The default listing is untouched.
    """
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

    disc_mode = mode == "disc"
    # the 压制 page picks videos only: audio and subtitle files have nothing
    # to re-encode
    video_mode = mode == "video"
    dirs, files, disc_dirs = [], [], []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append(entry.name)
                    if disc_mode and _is_disc_dir(entry):
                        disc_dirs.append(entry.name)
                elif disc_mode:
                    if entry.suffix.lower() == ".iso":
                        files.append({"name": entry.name, "size": entry.stat().st_size,
                                      "kind": "disc"})
                elif entry.suffix.lower() in SOURCE_EXTS and (
                        not video_mode or kind_of(entry) == "video"):
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
    listing = {"path": str(p), "parent": parent, "dirs": dirs, "files": files}
    if disc_mode:
        listing["disc_dirs"] = disc_dirs
        listing["is_disc"] = _is_disc_dir(p)
    return listing


def _is_disc_dir(folder: Path) -> bool:
    """A folder the 原盘 page can take as it is."""
    from app.services.disc.fs import _bd_root, _child, _dvd_dir

    if _bd_root(folder) is not None or _dvd_dir(folder) is not None:
        return True
    return folder.name.lower() == "bdmv" and _child(folder, "PLAYLIST") is not None


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
        summary=jobqueue.describe_entry(entry),
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
        result_files=list(entry.result_files),
        group_id=entry.group_id, group_title=entry.group_title,
    )


def _queue_view() -> QueueView:
    store = queue_manager.store
    current = jobqueue.settings_hash(jobqueue.snapshot())
    with store.lock:
        entries = [_entry_view(e, current) for e in store.entries]
        active = next((e.id for e in store.entries if e.status == "running"), "")
        return QueueView(paused=store.paused, cpu_yield=store.cpu_yield,
                         cpu_status=cpuyield.status(), memory_limit_gb=store.memory_limit_gb,
                         memory_status=memguard.status(), worker_alive=queue_manager.alive,
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
        scan = scan_media(
            req.directory, req.recursive, req.skip_existing_srt,
            req.target_language, req.subtitle_language)
        videos, skipped = scan.to_translate, scan.skipped
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


def _subtitle_template(subtitles: DiscSubtitles, item) -> JobRequest:
    """The translation request every MKV of this disc will be queued with
    (QueueEntry.then), video_path filled in per file when the remux ends.

    Built like a translation batch's entries (enqueue_batch): a file whose
    subtitle cannot be read falls back to speech recognition rather than
    failing. The series table is the disc's own — or the box set's, shared
    by its volumes — never the batch's: one batch of unrelated films would
    otherwise share one table of names, the first film's winning."""
    series_id = ""
    if subtitles.series_mode:
        key = item.volume.id if item.chained and item.volume is not None \
            else str(Path(item.disc.root))
        series_id = "disc-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return JobRequest(
        video_path="",
        text_source=subtitles.text_source,
        audio_language=subtitles.audio_language,
        subtitle_language=subtitles.subtitle_language,
        subtitle_fallback_asr=True,
        source_language=subtitles.source_language,
        target_language=subtitles.target_language.strip(),
        output_mode=subtitles.output_mode,
        embed_subtitle=subtitles.embed_subtitle,
        series_id=series_id,
    )


def _freeze_discs(items, output_mode, output_dir, settings, named: bool,
                  subtitles: Optional[DiscSubtitles] = None) -> list:
    """The DiscRequests to queue for *items* — (path, Answer, titles) — with
    everything the page showed resolved into them: the ticks (an empty
    selection becomes the analysis's defaults), the name, the episode and
    extra numbers, the output folder. The run analyses the disc again with
    exactly these, so it writes what the list showed even if the volume
    next to it is gone by then. Each comes paired with the translation to
    queue for its MKVs afterwards (None without *subtitles*).

    Everything that can be refused is refused now, not in an hour when the
    entry's turn comes; one disc refused refuses the whole call, so a batch
    is never half queued."""
    from app.services.disc import plan as disc_plan

    mode, custom = disc_plan.output_choice(output_mode, output_dir, settings)
    if mode == "custom" and not custom:
        raise HTTPException(status_code=400, detail="选了「指定的文件夹」，但还没有选是哪个文件夹")
    if subtitles is not None and not subtitles.target_language.strip():
        raise HTTPException(status_code=400, detail="做字幕的目标语言是空的")
    planned = disc_plan.plan([(path, answer) for path, answer, _ in items], settings)
    frozen, problems = [], []
    for item, (_, answer, titles) in zip(planned, items):
        who = f"{Path(item.path).name}：" if named else ""
        if item.disc is None:
            problems.append(who + item.error)
            continue
        report = disc_plan.report_of(item, settings, mode, custom)
        if report.encrypted:
            problems.append(who + "这张盘仍是加密状态，本程序不做解密，无法封装")
            continue
        if report.analysis_only:
            problems.append(who + "这是只有元数据的副本（没有视频文件），只能分析，不能封装")
            continue
        known = {t.id: t for t in report.titles}
        chosen = list(dict.fromkeys(titles)) or [t.id for t in report.titles if t.selected]
        unknown = [i for i in chosen if i not in known]
        if unknown:
            problems.append(who + f"光盘里没有这些标题：{'、'.join(unknown)}，请重新分析")
            continue
        refused = [i for i in chosen if not known[i].selectable]
        if refused:
            problems.append(who + "这些标题无法封装：" + "；".join(
                f"{i}（{known[i].reason}）" for i in refused))
            continue
        if not chosen:
            problems.append(who + "没有勾选任何标题")
            continue
        frozen.append((DiscRequest(
            path=report.path, titles=chosen, name=item.name, output_mode=mode,
            output_dir=custom, episode_start=item.episode_start,
            extra_start=item.extra_start, own_name=item.own_name,
            series=answer.series),
            _subtitle_template(subtitles, item) if subtitles is not None else None))
    if problems:
        raise HTTPException(status_code=400, detail="；".join(problems))
    return frozen


def _queue_room(n: int) -> None:
    waiting = sum(1 for e in queue_manager.store.entries if e.status == "queued")
    if waiting + n > jobqueue.MAX_QUEUED:
        raise HTTPException(status_code=400,
                            detail=f"列队已满（最多 {jobqueue.MAX_QUEUED} 条等待中的任务）")


@router.post("/queue/disc")
def enqueue_disc(req: DiscEnqueueRequest) -> dict:
    """Add one disc's remux to the queue — 加入列队, or with `subtitles`
    加入列队并做字幕: its MKVs then go into the queue for translation as
    soon as they are written (jobqueue.QueueManager._queue_subtitles)."""
    from app.services.disc.plan import Answer

    settings = jobqueue.snapshot()
    then_encode = (_encode_template(req.encode, req.subtitles)
                   if req.encode is not None else None)
    frozen, then = _freeze_discs(
        [(req.path, Answer(req.series, req.name, req.episode_start), req.titles)],
        req.output_mode, req.output_dir, settings, named=False,
        subtitles=req.subtitles)[0]
    try:
        entry = queue_manager.store.add_disc(frozen, settings, then=then,
                                             then_encode=then_encode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    queue_manager.nudge()
    waiting = [e for e in queue_manager.store.entries if e.status == "queued"]
    position = next((i + 1 for i, e in enumerate(waiting) if e.id == entry.id), 0)
    return {"entry": _entry_view(entry, jobqueue.settings_hash(settings)),
            "position": position}


@router.post("/queue/disc-batch")
def enqueue_disc_batch(req: DiscBatchRequest) -> dict:
    """The batch mode's 加入列队: one entry per disc, grouped, sharing one
    settings snapshot — like a translation batch, so one disc failing costs
    no other and each can be retried or dropped on its own."""
    wanted = [a for a in req.discs if a.include]
    if not wanted:
        raise HTTPException(status_code=400, detail="没有勾选任何光盘")
    settings = jobqueue.snapshot()
    then_encode = (_encode_template(req.encode, req.subtitles)
                   if req.encode is not None else None)
    frozen = _freeze_discs(
        [(a.path, _disc_answer(a), a.titles) for a in wanted],
        req.output_mode, req.output_dir, settings, named=True,
        subtitles=req.subtitles)
    _queue_room(len(frozen))
    group_id = uuid.uuid4().hex[:12]
    added = []
    try:
        for request, then in frozen:
            added.append(queue_manager.store.add_disc(
                request, settings, group_id=group_id, group_title=req.path, then=then,
                then_encode=then_encode))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    queue_manager.nudge()
    current = jobqueue.settings_hash(settings)
    waiting = [e for e in queue_manager.store.entries if e.status == "queued"]
    position = next((i + 1 for i, e in enumerate(waiting) if e.id == added[0].id), 0)
    return {"entries": [_entry_view(e, current) for e in added],
            "count": len(added), "position": position}


# ------------------------------------------------------------------ encode


# a batch this large is almost certainly the wrong folder (a whole library),
# and every file is opened to be listed
MAX_ENCODE_FILES = 500


def _encode_template(choice, subtitles: Optional[DiscSubtitles]) -> EncodeRequest:
    """The encode every MKV of a disc gets once remuxed (QueueEntry.then_encode),
    source filled in per file. It replaces the lossless MKV unless the
    dialog said to keep it."""
    if choice.options.container != "mkv":
        raise HTTPException(status_code=400, detail=(
            "全流程只能压成 MKV：MP4 装不下光盘的图形字幕，也就换不下那份无损 MKV"))
    _check_encode(choice.options, subtitles)
    return EncodeRequest(source="", options=choice.options, output_mode="beside",
                         replace_source=not choice.keep_lossless)


def _check_encode(options: EncodeOptions, subtitles: Optional[DiscSubtitles]) -> None:
    """Refuse now what would fail in an hour: an encoder this machine lacks,
    and a subtitle step whose track the encode throws away."""
    try:
        encode.check_options(options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if subtitles is None:
        return
    if not subtitles.target_language.strip():
        raise HTTPException(status_code=400, detail="做字幕的目标语言是空的")
    if subtitles.text_source == "subtitle":
        if options.subtitles == "none" or options.container == "mp4":
            raise HTTPException(status_code=400, detail=(
                "做字幕要读片源里的字幕轨，可压制的设置不保留字幕轨"
                + ("（MP4 装不下字幕轨）" if options.container == "mp4" else "")))
        wanted = subtitles.subtitle_language
        kept = [audio.canon_language(x) for x in options.subtitle_languages]
        if options.subtitles == "languages" and wanted \
                and audio.canon_language(wanted) not in kept:
            raise HTTPException(status_code=400, detail=(
                f"做字幕要读 {audio.language_name(wanted)} 字幕轨，"
                f"可压制只保留 {'、'.join(options.subtitle_languages) or '（无）'} 的字幕轨"))
    wanted = subtitles.audio_language
    kept = [audio.canon_language(x) for x in options.audio_languages]
    if kept and wanted and audio.canon_language(wanted) not in kept:
        raise HTTPException(status_code=400, detail=(
            f"做字幕要用 {audio.language_name(wanted)} 音轨，"
            f"可压制只保留 {'、'.join(options.audio_languages)} 的音轨"))


def _subtitles_after_encode(subtitles: DiscSubtitles, series_id: str) -> JobRequest:
    """The translation an encode queues for its output (QueueEntry.then) —
    built like the disc's (_subtitle_template), with the batch's own table
    when the page asked for one."""
    return JobRequest(
        video_path="",
        text_source=subtitles.text_source,
        audio_language=subtitles.audio_language,
        subtitle_language=subtitles.subtitle_language,
        subtitle_fallback_asr=True,
        source_language=subtitles.source_language,
        target_language=subtitles.target_language.strip(),
        output_mode=subtitles.output_mode,
        embed_subtitle=subtitles.embed_subtitle,
        series_id=series_id,
    )


def _clean_path(raw: str) -> Path:
    return Path(raw.strip().strip('"').strip("'")).expanduser()


def _custom_dir(output_mode: str, output_dir: str) -> Optional[Path]:
    if output_mode != "custom":
        return None
    if not output_dir.strip():
        raise HTTPException(status_code=400, detail="选了「指定的文件夹」，但还没有选是哪个文件夹")
    return _clean_path(output_dir)


def _same_folder(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _video_files(folder: Path, recursive: bool) -> tuple[list, list]:
    """(videos, disc roots) under *folder*: never inside a disc, whose
    m2ts / VOB files are pieces, not films (the 原盘 page is for those)."""
    videos, discs = [], set()
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if not recursive:
            dirs[:] = []
        for name in sorted(names):
            path = Path(root) / name
            if name.startswith(".") or name.endswith(".part"):
                continue
            disc = disc_root_of(path)
            if disc is not None:
                discs.add(str(disc))
                continue
            if kind_of(path) == "video":
                videos.append(path)
    return videos, sorted(discs)


@router.get("/encode/probe")
def encode_probe(path: str) -> dict:
    """One file for the 压制 page: its streams, plus what only decoded
    frames tell — interlacing, HDR metadata."""
    source = _clean_path(path)
    if not source.is_file():
        raise HTTPException(status_code=400, detail=f"文件不存在：{source}")
    try:
        info = encode.probe(source, sample_frames=60)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"读不出这个文件：{exc}") from exc
    if info["video"] is None:
        raise HTTPException(status_code=400, detail="这个文件里没有画面，没有可压制的视频")
    return info


@router.post("/encode/pick")
def encode_pick(body: dict) -> dict:
    """按画面自动选编码, previewed: the 压制 page asks as soon as a file is
    chosen, so the choice is in the form before it is queued and can still
    be changed. The same judgement an auto encode makes when it starts
    (encodepick), on the settings as they stand; *options* is the form, of
    which only the picture's fields are replaced.
    """
    import base64

    from app.services import encodepick

    path = Path(str(body.get("path", "")).strip())
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"找不到这个文件：{path}")
    try:
        options = EncodeOptions(**(body.get("options") or {}))
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        verdict, image = encodepick.analyze(path, config.load_settings())
        picked = encodepick.apply(verdict, options)
        encode.check_options(picked)
    except Exception as exc:  # noqa: BLE001 — the page shows it and keeps its own
        raise HTTPException(status_code=400, detail=f"视觉模型没判断出来：{exc}") from exc
    return {
        "content": verdict.content, "grain": verdict.grain, "reason": verdict.reason,
        "summary": verdict.describe(), "options": picked.model_dump(),
        "described": encode.describe_options(picked),
        "image": "data:image/jpeg;base64," + base64.b64encode(image).decode(),
    }


@router.post("/encode/scan")
def encode_scan(req: EncodeScanRequest) -> dict:
    """The 压制 page's batch mode: every video in a folder, headers only."""
    folder = _clean_path(req.path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"不是有效目录：{folder}")
    found, discs = _video_files(folder, req.recursive)
    if len(found) > MAX_ENCODE_FILES:
        raise HTTPException(status_code=400, detail=(
            f"这个文件夹里有 {len(found)} 个视频，一次最多 {MAX_ENCODE_FILES} 个，"
            f"请选一个小一点的文件夹"))
    files, skipped = [], []
    for path in found:
        try:
            info = encode.probe(path)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(path), "reason": f"读不出来：{exc}"})
            continue
        if info["video"] is None:
            skipped.append({"path": str(path), "reason": "没有画面"})
            continue
        info["relative"] = str(path.relative_to(folder))
        files.append(info)
    return {"path": str(folder), "files": files, "skipped": skipped, "discs": discs}


def _enqueued(entries: list, settings) -> dict:
    queue_manager.nudge()
    current = jobqueue.settings_hash(settings)
    waiting = [e for e in queue_manager.store.entries if e.status == "queued"]
    position = next((i + 1 for i, e in enumerate(waiting) if e.id == entries[0].id), 0)
    return {"entries": [_entry_view(e, current) for e in entries],
            "entry": _entry_view(entries[0], current),
            "count": len(entries), "position": position}


@router.post("/queue/encode")
def enqueue_encode(req: EncodeEnqueueRequest) -> dict:
    """Add one video's encode to the queue — 加入列队, or with `subtitles`
    加入列队并做字幕: the encode is translated once it is written."""
    source = _clean_path(req.source)
    if not source.is_file():
        raise HTTPException(status_code=400, detail=f"文件不存在：{source}")
    _check_encode(req.options, req.subtitles)
    if not audio.has_picture(source):
        raise HTTPException(status_code=400, detail="这个文件里没有画面，没有可压制的视频")
    custom = _custom_dir(req.output_mode, req.output_dir)
    if custom is not None and _same_folder(custom, source.parent):
        raise HTTPException(status_code=400, detail=(
            "指定的文件夹就是原文件所在的文件夹：请改选「放在原文件旁边」"))
    request = EncodeRequest(source=str(source), options=req.options,
                            output_mode=req.output_mode,
                            output_dir=str(custom) if custom is not None else "")
    then = _subtitles_after_encode(req.subtitles, "") if req.subtitles else None
    settings = jobqueue.snapshot()
    try:
        entry = queue_manager.store.add_encode(request, settings, then=then)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _enqueued([entry], settings)


@router.post("/queue/encode-batch")
def enqueue_encode_batch(req: EncodeBatchRequest) -> dict:
    """The batch mode's 加入列队: one entry per file, one group, one
    settings snapshot. Everything is checked first — a batch is never half
    queued. With a folder chosen, each file keeps its sub-folder under it,
    so two 01.mkv from two seasons do not end up as 01.mkv and 01.2.mkv."""
    folder = _clean_path(req.path)
    files = list(dict.fromkeys(req.files))
    if not files:
        raise HTTPException(status_code=400, detail="没有勾选任何文件")
    _check_encode(req.options, req.subtitles)
    custom = _custom_dir(req.output_mode, req.output_dir)
    requests, problems = [], []
    for name in files:
        source = _clean_path(name)
        if not source.is_file():
            problems.append(f"{source.name}：文件不存在")
            continue
        if kind_of(source) != "video":
            problems.append(f"{source.name}：不是视频文件")
            continue
        out_dir = ""
        if custom is not None:
            try:
                where = custom / source.parent.relative_to(folder)
            except ValueError:
                where = custom
            if _same_folder(where, source.parent):
                problems.append(f"{source.name}：指定的文件夹就是它所在的文件夹")
                continue
            out_dir = str(where)
        requests.append(EncodeRequest(source=str(source), options=req.options,
                                      output_mode=req.output_mode, output_dir=out_dir))
    if problems:
        raise HTTPException(status_code=400, detail="；".join(problems))
    _queue_room(len(requests))
    group_id = uuid.uuid4().hex[:12]
    then = None
    if req.subtitles is not None:
        then = _subtitles_after_encode(
            req.subtitles, f"enc-{group_id}" if req.subtitles.series_mode else "")
    settings = jobqueue.snapshot()
    added = []
    try:
        for request in requests:
            added.append(queue_manager.store.add_encode(
                request, settings, group_id=group_id, group_title=str(folder), then=then))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _enqueued(added, settings)


@router.post("/queue/pause")
def pause_queue(body: dict) -> dict:
    queue_manager.store.set_paused(bool(body.get("paused", True)))
    if not queue_manager.store.paused:
        queue_manager.nudge()
    return {"paused": queue_manager.store.paused}


@router.post("/queue/cpu-yield")
def set_cpu_yield(body: dict) -> dict:
    """压制让出 CPU：开着时压制以最低优先级运行，其他程序占用 CPU 超过 20%
    时限速。正在跑的压制一秒内跟上（services/cpuyield.py）。"""
    queue_manager.store.set_cpu_yield(bool(body.get("enabled", True)))
    return {"enabled": queue_manager.store.cpu_yield}


@router.post("/queue/memory-limit")
def set_memory_limit(body: dict) -> dict:
    """压制内存上限（GB，0 = 自动：物理内存的一半）。压制自己用的超过它就停下；
    正在跑的压制下一次检查（两秒内）就按新值算（services/memguard.py）。"""
    try:
        gb = float(body.get("gb", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="上限要是一个数（GB），0 表示自动")
    got = memguard.machine()
    total = got[0] / memguard.GIB if got else 0.0
    if gb < 0 or (gb and gb < 1) or (total and gb > total):
        raise HTTPException(status_code=400, detail=(
            f"上限要在 1 GB 到物理内存（{total:.0f} GB）之间，0 表示自动" if total
            else "上限至少 1 GB，0 表示自动"))
    queue_manager.store.set_memory_limit(gb)
    return {"gb": queue_manager.store.memory_limit_gb}


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
