"""MCP server: drive a translation from another machine.

Mounted at /mcp over streamable HTTP, so any MCP client that can reach the
port — on the LAN, or through a tunnel — can queue films on the GPU box and
collect the subtitles. Off by default (`MCPSettings.enabled`); the access
guard answers 404 while it is.

The tools are the REST surface, minus the settings page. A client that
could rewrite `asr.model_size` or the LLM key from a sentence of prose is a
much larger blast radius than one that can only start, watch and cancel
jobs — and the jobs are the point. Everything the other pages do is here:
translating now or through the queue, and the queue-only pages (原盘, 压制,
修复, 音频) with the analyses that come before them.

Nothing here reimplements the pipeline or the web's checks: the queue and
analysis tools call the route functions themselves (`app.api.routes`, a
refusal there comes back as {"error": …}), so what is refused, what is
frozen and when, is the same as the page by construction. That module
imports this one, so it is imported inside build().

The dependency is imported defensively: an installation whose venv predates
this feature still starts, with `mcp` left as None and the reason reported
through /api/server/info.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import anyio.to_thread
from pydantic import BaseModel, ValidationError

from app.core import joblog
from app.core.config import load_settings
from app.core.media import kind_of
from app.models.schemas import (
    GEMINI_ASR_MODEL,
    AudioBatchRequest,
    AudioOptions,
    AudioRequest,
    BatchRequest,
    DiscAnswer,
    DiscBatchRequest,
    DiscEncode,
    DiscEnqueueRequest,
    DiscOutputMode,
    DiscSubtitles,
    EmbedSettings,
    EncodeBatchRequest,
    EncodeEnqueueRequest,
    EncodeOptions,
    EncodeScanRequest,
    JobRequest,
)
from app.services import audio, jobqueue, mux, subsource
from app.services.batch import batch_manager
from app.services.pipeline import manager

SERVER_NAME = "MovieTranslator"
IMPORT_ERROR: str = ""

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
except Exception as exc:  # noqa: BLE001 — any import failure must not stop the app
    FastMCP = None  # type: ignore[assignment]
    IMPORT_ERROR = str(exc)

AVAILABLE = FastMCP is not None

# how much subtitle text one tool call may return; a two-hour bilingual SRT
# is well under this, but a client should never be handed an unbounded blob
MAX_SUBTITLE_CHARS = 200_000


def _container_error(container: str) -> str:
    """Why this container cannot be used, or empty.

    Kept out of pydantic's hands on purpose: EmbedSettings.container is a
    Literal, and a ValidationError traceback is a poor answer for a client
    that simply mistyped the format.
    """
    if container not in mux.CONTAINERS:
        return f"container 只能是 {' 或 '.join(mux.CONTAINERS)}"
    return ""


def _embed_error(container: str, video_codec: str) -> str:
    """Why this container/codec pair cannot be used here, or empty."""
    bad = _container_error(container)
    if bad:
        return bad
    if video_codec == mux.COPY:
        return ""
    usable = {enc["id"] for enc in mux.available_encoders()}
    if video_codec not in usable:
        return (f"本机无法使用编码器 {video_codec}。"
                f"可用的有：{'、'.join(sorted(usable)) or '（无）'}")
    return ""


# 手写的白名单，必须跟着 JobRequest.output_mode 那个 Literal 走。写成一处而
# 不是两处：从前 translate_video 和 translate_directory 各抄了一份，加了模式
# 只改一处就是客户端用不上另一个工具。
OUTPUT_MODES = ("bilingual", "translation_only", "original_only", "bilingual_split")

# files whose text get_queue_subtitle hands back; anything else (a muxed
# video) is answered with its path
SUBTITLE_SUFFIXES = (".srt", ".ass", ".ssa", ".vtt")


def _plain(value):
    """Route results as JSON: pydantic models anywhere inside become dicts."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _answer(fn, *args, **kwargs) -> dict:
    """Call a route function the way the page would, and turn its refusal
    into the {"error": …} every tool here answers with — the route's own
    words, which are written for a person."""
    from fastapi import HTTPException

    try:
        out = fn(*args, **kwargs)
    except HTTPException as exc:
        return {"error": exc.detail if isinstance(exc.detail, str) else str(exc.detail)}
    except ValidationError as exc:
        return {"error": str(exc)}
    out = _plain(out)
    return out if isinstance(out, dict) else {"result": out}


async def _in_thread(fn, *args, **kwargs) -> dict:
    """The routes are sync and most of them open files: off the event loop."""
    return await anyio.to_thread.run_sync(lambda: _answer(fn, *args, **kwargs))


def _encode_options(given: Optional[EncodeOptions]) -> EncodeOptions:
    """The 压制/修复 options a tool runs with: the settings page's defaults
    (AppSettings.encode — where the page's form starts too), with only the
    fields the client actually sent laid over them. A client asking for
    {"preset": "slow"} gets slow on top of the user's defaults, not on top
    of the model's."""
    base = load_settings().encode.model_copy(deep=True)
    if given is None:
        return base
    return base.model_copy(update={k: getattr(given, k) for k in given.model_fields_set})


def _job_request(video_path: str, **fields) -> JobRequest | str:
    """A JobRequest from translate_video's arguments, or why not — the
    checks the two translate tools and their queue twins share."""
    if fields["output_mode"] not in OUTPUT_MODES:
        return f"output_mode 只能是 {'、'.join(OUTPUT_MODES)}"
    if fields["text_source"] not in ("asr", "subtitle"):
        return "text_source 只能是 asr 或 subtitle"
    # the encoder half is left to manager.create, which knows whether this
    # source has a picture at all — an audio file degrades rather than
    # being turned away over an encoder it will never reach
    bad = _container_error(fields["container"])
    if bad:
        return bad
    container = fields.pop("container")
    video_codec = fields.pop("video_codec")
    return JobRequest(video_path=video_path,
                      embed=EmbedSettings(container=container, video_codec=video_codec),
                      **fields)


def _batch_request(directory: str, **fields) -> BatchRequest | str:
    if fields["output_mode"] not in OUTPUT_MODES:
        return f"output_mode 只能是 {'、'.join(OUTPUT_MODES)}"
    if fields["text_source"] not in ("asr", "subtitle"):
        return "text_source 只能是 asr 或 subtitle"
    bad = _embed_error(fields["container"], fields["video_codec"])
    if bad:
        return bad
    container = fields.pop("container")
    video_codec = fields.pop("video_codec")
    return BatchRequest(directory=directory,
                        embed=EmbedSettings(container=container, video_codec=video_codec),
                        **fields)


def _preview_view(preview) -> dict:
    """The 试看 as the page polls it, plus where its files are on disk —
    the page plays them over HTTP; an MCP client on this machine opens
    them."""
    view = preview.view()
    folder = preview.folder
    view["file_paths"] = (
        {k: str(folder / name) for k, name in preview.files.items()} if folder else {})
    return view


def _job_view(job) -> dict:
    st = job.status
    return {
        "job_id": st.id,
        "stage": st.stage,
        "progress": st.progress,
        "message": st.message,
        "error": st.error,
        "video_path": st.video_path,
        "subtitle_path": st.srt_filename,
        # 双文件模式的另一半；其余模式是空串
        "original_subtitle_path": st.original_srt_filename,
        "finished": st.stage in ("done", "failed", "cancelled"),
    }


def build() -> Optional["FastMCP"]:
    """A fresh server instance.

    Built per lifespan rather than once at import: a session manager may
    only be run once, so a reused module-level instance would refuse to
    start a second time — which is exactly what a test process that opens
    more than one client does.
    """
    if FastMCP is None:
        return None

    mcp = FastMCP(
        name=SERVER_NAME,
        stateless_http=True,
        json_response=True,
        # The SDK's DNS-rebinding guard checks the Host header against a
        # list, and that list cannot be written down here: the whole point
        # of this feature is being reached at whatever address the user's
        # network hands out, plus any reverse proxy in front. Left on, it
        # answers every LAN client with an unexplained 421.
        # The protection it provides is kept, just moved: AccessGuard
        # refuses any /mcp request carrying an Origin header, which is the
        # header a rebinding attack cannot avoid and a real MCP client
        # never sends.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    # without this the client URL would be /mcp/mcp
    mcp.settings.streamable_http_path = "/"

    from app.api import routes
    from app.services import restore_preview

    # ---------------------------------------------------------- discovery

    @mcp.tool()
    async def list_videos(
        directory: str, recursive: bool = True, skip_translated: bool = True,
        target_language: str = "", subtitle_language: str = "",
    ) -> dict:
        """列出某个目录下可翻译的视频、音频与字幕文件（首页「批量」的预览）。

        directory: 服务器本机的目录绝对路径。
        recursive: 是否递归子目录。
        skip_translated: 跳过已有目标语言字幕的文件。填了 target_language 时按字幕
            的**语言**判断（别的语言的字幕不算翻完，而是可用的原文，列在
            with_source 里）；不填则按旧规则判断。
        subtitle_language: 打算用片源字幕当原文时，按这个语言挑轨。

        audio: 上面这批里的纯音频文件——它们只能产出字幕文件，
            不支持内嵌合成新视频，也没有画面翻译。
        shadowed: 与同名视频重名而被让位的音频文件（两者会写同一份字幕）。
        replaced: 被同名字幕文件顶替的视频/音频（更快，但不会产出内嵌视频、不会用语音识别）。
        with_source: 已经带有别的语言字幕的文件——可以用 text_source="subtitle" 直接翻那份。
        discs: 目录里的原盘（BDMV / VIDEO_TS / .iso），不会被逐个文件翻译，
            要用 analyze_disc / queue_disc。
        """
        out = await _in_thread(routes.batch_scan, directory, recursive, skip_translated,
                               target_language, subtitle_language)
        if "error" not in out:
            out["subtitle"] = [v for v in out["videos"] if kind_of(Path(v)) == "subtitle"]
        return out

    @mcp.tool()
    async def browse_files(path: str = "", mode: str = "") -> dict:
        """浏览服务器本机的一个目录（网页的文件选择框）。

        path: 目录绝对路径；留空为主目录（Windows 为盘符列表）。
        mode: ""（视频/音频/字幕，翻译用）、"video"（只列视频，压制/修复/音频用）、
            "disc"（只列 .iso，并在 disc_dirs 里标出哪些子文件夹本身就是原盘，
            is_disc 表示当前目录就是原盘）。
        """
        return await _in_thread(routes.fs_browse, path, mode)

    @mcp.tool()
    async def list_audio_tracks(video_path: str) -> dict:
        """列出视频的音轨，用于挑选要识别的那一条。

        返回的 index 即 translate_video 的 audio_track 参数。
        """
        path = Path(video_path).expanduser()
        if not path.is_file():
            return {"error": f"文件不存在: {path}"}
        tracks = await anyio.to_thread.run_sync(audio.list_tracks, path)
        return {"tracks": tracks}

    @mcp.tool()
    async def list_subtitle_tracks(video_path: str) -> dict:
        """列出视频已有的字幕（内嵌轨 + 同目录同名字幕文件）。

        把其中一条交给 translate_video（text_source="subtitle" 加上
        subtitle_track 或 subtitle_file），就能跳过语音识别直接翻译现成的
        原文——又快又准。text 为 false 的是图形字幕（PGS/VobSub），会先用 OCR
        认成文字，比文字字幕慢得多。
        """
        path = Path(video_path).expanduser()
        if not path.is_file():
            return {"error": f"文件不存在: {path}"}
        tracks = await anyio.to_thread.run_sync(subsource.all_tracks, path)
        return {"tracks": tracks}

    @mcp.tool()
    async def list_encoders() -> dict:
        """本机能用的容器、视频编码器、音频编码与各编码器的画质刻度。

        video 里的 id 就是 translate_video 的 video_codec、以及压制/修复
        options.video_codec 可填的值（硬件编码器只有本机真的能打开才会列出）；
        capabilities 说明每个编码器的 quality 刻度、能不能 10bit、有哪些 tune；
        deinterlace 表示本机能不能反交错。
        """
        return await _in_thread(routes.media_encoders)

    @mcp.tool()
    async def get_server_status() -> dict:
        """查询本机翻译服务的版本、识别配置、列队与当前任务情况。不返回任何密钥。"""
        settings = load_settings()
        jobs = list(manager.jobs.values())
        running = [_job_view(j) for j in jobs if j.status.stage not in
                   ("done", "failed", "cancelled")]
        store = routes.queue_manager.store
        with store.lock:
            waiting = sum(1 for e in store.entries if e.status == "queued")
            active = next((e.id for e in store.entries if e.status == "running"), "")
        return {
            "version": joblog.APP_VERSION,
            "asr_engine": settings.asr.engine,
            "asr_model": (
                GEMINI_ASR_MODEL
                if settings.asr.engine == "api"
                else settings.asr.model_path.strip() or settings.asr.model_size
            ),
            "asr_device": settings.asr.device,
            "llm_model": settings.llm.model,
            "second_pass": settings.asr.second_pass,
            "refine_enabled": settings.prompts.refine_enabled,
            "mark_lyrics": settings.prompts.mark_lyrics,
            # 修复的 AI 放大要单独的引擎环境；没配时只能用 Lanczos
            "restore_engine_configured": bool(settings.restore.engine_python.strip()),
            "encode_defaults": settings.encode.model_dump(),
            "queue": {"paused": store.paused, "waiting": waiting, "running_entry": active},
            "active_jobs": running,
            "total_jobs": len(jobs),
        }

    # ------------------------------------------------------------- start

    @mcp.tool()
    async def translate_video(
        video_path: str,
        target_language: str = "简体中文",
        source_language: str = "auto",
        synopsis: str = "",
        output_mode: str = "bilingual",
        audio_track: Optional[int] = None,
        audio_language: str = "",
        embed_subtitle: bool = False,
        container: str = "mkv",
        video_codec: str = "copy",
        text_source: str = "asr",
        subtitle_track: Optional[int] = None,
        subtitle_file: str = "",
    ) -> dict:
        """为一个视频文件**立即**启动字幕翻译（首页「开始翻译」），立即返回 job_id，不等待完成。

        不进列队：它插在列队里还没开始的任务前面（同一时间仍只跑一个重任务），
        服务重启后也不会自动重跑。要排队请用 queue_translate_video。

        整个流程需要几分钟到几小时，请随后用 get_job 轮询进度，
        完成后用 get_subtitle 取回字幕。用的是调用这一刻的设置。

        video_path: 服务器本机的视频、音频或**字幕文件**绝对路径。字幕文件（.srt/.ass/.vtt/.sub/.sup/.idx）可以单独翻译——没有画面，所以不会产出内嵌视频，也不会用语音识别。
        source_language: 影片原始语言代码（如 ja / en），auto 为自动判断。
        synopsis: 剧情简介，可显著提升人名与代词的翻译准确度。
        output_mode: bilingual 双语，translation_only 只要译文，
            original_only 只要原文——跳过 AI 翻译，但转写预处理、歌词识别、
            二次识别复核、图形字幕 OCR 与校对照常执行。产物带源语言后缀
            （片名.ja.srt，内嵌时 片名.ja.mkv，轨道标为该语言）。此时
            target_language 只作用于画面翻译，对白保持原文。
            bilingual_split 内容与 bilingual 相同，但写成两个单语文件
            （片名.zh.srt + 片名.ja.srt）；内嵌时是一个视频里两条字幕轨，
            译文默认打开。原文那一份的路径在 get_job 的
            original_subtitle_path 里。
            **四种模式的字幕都带语言后缀，后缀就是文件里的内容**：双语两个
            都写、屏幕上哪一行在上哪个就在前（片名.ja-zh.srt），纯译文写目标
            语言，纯原文写源语言。片名.srt 这个名字不再产出。
            同名文件已存在时产物带编号让路（片名.zh.2.srt）：本程序绝不覆盖
            片源目录里任何已经存在的文件。
        audio_track: 音轨的容器序号，留空用默认音轨。
        audio_language: 按语言标签选音轨（如 jpn），仅在 audio_track 留空时生效。
        text_source: asr 走语音识别；subtitle 直接读片源已有的字幕，跳过识别——
            片源自带外文字幕时又快又准，先用 list_subtitle_tracks 看有哪些。
        subtitle_track: 要读取的字幕轨容器序号（text_source="subtitle" 时）。
            指向图形字幕（PGS/VobSub）会先用 OCR 认成文字，比文字字幕慢得多。
        subtitle_file: 改为读取这个外挂字幕文件（.srt/.ass），优先于 subtitle_track。
        embed_subtitle: 开启后不生成字幕文件，而是在同目录产出一个内嵌软字幕的
            新视频（片名.zh.mkv，纯原文时后缀是源语言；音视频不重编码）。会完整复制一份视频，注意磁盘空间。
            片源原有的字幕轨会保留。
            纯音频片源没有画面可合成，会自动改为生成字幕文件。
        container: 新视频的容器，mkv 或 mp4。mp4 兼容性最好，但只能带纯文本字幕轨，
            也装不下片源自带的字幕轨与字体附件。仅在 embed_subtitle 开启时有意义。
        video_codec: copy 表示原样拷贝不重编码（默认，唯一不损失画质的选项）。
            要重编码就填编码器 id，可用的用 list_encoders 查。重编码整部影片
            动辄数小时，画质只减不增。
        """
        request = _job_request(
            video_path, audio_track=audio_track, audio_language=audio_language,
            text_source=text_source, subtitle_track=subtitle_track,
            subtitle_file=subtitle_file, source_language=source_language,
            target_language=target_language, synopsis=synopsis,
            output_mode=output_mode, embed_subtitle=embed_subtitle,
            container=container, video_codec=video_codec)
        if isinstance(request, str):
            return {"error": request}
        try:
            # what the settings say now is what this job runs with, as when
            # the page's button is pressed (routes.create_job)
            job = manager.create(request, settings=jobqueue.snapshot())
        except (FileNotFoundError, ValueError) as exc:
            return {"error": str(exc)}
        return _job_view(job)

    @mcp.tool()
    async def translate_directory(
        directory: str,
        target_language: str = "简体中文",
        source_language: str = "auto",
        synopsis: str = "",
        output_mode: str = "bilingual",
        recursive: bool = True,
        skip_translated: bool = True,
        audio_language: str = "",
        series_mode: bool = False,
        embed_subtitle: bool = False,
        container: str = "mkv",
        video_codec: str = "copy",
        text_source: str = "asr",
        subtitle_language: str = "",
    ) -> dict:
        """为一个目录下的所有视频、音频与字幕文件**立即**启动批量翻译，立即返回 batch_id。

        不进列队（同 translate_video）；要排队请用 queue_translate_directory。
        影片逐个串行处理（语音识别占满 CPU/GPU，同时只跑一个），整批用调用
        这一刻的同一份设置。用 get_batch 查看整体进度。

        series_mode（剧集模式）：整个目录是同一部剧的多集时开启，先译出的
        人名/术语译法会强制沿用到后续每一集；目录里是互不相干的影片则不要
        开启。累积的对照表可在 get_batch 的 glossary 字段查看。

        text_source="subtitle"：直接读每个视频已有的字幕，跳过语音识别。
        subtitle_language 按语言标签挑轨（如 eng；各文件的轨道序号不同）。
        找不到可读字幕的那个文件会自动改用语音识别，不影响整批。

        output_mode：同 translate_video，整批共用。original_only（纯原文）跳过
        AI 翻译，每个文件产出带源语言后缀的原文字幕（片名.ja.srt）；此时剧集模式
        不会积累新的译名表，因为没有译文。bilingual_split（双文件）每个文件产出
        译文、原文各一份。四种模式的产物都带语言后缀，且绝不覆盖目录里已经存在
        的文件——详见 translate_video。

        container / video_codec：同 translate_video，整批共用。
        embed_subtitle：同 translate_video——每个视频产出一个内嵌软字幕的新 mkv，
        不生成字幕文件。整季剧集会因此多占一整份磁盘空间。目录里的纯音频文件
        没有画面可合成，会各自改为生成字幕文件，不影响整批。
        """
        request = _batch_request(
            directory, recursive=recursive, skip_existing_srt=skip_translated,
            audio_language=audio_language, text_source=text_source,
            subtitle_language=subtitle_language, source_language=source_language,
            target_language=target_language, synopsis=synopsis,
            output_mode=output_mode, series_mode=series_mode,
            embed_subtitle=embed_subtitle, container=container, video_codec=video_codec)
        if isinstance(request, str):
            return {"error": request}
        try:
            # one snapshot for the whole batch (routes.create_batch)
            status = await anyio.to_thread.run_sync(
                lambda: batch_manager.create(request, settings=jobqueue.snapshot()))
        except (NotADirectoryError, ValueError) as exc:
            return {"error": str(exc)}
        return {
            "batch_id": status.id,
            "total": status.total,
            "job_ids": [j.id for j in status.jobs],
            "skipped": status.skipped,
        }

    # ------------------------------------------------------------ follow

    @mcp.tool()
    async def get_job(job_id: str) -> dict:
        """查询单个任务的阶段与进度（立即开始的翻译，或列队条目的 job_id）。"""
        try:
            return _job_view(manager.get(job_id))
        except KeyError:
            return {"error": f"没有这个任务: {job_id}"}

    @mcp.tool()
    async def get_batch(batch_id: str) -> dict:
        """查询批量翻译的整体进度与每个任务的状态。"""
        try:
            status = batch_manager.status(batch_id)
        except KeyError:
            return {"error": f"没有这个批量任务: {batch_id}"}
        return status.model_dump()

    @mcp.tool()
    async def cancel_job(job_id: str) -> dict:
        """取消一个立即开始的翻译任务（列队条目请用 cancel_queue_entry）。"""
        try:
            manager.cancel(job_id)
        except KeyError:
            return {"error": f"没有这个任务: {job_id}"}
        return {"ok": True, "job_id": job_id}

    @mcp.tool()
    async def cancel_batch(batch_id: str) -> dict:
        """取消一个批量任务下所有尚未完成的影片。"""
        try:
            batch_manager.cancel(batch_id)
        except KeyError:
            return {"error": f"没有这个批量任务: {batch_id}"}
        return {"ok": True, "batch_id": batch_id}

    # ------------------------------------------------------------ result

    @mcp.tool()
    async def get_subtitle(job_id: str, part: str = "translation") -> dict:
        """取回已完成任务的字幕全文（SRT 或 ASS）。

        字幕文件本身也已经写在视频旁边，路径见 subtitle_path。
        part: translation（默认）；original 取双文件模式（bilingual_split）的原文那一份。
        列队条目在服务重启后也能取，用 get_queue_subtitle。
        """
        try:
            job = manager.get(job_id)
        except KeyError:
            return {"error": f"没有这个任务: {job_id}"}
        if job.status.stage != "done" or not job.srt_path:
            return {
                "error": f"任务尚未完成（当前阶段: {job.status.stage}）",
                "stage": job.status.stage,
                "progress": job.status.progress,
            }
        if part == "translation":
            path = Path(job.srt_path)
        elif part == "original":
            path = Path(job.status.original_srt_filename or "")
            if not path.name:
                return {"error": "这个任务没有单独的原文字幕"}
        else:
            return {"error": "part 只能是 translation 或 original"}
        text = await anyio.to_thread.run_sync(lambda: path.read_text(encoding="utf-8"))
        truncated = len(text) > MAX_SUBTITLE_CHARS
        return {
            "subtitle_path": str(path) if part == "original" else job.status.srt_filename,
            "content": text[:MAX_SUBTITLE_CHARS],
            "truncated": truncated,
            "total_chars": len(text),
        }

    @mcp.tool()
    async def get_job_log(job_id: str, tail_lines: int = 200) -> dict:
        """取回任务日志的末尾若干行，用于排查失败原因。日志不含 API key。

        列队条目（含原盘、压制、修复、音频）的日志也用它，job_id 见 get_queue。
        """
        path = joblog.find_log(job_id)
        if path is None:
            return {"error": f"该任务没有日志文件: {job_id}"}
        lines = await anyio.to_thread.run_sync(
            lambda: path.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        tail = lines[-max(tail_lines, 1):]
        return {
            "log_file": path.name,
            "total_lines": len(lines),
            "lines": tail,
        }

    @mcp.tool()
    async def list_logs() -> dict:
        """最近的任务日志文件（新的在前，保留最近 20 个），以及日志所在目录。"""
        return await _in_thread(routes.list_logs)

    @mcp.tool()
    async def get_log_file(name: str, tail_lines: int = 200) -> dict:
        """按文件名（list_logs 给出的 name）取回一份任务日志的末尾若干行。"""
        path = joblog.logs_dir() / Path(name).name
        if not path.is_file():
            return {"error": f"日志文件不存在: {name}"}
        lines = await anyio.to_thread.run_sync(
            lambda: path.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        return {"log_file": path.name, "total_lines": len(lines),
                "lines": lines[-max(tail_lines, 1):]}

    # ------------------------------------------------------------- queue

    @mcp.tool()
    async def get_queue(status: str = "") -> dict:
        """查看列队（网页「列队」页）：每条任务的种类、状态、进度、产物与开关状态。

        status: 只看某种状态的条目（queued / running / done / failed / cancelled），留空看全部。
        kind: job（翻译）/ disc（原盘封装）/ encode（压制与修复）/ audio（音频提取）。
        每条任务用的是**加入列队那一刻**的设置；settings_differs 表示与现在的设置不同。
        产物：翻译看 result_srt / result_video，其余看 result_files（都是本机路径）。
        job_id 可交给 get_job 看实时进度、交给 get_job_log 看日志。
        """
        if status and status not in ("queued", "running", "done", "failed", "cancelled"):
            return {"error": "status 只能是 queued / running / done / failed / cancelled，或留空"}
        out = await _in_thread(routes._queue_view)
        if status and "entries" in out:
            out["entries"] = [e for e in out["entries"] if e["status"] == status]
        return out

    @mcp.tool()
    async def get_queue_entry_snapshot(entry_id: str) -> dict:
        """这条列队任务冻结的设置（与任务日志里那一节逐字相同，不含密钥），
        以及其中与现在的设置不同的行（differs）。"""
        return await _in_thread(routes.queue_entry_settings, entry_id)

    @mcp.tool()
    async def get_queue_subtitle(entry_id: str, part: str = "translation") -> dict:
        """取回一条已完成的翻译列队任务的字幕全文；服务重启后也能取。

        part: translation（默认）或 original（双文件模式的原文那一份）。
        产物是内嵌字幕的视频时只返回它的路径。
        """
        def read():
            path = routes.queue_result_path(entry_id, part)
            if path.suffix.lower() not in SUBTITLE_SUFFIXES:
                return {"path": str(path), "content": None,
                        "note": "这条任务的产物是视频，不是字幕文件"}
            text = path.read_text(encoding="utf-8", errors="replace")
            return {"path": str(path), "content": text[:MAX_SUBTITLE_CHARS],
                    "truncated": len(text) > MAX_SUBTITLE_CHARS, "total_chars": len(text)}

        return await _in_thread(read)

    @mcp.tool()
    async def cancel_queue_entry(entry_id: str) -> dict:
        """取消一条列队任务：还没开始的直接取消，正在跑的会停下并删掉半成品。"""
        return await _in_thread(lambda: {"status": routes.cancel_queue_entry(entry_id)["status"]})

    @mcp.tool()
    async def retry_queue_entry(entry_id: str, use_current_settings: bool = False) -> dict:
        """把一条已结束（失败/取消/完成）的列队任务重新加到队尾。

        use_current_settings: false（默认）沿用它当初冻结的设置；true 改用现在的设置。
        原盘、压制重跑时已写好的产物会被认出来跳过，不会重写一份 .2。
        """
        return await _in_thread(routes.retry_queue_entry, entry_id,
                                {"fresh": use_current_settings})

    @mcp.tool()
    async def remove_queue_entry(entry_id: str) -> dict:
        """从列队里删掉一条任务（正在跑的要先取消）。不删除任何已写出的文件。"""
        return await _in_thread(routes.remove_queue_entry, entry_id)

    @mcp.tool()
    async def clear_finished_queue() -> dict:
        """从列队里删掉所有已结束（完成/失败/取消）的条目。不删除任何文件。"""
        return await _in_thread(routes.clear_finished)

    @mcp.tool()
    async def reorder_queue(ids: list[str]) -> dict:
        """调整等待中任务的顺序。

        ids: **全部**等待中（queued）条目的 id，按想要的顺序排好——少一个、多一个
            都会被拒绝（列队已经变了），请先 get_queue(status="queued") 再排。
        """
        return await _in_thread(routes.reorder_queue, {"ids": ids})

    @mcp.tool()
    async def set_queue_options(
        paused: Optional[bool] = None,
        cpu_yield: Optional[bool] = None,
        memory_limit_gb: Optional[float] = None,
    ) -> dict:
        """列队页的三个开关，只改传了的那几个，返回改完后的状态。

        paused: 暂停列队——正在跑的那条跑完，之后不再开始新的。
        cpu_yield: 压制让出 CPU——压制以最低优先级运行，其他程序占用 CPU 超过 20%
            时限速（最低用 10%）。只管压制（含修复），立即生效。
        memory_limit_gb: 压制内存上限（GB），0 表示自动（物理内存的一半）；
            要在 1 到物理内存之间。压制自己用的超过它就停下。
        """
        def apply():
            if paused is not None:
                routes.pause_queue({"paused": paused})
            if cpu_yield is not None:
                routes.set_cpu_yield({"enabled": cpu_yield})
            if memory_limit_gb is not None:
                routes.set_memory_limit({"gb": memory_limit_gb})
            store = routes.queue_manager.store
            return {"paused": store.paused, "cpu_yield": store.cpu_yield,
                    "memory_limit_gb": store.memory_limit_gb}

        return await _in_thread(apply)

    @mcp.tool()
    async def queue_translate_video(
        video_path: str,
        target_language: str = "简体中文",
        source_language: str = "auto",
        synopsis: str = "",
        output_mode: str = "bilingual",
        audio_track: Optional[int] = None,
        audio_language: str = "",
        embed_subtitle: bool = False,
        container: str = "mkv",
        video_codec: str = "copy",
        text_source: str = "asr",
        subtitle_track: Optional[int] = None,
        subtitle_file: str = "",
    ) -> dict:
        """把一个文件的翻译**加入列队**（首页「加入列队」）。参数同 translate_video。

        与 translate_video 的区别：排在列队末尾依次跑；用的是**加入这一刻**的设置；
        服务重启后会从头重跑而不是丢掉。返回列队条目与它在等待中的位置，
        用 get_queue 跟进。
        """
        request = _job_request(
            video_path, audio_track=audio_track, audio_language=audio_language,
            text_source=text_source, subtitle_track=subtitle_track,
            subtitle_file=subtitle_file, source_language=source_language,
            target_language=target_language, synopsis=synopsis,
            output_mode=output_mode, embed_subtitle=embed_subtitle,
            container=container, video_codec=video_codec)
        if isinstance(request, str):
            return {"error": request}
        return await _in_thread(routes.enqueue_job, request)

    @mcp.tool()
    async def queue_translate_directory(
        directory: str,
        target_language: str = "简体中文",
        source_language: str = "auto",
        synopsis: str = "",
        output_mode: str = "bilingual",
        recursive: bool = True,
        skip_translated: bool = True,
        audio_language: str = "",
        series_mode: bool = False,
        embed_subtitle: bool = False,
        container: str = "mkv",
        video_codec: str = "copy",
        text_source: str = "asr",
        subtitle_language: str = "",
    ) -> dict:
        """把一个目录的批量翻译**加入列队**（首页批量的「加入列队」）。参数同 translate_directory。

        加入时就展开成每个文件一条（同一组、同一份设置快照），可以单独删除、调序、重试。
        剧集模式的译名表会落盘，跨重启接着用。
        """
        request = _batch_request(
            directory, recursive=recursive, skip_existing_srt=skip_translated,
            audio_language=audio_language, text_source=text_source,
            subtitle_language=subtitle_language, source_language=source_language,
            target_language=target_language, synopsis=synopsis,
            output_mode=output_mode, series_mode=series_mode,
            embed_subtitle=embed_subtitle, container=container, video_codec=video_codec)
        if isinstance(request, str):
            return {"error": request}
        return await _in_thread(routes.enqueue_batch, request)

    # -------------------------------------------------------------- 原盘

    @mcp.tool()
    async def analyze_disc(
        path: str,
        series: Optional[bool] = None,
        episode_start: Optional[int] = None,
        name: str = "",
        output_mode: Optional[DiscOutputMode] = None,
        output_dir: str = "",
    ) -> dict:
        """分析一张原盘（蓝光文件夹 / VIDEO_TS / .iso），列出每个标题及其判定（「原盘」页）。

        只读元数据，几毫秒。titles 里每个标题有 id、label（正片/第 N 集/花絮/重复…）、
        category、理由、selected（默认是否勾选）、selectable、计划写出的文件名。
        加密盘与只有元数据的副本会标出来，不能封装。

        series: 整片还是分集——null 由盘的结构决定（question 字段非空时表示这张盘
            需要回答），true 按分集、false 按整片。改了它标题 id 可能不同，
            入队时要传同样的值。
        episode_start: 起始集号；name: 片名（默认取文件夹名）；
        output_mode: beside（光盘所在文件夹，默认）/ inside（光盘自己的文件夹）/
            custom（output_dir 指定的文件夹）。
        """
        return await _in_thread(routes.disc_scan, path, series, episode_start, name,
                                output_mode, output_dir)

    @mcp.tool()
    async def analyze_disc_folder(
        path: str,
        recursive: bool = True,
        output_mode: Optional[DiscOutputMode] = None,
        output_dir: str = "",
        discs: Optional[list[DiscAnswer]] = None,
    ) -> dict:
        """分析一个文件夹里的所有原盘（「原盘」页的批量模式），每张盘同 analyze_disc。

        多卷合集（VOL01/VOL02…）会被认成一套：共用片名，集号与花絮编号跨卷接着编。
        读不出来的盘列在 skipped 里并写明原因。
        discs: 已经给过的回答（每张盘的 path、series、name、episode_start），没给的用默认。
        """
        request = DiscBatchRequest(path=path, recursive=recursive, output_mode=output_mode,
                                   output_dir=output_dir, discs=discs or [])
        return await _in_thread(routes.disc_batch_scan, request)

    @mcp.tool()
    async def queue_disc(
        path: str,
        titles: Optional[list[str]] = None,
        series: Optional[bool] = None,
        name: str = "",
        episode_start: Optional[int] = None,
        output_mode: Optional[DiscOutputMode] = None,
        output_dir: str = "",
        subtitles: Optional[DiscSubtitles] = None,
        encode: Optional[EncodeOptions] = None,
        keep_lossless: bool = False,
    ) -> dict:
        """把一张原盘的封装（只换容器成 MKV，不重编码、不解密）**加入列队**。

        先用 analyze_disc 看标题。series / name / episode_start / output_mode 要与
        分析时一致。titles: 要封装的标题 id；留空＝分析结果默认勾选的那些。

        三种用法对应页面的三个按钮：
        - 只传上面这些：「加入列队」，只封装。
        - 加 subtitles：「加入列队并做字幕」，每个 MKV（含花絮）写完后自动加一条
          翻译任务到列队末尾。subtitles 的字段与默认值同首页批量
          （target_language 默认简体中文，series_mode 默认开，按盘/合集共用译名表）。
        - 再加 encode：「加入列队并压制、做字幕」，每个 MKV 先压制再做字幕。
          encode 是压制参数（同 queue_encode 的 options，未填的字段取设置页的
          压制默认值，传 {} 即全用默认；容器必须是 mkv）。压制校验通过后默认
          **用压制版替换无损 MKV**并沿用原名；keep_lossless=true 则保留无损版、
          压制版另起名。
        """
        def run():
            request = DiscEnqueueRequest(
                path=path, titles=titles or [], series=series, name=name,
                episode_start=episode_start, output_mode=output_mode,
                output_dir=output_dir, subtitles=subtitles,
                encode=(DiscEncode(options=_encode_options(encode), keep_lossless=keep_lossless)
                        if encode is not None else None))
            return routes.enqueue_disc(request)

        return await _in_thread(run)

    @mcp.tool()
    async def queue_disc_folder(
        path: str,
        discs: list[DiscAnswer],
        recursive: bool = True,
        output_mode: Optional[DiscOutputMode] = None,
        output_dir: str = "",
        subtitles: Optional[DiscSubtitles] = None,
        encode: Optional[EncodeOptions] = None,
        keep_lossless: bool = False,
    ) -> dict:
        """把一个文件夹里的原盘**加入列队**（批量模式），每张盘一条、同一组。

        先用 analyze_disc_folder 分析。discs: 要入队的盘——只入队这里点名且
        include=true 的那几张（不是文件夹现在有的全部）；每张可带 series、name、
        episode_start、titles（留空＝默认勾选）。只要有一张被拒，整批都不入队。
        subtitles / encode / keep_lossless 同 queue_disc，整批共用。
        """
        def run():
            request = DiscBatchRequest(
                path=path, recursive=recursive, output_mode=output_mode,
                output_dir=output_dir, discs=discs, subtitles=subtitles,
                encode=(DiscEncode(options=_encode_options(encode), keep_lossless=keep_lossless)
                        if encode is not None else None))
            return routes.enqueue_disc_batch(request)

        return await _in_thread(run)

    # ------------------------------------------------------------ 压制/修复

    @mcp.tool()
    async def probe_video(path: str) -> dict:
        """读一个视频的流信息（「压制」页选好文件后显示的那些）：分辨率、帧率、
        编码、HDR、是否隔行、每条音轨/字幕轨。"""
        return await _in_thread(routes.encode_probe, path)

    @mcp.tool()
    async def scan_encode_folder(path: str, recursive: bool = True) -> dict:
        """列出一个文件夹里可压制/修复的视频及其流信息（「压制」「修复」页的批量模式）。

        不进原盘里找（原盘列在 discs 里，要用原盘工具）。一次最多 500 个视频。
        files 里的 path 就是 queue_encode_batch 的 files。
        """
        return await _in_thread(routes.encode_scan, EncodeScanRequest(path=path, recursive=recursive))

    @mcp.tool()
    async def pick_encode(path: str, options: Optional[EncodeOptions] = None) -> dict:
        """按画面选编码：截几帧拼成一张图问视觉模型（动画还是真人、颗粒轻重），
        按实测表给出编码、速度档、质量与 tune（「压制」页选好单个文件时做的事）。

        options: 当前的压制参数（未填的取设置页默认值），只有画面那几项会被替换。
        返回 content / grain / reason，以及替换后的完整 options——拿它去 queue_encode。
        要视觉模型（设置里配置），会花一次请求的钱。拼图本身不返回。
        """
        def run():
            out = routes.encode_pick({"path": path,
                                      "options": _encode_options(options).model_dump()})
            out.pop("image", None)
            return out

        return await _in_thread(run)

    @mcp.tool()
    async def queue_encode(
        source: str,
        options: Optional[EncodeOptions] = None,
        output_mode: Literal["beside", "custom"] = "beside",
        output_dir: str = "",
        subtitles: Optional[DiscSubtitles] = None,
    ) -> dict:
        """把一个视频的压制或**修复****加入列队**（「压制」「修复」页的「加入列队」）。

        永不删除、不替换原文件：beside 时在原文件旁边另起名（片名.HEVC.mkv，修复放大后
        如 片名.1080p.60fps.HEVC.mkv）；custom 时写进 output_dir、保持原名。

        options: 压制参数，**未填的字段取设置页的压制默认值**（get_server_status 的
        encode_defaults），可用的编码器见 list_encoders。常用字段：
        - container mkv/mp4；video_codec 编码器 id（copy＝画面不动）；rate_control
          quality/bitrate；quality（CRF，越小越好）；bitrate_kbps；preset
          ultrafast…veryslow；tune ""/film/animation/grain；bit_depth auto/8/10；
          max_height 缩小到的短边（0 不变）；deinterlace auto/off/all/ivtc/bob/match/detect。
        - 音频：audio_codec copy/aac/libopus/ac3/eac3/flac，audio_scope lossless/all，
          audio_bitrate_kbps，audio_mixdown keep/stereo，audio_languages 保留的语言；
          subtitles all/languages/none 与 subtitle_languages。
        - auto_pick=true：开压时让视觉模型按画面自动选编码（见 pick_encode）。
        - **修复**（老 VHS / DVD，先用 analyze_restore 看建议）：deinterlace="bob"
          把隔行还原成 59.94 帧、"match" 只做场匹配、"ivtc" 还原 24 帧、"detect" 开压时
          自动判断；field_order auto/tff/bff；crop_top/bottom/left/right（双数）与
          crop_auto 自动裁黑边；aspect auto/4:3/16:9；denoise off/light/strong；
          upscale 放大到的短边 0/720/1080/1440/2160（与 max_height 互斥）；
          ai_model ""（Lanczos）/ realviformer / realbasicvsr / liveaction_span——
          AI 放大要设置里配好修复引擎（get_server_status 的 restore_engine_configured），
          可先用 start_restore_preview 试看对比。
        subtitles: 「加入列队并做字幕」——压完后把产物加一条翻译任务到列队末尾
            （字段同 queue_disc 的 subtitles）。
        """
        def run():
            return routes.enqueue_encode(EncodeEnqueueRequest(
                source=source, options=_encode_options(options), output_mode=output_mode,
                output_dir=output_dir, subtitles=subtitles))

        return await _in_thread(run)

    @mcp.tool()
    async def queue_encode_batch(
        path: str,
        files: list[str],
        options: Optional[EncodeOptions] = None,
        output_mode: Literal["beside", "custom"] = "beside",
        output_dir: str = "",
        subtitles: Optional[DiscSubtitles] = None,
    ) -> dict:
        """把一个文件夹里选中的视频的压制或修复**加入列队**（批量模式），每个文件一条、同一组。

        path: 扫描的文件夹（scan_encode_folder 的 path）；files: 要压的视频的完整路径。
        custom 时每个文件保留它在 path 下的子文件夹。只要有一个被拒，整批都不入队。
        options / subtitles 同 queue_encode，整批共用（subtitles.series_mode 为整批共用一张译名表）。
        """
        def run():
            return routes.enqueue_encode_batch(EncodeBatchRequest(
                path=path, files=files, options=_encode_options(options),
                output_mode=output_mode, output_dir=output_dir, subtitles=subtitles))

        return await _in_thread(run)

    @mcp.tool()
    async def analyze_restore(path: str) -> dict:
        """「修复」页的片源分析：从画面判断片源节奏（硬过带 24p / 30p 错场 / 真隔行 /
        混场）、黑边、画面比例是否缺失，并给出建议的反交错方式与裁边——即开压时
        「自动判断」会量出来的东西，入队前先看。"""
        return await _in_thread(routes.restore_analyze, path)

    @mcp.tool()
    async def start_restore_preview(source: str, options: Optional[EncodeOptions] = None) -> dict:
        """修复的「试看」：取片长 25/50/75% 处各 3 秒，用 Lanczos 与三个 AI 模型各跑一遍，
        拼成 2×2 对比视频与静帧图，并按实测速度估整部片要多久。后台运行，立即返回；
        用 get_restore_preview 跟进。同时只有一个试看，开新的会取消旧的。

        options: 修复参数（同 queue_encode，未填的取设置页默认值），必须设 upscale、
        且画面不能是 copy；要先在设置里配好修复引擎。和翻译、压制共用同一个重任务位置。
        试看从不替你选模型：看完后把选中的 ai_model 填进 queue_encode。
        """
        def run():
            try:
                preview = restore_preview.start(
                    Path(source.strip().strip('"').strip("'")).expanduser(),
                    _encode_options(options), load_settings().restore)
            except restore_preview.PreviewError as exc:
                return {"error": str(exc)}
            return _preview_view(preview)

        return await _in_thread(run)

    @mcp.tool()
    async def get_restore_preview() -> dict:
        """当前试看的状态：进度、量出来的片源参数（resolved）、每个方法的速度与
        闪烁（methods，含整部片的预计用时），完成后 file_paths 是对比视频与静帧图的本机路径。"""
        preview = restore_preview.current()
        return _preview_view(preview) if preview else {"status": "idle"}

    @mcp.tool()
    async def cancel_restore_preview() -> dict:
        """取消正在跑的试看。"""
        preview = restore_preview.cancel()
        return _preview_view(preview) if preview else {"status": "idle"}

    # -------------------------------------------------------------- 音频

    @mcp.tool()
    async def probe_audio(path: str) -> dict:
        """「音频」页读一个视频：时长、音轨列表、默认会取哪条（default_track），
        以及旁边是否已经有本程序提取过的音频（extracted）。"""
        return await _in_thread(routes.audio_probe, path)

    @mcp.tool()
    async def scan_audio_folder(path: str, recursive: bool = True) -> dict:
        """列出一个文件夹里的视频及其音轨（「音频」页的批量模式）。只列视频；
        extracted 非空的是已经提取过的。files 里的 path 就是 queue_audio_batch 的 files。"""
        return await _in_thread(routes.audio_scan, EncodeScanRequest(path=path, recursive=recursive))

    @mcp.tool()
    async def queue_audio(
        source: str,
        options: Optional[AudioOptions] = None,
        track: Optional[int] = None,
        language: str = "",
        output_mode: Literal["beside", "custom"] = "beside",
        output_dir: str = "",
    ) -> dict:
        """把一个视频的音轨提取**加入列队**：16 kHz 单声道，给翻译用（「音频」页）。

        options.format: flac（默认，无损——与直接翻视频时识别听到的逐采样相同）或
            opus（约六分之一大小，略有损失）；options.bitrate_kbps 只对 opus 有效，0 为默认 24k。
        track: 音轨容器序号（probe_audio 列出）；留空按 language（ISO 639，如 jpn）选，
            再不行用默认音轨。
        产物 片名.flac / 片名.opus（不带语言后缀，批量翻译时与视频算同一部片）；
        beside 放原文件旁边，custom 放 output_dir；绝不覆盖已有文件。
        """
        def run():
            return routes.enqueue_audio(AudioRequest(
                source=source, options=options or AudioOptions(), track=track,
                language=language, output_mode=output_mode, output_dir=output_dir))

        return await _in_thread(run)

    @mcp.tool()
    async def queue_audio_batch(
        path: str,
        files: list[str],
        options: Optional[AudioOptions] = None,
        language: str = "",
        output_mode: Literal["beside", "custom"] = "beside",
        output_dir: str = "",
    ) -> dict:
        """把一个文件夹里选中视频的音频提取**加入列队**（批量），每个文件一条、同一组。

        path: 扫描的文件夹（scan_audio_folder 的 path）；files: 视频完整路径。
        各文件的音轨编号不同，所以按 language 选；没有该语言的用默认音轨并在日志里说明。
        options / output_mode 同 queue_audio；custom 时保留子文件夹。
        """
        def run():
            return routes.enqueue_audio_batch(AudioBatchRequest(
                path=path, files=files, options=options or AudioOptions(),
                language=language, output_mode=output_mode, output_dir=output_dir))

        return await _in_thread(run)

    return mcp


class Mount:
    """The ASGI app parked at /mcp for the process's whole life.

    It holds no server of its own — each lifespan builds one and hands it
    over — so the FastAPI app object stays reusable across restarts of the
    lifespan while /mcp keeps a single stable mount point.
    """

    def __init__(self):
        self.inner = None

    async def __call__(self, scope, receive, send):
        if self.inner is None:
            body = b'{"detail":"MCP \\u670d\\u52a1\\u5c1a\\u672a\\u5c31\\u7eea"}'
            await send({
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return
        await self.inner(scope, receive, send)


mount = Mount()
