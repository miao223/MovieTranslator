"""MCP server: drive a translation from another machine.

Mounted at /mcp over streamable HTTP, so any MCP client that can reach the
port — on the LAN, or through a tunnel — can queue films on the GPU box and
collect the subtitles. Off by default (`MCPSettings.enabled`); the access
guard answers 404 while it is.

The tools are the REST surface, minus anything that writes settings. A
client that could rewrite `asr.model_size` or the LLM key from a sentence
of prose is a much larger blast radius than one that can only start, watch
and cancel jobs — and the jobs are the point.

Nothing here reimplements the pipeline: every tool calls the same manager
the HTTP routes call, so job semantics (one CPU-heavy job at a time,
cancellation, logging) are shared by construction.

The dependency is imported defensively: an installation whose venv predates
this feature still starts, with `mcp` left as None and the reason reported
through /api/server/info.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anyio.to_thread

from app.core import joblog
from app.core.config import load_settings
from app.core.media import kind_of, scan_media
from app.models.schemas import BatchRequest, EmbedSettings, JobRequest
from app.services import audio, mux, subsource
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

    # ---------------------------------------------------------- discovery

    @mcp.tool()
    async def list_videos(
        directory: str, recursive: bool = True, skip_translated: bool = True
    ) -> dict:
        """列出某个目录下可翻译的视频与音频文件。

        directory: 服务器本机的目录绝对路径。
        recursive: 是否递归子目录。
        skip_translated: 跳过同名字幕已存在的文件。

        audio: 上面这批里的纯音频文件——它们只能产出字幕文件，
            不支持内嵌合成新视频，也没有画面翻译。
        shadowed: 与同名视频重名而被让位的音频文件（两者会写同一份字幕）。
        """
        found, skipped, shadowed, _with_source = await anyio.to_thread.run_sync(
            scan_media, directory, recursive, skip_translated
        )
        return {
            "videos": [str(v) for v in found],
            "skipped": [str(s) for s in skipped],
            "total": len(found),
            "audio": [str(f) for f in found if kind_of(f) == "audio"],
            "shadowed": [str(s) for s in shadowed],
        }

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
        原文——又快又准。text 为 false 的是图形字幕（PGS/VobSub），只有图片
        没有文字，不能用。
        """
        path = Path(video_path).expanduser()
        if not path.is_file():
            return {"error": f"文件不存在: {path}"}
        tracks = await anyio.to_thread.run_sync(subsource.all_tracks, path)
        return {"tracks": tracks}

    @mcp.tool()
    async def get_server_status() -> dict:
        """查询本机翻译服务的版本、识别配置与当前任务情况。不返回任何密钥。"""
        settings = load_settings()
        jobs = list(manager.jobs.values())
        running = [_job_view(j) for j in jobs if j.status.stage not in
                   ("done", "failed", "cancelled")]
        return {
            "version": joblog.APP_VERSION,
            "asr_engine": settings.asr.engine,
            "asr_model": (
                (settings.llm.audio_model or settings.llm.model)
                if settings.asr.engine == "api"
                else settings.asr.model_path.strip() or settings.asr.model_size
            ),
            "asr_device": settings.asr.device,
            "llm_model": settings.llm.model,
            "second_pass": settings.asr.second_pass,
            "refine_enabled": settings.prompts.refine_enabled,
            "mark_lyrics": settings.prompts.mark_lyrics,
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
        """为一个视频文件启动字幕翻译，立即返回 job_id，不等待完成。

        整个流程需要几分钟到几小时，请随后用 get_job 轮询进度，
        完成后用 get_subtitle 取回字幕。

        video_path: 服务器本机的视频或音频文件绝对路径。
        source_language: 影片原始语言代码（如 ja / en），auto 为自动判断。
        synopsis: 剧情简介，可显著提升人名与代词的翻译准确度。
        output_mode: bilingual 双语，translation_only 只要译文，
            original_only 只要原文——跳过 AI 翻译，但转写预处理、歌词识别、
            二次识别复核、图形字幕 OCR 与校对照常执行。产物带源语言后缀
            （片名.ja.srt，内嵌时 片名.ja.mkv，轨道标为该语言）。此时
            target_language 只作用于画面翻译，对白保持原文。
            bilingual_split 内容与 bilingual 相同，但写成两个单语文件
            （片名.zh.srt + 片名.ja.srt，**译文这一份也带语言后缀**）；
            内嵌时是一个视频里两条字幕轨，译文默认打开。原文那一份的路径在
            get_job 的 original_subtitle_path 里。
        audio_track: 音轨的容器序号，留空用默认音轨。
        audio_language: 按语言标签选音轨（如 jpn），仅在 audio_track 留空时生效。
        text_source: asr 走语音识别；subtitle 直接读片源已有的字幕，跳过识别——
            片源自带外文字幕时又快又准，先用 list_subtitle_tracks 看有哪些。
        subtitle_track: 要读取的字幕轨容器序号（text_source="subtitle" 时）。
            指向图形字幕（PGS/VobSub）会直接失败，因为那里面没有文字。
        subtitle_file: 改为读取这个外挂字幕文件（.srt/.ass），优先于 subtitle_track。
        embed_subtitle: 开启后不生成字幕文件，而是在同目录产出一个内嵌软字幕的
            新视频（片名.zh.mkv，纯原文时后缀是源语言；音视频不重编码）。会完整复制一份视频，注意磁盘空间。
            片源原有的字幕轨会保留。生成路径见 get_job 的 video_filename 字段。
            纯音频片源没有画面可合成，会自动改为生成字幕文件。
        container: 新视频的容器，mkv 或 mp4。mp4 兼容性最好，但只能带纯文本字幕轨，
            也装不下片源自带的字幕轨与字体附件。仅在 embed_subtitle 开启时有意义。
        video_codec: copy 表示原样拷贝不重编码（默认，唯一不损失画质的选项）。
            要重编码就填编码器 id，可用的用 get_server_info 查。重编码整部影片
            动辄数小时，画质只减不增。
        """
        if output_mode not in OUTPUT_MODES:
            return {"error": f"output_mode 只能是 {'、'.join(OUTPUT_MODES)}"}
        if text_source not in ("asr", "subtitle"):
            return {"error": "text_source 只能是 asr 或 subtitle"}
        # the encoder half is left to manager.create, which knows whether this
        # source has a picture at all — an audio file degrades rather than
        # being turned away over an encoder it will never reach
        bad = _container_error(container)
        if bad:
            return {"error": bad}
        request = JobRequest(
            video_path=video_path,
            audio_track=audio_track,
            audio_language=audio_language,
            text_source=text_source,  # type: ignore[arg-type]
            subtitle_track=subtitle_track,
            subtitle_file=subtitle_file,
            source_language=source_language,
            target_language=target_language,
            synopsis=synopsis,
            output_mode=output_mode,  # type: ignore[arg-type]
            embed_subtitle=embed_subtitle,
            embed=EmbedSettings(container=container, video_codec=video_codec),
        )
        try:
            job = manager.create(request)
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
        """为一个目录下的所有视频与音频文件批量启动翻译，立即返回 batch_id。

        影片逐个串行处理（语音识别占满 CPU/GPU，同时只跑一个）。
        用 get_batch 查看整体进度。

        series_mode（剧集模式）：整个目录是同一部剧的多集时开启，先译出的
        人名/术语译法会强制沿用到后续每一集；目录里是互不相干的影片则不要
        开启。累积的对照表可在 get_batch 的 glossary 字段查看。

        text_source="subtitle"：直接读每个视频已有的字幕，跳过语音识别。
        subtitle_language 按语言标签挑轨（如 eng；各文件的轨道序号不同）。
        找不到可读字幕的那个文件会自动改用语音识别，不影响整批。

        output_mode：同 translate_video，整批共用。original_only（纯原文）跳过
        AI 翻译，每个文件产出带源语言后缀的原文字幕（片名.ja.srt）；此时剧集模式
        不会积累新的译名表，因为没有译文。bilingual_split（双文件）每个文件产出
        译文、原文各一份，两份都带语言后缀。

        container / video_codec：同 translate_video，整批共用。
        embed_subtitle：同 translate_video——每个视频产出一个内嵌软字幕的新 mkv，
        不生成字幕文件。整季剧集会因此多占一整份磁盘空间。目录里的纯音频文件
        没有画面可合成，会各自改为生成字幕文件，不影响整批。
        """
        if output_mode not in OUTPUT_MODES:
            return {"error": f"output_mode 只能是 {'、'.join(OUTPUT_MODES)}"}
        if text_source not in ("asr", "subtitle"):
            return {"error": "text_source 只能是 asr 或 subtitle"}
        bad = _embed_error(container, video_codec)
        if bad:
            return {"error": bad}
        request = BatchRequest(
            directory=directory,
            recursive=recursive,
            skip_existing_srt=skip_translated,
            audio_language=audio_language,
            text_source=text_source,  # type: ignore[arg-type]
            subtitle_language=subtitle_language,
            source_language=source_language,
            target_language=target_language,
            synopsis=synopsis,
            output_mode=output_mode,  # type: ignore[arg-type]
            series_mode=series_mode,
            embed_subtitle=embed_subtitle,
            embed=EmbedSettings(container=container, video_codec=video_codec),
        )
        try:
            status = await anyio.to_thread.run_sync(batch_manager.create, request)
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
        """查询单个翻译任务的阶段与进度。"""
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
        """取消一个翻译任务。"""
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
    async def get_subtitle(job_id: str) -> dict:
        """取回已完成任务的字幕全文（SRT 或 ASS）。

        字幕文件本身也已经写在视频旁边，路径见 subtitle_path。
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
        text = await anyio.to_thread.run_sync(
            lambda: Path(job.srt_path).read_text(encoding="utf-8")
        )
        truncated = len(text) > MAX_SUBTITLE_CHARS
        return {
            "subtitle_path": job.status.srt_filename,
            "content": text[:MAX_SUBTITLE_CHARS],
            "truncated": truncated,
            "total_chars": len(text),
        }

    @mcp.tool()
    async def get_job_log(job_id: str, tail_lines: int = 200) -> dict:
        """取回任务日志的末尾若干行，用于排查失败原因。日志不含 API key。"""
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
