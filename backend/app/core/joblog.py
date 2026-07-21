"""Per-job log files, written for post-mortem support.

The progress log in the browser is capped and disappears with the page, so
a failure a user wants help with is usually already gone. These files sit
next to the job cache but are NOT wiped on startup, carry a diagnostics
header (versions, media probe, effective settings), and can be downloaded
from the UI in one click.

Nothing secret is ever written: the API key is recorded as configured or
not, never its value.
"""

from __future__ import annotations

import platform
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.cache import _base_dir

APP_VERSION = "0.7.1"
LOG_DIR_NAME = "logs"
KEEP_LOGS = 20  # newest job logs to retain


def logs_dir() -> Path:
    d = _base_dir() / LOG_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def prune_logs(keep: int = KEEP_LOGS) -> None:
    try:
        files = sorted(
            logs_dir().glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for old in files[keep:]:
            old.unlink(missing_ok=True)
    except OSError:
        pass  # housekeeping must never break a job


def find_log(job_id: str) -> Optional[Path]:
    for path in logs_dir().glob(f"*_{job_id}.log"):
        return path
    return None


def _versions() -> list[str]:
    lines = [
        f"系统          : {platform.platform()} ({sys.platform})",
        f"Python        : {platform.python_version()}",
    ]
    try:
        import av

        lines.append(f"PyAV          : {av.__version__}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"PyAV          : 不可用 ({exc})")
    try:
        from importlib.metadata import version

        lines.append(f"faster-whisper: {version('faster-whisper')}")
    except Exception:  # noqa: BLE001
        lines.append("faster-whisper: 未安装")
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        lines.append(f"CUDA          : {f'可用，{count} 个设备' if count else '不可用'}")
    except Exception as exc:  # noqa: BLE001 — missing CUDA libs land here
        lines.append(f"CUDA          : 不可用 ({exc})")
    return lines


def _settings_lines(settings) -> list[str]:
    asr, llm, sub, net = settings.asr, settings.llm, settings.subtitle, settings.network
    model_src = asr.model_path.strip() or asr.model_size
    return [
        f"识别模型      : {model_src} ({asr.device}/{asr.compute_type}) "
        f"beam={asr.beam_size} 词级时间戳={'开' if asr.word_timestamps else '关'}",
        f"VAD           : {'开' if asr.vad_filter else '关'} 阈值={asr.vad_threshold} "
        f"填充={asr.vad_speech_pad_ms}ms 最短语音={asr.vad_min_speech_ms}ms "
        f"最短静默={asr.vad_min_silence_ms}ms",
        f"翻译模型      : {llm.model} @ {llm.base_url} "
        f"(API key: {'已配置' if llm.api_key.strip() else '未配置'}) "
        f"temp={llm.temperature} 每批={llm.batch_size} 上下文={llm.context_limit}",
        f"视觉模型      : {llm.vision_model or '（同主模型）'}",
        f"转写预处理    : {'开' if settings.prompts.refine_enabled else '关'}",
        f"字幕          : 每行{sub.max_chars_per_line}字 单条≤{sub.max_duration}s "
        f"{'样式(.ass)' if sub.style_enabled else '标准(.srt)'} {sub.bilingual_layout}",
        f"代理          : {net.proxy_url or '（未设置）'} "
        f"LLM={'走' if net.llm_via_proxy else '不走'} "
        f"模型下载={'走' if net.model_download_via_proxy else '不走'}",
    ]


class JobLogWriter:
    """Appends every published progress line to a file, with a header."""

    def __init__(self, job_id: str, video_path: str):
        self.path: Optional[Path] = None
        self._lock = threading.Lock()
        try:
            prune_logs()
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = logs_dir() / f"{stamp}_{job_id}.log"
            self._raw(f"=== MovieTranslator {APP_VERSION} 任务日志 ===")
            self._raw(f"任务 ID       : {job_id}")
            self._raw(f"开始时间      : {datetime.now():%Y-%m-%d %H:%M:%S}")
            self._raw(f"视频文件      : {video_path}")
        except OSError:
            self.path = None  # logging must never break a job

    # ------------------------------------------------------------ writing

    def _raw(self, text: str) -> None:
        if not self.path:
            return
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:
            self.path = None

    def section(self, title: str, lines: list[str]) -> None:
        self._raw("")
        self._raw(f"--- {title} ---")
        for line in lines:
            self._raw(line)

    def event(self, stage: str, progress: float, message: str, log: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        for text in (message, log):
            if text:
                for line in str(text).splitlines():
                    self._raw(f"[{stamp}] [{stage:<12} {progress:5.1f}%] {line}")

    # ------------------------------------------------------------ headers

    def write_environment(self) -> None:
        self.section("运行环境", _versions())

    def write_settings(self, settings) -> None:
        try:
            self.section("设置", _settings_lines(settings))
        except Exception as exc:  # noqa: BLE001
            self.section("设置", [f"（读取失败: {exc}）"])

    def write_media(self, video_path: str) -> None:
        from app.services.audio import describe_media

        try:
            self.section("媒体信息", describe_media(video_path))
        except Exception as exc:  # noqa: BLE001
            self.section("媒体信息", [f"（探测失败: {exc}）"])

    def write_request(self, request) -> None:
        self.section("任务参数", [
            f"源语言        : {request.source_language}",
            f"目标语言      : {request.target_language}",
            f"字幕形式      : {request.output_mode}",
            f"指定音轨      : {request.audio_track if request.audio_track is not None else '（自动）'}"
            + (f" 语言偏好={request.audio_language}" if request.audio_language else ""),
            f"画面翻译      : {len(request.frame_tasks)} 条"
            + ("（仅补充模式）" if request.frame_only else ""),
            f"剧情简介      : {'已填写 ' + str(len(request.synopsis)) + ' 字' if request.synopsis.strip() else '（无）'}",
        ])
