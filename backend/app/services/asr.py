"""faster-whisper wrapper with lazy model loading and progress callbacks."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from app.models.schemas import ASRSettings, NetworkSettings

# (start, end, text)
RawSegment = Tuple[float, float, str]

ProgressFn = Callable[[float], None]  # 0..1
LogFn = Callable[[str], None]

_model = None
_model_key: Optional[tuple] = None
_model_lock = threading.Lock()

# error signatures that indicate missing/broken CUDA runtime libraries
_CUDA_LIB_HINTS = ("cublas", "cudnn", "cuda", "cudart", "nvidia")

# friendly names for CT2-converted fine-tunes selectable in the UI,
# resolved to their HuggingFace repo ids
EXTRA_MODELS = {
    "kotoba-whisper-v2.0": "kotoba-tech/kotoba-whisper-v2.0-faster",
    "CrisperWhisper": "nyrahealth/faster_CrisperWhisper",
}


def resolve_model(model_size: str) -> str:
    """Map a UI model name to what faster-whisper expects (size or repo id)."""
    return EXTRA_MODELS.get(model_size, model_size)


def get_model_cache_dir() -> Optional[str]:
    """User-configured model storage dir, or None for the HF default cache."""
    from app.core import config  # lazy to avoid circular import

    d = config.load_settings().model_cache_dir.strip()
    return d or None


_dll_dirs_registered = False


def register_cuda_dll_dirs(log: Optional[LogFn] = None) -> None:
    """Windows: make pip-installed NVIDIA DLLs loadable by ctranslate2.

    `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` drops the DLLs into
    site-packages/nvidia/*/bin, which is NOT on the Windows DLL search path,
    so ctranslate2 fails with "cublas64_12.dll is not found". Register every
    such bin dir (add_dll_directory + PATH) before touching CUDA.
    """
    global _dll_dirs_registered
    if _dll_dirs_registered or sys.platform != "win32":
        return
    _dll_dirs_registered = True
    import site

    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for root in dict.fromkeys(roots):
        nvidia = Path(root) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in sorted(nvidia.glob("*/bin")):
            try:
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                if log:
                    log(f"已注册 CUDA DLL 目录: {bin_dir}")
            except OSError:
                continue


def _wrap_cuda_error(exc: Exception, settings: ASRSettings) -> Exception:
    """Map raw CUDA library errors to an actionable message."""
    message = str(exc)
    if settings.device in ("cuda", "auto") and any(
        hint in message.lower() for hint in _CUDA_LIB_HINTS
    ):
        return RuntimeError(
            "CUDA 运行库加载失败。请在 backend 目录执行 "
            '.venv\\Scripts\\pip install -e ".[gpu]"（Linux 为 .venv/bin/pip）'
            "安装 cuBLAS/cuDNN 后重启程序，程序会自动注册这些 DLL；"
            "若仍失败，可从 Purfview/whisper-standalone-win 的 Releases 下载 "
            "cuBLAS.and.cuDNN 压缩包，把 DLL 解压到 backend 目录或加入 PATH；"
            "或在设置中把设备切回 CPU。原始错误: " + message
        )
    return exc


@contextlib.contextmanager
def proxy_env(network: Optional[NetworkSettings]):
    """Route HuggingFace downloads through the proxy while inside the block.

    huggingface_hub's requests honour HTTP(S)_PROXY at request time; we set
    and restore them around download/load calls only, so the LLM traffic is
    unaffected (it has its own independent proxy switch).
    """
    if not (network and network.model_download_via_proxy and network.proxy_url.strip()):
        yield
        return
    proxy = network.proxy_url.strip()
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = proxy
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def is_local_model_dir(path: str) -> bool:
    """True if *path* looks like a CTranslate2 whisper model directory."""
    p = Path(path)
    return p.is_dir() and (p / "model.bin").is_file()


def is_model_cached(model_size: str) -> bool:
    """True if the model is already in the local HuggingFace cache."""
    from faster_whisper.utils import download_model

    try:
        download_model(
            resolve_model(model_size),
            local_files_only=True,
            cache_dir=get_model_cache_dir(),
        )
        return True
    except Exception:
        return False


def _get_model(
    settings: ASRSettings,
    log: Optional[LogFn] = None,
    network: Optional[NetworkSettings] = None,
):
    """Load (and cache) the WhisperModel; reload only when settings change.

    A non-empty settings.model_path takes priority and loads a local
    CTranslate2 directory (fully offline). Otherwise the model is downloaded
    only if not already in the local cache; a cached model is loaded offline
    (local_files_only=True).
    """
    global _model, _model_key
    use_path = settings.model_path.strip()
    key = (use_path, settings.model_size, settings.device, settings.compute_type)
    with _model_lock:
        if _model is None or _model_key != key:
            from faster_whisper import WhisperModel

            if use_path:
                if not is_local_model_dir(use_path):
                    raise RuntimeError(
                        f"本地模型目录无效: {use_path}"
                        "（需为 CTranslate2 格式的模型文件夹，至少包含 model.bin）"
                    )
                source, cached = use_path, True
                if log:
                    log(f"加载本地模型目录 {use_path} "
                        f"({settings.device}/{settings.compute_type})")
            else:
                source = resolve_model(settings.model_size)
                cached = is_model_cached(settings.model_size)
                if log:
                    if cached:
                        log(
                            f"加载本地已缓存的语音识别模型 {settings.model_size} "
                            f"({settings.device}/{settings.compute_type})"
                        )
                    else:
                        log(
                            f"本地未找到模型 {settings.model_size}，"
                            "开始从 HuggingFace 下载（仅首次需要，可能需要几分钟）…"
                        )
            if settings.device in ("cuda", "auto"):
                register_cuda_dll_dirs(log)
            try:
                with proxy_env(network if not cached else None):
                    _model = WhisperModel(
                        source,
                        device=settings.device,
                        compute_type=settings.compute_type,
                        local_files_only=cached,
                        download_root=None if use_path else get_model_cache_dir(),
                    )
            except Exception as exc:
                wrapped = _wrap_cuda_error(exc, settings)
                if wrapped is exc:
                    raise
                raise wrapped from exc
            _model_key = key
    return _model


def transcribe(
    wav_path: str,
    settings: ASRSettings,
    language: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    network: Optional[NetworkSettings] = None,
) -> Tuple[List[RawSegment], str]:
    """Transcribe *wav_path*; returns (segments, detected_language).

    *language* is a whisper language code, or None for auto-detection.
    """
    model = _get_model(settings, log, network)
    if log:
        log("模型就绪，开始识别（语言检测与首段解码可能需要等待一会儿）…")
    # CUDA libraries load lazily on the first encode (inside transcribe /
    # segment iteration), so the whole decode path needs the friendly wrap
    try:
        vad_parameters = None
        if settings.vad_filter:
            vad_parameters = dict(
                threshold=settings.vad_threshold,
                min_speech_duration_ms=settings.vad_min_speech_ms,
                min_silence_duration_ms=settings.vad_min_silence_ms,
                speech_pad_ms=settings.vad_speech_pad_ms,
            )
        segments_iter, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            vad_parameters=vad_parameters,
        )
        total = info.duration or 0.0
        if log:
            lang = language or f"{info.language} (置信度 {info.language_probability:.0%})"
            log(f"检测语言: {lang}，音频时长 {total:.0f}s")

        results: List[RawSegment] = []
        for seg in segments_iter:
            if should_cancel and should_cancel():
                raise InterruptedError("cancelled")
            text = seg.text.strip()
            if not text:
                continue
            results.append((float(seg.start), float(seg.end), text))
            if progress and total:
                progress(min(seg.end / total, 1.0))
            if log:
                log(f"[{seg.start:7.2f}s] {text}")
        return results, info.language
    except (InterruptedError, KeyboardInterrupt):
        raise
    except Exception as exc:
        wrapped = _wrap_cuda_error(exc, settings)
        if wrapped is exc:
            raise
        raise wrapped from exc
