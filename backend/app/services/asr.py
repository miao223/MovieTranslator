"""faster-whisper wrapper with lazy model loading and progress callbacks."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from app.models.schemas import ASRSettings, NetworkSettings


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)  # empty without word_timestamps
    recovered: bool = False  # came from the second pass, not the VAD-filtered one

ProgressFn = Callable[[float], None]  # 0..1
LogFn = Callable[[str], None]

# The coverage figure must stay comparable between runs, so its denominator
# is measured at a FIXED threshold rather than whatever the job used. With
# the user's setting, lowering the threshold inflated Silero's "speech"
# total by 334s of noise and the ratio fell from 77% to 63% — while the
# transcription had in fact improved by 60s.
REFERENCE_VAD_THRESHOLD = 0.35
# an uncovered run shorter than this is a pause, not a missing line
MIN_MISS_SECONDS = 2.0
# ...and outside the speech intervals, only report a stretch this long
MIN_UNDETECTED_SECONDS = 5.0
# how close to the level of known speech a stretch has to be before
# "Silero heard nothing there" becomes suspicious rather than expected
LEVEL_SUSPICIOUS_DB = 12.0
# Second pass (see second_pass): only revisit a blank at least this long —
# shorter ones are pauses, and a slice of a few seconds gives the decoder
# nothing to work with.
SECOND_PASS_MIN_BLANK = 15.0
SECOND_PASS_WINDOW = 300.0  # bounded slices; see _windows()
# gates for what the second pass is allowed to keep. no_speech_prob is
# deliberately absent — music inflates it on genuine dialogue.
SECOND_PASS_MAX_COMPRESSION = 2.4   # whisper's own repetition tell
SECOND_PASS_MIN_LOGPROB = -1.0      # whisper's own confidence floor

# Whisper was trained on subtitle files, and over non-speech it emits what
# those files contain: closing lines, station announcements, encyclopedia
# entries, cooking instructions. The second pass looks precisely where the
# VAD found no speech, so it meets them constantly — one film came back
# with "ご視聴ありがとうございました" twelve times at compression ratios of
# 0.86, perfectly ordinary sentences that no confidence gate can see
# through. Sorting those from the genuine dialogue the pass recovers takes
# knowledge of what the film is about, so it happens afterwards and
# elsewhere: see services/vet.py.

_model = None
_model_key: Optional[tuple] = None
_model_lock = threading.Lock()

# error signatures that indicate missing/broken CUDA runtime libraries
_CUDA_LIB_HINTS = ("cublas", "cudnn", "cuda", "cudart", "nvidia")

# friendly names for CT2-converted fine-tunes selectable in the UI,
# resolved to their HuggingFace repo ids. kotoba was removed from the UI
# (poor real-world results) but stays resolvable so saved settings keep
# working for users who already downloaded it.
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


def _blank_regions(
    segments: Sequence[Segment], duration: float, min_blank: float
) -> List[tuple[float, float]]:
    """Stretches of the timeline the first pass produced no WORDS for.

    Word positions, not segment spans: whisper's segment boundaries are
    routinely stretched across silence it never transcribed — one segment
    on a real film spanned 577 seconds while holding fifteen words, all of
    them at the far end. Measured by spans, that 9.6-minute hole counted as
    covered and the second pass never looked at it, which is exactly where
    the missing dialogue was. Measured by words, the same film's blanks go
    from 3563s to 5606s.
    """
    spans = sorted(
        (w.start, w.end) for seg in segments for w in seg.words
    ) or sorted((s.start, s.end) for s in segments)
    out: List[tuple[float, float]] = []
    cursor = 0.0
    for start, end in spans:
        if start - cursor >= min_blank:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= min_blank:
        out.append((cursor, duration))
    return out


def _windows(regions: Sequence[tuple[float, float]], size: float):
    """Split regions into slices no longer than *size*.

    Bounded slices are what makes a VAD-free pass safe: transcribing two
    hours with the VAD off collapsed into one sentence repeating, because
    each window primes the next. A few minutes at a time, with the priming
    turned off, cannot run away like that.
    """
    for start, end in regions:
        cursor = start
        while cursor < end:
            yield cursor, min(cursor + size, end)
            cursor += size


def second_pass(
    model,
    audio,
    segments: List[Segment],
    settings: ASRSettings,
    language: Optional[str],
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    debug=None,
) -> List[Segment]:
    """Re-transcribe what the VAD threw away, and merge what survives.

    Silero is a small model trained on clean speech; dialogue mixed under a
    music bed scores below any workable threshold, and lowering it only
    admits more music. Whisper's own encoder has no such trouble — on the
    film that prompted this, a stretch Silero called silent for nine and a
    half minutes transcribed into 35 lines of ordinary conversation once it
    was handed over directly.
    """
    duration = len(audio) / 16000.0
    regions = _blank_regions(segments, duration, SECOND_PASS_MIN_BLANK)
    if not regions:
        return segments
    todo = list(_windows(regions, SECOND_PASS_WINDOW))
    total = sum(e - s for s, e in todo)
    if log:
        log(f"二次识别：对 VAD 判定无语音的 {len(regions)} 段（共 {total:.0f}s）"
            "关闭 VAD 重新识别…")
    if debug is not None and debug.enabled:
        debug.section("二次识别（对 VAD 丢弃的区间关 VAD 重跑）")
        debug.kv("空白区间", f"{len(regions)} 段，切成 {len(todo)} 个窗口")
        debug.kv("重新识别时长", f"{total:.0f}s")

    recovered: List[Segment] = []
    dropped = 0
    for start, end in todo:
        if should_cancel and should_cancel():
            raise InterruptedError("cancelled")
        chunk = audio[int(start * 16000):int(end * 16000)]
        if len(chunk) < 16000:
            continue
        try:
            found, _info = model.transcribe(
                chunk,
                language=language,
                beam_size=settings.beam_size,
                word_timestamps=settings.word_timestamps,
                vad_filter=False,
                # each window must stand alone: priming from the previous one
                # is exactly what turns a quiet stretch into a repeat loop
                condition_on_previous_text=False,
                initial_prompt=settings.initial_prompt.strip() or None,
            )
            for seg in found:
                text = seg.text.strip()
                if not text:
                    continue
                # A music bed pushes no_speech_prob up even where the
                # dialogue is perfectly clear (0.85 on lines that turned out
                # to be ordinary conversation), so it cannot be a gate here.
                # Repetition and decoder confidence can.
                if getattr(seg, "compression_ratio", 0) > SECOND_PASS_MAX_COMPRESSION:
                    dropped += 1
                    continue
                if getattr(seg, "avg_logprob", 0) < SECOND_PASS_MIN_LOGPROB:
                    dropped += 1
                    continue
                words = [
                    Word(float(w.start) + start, float(w.end) + start, w.word)
                    for w in (seg.words or [])
                ]
                recovered.append(Segment(
                    float(seg.start) + start, float(seg.end) + start,
                    text, words, recovered=True,
                ))
                if debug is not None and debug.enabled:
                    _debug_segment(debug, seg, words, offset=start)
        except (InterruptedError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 — a bad window must not kill the job
            if log:
                log(f"⚠ 二次识别 {start:.0f}-{end:.0f}s 失败（跳过）: {exc}")

    # never overwrite what the first pass already covered
    kept = [s for s in recovered if not _overlaps_any(s, segments)]
    merged = sorted(segments + kept, key=lambda s: s.start)
    if log:
        log(f"二次识别完成：找回 {len(kept)} 段 / "
            f"{sum(s.end - s.start for s in kept):.0f}s"
            + (f"，丢弃 {dropped} 段可疑输出" if dropped else "")
            + "（待复核）")
    if debug is not None and debug.enabled:
        debug.kv("采纳", f"{len(kept)} 段 / {sum(s.end - s.start for s in kept):.0f}s")
        debug.kv("丢弃（重复或置信度过低）", dropped)
        debug.kv("因与第一遍重叠而丢弃", len(recovered) - len(kept))
    return merged


def _overlaps_any(seg: Segment, existing: Sequence[Segment]) -> bool:
    return any(
        min(seg.end, other.end) - max(seg.start, other.start) > 0.2
        for other in existing
    )


def _report_coverage(
    audio,
    settings: ASRSettings,
    segments: List[Segment],
    log: Optional[LogFn],
    debug,
) -> None:
    """How much of the speech actually made it into text.

    Purely diagnostic, and never allowed to fail a job: a two-hour film had
    15-20% of its speech silently absent, and three runs with different VAD
    settings each lost a *different* 15-20% — which no amount of parameter
    tuning fixes and nothing in the output revealed.
    """
    try:
        import time

        started = time.monotonic()
        duration = len(audio) / 16000.0
        intervals = speech_intervals_of(audio, settings)
        levels = level_profile(audio)
        if not intervals:
            return
        speech, covered, misses = coverage_report(intervals, segments)
        share = covered / speech if speech else 1.0
        elapsed = time.monotonic() - started

        # what does speech look like, level-wise, where we know there is some?
        speech_db = _median_level(
            levels, intervals[0][0], intervals[0][1]
        ) if levels else -140.0
        if levels:
            import statistics

            spoken = [
                _median_level(levels, s, e) for s, e in intervals if e - s >= 2.0
            ]
            if spoken:
                speech_db = statistics.median(spoken)

        # stretches Silero never called speech, but which are as loud as the
        # speech it did find — the case a threshold cannot reach
        undetected = []
        for s, e in _outside_intervals(intervals, duration, MIN_UNDETECTED_SECONDS):
            db = _median_level(levels, s, e) if levels else -140.0
            if db >= speech_db - LEVEL_SUSPICIOUS_DB:
                undetected.append((s, e, e - s, db))

        summary = (
            f"识别覆盖率 {share:.0%}（语音 {speech:.0f}s，转写 {covered:.0f}s，"
            f"漏识别 {len(misses)} 处 / {sum(d for _, _, d in misses):.0f}s）"
        )
        # `undetected` deliberately stays out of this line: on a film that is
        # 59% score and ambience, "as loud as speech" flags every music cue,
        # and a warning nobody can act on is worse than none. It is a lead to
        # follow in the debug log, not a verdict.
        if log:
            log(summary if share >= 0.9 else "⚠ " + summary)
        if debug is not None and debug.enabled:
            from app.core.debuglog import fmt_time, percentiles

            debug.section("语音检测覆盖（silero VAD vs 实际转写）")
            debug.line(
                f"分母固定用阈值 {REFERENCE_VAD_THRESHOLD} 测量，与本次任务的设置无关，\n"
                "否则调低阈值会同时抬高分母，覆盖率反而下降，跨配置无法比较。\n"
                f"区间内短于 {MIN_MISS_SECONDS}s 的空缺算正常停顿，不计入漏识别。\n"
            )
            debug.kv("silero 语音区间", f"{len(intervals)} 段，合计 {speech:.0f}s")
            debug.kv("实际转写覆盖", f"{covered:.0f}s")
            debug.kv("覆盖率", f"{share:.0%}")
            debug.kv("区间时长(s)", percentiles([e - s for s, e in intervals]))
            debug.kv("已知人声电平中位", f"{speech_db:.1f} dBFS")
            found = [g for g in segments if g.recovered]
            if found:
                debug.kv(
                    "其中来自二次识别",
                    f"{len(found)} 段 / {sum(g.end - g.start for g in found):.0f}s",
                )
            debug.kv("本次检测耗时", f"{elapsed:.1f}s")

            debug.line(
                f"\n① 有语音但没转写出来（{len(misses)} 处，"
                f"≥{MIN_MISS_SECONDS}s，按时长排序）："
            )
            debug.lines(
                f"  {fmt_time(s)} → {fmt_time(e)}  {d:6.1f}s  "
                f"{_median_level(levels, s, e):6.1f} dBFS"
                for s, e, d in sorted(misses, key=lambda m: -m[2])
            )
            debug.line(
                f"\n② 音量接近人声、却未被判定为语音（{len(undetected)} 处，"
                f"≥{MIN_UNDETECTED_SECONDS}s）——"
                "调阈值够不着的漏检就在这里："
            )
            debug.line(
                "  注意：配乐/环境音同样能达到人声响度，音乐多的片源这里会有很多条，"
                "不能直接当作漏识别，需结合时间点回看确认。"
            )
            debug.lines(
                f"  {fmt_time(s)} → {fmt_time(e)}  {d:6.1f}s  {db:6.1f} dBFS  "
                f"（比人声{'高' if db >= speech_db else '低'} "
                f"{abs(db - speech_db):.1f} dB）"
                for s, e, d, db in sorted(undetected, key=lambda m: -m[2])[:60]
            )
            debug.line("\nsilero 全部语音区间：")
            debug.lines(
                f"  {fmt_time(s)} → {fmt_time(e)}  {e - s:6.2f}s  "
                f"{_median_level(levels, s, e):6.1f} dBFS"
                for s, e in intervals
            )
            debug.line("\n每分钟音量剖面（dBFS，用于判断某段是否真的有声音）：")
            debug.lines(
                f"  {i:3d}min  {_median_level(levels, i * 60, (i + 1) * 60):6.1f}"
                for i in range(int(duration // 60) + 1)
            )
    except Exception as exc:  # noqa: BLE001 — diagnostics must never fail a job
        if log:
            log(f"（识别覆盖率统计未能完成: {exc}）")


def speech_intervals(wav_path: str, settings: ASRSettings) -> List[tuple[float, float]]:
    """Silero's own verdict on where the speech is, in seconds.

    faster-whisper runs this internally when vad_filter is on, but only
    reports the total. Having the intervals is what turns "it feels like
    some lines are missing" into a number: any interval that comes back
    with no transcribed words is a measured miss, not a guess.

    Padding is deliberately dropped (speech_pad_ms=0) — it only exists to
    give the decoder some run-up, and counting it here would inflate the
    speech total by 0.8s per interval and make the comparison meaningless.
    """
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(wav_path, sampling_rate=16000)
    return speech_intervals_of(audio, settings)


def speech_intervals_of(audio, settings: ASRSettings) -> List[tuple[float, float]]:
    """As above, for audio already decoded."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=REFERENCE_VAD_THRESHOLD,
        min_speech_duration_ms=settings.vad_min_speech_ms,
        min_silence_duration_ms=settings.vad_min_silence_ms,
        speech_pad_ms=0,
    )
    return [
        (chunk["start"] / 16000.0, chunk["end"] / 16000.0)
        for chunk in get_speech_timestamps(audio, options)
    ]


def _uncovered_within(
    intervals: Sequence[tuple[float, float]],
    words: Sequence[tuple[float, float]],
    min_gap: float,
) -> List[tuple[float, float]]:
    """Stretches inside a speech interval where no word landed.

    Only runs above *min_gap* count. Silero keeps pauses shorter than
    min_silence_duration_ms inside one interval, so short uncovered runs
    are breathing room, not missing dialogue — on one film they were 341
    of the 349 uncovered runs and 90% of the uncovered seconds.
    """
    out: List[tuple[float, float]] = []
    for start, end in intervals:
        inside = sorted(
            (max(start, ws), min(end, we)) for ws, we in words
            if we > start and ws < end
        )
        cursor = start
        for ws, we in inside:
            if ws - cursor >= min_gap:
                out.append((cursor, ws))
            cursor = max(cursor, we)
        if end - cursor >= min_gap:
            out.append((cursor, end))
    return out


def _outside_intervals(
    intervals: Sequence[tuple[float, float]], duration: float, min_gap: float
) -> List[tuple[float, float]]:
    """Stretches Silero did not call speech at all."""
    out: List[tuple[float, float]] = []
    cursor = 0.0
    for start, end in intervals:
        if start - cursor >= min_gap:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= min_gap:
        out.append((cursor, duration))
    return out


def level_profile(audio) -> "list":
    """RMS in dBFS for every second of *audio* (16 kHz mono float32)."""
    import numpy as np

    usable = len(audio) - len(audio) % 16000
    if usable <= 0:
        return []
    frames = np.asarray(audio[:usable], dtype=np.float32).reshape(-1, 16000)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return (20.0 * np.log10(np.maximum(rms, 1e-7))).tolist()


def _median_level(levels: Sequence[float], start: float, end: float) -> float:
    import statistics

    window = levels[int(start):max(int(end), int(start) + 1)]
    return statistics.median(window) if window else -140.0


def coverage_report(
    intervals: List[tuple[float, float]], segments: List[Segment]
) -> tuple[float, float, List[tuple[float, float, float]]]:
    """(speech total, transcribed total, misses) — all in seconds.

    A "miss" is a run of at least MIN_MISS_SECONDS inside a speech interval
    that received no words.
    """
    words = sorted(
        (w.start, w.end) for seg in segments for w in seg.words
    ) or sorted((s.start, s.end) for s in segments)

    speech = sum(e - s for s, e in intervals)
    covered = sum(
        max(0.0, min(end, we) - max(start, ws))
        for start, end in intervals
        for ws, we in words
        if we > start and ws < end
    )
    misses = [
        (s, e, e - s)
        for s, e in _uncovered_within(intervals, words, MIN_MISS_SECONDS)
    ]
    return speech, covered, misses


def _debug_segment(dbg, seg, words: List[Word], offset: float = 0.0) -> None:
    """Record one whisper segment verbatim, words and confidences included.

    The word timestamps are the point: a several-second gap reported in the
    middle of a single word is what makes the segmenter break there, and
    nothing downstream can tell that apart from a real pause afterwards.
    """
    dbg.line(
        f"\n[{seg.start + offset:8.2f} → {seg.end + offset:8.2f}] "
        f"logprob={getattr(seg, 'avg_logprob', float('nan')):.2f} "
        f"no_speech={getattr(seg, 'no_speech_prob', float('nan')):.2f} "
        f"compress={getattr(seg, 'compression_ratio', float('nan')):.2f}"
        f"\n    {seg.text.strip()}"
    )
    if not words:
        return
    parts = []
    for i, w in enumerate(words):
        gap = w.start - words[i - 1].end if i else 0.0
        flag = f" ⟨间隔{gap:.1f}s⟩" if gap > 1.0 else ""
        parts.append(f"{w.start:.2f}-{w.end:.2f}{w.text}{flag}")
    dbg.line("    词: " + "  ".join(parts))


def transcribe(
    wav_path: str,
    settings: ASRSettings,
    language: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    network: Optional[NetworkSettings] = None,
    debug=None,
) -> tuple[List[Segment], str]:
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
            word_timestamps=settings.word_timestamps,
            vad_filter=settings.vad_filter,
            vad_parameters=vad_parameters,
            initial_prompt=settings.initial_prompt.strip() or None,
        )
        total = info.duration or 0.0
        after_vad = getattr(info, "duration_after_vad", None)
        if log:
            lang = language or f"{info.language} (置信度 {info.language_probability:.0%})"
            log(f"检测语言: {lang}，音频时长 {total:.0f}s")
            if after_vad and total:
                log(f"VAD 保留语音 {after_vad:.0f}s（占音频 {after_vad / total:.0%}）")

        dbg = debug if debug is not None and debug.enabled else None
        if dbg:
            dbg.section("语音识别原始输出（faster-whisper）")
            dbg.kv("语言", f"{info.language} ({info.language_probability:.0%})")
            dbg.kv("音频时长", f"{total:.1f}s")
            dbg.kv("VAD 后语音时长", f"{after_vad:.1f}s" if after_vad else "（未提供）")
            dbg.kv("initial_prompt", settings.initial_prompt.strip() or "（未设置）")
            dbg.line(
                "\n每个 segment：起止、平均对数概率、无语音概率、压缩比"
                "（压缩比高或 no_speech 高 = 可疑/幻觉），随后是词级时间戳。"
                "\n词级时间戳里出现的异常大间隔，正是字幕被切成半个词的直接原因。\n"
            )

        results: List[Segment] = []
        for seg in segments_iter:
            if should_cancel and should_cancel():
                raise InterruptedError("cancelled")
            text = seg.text.strip()
            if not text:
                continue
            words = [
                Word(float(w.start), float(w.end), w.word)
                for w in (seg.words or [])
            ]
            results.append(Segment(float(seg.start), float(seg.end), text, words))
            if progress and total:
                progress(min(seg.end / total, 1.0))
            if log:
                log(f"[{seg.start:7.2f}s] {text}")
            if dbg:
                _debug_segment(dbg, seg, words)

        # one decode serves both the second pass and the coverage report
        audio = None
        try:
            from faster_whisper.audio import decode_audio

            audio = decode_audio(wav_path, sampling_rate=16000)
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"（音频复核未能加载: {exc}）")
        if audio is not None and settings.second_pass and settings.vad_filter:
            results = second_pass(
                model, audio, results, settings,
                language or info.language, log, should_cancel, debug,
            )
        if audio is not None:
            _report_coverage(audio, settings, results, log, debug)
        return results, info.language
    except (InterruptedError, KeyboardInterrupt):
        raise
    except Exception as exc:
        wrapped = _wrap_cuda_error(exc, settings)
        if wrapped is exc:
            raise
        raise wrapped from exc
