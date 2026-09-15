"""faster-whisper wrapper with lazy model loading and progress callbacks."""

from __future__ import annotations

import contextlib
import os
import re
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
# Silence between two words longer than this is not a pause inside an
# utterance — it is a misplaced timestamp. Same value the segmenter breaks
# lines at (segmenter.GAP_BREAK), so both agree on what a real gap is.
COVERED_GAP_BRIDGE = 1.5

# When silero stops working at all, five numbers derived from it go wrong
# together and none of them says so. On a VHS capture it called 199s of
# 5760 speech (3.5%) and the job log still reported 93% coverage, a
# chars-per-speech-second of 40, a correlation of +0.61 and a clock drift
# of -266 ms/s — every one of them measured against a yardstick that had
# collapsed.
#
# Two independent signals have to agree before the diagnostics are called
# blind, because either alone is wrong on a real film: a film that is
# mostly score has a low `vad_share` while silero still hears its dialogue
# perfectly, and a transcript can sit outside the VAD for the good reason
# that this pipeline transcribes under music.
#
# Measured on ten films — three that set this number and seven held out,
# both languages, every source grade:
#
#   source                       vad_share   transcript_in_vad   blind
#   VHS capture, badly degraded      0.04           0.09          yes
#   DVD (ja)                         0.21           0.63          no   *
#   VHS transfer (ja)                0.28           0.60          no   *
#   VHS transfer (ja)                0.30           0.62          no   *
#   VHSRip (en)                      0.30           0.66          no   *
#   Slipstream, Blu-ray (en)         0.33           0.65          no   *
#   DVD (ja)                         0.36           0.63          no
#   DVD (ja)                         0.39           0.56          no   *
#   DVDRip (en)                      0.42           0.82          no   *
#   Blu-ray (en)                     0.71           0.89          no
#                                                        (* = held out)
#
# Two things the held-out films settled. **The medium is not the
# predictor**: three ordinary VHS transfers sit at 0.28-0.30, one of them
# English, indistinguishable from the DVDs — the single blind source is a
# capture degraded until its dialogue stutters. **And `vad_share` alone is
# not the test**: an ordinary DVD measures 0.21, under the threshold. What
# separates is the second signal — 0.09 for the blind film against a worst
# sighted case of 0.56, a 6x gap where vad_share offers none at all. The
# AND is not a safety net here; it is the decision.
VAD_BLIND_BELOW = 0.25
VAD_BLIND_MIN_LOUD = 60.0  # under a minute of loud audio is not evidence

# A letter, a digit or a CJK/kana/hangul character — anything a viewer
# could read. `\w` minus the underscore covers every script.
_CONTENT_RE = re.compile(r"[^\W_]", re.UNICODE)


def has_content(text: str) -> bool:
    """Does *text* contain a single readable character?

    Whisper can decode a whole film into one meaningless symbol. On an 85
    minute flv it returned 577 segments that were all exactly "-": no
    dialogue at all, yet every quality gate waved them through — a lone
    dash compresses to 0.33 and decodes at logprob -0.15, so the
    repetition and confidence thresholds see a perfectly good transcript.
    The damage was not just 224 dashes on screen (58% of the finished
    cues, 577 seconds of screen time) and the tokens spent translating
    them: those segments also count as transcribed, so the second pass
    considered those stretches covered and never revisited them, and the
    LLM review in vet.py was asked to judge recovered lines against a
    "confirmed" transcript consisting entirely of dashes.

    Dropping these is a local, deterministic rule — a cue with nothing to
    read is not a cue — and stays within the same authority as the
    `if not text` check next to it. It is not the LLM-side deletion that
    vet.py is deliberately barred from applying to the first pass.
    """
    return bool(_CONTENT_RE.search(text))


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


def _word_spans(segments: Sequence[Segment]) -> List[tuple[float, float]]:
    """Where each word sits — or each segment, without word timestamps.

    The single source of truth for "the first pass has text here". Segment
    spans cannot answer that: whisper routinely pins a word it failed to
    align to the end of the previous utterance, leaving a segment that
    claims minutes of timeline it never transcribed.
    """
    return sorted(
        (w.start, w.end) for seg in segments for w in seg.words
    ) or sorted((s.start, s.end) for s in segments)


def _covered_intervals(segments: Sequence[Segment]) -> List[tuple[float, float]]:
    """The timeline the first pass really put words on, gaps bridged.

    Silence up to ``COVERED_GAP_BRIDGE`` between two words is a pause
    inside an utterance: nothing is missing there even though no word
    occupies the instant. Anything longer is a misplaced timestamp, and
    what lies in it is genuinely untranscribed.
    """
    merged: List[tuple[float, float]] = []
    for start, end in _word_spans(segments):
        if merged and start - merged[-1][1] <= COVERED_GAP_BRIDGE:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


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
    spans = _word_spans(segments)
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


# A sentence or two of the spoken language, written the way a subtitle is:
# capitalised, punctuated, one sentence ending and another beginning.
# Whisper conditions its output on `initial_prompt`, so the shape of the
# prompt is the shape it tends to produce — and two Japanese films came
# back with 72% and 57% of their lines not ending a sentence, which is what
# tips the segmenter onto its "merge nothing, let refine do it" path.
#
# A language with no exemplar gets none: the prompt must be in the spoken
# language, or the whole transcript drifts toward the prompt's language.
STYLE_EXEMPLARS = {
    "en": "Hello, how are you? I'm fine, thank you.",
    "ja": "こんにちは、お元気ですか。はい、おかげさまで元気です。",
    "zh": "你好，最近怎么样？我很好，谢谢。",
    "ko": "안녕하세요, 잘 지내셨어요? 네, 잘 지냈습니다.",
    "es": "Hola, ¿cómo estás? Estoy bien, gracias.",
    "fr": "Bonjour, comment allez-vous ? Je vais bien, merci.",
    "de": "Hallo, wie geht es dir? Mir geht es gut, danke.",
    "it": "Ciao, come stai? Sto bene, grazie.",
    "pt": "Olá, como vai? Estou bem, obrigado.",
    "ru": "Здравствуйте, как дела? Хорошо, спасибо.",
}
# faster-whisper's own floor for trusting a language detection.
STYLE_DETECT_MIN_PROBABILITY = 0.5
# ...but it has to be asked the question properly first. `detect_language`
# defaults to `language_detection_segments=1` and `vad_filter=False`: one
# 30-second window from the very start of the file, silence and all. On a
# film that window is the distributor logo, the titles and the score, so
# the detector is being shown the one stretch with no speech in it.
# Measured on three films, default call vs this one:
#
#   可愛い悪魔（VHS 日语）   ja 20%  →  ja 93%
#   K9（BD 英语）           ru 49%  →  en 99%
#   午前零時（DVD 日语）     ja 75%  →  ja 93%
#
# The middle row is the one to read twice: asked its default way the
# detector called an English film **Russian**, and the 0.5 floor turned it
# down by a single point. Had the floor been 0.45 this feature would have
# prepended a Russian sentence to an English transcript — the one thing
# its red line forbids. The floor was doing real work, but it was the only
# thing standing there; with the VAD on and several windows sampled the
# detector is shown speech instead of the distributor logo, and is both
# sure and right. Two arms that had the exemplar switched on got no
# exemplar at all because of this, which is how it was found.
STYLE_DETECT_SEGMENTS = 4

_ECHO_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)


def style_exemplar(language: Optional[str]) -> Optional[str]:
    return STYLE_EXEMPLARS.get((language or "").strip().lower()) or None


def build_initial_prompt(settings: ASRSettings,
                         exemplar: Optional[str] = None) -> Optional[str]:
    """What to hand whisper as `initial_prompt`.

    The exemplar goes first and the user's own text (character names) last,
    because whisper keeps the *tail* of an over-long prompt
    (``previous_tokens[-(max_length // 2 - 1):]`` in its get_prompt) — so
    the names are the part that survives truncation.

    Worth knowing and not worth fighting: with
    ``condition_on_previous_text=False`` whisper resets the prompt after the
    first 30-second window, so the exemplar only shapes the opening of each
    ``transcribe()`` call. The conditioned first pass carries the style on
    by its own output; the windowed decodes get one window's worth each.
    """
    own = settings.initial_prompt.strip()
    if not exemplar:
        return own or None
    return f"{exemplar} {own}" if own else exemplar


def _is_prompt_echo(text: str, exemplar: Optional[str]) -> bool:
    """Did whisper simply repeat the exemplar back?

    A local, deterministic rule of the same kind as ``has_content`` — a cue
    that is exactly the prompt is not something the film said. Deliberately
    strict: only the whole exemplar, or one of its sentences, counts.
    """
    if not exemplar:
        return False
    flat = _ECHO_STRIP.sub("", text).lower()
    if not flat:
        return False
    parts = [exemplar] + re.split(r"[.!?。！？]", exemplar)
    return any(flat == _ECHO_STRIP.sub("", part).lower()
               for part in parts if part.strip())


def _pick_exemplar(model, language: Optional[str], soundtrack,
                   log: Optional[LogFn]) -> Optional[str]:
    """The sample sentence for this film's language, or None.

    With the language given, a table lookup. With ``auto``, whisper's own
    detector is asked first — **only to choose the exemplar**. The decode
    still runs with ``language=None``, so whisper's language judgement is
    unchanged by this; what would be unsafe is guessing, because a prompt
    in the wrong language drags the whole transcript toward it.
    """
    if language:
        found = style_exemplar(language)
        if log:
            # Say so either way. With the language given this used to
            # return in silence, so the job log showed the switch was on
            # and nothing about whether an exemplar was actually added —
            # which is the only part that affects the transcript.
            log(f"标点示例句：按指定的源语言 {language} 加在识别提示词前" if found
                else f"（标点示例句：没有 {language} 的示例句，本次不加）")
        return found
    try:
        detected, probability, _ = model.detect_language(
            audio=soundtrack(),
            vad_filter=True,
            language_detection_segments=STYLE_DETECT_SEGMENTS,
        )
    except Exception as exc:  # noqa: BLE001 — a nicety must not fail a job
        if log:
            log(f"（标点示例句：语言预检测未能进行，本次不加: {exc}）")
        return None
    if probability < STYLE_DETECT_MIN_PROBABILITY:
        if log:
            log(f"（标点示例句：语言预检测只有 {probability:.0%} 把握，本次不加）")
        return None
    found = style_exemplar(detected)
    if log:
        log(f"标点示例句：按预检测的语言 {detected}（{probability:.0%}）"
            + ("加在识别提示词前" if found else "查表没有示例句，本次不加"))
    return found


def _decode_windows(
    model,
    audio,
    windows: Sequence[tuple[float, float]],
    settings: ASRSettings,
    language: Optional[str],
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    debug=None,
    progress: Optional[ProgressFn] = None,
    what: str = "识别",
    exemplar: Optional[str] = None,
) -> tuple[List[Segment], int]:
    """Decode a list of windows with the VAD off; (segments, dropped).

    Lifted out of ``second_pass`` unchanged so a second caller
    (``windowed_first_pass``) runs exactly the same decode with exactly the
    same gates. Timestamps come back absolute. ``recovered`` is left False —
    whether these count as a first pass or as material for review is the
    caller's decision, and the two callers answer it differently.
    """
    out: List[Segment] = []
    dropped = 0
    done = 0.0
    span = sum(e - s for s, e in windows) or 1.0
    for start, end in windows:
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
                initial_prompt=build_initial_prompt(settings, exemplar),
            )
            for seg in found:
                text = seg.text.strip()
                if not text or not has_content(text):
                    continue
                if _is_prompt_echo(text, exemplar):
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
                out.append(Segment(
                    float(seg.start) + start, float(seg.end) + start, text, words,
                ))
                if debug is not None and debug.enabled:
                    _debug_segment(debug, seg, words, offset=start)
        except (InterruptedError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 — a bad window must not kill the job
            if log:
                log(f"⚠ {what} {start:.0f}-{end:.0f}s 失败（跳过）: {exc}")
        done += end - start
        if progress:
            progress(min(done / span, 1.0))
    return out, dropped


def second_pass(
    model,
    audio,
    segments: List[Segment],
    settings: ASRSettings,
    language: Optional[str],
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    debug=None,
    exemplar: Optional[str] = None,
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

    recovered, dropped = _decode_windows(
        model, audio, todo, settings, language, log, should_cancel, debug,
        what="二次识别", exemplar=exemplar,
    )
    for seg in recovered:
        seg.recovered = True

    # never overwrite what the first pass already covered — measured by
    # where its words are, the same way _blank_regions chose where to look.
    # Read as spans instead, one film's 29 stretched segments claimed 2180s
    # they never transcribed and this line discarded 147 recovered segments
    # (377s) found inside them, a twelve-turn conversation among them.
    covered = _covered_intervals(segments)
    kept = [s for s in recovered if not _overlaps_any(s, covered)]
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
        # Whisper prefixes ♪ only with the VAD off, never with it on (0 of
        # 434 vs 90 of 216 on one film) — so this counts the decoding mode,
        # not the singing. Recorded because it is still a useful second
        # opinion for the lyrics pass to be scored against.
        debug.kv(
            "其中 whisper 自带 ♪ 标记的",
            f"{sum(1 for s in kept if '♪' in s.text)} 段"
            f"（第一遍: {sum(1 for s in segments if '♪' in s.text)} 段）",
        )
    return merged


# Under this share of the film kept by the VAD, the first pass is too thin
# to anchor a review of everything else. Measured on three films: a VHS
# capture kept 5.2%, a Japanese DVD 28%, an English Blu-ray 64% — the same
# figure the job log already prints as "VAD 保留语音 …（占音频 X%）".
WINDOWED_FIRST_PASS_BELOW = 0.10
# ...and an absolute floor, because the other test is a *relative* one and
# a track with nothing on it passes every relative test trivially: with no
# speech to measure against, its own hiss is "as loud as speech". Film
# dialogue sits far above this — measured on the same three films, -18.2
# (VHS), -28.9 (Blu-ray) and -38.1 (DVD) dBFS — while an empty commentary
# stream, a muted capture or the wrong track sits at or under it. Handing
# one of those to a VAD-off whisper is how you get a film's worth of
# invented dialogue.
WINDOWED_FIRST_PASS_MIN_DB = -55.0


def _vad_intervals(audio, vad_parameters: Optional[dict]) -> List[tuple[float, float]]:
    """Silero with the *job's* parameters — what the first pass would have kept.

    Not ``speech_intervals_of``, which deliberately pins the reference
    threshold so coverage figures compare across runs. Here the question is
    the other one: which of these segments would a VAD-gated first pass have
    produced anyway.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(**(vad_parameters or {}))
    return [
        (chunk["start"] / 16000.0, chunk["end"] / 16000.0)
        for chunk in get_speech_timestamps(audio, options)
    ]


def should_window_first_pass(
    audio, settings: ASRSettings, after_vad: Optional[float], duration: float,
) -> tuple[bool, str]:
    """(engage, why) for the VAD-off first pass. Both conditions must hold.

    The first says the VAD has collapsed; the second says there is really
    audio out there to transcribe. Without the second, a film with a silent
    or wrongly-selected track — which also keeps ~0% — would be handed to a
    VAD-off whisper, and a VAD-off whisper on an empty track writes
    dialogue out of its training data. That is the failure this whole
    module spends `vet.py` guarding against; there is no reason to invite
    more of it.
    """
    if not (settings.windowed_first_pass and settings.vad_filter):
        return False, ""
    if not after_vad or not duration:
        return False, "VAD 未报告保留时长"
    share = after_vad / duration
    if share >= WINDOWED_FIRST_PASS_BELOW:
        return False, (f"VAD 保留了 {share:.0%} 的音频（阈值 "
                       f"{WINDOWED_FIRST_PASS_BELOW:.0%}），不需要兜底")
    intervals = speech_intervals_of(audio, settings)
    levels = level_profile(audio)
    speech_db = _speech_level(levels, intervals)
    if speech_db < WINDOWED_FIRST_PASS_MIN_DB:
        return False, (f"VAD 只保留了 {share:.0%}，但整条音轨的电平只有 "
                       f"{speech_db:.0f} dBFS（低于 {WINDOWED_FIRST_PASS_MIN_DB:.0f}），"
                       "更像是音轨本身没有内容，不做分窗兜底")
    loud = sum(d for _, _, d, _ in
               _loud_undetected(levels, intervals, duration, speech_db))
    if loud < after_vad:
        return False, (f"VAD 只保留了 {share:.0%}，但 VAD 之外接近人声电平的音频"
                       f"只有 {loud:.0f}s（少于 VAD 保留的 {after_vad:.0f}s），"
                       "更像是音轨本身没有内容，不做分窗兜底")
    return True, (f"VAD 只保留了 {share:.0%} 的音频（{after_vad:.0f}s），"
                  f"而 VAD 之外还有 {loud:.0f}s 的音量接近人声")


def windowed_first_pass(
    model,
    audio,
    settings: ASRSettings,
    language: Optional[str],
    vad_parameters: Optional[dict],
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    debug=None,
    progress: Optional[ProgressFn] = None,
    exemplar: Optional[str] = None,
) -> List[Segment]:
    """Decode the whole timeline as VAD-off windows, and mark their source.

    The marking is the load-bearing part, and it is deliberately neither
    "all first pass" nor "all recovered":

    * all first pass would hand a transcript made entirely with the VAD off
      to the rest of the pipeline unreviewed, and a VAD-off whisper over
      non-speech emits its training data — one run produced 547 such
      segments of which the review dropped 149, boilerplate among them;
    * all recovered would leave ``vet.py`` with nothing confirmed to judge
      against (its ``no_context`` path, which it says outright is less
      accurate) and send the entire film back for review in one request.

    So a segment counts as first pass exactly when a VAD-gated run would
    have seen it — its words overlap what silero, with the job's own
    parameters, calls speech — and as recovered otherwise. That reproduces
    today's authority boundary line for line: review may delete only what
    only the VAD-off decode could see.
    """
    duration = len(audio) / 16000.0
    windows = list(_windows([(0.0, duration)], SECOND_PASS_WINDOW))
    if log:
        log(f"分窗识别兜底：把整条时间轴切成 {len(windows)} 个窗口、关闭 VAD 重跑…")
    if debug is not None and debug.enabled:
        debug.section("分窗识别（VAD 失明兜底）")
        debug.kv("窗口", f"{len(windows)} 个，合计 {duration:.0f}s")
    found, dropped = _decode_windows(
        model, audio, windows, settings, language, log, should_cancel, debug,
        progress=progress, what="分窗识别", exemplar=exemplar,
    )
    heard = _vad_intervals(audio, vad_parameters)
    for seg in found:
        seg.recovered = not _heard_by_vad(seg, heard)
    review = [s for s in found if s.recovered]
    if log:
        log(f"分窗识别完成：{len(found)} 段 / "
            f"{sum(s.end - s.start for s in found):.0f}s，其中 {len(review)} 段 / "
            f"{sum(s.end - s.start for s in review):.0f}s 落在 VAD 判定的语音之外"
            "（待复核）"
            + (f"；另丢弃 {dropped} 段可疑输出" if dropped else ""))
    if debug is not None and debug.enabled:
        debug.kv("识别结果", f"{len(found)} 段 / "
                             f"{sum(s.end - s.start for s in found):.0f}s")
        debug.kv("VAD 也听得到（算第一遍）", f"{len(found) - len(review)} 段")
        debug.kv("只有关 VAD 才看得见（待复核）", f"{len(review)} 段")
        debug.kv("丢弃（重复或置信度过低）", dropped)
    return sorted(found, key=lambda s: s.start)


def _overlaps_any(seg: Segment, covered: Sequence[tuple[float, float]]) -> bool:
    """Does *seg* land on timeline the first pass already has text for?"""
    return any(
        min(seg.end, end) - max(seg.start, start) > 0.2
        for start, end in covered
    )


def _heard_by_vad(seg: Segment, heard: Sequence[tuple[float, float]]) -> bool:
    """Would a VAD-gated run have seen this segment at all?

    By where its words are, for the same reason ``_word_spans`` exists: a
    segment's own span is routinely stretched across silence it never
    transcribed, and judging by that span would quietly promote a line
    whisper found only with the VAD off.
    """
    spans = sorted((w.start, w.end) for w in seg.words) or [(seg.start, seg.end)]
    return any(
        min(end, heard_end) - max(start, heard_start) > 0.2
        for start, end in spans
        for heard_start, heard_end in heard
    )


@dataclass
class CoverageVerdict:
    """What the coverage pass measured, for callers that need to know.

    ``vad_blind`` is the one field with consequences: the API engine's
    density, correlation and drift checks are all measured against silero,
    so when silero is the thing that failed they have to say so rather than
    accuse the transcript.
    """

    speech: float = 0.0
    loud: float = 0.0
    transcribed: float = 0.0
    inside: float = 0.0
    vad_share: float = float("nan")
    transcript_in_vad: float = float("nan")
    coverage: Optional[float] = None
    vad_blind: bool = False


def _speech_level(levels: Sequence[float],
                  intervals: Sequence[tuple[float, float]]) -> float:
    """How loud this film's speech is, in dBFS.

    The median over the intervals silero is sure about; with no intervals
    at all there is nothing to take a median of, so the loud end of the
    film stands in — enough to say whether there is any audio here.
    """
    if not levels:
        return -140.0
    if not intervals:
        ordered = sorted(levels)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    import statistics

    spoken = [_median_level(levels, s, e) for s, e in intervals if e - s >= 2.0]
    if spoken:
        return statistics.median(spoken)
    return _median_level(levels, intervals[0][0], intervals[0][1])


def loud_seconds(levels: Sequence[float], speech_db: float) -> float:
    """Seconds whose level is within LEVEL_SUSPICIOUS_DB of known speech.

    The denominator for "how much of the audible film did silero hear" —
    the whole film would count silence against it.
    """
    floor = speech_db - LEVEL_SUSPICIOUS_DB
    return float(sum(1 for db in levels if db >= floor))


def _loud_undetected(
    levels: Sequence[float],
    intervals: Sequence[tuple[float, float]],
    duration: float,
    speech_db: float,
    min_gap: float = MIN_UNDETECTED_SECONDS,
) -> List[tuple[float, float, float, float]]:
    """Stretches silero never called speech that are as loud as the speech.

    The kind of miss a threshold cannot reach: lowering it only lets more
    music in. Music and ambience reach speech loudness too, so this is a
    lead, never a verdict.
    """
    out: List[tuple[float, float, float, float]] = []
    for s, e in _outside_intervals(intervals, duration, min_gap):
        db = _median_level(levels, s, e) if levels else -140.0
        if db >= speech_db - LEVEL_SUSPICIOUS_DB:
            out.append((s, e, e - s, db))
    return out


def vad_blindness(
    levels: Sequence[float],
    intervals: Sequence[tuple[float, float]],
    segments: Sequence[Segment],
    speech_db: float,
) -> tuple[float, float, float, bool]:
    """(loud, vad_share, transcript_in_vad, blind) — see VAD_BLIND_BELOW."""
    loud = loud_seconds(levels, speech_db)
    speech = sum(e - s for s, e in intervals)
    spans = _covered_intervals(segments)
    transcribed = sum(e - s for s, e in spans)
    inside = sum(
        max(0.0, min(end, we) - max(start, ws))
        for start, end in intervals
        for ws, we in spans
        if we > start and ws < end
    )
    vad_share = (speech / loud) if loud else float("nan")
    in_vad = (inside / transcribed) if transcribed else float("nan")
    blind = bool(
        loud >= VAD_BLIND_MIN_LOUD
        and vad_share == vad_share and vad_share < VAD_BLIND_BELOW
        and in_vad == in_vad and in_vad < VAD_BLIND_BELOW
    )
    return loud, vad_share, in_vad, blind


def _report_coverage(
    audio,
    settings: ASRSettings,
    segments: List[Segment],
    log: Optional[LogFn],
    debug,
    intervals: Optional[List[tuple[float, float]]] = None,
) -> Optional[CoverageVerdict]:
    """How much of the speech actually made it into text.

    Purely diagnostic, and never allowed to fail a job: a two-hour film had
    15-20% of its speech silently absent, and three runs with different VAD
    settings each lost a *different* 15-20% — which no amount of parameter
    tuning fixes and nothing in the output revealed.

    *intervals* lets a caller that already ran silero pass its answer in;
    the API engine does, which saves a second pass over the whole film.
    """
    try:
        import time

        started = time.monotonic()
        duration = len(audio) / 16000.0
        if intervals is None:
            intervals = speech_intervals_of(audio, settings)
        levels = level_profile(audio)
        speech_db = _speech_level(levels, intervals)
        loud, vad_share, in_vad, blind = vad_blindness(
            levels, intervals, segments, speech_db)
        verdict = CoverageVerdict(
            speech=sum(e - s for s, e in intervals), loud=loud,
            vad_share=vad_share, transcript_in_vad=in_vad, vad_blind=blind)
        if not intervals:
            # Silence here used to mean no coverage line at all, on exactly
            # the films where the reader most needs to know why.
            if log:
                log(f"⚠ silero 在整条音轨上一段语音都没找到（参考阈值 "
                    f"{REFERENCE_VAD_THRESHOLD}），本次无法计算识别覆盖率；"
                    f"每秒电平 P90 {speech_db:.1f} dBFS，"
                    f"音量接近人声的音频 {loud:.0f}s / 全片 {duration:.0f}s")
            _debug_coverage_context(debug, levels, intervals, duration, speech_db,
                                    verdict, time.monotonic() - started)
            return verdict
        speech, covered, misses = coverage_report(intervals, segments)
        share = covered / speech if speech else 1.0
        verdict.transcribed = covered
        verdict.coverage = share
        elapsed = time.monotonic() - started

        # stretches Silero never called speech, but which are as loud as the
        # speech it did find — the case a threshold cannot reach
        undetected = _loud_undetected(levels, intervals, duration, speech_db)

        summary = (
            f"识别覆盖率 {share:.0%}（语音 {speech:.0f}s，转写 {covered:.0f}s，"
            f"漏识别 {len(misses)} 处 / {sum(d for _, _, d in misses):.0f}s）"
        )
        # `undetected` deliberately stays out of this line: on a film that is
        # 59% score and ambience, "as loud as speech" flags every music cue,
        # and a warning nobody can act on is worse than none. It is a lead to
        # follow in the debug log, not a verdict.
        if log and blind:
            log(f"⚠ 本片源上 silero 基本失聪：音量接近人声的音频有 {loud:.0f}s，"
                f"silero 只把其中 {speech:.0f}s（{vad_share:.0%}）判成语音；"
                f"转写出的内容也只有 {in_vad:.0%} 落在它的语音区间内"
                "（严重劣化的采集、噪声底高的片源会这样）。"
                "下面的覆盖率，以及字数/语音秒、时钟漂移、相关性这些以它为基准的读数，"
                "本次只作参考、不据此告警")
            log(summary + "（VAD 失聪，仅供参考）")
        elif log:
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
            debug.kv("音量接近人声的秒数", f"{loud:.0f}s"
                     f"（离人声 {LEVEL_SUSPICIOUS_DB:.0f} dB 以内）")
            debug.kv("silero 占其中", f"{vad_share:.0%}")
            debug.kv("转写落在 VAD 内", f"{in_vad:.0%}")
            if blind:
                debug.kv("VAD 失聪", f"是——两项都低于 {VAD_BLIND_BELOW:.0%}，"
                                     "以 silero 为基准的读数本次只作参考")
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
        return verdict
    except Exception as exc:  # noqa: BLE001 — diagnostics must never fail a job
        if log:
            log(f"（识别覆盖率统计未能完成: {exc}）")
        return None


def _debug_coverage_context(debug, levels, intervals, duration, speech_db,
                            verdict: CoverageVerdict, elapsed: float) -> None:
    """The debug section for a film silero found no speech in at all.

    The coverage table cannot be written — there is no denominator — but
    the two things that say whether the audio is there at all (the loud
    stretches silero missed, and the per-minute level profile) can, and
    without them the reader is left with a job log that simply says
    nothing about the transcript's quality.
    """
    if debug is None or not getattr(debug, "enabled", False):
        return
    from app.core.debuglog import fmt_time

    debug.section("语音检测覆盖（silero VAD vs 实际转写）")
    debug.line(f"silero 在整条音轨上一段语音都没找到（参考阈值 "
               f"{REFERENCE_VAD_THRESHOLD}），覆盖率无法计算。\n"
               "下面两项仍然有意义：音轨里到底有没有声音、声音在哪。\n")
    debug.kv("每秒电平 P90", f"{speech_db:.1f} dBFS")
    debug.kv("音量接近人声的秒数", f"{verdict.loud:.0f}s / 全片 {duration:.0f}s")
    debug.kv("本次检测耗时", f"{elapsed:.1f}s")
    undetected = _loud_undetected(levels, intervals, duration, speech_db)
    debug.line(f"\n② 音量接近人声、却未被判定为语音（{len(undetected)} 处，"
               f"≥{MIN_UNDETECTED_SECONDS}s）：")
    debug.lines(
        f"  {fmt_time(s)} → {fmt_time(e)}  {d:6.1f}s  {db:6.1f} dBFS"
        for s, e, d, db in sorted(undetected, key=lambda m: -m[2])[:60]
    )
    debug.line("\n每分钟音量剖面（dBFS，用于判断某段是否真的有声音）：")
    debug.lines(
        f"  {i:3d}min  {_median_level(levels, i * 60, (i + 1) * 60):6.1f}"
        for i in range(int(duration // 60) + 1)
    )


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


def _debug_segment(
    dbg, seg, words: List[Word], offset: float = 0.0, note: str = ""
) -> None:
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
        + (f"  ⚠ {note}" if note else "")
        + f"\n    {seg.text.strip()}"
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
    llm=None,
    client=None,  # injectable for tests (api engine only)
    usage: Optional[dict] = None,
) -> tuple[List[Segment], str]:
    """Transcribe *wav_path*; returns (segments, detected_language).

    *language* is a whisper language code, or None for auto-detection.

    With ``settings.engine == "api"`` the work goes to a multimodal LLM
    instead (services/asr_api.py) and *llm* carries its credentials. The
    branch sits here rather than in the pipeline so that everything either
    engine needs — the extracted WAV, the language, cancellation, progress
    — is already in one place, the way ocr.read_cues dispatches its two.
    """
    if settings.engine == "api":
        from app.services import asr_api

        if llm is None:
            raise ValueError("API 语音识别引擎需要先配置 LLM")
        return asr_api.transcribe(
            wav_path, settings, llm, language=language, progress=progress,
            log=log, should_cancel=should_cancel, network=network,
            debug=debug, client=client, usage=usage,
        )

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

        # one decode of the audio serves the style detection, the windowed
        # fallback, the second pass and the coverage report
        audio = None

        def soundtrack():
            nonlocal audio
            if audio is None:
                from faster_whisper.audio import decode_audio

                audio = decode_audio(wav_path, sampling_rate=16000)
            return audio

        exemplar = None
        if settings.style_prompt:
            exemplar = _pick_exemplar(model, language, soundtrack, log)

        def decode(condition: bool = True):
            return model.transcribe(
                wav_path,
                language=language,
                beam_size=settings.beam_size,
                word_timestamps=settings.word_timestamps,
                vad_filter=settings.vad_filter,
                vad_parameters=vad_parameters,
                initial_prompt=build_initial_prompt(settings, exemplar),
                condition_on_previous_text=condition,
            )

        segments_iter, info = decode()
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
            if exemplar:
                dbg.kv("标点示例句", exemplar)
            dbg.line(
                "\n每个 segment：起止、平均对数概率、无语音概率、压缩比"
                "（压缩比高或 no_speech 高 = 可疑/幻觉），随后是词级时间戳。"
                "\n词级时间戳里出现的异常大间隔，正是字幕被切成半个词的直接原因。\n"
            )

        echoed: List[str] = []

        def collect(segments_iter) -> tuple[List[Segment], int]:
            out: List[Segment] = []
            blank = 0
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
                if progress and total:
                    progress(min(seg.end / total, 1.0))
                if not has_content(text):
                    # kept in the debug log — what whisper said is the
                    # evidence — but never handed on. See has_content().
                    blank += 1
                    if dbg:
                        _debug_segment(dbg, seg, words, note="无文字内容，已丢弃")
                    continue
                if _is_prompt_echo(text, exemplar):
                    echoed.append(text)
                    if dbg:
                        _debug_segment(dbg, seg, words, note="示例句回声，已丢弃")
                    continue
                out.append(Segment(float(seg.start), float(seg.end), text, words))
                if log:
                    log(f"[{seg.start:7.2f}s] {text}")
                if dbg:
                    _debug_segment(dbg, seg, words)
            return out, blank

        windowed = False
        if settings.windowed_first_pass and settings.vad_filter:
            # The generator has only detected the language so far — decoding
            # is lazy — so not iterating it costs nothing and skips the pass
            # whose output would be discarded anyway.
            try:
                engage, why = should_window_first_pass(
                    soundtrack(), settings, after_vad, total)
                if log and why:
                    log(("分窗识别兜底：" + why) if engage
                        else f"（未启用分窗识别兜底：{why}）")
                if engage:
                    results = windowed_first_pass(
                        model, soundtrack(), settings, language or info.language,
                        vad_parameters, log, should_cancel, debug, progress,
                        exemplar=exemplar)
                    blank = 0
                    windowed = True
            except (InterruptedError, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001 — fall back to the normal pass
                if log:
                    log(f"（分窗识别兜底未能进行，按常规流程继续: {exc}）")

        if not windowed:
            results, blank = collect(segments_iter)

        if blank and not results:
            # Not a quiet film — a decode loop. Whisper primes each window
            # with what it decoded in the last one, so once it emits a
            # meaningless symbol the prompt keeps it there for the rest of
            # the film. Every gate waves it through: a lone dash compresses
            # to 0.33 at logprob -0.15. The second pass already runs each
            # window unprimed for exactly this reason; one more decode is a
            # cheap price for the difference between a subtitle and none.
            if log:
                log(f"⚠ 识别结果的 {blank} 段全都没有文字内容（如「-」「…」），"
                    "判定为解码陷入循环，正在关闭上文关联重新识别…")
            if dbg:
                dbg.kv("整片无文字内容", f"{blank} 段 → 关闭上文关联重试")
            retried, blank = collect(decode(condition=False)[0])
            if retried:
                if log:
                    log(f"重试成功，得到 {len(retried)} 段文字")
                results = retried
            elif log:
                log("⚠ 重试后仍然没有任何文字，请检查音轨是否正确、"
                    "或改用其它识别模型重试")

        if blank:
            share = blank / (blank + len(results))
            note = f"语音识别丢弃了 {blank} 段没有任何文字内容的输出（如「-」「…」「♪」）"
            if log:
                log(f"⚠ {note}，占全部输出的 {share:.0%}" if share >= 0.3 else note)
            if dbg:
                dbg.kv("无文字内容而丢弃的 segment", f"{blank} 段（占 {share:.0%}）")

        try:
            soundtrack()
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"（音频复核未能加载: {exc}）")
        if windowed and settings.second_pass and log:
            log("第一遍已按窗口关 VAD 覆盖整条时间轴，本次跳过二次识别")
        if (audio is not None and settings.second_pass and settings.vad_filter
                and not windowed):
            results = second_pass(
                model, audio, results, settings,
                language or info.language, log, should_cancel, debug,
                exemplar=exemplar,
            )
        if echoed and log:
            log(f"（丢弃了 {len(echoed)} 段与标点示例句完全相同的输出）")
        if dbg and exemplar:
            dbg.kv("示例句回声而丢弃", f"{len(echoed)} 段")
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
