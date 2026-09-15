"""Speech recognition through a multimodal LLM that accepts audio input.

The alternative to faster-whisper (ASRSettings.engine == "api"): cut the
soundtrack into windows locally, send each one to an OpenAI-compatible
endpoint as an `input_audio` part, and ask for the lines it hears.

Three things measured on gemini-3.8-flash through a relay decide the whole
design, and none of them are guessable from the documentation:

1. **Audio costs exactly 25 tokens per second** (checked at 2 / 30 / 120 /
   720 / 1046 / 2400s — perfectly linear). That is the only evidence that
   the audio actually reached the model: a relay that drops the audio part
   still answers, fluently and plausibly, from the text prompt alone. Hence
   the token check in `validate` — a reply that reads well proves nothing.
2. **`max_tokens` is silently ignored and `finish_reason` is always
   "stop"**, even on a reply capped at 200 tokens that emitted 1941. A
   truncated transcript therefore cannot be detected from the response
   envelope; it has to be caught by content, against the VAD.
3. **The clock does not drift on this model** (+0.22 ms/s over 40 minutes)
   — but it is documented to drift catastrophically on others (reports of
   −157s over 12 minutes, a clock running 22% fast). So drift is treated as
   a measured quantity, not an assumption: every job computes the slope,
   warns above 20 ms/s and fails above 50. Measured noise is 0.1–1.6 and
   the reported failure is 218; the threshold sits in the empty middle.

And one property that is the reason to use this engine at all: it
transcribes dialogue mixed 3 dB *under* an orchestral bed, where silero
scores zero. That is exactly the hole `second_pass` + `vet` exist to patch
around in the local engine — and it is why the VAD here only ever decides
*where to cut*, never *what to drop*. The windows tile the timeline.
"""

from __future__ import annotations

import base64
import io
import re
import statistics
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from app.models.schemas import ASRSettings, LLMSettings, NetworkSettings
from app.services import asr
from app.services.asr import LogFn, ProgressFn, Segment

SAMPLE_RATE = 16_000

# Measured, exactly linear. Used to prove the audio reached the model.
AUDIO_TOKENS_PER_SECOND = 25.0
# ...allowing for the endpoint counting slightly differently.
MIN_AUDIO_TOKEN_SHARE = 0.8

# Silero options for *cutting*. Deliberately not the job's VAD settings:
# the project default (min_silence=2000ms) found 4 intervals in 17 minutes
# of narration, which is useless for choosing cut points. The coverage
# report keeps using the job's settings so its numbers stay comparable
# across runs — two questions, two parameter sets.
CUT_MIN_SPEECH_MS = 100
CUT_MIN_SILENCE_MS = 300
MIN_CUT_SILENCE = 0.35  # shorter than silero's own floor is not a candidate

WINDOW_MIN_RATIO = 0.4  # do not shorten a window just because a gap appeared
WINDOW_MAX_RATIO = 1.4  # ...and never run past this without one
WINDOW_HARD_MAX = 420.0  # whatever the target, no window runs longer than this
WINDOW_FLOOR = 30.0     # halving stops here

WINDOW_ATTEMPTS = 2     # one retry, as in ocr._read_sheet
MAX_CALL_FACTOR = 4     # total requests cap = this × number of windows
# How much of a film may go untranscribed before the stage gives up.
#
# The ladder used to end at "fail the whole stage", on the principle that
# a subtitle with holes in it is worse than no subtitle. That is right
# about a relay refusing everything and wrong about one bad stretch: a
# window that nothing gets through destroys an hour of work on the other
# ninety minutes. Measured, a stretch that neither container will answer
# does exist (Donor 49:03–52:34, empty in mp3 and in wav alike), while its
# own halves transcribe normally — so gaps are rare, real, and mostly
# recoverable by splitting.
#
# So a window that runs out of ladder becomes a *named* gap and the film
# carries on, up to this share of it; past that the endpoint is refusing
# systematically and a holed transcript is not worth delivering. Silence
# is never an option either way: every missing range is printed.
MAX_MISSING_SHARE = 0.02
PROSE_RUN = 3           # this many prose replies in a row = a refusal, not a format error
# An empty reply buys one extra attempt, spent on the other container.
#
# What the empty reply is keyed to is the exact (audio bytes, prompt)
# pair, and perturbing either side gets an answer. Measured on the
# hardest known clip (Donor, 49:03–50:35, 92 seconds), ten trials each:
# **as mp3 with the plain prompt it came back with an empty candidate
# list 10 times out of 10; as wav with the same prompt it transcribed 10
# times out of 10**. Every refusal reported `prompt_tokens=2755,
# completion_tokens=0` — 92s × 25 tokens plus the prompt — so the audio
# arrived and was charged for and nothing came back. A neighbouring clip
# of the same length answers identically in both containers, so this is
# not mp3 being worse; it is one stretch of audio the pair will not
# answer.
#
# The prompt side is why the old ladder appeared to work: its second
# attempt carried a `complaint`, and in the recorded run all four
# recovered windows came back on exactly that attempt. But that sentence
# tells the model its last answer fell short when there was no answer at
# all, so the perturbation this uses is the honest one — same seconds,
# different bytes. A plain resend changes neither and is worth nothing
# (0/10 above), so the switch gets one try and no spare. Splitting
# follows, and perturbs the bytes again by changing the seconds.
OTHER_FORMAT = {"mp3": "wav", "wav": "mp3"}

TIME_SLACK = 1.0            # seconds past the window end that get clamped
MIN_CUE_SECONDS = 0.2       # subtitle.py must never see a zero-length cue
MAX_INVERSION_SHARE = 0.05  # more out-of-order lines than this = unusable
MAX_REPEAT_SHARE = 0.30     # identical neighbours: a decode loop
MIN_SPEECH_FOR_TEXT = 3.0   # below this, "nothing here" needs no defence
DISPUTE_SPEECH = 10.0       # ...above this, disagreeing with the model is worth a log line
PROSE_MIN_CHARS = 20        # a paragraph where a transcript was asked for
# Truncation is measured against the window's own speech, not in absolute
# seconds: silero calls plenty of music "speech", so a window whose last
# stretch is score would fail a fixed threshold every time. Deliberately
# generous — a reply that really was cut off loses most of its window,
# while the cost of crying truncation is a retry, a split and finally a
# failed job. Small holes are for the coverage report to surface.
TRUNCATION_MIN_TAIL = 5.0
TRUNCATION_TAIL_SHARE = 0.35

DENSITY_BAND = 3.0   # chars/speech-second outside median/3..median*3 → noted
DRIFT_WARN_MS = 20.0
DRIFT_FAIL_MS = 50.0
ALIGNMENT_MIN_CORRELATION = 0.45  # same number, same meaning as translator's
ALIGNMENT_MIN_WINDOWS = 6         # below this the correlation is noise

MP3_BITRATE = 64_000
MAX_INLINE_BYTES = 18 * 1024 * 1024  # headroom under the ~20MB inline cap

NO_SPEECH = "NO SPEECH"

_FENCE_RE = re.compile(r"^\s*```")
_LANG_RE = re.compile(r"^\s*#?\s*lang\s*[:：]\s*([A-Za-z]{2,3})\b", re.I)
# [MM:SS.mmm --> MM:SS.mmm] text — with HH:, bare seconds, "->", and the
# usual bullet/quote noise tolerated, the way translator._LINE_RE is.
_CUE_RE = re.compile(
    r"^\s*[*\-•]?\s*[\[(]?\s*"
    r"(?P<a>(?:\d+:)?\d{1,2}:\d{1,2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d{1,3})?)"
    r"\s*(?:-{1,2}>|→|~|—)\s*"
    r"(?P<b>(?:\d+:)?\d{1,2}:\d{1,2}(?:[.,]\d{1,3})?|\d+(?:[.,]\d{1,3})?)"
    r"\s*[\])]?\s*[:：]?\s*(?P<text>.*)$"
)

Cue = Tuple[float, float, str]


class RefusalError(RuntimeError):
    """The model answered in prose instead of transcribing, several times."""


# ------------------------------------------------------------- audio bytes


def audio_part(data: bytes, fmt: str) -> dict:
    """The OpenAI-compatible content part carrying one clip.

    Shared verbatim with the settings-page probe, for the same reason
    test-vision reuses ocr._png_data_url: a probe that builds its own
    request proves the probe works, not the feature.
    """
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(data).decode("ascii"),
            "format": fmt,
        },
    }


def encode_window(audio, start: float, end: float, fmt: str = "mp3") -> bytes:
    """Encode audio[start:end] (a float32 mono 16kHz array) to *fmt* bytes."""
    lo = max(0, int(start * SAMPLE_RATE))
    hi = min(len(audio), int(end * SAMPLE_RATE))
    return encode_samples(audio[lo:hi], fmt)


def encode_samples(samples, fmt: str = "mp3") -> bytes:
    """As above, for samples already sliced. Same shape as audio.py's writer."""
    import fractions

    import av

    codec = "libmp3lame" if fmt == "mp3" else "pcm_s16le"
    buf = io.BytesIO()
    with av.open(buf, mode="w", format=fmt) as container:
        stream = container.add_stream(codec, rate=SAMPLE_RATE)
        stream.layout = "mono"
        if fmt == "mp3":
            stream.bit_rate = MP3_BITRATE
        resampler = av.AudioResampler(
            format=stream.format.name, layout="mono", rate=SAMPLE_RATE
        )
        step = 1024
        pts = 0
        for i in range(0, len(samples), step):
            chunk = samples[i:i + step]
            frame = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1), format="fltp", layout="mono"
            )
            frame.sample_rate = SAMPLE_RATE
            frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
            frame.pts = pts
            pts += len(chunk)
            for resampled in resampler.resample(frame):
                for packet in stream.encode(resampled):
                    container.mux(packet)
        for resampled in resampler.resample(None):
            for packet in stream.encode(resampled):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buf.getvalue()


# ---------------------------------------------------------- window planning


def cut_intervals(audio) -> List[tuple[float, float]]:
    """Silero speech intervals at the resolution cutting needs."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=asr.REFERENCE_VAD_THRESHOLD,
        min_speech_duration_ms=CUT_MIN_SPEECH_MS,
        min_silence_duration_ms=CUT_MIN_SILENCE_MS,
        speech_pad_ms=0,
    )
    return [
        (chunk["start"] / float(SAMPLE_RATE), chunk["end"] / float(SAMPLE_RATE))
        for chunk in get_speech_timestamps(audio, options)
    ]


def plan_windows(
    intervals: Sequence[tuple[float, float]], duration: float, target: float
) -> tuple[List[tuple[float, float]], int]:
    """Tile [0, duration) with windows that break only in silence.

    Returns (windows, forced) — *forced* counts the cuts that had to be made
    mid-speech because no silence existed in the allowed band.

    The windows are contiguous and exhaustive by construction. That is the
    "never silently drop content" guarantee at this level: the VAD decides
    only *where* the seams go. Dialogue it scores as silence — under music,
    the very case this engine is better at than whisper — stays inside a
    window and gets transcribed anyway.
    """
    if duration <= 0:
        return [], 0

    lo_span = max(WINDOW_FLOOR, target * WINDOW_MIN_RATIO)
    hi_span = max(lo_span + 1.0, min(target * WINDOW_MAX_RATIO, WINDOW_HARD_MAX))

    gaps: List[tuple[float, float]] = []
    for before, after in zip(intervals, list(intervals)[1:]):
        if after[0] - before[1] >= MIN_CUT_SILENCE:
            gaps.append((before[1], after[0]))
    if intervals:
        if intervals[0][0] >= MIN_CUT_SILENCE:
            gaps.insert(0, (0.0, intervals[0][0]))
        if duration - intervals[-1][1] >= MIN_CUT_SILENCE:
            gaps.append((intervals[-1][1], duration))

    cuts: List[float] = []
    cursor = 0.0
    forced = 0
    while duration - cursor > hi_span:
        lo, hi = cursor + lo_span, cursor + hi_span
        band = [g for g in gaps if lo <= (g[0] + g[1]) / 2 <= hi]
        if band:
            # The longest silence wins: a three-second pause is a scene
            # boundary, a 0.4s one may be inside a sentence. Ties go to
            # whichever sits closest to the target length.
            best = max(band, key=lambda g: (round(g[1] - g[0], 1),
                                            -abs((g[0] + g[1]) / 2 - (cursor + target))))
            cut = (best[0] + best[1]) / 2
        else:
            cut = cursor + hi_span
            forced += 1
        cuts.append(cut)
        cursor = cut
    cuts.append(duration)
    return [(a, b) for a, b in zip([0.0] + cuts[:-1], cuts)], forced


def speech_inside(
    intervals: Sequence[tuple[float, float]], start: float, end: float
) -> float:
    """Seconds of silero-detected speech inside [start, end)."""
    return sum(
        max(0.0, min(end, e) - max(start, s)) for s, e in intervals
    )


def split_point(
    intervals: Sequence[tuple[float, float]], start: float, end: float
) -> float:
    """Where to halve a window that will not verify — the best silence near
    the middle, or the middle itself."""
    middle = (start + end) / 2
    gaps = [
        ((before[1] + after[0]) / 2, after[0] - before[1])
        for before, after in zip(intervals, list(intervals)[1:])
        if before[1] > start + WINDOW_FLOOR / 2 and after[0] < end - WINDOW_FLOOR / 2
    ]
    if not gaps:
        return middle
    return min(gaps, key=lambda g: (abs(g[0] - middle) - g[1]))[0]


# ----------------------------------------------------------------- prompt


def build_prompt(seconds: float, language: str = "", proper_nouns: str = "",
                 complaint: str = "") -> str:
    """What to ask for one window.

    Deliberately asks for the format the model already produces
    (`[MM:SS.mmm --> MM:SS.mmm]`): it ignored every request for decimal
    seconds, and fighting that only buys retries.
    """
    hint = f"（原文语言：{language}）" if language else ""
    clock = f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"
    lines = [
        f"你是一名影片字幕转写员{hint}。下面是一部影片音轨里截出来的一段音频。",
        "请把这段音频里**所有说出来的话**按时间顺序完整转写下来，一个字都不要漏。",
        "",
        "要求：",
        "- 只转写、不翻译，输出必须与说话人使用的语言一致；",
        "- 唱出来的歌词也要转写，但不要加 ♪ 之类的标记；",
        # Without this the model returns bare clauses: two Japanese films
        # came back with 72% and 57% of their lines not ending a sentence,
        # which trips segmenter.is_effectively_unpunctuated (>50%) and
        # defers every sentence merge to the refine pass. The second half
        # is the load-bearing half — a full stop on *every* line drives
        # open_ended_ratio to 0 and leaves the merger no evidence at all
        # that two lines belong to one sentence.
        "- 每行末尾按该语言的习惯写句末标点（中文/日文用 。！？，英文等用 .!?）；"
        "一句话被拆成几行时，只有句子真正结束的那一行以句末标点结尾，其余行末不写句末标点；",
        "- 不要描述音效、音乐、环境声，不要加任何注释或说明；",
        "- 这段音频可能从半句话开始、也可能在半句话处结束，听到半句就写半句；",
        # Measured: on a stretch of score with no voice in it, five runs
        # invented dialogue four times without this line and twice with it.
        # It does not solve the problem — nothing local can, because silero
        # hears the same drone as speech — but it is one line of prompt.
        "- 这段音频可能只有音乐、环境噪音或静默，完全没有人说话。"
        "那种情况下**绝对不要**凭空写出任何台词；",
        f"- 时间从**这段音频的开头**算起（不是影片开头）。这段音频长 {seconds:.2f} 秒"
        f"（{clock}），任何时间都不得超过它，且必须逐行递增；",
        "- 每行时间跨度尽量不超过 6 秒，一句话太长就在自然停顿处拆成多行。",
        "",
        "输出格式：每句一行，写 `[MM:SS.mmm --> MM:SS.mmm] 原文`",
        "例：`[00:12.340 --> 00:15.800] 你怎么还在这儿`",
        f"第一行先写这段音频的语言：`#lang: xx`（ISO 639-1 两位小写字母，如 ja / en / zh）。",
        f"这段音频里完全没有人说话时，只输出一行：{NO_SPEECH}",
        "不要输出解释、代码块标记或任何多余内容。",
    ]
    if language:
        lines.append(f"这段音频的语言是 {language}，请只用该语言书写。")
    if proper_nouns.strip():
        lines.append(
            "片中反复出现的人名/专有名词（听到时请按这里的写法）："
            + proper_nouns.strip()
        )
    if complaint:
        lines.append(
            f"上一次的回答不符合要求：{complaint}。请重新输出，"
            f"每行必须是 [MM:SS.mmm --> MM:SS.mmm] 文字，时间不得超过 {seconds:.2f} 秒。"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------- parsing


def _seconds(value: str) -> Optional[float]:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for number in numbers:
        total = total * 60 + number
    return total


def says_no_speech(reply: str) -> bool:
    """Did the model report an empty window? Its own words, exactly."""
    body = "\n".join(
        line for line in reply.splitlines()
        if not _FENCE_RE.match(line) and not _LANG_RE.match(line)
    )
    squashed = "".join(body.split()).upper().strip(".。")
    return squashed in ("NOSPEECH", "无", "NOSPEECH.", "NONE")


def parse_reply(reply: str) -> tuple[str, List[Cue]]:
    """(language, cues) out of one window's reply. Times stay relative."""
    language = ""
    cues: List[Cue] = []
    for raw in reply.splitlines():
        if _FENCE_RE.match(raw):
            continue
        tag = _LANG_RE.match(raw)
        if tag:
            language = tag.group(1).lower()[:2]
            continue
        found = _CUE_RE.match(raw)
        if not found:
            continue
        start, end = _seconds(found.group("a")), _seconds(found.group("b"))
        text = found.group("text").strip()
        if start is None or end is None or not text:
            continue
        cues.append((start, end, text))
    return language, cues


def looks_like_prose(reply: str, cues: Sequence[Cue]) -> bool:
    """A well-formed paragraph where a transcript was asked for.

    That is what a content refusal looks like: polite, fluent, and nothing
    like a format error — which is what the user would otherwise go hunting.
    """
    return (not cues and not says_no_speech(reply)
            and len(reply.strip()) >= PROSE_MIN_CHARS)


# -------------------------------------------------------------- validation


def audio_reached_the_model(usage, seconds: float) -> Optional[bool]:
    """Did the request actually carry audio? None when usage is absent.

    The only mechanical answer to "did the relay pass the audio part
    through". Measured: 25 tokens/s, so a 300s window reports ~7,772 prompt
    tokens; stripped of its audio the same request reports ~272 — and the
    model still answers as if it had listened.
    """
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if not prompt_tokens:
        return None
    return prompt_tokens >= MIN_AUDIO_TOKEN_SHARE * AUDIO_TOKENS_PER_SECOND * seconds


# ------------------------------------------------------- where speech starts
# This engine has no word-level timestamps, so segmenter._trustworthy_start
# cannot reach its cues: the model gives one span per utterance and the
# line splitter divides it by character count. Measured on a Japanese DVD
# source, 89 of its cues start on audio quieter than -45 dBFS (whisper: 57),
# and where the two engines disagree about a start by more than 1.5s the
# audio sides with whisper 20 times against 11.
#
# The fix is local and deterministic: look at the cue's own audio and move
# the start forward to where sound actually begins. These two functions are
# the measurement half, kept pure so the comparison tool in backend/tools
# reports the very quantity the engine acts on.

ONSET_FRAME = 0.02  # a syllable runs ~80ms; RMS over 20ms is still stable
# "within 12 dB of known speech counts as speech" — the same constant, with
# the same meaning, that the coverage report uses to decide whether silero
# missing a stretch is suspicious. Measured spread: a misplaced start sits
# >=25 dB under its own cue, a soft first syllable 6-8 dB under, VHS floor
# noise 20+ dB under. 12 separates all three.
ONSET_DB = asr.LEVEL_SUSPICIOUS_DB
# ...measured against the cue's *loud* end, not its median. A cue that
# begins a second early is mostly silence, so its median IS the room tone
# and a median-based threshold waves the room tone through: measured on a
# Japanese DVD source that definition called 80% of the API's starts
# perfectly placed, and the defect it is meant to find disappeared. Against
# P90 the same cues show a median lag of 0.18s and a p90 of 0.62s, against
# whisper's 0.20s / 0.52s — the gap the comparison is looking for.
ONSET_REFERENCE_Q = 0.9
ONSET_SUSTAIN = (2, 4)  # >=2 of the next 4 frames must hold: a click is one frame
ONSET_LEAD = 0.10       # land this far before the first sound, not on top of it
# Past this the cue is not late, it is misplaced, and a snap would be a
# guess. Same number as segmenter.GAP_BREAK / asr.COVERED_GAP_BRIDGE, which
# both mean "longer than this is not a pause". Over the limit we refuse to
# move rather than clamp to it.
MAX_START_MOVE = 1.5
SNAP_SILENT_DB = -60.0  # a cue whose own median is under this has nothing to align to


def frame_levels(audio, start: float, end: float, frame: float = ONSET_FRAME):
    """RMS in dBFS per *frame* seconds over ``audio[start:end]``.

    asr.level_profile is one value per second — far too coarse to find the
    onset inside a cue — so this is the fine-grained twin, with the same
    -140 dBFS floor for digital silence.
    """
    import numpy as np

    step = max(1, int(round(frame * SAMPLE_RATE)))
    first = max(0, int(round(start * SAMPLE_RATE)))
    last = min(len(audio), int(round(end * SAMPLE_RATE)))
    usable = (last - first) - (last - first) % step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    block = np.asarray(audio[first:first + usable], dtype=np.float32).reshape(-1, step)
    rms = np.sqrt(np.mean(np.square(block), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-7))


def onset_reference(levels) -> float:
    """How loud this cue gets when it is speaking, in dBFS.

    One number, so the engine and the comparison tool cannot drift apart
    on the definition.
    """
    if len(levels) == 0:
        return -140.0
    ordered = sorted(float(db) for db in levels)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ONSET_REFERENCE_Q))]


def speech_onset(levels, threshold_db: float,
                 sustain: tuple[int, int] = ONSET_SUSTAIN) -> Optional[int]:
    """Index of the first frame where sound starts and keeps going.

    A door click, a tape dropout or one frame of hiss all reach the
    threshold for exactly one frame; requiring *need* of the following
    *window* frames to hold makes them cost nothing. Returns None when the
    stretch never rises to the threshold.
    """
    need, window = sustain
    total = len(levels)
    for index in range(total):
        if levels[index] < threshold_db:
            continue
        after = levels[index + 1:index + 1 + window]
        enough = min(need, len(after))
        if sum(1 for db in after if db >= threshold_db) >= enough:
            return index
    return None


@dataclass
class SnapStats:
    """What ``snap_starts`` did, for the job log and the debug log."""

    total: int = 0
    moved: List[float] = field(default_factory=list)
    silent: int = 0         # nothing above the noise floor in the whole cue
    on_speech: int = 0      # already starts on sound: left alone
    beyond_bound: int = 0   # sound starts more than MAX_START_MOVE later
    no_room: int = 0        # moving would leave the cue too short to read

    def describe(self) -> str:
        if not self.moved:
            return f"0/{self.total} 行"
        ordered = sorted(self.moved)
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
        return (f"{len(self.moved)}/{self.total} 行 "
                f"中位 +{statistics.median(ordered):.2f}s "
                f"p90 +{p90:.2f}s 最大 +{max(ordered):.2f}s")


def snap_start(cue: Cue, audio) -> tuple[Cue, float, str]:
    """Move one cue's start forward to where its sound actually begins.

    Forward only, and never past ``MAX_START_MOVE``: beyond that the cue is
    not late, it is in the wrong place, and moving it would be a guess
    dressed up as a correction. The end is deliberately untouched — the
    defect was only ever measured at the start, a relative threshold would
    clip breathy endings and Japanese sentence-final particles, and a late
    end costs nothing while an early one costs reading time.
    """
    start, end, text = cue
    levels = frame_levels(audio, start, end)
    if len(levels) < 2:
        return cue, 0.0, "too_short"
    reference = onset_reference(levels)
    if reference < SNAP_SILENT_DB:
        return cue, 0.0, "silent"
    threshold = reference - ONSET_DB
    index = speech_onset(levels, threshold)
    if index is None:
        return cue, 0.0, "silent"
    if index == 0:
        return cue, 0.0, "on_speech"
    if index * ONSET_FRAME > MAX_START_MOVE:
        return cue, 0.0, "beyond_bound"
    moved_to = start + index * ONSET_FRAME - ONSET_LEAD
    # keep the cue long enough that the segmenter will not pull it back
    limit = end - max(MIN_CUE_SECONDS, 0.5)
    if moved_to > limit:
        if limit <= start:
            return cue, 0.0, "no_room"
        moved_to = limit
    if moved_to <= start:
        return cue, 0.0, "on_speech"
    return (moved_to, end, text), moved_to - start, "moved"


def snap_starts(cues: Sequence[Cue], audio) -> tuple[List[Cue], SnapStats]:
    """``snap_start`` over a globally sorted list of cues.

    Order matters: this runs after every window has come back and the cues
    are in one sorted sequence, so the neighbour relations hold. Because a
    start only ever moves later, and ``validate`` already guarantees
    ``start < end`` within a window, no cue can be moved across another.
    """
    out: List[Cue] = []
    stats = SnapStats(total=len(cues))
    for cue in cues:
        moved_cue, distance, why = snap_start(cue, audio)
        out.append(moved_cue)
        if why == "moved":
            stats.moved.append(distance)
        elif why == "silent":
            stats.silent += 1
        elif why == "on_speech":
            stats.on_speech += 1
        elif why == "beyond_bound":
            stats.beyond_bound += 1
        elif why == "no_room":
            stats.no_room += 1
    return out, stats


def validate(
    cues: Sequence[Cue], seconds: float, speech: float
) -> tuple[List[Cue], List[str], List[str]]:
    """(kept, fatal, notes) for one window's cues, times still relative.

    *fatal* non-empty means the window has to be retried or split; *notes*
    are repairs worth recording but not worth another request.
    """
    fatal: List[str] = []
    notes: List[str] = []

    ordered = list(cues)
    inversions = sum(
        1 for a, b in zip(ordered, ordered[1:]) if b[0] < a[0]
    )
    if ordered and inversions / len(ordered) > MAX_INVERSION_SHARE:
        fatal.append(f"{inversions}/{len(ordered)} 行时间倒序")
        return [], fatal, notes
    if inversions:
        notes.append(f"{inversions} 行时间倒序，已重新排序")
        ordered.sort(key=lambda c: c[0])

    kept: List[Cue] = []
    outside = 0
    for start, end, text in ordered:
        if not asr.has_content(text):
            continue
        if start < -0.05 or start >= seconds + TIME_SLACK:
            outside += 1
            continue
        start = max(0.0, start)
        if end > seconds:
            if end > seconds + TIME_SLACK:
                outside += 1
                continue
            end = seconds
        if end - start < MIN_CUE_SECONDS:
            end = min(start + MIN_CUE_SECONDS, seconds)
        if end <= start:  # nothing left of it after clamping to the window
            outside += 1
            continue
        if end - start > 30.0:
            notes.append(f"一行长达 {end - start:.0f}s，断句将由 segmenter 按字数切")
        if kept and start < kept[-1][1]:
            overlap = kept[-1][1] - start
            if overlap > 0.3:
                notes.append(f"{overlap:.1f}s 重叠，已切齐")
            previous = kept[-1]
            kept[-1] = (previous[0], max(previous[0] + MIN_CUE_SECONDS * 0.5, start),
                        previous[2])
        kept.append((start, end, text))

    if outside:
        notes.append(f"{outside} 行落在本段之外，已丢弃")

    repeats = sum(1 for a, b in zip(kept, kept[1:]) if a[2] == b[2])
    if kept and repeats / len(kept) > MAX_REPEAT_SHARE:
        fatal.append(f"{repeats}/{len(kept)} 行与上一行完全相同（疑似解码陷环）")
        return [], fatal, notes

    if not kept and speech >= MIN_SPEECH_FOR_TEXT:
        fatal.append(f"本段 silero 认定有 {speech:.0f}s 语音，却一行都没转出")
        return [], fatal, notes

    # Truncation is checked by the caller (unclaimed_tail), which has the
    # speech intervals: `max_tokens` is ignored and `finish_reason` always
    # says "stop", so content against the VAD is the only tell there is.
    return kept, fatal, notes


def unclaimed_tail(
    intervals: Sequence[tuple[float, float]],
    window: tuple[float, float],
    last_end: float,
) -> float:
    """Silero-confirmed speech after the last transcribed line, in seconds."""
    start, end = window
    return speech_inside(intervals, start + last_end, end)


def looks_truncated(tail: float, speech: float) -> bool:
    """Was the reply cut off? The only question the envelope cannot answer.

    `max_tokens` is ignored and `finish_reason` says "stop" either way, so
    the tell is content: speech the VAD is sure of, past where the
    transcript stopped, and enough of it to outweigh silero's habit of
    hearing music as speech.
    """
    return tail >= max(TRUNCATION_MIN_TAIL, TRUNCATION_TAIL_SHARE * speech)


def speech_end_within(
    intervals: Sequence[tuple[float, float]], start: float, end: float
) -> float:
    """Where the last speech in this window ends, relative to its start."""
    ends = [min(e, end) for s, e in intervals if s < end and e > start]
    return max(ends) - start if ends else 0.0


def clock_error(speech_end: float, last_cue_end: float) -> float:
    """How fast the model's clock ran in this window, in ms per second.

    Positive means it reported events earlier than they happened. Measured
    this way rather than by matching each line to the nearest VAD boundary,
    because that matching needs a radius — and a clock off by more than the
    radius (exactly the case worth catching) stops matching at all.
    """
    if speech_end <= 0:
        return 0.0
    return (speech_end - last_cue_end) / speech_end * 1000.0


# ----------------------------------------------------------- diagnostics


DRIFT_MIN_LEVER = 30.0   # a window needs this much speech to measure a rate
DRIFT_MIN_WINDOWS = 4    # ...and this many windows before it may fail a job


def drift_rate(records: Sequence[dict]) -> tuple[Optional[float], int]:
    """(ms of error per second of audio, how many windows it came from).

    Positive = the transcript ends earlier than the audio's last speech,
    i.e. a clock running fast. Negative = it ends later, which happens
    whenever the model hears speech silero missed and is not a defect.

    A least-squares fit through the origin of "seconds missing at the end"
    against "seconds of audio to be wrong about", so a long window counts
    for more than a short one and no single window decides. That matters:
    silero hears music as speech, so a scene ending on score looks exactly
    like a transcript that stopped early. A clock that genuinely runs fast
    does it in every window, and the fit sees that.
    """
    pairs = [
        (record["speech_end"], record["speech_end"] - record["last_cue_end"])
        for record in records
        if record.get("speech_end", 0.0) >= DRIFT_MIN_LEVER
        and record.get("last_cue_end") is not None
    ]
    if len(pairs) < 2:
        return None, len(pairs)
    lever = sum(s * s for s, _ in pairs)
    if not lever:
        return None, len(pairs)
    return sum(s * e for s, e in pairs) / lever * 1000.0, len(pairs)


def boundary_errors(
    segments: Sequence[Segment], intervals: Sequence[tuple[float, float]]
) -> List[tuple[float, float]]:
    """(time, signed distance to the nearest silero boundary) per segment."""
    edges = sorted([s for s, _ in intervals] + [e for _, e in intervals])
    if not edges:
        return []
    out: List[tuple[float, float]] = []
    for segment in segments:
        nearest = min(edges, key=lambda edge: abs(edge - segment.start))
        if abs(nearest - segment.start) <= 5.0:
            out.append((segment.start, segment.start - nearest))
    return out


def correlation(pairs: Sequence[tuple[float, float]]) -> Optional[float]:
    """Pearson r — the same check translator._verify_alignment makes, moved
    to the audio side: per window, speech seconds against characters. The
    one failure every per-window check passes is a window whose offset was
    applied wrongly, and this is what sees it."""
    if len(pairs) < ALIGNMENT_MIN_WINDOWS:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    dx = sum((p[0] - mx) ** 2 for p in pairs) ** 0.5
    dy = sum((p[1] - my) ** 2 for p in pairs) ** 0.5
    if not dx or not dy:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pairs) / (dx * dy)


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# ------------------------------------------------------------- entry point


def transcribe(
    wav_path: str,
    settings: ASRSettings,
    llm: LLMSettings,
    language: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    log: Optional[LogFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    network: Optional[NetworkSettings] = None,
    debug=None,
    client=None,  # injectable for tests
    usage: Optional[dict] = None,
) -> tuple[List[Segment], str]:
    """Transcribe *wav_path* with the audio model; (segments, language)."""
    from faster_whisper.audio import decode_audio

    from app.services.translator import make_audio_client

    audio = decode_audio(wav_path, sampling_rate=SAMPLE_RATE)
    duration = len(audio) / float(SAMPLE_RATE)
    model = llm.audio_model.strip() or llm.model
    endpoint = llm.audio_base_url.strip() or llm.base_url
    if client is None:
        client = make_audio_client(llm, network)

    cuts = cut_intervals(audio)
    windows, forced = plan_windows(cuts, duration, settings.api_window_seconds)
    # The job's own VAD settings, for everything that is measurement rather
    # than cutting — so the coverage figure stays comparable across runs.
    intervals = asr.speech_intervals_of(audio, settings)

    if log:
        log(f"API 语音识别：{model} @ {endpoint}")
        log(
            f"音频 {_clock(duration)}，切成 {len(windows)} 段"
            f"（目标 {settings.api_window_seconds:.0f}s，格式 {settings.api_audio_format}，"
            f"并发 {settings.api_concurrency}）"
            + (f"，其中 {forced} 处无静音可切、按时长硬切" if forced else "")
        )
        log("本引擎不使用二次识别与其复核：窗口已覆盖整条时间轴，"
            "设置里的「二次识别」开关本次无效")
        if not cuts:
            log("⚠ silero 在整条音轨上没有找到语音，已按固定时长均分——"
                "并不因此跳过任何一段音频")

    tally = usage if usage is not None else {}
    for key in ("calls", "prompt", "completion", "empty"):
        tally.setdefault(key, 0)
    tally_lock = threading.Lock()
    prose_run = 0
    languages: List[str] = []
    per_window: List[tuple[float, float, int]] = []  # speech, chars, forced-index
    records: List[dict] = []

    def one_window(window: tuple[float, float]) -> dict:
        """Two attempts at one window, three if the audio format is
        changed. Raises RefusalError; returns a dict with `cues`
        (absolute) or `fatal` when it could not be verified."""
        nonlocal prose_run
        start, end = window
        seconds = end - start
        speech = speech_inside(intervals, start, end)
        fmt = settings.api_audio_format
        data = encode_window(audio, start, end, fmt)
        part = audio_part(data, fmt)
        where = f"[{_clock(start)}–{_clock(end)}]"
        if len(part["input_audio"]["data"]) > MAX_INLINE_BYTES:
            return {"window": window, "fatal": "请求超过内联上限，需要更短的窗口",
                    "attempts": 0, "reply": ""}

        # `complaint` is model-facing — it becomes "your last answer was
        # wrong because …" in the next prompt — while `failure` is why
        # this window failed, for the log and the ladder. An empty reply
        # sets only the second: there was no answer to fault, and a resend
        # whose prompt changed would no longer be a resend.
        complaint = ""
        failure = ""
        empty_billed = 0
        switched = False
        attempt = 0
        allowed = WINDOW_ATTEMPTS
        while attempt < allowed:
            attempt += 1
            if should_cancel and should_cancel():
                raise InterruptedError
            prompt = build_prompt(seconds, language or "",
                                  settings.initial_prompt, complaint)
            try:
                from app.services.translator import (
                    EmptyReplyError, chat_completion, reply_text)

                resp, _ = chat_completion(
                    client, model=model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt}, part,
                    ]}],
                    temperature=0,
                    no_thinking=llm.disable_thinking,
                )
                reply = reply_text(resp)
            except EmptyReplyError as exc:
                # A 200 carrying no candidate at all. Named for what it is,
                # because the wording this replaces sent people off to
                # change models over something the model never did.
                #
                # Three states, and the middle one matters: no usage block
                # is "cannot tell", not "the audio was lost". And only the
                # billed kind is the known fault — a server answering
                # "model not loaded" arrives here without choices too, and
                # for that one the advice dropped below (check the model,
                # check the endpoint) is the right advice. A server that
                # never processed the clip does not charge 25 tokens a
                # second for it, so the usage block tells them apart.
                failure = str(exc)
                carried = audio_reached_the_model(exc.usage, seconds)
                heard = {True: "音频已到服务端并已计费",
                         False: "服务端只收到文字量的 token，音频可能被中转丢了",
                         None: "服务端没给 usage，无法判断音频是否送达"}[carried]
                if carried is True:
                    empty_billed += 1
                with tally_lock:
                    # a request that was made, and for the billed kind, paid
                    # for: those prompt tokens are the audio, with nothing
                    # returned for them
                    tally["calls"] += 1
                    tally["empty"] += 1 if carried is True else 0
                    tally["prompt"] += getattr(exc.usage, "prompt_tokens", 0) or 0
                if log:
                    log(f"⚠ {where} 第 {attempt} 次模型返回了空回复"
                        f"（{heard}；{fmt} {len(data) // 1024}KB）")
                other = OTHER_FORMAT.get(fmt)
                if other and not switched:
                    retry = encode_window(audio, start, end, other)
                    candidate = audio_part(retry, other)
                    if len(candidate["input_audio"]["data"]) <= MAX_INLINE_BYTES:
                        fmt, data, part = other, retry, candidate
                        switched = True
                        # exactly one try for the new format, never a spare:
                        # measured, the refusal is deterministic per clip and
                        # format (ten of ten), so a plain resend buys nothing
                        allowed = max(allowed, attempt + 1)
                        if log:
                            log(f"  {where} 换成 {other} 重发同一段"
                                f"（{len(data) // 1024}KB）")
                continue
            except Exception as exc:  # noqa: BLE001 — the server's own words
                complaint = failure = str(exc)
                if log:
                    log(f"⚠ {where} 第 {attempt} 次请求失败：{exc}"
                        f"（音频 {len(data) // 1024}KB）")
                continue

            with tally_lock:
                tally["calls"] += 1
                usage_obj = getattr(resp, "usage", None)
                tally["prompt"] += getattr(usage_obj, "prompt_tokens", 0) or 0
                tally["completion"] += getattr(usage_obj, "completion_tokens", 0) or 0

            carried = audio_reached_the_model(getattr(resp, "usage", None), seconds)
            if carried is False:
                complaint = failure = "请求里似乎没有音频"
                if log:
                    log(f"⚠ {where} 服务端只收到文字的 token 量"
                        f"（{getattr(getattr(resp, 'usage', None), 'prompt_tokens', 0)}，"
                        f"应约 {int(AUDIO_TOKENS_PER_SECOND * seconds)}）——"
                        "中转可能把音频丢了")
                continue

            lang, cues = parse_reply(reply)
            if lang:
                with tally_lock:
                    languages.append(lang)
            if looks_like_prose(reply, cues):
                with tally_lock:
                    prose_run += 1
                    run = prose_run
                if run >= PROSE_RUN:
                    raise RefusalError(
                        f"连续 {run} 段音频得到的是说明文字而不是转写，"
                        "模型可能拒绝转写本片内容；可改用本地识别引擎或换一个模型"
                    )
                complaint = failure = "回答的是说明文字，不是逐句转写"
                if log:
                    log(f"⚠ {where} 收到的是说明文字而不是转写")
                continue
            with tally_lock:
                prose_run = 0

            if not cues and says_no_speech(reply):
                # Believed, always — even when silero insists there is
                # speech here. Measured: asking again over a stretch of
                # score made the model produce fifteen lines of invented
                # dialogue where its first answer had been "no speech",
                # and silero was the one hearing things. Invented lines are
                # the worse failure (the whole reason vet.py exists), so
                # the disagreement is recorded for a human and obeyed by
                # nobody — the same standing whisper's ♪ has in lyrics.py.
                if speech >= DISPUTE_SPEECH and log:
                    log(f"  {where} 模型说这段没有人说话，而 silero 认为有 "
                        f"{speech:.0f}s 语音——采信模型，分歧已记录")
                return {"window": window, "cues": [], "attempts": attempt,
                        "reply": reply, "speech": speech, "no_speech": True,
                        "disputed": speech >= DISPUTE_SPEECH}

            kept, fatal, notes = validate(cues, seconds, speech)
            if kept and not fatal:
                tail = unclaimed_tail(intervals, window, kept[-1][1])
                if looks_truncated(tail, speech):
                    fatal = [f"转写停在 {kept[-1][1]:.0f}s，其后仍有约 {tail:.0f}s 语音"
                             "（疑似截断）"]
            if fatal:
                complaint = failure = "；".join(fatal)
                if log:
                    log(f"⚠ {where} 第 {attempt} 次校验未通过：{complaint}")
                continue
            if notes and log:
                log(f"  {where} {'；'.join(notes)}")
            return {
                "window": window, "attempts": attempt, "reply": reply,
                "speech": speech, "no_speech": False,
                "speech_end": speech_end_within(intervals, start, end),
                "last_cue_end": kept[-1][1],
                "cues": [(start + a, start + b, text) for a, b, text in kept],
            }
        return {"window": window, "fatal": failure or "校验未通过",
                "attempts": attempt, "reply": "",
                "empty_billed": empty_billed}

    # ---- run the windows, splitting the ones that will not verify --------
    pending = deque(windows)
    done_seconds = 0.0
    budget = MAX_CALL_FACTOR * max(1, len(windows))
    collected: List[Cue] = []
    missing: List[tuple[float, float]] = []   # stretches nothing got through
    allowed_missing = MAX_MISSING_SHARE * duration if duration else 0.0
    last_failure = ""   # the server's own words, for whichever exit fires

    def give_up(start: float, end: float, why: str) -> None:
        """Record a stretch as untranscribed, or stop the stage.

        Raises once the gaps outgrow `allowed_missing` — immediately, so a
        systematically refusing endpoint stops costing tokens at the point
        it is recognised rather than after the whole budget is spent.
        """
        missing.append((start, end))
        lost = sum(b - a for a, b in missing)
        if log:
            log(f"⚠ [{_clock(start)}–{_clock(end)}] 放弃这一段（{why}）；"
                f"累计未转写 {lost:.0f}s，占全片 "
                f"{(lost / duration if duration else 0):.1%}")
        if lost > allowed_missing:
            ranges = "、".join(f"{_clock(a)}–{_clock(b)}" for a, b in missing)
            raise RuntimeError(
                f"语音识别有 {lost:.0f}s 没能转出来（{ranges}），"
                f"超过全片的 {MAX_MISSING_SHARE:.0%}——接口在持续拒绝，"
                f"这份转写已经不值得交付（最后一次失败：{why}）；"
                "可稍后重试或改用本地识别引擎"
            )
    pool = ThreadPoolExecutor(max_workers=max(1, settings.api_concurrency))
    inflight: dict = {}
    try:
        while pending or inflight:
            if should_cancel and should_cancel():
                raise InterruptedError
            while pending and len(inflight) < max(1, settings.api_concurrency):
                if tally.get("calls", 0) >= budget:
                    # Out of requests. Same policy as any other stretch we
                    # cannot get through: name what is missing and let the
                    # ceiling decide. The server's own words travel with it
                    # — this is the only place the complaint would survive.
                    empty_all = tally.get("empty", 0) >= tally.get("calls", 0)
                    why = (f"请求数已达上限（{budget} 次）"
                           + ("，全部收到空回复" if empty_all else "")
                           + (f"，最后一次：{last_failure}" if last_failure else ""))
                    if log:
                        log(f"⚠ 请求数已达上限（{budget} 次），仍有 {len(pending)} 段未完成")
                    while pending:
                        give_up(*pending.popleft(), why)
                    break
                window = pending.popleft()
                inflight[pool.submit(one_window, window)] = window
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for future in finished:
                window = inflight.pop(future)
                result = future.result()  # re-raises InterruptedError/RefusalError
                start, end = window
                if "fatal" in result:
                    last_failure = result["fatal"]
                    if end - start > WINDOW_FLOOR * 2:
                        middle = split_point(cuts, start, end)
                        pending.appendleft((middle, end))
                        pending.appendleft((start, middle))
                        if log:
                            log(f"  [{_clock(start)}–{_clock(end)}] 拆成两段再试")
                        continue
                    # Out of ladder on a stretch this short. Let the film
                    # keep its other ninety minutes; give_up decides whether
                    # the gaps have grown past what is worth delivering.
                    give_up(start, end,
                            "连续收到空回复，换音频格式与拆短窗口都没能绕开"
                            if result.get("empty_billed", 0) >= result.get("attempts", 1)
                            else result["fatal"])
                    done_seconds += end - start
                    if progress:
                        progress(min(done_seconds / duration, 1.0) if duration else 1.0)
                    continue
                collected.extend(result["cues"])
                records.append(result)
                per_window.append((
                    result.get("speech", 0.0),
                    float(sum(len(text) for _, _, text in result["cues"])),
                    0,
                ))
                done_seconds += end - start
                if progress:
                    progress(min(done_seconds / duration, 1.0) if duration else 1.0)
                if log:
                    log(f"  [{_clock(start)}–{_clock(end)}] "
                        + (f"{len(result['cues'])} 行" if result["cues"] else "无人说话")
                        + (f"（第 {result['attempts']} 次通过）"
                           if result["attempts"] > 1 else ""))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # Starts get corrected here, not per window: the cues have to be in one
    # globally sorted sequence before a neighbour relation means anything.
    ordered, snapped = snap_starts(sorted(collected, key=lambda c: c[0]), audio)
    segments = [Segment(start=a, end=b, text=text) for a, b, text in ordered]
    if log and snapped.moved:
        log(f"已按音频把 {snapped.describe()} 的起点后移到实际发声处")

    detected = language or ""
    if not detected and languages:
        detected = statistics.mode(languages)

    if log:
        log(f"识别完成：{len(segments)} 段，共 {tally.get('calls', 0)} 次请求"
            f"（输入 {tally.get('prompt', 0)} tokens，输出 {tally.get('completion', 0)}）")
        if missing:
            # A gap that only showed up in a line scrolled past an hour ago
            # is a silent gap. It gets said again, at the end, in full.
            lost = sum(b - a for a, b in missing)
            log(f"⚠ 有 {len(missing)} 段共 {lost:.0f}s 没能转出来"
                f"（占全片 {(lost / duration if duration else 0):.1%}，"
                f"在容许的 {MAX_MISSING_SHARE:.0%} 以内）——"
                "这些时间段的字幕是空的，不是那里没有人说话：")
            for a, b in missing:
                log(f"    {_clock(a)}–{_clock(b)}（{b - a:.0f}s）")
        log("注：本引擎没有词级时间戳，下面的覆盖率按 segment 区间计量，"
            "读数偏乐观，与本地引擎的数字不可直接比较")
    # Coverage first, diagnostics second — deliberately. Every check in
    # `_report` is measured against silero, so whether silero worked at all
    # has to be established (and said out loud) before its readings are used
    # to accuse the transcript. The intervals are handed over rather than
    # recomputed: one pass of silero per job.
    verdict = asr._report_coverage(audio, settings, segments, log, debug,
                                   intervals=intervals)
    _report(segments, cuts, per_window, records, forced, log, debug, snapped,
            vad_blind=bool(verdict is not None and verdict.vad_blind))
    return segments, detected


def _report(segments, cuts, per_window, records, forced, log, debug,
            snapped: Optional[SnapStats] = None, vad_blind: bool = False) -> None:
    """The cross-check against silero: diagnostics, plus the drift interlock.

    Modelled on pipeline._debug_lyric_agreement — an independent second
    opinion that is never obeyed. Silero cannot hear dialogue under music
    and this engine can, so a disagreement is as likely to be silero's
    fault as the model's; the only thing acted on here is a clock that
    slides, which is not a matter of opinion.
    """
    # against the fine-grained cut intervals, not the job's VAD settings:
    # at min_silence=2000ms silero reports a handful of boundaries for a
    # whole film and every line is "far" from one, which says nothing
    errors = boundary_errors(segments, cuts)
    slope, measured_on = drift_rate(records)
    deltas = sorted(abs(e) for _, e in errors)
    density = [chars / speech for speech, chars, _ in per_window
               if speech >= 1.0 and chars]
    pairs = [(speech, chars) for speech, chars, _ in per_window]
    r = correlation(pairs)

    if debug is not None and getattr(debug, "enabled", False):
        _write_debug(debug, records, forced, slope, measured_on,
                     deltas, density, r, snapped, vad_blind)
    # Every check below compares this engine against silero. On a source
    # silero cannot hear (asr.vad_blindness), all three fire at once and all
    # three are wrong — measured on a VHS capture: 40.4 chars/speech-second,
    # r=+0.61, -266 ms/s, against a VAD that found 3.5% of the film. The
    # numbers stay in the debug log; what is suppressed is calling them
    # faults of the transcript. The coverage report has already said, in
    # the job log, that the yardstick is the thing that broke.
    if density and log and not vad_blind:
        middle = statistics.median(density)
        strange = [d for d in density
                   if d > middle * DENSITY_BAND or d * DENSITY_BAND < middle]
        if strange:
            log(f"⚠ {len(strange)} 段的字数/语音秒比与全片中位数（{middle:.1f}）"
                "相差三倍以上，值得在调试日志里看一眼")
    if r is not None and r < ALIGNMENT_MIN_CORRELATION and log and not vad_blind:
        log(f"⚠ 各段「语音秒数 vs 转出字数」相关性只有 {r:+.2f}"
            f"（阈值 {ALIGNMENT_MIN_CORRELATION}）——时间轴可能整体错位")
    if slope is None:
        return
    # One-sided on purpose. A POSITIVE rate means the transcript stops
    # earlier and earlier relative to the audio — the documented failure,
    # a clock running fast. A negative one means the model's last line
    # lands *after* silero's last speech, which is not a defect but this
    # engine's whole advantage: it hears dialogue under music that silero
    # scores as silence. Measured on a real film: −38 ms/s with the line
    # starts sitting 0.40s (median) from the nearest VAD boundary and a
    # window-alignment correlation of +0.93 — i.e. the timestamps were
    # fine and only the VAD was deaf. Warning on that would cry wolf on
    # every film with a score.
    if slope > DRIFT_FAIL_MS and measured_on >= DRIFT_MIN_WINDOWS and vad_blind:
        # The interlock rests on "silero's last speech is where speech
        # ends". On a blind source that premise has just been measured
        # false, so the slope is not evidence of anything — it is reported
        # and not acted on, the same rule the ♪ cross-check follows.
        if log:
            log(f"⚠ 时钟漂移测得 {slope:+.0f} ms/s，但本片 silero 基本失聪、"
                "这个斜率正是以它为基准算出来的，无法判定，本次不据此中止识别")
        return
    if slope > DRIFT_FAIL_MS and measured_on >= DRIFT_MIN_WINDOWS:
        raise RuntimeError(
            f"模型返回的时间戳与音频持续对不上（{measured_on} 段的中位数 "
            f"{slope:+.0f} ms/s，一段 5 分钟就会偏 {abs(slope) * 0.3:.0f} 秒）。"
            "可能是这个模型的时钟会漂，也可能是整片大段没有人声、模型凭空编了台词；"
            "无论哪种，这样的时间轴都不能用来做字幕——请改用本地识别引擎，或换一个模型"
        )
    if slope > DRIFT_WARN_MS and log and not vad_blind:
        log(f"⚠ 时间戳有系统性漂移（各段 {slope:+.0f} ms/s），"
            "建议抽查片尾几条字幕的时间是否对得上"
            + ("；样本太少，本次不据此中止" if measured_on < DRIFT_MIN_WINDOWS else ""))


def _write_debug(debug, records, forced, slope, measured_on,
                 deltas, density, r, snapped=None, vad_blind=False) -> None:
    """The cross-check, verbatim replies and all. Never allowed to fail a
    job — a diagnostic that takes the transcript down with it is worse than
    no diagnostic (asr._report_coverage has the same guard)."""
    try:
        debug.section("LLM 转写 vs silero VAD（只作参考，不改时间戳）")
        debug.kv("窗口", f"{len(records)}（硬切 {forced}，"
                        f"无人说话 {sum(1 for x in records if x.get('no_speech'))}，"
                        f"重试 {sum(1 for x in records if x.get('attempts', 1) > 1)}）")
        if deltas:
            debug.kv("起点与最近 VAD 边界偏差",
                     f"中位 {statistics.median(deltas):.2f}s  "
                     f"p90 {deltas[int(len(deltas) * 0.9) - 1]:.2f}s")
        debug.kv("时钟漂移（各段中位）",
                 "样本不足" if slope is None
                 else f"{slope:+.1f} ms/s（{measured_on} 段）")
        if density:
            debug.kv("每窗 字符/语音秒",
                     f"中位 {statistics.median(density):.1f}"
                     f"（{min(density):.1f}–{max(density):.1f}）")
        debug.kv("语音秒 vs 字符数 相关性 r",
                 "样本不足" if r is None else f"{r:+.2f}")
        if vad_blind:
            debug.kv("VAD 失聪", "是——以上各项以 silero 为基准，本次只作参考")
        if snapped is not None:
            debug.kv("起点按音频后移", snapped.describe()
                     + (f"；超出 {MAX_START_MOVE}s 上限未动 {snapped.beyond_bound} 行"
                        if snapped.beyond_bound else "")
                     + (f"，整条无声 {snapped.silent} 行" if snapped.silent else "")
                     + (f"，已在发声处 {snapped.on_speech} 行" if snapped.on_speech else ""))
        # Millisecond precision on purpose: the replies inside carry times
        # relative to the window, so this heading is the only anchor that
        # turns them back into absolute ones. At MM:SS every cue in the
        # debug log was good to about a second.
        from app.core.debuglog import fmt_time

        debug.section("语音识别原始输出（API 引擎）")
        for record in sorted(records, key=lambda x: x["window"][0]):
            start, end = record["window"]
            debug.line(f"\n--- [{fmt_time(start)}–{fmt_time(end)}] "
                       f"{len(record.get('cues', []))} 行 "
                       f"第 {record.get('attempts', 1)} 次通过 ---")
            debug.block(f"{fmt_time(start)}–{fmt_time(end)}", record.get("reply", ""))
    except Exception:  # noqa: BLE001 — diagnostics must never fail a job
        pass
