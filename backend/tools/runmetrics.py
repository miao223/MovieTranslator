"""The numbers two runs can be judged on, and nothing else.

Every figure here was computed at least once by a throwaway script while
comparing the local and API engines on three films; this is the same
arithmetic with the definitions pinned down, so that the next comparison
measures the same thing.

Two rules the module keeps:

* the audio-side measurements call into ``app.services`` rather than
  reimplementing them — ``onset_delays`` uses ``asr_api.frame_levels`` and
  ``asr_api.speech_onset``, which is exactly what the engine acts on, and
  the VAD picture uses ``asr.speech_intervals_of`` at its fixed reference
  threshold, which is what every previous coverage figure used;
* nothing here decides pass or fail. ``compare_runs`` does that.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Span = Tuple[float, float]

# ------------------------------------------------------------- interval math


def merge(spans: Sequence[Span], bridge: float = 0.0) -> List[Span]:
    out: List[List[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in spans):
        if out and start <= out[-1][1] + bridge:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def subtract(a: Sequence[Span], b: Sequence[Span]) -> List[Span]:
    out: List[Span] = []
    for start, end in a:
        cursor = start
        for other_start, other_end in b:
            if other_end <= cursor or other_start >= end:
                continue
            if other_start > cursor:
                out.append((cursor, min(other_start, end)))
            cursor = max(cursor, other_end)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return [(s, e) for s, e in out if e - s > 0.01]


def intersect(a: Sequence[Span], b: Sequence[Span]) -> float:
    """Seconds two interval sets have in common. Both must be merged."""
    total_seconds = 0.0
    index = 0
    for start, end in a:
        while index and b[index - 1][1] > start:
            index -= 1
        while index < len(b) and b[index][1] <= start:
            index += 1
        cursor = index
        while cursor < len(b) and b[cursor][0] < end:
            total_seconds += max(0.0, min(end, b[cursor][1]) - max(start, b[cursor][0]))
            cursor += 1
    return total_seconds


def total(spans: Sequence[Span]) -> float:
    return sum(e - s for s, e in spans)


def spans_of(cues) -> List[Span]:
    return merge([(c.start, c.end) for c in cues])


def clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


# ------------------------------------------------------------- text pairing

_CJK_RE = re.compile(r"[぀-ヿ一-鿿]")
_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def grams(text: str) -> set:
    """Character bigrams for CJK, words for everything else.

    ``words()`` on Japanese returns whole clauses, so two renderings of the
    same line share nothing at all and every pair is lost. Bigrams are the
    standard answer and behave for Latin text too.
    """
    if _CJK_RE.search(text):
        flat = _STRIP_RE.sub("", text)
        return {flat[i:i + 2] for i in range(len(flat) - 1)} or ({flat} if flat else set())
    return set(_WORD_RE.findall(text.lower()))


def pair_by_text(left, right, window: float = 30.0, floor: float = 0.4,
                 least: int = 3) -> List[tuple]:
    """[(left_cue, right_cue, jaccard)] — the same line as both runs saw it.

    Ties break towards the nearer start: a film that says "K9" forty times
    gives forty perfect-but-wrong candidates otherwise, which is where a
    2% mispairing rate came from the first time this was measured.
    """
    pairs = []
    for cue in left:
        mine = grams(cue.text)
        if len(mine) < least:
            continue
        best, score = None, 0.0
        for other in right:
            if abs(other.start - cue.start) > window:
                continue
            theirs = grams(other.text)
            if not theirs:
                continue
            overlap = len(mine & theirs) / len(mine | theirs)
            closer = best is not None and abs(other.start - cue.start) < abs(best.start - cue.start)
            if overlap > score or (overlap == score and closer):
                best, score = other, overlap
        if best is not None and score >= floor:
            pairs.append((cue, best, score))
    return pairs


# --------------------------------------------------------------- the audio


def rms_db(audio, start: float, seconds: float) -> float:
    """dBFS over *seconds* of audio from *start*."""
    from app.services import asr_api

    levels = asr_api.frame_levels(audio, start, start + seconds)
    if len(levels) == 0:
        return -140.0
    return float(statistics.median(levels))


# A start this far ahead of the sound is sitting in silence, not merely
# early. asr_api.ONSET_LEAD (0.10s) is where a corrected start is *meant*
# to land, so anything at or near it is right by construction — which is
# why the median stops being informative once snapping is on and the
# verdict is taken from the tail instead.
LATE_START = 0.30


@dataclass
class OnsetStats:
    delays: List[float] = field(default_factory=list)
    silent: int = 0        # the cue has nothing above the noise floor
    unfound: int = 0       # nothing ever reached the threshold inside it
    measured: int = 0

    @property
    def median(self) -> float:
        return statistics.median(self.delays) if self.delays else float("nan")

    @property
    def p90(self) -> float:
        return percentile(self.delays, 0.9) if self.delays else float("nan")

    @property
    def late_share(self) -> float:
        """Share of starts that land in silence rather than on the sound."""
        if not self.delays:
            return float("nan")
        return sum(1 for d in self.delays if d > LATE_START) / len(self.delays)

    def describe(self) -> str:
        if not self.delays:
            return "n/a"
        return (f"中位 {self.median:.2f}s  p90 {self.p90:.2f}s  "
                f"落在静音里(>{LATE_START:.1f}s) {self.late_share:.1%}  "
                f"(n={self.measured}，无声 {self.silent}，未找到起声 {self.unfound})")


def onset_delays(cues, audio) -> OnsetStats:
    """How long after each cue starts does sound actually begin.

    The threshold is relative to the cue's own loud end (``onset_reference``),
    so a quiet scene is judged against itself rather than against the film,
    and a cue that is mostly silence is still judged against its speech.
    This is the quantity ``asr_api.snap_start`` moves, measured the same way.
    """
    from app.services import asr_api

    stats = OnsetStats()
    for cue in cues:
        levels = asr_api.frame_levels(audio, cue.start, cue.end)
        if len(levels) < 2:
            continue
        reference = asr_api.onset_reference(levels)
        if reference < asr_api.SNAP_SILENT_DB:
            stats.silent += 1
            continue
        index = asr_api.speech_onset(levels, reference - asr_api.ONSET_DB)
        if index is None:
            stats.unfound += 1
            continue
        stats.measured += 1
        stats.delays.append(index * asr_api.ONSET_FRAME)
    return stats


@dataclass
class Arbitration:
    left: int = 0
    right: int = 0
    tie: int = 0
    disputed: int = 0

    def describe(self, left_name: str, right_name: str) -> str:
        if not self.disputed:
            return "无争议起点"
        return (f"{left_name} {self.left} : {self.right} {right_name}"
                f"（平手 {self.tie} / 共 {self.disputed}）")


def arbitrate(pairs, audio, apart: float = 1.5, probe: float = 0.6,
              margin_db: float = 6.0) -> Arbitration:
    """Where two runs disagree about a start, ask the audio which is right.

    Whoever's start lands on sound wins; *margin_db* keeps a coin-flip from
    counting as a win. Only starts at least *apart* seconds apart are put
    to the question — below that the disagreement is not about which
    utterance it is.
    """
    verdict = Arbitration()
    for left, right, _ in pairs:
        if abs(left.start - right.start) < apart:
            continue
        verdict.disputed += 1
        left_db = rms_db(audio, left.start, probe)
        right_db = rms_db(audio, right.start, probe)
        if left_db - right_db >= margin_db:
            verdict.left += 1
        elif right_db - left_db >= margin_db:
            verdict.right += 1
        else:
            verdict.tie += 1
    return verdict


@dataclass
class VadPicture:
    duration: float = 0.0
    intervals: List[Span] = field(default_factory=list)
    levels: List[float] = field(default_factory=list)
    speech: float = 0.0
    speech_db: float = -140.0
    loud: float = 0.0

    @property
    def vad_share(self) -> float:
        return self.speech / self.loud if self.loud else float("nan")


def vad_picture(audio, settings=None) -> VadPicture:
    """Silero at the fixed reference threshold, plus the level profile.

    The threshold is ``asr.REFERENCE_VAD_THRESHOLD`` whatever the job used
    — that is the whole point of the constant, and the only way two runs
    are comparable.
    """
    from app.models.schemas import ASRSettings
    from app.services import asr

    settings = settings or ASRSettings()
    intervals = asr.speech_intervals_of(audio, settings)
    levels = asr.level_profile(audio)
    duration = len(audio) / 16000.0
    picture = VadPicture(duration=duration, intervals=intervals, levels=levels,
                         speech=total(intervals))
    if levels and intervals:
        spoken = [asr._median_level(levels, s, e) for s, e in intervals if e - s >= 2.0]
        picture.speech_db = statistics.median(spoken) if spoken else asr._median_level(
            levels, intervals[0][0], intervals[0][1])
    elif levels:
        picture.speech_db = percentile(levels, 0.9)
    floor = picture.speech_db - asr.LEVEL_SUSPICIOUS_DB
    picture.loud = float(sum(1 for db in levels if db >= floor))
    return picture


@dataclass
class Coverage:
    speech: float = 0.0
    transcribed: float = 0.0
    inside: float = 0.0
    share: Optional[float] = None       # inside / speech, the same yardstick both sides
    transcript_in_vad: float = float("nan")
    vad_share: float = float("nan")
    blind: bool = False


# Two independent signals have to agree before the diagnostics are called
# blind: a film can be mostly music (low vad_share) and still have silero
# hearing its dialogue perfectly (high transcript_in_vad).
VAD_BLIND_BELOW = 0.25
VAD_BLIND_MIN_LOUD = 60.0


def coverage(picture: VadPicture, cues) -> Coverage:
    """Transcribed seconds against silero's, measured the same way twice.

    Deliberately by segment span on both sides even where word timestamps
    exist: the API engine has none, and a figure that flatters one engine
    is worse than a pessimistic one that compares.
    """
    spans = spans_of(cues)
    out = Coverage(speech=picture.speech, transcribed=total(spans))
    out.inside = intersect(picture.intervals, spans) if picture.intervals else 0.0
    out.share = (out.inside / picture.speech) if picture.speech else None
    out.transcript_in_vad = (out.inside / out.transcribed) if out.transcribed else float("nan")
    out.vad_share = picture.vad_share
    out.blind = bool(
        picture.loud >= VAD_BLIND_MIN_LOUD
        and out.vad_share == out.vad_share and out.vad_share < VAD_BLIND_BELOW
        and out.transcript_in_vad == out.transcript_in_vad
        and out.transcript_in_vad < VAD_BLIND_BELOW
    )
    return out


# ------------------------------------------------------------- the subtitles

_SENTENCE_END = re.compile(r"[.!?。！？…]$")


def open_ended_ratio(cues) -> float:
    """Share of lines that do not end a sentence — the segmenter's own test.

    Above 0.5 the segmenter stops merging sentences and defers to refine
    (segmenter.is_effectively_unpunctuated), so this number decides which
    of two code paths a film takes.
    """
    lines = [c for c in cues if c.text.strip()]
    if not lines:
        return float("nan")
    return sum(1 for c in lines if not _SENTENCE_END.search(c.text.strip())) / len(lines)


def order_faults(cues) -> Dict[str, int]:
    """Overlapping or out-of-order cues: a hard invariant, not a metric."""
    ordered = list(cues)
    return {
        "inverted": sum(1 for a, b in zip(ordered, ordered[1:]) if b.start < a.start),
        "overlapping": sum(1 for a, b in zip(ordered, ordered[1:]) if b.start < a.end - 0.001),
        "empty": sum(1 for c in ordered if c.end - c.start <= 0.0),
    }


def exclusive(left, right, bridge: float = 0.3, least: float = 1.5) -> List[Span]:
    """Stretches only *left* has subtitles on."""
    return [g for g in subtract(merge([(c.start, c.end) for c in left], bridge),
                                merge([(c.start, c.end) for c in right], bridge))
            if g[1] - g[0] >= least]


def said_between(cues, start: float, end: float, limit: int = 70) -> str:
    return " ".join(c.text for c in cues if c.end > start and c.start < end)[:limit]


def characters(cues) -> int:
    return sum(len(c.text) for c in cues)


# A pair only says something about *placement* when both sides are the same
# utterance cut the same way. Merge two lines into one sentence — which is
# exactly what asking the model for punctuation does — and the merged cue
# legitimately starts seconds earlier than the fragment it pairs with.
# Measured on one film: 7.9% of all pairs differ by >2s, but only 0.7% of
# the pairs that are the same words at the same length.
COMPARABLE_JACCARD = 0.85
COMPARABLE_DURATION_SLACK = 0.5


def comparable(pairs) -> List[tuple]:
    """Pairs that are the same line, segmented the same way, on both sides."""
    out = []
    for left, right, score in pairs:
        if score < COMPARABLE_JACCARD:
            continue
        span = max(left.end - left.start, 0.1)
        if abs((right.end - right.start) - (left.end - left.start)) > \
                COMPARABLE_DURATION_SLACK * span:
            continue
        out.append((left, right, score))
    return out


def far_share(pairs, apart: float = 2.0) -> Optional[float]:
    """Share of comparable pairs whose starts are more than *apart* apart.

    Above that the cue is not late, it is in the wrong place — nothing in
    this pipeline moves a start by more than 1.5s on purpose.
    """
    chosen = comparable(pairs)
    if not chosen:
        return None
    return sum(1 for left, right, _ in chosen
               if abs(right.start - left.start) > apart) / len(chosen)


def start_deltas(pairs) -> List[float]:
    return [right.start - left.start for left, right, _ in pairs]


def describe_deltas(deltas: Sequence[float]) -> str:
    if not deltas:
        return "n/a"
    ordered = sorted(deltas)
    return (f"中位 {percentile(ordered, 0.5):+.2f}s  p10 {percentile(ordered, 0.1):+.2f}s  "
            f"p90 {percentile(ordered, 0.9):+.2f}s  "
            f"|Δ|≤0.5s {sum(1 for d in ordered if abs(d) <= 0.5) / len(ordered):.0%}  "
            f"|Δ|≤1s {sum(1 for d in ordered if abs(d) <= 1.0) / len(ordered):.0%}  "
            f"|Δ|>2s {sum(1 for d in ordered if abs(d) > 2.0) / len(ordered):.0%}")


def load_audio(path):
    """Decode to 16 kHz mono — the same call the pipeline makes."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=16000)


def nan_safe(value: float) -> bool:
    return value == value and not math.isinf(value)
