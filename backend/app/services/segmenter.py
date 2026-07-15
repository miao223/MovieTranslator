"""Turn whisper segments into subtitle lines.

Preferred path: word-level timestamps. Lines are assembled word by word and
broken at sentence punctuation, silence gaps, or when the length/duration
limits are hit — every cue keeps REAL start/end times, so stretched or
fabricated timings cannot occur.

Fallback path (word timestamps disabled): the classic text splitter, with
two guards against whisper's stretched-segment pathology: a duration far
beyond what the text could occupy is capped to a speaking-rate estimate,
and short texts are never hard-cut into character fragments.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Union

from app.models.schemas import SubtitleLine, SubtitleSettings
from app.services.asr import Segment, Word

# split preference: sentence enders > clause punctuation > spaces
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？…])\s*")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:，；：、])\s*")
_SENTENCE_END = re.compile(r"[.!?。！？…]$")

MIN_LINE_DURATION = 0.5  # seconds
GAP_BREAK = 1.5  # silence between words that forces a new line (s)
NO_HARD_SPLIT_BELOW = 12  # never character-cut texts shorter than this
CHARS_PER_SECOND = 3.0  # conservative speaking-rate floor for bogus-duration cap
MERGE_FRAGMENT_CHARS = 2  # trailing fragments this short get merged back
MERGE_FRAGMENT_GAP = 0.6  # ...when the time gap to the previous line is below (s)


def _cap_bogus_duration(text: str, start: float, end: float) -> float:
    """Return a sane end time: whisper sometimes attaches long silence."""
    duration = end - start
    plausible = max(len(text) / CHARS_PER_SECOND, MIN_LINE_DURATION) * 2
    if duration > plausible:
        return start + max(len(text) / CHARS_PER_SECOND, MIN_LINE_DURATION)
    return end


# ------------------------------------------------------------- word path


def _lines_from_words(words: Sequence[Word], settings: SubtitleSettings) -> List[SubtitleLine]:
    lines: List[SubtitleLine] = []
    buf: List[Word] = []

    def flush():
        if not buf:
            return
        text = "".join(w.text for w in buf).strip()
        if text:
            lines.append(
                SubtitleLine(
                    index=0,
                    start=round(buf[0].start, 3),
                    end=round(max(buf[-1].end, buf[0].start + MIN_LINE_DURATION), 3),
                    text=text,
                )
            )
        buf.clear()

    for word in words:
        if buf:
            candidate_len = len("".join(w.text for w in buf) + word.text.rstrip())
            too_long = candidate_len > settings.max_chars_per_line
            too_slow = word.end - buf[0].start > settings.max_duration
            gap = word.start - buf[-1].end > GAP_BREAK
            if too_long or too_slow or gap:
                flush()
        buf.append(word)
        if _SENTENCE_END.search(word.text.strip()):
            flush()
    flush()
    return lines


# ------------------------------------------------------------- text path


def _split_text(text: str, max_chars: int) -> List[str]:
    """Split *text* into chunks of at most *max_chars*, at natural boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    for pattern in (_SENTENCE_BREAK, _CLAUSE_BREAK):
        parts = [p for p in pattern.split(text) if p.strip()]
        if len(parts) > 1:
            chunks: List[str] = []
            current = ""
            for part in parts:
                candidate = (current + " " + part).strip() if current else part.strip()
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part.strip()
            if current:
                chunks.append(current)
            out: List[str] = []
            for c in chunks:
                out.extend(_split_text(c, max_chars) if len(c) > max_chars else [c])
            return out

    words = text.split(" ")
    if len(words) > 1:
        chunks, current = [], ""
        for w in words:
            candidate = (current + " " + w).strip() if current else w
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = w
        if current:
            chunks.append(current)
        out = []
        for c in chunks:
            out.extend(_split_text(c, max_chars) if len(c) > max_chars else [c])
        return out

    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _lines_from_text(
    start: float, end: float, text: str, settings: SubtitleSettings
) -> List[SubtitleLine]:
    end = _cap_bogus_duration(text, start, end)
    duration = max(end - start, MIN_LINE_DURATION)
    chunks = _split_text(text, settings.max_chars_per_line)
    # honour max_duration only when the text is long enough to survive the
    # extra splitting — short texts with inflated durations were the source
    # of the 2-chars-per-5-seconds fragment bug
    if chunks and duration / len(chunks) > settings.max_duration:
        import math

        need = math.ceil(duration / settings.max_duration)
        while len(chunks) < need:
            longest = max(range(len(chunks)), key=lambda i: len(chunks[i]))
            if len(chunks[longest]) < NO_HARD_SPLIT_BELOW:
                break
            sub = _split_text(chunks[longest], max(len(chunks[longest]) // 2, 1))
            if len(sub) <= 1:
                break
            chunks[longest : longest + 1] = sub

    total_chars = sum(len(c) for c in chunks) or 1
    lines: List[SubtitleLine] = []
    t = start
    for chunk in chunks:
        share = duration * len(chunk) / total_chars
        lines.append(
            SubtitleLine(
                index=0,
                start=round(t, 3),
                end=round(min(t + share, end), 3),
                text=chunk,
            )
        )
        t += share
    return lines


# ---------------------------------------------------------------- public


def segment_lines(
    segments: Sequence[Union[Segment, tuple]], settings: SubtitleSettings
) -> List[SubtitleLine]:
    lines: List[SubtitleLine] = []
    for seg in segments:
        if isinstance(seg, tuple):  # legacy (start, end, text) form
            seg = Segment(seg[0], seg[1], seg[2])
        if not seg.text.strip():
            continue
        if seg.words:
            lines.extend(_lines_from_words(seg.words, settings))
        else:
            lines.extend(_lines_from_text(seg.start, seg.end, seg.text, settings))

    # merge stray 1-2 char fragments into the previous line
    merged: List[SubtitleLine] = []
    for line in lines:
        if (
            merged
            and len(line.text) <= MERGE_FRAGMENT_CHARS
            and line.start - merged[-1].end <= MERGE_FRAGMENT_GAP
            and len(merged[-1].text) + len(line.text) <= settings.max_chars_per_line
        ):
            merged[-1].text += line.text
            merged[-1].end = line.end
        else:
            merged.append(line)

    for i, line in enumerate(merged, start=1):
        line.index = i
    return merged
