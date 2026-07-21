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
from typing import List, Optional, Sequence, Union

from app.models.schemas import SubtitleLine, SubtitleSettings
from app.services.asr import Segment, Word

# split preference: sentence enders > clause punctuation > spaces
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？…])\s*")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:，；：、])\s*")
_SENTENCE_END = re.compile(r"[.!?。！？…]$")
# same punctuation as the two regexes above, as sets for last-char tests
_SENTENCE_CHARS = ".!?。！？…"
_CLAUSE_CHARS = ",;:，；：、"

MIN_LINE_DURATION = 0.5  # seconds
GAP_BREAK = 1.5  # silence between words that forces a new line (s)
NO_HARD_SPLIT_BELOW = 12  # never character-cut texts shorter than this
CHARS_PER_SECOND = 3.0  # conservative speaking-rate floor for bogus-duration cap
MERGE_FRAGMENT_CHARS = 2  # trailing fragments this short get merged back
MERGE_FRAGMENT_GAP = 0.6  # ...when the time gap to the previous line is below (s)

# A cue may occupy two display lines (the 42-chars figure of subtitle style
# guides is per DISPLAY line, not per cue), so accumulation gets twice the
# budget and max_chars_per_line only governs wrapping in subtitle.py.
# Enforcing it per cue is what shredded normal English sentences into
# "…never entered my" + "mind." — which in turn let the translator reflow
# content across lines and shift the whole file.
CUE_CHAR_BUDGET_RATIO = 2
MERGE_GAP = 0.8  # max silence between lines of the same sentence (s)
MERGE_MAX_DURATION = 7.0  # a merged cue never lasts longer than this (s)
# a clause-boundary break must leave at least this share of the buffer behind,
# otherwise breaking at the overflowing word is the lesser evil
MIN_CLAUSE_BREAK_RATIO = 0.4


def _cap_bogus_duration(text: str, start: float, end: float) -> float:
    """Return a sane end time: whisper sometimes attaches long silence."""
    duration = end - start
    plausible = max(len(text) / CHARS_PER_SECOND, MIN_LINE_DURATION) * 2
    if duration > plausible:
        return start + max(len(text) / CHARS_PER_SECOND, MIN_LINE_DURATION)
    return end


# ------------------------------------------------------------- word path


def _clause_break_index(buf: Sequence[Word]) -> Optional[int]:
    """Index to split *buf* at so the first part ends on a natural boundary.

    Prefers the latest sentence ender, then the latest clause punctuation.
    Returns None when no boundary leaves a substantial enough first part —
    breaking mid-phrase is then the lesser evil.
    """
    total = len("".join(w.text for w in buf).strip())
    if total == 0:
        return None
    for chars in (_SENTENCE_CHARS, _CLAUSE_CHARS):
        for i in range(len(buf) - 1, 0, -1):
            head = "".join(w.text for w in buf[:i]).strip()
            if len(head) < total * MIN_CLAUSE_BREAK_RATIO:
                break
            if head[-1] in chars:
                return i
    return None


def _lines_from_words(words: Sequence[Word], settings: SubtitleSettings) -> List[SubtitleLine]:
    lines: List[SubtitleLine] = []
    buf: List[Word] = []
    budget = max(settings.max_chars_per_line * CUE_CHAR_BUDGET_RATIO, 1)

    def flush(split_at: Optional[int] = None):
        """Emit ``buf[:split_at]`` as a line; whatever is left stays buffered."""
        if not buf:
            return
        take = buf[:split_at] if split_at is not None else list(buf)
        rest = buf[split_at:] if split_at is not None else []
        text = "".join(w.text for w in take).strip()
        if text:
            lines.append(
                SubtitleLine(
                    index=0,
                    start=round(take[0].start, 3),
                    end=round(max(take[-1].end, take[0].start + MIN_LINE_DURATION), 3),
                    text=text,
                )
            )
        else:
            rest = list(buf) if split_at is not None else []
        buf.clear()
        buf.extend(rest)

    for word in words:
        if buf:
            candidate_len = len("".join(w.text for w in buf) + word.text.rstrip())
            too_long = candidate_len > budget
            too_slow = word.end - buf[0].start > settings.max_duration
            gap = word.start - buf[-1].end > GAP_BREAK
            if gap or too_slow:
                flush()  # time-driven breaks always cut the whole buffer
            elif too_long:
                flush(_clause_break_index(buf))
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


# ------------------------------------------------------- sentence merging


def _needs_space(ch: str) -> bool:
    """CJK scripts write without spaces between words; latin scripts don't."""
    return not ("　" <= ch <= "鿿" or "＀" <= ch <= "￯")


def _join_text(a: str, b: str) -> str:
    if not a or not b:
        return a or b
    if _needs_space(a[-1]) and _needs_space(b[0]):
        return f"{a} {b}"
    return a + b


def _can_merge(
    prev: SubtitleLine, line: SubtitleLine, budget: int, max_duration: float
) -> bool:
    gap = line.start - prev.end
    if len(line.text) <= MERGE_FRAGMENT_CHARS:
        if gap > MERGE_FRAGMENT_GAP:  # stray 1-2 char fragment (CJK)
            return False
    else:
        if _SENTENCE_END.search(prev.text):
            return False  # previous cue is a complete sentence, leave it alone
        if gap > MERGE_GAP:
            return False  # a real pause: same sentence or not, don't glue cues
    if len(_join_text(prev.text, line.text)) > budget:
        return False
    return line.end - prev.start <= max_duration


def _merge_sentence_units(
    lines: List[SubtitleLine], settings: SubtitleSettings
) -> List[SubtitleLine]:
    """Glue cues that whisper cut mid-sentence back into whole sentences.

    A cue ending without sentence punctuation is the tell: whatever follows
    it after a short gap is the rest of that sentence. Leaving those halves
    apart is what pushes the translator into re-flowing content across lines.
    """
    budget = max(settings.max_chars_per_line * CUE_CHAR_BUDGET_RATIO, 1)
    max_duration = max(settings.max_duration, MERGE_MAX_DURATION)
    merged: List[SubtitleLine] = []
    for line in lines:
        if merged and _can_merge(merged[-1], line, budget, max_duration):
            merged[-1].text = _join_text(merged[-1].text, line.text)
            merged[-1].end = line.end
        else:
            merged.append(line)
    return merged


def open_ended_ratio(lines: Sequence[SubtitleLine]) -> float:
    """Share of cues not ending on sentence punctuation — the fragmentation
    metric reported in the job log (a healthy film sits below ~0.1)."""
    if not lines:
        return 0.0
    open_ended = sum(1 for l in lines if not _SENTENCE_END.search(l.text.strip()))
    return open_ended / len(lines)


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

    merged = _merge_sentence_units(lines, settings)

    for i, line in enumerate(merged, start=1):
        line.index = i
    return merged
