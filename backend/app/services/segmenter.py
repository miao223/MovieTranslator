"""Turn raw whisper segments into subtitle lines.

Rules: a line must not exceed *max_chars_per_line* characters nor
*max_duration* seconds; overly long segments are split at punctuation /
space boundaries with time allocated proportionally to text length.
"""

from __future__ import annotations

import re
from typing import List

from app.models.schemas import SubtitleLine, SubtitleSettings
from app.services.asr import RawSegment

# split preference: sentence enders > clause punctuation > spaces
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？…])\s*")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:，；：、])\s*")

MIN_LINE_DURATION = 0.5  # seconds


def _split_text(text: str, max_chars: int) -> List[str]:
    """Split *text* into chunks of at most *max_chars*, at natural boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    for pattern in (_SENTENCE_BREAK, _CLAUSE_BREAK):
        parts = [p for p in pattern.split(text) if p.strip()]
        if len(parts) > 1:
            # greedily repack parts into chunks <= max_chars
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
            # recurse in case an individual part is still too long
            out: List[str] = []
            for c in chunks:
                out.extend(_split_text(c, max_chars) if len(c) > max_chars else [c])
            return out

    # no punctuation: fall back to splitting at spaces, then hard cuts
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


def segment_lines(
    segments: List[RawSegment], settings: SubtitleSettings
) -> List[SubtitleLine]:
    lines: List[SubtitleLine] = []
    for start, end, text in segments:
        text = text.strip()
        if not text:
            continue
        duration = max(end - start, MIN_LINE_DURATION)
        chunks = _split_text(text, settings.max_chars_per_line)
        # also honour max_duration: ensure enough chunks that each piece fits
        if chunks and duration / len(chunks) > settings.max_duration:
            import math

            need = math.ceil(duration / settings.max_duration)
            while len(chunks) < need:
                # split the longest chunk further
                longest = max(range(len(chunks)), key=lambda i: len(chunks[i]))
                sub = _split_text(
                    chunks[longest], max(len(chunks[longest]) // 2, 1)
                )
                if len(sub) <= 1:
                    break
                chunks[longest : longest + 1] = sub

        total_chars = sum(len(c) for c in chunks) or 1
        t = start
        for chunk in chunks:
            share = duration * len(chunk) / total_chars
            lines.append(
                SubtitleLine(
                    index=0,  # renumbered below
                    start=round(t, 3),
                    end=round(min(t + share, end), 3),
                    text=chunk,
                )
            )
            t += share

    for i, line in enumerate(lines, start=1):
        line.index = i
    return lines
