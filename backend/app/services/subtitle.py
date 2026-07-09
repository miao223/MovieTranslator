"""SRT serialization: mono-language or bilingual cues.

Timestamps live only here — the LLM never sees them; lines are matched
back to their timestamps via the 1-based line index.
"""

from __future__ import annotations

from typing import List

from app.models.schemas import SubtitleLine, SubtitleSettings


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(
    lines: List[SubtitleLine],
    settings: SubtitleSettings,
    mode: str = "bilingual",
) -> str:
    """Render *lines* as SRT text.

    mode: "bilingual" (original + translation) or "translation_only".
    Lines with an empty translation fall back to the original text.
    """
    blocks: List[str] = []
    for n, line in enumerate(lines, start=1):
        translation = line.translation.strip() or line.text
        if mode == "translation_only":
            text = translation
        else:
            if settings.bilingual_layout == "translation_top":
                text = f"{translation}\n{line.text}"
            else:
                text = f"{line.text}\n{translation}"
        blocks.append(
            f"{n}\n{format_timestamp(line.start)} --> {format_timestamp(line.end)}\n{text}\n"
        )
    return "\n".join(blocks)
