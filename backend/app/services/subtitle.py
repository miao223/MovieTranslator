"""SRT/ASS serialization and in-place frame-cue insertion.

Timestamps live only here — the LLM never sees them; lines are matched
back to their timestamps via the 1-based line index.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from app.models.schemas import SubtitleLine, SubtitleSettings

# (start_seconds, end_seconds, translated_text)
FrameCue = Tuple[float, float, str]

_SRT_TIMING = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*$"
)
_ASS_DIALOGUE_START = re.compile(r"^Dialogue:\s*\d+\s*,\s*(\d+):(\d{2}):(\d{2})\.(\d{2})\s*,")


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_color(hex_color: str) -> str:
    """'#RRGGBB' → ASS '&H00BBGGRR' (BGR order, leading alpha)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def format_ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    cs = round(seconds * 100)
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    # braces would be parsed as override tags
    return text.replace("{", "（").replace("}", "）")


def wrap_display_text(text: str, max_chars: int) -> List[str]:
    """Break *text* into at most two display lines of ~max_chars each.

    A cue holds a whole sentence now (see segmenter), which can exceed one
    display line. Wrapping here rather than leaving it to the player keeps
    the result identical across VLC / mpv / PotPlayer. Text that cannot be
    split sensibly is returned as-is — never hard-cut mid-word.
    """
    text = text.strip()
    if len(text) <= max_chars or max_chars <= 0:
        return [text]

    # prefer breaking near the middle so the two lines look balanced
    target = len(text) / 2
    best: int | None = None       # closest boundary that also fits max_chars
    fallback: int | None = None   # closest boundary, fitting or not
    for i, ch in enumerate(text):
        if ch not in " ，。、！？；：,.!?;:" or not 0 < i < len(text) - 1:
            continue
        # cut after the character, except for spaces which are dropped
        cut = i if ch == " " else i + 1
        if not 0 < cut < len(text):
            continue
        if fallback is None or abs(cut - target) < abs(fallback - target):
            fallback = cut
        if max(len(text[:cut].strip()), len(text[cut:].strip())) > max_chars:
            continue  # still too wide, but keep it as a fallback candidate
        if best is None or abs(cut - target) < abs(best - target):
            best = cut
    # two overlong lines still beat one very long line
    cut = best if best is not None else fallback
    if cut is None:
        return [text]
    return [text[:cut].strip(), text[cut:].strip()]


def build_ass(
    lines: List[SubtitleLine],
    settings: SubtitleSettings,
    mode: str = "bilingual",
) -> str:
    """Render *lines* as an ASS file with the configured size/colors."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{settings.font_size},{_ass_color(settings.translation_color)},&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    orig_tag = (
        f"{{\\fs{settings.original_font_size}"
        f"\\c{_ass_color(settings.original_color)}&}}"
    )
    events = []
    wrap = settings.max_chars_per_line
    for line in lines:
        # wrap first, escape after: _ass_escape only touches braces, so the
        # \N separators inserted here survive untouched
        translation = "\\N".join(
            _ass_escape(part)
            for part in wrap_display_text(line.translation.strip() or line.text, wrap)
        )
        original = "\\N".join(
            _ass_escape(part) for part in wrap_display_text(line.text, wrap)
        )
        if line.is_frame:
            text = "{\\an7}" + translation
        elif mode == "translation_only":
            text = translation
        elif settings.bilingual_layout == "translation_top":
            text = f"{translation}\\N{orig_tag}{original}"
        else:
            text = f"{orig_tag}{original}{{\\r}}\\N{translation}"
        events.append(
            f"Dialogue: 0,{format_ass_timestamp(line.start)},"
            f"{format_ass_timestamp(line.end)},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


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
    wrap = settings.max_chars_per_line
    for n, line in enumerate(lines, start=1):
        translation = "\n".join(
            wrap_display_text(line.translation.strip() or line.text, wrap)
        )
        original = "\n".join(wrap_display_text(line.text, wrap))
        if line.is_frame:
            # on-screen text cue: top-left ({\an7} is honoured by VLC /
            # PotPlayer / mpv even in SRT), translation only
            text = "{\\an7}" + translation
        elif mode == "translation_only":
            text = translation
        else:
            if settings.bilingual_layout == "translation_top":
                text = f"{translation}\n{original}"
            else:
                text = f"{original}\n{translation}"
        blocks.append(
            f"{n}\n{format_timestamp(line.start)} --> {format_timestamp(line.end)}\n{text}\n"
        )
    return "\n".join(blocks)


# ------------------------------------------------- in-place cue insertion


def insert_frame_cues_srt(srt_text: str, cues: Sequence[FrameCue]) -> str:
    """Insert frame cues (as {\\an7} blocks) into existing SRT text.

    Existing block text is preserved verbatim; only numbering is redone.
    """
    blocks = []  # (start_seconds, [timing_line, *text_lines])
    for raw in re.split(r"\n\s*\n", srt_text.strip()):
        lines = raw.splitlines()
        if not lines:
            continue
        # first line is usually the index — timing may be on line 0 or 1
        timing_idx = next(
            (i for i, l in enumerate(lines[:2]) if _SRT_TIMING.match(l.strip())),
            None,
        )
        if timing_idx is None:
            continue  # malformed block: drop rather than corrupt output
        m = _SRT_TIMING.match(lines[timing_idx].strip())
        start = (
            int(m.group(1)) * 3600 + int(m.group(2)) * 60
            + int(m.group(3)) + int(m.group(4)) / 1000
        )
        blocks.append((start, lines[timing_idx:]))

    for start, end, text in cues:
        timing = f"{format_timestamp(start)} --> {format_timestamp(end)}"
        blocks.append((start, [timing, "{\\an7}" + text]))

    blocks.sort(key=lambda b: b[0])
    out = []
    for n, (_start, lines) in enumerate(blocks, start=1):
        out.append(str(n) + "\n" + "\n".join(lines) + "\n")
    return "\n".join(out)


def _ass_start_seconds(line: str) -> float | None:
    m = _ASS_DIALOGUE_START.match(line)
    if not m:
        return None
    return (
        int(m.group(1)) * 3600 + int(m.group(2)) * 60
        + int(m.group(3)) + int(m.group(4)) / 100
    )


def insert_frame_cues_ass(ass_text: str, cues: Sequence[FrameCue]) -> str:
    """Insert frame cues as Dialogue events, sorted by start time.

    Header/style sections are untouched. If no Dialogue lines are found
    (unexpected format), the new events are appended at the end — ASS
    renderers do not require chronological event order.
    """
    new_events = [
        f"Dialogue: 0,{format_ass_timestamp(start)},{format_ass_timestamp(end)},"
        f"Default,,0,0,0,,{{\\an7}}{_ass_escape(text)}"
        for start, end, text in cues
    ]
    lines = ass_text.splitlines()
    dialogue_idx = [
        (i, s) for i, l in enumerate(lines)
        if (s := _ass_start_seconds(l)) is not None
    ]
    if not dialogue_idx:
        return ass_text.rstrip("\n") + "\n" + "\n".join(new_events) + "\n"

    for event, (start, _e, _t) in sorted(
        zip(new_events, cues), key=lambda p: p[1][0], reverse=True
    ):
        # insert before the first existing dialogue that starts later
        insert_at = dialogue_idx[-1][0] + 1
        for i, s in dialogue_idx:
            if s > start:
                insert_at = i
                break
        lines.insert(insert_at, event)
        dialogue_idx = [
            (i, s) for i, l in enumerate(lines)
            if (s := _ass_start_seconds(l)) is not None
        ]
    return "\n".join(lines) + "\n"
