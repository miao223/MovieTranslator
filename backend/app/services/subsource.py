"""Read an existing subtitle as the source text, instead of transcribing.

Many releases already carry the dialogue: an embedded soft-subtitle track,
or a same-named .srt/.ass beside the file. That text was typed by a person
against the picture — the line breaks, the punctuation and the names are
right, and the timings were nudged to the cut. It beats anything this
program can recognise from the audio, and it arrives in seconds with no
model, no GPU and no hour of decoding.

The module is deliberately shaped like ``services/audio.py``: list_tracks /
pick_track / describe_track have the same contract, return the same dict
shape, and share its language tables — the two are the same choice made
twice (which stream of this file do we take the words from).

Reading the cues, verified against PyAV 18:

* **``container.decode(stream)`` throws the timing away.** It flattens each
  ``SubtitleSet`` into its rects (av/container/input.py), and a rect carries
  no pts at all — ``Subtitle`` holds only proxy/ptr/type. The type stub
  (av/container/input.pyi) claims ``Iterator[SubtitleSet]``; it is wrong.
  So the packets are demuxed by hand for their timing and decoded one at a
  time with ``decode2`` for their text.
* **The decoder's own clocks disagree with each other**: ``SubtitleSet.pts``
  is always rescaled to ``av.time_base`` (microseconds) while
  ``start_display_time``/``end_display_time`` stay in milliseconds. The
  packet's own pts/duration are in the stream's time_base and need no
  rescaling guesswork, so they are what this uses.
* **``AssSubtitle.dialogue`` cannot be used.** Its state machine enters
  skip-mode on ``{`` only when the next character is *not* a backslash, so
  it strips ``{comments}`` and keeps ``{\\an8}`` — the opposite of what a
  translator needs. The raw ``.ass`` event is cleaned here instead.

Every text codec (subrip, ass/ssa, mov_text, webvtt, microdvd…) is
normalised to an ASS event by ffmpeg, so one cleaner covers them all.
Bitmap tracks (PGS, VobSub, DVB) hold pictures and no text; they are listed,
marked, and refused.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional

import av

from app.models.schemas import SubtitleLine
from app.services import lyrics, mux
from app.services.asr import has_content
from app.services.audio import canon_language, language_name

LogFn = Callable[[str], None]

AV_DISPOSITION_DEFAULT = 1 << 0
AV_DISPOSITION_FORCED = 1 << 6

# Pictures, not words. Turning these into text is OCR, which is a different
# program; they are listed so the user can see why they are not offered.
BITMAP_CODECS = {
    "hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvdsub",
    "dvb_subtitle", "dvbsub", "xsub",
}

# Subtitle files that may sit next to the video, best format first
SIDECAR_SUFFIXES = (".ass", ".ssa", ".srt", ".vtt")

# ffmpeg hands every text codec over as an ASS event, whose fields are
# ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text — eight
# commas, then the line. None of those fields may itself contain a comma.
_ASS_FIELDS = 8
_OVERRIDE = re.compile(r"\{[^}]*\}")          # {\an8}, {\pos(..)}, {comments}
_HTML = re.compile(r"</?[a-zA-Z][^>]*>")      # <i>, </b>, <font color=…>
# {\p1} switches the cue body from text to vector drawing commands, and
# "m 0 0 l 100 0" reads as content to any character-level test. The digit
# is what distinguishes it from {\pos(..)}.
_DRAWING = re.compile(r"\{[^}]*\\p[1-9]")
# two cues this far apart are two utterances, however alike they read
_DUPLICATE_GAP = 0.1


# --------------------------------------------------------------- tracks


def _track_info(stream) -> dict:
    codec = (stream.codec_context.name or "").lower()
    disposition = int(stream.disposition or 0)
    return {
        "index": stream.index,  # container-wide stream index, as ffmpeg reports it
        "path": "",  # set only for a sidecar file
        "codec": codec.upper(),
        "language": canon_language(stream.language or ""),
        "language_name": language_name(stream.language or ""),
        "title": (stream.metadata.get("title") or "").strip(),
        "default": bool(disposition & AV_DISPOSITION_DEFAULT),
        # a forced track carries only signs and foreign lines — a few dozen
        # cues where a full one has thousands, so never auto-select it
        "forced": bool(disposition & AV_DISPOSITION_FORCED),
        "text": codec not in BITMAP_CODECS,
    }


def list_tracks(video_path: str | Path) -> list[dict]:
    """The video's own subtitle streams. Empty is a normal answer."""
    with av.open(str(video_path)) as container:
        return [_track_info(s) for s in container.streams.subtitles]


def sidecar_tracks(video_path: str | Path) -> list[dict]:
    """Subtitle files sitting next to the video under the same stem.

    Both ``film.srt`` and the language-suffixed ``film.en.srt`` that most
    downloads use. The language is taken from that suffix when it names one.
    """
    video_path = Path(video_path)
    found: list[dict] = []
    try:
        siblings = sorted(video_path.parent.iterdir())
    except OSError:
        return found
    for path in siblings:
        if path.suffix.lower() not in SIDECAR_SUFFIXES or not path.is_file():
            continue
        if path.stem != video_path.stem and not path.stem.startswith(
            video_path.stem + "."
        ):
            continue
        # film.en.srt -> "en"; film.srt -> ""
        tag = path.stem[len(video_path.stem) + 1:] if path.stem != video_path.stem else ""
        language = canon_language(tag) if tag and len(tag) <= 3 else ""
        found.append({
            "index": -1,  # not a container stream
            "path": str(path),
            "codec": path.suffix.lstrip(".").upper(),
            "language": language,
            "language_name": language_name(language) if language else "未标注语言",
            "title": path.name,
            "default": False,
            "forced": False,
            "text": True,
        })
    return found


def all_tracks(video_path: str | Path) -> list[dict]:
    """Everything that could serve as the source text, embedded first."""
    try:
        embedded = list_tracks(video_path)
    except Exception:  # noqa: BLE001 — an unreadable container still has siblings
        embedded = []
    return embedded + sidecar_tracks(video_path)


def describe_track(track: dict) -> str:
    """One line for the log, e.g. '字幕轨 #3 英语「Full」SUBRIP (默认)'."""
    if track["path"]:
        head = f"外挂字幕 {Path(track['path']).name}"
        parts = [head, track["language_name"], track["codec"]]
    else:
        named = track["language_name"] + (
            f"「{track['title']}」" if track["title"] else ""
        )
        parts = [f"字幕轨 #{track['index']}", named, track["codec"]]
    if track["forced"]:
        parts.append("(强制/仅告示牌)")
    if track["default"]:
        parts.append("(默认)")
    if not track["text"]:
        parts.append("(图形字幕，无法读取文字)")
    return " ".join(p for p in parts if p)


def describe_stream(stream) -> str:
    """Describe a PyAV subtitle stream the way the picker would name it."""
    return describe_track(_track_info(stream))


def bitmap_error(track: dict) -> str:
    return (
        f"{describe_track(track)} 只有图片没有文字，"
        "读不出可翻译的内容（需要 OCR）；请改选其它字幕轨或改用语音识别"
    )


def pick_track(
    tracks: list[dict],
    index: Optional[int] = None,
    file: str = "",
    language: str = "",
) -> dict:
    """Choose one: explicit file > explicit index > language > default > first.

    Raises ValueError when nothing readable is on offer, or when the caller
    named a bitmap track — the two cases the pipeline has to tell apart.

    An index that no longer exists falls back rather than raising, matching
    audio.pick_track: the video may have been re-picked after the选择.
    """
    if file:
        for t in tracks:
            if t["path"] == file:
                return t
    if index is not None:
        for t in tracks:
            if t["index"] == index and not t["path"]:
                if not t["text"]:
                    raise ValueError(bitmap_error(t))
                return t
    readable = [t for t in tracks if t["text"]]
    if not readable:
        if tracks:
            raise ValueError(
                "这个视频只有图形字幕（PGS/VobSub 之类），读不出文字；请改用语音识别"
            )
        raise ValueError("这个视频没有内建字幕轨，也没有同名的外挂字幕文件")
    # a forced track is a legal answer only when it is the only one
    full = [t for t in readable if not t["forced"]] or readable
    wanted = canon_language(language)
    if wanted:
        for t in full:
            if t["language"] == wanted:
                return t
    for t in full:
        if t["default"]:
            return t
    return full[0]


# ----------------------------------------------------------------- cues


def _plain_text(rect) -> str:
    """The readable line inside one decoded subtitle rect."""
    raw = bytes(rect.ass or b"") or bytes(rect.text or b"")
    body = raw.decode("utf-8", "replace")
    if rect.ass:
        parts = body.split(",", _ASS_FIELDS)
        if len(parts) > _ASS_FIELDS:
            body = parts[_ASS_FIELDS]
    if _DRAWING.search(body):
        return ""  # a shape, not a line — see _DRAWING
    body = body.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    body = _OVERRIDE.sub("", body)
    body = _HTML.sub("", body)
    # one cue is one line for the LLM; the renderer re-wraps for display
    return " ".join(body.split())


def _dedupe(cues: list[tuple[float, float, str]]) -> tuple[list, int]:
    """Fold karaoke-style repeats — the same words re-emitted while still
    on screen — into one cue. A repeat after a real gap is left alone: a
    character saying the same thing twice is a subtitle, not an artifact."""
    merged: list[list] = []
    folded = 0
    for start, end, text in cues:
        if merged and merged[-1][2] == text and start <= merged[-1][1] + _DUPLICATE_GAP:
            merged[-1][1] = max(merged[-1][1], end)
            folded += 1
            continue
        merged.append([start, end, text])
    return merged, folded


def read_cues(
    video_path: str | Path,
    track: dict,
    log: Optional[LogFn] = None,
    stats: Optional[dict] = None,
) -> List[SubtitleLine]:
    """Decode *track* into subtitle lines, ready for translation.

    Raises ValueError for a bitmap track or an empty one.
    """
    if not track["text"]:
        raise ValueError(bitmap_error(track))
    source = Path(track["path"] or video_path)
    blank = 0
    raw: list[tuple[float, float, str]] = []

    with av.open(str(source)) as container:
        streams = container.streams.subtitles
        if not streams:
            raise ValueError(f"{source.name} 里没有字幕流")
        stream = next(
            (s for s in streams if s.index == track["index"]), streams[0]
        )
        # A sidecar file's times are already the times the player shows. Only
        # an embedded track shares the video's clock, and then it has to move
        # with it — the same shift services/mux.py applies when writing back.
        offset = 0.0 if track["path"] else mux.start_offset(container, source)

        for packet in container.demux(stream):
            # the empty packet demux emits at end-of-stream. Testing
            # `dts is None` instead would throw away real cues: matroska
            # stores no DTS (see services/mux.py for the same trap).
            if not packet.size or packet.pts is None:
                continue
            start = float(packet.pts * packet.time_base) - offset
            subset = stream.decode2(packet)
            if subset is None:
                continue
            if packet.duration:
                end = start + float(packet.duration * packet.time_base)
            else:
                # display times are milliseconds relative to pts, whatever
                # the stream's time_base says
                end = start + max(subset.end_display_time, 0) / 1000.0
            for rect in subset:
                if rect.type == b"bitmap":
                    raise ValueError(bitmap_error(track))
                text = _plain_text(rect)
                # Drawing commands ({\p1}m 0 0 l 100 0) and bare ♪♪ leave
                # nothing to read. Same local, deterministic rule the ASR
                # side applies to its own output — see asr.has_content.
                if not has_content(text):
                    blank += 1
                    continue
                raw.append((max(start, 0.0), max(end, start + 0.1), text))

    raw.sort(key=lambda c: (c[0], c[1]))
    merged, folded = _dedupe(raw)
    if not merged:
        raise ValueError(f"{describe_track(track)} 里没有可读的字幕内容")

    lines = [
        SubtitleLine(index=n, start=start, end=end, text=text)
        for n, (start, end, text) in enumerate(merged, start=1)
    ]
    # The subtitle already says which lines are sung. Keep that: mark_lyrics
    # only ever sets is_lyric True, so a pre-set flag survives it, and
    # apply_marks puts the ♪ back around the final text.
    sung = 0
    for line in lines:
        if lyrics.MARK in line.text:
            line.is_lyric = True
            line.text = lyrics.strip_marks(line.text)
            sung += 1

    if stats is not None:
        stats.update({"blank": blank, "folded": folded, "lyric": sung,
                      "offset": offset})
    if log:
        detail = []
        if blank:
            detail.append(f"丢弃无文字内容 {blank} 条")
        if folded:
            detail.append(f"合并重叠重复 {folded} 条")
        if sung:
            detail.append(f"已标记为歌词 {sung} 条")
        if offset >= 1.0:
            detail.append(f"片源时间轴从 {offset:.1f}s 开始，已对齐到 0")
        log(f"读取 {describe_track(track)}：{len(lines)} 条字幕"
            + ("（" + "，".join(detail) + "）" if detail else ""))
    return lines


# ------------------------------------------------------------- language


# canonical ISO 639-2/B -> the two-letter code whisper and the prompts use
_ISO2 = {
    "jpn": "ja", "eng": "en", "chi": "zh", "kor": "ko", "fre": "fr",
    "ger": "de", "spa": "es", "rus": "ru", "ita": "it", "por": "pt",
    "tha": "th", "vie": "vi", "ara": "ar", "hin": "hi",
}

_SCRIPTS = (
    ("ja", ("぀", "ヿ")),   # kana settles Japanese before the kanji test
    ("ko", ("가", "힯")),
    ("ko", ("ᄀ", "ᇿ")),
    ("th", ("฀", "๿")),
    ("ru", ("Ѐ", "ӿ")),
    ("ar", ("؀", "ۿ")),
    ("he", ("֐", "׿")),
    ("el", ("Ͱ", "Ͽ")),
    ("hi", ("ऀ", "ॿ")),
    ("zh", ("一", "鿿")),   # ideographs without kana
)

# Deliberately small. This only fills the "（原文语言：xx）" line in a prompt
# that also contains the text itself, so a near miss costs almost nothing —
# and a container's own tag is used whenever there is one.
_STOPWORDS = {
    "en": ("the", "and", "you", "that", "what", "this", "with", "have", "not"),
    "fr": ("vous", "est", "pas", "que", "les", "une", "pour", "dans", "je"),
    "de": ("nicht", "ich", "und", "ist", "sie", "der", "das", "wir", "ein"),
    "es": ("que", "está", "qué", "muy", "pero", "como", "por", "una", "sí"),
    "it": ("che", "sono", "perché", "cosa", "questo", "non", "una", "anche"),
    "pt": ("você", "não", "está", "mas", "muito", "por", "uma", "isso"),
}

_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)


def iso2(language: str) -> str:
    """'jpn'/'ja' -> 'ja'. Unknown tags come back as given."""
    canon = canon_language(language)
    return _ISO2.get(canon, canon[:2] if canon else "")


def detect_language(lines: List[SubtitleLine]) -> str:
    """A two-letter guess at what language *lines* are written in.

    Script first, because it is decisive: kana means Japanese, hangul means
    Korean, ideographs without kana mean Chinese. Only the Latin alphabet
    needs word evidence, and there a handful of stopwords separates the
    common cases well enough for a prompt hint.
    """
    text = " ".join(line.text for line in lines[:400])
    if not text.strip():
        return ""
    counts: dict[str, int] = {}
    letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        for code, (lo, hi) in _SCRIPTS:
            if lo <= ch <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts:
        code, hits = max(counts.items(), key=lambda kv: kv[1])
        # a share rather than a count: a two-line file has to be decidable,
        # and a stray kanji in an English line must not decide anything
        if hits >= 2 and hits >= letters * 0.3:
            return code

    words = {w.lower() for w in _WORDS.findall(text)}
    scored = sorted(
        ((sum(w in words for w in stop), code) for code, stop in _STOPWORDS.items()),
        reverse=True,
    )
    if scored and scored[0][0]:
        return scored[0][1]
    # Latin script with nothing to go on: English is both the likeliest and
    # the least costly wrong answer for a hint.
    return "en" if any(ch.isascii() and ch.isalpha() for ch in text) else ""
