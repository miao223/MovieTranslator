"""Media file constants and directory scanning."""

from __future__ import annotations

import glob
import re

from pathlib import Path
from typing import List, Tuple

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".ts", ".m2ts", ".mpg", ".mpeg", ".rmvb", ".m4v",
}

# Releases are often shared as the audio track alone — a film is tens of GB,
# its audio a few dozen MB — and the pipeline's first step throws the picture
# away anyway. A few of these (.mka, .ogg) can legally carry video; the suffix
# only decides what gets listed and offered, `audio.has_picture` decides how a
# file is actually run.
AUDIO_EXTS = {
    ".mp3", ".m4a", ".m4b", ".aac", ".flac", ".wav", ".ogg", ".oga",
    ".opus", ".wma", ".ape", ".alac", ".mka", ".dts", ".ac3", ".eac3",
    ".amr", ".aiff", ".aif", ".wv", ".tta", ".mpc", ".caf", ".w64", ".mp2",
}

MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS


def is_audio_ext(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def kind_of(path: str | Path) -> str:
    """'video' / 'audio' / '' — what the file picker labels an entry with."""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return ""


def scan_media(
    directory: str | Path,
    recursive: bool = True,
    skip_existing_srt: bool = True,
    target_language: str = "",
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """Find video and audio files under *directory*.

    Returns (to_translate, skipped, shadowed, with_source):

    *skipped* are files already translated into *target_language*, decided
    by reading the subtitle next to them rather than by its name (see
    subtitle_state). *with_source* is the subset of to_translate that has a
    subtitle in some **other** language — those can be translated from that
    text instead of from speech, which is faster and more accurate. They
    are reported, never acted on: which source to use is the user's call.

    Without a *target_language* the old rule applies and any subtitle
    counts as done, because there is nothing to compare against. *shadowed* are audio files dropped because a video
    of the same stem sits in the same folder — both would write the same .srt,
    so the one with the picture wins and the audio (usually extracted from it)
    steps aside. Hidden directories/files (dot-prefixed) are ignored.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"不是有效目录: {directory}")

    files: List[Path] = []
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if not p.is_file() or p.suffix.lower() not in MEDIA_EXTS:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        files.append(p)
    files.sort()

    # done before the .srt check: were it after, a video whose subtitle already
    # exists would be skipped and its audio twin would live on to overwrite it
    stems = {(p.parent, p.stem.lower()) for p in files if not is_audio_ext(p)}
    shadowed = [p for p in files if is_audio_ext(p) and (p.parent, p.stem.lower()) in stems]
    if shadowed:
        dropped = set(shadowed)
        files = [p for p in files if p not in dropped]

    to_translate, skipped, with_source = [], [], []
    for f in files:
        state, _lang = subtitle_state(f, target_language)
        if state == "done" and skip_existing_srt:
            skipped.append(f)
            continue
        to_translate.append(f)
        if state == "source":
            with_source.append(f)
    return to_translate, skipped, shadowed, with_source


# A language suffix as original_only writes it: two or three letters, or the
# "orig"/"und" it falls back to when the language could not be determined.
_LANG_SUFFIX = re.compile(r"^[a-z]{2,3}$|^orig$|^und$")


def subtitles_beside(video: Path) -> List[Path]:
    """Every subtitle file sitting next to this one, ours or anyone else's."""
    stem = video.stem
    found = [p for p in (video.with_suffix(".srt"), video.with_suffix(".ass"))
             if p.is_file()]
    for sibling in sorted(video.parent.glob(f"{glob.escape(stem)}.*")):
        if sibling.suffix.lower() not in (".srt", ".ass") or not sibling.is_file():
            continue
        middle = sibling.name[len(stem) + 1: -len(sibling.suffix)]
        if _LANG_SUFFIX.match(middle.lower()):
            found.append(sibling)
    return found


def subtitle_state(video: Path, target_language: str = "") -> tuple[str, str]:
    """What the subtitle next to *video* means: (state, its language).

    ``done``   already in the target language — this film is translated.
    ``source`` in some other language — not a result but **material**, and
               better material than speech recognition, since a person
               typed it against the picture.
    ``none``   nothing there.

    The distinction cannot be made from the filename: this program's own
    original_only output and a subtitle downloaded from anywhere else are
    both ``film.ja.srt``. So the file is read — a few dozen KB, and only
    its opening lines.

    Unsure always means ``done``. An unreadable subtitle, or a target
    language this build has no code for, leaves no way to compare, and
    re-translating a whole season is the more expensive mistake of the two.
    """
    from app.services import mux, subsource

    want = mux.language_of(target_language)[0] if target_language else ""
    if want == mux.FALLBACK[0]:
        want = ""                       # no code for this target language
    for path in subtitles_beside(video):
        tag = path.name[len(video.stem) + 1: -len(path.suffix)]
        language = ""
        if _LANG_SUFFIX.match(tag.lower()):
            language = subsource.iso2(tag)
        language = language or _language_of_file(path)
        if not language or not want or language == want:
            return "done", language
        return "source", language
    return "none", ""


def _language_of_file(path: Path) -> str:
    """Guess a subtitle file's language from its text.

    Parsed with a few lines here rather than through subsource's real
    reader: that one opens a container per file, and a scan may cover a
    hundred of them for an answer that only needs some words.
    """
    from app.models.schemas import SubtitleLine
    from app.services import subsource

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read(120_000)          # the opening lines settle it
    except OSError:
        return ""
    texts: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "-->" in line or line.isdigit():
            continue                        # srt index and timing lines
        if line.startswith("Dialogue:"):    # ass event: text is the last field
            line = line.split(",", 9)[-1]
        elif line.split(":", 1)[0].isalpha() and ":" in line[:14] and "," in line:
            continue                        # other ass header/event fields
        texts.append(subsource.event_text(line, ass=path.suffix.lower() == ".ass"))
    lines = [SubtitleLine(index=i, start=0.0, end=0.0, text=t)
             for i, t in enumerate(texts[:400], 1) if t.strip()]
    return subsource.detect_language(lines) if lines else ""


def scan_media(
    directory: str | Path,
    recursive: bool = True,
    skip_existing_srt: bool = True,
    target_language: str = "",
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """Find video and audio files under *directory*.

    Returns (to_translate, skipped, shadowed, with_source):

    *skipped* are files already translated into *target_language*, decided
    by reading the subtitle next to them rather than by its name (see
    subtitle_state). *with_source* is the subset of to_translate that has a
    subtitle in some **other** language — those can be translated from that
    text instead of from speech, which is faster and more accurate. They
    are reported, never acted on: which source to use is the user's call.

    Without a *target_language* the old rule applies and any subtitle
    counts as done, because there is nothing to compare against. *shadowed* are audio files dropped because a video
    of the same stem sits in the same folder — both would write the same .srt,
    so the one with the picture wins and the audio (usually extracted from it)
    steps aside. Hidden directories/files (dot-prefixed) are ignored.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"不是有效目录: {directory}")

    files: List[Path] = []
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if not p.is_file() or p.suffix.lower() not in MEDIA_EXTS:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        files.append(p)
    files.sort()

    # done before the .srt check: were it after, a video whose subtitle already
    # exists would be skipped and its audio twin would live on to overwrite it
    stems = {(p.parent, p.stem.lower()) for p in files if not is_audio_ext(p)}
    shadowed = [p for p in files if is_audio_ext(p) and (p.parent, p.stem.lower()) in stems]
    if shadowed:
        dropped = set(shadowed)
        files = [p for p in files if p not in dropped]

    to_translate, skipped, with_source = [], [], []
    for f in files:
        state, _lang = subtitle_state(f, target_language)
        if state == "done" and skip_existing_srt:
            skipped.append(f)
            continue
        to_translate.append(f)
        if state == "source":
            with_source.append(f)
    return to_translate, skipped, shadowed, with_source


# A language suffix as original_only writes it: two or three letters, or the
# "orig"/"und" it falls back to when the language could not be determined.
_LANG_SUFFIX = re.compile(r"^[a-z]{2,3}$|^orig$|^und$")


def subtitles_beside(video: Path) -> List[Path]:
    """Every subtitle file sitting next to this one, ours or anyone else's."""
    stem = video.stem
    found = [p for p in (video.with_suffix(".srt"), video.with_suffix(".ass"))
             if p.is_file()]
    for sibling in sorted(video.parent.glob(f"{glob.escape(stem)}.*")):
        if sibling.suffix.lower() not in (".srt", ".ass") or not sibling.is_file():
            continue
        middle = sibling.name[len(stem) + 1: -len(sibling.suffix)]
        if _LANG_SUFFIX.match(middle.lower()):
            found.append(sibling)
    return found


def subtitle_state(video: Path, target_language: str = "") -> tuple[str, str]:
    """What the subtitle next to *video* means: (state, its language).

    ``done``   already in the target language — this film is translated.
    ``source`` in some other language — not a result but **material**, and
               better material than speech recognition, since a person
               typed it against the picture.
    ``none``   nothing there.

    The distinction cannot be made from the filename: this program's own
    original_only output and a subtitle downloaded from anywhere else are
    both ``film.ja.srt``. So the file is read — a few dozen KB, and only
    its opening lines.

    Unsure always means ``done``. An unreadable subtitle, or a target
    language this build has no code for, leaves no way to compare, and
    re-translating a whole season is the more expensive mistake of the two.
    """
    from app.services import mux, subsource

    want = mux.language_of(target_language)[0] if target_language else ""
    if want == mux.FALLBACK[0]:
        want = ""                       # no code for this target language
    for path in subtitles_beside(video):
        tag = path.name[len(video.stem) + 1: -len(path.suffix)]
        language = ""
        if _LANG_SUFFIX.match(tag.lower()):
            language = subsource.iso2(tag)
        language = language or _language_of_file(path)
        if not language or not want or language == want:
            return "done", language
        return "source", language
    return "none", ""


def _language_of_file(path: Path) -> str:
    """Guess a subtitle file's language from its text.

    Parsed with a few lines here rather than through subsource's real
    reader: that one opens a container per file, and a scan may cover a
    hundred of them for an answer that only needs some words.
    """
    from app.models.schemas import SubtitleLine
    from app.services import subsource

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read(120_000)          # the opening lines settle it
    except OSError:
        return ""
    texts: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "-->" in line or line.isdigit():
            continue                        # srt index and timing lines
        if line.startswith("Dialogue:"):    # ass event: text is the last field
            line = line.split(",", 9)[-1]
        elif line.split(":", 1)[0].isalpha() and ":" in line[:14] and "," in line:
            continue                        # other ass header/event fields
        texts.append(subsource.event_text(line, ass=path.suffix.lower() == ".ass"))
    lines = [SubtitleLine(index=i, start=0.0, end=0.0, text=t)
             for i, t in enumerate(texts[:400], 1) if t.strip()]
    return subsource.detect_language(lines) if lines else ""


def has_subtitle(video: Path) -> bool:
    """Is there already a subtitle for this file next to it?

    Both shapes count. Translated output is ``film.srt``, but original_only
    writes ``film.ja.srt`` — and only checking the first meant a season run
    in that mode was rescanned as untranslated every time, re-running the
    lot. The language is not knowable at scan time when source_language is
    auto, so the suffix is matched by shape rather than by value.

    Deliberately narrow: ``film.backup.srt`` and ``film.v2.srt`` are not
    subtitles this program wrote, and skipping a file because of one would
    be the worse mistake.
    """
    if video.with_suffix(".srt").exists():
        return True
    stem = video.stem
    for sibling in video.parent.glob(f"{glob.escape(stem)}.*.srt"):
        middle = sibling.name[len(stem) + 1: -len(".srt")]
        if _LANG_SUFFIX.match(middle.lower()):
            return True
    return False
