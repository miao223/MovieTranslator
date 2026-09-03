"""Media file constants and directory scanning."""

from __future__ import annotations

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
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Find video and audio files under *directory*.

    Returns (to_translate, skipped, shadowed):

    *skipped* are files that already have a same-stem .srt next to them (only
    when skip_existing_srt). *shadowed* are audio files dropped because a video
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

    if not skip_existing_srt:
        return files, [], shadowed
    to_translate, skipped = [], []
    for f in files:
        (skipped if f.with_suffix(".srt").exists() else to_translate).append(f)
    return to_translate, skipped, shadowed
