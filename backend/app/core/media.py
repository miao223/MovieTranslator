"""Video file constants and directory scanning."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".ts", ".m2ts", ".mpg", ".mpeg", ".rmvb", ".m4v",
}


def scan_videos(
    directory: str | Path,
    recursive: bool = True,
    skip_existing_srt: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """Find video files under *directory*.

    Returns (to_translate, skipped): *skipped* are videos that already have
    a same-stem .srt next to them (only when skip_existing_srt). Hidden
    directories/files (dot-prefixed) are ignored.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"不是有效目录: {directory}")

    videos: List[Path] = []
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        videos.append(p)
    videos.sort()

    if not skip_existing_srt:
        return videos, []
    to_translate, skipped = [], []
    for v in videos:
        (skipped if v.with_suffix(".srt").exists() else to_translate).append(v)
    return to_translate, skipped
