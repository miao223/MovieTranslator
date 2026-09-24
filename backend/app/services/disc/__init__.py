"""Blu-ray / DVD / ISO discs: read their structure, pick titles, remux to MKV.

No decryption of any kind lives here or anywhere in this program: a disc
that is still encrypted is recognised and refused (bdmv.looks_encrypted,
dvd.looks_scrambled).
"""

from __future__ import annotations

from app.services.disc.binary import DiscError
from app.services.disc.fs import locate
from app.services.disc.model import Disc


def open_disc(path: str) -> Disc:
    """Read the disc *path* points at (folder, BDMV/VIDEO_TS, file inside, .iso)."""
    located = locate(path)
    if located.kind == "bd":
        from app.services.disc import bdmv

        disc = bdmv.load(located)
    else:
        from app.services.disc import dvd

        disc = dvd.load(located)
    disc.path = (path or "").strip().strip('"').strip("'").strip()
    if not disc.titles:
        raise DiscError("这张盘里没有找到任何可以播放的标题")
    if disc.analysis_only:
        disc.warnings.insert(0, "这是只有元数据的副本（没有视频文件）：可以分析，不能封装")
    return disc


__all__ = ["Disc", "DiscError", "open_disc"]
