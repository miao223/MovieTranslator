"""Where a disc's files come from: a folder on disk, or an ISO image.

Everything above this module names files the way the disc does
("BDMV/PLAYLIST/00800.mpls", "VTS_01_1.VOB") and never asks what they are
stored in. Two things make that worth an abstraction:

* **Case.** Discs are authored with upper-case names, copies of them end up
  in every case there is (00800.MPLS, bdmv/, Video_ts/). Lookups here are
  case-insensitive, and the rest of the code never has to think about it.
* **ISO images.** A file inside an image is a list of byte ranges in the
  image, not a file the OS can open. `extents()` answers that question for
  both kinds, and `ConcatReader` turns any list of ranges into one seekable
  read-only file object — which is also exactly what a DVD cell is (a range
  of sectors spread over VTS_01_1.VOB … VTS_01_9.VOB).

`ConcatReader` counts the bytes it hands out, which is how remux progress
is measured: exactly, in bytes of the disc actually read.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.services.disc.binary import DiscError


@dataclass(frozen=True)
class Extent:
    file: str       # a real file on this machine
    offset: int     # where the range starts in it
    length: int


def sub_extents(extents: List[Extent], start: int, length: int) -> List[Extent]:
    """The ranges covering bytes [start, start+length) of *extents* laid end to end."""
    out: List[Extent] = []
    pos = 0
    end = start + length
    for ext in extents:
        lo, hi = pos, pos + ext.length
        pos = hi
        if hi <= start:
            continue
        if lo >= end:
            break
        a, b = max(lo, start), min(hi, end)
        out.append(Extent(ext.file, ext.offset + (a - lo), b - a))
    return out


class ConcatReader(io.RawIOBase):
    """One read-only, seekable file made of byte ranges of other files."""

    def __init__(self, extents: List[Extent], name: str = "", prefix: bytes = b""):
        super().__init__()
        self.extents = [e for e in extents if e.length > 0]
        self.name = name
        # bytes served before the first extent, as if they were in the file
        # (remux.py primes the DVD demuxer with them)
        self.prefix = prefix
        self.total = len(prefix) + sum(e.length for e in self.extents)
        self._starts = []
        pos = len(prefix)
        for e in self.extents:
            self._starts.append(pos)
            pos += e.length
        self._pos = 0
        self._handles: Dict[str, object] = {}
        # bytes handed to the reader so far — the remux progress measure
        self.consumed = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self.total + offset
        else:
            raise ValueError(f"bad whence {whence}")
        self._pos = max(0, pos)
        return self._pos

    def _handle(self, path: str):
        fh = self._handles.get(path)
        if fh is None:
            fh = open(path, "rb")  # noqa: SIM115 — closed in close()
            self._handles[path] = fh
        return fh

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        want = len(view)
        done = 0
        if self._pos < len(self.prefix):
            n = min(want, len(self.prefix) - self._pos)
            view[:n] = self.prefix[self._pos:self._pos + n]
            done += n
            self._pos += n
        while done < want and self._pos < self.total:
            # the extent holding self._pos
            lo, hi = 0, len(self._starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._starts[mid] <= self._pos:
                    lo = mid
                else:
                    hi = mid - 1
            ext = self.extents[lo]
            inside = self._pos - self._starts[lo]
            n = min(want - done, ext.length - inside)
            fh = self._handle(ext.file)
            fh.seek(ext.offset + inside)
            got = fh.readinto(view[done:done + n])
            if not got:
                raise DiscError(f"读取 {self.name or ext.file} 时文件提前结束（文件被截断？）")
            done += got
            self._pos += got
        self.consumed += done
        return done

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.total - self._pos
        buf = bytearray(max(0, min(size, self.total - self._pos)))
        n = self.readinto(buf)
        return bytes(buf[:n])

    def close(self) -> None:
        for fh in self._handles.values():
            try:
                fh.close()
            except OSError:
                pass
        self._handles.clear()
        super().close()


class DiscFS:
    """Read access to one disc's files. Names use "/" and ignore case."""

    source = ""

    def lookup(self, rel: str) -> Optional[str]:
        raise NotImplementedError

    def is_dir(self, rel: str) -> bool:
        raise NotImplementedError

    def listdir(self, rel: str) -> List[str]:
        raise NotImplementedError

    def size(self, rel: str) -> int:
        raise NotImplementedError

    def extents(self, rel: str) -> List[Extent]:
        raise NotImplementedError

    def exists(self, rel: str) -> bool:
        return self.lookup(rel) is not None

    def anchor(self) -> Optional[Path]:
        """A real file whose stat says "this disc, unchanged" — for resuming."""
        return None

    def read(self, rel: str, limit: Optional[int] = None) -> bytes:
        found = self.lookup(rel)
        if found is None:
            raise DiscError(f"光盘里缺少文件：{rel}")
        with self.open(found) as fh:
            return fh.read(-1 if limit is None else limit)

    def open(self, rel: str) -> ConcatReader:
        found = self.lookup(rel)
        if found is None:
            raise DiscError(f"光盘里缺少文件：{rel}")
        return ConcatReader(self.extents(found), name=found)


class DirFS(DiscFS):
    """A disc copied to a folder."""

    source = "dir"

    def __init__(self, base: Path):
        self.base = Path(base)
        self._listing: Dict[str, Dict[str, str]] = {}

    def _names(self, rel_dir: str) -> Dict[str, str]:
        cached = self._listing.get(rel_dir)
        if cached is None:
            folder = self.base / rel_dir if rel_dir else self.base
            try:
                cached = {n.lower(): n for n in os.listdir(folder)}
            except OSError:
                cached = {}
            self._listing[rel_dir] = cached
        return cached

    def lookup(self, rel: str) -> Optional[str]:
        parts = [p for p in re.split(r"[\\/]+", rel) if p]
        done: List[str] = []
        for part in parts:
            real = self._names("/".join(done)).get(part.lower())
            if real is None:
                return None
            done.append(real)
        return "/".join(done)

    def _path(self, rel: str) -> Path:
        found = self.lookup(rel)
        if found is None:
            raise DiscError(f"光盘里缺少文件：{rel}")
        return self.base / found

    def is_dir(self, rel: str) -> bool:
        found = self.lookup(rel)
        return found is not None and (self.base / found).is_dir()

    def listdir(self, rel: str) -> List[str]:
        found = self.lookup(rel) if rel else ""
        if found is None:
            return []
        return sorted(self._names(found).values())

    def size(self, rel: str) -> int:
        return self._path(rel).stat().st_size

    def extents(self, rel: str) -> List[Extent]:
        path = self._path(rel)
        return [Extent(str(path), 0, path.stat().st_size)]

    def anchor(self) -> Optional[Path]:
        for rel in ("BDMV/index.bdmv", "BDMV/PLAYLIST", "VIDEO_TS.IFO"):
            found = self.lookup(rel)
            if found:
                return self.base / found
        return None


# ------------------------------------------------------------ finding a disc

@dataclass
class Located:
    fs: DiscFS
    kind: str        # "bd" | "dvd"
    root: Path       # the folder the outputs go next to / the .iso itself
    name: str        # default base name for the outputs
    # DVD only: where VIDEO_TS.IFO lives inside fs ("" or "VIDEO_TS")
    ifo_dir: str = ""


# Folder names that say nothing about the film: a box set's "DISC1" or a
# bare "BD". The film's name is one level up.
_GENERIC = re.compile(r"^(?:disc|disk|bd|dvd|cd|vol(?:ume)?)[\s._-]*\d*$", re.IGNORECASE)


def _name_for(folder: Path) -> str:
    name = folder.name or "disc"
    if _GENERIC.match(name) and folder.parent.name:
        return f"{folder.parent.name} {name}"
    return name


def _child(folder: Path, name: str) -> Optional[Path]:
    try:
        for entry in os.listdir(folder):
            if entry.lower() == name.lower():
                return folder / entry
    except OSError:
        return None
    return None


def _bd_root(folder: Path) -> Optional[Path]:
    """*folder* if it holds a BDMV directory with playlists in it."""
    bdmv = _child(folder, "BDMV")
    if bdmv is not None and bdmv.is_dir() and _child(bdmv, "PLAYLIST") is not None:
        return folder
    return None


def _dvd_dir(folder: Path) -> Optional[Path]:
    """The folder holding VIDEO_TS.IFO: *folder* itself or its VIDEO_TS."""
    if _child(folder, "VIDEO_TS.IFO") is not None:
        return folder
    vts = _child(folder, "VIDEO_TS")
    if vts is not None and vts.is_dir() and _child(vts, "VIDEO_TS.IFO") is not None:
        return vts
    return None


# A folder picked for the batch mode by mistake — a whole drive — would be
# walked and every disc on it analysed before the page answered.
MAX_BATCH_DISCS = 200


def _natural(path: Path):
    """VOL2 before VOL10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))]


def find_discs(folder: str, recursive: bool = True) -> List[Path]:
    """Every disc in *folder* (the 原盘 page's batch mode): disc folders and
    .iso images, in and — when *recursive* — under it. A disc is not looked
    inside: its BDMV/STREAM holds the film's pieces, not more discs."""
    raw = (folder or "").strip().strip('"').strip("'").strip()
    if not raw:
        raise DiscError("路径为空")
    top = Path(raw).expanduser()
    if top.is_file() and top.suffix.lower() == ".iso":
        return [top]
    if not top.is_dir():
        raise DiscError(f"不是有效的文件夹：{top}")
    if top.name.lower() in ("bdmv", "video_ts"):
        top = top.parent
    if _bd_root(top) is not None or _dvd_dir(top) is not None:
        return [top]
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(top):
        here = Path(dirpath)
        deeper = []
        for name in sorted(dirnames):
            if name.startswith("."):
                continue
            child = here / name
            if _bd_root(child) is not None or _dvd_dir(child) is not None:
                found.append(child)
            else:
                deeper.append(name)
        for name in sorted(filenames):
            if name.lower().endswith(".iso") and not name.startswith("."):
                found.append(here / name)
        dirnames[:] = deeper if recursive else []
        if len(found) > MAX_BATCH_DISCS:
            raise DiscError(f"这个文件夹里的原盘超过 {MAX_BATCH_DISCS} 张，"
                            "请选范围小一点的文件夹")
    return sorted(found, key=_natural)


def locate(path: str) -> Located:
    """Work out which disc *path* means.

    Accepted: the folder holding BDMV/ or VIDEO_TS/, the BDMV or VIDEO_TS
    folder itself, any file inside them (index.bdmv, a .mpls, an .IFO), or
    an .iso image.
    """
    raw = (path or "").strip().strip('"').strip("'").strip()
    if not raw:
        raise DiscError("路径为空")
    p = Path(raw).expanduser()
    if p.is_file() and p.suffix.lower() == ".iso":
        from app.services.disc.udf import open_iso

        return open_iso(p)
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        raise DiscError(f"路径不存在：{p}")

    # the BDMV folder itself, or one of its subfolders (PLAYLIST/STREAM…)
    for folder in (p, p.parent, p.parent.parent):
        if folder.name.lower() == "bdmv" and _child(folder, "PLAYLIST") is not None:
            p = folder.parent
            break
    root = _bd_root(p)
    if root is not None:
        return Located(DirFS(root), "bd", root, _name_for(root))
    ifo_dir = _dvd_dir(p)
    if ifo_dir is not None:
        root = ifo_dir.parent if ifo_dir.name.lower() == "video_ts" else ifo_dir
        return Located(DirFS(ifo_dir), "dvd", root, _name_for(root))

    # A box set: the folder holds several discs. Name them instead of
    # saying "not a disc" about a folder that plainly has discs in it.
    inside = []
    try:
        for entry in sorted(os.listdir(p)):
            child = p / entry
            if child.is_dir() and (_bd_root(child) or _dvd_dir(child)):
                inside.append(entry)
            elif child.suffix.lower() == ".iso":
                inside.append(entry)
    except OSError:
        pass
    if inside:
        raise DiscError(
            f"这个文件夹里有 {len(inside)} 张原盘，请选其中一张："
            + "、".join(inside[:8]) + ("…" if len(inside) > 8 else ""))
    raise DiscError(f"这里不是原盘：{p} 里没有找到 BDMV 或 VIDEO_TS")
