"""Read the files inside a disc image (.iso) — read-only UDF, as discs use it.

pycdlib, the pure-Python ISO library, cannot open a Blu-ray image
(clalancette/pycdlib#19): Blu-ray keeps its whole file tree in a UDF 2.50
*metadata partition*, a partition that is itself stored as a file. This is
the small subset of ECMA-167 / OSTA UDF 1.02–2.60 a pressed or ripped disc
actually uses:

  anchor (sector 256) -> volume descriptor sequence -> partition +
  logical volume (partition maps) -> file set -> directories -> files

* partition maps: type 1 (a physical partition — DVD, UDF 1.02) and type 2
  "*UDF Metadata Partition" (Blu-ray, UDF 2.50/2.60). Sparable and virtual
  partitions are for rewritable media and are refused by name.
* file entries: FE and EFE; allocation descriptors short, long and
  embedded, with continuation (AED) for heavily fragmented files.
* names: OSTA CS0, 8- and 16-bit.

A file comes out as a list of byte ranges of the image (fs.Extent), which
ConcatReader turns into one seekable file — so a clip inside an image is
read exactly like a clip on disk, without mounting or extracting anything.

Everything in UDF is little-endian, unlike the Blu-ray and DVD tables.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.services.disc.binary import DiscError
from app.services.disc.fs import DiscFS, Extent, Located, _name_for

SECTOR = 2048
ANCHOR = 256

TAG_PD, TAG_LVD, TAG_TD = 5, 6, 8
TAG_FSD, TAG_FID, TAG_AED, TAG_FE, TAG_EFE = 256, 257, 258, 261, 266
FILE_TYPE_DIR = 4

AD_SHORT, AD_LONG, AD_EXT, AD_EMBEDDED = 0, 1, 2, 3


def _u16(b: bytes, at: int) -> int:
    return struct.unpack_from("<H", b, at)[0]


def _u32(b: bytes, at: int) -> int:
    return struct.unpack_from("<I", b, at)[0]


def _u64(b: bytes, at: int) -> int:
    return struct.unpack_from("<Q", b, at)[0]


def _tag(b: bytes) -> int:
    """The descriptor's tag id, or -1 when the checksum says it is not one."""
    if len(b) < 16:
        return -1
    if sum(b[0:4]) + sum(b[5:16]) & 0xFF != b[4]:
        return -1
    return _u16(b, 0)


def _cs0(raw: bytes) -> str:
    """OSTA compressed unicode: a leading 8 (one byte per char) or 16 (UCS-2 BE)."""
    if not raw:
        return ""
    kind, body = raw[0], raw[1:]
    if kind in (16, 255):
        return body[: len(body) // 2 * 2].decode("utf-16-be", "replace")
    return body.decode("latin-1")


@dataclass
class Node:
    name: str
    is_dir: bool
    size: int
    # (partition ref, logical block, length) runs, or the data itself
    runs: List[Tuple[int, int, int]] = field(default_factory=list)
    embedded: Optional[bytes] = None


class UdfImage:
    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.size = self.path.stat().st_size
            self._fh = open(self.path, "rb")  # noqa: SIM115 — closed by close()
        except OSError as exc:
            raise DiscError(f"打不开镜像 {self.path.name}：{exc}") from exc
        self._dirs: Dict[Tuple[int, int], List[Node]] = {}
        try:
            self._volume()
        except DiscError:
            self.close()
            raise

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass

    # ----------------------------------------------------------- sectors

    def _sector(self, n: int, count: int = 1) -> bytes:
        if n < 0 or (n + count) * SECTOR > self.size:
            raise DiscError(f"镜像 {self.path.name} 不完整：需要第 {n} 扇区，文件只有 "
                            f"{self.size // SECTOR} 扇区")
        self._fh.seek(n * SECTOR)
        return self._fh.read(count * SECTOR)

    # ------------------------------------------------------------ volume

    def _volume(self) -> None:
        anchor = self._sector(ANCHOR) if self.size > (ANCHOR + 1) * SECTOR else b""
        if _tag(anchor) != 2:
            raise DiscError(f"{self.path.name} 不是 UDF 镜像（找不到第 256 扇区的锚点）；"
                            "请先挂载或解开它")
        length, where = _u32(anchor, 16), _u32(anchor, 20)
        partitions: Dict[int, int] = {}
        lvd = None
        for i in range(max(length // SECTOR, 1)):
            d = self._sector(where + i)
            tag = _tag(d)
            if tag == TAG_PD:
                partitions[_u16(d, 22)] = _u32(d, 188)
            elif tag == TAG_LVD:
                lvd = d
            elif tag == TAG_TD:
                break
        if lvd is None or not partitions:
            raise DiscError(f"{self.path.name} 的 UDF 卷描述不完整（缺少分区或逻辑卷）")
        if _u32(lvd, 212) != SECTOR:
            raise DiscError(f"{self.path.name} 的逻辑块大小不是 2048，不是光盘镜像")
        self._maps: List[dict] = []
        at, n_maps = 440, _u32(lvd, 268)
        for _ in range(n_maps):
            kind, size = lvd[at], lvd[at + 1]
            if kind == 1:
                number = _u16(lvd, at + 4)
                self._maps.append({"kind": "physical", "start": partitions.get(number, 0)})
            elif kind == 2:
                ident = lvd[at + 5:at + 28].split(b"\x00")[0].decode("ascii", "replace")
                number = _u16(lvd, at + 38)
                if "Metadata" not in ident:
                    raise DiscError(f"{self.path.name} 用了 {ident}（可擦写光盘的格式），暂不支持")
                self._maps.append({"kind": "metadata", "start": partitions.get(number, 0),
                                   "file": _u32(lvd, at + 40), "runs": None})
            else:
                raise DiscError(f"{self.path.name} 有未知的分区映射类型 {kind}")
            at += size
        # the metadata partition is a file in the physical one: read its
        # extents once, and map its blocks through them from then on
        for i, m in enumerate(self._maps):
            if m["kind"] == "metadata":
                phys = next(j for j, x in enumerate(self._maps) if x["kind"] == "physical")
                node = self._entry(phys, m["file"], "")
                m["runs"] = [(lbn, length) for _p, lbn, length in node.runs]
                m["phys"] = phys
        fsd_len, fsd_lbn, fsd_ref = _u32(lvd, 248), _u32(lvd, 252), _u16(lvd, 256)
        del fsd_len
        fsd = self._block(fsd_ref, fsd_lbn)
        if _tag(fsd) != TAG_FSD:
            raise DiscError(f"{self.path.name} 找不到文件集描述符")
        root_lbn, root_ref = _u32(fsd, 404), _u16(fsd, 408)
        self.root = self._entry(root_ref, root_lbn, "")

    # ------------------------------------------------------------ blocks

    def _absolute(self, ref: int, lbn: int, length: int) -> List[Tuple[int, int]]:
        """Byte ranges of the image holding *length* bytes from block *lbn*
        of partition map *ref*."""
        if not 0 <= ref < len(self._maps):
            raise DiscError(f"{self.path.name} 引用了不存在的分区 {ref}")
        m = self._maps[ref]
        if m["kind"] == "physical":
            return [((m["start"] + lbn) * SECTOR, length)]
        out: List[Tuple[int, int]] = []
        want = lbn * SECTOR
        left = length
        start = self._maps[m["phys"]]["start"]
        pos = 0
        for run_lbn, run_len in m["runs"]:
            if left <= 0:
                break
            if want < pos + run_len:
                inside = max(want - pos, 0)
                take = min(run_len - inside, left)
                out.append(((start + run_lbn) * SECTOR + inside, take))
                want += take
                left -= take
            pos += run_len
        if left > 0:
            raise DiscError(f"{self.path.name} 的元数据分区比引用的位置短")
        return out

    def _read(self, ref: int, lbn: int, length: int) -> bytes:
        chunks = []
        for offset, size in self._absolute(ref, lbn, length):
            self._fh.seek(offset)
            chunks.append(self._fh.read(size))
        data = b"".join(chunks)
        if len(data) < length:
            raise DiscError(f"镜像 {self.path.name} 不完整")
        return data

    def _block(self, ref: int, lbn: int) -> bytes:
        return self._read(ref, lbn, SECTOR)

    # ------------------------------------------------------------- files

    def _ads(self, ref: int, raw: bytes, kind: int, runs: List[Tuple[int, int, int]],
             depth: int = 0) -> None:
        step = {AD_SHORT: 8, AD_LONG: 16, AD_EXT: 20}.get(kind)
        if step is None:
            raise DiscError(f"{self.path.name} 用了未知的分配描述符类型 {kind}")
        for at in range(0, len(raw) - step + 1, step):
            word = _u32(raw, at)
            length, typ = word & 0x3FFFFFFF, word >> 30
            if length == 0:
                break
            if kind == AD_SHORT:
                lbn, part = _u32(raw, at + 4), ref
            elif kind == AD_LONG:
                lbn, part = _u32(raw, at + 4), _u16(raw, at + 8)
            else:
                lbn, part = _u32(raw, at + 12), _u16(raw, at + 16)
            if typ == 3:                       # continues in an AED
                if depth > 64:
                    raise DiscError(f"{self.path.name} 的分配描述符链太长")
                aed = self._block(part, lbn)
                if _tag(aed) != TAG_AED:
                    raise DiscError(f"{self.path.name} 的分配描述符续表损坏")
                n = _u32(aed, 20)
                self._ads(part, aed[24:24 + n], kind, runs, depth + 1)
                return
            if typ in (0, 1):                  # recorded (1 = allocated, unwritten)
                runs.append((part, lbn, length))
            else:                              # not allocated: a hole
                runs.append((-1, 0, length))

    def _entry(self, ref: int, lbn: int, name: str) -> Node:
        d = self._block(ref, lbn)
        tag = _tag(d)
        if tag == TAG_FE:
            size, l_ea, l_ad, base = _u64(d, 56), _u32(d, 168), _u32(d, 172), 176
        elif tag == TAG_EFE:
            size, l_ea, l_ad, base = _u64(d, 56), _u32(d, 208), _u32(d, 212), 216
        else:
            raise DiscError(f"{self.path.name} 里 {name or '一个文件'} 的文件入口损坏")
        file_type, flags = d[27], _u16(d, 34)
        node = Node(name, file_type == FILE_TYPE_DIR, size)
        area = d[base + l_ea: base + l_ea + l_ad]
        if flags & 7 == AD_EMBEDDED:
            node.embedded = area[:size]
        else:
            self._ads(ref, area, flags & 7, node.runs)
        return node

    def data(self, node: Node) -> bytes:
        if node.embedded is not None:
            return node.embedded
        return b"".join(
            b"\x00" * length if part < 0 else self._read(part, lbn, length)
            for part, lbn, length in node.runs)[: node.size]

    def children(self, node: Node) -> List[Node]:
        key = (id(node), 0)
        cached = self._dirs.get(key)
        if cached is not None:
            return cached
        raw = self.data(node)
        out: List[Node] = []
        at = 0
        while at + 38 <= len(raw):
            if _u16(raw, at) != TAG_FID:
                break
            chars, l_fi = raw[at + 18], raw[at + 19]
            icb_lbn, icb_ref = _u32(raw, at + 24), _u16(raw, at + 28)
            l_iu = _u16(raw, at + 36)
            ident = raw[at + 38 + l_iu: at + 38 + l_iu + l_fi]
            at += (38 + l_iu + l_fi + 3) // 4 * 4
            if chars & 0x0C:                   # deleted, or the parent entry
                continue
            out.append(self._entry(icb_ref, icb_lbn, _cs0(ident)))
        self._dirs[key] = out
        return out

    def extents(self, node: Node) -> List[Extent]:
        """Where *node*'s bytes are in the image. Holes are not expected in
        disc files and are refused rather than read as garbage."""
        if node.embedded is not None:
            raise DiscError(f"{node.name} 太小，存在文件入口里，不是视频文件")
        out: List[Extent] = []
        left = node.size
        for part, lbn, length in node.runs:
            if left <= 0:
                break
            take = min(length, left)
            if part < 0:
                raise DiscError(f"镜像里的 {node.name} 有未写入的空洞，文件不完整")
            for offset, size in self._absolute(part, lbn, take):
                if out and out[-1].offset + out[-1].length == offset:
                    out[-1] = Extent(out[-1].file, out[-1].offset, out[-1].length + size)
                else:
                    out.append(Extent(str(self.path), offset, size))
            left -= take
        return out


class IsoFS(DiscFS):
    """A disc image, read in place."""

    source = "iso"

    def __init__(self, image: UdfImage):
        self.image = image
        self._nodes: Dict[str, Node] = {"": image.root}

    def _walk(self, rel: str) -> Optional[Tuple[str, Node]]:
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        path, node = "", self.image.root
        for part in parts:
            if not node.is_dir:
                return None
            match = next((c for c in self.image.children(node)
                          if c.name.lower() == part.lower()), None)
            if match is None:
                return None
            path = f"{path}/{match.name}" if path else match.name
            node = match
            self._nodes[path] = node
        return path, node

    def lookup(self, rel: str) -> Optional[str]:
        found = self._walk(rel)
        return None if found is None else found[0]

    def _node(self, rel: str) -> Node:
        found = self._walk(rel)
        if found is None:
            raise DiscError(f"镜像里缺少文件：{rel}")
        return found[1]

    def is_dir(self, rel: str) -> bool:
        found = self._walk(rel)
        return found is not None and found[1].is_dir

    def listdir(self, rel: str) -> List[str]:
        found = self._walk(rel)
        if found is None or not found[1].is_dir:
            return []
        return sorted(c.name for c in self.image.children(found[1]))

    def size(self, rel: str) -> int:
        return self._node(rel).size

    def extents(self, rel: str) -> List[Extent]:
        return self.image.extents(self._node(rel))

    def read(self, rel: str, limit: Optional[int] = None) -> bytes:
        node = self._node(rel)
        if node.embedded is not None:
            return node.embedded[:limit] if limit is not None else node.embedded
        return super().read(rel, limit)

    def anchor(self) -> Optional[Path]:
        return self.image.path


def open_iso(path: Path) -> Located:
    image = UdfImage(path)
    fs = IsoFS(image)
    name = _name_for(Path(path).with_suffix("")) if Path(path).suffix.lower() == ".iso" \
        else Path(path).name
    if fs.lookup("BDMV/PLAYLIST"):
        return Located(fs, "bd", Path(path), name)
    if fs.lookup("VIDEO_TS/VIDEO_TS.IFO"):
        return Located(fs, "dvd", Path(path), name, ifo_dir=fs.lookup("VIDEO_TS") or "VIDEO_TS")
    image.close()
    raise DiscError(f"{Path(path).name} 里没有 BDMV 或 VIDEO_TS，不是蓝光或 DVD 的镜像")
