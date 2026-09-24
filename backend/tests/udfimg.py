"""UDF disc images written from scratch, for testing app/services/disc/udf.py.

Real images cannot ship, and the tools that make UDF images are neither in
PyAV nor on CI — so, as with tests/pgs.py and tests/discgen.py, the format is
written here field by field. Two layouts, the two the reader has to handle:

* plain (UDF 1.02, what a DVD image is): one physical partition, short
  allocation descriptors, everything in it;
* metadata (UDF 2.50, what a Blu-ray image is): the file tree lives in a
  *metadata partition* — itself a file of the physical partition — and file
  entries point at their data with long descriptors.

Options make the reader's harder paths happen on purpose: files split into
many extents stored out of order, descriptor lists that overflow into an
AED, extended file entries, directories embedded in their entry, 16-bit
names.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

SECTOR = 2048
ANCHOR = 256
VDS = 32
PART_START = 300


def _tag(tag_id: int, location: int, body: bytes) -> bytes:
    """A descriptor: 16-byte tag (checksum filled in) + body, padded to a sector."""
    head = bytearray(struct.pack("<HHBBHHHI", tag_id, 3, 0, 0, 0, 0, 0, location))
    head[4] = (sum(head[0:4]) + sum(head[5:16])) & 0xFF
    out = bytes(head) + body
    return out + b"\x00" * (-len(out) % SECTOR)


def _regid(ident: str, suffix: bytes = b"") -> bytes:
    return b"\x00" + ident.encode("ascii").ljust(23, b"\x00") + suffix.ljust(8, b"\x00")


def _cs0(name: str, wide: bool) -> bytes:
    if wide or any(ord(c) > 255 for c in name):
        return b"\x10" + name.encode("utf-16-be")
    return b"\x08" + name.encode("latin-1")


def _short(length: int, lbn: int, typ: int = 0) -> bytes:
    return struct.pack("<II", (typ << 30) | length, lbn)


def _long(length: int, lbn: int, ref: int, typ: int = 0) -> bytes:
    return struct.pack("<IIH", (typ << 30) | length, lbn, ref) + b"\x00" * 6


class _Dir:
    def __init__(self, name: str):
        self.name = name
        self.dirs: Dict[str, "_Dir"] = {}
        self.files: Dict[str, bytes] = {}


def build(files: Dict[str, bytes], *, metadata: bool = False, extent_blocks: int = 0,
          scramble: bool = False, aed_after: int = 0, efe: bool = False,
          embed_dirs: bool = False, wide_names: bool = False) -> bytes:
    """An image holding *files* ("BDMV/PLAYLIST/00800.mpls" -> bytes)."""
    root = _Dir("")
    for path, data in files.items():
        parts = [p for p in path.split("/") if p]
        node = root
        for part in parts[:-1]:
            node = node.dirs.setdefault(part, _Dir(part))
        node.files[parts[-1]] = data

    meta_ref = 1 if metadata else 0      # partition map index holding the tree
    data_ref = 0                         # the physical partition

    # ---- pass 1: number every metadata block --------------------------------
    meta: List[Optional[bytes]] = []     # filled in pass 2
    lbns: Dict[int, int] = {}            # id(obj) -> its entry's lbn

    def alloc() -> int:
        meta.append(None)
        return len(meta) - 1

    fsd_lbn = alloc()
    plan: List[Tuple[str, object, int]] = []

    def number(d: _Dir, parent_lbn: Optional[int]) -> None:
        lbns[id(d)] = alloc()
        plan.append(("dir", d, parent_lbn if parent_lbn is not None else lbns[id(d)]))
        for name, data in d.files.items():
            key = (id(d), name)
            lbns[hash(key)] = alloc()
            plan.append(("file", key, 0))
        for sub in d.dirs.values():
            number(sub, lbns[id(d)])

    number(root, None)

    # ---- data layout (physical partition) ------------------------------------
    chunks: List[Tuple[Tuple[int, str], int, bytes]] = []   # (file key, index, bytes)
    runs_of: Dict[Tuple[int, str], List[Tuple[int, int]]] = {}
    for kind, obj, _ in plan:
        if kind != "file":
            continue
        d_id, name = obj
        data = next(d for d in _dirs(root) if id(d) == d_id).files[name]
        step = (extent_blocks or max(1, -(-len(data) // SECTOR))) * SECTOR
        pieces = [data[i:i + step] for i in range(0, max(len(data), 1), step)] or [b""]
        for i, piece in enumerate(pieces):
            chunks.append((obj, i, piece))
    order = list(reversed(chunks)) if scramble else chunks

    # directory data blocks and AEDs are metadata blocks too
    dir_data: Dict[int, Tuple[bytes, Optional[int], int]] = {}
    aeds: Dict[Tuple[int, str], int] = {}

    def fids(d: _Dir, parent_lbn: int) -> bytes:
        out = b""

        def one(chars: int, ident: bytes, lbn: int) -> bytes:
            body = struct.pack("<HBB", 1, chars, len(ident)) + _long(SECTOR, lbn, meta_ref) \
                + struct.pack("<H", 0) + ident
            raw = _tag(257, 0, body)[: 16 + len(body)]
            return raw + b"\x00" * (-len(raw) % 4)

        out += one(0x0A, b"", parent_lbn)
        for sub in d.dirs.values():
            out += one(0x02, _cs0(sub.name, wide_names), lbns[id(sub)])
        for name in d.files:
            out += one(0x00, _cs0(name, wide_names), lbns[hash((id(d), name))])
        return out

    for kind, obj, parent in plan:
        if kind == "dir":
            raw = fids(obj, parent)
            if embed_dirs and len(raw) <= SECTOR - 216:
                dir_data[id(obj)] = (raw, None, 0)
            else:
                first = len(meta)
                for _ in range(-(-len(raw) // SECTOR)):
                    alloc()
                dir_data[id(obj)] = (raw, first, len(raw))
        elif aed_after:
            n_runs = sum(1 for key, _i, _p in chunks if key == obj)
            if n_runs > aed_after:
                aeds[obj] = alloc()

    k = len(meta)
    data_base = (1 + k) if metadata else k
    at = data_base
    for key, i, piece in order:
        blocks = max(1, -(-len(piece) // SECTOR))
        runs_of.setdefault(key, []).append((i, at, len(piece), blocks))
        at += blocks
    total_blocks = at

    # ---- pass 2: write every metadata block ----------------------------------
    # UDF 2.50 requires extended file entries (libudfread refuses a metadata
    # file recorded as a plain FE), so the Blu-ray layout always uses them
    use_efe = efe or metadata

    def entry(file_type: int, size: int, ads: bytes, ad_kind: int, loc: int,
              blocks: int) -> bytes:
        icb = struct.pack("<IHHHBB", 0, 4, 0, 1, 0, file_type) + b"\x00" * 6 \
            + struct.pack("<H", ad_kind)
        if use_efe:
            body = icb + struct.pack("<IIIHBBI", 0, 0, 0x14A5, 1, 0, 0, 0) \
                + struct.pack("<QQQ", size, size, blocks) + b"\x00" * 48 + b"\x00" * 8 \
                + b"\x00" * 32 + _regid("*MovieTranslator") + b"\x00" * 8 \
                + struct.pack("<II", 0, len(ads)) + ads
            return _tag(266, loc, body)
        body = icb + struct.pack("<IIIHBBI", 0, 0, 0x14A5, 1, 0, 0, 0) \
            + struct.pack("<QQ", size, blocks) + b"\x00" * 36 + b"\x00" * 4 \
            + b"\x00" * 16 + _regid("*MovieTranslator") + b"\x00" * 8 \
            + struct.pack("<II", 0, len(ads)) + ads
        return _tag(261, loc, body)

    for kind, obj, _ in plan:
        if kind == "dir":
            raw, first, length = dir_data[id(obj)]
            lbn = lbns[id(obj)]
            if first is None:
                meta[lbn] = entry(4, len(raw), raw, 3, lbn, 0)
            else:
                meta[lbn] = entry(4, len(raw), _short(length, first), 0, lbn,
                                  -(-length // SECTOR))
                for i in range(-(-length // SECTOR)):
                    meta[first + i] = raw[i * SECTOR:(i + 1) * SECTOR].ljust(SECTOR, b"\x00")
        else:
            lbn = lbns[hash(obj)]
            runs = sorted(runs_of[obj])
            size = sum(r[2] for r in runs)
            if metadata:
                ads_list = [_long(length, where, data_ref) for _i, where, length, _b in runs]
                ad_kind, cont = 1, lambda where: _long(SECTOR, where, meta_ref, typ=3)
            else:
                ads_list = [_short(length, where) for _i, where, length, _b in runs]
                ad_kind, cont = 0, lambda where: _short(SECTOR, where, typ=3)
            if obj in aeds:
                head, tail = ads_list[:aed_after], ads_list[aed_after:]
                aed_lbn = aeds[obj]
                inner = b"".join(tail)
                meta[aed_lbn] = _tag(258, aed_lbn, struct.pack("<II", 0, len(inner)) + inner)
                ads = b"".join(head) + cont(aed_lbn)
            else:
                ads = b"".join(ads_list)
            meta[lbn] = entry(5, size, ads, ad_kind, lbn, sum(r[3] for r in runs))
    fsd = b"\x00" * 12 + struct.pack("<HHII", 3, 3, 1, 1) + struct.pack("<II", 0, 0) \
        + b"\x00" * 64 + b"\x00" * 128 + b"\x00" * 64 + b"\x00" * 32 + b"\x00" * 32 \
        + b"\x00" * 32 + _long(SECTOR, lbns[id(root)], meta_ref) + _regid("*OSTA UDF Compliant")
    meta[fsd_lbn] = _tag(256, fsd_lbn, fsd)

    # ---- the partition -------------------------------------------------------
    blocks = bytearray(total_blocks * SECTOR)
    if metadata:
        mfe = entry(250, k * SECTOR, _short(k * SECTOR, 1), 0, 0, k)
        blocks[0:SECTOR] = mfe
        base = 1
    else:
        base = 0
    for i, block in enumerate(meta):
        blocks[(base + i) * SECTOR:(base + i + 1) * SECTOR] = (block or b"\x00" * SECTOR)[:SECTOR]
    for key, i, piece in order:
        _i, where, length, n = next(r for r in runs_of[key] if r[0] == i)
        blocks[where * SECTOR: where * SECTOR + len(piece)] = piece

    # ---- volume structures ---------------------------------------------------
    image = bytearray(PART_START * SECTOR) + blocks + bytearray(SECTOR * 2)
    # ECMA-167 volume recognition: the reader here does not need it, but
    # every real image has it and libudfread refuses an image without it —
    # which is how its absence was found (libbluray as the reference)
    for i, ident in enumerate((b"BEA01", b"NSR03" if metadata else b"NSR02", b"TEA01")):
        image[(16 + i) * SECTOR:(16 + i) * SECTOR + 7] = b"\x00" + ident + b"\x01"
    avdp = _tag(2, ANCHOR, struct.pack("<IIII", 16 * SECTOR, VDS, 16 * SECTOR, VDS))
    image[ANCHOR * SECTOR:(ANCHOR + 1) * SECTOR] = avdp
    pd = struct.pack("<IHH", 1, 1, 0) + _regid("+NSR03") + b"\x00" * 128 \
        + struct.pack("<III", 1, PART_START, total_blocks)
    image[VDS * SECTOR:(VDS + 1) * SECTOR] = _tag(5, VDS, pd)
    maps = bytes([1, 6]) + struct.pack("<HH", 1, 0)
    if metadata:
        maps += bytes([2, 64, 0, 0]) + _regid("*UDF Metadata Partition", b"\x50\x02") \
            + struct.pack("<HHIIIIHB", 1, 0, 0, 0, 0xFFFFFFFF, 32, 1, 0) + b"\x00" * 5
    lvd = struct.pack("<I", 2) + b"\x00" * 64 + b"\x00" * 128 + struct.pack("<I", SECTOR) \
        + _regid("*OSTA UDF Compliant") + _long(SECTOR, fsd_lbn, meta_ref) \
        + struct.pack("<II", len(maps), 2 if metadata else 1) \
        + _regid("*MovieTranslator") + b"\x00" * 128 + b"\x00" * 8 + maps
    image[(VDS + 1) * SECTOR:(VDS + 2) * SECTOR] = _tag(6, VDS + 1, lvd)
    image[(VDS + 2) * SECTOR:(VDS + 3) * SECTOR] = _tag(8, VDS + 2, b"")
    return bytes(image)


def _dirs(d: _Dir):
    yield d
    for sub in d.dirs.values():
        yield from _dirs(sub)


def disc_files(root) -> Dict[str, bytes]:
    """Every file under a disc folder, as the image's paths."""
    from pathlib import Path

    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}
