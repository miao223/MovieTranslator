"""DVD-Video (VIDEO_TS) metadata: titles, program chains, cells, streams.

The IFO files are tables of fixed-width big-endian fields; offsets and bit
layouts follow libdvdread's ifo_types.h, and the stream-id mapping follows
FFmpeg's dvdvideo demuxer (libavformat/dvdvideodec.c), which is the
reference this program's DVD support is checked against.

The model a title reduces to: the cells of its program chain(s), each a
range of sectors in its title set's VOB files. Two titles are the same
content exactly where they share sectors of the same title set, which is
why cells are described by sector ranges, not by title numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services import audio
from app.services.disc.binary import DiscError, Reader
from app.services.disc.fs import ConcatReader, DiscFS, Extent, Located, sub_extents
from app.services.disc.model import Disc, Segment, Stream, Title

SECTOR = 2048

# VTS audio coding mode -> (display name, first private stream id)
AUDIO_FORMATS = {
    0: ("AC-3", 0x80), 2: ("MPEG 音频", 0x1C0), 3: ("MPEG 音频", 0x1C0),
    4: ("LPCM", 0xA0), 6: ("DTS", 0x88),
}
CHANNEL_NAMES = {1: "单声道", 2: "立体声", 6: "5.1"}

# ISO 639-1 (what a DVD stores) -> ISO 639-2/B (what an MKV track carries).
# audio.canon_language covers the common ones; this fills in the rest a DVD
# is likely to name. Anything unknown becomes "und" — a two-letter code in a
# Matroska language field is simply wrong.
_ISO1_TO_2B = {
    "af": "afr", "sq": "alb", "am": "amh", "hy": "arm", "az": "aze", "eu": "baq",
    "be": "bel", "bn": "ben", "bs": "bos", "bg": "bul", "my": "bur", "ca": "cat",
    "hr": "hrv", "et": "est", "fo": "fao", "tl": "tgl", "gl": "glg", "ka": "geo",
    "gu": "guj", "ht": "hat", "ha": "hau", "is": "ice", "ga": "gle", "jv": "jav",
    "kn": "kan", "kk": "kaz", "km": "khm", "ky": "kir", "ku": "kur", "lo": "lao",
    "la": "lat", "lv": "lav", "lt": "lit", "lb": "ltz", "mk": "mac", "mg": "mlg",
    "ml": "mal", "mt": "mlt", "mi": "mao", "mr": "mar", "mn": "mon", "ne": "nep",
    "nb": "nob", "nn": "nno", "ps": "pus", "pa": "pan", "qu": "que", "sa": "san",
    "gd": "gla", "sr": "srp", "sd": "snd", "si": "sin", "sk": "slo", "sl": "slv",
    "so": "som", "sw": "swa", "tg": "tgk", "ta": "tam", "tt": "tat", "te": "tel",
    "bo": "tib", "ur": "urd", "uz": "uzb", "cy": "wel", "xh": "xho", "yi": "yid",
    "yo": "yor", "zu": "zul", "iw": "heb", "in": "ind", "ji": "yid",
}


def iso639_2(code: str) -> str:
    code = (code or "").strip().lower()
    if not re.fullmatch(r"[a-z]{2}", code):
        return ""
    canon = audio.canon_language(code)
    if len(canon) == 3:
        return canon
    return _ISO1_TO_2B.get(code, "und")


@dataclass
class DvdAudio:
    fmt: int
    channels: int
    language: str
    code_ext: int
    quant: int = 0
    rate: int = 0


@dataclass
class DvdSubp:
    language: str
    code_ext: int
    lang_ext: int = 0


@dataclass
class Cell:
    first: int            # first VOBU start sector, in the title set's VOB space
    last: int             # last VOBU end sector (inclusive)
    seconds: float
    vob_id: int = 0
    cell_id: int = 0
    block_mode: int = 0   # 0 none, 1 first, 2 middle, 3 last
    block_type: int = 0   # 1 = angle block
    seamless: bool = False
    interleaved: bool = False
    stc_discontinuity: bool = False


@dataclass
class Pgc:
    seconds: float
    programs: List[int]            # entry cell number (1-based) per program
    cells: List[Cell]
    audio_control: List[int]
    subp_control: List[int]
    palette: List[int]             # 16 x 0x00YYCrCb


@dataclass
class Vts:
    number: int
    video: int                     # the 16-bit video attribute word
    audio: List[DvdAudio]
    subp: List[DvdSubp]
    ptts: List[List[Tuple[int, int]]]
    pgcs: List[Pgc]

    @property
    def mpeg(self) -> int:
        return 1 if (self.video >> 14) & 3 == 0 else 2

    @property
    def pal(self) -> bool:
        return (self.video >> 12) & 3 == 1

    @property
    def widescreen(self) -> bool:
        return (self.video >> 10) & 3 == 3

    @property
    def size(self) -> Tuple[int, int]:
        height = 576 if self.pal else 480
        picture = (self.video >> 2) & 3
        width = {0: 720, 1: 704, 2: 352, 3: 352}[picture]
        return width, height // 2 if picture == 3 else height


@dataclass
class VmgTitle:
    number: int
    angles: int
    chapters: int
    vts: int
    vts_ttn: int


@dataclass
class DvdPlan:
    """What remux.py needs to write one title."""

    vts: int
    vobs: List[str]                         # fs paths of VTS_nn_1..9.VOB
    cells: List[Cell]                       # in playing order (angle 1)
    chapters: List[float]
    audio: List[Tuple[int, DvdAudio]]       # (stream id, attributes)
    subp: List[Tuple[int, DvdSubp]]
    palette: List[int]
    size: Tuple[int, int]
    widescreen: bool


# ----------------------------------------------------------------- parsing

def _bcd(b: int) -> int:
    return (b >> 4) * 10 + (b & 0x0F)


def bcd_seconds(raw: bytes) -> float:
    h, m, s, f = raw
    fps = {1: 25.0, 3: 30000 / 1001}.get(f >> 6, 30.0)
    return _bcd(h) * 3600 + _bcd(m) * 60 + _bcd(s) + _bcd(f & 0x3F) / fps


def parse_vmg(data: bytes) -> List[VmgTitle]:
    r = Reader(data, "VIDEO_TS.IFO")
    if r.text(12) != "DVDVIDEO-VMG":
        raise DiscError("VIDEO_TS.IFO 不是 DVD 的索引文件")
    t = r.at(r.at(0xC4).u32() * SECTOR)
    n = t.u16()
    t.u16()
    t.u32()
    titles = []
    for i in range(n):
        t.u8()                             # playback type
        angles = t.u8()
        chapters = t.u16()
        t.u16()                            # parental mask
        vts, ttn = t.u8(), t.u8()
        t.u32()                            # VTS start sector
        titles.append(VmgTitle(i + 1, max(angles, 1), chapters, vts, ttn))
    return titles


def _pgc(r: Reader, start: int) -> Pgc:
    q = r.at(start)
    q.u16()
    n_programs, n_cells = q.u8(), q.u8()
    seconds = bcd_seconds(q.raw(4))
    q.u32()                                # prohibited user operations
    audio_control = [q.u16() for _ in range(8)]
    subp_control = [q.u32() for _ in range(32)]
    q.skip(6)                              # next / prev / go-up PGC
    q.u8()
    q.u8()
    palette = [q.u32() for _ in range(16)]
    q.u16()                                # command table offset
    prog_off, play_off, pos_off = q.u16(), q.u16(), q.u16()
    programs = []
    if prog_off and n_programs:
        p = r.at(start + prog_off)
        programs = [p.u8() for _ in range(n_programs)]
    cells = []
    if play_off and n_cells:
        c = r.at(start + play_off)
        for _ in range(n_cells):
            b0 = c.u8()
            c.skip(3)
            t = bcd_seconds(c.raw(4))
            first = c.u32()
            c.u32()                        # first ILVU end sector
            c.u32()                        # last VOBU start sector
            last = c.u32()
            cells.append(Cell(first, last, t, block_mode=b0 >> 6,
                              block_type=(b0 >> 4) & 3,
                              seamless=bool((b0 >> 3) & 1),
                              interleaved=bool((b0 >> 2) & 1),
                              stc_discontinuity=bool((b0 >> 1) & 1)))
        if pos_off:
            p = r.at(start + pos_off)
            for cell in cells:
                cell.vob_id = p.u16()
                p.u8()
                cell.cell_id = p.u8()
    return Pgc(seconds, programs, cells, audio_control, subp_control, palette)


def parse_vts(data: bytes, number: int) -> Vts:
    label = f"VTS_{number:02d}_0.IFO"
    r = Reader(data, label)
    if r.text(12) != "DVDVIDEO-VTS":
        raise DiscError(f"{label} 不是 DVD 标题集文件")
    ptt_sector = r.at(0xC8).u32()
    pgcit_sector = r.at(0xCC).u32()
    video = r.at(0x200).u16()

    a = r.at(0x202)
    n_audio = min(a.u16(), 8)
    streams = []
    for _ in range(n_audio):
        raw = a.raw(8)
        lang = raw[2:4].decode("ascii", "replace") if (raw[0] >> 2) & 3 == 1 else ""
        streams.append(DvdAudio(
            fmt=raw[0] >> 5, channels=(raw[1] & 7) + 1, language=iso639_2(lang),
            code_ext=raw[5], quant=raw[1] >> 6, rate=(raw[1] >> 4) & 3))
    s = r.at(0x254)
    n_subp = min(s.u16(), 32)
    subp = []
    for _ in range(n_subp):
        raw = s.raw(6)
        lang = raw[2:4].decode("ascii", "replace") if raw[0] & 3 == 1 else ""
        subp.append(DvdSubp(iso639_2(lang), raw[5], raw[4]))

    ptts: List[List[Tuple[int, int]]] = []
    if ptt_sector:
        base = ptt_sector * SECTOR
        p = r.at(base)
        n_titles = p.u16()
        p.u16()
        end = p.u32()
        offsets = [p.u32() for _ in range(n_titles)]
        for i, off in enumerate(offsets):
            stop = offsets[i + 1] if i + 1 < n_titles else end + 1
            q = r.at(base + off)
            ptts.append([(q.u16(), q.u16()) for _ in range(max(stop - off, 0) // 4)])

    pgcs: List[Pgc] = []
    if pgcit_sector:
        base = pgcit_sector * SECTOR
        p = r.at(base)
        n_pgcs = p.u16()
        p.u16()
        p.u32()
        for _ in range(n_pgcs):
            p.u32()                        # category
            pgcs.append(_pgc(r, base + p.u32()))
    return Vts(number, video, streams, subp, ptts, pgcs)


# ------------------------------------------------------------- the disc

def _title_cells(vts: Vts, ttn: int) -> Tuple[List[Tuple[int, Cell]], List[float], Optional[Pgc]]:
    """The cells a title plays (angle 1), with its chapter start times.

    A program chain plays its cells in order once entered, so a title is
    every cell of each of its chains from the first program it names
    onwards — not only the cells its chapter list happens to point at.
    Inside an angle block only the first cell (angle 1) plays.
    """
    if not 1 <= ttn <= len(vts.ptts):
        return [], [], None
    ptts = vts.ptts[ttn - 1]
    first_pg: Dict[int, int] = {}
    order: List[int] = []
    for pgcn, pgn in ptts:
        if 1 <= pgcn <= len(vts.pgcs) and pgcn not in first_pg:
            first_pg[pgcn] = pgn
            order.append(pgcn)
    played: List[Tuple[int, int, Cell]] = []
    for pgcn in order:
        pgc = vts.pgcs[pgcn - 1]
        pg = first_pg[pgcn]
        entry = pgc.programs[pg - 1] if 1 <= pg <= len(pgc.programs) else 1
        for idx in range(max(entry, 1), len(pgc.cells) + 1):
            cell = pgc.cells[idx - 1]
            if cell.block_type == 1 and cell.block_mode not in (0, 1):
                continue                   # angles 2.. of an angle block
            played.append((pgcn, idx, cell))
    starts: Dict[Tuple[int, int], float] = {}
    t = 0.0
    for pgcn, idx, cell in played:
        starts[(pgcn, idx)] = t
        t += cell.seconds
    chapters: List[float] = []
    for pgcn, pgn in ptts:
        if pgcn not in first_pg:
            continue
        pgc = vts.pgcs[pgcn - 1]
        if not 1 <= pgn <= len(pgc.programs):
            continue
        at = starts.get((pgcn, pgc.programs[pgn - 1]))
        if at is not None and (not chapters or at - chapters[-1] >= 0.5):
            chapters.append(round(at, 3))
    main = vts.pgcs[order[0] - 1] if order else None
    return [(pgcn, cell) for pgcn, _, cell in played], chapters or [0.0], main


def _streams(vts: Vts, pgc: Pgc) -> Tuple[List[Stream], List[Tuple[int, DvdAudio]], List[Tuple[int, DvdSubp]]]:
    width, height = vts.size
    video = Stream("video", f"MPEG-{vts.mpeg}", 0x1E0,
                   detail=f"{width}x{height} {'16:9' if vts.widescreen else '4:3'} "
                          f"{'PAL' if vts.pal else 'NTSC'}")
    out = [video]
    audios: List[Tuple[int, DvdAudio]] = []
    for i, control in enumerate(pgc.audio_control):
        if not control & 0x8000 or i >= len(vts.audio):
            continue
        attr = vts.audio[i]
        name, base = AUDIO_FORMATS.get(attr.fmt, (f"格式 {attr.fmt}", 0))
        position = (control >> 8) & 0x07
        sid = base + position if base else 0
        s = Stream("audio", name, sid, language=attr.language,
                   detail=CHANNEL_NAMES.get(attr.channels, f"{attr.channels} 声道"),
                   channels=attr.channels, commentary=attr.code_ext in (3, 4))
        if not sid:
            s.carried, s.note = False, "未知的音频格式"
        elif attr.fmt == 4:
            s.note = "LPCM 会无损转成 FLAC（MKV 装不下 DVD 的 LPCM 格式）"
        out.append(s)
        if s.carried:
            audios.append((sid, attr))
    subs: List[Tuple[int, DvdSubp]] = []
    for i, control in enumerate(pgc.subp_control):
        if not control & 0x80000000 or i >= len(vts.subp):
            continue
        # the stream id to read depends on the picture: the widescreen
        # variant for 16:9 titles, the 4:3 one otherwise (what a player
        # shows on a 16:9 screen, and what FFmpeg's demuxer picks)
        position = (control >> 16) & 0x1F if vts.widescreen else (control >> 24) & 0x1F
        attr = vts.subp[i]
        sid = 0x20 + position
        if any(sid == got for got, _ in subs):
            continue
        out.append(Stream("subtitle", "VobSub", sid, language=attr.language,
                          forced=9 in (attr.code_ext, attr.lang_ext)))
        subs.append((sid, attr))
    return out, audios, subs


def _vob_files(fs: DiscFS, prefix: str, vts: int) -> List[str]:
    found = []
    for n in range(1, 10):
        path = fs.lookup(f"{prefix}VTS_{vts:02d}_{n}.VOB")
        if path:
            found.append(path)
    return found


def _read_ifo(fs: DiscFS, prefix: str, name: str) -> bytes:
    """An IFO, or its .BUP twin when the IFO itself is unreadable."""
    try:
        return fs.read(prefix + name)
    except (DiscError, OSError) as first:
        try:
            return fs.read(prefix + name[:-4] + ".BUP")
        except (DiscError, OSError):
            raise first


def load(located: Located) -> Disc:
    fs = located.fs
    prefix = f"{located.ifo_dir}/" if located.ifo_dir else ""
    disc = Disc(kind="dvd", source=fs.source, path=str(located.root),
                root=str(located.root), name=located.name, titles=[], fs=fs)
    vmg = parse_vmg(_read_ifo(fs, prefix, "VIDEO_TS.IFO"))
    sets: Dict[int, Optional[Vts]] = {}
    vob_sets: Dict[int, List[str]] = {}
    bad: List[str] = []
    for vt in vmg:
        if vt.vts not in sets:
            try:
                sets[vt.vts] = parse_vts(
                    _read_ifo(fs, prefix, f"VTS_{vt.vts:02d}_0.IFO"), vt.vts)
            except DiscError as exc:
                sets[vt.vts] = None
                bad.append(str(exc))
            vob_sets[vt.vts] = _vob_files(fs, prefix, vt.vts)
        vts = sets[vt.vts]
        if vts is None:
            continue
        played, chapters, pgc = _title_cells(vts, vt.vts_ttn)
        if not played or pgc is None:
            continue
        streams, audios, subs = _streams(vts, pgc)
        segments = [Segment(f"vts{vt.vts}", cell.first, cell.last + 1, cell.seconds,
                            seamless=n > 0 and cell.seamless)
                    for n, (_, cell) in enumerate(played)]
        title = Title(
            id=f"title{vt.number:02d}",
            number=vt.number,
            duration=round(sum(c.seconds for _, c in played), 3),
            segments=segments,
            chapters=chapters,
            streams=streams,
            angles=vt.angles,
            plan=DvdPlan(vt.vts, vob_sets[vt.vts], [c for _, c in played], chapters,
                         audios, subs, pgc.palette, vts.size, vts.widescreen),
        )
        title.size = sum((c.last - c.first + 1) * SECTOR for _, c in played)
        if vt.angles > 1 or any(c.block_type == 1 for _, c in played):
            title.problems.append("多角度标题：各角度的画面交错存放，暂不支持导出")
        disc.titles.append(title)
    if bad:
        disc.warnings.append("以下标题集读不出来，其中的标题已忽略：" + "；".join(bad[:5]))
    disc.analysis_only = not any(vob_sets.values())
    for title in disc.titles:
        if disc.analysis_only:
            title.size = None
        elif not title.plan.vobs:
            title.problems.append(f"缺少 VTS_{title.plan.vts:02d}_1.VOB 等视频文件")
            title.size = None
    disc.encrypted = _encrypted(fs, disc)
    return disc


def vob_extents(fs: DiscFS, plan: DvdPlan) -> List[Extent]:
    """The title set's VOB files end to end: the space cell sectors count in."""
    return [e for vob in plan.vobs for e in fs.extents(vob)]


def _encrypted(fs: DiscFS, disc: Disc) -> bool:
    """Checked on the first 64 sectors of the longest title."""
    if disc.analysis_only or not disc.titles:
        return False
    plan = max(disc.titles, key=lambda t: t.duration).plan
    if not plan.vobs or not plan.cells:
        return False
    try:
        start = plan.cells[0].first * SECTOR
        with ConcatReader(sub_extents(vob_extents(fs, plan), start, 64 * SECTOR)) as fh:
            return looks_scrambled(fh.read())
    except (DiscError, OSError):
        return False


def looks_scrambled(data: bytes) -> bool:
    """Is any pack in *data* CSS-scrambled?

    Scrambling is flagged per PES packet (PES_scrambling_control, two bits
    of the header's first flag byte). NAV packs never are, so a clean first
    sector proves nothing — the caller hands over several.
    """
    for base in range(0, len(data) - 14, SECTOR):
        pack = data[base:base + SECTOR]
        if pack[:4] != b"\x00\x00\x01\xba":
            continue
        pos = 14 + (pack[13] & 7) if pack[4] >> 6 == 1 else 12
        while pos + 9 <= len(pack) and pack[pos:pos + 3] == b"\x00\x00\x01":
            sid = pack[pos + 3]
            length = int.from_bytes(pack[pos + 4:pos + 6], "big")
            if (sid == 0xBD or 0xC0 <= sid <= 0xEF) and (pack[pos + 6] >> 4) & 3:
                return True
            pos += 6 + length
    return False
