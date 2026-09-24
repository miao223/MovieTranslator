"""Blu-ray and DVD metadata written from scratch, for tests.

A real disc cannot ship in the repository — not even its metadata, which
is taken from someone's film — so the parsers are tested against files
built here field by field, the way tests/pgs.py builds PGS bitstreams.
Each writer is the inverse of one parser in app/services/disc and follows
the same libbluray / libdvdread layouts, so a round trip checks both.

Only what the parsers read is filled in; everything else is zero, which is
also what a real disc has in most reserved fields.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TICKS = 45000
SECTOR = 2048


def T(seconds: float) -> int:
    """Seconds -> MPLS/CLPI 45 kHz ticks."""
    return int(round(seconds * TICKS))


def _u8(v): return struct.pack(">B", v)
def _u16(v): return struct.pack(">H", v)
def _u32(v): return struct.pack(">I", v)


# ================================================================ Blu-ray

@dataclass
class S:
    """One STN stream entry."""

    group: str                 # video | audio | pg | ig | textst
    pid: int
    coding: int
    lang: str = ""
    fmt: int = 0
    rate: int = 0
    stream_type: int = 1


DEFAULT_STREAMS = (
    S("video", 0x1011, 0x1B, fmt=6, rate=1),          # H.264 1080p 23.976
    S("audio", 0x1100, 0x81, "jpn", fmt=6, rate=1),   # AC-3 5.1 Japanese
    S("pg", 0x1200, 0x90, "jpn"),
)


@dataclass
class Item:
    clip: str
    in_time: int
    out_time: int
    angles: Sequence[str] = ()
    connection: int = 1
    streams: Sequence[S] = DEFAULT_STREAMS
    secondary: int = 0
    dolby_vision: int = 0


def _stream_entry(s: S) -> bytes:
    if s.stream_type == 1:
        entry = _u8(1) + _u16(s.pid)
    else:
        entry = _u8(s.stream_type) + _u8(0) + _u8(0) + _u16(s.pid)
    lang = s.lang.encode("ascii").ljust(3, b"\x00")[:3]
    if s.group == "video":
        attrs = _u8(s.coding) + _u8((s.fmt << 4) | s.rate)
    elif s.group == "audio":
        attrs = _u8(s.coding) + _u8((s.fmt << 4) | s.rate) + lang
    elif s.group == "textst":
        attrs = _u8(s.coding) + _u8(1) + lang
    else:
        attrs = _u8(s.coding) + lang
    return _u8(len(entry)) + entry + _u8(len(attrs)) + attrs


def _stn(item: Item) -> bytes:
    groups = {g: [s for s in item.streams if s.group == g]
              for g in ("video", "audio", "pg", "textst", "ig")}
    subs = groups["pg"] + groups["textst"]
    body = (_u16(0) + _u8(len(groups["video"])) + _u8(len(groups["audio"]))
            + _u8(len(subs)) + _u8(len(groups["ig"]))
            + _u8(item.secondary) + _u8(0) + _u8(0) + _u8(item.dolby_vision)
            + b"\x00" * 4)
    for s in groups["video"] + groups["audio"] + subs + groups["ig"]:
        body += _stream_entry(s)
    return _u16(len(body)) + body


def _play_item(item: Item) -> bytes:
    multi = 1 if item.angles else 0
    body = (item.clip.encode("ascii") + b"M2TS"
            + _u16((multi << 4) | item.connection) + _u8(0)
            + _u32(item.in_time) + _u32(item.out_time)
            + b"\x00" * 8 + _u8(0) + _u8(0) + _u16(0))
    if multi:
        body += _u8(len(item.angles) + 1) + _u8(0)
        for clip in item.angles:
            body += clip.encode("ascii") + b"M2TS" + _u8(0)
    body += _stn(item)
    return _u16(len(body)) + body


def mpls(items: Sequence[Item], marks: Sequence[Tuple[int, int, int]] = (),
         chapters_every: Optional[float] = None) -> bytes:
    """A playlist. *marks* are (type, play item index, 45 kHz time);
    *chapters_every* adds an entry mark every so many seconds instead."""
    marks = list(marks)
    if chapters_every and not marks:
        for i, it in enumerate(items):
            t = it.in_time
            while t < it.out_time:
                marks.append((1, i, t))
                t += T(chapters_every)
    playlist = _u16(0) + _u16(len(items)) + _u16(0)
    for it in items:
        playlist += _play_item(it)
    playlist = _u32(len(playlist)) + playlist
    mark_table = _u16(len(marks)) + b"".join(
        _u8(0) + _u8(kind) + _u16(item) + _u32(time) + _u16(0xFFFF) + _u32(0)
        for kind, item, time in marks)
    mark_table = _u32(len(mark_table)) + mark_table
    appinfo = _u8(0) + _u8(1) + _u16(0) + b"\x00" * 8 + _u16(0)
    appinfo = _u32(len(appinfo)) + appinfo
    list_start = 0x28 + len(appinfo)
    mark_start = list_start + len(playlist)
    head = b"MPLS0200" + _u32(list_start) + _u32(mark_start) + _u32(0) + b"\x00" * 20
    return head + appinfo + playlist + mark_table


def clpi(start: int, end: int, streams: Sequence[S] = DEFAULT_STREAMS,
         ep: Sequence[Tuple[int, int]] = (), packets: int = 0,
         ep_pid: int = 0x1011) -> bytes:
    """Clip info: presentation start/end (45 kHz), per-PID coding and
    language, and optionally an EP map of (pts 45 kHz, SPN) points."""
    clipinfo = (_u16(0) + _u8(1) + _u8(1) + _u32(0) + _u32(6_000_000)
                + _u32(packets) + b"\x00" * 128)
    clipinfo = _u32(len(clipinfo)) + clipinfo
    seq = (_u8(0) + _u8(1) + _u32(0) + _u8(1) + _u8(0)
           + _u16(0x1001) + _u32(0) + _u32(start) + _u32(end))
    seq = _u32(len(seq)) + seq
    prog = _u8(0) + _u8(1) + _u32(0) + _u16(0x100) + _u8(len(streams)) + _u8(0)
    for s in streams:
        lang = s.lang.encode("ascii").ljust(3, b"\x00")[:3]
        if s.group == "video":
            attrs = _u8(s.coding) + _u8((s.fmt << 4) | s.rate) + _u8(0x30) + _u8(0)
        elif s.group == "audio":
            attrs = _u8(s.coding) + _u8((s.fmt << 4) | s.rate) + lang
        elif s.group == "textst":
            attrs = _u8(s.coding) + _u8(1) + lang
        else:
            attrs = _u8(s.coding) + lang
        prog += _u16(s.pid) + _u8(len(attrs)) + attrs
    prog = _u32(len(prog)) + prog
    if ep:
        # one coarse entry per point keeps the writer simple; the reader
        # has to combine coarse and fine exactly the same way either way
        coarse = b"".join(
            _u32((i << 14) | ((pts >> 18) & 0x3FFF)) + _u32(spn)
            for i, (pts, spn) in enumerate(ep))
        fine = b"".join(
            _u32((((pts >> 8) & 0x7FF) << 17) | (spn & 0x1FFFF)) for pts, spn in ep)
        block = _u32(4 + len(coarse)) + coarse + fine
        bits = (1 << 34) | (len(ep) << 18) | len(ep)       # type 1 | coarse | fine
        head = _u8(0) + _u8(1) + _u16(ep_pid) + bits.to_bytes(6, "big")
        head += _u32(len(head) + 4)                          # block follows the header
        cpi_body = _u16(1) + head + block
    else:
        cpi_body = b""
    cpi = _u32(len(cpi_body)) + cpi_body if cpi_body else _u32(0)
    seq_start = 0x28 + len(clipinfo)
    prog_start = seq_start + len(seq)
    cpi_start = prog_start + len(prog)
    head = (b"HDMV0200" + _u32(seq_start) + _u32(prog_start) + _u32(cpi_start)
            + _u32(0) + _u32(0) + b"\x00" * 12)
    return head + clipinfo + seq + prog + cpi


def index_bdmv(titles: Sequence[Tuple[str, int]] = (("hdmv", 0),)) -> bytes:
    """Titles as ("hdmv", movie object id) / ("bdj", _)."""
    def obj(kind: str, ref: int) -> bytes:
        if kind == "hdmv":
            return _u32(1 << 30) + _u16(0) + _u16(ref) + _u32(0)
        return _u32(2 << 30) + _u16(0) + b"00000" + _u8(0)

    body = obj(titles[0][0], 0) + obj(titles[0][0], 0) + _u16(len(titles))
    for kind, ref in titles:
        if kind == "hdmv":
            body += _u32(1 << 30) + _u16(0) + _u16(ref) + _u32(0)
        else:
            body += _u32(2 << 30) + _u16(0) + b"00001" + _u8(0)
    body = _u32(len(body)) + body
    appinfo = bytes(34)             # AppInfoBDMV: libbluray expects exactly 34
    start = 0x28 + 4 + len(appinfo)
    return b"INDX0200" + _u32(start) + _u32(0) + b"\x00" * 24 + _u32(len(appinfo)) + appinfo + body


def movie_objects(objects: Sequence[Sequence[int]]) -> bytes:
    """Movie objects, each a list of playlists it plays (PlayPL, immediate)."""
    body = _u32(0) + _u16(len(objects))
    for playlists in objects:
        body += _u16(0) + _u16(len(playlists))
        for n in playlists:
            # op_cnt 1 | grp BRANCH | sub PLAY ; imm_op1 | PLAY_PL
            body += bytes([(1 << 5) | 2, 0x80, 0, 0]) + _u32(n) + _u32(0)
    body = _u32(len(body)) + body
    return b"MOBJ0200" + _u32(0) + b"\x00" * 28 + body


def write_bd(root: Path, playlists: Dict[str, bytes], clips: Dict[str, bytes],
             streams: Optional[Dict[str, bytes]] = None,
             index: Optional[bytes] = None, mobj: Optional[bytes] = None,
             meta_title: str = "") -> Path:
    """A BDMV folder. *streams* maps clip id -> m2ts bytes; None = metadata only."""
    bdmv = root / "BDMV"
    for sub in ("PLAYLIST", "CLIPINF", "STREAM"):
        (bdmv / sub).mkdir(parents=True, exist_ok=True)
    for name, data in playlists.items():
        (bdmv / "PLAYLIST" / f"{name}.mpls").write_bytes(data)
    for name, data in clips.items():
        (bdmv / "CLIPINF" / f"{name}.clpi").write_bytes(data)
    for name, data in (streams or {}).items():
        (bdmv / "STREAM" / f"{name}.m2ts").write_bytes(data)
    (bdmv / "index.bdmv").write_bytes(index or index_bdmv())
    (bdmv / "MovieObject.bdmv").write_bytes(mobj or movie_objects([[]]))
    if meta_title:
        (bdmv / "META" / "DL").mkdir(parents=True, exist_ok=True)
        (bdmv / "META" / "DL" / "bdmt_eng.xml").write_text(
            f'<?xml version="1.0"?><disclib><di:discinfo><di:title>'
            f"<di:name>{meta_title}</di:name></di:title></di:discinfo></disclib>",
            encoding="utf-8")
    return root


def simple_bd(root: Path, playlists: Dict[str, Sequence[Tuple[str, float, float]]],
              clip_seconds: Optional[Dict[str, float]] = None, **kw) -> Path:
    """A BDMV folder from plain numbers: playlist -> [(clip, in s, out s)]."""
    clip_seconds = dict(clip_seconds or {})
    for items in playlists.values():
        for clip, _a, b in items:
            clip_seconds[clip] = max(clip_seconds.get(clip, 0.0), b)
    pls = {name: mpls([Item(c, T(a), T(b)) for c, a, b in items], chapters_every=300)
           for name, items in playlists.items()}
    clips = {c: clpi(0, T(s)) for c, s in clip_seconds.items()}
    return write_bd(root, pls, clips, **kw)


# ==================================================================== DVD

def bcd_time(seconds: float, fps: int = 25) -> bytes:
    whole = int(seconds)
    frames = int(round((seconds - whole) * fps))
    h, m, s = whole // 3600, (whole // 60) % 60, whole % 60

    def bcd(v):
        return ((v // 10) << 4) | (v % 10)

    return bytes([bcd(h), bcd(m), bcd(s), (1 << 6 if fps == 25 else 3 << 6) | bcd(frames)])


@dataclass
class C:
    """One cell: sectors [first, last] of its title set, playing *seconds*."""

    first: int
    last: int
    seconds: float
    vob_id: int = 1
    cell_id: int = 1
    block_mode: int = 0
    block_type: int = 0


@dataclass
class P:
    """One program chain."""

    cells: Sequence[C]
    programs: Sequence[int] = (1,)
    audio: Sequence[int] = (0x8000,)              # per stream: available | position<<8
    subp: Sequence[int] = (0x80000000,)           # available | 4:3<<24 | wide<<16 …
    palette: Sequence[int] = tuple(range(0x108080, 0x108080 + 16))


def _pgc(p: P) -> bytes:
    seconds = sum(c.seconds for c in p.cells)
    head = (_u16(0) + _u8(len(p.programs)) + _u8(len(p.cells)) + bcd_time(seconds)
            + _u32(0)
            + b"".join(_u16(v) for v in list(p.audio) + [0] * (8 - len(p.audio)))
            + b"".join(_u32(v) for v in list(p.subp) + [0] * (32 - len(p.subp)))
            + _u16(0) * 3 + _u8(0) + _u8(0)
            + b"".join(_u32(v) for v in p.palette))
    fixed = len(head) + 8
    programs = bytes(p.programs) + (b"\x00" if len(p.programs) % 2 else b"")
    playback = b"".join(
        _u8((c.block_mode << 6) | (c.block_type << 4)) + b"\x00\x00\x00"
        + bcd_time(c.seconds) + _u32(c.first) + _u32(0) + _u32(c.last) + _u32(c.last)
        for c in p.cells)
    positions = b"".join(_u16(c.vob_id) + _u8(0) + _u8(c.cell_id) for c in p.cells)
    prog_off = fixed
    play_off = prog_off + len(programs)
    pos_off = play_off + len(playback)
    return head + _u16(0) + _u16(prog_off) + _u16(play_off) + _u16(pos_off) \
        + programs + playback + positions


def _sectors(*blocks: bytes) -> bytes:
    return b"".join(b + b"\x00" * (-len(b) % SECTOR) for b in blocks)


def vmg_ifo(titles: Sequence[Tuple[int, int, int, int]]) -> bytes:
    """VIDEO_TS.IFO for titles given as (angles, chapters, vts, vts_ttn)."""
    head = bytearray(SECTOR)
    head[0:12] = b"DVDVIDEO-VMG"
    head[0xC4:0xC8] = _u32(1)
    table = _u16(len(titles)) + _u16(0) + _u32(8 + 12 * len(titles) - 1)
    for angles, chapters, vts, ttn in titles:
        table += _u8(0x3C) + _u8(angles) + _u16(chapters) + _u16(0) + _u8(vts) + _u8(ttn) + _u32(0)
    return _sectors(bytes(head), table)


def vts_ifo(pgcs: Sequence[P], ptts: Sequence[Sequence[Tuple[int, int]]],
            audio: Sequence[Tuple[int, int, str, int]] = ((0, 6, "ja", 1),),
            subp: Sequence[Tuple[str, int]] = (("zh", 1),),
            video: int = (1 << 14) | (0 << 12) | (3 << 10)) -> bytes:
    """VTS_nn_0.IFO. audio: (format, channels, lang, code_ext); subp: (lang, code_ext).
    Default video word: MPEG-2, NTSC, 16:9."""
    head = bytearray(SECTOR)
    head[0:12] = b"DVDVIDEO-VTS"
    head[0xC8:0xCC] = _u32(1)
    head[0x200:0x202] = _u16(video)
    head[0x202:0x204] = _u16(len(audio))
    for i, (fmt, channels, lang, ext) in enumerate(audio):
        at = 0x204 + 8 * i
        head[at:at + 8] = bytes([(fmt << 5) | (1 << 2), channels - 1]) \
            + lang.encode("ascii") + b"\x00" + bytes([ext]) + b"\x00\x00"
    head[0x254:0x256] = _u16(len(subp))
    for i, (lang, ext) in enumerate(subp):
        at = 0x256 + 6 * i
        head[at:at + 6] = bytes([1, 0]) + lang.encode("ascii") + b"\x00" + bytes([ext])

    offsets, entries = [], b""
    base = 8 + 4 * len(ptts)
    for chapters in ptts:
        offsets.append(base + len(entries))
        entries += b"".join(_u16(pgcn) + _u16(pgn) for pgcn, pgn in chapters)
    ptt = _u16(len(ptts)) + _u16(0) + _u32(base + len(entries) - 1) \
        + b"".join(_u32(o) for o in offsets) + entries

    bodies = [_pgc(p) for p in pgcs]
    table_head = 8 + 8 * len(bodies)
    pos, pointers = table_head, b""
    for body in bodies:
        pointers += _u32(0x81000000) + _u32(pos)
        pos += len(body)
    pgcit = _u16(len(bodies)) + _u16(0) + _u32(pos - 1) + pointers + b"".join(bodies)

    ptt_sectors = -(-len(ptt) // SECTOR)
    head[0xCC:0xD0] = _u32(1 + ptt_sectors)
    return _sectors(bytes(head), ptt, pgcit)


def write_dvd(root: Path, vmg: bytes, sets: Dict[int, bytes],
              vobs: Optional[Dict[int, Sequence[bytes]]] = None,
              in_video_ts: bool = True) -> Path:
    """A DVD folder: VIDEO_TS/ with the IFOs (and VOBs when given)."""
    folder = root / "VIDEO_TS" if in_video_ts else root
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "VIDEO_TS.IFO").write_bytes(vmg)
    for n, data in sets.items():
        (folder / f"VTS_{n:02d}_0.IFO").write_bytes(data)
    for n, parts in (vobs or {}).items():
        for i, data in enumerate(parts, start=1):
            (folder / f"VTS_{n:02d}_{i}.VOB").write_bytes(data)
    return root


# ============================================================ streams

@dataclass
class Clip:
    """A synthetic stream file and what an authoring tool would record about it."""

    data: bytes
    start: int                     # first picture, 45 kHz (MPLS/CLPI units)
    end: int                       # end of the last picture, 45 kHz
    frames: int
    pids: Dict[str, List[int]] = field(default_factory=dict)   # kind -> ids
    keyframes: List[Tuple[int, int]] = field(default_factory=list)  # (pts 45k, SPN)


def _tone(n: int, rate: int, channels: int, fmt: str):
    import numpy as np

    t = np.arange(n) / rate
    wave = 0.3 * np.sin(2 * np.pi * 440 * t)
    if fmt == "fltp":
        return np.repeat(wave[None, :], channels, axis=0).astype(np.float32)
    if fmt == "s16":
        return (np.repeat(wave[:, None], channels, axis=1).reshape(1, -1) * 32767).astype(np.int16)
    # s32 holding 24-bit samples, as a disc's 24-bit LPCM decodes
    samples = (np.repeat(wave[:, None], channels, axis=1).reshape(1, -1) * (2 ** 23 - 1)).astype(np.int32)
    return samples << 8


def _probe(data: bytes, fmt: str, fps: int) -> Tuple[int, int, int, Dict[str, List[int]], List[Tuple[int, int]]]:
    import io

    import av

    with av.open(io.BytesIO(data), format=fmt) as c:
        pids: Dict[str, List[int]] = {}
        for s in c.streams:
            pids.setdefault(s.type, []).append(s.id)
        pts, keys = [], []
        # a VOB whose subtitle stream first appears mid-file trips PyAV's
        # end-of-stream flush (see app/services/disc/remux.packets)
        from app.services.disc.remux import packets

        for p in packets(c):
            if p.stream.type != "video" or p.pts is None or not p.size:
                continue
            pts.append(p.pts)
            if p.is_keyframe and p.pos is not None:
                # EP maps count 192-byte source packets, not bytes
                keys.append((p.pts // 2, p.pos // 192))
    step = 90000 // fps
    return min(pts) // 2, (max(pts) + step) // 2, len(pts), pids, keys


def make_m2ts(seconds: float = 2.0, *, fps: int = 24, audio: str = "ac3",
              channels: int = 2, wide: bool = False, start: float = 0.0,
              cues: Sequence[Tuple[float, float, str]] = (),
              tmp: Optional[Path] = None) -> Clip:
    """A short Blu-ray m2ts: H.264 with B-frames, one audio track (ac3 or
    pcm_bluray), and PGS subtitles when *cues* are given (times relative to
    the clip). *start* moves the clip's own timeline, as authoring does."""
    import io

    import av
    import numpy as np

    from tests import pgs

    frames = int(seconds * fps)
    buf = io.BytesIO()
    sup = None
    if cues:
        path = (tmp or Path(".")) / f"cues_{abs(hash((seconds, start, tuple(cues))))}.sup"
        pgs.write_sup(path, [(a + start, b + start, text) for a, b, text in cues])
        sup = av.open(str(path))
    try:
        with av.open(buf, "w", format="mpegts",
                     container_options={"mpegts_m2ts_mode": "1"}) as out:
            v = out.add_stream("libx264", rate=fps)
            v.width, v.height, v.pix_fmt = 128, 96, "yuv420p"
            v.options = {"bf": "3", "g": str(fps)}
            a = out.add_stream(audio, rate=48000)
            a.layout = "stereo" if channels == 2 else "5.1(side)" if channels == 6 else "mono"
            fmt = "fltp" if audio == "ac3" else ("s32" if wide else "s16")
            a.format = fmt
            s = out.add_stream_from_template(sup.streams[0]) if sup else None
            base = int(start * fps)
            per = 1536 if audio == "ac3" else 240
            total = int(seconds * 48000)
            offset = int(start * 48000)
            sent = 0

            def audio_until(t: float) -> None:
                # interleaved as a disc is: audio up to the picture being
                # written, not all of it after the video (a demuxer's probe
                # would never reach it)
                nonlocal sent
                while sent < total and sent / 48000 <= t:
                    n = min(per, total - sent)
                    af = av.AudioFrame.from_ndarray(_tone(n, 48000, channels, fmt),
                                                    format=fmt, layout=a.layout.name)
                    af.sample_rate = 48000
                    af.pts = offset + sent
                    for p in a.encode(af):
                        out.mux(p)
                    sent += n

            for i in range(frames):
                img = np.zeros((96, 128, 3), dtype=np.uint8)
                img[:, (i * 4) % 128:((i * 4) % 128) + 16] = 255
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pts = base + i
                for p in v.encode(frame):
                    out.mux(p)
                audio_until((i + 1) / fps)
            for p in v.encode(None):
                out.mux(p)
            audio_until(seconds + 1)
            for p in a.encode(None):
                out.mux(p)
            if sup:
                for p in sup.demux(sup.streams[0]):
                    if p.size:
                        p.stream = s
                        out.mux(p)
    finally:
        if sup:
            sup.close()
    data = buf.getvalue()
    first, end, count, pids, keys = _probe(data, "mpegts", fps)
    return Clip(data, first, end, count, pids, keys)


def make_vob(seconds: float = 2.0, *, fps: int = 25, audio: str = "ac3",
             subtitles: Sequence[float] = (), start: float = 0.0) -> Clip:
    """A DVD VOB (the `dvd` muxer: 2048-byte packs with NAV packs), MPEG-2
    video, ac3 or pcm_dvd audio, and a VobSub packet at each of *subtitles*
    (seconds into the clip). Sizes are whole sectors, as cells need."""
    import io

    import av
    import numpy as np

    frames = int(seconds * fps)
    buf = io.BytesIO()
    with av.open(buf, "w", format="dvd") as out:
        v = out.add_stream("mpeg2video", rate=fps)
        v.width, v.height, v.pix_fmt = 352, 288, "yuv420p"
        a = out.add_stream(audio, rate=48000)
        a.layout = "stereo"
        fmt = "fltp" if audio == "ac3" else "s16"
        a.format = fmt
        s = out.add_mux_stream("dvd_subtitle") if subtitles else None
        if s is not None:
            from fractions import Fraction

            s.time_base = Fraction(1, 90000)
        base = int(start * fps)
        pending = sorted(subtitles)
        per = 1536 if audio == "ac3" else 240
        total = int(seconds * 48000)
        sent = 0

        def audio_until(t: float) -> None:
            nonlocal sent
            while sent < total and sent / 48000 <= t:
                n = min(per, total - sent)
                af = av.AudioFrame.from_ndarray(_tone(n, 48000, 2, fmt), format=fmt,
                                                layout="stereo")
                af.sample_rate = 48000
                af.pts = int(start * 48000) + sent
                for p in a.encode(af):
                    out.mux(p)
                sent += n

        for i in range(frames):
            img = np.zeros((288, 352, 3), dtype=np.uint8)
            img[:, (i * 8) % 352:((i * 8) % 352) + 32] = 200
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = base + i
            for p in v.encode(frame):
                out.mux(p)
            audio_until((i + 1) / fps)
            while pending and s is not None and pending[0] <= i / fps:
                at = int((pending.pop(0) + start) * 90000)
                pkt = av.Packet(bytes([0x00, 0x10] + [0] * 14))
                pkt.stream, pkt.pts, pkt.dts = s, at, at
                out.mux(pkt)
        for p in v.encode(None):
            out.mux(p)
        audio_until(seconds + 1)
        for p in a.encode(None):
            out.mux(p)
    data = buf.getvalue()
    data += b"\x00" * (-len(data) % SECTOR)
    first, end, count, pids, keys = _probe(data, "mpeg", fps)
    return Clip(data, first, end, count, pids, keys)
