"""Write one disc title as an MKV: every stream copied, the pieces stitched.

A title is a run of pieces of content — Blu-ray play items (whole m2ts
clips, or stretches of them), DVD cells (sector ranges of the VOB files).
Each piece has its own timestamps: a clip starts wherever its authoring
put it, a DVD cell after a discontinuity starts again from anywhere. So
every piece is opened as a container of its own and shifted onto one
output timeline, exactly the way mux.py shifts one file — once per piece
instead of once per film.

What is copied is decided by the disc's own stream table (Blu-ray STN, DVD
IFO), not by what the demuxer happens to find: that table is where the
languages are (neither m2ts nor VOB carries them for subtitles), and it is
the list of what the title *plays* — an m2ts also holds menus and
picture-in-picture streams that are not part of the film.

Measured facts this relies on (PyAV 18, see CLAUDE.md for the rest):

* Matroska refuses pcm_bluray / pcm_dvd, so a disc's LPCM is decoded and
  written as FLAC (or plain PCM) — losslessly either way. It refuses VC-1
  too, and PyAV hard-codes the compliance check; such titles never reach
  this module (bdmv marks them unsupported).
* A VobSub track needs the DVD's palette in its CodecPrivate, which is set
  through codec_context.extradata on a stream built from a template; the
  palette lives in the IFO, not in the VOB.
* Only the demuxer's empty end-of-stream packet may be dropped from a
  copied stream (mux.py: `not packet.size`). PGS "clear" display sets are
  small but real, and every subtitle's end time depends on them.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import av

from app.services.disc.binary import DiscError
from app.services.disc.fs import ConcatReader, sub_extents
from app.services.disc.model import Disc, Title

LogFn = Callable[[str], None]
ProgressFn = Callable[[float], None]   # 0..1 for this title

PROGRESS_EVERY = 200
BUFFER = 1 << 20          # PyAV read size for file-like inputs
PROBE = {"probesize": str(32 << 20), "analyzeduration": str(20_000_000)}
# second try, when a track's parameters were not in the first 32 MB: a
# matroska header cannot be written for an audio track with no sample rate
DEEP_PROBE = {"probesize": str(1 << 30), "analyzeduration": str(600_000_000)}
LPCM = ("pcm_bluray", "pcm_dvd")


@dataclass
class Piece:
    """One stretch of the title, and where it lands on the output timeline."""

    open: Callable[[], ConcatReader]
    length: int                          # bytes it covers
    fmt: str                             # "mpegts" | "mpeg"
    label: str
    # output seconds where this piece begins; None = right after the
    # previous piece's picture ends (DVD runs, whose true length only the
    # stream knows)
    start: Optional[float]
    # the piece's own timestamp that maps to `start`; None = its first
    # video frame
    origin: Optional[float]
    # the piece's own timestamps to keep, [a, b); None = all of it
    window: Optional[Tuple[float, float]] = None
    # bytes of synthetic packets ahead of the real data (subtitle_primer)
    primer: int = 0


@dataclass
class Want:
    """One output track, as the disc's stream table describes it."""

    kind: str                            # video | audio | subtitle
    key: int                             # BD PID / DVD stream id
    language: str = ""
    forced: bool = False
    commentary: bool = False
    extradata: Optional[bytes] = None    # VobSub: size + palette


@dataclass
class Route:
    out: object                          # output stream
    kind: str
    encoder: bool = False                # LPCM being converted
    fmt: str = ""                        # the encoder's sample format
    packets: int = 0
    next_pts: int = 0                    # encoder: first sample not yet written


@dataclass
class Result:
    path: Path
    bytes: int
    seconds: float
    tracks: List[str]
    chapters: int
    adjusted: int = 0
    dropped: int = 0
    notes: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ plans

def _bd_plan(disc: Disc, title: Title) -> Tuple[List[Piece], List[Want]]:
    from app.services.disc.bdmv import TICKS

    plan = title.plan
    pieces: List[Piece] = []
    at = 0.0
    for item, path in plan.items:
        if not path:
            raise DiscError(f"缺少片段文件 {item.clip}.m2ts")
        extents = disc.fs.extents(path)
        total = sum(e.length for e in extents)
        clip = plan.clips.get(item.clip)
        begin = 0
        window = None
        if clip is not None and clip.end > clip.start:
            partial = item.in_time > clip.start + TICKS or item.out_time < clip.end - TICKS
            if partial:
                window = (item.in_time / TICKS, item.out_time / TICKS)
                spn = clip.spn_at(item.in_time)
                if spn and item.in_time > clip.start + TICKS:
                    begin = min(spn * 192, total)
        span = sub_extents(extents, begin, total - begin)
        pieces.append(Piece(
            open=lambda span=span, name=path: ConcatReader(span, name=name),
            length=total - begin, fmt="mpegts", label=f"{item.clip}.m2ts",
            start=at, origin=item.in_time / TICKS, window=window))
        at += item.seconds
    wants = []
    for s in title.streams:
        if s.carried:
            wants.append(Want(s.kind, s.key, s.language, s.forced, s.commentary))
    return pieces, wants


def _ycrcb_to_rgb(value: int) -> str:
    """One IFO palette entry (0x00YYCrCb, studio range) as rrggbb."""
    y, cr, cb = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
    yy = (y - 16) * 255 / 219
    crr = (cr - 128) * 255 / 224
    cbb = (cb - 128) * 255 / 224
    rgb = (yy + 1.402 * crr, yy - 0.344136 * cbb - 0.714136 * crr, yy + 1.772 * cbb)
    return "".join(f"{max(0, min(255, round(c))):02x}" for c in rgb)


def vobsub_private(size: Tuple[int, int], palette: List[int]) -> bytes:
    """What a Matroska VobSub track carries as CodecPrivate: the .idx header."""
    colours = ", ".join(_ycrcb_to_rgb(v) for v in (list(palette) + [0] * 16)[:16])
    return f"size: {size[0]}x{size[1]}\npalette: {colours}\n".encode("ascii")


def _pack(payload: bytes, stream_id: int = 0xBD) -> bytes:
    """One 2048-byte MPEG-2 program-stream pack holding one PES packet."""
    head = b"\x00\x00\x01\xba\x44\x00\x04\x00\x04\x01\x01\x89\xc3\xf8"
    pes = b"\x00\x00\x01" + bytes([stream_id]) + (3 + len(payload)).to_bytes(2, "big") \
        + b"\x81\x00\x00" + payload
    pad = 2048 - len(head) - len(pes) - 6
    return head + pes + b"\x00\x00\x01\xbe" + pad.to_bytes(2, "big") + b"\xff" * pad


def subtitle_primer(sids: List[int]) -> bytes:
    """Packs announcing each VobSub stream before the real data starts.

    A VOB declares nothing up front: the MPEG demuxer creates a stream when
    its first packet goes by, and PyAV only ever hands out packets of the
    streams that existed when the file was opened — a stream first met
    later has every packet silently dropped (InputContainer.demux compares
    the index against the stream list it built at open). Measured on a
    real DVD: both of a film's subtitle tracks came out empty, because
    their first line is further in than the demuxer's probe reads. So each
    piece opens with one empty subpicture packet per stream the IFO names,
    which the probe sees; remux_title drops them again by position.
    """
    # an empty but well-formed SPU: size 9, control sequence at 4 that ends
    empty_spu = bytes([0x00, 0x09, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0xFF])
    return b"".join(_pack(bytes([sid]) + empty_spu) for sid in sids)


def _dvd_plan(disc: Disc, title: Title) -> Tuple[List[Piece], List[Want]]:
    from app.services.disc.dvd import SECTOR, vob_extents

    plan = title.plan
    if not plan.vobs:
        raise DiscError(f"缺少 VTS_{plan.vts:02d}_1.VOB 等视频文件")
    extents = vob_extents(disc.fs, plan)
    total = sum(e.length for e in extents)
    # FFmpeg's `trim`: padding cells under a second at the start are black
    cells = list(plan.cells)
    while len(cells) > 1 and cells[0].seconds < 1.0:
        cells.pop(0)
    runs: List[List] = []
    for cell in cells:
        if runs and cell.first == runs[-1][-1].last + 1 \
                and cell.vob_id == runs[-1][-1].vob_id and not cell.stc_discontinuity:
            runs[-1].append(cell)
        else:
            runs.append([cell])
    primer = subtitle_primer([sid for sid, _attr in plan.subp])
    pieces = []
    for n, run in enumerate(runs):
        start = run[0].first * SECTOR
        end = min((run[-1].last + 1) * SECTOR, total)
        if start >= total:
            raise DiscError(f"单元指向第 {run[0].first} 扇区，但 VOB 文件只有 {total // SECTOR} 扇区"
                            "（VOB 文件不完整？）")
        span = sub_extents(extents, start, end - start)
        pieces.append(Piece(
            open=lambda span=span: ConcatReader(span, name=f"VTS_{plan.vts:02d}", prefix=primer),
            length=end - start, fmt="mpeg", primer=len(primer),
            label=f"VTS_{plan.vts:02d} 扇区 {run[0].first}–{run[-1].last}",
            start=0.0 if n == 0 else None, origin=None))
    private = vobsub_private(plan.size, plan.palette)
    wants = [Want("video", 0x1E0)]
    for sid, attr in plan.audio:
        wants.append(Want("audio", sid, attr.language, commentary=attr.code_ext in (3, 4)))
    for sid, attr in plan.subp:
        wants.append(Want("subtitle", sid, attr.language,
                          forced=9 in (attr.code_ext, attr.lang_ext), extradata=private))
    return pieces, wants


# ----------------------------------------------------------------- engine

def packets(container):
    """container.demux(), minus a PyAV bug that only a VOB triggers.

    PyAV sizes its per-stream table when demux() starts. A stream the MPEG
    demuxer creates later (a VOB declares nothing up front) has its packets
    dropped — which subtitle_primer prevents for the streams the IFO names
    — and the end-of-stream flush then indexes that table with the *new*
    stream count, reading memory it never set: sometimes nothing happens,
    sometimes IndexError (av/container/input.py, "TODO: find better way").
    By then every real packet has been delivered, so it ends the loop.
    """
    it = container.demux()
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except IndexError:
            return


def _open(piece: Piece, deep: bool = False):
    reader = piece.open()
    try:
        container = av.open(reader, format=piece.fmt, options=DEEP_PROBE if deep else PROBE,
                            buffer_size=BUFFER)
    except Exception:
        reader.close()
        raise
    return reader, container


def _vobsub_template():
    """A dvd_subtitle stream to copy codec parameters from, for a VobSub
    track the first piece does not reach — most films have minutes of
    dialogue-free opening, and a demuxer only knows the streams it saw."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="vob") as o:
        v = o.add_stream("mpeg2video", rate=25)
        v.width, v.height, v.pix_fmt = 64, 64, "yuv420p"
        s = o.add_mux_stream("dvd_subtitle")
        s.time_base = Fraction(1, 90000)
        frame = av.VideoFrame(64, 64, "yuv420p")
        for i in range(3):
            frame.pts = i
            for p in v.encode(frame):
                o.mux(p)
        pkt = av.Packet(bytes([0x00, 0x10] + [0] * 14))
        pkt.stream, pkt.pts, pkt.dts = s, 3600, 3600
        o.mux(pkt)
        for p in v.encode(None):
            o.mux(p)
    buf.seek(0)
    container = av.open(buf, format="mpeg")
    return container, next(st for st in container.streams if st.type == "subtitle")


def _lpcm_codec(stream, choice: str) -> str:
    """Which encoder takes this disc LPCM track: FLAC when asked and the
    channel layout suits it, PCM otherwise. Tried in a scratch container so
    a refusal never leaves a dead track in the real output."""
    cc = stream.codec_context
    wide = cc.format is not None and cc.format.name.startswith("s32")
    pcm = "pcm_s24le" if wide else "pcm_s16le"
    if choice != "flac":
        return pcm
    try:
        with av.open(io.BytesIO(), "w", format="matroska") as scratch:
            enc = scratch.add_stream("flac", rate=cc.sample_rate)
            enc.layout = cc.layout.name
            enc.format = "s32" if wide else "s16"
            enc.codec_context.open()
        return "flac"
    except Exception:  # noqa: BLE001 — FLAC cannot take it; PCM can
        return pcm


def _lpcm_encoder(out, stream, choice: str, log: LogFn):
    """The output stream a disc LPCM track is converted into."""
    cc = stream.codec_context
    name = _lpcm_codec(stream, choice)
    if choice == "flac" and name != "flac":
        log(f"⚠ 这条 LPCM 音轨（{cc.layout.name}）FLAC 装不下，改为无损 PCM")
    enc = out.add_stream(name, rate=cc.sample_rate)
    enc.layout = cc.layout.name
    enc.format = "s32" if cc.format.name.startswith("s32") else "s16"
    return enc, enc.format.name


def _disposition(route_kind: str, want: Want, first: bool):
    d = av.stream.Disposition
    value = d(0)
    if route_kind in ("video", "audio") and first:
        value |= d.default
    if want.forced:
        value |= d.forced
    if want.commentary:
        value |= d.comment
    return value


def remux_title(disc: Disc, title: Title, out_path: Path, *, lpcm: str = "flac",
                log: Optional[LogFn] = None, progress: Optional[ProgressFn] = None,
                should_cancel: Optional[Callable[[], bool]] = None,
                tags: Optional[Dict[str, str]] = None) -> Result:
    log = log or (lambda _m: None)
    should_cancel = should_cancel or (lambda: False)
    pieces, wants = (_bd_plan if disc.kind == "bd" else _dvd_plan)(disc, title)
    total = sum(p.length for p in pieces) or 1
    out_path = Path(out_path)
    part = out_path.with_name(out_path.name + ".part")
    if part.exists():
        log(f"⚠ 发现上次未写完的残留文件（{part.stat().st_size / 1e9:.1f} GB），已删除：{part.name}")
        part.unlink(missing_ok=True)
    started = time.monotonic()
    tracks: List[str] = []
    notes: List[str] = []
    adjusted = dropped = 0
    last_dts: Dict[int, float] = {}
    routes: Dict[Tuple[int, str], Route] = {}

    try:
        with av.open(str(part), mode="w", format="matroska") as out:
            # global tags, written with the header: the pipeline's way of
            # recognising this file as the remux of this title later
            # (encode.REMUX_TAG)
            out.metadata.update(tags or {})
            # Tracks are built from the first piece, and from later ones for
            # whatever it does not hold: a play item's clip need not carry
            # every stream of the title (a logo clip ahead of the film has
            # the picture alone), and a track only ever met in a later
            # piece would otherwise have no output to go to.
            opened = []
            try:
                for n, piece in enumerate(pieces):
                    if n and not _missing(opened, wants, disc.kind):
                        break
                    reader, container = _open(piece)
                    if _unprobed(container, wants):
                        container.close()
                        reader.close()
                        log(f"{piece.label}：部分音轨的参数在片段开头读不出来，扩大探测范围重读一次…")
                        reader, container = _open(piece, deep=True)
                    opened.append((reader, container))
                    if n:
                        log(f"第一个片段里没有的轨道到 {piece.label} 里找")
                routes, tracks = _build_outputs(out, [c for _r, c in opened], wants,
                                                disc.kind, lpcm, log, notes)
                lead = _lead(opened[0][1], pieces[0])
            finally:
                for reader, container in opened:
                    container.close()
                    reader.close()
            if not any(r.kind == "video" for r in routes.values()):
                raise DiscError("第一个片段里找不到视频流")
            chapters = [c + lead for c in title.chapters if c < title.duration]
            if chapters:
                out.set_chapters([{
                    "id": i + 1,
                    "start": round(c * 1000),
                    "end": round((chapters[i + 1] if i + 1 < len(chapters)
                                  else title.duration + lead) * 1000),
                    "time_base": Fraction(1, 1000),
                    "metadata": {"title": f"第 {i + 1:02d} 章"},
                } for i, c in enumerate(chapters)])

            done = 0
            cursor = lead
            seen = 0
            for piece in pieces:
                reader, container = _open(piece)
                try:
                    start = cursor if piece.start is None else piece.start + lead
                    origin = piece.origin
                    if origin is None:
                        origin = _first_video_pts(container, piece)
                        container.close()
                        reader.close()
                        reader, container = _open(piece)
                    shift_s = start - origin
                    video_end = start
                    window = piece.window
                    # DVD pieces: the open GOP's leading B-frames (see
                    # _first_video_pts). A Blu-ray clip is joined seamlessly
                    # to the one before it, so its leading frames do have
                    # their reference and are kept.
                    trim_leading = piece.fmt == "mpeg"
                    entry_pts = None
                    for packet in packets(container):
                        if should_cancel():
                            raise InterruptedError
                        stream = packet.stream
                        route = routes.get(_key(stream, disc.kind))
                        seen += 1
                        if progress and seen % PROGRESS_EVERY == 0:
                            progress(min((done + reader.consumed) / total, 1.0))
                        if route is None:
                            continue
                        if piece.primer and packet.pos is not None and 0 <= packet.pos < piece.primer:
                            continue                 # one of ours, not the disc's
                        tb = stream.time_base
                        if window is not None and packet.pts is not None:
                            at = float(packet.pts * tb)
                            if at < window[0] - 0.001 or at >= window[1]:
                                if route.kind == "video" and at > window[1] + 2.0:
                                    break
                                dropped += 1
                                continue
                        if route.encoder:
                            if packet.pts is None:
                                continue
                            for frame in packet.decode():
                                if frame.pts is None:
                                    continue
                                rate = route.out.rate
                                at = round((float(frame.pts * frame.time_base) + shift_s) * rate)
                                if at < 0 or at + frame.samples <= route.next_pts:
                                    dropped += 1          # wholly before what is written
                                    continue
                                # a sliver of overlap at a join: butt it up
                                frame.pts = max(at, route.next_pts)
                                frame.time_base = Fraction(1, rate)
                                route.next_pts = frame.pts + frame.samples
                                for made in route.out.encode(frame):
                                    out.mux(made)
                                    route.packets += 1
                            continue
                        if not packet.size:
                            continue
                        if trim_leading and route.kind == "video":
                            if entry_pts is None:
                                if not packet.is_keyframe or packet.pts is None:
                                    dropped += 1
                                    continue
                                entry_pts = packet.pts
                            elif packet.pts is not None and packet.pts < entry_pts:
                                dropped += 1
                                continue
                        shift = round(shift_s / float(tb))
                        if packet.pts is not None:
                            packet.pts += shift
                        if packet.dts is not None:
                            packet.dts += shift
                        key = id(route.out)
                        ref = packet.dts if packet.dts is not None else packet.pts
                        if ref is not None:
                            t = float(ref * tb)
                            prev = last_dts.get(key)
                            # Equal is fine (every segment of a PGS display
                            # set shares one timestamp; matroska accepts it).
                            # Earlier is not: overlap at a seamless join.
                            if t < 0 or (prev is not None and t < prev - 1e-6):
                                if route.kind != "video":
                                    dropped += 1
                                    continue
                                adjusted += 1
                                delta = round(((prev or 0.0) - t) / float(tb)) + 1
                                if packet.dts is not None:
                                    packet.dts += delta
                                if packet.pts is not None:
                                    packet.pts = max(packet.pts, packet.dts
                                                     if packet.dts is not None else packet.pts)
                                t = float((packet.dts if packet.dts is not None
                                           else packet.pts) * tb)
                            last_dts[key] = t
                        if route.kind == "video" and packet.pts is not None:
                            dur = float((packet.duration or 0) * tb)
                            video_end = max(video_end, float(packet.pts * tb) + dur)
                        packet.stream = route.out
                        out.mux(packet)
                        route.packets += 1
                    done += piece.length
                    cursor = video_end
                finally:
                    container.close()
                    reader.close()
            for route in routes.values():
                if route.encoder:
                    for made in route.out.encode(None):
                        out.mux(made)
                        route.packets += 1
            # The disc says this title has sound; a file without any is not
            # a remux of it, however cleanly it was written. A ⚠ in the log
            # under a job marked done is how a silent film once got through.
            if any(w.kind == "audio" for w in wants) and not any(
                    r.kind == "audio" and r.packets for r in routes.values()):
                raise DiscError("光盘流表里有音轨，但一条音轨都没有写进去"
                                "（这部片的片段里找不到它们），不交出一个没有声音的文件")
        os.replace(part, out_path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    if progress:
        progress(1.0)
    empty = [r for r in routes.values() if r.packets == 0 and r.kind != "subtitle"]
    for r in empty:
        notes.append(f"⚠ 有一条{'视频' if r.kind == 'video' else '音轨'}一个数据包都没有写进去")
    size = out_path.stat().st_size
    written = _duration(out_path)
    if written and abs(written - title.duration) > max(2.0, 0.01 * title.duration):
        notes.append(f"⚠ 写出的时长 {written:.1f}s，与光盘标称的 {title.duration:.1f}s 不一致")
    return Result(out_path, size, time.monotonic() - started, tracks,
                  len(title.chapters), adjusted, dropped, notes)


def _audio_ready(stream) -> bool:
    cc = stream.codec_context
    return cc is not None and bool(cc.sample_rate) and bool(getattr(cc.layout, "nb_channels", 0))


def _unprobed(container, wants: List[Want]) -> bool:
    keys = {w.key for w in wants if w.kind == "audio"}
    return any(s.type == "audio" and s.id in keys and not _audio_ready(s)
               for s in container.streams)


def _key(stream, kind: str) -> Tuple[int, str]:
    name = stream.codec_context.name if stream.codec_context is not None else ""
    return (stream.id, name)


def _first_video_pts(container, piece: Piece) -> float:
    """Where a DVD piece's picture starts: its first I-frame.

    Not the earliest timestamp: a cell usually opens with an *open* GOP,
    whose leading B-frames display before that I-frame but predict from the
    GOP before it — which, at the start of a title or after a discontinuity,
    is not there. Those frames are dropped (see remux_title), so the picture
    starts at the I-frame. FFmpeg's dvdvideo demuxer drops them too; that
    is where the only packet-count difference against it came from."""
    for packet in packets(container):
        if packet.stream.type == "video" and packet.pts is not None and packet.is_keyframe:
            return float(packet.pts * packet.stream.time_base)
    raise DiscError(f"{piece.label} 里找不到视频")


def _lead(container, piece: Piece) -> float:
    """How far the output timeline must start after zero so that no packet
    lands before it: with B-frames the first decode timestamp is earlier
    than the first picture, and Matroska cannot store negative times —
    it drops those packets, the opening keyframe first (mux.py measured
    that). One constant for the whole title, so nothing drifts."""
    origin = piece.origin
    lowest = None
    first_video = None
    for n, packet in enumerate(packets(container)):
        if n > 600:
            break
        tb = packet.stream.time_base
        # a play item that starts mid-clip: what comes before its in-point
        # is dropped, so it must not push the timeline back either
        if piece.window is not None and packet.pts is not None \
                and float(packet.pts * tb) < piece.window[0] - 0.001:
            continue
        if packet.dts is not None and packet.size:
            t = float(packet.dts * tb)
            lowest = t if lowest is None else min(lowest, t)
        if first_video is None and packet.stream.type == "video" and packet.pts is not None:
            first_video = float(packet.pts * tb)
    if origin is None:
        origin = first_video if first_video is not None else (lowest or 0.0)
    if lowest is None:
        return 0.0
    return round(max(0.0, origin - lowest), 6)


def _streams_by_id(containers, kind_of_want: Dict[int, str]) -> Dict[int, list]:
    """Each stream id's streams, from the first container that has it as
    the kind the disc says it is (a TrueHD PID holds two streams: both
    come from the same container)."""
    by_id: Dict[int, list] = {}
    for container in containers:
        found: Dict[int, list] = {}
        for s in container.streams:
            if s.id not in by_id and kind_of_want.get(s.id, s.type) == s.type:
                found.setdefault(s.id, []).append(s)
        by_id.update(found)
    return by_id


def _missing(opened, wants: List[Want], kind: str) -> bool:
    """Does some wanted track appear in none of the pieces opened so far?
    (A DVD's VobSub is made from a template when absent, so it never is.)"""
    if not opened:
        return True
    by_id = _streams_by_id([c for _r, c in opened], {w.key: w.kind for w in wants})
    return any(w.key not in by_id for w in wants
               if not (w.kind == "subtitle" and kind == "dvd"))


def _build_outputs(out, containers, wants: List[Want], kind: str, lpcm: str,
                   log: LogFn, notes: List[str]):
    routes: Dict[Tuple[int, str], Route] = {}
    tracks: List[str] = []
    firsts = {"video": True, "audio": True, "subtitle": True}
    template_box = None
    by_id = _streams_by_id(containers, {w.key: w.kind for w in wants})
    try:
        for want in wants:
            matches = [s for s in by_id.get(want.key, []) if s.type == want.kind]
            if not matches and want.kind == "subtitle" and kind == "dvd":
                if template_box is None:
                    template_box = _vobsub_template()
                matches = [("template", template_box[1])]
            if not matches:
                notes.append(f"⚠ 光盘流表里的 {want.kind} 0x{want.key:x} 在片段里找不到，已跳过")
                continue
            # TrueHD arrives as two streams under one PID: the lossless
            # track and its AC-3 core. Both are kept, TrueHD first.
            for m in sorted(matches, key=lambda x: 0 if not isinstance(x, tuple) and
                            x.codec_context.name == "truehd" else 1):
                synthetic = isinstance(m, tuple)
                stream = m[1] if synthetic else m
                name = stream.codec_context.name
                if want.kind == "audio" and not _audio_ready(stream):
                    # one track the demuxer cannot describe must not cost
                    # the film: matroska would refuse the whole header
                    notes.append(f"⚠ 音轨 0x{want.key:x}（{name}）读不出采样率/声道，已跳过")
                    continue
                is_first = firsts[want.kind]
                if name in LPCM:
                    o, fmt = _lpcm_encoder(out, stream, lpcm, log)
                    route = Route(o, want.kind, encoder=True, fmt=fmt)
                    label = f"{'FLAC' if o.codec_context.name == 'flac' else 'PCM'}（由 LPCM 无损转换）"
                else:
                    o = out.add_stream_from_template(stream)
                    route = Route(o, want.kind)
                    label = name
                o.metadata["language"] = want.language or "und"
                if want.commentary:
                    o.metadata["title"] = "评论音轨"
                elif want.kind == "audio" and name == "ac3" and len(matches) > 1:
                    o.metadata["title"] = "AC-3 兼容音轨"
                    is_first = False
                elif want.forced:
                    o.metadata["title"] = "强制字幕"
                if want.extradata is not None and o.codec_context is not None:
                    o.codec_context.extradata = want.extradata
                o.disposition = _disposition(want.kind, want, is_first)
                firsts[want.kind] = False
                key = (want.key, "dvdsub") if synthetic else _key(stream, kind)
                routes[key] = route
                tracks.append(f"{want.kind}:{label}:{want.language or 'und'}")
    finally:
        if template_box is not None:
            template_box[0].close()
    return routes, tracks


def _duration(path: Path) -> Optional[float]:
    try:
        with av.open(str(path)) as c:
            return float(c.duration / av.time_base) if c.duration else None
    except Exception:  # noqa: BLE001 — a check, not a requirement
        return None
