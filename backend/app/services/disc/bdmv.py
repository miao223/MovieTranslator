"""Blu-ray (BDMV) metadata: playlists, clip info, index and movie objects.

Only the small files beside the streams are read here — never the streams
— so a whole disc is analysed in milliseconds, and a metadata-only copy of
it (BDMV without STREAM/) analyses exactly like the real thing. Field
layouts follow libbluray (mpls_parse.c, clpi_parse.c, index_parse.c,
mobj_parse.c); comments name the libbluray field where it helps.

What a playlist *is* for the rest of the program: an ordered list of play
items, each "clip 00012 from in_time to out_time" (45 kHz ticks). Two
playlists showing the same film are two lists over the same clips, which is
why analysis compares clips and times, never playlist numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.services import audio
from app.services.disc.binary import DiscError, Reader
from app.services.disc.fs import DiscFS, Located
from app.services.disc.model import Disc, Segment, Stream, Title

TICKS = 45000  # MPLS/CLPI time unit, per second

# stream_coding_type (MPLS STN / CLPI ProgramInfo)
VIDEO_CODECS = {
    0x01: "MPEG-1", 0x02: "MPEG-2", 0x1B: "H.264", 0x20: "MVC",
    0x24: "HEVC", 0xEA: "VC-1",
}
AUDIO_CODECS = {
    0x03: "MPEG 音频", 0x04: "MPEG 音频", 0x80: "LPCM", 0x81: "AC-3",
    0x82: "DTS", 0x83: "TrueHD", 0x84: "E-AC-3", 0x85: "DTS-HD HRA",
    0x86: "DTS-HD MA", 0xA1: "E-AC-3", 0xA2: "DTS-HD",
}
PG, IG, TEXTST = 0x90, 0x91, 0x92

VIDEO_FORMATS = {1: "480i", 2: "576i", 3: "480p", 4: "1080i", 5: "720p",
                 6: "1080p", 7: "576p", 8: "2160p"}
FRAME_RATES = {1: "23.976", 2: "24", 3: "25", 4: "29.97", 6: "50", 7: "59.94"}
AUDIO_LAYOUTS = {1: "单声道", 3: "立体声", 6: "多声道", 12: "立体声+多声道"}

# HDMV navigation command fields (hdmv_insn_t): the "play playlist" family
_GRP_BRANCH, _SUB_PLAY = 0, 2
_PLAY_PL, _PLAY_PL_PI, _PLAY_PL_PM = 0, 1, 2


@dataclass
class StreamEntry:
    group: str          # "video" | "audio" | "pg" | "ig" | "textst"
    stream_type: int    # 1 = in this play item's clip; 2-4 = a sub path's clip
    pid: int
    coding_type: int
    fmt: int = 0        # video_format / audio_presentation_type
    rate: int = 0       # frame_rate / sampling_frequency
    language: str = ""


@dataclass
class PlayItem:
    clip: str                                   # "00012"
    in_time: int
    out_time: int
    connection: int = 1
    angles: List[str] = field(default_factory=list)   # angles 2.. (clip ids)
    streams: List[StreamEntry] = field(default_factory=list)
    secondary: int = 0   # secondary audio + video + picture-in-picture PG
    dolby_vision: int = 0

    @property
    def seconds(self) -> float:
        return max(self.out_time - self.in_time, 0) / TICKS


@dataclass
class Mark:
    kind: int      # 1 = entry (chapter), 2 = link point
    item: int
    time: int


@dataclass
class Playlist:
    name: str                      # "00800"
    items: List[PlayItem]
    marks: List[Mark]
    playback_type: int = 1
    subpaths: int = 0


@dataclass
class ClipInfo:
    name: str
    application_type: int = 1
    ts_rate: int = 0
    source_packets: int = 0
    start: int = 0                  # presentation start, 45 kHz
    end: int = 0
    languages: Dict[int, str] = field(default_factory=dict)   # pid -> lang
    codings: Dict[int, int] = field(default_factory=dict)     # pid -> coding
    # pid -> [(pts 45 kHz, spn)], ascending — for seeking into a clip
    ep: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)

    def spn_at(self, time: int) -> Optional[int]:
        """Source packet number of the last entry point at or before *time*."""
        for entries in self.ep.values():
            if not entries:
                continue
            best = None
            for pts, spn in entries:
                if pts > time:
                    break
                best = spn
            return best if best is not None else entries[0][1]
        return None


@dataclass
class BdPlan:
    """What remux.py needs to write one playlist."""

    playlist: str
    items: List[Tuple[PlayItem, str]]          # (item, m2ts path in the fs)
    clips: Dict[str, ClipInfo]
    marks: List[Mark]


# ----------------------------------------------------------------- parsing

def _stream(r: Reader, group: str) -> StreamEntry:
    n = r.u8()
    end = r.pos + n
    stream_type = r.u8()
    pid = 0
    if stream_type == 1:
        pid = r.u16()
    elif stream_type in (2, 4):
        r.u8()
        r.u8()
        pid = r.u16()
    elif stream_type == 3:
        r.u8()
        pid = r.u16()
    r.seek(end)
    n = r.u8()
    end = r.pos + n
    coding = r.u8()
    fmt = rate = 0
    lang = ""
    if coding in VIDEO_CODECS:
        b = r.u8()
        fmt, rate = b >> 4, b & 0x0F
    elif coding in AUDIO_CODECS:
        b = r.u8()
        fmt, rate = b >> 4, b & 0x0F
        lang = r.text(3)
    elif coding in (PG, IG):
        lang = r.text(3)
    elif coding == TEXTST:
        r.u8()
        lang = r.text(3)
    r.seek(end)
    return StreamEntry(group, stream_type, pid, coding, fmt, rate,
                       lang.strip("\x00 ").lower())


def _stn(r: Reader, item: PlayItem) -> None:
    n = r.u16()
    end = r.pos + n
    r.skip(2)
    n_video, n_audio, n_pg, n_ig = r.u8(), r.u8(), r.u8(), r.u8()
    n_sec_audio, n_sec_video, n_pip_pg, n_dv = r.u8(), r.u8(), r.u8(), r.u8()
    r.skip(4)
    streams = [_stream(r, "video") for _ in range(n_video)]
    streams += [_stream(r, "audio") for _ in range(n_audio)]
    # text subtitles share the PG slots; the picture-in-picture ones come
    # after the primary ones and are not subtitles of this picture
    pg = [_stream(r, "pg") for _ in range(n_pg + n_pip_pg)]
    for s in pg:
        if s.coding_type == TEXTST:
            s.group = "textst"
    streams += pg[:n_pg]
    streams += [_stream(r, "ig") for _ in range(n_ig)]
    item.streams = streams
    item.secondary = n_sec_audio + n_sec_video + n_pip_pg
    item.dolby_vision = n_dv
    r.seek(end)


def _play_item(r: Reader) -> PlayItem:
    n = r.u16()
    end = r.pos + n
    clip = r.text(5)
    r.skip(4)                              # "M2TS"
    flags = r.u16()
    multi_angle = (flags >> 4) & 1
    connection = flags & 0x0F
    r.u8()                                 # stc_id
    item = PlayItem(clip, r.u32(), r.u32(), connection)
    r.skip(8)                              # UO mask
    r.u8()                                 # random access flag
    r.u8()                                 # still mode
    r.u16()                                # still time
    if multi_angle:
        count = max(r.u8(), 1)
        r.u8()
        for _ in range(count - 1):
            item.angles.append(r.text(5))
            r.skip(4)
            r.u8()
    _stn(r, item)
    r.seek(end)
    return item


def parse_mpls(data: bytes, name: str) -> Playlist:
    label = f"播放列表 {name}.mpls"
    r = Reader(data, label)
    if r.text(4) != "MPLS":
        raise DiscError(f"{label} 不是播放列表文件")
    r.skip(4)                              # version
    list_start, mark_start = r.u32(), r.u32()
    r.seek(0x28)
    r.u32()
    r.u8()
    playback_type = r.u8()

    p = r.at(list_start)
    p.u32()
    p.skip(2)
    n_items, n_subpaths = p.u16(), p.u16()
    items = [_play_item(p) for _ in range(n_items)]

    marks: List[Mark] = []
    if mark_start:
        m = r.at(mark_start)
        m.u32()
        for _ in range(m.u16()):
            m.u8()
            kind, item, time = m.u8(), m.u16(), m.u32()
            m.u16()
            m.u32()
            marks.append(Mark(kind, item, time))
    return Playlist(name, items, marks, playback_type, n_subpaths)


def parse_clpi(data: bytes, name: str) -> ClipInfo:
    label = f"片段信息 {name}.clpi"
    r = Reader(data, label)
    if r.text(4) != "HDMV":
        raise DiscError(f"{label} 不是片段信息文件")
    r.skip(4)
    seq_start, prog_start, cpi_start = r.u32(), r.u32(), r.u32()
    info = ClipInfo(name)
    r.seek(0x28)
    r.u32()
    r.skip(2)
    r.u8()                                 # clip_stream_type
    info.application_type = r.u8()
    r.u32()                                # reserved + is_ATC_delta
    info.ts_rate, info.source_packets = r.u32(), r.u32()

    if seq_start:
        s = r.at(seq_start)
        s.skip(5)
        starts, ends = [], []
        for _ in range(s.u8()):
            s.u32()                        # SPN_ATC_start
            n_stc = s.u8()
            s.u8()
            for _ in range(n_stc):
                s.u16()
                s.u32()
                starts.append(s.u32())
                ends.append(s.u32())
        if starts:
            info.start, info.end = min(starts), max(ends)

    if prog_start:
        p = r.at(prog_start)
        p.skip(5)
        for _ in range(p.u8()):
            p.u32()
            p.u16()
            n_streams = p.u8()
            p.u8()
            for _ in range(n_streams):
                pid = p.u16()
                n = p.u8()
                end = p.pos + n
                coding = p.u8()
                info.codings[pid] = coding
                if coding in AUDIO_CODECS:
                    p.u8()
                    info.languages[pid] = p.text(3).strip("\x00 ").lower()
                elif coding in (PG, IG):
                    info.languages[pid] = p.text(3).strip("\x00 ").lower()
                elif coding == TEXTST:
                    p.u8()
                    info.languages[pid] = p.text(3).strip("\x00 ").lower()
                p.seek(end)

    if cpi_start:
        c = r.at(cpi_start)
        if c.u32():
            c.u16()                        # reserved + CPI_type
            base = c.pos
            c.u8()
            heads = []
            for _ in range(c.u8()):
                pid = c.u16()
                # reserved 10 | EP_stream_type 4 | coarse 16 | fine 18
                bits = int.from_bytes(c.raw(6), "big")
                n_coarse = (bits >> 18) & 0xFFFF
                n_fine = bits & 0x3FFFF
                heads.append((pid, n_coarse, n_fine, base + c.u32()))
            for pid, n_coarse, n_fine, at in heads:
                e = r.at(at)
                fine_start = e.u32()
                coarse = []
                for _ in range(n_coarse):
                    w = e.u32()
                    coarse.append((w >> 14, w & 0x3FFF, e.u32()))
                e.seek(at + fine_start)
                fine = []
                for _ in range(n_fine):
                    w = e.u32()
                    fine.append(((w >> 17) & 0x7FF, w & 0x1FFFF))
                points = []
                for i, (ref, c_pts, c_spn) in enumerate(coarse):
                    stop = coarse[i + 1][0] if i + 1 < len(coarse) else len(fine)
                    for f_pts, f_spn in fine[ref:stop]:
                        pts = ((c_pts & ~1) << 18) + (f_pts << 8)
                        spn = (c_spn & ~0x1FFFF) + f_spn
                        points.append((pts, spn))
                info.ep[pid] = points
    return info


def parse_index(data: bytes) -> Tuple[List[Tuple[str, int]], bool]:
    """The disc's titles as (kind, movie object id), and whether BD-J is used."""
    r = Reader(data, "index.bdmv")
    if r.text(4) != "INDX":
        raise DiscError("index.bdmv 不是索引文件")
    r.skip(4)
    start = r.u32()
    r = r.at(start)
    r.u32()
    bdj = False
    for _ in range(2):                     # first play, top menu
        kind = r.u32() >> 30
        bdj |= kind == 2
        r.skip(8)
    titles = []
    for _ in range(r.u16()):
        kind = r.u32() >> 30
        if kind == 1:
            r.u16()
            titles.append(("hdmv", r.u16()))
            r.u32()
        else:
            bdj |= kind == 2
            titles.append(("bdj", -1))
            r.skip(8)
    return titles, bdj


def parse_movie_objects(data: bytes) -> List[Set[int]]:
    """Per movie object, the playlists it plays by immediate number."""
    r = Reader(data, "MovieObject.bdmv")
    if r.text(4) != "MOBJ":
        raise DiscError("MovieObject.bdmv 不是导航文件")
    r.seek(0x28)
    r.u32()
    r.u32()
    objects = []
    for _ in range(r.u16()):
        r.u16()
        played: Set[int] = set()
        for _ in range(r.u16()):
            b0, b1 = r.u8(), r.u8()
            r.u16()
            dst = r.u32()
            r.u32()
            grp, sub, imm1, opt = (b0 >> 3) & 3, b0 & 7, b1 >> 7, b1 & 0x0F
            if (grp, sub) == (_GRP_BRANCH, _SUB_PLAY) and imm1 and opt in (
                    _PLAY_PL, _PLAY_PL_PI, _PLAY_PL_PM):
                played.add(dst)
        objects.append(played)
    return objects


# ------------------------------------------------------------- the disc

def _stream_of(entry: StreamEntry, clip: Optional[ClipInfo]) -> Stream:
    lang = entry.language or (clip.languages.get(entry.pid, "") if clip else "")
    lang = audio.canon_language(lang) if lang else ""
    if entry.group == "video":
        codec = VIDEO_CODECS.get(entry.coding_type, f"0x{entry.coding_type:02x}")
        detail = " ".join(x for x in (VIDEO_FORMATS.get(entry.fmt, ""),
                                      FRAME_RATES.get(entry.rate, "")) if x)
        s = Stream("video", codec, entry.pid, detail=detail)
        if entry.coding_type == 0x20:
            s.carried, s.note = False, "3D 右眼视图（MVC），MKV 只保留 2D"
        return s
    if entry.group == "audio":
        codec = AUDIO_CODECS.get(entry.coding_type, f"0x{entry.coding_type:02x}")
        s = Stream("audio", codec, entry.pid, language=lang,
                   detail=AUDIO_LAYOUTS.get(entry.fmt, ""))
        if entry.coding_type == 0x80:
            s.note = "LPCM 会无损转成 FLAC（MKV 装不下蓝光的 LPCM 格式）"
        elif entry.coding_type == 0x83:
            s.note = "TrueHD 自带的 AC-3 兼容内核会作为另一条音轨一起保留"
        if entry.stream_type != 1:
            s.carried, s.note = False, "这条音轨在另一个片段文件里（子路径），暂不支持"
        return s
    if entry.group == "pg":
        s = Stream("subtitle", "PGS", entry.pid, language=lang)
        if entry.stream_type != 1:
            s.carried, s.note = False, "这条字幕在另一个片段文件里（子路径），暂不支持"
        return s
    if entry.group == "textst":
        return Stream("subtitle", "TextST", entry.pid, language=lang,
                      carried=False, note="蓝光文字字幕（TextST）MKV 里没有对应格式，暂不支持")
    return Stream("subtitle", "IG", entry.pid, language=lang, carried=False,
                  note="互动菜单（IG），不是字幕")


def _reachable(fs: DiscFS) -> Tuple[Set[int], List[str]]:
    notes: List[str] = []
    played: Set[int] = set()
    try:
        _titles, bdj = parse_index(fs.read("BDMV/index.bdmv"))
    except DiscError as exc:
        return played, [f"读不出 index.bdmv（{exc}），无法判断哪些播放列表在菜单里"]
    try:
        objects = parse_movie_objects(fs.read("BDMV/MovieObject.bdmv"))
    except DiscError:
        objects = []
    for obj in objects:
        played |= obj
    if bdj and not played:
        notes.append("这张盘的菜单是 BD-J（Java）写的，看不出菜单播放哪个列表，只能按内容判断")
    return played, notes


def _label(fs: DiscFS) -> str:
    folder = "BDMV/META/DL"
    for name in fs.listdir(folder):
        if not re.match(r"bdmt_\w+\.xml$", name, re.IGNORECASE):
            continue
        try:
            text = fs.read(f"{folder}/{name}", limit=65536).decode("utf-8", "replace")
        except (DiscError, OSError):
            continue
        m = re.search(r"<di:name>(.*?)</di:name>", text, re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""


def _chapters(pl: Playlist) -> List[float]:
    offsets, t = [], 0.0
    for item in pl.items:
        offsets.append(t)
        t += item.seconds
    out: List[float] = []
    for mark in pl.marks:
        if mark.kind != 1 or not 0 <= mark.item < len(pl.items):
            continue
        item = pl.items[mark.item]
        at = offsets[mark.item] + max(mark.time - item.in_time, 0) / TICKS
        if at >= t - 1.0 and out:
            continue                       # a mark at the very end is not a chapter
        if not out or at - out[-1] >= 0.5:
            out.append(round(at, 3))
    if out and out[0] > 0.5:
        out.insert(0, 0.0)
    return out or [0.0]


def _item_bytes(item: PlayItem, clip: Optional[ClipInfo], file_size: Optional[int]) -> Optional[int]:
    if file_size is None:
        return None
    if clip is not None and clip.ep:
        a = clip.spn_at(item.in_time)
        b = clip.spn_at(item.out_time)
        if a is not None and b is not None and b > a:
            whole = clip.source_packets or file_size // 192
            # the last entry point is a GOP before out_time: count the tail
            # to the end of the clip when out_time is the clip's own end
            if clip.end and item.out_time >= clip.end - TICKS:
                b = whole
            return max(b - a, 0) * 192
    if clip is not None and clip.end > clip.start:
        share = (item.out_time - item.in_time) / (clip.end - clip.start)
        return int(file_size * min(max(share, 0.0), 1.0))
    return file_size


def load(located: Located) -> Disc:
    fs = located.fs
    disc = Disc(kind="bd", source=fs.source, path=str(located.root),
                root=str(located.root), name=located.name, titles=[], fs=fs)
    disc.label = _label(fs)
    played, notes = _reachable(fs)
    disc.warnings.extend(notes)

    stream_dir = fs.lookup("BDMV/STREAM")
    disc.analysis_only = stream_dir is None or not fs.listdir("BDMV/STREAM")
    clips: Dict[str, Optional[ClipInfo]] = {}

    def clip_info(clip_id: str) -> Optional[ClipInfo]:
        if clip_id not in clips:
            try:
                clips[clip_id] = parse_clpi(
                    fs.read(f"BDMV/CLIPINF/{clip_id}.clpi"), clip_id)
            except DiscError:
                clips[clip_id] = None
        return clips[clip_id]

    def stream_file(clip_id: str) -> Optional[str]:
        for ext in ("m2ts", "mts", "ssif"):
            found = fs.lookup(f"BDMV/STREAM/{clip_id}.{ext}")
            if found:
                return found
        return None

    bad: List[str] = []
    for name in fs.listdir("BDMV/PLAYLIST"):
        m = re.match(r"(\d{5})\.mpls$", name, re.IGNORECASE)
        if not m:
            continue
        try:
            pl = parse_mpls(fs.read(f"BDMV/PLAYLIST/{name}"), m.group(1))
        except DiscError as exc:
            bad.append(f"{name}（{exc}）")
            continue
        if not pl.items:
            continue
        title = _title(pl, clip_info, stream_file, fs, disc.analysis_only)
        title.reachable = int(pl.name) in played
        disc.titles.append(title)
    if bad:
        disc.warnings.append("以下播放列表读不出来，已忽略：" + "；".join(bad[:5]))
    disc.titles.sort(key=lambda t: t.number)
    disc.encrypted = _encrypted(fs, disc)
    return disc


def _encrypted(fs: DiscFS, disc: Disc) -> bool:
    """Checked on the longest title's first clip: one aligned unit decides."""
    if disc.analysis_only or not disc.titles:
        return False
    longest = max(disc.titles, key=lambda t: t.duration)
    for _item, path in longest.plan.items:
        if path:
            try:
                return looks_encrypted(fs.read(path, limit=6144))
            except (DiscError, OSError):
                return False
    return False


def split_title(title: Title) -> List[Title]:
    """One title per play item of *title* — a "play all" cut back into the
    episodes it was made of. Ids are "<playlist>#<n>"; the same cut of the
    same disc always yields the same ids, which is what lets a queued
    request name them."""
    plan: BdPlan = title.plan
    out: List[Title] = []
    for k, (item, path) in enumerate(plan.items, start=1):
        marks = [Mark(m.kind, 0, m.time) for m in plan.marks if m.item == k - 1]
        share = item.seconds / title.duration if title.duration else 0.0
        out.append(Title(
            id=f"{title.id}#{k}",
            number=title.number,
            duration=round(item.seconds, 3),
            segments=[Segment(item.clip, item.in_time, item.out_time, item.seconds)],
            chapters=_chapters(Playlist(plan.playlist, [item], marks)),
            streams=title.streams,
            angles=len(item.angles) + 1,
            size=None if title.size is None else int(title.size * share),
            problems=list(title.problems),
            notes=list(title.notes),
            plan=BdPlan(plan.playlist, [(item, path)], plan.clips, marks),
        ))
    return out


_GROUP_ORDER = {"video": 0, "audio": 1, "pg": 2, "textst": 2, "ig": 3}


def _title_streams(pl: Playlist, clip_info) -> List[Stream]:
    """The title's stream table: every play item's STN, merged.

    Each play item carries its own table, and they need not agree. Measured
    on a real disc (Nighty Night, 1986): the film's playlist opens with a
    7-second studio logo whose table lists the picture alone, and its LPCM
    and PGS are only in the second item's — reading the first item's table
    remuxed a 74-minute film with no sound and no subtitles, and called it
    done. So the longest item (the film itself) sets the order, which is
    the disc's own (its first audio track is the default one), and anything
    another item lists that it does not is added after its own kind."""
    order = sorted(range(len(pl.items)), key=lambda n: (-pl.items[n].seconds, n))
    seen: Set[Tuple[str, int]] = set()
    picked: List[Tuple[int, int, Stream]] = []
    for rank, n in enumerate(order):
        item = pl.items[n]
        clip = clip_info(item.clip)
        for k, entry in enumerate(item.streams):
            if (entry.group, entry.pid) in seen:
                continue
            seen.add((entry.group, entry.pid))
            picked.append((_GROUP_ORDER.get(entry.group, 9), rank * 1000 + k,
                           _stream_of(entry, clip)))
    picked.sort(key=lambda p: (p[0], p[1]))
    return [s for _g, _k, s in picked]


def _title(pl: Playlist, clip_info, stream_file, fs: DiscFS, analysis_only: bool) -> Title:
    segments, items, problems, notes = [], [], [], []
    size: Optional[int] = 0
    missing = []
    for n, item in enumerate(pl.items):
        segments.append(Segment(item.clip, item.in_time, item.out_time, item.seconds,
                                seamless=n > 0 and item.connection in (5, 6)))
        clip = clip_info(item.clip)
        path = stream_file(item.clip)
        if path is None:
            if not analysis_only:
                missing.append(item.clip)
            size = None
        elif size is not None:
            got = _item_bytes(item, clip, fs.size(path))
            size = None if got is None else size + got
        items.append((item, path or ""))
    if missing:
        problems.append("缺少片段文件：" + "、".join(f"{c}.m2ts" for c in sorted(set(missing))[:5]))

    streams = _title_streams(pl, clip_info)
    if not any(s.kind == "video" for s in streams):
        problems.append("播放列表里没有视频流")
    if any(s.kind == "video" and s.codec == "VC-1" for s in streams):
        problems.append("VC-1 视频：PyAV 不允许把它写进 MKV，暂不支持（以后可以转码）")
    secondary = max(i.secondary for i in pl.items)
    if secondary:
        notes.append(f"另有 {secondary} 条画中画/副音轨，MKV 里不保留")
    if any(i.dolby_vision for i in pl.items):
        notes.append("杜比视界增强层不保留（保留基础层和 HDR10）")
    angles = max((len(i.angles) + 1 for i in pl.items), default=1)
    if angles > 1:
        notes.append(f"多角度（{angles} 个），只导出角度 1")

    clips = {i.clip: clip_info(i.clip) for i in pl.items}
    kinds = {c.application_type for c in clips.values() if c is not None}
    return Title(
        id=f"{pl.name}.mpls",
        number=int(pl.name),
        duration=round(sum(i.seconds for i in pl.items), 3),
        segments=segments,
        chapters=_chapters(pl),
        streams=streams,
        angles=angles,
        size=None if analysis_only else size,
        problems=problems,
        notes=notes,
        still=bool(kinds) and kinds <= {2, 3},
        plan=BdPlan(pl.name, items, {k: v for k, v in clips.items() if v}, pl.marks),
    )


def looks_encrypted(head: bytes) -> bool:
    """Is this the start of an AACS-encrypted m2ts?

    AACS encrypts each 6144-byte aligned unit (32 source packets) except its
    first 16 bytes. In the clear, every 192-byte source packet carries the
    TS sync byte 0x47 at offset 4; in an encrypted unit only the first one
    still does. The copy-permission bits are deliberately not consulted:
    decrypted backups do not all clear them, and a false "encrypted" would
    refuse a disc that is perfectly readable.
    """
    packets = [head[i:i + 192] for i in range(0, min(len(head), 6144), 192)]
    packets = [p for p in packets if len(p) == 192]
    if len(packets) < 2:
        return False
    return any(p[4] != 0x47 for p in packets[1:])
