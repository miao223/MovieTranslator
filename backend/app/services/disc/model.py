"""What a disc holds, in terms both formats share.

A Blu-ray playlist and a DVD title are the same thing to everything above
the parsers: an ordered run of *content* with a duration, chapters and a
stream table. What makes two titles "the same film" or "a piece of the
film" is the content they point at, not their names or their files — so
each title is described as a list of `Segment`s over content units, and
the analysis (analyze.py) does its interval arithmetic on those alone.

The engine-specific half (which m2ts, which sectors, which PIDs) rides
along in `Title.plan` and is only read by remux.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Stream:
    kind: str                  # "video" | "audio" | "subtitle"
    codec: str                 # display name: "H.264", "TrueHD", "PGS", "VobSub"…
    key: int                   # how the engine finds it: BD PID / DVD stream id
    language: str = ""         # ISO 639-2/B, "" when the disc does not say
    detail: str = ""           # "1080p 23.976" / "5.1" …
    channels: int = 0
    forced: bool = False
    commentary: bool = False
    # False: the disc lists it but it will not be in the MKV (menus,
    # picture-in-picture, text subtitles…). `note` says why.
    carried: bool = True
    note: str = ""


@dataclass
class Segment:
    """One stretch of content, in the coordinates of its unit.

    unit: which content — a clip id on Blu-ray, a title set's sector space
          on DVD. Two titles overlap only where they share a unit.
    start/end: half-open, in unit ticks (BD 45 kHz, DVD sectors).
    seconds: how long this stretch plays. Ticks are not seconds on DVD
          (bitrate varies), so overlap is converted proportionally.
    """

    unit: str
    start: int
    end: int
    seconds: float
    # plays straight on from the previous segment, with no break. A film
    # stored as several clips is joined seamlessly throughout; episodes
    # strung together by a "play all" usually are not.
    seamless: bool = False


@dataclass
class Title:
    id: str                    # "00800.mpls" / "标题 3"
    number: int                # disc order: playlist number / title number
    duration: float
    segments: List[Segment]
    chapters: List[float]      # chapter starts in seconds
    streams: List[Stream]
    angles: int = 1
    # A disc menu or title plays this playlist by number (Blu-ray HDMV
    # movie objects). Only ever a positive signal: BD-J discs and menus that
    # pick the playlist from a register say nothing either way.
    reachable: bool = False
    size: Optional[int] = None  # bytes it will read, None when unknown
    # the disc itself labels it a slideshow (Blu-ray CLPI application type
    # 2/3): pictures to page through, not a film
    still: bool = False
    # Why this title cannot be remuxed at all (VC-1, multi-angle DVD,
    # missing stream files). Empty = it can.
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    plan: object = None        # BdPlan / DvdPlan, for remux.py

    @property
    def video(self) -> str:
        for s in self.streams:
            if s.kind == "video":
                return " ".join(p for p in (s.codec, s.detail) if p)
        return ""

    def units(self) -> set:
        return {seg.unit for seg in self.segments}


@dataclass
class Disc:
    kind: str                  # "bd" | "dvd"
    source: str                # "dir" | "iso"
    path: str                  # what the user pointed at
    root: str                  # folder holding BDMV/VIDEO_TS, or the .iso
    name: str                  # default base name for the outputs
    titles: List[Title]
    fs: object = None          # fs.DiscFS
    label: str = ""            # the disc's own title (BD META), shown only
    # The stream files are not there — a metadata-only copy of the disc.
    # Analysis works exactly the same; sizes are unknown and nothing can be
    # remuxed.
    analysis_only: bool = False
    encrypted: bool = False
    warnings: List[str] = field(default_factory=list)
