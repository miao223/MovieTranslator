"""Box sets: one series spread over several discs side by side.

A season on Blu-ray comes as SHOW_VOL01, SHOW_VOL02… in one folder.
Remuxed one at a time, every volume would number its episodes from E01 and
be named after its own folder — the second volume's first episode written
as SHOW_VOL02.E01.mkv where SHOW.E07.mkv was meant. Recognised here
as one set, the volumes share a name and number on from each other
(plan.py does the numbering; this module only says which discs belong
together and in what order).

Recognised by name, and narrowly: discs in one folder whose names differ
only in a *volume* number — a number behind a volume word (VOL, Volume,
Disc, Disk, D, BD, DVD, 第N巻) — or bare volume names (DISC1, DISC2). A bare
trailing number is not enough: "Rocky 3" and "Rocky 4" side by side are
sequels, and one set would write both films as Rocky.mkv. The volume word is
part of the key too, so a show's BD1 and DVD1 — the same episodes twice —
stay apart.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.services.disc.fs import _bd_root, _dvd_dir

# tags after the name that are not part of it: [BDMV] [FRA] (1992) 【特典】
_TAGS = re.compile(r"\s*[\[(（【][^\[\]()（）【】]*[\])）】]\s*$")
_VOLUME = re.compile(
    r"^(?P<stem>.*?)(?:(?<=[\s._\-\d])|^)"
    r"(?P<word>vol(?:ume)?|dis[ck]|dvd|bd|d)[\s._\-]*0*(?P<n>\d{1,3})$",
    re.IGNORECASE)
_VOLUME_CJK = re.compile(r"^(?P<stem>.*?)[\s._\-]*第\s*0*(?P<n>\d{1,3})\s*[巻卷]$")
_WORD = {"vol": "vol", "volume": "vol", "disc": "disc", "disk": "disc",
         "dvd": "dvd", "bd": "bd", "d": "d"}


@dataclass
class Volume:
    root: Path          # the disc: folder holding BDMV/VIDEO_TS, or the .iso
    number: int
    token: str          # the volume part as written, separators dropped: "VOL02"


@dataclass
class VolumeSet:
    id: str             # the same for every member: folder + name key
    name: str           # the shared name: "SHOW"
    folder: Path
    volumes: List[Volume]

    def index(self, root: Path) -> int:
        """1-based position of *root* in the set, 0 if not a member."""
        for n, vol in enumerate(self.volumes, start=1):
            if vol.root == root:
                return n
        return 0


@dataclass
class _Parsed:
    key: str
    display: str
    number: int
    token: str


def disc_name(root: Path) -> str:
    """The name a disc goes by in its folder: the folder's, or the image's
    without .iso."""
    return root.stem if root.suffix.lower() == ".iso" else root.name


def parse(name: str) -> Optional[_Parsed]:
    """(what the set is called, which volume) — or None when *name* carries
    no volume number."""
    bare = name
    while True:
        shorter = _TAGS.sub("", bare)
        if shorter == bare:
            break
        bare = shorter
    m = _VOLUME.match(bare.strip())
    word = ""
    if m:
        word = _WORD[m.group("word").lower()]
    else:
        m = _VOLUME_CJK.match(bare.strip())
        if not m:
            return None
        word = "巻"
    stem = re.sub(r"[\s._\-]+$", "", m.group("stem"))
    key = re.sub(r"[\s._\-]+", " ", stem).strip().lower() + "|" + word
    token = re.sub(r"[\s._\-]+", "", bare.strip()[len(m.group("stem")):])
    return _Parsed(key, stem, int(m.group("n")), token)


def discs_in(folder: Path) -> List[Path]:
    """The discs directly in *folder*: disc folders and .iso images."""
    out = []
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return out
    for entry in entries:
        if entry.startswith("."):
            continue
        p = folder / entry
        if p.suffix.lower() == ".iso" and p.is_file():
            out.append(p)
        elif p.is_dir() and (_bd_root(p) is not None or _dvd_dir(p) is not None):
            out.append(p)
    return out


def volume_set(root: Path) -> Optional[VolumeSet]:
    """The set *root* is a volume of, or None when it stands alone."""
    root = Path(root)
    mine = parse(disc_name(root))
    if mine is None:
        return None
    members = []
    for sibling in discs_in(root.parent):
        other = parse(disc_name(sibling))
        if other is not None and other.key == mine.key:
            members.append(Volume(sibling, other.number, other.token))
    numbers = [v.number for v in members]
    # two "VOL01"s (a folder and its .iso, say) is not a set anyone can
    # number — better no set than a wrong one
    if len(members) < 2 or len(set(numbers)) != len(numbers) \
            or not any(v.root == root for v in members):
        return None
    members.sort(key=lambda v: v.number)
    # DISC1 / DISC2 name nothing; the set is the folder they are in
    name = mine.display or root.parent.name or "disc"
    return VolumeSet(f"{root.parent}|{mine.key}", name, root.parent, members)
