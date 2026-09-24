"""From "the discs the user picked" to analysed discs with their names settled.

The 原盘 page's two scans (one disc, a whole folder) and both of its
加入列队 calls go through here. What makes that more than calling analyze()
is the box set (volumes.py): the name and the numbers a volume gets depend
on the volumes beside it — VOL02's first episode is E07 only because VOL01
has six — and a disc must come out under the same names whether it was
scanned alone or with its whole folder. So the set is always read from the
disk, never from what the batch happened to include.

Only series volumes take part. A volume judged a film — the bonus disc of
a box, a film split over two discs — keeps its own name and numbering: in
a shared name its "main" title would be written as the same 片名.mkv as
the next one's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.models.schemas import DiscReport, DiscVolumeInfo
from app.services.disc import report as disc_report
from app.services.disc.analyze import Analysis, analyze
from app.services.disc.binary import DiscError
from app.services.disc.model import Disc
from app.services.disc.volumes import VolumeSet, disc_name, volume_set


@dataclass
class Answer:
    """What the page was told about one disc."""

    series: Optional[bool] = None
    name: str = ""
    episode_start: Optional[int] = None


@dataclass
class Planned:
    path: str
    answer: Answer
    disc: Optional[Disc] = None
    analysis: Optional[Analysis] = None
    error: str = ""
    # the naming in effect, what a queued entry freezes
    name: str = ""
    episode_start: int = 1
    extra_start: int = 1
    own_name: str = ""
    volume: Optional[VolumeSet] = None
    index: int = 0
    chained: bool = False
    note: str = ""


def output_choice(mode: Optional[str], output_dir: str, settings) -> Tuple[str, str]:
    """(mode, custom folder) with the settings' default filled in."""
    mode = mode or settings.disc.output_mode
    if mode == "custom":
        return mode, (output_dir or "").strip() or settings.disc.output_dir.strip()
    return mode, ""


def _analyze(disc: Disc, settings, answer: Answer, **naming) -> Analysis:
    return analyze(disc, export_extras=settings.disc.export_extras,
                   min_seconds=settings.disc.min_title_seconds,
                   series=answer.series, **naming)


def _key(path) -> str:
    return str(Path(path))


def plan(requests: List[Tuple[str, Answer]], settings) -> List[Planned]:
    """Open, number and analyse the discs *requests* name, in order."""
    from app.services.disc import open_disc

    out: List[Planned] = []
    for path, answer in requests:
        item = Planned(path=path, answer=answer)
        try:
            item.disc = open_disc(path)
        except (DiscError, OSError) as exc:
            item.error = str(exc)
        out.append(item)

    by_root = {_key(p.disc.root): p for p in out if p.disc is not None}
    sets: Dict[str, VolumeSet] = {}
    for item in out:
        if item.disc is None:
            continue
        found = volume_set(Path(item.disc.root))
        if found is not None:
            item.volume = sets.setdefault(found.id, found)
            item.index = item.volume.index(Path(item.disc.root))
    for vset in sets.values():
        _number(vset, by_root, settings)

    for item in out:
        if item.disc is None:
            continue
        if not item.chained:
            item.name = item.answer.name.strip() or item.disc.name
            item.episode_start = (item.answer.episode_start
                                  if item.answer.episode_start is not None else 1)
            item.extra_start, item.own_name = 1, ""
        item.analysis = _analyze(
            item.disc, settings, item.answer, name=item.name,
            episode_start=item.episode_start, extra_start=item.extra_start,
            own_name=item.own_name)
    return out


def _number(vset: VolumeSet, by_root: Dict[str, Planned], settings) -> None:
    """Share the set's name and number the series volumes on from each
    other, counting every episode and extra of the volumes before —
    ticked or not, the same rule as extras within one disc."""
    from app.services.disc import open_disc

    members = [disc_name(v.root) for v in vset.volumes]
    head = f"多卷合集的第 {{n}}/{len(members)} 卷（{'、'.join(members)}）"
    next_episode, next_extra = 1, 1
    last: Optional[Tuple[int, int]] = None      # (volume, episodes) numbered before
    unreadable: List[str] = []
    for n, vol in enumerate(vset.volumes, start=1):
        item = by_root.get(_key(vol.root))
        if item is not None:
            disc, answer = item.disc, item.answer
        else:
            try:
                disc, answer = open_disc(str(vol.root)), Answer()
            except (DiscError, OSError):
                unreadable.append(f"第 {n} 卷")
                continue
        analysis = _analyze(disc, settings, answer)
        if analysis.mode != "series":
            if item is not None:
                item.note = (head.format(n=n) + "，但这一卷判断为电影、不是剧集，"
                             "按单独一张盘命名，不和其他卷接着编号")
            continue
        episodes = sum(1 for v in analysis.verdicts.values() if v.category == "episode")
        extras = sum(1 for v in analysis.verdicts.values() if v.category == "extra")
        start = answer.episode_start if answer.episode_start is not None else next_episode
        if item is not None:
            item.chained = True
            item.name = answer.name.strip() or vset.name
            item.episode_start, item.extra_start = start, next_extra
            vol_tag = vol.token or f"{n:02d}"
            item.own_name = f"{item.name}.{vol_tag}"
            note = head.format(n=n) + f"：各卷共用片名「{item.name}」"
            if answer.episode_start is not None:
                note += f"，从你指定的第 {start} 集编起"
            elif last is not None:
                note += f"，接着第 {last[0]} 卷的 {last[1]} 集，从第 {start} 集编起"
            else:
                note += "，集号和花絮编号在各卷之间接着编"
            if extras and next_extra > 1:
                note += f"，花絮从 {next_extra:02d} 编起"
            if unreadable:
                note += (f"。⚠ {'、'.join(unreadable)}读不出来，没算进编号，"
                         "集号可能对不上，请核对起始集号")
            item.note = note
        next_episode, next_extra = start + episodes, next_extra + extras
        last = (n, episodes)


def report_of(item: Planned, settings, output_mode: str, output_dir: str) -> DiscReport:
    volume = None
    if item.volume is not None:
        volume = DiscVolumeInfo(
            id=item.volume.id, name=item.name if item.chained else item.volume.name,
            index=item.index, count=len(item.volume.volumes),
            members=[disc_name(v.root) for v in item.volume.volumes],
            chained=item.chained, note=item.note)
    return disc_report.build(
        item.disc, item.analysis, name=item.name, output_mode=output_mode,
        output_dir=output_dir, series=item.answer.series,
        episode_start=item.episode_start, extra_start=item.extra_start, volume=volume)
