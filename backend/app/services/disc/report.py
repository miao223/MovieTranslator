"""A disc's analysis as the page and the job log see it.

The same function builds the report the 原盘 page shows and the one the job
re-derives when it actually starts, so "what the list said" and "what was
written" cannot drift apart: the names in the preview are computed here,
once, by the analysis.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from app.models.schemas import DiscReport, DiscStreamInfo, DiscTitleInfo, DiscVolumeInfo
from app.services import audio
from app.services.disc.analyze import Analysis, Verdict, hms
from app.services.disc.model import Disc

# the order categories are listed in: what will be written first
_ORDER = {"main": 0, "episode": 1, "extra": 2, "variant": 3,
          "duplicate": 4, "silent": 5, "still": 6, "short": 7, "loop": 8}


def label(v: Verdict) -> str:
    return {
        "main": "正片",
        "episode": f"第 {v.number} 集",
        "extra": f"花絮 {v.number:02d}",
        "variant": f"版本 {v.number}",
        "duplicate": "重复",
        "short": "过短",
        "loop": "循环",
        "still": "静态图",
        "silent": "无声",
    }.get(v.category, v.category)


# DiscRequest.output_mode as the log, the queue and tools/disc_report say it
OUTPUT_MODES = {
    "beside": "光盘所在的文件夹",
    "inside": "光盘自己的文件夹",
    "custom": "指定的文件夹",
}


def output_dir_for(disc: Disc, mode: str = "beside", custom: str = "") -> Optional[Path]:
    """Where the MKVs go (DiscRequest.output_mode).

    beside: the folder the disc is in — Film (1992)/ → Film (1992).mkv next
    to it, Film.iso → Film.mkv next to it. A box set's volumes are in one
    folder, so this also puts a season together.
    inside: the disc's own folder, the one holding BDMV/VIDEO_TS; an image
    has none, so it gets one beside it, named after it.
    custom: *custom*; None while it has not been chosen.
    """
    root = Path(disc.root)
    if mode == "custom":
        return Path(custom.strip()).expanduser() if custom.strip() else None
    if mode == "inside":
        return root if disc.source == "dir" else root.with_suffix("")
    return root.parent


def free_bytes(folder: Optional[Path]) -> Optional[int]:
    """Free space where *folder* is or will be: an image's own folder
    ("inside") does not exist until the first MKV is written into it."""
    if folder is None:
        return None
    for candidate in (folder, *folder.parents):
        try:
            return shutil.disk_usage(candidate).free
        except OSError:
            continue
    return None


def build(disc: Disc, analysis: Analysis, *, name: str = "", output_mode: str = "beside",
          output_dir: str = "", series: Optional[bool] = None, episode_start: int = 1,
          extra_start: int = 1, volume: Optional[DiscVolumeInfo] = None) -> DiscReport:
    out_dir = output_dir_for(disc, output_mode, output_dir)
    titles: List[DiscTitleInfo] = []
    for t in analysis.titles:
        v = analysis.verdicts[t.id]
        titles.append(DiscTitleInfo(
            id=t.id, number=t.number, label=label(v), category=v.category,
            selected=v.selected, selectable=v.selectable, reason=v.reason,
            duplicate_of=v.duplicate_of, output=v.output,
            ordinal=v.number if v.category in ("episode", "extra", "variant") else 0,
            duration=t.duration, chapters=len(t.chapters), size=t.size,
            video=t.video,
            streams=[DiscStreamInfo(
                kind=s.kind, codec=s.codec, language=s.language,
                language_name=audio.language_name(s.language) if s.language else "",
                detail=s.detail, forced=s.forced, commentary=s.commentary,
                carried=s.carried, note=s.note) for s in t.streams],
            angles=t.angles, segments=len(t.segments), reachable=t.reachable,
            notes=list(t.notes),
        ))
    titles.sort(key=lambda x: (_ORDER.get(x.category, 9),
                               x.number if x.category != "episode" else 0,
                               analysis.verdicts[x.id].number, x.id))
    warnings = list(disc.warnings) + list(analysis.warnings)
    if out_dir is None:
        warnings.append("还没有选输出文件夹")
    return DiscReport(
        path=disc.path, root=disc.root, kind=disc.kind, source=disc.source,
        name=(name or disc.name).strip() or "disc", label=disc.label,
        output_mode=output_mode, output_dir=str(out_dir) if out_dir else "",
        mode=analysis.mode, mode_reason=analysis.mode_reason,
        series_choice=analysis.series_choice, series_default=analysis.series_default,
        series=series, episode_start=episode_start, extra_start=extra_start,
        volume=volume, titles=titles,
        analysis_only=disc.analysis_only, encrypted=disc.encrypted,
        free_bytes=free_bytes(out_dir), warnings=warnings,
    )


def lines(report: DiscReport) -> List[str]:
    """The report as plain text, for the job log and tools/disc_report."""
    kind = {"bd": "蓝光", "dvd": "DVD"}[report.kind]
    src = "镜像" if report.source == "iso" else "文件夹"
    out = [
        f"光盘          : {kind}{src} {report.root}",
        f"片名          : {report.name}" + (f"（盘上的标题：{report.label}）" if report.label else ""),
        f"判断          : {'剧集' if report.mode == 'series' else '电影'}——{report.mode_reason}",
        f"输出目录      : {report.output_dir or '（未选择）'}"
        f"（{OUTPUT_MODES.get(report.output_mode, report.output_mode)}）",
    ]
    if report.volume is not None:
        out.append(f"多卷合集      : {report.volume.note}")
    for warning in report.warnings:
        out.append(f"⚠ {warning}")
    for t in report.titles:
        tick = "☑" if t.selected else ("☐" if t.selectable else "✕")
        size = f"{t.size / (1 << 30):.1f} GB" if t.size else "大小未知"
        out.append(f"{tick} {t.id:<14} {t.label:<6} {hms(t.duration):>8}  "
                   f"{t.chapters} 章  {size}  {t.video}")
        audios = [s for s in t.streams if s.kind == "audio"]
        subs = [s for s in t.streams if s.kind == "subtitle"]
        if audios:
            out.append("      音轨: " + "；".join(
                f"{s.language_name or '未标注'} {s.codec} {s.detail}".strip()
                + ("（评论）" if s.commentary else "")
                + ("（不带）" if not s.carried else "") for s in audios))
        if subs:
            out.append("      字幕: " + "；".join(
                f"{s.language_name or '未标注'} {s.codec}"
                + ("（强制）" if s.forced else "")
                + ("（不带）" if not s.carried else "") for s in subs))
        out.append(f"      {t.reason}")
        if t.output:
            out.append(f"      → {t.output}")
        for note in t.notes:
            out.append(f"      · {note}")
    return out
