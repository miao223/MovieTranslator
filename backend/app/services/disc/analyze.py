"""Which titles of a disc to export, and why — the answer to "the whole film
came out twice" and "some people want the extras, some do not".

Tools that remux discs decide by *titles*: MakeMKV and libbluray drop a
playlist only when it is identical to another, or shorter than a minimum.
Neither notices that one playlist is a piece of another — so a disc that
offers the film and also each of its segments yields the film plus the film
again in pieces. Here every title is reduced to the content it points at
(model.Segment: clip + time range on Blu-ray, sectors on DVD) and the
decisions are interval arithmetic over that content:

1. loop       the same stretch repeated more than twice (libbluray's rule):
              a menu background, not a film
2. identical  the same content in the same order — alternate playlists that
              only differ in which audio track is the default
3. play-all   one title made of several others end to end. Which side to
              keep is the one real judgement call on a disc: episodes of a
              series (keep the parts) or segments of a film (keep the whole)
4. contained  everything it shows is already in something kept
5. variant    shares most of the film but adds minutes of its own — a
              different cut, never picked by default

Nothing is hidden. Every title is listed with a category, a default tick and
a sentence saying why; the thresholds are named constants with their
reasons, to be calibrated on real discs the way every other threshold in
this project is. HandBrake's lesson (#5985) is the reason: a de-duplicator
that silently removes titles eventually removes the one the user wanted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from app.services.disc.model import Disc, Segment, Title

# A film's main feature is at least this long, and stands out: at least
# DOMINANCE times the next title. Below that, two long titles are two of
# something, and 整片/分集 becomes a question for the user.
MAIN_MIN = 40 * 60.0
DOMINANCE = 1.5
# The shortest thing that can be an episode. Creditless openings, previews
# and trailers all sit below it.
EPISODE_MIN = 10 * 60.0
# Episodes of one series run about the same length: "similar" means the
# longest is at most this many times the shortest.
SERIES_SPREAD = 1.6
# Three or more titles this close in length are episodes even without any
# other evidence — unless they play on from each other seamlessly (see
# _continuous). A film cut into three pieces is rarely this even.
TIGHT_SPREAD = 1.2
# "Nothing new": what a title adds beyond what is already kept is at most
# this much. The absolute floor absorbs rounding at clip boundaries, the
# relative part long titles.
NOTHING_NEW = (2.0, 0.01)
# A title is made of others when together they cover this much of it.
PLAY_ALL_COVER = 0.9
# A different cut: shares at least this much of its running time with what
# is kept, and still has this many seconds of its own.
VARIANT_SHARE = 0.5
VARIANT_UNIQUE = 60.0
# libbluray _filter_repeats: a stretch played more than twice is a loop —
# but only when those repeats are most of the title. A "play all" of
# episodes repeats the shared opening and ending once per episode, and that
# is a play-all, not a menu background (libbluray's filter drops it anyway,
# which is harmless there and wrong here, where the reason is shown).
LOOP_REPEATS = 2
LOOP_SHARE = 0.5

# words in a disc's name that say "series"
SERIES_WORDS = re.compile(
    r"(?i)(?:\b(?:ep(?:isodes?)?|vol(?:ume)?|season|s\d{1,2}|tv)\b"
    r"|第.{1,4}[巻卷话話集季]|[巻卷])")


@dataclass
class Verdict:
    category: str            # main episode extra variant duplicate short loop still silent
    selected: bool
    reason: str
    selectable: bool = True
    duplicate_of: str = ""
    number: int = 0          # episode / extra / variant number
    output: str = ""         # the file name it would be written as


@dataclass
class Analysis:
    titles: List[Title]                  # the disc's, plus split-out episodes
    verdicts: Dict[str, Verdict]
    mode: str                            # "movie" | "series"
    mode_reason: str
    # whether 整片/分集 is a real question on this disc (the page shows the
    # switch only then), and what the evidence alone would answer
    series_choice: bool = False
    series_default: bool = False
    warnings: List[str] = field(default_factory=list)


def hms(seconds: float) -> str:
    whole = int(round(seconds))
    h, rest = divmod(whole, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --------------------------------------------------------------- intervals

Union = Dict[str, List[Tuple[int, int]]]


def _add(union: Union, title: Title) -> None:
    for seg in title.segments:
        if seg.end <= seg.start:
            continue
        spans = sorted(union.get(seg.unit, []) + [(seg.start, seg.end)])
        merged: List[Tuple[int, int]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        union[seg.unit] = merged


def _union(titles: Sequence[Title]) -> Union:
    out: Union = {}
    for t in titles:
        _add(out, t)
    return out


def _covered_segment(seg: Segment, union: Union) -> float:
    span = seg.end - seg.start
    if span <= 0:
        return 0.0
    overlap = sum(max(0, min(b, seg.end) - max(a, seg.start))
                  for a, b in union.get(seg.unit, ()))
    return seg.seconds * min(overlap, span) / span


def covered(title: Title, union: Union) -> float:
    """Seconds of *title* that *union* already holds."""
    return sum(_covered_segment(seg, union) for seg in title.segments)


def _slack(title: Title) -> float:
    """How much of *title* may be unaccounted for and still count as
    "nothing new". Never more than a quarter of the title: the absolute
    floor is for rounding at clip boundaries, and must not make every
    title shorter than it look contained in anything at all."""
    return min(max(NOTHING_NEW[0], NOTHING_NEW[1] * title.duration), 0.25 * title.duration)


def _nothing_new(title: Title, union: Union) -> bool:
    return title.duration - covered(title, union) <= _slack(title)


def _similar(titles: Sequence[Title], spread: float) -> bool:
    durations = [t.duration for t in titles]
    return bool(durations) and min(durations) > 0 and \
        max(durations) / min(durations) <= spread


def _named_like_series(disc: Disc) -> bool:
    return bool(SERIES_WORDS.search(f"{disc.name} {disc.label}"))


def _continuous(title: Title) -> bool:
    """Every join inside *title* is seamless — one piece of film stored as
    several clips. A "play all" of episodes stops between them."""
    return len(title.segments) >= 2 and all(s.seamless for s in title.segments[1:])


def series_evidence(parts: Sequence[Title], disc: Disc,
                    whole: Optional[Title] = None) -> str:
    """Why *parts* look like episodes of a series, or "" when they do not.

    *whole* is the title they make up together, when there is one: if it
    plays through them seamlessly, only a shared opening/ending still
    counts — similar lengths and a series-like name are exactly what a film
    split into clips can also have.
    """
    if len(parts) < 2 or any(t.duration < EPISODE_MIN for t in parts):
        return ""
    if not _similar(parts, SERIES_SPREAD):
        return ""
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if covered(a, _union([b])) > NOTHING_NEW[0]:
                return "各集共用同一段片头/片尾"
    if whole is not None and _continuous(whole):
        return ""
    if len(parts) >= 3 and _similar(parts, TIGHT_SPREAD):
        shortest = min(t.duration for t in parts)
        longest = max(t.duration for t in parts)
        return f"{len(parts)} 段时长几乎一样（{hms(shortest)}–{hms(longest)}）"
    if _named_like_series(disc):
        return "盘名看起来是剧集"
    return ""


# ---------------------------------------------------------------- the rules

def _placement(title: Title, owner: Title) -> List[Tuple[float, float]]:
    """Where *title*'s content sits on *owner*'s timeline, in seconds."""
    spans: List[Tuple[float, float]] = []
    t0 = 0.0
    timeline = []
    for seg in owner.segments:
        timeline.append((seg, t0))
        t0 += seg.seconds
    for seg in title.segments:
        for own, at in timeline:
            if own.unit != seg.unit:
                continue
            a, b = max(own.start, seg.start), min(own.end, seg.end)
            if b <= a:
                continue
            width = own.end - own.start
            spans.append((at + own.seconds * (a - own.start) / width,
                          at + own.seconds * (b - own.start) / width))
    merged: List[Tuple[float, float]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + NOTHING_NEW[0]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _is_piece(title: Title, owner: Title) -> bool:
    """*title* is one unbroken stretch of *owner* — a segment of the film,
    not a different edit of it that skips parts."""
    return any(b - a >= title.duration - _slack(title) for a, b in _placement(title, owner))


def _repeats(title: Title) -> int:
    """How often the title's most repeated stretch plays, when repeats are
    what the title mostly is; 0 otherwise."""
    counts: Dict[Tuple[str, int, int], int] = {}
    seconds: Dict[Tuple[str, int, int], float] = {}
    for seg in title.segments:
        key = (seg.unit, seg.start, seg.end)
        counts[key] = counts.get(key, 0) + 1
        seconds[key] = seg.seconds
    looping = [k for k, n in counts.items() if n > LOOP_REPEATS]
    share = sum(seconds[k] * counts[k] for k in looping)
    if not looping or share < LOOP_SHARE * max(title.duration, 1e-9):
        return 0
    return max(counts[k] for k in looping)


def _rank(title: Title):
    """Which of several equal titles represents them: the menu's own choice
    first, then the one carrying more tracks, then disc order."""
    return (not title.reachable, -sum(1 for s in title.streams if s.carried),
            title.number, title.id)


def _longest_first(titles: Sequence[Title]) -> List[Title]:
    return sorted(titles, key=lambda t: (-t.duration, _rank(t)))


def _split_play_items(title: Title, disc: Disc) -> List[Title]:
    """A lone "play all" title cut into its play items (Blu-ray only)."""
    if disc.kind != "bd" or len(title.segments) < 2:
        return []
    from app.services.disc import bdmv

    return bdmv.split_title(title)


def _chosen(series: Optional[bool]) -> str:
    return "按你的选择：" + ("分集导出" if series else "整片导出")


def analyze(disc: Disc, *, export_extras: bool = True, min_seconds: float = 60.0,
            series: Optional[bool] = None, episode_start: int = 1,
            name: str = "", extra_start: int = 1, own_name: str = "") -> Analysis:
    """Classify every title of *disc*.

    series: None decides 整片 vs 分集 from the evidence; True / False is the
    user's answer from the switch. It only applies where the disc actually
    poses the question (Analysis.series_choice).

    A volume of a box set (volumes.py) shares *name* with the other volumes
    and numbers on from them: *episode_start* and *extra_start* say where.
    What is numbered per disc — another version (版本2), a title named
    after its number on the disc — would then collide across the volumes,
    so it is named with *own_name* ("SHOW.VOL02") instead.
    """
    base = (name or disc.name).strip() or "disc"
    own = own_name.strip() or base
    verdicts: Dict[str, Verdict] = {}
    warnings: List[str] = []
    choice = False

    def mark(t: Title, category: str, reason: str, of: str = "") -> None:
        verdicts[t.id] = Verdict(category, False, reason, duplicate_of=of)

    # 1. loops and empty titles
    live: List[Title] = []
    for t in sorted(disc.titles, key=lambda t: t.number):
        repeats = _repeats(t)
        if repeats:
            mark(t, "loop", f"循环：同一段画面重复 {repeats} 次，通常是菜单背景")
        elif t.still:
            mark(t, "still", "静态图：光盘把它标成幻灯片（一页页的图），不是视频")
        elif t.duration < 1.0:
            mark(t, "short", "空的：几乎没有时长（静止画面或占位）")
        else:
            live.append(t)

    # 2. identical content
    same: Dict[Tuple, List[Title]] = {}
    for t in live:
        same.setdefault(tuple((s.unit, s.start, s.end) for s in t.segments), []).append(t)
    live = []
    for members in same.values():
        members.sort(key=_rank)
        live.append(members[0])
        for other in members[1:]:
            mark(other, "duplicate",
                 f"重复：与 {members[0].id} 内容完全相同（通常只是默认音轨/字幕不同）",
                 members[0].id)
    live = _longest_first(live)
    longest = live[0] if live else None

    # 3. play-all titles
    # (split?, why, what the evidence alone says) for the film itself
    decision: Optional[Tuple[bool, str, bool]] = None
    for c in list(live):
        if c.id in verdicts:
            continue
        inside = [p for p in live
                  if p is not c and p.id not in verdicts and p.duration < c.duration
                  and _nothing_new(p, _union([c]))]
        # the maximal pieces only: an "opening only" playlist inside every
        # episode is not itself a part of the play-all
        parts = [p for p in inside
                 if not any(q is not p and q.duration > p.duration
                            and _nothing_new(p, _union([q])) for q in inside)]
        if len(parts) < 2 or covered(c, _union(parts)) < PLAY_ALL_COVER * c.duration:
            continue
        is_main = c is longest
        asked = is_main and sum(p.duration >= EPISODE_MIN for p in parts) >= 2
        choice = choice or asked
        evidence = series_evidence(parts, disc, whole=c)
        if evidence:
            auto, why = True, f"像剧集（{evidence}）"
        elif is_main:
            auto, why = False, "它是正片，其余几段都是它的一部分"
        else:
            auto, why = True, "各段单独导出更好用"
        split = auto
        if asked and series is not None:
            split, why = series, _chosen(series)
        if split:
            names = "、".join(p.id for p in sorted(parts, key=lambda t: t.number))
            mark(c, "duplicate", f"重复：「全部播放」，由 {names} 拼成——{why}，已按各段导出")
        else:
            for p in inside:
                mark(p, "duplicate", f"重复：整段都在 {c.id} 里，是它的一段——{why}", c.id)
        if is_main:
            decision = (split, why, auto)
    live = [t for t in live if t.id not in verdicts]

    # a lone long title that may be several episodes end to end
    pieces: List[Title] = []
    if live and decision is None and not _continuous(_longest_first(live)[0]):
        head = _longest_first(live)[0]
        cut = _split_play_items(head, disc)
        long_cut = [p for p in cut if p.duration >= EPISODE_MIN]
        if len(long_cut) >= 2 and _similar(long_cut, SERIES_SPREAD):
            choice = True
            evidence = series_evidence(long_cut, disc)
            # Two independent signs must agree before a title is cut up: a
            # film is often stored as a few clips of similar length, and
            # cutting a film into pieces is precisely the failure this
            # module exists to prevent.
            auto = bool(evidence) and _named_like_series(disc) and len(long_cut) >= 3
            why = (f"像剧集（{evidence}）" if auto else
                   f"由 {len(cut)} 段时长相近的片段组成，可能是剧集的「全部播放」，"
                   "默认按整片导出")
            split = auto
            if series is not None:
                split, why = series, _chosen(series)
            decision = (split, why, auto)
            if split:
                mark(head, "duplicate",
                     f"重复：「全部播放」，已按其中的 {len(cut)} 段分别导出——{why}")
                pieces = cut
                live = [t for t in live if t is not head] + cut

    # 4 + 5. contained / different cuts, longest first
    kept: List[Title] = []
    union: Union = {}
    variants: List[Title] = []
    cuts: Dict[str, str] = {}          # shorter cut -> the title it cuts down
    for t in _longest_first(live):
        if kept and _nothing_new(t, union):
            owner = max(kept, key=lambda k: covered(t, _union([k])))
            if not _nothing_new(t, _union([owner])):
                mark(t, "duplicate", "重复：内容都已在其他标题里，没有新内容", owner.id)
            elif _is_piece(t, owner):
                mark(t, "duplicate", f"重复：整段都在 {owner.id} 里，是它的一段", owner.id)
            else:
                # everything it shows is in the owner, but not in one
                # stretch: the same film with parts left out
                variants.append(t)
                cuts[t.id] = owner.id
            continue
        shared = covered(t, union) if kept else 0.0
        if kept and shared >= VARIANT_SHARE * t.duration \
                and t.duration - shared >= VARIANT_UNIQUE:
            variants.append(t)
        kept.append(t)
        _add(union, t)

    # 6. a film, or a series
    body = [t for t in kept if t not in variants]
    long = _longest_first([t for t in body if t.duration >= EPISODE_MIN])
    dominant = bool(long) and long[0].duration >= MAIN_MIN and (
        len(long) == 1 or long[0].duration >= DOMINANCE * long[1].duration)
    if decision is not None:
        is_series, mode_reason, evidence_default = decision
    elif len(long) >= 2:
        # A film that dwarfs everything else settles it; only two or more
        # comparable long titles make 整片/分集 a question worth asking.
        choice = choice or not dominant
        evidence = "" if dominant else series_evidence(long, disc)
        evidence_default = bool(evidence)
        if series is not None and not dominant:
            is_series, mode_reason = series, _chosen(series)
        elif evidence:
            is_series, mode_reason = True, f"像剧集（{evidence}）"
        elif dominant:
            is_series = False
            mode_reason = (f"正片是 {long[0].id}：全盘最长（{hms(long[0].duration)}），"
                           f"是第二长的 {long[0].duration / long[1].duration:.1f} 倍")
        else:
            is_series = False
            mode_reason = "看不出是剧集，按电影处理：最长的当正片"
    else:
        is_series, evidence_default = False, False
        mode_reason = (f"正片是 {long[0].id}：全盘最长（{hms(long[0].duration)}）"
                       + ("，菜单直接播放它" if long[0].reachable else "")
                       if long else "没有长于 10 分钟的内容，最长的当正片")

    # 7. categories and names
    episodes: List[Title] = []
    main: Optional[Title] = None
    if is_series:
        episodes = list(long) or _longest_first(body)[:1]
    else:
        main = (long or _longest_first(body) or [None])[0]

    def body_key(t: Title):
        """Episodes in the order of their own content: the first stretch
        no other episode shares (the shared ones are the opening)."""
        others = _union([e for e in episodes if e is not t])
        own = [s for s in t.segments if _covered_segment(s, others) <= 0]
        first = (own or t.segments)[0]
        return (first.unit, first.start, t.number, t.id)

    for i, t in enumerate(sorted(episodes, key=body_key)):
        n = episode_start + i
        verdicts[t.id] = Verdict("episode", True, f"第 {n} 集：{mode_reason}",
                                 number=n, output=f"{base}.E{n:02d}.mkv")
    if main is not None:
        verdicts[main.id] = Verdict("main", True, f"正片：{mode_reason}",
                                    output=f"{base}.mkv")

    for k, t in enumerate(sorted(variants, key=lambda t: (t.number, t.id)), start=2):
        if t.id in cuts:
            reason = (f"另一版本：内容都在 {cuts[t.id]} 里，但跳过了其中一些段落"
                      "（可能是剧场版/删减版）")
        else:
            others = _union([x for x in kept if x is not t])
            share = covered(t, others) / t.duration
            reason = (f"另一版本：与其他内容共享 {share:.0%}，另有 "
                      f"{hms(t.duration * (1 - share))} 独有的片段（可能是加长版或导演剪辑版）")
        verdicts[t.id] = Verdict("variant", False, reason, number=k,
                                 output=f"{own}.版本{k}.mkv")

    # extras are numbered over every extra in disc order, ticked or not, so
    # unticking one never renumbers the rest and exporting it later cannot
    # collide with a name already written
    n_extra = max(extra_start, 1) - 1
    for t in sorted((t for t in body if t.id not in verdicts),
                    key=lambda t: (t.number, t.id)):
        if t.duration < min_seconds:
            mark(t, "short", f"过短：只有 {hms(t.duration)}（短于 {int(min_seconds)} 秒的"
                             "一般是片头 logo、警告或菜单背景）")
            continue
        if not any(x.kind == "audio" and x.carried for x in t.streams):
            # A featurette or a trailer always has sound. Measured on a
            # series Blu-ray: its one silent title was a minute of menu
            # background, listed as 花絮01 before this rule.
            mark(t, "silent", f"无声：{hms(t.duration)} 的画面但一条音轨都没有，多半是菜单背景")
            continue
        n_extra += 1
        verdicts[t.id] = Verdict("extra", export_extras,
                                 f"花絮：非正片内容（{hms(t.duration)}）",
                                 number=n_extra, output=f"{base}.花絮{n_extra:02d}.mkv")

    all_titles = sorted(disc.titles, key=lambda t: t.number) + pieces
    for t in all_titles:
        v = verdicts[t.id]
        if not v.output:
            # not written by default, but the user may tick it anyway: it is
            # then named after its number on the disc ("片名.00012.mkv")
            tag = t.id.rsplit(".", 1)[0] if t.id.endswith(".mpls") else t.id
            v.output = f"{own}.{tag.replace('.mpls#', '-').replace('#', '-')}.mkv"
        if t.problems:
            v.selectable = False
            v.selected = False
            v.reason = "不支持：" + "；".join(t.problems) + "。" + v.reason
    if disc.encrypted:
        warnings.append("这张盘仍是加密状态，本程序不做解密，无法封装")
        for v in verdicts.values():
            v.selectable = False
            v.selected = False
    return Analysis(all_titles, verdicts, "series" if is_series else "movie",
                    mode_reason, series_choice=choice,
                    series_default=bool(evidence_default), warnings=warnings)
