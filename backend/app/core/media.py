"""Media file constants and directory scanning."""

from __future__ import annotations

import glob
import re

from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".ts", ".m2ts", ".mpg", ".mpeg", ".rmvb", ".m4v",
}

# Releases are often shared as the audio track alone — a film is tens of GB,
# its audio a few dozen MB — and the pipeline's first step throws the picture
# away anyway. A few of these (.mka, .ogg) can legally carry video; the suffix
# only decides what gets listed and offered, `audio.has_picture` decides how a
# file is actually run.
AUDIO_EXTS = {
    ".mp3", ".m4a", ".m4b", ".aac", ".flac", ".wav", ".ogg", ".oga",
    ".opus", ".wma", ".ape", ".alac", ".mka", ".dts", ".ac3", ".eac3",
    ".amr", ".aiff", ".aif", ".wv", ".tta", ".mpc", ".caf", ".w64", ".mp2",
}

MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS

# A subtitle can be the job's source all by itself — the film is often
# somewhere else. The text ones are read directly; the graphic ones hold
# pictures of words and go through OCR, exactly as a graphic track inside a
# video does. `.sub` is in both worlds: MicroDVD/SubViewer store text under
# that name, while VobSub uses it for the binary half of an .idx/.sub pair —
# which of the two it is cannot be decided here (see subsource.file_track).
SUBTITLE_TEXT_EXTS = {".srt", ".ass", ".ssa", ".vtt", ".sub"}
SUBTITLE_GRAPHIC_EXTS = {".sup", ".idx"}
SUBTITLE_EXTS = SUBTITLE_TEXT_EXTS | SUBTITLE_GRAPHIC_EXTS

# Everything this program will accept as a source. MEDIA_EXTS keeps meaning
# "has a soundtrack"; this is the wider table the file picker and the scan ask.
SOURCE_EXTS = MEDIA_EXTS | SUBTITLE_EXTS


def is_audio_ext(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def is_subtitle_ext(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUBTITLE_EXTS


def kind_of(path: str | Path) -> str:
    """'video' / 'audio' / 'subtitle' / '' — what the picker labels an entry."""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in SUBTITLE_EXTS:
        return "subtitle"
    return ""


def probe_kind(path: str | Path) -> str:
    """What this job is really reading: 'video' / 'audio' / 'subtitle'.

    The suffix decides the subtitle case and deliberately gets the first
    word, against this project's usual "the suffix lists, PyAV decides"
    rule. The reason is that the probe cannot be trusted here:
    audio.has_picture answers `not is_audio_ext(path)` — i.e. True, "this
    is a film" — for anything it fails to open, and subtitles fail to open
    routinely. ffmpeg's MicroDVD prober needs three cue lines, so a short
    .sub is unreadable; so is a .idx whose .sub is missing. Calling those a
    film would arm the encoder check and let frame_only through, which is
    the one path that rewrites an existing subtitle in place.
    """
    from app.services import audio

    if is_subtitle_ext(path):
        return "subtitle"
    return "video" if audio.has_picture(path) else "audio"


class ScanResult(NamedTuple):
    """What a directory holds, once每部片只留一个翻译对象。

    *skipped* 已经译成目标语言了（读文件判的，不是看名字）。*shadowed* 是同一
    部片里让开的其它文件（音频让给视频）。*replaced* 是**被同名字幕顶替掉的
    视频/音频**——单独列出来是因为它要说的话完全不同：更快了，但这一批不会
    产出内嵌视频、也不会用语音识别。*with_source* 仍是「有外语字幕可当原料」
    的那些，只报告不替人做主。
    """

    to_translate: List[Path]
    skipped: List[Path]
    shadowed: List[Path]
    with_source: List[Path]
    replaced: List[Path]


# 让路留下的副本（film.zh.2.srt）不是成品，也不该被当成下一次的原料
_STEPPED_ASIDE = re.compile(
    r"\.(?:[a-z]{2,3}|orig)(?:-(?:[a-z]{2,3}|orig))?\.\d+$", re.IGNORECASE)

# 同一组里挑哪一份字幕当原文：文字快而准，图形要 OCR，按分钟计
_FORMAT_ORDER = (".ass", ".ssa", ".srt", ".vtt", ".sub", ".sup", ".idx")


def _group_key(path: Path, media_stems: set[str]) -> str:
    """这个文件说的是哪部片。

    字幕要认领到它伺候的那部片上：film.mkv 旁边的 film.backup.srt 属于
    film，而不是自成一部叫「film.backup」的片子——否则它会被当成一份独立
    字幕翻译，而它只是别人的备份。
    """
    stem = path.stem.lower()
    if not is_subtitle_ext(path):
        return stem
    if stem in media_stems:
        return stem
    base = base_stem(path).lower()
    if base in media_stems:
        return base
    owners = [m for m in media_stems if stem.startswith(m + ".")]
    return max(owners, key=len) if owners else base


def _usable_source(path: Path, key: str) -> bool:
    """这份字幕能不能当这部片的原文。

    名字必须正好是「这部片 + 一个语言后缀」：film.backup.srt 不是（backup
    不是语言），film.zh.2.srt 更不是（那是防覆盖留下的副本）。
    """
    return (base_stem(path).lower() == key
            and not _STEPPED_ASIDE.search(path.stem))


def scan_media(
    directory: str | Path,
    recursive: bool = True,
    skip_existing_srt: bool = True,
    target_language: str = "",
    prefer_language: str = "",
) -> ScanResult:
    """Find everything under *directory* that can be translated.

    每个 (目录, 片名主干) 只产出一个翻译对象，组内优先级是
    **字幕 > 视频 > 音频**：一份现成的字幕十几秒读完、不占 GPU，而且是人对着
    画面敲的。代价是这一组拿不到内嵌视频、也不会走语音识别——所以被顶替的
    那些单独装进 `replaced`，调用方必须说出来。

    音频让给同名视频是这条规则的一个特例（两者会写出同一份字幕，有画面的赢）。

    Hidden directories/files (dot-prefixed) are ignored.
    """
    from app.services.audio import canon_language

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"不是有效目录: {directory}")

    files: List[Path] = []
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if not p.is_file() or p.suffix.lower() not in SOURCE_EXTS:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        files.append(p)
    files.sort()

    # VobSub 的 .idx + .sub 是一对，只留索引那一半，否则一对会变成两个任务
    paired = {p.with_suffix(".sub") for p in files if p.suffix.lower() == ".idx"}
    files = [p for p in files if p not in paired]

    groups: Dict[Tuple[Path, str], List[Path]] = {}
    for parent in {p.parent for p in files}:
        stems = {p.stem.lower() for p in files
                 if p.parent == parent and not is_subtitle_ext(p)}
        for p in (f for f in files if f.parent == parent):
            groups.setdefault((parent, _group_key(p, stems)), []).append(p)

    to_translate, skipped, shadowed, with_source, replaced = [], [], [], [], []
    known: Dict[Tuple[Path, str], tuple[str, str]] = {}
    for (parent, key), members in groups.items():
        subs = [m for m in members
                if is_subtitle_ext(m) and _usable_source(m, key)]
        videos = [m for m in members if kind_of(m) == "video"]
        audios = [m for m in members if kind_of(m) == "audio"]
        if not (subs or videos or audios):
            # 这一组里一个能用的都没有——只有让路留下的副本，或者别人的备份。
            # 既不是待翻译的，也不是「已完成」的证据，索性当它不存在。
            continue
        if (parent, key) not in known:
            known[(parent, key)] = subtitle_state(parent, key, target_language)
        state, _lang = known[(parent, key)]

        wanted = canon_language(prefer_language) if prefer_language else ""
        subs.sort(key=lambda m: (
            m.suffix.lower() in SUBTITLE_GRAPHIC_EXTS,
            bool(wanted) and canon_language(split_language_tag(m)[1]) != wanted,
            _FORMAT_ORDER.index(m.suffix.lower())
            if m.suffix.lower() in _FORMAT_ORDER else len(_FORMAT_ORDER),
            m.name,
        ))
        rest = videos + audios
        if state == "done" and skip_existing_srt:
            # 报告的代表仍是片子本身：用户认的是片子，不是它旁边那份字幕
            first = (videos or audios or subs)[0]
            skipped.append(first)
            shadowed.extend(m for m in rest if m is not first)
            continue
        pick = (subs or videos or audios)[0]
        to_translate.append(pick)
        if subs and rest:
            replaced.extend(rest)          # 被字幕顶替掉的视频/音频
        else:
            shadowed.extend(m for m in rest if m is not pick)
        if state == "source" and not is_subtitle_ext(pick):
            with_source.append(pick)

    order = {p: n for n, p in enumerate(files)}
    for group in (to_translate, skipped, shadowed, with_source, replaced):
        group.sort(key=lambda p: order.get(p, 0))
    return ScanResult(to_translate, skipped, shadowed, with_source, replaced)


# A language suffix as this program writes it: two or three letters (which
# covers the "und"/"sub" fallbacks), the longer "orig", or a bilingual pair
# of those joined by a hyphen — film.en-zh.srt. Region tags like zh-cn come
# along for free and land right: either half naming the target is enough.
# Still deliberately narrow — film.backup.srt, film.v2.srt and the
# film.zh.2.srt a step-aside leaves behind are not results and must not be
# read as any language.
_LANG_SUFFIX = re.compile(r"^(?:[a-z]{2,3}|orig)(?:-(?:[a-z]{2,3}|orig))?$")

# 名字里这几个词不是语言，是「说不出是什么语言」。必须在 iso2 之前滤掉：
# iso2("orig") 会顺着前两个字母给出 "or"（奥里亚语），iso2("sub") 给 "su"。
_NOT_A_LANGUAGE = ("orig", "und", "sub")


def split_language_tag(path: str | Path) -> tuple[str, str]:
    """('film', 'en') for film.en.srt; ('film', '') for film.srt.

    只看最后一段，且只认 _LANG_SUFFIX 那条窄规则——发行版片名里全是点
    （Movie.2019.1080p.srt 的 1080p 不是语言），剥错一段就会把产物写到
    别人的名字上。
    """
    stem = Path(path).stem
    head, sep, tail = stem.rpartition(".")
    return (head, tail) if sep and _LANG_SUFFIX.match(tail.lower()) else (stem, "")


def base_stem(path: str | Path) -> str:
    """一份字幕说的是哪部片：film.en.srt 与 film.mkv 都是 film。"""
    return split_language_tag(path)[0]


def subtitles_beside(folder: Path, stem: str) -> List[Path]:
    """Every subtitle file sitting beside *stem*, ours or anyone else's.

    收的是 (目录, 片名主干) 而不是一个 Path：调用方常常只有一个主干，而
    Path("/x/Movie.2019.1080p").with_suffix(".srt") 会得到 Movie.2019.srt
    ——发行版片名里全是点，这个坑一碰就是系统性的。
    """
    found = [p for p in (folder / f"{stem}.srt", folder / f"{stem}.ass")
             if p.is_file()]
    for sibling in sorted(folder.glob(f"{glob.escape(stem)}.*")):
        if sibling.suffix.lower() not in (".srt", ".ass") or not sibling.is_file():
            continue
        middle = sibling.name[len(stem) + 1: -len(sibling.suffix)]
        if _LANG_SUFFIX.match(middle.lower()):
            found.append(sibling)
    return found


def subtitle_state(folder: Path, stem: str,
                   target_language: str = "") -> tuple[str, str]:
    """What the subtitle next to *video* means: (state, its language).

    ``done``   already in the target language — this film is translated.
    ``source`` in some other language — not a result but **material**, and
               better material than speech recognition, since a person
               typed it against the picture.
    ``none``   nothing there.

    The distinction cannot be made from the filename: this program's own
    original_only output and a subtitle downloaded from anywhere else are
    both ``film.ja.srt``. So the file is read — a few dozen KB, and only
    its opening lines.

    Unsure always means ``done``. An unreadable subtitle, or a target
    language this build has no code for, leaves no way to compare, and
    re-translating a whole season is the more expensive mistake of the two.

    One file in the target language settles it, but ``source`` is only the
    answer once **every** sibling has been looked at. A film can have two
    subtitles beside it — the 双文件 mode leaves film.en.srt and
    film.zh.srt together — and the glob is alphabetical, so the original is
    what turns up first. Stopping there called a finished film untranslated
    and translated it again, every scan, forever.
    """
    from app.services import mux, subsource

    want = mux.language_of(target_language)[0] if target_language else ""
    if want == mux.FALLBACK[0]:
        want = ""                       # no code for this target language
    state, found = "none", ""
    for path in subtitles_beside(folder, stem):
        tag = path.name[len(stem) + 1: -len(path.suffix)].lower()
        languages: tuple[str, ...] = ()
        if _LANG_SUFFIX.match(tag):
            languages = tuple(
                code for code in (subsource.iso2(part) for part in tag.split("-")
                                  if part not in _NOT_A_LANGUAGE)
                if code
            )
        if not languages:
            one = _language_of_file(path)
            languages = (one,) if one else ()
        if not languages or not want or want in languages:
            return "done", want if want in languages else (
                languages[0] if languages else "")
        if state == "none":
            # 配对时报第一段：默认排版下那是原文，而 source 说的正是「这里有
            # 可以拿来译的材料」
            state, found = "source", languages[0]
    return state, found


def _language_of_file(path: Path) -> str:
    """Guess a subtitle file's language from its text.

    Parsed with a few lines here rather than through subsource's real
    reader: that one opens a container per file, and a scan may cover a
    hundred of them for an answer that only needs some words.
    """
    from app.models.schemas import SubtitleLine
    from app.services import subsource

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read(120_000)          # the opening lines settle it
    except OSError:
        return ""
    texts: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "-->" in line or line.isdigit():
            continue                        # srt index and timing lines
        if line.startswith("Dialogue:"):    # ass event: text is the last field
            line = line.split(",", 9)[-1]
        elif line.split(":", 1)[0].isalpha() and ":" in line[:14] and "," in line:
            continue                        # other ass header/event fields
        texts.append(subsource.event_text(line, ass=path.suffix.lower() == ".ass"))
    lines = [SubtitleLine(index=i, start=0.0, end=0.0, text=t)
             for i, t in enumerate(texts[:400], 1) if t.strip()]
    return subsource.detect_language(lines) if lines else ""
