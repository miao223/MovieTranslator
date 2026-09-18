"""Read a graphic subtitle track by recognising its pictures.

PGS (Blu-ray), VobSub (DVD) and DVB subtitles carry no text at all — each
cue is a bitmap. That is most of what real releases ship, so refusing them
means falling back to an hour of speech recognition on a film that already
has its dialogue typed out.

What makes this tractable is that the input is *rendered* text, not a
photograph: one typeface per disc, no noise, no perspective, huge contrast.
And the timing comes free — every cue already carries its own start and end,
so recognition only has to solve the words, never the boundaries.

Getting a clean image out is the part that decides the quality, and it has
one obstacle: **the palette is unreachable**. FFmpeg puts it in
``AVSubtitleRect.data[1]`` but leaves ``linesize[1]`` at zero, and PyAV
enumerates planes by linesize, so only the index bitmap is exposed. It turns
out not to matter. The fill and the outline are *different indices*, and the
outline wraps around the fill, so ranking indices by how far their pixels
sit from the transparent background separates them by geometry rather than
by colour: measured on generated discs the fill sits around 6.1 and the
outline around 3.3, anti-aliased edges included — and on a real DVD, whose
glyphs are a third the size, 3.4 against 2.2.
"""

from __future__ import annotations

import base64
import io
import re
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models.schemas import (
    LLMSettings,
    NetworkSettings,
    OcrSettings,
    SubtitleLine,
)
from app.services import mux
from app.services.asr import has_content

LogFn = Callable[[str], None]
ProgressFn = Callable[[float], None]  # 0..1

MISSING_DEPENDENCY = (
    "图形字幕需要 OCR 组件，请先安装：\n"
    '  Windows: .venv\\Scripts\\pip install -e ".[ocr]"\n'
    '  Linux/macOS: .venv/bin/pip install -e ".[ocr]"\n'
    "（或在设置里把「图形字幕 OCR」的引擎改成「视觉大模型」，那条路不需要本地组件）"
)

# A rect with essentially no transparent pixels is a solid plate with the
# glyphs knocked out of it, not glyphs floating on nothing. The depth rule
# cannot see the difference, so it is asked first.
OPAQUE_BELOW = 0.02
# An index this much of the way to the deepest one is part of the glyph.
# Two measurements set it. On a Blu-ray's big glyphs the fill sits around
# 6.1 and the outline around 3.3 — a ratio of 0.54. On a real 720x576 DVD
# the glyphs are small and the outline thin, and the same two numbers come
# out 3.4 and 2.2 — 0.65. A threshold of 0.6 therefore split Blu-ray
# correctly and called a DVD's outline part of the letter, which then
# tripped the "too much ink" inversion below and emptied the mask: 602 of
# that disc's 1144 subtitles vanished, because bitmap_cues reads an empty
# mask as a clear event. 0.7 clears both, and the floor below catches
# whatever a third format does.
DEPTH_RATIO = 0.7
# indices thinner than this share of the opaque area are edge blending, and
# their means are too noisy to rank
MIN_INDEX_SHARE = 0.01
MAX_DEPTH_ROUNDS = 40
# ink covering more than this is a background, whatever the geometry said
INK_CEILING = 0.5
# A cue whose end never arrives gets this. PGS ends a cue with an explicit
# clear, so this only covers a truncated stream.
DEFAULT_CUE_SECONDS = 4.0
MAX_CUE_SECONDS = 30.0
# A repeat of the same picture within this of the last one is the disc
# re-sending a subtitle that never left the screen (see `_fold_repeats`).
REPEAT_GAP = 0.1
# PP-OCR's detector normalises the *short* side to 736px, so a bare
# subtitle strip is blown up out of all proportion: a measured 1646x276 cue
# reaches 4390x736 and the glyphs outgrow what the detector can take in. On
# real Blu-ray cues it then returned one box swallowing both lines, or no
# boxes at all — 6.3% of a film came back blank, every one of them a short
# line. A white margin fixes both and is *faster*, because the short side
# needs less enlarging: measured over 30 cues, 90.2% -> 96.3% per character,
# no blanks left, and 1677ms -> 1047ms each.
DETECTOR_MARGIN = 0.5  # of the picture's own height, on every side
MIN_MARGIN = 24
# cues are reported this often; recognition of a film runs to thousands
PROGRESS_EVERY = 25
# vision engine: a sheet holding more rows than this is asking the model to
# lose count, and losing count voids the whole sheet
MAX_SHEET_ROWS = 40
LABEL_WIDTH = 130  # room for "[1234]" beside each row of a sheet
# The number is downscaled by the server exactly as much as the subtitle
# is, and a model that cannot read the labels fails the coverage check on
# every sheet. PIL's unscaled default font is 11px.
LABEL_SIZE = 28
# how many binarised cues the debug log keeps as PNGs
DEBUG_IMAGES = 30

_LINE_RE = re.compile(r"^\s*[*\-•]?\s*\[?\s*(\d+)\s*\]?\s*[:：.、]?\s*(.*?)\s*$")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$")
_NOTHING = "[无]"


@dataclass
class Cue:
    """One subtitle event, already reduced to black text on white."""

    start: float
    end: float
    image: Image.Image


# ------------------------------------------------------------ binarisation


def _erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def depth_map(opaque: np.ndarray) -> np.ndarray:
    """How far inside the shape each pixel is, in pixels.

    Repeated erosion, counting the rounds each pixel survives — a chessboard
    distance transform, which is all this needs and avoids a scipy import.
    """
    depth = np.zeros(opaque.shape, dtype=np.int32)
    current = opaque
    for _ in range(MAX_DEPTH_ROUNDS):
        if not current.any():
            break
        depth += current
        current = _erode(current)
    return depth


def ink_mask(indices: np.ndarray) -> np.ndarray:
    """Which pixels of an indexed subtitle bitmap are the letters.

    Two shapes of subtitle, told apart by whether the rect has any
    transparency at all:

    * **glyphs on nothing** — the outline wraps the fill, so the fill lies
      deeper inside the opaque shape. Ranking the indices by their mean
      depth separates them without ever knowing a colour. It has to be a
      mean depth rather than "survives N erosions": once the edges are
      anti-aliased the outline is split across several indices, and a fixed
      erosion depth lands right on top of the widest one — measured, that
      let the outline through and closed the counters of every ``e`` and
      ``o``.
    * **a solid plate with the text knocked out** — nothing to measure depth
      against, so the plate is simply the index covering the most ground and
      the letters are everything else.
    """
    if (indices == 0).mean() < OPAQUE_BELOW:
        counts = np.bincount(indices.ravel(), minlength=256)
        return indices != int(counts.argmax())

    opaque = indices != 0
    if not opaque.any():
        return opaque
    depth = depth_map(opaque)
    counts = np.bincount(indices[opaque], minlength=256)
    floor = max(1, int(opaque.sum() * MIN_INDEX_SHARE))
    means = {
        i: float(depth[indices == i].mean())
        for i in range(1, 256) if counts[i] >= floor
    }
    if not means:
        return opaque
    deepest = max(means.values())
    fill = [i for i, m in means.items() if m >= deepest * DEPTH_RATIO]
    ink = np.isin(indices, fill)
    if ink.mean() > INK_CEILING:
        # the "interior" turned out to be a plate sitting on a transparent
        # frame; the letters are the opaque pixels it does not cover
        ink = opaque & ~ink
    if not ink.any():
        # 分不出填充与描边时，整个不透明形状交出去——字形带着描边仍然认得出，
        # 而空掩码会让调用方把这一条当成「清屏事件」整个丢掉。实测一张真 DVD
        # （720x576 法语 VobSub）上，这两件事一起发生过 602 次，占全片一半。
        return opaque
    return ink


def _rect_image(rect) -> Optional[np.ndarray]:
    if not rect.planes:
        return None
    buf = bytes(memoryview(rect.planes[0]))
    if len(buf) < rect.width * rect.height:
        return None
    indices = np.frombuffer(buf, dtype=np.uint8)[: rect.width * rect.height]
    return indices.reshape(rect.height, rect.width)


def _compose(rects: Sequence[tuple[int, int, np.ndarray]], upscale: int) -> Image.Image:
    """Lay every region of one cue back out where it was on screen.

    A disc often puts each line of a two-line subtitle in its own region;
    pasting them at their own coordinates is what keeps them two lines
    instead of one run-on.
    """
    left = min(x for x, _y, _m in rects)
    top = min(y for _x, y, _m in rects)
    right = max(x + m.shape[1] for x, _y, m in rects)
    bottom = max(y + m.shape[0] for _x, y, m in rects)
    canvas = np.zeros((bottom - top, right - left), dtype=bool)
    for x, y, m in rects:
        h, w = m.shape
        canvas[y - top: y - top + h, x - left: x - left + w] |= m
    # a margin: recognisers are trained on text with room around it
    pad = 8
    padded = np.zeros((canvas.shape[0] + pad * 2, canvas.shape[1] + pad * 2), bool)
    padded[pad:-pad, pad:-pad] = canvas
    img = Image.fromarray(np.where(padded, 0, 255).astype(np.uint8), "L")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    return img


def bitmap_cues(
    video_path: str | Path,
    track: dict,
    upscale: int = 2,
    should_cancel: Optional[Callable[[], bool]] = None,
    stats: Optional[dict] = None,
) -> List[Cue]:
    """Decode *track*'s bitmaps into black-on-white images with timings."""
    source = Path(track["path"] or video_path)
    cues: List[Cue] = []
    with av.open(str(source)) as container:
        streams = container.streams.subtitles
        if not streams:
            raise ValueError(f"{source.name} 里没有字幕流")
        stream = next((s for s in streams if s.index == track["index"]), streams[0])
        offset = 0.0 if track["path"] else mux.start_offset(container, source)

        open_cue: Optional[Cue] = None
        open_timed = False      # 这条 cue 自己说得出时长吗
        for packet in container.demux(stream):
            if should_cancel and should_cancel():
                raise InterruptedError
            # the empty packet demux emits at end-of-stream. Testing
            # `dts is None` would throw away real cues — matroska stores no
            # DTS at all (see services/mux.py for the same trap).
            if not packet.size or packet.pts is None:
                continue
            subset = stream.decode2(packet)
            if subset is None:
                continue
            at = max(float(packet.pts * packet.time_base) - offset, 0.0)
            rects = []
            for rect in subset:
                if rect.type != b"bitmap":
                    continue
                indices = _rect_image(rect)
                if indices is None or not indices.size:
                    continue
                rects.append((rect.x, rect.y, ink_mask(indices)))

            if open_cue is not None and at > open_cue.start:
                # PGS ends a cue with its own clear event — an empty
                # composition at the moment the words leave the screen. The
                # packet duration is not to be trusted for this: on a .sup
                # it comes back as 0xFFFFFFFF, i.e. a cue lasting 49 days.
                #
                # VobSub has no clear events at all and writes the real
                # duration in end_display_time instead. When a cue said how
                # long it lasts, the next one may only **cut it short**:
                # stretching it to the next cue leaves a line on screen
                # through every silence — measured on a real DVD, one cue
                # was told to last 1.4s and would have run 2.8s.
                open_cue.end = min(open_cue.end, at) if open_timed else at
                cues.append(open_cue)
                open_cue = None
            if not rects or not any(m.any() for _x, _y, m in rects):
                continue  # the clear itself carries no picture
            end = at + DEFAULT_CUE_SECONDS
            span = float((packet.duration or 0) * packet.time_base)
            if not 0 < span <= MAX_CUE_SECONDS:
                # milliseconds relative to pts whatever the time_base says —
                # the same field subsource.read_cues falls back to. PGS
                # leaves it at 0xFFFFFFFF, which this range check throws out
                # exactly as it throws out the packet duration.
                span = max(subset.end_display_time, 0) / 1000.0
            open_timed = 0 < span <= MAX_CUE_SECONDS
            if open_timed:
                end = at + span
            open_cue = Cue(at, end, _compose(rects, upscale))
        if open_cue is not None:
            cues.append(open_cue)
    cues.sort(key=lambda c: (c.start, c.end))
    folded = _fold_repeats(cues)
    if stats is not None:
        stats.update({"events": len(cues), "cues": len(folded)})
    return folded


def _fold_repeats(cues: List[Cue]) -> List[Cue]:
    """Fold a picture the disc re-sends while it is still on screen.

    Measured on a Blu-ray: 1165 of 2304 consecutive cues were exactly one
    second apart and carried a byte-identical bitmap — 2305 events for
    1088 subtitles. Every repeat was recognised again, proofread again and
    translated again, and the finished file showed each line broken into a
    row of one-second duplicates.

    The rule is subsource's, which was written for karaoke-timed ASS: same
    content still on screen folds, the same content after a real gap does
    not. A character saying a line twice is two subtitles.
    """
    folded: List[Cue] = []
    previous = None
    for cue in cues:
        picture = (cue.image.size, cue.image.tobytes())
        if (folded and picture == previous
                and cue.start <= folded[-1].end + REPEAT_GAP):
            folded[-1].end = max(folded[-1].end, cue.end)
            continue
        folded.append(cue)
        previous = picture
    return folded


# --------------------------------------------------------------- engines


_engine = None
_engine_key: Optional[tuple] = None
_engine_label = ""  # which model answered, for the debug log
_engine_lock = threading.Lock()

# Which recognition model reads which language. The Japanese dictionary
# includes ASCII, so it is also what an unknown track is probed with.
_REC_LANG = {
    "ja": "japan", "zh": "ch", "ko": "korean", "en": "en",
    "ru": "cyrillic", "fr": "latin", "de": "latin", "es": "latin",
    "it": "latin", "pt": "latin",
}
PROBE_CUES = 10

# Which model reads which language, best first — see `_engine_params`.
_REC_MODELS = {
    "japan": (("PP-OCRv6", "small"), ("PP-OCRv4", "mobile")),
    "en": (("PP-OCRv6", "small"), ("PP-OCRv4", "mobile")),
    "ch": (("PP-OCRv6", "small"), ("PP-OCRv4", "mobile")),
    "korean": (("PP-OCRv4", "mobile"),),
    "cyrillic": (("PP-OCRv4", "mobile"),),
    "latin": (("PP-OCRv4", "mobile"),),
}
_REC_FALLBACK = (("PP-OCRv4", "mobile"),)


def rec_language(code: str) -> str:
    return _REC_LANG.get((code or "").lower()[:2], "en")


def _get_engine(lang: str, network: Optional[NetworkSettings], log: Optional[LogFn]):
    """Load RapidOCR once and keep it, the way asr.py keeps whisper.

    Every candidate model is tried in turn: which recognition models a
    release ships moves around, and the library only finds out it cannot
    serve a combination when it goes looking for the file.
    """
    global _engine, _engine_key, _engine_label
    with _engine_lock:
        if _engine is not None and _engine_key == lang:
            return _engine
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(MISSING_DEPENDENCY) from exc

        from app.services.asr import proxy_env

        candidates = _engine_params(lang)
        if not candidates:
            raise RuntimeError(
                f"OCR 模型加载失败：安装的 rapidocr 没有可用于「{lang}」的识别模型，"
                "请升级 OCR 组件，或把引擎改成「视觉大模型」"
            )
        failure: Optional[Exception] = None
        for label, params in candidates:
            if log:
                log(f"加载 OCR 模型（{lang} / {label}）…（仅首次需要下载，约 20–40MB）")
            try:
                # the download goes out over requests, so the model-download
                # proxy switch applies here exactly as it does to whisper
                with proxy_env(network):
                    _engine = RapidOCR(params=params)
            except Exception as exc:  # noqa: BLE001 — on to the next model
                failure = exc
                if log:
                    log(f"  {label} 不可用：{exc}")
                continue
            _engine_key, _engine_label = lang, label
            return _engine
        raise RuntimeError(
            f"OCR 模型加载失败（语言 {lang}）：{failure}\n"
            "若是下载失败，可在设置里开启「模型下载走代理」后重试"
        ) from failure


def _enum_value(name: str, value: str):
    """Look up one of RapidOCR's enum members by the value it stands for.

    These parameters are refused unless they arrive as enum instances —
    passing the string ``"PP-OCRv4"`` fails with *The value of
    Rec.ocr_version must be Enum Type*. Which members exist changes between
    releases, so a missing one drops that candidate rather than failing.
    """
    import rapidocr

    enum = getattr(rapidocr, name, None)
    if enum is None:
        return None
    try:
        return enum(value)
    except ValueError:
        return None


def _engine_params(lang: str) -> List[tuple[str, dict]]:
    """Every RapidOCR configuration worth trying for one language, best first.

    The version and the model size are not free choices: RapidOCR checks the
    whole (version, language, size) triple against its own model list. The
    v6 line is a single multilingual model that covers Latin, Japanese and
    Chinese; v5 has no Japanese model at all; v4 ships one model per
    language and only in the ``mobile`` size.
    """
    from app.services.asr import get_model_cache_dir

    extras = {
        # a subtitle is never upside down, so the orientation classifier has
        # nothing to gain and one way to lose
        "Global.use_cls": False,
    }
    cache = get_model_cache_dir()
    if cache:
        # never the core/cache.py tree — that one is wiped on every startup.
        # `model_root_dir` is the only directory Global has; anything else
        # there is rejected as an unknown key.
        extras["Global.model_root_dir"] = str(Path(cache) / "rapidocr")

    out: List[tuple[str, dict]] = []
    for version, size in _REC_MODELS.get(lang, _REC_FALLBACK):
        ocr_version = _enum_value("OCRVersion", version)
        model_type = _enum_value("ModelType", size)
        if ocr_version is None or model_type is None:
            continue  # this rapidocr has never heard of that combination
        out.append((f"{version} {size}", {
            # the only one still allowed to be a plain string, but the enum
            # is what the library's own examples pass
            "Rec.lang_type": _enum_value("LangRec", lang) or lang,
            "Rec.ocr_version": ocr_version,
            "Rec.model_type": model_type,
            **extras,
        }))
    if out:
        # last resort: everything above is rejected if a single key has been
        # renamed, and a recogniser that keeps its models in the wrong folder
        # still beats a job that fails
        label, params = out[-1]
        out.append((f"{label}（默认设置）",
                    {k: v for k, v in params.items() if not k.startswith("Global.")}))
    return out


def _read_rapidocr(cues: Sequence[Cue], lang: str,
                   network: Optional[NetworkSettings] = None,
                   log: Optional[LogFn] = None,
                   progress: Optional[ProgressFn] = None,
                   should_cancel: Optional[Callable[[], bool]] = None,
                   ) -> List[tuple[str, float]]:
    engine = _get_engine(lang, network, log)
    out: List[tuple[str, float]] = []
    for n, cue in enumerate(cues, start=1):
        if should_cancel and should_cancel():
            raise InterruptedError
        out.append(_one(engine, cue.image))
        if progress and n % PROGRESS_EVERY == 0:
            progress(n / len(cues))
    return out


def reading_order(boxes: Sequence) -> List[int]:
    """Group the detector's boxes into lines, each read left to right.

    Sorting on the top edge in fixed-size bands does not work: within one
    line of Japanese the top of ``こ`` and the top of ``閒`` are tens of
    pixels apart at this scale, so a band narrow enough to separate two
    lines also cuts one line into several, and the fragments then come back
    sorted by height instead of by position. Measured on a Blu-ray, 82 cues
    came out as an exact anagram of themselves — ``ねえ これ知ってる？``
    recognised as ``れ知って てる ねえ ?``.

    So the lines are found first, by clustering the box centres with a
    tolerance taken from the boxes' own height, and only then is each line
    read across.
    """
    edges = [np.asarray(b, dtype=float) for b in boxes]
    mid = [float(b[:, 1].min() + b[:, 1].max()) / 2 for b in edges]
    left = [float(b[:, 0].min()) for b in edges]
    tall = [float(b[:, 1].max() - b[:, 1].min()) for b in edges]
    tol = max(1.0, statistics.median(tall) * 0.5)

    order: List[int] = []
    line: List[int] = []
    anchor = 0.0
    for i in sorted(range(len(edges)), key=lambda i: mid[i]):
        if line and mid[i] - anchor > tol:
            order.extend(sorted(line, key=lambda j: left[j]))
            line = []
        if not line:
            anchor = mid[i]
        line.append(i)
    order.extend(sorted(line, key=lambda j: left[j]))
    return order


def bordered(image: Image.Image) -> Image.Image:
    """The cue with room around it, which is what the detector needs."""
    margin = max(MIN_MARGIN, int(image.height * DETECTOR_MARGIN))
    out = Image.new("L", (image.width + margin * 2, image.height + margin * 2), 255)
    out.paste(image.convert("L"), (margin, margin))
    return out


def _one(engine, image: Image.Image) -> tuple[str, float]:
    """Recognise one cue: every box the detector found, in reading order."""
    result = engine(np.array(bordered(image).convert("RGB")))
    boxes = getattr(result, "boxes", None)
    texts = list(getattr(result, "txts", None) or [])
    scores = list(getattr(result, "scores", None) or [])
    if not texts:
        return "", 0.0
    if boxes is not None and len(boxes) == len(texts):
        order = reading_order(boxes)
        texts = [texts[i] for i in order]
        scores = [scores[i] for i in order] if len(scores) == len(order) else scores
    text = " ".join(t.strip() for t in texts if t and t.strip())
    return text, float(sum(scores) / len(scores)) if scores else 0.0


# ------------------------------------------------------- the vision engine


def _label_font():
    try:
        return ImageFont.load_default(size=LABEL_SIZE)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _sheet(cues: Sequence[Cue], first: int) -> Image.Image:
    """Stack cues into one labelled image, so one call reads many lines."""
    width = LABEL_WIDTH + max(c.image.width for c in cues)
    height = sum(c.image.height + 16 for c in cues)
    sheet = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(sheet)
    font = _label_font()
    y = 0
    for n, cue in enumerate(cues, start=first):
        sheet.paste(cue.image, (LABEL_WIDTH, y))
        draw.text((8, y + cue.image.height // 2 - LABEL_SIZE // 2),
                  f"[{n}]", fill=0, font=font)
        y += cue.image.height + 16
        draw.line((0, y - 8, width, y - 8), fill=160)
    return sheet


def _png_data_url(image: Image.Image) -> str:
    """PNG, never JPEG: ringing around one-bit glyphs is lost accuracy."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_vision_prompt(language_hint: str = "") -> str:
    lang = f"（字幕语言：{language_hint}）" if language_hint else ""
    return "\n".join([
        f"这是一部影片的图形字幕{lang}，每一条字幕单独一行，左侧标着它的编号。",
        "请把每一条里的文字**原样抄写**出来。",
        "要求：",
        "- 只抄写，不翻译、不改写、不补全、不加标点；",
        "- 一条字幕即使原本折成两行，也合并成一行输出；",
        f"- 某一条实在看不出文字时，输出 {_NOTHING}，不要跳过它。",
        "输出格式：每条一行，写 `[编号] 文字`。",
        "必须且只能覆盖图中出现的全部编号，不得新增、遗漏或改写编号。",
        "不要输出解释、代码块标记或任何多余内容。",
    ])


def parse_sheet(reply: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in (reply or "").splitlines():
        line = raw.strip()
        if not line or _FENCE_RE.match(line):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        out[int(m.group(1))] = m.group(2).strip()
    return out


def _read_vision(cues: Sequence[Cue], llm: LLMSettings, settings: OcrSettings,
                 network: Optional[NetworkSettings] = None,
                 language_hint: str = "",
                 log: Optional[LogFn] = None,
                 progress: Optional[ProgressFn] = None,
                 should_cancel: Optional[Callable[[], bool]] = None,
                 client=None,  # injectable for tests
                 ) -> List[tuple[str, float]]:
    """Transcribe the cues with a vision model, a sheet at a time.

    Verified the way every other model answer in this project is: the reply
    must cover exactly the numbers that were on the sheet.

    A sheet that will not verify is halved and tried again. The likeliest
    reason a self-hosted endpoint turns one down is that the picture, or the
    reply it would need to write, is over some limit of its own, and half a
    sheet is under it. Only a single cue failing twice aborts the pass —
    handing back a subtitle with holes in it is the one outcome worse than
    stopping.
    """
    from app.services.translator import make_vision_client

    if client is None:
        client = make_vision_client(llm, network)
    model = llm.vision_model.strip() or llm.model
    system = build_vision_prompt(language_hint)
    size = max(1, min(settings.vision_batch, MAX_SHEET_ROWS))

    texts: dict[int, str] = {}
    queue = deque((start, list(cues[start: start + size]))
                  for start in range(0, len(cues), size))
    while queue:
        if should_cancel and should_cancel():
            raise InterruptedError
        offset, batch = queue.popleft()
        parsed = _read_sheet(client, model, system, batch, offset + 1, log)
        if parsed is None:
            if len(batch) > 1:
                half = len(batch) // 2
                queue.appendleft((offset + half, batch[half:]))
                queue.appendleft((offset, batch[:half]))
                if log:
                    log(f"  拆成 {half} + {len(batch) - half} 条再试")
                continue
            raise RuntimeError(
                f"视觉识别在第 {offset + 1} 条上连续失败，已中止；"
                "可改用本地 OCR 引擎，或换一个视觉模型"
            )
        texts.update(parsed)
        if progress:
            progress(min(len(texts) / len(cues), 1.0))
    return [("" if texts[n] == _NOTHING else texts[n], 1.0)
            for n in range(1, len(cues) + 1)]


def _read_sheet(client, model: str, system: str, batch: Sequence[Cue],
                first: int, log: Optional[LogFn]) -> Optional[dict[int, str]]:
    """One sheet, one retry. None means it could not be verified."""
    from app.services.translator import reply_text

    expected = set(range(first, first + len(batch)))
    image = _sheet(batch, first)
    url = _png_data_url(image)
    where = (f"第 {first}–{first + len(batch) - 1} 条" if len(batch) > 1
             else f"第 {first} 条")
    for attempt in (1, 2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }],
                temperature=0,
            )
            parsed = parse_sheet(reply_text(resp))
        except Exception as exc:  # noqa: BLE001
            if log:
                # the picture's size is half the diagnosis when an endpoint
                # refuses a request outright
                log(f"⚠ 视觉识别{where}失败（第 {attempt} 次）：{exc}"
                    f"（图 {image.width}×{image.height}，请求 {len(url) // 1024}KB）")
            continue
        if set(parsed) == expected:
            return parsed
        if log:
            log(f"⚠ {where}的编号覆盖校验未通过"
                f"（应有 {len(expected)} 条，收到 {len(parsed)} 条）")
    return None


# ------------------------------------------------------------- entry point


def read_cues(
    video_path: str | Path,
    track: dict,
    settings: OcrSettings,
    llm: Optional[LLMSettings] = None,
    network: Optional[NetworkSettings] = None,
    language: str = "",
    log: Optional[LogFn] = None,
    progress: Optional[ProgressFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    debug=None,
    stats: Optional[dict] = None,
    engine=None,  # injectable for tests: (cues, lang) -> [(text, score)]
) -> List[SubtitleLine]:
    """Recognise *track* into subtitle lines, ready for proofreading."""
    began = time.monotonic()
    shape: dict = {}
    cues = bitmap_cues(video_path, track, settings.upscale, should_cancel, shape)
    if not cues:
        raise ValueError("这条图形字幕轨里没有任何画面")
    repeats = shape.get("events", len(cues)) - len(cues)
    if log:
        log(f"图形字幕：{len(cues)} 条"
            + (f"（片源重复发送的 {repeats} 条同图已合并）" if repeats else "")
            + f"，开始 OCR（引擎：{settings.engine}）…")

    lang = language
    if engine is not None:
        results = engine(cues, lang)
    elif settings.engine == "vision":
        if llm is None:
            raise ValueError("视觉大模型 OCR 需要先配置 LLM")
        results = _read_vision(cues, llm, settings, network, lang,
                               log, progress, should_cancel)
    else:
        if not lang:
            lang, note = _probe_language(cues, network, log, should_cancel)
        else:
            note = "已知字幕语言"
        if log:
            log(f"OCR 识别语言：{rec_language(lang)}（{note}）")
        results = _read_rapidocr(cues, rec_language(lang), network,
                                 log, progress, should_cancel)

    lines: List[SubtitleLine] = []
    raw: List[tuple[float, str, float]] = []
    blank = 0
    for cue, (text, score) in zip(cues, results):
        raw.append((cue.start, text, score))
        if not has_content(text):
            blank += 1
            continue
        lines.append(SubtitleLine(index=len(lines) + 1, start=cue.start,
                                  end=cue.end, text=text))
    if not lines:
        raise ValueError("OCR 没有识别出任何文字；请检查这条轨是否真的是字幕，"
                         "或改用语音识别")

    took = time.monotonic() - began
    if stats is not None:
        stats.update({"cues": len(cues), "blank": blank, "seconds": took,
                      "language": lang, "repeats": repeats})
    if log:
        log(f"OCR 完成：{len(lines)} 条"
            + (f"，{blank} 条无文字已丢弃" if blank else "")
            + f"，耗时 {took:.0f}s")
    _write_debug(debug, track, settings, lang, cues, raw, took, repeats)
    return lines


def _probe_language(cues: Sequence[Cue], network, log, should_cancel) -> tuple[str, str]:
    """Read a few cues with the Japanese model, then look at what came out.

    Its dictionary covers ASCII too, so it is legible enough on a Latin
    subtitle for the script test to tell them apart.
    """
    from app.services.subsource import detect_language

    sample = _read_rapidocr(cues[:PROBE_CUES], "japan", network,
                            should_cancel=should_cancel)
    probe = [SubtitleLine(index=i, start=0.0, end=1.0, text=t)
             for i, (t, _s) in enumerate(sample, start=1) if t]
    found = detect_language(probe) if probe else ""
    return (found or "en"), ("试识别前几条后判定" if found else "无法判定，按英文处理")


def _write_debug(debug, track, settings: OcrSettings, lang: str,
                 cues: Sequence[Cue], raw: Sequence[tuple[float, str, float]],
                 took: float, repeats: int = 0) -> None:
    """Record what the engine saw and what it said.

    The images matter as much as the text: without them there is no telling
    a binarisation that went wrong from a recogniser that misread a clean
    picture, and those have opposite fixes.
    """
    if debug is None or not getattr(debug, "enabled", False):
        return
    from app.services.subsource import describe_track

    debug.section("图形字幕 OCR")
    debug.kv("选用", describe_track(track))
    debug.kv("引擎", settings.engine + (f"（{_engine_label}）" if _engine_label else ""))
    debug.kv("识别语言", f"{lang or '未知'} → {rec_language(lang)}")
    debug.kv("放大倍数", f"{settings.upscale}x")
    debug.kv("条数 / 耗时", f"{len(cues)} 条 / {took:.0f}s")
    if repeats:
        debug.kv("片源重复发送", f"{repeats} 条同图已合并（原 {len(cues) + repeats} 个事件）")

    folder = None
    if debug.path is not None:
        folder = Path(debug.path).with_suffix(".ocr")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            for n, cue in enumerate(cues[:DEBUG_IMAGES], start=1):
                cue.image.save(folder / f"cue{n:04d}.png")
            debug.kv("二值化图片", f"前 {min(len(cues), DEBUG_IMAGES)} 条 → {folder}")
        except OSError as exc:
            debug.kv("二值化图片", f"（写入失败: {exc}）")

    debug.line("\n每条的原始识别结果（校对之前）：")
    debug.lines(
        f"[{n:4d}] {start:8.2f} 置信度={score:.2f} | {text}"
        for n, (start, text, score) in enumerate(raw, start=1)
    )
