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
outline wraps around the fill, so eroding the non-transparent mask and
asking which indices survive separates them by geometry rather than by
colour: measured on generated discs, the fill scores 1.00 and the outline
0.29–0.33. Anything above 0.5 is the glyph.
"""

from __future__ import annotations

import base64
import io
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import av
import numpy as np
from PIL import Image, ImageDraw

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
# Measured on generated discs, the fill sits around 6.1 and the outline
# around 3.3 whether or not the edges are anti-aliased, so 0.6 lands in the
# gap with room on both sides.
DEPTH_RATIO = 0.6
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
# cues are reported this often; recognition of a film runs to thousands
PROGRESS_EVERY = 25
# vision engine: a sheet holding more rows than this is asking the model to
# lose count, and losing count voids the whole sheet
MAX_SHEET_ROWS = 40
LABEL_WIDTH = 90  # room for "[123]" beside each row of a sheet
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
                open_cue.end = at
                cues.append(open_cue)
                open_cue = None
            if not rects or not any(m.any() for _x, _y, m in rects):
                continue  # the clear itself carries no picture
            end = at + DEFAULT_CUE_SECONDS
            span = float((packet.duration or 0) * packet.time_base)
            if 0 < span <= MAX_CUE_SECONDS:
                end = at + span
            open_cue = Cue(at, end, _compose(rects, upscale))
        if open_cue is not None:
            cues.append(open_cue)
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


# --------------------------------------------------------------- engines


_engine = None
_engine_key: Optional[tuple] = None
_engine_lock = threading.Lock()

# Which recognition model reads which language. The Japanese dictionary
# includes ASCII, so it is also what an unknown track is probed with.
_REC_LANG = {
    "ja": "japan", "zh": "ch", "ko": "korean", "en": "en",
    "ru": "cyrillic", "fr": "latin", "de": "latin", "es": "latin",
    "it": "latin", "pt": "latin",
}
PROBE_CUES = 10


def rec_language(code: str) -> str:
    return _REC_LANG.get((code or "").lower()[:2], "en")


def _get_engine(lang: str, network: Optional[NetworkSettings], log: Optional[LogFn]):
    """Load RapidOCR once and keep it, the way asr.py keeps whisper."""
    global _engine, _engine_key
    with _engine_lock:
        if _engine is not None and _engine_key == lang:
            return _engine
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(MISSING_DEPENDENCY) from exc

        from app.services.asr import proxy_env

        if log:
            log(f"加载 OCR 模型（{lang}）…（仅首次需要下载，约 10–20MB）")
        params = _engine_params(lang)
        try:
            # the download goes out over requests, so the model-download
            # proxy switch applies here exactly as it does to whisper
            with proxy_env(network):
                _engine = RapidOCR(params=params)
        except Exception as exc:  # noqa: BLE001 — the wording is the point
            raise RuntimeError(
                f"OCR 模型加载失败（语言 {lang}）：{exc}\n"
                "若是下载失败，可在设置里开启「模型下载走代理」后重试"
            ) from exc
        _engine_key = lang
        return _engine


def _engine_params(lang: str) -> dict:
    """RapidOCR configuration for one language.

    Pinned to PP-OCRv4: the v5 line is reported to be missing the Japanese
    recognition model, and Japanese is half of what this feature is for.
    """
    from app.services.asr import get_model_cache_dir

    params = {"Rec.lang_type": lang, "Rec.ocr_version": "PP-OCRv4"}
    cache = get_model_cache_dir()
    if cache:
        # never the core/cache.py tree — that one is wiped on every startup
        params["Global.model_dir"] = str(Path(cache) / "rapidocr")
    return params


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


def _one(engine, image: Image.Image) -> tuple[str, float]:
    """Recognise one cue: every box the detector found, in reading order."""
    result = engine(np.array(image.convert("RGB")))
    boxes = getattr(result, "boxes", None)
    texts = list(getattr(result, "txts", None) or [])
    scores = list(getattr(result, "scores", None) or [])
    if not texts:
        return "", 0.0
    if boxes is not None and len(boxes) == len(texts):
        # top to bottom, then left to right — the order a person reads
        order = sorted(
            range(len(texts)),
            key=lambda i: (round(float(np.min(np.asarray(boxes[i])[:, 1])) / 12),
                           float(np.min(np.asarray(boxes[i])[:, 0]))),
        )
        texts = [texts[i] for i in order]
        scores = [scores[i] for i in order] if len(scores) == len(order) else scores
    text = " ".join(t.strip() for t in texts if t and t.strip())
    return text, float(sum(scores) / len(scores)) if scores else 0.0


# ------------------------------------------------------- the vision engine


def _sheet(cues: Sequence[Cue], first: int) -> Image.Image:
    """Stack cues into one labelled image, so one call reads many lines."""
    width = LABEL_WIDTH + max(c.image.width for c in cues)
    height = sum(c.image.height + 16 for c in cues)
    sheet = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(sheet)
    y = 0
    for n, cue in enumerate(cues, start=first):
        sheet.paste(cue.image, (LABEL_WIDTH, y))
        draw.text((8, y + cue.image.height // 2 - 6), f"[{n}]", fill=0)
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
    must cover exactly the numbers that were on the sheet. A sheet that
    cannot be verified after one retry aborts the whole pass rather than
    handing back a subtitle with holes in it.
    """
    from app.services.translator import make_openai_client

    if client is None:
        client = make_openai_client(llm, network)
    model = llm.vision_model.strip() or llm.model
    system = build_vision_prompt(language_hint)
    size = max(1, min(settings.vision_batch, MAX_SHEET_ROWS))
    out: List[tuple[str, float]] = []

    for start in range(0, len(cues), size):
        if should_cancel and should_cancel():
            raise InterruptedError
        batch = cues[start: start + size]
        expected = list(range(start + 1, start + 1 + len(batch)))
        url = _png_data_url(_sheet(batch, expected[0]))
        parsed: dict[int, str] = {}
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
                parsed = parse_sheet(resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                if log:
                    log(f"⚠ 视觉识别第 {expected[0]}–{expected[-1]} 条失败"
                        f"（第 {attempt} 次）：{exc}")
                parsed = {}
            if set(parsed) == set(expected):
                break
            if log and parsed:
                log(f"⚠ 第 {expected[0]}–{expected[-1]} 条的编号覆盖校验未通过"
                    f"（应有 {len(expected)} 条，收到 {len(parsed)} 条）")
            parsed = {}
        if not parsed:
            raise RuntimeError(
                f"视觉识别在第 {expected[0]}–{expected[-1]} 条上连续失败，已中止；"
                "可改用本地 OCR 引擎，或换一个视觉模型"
            )
        for n in expected:
            text = parsed[n]
            out.append(("" if text == _NOTHING else text, 1.0))
        if progress:
            progress(min((start + len(batch)) / len(cues), 1.0))
    return out


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
    cues = bitmap_cues(video_path, track, settings.upscale, should_cancel)
    if not cues:
        raise ValueError("这条图形字幕轨里没有任何画面")
    if log:
        log(f"图形字幕：{len(cues)} 条，开始 OCR（引擎：{settings.engine}）…")

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
                      "language": lang})
    if log:
        log(f"OCR 完成：{len(lines)} 条"
            + (f"，{blank} 条无文字已丢弃" if blank else "")
            + f"，耗时 {took:.0f}s")
    _write_debug(debug, track, settings, lang, cues, raw, took)
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
                 took: float) -> None:
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
    debug.kv("引擎", settings.engine)
    debug.kv("识别语言", f"{lang or '未知'} → {rec_language(lang)}")
    debug.kv("放大倍数", f"{settings.upscale}x")
    debug.kv("条数 / 耗时", f"{len(cues)} 条 / {took:.0f}s")

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
