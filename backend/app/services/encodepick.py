"""按画面自动选编码（EncodeOptions.auto_pick；设置 → 视频压制里可开，默认关）。

开压前从片子里取 6 帧拼成一张图交给视觉模型。**模型只回答画面是什么**——动画
还是真人实拍、颗粒/噪点有多重；编码、速度、质量、内容类型由程序按实测对照出来
（CLAUDE.md 的画质实测表，压制页下方写的是同一份建议）。不让模型直接报 CRF：
数字是量出来的，模型报的数字无从核对；而「这是动画」「颗粒很重」对着拼图一眼就能
核实，判断和理由都写进日志。

    动画，干净或颗粒轻      → AV1（SVT-AV1）最慢 CRF 25     实测 99.55 分，体积不到 H.265 的一半
    动画，颗粒或噪点重      → H.265 较慢 CRF 18             SVT-AV1 会把颗粒当噪声抹掉
    真人，干净或颗粒轻      → H.265 较慢 CRF 18             实测 95.3–97.0 分
    真人，颗粒重            → H.265 较慢 CRF 18 + 保留胶片颗粒   最接近原片，体积约翻倍

**拼图要看得出颗粒**：整帧缩到 640 宽，颗粒就没了。所以上两行是 6 个时间点的整帧
缩略图（看内容），最下一行是其中细节最多的 3 帧正中的一块、1:1 原始像素（看颗粒）。
时间点避开片头片尾（片长的 12%–82%），那里常是黑场、片商标志和滚动字幕。

任何失败（没配能看图的模型、接口报错、回答不合格式、选出的编码器本机没有）都退回
表单里填的参数，并写进日志——和 lyrics / vet 同一条规矩：辅助判断做不成，不能让
压制本身失败。
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import av
import numpy as np
from PIL import Image, ImageDraw

from app.models.schemas import AppSettings, EncodeOptions, LLMSettings, NetworkSettings

POINTS = (0.12, 0.26, 0.40, 0.54, 0.68, 0.82)   # 取帧的位置（片长的比例）
TILE = (640, 360)                                # 拼图每格的大小
CROPS = 3                                        # 最下一行：几块 1:1 原始像素
MAX_DECODE = 600                                 # 一次跳转后最多解多少帧去找目标时间
PQ, HLG = 16, 18

# (内容, 颗粒重不重) → 覆盖进 EncodeOptions 的几项。数字来自实测，不来自模型
PICKS = {
    ("animation", False): {"video_codec": "libsvtav1", "preset": "veryslow", "quality": 25,
                           "tune": ""},
    ("animation", True): {"video_codec": "libx265", "preset": "slow", "quality": 18, "tune": ""},
    ("live", False): {"video_codec": "libx265", "preset": "slow", "quality": 18, "tune": ""},
    ("live", True): {"video_codec": "libx265", "preset": "slow", "quality": 18, "tune": "grain"},
}
CONTENT = {"animation": "动画", "live": "真人实拍"}
GRAIN = {"none": "画面干净", "light": "颗粒轻", "heavy": "颗粒重"}
# 模型偶尔换个说法；认几个明摆着的同义词，别的一律算读不出
_CONTENT_WORDS = {"animation": "animation", "anime": "animation", "cartoon": "animation",
                  "cg": "animation", "live": "live", "live-action": "live",
                  "live_action": "live", "liveaction": "live", "film": "live"}
_GRAIN_WORDS = {"none": "none", "clean": "none", "light": "light", "low": "light",
                "heavy": "heavy", "high": "heavy"}


class PickError(RuntimeError):
    pass


@dataclass
class Verdict:
    content: str      # animation | live
    grain: str        # none | light | heavy
    reason: str

    def describe(self) -> str:
        return f"{CONTENT[self.content]}，{GRAIN[self.grain]}"


# ------------------------------------------------------------------ 拼图


@dataclass
class _Shot:
    overview: Image.Image     # 整帧，按显示比例缩进一格
    crop: Image.Image         # 正中一块，1:1 原始像素
    detail: float             # 那一块有多少细节：挑最下一行用


def _detail(crop: Image.Image) -> float:
    g = np.asarray(crop.convert("L"), dtype=np.float32)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def _shot(image: Image.Image, sar: float) -> _Shot:
    w, h = image.size
    tw, th = TILE
    display_w = w * sar
    scale = min(tw / display_w, th / h)
    size = (max(1, round(display_w * scale)), max(1, round(h * scale)))
    overview = image.resize(size, Image.LANCZOS)
    cw, ch = min(tw, w), min(th, h)
    left, top = (w - cw) // 2, (h - ch) // 2
    crop = image.crop((left, top, left + cw, top + ch))
    return _Shot(overview, crop, _detail(crop))


def grab(path: str | Path) -> Tuple[List[_Shot], dict]:
    """片长 12%–82% 之间的 6 帧，外加提示词要知道的事（隔行、HDR）。"""
    from app.services.audio import picture_stream

    shots: List[_Shot] = []
    facts = {"interlaced": False, "hdr": False, "times": []}
    with av.open(str(path)) as container:
        stream = picture_stream(container)
        if stream is None:
            raise PickError("片源里没有画面")
        stream.thread_type = "AUTO"
        start = container.start_time / av.time_base if container.start_time else 0.0
        duration = container.duration / av.time_base if container.duration else 0.0
        if duration <= 0:
            raise PickError("读不出片长")
        sar = float(stream.sample_aspect_ratio or 1) or 1.0
        facts["hdr"] = getattr(stream.codec_context, "color_trc", 0) in (PQ, HLG)
        for point in POINTS:
            target = start + duration * point
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            frame = None
            for decoded, got in zip(range(MAX_DECODE), container.decode(stream)):
                frame = got
                if got.time is None or got.time >= target:
                    break
            if frame is None:
                continue
            facts["interlaced"] |= bool(getattr(frame, "interlaced_frame", False))
            facts["times"].append(round((frame.time or target) - start, 2))
            shots.append(_shot(frame.to_image(), sar))
    if len(shots) < 3:
        raise PickError(f"只取到 {len(shots)} 帧画面")
    return shots, facts


def contact_sheet(shots: List[_Shot]) -> Image.Image:
    """3 列 × 3 行：上两行整帧缩略图，最下一行细节最多的几帧的 1:1 原始像素。"""
    tw, th = TILE
    sheet = Image.new("RGB", (tw * 3, th * 3), (24, 24, 24))
    for i, shot in enumerate(shots[:6]):
        x, y = (i % 3) * tw, (i // 3) * th
        ow, oh = shot.overview.size
        sheet.paste(shot.overview, (x + (tw - ow) // 2, y + (th - oh) // 2))
    detailed = sorted(sorted(range(len(shots)), key=lambda i: -shots[i].detail)[:CROPS])
    for j, i in enumerate(detailed):
        crop = shots[i].crop
        sheet.paste(crop, (j * tw + (tw - crop.width) // 2, 2 * th + (th - crop.height) // 2))
    draw = ImageDraw.Draw(sheet)
    for k in (1, 2):   # 格子之间的分隔线，别让模型把两格读成一张图
        draw.line([(k * tw, 0), (k * tw, 3 * th)], fill=(128, 128, 128), width=2)
        draw.line([(0, k * th), (3 * tw, k * th)], fill=(128, 128, 128), width=2)
    return sheet


def jpeg(sheet: Image.Image) -> bytes:
    """JPEG 质量 95、不做色度抽样：颗粒还在，一张图 1 MB 上下。PNG 装满颗粒的
    1920×1080 要 4 MB 往上，有的中转会拒收。"""
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue()


# ------------------------------------------------------------------ 问模型


def build_prompt(facts: dict) -> str:
    notes = []
    if facts.get("interlaced"):
        notes.append("片源是隔行扫描的，截图没有反交错：横向的梳齿纹是隔行造成的，不是颗粒或噪点。")
    if facts.get("hdr"):
        notes.append("片源是 HDR，截图没有做色调映射：颜色发灰、发淡是正常的，不要据此判断。")
    return (
        "这是一部视频的截图拼图，用来决定怎么压制它。拼图是 3×3 格：\n"
        "- 上面两行：片中 6 个时间点的整帧，缩小过，用来看内容；\n"
        "- 最下面一行：其中 3 帧正中的一块，1:1 原始像素、没有缩放，用来看颗粒和噪点。\n"
        + "".join(f"注意：{n}\n" for n in notes)
        + "请判断两件事：\n"
        "1. content：animation（2D 动画、卡通、3D/CG 动画）或 live（真人实拍的电影、电视剧、纪录片）；"
        "两种都有时按占多数的算。\n"
        "2. grain：看最下面一行的原始像素——none（干净平滑，几乎看不到颗粒）、"
        "light（有细微的颗粒或噪点）、heavy（明显的胶片颗粒或噪点，是画面质感的一部分）。\n"
        '只输出一行 JSON，不要别的：{"content": "animation 或 live", '
        '"grain": "none、light 或 heavy", "reason": "一句中文，说你看到了什么"}'
    )


def parse(reply: str) -> Optional[Verdict]:
    match = re.search(r"\{.*\}", reply or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    content = _CONTENT_WORDS.get(str(data.get("content", "")).strip().lower())
    grain = _GRAIN_WORDS.get(str(data.get("grain", "")).strip().lower())
    if not content or not grain:
        return None
    return Verdict(content, grain, str(data.get("reason", "")).strip()[:200])


def ask(image: bytes, facts: dict, llm: LLMSettings,
        network: Optional[NetworkSettings] = None, client=None) -> Verdict:
    from app.services.translator import make_vision_client, reply_text

    client = client or make_vision_client(llm, network)
    model = llm.vision_model.strip() or llm.model
    url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": build_prompt(facts)},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]
    reply = ""
    for _ in range(2):
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
        reply = reply_text(resp)
        verdict = parse(reply)
        if verdict is not None:
            return verdict
        messages += [{"role": "assistant", "content": reply or ""},
                     {"role": "user", "content": "只输出一行 JSON：content 取 animation 或 live，"
                                                 "grain 取 none、light 或 heavy。"}]
    raise PickError(f"视觉模型的回答读不出判断：{(reply or '（空）')[:120]}")


# ------------------------------------------------------------------ 对照


def apply(verdict: Verdict, options: EncodeOptions) -> EncodeOptions:
    """容器、音频、分辨率、反交错、保留哪些轨照旧，只换画面的几项。"""
    choice = PICKS[(verdict.content, verdict.grain == "heavy")]
    return options.model_copy(update={**choice, "rate_control": "quality",
                                      "auto_pick": False})


def analyze(path: str | Path, settings: AppSettings,
            client=None) -> Tuple[Verdict, bytes]:
    shots, facts = grab(path)
    image = jpeg(contact_sheet(shots))
    return ask(image, facts, settings.llm, settings.network, client=client), image


def decide(path: str | Path, options: EncodeOptions, settings: AppSettings, *,
           log: Callable[[str], None], client=None) -> EncodeOptions:
    """这次压制该用的参数：判断出来的那套，判断不了就是 *options* 本身。"""
    from app.services import encode

    fallback = options.model_copy(update={"auto_pick": False})
    try:
        verdict, _ = analyze(path, settings, client=client)
        picked = apply(verdict, options)
        encode.check_options(picked)
    except Exception as exc:  # noqa: BLE001 — any failure falls back, and says so
        log(f"⚠ 按画面自动选编码没做成（{exc}），用表单里的参数："
            f"{encode.describe_options(fallback)}")
        return fallback
    log(f"按画面自动选编码：{verdict.describe()}"
        + (f"（{verdict.reason}）" if verdict.reason else "")
        + f" → {encode.describe_options(picked)}")
    return picked
