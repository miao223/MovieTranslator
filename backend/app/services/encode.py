"""压制：把一个视频重新编码成另一个（「压制」页，和原盘页的全流程）。

The engine under the queue's "encode" entries. It sits on the same loop as
`mux.embed` (mux.pump), so every rule that loop learned the hard way —
copied streams drop only the empty end-of-stream packet, transcoded ones
must be fed it; timestamps shifted by the smallest DTS; the encoder's time
base on the codec context — holds here without being written twice.

What this adds is everything a re-encode needs and a subtitle embed never
did: a filter graph (deinterlace, scale, pixel format), rate control,
per-track audio decisions, chapters, and a verification pass whose result
decides whether the pipeline's lossless intermediate may be replaced.

Measured facts that shaped it (PyAV 18 / FFmpeg 8.1, see CLAUDE.md):

- a filter graph must be built by hand: `Graph.add_buffer` pins the pixel
  aspect to 1:1 and gives no frame rate, which decimate refuses;
- frames leave bwdif with a halved time base and doubled pts; PyAV's
  encode() rescales into the encoder's time base, so VFR stays exact;
- bwdif `deint=interlaced` touches only frames flagged interlaced —
  progressive frames come out byte-identical;
- x265 refuses `tune=film` at open; unknown `x265-params` keys are
  swallowed silently, so nothing free-form is ever passed through;
- AV1 and VP9 lose the sample aspect ratio in MKV — anamorphic sources are
  scaled to square pixels for them;
- a decoder can throw on one bad DTS packet in a whole film: decode errors
  are counted and skipped, and fail the job only when they are not rare.

Two tags make the program recognise its own files (read_tags):
REMUX_TAG is written by the disc remux, ENCODE_TAG by this module. The
second one is what lets a re-run after a crash see that its work is
already done, and the first is what licenses replacing a file at all —
only a lossless MKV this program remuxed may be swapped for its encode.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import av
from av.sidedata.sidedata import Type as SideType
from av.video.format import VideoFormat
from av.video.reformatter import VideoReformatter

from app.models.schemas import EncodeOptions, EncodeRequest
from app.services import audio, mux, restore

# SVT-AV1 writes a banner of a dozen lines to stderr for every encoder it
# opens; errors are all the server console needs from it.
os.environ.setdefault("SVT_LOG", "1")

LogFn = Callable[[str], None]

ENCODE_TAG = "MOVIETRANSLATOR_ENCODE"
REMUX_TAG = "MOVIETRANSLATOR_REMUX"

# the name an encode gets next to its source: 片名.HEVC.mkv
FAMILY_TAGS = {"H.264": "H264", "H.265": "HEVC", "AV1": "AV1", "VP9": "VP9"}
ENCODER_NAMES = {"libx264": "x264", "libx265": "x265", "libsvtav1": "SVT-AV1",
                 "libvpx-vp9": "libvpx"}
VENDOR_NAMES = {"nvenc": "NVENC", "qsv": "QSV", "amf": "AMF"}

# content tunings each encoder accepts. x265 has no "film" — it fails to
# open with it — and nothing else takes these names at all.
TUNES = {"libx264": ("film", "animation", "grain"), "libx265": ("animation", "grain")}

AUDIO_NAMES = {"copy": "原样", "aac": "AAC", "libopus": "Opus", "ac3": "AC-3",
               "eac3": "E-AC-3", "flac": "FLAC"}
# decoded streams call the Opus codec "opus"; the encoder is libopus
DECODED_AS = {"libopus": "opus"}
# kbit/s by channel count when the user leaves it to us. E-AC-3 and AC-3
# top out at 5.1 here, so their 7.1 sources arrive as six channels.
AUTO_KBPS = {
    "eac3": {1: 128, 2: 224, 6: 640},
    "ac3": {1: 128, 2: 224, 6: 640},
    "aac": {1: 128, 2: 192, 6: 384, 8: 512},
    "libopus": {1: 96, 2: 128, 6: 256, 8: 320},
}
AC3_MAX_KBPS = 640

# MP4 cannot carry these as they are
MP4_REFUSES_AUDIO = ("truehd", "mlp")
# the AC-3 core our remux keeps next to a TrueHD track (disc/remux.py)
TRUEHD_CORE_TITLE = "AC-3 兼容音轨"

# mkvmerge's statistics tags describe the bitstream they were written for:
# on a re-encoded track they are simply wrong (BPS-eng, NUMBER_OF_BYTES-eng…)
_STAT_TAGS = {"BPS", "NUMBER_OF_FRAMES", "NUMBER_OF_BYTES", "DURATION",
              "_STATISTICS_WRITING_APP", "_STATISTICS_WRITING_DATE_UTC",
              "_STATISTICS_TAGS", "ENCODER"}
# a track title naming the codec it had ("DTS-HD MA 7.1", "Dolby TrueHD
# Atmos") would lie about the one it has now
_CODEC_WORDS = re.compile(
    r"truehd|dts|flac|pcm|lossless|无损|atmos|dolby|\bac-?3\b|\baac\b|\bopus\b",
    re.IGNORECASE)

# The floor below which an encode stops rather than fill the disk: past it
# even queue.json cannot be saved any more.
MIN_FREE_BYTES = 1 << 30
SPACE_CHECK_SECONDS = 30
# decode errors tolerated before the source counts as broken (per stream)
BAD_PACKET_SHARE = 0.001
BAD_PACKET_FLOOR = 50

PQ, HLG = 16, 18  # AVColorTransferCharacteristic


class EncodeError(RuntimeError):
    """A source or a choice this engine will not encode, with the reason."""


# ------------------------------------------------------------- encoders


def _family(codec: str) -> str:
    for name, family, _vendor in mux.ENCODERS:
        if name == codec:
            return family
    return ""


def _hardware(codec: str) -> bool:
    return codec.rsplit("_", 1)[-1] in VENDOR_NAMES


def encoder_name(codec: str) -> str:
    """x265, SVT-AV1, NVENC… — the part of a label that says which engine."""
    if codec in ENCODER_NAMES:
        return ENCODER_NAMES[codec]
    return VENDOR_NAMES.get(codec.rsplit("_", 1)[-1], codec)


def name_tag(codec: str) -> str:
    """What goes into the name of an encode written next to its source."""
    if codec == mux.COPY:
        return "remux"
    return FAMILY_TAGS.get(_family(codec), codec)


def quality_scale(codec: str) -> Tuple[int, int, int, str]:
    """(lowest, highest, recommended, what the encoder calls it)."""
    if codec == "libx264":
        return 0, 51, 20, "CRF"
    if codec == "libx265":
        return 0, 51, 22, "CRF"
    if codec == "libsvtav1":
        return 0, 63, 30, "CRF"
    if codec == "libvpx-vp9":
        return 0, 63, 32, "CRF"
    if codec == "av1_amf":
        return 0, 255, 100, "QP"     # AV1 qindex scale
    if codec.endswith("_amf"):
        return 0, 51, 24, "QP"
    if codec == "av1_nvenc":
        return 0, 63, 32, "CQ"
    if codec.endswith("_nvenc"):
        return 0, 51, 24, "CQ"
    if codec.endswith("_qsv"):
        return 1, 51, 24, "ICQ"
    return 0, 51, 23, "CRF"


def _opens(codec: str, pix_fmt: str) -> bool:
    try:
        ctx = av.codec.context.CodecContext.create(codec, "w")
        # NVENC refuses frames below about 129x33 — see mux._open_encoder
        ctx.width, ctx.height = 256, 144
        ctx.pix_fmt = pix_fmt
        ctx.time_base = Fraction(1, 25)
        ctx.open()
    except Exception:  # noqa: BLE001 — any failure means "not like this"
        return False
    return True


_ten_bit: Dict[str, Optional[str]] = {}


def ten_bit_format(codec: str) -> Optional[str]:
    """The 10-bit pixel format *codec* really takes here, or None.

    A hardware encoder's own list is not the answer: h264_nvenc lists
    p010le, and most cards refuse it at open. So those are opened for real,
    once per process, like mux.available_encoders does.
    """
    if codec not in _ten_bit:
        try:
            formats = [f.name for f in (av.codec.Codec(codec, "w").video_formats or [])]
        except Exception:  # noqa: BLE001
            formats = []
        found = next((f for f in ("yuv420p10le", "p010le") if f in formats), None)
        if found and _hardware(codec) and not _opens(codec, found):
            found = None
        _ten_bit[codec] = found
    return _ten_bit[codec]


def capabilities() -> List[dict]:
    """mux.available_encoders(), plus what the 压制 form needs to know
    about each: 10-bit, tunings, and the quality scale."""
    found = []
    for enc in mux.available_encoders():
        low, high, default, label = quality_scale(enc["id"])
        found.append({
            **enc,
            "engine": encoder_name(enc["id"]),
            "ten_bit": ten_bit_format(enc["id"]) is not None,
            "tunes": list(TUNES.get(enc["id"], ())),
            "quality": {"min": low, "max": high, "default": default, "name": label},
        })
    return found


def wants_ten_bit(opts: EncodeOptions) -> bool:
    if opts.bit_depth == "auto":
        return _family(opts.video_codec) in ("H.265", "AV1")
    return opts.bit_depth == "10"


DEINTERLACE_NAMES = {"auto": "自动反交错", "off": "不反交错", "all": "全部反交错",
                     "ivtc": "反胶片过带", "bob": "还原 60 帧", "match": "只做场匹配",
                     "detect": "按片源节奏自动选反交错"}


def check_options(opts: EncodeOptions) -> None:
    """Refuse, at enqueue time, what this machine cannot encode.

    Raises ValueError with the reason. Run again when the job starts —
    the same queue entry can outlive a GPU driver.
    """
    if opts.video_codec == mux.COPY:
        if restore.restoring(opts):
            raise ValueError("还原 60 帧、裁边、降噪、放大都要重编码画面，不能和「保持原样」一起用")
        return
    if opts.upscale and opts.max_height:
        raise ValueError("「放大到」和「分辨率上限」不能同时设置")
    if opts.ai_model and not opts.upscale:
        raise ValueError("AI 放大要先选「放大到」多少")
    usable = {enc["id"] for enc in mux.available_encoders()}
    if opts.video_codec not in usable:
        raise ValueError(f"本机无法使用编码器 {opts.video_codec}。"
                         f"可用的有：{'、'.join(sorted(usable)) or '（无）'}")
    if opts.bit_depth == "10" and ten_bit_format(opts.video_codec) is None:
        raise ValueError(f"{mux.encoder_label(opts.video_codec)} 在本机不支持 10bit，"
                         f"请改选 8bit 或「自动」")
    if opts.tune and opts.tune not in TUNES.get(opts.video_codec, ()):
        raise ValueError(f"{mux.encoder_label(opts.video_codec)} 没有「{opts.tune}」这种内容类型")


def describe_options(opts: EncodeOptions) -> str:
    """One line for the queue and the log: 'H.265 10bit（x265）· CRF 22 · medium · …'."""
    parts = []
    if opts.video_codec == mux.COPY:
        parts.append("按画面自动选编码（判断不了时画面原样）" if opts.auto_pick else "画面原样")
    else:
        depth = "10bit" if wants_ten_bit(opts) else "8bit"
        family = _family(opts.video_codec) or opts.video_codec
        picture = [f"{family} {depth}（{encoder_name(opts.video_codec)}）"]
        if opts.rate_control == "bitrate":
            picture.append(f"{opts.bitrate_kbps} kbps")
        else:
            picture.append(f"{quality_scale(opts.video_codec)[3]} {opts.quality}")
        picture.append(opts.preset)
        if opts.tune:
            picture.append(f"tune {opts.tune}")
        if opts.auto_pick:
            # decided when the encode starts (encodepick); these are the fallback
            parts.append(f"按画面自动选编码（判断不了时 {' · '.join(picture)}）")
        else:
            parts.extend(picture)
        if opts.max_height:
            parts.append(f"≤{opts.max_height}p")
        if opts.deinterlace != "auto":
            parts.append(DEINTERLACE_NAMES[opts.deinterlace])
        if opts.field_order != "auto":
            parts.append({"tff": "上场优先", "bff": "下场优先"}[opts.field_order])
        if opts.crop_auto:
            parts.append("自动裁黑边")
        if restore.cropped(opts):
            parts.append(f"裁边 上{opts.crop_top} 下{opts.crop_bottom} "
                         f"左{opts.crop_left} 右{opts.crop_right}")
        if opts.aspect != "auto":
            parts.append(f"画面 {opts.aspect}")
        if opts.denoise != "off":
            parts.append(restore.DENOISE_NAMES[opts.denoise])
        if opts.upscale:
            model = restore.AI_MODELS.get(opts.ai_model, "")
            parts.append(f"{'用 ' + model + ' ' if model else ''}放大到 {opts.upscale}p")
    if opts.audio_codec == "copy":
        parts.append("音频原样")
    else:
        scope = "无损音轨" if opts.audio_scope == "lossless" else "全部音轨"
        mix = "（立体声）" if opts.audio_mixdown == "stereo" else ""
        parts.append(f"{scope}→{AUDIO_NAMES[opts.audio_codec]}{mix}")
    if opts.audio_languages:
        parts.append("音轨：" + "/".join(opts.audio_languages))
    if opts.subtitles == "none":
        parts.append("不要字幕轨")
    elif opts.subtitles == "languages":
        parts.append("字幕：" + ("/".join(opts.subtitle_languages) or "无"))
    if opts.container == "mp4":
        parts.append("MP4")
    return " · ".join(parts)


# ------------------------------------------------------------------- tags


def options_hash(opts: EncodeOptions) -> str:
    # the 修复 fields are left out while unused: an encode tagged before they
    # existed must still read as this request's finished work
    unused = {k for k, v in restore.RESTORE_DEFAULTS.items() if getattr(opts, k) == v}
    return hashlib.sha1(opts.model_dump_json(exclude=unused).encode("utf-8")).hexdigest()[:12]


def disc_key(root: str | Path) -> str:
    """The id a remux tag names its disc by (the same hash routes uses for
    a disc's series id)."""
    return hashlib.sha1(str(Path(root)).encode("utf-8")).hexdigest()[:12]


def remux_tag(root: str | Path, title_id: str) -> str:
    return json.dumps({"disc": disc_key(root), "title": title_id}, ensure_ascii=False)


def read_tags(path: str | Path) -> dict:
    """{"encode": {...} | None, "remux": {...} | None} for a file of ours.

    Unreadable files, and tags that are not our JSON, read as absent: the
    tags only ever unlock something (skip work, replace a file), so a
    doubtful one must not.
    """
    found: dict = {"encode": None, "remux": None}
    try:
        with av.open(str(path)) as container:
            meta = {str(k).upper(): v for k, v in dict(container.metadata).items()}
    except Exception:  # noqa: BLE001
        return found
    for key, name in ((ENCODE_TAG, "encode"), (REMUX_TAG, "remux")):
        raw = meta.get(key)
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if isinstance(value, dict):
            found[name] = value
    return found


# ------------------------------------------------------------------ audio


def is_lossless(stream) -> bool:
    """TrueHD, DTS-HD MA, FLAC, PCM, ALAC… — the tracks worth re-encoding.

    DTS is one codec either way; only its profile says whether the lossless
    extension is there ('DTS-HD MA', 'DTS-HD MA + DTS:X'). The rest carry a
    flag in their descriptor.
    """
    cc = stream.codec_context
    name = cc.name or ""
    if name in ("dca", "dts"):
        return str(cc.profile or "").startswith("DTS-HD MA")
    try:
        codec = av.codec.Codec(name, "r")
        return bool(codec.lossless) and not bool(codec.lossy)
    except Exception:  # noqa: BLE001
        return name.startswith("pcm_")


def _channels(stream) -> int:
    layout = stream.codec_context.layout
    return layout.nb_channels if layout is not None else 0


def _channel_name(n: int) -> str:
    return {1: "单声道", 2: "立体声", 6: "5.1", 8: "7.1"}.get(n, f"{n} 声道")


def auto_kbps(codec: str, channels: int) -> int:
    table = AUTO_KBPS.get(codec)
    if not table:
        return 0
    fits = [n for n in table if n <= max(channels, 1)]
    return table[max(fits)] if fits else table[min(table)]


def _clean_tags(meta: dict) -> dict:
    return {k: v for k, v in meta.items()
            if str(k).upper().split("-")[0] not in _STAT_TAGS}


def _language(stream) -> str:
    return audio.canon_language(stream.metadata.get("language", "") or "")


def _keeps(stream, wanted: List[str]) -> bool:
    """Whether a language filter keeps *stream*. Untagged always stays."""
    lang = _language(stream)
    if lang in ("", "und"):
        return True
    return lang in {audio.canon_language(w) for w in wanted}


# ------------------------------------------------------------------ probe


@dataclass
class VideoSample:
    """What only decoded frames can tell: taken once, from a third of the
    way in (a film's first minute is logos and black)."""

    frames: int = 0
    interlaced: int = 0
    regular: bool = True          # pts steps uniform (hard telecine, not soft)
    step: float = 0.0             # mean seconds from one frame to the next
    mastering: Optional[bytes] = None
    light: Optional[bytes] = None
    dovi: bool = False
    hdr10plus: bool = False
    # (top-first, bottom-first) votes from the picture (restore.FieldVotes),
    # and over how many frames; only gathered when asked for
    fields: Tuple[int, int] = (0, 0)
    field_frames: int = 0


def _side(frame, kind) -> Optional[bytes]:
    try:
        data = frame.side_data.get(kind)
    except Exception:  # noqa: BLE001
        return None
    return bytes(data) if data is not None else None


def sample_video(path: str | Path, index: int, count: int = 90,
                 fields: bool = False, at: float = 1 / 3) -> VideoSample:
    sample = VideoSample()
    steps: List[int] = []
    votes = restore.FieldVotes() if fields else None
    # one converter for the whole sample: frame.to_ndarray(format=…) makes a
    # new one per frame, with this FFmpeg's swscale threads (24 on a 12-core
    # machine), and a frame whose side data has been read sits in a reference
    # cycle that keeps it — and them — until the garbage collector comes
    # round. Measured: 90 sampled frames, 2,160 threads; on a GPU box capped
    # at 512 tasks the next decoder and x264 then failed to open
    gray = VideoReformatter() if fields else None
    last = None
    tb = None
    try:
        with av.open(str(path)) as container:
            stream = container.streams[index]
            stream.thread_type = "AUTO"
            tb = stream.time_base
            clock = mux.FrameClock.of(stream)
            if container.duration and container.duration / av.time_base > 60:
                container.seek(int(container.duration * at))
            for frame in container.decode(stream):
                sample.frames += 1
                sample.interlaced += bool(frame.interlaced_frame)
                pts = clock(frame)
                if pts is not None:
                    if last is not None:
                        steps.append(pts - last)
                    last = pts
                if sample.mastering is None:
                    sample.mastering = _side(frame, SideType.MASTERING_DISPLAY_METADATA)
                if sample.light is None:
                    sample.light = _side(frame, SideType.CONTENT_LIGHT_LEVEL)
                sample.dovi = sample.dovi or bool(
                    _side(frame, SideType.DOVI_RPU_BUFFER)
                    or _side(frame, SideType.DOVI_METADATA))
                sample.hdr10plus = sample.hdr10plus or bool(
                    _side(frame, SideType.DYNAMIC_HDR_PLUS))
                if votes is not None:
                    votes.add(gray.reformat(frame, format="gray").to_ndarray())
                if sample.frames >= count:
                    break
    except Exception:  # noqa: BLE001 — a sample is advice, not a requirement
        return sample
    if votes is not None:
        sample.fields, sample.field_frames = votes.votes, sample.frames
    steps = [s for s in steps if s > 0]
    if steps:
        # soft telecine decodes as progressive frames whose steps alternate
        # (3003/4504 at 1/90000); decimate would throw real frames away there
        sample.regular = max(steps) - min(steps) <= max(1, min(steps) // 50)
        # the mean, not the median: soft telecine's alternating steps have
        # no middle value that is the rate, their average is
        sample.step = float(sum(steps) / len(steps) * tb) if tb else 0.0
    return sample


# what a measured rate is snapped to, when it is within 1% of one
STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                  Fraction(30), Fraction(48), Fraction(50), Fraction(60000, 1001),
                  Fraction(60), Fraction(120)]


def nominal_rate(stream, sample: Optional["VideoSample"] = None) -> Fraction:
    """The picture's frame rate, as the encoder and the output header state it.

    Measured from decoded frames when there are some: the container's own
    average can be an estimate from a probing window — a remuxed Blu-ray
    extra reads 293/12 (24.42) where its frames step at 24000/1001 — and
    whatever goes in here becomes the new file's stated frame rate.
    """
    if sample is not None and sample.step > 0:
        measured = 1 / sample.step
        best = min(STANDARD_RATES, key=lambda r: abs(float(r) - measured))
        if abs(float(best) - measured) <= measured * 0.01:
            return best
        return Fraction(measured).limit_denominator(1001)
    for rate in (stream.guessed_rate, stream.average_rate):
        if rate and 1 <= float(rate) <= 300:
            return Fraction(rate)
    return Fraction(25)


def _bit_depth(pix_fmt: str) -> int:
    try:
        return VideoFormat(pix_fmt).components[0].bits
    except Exception:  # noqa: BLE001
        return 8


def _fraction_text(value) -> str:
    return f"{value.numerator}:{value.denominator}" if value else ""


def probe(path: str | Path, sample_frames: int = 0) -> dict:
    """What the 压制 page shows about a file, and what a batch lists.

    *sample_frames* > 0 decodes that many frames for the interlace / HDR
    details; the batch scan leaves it at 0 and reads headers only.
    """
    path = Path(path)
    size = path.stat().st_size
    tags = read_tags(path)
    info: dict = {
        "path": str(path), "name": path.name, "size": size,
        "container": "", "duration": 0.0,
        "video": None, "audio": [], "subtitles": [],
        "chapters": 0, "attachments": 0,
        "encoded": tags["encode"] is not None,
        "remuxed": tags["remux"] is not None,
    }
    with av.open(str(path)) as container:
        info["container"] = container.format.name
        info["duration"] = (float(container.duration / av.time_base)
                            if container.duration else 0.0)
        picture = audio.picture_stream(container)
        if picture is not None:
            cc = picture.codec_context
            rate = picture.guessed_rate or picture.average_rate
            trc = getattr(cc, "color_trc", 0)
            info["video"] = {
                "index": picture.index,
                "codec": cc.name,
                "profile": str(cc.profile or ""),
                "width": cc.width, "height": cc.height,
                "fps": round(float(rate), 3) if rate else 0.0,
                "pix_fmt": cc.pix_fmt or "",
                "bit_depth": _bit_depth(cc.pix_fmt) if cc.pix_fmt else 8,
                "sar": _fraction_text(picture.sample_aspect_ratio)
                if picture.sample_aspect_ratio and picture.sample_aspect_ratio != 1 else "",
                "hdr": {PQ: "HDR10", HLG: "HLG"}.get(trc, ""),
                "interlaced": None,
            }
        for stream in container.streams.audio:
            cc = stream.codec_context
            profile = str(cc.profile or "")
            info["audio"].append({
                "index": stream.index,
                "codec": cc.name,
                "profile": profile,
                "channels": _channels(stream),
                "rate": cc.sample_rate or 0,
                "language": _language(stream),
                "language_name": audio.language_name(stream.metadata.get("language", "")),
                "title": (stream.metadata.get("title") or "").strip(),
                "default": bool(stream.disposition & av.stream.Disposition.default),
                "lossless": is_lossless(stream),
                "atmos": "atmos" in profile.lower() or "dts:x" in profile.lower(),
            })
        from app.services.subsource import BITMAP_CODECS
        for stream in container.streams.subtitles:
            info["subtitles"].append({
                "index": stream.index,
                "codec": stream.codec_context.name if stream.codec_context else "",
                "language": _language(stream),
                "language_name": audio.language_name(stream.metadata.get("language", "")),
                "title": (stream.metadata.get("title") or "").strip(),
                "forced": bool(stream.disposition & av.stream.Disposition.forced),
                "bitmap": (stream.codec_context.name if stream.codec_context else "")
                in BITMAP_CODECS,
            })
        info["chapters"] = len(container.chapters())
        info["attachments"] = sum(1 for s in container.streams if s.type == "attachment")
    if sample_frames and info["video"]:
        sample = sample_video(path, info["video"]["index"], sample_frames)
        video = info["video"]
        video["interlaced"] = (round(sample.interlaced / sample.frames, 2)
                               if sample.frames else None)
        if sample.step > 0:
            video["fps"] = round(float(nominal_rate(None, sample)), 3)
        if sample.dovi:
            video["hdr"] = "杜比视界" + ("" if video["hdr"] else "（profile 5）")
        elif sample.hdr10plus:
            video["hdr"] = "HDR10+"
    return info


# ------------------------------------------------------------------- plan


@dataclass
class StreamPlan:
    index: int
    kind: str                  # video / audio / subtitle / attachment / data
    action: str                # encode / copy / drop
    codec: str = ""            # the encoder, for "encode"
    note: str = ""             # why, for the log


def _describe_stream(stream) -> str:
    cc = stream.codec_context
    name = (cc.name if cc is not None else "") or stream.type
    lang = stream.metadata.get("language", "") or ""
    title = (stream.metadata.get("title") or "").strip()
    bits = [f"#{stream.index}", stream.type, name]
    if stream.type == "audio":
        profile = str(cc.profile or "")
        if profile and profile.lower() != name.lower():
            bits.append(profile)
        bits.append(_channel_name(_channels(stream)))
    if lang:
        bits.append(lang)
    if title:
        bits.append(f"「{title}」")
    return " ".join(bits)


def plan_streams(source, opts: EncodeOptions) -> Tuple[List[StreamPlan], List[str]]:
    """Where every stream of *source* goes, and the notes worth logging.

    One plan for the log and for the run, so the lines the log shows are
    the decisions that were made, not a description of them.
    """
    mp4 = opts.container == "mp4"
    real = audio.picture_stream(source)
    notes: List[str] = []
    audios = list(source.streams.audio)
    keep_audio = {s.index for s in audios}
    if opts.audio_languages:
        chosen = {s.index for s in audios if _keeps(s, opts.audio_languages)}
        if chosen:
            keep_audio = chosen
        elif audios:
            notes.append(f"⚠ 按语言（{'/'.join(opts.audio_languages)}）筛完一条音轨都不剩，"
                         f"已保留全部音轨")
    plans: List[StreamPlan] = []
    for stream in source.streams:
        kind = stream.type
        if kind == "video":
            if real is not None and stream.index == real.index:
                if opts.video_codec == mux.COPY:
                    plans.append(StreamPlan(stream.index, kind, "copy", note="画面原样复制"))
                else:
                    plans.append(StreamPlan(stream.index, kind, "encode", opts.video_codec))
            else:
                # a poster (attached_pic) travels as a copy, never as a film
                plans.append(StreamPlan(stream.index, kind, "copy", note="封面图，原样复制"))
        elif kind == "audio":
            name = stream.codec_context.name or ""
            if stream.index not in keep_audio:
                plans.append(StreamPlan(stream.index, kind, "drop", note="不在要保留的语言里"))
                continue
            target = opts.audio_codec
            # 降为立体声 is about what comes out, so it reaches every track
            # with more than two channels, lossy or not
            mix = target != "copy" and opts.audio_mixdown == "stereo" and _channels(stream) > 2
            same = name == DECODED_AS.get(target, target) and not mix
            if target != "copy" and not same and (
                    opts.audio_scope == "all" or mix or is_lossless(stream)):
                plans.append(StreamPlan(stream.index, kind, "encode", target))
            elif mp4 and (name in MP4_REFUSES_AUDIO or name.startswith("pcm_")):
                # MP4 takes the AAC fallback mux.embed uses; TrueHD passes
                # the container check and only fails writing the header
                to = target if target not in ("copy", "flac") else "aac"
                plans.append(StreamPlan(stream.index, kind, "encode", to,
                                        note=f"MP4 装不下 {name}"))
            else:
                plans.append(StreamPlan(stream.index, kind, "copy"))
        elif kind == "subtitle":
            if mp4:
                plans.append(StreamPlan(stream.index, kind, "drop", note="MP4 装不下"))
            elif opts.subtitles == "none":
                plans.append(StreamPlan(stream.index, kind, "drop", note="选了不要字幕轨"))
            elif opts.subtitles == "languages" and not _keeps(stream, opts.subtitle_languages):
                plans.append(StreamPlan(stream.index, kind, "drop", note="不在要保留的语言里"))
            else:
                plans.append(StreamPlan(stream.index, kind, "copy"))
        elif kind == "attachment":
            if mp4:
                plans.append(StreamPlan(stream.index, kind, "drop", note="MP4 装不下"))
            else:
                plans.append(StreamPlan(stream.index, kind, "copy"))
        else:
            plans.append(StreamPlan(stream.index, kind, "drop", note="数据流"))
    # The AC-3 core our remux keeps beside a TrueHD track is a second, lossy
    # copy of the same sound; once the TrueHD itself is re-encoded, keeping
    # the core only doubles it.
    by_index = {p.index: p for p in plans}
    for plan in plans:
        stream = source.streams[plan.index]
        if (plan.kind == "audio" and plan.action == "encode"
                and stream.codec_context.name == "truehd"):
            nxt = by_index.get(plan.index + 1)
            if nxt is None or nxt.kind != "audio" or nxt.action == "drop":
                continue
            core = source.streams[nxt.index]
            if (core.codec_context.name == "ac3"
                    and (core.metadata.get("title") or "").strip() == TRUEHD_CORE_TITLE
                    and _language(core) == _language(stream)):
                nxt.action, nxt.note = "drop", "TrueHD 已转码，它的 AC-3 内核不再需要"
    dropped = sum(1 for p in plans if p.action == "drop" and p.note == "MP4 装不下")
    if dropped:
        notes.append(f"⚠ MP4 装不下片源的 {dropped} 条字幕 / 字体附件，已略过（需要保留请改用 MKV）")
    return plans, notes


# --------------------------------------------------------------- geometry


def _even(value) -> int:
    return max(2, int(round(value / 2)) * 2)


def geometry(width: int, height: int, sar, opts: EncodeOptions) -> Tuple[int, int, Fraction]:
    """Output size and sample aspect ratio: the shorter side capped at
    max_height (never enlarged), sides even, the display aspect unchanged.

    AV1 and VP9 cannot carry a sample aspect ratio into MKV — it reads back
    as 1 — so a non-square source is scaled to square pixels for them
    (720x480 at 32:27 -> 854x480) instead of being shown squashed.

    *width* x *height* is the picture after any crop. With ``upscale`` the
    picture is made square-pixelled first and then enlarged until its
    shorter side is that long (720x480 at 32:27 -> 1920x1080): an enlarging
    model must never see anamorphic pixels, and nothing after it should
    have to stretch them again. A picture already that large is left to
    the rules below — upscale never shrinks.
    """
    sar = Fraction(sar) if sar else Fraction(1)
    if sar <= 0:
        sar = Fraction(1)
    if opts.upscale:
        shown = width * sar
        short = min(shown, height)
        if short < opts.upscale:
            factor = Fraction(opts.upscale) / short
            return _even(shown * factor), _even(height * factor), Fraction(1)
    scale = Fraction(1)
    short = min(width, height)
    if opts.max_height and short > opts.max_height:
        scale = Fraction(opts.max_height, short)
    w, h = _even(width * scale), _even(height * scale)
    out_sar = sar * Fraction(h * width, w * height)
    if (opts.container == "mkv" and _family(opts.video_codec) in ("AV1", "VP9")
            and out_sar != 1):
        w, out_sar = _even(w * out_sar), Fraction(1)
    return w, h, out_sar


def pixel_format(codec: str, opts: EncodeOptions, hdr: bool) -> Tuple[str, str]:
    """(pixel format, note). An HDR source must stay 10-bit."""
    note = ""
    if hdr or wants_ten_bit(opts):
        ten = ten_bit_format(codec)
        if ten:
            return ten, note
        if hdr:
            raise EncodeError(f"片源是 HDR，需要 10bit 编码，而 {mux.encoder_label(codec)} "
                              f"在本机不支持 10bit。请换一个编码器，或选「保持原编码」")
        note = f"⚠ {mux.encoder_label(codec)} 在本机不支持 10bit，改用 8bit"
    try:
        formats = [f.name for f in (av.codec.Codec(codec, "w").video_formats or [])]
    except Exception:  # noqa: BLE001
        formats = []
    for want in ("yuv420p", "nv12"):
        if want in formats:
            return want, note
    return (formats[0] if formats else "yuv420p"), note


# ------------------------------------------------------------- rate control


_VP9_SPEED = {"ultrafast": "5", "fast": "4", "medium": "2", "slow": "1", "veryslow": "0"}


def _mastering_display(raw: bytes, svt: bool) -> str:
    """AVMasteringDisplayMetadata (22 int32) in x265's or SVT-AV1's syntax."""
    v = struct.unpack("<22i", raw[:88])

    def ratio(i: int) -> float:
        return v[i] / v[i + 1] if v[i + 1] else 0.0

    # FFmpeg orders the primaries R, G, B; both encoders want G, B, R
    (rx, ry), (gx, gy), (bx, by) = (ratio(0), ratio(2)), (ratio(4), ratio(6)), (ratio(8), ratio(10))
    wx, wy, lo, hi = ratio(12), ratio(14), ratio(16), ratio(18)
    if svt:
        return (f"G({gx:.4f},{gy:.4f})B({bx:.4f},{by:.4f})R({rx:.4f},{ry:.4f})"
                f"WP({wx:.4f},{wy:.4f})L({hi:.4f},{lo:.4f})")

    def xy(value: float) -> int:  # units of 0.00002
        return round(value * 50000)

    return (f"G({xy(gx)},{xy(gy)})B({xy(bx)},{xy(by)})R({xy(rx)},{xy(ry)})"
            f"WP({xy(wx)},{xy(wy)})L({round(hi * 10000)},{round(lo * 10000)})")


def hdr_params(codec: str, sample: VideoSample, trc: int) -> str:
    """HDR10 static metadata for the encoders that take it as parameters.

    Built only from our own keys — an unknown x265-params key is swallowed
    without a word, so nothing from outside is ever passed through.
    """
    if trc != PQ or codec not in ("libx265", "libsvtav1"):
        return ""
    svt = codec == "libsvtav1"
    parts = ["enable-hdr=1" if svt else "hdr10=1"]
    if sample.mastering and len(sample.mastering) >= 88:
        v = struct.unpack("<22i", sample.mastering[:88])
        if v[20] and v[21]:  # has_primaries, has_luminance
            parts.append(("mastering-display=" if svt else "master-display=")
                         + _mastering_display(sample.mastering, svt))
    if sample.light and len(sample.light) >= 8:
        cll, fall = struct.unpack("<2I", sample.light[:8])
        if cll:
            parts.append(f"{'content-light' if svt else 'max-cll'}={cll},{fall}")
    return ":".join(parts)


def video_options(codec: str, opts: EncodeOptions, extra: str = "") -> Tuple[dict, int]:
    """(codec options, bit_rate) for *codec*, from the abstract choices."""
    low, high, _default, _name = quality_scale(codec)
    q = str(min(max(opts.quality, low), high))
    preset = opts.preset
    options: dict = {}
    bit_rate = 0
    if opts.rate_control == "bitrate":
        bit_rate = opts.bitrate_kbps * 1000
        peaks = {"maxrate": str(bit_rate * 3 // 2), "bufsize": str(bit_rate * 2)}
        if codec in ("libx264", "libx265"):
            options.update(preset=preset, **peaks)
        elif codec == "libsvtav1":
            options["preset"] = mux._SVT_PRESET[preset]
        elif codec == "libvpx-vp9":
            options.update(deadline="good", **{"cpu-used": _VP9_SPEED[preset], "row-mt": "1"})
        elif codec.endswith("_nvenc"):
            options.update(preset=mux._NVENC_PRESET[preset], rc="vbr", **peaks)
        elif codec.endswith("_qsv"):
            options.update(preset=mux._QSV_PRESET[preset], **peaks)
        elif codec.endswith("_amf"):
            options.update(quality=mux._AMF_QUALITY[preset], rc="vbr_peak", **peaks)
    else:
        if codec.endswith("_nvenc"):
            options.update(cq=q, preset=mux._NVENC_PRESET[preset], rc="vbr")
        elif codec.endswith("_qsv"):
            options.update(global_quality=q, preset=mux._QSV_PRESET[preset])
        elif codec.endswith("_amf"):
            options.update(quality=mux._AMF_QUALITY[preset], rc="cqp", qp_i=q, qp_p=q)
            if codec in ("h264_amf", "av1_amf"):
                options["qp_b"] = q
        elif codec == "libsvtav1":
            options.update(crf=q, preset=mux._SVT_PRESET[preset])
        elif codec == "libvpx-vp9":
            # vp9 needs the bitrate pinned to 0 for crf to mean constant quality
            options.update(crf=q, b="0", deadline="good",
                           **{"cpu-used": _VP9_SPEED[preset], "row-mt": "1"})
        else:
            options.update(crf=q, preset=preset)
    if opts.tune and opts.tune in TUNES.get(codec, ()):
        options["tune"] = opts.tune
    if codec == "libx265":
        options["x265-params"] = ":".join(p for p in ("log-level=error", extra) if p)
    elif codec == "libsvtav1" and extra:
        options["svtav1-params"] = extra
    return options, bit_rate


# ------------------------------------------------------------------ pipes


@dataclass
class Stats:
    frames_in: int = 0          # decoded from the source's picture
    frames_out: int = 0         # handed to the encoder (after the filters)
    video_packets: int = 0      # what the encoder gave back
    copied: Dict[int, int] = field(default_factory=dict)   # source index -> packets
    dropped: Dict[int, int] = field(default_factory=dict)  # left out of a copy's opening
    bad: Dict[int, int] = field(default_factory=dict)      # source index -> decode errors
    seen: Dict[int, int] = field(default_factory=dict)     # source index -> packets
    spans: Dict[int, List[float]] = field(default_factory=dict)  # source index -> [start, end]
    started: float = 0.0
    seconds: float = 0.0


class VideoFilter:
    """Deinterlace / scale / convert a stream's frames, their pts intact.

    Built from the first frame (and rebuilt if the picture changes size or
    format mid-stream), by hand: `Graph.add_buffer` fixes the pixel aspect
    at 1:1 and passes no frame rate, which decimate refuses.
    """

    def __init__(self, stream, chain: List[Tuple[str, str]], sar: Fraction,
                 rate: Fraction, ivtc: bool = False):
        self.stream = stream
        self.chain = chain
        self.sar = sar or Fraction(1)
        self.rate = rate
        self.ivtc = ivtc
        self.graph = None
        self.shape = None
        self.first_in: Optional[Fraction] = None
        self.delta: Optional[int] = None

    def _time_base(self, frame):
        # a frame from another filter graph (after bwdif, the time base is
        # halved) states its own; a decoded one has the stream's
        return frame.time_base or self.stream.time_base

    def _build(self, frame) -> None:
        graph = av.filter.Graph()
        src = graph.add(
            "buffer",
            video_size=f"{frame.width}x{frame.height}",
            pix_fmt=str(int(VideoFormat(frame.format.name))),
            time_base=str(self._time_base(frame)),
            pixel_aspect=f"{self.sar.numerator}/{self.sar.denominator}",
            frame_rate=f"{self.rate.numerator}/{self.rate.denominator}",
        )
        nodes = [src] + [graph.add(name, args) for name, args in self.chain]
        sink = graph.add("buffersink")
        graph.link_nodes(*nodes, sink)
        graph.configure()
        self.graph, self.src, self.sink = graph, src, sink
        self.shape = (frame.width, frame.height, frame.format.name)

    def _pull(self) -> list:
        got = []
        while True:
            try:
                got.append(self.sink.pull())
            except av.error.BlockingIOError:   # wants more input
                return got
            except av.error.EOFError:          # flushed
                return got

    def _drain(self) -> list:
        self.src.push(None)
        got = self._pull()
        self.graph = None
        return got

    def _fix(self, frames: list) -> list:
        # decimate emits its first frame one frame late: pull the whole
        # stream back so it starts where the source did
        if not self.ivtc or not frames:
            return frames
        if self.delta is None and self.first_in is not None:
            first = frames[0]
            self.delta = int(first.pts - round(self.first_in / first.time_base))
        if self.delta:
            for frame in frames:
                frame.pts -= self.delta
        return frames

    def push(self, frame) -> list:
        if not self.chain:
            return [frame]
        out = []
        shape = (frame.width, frame.height, frame.format.name)
        if self.graph is not None and shape != self.shape:
            out.extend(self._drain())
        if self.graph is None:
            self._build(frame)
        if self.first_in is None and frame.pts is not None:
            self.first_in = frame.pts * self._time_base(frame)
        self.src.push(frame)
        out.extend(self._pull())
        return self._fix(out)

    def flush(self) -> list:
        if not self.chain or self.graph is None:
            return []
        return self._fix(self._drain())


class _Tolerant:
    """Decode errors are counted, not fatal: a single DTS packet the decoder
    chokes on ('Residual encoded channels are present without core') must
    not throw away hours of encoding. Too many of them fail the job
    (encode_file checks the share)."""

    stats: Stats
    index: int

    def _decode(self, packet) -> list:
        try:
            return packet.decode()
        except av.error.FFmpegError:
            self.stats.bad[self.index] = self.stats.bad.get(self.index, 0) + 1
            return []


class VideoPipe(_Tolerant):
    """decoder → filter graph → [frame stage → second filter graph] → encoder.

    The bracketed part is there only when a model enlarges the picture
    (restore.FrameStage): the first graph then ends in RGB, and the second
    scales the model's output to the encoder's size and pixel format.
    """

    def __init__(self, enc, shift: int, filt: VideoFilter, stats: Stats, index: int,
                 step: int, stage: Optional["restore.FrameStage"] = None,
                 post: Optional[VideoFilter] = None,
                 clock: Optional[mux.FrameClock] = None):
        self.enc, self.shift, self.filter = enc, shift, filt
        self.stats, self.index, self.step = stats, index, step
        self.stage, self.post = stage, post
        self.clock = clock
        self.last: Optional[int] = None

    def _staged(self, frames: list, last: bool = False) -> list:
        if self.stage is None:
            return frames
        out = []
        for frame in frames:
            for made in self.stage.push(frame):
                out.extend(self.post.push(made))
        if last:
            for made in self.stage.flush():
                out.extend(self.post.push(made))
            out.extend(self.post.flush())
        return out

    def wants(self, packet) -> bool:
        return True  # the empty end-of-stream packet flushes the decoder

    def _encode(self, frame) -> list:
        self.stats.frames_out += 1
        made = self.enc.encode(frame)
        self.stats.video_packets += len(made)
        return made

    def feed(self, packet) -> list:
        made = []
        for frame in self._decode(packet):
            self.stats.frames_in += 1
            if self.clock is not None:
                frame.pts = self.clock(frame)
            # PyAV turns a missing pts into a frame counter, which is wrong
            # in any time base but 1/fps
            if frame.pts is None:
                frame.pts = 0 if self.last is None else self.last + self.step
            self.last = frame.pts
            if self.shift:
                frame.pts -= self.shift
            for ready in self._staged(self.filter.push(frame)):
                made.extend(self._encode(ready))
        return made

    def flush(self) -> list:
        made = []
        for ready in self._staged(self.filter.flush(), last=True):
            made.extend(self._encode(ready))
        tail = self.enc.encode(None)
        self.stats.video_packets += len(tail)
        return made + tail


class AudioPipe(_Tolerant, mux.TranscodePipe):
    def __init__(self, enc, shift: int, resampler, stats: Stats, index: int):
        mux.TranscodePipe.__init__(self, enc, shift, resampler)
        self.stats, self.index = stats, index

    def feed(self, packet) -> list:
        made = []
        for frame in self._decode(packet):
            made.extend(mux._encode(self.enc, frame, self.shift, self.resampler))
        return made


class CountingCopy(mux.CopyPipe):
    def __init__(self, target, shift: int, stats: Stats, index: int):
        super().__init__(target, shift)
        self.stats, self.index = stats, index

    def feed(self, packet) -> list:
        return self._count(super().feed(packet))

    def flush(self) -> list:
        return self._count(super().flush())

    def _count(self, made: list) -> list:
        # what reaches the file, which is what verify() compares: the copy may
        # leave out a cut's undecodable opening (mux.CopyPipe)
        self.stats.copied[self.index] = self.stats.copied.get(self.index, 0) + len(made)
        self.stats.dropped[self.index] = self.dropped
        return made


# -------------------------------------------------------------- the encode


@dataclass
class Result:
    """What encode_file did — the input verify() checks the file against."""

    stats: Stats
    plans: List[StreamPlan]
    mapping: Dict[int, int]            # source index -> output index
    chapters: int
    video_index: Optional[int]         # source index of the picture
    video_encoded: bool
    frame_step: float                  # seconds per output frame, for tolerances
    duration: float
    lines: List[str] = field(default_factory=list)
    # the 修复 engine's own account (frames, seconds, fps, peak VRAM), when a
    # model ran: what the 试看 page estimates a whole film from
    engine: Dict[str, float] = field(default_factory=dict)


class LowDiskSpace(EncodeError):
    pass


def _more_fields(path: Path, index: int, sample: VideoSample, duration: float) -> None:
    """A still first window says nothing about field order; look again,
    longer and later, before falling back to flags an analog capture often
    does not have. A short file is sampled from its start both times, so
    the longer look replaces the first rather than counting it twice."""
    more = sample_video(path, index, count=300, fields=True, at=2 / 3)
    if duration > 60:
        sample.fields = (sample.fields[0] + more.fields[0], sample.fields[1] + more.fields[1])
        sample.field_frames += more.field_frames
    elif more.field_frames > sample.field_frames:
        sample.fields, sample.field_frames = more.fields, more.field_frames


def _deinterlace_chain(opts: EncodeOptions, stream, sample: VideoSample,
                       log: LogFn) -> Tuple[List[Tuple[str, str]], Fraction]:
    """(filter chain, how many output frames per source frame) for the
    chosen deinterlace mode: 1, 4/5 for an IVTC, 2 for bob."""
    parity = opts.field_order if opts.field_order != "auto" else "auto"
    bwdif = ("bwdif", f"mode=send_frame:parity={parity}:deint=interlaced")
    if opts.deinterlace == "off":
        return [], Fraction(1)
    if opts.deinterlace == "all":
        return [("bwdif", f"mode=send_frame:parity={parity}:deint=all")], Fraction(1)
    if opts.deinterlace == "ivtc":
        ntsc = abs(float(nominal_rate(stream, sample)) - 30000 / 1001) < 0.01
        if ntsc and sample.regular:
            if sample.frames and sample.interlaced * 2 < sample.frames:
                log("⚠ 抽样里大部分帧没有标记为隔行，反胶片过带仍会按 5 取 4 删帧")
            return [("fieldmatch", f"order={parity}:combmatch=full"), bwdif,
                    ("decimate", "")], Fraction(4, 5)
        why = ("帧率不是 29.97" if not ntsc else
               "时间戳步长不均（软电视电影：画面本来就是逐行的）")
        log(f"⚠ 这个片源不适合反胶片过带（{why}），改为自动反交错")
    if opts.deinterlace == "match":
        # field matching alone: 30p pictures a field out of step come back
        # whole; what it cannot match is left combed and bwdif takes it
        return [("fieldmatch", f"order={parity}:combmatch=full"), bwdif], Fraction(1)
    if opts.deinterlace == "bob":
        rate = float(nominal_rate(stream, sample))
        if rate > restore.BOB_MAX_RATE:
            log(f"⚠ 片源已经是每秒 {rate:.2f} 帧，不需要还原 60 帧，改为自动反交错")
            return [bwdif], Fraction(1)
        measured = restore.field_order(sample.fields)
        tff, bff = sample.fields
        if opts.field_order == "auto":
            if measured:
                parity = measured
                log(f"场序：从画面测得{'上场' if measured == 'tff' else '下场'}优先"
                    f"（抽样 {sample.field_frames} 帧，投票 上场 {tff} : 下场 {bff}）")
            else:
                log(f"⚠ 场序：画面里测不出（抽样 {sample.field_frames} 帧，投票 上场 {tff} : "
                    f"下场 {bff}，多半是画面太静），按片源的标记——没有标记的帧按上场优先处理。"
                    f"成品里动作若一前一后地抖，请在「场序」里改选下场优先重压")
        elif measured and measured != opts.field_order:
            log(f"⚠ 指定的是{'上场' if opts.field_order == 'tff' else '下场'}优先，"
                f"但画面测得的是{'上场' if measured == 'tff' else '下场'}优先"
                f"（投票 上场 {tff} : 下场 {bff}）：动作若前后抖动，请改选另一种")
        if not measured and sample.frames and sample.interlaced * 2 < sample.frames:
            log("⚠ 抽样里画面看不出隔行、多数帧也没有隔行标记：片源可能本来就是逐行的，"
                "还原出的相邻两帧会几乎一样")
        return [("bwdif", f"mode=send_field:parity={parity}:deint=all")], Fraction(2)
    return [bwdif], Fraction(1)


def encode_file(source_path: str | Path, part: str | Path, request: EncodeRequest, *,
                log: Optional[LogFn] = None,
                progress: Optional[Callable[[float, float, float], None]] = None,
                should_cancel: Optional[Callable[[], bool]] = None,
                pace: Optional[Callable[[], None]] = None,
                engine=None) -> Result:
    """Encode *source_path* into *part* as *request* says.

    The file is left at *part* for verify() and the caller's rename;
    anything that goes wrong — cancel included (InterruptedError) — removes
    it. *progress(fraction, fps, speed)* is called on every video packet;
    throttling it is the caller's business. *pace()* is called on every
    packet too, and may sleep: that is how 压制让路 (cpuyield) slows the
    whole encode down, since the codec threads idle once nothing feeds them.
    """
    log = log or (lambda _m: None)
    should_cancel = should_cancel or (lambda: False)
    source_path, part = Path(source_path), Path(part)
    # the pipeline has already done this (the name depends on it); a direct
    # caller gets the same measurement here
    opts = restore.resolve(source_path, request.options, log)
    if part.exists():
        stale = part.stat().st_size
        log(f"⚠ 发现上次未写完的残留文件（{stale / 1e9:.1f} GB），已删除：{part.name}")
        part.unlink(missing_ok=True)
    stats = Stats(started=time.monotonic())
    lines: List[str] = []
    try:
        with av.open(str(source_path)) as source:
            plans, notes = plan_streams(source, opts)
            for note in notes:
                log(note)
            real = audio.picture_stream(source)
            if real is None:
                raise EncodeError("片源里没有画面")
            # read now: a stream's attributes are garbage once the container
            # is closed, and the Result outlives it
            video_index = real.index
            duration = float(source.duration / av.time_base) if source.duration else 0.0
            video_plan = next(p for p in plans if p.index == real.index)
            encoding = video_plan.action == "encode"
            sample = VideoSample()
            if encoding:
                sample = sample_video(source_path, real.index,
                                      fields=opts.deinterlace == "bob")
                if (opts.deinterlace == "bob" and opts.field_order == "auto"
                        and not restore.field_order(sample.fields)):
                    _more_fields(source_path, real.index, sample, duration)
                cc = real.codec_context
                trc = getattr(cc, "color_trc", 0)
                if sample.dovi and trc not in (PQ, HLG):
                    raise EncodeError(
                        "片源是杜比视界 profile 5（没有 HDR10 基础层），重编码后颜色会出错。"
                        "请改用「保持原编码（不重编码）」")
                if sample.dovi:
                    log("⚠ 杜比视界层会丢失，只保留 HDR10 基础层")
                if sample.hdr10plus:
                    log("⚠ HDR10+ 动态元数据会丢失，只保留 HDR10 静态元数据")
                if trc in (PQ, HLG) and _hardware(opts.video_codec):
                    log("⚠ 显卡编码器不接收 HDR10 静态元数据，只保留色彩标记")
            rate = nominal_rate(real, sample)

            fmt = mux.CONTAINERS.get(opts.container, "matroska")
            container_options = ({"movflags": "use_metadata_tags"}
                                 if fmt == "mp4" else None)
            with av.open(str(part), mode="w", format=fmt,
                         container_options=container_options) as out:
                out.metadata.update(dict(source.metadata))
                out.metadata[ENCODE_TAG] = json.dumps({
                    "source": source_path.name,
                    "size": source_path.stat().st_size,
                    "options": options_hash(opts),
                    "version": _version(),
                }, ensure_ascii=False)
                offset = mux.start_offset(source, source_path)
                if offset >= 1.0:
                    log(f"片源时间轴从 {offset:.1f}s 开始，已整体对齐到 0")
                shifts = {s.index: round(offset / float(s.time_base)) if offset else 0
                          for s in source.streams}
                pipes: Dict[int, object] = {}
                mapping: Dict[int, int] = {}
                step_seconds = float(1 / rate)
                for plan in plans:
                    stream = source.streams[plan.index]
                    shift = shifts.get(plan.index, 0)
                    if plan.action == "drop":
                        lines.append(f"{_describe_stream(stream)} → 略去（{plan.note}）")
                        continue
                    if plan.kind == "video" and plan.action == "encode":
                        enc, step_seconds, line = _video_encoder(
                            out, stream, opts, sample, rate, stats, shift, log, pipes, engine)
                        lines.append(f"{_describe_stream(stream)} → {line}")
                        enc.metadata.update(_clean_tags(dict(stream.metadata)))
                        enc.disposition = stream.disposition
                        mapping[plan.index] = enc.index
                        continue
                    if plan.kind == "audio" and plan.action == "encode":
                        enc, line = _audio_encoder(out, stream, plan.codec, opts, stats,
                                                   shift, pipes)
                        lines.append(f"{_describe_stream(stream)} → {line}"
                                     + (f"（{plan.note}）" if plan.note else ""))
                        mapping[plan.index] = enc.index
                        continue
                    try:
                        copy = out.add_stream_from_template(stream)
                    except Exception as exc:  # noqa: BLE001
                        if plan.kind == "audio":
                            enc, line = _audio_encoder(out, stream, "aac", opts, stats,
                                                       shift, pipes)
                            lines.append(f"{_describe_stream(stream)} → {line}"
                                         f"（{opts.container} 装不下原编码）")
                            mapping[plan.index] = enc.index
                            continue
                        if stream.index == real.index:
                            raise EncodeError(
                                f"{opts.container} 容器无法容纳片源的画面编码 "
                                f"{stream.codec_context.name}：{exc}") from exc
                        lines.append(f"{_describe_stream(stream)} → 略去（无法复制：{exc}）")
                        continue
                    copy.metadata.update(dict(stream.metadata))
                    copy.disposition = stream.disposition
                    mapping[plan.index] = copy.index
                    lines.append(f"{_describe_stream(stream)} → 原样复制"
                                 + (f"（{plan.note}）" if plan.note else ""))
                    if plan.kind != "attachment":
                        # an attachment's payload is in the stream header;
                        # its packet is refused by the muxer (mux.embed)
                        pipes[plan.index] = CountingCopy(copy, shift, stats, plan.index)
                chapters = mux.copy_chapters(source, out, offset)
                for line in lines:
                    log(line)
                if chapters:
                    log(f"章节：{chapters} 个，原样带上")

                folder = part.parent
                checked = {"at": time.monotonic()}

                def on_packet(packet, pipe) -> None:
                    if pace is not None:
                        pace()
                    index = packet.stream.index
                    stats.seen[index] = stats.seen.get(index, 0) + 1
                    if packet.pts is not None:
                        tb = packet.time_base
                        start = float((packet.pts - pipe.shift) * tb)
                        end = start + float((packet.duration or 0) * tb)
                        span = stats.spans.get(index)
                        if span is None:
                            stats.spans[index] = [start, end]
                        else:
                            span[0] = min(span[0], start)
                            span[1] = max(span[1], end)
                        if index == video_index and progress and duration:
                            elapsed = max(time.monotonic() - stats.started, 1e-6)
                            done = min(max(start, 0.0) / duration, 1.0)
                            progress(done, stats.frames_out / elapsed, max(start, 0.0) / elapsed)
                    now = time.monotonic()
                    if now - checked["at"] >= SPACE_CHECK_SECONDS:
                        checked["at"] = now
                        _check_space(folder)
                        _check_bad(stats)

                mux.pump(source, out, pipes, should_cancel, on_packet)
                _check_bad(stats)
                engine_stats: Dict[str, float] = {}
                for pipe in pipes.values():
                    stage = getattr(pipe, "stage", None)
                    summary = getattr(stage, "summary", None)
                    if summary is not None and summary():
                        log(summary())
                    engine_stats.update(getattr(stage, "stats", None) or {})
                for index, dropped in stats.dropped.items():
                    if dropped:
                        log(f"流 #{index} 开头是从 GOP 中间切开的：丢掉 {dropped} 个解不出来的前导帧"
                            f"（它们参照的是被切掉的上一段）")
        stats.seconds = time.monotonic() - stats.started
    except restore.EngineError as exc:
        part.unlink(missing_ok=True)
        raise EncodeError(str(exc)) from exc
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        # an engine process must not outlive its encode, cancelled or not
        for pipe in locals().get("pipes", {}).values():
            close = getattr(getattr(pipe, "stage", None), "close", None)
            if close is not None:
                close()
    return Result(stats=stats, plans=plans, mapping=mapping, chapters=chapters,
                  video_index=video_index, video_encoded=encoding,
                  frame_step=step_seconds, duration=duration, lines=lines,
                  engine=engine_stats)


def _version() -> str:
    from app.core.joblog import APP_VERSION
    return APP_VERSION


def _check_space(folder: Path) -> None:
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        return
    if free < MIN_FREE_BYTES:
        raise LowDiskSpace(
            f"输出目录所在的磁盘只剩 {free / (1 << 30):.2f} GB，已停止压制"
            f"（再写下去连列队都存不了）。腾出空间后重试即可")


def _check_bad(stats: Stats) -> None:
    for index, bad in stats.bad.items():
        seen = stats.seen.get(index, 0) or 1
        if bad > BAD_PACKET_FLOOR and bad / seen > BAD_PACKET_SHARE:
            raise EncodeError(f"流 #{index} 有 {bad} 个包解码失败（共 {seen} 个），"
                              f"片源可能已损坏")


def _video_encoder(out, stream, opts: EncodeOptions, sample: VideoSample, rate,
                   stats: Stats, shift: int, log: LogFn, pipes: dict, engine=None):
    """Open the picture's encoder and its pipe; returns (stream, seconds per
    output frame, the log line)."""
    cc = stream.codec_context
    codec = opts.video_codec
    stream.thread_type = "AUTO"  # frame-threaded decoding: +50% on H.264
    trc = getattr(cc, "color_trc", 0)
    hdr = trc in (PQ, HLG)
    pix_fmt, note = pixel_format(codec, opts, hdr)
    if note:
        log(note)
    sar = (restore.aspect_sar(opts.aspect, cc.width, cc.height)
           or stream.sample_aspect_ratio or Fraction(1))
    chain, per_frame = _deinterlace_chain(opts, stream, sample, log)
    ivtc, bob = per_frame == Fraction(4, 5), per_frame == 2
    out_rate = rate * per_frame
    try:
        picture, in_w, in_h = restore.picture_chain(opts, cc.width, cc.height)
    except ValueError as exc:
        raise EncodeError(str(exc)) from exc
    chain.extend(picture)
    width, height, out_sar = geometry(in_w, in_h, sar, opts)
    resized = (width, height) != (in_w, in_h)
    stage = restore.make_stage(opts, log, engine) if opts.upscale and resized else None
    post = None
    if stage is not None:
        # a model must never see stretched pixels: square them first, never
        # by shrinking (720x480 at 32:27 -> 854x480, at 8:9 -> 720x540)
        if sar != 1:
            sq_w, sq_h = ((_even(in_w * sar), in_h) if sar > 1
                          else (in_w, _even(in_h / sar)))
            chain.append(("scale", f"{sq_w}:{sq_h}:flags=lanczos"))
            chain.append(("setsar", "1"))
        # the model works in RGB at the depth it is given; the second graph
        # takes its output to the encoder's size and format
        chain.append(("format", "rgb48le" if _bit_depth(pix_fmt) > 8 else "rgb24"))
        post = VideoFilter(stream, [("scale", f"{width}:{height}:flags=lanczos"),
                                    ("format", pix_fmt)], sar, out_rate)
    elif resized:
        chain.append(("scale", f"{width}:{height}:flags=lanczos"))
    if chain and stage is None:
        chain.append(("format", pix_fmt))
    options, bit_rate = video_options(codec, opts, hdr_params(codec, sample, trc))
    tag = "hvc1" if opts.container == "mp4" and _family(codec) == "H.265" else ""
    enc = mux.open_video_encoder(
        out, codec, stream, width=width, height=height, pix_fmt=pix_fmt,
        rate=out_rate, options=options, sar=out_sar if out_sar != 1 else None,
        bit_rate=bit_rate, codec_tag=tag,
        # a field per frame needs a clock twice as fine: an AVI capture's
        # time base is 1001/30000, one tick per source frame, and the second
        # field of each would land on the first's timestamp
        time_base=stream.time_base / 2 if bob and stream.time_base else None,
        # frame threads: x264 runs ~40% faster than with PyAV's default
        # slice threading; the others ignore it
        thread_type="AUTO" if not _hardware(codec) else "", log=log)
    step = max(1, round(1 / (rate * stream.time_base))) if stream.time_base else 1
    pipes[stream.index] = VideoPipe(
        enc, shift, VideoFilter(stream, chain, sar, rate, ivtc), stats, stream.index, step,
        stage=stage, post=post, clock=mux.FrameClock.of(stream))
    depth = "10bit" if _bit_depth(pix_fmt) > 8 else "8bit"
    low, high, _default, scale_name = quality_scale(codec)
    what = [f"{mux.encoder_label(codec)} {depth}",
            f"{width}x{height}" + (f"（SAR {_fraction_text(out_sar)}）" if out_sar != 1 else ""),
            (f"{opts.bitrate_kbps} kbps" if opts.rate_control == "bitrate"
             else f"{scale_name} {min(max(opts.quality, low), high)}"),
            opts.preset]
    if ivtc:
        what.append(f"反胶片过带 → {float(out_rate):.3f}fps")
    elif bob:
        what.append(f"还原 60 帧（每一场一帧）→ {float(out_rate):.3f}fps")
    elif opts.deinterlace == "match":
        what.append("只做场匹配（30p 错场还原成逐行）")
    elif chain and chain[0][0] == "bwdif":
        what.append("自动反交错（只处理标记为隔行的帧）" if "deint=interlaced" in chain[0][1]
                    else "每一帧都反交错")
    if restore.cropped(opts):
        what.append(f"裁边后 {in_w}x{in_h}")
    if opts.denoise != "off":
        what.append(restore.DENOISE_NAMES[opts.denoise])
    if opts.upscale:
        what.append((f"用 {stage.name} 放大" if stage is not None else "Lanczos 放大")
                    if resized else f"已不小于 {opts.upscale}p，不放大")
    if "tune" in options:
        what.append(f"tune {options['tune']}")
    if hdr:
        what.append({PQ: "HDR10", HLG: "HLG"}[trc]
                    + ("（带母版元数据）" if "master" in options.get("x265-params", "")
                       + options.get("svtav1-params", "") else ""))
    return enc, float(1 / out_rate), "重编码为 " + "，".join(what)


def _audio_encoder(out, stream, codec: str, opts: EncodeOptions, stats: Stats,
                   shift: int, pipes: dict):
    """Open one audio track's encoder and its pipe; returns (stream, log line)."""
    mixdown = opts.audio_mixdown == "stereo"
    lossy = codec != "flac"
    kbps = 0
    if lossy:
        # the channel count it will have, for the automatic rate
        rate_, layout, _fmt = mux.fit_audio(codec, stream, mixdown=mixdown,
                                            max_rate=mux.LOSSY_MAX_RATE)
        channels = av.AudioLayout(layout).nb_channels
        kbps = opts.audio_bitrate_kbps or auto_kbps(codec, channels)
        if codec == "ac3":
            kbps = min(kbps, AC3_MAX_KBPS)
    enc, resampler = mux._add_audio_encoder(
        out, stream, codec, bit_rate=kbps * 1000, mixdown=mixdown,
        max_rate=mux.LOSSY_MAX_RATE if lossy else 0)
    try:
        enc.codec_context.open()  # fail here, before a byte is written
    except Exception as exc:  # noqa: BLE001
        raise EncodeError(f"音轨 #{stream.index} 无法用 {AUDIO_NAMES.get(codec, codec)} "
                          f"编码（{enc.layout.name}，{enc.rate}Hz）：{exc}") from exc
    meta = _clean_tags(dict(stream.metadata))
    title = (meta.get("title") or "").strip()
    new_desc = f"{AUDIO_NAMES.get(codec, codec)} {_channel_name(enc.layout.nb_channels)}"
    if title and _CODEC_WORDS.search(title):
        meta["title"] = new_desc
    enc.metadata.update(meta)
    enc.disposition = stream.disposition
    pipes[stream.index] = AudioPipe(enc, shift, resampler, stats, stream.index)
    before = _channels(stream)
    line = f"转为 {new_desc}" + (f" {kbps}kbps" if kbps else "")
    if enc.layout.nb_channels < before:
        line += f"（{_channel_name(before)} 降为 {_channel_name(enc.layout.nb_channels)}）"
    if enc.rate != stream.codec_context.sample_rate:
        line += f"（{stream.codec_context.sample_rate}Hz → {enc.rate}Hz）"
    profile = str(stream.codec_context.profile or "").lower()
    if "atmos" in profile or "dts:x" in profile:
        line += "（Atmos / DTS:X 的声音对象信息会丢失）"
    return enc, line


# ------------------------------------------------------------------ verify


def verify(part: str | Path, result: Result) -> List[str]:
    """What is wrong with the file just written, as sentences; [] = sound.

    The lossless source may be deleted on the strength of this, so it
    checks the file itself, reopened, not the counters alone.
    """
    problems: List[str] = []
    stats = result.stats
    if result.video_encoded and stats.frames_out != stats.video_packets:
        problems.append(f"送进编码器 {stats.frames_out} 帧，编码器只交回 {stats.video_packets} 个包")
    try:
        with av.open(str(part)) as check:
            present = [s for s in check.streams if s.type != "data"]
            packets: Dict[int, int] = {}
            spans: Dict[int, List[float]] = {}
            for packet in check.demux():
                if not packet.size or packet.stream.type == "data":
                    continue
                index = packet.stream.index
                packets[index] = packets.get(index, 0) + 1
                if packet.pts is None:
                    continue
                start = float(packet.pts * packet.time_base)
                end = start + float((packet.duration or 0) * packet.time_base)
                span = spans.setdefault(index, [start, end])
                span[0], span[1] = min(span[0], start), max(span[1], end)
            tags = {str(k).upper() for k in dict(check.metadata)}
            chapters = len(check.chapters())
    except Exception as exc:  # noqa: BLE001
        return [f"压制出的文件打不开：{exc}"]
    if len(present) != len(result.mapping):
        problems.append(f"应有 {len(result.mapping)} 条流，文件里有 {len(present)} 条")
    if ENCODE_TAG not in tags:
        problems.append("文件里没有压制标记")
    if chapters != result.chapters:
        problems.append(f"应有 {result.chapters} 个章节，文件里有 {chapters} 个")
    by_out = {out: src for src, out in result.mapping.items()}
    for out_index, src_index in by_out.items():
        kind = next((p.kind for p in result.plans if p.index == src_index), "")
        got = packets.get(out_index, 0)
        if src_index == result.video_index and result.video_encoded:
            if got != stats.video_packets:
                problems.append(f"画面应有 {stats.video_packets} 个包，文件里有 {got} 个")
        elif src_index in stats.copied and got != stats.copied[src_index]:
            problems.append(f"流 #{src_index} 复制了 {stats.copied[src_index]} 个包，"
                            f"文件里有 {got} 个")
        want, have = stats.spans.get(src_index), spans.get(out_index)
        if not want or not (kind == "audio" or src_index == result.video_index):
            continue   # subtitles are sparse, a cover image is a single still
        if not have:
            problems.append(f"流 #{src_index} 在文件里一个包都没有")
            continue
        length_in, length_out = want[1] - want[0], have[1] - have[0]
        if kind == "video":
            slack = max(3 * result.frame_step, 0.001 * length_in)
            slack += stats.dropped.get(src_index, 0) * result.frame_step
        else:
            slack = 0.5
        if abs(length_in - length_out) > slack:
            problems.append(f"流 #{src_index} 时长 {length_out:.2f}s，片源是 {length_in:.2f}s")
    return problems


# ------------------------------------------------------------ names, files


def _header_rate(source: Path) -> Optional[float]:
    """The picture's frame rate as the file's header states it, for a name."""
    try:
        with av.open(str(source)) as container:
            picture = audio.picture_stream(container)
            rate = picture and (picture.guessed_rate or picture.average_rate)
            return float(rate) if rate else None
    except Exception:  # noqa: BLE001
        return None


def _base_target(source: Path, request: EncodeRequest) -> Path:
    opts = request.options
    ext = opts.container
    if request.output_mode == "custom" and request.output_dir.strip():
        folder = Path(request.output_dir.strip().strip('"').strip("'")).expanduser()
        return folder / f"{source.stem}.{ext}"
    # 片名.1080p.60fps.HEVC.mkv: a restore says what it did to the picture
    rate = _header_rate(source) if opts.deinterlace == "bob" else None
    tags = restore.name_tags(opts, rate) + [name_tag(opts.video_codec)]
    return source.parent / f"{source.stem}.{'.'.join(tags)}.{ext}"


def output_target(source: Path, request: EncodeRequest) -> Path:
    """Where the encode is written: never over an existing file."""
    return mux.free_path(_base_target(source, request))


def existing_encode(source: Path, request: EncodeRequest) -> Optional[Path]:
    """This request's finished encode, if a previous run already wrote it.

    A replacing run is done once its source carries our encode tag — the
    swap happened, only the queue did not get to record it. A run that
    keeps its source is done when a file where it would have written one
    carries a tag naming this source, at this size, with these options.
    """
    if request.replace_source:
        if source.exists() and read_tags(source)["encode"] is not None:
            return source
        return None
    if not source.exists():
        return None
    size = source.stat().st_size
    base = _base_target(source, request)
    candidate, n = base, 2
    while candidate.exists() and n < 100:
        tag = read_tags(candidate)["encode"]
        if (tag and tag.get("source") == source.name and tag.get("size") == size
                and tag.get("options") == options_hash(request.options)):
            return candidate
        candidate = base.with_name(f"{base.stem}.{n}{base.suffix}")
        n += 1
    return None


def replaceable(source: Path, before: os.stat_result) -> str:
    """Why *source* must NOT be replaced by its encode, or "" when it may.

    Only a lossless MKV this program remuxed (the REMUX tag), not already an
    encode, and untouched since the encode started.
    """
    tags = read_tags(source)
    if tags["remux"] is None:
        return "它不是本程序封装出来的无损 MKV"
    if tags["encode"] is not None:
        return "它已经是压制版"
    try:
        now = source.stat()
    except OSError as exc:
        return f"读不到它（{exc}）"
    if (now.st_size, now.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        return "压制期间它被改动过"
    return ""


def _fsync(path: Path) -> None:
    try:
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass


def _fsync_dir(folder: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def finalize(part: Path, source: Path, target: Path, request: EncodeRequest,
             before: Optional[os.stat_result], retries: int = 5,
             wait: float = 2.0) -> Tuple[Path, str]:
    """Put a verified encode in its place. Returns (where it is, why the
    source was kept when it was meant to be replaced — "" otherwise).

    Replacing is one os.replace onto the lossless file: at every moment one
    complete file has that name. A file held open elsewhere (Windows, a
    media server scanning it) gets a few tries, then both are kept.
    """
    _fsync(part)
    if request.replace_source and before is not None:
        why = replaceable(source, before)
        if not why:
            for attempt in range(retries + 1):
                try:
                    os.replace(part, source)
                    _fsync_dir(source.parent)
                    return source, ""
                except PermissionError as exc:
                    if attempt == retries:
                        why = f"无损版被别的程序占用，换不下来（{exc}）"
                        break
                    time.sleep(wait)
        os.replace(part, target)
        _fsync_dir(target.parent)
        return target, why
    os.replace(part, target)
    _fsync_dir(target.parent)
    return target, ""
