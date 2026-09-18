"""Remux — or re-encode — a video with the finished subtitle as a soft track.

The alternative to writing a .srt next to the film: one new file that plays
with subtitles anywhere, with the track still switchable and the picture
untouched. By default audio and video packets are copied verbatim — a change
of container, not a transcode, so a two-hour film takes the time of a file
copy rather than the time of an encode.

`EmbedSettings` can ask for more than that: a different container, and a
real re-encode of the picture (H.264/H.265/AV1/VP9, on the CPU or on a
hardware encoder). Both default to the copy behaviour, so nothing re-encodes
that was not asked to.

The subtitle is not assembled packet by packet here. It is written as a
normal .srt/.ass first and then opened as an *input container*, so
`add_stream_from_template` copies the codec parameters — including the ASS
`[Script Info]`/`[V4+ Styles]` header, which lives in codecpar.extradata
and cannot be set from Python at all. Everything the file version does —
bilingual layout, wrapping, ♪, {\\an7} frame cues — reaches the embedded
track unchanged, because it is literally the same file.

MKV is still the container to prefer: it accepts essentially any audio/video
codec, which matters when the input is only being copied, and it is the only
common one that can carry styled ASS. MP4 is offered because phones, TVs and
browsers often accept nothing else. It cannot hold ASS or SRT at all, so its
subtitle track is tx3g ("mov_text") — plain text, no styling. PyAV cannot
encode subtitles (SubtitleCodecContext.encode raises NotImplementedError),
so that track is built by hand on top of `add_mux_stream`, which exists to
make a stream with no codec context for pre-encoded data. A tx3g sample is
just a big-endian uint16 length followed by UTF-8, and the MP4 muxer fills
the gaps between cues with empty samples on its own.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import av

from app.models.schemas import EmbedSettings
from app.services import audio

LogFn = Callable[[str], None]
ProgressFn = Callable[[float], None]  # 0..1

# target language -> (filename suffix, ISO 639-2/B tag for the track)
LANGUAGES = {
    "简体中文": ("zh", "chi"),
    "繁體中文": ("zh", "chi"),
    "English": ("en", "eng"),
    "日本語": ("ja", "jpn"),
    "한국어": ("ko", "kor"),
    "Français": ("fr", "fre"),
    "Deutsch": ("de", "ger"),
    "Português": ("pt", "por"),
    "Italiano": ("it", "ita"),
    "ไทย": ("th", "tha"),
    "Tiếng Việt": ("vi", "vie"),
    "العربية": ("ar", "ara"),
    "हिन्दी": ("hi", "hin"),
}
FALLBACK = ("sub", "und")

# stream types worth carrying over. Attachments matter more than they look:
# they are where a release keeps the fonts its subtitles are styled with.
COPIED = ("video", "audio", "subtitle", "attachment")

# Cancellation is checked on every packet — an attribute read against a
# copy that can run for minutes — while progress, which reaches the job log
# and the SSE stream, is throttled.
PROGRESS_EVERY = 200

# container id (as the UI and EmbedSettings spell it) -> libavformat name
CONTAINERS = {"mkv": "matroska", "mp4": "mp4"}
COPY = "copy"

# tx3g timestamps are milliseconds
MOV_TEXT_TB = Fraction(1, 1000)
# what MP4 can carry alongside the picture. Everything else in the source —
# its own SRT/ASS tracks, its attached fonts — has nowhere to go.
MP4_KEEPS = ("video", "audio")

# (encoder id, human family name, vendor). Vendor empty = software.
# Order is the order the UI shows: software first, then hardware.
ENCODERS = (
    ("libx264", "H.264", ""),
    ("libx265", "H.265", ""),
    ("libsvtav1", "AV1", ""),
    ("libvpx-vp9", "VP9", ""),
    ("h264_nvenc", "H.264", "NVIDIA"),
    ("hevc_nvenc", "H.265", "NVIDIA"),
    ("av1_nvenc", "AV1", "NVIDIA"),
    ("h264_qsv", "H.264", "Intel"),
    ("hevc_qsv", "H.265", "Intel"),
    ("av1_qsv", "AV1", "Intel"),
    ("h264_amf", "H.264", "AMD"),
    ("hevc_amf", "H.265", "AMD"),
    ("av1_amf", "AV1", "AMD"),
)

# speed tiers, mapped onto what each encoder family actually calls them.
# SVT-AV1 counts the other way (0 slowest, 13 fastest); NVENC uses p1..p7.
_SVT_PRESET = {
    "ultrafast": "12", "fast": "10", "medium": "8", "slow": "6", "veryslow": "4",
}
_NVENC_PRESET = {
    "ultrafast": "p1", "fast": "p2", "medium": "p4", "slow": "p6", "veryslow": "p7",
}
_QSV_PRESET = {
    "ultrafast": "veryfast", "fast": "fast", "medium": "medium",
    "slow": "slow", "veryslow": "veryslow",
}
_AMF_QUALITY = {
    "ultrafast": "speed", "fast": "speed", "medium": "balanced",
    "slow": "quality", "veryslow": "quality",
}
# x26x and NVENC top out at 51; SVT-AV1 and VP9 go to 63
_CRF_MAX = 51


def language_of(target_language: str) -> tuple[str, str]:
    """(文件名后缀, 轨道 ISO-639-2/B 标签) —— 目标语言那一侧。

    表里没有的语言也接受一个直接写下的语言代码：界面允许用户手输目标语言，
    而 pt / nl 这种写法既是他想要的语言、也正好是这套后缀的形状。认不出来
    才落到 FALLBACK —— 猜一个语言比承认不知道更糟，与 source_language_of
    同一条规矩。
    """
    name = target_language.strip()
    if name in LANGUAGES:
        return LANGUAGES[name]
    code = name.lower()
    if 2 <= len(code) <= 3 and code.isalpha():
        tag = audio.canon_language(code)
        if audio.language_name(tag) != tag:     # 认得出名字才算认得这个代码
            return code, tag
    return FALLBACK


def source_language_of(detected: str) -> tuple[str, str, str]:
    """(文件名后缀, 轨道 ISO-639-2/B 标签, 轨道标题) —— language_of 的镜像。

    纯原文模式（output_mode="original_only"）的产物是片子自己的语言，那不是
    LANGUAGES 里那七个可选目标之一，而是 pipeline 一路传下来的 `detected`。
    它始终是两字母码（whisper 自己的、subsource.iso2 过的轨道标签、用户在源
    语言里选的，或 subsource.detect_language 判的），所以后缀就是它本身，标
    签取它的 ISO 639-2/B 形式。

    空字符串＝什么都没判定出来。此时给 orig / und 而不是猜一个：一条诚实标着
    「未知」的轨道，比一条标着 eng 的日语轨道有用得多。
    """
    code = (detected or "").strip().lower()
    if not code:
        return ("orig", "und", "原文字幕")
    return (code, audio.canon_language(code), f"{audio.language_name(code)}字幕")


def _start_time(container) -> float:
    st = container.start_time
    # AV_NOPTS_VALUE comes back as a huge number rather than None
    return float(st) / av.time_base if st and 0 < st < 1 << 62 else 0.0


def _first_dts(video_path: Path) -> float:
    """The earliest decode timestamp, in seconds — what has to become zero."""
    with av.open(str(video_path)) as probe:
        for packet in probe.demux():
            if packet.dts is not None:
                return float(packet.dts * packet.time_base)
    return 0.0


def start_offset(container, video_path: Path) -> float:
    """Where this container's timeline begins, in seconds.

    A container's timeline need not start at zero — TS captures and edit
    lists routinely start hours in — while the subtitles always do, because
    the transcript comes from audio decoded from the first frame whatever
    its timestamp. Copying the source's timestamps unshifted would slide
    every cue by that much.

    The smallest DTS, not start_time: start_time is a PTS, and with B-frames
    the first DTS is smaller still (often negative). Subtracting the larger
    of the two pushes DTS below zero, which matroska cannot store — it drops
    those packets, and the first of them is the opening keyframe.
    """
    return min(_start_time(container), _first_dts(video_path))


def free_path(candidate: Path) -> Path:
    """*candidate*, or the first free 片名.zh.2.ext / .3 / … beside it.

    本程序**绝不覆盖任何已经存在的文件**。一个名字撞上了，让路的永远是新写的
    那一份：用户目录里的东西——他自己下载的字幕、上一次跑出来的成品、同名的
    别的什么——都不是本程序的，而一次静默覆盖是没有下一次机会的。

    编号插在扩展名之前，所以语言后缀留在名字上（film.zh.srt → film.zh.2.srt）。
    只对**写进片源目录的成品**用它：工作目录里的副本、.part/.tmp 这些自己的
    中间文件，本来就该被重写。
    """
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        numbered = candidate.with_name(f"{candidate.stem}.{n}{candidate.suffix}")
        if not numbered.exists():
            return numbered
        n += 1


def output_path(video: Path, target_language: str, container: str = "mkv",
                suffix: str = "") -> Path:
    """Where the muxed film goes: film.mkv -> film.zh.mkv, same folder.

    *suffix* overrides the one derived from *target_language*: 纯原文按片子自
    己的语言命名（film.ja.mkv），而那个语言不在 LANGUAGES 里——见
    source_language_of.

    Never returns a path that already exists — overwriting the user's video
    library is not a risk worth taking for a naming collision.
    """
    suffix = suffix.strip() or language_of(target_language)[0]
    ext = container if container in CONTAINERS else "mkv"
    return free_path(video.parent / f"{video.stem}.{suffix}.{ext}")


# ------------------------------------------------------------- encoders


def _open_encoder(name: str, options: Optional[dict] = None) -> bool:
    """Whether this machine can actually open *name* as a video encoder."""
    try:
        ctx = av.codec.context.CodecContext.create(name, "w")
        # 256x144, not something tiny: NVENC refuses frames below roughly
        # 129x33, so a 64x64 probe would report a perfectly good card as
        # unusable — and this machine has no GPU to notice that on.
        ctx.width, ctx.height = 256, 144
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, 25)
        if options:
            ctx.options = dict(options)
        ctx.open()
    except Exception:  # noqa: BLE001 — any failure means "cannot use it"
        return False
    return True


def _usable(name: str, hardware: bool) -> bool:
    try:
        av.codec.Codec(name, "w")
    except Exception:  # noqa: BLE001
        return False
    if not hardware:
        # compiled into the PyAV wheel: registered means it will open, and
        # opening libx265/SVT-AV1 for real costs 30-50ms and prints a banner
        return True
    # A hardware encoder registers whether or not the device is there —
    # h264_nvenc is present on a machine with no NVIDIA card at all and only
    # fails at open (~1ms). That open is the whole point of this probe.
    return _open_encoder(name)


_available: Optional[List[dict]] = None


def available_encoders() -> List[dict]:
    """The video encoders this machine can really use, in display order.

    Cached for the life of the process: a GPU does not appear or disappear
    while the server runs, and the answer decides what a dropdown shows.
    """
    global _available
    if _available is None:
        found = []
        for name, family, vendor in ENCODERS:
            if not _usable(name, bool(vendor)):
                continue
            found.append({
                "id": name,
                "label": f"{family}（{vendor} 硬件加速）" if vendor else family,
                "family": family,
                "hardware": bool(vendor),
            })
        _available = found
    return list(_available)


def encoder_label(codec: str) -> str:
    """How to name *codec* in a log line."""
    if codec == COPY:
        return "原编码"
    for enc in available_encoders():
        if enc["id"] == codec:
            return f"{enc['label']}（{codec}）"
    return codec


def _quality_options(name: str, quality: int, preset: str) -> dict:
    """Translate the abstract quality/preset into this encoder's own knobs."""
    if name.endswith("_nvenc"):
        return {
            "cq": str(min(quality, _CRF_MAX)),
            "preset": _NVENC_PRESET[preset],
            "rc": "vbr",
        }
    if name.endswith("_qsv"):
        return {
            "global_quality": str(min(quality, _CRF_MAX)),
            "preset": _QSV_PRESET[preset],
        }
    if name.endswith("_amf"):
        return {"quality": _AMF_QUALITY[preset], "rc": "cqp",
                "qp_i": str(min(quality, _CRF_MAX)),
                "qp_p": str(min(quality, _CRF_MAX))}
    if name == "libsvtav1":
        return {"crf": str(quality), "preset": _SVT_PRESET[preset]}
    if name == "libvpx-vp9":
        # vp9 needs the bitrate pinned to 0 for crf to mean constant quality
        return {"crf": str(quality), "b": "0"}
    return {"crf": str(min(quality, _CRF_MAX)), "preset": preset}


def encoder_options(name: str, quality: int, preset: str) -> dict:
    """Quality/preset spelled the way *name* wants to hear it."""
    return _quality_options(name, quality, preset)


def _pix_fmt(name: str, source_fmt: Optional[str]) -> Optional[str]:
    """A pixel format *name* accepts, staying as close to the source as it can.

    Never guessed: `Codec.video_formats` is the encoder's own list. Writing
    yuv420p unconditionally would silently throw away a 10-bit source, and
    would be wrong anyway for NVENC, whose 10-bit format is p010le rather
    than yuv420p10le.
    """
    try:
        formats = [f.name for f in (av.codec.Codec(name, "w").video_formats or [])]
    except Exception:  # noqa: BLE001
        formats = []
    if not formats or not source_fmt or source_fmt in formats:
        return source_fmt
    deep = any(source_fmt.endswith(s) for s in ("10le", "10be", "12le", "12be", "16le"))
    for want in (["yuv420p10le", "p010le"] if deep else []) + ["yuv420p", "nv12"]:
        if want in formats:
            return want
    return formats[0]


# --------------------------------------------------------------- muxing


def _mov_text_cues(subs_in, sub_in) -> List[Tuple[float, float, str]]:
    """The finished subtitle as plain (start, end, text), ready for tx3g."""
    # imported inside the function on purpose: subsource imports this module
    # for start_offset(), so at import time the dependency runs one way only.
    from app.services import subsource

    ass = (sub_in.codec_context.name or "") in ("ass", "ssa")
    found: List[Tuple[float, float, str]] = []
    for packet in subs_in.demux(sub_in):
        if not packet.size:
            continue
        body = bytes(packet).decode("utf-8", "replace")
        text = "\n".join(
            line.strip()
            for line in subsource.event_text(body, ass=ass).splitlines()
            if line.strip()
        )
        if not text:
            continue
        start = float(packet.pts * packet.time_base) if packet.pts is not None else 0.0
        found.append((start, start + float((packet.duration or 0) * packet.time_base), text))

    merged: List[Tuple[float, float, str]] = []
    for start, end, text in sorted(found):
        if merged and start < merged[-1][1] - 1e-6:
            # A tx3g track shows one sample at a time, so two cues cannot
            # overlap the way a {\an7} frame note overlaps dialogue in ASS.
            # Joining them is what the viewer would have seen anyway; the
            # alternative is dropping one of the two.
            was_start, was_end, was_text = merged[-1]
            merged[-1] = (was_start, max(was_end, end), f"{was_text}\n{text}")
            continue
        merged.append((start, end, text))
    return merged


def _add_video_encoder(out, stream, opts: EmbedSettings, log: Optional[LogFn]):
    cc = stream.codec_context
    rate = stream.average_rate or stream.guessed_rate or Fraction(25, 1)
    enc = out.add_stream(opts.video_codec, rate=rate)
    ctx = enc.codec_context
    ctx.width, ctx.height = cc.width, cc.height
    pix_fmt = _pix_fmt(opts.video_codec, cc.pix_fmt)
    if pix_fmt:
        ctx.pix_fmt = pix_fmt
        if log and pix_fmt != cc.pix_fmt:
            log(f"⚠ {opts.video_codec} 不支持片源的像素格式 {cc.pix_fmt}，"
                f"画面将转成 {pix_fmt}（10bit 降到 8bit 会带来轻微色带）")
    # ---- the single most load-bearing line in the transcode path ----
    # It must be codec_context.time_base, NOT enc.time_base. Assigning the
    # stream's looks like it works — it reads back as what you set — but the
    # encoder never consults it, and open() then derives 1/fps from the frame
    # rate instead. For constant-rate material that is the same number, so
    # every test fixture passes; a variable-rate source (pulldown anime,
    # screen captures, some WEB-DL) gets its timestamps quantised. Measured:
    # source pts 0,10,12,40,41,100… came back 0,0,0,42,42,83…, which drifts
    # the audio and slides every cue.
    ctx.time_base = stream.time_base
    if cc.sample_aspect_ratio:  # assigning None raises inside PyAV
        ctx.sample_aspect_ratio = cc.sample_aspect_ratio
    for attr in ("color_range", "color_primaries", "color_trc", "colorspace"):
        try:  # anamorphic and wide-gamut sources deserve the attempt
            setattr(ctx, attr, getattr(cc, attr))
        except Exception:  # noqa: BLE001
            pass
    ctx.options = encoder_options(opts.video_codec, opts.quality, opts.preset)
    # Open now, before a single packet is written. A hardware encoder builds
    # a stream quite happily on a machine with no such card — h264_nvenc only
    # fails at avcodec_open2 — and without this the failure would land after
    # the header and some audio were already in the .part file.
    try:
        ctx.open()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"本机无法使用编码器 {encoder_label(opts.video_codec)}：{exc}。"
            f"请改用软件编码或「保持原编码（不重编码）」"
        ) from exc
    if ctx.options and log:
        # libavcodec drops options it does not recognise instead of
        # complaining, leaving them here. Silence would mean encoding at some
        # default quality while the UI claimed otherwise.
        log(f"⚠ 编码器 {opts.video_codec} 不认识这些参数，已忽略：{ctx.options}")
    if log:
        log(f"开始重编码：{cc.width}x{cc.height} @{float(rate):.3f}fps → "
            f"{encoder_label(opts.video_codec)}，质量 {opts.quality}／速度 {opts.preset}")
        if getattr(cc, "color_trc", None) in (16, 18):  # PQ / HLG
            log("⚠ 片源是 HDR：重编码只保留色彩标记，HDR10 静态元数据可能丢失、"
                "杜比视界层必然丢失。想原样保留请选「保持原编码（不重编码）」")
    return enc


def _add_audio_encoder(out, stream, codec: str):
    cc = stream.codec_context
    enc = out.add_stream(codec, rate=cc.sample_rate)
    try:
        enc.layout = cc.layout.name
    except Exception:  # noqa: BLE001 — the encoder keeps its own default
        pass
    resampler = av.AudioResampler(
        format=enc.format.name, layout=enc.layout.name, rate=enc.rate
    )
    return enc, resampler


def _encode(enc, frame, shift: int, resampler) -> list:
    if frame.pts is not None and shift:
        frame.pts -= shift
    made = []
    for ready in (resampler.resample(frame) if resampler is not None else (frame,)):
        made.extend(enc.encode(ready))
    return made


def embed(
    video_path: str | Path,
    subtitle_path: str | Path,
    out_path: str | Path,
    target_language: str = "",
    opts: Optional[EmbedSettings] = None,
    log: Optional[LogFn] = None,
    progress: Optional[ProgressFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    track_title: str = "",
    track_language: str = "",
    extra_tracks: Sequence[Tuple[str | Path, str, str]] = (),
) -> Path:
    """Write *video_path* to *out_path* with *subtitle_path* as a track.

    With the default *opts* every stream is copied into an mkv, which is all
    this did before there was anything to choose. Raises InterruptedError
    when cancelled, and anything PyAV raises when the copy or the encode
    fails — the caller is expected to fall back to writing the subtitle file
    rather than losing the job.

    *track_language* is the ISO-639-2/B tag for the subtitle track, already
    worked out by the caller; it overrides the one derived from
    *target_language*. 纯原文的轨道语言是片子自己的语言，不是翻译目标。

    *extra_tracks* 是额外的字幕文件，每个 (路径, 标题, 语言标签) 变成一条独立
    轨道，排在主轨之后——双文件模式的原文就是这么进来的。**default 标志只给
    主轨**：两条 default 等于把「先显示哪一条」交还给播放器，而这个模式的全部
    意思就是译文先出来。
    """
    opts = opts or EmbedSettings()
    video_path, subtitle_path, out_path = (
        Path(video_path), Path(subtitle_path), Path(out_path)
    )
    should_cancel = should_cancel or (lambda: False)
    # written under a temporary name: a half-copied film left in the user's
    # library looks exactly like a real one until they try to play it
    part = out_path.with_name(out_path.name + ".part")
    # A previous run killed outright (SIGKILL, power cut) leaves this behind,
    # and it can be several gigabytes. Opening for write would truncate it
    # anyway; it is removed explicitly so the log can say it was there,
    # because otherwise nothing ever tells anyone. The file is named exactly
    # after the output this run is about to produce, so nothing else can be
    # caught by it — a stale .part for a film that is never translated again
    # is still left alone, deliberately: guessing at other people's
    # half-written files is not this program's business.
    if part.exists():
        stale = part.stat().st_size
        if log:
            log(f"⚠ 发现上次未写完的残留文件（{stale / 1_000_000_000:.1f} GB），已删除："
                f"{part.name}")
        part.unlink(missing_ok=True)
    lang = track_language.strip() or language_of(target_language)[1]
    # 主轨 + 附加轨，一视同仁地往下走。附加轨的语言标签认不出来时给 und 而不是
    # 目标语言：它按定义就不是译文，挂上译文的标签是 source_language_of 一直在
    # 防的那种谎。
    wanted = [(Path(subtitle_path), track_title, lang)] + [
        (Path(path), title, language.strip() or FALLBACK[1])
        for path, title, language in extra_tracks
    ]
    fmt = CONTAINERS.get(opts.container, "matroska")
    mp4 = fmt == "mp4"

    try:
        with ExitStack() as stack:
            source = stack.enter_context(av.open(str(video_path)))
            subs_in = [stack.enter_context(av.open(str(path)))
                       for path, _, _ in wanted]
            out = stack.enter_context(av.open(str(part), mode="w", format=fmt))
            for container, (path, _, _) in zip(subs_in, wanted):
                if not container.streams.subtitles:
                    raise ValueError(f"字幕文件无法解析: {path.name}")

            out.metadata.update(dict(source.metadata))
            # Releases routinely carry a poster as a second video stream
            # flagged attached_pic. Re-encoding that as if it were the film
            # is both pointless and wrong, so only the real picture is
            # transcoded and the cover travels as a copy like any other.
            real = audio.picture_stream(source)
            picture = real.index if real is not None else None
            copies: Dict[int, object] = {}       # source index -> copied stream
            encoders: Dict[int, object] = {}     # source index -> encoding stream
            resamplers: Dict[int, object] = {}   # source index -> AudioResampler
            dropped = 0
            for stream in source.streams:
                if stream.type not in COPIED:
                    continue
                if mp4 and stream.type not in MP4_KEEPS:
                    # MP4 has nowhere to put the release's own SRT/ASS tracks
                    # or its attached fonts. Saying so beats a silent loss.
                    dropped += 1
                    continue
                if (stream.type == "video" and opts.video_codec != COPY
                        and stream.index == picture):
                    encoders[stream.index] = _add_video_encoder(
                        out, stream, opts, log)
                    continue
                if stream.type == "audio" and opts.audio_codec != COPY:
                    enc, resampler = _add_audio_encoder(
                        out, stream, opts.audio_codec)
                    enc.metadata.update(dict(stream.metadata))
                    encoders[stream.index] = enc
                    resamplers[stream.index] = resampler
                    continue
                try:
                    copy = out.add_stream_from_template(stream)
                except Exception as exc:  # noqa: BLE001
                    if stream.type == "audio":
                        # The container cannot hold this codec — MP4 has no
                        # room for DTS or TrueHD, and PyAV's ffmpeg has no
                        # encoder for either, so copying was never going to
                        # work. Re-encoding one audio track is a far smaller
                        # loss than throwing away the hour of transcription
                        # and translation standing behind this step.
                        enc, resampler = _add_audio_encoder(out, stream, "aac")
                        enc.metadata.update(dict(stream.metadata))
                        encoders[stream.index] = enc
                        resamplers[stream.index] = resampler
                        if log:
                            log(f"⚠ {opts.container} 无法容纳音轨 #{stream.index} 的"
                                f"{stream.codec_context.name} 编码，已转为 AAC")
                        continue
                    if stream.type == "video":
                        # Never skip these. PyAV asks libavformat at normal
                        # compliance, which turns down a few old codecs MKV
                        # can technically carry (msmpeg4v2, wmv, vc1, adpcm).
                        # Skipping produced a file with no picture at all —
                        # far worse than falling back to a subtitle file.
                        raise ValueError(
                            f"{opts.container} 容器无法容纳片源的{stream.type}编码 "
                            f"{stream.codec_context.name}: {exc}"
                        ) from exc
                    # a font or a data stream is a different matter
                    if log:
                        log(f"⚠ 跳过无法复制的流 #{stream.index}（{stream.type}）：{exc}")
                    continue
                # add_stream_from_template copies codec parameters but not
                # metadata, and dropping it would strip every audio track's
                # language tag — the thing multi-track releases are picked by
                copy.metadata.update(dict(stream.metadata))
                copy.disposition = stream.disposition
                if stream.type == "subtitle":
                    # the release's own subtitle may be flagged default; ours
                    # is about to be, and two defaults means the player picks
                    copy.disposition &= ~av.stream.Disposition.default
                if stream.type != "attachment":
                    # An attachment's payload travels in the stream header,
                    # already copied above. It also emits a packet, which the
                    # muxer refuses — so it is copied but never fed.
                    copies[stream.index] = copy
            if dropped and log:
                log(f"⚠ mp4 无法容纳片源自带的 {dropped} 条字幕/附件流，已略过"
                    f"（需要保留请改用 mkv）")

            sub_streams = []
            for container, (_path, title, language) in zip(subs_in, wanted):
                sub_in = container.streams.subtitles[0]
                if mp4:
                    # PyAV cannot encode subtitles at all, so the tx3g track is
                    # built by hand: add_mux_stream makes a stream with no codec
                    # context, for packets that are already in their final form.
                    subtitle_stream = out.add_mux_stream("mov_text")
                    subtitle_stream.time_base = MOV_TEXT_TB
                else:
                    subtitle_stream = out.add_stream_from_template(sub_in)
                subtitle_stream.metadata["language"] = language
                if title:
                    subtitle_stream.metadata["title"] = title
                # cleared explicitly: add_stream_from_template brings the
                # template's disposition along, and matroska treats a missing
                # FlagDefault as set
                subtitle_stream.disposition &= ~av.stream.Disposition.default
                sub_streams.append(subtitle_stream)
            # the point of the whole exercise is that it shows up on its own —
            # 只给主轨，两条 default 等于把「先显示哪一条」交还给播放器
            sub_streams[0].disposition = av.stream.Disposition.default

            # a film's worth of cues is a few hundred KB; reading them up
            # front is what lets them be merged into the copy in time order.
            # 两条轨也是同一张表：每条 cue 自己带着要去哪条轨，所以下面的游标
            # 仍然只有一个。
            cues: List[Tuple[float, object, object]] = []
            for container, subtitle_stream in zip(subs_in, sub_streams):
                sub_in = container.streams.subtitles[0]
                if mp4:
                    for start, end, text in _mov_text_cues(container, sub_in):
                        raw = text.encode("utf-8")
                        # a tx3g sample is a big-endian uint16 length, then UTF-8
                        cue = av.Packet(len(raw).to_bytes(2, "big") + raw)
                        cue.stream = subtitle_stream
                        cue.time_base = MOV_TEXT_TB
                        cue.pts = cue.dts = int(round(start * 1000))
                        cue.duration = max(1, int(round((end - start) * 1000)))
                        cues.append((start, cue, subtitle_stream))
                else:
                    for cue in container.demux(sub_in):
                        if not cue.size:
                            continue
                        cues.append(
                            (float(cue.pts * cue.time_base), cue, subtitle_stream))
            cues.sort(key=lambda c: c[0])

            duration = (
                float(source.duration / av.time_base) if source.duration else 0.0
            )
            offset = start_offset(source, video_path)
            if log and offset >= 1.0:
                log(f"片源时间轴从 {offset:.1f}s 开始，已整体对齐到 0 以匹配字幕")
            shifts = {
                s.index: round(offset / float(s.time_base)) if offset else 0
                for s in source.streams
            }
            seen = 0
            next_cue = 0
            for packet in source.demux():
                if should_cancel():
                    raise InterruptedError
                index = packet.stream.index
                enc = encoders.get(index)
                target = copies.get(index)
                if enc is None and target is None:
                    continue
                if enc is None and not packet.size:
                    # Only the empty packet demux emits at end-of-stream may
                    # be dropped, and only for a stream being copied — see
                    # the encode branch below for why. Testing `dts is None`
                    # instead — the idiom in every remux example — silently
                    # threw away real frames: matroska stores no DTS at all,
                    # so its demuxer hands back content with dts unset, and
                    # the very first such packet is the opening keyframe.
                    # Measured on a real mkv: 24 of 1104 frames gone, the
                    # file starting with an undecodable GOP.
                    continue

                shift = shifts.get(index, 0)
                if packet.pts is not None:
                    at = float((packet.pts - shift) * packet.time_base)
                else:
                    at = 0.0
                seen += 1
                if seen % PROGRESS_EVERY == 0 and progress and duration \
                        and packet.pts is not None:
                    progress(min(at / duration, 1.0))
                while next_cue < len(cues) and cues[next_cue][0] <= at:
                    _, cue, cue_stream = cues[next_cue]
                    cue.stream = cue_stream
                    out.mux(cue)
                    next_cue += 1

                if enc is not None:
                    # The empty end-of-stream packet is deliberately NOT
                    # filtered out here: feeding it to the decoder is what
                    # releases the frames it holds back for B-frame reorder.
                    # Reusing the copy path's `not packet.size` guard turned
                    # 48 frames into 31 in testing.
                    for frame in packet.decode():
                        for made in _encode(enc, frame, shift, resamplers.get(index)):
                            out.mux(made)
                    continue

                if shift:
                    if packet.pts is not None:
                        packet.pts -= shift
                    if packet.dts is not None:
                        packet.dts -= shift
                packet.stream = target
                out.mux(packet)

            for index, enc in encoders.items():
                resampler = resamplers.get(index)
                if resampler is not None:
                    for ready in resampler.resample(None):
                        for made in enc.encode(ready):
                            out.mux(made)
                for made in enc.encode(None):
                    out.mux(made)
            for _, cue, cue_stream in cues[next_cue:]:
                cue.stream = cue_stream
                out.mux(cue)

        os.replace(part, out_path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    if progress:
        progress(1.0)
    if log:
        size = out_path.stat().st_size
        shown = (
            f"{size / (1 << 30):.2f} GB" if size >= 1 << 30
            else f"{size / (1 << 20):.1f} MB"
        )
        if opts.video_codec == COPY and opts.audio_codec == COPY:
            how = "音视频未重编码"
        else:
            how = "，".join([
                "视频未重编码" if opts.video_codec == COPY
                else f"视频已重编码为 {encoder_label(opts.video_codec)}",
                "音频未重编码" if opts.audio_codec == COPY
                else f"音频已重编码为 {opts.audio_codec}",
            ])
        log(f"已生成带字幕的视频: {out_path.name}（{shown}，{how}）")
    return out_path
