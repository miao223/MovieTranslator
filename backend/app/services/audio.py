"""Audio extraction with PyAV: any video container → 16 kHz mono s16 WAV.

Multi-track releases (original + dub, or commentary tracks) are common, so
tracks are enumerated and one is picked explicitly instead of blindly taking
the first stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import av

SAMPLE_RATE = 16_000

ProgressFn = Callable[[float], None]  # 0..1
LogFn = Callable[[str], None]

AV_DISPOSITION_DEFAULT = 1 << 0

# ISO 639-2/B is what containers usually carry; fold the common variants in
_LANG_CANON = {
    "ja": "jpn", "jp": "jpn", "jap": "jpn",
    "en": "eng",
    "zh": "chi", "zho": "chi", "cmn": "chi",
    "ko": "kor",
    "fr": "fre", "fra": "fre",
    "de": "ger", "deu": "ger",
    "es": "spa",
    "ru": "rus",
    "it": "ita",
    "pt": "por",
    "th": "tha",
    "vi": "vie",
    "ar": "ara",
    "hi": "hin",
}

_LANG_NAMES = {
    "jpn": "日语", "eng": "英语", "chi": "中文", "kor": "韩语",
    "fre": "法语", "ger": "德语", "spa": "西班牙语", "rus": "俄语",
    "ita": "意大利语", "por": "葡萄牙语", "tha": "泰语", "vie": "越南语",
    "ara": "阿拉伯语", "hin": "印地语", "und": "未标注语言",
}

_CHANNEL_NAMES = {1: "单声道", 2: "立体声", 6: "5.1 声道", 8: "7.1 声道"}


def canon_language(code: str) -> str:
    """Normalize a language tag so 'ja' / 'jpn' / 'jap' compare equal."""
    code = (code or "").strip().lower()
    return _LANG_CANON.get(code, code)


def language_name(code: str) -> str:
    canon = canon_language(code)
    if not canon:
        return "未标注语言"
    return _LANG_NAMES.get(canon, canon)


def _track_info(stream) -> dict:
    cc = stream.codec_context
    layout = getattr(cc, "layout", None)
    channels = getattr(layout, "nb_channels", 0) or 0
    return {
        "index": stream.index,  # container-wide stream index, as ffmpeg reports it
        "codec": (cc.name or "").upper(),
        "language": canon_language(stream.language or ""),
        "language_name": language_name(stream.language or ""),
        "title": (stream.metadata.get("title") or "").strip(),
        "channels": channels,
        "channel_name": _CHANNEL_NAMES.get(channels, f"{channels} 声道" if channels else ""),
        "sample_rate": cc.sample_rate or 0,
        "default": bool((stream.disposition or 0) & AV_DISPOSITION_DEFAULT),
    }


def describe_track(track: dict) -> str:
    """Human-readable one-liner, e.g. '音轨 #1 日语「导演评论」AAC 立体声'."""
    parts = [f"音轨 #{track['index']}", track["language_name"]]
    if track["title"]:
        parts.append(f"「{track['title']}」")
    tail = " ".join(p for p in (track["codec"], track["channel_name"]) if p)
    if tail:
        parts.append(tail)
    if track["default"]:
        parts.append("(默认)")
    return " ".join(parts)


def list_tracks(video_path: str | Path) -> list[dict]:
    """Enumerate the audio tracks of *video_path*.

    Raises ValueError if the file carries no audio at all.
    """
    video_path = Path(video_path)
    with av.open(str(video_path)) as container:
        tracks = [_track_info(s) for s in container.streams.audio]
    if not tracks:
        raise ValueError(f"视频中没有音频流: {video_path.name}")
    return tracks


def describe_media(video_path: str | Path) -> list[str]:
    """Container / video / audio summary for the job log.

    Every audio track is listed, not just the chosen one — a decode failure
    is usually answered by "which codec was it actually handed".
    """
    video_path = Path(video_path)
    lines: list[str] = []
    try:
        size = video_path.stat().st_size
        lines.append(f"文件大小      : {size / (1 << 30):.2f} GB ({size} 字节)")
    except OSError as exc:
        lines.append(f"文件大小      : 读取失败 ({exc})")

    with av.open(str(video_path)) as container:
        duration = (
            float(container.duration / av.time_base) if container.duration else 0.0
        )
        lines.append(
            f"容器格式      : {container.format.name} "
            f"时长 {int(duration // 3600):02d}:{int(duration % 3600 // 60):02d}:{int(duration % 60):02d}"
        )
        for stream in container.streams.video:
            cc = stream.codec_context
            lines.append(
                f"视频流        : #{stream.index} {cc.name} "
                f"{cc.width}x{cc.height} {float(stream.average_rate or 0):.3f} fps"
            )
        for track in (_track_info(s) for s in container.streams.audio):
            lines.append(f"音轨          : {describe_track(track)} @{track['sample_rate']}Hz")
        for stream in container.streams.subtitles:
            lines.append(
                f"内嵌字幕      : #{stream.index} {stream.codec_context.name} "
                f"{stream.language or '未标注'}"
            )
    return lines


def pick_track(
    tracks: list[dict],
    index: Optional[int] = None,
    language: str = "",
) -> dict:
    """Choose one track: explicit index > language preference > default > first.

    An index that no longer exists (e.g. the video path changed after the user
    picked a track) falls back instead of raising — the caller logs the switch
    rather than losing a long-running job.
    """
    if index is not None:
        for t in tracks:
            if t["index"] == index:
                return t
    wanted = canon_language(language)
    if wanted:
        for t in tracks:
            if t["language"] == wanted:
                return t
    for t in tracks:
        if t["default"]:
            return t
    return tracks[0]


def extract_audio(
    video_path: str | Path,
    out_wav: str | Path,
    progress: Optional[ProgressFn] = None,
    track_index: Optional[int] = None,
    log: Optional[LogFn] = None,
) -> Path:
    """Decode one audio stream of *video_path* into a 16 kHz mono WAV.

    *track_index* is a container stream index (see `list_tracks`); None takes
    the first audio stream. Damaged packets are skipped rather than aborting
    the job. Raises ValueError if the file has no audio stream, the requested
    index is not an audio stream, or nothing at all could be decoded.
    """
    video_path = Path(video_path)
    out_wav = Path(out_wav)

    with av.open(str(video_path)) as in_container:
        if not in_container.streams.audio:
            raise ValueError(f"视频中没有音频流: {video_path.name}")
        in_stream = in_container.streams.audio[0]
        if track_index is not None:
            match = next(
                (s for s in in_container.streams.audio if s.index == track_index), None
            )
            if match is None:
                raise ValueError(f"音轨 #{track_index} 不存在于 {video_path.name}")
            in_stream = match
        duration = float(in_container.duration / av.time_base) if in_container.duration else 0.0

        with av.open(str(out_wav), mode="w", format="wav") as out_container:
            out_stream = out_container.add_stream("pcm_s16le", rate=SAMPLE_RATE)
            out_stream.layout = "mono"
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

            decoded = bad_packets = 0
            reached = 0.0
            try:
                for packet in in_container.demux(in_stream):
                    try:
                        frames = packet.decode()
                    except av.error.FFmpegError:
                        # a damaged packet must not cost the whole film —
                        # ffmpeg's own CLI logs and moves on
                        bad_packets += 1
                        continue
                    for frame in frames:
                        decoded += 1
                        if duration and frame.pts is not None:
                            reached = float(frame.pts * in_stream.time_base)
                            if progress:
                                progress(min(reached / duration, 1.0))
                        for out_frame in resampler.resample(frame):
                            for out_packet in out_stream.encode(out_frame):
                                out_container.mux(out_packet)
            except av.error.FFmpegError as exc:
                # the container itself broke down mid-file
                if not decoded:
                    raise
                if log:
                    log(
                        f"⚠ 视频文件在 {reached:.0f}s 处损坏（{exc}），"
                        "已用此前解码出的音频继续；该时间点之后不会有字幕"
                    )
            # flush resampler and encoder
            for out_frame in resampler.resample(None):
                for out_packet in out_stream.encode(out_frame):
                    out_container.mux(out_packet)
            for out_packet in out_stream.encode(None):
                out_container.mux(out_packet)

    if not decoded:
        raise ValueError(
            f"音轨 #{in_stream.index}（{(in_stream.codec_context.name or '未知').upper()}）"
            "无法解码，可能是编码格式不受支持或该音轨已损坏；"
            "请在任务页改选其他音轨后重试"
        )
    if log:
        log(
            f"音轨 #{in_stream.index} 解码完成：{decoded} 个音频帧，"
            f"覆盖到 {reached:.0f}s / 共 {duration:.0f}s"
            + (f"，跳过 {bad_packets} 个损坏的包" if bad_packets else "")
        )
    if bad_packets and log:
        log("⚠ 文件存在轻微损坏，跳过的部分不会有字幕；其余部分已正常提取")
    if progress:
        progress(1.0)
    return out_wav
