"""Remux a video with the finished subtitle as a soft track (no re-encode).

The alternative to writing a .srt next to the film: one new file that plays
with subtitles anywhere, with the track still switchable and the picture
untouched. Audio and video packets are copied verbatim — this is a change
of container, not a transcode, so a two-hour film takes the time of a file
copy rather than the time of an encode.

The subtitle is not assembled packet by packet here. It is written as a
normal .srt/.ass first and then opened as an *input container*, so
`add_stream_from_template` copies the codec parameters — including the ASS
`[Script Info]`/`[V4+ Styles]` header, which lives in codecpar.extradata
and cannot be set from Python at all. Everything the file version does —
bilingual layout, wrapping, ♪, {\\an7} frame cues — reaches the embedded
track unchanged, because it is literally the same file.

MKV is the only output container: it accepts essentially any audio/video
codec, which matters when the input is only being copied, and it is the
only common one that can carry styled ASS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import av

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
}
FALLBACK = ("sub", "und")

# stream types worth carrying over. Attachments matter more than they look:
# they are where a release keeps the fonts its subtitles are styled with.
COPIED = ("video", "audio", "subtitle", "attachment")

# Cancellation is checked on every packet — an attribute read against a
# copy that can run for minutes — while progress, which reaches the job log
# and the SSE stream, is throttled.
PROGRESS_EVERY = 200


def language_of(target_language: str) -> tuple[str, str]:
    return LANGUAGES.get(target_language.strip(), FALLBACK)


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


def output_path(video: Path, target_language: str) -> Path:
    """Where the muxed film goes: film.mkv -> film.zh.mkv, same folder.

    Never returns a path that already exists — overwriting the user's video
    library is not a risk worth taking for a naming collision.
    """
    suffix = language_of(target_language)[0]
    candidate = video.parent / f"{video.stem}.{suffix}.mkv"
    n = 2
    while candidate.exists():
        candidate = video.parent / f"{video.stem}.{suffix}.{n}.mkv"
        n += 1
    return candidate


def embed(
    video_path: str | Path,
    subtitle_path: str | Path,
    out_path: str | Path,
    target_language: str = "",
    log: Optional[LogFn] = None,
    progress: Optional[ProgressFn] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    track_title: str = "",
) -> Path:
    """Copy *video_path* into *out_path*, adding *subtitle_path* as a track.

    Raises InterruptedError when cancelled, and anything PyAV raises when
    the copy itself fails — the caller is expected to fall back to writing
    the subtitle file rather than losing the job.
    """
    video_path, subtitle_path, out_path = (
        Path(video_path), Path(subtitle_path), Path(out_path)
    )
    should_cancel = should_cancel or (lambda: False)
    # written under a temporary name: a half-copied film left in the user's
    # library looks exactly like a real one until they try to play it
    part = out_path.with_name(out_path.name + ".part")
    lang = language_of(target_language)[1]

    try:
        with av.open(str(video_path)) as source, \
             av.open(str(subtitle_path)) as subs_in, \
             av.open(str(part), mode="w", format="matroska") as out:
            if not subs_in.streams.subtitles:
                raise ValueError(f"字幕文件无法解析: {subtitle_path.name}")
            sub_in = subs_in.streams.subtitles[0]

            out.metadata.update(dict(source.metadata))
            mapping: Dict[int, object] = {}
            for stream in source.streams:
                if stream.type not in COPIED:
                    continue
                try:
                    copy = out.add_stream_from_template(stream)
                except Exception as exc:  # noqa: BLE001
                    if stream.type in ("video", "audio"):
                        # Never skip these. PyAV asks libavformat at normal
                        # compliance, which turns down a few old codecs MKV
                        # can technically carry (msmpeg4v2, wmv, vc1, adpcm).
                        # Skipping produced a file with no picture at all —
                        # far worse than falling back to a subtitle file.
                        raise ValueError(
                            f"mkv 容器无法容纳片源的{stream.type}编码 "
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
                    mapping[stream.index] = copy

            subtitle_stream = out.add_stream_from_template(sub_in)
            subtitle_stream.metadata["language"] = lang
            if track_title:
                subtitle_stream.metadata["title"] = track_title
            # the point of the whole exercise is that it shows up on its own
            subtitle_stream.disposition = av.stream.Disposition.default

            # a film's worth of cues is a few hundred KB; reading them up
            # front is what lets them be merged into the copy in time order
            cues = [p for p in subs_in.demux(sub_in) if p.size]
            cues.sort(key=lambda p: p.pts or 0)

            duration = (
                float(source.duration / av.time_base) if source.duration else 0.0
            )
            # A container's timeline need not start at zero — TS captures and
            # edit lists routinely start hours in. The subtitles do start at
            # zero, because the transcript comes from audio decoded from the
            # first frame regardless of its timestamp, so copying the source's
            # timestamps unshifted would slide every cue by that much. This is
            # what ffmpeg does by default for each input.
            #
            # The shift is the smallest DTS, not start_time: start_time is a
            # PTS, and with B-frames the first DTS is smaller still (often
            # negative). Subtracting the larger of the two pushes DTS below
            # zero, which matroska cannot store — it drops those packets, and
            # the first of them is the opening keyframe.
            offset = min(_start_time(source), _first_dts(video_path))
            if log and offset >= 1.0:
                log(f"片源时间轴从 {offset:.1f}s 开始，已整体对齐到 0 以匹配字幕")
            seen = 0
            next_cue = 0
            for packet in source.demux():
                if should_cancel():
                    raise InterruptedError
                # Only the empty packet demux emits at end-of-stream may be
                # dropped. Testing `dts is None` instead — the idiom in every
                # remux example — silently threw away real frames: matroska
                # stores no DTS at all, so its demuxer hands back content with
                # dts unset, and the very first such packet is the opening
                # keyframe. Measured on a real mkv: 24 of 1104 frames gone,
                # the file starting with an undecodable GOP.
                if not packet.size:
                    continue
                target = mapping.get(packet.stream.index)
                if target is None:
                    continue
                if offset:
                    shift = round(offset / float(packet.time_base))
                    if packet.pts is not None:
                        packet.pts -= shift
                    if packet.dts is not None:
                        packet.dts -= shift
                seen += 1
                if seen % PROGRESS_EVERY == 0 and progress and duration \
                        and packet.pts is not None:
                    progress(min(float(packet.pts * packet.time_base) / duration, 1.0))

                at = (
                    float(packet.pts * packet.time_base)
                    if packet.pts is not None else 0.0
                )
                while next_cue < len(cues):
                    cue = cues[next_cue]
                    if float(cue.pts * cue.time_base) > at:
                        break
                    cue.stream = subtitle_stream
                    out.mux(cue)
                    next_cue += 1

                packet.stream = target
                out.mux(packet)

            for cue in cues[next_cue:]:
                cue.stream = subtitle_stream
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
        log(f"已生成带字幕的视频: {out_path.name}（{shown}，音视频未重编码）")
    return out_path
