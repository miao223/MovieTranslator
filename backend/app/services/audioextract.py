"""音频: one audio track of a video, as a file for translation.

The 音频 page's engine. What it writes is what the pipeline itself would
transcribe — 16 kHz mono. By default FLAC: lossless, it decodes sample for
sample to the WAV a direct translation of the video transcribes, at about
half its size (an 87-minute film, 82 MB). Opus 24 kbps is the small option
(15 MB), which the local speech detection hears slightly differently —
97.6% of the same speech on the one film measured. MP3 was offered once and
dropped: at 32 and 64 kbps alike silero found only 83–84% of it.

The file feeds straight back into 翻译任务 as an audio-only source, or goes
to a machine that has no room for the film.

Every decoding rule lives in audio.write_track, the loop the pipeline's WAV
extraction has always used; this module only names the file, tags it, and
recognises it again.

Written as 片名.flac beside the video (or in a chosen folder), never over
anything (mux.free_path). The batch scan groups 片名.mkv and 片名.flac as
one film and keeps the video, so extracting next to a library does not make
a later batch translation run twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import av

from app.models.schemas import AudioOptions, AudioRequest
from app.services import audio, mux

# default_kbps is what the file is written at; 0 = lossless, no target (the
# page estimates FLAC by the rate measured on a film: 81.7 MB in 5194 s)
FORMATS = {
    "flac": {"container": "flac", "codec": "flac", "suffix": ".flac", "name": "FLAC",
             "default_kbps": 0},
    "opus": {"container": "opus", "codec": "libopus", "suffix": ".opus", "name": "Opus",
             "default_kbps": 24},
}

# The file says what it was made from, so a retried entry — or the same
# file queued again — finds it instead of writing 片名.2.opus next to it.
# The encode's tag (encode.read_tags) does the same for 压制.
TAG = "MOVIETRANSLATOR_AUDIO"


def kbps(options: AudioOptions) -> int:
    """The encoder's target rate; 0 for FLAC, which has none."""
    fmt = FORMATS[options.format]
    if not fmt["default_kbps"]:
        return 0
    return options.bitrate_kbps or fmt["default_kbps"]


def describe(options: AudioOptions) -> str:
    """"FLAC 无损 · 16 kHz 单声道" / "Opus 24 kbps · 16 kHz 单声道"."""
    rate = kbps(options)
    return (f"{FORMATS[options.format]['name']} {f'{rate} kbps' if rate else '无损'}"
            " · 16 kHz 单声道")


def probe(path: Path) -> dict:
    """What the 音频 page shows of one file: length, size, audio tracks,
    and whether an earlier extraction of it is already sitting beside it."""
    with av.open(str(path)) as container:
        tracks = [audio.track_info(s) for s in container.streams.audio]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        name = container.format.name
    if not tracks:
        raise ValueError("这个文件里没有音轨")
    done = extracted_beside(path)
    return {
        "path": str(path), "name": path.name, "size": path.stat().st_size,
        "duration": duration, "container": name, "tracks": tracks,
        # what the job will take when nothing is chosen
        "default_track": audio.pick_track(tracks)["index"],
        "extracted": str(done) if done else "",
    }


def extracted_beside(source: Path) -> Optional[Path]:
    """An audio file beside *source* that this program extracted from it,
    in any format and from any track — what the page marks 已提取 and
    leaves unticked."""
    size = source.stat().st_size
    for fmt in FORMATS.values():
        base = source.with_name(source.stem + fmt["suffix"])
        candidate, n = base, 2
        while candidate.exists() and n < 100:
            tag = read_tag(candidate)
            if tag and tag.get("source") == source.name and tag.get("size") == size:
                return candidate
            candidate = base.with_name(f"{base.stem}.{n}{base.suffix}")
            n += 1
    return None


def _base_target(source: Path, request: AudioRequest) -> Path:
    folder = (Path(request.output_dir) if request.output_mode == "custom"
              else source.parent)
    return folder / (source.stem + FORMATS[request.options.format]["suffix"])


def output_target(source: Path, request: AudioRequest) -> Path:
    """Where the audio is written: never over an existing file."""
    return mux.free_path(_base_target(source, request))


def tag_for(source: Path, request: AudioRequest, track: int) -> dict:
    return {"source": source.name, "size": source.stat().st_size, "track": track,
            "format": request.options.format, "kbps": kbps(request.options)}


def read_tag(path: Path) -> Optional[dict]:
    """Our tag on *path*, or None. Ogg keeps it on the stream (as a Vorbis
    comment), FLAC on the file, so both are looked at."""
    try:
        with av.open(str(path)) as container:
            found = [container.metadata] + [s.metadata for s in container.streams]
            raw = next((m[TAG] for m in found if TAG in m), None)
    except (av.error.FFmpegError, OSError):
        return None
    if raw is None:
        return None
    try:
        tag = json.loads(raw)
    except ValueError:
        return None
    return tag if isinstance(tag, dict) else None


def existing(source: Path, request: AudioRequest, track: int) -> Optional[Path]:
    """This request's finished file, if an earlier run already wrote it:
    one where it would have gone (片名.opus, 片名.2.opus…) whose tag names
    this source, at this size, this track and these options."""
    if not source.exists():
        return None
    want = tag_for(source, request, track)
    base = _base_target(source, request)
    candidate, n = base, 2
    while candidate.exists() and n < 100:
        if read_tag(candidate) == want:
            return candidate
        candidate = base.with_name(f"{base.stem}.{n}{base.suffix}")
        n += 1
    return None


def extract(source: Path, out: Path, request: AudioRequest, track: int,
            log: Optional[Callable[[str], None]] = None,
            progress: Optional[Callable[[float], None]] = None,
            should_cancel: Optional[Callable[[], bool]] = None) -> Path:
    """Write *track* of *source* to *out* in the request's format, tagged."""
    fmt = FORMATS[request.options.format]
    return audio.write_track(
        source, out, progress=progress, track_index=track, log=log,
        fmt=fmt["container"], codec=fmt["codec"], bit_rate=kbps(request.options) * 1000,
        metadata={TAG: json.dumps(tag_for(source, request, track), ensure_ascii=False)},
        should_cancel=should_cancel)
