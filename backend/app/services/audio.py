"""Audio extraction with PyAV: any video container → 16 kHz mono s16 WAV."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import av

SAMPLE_RATE = 16_000

ProgressFn = Callable[[float], None]  # 0..1


def extract_audio(
    video_path: str | Path,
    out_wav: str | Path,
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Decode the first audio stream of *video_path* into a 16 kHz mono WAV.

    Raises ValueError if the file has no audio stream.
    """
    video_path = Path(video_path)
    out_wav = Path(out_wav)

    with av.open(str(video_path)) as in_container:
        if not in_container.streams.audio:
            raise ValueError(f"no audio stream in {video_path}")
        in_stream = in_container.streams.audio[0]
        duration = float(in_container.duration / av.time_base) if in_container.duration else 0.0

        with av.open(str(out_wav), mode="w", format="wav") as out_container:
            out_stream = out_container.add_stream("pcm_s16le", rate=SAMPLE_RATE)
            out_stream.layout = "mono"
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

            for frame in in_container.decode(in_stream):
                if progress and duration and frame.pts is not None:
                    ts = float(frame.pts * in_stream.time_base)
                    progress(min(ts / duration, 1.0))
                for out_frame in resampler.resample(frame):
                    for packet in out_stream.encode(out_frame):
                        out_container.mux(packet)
            # flush resampler and encoder
            for out_frame in resampler.resample(None):
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)
            for packet in out_stream.encode(None):
                out_container.mux(packet)

    if progress:
        progress(1.0)
    return out_wav
