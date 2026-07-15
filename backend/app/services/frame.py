"""Frame extraction for on-screen text translation."""

from __future__ import annotations

from pathlib import Path

import av

MAX_WIDTH = 1280  # frames are downscaled to save vision-model tokens


def parse_time(value: str) -> float:
    """Parse '1:23:45' / '23:45' / '85' (h:m:s, m:s, plain seconds)."""
    parts = [p.strip() for p in str(value).strip().split(":")]
    if not 1 <= len(parts) <= 3 or any(not p for p in parts):
        raise ValueError(f"无法解析时间: {value!r}")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"无法解析时间: {value!r}") from exc
    if any(n < 0 for n in nums):
        raise ValueError(f"时间不能为负: {value!r}")
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def extract_frame(video_path: str | Path, seconds: float, out_jpg: str | Path) -> Path:
    """Save the frame at *seconds* as a JPEG (width capped at MAX_WIDTH)."""
    out_jpg = Path(out_jpg)
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("视频中没有画面流")
        duration = (
            float(container.duration / av.time_base) if container.duration else None
        )
        if duration is not None and seconds > duration:
            raise ValueError(
                f"时间点 {seconds:.0f}s 超出视频时长 {duration:.0f}s"
            )
        stream = container.streams.video[0]
        container.seek(int(seconds * av.time_base))
        frame_found = None
        for frame in container.decode(stream):
            ts = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            frame_found = frame
            if ts >= seconds:
                break
        if frame_found is None:
            raise ValueError(f"无法在 {seconds:.0f}s 处解码到画面帧")
        img = frame_found.to_image()
        if img.width > MAX_WIDTH:
            img.thumbnail((MAX_WIDTH, MAX_WIDTH * 4))
        img.save(out_jpg, quality=85)
    return out_jpg
