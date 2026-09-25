"""Small films made on the spot, for the tests that encode, remux or embed.

A real release cannot live in the repository, but its shape can: a picture
with B-frames (interlaced or not, anamorphic or not, with any timestamps),
audio tracks in any codec and layout, a subtitle track, fonts, chapters, a
late start, container tags. Pure PyAV — nothing here depends on the code
under test.
"""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def make_source(path: Path, *, frames: int = 24, rate=24, size=(128, 96), sar=None,
                interlaced: bool = False, pts=None, time_base=None, offset: float = 0.0,
                audio=(), chapters=(), font: bool = False, subtitle: Path | None = None,
                tags=None) -> Path:
    """A small film to encode. *audio* is a list of dicts:
    codec, layout, rate, format, language, title, amp."""
    rate = Fraction(rate)
    with av.open(str(path), "w", format="matroska") as c:
        if tags:
            c.metadata.update(tags)
        v = c.add_stream("libx264", rate=rate)
        v.width, v.height = size
        v.pix_fmt = "yuv420p"
        options = {"bf": "2", "g": "12"}
        if interlaced:
            options["flags"] = "+ildct+ilme"
        v.options = options
        if sar:
            v.codec_context.sample_aspect_ratio = sar
        if time_base:
            v.time_base = time_base
            v.codec_context.time_base = time_base
        streams = []
        for spec in audio:
            st = c.add_stream(spec["codec"], rate=spec.get("rate", 48000))
            st.layout = spec.get("layout", "stereo")
            if spec.get("format"):
                st.format = spec["format"]
            st.metadata["language"] = spec.get("language", "und")
            if spec.get("title"):
                st.metadata["title"] = spec["title"]
            streams.append((st, spec))
        sub_in = None
        if subtitle is not None:
            sub_in = av.open(str(subtitle))
            sub_out = c.add_stream_from_template(sub_in.streams.subtitles[0])
            sub_out.metadata["language"] = "eng"
        if font:
            c.add_attachment("Font.ttf", "application/x-truetype-font", b"FONT" * 16)
        if chapters:
            c.set_chapters([{"id": i + 1, "start": int(s * 1000), "end": int(e * 1000),
                             "time_base": Fraction(1, 1000),
                             "metadata": {"title": f"第 {i + 1:02d} 章"}}
                            for i, (s, e) in enumerate(chapters)])
        w, h = size
        packets = []
        for i in range(frames):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, (i * 5) % w:((i * 5) % w) + 12] = 255       # motion to encode
            img[::2, :, 1] = (i * 9) % 255                      # a comb, for bwdif
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            if pts is not None:
                frame.pts, frame.time_base = pts[i], time_base
            else:
                frame.pts = i
            packets += list(v.encode(frame))
        packets += list(v.encode(None))
        seconds = float(frames / rate)
        for st, spec in streams:
            layout = av.AudioLayout(spec.get("layout", "stereo"))
            channels = layout.nb_channels
            sr = spec.get("rate", 48000)
            fmt = spec.get("format") or st.format.name
            packed = not fmt.endswith("p")
            done = 0
            total = int(seconds * sr)
            while done < total:
                n = min(1024, total - done)
                t = (np.arange(n) + done) / sr
                wave = spec.get("amp", 0.5) * np.sin(2 * np.pi * 440 * t)
                data = np.tile(wave, (channels, 1))
                if fmt.startswith("s16"):
                    data = (data * 32767).astype(np.int16)
                elif fmt.startswith("s32"):
                    data = (data * 2147483647).astype(np.int32)
                else:
                    data = data.astype(np.float32)
                if packed:
                    data = data.T.reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(data, format=fmt, layout=layout.name)
                frame.sample_rate = sr
                frame.time_base = Fraction(1, sr)
                frame.pts = done
                done += n
                packets += list(st.encode(frame))
            packets += list(st.encode(None))
        if sub_in is not None:
            for packet in sub_in.demux(sub_in.streams.subtitles[0]):
                if packet.size:
                    packet.stream = sub_out
                    packets.append(packet)
        for packet in packets:
            if offset:
                shift = round(offset / float(packet.time_base))
                packet.pts += shift
                if packet.dts is not None:
                    packet.dts += shift
            c.mux(packet)
        if sub_in is not None:
            sub_in.close()
    return path
