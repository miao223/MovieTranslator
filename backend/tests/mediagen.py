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


def make_open_gop_cut(path: Path, frames: int = 96) -> Path:
    """What `ffmpeg -ss … -c copy` leaves of a film with open GOPs: it starts on
    a keyframe followed, in decode order, by two B-frames that are shown
    before it and predicted from the GOP the cut removed. Read back, the
    matroska demuxer makes up DTS for them that the muxer then refuses
    (mux.CopyPipe). The x264 settings are the ones that put B-frames right
    before a keyframe: without b-adapt=0 it ends each GOP on a P-frame."""
    whole = path.with_name(path.stem + ".whole.mkv")
    with av.open(str(whole), "w", format="matroska") as c:
        v = c.add_stream("libx264", rate=24)
        v.width, v.height, v.pix_fmt = 160, 120, "yuv420p"
        v.options = {"x264-params":
                     "open-gop=1:bframes=3:b-adapt=0:keyint=23:min-keyint=23:scenecut=0"}
        for i in range(frames):
            img = np.zeros((120, 160, 3), np.uint8)
            img[:, :, 0] = (i * 7) % 256
            img[(i * 3) % 120, :, 1] = 255
            img[:, (i * 5) % 160, 2] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for p in v.encode(frame):
                c.mux(p)
        for p in v.encode(None):
            c.mux(p)
    with av.open(str(whole)) as s, av.open(str(path), "w", format="matroska") as o:
        vs = s.streams.video[0]
        ov = o.add_stream_from_template(vs)
        keys = 0
        for p in s.demux(vs):
            if not p.size:
                continue
            keys += p.is_keyframe
            if keys >= 2:          # from the second keyframe on
                p.stream = ov
                o.mux(p)
    whole.unlink()
    return path


def make_fields(path: Path, *, frames: int = 30, rate=Fraction(30000, 1001),
                size=(128, 96), bottom_first: bool = False, flagged: bool = True,
                fmt: str = "matroska", sar=None, still: int = 0) -> Path:
    """An interlaced recording whose two fields were shot at different
    moments: a bar moves 8 px a frame, and the later field of each frame
    shows it 4 px further on. Played a field at a time in the right order,
    the bar's left edge steps 0, 4, 8, 12…; in the wrong order it jitters.

    *flagged* encodes it as interlaced (x264 ildct/ilme, and bff=1 for
    bottom-first), so the decoder says which field comes first; unflagged
    it is the analog capture that says nothing — bwdif then assumes
    top-first. *fmt* "avi" gives the capture's own time base, one tick
    per frame (1001/30000), with no room for a second field in between.
    The first *still* frames hold the bar where it is, in both fields: a
    stretch with no motion, which says nothing about field order.
    """
    rate = Fraction(rate)
    w, h = size
    with av.open(str(path), "w", format=fmt) as c:
        v = c.add_stream("libx264", rate=rate)
        v.width, v.height = size
        v.pix_fmt = "yuv420p"
        # no B-frames in AVI: it stores no presentation timestamps of its own
        options = {"bf": "0" if fmt == "avi" else "2", "g": "12"}
        if flagged:
            options["flags"] = "+ildct+ilme"
            if bottom_first:
                options["x264-params"] = "bff=1"
        v.options = options
        if sar:
            v.codec_context.sample_aspect_ratio = sar
        earlier, later = (1, 0) if bottom_first else (0, 1)   # row parity
        for i in range(frames):
            img = np.full((h, w, 3), 16, dtype=np.uint8)
            if i < still:
                img[:, 0:6] = 235
            else:
                x = (i * 8) % (w - 16)
                img[earlier::2, x:x + 6] = 235
                img[later::2, x + 4:x + 10] = 235
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in v.encode(frame):
                c.mux(packet)
        for packet in v.encode(None):
            c.mux(packet)
    return path


def bar_positions(path: Path, limit: int = 0) -> list:
    """The moving bar's left edge in each decoded frame (make_fields)."""
    found = []
    with av.open(str(path)) as c:
        for frame in c.decode(c.streams.video[0]):
            gray = frame.to_ndarray(format="gray")
            middle = gray[gray.shape[0] // 3: gray.shape[0] * 2 // 3].mean(axis=0)
            found.append(int(np.argmax(middle > 120)))
            if limit and len(found) >= limit:
                break
    return found


def make_avi(path: Path, *, frames: int = 30, rate=25, size=(64, 48), bframes: int = 2) -> Path:
    """H.264 with B-frames in AVI, the way capture software writes it. AVI
    stores no presentation times: FFmpeg's demuxer labels each packet with
    a frame counter, so the decoder hands the pictures back in display
    order but labelled in decode order (1,3,4,2,6,7,5…)."""
    w, h = size
    with av.open(str(path), "w", format="avi") as c:
        v = c.add_stream("libx264", rate=rate)
        v.width, v.height = size
        v.pix_fmt = "yuv420p"
        v.options = {"bf": str(bframes), "g": "12"}
        for i in range(frames):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, (i * 2) % (w - 4):(i * 2) % (w - 4) + 4] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in v.encode(frame):
                c.mux(packet)
        for packet in v.encode(None):
            c.mux(packet)
    return path


def _bars(t: float, w: int, h: int) -> np.ndarray:
    """The scene at *t* fields in: sharp vertical bars moving 6 px a field,
    over a gentle gradient so it is not all one colour."""
    x = (np.arange(w) + 6 * t) % 32
    row = np.where(x < 12, 225.0, 30.0)
    img = row[None, :] + np.linspace(0, 20, h)[:, None]
    return np.repeat(img[:, :, None], 3, axis=2)


def make_cadence(path: Path, kind: str, *, frames: int = 300, size=(160, 120)) -> Path:
    """An NTSC disc's picture in one of the cadences DVDs come in, flagged
    interlaced top-field-first as every DVD is, whatever it holds:

    interlaced  each field its own moment (a video camera)
    telecine    24p film by 3:2 pulldown: 2 frames in 5 weave two pictures
    shifted     30p pictures stored a field out of step: every frame combed,
                every one recoverable by field matching
    progressive 30p pictures, both fields of one moment
    """
    w, h = size
    with av.open(str(path), "w", format="matroska") as c:
        v = c.add_stream("libx264", rate=Fraction(30000, 1001))
        v.width, v.height = size
        v.pix_fmt = "yuv420p"
        v.options = {"bf": "2", "g": "12", "flags": "+ildct+ilme", "crf": "12"}
        film = []                       # 3:2 pulldown: which film frame each field shows
        n = 0
        while len(film) < 2 * frames + 4:
            film += [n] * (2 if n % 2 == 0 else 3)
            n += 1
        for i in range(frames):
            if kind == "interlaced":
                top, bottom = _bars(2 * i, w, h), _bars(2 * i + 1, w, h)
            elif kind == "telecine":
                top, bottom = (_bars(2.5 * film[2 * i], w, h), _bars(2.5 * film[2 * i + 1], w, h))
            elif kind == "shifted":
                top, bottom = _bars(2 * (i + 1), w, h), _bars(2 * i, w, h)
            else:
                top = bottom = _bars(2 * i, w, h)
            img = np.empty((h, w, 3))
            img[0::2], img[1::2] = top[0::2], bottom[1::2]
            frame = av.VideoFrame.from_ndarray(img.astype(np.uint8), format="rgb24")
            frame.pts = i
            for packet in v.encode(frame):
                c.mux(packet)
        for packet in v.encode(None):
            c.mux(packet)
    return path
