"""Write real HDMV PGS (.sup) streams, so the OCR path is testable in CI.

Bitmap subtitles cannot be synthesized with ffmpeg — it refuses to encode
text into a bitmap codec ("only possible from text to text or bitmap to
bitmap") — and shipping a Blu-ray sample is out of the question. So this
builds the bitstream: PCS/WDS/PDS/ODS segments and PGS run-length encoding,
with the glyphs drawn by PIL. ffprobe reads the result as
``hdmv_pgs_subtitle``, and PyAV decodes it like any disc's.

Three palette styles, because the binarizer's whole job is telling the fill
apart from everything else and each style attacks that differently:

``outline``
    Three indices — transparent, black outline, white fill. The textbook
    case, and the easiest.
``antialiased``
    What discs actually ship: the glyph edges are blended, so a dozen-odd
    intermediate entries sit between fill, outline and transparency.
``box``
    A solid background plate with the text knocked out of it. Here the fill
    is the *background*, and a binarizer that just takes "whatever is in the
    middle" gets a black rectangle.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

CANVAS = (1920, 1080)
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
SUPERSAMPLE = 3  # rendered this many times over, then shrunk, to get edges


def _font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _masks(text: str, size: int, stroke: int) -> tuple[Image.Image, Image.Image]:
    """(alpha, luma) masks: everything drawn, and the fill alone."""
    font = _font(size * SUPERSAMPLE)
    pad = (12 + stroke) * SUPERSAMPLE
    gap = 10 * SUPERSAMPLE
    lines = text.split("\n")
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    boxes = [probe.textbbox((0, 0), line, font=font) for line in lines]
    w = max(b[2] - b[0] for b in boxes) + pad * 2
    h = sum(b[3] - b[1] for b in boxes) + pad * 2 + gap * (len(lines) - 1)

    alpha = Image.new("L", (w, h), 0)
    luma = Image.new("L", (w, h), 0)
    da, dl = ImageDraw.Draw(alpha), ImageDraw.Draw(luma)
    y = pad
    for line, box in zip(lines, boxes):
        x = (w - (box[2] - box[0])) // 2 - box[0]
        da.text((x, y - box[1]), line, font=font, fill=255,
                stroke_width=stroke * SUPERSAMPLE, stroke_fill=255)
        dl.text((x, y - box[1]), line, font=font, fill=255)
        y += box[3] - box[1] + gap
    small = (w // SUPERSAMPLE, h // SUPERSAMPLE)
    return (alpha.resize(small, Image.LANCZOS), luma.resize(small, Image.LANCZOS))


def render(text: str, style: str = "outline", size: int = 52,
           stroke: int = 3) -> Image.Image:
    """An indexed bitmap of *text*, shaped like the chosen palette style.

    *size* and *stroke* together set how thick the outline is relative to
    the glyph, which is the whole difference between a Blu-ray cue and a
    DVD one: at 52/3 the fill sits about twice as deep as the outline, at
    20/1 only about half again as deep.
    """
    alpha, luma = _masks(text, size, stroke=stroke)
    w, h = alpha.size
    img = Image.new("P", (w, h), 0)
    px, ap, lp = img.load(), alpha.load(), luma.load()

    if style == "box":
        # a plate covering the whole rect, with the glyphs cut out of it
        for y in range(h):
            for x in range(w):
                px[x, y] = 2 if lp[x, y] > 128 else 1
        return img

    steps = 1 if style == "outline" else 4  # quantisation levels per axis
    index: dict[tuple[int, int], int] = {}
    for y in range(h):
        for x in range(w):
            a, l = ap[x, y], lp[x, y]
            if a < 16:
                continue  # transparent
            if style == "outline":
                px[x, y] = 2 if l > 128 else 1
                continue
            key = (min(a * steps // 256, steps - 1), min(l * steps // 256, steps - 1))
            px[x, y] = index.setdefault(key, len(index) + 1)
    return img


def rle(img: Image.Image) -> bytes:
    """PGS run-length encoding, one scan line at a time."""
    px = img.load()
    w, h = img.size
    out = bytearray()
    for y in range(h):
        x = 0
        while x < w:
            colour = px[x, y]
            n = 1
            while x + n < w and px[x + n, y] == colour and n < 16383:
                n += 1
            if colour == 0:
                out += (bytes((0x00, n)) if n <= 63
                        else bytes((0x00, 0x40 | (n >> 8), n & 0xFF)))
            elif n <= 2:
                out += bytes((colour,)) * n
            elif n <= 63:
                out += bytes((0x00, 0x80 | n, colour))
            else:
                out += bytes((0x00, 0xC0 | (n >> 8), n & 0xFF, colour))
            x += n
        out += b"\x00\x00"  # end of line
    return bytes(out)


def _segment(pts90: int, kind: int, payload: bytes) -> bytes:
    return b"PG" + struct.pack(">IIBH", pts90, 0, kind, len(payload)) + payload


def _pcs(state: int, objects: Sequence[tuple[int, int, int, int]]) -> bytes:
    out = struct.pack(">HHBHBBBB", CANVAS[0], CANVAS[1], 0x10, 0, state, 0, 0,
                      len(objects))
    for obj_id, win_id, x, y in objects:
        out += struct.pack(">HBBHH", obj_id, win_id, 0, x, y)
    return out


def _wds(windows: Sequence[tuple[int, int, int, int, int]]) -> bytes:
    out = bytes((len(windows),))
    for win_id, x, y, w, h in windows:
        out += struct.pack(">BHHHH", win_id, x, y, w, h)
    return out


def _pds(img: Image.Image, style: str) -> bytes:
    """A palette covering every index the bitmap actually uses."""
    used = sorted(set(img.tobytes()) - {0})
    out = bytes((0, 0))  # palette id, version
    for i in used:
        if style == "outline" or style == "box":
            y, a = (235, 255) if i == 2 else (16, 255)
        else:
            # indices were handed out as (alpha level, luma level) pairs, so
            # recovering a plausible colour only needs the same arithmetic
            a_level, l_level = divmod(i - 1, 4)
            y = 16 + (235 - 16) * l_level // 3
            a = 64 + (255 - 64) * a_level // 3
        out += bytes((i, y, 128, 128, a))
    return out


def _ods(img: Image.Image, obj_id: int = 0) -> bytes:
    data = rle(img)
    return (struct.pack(">HBB", obj_id, 0, 0xC0)
            + (len(data) + 4).to_bytes(3, "big")
            + struct.pack(">HH", *img.size) + data)


def write_sup(path: Path, cues: Sequence[tuple[float, float, str]],
              style: str = "outline") -> Path:
    """cues: [(start_seconds, end_seconds, text), ...] -> a .sup file."""
    out = bytearray()
    for start, end, text in cues:
        img = render(text, style)
        w, h = img.size
        x, y = (CANVAS[0] - w) // 2, CANVAS[1] - h - 60
        t0, t1 = int(start * 90000), int(end * 90000)
        out += _segment(t0, 0x16, _pcs(0x80, [(0, 0, x, y)]))
        out += _segment(t0, 0x17, _wds([(0, x, y, w, h)]))
        out += _segment(t0, 0x14, _pds(img, style))
        out += _segment(t0, 0x15, _ods(img))
        out += _segment(t0, 0x80, b"")
        out += _segment(t1, 0x16, _pcs(0x00, []))
        out += _segment(t1, 0x17, _wds([(0, x, y, w, h)]))
        out += _segment(t1, 0x80, b"")
    Path(path).write_bytes(bytes(out))
    return Path(path)


def write_two_region_sup(path: Path, start: float, end: float,
                         top: str, bottom: str) -> Path:
    """One cue whose two lines are separate regions, as discs often do."""
    a, b = render(top), render(bottom)
    x_a, x_b = (CANVAS[0] - a.width) // 2, (CANVAS[0] - b.width) // 2
    y_b = CANVAS[1] - b.height - 60
    y_a = y_b - a.height - 8
    t0, t1 = int(start * 90000), int(end * 90000)
    out = bytearray()
    out += _segment(t0, 0x16, _pcs(0x80, [(0, 0, x_a, y_a), (1, 1, x_b, y_b)]))
    out += _segment(t0, 0x17, _wds([(0, x_a, y_a, a.width, a.height),
                                    (1, x_b, y_b, b.width, b.height)]))
    out += _segment(t0, 0x14, _pds(a, "outline"))
    out += _segment(t0, 0x15, _ods(a, 0))
    out += _segment(t0, 0x15, _ods(b, 1))
    out += _segment(t0, 0x80, b"")
    out += _segment(t1, 0x16, _pcs(0x00, []))
    out += _segment(t1, 0x17, _wds([(0, x_a, y_a, a.width, a.height)]))
    out += _segment(t1, 0x80, b"")
    Path(path).write_bytes(bytes(out))
    return Path(path)


def write_repeating_sup(path: Path, start: float, end: float, text: str,
                        every: float = 1.0) -> Path:
    """One subtitle, re-sent every *every* seconds without ever clearing.

    Real discs do this — measured on a Blu-ray, 1165 of 2304 consecutive
    compositions were exactly a second apart with a byte-identical bitmap.
    """
    img = render(text)
    w, h = img.size
    x, y = (CANVAS[0] - w) // 2, CANVAS[1] - h - 60
    out = bytearray()
    at = start
    while at < end - 1e-6:
        t = int(at * 90000)
        out += _segment(t, 0x16, _pcs(0x80, [(0, 0, x, y)]))
        out += _segment(t, 0x17, _wds([(0, x, y, w, h)]))
        out += _segment(t, 0x14, _pds(img, "outline"))
        out += _segment(t, 0x15, _ods(img))
        out += _segment(t, 0x80, b"")
        at += every
    t1 = int(end * 90000)
    out += _segment(t1, 0x16, _pcs(0x00, []))
    out += _segment(t1, 0x17, _wds([(0, x, y, w, h)]))
    out += _segment(t1, 0x80, b"")
    Path(path).write_bytes(bytes(out))
    return Path(path)
