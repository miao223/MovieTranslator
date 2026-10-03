"""试看 (the 修复 page): before a film is queued, a few short moments of it
are run through Lanczos and each AI model on this machine's own GPU and put
side by side — the same 2×2 comparison video the GPU bench made, a sheet of
1:1 crops with a flicker number for each, and each model's measured speed
turned into what the whole film would take.

The user's own choice, after watching the bench's comparisons: the models
differ from film to film ("每个模型各有优劣，不太好判断"), so the choice
is made per film, by eye, on that film.

* One preview at a time; starting another cancels the one running.
* It takes the heavy-job slot (pipeline._run_slot) like a translation or an
  encode: never two models on one card, never a preview racing a queued
  encode for the CPU. While it waits its status is "waiting".
* The film's cadence, black bars and field order are measured once on the
  whole film and handed to every segment as explicit options. On a
  3-second clip they come out differently or not at all (classify needs 60
  frames of motion; the field vote needs movement) — and differently from
  what the real encode, which measures the whole film, would use.
* Files live under the cache's jobs/ folder (wiped at startup); a new
  preview deletes the previous one's.
"""

from __future__ import annotations

import gc
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import av
import numpy as np

from app.core import cache
from app.models.schemas import EncodeOptions, EncodeRequest, RestoreSettings
from app.services import audio, encode, mux, restore

LogFn = Callable[[str], None]

# where in the film, and how long: three moments a few seconds each (the
# bench's clips were 6 s; three shorter ones show more kinds of picture)
POINTS = (0.25, 0.5, 0.75)
SEGMENT_SECONDS = 3.0
METHODS = ["lanczos", "realbasicvsr", "realviformer", "liveaction_span"]
LABELS = {"lanczos": "Original (Lanczos)", "realbasicvsr": "RealBasicVSR",
          "realviformer": "RealViformer", "liveaction_span": "SPAN 2xLiveActionV1"}
# what each method's segment is encoded with: only the model may differ
PREVIEW_ENCODE = {"container": "mkv", "video_codec": "libx264", "rate_control": "quality",
                  "quality": 14, "preset": "fast", "tune": "", "bit_depth": "8",
                  "max_height": 0, "auto_pick": False, "subtitles": "none"}
GRID_CRF = 15
CROP = 360                      # the sheet's 1:1 crops
ENCODE_SHARE = 0.85             # of the progress bar; the rest is the grid


class PreviewError(Exception):
    pass


@dataclass
class Preview:
    id: str
    source: str
    options: EncodeOptions
    status: str = "waiting"         # waiting | running | done | failed | cancelled
    stage: str = "等待开始"
    progress: float = 0.0
    error: str = ""
    created: float = field(default_factory=time.time)
    resolved: List[str] = field(default_factory=list)   # what the film measured as
    segments: List[Tuple[float, float]] = field(default_factory=list)  # (start, seconds) in the film
    methods: Dict[str, dict] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)  # "video" / "sheet" -> file name
    film_seconds: float = 0.0
    log: List[str] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    folder: Optional[Path] = field(default=None, repr=False)

    def view(self) -> dict:
        return {"id": self.id, "source": self.source, "status": self.status, "stage": self.stage,
                "progress": round(self.progress, 4), "error": self.error, "created": self.created,
                "resolved": self.resolved, "segments": self.segments, "methods": self.methods,
                "files": self.files, "film_seconds": self.film_seconds, "log": self.log[-40:],
                "options": self.options.model_dump()}


_lock = threading.Lock()
_current: Optional[Preview] = None


def root() -> Path:
    path = cache.cache_root() / "preview"
    path.mkdir(parents=True, exist_ok=True)
    return path


def current() -> Optional[Preview]:
    return _current


def file_path(name: str) -> Optional[Path]:
    """A file of the current preview, by the name its view lists — nothing
    else is served."""
    p = _current
    if p is None or p.folder is None or name not in p.files.values():
        return None
    path = p.folder / name
    return path if path.is_file() else None


def start(source: Path, options: EncodeOptions, engine: RestoreSettings) -> Preview:
    """Check what can be checked now, cancel the running preview, and start
    this one in its own thread. Raises PreviewError for what the page should
    say instead."""
    global _current
    if not source.is_file():
        raise PreviewError(f"文件不存在：{source}")
    if not options.upscale:
        raise PreviewError("先在「放大到」里选一个尺寸：试看比的是放大的方法")
    if not engine.engine_python:
        raise PreviewError("AI 模型要先在「设置 → 修复引擎」里配置引擎")
    if options.video_codec == "copy":
        raise PreviewError("画面保持原样时没有可比的")
    with _lock:
        old = _current
        preview = Preview(id=uuid.uuid4().hex[:12], source=str(source), options=options)
        _current = preview
    if old is not None:
        old.cancel.set()
        if old.thread is not None:
            old.thread.join(timeout=60)
    for stale in root().iterdir():
        if stale.name != preview.id:
            shutil.rmtree(stale, ignore_errors=True)
    preview.folder = root() / preview.id
    preview.folder.mkdir(parents=True, exist_ok=True)
    preview.thread = threading.Thread(target=_run, args=(preview, source, engine),
                                      name=f"preview-{preview.id}", daemon=True)
    preview.thread.start()
    return preview


def cancel() -> Optional[Preview]:
    p = _current
    if p is not None and p.status in ("waiting", "running"):
        p.cancel.set()
    return p


# ------------------------------------------------------------------ the run

def _run(p: Preview, source: Path, engine: RestoreSettings) -> None:
    from app.services import memguard, pipeline

    def log(line: str) -> None:
        p.log.append(line)

    p.stage = "等待正在跑的任务结束（翻译、压制和试看不同时用显卡）"
    while not pipeline._run_slot.acquire(timeout=0.2):
        if p.cancel.is_set():
            p.status, p.stage = "cancelled", "已取消"
            return
    try:
        p.status = "running"
        _work(p, source, engine, log)
        p.status, p.stage, p.progress = "done", "完成", 1.0
    except InterruptedError:
        p.status, p.stage = "cancelled", "已取消"
    except Exception as exc:  # noqa: BLE001 — the page shows it
        p.status, p.error = "failed", str(exc) or type(exc).__name__
        p.stage = "失败"
        log(f"✗ {p.error}")
    finally:
        pipeline._run_slot.release()
        gc.collect()
        memguard.trim()


def _check(p: Preview) -> None:
    if p.cancel.is_set():
        raise InterruptedError


def _work(p: Preview, source: Path, engine: RestoreSettings, log: LogFn) -> None:
    assert p.folder is not None
    p.stage = "在整部片上量节奏、黑边和场序（约 20 秒）"
    opts, duration = measure(source, p.options, log)
    p.film_seconds = duration
    p.resolved = describe(opts)
    _check(p)

    p.segments = moments(duration)
    clips: List[Path] = []
    suffix = ".avi" if _is_avi(source) else ".mkv"
    for i, (start, seconds) in enumerate(p.segments):
        p.stage = f"截取片段 {i + 1}/{len(p.segments)}"
        clip = p.folder / f"seg{i}{suffix}"
        cut(source, clip, start, seconds, p.cancel.is_set)
        clips.append(clip)

    units = len(METHODS) * len(clips)
    done = 0
    outputs: Dict[str, List[Path]] = {}
    for method in METHODS:
        update = dict(PREVIEW_ENCODE, ai_model="" if method == "lanczos" else method)
        method_opts = opts.model_copy(update=update)
        totals = {"frames": 0, "wall_s": 0.0, "clip_seconds": 0.0, "engine_frames": 0,
                  "engine_seconds": 0.0, "peak_vram_gb": 0.0, "peak_reserved_gb": 0.0}
        outputs[method] = []
        for i, clip in enumerate(clips):
            p.stage = f"{LABELS[method]}：片段 {i + 1}/{len(clips)}"
            out = p.folder / f"seg{i}.{method}.mkv"

            def progress(fraction, fps, speed, base=done):
                p.progress = ENCODE_SHARE * (base + min(max(fraction, 0.0), 1.0)) / units

            started = time.monotonic()
            result = encode.encode_file(clip, out, EncodeRequest(source=str(clip), options=method_opts),
                                        log=log, progress=progress, should_cancel=p.cancel.is_set,
                                        engine=engine)
            totals["wall_s"] += time.monotonic() - started
            totals["frames"] += result.stats.frames_out
            totals["clip_seconds"] += result.duration
            st = result.engine or {}
            totals["engine_frames"] += int(st.get("frames", 0) or 0)
            totals["engine_seconds"] += float(st.get("seconds", 0.0) or 0.0)
            for key in ("peak_vram_gb", "peak_reserved_gb"):
                totals[key] = max(totals[key], float(st.get(key, 0.0) or 0.0))
            outputs[method].append(out)
            done += 1
            p.progress = ENCODE_SHARE * done / units
            gc.collect()        # an encode's threads sit in reference cycles until collected
        p.methods[method] = _figures(totals, duration)

    p.stage = "拼对比视频"
    flicker = compose(p.folder / "compare.mp4", p.folder / "sheet.jpg", outputs,
                      [s for s, _ in p.segments], p.cancel.is_set,
                      lambda f: setattr(p, "progress", ENCODE_SHARE + (1 - ENCODE_SHARE) * f))
    for method, value in flicker.items():
        p.methods[method]["flicker"] = round(value, 3)
    p.files = {"video": "compare.mp4", "sheet": "sheet.jpg"}


def _figures(totals: dict, film_seconds: float) -> dict:
    """A method's numbers, and the film they come to: the model's own speed
    (decode and encode not counted) is what the engine reports; the frame
    rate out of the deinterlacer is what the clips made per second of film."""
    out_rate = totals["frames"] / totals["clip_seconds"] if totals["clip_seconds"] else 0.0
    film_frames = out_rate * film_seconds
    fig = {"frames": totals["frames"], "wall_s": round(totals["wall_s"], 2),
           "out_fps": round(out_rate, 3)}
    if totals["engine_seconds"]:
        fps = totals["engine_frames"] / totals["engine_seconds"]
        fig.update(model_fps=round(fps, 3), peak_vram_gb=round(totals["peak_vram_gb"], 2),
                   peak_reserved_gb=round(totals["peak_reserved_gb"], 2),
                   film_model_hours=round(film_frames / fps / 3600, 2) if fps else None)
    if totals["frames"] and totals["wall_s"]:
        # everything this run did per frame: decode, filters, model, x264 fast
        fig["film_total_hours"] = round(film_frames * totals["wall_s"] / totals["frames"] / 3600, 2)
    return fig


# --------------------------------------------------------- measuring once

def measure(source: Path, options: EncodeOptions, log: LogFn) -> Tuple[EncodeOptions, float]:
    """The options a real encode of the whole film would settle on, written
    out: cadence and black bars (restore.resolve), and for bob the field
    order — by the very calls encode_file makes on its input."""
    opts = restore.resolve(source, options, log)
    with av.open(str(source)) as container:
        picture = audio.picture_stream(container)
        if picture is None:
            raise PreviewError("片源里没有画面")
        index = picture.index
        duration = float(container.duration / av.time_base) if container.duration else 0.0
    if opts.deinterlace == "bob" and opts.field_order == "auto":
        sample = encode.sample_video(source, index, fields=True)
        if not restore.field_order(sample.fields):
            encode._more_fields(source, index, sample, duration)
        measured = restore.field_order(sample.fields)
        tff, bff = sample.fields
        if measured:
            opts = opts.model_copy(update={"field_order": measured})
            log(f"场序：从整部片测得{'上场' if measured == 'tff' else '下场'}优先（投票 上场 {tff} : 下场 {bff}）")
        else:
            log(f"⚠ 场序：整部片里也测不出（投票 上场 {tff} : 下场 {bff}），按片源的标记")
    if duration <= 0:
        raise PreviewError("读不出片长")
    return opts, duration


def describe(opts: EncodeOptions) -> List[str]:
    """What the segments are run with, as the page shows it."""
    lines = [f"反交错：{encode.DEINTERLACE_NAMES.get(opts.deinterlace, opts.deinterlace)}"]
    if opts.field_order != "auto":
        lines.append(f"场序：{'上场' if opts.field_order == 'tff' else '下场'}优先")
    crop = (opts.crop_top, opts.crop_bottom, opts.crop_left, opts.crop_right)
    if any(crop):
        lines.append("裁边：上 {} 下 {} 左 {} 右 {}".format(*crop))
    if opts.aspect != "auto":
        lines.append(f"画面比例：{opts.aspect}")
    if opts.denoise != "off":
        lines.append(f"降噪：{'轻度' if opts.denoise == 'light' else '强力'}")
    lines.append(f"放大到：{opts.upscale}p")
    return lines


def moments(duration: float) -> List[Tuple[float, float]]:
    """Where to look: POINTS of the way in, SEGMENT_SECONDS each, centred;
    a film too short for that is looked at once, in its middle."""
    seconds = SEGMENT_SECONDS
    if duration < seconds * (len(POINTS) + 1):
        seconds = min(seconds, duration)
        return [(max(0.0, duration / 2 - seconds / 2), seconds)]
    return [(max(0.0, duration * point - seconds / 2), seconds) for point in POINTS]


# ---------------------------------------------------------------- cutting

def _is_avi(path: Path) -> bool:
    try:
        with av.open(str(path)) as c:
            return c.format.name == "avi"
    except Exception:  # noqa: BLE001
        return False


def cut(source: Path, out: Path, start: float, seconds: float,
        should_cancel: Callable[[], bool]) -> None:
    """[start, start + seconds) of the picture by stream copy, from the
    keyframe at or before *start* (seconds into the film, wherever its
    timeline begins). An AVI is cut to AVI: its timestamps are only right
    through mux.FrameClock, which the encode applies to AVI alone."""
    with av.open(str(source)) as src:
        picture = audio.picture_stream(src)
        if picture is None:
            raise PreviewError("片源里没有画面")
        offset = mux.start_offset(src, source)
        with av.open(str(out), "w", format="avi" if out.suffix == ".avi" else "matroska") as dst:
            target = dst.add_stream_from_template(picture)
            target.metadata.update(dict(picture.metadata))
            src.seek(int((offset + start) * av.time_base), backward=True, any_frame=False)
            pipe, zero, base = None, None, 0
            for packet in src.demux(picture):
                if should_cancel():
                    raise InterruptedError
                if not packet.size:
                    continue
                stamp = packet.pts if packet.pts is not None else packet.dts
                t = float(stamp * packet.time_base) if stamp is not None else None
                if zero is None:
                    if not packet.is_keyframe or t is None:
                        continue                  # begin on a keyframe
                    zero, base = t, stamp
                elif t is not None and t > zero + seconds:
                    break
                if packet.pts is not None:
                    packet.pts -= base
                if packet.dts is not None:
                    packet.dts -= base
                if pipe is None:
                    pipe = mux.CopyPipe(target, 0)
                for made in pipe.feed(packet):
                    dst.mux(made)
            if pipe is None:
                raise PreviewError(f"在 {start:.0f} 秒处截不到画面")
            for made in pipe.flush():
                dst.mux(made)


# --------------------------------------------------- the grid and the sheet

def _font(size: int):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)     # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _band(text: str, font) -> np.ndarray:
    from PIL import Image, ImageDraw
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    band = Image.new("RGB", (right - left + 24, bottom - top + 16), (0, 0, 0))
    ImageDraw.Draw(band).text((12 - left, 8 - top), text, fill=(255, 255, 255), font=font)
    return np.asarray(band)


def _timecode(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def compose(video: Path, sheet: Path, outputs: Dict[str, List[Path]], starts: List[float],
            should_cancel: Callable[[], bool], progress: Callable[[float], None]) -> Dict[str, float]:
    """The segments one after another, each a 2×2 grid of the methods at
    their own size (four 1080p panels make 3840×2160); the sheet: one 1:1
    crop per segment where its middle frame has the most detail. Returns
    each method's flicker: the mean change between consecutive frames over
    the pixels the Lanczos version keeps still — the film's own floor."""
    from PIL import Image, ImageDraw

    methods = [m for m in METHODS if m in outputs]
    anchor = methods[0]
    total = sum(_count(outputs[anchor][i]) for i in range(len(starts))) or 1
    sums: Dict[str, List[float]] = {m: [] for m in methods}
    crops: List[Dict[str, np.ndarray]] = []
    out = None
    stream = None
    n = 0
    try:
        for i, start in enumerate(starts):
            containers = [av.open(str(outputs[m][i])) for m in methods]
            try:
                streams = [c.streams.video[0] for c in containers]
                for s in streams:
                    s.thread_type = "AUTO"
                length = min(_count(outputs[m][i]) for m in methods)
                middle = length // 2
                if out is None:
                    rate = Fraction(streams[0].average_rate or Fraction(30000, 1001))
                    w, h = streams[0].codec_context.width, streams[0].codec_context.height
                    font = _font(max(18, h // 28))
                    out = av.open(str(video), "w", options={"movflags": "+faststart"})
                    stream = out.add_stream("libx264", rate=rate)
                    stream.width, stream.height, stream.pix_fmt = w * 2, h * 2, "yuv420p"
                    stream.options = {"crf": str(GRID_CRF), "preset": "fast"}
                    stream.codec_context.time_base = 1 / rate
                    canvas = np.zeros((h * 2, w * 2, 3), np.uint8)
                bands = [_band(f"{LABELS[m]} · {_timecode(start)}", font) for m in methods]
                previous = None
                decoders = [c.decode(s) for c, s in zip(containers, streams)]
                for t in range(length):
                    if should_cancel():
                        raise InterruptedError
                    try:
                        pictures = [next(d).to_ndarray(format="rgb24") for d in decoders]
                    except StopIteration:
                        break
                    for k, (picture, band) in enumerate(zip(pictures, bands)):
                        y, x = (k // 2) * h, (k % 2) * w
                        ph, pw = min(h, picture.shape[0]), min(w, picture.shape[1])
                        canvas[y:y + h, x:x + w] = 0
                        canvas[y:y + ph, x:x + pw] = picture[:ph, :pw]
                        bh, bw = min(band.shape[0], h), min(band.shape[1], w)
                        canvas[y:y + bh, x:x + bw] = band[:bh, :bw]
                    frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
                    frame.pts = n
                    for packet in stream.encode(frame):
                        out.mux(packet)
                    n += 1
                    progress(n / total)
                    if previous is not None:
                        a = pictures[0].astype(np.int16)
                        still = np.abs(a - previous[0].astype(np.int16)).max(axis=2) < 3
                        if still.mean() >= 0.05:
                            for m, now, before in zip(methods, pictures, previous):
                                d = np.abs(now.astype(np.int16) - before.astype(np.int16)).mean(axis=2)
                                sums[m].append(float(d[still].mean()))
                    if t == middle:
                        crops.append(_crops(dict(zip(methods, pictures))))
                    previous = pictures
            finally:
                for c in containers:
                    c.close()
        if stream is not None:
            for packet in stream.encode():
                out.mux(packet)
    finally:
        if out is not None:
            out.close()

    flicker = {m: (float(np.mean(v)) if v else float("nan")) for m, v in sums.items()}
    # the sheet: a header of method + flicker, then a row per segment
    font = _font(18)
    header = 34
    image = Image.new("RGB", (CROP * len(methods), header + CROP * len(crops)), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    for k, m in enumerate(methods):
        draw.text((k * CROP + 8, 8), f"{LABELS[m]}  flicker {flicker[m]:.2f}", fill=(255, 255, 255), font=font)
    for r, row in enumerate(crops):
        for k, m in enumerate(methods):
            image.paste(Image.fromarray(row[m]), (k * CROP, header + r * CROP))
        draw.text((8, header + r * CROP + 8), _timecode(starts[r]), fill=(255, 255, 0), font=font)
    image.save(sheet, quality=90)
    return flicker


def _crops(pictures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The CROP-square of the anchor picture with the most detail (sum of
    horizontal differences), cut from every method at the same place."""
    anchor = next(iter(pictures.values()))
    gray = anchor.mean(axis=2)
    detail = np.abs(np.diff(gray, axis=1))
    h, w = gray.shape
    size = min(CROP, h, w - 1)
    best, at = -1.0, (0, 0)
    for y in range(0, max(1, h - size + 1), max(1, size // 3)):
        for x in range(0, max(1, w - size), max(1, size // 3)):
            v = float(detail[y:y + size, x:x + size - 1].mean())
            if v > best:
                best, at = v, (y, x)
    y, x = at
    out = {}
    for m, picture in pictures.items():
        tile = np.zeros((CROP, CROP, 3), np.uint8)
        part = picture[y:y + size, x:x + size]
        tile[:part.shape[0], :part.shape[1]] = part
        out[m] = tile
    return out


def _count(path: Path) -> int:
    with av.open(str(path)) as c:
        return sum(1 for p in c.demux(c.streams.video[0]) if p.size)
