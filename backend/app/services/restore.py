"""修复: what the restore page adds to a re-encode (services/encode.py).

A restore is an encode with more done to the picture first, so it runs in
the same engine, the same queue entry kind and the same verify/replace path
— only these steps are its own:

* **bob deinterlacing** to 59.94 / 50 frames a second (encode's
  ``deinterlace="bob"``). A camera-shot interlaced recording holds that many
  real pictures, one per field; nothing is invented. Which field came first
  is measured from the picture here (:func:`field_order`), because an analog
  capture's flags are often absent — a lossless AVI capture (Huffyuv,
  Lagarith) has no field flags at all — and bwdif then takes every frame as
  top-first, which plays a bottom-first capture backwards pair by pair. A
  still first look is followed by a longer, later one (encode._more_fields).
* **crop** and **denoise**, ffmpeg's own filters (:func:`picture_chain`).
* **enlarging**, by Lanczos in the filter graph, or by a model through a
  :class:`FrameStage` between the filter graph and the encoder
  (:func:`make_stage`). Which model, and on what runtime, is decided by
  measurement on real tapes and discs before anything is wired in; until
  then there is none, and the hook exists so the plumbing around it — sizes,
  timestamps, the second filter graph — is tested on its own.
"""

from __future__ import annotations

import collections
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Deque, List, Optional, Protocol, Sequence, Tuple

import av
import numpy as np
from av.video.format import VideoFormat

from app.models.schemas import EncodeOptions

LogFn = Callable[[str], None]

# a pair of fields counts as a vote when one ordering is clearly nearer in
# time than the other; a still picture votes for neither
VOTE_MARGIN = 0.9
# a verdict needs this many votes, and this many times the other side's
MIN_VOTES = 8
VOTE_RATIO = 3

# the 修复 fields of EncodeOptions, and what each one is when it is "not used"
RESTORE_DEFAULTS = {
    "field_order": "auto",
    "crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 0,
    "crop_auto": False,
    "aspect": "auto",
    "denoise": "off",
    "upscale": 0,
    "ai_model": "",
}

DENOISE = {
    # temporal: averages each pixel with its neighbours in time where they
    # agree — tape noise goes, edges in motion stay
    "light": [("atadenoise", "")],
    # FFT over the frame and its two neighbours: takes out more, softens more
    "strong": [("fftdnoiz", "sigma=8:prev=1:next=1")],
}
DENOISE_NAMES = {"light": "轻度降噪（时域）", "strong": "强力降噪（频域 + 时域）"}


def restoring(opts: EncodeOptions) -> bool:
    """Whether *opts* asks for anything beyond a plain encode."""
    return (opts.deinterlace in ("bob", "match", "detect")
            or any(getattr(opts, k) != v for k, v in RESTORE_DEFAULTS.items()
                   if k != "field_order"))


def cropped(opts: EncodeOptions) -> bool:
    return any((opts.crop_top, opts.crop_bottom, opts.crop_left, opts.crop_right))


def picture_chain(opts: EncodeOptions, width: int, height: int) -> Tuple[List[Tuple[str, str]], int, int]:
    """(filters, width, height after them) for the crop and denoise steps,
    which go right after deinterlacing."""
    chain: List[Tuple[str, str]] = []
    w = width - opts.crop_left - opts.crop_right
    h = height - opts.crop_top - opts.crop_bottom
    if cropped(opts):
        if w < 16 or h < 16:
            raise ValueError(f"裁边太多：{width}×{height} 的画面裁完只剩 {w}×{h}")
        chain.append(("crop", f"{w}:{h}:{opts.crop_left}:{opts.crop_top}"))
    chain.extend(DENOISE.get(opts.denoise, []))
    return chain, w, h


# ------------------------------------------------------------ field order


class FieldVotes:
    """Which field was shot first, voted on by consecutive frames.

    Top field first means a frame's bottom field and the next frame's top
    field are neighbours in time (… T0 B0 | T1 B1 …); bottom first makes the
    top field and the next bottom one neighbours. The pair nearer in time
    differs less. Each comparison is between rows one line apart, which
    biases both sides the same way. Frames are fed one at a time so a
    sample of a 1080i film is never held whole.
    """

    def __init__(self):
        self.tff = self.bff = 0
        self.prev = None

    def add(self, gray: np.ndarray) -> None:
        h = gray.shape[0] // 2 * 2
        top, bottom = gray[0:h:2].astype(np.float32), gray[1:h:2].astype(np.float32)
        prev, self.prev = self.prev, (top, bottom)
        if prev is None or prev[0].shape != top.shape:
            return
        after_bottom = float(np.abs(prev[1] - top).mean())
        after_top = float(np.abs(prev[0] - bottom).mean())
        if after_bottom < after_top * VOTE_MARGIN:
            self.tff += 1
        elif after_top < after_bottom * VOTE_MARGIN:
            self.bff += 1

    @property
    def votes(self) -> Tuple[int, int]:
        return self.tff, self.bff


def field_votes(frames: Sequence[np.ndarray]) -> Tuple[int, int]:
    """(top-first votes, bottom-first votes) over consecutive grey frames."""
    counter = FieldVotes()
    for gray in frames:
        counter.add(gray)
    return counter.votes


def field_order(votes: Tuple[int, int]) -> str:
    """"tff", "bff", or "" when the picture does not say."""
    tff, bff = votes
    if tff >= MIN_VOTES and tff >= VOTE_RATIO * max(bff, 1):
        return "tff"
    if bff >= MIN_VOTES and bff >= VOTE_RATIO * max(tff, 1):
        return "bff"
    return ""


# ------------------------------------------------------------- the model


class FrameStage(Protocol):
    """A step that works on whole frames between the filter graph and the
    encoder — the place an enlarging model goes.

    The same contract as encode.VideoFilter: ``push`` takes one RGB frame
    (rgb24, or rgb48le for a 10-bit / HDR source) and returns the frames
    that are ready, possibly none while it gathers a batch; ``flush`` returns
    the rest at the end. Every frame it returns keeps the pts and time base
    of the frame it was made from: the second filter graph after it scales
    to the encoder's size and format, and the timestamps go straight on.
    """

    name: str

    def push(self, frame) -> list: ...

    def flush(self) -> list: ...

    def close(self) -> None:
        """Called whatever happens — done, failed or cancelled; an engine
        process must not outlive its encode."""


# what the 修复 page offers, in the order offered; the engine knows the rest
# (restore_engine/models). Kept here as plain data: the app never imports
# the engine's model code, which needs PyTorch.
AI_MODELS = {
    "realviformer": "RealViformer",
    "realbasicvsr": "RealBasicVSR",
    "liveaction_span": "2xLiveActionV1_SPAN",
}

BACKEND = Path(__file__).resolve().parents[2]


class EngineError(RuntimeError):
    pass


class EngineClient:
    """One engine process (``python -m restore_engine.worker``) and its pipe.

    A reader thread drains the engine's output the whole time. Without it
    the two ends block each other: the engine stops reading frames while its
    output pipe is full, and the app stops reading output while it is still
    writing frames — measured, with a window model, on the first try.
    """

    def __init__(self, python: str, init: dict, log: LogFn,
                 module: str = "restore_engine.worker", cwd: Optional[Path] = None):
        if not python or not Path(python).exists():
            raise EngineError(f"修复引擎的 Python 不存在：{python or '（没有配置）'}。"
                              f"请在「设置 → 修复引擎」里填写")
        self.log = log
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([python, "-m", module], cwd=str(cwd or BACKEND),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, creationflags=flags)
        self.messages: "queue.Queue" = queue.Queue()
        self.stderr: Deque[str] = collections.deque(maxlen=30)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.stats: dict = {}
        self._send(lambda: _protocol().send_json(self.proc.stdin, {"init": init}))
        self.info = self._until("ready")

    def _read(self) -> None:
        proto = _protocol()
        while True:
            try:
                kind, payload = proto.receive(self.proc.stdout)
            except Exception as exc:  # noqa: BLE001 — a torn pipe ends the conversation
                kind, payload = b"J", {"error": f"读不懂引擎的输出：{exc}"}
            self.messages.put((kind, payload))
            if kind is None or (kind == b"J" and ("error" in payload or "done" in payload)):
                return

    def _read_stderr(self) -> None:
        for raw in self.proc.stderr:
            self.stderr.append(raw.decode("utf-8", "replace").rstrip())

    def _why(self) -> str:
        tail = [line for line in self.stderr if line.strip()][-5:]
        return "；".join(tail) or f"引擎进程退出（返回码 {self.proc.poll()}）"

    def _send(self, write) -> None:
        try:
            write()
        except (BrokenPipeError, OSError):
            # the engine died. If it said why, that is on the queue once the
            # reader has read it — not in stderr, whose tail is library
            # warnings (a NaN stop was once reported as a meshgrid warning)
            self.reader.join(timeout=5)
            for kind, payload in self._drain(block=False):
                if kind == b"J" and "error" in payload:
                    raise EngineError(f"修复引擎出错：{payload['error']}") from None
            raise EngineError(f"修复引擎中途退出：{self._why()}") from None

    def _handle(self, kind, payload, frames: list, want: str = "") -> bool:
        """True when *want* arrived."""
        if kind is None:
            raise EngineError(f"修复引擎中途退出：{self._why()}")
        if kind == b"F":
            frames.append(payload)
            return False
        if "log" in payload:
            self.log(payload["log"])
            return False
        if "error" in payload:
            raise EngineError(f"修复引擎出错：{payload['error']}")
        if "done" in payload:
            self.stats = payload.get("stats") or {}
        return bool(want) and want in payload

    def _drain(self, block: bool):
        while True:
            try:
                yield self.messages.get(block=block, timeout=None)
            except queue.Empty:
                return

    def _until(self, want: str) -> dict:
        frames: list = []
        while True:
            kind, payload = self.messages.get()
            if self._handle(kind, payload, frames, want):
                return payload

    def send(self, array) -> None:
        self._send(lambda: _protocol().send_frame(self.proc.stdin, array))

    def take(self) -> list:
        """Whatever upscaled frames have arrived, without waiting."""
        frames: list = []
        for kind, payload in self._drain(block=False):
            self._handle(kind, payload, frames)
        return frames

    def finish(self) -> list:
        """Flush the engine and wait for every frame it still owes."""
        frames: list = []
        self._send(lambda: _protocol().send_json(self.proc.stdin, {"end": True}))
        while True:
            kind, payload = self.messages.get()
            if self._handle(kind, payload, frames, "done"):
                return frames

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _protocol():
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from restore_engine import protocol
    return protocol


class EngineStage:
    """A FrameStage backed by the engine process: frames out as RGB arrays,
    upscaled arrays back as frames carrying the timestamps of the frames
    they were made from (the engine never sees a timestamp)."""

    def __init__(self, client: EngineClient, name: str):
        self.client, self.name = client, name
        self.pending: Deque[Tuple[int, object, str]] = collections.deque()

    def _frames(self, arrays: list) -> list:
        made = []
        for array in arrays:
            pts, time_base, fmt = self.pending.popleft()
            frame = av.VideoFrame.from_ndarray(array, format=fmt)
            frame.pts, frame.time_base = pts, time_base
            made.append(frame)
        return made

    def push(self, frame) -> list:
        fmt = frame.format.name
        self.pending.append((frame.pts, frame.time_base, fmt))
        self.client.send(frame.to_ndarray())
        return self._frames(self.client.take())

    def flush(self) -> list:
        return self._frames(self.client.finish())

    def close(self) -> None:
        self.client.close()

    @property
    def stats(self) -> dict:
        """The engine's own numbers once it has said done (empty before)."""
        return dict(self.client.stats or {})

    def summary(self) -> str:
        """One line for the log once the engine has said done: how fast the
        model itself went (decode and encode not counted) and its peak VRAM."""
        st = self.client.stats
        if not st:
            return ""
        line = f"修复引擎：{st.get('frames', 0)} 帧，模型部分 {st.get('fps', 0):.2f} 帧/秒"
        if st.get("peak_vram_gb"):
            line += f"，显存峰值 {st['peak_vram_gb']:.2f} GB"
            if st.get("peak_reserved_gb"):
                line += f"（占用 {st['peak_reserved_gb']:.2f} GB）"
        return line


def make_stage(opts: EncodeOptions, log: LogFn, engine=None) -> Optional[FrameStage]:
    """The model this encode enlarges with (opts.ai_model, run by the engine
    *engine* names — a RestoreSettings), or None for Lanczos."""
    if not opts.ai_model:
        return None
    if engine is None:
        raise EngineError("这个压制选了 AI 放大，但没有修复引擎的设置")
    from app.services import asr
    cache = asr.get_model_cache_dir()          # None: no model folder set
    # empty: the engine's own default (<home>/.cache/MovieTranslator/restore)
    models_dir = engine.models_dir or (str(Path(cache) / "restore") if cache else "")
    client = EngineClient(engine.engine_python, {
        "model": opts.ai_model, "device": engine.device, "precision": engine.precision,
        "models_dir": models_dir,
    }, log)
    info = client.info
    log(f"修复引擎：{AI_MODELS.get(opts.ai_model, opts.ai_model)}，{info.get('device')}，"
        f"{info.get('precision')}，torch {info.get('torch')}（CUDA {info.get('cuda') or '无'}）")
    return EngineStage(client, AI_MODELS.get(opts.ai_model, opts.ai_model))


def name_tags(opts: EncodeOptions, rate: Optional[float]) -> List[str]:
    """What a restore adds to the name of an encode written next to its
    source: 片名.1080p.60fps.HEVC.mkv. *rate* is the source's frame rate as
    its header states it."""
    tags = []
    if opts.upscale:
        tags.append(f"{opts.upscale}p")
    if opts.deinterlace == "bob" and rate and rate <= BOB_MAX_RATE:
        tags.append(f"{round(rate * 2)}fps")
    return tags


# bob makes sense for interlaced material at 25 / 29.97 / 30 frames a second;
# anything faster already has a picture per field
BOB_MAX_RATE = 31.0


# --------------------------------------------------------- what the disc is
#
# Every DVD flags its frames interlaced, and that says nothing: measured on
# the user's own discs, one was 24p film hard-telecined (every 5th and 4th
# frame combed, all of it recoverable), one 30p video stored a field out of
# step (every frame combed, all recoverable by field matching alone), three
# truly interlaced (nothing recoverable) and one a film whose fields were
# blended in a standards conversion (combed 4 frames in 5, nothing
# recoverable). Each needs a different mode, and the wrong one costs either
# real motion (single-rate on true interlace), real frames (decimating 30p),
# twice the frames for nothing (bob on 30p) or ghosts sharpened by a model.

CADENCE_NAMES = {
    "progressive": "逐行",
    "telecine": "24p 硬过带（胶片拍摄）",
    "shifted": "30p 逐行、两场错开一场",
    "interlaced": "真隔行（摄像机拍摄）",
    "mixed": "多数是 30p 错场，夹着真隔行的片段",
    "blended": "疑似混场转制",
}
CADENCE_MODES = {
    "progressive": "off", "telecine": "ivtc", "shifted": "match",
    "interlaced": "bob", "mixed": "match", "blended": "bob",
}

# a pixel is combed when it differs from both lines around it the same way
# while those two agree with each other; a frame is combed when a 16x16
# block holds this many such pixels. Counting "differs from both neighbours"
# alone is fooled by fine horizontal detail, text and grain — a first
# version called a cleanly telecined disc interlaced that way.
COMB_DIFF = 25
COMB_FLAT = 12
COMB_BLOCK = 60
# where fewer frames than this are combed, there is no interlacing to undo
PROGRESSIVE_BELOW = 0.05
# ...but only where the picture moves: a still picture shows no combing
# whatever it is (measured: a 2 s cut of a telecined film, all of it one
# held shot, read 0% combed and was called progressive). Mean absolute
# difference between frames, quarter size, 0–255.
STILL_BELOW = 1.0
# field matching "recovers" a disc when it takes away at least this share
# of the combing it was given, and recovers a good part of it from this
# share on: a 2010 TV omnibus measured 57% — 30p drama with interlaced
# segments between, which field matching serves better than bob (the 30p
# keeps every line; bob would only show each picture twice, softer)
RECOVERED = 0.8
PARTLY_RECOVERED = 0.4


def combed(gray: np.ndarray) -> bool:
    y = gray.astype(np.float32)
    a, b, c = y[:-2], y[1:-1], y[2:]
    mask = ((b - a) * (b - c) > COMB_DIFF * COMB_DIFF) & (np.abs(a - c) < COMB_FLAT)
    h, w = mask.shape
    h, w = h - h % 16, w - w % 16
    if not h or not w:
        return False
    blocks = mask[:h, :w].reshape(h // 16, 16, w // 16, 16).sum(axis=(1, 3))
    return bool(blocks.max() > COMB_BLOCK)


def five_cycle(pattern: Sequence[bool]) -> bool:
    """Whether combing comes and goes with a 5-frame period — telecine's
    two combed frames in five, a blended conversion's four."""
    if len(pattern) < 25:
        return False
    flags = np.array(pattern, dtype=np.float32)
    share = flags.mean()
    if share < 0.15 or share > 0.95:
        return False
    phases = [flags[p::5].mean() for p in range(5)]
    return bool(max(phases) - min(phases) > 0.6)


@dataclass
class Cadence:
    kind: str = ""                 # a CADENCE_NAMES key; "" when the picture never moved enough
    combed: float = 0.0            # share of the sampled frames combed as decoded
    matched: float = 0.0           # the same after field matching, better field order of the two
    cycle: bool = False
    frames: int = 0
    votes: Tuple[int, int] = (0, 0)

    @property
    def name(self) -> str:
        return CADENCE_NAMES.get(self.kind, "测不出")

    @property
    def mode(self) -> str:
        return CADENCE_MODES.get(self.kind, "")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "mode": self.mode,
                "combed": round(self.combed, 2), "matched": round(self.matched, 2),
                "cycle": self.cycle, "frames": self.frames}


def classify(combed_share: float, matched_share: float, cycle: bool, frames: int,
             motion: float = 99.0) -> str:
    if frames < 60 or motion < STILL_BELOW:
        return ""
    if combed_share < PROGRESSIVE_BELOW:
        return "progressive"
    recovered = 1 - matched_share / combed_share
    if recovered >= RECOVERED:
        return "telecine" if cycle else "shifted"
    if recovered >= PARTLY_RECOVERED:
        return "telecine" if cycle else "mixed"
    return "blended" if cycle else "interlaced"


class _Graph:
    """One filter graph, built from the first frame it is given."""

    def __init__(self, chain: List[Tuple[str, str]], time_base, rate: Fraction):
        self.chain, self.time_base, self.rate = chain, time_base, rate
        self.graph = None

    def push(self, frame) -> list:
        if self.graph is None:
            graph = av.filter.Graph()
            src = graph.add(
                "buffer", video_size=f"{frame.width}x{frame.height}",
                pix_fmt=str(int(VideoFormat(frame.format.name))),
                time_base=str(self.time_base), pixel_aspect="1/1",
                frame_rate=f"{self.rate.numerator}/{self.rate.denominator}")
            nodes = [src] + [graph.add(name, args) for name, args in self.chain]
            sink = graph.add("buffersink")
            graph.link_nodes(*nodes, sink)
            graph.configure()
            self.graph, self.src, self.sink = graph, src, sink
        self.src.push(frame)
        out = []
        while True:
            try:
                out.append(self.sink.pull())
            except (av.error.BlockingIOError, av.error.EOFError):
                return out


def _busiest(container, stream, positions: int, keep: int) -> List[Tuple[float, int]]:
    """(motion, seek point) for the *keep* seek points (of *positions* spread
    over the film) where the picture moves most steadily — the median, so one
    cut does not count."""
    found = []
    duration = container.duration or 0
    for k in range(1, positions + 1):
        at = int(duration * k / (positions + 1))
        try:
            container.seek(at)
        except av.error.FFmpegError:
            continue
        prev, moves = None, []
        for frame in container.decode(stream):
            y = frame.to_ndarray(format="gray")[::4, ::4].astype(np.int16)
            if prev is not None:
                moves.append(float(np.abs(y - prev).mean()))
            prev = y
            if len(moves) >= 16:
                break
        if moves:
            found.append((float(np.median(moves)), at))
    found.sort(reverse=True)
    return found[:keep]


def analyze_cadence(path: str | Path, index: int, segments: int = 5,
                    frames: int = 120) -> Cadence:
    """What the picture is, from where it moves most: as decoded, and after
    field matching in either field order (the better one counts)."""
    raw: List[bool] = []
    matched = {"tff": [], "bff": []}
    patterns: List[List[bool]] = []
    votes = FieldVotes()
    with av.open(str(path)) as container:
        stream = container.streams[index]
        stream.thread_type = "AUTO"
        rate = Fraction(stream.guessed_rate or stream.average_rate or 30000 / 1001
                        ).limit_denominator(1001)
        busiest = _busiest(container, stream, 30, segments)
        motion = max((m for m, _ in busiest), default=0.0)
        for _, at in busiest:
            container.seek(at)
            graphs = {order: _Graph([("fieldmatch", f"order={order}:combmatch=full")],
                                    stream.time_base, rate) for order in matched}
            pattern: List[bool] = []
            votes.prev = None
            n = 0
            for frame in container.decode(stream):
                gray = frame.to_ndarray(format="gray")
                pattern.append(combed(gray))
                votes.add(gray)
                for order, graph in graphs.items():
                    for out in graph.push(frame):
                        matched[order].append(combed(out.to_ndarray(format="gray")))
                n += 1
                if n >= frames:
                    break
            raw.extend(pattern)
            patterns.append(pattern)
    if not raw:
        return Cadence()
    combed_share = float(np.mean(raw))
    matched_share = min(float(np.mean(v)) if v else 1.0 for v in matched.values())
    cycling = [five_cycle(p) for p in patterns if p and 0.15 <= np.mean(p) <= 0.95]
    cycle = bool(cycling) and sum(cycling) * 2 > len(cycling)
    return Cadence(kind=classify(combed_share, matched_share, cycle, len(raw), motion),
                   combed=combed_share, matched=matched_share, cycle=cycle,
                   frames=len(raw), votes=votes.votes)


# ---------------------------------------------------------- black borders

DARK = 30            # a row or column whose mean is below this is black
MIN_BORDER = 4       # thinner than this is the picture's own edge, not a bar


def _dark_run(means: np.ndarray) -> int:
    run = 0
    for value in means:
        if value >= DARK:
            break
        run += 1
    return run


def detect_borders(path: str | Path, index: int, positions: int = 24,
                   per: int = 6) -> Tuple[int, int, int, int]:
    """(top, bottom, left, right) black bars, in pixels of the coded frame.

    Only bright frames vote, and a bar is as thin as it is on the brightest
    of them: a dark scene is black to the edge without being letterboxed.
    """
    runs = []
    with av.open(str(path)) as container:
        stream = container.streams[index]
        stream.thread_type = "AUTO"
        height, width = stream.codec_context.height, stream.codec_context.width
        duration = container.duration or 0
        for k in range(1, positions + 1):
            try:
                container.seek(int(duration * k / (positions + 1)))
            except av.error.FFmpegError:
                continue
            n = 0
            for frame in container.decode(stream):
                y = frame.to_ndarray(format="gray").astype(np.float32)
                n += 1
                if y.mean() > 40:
                    rows, cols = y.mean(axis=1), y.mean(axis=0)
                    runs.append((_dark_run(rows), _dark_run(rows[::-1]),
                                 _dark_run(cols), _dark_run(cols[::-1])))
                if n >= per:
                    break
    if not runs:
        return (0, 0, 0, 0)
    least = np.min(np.array(runs), axis=0)
    limits = (height // 3, height // 3, width // 3, width // 3)
    return tuple(int(v) // 2 * 2 if MIN_BORDER <= v <= limit else 0
                 for v, limit in zip(least, limits))


def missing_aspect(width: int, height: int, sar) -> bool:
    """A standard-definition frame that claims square pixels: almost always a
    capture or re-encode that dropped the flag (720x576 would show 5:4)."""
    return (not sar or Fraction(sar) == 1) and height in (480, 486, 576) and width in (704, 720)


def aspect_sar(aspect: str, width: int, height: int) -> Optional[Fraction]:
    """The sample aspect ratio that makes the whole coded frame *aspect*."""
    if aspect == "4:3":
        return Fraction(4, 3) * Fraction(height, width)
    if aspect == "16:9":
        return Fraction(16, 9) * Fraction(height, width)
    return None


def analyze(path: str | Path) -> dict:
    """What the 修复 page shows about one file, and what it suggests."""
    from app.services import audio
    with av.open(str(path)) as container:
        picture = audio.picture_stream(container)
        if picture is None:
            raise ValueError("片源里没有画面")
        index = picture.index
        cc = picture.codec_context
        width, height = cc.width, cc.height
        sar = picture.sample_aspect_ratio
    cadence = analyze_cadence(path, index)
    borders = detect_borders(path, index)
    return {
        "cadence": cadence.as_dict(),
        "borders": list(borders),
        "aspect_missing": missing_aspect(width, height, sar),
        "width": width, "height": height,
    }


def resolve(source: str | Path, opts: EncodeOptions, log: LogFn) -> EncodeOptions:
    """*opts* with "measure it" turned into what was measured: the cadence's
    deinterlace mode for ``detect``, the bars added to the crop for
    ``crop_auto``. Run before the output is named — a bob'd encode is named
    for its frame rate — and logged, so a wrong guess can be read back."""
    if opts.deinterlace != "detect" and not opts.crop_auto:
        return opts
    from app.services import audio
    with av.open(str(source)) as container:
        picture = audio.picture_stream(container)
        if picture is None:
            return opts
        index = picture.index
        flagged = None
        for frame in container.decode(picture):
            flagged = bool(frame.interlaced_frame)
            break
    update: dict = {}
    if opts.deinterlace == "detect":
        cadence = analyze_cadence(source, index)
        mode = cadence.mode
        if mode:
            log(f"片源节奏：{cadence.name}（梳齿帧 {cadence.combed:.0%}，场匹配后 "
                f"{cadence.matched:.0%}，抽样 {cadence.frames} 帧）→ "
                + {"off": "不反交错", "ivtc": "反胶片过带", "match": "只做场匹配",
                   "bob": "还原 60 帧"}[mode])
        else:
            mode = "bob" if flagged else "off"
            log("⚠ 片源节奏：画面里动得太少，测不出——按"
                + ("隔行处理（还原 60 帧）" if flagged else "逐行处理"))
        if cadence.kind == "blended":
            log("⚠ 疑似混场转制：胶片转录像带时场被混在一起，找不回原来的帧。"
                "画面里会有重影，AI 放大会把重影一起锐化")
        update["deinterlace"] = mode
    if opts.crop_auto:
        top, bottom, left, right = detect_borders(source, index)
        if any((top, bottom, left, right)):
            log(f"检测到黑边：上 {top} 下 {bottom} 左 {left} 右 {right}，一并裁掉")
        else:
            log("没有检测到黑边")
        update.update(crop_top=opts.crop_top + top, crop_bottom=opts.crop_bottom + bottom,
                      crop_left=opts.crop_left + left, crop_right=opts.crop_right + right,
                      crop_auto=False)
    return opts.model_copy(update=update)
