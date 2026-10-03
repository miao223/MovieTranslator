"""The engine process: ``python -m restore_engine.worker``, spoken to over
stdin/stdout (protocol.py).

1. the app sends ``{"init": {...}}``: model id, device, precision, the
   models folder, tuning (chunk / window / overlap / batch);
2. the worker loads the model (downloading its weights the first time) and
   answers ``{"ready": true, "scale": …, "device": …}``;
3. frames go in, upscaled frames come out — in order, one for one, but not
   in lockstep: a window model holds a window's worth before it answers;
4. ``{"end": true}`` flushes what is held and is answered with
   ``{"done": true, "stats": …}``.

Anything that goes wrong is ``{"error": …, "trace": …}`` and the process
exits. Library chatter (download bars, warnings) goes to stderr: stdout is
the protocol and nothing else.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List

import numpy as np

from restore_engine import protocol
from restore_engine.models import MODELS, Spec, loader

LogFn = Callable[[str], None]


def _torch():
    import torch
    return torch


class Engine:
    """Turns frames (HxWx3 uint8/uint16 arrays) into upscaled frames."""

    def __init__(self, spec: Spec, cfg: dict, log: LogFn):
        torch = _torch()
        self.spec, self.log = spec, log
        self.tuning = {**spec.defaults, **{k: int(v) for k, v in (cfg.get("tuning") or {}).items()}}
        wanted = cfg.get("device", "auto")
        if wanted == "auto":
            wanted = "cuda" if torch.cuda.is_available() else "cpu"
        if wanted == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("这个 PyTorch 用不了显卡（CUDA 不可用）。"
                               f"torch {torch.__version__}，CUDA 构建 {torch.version.cuda}")
        self.device = torch.device(wanted)
        precision = cfg.get("precision", "auto")
        if precision == "auto":
            precision = spec.gpu_precision if self.device.type == "cuda" else "fp32"
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                precision = "fp32"                  # pre-Ampere cards have no bf16
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        self.precision = precision
        if self.device.type == "cpu":
            if cfg.get("threads"):
                torch.set_num_threads(int(cfg["threads"]))
            # in RAM rather than VRAM, a batch of 4 SD frames through SPAN took
            # over 6 GB and was OOM-killed: one at a time unless told otherwise
            given = cfg.get("tuning") or {}
            for key, small in (("batch", 1), ("chunk", 4)):
                if key in self.tuning and key not in given:
                    self.tuning[key] = small
        from restore_engine import weights
        folder = Path(cfg.get("models_dir") or Path.home() / ".cache" / "MovieTranslator" / "restore")
        paths = weights.fetch(spec.files, folder, log)
        net = loader(spec)(paths)
        self.net = net.to(self.device, dtype=self.dtype).eval()
        if spec.channels_last and self.device.type == "cuda":
            self.net = self.net.to(memory_format=torch.channels_last)
        self.frames = 0
        self.busy = 0.0
        self.depth = 1
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def info(self) -> dict:
        torch = _torch()
        name = (torch.cuda.get_device_name(self.device) if self.device.type == "cuda"
                else "CPU")
        return {"model": self.spec.id, "scale": self.spec.scale, "device": name,
                "precision": self.precision, "torch": torch.__version__,
                "cuda": torch.version.cuda, "tuning": self.tuning}

    def stats(self) -> dict:
        torch = _torch()
        cuda = self.device.type == "cuda"
        peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if cuda else 0.0
        # what the card actually has to hold: the allocator keeps more than
        # it hands out, and the CUDA context comes on top of both
        reserved = torch.cuda.max_memory_reserved(self.device) / 2**30 if cuda else 0.0
        return {"frames": self.frames, "seconds": round(self.busy, 3),
                "fps": round(self.frames / self.busy, 3) if self.busy else 0.0,
                "peak_vram_gb": round(peak, 3), "peak_reserved_gb": round(reserved, 3)}

    # ---- array <-> tensor

    def _tensor(self, arrays: List[np.ndarray]):
        torch = _torch()
        self.depth = arrays[0].dtype.itemsize
        scale = 255.0 if self.depth == 1 else 65535.0
        x = torch.from_numpy(np.stack(arrays)).to(self.device)
        x = x.permute(0, 3, 1, 2).to(self.dtype).div_(scale)
        return x                               # (n, 3, h, w)

    def _arrays(self, y) -> List[np.ndarray]:
        torch = _torch()
        top = 255.0 if self.depth == 1 else 65535.0
        with torch.inference_mode():
            if not bool(torch.isfinite(y).all()):
                # NaN survives clamp and casts to garbage: a black film that
                # every count and timestamp check would pass
                raise RuntimeError(f"模型输出里出现了 NaN/inf（{self.precision} 溢出？），"
                                   f"请把计算精度改成 bf16 或 fp32 再试")
        with torch.inference_mode():       # y came out of inference mode: no in-place ops outside it
            y = y.float().clamp(0, 1).mul(top).round()
            y = y.to(torch.uint8 if self.depth == 1 else torch.int32).permute(0, 2, 3, 1).cpu().numpy()
        if self.depth != 1:
            y = y.astype(np.uint16)
        return list(y)

    def _pad(self, x):
        """Reflect-pad h and w up to the model's multiple, at the top and left
        as RealViformer's own script does; returns (x, pad_h, pad_w)."""
        import torch.nn.functional as F
        m = self.spec.pad
        h, w = x.shape[-2:]
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (pw, 0, ph, 0), mode="reflect")       # x is (n, 3, h, w)
        return x, ph, pw

    def _run(self, fn, frames: int):
        torch = _torch()
        started = time.perf_counter()
        with torch.inference_mode():
            y = fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.busy += time.perf_counter() - started
        self.frames += frames
        return y

    def push(self, array: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError

    def flush(self) -> List[np.ndarray]:
        raise NotImplementedError


class FrameEngine(Engine):
    def __init__(self, spec, cfg, log):
        super().__init__(spec, cfg, log)
        self.held: List[np.ndarray] = []

    def _go(self) -> List[np.ndarray]:
        held, self.held = self.held, []
        if not held:
            return []
        x, ph, pw = self._pad(self._tensor(held))
        y = self._run(lambda: self.net(x), len(held))
        s = self.spec.scale
        return self._arrays(y[..., ph * s:, pw * s:])

    def push(self, array):
        self.held.append(array)
        return self._go() if len(self.held) >= self.tuning.get("batch", 1) else []

    def flush(self):
        return self._go()


class StreamEngine(Engine):
    def __init__(self, spec, cfg, log):
        super().__init__(spec, cfg, log)
        from restore_engine.models import realviformer
        self.stream = realviformer.Stream(self.net)
        self.held: List[np.ndarray] = []

    def _go(self) -> List[np.ndarray]:
        held, self.held = self.held, []
        if not held:
            return []
        x, ph, pw = self._pad(self._tensor(held))
        y = self._run(lambda: self.stream(x.unsqueeze(0))[0], len(held))
        s = self.spec.scale
        return self._arrays(y[..., ph * s:, pw * s:])

    def push(self, array):
        self.held.append(array)
        return self._go() if len(self.held) >= self.tuning.get("chunk", 8) else []

    def flush(self):
        return self._go()


class WindowEngine(Engine):
    """Bidirectional: each window of `window` frames is run with `overlap`
    frames of context on either side, and only its middle is kept."""

    def __init__(self, spec, cfg, log):
        super().__init__(spec, cfg, log)
        self.window = max(1, self.tuning.get("window", 30))
        self.overlap = max(0, self.tuning.get("overlap", 4))
        self.frames_in: List[np.ndarray] = []    # from first - kept_before on
        self.first = 0                           # index of the next frame to emit
        self.base = 0                            # index of frames_in[0]

    def _process(self, end: int) -> List[np.ndarray]:
        start = max(self.base, self.first - self.overlap)
        stop = min(self.base + len(self.frames_in), end + self.overlap)
        chunk = self.frames_in[start - self.base:stop - self.base]
        x, ph, pw = self._pad(self._tensor(chunk))
        y = self._run(lambda: self.net(x.unsqueeze(0))[0], end - self.first)
        s = self.spec.scale
        y = y[self.first - start:end - start, :, ph * s:, pw * s:]
        out = self._arrays(y)
        self.first = end
        keep_from = max(self.base, self.first - self.overlap)
        self.frames_in = self.frames_in[keep_from - self.base:]
        self.base = keep_from
        return out

    def push(self, array):
        self.frames_in.append(array)
        have = self.base + len(self.frames_in)
        if have >= self.first + self.window + self.overlap:
            return self._process(self.first + self.window)
        return []

    def flush(self):
        out = []
        have = self.base + len(self.frames_in)
        while self.first < have:
            out.extend(self._process(min(have, self.first + self.window)))
        return out


ENGINES = {"frame": FrameEngine, "stream": StreamEngine, "window": WindowEngine}


def make_engine(cfg: dict, log: LogFn) -> Engine:
    model = cfg.get("model", "")
    if model not in MODELS:
        raise ValueError(f"不认识的模型：{model}（有：{'、'.join(MODELS)}）")
    spec = MODELS[model]
    return ENGINES[spec.kind](spec, cfg, log)


def main() -> int:
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                     # the pipe carries the protocol only
    inp = sys.stdin.buffer

    def log(message: str) -> None:
        protocol.send_json(out, {"log": message})

    def fail(exc: BaseException) -> int:
        protocol.send_json(out, {"error": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc()})
        return 1

    kind, message = protocol.receive(inp)
    if kind != b"J" or "init" not in (message or {}):
        return 2
    try:
        engine = make_engine(message["init"], log)
    except BaseException as exc:  # noqa: BLE001 — reported, then the process ends
        return fail(exc)
    protocol.send_json(out, {"ready": True, **engine.info})
    try:
        while True:
            kind, payload = protocol.receive(inp)
            if kind is None:
                return 0                        # the app went away: nothing to answer
            if kind == b"F":
                for array in engine.push(payload):
                    protocol.send_frame(out, array)
            elif payload.get("end"):
                for array in engine.flush():
                    protocol.send_frame(out, array)
                protocol.send_json(out, {"done": True, "stats": engine.stats()})
                return 0
    except BaseException as exc:  # noqa: BLE001
        return fail(exc)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.exit(main())
