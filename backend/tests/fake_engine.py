"""A stand-in for restore_engine.worker that needs no PyTorch: the same
protocol, nearest-neighbour x2, held in windows of three frames so the
app's end has to cope with output that lags input (as a real window
model's does). ``{"init": {"model": "fail"}}`` fails the way a broken
engine does; ``"die"`` exits mid-stream without a word; ``"nan"`` stops
mid-stream saying why, as the NaN guard does."""

import sys

import numpy as np

from restore_engine import protocol


def main() -> int:
    out, inp = sys.stdout.buffer, sys.stdin.buffer
    kind, message = protocol.receive(inp)
    model = message["init"]["model"]
    if model == "fail":
        protocol.send_json(out, {"error": "RuntimeError: 显存不足"})
        return 1
    protocol.send_json(out, {"log": "假引擎：加载完成"})
    protocol.send_json(out, {"ready": True, "scale": 2, "device": "fake", "precision": "fp32",
                             "torch": "none", "cuda": None})
    held, count = [], 0
    while True:
        kind, payload = protocol.receive(inp)
        if kind is None:
            return 0
        if kind == b"F":
            count += 1
            if model == "die" and count == 4:
                return 3
            if model == "nan" and count == 4:
                print("UserWarning: torch.meshgrid: in an upcoming release…", file=sys.stderr)
                protocol.send_json(out, {"error": "RuntimeError: 模型输出里出现了 NaN/inf（fp16 溢出？）"})
                return 1
            held.append(payload)
            if len(held) == 3:
                for array in held:
                    protocol.send_frame(out, array.repeat(2, 0).repeat(2, 1))
                held = []
        else:
            for array in held:
                protocol.send_frame(out, array.repeat(2, 0).repeat(2, 1))
            # the shape the real worker reports (Engine.stats)
            protocol.send_json(out, {"done": True, "stats": {
                "frames": count, "seconds": round(count * 0.01, 3), "fps": 100.0,
                "peak_vram_gb": 1.5, "peak_reserved_gb": 2.0}})
            return 0


if __name__ == "__main__":
    sys.exit(main())
