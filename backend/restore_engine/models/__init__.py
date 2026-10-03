"""The models the engine can run, and how each one wants its frames.

kind "frame":  one picture at a time (batched for the GPU) — a single-image
               model; nothing carries from frame to frame.
kind "stream": causal, state carried forward — fed in pieces of any length.
kind "window": looks both ways in time (bidirectional) — run on overlapping
               windows; the frames at each edge are context only, and only
               the middle of each window is kept, so no seam is a hard cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    scale: int
    kind: str
    files: List[dict]                  # what weights.fetch downloads, in order
    licence: str
    note: str = ""
    pad: int = 1                       # input sides must be multiples of this
    defaults: Dict[str, int] = field(default_factory=dict)
    gpu_precision: str = "fp16"        # what precision "auto" means on a GPU
    channels_last: bool = False        # NHWC on a GPU: faster where measured so


MODELS: Dict[str, Spec] = {
    "realviformer": Spec(
        id="realviformer", name="RealViformer", scale=4, kind="stream", pad=4,
        files=[{"file": "realviformer_weights.pth", "sha256": "25d49a0128b1ecf2",
                "urls": ["gdrive:1XM-A59MUeSoT0vZ2NCvLVYfvDXnwNGSG"]}],
        licence="MIT（Yuehan Zhang, RealViformer, ECCV 2024）",
        note="视频模型（只看之前的帧），闪烁最少",
        defaults={"chunk": 8},
        # fp16 overflows (NaN from the eighth frame of a real DVD); bf16 runs
        # but drifts from fp32 by 3.7 levels on average (99th percentile 27,
        # RTX 5070 Ti, 2026-10-03) through the state it carries. fp32 is what
        # the model computes: 2.6 fps, 9.3 GB — the user's choice, quality first
        gpu_precision="fp32"),
    "realbasicvsr": Spec(
        id="realbasicvsr", name="RealBasicVSR", scale=4, kind="window",
        files=[{"file": "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth",
                "sha256": "52f77c2c835aaa3f",
                "urls": ["https://download.openmmlab.com/mmediting/restorers/real_basicvsr/"
                         "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth"]}],
        licence="Apache-2.0（OpenMMLab mmagic / RealBasicVSR, CVPR 2022）",
        note="视频模型（前后都看），按窗口处理",
        defaults={"window": 30, "overlap": 4},
        # 2.03 → 2.72 fps and 9.5 → 7.9 GB on an RTX 5070 Ti, as far from fp32
        # as without it (mean 0.14 levels); RealViformer got slower, SPAN no change
        channels_last=True),
    "liveaction_span": Spec(
        id="liveaction_span", name="2xLiveActionV1_SPAN", scale=2, kind="frame",
        files=[{"file": "2xLiveActionV1_SPAN_490000.pth",
                "sha256": "8b166c75831ea7f694d9058ee9c8df8148af8cc1d2b57e69e6581b15cab572f7",
                "urls": ["https://raw.githubusercontent.com/jcj83429/upscaling/"
                         "f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/"
                         "2xLiveActionV1_SPAN_490000.pth"]}],
        licence="CC-BY-NC-SA-4.0（jcj83429）：仅限非商业使用",
        note="单帧模型，专门修 DVD 的压缩、光晕、色度，不降噪",
        defaults={"batch": 4}),
}


def loader(spec: Spec) -> Callable[[List[str]], object]:
    """The function that turns the downloaded files into a ready network."""
    if spec.id == "realviformer":
        from restore_engine.models import realviformer
        return lambda paths: realviformer.load(paths[0])
    if spec.id == "realbasicvsr":
        from restore_engine.models import realbasicvsr
        return lambda paths: realbasicvsr.load(paths[0])

    def single(paths):
        import spandrel
        return spandrel.ModelLoader().load_from_file(paths[0]).model.eval()
    return single
