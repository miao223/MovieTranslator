"""RealViformer (Zhang & Yao, ECCV 2024), streamed.

realviformer_arch.py, arch_util.py and spynet_arch.py are the authors' files
unchanged (MIT, LICENSE beside them; https://github.com/Yuehan717/RealViformer
at bd5f88d, 2024-07-22). What is added here is only how they are driven:

* **streaming**: the network is causal — each frame is made from the frames
  before it — and the authors' script cuts a video into independent
  100-frame pieces, so the picture jumps every 100 frames where the
  recurrent state starts from zero again. :class:`Stream` carries that state
  (the propagated features and the last frame, for the next flow) from one
  piece to the next, so an arbitrarily long film is one unbroken sequence;
  the result is the same as running the whole film through the authors'
  forward at once (checked in text/restore-bench).
* **weights**: the authors' script loads ``params`` with strict=False. Here
  every model parameter must be present, and the one key the checkpoint
  has that the model does not (``attn_merge.attn.masktemp``) is the only one
  allowed to go unused.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .arch_util import flow_warp
from .realviformer_arch import RealViformer

WEIGHTS = {
    # the authors publish it on Google Drive only; MIT allows a mirror, and
    # the app downloads from the mirror (see restore_engine/weights.py)
    "file": "realviformer_weights.pth",
    "sha256_prefix": "25d49a0128b1ecf2",
    "licence": "MIT（Yuehan Zhang）",
}

UNUSED_KEYS = {"attn_merge.attn.masktemp"}


def build() -> RealViformer:
    return RealViformer(num_feat=48, num_blocks=[2, 3, 4, 1], spynet_path=None, heads=[1, 2, 4],
                        ffn_expansion_factor=2.66, merge_head=2, bias=False,
                        LayerNorm_type="BiasFree", ch_compress=True, squeeze_factor=[4, 4, 4],
                        masked=True)


def load(path: str) -> RealViformer:
    net = build()
    state = torch.load(path, map_location="cpu", weights_only=False)["params"]
    result = net.load_state_dict(state, strict=False)
    if result.missing_keys or set(result.unexpected_keys) - UNUSED_KEYS:
        raise ValueError(f"RealViformer 权重和网络对不上：缺 {result.missing_keys}，"
                         f"多出 {result.unexpected_keys}")
    return net.eval()


class Stream:
    """Feed frames in pieces of any length; get the same frames back as one
    long forward() would give. Inputs (1, n, 3, h, w) in [0, 1], RGB, h and w
    multiples of 4 (the caller pads); outputs 4x."""

    def __init__(self, net: RealViformer):
        self.net = net
        self.prev = None        # the last frame of the previous piece, for the flow into this one
        self.feat = None        # propagated features after that frame

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net
        b, n, _, h, w = x.size()
        seq = x if self.prev is None else torch.cat([self.prev, x], dim=1)
        start = seq.size(1) - n
        flows = net.get_flow(seq.clone()) if seq.size(1) > 1 else None
        feat_prop = self.feat if self.feat is not None else x.new_zeros(b, net.num_feat, h, w)
        out_l = []
        for i in range(start, seq.size(1)):
            x_i = seq[:, i, :, :, :]
            feat_shallow = net.shallow_extraction(x_i)
            if i > 0:
                flow = flows[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = net.attn_merge(feat_shallow, feat_prop)
            out_enc_level1 = net.encoder_level1(feat_prop)
            inp_enc_level2 = net.down1_2(out_enc_level1)
            out_enc_level2 = net.encoder_level2(inp_enc_level2)
            inp_enc_level3 = net.down2_3(out_enc_level2)
            latent = net.latent(inp_enc_level3)
            inp_dec_level2 = net.up3_2(latent)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = net.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = net.decoder_level2(inp_dec_level2)
            inp_dec_level1 = net.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            out_dec_level1 = net.decoder_level1(inp_dec_level1)
            out = net.refinement(out_dec_level1)
            feat_prop = net.compress(out)
            out = net.lrelu(net.pixel_shuffle(net.upconv1(out)))
            out = net.lrelu(net.pixel_shuffle(net.upconv2(out)))
            out = net.lrelu(net.conv_hr(out))
            out = net.conv_last(out)
            out += F.interpolate(x_i, scale_factor=4, mode="bilinear", align_corners=False)
            out_l.append(out)
        self.prev, self.feat = seq[:, -1:].clone(), feat_prop
        return torch.stack(out_l, dim=1)
