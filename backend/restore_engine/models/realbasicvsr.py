"""RealBasicVSR (Chan et al., CVPR 2022), in plain PyTorch.

A line-for-line port of mmagic's RealBasicVSRNet / BasicVSRNet / SPyNet
(Apache-2.0, https://github.com/open-mmlab/mmagic, models/editors/
basicvsr/basicvsr_net.py and real_basicvsr/real_basicvsr_net.py): the same
module and parameter names, so the published checkpoint loads strictly, and
none of mmcv / mmengine, whose prebuilt wheels stop at torch 2.4 — too old
for a Blackwell card. Checked against the original source on real frames
(text/restore-bench).

Inference uses the EMA generator (``generator_ema.*``): the released config
sets is_use_ema=True and mmagic's forward_tensor then runs generator_ema,
not generator.

One departure, in half precision only: flow_warp samples in float32 (see
there). In fp32 the port is the original to the bit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = {
    "url": "https://download.openmmlab.com/mmediting/restorers/real_basicvsr/"
           "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth",
    "file": "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth",
    "sha256_prefix": "52f77c2c",
    "licence": "Apache-2.0（OpenMMLab mmagic）",
}


def flow_warp(x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    # Where to sample is worked out, and sampled, in float32 whatever the
    # model runs in (mmagic builds the grid in x's dtype): in half precision
    # a pixel coordinate above 512 only moves in steps of 0.5 px (855 + 0.3
    # is 855.5), so in fp16 every warp on the right of a 16:9 frame was off
    # by up to a quarter pixel. In fp32 this is the original, to the bit.
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=flow.device, dtype=torch.float32),
        torch.arange(0, w, device=flow.device, dtype=torch.float32), indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2)
    grid_flow = grid + flow.float()
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(x.float(), grid_flow, mode=interpolation, padding_mode=padding_mode,
                         align_corners=align_corners).to(x.dtype)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(mid_channels=out_channels)
                            for _ in range(num_blocks)]))

    def forward(self, feat):
        return self.main(feat)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor,
                                       upsample_kernel, padding=(upsample_kernel - 1) // 2)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class _ConvAct(nn.Module):
    """mmcv's ConvModule as SPyNet uses it: conv, then ReLU or nothing."""

    def __init__(self, in_channels, out_channels, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 1, 3)
        self.activate = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        return self.activate(x) if self.activate is not None else x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvAct(8, 32), _ConvAct(32, 64), _ConvAct(64, 32), _ConvAct(32, 16),
            _ConvAct(16, 2, relu=False))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _level in range(5):
            ref.append(F.avg_pool2d(input=ref[-1], kernel_size=2, stride=2,
                                    count_include_pad=False))
            supp.append(F.avg_pool2d(input=supp[-1], kernel_size=2, stride=2,
                                     count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(input=flow, scale_factor=2, mode="bilinear",
                                        align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode="border"),
                flow_up], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(input=ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(input=supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(input=self.compute_flow(ref, supp), size=(h, w),
                             mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


class BasicVSRNet(nn.Module):
    def __init__(self, mid_channels=64, num_blocks=30):
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels,
                                                              num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels,
                                                             num_blocks)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        self.check_if_mirror_extended(lrs)
        flows_forward, flows_backward = self.compute_flow(lrs)
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            out += self.img_upsample(lr_curr)
            outputs[i] = out
        return torch.stack(outputs, dim=1)


class RealBasicVSRNet(nn.Module):
    """lqs (n, t, 3, h, w) in [0, 1], RGB → (n, t, 3, 4h, 4w).

    dynamic_refine_thres: 5 at inference (the released config's own note:
    "change to 5 for test"); 255 is the training value.
    """

    def __init__(self, mid_channels=64, num_propagation_blocks=20, num_cleaning_blocks=20,
                 dynamic_refine_thres=5):
        super().__init__()
        self.dynamic_refine_thres = dynamic_refine_thres / 255.
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True))
        self.basicvsr = BasicVSRNet(mid_channels, num_propagation_blocks)

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        for _ in range(0, 3):
            lqs = lqs.view(-1, c, h, w)
            residues = self.image_cleaning(lqs)
            lqs = (lqs + residues).view(n, t, c, h, w)
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break
        return self.basicvsr(lqs)


def load(path: str) -> RealBasicVSRNet:
    net = RealBasicVSRNet()
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    prefix = "generator_ema." if any(k.startswith("generator_ema.") for k in state) else "generator."
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    net.load_state_dict(weights, strict=True)
    return net.eval()
