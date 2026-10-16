# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.
"""BigVGAN v2 (nvidia/bigvgan_v2_22khz_80band_256x), inference only.

Always loaded from a local directory with config.json and bigvgan_generator.pt, so the
huggingface_hub mixin is not needed. NVIDIA's fused CUDA kernel was removed: it required
nvcc/ninja for the same CUDA version as torch, and the vocoder costs only ~40 ms per segment.
"""
import json
import os

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d

from indextts25.utils.checkpoint import fold_weight_norm

from . import activations
from .alias_free_activation.torch.act import Activation1d
from .env import AttrDict
from .utils import get_padding


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        return AttrDict(json.load(f))


def _snake(h, channels):
    if h.activation == "snake":
        return activations.Snake(channels, alpha_logscale=h.snake_logscale)
    if h.activation == "snakebeta":
        return activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale)
    raise NotImplementedError(f"activation {h.activation!r} not supported")


class AMPBlock1(torch.nn.Module):
    """Anti-aliased Snake/SnakeBeta + dilated convs (convs1) and dilation-1 convs (convs2)."""

    def __init__(self, h: AttrDict, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))
            for _ in dilation
        ])
        self.activations = nn.ModuleList(
            [Activation1d(activation=_snake(h, channels)) for _ in range(2 * len(dilation))])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            x = c2(a2(c1(a1(x)))) + x
        return x


class BigVGAN(torch.nn.Module):
    """Neural vocoder with anti-aliased periodic activations (AMP)."""

    def __init__(self, h: AttrDict):
        super().__init__()
        if h.resblock != "1":
            raise NotImplementedError(f"resblock {h.resblock!r} not supported")
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)

        # Transposed-conv upsamplers (no anti-aliasing).
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([ConvTranspose1d(
                h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2)]))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(h, ch, k, d))

        self.activation_post = Activation1d(activation=_snake(h, ch))
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    @classmethod
    def from_dir(cls, directory, device="cpu"):
        """Builds the model on `device` from config.json + bigvgan_generator.pt."""
        with torch.device(device):
            model = cls(load_hparams_from_json(os.path.join(directory, "config.json")))
        sd = torch.load(os.path.join(directory, "bigvgan_generator.pt"), map_location="cpu", mmap=True)
        model.load_state_dict(fold_weight_norm(sd["generator"], device))
        return model.to(device)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.conv_post(self.activation_post(x))
        if self.use_tanh_at_final:
            return torch.tanh(x)
        return torch.clamp(x, min=-1.0, max=1.0)
