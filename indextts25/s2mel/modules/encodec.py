# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Conv1d with asymmetric or causal padding (from EnCodec), as used by the s2mel WaveNet."""

import math
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """Extra padding at the end so that the last window is complete."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    if paddings == (0, 0):
        return x  # F.pad would copy the whole tensor anyway
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class NormConv1d(nn.Module):
    """Conv1d. `norm='weight_norm'` is accepted for config compatibility: the
    checkpoint's weight norm is folded into `weight` at load time (fold_weight_norm)."""

    def __init__(self, *args, causal: bool = False, norm: str = 'none', **kwargs):
        super().__init__()
        if norm not in ('none', 'weight_norm'):
            raise NotImplementedError(f"norm {norm!r} not supported")
        self.conv = nn.Conv1d(*args, **kwargs)
        self.norm = nn.Identity()
        self.norm_type = norm

    def forward(self, x):
        return self.norm(self.conv(x))


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', pad_mode: str = 'reflect', **kwargs):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal, norm=norm)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)
