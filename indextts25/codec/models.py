# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""IndexTTS-2.5 semantic codec, only the decoding half.

At inference the GPT produces codes and the only thing that runs is `decode()`: codes ->
w2v-bert-style features at 50 Hz. The encoder, the downsample and Amphion's alternative
quantizers were training-only. Module names match the checkpoint's
(`quantizer.quantizers.0.*`, `decoder.*`, `up.*`).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from indextts25.utils.checkpoint import fold_weight_norm


class ConvNeXtBlock(nn.Module):
    """1D ConvNeXt block from Vocos (https://github.com/facebookresearch/ConvNeXt)."""

    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        return residual + (self.gamma * x).transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, input_channels: int, dim: int, intermediate_dim: int, num_layers: int):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, 1 / num_layers) for _ in range(num_layers)])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class FactorizedVectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.out_project = nn.Conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def vq2emb(self, vq):
        return self.out_project(F.embedding(vq, self.codebook.weight).transpose(1, 2))


class ResidualVQ(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.quantizers = nn.ModuleList([FactorizedVectorQuantize(input_dim, codebook_size, codebook_dim)])

    def vq2emb(self, vq):
        return self.quantizers[0].vq2emb(vq[0])


class EnhancedCodec(nn.Module):
    def __init__(self, codebook_size=8192, hidden_size=1024, codebook_dim=8, vocos_dim=384,
                 vocos_intermediate_dim=2048, vocos_num_layers=12):
        super().__init__()
        self.quantizer = ResidualVQ(hidden_size, codebook_size, codebook_dim)
        self.decoder = nn.Sequential(
            VocosBackbone(hidden_size, vocos_dim, vocos_intermediate_dim, vocos_num_layers),
            nn.Linear(vocos_dim, hidden_size),
        )
        self.up = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

    def decode(self, codes):
        """Codes (B, T) at 25 Hz -> features (B, 2T, hidden_size) at 50 Hz."""
        x = self.decoder(self.quantizer.vq2emb(codes.unsqueeze(0)))
        x = F.interpolate(x.transpose(1, 2), scale_factor=2, mode="nearest")
        return self.up(x).transpose(1, 2)

    def load_checkpoint(self, checkpoint_path, device="cpu"):
        # mmap: codec.pth also carries ~400 MB of optimizer state that is never read.
        sd = fold_weight_norm(torch.load(checkpoint_path, map_location="cpu", mmap=True)["model"], device)
        # strict=False only because of the encoder keys that don't exist here. A missing
        # key would leave random weights and garbage audio: that one IS an error.
        missing = self.load_state_dict(sd, strict=False).missing_keys
        if missing:
            raise RuntimeError(f"codec checkpoint is missing {missing}")
