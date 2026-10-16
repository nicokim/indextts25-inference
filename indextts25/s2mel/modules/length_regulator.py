
import torch
import torch.nn as nn
from torch.nn import functional as F

from indextts25.s2mel.modules.commons import sequence_mask


class InterpolateRegulator(nn.Module):
    """Projects the continuous semantic features and stretches them to `ylens` mel frames.

    This is the only variant IndexTTS-2.5 uses (continuous input, no VQ, no f0).
    A config for another variant won't load: the checkpoint has keys that don't exist here.
    """

    def __init__(self, channels: int, sampling_ratios: tuple, in_channels: int,
                 codebook_size: int = 1024, groups: int = 1):
        super().__init__()
        model = []
        for _ in sampling_ratios:
            model += [nn.Conv1d(channels, channels, 3, 1, 1), nn.GroupNorm(groups, channels), nn.Mish()]
        model.append(nn.Conv1d(channels, channels, 1, 1))
        self.model = nn.Sequential(*model)
        # Unused with continuous input, but present in the checkpoint.
        self.embedding = nn.Embedding(codebook_size, channels)
        self.mask_token = nn.Parameter(torch.zeros(1, channels))
        self.content_in_proj = nn.Linear(in_channels, channels)

    def forward(self, x, ylens):
        x = self.content_in_proj(x)
        mask = sequence_mask(ylens).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')
        return self.model(x).transpose(1, 2).contiguous() * mask
