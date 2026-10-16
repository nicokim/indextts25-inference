import torch
from torch import nn


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    # use torch.split to avoid dynamic slicing
    t_act_part, s_act_part = torch.split(in_act, n_channels_int, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    acts = t_act * s_act
    return acts


class MyModel(nn.Module):
    """The inference s2mel: {cfm: CFM with the DiT, length_regulator}."""

    def __init__(self, args):
        super().__init__()
        from indextts25.s2mel.modules.flow_matching import CFM
        from indextts25.s2mel.modules.length_regulator import InterpolateRegulator

        lr = args.length_regulator
        self.models = nn.ModuleDict({
            'cfm': CFM(args),
            'length_regulator': InterpolateRegulator(
                channels=lr.channels,
                sampling_ratios=lr.sampling_ratios,
                in_channels=lr.in_channels,
                codebook_size=lr.content_codebook_size,
            ),
        })
