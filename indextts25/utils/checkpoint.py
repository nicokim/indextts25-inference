# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> None:
    """Loads weights, ignoring keys the model does not have (e.g. the GPT's training
    text_head). A missing key is an error: it would leave randomly initialized layers."""
    checkpoint = torch.load(model_pth, map_location='cpu', mmap=True)
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    missing = model.load_state_dict(checkpoint, strict=False).missing_keys
    if missing:
        raise RuntimeError(f"{model_pth}: missing {len(missing)} keys, e.g. {missing[:5]}")


def fold_weight_norm(state_dict: dict, device) -> dict:
    """Folds each `*.weight_g`/`*.weight_v` pair (torch.nn.utils.weight_norm, dim=0)
    into a plain `*.weight`, computed on `device` with the same operation as the hook.

    The modules are defined without weight_norm: inference does not need to recompute
    g·v/||v|| on every forward, and the legacy API is deprecated in torch.
    """
    folded = {}
    for key, value in state_dict.items():
        name = key.rsplit(".", 1)[-1]
        if name == "weight_g":
            base = key[: -len("_g")]
            folded[base] = torch._weight_norm(state_dict[base + "_v"].to(device), value.to(device), 0)
        elif name != "weight_v":
            folded[key] = value
    return folded
