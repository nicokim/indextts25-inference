from collections import deque
from typing import List

import torch


class Seq:
    """A sequence in the paged pool: how many tokens it has and which blocks hold them."""

    def __init__(self, num_prompt_tokens: int, last_token: int, block_size: int = 256):
        self.num_tokens = num_prompt_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.last_token = last_token
        self.block_size = block_size
        self.block_table: List[int] = []

    def __len__(self):
        return self.num_tokens

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def append_token(self, token_id: int):
        self.last_token = token_id
        self.num_tokens += 1


class KVCacheManager:
    """fp16 KV pool, [2, layers, blocks, block_size, heads, head_dim], handed out
    block by block. No prefix caching: in TTS the real prompt is embeddings and the ids
    are filler, so two long prompts "matched" and overwrote each other."""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.free_block_ids: deque = deque(range(num_blocks))
        self.kv_cache = torch.empty(
            2, num_layers, num_blocks, block_size, num_heads, head_dim, dtype=torch.float16, device="cuda")

    def allocate(self, seq: Seq):
        for _ in range(seq.num_blocks):
            seq.block_table.append(self.free_block_ids.popleft())

    def append_to_seq(self, seq: Seq):
        # The just-appended token opens a new block.
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self.free_block_ids.popleft())

    def free(self, seq: Seq):
        self.free_block_ids.extend(reversed(seq.block_table))
        seq.block_table.clear()

    def wire_kv_cache_to_model(self, model):
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
