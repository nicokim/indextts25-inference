---
type: pipeline-stage
title: "Flash engine: paged KV cache and CUDA graphs"
description: The accelerated batch-1 GPT decoder behind attn_backend="flash" — weight sharing with the main GPT, the paged fp16 KV pool, the Attention module with Triton KV stores and flash-attn kernels, prefill and CUDA-graph decode, the sampler, stop handling, and its limits.
tags: [flash-attn, cuda-graphs, kv-cache, decoding, performance]
sources:
  - id: openwiki-source-0719484fdaa00f6fd5bf6bb4
    resource: repo://indextts25/accel/accel_engine.py
  - id: openwiki-source-10054e7cff1fd57914a88ac5
    resource: repo://indextts25/accel/attention.py
  - id: openwiki-source-2f2d0a9da3bed25a4ef63088
    resource: repo://indextts25/accel/gpt2_accel.py
  - id: openwiki-source-adceae00dfd760547de770dd
    resource: repo://indextts25/accel/kv_manager.py
  - id: openwiki-source-0908f962bcb3f4aab17c1ae5
    resource: repo://indextts25/gpt/model_v2.py
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T06:43:45.974Z
---

# Flash engine: paged KV cache and CUDA graphs

With `attn_backend="flash"`, `UnifiedVoice.inference_speech` hands generation to
`AccelInferenceEngine` (`indextts25/accel/`) instead of transformers `generate()`. The engine
decodes one sequence at a time, keeps keys and values in a paged fp16 pool, runs attention
with flash-attn and replays each decode step from a captured CUDA graph, which removes the
per-token Python and launch overhead of `generate()`. It produces the same kind of output
as the `sdpa` path ([GPT semantic-code generation](gpt-generation.md)): the prompt ids
followed by the generated codes, stop code included.

## Construction and weight sharing

`UnifiedVoice.post_init_gpt2_config` builds the engine only when `use_accel` is set and the
GPT lives on a CUDA device; on any other device the flag has no effect and the `sdpa` path
runs. The accelerated GPT-2 (`GPT2AccelModel`) is instantiated on the `meta` device and
filled with `load_state_dict(..., assign=True)`, so its parameters *are* the main GPT's
tensors, in whatever dtype they have (fp32, or bf16 with `fp16=True`). No second copy of the
weights exists. Its token and position embedding tables are deleted: codes are embedded
outside with the GPT's mel embedding and mel position embedding.

`GPT2AccelModel` takes only `inputs_embeds` and returns the final hidden state after
`ln_f`. Each `GPT2AccelBlock` has its own minimal forward (pre-norm attention and MLP with
residuals), which keeps it independent of transformers' changing `GPT2Block` contract.
`GPT2AccelAttention` computes q/k/v with the original `c_attn` weights, flattens them to
`[tokens, heads, head_dim]`, casts them to fp16 for flash-attn and casts the output back.

## Paged KV pool

`KVCacheManager` (`accel/kv_manager.py`) allocates one fp16 tensor
`[2, layers, blocks, block_size, heads, head_dim]` and wires per-layer views into every
attention module. Blocks hold 256 tokens. The pool is sized for the longest possible
sequence, `ceil((max_text_tokens + max_mel_tokens + 8) / 256)` blocks. A `Seq` tracks the
token count and its block table; a new block is taken from the free list whenever a token
opens one, and all blocks return to the free list in a `finally` when generation ends,
including on errors.

There is deliberately no prefix cache: the real prompt is passed as embeddings behind
placeholder ids, so different prompts look identical by id and a prefix cache would reuse
the wrong keys and values.

## Attention and the forward context

`Attention.forward` (`accel/attention.py`) first stores the new keys and values into their
slots with a small Triton kernel (`store_kvcache_kernel`), then runs:

- **prefill:** `flash_attn_varlen_func` over the whole prompt, causal;
- **decode:** `flash_attn_with_kvcache` for the single new query against the paged cache,
  addressed through the block table and the current context length.

The per-call metadata (prefill flag, cumulative lengths, slot mapping, context lengths,
block tables) is not passed through the model: the engine sets a module-level
`ForwardContext` before each forward and resets it afterwards. This is why two engines must
never run concurrently in one process.

## Generation flow

1. **Prefill.** The prompt embeddings `[pad][cond][text]` from `prepare_gpt_inputs` are
   concatenated with the start-of-mel embedding, which receives mel position 0 as in
   training. One forward pass fills the KV cache, and the last hidden state goes through
   `final_norm` and `mel_head` to the first logits.
2. **Sampling** (`_sample`), in the same order as transformers: repetition penalty on the
   raw logits (over every id seen so far, including the placeholder prompt ids), then
   temperature, top-k, top-p, and finally sampling by the exponential-race trick.
   Temperature ≤ 0 means greedy argmax; the GPT maps `do_sample=False` to that.
3. **Decode loop.** Each new code is appended, a block is opened if needed, and one decode
   step embeds the code with mel position *k* for the *k*-th generated code (the start code
   holds position 0) and runs the model. Generation stops on the stop code, which is kept in the output, or after
   `max_new_tokens`.

## CUDA graph

On first use the engine captures one decode step into a CUDA graph with static input
buffers (code id, position, slot, context length, block table and the hidden-state output).
Each step then only writes these buffers and replays the graph. The block-table width
captured in the graph is fixed at 2048 tokens because flash-attn chooses its work split from
that width and changing it would change results in the last bit; sequences longer than
that continue on an eager path that builds the same inputs per step. Capture happens during
`warm()` if it is called.

## Limits

- No beam search: `num_beams`, `length_penalty` and `typical_sampling` are ignored; sampling
  uses only temperature, top-k, top-p and repetition penalty.
- CUDA only, batch 1, one decode at a time per process.
- The fp16 KV pool and the graph's static buffers are allocated once and reused across
  calls.

## Related pages

- [GPT semantic-code generation](gpt-generation.md)
- [Dependencies, extras and version compatibility](../operations/dependencies-and-compatibility.md)
- [Testing and benchmarks](../testing/testing-and-benchmarks.md)
