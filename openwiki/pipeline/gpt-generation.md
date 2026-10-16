---
type: pipeline-stage
title: GPT semantic-code generation
description: How the autoregressive GPT turns a text segment plus speaker and emotion conditioning into semantic codes — the input layout built by prepare_gpt_inputs, the sdpa path through transformers generate() with its position and cache handling, sampling defaults, stop-code handling, and the hand-off to the flash engine.
tags: [gpt, generation, transformers, sampling, positions]
sources:
  - id: openwiki-source-0908f962bcb3f4aab17c1ae5
    resource: repo://indextts25/gpt/model_v2.py
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
  - id: openwiki-source-c0041d170907b684158aef1e
    resource: repo://indextts25/utils/typical_sampling.py
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T07:18:19.761Z
---

# GPT semantic-code generation

`UnifiedVoice` (`indextts25/gpt/model_v2.py`) is a GPT-2 decoder (built from transformers'
`GPT2Model`, without its own token or position embeddings) that predicts *semantic codes*:
discrete tokens at 25 Hz from an 8192-entry codebook plus a start and a stop code. One call
of `inference_speech` generates the codes for one text segment; `infer_generator` calls it
once per segment.

## Inputs

For each segment the engine passes:

- `text_inputs`: the token ids of `<|lang|> ` plus the segment text
  ([Text front-end](text-frontend.md));
- `langs`: the language id from the tokenizer's language table;
- `emo_vec`: the merged emotion vector ([Voice and emotion conditioning](voice-and-emotion-conditioning.md));
- `campplus_embedding`: the 192-dimensional speaker style.

The conditioning prefix is `spk_emb_proj(style) + emo_vec` followed by two zero vectors.
`prepare_gpt_inputs` then builds the text part: it removes any stop-text ids, wraps the
text in start and stop text tokens, adds learned text-position embeddings and the language
embedding to every text position, and lays the prompt out as

```text
[pad?][conditioning (3)][start_text, text..., stop_text] + start_mel
```

Left padding only appears when stop ids had to be removed; it is masked out in the
attention mask. The start-text token has id 0, which in this tokenizer is also `"!"`, so
only stop ids are filtered: filtering id 0 as well would delete every exclamation mark.
The prompt is returned as embeddings plus placeholder `input_ids` (all ones, ending in
the start-mel code), because transformers' `generate()` needs ids to drive the loop.

## The sdpa backend: transformers `generate()`

`GPT2InferenceModel` adapts the GPT to `generate()`:

- **Prefix cache.** `store_mel_emb` keeps the prompt embeddings. On the first forward pass
  the placeholder ids are replaced by them, and the start-mel code is embedded with mel
  position 0.
- **Decode steps.** Each new code is embedded with the mel embedding plus the mel position
  `past_length − prefix_length`, so the *k*-th generated code sits at mel position *k*, as
  in training. Upstream was off by one. The position is derived from the KV-cache length
  rather than from `attention_mask`, which transformers 5.x omits when it is all ones.
- **Cache helpers.** `_past_len` reads the cached length from both `Cache` objects
  (transformers ≥ 5) and legacy tuples without relying on truthiness, and
  `_reorder_cache` reorders either form between beams.
- The GPT's position-embedding hook returns a scalar zero, so the positions passed by
  `generate()` have no effect and cannot promote bf16 activations to fp32.

`inference_speech` calls `generate()` with the mel start/stop codes as bos/eos/pad ids,
`max_length = prompt + max_mel_tokens`, and the sampling settings from the caller.
`typical_sampling=True` adds a typical-decoding logits warper. The returned tensor is cut
at the prompt length, leaving the generated codes (stop code included).

## Sampling defaults

`infer_generator` fixes the defaults: `do_sample=True`, `temperature=0.8`, `top_k=30`,
`top_p=0.8`, `repetition_penalty=10.0`, `num_beams=3`, `length_penalty=0.0`,
`max_mel_tokens=1500`. With `sdpa` this is beam sampling over three beams. The GPT runs
under bf16 autocast when the engine was built with `fp16=True`.

## Stop handling

After each call `infer_generator`:

1. emits one `RuntimeWarning` per call if a segment ended without the stop code (it hit
   `max_mel_tokens`; lowering `segment_tokens` usually avoids it);
2. cuts the codes at the first stop code;
3. skips the segment entirely if nothing is left.

The codes then go to the acoustic decoder ([Acoustic decoding](acoustic-decoding.md)).

## Hand-off to the flash engine

When the GPT was built with the flash engine, `inference_speech` calls
`AccelInferenceEngine.generate` with the same prompt embeddings and forwards only
`temperature` (0 when `do_sample=False`), `top_k`, `top_p` and `repetition_penalty`. Beams,
`length_penalty` and typical sampling do not apply there. See
[Flash engine](flash-engine.md).

## Related pages

- [Flash engine](flash-engine.md)
- [Voice and emotion conditioning](voice-and-emotion-conditioning.md)
- [Text front-end](text-frontend.md)
- [Acoustic decoding](acoustic-decoding.md)
