---
type: api-reference
title: "Public API: IndexTTS25 and Chunk"
description: How to construct IndexTTS25, warm it up, synthesize to a WAV file or stream Chunk objects, which keyword arguments reach the generator, which errors to expect, and the threading contract.
tags: [api, synthesis, streaming, configuration, threading]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T14:28:52.827Z
sources:
  - id: openwiki-source-f99a73fe01bdf87e2b7d0a4d
    resource: repo://indextts25/__init__.py
  - id: openwiki-source-0719484fdaa00f6fd5bf6bb4
    resource: repo://indextts25/accel/accel_engine.py
  - id: openwiki-source-10054e7cff1fd57914a88ac5
    resource: repo://indextts25/accel/attention.py
  - id: openwiki-source-b78c0c2b02855c6c7d92411c
    resource: repo://indextts25/api.py
  - id: openwiki-source-0908f962bcb3f4aab17c1ae5
    resource: repo://indextts25/gpt/model_v2.py
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
---

# Public API: IndexTTS25 and Chunk

The package exports three names: `IndexTTS25`, `Chunk` and `STREAM_SEGMENT_TOKENS`
(`indextts25/__init__.py`). `IndexTTS25` (`indextts25/api.py`) is a thin facade over the
engine class `IndexTTS2` (`indextts25/infer_v2_5.py`): it validates inputs, maps friendly
argument names onto the engine's, and turns the engine's raw generator into `Chunk`
objects. Everything heavy (torch models, transformers, NeMo) is imported only when an
`IndexTTS25` is constructed, so `import indextts25` itself is cheap.

## Construction

```python
IndexTTS25(model_dir="checkpoints", config=None, attn_backend="sdpa",
           fp16=False, device=None, verbose=False)
```

| argument | meaning |
|---|---|
| `model_dir` | Directory with `gpt.pth`, `s2mel.pth`, `codec.pth`, the tiktoken vocabulary and optionally `hf_cache/` (see [Model directory, loading and caches](../operations/model-loading-and-caches.md)). |
| `config` | YAML to load; defaults to `model_dir/config.yaml`. A config whose `gpt_checkpoint:` points at other weights loads a fine-tuned GPT. |
| `attn_backend` | `"sdpa"` (transformers `generate()`) or `"flash"` (the accelerated engine, see [Flash engine](../pipeline/flash-engine.md)). |
| `fp16` | Runs the GPT in bf16 (the engine's `use_bf16`). Saves VRAM, not time. |
| `device` | Torch device string. When omitted the engine picks `cuda:0`, then `xpu`, then `mps`, then `cpu`. |
| `verbose` | Prints the normalized text, segments and per-stage timings of every synthesis. |

The constructor fails fast, before any model is loaded: a missing config raises
`FileNotFoundError`, and `attn_backend="flash"` without an importable `flash_attn` raises
`ImportError` with the install hint. bf16 is silently ignored on CPU and MPS.

## `warm(voice, lang="ES")`

Runs one complete synthesis of the text `"Hola."` through the streaming generator and
discards the audio. This moves the one-time costs out of the first real request: the
reference-voice conditioning (cached afterwards), loading or compiling the language's
normalization grammar, and, with `"flash"`, capturing the CUDA graph. Pass the language you
will use, since grammars are per language.

## `synthesize(text, voice, out, ...)`

Synthesizes the whole text into a 16-bit PCM WAV at 22 050 Hz and returns `Path(out)`.
Segments are joined with 200 ms of silence. Arguments: `lang` (default `"ES"`), `speed`
(the s2mel `duration_factor`; below 1 is faster), `emotion` (a separate emotion reference
clip), `emotion_strength` (the mixing weight `emo_alpha`) and `segment_tokens` (default
120, the per-segment token budget). Text without any alphanumeric character raises
`ValueError`.

## `stream(text, voice, ...)` and `Chunk`

`stream()` is a generator with the same arguments plus `include_silence`. It yields one
`Chunk` per text segment as soon as that segment is vocoded:

| field | content |
|---|---|
| `audio` | `float32` mono samples in [-1, 1] (the engine's int16-scaled waveform divided by 32767) |
| `text` | the segment as the model saw it: normalized, numbers spelled out, upper-cased in Spanish |
| `sample_rate` | 22050 |
| `elapsed` | seconds since `stream()` started |
| `is_silence` | `True` only for the pauses between segments |
| `duration` | property: `len(audio) / sample_rate` |

The 200 ms pauses between segments are dropped unless `include_silence=True`. Text without
anything pronounceable yields no chunks at all (no exception, unlike `synthesize()`).
Streaming defaults to `segment_tokens=STREAM_SEGMENT_TOKENS` (60): smaller segments reach
the first chunk sooner and give finer-grained text, at some cost in throughput. How text is
cut into segments is described in [Text front-end](../pipeline/text-frontend.md).

## Keyword passthrough to the generator

Both `synthesize()` and `stream()` forward unknown keyword arguments to
`IndexTTS2.infer_generator`. Some are explicit parameters there:

- `emo_vector`: eight floats, `[happy, angry, sad, afraid, disgusted, melancholic,
  surprised, calm]`, used instead of an emotion clip. `emotion_strength` is clamped to
  [0, 1] and scales the vector. See [Voice and emotion conditioning](../pipeline/voice-and-emotion-conditioning.md).
- `use_random`: with `emo_vector`, pick random rows of the emotion matrices instead of the
  ones closest to the speaker style.
- `interval_silence`: the pause between segments in milliseconds (200).

The rest are generation settings, with these defaults:

| knob | default | used by |
|---|---|---|
| `do_sample` | `True` | both; `False` gives beam search with `sdpa` and greedy decoding with `flash` |
| `temperature` | 0.8 | both |
| `top_k` | 30 | both |
| `top_p` | 0.8 | both |
| `repetition_penalty` | 10.0 | both |
| `num_beams` | 3 | `sdpa` only; `flash` decodes a single beam |
| `length_penalty` | 0.0 | `sdpa` beam search |
| `max_mel_tokens` | 1500 | cap on generated codes per segment |
| `diffusion_steps` | 25 | Euler steps of the flow-matching decoder |
| `inference_cfg_rate` | 0.7 | classifier-free guidance weight of the decoder |

Anything else is passed through to the GPT (for example `typical_sampling`). Effects on the
models are explained in [GPT semantic-code generation](../pipeline/gpt-generation.md) and
[Acoustic decoding](../pipeline/acoustic-decoding.md).

## Errors and warnings

- `ValueError` from `synthesize()` for text with nothing to say.
- `ValueError` from the engine for a `lang` that is neither a tokenizer language code nor
  `"zhen"`.
- A `RuntimeWarning` (once per call) when a segment hits `max_mel_tokens` without producing
  the stop code; the audio of that segment is cut there.
- A segment for which the GPT emits the stop code immediately is skipped silently.

## Threading and side effects

An instance is not thread-safe: serialize calls, for example through one worker thread.
Which thread does not matter, because the library does not redirect `stdout`/`stderr`,
install warning filters or set environment variables, and prints nothing unless
`verbose=True`. The `flash` engine additionally keeps its per-step attention context in a
module-level variable, so two flash instances must not decode concurrently either.

## Related pages

- [Architecture overview](../architecture/overview.md)
- [Text front-end](../pipeline/text-frontend.md)
- [Voice and emotion conditioning](../pipeline/voice-and-emotion-conditioning.md)
- [GPT semantic-code generation](../pipeline/gpt-generation.md)
