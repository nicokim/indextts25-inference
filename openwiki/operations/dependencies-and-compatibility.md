---
type: operations
title: Dependencies, extras and version compatibility
description: What each runtime dependency and optional extra is for, which upstream dependencies were replaced and how, the code seams that keep the library working on transformers 4.x and 5.x, flash-attn wheel constraints, and the stacks that were tested.
tags: [dependencies, packaging, compatibility, transformers, flash-attn]
sources:
  - id: openwiki-source-2f2d0a9da3bed25a4ef63088
    resource: repo://indextts25/accel/gpt2_accel.py
  - id: openwiki-source-0908f962bcb3f4aab17c1ae5
    resource: repo://indextts25/gpt/model_v2.py
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
  - id: openwiki-source-9aaef6e8e78299ad4d6be067
    resource: repo://indextts25/utils/common.py
  - id: openwiki-source-8ec925e6ba68e143e4b0d6c5
    resource: repo://indextts25/utils/nemo_tn.py
  - id: openwiki-source-01840b5abb4687ea7b428fe2
    resource: repo://indextts25/utils/tokenizer.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-310a7347cd31a1985b43c333
    resource: repo://tests/test_regressions.py
generated: { by: "claude-code", at: "2026-10-16T07:18:19.761Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T07:18:19.761Z
---

# Dependencies, extras and version compatibility

The package is built with hatchling and ships only the `indextts25` directory. Python 3.10
or newer is required; Linux is the supported platform.

## Runtime dependencies

| package | why it is needed |
|---|---|
| `torch>=2.8` | every model |
| `transformers>=4.52` | the GPT-2 backbone and `generate()` for the `sdpa` backend; `Wav2Vec2BertModel` and the SeamlessM4T feature extractor for voice conditioning |
| `safetensors>=0.4.3` | reading only the w2v-bert tensors the 17-layer model needs |
| `huggingface_hub>=0.30` | fetching w2v-bert-2.0, CAMPPlus and BigVGAN when `model_dir/hf_cache/` lacks them |
| `tiktoken>=0.7` | the text tokenizer, built from the vocabulary file in the model directory |
| `librosa>=0.10` | loading and resampling the reference clips |
| `omegaconf>=2.3` | reading `config.yaml` |
| `pyyaml>=5.1` | the optional `glossary.yaml` of term pronunciations |
| `einops>=0.8` | the perceiver resampler of the emotion encoder |
| `numpy>=1.26` | chunk audio and feature extraction |
| `nemo-text-processing>=1.1` | Spanish text normalization, and English/Chinese when the `zh` extra is absent |

`nemo-text-processing` carries the marker `sys_platform == 'linux' and platform_machine ==
'x86_64' and python_version < '3.14'` because it pins a `pynini` that only ships wheels
there. Where it is missing, normalization degrades to passing the text through unchanged
with a logged warning, so digits reach the tokenizer raw (see
[Text front-end](../pipeline/text-frontend.md)).

## Optional extras

| extra | adds | effect |
|---|---|---|
| `flash` | `flash-attn>=2.8.3`, `triton>=3.0` | enables `attn_backend="flash"`; Triton compiles the KV-cache store kernel |
| `zh` | `WeTextProcessing>=1.2` on Linux, `wetext>=0.1` elsewhere | WeTextProcessing normalization for Chinese and English |
| `ja` | `fugashi>=1.2`, `unidic-lite>=1.0` | Japanese word segmentation; Japanese input fails without it (or a system MeCab) |

The `test` dependency group adds `pytest` and `pytest-cov`, and the `dev` group adds `ruff`.

flash-attn is published on PyPI only as a source distribution; prebuilt wheels live on its
GitHub releases for specific torch/CUDA/Python combinations (CUDA 13 wheels exist only for
torch 2.9 and 2.10 on CPython 3.12; CUDA 12 wheels cover torch up to 2.8 on every Python).
Installing torch and the matching wheel before the extra avoids a source build. The
README's *Recommended stack* section has the exact commands.

## Dependencies that were replaced

- **torchaudio.** Only two of its functions were used. `utils/common.py` vendors the sinc
  resampler and the Kaldi filterbank (BSD-2-Clause), specialized to the arguments
  inference uses and documented as bit-identical; a test compares them with torchaudio
  when it happens to be installed. WAV files are written with the standard-library `wave`
  module, because `torchaudio.save()` routes through TorchCodec since torchaudio 2.9.
- **openai-whisper.** It was only used for its tokenizer constructor; `utils/tokenizer.py`
  builds the same `tiktoken.Encoding` directly, with the Whisper-style special tokens in a
  fixed order.
- **NVIDIA's fused BigVGAN CUDA kernel** and its Hugging Face mixin are gone; the vocoder
  loads from a local directory with plain PyTorch ops.

## transformers 4.x and 5.x

The same code runs on transformers 4.52 through 5.20. The seams, all in
`gpt/model_v2.py` unless noted:

1. `GPT2InferenceModel` inherits `GenerationMixin` explicitly; 5.x no longer provides it
   through `PreTrainedModel`.
2. `_past_len` reads the cached length from a `Cache` object or a legacy tuple. It never
   tests truthiness: an empty `Cache` is truthy in 5.x, which used to make generation drop
   the conditioning prefix after the first step.
3. Mel positions during decoding come from the cache length, because 5.x stops passing an
   all-ones `attention_mask` (see [GPT generation](../pipeline/gpt-generation.md)).
4. `_reorder_cache` calls `Cache.reorder_cache` when it exists and reorders nested tuples
   otherwise.
5. `accel/gpt2_accel.py`: `GPT2AccelBlock` has its own `forward`, since `GPT2Block` returns
   a tuple up to 5.2 and a tensor from 5.3 on.
6. Both `GPT2Config`s set `bos_token_id`/`eos_token_id` to `None`; the GPT-2 defaults
   (50256) lie outside these vocabularies and 5.x warns about them on stderr. `generate()`
   receives the mel start/stop tokens explicitly, so decoding is unaffected.
7. The per-layer causal-mask buffer that 4.x registers (about 134 MB in total) is removed,
   since the model runs with SDPA.
8. `infer_v2_5.py`: w2v-bert is built from its config with 17 layers and loaded from a
   filtered `safetensors` read instead of `from_pretrained`, which in 5.x also draws a
   progress bar on stderr. Loading is strict.

## Tested stacks

Each stack below ran the benchmark and a six-case synthesis check (Spanish sentence,
paragraph and stream, English, Spanish with an emotion clip, Japanese) with Whisper
transcription:

| Python | torch | transformers | flash-attn | backends |
|---|---|---|---|---|
| 3.11 | 2.8.0+cu128 | 4.52.1 | 2.8.3.post1 (cu12) | sdpa, flash |
| 3.12 | 2.9.1+cu130 | 4.57.6 | 2.8.3 (cu13) | sdpa, flash |
| 3.12 | 2.10.0+cu130 | 4.57.6 and 5.20.0 | 2.8.3 (cu13) | flash |
| 3.12 | 2.14.1+cu130 | 5.20.0 | — | sdpa |

With a fixed seed, the torch 2.8 stack produced bit-identical Spanish and Japanese audio to
torch 2.9.1. Python 3.10 and 3.13 install and pass the CPU tests. The bounds in
`pyproject.toml` are the minimum stack.

## Related pages

- [Flash engine](../pipeline/flash-engine.md)
- [Model directory, loading and caches](model-loading-and-caches.md)
- [Testing and benchmarks](../testing/testing-and-benchmarks.md)
