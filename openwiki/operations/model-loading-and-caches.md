---
type: operations
title: Model directory, loading and caches
description: The files the library reads from model_dir and config.yaml, how auxiliary models are resolved or downloaded, the order and rules of model loading (on-device construction, mmap, strict loading, weight-norm folding, bf16), and every in-memory and on-disk cache.
tags: [loading, checkpoints, caching, configuration, operations]
sources:
  - id: openwiki-source-42e695b3905ca2603655892f
    resource: repo://indextts25/codec/models.py
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
  - id: openwiki-source-93314986bcd35f2fb0820754
    resource: repo://indextts25/s2mel/modules/bigvgan/bigvgan.py
  - id: openwiki-source-b0f3c8ff31bb9a743a82597f
    resource: repo://indextts25/utils/checkpoint.py
  - id: openwiki-source-3672c72f276515deb6a3d235
    resource: repo://indextts25/utils/front.py
  - id: openwiki-source-7688cac5b090cd6e6bba0a20
    resource: repo://indextts25/utils/model_download.py
  - id: openwiki-source-8ec925e6ba68e143e4b0d6c5
    resource: repo://indextts25/utils/nemo_tn.py
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T07:18:19.761Z
---

# Model directory, loading and caches

All loading happens once, in `IndexTTS2._load_models` (`indextts25/infer_v2_5.py`), called
from the constructor inside `torch.random.fork_rng` so that the random initialization of
modules (later overwritten by checkpoints) does not disturb the caller's RNG.

## What `model_dir` must contain

| file | how it is found | used for |
|---|---|---|
| `config.yaml` | `IndexTTS25(config=...)`, default `model_dir/config.yaml` | every hyper-parameter below |
| GPT weights (`gpt.pth`) | `gpt_checkpoint:` in the config | the autoregressive GPT |
| `s2mel.pth` | `s2mel_checkpoint:` | length regulator and CFM/DiT (`net` dict) |
| `codec.pth` | fixed name | semantic codec decoder |
| `wav2vec2bert_stats.pt` | `w2v_stat:` | mean and variance used to normalize w2v-bert features |
| `feat1.pt`, `feat2.pt` | `spk_matrix:`, `emo_matrix:` | speaker and emotion matrices for `emo_vector`, split by `emo_num` |
| `multilingual_zh_ja_yue_char_del.tiktoken` | fixed name | text tokenizer vocabulary |
| `glossary.yaml` (optional) | fixed name | term pronunciations for the WeTextProcessing path |
| `hf_cache/` (optional) | fixed layout | local copies of the auxiliary models |

A fine-tuned GPT is loaded by pointing a copy of the config at other weights through
`gpt_checkpoint:` and passing that YAML as `config=`; everything else is shared.

## Auxiliary models

`ensure_models_available` (`utils/model_download.py`) returns local paths for w2v-bert-2.0,
CAMPPlus and BigVGAN. It checks for the actual weight files in `model_dir/hf_cache/` (not
just the directories, so an interrupted download is retried) and otherwise downloads from
the Hugging Face Hub into its standard cache, restricted with `allow_patterns` to the files
the library reads. `HF_ENDPOINT` selects a mirror; the library never changes `HF_HUB_CACHE`.

## Load sequence

1. **GPT.** `UnifiedVoice` is built on the device, then `load_checkpoint` reads the
   checkpoint memory-mapped. Keys the model lacks (the training-only text head) are ignored;
   a key the checkpoint lacks raises. With `fp16=True` the GPT is cast to bf16.
   `post_init_gpt2_config` then builds the `generate()` wrapper and, for `"flash"`, the
   accelerated engine sharing the same weights ([Flash engine](../pipeline/flash-engine.md)).
2. **w2v-bert.** The feature extractor comes from `from_pretrained` (local files only). The
   model is built from its config with 17 layers and loaded strictly from the matching
   tensors of `model.safetensors`; the 7 unused layers are never read.
3. **Codec.** Only the decoding half exists. Its checkpoint is memory-mapped (it also holds
   optimizer state that is never read), weight norm is folded, and missing keys raise.
4. **s2mel.** Each submodule loads strictly from the checkpoint's `net` dict after weight
   norm is folded; the DiT then precomputes its rotary-embedding tables.
5. **CAMPPlus** from its state dict, **BigVGAN** through `BigVGAN.from_dir` (config plus
   memory-mapped generator weights, folded).
6. Tokenizer, the WeTextProcessing wrapper (plus `glossary.yaml` when present), the
   emotion and speaker matrices, and the mel-spectrogram function.

Modules are defined without `torch.nn.utils.weight_norm`. `fold_weight_norm`
(`utils/checkpoint.py`) turns every `weight_g`/`weight_v` pair of a checkpoint into a plain
`weight` with `torch._weight_norm`, the same operation the hook used, so the result is
identical and the norm is not recomputed on every forward pass.

## Caches

**In memory, per engine.** Speaker conditioning, emotion conditioning and merged emotion
vectors are cached in dicts keyed by `(absolute path, st_mtime_ns)` of the clip, so
overwriting a reference file invalidates its entry. Each cache keeps at most eight entries
and evicts the oldest first.

**In memory, per process.** The NeMo normalizer is a module-level singleton, one grammar per
language, and the tokenizer is memoized with `lru_cache`.

**On disk** (under `$XDG_CACHE_HOME`, default `~/.cache`):

- `indextts25/nemo_tn/{lang}-{case}-nemo{version}-pynini{version}.pkl` holds a pickled,
  fully built NeMo `Normalizer`. Building a grammar takes tens of seconds; loading the
  pickle takes well under one. The file name changes with the NeMo and pynini versions, a
  corrupt or incompatible file is rebuilt, writes go to a temporary file followed by an
  atomic `os.replace`, and an unwritable directory only logs a warning. NeMo's own
  `cache_dir` export is not used because it re-optimizes the grammar and changes how some
  decimals are read.
- `indextts25/wetext_tagger/` holds the Chinese WeTextProcessing grammar, compiled with
  non-default options (interjections and erhua kept), outside the possibly read-only
  installed package. English uses the FSTs that ship with WeTextProcessing.
- The Hugging Face cache holds auxiliary models downloaded from the Hub.

## Related pages

- [Architecture overview](../architecture/overview.md)
- [Dependencies, extras and version compatibility](dependencies-and-compatibility.md)
- [Voice and emotion conditioning](../pipeline/voice-and-emotion-conditioning.md)
- [Text front-end](../pipeline/text-frontend.md)
