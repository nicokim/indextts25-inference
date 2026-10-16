---
type: quickstart
title: Quickstart
description: What indextts25-inference is, the minimal install-and-synthesize path, the handful of concepts you need (reference voice, backends, segments and chunks), and where in this wiki to look for each common task.
tags: [quickstart, overview, navigation]
sources:
  - id: openwiki-source-f99a73fe01bdf87e2b7d0a4d
    resource: repo://indextts25/__init__.py
  - id: openwiki-source-b78c0c2b02855c6c7d92411c
    resource: repo://indextts25/api.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T06:43:45.974Z
---

# Quickstart

indextts25-inference is an inference-only Python library for **IndexTTS-2.5**: zero-shot
text-to-speech that clones a voice from one reference clip, with emotion and speed control
and streaming that yields each audio chunk together with the text that produced it. It is
derived from index-tts and keeps only what inference needs.

## Minimal use

```bash
pip install indextts25-inference
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints --exclude "qwen0.6bemo4-merge/*"
```

```python
from indextts25 import IndexTTS25

tts = IndexTTS25(model_dir="checkpoints")          # attn_backend="flash" with the [flash] extra
tts.warm("voice.wav")                              # one-time costs before real traffic
tts.synthesize("Son las 14:30.", voice="voice.wav", out="out.wav")

for chunk in tts.stream("A longer text…", voice="voice.wav", lang="EN"):
    play(chunk.audio)                              # float32 mono, 22 050 Hz
    show(chunk.text)                               # the normalized segment text
```

## Concepts

- **Reference voice.** Any clip (only its first 15 s are used). Its conditioning is
  computed once and cached by path and modification time; an optional second clip, or an
  eight-value `emo_vector`, sets the emotion.
- **Language.** `lang` (default `"ES"`) picks the text normalizer, casing and language
  token. Spanish, English, Chinese, Japanese and Arabic are what the model speaks.
- **Segments and chunks.** Text is normalized, then cut into token-limited segments. Each
  segment runs through GPT → codec → flow matching → vocoder and becomes one `Chunk`.
  `segment_tokens` trades latency for throughput.
- **Backends.** `"sdpa"` uses transformers `generate()` with beam sampling; `"flash"` uses
  an accelerated single-beam decoder with a paged KV cache and CUDA graphs. Both score the
  same on intelligibility and speaker similarity.
- **Side effects.** None process-wide, and silent unless `verbose=True`. An instance is not
  thread-safe.

## Where to look

| I want to… | read |
|---|---|
| know every argument, return value and error | [Public API](api/public-api.md) |
| understand how the pieces fit | [Architecture overview](architecture/overview.md) |
| fix how numbers, times or symbols are read, or change segmentation | [Text front-end](pipeline/text-frontend.md) |
| work with reference voices, emotion clips or `emo_vector` | [Voice and emotion conditioning](pipeline/voice-and-emotion-conditioning.md) |
| tune sampling or understand stop codes | [GPT semantic-code generation](pipeline/gpt-generation.md) |
| debug or modify the `flash` backend | [Flash engine](pipeline/flash-engine.md) |
| change speed, diffusion steps or the vocoder | [Acoustic decoding](pipeline/acoustic-decoding.md) |
| load a fine-tuned GPT, find caches or offline files | [Model directory, loading and caches](operations/model-loading-and-caches.md) |
| pick torch/transformers/flash-attn versions | [Dependencies, extras and version compatibility](operations/dependencies-and-compatibility.md) |
| run tests or reproduce the benchmark | [Testing and benchmarks](testing/testing-and-benchmarks.md) |
