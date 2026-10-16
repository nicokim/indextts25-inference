---
type: architecture
title: Architecture overview
description: How indextts25-inference is layered (IndexTTS25 facade, IndexTTS2 engine, model components), how text plus a reference clip flows to a 22.05 kHz waveform, where state lives, and the design constraints every module follows.
tags: [architecture, pipeline, data-flow, design-constraints]
sources:
  - id: openwiki-source-0719484fdaa00f6fd5bf6bb4
    resource: repo://indextts25/accel/accel_engine.py
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
  - id: openwiki-source-8ec925e6ba68e143e4b0d6c5
    resource: repo://indextts25/utils/nemo_tn.py
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T07:02:23.388Z
---

# Architecture overview

indextts25-inference is an inference-only packaging of IndexTTS-2.5 (derived from index-tts
commit `ee40fa7`). It turns a text and a reference voice clip into speech in that voice, one
text segment at a time, so audio can be streamed while later segments are still being
generated.

## Layers

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    A[IndexTTS25<br/>indextts25/api.py] --> B[IndexTTS2 engine<br/>indextts25/infer_v2_5.py]
    B --> T[Text front-end<br/>utils/front.py, utils/nemo_tn.py,<br/>utils/tokenizer.py, utils/ja_g2p.py]
    B --> C[Conditioning models<br/>w2v-bert-2.0, CAMPPlus, mel, length regulator]
    B --> G[GPT<br/>gpt/model_v2.py UnifiedVoice]
    G --> S[sdpa: transformers generate]
    G --> F[flash: accel/ engine]
    B --> D[Acoustic decoder<br/>codec/models.py, s2mel/, BigVGAN]
```

1. **Facade** — `IndexTTS25` validates inputs, names the backends, and wraps the engine's
   generator into `Chunk` objects ([Public API](../api/public-api.md)).
2. **Engine** — `IndexTTS2` loads every model once, owns the per-voice caches, and runs the
   pipeline in `infer_generator`. `infer()` is simply its first result.
3. **Components** — plain `nn.Module`s with no knowledge of each other; the engine moves
   tensors between them.

Module map:

| area | modules |
|---|---|
| text | `utils/front.py` (WeTextProcessing wrapper, punctuation map), `utils/nemo_tn.py` (NeMo normalizer), `utils/tokenizer.py` (tiktoken with Whisper-style special tokens), `utils/ja_g2p.py` (Japanese segmentation) |
| conditioning | `transformers.Wav2Vec2BertModel` (17 layers), `s2mel/modules/campplus/`, `s2mel/modules/audio.py` (mel), `utils/common.py` (vendored resampler and Kaldi fbank) |
| GPT | `gpt/model_v2.py` (`UnifiedVoice`, `GPT2InferenceModel`), `gpt/conformer_encoder.py` + `gpt/perceiver.py` (emotion encoder), `accel/` (flash engine) |
| acoustic | `codec/models.py` (semantic codec, decoder half), `s2mel/modules/` (length regulator, CFM, DiT, WaveNet, gpt_fast transformer), `s2mel/modules/bigvgan/` |
| loading | `utils/model_download.py`, `utils/checkpoint.py` |

## End-to-end flow of one call

`infer_generator` does, in order:

1. **Validate and resolve emotion inputs.** The language must be a tokenizer language code
   or `zhen`. Without an emotion clip the speaker clip doubles as the emotion source.
2. **Conditioning** ([Voice and emotion conditioning](../pipeline/voice-and-emotion-conditioning.md)).
   The speaker clip yields w2v-bert features, the CAMPPlus style vector, the length-regulated
   prompt condition and the reference mel; the emotion clip yields its own w2v-bert
   features. Both are cached by absolute path and modification time.
3. **Text** ([Text front-end](../pipeline/text-frontend.md)). Normalization, casing,
   annotations, then `split_text_by_tokens`; each segment is encoded as `<|lang|> ` plus the
   segment text.
4. **Emotion vector.** `merge_emovec` mixes speaker and emotion features with
   `emotion_strength`; the result is cached per (voice, emotion, strength). An explicit
   `emo_vector` is blended in afterwards.
5. **Per segment:**
   - the GPT generates semantic codes until the stop code ([GPT generation](../pipeline/gpt-generation.md),
     [Flash engine](../pipeline/flash-engine.md));
   - codes are cut at the first stop code; an empty result skips the segment;
   - the codec decodes codes into 50 Hz features, the length regulator stretches them to
     `len × 1.72 × duration_factor` mel frames, the prompt condition is prepended, the CFM
     solver produces the mel, the prompt frames are cut off, and BigVGAN renders the
     waveform ([Acoustic decoding](../pipeline/acoustic-decoding.md));
   - the waveform is clamped to the int16 range and moved to the CPU. In streaming mode it is
     yielded immediately, followed by `interval_silence` ms of silence unless it was the
     last segment.
6. **Non-streaming output.** Segment waveforms are joined with the silence and either
   written as a 16-bit WAV or returned as `(sample_rate, int16 array)`.

## Where state lives

- **Engine instance:** all models, the three conditioning caches (FIFO, eight entries each),
  the WeTextProcessing wrapper and the lazily created Japanese processor.
- **Module level:** the NeMo normalizer singleton (`utils/nemo_tn.py`), the tokenizer
  (`lru_cache` in `utils/tokenizer.py`) and the flash engine's forward context
  (`accel/attention.py`).
- **Disk:** compiled NeMo grammars and WeTextProcessing taggers under
  `~/.cache/indextts25/`, auxiliary models in `model_dir/hf_cache/` or the Hugging Face cache
  ([Model directory, loading and caches](../operations/model-loading-and-caches.md)).

## Design constraints

- **Batch 1.** One request, one sequence: the public API never batches texts and the flash
  engine captures a batch-1 CUDA graph. The batch only widens inside a stage, for `sdpa`
  beam search (3 beams by default) and for classifier-free guidance in the CFM (2).
- **No process-wide side effects.** No `stdout`/`stderr` redirection, warning filters or
  environment variables; output only with `verbose=True`. The single exception is lowering
  NeMo's and WeTextProcessing's own loggers to WARNING.
- **Reproducible.** With a fixed seed, the same stack produces bit-identical audio; model
  construction runs inside `torch.random.fork_rng` so building models does not move the
  caller's RNG. Refactors are verified by hashing the PCM ([Testing and benchmarks](../testing/testing-and-benchmarks.md)).
- **Strict weights.** Every checkpoint load fails on a missing key instead of leaving random
  weights; weights inference never reads are not built at all.
- **Built on the device.** Models are constructed directly on the target device inside
  `torch.device(...)`, then loaded from memory-mapped checkpoints.

## Related pages

- [Public API](../api/public-api.md)
- [Text front-end](../pipeline/text-frontend.md)
- [Voice and emotion conditioning](../pipeline/voice-and-emotion-conditioning.md)
- [GPT semantic-code generation](../pipeline/gpt-generation.md)
- [Flash engine](../pipeline/flash-engine.md)
- [Acoustic decoding](../pipeline/acoustic-decoding.md)
- [Model directory, loading and caches](../operations/model-loading-and-caches.md)
