---
type: pipeline-stage
title: "Acoustic decoding: codec, flow matching and BigVGAN"
description: How one segment's semantic codes become a 22.05 kHz waveform — codec decoding to 50 Hz features, length regulation with the speed factor, the conditional flow-matching Euler solver with classifier-free guidance over the DiT, prompt-mel handling, BigVGAN vocoding and PCM output.
tags: [s2mel, flow-matching, dit, vocoder, codec, audio]
sources:
  - id: openwiki-source-42e695b3905ca2603655892f
    resource: repo://indextts25/codec/models.py
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
  - id: openwiki-source-93314986bcd35f2fb0820754
    resource: repo://indextts25/s2mel/modules/bigvgan/bigvgan.py
  - id: openwiki-source-c0349bb60013fe10d9289a76
    resource: repo://indextts25/s2mel/modules/diffusion_transformer.py
  - id: openwiki-source-665e5ec49f3fe5c6fa9c2ccc
    resource: repo://indextts25/s2mel/modules/flow_matching.py
  - id: openwiki-source-abe27bd9a9214db9f982c018
    resource: repo://indextts25/s2mel/modules/length_regulator.py
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T06:43:45.974Z
---

# Acoustic decoding: codec, flow matching and BigVGAN

After the GPT has produced a segment's semantic codes and they have been cut at the first
stop code, `IndexTTS2.infer_generator` turns them into audio in four steps. All of this runs
under `torch.no_grad`, in fp32.

```text
codes (25 Hz) --codec.decode--> features (50 Hz, 1024-d)
  --length regulator--> condition (mel frame rate)    prompt condition (from the voice)
                              \__________ concat __________/
                                         |
                      CFM Euler solver over the DiT  (+ reference mel, style)
                                         |
                        mel (80 bands), prompt frames removed
                                         |
                                   BigVGAN v2 --> waveform (22 050 Hz)
```

## 1. Semantic codec decoding

`EnhancedCodec` (`codec/models.py`) contains only the decoding half of the semantic codec
from Amphion's MaskGCT: a factorized VQ codebook lookup with an output projection, a Vocos
ConvNeXt backbone, a linear layer back to 1024 dimensions, nearest-neighbour upsampling by
2 and a final convolution. It maps codes at 25 Hz to w2v-bert-like features at 50 Hz, the
same space as the conditioning features of the reference voice.

## 2. Length regulation and speed

The number of mel frames for the segment is `int(feature_frames × 1.72 × duration_factor)`;
`duration_factor` is the API's `speed`, so values below 1 produce fewer frames and faster
speech. `InterpolateRegulator` (`s2mel/modules/length_regulator.py`) projects the features,
stretches them to that length with nearest-neighbour interpolation and refines them with
convolution, GroupNorm and Mish blocks. This is the only regulator variant IndexTTS-2.5
uses (continuous input, no VQ, no f0).

The voice's own *prompt condition*, produced the same way from the reference clip during
conditioning ([Voice and emotion conditioning](voice-and-emotion-conditioning.md)), is
prepended, so the decoder sees `[prompt | target]` as one sequence.

## 3. Conditional flow matching

`CFM.inference` (`s2mel/modules/flow_matching.py`) integrates an ODE from Gaussian noise
with an explicit Euler scheme of `diffusion_steps` steps (default 25) over `t ∈ [0, 1]`:

- the prompt region of the state is pinned to zero at every step, and the reference mel
  is supplied separately as `prompt_x` for those frames;
- with `inference_cfg_rate > 0` (default 0.7) each step evaluates the estimator once on a
  batch of two, the conditional input and a copy with prompt, style and content zeroed,
  and combines them as `(1 + w) · v_cond − w · v_uncond`. The unconditional branch is
  assembled once, not per step;
- the estimator is the DiT below; its output is the velocity `dφ/dt`.

The cost of the stage scales linearly with `diffusion_steps`, which is why the API exposes
it.

### The DiT estimator

`DiT` (`s2mel/modules/diffusion_transformer.py`) concatenates, per frame, the noisy state,
the prompt mel, the projected content condition and the 192-dimensional style vector, merges
them with a linear layer, and runs a non-causal transformer (`s2mel/modules/gpt_fast/`,
rotary positions, timestep-conditioned normalization, U-ViT skip connections when
configured). A long skip connection re-injects the noisy input, and the configured final
layer is a WaveNet stack (`s2mel/modules/wavenet.py`) conditioned on a second timestep
embedding. Because inference always passes the full length, no attention mask is built.
A causal DiT configuration is rejected at construction.

## 4. Vocoding and output

The prompt frames are cut from the generated mel, and BigVGAN v2
(`s2mel/modules/bigvgan/bigvgan.py`, `nvidia/bigvgan_v2_22khz_80band_256x`) renders the
remaining 80-band mel to a waveform at 22 050 Hz with anti-aliased Snake activations and
plain PyTorch ops (no fused CUDA kernel). The output is clipped to [-1, 1] (or passed
through `tanh` if the config says so).

The engine then scales the waveform by 32767, clamps it to the int16 range and moves it to
the CPU. In streaming mode the segment is yielded at once (with its text when
`emit_text=True`), followed by `interval_silence` ms of zeros unless it is the last
segment; otherwise segments are collected and joined with the same silence at the end
([Public API](../api/public-api.md)).

## Related pages

- [GPT semantic-code generation](gpt-generation.md)
- [Voice and emotion conditioning](voice-and-emotion-conditioning.md)
- [Model directory, loading and caches](../operations/model-loading-and-caches.md)
