---
type: pipeline-stage
title: Voice and emotion conditioning
description: How a reference clip becomes the conditioning the GPT and the acoustic decoder use — audio loading and truncation, w2v-bert layer-17 features, the CAMPPlus style vector, the reference mel and prompt condition, emotion encoding and mixing, the emo_vector path, and the per-file caches.
tags: [conditioning, voice-cloning, emotion, w2v-bert, campplus, caching]
sources:
  - id: openwiki-source-0908f962bcb3f4aab17c1ae5
    resource: repo://indextts25/gpt/model_v2.py
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
  - id: openwiki-source-310a7347cd31a1985b43c333
    resource: repo://tests/test_regressions.py
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T07:18:19.761Z
---

# Voice and emotion conditioning

IndexTTS-2.5 clones a voice from one reference clip and takes its emotion either from the
same clip, from a second clip, or from an explicit eight-value emotion vector. All of this
is computed in `IndexTTS2` (`indextts25/infer_v2_5.py`) before the first segment is
generated, without autograd, and cached per file.

## Speaker clip

`_speaker_conditioning` produces four tensors from the voice clip:

1. **Loading.** librosa loads the clip resampled to 22 050 Hz, and anything beyond 15 s is
   cut off.
2. **Semantic features (`spk_cond_emb`).** The clip is resampled to 16 kHz with the
   vendored sinc resampler, turned into SeamlessM4T input features and run through
   w2v-bert-2.0. `get_emb` takes the hidden state after layer 17 and normalizes it with the
   mean and standard deviation from `wav2vec2bert_stats.pt`. Only 17 layers are built,
   since later layers never influence that output.
3. **Style (`style`).** An 80-bin Kaldi filterbank of the 16 kHz audio, mean-normalized over
   time, goes through CAMPPlus (`s2mel/modules/campplus/`) to give a 192-dimensional speaker
   embedding. It conditions the GPT (through `spk_emb_proj`), the DiT and the emotion-vector
   lookup.
4. **Reference mel and prompt condition.** The mel spectrogram of the 22.05 kHz audio is the
   prompt the CFM continues from; the length regulator stretches `spk_cond_emb` to the mel's
   frame count to form the matching `prompt_condition`
   ([Acoustic decoding](acoustic-decoding.md)).

## Emotion clip

`_emotion_conditioning` loads the emotion clip directly at 16 kHz (also capped at 15 s) and
keeps only its w2v-bert layer-17 features. When no emotion clip is given, the voice clip
is used and `emotion_strength` is forced to 1.

## Mixing emotion

The GPT turns features into an emotion vector with its own encoder: a Conformer
(`gpt/conformer_encoder.py`) followed by a one-latent perceiver resampler
(`gpt/perceiver.py`) and two linear layers. `merge_emovec` encodes both the voice and the
emotion features and interpolates:

```text
emovec = base + alpha * (emotion - base)        alpha = emotion_strength
```

The result is added to the projected speaker style to form the GPT's conditioning prefix
([GPT semantic-code generation](gpt-generation.md)). It depends only on the voice, the
emotion source and alpha, so it is computed once and cached.

## Emotion vectors

With `emo_vector` (eight weights in the order happy, angry, sad, afraid, disgusted,
melancholic, surprised, calm), the emotion clip is ignored and:

1. `emotion_strength`, clamped to [0, 1], scales the weights;
2. for each of the eight emotion groups of `feat2.pt` (sizes from the config's `emo_num`),
   one row is chosen: the row whose counterpart in the speaker matrix `feat1.pt` is most
   cosine-similar to the voice's style, or a random row with `use_random=True`;
3. the weighted sum of those rows is blended with the voice's own emotion vector:
   `emovec_mat + (1 − Σw) · emovec`. Weights summing above 1 give the voice's own emotion
   a negative weight, which is why the README advises keeping the sum at 0.8 or less.

## Caching

Conditioning is the most expensive per-voice work, so three dicts on the engine cache it:

| cache | key | value |
|---|---|---|
| `_spk_cache` | voice file | `spk_cond_emb`, `style`, `prompt_condition`, reference mel |
| `_emo_cache` | emotion file | emotion `spk_cond_emb` |
| `_emovec_cache` | (voice file, emotion file, alpha) | merged emotion vector |

A file key is its absolute path plus `st_mtime_ns`, so editing a reference clip in place
invalidates it. Each cache keeps the eight most recent insertions (oldest evicted first).
Everything runs under `torch.no_grad`: no autograd graph is kept alive inside the caches,
which a regression test checks. `warm()` fills the caches for its voice.

## Related pages

- [Public API](../api/public-api.md)
- [GPT semantic-code generation](gpt-generation.md)
- [Acoustic decoding](acoustic-decoding.md)
- [Model directory, loading and caches](../operations/model-loading-and-caches.md)
