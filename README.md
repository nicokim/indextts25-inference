# indextts25-inference

[![CI](https://github.com/nicokim/indextts25-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/nicokim/indextts25-inference/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/indextts25-inference)](https://pypi.org/project/indextts25-inference/)

Inference library for **IndexTTS-2.5**: zero-shot text-to-speech (voice cloning from one
reference clip) with emotion and speed control, and streaming that yields every audio chunk
together with the text that produced it.

It packages the inference code of [index-tts](https://github.com/index-tts/index-tts)
(commit `ee40fa7`) behind a small API, fixes a number of upstream bugs (several of them
changed what the model actually hears) and drops everything inference does not use, about
80 % of the vendored code.

## Installation

```bash
pip install indextts25-inference
```

Python 3.10–3.15 and a CUDA GPU with about 6 GB of free VRAM. The weights are downloaded
separately; the `qwen0.6bemo4-merge/` text-emotion model in that repo is not used by this
library and can be skipped:

```bash
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints --exclude "qwen0.6bemo4-merge/*"
```

One-time costs on a fresh machine:

- **Auxiliary models.** w2v-bert-2.0, CAMPPlus and BigVGAN are read from
  `checkpoints/hf_cache/` if they are there; otherwise only the files the library reads are
  fetched into the standard Hugging Face cache. Set `HF_ENDPOINT` to use a mirror.
- **Text-normalization grammar.** The first synthesis in a language compiles NeMo's grammar
  for it (about 25 s). It is cached in `~/.cache/indextts25/nemo_tn/` (or under
  `$XDG_CACHE_HOME`), and later processes load it in under a second.

### Optional extras

| extra | what it adds |
|---|---|
| `flash` | the accelerated GPT engine behind `attn_backend="flash"` (see [Backends](#backends) and [Recommended stack](#recommended-stack)) |
| `zh` | WeTextProcessing normalization for Chinese and English; without it, both use NeMo |
| `ja` | Japanese word segmentation (`fugashi` + `unidic-lite`), required for `lang="JA"` |

`nemo-text-processing` pins a `pynini` that only ships wheels for Linux x86_64 up to
CPython 3.13, so it is not installed elsewhere, Python 3.14 and 3.15 included. Without it,
text goes to the model unnormalized and digits are read poorly: write numbers out, or
install `pynini` from conda-forge and then `nemo-text-processing`. For Spanish, Python 3.12
or 3.13 is the comfortable choice.

On Python 3.15 PyPI has no torch wheels yet: install torch from the PyTorch index first
(`pip install torch --index-url https://download.pytorch.org/whl/cu130`), then this package.

## Usage

```python
from indextts25 import IndexTTS25

tts = IndexTTS25(model_dir="checkpoints", attn_backend="flash")
tts.warm("voice.wav")           # pay the one-time costs before the first real request

tts.synthesize("Hola, ¿cómo va?", voice="voice.wav", out="out.wav", speed=0.85)
```

`lang` defaults to `"ES"`; the model also speaks `"EN"`, `"ZH"`, `"JA"` and `"AR"` (Arabic
text is not normalized). `warm()` runs one short synthesis: it extracts the voice
conditioning, loads the language's normalization grammar and, with `"flash"`, captures the
CUDA graph. Pass it `lang=` if you will not be synthesizing Spanish.

### Spanish numbers

Spanish text goes through NeMo plus a layer of rules for what NeMo reads wrong
(`indextts25/utils/es_rules.py`), following Argentine usage. `$` is pesos, the Latin American
convention, before or after the amount (`$1.250,50` → "mil doscientos cincuenta pesos con
cincuenta", `1500$` → "mil quinientos pesos"); `US$`, `U$S` and `USD` are dollars, also in
`USD $500` and `$500 USD`. Numbers agree with their noun ("ciento una personas", "a la una",
"la primera vez" for `la 1° vez`) and keep their full form as labels ("capítulo uno") and before
a verb ("en dos mil veintiuno empecé"), unless the label or name is followed by what they count
("ToxicMati un bit"). Heights read "uno setenta", scores "tres a uno", ranges "diez a quince
kilómetros", ordinals "1era" → "primera". `H.MM` is a time (`te espero 21.30`, `21.30 hs` →
"veintiuna y media") unless a unit or a measure follows (`9.58 segundos`, `3.15 %`, `2.50 kms`)
or a version or a grade precedes (`Python 3.11`, `promedio de 7.50`); then it is a decimal, with
"coma" before a unit ("tres coma quince por ciento", `7.5 hs` → "siete coma cinco horas") and
"punto" otherwise. Roman numerals are read only in clear contexts ("siglo veintiuno", "del uno
al doce") and after a regnal name from a fixed list ("Juan Pablo segundo", "Isabel segunda",
"Luis catorce"); other capitals such as "CV", "MI", "GTA V", "Juan M" or the initial in "Juan
D. Perón" stay letters.

### Streaming with synchronized text

Each chunk carries the text it was generated from, which is what subtitles or an on-screen
transcript need:

```python
for chunk in tts.stream("A long text…", voice="voice.wav", lang="EN"):
    play(chunk.audio)               # float32 mono at 22050 Hz
    show(chunk.text)                # the segment that produced this audio
    print(chunk.elapsed, chunk.duration)
```

`chunk.text` is the model input, not your original string: it is normalized (numbers and
times spelled out) and, for Spanish, upper-cased. Text with nothing to pronounce (only
punctuation or emoji) yields no chunks, and `synthesize()` raises `ValueError` for it.
`include_silence=True` also yields the 200 ms pauses between segments.

`segment_tokens` (60 by default for streaming) sets latency and text granularity at the same
time: smaller segments return the first chunk sooner and give finer subtitles. Segments are
cut at punctuation and, when a stretch has none, between words.

### Fine-tuned models

```python
tts = IndexTTS25(model_dir="checkpoints", config="checkpoints/config_my_voice.yaml")
```

A YAML with `gpt_checkpoint:` pointing at your weights. It works for a merged LoRA: the
resulting model has exactly the size and speed of the base one. Checkpoints are loaded
strictly, so a missing key is an error instead of silently leaving random weights.

### Emotion and speed

```python
tts.synthesize(text, voice="voice.wav", out="o.wav",
               emotion="sad_reference.wav", emotion_strength=0.6,
               speed=0.85)              # <1 is faster; below 0.78 it saturates

# or an emotion vector instead of a reference clip, in the order
# [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
tts.synthesize(text, voice="voice.wav", out="o.wav", emo_vector=[0, 0, 0.6, 0, 0, 0, 0, 0])
```

With `emo_vector`, the weight left over (`1 - sum`) goes to the voice's own emotion, so keep
the sum at 0.8 or less; `emotion_strength` scales the vector. Voice and emotion conditioning
are cached per file (path and modification time), so alternating between a few references
costs nothing after the first use of each.

### Generation knobs

`synthesize()` and `stream()` forward extra keyword arguments to the generator:
`temperature` (0.8), `top_p` (0.8), `top_k` (30), `num_beams` (3), `repetition_penalty` (10),
`do_sample` (True), `max_mel_tokens` (1500) and, for the diffusion stage, `diffusion_steps`
(25) and `inference_cfg_rate` (0.7). Fewer diffusion steps trade quality for latency.

### Threads

An instance is not thread-safe: call it from one thread at a time, for example from a single
worker thread. Which thread does not matter, because the library leaves process-wide state
alone: it does not redirect `stdout`/`stderr`, install warning filters or set environment
variables, and it prints nothing unless you pass `verbose=True`.

## Backends

| | `"sdpa"` (default) | `"flash"` |
|---|---|---|
| GPT engine | transformers `generate()` | own decoder: paged KV cache, each step replayed as a CUDA graph |
| decoding | beam sampling (3 beams by default) | sampling, a single beam |
| sampling knobs | repetition penalty, temperature, top-k, top-p | the same, applied in the same order |
| needs | nothing extra | a `flash-attn` wheel matching your torch and CUDA |

`"flash"` ignores `num_beams`. Measured below, it is faster and scores the same as `"sdpa"`
on intelligibility and speaker similarity.

## Performance

RTX 5090 (driver 595, CUDA 13.2), Ryzen 9 9900X, torch 2.9.1+cu130, transformers 4.57.6;
medians of 5 runs of `scripts/benchmark.py` with a Spanish voice. The sentence has 49
characters and the paragraph 425 (34 s of audio); TTFA is the time to the first `stream()`
chunk.

| backend | version | load | warm | TTFA sentence | TTFA paragraph | paragraph | peak VRAM |
|---|---|---|---|---|---|---|---|
| sdpa | 0.1.0 | 9.3 s | 25.6 s | 1.23 s | 2.36 s | 8.94 s | 22.8 GB |
| sdpa | **0.2.0** | 3.0 s | 1.8 s | 0.85 s | 1.43 s | 6.17 s | 6.0 GB |
| sdpa + fp16 | 0.2.0 | 3.0 s | 1.8 s | 0.94 s | 1.54 s | 6.27 s | 4.3 GB |
| flash | 0.1.0 | 14.3 s | 25.7 s | 0.81 s | 1.64 s | 4.67 s | 25.3 GB |
| flash | **0.2.0** | 3.0 s | 1.9 s | 0.73 s | 1.16 s | 4.54 s | 6.3 GB |
| flash + fp16 | 0.2.0 | 3.2 s | 1.9 s | 0.72 s | 1.19 s | 4.56 s | 4.6 GB |

- `warm` is the first synthesis in a new process: 0.1.0 compiled the text-normalization
  grammar every time, 0.2.0 loads it from its cache.
- The peak VRAM of 0.1.0 came mostly from autograd graphs kept alive in its voice cache.
- `fp16=True` runs the GPT in bf16: 1.7 GB less VRAM, but no faster (with `sdpa`, about
  0.1 s more to the first chunk).
- On the recommended stacks below, `sdpa` with torch 2.14.1 measured 0.89 s / 6.39 s
  (TTFA sentence / paragraph) and `flash` with torch 2.10, 0.71 s / 4.30 s.

## Recommended stack

**`"sdpa"`**: nothing to choose. A plain `pip install indextts25-inference` resolves to the
newest releases (torch 2.14.1 and transformers 5.20 at the time of writing), which are
tested.

**`"flash"`**: flash-attn ships only a source distribution on PyPI and prebuilt wheels on
GitHub for a few torch/CUDA/Python combinations. Install torch and the matching wheel
*before* the extra; otherwise pip tries to compile flash-attn. With an NVIDIA driver ≥ 580
(CUDA 13) and Python 3.12:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install "indextts25-inference[flash]"
```

The newest CUDA 13 wheels are for torch 2.10; there are none for torch 2.11 or later. With
an older driver, or another Python version, use torch 2.8 (CUDA 12.8) and the
`cu12torch2.8` wheel for your Python from the
[v2.8.3.post1 release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1);
that is the minimum tested stack below.

## Version compatibility

Each stack ran the benchmark and six cases (Spanish sentence, paragraph and stream,
English, Spanish with an emotion reference, Japanese) on the backends listed, and the audio
was checked with Whisper. With a fixed seed the minimum stack produces bit-identical Spanish
and Japanese audio to torch 2.9.1.

| Python | torch | transformers | flash-attn | backends |
|---|---|---|---|---|
| 3.11 | 2.8.0+cu128 | 4.52.1 | 2.8.3.post1 (cu12) | sdpa, flash |
| 3.12 | 2.9.1+cu130 | 4.57.6 | 2.8.3 (cu13) | sdpa, flash |
| 3.12 | 2.10.0+cu130 | 4.57.6 and 5.20.0 | 2.8.3 (cu13) | flash |
| 3.12 | 2.14.1+cu130 | 5.20.0 | — | sdpa |

Python 3.10, 3.13, 3.14 and 3.15 install and pass the CPU tests (torch 2.14.1); on 3.15 a
CPU synthesis was also checked. CUDA 13 builds of torch need an NVIDIA driver ≥ 580.

## Differences from upstream

**Text**

1. **`nemo-text-processing` is a real dependency.** Upstream does not declare it although it
   needs it for Spanish; without it normalization fails silently and digits reach the
   tokenizer raw.
2. **Normalization order.** Upstream applies its punctuation map (which turns `":"` into
   `","`) *before* normalizing, so `14:30` was read "catorce **coma** treinta". It now runs
   after NeMo ("catorce **y media**"); for zh/en the colon is left to the normalizer, so
   "8:00 AM" is no longer "eight, zero AM".
3. **Exclamation marks reach the model.** In this tokenizer `"!"` has id 0, the same as the
   start-of-text token, and upstream's padding filter deleted every one of them.
4. English and Chinese without WeTextProcessing fall back to NeMo instead of being
   normalized to an empty string (which synthesized about 2 s of noise). With `lang="EN"`,
   text without letters is no longer routed to the Chinese normalizer.
5. A stretch of text without punctuation is split between words, never inside one.

**Generation**

6. **Mel positions.** Generated code *k* now gets mel position *k*, as in training; upstream
   was off by one, which costs about 0.5 nats per token of likelihood on real audio.
7. **No stray stop token.** Upstream appended an extra stop token to every segment; it became
   left padding, and the `flash` engine attended to it.
8. **`flash` samples like transformers.** It applies repetition penalty, temperature, top-k and
   top-p; upstream's engine sampled with temperature alone.

   Together, 3 and 6–8 (plus the GELU kernel in 14) take `flash` from 3.2 % WER and 0.899
   speaker similarity to 2.4 % and 0.908, on par with `sdpa` (2.4 %, 0.907 before and
   after): 48 utterances (20 Spanish and 4 English sentences, two seeds), Whisper
   large-v3-turbo for WER, CAMPPlus cosine for similarity.
9. The `flash` engine works with transformers 5.x (it has its own block forward), and so
   does `sdpa` on 5.x releases that stopped passing an attention mask.
10. Fixed two `flash` crashes: a false prefix-cache hit with segments of 250+ tokens, and
    sequences longer than 2048 tokens.

**Memory and speed**

11. **No autograd.** Upstream computes the CAMPPlus style and the prompt condition with
    autograd on and keeps their graphs alive in its voice cache.
12. Weights inference never uses are not loaded: w2v-bert layers 18–24, the GPT text head,
    the codec encoder, and the second copy of the GPT that the `flash` engine used to make.
13. Models are built directly on the GPU from memory-mapped checkpoints, with weight norm
    folded at load time.
14. The GPT uses the fused tanh-GELU kernel (`gelu_pytorch_tanh`, same formula as
    `gelu_new`): 5–6 % faster.
15. The emotion vector of a (voice, emotion, strength) triple is computed once and reused.

**Library hygiene**

16. **No global side effects.** Upstream sets `HF_HUB_CACHE` (relative to the working
    directory) and silences every `FutureWarning` and `UserWarning` on import; its prints are
    behind `verbose` here. The only global change left is lowering NeMo's and
    WeTextProcessing's loggers, which both set to INFO with their own handler, to WARNING.
17. `attn_backend` by name instead of a `use_accel` boolean; streaming yields `(audio, text)`.
18. Packaged without IndexTTS-2.0, MaskGCT, training and web-UI code, and without
    `openai-whisper`, `torchaudio`, `munch`, `matplotlib` or `modelscope`. The auxiliary
    download fetches only the files it reads (2.5 GB less on a fresh install).

## Development

```bash
pip install -e . pytest
pytest                                   # fast checks: CPU, no weights
INDEXTTS25_MODEL_DIR=checkpoints INDEXTTS25_VOICE=voice.wav pytest   # + end to end on GPU
python scripts/benchmark.py --voice voice.wav --backend flash        # the numbers above
```

## License

Bilibili Model Use License Agreement — see `LICENSE`. This library is a derivative work;
its modifications are not endorsed or guaranteed by the original authors.
