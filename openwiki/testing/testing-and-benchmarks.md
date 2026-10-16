---
type: testing
title: Testing and benchmarks
description: The regression test suite (fast CPU checks and a GPU end-to-end test enabled by environment variables), what each test guards against, how to run it, the benchmark script behind the README numbers, and the seed-reproducibility technique used to verify refactors.
tags: [testing, pytest, benchmarks, reproducibility, regression]
sources:
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-46fa34397e41ebf7491c7359
    resource: repo://.github/workflows/release-please.yml
  - id: openwiki-source-4d1d392666be6dfdd7a91a2e
    resource: repo://.github/workflows/release.yml
  - id: openwiki-source-f99a73fe01bdf87e2b7d0a4d
    resource: repo://indextts25/__init__.py
  - id: openwiki-source-0719484fdaa00f6fd5bf6bb4
    resource: repo://indextts25/accel/accel_engine.py
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
  - id: openwiki-source-6192c6338f9c3e6860ecd660
    resource: repo://scripts/benchmark.py
  - id: openwiki-source-310a7347cd31a1985b43c333
    resource: repo://tests/test_regressions.py
generated: { by: "claude-code", at: "2026-10-16T07:18:19.761Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T14:28:52.827Z
---

# Testing and benchmarks

## Running the tests

```bash
pip install -e . pytest                 # or: uv sync --group test
pytest                                  # CPU checks, no weights needed
INDEXTTS25_MODEL_DIR=checkpoints INDEXTTS25_VOICE=voice.wav pytest   # + GPU end to end
```

`pyproject.toml` points pytest at `tests/`. Everything lives in `tests/test_regressions.py`.
Each test is named after the bug it prevents from coming back.

## CPU checks

They need neither weights nor a GPU and run in a few seconds:

| test | guards against |
|---|---|
| `test_ja_text_process_is_a_property` | a decorator placed above `@property` turned the Japanese processor into a plain function, so Japanese crashed |
| `test_get_emb_does_not_record_autograd` | w2v-bert features recorded autograd and the voice cache kept the graph alive (gigabytes of VRAM) |
| `test_normalizer_without_wetext_keeps_the_text` | without WeTextProcessing, English/Chinese text was normalized to an empty string |
| `test_split_text_by_tokens_cuts_between_words` | segmentation cutting inside words, or losing characters |
| `test_split_text_by_tokens_keeps_annotations_whole` | segmentation splitting a `<|SPECIAL_TOKEN_n|>` annotation |
| `test_pronunciation_annotations` | the `<word|reading>` expansion |
| `test_past_len_on_both_cache_types` | cache-length detection on legacy tuples and on an empty transformers `Cache` (which is truthy) |
| `test_fold_weight_norm_matches_the_hook` | folded weight-norm weights giving a different output from `torch.nn.utils.weight_norm` |
| `test_vendored_audio_matches_torchaudio` | the vendored resampler and filterbank drifting from torchaudio (skipped when torchaudio is not installed) |

Several of these use small stand-ins instead of real models, for example a fake w2v-bert or
a tokenizer that counts one token per character.

## GPU end-to-end test

`test_end_to_end` runs only when `INDEXTTS25_MODEL_DIR` and `INDEXTTS25_VOICE` are set and
CUDA is available, once per backend (`flash` is skipped without flash-attn). With the real
model it checks that:

- `synthesize()` writes more than about half a second of audio for a time expression;
- `stream()` yields chunks that all carry text and audio, yields nothing for `"..."`, and
  `synthesize("!!!")` raises `ValueError`;
- the cached voice conditioning holds no autograd state;
- a segment of more than 256 tokens synthesizes (the old flash prefix cache crashed on the
  second such segment);
- Japanese synthesizes when fugashi is installed.

It asserts behavior, not exact audio, because output differs across torch versions and GPUs.

## Benchmark

`scripts/benchmark.py` reproduces the README's performance table:

```bash
python scripts/benchmark.py --model-dir checkpoints --voice voice.wav --backend flash [--fp16] [--reps 5]
```

It reports the time to import and construct `IndexTTS25`, the time of `warm()`, the median
time to the first `stream()` chunk for a one-sentence and a 425-character Spanish text, the
median time to `synthesize()` the paragraph (with characters per second and real-time
factor) and the peak allocated VRAM. Each repetition reseeds torch.

## Verifying refactors: bit-identical output

With a fixed seed and the same software stack, synthesis is deterministic down to the bit,
so refactors were verified by hashing the int16 PCM of a fixed set of cases before and
after a change, on every supported stack. Two details keep that property:

- model construction runs inside `torch.random.fork_rng`, so seeding before or after
  creating `IndexTTS25` gives the same audio;
- numerically neutral changes are kept neutral on purpose: for example, the flash engine's
  CUDA graph keeps a 2048-token block table because flash-attn splits work by that width,
  and weight-norm folding uses the same operation as the original hook.

Changes that alter the numbers (the generation fixes listed in the README) were instead
evaluated by intelligibility and speaker similarity on a fixed set of utterances.

## Continuous integration and releases

On every pull request and push to `main`, `.github/workflows/ci.yml` runs two jobs with uv:
`ruff check` (the `dev` group) and the CPU test suite with coverage (the `test` group),
uploading the report to Codecov. GitHub runners have no GPU, so the end-to-end test is
skipped there.

Releases follow conventional commits. On pushes to `main`, release-please keeps a release
pull request up to date with the next version and the changelog; it updates the version in
`pyproject.toml` and `__version__` in `indextts25/__init__.py`. Merging that pull request
creates the `vX.Y.Z` tag, and the tag triggers `.github/workflows/release.yml`, which builds
the package with `uv build` and publishes it to PyPI through trusted publishing (the `pypi`
environment, no stored PyPI token).

## Related pages

- [Public API](../api/public-api.md)
- [Flash engine](../pipeline/flash-engine.md)
- [Dependencies, extras and version compatibility](../operations/dependencies-and-compatibility.md)
