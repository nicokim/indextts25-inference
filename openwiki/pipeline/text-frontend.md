---
type: pipeline-stage
title: "Text front-end: normalization, casing and segmentation"
description: How raw input text becomes the token segments the GPT reads — language validation, per-language normalization with NeMo or WeTextProcessing, why the punctuation map runs after normalization, casing, pronunciation annotations, Japanese segmentation, the tiktoken tokenizer, and token-budget segmentation.
tags: [text-normalization, nemo, wetextprocessing, tokenizer, segmentation, i18n]
sources:
  - id: openwiki-source-bf6745cef9c9e3b63f473128
    resource: repo://indextts25/infer_v2_5.py
  - id: openwiki-source-3672c72f276515deb6a3d235
    resource: repo://indextts25/utils/front.py
  - id: openwiki-source-8ec925e6ba68e143e4b0d6c5
    resource: repo://indextts25/utils/nemo_tn.py
  - id: openwiki-source-01840b5abb4687ea7b428fe2
    resource: repo://indextts25/utils/tokenizer.py
  - id: openwiki-source-310a7347cd31a1985b43c333
    resource: repo://tests/test_regressions.py
generated: { by: "claude-code", at: "2026-10-16T06:02:56.648Z" }
verified:
  - by: openwiki/0.7.2
    at: 2026-10-16T07:18:19.761Z
---

# Text front-end: normalization, casing and segmentation

The model reads text in a canonical written form: numbers, times and symbols spelled out,
punctuation reduced to a small set, a fixed casing per language. `IndexTTS2._prepare_text`
(`indextts25/infer_v2_5.py`) produces that form, and `split_text_by_tokens` cuts it into
segments the GPT can handle. The text of each segment is also what `stream()` returns as
`Chunk.text`.

## Language codes

`lang` is case-insensitive and must be a key of the tokenizer's language table (the
Whisper list of about a hundred ISO 639-1 codes) or `"zhen"` for mixed Chinese-English;
anything else raises `ValueError`. The model itself is trained for Chinese, English,
Japanese, Spanish and Arabic. The language selects the normalizer, the casing, the
`<|lang|> ` prefix of every segment and the GPT's language embedding.

## Normalization

| language | normalizer |
|---|---|
| `es` | NeMo (`utils/nemo_tn.py`) |
| `en`, `zh` | WeTextProcessing (`utils/front.py`) when the `zh` extra is installed, otherwise NeMo |
| `zhen` | WeTextProcessing when installed, otherwise none |
| `ja` | none (NeMo's Japanese grammar is deliberately not enabled) |
| anything else | none |

**NeMo.** `TextNormalizer` in `utils/nemo_tn.py` maps service codes (and ISO 639-3 codes
such as `spa`) to NeMo languages, builds one cased `Normalizer` per language on first use
and keeps it for the life of the process (see
[Model directory, loading and caches](../operations/model-loading-and-caches.md) for the
on-disk pickle). If NeMo is missing, a grammar fails to build or a call raises, it logs a
warning and returns the text unchanged, so synthesis never fails because of normalization.

**WeTextProcessing.** `TextNormalizer.load()` in `utils/front.py` imports the Chinese and
English normalizers lazily on the first zh/en request and remembers an import failure, after
which zh/en fall back to NeMo. Before normalizing it protects things the grammars would
damage: `<word|reading>` annotations, English contractions like *what's*, hyphenated technical
terms (*GPT-5-nano*), pinyin with tone digits and Chinese names, plus optional glossary
terms from `model_dir/glossary.yaml`. With `lang="en"` it never routes text to the Chinese
grammar, even when the text has no Latin letters.

## The punctuation map and its ordering

`char_rep_map` in `utils/front.py` reduces full-width and typographic punctuation to a
small set (`：；，` to `,`, quotes and brackets to `'`, dashes and tildes to `-`, ellipses to
`…`, newlines to spaces) and also maps `":"` to `","`. Applied before normalization, that
turned `14:30` into `14,30`, which NeMo reads as a decimal ("catorce coma treinta"). So:

- **es, ja:** the map runs only after normalization.
- **zh, zhen, en:** the map runs before normalization (the grammars need the other
  replacements) but keeps `:` and `：`, which the normalizer interprets as times; then it
  runs again after normalization.
- **other languages:** the map runs before (no normalization happens).

## Casing, annotations and Japanese

After normalization the text is lower-cased for `ja`, `zh`, `zhen` and `en` and upper-cased
for `es`. Pronunciation annotations `<word|reading>` become the reading wrapped in
`<|SPECIAL_TOKEN_1|>` (non-Chinese word, e.g. CMU phonemes) or `<|SPECIAL_TOKEN_2|>`
(Chinese word, e.g. pinyin), upper-cased; a kana reading is inserted directly. Japanese text
is then segmented into words separated by spaces with fugashi (or a system MeCab), which is
why the `ja` extra is required for Japanese. Finally every `<|...|>` token name is
upper-cased.

## Tokenizer

`get_encoding` (`utils/tokenizer.py`) builds a `tiktoken.Encoding` from
`multilingual_zh_ja_yue_char_del.tiktoken` in the model directory, with Whisper-style
special tokens appended in a fixed order (end-of-text, start-of-transcript, the language
tokens, audio events, emotions, task tokens, `SPECIAL_TOKEN_1…30`, TTS tokens and
timestamps). The order defines their ids, so it must not change. Text is encoded with
`allowed_special='all'`. In this vocabulary id 0 is `"!"`, which is also the GPT's
start-of-text id ([GPT generation](gpt-generation.md)).

## Segmentation

`split_text_by_tokens(text, max_tokens, lang_prefix)` returns the text unchanged when it fits
the budget: `min(max_tokens, text-position capacity − 2)` minus the tokens of the language
prefix. Otherwise it:

1. keeps every `<|SPECIAL_TOKEN_n|>…<|SPECIAL_TOKEN_n|>` span as an indivisible piece;
2. splits the rest after sentence and clause punctuation (Latin and CJK);
3. splits any piece that is still too long between words, and only a single word that alone
   exceeds the budget (for example Chinese without spaces) between characters;
4. greedily packs the pieces into segments up to the budget.

Concatenating the segments gives back the input text. `segment_tokens` in the public API is
this `max_tokens`: 60 for `stream()` and 120 for `synthesize()` by default.

## Related pages

- [Public API](../api/public-api.md)
- [GPT semantic-code generation](gpt-generation.md)
- [Model directory, loading and caches](../operations/model-loading-and-caches.md)
