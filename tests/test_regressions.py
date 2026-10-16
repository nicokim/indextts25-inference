"""Regression checks.

The fast ones run on CPU without weights. The end-to-end one needs a GPU and the model:

    pytest -q
    INDEXTTS25_MODEL_DIR=checkpoints INDEXTTS25_VOICE=voice.wav pytest -q
"""
import importlib.util
import os
import re
from types import SimpleNamespace as NS

import pytest
import torch

from indextts25.gpt.model_v2 import _past_len
from indextts25.infer_v2_5 import IndexTTS2, apply_pronunciation_annotations
from indextts25.utils.checkpoint import fold_weight_norm
from indextts25.utils.common import kaldi_fbank, resample
from indextts25.utils.front import TextNormalizer


def test_ja_text_process_is_a_property():
    # A @torch.no_grad() on top of the @property turned it into a function: Japanese crashed.
    assert isinstance(IndexTTS2.__dict__["ja_text_process"], property)


def test_get_emb_does_not_record_autograd():
    # Without no_grad, the voice cache kept the w2v-bert graph alive (~10 GB with a 15 s reference).
    lin = torch.nn.Linear(4, 4)
    fake = NS(
        semantic_model=lambda input_features, attention_mask, output_hidden_states: NS(
            hidden_states=[lin(input_features)] * 18),
        semantic_mean=torch.zeros(4), semantic_std=torch.ones(4),
    )
    assert not IndexTTS2.get_emb(fake, torch.randn(1, 3, 4), None).requires_grad


def test_normalizer_without_wetext_keeps_the_text():
    # Without WeTextProcessing, normalize() returned "" and ~2 s of noise got synthesized.
    tn = TextNormalizer()
    tn._zh_en_error = ImportError("simulated")
    assert tn.normalize("It costs 3 dollars.", "en") == "It costs 3 dollars."


def _splitter():
    tts = object.__new__(IndexTTS2)
    tts.tokenizer = NS(encode=lambda text, allowed_special=None: list(text))  # 1 token per character
    tts.gpt = NS(text_pos_embedding=NS(emb=NS(num_embeddings=602)))
    return tts


def test_split_text_by_tokens_cuts_between_words():
    text = "una frase larga sin puntuacion que obliga a cortar entre palabras " * 3
    segments = _splitter().split_text_by_tokens(text, 20)
    assert "".join(segments) == text
    assert all(len(s) <= 20 for s in segments)
    for a, b in zip(segments, segments[1:]):
        assert not (a[-1].isalnum() and b[0].isalnum()), (a, b)


def test_split_text_by_tokens_keeps_annotations_whole():
    tag = "<|SPECIAL_TOKEN_1|>G OW1<|SPECIAL_TOKEN_1|>"
    segments = _splitter().split_text_by_tokens("hola. " * 5 + tag + " chau. " * 5, 40)
    assert any(tag in s for s in segments)


def test_pronunciation_annotations():
    assert apply_pronunciation_annotations("<going|g ow1>") == "<|SPECIAL_TOKEN_1|>G OW1<|SPECIAL_TOKEN_1|>"


def test_past_len_on_both_cache_types():
    from transformers import DynamicCache

    assert _past_len(None) == 0
    assert _past_len(((torch.zeros(1, 2, 5, 4), torch.zeros(1, 2, 5, 4)),)) == 5
    # In transformers 5.x an empty Cache is truthy: input_ids must not be trimmed.
    assert _past_len(DynamicCache()) == 0


def test_fold_weight_norm_matches_the_hook():
    with pytest.warns(FutureWarning):
        conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(4, 6, 3))
    plain = torch.nn.Conv1d(4, 6, 3)
    plain.load_state_dict(fold_weight_norm(conv.state_dict(), "cpu"))
    x = torch.randn(1, 4, 10)
    assert torch.equal(plain(x), conv(x))


def test_vendored_audio_matches_torchaudio():
    ta = pytest.importorskip("torchaudio")
    x = torch.randn(1, 24000)
    y = resample(x, 24000, 16000)
    assert torch.equal(y, ta.transforms.Resample(24000, 16000)(x))
    assert torch.equal(kaldi_fbank(y), ta.compliance.kaldi.fbank(y, num_mel_bins=80, dither=0, sample_frequency=16000))


@pytest.mark.skipif(
    not (os.environ.get("INDEXTTS25_MODEL_DIR") and os.environ.get("INDEXTTS25_VOICE") and torch.cuda.is_available()),
    reason="needs a GPU, INDEXTTS25_MODEL_DIR and INDEXTTS25_VOICE",
)
@pytest.mark.parametrize("backend", ["sdpa", "flash"])
def test_end_to_end(tmp_path, backend):
    if backend == "flash" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash-attn not installed")
    from indextts25 import IndexTTS25

    voice = os.environ["INDEXTTS25_VOICE"]
    tts = IndexTTS25(model_dir=os.environ["INDEXTTS25_MODEL_DIR"], attn_backend=backend)

    out = tts.synthesize("Son las 14:30.", voice=voice, out=tmp_path / "es.wav")
    assert out.stat().st_size > 22050  # more than ~0.5 s of audio

    chunks = list(tts.stream("Hola. ¿Cómo va?", voice=voice))
    assert chunks and all(c.text and len(c.audio) for c in chunks)
    assert list(tts.stream("...", voice=voice)) == []
    with pytest.raises(ValueError):
        tts.synthesize("!!!", voice=voice, out=tmp_path / "empty.wav")

    # The voice cache does not retain autograd graphs.
    assert not any(t.requires_grad for t in next(iter(tts._tts._spk_cache.values())))

    # Segments over 256 tokens: the flash engine's prefix cache crashed on the 2nd one.
    long_text = " ".join(re.findall(r"\w+", "La inteligencia artificial cambió la forma en que trabajamos " * 40))
    tts.synthesize(long_text + ".", voice=voice, out=tmp_path / "long.wav", segment_tokens=300)

    if importlib.util.find_spec("fugashi"):
        tts.synthesize("こんにちは、今日はいい天気ですね。", voice=voice, out=tmp_path / "ja.wav", lang="JA")
