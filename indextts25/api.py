"""High-level API for IndexTTS-2.5.

Wraps `infer_v2_5.IndexTTS2` with what real use needs: attention backends
selected by name, input validation, streaming that emits the text alongside the
audio, and selectable finetuned models. The engine caches each voice's
conditioning per file.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

AttnBackend = Literal["sdpa", "flash"]

#: Segment size measured as optimal for streaming on an RTX 5090: same
#: time-to-first-audio as 40 but with a better RTF.
STREAM_SEGMENT_TOKENS = 60


def _has_speech(text: str) -> bool:
    # isalnum() covers CJK and kana: text made only of punctuation or emojis has nothing to say.
    return any(c.isalnum() for c in text)


@dataclass(frozen=True)
class Chunk:
    """An audio chunk from the stream, with the text that produced it."""
    audio: np.ndarray          # float32 in [-1, 1], mono
    text: str                  # model input (normalized; "" for silences)
    sample_rate: int
    elapsed: float             # seconds since synthesis started
    is_silence: bool

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


class IndexTTS25:
    """IndexTTS-2.5 synthesis.

        tts = IndexTTS25(attn_backend="flash")
        tts.synthesize("Hola, ¿cómo va?", voice="ref.wav", out="out.wav")

        for chunk in tts.stream("Un texto largo...", voice="ref.wav"):
            play(chunk.audio); show(chunk.text)

    An instance is not thread-safe: call it from one thread at a time (any thread:
    the library touches no process-global state).
    """

    SAMPLE_RATE = 22050

    def __init__(
        self,
        model_dir: str | Path = "checkpoints",
        config: str | Path | None = None,
        attn_backend: AttnBackend = "sdpa",
        fp16: bool = False,
        device: str | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            model_dir: folder with gpt.pth, s2mel.pth, codec.pth, the .tiktoken file and hf_cache/.
            config: YAML to use. Defaults to `model_dir/config.yaml`. Point it at a
                config with its own `gpt_checkpoint:` to load a finetuned model.
            attn_backend: "sdpa" (default, no extra dependencies) or "flash"
                (accelerated GPT2 engine with paged KV cache and CUDA graphs; requires
                `pip install indextts25-inference[flash]`). "flash" does no beam search:
                it ignores `num_beams`.
            fp16: GPT in bf16. Saves ~1.7 GB of VRAM but is no faster (with "sdpa" the
                first chunk takes ~0.1 s longer).
            device: "cuda:0", "cpu", ... Defaults to the first available GPU.
            verbose: print the details of each synthesis (normalized text, segments,
                per-stage timings). Without it the library writes nothing to stdout.
        """
        model_dir = Path(model_dir)
        cfg = Path(config) if config else model_dir / "config.yaml"
        if not cfg.exists():
            raise FileNotFoundError(f"Config not found: {cfg}")

        if attn_backend == "flash":
            try:
                import flash_attn  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    'attn_backend="flash" requires flash-attn. Install '
                    "indextts25-inference[flash], or use attn_backend=\"sdpa\"."
                ) from e

        from .infer_v2_5 import IndexTTS2

        self._verbose = verbose
        self._tts = IndexTTS2(
            cfg_path=str(cfg), model_dir=str(model_dir), device=device,
            use_accel=(attn_backend == "flash"), use_bf16=fp16,
        )
        self.attn_backend = attn_backend

    # ---- voices ---------------------------------------------------------

    def warm(self, voice: str | Path, lang: str = "ES") -> None:
        """Run a short synthesis to take everything that is paid only once off the
        critical path: the voice conditioning (w2v-bert, CAMPPlus), the language's
        normalization grammar and, with "flash", the CUDA graph capture.
        """
        for _ in self._tts.infer_generator(
            spk_audio_prompt=str(voice), text="Hola.", output_path=None, lang=lang,
            stream_return=True, verbose=self._verbose,
        ):
            pass

    # ---- synthesis ------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: str | Path,
        out: str | Path,
        lang: str = "ES",
        speed: float = 1.0,
        emotion: str | Path | None = None,
        emotion_strength: float = 1.0,
        segment_tokens: int = 120,
        **sampling,
    ) -> Path:
        """Synthesize the whole `text` to a WAV file.

        Args:
            speed: 1.0 is the native pace; <1 speeds up. IndexTTS-2.5's `duration_factor`,
                applied in the s2mel. Below 0.78 it saturates and stops speeding up.
            emotion: emotion reference audio, separate from the timbre one.
            sampling: passed to the generator (temperature, top_p, top_k, num_beams,
                repetition_penalty, max_mel_tokens, diffusion_steps, ...).
        """
        if not _has_speech(text):
            raise ValueError("the text has nothing to synthesize")
        self._tts.infer(
            spk_audio_prompt=str(voice), text=text, output_path=str(out), lang=lang,
            emo_audio_prompt=str(emotion) if emotion else None,
            emo_alpha=emotion_strength, duration_factor=speed,
            max_text_tokens_per_segment=segment_tokens,
            text_normalization=True, verbose=self._verbose, **sampling,
        )
        return Path(out)

    def stream(
        self,
        text: str,
        voice: str | Path,
        lang: str = "ES",
        speed: float = 1.0,
        emotion: str | Path | None = None,
        emotion_strength: float = 1.0,
        segment_tokens: int = STREAM_SEGMENT_TOKENS,
        include_silence: bool = False,
        **sampling,
    ) -> Iterator[Chunk]:
        """Generate the audio in chunks, each with the text that produced it.

        `segment_tokens` governs both latency and text granularity: smaller returns
        the first chunk sooner and gives finer-grained subtitles. Each chunk's text is
        the model input: normalized (numbers spelled out) and, for Spanish,
        uppercased. Text with nothing pronounceable yields no chunks.
        """
        if not _has_speech(text):
            return
        import torch

        t0 = time.perf_counter()
        gen = self._tts.infer_generator(
            spk_audio_prompt=str(voice), text=text, output_path=None, lang=lang,
            emo_audio_prompt=str(emotion) if emotion else None,
            emo_alpha=emotion_strength, duration_factor=speed,
            stream_return=True, emit_text=True,
            max_text_tokens_per_segment=segment_tokens,
            text_normalization=True, verbose=self._verbose, **sampling,
        )
        for wav, seg_text in gen:
            silent = not seg_text
            if silent and not include_silence:
                continue
            audio = (wav.reshape(-1).to(torch.float32) / 32767.0).numpy()
            yield Chunk(audio=audio, text=seg_text.strip(), sample_rate=self.SAMPLE_RATE,
                        elapsed=time.perf_counter() - t0, is_silence=silent)
