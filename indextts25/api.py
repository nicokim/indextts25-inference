"""API de alto nivel para IndexTTS-2.5.

Envuelve `infer_v2_5.IndexTTS2` con lo que hace falta para uso real: backends de
atención por nombre, voces precomputadas (en 2.5 el hablante es un vector CAMPPlus
de 192 flotantes, así que una voz cacheada pesa 768 bytes), streaming que emite el
texto junto al audio, y modelos finetuneados seleccionables.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np

AttnBackend = Literal["sdpa", "flash"]

#: Tamaño de segmento medido como óptimo para streaming en RTX 5090: mismo
#: time-to-first-audio que 40 pero sin partir palabras y con mejor RTF.
STREAM_SEGMENT_TOKENS = 60


class _Quiet:
    """Silencia los print() y las barras tqdm de upstream.

    El código de upstream imprime sin condicionar (">> gpt_gen_time", progreso del
    difusor, avisos de carga). Una librería no debe escupir eso al stdout de quien la
    usa. Se aplica por operación, no globalmente, así el `verbose=True` sigue sirviendo.
    """

    def __init__(self, activo: bool):
        self.activo = activo
        self._o = self._e = None

    def __enter__(self):
        if self.activo:
            self._o, self._e = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        return self

    def __exit__(self, *exc):
        if self.activo:
            sys.stdout, sys.stderr = self._o, self._e
        return False


@dataclass(frozen=True)
class Chunk:
    """Un trozo de audio del stream, con el texto que lo produjo."""
    audio: np.ndarray          # float32 en [-1, 1], mono
    text: str                  # "" en los silencios de separación
    sample_rate: int
    elapsed: float             # segundos desde que arrancó la síntesis
    is_silence: bool

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


class IndexTTS25:
    """Síntesis IndexTTS-2.5.

        tts = IndexTTS25(attn_backend="flash")
        tts.synthesize("Hola, ¿cómo va?", voice="ref.wav", out="salida.wav")

        for chunk in tts.stream("Un texto largo...", voice="ref.wav"):
            reproducir(chunk.audio); mostrar(chunk.text)
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
            model_dir: carpeta con gpt.pth, s2mel.pth, codec.pth, el tiktoken y hf_cache/.
            config: YAML a usar. Por defecto `model_dir/config.yaml`. Apuntá acá un
                config con `gpt_checkpoint:` propio para cargar un modelo finetuneado.
            attn_backend: "sdpa" (por defecto, sin dependencias extra) o "flash"
                (motor GPT2 acelerado con KV cache paginada y CUDA graphs; requiere
                `pip install indextts25-inference[flash]`). En una RTX 5090 "flash"
                mide 1,77x en párrafos y baja el time-to-first-audio de 1,06 a 0,82 s.
            fp16: autocast en bfloat16. Ahorra ~23 % de VRAM pero **es más lento**
                con batch 1 (los casts no se amortizan); activalo sólo si te falta memoria.
        """
        from .infer_v2_5 import IndexTTS2

        model_dir = Path(model_dir)
        cfg = Path(config) if config else model_dir / "config.yaml"
        if not cfg.exists():
            raise FileNotFoundError(f"No existe el config: {cfg}")

        if attn_backend == "flash":
            try:
                import flash_attn  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    'attn_backend="flash" necesita flash-attn. Instalá '
                    "indextts25-inference[flash], o usá attn_backend=\"sdpa\"."
                ) from e

        self._verbose = verbose
        with _Quiet(not verbose):
            self._tts = IndexTTS2(
                cfg_path=str(cfg), model_dir=str(model_dir), device=device,
                use_cuda_kernel=False, use_accel=(attn_backend == "flash"), use_bf16=fp16,
            )
        self.attn_backend = attn_backend

    # ---- voces ----------------------------------------------------------

    def warm(self, voice: str | Path) -> None:
        """Precalcula el condicionamiento de una voz.

        La primera síntesis con una voz nueva paga la extracción w2v-bert del audio de
        referencia, que en streaming se cobra entera en el primer chunk. Llamar a `warm()`
        al arrancar mueve ese costo fuera del camino crítico.
        """
        with _Quiet(not self._verbose):
            for _ in self._tts.infer_generator(
                spk_audio_prompt=str(voice), text="Hola.", output_path=None, lang="ES",
                stream_return=True, verbose=False,
            ):
                pass

    # ---- síntesis -------------------------------------------------------

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
        """Sintetiza `text` completo a un archivo.

        Args:
            speed: 1.0 es el ritmo nativo; <1 acelera. `duration_factor` de 2.5 aplicado
                en el s2mel. Por debajo de 0,78 satura y deja de acelerar.
            emotion: audio de referencia de emoción, separado del de timbre.
        """
        with _Quiet(not self._verbose):
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
        """Genera el audio por trozos, cada uno con el texto que lo produjo.

        `segment_tokens` gobierna a la vez la latencia y la granularidad del texto:
        más chico devuelve antes el primer chunk y da subtítulos más finos, pero por
        debajo de ~40 tokens la partición corta palabras al medio.
        """
        import torch

        t0 = time.perf_counter()
        quiet = _Quiet(not self._verbose)
        with quiet:
            gen = self._tts.infer_generator(
                spk_audio_prompt=str(voice), text=text, output_path=None, lang=lang,
                emo_audio_prompt=str(emotion) if emotion else None,
                emo_alpha=emotion_strength, duration_factor=speed,
                stream_return=True, emit_text=True,
                max_text_tokens_per_segment=segment_tokens,
                text_normalization=True, verbose=self._verbose, **sampling,
            )
        while True:
            with quiet:
                item = next(gen, None)
            if item is None:
                break
            if not isinstance(item, tuple):
                break                      # el último yield es el wav completo
            wav, seg_text = item
            if not torch.is_tensor(wav):
                break
            silencio = not seg_text
            if silencio and not include_silence:
                continue
            audio = (wav.reshape(-1).to(torch.float32) / 32767.0).numpy()
            yield Chunk(audio=audio, text=seg_text.strip(), sample_rate=self.SAMPLE_RATE,
                        elapsed=time.perf_counter() - t0, is_silence=silencio)
