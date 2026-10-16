"""Measures what the README reports: load, warm(), time-to-first-audio and throughput.

    python scripts/benchmark.py --model-dir checkpoints --voice voice.wav --backend flash

Prints medians over --reps repetitions and the peak VRAM.
"""
import argparse
import statistics
import tempfile
import time
import wave
from pathlib import Path

import torch

SHORT = "Hola, ¿cómo va? Gracias por pasar por el stream."
PARAGRAPH = (
    "La inteligencia artificial cambió la forma en que trabajamos. Hace apenas unos años, "
    "nadie imaginaba que una computadora pudiera mantener una conversación fluida, escribir "
    "código o componer música. Hoy, el 14 de marzo de 2026 a las 18:45, más de 3.500 personas "
    "siguieron la transmisión en vivo, y el récord anterior era de 2.100 espectadores. "
    "¿Qué viene después? Nadie lo sabe con certeza, pero vale la pena prestar atención."
)


def ttfa(tts, text, voice):
    t0 = time.perf_counter()
    for _ in tts.stream(text, voice=voice):
        return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="checkpoints")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--backend", choices=["sdpa", "flash"], default="sdpa")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    t0 = time.perf_counter()
    from indextts25 import IndexTTS25  # load time includes the import

    tts = IndexTTS25(model_dir=a.model_dir, attn_backend=a.backend, fp16=a.fp16)
    torch.cuda.synchronize()
    load = time.perf_counter() - t0
    t0 = time.perf_counter()
    tts.warm(a.voice)
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0

    short, para, synth, audio = [], [], [], []
    out = Path(tempfile.mkdtemp()) / "bench.wav"
    for r in range(a.reps):
        torch.manual_seed(r)
        short.append(ttfa(tts, SHORT, a.voice))
        para.append(ttfa(tts, PARAGRAPH, a.voice))
        t0 = time.perf_counter()
        tts.synthesize(PARAGRAPH, voice=a.voice, out=out)
        synth.append(time.perf_counter() - t0)
        with wave.open(str(out)) as w:
            audio.append(w.getnframes() / w.getframerate())

    med = statistics.median
    print(f"backend={a.backend} fp16={a.fp16} torch={torch.__version__}")
    print(f"load {load:.1f} s | warm {warm:.2f} s")
    print(f"TTFA short {med(short):.3f} s | TTFA paragraph {med(para):.3f} s")
    print(f"paragraph {med(synth):.2f} s ({len(PARAGRAPH) / med(synth):.0f} chars/s, "
          f"RTF {med(synth) / med(audio):.3f})")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
