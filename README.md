# indextts25-inference

Librería de inferencia para **IndexTTS-2.5**: síntesis zero-shot con control de emoción,
velocidad y streaming con texto sincronizado.

Empaqueta el código de [index-tts](https://github.com/index-tts/index-tts) (commit `ee40fa7`)
con una API de alto nivel y dos correcciones sobre upstream.

## Instalación

```bash
pip install indextts25-inference

# con el motor GPT2 acelerado (recomendado): 1,77x en párrafos
pip install "indextts25-inference[flash]"
```

Los pesos van aparte:

```bash
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
```

## Uso

```python
from indextts25 import IndexTTS25

tts = IndexTTS25(model_dir="checkpoints", attn_backend="flash")
tts.warm("voz.wav")                      # saca la extracción del hablante del camino crítico

tts.synthesize("Hola, ¿cómo va?", voice="voz.wav", out="salida.wav", speed=0.85)
```

### Streaming con texto sincronizado

Cada trozo llega con el texto que lo produjo, que es lo que hace falta para subtítulos
o para mostrar lo que se está diciendo:

```python
for chunk in tts.stream("Un texto largo…", voice="voz.wav"):
    reproducir(chunk.audio)              # np.float32 mono a 22050 Hz
    mostrar(chunk.text)                  # el segmento correspondiente
    print(chunk.elapsed, chunk.duration)
```

`segment_tokens` gobierna latencia y granularidad a la vez. Por debajo de ~40 la partición
empieza a cortar palabras al medio:

| `segment_tokens` | time-to-first-audio | palabras cortadas |
|---|---|---|
| 120 (default de upstream) | 1,68 s | no |
| **60 (default de esta lib)** | **1,06 s** | **no** |
| 40 | 1,07 s | sí |

### Modelos finetuneados

```python
tts = IndexTTS25(model_dir="checkpoints", config="checkpoints/config_mi_voz.yaml")
```

Un YAML con `gpt_checkpoint:` apuntando a tus pesos. Sirve para un LoRA ya fundido:
el modelo resultante tiene exactamente el mismo tamaño y velocidad que el base.

### Emoción y velocidad

```python
tts.synthesize(texto, voice="voz.wav", out="o.wav",
               emotion="referencia_triste.wav", emotion_strength=0.6,
               speed=0.85)              # <1 acelera; por debajo de 0,78 satura
```

## Rendimiento

RTX 5090, texto de 824 caracteres, mediana de 3 corridas:

| configuración | TTFA | párrafo | VRAM |
|---|---|---|---|
| `attn_backend="sdpa"` | 1,06 s | 13,85 s | 8,0 GB |
| **`attn_backend="flash"`** | **0,82 s** | **7,84 s** | 9,8 GB |
| `flash` + `fp16=True` | 1,00 s | 10,61 s | 7,2 GB |
| `sdpa` + `fp16=True` | 1,30 s | 16,13 s | **6,2 GB** |

**`fp16` no acelera**: con batch 1 el cuello es ancho de banda y lanzamiento de kernels, y
los casts del autocast no se amortizan. Sirve sólo para ahorrar VRAM.

## Diferencias con upstream

1. **`nemo-text-processing` es dependencia real.** Upstream no la declara aunque
   `utils/nemo_tn.py` la necesita para español y japonés; sin ella la normalización falla
   **en silencio** y los números van crudos al tokenizer.
2. **Orden de normalización corregido.** Upstream aplica `char_rep_map` (que mapea `":"`
   a `","`) *antes* de normalizar, así que `14:30` se lee "catorce **coma** treinta".
   Acá la sustitución va después: "catorce **y media**".
3. `attn_backend` por nombre en vez de `use_accel` booleano.
4. Streaming que emite `(audio, texto)` en vez de sólo audio.
5. Sin ruido en stdout/stderr: los `print()` de upstream quedan tras `verbose`.
6. Empaquetado sin el código de 2.0 ni MaskGCT: 2,2 MB en vez de 29 MB.

## Licencia

Bilibili Model Use License Agreement — ver `LICENSE`. Esta librería es un trabajo derivado;
las modificaciones no están avaladas ni garantizadas por los autores originales.
