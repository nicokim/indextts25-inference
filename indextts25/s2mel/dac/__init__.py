"""Subconjunto de `descript-audio-codec` usado por el s2mel.

De todo dac acá sólo hace falta `dac.nn.quantize.VectorQuantize` (lo importa
`s2mel/modules/length_regulator.py`). El `__init__` original importaba `audiotools`
para registrar rutas de serialización de sus propios modelos, y `model`/`utils`, que
arrastran el codificador completo, la CLI y audiotools. Nada de eso interviene en
inferencia, así que se dejan afuera.
"""
__version__ = "1.0.0"
__model_version__ = "latest"

from . import nn  # noqa: F401
