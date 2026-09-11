"""IndexTTS-2.5: síntesis de voz zero-shot con control de emoción y velocidad."""
from .api import IndexTTS25, Chunk, STREAM_SEGMENT_TOKENS

__all__ = ["IndexTTS25", "Chunk", "STREAM_SEGMENT_TOKENS"]
__version__ = "0.1.0"
