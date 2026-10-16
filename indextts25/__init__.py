"""IndexTTS-2.5: zero-shot speech synthesis with emotion and speed control."""
from .api import IndexTTS25, Chunk, STREAM_SEGMENT_TOKENS

__all__ = ["IndexTTS25", "Chunk", "STREAM_SEGMENT_TOKENS"]
__version__ = "0.2.0"
