"""IndexTTS-2.5: zero-shot speech synthesis with emotion and speed control."""
from .api import STREAM_SEGMENT_TOKENS, Chunk, IndexTTS25

__all__ = ["IndexTTS25", "Chunk", "STREAM_SEGMENT_TOKENS"]
__version__ = "0.1.1"  # x-release-please-version
