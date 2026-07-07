"""TTS provider abstraction (issue #3 v2).

Pluggable backends for self-voice-clone TTS, selected via config.yaml:

    tts:
      provider: "dashscope_vc"   # or "faster_qwen3"

Adding a new TTS backend:
  1. Subclass `TTSProvider` in this package.
  2. Implement `generate_voice_clone()` returning (np.ndarray, sample_rate).
  3. Register it in `get_tts_provider()` below with its config key.
  4. Add matching section to scripts/config.yaml.
"""

from .base import TTSProvider, TTSResult
from .factory import get_tts_provider

__all__ = ["TTSProvider", "TTSResult", "get_tts_provider"]