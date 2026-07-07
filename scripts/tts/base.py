"""Abstract TTS provider interface.

All backends return raw audio as a numpy array plus its sample rate.
Downstream code (02_tts.py) handles disk I/O and metadata, so providers
stay testable in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TTSResult:
    """Output of a single TTS generation."""

    audio: np.ndarray         # (N,) float32 or int16, mono
    sample_rate: int          # Hz
    duration_s: float        # len(audio)/sample_rate
    backend_meta: dict       # provider-specific extras (voice_id, model, ...)


class TTSProvider(ABC):
    """Abstract base for pluggable TTS backends.

    Concrete classes must:
      • Set `name` (short id, used in logs and meta)
      • Implement `generate_voice_clone`
      • Accept config via __init__(cfg: dict, env: dict[str, str])
    """

    name: str

    def __init__(self, cfg: dict[str, Any], env: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.env = env or {}

    @abstractmethod
    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,
        language: str = "Chinese",
    ) -> TTSResult:
        """Synthesize `text` in the voice cloned from `ref_audio_path`.

        Args:
            text: target transcript (what to synthesize).
            ref_audio_path: source speaker audio (readable local file).
            ref_text: transcript of `ref_audio_path`. May equal `text` for
                self-clone (ICL backends), or be ignored (VC backends that only
                need audio).
            language: target BCP-47-like language token ("Chinese", "English", ...).

        Returns:
            TTSResult with mono audio + sample_rate.

        Notes:
            * Deterministic: callers may set torch/np seeds before calling;
              backends must NOT re-seed globally (could break callers).
            * Side effects (e.g. DashScope voice registration) should cache,
              see dashscope_vc.py.
        """
        ...