"""F5-TTS local provider (R2 – non-autoregressive flow-matching).

Wraps `f5_tts.api.F5TTS` for zero-shot voice cloning.
Model is loaded lazily on first call so module import is cheap.

Install: `pip install f5-tts` (code MIT; weights CC-BY-NC).

API reference:
    https://github.com/SWivid/F5-TTS#python-api
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import TTSProvider, TTSResult


class F5TTSProvider(TTSProvider):
    name = "f5_tts"

    def __init__(
        self,
        cfg: dict[str, Any],
        env: dict[str, str] | None = None,
        run_id: str = "",
        repo_root: Path | None = None,
    ) -> None:
        super().__init__(cfg, env)
        self.sub = cfg.get("f5_tts", {})
        self.model_name = self.sub.get("model_name", "F5TTS_Base")
        self.nfe_step = self.sub.get("nfe_step", 32)
        self.cfg_strength = self.sub.get("cfg_strength", 2.0)
        self.device = self.sub.get("device", "cuda")
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from f5_tts.api import F5TTS

            self._model = F5TTS(
                model=self.model_name,
                device=self.device,
            )
        return self._model

    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,
        language: str = "Chinese",
    ) -> TTSResult:
        model = self._ensure_model()
        audio, sr, _ = model.infer(
            ref_file=str(ref_audio_path),
            ref_text=ref_text,
            gen_text=text,
            nfe_step=self.nfe_step,
            cfg_strength=self.cfg_strength,
        )
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr is None:
            sr = 24000  # F5-TTS default
        return TTSResult(
            audio=audio_np,
            sample_rate=int(sr),
            duration_s=round(len(audio_np) / int(sr), 3),
            backend_meta={
                "backend": self.name,
                "model": self.model_name,
                "nfe_step": self.nfe_step,
                "cfg_strength": self.cfg_strength,
            },
        )
