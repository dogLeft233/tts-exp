"""Local faster-qwen3-tts ICL provider (former behavior preserved).

Wraps `FasterQwen3TTS.from_pretrained(model_id)` and calls
`generate_voice_clone(text=, ref_audio=, ref_text=, language=)`.
Model is loaded lazily on first call so module import is cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import TTSProvider, TTSResult


class FasterQwen3TTSProvider(TTSProvider):
    name = "faster_qwen3"

    def __init__(
        self,
        cfg: dict[str, Any],
        env: dict[str, str] | None = None,
        run_id: str = "",
        repo_root: Path | None = None,
    ) -> None:
        super().__init__(cfg, env)
        self.sub = cfg.get("faster_qwen3", {})
        self.model_id = self.sub.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from faster_qwen3_tts import FasterQwen3TTS
                self._model = FasterQwen3TTS.from_pretrained(self.model_id)
            except (ImportError, ValueError, RuntimeError):
                from qwen_tts import Qwen3TTSModel
                self._model = Qwen3TTSModel.from_pretrained(self.model_id)
        return self._model

    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,
        language: str = "Chinese",
    ) -> TTSResult:
        model = self._ensure_model()
        generation_kwargs = {}
        if self.sub.get("max_new_tokens") is not None:
            generation_kwargs["max_new_tokens"] = int(self.sub["max_new_tokens"])
        audio_list, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
            **generation_kwargs,
        )
        audio = audio_list[0] if isinstance(audio_list, list) else audio_list
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:  # force mono
            audio_np = audio_np.mean(axis=tuple(range(1, audio_np.ndim)))
        return TTSResult(
            audio=audio_np,
            sample_rate=int(sr),
            duration_s=round(len(audio_np) / int(sr), 3),
            backend_meta={
                "backend": self.name,
                "model_id": self.model_id,
                "clone_mode": "icl",
            },
        )
