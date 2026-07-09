"""Boson Higgs TTS 3 provider (cloud API).

Single-step protocol:
  POST /v1/audio/speech (multipart/form-data with ref_audio upload)
  → audio bytes (wav, 24kHz)

No separate model training step needed — voice cloning is in-line via ref_audio.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .base import TTSProvider, TTSResult


class HiggsTTSProvider(TTSProvider):
    name = "higgs_tts"

    BASE_URL = "https://api.boson.ai"
    DEFAULT_MODEL = "higgs-tts-3"

    def __init__(
        self,
        cfg: dict[str, Any],
        env: dict[str, str] | None = None,
        run_id: str = "",
        repo_root: Path | None = None,
    ) -> None:
        super().__init__(cfg, env)
        self.sub = cfg.get("higgs_tts", {})
        env_map = env or {}
        self.api_key = env_map.get(
            self.sub.get("api_key_env", "BOSON_API_KEY"), ""
        )
        if not self.api_key:
            raise RuntimeError(
                "HiggsTTSProvider: missing BOSON_API_KEY env var"
            )
        self.model = self.sub.get("model", self.DEFAULT_MODEL)
        self.response_format = self.sub.get("response_format", "wav")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _synthesize(self, text: str, ref_audio_path: Path, ref_text: str) -> tuple[np.ndarray, int]:
        url = f"{self.BASE_URL}/v1/audio/speech"
        with open(ref_audio_path, "rb") as f:
            audio_bytes = f.read()
        ext = ref_audio_path.suffix.lower()
        mime_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
        mime = mime_map.get(ext, "audio/wav")

        files = {
            "ref_audio": (ref_audio_path.name, audio_bytes, mime),
        }
        data = {
            "model": self.model,
            "input": text.strip(),
            "voice": "default",
            "response_format": self.response_format,
            "ref_text": ref_text,
        }

        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(
                    url,
                    data=data,
                    files=files,
                    headers=self._headers(),
                    timeout=120,
                )
                if r.status_code == 200:
                    return self._decode_audio(r.content)
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Higgs TTS synthesize failed: {last_err}")

    @staticmethod
    def _decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,
        language: str = "Chinese",
        sample_id: int | None = None,
    ) -> TTSResult:
        audio, sr = self._synthesize(text, ref_audio_path, ref_text)
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_s=round(len(audio) / sr, 3),
            backend_meta={
                "backend": self.name,
                "model": self.model,
                "response_format": self.response_format,
            },
        )
