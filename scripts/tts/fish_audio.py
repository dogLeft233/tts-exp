"""Fish Audio S2 Pro provider (cloud API).

Two-step protocol:
  1. POST /model (multipart/form-data) → voice model_id (persistent, fast train)
  2. POST /v1/tts (JSON) with reference_id → audio bytes (wav, 24kHz)

Model IDs are cached per (sample_id, audio_sha256) in voice_registry.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .base import TTSProvider, TTSResult


class FishAudioProvider(TTSProvider):
    name = "fish_audio"

    BASE_URL = "https://api.fish.audio"
    DEFAULT_MODEL = "s2-pro"  # header model for /v1/tts

    def __init__(
        self,
        cfg: dict[str, Any],
        env: dict[str, str] | None = None,
        run_id: str = "",
        repo_root: Path | None = None,
    ) -> None:
        super().__init__(cfg, env)
        self.sub = cfg.get("fish_audio", {})
        env_map = env or {}
        self.api_key = env_map.get(
            self.sub.get("api_key_env", "FISH_AUDIO_API_KEY"), ""
        )
        if not self.api_key:
            raise RuntimeError(
                "FishAudioProvider: missing FISH_AUDIO_API_KEY env var"
            )
        self.visibility = self.sub.get("visibility", "private")
        self.voice_prefix = self.sub.get("voice_prefix", "ttsexp_fish")
        self.tts_model = self.sub.get("tts_model", self.DEFAULT_MODEL)
        self.format = self.sub.get("format", "wav")
        self.sample_rate = self.sub.get("sample_rate", 24000)
        self.latency = self.sub.get("latency", "normal")
        self.normalize_loudness = self.sub.get(
            "normalize_loudness", False
        )  # off for strict experiment
        self._init_registry(repo_root, run_id)

    def _headers(self, content_type: str | None = None) -> dict:
        h = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            h["Content-Type"] = content_type
        return h

    # -------- registry cache -------------------------------------------

    def _init_registry(self, repo_root: Path | None, run_id: str) -> None:
        repo = repo_root or Path(__file__).resolve().parents[2]
        self.registry_path = (
            repo / "runs" / run_id / "02_tts" / "voice_registry_fish.json"
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, dict] = {}
        if self.registry_path.exists():
            try:
                self._registry = json.loads(self.registry_path.read_text())
            except json.JSONDecodeError:
                self._registry = {}

    def _save_registry(self) -> None:
        self.registry_path.write_text(json.dumps(self._registry, indent=2))

    @staticmethod
    def _audio_sha(ref_audio_path: Path) -> str:
        h = hashlib.sha256()
        with open(ref_audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    # -------- step 1: voice model creation -----------------------------

    def _create_model(self, ref_audio_path: Path, title: str) -> str:
        """POST /model with multipart/form-data → model_id."""
        url = f"{self.BASE_URL}/model"
        with open(ref_audio_path, "rb") as f:
            audio_bytes = f.read()
        ext = ref_audio_path.suffix.lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".opus": "audio/ogg",
        }
        mime = mime_map.get(ext, "audio/wav")

        files = {
            "voices": (ref_audio_path.name, audio_bytes, mime),
        }
        data = {
            "type": "tts",
            "title": title,
            "train_mode": "fast",
            "visibility": self.visibility,
            "enhance_audio_quality": "true",
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
                if r.status_code in (200, 201):
                    body = r.json()
                    return body["_id"]
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Fish model create failed: {last_err}")

    # -------- step 2: synthesize ---------------------------------------

    def _synthesize(self, text: str, model_id: str) -> tuple[np.ndarray, int]:
        """POST /v1/tts → (audio_np, sample_rate)."""
        url = f"{self.BASE_URL}/v1/tts"
        payload = {
            "text": text,
            "reference_id": model_id,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "latency": self.latency,
            "normalize": False,
            "prosody": {
                "speed": 1.0,
                "volume": 0,
                "normalize_loudness": self.normalize_loudness,
            },
            "chunk_length": 300,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.2,
        }
        headers = self._headers("application/json")
        headers["model"] = self.tts_model

        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(
                    url, json=payload, headers=headers, timeout=120
                )
                if r.status_code == 200:
                    return self._decode_audio(r.content)
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Fish TTS synthesize failed: {last_err}")

    @staticmethod
    def _decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    # -------- public API -----------------------------------------------

    def get_model_id(self, sample_id: int, ref_audio_path: Path) -> str:
        sha = self._audio_sha(ref_audio_path)
        cached = self._registry.get(str(sample_id))
        if cached and cached.get("audio_sha") == sha:
            return cached["model_id"]
        title = f"{self.voice_prefix}_{sample_id}"
        model_id = self._create_model(ref_audio_path, title)
        self._registry[str(sample_id)] = {
            "audio_sha": sha,
            "model_id": model_id,
            "ts": int(time.time()),
        }
        self._save_registry()
        return model_id

    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,
        language: str = "Chinese",
        sample_id: int | None = None,
    ) -> TTSResult:
        if sample_id is None:
            sample_id = abs(hash(str(ref_audio_path))) % (10**8)
        model_id = self.get_model_id(sample_id, ref_audio_path)
        audio, sr = self._synthesize(text, model_id)
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_s=round(len(audio) / sr, 3),
            backend_meta={
                "backend": self.name,
                "tts_model": self.tts_model,
                "model_id": model_id,
                "format": self.format,
            },
        )
