"""SiliconFlow cloud TTS provider.

Two-step voice cloning:
  1. POST /v1/uploads/audio/voice (base64) → uri (cached)
  2. POST /v1/audio/speech with voice=uri → audio bytes
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .base import TTSProvider, TTSResult


class SiliconFlowProvider(TTSProvider):
    name = "siliconflow"

    BASE_URL = "https://api.siliconflow.cn/v1"

    def __init__(
        self,
        cfg: dict[str, Any],
        env: dict[str, str] | None = None,
        run_id: str = "",
        repo_root: Path | None = None,
    ) -> None:
        super().__init__(cfg, env)
        self.sub = cfg.get("siliconflow", {})
        env_map = env or {}
        self.api_key = env_map.get(
            self.sub.get("api_key_env", "SILICONFLOW_API_KEY"), ""
        )
        if not self.api_key:
            raise RuntimeError(
                "SiliconFlowProvider: missing SILICONFLOW_API_KEY env var"
            )
        self.model = self.sub.get(
            "model", "FunAudioLLM/CosyVoice2-0.5B"
        )
        self.speed = self.sub.get("speed", 1.0)
        self.gain = self.sub.get("gain", 0.0)
        self.response_format = self.sub.get("response_format", "wav")
        self._init_registry(repo_root, run_id)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # -------- registry cache -------------------------------------------

    def _init_registry(self, repo_root: Path | None, run_id: str) -> None:
        repo = repo_root or Path(__file__).resolve().parents[2]
        self.registry_path = (
            repo / "runs" / run_id / "02_tts" / "voice_registry_sf.json"
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

    # -------- step 1: upload voice -------------------------------------

    def _upload_voice(self, ref_audio_path: Path, ref_text: str, name: str) -> str:
        """POST /v1/uploads/audio/voice → uri."""
        with open(ref_audio_path, "rb") as f:
            ref_bytes = f.read()
        ext = ref_audio_path.suffix.lower().lstrip(".")
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(
            ext, "audio/wav"
        )
        audio_data_uri = f"data:{mime};base64,{base64.b64encode(ref_bytes).decode()}"

        url = f"{self.BASE_URL}/uploads/audio/voice"
        payload = {
            "model": self.model,
            "customName": name,
            "audio": audio_data_uri,
            "text": ref_text or "Hello, this is a reference voice.",
        }
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(
                    url, json=payload, headers=self._headers(), timeout=60
                )
                if r.status_code == 200:
                    body = r.json()
                    uri = body.get("uri", "")
                    if not uri:
                        raise RuntimeError(f"No uri in response: {body}")
                    return uri
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"SiliconFlow voice upload failed: {last_err}")

    # -------- step 2: synthesize ---------------------------------------

    def _synthesize(self, text: str, voice_uri: str) -> tuple[np.ndarray, int]:
        """POST /v1/audio/speech → (audio_np, sample_rate)."""
        url = f"{self.BASE_URL}/audio/speech"
        payload = {
            "model": self.model,
            "input": text,
            "voice": voice_uri,
            "response_format": self.response_format,
            "speed": self.speed,
            "gain": self.gain,
        }
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(
                    url, json=payload, headers=self._headers(), timeout=120
                )
                if r.status_code == 200:
                    return self._decode_audio(r.content)
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"SiliconFlow TTS failed: {last_err}")

    @staticmethod
    def _decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    # -------- public API -----------------------------------------------

    def get_uri(self, sample_id: int, ref_audio_path: Path, ref_text: str) -> str:
        sha = self._audio_sha(ref_audio_path)
        cached = self._registry.get(str(sample_id))
        if cached and cached.get("audio_sha") == sha:
            return cached["uri"]
        name = f"ttsexp_sf_{sample_id}"
        uri = self._upload_voice(ref_audio_path, ref_text, name)
        self._registry[str(sample_id)] = {
            "audio_sha": sha,
            "uri": uri,
            "ts": int(time.time()),
        }
        self._save_registry()
        return uri

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
        uri = self.get_uri(sample_id, ref_audio_path, ref_text)
        audio, sr = self._synthesize(text, uri)
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_s=round(len(audio) / sr, 3),
            backend_meta={
                "backend": self.name,
                "model": self.model,
                "uri": uri,
            },
        )
