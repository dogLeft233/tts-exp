"""DashScope cloud Qwen3-TTS-VC provider.

Two-step protocol (per-sample voice cloning):
  1. POST /services/audio/tts/customization with `action: "create"` and a
     base64-encoded local wav → returns `voice_id` bound to `target_model`.
  2. POST /services/aigc/multimodal-generation/generation with model=
     `target_model`, `voice=voice_id`, `text=...` → returns audio URL (wav,
     valid 24h); download and decode to numpy.

Voice IDs are cached per (sample_id, audio_sha256) in `voice_registry.json`
so re-runs of the same run_id don't re-register (saves API quota).

API reference:
  https://help.aliyun.com/zh/model-studio/voice-cloning-user-guide
  https://help.aliyun.com/zh/model-studio/non-realtime-tts-user-guide
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


class DashScopeQwen3VCProvider(TTSProvider):
    name = "dashscope_vc"

    def __init__(
        self,
        cfg: dict[str, Any],
        env: dict[str, str] | None = None,
        run_id: str = "",
        repo_root: Path | None = None,
    ) -> None:
        super().__init__(cfg, env)
        self.sub = cfg.get("dashscope_vc", {})
        self.target_model = self.sub.get(
            "target_model", "qwen3-tts-vc-2026-01-22"
        )
        self.voice_prefix = self.sub.get("voice_prefix", "ttsexp")
        env_map = env or {}
        self.api_key = env_map.get(
            self.sub.get("api_key_env", "DASHSCOPE_API_KEY"), ""
        )
        if not self.api_key:
            raise RuntimeError(
                "DashScopeQwen3VCProvider: missing DASHSCOPE_API_KEY env var"
            )
        # Default DashScope Beijing endpoint; user can override per workspace
        self.base_url = self.sub.get(
            "base_url", "https://dashscope.aliyuncs.com/api/v1"
        ).rstrip("/")
        self._init_registry(repo_root, run_id)

    # -------- voice registry (cache) -----------------------------------

    def _init_registry(self, repo_root: Path | None, run_id: str) -> None:
        repo = repo_root or Path(__file__).resolve().parents[2]
        self.registry_path = (
            repo / "runs" / run_id / "02_tts" / "voice_registry.json"
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

    def _audio_sha(self, ref_audio_path: Path) -> str:
        h = hashlib.sha256()
        with open(ref_audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    # -------- step 1: voice registration -------------------------------

    def _register_voice(
        self, ref_audio_path: Path, voice_name: str
    ) -> str:
        """POST /services/audio/tts/customization → voice_id."""
        base64_str = base64.b64encode(ref_audio_path.read_bytes()).decode()
        # Probe mime from extension (DashScope accepts the data URI scheme).
        ext = ref_audio_path.suffix.lower().lstrip(".")
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(
            ext, "audio/wav"
        )
        data_uri = f"data:{mime};base64,{base64_str}"

        url = f"{self.base_url}/services/audio/tts/customization"
        payload = {
            "model": "qwen-voice-enrollment",  # must not change
            "input": {
                "action": "create",
                "target_model": self.target_model,
                "preferred_name": voice_name,
                "audio": {"data": data_uri},
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=120)
                if r.status_code == 200:
                    return r.json()["output"]["voice"]
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"voice register failed: {last_err}")

    # -------- step 2: synthesis ----------------------------------------

    def _synthesize(self, text: str, voice_id: str, language: str) -> tuple[np.ndarray, int]:
        """POST /services/aigc/multimodal-generation/generation → (audio_np, sr)."""
        url = f"{self.base_url}/services/aigc/multimodal-generation/generation"
        payload = {
            "model": self.target_model,
            "input": {
                "text": text,
                "voice": voice_id,
                "language_type": language,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=120)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(1.5 * (attempt + 1))
                    continue
                body = r.json()
                audio_url = body["output"]["audio"]["url"]
                return self._download_audio(audio_url)
            except requests.RequestException as e:
                last_err = str(e)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"synthesize failed: {last_err}")

    def _download_audio(self, url: str) -> tuple[np.ndarray, int]:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    # -------- public API -----------------------------------------------

    def get_voice_id(self, sample_id: int, ref_audio_path: Path) -> str:
        """Get or register a voice_id for a sample, caching by audio hash."""
        sha = self._audio_sha(ref_audio_path)
        cached = self._registry.get(str(sample_id))
        if cached and cached.get("audio_sha") == sha:
            return cached["voice_id"]
        voice_name = f"{self.voice_prefix}_{sample_id}"
        voice_id = self._register_voice(ref_audio_path, voice_name)
        self._registry[str(sample_id)] = {
            "audio_sha": sha,
            "voice_id": voice_id,
            "ts": int(time.time()),
        }
        self._save_registry()
        return voice_id

    def generate_voice_clone(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,  # ref_text unused by VC; kept for interface parity
        language: str = "Chinese",
        sample_id: int | None = None,
    ) -> TTSResult:
        if sample_id is None:
            sample_id = abs(hash(str(ref_audio_path))) % (10**8)
        voice_id = self.get_voice_id(sample_id, ref_audio_path)
        audio, sr = self._synthesize(text, voice_id, language)
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_s=round(len(audio) / sr, 3),
            backend_meta={
                "backend": self.name,
                "model": self.target_model,
                "voice_id": voice_id,
                "language": language,
            },
        )