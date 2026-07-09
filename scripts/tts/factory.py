"""Provider factory — selects and instantiates a TTS backend from config.

Config contract (scripts/config.yaml):

    tts:
      provider: "dashscope_vc"     # one of: faster_qwen3, dashscope_vc

      faster_qwen3:
        model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        append_silence: true
        x_vector_only: false

      dashscope_vc:
        api_key_env: "DASHSCOPE_API_KEY"
        target_model: "qwen3-tts-vc-2026-01-22"
        voice_prefix: "ttsexp"
        voice_cache_path: "runs/<run_id>/02_tts/voice_registry.json"
"""

from __future__ import annotations

from pathlib import Path

from .base import TTSProvider
from .dashscope_vc import DashScopeQwen3VCProvider
from .f5_tts import F5TTSProvider
from .faster_qwen3 import FasterQwen3TTSProvider
from .fish_audio import FishAudioProvider
from .higgs_tts import HiggsTTSProvider
from .siliconflow import SiliconFlowProvider


_REGISTRY: dict[str, type[TTSProvider]] = {
    "faster_qwen3": FasterQwen3TTSProvider,
    "dashscope_vc": DashScopeQwen3VCProvider,
    "fish_audio": FishAudioProvider,
    "f5_tts": F5TTSProvider,
    "higgs_tts": HiggsTTSProvider,
    "siliconflow": SiliconFlowProvider,
}


def get_tts_provider(
    tts_cfg: dict,
    run_id: str | None = None,
    repo_root: Path | None = None,
    env: dict[str, str] | None = None,
) -> TTSProvider:
    """Build provider instance based on tts.provider config entry.

    Args:
        tts_cfg: top-level cfg["tts"] dict.
        run_id: current run_id (used for voice cache path resolution).
        repo_root: repo root (defaults to 3 levels up from this file).
        env: environment variables (primarily for DASHSCOPE_API_KEY).
    """
    repo = repo_root or Path(__file__).resolve().parents[2]
    name = tts_cfg.get("provider", "faster_qwen3")
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown TTS provider '{name}'. Known: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](
        cfg=tts_cfg,
        env=env or {},
        run_id=run_id or "",
        repo_root=repo,
    )