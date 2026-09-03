"""Frozen configuration and operator-facing validation for the experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ASR_MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
SYNCNET_CHECKPOINT_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
DEFAULT_SEED = 20260831
DEFAULT_RUN_NAME = "lrs3_asr_sync_error_correlation_20260831"
PARENT_COHORT_MANIFEST = "runs/lrs3_mfa_linear_replacement_20260824/00_protocol_lock_retry1/manifest.json"
CANONICAL_SOURCE_MANIFEST = "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json"
CANONICAL_TTS_METADATA = "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json"
SYNCNET_ENV_PYTHON = Path.home() / ".venvs" / "syncnet" / "bin" / "python"
SYNCNET_DIR = Path("third_party/syncnet_python")

ALLOWED_STAGES = ("preflight", "prepare", "mux", "asr", "sync", "align", "analyze", "plot")
ALLOWED_DEVICES = ("auto", "cuda", "cpu")


@dataclass(frozen=True)
class RuntimeOptions:
    repo_root: Path
    run_dir: Path
    stage: str | None
    device: str
    acquire_asr_model: bool
    resume: bool


def frozen_config(repo_root: Path) -> dict[str, Any]:
    """Return the only scientific configuration accepted by this experiment."""
    return {
        "schema_version": 1,
        "seed": DEFAULT_SEED,
        "arms": ["natural", "tts"],
        "cohort": {
            "parent_manifest": PARENT_COHORT_MANIFEST,
            "source_manifest": CANONICAL_SOURCE_MANIFEST,
            "tts_metadata": CANONICAL_TTS_METADATA,
            "expected_samples": 24,
            "expected_source_groups": 24,
            "split": "fit",
            "cohort_label": "fresh_confirmation",
        },
        "asr": {
            "model_id": ASR_MODEL_ID,
            "decoder": "ctc_argmax_greedy",
            "sample_rate_hz": 16000,
            "batch_size": 1,
            "dtype": "float32",
            "blank_policy": "tokenizer_blank_id",
            "delimiter_policy": "tokenizer_word_delimiter",
        },
        "syncnet": {
            "fps": 25,
            "vshift": 15,
            "min_track": 50,
            "embedding_video_frames": 5,
            "embedding_audio_mfcc_frames": 20,
            "median_filter_width": 9,
            "checkpoint_sha256": SYNCNET_CHECKPOINT_SHA256,
        },
        "analysis": {
            "grid_hz": 25,
            "low_score_k": 1.0,
            "low_score_comparison": "strict_less_than",
            "std_ddof": 0,
            "bootstrap_draws": 10000,
            "bootstrap_seed": DEFAULT_SEED,
            "min_defined_records": 12,
            "min_error_cells": 50,
            "min_source_groups": 8,
            "confidence_interval": 0.95,
        },
        "paths": {
            "asr_cache": str((repo_root / "checkpoints" / "huggingface").resolve()),
            "syncnet_python": str(repo_root / SYNCNET_ENV_PYTHON),
            "syncnet_dir": str((repo_root / SYNCNET_DIR).resolve()),
        },
    }


def _reject_scientific_override(key: str, value: Any, expected: Any) -> None:
    if value != expected:
        raise ValueError(f"scientific override forbidden: {key} must remain {expected!r}")


def validate_frozen_config(config: Mapping[str, Any], repo_root: Path | None = None) -> None:
    expected = frozen_config(repo_root or Path.cwd())
    for section in ("asr", "syncnet", "analysis", "cohort"):
        values = config.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"missing frozen config section: {section}")
        for key, expected_value in expected[section].items():
            if key not in values:
                raise ValueError(f"missing frozen config key: {section}.{key}")
            _reject_scientific_override(f"{section}.{key}", values[key], expected_value)
    if list(config.get("arms", [])) != expected["arms"]:
        raise ValueError("scientific override forbidden: arms")
    if config.get("schema_version") != expected["schema_version"]:
        raise ValueError("unsupported config schema_version")


def validate_decoder_options(values: Mapping[str, Any]) -> None:
    forbidden = {
        "beam_search": False,
        "language_model": None,
        "lexicon": None,
        "spell_correction": False,
        "inverse_text_normalization": False,
        "prompt": None,
        "reference_conditioned": False,
    }
    for key, safe in forbidden.items():
        if key in values and values[key] != safe:
            raise ValueError(f"correction-enabled decoder option forbidden: {key}")
    decoder = values.get("decoder", "ctc_argmax_greedy")
    if decoder != "ctc_argmax_greedy":
        raise ValueError(f"unsupported decoder: {decoder}")


def parse_args(argv: list[str] | None = None) -> RuntimeOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=ALLOWED_STAGES)
    parser.add_argument("--run-dir", type=Path, default=Path("runs") / DEFAULT_RUN_NAME)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", choices=ALLOWED_DEVICES, default="auto")
    parser.add_argument("--acquire-asr-model", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    repo_root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else (repo_root / args.run_dir)
    return RuntimeOptions(repo_root, run_dir.resolve(), args.stage, args.device, args.acquire_asr_model, args.resume)


def config_json(config: Mapping[str, Any]) -> str:
    validate_frozen_config(config)
    return json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
