from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ASR_MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
ASR_MODEL_REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
ASR_SNAPSHOT_HASH = "4fb78336778731da7ffa74f6438d7a5c8ef81366f367569cbe9cbc74cb9217de"
ASR_TOKENIZER_VOCABULARY_HASH = "267dc46d0a7d6dd6a91a89e5b747f771e99643693fc7438568294a3674c7f28c"
SYNCNET_CHECKPOINT_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SEED = 20260901
PARENT_ASR_RUN = "runs/lrs3_asr_sync_error_correlation_20260831"
PARENT_ASR_MANIFEST = f"{PARENT_ASR_RUN}/01_manifest/manifest.json"
PARENT_ASR_CONFIG = f"{PARENT_ASR_RUN}/00_preflight/config.json"
PARENT_ASR_MODEL_LOCK = f"{PARENT_ASR_RUN}/00_preflight/asr_model_lock.json"
PARENT_ASR_ASSETS = f"{PARENT_ASR_RUN}/00_preflight/assets.json"
PARENT_ASR_MANIFEST_DECISION = f"{PARENT_ASR_RUN}/01_manifest/decision.json"
PARENT_ASR_DECISION = f"{PARENT_ASR_RUN}/03_asr/decision.json"
PARENT_ASR_RECORDS = f"{PARENT_ASR_RUN}/03_asr/records"
PARENT_MANIFEST = "runs/lrs3_mfa_linear_replacement_20260824/00_protocol_lock_retry1/manifest.json"
PARENT_TEST_LOCK = "runs/lrs3_mfa_linear_replacement_20260824/00_protocol_lock_retry1/test_lock.json"
CONDITIONS = ("natural", "asr_targeted", "target_control_0", "target_control_1")
ALLOWED_STAGES = ("lock", "candidates", "audio", "render", "replacement", "sync", "analysis", "all")
ALLOWED_DEVICES = ("auto", "cuda", "cpu")


@dataclass(frozen=True)
class RuntimeOptions:
    repo_root: Path
    run_dir: Path
    stage: str
    device: str
    resume: bool


def frozen_config(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    return {
        "schema_version": 1,
        "experiment": "lrs3_asr_targeted_local_replacement_prototype",
        "seed": SEED,
        "parent": {
            "asr_manifest": PARENT_ASR_MANIFEST,
            "asr_config": PARENT_ASR_CONFIG,
            "asr_model_lock": PARENT_ASR_MODEL_LOCK,
            "asr_assets": PARENT_ASR_ASSETS,
            "asr_manifest_decision": PARENT_ASR_MANIFEST_DECISION,
            "asr_decision": PARENT_ASR_DECISION,
            "asr_records": PARENT_ASR_RECORDS,
            "cohort_manifest": PARENT_MANIFEST,
            "test_lock": PARENT_TEST_LOCK,
            "expected_samples": 24,
            "expected_source_groups": 24,
            "expected_asr_cells": 48,
            "split": "fit",
        },
        "model": {
            "asr_model_id": ASR_MODEL_ID,
            "asr_revision": ASR_MODEL_REVISION,
            "asr_snapshot_hash": ASR_SNAPSHOT_HASH,
            "asr_tokenizer_vocabulary_hash": ASR_TOKENIZER_VOCABULARY_HASH,
            "syncnet_checkpoint_sha256": SYNCNET_CHECKPOINT_SHA256,
        },
        "patch": {
            "sample_rate_hz": 16000,
            "channels": 1,
            "resampler": "scipy.signal.resample",
            "scipy_version": "1.18.0",
            "boundary_ramp": "raised_cosine_in_segment",
            "max_ramp_samples": 320,
            "length_policy": "exact_destination_sample_count",
            "loudness_normalization": False,
            "global_time_transform": False,
        },
        "render": {
            "conditions": list(CONDITIONS),
            "face_det_batch_size": 4,
            "wav2lip_batch_size": 4,
            "nosmooth": True,
            "video_codec": "copy_for_replacement_mux",
        },
        "syncnet": {
            "vshift": 15,
            "min_track": 50,
            "fps": 25.0,
            "median_filter_width": 9,
            "local_guard_s": 0.20,
        },
        "analysis": {
            "bootstrap_draws": 10000,
            "bootstrap_seed": SEED,
            "confidence_interval": 0.95,
            "min_samples": 12,
            "min_source_groups": 12,
        },
        "paths": {
            "repo_root": str(root),
            "syncnet_dir": str(root / "third_party/syncnet_python"),
            "syncnet_python": str(Path.home() / ".venvs/syncnet/bin/python"),
            "wav2lip_dir": str(root / "third_party/Wav2Lip"),
            "wav2lip_python": str(Path.home() / ".venvs/wav2lip/bin/python"),
            "wav2lip_checkpoint": str(root / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"),
            "syncnet_checkpoint": str(root / "third_party/syncnet_python/data/syncnet_v2.model"),
        },
    }


def validate_frozen_config(config: Mapping[str, Any], repo_root: Path) -> None:
    expected = frozen_config(repo_root)
    if config != expected:
        raise ValueError("prototype config differs from frozen configuration")


def parse_args(argv: list[str] | None = None) -> RuntimeOptions:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=ALLOWED_STAGES, default="all")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/lrs3_asr_targeted_local_replacement_prototype_20260901"))
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", choices=ALLOWED_DEVICES, default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    repo_root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    return RuntimeOptions(repo_root, run_dir.resolve(), str(args.stage), str(args.device), bool(args.resume))


def config_json(repo_root: Path) -> str:
    return json.dumps(frozen_config(repo_root), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
