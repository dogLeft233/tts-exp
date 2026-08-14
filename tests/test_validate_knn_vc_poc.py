from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from knn_vc_poc_schema import (  # noqa: E402
    EXPECTED_CONDITIONS,
    canonical_sha256,
    file_sha256,
    write_json,
)
from validate_knn_vc_poc import validate_run  # noqa: E402


def test_validator_accepts_complete_synthetic_run(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    feature_dir = tmp_path / "features"
    audio_dir.mkdir()
    feature_dir.mkdir()
    rows = []
    for index, condition in enumerate(EXPECTED_CONDITIONS):
        wav_path = audio_dir / f"{condition}.wav"
        values = 0.01 * np.sin(np.linspace(0, 4 * np.pi, 640, dtype=np.float32))
        sf.write(wav_path, values, 16_000, subtype="FLOAT")
        feature_path = feature_dir / f"{condition}.pt"
        feature_path.write_bytes(b"feature")
        aligned_distance = 0.2 if condition != "paired_tts_wrong_phone" else 0.8
        rows.append({
            "condition": condition,
            "paired_oracle": True,
            "candidate_scope": "synthetic",
            "retrieval": {"coverage": 1.0, "fallback_frames": 0},
            "conditioning_shape": [2, 1024],
            "conditioning_sha256": file_sha256(feature_path),
            "feature_path": str(feature_path),
            "length_adjustment": {
                "raw_decoder_sample_count": 640,
                "target_natural_sample_count": 640,
                "action": "none",
                "adjustment_sample_count": 0,
            },
            "audio": {
                "sample_count": 640, "natural_sample_count": 640,
                "exact_natural_sample_count": True, "finite": True,
                "peak": float(np.abs(values).max()), "rms": float(np.sqrt(np.mean(values ** 2))),
                "clipped_sample_count": 0,
            },
            "output_path": str(wav_path),
            "output_sha256": file_sha256(wav_path),
            "wavlm_synthesis_1024": {
                "output_to_conditioning": {"cosine": 0.1, "mse": 0.1, "compared_frames": 2},
                "output_to_natural_query": {"cosine": 0.1, "mse": 0.1, "compared_frames": 2},
                "output_to_aligned_tts": {"cosine": aligned_distance, "mse": 0.1, "compared_frames": 2},
                "natural_query_to_aligned_tts": {"cosine": 0.5, "mse": 0.1, "compared_frames": 2},
            },
        })
    keys = ["pair-a"]
    summary = {
        "schema_version": 1, "experiment_type": "wavlm_knn_vc_valid_poc", "status": "completed",
        "manifest": {},
        "selection": {
            "split": "valid", "speaker_group": "S0765", "ordered_paired_keys": keys,
            "ordered_paired_keys_sha256": canonical_sha256(keys),
        },
        "model": {}, "policies": {}, "runtime": {},
        "conditions": list(EXPECTED_CONDITIONS),
        "items": [{"paired_key": "pair-a", "conditions": rows}],
        "heldout_excluded": True,
    }
    summary_path = tmp_path / "summary.json"
    write_json(summary_path, summary)
    report = validate_run(summary_path, tmp_path / "gate.json")
    assert report["gate_pass"]
    assert report["negative_control_pass"]
