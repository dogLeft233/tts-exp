"""Tests for target caching and provenance auditing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pnp_cache_targets import audit_manifest, cache_phone_local_targets, file_sha256


def _write_pair(tmp_path: Path) -> Path:
    natural_path = tmp_path / "natural.wav"
    tts_path = tmp_path / "tts.wav"
    sr = 1000
    natural = np.zeros(1000, dtype=np.float32)
    natural[100:300] = 0.2
    tts = np.concatenate(
        [np.full(100, 0.7, dtype=np.float32), np.zeros(50, dtype=np.float32), np.full(300, -0.7, dtype=np.float32)]
    )
    sf.write(natural_path, natural, sr)
    sf.write(tts_path, tts, sr)
    records = {
        "schema_version": 1,
        "records": [
            {
                "paired_key": "1",
                "sample_id": 1,
                "condition": "natural",
                "audio_path": str(natural_path),
                "speaker_id": "speaker-a",
                "transcript": "a b",
                "alignment_source": "mfa",
                "alignment_confidence": 1.0,
                "sample_rate": sr,
                "tokens": [
                    {"token": "a", "start_s": 0.1, "end_s": 0.3, "confidence": 1.0},
                    {"token": "b", "start_s": 0.5, "end_s": 0.7, "confidence": 1.0},
                ],
            },
            {
                "paired_key": "1",
                "sample_id": 1,
                "condition": "tts",
                "tts_provider": "provider-a",
                "audio_path": str(tts_path),
                "speaker_id": "speaker-a",
                "transcript": "a b",
                "alignment_source": "mfa",
                "alignment_confidence": 1.0,
                "sample_rate": sr,
                "tokens": [
                    {"token": "a", "start_s": 0.0, "end_s": 0.1, "confidence": 1.0},
                    {"token": "b", "start_s": 0.15, "end_s": 0.45, "confidence": 1.0},
                ],
            },
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    return manifest


def test_audit_reports_provenance_and_coverage(tmp_path: Path) -> None:
    manifest = _write_pair(tmp_path)
    result = audit_manifest(manifest)
    assert result["pair_count"] == 1
    assert result["row_count"] == 1
    row = result["rows"][0]
    assert row["provider"] == "provider-a"
    assert row["matched_token_count"] == 2
    assert row["coverage"] == 1.0
    assert len(row["natural_sha256"]) == 64


def test_audit_uses_warp_matching_for_inserted_token(tmp_path: Path) -> None:
    manifest = _write_pair(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][1]["tokens"].insert(
        1,
        {"token": "extra", "start_s": 0.1, "end_s": 0.14, "confidence": 1.0},
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    row = audit_manifest(manifest)["rows"][0]
    assert row["matched_token_count"] == 2
    assert row["coverage"] == 1.0
    assert row["label_mismatch_count"] == 1


    manifest = _write_pair(tmp_path)
    output_dir = tmp_path / "cache"
    result = cache_phone_local_targets(manifest, output_dir)
    assert result["cached_count"] == 1
    item = result["items"][0]
    audio, sr = sf.read(item["audio_path"], dtype="float32")
    assert sr == 16000
    assert len(audio) == 16000
    metadata = json.loads(Path(item["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["target_type"] == "tts_phone_local_warp"
    assert metadata["exact_sample_count"] is True
    assert metadata["output_sha256"] == file_sha256(item["audio_path"])
    assert (output_dir / "manifest.json").exists()
