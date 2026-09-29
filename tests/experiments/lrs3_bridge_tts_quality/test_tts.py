from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from scripts.experiments.lrs3_bridge_tts_quality.common import (
    file_sha256,
    write_self_hashed_json,
)
from scripts.experiments.lrs3_bridge_tts_quality.tts import (
    _cloud_row,
    _copy_existing_local,
    _load_existing_local_rows,
    _read_current_manifest,
)


def _write_pcm16(path: Path, samples: int = 1600) -> None:
    values = np.arange(samples, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(values.tobytes())


def test_cloud_row_flattens_canonical_audio_binding(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.wav"
    provider = tmp_path / "provider.wav"
    _write_pcm16(canonical)
    _write_pcm16(provider)
    canonical_hash = file_sha256(canonical)
    provider_hash = file_sha256(provider)

    record = {
        "sample_id": "sample-1",
        "source_group": "group-1",
        "transcript": "HELLO WORLD",
        "natural_audio": {"path": str(canonical), "sha256": canonical_hash},
        "cloud_tts": {
            "canonical_audio": {"path": str(canonical), "sha256": canonical_hash},
            "provider_audio": {"path": str(provider), "sha256": provider_hash},
            "canonical_samples": 1600,
            "voice_id": "voice-1",
            "metadata_record_sha256": "metadata-hash",
        },
    }

    row = _cloud_row(record)

    assert row["canonical_audio"] == str(canonical)
    assert row["canonical_audio_sha256"] == canonical_hash


def test_current_manifest_indexes_local_provider_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "tts_manifest.json"
    write_self_hashed_json(
        manifest,
        {
            "rows": [
                {
                    "sample_id": "sample-1",
                    "LOCAL": {"canonical_audio": "/tmp/local.wav"},
                    "CLOUD": {"canonical_audio": "/tmp/cloud.wav"},
                }
            ]
        },
    )

    rows = _read_current_manifest(manifest)

    assert rows["sample-1"]["canonical_audio"] == "/tmp/local.wav"


def test_external_manifest_indexes_nested_local_provider_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "tts_manifest.json"
    write_self_hashed_json(
        manifest,
        {
            "rows": [
                {
                    "sample_id": "sample-1",
                    "LOCAL": {"provider": "faster_qwen3", "canonical_audio": "/tmp/local.wav"},
                    "CLOUD": {"provider": "dashscope_vc", "canonical_audio": "/tmp/cloud.wav"},
                }
            ]
        },
    )

    rows = _load_existing_local_rows(manifest)

    assert rows["sample-1"]["provider"] == "faster_qwen3"


def test_copy_existing_local_accepts_string_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    raw = tmp_path / "raw.npy"
    destination = tmp_path / "copied.wav"
    _write_pcm16(source)
    np.save(raw, np.ones(8, dtype=np.float32))
    source_hash = file_sha256(source)
    record = {
        "sample_id": "sample-1",
        "source_group": "group-1",
        "transcript": "HELLO WORLD",
        "natural_audio": {"path": str(source), "sha256": source_hash},
    }
    row = {
        "provider": "faster_qwen3",
        "canonical_audio": str(source),
        "canonical_audio_sha256": source_hash,
        "raw_waveform": str(raw),
        "raw_waveform_sha256": file_sha256(raw),
        "provider_class": "provider",
        "resolved_model_class": "model",
        "provider_implementation_sha256": "implementation",
    }

    copied = _copy_existing_local(row, record, destination)

    assert copied["canonical_audio"] == str(destination.resolve())
    assert destination.is_file()
