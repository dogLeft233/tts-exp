#!/usr/bin/env python3
"""Tests for the strict two-stage HuBERT pair manifest boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_waveform_hubert_targets import build_targets, file_sha256  # noqa: E402
from prepare_two_stage_hubert_manifest import (  # noqa: E402
    STAGE_CONTRACT,
    STAGE_MANIFEST_TYPE,
    build_two_stage_manifest,
    load_two_stage_manifest,
)


def _tokens() -> list[dict[str, object]]:
    return [
        {"label": "a", "start_s": 0.0, "end_s": 0.5, "confidence": 1.0},
        {"label": "b", "start_s": 0.5, "end_s": 1.0, "confidence": 1.0},
    ]


def _record(
    *,
    paired_key: str,
    condition: str,
    path: Path,
    transcript: str,
    split: str,
    speaker_group: str,
) -> dict[str, object]:
    return {
        "paired_key": paired_key,
        "condition": condition,
        "variant": "raw",
        "tts_provider": "faster_qwen3" if condition == "tts" else None,
        "audio_path": str(path),
        "filepath": str(path),
        "source_sha256": file_sha256(path),
        "transcript": transcript,
        "split": split,
        "speaker_group": speaker_group,
        "alignment_source": "mfa",
        "tokens": _tokens(),
    }


def _build_parent_manifest(
    tmp_path: Path,
    specs: list[tuple[str, str, str, str]],
) -> tuple[Path, Path]:
    sample_rate = 16_000
    records: list[dict[str, object]] = []
    for index, (paired_key, split, speaker_group, transcript) in enumerate(specs):
        natural_path = tmp_path / f"{paired_key}_natural.wav"
        tts_path = tmp_path / f"{paired_key}_tts.wav"
        natural = np.sin(np.linspace(0, 2 * np.pi * (index + 1), sample_rate, dtype=np.float32))
        tts = np.cos(np.linspace(0, 2 * np.pi * (index + 1), sample_rate, dtype=np.float32))
        sf.write(natural_path, natural, sample_rate)
        sf.write(tts_path, tts, sample_rate)
        records.extend(
            [
                _record(
                    paired_key=paired_key,
                    condition="natural",
                    path=natural_path,
                    transcript=transcript,
                    split=split,
                    speaker_group=speaker_group,
                ),
                _record(
                    paired_key=paired_key,
                    condition="tts",
                    path=tts_path,
                    transcript=transcript,
                    split=split,
                    speaker_group=speaker_group,
                ),
            ]
        )
    source_manifest = tmp_path / "source_conditions.json"
    source_manifest.write_text(json.dumps({"records": records}, allow_nan=False), encoding="utf-8")
    target_dir = tmp_path / "fresh_targets"
    target_payload = build_targets(source_manifest, target_dir)
    assert len(target_payload["accepted"]) == len(specs)
    return target_dir / "target_manifest.json", source_manifest


def _default_specs() -> list[tuple[str, str, str, str]]:
    return [
        ("pair_train", "train", "speaker_train", "train transcript"),
        ("pair_valid", "valid", "speaker_valid", "valid transcript"),
        ("pair_heldout", "heldout", "speaker_heldout", "heldout transcript"),
    ]


def test_builds_train_valid_manifest_without_weak_target_paths(tmp_path: Path) -> None:
    parent_manifest, _ = _build_parent_manifest(tmp_path, _default_specs())
    output = tmp_path / "two_stage_manifest.json"

    payload = build_two_stage_manifest(parent_manifest, output)
    records = load_two_stage_manifest(output)

    assert payload["manifest_type"] == STAGE_MANIFEST_TYPE
    assert payload["contract"] == STAGE_CONTRACT
    assert {row["paired_key"] for row in records} == {"pair_train", "pair_valid"}
    assert {row["split"] for row in records} == {"train", "valid"}
    assert len(payload["excluded_provenance"]["heldout"]) == 1
    assert payload["excluded_provenance"]["heldout"][0]["paired_key"] == "pair_heldout"
    for row in records:
        assert "target_path" not in row
        assert "target_sha256" not in row
        assert "weak_target_warning" not in row
        assert row["provenance"]["natural"] == {
            "condition": "natural",
            "variant": "raw",
            "alignment_source": "mfa",
        }
        assert row["provenance"]["tts"]["tts_provider"] == "faster_qwen3"
        assert row["matched_spans"]


def test_builder_rejects_cross_split_transcript_leakage(tmp_path: Path) -> None:
    parent_manifest, _ = _build_parent_manifest(
        tmp_path,
        [
            ("pair_train", "train", "speaker_train", "shared transcript"),
            ("pair_valid", "valid", "speaker_valid", "shared transcript"),
        ],
    )

    with pytest.raises(ValueError, match="cross-split transcript_sha256 leakage"):
        build_two_stage_manifest(parent_manifest, tmp_path / "two_stage_manifest.json")


def test_builder_rejects_non_raw_or_non_mfa_source_provenance(tmp_path: Path) -> None:
    parent_manifest, source_manifest = _build_parent_manifest(tmp_path, _default_specs())
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source["records"][1]["variant"] = "postprocessed"
    source_manifest.write_text(json.dumps(source, allow_nan=False), encoding="utf-8")
    parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
    parent["source_manifest_sha256"] = file_sha256(source_manifest)
    parent_manifest.write_text(json.dumps(parent, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must be raw"):
        build_two_stage_manifest(parent_manifest, tmp_path / "two_stage_manifest.json")


def test_builder_rejects_malformed_overlapping_matched_spans(tmp_path: Path) -> None:
    parent_manifest, _ = _build_parent_manifest(tmp_path, _default_specs())
    parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
    row = parent["accepted"][0]
    metadata_path = Path(row["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metadata"]["spans"][1]["natural_start_s"] = 0.25
    metadata_path.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
    row["metadata"] = metadata["metadata"]
    parent_manifest.write_text(json.dumps(parent, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap or are unordered"):
        build_two_stage_manifest(parent_manifest, tmp_path / "two_stage_manifest.json")


def test_loader_rejects_heldout_record_and_metadata_hash_change(tmp_path: Path) -> None:
    parent_manifest, _ = _build_parent_manifest(tmp_path, _default_specs())
    output = tmp_path / "two_stage_manifest.json"
    build_two_stage_manifest(parent_manifest, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["records"][0]["split"] = "heldout"
    output.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load split"):
        load_two_stage_manifest(output)

    build_two_stage_manifest(parent_manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    metadata_path = Path(payload["records"][0]["matched_span_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metadata"]["matched_count"] += 1
    metadata_path.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        load_two_stage_manifest(output)


def test_builder_rejects_audio_hash_mismatch_and_source_speaker_mismatch(tmp_path: Path) -> None:
    parent_manifest, source_manifest = _build_parent_manifest(tmp_path, _default_specs())
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source["records"][0]["speaker_group"] = "wrong_speaker"
    source_manifest.write_text(json.dumps(source, allow_nan=False), encoding="utf-8")
    parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
    parent["source_manifest_sha256"] = file_sha256(source_manifest)
    parent_manifest.write_text(json.dumps(parent, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="speaker group mismatch"):
        build_two_stage_manifest(parent_manifest, tmp_path / "two_stage_manifest.json")
