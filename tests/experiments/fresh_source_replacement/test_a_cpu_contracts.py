from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from scripts.experiments.fresh_source_replacement.common import (
    branch_dir,
    cell_key,
    file_sha256,
    write_self_hashed,
)
from scripts.experiments.fresh_source_replacement.scoring import (
    fixed_metrics,
    make_bootstrap_indices,
    synthetic_delay_control,
)
from scripts.experiments.fresh_source_replacement.validate import (
    _critical_analysis,
    validate,
)


def test_fixed_support_tie_and_five_frame_shift() -> None:
    rng = np.random.default_rng(19)
    audio = rng.normal(size=(180, 8)).astype(np.float32)
    visual = audio.copy()
    natural = fixed_metrics(visual, audio)
    shifted = synthetic_delay_control(visual, audio)

    assert natural["best_lag"] == 0
    assert natural["offset"] == 0
    assert shifted["best_index_shift"] == 5
    assert shifted["offset_change"] == -5
    assert shifted["uncompensated_damage"] > 0

    # np.argmin's first-column tie is part of the frozen endpoint.
    flat = np.zeros((31,), dtype=np.float64)
    from scripts.experiments.fresh_source_replacement.scoring import curve_metrics

    assert curve_metrics(flat)["best_index"] == 0


def test_bootstrap_indices_are_pcg64_repeatable() -> None:
    first = make_bootstrap_indices(12)
    second = make_bootstrap_indices(12)
    assert first.dtype == np.int64
    assert first.shape == (20_000, 12)
    np.testing.assert_array_equal(first, second)


def test_independent_validator_rejects_resigned_analysis(tmp_path: Path, monkeypatch) -> None:
    """Raw arrays, not the producer's signed summary, own the endpoint."""

    repo = Path(__file__).resolve().parents[3]
    source_run = repo / "runs/fresh_source_cross_generator_20260911_unblock_r4"
    (tmp_path / "run").mkdir()
    os.symlink(source_run / "run" / "shared", tmp_path / "run" / "shared", target_is_directory=True)
    aroot = branch_dir(tmp_path, "A")
    aroot.mkdir(parents=True)

    shared_inputs = json.loads((source_run / "run" / "shared" / "inputs.json").read_text())
    formal = shared_inputs["records"][:12]
    videos: dict[str, dict] = {}
    scores: dict[str, dict] = {}
    array = np.random.default_rng(3).normal(size=(180, 8)).astype(np.float32)
    for record in formal:
        sid = str(record["sample_id"])
        natural_media = repo / "runs/fresh_source_cross_generator_20260911_unblock_r4/run/replacement/ditto/natural_raw" / f"{sid}.mkv"
        direct_media = repo / "runs/fresh_source_cross_generator_20260911_unblock_r4/run/replacement/ditto/direct_raw" / f"{sid}.mkv"
        for arm, media in (("N", natural_media), ("C", direct_media)):
            for seed in (42, 43):
                key = cell_key(sid, "wav2lip", arm, seed, 0)
                videos[key] = {
                    "key": key, "sample_id": sid, "source_group": record["source_group"], "model": "wav2lip",
                    "arm": arm, "seed": seed, "repeat_index": 0, "status": "complete", "path": str(media),
                    "sha256": file_sha256(media), "frame_count": 140, "fps": 25,
                }
                emb = aroot / "embeddings" / f"{key.replace('/', '__')}.npz"
                emb.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(emb, visual=array, audio=array)
                scores[key] = {
                    "key": key, "sample_id": sid, "source_group": record["source_group"], "model": "wav2lip",
                    "arm": arm, "seed": seed, "repeat_index": 0, "status": "complete",
                    "embedding_path": str(emb), "embedding_sha256": file_sha256(emb),
                }
        key = cell_key(sid, "wav2lip", "N", 42, 1)
        media = natural_media
        videos[key] = {
            "key": key, "sample_id": sid, "source_group": record["source_group"], "model": "wav2lip",
            "arm": "N", "seed": 42, "repeat_index": 1, "status": "complete", "path": str(media),
            "sha256": file_sha256(media), "frame_count": 140, "fps": 25,
        }
        emb = aroot / "embeddings" / f"{key.replace('/', '__')}.npz"
        np.savez_compressed(emb, visual=array, audio=array)
        scores[key] = {
            "key": key, "sample_id": sid, "source_group": record["source_group"], "model": "wav2lip",
            "arm": "N", "seed": 42, "repeat_index": 1, "status": "complete",
            "embedding_path": str(emb), "embedding_sha256": file_sha256(emb),
        }
    write_self_hashed(aroot / "videos.json", {"schema_version": 1, "model": "wav2lip", "status": "COMPLETE", "cells": videos, "records": list(videos.values())})
    write_self_hashed(aroot / "scores.json", {"schema_version": 1, "model": "wav2lip", "status": "COMPLETE", "videos_sha256": file_sha256(aroot / "videos.json"), "cells": scores})
    write_self_hashed(
        aroot / "ditto_ingest.json",
        {
            "schema_version": 1,
            "model": "ditto",
            "status": "BLOCKED_CROSS_GENERATOR",
            "read_only": True,
            "record_count": 0,
            "expected_record_count": 28,
            "errors": ["synthetic fixture isolates Wav2Lip validation"],
            "replacement_confirmed": False,
        },
    )

    expected = _critical_analysis(
        {"formal_records": [{**row, "_natural_audio_path": Path(row["natural_audio"]), "_direct_audio_path": Path(row["direct_audio"]["output_path"])} for row in formal]},
        scores,
    )
    expected.update({
        "inputs_sha256": file_sha256(source_run / "run" / "shared" / "inputs.json"),
        "scores_sha256": file_sha256(aroot / "scores.json"),
        "videos_sha256": file_sha256(aroot / "videos.json"),
        "cross_generator_status": "BLOCKED_CROSS_GENERATOR",
        "ditto_ingest_sha256": file_sha256(aroot / "ditto_ingest.json"),
    })
    write_self_hashed(aroot / "analysis.json", expected)
    # This unit isolates signed-analysis reconciliation; media/mux contracts
    # have their own strict checks and are intentionally not bypassed in the
    # real validator.
    monkeypatch.setattr("scripts.experiments.fresh_source_replacement.validate._validate_media", lambda *_args: None)
    first_validation = validate(tmp_path)
    assert first_validation["status"] == "GO"

    tampered = json.loads((aroot / "analysis.json").read_text())
    tampered["gains"]["sync_c"][0] = float(tampered["gains"]["sync_c"][0]) + 1.0
    write_self_hashed(aroot / "analysis.json", tampered)
    result = validate(tmp_path)
    assert result["status"] == "NO_GO"
    assert any("analysis.gains.sync_c" in error for error in result["errors"])
