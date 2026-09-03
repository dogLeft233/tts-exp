from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.masked_tts_reconstruction.features import build_example, fit_normalization, phone_phase_linear
from scripts.experiments.masked_tts_reconstruction.protocol import (
    assign_shuffled_donors,
    build_mask_manifest,
    edit_align_phones,
    is_administrative,
    leakage_audit,
    normalize_phone,
)


def _alignment(tmp_path: Path) -> Path:
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps({
        "natural_phones": [{"phone": "a", "start": 0.5, "end": 0.75}],
        "tts_phones": [{"phone": "a", "start": 0.4, "end": 0.8}],
    }), encoding="utf-8")
    return path


def _lock(tmp_path: Path, sample_id: str = "sample") -> dict:
    alignment = _alignment(tmp_path)
    return {"records": [{"sample_id": sample_id, "source_group": "6WeS1bXRBOk", "prototype_split": "train", "paths": {"alignment": str(alignment)}}]}


def test_phone_normalization_and_tie_order() -> None:
    assert normalize_phone(" é ") == "é"
    assert is_administrative(" SPN ")
    operations = edit_align_phones([{"phone": "a"}, {"phone": "b"}], [{"phone": "a"}, {"phone": "c"}])
    assert [row["operation"] for row in operations] == ["equal", "substitute"]
    operations = edit_align_phones([{"phone": "a"}], [{"phone": "a"}, {"phone": "b"}])
    assert [row["operation"] for row in operations] == ["equal", "insert"]


def test_mask_manifest_and_phase_alignment(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    mel = {"sample": np.zeros((80, 308), dtype=np.float32)}
    tts_lengths = {"sample": 50}
    manifest = build_mask_manifest(lock, mel, tts_lengths)
    assert manifest["counts"]["masks"] == 1
    row = manifest["masks"][0]
    assert row["natural_frame_count"] == 20
    assert row["core_end"] - row["core_start"] == 20
    assert row["mask_start"] == max(0, row["core_start"] - 4)
    assert row["mask_end"] == min(96, row["core_end"] + 4)
    source = np.arange(20 * 1024, dtype=np.float32).reshape(20, 1024)
    mapped, provenance = phone_phase_linear(source, 2, 10, 4)
    assert mapped.shape == (4, 1024)
    assert len(provenance) == 4
    assert provenance[0]["source_index_left"] == 2
    assert provenance[-1]["source_index_right"] <= 9


def test_example_zeroes_outside_core_and_leakage(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    mel = {"sample": np.ones((80, 308), dtype=np.float32)}
    tts = {"sample": np.ones((50, 1024), dtype=np.float32)}
    manifest = build_mask_manifest(lock, mel, {"sample": 50})
    stats = fit_normalization(mel, tts, {"sample"})
    example = build_example(manifest["masks"][0], mel, tts, stats)
    row = manifest["masks"][0]
    outside = np.ones(96, dtype=bool)
    outside[row["core_start"]:row["core_end"]] = False
    assert np.array_equal(example["tts_features"][outside], np.zeros((int(outside.sum()), 1024), dtype=np.float32))
    batch = {key: example[key] for key in ("natural_mel", "masked_support", "target_core", "tts_features", "target")}
    report = leakage_audit(batch)
    assert report["status"] == "GO"
    assert report["target_changes_without_input_changes"]
    with pytest.raises(ValueError):
        leakage_audit({**example, "natural_wavlm": np.zeros((96, 1024), dtype=np.float32)})


def test_shuffled_donor_is_train_only(tmp_path: Path) -> None:
    lock = {"records": []}
    masks = []
    for index, (group, split, label) in enumerate([
        ("6WeS1bXRBOk", "train", "a"),
        ("6tSlMoMNSlY", "evaluation", "b"),
    ]):
        masks.append({"mask_sha256": f"hash{index}", "sample_id": f"sample{index}", "source_group": group, "prototype_split": split, "label": label, "tts_duration_s": 0.2, "tts_frame_start": 1, "tts_frame_end": 4})
    result = assign_shuffled_donors({"masks": masks})
    target = next(row for row in result["masks"] if row["prototype_split"] == "evaluation")
    assert target["shuffled_donor"]["status"] == "assigned"
    assert target["shuffled_donor"]["donor_source_group"] == "6WeS1bXRBOk"
