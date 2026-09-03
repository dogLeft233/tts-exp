from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

from scripts.experiments.masked_tts_tfg_probe.cohort import EXPECTED_GROUPS, freeze_cohort, selection_key
from scripts.experiments.masked_tts_tfg_probe.direct_mel import chunk_mels
from scripts.experiments.masked_tts_tfg_probe.mel_drivers import patch_normalized_mel
from scripts.experiments.masked_tts_tfg_probe.phase_a import SEEDS


REPO = Path(__file__).resolve().parents[3]
PARENT = REPO / "runs/lrs3_masked_tts_retention_exploratory_20260901"
RUN = REPO / "runs/lrs3_masked_tts_tfg_probe_20260902"


def test_selection_key_is_salted_sample_id_hash() -> None:
    sample_id = "lrs3_6ORDQFh0Byw_00012"
    expected = hashlib.sha256(("masked-tts-tfg-probe-v1\0" + sample_id).encode()).hexdigest()
    assert selection_key(sample_id) == expected


def test_frozen_cohort_is_one_maximum_coverage_record_per_group() -> None:
    diagnosis = RUN / "00_diagnosis/analysis.json"
    with tempfile.TemporaryDirectory() as directory:
        manifest = freeze_cohort(PARENT, diagnosis, Path(directory) / "manifest.json")
    assert manifest["record_count"] == 8
    assert tuple(row["source_group"] for row in manifest["records"]) == EXPECTED_GROUPS
    assert {row["source_group"] for row in manifest["records"]} >= {"6ORDQFh0Byw", "6yR5OUVb2gY"}
    assert all(row["evaluation_mask_count"] > 0 for row in manifest["records"])


def test_frozen_chunk_rule_uses_last_window_without_padding() -> None:
    mel = np.arange(80 * 308, dtype=np.float32).reshape(80, 308)
    chunks = chunk_mels(mel, 25.0)
    assert len(chunks) == 93
    assert np.array_equal(chunks[0], mel[:, :16])
    assert np.array_equal(chunks[-1], mel[:, -16:])


def test_mel_patch_preserves_uncovered_frames_and_averages_overlap() -> None:
    natural = np.zeros((80, 10), dtype=np.float32)
    first = np.ones((3, 80), dtype=np.float32)
    second = np.full((3, 80), 3.0, dtype=np.float32)
    result, counts = patch_normalized_mel(natural, [(2, 5, first), (4, 7, second)])
    assert np.array_equal(counts, np.array([0, 0, 1, 1, 2, 1, 1, 0, 0, 0], dtype=np.int32))
    assert np.array_equal(result[:, :2], natural[:, :2])
    assert np.array_equal(result[:, 2:4], np.ones((80, 2), dtype=np.float32))
    assert np.array_equal(result[:, 4:5], np.full((80, 1), 2.0, dtype=np.float32))
    assert np.array_equal(result[:, 5:7], np.full((80, 2), 3.0, dtype=np.float32))
    assert np.array_equal(result[:, 7:], natural[:, 7:])


def test_phase_a_has_three_frozen_seeds() -> None:
    analysis = RUN / "00_diagnosis/analysis.json"
    assert analysis.is_file()
    assert SEEDS == (20260901, 20260902, 20260903)
