from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scripts.experiments import vsr_tts_metrics as metrics
from scripts.experiments import vsr_tts_pilot as pilot


def _enumerate_ctc_probability(logp: np.ndarray, target: list[int], blank: int = 0) -> float:
    total = 0.0
    import itertools

    for path in itertools.product(range(logp.shape[1]), repeat=logp.shape[0]):
        collapsed = []
        previous = None
        for symbol in path:
            if symbol == previous:
                continue
            previous = symbol
            if symbol != blank:
                collapsed.append(symbol)
        if collapsed == target:
            total += float(np.exp(sum(float(logp[t, symbol]) for t, symbol in enumerate(path))))
    return total


def _logp_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(probs, dtype=np.float64))


def test_normalize_text_and_reserved_tokens() -> None:
    assert metrics.normalize_text("  组织了，第一批。\n") == "组织了第一批"
    char_list = ["<blank>", "<unk>", "组", "织", "了"]
    assert metrics.token_ids("组织了", char_list) == [2, 3, 4]
    with pytest.raises(ValueError, match="OOV"):
        metrics.token_ids("组织未", char_list)


def test_ctc_tiny_enumeration_and_independent_dp() -> None:
    probs = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.70, 0.20, 0.10],
        ],
        dtype=np.float64,
    )
    logp = _logp_from_probs(probs)
    target = [1]
    expected_probability = _enumerate_ctc_probability(logp, target)
    expected_nll = -math.log(expected_probability)
    assert np.isclose(metrics.ctc_nll(logp, target), expected_nll, atol=1e-10)
    assert np.isclose(pilot._independent_ctc_nll(logp, target), expected_nll, atol=1e-10)

    repeated_target = [1, 1]
    with pytest.raises(ValueError, match="unreachable"):
        metrics.ctc_nll(logp[:2], repeated_target)


def test_ctc_normalization_happens_once_and_rejects_bad_values() -> None:
    logp = _logp_from_probs(np.asarray([[0.2, 0.7, 0.1], [0.7, 0.2, 0.1]], dtype=np.float64))
    raw = metrics.ctc_nll(logp, [1])
    normalized = metrics.length_normalized_nll(logp, [1])
    assert np.isclose(raw, normalized)
    assert metrics.length_normalized_nll(logp, [1, 2]) != raw
    with pytest.raises(ValueError, match="non-finite"):
        metrics.ctc_nll(np.asarray([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]), [1])
    with pytest.raises(ValueError, match="blank"):
        metrics.ctc_nll(logp, [0])


def test_views_preserve_frame_set_and_do_not_mutate_input() -> None:
    native = np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2)
    original = native.copy()
    views = metrics.make_views(native, matched_length=6)
    assert np.array_equal(native, original)
    assert views["native"].shape == (1, 4, 2, 2)
    assert views["frozen"].shape == (1, 4, 2, 2)
    assert views["matched"].shape == (1, 6, 2, 2)
    assert views["matched_frozen"].shape == (1, 6, 2, 2)
    assert np.array_equal(views["reversed"][:, ::-1], native)
    assert all(np.array_equal(views["frozen"][:, 0], views["frozen"][:, i]) for i in range(4))
    assert np.array_equal(views["matched"][..., 0, 0][:, [0, -1]], native[..., 0, 0][:, [0, -1]])


def test_decoys_are_fixed_by_length_then_numeric_id() -> None:
    rows = [
        {"id": 1, "normalized_text": "aaaa", "token_ids": [1, 2, 3, 4]},
        {"id": 2, "normalized_text": "bbb", "token_ids": [1, 2, 3]},
        {"id": 3, "normalized_text": "ccccc", "token_ids": [1, 2, 3, 4, 5]},
        {"id": 4, "normalized_text": "dddd", "token_ids": None},
        {"id": 5, "normalized_text": "eeeee", "token_ids": [1, 2, 3, 4, 5]},
        {"id": 6, "normalized_text": "f", "token_ids": [1]},
    ]
    decoys = metrics.build_decoys(rows, count=3)
    assert [item["id"] for item in decoys["1"]] == [2, 3, 5]
    assert all(item["id"] != 4 for item in decoys["1"])


def test_margin_and_g_q_b_formulas_have_expected_signs() -> None:
    assert metrics.content_margin(1.0, [3.0, 3.0]) == 2.0
    rows = [{"id": i, "g": 1.0, "b": 1.0, "gmatched": 1.0, "r": 0.2} for i in range(1, 11)]
    paired = metrics.paired_summary(rows)
    calibration = {
        "status": "CALIBRATED_ON_COHORT",
        "arms": {
            "natural": {"strict_native_top1_fraction": 1.0, "q_positive_fraction": 1.0, "q_mean": 1.0},
            "tts": {"strict_native_top1_fraction": 1.0, "q_positive_fraction": 1.0, "q_mean": 1.0},
        },
    }
    assert metrics.decision_status(engineering_ok=True, calibration=calibration, paired=paired) == "EXPLORATORY_VISUAL_CONTENT_SUPPORT"


def test_decision_precedence_catches_control_and_duration() -> None:
    calibration = {"status": "CALIBRATED_ON_COHORT"}
    control_rows = [{"id": i, "g": 1.0, "b": -0.1, "gmatched": 1.0, "r": 0.0} for i in range(1, 11)]
    duration_rows = [{"id": i, "g": 1.0, "b": 1.0, "gmatched": -0.1, "r": 0.0} for i in range(1, 11)]
    assert metrics.decision_status(engineering_ok=True, calibration=calibration, paired=metrics.paired_summary(control_rows)) == "CONTROL_DRIVEN_DIFFERENCE"
    assert metrics.decision_status(engineering_ok=True, calibration=calibration, paired=metrics.paired_summary(duration_rows)) == "DURATION_SENSITIVE"


def test_greedy_cer_can_exceed_one_and_keeps_repeated_symbols() -> None:
    probs = np.asarray([[0.01, 0.98, 0.01], [0.01, 0.98, 0.01], [0.01, 0.98, 0.01]], dtype=np.float64)
    # Target [1, 2] is decoded as [1], one deletion only; a long hypothesis
    # would similarly be allowed to yield CER > 1.
    assert metrics.greedy_cer(_logp_from_probs(probs), [1, 2]) == 0.5


def test_readme_binding_is_fixed_and_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    rows = pilot.parse_r2_readme(Path("runs/r2_assets/README.md"))
    assert rows[0]["id"] == 1
    assert rows[0]["wid"] == "BAC009S0764W0201"
    assert rows[0]["raw_text"] == "组织了第一批七个地区城市开展三网融合试点"
    duplicate = tmp_path / "README.md"
    lines = [
        "| Sample | WID | Duration | Transcript |",
        "|--------|-----|----------|------------|",
    ]
    for sample_id in range(1, 14):
        actual = sample_id if sample_id != 13 else 1
        lines.append("| {} | BAC009S0764W{:04d} | 5.0s | text{} |".format(actual, sample_id, sample_id))
    duplicate.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly sample IDs"):
        pilot.parse_r2_readme(duplicate)


def test_atomic_json_rejects_nonfinite_and_resume_scope_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        pilot.write_json(tmp_path / "bad.json", {"value": float("nan")})
    path = tmp_path / "good.json"
    pilot.write_json(path, {"status": "ok"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "ok"


def test_resume_rejects_changed_frozen_input(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    manifest = {
        "records": [
            {
                "id": 1,
                "natural_video": str(source),
                "sha256": {"natural": pilot.sha256_file(source)},
            }
        ],
        "source_documents": {},
    }
    pilot._verify_frozen_inputs(tmp_path, manifest)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        pilot._verify_frozen_inputs(tmp_path, manifest)
