"""Scientific contract tests for the F0 contour exchange runner.

These tests use synthetic World arrays and never treat a production output as
the expected value.  They protect the time mapping, taper, gain, score matrix,
and speaker bootstrap contracts independently of the optional GPU backends.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "experiments" / "tts_f0_swap.py"
SPEC = importlib.util.spec_from_file_location("tts_f0_swap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RECOMPUTE_SCRIPT = Path(__file__).parents[2] / "scripts" / "experiments" / "tts_f0_swap_recompute.py"
RECOMPUTE_SPEC = importlib.util.spec_from_file_location("tts_f0_swap_recompute", RECOMPUTE_SCRIPT)
assert RECOMPUTE_SPEC is not None and RECOMPUTE_SPEC.loader is not None
RECOMPUTE = importlib.util.module_from_spec(RECOMPUTE_SPEC)
RECOMPUTE_SPEC.loader.exec_module(RECOMPUTE)


def _world(f0: np.ndarray, *, offset: float = 0.0) -> MODULE.WorldParams:
    f0 = np.asarray(f0, dtype=np.float64)
    time = np.arange(f0.size, dtype=np.float64) * 0.005 + 0.0025 + offset
    sp = np.ones((f0.size, 4), dtype=np.float64)
    ap = np.zeros_like(sp)
    return MODULE.WorldParams(f0=f0, time=time, sp=sp, ap=ap, sample_count=f0.size * 320)


def _grid(start: float, end: float, label: str = "a") -> dict[str, object]:
    return {"phones": [{"index": 0, "label": "", "normalized": "", "start_s": 0.0, "end_s": start, "silence": True, "unknown": False}, {"index": 1, "label": label, "normalized": label, "start_s": start, "end_s": end, "silence": False, "unknown": False}, {"index": 2, "label": "", "normalized": "", "start_s": end, "end_s": end + 0.05, "silence": True, "unknown": False}], "words": []}


def test_mapping_has_phone_local_taper_and_does_not_cross_silence() -> None:
    n = _world(np.full(45, 200.0))
    d = _world(np.full(45, 220.0), offset=0.0)
    mapping = MODULE.build_f0_mapping(n, d, _grid(0.05, 0.225), _grid(0.05, 0.225))
    assert mapping.valid.sum() > 0
    speech = np.flatnonzero(mapping.phone_index == 1)
    assert np.all(mapping.phone_index[mapping.valid] == 1)
    assert np.all(mapping.weights[:10] == 0.0)  # silence remains untouched
    assert np.all(mapping.weights[speech[0]] == 0.0)
    assert np.all(mapping.weights[speech[-1]] == 0.0)
    assert not np.any(mapping.valid[:10])


def test_mapping_and_intervention_support_different_receiver_donor_lengths() -> None:
    receiver = _world(np.full(60, 180.0))
    donor = _world(np.linspace(150.0, 250.0, 80))
    mapping = MODULE.build_f0_mapping(receiver, donor, _grid(0.0, 0.30), _grid(0.0, 0.40))
    result = MODULE.f0_interventions(receiver, donor, mapping)
    assert result["target_semitones"]["DONOR"].shape == receiver.f0.shape
    assert result["f0"]["CONTOUR"].shape == receiver.f0.shape
    assert np.isfinite(result["target_semitones"]["DONOR"][mapping.valid]).all()


def test_constant_donor_offset_is_level_only_and_contour_preserves_mean() -> None:
    n_f0 = np.full(50, 180.0)
    d_f0 = np.full(50, 360.0)
    # Make the time domains identical so each donor sample is directly mapped.
    n, d = _world(n_f0), _world(d_f0)
    grid = _grid(0.0, 0.25)
    mapping = MODULE.build_f0_mapping(n, d, grid, grid)
    result = MODULE.f0_interventions(n, d, mapping)
    np.testing.assert_allclose(result["f0"]["ID"], n_f0)
    valid = mapping.valid
    assert result["dose_contour_rms_st"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(result["f0"]["CONTOUR"][valid], n_f0[valid], atol=1e-10)
    assert result["f0"]["LEVEL"][valid].mean() > n_f0[valid].mean()
    assert abs(result["identity_error"]) <= 1e-12


def test_contour_changes_rhythm_without_changing_ordinary_mean() -> None:
    n_f0 = np.full(60, 180.0)
    d_f0 = np.linspace(140.0, 260.0, 60)
    n, d = _world(n_f0), _world(d_f0)
    grid = _grid(0.0, 0.30)
    mapping = MODULE.build_f0_mapping(n, d, grid, grid)
    result = MODULE.f0_interventions(n, d, mapping)
    assert result["dose_contour_rms_st"] > 0.5
    valid = mapping.valid
    ordinary_delta = np.mean(12 * np.log2(result["f0"]["CONTOUR"][n_f0 > 0.0]) - 12 * np.log2(n_f0[n_f0 > 0.0]))
    assert ordinary_delta == pytest.approx(0.0, abs=1e-10)
    # The repair contract does not force the taper-weighted mean to zero.
    weighted_delta = np.sum(mapping.weights[valid] * (12 * np.log2(result["f0"]["CONTOUR"][valid]) - 12 * np.log2(n_f0[valid]))) / np.sum(mapping.weights[valid])
    assert weighted_delta != pytest.approx(0.0, abs=1e-10)
    assert np.max(np.abs(result["f0"]["CONTOUR"][valid] - n_f0[valid])) > 1.0


def test_nonuniform_taper_uses_plain_mean_contract_and_v2_dose() -> None:
    # Hand-check the repair spec's counterexample.  The old w**2 correction
    # would make the weighted mean zero and therefore fail this expectation.
    receiver_st = np.full(5, 80.0)
    donor_st = receiver_st + np.array([0.0, 0.0, 2.0, 0.0, 0.0])
    receiver = _world(2.0 ** (receiver_st / 12.0))
    donor = _world(2.0 ** (donor_st / 12.0))
    mapping = MODULE.F0Mapping(
        receiver_indices=np.arange(5),
        donor_left=np.arange(5), donor_right=np.arange(5), donor_alpha=np.zeros(5),
        donor_time=donor.time.copy(), phone_index=np.zeros(5, dtype=np.int64),
        weights=np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
        valid=np.array([False, True, True, True, False]), invalid_reasons=("fade_endpoint",) * 5,
    )
    result = MODULE.f0_interventions(receiver, donor, mapping)
    delta = result["target_deltas"]["CONTOUR"]
    np.testing.assert_allclose(delta, np.array([0.0, -0.5, 1.0, -0.5, 0.0]))
    assert result["ordinary_identity_error"] == pytest.approx(0.0, abs=1e-12)
    assert result["weighted_identity_error"] == pytest.approx(0.25, abs=1e-12)
    assert result["dose_contour_rms_st"] == pytest.approx(np.sqrt(0.5), abs=1e-12)
    np.testing.assert_array_equal(result["f0"]["CONTOUR"][~mapping.valid], receiver.f0[~mapping.valid])


def test_stonemask_floor_tolerance_preserves_low_pitch_carrier_without_clipping() -> None:
    receiver = _world(np.array([55.0, 80.0, 100.0, 80.0, 55.0]))
    donor = _world(np.array([55.0, 80.0, 100.0, 80.0, 55.0]))
    mapping = MODULE.F0Mapping(
        receiver_indices=np.arange(5), donor_left=np.arange(5), donor_right=np.arange(5), donor_alpha=np.zeros(5),
        donor_time=donor.time.copy(), phone_index=np.zeros(5, dtype=np.int64),
        weights=np.ones(5, dtype=np.float64), valid=np.ones(5, dtype=bool), invalid_reasons=("valid",) * 5,
    )
    result = MODULE.f0_interventions(receiver, donor, mapping)
    assert result["f0"]["ID"][0] == pytest.approx(55.0)
    assert result["f0"]["CONTOUR"][0] == pytest.approx(55.0)
    with pytest.raises(MODULE.ProtocolError, match="StoneMask tolerance"):
        MODULE.f0_interventions(_world(np.array([49.0] * 5)), donor, mapping)


def test_length_and_common_peak_scaling_are_global() -> None:
    values = np.ones(100, dtype=np.float64) * 0.9
    cropped, crop_meta = MODULE.exact_length(values, 80)
    assert cropped.size == 80 and crop_meta["action"] == "right_crop"
    arrays, scale = MODULE._common_safe_scale({"RAW": values, "ID": values * 0.5})
    assert scale == pytest.approx(1.0)
    assert arrays["RAW"].max() == pytest.approx(0.9)
    assert arrays["ID"].max() == pytest.approx(0.45)


def test_distance_matrix_curve_and_expected_cells() -> None:
    visual = np.zeros((80, 1024), dtype=np.float32)
    audio = np.zeros((80, 1024), dtype=np.float32)
    visual[:, 0] = np.arange(80, dtype=np.float32)
    audio[:, 0] = np.arange(80, dtype=np.float32)
    matrix = MODULE.syncnet_distance_matrix(visual, audio)
    support = np.arange(15, 65, dtype=np.int64)
    metrics = MODULE.curve_metrics(matrix, support, k0=0)
    assert metrics["k_star"] == 0
    assert metrics["C"] >= 0.0
    # The frozen distance adds +1e-6 to each of 1024 dimensions, so the
    # identical-row distance is the deterministic epsilon norm (~3.2e-5).
    assert metrics["d_k0"] == pytest.approx(32e-6, abs=1e-7)
    assert len(MODULE.expected_score_cells()) == 10
    assert ("CONTOUR", "ID") in MODULE.expected_score_cells()


def test_delay_control_supports_both_signed_directions() -> None:
    values = np.arange(10, dtype=np.float64)
    np.testing.assert_array_equal(MODULE.delayed_audio(values, 2), np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float64))
    np.testing.assert_array_equal(MODULE.delayed_audio(values, -2), np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0], dtype=np.float64))


def test_audio_stage_selects_formal_or_pilot_role_without_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_build(record: dict[str, object], _paths: MODULE.RunPaths) -> dict[str, object]:
        calls.append(str(record["paired_key"]))
        return {
            "paired_key": record["paired_key"],
            "sample_id": record["sample_id"],
            "speaker_id": record["speaker_id"],
            "audio": {},
            "parameter_path": "synthetic.npz",
            "parameter_sha256": "synthetic",
            "qc": [
                {"paired_key": record["paired_key"], "sample_id": record["sample_id"], "speaker_id": record["speaker_id"], "receiver": receiver, "status": "PASS"}
                for receiver in ("N", "T")
            ],
            "world": {},
        }

    monkeypatch.setattr(MODULE, "build_audio_for_pair", fake_build)
    monkeypatch.setattr(MODULE, "_audio_backend_info", lambda: {"name": "synthetic"})
    inputs = {
        "records": [
            {"paired_key": "pilot", "sample_id": 1, "speaker_id": "S1", "role": "pilot"},
            {"paired_key": "formal", "sample_id": 2, "speaker_id": "S2", "role": "formal"},
        ]
    }
    paths = MODULE.RunPaths(tmp_path / "run")
    formal = MODULE.run_audio_stage(paths, inputs, pilot_only=False)
    assert calls == ["formal"]
    assert formal["pair_count"] == 1 and formal["expected_pair_count"] == 1
    pilot = MODULE.run_audio_stage(paths, inputs, pilot_only=True)
    assert calls == ["formal", "pilot"]
    assert pilot["pair_count"] == 1 and pilot["expected_pair_count"] == 1


def test_pilot_gate_counts_frozen_missing_pairs_as_failures() -> None:
    gate = MODULE.pilot_gate(
        {"expected_pair_count": 4, "manifests": [{"paired_key": "a", "qc": [{"receiver": "N", "status": "PASS"}, {"receiver": "T", "status": "PASS"}]}]},
        expected_keys=["a", "b", "c", "d"],
    )
    assert gate["pilot_pair_count"] == 4
    assert gate["passed_pairs"] == 1
    assert gate["status"] == "MANIPULATION_NOT_VALIDATED"


def test_independent_recompute_uses_explicit_score_cells() -> None:
    rows: list[dict[str, object]] = []
    def add(receiver: str, category: str, video: str, audio: str, value: float) -> None:
        rows.append({"status": "complete", "paired_key": "p", "receiver": receiver, "category": category, "video_arm": f"{receiver}_{video}", "audio_arm": f"{receiver}_{audio}", "C": value, "D": 0.0, "B": value, "d_k0": value})
    for receiver, offset in (("N", 0.0), ("T", 10.0)):
        for arm, value in zip(("RAW", "ID", "LEVEL", "CONTOUR"), (1.0, 2.0, 3.0, 4.0), strict=True):
            add(receiver, "own", arm, arm, value + offset)
        add(receiver, "fixed_audio", "LEVEL", "ID", 5.0 + offset)
        add(receiver, "fixed_audio", "CONTOUR", "ID", 7.0 + offset)
        add(receiver, "fixed_video", "ID", "LEVEL", 8.0 + offset)
        add(receiver, "fixed_video", "ID", "CONTOUR", 9.0 + offset)
        add(receiver, "raw_replacement", "ID", "RAW", 10.0 + offset)
        add(receiver, "raw_replacement", "CONTOUR", "RAW", 11.0 + offset)
    index = RECOMPUTE._row_index(rows)
    values = RECOMPUTE._record_contrasts("p", "S", index)
    assert values["g_N"] == pytest.approx(5.0)
    assert values["l_T"] == pytest.approx(-5.0)
    assert values["h_N"] == pytest.approx(2.0)
    assert values["h_T"] == pytest.approx(-2.0)
    assert values["G_raw"] == pytest.approx(10.0)
    assert values["G_id"] == pytest.approx(10.0)


def test_analyze_scores_keeps_both_raw_replacement_video_cells(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for receiver in ("N", "T"):
        for arm, value in zip(("RAW", "ID", "LEVEL", "CONTOUR"), (1.0, 2.0, 3.0, 4.0), strict=True):
            rows.append({"paired_key": "p", "receiver": receiver, "category": "own", "video_arm": f"{receiver}_{arm}", "audio_arm": f"{receiver}_{arm}", "C": value})
        rows.extend([
            {"paired_key": "p", "receiver": receiver, "category": "fixed_audio", "video_arm": f"{receiver}_LEVEL", "audio_arm": f"{receiver}_ID", "C": 5.0},
            {"paired_key": "p", "receiver": receiver, "category": "fixed_audio", "video_arm": f"{receiver}_CONTOUR", "audio_arm": f"{receiver}_ID", "C": 6.0},
            {"paired_key": "p", "receiver": receiver, "category": "fixed_video", "video_arm": f"{receiver}_ID", "audio_arm": f"{receiver}_LEVEL", "C": 7.0},
            {"paired_key": "p", "receiver": receiver, "category": "fixed_video", "video_arm": f"{receiver}_ID", "audio_arm": f"{receiver}_CONTOUR", "C": 8.0},
            {"paired_key": "p", "receiver": receiver, "category": "raw_replacement", "video_arm": f"{receiver}_ID", "audio_arm": f"{receiver}_RAW", "C": 9.0},
            {"paired_key": "p", "receiver": receiver, "category": "raw_replacement", "video_arm": f"{receiver}_CONTOUR", "audio_arm": f"{receiver}_RAW", "C": 10.0},
        ])
    result = MODULE.analyze_scores(
        MODULE.RunPaths(tmp_path / "run"),
        {"records": [{"paired_key": "p", "speaker_id": "S", "role": "formal"}]},
        {"rows": rows},
        {},
        controls={"status": "PASS"},
    )
    assert result["complete_pair_count"] == 1
    assert "missing_reason" not in result["records"][0]
    assert result["records"][0]["raw_replace_N"] == pytest.approx(1.0)


def test_speaker_bootstrap_is_reproducible_and_equal_weighted() -> None:
    values = [1.0, 3.0, 9.0, 11.0]
    speakers = ["b", "b", "a", "a"]
    first = MODULE.speaker_bootstrap(values, speakers)
    second = MODULE.speaker_bootstrap(values, speakers)
    assert first == second
    assert first["mean"] == pytest.approx((2.0 + 10.0) / 2.0)
    assert first["speaker_labels"] == ["a", "b"]
    assert first["ci98_75_bonferroni"][0] <= first["mean"] <= first["ci98_75_bonferroni"][1]


def test_textgrid_normalization_keeps_tone_marks_and_unknown_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "x.TextGrid"
    path.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n'
        'xmin = 0\nxmax = 0.3\nsize = 1\nitem []:\n item [1]:\n'
        ' class = "IntervalTier"\n name = "phones"\n xmin = 0\n xmax = 0.3\n intervals: size = 2\n'
        ' intervals [1]:\n xmin = 0\n xmax = 0.1\n text = " "\n'
        ' intervals [2]:\n xmin = 0.1\n xmax = 0.3\n text = "a˧˥"\n', encoding="utf-8")
    parsed = MODULE.parse_textgrid(path)
    assert MODULE.phone_sequence(parsed) == ["a˧˥"]
    assert parsed["phones"][0]["silence"] is True


def test_independent_measurement_rejects_time_grid_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWorld:
        @staticmethod
        def dio(values: np.ndarray, sample_rate: int, *, frame_period: float, f0_floor: float, f0_ceil: float) -> tuple[np.ndarray, np.ndarray]:
            del values, sample_rate, frame_period, f0_floor, f0_ceil
            return np.full(59, 180.0), np.arange(59, dtype=np.float64) * 0.005

        @staticmethod
        def stonemask(values: np.ndarray, f0: np.ndarray, time: np.ndarray, sample_rate: int) -> np.ndarray:
            del values, time, sample_rate
            return f0

    params = _world(np.full(60, 180.0))
    monkeypatch.setattr(MODULE, "_pyworld", lambda: FakeWorld())
    with pytest.raises(MODULE.ProtocolError, match="TIME_GRID_MISMATCH"):
        MODULE._independent_f0_measure(params, np.zeros(params.sample_count, dtype=np.float64))


def test_four_cell_diagnostic_preserves_fixed_reference_denominator() -> None:
    reference = np.array([180.0, 180.0, 0.0, 0.0])
    raw = np.array([180.0, 180.0, 180.0, 0.0])
    identity = np.array([180.0, 0.0, 0.0, 180.0])
    result = MODULE._four_cell_counts(reference, raw, identity)
    assert result["reference_voiced"] == 2
    assert result["both_raw_and_id_voiced"] == 1
    assert result["only_raw_voiced"] == 1
    assert result["only_id_voiced"] == 0
    assert result["both_unvoiced"] == 0
