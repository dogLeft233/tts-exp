from __future__ import annotations

import importlib.util
from dataclasses import replace
import numpy as np
import unittest

from scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt06 import (
    FitRoster,
    FitScoreRow,
    FitSelectionError,
    MethodSpec,
    METRIC_PARITY_ATOL,
    METRIC_PARITY_RTOL,
    ProposalEvaluation,
    SPSAController,
    SnapshotSpec,
    _sha256,
    _metric_on_keys,
    canonical_input_pins_sha256,
    expected_score_accounting,
    full_fit_candidate_binding_sha256,
    objective_from_effects,
    official_stagea_score_row,
    select_snapshot_family,
)


MODELS = ("wav2lip", "musetalk", "ditto")
SAMPLES = {f"sp{speaker:02d}_clip{clip}": f"sp{speaker:02d}" for speaker in range(20) for clip in range(2)}
ROSTER = FitRoster(SAMPLES, MODELS)
PINS = {"plan": {"path": "/frozen/plan.json", "bytes": 12, "sha256": _sha256(b"plan")}}
PINS_SHA = canonical_input_pins_sha256(PINS)
KEYS = tuple((i, i, 25, i) for i in range(27))
VALID = (True,) * 27


def _digest(label: str) -> str:
    return _sha256(label.encode("utf-8"))


def _curve(kind: str) -> np.ndarray:
    curve = np.ones(31, dtype=np.float32)
    if kind in {"identity", "rms", "rms_fixed", "equal"}:
        curve[4] = 0.8
        curve[15] = 1.0
    elif kind == "good":
        curve[4] = 0.6
        curve[15] = 0.95
    elif kind == "weak":
        curve[4] = 0.82
        curve[15] = 1.01
    else:
        raise AssertionError(kind)
    return curve


def _spec(method_id: str, artifact_label: str | None = None, binding_label: str | None = None) -> MethodSpec:
    return MethodSpec(method_id, _digest(artifact_label or f"artifact:{method_id}"),
                      _digest(binding_label or f"binding:{method_id}"))


def _row(
    sample_id: str,
    model_id: str,
    spec: MethodSpec,
    roster: FitRoster = ROSTER,
    *,
    curve_kind: str | None = None,
    keys: tuple[tuple[int, int, int, int], ...] = KEYS,
    valid: tuple[bool, ...] = VALID,
    status: str = "PASS",
    natural_sha: str | None = None,
    input_sha: str = PINS_SHA,
) -> FitScoreRow:
    matrix = None
    if status == "PASS":
        curve = _curve(curve_kind or spec.method_id)
        matrix = np.tile(curve[None, :], (len(keys), 1)).astype("<f4")
    speaker = roster.sample_to_speaker[sample_id]
    return FitScoreRow(
        row_id=f"{spec.method_id}:{sample_id}:{model_id}", split="fit", sample_id=sample_id,
        speaker_id=speaker, model_id=model_id, face_id=f"face:{sample_id}", seed=20261003,
        method_id=spec.method_id, status=status, natural_pcm_sha256=natural_sha or _digest(f"pcm:{sample_id}"),
        source_video_sha256=_digest(f"video:{sample_id}"), source_face_sha256=_digest(f"face:{sample_id}"),
        artifact_sha256=spec.artifact_sha256, binding_sha256=spec.binding_sha256,
        input_pins_sha256=input_sha, query_keys=keys if status == "PASS" else (),
        valid_query_mask=valid if status == "PASS" else (), distance_by_query_lag=matrix,
        failure_reason=None if status == "PASS" else "synthetic test failure",
    )


def _rows(specs, *, roster=ROSTER, sample_ids=None, curve_kinds=None, mask=None, failed=None, pcm_override=None):
    curve_kinds = curve_kinds or {}
    mask = mask or {}
    failed = failed or set()
    pcm_override = pcm_override or {}
    samples = tuple(roster.sample_ids if sample_ids is None else sample_ids)
    rows = []
    for spec in specs:
        default_curve = spec.method_id if spec.method_id in {"identity", "rms", "rms_fixed"} else "good"
        for sample_id in samples:
            for model_id in roster.model_ids:
                key = (spec.method_id, sample_id, model_id)
                valid = list(VALID)
                for index in mask.get(key, ()):
                    valid[index] = False
                status = "FAILED" if key in failed else "PASS"
                rows.append(_row(sample_id, model_id, spec, roster,
                                 curve_kind=curve_kinds.get(key, curve_kinds.get(spec.method_id, default_curve)),
                                 valid=tuple(valid), status=status,
                                 natural_sha=pcm_override.get(key)))
    return rows


def _snapshot(method_id: str, step: int, *, curve_kind: str = "good") -> SnapshotSpec:
    artifact = _digest(f"artifact:{method_id}")
    theta = (0.0,) * 9
    binding = full_fit_candidate_binding_sha256(method_id=method_id, artifact_sha256=artifact,
                                               input_pins_sha256=PINS_SHA, source_step=step,
                                               theta=theta, point_kind="snapshot")
    return SnapshotSpec(method_id, artifact, binding, step, theta, "snapshot")


def _initial_family(*, mask=None, drop=None, curve_kinds=None, metric_backend="numpy_diagnostic",
                    rms_method_id="rms"):
    refs = (_spec("identity"), _spec(rms_method_id))
    candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
    specs = (*refs, *candidates)
    rows = _rows(specs, curve_kinds=curve_kinds, mask=mask)
    if drop is not None:
        rows = [row for row in rows if (row.method_id, row.sample_id, row.model_id) != drop]
    result = select_snapshot_family(rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
                                    expected_input_pins_sha256=PINS_SHA,
                                    metric_backend=metric_backend)
    return refs, candidates, rows, result


class FitSelectorTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "canonical selector admission requires PyTorch CPU")
    def test_snapshot_family_uses_one_six_method_support_and_correct_effect_directions(self) -> None:
        mask = {("step120", "sp00_clip0", "wav2lip"): (26,)}
        _refs, _candidates, _rows_, result = _initial_family(mask=mask, metric_backend="torch_cpu")

        self.assertEqual(result.status, "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV")
        self.assertEqual(result.metric_backend, "torch_cpu")
        self.assertEqual(result.selected_method_id, "step030")  # exact tie breaks by source step
        self.assertEqual(result.accounting["unique_official_score_rows"], 720)
        self.assertEqual(result.accounting["logical_rows_per_candidate_bundle"], 360)
        self.assertEqual(result.accounting["common_query_keys_total"], 40 * 3 * 27 - 1)
        first = result.candidate_decisions[0]
        self.assertAlmostEqual(first.gates["models"]["wav2lip"]["identity"]["Cbest"], 0.2, delta=1e-6)
        self.assertAlmostEqual(first.gates["models"]["wav2lip"]["identity"]["D0"], 0.05, delta=1e-6)
        self.assertAlmostEqual(first.gates["models"]["wav2lip"]["identity"]["Canchor"], 0.05, delta=1e-6)
        self.assertTrue(all(d.feasible for d in result.candidate_decisions))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "canonical selector admission requires PyTorch CPU")
    def test_rms_fixed_is_bound_as_the_raw_stagea_rms_reference_id(self) -> None:
        _refs, _candidates, _rows_, result = _initial_family(
            metric_backend="torch_cpu", rms_method_id="rms_fixed")
        self.assertEqual(result.status, "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV")
        self.assertEqual(result.accounting["reference_role_mapping"],
                         {"identity": "identity", "rms": "rms_fixed"})
        self.assertEqual(result.candidate_decisions[0].gates["reference_role_mapping"],
                         {"identity": "identity", "rms": "rms_fixed"})

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "canonical Stage-A family selection requires PyTorch CPU")
    def test_actual_stagea_row_adapter_preserves_rms_fixed_through_full_selection(self) -> None:
        refs = (_spec("identity"), _spec("rms_fixed"))
        candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
        specs = (*refs, *candidates)
        arrays = {
            "query_frame_index": np.arange(27, dtype="<i4"),
            "query_timestamp_num": np.arange(27, dtype="<i8"),
            "query_timestamp_den": np.full(27, 25, dtype="<i8"),
            "track_frame_index": np.arange(27, dtype="<i4"),
            "valid_query_mask": np.ones(27, dtype=np.bool_),
            "distance_by_query_lag": np.tile(_curve("good")[None, :], (27, 1)).astype("<f4"),
            "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
        }
        array_meta = {
            name: {
                "dtype": np.asarray(value).dtype.str,
                "shape": list(np.asarray(value).shape),
                "array_sha256": _sha256(np.ascontiguousarray(value).tobytes(order="C")),
            }
            for name, value in arrays.items()
        }
        rows = []
        for spec in specs:
            curve_kind = spec.method_id if spec.method_id in {"identity", "rms_fixed"} else "good"
            row_arrays = {
                **arrays,
                "distance_by_query_lag": np.tile(_curve(curve_kind)[None, :], (27, 1)).astype("<f4"),
            }
            row_array_meta = {
                **array_meta,
                "distance_by_query_lag": {
                    "dtype": row_arrays["distance_by_query_lag"].dtype.str,
                    "shape": list(row_arrays["distance_by_query_lag"].shape),
                    "array_sha256": _sha256(
                        np.ascontiguousarray(row_arrays["distance_by_query_lag"]).tobytes(order="C")),
                },
            }
            for sample_id, speaker_id in ROSTER.sample_to_speaker.items():
                for model_id in ROSTER.model_ids:
                    natural_sha = _digest(f"natural:{sample_id}")
                    record = {
                        "schema_version": "cross_tfg_audio_head_stagea_v1",
                        "status": "PASS", "failure_reason": None,
                        "row_id": f"{spec.method_id}:{sample_id}:{model_id}", "split": "fit",
                        "sample_id": sample_id, "speaker_id": speaker_id, "model_id": model_id,
                        "face_id": f"face:{sample_id}", "seed": 20261003,
                        "method_id": spec.method_id, "natural_pcm_sha256": natural_sha,
                        "score_audio_sha256": natural_sha,
                        "source_video_sha256": _digest(f"video:{sample_id}"),
                        "source_face_sha256": _digest(f"face:{sample_id}"),
                        "score": {
                            "raw_matrix_npz": {
                                "path": f"/qualified/{spec.method_id}/{sample_id}/{model_id}.npz",
                                "file_sha256": _digest(f"npz:{spec.method_id}:{sample_id}:{model_id}"),
                                "arrays": row_array_meta,
                            },
                            "support_by_support": {
                                "full": {"query_count": 27, "support_mask_sha256": _digest("full-support")},
                            },
                            "metrics_by_support": {
                                "full": {"status": "COMPLETE", "anchor_column": 15,
                                         "zero_lag_column": 15},
                            },
                        },
                    }
                    rows.append(official_stagea_score_row(
                        record, row_arrays, artifact_sha256=spec.artifact_sha256,
                        binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                        expected_anchor_column=15,
                    ))
        result = select_snapshot_family(
            rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
            expected_input_pins_sha256=PINS_SHA, metric_backend="torch_cpu",
        )
        self.assertEqual(result.status, "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV")
        self.assertEqual(result.accounting["observed_unique_rows"], 720)
        self.assertEqual(result.accounting["reference_role_mapping"],
                         {"identity": "identity", "rms": "rms_fixed"})

    def test_numpy_diagnostic_cannot_select_or_seed_refinement(self) -> None:
        _refs, _candidates, _rows_, result = _initial_family(metric_backend="numpy_diagnostic")
        self.assertEqual(result.status, "NUMPY_DIAGNOSTIC_ONLY_NOT_ADMITTED")
        self.assertIsNone(result.selected_method_id)
        self.assertIsNone(result.refinement_seed_method_id)
        self.assertEqual(result.candidate_decisions, ())
        self.assertTrue(result.accounting["diagnostic_only"])

    def test_any_missing_full_family_row_blocks_selection_and_refinement_seed(self) -> None:
        _refs, _candidates, _rows_, result = _initial_family(drop=("step090", "sp11_clip1", "ditto"))

        self.assertEqual(result.status, "CONTROL_FAILED_MISSING_FIT_EVIDENCE")
        self.assertIsNone(result.selected_method_id)
        self.assertIsNone(result.refinement_seed_method_id)
        self.assertEqual(result.candidate_decisions, ())
        self.assertTrue(any("missing 1 planned rows" in error for error in result.errors))

    def test_failed_row_preserves_the_full_planned_denominator(self) -> None:
        refs = (_spec("identity"), _spec("rms"))
        candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
        failed_key = ("step060", "sp03_clip1", "musetalk")
        rows = _rows((*refs, *candidates), failed={failed_key})
        result = select_snapshot_family(rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
                                        expected_input_pins_sha256=PINS_SHA,
                                        metric_backend="numpy_diagnostic")

        self.assertEqual(result.status, "CONTROL_FAILED_MISSING_FIT_EVIDENCE")
        self.assertEqual(result.accounting["unique_official_score_rows"], 720)
        self.assertEqual(result.accounting["observed_unique_rows"], 720)
        self.assertEqual(result.accounting["failed_score_rows"], 1)
        self.assertEqual(result.accounting["missing_score_rows"], 0)
        self.assertIsNone(result.refinement_seed_method_id)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "canonical selector admission requires PyTorch CPU")
    def test_zero_cbest_is_not_feasible_and_only_seeds_refinement(self) -> None:
        candidates = tuple(_snapshot(f"flat{step}", step) for step in (30, 60, 90, 120))
        refs = (_spec("identity"), _spec("rms"))
        rows = _rows((*refs, *candidates), curve_kinds={candidate.method_id: "equal" for candidate in candidates})
        result = select_snapshot_family(rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
                                        expected_input_pins_sha256=PINS_SHA,
                                        metric_backend="torch_cpu")

        self.assertEqual(result.status, "NOT_ADMITTED")
        self.assertIsNone(result.selected_method_id)
        self.assertEqual(result.refinement_seed_method_id, "flat30")
        self.assertFalse(result.candidate_decisions[0].feasible)
        self.assertGreater(result.candidate_decisions[0].failed_gate_count, 0)

    def test_snapshot_selector_rejects_minibatch_evidence(self) -> None:
        refs = (_spec("identity"), _spec("rms"))
        candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
        rows = _rows((*refs, *candidates))
        with self.assertRaisesRegex(FitSelectionError, "full-Fit only"):
            select_snapshot_family(rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
                                   expected_input_pins_sha256=PINS_SHA, metric_backend="numpy_diagnostic",
                                   sample_ids=tuple(ROSTER.sample_ids[:8]))

    def test_fit_family_rejects_dev_or_confirm_rows_even_with_exact_roster(self) -> None:
        refs = (_spec("identity"), _spec("rms"))
        candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
        original_rows = _rows((*refs, *candidates))
        for wrong_split in ("dev", "confirm"):
            with self.subTest(split=wrong_split):
                rows = [replace(row, split=wrong_split) for row in original_rows]
                result = select_snapshot_family(
                    rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
                    expected_input_pins_sha256=PINS_SHA, metric_backend="numpy_diagnostic",
                )
                self.assertEqual(result.status, "CONTROL_FAILED_MISSING_FIT_EVIDENCE")
                self.assertIsNone(result.selected_method_id)
                self.assertIsNone(result.refinement_seed_method_id)
                self.assertTrue(any("non-fit split row" in error for error in result.errors))


def _effect_fixture(worst_model_delta: float):
    from scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt06 import MethodEffects

    models = {model: {ref: {"D0": 0.02, "Cbest": 0.10, "Canchor": 0.02}
                      for ref in ("identity", "rms")} for model in MODELS}
    models[MODELS[-1]]["rms"]["Cbest"] = worst_model_delta
    equal = {
        ref: {metric: float(np.mean([models[m][ref][metric] for m in MODELS]))
              for metric in ("D0", "Cbest", "Canchor")}
        for ref in ("identity", "rms")
    }
    cbest_by_model = {m: models[m]["rms"]["Cbest"] for m in MODELS}
    return MethodEffects(models, cbest_by_model, equal, min(cbest_by_model.values()), _digest("support"),
                         {"common_contexts": 120, "common_query_keys": 3240,
                          "min_query_keys_per_context": 27, "max_query_keys_per_context": 27}, True, "COMPLETE")


def _controller(metric_backend="numpy_diagnostic", rms_method_id="rms") -> SPSAController:
    refs = (_spec("identity"), _spec(rms_method_id))
    return SPSAController(initial_theta=(0.0,) * 9, input_pins=PINS,
                          candidate_artifact_sha256=_digest("frozen-d0-head"), roster=ROSTER,
                          reference_specs=refs, metric_backend=metric_backend)


def _evaluation(proposal, refs, roster, *, curve_kind="good", failed=(), pcm_override=None, batch_id=None):
    candidate = MethodSpec(proposal.method_id, proposal.artifact_sha256, proposal.binding_sha256)
    specs = (*refs, candidate)
    rows = _rows(specs, roster=roster, sample_ids=proposal.sample_ids,
                 curve_kinds={proposal.method_id: curve_kind}, failed=set(failed),
                 pcm_override=pcm_override)
    return ProposalEvaluation(
        proposal_id=proposal.proposal_id, pair_id=proposal.pair_id, side=proposal.side,
        method_id=proposal.method_id, theta=proposal.theta, theta_sha256=proposal.theta_sha256,
        artifact_sha256=proposal.artifact_sha256, binding_sha256=proposal.binding_sha256,
        input_pins_sha256=proposal.input_pins_sha256, batch_id=batch_id or proposal.batch_id,
        rows=tuple(rows),
    )


class SPSATests(unittest.TestCase):
    def test_objective_penalizes_a_weak_worst_model(self) -> None:
        theta = (0.0,) * 9
        good = objective_from_effects(_effect_fixture(0.10), theta, theta)
        weak = objective_from_effects(_effect_fixture(-0.05), theta, theta)

        self.assertGreater(weak["L_c"], good["L_c"])
        self.assertGreater(weak["L_total"], good["L_total"])

    def test_spsa_ask_is_deterministic_paired_and_uses_nine_unclipped_directions(self) -> None:
        left, right = _controller("torch_cpu"), _controller("torch_cpu")
        pair_a, pair_b = left.ask(), right.ask()

        self.assertEqual(pair_a.direction, pair_b.direction)
        self.assertEqual(pair_a.plus.theta, pair_b.plus.theta)
        self.assertEqual(pair_a.minus.theta, pair_b.minus.theta)
        self.assertEqual(len(pair_a.plus.theta), 9)
        self.assertEqual(pair_a.plus.batch_id, pair_a.minus.batch_id)
        self.assertEqual(len(pair_a.sample_ids), 8)
        self.assertEqual(len(pair_a.speakers), 4)
        self.assertAlmostEqual(abs(pair_a.plus.theta[0] - pair_a.center_theta[0]), pair_a.c_k)
        self.assertAlmostEqual(abs(pair_a.minus.theta[0] - pair_a.center_theta[0]), pair_a.c_k)
        accounting = expected_score_accounting(2, context_count=24)
        self.assertEqual(accounting["unique_official_score_rows"], 96)
        full_pair = expected_score_accounting(2, context_count=120)
        self.assertEqual(full_pair["shared_reference_rows"], 240)
        self.assertEqual(full_pair["unique_official_score_rows"], 480)

    def test_numpy_diagnostic_cannot_issue_spsa_proposals(self) -> None:
        controller = _controller("numpy_diagnostic")
        before = controller.theta
        with self.assertRaisesRegex(FitSelectionError, "NUMPY_DIAGNOSTIC_ONLY"):
            controller.ask()
        self.assertEqual(controller.theta, before)
        self.assertEqual(controller.iteration, 0)
        self.assertFalse(controller.halted)

    def test_spsa_successful_tell_updates_once_and_requests_full_centers_only_at_frozen_steps(self) -> None:
        controller, replay = (_controller("torch_cpu", "rms_fixed"),
                              _controller("torch_cpu", "rms_fixed"))
        pair, replay_pair = controller.ask(), replay.ask()
        refs = controller.reference_specs
        plus = _evaluation(pair.plus, refs, ROSTER, curve_kind="good")
        minus = _evaluation(pair.minus, refs, ROSTER, curve_kind="weak")
        replay_outcome = replay.tell(_evaluation(replay_pair.plus, refs, ROSTER, curve_kind="good"),
                                     _evaluation(replay_pair.minus, refs, ROSTER, curve_kind="weak"))
        outcome = controller.tell(plus, minus)

        self.assertEqual(outcome.status, "UPDATED")
        self.assertNotEqual(outcome.theta_after, outcome.theta_before)
        self.assertEqual(controller.iteration, 1)
        self.assertEqual(outcome.theta_after, replay_outcome.theta_after)
        self.assertFalse(outcome.full_center_score_due)
        self.assertIsNone(outcome.full_center_spec)
        self.assertIsNotNone(outcome.support_sha256)

    def test_spsa_emits_full_center_identity_only_at_k5_k10_k15_k20(self) -> None:
        controller = _controller("torch_cpu")
        last = None
        for _ in range(5):
            pair = controller.ask()
            plus = _evaluation(pair.plus, controller.reference_specs, ROSTER, curve_kind="good")
            minus = _evaluation(pair.minus, controller.reference_specs, ROSTER, curve_kind="good")
            last = controller.tell(plus, minus)
            self.assertEqual(last.status, "UPDATED")
        self.assertTrue(last.full_center_score_due)
        self.assertEqual(last.full_center_spec.method_id, "fit40_d0_spsa_center_k05")
        self.assertEqual(last.full_center_spec.point_kind, "spsa_center")
        self.assertEqual(last.full_center_spec.theta, controller.theta)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "canonical center selection requires PyTorch CPU")
    def test_center_family_requires_all_four_predeclared_k_values(self) -> None:
        refs = (_spec("identity"), _spec("rms"))
        centers = tuple(
            SnapshotSpec(
                f"center{k:02d}", _digest(f"artifact:center{k}"),
                full_fit_candidate_binding_sha256(
                    method_id=f"center{k:02d}", artifact_sha256=_digest(f"artifact:center{k}"),
                    input_pins_sha256=PINS_SHA, source_step=k, theta=(float(k),) * 9,
                    point_kind="spsa_center"),
                k, (float(k),) * 9, "spsa_center")
            for k in (5, 10, 15, 20)
        )
        rows = _rows((*refs, *centers))
        result = select_snapshot_family(rows, roster=ROSTER, reference_specs=refs, candidates=centers,
                                        expected_input_pins_sha256=PINS_SHA,
                                        metric_backend="torch_cpu")
        self.assertEqual(result.status, "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV")
        self.assertIsNone(result.refinement_seed_method_id)
        self.assertEqual(result.accounting["unique_official_score_rows"], 720)

    def test_spsa_failed_or_mismatched_pair_never_changes_theta_or_retries(self) -> None:
        controller = _controller("torch_cpu")
        pair = controller.ask()
        refs = controller.reference_specs
        plus = _evaluation(pair.plus, refs, ROSTER, curve_kind="good")
        bad_minus = _evaluation(pair.minus, refs, ROSTER, curve_kind="good", batch_id="wrong-batch")
        before = controller.theta
        outcome = controller.tell(plus, bad_minus)

        self.assertEqual(outcome.status, "NO_UPDATE")
        self.assertEqual(outcome.theta_after, before)
        self.assertEqual(controller.theta, before)
        self.assertTrue(controller.halted)
        with self.assertRaisesRegex(FitSelectionError, "halted"):
            controller.ask()

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "canonical SPSA family validation requires PyTorch CPU")
    def test_spsa_incomplete_candidate_or_changed_source_pair_returns_no_update(self) -> None:
        controller = _controller("torch_cpu")
        pair = controller.ask()
        refs = controller.reference_specs
        bad_key = (pair.plus.method_id, pair.sample_ids[0], MODELS[0])
        plus = _evaluation(pair.plus, refs, ROSTER, curve_kind="good", failed={bad_key})
        minus = _evaluation(pair.minus, refs, ROSTER, curve_kind="good")
        before = controller.theta
        outcome = controller.tell(plus, minus)
        self.assertEqual(outcome.status, "NO_UPDATE")
        self.assertEqual(controller.theta, before)

        controller = _controller("torch_cpu")
        pair = controller.ask()
        plus = _evaluation(pair.plus, refs, ROSTER, curve_kind="good")
        wrong_pcm_key = (pair.minus.method_id, pair.sample_ids[0], MODELS[0])
        minus = _evaluation(pair.minus, refs, ROSTER, curve_kind="good",
                            pcm_override={wrong_pcm_key: _digest("different-source-pcm")})
        outcome = controller.tell(plus, minus)
        self.assertEqual(outcome.status, "NO_UPDATE")
        self.assertIn("natural_pcm_sha256 differs", outcome.reason or "")

    def test_official_adapter_accepts_stagea_shape_and_rejects_w2l_proxy_shape(self) -> None:
        spec = _spec("rms_fixed")
        record = {
            "schema_version": "cross_tfg_audio_head_stagea_v1", "status": "PASS", "failure_reason": None,
            "row_id": "r1", "split": "fit", "sample_id": "sp00_clip0", "speaker_id": "sp00",
            "model_id": MODELS[0], "face_id": "face:sp00_clip0", "seed": 20261003, "method_id": "rms_fixed",
            "natural_pcm_sha256": _digest("pcm"), "score_audio_sha256": _digest("pcm"),
            "source_video_sha256": _digest("video"), "source_face_sha256": _digest("face"),
        }
        arrays = {
            "query_frame_index": np.arange(27, dtype="<i4"),
            "query_timestamp_num": np.arange(27, dtype="<i8"),
            "query_timestamp_den": np.full(27, 25, dtype="<i8"),
            "track_frame_index": np.arange(27, dtype="<i4"),
            "valid_query_mask": np.ones(27, dtype=np.bool_),
            "distance_by_query_lag": np.ones((27, 31), dtype="<f4"),
            "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
        }
        record["score"] = {
            "raw_matrix_npz": {
                "path": "/qualified/raw_score_arrays.npz",
                "file_sha256": _digest("raw-npz-file"),
                "arrays": {
                    name: {
                        "dtype": np.asarray(array).dtype.str,
                        "shape": list(np.asarray(array).shape),
                        "array_sha256": _sha256(np.ascontiguousarray(array).tobytes(order="C")),
                    }
                    for name, array in arrays.items()
                },
            },
            # This is the Stage-A family common-support count, not raw NPZ N.
            "support_by_support": {"full": {"query_count": 18, "support_mask_sha256": _digest("support")}},
            "metrics_by_support": {
                "full": {"anchor_column": 15, "zero_lag_column": 15, "status": "COMPLETE"},
            },
        }
        adapted = official_stagea_score_row(record, arrays, artifact_sha256=spec.artifact_sha256,
                                            binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                            expected_anchor_column=15)
        self.assertEqual(adapted.distance_by_query_lag.shape, (27, 31))
        self.assertEqual(adapted.method_id, "rms_fixed")
        self.assertEqual(adapted.query_keys[1], (1, 1, 25, 1))
        self.assertEqual(adapted.anchor_column, 15)
        with self.assertRaisesRegex(FitSelectionError, "metadata does not bind"):
            bad_record = {**record, "score": {**record["score"]}}
            bad_record["score"]["raw_matrix_npz"] = {**record["score"]["raw_matrix_npz"]}
            bad_record["score"]["raw_matrix_npz"]["arrays"] = {
                **record["score"]["raw_matrix_npz"]["arrays"]}
            bad_record["score"]["raw_matrix_npz"]["arrays"]["distance_by_query_lag"] = {
                **record["score"]["raw_matrix_npz"]["arrays"]["distance_by_query_lag"],
                "array_sha256": _digest("wrong-array"),
            }
            official_stagea_score_row(bad_record, arrays, artifact_sha256=spec.artifact_sha256,
                                      binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                      expected_anchor_column=15)
        with self.assertRaisesRegex(FitSelectionError, "bound Stage-A plan anchor"):
            official_stagea_score_row(record, arrays, artifact_sha256=spec.artifact_sha256,
                                      binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                      expected_anchor_column=14)
        bad_anchor = {**record, "score": {**record["score"]}}
        bad_anchor["score"]["metrics_by_support"] = {"full": {"anchor_column": 14, "zero_lag_column": 15}}
        with self.assertRaisesRegex(FitSelectionError, "anchor/zero-lag"):
            official_stagea_score_row(bad_anchor, arrays, artifact_sha256=spec.artifact_sha256,
                                      binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                      expected_anchor_column=15)
        with self.assertRaisesRegex(FitSelectionError, r"\[N,31\]"):
            wrong = dict(arrays)
            wrong["distance_by_query_lag"] = np.ones((31, 27), dtype="<f4")
            official_stagea_score_row(record, wrong, artifact_sha256=spec.artifact_sha256,
                                      binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                      expected_anchor_column=15)

    def test_static_portrait_rows_require_explicit_kind_and_real_face_pin(self) -> None:
        spec = _spec("rms_fixed")
        video_row = _row("sp00_clip0", MODELS[0], spec)
        static_row = replace(video_row, source_media_kind="static_portrait", source_video_sha256=None)
        self.assertEqual(static_row.source_media_kind, "static_portrait")
        self.assertIsNone(static_row.source_video_sha256)
        self.assertRegex(static_row.source_face_sha256, r"^[0-9a-f]{64}$")

        with self.assertRaisesRegex(FitSelectionError, "real source video"):
            replace(video_row, source_video_sha256=None)
        with self.assertRaisesRegex(FitSelectionError, "must not claim a source video"):
            replace(static_row, source_video_sha256=_digest("invented-static-video"))
        with self.assertRaisesRegex(FitSelectionError, "source face SHA-256"):
            replace(static_row, source_face_sha256=None)

    def test_official_stagea_adapter_preserves_static_portrait_media_type(self) -> None:
        spec = _spec("rms_fixed")
        natural_sha = _digest("static-natural-pcm")
        face_sha = _digest("qualified-static-portrait")
        arrays = {
            "query_frame_index": np.arange(3, dtype="<i4"),
            "query_timestamp_num": np.arange(3, dtype="<i8"),
            "query_timestamp_den": np.full(3, 25, dtype="<i8"),
            "track_frame_index": np.arange(3, dtype="<i4"),
            "valid_query_mask": np.ones(3, dtype=np.bool_),
            "distance_by_query_lag": np.ones((3, 31), dtype="<f4"),
            "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
        }
        record = {
            "schema_version": "cross_tfg_audio_head_stagea_v1", "status": "PASS", "failure_reason": None,
            "row_id": "static-rms", "split": "fit", "sample_id": "sp00_clip0", "speaker_id": "sp00",
            "model_id": MODELS[0], "face_id": "primary_face_static_portrait_v1", "seed": 20261003,
            "method_id": "rms_fixed", "source_media_kind": "static_portrait",
            "natural_pcm_sha256": natural_sha, "score_audio_sha256": natural_sha,
            "source_video_sha256": None, "source_face_sha256": face_sha,
            "score": {
                "raw_matrix_npz": {
                    "path": "/qualified/static-rms.npz", "file_sha256": _digest("static-npz"),
                    "arrays": {
                        name: {"dtype": np.asarray(value).dtype.str, "shape": list(np.asarray(value).shape),
                               "array_sha256": _sha256(np.ascontiguousarray(value).tobytes(order="C"))}
                        for name, value in arrays.items()
                    },
                },
                "support_by_support": {"full": {"query_count": 3, "support_mask_sha256": _digest("support")}},
                "metrics_by_support": {"full": {"anchor_column": 15, "zero_lag_column": 15,
                                                   "status": "COMPLETE"}},
            },
        }
        adapted = official_stagea_score_row(record, arrays, artifact_sha256=spec.artifact_sha256,
                                            binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                            expected_anchor_column=15)
        self.assertEqual(adapted.source_media_kind, "static_portrait")
        self.assertIsNone(adapted.source_video_sha256)
        self.assertEqual(adapted.source_face_sha256, face_sha)
        self.assertEqual(adapted.query_keys, ((0, 0, 1, 0), (1, 1, 25, 1), (2, 2, 25, 2)))

        fake_video = {**record, "source_video_sha256": _digest("invented-static-video")}
        with self.assertRaisesRegex(FitSelectionError, "must not claim a source video"):
            official_stagea_score_row(fake_video, arrays, artifact_sha256=spec.artifact_sha256,
                                      binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                      expected_anchor_column=15)
        missing_face = {**record, "source_face_sha256": None}
        with self.assertRaisesRegex(FitSelectionError, "source face SHA-256"):
            official_stagea_score_row(missing_face, arrays, artifact_sha256=spec.artifact_sha256,
                                      binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                      expected_anchor_column=15)
        implicit_static = {key: value for key, value in record.items() if key != "source_media_kind"}
        with self.assertRaisesRegex(FitSelectionError, "real source video"):
            official_stagea_score_row(implicit_static, arrays, artifact_sha256=spec.artifact_sha256,
                                      binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA,
                                      expected_anchor_column=15)

    def test_static_media_kind_must_match_across_all_methods_and_models(self) -> None:
        refs = (_spec("identity"), _spec("rms_fixed"))
        candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
        specs = (*refs, *candidates)
        rows = _rows(specs)
        all_static = [replace(row, source_media_kind="static_portrait", source_video_sha256=None)
                      for row in rows]
        valid = select_snapshot_family(
            all_static, roster=ROSTER, reference_specs=refs, candidates=candidates,
            expected_input_pins_sha256=PINS_SHA, metric_backend="numpy_diagnostic",
        )
        self.assertEqual(valid.status, "NUMPY_DIAGNOSTIC_ONLY_NOT_ADMITTED")
        self.assertTrue(valid.accounting["diagnostic_only"])
        self.assertEqual(valid.accounting["observed_row_records"], len(specs) * len(SAMPLES) * len(MODELS))
        self.assertEqual(valid.accounting["common_query_keys_total"], len(SAMPLES) * len(MODELS) * len(KEYS))

        mixed = list(all_static)
        mixed[0] = replace(mixed[0], source_media_kind="video", source_video_sha256=_digest("video"))
        invalid = select_snapshot_family(
            mixed, roster=ROSTER, reference_specs=refs, candidates=candidates,
            expected_input_pins_sha256=PINS_SHA, metric_backend="numpy_diagnostic",
        )
        self.assertEqual(invalid.status, "CONTROL_FAILED_MISSING_FIT_EVIDENCE")
        self.assertTrue(any("source_media_kind differs" in error for error in invalid.errors))

    def test_source_identity_must_match_when_one_entire_model_uses_video_kind(self) -> None:
        refs = (_spec("identity"), _spec("rms_fixed"))
        candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
        specs = (*refs, *candidates)
        rows = [replace(row, source_media_kind="static_portrait", source_video_sha256=None)
                for row in _rows(specs)]
        changed_model = MODELS[1]
        rows = [
            replace(row, source_media_kind="video", source_video_sha256=_digest("video:shared-input"))
            if row.model_id == changed_model else row
            for row in rows
        ]

        outcome = select_snapshot_family(
            rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
            expected_input_pins_sha256=PINS_SHA, metric_backend="numpy_diagnostic",
        )
        self.assertEqual(outcome.status, "CONTROL_FAILED_MISSING_FIT_EVIDENCE")
        self.assertTrue(any("shared source identity differs across models/methods" in error
                            for error in outcome.errors))

    def test_natural_pcm_identity_must_match_across_models(self) -> None:
        refs = (_spec("identity"), _spec("rms_fixed"))
        candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
        specs = (*refs, *candidates)
        changed_model = MODELS[2]
        rows = [
            replace(row, natural_pcm_sha256=_digest("wrong-pcm"))
            if row.model_id == changed_model else row
            for row in _rows(specs)
        ]

        outcome = select_snapshot_family(
            rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
            expected_input_pins_sha256=PINS_SHA, metric_backend="numpy_diagnostic",
        )
        self.assertEqual(outcome.status, "CONTROL_FAILED_MISSING_FIT_EVIDENCE")
        self.assertTrue(any("shared source identity differs across models/methods" in error
                            for error in outcome.errors))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "optional PyTorch CPU parity check; torch is not installed here")
class TorchMetricConformanceTests(unittest.TestCase):
    def test_numpy_stagea_metrics_match_torch_cpu_within_explicit_tolerance(self) -> None:
        import torch

        rng = np.random.default_rng(20261010)
        matrix = rng.uniform(0.15, 1.25, size=(127, 31)).astype("<f4")
        raw_keys = tuple((i, i, 25, i + 3) for i in range(len(matrix)))
        row = FitScoreRow(
            row_id="torch-conformance", split="fit", sample_id="sp00_clip0", speaker_id="sp00",
            model_id="wav2lip", face_id="face:sp00_clip0", seed=20261003, method_id="identity",
            status="PASS", natural_pcm_sha256=_digest("torch:pcm"),
            source_video_sha256=_digest("torch:video"), source_face_sha256=_digest("torch:face"),
            artifact_sha256=_digest("torch:artifact"), binding_sha256=_digest("torch:binding"),
            input_pins_sha256=PINS_SHA, query_keys=raw_keys, valid_query_mask=(True,) * len(raw_keys),
            distance_by_query_lag=matrix,
        )
        keys = row.query_keys

        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            # Mirrors Stage-A _metric_summary: transpose contiguous [N,31]
            # to [31,N], then perform its float32 torch.mean over queries.
            torch_curve = torch.mean(torch.from_numpy(np.ascontiguousarray(matrix.T)), dim=1)
            torch_b = torch.median(torch_curve)
            torch_dmin = torch.min(torch_curve, dim=0).values
            torch_d0 = torch_curve[15]
            expected = np.asarray([
                torch_b.item(), torch_dmin.item(), torch_d0.item(),
                (torch_b - torch_dmin).item(), (torch_b - torch_curve[15]).item(),
            ], dtype=np.float64)
            canonical = _metric_on_keys(row, keys, metric_backend="torch_cpu")
            canonical_values = np.asarray([
                canonical.b_median, canonical.dmin, canonical.d0, canonical.cbest, canonical.canchor,
            ], dtype=np.float64)
            np.testing.assert_array_equal(canonical_values, expected)

            diagnostic = _metric_on_keys(row, keys, metric_backend="numpy_diagnostic")
            observed = np.asarray([
                diagnostic.b_median, diagnostic.dmin, diagnostic.d0, diagnostic.cbest, diagnostic.canchor,
            ], dtype=np.float64)
            np.testing.assert_allclose(
                observed, expected, atol=METRIC_PARITY_ATOL, rtol=METRIC_PARITY_RTOL,
                err_msg="NumPy CPU and official Stage-A PyTorch CPU summaries exceed the conformance tolerance",
            )
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
