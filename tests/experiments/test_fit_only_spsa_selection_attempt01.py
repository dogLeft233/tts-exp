from __future__ import annotations

import numpy as np
import unittest

from scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt01 import (
    FitRoster,
    FitScoreRow,
    FitSelectionError,
    MethodSpec,
    ProposalEvaluation,
    SPSAController,
    SnapshotSpec,
    _sha256,
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
    if kind in {"identity", "rms", "equal"}:
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
        row_id=f"{spec.method_id}:{sample_id}:{model_id}", split="dev", sample_id=sample_id,
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
        default_curve = spec.method_id if spec.method_id in {"identity", "rms"} else "good"
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


def _initial_family(*, mask=None, drop=None, curve_kinds=None):
    refs = (_spec("identity"), _spec("rms"))
    candidates = tuple(_snapshot(f"step{step:03d}", step) for step in (30, 60, 90, 120))
    specs = (*refs, *candidates)
    rows = _rows(specs, curve_kinds=curve_kinds, mask=mask)
    if drop is not None:
        rows = [row for row in rows if (row.method_id, row.sample_id, row.model_id) != drop]
    result = select_snapshot_family(rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
                                    expected_input_pins_sha256=PINS_SHA)
    return refs, candidates, rows, result


class FitSelectorTests(unittest.TestCase):
    def test_snapshot_family_uses_one_six_method_support_and_correct_effect_directions(self) -> None:
        mask = {("step120", "sp00_clip0", "wav2lip"): (26,)}
        _refs, _candidates, _rows_, result = _initial_family(mask=mask)

        self.assertEqual(result.status, "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV")
        self.assertEqual(result.selected_method_id, "step030")  # exact tie breaks by source step
        self.assertEqual(result.accounting["unique_official_score_rows"], 720)
        self.assertEqual(result.accounting["logical_rows_per_candidate_bundle"], 360)
        self.assertEqual(result.accounting["common_query_keys_total"], 40 * 3 * 27 - 1)
        first = result.candidate_decisions[0]
        self.assertAlmostEqual(first.gates["models"]["wav2lip"]["identity"]["Cbest"], 0.2, delta=1e-6)
        self.assertAlmostEqual(first.gates["models"]["wav2lip"]["identity"]["D0"], 0.05, delta=1e-6)
        self.assertAlmostEqual(first.gates["models"]["wav2lip"]["identity"]["Canchor"], 0.05, delta=1e-6)
        self.assertTrue(all(d.feasible for d in result.candidate_decisions))

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
                                        expected_input_pins_sha256=PINS_SHA)

        self.assertEqual(result.status, "CONTROL_FAILED_MISSING_FIT_EVIDENCE")
        self.assertEqual(result.accounting["unique_official_score_rows"], 720)
        self.assertEqual(result.accounting["observed_unique_rows"], 720)
        self.assertEqual(result.accounting["failed_score_rows"], 1)
        self.assertEqual(result.accounting["missing_score_rows"], 0)
        self.assertIsNone(result.refinement_seed_method_id)

    def test_zero_cbest_is_not_feasible_and_only_seeds_refinement(self) -> None:
        candidates = tuple(_snapshot(f"flat{step}", step) for step in (30, 60, 90, 120))
        refs = (_spec("identity"), _spec("rms"))
        rows = _rows((*refs, *candidates), curve_kinds={candidate.method_id: "equal" for candidate in candidates})
        result = select_snapshot_family(rows, roster=ROSTER, reference_specs=refs, candidates=candidates,
                                        expected_input_pins_sha256=PINS_SHA)

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
                                   expected_input_pins_sha256=PINS_SHA, sample_ids=tuple(ROSTER.sample_ids[:8]))


def _effect_fixture(worst_model_delta: float):
    from scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt01 import MethodEffects

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


def _controller() -> SPSAController:
    refs = (_spec("identity"), _spec("rms"))
    return SPSAController(initial_theta=(0.0,) * 9, input_pins=PINS,
                          candidate_artifact_sha256=_digest("frozen-d0-head"), roster=ROSTER,
                          reference_specs=refs)


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
        left, right = _controller(), _controller()
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

    def test_spsa_successful_tell_updates_once_and_requests_full_centers_only_at_frozen_steps(self) -> None:
        controller, replay = _controller(), _controller()
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
        controller = _controller()
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
                                        expected_input_pins_sha256=PINS_SHA)
        self.assertEqual(result.status, "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV")
        self.assertEqual(result.accounting["unique_official_score_rows"], 720)

    def test_spsa_failed_or_mismatched_pair_never_changes_theta_or_retries(self) -> None:
        controller = _controller()
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

    def test_spsa_incomplete_candidate_or_changed_source_pair_returns_no_update(self) -> None:
        controller = _controller()
        pair = controller.ask()
        refs = controller.reference_specs
        bad_key = (pair.plus.method_id, pair.sample_ids[0], MODELS[0])
        plus = _evaluation(pair.plus, refs, ROSTER, curve_kind="good", failed={bad_key})
        minus = _evaluation(pair.minus, refs, ROSTER, curve_kind="good")
        before = controller.theta
        outcome = controller.tell(plus, minus)
        self.assertEqual(outcome.status, "NO_UPDATE")
        self.assertEqual(controller.theta, before)

        controller = _controller()
        pair = controller.ask()
        plus = _evaluation(pair.plus, refs, ROSTER, curve_kind="good")
        wrong_pcm_key = (pair.minus.method_id, pair.sample_ids[0], MODELS[0])
        minus = _evaluation(pair.minus, refs, ROSTER, curve_kind="good",
                            pcm_override={wrong_pcm_key: _digest("different-source-pcm")})
        outcome = controller.tell(plus, minus)
        self.assertEqual(outcome.status, "NO_UPDATE")
        self.assertIn("natural_pcm_sha256 differs", outcome.reason or "")

    def test_official_adapter_accepts_stagea_shape_and_rejects_w2l_proxy_shape(self) -> None:
        spec = _spec("identity")
        record = {
            "schema_version": "cross_tfg_audio_head_stagea_v1", "status": "PASS", "failure_reason": None,
            "row_id": "r1", "split": "dev", "sample_id": "sp00_clip0", "speaker_id": "sp00",
            "model_id": MODELS[0], "face_id": "face:sp00_clip0", "seed": 20261003, "method_id": "identity",
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
        adapted = official_stagea_score_row(record, arrays, artifact_sha256=spec.artifact_sha256,
                                            binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA)
        self.assertEqual(adapted.distance_by_query_lag.shape, (27, 31))
        self.assertEqual(adapted.query_keys[1], (1, 1, 25, 1))
        with self.assertRaisesRegex(FitSelectionError, r"\[N,31\]"):
            wrong = dict(arrays)
            wrong["distance_by_query_lag"] = np.ones((31, 27), dtype="<f4")
            official_stagea_score_row(record, wrong, artifact_sha256=spec.artifact_sha256,
                                      binding_sha256=spec.binding_sha256, input_pins_sha256=PINS_SHA)


if __name__ == "__main__":
    unittest.main()
