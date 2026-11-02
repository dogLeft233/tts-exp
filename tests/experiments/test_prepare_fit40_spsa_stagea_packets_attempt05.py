from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt06 import (
    FitRoster,
    FitScoreRow,
    MethodSpec,
    ProposalEvaluation,
    SPSAController,
    SnapshotSelection,
    SnapshotSpec,
    canonical_input_pins_sha256,
    full_fit_candidate_binding_sha256,
)
from scripts.experiments.cross_tfg_audio_head_20261003.fit_only_stagea_ingest_attempt03 import (
    FamilyInputs,
    FamilySelection,
)
from scripts.experiments.cross_tfg_audio_head_20261003.prepare_fit40_spsa_stagea_packets_attempt05 import (
    EXPECTED_MODEL_ORDER,
    EXPECTED_SUPPORTS,
    PacketError,
    _method_for_center,
    _resolve_seed_method,
    _seed_snapshot_lineage,
    compose_three_model_family_plan,
    prepare_center_packet,
    prepare_pair_packets,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pin(label: str, *, self_sha256: str | None = None) -> dict[str, object]:
    raw = label.encode("utf-8")
    result: dict[str, object] = {
        "path": f"/frozen/{label}", "bytes": len(raw), "sha256": _sha(raw),
    }
    if self_sha256 is not None:
        result["self_sha256"] = self_sha256
    return result


def _input_pins() -> dict[str, dict[str, object]]:
    pins = {
        "adr_022_fit_native_scoring": _pin("adr022.md"),
        "adr_023_decoded_window_support": _pin("adr023.md"),
        "fit40_common_score_protocol": _pin("score_protocol.json"),
        "fit40_method_configs": _pin("method_configs.json"),
        "fit40_scientific_context": _pin("scientific_context.json"),
        "fit_source_manifest": _pin("fit_source_manifest.json"),
        "fixed_portrait": _pin("portrait.png"),
        "native_geometry_manifest": _pin("geometry.json"),
    }
    for step in (30, 60, 90, 120):
        artifact = _sha(f"snapshot-artifact-{step}".encode())
        pins[f"snapshot_step_{step:03d}"] = _pin(f"snapshot_step_{step:03d}.json",
                                                    self_sha256=artifact)
    pins["snapshot_index"] = _pin("snapshot_index.json")
    return pins


def _method_roster() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for method_id, kind, config in (
        ("identity", "identity", {"method": "identity"}),
        ("rms_fixed", "rms_fixed", {"method": "rms", "target_rms": 0.0376838172675746}),
        *[(f"fit40_native_transition_head_v3_attempt11_step{step:03d}", "fit_snapshot", {
            "method": "precomputed_fit_snapshot_waveform",
            "source_optimizer_step": step,
            "snapshot_sha256": _input_pins()[f"snapshot_step_{step:03d}"]["self_sha256"],
        }) for step in (30, 60, 90, 120)],
    ):
        result.append({
            "method_id": method_id,
            "kind": kind,
            "config": config,
            "config_sha256": _sha(_canonical(config)),
            "checkpoint_sha256": None,
        })
    return result


def _synthetic_plan(model_id: str, model_order_index: int) -> dict[str, object]:
    pins = _input_pins()
    methods = _method_roster()
    model = {
        "model_id": model_id,
        "audio_frontend": f"synthetic_frontend_{model_order_index}",
        "fps_num": 25,
        "fps_den": 1,
        "checkpoint_sha256": _sha(f"weights:{model_id}".encode()),
    }
    model_bindings = {
        EXPECTED_MODEL_ORDER[0]: "w2l_runtime_binding",
        EXPECTED_MODEL_ORDER[1]: "muse_runtime_binding",
        EXPECTED_MODEL_ORDER[2]: "ditto_runtime_binding",
    }
    plan = {
        "schema_version": "cross-tfg-audio-eval-v1",
        "status": "FROZEN",
        "run_id": f"synthetic-fit-{model_id}",
        "plan_sha256": "",
        "scientific_input_pins": pins,
        "scientific_input_pins_sha256": canonical_input_pins_sha256(pins),
        "models": [model],
        "methods": methods,
        "protocol": {
            "syncnet": {
                "anchor_column": 15,
                "zero_lag_column": 15,
                "lag_offsets_frames": list(range(-15, 16)),
            },
            "generator": {"native_model_id": model_id, "renderer_sha256": _sha(model_id.encode())},
        },
        "supports": copy.deepcopy(EXPECTED_SUPPORTS),
        "generation_repeat_qc": [],
        "rows": [],
        "execution": {
            "planned_rows": 240,
            "model_ids": [model_id],
            "primary_score": "native_roi_original_pcm16",
            "scientific_input_pins_sha256": canonical_input_pins_sha256(pins),
            "dev_or_confirm_access": False,
            "candidate_selection": "not performed",
            model_bindings[model_id]: {
                "path": f"/runtime/{model_id}.json",
                "raw_sha256": _sha(f"runtime:{model_id}".encode()),
            },
        },
    }
    speakers = {f"S{index:04d}": f"BAC009{index:04d}" for index in range(20)}
    for speaker_index, (speaker_id, prefix) in enumerate(speakers.items()):
        for clip in range(2):
            sample_id = f"{prefix}W{speaker_index:04d}{clip:02d}"
            natural = {
                "pcm_sha256": _sha(f"pcm:{sample_id}".encode()),
                "sample_count": 80000 + speaker_index * 2 + clip,
                "sample_rate_hz": 16000,
                "channels": 1,
            }
            source_face = {"file_sha256": _sha(b"portrait"), "path": "/source/portrait.png"}
            source_video = {"file_sha256": None, "path": None, "role": "not_applicable_static_portrait"}
            for method in methods:
                row = {
                    "row_id": f"fit_{speaker_id}_{sample_id}_{model_id}_{method['method_id']}",
                    "split": "fit",
                    "sample_id": sample_id,
                    "speaker_id": speaker_id,
                    "model_id": model_id,
                    "face_id": "neutral_01",
                    "seed": 20261003,
                    "method_id": method["method_id"],
                    "method_config_sha256": method["config_sha256"],
                    "checkpoint_sha256": None,
                    "audio_frontend": model["audio_frontend"],
                    "natural_pcm": copy.deepcopy(natural),
                    "score_audio": copy.deepcopy(natural),
                    "source_face": copy.deepcopy(source_face),
                    "source_video": copy.deepcopy(source_video),
                    "source_media_kind": "static_portrait",
                }
                plan["rows"].append(row)
    plan["plan_sha256"] = _sha(_canonical({k: v for k, v in plan.items() if k != "plan_sha256"}))
    return plan


def _write_plan_and_rows(root: Path, plan: dict[str, object], label: str) -> tuple[Path, str, Path, str]:
    plan_path = root / f"{label}.json"
    rows_path = root / f"{label}.planned_rows.jsonl"
    plan_raw = (json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False,
                           allow_nan=False) + "\n").encode("utf-8")
    rows_raw = b"".join(_canonical(row) + b"\n" for row in plan["rows"])
    plan_path.write_bytes(plan_raw)
    rows_path.write_bytes(rows_raw)
    return plan_path, _sha(plan_raw), rows_path, _sha(rows_raw)


def _synthetic_admitted_family(root: Path):
    """Build engineering-only complete-shape inputs; never score evidence."""
    plan_inputs = [_write_plan_and_rows(root, _synthetic_plan(model, index), f"model{index}")
                   for index, model in enumerate(EXPECTED_MODEL_ORDER)]
    composed_path = root / "synthetic_composed_720_plan.json"
    compose_three_model_family_plan(
        model_plan_paths=[entry[0] for entry in plan_inputs],
        expected_plan_raw_sha256=[entry[1] for entry in plan_inputs],
        planned_rows_paths=[entry[2] for entry in plan_inputs],
        expected_planned_rows_raw_sha256=[entry[3] for entry in plan_inputs],
        output_path=composed_path,
    )
    plan = json.loads(composed_path.read_text(encoding="utf-8"))
    pins = plan["scientific_input_pins"]
    roster_map = {
        row["sample_id"]: row["speaker_id"]
        for row in plan["rows"] if row["method_id"] == "identity"
        and row["model_id"] == EXPECTED_MODEL_ORDER[0]
    }
    roster = FitRoster(sample_to_speaker=roster_map, model_ids=tuple(EXPECTED_MODEL_ORDER))
    input_sha = plan["scientific_input_pins_sha256"]
    candidates = []
    for method in plan["methods"][2:]:
        config = method["config"]
        step = int(config["source_optimizer_step"])
        theta = tuple(0.001 * step for _ in range(9))
        artifact = str(config["snapshot_sha256"])
        binding = full_fit_candidate_binding_sha256(
            method_id=method["method_id"], artifact_sha256=artifact,
            input_pins_sha256=input_sha, source_step=step, theta=theta,
            point_kind="snapshot",
        )
        candidates.append(SnapshotSpec(method["method_id"], artifact, binding, step, theta))
    method_by_id = {method["method_id"]: method for method in plan["methods"]}
    reference_specs = tuple(
        MethodSpec(method_id, _sha(_canonical(method_by_id[method_id]["config"])),
                   method_by_id[method_id]["config_sha256"])
        for method_id in ("identity", "rms_fixed")
    )
    family = FamilyInputs(
        rows=(), roster=roster, input_pins=pins, input_pins_sha256=input_sha,
        score_protocol_sha256=_sha(_canonical(plan["protocol"]["syncnet"])),
        reference_specs=reference_specs, candidates=tuple(candidates),
        expected_rows=720, actual_rows=720, missing_row_ids=(),
        producer_failure_rows=(), input_artifact_failures=(), verified_npz_files=0,
        score_npz_bindings=(), source_file_pins=(), integrity_errors=(),
    )
    accounting = {
        "unique_official_score_rows": 720, "observed_unique_rows": 720,
        "failed_score_rows": 0, "missing_score_rows": 0,
    }
    selection = SnapshotSelection(
        status="NOT_ADMITTED", selected_method_id=None,
        refinement_seed_method_id=candidates[-1].method_id,
        support_sha256=_sha(b"synthetic-only support; not observed"),
        accounting=accounting, candidate_decisions=(), metric_backend="torch_cpu",
    )
    family_selection = FamilySelection(
        family=family, selection=selection, status="NOT_ADMITTED",
        spsa_seed_method_id=candidates[-1].method_id, errors=(),
    )
    return plan, composed_path, family_selection


def _synthetic_evaluation(plan, pair, proposal, controller, side: str):
    method_by_id = {method["method_id"]: method for method in plan["methods"]}
    parent_by_context = {
        (row["sample_id"], row["model_id"], row["method_id"]): row
        for row in plan["rows"]
    }
    rows = []
    method_ids = ("identity", "rms_fixed", proposal.method_id)
    for method_id in method_ids:
        for sample_id in pair.sample_ids:
            for model_id in controller.roster.model_ids:
                parent = parent_by_context[(sample_id, model_id, "identity")]
                method_row = parent_by_context.get((sample_id, model_id, method_id))
                if method_id == proposal.method_id:
                    artifact, binding = proposal.artifact_sha256, proposal.binding_sha256
                    row_id = f"synthetic_{side}_{proposal.method_id}_{sample_id}_{model_id}"
                    candidate_factor = 0.84 if side == "plus" else 0.89
                else:
                    reference = method_by_id[method_id]
                    artifact = _sha(_canonical(reference["config"]))
                    binding = reference["config_sha256"]
                    row_id = method_row["row_id"]
                    candidate_factor = 1.0
                curve = np.linspace(0.5, 1.5, 31, dtype=np.float32) * candidate_factor
                matrix = np.stack((curve, curve + np.float32(0.01), curve + np.float32(0.02))).astype("<f4")
                query_keys = tuple((index, index, 25, index) for index in range(3))
                rows.append(FitScoreRow(
                    row_id=row_id, split="fit", sample_id=sample_id,
                    speaker_id=parent["speaker_id"], model_id=model_id,
                    face_id=parent["face_id"], seed=20261003, method_id=method_id,
                    status="PASS", natural_pcm_sha256=parent["natural_pcm"]["pcm_sha256"],
                    source_video_sha256=None,
                    source_face_sha256=parent["source_face"]["file_sha256"],
                    artifact_sha256=artifact, binding_sha256=binding,
                    input_pins_sha256=controller.input_pins_sha256,
                    query_keys=query_keys, valid_query_mask=(True, True, True),
                    distance_by_query_lag=matrix, source_media_kind="static_portrait",
                ))
    return ProposalEvaluation(
        proposal_id=proposal.proposal_id, pair_id=proposal.pair_id, side=side,
        method_id=proposal.method_id, theta=proposal.theta,
        theta_sha256=proposal.theta_sha256, artifact_sha256=proposal.artifact_sha256,
        binding_sha256=proposal.binding_sha256, input_pins_sha256=proposal.input_pins_sha256,
        batch_id=proposal.batch_id, rows=tuple(rows),
    )


def _synthetic_controller(plan, family_selection):
    seed_id = family_selection.spsa_seed_method_id
    seed_method = next(item for item in plan["methods"] if item["method_id"] == seed_id)
    seed = next(item for item in family_selection.family.candidates if item.method_id == seed_id)
    return SPSAController(
        initial_theta=seed.theta,
        input_pins=family_selection.family.input_pins,
        candidate_artifact_sha256=seed_method["config"]["snapshot_sha256"],
        roster=family_selection.family.roster,
        reference_specs=family_selection.family.reference_specs,
        metric_backend="torch_cpu",
    )


class Fit40SPSAPacketSourceTests(unittest.TestCase):
    def test_composes_three_actual_shaped_240_plans_without_claiming_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_inputs = [_write_plan_and_rows(root, _synthetic_plan(model, index), model)
                           for index, model in enumerate(EXPECTED_MODEL_ORDER)]
            result = compose_three_model_family_plan(
                model_plan_paths=[entry[0] for entry in plan_inputs],
                expected_plan_raw_sha256=[entry[1] for entry in plan_inputs],
                planned_rows_paths=[entry[2] for entry in plan_inputs],
                expected_planned_rows_raw_sha256=[entry[3] for entry in plan_inputs],
                output_path=root / "composed_720_plan.json",
            )
            composed = json.loads(Path(result["plan"]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(composed["execution"]["planned_rows"], 720)
            self.assertEqual([item["model_id"] for item in composed["models"]], list(EXPECTED_MODEL_ORDER))
            self.assertEqual(len(composed["rows"]), 720)
            self.assertEqual(len({row["row_id"] for row in composed["rows"]}), 720)
            self.assertFalse(result["execution_started"])
            self.assertFalse(result["model_calls_started"])
            self.assertEqual(set(composed["execution"]["runtime_bindings_by_model"]),
                             set(EXPECTED_MODEL_ORDER))
            self.assertEqual(set(composed["protocol"]["generators_by_model"]),
                             set(EXPECTED_MODEL_ORDER))
            self.assertEqual(composed["protocol"]["generators_by_model"][EXPECTED_MODEL_ORDER[1]]["native_model_id"],
                             EXPECTED_MODEL_ORDER[1])

    def test_rejects_snapshot_method_without_matching_pinned_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plans = [_synthetic_plan(model, index) for index, model in enumerate(EXPECTED_MODEL_ORDER)]
            for changed in plans:
                snapshot = changed["methods"][2]
                snapshot["config"]["snapshot_sha256"] = _sha(b"wrong snapshot")
                snapshot["config_sha256"] = _sha(_canonical(snapshot["config"]))
                for row in changed["rows"]:
                    if row["method_id"] == snapshot["method_id"]:
                        row["method_config_sha256"] = snapshot["config_sha256"]
                changed["plan_sha256"] = _sha(
                    _canonical({k: v for k, v in changed.items() if k != "plan_sha256"}))
            plan_inputs = [_write_plan_and_rows(root, plan, f"model{index}")
                           for index, plan in enumerate(plans)]
            with self.assertRaisesRegex(PacketError, "pinned optimizer-step artifact"):
                compose_three_model_family_plan(
                    model_plan_paths=[entry[0] for entry in plan_inputs],
                    expected_plan_raw_sha256=[entry[1] for entry in plan_inputs],
                    planned_rows_paths=[entry[2] for entry in plan_inputs],
                    expected_planned_rows_raw_sha256=[entry[3] for entry in plan_inputs],
                    output_path=root / "must_not_exist.json",
                )
            self.assertFalse((root / "must_not_exist.json").exists())

    def test_rejects_changed_anchor_or_guard_support(self) -> None:
        for change in ("anchor", "support"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                plans = [_synthetic_plan(model, index) for index, model in enumerate(EXPECTED_MODEL_ORDER)]
                if change == "anchor":
                    plans[0]["protocol"]["syncnet"]["zero_lag_column"] = 14
                else:
                    plans[0]["supports"][1]["minimum_queries"] = 40
                plans[0]["plan_sha256"] = _sha(_canonical({k: v for k, v in plans[0].items()
                                                             if k != "plan_sha256"}))
                inputs = [_write_plan_and_rows(root, plan, f"p{index}")
                          for index, plan in enumerate(plans)]
                with self.assertRaisesRegex(PacketError, "lag anchor|supports"):
                    compose_three_model_family_plan(
                        model_plan_paths=[entry[0] for entry in inputs],
                        expected_plan_raw_sha256=[entry[1] for entry in inputs],
                        planned_rows_paths=[entry[2] for entry in inputs],
                        expected_planned_rows_raw_sha256=[entry[3] for entry in inputs],
                        output_path=root / "must_not_exist.json",
                    )
                self.assertFalse((root / "must_not_exist.json").exists())

    def test_center_preserves_initial_snapshot_step_separately(self) -> None:
        artifact = _sha(b"verified-step-120-artifact")
        theta = (0.1,) * 9
        candidate_id = "fit40_native_transition_head_v3_attempt11_step120"
        candidate = SnapshotSpec(candidate_id, artifact, _sha(b"candidate-binding"), 120, theta)
        pins = {
            "snapshot_step_120": _pin("snapshot_step_120.json", self_sha256=artifact),
            "snapshot_index": _pin("snapshot_index.json"),
        }
        family = SimpleNamespace(candidates=(candidate,), input_pins=pins)
        family_selection = SimpleNamespace(family=family, spsa_seed_method_id=candidate_id)
        seed_method = {
            "method_id": candidate_id,
            "kind": "fit_snapshot",
            "checkpoint_sha256": None,
            "config": {
                "method": "precomputed_fit_snapshot_waveform",
                "source_optimizer_step": 120,
                "snapshot_sha256": artifact,
                "training_family": "fit40_native_transition_head_v3_attempt11",
            },
        }
        seed_id, resolved = _resolve_seed_method(family_selection, {candidate_id: seed_method})
        lineage = _seed_snapshot_lineage(family_selection, resolved)
        input_sha = _sha(b"frozen common inputs")
        center_theta = (0.2,) * 9
        center_method_id = "fit40_d0_spsa_center_k05"
        center = SnapshotSpec(
            center_method_id, artifact,
            full_fit_candidate_binding_sha256(
                method_id=center_method_id,
                artifact_sha256=artifact,
                input_pins_sha256=input_sha,
                source_step=5,
                theta=center_theta,
                point_kind="spsa_center",
            ),
            5, center_theta, "spsa_center",
        )
        method = _method_for_center(center, resolved, input_sha, lineage)
        self.assertEqual(seed_id, candidate_id)
        self.assertEqual(method["config"]["source_step"], 5)
        self.assertEqual(method["config"]["source_snapshot_step"], 120)
        self.assertEqual(method["config"]["source_snapshot_artifact_sha256"], artifact)
        self.assertIsNone(method["checkpoint_sha256"])

    def test_public_pair_and_center_use_controller_objects_without_relaxing_full_gate(self) -> None:
        """Synthetic engineering fixture only; no scientific score/admission claim."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, plan_path, family_selection = _synthetic_admitted_family(root)
            plan_raw = plan_path.read_bytes()
            controller = _synthetic_controller(plan, family_selection)
            pair = controller.ask()
            pair_packet = prepare_pair_packets(
                base_plan_path=plan_path,
                expected_base_plan_sha256=_sha(plan_raw),
                family_selection=family_selection,
                controller=controller,
                pair=pair,
                output_dir=root / "public_pair_packet",
            )
            self.assertEqual(pair_packet["planned_row_total"], 48)
            self.assertFalse(pair_packet["execution_started"])
            self.assertEqual([item["planned_row_count"] for item in pair_packet["plans"]], [24, 24])

            center_controller = _synthetic_controller(plan, family_selection)
            center_result = None
            for _ in range(5):
                pending = center_controller.ask()
                plus = _synthetic_evaluation(plan, pending, pending.plus, center_controller, "plus")
                minus = _synthetic_evaluation(plan, pending, pending.minus, center_controller, "minus")
                center_result = center_controller.tell(plus, minus)
                self.assertEqual(center_result.status, "UPDATED")
            self.assertIsNotNone(center_result)
            self.assertTrue(center_result.full_center_score_due)
            center_spec = center_result.full_center_spec
            self.assertIsInstance(center_spec, SnapshotSpec)
            center_packet = prepare_center_packet(
                base_plan_path=plan_path,
                expected_base_plan_sha256=_sha(plan_raw),
                family_selection=family_selection,
                center=center_spec,
                output_dir=root / "public_center_packet",
            )
            self.assertEqual(center_packet["planned_row_total"], 120)
            self.assertFalse(center_packet["execution_started"])
            self.assertEqual(center_packet["plans"][0]["planned_row_count"], 120)

            wrong_center = replace(center_spec, binding_sha256=_sha(b"wrong center binding"))
            wrong_output = root / "wrong_binding_must_not_write"
            with self.assertRaisesRegex(PacketError, "full-center method, theta, source artifact, or input-pin binding"):
                prepare_center_packet(
                    base_plan_path=plan_path,
                    expected_base_plan_sha256=_sha(plan_raw),
                    family_selection=family_selection,
                    center=wrong_center,
                    output_dir=wrong_output,
                )
            self.assertFalse(wrong_output.exists())


if __name__ == "__main__":
    unittest.main()
