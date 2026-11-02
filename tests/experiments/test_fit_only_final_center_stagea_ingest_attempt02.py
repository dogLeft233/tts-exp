from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit_only_final_center_stagea_ingest_attempt02 as ingest,
    fit_only_spsa_selection_attempt06 as selector,
    fit_only_stagea_ingest_attempt03 as initial_ingest,
    prepare_fit40_spsa_stagea_packets_attempt05 as packet_builder,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _seal(value, field: str) -> dict:
    out = dict(value)
    out[field] = _sha(_canonical(out))
    return out


def _roster() -> selector.FitRoster:
    return selector.FitRoster(
        {f"speaker{speaker:02d}_clip{clip}": f"speaker{speaker:02d}"
         for speaker in range(20) for clip in range(2)},
        ("wav2lip", "musetalk", "ditto"),
    )


class FinalCenterIngestAttempt02Tests(unittest.TestCase):
    def test_exact_predeclared_point_rosters(self):
        proposals = [
            {"point_id": f"p{i}", "method_id": f"p{i}", "point_kind": "spsa_proposal"}
            for i in range(40)
        ]
        centers = [
            {"point_id": f"fit40_d0_spsa_center_k{step:02d}",
             "method_id": f"fit40_d0_spsa_center_k{step:02d}",
             "point_kind": "spsa_center", "source_step": step}
            for step in ingest.CENTER_STEPS
        ]
        proposal_map, center_map = ingest._point_index_maps(
            {"proposal_points": proposals, "center_points": centers})
        self.assertEqual(len(proposal_map), 40)
        self.assertEqual(tuple(sorted(center_map)), (5, 10, 15, 20))
        with self.assertRaises(ingest.FinalCenterIngestError):
            ingest._point_index_maps({"proposal_points": proposals[:-1], "center_points": centers})

    def test_persisted_snapshot_lineage_for_real_proposal_and_center_shapes(self):
        controller_input_pins = {
            "exact": {"path": "/pinned/input.json", "bytes": 1, "sha256": _sha(b"x")}
        }
        input_sha = selector.canonical_input_pins_sha256(controller_input_pins)
        artifact_sha = _sha(b"persisted step120 snapshot artifact")
        seed_id = "fit40_native_transition_head_v3_attempt11_step120"
        seed_theta = tuple(float(i) / 10 for i in range(9))
        lineage = {
            "source_seed_method_id": seed_id,
            "source_snapshot_step": 120,
            "source_theta": list(seed_theta),
            "source_theta_sha256": _sha(_canonical(list(seed_theta))),
            "source_snapshot_artifact_sha256": artifact_sha,
            "source_snapshot_record": {"path": "/pinned/step120.json"},
            "source_snapshot_index": {"path": "/pinned/index.json"},
            "source_snapshot_record_raw_sha256": _sha(b"record"),
            "source_snapshot_index_raw_sha256": _sha(b"index"),
        }
        seed_method = {"method_id": seed_id,
                       "config": {"training_family": "fit40_native_transition_head_v3_attempt11"}}
        refs = (
            selector.MethodSpec("identity", _sha(b"identity"), _sha(b"identity binding")),
            selector.MethodSpec("rms_fixed", _sha(b"rms"), _sha(b"rms binding")),
        )
        controller = selector.SPSAController(
            initial_theta=seed_theta,
            input_pins=controller_input_pins,
            candidate_artifact_sha256=artifact_sha,
            roster=_roster(), reference_specs=refs, metric_backend="torch_cpu")
        pair = controller.ask()
        proposal = packet_builder._method_for_proposal(
            pair.plus, seed_method, input_sha, lineage)
        proposal_spec = ingest._point_method_spec(
            proposal, expected_kind="fit40_spsa_proposal", expected_id=pair.plus.method_id,
            input_sha=input_sha, expected_seed_id=seed_id, expected_seed_step=120,
            expected_seed_theta=seed_theta, expected_seed_artifact=artifact_sha)
        self.assertEqual(proposal_spec.method_id, pair.plus.method_id)
        self.assertEqual(proposal_spec.theta, pair.plus.theta)
        center_id = "fit40_d0_spsa_center_k05"
        center_theta = tuple(seed_theta[i] + 0.125 for i in range(9))
        center_binding = selector.full_fit_candidate_binding_sha256(
            method_id=center_id, artifact_sha256=artifact_sha,
            input_pins_sha256=input_sha, source_step=5, theta=center_theta,
            point_kind="spsa_center")
        center_spec = selector.SnapshotSpec(
            center_id, artifact_sha, center_binding, 5, center_theta, "spsa_center")
        center = packet_builder._method_for_center(center_spec, seed_method, input_sha, lineage)
        self.assertEqual(
            ingest._point_method_spec(
                center, expected_kind="fit40_spsa_center", expected_id=center_id,
                input_sha=input_sha, expected_seed_id=seed_id, expected_seed_step=120,
                expected_seed_theta=seed_theta, expected_seed_artifact=artifact_sha),
            center_spec,
        )
        tampered = dict(center)
        tampered_config = dict(center["config"])
        tampered_config["source_snapshot_step"] = 90
        tampered["config"] = tampered_config
        tampered["config_sha256"] = _sha(_canonical(tampered_config))
        with self.assertRaises(ingest.FinalCenterIngestError):
            ingest._point_method_spec(
                tampered, expected_kind="fit40_spsa_center", expected_id=center_id,
                input_sha=input_sha, expected_seed_id=seed_id, expected_seed_step=120,
                expected_seed_theta=seed_theta, expected_seed_artifact=artifact_sha)

    def test_execution_package_must_resolve_to_raw_verified_members(self):
        model_ids = ("wav2lip", "musetalk", "ditto")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {}

            def add(name: str, role: str) -> dict:
                path = root / f"{name}.bin"
                raw = f"fixture:{name}".encode("utf-8")
                path.write_bytes(raw)
                pin = initial_ingest._PinnedFile(
                    name, role, str(path), str(path), len(raw), _sha(raw))
                files[name] = pin
                return {"path": str(path), "bytes": len(raw), "sha256": _sha(raw)}

            family_raw = b"composed family"
            family_path = root / "family.json"
            family_path.write_bytes(family_raw)
            family_pin = initial_ingest._PinnedFile(
                "family", "composed_family_plan", str(family_path), str(family_path),
                len(family_raw), _sha(family_raw))
            family_self = _sha(b"family payload self")
            source_map = {}
            runtimes = {}
            for model in model_ids:
                runtime = add(f"{model}_runtime", "runtime_binding")
                runtimes[model] = {"path": runtime["path"], "bytes": runtime["bytes"],
                                   "raw_sha256": runtime["sha256"]}
                source_map[model] = {
                    "launcher_pin": add(f"{model}_launcher", "launcher"),
                    "runner_pin": add(f"{model}_runner", "runner"),
                    "adapter_pin": add(f"{model}_adapter", "adapter"),
                    "runtime_binding_pin": runtime,
                    "source_pins": {"module": add(f"{model}_module", "source_module")},
                }
            helper = add("shared_helper", "shared_helper")
            point_runner = add("point_runner", "point_runner")
            family_plan = {
                "models": [{"model_id": model} for model in model_ids],
                "execution": {"runtime_bindings_by_model": runtimes},
            }
            package = _seal({
                "schema": "fit40_spsa_point_execution_package_manifest_v1",
                "status": "FROZEN", "package_id": "fixture-package",
                "composed_family_plan_pin": {
                    "path": family_pin.declared_path, "bytes": family_pin.byte_count,
                    "sha256": family_pin.sha256, "self_sha256": family_self,
                },
                "shared_application_helper_pin": helper,
                "point_runner_pin": point_runner,
                "execution_sources_by_model": source_map,
            }, "manifest_sha256")
            package_raw = _canonical(package)
            package_path = root / "package.json"
            package_path.write_bytes(package_raw)
            package_pin = initial_ingest._PinnedFile(
                "package", "execution_package_manifest", str(package_path), str(package_path),
                len(package_raw), _sha(package_raw))
            package_file_table = dict(files)
            package_file_table["package"] = package_pin
            checked = ingest._validate_execution_package_manifest(
                package, package_pin, family_plan, family_pin, family_self, package_file_table)
            self.assertEqual(checked["point_runner_pin"], point_runner)
            self.assertEqual(set(checked["execution_sources_by_model"]), set(model_ids))
            bad = json.loads(json.dumps(package))
            bad["execution_sources_by_model"][model_ids[0]]["adapter_pin"]["sha256"] = _sha(b"not adapter")
            bad = _seal({k: v for k, v in bad.items() if k != "manifest_sha256"}, "manifest_sha256")
            with self.assertRaises(ingest.FinalCenterIngestError):
                ingest._validate_execution_package_manifest(
                    bad, package_pin, family_plan, family_pin, family_self, package_file_table)

    def test_actual_family03_per_model_generators_are_separately_bound(self):
        family_path = (Path(__file__).resolve().parents[2] / ".codex" / "scratch"
                       / "fit40_spsa_remote_family_attempt03" / "actual_readback_attempt01"
                       / "fit40_three_model_family_plan_attempt03.json")
        if not family_path.is_file():
            self.skipTest("actual family03 metadata fixture is not present in this checkout")
        family = json.loads(family_path.read_text(encoding="utf-8"))
        model_ids = [model["model_id"] for model in family["models"]]
        source_protocols = family["execution"]["source_model_protocols"]
        self.assertEqual(set(source_protocols), set(model_ids))
        generator_payloads = {
            json.dumps(source_protocols[model_id]["generator"], sort_keys=True)
            for model_id in model_ids
        }
        self.assertGreater(len(generator_payloads), 1)
        source_plans = {
            model_id: ({"methods": family["methods"],
                        "protocol": source_protocols[model_id]}, [])
            for model_id in model_ids
        }
        ingest._verify_composed_methods_and_protocol(family, source_plans)

        bad_score_protocol = json.loads(json.dumps(family))
        bad_score_protocol["protocol"]["syncnet"]["metric_atol"] += 1e-6
        with self.assertRaises(ingest.FinalCenterIngestError):
            ingest._verify_composed_methods_and_protocol(bad_score_protocol, source_plans)

        bad_generator_binding = json.loads(json.dumps(family))
        bad_generator_binding["protocol"]["generators_by_model"][model_ids[0]]["model_id"] = "wrong-model"
        with self.assertRaises(ingest.FinalCenterIngestError):
            ingest._verify_composed_methods_and_protocol(bad_generator_binding, source_plans)

    def test_utc_contract_rejects_local_or_malformed_time(self):
        parsed = ingest._validate_utc_timestamp("2026-10-09T00:00:00.000000Z", "fixture")
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)
        for bad in ("2026-10-09T00:00:00+08:00", "not-a-time", 1):
            with self.subTest(value=bad), self.assertRaises(ingest.FinalCenterIngestError):
                ingest._validate_utc_timestamp(bad, "fixture")

    def test_partial_final_family_cannot_reach_selector(self):
        proposals = [
            {"point_id": f"p{i}", "method_id": f"p{i}", "point_kind": "spsa_proposal"}
            for i in range(39)
        ]
        centers = [
            {"point_id": f"fit40_d0_spsa_center_k{step:02d}",
             "method_id": f"fit40_d0_spsa_center_k{step:02d}",
             "point_kind": "spsa_center", "source_step": step}
            for step in ingest.CENTER_STEPS
        ]
        index = _seal({
            "schema": ingest.SCHEMA,
            "status": ingest.STATUS_READY,
            "metric_backend": "torch_cpu", "selection_status": "NOT_SELECTED",
            "expected_initial_row_count": 720, "expected_final_row_count": 720,
            "expected_proposal_pair_count": 20, "expected_proposal_point_count": 40,
            "expected_center_count": 4, "expected_center_row_count": 480,
            "expected_proposal_row_count": 960,
            "files": [{"file_id": "unused", "role": "unused", "declared_path": "/unused",
                       "local_path": "/unused", "bytes": 1, "sha256": _sha(b"x")}],
            "proposal_points": proposals, "center_points": centers,
        }, "index_sha256")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "partial.json"
            path.write_bytes(json.dumps(index).encode("utf-8"))
            with mock.patch.object(ingest.selector, "select_snapshot_family",
                                   side_effect=AssertionError("selector must remain gated")):
                with self.assertRaises(ingest.FinalCenterIngestError):
                    ingest.ingest_final_center_family(path)


if __name__ == "__main__":
    unittest.main()
