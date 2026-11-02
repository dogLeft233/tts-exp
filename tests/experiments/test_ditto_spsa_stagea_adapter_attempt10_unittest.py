"""CPU-only wiring checks for the Ditto SPSA adapter attempt10.

Plan fixtures are synthetic and never represent an optimizer proposal or score.
The helper-source check reads the real pinned helper07 file but runs no model,
renderer, scorer, or CUDA operation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/ditto_spsa_stagea_adapter_attempt10.py"
SPEC = importlib.util.spec_from_file_location("ditto_spsa_adapter10_under_test", ADAPTER_PATH)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def self_hash(value, field):
    value[field] = ""
    value[field] = hashlib.sha256(canon({key: item for key, item in value.items()
                                         if key != field})).hexdigest()
    return value[field]


def write_plan(path: Path, value: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


def make_family(temp: Path) -> tuple[dict, dict, Path, dict]:
    model_ids = list(ADAPTER.EXPECTED_MODEL_ORDER)
    sample_ids = [f"sample-{index:02d}" for index in range(40)]
    speaker_by_sample = {sample: f"speaker-{index // 2:02d}"
                         for index, sample in enumerate(sample_ids)}
    input_sha = "a" * 64
    artifact_sha = "b" * 64
    record_pin = {"path": "evidence/snapshot_step120.json", "bytes": 20,
                  "sha256": "c" * 64, "self_sha256": artifact_sha}
    index_pin = {"path": "evidence/snapshot_set_index.json", "bytes": 21,
                 "sha256": "d" * 64, "self_sha256": "e" * 64}
    methods = [{"method_id": "identity", "kind": "identity", "config": {"method": "identity"}},
               {"method_id": "rms_fixed", "kind": "rms", "config": {"method": "rms"}}]
    for step in (30, 60, 90, 120):
        method_id = f"fit40_native_transition_head_v3_attempt11_step{step:03d}"
        methods.append({"method_id": method_id, "kind": "fit_snapshot", "checkpoint_sha256": None,
                        "config": {"method": "precomputed_fit_snapshot_waveform",
                                   "source_optimizer_step": step,
                                   "snapshot_sha256": artifact_sha if step == 120 else str(step) * 64,
                                   "training_family": "fit40_native_transition_head_v3_attempt11"}})
    pins = {"source_fixture": {"path": "evidence/source.fixture", "bytes": 1, "sha256": "f" * 64},
            "snapshot_step_120": record_pin, "snapshot_index": index_pin}
    rows = []
    for method in methods:
        for sample in sample_ids:
            for model_id in model_ids:
                natural = {"path": f"/fixture/{sample}.wav", "file_sha256": "1" * 64,
                           "pcm_sha256": "2" * 64, "sample_count": 62639,
                           "sample_rate_hz": 16000, "channels": 1, "subtype": "FLOAT_32"}
                rows.append({
                    "row_id": f"{method['method_id']}:{sample}:{model_id}",
                    "split": "fit", "sample_id": sample, "speaker_id": speaker_by_sample[sample],
                    "model_id": model_id, "face_id": "portrait", "seed": 20261003,
                    "method_id": method["method_id"], "method_config_sha256": "3" * 64,
                    "checkpoint_sha256": None, "source_video": {"path": None, "fps_num": 25, "fps_den": 1},
                    "source_face": {"path": "/fixture/face.png", "file_sha256": "4" * 64},
                    "natural_pcm": natural, "score_audio": copy.deepcopy(natural),
                    "audio_frontend": "test-only", "source_media_kind": "static_portrait",
                    "model_role": "candidate", "historical_exposed": False,
                    "generation_settings": {"fps_num": 25, "fps_den": 1, "seed": 20261003},
                    "wav2lip_native_frame_binding": {"sample_count": 62639,
                        "sample_rate_hz": 16000, "fps_num": 25, "fps_den": 1,
                        "expected_frame_count": 98},
                })
    family = {
        "schema_version": ADAPTER.PLAN_SCHEMA, "status": "FROZEN",
        "run_id": "synthetic-family-test-only", "models": [{"model_id": item} for item in model_ids],
        "methods": methods, "rows": rows, "scientific_input_pins": pins,
        "scientific_input_pins_sha256": input_sha,
        "protocol": {"generators_by_model": {item: {"generator": item} for item in model_ids}},
        "supports": {"scope": "synthetic fixture only"}, "generation_repeat_qc": [],
        "execution": {"planned_rows": 720, "model_ids": model_ids,
                      "dev_or_confirm_access": False, "candidate_selection": "not performed",
                      "scientific_input_pins_sha256": input_sha,
                      "scope": "test-only", "stage": "A"},
    }
    self_hash(family, "plan_sha256")
    family_path = temp / "family_plan.json"
    family_raw = write_plan(family_path, family)
    family_pin = {"path": str(family_path), "bytes": len(family_raw),
                  "sha256": hashlib.sha256(family_raw).hexdigest(),
                  "self_sha256": family["plan_sha256"]}
    return family, family_pin, family_path, {"sample_ids": sample_ids, "speaker_by_sample": speaker_by_sample,
                                              "record_pin": record_pin, "index_pin": index_pin,
                                              "artifact_sha": artifact_sha, "input_sha": input_sha}


def make_point(temp: Path, family: dict, family_pin: dict, facts: dict) -> tuple[dict, dict, dict]:
    samples = facts["sample_ids"][:8]
    method_id = "fit40_spsa_proposal_cpu_test_only"
    theta = [0.0] * 9
    theta_sha = hashlib.sha256(canon(theta)).hexdigest()
    config = {
        "method": "fit40_spsa_proposal", "schema": "fit40_spsa_proposal_transform_v1",
        "method_id": method_id, "proposal_id": "diagnostic-only-no-optimizer",
        "pair_id": "diagnostic-only-pair", "iteration": 0, "side": "plus",
        "batch_id": "diagnostic-only-batch", "c_k": 0.25,
        "theta": theta, "theta_sha256": theta_sha, "direction": [0.0] * 9,
        "artifact_sha256": facts["artifact_sha"],
        "source_snapshot_artifact_sha256": facts["artifact_sha"],
        "source_snapshot_step": 120,
        "source_seed_method_id": "fit40_native_transition_head_v3_attempt11_step120",
        "source_theta": theta, "source_theta_sha256": theta_sha,
        "source_snapshot_record": facts["record_pin"],
        "source_snapshot_index": facts["index_pin"],
        "source_snapshot_record_raw_sha256": facts["record_pin"]["sha256"],
        "source_snapshot_index_raw_sha256": facts["index_pin"]["sha256"],
        "head_family": "fit40_native_transition_head_v3_attempt11",
        "input_pins_sha256": facts["input_sha"], "sample_ids": samples,
    }
    binding_payload = {key: config[key] for key in (
        "method_id", "proposal_id", "pair_id", "iteration", "side", "batch_id",
        "artifact_sha256", "input_pins_sha256", "theta_sha256", "sample_ids")}
    config["binding_sha256"] = hashlib.sha256(canon(binding_payload)).hexdigest()
    method = {"method_id": method_id, "kind": config["method"], "config": config,
              "config_sha256": hashlib.sha256(canon(config)).hexdigest(), "checkpoint_sha256": None}
    scope = {
        "scope_kind": "spsa_pair_proposal", "iteration": 0, "pair_id": config["pair_id"],
        "proposal_id": config["proposal_id"], "side": "plus", "batch_id": config["batch_id"],
        "sample_ids": samples,
        "speaker_ids": list(dict.fromkeys(facts["speaker_by_sample"][sample] for sample in samples)),
        "source_seed_method_id": config["source_seed_method_id"], "method_id": method_id,
        "artifact_sha256": config["artifact_sha256"], "binding_sha256": config["binding_sha256"],
        "input_pins_sha256": config["input_pins_sha256"], "run_id": "diagnostic-point-test-only",
        "c_k": 0.25,
    }
    point = copy.deepcopy(family)
    point["run_id"] = scope["run_id"]
    point["methods"] = [method]
    identities = {(row["sample_id"], row["model_id"]): row for row in family["rows"]
                  if row["method_id"] == "identity"}
    point_rows = []
    for sample in samples:
        for model_id in ADAPTER.EXPECTED_MODEL_ORDER:
            row = copy.deepcopy(identities[(sample, model_id)])
            row.update({"row_id": f"diagnostic:{sample}:{model_id}:{method_id}",
                        "method_id": method_id, "method_config_sha256": method["config_sha256"],
                        "checkpoint_sha256": None,
                        "spsa_point_binding_sha256": config["binding_sha256"],
                        "score_audio": copy.deepcopy(row["natural_pcm"])})
            point_rows.append(row)
    point["rows"] = point_rows
    point["generation_repeat_qc"] = []
    point["execution"]["spsa_execution_parent_plan"] = family_pin
    point["execution"]["spsa_execution_scope"] = scope
    point["execution"]["runtime_readiness"] = "REQUIRED_SEPARATE_SPSA_BINDING_BEFORE_EXECUTION"
    point["execution"]["planned_rows"] = len(point_rows)
    point["analysis"] = {"selection_status": "NOT_SELECTED",
                         "scope": "fit_only_spsa_execution_packet",
                         "spsa_execution_scope": copy.deepcopy(scope)}
    self_hash(point, "plan_sha256")
    point_path = temp / "point_plan.json"
    point_raw = write_plan(point_path, point)
    point_pin = {"path": str(point_path), "bytes": len(point_raw),
                 "sha256": hashlib.sha256(point_raw).hexdigest(),
                 "self_sha256": point["plan_sha256"]}
    return point, point_pin, method


class DerivedPointPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.family, self.family_pin, _, self.facts = make_family(self.root)
        self.point, self.point_pin, self.method = make_point(
            self.root, self.family, self.family_pin, self.facts)

    def tearDown(self):
        self.temp.cleanup()

    def test_point_plan_is_bound_to_full_family_and_exact_ordered_subset(self):
        point, _, _, family, _, _, point_rows = ADAPTER._validate_point_plan(
            self.point, self.point_pin)
        self.assertEqual(len(family["rows"]), 720)
        self.assertEqual(len(point["rows"]), 24)
        self.assertEqual(len(point_rows), 24)
        self.assertEqual(point["methods"], [self.method])

    def test_scope_roster_mismatch_fails_even_with_updated_plan_self_hash(self):
        bad = copy.deepcopy(self.point)
        bad["execution"]["spsa_execution_scope"]["speaker_ids"] = ["wrong"]
        bad["analysis"]["spsa_execution_scope"] = copy.deepcopy(bad["execution"]["spsa_execution_scope"])
        self_hash(bad, "plan_sha256")
        path = self.root / "bad_point_plan.json"
        raw = write_plan(path, bad)
        pin = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
               "self_sha256": bad["plan_sha256"]}
        with self.assertRaises(ADAPTER.DittoSPSAAdapterError):
            ADAPTER._validate_point_plan(bad, pin)

    def test_row_source_context_cannot_be_changed_under_a_valid_point_hash(self):
        bad = copy.deepcopy(self.point)
        bad["rows"][0]["face_id"] = "other-face"
        self_hash(bad, "plan_sha256")
        path = self.root / "bad_row_plan.json"
        raw = write_plan(path, bad)
        pin = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
               "self_sha256": bad["plan_sha256"]}
        with self.assertRaises(ADAPTER.DittoSPSAAdapterError):
            ADAPTER._validate_point_plan(bad, pin)

    def test_adapter_loads_the_exact_shared_helper07_contract(self):
        helper, pin = ADAPTER._load_application_validator(ROOT)
        self.assertEqual(pin["path"], str((ROOT / ADAPTER.APPLICATION_VALIDATOR_REL).resolve()))
        self.assertEqual(pin["bytes"], 66046)
        self.assertEqual(pin["sha256"], "989b28c6239c2770bee6799afa85f0b0f0b48ec39bdd584cf68c035c464f1ec4")
        for name in (
            "apply_snapshot_theta_shared",
            "validate_shared_application_lineage",
            "project_shared_application_to_model_row",
            "validate_model_projection_lineage",
            "validate_snapshot_point",
        ):
            self.assertTrue(callable(getattr(helper, name, None)), name)

    def test_adapter_rejects_a_changed_helper_digest_before_loading_it(self):
        original = ADAPTER.APPLICATION_VALIDATOR_SHA256
        try:
            ADAPTER.APPLICATION_VALIDATOR_SHA256 = "0" * 64
            with self.assertRaisesRegex(ADAPTER.DittoSPSAAdapterError, "differs from its exact source pin"):
                ADAPTER._load_application_validator(ROOT)
        finally:
            ADAPTER.APPLICATION_VALIDATOR_SHA256 = original

    def test_validation_source_pins_include_the_exact_adapter_module(self):
        adapter = object.__new__(ADAPTER.DittoSPSAStageAAdapter)
        adapter.application_validator_pin = {"path": "helper", "bytes": 1, "sha256": "a" * 64}
        adapter.point_plan_pin = {"path": "point", "bytes": 2, "sha256": "b" * 64}
        adapter.family_plan_pin = {"path": "family", "bytes": 3, "sha256": "c" * 64}
        pins = adapter.validation_source_pins()
        actual = pins["ditto_spsa_adapter"]
        raw = ADAPTER_PATH.read_bytes()
        self.assertEqual(actual, {"path": str(ADAPTER_PATH.resolve()),
                                  "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})


if __name__ == "__main__":
    unittest.main()
