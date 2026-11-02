from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Optional

import numpy as np


SOURCE = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
          "cross_tfg_audio_head_20261003" / "fit40_spsa_point_execution_attempt04.py")
SPEC = importlib.util.spec_from_file_location("fit40_spsa_point_execution_attempt04_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load point execution runner")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canon(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _write_packet(path: Path, value: dict, field: str) -> dict:
    packet = dict(value)
    packet[field] = _sha(_canon(packet))
    raw = (json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": _sha(raw),
            "self_sha256": packet[field]}


def _write_file(path: Path, raw: bytes = b"pinned code fixture\n") -> dict:
    path.write_bytes(raw)
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": _sha(raw)}


def _fixture(root: Path, *, point_kind: str = "spsa_proposal"):
    models = list(RUNNER.MODEL_ORDER)
    input_sha = "a" * 64
    methods = ["identity", "rms_fixed", *sorted(RUNNER.SNAPSHOT_METHOD_IDS)]
    family_rows = []
    identity_by_context = {}
    for sample_index in range(40):
        sample = "S%04d" % (sample_index + 1)
        speaker = "SPK%02d" % (sample_index // 2 + 1)
        for model in models:
            pcm_sha = _sha((sample + model + "pcm").encode())
            natural = {
                "path": str((root / (sample + "_" + model + ".wav")).resolve()),
                "file_sha256": _sha((sample + model + "wav").encode()),
                "pcm_sha256": pcm_sha, "sample_count": 4, "sample_rate_hz": 16000,
                "channels": 1, "subtype": "IEEE_FLOAT_32_LE",
            }
            identity = {
                "row_id": "identity_%s_%s" % (sample, model), "split": "fit",
                "sample_id": sample, "speaker_id": speaker, "model_id": model,
                "face_id": "face_%s" % sample, "seed": 20261003,
                "method_id": "identity", "source_media_kind": "video",
                "natural_pcm": natural, "score_audio": natural,
                "source_face": {"file_sha256": _sha((sample + "face").encode())},
                "source_video": {"file_sha256": _sha((sample + "video").encode())},
                "face_box": [1, 2, 3, 4],
            }
            identity_by_context[(sample, model)] = identity
            family_rows.append(identity)
            for method in methods[1:]:
                extra = dict(identity)
                extra["row_id"] = "_%s_%s_%s" % (sample, model, method)
                extra["method_id"] = method
                family_rows.append(extra)
    family = {
        "schema_version": RUNNER.FAMILY_SCHEMA, "status": "FROZEN", "run_id": "family40",
        "models": [{"model_id": model} for model in models],
        "methods": [{"method_id": method} for method in methods],
        "rows": family_rows,
        "scientific_input_pins_sha256": input_sha,
        "execution": {"planned_rows": 720, "model_ids": models,
                      "candidate_selection": "not performed", "dev_or_confirm_access": False,
                      "runtime_bindings_by_model": {}},
    }
    for model in models:
        runtime_path = root / (model + "_runtime.json")
        runtime_pin = _write_file(runtime_path, b"runtime-" + model.encode())
        family["execution"]["runtime_bindings_by_model"][model] = {
            "path": runtime_pin["path"], "bytes": runtime_pin["bytes"],
            "raw_sha256": runtime_pin["sha256"],
        }
    family_path = root / "family.json"
    family_pin = _write_packet(family_path, family, "plan_sha256")
    family["plan_sha256"] = family_pin["self_sha256"]

    if point_kind == "spsa_proposal":
        samples = ["S%04d" % i for i in range(1, 9)]
        method_id = "fit40_spsa_proposal_fixture"
        config = {"method": "fit40_spsa_proposal", "proposal_id": "proposal-plus-01",
                  "pair_id": "pair-01", "iteration": 0, "side": "plus",
                  "binding_sha256": "b" * 64, "input_pins_sha256": input_sha}
        scope = {"scope_kind": "spsa_pair_proposal", "run_id": "point-proposal-plus",
                 "iteration": 0, "pair_id": "pair-01", "proposal_id": "proposal-plus-01",
                 "side": "plus", "method_id": method_id, "input_pins_sha256": input_sha}
    else:
        samples = ["S%04d" % i for i in range(1, 41)]
        method_id = "fit40_d0_spsa_center_k05"
        config = {"method": "fit40_spsa_center", "method_id": method_id,
                  "point_kind": "spsa_center", "source_step": 5,
                  "binding_sha256": "c" * 64, "input_pins_sha256": input_sha}
        scope = {"scope_kind": "spsa_full_center", "run_id": "point-center-05",
                 "iteration": 5, "source_step": 5, "method_id": method_id,
                 "input_pins_sha256": input_sha}
    method = {"method_id": method_id, "config": config, "config_sha256": _sha(_canon(config))}
    point_rows = []
    for sample in samples:
        for model in models:
            row = dict(identity_by_context[(sample, model)])
            row["row_id"] = "point_%s_%s_%s" % (sample, model, method_id)
            row["method_id"] = method_id
            row["method_config_sha256"] = method["config_sha256"]
            row["spsa_point_binding_sha256"] = config["binding_sha256"]
            point_rows.append(row)
    point = {
        "schema_version": RUNNER.FAMILY_SCHEMA, "status": "FROZEN",
        "run_id": scope["run_id"], "models": family["models"],
        "methods": [method], "rows": point_rows, "generation_repeat_qc": [],
        "scientific_input_pins_sha256": input_sha,
        "execution": {"planned_rows": len(point_rows),
                      "spsa_execution_parent_plan": {k: family_pin[k] for k in ("path", "bytes", "sha256")},
                      "spsa_execution_scope": scope},
    }
    point_path = root / "point.json"
    point_pin = _write_packet(point_path, point, "plan_sha256")

    ingest = {"schema": RUNNER.INITIAL_INGEST_SCHEMA,
              "models": [{"model_id": model} for model in models]}
    ingest_pin = _write_packet(root / "initial_ingest_index.json", ingest, "index_sha256")
    selection_status = "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV"
    selection = {
        "status": selection_status,
        "selected_method_id": "fit40_native_transition_head_v3_attempt11_step120",
        "refinement_seed_method_id": None, "metric_backend": "torch_cpu", "errors": [],
        "accounting": {"unique_official_score_rows": 720, "observed_unique_rows": 720,
                       "failed_score_rows": 0, "missing_score_rows": 0, "unexpected_score_rows": 0},
    }
    receipt = {
        "schema": RUNNER.INITIAL_RECEIPT_SCHEMA, "status": "INITIAL_720_SELECTOR_COMPLETE",
        "selection_status": selection_status, "selection": selection,
        "metric_backend": "torch_cpu", "initial_ingest_index": ingest_pin,
        "initial_selector_index_sha256": ingest_pin["sha256"],
        "initial_selector_index_self_sha256": ingest_pin["self_sha256"],
        "spsa_seed_method_id": selection["selected_method_id"],
        "scientific_input_pins_sha256": input_sha, "composed_family_plan": family_pin,
    }
    selector_pin = _write_packet(root / "initial_selector_receipt.json", receipt, "receipt_sha256")

    model_sources = {}
    source_file = _write_file(root / "stagea_core.py", b"core fixture\n")
    helper_file = _write_file(root / "shared_helper.py", b"helper fixture\n")
    launcher_file = _write_file(root / "launcher.py", b"launcher fixture\n")
    worker_file = _write_file(root / "worker.py", b"worker fixture\n")
    adapter_source = Path(__file__).resolve()
    adapter_raw = adapter_source.read_bytes()
    adapter_file = {"path": str(adapter_source), "bytes": len(adapter_raw), "sha256": _sha(adapter_raw)}
    source_raw = SOURCE.read_bytes()
    point_runner_file = {"path": str(SOURCE.resolve()), "bytes": len(source_raw),
                         "sha256": _sha(source_raw)}
    runtime_pin = {"path": family["execution"]["runtime_bindings_by_model"][models[0]]["path"],
                   "bytes": family["execution"]["runtime_bindings_by_model"][models[0]]["bytes"],
                   "sha256": family["execution"]["runtime_bindings_by_model"][models[0]]["raw_sha256"]}
    for model in models:
        model_sources[model] = {
            "launcher_pin": launcher_file, "runner_pin": worker_file,
            "adapter_pin": adapter_file,
            "runtime_binding_pin": {
                "path": family["execution"]["runtime_bindings_by_model"][model]["path"],
                "bytes": family["execution"]["runtime_bindings_by_model"][model]["bytes"],
                "sha256": family["execution"]["runtime_bindings_by_model"][model]["raw_sha256"],
            },
            "source_pins": {"stagea_core": source_file, "adapter_code": adapter_file,
                            "shared_helper": helper_file},
        }
    package = {
        "schema": RUNNER.EXECUTION_PACKAGE_SCHEMA, "status": "FROZEN", "package_id": "pkg01",
        "composed_family_plan_pin": family_pin,
        "shared_application_helper_pin": helper_file,
        "point_runner_pin": point_runner_file,
        "execution_sources_by_model": model_sources,
    }
    package_pin = _write_packet(root / "execution_package.json", package, "manifest_sha256")
    return family, family_pin, point, point_pin, selector_pin, package_pin, model_sources[models[0]]


def _arrays_and_npz(root: Path, row_id: str):
    arrays = {
        "query_frame_index": np.arange(3, dtype="<i4"),
        "query_timestamp_num": np.asarray([0, 1, 2], dtype="<i8"),
        "query_timestamp_den": np.asarray([1, 5, 5], dtype="<i8"),
        "track_frame_index": np.arange(3, dtype="<i4"),
        "valid_query_mask": np.ones(3, dtype=bool),
        "distance_by_query_lag": np.arange(93, dtype="<f4").reshape(3, 31),
        "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
    }
    meta = {}
    for key, value in arrays.items():
        meta[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
                     "array_sha256": _sha(np.ascontiguousarray(value).tobytes())}
    path = root / (row_id + ".npz")
    np.savez_compressed(path, **arrays)
    raw = path.read_bytes()
    ref = {"path": str(path.resolve()), "file_sha256": _sha(raw), "arrays": meta}
    return arrays, ref


class _SubsetError(RuntimeError):
    def __init__(self, completed, failed_row_id):
        super().__init__("synthetic render failure")
        self.completed = completed
        self.failed_row_id = failed_row_id
        self.unattempted_row_ids = []
        self.failed_result = {"failure_reason": "synthetic fixture failure"}


class _FakeAdapter:
    def __init__(self, root: Path, source_pin: dict, adapter_pin: dict, *, fail_at: Optional[int] = None):
        self.root = root
        self.source_pin = source_pin
        self.adapter_pin = adapter_pin
        self.fail_at = fail_at
        self.calls = 0

    def validation_source_pins(self):
        return {"stagea_core": self.source_pin,
                "adapter_code": {**self.adapter_pin,
                                 "declared_bytes": self.adapter_pin["bytes"],
                                 "self_sha256": "ignored-extension-field"}}

    def _render(self, candidate_rows):
        out = []
        for index, planned in enumerate(candidate_rows):
            if self.fail_at == index:
                raise _SubsetError(out, planned["row_id"])
            arrays, npz_ref = _arrays_and_npz(self.root, planned["row_id"])
            record = {
                "schema_version": "cross_tfg_audio_head_stagea_v1", "status": "PASS",
                "row_id": planned["row_id"], "model_id": planned["model_id"],
                "method_id": planned["method_id"],
                "score": {"raw_matrix_npz": npz_ref},
            }
            out.append({"row": record, "official_arrays": arrays})
        return out

    def render_subset(self, *, candidate_rows, method, output_root):
        return self._render(candidate_rows)


class _FakeW2LAdapter(_FakeAdapter):
    def render_subset(self, *, candidate_rows, output_root):
        return self._render(candidate_rows)


class PointRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.family, self.family_pin, self.point, self.point_pin,
         self.selector_pin, self.package_pin, self.model_package) = _fixture(self.root)
        self.source_pin = self.model_package["source_pins"]["stagea_core"]
        RUNNER._proc_start_ticks = lambda pid: 123456

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *, fail_at=None, name="output"):
        adapter = _FakeW2LAdapter(self.root, self.source_pin,
                               self.model_package["adapter_pin"], fail_at=fail_at)
        output = self.root / name
        result = RUNNER.execute_point_block(
            adapter_factory=lambda context: adapter,
            family_plan_pin=self.family_pin, point_plan_pin=self.point_pin,
            initial_selector_receipt_pin=self.selector_pin,
            execution_package_manifest_pin=self.package_pin,
            model_id=RUNNER.MODEL_ORDER[0], output_root=output,
        )
        return result, adapter, output

    def test_gate_fails_before_adapter_creation_or_output_on_incomplete_720(self):
        _path, receipt_raw = Path(self.selector_pin["path"]), None
        receipt = json.loads(_path.read_text())
        receipt["selection"]["accounting"]["missing_score_rows"] = 1
        resealed = dict(receipt)
        resealed["receipt_sha256"] = _sha(_canon({k: v for k, v in receipt.items() if k != "receipt_sha256"}))
        receipt_raw = (json.dumps(resealed, sort_keys=True, indent=2) + "\n").encode()
        bad_path = self.root / "initial_selector_bad.json"
        bad_path.write_bytes(receipt_raw)
        bad_pin = {"path": str(bad_path), "bytes": len(receipt_raw), "sha256": _sha(receipt_raw),
                   "self_sha256": resealed["receipt_sha256"]}
        called = []
        output = self.root / "must_not_exist"
        with self.assertRaises(RUNNER.PointExecutionError):
            RUNNER.execute_point_block(
                adapter_factory=lambda context: called.append(context),
                family_plan_pin=self.family_pin, point_plan_pin=self.point_pin,
                initial_selector_receipt_pin=bad_pin,
                execution_package_manifest_pin=self.package_pin,
                model_id=RUNNER.MODEL_ORDER[0], output_root=output)
        self.assertEqual(called, [])
        self.assertFalse(output.exists())

    def test_success_seals_exact_ordered_eight_row_block_and_actual_parent_receipt(self):
        result, _adapter, output = self._run()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["exit_code"], 0)
        rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual([row["row_id"] for row in rows], result["run_status"]["row_ids"])
        self.assertEqual(len(rows), 8)
        manifest = json.loads((output / "result_manifest.json").read_text())
        self.assertEqual(manifest["raw_score_npz_by_row"][rows[0]["row_id"]]["raw_matrix_npz"]["array_refs"].keys(),
                         RUNNER.ARRAY_NAMES)
        self.assertEqual(manifest["status_counts"], {"PASS": 8, "FAILED": 0, "INCOMPLETE": 0})
        self.assertEqual(result["run_status"]["initial_ingest_index_pin"]["path"].endswith("initial_ingest_index.json"), True)
        self.assertNotEqual(result["run_status"]["initial_ingest_index_pin"]["path"], self.selector_pin["path"])
        self.assertEqual(result["run_status"]["status"], "COMPLETE")
        self.assertEqual(result["run_status"]["planned_rows"], 8)
        self.assertEqual(result["run_status"]["completed_rows"], 8)
        self.assertEqual(result["run_status"]["failed_rows"], 0)
        self.assertEqual(result["run_status"]["incomplete_rows"], 0)
        self.assertEqual(set(manifest["raw_score_npz_by_row"]), {row["row_id"] for row in rows})
        self.assertEqual(manifest["row_ids"], result["run_status"]["row_ids"])
        launcher = self.model_package["launcher_pin"]
        worker = self.model_package["runner_pin"]
        adapter = self.model_package["adapter_pin"]
        runtime = self.model_package["runtime_binding_pin"]
        child_started = datetime.fromisoformat(result["run_status"]["started_utc"].replace("Z", "+00:00"))
        parent_started = (child_started - timedelta(seconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z")
        parent_process = {"child_pid": result["run_status"]["child_pid"],
                          "child_start_ticks": result["run_status"]["child_start_ticks"],
                          "child_exit_code": 0,
                          "started_utc": parent_started,
                          "ended_utc": RUNNER._utc_now()}
        status_raw_before = (output / "run_status.json").read_bytes()
        manifest_raw_before = (output / "result_manifest.json").read_bytes()
        finalized_status = json.loads(status_raw_before)
        self.assertGreaterEqual(finalized_status["started_utc"], parent_process["started_utc"])
        self.assertLessEqual(finalized_status["ended_utc"], parent_process["ended_utc"])
        receipt_pin = RUNNER.write_child_execution_receipt(
            receipt_path=output / "child_execution_receipt.json", context={},
            process=parent_process,
            launcher_pin=launcher, runner_pin=worker, runtime_binding_pin=runtime,
            adapter_pin=adapter,
            source_pins=self.model_package["source_pins"],
            execution_package_manifest_pin=self.package_pin,
            run_status_pin=result["run_status_pin"],
            result_manifest_pin=result["result_manifest_pin"],
            results_jsonl_pin=result["results_jsonl_pin"],
        )
        self.assertEqual((output / "run_status.json").read_bytes(), status_raw_before)
        self.assertEqual((output / "result_manifest.json").read_bytes(), manifest_raw_before)
        receipt = json.loads(Path(receipt_pin["path"]).read_text())
        self.assertEqual(receipt["schema"], RUNNER.CHILD_RECEIPT_SCHEMA)
        self.assertEqual(receipt["child_exit_code"], 0)
        self.assertEqual(receipt["block_index"], 0)
        self.assertEqual(receipt["execution_package_manifest_pin"], self.package_pin)
        self.assertEqual(receipt["row_ids"], finalized_status["row_ids"])
        receipt_started = datetime.fromisoformat(receipt["started_utc"].replace("Z", "+00:00"))
        child_ended = datetime.fromisoformat(finalized_status["ended_utc"].replace("Z", "+00:00"))
        receipt_ended = datetime.fromisoformat(receipt["ended_utc"].replace("Z", "+00:00"))
        self.assertLessEqual(receipt_started, child_started)
        self.assertGreaterEqual(receipt_ended, child_ended)

    def test_failure_seals_full_denominator_with_failed_and_incomplete_rows(self):
        result, _adapter, output = self._run(fail_at=2, name="failed_output")
        self.assertEqual(result["status"], "TERMINAL_WITH_FAILURES")
        self.assertEqual(result["exit_code"], 2)
        rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual([row["status"] for row in rows],
                         ["PASS", "PASS", "FAILED"] + ["INCOMPLETE"] * 5)
        manifest = json.loads((output / "result_manifest.json").read_text())
        self.assertEqual(manifest["status_counts"], {"PASS": 2, "FAILED": 1, "INCOMPLETE": 5})
        self.assertEqual(set(manifest["raw_score_npz_by_row"]), {rows[0]["row_id"], rows[1]["row_id"]})
        self.assertTrue(all(item["raw_matrix_npz"] is None for item in manifest["row_results"][2:]))

    def test_center_block_preserves_exact_forty_sample_denominator(self):
        center_root = self.root / "center_fixture"
        center_root.mkdir()
        (family, family_pin, point, point_pin, selector_pin,
         package_pin, model_package) = _fixture(center_root, point_kind="spsa_center")
        source_pin = model_package["source_pins"]["stagea_core"]
        adapter = _FakeW2LAdapter(center_root, source_pin, model_package["adapter_pin"])
        output = center_root / "center_output"
        result = RUNNER.execute_point_block(
            adapter_factory=lambda _context: adapter,
            family_plan_pin=family_pin, point_plan_pin=point_pin,
            initial_selector_receipt_pin=selector_pin,
            execution_package_manifest_pin=package_pin,
            model_id=RUNNER.MODEL_ORDER[0], output_root=output)
        ledger = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(ledger), 40)
        self.assertEqual(result["run_status"]["row_ids"], [row["row_id"] for row in ledger])

    def test_adapter_initialization_failure_still_seals_all_planned_rows(self):
        output = self.root / "adapter_init_failure"
        called = []

        def fail_after_gate(context):
            called.append(context["model_id"])
            raise RuntimeError("synthetic adapter initialization error")

        result = RUNNER.execute_point_block(
            adapter_factory=fail_after_gate,
            family_plan_pin=self.family_pin, point_plan_pin=self.point_pin,
            initial_selector_receipt_pin=self.selector_pin,
            execution_package_manifest_pin=self.package_pin,
            model_id=RUNNER.MODEL_ORDER[0], output_root=output)
        ledger = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual(called, [RUNNER.MODEL_ORDER[0]])
        self.assertEqual(len(ledger), 8)
        self.assertEqual([row["status"] for row in ledger], ["FAILED"] + ["INCOMPLETE"] * 7)
        self.assertEqual(result["run_status"]["status_counts"],
                         {"PASS": 0, "FAILED": 1, "INCOMPLETE": 7})

    def test_not_admitted_initial_selection_accepts_only_exact_fallback_seed(self):
        receipt_path = Path(self.selector_pin["path"])
        receipt = json.loads(receipt_path.read_text())
        seed = receipt["spsa_seed_method_id"]
        receipt["selection_status"] = "NOT_ADMITTED"
        receipt["selection"]["status"] = "NOT_ADMITTED"
        receipt["selection"]["selected_method_id"] = None
        receipt["selection"]["refinement_seed_method_id"] = seed
        receipt["receipt_sha256"] = _sha(_canon({k: v for k, v in receipt.items()
                                                   if k != "receipt_sha256"}))
        fallback_path = self.root / "initial_selector_fallback.json"
        raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
        fallback_path.write_bytes(raw)
        fallback_pin = {"path": str(fallback_path.resolve()), "bytes": len(raw),
                        "sha256": _sha(raw), "self_sha256": receipt["receipt_sha256"]}
        context = RUNNER.validate_point_scope(
            family_plan_pin=self.family_pin, point_plan_pin=self.point_pin,
            initial_selector_receipt_pin=fallback_pin,
            execution_package_manifest_pin=self.package_pin,
            model_id=RUNNER.MODEL_ORDER[0])
        self.assertEqual(context["initial_selector"]["selection_status"], "NOT_ADMITTED")
        receipt["selection"]["selected_method_id"] = seed
        receipt["receipt_sha256"] = _sha(_canon({k: v for k, v in receipt.items()
                                                   if k != "receipt_sha256"}))
        raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
        fallback_path.write_bytes(raw)
        fallback_pin.update(bytes=len(raw), sha256=_sha(raw), self_sha256=receipt["receipt_sha256"])
        with self.assertRaises(RUNNER.PointExecutionError):
            RUNNER.validate_point_scope(
                family_plan_pin=self.family_pin, point_plan_pin=self.point_pin,
                initial_selector_receipt_pin=fallback_pin,
                execution_package_manifest_pin=self.package_pin,
                model_id=RUNNER.MODEL_ORDER[0])

    def test_adapter_origin_is_derived_even_when_getter_is_absent(self):
        class AdapterWithoutGetter:
            pass

        context = RUNNER.validate_point_scope(
            family_plan_pin=self.family_pin, point_plan_pin=self.point_pin,
            initial_selector_receipt_pin=self.selector_pin,
            execution_package_manifest_pin=self.package_pin,
            model_id=RUNNER.MODEL_ORDER[0])
        pins = RUNNER._source_pin_table(AdapterWithoutGetter(), context)
        self.assertEqual(pins["adapter_implementation"], self.model_package["adapter_pin"])
        self.assertEqual(pins["family_plan"]["sha256"], self.family_pin["sha256"])
        self.assertEqual(pins["point_plan"]["sha256"], self.point_pin["sha256"])

    def test_adapter_origin_rejects_wrong_loaded_module_hash(self):
        import types

        wrong_source = self.root / "wrong_adapter.py"
        wrong_pin = _write_file(wrong_source, b"wrong module source\n")
        module_name = "fit40_wrong_adapter_origin_fixture"
        module = types.ModuleType(module_name)
        module.__file__ = wrong_pin["path"]
        sys.modules[module_name] = module
        WrongOriginAdapter = type("WrongOriginAdapter", (), {"__module__": module_name})
        try:
            context = RUNNER.validate_point_scope(
                family_plan_pin=self.family_pin, point_plan_pin=self.point_pin,
                initial_selector_receipt_pin=self.selector_pin,
                execution_package_manifest_pin=self.package_pin,
                model_id=RUNNER.MODEL_ORDER[0])
            with self.assertRaises(RUNNER.PointExecutionError):
                RUNNER._source_pin_table(WrongOriginAdapter(), context)
        finally:
            sys.modules.pop(module_name, None)

    def test_decoded_query_support_mask_preserves_invalid_and_empty_windows(self):
        arrays = {
            "query_frame_index": np.arange(3, dtype="<i4"),
            "query_timestamp_num": np.asarray([0, 1, 2], dtype="<i8"),
            "query_timestamp_den": np.asarray([1, 5, 5], dtype="<i8"),
            "track_frame_index": np.arange(3, dtype="<i4"),
            "valid_query_mask": np.asarray([True, False, True], dtype=bool),
            "distance_by_query_lag": np.zeros((3, 31), dtype="<f4"),
            "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
        }
        refs = RUNNER._validate_arrays(arrays)
        self.assertEqual(refs["valid_query_mask"]["shape"], [3])
        self.assertEqual(refs["valid_query_mask"]["array_sha256"],
                         _sha(np.ascontiguousarray(arrays["valid_query_mask"]).tobytes()))
        arrays["valid_query_mask"][:] = False
        empty_support_refs = RUNNER._validate_arrays(arrays)
        self.assertEqual(empty_support_refs["valid_query_mask"]["array_sha256"],
                         _sha(np.ascontiguousarray(arrays["valid_query_mask"]).tobytes()))


if __name__ == "__main__":
    unittest.main()
