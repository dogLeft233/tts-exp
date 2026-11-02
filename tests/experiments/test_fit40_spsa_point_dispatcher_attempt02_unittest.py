"""CPU-only engineering checks for the Fit40 point dispatcher.

Fixtures here are synthetic and exercise package/file-table plumbing only. They
are not Fit scores, model runs, or selector evidence.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
          "cross_tfg_audio_head_20261003" / "fit40_spsa_point_dispatcher_attempt02.py")
SPEC = importlib.util.spec_from_file_location("fit40_dispatcher_attempt02_test_module", SOURCE)
assert SPEC is not None and SPEC.loader is not None
dispatcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dispatcher
SPEC.loader.exec_module(dispatcher)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _write_self(path: Path, body: dict, field: str) -> dict:
    value = dict(body)
    value[field] = hashlib.sha256(_canonical(value)).hexdigest()
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return dispatcher.file_pin(path, self_field=field)


def _write_bytes(path: Path, data: bytes = b"fixture") -> dict:
    path.write_bytes(data)
    return dispatcher.file_pin(path)


class PointDispatcherEngineeringTests(unittest.TestCase):
    def test_execution_package_requires_model_specific_import_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "project"
            project.mkdir()
            common_runner = _write_bytes(root / "point_runner.py")
            helper = _write_bytes(root / "shared_helper.py")
            entries = {}
            runtime_pins = {}
            models = dispatcher.MODEL_ORDER
            for index, model_id in enumerate(models):
                runtime_pin = _write_self(root / f"runtime_{index}.json", {
                    "schema": "fixture-runtime", "model_id": model_id,
                }, "binding_sha256")
                runtime_raw = {key: runtime_pin[key] for key in ("path", "bytes", "sha256")}
                runtime_pins[model_id] = runtime_raw
                profile_body = {
                    "schema": dispatcher.PROFILE_SCHEMA, "status": "FROZEN",
                    "model_id": model_id,
                    "interpreter": {
                        "invocation_path": sys.executable,
                        "resolved_path": os.path.realpath(sys.executable),
                        "prefix": os.path.realpath(sys.prefix),
                        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
                    },
                    "runtime_binding_pin": runtime_raw,
                    "project_root": str(project.resolve()),
                    "python_path_entries": [str(project.resolve())],
                    "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
                    "adapter_args": {},
                    "timeouts": {"child_seconds": 60, "terminate_grace_seconds": 1.0},
                }
                profile_pin = _write_self(root / f"profile_{index}.json", profile_body,
                                          "profile_sha256")
                source_pins = {"launch_profile": profile_pin}
                if model_id == models[1]:
                    source_pins.update({
                        "musetalk_block_overlay": _write_bytes(root / "muse_overlay.py"),
                        "cross_frontend_runner": _write_bytes(root / "muse_cross.py"),
                        "resource_helper": _write_bytes(root / "muse_resource.py"),
                    })
                elif model_id == models[2]:
                    source_pins.update({
                        "ditto_block_overlay": _write_bytes(root / "ditto_overlay.py"),
                        "resource_helper": _write_bytes(root / "ditto_resource.py"),
                    })
                entries[model_id] = {
                    "launcher_pin": _write_bytes(root / f"launcher_{index}.py"),
                    "runner_pin": _write_bytes(root / f"runner_{index}.py"),
                    "adapter_pin": _write_bytes(root / f"adapter_{index}.py"),
                    "runtime_binding_pin": runtime_raw,
                    "source_pins": source_pins,
                }
            family_body = {
                "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
                "execution": {"runtime_bindings_by_model": runtime_pins},
            }
            family_pin = _write_self(root / "family.json", family_body, "plan_sha256")
            valid_package_path = root / "package.json"
            package_pin = dispatcher.seal_execution_package_manifest(
                valid_package_path, package_id="fixture", family_plan_pin=family_pin,
                point_runner_pin=common_runner,
                shared_application_helper_pin=helper,
                execution_sources_by_model=entries,
            )
            self.assertEqual(package_pin["self_sha256"], json.loads(valid_package_path.read_text())["manifest_sha256"])

            missing_role = {key: dict(value) for key, value in entries.items()}
            missing_role[models[1]]["source_pins"] = {
                key: value for key, value in missing_role[models[1]]["source_pins"].items()
                if key != "cross_frontend_runner"
            }
            rejected_path = root / "rejected.json"
            with self.assertRaises(dispatcher.DispatcherError):
                dispatcher.seal_execution_package_manifest(
                    rejected_path, package_id="fixture-bad", family_plan_pin=family_pin,
                    point_runner_pin=common_runner,
                    shared_application_helper_pin=helper,
                    execution_sources_by_model=missing_role,
                )
            self.assertFalse(rejected_path.exists())

    def _synthetic_gathered(self, root: Path, counts_by_model):
        models = dispatcher.MODEL_ORDER
        plan_path = root / "plan.json"
        plan_pin = _write_self(plan_path, {
            "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
            "run_id": "synthetic-engineering-fixture", "rows": [{"row_id": f"r{i}"} for i in range(24)],
        }, "plan_sha256")
        rows_path = root / "planned_rows.jsonl"
        rows_path.write_text("".join(json.dumps({"row_id": f"r{i}"}) + "\n" for i in range(24)), encoding="utf-8")
        rows_pin = dispatcher.file_pin(rows_path)
        index_path = root / "packet.json"
        packet_pin = _write_self(index_path, {"schema": "fixture"}, "packet_sha256")
        shared = {}
        for path, role, self_field in (
            (plan_path, "plan", "plan_sha256"),
            (rows_path, "planned_rows", None),
            (index_path, "proposal_packet_index", "packet_sha256"),
        ):
            dispatcher._add_file(shared, path, role, self_field=self_field)
        gathered = []
        for model_index, model_id in enumerate(models):
            files = dict(shared)
            for suffix, role in (("status", "run_status"), ("manifest", "result_manifest"),
                                 ("ledger", "results_jsonl"), ("child", "child_execution_receipt")):
                path = root / f"{model_index}_{suffix}.json"
                if suffix == "status":
                    pin = _write_self(path, {"fixture": True}, "status_sha256")
                elif suffix == "manifest":
                    pin = _write_self(path, {"fixture": True}, "manifest_sha256")
                elif suffix == "child":
                    pin = _write_self(path, {"fixture": True}, "receipt_sha256")
                else:
                    pin = _write_bytes(path)
                file_entry = dispatcher._file_entry(path, role,
                    self_field={"status": "status_sha256", "manifest": "manifest_sha256",
                                "child": "receipt_sha256"}.get(suffix))
                files[file_entry["file_id"]] = file_entry
            counts = counts_by_model[model_id]
            model_status = ("COMPLETE" if counts == {"PASS": 8, "FAILED": 0, "INCOMPLETE": 0}
                            else "TERMINAL_WITH_FAILURES")
            npz = {f"{model_id}:r{i}": {"path": f"/synthetic/{model_id}/{i}.npz", "bytes": 1,
                                         "sha256": "0" * 64, "array_refs": {}}
                   for i in range(counts["PASS"])}
            gathered.append({
                "block": {"model_id": model_id, "terminal_status": model_status,
                          "status_counts": dict(counts)},
                "files": files, "records": [], "raw_npz_by_row": npz,
            })
        point = SimpleNamespace(
            proposal_id="proposal-synthetic", pair_id="pair-synthetic", iteration=1,
            method_id="method-synthetic", side="plus", theta=(0.0,) * 9,
            theta_sha256="0" * 64, artifact_sha256="1" * 64,
            binding_sha256="2" * 64, input_pins_sha256="3" * 64,
            batch_id="batch-synthetic", sample_ids=(f"S{i:04d}" for i in range(8)),
            direction=(1,) * 9, c_k=0.1,
        )
        side_plan = dispatcher._PlanSide(
            "proposal-synthetic", "plus", plan_pin,
            {key: rows_pin[key] for key in ("path", "bytes", "sha256")}, packet_pin,
            {"run_id": "synthetic-engineering-fixture", "rows": [{"row_id": f"r{i}"} for i in range(24)]},
            tuple({"row_id": f"r{i}"} for i in range(24)),
        )
        obj = object.__new__(dispatcher.PointBlockDispatcher)
        for name, path, self_field in (
            ("family_pin", root / "family2.json", "plan_sha256"),
            ("package_pin", root / "package2.json", "manifest_sha256"),
            ("point_runner_pin", root / "point_runner2.py", None),
            ("helper_pin", root / "helper2.py", None),
        ):
            if self_field:
                _write_self(path, {"fixture": True}, self_field)
                pin = dispatcher.file_pin(path, self_field=self_field)
            else:
                pin = _write_bytes(path)
            setattr(obj, name, pin)
        ingest_path = root / "ingest.json"
        ingest_pin = _write_self(ingest_path, {"fixture": True}, "index_sha256")
        receipt_path = root / "receipt.json"
        receipt_pin = _write_self(receipt_path, {"initial_ingest_index": ingest_pin}, "receipt_sha256")
        obj.initial_receipt_pin = receipt_pin
        obj.initial_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return obj, point, side_plan, gathered

    def test_terminal_file_merge_and_durable_status_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            complete = {model: {"PASS": 8, "FAILED": 0, "INCOMPLETE": 0}
                        for model in dispatcher.MODEL_ORDER}
            obj, point, side_plan, gathered = self._synthetic_gathered(root, complete)
            terminal = obj._assemble_terminal(point, side_plan, gathered)
            obj._verify_terminal_files(terminal)
            self.assertEqual(terminal["status"], "COMPLETE")
            self.assertEqual(terminal["candidate_rows"], 24)
            paths = [(item["role"], item["declared_path"]) for item in terminal["file_entries"]]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual([item["model_id"] for item in terminal["point_entry"]["blocks"]],
                             list(dispatcher.MODEL_ORDER))
            altered = dict(terminal)
            altered["file_entries"] = [dict(item) for item in terminal["file_entries"]]
            altered["file_entries"][0]["file_id"] = "pt-forged"
            with self.assertRaises(dispatcher.DispatcherError):
                obj._verify_terminal_files(altered)

    def test_terminal_failures_preserve_denominator_without_claiming_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            counts = {model: {"PASS": 8, "FAILED": 0, "INCOMPLETE": 0}
                      for model in dispatcher.MODEL_ORDER}
            counts[dispatcher.MODEL_ORDER[0]] = {"PASS": 6, "FAILED": 2, "INCOMPLETE": 0}
            obj, point, side_plan, gathered = self._synthetic_gathered(root, counts)
            terminal = obj._assemble_terminal(point, side_plan, gathered)
            self.assertEqual(terminal["status"], "COMPLETE_WITH_FAILURES")
            self.assertEqual(terminal["planned_candidate_rows"], 24)
            self.assertEqual(terminal["candidate_rows"], 22)

            counts[dispatcher.MODEL_ORDER[1]] = {"PASS": 5, "FAILED": 1, "INCOMPLETE": 2}
            obj, point, side_plan, gathered = self._synthetic_gathered(root, counts)
            terminal = obj._assemble_terminal(point, side_plan, gathered)
            self.assertEqual(terminal["status"], "INCOMPLETE")
            self.assertEqual(terminal["planned_candidate_rows"], 24)
            self.assertEqual(terminal["candidate_rows"], 19)


if __name__ == "__main__":
    unittest.main()
