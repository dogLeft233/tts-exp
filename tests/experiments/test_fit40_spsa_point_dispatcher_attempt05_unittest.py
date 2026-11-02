"""CPU-only engineering checks for the Fit40 point dispatcher.

Fixtures here are synthetic and exercise package/file-table plumbing only. They
are not Fit scores, model runs, or selector evidence.
"""
from __future__ import annotations

import hashlib
import importlib.util
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
          "cross_tfg_audio_head_20261003" / "fit40_spsa_point_dispatcher_attempt05.py")
SPEC = importlib.util.spec_from_file_location("fit40_dispatcher_attempt05_test_module", SOURCE)
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
    @staticmethod
    def _proposal():
        return SimpleNamespace(
            proposal_id="proposal-synthetic", pair_id="pair-synthetic", iteration=1,
            method_id="method-synthetic", side="plus", theta=(0.0,) * 9,
            theta_sha256="0" * 64, artifact_sha256="1" * 64,
            binding_sha256="2" * 64, input_pins_sha256="3" * 64,
            batch_id="batch-synthetic", sample_ids=tuple(f"S{i:04d}" for i in range(8)),
            direction=(1,) * 9, c_k=0.1,
        )

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
                # Match the frozen family03 schema: Muse includes measured bytes,
                # while W2L and Ditto expose only path/raw_sha256/status. The
                # package sealer still measures and pins all three actual files.
                if index == 1:
                    runtime_pins[model_id] = {
                        "path": runtime_raw["path"], "bytes": runtime_raw["bytes"],
                        "raw_sha256": runtime_raw["sha256"], "status": "FROZEN_FIXTURE",
                    }
                else:
                    runtime_pins[model_id] = {
                        "path": runtime_raw["path"],
                        "raw_sha256": runtime_raw["sha256"], "status": "FROZEN_FIXTURE",
                    }
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

            measured_w2l_bytes = entries[models[0]]["runtime_binding_pin"]["bytes"]
            for invalid_size, suffix in ((True, "bool"), (measured_w2l_bytes + 1, "wrong")):
                bad_family_body = {
                    "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
                    "execution": {"runtime_bindings_by_model": {
                        **runtime_pins,
                        models[0]: {
                            **runtime_pins[models[0]], "bytes": invalid_size,
                        },
                    }},
                }
                bad_family_pin = _write_self(root / f"family_bad_{suffix}.json",
                                             bad_family_body, "plan_sha256")
                bad_package = root / f"bad_size_{suffix}.json"
                with self.assertRaises(dispatcher.DispatcherError):
                    dispatcher.seal_execution_package_manifest(
                        bad_package, package_id=f"fixture-bad-size-{suffix}",
                        family_plan_pin=bad_family_pin, point_runner_pin=common_runner,
                        shared_application_helper_pin=helper,
                        execution_sources_by_model=entries,
                    )
                self.assertFalse(bad_package.exists())

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
            npz = {}
            frame_maps = {}
            for i in range(counts["PASS"]):
                row_id = f"{model_id}:r{i}"
                npz[row_id] = {"path": f"/synthetic/{model_id}/{i}.npz", "bytes": 1,
                               "sha256": "0" * 64, "array_refs": {}}
                map_path = root / f"{model_index}_frame_map_{i}.json"
                map_body = {"schema": "cross-tfg-track-frame-map-v2", "fps_num": 25,
                            "fps_den": 1, "frames": []}
                map_raw = _canonical(map_body)
                map_path.write_bytes(map_raw)
                file_entry = dispatcher._file_entry(
                    map_path, "official_source_frame_map", pin_name=row_id)
                files[file_entry["file_id"]] = file_entry
                frame_maps[row_id] = file_entry["file_id"]
            gathered.append({
                "block": {"model_id": model_id, "terminal_status": model_status,
                          "status_counts": dict(counts),
                          "official_source_frame_map_file_ids": frame_maps},
                "files": files, "records": [], "raw_npz_by_row": npz,
            })
        point = self._proposal()
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
            self.assertEqual(set(terminal["point_entry"]["official_source_frame_map_file_ids"]),
                             set(terminal["raw_npz_by_row"]))
            self.assertEqual(sum(item["role"] == "official_source_frame_map"
                                 for item in terminal["file_entries"]), 24)
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

    def test_official_source_frame_map_is_raw_pinned_and_cross_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "frame_map.json"
            body = {"schema": "cross-tfg-track-frame-map-v2", "fps_num": 25,
                    "fps_den": 1, "frames": [
                        {"crop_frame_index": 0, "source_clock_frame_index": 0,
                         "timestamp_num": 0, "timestamp_den": 1,
                         "render_frame_index": 0, "frame_origin": "decoded_source"},
                    ]}
            raw = _canonical(body)
            path.write_bytes(raw)
            row = {
                "row_id": "candidate-r0",
                "score": {"primary_syncnet_input": {
                    "frame_map_path": str(path.resolve()),
                    "frame_map_sha256": hashlib.sha256(raw).hexdigest(),
                }},
                "render": {"selected_track_crop": {
                    "frame_map_path": str(path.resolve()),
                    "frame_map_sha256": hashlib.sha256(raw).hexdigest(),
                }},
            }
            entries = {}
            binding = dispatcher._add_official_source_frame_map(entries, row)
            self.assertEqual(len(binding), 1)
            file_id = binding["candidate-r0"]
            self.assertEqual(entries[file_id]["role"], "official_source_frame_map")
            self.assertEqual(entries[file_id]["pin_name"], "candidate-r0")
            self.assertEqual(entries[file_id]["sha256"], hashlib.sha256(raw).hexdigest())

            no_render_duplicate_row = json.loads(json.dumps(row))
            no_render_duplicate_row.pop("render")
            self.assertEqual(
                len(dispatcher._add_official_source_frame_map({}, no_render_duplicate_row)), 1)

            null_duplicate_row = json.loads(json.dumps(row))
            null_duplicate_row["render"]["selected_track_crop"] = None
            with self.assertRaises(dispatcher.DispatcherError):
                dispatcher._add_official_source_frame_map({}, null_duplicate_row)

            bad_hash_row = json.loads(json.dumps(row))
            bad_hash_row["score"]["primary_syncnet_input"]["frame_map_sha256"] = "0" * 64
            with self.assertRaises(dispatcher.DispatcherError):
                dispatcher._add_official_source_frame_map({}, bad_hash_row)
            bad_duplicate_row = json.loads(json.dumps(row))
            bad_duplicate_row["render"]["selected_track_crop"]["frame_map_path"] = str(root / "other.json")
            with self.assertRaises(dispatcher.DispatcherError):
                dispatcher._add_official_source_frame_map({}, bad_duplicate_row)

    def test_existing_partial_proposal_is_unknown_and_never_relaunched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            partial = root / "points" / "proposal-synthetic_plus"
            partial.mkdir(parents=True)
            obj = object.__new__(dispatcher.PointBlockDispatcher)
            obj.dispatch_root = root
            obj.point_outcome_factory = lambda **kwargs: SimpleNamespace(**kwargs)
            outcome = obj.readback(self._proposal())
            self.assertEqual(outcome.disposition, "UNKNOWN")
            before = sorted(str(path.relative_to(partial)) for path in partial.rglob("*"))
            with mock.patch.object(dispatcher.subprocess, "Popen") as popen:
                with self.assertRaises(dispatcher.DispatcherError):
                    obj.execute(self._proposal())
                popen.assert_not_called()
            after = sorted(str(path.relative_to(partial)) for path in partial.rglob("*"))
            self.assertEqual(after, before)

    def test_swapped_plan_or_package_intent_is_unknown_and_never_relaunched(self):
        for swapped_field in ("point_plan_pin", "execution_package_manifest_pin"):
            with self.subTest(swapped_field=swapped_field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                proposal = self._proposal()
                plan_body = {
                    "schema_version": "cross-tfg-audio-eval-v1",
                    "run_id": "synthetic-engineering-fixture",
                    "methods": [{"config": {
                        "proposal_id": proposal.proposal_id, "pair_id": proposal.pair_id,
                        "side": proposal.side, "method_id": proposal.method_id,
                    }}],
                    "rows": [],
                }
                plan_pin = _write_self(root / "point_plan.json", plan_body, "plan_sha256")
                rows_pin = _write_bytes(root / "planned_rows.jsonl", b"")
                packet_pin = _write_self(root / "packet.json", {"schema": "fixture"}, "packet_sha256")
                package_pin = _write_self(root / "package.json", {"schema": "fixture"}, "manifest_sha256")
                wrong_plan_pin = _write_self(root / "other_plan.json", {
                    **plan_body, "run_id": "different-fixture",
                }, "plan_sha256")
                wrong_package_pin = _write_self(root / "other_package.json", {
                    "schema": "different-fixture",
                }, "manifest_sha256")
                side_plan = dispatcher._PlanSide(
                    proposal.proposal_id, proposal.side, plan_pin, rows_pin, packet_pin,
                    plan_body, tuple(),
                )
                obj = object.__new__(dispatcher.PointBlockDispatcher)
                obj.dispatch_root = root
                obj.package_pin = package_pin
                obj._side_plans = {(proposal.proposal_id, proposal.side): side_plan}
                obj.point_outcome_factory = lambda **kwargs: SimpleNamespace(**kwargs)
                point_dir = root / "points" / (proposal.proposal_id + "_" + proposal.side)
                point_dir.mkdir(parents=True)
                intent = {
                    "schema": dispatcher.DISPATCH_INTENT_SCHEMA, "status": "DISPATCH_INTENT",
                    "proposal_id": proposal.proposal_id, "pair_id": proposal.pair_id,
                    "iteration": proposal.iteration, "side": proposal.side,
                    "point_plan_pin": wrong_plan_pin if swapped_field == "point_plan_pin" else plan_pin,
                    "planned_rows_pin": {key: rows_pin[key] for key in ("path", "bytes", "sha256")},
                    "packet_index_pin": packet_pin,
                    "execution_package_manifest_pin": (
                        wrong_package_pin if swapped_field == "execution_package_manifest_pin" else package_pin
                    ),
                    "created_utc": "2026-10-09T00:00:00.000000Z",
                }
                _write_self(point_dir / "dispatch_intent.json", intent, "intent_sha256")
                with mock.patch.object(dispatcher.subprocess, "Popen") as popen:
                    outcome = obj.readback(proposal)
                    self.assertEqual(outcome.disposition, "UNKNOWN")
                    with self.assertRaises(dispatcher.DispatcherError):
                        obj.execute(proposal)
                    popen.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "native process-group assertions require POSIX")
    def test_real_popen_numeric_exit_and_timeout_cleanup(self):
        success = subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(7)"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
            start_new_session=True,
        )
        self.assertGreater(dispatcher._proc_start_ticks(success.pid), 0)
        return_code, timed_out = dispatcher._wait_for_native_child(success, success.pid, 5, 0.2)
        self.assertEqual(return_code, 7)
        self.assertFalse(timed_out)

        slow = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
            start_new_session=True,
        )
        pid = slow.pid
        self.assertGreater(dispatcher._proc_start_ticks(pid), 0)
        return_code, timed_out = dispatcher._wait_for_native_child(slow, pid, 0.05, 0.2)
        self.assertTrue(timed_out)
        self.assertNotEqual(return_code, 0)
        with self.assertRaises(ProcessLookupError):
            os.killpg(pid, 0)

    @unittest.skipUnless(os.name == "posix", "native /proc and process-group assertions require POSIX")
    def test_run_block_popen_timeout_preserves_logs_and_terminal_unknown_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            launcher = root / "synthetic_child.py"
            launcher.write_text(
                "import time\nprint('synthetic-child-started', flush=True)\ntime.sleep(30)\n",
                encoding="utf-8",
            )
            runtime_path = root / "runtime.json"
            runtime_pin = _write_self(runtime_path, {"schema": "synthetic-runtime"}, "binding_sha256")
            runtime_raw = {key: runtime_pin[key] for key in ("path", "bytes", "sha256")}
            profile_path = root / "profile.json"
            profile_body = {
                "schema": dispatcher.PROFILE_SCHEMA, "status": "FROZEN",
                "model_id": dispatcher.MODEL_ORDER[0],
                "interpreter": {
                    "invocation_path": sys.executable,
                    "resolved_path": os.path.realpath(sys.executable),
                    "prefix": os.path.realpath(sys.prefix),
                    "python_version": ".".join(str(value) for value in sys.version_info[:3]),
                },
                "runtime_binding_pin": runtime_raw,
                "project_root": str(root), "python_path_entries": [str(root)],
                "environment": {
                    "PATH": os.defpath, "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                "adapter_args": {},
                "timeouts": {"child_seconds": 60, "terminate_grace_seconds": 1.0},
            }
            profile_pin = _write_self(profile_path, profile_body, "profile_sha256")
            profile_raw_pin = {key: profile_pin[key] for key in ("path", "bytes", "sha256")}
            runner_path, adapter_path = root / "runner.py", root / "adapter.py"
            runner_pin = _write_bytes(runner_path, b"runner fixture")
            adapter_pin = _write_bytes(adapter_path, b"adapter fixture")
            dispatcher_obj = object.__new__(dispatcher.PointBlockDispatcher)
            dispatcher_obj.model_entries = {dispatcher.MODEL_ORDER[0]: {
                "launcher_pin": dispatcher.file_pin(launcher),
                "adapter_pin": adapter_pin, "runner_pin": runner_pin,
                "runtime_binding_pin": runtime_raw,
                "source_pins": {"launch_profile": profile_raw_pin},
            }}
            dispatcher_obj.package_pin = _write_bytes(root / "package.json")
            dispatcher_obj.family_pin = _write_bytes(root / "family.json")
            dispatcher_obj.initial_receipt_pin = _write_bytes(root / "initial.json")
            dispatcher_obj.point_runner_pin = _write_bytes(root / "point_runner.py")
            dispatcher_obj.helper_pin = _write_bytes(root / "helper.py")
            dispatcher_obj.family = {"execution": {"source_model_execution": {
                dispatcher.MODEL_ORDER[0]: {"capacity_stage": "synthetic-test-only"},
            }}}
            block_root = root / "block"
            block_root.mkdir()
            side_plan = SimpleNamespace(
                plan_pin=_write_bytes(root / "point_plan.json"),
                plan={"run_id": "synthetic-engineering-fixture"},
                planned_rows_pin=_write_bytes(root / "model_planned_rows.jsonl"),
                packet_index_pin=_write_bytes(root / "packet.json"),
            )
            real_wait = dispatcher._wait_for_native_child

            def short_actual_wait(proc, pid, _timeout, grace):
                return real_wait(proc, pid, 0.05, min(float(grace), 0.2))

            with mock.patch.object(dispatcher, "_wait_for_native_child", side_effect=short_actual_wait):
                with self.assertRaisesRegex(dispatcher.DispatcherError, "exceeded its pinned timeout"):
                    dispatcher_obj._run_block(
                        self._proposal(), side_plan, dispatcher.MODEL_ORDER[0], block_root, {},
                    )
            self.assertIn("synthetic-child-started", (block_root / "stdout.log").read_text(encoding="utf-8"))
            started = json.loads((block_root / "process_started.json").read_text(encoding="utf-8"))
            terminal = json.loads((block_root / "process_terminal.json").read_text(encoding="utf-8"))
            self.assertTrue(terminal["timed_out"])
            self.assertEqual(terminal["process_started_pin"], dispatcher.file_pin(
                block_root / "process_started.json", self_field="started_sha256"))
            self.assertEqual((started["pid"], started["start_ticks"], started["started_utc"]),
                             (terminal["pid"], terminal["start_ticks"], terminal["started_utc"]))
            self.assertNotEqual(terminal["child_exit_code"], 0)
            with self.assertRaises(ProcessLookupError):
                os.killpg(terminal["pid"], 0)

    @unittest.skipUnless(os.name == "posix", "POSIX directory fsync is required")
    def test_directory_fsync_failure_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "receipt.json"
            with mock.patch.object(dispatcher.os, "fsync",
                                   side_effect=[None, OSError(errno.EIO, "injected directory fsync failure")]):
                with self.assertRaises(OSError):
                    dispatcher._write_exclusive(target, b"synthetic durable-write fixture")
            self.assertEqual(target.read_bytes(), b"synthetic durable-write fixture")


if __name__ == "__main__":
    unittest.main()
