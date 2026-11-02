"""Producer-to-center aggregation tests using the frozen point04 writer.

No model, scorer, GPU, or selector runs here. The per-model status/manifest
and successful score rows are emitted by the actual point04 worker with a
small deterministic adapter fixture. The parent receipt is created by
point04.write_child_execution_receipt, the same API used in production.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit40_spsa_center_dispatcher_attempt02 as center,
    fit40_spsa_point_dispatcher_attempt04 as parent_dispatch,
    fit40_spsa_point_execution_attempt04 as point04,
    fit_only_final_center_stagea_ingest_attempt04 as final_ingest,
    fit_only_spsa_selection_attempt06 as selector,
)


MODELS = list(center.MODEL_ORDER)
THETA = (0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _seal_json(path: Path, body: dict, field: str) -> dict:
    value = dict(body)
    value[field] = center._self_hash(value, field)
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return parent_dispatch.file_pin(path, self_field=field)


def _write_jsonl(path: Path, rows) -> dict:
    raw = b"".join(_canonical(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return parent_dispatch.file_pin(path)


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _arrays():
    return {
        "query_frame_index": np.asarray([0, 1], dtype="<i4"),
        "query_timestamp_num": np.asarray([0, 1], dtype="<i8"),
        "query_timestamp_den": np.asarray([25, 25], dtype="<i8"),
        "track_frame_index": np.asarray([2, 3], dtype="<i4"),
        "valid_query_mask": np.asarray([True, False], dtype="|b1"),
        "distance_by_query_lag": np.zeros((2, 31), dtype="<f4"),
        "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
    }


class _FixtureAdapter:
    """Produces fixture records through point04's unmodified success validator."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def render_subset(self, *, candidate_rows, output_root, method=None):
        result = []
        arrays = _arrays()
        array_refs = point04._validate_arrays(arrays)
        for planned in candidate_rows:
            stem = planned["row_id"]
            npz_path = Path(output_root) / (stem + ".npz")
            np.savez(npz_path, **arrays)
            npz_raw = npz_path.read_bytes()
            frame_path = Path(output_root) / (stem + ".frame-map.json")
            frame_raw = _canonical({
                "schema": "cross-tfg-track-frame-map-v2", "fps_num": 25,
                "fps_den": 1, "frames": [{"source_frame": 0}, {"source_frame": 1}],
            }) + b"\n"
            frame_path.write_bytes(frame_raw)
            row = dict(planned)
            row.update({
                "status": "PASS",
                "score": {
                    "raw_matrix_npz": {
                        "path": str(npz_path.resolve()),
                        "file_sha256": hashlib.sha256(npz_raw).hexdigest(),
                        "arrays": array_refs,
                    },
                    "primary_syncnet_input": {
                        "frame_map_path": str(frame_path.resolve()),
                        "frame_map_sha256": hashlib.sha256(frame_raw).hexdigest(),
                    },
                },
            })
            result.append({"row": row, "official_arrays": arrays})
        return result


class CenterPoint04AggregateTests(unittest.TestCase):
    def test_actual_point04_child_shape_aggregates_120_and_transports_to_consumer(self):
        with tempfile.TemporaryDirectory(prefix="fit40-center02-") as tmp:
            root = Path(tmp).resolve()
            source = root / "source"
            source.mkdir()
            center_spec = selector.SnapshotSpec(
                method_id="fit40_d0_spsa_center_k05", artifact_sha256="a" * 64,
                binding_sha256="b" * 64, source_step=5, theta=THETA,
                point_kind="spsa_center",
            )
            samples = ["BAC%04dS%04dW%04d" % (9, 52, index) for index in range(40)]
            rows = []
            for sample in samples:
                for model_id in MODELS:
                    natural = {"path": "/pinned/source_wav/%s.wav" % sample,
                               "file_sha256": "c" * 64, "pcm_sha256": "d" * 64,
                               "sample_count": 62639, "sample_rate_hz": 16000,
                               "channels": 1, "encoding": "FLOAT_32"}
                    rows.append({
                        "row_id": "%s_%s_%s" % (sample, model_id, center_spec.method_id),
                        "split": "fit", "sample_id": sample, "speaker_id": "S0052",
                        "model_id": model_id, "method_id": center_spec.method_id,
                        "natural_pcm": natural, "score_audio": dict(natural),
                    })

            point_method = {
                "method_id": center_spec.method_id,
                "config": {
                    "method": "fit40_spsa_center", "schema": "fit40_spsa_center_transform_v1",
                    "point_kind": "spsa_center", "source_step": 5,
                    "theta": list(THETA), "artifact_sha256": center_spec.artifact_sha256,
                    "binding_sha256": center_spec.binding_sha256,
                },
            }
            plan = {
                "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
                "run_id": "center-run-05", "rows": rows, "methods": [point_method],
                "models": [{"model_id": model_id} for model_id in MODELS],
            }
            scope = {
                "scope_kind": "spsa_full_center", "point_kind": "spsa_center",
                "source_step": 5, "method_id": center_spec.method_id,
                "artifact_sha256": center_spec.artifact_sha256,
                "binding_sha256": center_spec.binding_sha256, "run_id": plan["run_id"],
            }
            plan_pin = _seal_json(source / "point-plan.json", plan, "plan_sha256")
            family_pin = _seal_json(source / "family-plan.json", {
                "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
                "run_id": "family-run", "scientific_input_pins_sha256": "e" * 64,
            }, "plan_sha256")
            ingest_pin = _seal_json(source / "ingest-index.json", {
                "schema": point04.INITIAL_INGEST_SCHEMA, "models": [
                    {"model_id": model_id} for model_id in MODELS
                ],
            }, "index_sha256")
            initial_receipt_pin = _seal_json(source / "initial-receipt.json", {
                "schema": point04.INITIAL_RECEIPT_SCHEMA,
                "initial_ingest_index": ingest_pin,
            }, "receipt_sha256")
            package_pin = _seal_json(source / "execution-package.json", {
                "schema": point04.EXECUTION_PACKAGE_SCHEMA, "status": "FROZEN",
            }, "manifest_sha256")
            package_doc = json.loads((source / "execution-package.json").read_text(encoding="utf-8"))

            point_rows_pin = _write_jsonl(source / "point-rows.jsonl", rows)
            packet_index_pin = _seal_json(source / "packet-index.json", {
                "schema": "fit40_spsa_stagea_packet_index_v1", "status": "FROZEN",
            }, "packet_sha256")
            journal_pin = _seal_json(source / "journal.json", {
                "schema": "fit40_spsa_controller_journal_v1", "status": "COMPLETE_20_ASK_TELL_PAIRS",
            }, "journal_sha256")

            # Build exact code/runtime source pins used by each point04 child.
            adapter_pin = point04._file_pin(Path(__file__))
            point_runner_pin = point04._file_pin(Path(point04.__file__).resolve())
            helper_path = Path(__file__).resolve().parents[2] / "scripts" / "audio_enhancement" / "fit40_snapshot_spsa_application_attempt05.py"
            helper_pin = point04._file_pin(helper_path)
            entries = {}
            for model_id in MODELS:
                model_dir = source / "model-source" / model_id
                model_dir.mkdir(parents=True)
                launcher_path = model_dir / "launcher.py"
                launcher_path.write_text("# fixture launcher\n", encoding="utf-8")
                runner_path = model_dir / "runner.py"
                runner_path.write_text("# fixture runner\n", encoding="utf-8")
                runtime_path = model_dir / "runtime.json"
                runtime_file_pin = _seal_json(runtime_path, {"model_id": model_id}, "binding_sha256")
                runtime_pin = {key: runtime_file_pin[key] for key in ("path", "bytes", "sha256")}
                source_path = model_dir / "source.py"
                source_path.write_text("# fixture source\n", encoding="utf-8")
                source_pins = {"source": point04._file_pin(source_path)}
                entry = {
                    "launcher_pin": point04._file_pin(launcher_path),
                    "runner_pin": point04._file_pin(runner_path),
                    "adapter_pin": adapter_pin,
                    "runtime_binding_pin": runtime_pin,
                    "source_pins": source_pins,
                }
                entries[model_id] = entry
            package_doc.update({
                "composed_family_plan_pin": family_pin,
                "point_runner_pin": point_runner_pin,
                "shared_application_helper_pin": helper_pin,
                "execution_sources_by_model": entries,
            })
            package_pin = _seal_json(source / "execution-package.json", package_doc, "manifest_sha256")
            package_doc = json.loads((source / "execution-package.json").read_text(encoding="utf-8"))

            dispatcher = object.__new__(center.FitCenterDispatcher)
            dispatcher.model_entries = entries
            dispatcher.family_pin = family_pin
            dispatcher.initial_receipt_pin = initial_receipt_pin
            dispatcher.initial_receipt = {"initial_ingest_index": ingest_pin}
            dispatcher.package_pin = package_pin
            dispatcher.point_runner_pin = point_runner_pin
            dispatcher.helper_pin = helper_pin
            dispatcher.center_dispatcher_pin = parent_dispatch.file_pin(Path(center.__file__).resolve())
            dispatcher.runner = point04

            side_plan = SimpleNamespace(
                plan=plan, planned_rows=tuple(rows), plan_pin=plan_pin,
                planned_rows_pin=point_rows_pin, packet_index_pin=packet_index_pin,
            )
            dispatcher._side = side_plan
            dispatcher._packet_index_path = Path(packet_index_pin["path"])

            process_ticks = point04._proc_start_ticks(os.getpid())
            process_started_at = datetime.now(timezone.utc) - timedelta(seconds=2)
            process_started_text = _utc(process_started_at)
            gathered = []
            fake_identity = center._CenterIdentity(center_spec)

            for model_id in MODELS:
                block_root = root / "dispatch" / model_id
                output_root = block_root / "stagea_output"
                block_root.mkdir(parents=True)
                model_rows = [row for row in rows if row["model_id"] == model_id]
                entry = entries[model_id]
                context = {
                    "family_plan_pin": family_pin,
                    "point_plan_pin": plan_pin,
                    "point_kind": "spsa_center",
                    "point_id": center_spec.method_id,
                    "proposal_id": None, "pair_id": None,
                    "iteration": 5, "side": None, "source_step": 5,
                    "initial_selector": {"receipt": initial_receipt_pin},
                    "initial_ingest_index_pin": ingest_pin,
                    "execution_package_manifest_pin": package_pin,
                    "execution_package_manifest": entry,
                    "shared_application_helper_pin": helper_pin,
                    "method": point_method,
                    "point_plan": plan,
                    "planned_model_rows": model_rows,
                }
                with mock.patch.object(point04, "validate_point_scope", return_value=context):
                    result = point04.execute_point_block(
                        adapter_factory=lambda _context, mid=model_id: _FixtureAdapter(mid),
                        family_plan_pin=family_pin, point_plan_pin=plan_pin,
                        initial_selector_receipt_pin=initial_receipt_pin,
                        execution_package_manifest_pin=package_pin,
                        model_id=model_id, output_root=output_root,
                        start_utc=process_started_text,
                    )
                self.assertEqual(result["status"], "COMPLETE")
                self.assertEqual(result["exit_code"], 0)

                request_pin = parent_dispatch._write_self_json(block_root / "block_request.json", {
                    "schema": "fit40_test_parent_block_request_v1", "model_id": model_id,
                    "point_kind": "spsa_center", "source_step": 5,
                }, "request_sha256")
                started_pin = parent_dispatch._write_self_json(block_root / "process_started.json", {
                    "schema": parent_dispatch.PROCESS_START_SCHEMA, "status": "STARTED",
                    "pid": os.getpid(), "start_ticks": process_ticks,
                    "started_utc": process_started_text, "model_id": model_id,
                    "request_pin": request_pin,
                }, "started_sha256")
                ended_text = _utc(datetime.now(timezone.utc) + timedelta(seconds=2))
                parent_dispatch._write_self_json(block_root / "process_terminal.json", {
                    "schema": parent_dispatch.PROCESS_END_SCHEMA, "status": "TERMINAL",
                    "pid": os.getpid(), "start_ticks": process_ticks,
                    "ended_utc": ended_text, "timed_out": False,
                    "child_exit_code": 0, "process_started_pin": started_pin,
                }, "process_terminal_sha256")

                status_path = output_root / "run_status.json"
                manifest_path = output_root / "result_manifest.json"
                ledger_path = output_root / "results.jsonl"
                status_pin = point04._file_pin(status_path, self_field="status_sha256")
                manifest_pin = point04._file_pin(manifest_path, self_field="manifest_sha256")
                ledger_pin = point04._file_pin(ledger_path)
                status = json.loads(status_path.read_text(encoding="utf-8"))
                process_evidence = {
                    "child_pid": os.getpid(), "child_start_ticks": process_ticks,
                    "child_exit_code": 0, "started_utc": process_started_text,
                    "ended_utc": ended_text,
                }
                receipt_path = block_root / "child_execution_receipt.json"
                writer_args = {
                    "receipt_path": receipt_path, "context": context,
                    "process": process_evidence,
                    "launcher_pin": entry["launcher_pin"],
                    "adapter_pin": entry["adapter_pin"],
                    "runner_pin": entry["runner_pin"],
                    "runtime_binding_pin": entry["runtime_binding_pin"],
                    "source_pins": entry["source_pins"],
                    "execution_package_manifest_pin": package_pin,
                    "run_status_pin": status_pin,
                    "result_manifest_pin": manifest_pin,
                    "results_jsonl_pin": ledger_pin,
                }
                if model_id == MODELS[0]:
                    bad_process = dict(process_evidence, child_exit_code=1)
                    with self.assertRaises(point04.PointExecutionError):
                        point04.write_child_execution_receipt(
                            **dict(writer_args, process=bad_process))
                point04.write_child_execution_receipt(**writer_args)
                gathered.append(dispatcher._collect_block(
                    fake_identity, side_plan, model_id, block_root))

            terminal = dispatcher._assemble_center_terminal(
                center_spec, gathered, journal_pin)
            self.assertEqual(terminal["status"], "COMPLETE")
            self.assertEqual(terminal["planned_candidate_rows"], 120)
            self.assertEqual(terminal["candidate_rows"], 120)
            self.assertEqual(len(terminal["raw_npz_by_row"]), 120)
            entry = terminal["point_entry"]
            self.assertEqual([block["model_id"] for block in entry["blocks"]], MODELS)
            self.assertTrue(all(block["row_ids"] and len(block["row_ids"]) == 40
                                for block in entry["blocks"]))
            self.assertEqual(len(entry["official_source_frame_map_file_ids"]), 120)
            self.assertEqual(set(entry["official_source_frame_map_file_ids"]),
                             set(terminal["raw_npz_by_row"]))

            # Assert exact point04 center receipt identity: no fabricated
            # proposal-side fields, exact iteration, and planned_rows_pin.
            for model_id in MODELS:
                child_path = root / "dispatch" / model_id / "child_execution_receipt.json"
                child = json.loads(child_path.read_text(encoding="utf-8"))
                self.assertNotIn("planned_rows", child)
                self.assertEqual(len(child["row_ids"]), 40)
                self.assertEqual(child["iteration"], 5)
                self.assertIsNone(child["side"])
                self.assertIsNone(child["proposal_id"])
                self.assertIsNone(child["pair_id"])

            # The center aggregator rejects authenticated-but-tampered
            # bindings as well as raw NPZ bytes that changed after the actual
            # point04 child receipt was written.
            victim = root / "dispatch" / MODELS[0] / "child_execution_receipt.json"
            original = victim.read_bytes()
            for edit in (
                lambda value: value.update(planned_rows_pin={"path": "/wrong", "bytes": 1,
                                                             "sha256": "0" * 64}),
                lambda value: value.update(iteration=6),
                lambda value: value["row_ids"].__setitem__(0, "wrong-row-id"),
            ):
                changed = json.loads(original.decode("utf-8"))
                edit(changed)
                _seal_json(victim, changed, "receipt_sha256")
                with self.assertRaises(center.CenterDispatchError):
                    dispatcher._collect_block(fake_identity, side_plan, MODELS[0], victim.parent)
                victim.write_bytes(original)

            parent_terminal = root / "dispatch" / MODELS[0] / "process_terminal.json"
            parent_terminal_original = parent_terminal.read_bytes()
            changed_terminal = json.loads(parent_terminal_original.decode("utf-8"))
            changed_terminal["child_exit_code"] = 1
            _seal_json(parent_terminal, changed_terminal, "process_terminal_sha256")
            with self.assertRaises(center.CenterDispatchError):
                dispatcher._collect_block(fake_identity, side_plan, MODELS[0], parent_terminal.parent)
            parent_terminal.write_bytes(parent_terminal_original)

            ledger_records = [json.loads(line) for line in (
                root / "dispatch" / MODELS[0] / "stagea_output" / "results.jsonl"
            ).read_text(encoding="utf-8").splitlines() if line]
            score_row = next(row for row in ledger_records if row.get("status") == "PASS")
            npz_path = Path(score_row["score"]["raw_matrix_npz"]["path"])
            npz_original = npz_path.read_bytes()
            npz_path.write_bytes(npz_original + b"tamper")
            with self.assertRaises(center.CenterDispatchError):
                dispatcher._collect_block(fake_identity, side_plan, MODELS[0], npz_path.parents[2])
            npz_path.write_bytes(npz_original)

            roles_paths = [(item["role"], item["declared_path"])
                           for item in terminal["file_entries"]]
            self.assertEqual(len(roles_paths), len(set(roles_paths)))
            by_id = {item["file_id"]: item for item in terminal["file_entries"]}
            for row_id, file_id in entry["official_source_frame_map_file_ids"].items():
                mapped = by_id[file_id]
                self.assertEqual(mapped["role"], "official_source_frame_map")
                self.assertEqual(mapped["pin_name"], row_id)
            for row_id, npz in terminal["raw_npz_by_row"].items():
                self.assertEqual(npz["bytes"], Path(npz["path"]).stat().st_size)
                self.assertEqual(set(npz["array_refs"]), point04.ARRAY_NAMES)

            # Final-index transport shape consumed by consumer04. The one
            # produced center entry remains exact; other center slots and
            # proposal slots are shape-only siblings for this CPU unit test.
            center_entries = []
            for step in center.CENTER_STEPS:
                current = copy.deepcopy(entry)
                current["source_step"] = step
                current["point_id"] = "fit40_d0_spsa_center_k%02d" % step
                current["method_id"] = current["point_id"]
                if step != 5:
                    current["blocks"] = []
                center_entries.append(current)
            proposal_entries = [
                {"point_id": "proposal-%02d" % i, "method_id": "proposal-%02d" % i,
                 "point_kind": "spsa_proposal"} for i in range(40)
            ]
            index = {
                "schema": final_ingest.SCHEMA, "status": final_ingest.STATUS_READY,
                "metric_backend": "torch_cpu", "selection_status": "NOT_SELECTED",
                "expected_initial_row_count": 720, "expected_final_row_count": 720,
                "expected_proposal_pair_count": 20, "expected_proposal_point_count": 40,
                "expected_center_count": 4, "expected_center_row_count": 480,
                "expected_proposal_row_count": 960,
                "execution_package_manifest_file_id": "package-file-id",
                "proposal_points": proposal_entries, "center_points": center_entries,
                "files": terminal["file_entries"],
            }
            index["index_sha256"] = center._self_hash(index, "index_sha256")
            final_ingest._validate_final_index_header(index)
            _proposal_map, parsed_centers = final_ingest._point_index_maps(index)
            self.assertEqual(parsed_centers[5]["point_entry"] if "point_entry" in parsed_centers[5] else
                             parsed_centers[5]["method_id"], center_spec.method_id)

            # A present contradictory numeric denominator is a hard failure;
            # absence is the producer's actual receipt shape.
            victim = root / "dispatch" / MODELS[0] / "child_execution_receipt.json"
            original = victim.read_bytes()
            changed = json.loads(original.decode("utf-8"))
            changed["planned_rows"] = 39
            _seal_json(victim, changed, "receipt_sha256")
            with self.assertRaises(center.CenterDispatchError):
                dispatcher._collect_block(fake_identity, side_plan, MODELS[0], victim.parent)
            victim.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
