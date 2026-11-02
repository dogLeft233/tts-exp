from __future__ import annotations

import copy
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit40_durable_spsa_session_attempt02 as durable,
    fit40_spsa_point_execution_attempt04 as point_runner,
    fit_only_final_center_stagea_ingest_attempt03 as final_ingest,
    fit_only_stagea_ingest_attempt03 as initial_ingest,
    prepare_fit40_spsa_stagea_packets_attempt05 as packet_builder,
    fit_only_spsa_selection_attempt06 as selector,
)


_FIXTURE_PATH = Path(__file__).with_name("test_fit_only_stagea_ingest_attempt03.py")
_FIXTURE_SPEC = importlib.util.spec_from_file_location("_fit_ingest_fixture", _FIXTURE_PATH)
_FIXTURE_MODULE = importlib.util.module_from_spec(_FIXTURE_SPEC)
assert _FIXTURE_SPEC and _FIXTURE_SPEC.loader
_FIXTURE_SPEC.loader.exec_module(_FIXTURE_MODULE)

MODELS = packet_builder.EXPECTED_MODEL_ORDER
OLD_MODELS = _FIXTURE_MODULE.MODELS


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _seal(value, field):
    value[field] = _sha(_canonical({key: item for key, item in value.items() if key != field}))
    return value


class DurableSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("canonical initial-family selector requires PyTorch")

    def _make_complete_inputs(self, root: Path):
        """Build a producer-shaped 720 family using the real ingest fixture and composer."""
        index_path, index = _FIXTURE_MODULE.FitStageAIngestTests()._make_packet(root)
        file_by_id = {item["file_id"]: item for item in index["files"]}

        def save(file_id, document, *, field):
            item = file_by_id[file_id]
            raw = _canonical(_seal(document, field))
            local = Path(item["local_path"])
            _write(local, raw)
            item["bytes"] = len(raw)
            item["sha256"] = _sha(raw)
            return raw

        source_plans = []
        source_plan_paths = []
        source_plan_sha = []
        source_rows_paths = []
        source_rows_sha = []
        runtime_pins = {}
        model_rename = dict(zip(OLD_MODELS, MODELS))

        for model_entry in index["models"]:
            old_model = model_entry["model_id"]
            model_id = model_rename[old_model]
            model_entry["model_id"] = model_id
            plan_file = file_by_id[model_entry["plan_file_id"]]
            rows_file = file_by_id[model_entry["planned_rows_file_id"]]
            plan_path = Path(plan_file["local_path"])
            rows_path = Path(rows_file["local_path"])
            plan_file["declared_path"] = str(plan_path.resolve())
            rows_file["declared_path"] = str(rows_path.resolve())
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["models"][0]["model_id"] = model_id
            for row in plan["rows"]:
                row["model_id"] = model_id
                row["score_audio"] = dict(row["natural_pcm"])
            for method in plan["methods"]:
                config = method.get("config", {})
                if method.get("kind") == "fit_snapshot":
                    step = config["source_optimizer_step"]
                    config["snapshot_sha256"] = config["snapshot_sha256"]
                    plan["scientific_input_pins"][f"snapshot_step_{step:03d}"]["self_sha256"] = config["snapshot_sha256"]
            plan["scientific_input_pins_sha256"] = packet_builder.canonical_input_pins_sha256(
                plan["scientific_input_pins"])
            plan["supports"] = copy.deepcopy(packet_builder.EXPECTED_SUPPORTS)
            plan["generation_repeat_qc"] = []

            runtime_path = root / "package" / f"{model_id}_runtime.json"
            runtime_raw = _canonical({"synthetic_runtime": model_id})
            _write(runtime_path, runtime_raw)
            runtime_ref = {"path": str(runtime_path.resolve()), "bytes": len(runtime_raw),
                           "raw_sha256": _sha(runtime_raw)}
            runtime_pins[model_id] = runtime_ref
            plan["protocol"]["generator"] = {
                "model_id": model_id, "runtime_binding": runtime_ref,
                "generator_protocol": "synthetic-fixture-v1",
            }
            runtime_key = {
                MODELS[0]: "w2l_runtime_binding",
                MODELS[1]: "muse_runtime_binding",
                MODELS[2]: "ditto_runtime_binding",
            }[model_id]
            plan["execution"] = {
                "planned_rows": 240,
                "model_ids": [model_id],
                "primary_score": "native_roi_original_pcm16",
                "scientific_input_pins_sha256": plan["scientific_input_pins_sha256"],
                "dev_or_confirm_access": False,
                "candidate_selection": "not performed",
                runtime_key: runtime_ref,
            }
            plan_raw = save(plan_file["file_id"], plan, field="plan_sha256")
            rows_raw = b"".join(_canonical(row) + b"\n" for row in plan["rows"])
            _write(rows_path, rows_raw)
            rows_file["bytes"] = len(rows_raw)
            rows_file["sha256"] = _sha(rows_raw)

            # Rebind the 40×6 terminal ledgers and manifests to the actual model ID.
            for block in model_entry["blocks"]:
                ledger_file = file_by_id[block["results_file_id"]]
                records = [json.loads(line) for line in Path(ledger_file["local_path"]).read_text(
                    encoding="utf-8").splitlines()]
                for record in records:
                    record["model_id"] = model_id
                ledger_raw = b"".join(_canonical(record) + b"\n" for record in records)
                _write(Path(ledger_file["local_path"]), ledger_raw)
                ledger_file["bytes"] = len(ledger_raw)
                ledger_file["sha256"] = _sha(ledger_raw)

                manifest_file = file_by_id[block["result_manifest_file_id"]]
                manifest = json.loads(Path(manifest_file["local_path"]).read_text(encoding="utf-8"))
                for count in manifest["counts_by_model_method"]:
                    count["model_id"] = model_id
                manifest["plan_raw_sha256"] = _sha(plan_raw)
                manifest["plan_sha256"] = plan["plan_sha256"]
                manifest["results_jsonl_sha256"] = _sha(ledger_raw)
                manifest_raw = save(manifest_file["file_id"], manifest, field="manifest_sha256")

                status_file = file_by_id[block["run_status_file_id"]]
                status = json.loads(Path(status_file["local_path"]).read_text(encoding="utf-8"))
                status["plan_path"] = plan_file["declared_path"]
                status["planned_rows_path"] = rows_file["declared_path"]
                status["plan_raw_sha256"] = _sha(plan_raw)
                status["planned_rows_sha256"] = _sha(rows_raw)
                status["result_manifest_sha256"] = _sha(manifest_raw)
                # Run-status producer records are raw-pinned JSON without an embedded self hash.
                status_file_path = Path(status_file["local_path"])
                status_raw = _canonical(status)
                _write(status_file_path, status_raw)
                status_file["bytes"] = len(status_raw)
                status_file["sha256"] = _sha(status_raw)

            source_plans.append((model_id, plan, plan_file, rows_file))
            source_plan_paths.append(plan_path)
            source_plan_sha.append(plan_file["sha256"])
            source_rows_paths.append(Path(rows_file["local_path"]))
            source_rows_sha.append(rows_file["sha256"])

        index["models"].sort(key=lambda entry: MODELS.index(entry["model_id"]))
        index["scientific_input_pins_sha256"] = source_plans[0][1]["scientific_input_pins_sha256"]
        index_raw = (json.dumps(_seal(index, "index_sha256"), sort_keys=True, indent=2,
                                ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        _write(index_path, index_raw)

        composed_path = root / "composed_family_plan.json"
        packet_builder.compose_three_model_family_plan(
            model_plan_paths=source_plan_paths,
            expected_plan_raw_sha256=source_plan_sha,
            planned_rows_paths=source_rows_paths,
            expected_planned_rows_raw_sha256=source_rows_sha,
            output_path=composed_path,
        )
        composed, composed_raw = durable._read_json(composed_path, "fixture composed plan")
        composed_self = final_ingest._hash_object(composed, "plan_sha256", "fixture composed plan")

        package_dir = root / "package"
        pin_names = {
            "shared_application_helper_pin": "shared-helper.py",
            "point_runner_pin": "point-runner.py",
        }
        package = {
            "schema": "fit40_spsa_point_execution_package_manifest_v1",
            "status": "FROZEN",
            "package_id": "synthetic-execution-package-test-only",
            "composed_family_plan_pin": {
                "path": str(composed_path.resolve()), "bytes": len(composed_raw),
                "sha256": _sha(composed_raw), "self_sha256": composed_self,
            },
        }
        for field, name in pin_names.items():
            path = package_dir / name
            _write(path, f"{field}\n".encode())
            raw = path.read_bytes()
            package[field] = {"path": str(path.resolve()), "bytes": len(raw), "sha256": _sha(raw)}
        by_model = {}
        for model_id in MODELS:
            values = {}
            for role in ("launcher_pin", "runner_pin", "adapter_pin", "runtime_binding_pin"):
                if role == "runtime_binding_pin":
                    ref = runtime_pins[model_id]
                    values[role] = {"path": ref["path"], "bytes": ref["bytes"],
                                    "sha256": ref["raw_sha256"]}
                else:
                    path = package_dir / f"{model_id}_{role}.txt"
                    _write(path, f"{model_id}:{role}\n".encode())
                    raw = path.read_bytes()
                    values[role] = {"path": str(path.resolve()), "bytes": len(raw), "sha256": _sha(raw)}
            source_path = package_dir / f"{model_id}_source.py"
            _write(source_path, f"# source pin {model_id}\n".encode())
            source_raw = source_path.read_bytes()
            values["source_pins"] = {"synthetic_source": {
                "path": str(source_path.resolve()), "bytes": len(source_raw), "sha256": _sha(source_raw),
            }}
            by_model[model_id] = values
        package["execution_sources_by_model"] = by_model
        package = _seal(package, "manifest_sha256")
        package_path = root / "execution_package_manifest.json"
        _write(package_path, (json.dumps(package, sort_keys=True, indent=2,
                                         ensure_ascii=False, allow_nan=False) + "\n").encode())
        return index_path, composed_path, package_path

    def _load_identity_records(self, session):
        index, _raw = durable._read_json(session.inputs.initial_index_pin.local_path,
                                         "synthetic initial family index")
        files = initial_ingest._file_table(index)
        records = {}
        for model_entry in index["models"]:
            model_id = model_entry["model_id"]
            for block in model_entry["blocks"]:
                ledger_pin = initial_ingest._required_file(files, block["results_file_id"], "results_jsonl")
                for record in initial_ingest._parse_jsonl(initial_ingest._read_pinned(ledger_pin), "baseline ledger"):
                    if record["method_id"] == "identity":
                        records[(record["sample_id"], model_id)] = record
        if len(records) != 40 * 3:
            raise AssertionError("fixture did not expose all 120 identity ledger rows")
        return files, records

    def _make_proposal_outcome(self, session, proposal, root: Path,
                               identity_records, source_files, *, terminal_status="COMPLETE",
                               coverage_mode=None):
        inputs = session.inputs
        family = inputs.family_selection.family
        method = packet_builder._method_for_proposal(
            proposal, inputs.seed_method, family.input_pins_sha256, inputs.seed_lineage)
        run_id = f"synthetic_{proposal.method_id}"
        scope = {
            "scope_kind": "spsa_pair_proposal", "run_id": run_id,
            "iteration": proposal.iteration, "pair_id": proposal.pair_id,
            "proposal_id": proposal.proposal_id, "side": proposal.side,
            "batch_id": proposal.batch_id, "sample_ids": list(proposal.sample_ids),
            "source_seed_method_id": family and inputs.family_selection.spsa_seed_method_id,
            "method_id": proposal.method_id, "artifact_sha256": proposal.artifact_sha256,
            "binding_sha256": proposal.binding_sha256,
            "input_pins_sha256": proposal.input_pins_sha256,
        }
        base_rows = {(row["sample_id"], row["model_id"], row["method_id"]): row
                     for row in inputs.composed["rows"]}
        base_pin = durable._pin_link(inputs.composed_pin, inputs.composed_self)
        point_plan = packet_builder._execution_plan(
            inputs.composed, base_pin, base_rows, method, proposal.sample_ids, scope)
        point_root = root / "point_outputs" / proposal.method_id
        point_root.mkdir(parents=True, exist_ok=False)
        files = []
        file_prefix = proposal.method_id + "--"

        def add_file(file_id, role, leaf, raw):
            file_id = file_prefix + file_id
            path = point_root / leaf
            _write(path, raw)
            resolved = path.resolve()
            files.append({"file_id": file_id, "role": role,
                          "declared_path": str(resolved), "local_path": str(resolved),
                          "bytes": len(raw), "sha256": _sha(raw)})
            return {"file_id": file_id, "path": str(resolved),
                    "bytes": len(raw), "sha256": _sha(raw)}

        def add_existing(file_id, role, path, *, self_field=None):
            file_id = file_prefix + file_id
            path = Path(path)
            raw = path.read_bytes()
            resolved = path.resolve()
            pin = {"file_id": file_id, "role": role,
                   "declared_path": str(resolved), "local_path": str(resolved),
                   "bytes": len(raw), "sha256": _sha(raw)}
            if self_field is not None:
                packet = json.loads(raw.decode("utf-8"))
                pin["self_sha256"] = packet[self_field]
            files.append(pin)
            return {"file_id": file_id, "path": str(resolved),
                    "bytes": len(raw), "sha256": _sha(raw),
                    **({"self_sha256": pin["self_sha256"]} if self_field else {})}

        def raw_pin(pin, *, with_self=False):
            keys = ("path", "bytes", "sha256", "self_sha256") if with_self else (
                "path", "bytes", "sha256")
            return {key: pin[key] for key in keys}

        plan_raw = (json.dumps(point_plan, sort_keys=True, indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n").encode()
        point_plan_pin = add_file("point-plan", "plan", "plan.json", plan_raw)
        planned_rows = point_plan["rows"]
        planned_raw = b"".join(_canonical(row) + b"\n" for row in planned_rows)
        all_rows_pin = add_file("point-rows", "planned_rows", "planned_rows.jsonl", planned_raw)
        if terminal_status != "COMPLETE":
            failure_blocks = []
            for model_id in MODELS:
                model_rows = [row for row in planned_rows if row["model_id"] == model_id]
                model_rows_raw = b"".join(_canonical(row) + b"\n" for row in model_rows)
                model_rows_pin = add_file(f"{model_id}-rows", "model_planned_rows",
                                          f"{model_id}_planned_rows.jsonl", model_rows_raw)
                failure_blocks.append({"model_id": model_id, "block_index": 0,
                                       "row_ids": [row["row_id"] for row in model_rows],
                                       "planned_rows_file_id": model_rows_pin["file_id"]})
            failed_entry = {
                "point_id": proposal.method_id, "method_id": proposal.method_id,
                "point_kind": "spsa_proposal", "proposal_id": proposal.proposal_id,
                "pair_id": proposal.pair_id, "iteration": proposal.iteration,
                "side": proposal.side, "run_id": run_id,
                "plan_file_id": point_plan_pin["file_id"],
                "planned_rows_file_id": all_rows_pin["file_id"], "blocks": failure_blocks,
            }
            return durable.PointOutcome(
                disposition="TERMINAL", terminal_status=terminal_status,
                point_entry=failed_entry, files=tuple(files), raw_npz_by_row={},
                reason="synthetic terminal renderer failure",
            )
        raw_npz_by_row = {}
        blocks = []
        family_plan_pin = durable._pin_link(inputs.composed_pin, inputs.composed_self)
        initial_index_pin = durable._pin_link(inputs.initial_index_pin, inputs.initial_index_self)
        initial_receipt_pin = durable._pin_link(inputs.initial_receipt_pin, inputs.initial_receipt_self)
        package_pin = dict(inputs.execution_package["package_pin"])
        point_plan_link = {
            **raw_pin(point_plan_pin), "self_sha256": point_plan["plan_sha256"],
        }
        parent_started_utc = "2026-10-09T00:00:00.000000Z"
        child_started_utc = "2026-10-09T00:00:01.000000Z"
        child_ended_utc = "2026-10-09T00:00:03.000000Z"
        parent_ended_utc = "2026-10-09T00:00:04.000000Z"
        for model_id in MODELS:
            model_rows = [row for row in planned_rows if row["model_id"] == model_id]
            model_rows_raw = b"".join(_canonical(row) + b"\n" for row in model_rows)
            model_rows_pin = add_file(f"{model_id}-rows", "model_planned_rows",
                                      f"{model_id}_planned_rows.jsonl", model_rows_raw)
            records = []
            for planned in model_rows:
                sample = planned["sample_id"]
                identity = copy.deepcopy(identity_records[(sample, model_id)])
                identity["row_id"] = planned["row_id"]
                identity["method_id"] = proposal.method_id
                identity["method_config_sha256"] = method["config_sha256"]
                prior_npz = identity["score"]["raw_matrix_npz"]
                prior_pin = next(pin for pin in source_files.values()
                                 if pin.role == "raw_matrix_npz"
                                 and pin.declared_path == prior_npz["path"]
                                 and pin.sha256 == prior_npz["file_sha256"])
                npz_raw = initial_ingest._read_pinned(prior_pin)
                if coverage_mode and planned["method_id"] == proposal.method_id:
                    with np.load(BytesIO(npz_raw), allow_pickle=False) as archive:
                        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
                    mask = arrays["valid_query_mask"]
                    changed = sample == proposal.sample_ids[0] and model_id == MODELS[0]
                    if changed and coverage_mode == "partial":
                        if len(mask) < 3:
                            raise AssertionError("fixture needs at least three raw query rows")
                        mask[0] = False
                        arrays["track_frame_index"][1] += 1000
                        identity["score"]["support_by_support"]["full"]["query_count"] = int(mask.sum())
                    elif changed and coverage_mode == "empty":
                        mask[:] = False
                        identity["score"]["support_by_support"]["full"]["query_count"] = 0
                        full_metrics = identity["score"]["metrics_by_support"]["full"]
                        full_metrics["status"] = "EMPTY"
                        full_metrics["anchor_column"] = None
                        full_metrics["zero_lag_column"] = None
                    else:
                        arrays = None
                    if arrays is not None:
                        buffer = BytesIO()
                        np.savez(buffer, **arrays)
                        npz_raw = buffer.getvalue()
                        identity["score"]["raw_matrix_npz"]["arrays"] = {
                            name: {
                                "dtype": value.dtype.str,
                                "shape": list(value.shape),
                                "array_sha256": _sha(np.ascontiguousarray(value).tobytes(order="C")),
                            }
                            for name, value in arrays.items()
                        }
                npz_pin = add_file(f"npz-{planned['row_id']}", "raw_matrix_npz",
                                   f"npz/{planned['row_id']}.npz", npz_raw)
                prior_npz["path"] = npz_pin["path"]
                prior_npz["file_sha256"] = npz_pin["sha256"]
                raw_npz_by_row[planned["row_id"]] = {
                    "path": npz_pin["path"], "bytes": npz_pin["bytes"],
                    "sha256": npz_pin["sha256"], "array_refs": prior_npz["arrays"],
                }
                records.append(identity)
            ledger_raw = b"".join(_canonical(record) + b"\n" for record in records)
            ledger_pin = add_file(f"{model_id}-results", "results_jsonl",
                                  f"{model_id}_results.jsonl", ledger_raw)
            model_sources = inputs.execution_package["execution_sources_by_model"][model_id]
            source_pins = copy.deepcopy(model_sources["source_pins"])
            adapter_validation_pins = {
                "family_plan": {key: family_plan_pin[key] for key in ("path", "bytes", "sha256")},
                "point_plan": {key: point_plan_pin[key] for key in ("path", "bytes", "sha256")},
                "adapter_implementation": dict(model_sources["adapter_pin"]),
            }
            common = {
                "status": "COMPLETE", "run_id": run_id,
                "point_kind": "spsa_proposal", "point_id": proposal.method_id,
                "method_id": proposal.method_id,
                "proposal_id": proposal.proposal_id, "pair_id": proposal.pair_id,
                "iteration": proposal.iteration, "side": proposal.side, "source_step": None,
                "model_id": model_id, "block_index": 0,
                "family_plan_pin": family_plan_pin,
                "point_plan_pin": point_plan_link,
                "initial_selector_receipt_pin": initial_receipt_pin,
                "initial_ingest_index_pin": initial_index_pin,
                "execution_package_manifest_pin": package_pin,
                "planned_rows_pin": raw_pin(model_rows_pin),
                "row_ids": [row["row_id"] for row in model_rows],
                "planned_rows": len(model_rows), "completed_rows": len(model_rows),
                "failed_rows": 0, "incomplete_rows": 0,
                "status_counts": {"PASS": len(model_rows), "FAILED": 0, "INCOMPLETE": 0},
                "results_jsonl_pin": raw_pin(ledger_pin),
                "source_pins": source_pins,
                "adapter_validation_pins": adapter_validation_pins,
                "started_utc": child_started_utc,
                "terminalized_utc": child_ended_utc, "ended_utc": child_ended_utc,
                "child_pid": 42000 + MODELS.index(model_id),
                "child_start_ticks": 910000 + MODELS.index(model_id),
                "worker_error": None,
            }
            manifest_rows = []
            manifest_npz = {}
            for record in records:
                row_id = record["row_id"]
                npz = raw_npz_by_row[row_id]
                row_item = {
                    "row_id": row_id, "status": "PASS",
                    "raw_matrix_npz": dict(npz),
                }
                manifest_rows.append(row_item)
                manifest_npz[row_id] = row_item
            manifest_path = point_root / f"{model_id}_result_manifest.json"
            manifest_value = {
                "schema": point_runner.POINT_MANIFEST_SCHEMA,
                **common,
                "results_jsonl_pin": raw_pin(ledger_pin),
                "raw_score_npz_by_row": manifest_npz,
                "row_results": manifest_rows,
            }
            point_runner._write_self_json(
                manifest_path, manifest_value, "manifest_sha256")
            manifest_file_entry = add_existing(f"{model_id}-manifest", "result_manifest",
                                               manifest_path, self_field="manifest_sha256")
            manifest_pin = raw_pin(manifest_file_entry, with_self=True)
            status_path = point_root / f"{model_id}_run_status.json"
            status_value = {
                "schema": point_runner.POINT_STATUS_SCHEMA,
                **common,
                "result_manifest_pin": manifest_pin,
            }
            point_runner._write_self_json(
                status_path, status_value, "status_sha256")
            status_file_entry = add_existing(f"{model_id}-status", "run_status",
                                             status_path, self_field="status_sha256")
            status_pin = raw_pin(status_file_entry, with_self=True)
            child_path = point_root / f"{model_id}_child_receipt.json"
            process = {
                "child_pid": common["child_pid"],
                "child_start_ticks": common["child_start_ticks"],
                "child_exit_code": 0,
                "started_utc": parent_started_utc, "ended_utc": parent_ended_utc,
            }
            child_pin = point_runner.write_child_execution_receipt(
                receipt_path=child_path, context={}, process=process,
                launcher_pin=model_sources["launcher_pin"],
                adapter_pin=model_sources["adapter_pin"],
                runner_pin=model_sources["runner_pin"],
                runtime_binding_pin=model_sources["runtime_binding_pin"],
                source_pins=source_pins,
                execution_package_manifest_pin=package_pin,
                run_status_pin=status_pin,
                result_manifest_pin=manifest_pin,
                results_jsonl_pin=raw_pin(ledger_pin),
            )
            child_file_pin = add_existing(f"{model_id}-child", "child_execution_receipt",
                                          child_path, self_field="receipt_sha256")
            block = {
                "model_id": model_id, "block_index": 0,
                "row_ids": [row["row_id"] for row in model_rows],
                "planned_rows_file_id": model_rows_pin["file_id"],
                "results_file_id": ledger_pin["file_id"],
                "run_status_file_id": status_file_entry["file_id"],
                "result_manifest_file_id": manifest_file_entry["file_id"],
                "child_execution_receipt_file_id": child_file_pin["file_id"],
                "execution_package_manifest_pin": package_pin,
                "point_runner_pin": inputs.execution_package["point_runner_pin"],
                "shared_application_helper_pin": inputs.execution_package["shared_application_helper_pin"],
                "launcher_pin": model_sources["launcher_pin"],
                "runner_pin": model_sources["runner_pin"],
                "adapter_pin": model_sources["adapter_pin"],
                "runtime_binding_pin": model_sources["runtime_binding_pin"],
                "source_pins": source_pins,
            }
            blocks.append(block)
        point_entry = {
            "point_id": proposal.method_id, "method_id": proposal.method_id,
            "point_kind": "spsa_proposal", "proposal_id": proposal.proposal_id,
            "pair_id": proposal.pair_id, "iteration": proposal.iteration,
            "side": proposal.side, "run_id": run_id,
            "plan_file_id": point_plan_pin["file_id"],
            "planned_rows_file_id": all_rows_pin["file_id"], "blocks": blocks,
        }
        return durable.PointOutcome(
            disposition="TERMINAL", terminal_status=terminal_status,
            point_entry=point_entry, files=tuple(files),
            raw_npz_by_row=raw_npz_by_row,
        )

    class _Executor:
        def __init__(self, owner, session, root, *, first_unknown=False,
                     fail_first=False, unknown_after_execute=False,
                     unknown_readbacks=0, coverage_mode=None):
            self.owner = owner
            self.session = session
            self.root = root
            self.source_files, self.identity_records = owner._load_identity_records(session)
            self.first_unknown = first_unknown
            self.fail_first = fail_first
            self.unknown_after_execute = unknown_after_execute
            self.unknown_readbacks = unknown_readbacks
            self.coverage_mode = coverage_mode
            self.calls = []

        def readback(self, proposal):
            self.calls.append(("readback", proposal.proposal_id, proposal.side))
            if self.first_unknown or self.unknown_readbacks:
                self.first_unknown = False
                if self.unknown_readbacks:
                    self.unknown_readbacks -= 1
                return durable.PointOutcome(disposition="UNKNOWN", reason="synthetic ambiguous readback")
            return durable.PointOutcome(disposition="ABSENT")

        def execute(self, proposal):
            self.calls.append(("execute", proposal.proposal_id, proposal.side))
            if self.unknown_after_execute:
                self.unknown_after_execute = False
                return durable.PointOutcome(
                    disposition="UNKNOWN", reason="synthetic child may have launched; readback required")
            status = "FAILED_BEFORE_RENDER" if self.fail_first else "COMPLETE"
            self.fail_first = False
            return self.owner._make_proposal_outcome(
                self.session, proposal, self.root, self.identity_records, self.source_files,
                terminal_status=status, coverage_mode=self.coverage_mode)

    def test_initial_selector_receipt_uses_complete_torch_720_and_exact_composer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt_path = root / "initial_selector_receipt.json"
            pin = durable.produce_initial_selector_receipt(index, composed, receipt_path)
            receipt, raw = durable._read_json(receipt_path, "initial receipt")
            self.assertEqual(pin["sha256"], _sha(raw))
            self.assertEqual(receipt["schema"], durable.INITIAL_RECEIPT_SCHEMA)
            self.assertEqual(receipt["status"], "INITIAL_720_SELECTOR_COMPLETE")
            self.assertEqual(receipt["metric_backend"], "torch_cpu")
            self.assertIn(receipt["selection_status"], final_ingest.ALLOWED_INITIAL_SELECTION)
            self.assertEqual(receipt["spsa_seed_method_id"], receipt["seed_snapshot_spec"]["method_id"])
            if receipt["selection_status"] == "NOT_ADMITTED":
                self.assertIsNone(receipt["selection"]["selected_method_id"])
                self.assertEqual(receipt["selection"]["refinement_seed_method_id"],
                                 receipt["spsa_seed_method_id"])
            else:
                self.assertEqual(receipt["selection"]["selected_method_id"],
                                 receipt["spsa_seed_method_id"])
            verified = durable._load_verified_inputs(index, receipt_path, composed, package)
            self.assertEqual(verified.family_selection.family.actual_rows, 720)
            session = durable.DurableSPSASession.create(
                session_dir=root / "default-session", initial_index_path=index,
                initial_receipt_path=receipt_path, composed_plan_path=composed,
                execution_package_path=package)
            self.assertTrue(session.manifest["session_id"].startswith(
                f"fit40_spsa_{receipt['receipt_sha256'][:12]}_"))

    def test_dispatch_intent_and_orphan_point_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="orphan-recovery-test")
            executor = self._Executor(self, session, root)
            append = session._append_event

            def fail_terminal_event(event_type, data):
                if event_type == "POINT_TERMINAL":
                    raise OSError("synthetic crash after point file creation")
                return append(event_type, data)

            session._append_event = fail_terminal_event
            with self.assertRaisesRegex(OSError, "synthetic crash"):
                session.run_pending_pair(executor)
            self.assertTrue(session.poisoned)
            self.assertEqual(sum(call[0] == "execute" for call in executor.calls), 1)
            self.assertTrue(any(event["event_type"] == "POINT_DISPATCH_INTENT"
                                for event in session.events))
            self.assertEqual(len(list((root / "session" / "point_entries").glob("*.json"))), 1)
            with self.assertRaisesRegex(durable.DurableSessionError, "point-entry files"):
                durable.DurableSPSASession.resume(session_dir=root / "session")

    def test_incomplete_720_cannot_produce_initial_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, _package = self._make_complete_inputs(root)
            value = json.loads(index.read_text(encoding="utf-8"))
            value["files"] = [item for item in value["files"] if item["role"] != "results_jsonl"]
            index.write_bytes(_canonical(_seal(value, "index_sha256")))
            with self.assertRaises(Exception):
                durable.produce_initial_selector_receipt(index, composed, root / "bad_receipt.json")

    def test_pending_ask_is_persisted_and_unknown_outcome_is_not_relaunched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="unknown-readback-test")
            pair = session.ask_pair()
            events = len(session.events)
            self.assertEqual(session.ask_pair(), pair)
            self.assertEqual(len(session.events), events)
            executor = self._Executor(self, session, root, first_unknown=True)
            with self.assertRaises(durable.UnknownPointOutcome):
                session.run_pending_pair(executor)
            self.assertEqual(session.pending_pair, pair)
            self.assertEqual(sum(call[0] == "execute" for call in executor.calls), 0)
            resumed = durable.DurableSPSASession.resume(session_dir=root / "session")
            self.assertEqual(final_ingest.controller_state(resumed.controller),
                             final_ingest.controller_state(session.controller))
            with self.assertRaises(durable.DurableSessionError):
                session.write_controller_journal_exclusive(root / "incomplete_journal.json")
            self.assertFalse((root / "incomplete_journal.json").exists())

    def test_unknown_after_execute_never_duplicates_on_unknown_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="unknown-after-execute-test")
            executor = self._Executor(self, session, root, unknown_after_execute=True)
            with self.assertRaises(durable.UnknownPointOutcome):
                session.run_pending_pair(executor)
            self.assertEqual(sum(call[0] == "execute" for call in executor.calls), 1)
            executor.unknown_readbacks = 1
            with self.assertRaises(durable.UnknownPointOutcome):
                session.run_pending_pair(executor)
            self.assertEqual(sum(call[0] == "execute" for call in executor.calls), 1)
            resumed = durable.DurableSPSASession.resume(session_dir=root / "session")
            executor.session = resumed
            executor.unknown_readbacks = 1
            with self.assertRaises(durable.UnknownPointOutcome):
                resumed.run_pending_pair(executor)
            self.assertEqual(sum(call[0] == "execute" for call in executor.calls), 1)
            executor.unknown_readbacks = 0  # The persisted intent makes even ABSENT non-dispatchable.
            with self.assertRaises(durable.UnknownPointOutcome):
                resumed.run_pending_pair(executor)
            self.assertEqual(sum(call[0] == "execute" for call in executor.calls), 1)
            self.assertEqual(final_ingest.controller_state(resumed.controller),
                             final_ingest.controller_state(session.controller))

    def test_complete_pair_rebuilds_official_rows_before_tell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="complete-pair-test")
            executor = self._Executor(self, session, root)
            tell = session.run_pending_pair(executor)
            self.assertEqual(tell.status, "UPDATED")
            self.assertIsNone(session.pending_pair)
            self.assertEqual(sum(call[0] == "execute" for call in executor.calls), 2)
            self.assertEqual(sum(event["event_type"] == "PAIR_TOLD" for event in session.events), 1)
            resumed = durable.DurableSPSASession.resume(session_dir=root / "session")
            self.assertEqual(final_ingest.controller_state(resumed.controller),
                             final_ingest.controller_state(session.controller))
            self.assertEqual(len(resumed.point_entries), 2)
            first_record = next(iter(resumed.point_entries.values()))
            npz_pin = next(iter(first_record["point_doc"]["raw_npz_by_row"].values()))
            target = Path(npz_pin["path"])
            raw = bytearray(target.read_bytes())
            raw[-1] ^= 1
            target.write_bytes(raw)
            with self.assertRaises(durable.DurableSessionError):
                durable.DurableSPSASession.resume(session_dir=root / "session")

    def test_point04_terminal_replay_uses_final_consumer_receipt_and_source_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="consumer-point04-test")
            pair = session.ask_pair()
            proposal = pair.plus
            session._append_dispatch_intent(proposal)
            executor = self._Executor(self, session, root)
            outcome = self._make_proposal_outcome(
                session, proposal, root, executor.identity_records, executor.source_files)
            terminal = session._append_point_outcome(proposal, outcome)
            point_doc = terminal["point_doc"]

            # The real final-center consumer re-reads status, manifest, child
            # process interval, nested package source pins, ledgers, and NPZs
            # before a proposal can reach controller.tell.
            evaluation = session._load_official_evaluation(proposal, point_doc)
            self.assertEqual(len(evaluation.rows), len(proposal.sample_ids) * 3 * len(MODELS))

            bad_source = copy.deepcopy(point_doc)
            first_block = bad_source["point_entry"]["blocks"][0]
            first_source_role = next(iter(first_block["source_pins"]))
            first_block["source_pins"][first_source_role]["sha256"] = "0" * 64
            with self.assertRaisesRegex(durable.DurableSessionError, "final-consumer block/receipt validation"):
                session._load_official_evaluation(proposal, bad_source)

            # A valid child receipt whose claimed parent interval starts after
            # the immutable child interval must fail containment.
            bad_interval = copy.deepcopy(point_doc)
            target_block = bad_interval["point_entry"]["blocks"][0]
            child_entry = next(item for item in bad_interval["file_entries"]
                               if item["file_id"] == target_block["child_execution_receipt_file_id"])
            child_path = Path(child_entry["local_path"])
            child_value = json.loads(child_path.read_text(encoding="utf-8"))
            child_value["started_utc"] = "2026-10-09T00:00:02.500000Z"
            child_value = durable._seal_document(child_value, "receipt_sha256")
            child_raw = (json.dumps(child_value, sort_keys=True, indent=2,
                                    ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            altered_path = child_path.with_name("tampered_parent_interval_receipt.json")
            _write(altered_path, child_raw)
            altered_entry = dict(child_entry)
            altered_entry.update({
                "file_id": child_entry["file_id"] + "-interval-test",
                "declared_path": str(altered_path.resolve()),
                "local_path": str(altered_path.resolve()),
                "bytes": len(child_raw), "sha256": _sha(child_raw),
            })
            bad_interval["file_entries"].append(altered_entry)
            target_block["child_execution_receipt_file_id"] = altered_entry["file_id"]
            previous_active = session._active_files
            session._active_files = dict(previous_active)
            session._active_files.update(initial_ingest._file_table({"files": [altered_entry]}))
            try:
                with self.assertRaisesRegex(durable.DurableSessionError,
                                            "final-consumer block/receipt validation"):
                    session._load_official_evaluation(proposal, bad_interval)
            finally:
                session._active_files = previous_active

    def test_candidate_support_mask_and_track_coverage_reach_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="coverage-subset-test")
            executor = self._Executor(self, session, root, coverage_mode="partial")
            pair = session.ask_pair()
            tell = session.run_pending_pair(executor)
            self.assertEqual(tell.status, "UPDATED")
            self.assertIsNotNone(tell.support_sha256)
            candidate_eval = session._evaluation_for_point(
                pair.plus, session.point_entries[(pair.plus.proposal_id, "plus")])
            sample = pair.sample_ids[0]
            candidate = next(row for row in candidate_eval.rows
                             if row.method_id == pair.plus.method_id
                             and row.sample_id == sample and row.model_id == MODELS[0])
            identity = next(row for row in candidate_eval.rows
                            if row.method_id == "identity"
                            and row.sample_id == sample and row.model_id == MODELS[0])
            self.assertNotEqual(candidate.valid_query_mask, identity.valid_query_mask)
            self.assertEqual(
                {(key[0], key[1], key[2]) for key in candidate.query_keys},
                {(key[0], key[1], key[2]) for key in identity.query_keys})
            self.assertNotEqual(candidate.query_keys, identity.query_keys)

    def test_empty_candidate_support_is_scored_then_no_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="empty-support-test")
            executor = self._Executor(self, session, root, coverage_mode="empty")
            tell = session.run_pending_pair(executor)
            self.assertEqual(tell.status, "NO_UPDATE")
            self.assertIn("empty common valid source-clock support", tell.reason)
            self.assertTrue(session.halted)
            plus_record = session.point_entries[(session.events[1]["ask_pair"]["plus"]["proposal_id"], "plus")]
            # The terminal ledger remains complete and its all-false mask is kept
            # in the pinned NPZ; only selector support eligibility rejects it.
            first_raw = next(iter(plus_record["point_doc"]["raw_npz_by_row"].values()))
            point_npz = next(pin for pin in plus_record["point_doc"]["file_entries"]
                             if pin["role"] == "raw_matrix_npz"
                             and pin["declared_path"] == first_raw["path"])
            archive = np.load(BytesIO(Path(point_npz["local_path"]).read_bytes()), allow_pickle=False)
            try:
                self.assertFalse(bool(np.any(archive["valid_query_mask"])))
            finally:
                archive.close()

    def test_candidate_raw_npz_paths_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="npz-path-reuse-test")
            executor = self._Executor(self, session, root)
            pair = session.ask_pair()
            plus = pair.plus
            session._append_dispatch_intent(plus)
            plus_outcome = self._make_proposal_outcome(
                session, plus, root, executor.identity_records, executor.source_files)
            # Reusing one physical NPZ for two candidate rows in the same point.
            within_map = copy.deepcopy(plus_outcome.raw_npz_by_row)
            within_rows = list(within_map)
            within_map[within_rows[1]] = copy.deepcopy(within_map[within_rows[0]])
            within_body = durable._point_entry_body(
                plus_outcome.point_entry, plus_outcome.files, within_map,
                plus_outcome.terminal_status)
            with self.assertRaisesRegex(durable.DurableSessionError, "path is reused"):
                session._validate_point_document(plus, within_body)

            # Reusing any initial baseline NPZ under a new candidate row.
            initial_pin = next(pin for pin in session.inputs.files.values()
                               if pin.role == "raw_matrix_npz")
            initial_arrays = durable.DurableSPSASession._read_and_validate_npz(initial_pin)
            initial_file_entry = {
                "file_id": "initial-reused-npz-test-only", "role": "raw_matrix_npz",
                "declared_path": initial_pin.declared_path, "local_path": initial_pin.local_path,
                "bytes": initial_pin.byte_count, "sha256": initial_pin.sha256,
            }
            initial_map = copy.deepcopy(plus_outcome.raw_npz_by_row)
            initial_rows = list(initial_map)
            initial_map[initial_rows[0]] = {
                "path": initial_pin.declared_path, "bytes": initial_pin.byte_count,
                "sha256": initial_pin.sha256, "array_refs": initial_arrays,
            }
            initial_body = durable._point_entry_body(
                plus_outcome.point_entry, tuple(plus_outcome.files) + (initial_file_entry,),
                initial_map, plus_outcome.terminal_status)
            with self.assertRaisesRegex(durable.DurableSessionError, "path is reused"):
                session._validate_point_document(plus, initial_body)

            stored_plus = session._append_point_outcome(plus, plus_outcome)
            minus = pair.minus
            session._append_dispatch_intent(minus)
            minus_outcome = self._make_proposal_outcome(
                session, minus, root, executor.identity_records, executor.source_files)
            old_ref = next(iter(stored_plus["point_doc"]["raw_npz_by_row"].values()))
            old_entry = next(pin for pin in stored_plus["point_doc"]["file_entries"]
                             if pin["role"] == "raw_matrix_npz"
                             and pin["declared_path"] == old_ref["path"])
            duplicate_entry = dict(old_entry, file_id="cross-side-reused-npz-test-only")
            cross_map = copy.deepcopy(minus_outcome.raw_npz_by_row)
            cross_rows = list(cross_map)
            cross_map[cross_rows[0]] = copy.deepcopy(old_ref)
            cross_body = durable._point_entry_body(
                minus_outcome.point_entry, tuple(minus_outcome.files) + (duplicate_entry,),
                cross_map, minus_outcome.terminal_status)
            with self.assertRaisesRegex(durable.DurableSessionError, "path is reused"):
                session._validate_point_document(minus, cross_body)
            session._append_point_outcome(minus, minus_outcome)
            self.assertEqual(session.run_pending_pair(executor).status, "UPDATED")

            # Reuse a prior-iteration candidate NPZ at a fresh point path table.
            next_pair = session.ask_pair()
            next_plus = next_pair.plus
            session._append_dispatch_intent(next_plus)
            next_outcome = self._make_proposal_outcome(
                session, next_plus, root, executor.identity_records, executor.source_files)
            next_map = copy.deepcopy(next_outcome.raw_npz_by_row)
            next_rows = list(next_map)
            next_map[next_rows[0]] = copy.deepcopy(old_ref)
            prior_entry = dict(old_entry, file_id="prior-iteration-reused-npz-test-only")
            prior_body = durable._point_entry_body(
                next_outcome.point_entry, tuple(next_outcome.files) + (prior_entry,),
                next_map, next_outcome.terminal_status)
            with self.assertRaisesRegex(durable.DurableSessionError, "path is reused"):
                session._validate_point_document(next_plus, prior_body)

    def test_terminal_failure_does_not_launch_sibling_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="terminal-failure-test")
            executor = self._Executor(self, session, root, fail_first=True)
            tell = session.run_pending_pair(executor)
            self.assertEqual(tell.status, "NO_UPDATE")
            self.assertEqual(sum(call[0] == "execute" for call in executor.calls), 1)
            self.assertEqual(sum(call[0] == "execute" and call[2] == "minus"
                                 for call in executor.calls), 0)
            resumed = durable.DurableSPSASession.resume(session_dir=root / "session")
            replay_executor = self._Executor(self, resumed, root)
            self.assertTrue(resumed.halted)
            self.assertEqual(sum(call[0] == "execute" for call in replay_executor.calls), 0)
            self.assertEqual(final_ingest.controller_state(resumed.controller),
                             final_ingest.controller_state(session.controller))

    def test_twenty_pairs_roundtrip_save_restore_and_terminal_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="twenty-pair-roundtrip-test")
            executor = self._Executor(self, session, root)
            saved_state = None
            for iteration in range(selector.SPSA_ITERATIONS):
                if iteration == 10:
                    saved_state = final_ingest.controller_state(session.controller)
                    session = durable.DurableSPSASession.resume(session_dir=root / "session")
                    executor.session = session
                    self.assertEqual(final_ingest.controller_state(session.controller), saved_state)
                tell = session.run_pending_pair(executor)
                self.assertEqual(tell.status, "UPDATED")
                self.assertEqual(tell.iteration, iteration + 1)
            self.assertEqual(session.controller.iteration, selector.SPSA_ITERATIONS)
            self.assertFalse(session.halted)
            self.assertEqual(len(session.point_entries), 2 * selector.SPSA_ITERATIONS)
            journal = session.controller_journal_document()
            self.assertEqual(journal["status"], "COMPLETE_20_ASK_TELL_PAIRS")
            self.assertEqual(len(journal["entries"]), selector.SPSA_ITERATIONS)
            journal_path = root / "controller_journal.json"
            pin = session.write_controller_journal_exclusive(journal_path)
            self.assertEqual(session.write_controller_journal_exclusive(journal_path), pin)
            loaded, raw = durable._read_json(journal_path, "synthetic complete journal")
            self.assertEqual(loaded["journal_sha256"], pin["self_sha256"])
            self.assertEqual(_sha(raw), pin["sha256"])
            replayed = durable.DurableSPSASession.resume(session_dir=root / "session")
            self.assertEqual(final_ingest.controller_state(replayed.controller),
                             final_ingest.controller_state(session.controller))

    def test_event_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index, composed, package = self._make_complete_inputs(root)
            receipt = root / "initial_selector_receipt.json"
            durable.produce_initial_selector_receipt(index, composed, receipt)
            session = durable.DurableSPSASession.create(
                session_dir=root / "session", initial_index_path=index,
                initial_receipt_path=receipt, composed_plan_path=composed,
                execution_package_path=package, session_id="event-tamper-test")
            session.ask_pair()
            event_path = root / "session" / "events" / "event_000001.json"
            event = json.loads(event_path.read_text(encoding="utf-8"))
            event["ask_pair"]["pair_id"] = "0" * 64
            event_path.write_text(json.dumps(event, sort_keys=True, indent=2) + "\n",
                                  encoding="utf-8")
            with self.assertRaises(durable.DurableSessionError):
                durable.DurableSPSASession.resume(session_dir=root / "session")


if __name__ == "__main__":
    unittest.main()
