from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.experiments.cross_tfg_audio_head_20261003.fit_only_stagea_ingest_attempt01 import (
    SCHEMA,
    IngestError,
    ingest_family,
    initialize_spsa,
)


METHOD_IDS = (
    "identity", "rms_fixed",
    "fit40_native_transition_head_v3_attempt11_step030",
    "fit40_native_transition_head_v3_attempt11_step060",
    "fit40_native_transition_head_v3_attempt11_step090",
    "fit40_native_transition_head_v3_attempt11_step120",
)
STEPS = (30, 60, 90, 120)
MODELS = ("model_w2l", "model_muse", "model_ditto")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _seal(value, field):
    value[field] = _sha(_canonical({k: v for k, v in value.items() if k != field}))
    return value


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


class FitStageAIngestTests(unittest.TestCase):
    def _make_packet(self, root: Path, *, omitted_row: tuple[str, str] | None = None,
                     corrupt_snapshot_index_entry: bool = False,
                     wrong_model_descriptor: bool = False,
                     wrong_anchor_plan: bool = False):
        files = []
        input_pin_files = {}

        def add_file(file_id, role, declared_path, raw, *, pin_name=None, local_name=None):
            local = root / "files" / (local_name or file_id)
            _write(local, raw)
            item = {
                "file_id": file_id, "role": role, "declared_path": declared_path,
                "local_path": str(local.resolve()), "bytes": len(raw), "sha256": _sha(raw),
            }
            if pin_name is not None:
                item["pin_name"] = pin_name
            files.append(item)
            return item

        point_raw = {}
        point_artifact = {}
        snapshot_entries = []
        for step in STEPS:
            method_id = METHOD_IDS[2 + STEPS.index(step)]
            values = [[0.0, 0.0, 0.0] for _ in range(3)]
            values[0][0] = float(step) / 1000.0
            parameter_raw = np.asarray(values, dtype="<f4").tobytes(order="C")
            artifact_sha = _sha(f"snapshot-artifact-{step}".encode())
            record = {
                "schema": "cross_tfg_fit40_parameter_snapshot_v1",
                "method_id": method_id,
                "snapshot_sha256": artifact_sha,
                "source_optimizer_step": step,
                "training_family": "fit40_native_transition_head_v3_attempt11",
                "selection_status": "NOT_SELECTED",
                "optimizer_step_is_selected": False,
                "parameter": {
                    "shape": [3, 3], "values": values,
                    "little_endian_f32_sha256": _sha(parameter_raw),
                },
            }
            raw = _canonical(record)
            point_raw[step] = raw
            point_artifact[step] = artifact_sha
            snapshot_entries.append({
                "method_id": method_id, "source_optimizer_step": step,
                "snapshot_sha256": artifact_sha, "path": f"snapshot_step_{step:03d}.json",
                "bytes": len(raw), "sha256": _sha(raw),
            })

        snapshot_index = _seal({
            "schema": "cross_tfg_fit40_snapshot_set_index_v1",
            "status": "EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND",
            "snapshots": snapshot_entries,
        }, "index_sha256")
        if corrupt_snapshot_index_entry:
            snapshot_index["snapshots"][0]["bytes"] += 1
            snapshot_index = _seal(snapshot_index, "index_sha256")
        snapshot_index_raw = _canonical(snapshot_index)
        input_pin_bytes = {"snapshot_index": snapshot_index_raw}
        for step in STEPS:
            input_pin_bytes[f"snapshot_step_{step:03d}"] = point_raw[step]
        input_pins = {}
        for pin_name, raw in input_pin_bytes.items():
            relative = f"evidence/{pin_name}.json"
            input_pins[pin_name] = {"path": relative, "bytes": len(raw), "sha256": _sha(raw)}
            input_pin_files[pin_name] = add_file(
                f"pin-{pin_name}", "input_pin", relative, raw,
                pin_name=pin_name,
            )["file_id"]

        methods = []
        identity_config = {"method": "identity", "sample_rate_hz": 16000}
        rms_config = {"method": "rms", "peak_ceiling": None,
                      "sample_rate_hz": 16000, "target_rms": 0.0376838172675746}
        methods.append({"method_id": METHOD_IDS[0], "kind": "identity", "config": identity_config,
                        "config_sha256": _sha(_canonical(identity_config))})
        methods.append({"method_id": METHOD_IDS[1], "kind": "rms_fixed", "config": rms_config,
                        "config_sha256": _sha(_canonical(rms_config))})
        for step in STEPS:
            method_id = METHOD_IDS[2 + STEPS.index(step)]
            config = {
                "method": "precomputed_fit_snapshot_waveform", "sample_rate_hz": 16000,
                "selection_status": "NOT_SELECTED", "snapshot_sha256": point_artifact[step],
                "source_optimizer_step": step,
                "training_family": "fit40_native_transition_head_v3_attempt11",
            }
            methods.append({"method_id": method_id, "kind": "fit_snapshot", "config": config,
                            "config_sha256": _sha(_canonical(config))})

        samples = [f"sample_{i:03d}" for i in range(40)]
        speakers = {sample: f"speaker_{i // 2:02d}" for i, sample in enumerate(samples)}
        face_sha = _sha(b"same static face")
        pcm_by_sample = {sample: _sha(f"pcm-{sample}".encode()) for sample in samples}
        plan_rows_by_model = {}
        plan_files = {}
        audio_frontend = "synthetic_official_frontend_v1"
        score_model_sha = _sha(b"synthetic syncnet checkpoint")
        syncnet_protocol = {
            "anchor_column": 15,
            "zero_lag_column": 15,
            "lag_offsets_frames": list(range(-15, 16)),
            "score_audio_estimand": "native_roi_original_pcm16",
            "score_model_id": "syncnet_v2",
            "score_model_sha256": score_model_sha,
            "source_sha256": {"SyncNetInstance.py": _sha(b"source")},
            "primary_audio_input": {
                "code_sha256": _sha(b"audio decoder"),
                "ffmpeg_sha256": _sha(b"ffmpeg"),
                "codec": "pcm_s16le", "channels": 1, "sample_rate_hz": 16000,
            },
            "score_reduction": {"backend": "pytorch_cpu", "matrix_layout": "query_by_lag_float32"},
        }

        for model_id in MODELS:
            rows = []
            for sample in samples:
                for method in methods:
                    method_id = method["method_id"]
                    pcm_sha = pcm_by_sample[sample]
                    rows.append({
                        "row_id": f"fit_{sample}_{model_id}_{method_id}",
                        "split": "fit", "sample_id": sample,
                        "speaker_id": speakers[sample], "model_id": model_id,
                        "face_id": "static_face_v1", "seed": 20261003,
                        "method_id": method_id,
                        "method_config_sha256": method["config_sha256"],
                        "audio_frontend": audio_frontend,
                        "checkpoint_sha256": None,
                        "natural_pcm": {"pcm_sha256": pcm_sha, "file_sha256": _sha(f"file-{sample}".encode())},
                        "score_audio": {"pcm_sha256": pcm_sha},
                        "source_media_kind": "static_portrait",
                        "source_video": {"file_sha256": None},
                        "source_face": {"file_sha256": face_sha},
                    })
            plan_protocol = dict(syncnet_protocol)
            if wrong_anchor_plan and model_id == MODELS[0]:
                plan_protocol["anchor_column"] = 14
            plan = {
                "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
                "run_id": f"fit40_{model_id}_run01",
                "models": [{
                    "model_id": ("wrong_model_id" if wrong_model_descriptor and model_id == MODELS[0]
                                 else model_id),
                    "audio_frontend": audio_frontend,
                    "model_sha256": _sha(f"model-{model_id}".encode()),
                    "score_model_sha256": score_model_sha,
                }],
                "protocol": {
                    "syncnet": plan_protocol,
                    "roi": {"mode": "native_official", "estimand": "native_roi_original_pcm16"},
                },
                "rows": rows, "methods": methods,
                "scientific_input_pins": input_pins,
                "scientific_input_pins_sha256": _sha(_canonical(input_pins)),
                "generation_seed_family": {"fit": [20261003]},
            }
            _seal(plan, "plan_sha256")
            plan_raw = _canonical(plan)
            plan_rel = f"remote/{model_id}/plan.json"
            planned_raw = b"".join(_canonical(row) + b"\n" for row in rows)
            planned_rel = f"remote/{model_id}/planned_rows.jsonl"
            plan_pin = add_file(f"{model_id}-plan", "plan", plan_rel, plan_raw)
            planned_pin = add_file(f"{model_id}-planned", "planned_rows", planned_rel, planned_raw)
            plan_rows_by_model[model_id] = rows
            plan_files[model_id] = (
                plan_pin["file_id"], planned_pin["file_id"], _sha(plan_raw),
                plan["plan_sha256"], plan_rel, planned_rel,
            )

        arrays = {
            "query_frame_index": np.arange(27, dtype="<i4"),
            "query_timestamp_num": np.arange(27, dtype="<i8"),
            "query_timestamp_den": np.full(27, 25, dtype="<i8"),
            "track_frame_index": np.arange(27, dtype="<i4"),
            "valid_query_mask": np.ones(27, dtype=np.bool_),
            "distance_by_query_lag": np.ones((27, 31), dtype="<f4"),
            "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
        }
        arrays["distance_by_query_lag"][:, 4] = 0.6
        arrays["distance_by_query_lag"][:, 15] = 0.95
        array_meta = {
            name: {"dtype": value.dtype.str, "shape": list(value.shape),
                   "array_sha256": _sha(np.ascontiguousarray(value).tobytes(order="C"))}
            for name, value in arrays.items()
        }
        npz_stream = BytesIO()
        np.savez_compressed(npz_stream, **arrays)
        npz_raw = npz_stream.getvalue()
        npz_sha = _sha(npz_raw)
        model_entries = []

        for model_id in MODELS:
            plan_rows = plan_rows_by_model[model_id]
            plan_file_id, planned_file_id, plan_raw_sha, plan_sha, plan_rel, planned_rel = plan_files[model_id]
            model_planned_rows_raw = b"".join(_canonical(row) + b"\n" for row in plan_rows)
            blocks = []
            for block_index in range(40):
                block_rows = plan_rows[block_index * 6:(block_index + 1) * 6]
                records = []
                counts = []
                for plan_row in block_rows:
                    is_omitted = bool(omitted_row and omitted_row == (model_id, plan_row["row_id"]))
                    if is_omitted:
                        counts.append({
                            "method_id": plan_row["method_id"], "model_id": model_id,
                            "planned_rows": 1, "result_rows": 0,
                            "status_counts": {"PASS": 0, "FAILED": 0, "INCOMPLETE": 0},
                        })
                        continue
                    record = {
                        "schema_version": "cross_tfg_audio_head_stagea_v1",
                        "status": "PASS", "failure_reason": None,
                        "row_id": plan_row["row_id"], "split": "fit",
                        "sample_id": plan_row["sample_id"], "speaker_id": plan_row["speaker_id"],
                        "model_id": model_id, "face_id": plan_row["face_id"],
                        "seed": 20261003, "method_id": plan_row["method_id"],
                        "method_config_sha256": plan_row["method_config_sha256"],
                        "audio_frontend": audio_frontend,
                        "checkpoint_sha256": None,
                        "score_model_sha256": score_model_sha,
                        "natural_pcm_sha256": plan_row["natural_pcm"]["pcm_sha256"],
                        "score_audio_sha256": plan_row["natural_pcm"]["pcm_sha256"],
                        "source_media_kind": "static_portrait", "source_video_sha256": None,
                        "source_face_sha256": face_sha,
                        "score": {
                            "score_model_sha256": score_model_sha,
                            "preprocessing_sha256": syncnet_protocol["primary_audio_input"]["code_sha256"],
                            "primary_syncnet_input": {
                                "audio_codec": "pcm_s16le", "channels": 1, "sample_rate_hz": 16000,
                            },
                            "raw_matrix_npz": {
                                "path": f"/remote/{model_id}/rows/{plan_row['row_id']}/raw_score_arrays.npz",
                                "file_sha256": npz_sha, "arrays": array_meta,
                            },
                            "support_by_support": {"full": {"query_count": 27}},
                            "metrics_by_support": {"full": {"status": "COMPLETE",
                                                               "anchor_column": 15,
                                                               "zero_lag_column": 15}},
                        },
                    }
                    records.append(record)
                    counts.append({
                        "method_id": plan_row["method_id"], "model_id": model_id,
                        "planned_rows": 1, "result_rows": 1,
                        "status_counts": {"PASS": 1, "FAILED": 0, "INCOMPLETE": 0},
                    })
                    npz_pin = add_file(
                        f"npz-{model_id}-{plan_row['row_id']}", "raw_matrix_npz",
                        record["score"]["raw_matrix_npz"]["path"], npz_raw,
                        local_name=f"npz/{model_id}/{plan_row['row_id']}.npz",
                    )
                ledger_raw = b"".join(_canonical(record) + b"\n" for record in records)
                ledger_pin = add_file(
                    f"ledger-{model_id}-{block_index:02d}", "results_jsonl",
                    f"/remote/{model_id}/block{block_index:02d}/results.jsonl", ledger_raw,
                )
                status_counts = {"PASS": len(records), "FAILED": 0, "INCOMPLETE": 0}
                manifest = _seal({
                    "schema_version": "cross-tfg-audio-eval-v1",
                    "run_id": f"fit40_{model_id}_run01", "plan_raw_sha256": plan_raw_sha,
                    "plan_sha256": plan_sha, "parent_planned_rows": 240,
                    "planned_rows": 240, "block_index": block_index,
                    "block_start_index": block_index * 6, "block_rows": 6,
                    "result_rows": len(records), "results_jsonl_sha256": _sha(ledger_raw),
                    "counts_by_model_method": counts, "status_counts": status_counts,
                    "status": "BLOCK_COMPLETE",
                }, "manifest_sha256")
                manifest_raw = _canonical(manifest)
                manifest_pin = add_file(
                    f"manifest-{model_id}-{block_index:02d}", "result_manifest",
                    f"/remote/{model_id}/block{block_index:02d}/result_manifest.json", manifest_raw,
                )
                run_status = {
                    "status": "BLOCK_COMPLETE", "run_id": manifest["run_id"],
                    "plan_path": plan_rel, "planned_rows_path": planned_rel,
                    "plan_raw_sha256": plan_raw_sha, "planned_rows": 240,
                    "planned_rows_sha256": _sha(model_planned_rows_raw),
                    "block_index": block_index, "block_start_index": block_index * 6,
                    "block_rows": 6, "completed_rows": 6,
                    "result_manifest_sha256": _sha(manifest_raw),
                    "status_counts": status_counts,
                }
                status_pin = add_file(
                    f"status-{model_id}-{block_index:02d}", "run_status",
                    f"/remote/{model_id}/block{block_index:02d}/run_status.json",
                    _canonical(run_status),
                )
                blocks.append({
                    "block_index": block_index,
                    "run_status_file_id": status_pin["file_id"],
                    "result_manifest_file_id": manifest_pin["file_id"],
                    "results_file_id": ledger_pin["file_id"],
                })
            model_entries.append({
                "model_id": model_id, "plan_file_id": plan_file_id,
                "planned_rows_file_id": planned_file_id, "blocks": blocks,
            })
        index = {
            "schema": SCHEMA,
            "status": "READY_FOR_LOCAL_INGEST_NOT_SELECTION",
            "scientific_input_pins_sha256": _sha(_canonical(input_pins)),
            "expected_row_count": 720,
            "input_pin_files": input_pin_files,
            "files": files,
            "models": model_entries,
        }
        index = _seal(index, "index_sha256")
        index_path = root / "fit_family_ingest_index.json"
        _write(index_path, _canonical(index))
        return index_path, index

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "canonical Fit selector admission requires CPU PyTorch")
    def test_complete_720_native_family_initializes_cpu_spsa(self):
        with tempfile.TemporaryDirectory() as temp:
            index_path, _ = self._make_packet(Path(temp))
            result = ingest_family(index_path, metric_backend="torch_cpu")
            self.assertIn(result.status, {"ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV", "NOT_ADMITTED"})
            self.assertIsNotNone(result.spsa_seed_method_id)
            self.assertTrue(result.family.complete_transport)
            self.assertEqual(result.family.actual_rows, 720)
            self.assertEqual(result.family.verified_npz_files, 720)
            self.assertTrue(result.family.score_protocol_sha256)
            self.assertEqual(len(result.family.score_npz_bindings), 720)
            binding = result.family.score_npz_bindings[0]
            self.assertEqual(binding["declared_path"].startswith("/remote/"), True)
            self.assertTrue(Path(binding["local_path"]).is_absolute())
            self.assertEqual(binding["bytes"], len(Path(binding["local_path"]).read_bytes()))
            controller = initialize_spsa(result, metric_backend="torch_cpu")
            self.assertEqual(controller.iteration, 0)
            self.assertEqual(len(controller.theta), 9)

    def test_missing_result_stays_in_720_denominator_and_never_seeds_spsa(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing_id = "fit_sample_000_model_w2l_identity"
            index_path, _ = self._make_packet(root, omitted_row=(MODELS[0], missing_id))
            result = ingest_family(index_path, metric_backend="numpy_diagnostic")
            self.assertEqual(result.status, "CONTROL_FAILED_MISSING_FIT_EVIDENCE")
            self.assertEqual(result.family.expected_rows, 720)
            self.assertEqual(result.family.actual_rows, 719)
            self.assertEqual(result.family.missing_row_ids, (missing_id,))
            self.assertIsNone(result.spsa_seed_method_id)
            with self.assertRaisesRegex(Exception, "complete verified 720-row"):
                initialize_spsa(result, metric_backend="torch_cpu")

    def test_corrupt_whole_npz_fails_closed_before_family_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path, index = self._make_packet(root)
            npz_entry = next(item for item in index["files"] if item["role"] == "raw_matrix_npz")
            Path(npz_entry["local_path"]).write_bytes(b"corrupt archive")
            result = ingest_family(index_path, metric_backend="numpy_diagnostic")
            self.assertEqual(result.status, "CONTROL_FAILED_INPUT_EVIDENCE")
            self.assertIsNone(result.selection)
            self.assertEqual(result.family.expected_rows, 720)
            self.assertTrue(any("differs from exact file pin" in error
                                for error in result.family.input_artifact_failures))

    def test_snapshot_index_path_and_bytes_are_bound_to_materialized_input_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            index_path, _ = self._make_packet(Path(temp), corrupt_snapshot_index_entry=True)
            with self.assertRaisesRegex(IngestError, "snapshot index file reference differs"):
                ingest_family(index_path, metric_backend="numpy_diagnostic")

    def test_wrong_single_model_descriptor_model_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            index_path, _ = self._make_packet(Path(temp), wrong_model_descriptor=True)
            with self.assertRaisesRegex(IngestError, "model descriptor does not match"):
                ingest_family(index_path, metric_backend="numpy_diagnostic")

    def test_wrong_plan_anchor_is_rejected_before_score_adaptation(self):
        with tempfile.TemporaryDirectory() as temp:
            index_path, _ = self._make_packet(Path(temp), wrong_anchor_plan=True)
            with self.assertRaisesRegex(IngestError, "anchor/zero-lag/lag axis"):
                ingest_family(index_path, metric_backend="numpy_diagnostic")


if __name__ == "__main__":
    unittest.main()
