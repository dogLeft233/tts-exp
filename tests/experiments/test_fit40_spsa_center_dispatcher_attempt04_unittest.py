"""Current center03 → actual point04 → consumer05/session04 contract tests.

No model, scorer, GPU, selector, session, or dispatch root is started here.
The actual point04 worker emits each 40-row model block from a deterministic
fixture adapter; center03 assembles all three blocks; consumer05 verifies the
official source clock and raw NPZ links consumed by session04.
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
    fit40_spsa_center_dispatcher_attempt03 as center,
    fit40_spsa_point_dispatcher_attempt04 as parent_dispatch,
    fit40_spsa_point_execution_attempt04 as point04,
    fit_only_final_center_stagea_ingest_attempt05 as final_ingest,
    fit40_durable_spsa_session_attempt04 as durable,
    fit_only_stagea_ingest_attempt03 as initial_ingest,
    fit_only_spsa_selection_attempt06 as selector,
)


MODELS = list(center.MODEL_ORDER)
THETA = (0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09)
AUDIO_CODE_SHA = hashlib.sha256(b"fixture PCM conversion code").hexdigest()
FFMPEG_SHA = hashlib.sha256(b"fixture ffmpeg binary").hexdigest()
AUDIO_OPERATION = "float32_to_pcm16_fixture_round_clip"
SCORE_MODEL_SHA = hashlib.sha256(b"fixture SyncNet model").hexdigest()
PREPROCESSING_SHA = hashlib.sha256(b"fixture preprocessing").hexdigest()


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
    from fractions import Fraction
    frames = range(92)
    stamps = [Fraction(frame, 25) for frame in frames]
    return {
        "query_frame_index": np.asarray(list(frames), dtype="<i4"),
        "query_timestamp_num": np.asarray([stamp.numerator for stamp in stamps], dtype="<i8"),
        "query_timestamp_den": np.asarray([stamp.denominator for stamp in stamps], dtype="<i8"),
        "track_frame_index": np.asarray(list(frames), dtype="<i4"),
        "valid_query_mask": np.ones((92,), dtype="|b1"),
        "distance_by_query_lag": np.zeros((92, 31), dtype="<f4"),
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
            crop_path = "/fixture/%s/selected-crop.mkv" % stem
            render_path = "/fixture/%s/render.mkv" % stem
            crop_sha = hashlib.sha256((stem + ":crop").encode()).hexdigest()
            crop_es_sha = hashlib.sha256((stem + ":crop-es").encode()).hexdigest()
            render_sha = hashlib.sha256((stem + ":render").encode()).hexdigest()
            render_es_sha = hashlib.sha256((stem + ":render-es").encode()).hexdigest()
            timestamps_sha = hashlib.sha256((stem + ":timestamps").encode()).hexdigest()
            from fractions import Fraction
            frames = []
            for index in range(97):
                stamp = Fraction(index, 25)
                decoded = index < 95
                frames.append({
                    "crop_frame_index": index,
                    "source_clock_frame_index": index,
                    "timestamp_num": stamp.numerator,
                    "timestamp_den": stamp.denominator,
                    "render_frame_index": index if decoded else None,
                    "frame_origin": "decoded_render" if decoded else "nominal_clock_outside_render_decode",
                })
            frame_raw = _canonical({
                "schema_version": "cross-tfg-track-frame-map-v2",
                "selected_crop_video_path": crop_path,
                "selected_crop_video_file_sha256": crop_sha,
                "selected_crop_video_elementary_sha256": crop_es_sha,
                "selected_track_id": "track_00000", "selected_track_index": 0,
                "render_video_path": render_path,
                "render_video_file_sha256": render_sha,
                "render_video_elementary_sha256": render_es_sha,
                "render_timestamps_sha256": timestamps_sha,
                "fps_num": 25, "fps_den": 1,
                "decoded_render_frame_count": 95,
                "frames": frames,
            }) + b"\n"
            frame_path.write_bytes(frame_raw)
            row = dict(planned)
            natural = planned["natural_pcm"]
            map_sha = hashlib.sha256(frame_raw).hexdigest()
            pcm16_sha = hashlib.sha256((stem + ":pcm16").encode()).hexdigest()
            primary = {
                "path": "/fixture/%s/primary.mkv" % stem,
                "audio_codec": "pcm_s16le", "channels": 1,
                "sample_rate_hz": 16000,
                "audio_pcm16_sha256": pcm16_sha,
                "audio_sample_count": 62_080,
                "frame_map_path": str(frame_path.resolve()),
                "frame_map_sha256": map_sha,
                "crop_video_path": crop_path,
                "crop_video_file_sha256": crop_sha,
                "crop_video_elementary_sha256": crop_es_sha,
                "selected_track_id": "track_00000", "selected_track_index": 0,
            }
            conversion = {
                "code_sha256": AUDIO_CODE_SHA,
                "ffmpeg_sha256": FFMPEG_SHA,
                "operation": AUDIO_OPERATION,
            }
            row.update({
                "schema_version": "cross_tfg_audio_head_stagea_v1",
                "natural_pcm_sha256": natural["pcm_sha256"],
                "score_audio_sha256": natural["pcm_sha256"],
                "source_media_kind": "video",
                "source_video_sha256": planned["source_video"]["file_sha256"],
                "source_face_sha256": planned["source_face"]["file_sha256"],
                "render": {
                    "fps_num": 25, "fps_den": 1,
                    "frame_count": 95, "crop_frame_count": 95,
                    "timestamps_sha256": timestamps_sha,
                    "render_video_path": render_path,
                    "render_video_file_sha256": render_sha,
                    "selected_track_crop": {
                        "path": crop_path, "file_sha256": crop_sha,
                        "video_elementary_sha256": crop_es_sha,
                        "selected_track_id": "track_00000", "selected_track_index": 0,
                        "frame_map_path": str(frame_path.resolve()),
                        "frame_map_sha256": map_sha,
                    },
                },
                "status": "PASS",
                "score": {
                    "score_model_sha256": SCORE_MODEL_SHA,
                    "preprocessing_sha256": PREPROCESSING_SHA,
                    "support_by_support": {"full": {"query_count": 92}},
                    "metrics_by_support": {"full": {
                        "anchor_column": 15, "zero_lag_column": 15,
                    }},
                    "primary_syncnet_input": primary,
                    "primary_pcm16_window": {
                        "source_float32_pcm_sha256": natural["pcm_sha256"],
                        "sample_rate_hz": 16000, "channels": 1,
                        "codec": "pcm_s16le", "source_frame_start": 0,
                        "source_frame_stop_exclusive": 97,
                        "start_sample": 0,
                        "requested_stop_sample_exclusive": 62_080,
                        "stop_sample_exclusive": 62_080,
                        "sample_count": 62_080,
                        "tail_clipped_to_natural_pcm": False,
                        "source_start_pts_num": 0, "source_start_pts_den": 1,
                        "source_stop_pts_num": 97, "source_stop_pts_den": 25,
                        "pcm_sha256": pcm16_sha, "conversion": conversion,
                    },
                    "raw_matrix_npz": {
                        "path": str(npz_path.resolve()),
                        "file_sha256": hashlib.sha256(npz_raw).hexdigest(),
                        "arrays": array_refs,
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
            input_sha = "f" * 64
            point_config = {
                "method": "fit40_spsa_center", "schema": "fit40_spsa_center_transform_v1",
                "point_kind": "spsa_center", "source_step": 5,
                "theta": list(THETA), "artifact_sha256": center_spec.artifact_sha256,
                "binding_sha256": center_spec.binding_sha256,
                "input_pins_sha256": input_sha,
            }
            point_method = {
                "kind": "fit40_spsa_center", "method_id": center_spec.method_id,
                "config": point_config, "config_sha256": hashlib.sha256(_canonical(point_config)).hexdigest(),
            }
            descriptors = [{
                "model_id": model_id, "audio_frontend": "fixture_frontend_v1",
                "preprocessing_sha256": PREPROCESSING_SHA, "model_sha256": SCORE_MODEL_SHA,
                "score_model_sha256": SCORE_MODEL_SHA,
            } for model_id in MODELS]
            audio_protocol = {
                "codec": "pcm_s16le", "channels": 1, "sample_rate_hz": 16000,
                "code_sha256": AUDIO_CODE_SHA, "ffmpeg_sha256": FFMPEG_SHA,
                "operation": AUDIO_OPERATION,
            }
            protocol = {
                "syncnet": {
                    "anchor_column": 15, "zero_lag_column": 15,
                    "lag_offsets_frames": list(range(-15, 16)),
                    "score_audio_estimand": "native_roi_original_pcm16",
                    "score_model_id": "fixture_syncnet_v2",
                    "score_model_sha256": SCORE_MODEL_SHA,
                    "source_sha256": {"syncnet_runner": hashlib.sha256(b"syncnet runner").hexdigest()},
                    "primary_audio_input": audio_protocol,
                    "score_reduction": {"method": "fixture per-query mean", "dtype": "float32"},
                },
                "roi": {"mode": "native_official", "estimand": "native_roi_original_pcm16"},
            }
            rows = []
            for sample in samples:
                for model_id in MODELS:
                    natural = {"path": "/pinned/source_wav/%s.wav" % sample,
                               "file_sha256": "c" * 64, "pcm_sha256": "d" * 64,
                               "sample_count": 62639, "sample_rate_hz": 16000,
                               "channels": 1, "encoding": "FLOAT_32"}
                    face_sha = hashlib.sha256((sample + ":face").encode()).hexdigest()
                    video_sha = hashlib.sha256((sample + ":video").encode()).hexdigest()
                    rows.append({
                        "schema_version": "cross_tfg_audio_head_stagea_v1",
                        "row_id": "%s_%s_%s" % (sample, model_id, center_spec.method_id),
                        "split": "fit", "sample_id": sample, "speaker_id": "S0052",
                        "model_id": model_id, "method_id": center_spec.method_id,
                        "face_id": "face_%s" % sample,
                        "seed": selector.RENDERER_SEED,
                        "method_config_sha256": point_method["config_sha256"],
                        "audio_frontend": "fixture_frontend_v1", "checkpoint_sha256": None,
                        "natural_pcm": natural, "score_audio": dict(natural),
                        "source_media_kind": "video",
                        "source_face": {"file_sha256": face_sha},
                        "source_video": {"file_sha256": video_sha},
                    })

            plan = {
                "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
                "run_id": "center-run-05", "rows": rows, "methods": [point_method],
                "models": descriptors, "protocol": protocol,
            }
            scope = {
                "scope_kind": "spsa_full_center", "point_kind": "spsa_center",
                "source_step": 5, "method_id": center_spec.method_id,
                "artifact_sha256": center_spec.artifact_sha256,
                "binding_sha256": center_spec.binding_sha256, "run_id": plan["run_id"],
            }
            plan_pin = _seal_json(source / "point-plan.json", plan, "plan_sha256")
            plan["plan_sha256"] = plan_pin["self_sha256"]
            runtime_files = {}
            family_runtime = {}
            family_pin = _seal_json(source / "family-plan.json", {
                "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
                "run_id": "family-run", "scientific_input_pins_sha256": input_sha,
                "models": descriptors,
                "execution": {"runtime_bindings_by_model": family_runtime},
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
            family_path = source / "family-plan.json"
            family_body = json.loads(family_path.read_text(encoding="utf-8"))
            family_body["execution"]["runtime_bindings_by_model"] = {
                model_id: {
                    "path": entries[model_id]["runtime_binding_pin"]["path"],
                    "bytes": entries[model_id]["runtime_binding_pin"]["bytes"],
                    "raw_sha256": entries[model_id]["runtime_binding_pin"]["sha256"],
                }
                for model_id in MODELS
            }
            family_pin = _seal_json(family_path, family_body, "plan_sha256")
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

            # This is the exact consumer validator invoked by durable session04
            # before tell and again during replay. Feed it the actual point04
            # ledgers/NPZs/maps emitted above for all 3×40 candidate rows.
            self.assertIs(durable.final_ingest, final_ingest)
            file_table = initial_ingest._file_table({"files": terminal["file_entries"]})
            point_pin_file = initial_ingest._required_file(
                file_table, entry["plan_file_id"], "plan")
            point_rows_file = initial_ingest._required_file(
                file_table, entry["planned_rows_file_id"], "planned_rows")
            family_pin_file = next(pin for pin in file_table.values()
                                   if pin.role == "composed_family_plan")
            package_file = next(pin for pin in file_table.values()
                                if pin.role == "execution_package_manifest")
            package_value = json.loads(initial_ingest._read_pinned(package_file).decode("utf-8"))
            checked_package = final_ingest._validate_execution_package_manifest(
                package_value, package_file, family_body, family_pin_file,
                family_pin_file and family_body["plan_sha256"], file_table)
            npz_by_declared = {
                pin.declared_path: pin for pin in file_table.values()
                if pin.role == "raw_matrix_npz"
            }
            converted, checked_blocks = final_ingest._validate_point_blocks(
                entry, plan, point_pin_file, rows, point_rows_file,
                family_pin_file, family_body["plan_sha256"], file_table,
                point_method, input_sha, npz_by_declared, set(), checked_package,
                set(), outer_npz_by_row=terminal["raw_npz_by_row"])
            self.assertEqual(len(converted), 120)
            self.assertEqual([block["model_id"] for block in checked_blocks], MODELS)
            self.assertEqual(sum(block["status_counts"]["PASS"] for block in entry["blocks"]), 120)
            self.assertEqual(len(entry["official_source_frame_map_file_ids"]), 120)

            # The selected crop map can contain 97 source-clock frames while
            # only 95 are decoded into the render; the nominal tail gives 92
            # official SyncNet queries.  Session04 delegates to this same
            # consumer function, so exercise contradictory refs and vectors.
            first_row_id = next(iter(entry["official_source_frame_map_file_ids"]))
            first_record = next(record for block in gathered for record in block["records"]
                                if record["row_id"] == first_row_id)
            first_plan_row = next(row for row in rows if row["row_id"] == first_row_id)
            first_frame_pin = initial_ingest._required_file(
                file_table, entry["official_source_frame_map_file_ids"][first_row_id],
                "official_source_frame_map")
            first_npz_pin = npz_by_declared[first_record["score"]["raw_matrix_npz"]["path"]]
            first_arrays = initial_ingest._load_arrays(
                first_npz_pin, first_record["score"]["raw_matrix_npz"])
            good_map = final_ingest.validate_official_source_frame_map(
                first_record, first_plan_row, first_arrays, first_frame_pin)
            self.assertEqual((good_map["frame_count"], good_map["query_count"]), (97, 92))
            self.assertEqual(len(first_arrays["distance_by_query_lag"]), 92)

            wrong_duplicate = copy.deepcopy(first_record)
            wrong_duplicate["render"]["selected_track_crop"]["frame_map_sha256"] = "0" * 64
            with self.assertRaises(final_ingest.FinalCenterIngestError):
                final_ingest.validate_official_source_frame_map(
                    wrong_duplicate, first_plan_row, first_arrays, first_frame_pin)
            wrong_query = {name: np.array(value, copy=True) for name, value in first_arrays.items()}
            wrong_query["query_frame_index"][0] += 1
            with self.assertRaisesRegex(final_ingest.FinalCenterIngestError, "mapped by its own"):
                final_ingest.validate_official_source_frame_map(
                    first_record, first_plan_row, wrong_query, first_frame_pin)
            wrong_render_count = copy.deepcopy(first_record)
            wrong_render_count["render"]["frame_count"] = 94
            with self.assertRaisesRegex(final_ingest.FinalCenterIngestError, "decoded-render frame count"):
                final_ingest.validate_official_source_frame_map(
                    wrong_render_count, first_plan_row, first_arrays, first_frame_pin)

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
