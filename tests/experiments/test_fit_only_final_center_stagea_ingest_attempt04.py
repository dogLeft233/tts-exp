from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit_only_final_center_stagea_ingest_attempt04 as ingest,
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


class FinalCenterIngestAttempt04Tests(unittest.TestCase):
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


class OfficialSourceFrameMapTests(unittest.TestCase):
    def _fixture(self, root: Path, *, source_frames=range(2, 12)):
        source_frames = list(source_frames)
        row_id = "candidate-row"
        natural_count = 12_800
        natural_pcm_sha = _sha(b"original natural float32 PCM")
        frame_map_path = "/producer/row/syncnet/selected_track_frame_map.json"
        crop_path = "/producer/row/syncnet/selected_crop_video.mkv"
        render_path = "/producer/row/render/muxed.mkv"
        crop_file_sha = _sha(b"crop video")
        crop_es_sha = _sha(b"crop ES")
        render_sha = _sha(b"render video")
        timestamps_sha = _sha(b"timestamps")
        frames = []
        for index, source_frame in enumerate(source_frames):
            stamp = Fraction(source_frame, 25)
            frames.append({
                "crop_frame_index": index,
                "source_clock_frame_index": source_frame,
                "timestamp_num": stamp.numerator,
                "timestamp_den": stamp.denominator,
                "render_frame_index": source_frame,
                "frame_origin": "decoded_render",
            })
        frame_map = {
            "schema_version": "cross-tfg-track-frame-map-v2",
            "selected_crop_video_path": crop_path,
            "selected_crop_video_file_sha256": crop_file_sha,
            "selected_crop_video_elementary_sha256": crop_es_sha,
            "selected_track_id": "track_00000",
            "selected_track_index": 0,
            "render_video_path": render_path,
            "render_video_file_sha256": render_sha,
            "render_video_elementary_sha256": _sha(b"render ES"),
            "render_timestamps_sha256": timestamps_sha,
            "fps_num": 25,
            "fps_den": 1,
            "frames": frames,
        }
        raw = json.dumps(frame_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
        local_map = root / "selected_track_frame_map.json"
        local_map.write_bytes(raw)
        pin = initial_ingest._PinnedFile(
            row_id, "official_source_frame_map", frame_map_path, str(local_map),
            len(raw), _sha(raw), row_id)
        source_start, source_stop = source_frames[0], source_frames[-1] + 1
        start_sample, requested_stop = source_start * 640, source_stop * 640
        actual_stop = min(requested_stop, natural_count)
        pcm_count = actual_stop - start_sample
        pcm_sha = _sha(b"original-source PCM16 window")
        primary = {
            "path": "/producer/row/syncnet/primary.mkv",
            "frame_map_path": frame_map_path,
            "frame_map_sha256": pin.sha256,
            "crop_video_path": crop_path,
            "crop_video_file_sha256": crop_file_sha,
            "crop_video_elementary_sha256": crop_es_sha,
            "selected_track_id": "track_00000",
            "selected_track_index": 0,
            "audio_pcm16_sha256": pcm_sha,
            "audio_sample_count": pcm_count,
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        score = {
            "primary_syncnet_input": primary,
            "primary_pcm16_window": {
                "source_float32_pcm_sha256": natural_pcm_sha,
                "sample_rate_hz": 16000,
                "channels": 1,
                "codec": "pcm_s16le",
                "source_frame_start": source_start,
                "source_frame_stop_exclusive": source_stop,
                "start_sample": start_sample,
                "requested_stop_sample_exclusive": requested_stop,
                "stop_sample_exclusive": actual_stop,
                "sample_count": pcm_count,
                "tail_clipped_to_natural_pcm": actual_stop < requested_stop,
                "source_start_pts_num": Fraction(source_start, 25).numerator,
                "source_start_pts_den": Fraction(source_start, 25).denominator,
                "source_stop_pts_num": Fraction(source_stop, 25).numerator,
                "source_stop_pts_den": Fraction(source_stop, 25).denominator,
                "pcm_sha256": pcm_sha,
            },
        }
        render = {
            "fps_num": 25,
            "fps_den": 1,
            "crop_frame_count": len(frames),
            "timestamps_sha256": timestamps_sha,
            "render_video_path": render_path,
            "render_video_file_sha256": render_sha,
            "selected_track_crop": {
                "path": crop_path,
                "file_sha256": crop_file_sha,
                "video_elementary_sha256": crop_es_sha,
                "selected_track_id": "track_00000",
                "selected_track_index": 0,
                "frame_map_path": frame_map_path,
                "frame_map_sha256": pin.sha256,
            },
        }
        query_count = max(0, min(len(frames), pcm_count // 640) - 5)
        query_frames = frames[:query_count]
        arrays = {
            "query_frame_index": np.asarray([item["source_clock_frame_index"] for item in query_frames], dtype="<i4"),
            "query_timestamp_num": np.asarray([item["timestamp_num"] for item in query_frames], dtype="<i8"),
            "query_timestamp_den": np.asarray([item["timestamp_den"] for item in query_frames], dtype="<i8"),
            "track_frame_index": np.asarray([item["source_clock_frame_index"] for item in query_frames], dtype="<i4"),
            "valid_query_mask": np.ones((query_count,), dtype=np.bool_),
            "distance_by_query_lag": np.zeros((query_count, 31), dtype="<f4"),
            "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
        }
        record = {"row_id": row_id, "score": score, "render": render}
        plan_row = {"natural_pcm": {
            "sample_count": natural_count,
            "sample_rate_hz": 16000,
            "channels": 1,
            "pcm_sha256": natural_pcm_sha,
        }}
        return record, plan_row, arrays, pin, frame_map

    def test_candidate_clock_can_extend_beyond_identity_track_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            record, plan_row, arrays, pin, _ = self._fixture(Path(temp))
            result = ingest.validate_official_source_frame_map(record, plan_row, arrays, pin)
            candidate_frames = set(arrays["query_frame_index"].tolist())
            identity_frames = set(range(5))
            self.assertTrue(candidate_frames - identity_frames)
            self.assertEqual(result["source_frame_start"], 2)
            self.assertEqual(result["query_count"], 5)

    def test_frame_map_raw_hash_and_timestamp_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record, plan_row, arrays, pin, frame_map = self._fixture(root)
            bad_record = json.loads(json.dumps(record))
            bad_record["score"]["primary_syncnet_input"]["frame_map_sha256"] = _sha(b"wrong map")
            with self.assertRaises(ingest.FinalCenterIngestError):
                ingest.validate_official_source_frame_map(bad_record, plan_row, arrays, pin)

            frame_map["frames"][1]["timestamp_num"] += 1
            raw = json.dumps(frame_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
            local = Path(pin.local_path)
            local.write_bytes(raw)
            changed_pin = initial_ingest._PinnedFile(
                pin.file_id, pin.role, pin.declared_path, pin.local_path,
                len(raw), _sha(raw), pin.pin_name)
            bad_record = json.loads(json.dumps(record))
            bad_record["score"]["primary_syncnet_input"]["frame_map_sha256"] = changed_pin.sha256
            bad_record["render"]["selected_track_crop"]["frame_map_sha256"] = changed_pin.sha256
            with self.assertRaisesRegex(ingest.FinalCenterIngestError, "timestamp"):
                ingest.validate_official_source_frame_map(bad_record, plan_row, arrays, changed_pin)

    def test_official_query_arrays_must_follow_candidate_map(self):
        with tempfile.TemporaryDirectory() as temp:
            record, plan_row, arrays, pin, _ = self._fixture(Path(temp))
            arrays["query_frame_index"][0] += 1
            with self.assertRaisesRegex(ingest.FinalCenterIngestError, "mapped by its own"):
                ingest.validate_official_source_frame_map(record, plan_row, arrays, pin)

    def test_pcm_window_cannot_escape_original_natural_sample_count(self):
        with tempfile.TemporaryDirectory() as temp:
            record, plan_row, arrays, pin, _ = self._fixture(Path(temp), source_frames=range(20, 30))
            with self.assertRaisesRegex(ingest.FinalCenterIngestError, "no natural-audio window"):
                ingest.validate_official_source_frame_map(record, plan_row, arrays, pin)

    def test_missing_or_wrong_role_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            record, plan_row, arrays, pin, _ = self._fixture(Path(temp))
            wrong_role = initial_ingest._PinnedFile(
                pin.file_id, "raw_matrix_npz", pin.declared_path, pin.local_path,
                pin.byte_count, pin.sha256, pin.pin_name)
            with self.assertRaises(ingest.FinalCenterIngestError):
                ingest.validate_official_source_frame_map(record, plan_row, arrays, wrong_role)


if __name__ == "__main__":
    unittest.main()
