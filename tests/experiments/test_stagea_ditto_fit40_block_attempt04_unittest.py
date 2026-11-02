"""Stdlib-only contract checks for the Fit40 Ditto row adapter.

Fixtures in this file are synthetic and are not model scores or experiment
results.  The adapter import does not load renderer, model, CUDA, or media
dependencies.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt04.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("ditto_fit40_adapter_under_test", ADAPTER_PATH)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)


METHODS = [{"method_id": f"method_{index:02d}"} for index in range(6)]


def synthetic_parent_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip_index in range(40):
        speaker_id = f"speaker_{clip_index // 2:02d}"
        sample_id = f"clip_{clip_index:02d}"
        natural = {
            "path": f"/synthetic/input/{sample_id}.wav",
            "file_sha256": f"{clip_index + 1:064x}",
            "pcm_sha256": f"{clip_index + 101:064x}",
            "sample_count": 92_225 + clip_index,
            "sample_rate_hz": 16_000,
            "channels": 1,
        }
        face = {"path": "/synthetic/portrait.png", "file_sha256": ADAPTER.FACE_SHA256}
        source_video = {"path": "/synthetic/static_portrait.mp4", "file_sha256": "c" * 64}
        for method in METHODS:
            rows.append({
                "row_id": f"parent_{clip_index:02d}_{method['method_id']}",
                "sample_id": sample_id,
                "speaker_id": speaker_id,
                "split": "fit",
                "source_media_kind": "static_portrait",
                "seed": 20261003,
                "face_id": "fixed_face",
                "model_id": "w2l_parent",
                "audio_frontend": "w2l_parent_frontend",
                "method_id": method["method_id"],
                "natural_pcm": dict(natural),
                "score_audio": dict(natural),
                "source_face": dict(face),
                "source_video": dict(source_video),
                "generation_settings": {"frozen": True},
            })
    return rows


class DittoPlanProjectionTests(unittest.TestCase):
    def test_full_240_row_projection_keeps_six_method_contexts_and_native_ceil(self) -> None:
        parent = synthetic_parent_rows()
        projected = ADAPTER.project_fit_rows(parent, METHODS)
        self.assertEqual(len(projected), 240)
        self.assertEqual(len({row["sample_id"] for row in projected}), 40)
        self.assertEqual(len({row["speaker_id"] for row in projected}), 20)
        for start in range(0, 240, 6):
            source_block = parent[start:start + 6]
            block = ADAPTER.validate_block({"rows": projected, "methods": METHODS}, start)
            self.assertEqual([row["method_id"] for row in block], [m["method_id"] for m in METHODS])
            self.assertEqual({row["sample_id"] for row in block}, {source_block[0]["sample_id"]})
            self.assertEqual({row["speaker_id"] for row in block}, {source_block[0]["speaker_id"]})
            self.assertEqual({row["seed"] for row in block}, {20261003})
            for row, source in zip(block, source_block):
                self.assertEqual(row["natural_pcm"], source["natural_pcm"])
                self.assertEqual(row["score_audio"], source["score_audio"])
                self.assertEqual(row["source_face"], source["source_face"])
                self.assertEqual(row["source_video"], source["source_video"])
                self.assertEqual(row["ditto_native_frame_binding"]["expected_frame_count"],
                                 ADAPTER.expected_native_frame_count(source["natural_pcm"]["sample_count"]))

    def test_parent_denominator_and_method_order_are_not_relaxed(self) -> None:
        parent = synthetic_parent_rows()
        with self.assertRaises(ValueError):
            ADAPTER.project_fit_rows(parent[:-1], METHODS)
        bad = [dict(row) for row in parent]
        bad[1]["score_audio"] = {**bad[1]["score_audio"], "pcm_sha256": "f" * 64}
        with self.assertRaises(ValueError):
            ADAPTER.project_fit_rows(bad, METHODS)
        projected = ADAPTER.project_fit_rows(parent, METHODS)
        with self.assertRaises(ValueError):
            ADAPTER.validate_block({"rows": projected, "methods": METHODS}, 1)


class DittoFrameClockTests(unittest.TestCase):
    def test_native_count_uses_ceiling_without_trimming_audio(self) -> None:
        self.assertEqual(ADAPTER.expected_native_frame_count(92_225), 145)
        self.assertEqual(ADAPTER.expected_native_frame_count(1), 1)
        self.assertEqual(ADAPTER.expected_native_frame_count(640), 1)
        self.assertEqual(ADAPTER.expected_native_frame_count(641), 2)
        with self.assertRaises(ValueError):
            ADAPTER.expected_native_frame_count(0)

    def test_reduced_rational_pts_are_compared_by_cross_multiplication(self) -> None:
        self.assertTrue(ADAPTER.timestamp_matches_frame({"timestamp_num": 0, "timestamp_den": 1}, 0))
        self.assertTrue(ADAPTER.timestamp_matches_frame({"timestamp_num": 1, "timestamp_den": 25}, 1))
        self.assertTrue(ADAPTER.timestamp_matches_frame({"timestamp_num": 1, "timestamp_den": 5}, 5))
        self.assertTrue(ADAPTER.timestamp_matches_frame({"timestamp_num": 1, "timestamp_den": 1}, 25))
        self.assertFalse(ADAPTER.timestamp_matches_frame({"timestamp_num": 1, "timestamp_den": 25}, 5))
        self.assertFalse(ADAPTER.timestamp_matches_frame({"timestamp_num": 1, "timestamp_den": 0}, 5))


class DittoRendererReportTests(unittest.TestCase):
    def _valid_inputs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        sample_count = 92_225
        row = {
            "row_id": "synthetic_renderer_row",
            "seed": 20261003,
            "ditto_native_frame_binding": {
                "sample_count": sample_count,
                "expected_frame_count": 145,
            },
        }
        audio_ref = {"sample_count": sample_count, "file_sha256": "a" * 64, "pcm_sha256": "b" * 64}
        expected_face = {"path": "/synthetic/portrait.png", "bytes": ADAPTER.FACE_BYTES,
                         "file_sha256": ADAPTER.FACE_SHA256}
        expected_source = {"source_commit": "synthetic-commit", "source_files": {}, "weights": {}}
        report = {
            "status": "PASS_ENGINEERING_ONLY",
            "model": "ditto",
            "profile": ADAPTER.RENDERER_RESULT_PROFILE,
            "row_id": row["row_id"],
            "seed": {
                "seed": 20261003,
                "python_numpy_torch_seeded": True,
                "cuda_available": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "tf32": False,
            },
            "audio": {
                "file_sha256": audio_ref["file_sha256"],
                "pcm_float32_sha256": audio_ref["pcm_sha256"],
                "sample_count": sample_count,
                "sample_rate_hz": 16_000,
                "channels": 1,
            },
            "face": dict(expected_face),
            "source_and_weights": expected_source,
            "silent_video": {
                "path": "/synthetic/out.mp4",
                "bytes": 1234,
                "file_sha256": "d" * 64,
                "probe": {"frame_count": 145, "decoded_frame_count": 145,
                          "fps": "25/1", "decoded_pts_exact_25fps": True},
            },
        }
        return row, audio_ref, expected_face, expected_source, report

    def test_actual_seed_record_shape_is_required(self) -> None:
        row, audio, face, source, report = self._valid_inputs()
        ADAPTER.validate_renderer_input_report(row, audio_ref=audio, report=report,
                                               expected_source_and_weights=source, expected_face=face)
        for key, wrong in (("profile", "wrong"),):
            mutated = {**report, key: wrong}
            with self.assertRaises(RuntimeError):
                ADAPTER.validate_renderer_input_report(row, audio_ref=audio, report=mutated,
                                                       expected_source_and_weights=source, expected_face=face)
        for field, wrong in (("seed", 20261003), ("seed", {**report["seed"], "tf32": True})):
            mutated = {**report, field: wrong}
            with self.assertRaises(RuntimeError):
                ADAPTER.validate_renderer_input_report(row, audio_ref=audio, report=mutated,
                                                       expected_source_and_weights=source, expected_face=face)

    def test_pcm_and_decoded_frame_contracts_fail_closed(self) -> None:
        row, audio, face, source, report = self._valid_inputs()
        bad_audio = {**report, "audio": {**report["audio"], "pcm_float32_sha256": "f" * 64}}
        with self.assertRaises(RuntimeError):
            ADAPTER.validate_renderer_input_report(row, audio_ref=audio, report=bad_audio,
                                                   expected_source_and_weights=source, expected_face=face)
        bad_clock = {**report, "silent_video": {**report["silent_video"],
                    "probe": {**report["silent_video"]["probe"], "frame_count": 144}}}
        with self.assertRaises(RuntimeError):
            ADAPTER.validate_renderer_input_report(row, audio_ref=audio, report=bad_clock,
                                                   expected_source_and_weights=source, expected_face=face)


if __name__ == "__main__":
    unittest.main()
