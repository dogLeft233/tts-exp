from __future__ import annotations

import importlib
import json
import math
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "scripts/experiments/cross_tfg_audio_head_20261003"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

snapshots = importlib.import_module("fit40_training_snapshots_attempt02")
builder = importlib.import_module("build_fit40_snapshot_stagea_plan_attempt02")


def _fit_rows():
    rows = []
    for ordinal, (speaker, canonical_j) in enumerate(snapshots.FIT_PAIRS):
        sample_id = f"sample_{ordinal:02d}"
        rows.append({
            "row_id": f"{speaker}_j{canonical_j:02d}_{sample_id}",
            "sample_id": sample_id,
            "speaker_id": speaker,
            "canonical_j": canonical_j,
            "natural_pcm": {
                "sample_count": 64000 + ordinal,
                "sample_rate_hz": 16000,
                "channels": 1,
                "file_sha256": f"{ordinal + 1:064x}",
                "pcm_float32_sha256": f"{ordinal + 101:064x}",
            },
        })
    return rows


def _sealed(payload, field):
    return {**payload, field: snapshots.sha256_bytes(snapshots.canonical_bytes(payload))}


def _snapshot_index(tmp_path):
    profile = snapshots.PROFILES["fit40_native_attempt11"]
    fit_roster = _fit_rows()
    artifact_roles = {
        "checkpoint": "checkpoint", "training_record": "training_record",
        "runtime_source_pins": "runtime_source_pins", "training_steps": "training_steps",
        "strict_backward_qualification": "strict_backward_qualification", "completion": "completion",
        "status": "status",
    }
    artifact_pins = {
        role: {
            "path": f"/source/{role}.json",
            "bytes": 100 + index,
            "sha256": (profile["head_module"]["sha256"] if role == "head_module" else f"{index + 1:064x}"),
        }
        for index, role in enumerate(artifact_roles)
    }
    for role, expected in {
        "source_manifest": snapshots.STATIC_INPUTS["source_manifest"],
        "geometry_manifest": snapshots.STATIC_INPUTS["geometry_manifest"],
        "head_config": snapshots.STATIC_INPUTS["head_config"],
        "parent_plan": snapshots.STATIC_INPUTS["parent_plan"],
        "runner": profile["runner"],
        "head_module": profile["head_module"],
    }.items():
        artifact_pins[role] = {"path": f"/source/{role}.json", **expected}
    module_pins = {}
    for name, static_role in {
        "audio_io": "audio_io", "baselines": "baselines", "enhance_v3_cli": "enhance_v3_cli",
        "fit40_geometry_producer": "fit40_geometry_producer", "v2_runtime": "v2_runtime",
        "v4_runtime": "v4_runtime",
    }.items():
        expected = snapshots.STATIC_INPUTS[static_role]
        module_pins[name] = {"path": f"/source/{name}.py", "bytes": expected["bytes"], "sha256": expected["sha256"]}
    module_pins["v3_head"] = {"path": "/source/head.py", **profile["head_module"]}
    module_pins["v3_v4_adapter"] = {"path": "/source/adapter.py", **profile["v3_v4_adapter"]}
    artifact_pins["training_modules"] = module_pins
    entries = []
    for step in snapshots.SNAPSHOT_STEPS:
        source_fit = fit_roster[(step - 1) % 40]
        source_step = {
            "step": step,
            "row_id": source_fit["row_id"],
            "sample_id": source_fit["sample_id"],
            "speaker_id": source_fit["speaker_id"],
            "canonical_j": source_fit["canonical_j"],
            "source_pcm_float32_sha256": source_fit["natural_pcm"]["pcm_float32_sha256"],
            "theta_after_update": [[float(step)] * 3 for _ in range(3)],
        }
        values = [[float(step)] * 3 for _ in range(3)]
        parameter = {
            "name": "band_contrasts_db",
            "dtype": "float32",
            "shape": [3, 3],
            "temporal_components": ["static", "positive_flux", "negative_flux"],
            "non_reference_band_order": [0, 1, 2],
            "values": values,
            "little_endian_f32_sha256": snapshots.sha256_bytes(snapshots._theta_bytes(values)),
        }
        record = _sealed({
            "schema": "cross_tfg_fit40_parameter_snapshot_v1",
            "snapshot_id": f"{profile['family_id']}_source_step{step:03d}",
            "method_id": f"{profile['method_prefix']}_step{step:03d}",
            "training_family": profile["family_id"],
            "source_optimizer_step": step,
            "selection_status": "NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN",
            "parameter": parameter,
            "head_schema": profile["head_schema"],
            "head_constructor_sha256": snapshots.STATIC_INPUTS["head_config"]["constructor_sha256"],
            "head_implementation": artifact_pins["head_module"],
            "source_checkpoint": artifact_pins["checkpoint"],
            "source_training_record": artifact_pins["training_record"],
            "source_training_steps": artifact_pins["training_steps"],
            "source_completion": artifact_pins["completion"],
            "source_runtime_pins": artifact_pins["runtime_source_pins"],
            "source_fit_manifest": artifact_pins["source_manifest"],
            "source_geometry_manifest": artifact_pins["geometry_manifest"],
            "source_parent_plan": artifact_pins["parent_plan"],
            "source_training_step_record": source_step,
            "training_step_record_sha256": snapshots.sha256_bytes(snapshots.canonical_bytes(source_step)),
        }, "snapshot_sha256")
        path = tmp_path / f"snapshot_step{step:03d}.json"
        raw = json.dumps(record, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
        path.write_bytes(raw)
        entries.append({
            "snapshot_id": record["snapshot_id"],
            "method_id": record["method_id"],
            "source_optimizer_step": step,
            "path": path.name,
            "bytes": len(raw),
            "sha256": snapshots.sha256_bytes(raw),
            "snapshot_sha256": record["snapshot_sha256"],
        })
    index = _sealed({
        "schema": "cross_tfg_fit40_snapshot_set_index_v1",
        "status": "EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND",
        "snapshot_extractor_source": snapshots._repo_source_pin(Path(snapshots.__file__)),
        "training_family": profile["family_id"],
        "training_artifact_pins": artifact_pins,
        "training_counts": {
            "completed_optimizer_steps": 120,
            "source_fit_rows": 40,
            "fit_speakers": 20,
            "passes": 3,
            "snapshot_steps": list(snapshots.SNAPSHOT_STEPS),
        },
        "snapshots": entries,
        "native_endpoint": {
            "status": "UNBOUND_NATIVE_FIT_SCORES_NOT_RUN",
            "required_model_ids": list(snapshots.REQUIRED_MODEL_IDS),
            "required_fields": {"lag_offsets_frames": {"values": list(snapshots.LAG_OFFSETS)}},
        },
        "fit_roster": fit_roster,
    }, "index_sha256")
    path = tmp_path / "snapshot_set_index.json"
    raw = json.dumps(index, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    path.write_bytes(raw)
    return path, snapshots.sha256_bytes(raw)


def _base_fit_plan(index):
    methods = [
        {"method_id": "identity", "config": {"method": "identity"}},
        {"method_id": "rms_fixed", "config": {"method": "rms", "target_rms": 0.0376838172675746, "peak_ceiling": None}},
    ]
    models = [
        {"model_id": snapshots.REQUIRED_MODEL_IDS[0], "generator": {"native_entry_path": "/pinned/w2l.py", "native_entry_sha256": "a" * 64}},
        {"model_id": snapshots.REQUIRED_MODEL_IDS[1], "generator": {"contract": "cross_tfg_silent_video_cli_v1"}},
        {"model_id": snapshots.REQUIRED_MODEL_IDS[2], "generator": {"contract": "cross_tfg_silent_video_cli_v1"}},
    ]
    rows = []
    for source in index["fit_roster"]:
        natural = {
            "path": f"/canonical/{source['sample_id']}.wav",
            "sample_count": source["natural_pcm"]["sample_count"],
            "sample_rate_hz": 16000,
            "channels": 1,
            "file_sha256": source["natural_pcm"]["file_sha256"],
            "pcm_sha256": source["natural_pcm"]["pcm_float32_sha256"],
            "subtype": "IEEE_FLOAT_32_LE",
        }
        for model_id in snapshots.REQUIRED_MODEL_IDS:
            for method in methods:
                row = {
                    "row_id": f"{source['sample_id']}_{model_id}_{method['method_id']}",
                    "split": "fit",
                    "sample_id": source["sample_id"],
                    "speaker_id": source["speaker_id"],
                    "canonical_j": source["canonical_j"],
                    "model_id": model_id,
                    "method_id": method["method_id"],
                    "method_config_sha256": "b" * 64,
                    "score_audio": natural,
                    "natural_pcm": natural,
                    "seed": 20261003,
                    "face_id": "static",
                    "source_face": {"file_sha256": "c" * 64, "path": "/source/face.png"},
                    "source_video": {"file_sha256": "d" * 64, "path": "/source/portrait.mp4", "fps_num": 25, "fps_den": 1},
                    "generation_settings": {"fps_num": 25, "fps_den": 1},
                    "model_role": "candidate",
                    "audio_frontend": "native",
                }
                rows.append(row)
    return {
        "schema_version": "cross-tfg-audio-eval-v1",
        "status": "FROZEN",
        "methods": methods,
        "models": models,
        "rows": rows,
        "supports": [
            {"support_id": "full", "trim_each_end": 0, "minimum_queries": 1, "primary": True},
            {"support_id": "guard20", "trim_each_end": 20, "minimum_queries": 41, "primary": False},
        ],
        "protocol": {
            "audio": {"sample_rate_hz": 16000, "channels": 1, "dtype": "float32_le"},
            "syncnet": {
                "fps_num": 25,
                "fps_den": 1,
                "score_audio_estimand": "native_roi_original_pcm16",
                "lag_offsets_frames": list(range(-15, 16)),
                "zero_lag_column": 15,
                "anchor_column": 15,
            },
        },
    }


class FitSnapshotTests(unittest.TestCase):
    def test_training_family_id_resolves_profile_alias(self):
        profile = snapshots.profile_for_family_id("fit40_native_transition_head_v3_attempt11")
        self.assertEqual(profile["method_prefix"], "fit40_native_transition_head_v3_attempt11")
        with self.assertRaises(snapshots.SnapshotError):
            snapshots.profile_for_family_id("unknown-family")

    def test_theta_contract_is_finite_nine_parameter_float32(self):
        theta = [[0.0, 1.0, -1.0], [0.5, 0.25, -0.25], [2.0, -2.0, 0.0]]
        self.assertEqual(snapshots._normalize_theta(theta, "fixture"), theta)
        with self.assertRaises(snapshots.SnapshotError):
            snapshots._normalize_theta([[0.0] * 3] * 2, "bad-shape")
        with self.assertRaises(snapshots.SnapshotError):
            snapshots._normalize_theta([[math.nan] * 3 for _ in range(3)], "bad-value")


    def test_step_log_requires_all_ordered_rows_and_theta_continuity(self):
        rows = []
        for ordinal, (speaker, canonical_j) in enumerate(snapshots.FIT_PAIRS):
            rows.append({
                "row_id": f"{speaker}_{canonical_j}",
                "sample_id": f"sample_{ordinal}",
                "speaker_id": speaker,
                "canonical_j": canonical_j,
                "natural_pcm": {"pcm_float32_sha256": f"{ordinal + 1:064x}"},
            })
        profile = snapshots.PROFILES["fit40_native_attempt11"]
        records = []
        before = [[0.0] * 3 for _ in range(3)]
        optimizer = {"class": "Adam", "lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8}
        for step in range(1, 121):
            after = [[step / 1000.0] * 3 for _ in range(3)]
            row = rows[(step - 1) % 40]
            records.append({
                "step": step,
                "pass_index_zero_based": (step - 1) // 40,
                "row_index_zero_based": (step - 1) % 40,
                "row_id": row["row_id"],
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "canonical_j": row["canonical_j"],
                "source_pcm_float32_sha256": row["natural_pcm"]["pcm_float32_sha256"],
                "optimizer": optimizer,
                "gradient_nonzero_scalar_count": 1,
                "gradient_l2": 1.0,
                "gradient_max_abs": 1.0,
                "loss": 0.5,
                "theta_before_update": before,
                "theta_after_update": after,
            })
            before = after
        raw = b"".join(json.dumps(item, separators=(",", ":")).encode() + b"\n" for item in records)
        ordered, by_step = snapshots._verify_step_log(raw, rows, profile)
        self.assertEqual(len(ordered), 120)
        self.assertEqual(set(by_step), set(range(1, 121)))

        broken = list(records)
        broken[1] = {**broken[1], "theta_before_update": [[0.0] * 3 for _ in range(3)]}
        broken_raw = b"".join(json.dumps(item).encode() + b"\n" for item in broken)
        with self.assertRaisesRegex(snapshots.SnapshotError, "continuity"):
            snapshots._verify_step_log(broken_raw, rows, profile)
        with self.assertRaisesRegex(snapshots.SnapshotError, "exactly 120"):
            snapshots._verify_step_log(b"\n".join(raw.splitlines()[:-1]), rows, profile)


    def test_snapshot_index_binds_extractor_steps_roster_and_record_bytes(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            path, raw_sha = _snapshot_index(tmp_path)
            index, records = snapshots.verify_snapshot_index(path, raw_sha)
            self.assertEqual(index["training_counts"]["completed_optimizer_steps"], 120)
            self.assertEqual([item["source_optimizer_step"] for item in index["snapshots"]], [30, 60, 90, 120])
            self.assertEqual(len(records), 4)

            entry = index["snapshots"][1]
            record_path = tmp_path / entry["path"]
            record_path.write_bytes(record_path.read_bytes() + b" ")
            with self.assertRaisesRegex(snapshots.SnapshotError, "raw bytes"):
                snapshots.verify_snapshot_index(path, raw_sha)


    def test_base_plan_requires_exact_fit_roster_and_native_stagea_contract(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            index_path, index_sha = _snapshot_index(Path(directory))
            index, _ = snapshots.verify_snapshot_index(index_path, index_sha)
            plan = _base_fit_plan(index)
            rows, info = builder._validate_base_fit_plan(plan, index)
            self.assertEqual(len(rows), 240)
            self.assertEqual(len(info["contexts"]), 120)

            bad = json.loads(json.dumps(plan))
            bad["rows"][0]["split"] = "dev"
            with self.assertRaisesRegex(snapshots.SnapshotError, "non-Fit"):
                builder._validate_base_fit_plan(bad, index)

            bad = json.loads(json.dumps(plan))
            bad["protocol"]["syncnet"]["anchor_column"] = 14
            with self.assertRaisesRegex(snapshots.SnapshotError, "zero/anchor-15"):
                builder._validate_base_fit_plan(bad, index)


if __name__ == "__main__":
    unittest.main()
