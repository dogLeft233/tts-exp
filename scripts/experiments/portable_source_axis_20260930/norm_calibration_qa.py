#!/usr/bin/env python3
"""CPU-only calibration geometry QA for the frozen direction/radius runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import latent_angle_radius as radius
from . import ops

EXPECTED_SUMMARY_SHA = "34876249bb3463417f903ea49e11da03306b1c0785a28cb6a4f97da92d59f22a"
EXPECTED_AXES_SHA = "e71b669a67f435d951742cf67a9683e0385b77f5b97ab7a7408851389ade9244"
EXPECTED_GEOMETRY_SHA = "800d0a0d040df2337249958c907b28eb13ef75173392778ce7c674eed1fce17e"
EXPECTED_RADIUS_RUNNER_SHA = "cdc2f922903cfbb35cc9e8d57e24a3963c590a8d8d72b99f93a9cf476406ceb0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return ops.array_sha256(np.asarray(value))


def summarize(value: np.ndarray) -> dict[str, float]:
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    return {
        "count": int(x.size), "min": float(x.min()), "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)), "max": float(x.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--calibration-axes", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--radius-runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.calibration_summary.resolve(strict=True)
    axes_path = args.calibration_axes.resolve(strict=True)
    geometry_path = args.geometry_audit.resolve(strict=True)
    runner_path = args.radius_runner.resolve(strict=True)
    expected_hashes = {
        "calibration_summary": (summary_path, EXPECTED_SUMMARY_SHA),
        "calibration_axes": (axes_path, EXPECTED_AXES_SHA),
        "geometry_audit": (geometry_path, EXPECTED_GEOMETRY_SHA),
        "radius_runner": (runner_path, EXPECTED_RADIUS_RUNNER_SHA),
    }
    observed_hashes = {name: sha256(path) for name, (path, _) in expected_hashes.items()}
    mismatch = {name: {"expected": expected, "actual": observed_hashes[name]}
                for name, (_, expected) in expected_hashes.items() if observed_hashes[name] != expected}
    if mismatch:
        raise ValueError(f"frozen input/source hash mismatch: {mismatch}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "calibration_complete" or summary.get("global_anchor", {}).get("selected_k") != 3:
        raise ValueError("CAL summary status or frozen anchor is invalid")
    with np.load(axes_path, allow_pickle=False) as archive:
        d = np.asarray(archive["d_source"], dtype=np.float64)
        u = np.asarray(archive["u_level"], dtype=np.float64)
    if array_digest(d) != summary["axes"]["d_sha256"] or array_digest(u) != summary["axes"]["u_sha256"]:
        raise ValueError("CAL axis arrays do not match the frozen summary")

    feature_root = args.calibration_features.resolve(strict=True)
    feature_files = sorted(feature_root.glob("*.npz"))
    if len(feature_files) != 20:
        raise ValueError(f"expected 20 CAL feature archives, found {len(feature_files)}")
    speaker_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    all_q10_radius_errors, all_q01_radius_errors = [], []
    for path in feature_files:
        sample_id = path.stem
        speaker = sample_id.split("_", 1)[0]
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        with np.load(path, allow_pickle=False) as archive:
            z0 = np.asarray(archive["natural_z_fixed"], dtype=np.float32)
            mask = np.asarray(archive["natural_speech_mask_fixed"], dtype=np.uint8).astype(bool)
        z1, direction_meta = ops.apply_source_direction(z0, mask, d, u, sign=1, level_orthogonal=True)
        cells, radial_meta = radius._build_factorizations(z0, z1, u, mask)
        r0 = radius._per_frame_norm(z0)
        r1 = radius._per_frame_norm(z1)
        r_q10 = radius._per_frame_norm(cells["q10"])
        r_q01 = radius._per_frame_norm(cells["q01"])
        q10_error = np.abs(r_q10 - r0) / r0
        q01_error = np.abs(r_q01 - r1) / r1
        if not np.array_equal(cells["q00"], z0) or not np.array_equal(cells["q11"], z1):
            raise AssertionError(f"CAL endpoint identity failed for {sample_id}")
        if float(q10_error.max(initial=0.0)) > 1e-6 or float(q01_error.max(initial=0.0)) > 1e-6:
            raise AssertionError(f"CAL radius identity failed for {sample_id}")
        if any(not np.isfinite(value).all() or np.any(value < 0) for value in cells.values()):
            raise AssertionError(f"CAL factorization produced invalid latents for {sample_id}")
        all_q10_radius_errors.extend(q10_error.tolist())
        all_q01_radius_errors.extend(q01_error.tolist())
        rows.append({
            "sample_id": sample_id, "speaker_id": speaker, "feature_npz": str(path),
            "feature_npz_sha256": sha256(path), "frames": int(len(z0)),
            "fixed_mask_active_frames": int(np.count_nonzero(mask)),
            "calibration_direction": direction_meta,
            "factorization": radial_meta,
        })
    if len(speaker_counts) != 10 or set(speaker_counts.values()) != {2}:
        raise ValueError(f"CAL cohort must have 10 speakers x 2 clips: {speaker_counts}")

    output = {
        "schema_version": 1,
        "status": "passed",
        "classification": "CAL-only CPU geometry/identity QA; no new model forwards and no EVAL outcomes read",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "calibration_anchor": {"k": 3, "column": 18, "official_syncnet_offset": -3},
        "frozen_inputs": {
            name: {"path": str(path), "sha256": observed_hashes[name]}
            for name, (path, _) in expected_hashes.items()
        },
        "qa_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "calibration_features": {"directory": str(feature_root), "count": len(feature_files), "speakers": speaker_counts,
                                 "files": [{"path": str(path), "sha256": sha256(path)} for path in feature_files]},
        "calibration_axes": {"d_sha256": array_digest(d), "u_sha256": array_digest(u),
                             "d_norm": float(np.linalg.norm(d)), "u_norm": float(np.linalg.norm(u))},
        "numeric_qa": {
            "endpoint_copies_bit_exact": True,
            "all_arrays_finite_nonnegative": True,
            "q10_radius_relative_error": summarize(np.asarray(all_q10_radius_errors)),
            "q01_radius_relative_error": summarize(np.asarray(all_q01_radius_errors)),
            "radius_tolerance": 1e-6,
            "all_rows_passed": len(rows) == 20,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "clips": len(rows), "speakers": len(speaker_counts),
                      "q10_radius_error_max": output["numeric_qa"]["q10_radius_relative_error"]["max"],
                      "q01_radius_error_max": output["numeric_qa"]["q01_radius_relative_error"]["max"],
                      "sha256": sha256(args.out)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
