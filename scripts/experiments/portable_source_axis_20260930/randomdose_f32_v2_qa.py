"""CPU-only actual-float32 random-dose QA against sealed retry3 contexts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_RUN = Path(
    "/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/run/evaluation_20260930_retry3"
)
PRIVATE_ROOT = Path("/root/tts-research-system-20261001")
QA_ROOT = PRIVATE_ROOT / "independent_validation/randomdose_f32_v2_qa_attempt01"
SOURCE_PACKAGE_ROOT = PRIVATE_ROOT / "code/randomdose_f32_v2_qa/src"
EXPECTED_SPEC_SHA256 = "679b11c8a67b18ac397d63b95900d55865b6dbf1ad7e8662bf3a5b2e915c8c79"
PACKAGE_NAME = "portable_source_axis_20260930"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _dose_rows(base: np.ndarray, target: np.ndarray, output: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    selected = np.flatnonzero(mask)
    rows: list[dict[str, Any]] = []
    for index in selected:
        target_norm = float(np.linalg.norm(target[index].astype(np.float64) - base[index].astype(np.float64)))
        actual_norm = float(np.linalg.norm(output[index].astype(np.float64) - base[index].astype(np.float64)))
        tolerance = max(1e-6, 1e-6 * target_norm)
        error = actual_norm - target_norm
        rows.append({
            "frame_index": int(index), "target_norm": target_norm, "actual_norm": actual_norm,
            "absolute_error": abs(error), "tolerance": tolerance,
            "tolerance_ratio": abs(error) / tolerance, "within_tolerance": abs(error) <= tolerance,
        })
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([row["absolute_error"] for row in rows], dtype=np.float64)
    ratios = np.asarray([row["tolerance_ratio"] for row in rows], dtype=np.float64)
    return {
        "active_frames": len(rows),
        "tolerance_miss_frames": int(sum(not row["within_tolerance"] for row in rows)),
        "max_abs_dose_error": float(np.max(errors, initial=0.0)),
        "p95_abs_dose_error": float(np.percentile(errors, 95)) if len(errors) else 0.0,
        "max_tolerance_ratio": float(np.max(ratios, initial=0.0)),
        "p95_tolerance_ratio": float(np.percentile(ratios, 95)) if len(ratios) else 0.0,
    }


def main() -> int:
    sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))
    package = __import__(PACKAGE_NAME, fromlist=["ops", "randomdose_f32_v2"])
    from_package = __import__(f"{PACKAGE_NAME}.randomdose_f32_v2", fromlist=["apply_random_direction_f32_dose_v2"])
    ops = package.ops
    apply_v2 = from_package.apply_random_direction_f32_dose_v2

    if QA_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite QA attempt: {QA_ROOT}")
    QA_ROOT.mkdir(parents=True, mode=0o700)
    contexts = sorted((SOURCE_RUN / "samples").glob("*_context.npz"))
    if len(contexts) != 80:
        raise RuntimeError(f"expected 80 sealed retry3 contexts, found {len(contexts)}")
    spec_path = Path("/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/confirm-random-specificity-20261001_spec.md")
    if not spec_path.is_file() or _sha(spec_path) != EXPECTED_SPEC_SHA256:
        raise RuntimeError("random specificity frozen spec SHA mismatch")

    script_path = Path(__file__).resolve()
    helper_path = Path(from_package.__file__).resolve()
    ops_path = Path(package.ops.__file__).resolve()
    seal = {
        "schema_version": 1,
        "status": "running",
        "classification": "CPU-only numerical QA; no audio/video forward and no score/effect summaries",
        "source_run": str(SOURCE_RUN),
        "source_context_count": len(contexts),
        "source_contexts": [{"path": str(path), "sha256": _sha(path)} for path in contexts],
        "source_protocol_sha256": _sha(SOURCE_RUN / "evaluation_protocol.json"),
        "source_rows_sha256": _sha(SOURCE_RUN / "evaluation_rows.jsonl"),
        "source_results_sha256": _sha(SOURCE_RUN / "evaluation_results.jsonl"),
        "source_code_sha256": {
            "randomdose_f32_v2_qa.py": _sha(script_path),
            "randomdose_f32_v2.py": _sha(helper_path),
            "ops.py": _sha(ops_path),
        },
        "spec_path": str(spec_path),
        "spec_sha256": _sha(spec_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": _utc_now(),
    }
    _atomic_json(QA_ROOT / "qa_seal.json", seal)
    rows_path = QA_ROOT / "dose_qa_rows.jsonl"
    status_path = QA_ROOT / "run_status.json"
    _atomic_json(status_path, {"status": "running", "completed_samples": 0, "planned_samples": 80, "completed_arms": 0, "planned_arms": 240})

    started = time.perf_counter()
    per_arm: dict[str, list[dict[str, Any]]] = {f"random{i}": [] for i in range(1, 4)}
    old_aggregate: dict[str, list[dict[str, Any]]] = {f"random{i}": [] for i in range(1, 4)}
    scales_by_arm: dict[str, list[float]] = {f"random{i}": [] for i in range(1, 4)}
    capped_by_arm: dict[str, int] = {f"random{i}": 0 for i in range(1, 4)}
    failures: list[dict[str, Any]] = []
    completed_samples = completed_arms = 0
    for sample_index, path in enumerate(contexts):
        sample_id = path.name.removesuffix("_context.npz")
        context_sha = _sha(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                base = archive["axis_z_baseline_float32"]
                orth_plus = archive["axis_z_orth_plus_float32"]
                mask = archive["natural_speech_mask_fixed_uint8"].astype(bool)
                u = archive["calibration_u_level_float64"]
                inputs = {
                    "base_sha256": ops.array_sha256(base),
                    "orth_plus_sha256": ops.array_sha256(orth_plus),
                    "mask_sha256": ops.array_sha256(mask.astype(np.uint8)),
                    "u_sha256": ops.array_sha256(u),
                }
                for condition_index in range(1, 4):
                    condition = f"random{condition_index}"
                    direction = archive[f"axis_direction_{condition}_float64"]
                    old_output = archive[f"axis_z_{condition}_float32"]
                    output, diagnostics = apply_v2(base, mask, direction, u, orth_plus)
                    new_frame_rows = _dose_rows(base, orth_plus, output, mask)
                    old_frame_rows = _dose_rows(base, orth_plus, old_output, mask)
                    if len(diagnostics["frames"]) != int(mask.sum()):
                        raise AssertionError(f"active frame diagnostics lost for {sample_id}/{condition}")
                    if not np.array_equal(output[~mask], base[~mask]):
                        raise AssertionError(f"inactive frames changed for {sample_id}/{condition}")
                    if np.any(output < 0) or not np.isfinite(output).all():
                        raise AssertionError(f"invalid corrected latent for {sample_id}/{condition}")
                    if diagnostics["direction_sha256"] != ops.array_sha256(direction):
                        raise AssertionError(f"random direction identity changed for {sample_id}/{condition}")
                    row = {
                        "sample_index": sample_index,
                        "sample_id": sample_id,
                        "condition": condition,
                        "source_context_path": str(path),
                        "source_context_sha256": context_sha,
                        "input_arrays": inputs,
                        "direction_sha256": ops.array_sha256(direction),
                        "old_output_sha256": ops.array_sha256(old_output),
                        "corrected_output_sha256": ops.array_sha256(output),
                        "old_output_dose": _summary(old_frame_rows),
                        "corrected_output_dose": _summary(new_frame_rows),
                        "operation_diagnostics": diagnostics,
                        "recorded_at_utc": _utc_now(),
                    }
                    _append_jsonl(rows_path, row)
                    per_arm[condition].extend(new_frame_rows)
                    old_aggregate[condition].extend(old_frame_rows)
                    scales_by_arm[condition].extend(float(frame["scale"]) for frame in diagnostics["frames"])
                    capped_by_arm[condition] += int(diagnostics["capped_infeasible_frames"])
                    completed_arms += 1
            completed_samples += 1
            _atomic_json(status_path, {
                "status": "running", "completed_samples": completed_samples, "planned_samples": 80,
                "completed_arms": completed_arms, "planned_arms": 240,
                "failed_samples": len(failures), "last_sample_id": sample_id,
            })
        except Exception as exc:
            failure = {"sample_index": sample_index, "sample_id": sample_id, "context_sha256": context_sha,
                       "error_type": type(exc).__name__, "error": str(exc)[:2000], "recorded_at_utc": _utc_now()}
            failures.append(failure)
            _append_jsonl(QA_ROOT / "failures.jsonl", failure)
            _atomic_json(status_path, {
                "status": "running", "completed_samples": completed_samples, "planned_samples": 80,
                "completed_arms": completed_arms, "planned_arms": 240,
                "failed_samples": len(failures), "last_sample_id": sample_id,
            })

    final = {
        **seal,
        "status": "complete" if not failures and completed_samples == 80 and completed_arms == 240 else "complete_with_failures",
        "completed_samples": completed_samples,
        "failed_samples": len(failures),
        "completed_arms": completed_arms,
        "planned_arms": 240,
        "elapsed_seconds": time.perf_counter() - started,
        "dose_qa_by_arm": {
            condition: {
                "old_output": _summary(old_aggregate[condition]),
                "corrected_output": _summary(per_arm[condition]),
                "corrected_scale_min_median_p95_max": [
                    float(np.min(scales_by_arm[condition])) if scales_by_arm[condition] else 0.0,
                    float(np.median(scales_by_arm[condition])) if scales_by_arm[condition] else 0.0,
                    float(np.percentile(scales_by_arm[condition], 95)) if scales_by_arm[condition] else 0.0,
                    float(np.max(scales_by_arm[condition])) if scales_by_arm[condition] else 0.0,
                ],
                "corrected_scale_fraction_outside_half_to_two": float(np.mean(
                    (np.asarray(scales_by_arm[condition]) < 0.5)
                    | (np.asarray(scales_by_arm[condition]) > 2.0)
                )) if scales_by_arm[condition] else 0.0,
                "capped_infeasible_frames": capped_by_arm[condition],
            }
            for condition in per_arm
        },
        "failures": failures,
        "finished_at_utc": _utc_now(),
    }
    _atomic_json(QA_ROOT / "qa_summary.json", final)
    _atomic_json(status_path, final)
    inventory = {
        str(path.relative_to(QA_ROOT)): _sha(path)
        for path in sorted(QA_ROOT.rglob("*")) if path.is_file() and path.name != "all_files_sha256.json"
    }
    _atomic_json(QA_ROOT / "all_files_sha256.json", inventory)
    print(json.dumps({"status": final["status"], "completed_samples": completed_samples,
                      "completed_arms": completed_arms, "failed_samples": len(failures),
                      "qa_root": str(QA_ROOT)}, separators=(",", ":")), flush=True)
    return 0 if final["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
