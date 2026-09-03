from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.asr_sync_error_correlation.io import atomic_write_json, file_sha256, read_json
from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import mux_and_verify

from .config import CONDITIONS, SEED
from .evaluate import score_syncnet_cell


PROXY_ROOT = "06_candidate_proxy"
PROXY_SYNC_STAGE = f"{PROXY_ROOT}/sync"
PROXY_MEDIA_DIR = "media"
PROXY_MEDIA_RECORDS_DIR = "media_records"
PROXY_SYNC_RECORDS_DIR = "sync"
PROXY_ESTIMANDS = (
    "proxy_C_gain",
    "proxy_D_gain",
    "proxy_control_C_adv",
    "proxy_control_D_adv",
)


def _path(run_dir: Path, *parts: str) -> Path:
    return run_dir / PROXY_ROOT / Path(*parts)


def _candidate_audio_record(run_dir: Path, sample_id: str, condition: str) -> dict[str, Any]:
    return read_json(run_dir / "01_candidates" / "records" / sample_id / f"{condition}.json")


def _render_record(run_dir: Path, sample_id: str, condition: str) -> dict[str, Any]:
    return read_json(run_dir / "02_renders" / "records" / sample_id / f"{condition}.json")


def candidate_proxy_media_cell(
    *,
    run_dir: Path,
    sample_id: str,
    condition: str,
    rendered_video: Path,
    candidate_audio: Path,
    resume: bool = False,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unregistered condition: {condition}")
    output = _path(run_dir, PROXY_MEDIA_DIR, sample_id, f"{condition}.mkv").resolve()
    record_path = _path(run_dir, PROXY_MEDIA_RECORDS_DIR, sample_id, f"{condition}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if not rendered_video.is_file() or not candidate_audio.is_file():
        raise FileNotFoundError(f"proxy input missing for {sample_id}/{condition}")
    rendered_hash = file_sha256(rendered_video)
    candidate_hash = file_sha256(candidate_audio)
    if resume and output.is_file() and record_path.is_file():
        try:
            existing = read_json(record_path)
            contract = existing.get("proxy_contract", {})
            if (
                existing.get("copy_video_sha256") == rendered_hash
                and existing.get("candidate_driver_audio_sha256") == candidate_hash
                and existing.get("output_sha256") == file_sha256(output)
                and contract.get("audio_pcm_verified") is True
                and contract.get("video_stream_copy_verified") is True
            ):
                return {**existing, "resumed": True}
        except (OSError, ValueError, TypeError, KeyError):
            pass
    verification = mux_and_verify(source_video=rendered_video, expected_audio=candidate_audio, output_path=output)
    result = {
        "schema_version": 1,
        "endpoint": "candidate_driven_proxy",
        "sample_id": sample_id,
        "condition": condition,
        "copy_video": str(rendered_video),
        "copy_video_sha256": rendered_hash,
        "candidate_driver_audio": str(candidate_audio),
        "candidate_driver_audio_sha256": candidate_hash,
        "output_path": str(output),
        "output_sha256": file_sha256(output),
        "proxy_contract": verification,
        "resumed": False,
    }
    atomic_write_json(record_path, result)
    return result


def _compute_proxy_estimands(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    required = set(CONDITIONS)
    if set(scores) != required:
        raise ValueError(f"proxy score conditions differ from frozen contract: {sorted(scores)}")
    natural_c = float(scores["natural"]["sync_c"])
    targeted_c = float(scores["asr_targeted"]["sync_c"])
    natural_d = float(scores["natural"]["sync_d"])
    targeted_d = float(scores["asr_targeted"]["sync_d"])
    control_c = float(np.mean([float(scores[name]["sync_c"]) for name in ("target_control_0", "target_control_1")]))
    control_d = float(np.mean([float(scores[name]["sync_d"]) for name in ("target_control_0", "target_control_1")]))
    values = {
        "proxy_C_gain": targeted_c - natural_c,
        "proxy_D_gain": natural_d - targeted_d,
        "proxy_control_C_adv": targeted_c - control_c,
        "proxy_control_D_adv": control_d - targeted_d,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("proxy estimand is non-finite")
    return {name: float(value) for name, value in values.items()}


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no complete proxy rows")
    units = [str(row["unit_id"]) for row in rows]
    if len(set(units)) != len(units):
        raise ValueError("proxy source_group unit is not unique")
    estimands: dict[str, Any] = {}
    for name in PROXY_ESTIMANDS:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite proxy estimand: {name}")
        estimands[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "wins_gt_zero": int(np.count_nonzero(values > 0)),
            "wins_ge_zero": int(np.count_nonzero(values >= 0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {"count": len(rows), "unit_id": "source_group", "units": units, "estimands": estimands}


def _bootstrap(rows: Sequence[Mapping[str, Any]], *, draws: int = 10000, seed: int = SEED) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError("cannot bootstrap empty proxy rows")
    units = [str(row["unit_id"]) for row in rows]
    if len(set(units)) != len(units):
        raise ValueError("proxy bootstrap requires one row per source group")
    values = {name: np.asarray([float(row[name]) for row in rows], dtype=np.float64) for name in PROXY_ESTIMANDS}
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("proxy bootstrap contains non-finite values")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    indices = rng.integers(0, len(rows), size=(int(draws), len(rows)))
    result: dict[str, dict[str, Any]] = {}
    for name, array in values.items():
        medians = np.median(array[indices], axis=1)
        lower, upper = np.quantile(medians, [0.025, 0.975], method="linear")
        result[name] = {
            "observed_median": float(np.median(array)),
            "lower": float(lower),
            "upper": float(upper),
            "draws": int(draws),
            "seed": int(seed),
            "unit": "source_group",
            "percentile_method": "linear",
        }
    return result


def _load_strict_reference(run_dir: Path) -> dict[str, Any]:
    validation = read_json(run_dir / "validation.json")
    decision = read_json(run_dir / "decision.json")
    summary = read_json(run_dir / "summary.json")
    if validation.get("status") != "GO" or validation.get("sealed_splits_accessed") is not False:
        raise RuntimeError("strict prototype validation is not GO or records sealed access")
    if decision.get("engineering") != "GO":
        raise RuntimeError("strict prototype engineering decision is not GO")
    return {
        "decision": {
            "path": str((run_dir / "decision.json").resolve()),
            "sha256": file_sha256(run_dir / "decision.json"),
            "engineering": decision.get("engineering"),
            "science": decision.get("science"),
        },
        "summary": {
            "path": str((run_dir / "summary.json").resolve()),
            "sha256": file_sha256(run_dir / "summary.json"),
            "estimands": summary.get("analysis", {}).get("estimands", {}),
            "bootstrap": summary.get("bootstrap", {}),
        },
    }


def run_proxy(*, run_dir: Path, repo_root: Path, device: str, resume: bool = False) -> bool:
    strict_reference = _load_strict_reference(run_dir)
    candidates = read_json(run_dir / "01_candidates" / "manifest.json")
    lock = read_json(run_dir / "00_lock" / "lock.json")
    frozen_by_id = {str(row["sample_id"]): row for row in lock["frozen_inputs"]["samples"]}
    strict_analysis = read_json(run_dir / "05_analysis" / "analysis.json")
    strict_rows = {str(row["sample_id"]): row for row in strict_analysis["rows"]}
    samples = list(candidates["samples"])
    proxy_dir = run_dir / PROXY_ROOT
    proxy_dir.mkdir(parents=True, exist_ok=True)
    media_failures: list[dict[str, str]] = []
    media_records: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        media_records[sample_id] = {}
        for condition in CONDITIONS:
            try:
                render = _render_record(run_dir, sample_id, condition)
                audio = _candidate_audio_record(run_dir, sample_id, condition)
                media_records[sample_id][condition] = candidate_proxy_media_cell(
                    run_dir=run_dir,
                    sample_id=sample_id,
                    condition=condition,
                    rendered_video=Path(render["video_path"]),
                    candidate_audio=Path(audio["output_path"]),
                    resume=resume,
                )
            except Exception as exc:
                media_failures.append({"sample_id": sample_id, "condition": condition, "error": f"{type(exc).__name__}: {exc}"})
    if media_failures:
        atomic_write_json(proxy_dir / "media_failures.json", media_failures)
        atomic_write_json(proxy_dir / "decision.json", {"schema_version": 1, "endpoint": "candidate_driven_proxy", "status": "NO_GO", "stage": "media", "successful_cells": 4 * len(samples) - len(media_failures), "expected_cells": 4 * len(samples)})
        return False
    atomic_write_json(proxy_dir / "media_manifest.json", {"schema_version": 1, "endpoint": "candidate_driven_proxy", "conditions": list(CONDITIONS), "sample_count": len(samples), "media_records": media_records})

    sync_failures: list[dict[str, str]] = []
    sync_records: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        sync_records[sample_id] = {}
        natural_sync: Mapping[str, Any] | None = None
        for condition in CONDITIONS:
            try:
                patches = next(row["patch_blocks"] for row in sample["conditions"] if row["condition"] == condition)
                result = score_syncnet_cell(
                    repo_root=repo_root,
                    run_dir=run_dir,
                    sample_id=sample_id,
                    condition=condition,
                    replacement_media=Path(media_records[sample_id][condition]["output_path"]),
                    natural_sync_record=natural_sync,
                    patches=patches,
                    audio_duration_s=float(frozen_by_id[sample_id]["arms"]["natural"]["audio_duration_s"]),
                    device=device,
                    resume=resume,
                    sync_stage=PROXY_SYNC_STAGE,
                    media_label="candidate_proxy_media",
                    log_stage="candidate_proxy_sync",
                )
                sync_records[sample_id][condition] = result
                if condition == "natural":
                    natural_sync = result
            except Exception as exc:
                sync_failures.append({"sample_id": sample_id, "condition": condition, "error": f"{type(exc).__name__}: {exc}"})
    if sync_failures:
        atomic_write_json(proxy_dir / "sync_failures.json", sync_failures)
        atomic_write_json(proxy_dir / "decision.json", {"schema_version": 1, "endpoint": "candidate_driven_proxy", "status": "NO_GO", "stage": "sync", "successful_cells": 4 * len(samples) - len(sync_failures), "expected_cells": 4 * len(samples)})
        return False

    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        records = sync_records[sample_id]
        natural = records["natural"]
        track_keys = ("selected_track_index", "selected_frame_count", "track_start_frame", "track_end_frame", "track_ranges")
        if any(records[condition].get(key) != natural.get(key) for condition in CONDITIONS for key in track_keys):
            raise ValueError(f"proxy paired track mismatch: {sample_id}")
        scores = {condition: records[condition]["official"] for condition in CONDITIONS}
        row = {"sample_id": sample_id, "unit_id": str(sample["source_group"]), "source_group": str(sample["source_group"]), **_compute_proxy_estimands(scores)}
        row["condition_scores"] = {condition: dict(scores[condition]) for condition in CONDITIONS}
        strict_match = strict_rows[sample_id]
        row["strict_replacement_estimands"] = {name: strict_match[name] for name in ("baseline_C_gain", "baseline_D_gain", "control_C_adv", "control_D_adv")}
        rows.append(row)

    payload = {
        "schema_version": 1,
        "endpoint": "candidate_driven_proxy",
        "claim_boundary": "descriptive candidate-audio/video co-adaptation proxy; not replacement-safe visual evidence",
        "unit": "source_group",
        "rows": rows,
        "summary": _summarize(rows),
        "bootstrap": _bootstrap(rows),
        "strict_replacement_reference": strict_reference,
        "sealed_splits_accessed": False,
    }
    atomic_write_json(proxy_dir / "analysis.json", payload)
    atomic_write_json(proxy_dir / "summary.json", {"schema_version": 1, "endpoint": payload["endpoint"], "claim_boundary": payload["claim_boundary"], "analysis": payload["summary"], "bootstrap": payload["bootstrap"], "complete_samples": len(rows), "source_group_count": len({str(row["unit_id"]) for row in rows}), "sealed_splits_accessed": False, "strict_replacement_reference": strict_reference})
    atomic_write_json(proxy_dir / "decision.json", {"schema_version": 1, "endpoint": "candidate_driven_proxy", "status": "COMPLETE_DESCRIPTIVE_ONLY", "science": "NOT_A_REPLACEMENT_DECISION", "complete_units": len(rows), "strict_science_decision_unchanged": strict_reference["decision"]["science"], "claim_boundary": payload["claim_boundary"]})
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/lrs3_asr_targeted_local_replacement_prototype_20260901"))
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.repo_root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    args.run_dir = (args.run_dir if args.run_dir.is_absolute() else args.repo_root / args.run_dir).resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return 0 if run_proxy(run_dir=args.run_dir, repo_root=args.repo_root, device=args.device, resume=args.resume) else 1
    except Exception as exc:
        print(f"candidate-driven proxy failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
