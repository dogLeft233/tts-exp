"""Execute the masked-TTS natural-slot signal-path diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

import numpy as np
import soundfile as sf
import torch

from scripts.experiments.masked_tts_reconstruction.config import (
    MEL_BINS,
    SEEDS,
    WAVLM_DIM,
)
from scripts.experiments.masked_tts_reconstruction.evaluate import array_hash, score_condition
from scripts.experiments.masked_tts_reconstruction.features import (
    build_example,
    phone_phase_linear,
    standardize_tts,
)
from scripts.experiments.masked_tts_reconstruction.model import MaskedNaturalReconstructor
from scripts.experiments.masked_tts_reconstruction.protocol import (
    canonical_json,
    normalize_phone,
    sha256_file,
    sha256_text,
)
from scripts.experiments.masked_tts_reconstruction.run import _load_features, _load_mels
from scripts.experiments.masked_tts_reconstruction.train import load_checkpoint, set_deterministic_cpu
from scripts.experiments.masked_tts_tfg_probe.mel_drivers import MEL_MAX, MEL_MIN, patch_normalized_mel
from scripts.experiments.masked_tts_trajectory_specificity.confirm_new_records import (
    load_cohort,
    load_parent_donor_pool,
)
from scripts.experiments.masked_tts_trajectory_specificity.run import (
    _reverse_inside_core,
    render_controls_parallel,
    score_controls_parallel,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIRMATION = REPO / "runs/lrs3_masked_tts_new_confirmation_20260902"
DEFAULT_MODEL_RUN = REPO / "runs/lrs3_masked_tts_trajectory_specificity_20260902"
DEFAULT_FEATURE_RUN = REPO / "runs/lrs3_masked_tts_retention_exploratory_20260901"
DEFAULT_RUN = REPO / "runs/lrs3_masked_tts_natural_slot_signal_path_20260903"
NATURAL_SLOT = "NATURAL_WAVLM_IN_TTS_SLOT"
MATCHED_WRONG = "MATCHED_WRONG_RANK_1"
COMMON_PAIRED = "PAIRED_COMMON_MASKS"
EXPECTED_RECORDS = 16
EXPECTED_GROUPS = 8
EXPECTED_MASKS = 222
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260902
SAMPLE_RATE = 16_000
FRAME_STRIDE = 320


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_if_needed(path: Path, payload: Any, resume: bool) -> None:
    if path.exists() and resume:
        return
    write_json(path, payload)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {str(key): np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    if any(value.dtype.hasobject or not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"invalid NPZ arrays: {path}")
    return result


def _array_hash(value: np.ndarray) -> str:
    return array_hash(np.asarray(value, dtype=np.float32))


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    prefix = "bshall_knn-vc_"
    if root.name.startswith(prefix) and len(root.name) == len(prefix) + 40:
        return root.name[len(prefix):]
    raise ValueError(f"cannot resolve checkout revision: {root}")


def _bootstrap(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (EXPECTED_GROUPS,) or not np.isfinite(array).all():
        raise ValueError("bootstrap requires eight finite source-group values")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, EXPECTED_GROUPS, size=(BOOTSTRAP_DRAWS, EXPECTED_GROUPS))
    estimates = np.median(array[indices], axis=1)
    return {
        "median": float(np.median(array)),
        "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))],
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "unit": "whole_source_group",
        "method": "numpy_quantile_linear",
    }


def metric_summary(group_rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in group_rows]
    result = _bootstrap(values)
    result["positive_groups"] = int(sum(value > 0 for value in values))
    result["values"] = values
    result["pass"] = bool(result["ci95"][0] > 0 and result["positive_groups"] >= 7)
    return result


def _records_and_masks(confirmation: Path) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    _, records = load_cohort(confirmation)
    masks = read_json(confirmation / "03_data/mask_manifest.json")
    if masks.get("status") != "complete" or len(masks.get("masks", [])) != EXPECTED_MASKS:
        raise ValueError("confirmation mask manifest is incomplete")
    mask_map = {str(row["mask_sha256"]): row for row in masks["masks"]}
    if len(mask_map) != EXPECTED_MASKS:
        raise ValueError("duplicate confirmation mask hash")
    return records, mask_map


def _load_common(confirmation: Path, feature_run: Path) -> dict[str, Any]:
    records, masks = _records_and_masks(confirmation)
    natural = _load_npz(confirmation / "03_data/natural_mels.npz")
    new_tts = {}
    for record in records:
        sid = str(record["sample_id"])
        path = confirmation / "03_data/tts_features" / f"{sid}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != WAVLM_DIM or not np.isfinite(value).all():
            raise ValueError(f"invalid TTS feature array: {sid}")
        new_tts[sid] = value
    lock, parent_masks, parent_natural, parent_tts, stats = load_parent_donor_pool(feature_run)
    del parent_natural
    return {
        "records": records,
        "masks": masks,
        "natural": natural,
        "new_tts": new_tts,
        "parent_lock": lock,
        "parent_masks": parent_masks,
        "parent_tts": parent_tts,
        "stats": stats,
    }


def _parent_artifacts(confirmation: Path, model_run: Path, feature_run: Path) -> dict[str, Path]:
    files = {
        "cohort": confirmation / "00_cohort/manifest.json",
        "tts": confirmation / "01_tts/tts_meta.json",
        "alignment": confirmation / "02_alignment/alignment.json",
        "feature_manifest": confirmation / "03_data/feature_manifest.json",
        "mask_manifest": confirmation / "03_data/mask_manifest.json",
        "reconstruction": confirmation / "04_reconstruction/reconstruction.json",
        "drivers": confirmation / "05_drivers/drivers.json",
        "box_manifest": confirmation / "06_renders/box_manifest.json",
        "render_manifest": confirmation / "06_renders/render_manifest.json",
        "box_repairs": confirmation / "06_renders/box_repairs.json",
        "syncnet": confirmation / "05_syncnet/summary.json",
        "normalization": feature_run / "02_features/normalization.json",
    }
    for seed in SEEDS:
        files[f"checkpoint_{seed}"] = model_run / "04_training" / str(seed) / "hard_negative" / "checkpoint.pt"
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError([str(path) for path in files.values() if not path.is_file()])
    return files


def bind(run_dir: Path, confirmation: Path, model_run: Path, feature_run: Path, *, resume: bool) -> dict[str, Any]:
    path = run_dir / "00_binding/manifest.json"
    if path.exists() and resume:
        return read_json(path)
    files = _parent_artifacts(confirmation, model_run, feature_run)
    hub_root = Path(torch.hub.get_dir())
    checkout = hub_root / "bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358"
    checkpoint = hub_root / "checkpoints/WavLM-Large.pt"
    adapter_path = REPO / "scripts/wavlm_knn_vc_adapter.py"
    if not checkout.is_dir() or not checkpoint.is_file() or not adapter_path.is_file():
        raise FileNotFoundError("pinned local WavLM interface is unavailable")
    revision = _git_revision(checkout)
    expected_revision = "c616845c4e309e24d5927f15adbdf277a3d65358"
    if revision != expected_revision:
        raise ValueError(f"WavLM checkout revision changed: {revision}")
    records, masks = _records_and_masks(confirmation)
    cohort = read_json(files["cohort"])
    score = read_json(files["syncnet"])
    if len(records) != EXPECTED_RECORDS or len(cohort.get("groups", [])) != EXPECTED_GROUPS:
        raise ValueError("confirmation cohort counts changed")
    if int(score.get("score_count", -1)) != EXPECTED_RECORDS * len(SEEDS) * 4:
        raise ValueError("confirmation downstream matrix is incomplete")
    from scripts.wavlm_knn_vc_adapter import WavLMInterface
    interface = WavLMInterface().__dict__.copy()
    interface.update({
        "repository": "bshall/knn-vc",
        "revision": expected_revision,
        "sample_rate": SAMPLE_RATE,
        "frame_stride_samples": FRAME_STRIDE,
        "selected_layer": 6,
        "feature_dim": WAVLM_DIM,
        "vad_trigger_level": 0,
        "loudness_normalization": False,
        "waveform_policy": "complete_bound_natural_waveform_no_crop_pad_resample_or_amplitude_change",
        "phone_to_feature_span": "[round(start_s*16000/320), round(end_s*16000/320)) clipped to [0,feature_count]",
    })
    interface["wavlm_checkpoint_sha256"] = sha256_file(checkpoint)
    binding = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_natural_slot_signal_path_binding",
        "status": "complete",
        "confirmation_run": str(confirmation.resolve()),
        "model_run": str(model_run.resolve()),
        "feature_run": str(feature_run.resolve()),
        "files": {name: {"path": str(value.resolve()), "sha256": sha256_file(value)} for name, value in files.items()},
        "record_count": EXPECTED_RECORDS,
        "group_count": EXPECTED_GROUPS,
        "mask_count": len(masks),
        "seeds": list(SEEDS),
        "conditions": ["PAIRED_TTS", "SAME_PHONE_WRONG_INSTANCE", "WITHIN_PHONE_REVERSED", "NAT_ONLY"],
        "existing_downstream_cells": int(score["score_count"]),
        "adapter": {
            "path": str(adapter_path.resolve()),
            "sha256": sha256_file(adapter_path),
            "checkout": str(checkout.resolve()),
            "checkout_revision": revision,
            "checkout_source_sha256": _hash_tree(checkout),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "interface": interface,
        },
        "previously_inspected_records": True,
        "parent_artifacts_immutable": True,
        "sealed_splits_accessed": False,
    }
    write_json(path, binding)
    return binding


def _load_reconstruction_rows(confirmation: Path) -> list[Mapping[str, Any]]:
    result = read_json(confirmation / "04_reconstruction/reconstruction.json")
    rows = list(result.get("records", []))
    if result.get("status") != "complete" or len(rows) != EXPECTED_MASKS * len(SEEDS) * 4:
        raise ValueError("confirmation reconstruction matrix is incomplete")
    return rows


def _load_sync_rows(confirmation: Path) -> list[Mapping[str, Any]]:
    result = read_json(confirmation / "05_syncnet/summary.json")
    rows = list(result.get("scores", []))
    if result.get("status") != "complete" or len(rows) != EXPECTED_RECORDS * len(SEEDS) * 4:
        raise ValueError("confirmation SyncNet matrix is incomplete")
    return rows


def _aggregate_values(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], value: Callable[[Mapping[str, Any]], float], *, mask_level: bool) -> dict[str, Any]:
    by_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_seed[(str(row["sample_id"]), int(row["seed"]))].append(float(value(row)))
    if any(not values for values in by_seed.values()) or len(by_seed) != EXPECTED_RECORDS * len(SEEDS):
        raise ValueError("incomplete aggregation rows")
    per_seed = {key: float(np.median(values)) for key, values in by_seed.items()}
    record_rows = []
    for record in records:
        sid = str(record["sample_id"])
        values = [per_seed[(sid, int(seed))] for seed in SEEDS]
        record_rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "seed_values": values, "value": float(np.median(values))})
    group_rows = []
    seen: set[str] = set()
    for record in records:
        group = str(record["source_group"])
        if group in seen:
            continue
        seen.add(group)
        local = [row for row in record_rows if row["source_group"] == group]
        if len(local) != 2:
            raise ValueError(f"source group does not have two records: {group}")
        group_rows.append({"source_group": group, "record_count": 2, "value": float(np.median([row["value"] for row in local]))})
    del mask_level
    return {"record_rows": record_rows, "group_rows": group_rows, "summary": _bootstrap([row["value"] for row in group_rows])}


def _aggregate_named(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], metric: str, *, mask_level: bool) -> dict[str, Any]:
    result = _aggregate_values(records, rows, lambda row: float(row[metric]), mask_level=mask_level)
    for key in ("record_rows", "group_rows"):
        for row in result[key]:
            row[metric] = row.pop("value")
    result["summary"]["positive_groups"] = int(sum(row[metric] > 0 for row in result["group_rows"]))
    result["summary"]["pass"] = bool(result["summary"]["ci95"][0] > 0 and result["summary"]["positive_groups"] >= 7)
    return result


def correct_analysis(run_dir: Path, confirmation: Path, *, resume: bool) -> dict[str, Any]:
    path = run_dir / "01_correction/analysis.json"
    if path.exists() and resume:
        return read_json(path)
    records, _ = _records_and_masks(confirmation)
    rows = _load_reconstruction_rows(confirmation)
    output: dict[str, Any] = {}
    for condition in ("SAME_PHONE_WRONG_INSTANCE", "WITHIN_PHONE_REVERSED"):
        local = []
        for row in rows:
            if row["condition"] == condition:
                if "reconstruction_gain" not in row:
                    raise ValueError("stored reconstruction gain is missing")
                local.append({"sample_id": row["sample_id"], "seed": row["seed"], "mask_sha256": row["mask_sha256"], "gain": float(row["reconstruction_gain"])})
        output[condition] = _aggregate_named(records, local, "gain", mask_level=True)
    expected = {
        "SAME_PHONE_WRONG_INSTANCE": (0.14533, 0.06750, 0.17535),
        "WITHIN_PHONE_REVERSED": (0.04496, 0.03184, 0.06514),
    }
    checks = {}
    for condition, (median, low, high) in expected.items():
        summary = output[condition]["summary"]
        checks[condition] = {
            "median_abs_error": abs(summary["median"] - median),
            "ci_low_abs_error": abs(summary["ci95"][0] - low),
            "ci_high_abs_error": abs(summary["ci95"][1] - high),
            "positive_groups": summary["positive_groups"],
            "pass": bool(abs(summary["median"] - median) <= 1e-5 and abs(summary["ci95"][0] - low) <= 1e-5 and abs(summary["ci95"][1] - high) <= 1e-5 and summary["positive_groups"] == EXPECTED_GROUPS),
        }
    status = "complete" if all(row["pass"] for row in checks.values()) else "NOT_EVALUATED"
    result = {
        "schema_version": 1,
        "status": status,
        "part": "0",
        "pairing": "stored_same_mask_reconstruction_gain",
        "record_count": EXPECTED_RECORDS,
        "group_count": EXPECTED_GROUPS,
        "mask_count": EXPECTED_MASKS,
        "source_reconstruction_sha256": sha256_file(confirmation / "04_reconstruction/reconstruction.json"),
        "contrasts": output,
        "reproducibility_checks": checks,
        "rules": {"aggregation": "mask median, seed median, record median, two-record source-group median", "bootstrap": "10000 draws PCG64(20260902), NumPy linear quantiles", "prohibited_key": "sample_id + seed without mask_sha256"},
        "sealed_splits_accessed": False,
    }
    write_json(path, result)
    lines = ["# Part 0: corrected reconstruction analysis", "", f"Status: **{status}**.", "", "Controls are paired with their stored same-mask `reconstruction_gain`; parent artifacts were not modified.", "", "| Control | Median | 95% CI | Positive groups | Check |", "|---|---:|---|---:|---|"]
    for condition, check in checks.items():
        summary = output[condition]["summary"]
        lines.append(f"| {condition} | {summary['median']:.5f} | [{summary['ci95'][0]:.5f}, {summary['ci95'][1]:.5f}] | {summary['positive_groups']}/8 | {'pass' if check['pass'] else 'fail'} |")
    (run_dir / "01_correction/report.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "01_correction/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _load_prediction(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["prediction"], dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != MEL_BINS or not np.isfinite(value).all():
        raise ValueError(f"invalid prediction: {path}")
    return value


def _row_map(rows: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in keys)
        if key in result:
            raise ValueError(f"duplicate row identity: {key}")
        result[key] = row
    return result


def _old_control_examples(target: Mapping[str, Any], common: Mapping[str, Any], condition: str) -> dict[str, np.ndarray]:
    all_tts = {**common["parent_tts"], **common["new_tts"]}
    base = build_example(target, common["natural"], common["new_tts"], common["stats"])
    if condition == "PAIRED_TTS":
        return {key: np.asarray(base[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "tts_features", "target")}
    if condition == "NAT_ONLY":
        result = {key: np.asarray(base[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "tts_features", "target")}
        result["tts_features"] = np.zeros_like(result["tts_features"])
        return result
    raise ValueError(condition)


def _old_wrong_example(target: Mapping[str, Any], row: Mapping[str, Any], common: Mapping[str, Any]) -> dict[str, np.ndarray]:
    all_tts = {**common["parent_tts"], **common["new_tts"]}
    donor_hash = str(row["donor_source_mask_sha256"])
    donor_map = {str(item["mask_sha256"]): item for item in common["parent_masks"]["masks"]}
    donor = donor_map.get(donor_hash)
    if donor is None:
        raise ValueError(f"missing old donor mask: {donor_hash}")
    example = build_example(target, common["natural"], all_tts, common["stats"], tts_mask=donor)
    return {key: np.asarray(example[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "tts_features", "target_core", "target")}


def _trace_video_response(first: Path, second: Path, boxes_path: Path) -> tuple[float, int, tuple[int, int]]:
    import cv2
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    cap_a = cv2.VideoCapture(str(first))
    cap_b = cv2.VideoCapture(str(second))
    values = []
    frames = 0
    shape = None
    try:
        while True:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if ok_a != ok_b:
                raise ValueError("paired videos have different frame counts")
            if not ok_a:
                break
            if frame_a.shape != frame_b.shape:
                raise ValueError("paired videos have different frame shapes")
            if frames >= len(boxes):
                raise ValueError("face-box manifest is shorter than the video")
            x1, y1, x2, y2 = [int(value) for value in boxes[frames]]
            height, width = frame_a.shape[:2]
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
            midpoint = y1 + (y2 - y1) // 2
            if x2 <= x1 or y2 <= midpoint:
                raise ValueError("invalid lower-half face box")
            a = frame_a[midpoint:y2, x1:x2].astype(np.float32) / 255.0
            b = frame_b[midpoint:y2, x1:x2].astype(np.float32) / 255.0
            values.append(float(np.mean(np.abs(a - b))))
            frames += 1
            shape = (height, width)
    finally:
        cap_a.release()
        cap_b.release()
    if not values or shape is None:
        raise ValueError("empty rendered video")
    return float(np.mean(values)), frames, shape


def _descriptive_stage(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    return _aggregate_named(records, rows, metric, mask_level=False)


def trace(run_dir: Path, confirmation: Path, feature_run: Path, *, resume: bool) -> dict[str, Any]:
    path = run_dir / "02_trace/analysis.json"
    if path.exists() and resume:
        return read_json(path)
    common = _load_common(confirmation, feature_run)
    records, masks = common["records"], common["masks"]
    old_rows = _load_reconstruction_rows(confirmation)
    old_by_key = _row_map(old_rows, ("sample_id", "seed", "mask_sha256", "condition"))
    driver_manifest = read_json(confirmation / "05_drivers/drivers.json")
    driver_by_key = _row_map(driver_manifest["drivers"], ("sample_id", "seed", "condition"))
    render_manifest = read_json(confirmation / "06_renders/render_manifest.json")
    render_by_key = _row_map(render_manifest["renders"], ("sample_id", "seed", "condition"))
    scores = _load_sync_rows(confirmation)
    score_by_key = _row_map(scores, ("sample_id", "seed", "condition"))
    boxes_by_id = {str(row["sample_id"]): Path(str(row["boxes"])) for row in read_json(confirmation / "06_renders/box_manifest.json")["renders"]}
    condition_rows = {"SAME_PHONE_WRONG_INSTANCE": [], "WITHIN_PHONE_REVERSED": []}
    prediction_rows = {"SAME_PHONE_WRONG_INSTANCE": [], "WITHIN_PHONE_REVERSED": []}
    for record in records:
        sid = str(record["sample_id"])
        target_masks = [row for row in masks.values() if str(row["sample_id"]) == sid]
        for seed in SEEDS:
            for target in target_masks:
                mask_hash = str(target["mask_sha256"])
                paired_row = old_by_key[(sid, int(seed), mask_hash, "PAIRED_TTS")]
                paired = _old_control_examples(target, common, "PAIRED_TTS")
                core_start, core_end = int(target["core_start"]), int(target["core_end"])
                for condition in ("SAME_PHONE_WRONG_INSTANCE", "WITHIN_PHONE_REVERSED"):
                    control_row = old_by_key[(sid, int(seed), mask_hash, condition)]
                    if condition == "SAME_PHONE_WRONG_INSTANCE":
                        control = _old_wrong_example(target, control_row, common)
                    else:
                        control = {key: np.asarray(paired[key], dtype=np.float32) for key in paired}
                        control["tts_features"] = _reverse_inside_core(paired["tts_features"], core_start, core_end)
                    input_core = float(np.sqrt(np.mean((paired["tts_features"][core_start:core_end] - control["tts_features"][core_start:core_end]) ** 2)))
                    input_diff = np.diff(paired["tts_features"][core_start:core_end], axis=0) - np.diff(control["tts_features"][core_start:core_end], axis=0)
                    input_first = float(np.sqrt(np.mean(input_diff ** 2))) if input_diff.size else 0.0
                    paired_prediction = _load_prediction(confirmation / "04_reconstruction" / str(seed) / mask_hash / "PAIRED_TTS/prediction.npz")
                    control_prediction = _load_prediction(confirmation / "04_reconstruction" / str(seed) / mask_hash / f"{condition}/prediction.npz")
                    prediction_core = float(np.sqrt(np.mean((paired_prediction[core_start:core_end] - control_prediction[core_start:core_end]) ** 2)))
                    prediction_diff = np.diff(paired_prediction[core_start:core_end], axis=0) - np.diff(control_prediction[core_start:core_end], axis=0)
                    prediction_first = float(np.sqrt(np.mean(prediction_diff ** 2))) if prediction_diff.size else 0.0
                    prediction_rows[condition].append({"sample_id": sid, "source_group": record["source_group"], "seed": int(seed), "mask_sha256": mask_hash, "rms": prediction_core, "first_diff_rms": prediction_first})
                    condition_rows[condition].append({"sample_id": sid, "source_group": record["source_group"], "seed": int(seed), "mask_sha256": mask_hash, "rms": input_core, "first_diff_rms": input_first})
    stage_rows: dict[str, Any] = {"input_core": {}, "predicted_core": {}, "driver": {}, "rendered_response": {}, "endpoint": {}}
    for condition in condition_rows:
        stage_rows["input_core"][condition] = {
            "rms": _descriptive_stage(records, [{**row, "metric": row["rms"]} for row in condition_rows[condition]], "metric"),
            "first_diff_rms": _descriptive_stage(records, [{**row, "metric": row["first_diff_rms"]} for row in condition_rows[condition]], "metric"),
        }
        stage_rows["predicted_core"][condition] = {
            "rms": _descriptive_stage(records, [{**row, "metric": row["rms"]} for row in prediction_rows[condition]], "metric"),
            "first_diff_rms": _descriptive_stage(records, [{**row, "metric": row["first_diff_rms"]} for row in prediction_rows[condition]], "metric"),
        }
    for condition in condition_rows:
        driver_rows = []
        video_rows = []
        endpoint_rows = []
        for record in records:
            sid = str(record["sample_id"])
            target_masks = [row for row in masks.values() if str(row["sample_id"]) == sid]
            covered = np.zeros((int(common["natural"][sid].shape[1]),), dtype=bool)
            for target in target_masks:
                start = int(target["window_start_frame"]) + int(target["core_start"])
                end = int(target["window_start_frame"]) + int(target["core_end"])
                covered[start:end] = True
            for seed in SEEDS:
                paired_driver = np.asarray(np.load(driver_by_key[(sid, int(seed), "PAIRED_TTS")]["path"], allow_pickle=False), dtype=np.float32)
                control_driver = np.asarray(np.load(driver_by_key[(sid, int(seed), condition)]["path"], allow_pickle=False), dtype=np.float32)
                if paired_driver.shape != control_driver.shape or paired_driver.shape[1] != covered.shape[0]:
                    raise ValueError("driver shape mismatch")
                driver_rows.append({"sample_id": sid, "source_group": record["source_group"], "seed": int(seed), "metric": float(np.sqrt(np.mean((paired_driver[:, covered] - control_driver[:, covered]) ** 2)))})
                paired_render = render_by_key[(sid, int(seed), "PAIRED_TTS")]
                control_render = render_by_key[(sid, int(seed), condition)]
                if str(paired_render["face_sha256"]) != str(control_render["face_sha256"]):
                    raise ValueError("render face identity mismatch")
                response, frame_count, shape = _trace_video_response(Path(str(paired_render["video"])), Path(str(control_render["video"])), boxes_by_id[sid])
                video_rows.append({"sample_id": sid, "source_group": record["source_group"], "seed": int(seed), "metric": response, "frame_count": frame_count, "frame_shape": list(shape)})
                paired_score = score_by_key[(sid, int(seed), "PAIRED_TTS")]
                control_score = score_by_key[(sid, int(seed), condition)]
                endpoint_rows.append({"sample_id": sid, "source_group": record["source_group"], "seed": int(seed), "sync_c_gain": float(paired_score["sync_c"]) - float(control_score["sync_c"]), "sync_d_gain": float(control_score["sync_d"]) - float(paired_score["sync_d"])})
        stage_rows["driver"][condition] = _descriptive_stage(records, driver_rows, "metric")
        stage_rows["rendered_response"][condition] = _descriptive_stage(records, video_rows, "metric")
        stage_rows["endpoint"][condition] = {metric: _descriptive_stage(records, endpoint_rows, metric) for metric in ("sync_c_gain", "sync_d_gain")}
    ratios = {}
    for stage, values in stage_rows.items():
        ratios[stage] = {}
        reference = values.get("SAME_PHONE_WRONG_INSTANCE", {})
        for metric, reference_value in reference.items():
            if not isinstance(reference_value, dict) or "summary" not in reference_value:
                continue
            reversed_value = values.get("WITHIN_PHONE_REVERSED", {}).get(metric)
            if not isinstance(reversed_value, dict) or "summary" not in reversed_value:
                continue
            wrong = float(reference_value["summary"]["median"])
            reverse = float(reversed_value["summary"]["median"])
            ratios[stage][metric] = {"wrong_instance_magnitude": wrong, "reversed_magnitude": reverse, "ratio": None if reverse == 0 else abs(wrong) / abs(reverse)}
    result = {"schema_version": 1, "status": "complete", "part": "A", "recomputed_inference": False, "previously_inspected_records": True, "stages": stage_rows, "ratios": ratios, "rules": {"descriptive_only": True, "rgb_interpretation": "response magnitude including rendering and codec effects; not causal Wav2Lip localization"}, "sealed_splits_accessed": False}
    write_json(path, result)
    report = ["# Part A: existing signal-path audit", "", "Status: **complete**.", "", "No inference, rendering, replacement, or SyncNet scoring was rerun.", "", "| Stage | Metric | Wrong-instance median | Reversed median | Ratio |", "|---|---|---:|---:|---:|"]
    for stage, metrics in ratios.items():
        for metric, item in metrics.items():
            ratio = "n/a" if item["ratio"] is None else f"{item['ratio']:.5f}"
            report.append(f"| {stage} | {metric} | {item['wrong_instance_magnitude']:.5f} | {item['reversed_magnitude']:.5f} | {ratio} |")
    report.extend(["", "RGB differences are descriptive rendered-response magnitudes and do not identify Wav2Lip as a causal bottleneck.", ""])
    (run_dir / "02_trace/report.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "02_trace/report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def _natural_span(mask: Mapping[str, Any], feature_count: int, alignment: Mapping[str, Any]) -> tuple[int, int, Mapping[str, Any]]:
    sid = str(mask["sample_id"])
    row = next(item for item in alignment["records"] if str(item["sample_id"]) == sid)
    phones = list(row["natural_phones"])
    index = int(mask["natural_phone_index"])
    if index < 0 or index >= len(phones):
        raise ValueError(f"natural phone index out of range: {sid}/{index}")
    phone = phones[index]
    if normalize_phone(phone["phone"]) != normalize_phone(mask["label"]):
        raise ValueError(f"natural alignment label mismatch: {sid}/{index}")
    start = max(0, min(feature_count, int(round(float(phone["start"]) * SAMPLE_RATE / FRAME_STRIDE))))
    end = max(0, min(feature_count, int(round(float(phone["end"]) * SAMPLE_RATE / FRAME_STRIDE))))
    if end - start < 2:
        raise ValueError(f"natural phone WavLM span is too short: {sid}/{index}")
    return start, end, phone


def extract_natural_features(run_dir: Path, confirmation: Path, binding: Mapping[str, Any], *, device: str, resume: bool) -> dict[str, Any]:
    path = run_dir / "03_natural_features/manifest.json"
    if path.exists() and resume:
        return read_json(path)
    from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    adapter_info = binding["adapter"]
    adapter = WavLMKNNVCAdapter.load_pretrained(device=device, source=adapter_info["checkout"], revision=adapter_info["checkout_revision"])
    metadata = adapter.metadata()
    if metadata.get("revision") != adapter_info["checkout_revision"] or metadata.get("feature_dim") != WAVLM_DIM or metadata.get("selected_layer") != 6 or metadata.get("sample_rate") != SAMPLE_RATE or metadata.get("loudness_normalization") is not False:
        raise ValueError("loaded WavLM interface does not match binding")
    _, records = load_cohort(confirmation)
    rows = []
    for record in records:
        sid = str(record["sample_id"])
        audio = Path(str(record["natural_audio"]))
        if sha256_file(audio) != str(record["natural_audio_sha256"]):
            raise ValueError(f"natural audio hash changed: {sid}")
        waveform, sample_rate = sf.read(str(audio), dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if int(sample_rate) != SAMPLE_RATE or waveform.ndim != 1 or not np.isfinite(waveform).all() or waveform.shape[0] < 1024:
            raise ValueError(f"invalid natural waveform: {sid}")
        features = adapter.extract(torch.from_numpy(np.ascontiguousarray(waveform))).cpu().numpy().astype(np.float32, copy=False)
        if features.ndim != 2 or features.shape[1] != WAVLM_DIM or not np.isfinite(features).all():
            raise ValueError(f"invalid natural WavLM output: {sid}")
        feature_path = run_dir / "03_natural_features/wavlm" / f"{sid}.npy"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, features, allow_pickle=False)
        rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "audio": str(audio.resolve()), "audio_sha256": sha256_file(audio), "feature_path": str(feature_path.resolve()), "feature_sha256": sha256_file(feature_path), "feature_array_sha256": _array_hash(features), "shape": list(features.shape), "waveform_samples": int(waveform.shape[0]), "sample_rate": int(sample_rate)})
        print(f"natural WavLM {len(rows)}/{len(records)} {sid}", flush=True)
    result = {"schema_version": 1, "status": "complete", "record_count": len(rows), "features": rows, "extractor": adapter_info, "interface": metadata, "waveform_policy": "complete_bound_natural_waveform_no_crop_pad_resample_or_amplitude_change", "sealed_splits_accessed": False}
    write_json(path, result)
    return result


def build_natural_slot(run_dir: Path, confirmation: Path, feature_run: Path, natural_manifest: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "03_natural_features/aligned_manifest.json"
    if path.exists() and resume:
        return read_json(path)
    common = _load_common(confirmation, feature_run)
    alignment = read_json(confirmation / "02_alignment/alignment.json")
    feature_by_id = {str(row["sample_id"]): row for row in natural_manifest["features"]}
    rows = []
    for target in common["masks"].values():
        sid = str(target["sample_id"])
        feature = np.asarray(np.load(feature_by_id[sid]["feature_path"], allow_pickle=False), dtype=np.float32)
        start, end, phone = _natural_span(target, feature.shape[0], alignment)
        base = build_example(target, common["natural"], common["new_tts"], common["stats"])
        target_core_len = int(target["core_end"]) - int(target["core_start"])
        aligned_core, interpolation = phone_phase_linear(standardize_tts(feature, common["stats"]), start, end, target_core_len)
        aligned = np.zeros_like(np.asarray(base["tts_features"], dtype=np.float32))
        aligned[int(target["core_start"]):int(target["core_end"])] = aligned_core
        paired_hash = _array_hash(np.asarray(base["tts_features"], dtype=np.float32))
        aligned_hash = _array_hash(aligned)
        if aligned_hash == paired_hash or not np.isfinite(aligned).all() or np.any(aligned[:int(target["core_start"])] != 0) or np.any(aligned[int(target["core_end"]):] != 0):
            raise ValueError(f"invalid natural-slot aligned feature: {target['mask_sha256']}")
        output = run_dir / "03_natural_features/aligned" / f"{target['mask_sha256']}.npy"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, aligned, allow_pickle=False)
        rows.append({"mask_sha256": str(target["mask_sha256"]), "sample_id": sid, "source_group": str(target["source_group"]), "natural_phone_index": int(target["natural_phone_index"]), "natural_phone": {"phone": str(phone["phone"]), "start": float(phone["start"]), "end": float(phone["end"])}, "source_feature_sha256": str(feature_by_id[sid]["feature_array_sha256"]), "source_feature_path": str(feature_by_id[sid]["feature_path"]), "source_frame_start": start, "source_frame_end": end, "aligned_path": str(output.resolve()), "aligned_feature_sha256": sha256_file(output), "aligned_array_sha256": aligned_hash, "paired_input_array_sha256": paired_hash, "shape": list(aligned.shape), "core_start": int(target["core_start"]), "core_end": int(target["core_end"]), "interpolation": interpolation})
    result = {"schema_version": 1, "status": "complete", "condition": NATURAL_SLOT, "mask_count": len(rows), "records": rows, "natural_feature_manifest_sha256": sha256_file(run_dir / "03_natural_features/manifest.json"), "alignment_sha256": sha256_file(confirmation / "02_alignment/alignment.json"), "normalization_sha256": sha256_file(feature_run / "02_features/normalization.json"), "sealed_splits_accessed": False}
    if len(rows) != EXPECTED_MASKS:
        raise ValueError(f"natural-slot mask matrix incomplete: {len(rows)}")
    write_json(path, result)
    return result


def run_natural_reconstruction(run_dir: Path, confirmation: Path, model_run: Path, feature_run: Path, aligned: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "04_natural_slot/reconstruction.json"
    if path.exists() and resume:
        return read_json(path)
    common = _load_common(confirmation, feature_run)
    old_rows = _load_reconstruction_rows(confirmation)
    old_by_key = _row_map(old_rows, ("sample_id", "seed", "mask_sha256", "condition"))
    masks = common["masks"]
    aligned_by_hash = {str(row["mask_sha256"]): row for row in aligned["records"]}
    rows = []
    set_deterministic_cpu()
    for seed in SEEDS:
        checkpoint = model_run / "04_training" / str(seed) / "hard_negative" / "checkpoint.pt"
        model, payload = load_checkpoint(checkpoint)
        for target in masks.values():
            mask_hash = str(target["mask_sha256"])
            aligned_row = aligned_by_hash[mask_hash]
            aligned_features = np.asarray(np.load(aligned_row["aligned_path"], allow_pickle=False), dtype=np.float32)
            base = build_example(target, common["natural"], common["new_tts"], common["stats"])
            example = {key: np.asarray(base[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "target")}
            example["tts_features"] = aligned_features
            prediction, loss = score_condition(model, example, zero_tts=False)
            paired = old_by_key[(str(target["sample_id"]), int(seed), mask_hash, "PAIRED_TTS")]
            nat_only = old_by_key[(str(target["sample_id"]), int(seed), mask_hash, "NAT_ONLY")]
            cell = run_dir / "04_natural_slot/reconstruction" / str(seed) / mask_hash / NATURAL_SLOT
            cell.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cell / "prediction.npz", prediction=prediction)
            rows.append({"schema_version": 1, "status": "complete", "seed": int(seed), "sample_id": str(target["sample_id"]), "source_group": str(target["source_group"]), "mask_sha256": mask_hash, "condition": NATURAL_SLOT, "loss": loss, "paired_loss": paired["loss"], "nat_only_loss": nat_only["loss"], "over_tts_reconstruction_gain": float(paired["loss"]["total"]) - float(loss["total"]), "over_zero_reconstruction_gain": float(nat_only["loss"]["total"]) - float(loss["total"]), "prediction_shape": list(prediction.shape), "prediction_sha256": _array_hash(prediction), "aligned_feature_sha256": str(aligned_row["aligned_array_sha256"]), "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_initial_state_sha256": payload["initial_state_sha256"]})
        print(f"natural-slot reconstruction seed {seed} complete", flush=True)
    expected = EXPECTED_MASKS * len(SEEDS)
    if len(rows) != expected:
        raise ValueError(f"natural-slot reconstruction matrix incomplete: {len(rows)}/{expected}")
    result = {"schema_version": 1, "status": "complete", "condition": NATURAL_SLOT, "record_count": EXPECTED_RECORDS, "group_count": EXPECTED_GROUPS, "mask_count": EXPECTED_MASKS, "seed_count": len(SEEDS), "required_cells": expected, "records": rows, "model_run": str(model_run.resolve()), "aligned_manifest_sha256": sha256_file(run_dir / "03_natural_features/aligned_manifest.json"), "sealed_splits_accessed": False}
    write_json(path, result)
    return result


def build_natural_drivers(run_dir: Path, confirmation: Path, feature_run: Path, reconstruction: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "04_natural_slot/drivers.json"
    if path.exists() and resume:
        return read_json(path)
    common = _load_common(confirmation, feature_run)
    recon_by_key = _row_map(reconstruction["records"], ("sample_id", "seed", "mask_sha256", "condition"))
    rows = []
    for record in common["records"]:
        sid = str(record["sample_id"])
        natural = np.asarray(common["natural"][sid], dtype=np.float32)
        mean = np.asarray(common["stats"]["mel_mean"], dtype=np.float32)[:, None]
        std = np.asarray(common["stats"]["mel_std"], dtype=np.float32)[:, None]
        standardized = (natural - mean) / std
        target_masks = [row for row in common["masks"].values() if str(row["sample_id"]) == sid]
        for seed in SEEDS:
            contributions = []
            used = []
            for target in target_masks:
                mask_hash = str(target["mask_sha256"])
                row = recon_by_key[(sid, int(seed), mask_hash, NATURAL_SLOT)]
                prediction = _load_prediction(run_dir / "04_natural_slot/reconstruction" / str(seed) / mask_hash / NATURAL_SLOT / "prediction.npz")
                if _array_hash(prediction) != str(row["prediction_sha256"]):
                    raise ValueError("natural-slot prediction hash changed")
                start = int(target["window_start_frame"]) + int(target["core_start"])
                end = int(target["window_start_frame"]) + int(target["core_end"])
                contributions.append((start, end, prediction[int(target["core_start"]):int(target["core_end"])]))
                used.append({"mask_sha256": mask_hash, "global_start_frame": start, "global_end_frame": end, "prediction_sha256": str(row["prediction_sha256"])})
            candidate, counts = patch_normalized_mel(standardized, contributions)
            output = natural.copy()
            covered = counts > 0
            output[:, covered] = candidate[:, covered] * std + mean
            unclamped = output.copy()
            clamp = (output < MEL_MIN) | (output > MEL_MAX)
            output = np.clip(output, MEL_MIN, MEL_MAX).astype(np.float32, copy=False)
            if np.any(output[:, ~covered] != natural[:, ~covered]):
                raise ValueError(f"natural-slot driver changed outside core: {sid}")
            driver_id = f"{sid}__{NATURAL_SLOT}__{seed}"
            driver_path = run_dir / "04_natural_slot/drivers" / f"{driver_id}.npy"
            driver_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(driver_path, output, allow_pickle=False)
            rows.append({"driver_id": driver_id, "sample_id": sid, "source_group": str(record["source_group"]), "condition": NATURAL_SLOT, "seed": int(seed), "path": str(driver_path.resolve()), "sha256": sha256_file(driver_path), "shape": list(output.shape), "natural_mel_sha256": _array_hash(natural), "covered_frame_count": int(np.count_nonzero(covered)), "overlap_frame_count": int(np.count_nonzero(counts > 1)), "clamp_fraction": float(np.mean(clamp)), "clamped_values": int(np.count_nonzero(clamp)), "total_values": int(output.size), "mel_range_before_clamp": [float(unclamped.min()), float(unclamped.max())], "mel_range_after_clamp": [float(output.min()), float(output.max())], "used_masks": used, "outside_core_policy": "exact_natural_mel_source"})
    if len(rows) != EXPECTED_RECORDS * len(SEEDS):
        raise ValueError("natural-slot driver matrix incomplete")
    result = {"schema_version": 1, "manifest_type": "lrs3_masked_tts_natural_slot_direct_mel_drivers", "status": "complete", "condition": NATURAL_SLOT, "driver_count": len(rows), "record_count": EXPECTED_RECORDS, "seeds": list(SEEDS), "drivers": rows, "reconstruction_sha256": sha256_file(run_dir / "04_natural_slot/reconstruction.json"), "sealed_splits_accessed": False}
    write_json(path, result)
    return result


def render_and_score_natural(run_dir: Path, confirmation: Path, drivers: Mapping[str, Any], *, render_workers: int, score_workers: int, resume: bool) -> dict[str, Any]:
    stage = run_dir / "04_natural_slot"
    render_path = stage / "renders/render_manifest.json"
    if not render_path.exists() or not resume:
        render_controls_parallel(stage, confirmation / "00_cohort/manifest.json", stage / "drivers.json", confirmation / "06_renders/parity.json", confirmation / "06_renders/box_manifest.json", stage / "renders", workers=render_workers)
    if not (stage / "render_manifest.json").is_file():
        write_json(stage / "render_manifest.json", read_json(render_path))
    if not (stage / "05_syncnet/summary.json").is_file() or not resume:
        score_controls_parallel(stage, confirmation / "00_cohort/manifest.json", render_path, workers=score_workers)
    score = read_json(stage / "05_syncnet/summary.json")
    if score.get("status") != "complete" or int(score.get("score_count", -1)) != EXPECTED_RECORDS * len(SEEDS):
        raise ValueError("natural-slot downstream matrix incomplete")
    write_json(stage / "summary.json", score)
    return score


def _hierarchical_contrast(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], metric: str, *, mask_level: bool) -> dict[str, Any]:
    return _aggregate_named(records, rows, metric, mask_level=mask_level)


def analyze_natural(run_dir: Path, confirmation: Path, reconstruction: Mapping[str, Any], syncnet: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "04_natural_slot/analysis.json"
    if path.exists() and resume:
        return read_json(path)
    records, _ = _records_and_masks(confirmation)
    old_recon = _load_reconstruction_rows(confirmation)
    old_recon_by_key = _row_map(old_recon, ("sample_id", "seed", "mask_sha256", "condition"))
    recon_rows = []
    for row in reconstruction["records"]:
        paired = old_recon_by_key[(str(row["sample_id"]), int(row["seed"]), str(row["mask_sha256"]), "PAIRED_TTS")]
        nat_only = old_recon_by_key[(str(row["sample_id"]), int(row["seed"]), str(row["mask_sha256"]), "NAT_ONLY")]
        recon_rows.append({**row, "over_zero": float(nat_only["loss"]["total"]) - float(row["loss"]["total"]), "over_tts": float(paired["loss"]["total"]) - float(row["loss"]["total"])})
    reconstruction_contrasts = {"over_zero": _hierarchical_contrast(records, recon_rows, "over_zero", mask_level=True), "over_tts": _hierarchical_contrast(records, recon_rows, "over_tts", mask_level=True)}
    old_scores = _load_sync_rows(confirmation)
    old_score_by_key = _row_map(old_scores, ("sample_id", "seed", "condition"))
    new_scores = list(syncnet["scores"])
    if len(new_scores) != EXPECTED_RECORDS * len(SEEDS):
        raise ValueError("natural-slot score rows incomplete")
    downstream_rows = []
    for row in new_scores:
        paired = old_score_by_key[(str(row["sample_id"]), int(row["seed"]), "PAIRED_TTS")]
        zero = old_score_by_key[(str(row["sample_id"]), int(row["seed"]), "NAT_ONLY")]
        downstream_rows.append({**row, "over_zero_C": float(row["sync_c"]) - float(zero["sync_c"]), "over_zero_D": float(zero["sync_d"]) - float(row["sync_d"]), "over_tts_C": float(row["sync_c"]) - float(paired["sync_c"]), "over_tts_D": float(paired["sync_d"]) - float(row["sync_d"])})
    downstream = {contrast: {metric: _hierarchical_contrast(records, downstream_rows, f"{contrast}_{metric}", mask_level=False) for metric in ("C", "D")} for contrast in ("over_zero", "over_tts")}
    for contrast in downstream:
        for metric in downstream[contrast]:
            summary = downstream[contrast][metric]["summary"]
            summary["pass"] = bool(summary["ci95"][0] > 0 and summary["positive_groups"] >= 7)
    tfg = {contrast: bool(all(downstream[contrast][metric]["summary"]["pass"] for metric in ("C", "D"))) for contrast in downstream}
    recon_pass = {contrast: bool(reconstruction_contrasts[contrast]["summary"]["ci95"][0] > 0 and reconstruction_contrasts[contrast]["summary"].get("positive_groups", 0) >= 7) for contrast in reconstruction_contrasts}
    contrast_status = {}
    for contrast in ("over_zero", "over_tts"):
        contrast_status[contrast] = {"tfg": "NATURAL_SLOT_OVER_ZERO_TFG_PASS" if contrast == "over_zero" and tfg[contrast] else "NATURAL_SLOT_OVER_TTS_TFG_PASS" if contrast == "over_tts" and tfg[contrast] else "TFG_NOT_SHOWN", "reconstruction": "RECONSTRUCTION_PASS" if recon_pass[contrast] else "RECONSTRUCTION_NOT_SHOWN", "qualification": "RECONSTRUCTION_SUPPORTED" if tfg[contrast] and recon_pass[contrast] else "DOWNSTREAM_ONLY" if tfg[contrast] else "TFG_NOT_SHOWN"}
    result = {"schema_version": 1, "status": "complete", "part": "B", "condition": NATURAL_SLOT, "oracle_reference_condition": True, "previously_inspected_records": True, "reconstruction": reconstruction_contrasts, "downstream": downstream, "contrasts": contrast_status, "rules": {"positive_directions": {"over_zero": "NAT_ONLY - natural-slot loss; natural-slot Sync-C - NAT_ONLY Sync-C; NAT_ONLY Sync-D - natural-slot Sync-D", "over_tts": "PAIRED_TTS - natural-slot loss; natural-slot Sync-C - PAIRED_TTS Sync-C; PAIRED_TTS Sync-D - natural-slot Sync-D"}, "pass": "95% whole-group bootstrap lower bound > 0 and at least 7/8 source groups positive"}, "sealed_splits_accessed": False}
    write_json(path, result)
    lines = ["# Part B: natural WavLM in the TTS slot", "", "Status: **complete**.", "", "This is an oracle/reference condition: untouched natural audio is encoded into the existing TTS feature slot. It is not a text-only TTS result.", "", "| Contrast | Axis | Metric | Median | 95% CI | Positive groups | Pass |", "|---|---|---|---:|---|---:|---|"]
    labels = {"over_zero": "natural slot over zero input", "over_tts": "natural slot over paired TTS"}
    for contrast in ("over_zero", "over_tts"):
        item = reconstruction_contrasts[contrast]["summary"]
        lines.append(f"| {labels[contrast]} | reconstruction | total | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if recon_pass[contrast] else 'no'} |")
        for metric in ("C", "D"):
            item = downstream[contrast][metric]["summary"]
            lines.append(f"| {labels[contrast]} | frozen TFG | Sync-{metric} | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if item['pass'] else 'no'} |")
    (run_dir / "04_natural_slot/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _neighbor_mismatch(target: Mapping[str, Any], candidate: Mapping[str, Any], alignment_by_id: Mapping[str, Mapping[str, Any]]) -> int:
    def neighbors(row: Mapping[str, Any]) -> tuple[str, str]:
        phones = alignment_by_id[str(row["sample_id"])]["tts_phones"]
        index = int(row["tts_phone_index"])
        values = []
        for offset in (-1, 1):
            position = index + offset
            values.append(normalize_phone(phones[position]["phone"]) if 0 <= position < len(phones) else "__BOUNDARY__")
        return values[0], values[1]
    return int(sum(a != b for a, b in zip(neighbors(target), neighbors(candidate))))


def matched_preflight(run_dir: Path, confirmation: Path, feature_run: Path, *, resume: bool) -> dict[str, Any]:
    path = run_dir / "05_matched_control/preflight.json"
    if path.exists() and resume:
        return read_json(path)
    common = _load_common(confirmation, feature_run)
    alignment = read_json(confirmation / "02_alignment/alignment.json")
    alignment_by_id = {str(row["sample_id"]): row for row in alignment["records"]}
    occurrences: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in common["masks"].values():
        key = (str(row["sample_id"]), int(row["tts_phone_index"]))
        current = occurrences.get(key)
        if current is None or int(row["canonical_index"]) < int(current["canonical_index"]):
            occurrences[key] = row
    selected_rows = []
    invalid = []
    for target in common["masks"].values():
        target_occurrence = (str(target["sample_id"]), int(target["tts_phone_index"]))
        candidates = []
        for key, candidate in occurrences.items():
            if normalize_phone(candidate["label"]) != normalize_phone(target["label"]) or key == target_occurrence:
                continue
            candidate_sid = str(candidate["sample_id"])
            if candidate_sid == str(target["sample_id"]):
                tier = 0
            elif str(candidate["source_group"]) == str(target["source_group"]):
                tier = 1
            else:
                continue
            values = common["new_tts"].get(candidate_sid)
            start, end = int(candidate["tts_frame_start"]), int(candidate["tts_frame_end"])
            if values is None or values.ndim != 2 or values.shape[1] != WAVLM_DIM or start < 0 or end > values.shape[0] or end - start < 2:
                continue
            duration_diff = abs(float(candidate["tts_end_s"]) - float(candidate["tts_start_s"]) - (float(target["tts_end_s"]) - float(target["tts_start_s"])))
            candidates.append((tier, _neighbor_mismatch(target, candidate, alignment_by_id), duration_diff, candidate_sid, int(candidate["canonical_index"]), str(candidate["mask_sha256"]), candidate))
        candidates.sort(key=lambda item: item[:-1])
        distinct = []
        seen_occurrences: set[tuple[str, int]] = set()
        for item in candidates:
            occurrence = (str(item[-1]["sample_id"]), int(item[-1]["tts_phone_index"]))
            if occurrence in seen_occurrences:
                continue
            seen_occurrences.add(occurrence)
            distinct.append(item)
            if len(distinct) == 3:
                break
        if len(distinct) < 3:
            invalid.append({"mask_sha256": str(target["mask_sha256"]), "sample_id": str(target["sample_id"]), "reason": "fewer than three distinct same-phone donor occurrences"})
            continue
        selected_rows.append({"mask_sha256": str(target["mask_sha256"]), "sample_id": str(target["sample_id"]), "source_group": str(target["source_group"]), "donors": [{"rank": index + 1, "sample_id": str(item[-1]["sample_id"]), "source_group": str(item[-1]["source_group"]), "tts_phone_index": int(item[-1]["tts_phone_index"]), "mask_sha256": str(item[-1]["mask_sha256"]), "source_tier": int(item[0]), "neighbor_mismatch_count": int(item[1]), "duration_difference": float(item[2]), "canonical_index": int(item[4])} for index, item in enumerate(distinct)]})
    per_record = {str(record["sample_id"]): sum(row["sample_id"] == str(record["sample_id"]) for row in selected_rows) for record in common["records"]}
    represented_groups = sorted({str(row["source_group"]) for row in selected_rows})
    sufficient = not invalid and all(value >= 2 for value in per_record.values()) and len(represented_groups) == EXPECTED_GROUPS
    status = "complete" if sufficient else "INSUFFICIENT_MATCHED_DONOR_COVERAGE"
    result = {"schema_version": 1, "status": status, "part": "C", "record_count": EXPECTED_RECORDS, "group_count": EXPECTED_GROUPS, "eligible_mask_count": len(selected_rows), "invalid_masks": invalid, "per_record_eligible_masks": per_record, "represented_source_groups": represented_groups, "records": selected_rows, "selection_rule": ["same normalized phone", "distinct occurrence identity=(sample_id,tts_phone_index)", "same record before same source group", "neighbor mismatch", "duration difference", "sample ID", "canonical index", "mask hash"], "selection_does_not_use": ["reconstruction", "Wav2Lip", "SyncNet"], "downstream_allowed": sufficient, "sealed_splits_accessed": False}
    write_json(path, result)
    (run_dir / "05_matched_control").mkdir(parents=True, exist_ok=True)
    (run_dir / "05_matched_control/preflight.md").write_text(f"# Part C: matched donor preflight\n\nStatus: **{status}**.\n\nEligible masks: {len(selected_rows)}.\n", encoding="utf-8")
    return result


def _matched_reconstruction(run_dir: Path, confirmation: Path, model_run: Path, feature_run: Path, preflight: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "05_matched_control/reconstruction.json"
    if path.exists() and resume:
        return read_json(path)
    common = _load_common(confirmation, feature_run)
    old_rows = _load_reconstruction_rows(confirmation)
    old_by_key = _row_map(old_rows, ("sample_id", "seed", "mask_sha256", "condition"))
    target_by_hash = common["masks"]
    donors_by_hash = {str(row["mask_sha256"]): row for row in preflight["records"]}
    occurrence_mask = {(str(row["sample_id"]), int(row["tts_phone_index"])): row for row in common["masks"].values()}
    rows = []
    set_deterministic_cpu()
    for seed in SEEDS:
        checkpoint = model_run / "04_training" / str(seed) / "hard_negative" / "checkpoint.pt"
        model, payload = load_checkpoint(checkpoint)
        for mask_hash, donor_row in donors_by_hash.items():
            target = target_by_hash[mask_hash]
            paired = old_by_key[(str(target["sample_id"]), int(seed), mask_hash, "PAIRED_TTS")]
            for donor in donor_row["donors"]:
                donor_mask = occurrence_mask[(str(donor["sample_id"]), int(donor["tts_phone_index"]))]
                example_raw = build_example(target, common["natural"], common["new_tts"], common["stats"], tts_mask=donor_mask)
                example = {key: np.asarray(example_raw[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "tts_features", "target")}
                prediction, loss = score_condition(model, example, zero_tts=False)
                rank = int(donor["rank"])
                cell = run_dir / "05_matched_control/reconstruction" / str(seed) / mask_hash / f"MATCHED_WRONG_RANK_{rank}"
                cell.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cell / "prediction.npz", prediction=prediction)
                rows.append({"schema_version": 1, "status": "complete", "seed": int(seed), "sample_id": str(target["sample_id"]), "source_group": str(target["source_group"]), "mask_sha256": mask_hash, "condition": f"MATCHED_WRONG_RANK_{rank}", "donor_rank": rank, "donor_source_sample_id": str(donor["sample_id"]), "donor_source_mask_sha256": str(donor["mask_sha256"]), "loss": loss, "paired_loss": paired["loss"], "reconstruction_gain": float(loss["total"]) - float(paired["loss"]["total"]), "prediction_shape": list(prediction.shape), "prediction_sha256": _array_hash(prediction), "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_initial_state_sha256": payload["initial_state_sha256"]})
        print(f"matched reconstruction seed {seed} complete", flush=True)
    expected = len(donors_by_hash) * len(SEEDS) * 3
    if len(rows) != expected:
        raise ValueError(f"matched reconstruction matrix incomplete: {len(rows)}/{expected}")
    result = {"schema_version": 1, "status": "complete", "part": "C", "record_count": EXPECTED_RECORDS, "group_count": EXPECTED_GROUPS, "eligible_mask_count": len(donors_by_hash), "seed_count": len(SEEDS), "donor_rank_count": 3, "required_cells": expected, "records": rows, "sealed_splits_accessed": False}
    write_json(path, result)
    return result


def _build_matched_drivers(run_dir: Path, confirmation: Path, feature_run: Path, preflight: Mapping[str, Any], matched_recon: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "05_matched_control/drivers.json"
    if path.exists() and resume:
        return read_json(path)
    common = _load_common(confirmation, feature_run)
    paired_old = _load_reconstruction_rows(confirmation)
    paired_by_key = _row_map(paired_old, ("sample_id", "seed", "mask_sha256", "condition"))
    matched_by_key = _row_map([row for row in matched_recon["records"] if int(row["donor_rank"]) == 1], ("sample_id", "seed", "mask_sha256", "condition"))
    eligible = {str(row["mask_sha256"]): row for row in preflight["records"]}
    rows = []
    conditions = (COMMON_PAIRED, MATCHED_WRONG)
    for record in common["records"]:
        sid = str(record["sample_id"])
        natural = np.asarray(common["natural"][sid], dtype=np.float32)
        mean = np.asarray(common["stats"]["mel_mean"], dtype=np.float32)[:, None]
        std = np.asarray(common["stats"]["mel_std"], dtype=np.float32)[:, None]
        standardized = (natural - mean) / std
        mask_rows = [common["masks"][h] for h in eligible if str(common["masks"][h]["sample_id"]) == sid]
        for seed in SEEDS:
            contributions_by_condition = {condition: [] for condition in conditions}
            used_by_condition = {condition: [] for condition in conditions}
            for target in mask_rows:
                mask_hash = str(target["mask_sha256"])
                paired_prediction = _load_prediction(confirmation / "04_reconstruction" / str(seed) / mask_hash / "PAIRED_TTS/prediction.npz")
                matched_prediction = _load_prediction(run_dir / "05_matched_control/reconstruction" / str(seed) / mask_hash / "MATCHED_WRONG_RANK_1/prediction.npz")
                start = int(target["window_start_frame"]) + int(target["core_start"])
                end = int(target["window_start_frame"]) + int(target["core_end"])
                contributions_by_condition[COMMON_PAIRED].append((start, end, paired_prediction[int(target["core_start"]):int(target["core_end"])]))
                contributions_by_condition[MATCHED_WRONG].append((start, end, matched_prediction[int(target["core_start"]):int(target["core_end"])]))
                used_by_condition[COMMON_PAIRED].append({"mask_sha256": mask_hash, "prediction_sha256": _array_hash(paired_prediction)})
                used_by_condition[MATCHED_WRONG].append({"mask_sha256": mask_hash, "prediction_sha256": _array_hash(matched_prediction)})
            for condition in conditions:
                candidate, counts = patch_normalized_mel(standardized, contributions_by_condition[condition])
                output = natural.copy()
                covered = counts > 0
                output[:, covered] = candidate[:, covered] * std + mean
                unclamped = output.copy()
                clamp = (output < MEL_MIN) | (output > MEL_MAX)
                output = np.clip(output, MEL_MIN, MEL_MAX).astype(np.float32, copy=False)
                if np.any(output[:, ~covered] != natural[:, ~covered]):
                    raise ValueError(f"matched driver changed outside core: {sid} {condition}")
                driver_id = f"{sid}__{condition}__{seed}"
                driver_path = run_dir / "05_matched_control/drivers" / f"{driver_id}.npy"
                driver_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(driver_path, output, allow_pickle=False)
                rows.append({"driver_id": driver_id, "sample_id": sid, "source_group": str(record["source_group"]), "condition": condition, "seed": int(seed), "path": str(driver_path.resolve()), "sha256": sha256_file(driver_path), "shape": list(output.shape), "natural_mel_sha256": _array_hash(natural), "covered_frame_count": int(np.count_nonzero(covered)), "overlap_frame_count": int(np.count_nonzero(counts > 1)), "clamp_fraction": float(np.mean(clamp)), "clamped_values": int(np.count_nonzero(clamp)), "total_values": int(output.size), "mel_range_before_clamp": [float(unclamped.min()), float(unclamped.max())], "mel_range_after_clamp": [float(output.min()), float(output.max())], "used_masks": used_by_condition[condition], "outside_core_policy": "exact_natural_mel_source", "common_mask_subset": True})
    if len(rows) != EXPECTED_RECORDS * len(SEEDS) * 2:
        raise ValueError("matched driver matrix incomplete")
    result = {"schema_version": 1, "manifest_type": "lrs3_masked_tts_matched_common_mask_direct_mel_drivers", "status": "complete", "conditions": list(conditions), "driver_count": len(rows), "record_count": EXPECTED_RECORDS, "seeds": list(SEEDS), "drivers": rows, "eligible_mask_count": len(eligible), "reconstruction_sha256": sha256_file(run_dir / "05_matched_control/reconstruction.json"), "sealed_splits_accessed": False}
    write_json(path, result)
    return result


def _run_matched_downstream(run_dir: Path, confirmation: Path, *, render_workers: int, score_workers: int, resume: bool) -> dict[str, Any]:
    stage = run_dir / "05_matched_control"
    render_path = stage / "renders/render_manifest.json"
    if not render_path.exists() or not resume:
        render_controls_parallel(stage, confirmation / "00_cohort/manifest.json", stage / "drivers.json", confirmation / "06_renders/parity.json", confirmation / "06_renders/box_manifest.json", stage / "renders", workers=render_workers)
    if not (stage / "render_manifest.json").is_file():
        write_json(stage / "render_manifest.json", read_json(render_path))
    if not (stage / "05_syncnet/summary.json").is_file() or not resume:
        score_controls_parallel(stage, confirmation / "00_cohort/manifest.json", render_path, workers=score_workers)
    score = read_json(stage / "05_syncnet/summary.json")
    if score.get("status") != "complete" or int(score.get("score_count", -1)) != EXPECTED_RECORDS * len(SEEDS) * 2:
        raise ValueError("matched downstream matrix incomplete")
    write_json(stage / "summary.json", score)
    return score


def analyze_matched(run_dir: Path, confirmation: Path, preflight: Mapping[str, Any], matched_recon: Mapping[str, Any], syncnet: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "05_matched_control/analysis.json"
    if path.exists() and resume:
        return read_json(path)
    records, _ = _records_and_masks(confirmation)
    old = _load_reconstruction_rows(confirmation)
    old_by_key = _row_map(old, ("sample_id", "seed", "mask_sha256", "condition"))
    eligible = {str(row["mask_sha256"]) for row in preflight["records"]}
    recon_rows = list(matched_recon["records"])
    ranks = {}
    for rank in (1, 2, 3):
        local = []
        for row in recon_rows:
            if int(row["donor_rank"]) == rank:
                paired = old_by_key[(str(row["sample_id"]), int(row["seed"]), str(row["mask_sha256"]), "PAIRED_TTS")]
                local.append({"sample_id": row["sample_id"], "source_group": row["source_group"], "seed": row["seed"], "mask_sha256": row["mask_sha256"], "metric": float(row["loss"]["total"]) - float(paired["loss"]["total"])})
        ranks[str(rank)] = _aggregate_named(records, local, "metric", mask_level=True)
    all_rows = []
    by_target_seed: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in recon_rows:
        by_target_seed[(str(row["sample_id"]), int(row["seed"]), str(row["mask_sha256"]))].append(row)
    for key, values in by_target_seed.items():
        all_rows.append({
            "sample_id": key[0],
            "source_group": next(row["source_group"] for row in values),
            "seed": key[1],
            "mask_sha256": key[2],
            "metric": float(np.median([float(row["reconstruction_gain"]) for row in values])),
        })
    reconstruction_overall = _aggregate_named(records, all_rows, "metric", mask_level=True)
    score_rows = list(syncnet["scores"])
    if len(score_rows) != EXPECTED_RECORDS * len(SEEDS) * 2:
        raise ValueError("matched SyncNet rows incomplete")
    score_by_key = _row_map(score_rows, ("sample_id", "seed", "condition"))
    downstream_rows = []
    for record in records:
        sid = str(record["sample_id"])
        for seed in SEEDS:
            paired = score_by_key[(sid, int(seed), COMMON_PAIRED)]
            wrong = score_by_key[(sid, int(seed), MATCHED_WRONG)]
            downstream_rows.append({"sample_id": sid, "source_group": record["source_group"], "seed": int(seed), "sync_c_gain": float(paired["sync_c"]) - float(wrong["sync_c"]), "sync_d_gain": float(wrong["sync_d"]) - float(paired["sync_d"])})
    downstream = {metric: _aggregate_named(records, downstream_rows, metric, mask_level=False) for metric in ("sync_c_gain", "sync_d_gain")}
    for item in downstream.values():
        item["summary"]["pass"] = bool(item["summary"]["ci95"][0] > 0 and item["summary"]["positive_groups"] >= 7)
    reconstruction_pass = all(ranks[str(rank)]["summary"]["pass"] for rank in (1, 2, 3))
    downstream_pass = all(item["summary"]["pass"] for item in downstream.values())
    if downstream_pass:
        status = "MATCHED_INSTANCE_TFG_SIGNAL"
        qualification = "RECONSTRUCTION_SUPPORTED" if reconstruction_pass else "DOWNSTREAM_ONLY"
    elif reconstruction_pass:
        status = "MATCHED_INSTANCE_RECONSTRUCTION_ONLY"
        qualification = "RECONSTRUCTION_SUPPORTED"
    else:
        status = "NO_MATCHED_INSTANCE_SIGNAL"
        qualification = "TFG_NOT_SHOWN"
    result = {"schema_version": 1, "status": "complete", "part": "C", "matched_status": status, "qualification": qualification, "previously_inspected_records": True, "eligible_mask_count": len(eligible), "reconstruction": {"per_rank": ranks, "overall_three_rank_median": reconstruction_overall, "pass": reconstruction_pass}, "downstream": downstream, "downstream_pass": downstream_pass, "rules": {"reconstruction": "all three donor ranks pass the whole-group rule", "downstream": "rank-one common-mask paired baseline passes Sync-C and Sync-D under the whole-group rule"}, "sealed_splits_accessed": False}
    write_json(path, result)
    return result


def final_decision(run_dir: Path, correction: Mapping[str, Any], trace_result: Mapping[str, Any], natural: Mapping[str, Any], matched: Mapping[str, Any]) -> dict[str, Any]:
    if correction.get("status") != "complete" or trace_result.get("status") != "complete" or natural.get("status") != "complete":
        recommendation = "STOP_INVALID_EXPERIMENT"
    elif matched.get("matched_status") == "MATCHED_INSTANCE_TFG_SIGNAL":
        recommendation = "CONFIRM_MATCHED_INSTANCE_CONTROL_ON_FRESH_RECORDS"
    elif natural.get("contrasts", {}).get("over_zero", {}).get("tfg") == "NATURAL_SLOT_OVER_ZERO_TFG_PASS" and natural.get("contrasts", {}).get("over_tts", {}).get("tfg") == "NATURAL_SLOT_OVER_TTS_TFG_PASS":
        recommendation = "PIVOT_TO_NATURAL_REFERENCE_CONDITIONING"
    elif natural.get("contrasts", {}).get("over_zero", {}).get("tfg") == "NATURAL_SLOT_OVER_ZERO_TFG_PASS":
        recommendation = "RETAIN_MODALITY_ONLY_CLAIM"
    else:
        recommendation = "REDESIGN_CONDITIONING_OBJECTIVE_BEFORE_MORE_DATA"
    return {"schema_version": 1, "status": "complete", "recommendation": recommendation, "part_0_status": correction.get("status"), "part_a_status": trace_result.get("status"), "part_b_status": natural.get("status"), "part_c_status": matched.get("matched_status", matched.get("status")), "previously_inspected_records": True, "oracle_reference_condition": True, "bounded_claims": ["no audible retention", "no waveform recovery", "no deployability", "no population generalization", "matched control is exploratory on inspected records"], "waveform_decoder_gate": "CLOSED", "sealed_splits_accessed": False}


def execute(args: argparse.Namespace) -> int:
    run_dir = args.run.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise ValueError(f"refusing non-empty output directory without --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    confirmation = args.confirmation.resolve()
    model_run = args.model_run.resolve()
    feature_run = args.feature_run.resolve()
    binding = bind(run_dir, confirmation, model_run, feature_run, resume=args.resume)
    if args.stage == "bind":
        return 0
    correction = correct_analysis(run_dir, confirmation, resume=args.resume)
    if args.stage == "correct-analysis":
        return 0
    if correction.get("status") != "complete":
        raise RuntimeError("Part 0 is NOT_EVALUATED")
    trace_result = trace(run_dir, confirmation, feature_run, resume=args.resume)
    if args.stage == "trace":
        return 0
    natural_manifest = extract_natural_features(run_dir, confirmation, binding, device=args.device, resume=args.resume)
    aligned = build_natural_slot(run_dir, confirmation, feature_run, natural_manifest, resume=args.resume)
    natural_reconstruction = run_natural_reconstruction(run_dir, confirmation, model_run, feature_run, aligned, resume=args.resume)
    natural_drivers = build_natural_drivers(run_dir, confirmation, feature_run, natural_reconstruction, resume=args.resume)
    natural_scores = render_and_score_natural(run_dir, confirmation, natural_drivers, render_workers=args.render_workers, score_workers=args.score_workers, resume=args.resume)
    natural_analysis = analyze_natural(run_dir, confirmation, natural_reconstruction, natural_scores, resume=args.resume)
    if args.stage == "natural-slot":
        return 0
    preflight = matched_preflight(run_dir, confirmation, feature_run, resume=args.resume)
    if args.stage == "matched-preflight":
        return 0
    if preflight.get("status") == "complete":
        matched_reconstruction = _matched_reconstruction(run_dir, confirmation, model_run, feature_run, preflight, resume=args.resume)
        matched_drivers = _build_matched_drivers(run_dir, confirmation, feature_run, preflight, matched_reconstruction, resume=args.resume)
        matched_scores = _run_matched_downstream(run_dir, confirmation, render_workers=args.render_workers, score_workers=args.score_workers, resume=args.resume)
        matched_analysis = analyze_matched(run_dir, confirmation, preflight, matched_reconstruction, matched_scores, resume=args.resume)
    else:
        matched_analysis = {"schema_version": 1, "status": preflight["status"], "matched_status": preflight["status"], "reason": "coverage gate stopped before inference or rendering"}
        write_json(run_dir / "05_matched_control/analysis.json", matched_analysis)
    if args.stage == "matched-run":
        return 0
    decision = final_decision(run_dir, correction, trace_result, natural_analysis, matched_analysis)
    write_json(run_dir / "decision.json", decision)
    summary = {"schema_version": 1, "status": "complete", "binding_sha256": sha256_file(run_dir / "00_binding/manifest.json"), "part_0": {"status": correction["status"], "analysis_sha256": sha256_file(run_dir / "01_correction/analysis.json")}, "part_a": {"status": trace_result["status"], "analysis_sha256": sha256_file(run_dir / "02_trace/analysis.json")}, "part_b": {"status": natural_analysis["status"], "natural_slot_status": natural_analysis["contrasts"], "analysis_sha256": sha256_file(run_dir / "04_natural_slot/analysis.json"), "reconstruction_cells": natural_reconstruction["required_cells"], "downstream_cells": natural_scores["score_count"]}, "part_c": {"status": matched_analysis.get("matched_status", matched_analysis.get("status")), "preflight": preflight.get("status"), "eligible_mask_count": preflight.get("eligible_mask_count")}, "decision": decision, "sealed_splits_accessed": False}
    write_json(run_dir / "summary.json", summary)
    print(json.dumps({"part_0": correction["status"], "part_a": trace_result["status"], "part_b": natural_analysis["status"], "part_c": matched_analysis.get("matched_status", matched_analysis.get("status")), "recommendation": decision["recommendation"]}, ensure_ascii=False))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--model-run", type=Path, default=DEFAULT_MODEL_RUN)
    parser.add_argument("--feature-run", type=Path, default=DEFAULT_FEATURE_RUN)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("bind", "correct-analysis", "trace", "natural-slot", "matched-preflight", "matched-run", "analyze", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
