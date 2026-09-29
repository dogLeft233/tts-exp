"""Runner for the independent fresh-source visual (B) branch."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments.fresh_source_inputs.protocol import (
    ProtocolError,
    file_sha256,
    load_self_hashed,
    read_json,
    write_json,
)
from scripts.experiments.fresh_source_visual.common import (
    BLIND_SEED,
    BOOTSTRAP_COUNT,
    BOOTSTRAP_SEED,
    DEFAULT_LANDMARKER_ASSET,
    FPS,
    FRAME_COUNT,
    J0,
    MIN_OBSERVED_FRAMES,
    MODELS,
    FeatureArray,
    VisualProtocolError,
    _encode_silent_panels,
    _read_video_frames,
    _transform_frames,
    anonymise_name,
    bootstrap_metrics,
    crop_feature,
    fixed_clock_support,
    full_cohort_bounds,
    group_decomposition,
    import_video_features,
    load_feature,
    load_video_manifest,
    make_contact_sheet,
    media_metadata,
    normalise_video_rows,
    run_calibration,
    save_feature,
    verify_video_item,
)

_RATING_ALIASES = {
    "left": "left",
    "l": "left",
    "左": "left",
    "right": "right",
    "r": "right",
    "右": "right",
    "same": "same",
    "tie": "same",
    "equal": "same",
    "相同": "same",
    "unable": "unable",
    "unjudgeable": "unable",
    "无法判断": "unable",
    "missing": "missing",
    "": "missing",
}
_RATING_CHOICES = frozenset({"left", "right", "same", "unable", "missing"})


def _find_manifest(run_root: Path, name: str) -> Path | None:
    candidates = [run_root / "run/shared" / name, run_root / "shared" / name, run_root / name]
    if run_root.name == "shared":
        candidates.insert(0, run_root / name)
    return next((path for path in candidates if path.is_file()), None)


def _cohort_root(run_root: Path) -> Path:
    return run_root.parent.parent if run_root.name == "shared" else run_root


def _read_manifest(run_root: Path, name: str, *, required: bool = True) -> tuple[dict[str, Any], Path | None]:
    path = _find_manifest(run_root, name)
    if path is None:
        if required:
            raise VisualProtocolError(f"missing frozen manifest: {name}")
        return {}, None
    try:
        return load_self_hashed(path), path
    except ProtocolError:
        value = read_json(path)
        if value.get("status") not in {
            "BLOCKED_NEW_SOURCE", "BLOCKED_HISTORY_COVERAGE", "BLOCKED_SOURCE_ACCESS", "BLOCKED_ACQUISITION_BUDGET",
        }:
            raise
        return value, path


def _formal_records(run_root: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    cohort, _ = _read_manifest(run_root, "cohort.json")
    inputs, _ = _read_manifest(run_root, "inputs.json")
    if str(cohort.get("status")) not in {"GO", "COHORT_READY"} and str(cohort.get("readiness")) != "COHORT_READY":
        raise VisualProtocolError(f"visual branch is upstream-blocked: {cohort.get('status')}")
    formal = cohort.get("formal", [])
    groups = [str(item.get("source_group")) for item in formal if isinstance(item, dict)]
    if len(groups) != 12 or len(set(groups)) != 12:
        raise VisualProtocolError("COHORT_READY must contain 12 unique formal groups")
    by_group = {str(item.get("source_group")): dict(item) for item in inputs.get("records", []) if isinstance(item, dict)}
    if set(groups) - set(by_group):
        raise VisualProtocolError("inputs.json is missing formal groups")
    return groups, {group: by_group[group] for group in groups}


def _root_and_branch(run_root: Path) -> tuple[Path, Path]:
    root = _cohort_root(run_root.resolve())
    branch = root / "run" / "B"
    branch.mkdir(parents=True, exist_ok=True)
    return root, branch


def _array_sha(value: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _feature_path(branch: Path, sample_id: str, arm: str) -> Path:
    return branch / "features" / anonymise_name(sample_id) / f"{anonymise_name(arm)}.npz"


def _feature_entry(path: Path, *, sample_id: str, group: str, arm: str, source_path: Path | None = None) -> dict[str, Any]:
    feature = load_feature(path)
    return {
        "sample_id": sample_id,
        "source_group": group,
        "arm": arm,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "frame_count": feature.frame_count,
        "valid_fraction": float(np.mean(feature.valid)) if feature.valid.size else 0.0,
        "timestamps_s_sha256": _array_sha(feature.timestamps),
        "source_video": str(source_path.resolve()) if source_path else feature.metadata.get("video_path"),
        "extractor": feature.metadata.get("extractor"),
        "landmarker_asset_sha256": feature.metadata.get("landmarker_asset_sha256"),
    }


def _extract_feature(video: Path, asset: Path, output: Path, *, sample_id: str, group: str, arm: str, source_kind: str) -> dict[str, Any]:
    create_landmarker, extract_video_features = import_video_features()
    if not video.is_file():
        raise VisualProtocolError(f"video missing: {video}")
    handle = create_landmarker(asset)
    try:
        sequence = extract_video_features(video, asset, landmarker_handle=handle)
    finally:
        try:
            handle[1].close()
        except (AttributeError, IndexError, TypeError):
            pass
    cropped = crop_feature(sequence)
    metadata = dict(cropped.metadata)
    metadata.update(media_metadata(asset))
    metadata.update({
        "sample_id": sample_id,
        "source_group": group,
        "arm": arm,
        "source_kind": source_kind,
        "video_path": str(video.resolve()),
        "video_sha256": file_sha256(video),
    })
    save_feature(output, FeatureArray(cropped.mouth, cropped.valid, cropped.timestamps, metadata, cropped.landmarks))
    return _feature_entry(output, sample_id=sample_id, group=group, arm=arm, source_path=video)


def _cached_or_extract(video: Path, asset: Path, output: Path, *, sample_id: str, group: str, arm: str, source_kind: str) -> dict[str, Any]:
    if output.is_file():
        entry = _feature_entry(output, sample_id=sample_id, group=group, arm=arm, source_path=video)
        if entry["frame_count"] < FRAME_COUNT:
            raise VisualProtocolError(f"cached feature is shorter than 140 frames: {output}")
        return entry
    return _extract_feature(video, asset, output, sample_id=sample_id, group=group, arm=arm, source_kind=source_kind)


def extract_real_features(run_root: Path, groups: list[str], records: dict[str, dict[str, Any]], branch: Path) -> dict[str, Any]:
    asset = Path(os.environ.get("FRESH_SOURCE_VISUAL_LANDMARKER_ASSET", str(DEFAULT_LANDMARKER_ASSET))).resolve()
    if not asset.is_file():
        raise VisualProtocolError(f"MediaPipe landmarker asset missing: {asset}")
    rows: list[dict[str, Any]] = []
    for group in groups:
        record = records[group]
        sample_id = str(record.get("sample_id") or f"lrs3_{group}")
        video = Path(str(record.get("real_video", ""))).resolve()
        expected_sha = record.get("real_video_sha256")
        if expected_sha and (not video.is_file() or file_sha256(video) != str(expected_sha)):
            raise VisualProtocolError(f"frozen real video SHA mismatch: {sample_id}")
        output = _feature_path(branch, sample_id, "R")
        entry = _cached_or_extract(video, asset, output, sample_id=sample_id, group=group, arm="R", source_kind="real")
        feature = load_feature(output)
        entry["source_group"] = group
        overlay = branch / "overlays" / f"{len(rows)+1:02d}_{anonymise_name(group)}.png"
        overlay_sha = make_contact_sheet(video, feature, overlay)
        if overlay_sha:
            entry.update({"overlay_path": str(overlay.resolve()), "overlay_sha256": overlay_sha})
        rows.append(entry)
    return {"schema_version": 1, "fixed_clock_frames": FRAME_COUNT, "fps": FPS, "asset": media_metadata(asset), "rows": rows}


def _write_index(branch: Path, index: dict[str, Any]) -> dict[str, Any]:
    return write_json(branch / "features.json", index)


def _load_index(branch: Path) -> dict[str, Any]:
    return load_self_hashed(branch / "features.json")


def _feature_map(index: dict[str, Any]) -> dict[str, dict[str, FeatureArray]]:
    result: dict[str, dict[str, FeatureArray]] = {}
    for item in index.get("rows", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        if not path.is_file() or item.get("sha256") != file_sha256(path):
            raise VisualProtocolError(f"feature binding changed: {item.get('source_group')}/{item.get('arm')}")
        result.setdefault(str(item["source_group"]), {})[str(item["arm"])] = load_feature(path)
    return result


def _load_video_rows(run_root: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], Path | None]:
    payload, path = load_video_manifest(run_root)
    if payload is None or path is None:
        return {}, None
    return normalise_video_rows(payload), path


def extract_generated_features(run_root: Path, groups: list[str], records: dict[str, dict[str, Any]], branch: Path, index: dict[str, Any]) -> dict[str, Any]:
    video_rows, manifest_path = _load_video_rows(run_root)
    if manifest_path is None:
        index["video_manifest"] = None
        index["missing_video_manifest"] = True
        return index
    index["video_manifest"] = {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)}
    asset = Path(str(index["asset"]["landmarker_asset"])).resolve()
    generated = index.setdefault("generated", [])
    existing = {
        (str(item.get("sample_id")), str(item.get("model")), str(item.get("arm"))): index
        for index, item in enumerate(generated)
    }
    for group in groups:
        sample_id = str(records[group].get("sample_id") or f"lrs3_{group}")
        for model in MODELS:
            for arm in ("N42", "C42", "N43", "C43"):
                manifest_row = video_rows.get((sample_id, model, arm)) or video_rows.get((group, model, arm))
                item: dict[str, Any] = {"sample_id": sample_id, "source_group": group, "model": model, "arm": arm, "status": "missing", "reason": "video_cell_missing"}
                if manifest_row is not None:
                    ok, reason = verify_video_item(manifest_row)
                    item.update({"manifest": dict(manifest_row), "status": "complete" if ok else "missing", "reason": reason})
                    if ok:
                        try:
                            # A's final candidate is an audio/video mux.  Its
                            # audio duration can make OpenCV report a larger
                            # declared frame count than the fixed 140-frame
                            # video stream, while A's normalized_path is the
                            # exact frame-clock artifact.  Visual dynamics do
                            # not need the audio stream, so prefer that frozen
                            # normalized artifact when it is available.
                            video = Path(str(manifest_row.get("normalized_path") or manifest_row["path"])).resolve()
                            output = _feature_path(branch, sample_id, f"{model}_{arm}")
                            entry = _cached_or_extract(video, asset, output, sample_id=sample_id, group=group, arm=f"{model}:{arm}", source_kind=f"generated:{model}")
                            item.update({"path": entry["path"], "sha256": entry["sha256"], "frame_count": entry["frame_count"], "valid_fraction": entry["valid_fraction"], "status": "complete", "reason": None})
                        except (OSError, ValueError, VisualProtocolError) as exc:
                            item.update({"status": "missing", "reason": f"feature_extract_failed:{type(exc).__name__}"})
                key = (sample_id, model, arm)
                if key in existing:
                    # A is atomically updated over time.  Refresh a previously
                    # missing row when its complete video later appears.
                    generated[existing[key]] = item
                else:
                    generated.append(item)
                    existing[key] = len(generated) - 1
    return index


def _analysis_map(index: dict[str, Any]) -> dict[str, dict[str, FeatureArray]]:
    result = _feature_map(index)
    for item in index.get("generated", []):
        if item.get("status") != "complete":
            continue
        path = Path(str(item.get("path", "")))
        if not path.is_file() or item.get("sha256") != file_sha256(path):
            raise VisualProtocolError(f"generated feature binding changed: {item.get('source_group')}/{item.get('model')}/{item.get('arm')}")
        result.setdefault(str(item["source_group"]), {})[f"{item['model']}:{item['arm']}"] = load_feature(path)
    return result


def _model_rows(groups: list[str], features: dict[str, dict[str, FeatureArray]], model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        arms = features.get(group, {})
        named = {"real": arms.get("R")}
        named.update({arm: arms.get(f"{model}:{arm}") for arm in ("N42", "C42", "N43", "C43")})
        support = np.asarray([], dtype=np.int64)
        support_info: dict[str, Any] = {"requested_count": len(J0), "count": 0, "indices": [], "missing_by_arm": {key: len(J0) for key, value in named.items() if value is None}}
        if named["real"] is not None:
            support, support_info = fixed_clock_support(named, ("real", "N42", "C42", "N43", "C43"))
        row: dict[str, Any] = {"source_group": group, "observed": bool(len(support) >= MIN_OBSERVED_FRAMES), "support": support_info, "missing_arms": [key for key, value in named.items() if value is None]}
        if len(support) >= MIN_OBSERVED_FRAMES:
            row.update(group_decomposition(named, support))
            row["missing_reason"] = None
        else:
            row["missing_reason"] = "common_support_below_81" if named["real"] is not None else "feature_missing"
            for key in ("b_dynamic", "q_natural", "q_candidate", "e_total_natural", "e_total_candidate", "e_static_natural", "e_static_candidate", "e_dynamic_natural", "e_dynamic_candidate", "e_reverse_natural", "e_reverse_candidate"):
                row[key] = None
        rows.append(row)
    return rows


def analyse_model(groups: list[str], features: dict[str, dict[str, FeatureArray]], model: str, calibration: dict[str, Any]) -> dict[str, Any]:
    rows = _model_rows(groups, features, model)
    metrics = ("b_dynamic", "q_natural", "q_candidate")
    stats = bootstrap_metrics(rows, groups, metrics)
    bounds = {metric: full_cohort_bounds(rows, groups, metric) for metric in metrics}
    q_gate = bool(stats["q_natural"]["ci99"][0] is not None and stats["q_candidate"]["ci99"][0] is not None and stats["q_natural"]["ci99"][0] > 0 and stats["q_candidate"]["ci99"][0] > 0 and stats["q_natural"]["positive_groups"] >= 10 and stats["q_candidate"]["positive_groups"] >= 10)
    b = stats["b_dynamic"]
    observed_metric_gate = bool(b["mean"] is not None and b["mean"] > 0.02 and b["ci99"][0] is not None and b["ci99"][0] > 0 and b["positive_groups"] >= 10)
    coverage_gate = bool(stats["observed_group_count"] >= 11)
    full_pass = bool(bounds["b_dynamic"]["lower"] > 0)
    if not calibration.get("calibrated"):
        status = "METRIC_NOT_CALIBRATED"
    elif not q_gate:
        status = "GENERATED_TIMING_SPECIFICITY_UNRESOLVED"
    elif observed_metric_gate and coverage_gate and full_pass:
        status = "SIGNAL"
    elif observed_metric_gate:
        status = "MISSINGNESS_LIMITED"
    else:
        status = "NO_DYNAMIC_ADVANTAGE_ESTABLISHED"
    return {
        "model": model,
        "rows": rows,
        "observed_group_count": int(stats["observed_group_count"]),
        "missing_group_count": int(len(groups) - stats["observed_group_count"]),
        "stats": stats,
        "bounds": bounds,
        "gates": {"real_calibration": bool(calibration.get("calibrated")), "generated_timing_specificity": q_gate, "observed_dynamic": observed_metric_gate, "coverage": coverage_gate, "full_cohort_bound": full_pass},
        "status": status,
        "dynamic_signal": status == "SIGNAL",
        "replacement_confirmed": False,
        "training_authorized": False,
        "generalization_established": False,
    }


def _blind_specs(groups: list[str], records: dict[str, dict[str, Any]], rows: dict[tuple[str, str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary: list[dict[str, Any]] = []
    for group in groups:
        sample = str(records[group].get("sample_id") or f"lrs3_{group}")
        for model in MODELS:
            primary.append({"trial_id": f"trial_{len(primary)+1:03d}", "kind": "primary", "source_group": group, "sample_id": sample, "model": model, "real_video": records[group].get("real_video"), "natural": rows.get((sample, model, "N42")) or rows.get((group, model, "N42")), "candidate": rows.get((sample, model, "C42")) or rows.get((group, model, "C42"))})
    qc: list[dict[str, Any]] = []
    for group in groups[:2]:
        sample = str(records[group].get("sample_id") or f"lrs3_{group}")
        for transform in ("shift_plus5", "reverse"):
            qc.append({"trial_id": f"qc_{len(qc)+1:02d}", "kind": "qc", "source_group": group, "sample_id": sample, "transform": transform, "real_video": records[group].get("real_video")})
    return primary, qc


def _canonical_rating(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    if text not in _RATING_ALIASES:
        raise VisualProtocolError(f"unknown blind rating answer: {value!r}")
    answer = _RATING_ALIASES[text]
    if answer not in _RATING_CHOICES:
        raise VisualProtocolError(f"invalid canonical blind rating: {answer}")
    return answer


def _rating_input(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VisualProtocolError(f"human ratings file missing: {path}")
    raw = read_json(path)
    if "artifact_sha256" in raw:
        raw = load_self_hashed(path)
    if not isinstance(raw, Mapping):
        raise VisualProtocolError("human ratings must be a JSON object")
    return dict(raw)


def _rating_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[tuple[Any, Any]] = []
    ratings = payload.get("ratings", payload.get("rows"))
    if isinstance(ratings, list):
        values.extend((payload.get("reviewer", payload.get("reviewer_id")), item) for item in ratings)
    reviewers = payload.get("reviewers")
    if isinstance(reviewers, list):
        for block in reviewers:
            if not isinstance(block, Mapping):
                raise VisualProtocolError("ratings reviewer block must be an object")
            reviewer = block.get("reviewer", block.get("reviewer_id"))
            block_rows = block.get("ratings", block.get("rows", []))
            if not isinstance(block_rows, list):
                raise VisualProtocolError("ratings reviewer rows must be a list")
            values.extend((reviewer, item) for item in block_rows)
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for default_reviewer, value in values:
        if not isinstance(value, Mapping):
            raise VisualProtocolError("each human rating must be an object")
        reviewer_value = value.get("reviewer", value.get("reviewer_id", default_reviewer))
        try:
            reviewer = int(reviewer_value)
        except (TypeError, ValueError) as exc:
            raise VisualProtocolError("human rating reviewer must be an integer") from exc
        if reviewer not in (1, 2):
            raise VisualProtocolError(f"unexpected reviewer id: {reviewer}")
        trial_id = str(value.get("trial_id", value.get("id", ""))).strip()
        if not trial_id:
            raise VisualProtocolError("human rating trial_id is required")
        key = (reviewer, trial_id)
        if key in seen:
            raise VisualProtocolError(f"duplicate human rating: reviewer={reviewer}/{trial_id}")
        seen.add(key)
        answer = _canonical_rating(value.get("answer", value.get("rating", value.get("choice"))))
        result.append({"reviewer": reviewer, "trial_id": trial_id, "answer": answer})
    return result


def ingest_ratings(run_root: Path, ratings_path: Path) -> dict[str, Any]:
    """Normalize an external human-rating file without inventing missing rows."""

    _root, branch = _root_and_branch(run_root)
    package_path = branch / "blind" / "package.json"
    secret_path = branch / "blind" / "secret_mapping.json"
    if not package_path.is_file() or not secret_path.is_file():
        raise VisualProtocolError("blind package and secret mapping are required before ratings")
    package = load_self_hashed(package_path)
    secret = load_self_hashed(secret_path)
    trials = secret.get("trials")
    if not isinstance(trials, Mapping):
        raise VisualProtocolError("blind secret mapping has no trials")
    rows = _rating_rows(_rating_input(ratings_path))
    unknown = sorted({row["trial_id"] for row in rows if row["trial_id"] not in trials})
    if unknown:
        raise VisualProtocolError(f"human ratings contain unknown trials: {unknown}")
    normalized = write_json(
        branch / "blind" / "ratings.json",
        {
            "schema_version": 1,
            "status": "complete",
            "source_path": str(ratings_path.resolve()),
            "source_sha256": file_sha256(ratings_path),
            "ratings": rows,
        },
    )
    return {"ratings": normalized, "package": package, "secret": secret}


def _semantic_answer(answer: str, trial: Mapping[str, Any]) -> str:
    if answer in {"same", "unable", "missing"}:
        return answer
    side_map = trial.get("condition_by_side")
    if not isinstance(side_map, Mapping):
        return "unable"
    value = side_map.get(answer)
    return str(value) if value in {"N", "C", "identity", "transform"} else "unable"


def _human_bootstrap(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draw_indices = rng.integers(0, len(array), size=(BOOTSTRAP_COUNT, len(array)), dtype=np.int64)
    draws = array[draw_indices].mean(axis=1)
    return [
        float(np.quantile(draws, 0.005, method="linear")),
        float(np.quantile(draws, 0.995, method="linear")),
    ]


def summarize_human_ratings(
    branch: Path,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Qualify reviewers and compute blinded human group summaries."""

    ratings = load_self_hashed(branch / "blind" / "ratings.json")
    secret = load_self_hashed(branch / "blind" / "secret_mapping.json")
    trial_map = secret.get("trials")
    normalized = ratings.get("ratings")
    if not isinstance(trial_map, Mapping) or not isinstance(normalized, list):
        raise VisualProtocolError("ratings artifact is malformed")
    by_reviewer: dict[int, dict[str, str]] = {1: {}, 2: {}}
    for row in normalized:
        if not isinstance(row, Mapping):
            raise VisualProtocolError("ratings artifact contains a malformed row")
        reviewer = int(row.get("reviewer", -1))
        trial_id = str(row.get("trial_id", ""))
        answer = _canonical_rating(row.get("answer"))
        if reviewer not in by_reviewer or trial_id not in trial_map:
            raise VisualProtocolError("ratings artifact contains an unknown reviewer or trial")
        if trial_id in by_reviewer[reviewer]:
            raise VisualProtocolError(f"duplicate normalized rating: reviewer={reviewer}/{trial_id}")
        by_reviewer[reviewer][trial_id] = answer
    trial_ids = list(trial_map)
    qc_ids = [trial_id for trial_id in trial_ids if trial_map[trial_id].get("kind") == "qc"]
    primary_ids = [trial_id for trial_id in trial_ids if trial_map[trial_id].get("kind") == "primary"]
    reviewer_reports: list[dict[str, Any]] = []
    for reviewer in (1, 2):
        answers = by_reviewer[reviewer]
        missing = [trial_id for trial_id in trial_ids if trial_id not in answers]
        unjudgeable = [trial_id for trial_id, answer in answers.items() if answer in {"unable", "missing"}]
        qc_correct = sum(
            _semantic_answer(answers[trial_id], trial_map[trial_id]) == "identity"
            for trial_id in qc_ids
            if trial_id in answers
        )
        qc_answered = sum(answers.get(trial_id) in {"left", "right"} for trial_id in qc_ids)
        qualified = bool(qc_correct >= 3)
        complete = not missing
        reviewer_reports.append(
            {
                "reviewer": reviewer,
                "qc_total": len(qc_ids),
                "qc_answered": int(qc_answered),
                "qc_identity_correct": int(qc_correct),
                "qualified": qualified,
                "complete": complete,
                "qualified_complete": bool(qualified and complete),
                "missing_trial_ids": missing,
                "unjudgeable_trial_ids": unjudgeable,
                "rated_trial_count": len(answers),
            }
        )
    qualified = [report["reviewer"] for report in reviewer_reports if report["qualified_complete"]]
    agreement_by_model: dict[str, Any] = {}
    for model in MODELS:
        pairs: list[bool] = []
        model_ids = [
            trial_id
            for trial_id in primary_ids
            if str(trial_map[trial_id].get("model")) == model
        ]
        if len(qualified) >= 2:
            for trial_id in model_ids:
                pairs.append(
                    _semantic_answer(by_reviewer[qualified[0]][trial_id], trial_map[trial_id])
                    == _semantic_answer(by_reviewer[qualified[1]][trial_id], trial_map[trial_id])
                )
        agreement_by_model[model] = {
            "total": len(pairs),
            "agree": int(sum(pairs)),
            "fraction": float(np.mean(pairs)) if pairs else None,
        }
    model_reports: dict[str, Any] = {}
    for model in MODELS:
        model_trials = [
            (trial_id, trial_map[trial_id])
            for trial_id in primary_ids
            if str(trial_map[trial_id].get("model")) == model
        ]
        groups: list[str] = []
        group_values: list[float] = []
        if len(qualified) >= 2:
            for trial_id, trial in model_trials:
                semantic = [
                    _semantic_answer(by_reviewer[reviewer][trial_id], trial)
                    for reviewer in qualified[:2]
                ]
                scores = [1.0 if value == "C" else -1.0 if value == "N" else 0.0 for value in semantic]
                groups.append(str(trial.get("source_group", trial_id)))
                group_values.append(float(np.mean(scores)))
        mean = float(np.mean(group_values)) if group_values else None
        ci = _human_bootstrap(group_values)
        positive = int(sum(value > 0.0 for value in group_values))
        support = bool(mean is not None and ci[0] is not None and mean > 0.0 and ci[0] > 0.0 and positive >= 9)
        model_reports[model] = {
            "reviewer_count": len(qualified),
            "group_count": len(group_values),
            "group_values": group_values,
            "groups": groups,
            "mean": mean,
            "positive_groups": positive,
            "bootstrap_count": BOOTSTRAP_COUNT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci99": ci,
            "support": support,
        }
    human_status = "complete" if len(qualified) >= 2 else "insufficient"
    return {
        "schema_version": 1,
        "status": human_status,
        "human_status": human_status,
        "reviewers": reviewer_reports,
        "qualified_complete_reviewers": qualified,
        "agreement": agreement_by_model,
        "models": model_reports,
    }


def apply_human_ratings(run_root: Path, ratings_path: Path) -> dict[str, Any]:
    """Ingest ratings and update only human-dependent, hash-bound artifacts."""

    root, branch = _root_and_branch(run_root)
    ingest_ratings(root, ratings_path)
    analysis = load_self_hashed(branch / "analysis.json")
    summary = summarize_human_ratings(branch, analysis)
    models = analysis.get("models", {})
    verified_models: list[str] = []
    if isinstance(models, Mapping):
        for model in MODELS:
            report = summary["models"][model]
            model_row = dict(models.get(model, {}))
            model_row["human"] = report
            model_row["human_support"] = bool(report["support"])
            model_row["visual_verified"] = bool(model_row.get("dynamic_signal") and report["support"])
            if model_row["visual_verified"]:
                verified_models.append(model)
            models[model] = model_row
    analysis["models"] = models
    analysis["human_summary"] = summary
    analysis["human_status"] = summary["human_status"]
    analysis["visual_verified_models"] = verified_models
    analysis["visual_verified"] = bool(verified_models)
    analysis = write_json(branch / "analysis.json", analysis)
    final_path = branch / "final.json"
    if not final_path.is_file():
        raise VisualProtocolError("blind stage must create final.json before ratings")
    final = load_self_hashed(final_path)
    package = load_self_hashed(branch / "blind" / "package.json")
    final.update(
        {
            "analysis_sha256": analysis["artifact_sha256"],
            "blind_package_sha256": package["artifact_sha256"],
            "human_status": summary["human_status"],
            "visual_verified": bool(verified_models),
            "visual_verified_models": verified_models,
            "replacement_confirmed": False,
            "training_authorized": False,
            "generalization_established": False,
        }
    )
    final = write_json(final_path, final)
    return {"analysis": analysis, "summary": summary, "final": final}


def build_blind_package(run_root: Path, groups: list[str], records: dict[str, dict[str, Any]], branch: Path) -> dict[str, Any]:
    payload, _ = load_video_manifest(run_root)
    rows = normalise_video_rows(payload) if payload is not None else {}
    primary, qc = _blind_specs(groups, records, rows)
    blind = branch / "blind"
    blind.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(BLIND_SEED))
    secret: dict[str, Any] = {"schema_version": 1, "seed": BLIND_SEED, "trials": {}}
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    prompt = "哪侧嘴部运动在开合时刻和运动轨迹上更接近真实视频？左／右／相同／无法判断"
    for spec in primary + qc:
        trial_id = str(spec["trial_id"])
        internal = dict(spec)
        if spec["kind"] == "primary":
            base = {"left": "N", "right": "C"} if bool(rng.integers(0, 2)) else {"left": "C", "right": "N"}
            secret["trials"][trial_id] = {"kind": "primary", "source_group": spec["source_group"], "model": spec["model"], "condition_by_side": base}
            real = Path(str(spec["real_video"] or ""))
            available = bool(real.is_file() and spec.get("natural") and spec.get("candidate"))
            for reviewer in (1, 2):
                flip = bool(rng.integers(0, 2))
                media_rel: str | None = f"media/{trial_id}.mp4" if available else None
                media_sha: str | None = None
                if available:
                    try:
                        r_frames = _read_video_frames(real)
                        n_frames = _read_video_frames(Path(str(spec["natural"]["path"])))
                        c_frames = _read_video_frames(Path(str(spec["candidate"]["path"])))
                        left, right = (n_frames, c_frames) if base["left"] == "N" else (c_frames, n_frames)
                        if flip:
                            left, right = right, left
                        media_path = blind / f"reviewer_{reviewer}" / media_rel
                        media_sha = _encode_silent_panels(zip(r_frames, left, right), media_path)
                    except (OSError, ValueError, VisualProtocolError):
                        media_rel = None
                public.append({"trial_id": trial_id, "kind": "primary", "media": media_rel, "media_sha256": media_sha, "prompt": prompt, "options": ["left", "right", "same", "unable"], "reviewer": reviewer})
                internal.setdefault("reviewer_media", {})[str(reviewer)] = {"flip": flip, "path": str((blind / f"reviewer_{reviewer}" / media_rel).resolve()) if media_rel else None, "sha256": media_sha}
        else:
            base = {"left": "identity", "right": "transform"} if bool(rng.integers(0, 2)) else {"left": "transform", "right": "identity"}
            secret["trials"][trial_id] = {"kind": "qc", "source_group": spec["source_group"], "transform": spec["transform"], "condition_by_side": base}
            real = Path(str(spec["real_video"] or ""))
            for reviewer in (1, 2):
                flip = bool(rng.integers(0, 2))
                media_rel: str | None = f"media/{trial_id}.mp4" if real.is_file() else None
                media_sha: str | None = None
                if media_rel:
                    try:
                        r_frames = _read_video_frames(real)
                        transformed = _transform_frames(r_frames, str(spec["transform"]))
                        left, right = (r_frames, transformed) if base["left"] == "identity" else (transformed, r_frames)
                        if flip:
                            left, right = right, left
                        media_path = blind / f"reviewer_{reviewer}" / media_rel
                        media_sha = _encode_silent_panels(zip(r_frames, left, right), media_path)
                    except (OSError, ValueError, VisualProtocolError):
                        media_rel = None
                public.append({"trial_id": trial_id, "kind": "qc", "media": media_rel, "media_sha256": media_sha, "prompt": prompt, "options": ["left", "right", "same", "unable"], "reviewer": reviewer})
                internal.setdefault("reviewer_media", {})[str(reviewer)] = {"flip": flip, "path": str((blind / f"reviewer_{reviewer}" / media_rel).resolve()) if media_rel else None, "sha256": media_sha}
        private.append(internal)
    secret_path = blind / "secret_mapping.json"
    write_json(secret_path, secret)
    write_json(blind / "internal_trials.json", {"schema_version": 1, "trials": private})
    for reviewer in (1, 2):
        trials = [item for item in public if item["reviewer"] == reviewer]
        write_json(blind / f"reviewer_{reviewer}" / "package.json", {"schema_version": 1, "package_id": f"fresh_source_visual_reviewer_{reviewer}", "status": "READY" if all(item["media"] for item in trials) else "PENDING_MEDIA", "trial_count": len(trials), "primary_trials": sum(item["kind"] == "primary" for item in trials), "qc_trials": sum(item["kind"] == "qc" for item in trials), "trials": trials})
    # The branch-level manifest is an index with one 28-trial public view;
    # each reviewer directory contains its own independently flipped view.
    # Keeping the root list at 24+4 avoids counting reviewer duplicates as
    # additional trials while retaining both package paths explicitly.
    root_trials = [item for item in public if item["reviewer"] == 1]
    return write_json(blind / "package.json", {"schema_version": 1, "status": "READY" if all(item["media"] for item in public) else "PENDING_MEDIA", "primary_trials": len(primary), "qc_trials": len(qc), "reviewer_packages": [str((blind / f"reviewer_{n}" / "package.json").resolve()) for n in (1, 2)], "secret_mapping_path": str(secret_path.resolve()), "secret_mapping_sha256": file_sha256(secret_path), "human_status": "pending", "visual_verified": False, "trials": root_trials, "reviewer_prompt": prompt})


def _waiting(branch: Path, status: str, reason: list[str]) -> None:
    write_json(branch / "analysis.json", {"schema_version": 1, "status": status, "models": {}, "visual_verified": False, "human_status": "pending", "replacement_confirmed": False, "reason": reason})
    if not (branch / "blind" / "package.json").is_file():
        write_json(branch / "blind" / "package.json", {"schema_version": 1, "status": status, "primary_trials": 0, "qc_trials": 0, "human_status": "pending", "visual_verified": False, "trials": []})
    write_json(branch / "final.json", {"schema_version": 1, "status": status, "automated_status": status, "human_status": "pending", "visual_verified": False, "replacement_confirmed": False, "training_authorized": False, "generalization_established": False})


def run_blocked(run_root: Path) -> int:
    root = _cohort_root(run_root.resolve())
    path = _find_manifest(root, "cohort.json")
    cohort = read_json(path) if path else {"status": "BLOCKED_NEW_SOURCE", "blockers": ["cohort.json missing"]}
    branch = root / "run" / "B"
    reason = list(cohort.get("blockers", [])) or [str(cohort.get("status", "upstream cohort not ready"))]
    status = "BLOCKED_UPSTREAM_NEW_SOURCE"
    write_json(branch / "calibration.json", {"schema_version": 1, "status": status, "identity_mse": None, "calibrated": False, "reason": reason})
    _waiting(branch, status, reason)
    write_json(branch / "validation.json", {"schema_version": 1, "status": "GO", "scientific_cells": 0, "errors": [], "blocked_compatibility": True})
    print(f"B status={status}")
    return 0


def calibrate(run_root: Path) -> int:
    root, branch = _root_and_branch(run_root)
    try:
        groups, records = _formal_records(root)
        index = extract_real_features(root, groups, records, branch)
        _write_index(branch, index)
        features = _feature_map(index)
        calibration = run_calibration({group: arms["R"] for group, arms in features.items() if "R" in arms}, groups)
        calibration.update({"inputs_sha256": file_sha256(_find_manifest(root, "inputs.json")) if _find_manifest(root, "inputs.json") else None, "cohort_sha256": file_sha256(_find_manifest(root, "cohort.json")) if _find_manifest(root, "cohort.json") else None, "feature_index_sha256": file_sha256(branch / "features.json")})
        write_json(branch / "calibration.json", calibration)
        build_blind_package(root, groups, records, branch)
        _waiting(branch, "WAITING_FOR_GENERATED_VIDEOS", [] if calibration["calibrated"] else ["METRIC_NOT_CALIBRATED"])
        write_json(branch / "validation.json", {"schema_version": 1, "status": "PENDING_GENERATED_VIDEOS", "scientific_cells": 0, "errors": [], "calibration_sha256": file_sha256(branch / "calibration.json")})
        print(f"B calibration={calibration['status']} groups={len(groups)}")
        return 0
    except (OSError, ProtocolError, VisualProtocolError) as exc:
        _waiting(branch, "BLOCKED_ENGINEERING", [str(exc)])
        return 1


def extract(run_root: Path) -> int:
    root, branch = _root_and_branch(run_root)
    try:
        groups, records = _formal_records(root)
        index = _load_index(branch)
        index = extract_generated_features(root, groups, records, branch, index)
        _write_index(branch, index)
        print(f"B extract manifest={'present' if index.get('video_manifest') else 'missing'}")
        return 0
    except (OSError, ProtocolError, VisualProtocolError) as exc:
        _waiting(branch, "BLOCKED_ENGINEERING", [str(exc)])
        return 1


def analyse(run_root: Path) -> int:
    root, branch = _root_and_branch(run_root)
    try:
        groups, _records = _formal_records(root)
        calibration = load_self_hashed(branch / "calibration.json")
        index = _load_index(branch)
        # ``analyze`` is strictly model-free.  Generated videos are consumed
        # only after the explicit ``extract`` stage has materialised their
        # feature arrays; this keeps a statistical rerun from invoking
        # MediaPipe or touching A's media.
        features = _analysis_map(index)
        models = {model: analyse_model(groups, features, model, calibration) for model in MODELS}
        # One generator may be unavailable after its engineering smoke.  B is
        # still complete for the model with observed cells; the unavailable
        # model remains explicitly missing rather than blocking the other
        # model's descriptive visual result.
        complete = any(result["observed_group_count"] > 0 for result in models.values())
        analysis = write_json(branch / "analysis.json", {"schema_version": 1, "status": "complete" if complete else "WAITING_FOR_GENERATED_VIDEOS", "fixed_clock_frames": FRAME_COUNT, "J0": J0.tolist(), "groups": groups, "models": models, "automated_status": "complete" if complete else "pending", "human_status": "pending", "visual_verified": False, "replacement_confirmed": False, "training_authorized": False, "generalization_established": False, "calibration_sha256": file_sha256(branch / "calibration.json"), "feature_index_sha256": file_sha256(branch / "features.json")})
        write_json(branch / "validation.json", {"schema_version": 1, "status": "PENDING_BLIND" if complete else "PENDING_GENERATED_VIDEOS", "scientific_cells": sum(result["observed_group_count"] for result in models.values()), "errors": [], "analysis_sha256": analysis["artifact_sha256"]})
        print(f"B analysis={analysis['status']}")
        return 0
    except (OSError, ProtocolError, VisualProtocolError) as exc:
        _waiting(branch, "BLOCKED_ENGINEERING", [str(exc)])
        return 1


def blind(run_root: Path) -> int:
    root, branch = _root_and_branch(run_root)
    try:
        groups, records = _formal_records(root)
        package = build_blind_package(root, groups, records, branch)
        analysis = load_self_hashed(branch / "analysis.json") if (branch / "analysis.json").is_file() else {"automated_status": "pending"}
        status = "complete" if analysis.get("automated_status") == "complete" else "WAITING_FOR_GENERATED_VIDEOS"
        validation = write_json(branch / "validation.json", {"schema_version": 1, "status": "GO", "scientific_cells": sum(int(result.get("observed_group_count", 0)) for result in analysis.get("models", {}).values()), "errors": [], "analysis_sha256": analysis.get("artifact_sha256"), "blind_package_sha256": package.get("artifact_sha256")})
        write_json(branch / "final.json", {"schema_version": 1, "status": status, "automated_status": analysis.get("automated_status", "pending"), "human_status": "pending", "visual_verified": False, "replacement_confirmed": False, "training_authorized": False, "generalization_established": False, "analysis_sha256": analysis.get("artifact_sha256"), "blind_package_sha256": package.get("artifact_sha256"), "validation_sha256": validation.get("artifact_sha256")})
        print(f"B blind primary={package['primary_trials']} qc={package['qc_trials']} human=pending")
        return 0
    except (OSError, ProtocolError, VisualProtocolError) as exc:
        _waiting(branch, "BLOCKED_ENGINEERING", [str(exc)])
        return 1


# Public spelling used by the CLI/spec; retain ``analyse`` internally for
# compatibility with the project's existing British-spelled analysis modules.
analyze = analyse


def ratings(run_root: Path, ratings_path: Path) -> int:
    root, _branch = _root_and_branch(run_root)
    try:
        result = apply_human_ratings(root, ratings_path)
        summary = result["summary"]
        print(f"B ratings status={summary['human_status']} verified={','.join(result['final'].get('visual_verified_models', [])) or 'none'}")
        return 0
    except (OSError, ProtocolError, VisualProtocolError) as exc:
        print(f"B ratings blocked: {exc}", file=sys.stderr)
        return 1


def run(run_root: Path, stage: str, ratings_path: Path | None = None) -> int:
    root = _cohort_root(run_root.resolve())
    path = _find_manifest(root, "cohort.json")
    if path is None:
        return run_blocked(root)
    cohort = read_json(path)
    if str(cohort.get("status")) not in {"GO", "COHORT_READY"} and str(cohort.get("readiness")) != "COHORT_READY":
        return run_blocked(root)
    if stage == "ratings":
        if ratings_path is None:
            raise VisualProtocolError("--ratings is required for --stage ratings")
        return ratings(root, ratings_path)
    return {"calibrate": calibrate, "extract": extract, "analyze": analyse, "blind": blind}[stage](root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibrate", "extract", "analyze", "blind", "ratings"), default="calibrate")
    parser.add_argument("--ratings", type=Path, help="human ratings JSON for --stage ratings")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "ratings" and args.ratings is None:
        parser.error("--ratings is required for --stage ratings")
    return run(args.run_root, args.stage, args.ratings)


if __name__ == "__main__":
    raise SystemExit(main())
