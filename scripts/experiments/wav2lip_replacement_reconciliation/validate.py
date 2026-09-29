from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ReconciliationError,
    file_sha256,
    load_matrix,
    verify_self_hashed_json,
)

_DEFAULT_MEAN_DTYPE = np.dtype(np.float64)


def _summary(matrix: np.ndarray, rows: Sequence[int], *, mean_dtype: np.dtype[Any] = _DEFAULT_MEAN_DTYPE) -> dict[str, Any]:
    values = np.asarray(matrix)
    selected = [int(row) for row in rows]
    if values.ndim != 2 or values.shape[1] != config.MATRIX_COLUMNS or not selected or min(selected) < 0 or max(selected) >= values.shape[0]:
        raise ReconciliationError("validator endpoint support is invalid")
    curve = np.asarray(np.mean(values[selected, :], axis=0, dtype=mean_dtype), dtype=np.float64)
    index = int(np.argmin(curve))
    minimum = float(curve[index])
    return {"curve": curve, "curve_sha256": hashlib.sha256(np.ascontiguousarray(curve).tobytes()).hexdigest(), "min_index": index, "offset": int(config.VSHIFT - index), "sync_c": float(np.median(curve) - minimum), "sync_d": minimum}


def _rebuild_distance(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    visual_array = np.asarray(visual, dtype=np.float32)
    audio_array = np.asarray(audio, dtype=np.float32)
    if visual_array.ndim != 2 or audio_array.ndim != 2 or visual_array.shape != audio_array.shape:
        raise ReconciliationError("validator embedding shapes differ")
    padded = np.pad(audio_array, ((config.VSHIFT, config.VSHIFT), (0, 0)))
    rows = []
    for row in range(visual_array.shape[0]):
        diff = visual_array[row : row + 1] - padded[row : row + 2 * config.VSHIFT + 1]
        rows.append(np.sqrt(np.sum((diff + np.float32(1e-6)) ** 2, axis=1, dtype=np.float32))
        )
    return np.asarray(rows, dtype=np.float32)


def _bootstrap(values: Sequence[float], groups: Sequence[str], labels: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        by_group[str(group)].append(float(value))
    if sorted(by_group) != list(labels):
        raise ReconciliationError("validator bootstrap labels differ")
    group_means = np.asarray([np.mean(by_group[label], dtype=np.float64) for label in labels], dtype=np.float64)
    estimates = group_means[indices].mean(axis=1, dtype=np.float64)
    return {"mean": float(np.mean(values, dtype=np.float64)), "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))]}


def _close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def _root_and_binding_checks(root: Path, protocol: Mapping[str, Any], audit: Mapping[str, Any], errors: list[str]) -> None:
    if protocol.get("protocol_id") != config.PROTOCOL_ID or protocol.get("status") != "frozen":
        errors.append("protocol identity/status invalid")
    if protocol.get("input_audit_sha256") != file_sha256(root / "input_audit.json"):
        errors.append("protocol/input_audit binding differs")
    if audit.get("record_count") != config.EXPECTED_RECORD_COUNT or audit.get("matrix_count") != {"H": 66, "S": 110, "total": 176}:
        errors.append("audit counts differ")
    for label, (path, expected) in config.ROOT_INPUTS.items():
        try:
            if file_sha256(path) != expected:
                errors.append(f"root hash changed: {label}")
            verify_self_hashed_json(path)
        except Exception as exc:  # noqa: BLE001 - validator records every malformed root input
            errors.append(f"root input invalid: {label}: {exc}")
    for path in config.CODE_FILES:
        key = str(path.relative_to(config.REPO))
        if protocol.get("code_bindings", {}).get(key) != file_sha256(path):
            errors.append(f"code binding changed: {key}")
    for path in config.SPEC_FILES:
        key = str(path.relative_to(config.REPO))
        if protocol.get("spec_bindings", {}).get(key) != file_sha256(path):
            errors.append(f"spec binding changed: {key}")
    assets = protocol.get("asset_bindings", [])
    if not isinstance(assets, list) or len(assets) < 176:
        errors.append("asset binding list is incomplete")
    for item in assets:
        try:
            path = Path(str(item["path"]))
            if file_sha256(path) != str(item["sha256"]):
                errors.append(f"asset hash changed: {path}")
        except Exception as exc:  # noqa: BLE001 - validator records every malformed bound asset
            errors.append(f"asset binding invalid: {exc}")


def _endpoint_checks(root: Path, protocol: Mapping[str, Any], errors: list[str]) -> dict[tuple[str, str, str, str, str], Mapping[str, Any]]:
    payload = verify_self_hashed_json(root / "endpoints.json")
    endpoints = payload.get("endpoints", [])
    if len(endpoints) != config.EXPECTED_ENDPOINT_COUNT:
        errors.append("endpoint count differs from 528")
    endpoint_index: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    for row in endpoints:
        key = (str(row.get("sample_id")), str(row.get("origin")), str(row.get("video_arm")), str(row.get("audio_arm")), str(row.get("endpoint")))
        if key in endpoint_index:
            errors.append(f"duplicate endpoint: {key}")
        endpoint_index[key] = row
    records = list(protocol.get("records", []))
    binding_index = {(str(row["origin"]), str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in protocol.get("matrix_bindings", [])}
    for record in records:
        sid = str(record["sample_id"])
        for origin, arms in (("H", config.H_VIDEO_ARMS), ("S", config.S_VIDEO_ARMS)):
            for arm in arms:
                binding = binding_index[(origin, sid, arm, "N")]
                matrix = load_matrix(Path(str(binding["matrix"])), str(binding["matrix_sha256"]), f"validator matrix {origin}/{sid}/{arm}")
                for endpoint, rows in (("FULL", list(range(matrix.shape[0]))), ("COMMON_INTERIOR", list(record["common_interior_rows"])), ("U", list(record["u_rows"]))):
                    key = (sid, origin, arm, "N", endpoint)
                    row = endpoint_index.get(key)
                    if row is None:
                        errors.append(f"missing endpoint: {key}")
                        continue
                    expected = _summary(matrix, rows)
                    if int(row.get("row_count", -1)) != len(rows) or list(row.get("rows", [])) != rows:
                        errors.append(f"endpoint row support differs: {key}")
                    if int(row.get("offset", 999)) != expected["offset"] or not _close(float(row.get("sync_c")), expected["sync_c"]) or not _close(float(row.get("sync_d")), expected["sync_d"]):
                        errors.append(f"endpoint aggregate differs: {key}")
                    observed_curve = np.asarray(row.get("curve", []), dtype=np.float64)
                    if observed_curve.shape != (config.MATRIX_COLUMNS,) or not np.allclose(observed_curve, expected["curve"], atol=1e-12, rtol=0.0):
                        errors.append(f"endpoint curve differs: {key}")
                if origin == "S":
                    visual = np.load(Path(str(binding["visual"])), allow_pickle=False)
                    audio = np.load(Path(str(binding["audio_embedding"])), allow_pickle=False)
                    matrix = np.load(Path(str(binding["matrix"])), allow_pickle=False)
                    worker = verify_self_hashed_json(Path(str(binding["worker_result"])))
                    if worker.get("matrix_sha256") != binding.get("matrix_sha256") or not np.allclose(_rebuild_distance(visual, audio), matrix, atol=1e-4, rtol=0.0):
                        errors.append(f"S embedding-to-matrix rebuild differs: {key}")
                else:
                    try:
                        with Path(str(binding["track"])).open("rb") as handle:
                            tracks = pickle.load(handle)
                        if len(tracks) != 1 or len(np.asarray(tracks[0]["track"]["frame"])) != int(binding.get("track_frame_count", -1)):
                            errors.append(f"H track binding differs: {key}")
                    except Exception as exc:  # noqa: BLE001 - validator records malformed legacy caches
                        errors.append(f"H track cannot be loaded: {key}: {exc}")
    return endpoint_index


def _comparison_rows(endpoint_index: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]], records: Sequence[Mapping[str, Any]], origin: str, candidate: str, baseline: str, endpoint: str) -> list[dict[str, float | int | str | bool]]:
    rows = []
    for record in records:
        sid = str(record["sample_id"])
        candidate_row = endpoint_index[(sid, origin, candidate, "N", endpoint)]
        baseline_row = endpoint_index[(sid, origin, baseline, "N", endpoint)]
        c = float(candidate_row["sync_c"]) - float(baseline_row["sync_c"])
        d = float(baseline_row["sync_d"]) - float(candidate_row["sync_d"])
        offset_delta = int(candidate_row["offset"]) - int(baseline_row["offset"])
        rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "benefit_c": c, "benefit_d": d, "offset_delta": offset_delta, "offset_pass": abs(offset_delta) <= 1, "joint_win": c > 0 and d > 0})
    return rows


def _analysis_checks(root: Path, protocol: Mapping[str, Any], endpoints: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]], errors: list[str]) -> None:
    analysis = verify_self_hashed_json(root / "analysis.json")
    records = list(protocol["records"])
    labels = sorted({str(row["source_group"]) for row in records})
    indices = np.load(root / "bootstrap_indices.npy", allow_pickle=False)
    expected_indices = np.random.default_rng(config.BOOTSTRAP_SEED).integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), endpoint=False)
    if indices.shape != expected_indices.shape or not np.array_equal(indices, expected_indices):
        errors.append("bootstrap index matrix is not the frozen PCG64 matrix")
    specs = (
        ("H_BRIDGE_075_vs_N", "H", "BRIDGE_075", "N"),
        ("H_N_REPEAT_vs_N", "H", "N_REPEAT", "N"),
        ("S_MAG_vs_N", "S", "MAG", "N"),
        ("S_ENV_vs_N", "S", "ENV", "N"),
        ("S_MAG_vs_RT", "S", "MAG", "RT"),
        ("S_ENV_vs_RT", "S", "ENV", "RT"),
        ("S_N_REPEAT_vs_N", "S", "N_REPEAT", "N"),
        ("S_RT_vs_N", "S", "RT", "N"),
    )
    for name, origin, candidate, baseline in specs:
        for endpoint in config.ENDPOINTS:
            rows = _comparison_rows(endpoints, records, origin, candidate, baseline, endpoint)
            actual_c = _bootstrap([float(row["benefit_c"]) for row in rows], [str(row["source_group"]) for row in rows], labels, indices)
            actual_d = _bootstrap([float(row["benefit_d"]) for row in rows], [str(row["source_group"]) for row in rows], labels, indices)
            observed = analysis.get("comparisons", {}).get(name, {}).get(endpoint, {})
            for metric, actual in (("benefit_C", actual_c), ("benefit_D", actual_d)):
                observed_stat = observed.get(metric, {})
                if not _close(float(observed_stat.get("mean", 999)), actual["mean"]) or not all(_close(float(a), float(b)) for a, b in zip(observed_stat.get("ci95", []), actual["ci95"], strict=False)):
                    errors.append(f"analysis comparison differs: {name}/{endpoint}/{metric}")
            observed_records = observed.get("per_record", [])
            if len(observed_records) != len(rows):
                errors.append(f"analysis per-record count differs: {name}/{endpoint}")
            else:
                for actual_row, observed_row in zip(rows, observed_records, strict=True):
                    if not _close(float(actual_row["benefit_c"]), float(observed_row.get("benefit_c", 999))) or not _close(float(actual_row["benefit_d"]), float(observed_row.get("benefit_d", 999))):
                        errors.append(f"analysis per-record value differs: {name}/{endpoint}/{actual_row['sample_id']}")
            if observed.get("offset_agreement_count") != sum(bool(row["offset_pass"]) for row in rows) or observed.get("joint_win_count") != sum(bool(row["joint_win"]) for row in rows):
                errors.append(f"analysis counts differ: {name}/{endpoint}")
    for metric in ("C", "D"):
        observed = analysis.get("decomposition", {}).get(metric, {})
        terms = observed.get("terms", {})
        values = {name: [] for name in ("total_gap", "support_term", "window_term", "pipeline_term", "interaction", "common_pipeline_gap", "identity_check")}
        for record in records:
            sid = str(record["sample_id"])
            def benefit(origin: str, candidate: str, baseline: str, endpoint: str, sample_id: str = sid, selected_metric: str = metric) -> float:
                c = endpoints[(sample_id, origin, candidate, "N", endpoint)]
                b = endpoints[(sample_id, origin, baseline, "N", endpoint)]
                return (float(c["sync_c"]) - float(b["sync_c"])) if selected_metric == "C" else (float(b["sync_d"]) - float(c["sync_d"]))
            h_full = benefit("H", "BRIDGE_075", "N", "FULL")
            h_i = benefit("H", "BRIDGE_075", "N", "COMMON_INTERIOR")
            h_u = benefit("H", "BRIDGE_075", "N", "U")
            s_i = benefit("S", "MAG", "N", "COMMON_INTERIOR")
            s_u = benefit("S", "MAG", "N", "U")
            values["total_gap"].append(s_u - h_full)
            values["support_term"].append(h_i - h_full)
            values["window_term"].append(h_u - h_i)
            values["pipeline_term"].append(s_u - h_u)
            values["identity_check"].append((s_u - h_full) - ((h_i - h_full) + (h_u - h_i) + (s_u - h_u)))
            values["interaction"].append((s_u - s_i) - (h_u - h_i))
            values["common_pipeline_gap"].append(s_i - h_i)
        for name, vals in values.items():
            if name == "identity_check":
                if float(np.max(np.abs(vals))) > 1e-6 or not terms.get(name, {}).get("passes"):
                    errors.append(f"decomposition identity differs: {metric}")
                continue
            actual = _bootstrap(vals, [str(row["source_group"]) for row in records], labels, indices)
            observed_term = terms.get(name, {})
            if not _close(float(observed_term.get("mean", 999)), actual["mean"]) or not all(_close(float(a), float(b)) for a, b in zip(observed_term.get("ci95", []), actual["ci95"], strict=False)):
                errors.append(f"decomposition term differs: {metric}/{name}")


def _parity_parent_checks(protocol: Mapping[str, Any], endpoint_index: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]], errors: list[str]) -> None:
    parity = verify_self_hashed_json(config.RunPaths(Path(protocol.get("run_root", ""))).parity) if False else None
    del parity
    # The producer parity file is checked as an artifact; these direct checks make
    # the validator independent of its implementation for the most fragile anchors.
    h_scores = verify_self_hashed_json(config.ROOT_INPUTS["H/scores_manifest"][0])
    h_index = {(str(row["sample_id"]), str(row["cell"])): row for row in h_scores["scores"]}
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        for arm in config.H_VIDEO_ARMS:
            binding = next(row for row in protocol["matrix_bindings"] if row["origin"] == "H" and row["sample_id"] == sid and row["video_arm"] == arm)
            matrix = load_matrix(Path(str(binding["matrix"])), str(binding["matrix_sha256"]), "validator H parity matrix")
            actual = _summary(matrix, range(matrix.shape[0]), mean_dtype=np.dtype(np.float32))
            score = h_index[(sid, f"V_{arm}/A_N")]
            if abs(actual["sync_c"] - float(score["sync_c"])) > 0.000501 or abs(actual["sync_d"] - float(score["sync_d"])) > 0.000501 or actual["offset"] != int(score["av_offset"]):
                errors.append(f"independent H FULL parity differs: {sid}/{arm}")


def _core_validate(root: Path, *, require_final: bool) -> dict[str, Any]:
    errors: list[str] = []
    try:
        protocol = verify_self_hashed_json(root / "protocol.json")
        audit = verify_self_hashed_json(root / "input_audit.json")
        _root_and_binding_checks(root, protocol, audit, errors)
        endpoint_index = _endpoint_checks(root, protocol, errors)
        parity = verify_self_hashed_json(root / "parity.json")
        if parity.get("valid") is not True or parity.get("error_count") != 0:
            errors.append("parity artifact is not valid")
        _parity_parent_checks(protocol, endpoint_index, errors)
        _analysis_checks(root, protocol, endpoint_index, errors)
        review = verify_self_hashed_json(root / "review.json")
        if review.get("status") != "complete" or review.get("checks", {}).get("zero_new_media") is not True or review.get("checks", {}).get("zero_model_forward") is not True:
            errors.append("review artifact or zero-budget guard is invalid")
        if require_final:
            validation = verify_self_hashed_json(root / "validation.json")
            final = verify_self_hashed_json(root / "final.json")
            if validation.get("valid") is not True:
                errors.append("validation artifact is not valid")
            if final.get("status") != "complete" or final.get("engineering_decision") != "GO" or final.get("diagnostic_decision") != "RECONCILED" or final.get("scientific_decision") != "NOT_A_CONFIRMATION":
                errors.append("final terminal decision is invalid")
            if final.get("next_action") != "STOP_CURRENT_SPECTRAL_CONSTRUCTION":
                errors.append("final next_action is invalid")
            if final.get("training_authorized") is not False or final.get("generalization_established") is not False or final.get("historical_gate_repaired") is not False:
                errors.append("final authorization flags are invalid")
            for field, path in (("protocol_sha256", root / "protocol.json"), ("parity_sha256", root / "parity.json"), ("analysis_sha256", root / "analysis.json"), ("review_sha256", root / "review.json"), ("validation_sha256", root / "validation.json")):
                if final.get(field) != file_sha256(path):
                    errors.append(f"final binding differs: {field}")
            if not (root / "result.md").is_file():
                errors.append("result.md is missing")
        return {"schema_version": 1, "stage_id": "validation", "protocol_id": config.PROTOCOL_ID, "valid": not errors, "status": "valid" if not errors else "invalid", "error_count": len(errors), "errors": errors, "record_count": config.EXPECTED_RECORD_COUNT, "matrix_count": config.EXPECTED_MATRIX_COUNT, "endpoint_count": config.EXPECTED_ENDPOINT_COUNT, "new_media_count": 0, "model_forward_count": 0}
    except Exception as exc:  # noqa: BLE001 - validator must return a structured invalid result
        errors.append(f"{type(exc).__name__}: {exc}")
        return {"schema_version": 1, "stage_id": "validation", "protocol_id": config.PROTOCOL_ID, "valid": False, "status": "invalid", "error_count": len(errors), "errors": errors, "record_count": config.EXPECTED_RECORD_COUNT, "matrix_count": config.EXPECTED_MATRIX_COUNT, "endpoint_count": config.EXPECTED_ENDPOINT_COUNT, "new_media_count": 0, "model_forward_count": 0}


def validate_pre_final(root: Path) -> dict[str, Any]:
    return _core_validate(Path(root).resolve(), require_final=False)


def validate_run(root: Path) -> dict[str, Any]:
    return _core_validate(Path(root).resolve(), require_final=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate cache-only endpoint reconciliation")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
