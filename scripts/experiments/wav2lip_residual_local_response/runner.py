from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, file_sha256, group_bootstrap, load_self_hashed, read_json, write_json
from .support import exposure_for, row_supports, chunk_columns


def _paths(run_id: str) -> config.RunPaths:
    root = config.run_root_for(run_id); root.mkdir(parents=True, exist_ok=True); return config.RunPaths(root)


def _fixed() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name, path in {"drivers": config.DRIVERS, "parent_scores": config.PARENT_SCORES, "q_scores": config.Q_SCORES, "d_drivers": config.D_DRIVERS}.items():
        if not path.is_file() or file_sha256(path) != config.FIXED_HASHES[name]: raise ProtocolError(f"fixed asset changed or missing: {path}")
    if not config.Q_ANALYSIS.is_file() or file_sha256(config.Q_ANALYSIS) != config.FIXED_HASHES["q_analysis"]: raise ProtocolError("Q analysis binding changed")
    return load_self_hashed(config.DRIVERS), load_self_hashed(config.PARENT_SCORES), load_self_hashed(config.Q_SCORES), read_json(config.D_DRIVERS)


def _parent_score(manifest: dict[str, Any], sid: str, video: str, audio: str) -> dict[str, Any]:
    for row in manifest.get("rows", []):
        if str(row.get("sample_id")) == sid and str(row.get("video_arm")) == video and str(row.get("audio_arm")) == audio: return row.get("score", row)
    raise ProtocolError(f"score missing: {sid}/{video}/{audio}")


def _q_score(manifest: dict[str, Any], sid: str, arm: str) -> dict[str, Any]:
    for row in manifest.get("rows", []):
        if str(row.get("sample_id")) == sid and str(row.get("video_arm")) == arm and str(row.get("audio_arm")) == "N": return row["score"]
    raise ProtocolError(f"candidate score missing: {sid}/{arm}")


def _array_from_worker(score: dict[str, Any], key: str) -> np.ndarray:
    worker_path = Path(str(score["worker"])); worker = load_self_hashed(worker_path); path = Path(str(worker[key]))
    expected_hash = str(worker[f"{key}_sha256"])
    if file_sha256(path) != expected_hash: raise ProtocolError(f"worker array changed: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    expected = (config.MATRIX_ROWS, config.EMBEDDING_DIM) if key in {"visual", "audio_embedding"} else (config.MATRIX_ROWS, config.MATRIX_COLUMNS)
    if value.shape != expected or not np.isfinite(value).all(): raise ProtocolError(f"malformed worker array {key}: {path} {value.shape}")
    return value


def _rebuild_matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(audio, dtype=np.float32), ((config.VSHIFT, config.VSHIFT), (0, 0)), mode="constant")
    result = np.empty((config.MATRIX_ROWS, config.MATRIX_COLUMNS), dtype=np.float32)
    for row in range(config.MATRIX_ROWS):
        diff = np.asarray(visual[row : row + 1], dtype=np.float32) - padded[row : row + config.MATRIX_COLUMNS]
        result[row] = np.sqrt(np.sum(np.square(diff + np.float32(1e-6), dtype=np.float32), axis=1, dtype=np.float32), dtype=np.float32)
    return result


def _matrix(score: dict[str, Any]) -> tuple[np.ndarray, float]:
    rebuilt = _rebuild_matrix(_array_from_worker(score, "visual"), _array_from_worker(score, "audio_embedding"))
    cached_path = Path(str(score["matrix"])); cached = np.asarray(np.load(cached_path, allow_pickle=False), dtype=np.float32)
    if cached.shape != rebuilt.shape or file_sha256(cached_path) != str(score["matrix_sha256"]): raise ProtocolError(f"cached matrix binding changed: {cached_path}")
    error = float(np.max(np.abs(rebuilt.astype(np.float64) - cached.astype(np.float64))))
    if error > 1e-4: raise ProtocolError(f"independent matrix reconstruction mismatch: {cached_path}: {error}")
    return rebuilt, error


def _keys_from_d(drivers_d: dict[str, Any], sid: str) -> set[int]:
    rows = [row for row in drivers_d.get("drivers", []) if str(row.get("sample_id")) == sid]
    if len(rows) != 12: raise ProtocolError(f"expected 12 D driver rows for {sid}, got {len(rows)}")
    sets = []
    for row in rows:
        values: set[int] = set()
        for mask in row.get("used_masks", []): values.update(range(int(mask["global_start_frame"]), int(mask["global_end_frame"])))
        sets.append(values)
    if any(value != sets[0] for value in sets[1:]): raise ProtocolError(f"D used_masks disagree across conditions/seeds: {sid}")
    return sets[0]


def _prepare(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    drivers, parent_scores, q_scores, d_drivers = _fixed()
    rows = sorted(drivers.get("rows", []), key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    if len(rows) != 16 or len({str(row["source_group"]) for row in rows}) != 8: raise ProtocolError("frozen cohort is not 16/8")
    if paths.protocol.exists():
        if not resume: raise ProtocolError(f"run exists; use --resume: {paths.root}")
        return load_self_hashed(paths.protocol)
    exposure_rows = []; audit_rows = []; support = row_supports()
    for row in rows:
        sid = str(row["sample_id"]); p_k = set(int(value) for value in row["construct"]["K"]); d_k = _keys_from_d(d_drivers, sid)
        if p_k != d_k: raise ProtocolError(f"K mismatch between P and D: {sid}")
        exp = exposure_for(sorted(d_k)); exposure_rows.append({"sample_id": sid, "source_group": str(row["source_group"]), **exp}); audit_rows.append({"sample_id": sid, "source_group": str(row["source_group"]), "k_count": len(d_k), "k_sha256": hashlib.sha256(np.asarray(sorted(d_k), dtype=np.int64).tobytes()).hexdigest(), "parent_driver_sha256": row["construct"].get("K_sha256"), "d_driver_source": str(config.D_DRIVERS.resolve())})
    chunk_starts = np.asarray([int(i * 80.0 / config.FPS) for i in range(config.FRAME_COUNT)], dtype=np.int64)
    np.savez_compressed(paths.support, u_rows=np.asarray(config.U_ROWS, dtype=np.int64), chunk_starts=chunk_starts, **{f"row_{row}": support[row] for row in config.U_ROWS})
    protocol = {"schema_version": 1, "protocol_id": "wav2lip_residual_local_response", "protocol_revision": "fixed_support_local_response_v1", "status": "locked", "configuration": config.configuration(), "records": [{"sample_id": row["sample_id"], "source_group": row["source_group"]} for row in rows], "record_count": 16, "source_group_count": 8, "limits": {"fresh_videos": 0, "fresh_scores": 0, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}}
    write_json(paths.input_audit, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "complete", "record_count": 16, "source_group_count": 8, "rows": audit_rows})
    write_json(paths.exposure, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "complete", "rows": exposure_rows, "u_rows": list(config.U_ROWS), "support_npz": str(paths.support.resolve()), "support_npz_sha256": file_sha256(paths.support)})
    write_json(paths.reused_manifest, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "reused_only", "fresh_video_count": 0, "fresh_score_count": 0, "parent_assets": {name: {"path": str(path.resolve()), "sha256": config.FIXED_HASHES[name]} for name, path in {"drivers": config.DRIVERS, "parent_scores": config.PARENT_SCORES, "q_scores": config.Q_SCORES, "d_drivers": config.D_DRIVERS}.items()}})
    return write_json(paths.protocol, protocol)


def _metric_curve(matrix: np.ndarray, rows: tuple[int, ...]) -> np.ndarray:
    return np.mean(np.asarray(matrix)[np.asarray(rows, dtype=np.int64)], axis=0, dtype=np.float64)


def _analysis(paths: config.RunPaths, protocol: dict[str, Any], drivers: dict[str, Any], parent_scores: dict[str, Any], q_scores: dict[str, Any], d_drivers: dict[str, Any], resume: bool) -> dict[str, Any]:
    if paths.analysis.exists() and resume: return load_self_hashed(paths.analysis)
    exposure = load_self_hashed(paths.exposure); exposure_by_id = {str(row["sample_id"]): row for row in exposure["rows"]}; driver_by_id = {str(row["sample_id"]): row for row in drivers["rows"]}
    values: dict[str, list[dict[str, float]]] = {"CORRECT": [], "WRONG": [], "SHUFFLE": []}; group_records: dict[str, dict[str, list[float]]] = {arm: {} for arm in values}; records = []; matrix_errors = []; zero_rows = []
    for rec in sorted(protocol["records"], key=lambda row: (str(row["source_group"]), str(row["sample_id"]))):
        sid = str(rec["sample_id"]); group = str(rec["source_group"]); n_score = _parent_score(parent_scores, sid, "N", "N"); n_matrix, n_error = _matrix(n_score); n_curve = _metric_curve(n_matrix, config.U_ROWS); k0 = int(np.argmin(n_curve)); exp_rows = {int(row["row"]): row for row in exposure_by_id[sid]["rows"]}; row_out = {"sample_id": sid, "source_group": group, "k0": k0, "exposure_min": float(min(row["exposure"] for row in exp_rows.values())), "exposure_max": float(max(row["exposure"] for row in exp_rows.values())), "matrix_max_abs": {"N": n_error}}
        matrix_errors.append(n_error)
        for arm in ("CORRECT", "WRONG", "SHUFFLE"):
            score = _q_score(q_scores, sid, arm); matrix, error = _matrix(score); matrix_errors.append(error); row_out.setdefault("matrix_max_abs", {})[arm] = error; e = np.asarray([float(exp_rows[row]["exposure"]) for row in config.U_ROWS], dtype=np.float64); d = np.asarray([float(n_matrix[row, k0] - matrix[row, k0]) for row in config.U_ROWS], dtype=np.float64); centered_e = e - e.mean(); centered_d = d - d.mean(); v = float(np.mean(centered_e * centered_e)); c = float(np.mean(centered_e * centered_d)); local_num = float(np.sum(e * d)); local_den = float(np.sum(e)); entry = {"sample_id": sid, "source_group": group, "v": v, "c": c, "exposure_sum": local_den, "local_numerator": local_num, "d_mean": float(d.mean())}; values[arm].append(entry); group_records[arm].setdefault(group, []).append(entry); row_out[arm] = {"d_mean": float(d.mean()), "v": v, "c": c, "exposure_sum": local_den, "local_numerator": local_num}
            if arm == "CORRECT": row_out["parent_delta_A_component"] = float(d.mean())
            if np.all(e == 0.0) and arm == "CORRECT": zero_rows.append(sid)
            if np.all(e == 0.0) and float(np.max(np.abs(n_matrix[np.asarray(config.U_ROWS)] - matrix[np.asarray(config.U_ROWS)]))) > 1e-4: raise ProtocolError(f"zero-exposure row changed for {sid}/{arm}")
        records.append(row_out)
    group_stats: dict[str, Any] = {}; identifiable = True
    for arm, grouped in group_records.items():
        beta_values = []; local_values = []; labels = []
        for group in sorted(grouped):
            denom_v = sum(item["v"] for item in grouped[group]); denom_e = sum(item["exposure_sum"] for item in grouped[group]);
            if denom_v <= 1e-12 or denom_e <= 1e-12: identifiable = False
            beta_values.append(float(sum(item["c"] for item in grouped[group]) / denom_v) if denom_v > 1e-12 else 0.0); local_values.append(float(sum(item["local_numerator"] for item in grouped[group]) / denom_e) if denom_e > 1e-12 else 0.0); labels.append(group)
        beta = group_bootstrap(beta_values, labels); local = group_bootstrap(local_values, labels); group_stats[arm] = {"beta": beta, "local_gain": local, "group_values": {label: {"beta": value, "local_gain": gain} for label, value, gain in zip(labels, beta_values, local_values, strict=True)}}
    correct = group_stats["CORRECT"]
    pass_gate = bool(identifiable and correct["beta"]["ci99"][0] > 0 and correct["local_gain"]["ci99"][0] > 0 and correct["beta"]["group_positive_count"] >= 7 and correct["local_gain"]["group_positive_count"] >= 7)
    decision = "LOCAL_RESPONSE_ASSOCIATION_TO_CONFIRM" if pass_gate else "NO_LOCAL_POSITIVE_ASSOCIATION_ESTABLISHED" if identifiable else "LOCAL_SUPPORT_NOT_IDENTIFIABLE"
    parent_delta = float(np.mean([row["parent_delta_A_component"] for row in records]))
    q_analysis = load_self_hashed(config.Q_ANALYSIS); target = float(q_analysis["contrasts"]["CORRECT_vs_N"]["metrics"]["A"]["mean"])
    if abs(parent_delta - target) > 1e-6: raise ProtocolError(f"parent full-U delta A reproduction mismatch: {parent_delta} vs {target}")
    return write_json(paths.analysis, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "engineering_status": "GO", "scientific_decision": decision, "identifiable": identifiable, "zero_exposure_records": sorted(set(zero_rows)), "matrix_max_abs": float(max(matrix_errors)), "parent_delta_A": parent_delta, "parent_delta_A_target": target, "parent_delta_A_abs_error_to_reported": abs(parent_delta - target), "group_stats": group_stats, "records": records, "diagnostic_controls": {"WRONG": group_stats["WRONG"], "SHUFFLE": group_stats["SHUFFLE"]}, "fresh_video_count": 0, "fresh_score_count": 0, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False})


def _finalize(paths: config.RunPaths) -> None:
    analysis = load_self_hashed(paths.analysis); write_json(paths.validation, {"schema_version": 1, "protocol_id": analysis["protocol_id"], "status": "PENDING", "independent": True, "gpu_used": False, "fresh_scores": 0}); review = write_json(paths.review, {"schema_version": 1, "status": "self_reviewed", "checks": ["official chunk union", "K reconstructed from D used_masks", "matrix rebuilt from embeddings", "zero-exposure records retained", "no candidate generation"], "decision": analysis.get("scientific_decision")}); final = write_json(paths.final, {"schema_version": 1, "protocol_id": analysis["protocol_id"], "status": "complete", "engineering_status": "GO", "scientific_decision": analysis.get("scientific_decision"), "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False, "review_sha256": review["artifact_sha256"]}); paths.result.write_text(f"# Residual local response audit\\n\\n- decision: {final['scientific_decision']}\\n- fresh videos: 0\\n- fresh scores: 0\\n", encoding="utf-8")


def run(options: argparse.Namespace) -> int:
    paths = _paths(options.run_id)
    try:
        protocol = _prepare(paths, options.resume); drivers, parent, q, d = _fixed(); local_drivers = load_self_hashed(paths.p("drivers.json")) if paths.p("drivers.json").is_file() else load_self_hashed(config.DRIVERS)
        if options.stage in ("analyze", "all"): _analysis(paths, protocol, local_drivers, parent, q, d, options.resume); _finalize(paths)
        return 0
    except Exception as exc:
        write_json(paths.p("error.json"), {"schema_version": 1, "protocol_id": "wav2lip_residual_local_response", "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}); return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--stage", choices=("prepare", "analyze", "all"), default="all"); parser.add_argument("--resume", action="store_true"); return run(parser.parse_args(argv))


if __name__ == "__main__": raise SystemExit(main())
