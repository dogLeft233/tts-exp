from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, assert_finite, file_sha256, read_json, source_pcm16, verify_media_pair, verify_self_hashed_json, write_self_hashed_json


def _legacy_distance_matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float32)
    audio_value = np.asarray(audio, dtype=np.float32)
    if visual_value.shape != (config.EMBEDDING_ROWS, config.EMBEDDING_DIM) or audio_value.shape != visual_value.shape:
        raise ProtocolError(f"embedding shape mismatch: {visual_value.shape} vs {audio_value.shape}")
    padded = np.pad(audio_value, ((config.VSHIFT, config.VSHIFT), (0, 0)), mode="constant")
    output = np.empty((config.EMBEDDING_ROWS, config.MATRIX_COLUMNS), dtype=np.float32)
    epsilon = np.float32(1e-6)
    for row in range(config.EMBEDDING_ROWS):
        difference = visual_value[row : row + 1] - padded[row : row + config.MATRIX_COLUMNS]
        output[row] = np.sqrt(np.sum(np.square(difference + epsilon, dtype=np.float32), axis=1, dtype=np.float32)).astype(np.float32)
    return output


def _matched_distance_matrix(visual: np.ndarray, audio: np.ndarray, lag_start: int) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float32)
    audio_value = np.asarray(audio, dtype=np.float32)
    if visual_value.shape != (config.EMBEDDING_ROWS, config.EMBEDDING_DIM) or audio_value.shape != visual_value.shape:
        raise ProtocolError(f"embedding shape mismatch: {visual_value.shape} vs {audio_value.shape}")
    lags = list(range(int(lag_start), int(lag_start) + config.MATRIX_COLUMNS))
    output = np.empty((len(config.U_ROWS), config.MATRIX_COLUMNS), dtype=np.float32)
    epsilon = np.float32(1e-6)
    for output_row, visual_row in enumerate(config.U_ROWS):
        indices = np.asarray([visual_row + lag for lag in lags], dtype=np.int64)
        if int(indices.min()) < 0 or int(indices.max()) >= audio_value.shape[0]:
            raise ProtocolError(f"matched lag support is unavailable at row {visual_row}")
        difference = visual_value[visual_row : visual_row + 1] - audio_value[indices]
        output[output_row] = np.sqrt(np.sum(np.square(difference + epsilon, dtype=np.float32), axis=1, dtype=np.float32)).astype(np.float32)
    return output


def _metrics_from_curve(values: np.ndarray, offset_base: int) -> dict[str, Any]:
    curve = np.asarray(values, dtype=np.float64)
    if curve.shape != (config.MATRIX_COLUMNS,) or not np.isfinite(curve).all():
        raise ProtocolError(f"invalid curve shape: {curve.shape}")
    index = int(np.argmin(curve))
    return {"curve": [float(item) for item in curve], "min_index": index, "offset": int(offset_base - index), "D": float(curve[index]), "C": float(np.median(curve) - curve[index])}


def _bootstrap_group_means(values: list[float], groups: list[str], indices: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for value, group in zip(values, groups, strict=True):
        grouped.setdefault(str(group), []).append(float(value))
    labels = sorted(grouped)
    if len(labels) != config.EXPECTED_GROUP_COUNT or any(len(grouped[label]) != 2 for label in labels):
        raise ProtocolError("bootstrap group membership changed")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    if indices.shape != (config.BOOTSTRAP_DRAWS, config.EXPECTED_GROUP_COUNT):
        raise ProtocolError("bootstrap shape changed")
    estimates = np.mean(means[indices], axis=1, dtype=np.float64)
    return {
        "mean": float(np.mean(means, dtype=np.float64)),
        "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))],
        "group_labels": labels,
        "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)},
        "group_positive_count": int(np.sum(means > 0.0)),
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
    }


def _decision_flags(anchor_pass: bool, matched_pass_count: int, old_failure_ids: set[str], recovered_ids: set[str]) -> dict[str, Any]:
    corrected = bool(anchor_pass and int(matched_pass_count) >= 14)
    complete = bool(old_failure_ids == set(config.EXPECTED_ANOMALIES) and recovered_ids == set(config.EXPECTED_ANOMALIES) and matched_pass_count == config.EXPECTED_RECORD_COUNT)
    recovered = bool(corrected and complete)
    return {"corrected_control_pass": corrected, "boundary_explanation_complete": complete, "terminal_decision": "SEARCH_SUPPORT_RECOVERED" if recovered else "CONTROL_UNRESOLVED", "content_probe_revision_eligible": recovered}


def _fixed_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    for relative, expected in config.PARENT_HASHES.items():
        path = config.PARENT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"parent artifact hash changed: {path}")
    protocol = verify_self_hashed_json(config.PARENT_PROTOCOL)
    scores = verify_self_hashed_json(config.PARENT_SCORE_MANIFEST)
    verify_self_hashed_json(config.PARENT_CONTROL_ANALYSIS)
    verify_self_hashed_json(config.PARENT_CONTROL_VALIDATION)
    verify_self_hashed_json(config.PARENT_FINAL)
    return protocol, scores


def _parent_cells(protocol: dict[str, Any], score_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("parent record count changed")
    record_map = {str(row["sample_id"]): row for row in records}
    if len(record_map) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("parent sample ids are not unique")
    rows = []
    for row in score_manifest.get("rows", []):
        if str(row.get("video_arm")) != "N" or str(row.get("audio_arm")) not in {"N", "A_DELAY"}:
            continue
        sample_id = str(row.get("sample_id"))
        if sample_id not in record_map:
            raise ProtocolError(f"score row is outside parent cohort: {sample_id}")
        score = row.get("score")
        if not isinstance(score, dict):
            raise ProtocolError(f"score row is malformed: {sample_id}")
        worker_path = Path(str(score.get("worker", "")))
        worker = verify_self_hashed_json(worker_path)
        if worker.get("sample_id") != sample_id or worker.get("video_arm") != "N" or worker.get("audio_arm") != row.get("audio_arm"):
            raise ProtocolError(f"worker identity mismatch: {sample_id}/{row.get('audio_arm')}")
        cells = {
            "sample_id": sample_id,
            "source_group": str(record_map[sample_id]["source_group"]),
            "audio_arm": str(row["audio_arm"]),
            "score": score,
            "worker": worker,
        }
        for key in ("visual", "audio_embedding", "matrix"):
            path = Path(str(worker.get(key, "")))
            expected = str(worker.get(f"{key}_sha256", ""))
            if not path.is_file() or not expected or file_sha256(path) != expected:
                raise ProtocolError(f"cached {key} binding failed: {sample_id}/{row['audio_arm']}")
            value = np.asarray(np.load(path, allow_pickle=False))
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ProtocolError(f"cached {key} dtype/finite contract failed: {sample_id}/{row['audio_arm']}")
            expected_shape = (config.EMBEDDING_ROWS, config.EMBEDDING_DIM) if key != "matrix" else (config.EMBEDDING_ROWS, config.MATRIX_COLUMNS)
            if value.shape != expected_shape:
                raise ProtocolError(f"cached {key} shape changed: {sample_id}/{row['audio_arm']}: {value.shape}")
            cells[key] = value
        rows.append(cells)
    expected_keys = {(sample_id, arm) for sample_id in record_map for arm in ("N", "A_DELAY")}
    actual_keys = {(str(row["sample_id"]), str(row["audio_arm"])) for row in rows}
    if actual_keys != expected_keys or len(rows) != config.EXPECTED_CELL_COUNT:
        raise ProtocolError(f"cached cell set changed: {len(rows)}")
    rows.sort(key=lambda row: (row["source_group"], row["sample_id"], row["audio_arm"]))
    return rows


def _check_protocol(run_root: Path, protocol: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if protocol.get("status") != "locked" or protocol.get("record_count") != config.EXPECTED_RECORD_COUNT or protocol.get("cell_count") != config.EXPECTED_CELL_COUNT:
        raise ProtocolError("diagnostic protocol counts/status changed")
    if protocol.get("parent_file_hashes") != config.PARENT_HASHES:
        raise ProtocolError("diagnostic parent hash lock changed")
    coordinates = protocol.get("coordinates")
    expected_coordinates = {
        "u_rows": list(config.U_ROWS),
        "natural_lags": list(range(config.NATURAL_LAG_START, config.NATURAL_LAG_START + config.MATRIX_COLUMNS)),
        "delay_lags": list(range(config.DELAY_LAG_START, config.DELAY_LAG_START + config.MATRIX_COLUMNS)),
        "natural_offset": "15-j",
        "delay_offset": "10-j",
        "expected_delay_column": "k_N+5",
    }
    if coordinates != expected_coordinates:
        raise ProtocolError("diagnostic coordinates changed")
    protocol_rows = protocol.get("records")
    if not isinstance(protocol_rows, list) or len(protocol_rows) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("diagnostic protocol records are incomplete")
    expected = {(str(row["sample_id"]), str(row["audio_arm"])) for row in rows}
    actual: set[tuple[str, str]] = set()
    for record in protocol_rows:
        sample_id = str(record.get("sample_id"))
        for cell in record.get("cells", []):
            actual.add((sample_id, str(cell.get("audio_arm"))))
    if actual != expected:
        raise ProtocolError("diagnostic protocol cell membership changed")
    if not run_root.is_dir():
        raise ProtocolError("run root is missing")


def _load_matrix_manifest(run_root: Path) -> dict[tuple[str, str], np.ndarray]:
    manifest = verify_self_hashed_json(run_root / "matrices/manifest.json")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != config.EXPECTED_CELL_COUNT:
        raise ProtocolError("derived matrix manifest count changed")
    result: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("audio_arm")))
        path = Path(str(row.get("path", "")))
        if key in result or not path.is_file() or file_sha256(path) != str(row.get("sha256")):
            raise ProtocolError(f"derived matrix binding changed: {key}")
        value = np.asarray(np.load(path, allow_pickle=False))
        if value.dtype != np.float32 or value.shape != (len(config.U_ROWS), config.MATRIX_COLUMNS) or not np.isfinite(value).all():
            raise ProtocolError(f"derived matrix shape/dtype changed: {key}")
        result[key] = value
    return result


def _verify_media(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), {})[str(row["audio_arm"])] = row
    for sample_id, pair in grouped.items():
        n = pair["N"]
        d = pair["A_DELAY"]
        n_score = n["score"]
        d_score = d["score"]
        n_audio = Path(str(n_score["source_audio"]))
        d_audio = Path(str(d_score["source_audio"]))
        n_pcm = source_pcm16(n_audio)
        d_pcm = source_pcm16(d_audio)
        if len(n_pcm) != len(d_pcm) or d_pcm[: config.DELAY_SAMPLES * 2] != b"\0" * (config.DELAY_SAMPLES * 2) or d_pcm[config.DELAY_SAMPLES * 2 :] != n_pcm[: -config.DELAY_SAMPLES * 2]:
            raise ProtocolError(f"delayed PCM identity failed: {sample_id}")
        evidence = verify_media_pair(Path(str(n_score["media"])), Path(str(d_score["media"])), n_pcm, d_pcm)
        if evidence["N"]["sha256"] != str(n_score["media_sha256"]) or evidence["A_DELAY"]["sha256"] != str(d_score["media_sha256"]):
            raise ProtocolError(f"media manifest hash mismatch: {sample_id}")


def validate_run(run_root: Path) -> dict[str, Any]:
    try:
        protocol, score_manifest = _fixed_parent()
        rows = _parent_cells(protocol, score_manifest)
        run_protocol = verify_self_hashed_json(run_root / "protocol.json")
        _check_protocol(run_root, run_protocol, rows)
        input_audit = verify_self_hashed_json(run_root / "input_audit.json")
        if input_audit.get("status") != "GO" or input_audit.get("cell_count") != config.EXPECTED_CELL_COUNT:
            raise ProtocolError("input audit is incomplete")
        _verify_media(rows)
        matrices = _load_matrix_manifest(run_root)
        by_cell = {(row["sample_id"], row["audio_arm"]): row for row in rows}
        per_record: list[dict[str, Any]] = []
        natural_values: list[float] = []
        groups: list[str] = []
        for sample_id in sorted({key[0] for key in by_cell}, key=lambda sid: (str(by_cell[(sid, "N")]["source_group"]), sid)):
            n = by_cell[(sample_id, "N")]
            d = by_cell[(sample_id, "A_DELAY")]
            if not np.array_equal(n["visual"], d["visual"]):
                raise ProtocolError(f"visual embedding differs between paired cells: {sample_id}")
            rebuilt_n = _legacy_distance_matrix(n["visual"], n["audio_embedding"])
            rebuilt_d = _legacy_distance_matrix(d["visual"], d["audio_embedding"])
            cached_n = n["matrix"]
            cached_d = d["matrix"]
            old_n = _metrics_from_curve(np.mean(rebuilt_n[np.asarray(config.U_ROWS)], axis=0, dtype=np.float64), offset_base=config.VSHIFT)
            old_d = _metrics_from_curve(np.mean(rebuilt_d[np.asarray(config.U_ROWS)], axis=0, dtype=np.float64), offset_base=config.VSHIFT)
            b = _matched_distance_matrix(n["visual"], n["audio_embedding"], config.NATURAL_LAG_START)
            t = _matched_distance_matrix(d["visual"], d["audio_embedding"], config.DELAY_LAG_START)
            if not np.allclose(b, matrices[(sample_id, "N")], atol=config.MATRIX_TOLERANCE, rtol=0.0) or not np.allclose(t, matrices[(sample_id, "A_DELAY")], atol=config.MATRIX_TOLERANCE, rtol=0.0):
                raise ProtocolError(f"derived matrix differs from independent reconstruction: {sample_id}")
            z_b = np.mean(b.astype(np.float64), axis=0, dtype=np.float64)
            z_t = np.mean(t.astype(np.float64), axis=0, dtype=np.float64)
            m_b = _metrics_from_curve(z_b, offset_base=config.VSHIFT)
            m_t = _metrics_from_curve(z_t, offset_base=10)
            old_difference = int(old_d["offset"] - old_n["offset"])
            matched_difference = int(m_t["offset"] - m_b["offset"])
            row = {
                "sample_id": sample_id,
                "source_group": n["source_group"],
                "old_k_N": int(old_n["min_index"]),
                "expected_column": int(old_n["min_index"] + config.DELAY_FRAMES),
                "expected_in_legacy_domain": bool(0 <= old_n["min_index"] + config.DELAY_FRAMES < config.MATRIX_COLUMNS),
                "old_offset_difference": old_difference,
                "matched_offset_difference": matched_difference,
                "matched_offset_pass": bool(config.OFFSET_LOW <= matched_difference <= config.OFFSET_HIGH),
                "old_matrix_max_abs": float(max(np.max(np.abs(rebuilt_n.astype(np.float64) - cached_n.astype(np.float64))), np.max(np.abs(rebuilt_d.astype(np.float64) - cached_d.astype(np.float64))))),
                "natural_curve": m_b,
                "delay_curve": m_t,
                "anchor_damage": float(np.mean(rebuilt_d[np.asarray(config.U_ROWS)], axis=0, dtype=np.float64)[old_n["min_index"]] - np.mean(rebuilt_n[np.asarray(config.U_ROWS)], axis=0, dtype=np.float64)[old_n["min_index"]]),
            }
            per_record.append(row)
            natural_values.append(float(row["anchor_damage"]))
            groups.append(str(row["source_group"]))
        indices = np.load(run_root / "bootstrap_indices.npy", allow_pickle=False)
        expected_indices = np.random.default_rng(config.BOOTSTRAP_SEED).integers(0, config.EXPECTED_GROUP_COUNT, size=(config.BOOTSTRAP_DRAWS, config.EXPECTED_GROUP_COUNT), endpoint=False)
        if indices.dtype != expected_indices.dtype or indices.shape != expected_indices.shape or not np.array_equal(indices, expected_indices):
            raise ProtocolError("bootstrap sampling indices changed")
        anchor = _bootstrap_group_means(natural_values, groups, indices)
        matched_pass_count = sum(bool(row["matched_offset_pass"]) for row in per_record)
        old_failure_ids = {str(row["sample_id"]) for row in per_record if not (config.OFFSET_LOW <= int(row["old_offset_difference"]) <= config.OFFSET_HIGH)}
        matched_pass_ids = {str(row["sample_id"]) for row in per_record if bool(row["matched_offset_pass"])}
        recovered_ids = old_failure_ids & matched_pass_ids
        flags = _decision_flags(anchor_pass=bool(anchor["ci95"][0] > 0.0 and anchor["group_positive_count"] >= 7), matched_pass_count=matched_pass_count, old_failure_ids=old_failure_ids, recovered_ids=recovered_ids)
        analysis = verify_self_hashed_json(run_root / "analysis.json")
        analysis_records = analysis.get("records")
        if not isinstance(analysis_records, list) or len(analysis_records) != config.EXPECTED_RECORD_COUNT:
            raise ProtocolError("analysis record count changed")
        analysis_by_id = {str(row.get("sample_id")): row for row in analysis_records if isinstance(row, dict)}
        if len(analysis_by_id) != config.EXPECTED_RECORD_COUNT:
            raise ProtocolError("analysis sample membership changed")
        for row in per_record:
            produced = analysis_by_id.get(str(row["sample_id"]))
            if produced is None:
                raise ProtocolError(f"analysis record is missing: {row['sample_id']}")
            produced_legacy = produced.get("legacy", {})
            produced_matched = produced.get("matched", {})
            if (
                str(produced.get("source_group")) != str(row["source_group"])
                or int(produced_legacy.get("natural", {}).get("min_index", -1)) != int(row["old_k_N"])
                or int(produced_legacy.get("expected_column", -1)) != int(row["expected_column"])
                or int(produced_legacy.get("offset_difference", 10**9)) != int(row["old_offset_difference"])
                or int(produced_matched.get("natural", {}).get("min_index", -1)) != int(row["natural_curve"]["min_index"])
                or int(produced_matched.get("delay", {}).get("min_index", -1)) != int(row["delay_curve"]["min_index"])
                or int(produced_matched.get("offset_difference", 10**9)) != int(row["matched_offset_difference"])
                or bool(produced_matched.get("offset_pass")) != bool(row["matched_offset_pass"])
            ):
                raise ProtocolError(f"analysis record values changed: {row['sample_id']}")
        for key in ("corrected_control_pass", "boundary_explanation_complete", "terminal_decision", "content_probe_revision_eligible"):
            if analysis.get(key) != flags[key]:
                raise ProtocolError(f"analysis terminal flag mismatch: {key}")
        if analysis.get("record_count") != config.EXPECTED_RECORD_COUNT or analysis.get("matched_offset_pass_count") != matched_pass_count:
            raise ProtocolError("analysis aggregate count mismatch")
        if set(analysis.get("matched_pass_ids", [])) != matched_pass_ids or set(analysis.get("recovered_ids", [])) != recovered_ids:
            raise ProtocolError("analysis pass/recovery membership mismatch")
        assert_finite({"anchor": anchor, "records": per_record})
        return {
            "schema_version": 1,
            "protocol_id": "wav2lip_delay_search_support",
            "status": "PASS",
            "independent": True,
            "record_count": len(per_record),
            "cell_count": len(rows),
            "matched_offset_pass_count": matched_pass_count,
            "matched_pass_ids": sorted(matched_pass_ids),
            "old_failure_ids": sorted(old_failure_ids),
            "recovered_ids": sorted(recovered_ids),
            "anchor": anchor,
            "terminal": flags,
            "budget": config.config_payload()["budget"],
            "replacement_confirmed": False,
            "historical_shift_gate_repaired": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "stage_b_authorized": False,
        }
    except Exception as exc:  # noqa: BLE001 - validator must leave a truthful artifact
        return {
            "schema_version": 1,
            "protocol_id": "wav2lip_delay_search_support",
            "status": "FAIL",
            "independent": True,
            "error_count": 1,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "replacement_confirmed": False,
            "historical_shift_gate_repaired": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "stage_b_authorized": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent CPU validator for the Wav2Lip delay search-support diagnosis")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.run_root.resolve())
    write_self_hashed_json(args.run_root.resolve() / "validation.json", result)
    print(result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
