from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .analysis import analyze_record, bootstrap_group_means, bootstrap_indices, decision_flags
from .common import ProtocolError, assert_finite, file_sha256, read_json, source_pcm16, verify_media_pair, verify_self_hashed_json, write_json_atomic, write_self_hashed_json


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


def _cells(parent_protocol: dict[str, Any], score_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = parent_protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("parent record count is not 16")
    record_map = {str(row["sample_id"]): row for row in records}
    if len(record_map) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("parent sample ids are not unique")
    selected: list[dict[str, Any]] = []
    for row in score_manifest.get("rows", []):
        sample_id = str(row.get("sample_id"))
        video_arm = str(row.get("video_arm"))
        audio_arm = str(row.get("audio_arm"))
        if video_arm != "N" or audio_arm not in {"N", "A_DELAY"}:
            continue
        score = row.get("score")
        if not isinstance(score, dict):
            raise ProtocolError(f"score sidecar missing: {sample_id}/{audio_arm}")
        worker_path = Path(str(score.get("worker", "")))
        worker = verify_self_hashed_json(worker_path)
        if worker.get("sample_id") != sample_id or worker.get("video_arm") != video_arm or worker.get("audio_arm") != audio_arm:
            raise ProtocolError(f"worker identity mismatch: {sample_id}/{audio_arm}")
        arrays: dict[str, np.ndarray] = {}
        for key in ("visual", "audio_embedding", "matrix"):
            path = Path(str(worker.get(key, "")))
            expected_hash = str(worker.get(f"{key}_sha256", ""))
            if not path.is_file() or file_sha256(path) != expected_hash:
                raise ProtocolError(f"cached {key} hash mismatch: {sample_id}/{audio_arm}")
            value = np.asarray(np.load(path, allow_pickle=False))
            expected_shape = (config.EMBEDDING_ROWS, config.EMBEDDING_DIM) if key != "matrix" else (config.EMBEDDING_ROWS, config.MATRIX_COLUMNS)
            if value.dtype != np.float32 or value.shape != expected_shape or not np.isfinite(value).all():
                raise ProtocolError(f"cached {key} contract mismatch: {sample_id}/{audio_arm}")
            arrays[key] = value
        if sample_id not in record_map:
            raise ProtocolError(f"score row outside parent records: {sample_id}")
        selected.append({
            "sample_id": sample_id,
            "source_group": str(record_map[sample_id]["source_group"]),
            "audio_arm": audio_arm,
            "score": score,
            "worker": worker,
            "visual": arrays["visual"],
            "audio_embedding": arrays["audio_embedding"],
            "matrix": arrays["matrix"],
        })
    expected = {(sample_id, arm) for sample_id in record_map for arm in ("N", "A_DELAY")}
    actual = {(str(row["sample_id"]), str(row["audio_arm"])) for row in selected}
    if actual != expected or len(selected) != config.EXPECTED_CELL_COUNT:
        raise ProtocolError(f"expected exactly 32 N/A_DELAY cells, got {len(selected)}")
    selected.sort(key=lambda row: (row["source_group"], row["sample_id"], row["audio_arm"]))
    return selected


def _media_and_input_audit(cells: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_sample: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for cell in cells:
        sample_id = str(cell["sample_id"])
        score = cell["score"]
        audio_path = Path(str(score["source_audio"]))
        media_path = Path(str(score["media"]))
        if file_sha256(media_path) != str(score["media_sha256"]):
            raise ProtocolError(f"media hash mismatch: {sample_id}/{cell['audio_arm']}")
        pcm = source_pcm16(audio_path)
        if file_sha256(audio_path) != str(score["source_audio_sha256"]):
            raise ProtocolError(f"source audio hash mismatch: {sample_id}/{cell['audio_arm']}")
        if __import__('hashlib').sha256(pcm).hexdigest() != str(score["source_pcm_sha256"]):
            raise ProtocolError(f"source PCM hash mismatch: {sample_id}/{cell['audio_arm']}")
        cell["source_pcm"] = pcm
        by_sample.setdefault(sample_id, {})[str(cell["audio_arm"])] = cell
        audit_rows.append({
            "sample_id": sample_id,
            "source_group": str(cell["source_group"]),
            "audio_arm": str(cell["audio_arm"]),
            "media": str(media_path.resolve()),
            "media_sha256": str(score["media_sha256"]),
            "source_audio": str(audio_path.resolve()),
            "source_audio_sha256": str(score["source_audio_sha256"]),
            "source_pcm_sha256": str(score["source_pcm_sha256"]),
            "visual_sha256": str(cell["worker"]["visual_sha256"]),
            "audio_embedding_sha256": str(cell["worker"]["audio_embedding_sha256"]),
            "matrix_sha256": str(cell["worker"]["matrix_sha256"]),
            "worker": str(score["worker"]),
            "worker_sha256": str(score["worker_sha256"]),
        })
    if set(by_sample) != {str(cell["sample_id"]) for cell in cells} or any(set(pair) != {"N", "A_DELAY"} for pair in by_sample.values()):
        raise ProtocolError("paired input cells are incomplete")
    for sample_id, pair in by_sample.items():
        n = pair["N"]
        d = pair["A_DELAY"]
        if not np.array_equal(n["visual"], d["visual"]):
            raise ProtocolError(f"visual embeddings differ inside pair: {sample_id}")
        n_pcm = n["source_pcm"]
        d_pcm = d["source_pcm"]
        delay_bytes = config.DELAY_SAMPLES * 2
        if len(n_pcm) != len(d_pcm) or d_pcm[:delay_bytes] != b"\0" * delay_bytes or d_pcm[delay_bytes:] != n_pcm[:-delay_bytes]:
            raise ProtocolError(f"A_DELAY PCM relation failed: {sample_id}")
        media_evidence = verify_media_pair(Path(str(n["score"]["media"])), Path(str(d["score"]["media"])), n_pcm, d_pcm)
        for arm in ("N", "A_DELAY"):
            if media_evidence[arm]["sha256"] != str(pair[arm]["score"]["media_sha256"]):
                raise ProtocolError(f"decoded media hash mismatch: {sample_id}/{arm}")
        for row in audit_rows:
            if row["sample_id"] == sample_id:
                row["media_evidence"] = media_evidence[row["audio_arm"]]
        n["media_evidence"] = media_evidence["N"]
        d["media_evidence"] = media_evidence["A_DELAY"]
    audit = {
        "schema_version": 1,
        "protocol_id": "wav2lip_delay_search_support",
        "status": "GO",
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_GROUP_COUNT,
        "cell_count": len(audit_rows),
        "visual_pair_equal": True,
        "delay_samples": config.DELAY_SAMPLES,
        "rows": audit_rows,
        "budget": config.config_payload()["budget"],
    }
    return audit, by_sample


def _protocol(cells: list[dict[str, Any]], audit: dict[str, Any], run_id: str) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for cell in cells:
        sample_id = str(cell["sample_id"])
        record = records.setdefault(sample_id, {"sample_id": sample_id, "source_group": str(cell["source_group"]), "cells": []})
        record["cells"].append({
            "audio_arm": str(cell["audio_arm"]),
            "worker": str(cell["score"]["worker"]),
            "worker_sha256": str(cell["score"]["worker_sha256"]),
            "visual_sha256": str(cell["worker"]["visual_sha256"]),
            "audio_embedding_sha256": str(cell["worker"]["audio_embedding_sha256"]),
            "matrix_sha256": str(cell["worker"]["matrix_sha256"]),
            "source_audio": str(cell["score"]["source_audio"]),
            "source_audio_sha256": str(cell["score"]["source_audio_sha256"]),
            "media": str(cell["score"]["media"]),
            "media_sha256": str(cell["score"]["media_sha256"]),
        })
    record_list = sorted(records.values(), key=lambda row: (row["source_group"], row["sample_id"]))
    for record in record_list:
        record["cells"].sort(key=lambda row: row["audio_arm"])
    spec_bindings = {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in (("proposal", config.PROPOSAL), ("design", config.DESIGN), ("spec", config.SPEC))}
    code_paths = [Path(__file__), Path(__file__).with_name("analysis.py"), Path(__file__).with_name("validate.py"), Path(__file__).with_name("common.py"), Path(__file__).with_name("config.py")]
    return {
        "schema_version": 1,
        "protocol_id": "wav2lip_delay_search_support",
        "protocol_revision": "matched_delay_domain_v1",
        "run_id": run_id,
        "status": "locked",
        "parent_run": str(config.PARENT.resolve()),
        "parent_file_hashes": dict(config.PARENT_HASHES),
        "spec_bindings": spec_bindings,
        "code_bindings": {str(path.name): {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in code_paths},
        "record_count": len(record_list),
        "source_group_count": len({str(row["source_group"]) for row in record_list}),
        "cell_count": len(cells),
        "records": record_list,
        "coordinates": {
            "u_rows": list(config.U_ROWS),
            "natural_lags": list(range(config.NATURAL_LAG_START, config.NATURAL_LAG_START + config.MATRIX_COLUMNS)),
            "delay_lags": list(range(config.DELAY_LAG_START, config.DELAY_LAG_START + config.MATRIX_COLUMNS)),
            "natural_offset": "15-j",
            "delay_offset": "10-j",
            "expected_delay_column": "k_N+5",
        },
        "input_audit_sha256": audit.get("artifact_sha256"),
        "configuration": config.config_payload(),
        "fresh_work": {"new_videos": 0, "new_forward": 0, "training": 0, "gpu_calls": 0},
        "parent_control_final_unchanged": True,
    }


def _write_result(paths: config.RunPaths, analysis: dict[str, Any], validation: dict[str, Any], final: dict[str, Any]) -> None:
    anchor = analysis["anchor_damage"]
    lines = [
        "# Wav2Lip delay search support audit 2026-09-09",
        "",
        f"- 终态：`{final['scientific_decision']}`；工程状态：`{final['engineering_status']}`。",
        f"- 旧门禁：{analysis['old_offset_pass_count']}/16；旧异常：{', '.join(analysis['old_failure_ids']) or '无'}。",
        f"- 配对搜索域：{analysis['matched_offset_pass_count']}/16；恢复异常：{', '.join(analysis['recovered_ids']) or '无'}。",
        f"- anchor 损伤：mean={anchor['mean']:.12f}，95% CI=[{anchor['ci95'][0]:.12f}, {anchor['ci95'][1]:.12f}]，正组={anchor['group_positive_count']}/8。",
        f"- corrected_control_pass={analysis['corrected_control_pass']}；boundary_explanation_complete={analysis['boundary_explanation_complete']}；content_probe_revision_eligible={analysis['content_probe_revision_eligible']}。",
        f"- 独立验收：`{validation['status']}`。",
        "- 本轮没有执行候选视频、TFG forward、训练、TTS 或 GPU；父 run 的 `CONTROL_FAILED` 保持不变。",
        "",
        "下一步：仅当终态为 `SEARCH_SUPPORT_RECOVERED` 时，另建版本化 amendment 修订已知 delay 控制坐标，再决定是否运行原 48 个候选；本 run 不自动授权 Stage B。",
    ]
    paths.result.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_id: str, resume: bool = False) -> int:
    paths = config.RunPaths(config.run_root_for(run_id))
    if paths.root.exists() and any(paths.root.iterdir()) and not resume:
        raise ProtocolError(f"refusing non-empty output directory: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)
    parent_protocol, score_manifest = _fixed_parent()
    cells = _cells(parent_protocol, score_manifest)
    audit, by_sample = _media_and_input_audit(cells)
    audit = write_self_hashed_json(paths.input_audit, audit)
    protocol = write_self_hashed_json(paths.protocol, _protocol(cells, audit, run_id))
    indices = bootstrap_indices()
    paths.bootstrap_indices.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.bootstrap_indices, indices, allow_pickle=False)
    matrix_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    for sample_id in sorted(by_sample, key=lambda sid: (by_sample[sid]["N"]["source_group"], sid)):
        n = by_sample[sample_id]["N"]
        d = by_sample[sample_id]["A_DELAY"]
        result, natural_matched, delay_matched = analyze_record(
            sample_id=sample_id,
            source_group=str(n["source_group"]),
            visual=n["visual"],
            natural_audio=n["audio_embedding"],
            delayed_audio=d["audio_embedding"],
            cached_n_matrix=n["matrix"],
            cached_delay_matrix=d["matrix"],
        )
        for arm, value in (("N", natural_matched), ("A_DELAY", delay_matched)):
            path = paths.root / "matrices" / f"{sample_id}__{arm}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(value, dtype=np.float32), allow_pickle=False)
            matrix_rows.append({"sample_id": sample_id, "audio_arm": arm, "path": str(path.resolve()), "sha256": file_sha256(path), "shape": list(value.shape), "dtype": "float32", "derived_from": "cached_embeddings", "fresh_forward": False})
        record_rows.append(result)
    matrix_manifest = write_self_hashed_json(paths.matrix_manifest, {"schema_version": 1, "protocol_id": "wav2lip_delay_search_support", "status": "complete", "count": len(matrix_rows), "rows": matrix_rows})
    groups = [str(row["source_group"]) for row in record_rows]
    anchor = bootstrap_group_means([float(row["anchor_damage"]) for row in record_rows], groups, indices)
    old_failure_ids = {str(row["sample_id"]) for row in record_rows if bool(row["legacy"]["old_failure"])}
    matched_pass_ids = {str(row["sample_id"]) for row in record_rows if bool(row["matched"]["offset_pass"])}
    recovered_ids = old_failure_ids & matched_pass_ids
    matched_pass_count = len(matched_pass_ids)
    flags = decision_flags(anchor_pass=bool(anchor["ci95"][0] > 0.0 and anchor["group_positive_count"] >= 7), matched_pass_count=matched_pass_count, old_failure_ids=old_failure_ids, expected_failure_ids=set(config.EXPECTED_ANOMALIES), recovered_ids=recovered_ids)
    old_pass_count = config.EXPECTED_RECORD_COUNT - len(old_failure_ids)
    analysis = {
        "schema_version": 1,
        "protocol_id": "wav2lip_delay_search_support",
        "stage": "CPU_DELAY_SEARCH_SUPPORT",
        "engineering_status": "GO",
        "record_count": len(record_rows),
        "cell_count": len(cells),
        "records": record_rows,
        "anchor_damage": anchor,
        "old_offset_pass_count": old_pass_count,
        "old_failure_ids": sorted(old_failure_ids),
        "expected_old_failure_ids": sorted(config.EXPECTED_ANOMALIES),
        "matched_offset_pass_count": matched_pass_count,
        "matched_pass_ids": sorted(matched_pass_ids),
        "recovered_ids": sorted(recovered_ids),
        "anchor_pass": bool(anchor["ci95"][0] > 0.0 and anchor["group_positive_count"] >= 7),
        **flags,
        "matrix_manifest_sha256": file_sha256(paths.matrix_manifest),
        "bootstrap_indices_sha256": file_sha256(paths.bootstrap_indices),
        "budget": config.config_payload()["budget"],
        "stage_b_authorized": False,
        "replacement_confirmed": False,
        "historical_shift_gate_repaired": False,
        "waveform_head_authorized": False,
        "generalization_established": False,
    }
    assert_finite(analysis)
    analysis = write_self_hashed_json(paths.analysis, analysis)
    record_payload = {"schema_version": 1, "protocol_id": "wav2lip_delay_search_support", "records": record_rows, "artifact_source": str(paths.analysis.resolve())}
    write_self_hashed_json(paths.per_record, record_payload)
    from .validate import validate_run

    validation = validate_run(paths.root)
    validation = write_self_hashed_json(paths.validation, validation)
    review = write_self_hashed_json(paths.review, {"schema_version": 1, "protocol_id": "wav2lip_delay_search_support", "self_review": True, "independent_validation_status": validation["status"], "checks": {"parent_immutable": validation["status"] == "PASS", "flags_bounded": all(analysis.get(key) is False for key in ("stage_b_authorized", "replacement_confirmed", "historical_shift_gate_repaired", "waveform_head_authorized", "generalization_established")), "no_new_forward": True, "no_gpu": True}, "notes": "review is producer self-review; numerical acceptance comes from validation.json"})
    final_status = "complete" if validation["status"] == "PASS" else "blocked"
    final = write_self_hashed_json(paths.final, {
        "schema_version": 1,
        "protocol_id": "wav2lip_delay_search_support",
        "status": final_status,
        "engineering_status": "GO" if validation["status"] == "PASS" else "BLOCKED",
        "scientific_decision": analysis["terminal_decision"] if validation["status"] == "PASS" else "BLOCKED",
        "validation_status": validation["status"],
        "validation_sha256": file_sha256(paths.validation),
        "analysis_sha256": file_sha256(paths.analysis),
        "review_sha256": file_sha256(paths.review),
        "content_probe_revision_eligible": bool(analysis["content_probe_revision_eligible"] and validation["status"] == "PASS"),
        "stage_b_authorized": False,
        "replacement_confirmed": False,
        "historical_shift_gate_repaired": False,
        "waveform_head_authorized": False,
        "generalization_established": False,
        "parent_control_final_unchanged": True,
    })
    _write_result(paths, analysis, validation, final)
    print(json.dumps({"run_root": str(paths.root), "final": final, "validation": validation}, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CPU-only Wav2Lip delay search-support diagnosis")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args.run_id, resume=args.resume)
    except Exception as exc:  # noqa: BLE001 - preserve blocked evidence
        root = config.run_root_for(args.run_id)
        root.mkdir(parents=True, exist_ok=True)
        error = write_self_hashed_json(root / "error.json", {"schema_version": 1, "protocol_id": "wav2lip_delay_search_support", "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
        final = write_self_hashed_json(root / "final.json", {"schema_version": 1, "protocol_id": "wav2lip_delay_search_support", "status": "blocked", "engineering_status": "BLOCKED", "scientific_decision": "BLOCKED", "error_sha256": file_sha256(root / "error.json"), "content_probe_revision_eligible": False, "stage_b_authorized": False, "replacement_confirmed": False, "historical_shift_gate_repaired": False, "waveform_head_authorized": False, "generalization_established": False})
        print(json.dumps({"run_root": str(root), "error": error, "final": final}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
