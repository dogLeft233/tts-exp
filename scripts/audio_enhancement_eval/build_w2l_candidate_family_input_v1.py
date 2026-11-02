#!/usr/bin/env python3
"""Build a sealed W2L-only merged input for the candidate-family ASR guard.

This is a read-only source adapter: it never renders audio/video and never calls
ASR. Inputs are completed, hash-pinned Stage-A runs. It selects exactly one
model/frontend slice, joins methods across source runs by immutable sample,
speaker, seed, face, natural PCM, source video, and source clock, then emits the
cross-tfg-audio-eval-v1 ledger consumed by asr_content_guard_candidate_family_v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cross-tfg-audio-eval-v1"
SPEC_SCHEMA = "cross-tfg-asr-family-merge-spec-v1"
REGISTRY_SCHEMA = "cross-tfg-audio-head-content-candidate-registry-v1"
NATURAL_METHOD_ID = "identity"


class MergeError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical_json({key: item for key, item in value.items() if key != field}))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MergeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MergeError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json(dict(value)) + b"\n")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _context(row: Mapping[str, Any]) -> tuple[Any, ...]:
    natural = row.get("natural_pcm")
    score = row.get("score_audio")
    face = row.get("source_face")
    video = row.get("source_video")
    if not all(isinstance(value, dict) for value in (natural, score, face, video)):
        raise MergeError(f"{row.get('row_id')}: natural/score/face/video binding is incomplete")
    natural_key = tuple(natural.get(key) for key in (
        "file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype"
    ))
    score_key = tuple(score.get(key) for key in (
        "file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype"
    ))
    face_key = face.get("file_sha256")
    video_key = tuple(video.get(key) for key in ("file_sha256", "fps_num", "fps_den"))
    if not all(isinstance(value, str) and value for value in natural_key[:2]):
        raise MergeError(f"{row.get('row_id')}: natural PCM hashes are missing")
    if natural_key != score_key:
        raise MergeError(f"{row.get('row_id')}: score audio is not the exact natural PCM")
    if not isinstance(face_key, str) or not isinstance(video_key[0], str):
        raise MergeError(f"{row.get('row_id')}: face/video hashes are missing")
    return (
        row.get("split"),
        row.get("sample_id"),
        row.get("speaker_id"),
        row.get("model_id"),
        row.get("audio_frontend"),
        row.get("face_id"),
        row.get("seed"),
        natural_key,
        face_key,
        video_key,
    )


def validate_source(source: Mapping[str, Any], spec_path: Path, target: Mapping[str, Any]) -> dict[str, Any]:
    source_id = source.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise MergeError("source_id is missing")
    loaded: dict[str, Any] = {"source_id": source_id, "source_spec": dict(source)}
    paths: dict[str, Path] = {}
    raw_shas: dict[str, str] = {}
    for key, sha_key in (
        ("plan_path", "plan_raw_sha256"),
        ("results_path", "results_raw_sha256"),
        ("result_manifest_path", "result_manifest_raw_sha256"),
    ):
        raw_path = source.get(key)
        expected = source.get(sha_key)
        if not isinstance(raw_path, str) or not isinstance(expected, str) or len(expected) != 64:
            raise MergeError(f"{source_id}: {key}/{sha_key} is missing")
        p = Path(raw_path)
        p = (p if p.is_absolute() else spec_path.parent / p).resolve(strict=True)
        actual = sha_file(p)
        if actual != expected:
            raise MergeError(f"{source_id}: {key} raw SHA mismatch")
        paths[key] = p
        raw_shas[key] = actual
    plan = load_json(paths["plan_path"])
    results = load_jsonl(paths["results_path"])
    manifest = load_json(paths["result_manifest_path"])
    if plan.get("schema_version") not in ("cross_tfg-audio-stagea-v1", "cross-tfg-audio-eval-v1"):
        raise MergeError(f"{source_id}: unsupported plan schema")
    if plan.get("status") != "FROZEN" or plan.get("confirm_speaker_ids") or plan.get("generation_seed_family", {}).get("confirm"):
        raise MergeError(f"{source_id}: source must be frozen development-only with confirmation closed")
    plan_raw = raw_shas["plan_path"]
    results_raw = raw_shas["results_path"]
    manifest_raw = raw_shas["result_manifest_path"]
    embedded_plan_sha = plan.get("plan_sha256") or plan.get("plan_payload_sha256")
    if isinstance(plan.get("plan_sha256"), str) and plan["plan_sha256"] != payload_sha(plan, "plan_sha256"):
        raise MergeError(f"{source_id}: embedded plan payload SHA mismatch")
    if manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256"):
        raise MergeError(f"{source_id}: result manifest payload SHA mismatch")
    if (
        manifest.get("run_id") != plan.get("run_id")
        or manifest.get("plan_raw_sha256") != plan_raw
        or manifest.get("results_jsonl_sha256") != results_raw
        or manifest.get("planned_rows") != len(plan.get("rows", []))
        or manifest.get("result_rows") != len(results)
    ):
        raise MergeError(f"{source_id}: result manifest does not bind exact source rows")
    planned = plan.get("rows")
    if not isinstance(planned, list) or len(planned) != len(results):
        raise MergeError(f"{source_id}: planned/results row count mismatch")
    planned_ids = [row.get("row_id") for row in planned]
    result_ids = [row.get("row_id") for row in results]
    if len(set(planned_ids)) != len(planned_ids) or len(set(result_ids)) != len(result_ids) or planned_ids != result_ids:
        raise MergeError(f"{source_id}: result rows differ from planned row identities/order")
    methods = plan.get("methods")
    if not isinstance(methods, list):
        raise MergeError(f"{source_id}: method registry is missing")
    method_map = {item.get("method_id"): item for item in methods if isinstance(item, dict)}
    if len(method_map) != len(methods):
        raise MergeError(f"{source_id}: duplicate or malformed method registry")
    requested_methods = source.get("include_methods")
    if not isinstance(requested_methods, list) or not requested_methods or len(set(requested_methods)) != len(requested_methods):
        raise MergeError(f"{source_id}: include_methods must be a non-empty unique list")
    if any(method_id not in method_map for method_id in requested_methods):
        raise MergeError(f"{source_id}: requested method is absent from its sealed plan")
    target_model = target.get("model_id")
    target_frontend = target.get("audio_frontend")
    selected_plan: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    selected_results: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    result_by_id = {row.get("row_id"): row for row in results}
    for method_id in requested_methods:
        group_plan: dict[tuple[str, int], dict[str, Any]] = {}
        group_result: dict[tuple[str, int], dict[str, Any]] = {}
        for p_row in planned:
            if p_row.get("method_id") != method_id:
                continue
            if p_row.get("model_id") != target_model or p_row.get("audio_frontend") != target_frontend:
                continue
            if p_row.get("split") != "dev":
                continue
            key = (str(p_row.get("sample_id")), int(p_row.get("seed")))
            if key in group_plan:
                raise MergeError(f"{source_id}/{method_id}: duplicate sample/seed context {key}")
            r_row = result_by_id.get(p_row.get("row_id"))
            if not isinstance(r_row, dict):
                raise MergeError(f"{source_id}/{method_id}: missing result row")
            if (r_row.get("model_id"), r_row.get("audio_frontend"), r_row.get("method_id"),
                r_row.get("sample_id"), r_row.get("speaker_id"), r_row.get("seed")) != (
                target_model, target_frontend, method_id, p_row.get("sample_id"),
                p_row.get("speaker_id"), p_row.get("seed")
            ):
                raise MergeError(f"{source_id}/{method_id}: plan/result identity mismatch")
            group_plan[key] = p_row
            group_result[key] = r_row
        expected = int(target.get("expected_samples", 20)) * len(target.get("seed_family", []))
        if len(group_plan) != expected:
            raise MergeError(f"{source_id}/{method_id}: selected contexts {len(group_plan)} != expected {expected}")
        selected_plan[method_id] = group_plan
        selected_results[method_id] = group_result
    loaded.update({
        "plan": plan, "results": results, "manifest": manifest,
        "paths": {key: str(value) for key, value in paths.items()},
        "raw_shas": raw_shas, "method_map": method_map,
        "selected_plan": selected_plan, "selected_results": selected_results,
    })
    return loaded


def build(spec_path: Path, out_dir: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve(strict=True)
    spec = load_json(spec_path)
    if spec.get("schema_version") != SPEC_SCHEMA or spec.get("status") != "FROZEN":
        raise MergeError("merge spec must be frozen using the supported schema")
    target = spec.get("target")
    methods_spec = spec.get("methods")
    sources_spec = spec.get("sources")
    if not isinstance(target, dict) or not isinstance(methods_spec, list) or not isinstance(sources_spec, list):
        raise MergeError("target/methods/sources is malformed")
    if target.get("split") != "dev" or target.get("confirmation_closed") is not True:
        raise MergeError("only development scope with confirmation closed is accepted")
    seeds = target.get("seed_family")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise MergeError("target must bind exactly three distinct development seeds")
    source_by_id = {}
    for source in sources_spec:
        loaded = validate_source(source, spec_path, target)
        if loaded["source_id"] in source_by_id:
            raise MergeError(f"duplicate source_id: {loaded['source_id']}")
        source_by_id[loaded["source_id"]] = loaded
    if not source_by_id:
        raise MergeError("no source runs were supplied")
    method_ids = [entry.get("method_id") for entry in methods_spec if isinstance(entry, dict)]
    if len(method_ids) != len(methods_spec) or len(set(method_ids)) != len(method_ids):
        raise MergeError("method roster is malformed or duplicated")
    if NATURAL_METHOD_ID not in method_ids:
        raise MergeError("merged plan must include an actual identity source method")
    candidate_entries = [entry for entry in methods_spec if entry.get("role") in ("learned_v4", "shared_profile")]
    if sum(entry.get("role") == "learned_v4" for entry in candidate_entries) != 1:
        raise MergeError("candidate family requires exactly one learned v4 method")
    if sum(entry.get("role") == "shared_profile" for entry in candidate_entries) > 3:
        raise MergeError("at most three shared profile candidates can be selected")
    allowed_roles = {"natural_control", "rms_control", "local_control", "learned_v4", "shared_profile"}
    if any(entry.get("role") not in allowed_roles for entry in methods_spec):
        raise MergeError("unknown method role")
    method_source: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    all_contexts: dict[str, dict[tuple[str, int], tuple[Any, ...]]] = {}
    expected_contexts = {
        (str(sample_id), int(seed))
        for sample_id, seed in (
            (item, seed)
            for item in spec.get("sample_ids", [])
            for seed in seeds
        )
    }
    if len(expected_contexts) != int(target.get("expected_samples", 20)) * len(seeds):
        raise MergeError("sample_ids roster must contain the exact expected number of unique clips")
    for entry in methods_spec:
        method_id = entry["method_id"]
        source_id = entry.get("source_id")
        source = source_by_id.get(source_id)
        if source is None or method_id not in source["selected_plan"]:
            raise MergeError(f"{method_id}: source mapping is missing")
        source_method = source["method_map"][method_id]
        method_source[method_id] = (source, source_method)
        selected = source["selected_plan"][method_id]
        if set(selected) != expected_contexts:
            raise MergeError(f"{method_id}: selected sample/seed roster differs from frozen roster")
        by_context = {key: _context(row) for key, row in selected.items()}
        all_contexts[method_id] = by_context
    canonical_contexts = all_contexts[method_ids[0]]
    for method_id in method_ids[1:]:
        if all_contexts[method_id] != canonical_contexts:
            raise MergeError(f"{method_id}: context differs in PCM/face/video/frontend/clock binding")
    sample_speakers: dict[str, str] = {}
    for (sample_id, _seed), context in canonical_contexts.items():
        speaker = context[2]
        old = sample_speakers.setdefault(sample_id, str(speaker))
        if old != str(speaker):
            raise MergeError(f"{sample_id}: speaker identity differs across seeds")
    if len(sample_speakers) != int(target.get("expected_samples", 20)):
        raise MergeError("sample roster does not contain exactly expected sample count")
    if len(set(sample_speakers.values())) != int(target.get("expected_speakers", 10)):
        raise MergeError("speaker roster does not contain exactly expected speaker count")
    ordered_contexts = [(sample_id, int(seed)) for sample_id in spec["sample_ids"] for seed in seeds]
    rows_plan: list[dict[str, Any]] = []
    rows_result: list[dict[str, Any]] = []
    status_counts: dict[str, dict[str, int]] = {}
    for method_id in method_ids:
        source, source_method = method_source[method_id]
        status_counts[method_id] = Counter()
        for context_key in ordered_contexts:
            p_row = dict(source["selected_plan"][method_id][context_key])
            r_row = dict(source["selected_results"][method_id][context_key])
            source_row_id = str(p_row["row_id"])
            derived_id = f"{spec['run_id']}::{source['source_id']}::{source_row_id}"
            lineage = {
                "source_id": source["source_id"],
                "source_run_id": source["plan"].get("run_id"),
                "source_row_id": source_row_id,
                "source_plan_raw_sha256": source["raw_shas"]["plan_path"],
                "source_results_raw_sha256": source["raw_shas"]["results_path"],
                "source_result_manifest_raw_sha256": source["raw_shas"]["result_manifest_path"],
            }
            p_row["row_id"] = derived_id
            p_row["source_lineage"] = lineage
            r_row["row_id"] = derived_id
            r_row["source_lineage"] = lineage
            rows_plan.append(p_row)
            rows_result.append(r_row)
            status_counts[method_id][str(r_row.get("status", "UNKNOWN"))] += 1
    methods_out: list[dict[str, Any]] = []
    registry_candidates: list[dict[str, Any]] = []
    for entry in methods_spec:
        method_id = entry["method_id"]
        source, raw_method = method_source[method_id]
        method = dict(raw_method)
        method["method_id"] = method_id
        if not isinstance(method.get("config_sha256"), str) or len(method["config_sha256"]) != 64:
            raise MergeError(f"{method_id}: frozen source method config SHA is missing")
        method.setdefault("checkpoint_sha256", None)
        if entry["role"] == "learned_v4" and method.get("kind") != "learned_audio_head_candidate":
            raise MergeError(f"{method_id}: learned_v4 role does not match the frozen method kind")
        if entry["role"] == "shared_profile" and method.get("kind") != "shared_frequency_profile_candidate":
            raise MergeError(f"{method_id}: shared_profile role does not match the frozen method kind")
        method["source_id"] = source["source_id"]
        method["source_run_id"] = source["plan"].get("run_id")
        methods_out.append(method)
        if entry["role"] in ("learned_v4", "shared_profile"):
            registry_candidates.append({
                "method_id": method_id,
                "role": entry["role"],
                "config_sha256": method["config_sha256"],
                "checkpoint_sha256": method.get("checkpoint_sha256"),
                "source_id": source["source_id"],
                "source_run_id": source["plan"].get("run_id"),
                "source_plan_raw_sha256": source["raw_shas"]["plan_path"],
                "source_results_raw_sha256": source["raw_shas"]["results_path"],
            })
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    elif any(out_dir.iterdir()):
        raise MergeError(f"output directory must be fresh/empty: {out_dir}")
    source_registry = [
        {"source_id": item["source_id"], "run_id": item["plan"].get("run_id"),
         "plan_raw_sha256": item["raw_shas"]["plan_path"],
         "results_raw_sha256": item["raw_shas"]["results_path"],
         "result_manifest_raw_sha256": item["raw_shas"]["result_manifest_path"],
         "source_paths": item["paths"],
         "included_methods": list(item["source_spec"]["include_methods"])}
        for item in source_by_id.values()
    ]
    plan = {
        "schema_version": SCHEMA,
        "status": "FROZEN",
        "run_id": str(spec["run_id"]),
        "split": "dev",
        "model_id": target["model_id"],
        "audio_frontend": target["audio_frontend"],
        "face_id": target.get("face_id"),
        "sample_ids": list(spec["sample_ids"]),
        "speaker_ids": sorted(set(sample_speakers.values())),
        "confirm_speaker_ids": [],
        "generation_seed_family": {"fit": [], "dev": list(seeds), "confirm": []},
        "methods": methods_out,
        "rows": rows_plan,
        "source_runs": source_registry,
        "candidate_registry_role_policy": {
            "learned_v4": 1, "shared_profile_max": 3,
            "selection_scope": "development_only; candidate names frozen before ASR calls",
        },
    }
    plan["plan_sha256"] = payload_sha(plan, "plan_sha256")
    plan_path = out_dir / "plan.json"
    write_json(plan_path, plan)
    plan_raw = sha_file(plan_path)
    results_path = out_dir / "results.jsonl"
    write_jsonl(results_path, rows_result)
    results_raw = sha_file(results_path)
    manifest = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "run_id": plan["run_id"],
        "plan_raw_sha256": plan_raw,
        "plan_sha256": plan["plan_sha256"],
        "results_jsonl_sha256": results_raw,
        "planned_rows": len(rows_plan),
        "result_rows": len(rows_result),
        "status_counts_by_method": {key: dict(value) for key, value in status_counts.items()},
        "source_runs": source_registry,
        "confirmation_audio_read": False,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_path = out_dir / "result_manifest.json"
    write_json(manifest_path, manifest)
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "status": "FROZEN",
        "run_id": plan["run_id"],
        "plan_raw_sha256": plan_raw,
        "selection_scope": "development_only",
        "candidate_limit": {"learned_v4": 1, "shared_profile_max": 3},
        "candidates": registry_candidates,
    }
    registry["registry_sha256"] = payload_sha(registry, "registry_sha256")
    registry_path = out_dir / "candidate_registry.json"
    write_json(registry_path, registry)
    lineage = {
        "schema_version": "cross-tfg-asr-family-merge-lineage-v1",
        "run_id": plan["run_id"],
        "spec_path": str(spec_path),
        "spec_raw_sha256": sha_file(spec_path),
        "plan_raw_sha256": plan_raw,
        "results_raw_sha256": results_raw,
        "result_manifest_raw_sha256": sha_file(manifest_path),
        "candidate_registry_raw_sha256": sha_file(registry_path),
        "candidate_registry_payload_sha256": registry["registry_sha256"],
        "sources": source_registry,
        "method_order": method_ids,
        "sample_ids": list(spec["sample_ids"]),
        "seed_family": list(seeds),
        "selected_model_id": target["model_id"],
        "selected_audio_frontend": target["audio_frontend"],
        "confirm_audio_read": False,
        "asr_called": False,
    }
    write_json(out_dir / "lineage.json", lineage)
    return {
        "status": "PASS",
        "out_dir": str(out_dir),
        "plan_raw_sha256": plan_raw,
        "results_raw_sha256": results_raw,
        "result_manifest_raw_sha256": sha_file(manifest_path),
        "candidate_registry_raw_sha256": sha_file(registry_path),
        "planned_rows": len(rows_plan),
        "methods": method_ids,
        "status_counts_by_method": {key: dict(value) for key, value in status_counts.items()},
        "confirm_audio_read": False,
        "asr_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    try:
        result = build(Path(args.spec), Path(args.out_dir))
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=os.sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
