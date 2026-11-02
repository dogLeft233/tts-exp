#!/usr/bin/env python3
"""Join actual partial strict-repeat manifests into the frozen six-row QC06 denominator.

This is metadata-only. It opens only terminal pilot manifests and their sealed
per-backend qualification JSON receipts. It does not open audio, video, crops,
frames, tracks, score arrays, Dev/Confirm media, or model weights. A PASS is
written only when complementary partial runs cover all six repeat rows and all
three actual backend receipts pass the frozen equality gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
RUNNER_REL = BASE + "/run_v3_backend_strict_repeat_qualification_attempt04.py"
RUNNER_BYTES = 55_403
RUNNER_SHA256 = "d1c58060b415140ffd13aa156d3adc5f9f8b593a50e54bd24c5e96fc83bb502d"
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
FRONTENDS = {
    MODELS[0]: "wav2lip_official_16k_float32_mel",
    MODELS[1]: "musetalk_whisper_fixed_portrait_haar_expanded_box_fp16",
    MODELS[2]: "ditto_hubert_onnx_cpu_torch_cuda_static_portrait_attempt18",
}
RUN_ID = "cross_tfg_backend_strict_repeat_20261008_attempt03"
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
REPEAT_SEED = 20261003
QC06_MANIFEST_SHA256 = "f09cc4f339c43e96ea532cdaac4f9818fc42caec6ba7b212dcfe7ae4dc2c5436"
QC06_POLICY_SHA256 = "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
STRICT_BOOTSTRAP_SHA256 = "a2a56ad76a94f345e489bc186409958be201c2ad50bfe91c07a6f2902be59855"
STRICT_POLICY = {
    "pythonhashseed": "20261003", "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
    "numexpr_num_threads": "1", "torch_num_threads": 1, "torch_num_interop_threads": 1,
    "torch_use_deterministic_algorithms": True, "torch_cudnn_deterministic": True,
    "torch_cudnn_benchmark": False, "torch_allow_tf32": False,
}
GATE_FIELDS = (
    "generated_video_sha256_equal", "score_npz_sha256_equal", "crop_video_sha256_equal",
    "tracks_pickle_sha256_equal", "face_bbox_arrays_sha256_equal",
    "generated_video_elementary_stream_sha256_equal",
    "syncnet_query_distance_arrays_sha256_equal",
)
COMMON_AUDIO_FIELDS = ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels")
MAX_JSON_BYTES = 64 * 1024 * 1024


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_regular(path: Path, label: str) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > MAX_JSON_BYTES:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    raw = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=False)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(raw) != before.st_size):
        raise ValueError(f"{label} changed while being read")
    return raw


def read_json(path: Path, label: str, expected_raw_sha256: str | None = None) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label)
    digest = sha_bytes(raw)
    if expected_raw_sha256 is not None and digest != expected_raw_sha256.lower():
        raise ValueError(f"{label} raw SHA-256 differs from the caller pin")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, raw


def expected_rows() -> list[str]:
    return [f"strictrepeat_{model_id}_{repeat:02d}" for model_id in MODELS for repeat in (1, 2)]


def common_context(manifest: Mapping[str, Any]) -> dict[str, Any]:
    audio = manifest.get("canonical_qc06_audio")
    if not isinstance(audio, dict):
        raise ValueError("terminal repeat manifest lacks canonical QC06 audio metadata")
    context = {
        "run_id": manifest.get("run_id"), "seed": manifest.get("seed"),
        "policy_values": manifest.get("policy_values"),
        "determinism_policy_sha256": manifest.get("determinism_policy_sha256"),
        "qc06_engineering_manifest_sha256": manifest.get("qc06_engineering_manifest_sha256"),
        "dependency_template": manifest.get("dependency_template"),
        "input_paths": manifest.get("input_paths"),
        "canonical_qc06_audio": {key: audio.get(key) for key in COMMON_AUDIO_FIELDS},
        "wav2lip_python_identity": manifest.get("wav2lip_python_identity"),
        "syncnet_python_identity": manifest.get("syncnet_python_identity"),
    }
    if (context["run_id"] != RUN_ID or context["seed"] != REPEAT_SEED
            or context["policy_values"] != STRICT_POLICY
            or context["determinism_policy_sha256"] != QC06_POLICY_SHA256
            or context["qc06_engineering_manifest_sha256"] != QC06_MANIFEST_SHA256
            or not isinstance(context["dependency_template"], dict)
            or not isinstance(context["input_paths"], dict)):
        raise ValueError("repeat context differs from the fixed QC06 seed/source/determinism contract")
    if not all(context["canonical_qc06_audio"].get(key) not in (None, "") for key in COMMON_AUDIO_FIELDS):
        raise ValueError("canonical QC06 audio identity is incomplete")
    if (context["canonical_qc06_audio"].get("sample_rate_hz") != 16000
            or context["canonical_qc06_audio"].get("channels") != 1):
        raise ValueError("canonical QC06 audio must remain mono 16 kHz")
    return context


def check_qualification(model_id: str, entry: Any, manifest_rows: list[dict[str, Any]],
                        manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict) or entry.get("status") != "PASS_EXACT_MATCH":
        raise ValueError(f"manifest lacks a PASS_EXACT_MATCH qualification for {model_id}")
    ref = entry.get("receipt")
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ValueError(f"manifest lacks the actual qualification receipt reference for {model_id}")
    receipt_path = Path(ref["path"])
    if not receipt_path.is_absolute():
        raise ValueError(f"qualification receipt path is not absolute for {model_id}")
    receipt, raw = read_json(receipt_path, f"{model_id} qualification receipt", ref.get("sha256"))
    digest = sha_bytes(raw)
    if (len(raw) != ref.get("bytes") or receipt.get("schema") != "cross_tfg_backend_strict_determinism_qualification_v1"
            or receipt.get("status") != "PASS_EXACT_MATCH" or receipt.get("model_id") != model_id
            or receipt.get("run_id") != RUN_ID or receipt.get("gpu_uuid") != GPU_UUID
            or receipt.get("seed") not in (None, REPEAT_SEED)
            or receipt.get("determinism_policy_sha256") != QC06_POLICY_SHA256
            or receipt.get("policy_values") != STRICT_POLICY
            or receipt.get("strict_bootstrap_source_sha256") != STRICT_BOOTSTRAP_SHA256
            or receipt.get("qc06_engineering_manifest_sha256") != QC06_MANIFEST_SHA256
            or receipt.get("selection_eligible") is not False
            or receipt.get("dev_or_confirm_media_read") is not False
            or receipt.get("receipt_sha256") != payload_sha(receipt, "receipt_sha256")):
        raise ValueError(f"actual qualification receipt failed source/policy/self-hash gates: {model_id}")
    gate = receipt.get("repeat_gate")
    if (not isinstance(gate, dict) or gate.get("status") != "PASS_EXACT_MATCH"
            or gate.get("repeat_row_ids") != [f"strictrepeat_{model_id}_01", f"strictrepeat_{model_id}_02"]
            or any(gate.get(field) is not True for field in GATE_FIELDS)):
        raise ValueError(f"backend repeat equality gates are incomplete for {model_id}")
    rows = receipt.get("repeat_rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError(f"qualification receipt does not contain two actual rows for {model_id}")
    expected_ids = [f"strictrepeat_{model_id}_01", f"strictrepeat_{model_id}_02"]
    row_by_id = {row.get("row_id"): row for row in rows if isinstance(row, dict)}
    manifest_by_id = {row.get("row_id"): row for row in manifest_rows if row.get("model_id") == model_id}
    if set(row_by_id) != set(expected_ids) or set(manifest_by_id) != set(expected_ids):
        raise ValueError(f"qualification and terminal manifest row IDs disagree for {model_id}")
    for row_id in expected_ids:
        if (row_by_id[row_id].get("status") != "PASS" or row_by_id[row_id].get("model_id") != model_id
                or row_by_id[row_id].get("seed") != REPEAT_SEED
                or canonical(row_by_id[row_id]) != canonical(manifest_by_id[row_id])):
            raise ValueError(f"qualification row does not exactly match terminal manifest evidence: {row_id}")
    context = receipt.get("input_context")
    if (not isinstance(context, dict) or context.get("model_id") != model_id
            or context.get("seed") != REPEAT_SEED
            or context.get("scope") != "engineering_qc06_exposed_input_only"
            or context.get("strict_bootstrap_sha256") != STRICT_BOOTSTRAP_SHA256
            or context.get("determinism_policy_sha256") != QC06_POLICY_SHA256
            or receipt.get("input_context_sha256") != sha_bytes(canonical(context))):
        raise ValueError(f"qualification input context differs from fixed QC06 source/policy: {model_id}")
    manifest_context = common_context(manifest)
    for key in ("source_file_sha256", "face_file_sha256", "canonical_audio_file_sha256",
                "canonical_audio_pcm_sha256", "sample_rate_hz", "channels", "sample_count", "seed"):
        manifest_key = {"source_file_sha256": None, "face_file_sha256": None,
                        "canonical_audio_file_sha256": "file_sha256",
                        "canonical_audio_pcm_sha256": "pcm_sha256", "sample_rate_hz": "sample_rate_hz",
                        "channels": "channels", "sample_count": "sample_count", "seed": None}[key]
        if manifest_key is not None and context.get(key) != manifest_context["canonical_qc06_audio"].get(manifest_key):
            raise ValueError(f"qualification QC06 audio differs across partial runs: {model_id}/{key}")
    if context.get("source_file_sha256") != "263b19510c95c32db4874449aadefc9b37c008e4816c18028eaf8fef463828e5":
        raise ValueError(f"qualification natural_1 source SHA differs from QC06: {model_id}")
    if context.get("face_file_sha256") != "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6":
        raise ValueError(f"qualification portrait SHA differs from QC06: {model_id}")
    identity = receipt.get("backend_runtime_identity")
    runtime_map = manifest.get("runtime_identity", {}).get("backend_identity", {})
    if not isinstance(identity, dict) or canonical(identity) != canonical(runtime_map.get(model_id)):
        raise ValueError(f"qualification backend identity differs from its actual run manifest: {model_id}")
    return {
        "model_id": model_id, "status": receipt["status"],
        "receipt": {"path": str(receipt_path), "bytes": len(raw), "sha256": digest},
        "repeat_row_ids": expected_ids, "repeat_gate": {key: gate[key] for key in GATE_FIELDS},
        "input_context_sha256": receipt.get("input_context_sha256"),
        "backend_runtime_identity_sha256": sha_bytes(canonical(identity)),
    }


def load_terminal_manifest(path: Path, expected_sha: str) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    if path.name != "pilot_manifest_final.json":
        raise ValueError("only pilot_manifest_final.json inputs are eligible for the six-row terminal join")
    manifest, raw = read_json(path, "terminal repeat manifest", expected_sha)
    expected = expected_rows()
    if (manifest.get("schema") != "cross_tfg_backend_strict_repeat_pilot_manifest_v1"
            or manifest.get("run_id") != RUN_ID or manifest.get("planned_rows") != 6
            or manifest.get("row_ids") != expected or manifest.get("seed") != REPEAT_SEED
            or manifest.get("models") != list(MODELS)
            or manifest.get("status") != "INCOMPLETE_BACKEND_SUBSET_WITH_REQUESTED_BACKENDS_PASS"
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")):
        raise ValueError(f"terminal manifest schema, six-row denominator, run identity, or self-hash is invalid: {path}")
    rows = manifest.get("rows")
    requested_models = manifest.get("requested_models")
    requested_rows = manifest.get("requested_row_ids")
    if not isinstance(rows, list) or not isinstance(requested_models, list) or not isinstance(requested_rows, list):
        raise ValueError(f"terminal manifest lacks row or subset scope: {path}")
    if (len(requested_models) == 0 or len(requested_models) != len(set(requested_models))
            or any(model not in MODELS for model in requested_models)):
        raise ValueError(f"terminal manifest backend subset is malformed: {path}")
    if len(requested_models) >= len(MODELS):
        raise ValueError(f"six-row terminal join accepts complementary partial runs, not a full/duplicate run: {path}")
    expected_requested = [f"strictrepeat_{model}_{rep:02d}" for model in MODELS if model in requested_models for rep in (1, 2)]
    if (requested_rows != expected_requested or len(rows) != manifest.get("requested_render_rows")
            or manifest.get("completed_rows") != len(rows)):
        raise ValueError(f"terminal manifest selected rows do not match its requested backend subset: {path}")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"terminal manifest has a non-object row: {path}")
        row_id = row.get("row_id")
        if (row_id not in requested_rows or row_id in seen
                or row.get("model_id") not in requested_models
                or row_id != f"strictrepeat_{row.get('model_id')}_{1 if row_id.endswith('_01') else 2:02d}"
                or row.get("seed") != REPEAT_SEED):
            raise ValueError(f"terminal manifest has an unexpected/duplicate row identity: {path}")
        seen.add(row_id)
    if seen != set(requested_rows):
        raise ValueError(f"terminal manifest does not include every requested row: {path}")
    _ = common_context(manifest)
    return manifest, raw, rows


def join(manifest_refs: list[tuple[Path, str]], output: Path) -> dict[str, Any]:
    runner_path = ROOT / RUNNER_REL
    runner_raw = read_regular(runner_path, "pinned strict-repeat runner")
    if len(runner_raw) != RUNNER_BYTES or sha_bytes(runner_raw) != RUNNER_SHA256:
        raise ValueError("strict-repeat runner source does not match the frozen reviewed identity")
    manifests = []
    all_rows: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    contexts = []
    for path, digest in manifest_refs:
        manifest, raw, rows = load_terminal_manifest(path, digest)
        context = common_context(manifest)
        contexts.append(context)
        manifest_models = set(manifest["requested_models"])
        if manifest_models & set(by_model):
            raise ValueError("partial manifests overlap on requested backend(s)")
        for row in rows:
            all_rows[row["row_id"]] = row
        for model_id in manifest_models:
            entry = manifest["backend_qualifications"].get(model_id)
            qualification = check_qualification(model_id, entry, rows, manifest)
            by_model[model_id] = qualification
        manifests.append({
            "path": str(path.resolve(strict=True)), "bytes": len(raw), "raw_sha256": sha_bytes(raw),
            "status": manifest.get("status"), "requested_models": manifest["requested_models"],
            "requested_rows": manifest["requested_row_ids"], "completed_rows": manifest.get("completed_rows"),
            "row_ids": [row["row_id"] for row in rows],
        })
    if len(manifest_refs) < 2:
        raise ValueError("the current six-row plan requires joining complementary partial runs")
    if any(context != contexts[0] for context in contexts[1:]):
        raise ValueError("partial runs differ in QC06 input, seed, policy, environment, or dependency-template identity")
    expected_ids = expected_rows()
    if set(by_model) != set(MODELS) or set(all_rows) != set(expected_ids):
        raise ValueError("partial manifests do not cover every frozen backend and repeat row")
    for model_id in MODELS:
        expected_pair = [f"strictrepeat_{model_id}_01", f"strictrepeat_{model_id}_02"]
        if all_rows[expected_pair[0]].get("row_id") is None or all_rows[expected_pair[1]].get("row_id") is None:
            raise ValueError(f"all-six row pair is missing for {model_id}")
        if by_model[model_id].get("status") != "PASS_EXACT_MATCH":
            raise ValueError(f"strict repeat backend did not pass exact equality: {model_id}")
    row_evidence = []
    for row_id in expected_ids:
        row = all_rows[row_id]
        if row.get("status") != "PASS":
            raise ValueError(f"all-six terminal row failed: {row_id}")
        fresh = row.get("fresh_syncnet", {})
        if (row.get("audio_frontend") != FRONTENDS.get(row.get("model_id"))
                or row.get("source_audio_file_sha256") != contexts[0]["canonical_qc06_audio"]["file_sha256"]
                or row.get("source_face_sha256") != "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"):
            raise ValueError(f"all-six row has a source/audio frontend mismatch: {row_id}")
        pcm_readback = row.get("muxed_original_pcm_readback")
        if (not isinstance(pcm_readback, dict) or pcm_readback.get("matches_qc06_input") is not True
                or pcm_readback.get("pcm_float32_sha256") != contexts[0]["canonical_qc06_audio"]["pcm_sha256"]
                or pcm_readback.get("sample_count") != contexts[0]["canonical_qc06_audio"]["sample_count"]):
            raise ValueError(f"all-six mux row does not bind the exact QC06 original PCM: {row_id}")
        if not all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in (
                "generated_video_sha256", "generated_video_elementary_stream_sha256", "muxed_video_sha256")):
            raise ValueError(f"all-six row lacks generated media file/stream hash metadata: {row_id}")
        if not isinstance(fresh.get("distance_npz_sha256"), str) or len(fresh["distance_npz_sha256"]) != 64:
            raise ValueError(f"all-six row lacks the actual SyncNet distance array hash: {row_id}")
        row_evidence.append({
            "row_id": row_id, "model_id": row.get("model_id"), "audio_frontend": row.get("audio_frontend"),
            "seed": row.get("seed"), "status": row.get("status"),
            "source_audio_file_sha256": row.get("source_audio_file_sha256"),
            "source_face_sha256": row.get("source_face_sha256"),
            "generated_video_sha256": row.get("generated_video_sha256"),
            "generated_video_elementary_stream_sha256": row.get("generated_video_elementary_stream_sha256"),
            "muxed_video_sha256": row.get("muxed_video_sha256"),
            "muxed_original_pcm_readback": row.get("muxed_original_pcm_readback"),
            "syncnet_distance_npz_sha256": fresh.get("distance_npz_sha256"),
            "tracks_pickle_sha256": (fresh.get("tracking_artifacts") or {}).get("tracks_pickle_sha256"),
        })
    result: dict[str, Any] = {
        "schema": "cross_tfg_backend_strict_repeat_six_row_join_v1",
        "status": "PASS_ALL_SIX_ACTUAL_STRICT_REPEATS",
        "engineering_only": True, "selection_eligible": False,
        "dev_or_confirm_media_read": False, "model_execution_performed": False,
        "run_id": RUN_ID, "gpu_uuid": GPU_UUID, "seed": REPEAT_SEED,
        "determinism_policy_sha256": QC06_POLICY_SHA256,
        "strict_bootstrap_source_sha256": STRICT_BOOTSTRAP_SHA256,
        "qc06_engineering_manifest_sha256": QC06_MANIFEST_SHA256,
        "planned_rows": 6, "completed_rows": 6, "row_ids": expected_ids,
        "partial_manifests": manifests,
        "backend_qualifications": [by_model[model_id] for model_id in MODELS],
        "rows": row_evidence,
        "scope": {"json_metadata_opened_only": True, "waveform_or_video_content_opened": False,
                  "scores_recomputed": False, "criteria_changed": False},
    }
    result["join_sha256"] = payload_sha(result, "join_sha256")
    raw = pretty(result)
    target = Path(os.path.abspath(os.fspath(output)))
    if target.exists() or target.is_symlink() or getattr(target, "is_junction", lambda: False)():
        raise FileExistsError(f"refusing to replace six-row join output: {target}")
    cursor = Path(target.anchor)
    for part in target.parent.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"join output parent traverses a symlink/junction: {cursor}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": result["status"], "path": str(target.resolve()),
            "bytes": len(raw), "raw_sha256": sha_bytes(raw), "join_sha256": result["join_sha256"],
            "planned_rows": 6, "completed_rows": 6, "qualification_backend_count": len(MODELS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path,
                        help="terminal pilot_manifest_final.json; provide complementary partial run manifests")
    parser.add_argument("--manifest-raw-sha256", action="append", required=True,
                        help="exact raw file SHA for each --manifest, in the same order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if len(args.manifest) != len(args.manifest_raw_sha256) or len(args.manifest) < 2:
            raise ValueError("provide at least two manifest paths and matching raw SHA pins")
        result = join(list(zip(args.manifest, args.manifest_raw_sha256)), args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
