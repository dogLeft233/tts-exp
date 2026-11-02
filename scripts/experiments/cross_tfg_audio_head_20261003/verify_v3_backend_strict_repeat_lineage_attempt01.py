#!/usr/bin/env python3
"""Read-only lineage verifier for the three strict-repeat backend receipts.

Accepts each backend's own final manifest and qualification receipt. This is
intentional: a subset retry may qualify Wav2Lip separately while preserving
the passing MuseTalk and Ditto rows from the earlier six-row manifest. The
verifier checks JSON/receipt metadata only; it does not open rendered media,
run a model, call SyncNet, or alter the binder template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
SEED = 20261003
POLICY_SHA256 = "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
NATURAL_FILE_SHA256 = "263b19510c95c32db4874449aadefc9b37c008e4816c18028eaf8fef463828e5"
FACE_FILE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
CANONICAL_AUDIO_FILE_SHA256 = "d12ced7d44e7d70713a7e201abe47ff60ec190af20e9cd447cba5d325c11caa4"
CANONICAL_PCM_SHA256 = "b9a24dd67fd557e012e58339ebe5b1d22655240cf13dd3e31ae7173f3bd0d15a"
SYNCNET_CHECKPOINT_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SAMPLE_COUNT = 92225
SAMPLE_RATE_HZ = 16000

STRICT_POLICY = {
    "pythonhashseed": "20261003",
    "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "openblas_num_threads": "1",
    "numexpr_num_threads": "1",
    "torch_num_threads": 1,
    "torch_num_interop_threads": 1,
    "torch_use_deterministic_algorithms": True,
    "torch_cudnn_deterministic": True,
    "torch_cudnn_benchmark": False,
    "torch_allow_tf32": False,
}
FRONTENDS = {
    "wav2lip_gan_bac9a81e": "wav2lip_official_16k_float32_mel",
    "musetalk_fixed_portrait_attempt16": "musetalk_whisper_fixed_portrait_haar_expanded_box_fp16",
    "ditto_hubert_fixed_portrait_attempt18": "ditto_hubert_onnx_cpu_torch_cuda_static_portrait_attempt18",
}
BACKEND_ENTRIES = {
    "wav2lip_gan_bac9a81e": "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/inference.py",
    "musetalk_fixed_portrait_attempt16": "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt16/code/render_silent_video_attempt09.py",
    "ditto_hubert_fixed_portrait_attempt18": "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt18/code/render_silent_video_attempt07.py",
}
ADAPTER_PINS = {
    "musetalk_fixed_portrait_attempt16": "39d75863cdbd94fc8942c395a2ebf16cacbff74e13bed5dbc8cbd62b74239d14",
    "ditto_hubert_fixed_portrait_attempt18": "7522e08402dab6dc6386f20e4ef667e97fe6ea62febc232e49953ba99b74ddda",
}
EXPECTED_ROWS = {
    model_id: [f"strictrepeat_{model_id}_{repeat:02d}" for repeat in (1, 2)]
    for model_id in MODEL_ORDER
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def regular_file(path_text: str, label: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular non-symlink file: {path}")
    return path


def read_json(path_text: str, label: str) -> tuple[dict[str, Any], bytes, Path]:
    path = regular_file(path_text, label)
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value, raw, path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_bindings(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"{label} must be MODEL_ID=/absolute/path: {item!r}")
        model_id, path = item.split("=", 1)
        if model_id not in MODEL_ORDER or model_id in result or not path.startswith("/"):
            raise ValueError(f"invalid or duplicate {label} binding: {item!r}")
        result[model_id] = path
    if set(result) != set(MODEL_ORDER):
        raise ValueError(f"{label} bindings must name each frozen backend exactly once")
    return result


def verify_actual_file_ref(ref: Mapping[str, Any], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = regular_file(str(ref.get("path", "")), label)
    raw = path.read_bytes()
    digest = sha_bytes(raw)
    require(type(ref.get("bytes")) is int and ref["bytes"] == len(raw),
            f"{label} byte count differs from its sealed reference")
    require(isinstance(ref.get("sha256"), str) and ref["sha256"] == digest,
            f"{label} SHA differs from its sealed reference")
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value, {"path": str(path), "bytes": len(raw), "sha256": digest}


def verify_startup_receipts(row: Mapping[str, Any], model_id: str,
                            row_id: str) -> list[dict[str, Any]]:
    refs = row.get("strict_startup_process_receipts")
    require(isinstance(refs, list) and len(refs) >= 1,
            f"{row_id} has no strict renderer startup receipts")
    verified: list[dict[str, Any]] = []
    matching = []
    expected_entry = BACKEND_ENTRIES[model_id]
    for ref in refs:
        require(isinstance(ref, dict), f"{row_id} has an invalid startup receipt reference")
        record, file_ref = verify_actual_file_ref(ref, f"{row_id} startup receipt")
        require(record.get("schema") == "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1"
                and record.get("status") == "PASS"
                and record.get("receipt_sha256") == payload_sha(record, "receipt_sha256"),
                f"{row_id} startup receipt schema/status/self-hash failed")
        require(record.get("row_id") == row_id and record.get("model_id") == model_id
                and record.get("seed") == SEED
                and record.get("audio_frontend") == FRONTENDS[model_id],
                f"{row_id} startup receipt row/model/seed binding failed")
        require(record.get("bootstrap_source_sha256") == BOOTSTRAP_SHA256
                and record.get("policy") == STRICT_POLICY
                and record.get("policy_payload_sha256") == sha_bytes(canonical(STRICT_POLICY)),
                f"{row_id} startup receipt bootstrap/policy binding failed")
        require(ref.get("entry_matches_backend") is record.get("entry_matches_backend")
                and ref.get("entry_path") == record.get("entry_path")
                and ref.get("policy") == record.get("policy"),
                f"{row_id} startup receipt reference differs from its actual file")
        require(record.get("expected_backend_entry") == expected_entry,
                f"{row_id} startup receipt names another expected backend entry")
        require(type(record.get("pid")) is int and record["pid"] > 0
                and Path(str(ref.get("path", ""))).stem == str(record["pid"]),
                f"{row_id} startup receipt PID does not match its receipt filename")
        if record.get("entry_matches_backend") is True:
            require(record.get("entry_path") == expected_entry,
                    f"{row_id} native startup receipt entry path differs from the pinned backend")
            matching.append(file_ref)
        if model_id == "wav2lip_gan_bac9a81e":
            compatibility = record.get("renderer_compatibility")
            require(isinstance(compatibility, dict)
                    and compatibility.get("id") == "wav2lip_librosa_filters_mel_positional_sr_n_fft_to_keywords_v1"
                    and compatibility.get("installed_before_renderer_import") is True,
                    f"{row_id} lacks the pinned Wav2Lip librosa compatibility receipt")
        else:
            require(record.get("renderer_compatibility") is None,
                    f"{row_id} has an unexpected renderer compatibility change")
        verified.append({**file_ref, "entry_matches_backend": record["entry_matches_backend"],
                         "pid": record.get("pid")})
    require(len(matching) == 1,
            f"{row_id} must have exactly one actual native renderer startup receipt")
    return verified


def verify_native_adapter(row: Mapping[str, Any], model_id: str,
                          row_id: str) -> dict[str, Any] | None:
    ref = row.get("native_adapter_receipt")
    if model_id == "wav2lip_gan_bac9a81e":
        require(ref is None, f"{row_id} unexpectedly carries a native adapter receipt")
        return None
    require(isinstance(ref, dict), f"{row_id} lacks its native adapter receipt")
    receipt, file_ref = verify_actual_file_ref(ref, f"{row_id} native adapter receipt")
    require(receipt == ref.get("receipt"),
            f"{row_id} native adapter file differs from the row's inlined receipt")
    expected_schema = ("cross_tfg_musetalk_stagea_adapter_receipt_v1"
                       if model_id == "musetalk_fixed_portrait_attempt16"
                       else "cross_tfg_ditto_stagea_adapter_receipt_v1")
    expected_status = "PASS" if model_id == "musetalk_fixed_portrait_attempt16" else "PASS_ENGINEERING_ONLY"
    require(receipt.get("schema") == expected_schema and receipt.get("status") == expected_status
            and receipt.get("row_id") == row_id and receipt.get("model_id") == model_id
            and receipt.get("seed") == SEED,
            f"{row_id} native adapter receipt schema/status/identity failed")
    require(receipt.get("audio_frontend") == FRONTENDS[model_id],
            f"{row_id} native adapter receipt audio frontend changed")
    adapter_sha = receipt.get("adapter_sha256")
    if model_id == "ditto_hubert_fixed_portrait_attempt18":
        require(receipt.get("adapter_path", "").endswith("/ditto_five_arg_shim_attempt06.py"),
                f"{row_id} Ditto adapter path changed")
    require(adapter_sha == ADAPTER_PINS[model_id],
            f"{row_id} native adapter source SHA differs from the frozen shim")
    audio = receipt.get("input_audio")
    require(isinstance(audio, dict)
            and audio.get("file_sha256") == CANONICAL_AUDIO_FILE_SHA256
            and audio.get("sample_count") == SAMPLE_COUNT
            and audio.get("sample_rate_hz") == SAMPLE_RATE_HZ
            and audio.get("channels") == 1,
            f"{row_id} native adapter receipt does not bind the fixed mono QC06 audio")
    face = receipt.get("face") or receipt.get("source_face")
    require(isinstance(face, dict) and face.get("file_sha256") == FACE_FILE_SHA256,
            f"{row_id} native adapter receipt does not bind the fixed QC06 portrait")
    return {**file_ref, "schema": receipt["schema"], "status": receipt["status"],
            "row_id": row_id, "adapter_sha256": adapter_sha}


def verify_row(row: Mapping[str, Any], model_id: str, row_id: str) -> dict[str, Any]:
    require(row.get("row_id") == row_id and row.get("model_id") == model_id
            and row.get("seed") == SEED and row.get("status") == "PASS",
            f"{row_id} row identity/status/seed gate failed")
    require(row.get("audio_frontend") == FRONTENDS[model_id]
            and row.get("source_audio_file_sha256") == CANONICAL_AUDIO_FILE_SHA256
            and row.get("source_face_sha256") == FACE_FILE_SHA256,
            f"{row_id} row frontend or fixed source input pin changed")
    pcm = row.get("muxed_original_pcm_readback")
    require(isinstance(pcm, dict) and pcm.get("matches_qc06_input") is True
            and pcm.get("pcm_float32_sha256") == CANONICAL_PCM_SHA256
            and pcm.get("sample_count") == SAMPLE_COUNT
            and pcm.get("sample_rate_hz") == SAMPLE_RATE_HZ
            and pcm.get("channels") == 1,
            f"{row_id} muxed row does not preserve the exact QC06 PCM clock and samples")
    for field in ("resource_before", "resource_after"):
        snapshot = row.get(field)
        require(isinstance(snapshot, dict), f"{row_id} lacks {field} resource identity")
        gpus = snapshot.get("gpus")
        require(isinstance(gpus, list) and any(item.get("uuid") == GPU_UUID for item in gpus),
                f"{row_id} {field} does not contain the pinned GPU identity")
    startup = verify_startup_receipts(row, model_id, row_id)
    adapter = verify_native_adapter(row, model_id, row_id)
    return {"row_id": row_id, "seed": SEED, "status": "PASS",
            "native_renderer_startup_receipts": startup,
            "native_adapter_receipt": adapter}


def verify_backend(model_id: str, manifest_path_text: str,
                   qualification_path_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, manifest_raw, manifest_path = read_json(manifest_path_text, f"{model_id} final manifest")
    qual, qual_raw, qual_path = read_json(qualification_path_text, f"{model_id} qualification receipt")
    manifest_sha = sha_bytes(manifest_raw)
    qual_sha = sha_bytes(qual_raw)
    output_root = manifest_path.parent.resolve(strict=True)
    output_root_text = str(output_root)
    project_root = Path("/root/tts-research-system-20261001")
    forbidden_roots = (
        project_root / "scripts",
        project_root / "cross_tfg_audio_head_20261003" / "model_deploy_attempt16",
        project_root / "cross_tfg_audio_head_20261003" / "model_deploy_attempt18",
        Path("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip"),
        Path("/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs"),
    )
    require("/runs/" in output_root_text and output_root.name.startswith("backend_strict_repeat_"),
            f"{model_id} manifest output root is not a dedicated run-output directory")
    for forbidden in forbidden_roots:
        forbidden_text = str(forbidden.resolve())
        require(not (output_root_text == forbidden_text
                     or output_root_text.startswith(forbidden_text + os.sep)
                     or forbidden_text.startswith(output_root_text + os.sep)),
                f"{model_id} output root overlaps a protected source/model/input tree")
    require(manifest.get("schema") == "cross_tfg_backend_strict_repeat_pilot_manifest_v1"
            and manifest.get("manifest_sha256") == payload_sha(manifest, "manifest_sha256"),
            f"{model_id} final manifest schema/self-hash failed")
    expected_planned = [f"strictrepeat_{backend}_{repeat:02d}"
                        for backend in MODEL_ORDER for repeat in (1, 2)]
    require(manifest.get("planned_rows") == 6 and manifest.get("row_ids") == expected_planned
            and manifest.get("models") == list(MODEL_ORDER),
            f"{model_id} manifest does not preserve the fixed six-row denominator")
    require(model_id in manifest.get("requested_models", [])
            and manifest.get("dev_or_confirm_media_read") is False
            and manifest.get("selection_eligible") is False,
            f"{model_id} manifest is outside the engineering-only qualification scope")
    require(qual.get("schema") == "cross_tfg_backend_strict_determinism_qualification_v1"
            and qual.get("status") == "PASS_EXACT_MATCH"
            and qual.get("receipt_sha256") == payload_sha(qual, "receipt_sha256"),
            f"{model_id} qualification schema/status/self-hash failed")
    require(qual.get("model_id") == model_id and qual.get("run_id") == manifest.get("run_id")
            and qual.get("dev_or_confirm_media_read") is False
            and qual.get("selection_eligible") is False,
            f"{model_id} qualification does not bind the supplied manifest or frozen scope")
    ref = manifest.get("backend_qualifications", {}).get(model_id, {}).get("receipt")
    require(isinstance(ref, dict)
            and ref.get("path") == str(qual_path)
            and ref.get("bytes") == len(qual_raw)
            and ref.get("sha256") == qual_sha,
            f"{model_id} manifest does not pin the supplied qualification file")
    qualification_entry = manifest["backend_qualifications"][model_id]
    require(qualification_entry.get("status") == "PASS_EXACT_MATCH"
            and qualification_entry.get("model_id") == model_id
            and qualification_entry.get("run_id", manifest.get("run_id")) == manifest.get("run_id"),
            f"{model_id} manifest qualification entry is not an exact pass")
    require(qual.get("determinism_policy_sha256") == POLICY_SHA256
            and qual.get("strict_bootstrap_source_sha256") == BOOTSTRAP_SHA256
            and qual.get("gpu_uuid") == GPU_UUID
            and qual.get("policy_values") == STRICT_POLICY,
            f"{model_id} qualification does not match the shared GPU/bootstrap/policy pins")
    gate = qual.get("repeat_gate")
    require(isinstance(gate, dict) and gate.get("status") == "PASS_EXACT_MATCH"
            and gate.get("repeat_row_ids") == EXPECTED_ROWS[model_id],
            f"{model_id} qualification repeat gate is not bound to exactly its two planned rows")
    equality_fields = [key for key in gate if key.endswith("_equal")]
    require(equality_fields and all(gate.get(key) is True for key in equality_fields),
            f"{model_id} qualification has a failed repeat equality gate")
    expected_context = {
        "scope": "engineering_qc06_exposed_input_only",
        "sample_id": "natural_1",
        "seed": SEED,
        "source_file_sha256": NATURAL_FILE_SHA256,
        "face_file_sha256": FACE_FILE_SHA256,
        "canonical_audio_file_sha256": CANONICAL_AUDIO_FILE_SHA256,
        "canonical_audio_pcm_sha256": CANONICAL_PCM_SHA256,
        "sample_count": SAMPLE_COUNT,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": 1,
        "determinism_policy_sha256": POLICY_SHA256,
        "strict_bootstrap_sha256": BOOTSTRAP_SHA256,
        "syncnet_checkpoint_sha256": SYNCNET_CHECKPOINT_SHA256,
    }
    context = qual.get("input_context")
    require(isinstance(context, dict), f"{model_id} qualification lacks its fixed input context")
    normalized_context = {key: value for key, value in context.items() if key != "model_id"}
    require(normalized_context == expected_context,
            f"{model_id} qualification input context differs from the shared QC06 identity")
    require(qual.get("input_context_sha256") == sha_bytes(canonical(context)),
            f"{model_id} input-context self-hash failed")
    runtime = qual.get("backend_runtime_identity", {})
    require(runtime.get("model_binding", {}).get("model_id") == model_id,
            f"{model_id} backend runtime identity names another model")
    if model_id in ADAPTER_PINS:
        generator = runtime.get("generator", {})
        require(generator.get("adapter_sha256") == ADAPTER_PINS[model_id],
                f"{model_id} backend runtime identity adapter SHA changed")

    manifest_rows = manifest.get("rows")
    qual_rows = qual.get("repeat_rows")
    require(isinstance(manifest_rows, list) and isinstance(qual_rows, list),
            f"{model_id} manifest or qualification is missing row data")
    model_rows = {row.get("row_id"): row for row in manifest_rows
                  if isinstance(row, dict) and row.get("model_id") == model_id}
    require(len(model_rows) == 2 and set(model_rows) == set(EXPECTED_ROWS[model_id]),
            f"{model_id} manifest does not contain exactly its two requested rows")
    qual_rows_by_id = {row.get("row_id"): row for row in qual_rows if isinstance(row, dict)}
    require(len(qual_rows_by_id) == 2 and set(qual_rows_by_id) == set(EXPECTED_ROWS[model_id]),
            f"{model_id} qualification receipt does not contain exactly its two planned rows")
    require([row.get("row_id") for row in qual_rows] == EXPECTED_ROWS[model_id],
            f"{model_id} qualification row order differs from the frozen repeat order")
    require(all(qual_rows_by_id[row_id] == model_rows[row_id] for row_id in EXPECTED_ROWS[model_id]),
            f"{model_id} qualification repeat_rows differ from the exact rows in its manifest")
    rows = [verify_row(model_rows[row_id], model_id, row_id) for row_id in EXPECTED_ROWS[model_id]]
    return ({"model_id": model_id, "run_id": manifest["run_id"],
             "manifest": {"path": str(manifest_path), "bytes": len(manifest_raw), "sha256": manifest_sha},
             "output_root": output_root_text,
             "qualification": {"path": str(qual_path), "bytes": len(qual_raw), "sha256": qual_sha},
             "gpu_uuid": GPU_UUID, "seed": SEED, "input_context_sha256": qual["input_context_sha256"],
             "strict_policy_sha256": POLICY_SHA256, "bootstrap_sha256": BOOTSTRAP_SHA256,
             "rows": rows}, normalized_context)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, metavar="MODEL_ID=/absolute/path")
    parser.add_argument("--qualification", action="append", required=True, metavar="MODEL_ID=/absolute/path")
    args = parser.parse_args()
    try:
        manifests = parse_bindings(args.manifest, "--manifest")
        qualifications = parse_bindings(args.qualification, "--qualification")
        results = []
        common_context = None
        for model_id in MODEL_ORDER:
            result, context = verify_backend(model_id, manifests[model_id], qualifications[model_id])
            if common_context is None:
                common_context = context
            elif context != common_context:
                raise ValueError(f"{model_id} does not share the same exact QC06 input/clock context")
            results.append(result)
        output_roots_by_run: dict[str, str] = {}
        for result in results:
            previous = output_roots_by_run.setdefault(result["run_id"], result["output_root"])
            require(previous == result["output_root"],
                    f"run_id {result['run_id']} resolves to multiple output roots")
        distinct_run_outputs = list(output_roots_by_run.values())
        require(len(distinct_run_outputs) == len(set(distinct_run_outputs)),
                "distinct repeat run IDs share an output root")
        output = {"schema": "cross_tfg_backend_strict_repeat_lineage_v1",
                  "status": "PASS_ALL_THREE_BACKEND_REPEAT_QUALIFICATIONS",
                  "planned_rows": 6, "qualified_rows": 6, "models": results,
                  "shared_input_context": common_context,
                  "dev_or_confirm_media_read": False, "api_calls": False,
                  "model_execution": False, "syncnet_scoring": False,
                  "selection_eligible": False}
        output["lineage_sha256"] = sha_bytes(canonical(output))
        print(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        failure = {"schema": "cross_tfg_backend_strict_repeat_lineage_v1",
                   "status": "BLOCKED_OR_INVALID_LINEAGE", "error_type": type(exc).__name__,
                   "error": str(exc), "dev_or_confirm_media_read": False,
                   "api_calls": False, "model_execution": False,
                   "syncnet_scoring": False, "selection_eligible": False}
        failure["lineage_sha256"] = sha_bytes(canonical(failure))
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
