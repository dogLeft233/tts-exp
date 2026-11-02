#!/usr/bin/env python3
"""Read same-host identity/RMS references for the dynamic 540-row V3 join.

This is a narrow clone of the qualified V3 candidate builder adapted to the
fresh same-host 360-control archive. The dynamic joiner sets the archive byte
count and SHA before calling its reader functions. Do not run this adapter's
standalone builder entry point. No audio, video, checkpoint selection, or
confirmation data are accessed by the reader path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Iterator
from pathlib import PurePosixPath


BASE_PLAN_SHA256 = "bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585"
FRONTEND_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
STAGEA_CORE_SHA256 = "6fcef9ac59983af9aeb70fa9fe4ec8bdb6e16e2c2ff3368d142f2702f9352034"
CROSS_FRONTEND_RUNNER_SHA256 = "0183b1d09068df32fb71be42662a453930ed72529f904d82a14360de8ce3089a"
V3_METHOD_ID = "shared_recipient_transition_head_v3_fit40_step120"
V3_METHOD_CONFIG = "v3_parameter_checkpoint"
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
BASE_METHODS = ("identity", "rms_fixed", "rms_localcompress")
REQUIRED_BASELINE_METHODS = ("identity", "rms_fixed")
FIXED_TRAINING_STEPS = 120
BASELINE_EVIDENCE_ARCHIVE_BYTES = None  # dynamic raw pin set only by the same-host joiner

BASELINE_EVIDENCE_ARCHIVE_SHA256 = None  # dynamic raw pin set only by the same-host joiner

BASELINE_ARCHIVE_FRONTENDS = {
    "wav2lip_gan_bac9a81e": "w2l",
    "musetalk_fixed_portrait_attempt16": "muse",
    "ditto_hubert_fixed_portrait_attempt18": "ditto",
}
MODEL_BINDING_FIELDS = (
    "model_id", "audio_frontend", "model_sha256", "preprocessing_sha256",
    "generator_audio_preprocessing_sha256", "score_model_sha256", "fps_num", "fps_den",
)
QUERY_ARRAY_FIELDS = (
    "query_frame_index", "track_frame_index", "query_timestamp_num",
    "query_timestamp_den", "lag_offsets_frames", "valid_query_mask",
)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha(value: dict[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()


def load_frozen_plan(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if sha_bytes(raw) != expected_sha:
        raise ValueError(f"{label} raw SHA mismatch: {sha_bytes(raw)}")
    value = json.loads(raw.decode("utf-8"))
    if value.get("status") != "FROZEN" or value.get("plan_sha256") != payload_sha(value, "plan_sha256"):
        raise ValueError(f"{label} is not a valid frozen plan")
    return value, raw


def _safe_repo_file(project_root: Path, pin: dict[str, Any], label: str) -> tuple[Path, bytes]:
    if set(pin) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} must contain exactly path/bytes/sha256")
    rel = pin["path"]
    if not isinstance(rel, str) or not rel or "\\" in rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError(f"{label} path must be safe and repository-relative")
    lexical = project_root.joinpath(*Path(rel).parts)
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} has a symlink path component: {cursor}")
    path = lexical.resolve(strict=True)
    try:
        path.relative_to(project_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} escapes the declared project root") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular repository file")
    raw = path.read_bytes()
    if len(raw) != pin["bytes"] or sha_bytes(raw) != pin["sha256"]:
        raise ValueError(f"{label} file differs from its declared bytes/SHA")
    return path, raw


def _candidate_bundle(project_root: Path, cli_config_path: Path, expected_raw_sha: str) -> dict[str, Any]:
    config_raw = cli_config_path.read_bytes()
    if sha_bytes(config_raw) != expected_raw_sha:
        raise ValueError("V3 CLI config raw SHA differs from its supplied exact pin")
    config = json.loads(config_raw.decode("utf-8"))
    if config.get("schema") != "cross_tfg_v3_enhancement_cli_config_v1":
        raise ValueError("candidate config is not the frozen V3 CLI schema")
    if config.get("profile_role") != "METHOD_CANDIDATE":
        raise ValueError("Dev slate requires the frozen METHOD_CANDIDATE V3 CLI role")
    parameter = config.get("parameter")
    if not isinstance(parameter, dict) or parameter.get("shape") != [3, 3] or parameter.get("dtype") != "float32":
        raise ValueError("V3 candidate must contain the frozen nine-scalar float32 parameter")
    values = parameter.get("values")
    if not isinstance(values, list) or len(values) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in values):
        raise ValueError("V3 candidate must contain a 3×3 parameter matrix")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for row in values for v in row):
        raise ValueError("V3 candidate parameters must be nine finite JSON numbers")

    source_pins = config.get("source_pins")
    implementation = config.get("implementation")
    provenance = config.get("candidate_provenance")
    if not isinstance(source_pins, dict) or not isinstance(implementation, dict):
        raise ValueError("V3 CLI config is missing implementation/source pins")
    if not isinstance(provenance, dict) or set(provenance) != {"checkpoint", "training_record", "selection_record"}:
        raise ValueError("V3 candidate config requires checkpoint, training record, and fixed selection record")
    entry = source_pins.get("cli")
    if not isinstance(entry, dict) or entry.get("path") != "scripts/audio_enhancement/enhance_v3_attempt02.py":
        raise ValueError("candidate config must pin the exact supported V3 CLI entrypoint")
    for name, pin in source_pins.items():
        if not isinstance(pin, dict):
            raise ValueError(f"invalid V3 source pin: {name}")
        _safe_repo_file(project_root, pin, f"V3 source {name}")
    if implementation.get("path") != "scripts/audio_enhancement/shared_recipient_transition_head_v3.py":
        raise ValueError("candidate config implementation is not the frozen V3 head")
    if implementation != source_pins.get("head_v3"):
        raise ValueError("V3 implementation pin differs from its source closure")

    loaded: dict[str, tuple[dict[str, Any], bytes, Path]] = {}
    for name in ("checkpoint", "training_record", "selection_record"):
        pin = provenance[name]
        path, raw = _safe_repo_file(project_root, pin, f"candidate provenance {name}")
        record = json.loads(raw.decode("utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"candidate {name} must be a JSON object")
        loaded[name] = (record, raw, path)
    checkpoint, checkpoint_raw, checkpoint_path = loaded["checkpoint"]
    training, training_raw, training_path = loaded["training_record"]
    selection, selection_raw, selection_path = loaded["selection_record"]
    constructor = config.get("constructor")
    if not isinstance(constructor, dict):
        raise ValueError("V3 candidate constructor is missing")
    constructor_sha = sha_bytes(canonical(constructor))
    if checkpoint.get("schema") != "cross_tfg_v3_parameter_checkpoint_v1" or checkpoint.get("head_schema") != config.get("head_schema"):
        raise ValueError("step120 checkpoint schema/head mismatch")
    if checkpoint.get("implementation") != implementation or checkpoint.get("constructor_sha256") != constructor_sha:
        raise ValueError("step120 checkpoint implementation/constructor binding mismatch")
    if checkpoint.get("parameter") != parameter:
        raise ValueError("V3 CLI theta differs from the frozen step120 checkpoint")
    if checkpoint.get("selection_status") != "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED":
        raise ValueError("step120 fit checkpoint must not contain a Dev-selected weight")
    bindings = checkpoint.get("training_artifact_bindings")
    if not isinstance(bindings, dict) or bindings.get("completed_optimizer_steps") != FIXED_TRAINING_STEPS:
        raise ValueError("checkpoint does not bind exactly 120 optimizer steps")
    if training.get("schema") != "cross_tfg_v3_candidate_training_record_v1" or training.get("status") != "COMPLETE":
        raise ValueError("V3 training record is not complete")
    if training.get("completed_optimizer_steps") != FIXED_TRAINING_STEPS or training.get("planned_optimizer_steps") != FIXED_TRAINING_STEPS:
        raise ValueError("V3 training record is not fixed at the terminal 120 steps")
    if training.get("checkpoint_sha256") != sha_bytes(checkpoint_raw):
        raise ValueError("V3 training record does not bind the exact checkpoint JSON")
    if training.get("head_implementation_sha256") != implementation["sha256"]:
        raise ValueError("V3 training record does not bind the V3 implementation")
    if training.get("development_audio_opened") is not False or training.get("confirmation_audio_opened") is not False:
        raise ValueError("V3 training record must certify that Dev/Confirm stayed closed during fit")
    if training.get("selection_record_created") is not False or training.get("selection_status") != "NOT_SELECTED":
        raise ValueError("fit record must remain unselected before Dev evaluation")
    if selection.get("schema") != "cross_tfg_v3_candidate_selection_record_v1" or selection.get("status") != "FROZEN":
        raise ValueError("the V3 method freeze record must be present before Dev")
    if selection.get("checkpoint_sha256") != sha_bytes(checkpoint_raw):
        raise ValueError("V3 method freeze record does not bind the step120 checkpoint")
    if selection.get("training_record_sha256") != sha_bytes(training_raw):
        raise ValueError("V3 method freeze record does not bind its complete fit record")
    if selection.get("selection_rule") != "fixed_terminal_optimizer_step_120_no_development_selection":
        raise ValueError("V3 freeze record must name the predeclared fixed-step120 rule")
    if selection.get("selected_optimizer_step") != FIXED_TRAINING_STEPS:
        raise ValueError("V3 freeze record must bind the terminal optimizer step 120")
    if selection.get("development_audio_opened") is not False or selection.get("confirmation_audio_opened") is not False:
        raise ValueError("V3 fixed-step freeze record must certify Dev/Confirm were not used to choose weights")
    return {
        "config": config,
        "config_raw_sha256": expected_raw_sha,
        "checkpoint_sha256": sha_bytes(checkpoint_raw),
        "checkpoint_path": str(checkpoint_path),
        "training_record_sha256": sha_bytes(training_raw),
        "training_record_path": str(training_path),
        "selection_record_sha256": sha_bytes(selection_raw),
        "selection_record_path": str(selection_path),
        "optimizer_steps": FIXED_TRAINING_STEPS,
    }


def _archive_member_bytes(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo], name: str,
                          max_bytes: int) -> bytes:
    info = members.get(name)
    if info is None or not info.isfile() or info.size < 0 or info.size > max_bytes:
        raise ValueError(f"missing, nonregular, or oversized baseline archive member: {name}")
    stream = archive.extractfile(info)
    if stream is None:
        raise ValueError(f"could not read baseline archive member: {name}")
    raw = stream.read(max_bytes + 1)
    if len(raw) != info.size:
        raise ValueError(f"baseline archive member byte count changed: {name}")
    return raw


def _clock_descriptor(row: dict[str, Any], name: str) -> tuple[Any, ...]:
    arrays = row.get("score", {}).get("raw_matrix_npz", {}).get("arrays", {})
    descriptor = arrays.get(name)
    if not isinstance(descriptor, dict):
        raise ValueError(f"baseline score row has no raw matrix descriptor {name}")
    return (tuple(descriptor.get("shape", [])), descriptor.get("dtype"), descriptor.get("array_sha256"))


def _clock_signature(row: dict[str, Any]) -> dict[str, Any]:
    render = row.get("render", {})
    score = row.get("score", {})
    evaluator = score.get("official_evaluator", {})
    receipt = evaluator.get("receipt", {})
    matrix = _clock_descriptor(row, "distance_by_query_lag")
    if len(matrix[0]) != 2 or matrix[0][1] != 31:
        raise ValueError("baseline SyncNet matrix must retain the frozen 31-lag dimension")
    query_count = receipt.get("expected_query_count")
    if type(query_count) is not int or matrix[0][0] != query_count:
        raise ValueError("baseline evaluator query count and distance matrix shape disagree")
    arrays = {name: _clock_descriptor(row, name) for name in QUERY_ARRAY_FIELDS}
    for name in QUERY_ARRAY_FIELDS[:-1]:
        if arrays[name][0] != ((31,) if name == "lag_offsets_frames" else (query_count,)):
            raise ValueError(f"baseline query/lag array shape is invalid: {name}")
    if arrays["valid_query_mask"][0] != (query_count,):
        raise ValueError("baseline valid-query mask length differs from query support")
    if matrix[2] != receipt.get("matrix_sha256"):
        raise ValueError("baseline distance-matrix payload is not bound to evaluator receipt")
    return {
        "fps_num": render.get("fps_num"),
        "fps_den": render.get("fps_den"),
        "render_frame_count": render.get("frame_count"),
        "render_timestamps_sha256": render.get("timestamps_sha256"),
        "expected_query_count": query_count,
        "evaluator_video_frame_count": receipt.get("video_frame_count"),
        "audio_sample_count": receipt.get("audio_sample_count"),
        "audio_sample_rate_hz": receipt.get("audio_sample_rate_hz"),
        "distance_matrix_shape": matrix[0],
        "distance_matrix_dtype": matrix[1],
        "score_model_sha256": score.get("score_model_sha256"),
        "score_preprocessing_sha256": score.get("preprocessing_sha256"),
        "evaluator_checkpoint_sha256": receipt.get("checkpoint_sha256"),
        "query_and_lag_arrays": arrays,
    }


def _baseline_rows_from_evidence_archive(
    archive_path: Path,
    identity_contexts: list[dict[str, Any]],
    models: list[dict[str, Any]],
) -> tuple[dict[str, dict[tuple[str, str, str, int], dict[str, Any]]], dict[str, Any]]:
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("baseline evidence archive must be a regular non-symlink file")
    archive_path = archive_path.resolve(strict=True)
    if archive_path.stat().st_size != BASELINE_EVIDENCE_ARCHIVE_BYTES or sha_file(archive_path) != BASELINE_EVIDENCE_ARCHIVE_SHA256:
        raise ValueError("same-host evidence archive differs from its dynamic raw pin")

    expected_by_context: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in identity_contexts:
        for model in models:
            model_id = model["model_id"]
            key = (row["sample_id"], row["speaker_id"], model_id, int(row["seed"]))
            expected_by_context[key] = row
    if len(expected_by_context) != 20 * 3 * 3:
        raise ValueError("expected baseline denominator is not 20 clips × 3 seeds × 3 frontends")

    controls: dict[str, dict[tuple[str, str, str, int], dict[str, Any]]] = {m: {} for m in REQUIRED_BASELINE_METHODS}
    frontend_records: dict[str, Any] = {}
    run_id_by_model: dict[str, str] = {}
    pair_counts: dict[str, Any] = {}
    expected_model_by_id = {model["model_id"]: model for model in models}

    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"could not open frozen same-host control evidence archive: {exc}") from exc
    with archive:
        members: dict[str, tarfile.TarInfo] = {}
        for info in archive.getmembers():
            name = info.name
            path = PurePosixPath(name)
            if not name or "\\" in name or path.is_absolute() or ".." in path.parts or name in members:
                raise ValueError(f"unsafe or duplicate baseline archive member path: {name!r}")
            if not (info.isdir() or info.isfile()) or info.issym() or info.islnk():
                raise ValueError(f"baseline evidence archive contains a nonregular member: {name}")
            members[name] = info

        archive_manifest_raw = _archive_member_bytes(archive, members, "evidence_archive_manifest.json", 2 * 1024 * 1024)
        archive_manifest = json.loads(archive_manifest_raw.decode("utf-8"))
        if (
            archive_manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_evidence_archive_v1"
            or archive_manifest.get("role") != "read_only_raw_score_and_samehost_identity_rms_control_evidence"
            or archive_manifest.get("confirm_speakers_read") is not False
            or archive_manifest.get("row_count") != 360
            or archive_manifest.get("status") != "PASS"
            or archive_manifest.get("join_eligible") is not True
        ):
            raise ValueError("same-host archive manifest scope/schema is not a passing 360-row control package")

        for model_id, frontend_name in BASELINE_ARCHIVE_FRONTENDS.items():
            expected_model = expected_model_by_id.get(model_id)
            if expected_model is None:
                raise ValueError(f"frozen frontend plan is missing model {model_id}")
            prefix = f"producer/{frontend_name}/"
            plan_raw = _archive_member_bytes(archive, members, prefix + "plan.json", 2 * 1024 * 1024)
            result_manifest_raw = _archive_member_bytes(archive, members, prefix + "result_manifest.json", 128 * 1024)
            results_raw = _archive_member_bytes(archive, members, prefix + "results.jsonl", 16 * 1024 * 1024)
            plan = json.loads(plan_raw.decode("utf-8"))
            result_manifest = json.loads(result_manifest_raw.decode("utf-8"))
            plan_digest = plan.get("plan_sha256")
            if plan.get("status") != "FROZEN" or plan_digest != payload_sha(plan, "plan_sha256"):
                raise ValueError(f"baseline {frontend_name} plan payload digest/status mismatch")
            if plan.get("schema_version") != "cross-tfg-audio-eval-v1":
                raise ValueError(f"baseline {frontend_name} plan schema differs from the qualified row contract")
            plan_models = [x for x in plan.get("models", []) if x.get("model_id") == model_id]
            if len(plan_models) != 1:
                raise ValueError(f"baseline {frontend_name} plan does not have exactly one {model_id} record")
            archived_model = plan_models[0]
            model_mismatches = {
                field: {"baseline": archived_model.get(field), "frozen_fit": expected_model.get(field)}
                for field in MODEL_BINDING_FIELDS
                if archived_model.get(field) != expected_model.get(field)
            }
            if model_mismatches:
                raise ValueError(f"baseline model/source/scorer/frontend pins differ from frozen fit plan: {model_mismatches}")
            result_rows = [json.loads(line.decode("utf-8")) for line in results_raw.splitlines() if line.strip()]
            result_manifest_digest = result_manifest.get("results_jsonl_sha256")
            if (
                result_manifest.get("status") != "PASS"
                or result_manifest.get("run_id") != plan.get("run_id")
                or result_manifest.get("plan_sha256") != plan_digest
                or result_manifest.get("results_jsonl_sha256") != sha_bytes(results_raw)
                or result_manifest.get("result_rows") != result_manifest.get("planned_rows")
                or result_manifest.get("result_rows") != len(result_rows)
            ):
                raise ValueError(f"baseline {frontend_name} result manifest does not bind a complete ledger")
            if result_manifest.get("planned_rows") != plan.get("execution", {}).get("planned_rows"):
                raise ValueError(f"baseline {frontend_name} manifest denominator differs from frozen plan")

            plan_rows_by_id: dict[str, dict[str, Any]] = {}
            for planned in plan.get("rows", []):
                if planned.get("method_id") not in REQUIRED_BASELINE_METHODS:
                    continue
                row_id = planned.get("row_id")
                if not isinstance(row_id, str) or row_id in plan_rows_by_id:
                    raise ValueError(f"baseline {frontend_name} has missing/duplicate planned baseline row ID")
                plan_rows_by_id[row_id] = planned

            per_model_controls = {method: {} for method in REQUIRED_BASELINE_METHODS}
            product_identical_counts = {"generated_video_sha256": 0, "distance_matrix_payload_sha256": 0}
            local_pair_signatures: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = {}
            for line_no, row in enumerate(result_rows, 1):
                method_id = row.get("method_id")
                if method_id not in REQUIRED_BASELINE_METHODS:
                    continue
                key = (row.get("sample_id"), row.get("speaker_id"), row.get("model_id"), row.get("seed"))
                if key not in expected_by_context or row.get("model_id") != model_id:
                    raise ValueError(f"baseline {frontend_name} row {line_no} has an unexpected clip/model/seed")
                if key in per_model_controls[method_id]:
                    raise ValueError(f"baseline {frontend_name} duplicates {method_id} context {key}")
                plan_context = expected_by_context[key]
                planned = plan_rows_by_id.get(row.get("row_id"))
                if planned is None:
                    raise ValueError(f"baseline {frontend_name} result row is absent from its frozen plan: {row.get('row_id')}")
                for field in ("sample_id", "speaker_id", "model_id", "method_id", "seed", "audio_frontend", "face_id"):
                    if row.get(field) != planned.get(field):
                        raise ValueError(f"baseline {frontend_name} row/plan mismatch at {field}: {row.get('row_id')}")
                if row.get("status") != "PASS":
                    raise ValueError(f"frozen {method_id} baseline failed for {key}; no replacement allowed")
                source_pcm = plan_context["natural_pcm"]["pcm_sha256"]
                if row.get("natural_pcm_sha256") != source_pcm or row.get("score_audio_sha256") != source_pcm:
                    raise ValueError(f"baseline {frontend_name} does not score the exact original PCM for {key}")
                if row.get("source_face_sha256") != plan_context["source_face"]["file_sha256"]:
                    raise ValueError(f"baseline {frontend_name} portrait differs from frozen context for {key}")
                if row.get("source_video_sha256") != plan_context["source_video"]["file_sha256"]:
                    raise ValueError(f"baseline {frontend_name} source-video binding differs for {key}")
                if row.get("face_id") != plan_context["face_id"] or row.get("audio_frontend") != expected_model["audio_frontend"]:
                    raise ValueError(f"baseline {frontend_name} face/frontend identity differs for {key}")

                score = row.get("score", {})
                if score.get("score_model_sha256") != expected_model["score_model_sha256"]:
                    raise ValueError(f"baseline {frontend_name} score-model pin differs for {key}")
                if score.get("preprocessing_sha256") != expected_model["preprocessing_sha256"]:
                    raise ValueError(f"baseline {frontend_name} score preprocessing pin differs for {key}")
                evaluator = score.get("official_evaluator", {})
                receipt = evaluator.get("receipt", {})
                if receipt.get("checkpoint_sha256") != expected_model["score_model_sha256"]:
                    raise ValueError(f"baseline {frontend_name} evaluator checkpoint pin differs for {key}")
                if receipt.get("status") != "PASS":
                    raise ValueError(f"baseline {frontend_name} official evaluator receipt is not PASS for {key}")
                if evaluator.get("raw_npz_sha256") != receipt.get("distance_npz_sha256"):
                    raise ValueError(f"baseline {frontend_name} evaluator raw distance NPZ is not receipt-bound for {key}")
                render = row.get("render", {})
                row_video_sha = render.get("generated_video_sha256")
                row_score_npz_sha = score.get("raw_matrix_npz", {}).get("file_sha256")
                if not isinstance(row_video_sha, str) or len(row_video_sha) != 64:
                    raise ValueError(f"baseline {frontend_name} row has no generated-video SHA for {key}")
                if not isinstance(row_score_npz_sha, str) or len(row_score_npz_sha) != 64:
                    raise ValueError(f"baseline {frontend_name} row has no raw-score-array file SHA for {key}")
                if (render.get("fps_num"), render.get("fps_den")) != (expected_model.get("fps_num"), expected_model.get("fps_den")):
                    raise ValueError(f"baseline {frontend_name} render FPS differs from the frozen model contract for {key}")
                if not isinstance(render.get("timestamps_sha256"), str) or len(render["timestamps_sha256"]) != 64:
                    raise ValueError(f"baseline {frontend_name} has no pinned render timestamp hash for {key}")
                signature = _clock_signature(row)
                if signature["score_model_sha256"] != expected_model["score_model_sha256"] or signature["score_preprocessing_sha256"] != expected_model["preprocessing_sha256"]:
                    raise ValueError(f"baseline {frontend_name} clock signature model/preprocessing pin differs for {key}")
                per_model_controls[method_id][key] = row
                local_pair_signatures.setdefault(key, {})[method_id] = signature

            expected_keys = {key for key in expected_by_context if key[2] == model_id}
            for method_id in REQUIRED_BASELINE_METHODS:
                if set(per_model_controls[method_id]) != expected_keys or len(per_model_controls[method_id]) != 60:
                    raise ValueError(f"baseline {frontend_name} {method_id} denominator is not 60/60")
                controls[method_id].update(per_model_controls[method_id])

            clock_pairs = 0
            for key in expected_keys:
                pair = local_pair_signatures.get(key, {})
                if set(pair) != set(REQUIRED_BASELINE_METHODS):
                    raise ValueError(f"baseline {frontend_name} missing identity/RMS pair for {key}")
                if pair["identity"] != pair["rms_fixed"]:
                    unequal = [name for name in pair["identity"] if pair["identity"][name] != pair["rms_fixed"][name]]
                    raise ValueError(f"baseline {frontend_name} identity/RMS source clock/evaluator mismatch for {key}: {unequal}")
                clock_pairs += 1
                # Video and distance-matrix hashes are deliberately not part
                # of the pair signature: they are separate generated/scored
                # outputs for identity and RMS, and each remains row-bound.
                pair_rows = {method: per_model_controls[method][key] for method in REQUIRED_BASELINE_METHODS}
                if pair_rows["identity"]["render"].get("generated_video_sha256") == pair_rows["rms_fixed"]["render"].get("generated_video_sha256"):
                    product_identical_counts["generated_video_sha256"] += 1
                id_matrix = pair_rows["identity"]["score"]["official_evaluator"]["receipt"].get("matrix_sha256")
                rms_matrix = pair_rows["rms_fixed"]["score"]["official_evaluator"]["receipt"].get("matrix_sha256")
                if id_matrix == rms_matrix:
                    product_identical_counts["distance_matrix_payload_sha256"] += 1

            run_id = result_manifest["run_id"]
            run_id_by_model[model_id] = run_id
            frontend_records[model_id] = {
                "frontend_archive_key": frontend_name,
                "run_id": run_id,
                "plan_payload_sha256": plan_digest,
                "plan_raw_sha256": sha_bytes(plan_raw),
                "result_manifest_raw_sha256": sha_bytes(result_manifest_raw),
                "results_jsonl_sha256": result_manifest_digest,
                "results_row_count": len(result_rows),
                "identity_rows": len(per_model_controls["identity"]),
                "rms_fixed_rows": len(per_model_controls["rms_fixed"]),
                "identity_rms_shared_clock_pairs": clock_pairs,
                "identical_product_hash_counts_descriptive_only": product_identical_counts,
                "model_pin_fields_verified": list(MODEL_BINDING_FIELDS),
                "pair_clock_fields_verified": [
                    "fps_num", "fps_den", "render_frame_count", "render_timestamps_sha256",
                    "expected_query_count", "evaluator_video_frame_count", "audio_sample_count",
                    "audio_sample_rate_hz", "distance_matrix_shape", "distance_matrix_dtype",
                    "score_model_sha256", "score_preprocessing_sha256", "evaluator_checkpoint_sha256",
                    "query_frame_index", "track_frame_index", "query_timestamp_num", "query_timestamp_den",
                    "lag_offsets_frames", "valid_query_mask",
                ],
                "pair_clock_exclusions": [
                    "generated_video_sha256", "muxed_video_sha256", "distance_matrix_payload_sha256",
                    "distance_npz_sha256", "raw_npz_file_sha256", "distance_matrix_values",
                ],
            }

    for method_id in REQUIRED_BASELINE_METHODS:
        if set(controls[method_id]) != set(expected_by_context) or len(controls[method_id]) != 180:
            raise ValueError(f"baseline {method_id} denominator is not 180/180 across the three frontends")
    provenance = {
        "archive_path": str(archive_path),
        "archive_bytes": BASELINE_EVIDENCE_ARCHIVE_BYTES,
        "archive_raw_sha256": BASELINE_EVIDENCE_ARCHIVE_SHA256,
        "archive_manifest_raw_sha256": sha_bytes(archive_manifest_raw),
        "archive_manifest_schema": archive_manifest["schema"],
        "archive_confirm_speakers_read": False,
        "archive_source_rows": archive_manifest["row_count"],
        "frontend_ledgers": frontend_records,
        "run_ids_by_model": run_id_by_model,
        "control_rows": {method_id: len(rows) for method_id, rows in controls.items()},
        "baseline_pair_policy": "identity and RMS each retain their own generated-video and score-matrix hashes; pair support requires exact original PCM, source face/video, seed, model, scorer/preprocessing, FPS, frame/query clock, lag array and valid-mask agreement. No cross-method video/matrix hash or distance-value equality is required.",
    }
    return controls, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-dev-plan", type=Path, required=True)
    parser.add_argument("--frontend-roster-plan", type=Path, required=True)
    parser.add_argument("--candidate-cli-config", type=Path, required=True)
    parser.add_argument("--candidate-cli-config-raw-sha256", required=True)
    parser.add_argument("--baseline-evidence-archive", type=Path, required=True)
    parser.add_argument("--stagea-core", type=Path, required=True)
    parser.add_argument("--cross-frontend-runner", type=Path, required=True)
    parser.add_argument("--v3-dispatcher", type=Path, required=True)
    parser.add_argument("--v3-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve(strict=True)
    base, base_raw = load_frozen_plan(args.base_dev_plan, BASE_PLAN_SHA256, "base Dev plan")
    frontend, frontend_raw = load_frozen_plan(args.frontend_roster_plan, FRONTEND_PLAN_SHA256, "frontend roster plan")
    if base.get("schema_version") != "cross-tfg-audio-eval-v1" or frontend.get("schema") not in {
        "cross_tfg_audio_head_fit_plan_v1", "cross_tfg_audio_head_fit_plan_parent_overlay_v1", None
    }:
        raise ValueError("frozen base/frontend plan schemas differ from the qualified inputs")
    if base.get("generation_seed_family", {}).get("dev") != [20261003, 20261004, 20261005]:
        raise ValueError("frozen Dev render seed family changed")
    if base.get("generation_seed_family", {}).get("confirm") != []:
        raise ValueError("frozen base plan unexpectedly has confirmation seeds")
    if len(base.get("rows", [])) != 180 or tuple(method["method_id"] for method in base.get("methods", [])) != BASE_METHODS:
        raise ValueError("frozen Stage-A baseline plan row/method roster changed")
    models = frontend.get("models")
    if not isinstance(models, list) or tuple(model.get("model_id") for model in models) != EXPECTED_MODELS:
        raise ValueError("the exact Wav2Lip/MuseTalk/Ditto model roster changed")
    if {model.get("face_id", "primary_face_static_portrait_v1") for model in models} != {"primary_face_static_portrait_v1"}:
        raise ValueError("candidate Dev slate must retain the common static primary portrait")

    core_path = args.stagea_core.resolve(strict=True)
    cross_path = args.cross_frontend_runner.resolve(strict=True)
    dispatcher_path = args.v3_dispatcher.resolve(strict=True)
    cli_path = args.v3_cli.resolve(strict=True)
    if sha_file(core_path) != STAGEA_CORE_SHA256 or sha_file(cross_path) != CROSS_FRONTEND_RUNNER_SHA256:
        raise ValueError("Stage-A core/cross-frontend adapter differs from the frozen qualified sources")
    cli_sha = sha_file(cli_path)
    # The dispatcher embeds the qualified core/adapter hashes and checks the
    # CLI hash supplied by the plan at invocation. It must be pinned here.
    if len(cli_sha) != 64 or not dispatcher_path.is_file():
        raise ValueError("V3 CLI/dispatcher source identity could not be established")
    candidate = _candidate_bundle(
        project_root,
        args.candidate_cli_config.resolve(strict=True),
        args.candidate_cli_config_raw_sha256.lower(),
    )
    config_cli_pin = candidate["config"].get("source_pins", {}).get("cli", {})
    if config_cli_pin.get("path") != "scripts/audio_enhancement/enhance_v3_attempt02.py" or config_cli_pin.get("sha256") != cli_sha:
        raise ValueError("V3 config CLI pin does not match the exact CLI source passed to the candidate dispatcher")

    identity_rows = [row for row in base["rows"] if row.get("method_id") == "identity"]
    contexts = {(row["sample_id"], row["speaker_id"], int(row["seed"])) for row in identity_rows}
    if len(identity_rows) != 60 or len(contexts) != 60:
        raise ValueError("base plan must contain exactly one identity context for each 20 clips × 3 seeds")
    samples = {row["sample_id"] for row in identity_rows}
    speakers = {row["speaker_id"] for row in identity_rows}
    seeds = {int(row["seed"]) for row in identity_rows}
    if len(samples) != 20 or len(speakers) != 10 or seeds != {20261003, 20261004, 20261005}:
        raise ValueError("base identity contexts no longer match the frozen 20-clip/10-speaker/3-seed cohort")
    identity_rows.sort(key=lambda row: (row["sample_id"], int(row["seed"])))
    face_bindings = {canonical(row.get("source_face")) for row in identity_rows}
    video_bindings = {canonical(row.get("source_video")) for row in identity_rows}
    if len(face_bindings) != 1 or len(video_bindings) != 1:
        raise ValueError("baseline identity contexts do not share one frozen source portrait/video")
    for sample_id in samples:
        sample_rows = [row for row in identity_rows if row["sample_id"] == sample_id]
        if len(sample_rows) != 3 or len({row["speaker_id"] for row in sample_rows}) != 1:
            raise ValueError(f"incomplete speaker/seed context for {sample_id}")
        if len({canonical([row["natural_pcm"], row["score_audio"]]) for row in sample_rows}) != 1:
            raise ValueError(f"natural source/score PCM changed across seeds for {sample_id}")

    controls, control_provenance = _baseline_rows_from_evidence_archive(
        args.baseline_evidence_archive,
        identity_rows,
        models,
    )
    candidate_config_rel = args.candidate_cli_config.resolve(strict=True).relative_to(project_root).as_posix()
    method_config = {
        "method": V3_METHOD_CONFIG,
        "v3_cli_config_path": candidate_config_rel,
        "v3_cli_config_sha256": candidate["config_raw_sha256"],
        "head_schema": candidate["config"]["head_schema"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_training_record_sha256": candidate["training_record_sha256"],
        "candidate_fixed_step_selection_record_sha256": candidate["selection_record_sha256"],
        "fixed_optimizer_steps": FIXED_TRAINING_STEPS,
        "peak_policy": "use the exact frozen V3 CLI constructor; reject any configured peak violation without clipping/post-gain",
        "generator_audio_only": True,
        "score_audio": "exact original natural PCM16 source under the frozen native evaluator",
    }
    method = {
        "method_id": V3_METHOD_ID,
        "kind": "learned_audio_head_candidate",
        "config": method_config,
        "config_sha256": sha_bytes(canonical(method_config)),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "training_record_sha256": candidate["training_record_sha256"],
        "fixed_optimizer_steps": FIXED_TRAINING_STEPS,
        "selection_scope": "fixed terminal fit step 120; weights frozen before Dev and not chosen using Dev; confirmation remains closed",
    }

    # Use the 60 exact identity contexts as the row source and expand each
    # across the three pinned model records. Each planned row gets explicit
    # identity/RMS references from the already complete frozen ledger.
    rows: list[dict[str, Any]] = []
    expected_contexts = {(row["sample_id"], row["speaker_id"], int(row["seed"])) for row in identity_rows}
    model_by_id = {model["model_id"]: model for model in models}
    for source in identity_rows:
        for model_id in EXPECTED_MODELS:
            key = (source["sample_id"], source["speaker_id"], model_id, int(source["seed"]))
            identity_ref = controls["identity"][key]
            rms_ref = controls["rms_fixed"][key]
            model = model_by_id[model_id]
            row = copy.deepcopy(source)
            row["model_id"] = model_id
            row["audio_frontend"] = model["audio_frontend"]
            row["model_role"] = model.get("role", "candidate")
            row["method_id"] = V3_METHOD_ID
            row["method_config_sha256"] = method["config_sha256"]
            row["checkpoint_sha256"] = candidate["checkpoint_sha256"]
            row["baseline_reference_row_ids"] = {
                "identity": identity_ref["row_id"],
                "rms_fixed": rms_ref["row_id"],
            }
            row["baseline_reference_run_ids"] = {
                "identity": control_provenance["run_ids_by_model"][model_id],
                "rms_fixed": control_provenance["run_ids_by_model"][model_id],
            }
            row["generation_settings"] = {
                **row["generation_settings"],
                "generator": model_id,
                "input_audio": "V3 step120 candidate drives generator only; exact original natural PCM is muxed/scored",
                "audio_transform": V3_METHOD_ID,
                "v3_cli_config_sha256": candidate["config_raw_sha256"],
                "head_checkpoint_sha256": candidate["checkpoint_sha256"],
                "training_record_sha256": candidate["training_record_sha256"],
                "seed": int(source["seed"]),
            }
            row["row_id"] = (
                f"dev_{row['speaker_id']}_{row['sample_id']}_{model_id}_"
                f"{row['face_id']}_{int(row['seed'])}_{V3_METHOD_ID}"
            )
            rows.append(row)
    if len(rows) != 180 or len({row["row_id"] for row in rows}) != 180:
        raise ValueError("V3 candidate plan must contain exactly 180 unique learned rows")
    if len(expected_contexts) != 20 * 3:
        raise ValueError("candidate source context denominator changed")

    plan = copy.deepcopy(base)
    plan["run_id"] = "cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt01"
    plan["methods"] = [method]
    plan["models"] = copy.deepcopy(models)
    plan["rows"] = rows
    plan["generation_repeat_qc"] = []
    plan["contrasts"] = []
    plan["analysis"]["contrasts"] = []
    plan["analysis"]["summary_endpoints"] = []
    plan["analysis"]["simultaneous_family_ids"] = []
    plan["analysis"]["simultaneous_family_status"] = "NOT_APPLICABLE_CANDIDATE_ONLY_SUBRUN"
    plan["analysis"]["candidate_only_subrun"] = {
        "candidate_method_id": V3_METHOD_ID,
        "fixed_checkpoint_rule": "terminal_optimizer_step_120_from_fit_only_40_endpoint_roster",
        "weight_selection_using_dev": False,
        "baseline_join": "identity and rms_fixed are immutable reference rows pinned by row IDs in the full common-support control ledger",
    }
    plan["execution"] = {
        **plan.get("execution", {}),
        "stage": "v3_fixed_step120_candidate_only_development_render",
        "scope": "development only; 20 fixed Dev clips × 3 frozen seeds × 3 qualified native frontends",
        "planned_rows": 180,
        "candidate_only": True,
        "fresh_baseline_rerender": False,
        "baseline_controls": control_provenance,
        "baseline_reference_method_ids": list(REQUIRED_BASELINE_METHODS),
        "baseline_join_policy": control_provenance["baseline_pair_policy"],
        "v3_cli_sha256": cli_sha,
        "v3_dispatcher_sha256": sha_file(dispatcher_path),
        "stagea_core_sha256": STAGEA_CORE_SHA256,
        "stagea_runner_sha256": STAGEA_CORE_SHA256,
        "cross_frontend_runner_sha256": CROSS_FRONTEND_RUNNER_SHA256,
        "fixed_optimizer_steps": FIXED_TRAINING_STEPS,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_training_record_sha256": candidate["training_record_sha256"],
        "candidate_selection_record_sha256": candidate["selection_record_sha256"],
        "confirm_audio_read": False,
        "confirm_effects_read": False,
        "development_weights_selected": False,
    }
    plan["protocol"]["generator"]["stagea_runner_sha256"] = STAGEA_CORE_SHA256
    plan["protocol"]["generator"]["cross_frontend_runner_sha256"] = CROSS_FRONTEND_RUNNER_SHA256
    for model in plan["models"]:
        generator = model.get("generator")
        if isinstance(generator, dict):
            generator["runner_sha256"] = STAGEA_CORE_SHA256
    plan["source_reference_plans"] = {
        "base_dev_plan": {"path": str(args.base_dev_plan.resolve()), "raw_sha256": sha_bytes(base_raw), "plan_sha256": base["plan_sha256"]},
        "frontend_roster_plan": {"path": str(args.frontend_roster_plan.resolve()), "raw_sha256": sha_bytes(frontend_raw), "plan_sha256": frontend["plan_sha256"], "model_ids": list(EXPECTED_MODELS)},
        "v3_cli_config": {"path": candidate_config_rel, "raw_sha256": candidate["config_raw_sha256"], "checkpoint_sha256": candidate["checkpoint_sha256"], "training_record_sha256": candidate["training_record_sha256"], "selection_record_sha256": candidate["selection_record_sha256"]},
        "v3_cli": {"path": "scripts/audio_enhancement/enhance_v3_attempt02.py", "raw_sha256": cli_sha},
        "v3_candidate_dispatcher": {"path": str(dispatcher_path), "raw_sha256": sha_file(dispatcher_path)},
        "stagea_core": {"path": str(core_path), "raw_sha256": sha_file(core_path)},
        "cross_frontend_adapter_runner": {"path": str(cross_path), "raw_sha256": sha_file(cross_path)},
        "control_result_ledger": control_provenance,
    }
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = payload_sha(plan, "plan_sha256")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing plan output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    plan_raw = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    rows_raw = b"".join(canonical(row) + b"\n" for row in rows)
    write_new(args.output_dir / "plan.json", plan_raw)
    write_new(args.output_dir / "planned_rows.jsonl", rows_raw)
    manifest = {
        "schema": "cross_tfg_v3_candidate_only_dev_plan_v1",
        "run_id": plan["run_id"],
        "plan_sha256": plan["plan_sha256"],
        "plan_raw_sha256": sha_bytes(plan_raw),
        "planned_rows_sha256": sha_bytes(rows_raw),
        "planned_rows": len(rows),
        "method_counts": {V3_METHOD_ID: len(rows)},
        "model_counts": {model_id: sum(row["model_id"] == model_id for row in rows) for model_id in EXPECTED_MODELS},
        "sample_count": len(samples),
        "speaker_count": len(speakers),
        "render_seed_family": sorted(seeds),
        "fixed_optimizer_steps": FIXED_TRAINING_STEPS,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "identity_control_rows": 180,
        "rms_fixed_control_rows": 180,
        "fresh_baseline_rerender": False,
        "confirm_audio_read": False,
        "confirm_effects_read": False,
        "development_weights_selected": False,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    write_new(args.output_dir / "plan_manifest.json", (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"status": "FROZEN", "run_id": plan["run_id"], "plan_raw_sha256": sha_file(args.output_dir / "plan.json"),
                      "planned_rows_sha256": sha_file(args.output_dir / "planned_rows.jsonl"), "rows": 180,
                      "models": list(EXPECTED_MODELS), "checkpoint_sha256": candidate["checkpoint_sha256"],
                      "identity_control_rows": 180, "rms_fixed_control_rows": 180,
                      "confirm_audio_read": False, "development_weights_selected": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
