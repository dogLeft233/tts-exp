#!/usr/bin/env python3
"""W2L-only Stage-A adapter for prevalidated shared Fit40 SPSA waveforms.

The CPU coordinator validates the shared transform and all three model
projections before handing candidate rows to a model worker.  This module is
Python 3.8 compatible and deliberately does not import the Python 3.10 shared
transform helper.  It rechecks the raw/self-hashed shared and W2L-projection
receipts, the actual shared WAV file/PCM pin, original score audio, and the
exact family/point row context, then invokes the frozen Stage-A W2L
``_render_and_score`` primitive for one row at a time.

This adapter does not initialize or step SPSA, select a candidate, access
Dev/Confirm, or run the full 240-row Stage-A entry point.  The existing source
plan and runtime binding continue to govern native W2L rendering, capacity,
frame clocks, original-PCM delivery, and official SyncNet scoring.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


W2L_MODEL_ID = "wav2lip_gan_bac9a81e"
MODEL_ORDER = (
    W2L_MODEL_ID,
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
SHARED_SCHEMA = "cross_tfg_fit_snapshot_shared_application_v1"
SHARED_STATUS = "PASS_FIT_SNAPSHOT_SHARED_APPLICATION_ONLY"
PROJECTION_SCHEMA = "cross_tfg_fit_snapshot_model_projection_v1"
PROJECTION_STATUS = "PASS_FIT_SNAPSHOT_MODEL_PROJECTION_ONLY"
STAGEA_RUNNER_REL = Path("scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_w2l_block_attempt02.py")
STAGEA_RUNNER_BYTES = 96537
STAGEA_RUNNER_SHA256 = "6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e"
SHARED_FIELDS = (
    "proposal_id", "pair_id", "iteration", "side", "batch_id", "c_k",
    "point_kind", "source_step", "method_id", "method", "theta",
    "theta_sha256", "method_config_sha256", "artifact_sha256",
    "source_snapshot_artifact_sha256", "source_snapshot_step",
    "source_seed_method_id", "binding_sha256", "input_pins_sha256",
    "head_family", "source_theta", "source_theta_sha256",
)


class W2LSPSAAdapterError(RuntimeError):
    """A W2L SPSA row, source pin, or native execution violates its contract."""


class W2LSPSASubsetFailure(W2LSPSAAdapterError):
    """A bounded W2L subset failed while retaining completed row evidence."""

    def __init__(self, message: str, *, completed: Sequence[Mapping[str, Any]],
                 failed_row_id: str, unattempted_row_ids: Sequence[str],
                 failed_result: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.completed = list(completed)
        self.failed_row_id = failed_row_id
        self.unattempted_row_ids = list(unattempted_row_ids)
        self.failed_result = dict(failed_result) if isinstance(failed_result, Mapping) else None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha256(_canonical({key: item for key, item in value.items() if key != field}))


def _read_regular(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> bytes:
    if path.is_symlink():
        raise W2LSPSAAdapterError("refusing a symlink input: " + str(path))
    try:
        info = path.lstat()
    except OSError as exc:
        raise W2LSPSAAdapterError("required input is absent: " + str(path)) from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > max_bytes:
        raise W2LSPSAAdapterError("input is not a bounded nonempty regular file: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) != info.st_size or len(raw) > max_bytes:
        raise W2LSPSAAdapterError("input size changed or exceeded its read bound: " + str(path))
    return raw


def _read_file_pin(pin: Mapping[str, Any], label: str, *, self_field: Optional[str] = None,
                   max_bytes: int = 32 * 1024 * 1024) -> Tuple[Path, bytes, Dict[str, Any]]:
    required = {"path", "bytes", "sha256"}
    if self_field is not None:
        required.add(self_field)
    if not isinstance(pin, Mapping) or set(pin.keys()) != required:
        raise W2LSPSAAdapterError(label + " pin has missing or unexpected fields")
    path_text = pin.get("path")
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise W2LSPSAAdapterError(label + " path must be absolute")
    path = Path(path_text)
    raw = _read_regular(path, max_bytes=max_bytes)
    actual = {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": _sha256(raw)}
    if actual["bytes"] != pin.get("bytes") or actual["sha256"] != pin.get("sha256"):
        raise W2LSPSAAdapterError(label + " bytes or raw SHA differ from its file pin")
    return path.resolve(strict=True), raw, actual


def _read_json_pin(pin: Mapping[str, Any], label: str, *, self_field: Optional[str] = None,
                   max_bytes: int = 32 * 1024 * 1024) -> Tuple[Dict[str, Any], bytes, Dict[str, Any]]:
    _path, raw, actual = _read_file_pin(pin, label, self_field=self_field, max_bytes=max_bytes)
    try:
        packet = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W2LSPSAAdapterError(label + " is not valid UTF-8 JSON") from exc
    if not isinstance(packet, dict):
        raise W2LSPSAAdapterError(label + " root must be a JSON object")
    if self_field is not None and packet.get(self_field) != _self_hash(packet, self_field):
        raise W2LSPSAAdapterError(label + " canonical self hash is invalid")
    if self_field is not None and packet.get(self_field) != pin.get(self_field):
        raise W2LSPSAAdapterError(label + " self hash differs from its declared pin")
    return packet, raw, actual


def _pin_with_self(path: Path, raw: bytes, self_sha: str) -> Dict[str, Any]:
    return {"path": str(path.resolve(strict=True)), "bytes": len(raw),
            "sha256": _sha256(raw), "self_sha256": self_sha}


def _plan_pin(plan: Mapping[str, Any], pin: Mapping[str, Any], label: str) -> Dict[str, Any]:
    path, raw, actual = _read_file_pin(pin, label, self_field="self_sha256", max_bytes=32 * 1024 * 1024)
    try:
        disk = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W2LSPSAAdapterError(label + " is not JSON") from exc
    if (not isinstance(disk, dict) or disk.get("schema_version") != PLAN_SCHEMA
            or disk.get("status") != "FROZEN"
            or disk.get("plan_sha256") != _self_hash(disk, "plan_sha256")
            or pin.get("self_sha256") != disk.get("plan_sha256")
            or canonical_json(disk) != canonical_json(plan)):
        raise W2LSPSAAdapterError(label + " is not its exact frozen self-hashed plan")
    return {**actual, "self_sha256": disk["plan_sha256"]}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _model_context(family_plan: Mapping[str, Any], model_id: str) -> Dict[str, str]:
    execution = family_plan.get("execution", {})
    models = [item for item in family_plan.get("models", [])
              if isinstance(item, Mapping) and item.get("model_id") == model_id]
    protocols = execution.get("source_model_protocols", {})
    source_execution = execution.get("source_model_execution", {})
    runtime_refs = execution.get("runtime_bindings_by_model", {})
    if (len(models) != 1 or model_id not in protocols or model_id not in source_execution
            or model_id not in runtime_refs):
        raise W2LSPSAAdapterError("family lacks exact per-model descriptor/protocol/execution/runtime context")
    items = {
        "model_descriptor": models[0],
        "source_model_protocol": protocols[model_id],
        "source_model_execution": source_execution[model_id],
        "runtime_binding_reference": runtime_refs[model_id],
    }
    return {key + "_sha256": _sha256(_canonical(value)) for key, value in items.items()}


def _method_binding(config: Mapping[str, Any], config_sha: str) -> Dict[str, Any]:
    result = {key: config[key] for key in SHARED_FIELDS if key in config}
    for key in ("method_id", "method", "theta", "theta_sha256", "method_config_sha256",
                "artifact_sha256", "source_snapshot_artifact_sha256", "source_snapshot_step",
                "source_seed_method_id", "binding_sha256", "input_pins_sha256", "head_family",
                "source_theta", "source_theta_sha256"):
        if key in config:
            result[key] = config[key]
    result["method_config_sha256"] = config_sha
    return result


def _load_point_method(point_plan: Mapping[str, Any]) -> Mapping[str, Any]:
    methods = point_plan.get("methods")
    if not isinstance(methods, list) or len(methods) != 1 or not isinstance(methods[0], Mapping):
        raise W2LSPSAAdapterError("point plan must contain exactly one method")
    method = methods[0]
    config = method.get("config")
    if (not isinstance(config, Mapping)
            or config.get("method") not in {"fit40_spsa_proposal", "fit40_spsa_center"}
            or method.get("kind") != config.get("method")
            or config.get("method_id") != method.get("method_id")
            or method.get("config_sha256") != _sha256(_canonical(config))):
        raise W2LSPSAAdapterError("point method/config identity is not a frozen SPSA proposal or center")
    return method


def _identity_rows(family_plan: Mapping[str, Any]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    rows = family_plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 720:
        raise W2LSPSAAdapterError("family plan must preserve the complete 720-row Fit parent")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("method_id") != "identity":
            continue
        if row.get("split") != "fit" or row.get("score_audio") != row.get("natural_pcm"):
            raise W2LSPSAAdapterError("family identity parent is not Fit-only original score audio")
        key = (str(row.get("sample_id")), str(row.get("model_id")))
        if key in result:
            raise W2LSPSAAdapterError("family plan contains duplicate sample/model identity rows")
        result[key] = row
    if len(result) != 120:
        raise W2LSPSAAdapterError("family plan must contain 40 identity rows for each of three models")
    return result


def _validated_source_natural_context(core: Any, project_root: Path,
                                      family_plan: Mapping[str, Any],
                                      sample_id: str,
                                      parent: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-read one exact natural WAV through both the plan pin and frozen core."""
    natural = parent.get("natural_pcm")
    pins = family_plan.get("scientific_input_pins")
    if not isinstance(natural, Mapping) or not isinstance(pins, Mapping):
        raise W2LSPSAAdapterError("identity parent lacks its natural PCM or family input pins")
    pin = pins.get("natural_pcm_" + sample_id)
    if (not isinstance(pin, Mapping) or set(pin) != {"path", "bytes", "sha256"}
            or not isinstance(pin.get("path"), str) or not pin["path"]
            or type(pin.get("bytes")) is not int or pin["bytes"] <= 0
            or not isinstance(pin.get("sha256"), str) or len(pin["sha256"]) != 64):
        raise W2LSPSAAdapterError("identity parent has no exact pinned natural PCM input")
    rel = PurePosixPath(pin["path"])
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise W2LSPSAAdapterError("natural PCM plan pin is not a safe project-relative path")
    source_path = project_root.joinpath(*rel.parts)
    cursor = project_root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise W2LSPSAAdapterError("natural PCM input path traverses a symlink")
    if source_path.is_symlink() or not source_path.is_file():
        raise W2LSPSAAdapterError("pinned natural PCM is absent or not a regular file")
    source_path = source_path.resolve(strict=True)
    planned_path = natural.get("path")
    declared_path = Path(planned_path) if isinstance(planned_path, str) else None
    if (declared_path is None or not declared_path.is_absolute() or declared_path.is_symlink()
            or declared_path != source_path):
        raise W2LSPSAAdapterError("natural PCM row path is not the exact resolved family input path")
    raw = _read_regular(source_path, max_bytes=32 * 1024 * 1024)
    raw_sha = _sha256(raw)
    if (len(raw) != pin["bytes"] or raw_sha != pin["sha256"]
            or natural.get("file_sha256") != raw_sha):
        raise W2LSPSAAdapterError("natural PCM source bytes differ from the exact family pin")
    decoded_path, samples, actual = _read_audio_pin(core, natural, "family natural PCM")
    if decoded_path != source_path:
        raise W2LSPSAAdapterError("frozen core resolved the natural PCM to a different path")
    expected = {
        "path": str(source_path),
        "bytes": len(raw),
        "file_sha256": raw_sha,
        "pcm_sha256": natural.get("pcm_sha256"),
        "sample_count": natural.get("sample_count"),
        "sample_rate_hz": 16000,
        "channels": 1,
        "subtype": natural.get("subtype"),
    }
    if (actual.get("pcm_sha256") != expected["pcm_sha256"]
            or actual.get("sample_count") != expected["sample_count"]
            or actual.get("sample_rate_hz") != 16000 or actual.get("channels") != 1
            or samples.size != expected["sample_count"]
            or type(expected["sample_count"]) is not int or expected["sample_count"] <= 0
            or not isinstance(expected["subtype"], str) or not expected["subtype"]):
        raise W2LSPSAAdapterError("natural PCM decode, subtype, or source clock differs from the family row")
    return expected


def _load_w2l_source(family_plan: Mapping[str, Any], project_root: Path,
                     core: Any) -> Tuple[Dict[str, Any], bytes, Path, List[Dict[str, Any]], bytes,
                                         Path, Dict[str, Any]]:
    entries = [item for item in family_plan.get("execution", {}).get("source_model_plans", [])
               if isinstance(item, Mapping) and item.get("model_id") == W2L_MODEL_ID]
    if len(entries) != 1:
        raise W2LSPSAAdapterError("family plan must bind exactly one frozen W2L source plan")
    entry = entries[0]
    plan_ref = entry.get("plan")
    rows_ref = entry.get("planned_rows")
    if not isinstance(plan_ref, Mapping) or not isinstance(rows_ref, Mapping):
        raise W2LSPSAAdapterError("family W2L source entry omits exact plan/row pins")
    plan_path, plan_raw, _ = _read_file_pin(plan_ref, "frozen W2L source plan")
    rows_path, rows_raw, _ = _read_file_pin(rows_ref, "frozen W2L planned rows")
    try:
        source = json.loads(plan_raw.decode("utf-8"))
        planned_rows = [json.loads(line) for line in rows_raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W2LSPSAAdapterError("W2L source plan/rows are not valid UTF-8 JSON") from exc
    if (not isinstance(source, dict) or source.get("schema_version") != PLAN_SCHEMA
            or source.get("status") != "FROZEN"
            or source.get("plan_sha256") != _self_hash(source, "plan_sha256")
            or plan_ref.get("bytes") != len(plan_raw) or rows_ref.get("bytes") != len(rows_raw)
            or len(planned_rows) != 240 or planned_rows != source.get("rows")
            or entry.get("execution") != source.get("execution")
            or family_plan.get("execution", {}).get("source_model_execution", {}).get(W2L_MODEL_ID)
               != source.get("execution")
            or family_plan.get("execution", {}).get("source_model_protocols", {}).get(W2L_MODEL_ID)
               != source.get("protocol")
            or plan_ref.get("sha256") != _sha256(plan_raw)
            or rows_ref.get("sha256") != _sha256(rows_raw)):
        raise W2LSPSAAdapterError("W2L source plan/row bytes differ from the composed family refs")
    runner = project_root / STAGEA_RUNNER_REL
    if (runner.is_symlink() or not runner.is_file()
            or runner.lstat().st_size != STAGEA_RUNNER_BYTES
            or _sha256(_read_regular(runner, max_bytes=2 * 1024 * 1024)) != STAGEA_RUNNER_SHA256
            or Path(str(getattr(core, "__file__", ""))).resolve(strict=True) != runner.resolve(strict=True)
            or source.get("execution", {}).get("stagea_runner_sha256") != STAGEA_RUNNER_SHA256
            or entry.get("execution", {}).get("stagea_runner_sha256") != STAGEA_RUNNER_SHA256):
        raise W2LSPSAAdapterError("loaded Stage-A runner is not the exact frozen W2L core")
    model_list = source.get("models", [])
    if (not isinstance(model_list, list) or len(model_list) != 1
            or model_list[0].get("model_id") != W2L_MODEL_ID):
        raise W2LSPSAAdapterError("frozen W2L source plan has the wrong model descriptor")
    composed_models = [item for item in family_plan.get("models", [])
                       if item.get("model_id") == W2L_MODEL_ID]
    if len(composed_models) != 1 or composed_models[0].get("generator") != model_list[0].get("generator"):
        raise W2LSPSAAdapterError("composed W2L runtime descriptor differs from its exact source plan")
    return source, plan_raw, plan_path, planned_rows, rows_raw, rows_path, dict(model_list[0])


def _read_audio_pin(core: Any, ref: Mapping[str, Any], label: str) -> Tuple[Path, Any, Dict[str, Any]]:
    path_text = ref.get("path")
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise W2LSPSAAdapterError(label + " path must be absolute")
    path = Path(path_text)
    if path.is_symlink():
        raise W2LSPSAAdapterError(label + " must not be a symlink")
    samples_path, samples, actual = core._reference(ref)
    if (samples_path.resolve(strict=True) != path.resolve(strict=True)
            or actual.get("file_sha256") != ref.get("file_sha256")
            or actual.get("pcm_sha256") != ref.get("pcm_sha256")
            or actual.get("sample_count") != ref.get("sample_count")
            or actual.get("sample_rate_hz") != 16000 or actual.get("channels") != 1
            or samples.size != ref.get("sample_count")):
        raise W2LSPSAAdapterError(label + " file, canonical PCM, or source clock differs from its pin")
    return samples_path, samples, actual


def _validate_official_arrays(arrays: Mapping[str, Any], core: Any) -> Dict[str, int]:
    """Validate the full native SyncNet matrix and its source-clock axes."""
    expected = {
        "query_frame_index": ("<i4", 1),
        "query_timestamp_num": ("<i8", 1),
        "query_timestamp_den": ("<i8", 1),
        "track_frame_index": ("<i4", 1),
        "valid_query_mask": ("|b1", 1),
        "distance_by_query_lag": ("<f4", 2),
        "lag_offsets_frames": ("<i4", 1),
    }
    if not isinstance(arrays, Mapping) or set(arrays) != set(expected):
        raise W2LSPSAAdapterError("Stage-A returned a noncanonical official-array set")
    shapes = {}
    for name, (dtype, ndim) in expected.items():
        value = arrays[name]
        shape = getattr(value, "shape", None)
        dtype_str = getattr(getattr(value, "dtype", None), "str", None)
        if (not isinstance(shape, tuple) or len(shape) != ndim or dtype_str != dtype
                or any(type(size) is not int or size < 0 for size in shape)):
            raise W2LSPSAAdapterError("official array has an unexpected shape or dtype: " + name)
        shapes[name] = shape
    query_count = shapes["query_frame_index"][0]
    if (query_count <= 0 or shapes["distance_by_query_lag"] != (query_count, 31)
            or shapes["query_timestamp_num"] != (query_count,)
            or shapes["query_timestamp_den"] != (query_count,)
            or shapes["track_frame_index"] != (query_count,)
            or shapes["valid_query_mask"] != (query_count,)
            or shapes["lag_offsets_frames"] != (31,)
            or arrays["lag_offsets_frames"].tolist() != list(range(-15, 16))):
        raise W2LSPSAAdapterError("official SyncNet arrays do not have one actual-query × frozen-31-lag shape")
    denominators = arrays["query_timestamp_den"].tolist()
    if any(int(value) <= 0 for value in denominators):
        raise W2LSPSAAdapterError("official source-clock denominators must be positive")
    keys = []
    for frame, numerator, denominator, track in zip(
            arrays["query_frame_index"].tolist(), arrays["query_timestamp_num"].tolist(),
            denominators, arrays["track_frame_index"].tolist()):
        timestamp = Fraction(int(numerator), int(denominator))
        keys.append((int(frame), timestamp.numerator, timestamp.denominator, int(track)))
    if len(set(keys)) != query_count:
        raise W2LSPSAAdapterError("official query source-clock keys are not unique")
    matrix = arrays["distance_by_query_lag"]
    if any(not math.isfinite(float(value)) for value in matrix.flat):
        raise W2LSPSAAdapterError("official SyncNet matrix contains nonfinite values")
    mask = arrays["valid_query_mask"].tolist()
    if any(type(value) not in (bool, int) or int(value) not in (0, 1) for value in mask):
        raise W2LSPSAAdapterError("official valid-query mask is not boolean")
    # The frozen core owns the canonical clock-key derivation. Require it to
    # accept the same arrays after the independent shape/dtype checks above.
    core_keys = core._query_keys(arrays)
    if len(core_keys) != query_count or len(set(core_keys)) != query_count:
        raise W2LSPSAAdapterError("frozen core query-key derivation rejects or duplicates native queries")
    return {"query_count": query_count,
            "valid_query_count": sum(bool(value) for value in mask)}


def _validate_point_and_candidate(family_plan: Mapping[str, Any], family_pin: Mapping[str, Any],
                                  point_plan: Mapping[str, Any], point_pin: Mapping[str, Any],
                                  method: Mapping[str, Any], identity_by_context: Mapping[Tuple[str, str], Mapping[str, Any]],
                                  rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    config = method["config"]
    scope = point_plan.get("execution", {}).get("spsa_execution_scope", {})
    samples = config.get("sample_ids") if config.get("method") == "fit40_spsa_proposal" else scope.get("sample_ids")
    expected_n = 8 if config.get("method") == "fit40_spsa_proposal" else 40
    if (not isinstance(samples, list) or len(samples) != expected_n or len(set(samples)) != expected_n
            or scope.get("sample_ids") != samples):
        raise W2LSPSAAdapterError("point plan does not bind the exact ordered eight- or forty-clip sample roster")
    point_rows = point_plan.get("rows")
    if not isinstance(point_rows, list) or len(point_rows) != expected_n * len(MODEL_ORDER):
        raise W2LSPSAAdapterError("point plan row denominator is not the exact sample × three-model product")
    expected_point_by_context = {}
    expected_order = [(sample_id, model_id) for sample_id in samples for model_id in MODEL_ORDER]
    observed_order = []
    for point_row in point_rows:
        key = (str(point_row.get("sample_id")), str(point_row.get("model_id")))
        if key in expected_point_by_context:
            raise W2LSPSAAdapterError("point plan repeats a sample/model row")
        expected_point_by_context[key] = point_row
        observed_order.append(key)
    if observed_order != expected_order:
        raise W2LSPSAAdapterError("point plan rows differ from exact ordered sample × model roster")
    w2l_rows = [expected_point_by_context[(sample_id, W2L_MODEL_ID)] for sample_id in samples]
    if len(rows) != expected_n:
        raise W2LSPSAAdapterError("W2L candidate subset denominator differs from the point plan")
    candidate_ids = []
    for candidate, planned in zip(rows, w2l_rows):
        key = (str(candidate.get("sample_id")), W2L_MODEL_ID)
        parent = identity_by_context.get(key)
        if parent is None or candidate.get("row_id") != planned.get("row_id"):
            raise W2LSPSAAdapterError("candidate row lacks its exact W2L identity or point-plan row")
        candidate_projection = {key: value for key, value in candidate.items()
                                if key not in {"spsa_binding", "generator_condition_input"}}
        if canonical_json(candidate_projection) != canonical_json(planned):
            raise W2LSPSAAdapterError("candidate row changed fields outside its shared-wave projection")
        if (candidate.get("score_audio") != parent.get("natural_pcm")
                or candidate.get("natural_pcm") != parent.get("natural_pcm")
                or candidate.get("model_id") != W2L_MODEL_ID
                or candidate.get("split") != "fit"
                or candidate.get("spsa_point_binding_sha256") != config.get("binding_sha256")
                or candidate.get("method_config_sha256") != method.get("config_sha256")):
            raise W2LSPSAAdapterError("candidate row changed its W2L source, original score audio, or point binding")
        candidate_ids.append(candidate.get("row_id"))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise W2LSPSAAdapterError("candidate subset contains duplicate row IDs")
    expected_family_pin = {key: family_pin[key] for key in ("path", "bytes", "sha256", "self_sha256")}
    expected_point_pin = {key: point_pin[key] for key in ("path", "bytes", "sha256", "self_sha256")}
    declared_parent = point_plan.get("execution", {}).get("spsa_execution_parent_plan")
    if declared_parent != {key: family_pin[key] for key in ("path", "bytes", "sha256")}:
        raise W2LSPSAAdapterError("point plan does not bind the exact complete family-plan bytes")
    # Pins are returned to each receipt validator to avoid reconstructing a path
    # or trusting an unstaged in-memory plan.
    return w2l_rows, [expected_family_pin, expected_point_pin]


def _validate_shared_projection(core: Any, project_root: Path,
                                family_plan: Mapping[str, Any], family_pin: Mapping[str, Any],
                                point_plan: Mapping[str, Any], point_pin: Mapping[str, Any],
                                method: Mapping[str, Any], identity_by_context: Mapping[Tuple[str, str], Mapping[str, Any]],
                                candidate: Mapping[str, Any], point_row: Mapping[str, Any]) -> Dict[str, Any]:
    binding = candidate.get("spsa_binding")
    input_ref = candidate.get("generator_condition_input")
    if not isinstance(binding, Mapping) or not isinstance(input_ref, Mapping):
        raise W2LSPSAAdapterError("candidate row lacks shared application/projection bindings")
    shared_pin = binding.get("shared_application_receipt")
    projection_pin = binding.get("model_projection_receipt")
    if not isinstance(shared_pin, Mapping) or not isinstance(projection_pin, Mapping):
        raise W2LSPSAAdapterError("candidate row lacks explicit shared and model projection receipt pins")
    shared, _shared_raw, _shared_actual = _read_json_pin(
        shared_pin, "shared application receipt", self_field="self_sha256")
    projection, _projection_raw, _projection_actual = _read_json_pin(
        projection_pin, "W2L model projection receipt", self_field="self_sha256")
    if (shared.get("schema") != SHARED_SCHEMA or shared.get("status") != SHARED_STATUS
            or projection.get("schema") != PROJECTION_SCHEMA
            or projection.get("status") != PROJECTION_STATUS):
        raise W2LSPSAAdapterError("shared application/projection receipts have the wrong schema or status")
    sample_id = str(candidate.get("sample_id"))
    parent_rows = {model_id: identity_by_context.get((sample_id, model_id)) for model_id in MODEL_ORDER}
    if any(not isinstance(parent_rows[model_id], Mapping) for model_id in MODEL_ORDER):
        raise W2LSPSAAdapterError("shared waveform receipt lacks all three exact family identity parents")
    parent = parent_rows[W2L_MODEL_ID]
    natural_ref = parent.get("natural_pcm")
    source_context = _validated_source_natural_context(
        core, project_root, family_plan, sample_id, parent)
    if (shared.get("family_plan") != {key: family_pin[key] for key in ("path", "bytes", "sha256", "self_sha256")}
            or shared.get("point_plan") != {key: point_pin[key] for key in ("path", "bytes", "sha256", "self_sha256")}
            or shared.get("family_plan_self_sha256") != family_plan.get("plan_sha256")
            or shared.get("point_plan_self_sha256") != point_plan.get("plan_sha256")
            or shared.get("sample_id") != sample_id
            or shared.get("identity_parent_row_ids_by_model") != {
                model_id: parent_rows[model_id].get("row_id") for model_id in MODEL_ORDER}
            or shared.get("model_order") != list(MODEL_ORDER)
            or shared.get("source_natural_pcm") != source_context
            or shared.get("render_or_score_performed") is not False
            or shared.get("optimizer_created_or_stepped") is not False
            or shared.get("candidate_selection_performed") is not False
            or shared.get("shared_output_created_once") is not True
            or shared.get("shared_across_model_rows_as_physical_path") is not True):
        raise W2LSPSAAdapterError("shared receipt does not bind this exact family/point/sample identity")
    output = shared.get("output_generator_drive_wav")
    if not isinstance(output, Mapping):
        raise W2LSPSAAdapterError("shared receipt omits its output F32 WAV pin")
    output_path_text = output.get("path")
    if not isinstance(output_path_text, str) or not Path(output_path_text).is_absolute():
        raise W2LSPSAAdapterError("shared output WAV path must be absolute")
    output_path = Path(output_path_text)
    output_bytes = _read_regular(output_path, max_bytes=8 * 1024 * 1024)
    if (len(output_bytes) != output.get("bytes") or _sha256(output_bytes) != output.get("sha256")
            or input_ref.get("path") != str(output_path.resolve(strict=True))
            or input_ref.get("bytes") != output.get("bytes")
            or input_ref.get("sha256") != output.get("sha256")
            or input_ref.get("file_sha256") != output.get("file_sha256")
            or input_ref.get("pcm_sha256") != output.get("pcm_float32_sha256")
            or input_ref.get("sample_count") != output.get("sample_count")
            or input_ref.get("sample_rate_hz") != 16000 or input_ref.get("channels") != 1
            or input_ref.get("encoding") != "FLOAT_32"
            or output.get("sample_rate_hz") != 16000 or output.get("channels") != 1
            or output.get("encoding") != "FLOAT_32"
            or output.get("sample_count") != natural_ref.get("sample_count")):
        raise W2LSPSAAdapterError("candidate generator input differs from the actual shared output file pin")
    decoded_path, decoded_samples, decoded_ref = _read_audio_pin(
        core, input_ref, "shared candidate generator waveform")
    if (decoded_path.resolve(strict=True) != output_path.resolve(strict=True)
            or decoded_ref.get("file_sha256") != output.get("file_sha256")
            or decoded_ref.get("pcm_sha256") != output.get("pcm_float32_sha256")
            or decoded_samples.size != natural_ref.get("sample_count")
            or decoded_ref.get("sample_rate_hz") != 16000
            or decoded_ref.get("channels") != 1):
        raise W2LSPSAAdapterError("decoded shared waveform differs from the exact pinned original-PCM clock")
    origin_pin = shared.get("origin_v05_application_receipt")
    if not isinstance(origin_pin, Mapping):
        raise W2LSPSAAdapterError("shared receipt omits its original attempt05 application pin")
    origin, _origin_raw, _origin_actual = _read_json_pin(
        origin_pin, "attempt05 origin application receipt", self_field="self_sha256")
    if (origin.get("status") != "PASS_FIT_SNAPSHOT_GENERATOR_DRIVE_ONLY"
            or origin.get("schema") != "cross_tfg_fit_snapshot_generator_drive_result_v1"
            or origin.get("sample_id") != sample_id
            or origin.get("method_id") != method.get("method_id")
            or origin.get("method_config_sha256") != method.get("config_sha256")
            or origin.get("output_generator_drive_wav") != output
            or shared.get("origin_v05_self_sha256") != origin.get("self_sha256")):
        raise W2LSPSAAdapterError("shared receipt does not preserve the actual attempt05 origin output lineage")
    method_config = method.get("config", {})
    expected_method_binding = _method_binding(method_config, str(method.get("config_sha256")))
    projection_pin_base = {key: projection_pin[key] for key in ("path", "bytes", "sha256", "self_sha256")}
    if (projection.get("shared_application_receipt") != {key: shared_pin[key] for key in
            ("path", "bytes", "sha256", "self_sha256")}
            or projection.get("family_plan") != {key: family_pin[key] for key in
                ("path", "bytes", "sha256", "self_sha256")}
            or projection.get("point_plan") != {key: point_pin[key] for key in
                ("path", "bytes", "sha256", "self_sha256")}
            or projection.get("model_id") != W2L_MODEL_ID
            or projection.get("parent_row_id") != parent.get("row_id")
            or projection.get("point_row_id") != point_row.get("row_id")
            or projection.get("sample_id") != sample_id
            or projection.get("speaker_id") != parent.get("speaker_id")
            or projection.get("renderer_seed") != parent.get("seed")
            or projection.get("method_binding") != dict(expected_method_binding, **{
                "parent_row_id": parent.get("row_id"), "sample_id": parent.get("sample_id"),
                "speaker_id": parent.get("speaker_id"), "renderer_seed": parent.get("seed"),
                "model_id": W2L_MODEL_ID,
                "shared_application_receipt_raw_sha256": shared_pin.get("sha256"),
                "shared_application_receipt_self_sha256": shared_pin.get("self_sha256")})
            or projection.get("model_context") != _model_context(family_plan, W2L_MODEL_ID)
            or projection.get("source_natural_pcm") != shared.get("source_natural_pcm")
            or projection.get("shared_output_generator_drive_wav") != output
            or projection.get("score_audio") != parent.get("natural_pcm")
            or projection.get("render_or_score_performed") is not False
            or projection.get("optimizer_created_or_stepped") is not False):
        raise W2LSPSAAdapterError("model projection receipt differs from the exact W2L parent/point/output")
    if (binding.get("shared_application_receipt_raw_sha256") != shared_pin.get("sha256")
            or binding.get("shared_application_receipt_self_sha256") != shared_pin.get("self_sha256")
            or binding.get("model_projection_receipt_raw_sha256") != projection_pin.get("sha256")
            or binding.get("model_projection_receipt_self_sha256") != projection_pin.get("self_sha256")
            or binding.get("model_id") != W2L_MODEL_ID
            or binding.get("parent_row_id") != parent.get("row_id")
            or binding.get("sample_id") != sample_id
            or binding.get("binding_sha256") != method_config.get("binding_sha256")
            or binding.get("theta_sha256") != method_config.get("theta_sha256")
            or input_ref.get("shared_application_receipt") != dict(shared_pin)
            or input_ref.get("model_projection_receipt") != dict(projection_pin)
            or input_ref.get("shared_application_receipt_raw_sha256") != shared_pin.get("sha256")
            or input_ref.get("shared_application_receipt_self_sha256") != shared_pin.get("self_sha256")
            or input_ref.get("model_projection_receipt_raw_sha256") != projection_pin.get("sha256")
            or input_ref.get("model_projection_receipt_self_sha256") != projection_pin.get("self_sha256")
            or candidate.get("score_audio") != parent.get("natural_pcm")):
        raise W2LSPSAAdapterError("candidate row bindings do not match the validated receipt chain")
    return {"shared_application_receipt": dict(shared_pin),
            "model_projection_receipt": dict(projection_pin),
            "shared_output_generator_drive_wav": dict(output),
            "origin_v05_application_receipt": dict(origin_pin)}


class W2LSPSAStageAAdapter:
    """Render W2L rows from one validated 8-clip proposal or 40-clip center."""

    def __init__(self, *, core: Any, family_plan: Mapping[str, Any], family_plan_pin: Mapping[str, Any],
                 point_plan: Mapping[str, Any], point_plan_pin: Mapping[str, Any], args: Any):
        self.core = core
        self.args = args
        self.family_plan = dict(family_plan)
        self.point_plan = dict(point_plan)
        self.family_pin = _plan_pin(self.family_plan, family_plan_pin, "frozen 720-row family plan")
        self.point_pin = _plan_pin(self.point_plan, point_plan_pin, "single-method SPSA point plan")
        if (self.family_plan.get("execution", {}).get("planned_rows") != 720
                or self.family_plan.get("execution", {}).get("model_ids") != list(MODEL_ORDER)
                or self.family_plan.get("execution", {}).get("candidate_selection") != "not performed"
                or self.family_plan.get("execution", {}).get("dev_or_confirm_access") is not False
                or self.family_plan.get("generation_repeat_qc") != []):
            raise W2LSPSAAdapterError("family plan does not preserve its complete Fit-only 720-row scope")
        if len(self.family_plan.get("methods", [])) != 6:
            raise W2LSPSAAdapterError("family plan does not contain its six frozen source methods")
        self.method = _load_point_method(self.point_plan)
        self.point_method_config = dict(self.method["config"])
        if (self.point_plan.get("execution", {}).get("planned_rows") != len(self.point_plan.get("rows", []))
                or self.point_plan.get("generation_repeat_qc") != []
                or self.point_plan.get("scientific_input_pins_sha256") !=
                   self.family_plan.get("scientific_input_pins_sha256")):
            raise W2LSPSAAdapterError("point plan changed its frozen parent input identity or row scope")
        self.identity_by_context = _identity_rows(self.family_plan)
        self.source_project_root = Path(str(getattr(args, "project_root", "")))
        if (not self.source_project_root.is_absolute() or self.source_project_root.is_symlink()
                or not self.source_project_root.is_dir()):
            raise W2LSPSAAdapterError("W2L args.project_root must be the exact staged source project root")
        self.source_project_root = self.source_project_root.resolve(strict=True)
        (self.w2l_plan, self.w2l_plan_raw, self.w2l_plan_path, self.w2l_rows,
         self.w2l_rows_raw, self.w2l_rows_path, self.model) = _load_w2l_source(
            self.family_plan, self.source_project_root, self.core)
        self.runtime_ref = self.family_plan.get("execution", {}).get("runtime_bindings_by_model", {}).get(W2L_MODEL_ID)
        if not isinstance(self.runtime_ref, Mapping):
            raise W2LSPSAAdapterError("family plan omits the exact W2L runtime binding reference")
        self.runtime_path = Path(str(self.runtime_ref.get("path", "")))
        self.runtime_raw = _read_regular(self.runtime_path, max_bytes=8 * 1024 * 1024)
        if (("bytes" in self.runtime_ref and len(self.runtime_raw) != self.runtime_ref.get("bytes"))
                or _sha256(self.runtime_raw) != self.runtime_ref.get("raw_sha256")):
            raise W2LSPSAAdapterError("W2L runtime binding file differs from the family-plan pin")
        self.runtime_sha256 = _sha256(self.runtime_raw)
        self.model_order = tuple(MODEL_ORDER)

    def validation_source_pins(self) -> Dict[str, Dict[str, Any]]:
        """Return the exact code and plan pins used by this W2L worker."""
        core_path = Path(str(self.core.__file__)).resolve(strict=True)
        core_raw = _read_regular(core_path, max_bytes=2 * 1024 * 1024)
        return {
            "stagea_runner": {"path": str(core_path), "bytes": len(core_raw), "sha256": _sha256(core_raw)},
            "family_plan": dict(self.family_pin),
            "point_plan": dict(self.point_pin),
            "w2l_source_plan": {"path": str(self.w2l_plan_path), "bytes": len(self.w2l_plan_raw),
                                "sha256": _sha256(self.w2l_plan_raw),
                                "self_sha256": self.w2l_plan.get("plan_sha256")},
            "w2l_source_rows": {"path": str(self.w2l_rows_path), "bytes": len(self.w2l_rows_raw),
                                "sha256": _sha256(self.w2l_rows_raw)},
                "w2l_runtime_binding": {"path": str(self.runtime_path.resolve(strict=True)),
                                    "bytes": len(self.runtime_raw), "sha256": self.runtime_sha256,
                                    "declared_bytes": self.runtime_ref.get("bytes")},
        }

    def render_subset(self, *, candidate_rows: Sequence[Mapping[str, Any]],
                      output_root: str | Path) -> List[Dict[str, Any]]:
        """Render and score the ordered W2L portion of one SPSA point plan."""
        if not isinstance(candidate_rows, Sequence) or isinstance(candidate_rows, (str, bytes)):
            raise W2LSPSAAdapterError("candidate rows must be an ordered sequence")
        expected_point_rows, _pins = _validate_point_and_candidate(
            self.family_plan, self.family_pin, self.point_plan, self.point_pin,
            self.method, self.identity_by_context, candidate_rows)
        if len(candidate_rows) not in (8, 40):
            raise W2LSPSAAdapterError("W2L subset must contain exactly eight proposal or forty center rows")
        output = Path(output_root)
        if (not output.is_absolute() or output.is_symlink() or not output.is_dir()
                or not output.resolve(strict=True).is_dir()):
            raise W2LSPSAAdapterError("W2L subset output root must be an existing real directory")
        output = output.resolve(strict=True)
        if any(output.iterdir()):
            raise W2LSPSAAdapterError("W2L subset output root must be fresh and empty")
        row_ids = [str(row.get("row_id")) for row in candidate_rows]
        if (len(set(row_ids)) != len(row_ids)
                or any(not value or Path(value).name != value or value in {".", ".."}
                       or (output / value).exists() or (output / value).is_symlink() for value in row_ids)):
            raise W2LSPSAAdapterError("W2L candidate row output is unsafe or already exists")
        row_validations = []
        for row, point_row in zip(candidate_rows, expected_point_rows):
            row_validations.append(_validate_shared_projection(
                self.core, self.source_project_root, self.family_plan, self.family_pin,
                self.point_plan, self.point_pin,
                self.method, self.identity_by_context, row, point_row))
        if row_validations and len({item["shared_output_generator_drive_wav"]["path"]
                                    for item in row_validations}) != len(row_validations):
            raise W2LSPSAAdapterError("each sample must have one distinct shared output WAV path")

        saved_env = {}
        touched = ("CROSS_TFG_CAPACITY_STAGE", "CROSS_TFG_RUNTIME_BINDING_PATH",
                   "CROSS_TFG_RUNTIME_BINDING_RAW_SHA256")
        for key in touched:
            saved_env[key] = os.environ.get(key)
        os.environ["CROSS_TFG_CAPACITY_STAGE"] = "fit40_w2l240"
        os.environ["CROSS_TFG_RUNTIME_BINDING_PATH"] = str(self.runtime_path.resolve(strict=True))
        os.environ["CROSS_TFG_RUNTIME_BINDING_RAW_SHA256"] = self.runtime_sha256
        completed = []
        result_rows = []
        try:
            for index, (row, point_row, receipts) in enumerate(
                    zip(candidate_rows, expected_point_rows, row_validations)):
                row_id = str(row["row_id"])
                result = None
                try:
                    gate = self.core._row_resource_gate(self.w2l_plan, row, self.args, output)
                    if not isinstance(gate, Mapping) or gate.get("status") != "PASS":
                        raise W2LSPSAAdapterError("live W2L per-row capacity gate did not pass")
                    natural_path, natural_samples, natural_ref = _read_audio_pin(
                        self.core, row["natural_pcm"], "original natural PCM")
                    marker_method = copy.deepcopy(dict(self.method))
                    marker_config = copy.deepcopy(dict(self.point_method_config))
                    marker_config["method"] = "precomputed_fit_snapshot_waveform"
                    marker_method["config"] = marker_config
                    row_root = output / row_id
                    result = self.core._render_and_score(
                        row, self.model, marker_method, natural_path, natural_samples,
                        natural_ref, row_root, self.args, qc_repeat=False)
                    if (not isinstance(result, dict) or result.get("status") != "PASS"
                            or result.get("score_audio_sha256") != row["natural_pcm"].get("pcm_sha256")
                            or result.get("generator_pcm_sha256") !=
                               row["generator_condition_input"].get("pcm_sha256")):
                        raise W2LSPSAAdapterError("frozen W2L row result differs from shared-input/natural-score pins")
                    if not isinstance(result.get("_internal_arrays"), Mapping):
                        raise W2LSPSAAdapterError("frozen W2L row omitted its actual official score arrays")
                    array_qc = _validate_official_arrays(result["_internal_arrays"], self.core)
                    delivery = result.get("mux_audio_readback")
                    score_audio_sha = row["natural_pcm"].get("pcm_sha256")
                    if (not isinstance(delivery, Mapping) or delivery.get("status") != "PASS"
                            or delivery.get("decoded_float32_pcm_sha256") != score_audio_sha
                            or delivery.get("sample_count") != row["natural_pcm"].get("sample_count")
                            or result.get("natural_pcm_sha256") != score_audio_sha
                            or result.get("score_audio_sha256") != score_audio_sha):
                        raise W2LSPSAAdapterError("native video delivery did not preserve full original F32 PCM")
                    result_rows.append(result)
                    completed.append({
                        "candidate_row": row,
                        "row_output_root": str(row_root.resolve(strict=True)),
                        "parent_identity_row_id": self.identity_by_context[
                            (str(row.get("sample_id")), W2L_MODEL_ID)]["row_id"],
                        "receipts": receipts,
                        "resource_gate": dict(gate),
                        "official_array_qc": array_qc,
                    })
                except Exception as exc:
                    raise W2LSPSASubsetFailure(
                        "W2L SPSA row " + row_id + " failed: " + type(exc).__name__ + ": " + str(exc),
                        completed=completed,
                        failed_row_id=row_id,
                        unattempted_row_ids=row_ids[index + 1:],
                        failed_result=result,
                    ) from exc
            block_plan = copy.deepcopy(self.point_plan)
            block_plan["methods"] = [copy.deepcopy(dict(self.method))]
            block_plan["rows"] = [copy.deepcopy(dict(row)) for row in candidate_rows]
            self.core._finish_supports(block_plan, result_rows)
            for result, record in zip(result_rows, completed):
                row = record["candidate_row"]
                public = {key: value for key, value in result.items() if not key.startswith("_internal_")}
                public["spsa_binding"] = copy.deepcopy(dict(row["spsa_binding"]))
                public["spsa_point_binding_sha256"] = row["spsa_point_binding_sha256"]
                public["generator_condition_input"] = copy.deepcopy(dict(row["generator_condition_input"]))
                public["parent_identity_row_id"] = record["parent_identity_row_id"]
                record["row"] = public
                record["official_arrays"] = result["_internal_arrays"]
                record["support_scope"] = "single-W2L-arm descriptive only; not joint common support or admission"
                record["shared_application_receipt"] = record["receipts"]["shared_application_receipt"]
                record["model_projection_receipt"] = record["receipts"]["model_projection_receipt"]
                record["origin_v05_application_receipt"] = record["receipts"]["origin_v05_application_receipt"]
                record["validation_source_pins"] = self.validation_source_pins()
                record.pop("candidate_row", None)
                record.pop("receipts", None)
            return completed
        finally:
            for key in touched:
                previous = saved_env[key]
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous


__all__ = ["W2LSPSAAdapterError", "W2LSPSASubsetFailure", "W2LSPSAStageAAdapter"]
