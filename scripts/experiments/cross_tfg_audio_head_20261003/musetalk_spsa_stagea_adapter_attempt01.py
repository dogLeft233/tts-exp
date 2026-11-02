"""Muse-only Stage-A execution hooks for Fit SPSA proposals and centers.

This module does not construct or transform a head, select a candidate, or
start a model by itself.  The caller supplies the already-verified D0
generator-drive waveform and application receipt for each row.  We validate
that lineage, then call the frozen Stage-A ``_render_and_score`` function
directly with Muse's verified per-model protocol and the unmodified original
natural PCM as the scoring/delivery audio.

The Stage-A method dispatcher currently recognizes precomputed snapshot WAVs
by the config marker ``precomputed_fit_snapshot_waveform``.  SPSA plan methods
retain their frozen ``fit40_spsa_proposal`` / ``fit40_spsa_center`` configs;
only a private copy passed to that function receives the dispatcher marker.
The serialized plan, method config, and row are never rewritten.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


MUSE_MODEL_ID = "musetalk_fixed_portrait_attempt16"
EXPECTED_PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
EXPECTED_SCIENTIFIC_INPUT_PINS_SHA256 = "ae96def4c16fecabc6e49b2aeabea0a76a0308ea159fa6c7688f7908c62c7180"
EXPECTED_CAPACITY_STAGE = "fit40_musetalk240"
SUPPORTED_METHODS = {"fit40_spsa_proposal", "fit40_spsa_center"}
_PATCH_LOCK = threading.Lock()


class MuseSPSAAdapterError(RuntimeError):
    """A proposal, binding, or runtime input does not meet the Stage-A contract."""


class MuseSPSASubsetFailure(MuseSPSAAdapterError):
    """A render failed; carries completed and unattempted rows without replaying."""

    def __init__(self, message: str, *, completed: Sequence[Mapping[str, Any]],
                 failed_row_id: str, unattempted_row_ids: Sequence[str]):
        super().__init__(message)
        self.completed = list(completed)
        self.failed_row_id = failed_row_id
        self.unattempted_row_ids = list(unattempted_row_ids)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_native_frame_count(sample_count: int, fps_num: int = 25,
                                fps_den: int = 1, sample_rate_hz: int = 16000) -> int:
    """Return the frozen floor(N*fps/sample_rate) native frame count."""
    if (type(sample_count) is not int or sample_count <= 0
            or type(fps_num) is not int or fps_num <= 0
            or type(fps_den) is not int or fps_den <= 0
            or type(sample_rate_hz) is not int or sample_rate_hz <= 0):
        raise MuseSPSAAdapterError("sample count, frame rate, and sample rate must be positive integers")
    return (sample_count * fps_num) // (sample_rate_hz * fps_den)


def _require_sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise MuseSPSAAdapterError(f"{label} must be a lowercase SHA-256")
    return value


def _self_hash_matches(value: Mapping[str, Any], field: str) -> bool:
    expected = value.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    payload = {key: item for key, item in value.items() if key != field}
    return _sha256_bytes(_canonical_bytes(payload)) == expected


def _safe_regular_file(pin: Mapping[str, Any], label: str,
                       *, sha_field: str = "sha256") -> tuple[Path, bytes]:
    path_text = pin.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise MuseSPSAAdapterError(f"{label} path is absent")
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MuseSPSAAdapterError(f"{label} must be an absolute, non-symlink regular file")
    raw = path.read_bytes()
    expected_bytes = pin.get("bytes")
    expected_sha = pin.get(sha_field)
    if (type(expected_bytes) is not int or expected_bytes <= 0 or len(raw) != expected_bytes
            or _sha256_bytes(raw) != expected_sha):
        raise MuseSPSAAdapterError(f"{label} bytes or raw SHA-256 differ from the supplied pin")
    return path.resolve(strict=True), raw


def _method_config(method: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    config = method.get("config")
    if not isinstance(config, Mapping):
        raise MuseSPSAAdapterError("SPSA method lacks a config object")
    method_kind = config.get("method")
    if method_kind not in SUPPORTED_METHODS:
        raise MuseSPSAAdapterError("only frozen SPSA proposal or center methods may be rendered")
    config_sha = _require_sha(method.get("config_sha256"), "method.config_sha256")
    if _sha256_bytes(_canonical_bytes(config)) != config_sha:
        raise MuseSPSAAdapterError("method config raw canonical SHA differs from its declared pin")
    if method.get("method_id") != config.get("method_id"):
        raise MuseSPSAAdapterError("method ID differs between the method and its config")
    if method.get("kind") != method_kind:
        raise MuseSPSAAdapterError("method kind differs between the method and its config")
    if method.get("checkpoint_sha256") is not None:
        raise MuseSPSAAdapterError("SPSA waveform methods must not bind an alternate checkpoint")
    theta = config.get("theta")
    if (not isinstance(theta, (list, tuple)) or len(theta) != 9
            or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in theta)):
        raise MuseSPSAAdapterError("SPSA method must contain nine finite theta values")
    theta_sha = _require_sha(config.get("theta_sha256"), "method.config.theta_sha256")
    if _sha256_bytes(_canonical_bytes(list(theta))) != theta_sha:
        raise MuseSPSAAdapterError("theta canonical SHA differs from the method config")
    for name in ("artifact_sha256", "binding_sha256", "input_pins_sha256"):
        _require_sha(config.get(name), f"method.config.{name}")
    return config, config_sha


def _resolve_muse_protocol(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    models = plan.get("models")
    if not isinstance(models, list):
        raise MuseSPSAAdapterError("parent plan has no model descriptors")
    matches = [model for model in models if isinstance(model, Mapping)
               and model.get("model_id") == MUSE_MODEL_ID]
    if len(matches) != 1:
        raise MuseSPSAAdapterError("parent plan must contain exactly one MuseTalk descriptor")
    model = matches[0]
    execution = plan.get("execution", {})
    protocols = execution.get("source_model_protocols", {}) if isinstance(execution, Mapping) else {}
    protocol = protocols.get(MUSE_MODEL_ID) if isinstance(protocols, Mapping) else None
    if protocol is None:
        protocol = plan.get("protocol")
    if not isinstance(protocol, Mapping) or not isinstance(protocol.get("generator"), Mapping):
        raise MuseSPSAAdapterError("Muse's full per-model generator protocol is absent")
    generator = protocol["generator"]

    # A composed family plan carries per-model source protocols.  Require its
    # descriptor and source protocol to agree, then pass that exact model's
    # generator to the Muse wrapper rather than a composed/common generator.
    descriptor_generator = model.get("generator")
    if not isinstance(descriptor_generator, Mapping) or dict(descriptor_generator) != dict(generator):
        raise MuseSPSAAdapterError("Muse descriptor differs from its exact source_model_protocols generator")
    generators_by_model = plan.get("protocol", {}).get("generators_by_model", {})
    if isinstance(generators_by_model, Mapping) and generators_by_model:
        if dict(generators_by_model.get(MUSE_MODEL_ID, {})) != dict(generator):
            raise MuseSPSAAdapterError("composed generators_by_model Muse entry differs from source protocol")
    return model, generator


def _plan_binding_reference(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = plan.get("execution", {})
    if not isinstance(execution, Mapping):
        raise MuseSPSAAdapterError("parent plan execution block is malformed")
    direct = execution.get("muse_runtime_binding")
    if isinstance(direct, Mapping):
        return direct
    by_model = execution.get("runtime_bindings_by_model")
    if isinstance(by_model, Mapping) and isinstance(by_model.get(MUSE_MODEL_ID), Mapping):
        return by_model[MUSE_MODEL_ID]
    model_execution = execution.get("source_model_execution")
    if isinstance(model_execution, Mapping):
        source = model_execution.get(MUSE_MODEL_ID)
        if isinstance(source, Mapping):
            direct = source.get("muse_runtime_binding")
            if isinstance(direct, Mapping):
                return direct
    raise MuseSPSAAdapterError("parent plan does not bind the Muse runtime binding")


def _muse_source_execution(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = plan.get("execution", {})
    if not isinstance(execution, Mapping):
        raise MuseSPSAAdapterError("parent plan execution block is malformed")
    source_model_execution = execution.get("source_model_execution")
    if isinstance(source_model_execution, Mapping):
        value = source_model_execution.get(MUSE_MODEL_ID)
        if isinstance(value, Mapping):
            return value
    return execution


def _verify_module_pin(module: Any, ref: Mapping[str, Any], label: str) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not isinstance(ref.get("path"), str):
        raise MuseSPSAAdapterError(f"{label} has no concrete source path")
    path = Path(module_file)
    if path.is_symlink() or not path.is_file():
        raise MuseSPSAAdapterError(f"{label} source path is absent or symlinked")
    resolved = path.resolve(strict=True)
    if str(resolved) != ref.get("path"):
        raise MuseSPSAAdapterError(f"{label} loaded from a path other than the frozen plan")
    raw = resolved.read_bytes()
    expected_bytes = ref.get("bytes")
    expected_sha = ref.get("sha256")
    if type(expected_bytes) is not int or len(raw) != expected_bytes or _sha256_bytes(raw) != expected_sha:
        raise MuseSPSAAdapterError(f"{label} source bytes/SHA differ from the frozen plan")


def _validate_official_arrays(arrays: Mapping[str, Any]) -> dict[str, int]:
    expected = {
        "query_frame_index": ("<i4", 1),
        "query_timestamp_num": ("<i8", 1),
        "query_timestamp_den": ("<i8", 1),
        "track_frame_index": ("<i4", 1),
        "valid_query_mask": ("|b1", 1),
        "distance_by_query_lag": ("<f4", 2),
        "lag_offsets_frames": ("<i4", 1),
    }
    if set(arrays) != set(expected):
        raise MuseSPSAAdapterError("Stage-A returned an unexpected official NPZ array set")
    shapes: dict[str, tuple[int, ...]] = {}
    for key, (dtype, ndim) in expected.items():
        array = arrays[key]
        shape = getattr(array, "shape", None)
        actual_dtype = getattr(getattr(array, "dtype", None), "str", None)
        if (not isinstance(shape, tuple) or len(shape) != ndim or actual_dtype != dtype
                or any(type(dim) is not int or dim < 0 for dim in shape)):
            raise MuseSPSAAdapterError(f"official Stage-A array {key} has a noncanonical shape or dtype")
        shapes[key] = shape
    query_count = shapes["query_frame_index"][0]
    if query_count <= 0 or shapes["distance_by_query_lag"] != (query_count, 31):
        raise MuseSPSAAdapterError("official Stage-A score matrix must be [N,31] for its exact query clock")
    for key in ("query_timestamp_num", "query_timestamp_den", "track_frame_index", "valid_query_mask"):
        if shapes[key] != (query_count,):
            raise MuseSPSAAdapterError(f"official Stage-A query array {key} differs from the producer query count")
    if shapes["lag_offsets_frames"] != (31,):
        raise MuseSPSAAdapterError("official Stage-A lag vector must contain 31 offsets")
    lag_values = [int(value) for value in arrays["lag_offsets_frames"].tolist()]
    if lag_values != list(range(-15, 16)):
        raise MuseSPSAAdapterError("official Stage-A lag vector differs from the frozen -15..15 frame grid")
    numerators = arrays["query_timestamp_num"].tolist()
    denominators = arrays["query_timestamp_den"].tolist()
    if any(int(den) <= 0 for den in denominators):
        raise MuseSPSAAdapterError("official Stage-A timestamp denominator must be positive")
    keys = []
    for frame, numerator, denominator, track in zip(
            arrays["query_frame_index"].tolist(), numerators, denominators,
            arrays["track_frame_index"].tolist()):
        timestamp = Fraction(int(numerator), int(denominator))
        keys.append((int(frame), timestamp.numerator, timestamp.denominator, int(track)))
    if len(set(keys)) != query_count:
        raise MuseSPSAAdapterError("official Stage-A query clock contains duplicate normalized keys")
    matrix = arrays["distance_by_query_lag"]
    if not all(math.isfinite(float(value)) for value in matrix.flat):
        raise MuseSPSAAdapterError("official Stage-A distance matrix contains nonfinite values")
    mask = arrays["valid_query_mask"].tolist()
    if any(type(value) not in (bool, int) or int(value) not in (0, 1) for value in mask):
        raise MuseSPSAAdapterError("official Stage-A valid-query mask is not boolean")
    return {"query_count": query_count, "valid_query_count": sum(bool(value) for value in mask)}


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in
                 ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed"))


_ROW_MUTABLE_FIELDS = {
    "row_id", "method_id", "method_config_sha256", "checkpoint_sha256",
    "generator_condition_input", "spsa_binding", "spsa_point_binding_sha256",
}


def _stable_row_copy(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in _ROW_MUTABLE_FIELDS}


def _expected_sample_ids(plan: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, ...]:
    if config.get("method") == "fit40_spsa_proposal":
        sample_ids = config.get("sample_ids")
        if (not isinstance(sample_ids, list) or len(sample_ids) != 8
                or len(set(sample_ids)) != 8 or any(not isinstance(item, str) for item in sample_ids)):
            raise MuseSPSAAdapterError("proposal config must bind eight unique Fit sample IDs")
        return tuple(sample_ids)
    scope = plan.get("execution", {}).get("spsa_execution_scope", {})
    sample_ids = scope.get("sample_ids") if isinstance(scope, Mapping) else None
    if sample_ids is None:
        sample_ids = list(dict.fromkeys(
            str(row.get("sample_id")) for row in plan.get("rows", [])
            if row.get("model_id") == MUSE_MODEL_ID and row.get("method_id") == "identity"
        ))
    if (not isinstance(sample_ids, (list, tuple)) or len(sample_ids) != 40
            or len(set(sample_ids)) != 40 or any(not isinstance(item, str) for item in sample_ids)):
        raise MuseSPSAAdapterError("center plan must bind the full ordered 40-clip Fit roster")
    return tuple(sample_ids)


def _validate_plan_and_binding(plan: Mapping[str, Any], binding: Mapping[str, Any],
                               binding_raw: bytes, binding_path: Path) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    if plan.get("schema_version") != EXPECTED_PLAN_SCHEMA or plan.get("status") != "FROZEN":
        raise MuseSPSAAdapterError("parent plan is not a frozen Stage-A plan")
    if not _self_hash_matches(plan, "plan_sha256"):
        raise MuseSPSAAdapterError("parent plan self hash is invalid")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise MuseSPSAAdapterError("parent plan execution block is malformed")
    if execution.get("dev_or_confirm_access") is not False or execution.get("candidate_selection") != "not performed":
        raise MuseSPSAAdapterError("parent plan is not closed to Dev/Confirm or candidate selection")
    plan_pins = plan.get("scientific_input_pins_sha256")
    if plan_pins != EXPECTED_SCIENTIFIC_INPUT_PINS_SHA256:
        raise MuseSPSAAdapterError("parent plan scientific input pin differs from the frozen Fit roster")
    ref = _plan_binding_reference(plan)
    expected_binding_run_id = plan.get("run_id")
    source_plans = execution.get("source_model_plans")
    if isinstance(source_plans, list):
        muse_source = [item for item in source_plans if isinstance(item, Mapping)
                       and item.get("model_id") == MUSE_MODEL_ID]
        if len(muse_source) != 1 or not isinstance(muse_source[0].get("plan_run_id"), str):
            raise MuseSPSAAdapterError("composed plan does not retain one exact Muse source plan run ID")
        expected_binding_run_id = muse_source[0]["plan_run_id"]
    if (ref.get("path") != str(binding_path) or ref.get("raw_sha256") != _sha256_bytes(binding_raw)
            or ref.get("bytes") != len(binding_raw)):
        raise MuseSPSAAdapterError("parent plan does not bind the supplied Muse runtime binding bytes/path")
    if (binding.get("schema") != "cross_tfg_fit40_musetalk_runtime_binding_v1"
            or binding.get("status") != "PASS_ACTUAL_V100_MUSETALK_ROW_EVIDENCE_AND_MEASURED_ESTIMATE_WITH_LIVE_GATES"
            or binding.get("model_id") != MUSE_MODEL_ID
            or binding.get("run_id") != expected_binding_run_id
            or binding.get("scientific_input_pins_sha256") != plan_pins
            or not _self_hash_matches(binding, "binding_sha256")
            or _sha256_bytes(binding_raw) != ref.get("raw_sha256")):
        raise MuseSPSAAdapterError("Muse runtime binding does not match the frozen plan or its self/raw hashes")

    models = plan.get("models")
    methods = plan.get("methods")
    rows = plan.get("rows")
    if not isinstance(models, list) or not isinstance(methods, list) or not isinstance(rows, list):
        raise MuseSPSAAdapterError("parent plan models, methods, or rows are malformed")
    if len(methods) != 6 or len(rows) not in (240, 720):
        raise MuseSPSAAdapterError("parent plan must be a complete 240-row per-model or 720-row family plan")
    _resolve_muse_protocol(plan)
    muse_rows = [row for row in rows if isinstance(row, Mapping) and row.get("model_id") == MUSE_MODEL_ID]
    if len(muse_rows) != 240:
        raise MuseSPSAAdapterError("parent plan must contain exactly 240 Muse rows")
    expected_methods = [item.get("method_id") for item in methods if isinstance(item, Mapping)]
    if len(expected_methods) != 6 or len(set(expected_methods)) != 6:
        raise MuseSPSAAdapterError("parent plan method roster must contain six unique methods")
    identity_rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    roster_keys: set[tuple[str, str]] = set()
    per_method: dict[str, int] = {str(method_id): 0 for method_id in expected_methods}
    sample_speaker: dict[str, str] = {}
    speaker_samples: dict[str, set[str]] = {}
    for row in muse_rows:
        if row.get("split") != "fit" or row.get("score_audio") != row.get("natural_pcm"):
            raise MuseSPSAAdapterError("Muse parent rows must be Fit-only and score the original natural PCM")
        context_method = (str(row.get("sample_id")), str(row.get("method_id")))
        if context_method in roster_keys or row.get("method_id") not in per_method:
            raise MuseSPSAAdapterError("Muse parent plan contains duplicate or undeclared method/context rows")
        roster_keys.add(context_method)
        per_method[str(row["method_id"])] += 1
        if row.get("method_id") == "identity":
            key = _identity_key(row)
            if key in identity_rows:
                raise MuseSPSAAdapterError("parent plan repeats a Muse identity context")
            identity_rows[key] = row
            sample = str(row.get("sample_id"))
            speaker = str(row.get("speaker_id"))
            if sample in sample_speaker and sample_speaker[sample] != speaker:
                raise MuseSPSAAdapterError("parent sample maps to more than one speaker")
            sample_speaker[sample] = speaker
            speaker_samples.setdefault(speaker, set()).add(sample)
    if (len(roster_keys) != 240 or set(per_method.values()) != {40}
            or len(identity_rows) != 40 or len(sample_speaker) != 40 or len(speaker_samples) != 20
            or any(len(samples) != 2 for samples in speaker_samples.values())):
        raise MuseSPSAAdapterError("Muse identity rows do not preserve the frozen 40-clip/20-speaker Fit roster")
    return identity_rows


def _application_receipt(binding: Mapping[str, Any], input_ref: Mapping[str, Any],
                         natural_ref: Mapping[str, Any]) -> dict[str, Any]:
    pin = binding.get("application_receipt")
    if not isinstance(pin, Mapping):
        raise MuseSPSAAdapterError("SPSA row binding lacks its D0 application receipt pin")
    path, raw = _safe_regular_file(pin, "D0 application receipt")
    packet = json.loads(raw.decode("utf-8"))
    if (not isinstance(packet, dict) or not _self_hash_matches(packet, "self_sha256")
            or pin.get("self_sha256") != packet.get("self_sha256")):
        raise MuseSPSAAdapterError("D0 application receipt is not a valid self-hashed JSON object")
    if (packet.get("schema") != "cross_tfg_d0_gatedc_generator_drive_result_v1"
            or packet.get("status") != "PASS_D0_GENERATOR_DRIVE_ONLY"
            or packet.get("render_or_score_performed") is not False
            or packet.get("optimizer_created_or_stepped") is not False
            or packet.get("input_original_file", {}).get("sha256") != natural_ref.get("file_sha256")
            or packet.get("canonical_generator_input", {}).get("pcm_float32_sha256") != natural_ref.get("pcm_sha256")
            or packet.get("source_clock_preserved_after_canonicalization") is not True):
        raise MuseSPSAAdapterError("D0 application receipt does not bind this source PCM and generator-drive-only scope")
    _require_sha(binding.get("checkpoint_parameter_sha256"), "spsa_binding.checkpoint_parameter_sha256")
    if (binding.get("gate_mode") not in {"d0", "gate_dc"}
            or packet.get("checkpoint_parameter_sha256") != binding.get("checkpoint_parameter_sha256")
            or packet.get("gate_mode") != binding.get("gate_mode")):
        raise MuseSPSAAdapterError("D0 application receipt parameter or gate mode differs from the proposal binding")
    output_pin = packet.get("output_generator_drive_wav")
    if not isinstance(output_pin, Mapping):
        raise MuseSPSAAdapterError("D0 application receipt omits its output generator-drive WAV pin")
    if (input_ref.get("path") != output_pin.get("path")
            or input_ref.get("bytes") != output_pin.get("bytes")
            or input_ref.get("file_sha256") != output_pin.get("sha256")
            or input_ref.get("pcm_sha256") != output_pin.get("pcm_float32_sha256")
            or input_ref.get("sample_count") != output_pin.get("sample_count")
            or input_ref.get("sample_rate_hz") != output_pin.get("sample_rate_hz")
            or input_ref.get("channels") != output_pin.get("channels")
            or input_ref.get("encoding", input_ref.get("subtype")) != output_pin.get("encoding")
            or packet.get("output_pcm_float32_sha256") != input_ref.get("pcm_sha256")
            or packet.get("output_sample_count") != input_ref.get("sample_count")):
        raise MuseSPSAAdapterError("candidate generator-condition WAV differs from the D0 receipt output pin")
    if (binding.get("application_receipt_raw_sha256") != _sha256_bytes(raw)
            or binding.get("application_receipt_self_sha256") != packet.get("self_sha256")
            or input_ref.get("application_receipt_raw_sha256") != _sha256_bytes(raw)
            or input_ref.get("application_receipt_self_sha256") != packet.get("self_sha256")):
        raise MuseSPSAAdapterError("SPSA binding/input does not carry the exact application receipt raw/self hashes")
    return {"path": str(path), "bytes": len(raw), "sha256": _sha256_bytes(raw),
            "self_sha256": packet["self_sha256"]}


def _validate_spsa_row_binding(row: Mapping[str, Any], parent: Mapping[str, Any],
                               method: Mapping[str, Any], config: Mapping[str, Any],
                               config_sha: str) -> dict[str, Any]:
    binding = row.get("spsa_binding")
    if not isinstance(binding, Mapping):
        raise MuseSPSAAdapterError("candidate row lacks explicit spsa_binding metadata")
    point_sha = _require_sha(row.get("spsa_point_binding_sha256"), "row.spsa_point_binding_sha256")
    if point_sha != config.get("binding_sha256"):
        raise MuseSPSAAdapterError("row point binding does not match the serialized SPSA method")
    if (row.get("method_id") != method.get("method_id")
            or row.get("method_config_sha256") != config_sha
            or binding.get("method_id") != method.get("method_id")
            or binding.get("method_config_sha256") != config_sha
            or binding.get("parent_row_id") != parent.get("row_id")
            or binding.get("seed") != row.get("seed")):
        raise MuseSPSAAdapterError("candidate row, parent identity, method config, or seed binding differs")
    for name in ("artifact_sha256", "binding_sha256", "input_pins_sha256", "theta_sha256"):
        if binding.get(name) != config.get(name):
            raise MuseSPSAAdapterError(f"candidate row spsa_binding differs from method config field {name}")
    if list(binding.get("theta", [])) != list(config.get("theta", [])):
        raise MuseSPSAAdapterError("candidate row theta values differ from the exact SPSA method config")
    method_kind = config["method"]
    if method_kind == "fit40_spsa_proposal":
        for key in ("proposal_id", "pair_id", "iteration", "side", "batch_id", "c_k"):
            if binding.get(key) != config.get(key):
                raise MuseSPSAAdapterError(f"proposal row binding differs from method config field {key}")
        if binding.get("side") not in {"plus", "minus"}:
            raise MuseSPSAAdapterError("proposal row side must be plus or minus")
    else:
        if (binding.get("point_kind") != "spsa_center"
                or binding.get("source_step") != config.get("source_step")
                or binding.get("source_step") not in {5, 10, 15, 20}):
            raise MuseSPSAAdapterError("center row binding does not identify a predeclared full-score step")
    if _stable_row_copy(row) != _stable_row_copy(parent):
        raise MuseSPSAAdapterError("candidate row changed frozen identity, natural PCM, face, or clock metadata")
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id or row_id == parent.get("row_id"):
        raise MuseSPSAAdapterError("candidate row must have a fresh row_id distinct from its identity parent")
    return dict(binding)


def _validate_generator_input(row: Mapping[str, Any], parent: Mapping[str, Any],
                              core: Any, binding: Mapping[str, Any],
                              config: Mapping[str, Any], config_sha: str) -> tuple[dict[str, Any], Any]:
    natural_ref = row.get("natural_pcm")
    if not isinstance(natural_ref, Mapping) or row.get("score_audio") != natural_ref:
        raise MuseSPSAAdapterError("candidate must preserve natural_pcm as its exact scoring audio")
    input_ref = row.get("generator_condition_input")
    if not isinstance(input_ref, Mapping):
        raise MuseSPSAAdapterError("candidate row lacks its generated full-length generator_condition_input")
    if _stable_row_copy(row) != _stable_row_copy(parent):
        raise MuseSPSAAdapterError("candidate row differs from its identity parent outside declared SPSA/input fields")
    input_path, input_raw = _safe_regular_file(
        input_ref, "candidate generator-condition WAV", sha_field="file_sha256"
    )
    if _sha256_bytes(input_raw) != input_ref.get("file_sha256"):
        raise MuseSPSAAdapterError("candidate generator-condition WAV raw SHA is invalid")
    from scripts.audio_enhancement.audio_io import read_canonical_wav

    samples, info = read_canonical_wav(input_path)
    expected_natural = natural_ref.get("sample_count")
    if (info.sample_rate != 16000 or info.source_sample_rate != 16000 or info.source_channels != 1
            or info.frames != expected_natural or samples.size != expected_natural
            or input_ref.get("sample_count") != expected_natural
            or input_ref.get("sample_rate_hz") != 16000 or input_ref.get("channels") != 1
            or input_ref.get("pcm_sha256") != info.pcm_float32_sha256):
        raise MuseSPSAAdapterError("candidate generator-drive PCM is not full-length mono 16 kHz on the natural clock")
    encoding = input_ref.get("encoding", input_ref.get("subtype"))
    if (info.source_encoding != "FLOAT_32"
            or encoding not in ("FLOAT_32", "IEEE_FLOAT_32_LE", "FLOAT32", "float32")):
        raise MuseSPSAAdapterError("candidate generator-drive WAV must declare its canonical float32 subtype")
    native = row.get("musetalk_native_frame_binding")
    if (not isinstance(native, Mapping) or native.get("sample_count") != expected_natural
            or native.get("expected_frame_count") != expected_native_frame_count(expected_natural)):
        raise MuseSPSAAdapterError("candidate row native frame count differs from floor(N*25/16000)")

    spsa_binding = row.get("spsa_binding")
    if not isinstance(spsa_binding, Mapping):
        raise MuseSPSAAdapterError("candidate row lacks its explicit SPSA binding")
    if (input_ref.get("proposal_id") != spsa_binding.get("proposal_id")
            or input_ref.get("pair_id") != spsa_binding.get("pair_id")
            or input_ref.get("side") != spsa_binding.get("side")
            or input_ref.get("method_id") != config.get("method_id")
            or input_ref.get("method_config_sha256") != config_sha
            or input_ref.get("artifact_sha256") != config.get("artifact_sha256")
            or input_ref.get("binding_sha256") != config.get("binding_sha256")
            or input_ref.get("input_pins_sha256") != config.get("input_pins_sha256")
            or input_ref.get("theta_sha256") != config.get("theta_sha256")):
        raise MuseSPSAAdapterError("generator-condition waveform receipt is not bound to this SPSA method point")
    if input_ref.get("parent_row_id") != parent.get("row_id"):
        raise MuseSPSAAdapterError("generator-condition waveform receipt points to a different identity parent")
    if _sha256_bytes(_canonical_bytes(list(config["theta"]))) != config.get("theta_sha256"):
        raise MuseSPSAAdapterError("input theta hash differs from the exact proposal theta")

    natural_path, natural_samples, natural_actual = core._reference(natural_ref)
    if (natural_actual.get("file_sha256") != natural_ref.get("file_sha256")
            or natural_actual.get("pcm_sha256") != natural_ref.get("pcm_sha256")
            or natural_samples.size != expected_natural):
        raise MuseSPSAAdapterError("Stage-A natural PCM reader did not reproduce its frozen source pin")
    receipt_pin = _application_receipt(spsa_binding, input_ref, natural_ref)
    receipt_packet_raw = Path(receipt_pin["path"]).read_bytes()
    receipt_packet = json.loads(receipt_packet_raw.decode("utf-8"))
    # Keep the original Stage-A source path, which must be the same full natural
    # audio file used for SyncNet scoring and lossless delivery.
    if Path(str(natural_ref.get("path", ""))).resolve(strict=True) != natural_path:
        raise MuseSPSAAdapterError("natural PCM path changed during candidate input validation")
    return dict(input_ref), natural_actual


class MuseSPSAStageAAdapter:
    """Render only this model's exact SPSA candidate subset through Stage-A."""

    def __init__(self, *, core: Any, muse_overlay: Any, cross_frontend: Any,
                 parent_plan: Mapping[str, Any], runtime_binding: Mapping[str, Any],
                 runtime_binding_raw: bytes, runtime_binding_path: str | Path,
                 resource_helper: Any, args: Any, strict_receipt_dir: str | Path):
        self.core = core
        self.overlay = muse_overlay
        self.cross = cross_frontend
        self.plan = parent_plan
        self.binding = runtime_binding
        self.binding_raw = bytes(runtime_binding_raw)
        self.binding_path = Path(runtime_binding_path)
        self.resource_helper = resource_helper
        self.args = args
        self.strict_receipt_dir = Path(strict_receipt_dir)
        self.model, self.generator = _resolve_muse_protocol(parent_plan)
        self.identity_rows = _validate_plan_and_binding(
            parent_plan, runtime_binding, self.binding_raw, self.binding_path
        )
        if (not self.binding_path.is_absolute() or self.binding_path.is_symlink()
                or not self.binding_path.is_file()
                or _sha256_file(self.binding_path) != _sha256_bytes(self.binding_raw)):
            raise MuseSPSAAdapterError("runtime binding local path does not contain the supplied exact raw bytes")
        source_execution = _muse_source_execution(parent_plan)
        stagea_ref = source_execution.get("stagea_runner")
        overlay_ref = source_execution.get("musetalk_block_overlay")
        cross_ref = source_execution.get("cross_frontend_runner")
        if not all(isinstance(ref, Mapping) for ref in (stagea_ref, overlay_ref, cross_ref)):
            raise MuseSPSAAdapterError("parent plan lacks exact Stage-A, Muse overlay, or cross-runner source pins")
        _verify_module_pin(core, stagea_ref, "Stage-A core")
        _verify_module_pin(muse_overlay, overlay_ref, "Muse overlay")
        _verify_module_pin(cross_frontend, cross_ref, "cross-frontend runner")
        helper_ref = runtime_binding.get("resources", {}).get("resource_helper", {})
        if not isinstance(helper_ref, Mapping):
            raise MuseSPSAAdapterError("Muse runtime binding lacks its live resource helper pin")
        _verify_module_pin(resource_helper, helper_ref, "Muse live resource helper")
        if not self.strict_receipt_dir.is_absolute() or self.strict_receipt_dir.is_symlink() or not self.strict_receipt_dir.is_dir():
            raise MuseSPSAAdapterError("strict determinism receipt directory must be a pre-created real directory")
        if any(self.strict_receipt_dir.iterdir()):
            raise MuseSPSAAdapterError("each SPSA subset requires a fresh, empty strict receipt directory")
        self._validate_runtime_environment()

    def _validate_runtime_environment(self) -> None:
        expected = {
            "CROSS_TFG_CAPACITY_STAGE": EXPECTED_CAPACITY_STAGE,
            "CROSS_TFG_RUNTIME_BINDING_PATH": str(self.binding_path),
            "CROSS_TFG_RUNTIME_BINDING_RAW_SHA256": _sha256_bytes(self.binding_raw),
            "CROSS_TFG_DETERMINISM_RECEIPT_DIR": str(self.strict_receipt_dir.resolve(strict=True)),
        }
        for key, value in expected.items():
            if os.environ.get(key) != value:
                raise MuseSPSAAdapterError(f"caller runtime environment {key} differs from the frozen Muse binding")

    def _validate_rows(self, rows: Sequence[Mapping[str, Any]], method: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], Mapping[str, Any], str]:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise MuseSPSAAdapterError("candidate rows must be a nonempty ordered sequence")
        config, config_sha = _method_config(method)
        expected_samples = _expected_sample_ids(self.plan, config)
        if len(rows) != len(expected_samples) or len(rows) not in (8, 40):
            raise MuseSPSAAdapterError("Muse subset must contain exactly eight proposal rows or forty center rows")
        if (config["method"] == "fit40_spsa_proposal") != (len(rows) == 8):
            raise MuseSPSAAdapterError("proposal/center method kind differs from the required 8/40 row scope")
        observed_samples = [row.get("sample_id") for row in rows]
        if tuple(observed_samples) != expected_samples:
            raise MuseSPSAAdapterError("candidate rows do not preserve the method's exact ordered sample roster")
        if len(set(observed_samples)) != len(observed_samples):
            raise MuseSPSAAdapterError("candidate subset repeats a source sample")
        parents: list[Mapping[str, Any]] = []
        inputs: list[Mapping[str, Any]] = []
        seen_row_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or row.get("model_id") != MUSE_MODEL_ID:
                raise MuseSPSAAdapterError("Muse adapter accepts only Muse model rows")
            parent_key = _identity_key(row)
            parent = self.identity_rows.get(parent_key)
            if parent is None:
                raise MuseSPSAAdapterError("candidate row has no exact identity parent in the frozen Muse roster")
            binding = _validate_spsa_row_binding(row, parent, method, config, config_sha)
            input_ref, natural_actual = _validate_generator_input(
                row, parent, self.core, binding, config, config_sha
            )
            row_id = str(row.get("row_id"))
            if row_id in seen_row_ids:
                raise MuseSPSAAdapterError("candidate subset repeats a row ID")
            seen_row_ids.add(row_id)
            parents.append(parent)
            inputs.append(input_ref)
        return parents, inputs, config, config_sha

    def render_subset(self, *, candidate_rows: Sequence[Mapping[str, Any]],
                      method: Mapping[str, Any], output_root: str | Path) -> list[dict[str, Any]]:
        """Render and score one Muse subset (8 proposals or 40 full-center rows).

        The function preserves its created row outputs and strict receipts on
        failure.  It never cleans, retries, selects, or updates SPSA state.
        """
        self._validate_runtime_environment()
        parents, input_refs, config, config_sha = self._validate_rows(candidate_rows, method)
        out = Path(output_root)
        if not out.is_absolute() or out.is_symlink() or not out.is_dir():
            raise MuseSPSAAdapterError("subset output root must be an existing absolute real directory")
        if self.strict_receipt_dir.resolve(strict=True) == out.resolve(strict=True):
            raise MuseSPSAAdapterError("strict receipts and Stage-A row outputs require separate directories")
        for row in candidate_rows:
            row_id = row.get("row_id")
            if (not isinstance(row_id, str) or not row_id or Path(row_id).name != row_id
                    or row_id in {".", ".."}):
                raise MuseSPSAAdapterError("candidate row ID is not a safe single path component")
            row_path = out / row_id
            if row_path.exists() or row_path.is_symlink():
                raise MuseSPSAAdapterError("candidate row output already exists; refusing overwrite or replay")
        if not _PATCH_LOCK.acquire(blocking=False):
            raise MuseSPSAAdapterError("another Stage-A adapter call is active in this process")

        original_generator = self.core._run_generator
        original_frame_clock = self.core._frame_clock
        row_frames: dict[str, int] = {}
        runtime_rows = {"completed": 0}
        output: list[dict[str, Any]] = []
        gate_by_row_id: dict[str, Mapping[str, Any]] = {}
        try:
            # Per-row capacity checks use only this subset's remaining roster;
            # the frozen binding and resource helper still enforce the V100,
            # RAM, /dev/shm, cgroup, and 5 GiB reserve checks.
            def muse_generator(**kwargs: Any) -> dict[str, Any]:
                row = kwargs["row"]
                if row.get("model_id") != MUSE_MODEL_ID or kwargs["model"].get("model_id") != MUSE_MODEL_ID:
                    raise MuseSPSAAdapterError("Stage-A generator hook received a non-Muse row/model")
                index = runtime_rows["completed"]
                if index >= len(candidate_rows):
                    raise MuseSPSAAdapterError("Stage-A requested more Muse renders than the candidate subset")
                remaining = copy.copy(dict(self.plan))
                remaining["rows"] = [dict(item) for item in candidate_rows[index:]]
                gate = self.overlay._capacity_observation(
                    remaining, dict(self.binding), self.resource_helper, out, 0, whole_stage=False
                )
                if not isinstance(gate, Mapping) or gate.get("status") != "PASS":
                    raise MuseSPSAAdapterError("live Muse capacity gate failed before a proposal render")
                gate_by_row_id[str(row["row_id"])] = dict(gate)
                work = Path(kwargs["work"])
                receipt_names = {path.name for path in self.strict_receipt_dir.iterdir()
                                 if path.is_file() and not path.is_symlink()}
                if len(receipt_names) != index * 2:
                    raise MuseSPSAAdapterError("strict receipt directory does not contain exactly two receipts per completed row")
                execution = self.overlay._validate_runtime_evidence(
                    self.core, self.cross, **kwargs, generator=self.generator,
                    prior_receipt_names=receipt_names,
                )
                row_binding = row.get("musetalk_native_frame_binding", {})
                expected_frames = row_binding.get("expected_frame_count") if isinstance(row_binding, Mapping) else None
                expected_samples = row_binding.get("sample_count") if isinstance(row_binding, Mapping) else None
                if (type(expected_samples) is not int
                        or type(expected_frames) is not int
                        or expected_frames != expected_native_frame_count(expected_samples)):
                    raise MuseSPSAAdapterError("candidate Muse native frame binding is not floor(N*25/16000)")
                row_frames[str(work.resolve(strict=True))] = expected_frames
                runtime_rows["completed"] += 1
                return dict(execution)

            def muse_frame_clock(path: Path, ffprobe: str, fps_num: int, fps_den: int):
                frames, stream = original_frame_clock(path, ffprobe, fps_num, fps_den)
                row_root = str(Path(path).resolve(strict=False).parent.parent)
                expected = row_frames.get(row_root)
                if expected is not None and len(frames) != expected:
                    raise MuseSPSAAdapterError("decoded Muse video frame count differs from floor(N*25/16000)")
                return frames, stream

            self.core._run_generator = muse_generator
            self.core._frame_clock = muse_frame_clock
            for index, (row, parent, input_ref) in enumerate(zip(candidate_rows, parents, input_refs)):
                natural_path, natural_samples, natural_ref = self.core._reference(row["natural_pcm"])
                if row.get("score_audio") != row.get("natural_pcm"):
                    raise MuseSPSAAdapterError("candidate score audio changed from original natural PCM")
                row_root = out / str(row["row_id"])
                # Stage-A owns all original audio, mux, ROI, NPZ and JSON
                # creation. Only its internal dispatcher marker is adapted.
                render_method = copy.deepcopy(dict(method))
                render_config = copy.deepcopy(dict(config))
                render_config["method"] = "precomputed_fit_snapshot_waveform"
                render_method["config"] = render_config
                if render_method.get("config_sha256") != config_sha:
                    raise MuseSPSAAdapterError("private render adaptation changed the serialized method config pin")
                try:
                    result = self.core._render_and_score(
                        row, self.model, render_method, natural_path, natural_samples,
                        natural_ref, row_root, self.args, qc_repeat=False,
                    )
                except Exception as exc:
                    raise MuseSPSASubsetFailure(
                        f"Muse Stage-A row {row['row_id']} failed: {type(exc).__name__}: {exc}",
                        completed=output, failed_row_id=str(row["row_id"]),
                        unattempted_row_ids=[str(item["row_id"]) for item in candidate_rows[index + 1:]],
                    ) from exc
                if not isinstance(result, dict) or result.get("status") != "PASS":
                    raise MuseSPSAAdapterError("Stage-A Muse row did not return a PASS result")
                if result.get("score_audio_sha256") != row.get("natural_pcm", {}).get("pcm_sha256"):
                    raise MuseSPSAAdapterError("Stage-A result score audio is not the exact original natural PCM")
                if runtime_rows["completed"] != index + 1:
                    raise MuseSPSAAdapterError("Stage-A did not invoke exactly one Muse generator for the current row")
                after_names = {path.name for path in self.strict_receipt_dir.iterdir()
                               if path.is_file() and not path.is_symlink()}
                if len(after_names) != (index + 1) * 2:
                    raise MuseSPSAAdapterError("Muse renderer did not leave exactly two new strict bootstrap receipts")
                arrays = result.get("_internal_arrays")
                if not isinstance(arrays, Mapping):
                    raise MuseSPSAAdapterError("Stage-A did not return its official query-clock/SyncNet arrays")
                array_summary = _validate_official_arrays(arrays)
                public_result = {key: value for key, value in result.items()
                                 if not key.startswith("_internal_")}
                public_result["spsa_binding"] = copy.deepcopy(dict(row["spsa_binding"]))
                public_result["spsa_point_binding_sha256"] = row["spsa_point_binding_sha256"]
                public_result["generator_condition_input"] = copy.deepcopy(dict(input_ref))
                public_result["parent_identity_row_id"] = parent["row_id"]
                output.append({
                    "row": public_result,
                    "official_arrays": arrays,
                    "row_output_root": str(row_root.resolve(strict=True)),
                    "parent_identity_row_id": parent["row_id"],
                    "application_receipt": dict(row["spsa_binding"]["application_receipt"]),
                    "method_config_sha256": config_sha,
                    "resource_gate": gate_by_row_id[str(row["row_id"])],
                    "official_array_summary": array_summary,
                })
            return output
        finally:
            self.core._run_generator = original_generator
            self.core._frame_clock = original_frame_clock
            _PATCH_LOCK.release()


__all__ = ["MuseSPSAAdapterError", "MuseSPSASubsetFailure",
           "MuseSPSAStageAAdapter", "expected_native_frame_count"]
