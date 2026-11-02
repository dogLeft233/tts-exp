"""Ditto-only Stage-A hooks for Fit40 SPSA proposal and center rows.

The caller supplies a freshly applied, receipt-bound generator-drive WAV for
each candidate row.  This adapter validates that application against the
frozen natural PCM and SPSA point, then invokes the existing Stage-A
``_render_and_score`` one row at a time.  Ditto receives the transformed WAV;
official SyncNet scoring and lossless audio delivery continue to use the
original natural PCM.  No optimizer, selection, or retry is performed here.

Only a private method copy receives the Stage-A dispatcher marker
``precomputed_fit_snapshot_waveform``.  The serialized family plan, candidate
method, row, and their hashes are not changed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


DITTO_MODEL_ID = "ditto_hubert_fixed_portrait_attempt18"
PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
SUPPORTED_METHODS = {"fit40_spsa_proposal", "fit40_spsa_center"}
EXPECTED_MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    DITTO_MODEL_ID,
)
_PATCH_LOCK = threading.Lock()


class DittoSPSAAdapterError(RuntimeError):
    """A Ditto SPSA row or runtime input violates the frozen Stage-A contract."""


class DittoSPSASubsetFailure(DittoSPSAAdapterError):
    """A row failed while preserving completed and unattempted row evidence."""

    def __init__(self, message: str, *, completed: Sequence[Mapping[str, Any]],
                 failed_row_id: str, unattempted_row_ids: Sequence[str]):
        super().__init__(message)
        self.completed = list(completed)
        self.failed_row_id = failed_row_id
        self.unattempted_row_ids = list(unattempted_row_ids)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular(path: Path, *, limit: int = 32 * 1024 * 1024) -> bytes:
    if path.is_symlink():
        raise DittoSPSAAdapterError(f"refusing symlink input: {path}")
    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise DittoSPSAAdapterError(f"required input is absent: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > limit:
        raise DittoSPSAAdapterError(f"input is not a bounded nonempty regular file: {path}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size or len(raw) > limit:
        raise DittoSPSAAdapterError(f"input changed or exceeded its read bound: {path}")
    return raw


def _require_sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise DittoSPSAAdapterError(f"{label} must be a lowercase SHA-256")
    return value


def _self_hash_matches(value: Mapping[str, Any], field: str) -> bool:
    expected = value.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    return _sha256(_canonical({key: item for key, item in value.items() if key != field})) == expected


def _read_pinned_file(pin: Mapping[str, Any], label: str, *,
                      sha_field: str = "sha256", limit: int = 32 * 1024 * 1024) -> tuple[Path, bytes]:
    path_text = pin.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise DittoSPSAAdapterError(f"{label} path is absent")
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink():
        raise DittoSPSAAdapterError(f"{label} must be absolute and not a symlink")
    raw = _read_regular(path, limit=limit)
    if type(pin.get("bytes")) is not int or len(raw) != pin["bytes"]:
        raise DittoSPSAAdapterError(f"{label} byte count differs from its pin")
    if _sha256(raw) != pin.get(sha_field):
        raise DittoSPSAAdapterError(f"{label} raw SHA differs from its pin")
    return path.resolve(strict=True), raw


def expected_native_frame_count(sample_count: int, fps_num: int = 25,
                                fps_den: int = 1, sample_rate_hz: int = 16000) -> int:
    """Return Ditto's frozen ceil(N*fps/sample_rate) frame count."""
    if any(type(value) is not int or value <= 0
           for value in (sample_count, fps_num, fps_den, sample_rate_hz)):
        raise DittoSPSAAdapterError("sample and clock values must be positive integers")
    numerator = sample_count * fps_num
    denominator = sample_rate_hz * fps_den
    return (numerator + denominator - 1) // denominator


def _method_config(method: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    config = method.get("config")
    if not isinstance(config, Mapping) or config.get("method") not in SUPPORTED_METHODS:
        raise DittoSPSAAdapterError("only the frozen SPSA proposal or center method may be rendered")
    config_sha = _require_sha(method.get("config_sha256"), "method.config_sha256")
    if _sha256(_canonical(config)) != config_sha:
        raise DittoSPSAAdapterError("method config canonical SHA differs from its declared pin")
    if method.get("method_id") != config.get("method_id") or method.get("kind") != config.get("method"):
        raise DittoSPSAAdapterError("method identity differs between descriptor and config")
    if method.get("checkpoint_sha256") is not None:
        raise DittoSPSAAdapterError("SPSA waveform methods cannot substitute a checkpoint")
    theta = config.get("theta")
    if (not isinstance(theta, (list, tuple)) or len(theta) != 9
            or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in theta)):
        raise DittoSPSAAdapterError("SPSA method must bind exactly nine finite theta values")
    if _sha256(_canonical(list(theta))) != _require_sha(config.get("theta_sha256"), "theta_sha256"):
        raise DittoSPSAAdapterError("theta values differ from their canonical SHA")
    for key in ("artifact_sha256", "binding_sha256", "input_pins_sha256"):
        _require_sha(config.get(key), f"method.config.{key}")
    return config, config_sha


def _verify_plan_file_pin(plan: Mapping[str, Any], pin: Mapping[str, Any], label: str) -> None:
    pins = plan.get("scientific_input_pins")
    if not isinstance(pins, Mapping):
        raise DittoSPSAAdapterError("family plan lacks its scientific input pin map")
    matches = [candidate for candidate in pins.values() if isinstance(candidate, Mapping)
               and candidate.get("path") == pin.get("path")
               and candidate.get("bytes") == pin.get("bytes")
               and candidate.get("sha256") == pin.get("sha256")]
    if len(matches) != 1:
        raise DittoSPSAAdapterError(f"{label} is not uniquely pinned by the frozen family plan")


def _verify_snapshot_lineage(plan: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("input_pins_sha256") != plan.get("scientific_input_pins_sha256"):
        raise DittoSPSAAdapterError("method input pins differ from the frozen Fit family")
    artifact = _require_sha(config.get("artifact_sha256"), "artifact_sha256")
    source_artifact = _require_sha(config.get("source_snapshot_artifact_sha256"),
                                   "source_snapshot_artifact_sha256")
    if artifact != source_artifact:
        raise DittoSPSAAdapterError("SPSA point does not retain its selected seed snapshot artifact")
    source_step = config.get("source_snapshot_step")
    seed_method_id = config.get("source_seed_method_id")
    if type(source_step) is not int or source_step not in {30, 60, 90, 120} or not isinstance(seed_method_id, str):
        raise DittoSPSAAdapterError("SPSA point lacks a predeclared Fit snapshot source")
    seeds = [item for item in plan.get("methods", []) if isinstance(item, Mapping)
             and item.get("method_id") == seed_method_id]
    if len(seeds) != 1:
        raise DittoSPSAAdapterError("seed snapshot method is not unique in the frozen plan")
    seed_config = seeds[0].get("config")
    if (not isinstance(seed_config, Mapping)
            or seed_config.get("method") != "precomputed_fit_snapshot_waveform"
            or seed_config.get("source_optimizer_step") != source_step
            or seed_config.get("snapshot_sha256") != artifact
            or seed_config.get("training_family") != config.get("head_family")):
        raise DittoSPSAAdapterError("SPSA source snapshot differs from its original Fit method")

    record_pin = config.get("source_snapshot_record")
    index_pin = config.get("source_snapshot_index")
    if not isinstance(record_pin, Mapping) or not isinstance(index_pin, Mapping):
        raise DittoSPSAAdapterError("SPSA method lacks pinned source snapshot/index records")
    snapshot: dict[str, Any] | None = None
    for label, pin, raw_key, self_field in (
            ("source snapshot", record_pin, "source_snapshot_record_raw_sha256", "snapshot_sha256"),
            ("snapshot index", index_pin, "source_snapshot_index_raw_sha256", "index_sha256")):
        _verify_plan_file_pin(plan, pin, label)
        _, raw = _read_pinned_file(pin, label)
        if config.get(raw_key) != _sha256(raw):
            raise DittoSPSAAdapterError(f"{label} raw bytes differ from the method config")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DittoSPSAAdapterError(f"{label} is not valid JSON") from exc
        if not isinstance(value, dict) or not _self_hash_matches(value, self_field):
            raise DittoSPSAAdapterError(f"{label} canonical self hash is invalid")
        if pin.get("self_sha256") != value.get(self_field):
            raise DittoSPSAAdapterError(f"{label} self hash differs from its file pin")
        if label == "source snapshot":
            snapshot = value
            if (value.get("schema") != "cross_tfg_fit40_parameter_snapshot_v1"
                    or value.get("selection_status") != "NOT_SELECTED"
                    or value.get("optimizer_step_is_selected") is not False
                    or value.get("snapshot_sha256") != artifact
                    or value.get("method_id") != seed_method_id
                    or value.get("source_optimizer_step") != source_step
                    or value.get("training_family") != config.get("head_family")
                    or value.get("head_schema") != "cross_tfg_shared_recipient_transition_head_v3"):
                raise DittoSPSAAdapterError("seed snapshot status, source step, family, or head schema differs")
            parameter = value.get("parameter")
            source_theta = config.get("source_theta")
            if (not isinstance(parameter, Mapping) or parameter.get("values") != source_theta
                    or config.get("source_theta_sha256") != _sha256(_canonical(source_theta))
                    or list(source_theta or []) != list(seed_config.get("theta", []))):
                raise DittoSPSAAdapterError("serialized source theta differs from the exact seed snapshot")
            constructor = value.get("head_constructor")
            if value.get("head_constructor_sha256") != _sha256(_canonical(constructor)):
                raise DittoSPSAAdapterError("snapshot head constructor differs from its raw pin")
            implementation = value.get("head_implementation")
            if not isinstance(implementation, Mapping):
                raise DittoSPSAAdapterError("snapshot lacks the exact v3 head implementation pin")
            _verify_plan_file_pin(plan, implementation, "snapshot head implementation")
        else:
            indexed = [item for item in value.get("snapshots", []) if isinstance(item, Mapping)
                       and item.get("method_id") == seed_method_id]
            if (len(indexed) != 1 or indexed[0].get("snapshot_sha256") != artifact
                    or indexed[0].get("source_optimizer_step") != source_step
                    or indexed[0].get("sha256") != record_pin.get("sha256")
                    or indexed[0].get("bytes") != record_pin.get("bytes")):
                raise DittoSPSAAdapterError("snapshot index does not bind the exact seed record")
    if snapshot is None:
        raise DittoSPSAAdapterError("source snapshot record was not loaded")
    return snapshot


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(name) for name in
                 ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed"))


_ROW_MUTABLE_FIELDS = {
    "row_id", "method_id", "method_config_sha256", "checkpoint_sha256",
    "generator_condition_input", "spsa_binding", "spsa_point_binding_sha256",
}


def _stable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in _ROW_MUTABLE_FIELDS}


def _expected_sample_ids(plan: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, ...]:
    if config.get("method") == "fit40_spsa_proposal":
        samples = config.get("sample_ids")
        if (not isinstance(samples, list) or len(samples) != 8 or len(set(samples)) != 8
                or any(not isinstance(value, str) for value in samples)):
            raise DittoSPSAAdapterError("proposal must bind exactly eight ordered Fit samples")
        return tuple(samples)
    scope = plan.get("execution", {}).get("spsa_execution_scope", {})
    samples = scope.get("sample_ids") if isinstance(scope, Mapping) else None
    if samples is None:
        samples = list(dict.fromkeys(
            str(row.get("sample_id")) for row in plan.get("rows", [])
            if isinstance(row, Mapping) and row.get("model_id") == DITTO_MODEL_ID
            and row.get("method_id") == "identity"))
    if (not isinstance(samples, (list, tuple)) or len(samples) != 40
            or len(set(samples)) != 40 or any(not isinstance(value, str) for value in samples)):
        raise DittoSPSAAdapterError("center point must bind the full ordered 40-clip Fit roster")
    return tuple(samples)


def _validate_spsa_row_binding(row: Mapping[str, Any], parent: Mapping[str, Any],
                               method: Mapping[str, Any], config: Mapping[str, Any],
                               config_sha: str) -> dict[str, Any]:
    binding = row.get("spsa_binding")
    if not isinstance(binding, Mapping):
        raise DittoSPSAAdapterError("candidate row lacks its explicit SPSA point binding")
    point_sha = _require_sha(row.get("spsa_point_binding_sha256"), "spsa_point_binding_sha256")
    if (point_sha != config.get("binding_sha256")
            or row.get("method_id") != method.get("method_id")
            or row.get("method_config_sha256") != config_sha
            or binding.get("method_id") != method.get("method_id")
            or binding.get("method_config_sha256") != config_sha
            or binding.get("parent_row_id") != parent.get("row_id")
            or binding.get("seed") != row.get("seed")):
        raise DittoSPSAAdapterError("row, method, parent, or renderer seed differs from its SPSA binding")
    for name in ("artifact_sha256", "binding_sha256", "input_pins_sha256", "theta_sha256"):
        if binding.get(name) != config.get(name):
            raise DittoSPSAAdapterError(f"row SPSA binding differs from method field {name}")
    if list(binding.get("theta", [])) != list(config.get("theta", [])):
        raise DittoSPSAAdapterError("row theta differs from the serialized SPSA method")
    if config.get("method") == "fit40_spsa_proposal":
        for name in ("proposal_id", "pair_id", "iteration", "side", "batch_id", "c_k"):
            if binding.get(name) != config.get(name):
                raise DittoSPSAAdapterError(f"proposal row binding differs from method field {name}")
        if binding.get("side") not in {"plus", "minus"}:
            raise DittoSPSAAdapterError("proposal side must be plus or minus")
    else:
        if (binding.get("point_kind") != "spsa_center"
                or binding.get("source_step") != config.get("source_step")
                or binding.get("source_step") not in {5, 10, 15, 20}):
            raise DittoSPSAAdapterError("center row must bind one predeclared center step")
    if _stable_row(row) != _stable_row(parent):
        raise DittoSPSAAdapterError("candidate row altered frozen source, face, split, or clock metadata")
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id or row_id == parent.get("row_id"):
        raise DittoSPSAAdapterError("candidate row requires a new row_id distinct from its identity parent")
    return dict(binding)


def _validate_application_receipt(binding: Mapping[str, Any], input_ref: Mapping[str, Any],
                                  natural_ref: Mapping[str, Any], config: Mapping[str, Any],
                                  config_sha: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    receipt_pin = binding.get("application_receipt")
    if not isinstance(receipt_pin, Mapping):
        raise DittoSPSAAdapterError("row SPSA binding lacks the generator-drive application receipt pin")
    receipt_path, raw = _read_pinned_file(receipt_pin, "generator-drive application receipt")
    packet = json.loads(raw.decode("utf-8"))
    if (not isinstance(packet, dict) or not _self_hash_matches(packet, "self_sha256")
            or receipt_pin.get("self_sha256") != packet.get("self_sha256")):
        raise DittoSPSAAdapterError("application receipt is not a valid self-hashed JSON object")
    canonical_pcm = packet.get("canonical_original_pcm")
    original_file = packet.get("input_original_file")
    if (packet.get("schema") != "cross_tfg_fit_snapshot_generator_drive_result_v1"
            or packet.get("status") != "PASS_FIT_SNAPSHOT_GENERATOR_DRIVE_ONLY"
            or packet.get("render_or_score_performed") is not False
            or packet.get("optimizer_created_or_stepped") is not False
            or not isinstance(original_file, Mapping)
            or original_file.get("sha256") != natural_ref.get("file_sha256")
            or not isinstance(canonical_pcm, Mapping)
            or canonical_pcm.get("pcm_float32_sha256") != natural_ref.get("pcm_sha256")
            or canonical_pcm.get("sample_count") != natural_ref.get("sample_count")
            or packet.get("head_schema") != snapshot.get("head_schema")
            or packet.get("head_implementation") != snapshot.get("head_implementation")
            or packet.get("head_constructor_sha256") != snapshot.get("head_constructor_sha256")
            or packet.get("snapshot_artifact_sha256") != config.get("artifact_sha256")
            or packet.get("source_snapshot_step") != config.get("source_snapshot_step")
            or packet.get("source_snapshot_record_raw_sha256") != config.get("source_snapshot_record_raw_sha256")
            or packet.get("method_id") != config.get("method_id")
            or packet.get("method_config_sha256") != config_sha
            or packet.get("artifact_sha256") != config.get("artifact_sha256")
            or packet.get("binding_sha256") != config.get("binding_sha256")
            or packet.get("input_pins_sha256") != config.get("input_pins_sha256")
            or packet.get("theta") != config.get("theta")
            or packet.get("theta_sha256") != config.get("theta_sha256")
            or packet.get("source_clock_preserved_after_canonicalization") is not True):
        raise DittoSPSAAdapterError("application receipt differs from the exact source, snapshot, or SPSA theta")
    for name in ("proposal_id", "pair_id", "iteration", "side", "batch_id", "c_k",
                 "point_kind", "source_step", "parent_row_id", "sample_id", "speaker_id", "seed"):
        if name in config and packet.get(name) != config.get(name):
            raise DittoSPSAAdapterError(f"application receipt differs from SPSA point field {name}")
    rms = packet.get("rms_evidence")
    projection = packet.get("projection_evidence")
    constructor = snapshot.get("head_constructor")
    if not isinstance(constructor, Mapping) or not isinstance(rms, Mapping) or not isinstance(projection, Mapping):
        raise DittoSPSAAdapterError("application receipt omits constructor, RMS, or projection evidence")
    target_rms = constructor.get("target_rms")
    output_rms = rms.get("output_rms")
    max_mask_db = projection.get("postprojection_max_abs_mask_db")
    if (rms.get("sample_count") != natural_ref.get("sample_count")
            or rms.get("target_rms") != target_rms
            or rms.get("peak_ceiling") is not None
            or not isinstance(output_rms, (int, float)) or not math.isfinite(float(output_rms))
            or abs(float(output_rms) - float(target_rms)) > float(constructor.get("baseline_match_atol", 0.0))
            or not isinstance(max_mask_db, (int, float)) or not math.isfinite(float(max_mask_db))
            or float(max_mask_db) > float(constructor.get("max_abs_mask_db", 0.0)) + 1e-5):
        raise DittoSPSAAdapterError("application RMS, peak, or projected mask evidence violates the frozen constructor")
    output_pin = packet.get("output_generator_drive_wav")
    if not isinstance(output_pin, Mapping):
        raise DittoSPSAAdapterError("application receipt omits its output generator-drive WAV pin")
    output_matches = (
        input_ref.get("path") == output_pin.get("path")
        and input_ref.get("bytes") == output_pin.get("bytes")
        and input_ref.get("file_sha256") == output_pin.get("file_sha256")
        and input_ref.get("pcm_sha256") == output_pin.get("pcm_float32_sha256")
        and input_ref.get("sample_count") == output_pin.get("sample_count")
        and input_ref.get("sample_rate_hz") == output_pin.get("sample_rate_hz")
        and input_ref.get("channels") == output_pin.get("channels")
        and input_ref.get("encoding", input_ref.get("subtype")) == output_pin.get("encoding")
        and packet.get("output_pcm_float32_sha256") == input_ref.get("pcm_sha256")
        and packet.get("output_sample_count") == input_ref.get("sample_count")
    )
    raw_sha = _sha256(raw)
    if (not output_matches
            or binding.get("application_receipt_raw_sha256") != raw_sha
            or binding.get("application_receipt_self_sha256") != packet.get("self_sha256")
            or input_ref.get("application_receipt_raw_sha256") != raw_sha
            or input_ref.get("application_receipt_self_sha256") != packet.get("self_sha256")):
        raise DittoSPSAAdapterError("generator-drive WAV or receipt raw/self link differs from the SPSA row")
    return {"path": str(receipt_path), "bytes": len(raw), "sha256": raw_sha,
            "self_sha256": packet["self_sha256"]}


def _validate_generator_input(row: Mapping[str, Any], parent: Mapping[str, Any], core: Any,
                              binding: Mapping[str, Any], config: Mapping[str, Any],
                              config_sha: str, snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    natural_ref = row.get("natural_pcm")
    if not isinstance(natural_ref, Mapping) or row.get("score_audio") != natural_ref:
        raise DittoSPSAAdapterError("candidate must score the exact frozen natural PCM")
    input_ref = row.get("generator_condition_input")
    if not isinstance(input_ref, Mapping):
        raise DittoSPSAAdapterError("candidate row lacks its generated full-length generator-condition WAV")
    if _stable_row(row) != _stable_row(parent):
        raise DittoSPSAAdapterError("candidate row differs from identity parent outside SPSA/input fields")
    input_path, input_raw = _read_pinned_file(input_ref, "generator-condition WAV", sha_field="file_sha256",
                                              limit=64 * 1024 * 1024)
    from scripts.audio_enhancement.audio_io import read_canonical_wav

    transformed, info = read_canonical_wav(input_path)
    sample_count = natural_ref.get("sample_count")
    if (type(sample_count) is not int or sample_count <= 0
            or info.sample_rate != 16000 or info.source_sample_rate != 16000
            or info.source_channels != 1 or info.frames != sample_count
            or transformed.size != sample_count
            or input_ref.get("sample_count") != sample_count
            or input_ref.get("sample_rate_hz") != 16000 or input_ref.get("channels") != 1
            or input_ref.get("pcm_sha256") != info.pcm_float32_sha256):
        raise DittoSPSAAdapterError("generator-drive output changed canonical mono 16 kHz natural-clock length")
    encoding = input_ref.get("encoding", input_ref.get("subtype"))
    if (info.source_encoding != "FLOAT_32"
            or encoding not in {"FLOAT_32", "IEEE_FLOAT_32_LE", "FLOAT32", "float32"}):
        raise DittoSPSAAdapterError("generator-drive audio must be a full-length canonical float32 WAV")
    frame_binding = row.get("ditto_native_frame_binding")
    expected_frames = expected_native_frame_count(sample_count)
    if (not isinstance(frame_binding, Mapping)
            or frame_binding.get("sample_count") != sample_count
            or frame_binding.get("sample_rate_hz") != 16000
            or frame_binding.get("fps_num") != 25 or frame_binding.get("fps_den") != 1
            or frame_binding.get("expected_frame_count") != expected_frames):
        raise DittoSPSAAdapterError("Ditto native frame binding differs from ceil(N*25/16000)")
    if not all(math.isfinite(float(value)) for value in transformed):
        raise DittoSPSAAdapterError("generator-drive waveform contains nonfinite float32 samples")

    row_binding = row.get("spsa_binding")
    if not isinstance(row_binding, Mapping):
        raise DittoSPSAAdapterError("candidate row lacks SPSA binding metadata")
    point_fields = ("proposal_id", "pair_id", "side", "method_id", "method_config_sha256",
                    "artifact_sha256", "binding_sha256", "input_pins_sha256", "theta_sha256",
                    "parent_row_id")
    for name in point_fields:
        expected = row_binding.get(name)
        if name == "method_id":
            expected = config.get(name)
        elif name == "method_config_sha256":
            expected = config_sha
        elif name in {"artifact_sha256", "binding_sha256", "input_pins_sha256", "theta_sha256"}:
            expected = config.get(name)
        elif name == "parent_row_id":
            expected = parent.get("row_id")
        if name in row_binding and input_ref.get(name) != expected:
            raise DittoSPSAAdapterError(f"generator-drive WAV receipt differs from row point field {name}")
    if input_ref.get("method_id") != config.get("method_id"):
        raise DittoSPSAAdapterError("generator-drive WAV names a different SPSA method")
    for name in ("method_config_sha256", "artifact_sha256", "binding_sha256",
                 "input_pins_sha256", "theta_sha256"):
        if input_ref.get(name) != config.get(name) and name != "method_config_sha256":
            raise DittoSPSAAdapterError(f"generator-drive WAV differs from method field {name}")
    if input_ref.get("method_config_sha256") != config_sha or input_ref.get("parent_row_id") != parent.get("row_id"):
        raise DittoSPSAAdapterError("generator-drive WAV config or identity-parent link differs")
    if _sha256(_canonical(list(config["theta"]))) != config.get("theta_sha256"):
        raise DittoSPSAAdapterError("SPSA theta differs from its canonical hash")

    natural_path, natural_samples, natural_actual = core._reference(natural_ref)
    if (natural_actual.get("file_sha256") != natural_ref.get("file_sha256")
            or natural_actual.get("pcm_sha256") != natural_ref.get("pcm_sha256")
            or natural_samples.size != sample_count):
        raise DittoSPSAAdapterError("Stage-A natural PCM reader did not reproduce the frozen source pin")
    if Path(str(natural_ref.get("path", ""))).resolve(strict=True) != natural_path:
        raise DittoSPSAAdapterError("natural PCM path changed during generator-drive validation")
    receipt = _validate_application_receipt(row_binding, input_ref, natural_ref,
                                            config, config_sha, snapshot)
    if (input_ref.get("bytes") != len(input_raw)
            or input_ref.get("file_sha256") != info.file_sha256
            or input_ref.get("pcm_sha256") != info.pcm_float32_sha256):
        raise DittoSPSAAdapterError("generator-drive WAV file/raw/PCM identities disagree")
    return dict(input_ref), natural_actual


def _validate_official_arrays(arrays: Mapping[str, Any]) -> dict[str, int]:
    expected = {
        "query_frame_index": ("<i4", 1), "query_timestamp_num": ("<i8", 1),
        "query_timestamp_den": ("<i8", 1), "track_frame_index": ("<i4", 1),
        "valid_query_mask": ("|b1", 1), "distance_by_query_lag": ("<f4", 2),
        "lag_offsets_frames": ("<i4", 1),
    }
    if not isinstance(arrays, Mapping) or set(arrays) != set(expected):
        raise DittoSPSAAdapterError("Stage-A returned a noncanonical official-array set")
    shapes: dict[str, tuple[int, ...]] = {}
    for name, (dtype, ndim) in expected.items():
        array = arrays[name]
        shape = getattr(array, "shape", None)
        actual_dtype = getattr(getattr(array, "dtype", None), "str", None)
        if (not isinstance(shape, tuple) or len(shape) != ndim or actual_dtype != dtype
                or any(type(dim) is not int or dim < 0 for dim in shape)):
            raise DittoSPSAAdapterError(f"official array {name} has an invalid shape or dtype")
        shapes[name] = shape
    query_count = shapes["query_frame_index"][0]
    if query_count <= 0 or shapes["distance_by_query_lag"] != (query_count, 31):
        raise DittoSPSAAdapterError("official SyncNet matrix must have shape [actual_queries,31]")
    for name in ("query_timestamp_num", "query_timestamp_den", "track_frame_index", "valid_query_mask"):
        if shapes[name] != (query_count,):
            raise DittoSPSAAdapterError(f"official query metadata {name} differs from matrix query count")
    if shapes["lag_offsets_frames"] != (31,) or arrays["lag_offsets_frames"].tolist() != list(range(-15, 16)):
        raise DittoSPSAAdapterError("official SyncNet lag axis differs from the frozen -15..15 grid")
    den = arrays["query_timestamp_den"].tolist()
    if any(int(value) <= 0 for value in den):
        raise DittoSPSAAdapterError("official source-clock denominators must be positive")
    keys = []
    for frame, num, den_value, track in zip(
            arrays["query_frame_index"].tolist(), arrays["query_timestamp_num"].tolist(), den,
            arrays["track_frame_index"].tolist()):
        ts = Fraction(int(num), int(den_value))
        keys.append((int(frame), ts.numerator, ts.denominator, int(track)))
    if len(set(keys)) != query_count:
        raise DittoSPSAAdapterError("official query clock contains duplicate normalized keys")
    matrix = arrays["distance_by_query_lag"]
    if not all(math.isfinite(float(value)) for value in matrix.flat):
        raise DittoSPSAAdapterError("official SyncNet matrix contains nonfinite values")
    mask = arrays["valid_query_mask"].tolist()
    if any(type(value) not in (bool, int) or int(value) not in (0, 1) for value in mask):
        raise DittoSPSAAdapterError("official valid-query mask is not boolean")
    return {"query_count": query_count, "valid_query_count": sum(bool(value) for value in mask)}


def _module_file_sha(module: Any, expected_ref: Mapping[str, Any], label: str) -> tuple[Path, str]:
    file_value = getattr(module, "__file__", None)
    expected_path = expected_ref.get("path")
    if not isinstance(file_value, str) or not isinstance(expected_path, str):
        raise DittoSPSAAdapterError(f"{label} has no exact loaded source path")
    path = Path(file_value)
    if path.is_symlink() or not path.is_file():
        raise DittoSPSAAdapterError(f"{label} loaded from an absent or symlinked source")
    resolved = path.resolve(strict=True)
    if str(resolved) != expected_path:
        raise DittoSPSAAdapterError(f"{label} was loaded from an unexpected source path")
    raw = _read_regular(resolved, limit=32 * 1024 * 1024)
    if (type(expected_ref.get("bytes")) is not int or len(raw) != expected_ref.get("bytes")
            or _sha256(raw) != expected_ref.get("sha256")):
        raise DittoSPSAAdapterError(f"{label} source bytes differ from the frozen plan")
    return resolved, _sha256(raw)


def _load_jsonl_pin(pin: Mapping[str, Any], label: str) -> tuple[Path, bytes, list[dict[str, Any]]]:
    path, raw = _read_pinned_file(pin, label, limit=32 * 1024 * 1024)
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DittoSPSAAdapterError(f"{label} is not valid UTF-8 JSONL") from exc
    if b"".join(_canonical(row) + b"\n" for row in rows) != raw:
        raise DittoSPSAAdapterError(f"{label} is not canonical ordered JSONL")
    if any(not isinstance(row, dict) for row in rows):
        raise DittoSPSAAdapterError(f"{label} contains a non-object row")
    return path, raw, rows


def _source_model_entry(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = plan.get("execution", {}).get("source_model_plans")
    if not isinstance(entries, list):
        raise DittoSPSAAdapterError("composed family plan lacks source-model plan pins")
    matches = [entry for entry in entries if isinstance(entry, Mapping)
               and entry.get("model_id") == DITTO_MODEL_ID]
    if len(matches) != 1:
        raise DittoSPSAAdapterError("composed family plan must pin one source Ditto plan")
    return matches[0]


def _load_source_ditto_plan(family_plan: Mapping[str, Any]) -> tuple[dict[str, Any], bytes, Path, list[dict[str, Any]], bytes, Path]:
    entry = _source_model_entry(family_plan)
    plan_pin = entry.get("plan")
    rows_pin = entry.get("planned_rows")
    if not isinstance(plan_pin, Mapping) or not isinstance(rows_pin, Mapping):
        raise DittoSPSAAdapterError("composed family plan lacks Ditto plan or row-file pins")
    plan_path, plan_raw = _read_pinned_file(plan_pin, "source Ditto plan", limit=32 * 1024 * 1024)
    rows_path, rows_raw, rows = _load_jsonl_pin(rows_pin, "source Ditto planned rows")
    source = json.loads(plan_raw.decode("utf-8"))
    if (not isinstance(source, dict) or source.get("schema_version") != PLAN_SCHEMA
            or source.get("status") != "FROZEN" or not _self_hash_matches(source, "plan_sha256")
            or rows != source.get("rows") or len(rows) != 240):
        raise DittoSPSAAdapterError("source Ditto plan is not its exact frozen 240-row self-hashed producer")
    family_execution = family_plan.get("execution", {})
    model_execution = family_execution.get("source_model_execution", {})
    model_protocols = family_execution.get("source_model_protocols", {})
    if (model_execution.get(DITTO_MODEL_ID) != source.get("execution")
            or model_protocols.get(DITTO_MODEL_ID) != source.get("protocol")):
        raise DittoSPSAAdapterError("composed family plan changed the source Ditto execution or protocol")
    return source, plan_raw, plan_path, rows, rows_raw, rows_path


def _load_runtime_binding(family_plan: Mapping[str, Any], source_plan: Mapping[str, Any],
                          source_plan_path: Path) -> tuple[dict[str, Any], bytes, Path]:
    refs = family_plan.get("execution", {}).get("runtime_bindings_by_model", {})
    ref = refs.get(DITTO_MODEL_ID) if isinstance(refs, Mapping) else None
    source_ref = source_plan.get("execution", {}).get("ditto_runtime_binding")
    if not isinstance(ref, Mapping) or not isinstance(source_ref, Mapping):
        raise DittoSPSAAdapterError("family and Ditto source plans do not both bind the runtime receipt")
    if ref.get("path") != source_ref.get("path") or ref.get("raw_sha256") != source_ref.get("raw_sha256"):
        raise DittoSPSAAdapterError("composed runtime binding ref differs from Ditto's frozen source ref")
    path_input = Path(str(ref.get("path", "")))
    if not path_input.is_absolute() or path_input.is_symlink():
        raise DittoSPSAAdapterError("Ditto runtime binding path must be absolute and non-symlink")
    path = path_input.resolve(strict=True)
    raw = _read_regular(path, limit=8 * 1024 * 1024)
    expected = ref.get("raw_sha256")
    if not isinstance(expected, str) or _sha256(raw) != expected:
        raise DittoSPSAAdapterError("Ditto runtime binding raw bytes differ from the family/source plan")
    binding = json.loads(raw.decode("utf-8"))
    if not isinstance(binding, dict) or not _self_hash_matches(binding, "binding_sha256"):
        raise DittoSPSAAdapterError("Ditto runtime binding is not a self-hashed JSON object")
    if binding.get("model_id") != DITTO_MODEL_ID or binding.get("run_id") != source_plan.get("run_id"):
        raise DittoSPSAAdapterError("Ditto runtime binding belongs to another model or source plan")
    return binding, raw, path


def _validate_family_plan(plan: Mapping[str, Any]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "FROZEN":
        raise DittoSPSAAdapterError("base Fit plan is not a frozen Stage-A family plan")
    if not _self_hash_matches(plan, "plan_sha256"):
        raise DittoSPSAAdapterError("base Fit family plan self hash is invalid")
    execution = plan.get("execution")
    if (not isinstance(execution, Mapping) or execution.get("planned_rows") != 720
            or execution.get("model_ids") != list(EXPECTED_MODEL_ORDER)
            or execution.get("dev_or_confirm_access") is not False
            or execution.get("candidate_selection") != "not performed"
            or execution.get("scientific_input_pins_sha256") != plan.get("scientific_input_pins_sha256")
            or plan.get("generation_repeat_qc") != []):
        raise DittoSPSAAdapterError("base plan does not preserve the frozen 720-row Fit-only family scope")
    models, methods, rows = plan.get("models"), plan.get("methods"), plan.get("rows")
    if (not isinstance(models, list) or tuple(item.get("model_id") for item in models) != EXPECTED_MODEL_ORDER
            or not isinstance(methods, list) or len(methods) != 6
            or not isinstance(rows, list) or len(rows) != 720):
        raise DittoSPSAAdapterError("base family plan model/method/row roster is incomplete")
    method_ids = [item.get("method_id") for item in methods if isinstance(item, Mapping)]
    if len(method_ids) != 6 or len(set(method_ids)) != 6:
        raise DittoSPSAAdapterError("base family plan must contain six unique source methods")
    model = next(item for item in models if item.get("model_id") == DITTO_MODEL_ID)
    ditto_rows = [row for row in rows if isinstance(row, Mapping) and row.get("model_id") == DITTO_MODEL_ID]
    if len(ditto_rows) != 240:
        raise DittoSPSAAdapterError("base family plan must contain exactly 240 Ditto source rows")
    identities: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    contexts: set[tuple[str, str]] = set()
    per_method = {method_id: 0 for method_id in method_ids}
    sample_speaker: dict[str, str] = {}
    speaker_samples: dict[str, set[str]] = {}
    for row in ditto_rows:
        method_id = row.get("method_id")
        sample_id = row.get("sample_id")
        if row.get("split") != "fit" or row.get("score_audio") != row.get("natural_pcm"):
            raise DittoSPSAAdapterError("Ditto base rows must be Fit-only and score the exact natural PCM")
        key = (str(sample_id), str(method_id))
        if key in contexts or method_id not in per_method:
            raise DittoSPSAAdapterError("Ditto family plan repeats or declares an unknown context/method")
        contexts.add(key)
        per_method[method_id] += 1
        if method_id == "identity":
            identity_key = _identity_key(row)
            if identity_key in identities:
                raise DittoSPSAAdapterError("base Ditto plan repeats an identity context")
            identities[identity_key] = row
            speaker = str(row.get("speaker_id"))
            sample = str(sample_id)
            if sample in sample_speaker and sample_speaker[sample] != speaker:
                raise DittoSPSAAdapterError("Fit sample maps to more than one speaker")
            sample_speaker[sample] = speaker
            speaker_samples.setdefault(speaker, set()).add(sample)
    if (len(contexts) != 240 or set(per_method.values()) != {40} or len(identities) != 40
            or len(sample_speaker) != 40 or len(speaker_samples) != 20
            or any(len(items) != 2 for items in speaker_samples.values())):
        raise DittoSPSAAdapterError("Ditto identity roster must preserve 40 clips across 20 two-clip speakers")
    if (model.get("generator") != plan.get("execution", {}).get("source_model_protocols", {})
            and False):
        # Keep model/protocol checks in _validate_composed_source_plan below;
        # this branch deliberately has no alternate acceptance path.
        raise DittoSPSAAdapterError("unreachable model descriptor mismatch")
    return identities


def _validate_composed_source_plan(family_plan: Mapping[str, Any], source_plan: Mapping[str, Any]) -> None:
    models = family_plan["models"]
    model = next(item for item in models if item.get("model_id") == DITTO_MODEL_ID)
    execution = family_plan["execution"]
    source_protocol = source_plan.get("protocol")
    generator = source_protocol.get("generator") if isinstance(source_protocol, Mapping) else None
    generators_by_model = family_plan.get("protocol", {}).get("generators_by_model", {})
    if (not isinstance(generator, Mapping) or model.get("generator") != generator
            or not isinstance(generators_by_model, Mapping)
            or generators_by_model.get(DITTO_MODEL_ID) != generator):
        raise DittoSPSAAdapterError("family Ditto descriptor is not the exact source-plan generator protocol")
    if execution.get("source_model_execution", {}).get(DITTO_MODEL_ID) != source_plan.get("execution"):
        raise DittoSPSAAdapterError("family plan changed the actual Ditto runtime/source execution record")
    if source_plan.get("scientific_input_pins_sha256") != family_plan.get("scientific_input_pins_sha256"):
        raise DittoSPSAAdapterError("Ditto source plan differs from the family's scientific input identity")


class DittoSPSAStageAAdapter:
    """Render only Ditto's ordered 8-row proposal or 40-row center subset."""

    def __init__(self, *, core: Any, ditto_overlay: Any,
                 family_plan: Mapping[str, Any], resource_helper: Any, args: Any):
        self.core = core
        self.overlay = ditto_overlay
        self.plan = family_plan
        self.args = args
        self.identity_rows = _validate_family_plan(family_plan)
        (self.source_plan, self.source_plan_raw, self.source_plan_path,
         self.source_rows, self.source_rows_raw, self.source_rows_path) = _load_source_ditto_plan(family_plan)
        _validate_composed_source_plan(family_plan, self.source_plan)
        self.binding, self.binding_raw, self.binding_path = _load_runtime_binding(
            family_plan, self.source_plan, self.source_plan_path)
        self.model, self.generator, self.renderer, self.source_and_weights = self._validate_source_runtime()
        self.resource_helper = resource_helper
        helper_pin = self.binding.get("resources", {}).get("resource_helper", {})
        _module_file_sha(resource_helper, helper_pin, "Ditto resource helper")
        project_root = Path(str(getattr(args, "project_root", ""))).resolve(strict=True)
        self.overlay._validate_scoring_runtime(args, project_root, self.source_plan)
        self._validate_runtime_environment_inputs()

    def _validate_source_runtime(self) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
        execution = self.source_plan.get("execution", {})
        parent_plan_ref = execution.get("parent_w2l_plan")
        parent_rows_ref = execution.get("parent_w2l_planned_rows")
        projection = self.source_plan.get("ditto_fit_projection", {})
        if not isinstance(parent_plan_ref, Mapping) or not isinstance(parent_rows_ref, Mapping):
            raise DittoSPSAAdapterError("frozen Ditto source plan lacks its W2L projection parent refs")
        parent, parent_raw, parent_path, parent_rows, parent_rows_raw, _ = self.overlay._load_frozen_parent(
            str(parent_plan_ref.get("path")), str(parent_plan_ref.get("raw_sha256")),
            str(parent_rows_ref.get("path")), str(parent_rows_ref.get("raw_sha256")))
        if (parent_plan_ref.get("bytes") != len(parent_raw)
                or parent_rows_ref.get("bytes") != len(parent_rows_raw)
                or projection.get("parent_plan_path") != str(parent_path)
                or projection.get("parent_rows_path") != str(Path(parent_rows_ref["path"]).resolve(strict=True))):
            raise DittoSPSAAdapterError("Ditto projection refs differ from the exact parent files")
        core_path, core_sha = _module_file_sha(self.core, execution.get("stagea_runner", {}), "Stage-A core")
        overlay_path, overlay_sha = _module_file_sha(self.overlay, execution.get("ditto_block_overlay", {}),
                                                     "Ditto block overlay")
        model, generator, renderer, source_and_weights = self.overlay._validate_model_and_binding(
            self.source_plan, self.binding, self.binding_raw, self.binding_path,
            parent, parent_raw, parent_path, parent_rows_raw, core_path, core_sha,
            overlay_path, overlay_sha)
        self.overlay._verify_parent_and_projection(
            self.source_plan, parent, parent_rows,
            parent_plan_raw=parent_raw, parent_rows_raw=parent_rows_raw)
        helper_ref = self.binding.get("resources", {}).get("resource_helper", {})
        project_root = Path(str(getattr(self.args, "project_root", ""))).resolve(strict=True)
        expected_helper = project_root / self.overlay.RESOURCE_HELPER_REL
        if (expected_helper.is_symlink() or not expected_helper.is_file()
                or expected_helper.stat(follow_symlinks=False).st_size != self.overlay.RESOURCE_HELPER_BYTES
                or _sha256_file(expected_helper) != self.overlay.RESOURCE_HELPER_SHA256
                or helper_ref.get("path") != str(expected_helper)
                or helper_ref.get("bytes") != self.overlay.RESOURCE_HELPER_BYTES
                or helper_ref.get("sha256") != self.overlay.RESOURCE_HELPER_SHA256):
            raise DittoSPSAAdapterError("resource helper differs from the live Ditto binding")
        return dict(model), dict(generator), renderer, source_and_weights

    def _validate_runtime_environment_inputs(self) -> None:
        if not hasattr(self.args, "generator_timeout") or not hasattr(self.args, "ffmpeg_timeout"):
            raise DittoSPSAAdapterError("Stage-A args omit frozen renderer or FFmpeg timeout fields")
        if (str(getattr(self.args, "project_root", "")) != "/home/wjj/tts-audio/tts-exp"
                or self.binding.get("gpu_uuid") != self.source_plan.get("execution", {}).get("gpu_uuid")):
            raise DittoSPSAAdapterError("Ditto project root or V100 identity differs from source plan and binding")

    def _validate_rows(self, rows: Sequence[Mapping[str, Any]], method: Mapping[str, Any]):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise DittoSPSAAdapterError("candidate rows must be an ordered sequence")
        config, config_sha = _method_config(method)
        snapshot = _verify_snapshot_lineage(self.plan, config)
        expected_samples = _expected_sample_ids(self.plan, config)
        if len(rows) != len(expected_samples) or len(rows) not in (8, 40):
            raise DittoSPSAAdapterError("Ditto subset must contain exactly eight proposal or forty center rows")
        if (config.get("method") == "fit40_spsa_proposal") != (len(rows) == 8):
            raise DittoSPSAAdapterError("SPSA method kind differs from the 8/40 row subset")
        observed = [row.get("sample_id") for row in rows]
        if tuple(observed) != expected_samples or len(set(observed)) != len(observed):
            raise DittoSPSAAdapterError("Ditto candidate rows differ from exact ordered SPSA sample roster")
        parents: list[Mapping[str, Any]] = []
        input_refs: list[Mapping[str, Any]] = []
        row_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or row.get("model_id") != DITTO_MODEL_ID:
                raise DittoSPSAAdapterError("Ditto adapter accepts only Ditto model rows")
            parent = self.identity_rows.get(_identity_key(row))
            if parent is None:
                raise DittoSPSAAdapterError("candidate row has no exact Ditto identity parent")
            row_binding = _validate_spsa_row_binding(row, parent, method, config, config_sha)
            input_ref, _natural_actual = _validate_generator_input(
                row, parent, self.core, row_binding, config, config_sha, snapshot)
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or row_id in row_ids:
                raise DittoSPSAAdapterError("candidate subset has a missing or duplicate row ID")
            row_ids.add(row_id)
            parents.append(parent)
            input_refs.append(input_ref)
        return parents, input_refs, config, config_sha

    def render_subset(self, *, candidate_rows: Sequence[Mapping[str, Any]],
                      method: Mapping[str, Any], output_root: str | Path) -> list[dict[str, Any]]:
        """Render and officially score one eight-row proposal or 40-row center.

        Each row is appended only after native render, lossless natural-audio
        delivery, source-clock checks, and official Stage-A SyncNet scoring all
        return PASS.  On failure, the raised subset exception retains every
        already completed row and names the failed/unattempted IDs.
        """
        parents, input_refs, config, config_sha = self._validate_rows(candidate_rows, method)
        output_root = Path(output_root)
        if not output_root.is_absolute() or output_root.is_symlink() or not output_root.is_dir():
            raise DittoSPSAAdapterError("subset output root must be an existing real directory")
        for row in candidate_rows:
            row_id = row.get("row_id")
            if (not isinstance(row_id, str) or Path(row_id).name != row_id or row_id in {".", ".."}
                    or (output_root / row_id).exists() or (output_root / row_id).is_symlink()):
                raise DittoSPSAAdapterError("candidate row path is unsafe or already exists; refusing replay")
        if not _PATCH_LOCK.acquire(blocking=False):
            raise DittoSPSAAdapterError("another per-model Stage-A adapter call is active in this process")

        original_generator = self.core._run_generator
        original_frame_clock = self.core._frame_clock
        expected_by_row_root: dict[str, int] = {}
        processed = {"count": 0}
        gate_by_row: dict[str, Mapping[str, Any]] = {}
        output: list[dict[str, Any]] = []
        saved_env = {key: value for key, value in os.environ.items() if key.startswith("CROSS_TFG_")}
        try:
            for key in list(os.environ):
                if key.startswith("CROSS_TFG_"):
                    os.environ.pop(key, None)
            os.environ["CROSS_TFG_CAPACITY_STAGE"] = "fit40_ditto240"
            os.environ["CROSS_TFG_RUNTIME_BINDING_PATH"] = str(self.binding_path)
            os.environ["CROSS_TFG_RUNTIME_BINDING_RAW_SHA256"] = _sha256(self.binding_raw)
            os.environ["PYTHONNOUSERSITE"] = "1"
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

            def ditto_generator(**kwargs: Any) -> dict[str, Any]:
                row = kwargs.get("row")
                if (not isinstance(row, Mapping) or row.get("model_id") != DITTO_MODEL_ID
                        or kwargs.get("model", {}).get("model_id") != DITTO_MODEL_ID):
                    raise DittoSPSAAdapterError("Stage-A generator hook received a non-Ditto row/model")
                return self.overlay._run_ditto_generator(
                    core=self.core, generator=self.generator,
                    expected_source_and_weights=self.source_and_weights,
                    expected_frames_by_row_root=expected_by_row_root,
                    processed_rows=processed, **kwargs)

            def ditto_frame_clock(path: Path, ffprobe: str, fps_num: int, fps_den: int):
                frames, stream = original_frame_clock(path, ffprobe, fps_num, fps_den)
                row_root = str(Path(path).resolve(strict=False).parent.parent)
                expected = expected_by_row_root.get(row_root)
                if expected is not None:
                    if len(frames) != expected:
                        raise DittoSPSAAdapterError("decoded Ditto frame count differs from ceil(N*25/16000)")
                    for index, frame in enumerate(frames):
                        if (frame.get("frame_index") != index
                                or not self.overlay.timestamp_matches_frame(frame, index, 25, 1)):
                            raise DittoSPSAAdapterError("decoded Ditto frame PTS differs from the native 25 fps clock")
                return frames, stream

            self.core._run_generator = ditto_generator
            self.core._frame_clock = ditto_frame_clock
            for index, (row, parent, input_ref) in enumerate(zip(candidate_rows, parents, input_refs)):
                remaining = copy.copy(dict(self.source_plan))
                remaining["rows"] = [copy.deepcopy(dict(item)) for item in candidate_rows[index:]]
                gate = self.overlay._capacity_observation(
                    remaining, self.binding, self.resource_helper, output_root, 0, whole_stage=False)
                if not isinstance(gate, Mapping) or gate.get("status") != "PASS":
                    raise DittoSPSASubsetFailure(
                        f"Ditto live resource gate failed before row {row['row_id']}",
                        completed=output, failed_row_id=str(row["row_id"]),
                        unattempted_row_ids=[str(item["row_id"]) for item in candidate_rows[index:]])
                gate_by_row[str(row["row_id"])] = dict(gate)
                natural_path, natural_samples, natural_actual = self.core._reference(row["natural_pcm"])
                if (row.get("score_audio") != row.get("natural_pcm")
                        or natural_actual.get("pcm_sha256") != row["natural_pcm"].get("pcm_sha256")):
                    raise DittoSPSAAdapterError("candidate score audio is not the exact natural PCM")
                expected_frames = expected_native_frame_count(int(natural_ref_count := row["natural_pcm"]["sample_count"]))
                row_root = output_root / str(row["row_id"])
                render_method = copy.deepcopy(dict(method))
                render_config = copy.deepcopy(dict(config))
                render_config["method"] = "precomputed_fit_snapshot_waveform"
                render_method["config"] = render_config
                if render_method.get("config_sha256") != config_sha:
                    raise DittoSPSAAdapterError("private dispatcher marker changed the serialized method pin")
                expected_by_row_root[str(row_root.resolve(strict=False))] = expected_frames
                try:
                    result = self.core._render_and_score(
                        row, self.model, render_method, natural_path, natural_samples,
                        natural_actual, row_root, self.args, qc_repeat=False)
                except Exception as exc:
                    raise DittoSPSASubsetFailure(
                        f"Ditto Stage-A row {row['row_id']} failed: {type(exc).__name__}: {exc}",
                        completed=output, failed_row_id=str(row["row_id"]),
                        unattempted_row_ids=[str(item["row_id"]) for item in candidate_rows[index + 1:]]) from exc
                if not isinstance(result, dict) or result.get("status") != "PASS":
                    raise DittoSPSAAdapterError("Stage-A Ditto row did not return PASS")
                if result.get("score_audio_sha256") != row["natural_pcm"].get("pcm_sha256"):
                    raise DittoSPSAAdapterError("Stage-A SyncNet score audio differs from the exact natural PCM")
                if processed["count"] != index + 1:
                    raise DittoSPSAAdapterError("Stage-A did not execute exactly one Ditto renderer for the row")
                strict_dir = row_root / "runtime_determinism_receipts"
                receipts = [path for path in strict_dir.iterdir() if path.is_file() and not path.is_symlink()] \
                    if strict_dir.is_dir() and not strict_dir.is_symlink() else []
                generation = result.get("render", {}).get("generation", {})
                receipt_meta = generation.get("strict_startup_receipt") if isinstance(generation, Mapping) else None
                if len(receipts) != 1 or not isinstance(receipt_meta, Mapping):
                    raise DittoSPSAAdapterError("Ditto row did not leave exactly one strict startup receipt")
                arrays = result.get("_internal_arrays")
                if not isinstance(arrays, Mapping):
                    raise DittoSPSAAdapterError("Stage-A omitted official source-clock/SyncNet arrays")
                array_summary = _validate_official_arrays(arrays)
                public = {key: value for key, value in result.items() if not key.startswith("_internal_")}
                public["spsa_binding"] = copy.deepcopy(dict(row["spsa_binding"]))
                public["spsa_point_binding_sha256"] = row["spsa_point_binding_sha256"]
                public["generator_condition_input"] = copy.deepcopy(dict(input_ref))
                public["parent_identity_row_id"] = parent["row_id"]
                output.append({
                    "row": public,
                    "official_arrays": arrays,
                    "row_output_root": str(row_root.resolve(strict=True)),
                    "parent_identity_row_id": parent["row_id"],
                    "application_receipt": copy.deepcopy(dict(row["spsa_binding"]["application_receipt"])),
                    "method_config_sha256": config_sha,
                    "resource_gate": gate_by_row[str(row["row_id"])],
                    "official_array_summary": array_summary,
                })
            return output
        finally:
            self.core._run_generator = original_generator
            self.core._frame_clock = original_frame_clock
            for key in list(os.environ):
                if key.startswith("CROSS_TFG_"):
                    os.environ.pop(key, None)
            os.environ.update(saved_env)
            _PATCH_LOCK.release()


__all__ = ["DittoSPSAAdapterError", "DittoSPSASubsetFailure",
           "DittoSPSAStageAAdapter", "expected_native_frame_count"]
