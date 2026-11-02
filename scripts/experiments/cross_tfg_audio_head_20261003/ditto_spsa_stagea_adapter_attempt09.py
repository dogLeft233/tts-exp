"""Ditto-only Stage-A hooks for shared Fit40 SPSA proposal and center WAVs.

The coordinator applies the selected snapshot once per sample and creates
separate model-projection receipts for the three backend identity rows.  This
adapter independently validates the Ditto projection through the pinned
shared application validator, then invokes the existing Stage-A
``_render_and_score`` one row at a time.  Ditto receives the shared transformed
WAV; official SyncNet scoring and lossless audio delivery continue to use the
original natural PCM.  No optimizer, selection, or retry is performed here.

The constructor takes an immutable, self-hashed single-method point plan and
its exact file pin.  Its embedded ``spsa_execution_parent_plan`` binds the
separate, complete 720-row six-method family.  The point plan is passed to the
shared waveform validator; the family plan supplies identity rows and native
model/runtime provenance.  Only a private method copy receives the Stage-A
dispatcher marker ``precomputed_fit_snapshot_waveform``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
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
APPLICATION_VALIDATOR_REL = Path("scripts/audio_enhancement/fit40_snapshot_spsa_application_attempt06.py")
APPLICATION_VALIDATOR_BYTES = 64266
APPLICATION_VALIDATOR_SHA256 = "6c9b4fbfe13999f37625d73bc7e302dfdc16b8e970371f28010cff10ce65548e"
APPLICATION_VALIDATOR_MODULE = "fit40_snapshot_spsa_application_attempt06_pinned"
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


def _load_application_validator(project_root: Path) -> tuple[Any, dict[str, Any]]:
    """Load the exact shared snapshot/PCM validator from the persistent source project."""
    root = project_root.resolve(strict=True)
    path = root
    for part in APPLICATION_VALIDATOR_REL.parts:
        path = path / part
        if path.is_symlink():
            raise DittoSPSAAdapterError("shared snapshot validator path traverses a symlink")
    if not path.is_file():
        raise DittoSPSAAdapterError("shared snapshot validator is absent or symlinked")
    raw = _read_regular(path, limit=2 * 1024 * 1024)
    digest = _sha256(raw)
    if len(raw) != APPLICATION_VALIDATOR_BYTES or digest != APPLICATION_VALIDATOR_SHA256:
        raise DittoSPSAAdapterError("shared snapshot validator differs from its exact source pin")
    resolved = path.resolve(strict=True)
    existing = sys.modules.get(APPLICATION_VALIDATOR_MODULE)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if not isinstance(existing_path, str) or Path(existing_path).resolve(strict=True) != resolved:
            raise DittoSPSAAdapterError("shared snapshot validator module name is already bound elsewhere")
        module = existing
    else:
        spec = importlib.util.spec_from_file_location(APPLICATION_VALIDATOR_MODULE, resolved)
        if spec is None or spec.loader is None:
            raise DittoSPSAAdapterError("cannot load the exact shared snapshot validator")
        module = importlib.util.module_from_spec(spec)
        sys.modules[APPLICATION_VALIDATOR_MODULE] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(APPLICATION_VALIDATOR_MODULE, None)
            raise
    if (getattr(module, "__file__", None) is None
            or Path(module.__file__).resolve(strict=True) != resolved):
        raise DittoSPSAAdapterError("loaded shared validator origin differs from the pinned file")
    for name in ("apply_snapshot_theta_shared", "validate_shared_application_lineage",
                 "project_shared_application_to_model_row", "validate_model_projection_lineage"):
        if not callable(getattr(module, name, None)):
            raise DittoSPSAAdapterError(f"shared snapshot validator omits {name}")
    return module, {"path": str(resolved), "bytes": len(raw), "sha256": digest}


def expected_native_frame_count(sample_count: int, fps_num: int = 25,
                                fps_den: int = 1, sample_rate_hz: int = 16000) -> int:
    """Return Ditto's frozen ceil(N*fps/sample_rate) frame count."""
    if any(type(value) is not int or value <= 0
           for value in (sample_count, fps_num, fps_den, sample_rate_hz)):
        raise DittoSPSAAdapterError("sample and clock values must be positive integers")
    numerator = sample_count * fps_num
    denominator = sample_rate_hz * fps_den
    return (numerator + denominator - 1) // denominator


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
    return identities


def _load_pinned_plan(pin: Mapping[str, Any], *, label: str,
                      expected_object: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], bytes, Path]:
    """Load a self-hashed Stage-A plan through an exact raw-byte pin."""
    if not isinstance(pin, Mapping):
        raise DittoSPSAAdapterError(f"{label} pin must be an object")
    path, raw = _read_pinned_file(pin, label, limit=32 * 1024 * 1024)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DittoSPSAAdapterError(f"{label} is not valid UTF-8 JSON") from exc
    if (not isinstance(value, dict) or value.get("schema_version") != PLAN_SCHEMA
            or value.get("status") != "FROZEN" or not _self_hash_matches(value, "plan_sha256")):
        raise DittoSPSAAdapterError(f"{label} is not a frozen self-hashed Stage-A plan")
    if pin.get("self_sha256") is not None and pin.get("self_sha256") != value.get("plan_sha256"):
        raise DittoSPSAAdapterError(f"{label} self-hash differs from its declared pin")
    if expected_object is not None and _canonical(value) != _canonical(expected_object):
        raise DittoSPSAAdapterError(f"{label} file bytes decode to a different in-memory plan")
    return value, raw, path


def _validate_point_plan(point_plan: Mapping[str, Any], point_plan_pin: Mapping[str, Any]
                         ) -> tuple[dict[str, Any], bytes, Path,
                                    dict[str, Any], bytes, Path,
                                    dict[tuple[str, str], Mapping[str, Any]]]:
    """Bind a one-method proposal/center plan to its complete frozen family.

    Returns the exact 720-row family plan, the point/family raw files and paths,
    and the derived point rows indexed by (sample, model).  The raw point pin
    is an explicit constructor input so the adapter never certifies an
    untraceable in-memory plan object.
    """
    if not isinstance(point_plan, Mapping):
        raise DittoSPSAAdapterError("SPSA point plan must be an object")
    point, point_raw, point_path = _load_pinned_plan(
        point_plan_pin, label="derived SPSA point plan", expected_object=point_plan)
    point_exec = point.get("execution")
    family_ref = point_exec.get("spsa_execution_parent_plan") if isinstance(point_exec, Mapping) else None
    family, family_raw, family_path = _load_pinned_plan(
        family_ref, label="frozen 720-row SPSA parent family")
    identities = _validate_family_plan(family)

    methods = point.get("methods")
    models = point.get("models")
    rows = point.get("rows")
    execution = point.get("execution")
    if (not isinstance(methods, list) or len(methods) != 1 or not isinstance(methods[0], Mapping)
            or not isinstance(models, list) or not isinstance(rows, list)
            or not isinstance(execution, Mapping)):
        raise DittoSPSAAdapterError("derived point plan must contain one method, model roster, rows, and execution")
    method = methods[0]
    config = method.get("config")
    method_id = method.get("method_id")
    expected_schema = {
        "fit40_spsa_proposal": "fit40_spsa_proposal_transform_v1",
        "fit40_spsa_center": "fit40_spsa_center_transform_v1",
    }
    if (not isinstance(config, Mapping) or config.get("method") not in SUPPORTED_METHODS
            or not isinstance(method_id, str) or not method_id
            or config.get("method_id") != method_id
            or config.get("schema") != expected_schema.get(config.get("method"))
            or method.get("kind") != config.get("method")
            or method.get("checkpoint_sha256") is not None
            or method.get("config_sha256") != _sha256(_canonical(config))):
        raise DittoSPSAAdapterError("derived point plan method/config does not have its exact SPSA point identity")
    if (point.get("schema_version") != PLAN_SCHEMA or point.get("status") != "FROZEN"
            or not _self_hash_matches(point, "plan_sha256")
            or point.get("scientific_input_pins_sha256") != family.get("scientific_input_pins_sha256")
            or point.get("scientific_input_pins") != family.get("scientific_input_pins")
            or point.get("models") != family.get("models")
            or point.get("protocol") != family.get("protocol")
            or point.get("supports") != family.get("supports")
            or point.get("generation_repeat_qc") != []):
        raise DittoSPSAAdapterError("derived point plan changed frozen family model/protocol/input/support identity")
    allowed_execution_changes = {
        "planned_rows", "spsa_execution_parent_plan", "spsa_execution_scope", "runtime_readiness",
    }
    for key, value in family["execution"].items():
        if key not in allowed_execution_changes and execution.get(key) != value:
            raise DittoSPSAAdapterError(f"derived point plan changed frozen family execution field {key}")
    if (execution.get("planned_rows") != len(rows)
            or execution.get("model_ids") != list(EXPECTED_MODEL_ORDER)
            or execution.get("dev_or_confirm_access") is not False
            or execution.get("candidate_selection") != "not performed"
            or execution.get("runtime_readiness") != "REQUIRED_SEPARATE_SPSA_BINDING_BEFORE_EXECUTION"):
        raise DittoSPSAAdapterError("derived point execution metadata differs from its single-method scope")
    if (config.get("input_pins_sha256") != family.get("scientific_input_pins_sha256")
            or config.get("artifact_sha256") != config.get("source_snapshot_artifact_sha256")):
        raise DittoSPSAAdapterError("derived point method changed source snapshot or input-pins identity")
    scope = execution.get("spsa_execution_scope")
    analysis = point.get("analysis")
    if (not isinstance(scope, Mapping) or not isinstance(analysis, Mapping)
            or analysis.get("selection_status") != "NOT_SELECTED"
            or analysis.get("scope") != "fit_only_spsa_execution_packet"
            or analysis.get("spsa_execution_scope") != scope
            or point.get("run_id") != scope.get("run_id")
            or scope.get("method_id") != method_id
            or scope.get("input_pins_sha256") != family.get("scientific_input_pins_sha256")
            or scope.get("artifact_sha256") != config.get("artifact_sha256")
            or scope.get("binding_sha256") != config.get("binding_sha256")):
        raise DittoSPSAAdapterError("derived point scope or analysis metadata differs from its method binding")
    if config.get("method") == "fit40_spsa_proposal":
        if scope.get("scope_kind") != "spsa_pair_proposal":
            raise DittoSPSAAdapterError("proposal execution scope has the wrong scope kind")
        # The frozen composer04 producer intentionally emits the identity fields
        # below in spsa_execution_scope; its method config additionally carries
        # c_k, but the scope schema has no c_k key. Validate c_k on the method
        # config itself against the frozen schedule instead of requiring a
        # producer-side duplicate alias that is absent from actual packets.
        for field in ("proposal_id", "pair_id", "iteration", "side", "batch_id"):
            if scope.get(field) != config.get(field):
                raise DittoSPSAAdapterError(f"proposal execution scope differs from method field {field}")
        iteration = config.get("iteration")
        c_k = config.get("c_k")
        if (type(iteration) is not int or not 0 <= iteration < 20
                or isinstance(c_k, bool) or not isinstance(c_k, (int, float))
                or not math.isfinite(float(c_k)) or float(c_k) <= 0.0
                or float(c_k) != 0.25 / ((iteration + 1) ** 0.101)):
            raise DittoSPSAAdapterError("proposal method config c_k differs from the frozen SPSA schedule")
    elif (scope.get("scope_kind") != "spsa_full_center"
            or scope.get("point_kind") != "spsa_center"
            or scope.get("source_step") != config.get("source_step")):
        raise DittoSPSAAdapterError("center execution scope differs from its source center point")
    seed_method_id = config.get("source_seed_method_id")
    seed_methods = [item for item in family.get("methods", [])
                    if isinstance(item, Mapping) and item.get("method_id") == seed_method_id]
    if len(seed_methods) != 1:
        raise DittoSPSAAdapterError("derived point does not bind one exact source snapshot seed method")
    seed_config = seed_methods[0].get("config")
    if (not isinstance(seed_config, Mapping)
            or seed_config.get("method") != "precomputed_fit_snapshot_waveform"
            or seed_config.get("source_optimizer_step") != config.get("source_snapshot_step")
            or seed_config.get("snapshot_sha256") != config.get("source_snapshot_artifact_sha256")
            or seed_config.get("training_family") != config.get("head_family")):
        raise DittoSPSAAdapterError("derived point snapshot lineage differs from its exact family seed method")

    expected_samples = _expected_sample_ids(point, config)
    expected_row_count = len(expected_samples) * len(EXPECTED_MODEL_ORDER)
    if len(rows) != expected_row_count:
        raise DittoSPSAAdapterError("derived point plan does not contain its complete 8/40 × 3-model subset")
    if (config.get("method") == "fit40_spsa_proposal" and len(expected_samples) != 8
            or config.get("method") == "fit40_spsa_center" and len(expected_samples) != 40):
        raise DittoSPSAAdapterError("derived point plan sample denominator differs from proposal/center scope")
    if scope.get("sample_ids") != list(expected_samples):
        raise DittoSPSAAdapterError("derived point execution scope differs from the exact method sample roster")
    family_identity_by_sample = {
        str(row.get("sample_id")): row
        for row in family["rows"]
        if (isinstance(row, Mapping) and row.get("model_id") == EXPECTED_MODEL_ORDER[0]
            and row.get("method_id") == "identity")
    }
    if any(sample not in family_identity_by_sample for sample in expected_samples):
        raise DittoSPSAAdapterError("point sample roster contains a sample outside the frozen source roster")
    expected_speakers = list(dict.fromkeys(
        str(family_identity_by_sample[sample].get("speaker_id")) for sample in expected_samples
    ))
    if scope.get("speaker_ids") != expected_speakers:
        raise DittoSPSAAdapterError("derived point execution scope differs from the exact ordered speaker roster")
    if config.get("method") == "fit40_spsa_proposal":
        speakers = [str(family_identity_by_sample[sample].get("speaker_id")) for sample in expected_samples]
        if len(set(speakers)) != 4 or any(speakers.count(speaker) != 2 for speaker in set(speakers)):
            raise DittoSPSAAdapterError("proposal subset must preserve four speakers with two clips each")

    family_identity_by_context = {
        (str(row.get("sample_id")), str(row.get("model_id"))): row
        for row in family.get("rows", [])
        if isinstance(row, Mapping) and row.get("method_id") == "identity"
    }
    point_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_order = [(sample, model_id) for sample in expected_samples for model_id in EXPECTED_MODEL_ORDER]
    observed_order = []
    row_ids = set()
    for row in rows:
        if (not isinstance(row, Mapping) or row.get("method_id") != method_id
                or row.get("method_config_sha256") != method.get("config_sha256")
                or row.get("checkpoint_sha256") is not None
                or row.get("spsa_point_binding_sha256") != config.get("binding_sha256")):
            raise DittoSPSAAdapterError("derived point row changed its method/config/point binding")
        sample = row.get("sample_id")
        model_id = row.get("model_id")
        key = (str(sample), str(model_id))
        parent = family_identity_by_context.get(key)
        row_id = row.get("row_id")
        if (parent is None or not isinstance(row_id, str) or not row_id or row_id == parent.get("row_id")
                or row_id in row_ids or _stable_row(row) != _stable_row(parent)):
            raise DittoSPSAAdapterError("derived point row is not an exact source identity context")
        row_ids.add(row_id)
        point_rows[key] = row
        observed_order.append(key)
    if observed_order != expected_order or len(point_rows) != expected_row_count:
        raise DittoSPSAAdapterError("derived point rows are not the exact ordered sample × model subset")
    # Ensure the Ditto identity map returned by the family validator names the
    # same source contexts; this guards accidental remapping by duplicate IDs.
    if any(_identity_key(family_identity_by_context[(sample, DITTO_MODEL_ID)]) not in identities
           for sample in expected_samples):
        raise DittoSPSAAdapterError("derived point sample roster is not covered by exact Ditto identity rows")
    return point, point_raw, point_path, family, family_raw, family_path, point_rows


def _validate_composed_source_plan(family_plan: Mapping[str, Any], source_plan: Mapping[str, Any]) -> None:
    models = family_plan["models"]
    model = next(item for item in models if item.get("model_id") == DITTO_MODEL_ID)
    execution = family_plan["execution"]
    source_protocol = source_plan.get("protocol")
    protocol_generator = source_protocol.get("generator") if isinstance(source_protocol, Mapping) else None
    source_models = [item for item in source_plan.get("models", [])
                     if isinstance(item, Mapping) and item.get("model_id") == DITTO_MODEL_ID]
    source_model_descriptor = source_models[0].get("generator") if len(source_models) == 1 else None
    generators_by_model = family_plan.get("protocol", {}).get("generators_by_model", {})
    if (not isinstance(source_model_descriptor, Mapping)
            or model.get("generator") != source_model_descriptor):
        raise DittoSPSAAdapterError(
            "family Ditto runtime descriptor differs from the exact source-plan model descriptor")
    if (not isinstance(protocol_generator, Mapping)
            or not isinstance(generators_by_model, Mapping)
            or generators_by_model.get(DITTO_MODEL_ID) != protocol_generator):
        raise DittoSPSAAdapterError(
            "family Ditto per-model protocol differs from the exact source-plan protocol generator")
    if execution.get("source_model_execution", {}).get(DITTO_MODEL_ID) != source_plan.get("execution"):
        raise DittoSPSAAdapterError("family plan changed the actual Ditto runtime/source execution record")
    if source_plan.get("scientific_input_pins_sha256") != family_plan.get("scientific_input_pins_sha256"):
        raise DittoSPSAAdapterError("Ditto source plan differs from the family's scientific input identity")


class DittoSPSAStageAAdapter:
    """Render only Ditto's ordered 8-row proposal or 40-row center subset."""

    def __init__(self, *, core: Any, ditto_overlay: Any,
                 point_plan: Mapping[str, Any], point_plan_pin: Mapping[str, Any],
                 resource_helper: Any, args: Any):
        self.core = core
        self.overlay = ditto_overlay
        (self.point_plan, self.point_plan_raw, self.point_plan_path,
         self.family_plan, self.family_plan_raw, self.family_plan_path,
         self.point_rows) = _validate_point_plan(point_plan, point_plan_pin)
        self.plan = self.point_plan
        self.point_plan_pin = {
            "path": str(self.point_plan_path), "bytes": len(self.point_plan_raw),
            "sha256": _sha256(self.point_plan_raw),
            "self_sha256": self.point_plan["plan_sha256"],
        }
        self.family_plan_pin = {
            "path": str(self.family_plan_path), "bytes": len(self.family_plan_raw),
            "sha256": _sha256(self.family_plan_raw),
            "self_sha256": self.family_plan["plan_sha256"],
        }
        self.args = args
        self.identity_rows = _validate_family_plan(self.family_plan)
        self.identity_rows_by_sample_model = {
            (str(row.get("sample_id")), str(row.get("model_id"))): row
            for row in self.family_plan.get("rows", [])
            if isinstance(row, Mapping) and row.get("method_id") == "identity"
        }
        (self.source_plan, self.source_plan_raw, self.source_plan_path,
         self.source_rows, self.source_rows_raw, self.source_rows_path) = _load_source_ditto_plan(self.family_plan)
        _validate_composed_source_plan(self.family_plan, self.source_plan)
        self.binding, self.binding_raw, self.binding_path = _load_runtime_binding(
            self.family_plan, self.source_plan, self.source_plan_path)
        self.model, self.generator, self.renderer, self.source_and_weights = self._validate_source_runtime()
        parent_plan_ref = self.source_plan.get("execution", {}).get("parent_w2l_plan", {})
        parent_plan_path = Path(str(parent_plan_ref.get("path", "")))
        if not parent_plan_path.is_absolute() or parent_plan_path.is_symlink():
            raise DittoSPSAAdapterError("snapshot project root must come from the exact absolute W2L parent plan")
        parent_plan_path = parent_plan_path.resolve(strict=True)
        self.snapshot_project_root = parent_plan_path.parent
        if self.snapshot_project_root.is_symlink() or not self.snapshot_project_root.is_dir():
            raise DittoSPSAAdapterError("snapshot project root must be the real staged W2L project directory")
        source_project_root = Path(str(getattr(args, "project_root", ""))).resolve(strict=True)
        self.application_validator, self.application_validator_pin = _load_application_validator(
            source_project_root)
        self.resource_helper = resource_helper
        helper_pin = self.binding.get("resources", {}).get("resource_helper", {})
        _module_file_sha(resource_helper, helper_pin, "Ditto resource helper")
        project_root = Path(str(getattr(args, "project_root", ""))).resolve(strict=True)
        self.overlay._validate_scoring_runtime(args, project_root, self.source_plan)
        self._validate_runtime_environment_inputs()

    def validation_source_pins(self) -> dict[str, dict[str, Any]]:
        """Expose validator and both plan lineage pins for the block manifest."""
        return {
            "fit_snapshot_spsa_shared_application_validator": copy.deepcopy(self.application_validator_pin),
            "spsa_point_plan": copy.deepcopy(self.point_plan_pin),
            "spsa_parent_family_plan": copy.deepcopy(self.family_plan_pin),
        }

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
        config = method.get("config")
        config_sha = method.get("config_sha256")
        if (not isinstance(config, Mapping) or not isinstance(config_sha, str)
                or _sha256(_canonical(config)) != config_sha
                or self.point_plan.get("methods") != [dict(method)]):
            raise DittoSPSAAdapterError("candidate method differs from the exact self-pinned point-plan method")
        expected_samples = _expected_sample_ids(self.point_plan, config)
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
            planned_row = self.point_rows.get((str(row.get("sample_id")), DITTO_MODEL_ID))
            if (planned_row is None or row.get("row_id") != planned_row.get("row_id")
                    or row.get("method_id") != method.get("method_id")
                    or row.get("method_config_sha256") != config_sha
                    or row.get("spsa_point_binding_sha256") != config.get("binding_sha256")):
                raise DittoSPSAAdapterError(
                    "candidate row differs from its exact derived point-plan row/method binding"
                )
            if _stable_row(row) != _stable_row(parent):
                raise DittoSPSAAdapterError(
                    "candidate row altered frozen source, face, split, speaker, or clock metadata"
                )
            parent_rows_by_model = {
                model_id: self.identity_rows_by_sample_model.get((str(row.get("sample_id")), model_id))
                for model_id in EXPECTED_MODEL_ORDER
            }
            if any(not isinstance(value, Mapping) for value in parent_rows_by_model.values()):
                raise DittoSPSAAdapterError("candidate sample lacks one or more exact three-model family parents")
            spsa_binding = row.get("spsa_binding")
            if not isinstance(spsa_binding, Mapping):
                raise DittoSPSAAdapterError("candidate row omits its shared-application and model-projection bindings")
            shared_pin = spsa_binding.get("shared_application_receipt")
            projection_pin = spsa_binding.get("model_projection_receipt")
            if not isinstance(shared_pin, Mapping) or not isinstance(projection_pin, Mapping):
                raise DittoSPSAAdapterError("candidate row lacks exact shared/projection receipt pins")
            try:
                lineage = self.application_validator.validate_model_projection_lineage(
                    project_root=self.snapshot_project_root,
                    family_plan=self.family_plan,
                    family_plan_pin=self.family_plan_pin,
                    point_plan=self.point_plan,
                    point_plan_pin=self.point_plan_pin,
                    method=method,
                    model_id=DITTO_MODEL_ID,
                    parent_rows_by_model=parent_rows_by_model,
                    parent_row=parent,
                    point_row=planned_row,
                    candidate_row=row,
                    shared_application_receipt_pin=shared_pin,
                    model_projection_receipt_pin=projection_pin,
                )
            except Exception as exc:
                raise DittoSPSAAdapterError(
                    f"shared application/projection validator rejected row {row.get('row_id')}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            input_ref = lineage.get("generator_condition_input")
            sample_count = lineage.get("sample_count")
            if not isinstance(input_ref, Mapping) or type(sample_count) is not int or sample_count <= 0:
                raise DittoSPSAAdapterError("shared application validator returned incomplete row lineage")
            frame_binding = row.get("ditto_native_frame_binding")
            if (not isinstance(frame_binding, Mapping)
                    or frame_binding.get("sample_count") != sample_count
                    or frame_binding.get("sample_rate_hz") != 16000
                    or frame_binding.get("fps_num") != 25
                    or frame_binding.get("fps_den") != 1
                    or frame_binding.get("expected_frame_count") != expected_native_frame_count(sample_count)):
                raise DittoSPSAAdapterError(
                    "Ditto native frame binding differs from ceil(N*25/16000) for this exact input"
                )
            natural_path, natural_samples, natural_actual = self.core._reference(row["natural_pcm"])
            natural_ref = row.get("natural_pcm", {})
            if (row.get("score_audio") != natural_ref
                    or natural_actual.get("file_sha256") != natural_ref.get("file_sha256")
                    or natural_actual.get("pcm_sha256") != natural_ref.get("pcm_sha256")
                    or natural_samples.size != sample_count
                    or Path(str(natural_ref.get("path", ""))).resolve(strict=True) != natural_path):
                raise DittoSPSAAdapterError(
                    "Stage-A natural PCM reader differs from the shared validator's pinned source lineage"
                )
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or row_id in row_ids:
                raise DittoSPSAAdapterError("candidate subset has a missing or duplicate row ID")
            row_ids.add(row_id)
            parents.append(parent)
            input_refs.append(dict(input_ref))
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
        tracked_env_keys = {
            key for key in os.environ
            if key.startswith("CROSS_TFG_") or key in {"PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"}
        }
        saved_env = {key: os.environ[key] for key in tracked_env_keys}
        try:
            for key in list(os.environ):
                if key.startswith("CROSS_TFG_") or key in {"PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"}:
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
                remaining = copy.deepcopy(dict(self.point_plan))
                remaining["rows"] = [copy.deepcopy(dict(item)) for item in candidate_rows[index:]]
                remaining["execution"]["planned_rows"] = len(remaining["rows"])
                remaining["plan_sha256"] = _sha256(_canonical({
                    key: value for key, value in remaining.items() if key != "plan_sha256"
                }))
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
                    "shared_application_receipt": copy.deepcopy(
                        dict(row["spsa_binding"]["shared_application_receipt"])),
                    "model_projection_receipt": copy.deepcopy(
                        dict(row["spsa_binding"]["model_projection_receipt"])),
                    "method_config_sha256": config_sha,
                    "resource_gate": gate_by_row[str(row["row_id"])],
                    "official_array_summary": array_summary,
                    "validation_source_pins": self.validation_source_pins(),
                })
            return output
        finally:
            self.core._run_generator = original_generator
            self.core._frame_clock = original_frame_clock
            for key in list(os.environ):
                if key.startswith("CROSS_TFG_") or key in {"PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"}:
                    os.environ.pop(key, None)
            os.environ.update(saved_env)
            _PATCH_LOCK.release()


__all__ = ["DittoSPSAAdapterError", "DittoSPSASubsetFailure",
           "DittoSPSAStageAAdapter", "expected_native_frame_count"]
