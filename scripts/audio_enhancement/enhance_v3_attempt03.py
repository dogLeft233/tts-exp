"""Pinned V3 waveform enhancement CLI, revision 2.

This entrypoint is intentionally separate from the existing spectral-head CLI.
It accepts only the shared-recipient V3 parameter/configuration contract. The
module performs no model loading or inference when imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import pathlib
import re
import stat
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any


SCHEMA = "cross_tfg_v3_enhancement_cli_config_v1"
HEAD_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3"
RESULT_SCHEMA = "cross_tfg_v3_enhancement_cli_result_v1"
CHECKPOINT_SCHEMA = "cross_tfg_v3_parameter_checkpoint_v1"
TRAINING_RECORD_SCHEMA = "cross_tfg_v3_candidate_training_record_v1"
SELECTION_RECORD_SCHEMA = "cross_tfg_v3_candidate_selection_record_v1"
ENGINEERING_ONLY = "ENGINEERING_ONLY"
METHOD_CANDIDATE = "METHOD_CANDIDATE"
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_PIN_KEYS = {"path", "bytes", "sha256"}
_PARAMETER_KEYS = {
    "name",
    "shape",
    "dtype",
    "temporal_components",
    "non_reference_band_order",
    "values",
}
_REQUIRED_RUNTIME_MODULES = {
    "scripts.audio_enhancement": "scripts/audio_enhancement/__init__.py",
    "scripts.audio_enhancement.audio_io": "scripts/audio_enhancement/audio_io.py",
    "scripts.audio_enhancement.baselines": "scripts/audio_enhancement/baselines.py",
    "scripts.audio_enhancement.pipeline": "scripts/audio_enhancement/pipeline.py",
    "scripts.audio_enhancement.overdub": "scripts/audio_enhancement/overdub.py",
    "scripts.audio_enhancement.shared_recipient_transition_head_v3": (
        "scripts/audio_enhancement/shared_recipient_transition_head_v3.py"
    ),
}

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ENTRYPOINT_RELATIVE = pathlib.Path(__file__).resolve().relative_to(_ROOT).as_posix()
sys.dont_write_bytecode = True


class V3CLIError(ValueError):
    """Invalid config, provenance, waveform, or output contract."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_json_object(path: pathlib.Path, label: str, *, max_bytes: int = 2 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise V3CLIError(f"cannot read {label}: {exc}") from exc
    if len(raw) > max_bytes:
        raise V3CLIError(f"{label} exceeds the {max_bytes}-byte JSON bound")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3CLIError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise V3CLIError(f"{label} must be a JSON object")
    return value, raw


def _decode_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3CLIError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise V3CLIError(f"{label} must be a JSON object")
    return value


def _reject_symlink_components(path: pathlib.Path) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    cursor = pathlib.Path(parts[0])
    for part in parts[1:]:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise V3CLIError(f"symlink path component is not accepted: {cursor}")


def _pinned_repo_path(pin: dict[str, Any], label: str) -> tuple[pathlib.Path, bytes]:
    if set(pin) != _PIN_KEYS:
        raise V3CLIError(f"{label} pin must contain exactly path, bytes, sha256")
    rel = pin.get("path")
    size = pin.get("bytes")
    expected_sha = pin.get("sha256")
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise V3CLIError(f"{label} path must be a nonempty repository-relative POSIX path")
    pure = pathlib.PurePosixPath(rel)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise V3CLIError(f"{label} path is not a safe repository-relative path")
    if type(size) is not int or size < 0:
        raise V3CLIError(f"{label} byte count must be a nonnegative integer")
    if not isinstance(expected_sha, str) or not _HEX_256.fullmatch(expected_sha):
        raise V3CLIError(f"{label} sha256 must be 64 lowercase hexadecimal characters")
    candidate = _ROOT.joinpath(*pure.parts)
    _reject_symlink_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(_ROOT)
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode):
            raise V3CLIError(f"{label} is not a regular file")
        raw = resolved.read_bytes()
        after = resolved.stat()
    except (OSError, ValueError) as exc:
        raise V3CLIError(f"cannot resolve/read pinned {label}: {exc}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise V3CLIError(f"{label} changed while it was being read")
    if len(raw) != size or _sha256(raw) != expected_sha:
        raise V3CLIError(f"{label} byte count or SHA-256 does not match its pin")
    return resolved, raw


def _validate_config(path: pathlib.Path) -> tuple[dict[str, Any], bytes, dict[str, dict[str, Any]]]:
    if path.is_symlink():
        raise V3CLIError("configuration path must not be a symlink")
    config_path = path.resolve(strict=True)
    config, config_raw = _read_json_object(config_path, "configuration", max_bytes=256 * 1024)
    if config.get("schema") != SCHEMA:
        raise V3CLIError(f"unsupported configuration schema: {config.get('schema')!r}")
    if config.get("head_schema") != HEAD_SCHEMA:
        raise V3CLIError("configuration head_schema does not match V3")

    implementation = config.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != _PIN_KEYS:
        raise V3CLIError("implementation must be a complete path/bytes/sha256 pin")
    pins = config.get("source_pins")
    if not isinstance(pins, dict) or not pins:
        raise V3CLIError("source_pins must be a nonempty object")
    pin_by_path: dict[str, dict[str, Any]] = {}
    for name, pin in pins.items():
        if not isinstance(name, str) or not isinstance(pin, dict):
            raise V3CLIError("source_pins entries must be named pin objects")
        if pin.get("path") in pin_by_path and pin_by_path[pin["path"]] != pin:
            raise V3CLIError(f"source_pins contains conflicting bindings for {pin['path']}")
        pin_by_path[pin["path"]] = pin
        _resolved, _raw = _pinned_repo_path(pin, f"source_pins.{name}")

    _implementation_path, implementation_raw = _pinned_repo_path(implementation, "implementation")
    if implementation["path"] not in pin_by_path:
        raise V3CLIError("the V3 implementation must also appear in source_pins")
    _implementation_in_pins_path, implementation_in_pins_raw = _pinned_repo_path(
        pin_by_path[implementation["path"]], "source_pins V3 implementation"
    )
    if implementation_in_pins_raw != implementation_raw:
        raise V3CLIError("implementation and source_pins bytes disagree")
    for module_path in _REQUIRED_RUNTIME_MODULES.values():
        if module_path not in pin_by_path:
            raise V3CLIError(f"runtime source closure is missing {module_path}")
    if implementation["path"] != _REQUIRED_RUNTIME_MODULES[
        "scripts.audio_enhancement.shared_recipient_transition_head_v3"
    ]:
        raise V3CLIError("implementation path is not the pinned V3 head module")
    if implementation["path"] not in pin_by_path or pin_by_path[implementation["path"]] != implementation:
        raise V3CLIError("implementation pin differs from its source_pins entry")
    if _ENTRYPOINT_RELATIVE not in pin_by_path:
        raise V3CLIError("source_pins must bind this exact CLI entrypoint")
    entry_pin = pin_by_path[_ENTRYPOINT_RELATIVE]
    if entry_pin["bytes"] != len(pathlib.Path(__file__).resolve().read_bytes()) or entry_pin[
        "sha256"
    ] != _sha256(pathlib.Path(__file__).resolve().read_bytes()):
        raise V3CLIError("CLI entrypoint source pin is stale")

    allowed_config_keys = {
        "schema",
        "profile_role",
        "head_schema",
        "implementation",
        "constructor",
        "parameter",
        "source_pins",
        "runtime",
        "input",
        "output",
        "candidate_provenance",
    }
    if set(config) - allowed_config_keys:
        raise V3CLIError(f"configuration contains unknown fields: {sorted(set(config) - allowed_config_keys)}")
    required_config_keys = allowed_config_keys - {"candidate_provenance"}
    if required_config_keys - set(config):
        raise V3CLIError(f"configuration is missing fields: {sorted(required_config_keys - set(config))}")

    constructor = config.get("constructor")
    parameter = config.get("parameter")
    runtime = config.get("runtime")
    input_contract = config.get("input")
    output_contract = config.get("output")
    if not isinstance(constructor, dict) or not isinstance(parameter, dict):
        raise V3CLIError("constructor and parameter must be objects")
    if not isinstance(runtime, dict) or not isinstance(input_contract, dict) or not isinstance(output_contract, dict):
        raise V3CLIError("runtime, input, and output contracts must be objects")
    if set(parameter) != _PARAMETER_KEYS:
        raise V3CLIError(f"parameter must contain exactly {sorted(_PARAMETER_KEYS)}")
    if parameter.get("name") != "band_contrasts_db" or parameter.get("shape") != [3, 3]:
        raise V3CLIError("V3 requires the band_contrasts_db parameter with shape [3,3]")
    if parameter.get("dtype") != "float32":
        raise V3CLIError("V3 parameter dtype must be float32")
    if parameter.get("temporal_components") != ["static", "positive_flux", "negative_flux"]:
        raise V3CLIError("V3 temporal component order does not match the pinned head")
    reference_band = constructor.get("reference_band_index")
    if type(reference_band) is not int or not 0 <= reference_band < 4:
        raise V3CLIError("constructor.reference_band_index must be an integer in [0,3]")
    expected_band_order = [band for band in range(4) if band != reference_band]
    if parameter.get("non_reference_band_order") != expected_band_order:
        raise V3CLIError(f"V3 non-reference band order must be {expected_band_order}")
    values = parameter.get("values")
    if not isinstance(values, list) or len(values) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in values
    ):
        raise V3CLIError("band_contrasts_db values must be a 3 by 3 list")
    for row in values:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise V3CLIError("all nine parameter values must be finite JSON numbers")

    role = config.get("profile_role")
    candidate_provenance = config.get("candidate_provenance")
    if role == ENGINEERING_ONLY:
        if any(float(value) != 0.0 for row in values for value in row):
            raise V3CLIError("ENGINEERING_ONLY profile must contain exactly zero theta")
        if "candidate_provenance" in config:
            raise V3CLIError("ENGINEERING_ONLY profile must not carry candidate provenance")
    elif role == METHOD_CANDIDATE:
        if not isinstance(candidate_provenance, dict) or set(candidate_provenance) != {
            "checkpoint",
            "training_record",
            "selection_record",
        }:
            raise V3CLIError("METHOD_CANDIDATE requires pinned checkpoint, training, and selection records")
    else:
        raise V3CLIError("profile_role must be ENGINEERING_ONLY or METHOD_CANDIDATE")

    if input_contract != {
        "container": "RIFF_WAVE",
        "accepted_encodings": ["PCM", "IEEE_FLOAT"],
        "target_sample_rate_hz": 16000,
        "target_channels": 1,
        "downmix": "arithmetic_channel_mean",
        "resampler": "numpy_kaiser_sinc_beta8.6_radius12_over_cutoff",
    }:
        raise V3CLIError("input policy differs from the frozen V3 decoder contract")
    if output_contract != {
        "container": "RIFF_WAVE",
        "encoding": "IEEE_FLOAT_32_LE",
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_count_policy": "exact_canonical_input_count",
        "peak_violation": "reject_without_clipping_or_post_gain",
        "overwrite": False,
    }:
        raise V3CLIError("output policy differs from the frozen V3 CLI contract")
    return config, config_raw, pin_by_path


def _validate_runtime(runtime: dict[str, Any], torch: Any, np: Any) -> str:
    expected_python = runtime.get("python_version")
    if not isinstance(expected_python, list) or len(expected_python) != 3 or any(
        type(x) is not int for x in expected_python
    ):
        raise V3CLIError("runtime.python_version must be a three-integer list")
    if list(sys.version_info[:3]) != expected_python:
        raise V3CLIError(f"Python version mismatch: expected {expected_python}, got {list(sys.version_info[:3])}")
    if runtime.get("numpy_version") != np.__version__:
        raise V3CLIError(f"NumPy version mismatch: expected {runtime.get('numpy_version')}, got {np.__version__}")
    if runtime.get("torch_version") != str(torch.__version__):
        raise V3CLIError(f"PyTorch version mismatch: expected {runtime.get('torch_version')}, got {torch.__version__}")
    device = runtime.get("device")
    if not isinstance(device, str) or not (device == "cpu" or re.fullmatch(r"cuda(?::[0-9]+)?", device)):
        raise V3CLIError("runtime.device must be cpu or an explicit cuda[:index] device")
    intra = runtime.get("torch_intraop_threads")
    inter = runtime.get("torch_interop_threads")
    if type(intra) is not int or intra < 1 or type(inter) is not int or inter < 1:
        raise V3CLIError("runtime thread counts must be positive integers")
    # The strict startup gate establishes these values before the candidate
    # dispatcher runs. PyTorch forbids changing interop threads after parallel
    # work has started, so only invoke the setter when the configured value
    # differs; the equality check below remains the fail-closed policy gate.
    if torch.get_num_threads() != intra:
        torch.set_num_threads(intra)
    if torch.get_num_interop_threads() != inter:
        try:
            torch.set_num_interop_threads(inter)
        except RuntimeError as exc:
            raise V3CLIError(f"cannot establish configured PyTorch interop threads: {exc}") from exc
    if torch.get_num_threads() != intra or torch.get_num_interop_threads() != inter:
        raise V3CLIError("PyTorch thread settings did not match the configuration")
    if device != "cpu" and not torch.cuda.is_available():
        raise V3CLIError(f"configured CUDA device {device} is unavailable; no automatic CPU fallback is allowed")
    if runtime.get("require_mkl", False):
        mkl_backend = getattr(torch.backends, "mkl", None)
        is_available = getattr(mkl_backend, "is_available", None)
        if not callable(is_available) or not bool(is_available()):
            raise V3CLIError("configuration requires MKL but the loaded PyTorch runtime has no MKL")
    return device


def _verify_imported_sources(pins: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sys.path.insert(0, str(_ROOT))
    checked: dict[str, dict[str, Any]] = {}
    for module_name, expected_path in _REQUIRED_RUNTIME_MODULES.items():
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise V3CLIError(f"imported module has no file path: {module_name}")
        resolved = pathlib.Path(module_file).resolve(strict=True)
        try:
            rel = resolved.relative_to(_ROOT).as_posix()
        except ValueError as exc:
            raise V3CLIError(f"imported module resolved outside the pinned repository: {module_name}") from exc
        pin = pins.get(rel)
        if rel != expected_path or pin is None:
            raise V3CLIError(f"imported module path is not in its declared source closure: {module_name}")
        raw = resolved.read_bytes()
        if len(raw) != pin["bytes"] or _sha256(raw) != pin["sha256"]:
            raise V3CLIError(f"imported module source differs from its declared pin: {module_name}")
        checked[module_name] = {"path": rel, "bytes": len(raw), "sha256": _sha256(raw)}
    return checked


def _canonical_constructor_sha(constructor: dict[str, Any]) -> str:
    return _sha256(_json_bytes(constructor))


def _candidate_parameter_values(
    config: dict[str, Any], np: Any
) -> dict[str, Any] | None:
    if config["profile_role"] == ENGINEERING_ONLY:
        return None
    provenance = config["candidate_provenance"]
    loaded: dict[str, tuple[dict[str, Any], bytes]] = {}
    for name in ("checkpoint", "training_record", "selection_record"):
        pin = provenance[name]
        if not isinstance(pin, dict):
            raise V3CLIError(f"candidate_provenance.{name} must be a complete file pin")
        path, raw = _pinned_repo_path(pin, f"candidate_provenance.{name}")
        loaded[name] = (_decode_json_object(raw, f"candidate {name}"), raw)
    checkpoint, checkpoint_raw = loaded["checkpoint"]
    training_record, _ = loaded["training_record"]
    selection_record, _ = loaded["selection_record"]
    implementation = config["implementation"]
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("head_schema") != HEAD_SCHEMA:
        raise V3CLIError("candidate checkpoint schema/head does not match V3")
    if checkpoint.get("implementation") != implementation:
        raise V3CLIError("candidate checkpoint implementation pin differs from config")
    if checkpoint.get("constructor_sha256") != _canonical_constructor_sha(config["constructor"]):
        raise V3CLIError("candidate checkpoint constructor digest differs from config")
    checkpoint_parameter = checkpoint.get("parameter")
    if not isinstance(checkpoint_parameter, dict):
        raise V3CLIError("candidate checkpoint has no parameter object")
    if checkpoint_parameter != config["parameter"]:
        raise V3CLIError("candidate checkpoint parameter does not exactly match config")
    if training_record.get("schema") != TRAINING_RECORD_SCHEMA or training_record.get("status") != "COMPLETE":
        raise V3CLIError("candidate training record is not a completed V3 training record")
    steps = training_record.get("completed_optimizer_steps")
    if type(steps) is not int or steps <= 0:
        raise V3CLIError("candidate training record has no positive completed optimizer-step count")
    if training_record.get("checkpoint_sha256") != _sha256(checkpoint_raw):
        raise V3CLIError("candidate training record does not bind the supplied checkpoint")
    if training_record.get("head_implementation_sha256") != implementation["sha256"]:
        raise V3CLIError("candidate training record does not bind the V3 source")
    if selection_record.get("schema") != SELECTION_RECORD_SCHEMA or selection_record.get("status") != "FROZEN":
        raise V3CLIError("candidate selection record is not frozen")
    if selection_record.get("checkpoint_sha256") != _sha256(checkpoint_raw):
        raise V3CLIError("candidate selection record does not bind the supplied checkpoint")
    if selection_record.get("training_record_sha256") != _sha256(loaded["training_record"][1]):
        raise V3CLIError("candidate selection record does not bind the training record")
    values = config["parameter"]["values"]
    arr = np.asarray(values, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise V3CLIError("candidate float32 parameter values overflow to non-finite values")
    return {
        "checkpoint_sha256": _sha256(checkpoint_raw),
        "training_record_sha256": _sha256(loaded["training_record"][1]),
        "selection_record_sha256": _sha256(loaded["selection_record"][1]),
        "completed_optimizer_steps": steps,
        "pins": provenance,
    }


def _constructor_fields(head_class: Any, constructor: dict[str, Any]) -> None:
    signature = inspect.signature(head_class.__init__)
    expected = {name for name in signature.parameters if name != "self"}
    received = set(constructor)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise V3CLIError(f"constructor fields differ from the pinned source; missing={missing}, extra={extra}")
    int_fields = {
        "sample_rate_hz",
        "n_fft",
        "hop_length",
        "win_length",
        "reference_band_index",
        "gate_smoothing_frames",
        "mask_smoothing_frequency_bins",
        "mask_smoothing_time_frames",
        "video_fps_num",
        "video_fps_den",
    }
    bool_fields = {
        "window_periodic",
        "stft_center",
        "stft_normalized",
        "stft_onesided",
        "stft_return_complex",
    }
    str_fields = {
        "window_kind",
        "stft_pad_mode",
        "gate_smoothing_pad_mode",
        "mask_smoothing_pad_mode",
        "frequency_common_mode_policy",
    }
    for name in int_fields:
        if type(constructor[name]) is not int:
            raise V3CLIError(f"constructor.{name} must be an integer")
    for name in bool_fields:
        if type(constructor[name]) is not bool:
            raise V3CLIError(f"constructor.{name} must be a boolean")
    for name in str_fields:
        if not isinstance(constructor[name], str):
            raise V3CLIError(f"constructor.{name} must be a string")
    edges = constructor.get("band_edges_hz")
    if not isinstance(edges, list) or len(edges) != 5 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in edges
    ):
        raise V3CLIError("constructor.band_edges_hz must contain five finite numbers")
    numeric_fields = {
        "flux_denominator_floor",
        "activity_floor",
        "flux_normalization_scale",
        "positive_flux_quantile",
        "negative_flux_quantile",
        "gate_temperature",
        "max_abs_mask_db",
        "target_rms",
        "rms_epsilon",
        "baseline_match_atol",
        "frame_power_floor",
    }
    for name in numeric_fields:
        value = constructor[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise V3CLIError(f"constructor.{name} must be finite numeric")
    peak = constructor["peak_ceiling"]
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(float(peak)) or peak <= 0
    ):
        raise V3CLIError("constructor.peak_ceiling must be positive finite or null")


_MAX_DIAGNOSTIC_ARRAY_ELEMENTS = 4096


def _json_diagnostic_value(value: Any, path: str) -> Any:
    """Convert nested head diagnostics to bounded JSON-safe finite values.

    Scalar tensors become JSON scalars. Small non-scalar tensors/NumPy arrays
    keep their dtype and shape alongside recursively checked values. Large
    arrays are rejected so a diagnostic sidecar cannot grow without bound.
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise V3CLIError(f"non-finite diagnostic value from V3 head at {path}")
        return float(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise V3CLIError(f"diagnostic mapping key is not a string at {path}")
            result[key] = _json_diagnostic_value(value[key], f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _json_diagnostic_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if hasattr(value, "detach") and hasattr(value, "shape") and hasattr(value, "tolist"):
        tensor = value.detach().cpu()
        shape = [int(dimension) for dimension in tensor.shape]
        if not shape:
            return _json_diagnostic_value(tensor.item(), path)
        if int(tensor.numel()) > _MAX_DIAGNOSTIC_ARRAY_ELEMENTS:
            raise V3CLIError(f"diagnostic tensor exceeds bounded metadata size at {path}")
        return {
            "dtype": str(tensor.dtype),
            "shape": shape,
            "values": _json_diagnostic_value(tensor.tolist(), f"{path}.values"),
        }
    module_name = type(value).__module__
    if module_name.startswith("numpy") and hasattr(value, "shape") and hasattr(value, "tolist"):
        shape = [int(dimension) for dimension in value.shape]
        if not shape:
            return _json_diagnostic_value(value.item(), path)
        if int(value.size) > _MAX_DIAGNOSTIC_ARRAY_ELEMENTS:
            raise V3CLIError(f"diagnostic NumPy array exceeds bounded metadata size at {path}")
        return {
            "dtype": str(value.dtype),
            "shape": shape,
            "values": _json_diagnostic_value(value.tolist(), f"{path}.values"),
        }
    if hasattr(value, "item") and callable(value.item):
        try:
            scalar = value.item()
        except (ValueError, TypeError):
            scalar = None
        if scalar is not None:
            return _json_diagnostic_value(scalar, path)
    raise V3CLIError(f"unsupported diagnostic value at {path}: {type(value).__name__}")


def _stat_binding(path: pathlib.Path) -> tuple[int, int, int, int]:
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _write_metadata_exclusive(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def run(input_path: pathlib.Path, output_path: pathlib.Path, config_path: pathlib.Path, metadata_path: pathlib.Path) -> dict[str, Any]:
    total_start = time.perf_counter_ns()
    config_start = time.perf_counter_ns()
    input_resolved = input_path.resolve(strict=True)
    config_resolved = config_path.resolve(strict=True)
    config_before_stat = _stat_binding(config_resolved)
    _reject_symlink_components(input_path.absolute())
    _reject_symlink_components(config_path.absolute())
    _reject_symlink_components(output_path.absolute())
    _reject_symlink_components(metadata_path.absolute())
    output_resolved = output_path.resolve(strict=False)
    metadata_resolved = metadata_path.resolve(strict=False)
    if not input_resolved.is_file():
        raise V3CLIError("input must be a regular audio file")
    if input_resolved in {output_resolved, metadata_resolved}:
        raise V3CLIError("output and metadata paths must not replace the original input")
    if config_resolved in {output_resolved, metadata_resolved}:
        raise V3CLIError("output paths must not replace the configuration")
    if output_resolved == metadata_resolved:
        raise V3CLIError("float WAV and JSON metadata require distinct paths")
    if output_resolved.suffix.lower() != ".wav" or metadata_resolved.suffix.lower() != ".json":
        raise V3CLIError("output must use .wav and metadata must use .json")
    if output_resolved.exists() or metadata_resolved.exists():
        raise FileExistsError("refusing to overwrite an existing output or metadata file")
    input_before_stat = _stat_binding(input_resolved)
    config, config_raw, pinned_files = _validate_config(config_resolved)
    config_after_stat = _stat_binding(config_resolved)
    if config_after_stat != config_before_stat:
        raise V3CLIError("configuration changed during validation")
    config_ms = (time.perf_counter_ns() - config_start) / 1_000_000.0

    import_start = time.perf_counter_ns()
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    imported_sources = _verify_imported_sources(pinned_files)
    device = _validate_runtime(config["runtime"], torch, np)
    if config["profile_role"] == ENGINEERING_ONLY and device != "cpu":
        raise V3CLIError("ENGINEERING_ONLY profile is restricted to explicit CPU execution")
    if config["profile_role"] == ENGINEERING_ONLY and torch.cuda.is_initialized():
        raise V3CLIError("ENGINEERING_ONLY profile requires CUDA to remain uninitialized")
    head_module = importlib.import_module(
        "scripts.audio_enhancement.shared_recipient_transition_head_v3"
    )
    head_class = getattr(head_module, "SharedRecipientTransitionHeadV3", None)
    if head_class is None:
        raise V3CLIError("pinned V3 source does not expose SharedRecipientTransitionHeadV3")
    _constructor_fields(head_class, config["constructor"])
    candidate_binding = _candidate_parameter_values(config, np)
    import_load_ms = (time.perf_counter_ns() - import_start) / 1_000_000.0

    from scripts.audio_enhancement.audio_io import (
        CANONICAL_SAMPLE_RATE,
        pcm_float32_sha256,
        read_canonical_wav,
        write_float32_wav,
    )
    from scripts.audio_enhancement.baselines import apply_global_rms, waveform_rms

    input_start = time.perf_counter_ns()
    source, read_info = read_canonical_wav(input_resolved, target_rate=CANONICAL_SAMPLE_RATE)
    input_read_ms = (time.perf_counter_ns() - input_start) / 1_000_000.0
    input_after_stat = _stat_binding(input_resolved)
    if input_before_stat != input_after_stat:
        raise V3CLIError("original input changed while being read")
    if source.dtype != np.float32 or source.ndim != 1 or source.size == 0:
        raise V3CLIError("canonical input must be nonempty mono float32")
    if not np.isfinite(source).all():
        raise V3CLIError("canonical input contains non-finite values")
    if config["constructor"]["stft_center"] and config["constructor"]["stft_pad_mode"] == "reflect":
        if source.size <= int(config["constructor"]["n_fft"]) // 2:
            raise V3CLIError("canonical input is too short for centered reflect-padded STFT")

    baseline_start = time.perf_counter_ns()
    baseline, baseline_info = apply_global_rms(
        source,
        target_rms=float(config["constructor"]["target_rms"]),
        peak_ceiling=config["constructor"]["peak_ceiling"],
        silence_epsilon=float(config["constructor"]["rms_epsilon"]),
    )
    if baseline.shape != source.shape or baseline.dtype != np.float32 or not np.isfinite(baseline).all():
        raise V3CLIError("canonical RMS baseline violated exact-length finite float32 contract")
    baseline_ms = (time.perf_counter_ns() - baseline_start) / 1_000_000.0

    constructor_start = time.perf_counter_ns()
    head = head_class(**config["constructor"]).to(device)
    theta = np.asarray(config["parameter"]["values"], dtype=np.float32)
    if theta.shape != (3, 3) or not np.isfinite(theta).all():
        raise V3CLIError("V3 theta must be nine finite float32 values")
    with torch.no_grad():
        if tuple(head.band_contrasts_db.shape) != (3, 3):
            raise V3CLIError("constructed V3 parameter has unexpected shape")
        head.band_contrasts_db.copy_(torch.as_tensor(theta, dtype=torch.float32, device=device))
    if not bool(torch.isfinite(head.band_contrasts_db).all().item()):
        raise V3CLIError("constructed V3 theta contains non-finite values")
    head.eval()
    constructor_ms = (time.perf_counter_ns() - constructor_start) / 1_000_000.0

    forward_start = time.perf_counter_ns()
    source_tensor = torch.from_numpy(np.ascontiguousarray(source)).to(device=device, dtype=torch.float32)
    baseline_tensor = torch.from_numpy(np.ascontiguousarray(baseline)).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        head_result = head(source_tensor, baseline_tensor)
    if config["profile_role"] == ENGINEERING_ONLY and torch.cuda.is_initialized():
        raise V3CLIError("ENGINEERING_ONLY CPU profile unexpectedly initialized CUDA")
    if head_result.get("schema") != HEAD_SCHEMA:
        raise V3CLIError("V3 head returned an unexpected result schema")
    output_tensor = head_result.get("output_pcm_float32")
    if not isinstance(output_tensor, torch.Tensor) or output_tensor.dtype != torch.float32:
        raise V3CLIError("V3 head output must be a float32 tensor")
    if output_tensor.ndim != 1 or output_tensor.numel() != source.size:
        raise V3CLIError("V3 head changed waveform rank or sample count")
    if not bool(torch.isfinite(output_tensor).all().item()):
        raise V3CLIError("V3 head output contains NaN or infinity")
    enhanced = np.ascontiguousarray(output_tensor.detach().cpu().numpy(), dtype=np.float32)
    if enhanced.shape != source.shape or not np.isfinite(enhanced).all():
        raise V3CLIError("V3 output violates finite exact-length float32 contract")
    if config["profile_role"] == ENGINEERING_ONLY and not np.array_equal(enhanced, baseline):
        raise V3CLIError("zero-theta ENGINEERING_ONLY output must equal its canonical RMS baseline")
    output_rms = waveform_rms(enhanced)
    output_peak = float(np.max(np.abs(enhanced.astype(np.float64))))
    peak_ceiling = config["constructor"]["peak_ceiling"]
    if peak_ceiling is not None and output_peak > float(peak_ceiling):
        raise V3CLIError(
            f"V3 output peak {output_peak:.9g} exceeds configured ceiling {float(peak_ceiling):.9g}; no clipping or post-gain applied"
        )
    forward_ms = (time.perf_counter_ns() - forward_start) / 1_000_000.0

    output_start = time.perf_counter_ns()
    write_info = write_float32_wav(
        output_resolved,
        enhanced,
        sample_rate=CANONICAL_SAMPLE_RATE,
        peak_ceiling=None,
        overwrite=False,
    )
    reread, reread_info = read_canonical_wav(output_resolved, target_rate=CANONICAL_SAMPLE_RATE)
    if reread.shape != enhanced.shape or not np.array_equal(reread, enhanced):
        raise V3CLIError("written float WAV did not decode to bit-identical output PCM")
    if reread_info.source_sample_rate != CANONICAL_SAMPLE_RATE or reread_info.source_channels != 1:
        raise V3CLIError("written float WAV readback header is not mono 16 kHz")
    if reread_info.source_encoding != "FLOAT_32":
        raise V3CLIError("written float WAV readback is not IEEE float32")
    if reread_info.pcm_float32_sha256 != pcm_float32_sha256(enhanced):
        raise V3CLIError("written float WAV decoded PCM hash mismatch")
    output_write_readback_ms = (time.perf_counter_ns() - output_start) / 1_000_000.0

    input_final_stat = _stat_binding(input_resolved)
    input_final_sha = _sha256_file(input_resolved)
    if input_final_stat != input_before_stat or input_final_sha != read_info.file_sha256:
        raise V3CLIError("original input file changed during enhancement")

    source_duration_num = int(read_info.source_frames)
    source_duration_den = int(read_info.source_sample_rate)
    canonical_duration_num = int(source.size)
    canonical_duration_den = int(CANONICAL_SAMPLE_RATE)
    total_ms = (time.perf_counter_ns() - total_start) / 1_000_000.0
    canonical_duration = canonical_duration_num / canonical_duration_den
    meta = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "profile_role": config["profile_role"],
        "engineering_only": config["profile_role"] == ENGINEERING_ONLY,
        "enhanced_wav_is_generator_drive_only": True,
        "final_mux_policy": "use the exact original input file as delivery/scoring audio",
        "config": {
            "path": str(config_resolved),
            "bytes": len(config_raw),
            "sha256": _sha256(config_raw),
        },
        "head": {
            "schema": HEAD_SCHEMA,
            "implementation": config["implementation"],
            "constructor": head.configuration(),
            "parameter": {
                "name": "band_contrasts_db",
                "shape": [3, 3],
                "dtype": "float32",
                "temporal_components": config["parameter"]["temporal_components"],
                "non_reference_band_order": config["parameter"]["non_reference_band_order"],
                "values": theta.tolist(),
            },
            "candidate_provenance": candidate_binding,
        },
        "source_pins": config["source_pins"],
        "runtime": {
            "python_version": list(sys.version_info[:3]),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "device": device,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_initialized": bool(torch.cuda.is_initialized()),
        },
        "verified_import_sources": imported_sources,
        "source_audio": {
            "path": str(input_resolved),
            "file_sha256": read_info.file_sha256,
            "source_encoding": read_info.source_encoding,
            "sample_rate_hz": read_info.source_sample_rate,
            "channels": read_info.source_channels,
            "sample_count": read_info.source_frames,
            "duration_seconds_rational": {
                "numerator": source_duration_num,
                "denominator": source_duration_den,
            },
        },
        "canonical_clock": {
            "origin": "source_sample_0",
            "sample_rate_hz": CANONICAL_SAMPLE_RATE,
            "sample_count": int(source.size),
            "duration_seconds_rational": {
                "numerator": canonical_duration_num,
                "denominator": canonical_duration_den,
            },
            "canonical_sample_to_nominal_source_sample_ratio": {
                "numerator": read_info.source_sample_rate,
                "denominator": CANONICAL_SAMPLE_RATE,
            },
            "resampler": read_info.resampler,
            "retiming": False,
        },
        "canonical_input_pcm": {
            "dtype": "float32-le",
            "channels": 1,
            "sample_rate_hz": CANONICAL_SAMPLE_RATE,
            "sample_count": int(source.size),
            "pcm_sha256": read_info.pcm_float32_sha256,
        },
        "rms_baseline": {
            **asdict(baseline_info),
            "measured_peak": float(np.max(np.abs(baseline.astype(np.float64)))),
        },
        "enhanced_output": {
            "path": str(output_resolved),
            "file_sha256": write_info.file_sha256,
            "pcm_float32_sha256": write_info.pcm_float32_sha256,
            "container": "RIFF_WAVE",
            "encoding": write_info.subtype,
            "sample_rate_hz": write_info.sample_rate,
            "channels": write_info.channels,
            "sample_count": write_info.frames,
            "rms": output_rms,
            "peak": output_peak,
            "writer_peak_gain": write_info.applied_peak_gain,
            "readback_bit_identical": True,
        },
        "head_diagnostics": {
            "normalization": {
                key: _json_diagnostic_value(value, f"head_diagnostics.normalization.{key}")
                for key, value in head_result["normalization"].items()
                if key not in {"normalization_residual_added_to_canonical_baseline"}
            },
            "positive_flux_valid_transition_count": int(
                head_result["positive_flux_valid_transition_count"]
            ),
            "negative_flux_valid_transition_count": int(
                head_result["negative_flux_valid_transition_count"]
            ),
            "positive_flux_has_valid_support": bool(head_result["positive_flux_has_valid_support"]),
            "negative_flux_has_valid_support": bool(head_result["negative_flux_has_valid_support"]),
        },
        "timing_ms": {
            "config_and_source_pin_validation": config_ms,
            "imports_runtime_validation_and_head_initialization": import_load_ms + constructor_ms,
            "input_read_decode_canonicalize": input_read_ms,
            "rms_baseline": baseline_ms,
            "head_forward_and_output_checks": forward_ms,
            "float_wav_write_and_exact_readback": output_write_readback_ms,
            "total_wall_before_metadata_write": total_ms,
            "real_time_factor_before_metadata_write": total_ms / (canonical_duration * 1000.0),
            "metadata_serialization_and_write_included": False,
            "profiling_scope": "single cold invocation; not a latency benchmark",
        },
    }
    metadata_raw = json.dumps(meta, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    _write_metadata_exclusive(metadata_resolved, metadata_raw)
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the pinned V3 waveform head; the enhanced WAV is generator-drive audio only."
    )
    parser.add_argument("--input", required=True, type=pathlib.Path, help="original RIFF/WAVE audio to read without modification")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="new canonical mono 16 kHz float32 WAV")
    parser.add_argument("--config", required=True, type=pathlib.Path, help="pinned V3 JSON profile configuration")
    parser.add_argument("--metadata", required=True, type=pathlib.Path, help="new JSON provenance sidecar")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args.input, args.output, args.config, args.metadata)
    except Exception as exc:
        print(
            json.dumps(
                {"schema": RESULT_SCHEMA, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"schema": RESULT_SCHEMA, "status": result["status"], "metadata_path": str(args.metadata.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
