"""Apply a sealed Fit V3 snapshot to one arbitrary mono 16 kHz WAV.

This is a waveform application API only.  It does not select a snapshot,
claim native TFG qualification, render video, or score SyncNet.  The artifact
format deliberately labels the currently packageable source snapshots as
provisional and unselected; a future selected release must be packaged with
its own selection evidence rather than mutating a snapshot record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ARTIFACT_SCHEMA = "cross_tfg_v3_snapshot_application_artifact_v1"
ARTIFACT_STATUS = "PROVISIONAL_FIT_SNAPSHOT_APPLICATION_ONLY_NOT_SELECTED"
RECEIPT_SCHEMA = "cross_tfg_v3_portable_snapshot_wav_application_v1"
RECEIPT_STATUS = "PASS_APPLICATION_ONLY_NOT_A_SELECTION_OR_TFG_QUALIFICATION"
SNAPSHOT_SCHEMA = "cross_tfg_fit40_parameter_snapshot_v1"
SNAPSHOT_INDEX_SCHEMA = "cross_tfg_fit40_snapshot_set_index_v1"
HEAD_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3"
TRAINING_FAMILY = "fit40_native_transition_head_v3_attempt11"
EXPECTED_CONSTRUCTOR_SHA256 = "eaf51303012125c2de40dc102fc04a983c130b6d2aac31cd96b6e5648c3e27b8"

SOURCE_PINS = {
    "package_init": ("scripts/audio_enhancement/__init__.py", 1503,
                     "db520f7b04343682e6df4ba53b22503de806287e70f2b8e41e84f16912d361fc"),
    "audio_io": ("scripts/audio_enhancement/audio_io.py", 14294,
                 "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"),
    "baselines": ("scripts/audio_enhancement/baselines.py", 15123,
                  "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"),
    "pipeline": ("scripts/audio_enhancement/pipeline.py", 5975,
                 "b40ddce9b09ea42d56945eb66acca3326be6784acb6c1e6ab134b1b2899fb6c3"),
    "overdub": ("scripts/audio_enhancement/overdub.py", 12002,
                "0febe6f2b43ed4fb089bdf36e2e208cc1cd61681d8e0ebfd7f59cd56dd00b4b6"),
    "head": ("scripts/audio_enhancement/shared_recipient_transition_head_v3.py", 32233,
             "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87"),
    "snapshot_extractor": (
        "scripts/experiments/cross_tfg_audio_head_20261003/fit40_training_snapshots_attempt02.py",
        68545, "7fe041e078d1f03c10c1735b572f13202315ed17c312918a52b4939863850d66"),
}
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_WAV_BYTES = 256 * 1024 * 1024


class PortableApplyError(RuntimeError):
    """A snapshot artifact, source input, or output failed its application contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash_matches(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    return (isinstance(declared, str) and len(declared) == 64
            and sha256_bytes(canonical_bytes({key: item for key, item in value.items()
                                               if key != field})) == declared)


def _read_regular(path: Path, *, limit: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PortableApplyError(f"{label} must be a regular non-symlink file: {path}")
    info = path.stat(follow_symlinks=False)
    if info.st_size <= 0 or info.st_size > limit:
        raise PortableApplyError(f"{label} is empty or exceeds the {limit}-byte read limit")
    raw = path.read_bytes()
    if len(raw) != info.st_size or len(raw) > limit:
        raise PortableApplyError(f"{label} changed during bounded read")
    return raw


def _load_json(path: Path, *, limit: int, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, limit=limit, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableApplyError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PortableApplyError(f"{label} must contain a JSON object")
    return value, raw


def _resolve_root(root: str | Path) -> Path:
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise PortableApplyError("source root must be an existing real directory")
    return path.resolve(strict=True)


def _check_pinned_source(root: Path, role: str) -> tuple[Path, dict[str, Any]]:
    relative, expected_bytes, expected_sha = SOURCE_PINS[role]
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise PortableApplyError(f"invalid built-in source path for {role}")
    path = root.joinpath(*rel.parts)
    cursor = root
    for component in rel.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise PortableApplyError(f"pinned source path traverses a symlink: {role}")
    raw = _read_regular(path, limit=2 * 1024 * 1024, label=f"pinned source {role}")
    actual_sha = sha256_bytes(raw)
    if (len(raw), actual_sha) != (expected_bytes, expected_sha):
        raise PortableApplyError(f"pinned source differs from the application contract: {role}")
    return path.resolve(strict=True), {"path": relative, "bytes": len(raw), "sha256": actual_sha}


def _verify_snapshot(snapshot: Mapping[str, Any]) -> None:
    if (snapshot.get("schema") != SNAPSHOT_SCHEMA
            or snapshot.get("training_family") != TRAINING_FAMILY
            or snapshot.get("head_schema") != HEAD_SCHEMA
            or not _self_hash_matches(snapshot, "snapshot_sha256")):
        raise PortableApplyError("snapshot schema, family, head, or self-hash is invalid")
    step = snapshot.get("source_optimizer_step")
    method_id = snapshot.get("method_id")
    if (type(step) is not int or step not in (30, 60, 90, 120)
            or method_id != f"{TRAINING_FAMILY}_step{step:03d}"):
        raise PortableApplyError("snapshot is not one of the four pinned Fit40 source steps")
    if (snapshot.get("selection_status") != "NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN"
            or snapshot.get("optimizer_step_is_selected") is not False):
        raise PortableApplyError("this provisional artifact builder accepts only explicitly unselected snapshots")
    parameter = snapshot.get("parameter")
    values = parameter.get("values") if isinstance(parameter, Mapping) else None
    if (not isinstance(parameter, Mapping) or parameter.get("name") != "band_contrasts_db"
            or parameter.get("dtype") != "float32" or parameter.get("shape") != [3, 3]
            or not isinstance(values, list) or len(values) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in values)):
        raise PortableApplyError("snapshot parameter is not the expected [3,3] float32 tensor")
    flat: list[float] = []
    for row in values:
        for item in row:
            if type(item) not in (int, float) or not math.isfinite(float(item)):
                raise PortableApplyError("snapshot theta contains a non-finite or non-numeric value")
            flat.append(float(item))
    theta_raw = struct.pack("<9f", *flat)
    if sha256_bytes(theta_raw) != parameter.get("little_endian_f32_sha256"):
        raise PortableApplyError("snapshot theta differs from its little-endian float32 digest")
    constructor = snapshot.get("head_constructor")
    if (not isinstance(constructor, Mapping)
            or sha256_bytes(canonical_bytes(constructor)) != snapshot.get("head_constructor_sha256")
            or snapshot.get("head_constructor_sha256") != EXPECTED_CONSTRUCTOR_SHA256
            or constructor.get("sample_rate_hz") != 16000
            or constructor.get("n_fft") != 512 or constructor.get("hop_length") != 128
            or constructor.get("max_abs_mask_db") != 3.0
            or constructor.get("peak_ceiling") is not None):
        raise PortableApplyError("snapshot constructor differs from the pinned V3 waveform contract")
    head = snapshot.get("head_implementation")
    expected_head = SOURCE_PINS["head"]
    if (not isinstance(head, Mapping)
            or (head.get("bytes"), head.get("sha256")) != (expected_head[1], expected_head[2])):
        raise PortableApplyError("snapshot head source pin differs from the pinned implementation")


def _verify_index(index: Mapping[str, Any], snapshot: Mapping[str, Any],
                  snapshot_raw_sha: str, snapshot_bytes: int) -> None:
    if (index.get("schema") != SNAPSHOT_INDEX_SCHEMA
            or index.get("status") != "EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND"
            or not _self_hash_matches(index, "index_sha256")
            or index.get("training_family") != TRAINING_FAMILY):
        raise PortableApplyError("snapshot index schema, status, family, or self-hash is invalid")
    entries = index.get("snapshots")
    matches = [entry for entry in entries if isinstance(entry, Mapping)
               and entry.get("method_id") == snapshot.get("method_id")]
    if (len(matches) != 1
            or matches[0].get("source_optimizer_step") != snapshot.get("source_optimizer_step")
            or matches[0].get("snapshot_sha256") != snapshot.get("snapshot_sha256")
            or matches[0].get("bytes") != snapshot_bytes
            or matches[0].get("sha256") != snapshot_raw_sha):
        raise PortableApplyError("snapshot index does not bind the exact snapshot record bytes")
    training = index.get("training_artifact_pins")
    modules = training.get("training_modules") if isinstance(training, Mapping) else None
    expected_index_sources = {
        "audio_io": SOURCE_PINS["audio_io"],
        "baselines": SOURCE_PINS["baselines"],
        "v3_head": SOURCE_PINS["head"],
    }
    if not isinstance(modules, Mapping):
        raise PortableApplyError("snapshot index lacks its pinned training module map")
    for role, (_relative, expected_bytes, expected_sha) in expected_index_sources.items():
        pin = modules.get(role)
        if (not isinstance(pin, Mapping)
                or (pin.get("bytes"), pin.get("sha256")) != (expected_bytes, expected_sha)):
            raise PortableApplyError(f"snapshot index source module pin differs: {role}")
    extractor_pin = index.get("snapshot_extractor_source")
    expected_extractor = SOURCE_PINS["snapshot_extractor"]
    if (not isinstance(extractor_pin, Mapping)
            or (extractor_pin.get("bytes"), extractor_pin.get("sha256"))
               != (expected_extractor[1], expected_extractor[2])):
        raise PortableApplyError("snapshot index does not bind this exact snapshot extractor")
    endpoint = index.get("native_endpoint")
    if (not isinstance(endpoint, Mapping)
            or endpoint.get("status") != "UNBOUND_NATIVE_FIT_SCORES_NOT_RUN"):
        raise PortableApplyError("source snapshot index unexpectedly claims native endpoint qualification")


def seal_provisional_artifact(*, snapshot_path: str | Path, index_path: str | Path,
                              source_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Create a portable, explicitly unselected application envelope.

    The exact snapshot/index contents are embedded so the artifact does not
    depend on the original machine's absolute paths.  Model code remains a
    separately installed, hash-pinned source closure.
    """
    root = _resolve_root(source_root)
    snap_file = Path(snapshot_path)
    idx_file = Path(index_path)
    snapshot, snapshot_raw = _load_json(snap_file, limit=MAX_JSON_BYTES, label="snapshot record")
    index, index_raw = _load_json(idx_file, limit=MAX_JSON_BYTES, label="snapshot index")
    _verify_snapshot(snapshot)
    snapshot_raw_sha = sha256_bytes(snapshot_raw)
    _verify_index(index, snapshot, snapshot_raw_sha, len(snapshot_raw))
    code_pins: dict[str, dict[str, Any]] = {}
    for role in SOURCE_PINS:
        _path, pin = _check_pinned_source(root, role)
        code_pins[role] = pin
    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "status": ARTIFACT_STATUS,
        "selection_status": snapshot["selection_status"],
        "native_tfg_qualification_claim": False,
        "snapshot_record": snapshot,
        "snapshot_record_pin": {
            "bytes": len(snapshot_raw), "sha256": snapshot_raw_sha,
            "self_sha256": snapshot["snapshot_sha256"],
        },
        "snapshot_index": index,
        "snapshot_index_pin": {
            "bytes": len(index_raw), "sha256": sha256_bytes(index_raw),
            "self_sha256": index["index_sha256"],
        },
        "source_code_pins": code_pins,
        "waveform_contract": {
            "input": "RIFF/WAVE, mono, 16 kHz; supported PCM or IEEE float encodings",
            "output": "mono IEEE float32 WAV, same 16 kHz sample count and source clock",
            "peak_ceiling": None,
            "global_projected_log_magnitude_mask_bound_db": 3.0,
        },
    }
    sealed = dict(payload)
    sealed["artifact_sha256"] = sha256_bytes(canonical_bytes(payload))
    raw = json.dumps(sealed, sort_keys=True, ensure_ascii=False, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    destination = Path(output_path)
    _write_exclusive_json(destination, raw, "provisional application artifact")
    return {"path": str(destination.resolve(strict=True)), "bytes": len(raw),
            "sha256": sha256_bytes(raw), "artifact_sha256": sealed["artifact_sha256"],
            "status": sealed["status"], "snapshot_id": snapshot["snapshot_id"],
            "source_optimizer_step": snapshot["source_optimizer_step"]}


def _write_exclusive_json(path: Path, raw: bytes, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise PortableApplyError(f"{label} destination must be fresh, absolute, and non-symlink")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PortableApplyError(f"{label} parent must already be a real directory")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _load_artifact(path: Path) -> tuple[dict[str, Any], bytes]:
    artifact, raw = _load_json(path, limit=MAX_JSON_BYTES, label="application artifact")
    if (artifact.get("schema") != ARTIFACT_SCHEMA
            or artifact.get("status") != ARTIFACT_STATUS
            or artifact.get("native_tfg_qualification_claim") is not False
            or not _self_hash_matches(artifact, "artifact_sha256")):
        raise PortableApplyError("application artifact is not a valid unselected provisional envelope")
    snapshot = artifact.get("snapshot_record")
    index = artifact.get("snapshot_index")
    snap_pin = artifact.get("snapshot_record_pin")
    index_pin = artifact.get("snapshot_index_pin")
    if not all(isinstance(value, Mapping) for value in (snapshot, index, snap_pin, index_pin)):
        raise PortableApplyError("artifact lacks its embedded snapshot/index evidence")
    _verify_snapshot(snapshot)
    snapshot_raw = (json.dumps(snapshot, sort_keys=True, ensure_ascii=False, indent=2,
                               allow_nan=False).encode("utf-8") + b"\n")
    index_raw = (json.dumps(index, sort_keys=True, ensure_ascii=False, indent=2,
                            allow_nan=False).encode("utf-8") + b"\n")
    if (snap_pin.get("bytes") != len(snapshot_raw)
            or snap_pin.get("sha256") != sha256_bytes(snapshot_raw)
            or snap_pin.get("self_sha256") != snapshot.get("snapshot_sha256")
            or index_pin.get("bytes") != len(index_raw)
            or index_pin.get("sha256") != sha256_bytes(index_raw)
            or index_pin.get("self_sha256") != index.get("index_sha256")):
        raise PortableApplyError("embedded snapshot or index bytes differ from their sealed pins")
    _verify_index(index, snapshot, snap_pin["sha256"], snap_pin["bytes"])
    return artifact, raw


def _load_runtime_modules(root: Path, expected_code_pins: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    actual: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for role in SOURCE_PINS:
        path, pin = _check_pinned_source(root, role)
        if expected_code_pins.get(role) != pin:
            raise PortableApplyError(f"installed source differs from artifact pin: {role}")
        paths[role] = path
        actual[role] = pin
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    package = importlib.import_module("scripts.audio_enhancement")
    if Path(str(package.__file__)).resolve(strict=True) != paths["package_init"]:
        raise PortableApplyError("audio enhancement package loaded from an unexpected path")
    for module_name, role in (("scripts.audio_enhancement.audio_io", "audio_io"),
                              ("scripts.audio_enhancement.baselines", "baselines"),
                              ("scripts.audio_enhancement.pipeline", "pipeline"),
                              ("scripts.audio_enhancement.overdub", "overdub")):
        module = importlib.import_module(module_name)
        if Path(str(module.__file__)).resolve(strict=True) != paths[role]:
            raise PortableApplyError(f"runtime import origin differs from the pinned source: {module_name}")
    extractor_path = paths["snapshot_extractor"]
    module_name = "portable_fit40_snapshot_extractor_attempt01_pinned"
    loaded = sys.modules.get(module_name)
    if loaded is None:
        spec = importlib.util.spec_from_file_location(module_name, extractor_path)
        if spec is None or spec.loader is None:
            raise PortableApplyError("cannot import the exact snapshot transform implementation")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        try:
            spec.loader.exec_module(loaded)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    if Path(str(getattr(loaded, "__file__", ""))).resolve(strict=True) != extractor_path:
        raise PortableApplyError("snapshot extractor was already imported from another source")
    transform_class = getattr(loaded, "FitSnapshotTransform", None)
    if not callable(transform_class):
        raise PortableApplyError("pinned snapshot extractor lacks FitSnapshotTransform")
    return transform_class, importlib.import_module("scripts.audio_enhancement.audio_io"), actual


def _safe_output_path(path_value: str | Path, *, input_path: Path, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise PortableApplyError(f"{label} must be a new absolute non-symlink path")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PortableApplyError(f"{label} parent must exist and may not be a symlink")
    if path.resolve(strict=False) == input_path.resolve(strict=True):
        raise PortableApplyError(f"{label} may not overwrite the input WAV")
    return path


def apply_artifact(*, artifact_path: str | Path, source_root: str | Path,
                   input_wav: str | Path, output_wav: str | Path,
                   receipt_path: str | Path) -> dict[str, Any]:
    """Apply a provisional sealed nine-theta artifact to an exact mono 16 kHz WAV."""
    import numpy as np
    import torch

    root = _resolve_root(source_root)
    artifact_path = Path(artifact_path)
    artifact, artifact_raw = _load_artifact(artifact_path)
    snapshot = artifact["snapshot_record"]
    transform_class, audio_io, code_pins = _load_runtime_modules(root, artifact["source_code_pins"])
    input_path = Path(input_wav)
    if not input_path.is_absolute() or input_path.is_symlink() or not input_path.is_file():
        raise PortableApplyError("input WAV must be an absolute, regular non-symlink file")
    input_raw = _read_regular(input_path, limit=MAX_WAV_BYTES, label="input WAV")
    input_sha = sha256_bytes(input_raw)
    output_path = _safe_output_path(output_wav, input_path=input_path, label="output WAV")
    receipt_out = _safe_output_path(receipt_path, input_path=input_path, label="receipt")
    if output_path.resolve(strict=False) == receipt_out.resolve(strict=False):
        raise PortableApplyError("output WAV and receipt must be distinct paths")
    natural, read_info = audio_io.read_canonical_wav(input_path, target_rate=16000)
    if (read_info.file_sha256 != input_sha or read_info.sample_rate != 16000
            or read_info.source_sample_rate != 16000 or read_info.source_channels != 1
            or read_info.frames <= 0 or natural.ndim != 1 or not np.isfinite(natural).all()):
        raise PortableApplyError("input must decode as finite mono 16 kHz PCM without resampling")
    head_path = root.joinpath(*PurePosixPath(SOURCE_PINS["head"][0]).parts)
    baseline_path = root.joinpath(*PurePosixPath(SOURCE_PINS["baselines"][0]).parts)
    transform = transform_class(snapshot=snapshot, head_module_path=head_path,
                                head_module_sha256=SOURCE_PINS["head"][2],
                                device="cpu", baselines_path=baseline_path)
    hook_state: dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, dict):
            raise PortableApplyError("pinned V3 head emitted no diagnostics mapping")
        mask = output.get("projected_log_magnitude_mask_db")
        if mask is None or not bool(torch.isfinite(mask).all().item()):
            raise PortableApplyError("projected mask is absent or non-finite")
        hook_state.clear()
        hook_state.update({
            "preprojection_max_abs_mask_db": float(output["preprojection_max_abs_mask_db"].detach().cpu().item()),
            "postprojection_max_abs_mask_db": float(output["postprojection_max_abs_mask_db"].detach().cpu().item()),
            "valid_transition_count": int(output["valid_transition_mask"].sum().detach().cpu().item()),
            "stft_frame_count": int(mask.shape[-1]),
        })

    hook = transform.head.register_forward_hook(capture)
    try:
        output, transform_info = transform.transform(natural)
    finally:
        hook.remove()
    output = np.ascontiguousarray(output, dtype=np.float32)
    if output.shape != natural.shape or not bool(np.isfinite(output).all()) or not hook_state:
        raise PortableApplyError("transform changed sample count or emitted non-finite/empty output")
    constructor = snapshot["head_constructor"]
    baseline = transform_info.get("canonical_rms_baseline")
    output_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2, dtype=np.float64)))
    target_rms = float(constructor["target_rms"])
    if (not isinstance(baseline, Mapping) or not math.isfinite(output_rms)
            or abs(output_rms - float(baseline.get("output_rms", float("nan")))) > 1e-6
            or abs(output_rms - target_rms) > float(constructor["baseline_match_atol"]) + 1e-6
            or hook_state["postprojection_max_abs_mask_db"] > 3.0 + 1e-5):
        raise PortableApplyError("output failed frozen global RMS or 3 dB projection checks")

    # The canonical writer uses a same-directory temporary and atomic replace.
    # Write that temporary under a unique name, then hard-link to the final
    # destination so an existing output can never be overwritten by a race.
    temp_path = output_path.with_name(f".{output_path.name}.{os.urandom(8).hex()}.pending.wav")
    if temp_path.exists() or temp_path.is_symlink():
        raise PortableApplyError("temporary output name collision")
    write_info = audio_io.write_float32_wav(temp_path, output, sample_rate=16000,
                                            peak_ceiling=None, overwrite=False)
    try:
        os.link(temp_path, output_path)
    except Exception:
        # Keep an already-linked output as evidence; only remove our private
        # temporary after a successful final link.
        raise
    temp_path.unlink()
    output_raw = _read_regular(output_path, limit=MAX_WAV_BYTES, label="written output WAV")
    reread, reread_info = audio_io.read_canonical_wav(output_path, target_rate=16000)
    output_pcm_sha = sha256_bytes(output.astype("<f4", copy=False).tobytes())
    if (reread_info.file_sha256 != sha256_bytes(output_raw)
            or reread_info.pcm_float32_sha256 != output_pcm_sha
            or reread_info.frames != read_info.frames
            or not np.array_equal(reread, output)):
        raise PortableApplyError("written output failed exact float32 PCM/source-clock readback")
    receipt_payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "selection_status": snapshot["selection_status"],
        "native_tfg_qualification_claim": False,
        "artifact": {"path": str(artifact_path.resolve(strict=True)),
                     "bytes": len(artifact_raw), "sha256": sha256_bytes(artifact_raw),
                     "artifact_sha256": artifact["artifact_sha256"],
                     "snapshot_id": snapshot["snapshot_id"],
                     "snapshot_sha256": snapshot["snapshot_sha256"],
                     "source_optimizer_step": snapshot["source_optimizer_step"],
                     "method_id": snapshot["method_id"]},
        "input_wav": {"path": str(input_path.resolve(strict=True)), "bytes": len(input_raw),
                      "sha256": input_sha, "pcm_float32_sha256": read_info.pcm_float32_sha256,
                      "sample_count": int(read_info.frames), "sample_rate_hz": 16000,
                      "channels": 1, "source_encoding": read_info.source_encoding},
        "output_wav": {"path": str(output_path.resolve(strict=True)),
                       "bytes": len(output_raw), "sha256": sha256_bytes(output_raw),
                       "pcm_float32_sha256": output_pcm_sha,
                       "sample_count": int(reread_info.frames), "sample_rate_hz": 16000,
                       "channels": 1, "encoding": "FLOAT_32"},
        "clock": {"sample_rate_hz": 16000, "input_samples": int(read_info.frames),
                  "output_samples": int(reread_info.frames), "preserved": True},
        "theta": snapshot["parameter"]["values"],
        "theta_little_endian_f32_sha256": snapshot["parameter"]["little_endian_f32_sha256"],
        "constructor_sha256": snapshot["head_constructor_sha256"],
        "runtime_source_pins": code_pins,
        "rms": {"input_rms": float(baseline["input_rms"]),
                "target_rms": target_rms, "output_rms": output_rms,
                "output_rms_error": abs(output_rms-target_rms), "peak_ceiling": None,
                "peak_limiter_applied": False},
        "projection": {**hook_state, "bound_db": 3.0,
                       "within_bound": True, "projection_is_global_scale": True},
        "transform_summary": transform_info,
        "runtime": {"python_version": platform.python_version(),
                    "python_executable": str(Path(sys.executable).resolve()),
                    "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
                    "device": "cpu"},
        "scope": "waveform_application_only; no TFG renderer, scorer, selection, or qualification",
    }
    receipt = dict(receipt_payload)
    receipt["self_sha256"] = sha256_bytes(canonical_bytes(receipt_payload))
    receipt_raw = json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2,
                             allow_nan=False).encode("utf-8") + b"\n"
    _write_exclusive_json(receipt_out, receipt_raw, "application receipt")
    return {"path": str(receipt_out.resolve(strict=True)), "bytes": len(receipt_raw),
            "sha256": sha256_bytes(receipt_raw), "self_sha256": receipt["self_sha256"],
            "status": receipt["status"], "output_wav": receipt["output_wav"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal-provisional", help="package one exact unselected snapshot for application-only use")
    seal.add_argument("--snapshot", type=Path, required=True)
    seal.add_argument("--index", type=Path, required=True)
    seal.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    seal.add_argument("--output-artifact", type=Path, required=True)
    apply = sub.add_parser("apply", help="apply the artifact to a mono 16 kHz WAV (CPU only)")
    apply.add_argument("--artifact", type=Path, required=True)
    apply.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    apply.add_argument("--input-wav", type=Path, required=True)
    apply.add_argument("--output-wav", type=Path, required=True)
    apply.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal-provisional":
            result = seal_provisional_artifact(snapshot_path=args.snapshot,
                                               index_path=args.index,
                                               source_root=args.source_root,
                                               output_path=args.output_artifact)
        else:
            result = apply_artifact(artifact_path=args.artifact,
                                    source_root=args.source_root,
                                    input_wav=args.input_wav,
                                    output_wav=args.output_wav,
                                    receipt_path=args.receipt)
    except Exception as exc:
        parser.exit(2, f"portable snapshot application failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
