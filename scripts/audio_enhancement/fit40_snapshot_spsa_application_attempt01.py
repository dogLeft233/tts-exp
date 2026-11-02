"""Apply one frozen Fit snapshot head at a bound SPSA theta, without scoring.

The transformation is the existing ``FitSnapshotTransform`` from the pinned
snapshot extractor and the exact V3 head implementation.  This module adds
strict input/point/output lineage around that operation; it does not train,
render video, score SyncNet, or select a candidate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import stat
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


SCHEMA = "cross_tfg_fit_snapshot_generator_drive_result_v1"
STATUS = "PASS_FIT_SNAPSHOT_GENERATOR_DRIVE_ONLY"
SNAPSHOT_SCHEMA = "cross_tfg_fit40_parameter_snapshot_v1"
HEAD_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3"
HEAD_REL = PurePosixPath("scripts/audio_enhancement/shared_recipient_transition_head_v3.py")
EXTRACTOR_REL = PurePosixPath(
    "scripts/experiments/cross_tfg_audio_head_20261003/fit40_training_snapshots_attempt02.py"
)
BASELINES_REL = PurePosixPath("scripts/audio_enhancement/baselines.py")
EXTRACTOR_BYTES = 68_545
EXTRACTOR_SHA256 = "7fe041e078d1f03c10c1735b572f13202315ed17c312918a52b4939863850d66"
HEAD_BYTES = 32_233
HEAD_SHA256 = "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87"
BASELINES_BYTES = 15_123
BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
ALLOWED_SOURCE_STEPS = {30, 60, 90, 120}
ALLOWED_POINT_STEPS = {5, 10, 15, 20}


class ApplicationError(RuntimeError):
    """A snapshot, point, audio, or runtime input failed a frozen check."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pin(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ApplicationError(f"expected a non-symlink regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ApplicationError(f"source exceeds configured byte limit: {path}")
    return {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _self_hash(value: Mapping[str, Any], field: str) -> bool:
    expected = value.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    return sha256_bytes(canonical_bytes({k: v for k, v in value.items() if k != field})) == expected


def _safe_under(root: Path, path_text: Any, label: str, *, required: bool = True) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise ApplicationError(f"{label} path is missing")
    path = Path(path_text)
    if not path.is_absolute():
        pure = PurePosixPath(path_text.replace("\\", "/"))
        if pure.is_absolute() or not pure.parts or any(p in ("", ".", "..") for p in pure.parts):
            raise ApplicationError(f"{label} has an unsafe relative path")
        path = root.joinpath(*pure.parts)
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=required)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ApplicationError(f"{label} escaped the pinned project root") from exc
    if required and (path.is_symlink() or not resolved.is_file()):
        raise ApplicationError(f"{label} must be a non-symlink regular file")
    return resolved


def _read_pin(root: Path, pin: Mapping[str, Any], label: str, *,
              self_field: str | None = None) -> tuple[dict[str, Any], bytes, Path]:
    path = _safe_under(root, pin.get("path"), label)
    raw = path.read_bytes()
    if type(pin.get("bytes")) is not int or len(raw) != pin["bytes"]:
        raise ApplicationError(f"{label} byte count differs from its frozen plan pin")
    raw_sha = sha256_bytes(raw)
    if pin.get("sha256") != raw_sha:
        raise ApplicationError(f"{label} raw SHA differs from its frozen plan pin")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApplicationError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ApplicationError(f"{label} JSON root is not an object")
    if self_field is not None:
        expected_self = pin.get("self_sha256")
        if (not _self_hash(value, self_field) or expected_self != value.get(self_field)):
            raise ApplicationError(f"{label} canonical self hash differs from its plan pin")
    return value, raw, path


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ApplicationError(f"{label} must be a lowercase SHA-256")
    return value


def _theta_matrix(values: Any, np: Any) -> Any:
    if not isinstance(values, (list, tuple)) or len(values) != 9:
        raise ApplicationError("candidate theta must contain the frozen nine parameters")
    try:
        flat = np.asarray([float(x) for x in values], dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApplicationError("candidate theta contains a nonnumeric parameter") from exc
    if flat.shape != (9,) or not bool(np.isfinite(flat).all()):
        raise ApplicationError("candidate theta must be nine finite float32 values")
    return flat.reshape(3, 3)


def _find_unique_plan_pin(plan: Mapping[str, Any], pin: Mapping[str, Any], label: str) -> None:
    candidates = plan.get("scientific_input_pins")
    if not isinstance(candidates, Mapping):
        raise ApplicationError("plan omits its scientific input pin map")
    matches = [item for item in candidates.values()
               if isinstance(item, Mapping)
               and item.get("path") == pin.get("path")
               and item.get("bytes") == pin.get("bytes")
               and item.get("sha256") == pin.get("sha256")
               and item.get("self_sha256") == pin.get("self_sha256")]
    if len(matches) != 1:
        raise ApplicationError(f"{label} is not uniquely bound by the parent plan")


def _load_pinned_extractor(root: Path) -> Any:
    path = root.joinpath(*EXTRACTOR_REL.parts)
    actual = _pin(path, max_bytes=EXTRACTOR_BYTES)
    if (actual["bytes"], actual["sha256"]) != (EXTRACTOR_BYTES, EXTRACTOR_SHA256):
        raise ApplicationError("snapshot extractor source differs from its frozen exact pin")
    name = "fit40_training_snapshots_attempt02_spsa_application"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ApplicationError("cannot import the pinned snapshot extractor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_point(plan: Mapping[str, Any], method: Mapping[str, Any],
                    parent_row: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    config = method.get("config")
    if not isinstance(config, Mapping) or config.get("method") not in {
            "fit40_spsa_proposal", "fit40_spsa_center"}:
        raise ApplicationError("method must be one declared Fit SPSA proposal or center")
    method_id = method.get("method_id")
    if not isinstance(method_id, str) or config.get("method_id") != method_id:
        raise ApplicationError("method ID differs between the plan method and config")
    config_sha = _require_sha(method.get("config_sha256"), "method.config_sha256")
    if sha256_bytes(canonical_bytes(config)) != config_sha:
        raise ApplicationError("SPSA method config canonical hash differs")
    if method.get("checkpoint_sha256") is not None:
        raise ApplicationError("snapshot SPSA transform cannot substitute a checkpoint")
    plan_methods = [m for m in plan.get("methods", []) if isinstance(m, Mapping)
                    and m.get("method_id") == method_id]
    if len(plan_methods) != 1 or canonical_bytes(plan_methods[0]) != canonical_bytes(method):
        raise ApplicationError("method differs from the unique method frozen into the parent plan")
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN":
        raise ApplicationError("parent Stage-A plan is not frozen")
    if not _self_hash(plan, "plan_sha256"):
        raise ApplicationError("parent Stage-A plan self hash is invalid")
    if (config.get("input_pins_sha256") != plan.get("scientific_input_pins_sha256")
            or config.get("artifact_sha256") != config.get("source_snapshot_artifact_sha256")):
        raise ApplicationError("SPSA config input or selected seed snapshot artifact differs from plan")
    step = config.get("source_snapshot_step")
    if type(step) is not int or step not in ALLOWED_SOURCE_STEPS:
        raise ApplicationError("source snapshot step is not one of the frozen 30/60/90/120 checkpoints")
    source_seed_method_id = config.get("source_seed_method_id")
    if not isinstance(source_seed_method_id, str):
        raise ApplicationError("SPSA config does not identify its exact seed method")
    seeds = [m for m in plan.get("methods", []) if isinstance(m, Mapping)
             and m.get("method_id") == source_seed_method_id]
    if len(seeds) != 1:
        raise ApplicationError("source seed method is not uniquely present in the frozen plan")
    seed_config = seeds[0].get("config")
    if (not isinstance(seed_config, Mapping)
            or seeds[0].get("kind") != "fit_snapshot"
            or seeds[0].get("checkpoint_sha256") is not None
            or seed_config.get("method") != "precomputed_fit_snapshot_waveform"
            or seed_config.get("source_optimizer_step") != step
            or seed_config.get("snapshot_sha256") != config.get("artifact_sha256")
            or seed_config.get("training_family") != config.get("head_family")):
        raise ApplicationError("SPSA seed identity does not match its exact frozen snapshot method")
    if config.get("method") == "fit40_spsa_proposal":
        if config.get("side") not in {"plus", "minus"} or not isinstance(config.get("proposal_id"), str):
            raise ApplicationError("proposal point lacks its frozen plus/minus identity")
    elif config.get("point_kind") != "spsa_center" or config.get("source_step") not in ALLOWED_POINT_STEPS:
        raise ApplicationError("center is not one of the predeclared full-score centers 5/10/15/20")
    _theta_matrix(config.get("theta"), __import__("numpy"))
    theta_sha = _require_sha(config.get("theta_sha256"), "SPSA theta SHA")
    if sha256_bytes(canonical_bytes(list(config["theta"]))) != theta_sha:
        raise ApplicationError("SPSA theta canonical hash differs")
    for key in ("artifact_sha256", "binding_sha256", "input_pins_sha256"):
        _require_sha(config.get(key), f"SPSA config {key}")
    record_pin = config.get("source_snapshot_record")
    index_pin = config.get("source_snapshot_index")
    if not isinstance(record_pin, Mapping) or not isinstance(index_pin, Mapping):
        raise ApplicationError("SPSA config lacks exact snapshot record/index pins")
    _find_unique_plan_pin(plan, record_pin, "source snapshot record")
    _find_unique_plan_pin(plan, index_pin, "snapshot set index")
    if (config.get("source_snapshot_record_raw_sha256") != record_pin.get("sha256")
            or config.get("source_snapshot_index_raw_sha256") != index_pin.get("sha256")):
        raise ApplicationError("SPSA config raw snapshot/index SHA refs differ from their pins")
    if (parent_row.get("split") != "fit" or parent_row.get("method_id") != "identity"
            or parent_row.get("model_id") != "musetalk_fixed_portrait_attempt16"
            or parent_row.get("score_audio") != parent_row.get("natural_pcm")):
        raise ApplicationError("parent row must be the same Fit Muse identity row scoring natural PCM")
    if config.get("method") == "fit40_spsa_proposal":
        if parent_row.get("sample_id") not in config.get("sample_ids", []):
            raise ApplicationError("proposal method does not contain the parent sample")
    return dict(config), config_sha, dict(record_pin), dict(index_pin)


def apply_snapshot_theta(*, project_root: str | Path, plan: Mapping[str, Any],
                         method: Mapping[str, Any], parent_row: Mapping[str, Any],
                         output_wav_path: str | Path, receipt_path: str | Path,
                         device: str = "cpu",
                         artifact_path_overrides: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
    """Apply the method's nine theta values to one exact natural PCM row.

    ``artifact_path_overrides`` can remap only the snapshot record/index to
    byte-identical files in a local readback tree; hashes and plan pins remain
    authoritative. Production project payloads normally need no overrides.
    """
    import numpy as np
    import torch

    root = Path(project_root)
    if root.is_symlink() or not root.is_dir():
        raise ApplicationError("project root must be a real directory")
    root = root.resolve(strict=True)
    config, config_sha, record_pin, index_pin = _validate_point(plan, method, parent_row)
    if torch.device(device).type not in {"cpu", "cuda"}:
        raise ApplicationError("snapshot transform device must be explicit CPU or CUDA")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ApplicationError("requested CUDA device is unavailable")

    overrides = dict(artifact_path_overrides or {})
    record_override = overrides.get("source_snapshot_record")
    index_override = overrides.get("source_snapshot_index")
    if record_override is not None:
        record_pin = dict(record_pin, path=str(Path(record_override).resolve(strict=True)))
    if index_override is not None:
        index_pin = dict(index_pin, path=str(Path(index_override).resolve(strict=True)))
    snapshot, record_raw, record_path = _read_pin(root, record_pin, "source snapshot", self_field="snapshot_sha256")
    index, index_raw, index_path = _read_pin(root, index_pin, "snapshot index", self_field="index_sha256")
    source_step = config["source_snapshot_step"]
    seed_id = config["source_seed_method_id"]
    record_entry = [item for item in index.get("snapshots", []) if isinstance(item, Mapping)
                    and item.get("method_id") == seed_id]
    if (len(record_entry) != 1 or record_entry[0].get("source_optimizer_step") != source_step
            or record_entry[0].get("snapshot_sha256") != config.get("artifact_sha256")
            or record_entry[0].get("bytes") != len(record_raw)
            or record_entry[0].get("sha256") != sha256_bytes(record_raw)):
        raise ApplicationError("snapshot set index does not bind the exact selected snapshot record")
    if (snapshot.get("schema") != SNAPSHOT_SCHEMA
            or snapshot.get("selection_status") != "NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN"
            or snapshot.get("optimizer_step_is_selected") is not False
            or snapshot.get("training_family") != config.get("head_family")
            or snapshot.get("method_id") != seed_id
            or snapshot.get("source_optimizer_step") != source_step
            or snapshot.get("snapshot_sha256") != config.get("artifact_sha256")
            or snapshot.get("head_schema") != HEAD_SCHEMA):
        raise ApplicationError("snapshot identity or unselected state differs from the method lineage")
    source_theta = snapshot.get("parameter", {}).get("values")
    source_theta_flat = np.asarray(source_theta, dtype=np.float64).reshape(-1).tolist()
    # The packet builder hashes the flattened nine-value controller view,
    # while the snapshot file stores those same values as a [3,3] matrix.
    if (config.get("source_theta") != source_theta_flat
            or config.get("source_theta_sha256") != sha256_bytes(canonical_bytes(source_theta_flat))):
        raise ApplicationError("method's recorded seed theta differs from exact snapshot parameters")
    seed_method = next(m for m in plan["methods"] if m.get("method_id") == seed_id)
    if seed_method.get("config", {}).get("training_family") != snapshot.get("training_family"):
        raise ApplicationError("snapshot seed family differs from the plan's seed method")
    constructor = snapshot.get("head_constructor")
    if (not isinstance(constructor, Mapping)
            or snapshot.get("head_constructor_sha256") != sha256_bytes(canonical_bytes(constructor))
            or constructor.get("sample_rate_hz") != 16000
            or constructor.get("n_fft") != 512 or constructor.get("hop_length") != 128
            or constructor.get("max_abs_mask_db") != 3.0
            or constructor.get("peak_ceiling") is not None):
        raise ApplicationError("snapshot constructor differs from the frozen V3 audio/STFT/projection setup")

    natural = parent_row.get("natural_pcm")
    if not isinstance(natural, Mapping):
        raise ApplicationError("identity parent lacks its canonical natural PCM pin")
    natural_path_text = natural.get("path")
    if not isinstance(natural_path_text, str) or not natural_path_text:
        raise ApplicationError("original natural PCM row has no source path")
    natural_path_candidate = Path(natural_path_text)
    if natural_path_candidate.is_absolute():
        input_path = natural_path_candidate.resolve(strict=True)
    else:
        input_path = _safe_under(root, natural_path_text, "original natural PCM")
    if input_path.is_symlink() or not input_path.is_file():
        raise ApplicationError("original natural PCM path must be a non-symlink regular file")
    input_raw = input_path.read_bytes()
    input_file_sha = sha256_bytes(input_raw)
    if input_file_sha != natural.get("file_sha256"):
        raise ApplicationError("original natural WAV bytes differ from the identity row")
    natural_pin = plan.get("scientific_input_pins", {}).get(f"natural_pcm_{parent_row.get('sample_id')}")
    if (not isinstance(natural_pin, Mapping) or natural_pin.get("sha256") != input_file_sha
            or natural_pin.get("bytes") != len(input_raw)):
        raise ApplicationError("natural WAV file bytes are not bound by its sample-specific plan input pin")

    source_root = root
    head_path = source_root.joinpath(*HEAD_REL.parts)
    baselines_path = source_root.joinpath(*BASELINES_REL.parts)
    extractor_path = source_root.joinpath(*EXTRACTOR_REL.parts)
    head_pin = _pin(head_path)
    baseline_pin = _pin(baselines_path)
    extractor_pin = _pin(extractor_path, max_bytes=EXTRACTOR_BYTES)
    source_head_pin = snapshot.get("head_implementation")
    if (not isinstance(source_head_pin, Mapping)
            or (source_head_pin.get("bytes"), source_head_pin.get("sha256")) != (HEAD_BYTES, HEAD_SHA256)
            or (head_pin["bytes"], head_pin["sha256"]) != (HEAD_BYTES, HEAD_SHA256)):
        raise ApplicationError("loaded V3 head bytes differ from the seed snapshot's source pin")
    training_modules = index.get("training_artifact_pins", {}).get("training_modules", {})
    expected_audio_io = training_modules.get("audio_io", {}) if isinstance(training_modules, Mapping) else {}
    expected_baselines = training_modules.get("baselines", {}) if isinstance(training_modules, Mapping) else {}
    expected_head = training_modules.get("v3_head", {}) if isinstance(training_modules, Mapping) else {}
    if (baseline_pin["bytes"], baseline_pin["sha256"]) != (BASELINES_BYTES, BASELINES_SHA256):
        raise ApplicationError("loaded global RMS baseline bytes differ from the frozen source pin")
    if any((ref.get("bytes"), ref.get("sha256")) != (pin["bytes"], pin["sha256"])
           for ref, pin in ((expected_baselines, baseline_pin), (expected_head, head_pin))):
        raise ApplicationError("loaded baseline or head module differs from the snapshot index source pins")
    audio_io_path = source_root.joinpath("scripts", "audio_enhancement", "audio_io.py")
    audio_io_pin = _pin(audio_io_path)
    if any((expected_audio_io.get(key)) != audio_io_pin.get(key) for key in ("bytes", "sha256")):
        raise ApplicationError("canonical WAV reader differs from the source snapshot pin")

    extractor = _load_pinned_extractor(source_root)
    transform = extractor.FitSnapshotTransform(
        snapshot=snapshot, head_module_path=head_path,
        head_module_sha256=HEAD_SHA256, baselines_path=baselines_path, device=device,
    )
    with torch.no_grad():
        transform.head.band_contrasts_db.copy_(torch.as_tensor(theta, dtype=torch.float32, device=device))
    transform.head.eval()

    # Load the exact canonical reader from the staged project. This import is
    # source pinned by the snapshot index and does not access model packages.
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from scripts.audio_enhancement.audio_io import read_canonical_wav, write_float32_wav

    original, original_info = read_canonical_wav(input_path, target_rate=16000)
    if (original_info.file_sha256 != natural.get("file_sha256")
            or original_info.pcm_float32_sha256 != natural.get("pcm_sha256")
            or original_info.frames != natural.get("sample_count")
            or original_info.sample_rate != 16000
            or original_info.source_sample_rate != 16000
            or original_info.source_channels != 1):
        raise ApplicationError("original natural PCM does not reproduce the row's exact 16 kHz clock pin")
    if natural.get("subtype") not in (None, original_info.source_encoding):
        raise ApplicationError("original WAV encoding differs from its source row pin")

    hook_state: dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, dict):
            raise ApplicationError("head did not return its pinned V3 diagnostics object")
        mask = output.get("projected_log_magnitude_mask_db")
        if mask is None or not bool(torch.isfinite(mask).all().item()):
            raise ApplicationError("projected V3 mask is absent or nonfinite")
        hook_state.clear()
        hook_state.update({
            "preprojection_max_abs_mask_db": float(output["preprojection_max_abs_mask_db"].detach().cpu().item()),
            "postprojection_max_abs_mask_db": float(output["postprojection_max_abs_mask_db"].detach().cpu().item()),
            "projection_scale": float(output["smooth_projection_scale"].detach().cpu().item()),
            "positive_flux_valid_transition_count": int(output["positive_flux_valid_transition_count"].detach().cpu().item()),
            "negative_flux_valid_transition_count": int(output["negative_flux_valid_transition_count"].detach().cpu().item()),
            "positive_flux_has_valid_support": bool(output["positive_flux_has_valid_support"]),
            "negative_flux_has_valid_support": bool(output["negative_flux_has_valid_support"]),
            "stft_frame_count": int(mask.shape[-1]),
            "valid_transition_count": int(output["valid_transition_mask"].sum().detach().cpu().item()),
        })

    hook = transform.head.register_forward_hook(capture)
    try:
        output_samples, transform_info = transform.transform(original)
    finally:
        hook.remove()
    if not hook_state:
        raise ApplicationError("pinned head forward did not emit its projection evidence")
    output_samples = np.ascontiguousarray(output_samples, dtype=np.float32)
    if (output_samples.shape != original.shape or not bool(np.isfinite(output_samples).all())
            or len(output_samples) != original_info.frames):
        raise ApplicationError("candidate output changed sample count or contains nonfinite PCM")

    baseline = transform_info.get("canonical_rms_baseline")
    output_rms = float(np.sqrt(np.mean(output_samples.astype(np.float64) ** 2, dtype=np.float64)))
    target_rms = float(constructor["target_rms"])
    if (not isinstance(baseline, Mapping) or not math.isfinite(output_rms)
            or abs(output_rms - float(baseline.get("output_rms", float("nan")))) > 1e-6
            or abs(output_rms - target_rms) > float(constructor["baseline_match_atol"]) + 1e-6):
        raise ApplicationError("candidate output failed the exact frozen global RMS baseline contract")
    if hook_state["postprojection_max_abs_mask_db"] > float(constructor["max_abs_mask_db"]) + 1e-5:
        raise ApplicationError("candidate V3 mask exceeds its frozen postprojection bound")

    out_path = Path(output_wav_path)
    receipt_out = Path(receipt_path)
    for path, label in ((out_path, "output WAV"), (receipt_out, "application receipt")):
        if not path.is_absolute() or path.is_symlink() or path.exists():
            raise ApplicationError(f"{label} must have a fresh absolute destination: {path}")
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise ApplicationError(f"{label} parent must be an existing real directory")
    write_info = write_float32_wav(out_path, output_samples, sample_rate=16000,
                                   peak_ceiling=None, overwrite=False)
    output_pin = _pin(out_path, max_bytes=8 * 1024 * 1024)
    reread, reread_info = read_canonical_wav(out_path, target_rate=16000)
    output_pcm_sha = sha256_bytes(output_samples.astype("<f4", copy=False).tobytes())
    if (reread_info.file_sha256 != output_pin["sha256"]
            or reread_info.pcm_float32_sha256 != output_pcm_sha
            or reread_info.frames != original_info.frames
            or not np.array_equal(reread, output_samples)):
        raise ApplicationError("written generator WAV did not round-trip bit-exactly")

    method_point_fields = ("proposal_id", "pair_id", "iteration", "side", "batch_id", "c_k",
                           "point_kind", "source_step")
    packet: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "render_or_score_performed": False,
        "optimizer_created_or_stepped": False,
        "method_id": method["method_id"],
        "method_config_sha256": config_sha,
        "artifact_sha256": config["artifact_sha256"],
        "source_snapshot_artifact_sha256": config["source_snapshot_artifact_sha256"],
        "source_seed_method_id": config["source_seed_method_id"],
        "source_snapshot_step": config["source_snapshot_step"],
        "source_snapshot_record_raw_sha256": sha256_bytes(record_raw),
        "source_snapshot_record_self_sha256": snapshot["snapshot_sha256"],
        "source_snapshot_index_raw_sha256": sha256_bytes(index_raw),
        "source_snapshot_index_self_sha256": index["index_sha256"],
        "snapshot_record_runtime_path": str(record_path),
        "snapshot_index_runtime_path": str(index_path),
        "head_schema": snapshot["head_schema"],
        "head_implementation": head_pin,
        "source_head_implementation_pin": dict(source_head_pin),
        "head_constructor_sha256": snapshot["head_constructor_sha256"],
        "method_kind": config["method"],
        "theta": list(config["theta"]),
        "theta_sha256": config["theta_sha256"],
        "theta_float32_sha256": sha256_bytes(theta.astype("<f4", copy=False).tobytes()),
        "binding_sha256": config["binding_sha256"],
        "input_pins_sha256": config["input_pins_sha256"],
        "source_clock": {"sample_rate_hz": 16000, "sample_count": int(original_info.frames),
                         "channels": 1, "preserved": True},
        "input_original_file": {"path": str(input_path), "bytes": len(input_raw),
                                 "file_sha256": input_file_sha},
        "canonical_original_pcm": {"pcm_float32_sha256": original_info.pcm_float32_sha256,
                                    "sample_count": int(original_info.frames),
                                    "sample_rate_hz": 16000, "channels": 1,
                                    "source_encoding": original_info.source_encoding},
        "canonical_generator_input": {"pcm_float32_sha256": transform_info["input_pcm_float32_sha256"],
                                      "sample_count": int(original_info.frames),
                                      "sample_rate_hz": 16000, "channels": 1},
        "output_generator_drive_wav": {
            **output_pin, "file_sha256": output_pin["sha256"],
            "pcm_float32_sha256": write_info.pcm_float32_sha256,
            "sample_count": int(write_info.frames), "sample_rate_hz": 16000,
            "channels": 1, "encoding": "FLOAT_32",
        },
        "output_pcm_float32_sha256": write_info.pcm_float32_sha256,
        "output_sample_count": int(write_info.frames),
        "source_clock_preserved_after_canonicalization": True,
        "rms_evidence": {
            "sample_count": int(write_info.frames), "target_rms": target_rms,
            "input_rms": float(baseline["input_rms"]),
            "canonical_baseline_output_rms": float(baseline["output_rms"]),
            "output_rms": output_rms, "output_rms_error": abs(output_rms-target_rms),
            "peak_ceiling": None, "peak_limiter_applied": False,
        },
        "projection_evidence": {
            **hook_state, "declared_postprojection_bound_db": float(constructor["max_abs_mask_db"]),
            "postprojection_within_bound": True,
            "projection_is_global_scale": True,
        },
        "transform_summary": transform_info,
        "runtime": {
            "python_version": platform.python_version(), "python_executable": str(Path(sys.executable).resolve()),
            "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "numpy_version": np.__version__, "device": str(torch.device(device)),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        },
    }
    for field in method_point_fields:
        if field in config:
            packet[field] = config[field]
    packet["parent_row_id"] = parent_row.get("row_id")
    packet["sample_id"] = parent_row.get("sample_id")
    packet["speaker_id"] = parent_row.get("speaker_id")
    packet["seed"] = parent_row.get("seed")
    packet["self_sha256"] = sha256_bytes(canonical_bytes(packet))
    raw = json.dumps(packet, sort_keys=True, ensure_ascii=False, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(receipt_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    receipt_pin = {"path": str(receipt_out.resolve(strict=True)), "bytes": len(raw),
                   "sha256": sha256_bytes(raw), "self_sha256": packet["self_sha256"]}
    generator_condition_input = {
        **packet["output_generator_drive_wav"],
        "pcm_sha256": packet["output_generator_drive_wav"]["pcm_float32_sha256"],
        "proposal_id": config.get("proposal_id"), "pair_id": config.get("pair_id"),
        "side": config.get("side"), "method_id": method["method_id"],
        "method_config_sha256": config_sha, "artifact_sha256": config["artifact_sha256"],
        "binding_sha256": config["binding_sha256"], "input_pins_sha256": config["input_pins_sha256"],
        "theta_sha256": config["theta_sha256"], "parent_row_id": parent_row.get("row_id"),
        "seed": parent_row.get("seed"),
        "application_receipt_raw_sha256": receipt_pin["sha256"],
        "application_receipt_self_sha256": receipt_pin["self_sha256"],
    }
    spsa_binding = {
        **{key: config[key] for key in method_point_fields if key in config},
        "method_id": method["method_id"], "method_config_sha256": config_sha,
        "parent_row_id": parent_row.get("row_id"), "seed": parent_row.get("seed"),
        "theta": list(config["theta"]), "theta_sha256": config["theta_sha256"],
        "artifact_sha256": config["artifact_sha256"], "binding_sha256": config["binding_sha256"],
        "input_pins_sha256": config["input_pins_sha256"],
        "source_snapshot_artifact_sha256": config["source_snapshot_artifact_sha256"],
        "source_seed_method_id": config["source_seed_method_id"],
        "source_snapshot_step": config["source_snapshot_step"],
        "application_receipt": receipt_pin,
        "application_receipt_raw_sha256": receipt_pin["sha256"],
        "application_receipt_self_sha256": receipt_pin["self_sha256"],
    }
    return {
        "application_receipt": receipt_pin,
        "application_packet": packet,
        "generator_condition_input": generator_condition_input,
        "spsa_binding": spsa_binding,
        "output_samples": output_samples,
        "output_pin": packet["output_generator_drive_wav"],
    }


__all__ = ["ApplicationError", "apply_snapshot_theta", "canonical_bytes", "sha256_bytes"]
