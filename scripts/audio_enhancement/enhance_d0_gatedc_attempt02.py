"""Apply the unselected Fit120 D0/GateDC head to generator-drive audio.

This entrypoint is intentionally separate from ``enhance_v3_attempt03.py``:
the D0 checkpoint has a distinct head/checkpoint schema and is still
UNSELECTED. Importing this module does not import Torch or load a checkpoint.
Execution requires an actual root-reviewed completion binding produced after
Fit40/Adam120; no selection record is accepted or synthesized here.

The canonical 16 kHz output is generator input only. Keep the complete
original audio file unchanged for delivery/scoring mux. ``gate_dc`` is an
explicit Fit-only same-theta score-control mode; it is never the default.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pathlib
import re
import stat
import sys
import time
from typing import Any


CONTRACT_SCHEMA = "cross_tfg_d0_gatedc_inference_source_contract_v1"
BINDING_SCHEMA = "cross_tfg_d0_gatedc_completed_fit_inference_binding_v1"
RESULT_SCHEMA = "cross_tfg_d0_gatedc_generator_drive_result_v1"
HEAD_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_d0_gate_dc_attempt01"
CHECKPOINT_SCHEMA = "cross_tfg_v3_d0_direction_gatedc_parameter_checkpoint_v1"
TRAINING_SCHEMA = "cross_tfg_v3_d0_direction_gatedc_training_record_v1"
COMPLETION_SCHEMA = "cross_tfg_fit40_native_transition_head_v3_d0_direction_gatedc_runner_attempt02"
COMPLETION_STATUS = "COMPLETE_FIT40_DIRECT_D0_DIRECTION_GATE_DC_120_STEPS"
ROOT_QC_STATUS = "PASS_ROOT_INDEPENDENT_D0_COMPLETION_QC"
TARGET_RMS = 0.0376838172675746
_HEX256 = re.compile(r"^[0-9a-f]{64}$")
_PIN_KEYS = {"path", "bytes", "sha256"}


class D0CLIError(ValueError):
    """Invalid lineage, runtime, audio, or output contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _pin_shape(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _PIN_KEYS:
        raise D0CLIError(f"{label} must be an exact path/bytes/sha256 pin")
    if not isinstance(value["path"], str) or not value["path"]:
        raise D0CLIError(f"{label}.path must be nonempty text")
    if type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise D0CLIError(f"{label}.bytes must be a positive integer")
    if not isinstance(value["sha256"], str) or not _HEX256.fullmatch(value["sha256"]):
        raise D0CLIError(f"{label}.sha256 must be lowercase hexadecimal SHA-256")
    return {"path": value["path"], "bytes": value["bytes"], "sha256": value["sha256"]}


def _reject_symlink_components(path: pathlib.Path, *, allow_missing_leaf: bool = False) -> None:
    absolute = path.absolute()
    parts = [absolute.anchor, *absolute.parts[1:]]
    cursor = pathlib.Path(absolute.anchor)
    for index, part in enumerate(parts[1:]):
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 2:
                return
            raise
        if stat.S_ISLNK(mode):
            raise D0CLIError(f"symlink path component is forbidden: {cursor}")


def _read_regular(
    path: pathlib.Path,
    *,
    maximum: int,
    expected_pin: dict[str, Any] | None = None,
    label: str,
) -> bytes:
    _reject_symlink_components(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise D0CLIError(f"cannot stat {label}: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise D0CLIError(f"{label} must be a regular non-symlink file: {path}")
    if info.st_size > maximum:
        raise D0CLIError(f"{label} exceeds its bounded byte limit")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise D0CLIError(f"{label} changed while being read")
    if expected_pin is not None:
        pin = _pin_shape(expected_pin, label)
        if len(raw) != pin["bytes"] or _sha(raw) != pin["sha256"]:
            raise D0CLIError(f"{label} does not match its actual whole-file pin")
    return raw


def _read_json(path: pathlib.Path, *, maximum: int, expected_pin: dict[str, Any] | None, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, maximum=maximum, expected_pin=expected_pin, label=label)
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D0CLIError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(obj, dict):
        raise D0CLIError(f"{label} must decode to an object")
    return obj, raw


def _verify_self_hash(obj: dict[str, Any], field: str, label: str) -> None:
    expected = obj.get(field)
    payload = {key: value for key, value in obj.items() if key != field}
    if not isinstance(expected, str) or expected != _sha(_canonical(payload)):
        raise D0CLIError(f"{label} canonical self-hash mismatch")


def _same_pin(left: Any, right: Any, label: str) -> dict[str, Any]:
    a = _pin_shape(left, f"{label}.left")
    b = _pin_shape(right, f"{label}.right")
    if a != b:
        raise D0CLIError(f"{label} pins differ")
    return a


def _resolved_pin_path(pin: dict[str, Any], label: str) -> pathlib.Path:
    normalized = _pin_shape(pin, label)
    path = pathlib.Path(normalized["path"])
    _reject_symlink_components(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise D0CLIError(f"{label} is not a regular file")
    return resolved


def _verify_pin_file(pin: dict[str, Any], *, label: str, maximum: int = 2 * 1024**3) -> bytes:
    path = _resolved_pin_path(pin, label)
    return _read_regular(path, maximum=maximum, expected_pin=pin, label=label)


def _verify_contract(contract_path: pathlib.Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    if not _HEX256.fullmatch(expected_sha256):
        raise D0CLIError("--expected-contract-sha256 must be a full lowercase SHA-256")
    contract, raw = _read_json(contract_path, maximum=256 * 1024, expected_pin=None, label="source contract")
    if _sha(raw) != expected_sha256:
        raise D0CLIError("source contract raw bytes differ from --expected-contract-sha256")
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != "SOURCE_CONTRACT_READY_DYNAMIC_COMPLETION_REQUIRED":
        raise D0CLIError("source contract schema/status is not the sealed D0 inference contract")
    _verify_self_hash(contract, "self_sha256", "source contract")
    cli_pin = _pin_shape(contract.get("cli_source"), "source_contract.cli_source")
    this_file = pathlib.Path(__file__).resolve(strict=True)
    runtime_cli_pin = {"path": str(this_file), "bytes": cli_pin["bytes"], "sha256": cli_pin["sha256"]}
    this_raw = _read_regular(this_file, maximum=1024 * 1024, expected_pin=runtime_cli_pin, label="D0 inference CLI source")
    if this_raw != this_file.read_bytes():
        raise D0CLIError("D0 inference CLI changed during source verification")
    return contract, raw


def _verify_source_stage(contract: dict[str, Any], receipt_path: pathlib.Path) -> tuple[dict[str, Any], bytes, pathlib.Path, dict[str, dict[str, Any]]]:
    receipt_pin = _pin_shape(contract.get("source_stage_receipt"), "source_contract.source_stage_receipt")
    receipt, receipt_raw = _read_json(receipt_path, maximum=2 * 1024**2, expected_pin=receipt_pin, label="source-stage receipt")
    _verify_self_hash(receipt, "self_sha256", "source-stage receipt")
    if receipt.get("schema") != "fit40_d0_gatedc_source_stage_receipt_attempt02_v1" or receipt.get("status") != "PASS_SOURCE_CLOSURE_EXTRACTED_NO_MODEL_OR_GPU":
        raise D0CLIError("source-stage receipt is not the actual source-only D0 closure receipt")
    scope = receipt.get("scope")
    if not isinstance(scope, dict) or any(scope.get(key) is not False for key in (
        "cuda_initialized", "model_imported", "optimizer_updated", "render_or_score",
        "development_audio_opened", "confirmation_audio_opened",
    )):
        raise D0CLIError("source-stage receipt scope is not source-only")
    root = pathlib.Path(str(receipt.get("destination", "")))
    _reject_symlink_components(root)
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise D0CLIError("source-stage extraction root is absent, symlinked, or noncanonical")
    actual_members = receipt.get("members")
    contract_members = contract.get("source_members")
    if not isinstance(actual_members, list) or not isinstance(contract_members, list) or len(actual_members) != 23 or len(contract_members) != 23:
        raise D0CLIError("source-stage contract requires the exact 23-file source closure")
    receipt_by_rel: dict[str, dict[str, Any]] = {}
    for member in actual_members:
        if not isinstance(member, dict):
            raise D0CLIError("source-stage receipt contains a malformed member")
        rel = member.get("relative_path")
        if not isinstance(rel, str) or rel in receipt_by_rel:
            raise D0CLIError("source-stage receipt has duplicate/invalid relative member paths")
        receipt_by_rel[rel] = member
    contract_by_rel: dict[str, dict[str, Any]] = {}
    for member in contract_members:
        if not isinstance(member, dict) or not isinstance(member.get("relative_path"), str):
            raise D0CLIError("source contract has a malformed source member")
        rel = member["relative_path"]
        if rel in contract_by_rel:
            raise D0CLIError("source contract has duplicate member paths")
        contract_by_rel[rel] = member
    if set(receipt_by_rel) != set(contract_by_rel):
        raise D0CLIError("source contract members differ from the actual extracted 23-file closure")
    verified: dict[str, dict[str, Any]] = {}
    for rel, expected in contract_by_rel.items():
        parts = pathlib.PurePosixPath(rel).parts
        if not parts or pathlib.PurePosixPath(rel).is_absolute() or any(part in ("", ".", "..") for part in parts):
            raise D0CLIError(f"unsafe source member path: {rel}")
        target = root.joinpath(*parts)
        _reject_symlink_components(target)
        expected_pin = {"path": str(target), "bytes": expected.get("bytes"), "sha256": expected.get("sha256")}
        actual_receipt_row = receipt_by_rel[rel]
        _same_pin(expected_pin, {"path": actual_receipt_row.get("path"), "bytes": actual_receipt_row.get("bytes"), "sha256": actual_receipt_row.get("sha256")}, f"source member {rel}")
        raw = _read_regular(target, maximum=16 * 1024**2, expected_pin=expected_pin, label=f"source member {rel}")
        verified[rel] = {"path": str(target), "bytes": len(raw), "sha256": _sha(raw)}
    root_qc = contract.get("source_stage_root_qc")
    if not isinstance(root_qc, dict):
        raise D0CLIError("source contract lacks the independent root QC summary")
    _pin_shape(root_qc.get("artifact"), "source_contract.source_stage_root_qc.artifact")
    if (
        root_qc.get("status") != "PASS_ACTUAL_STAGE_RECEIPT_AND_ALL23_SOURCE_BYTES"
        or root_qc.get("receipt_raw_sha256") != receipt_pin["sha256"]
        or root_qc.get("member_count") != 23
        or root_qc.get("all_no_gpu_scope_flags_false") is not True
    ):
        raise D0CLIError("independent source-stage QC does not bind the exact receipt and all 23 files")
    return receipt, receipt_raw, root, verified


def _check_runtime_policy(torch: Any, np: Any, completion: dict[str, Any], *, requested_device: str | None) -> dict[str, Any]:
    identity = completion.get("gpu_identity")
    if not isinstance(identity, dict):
        raise D0CLIError("completion lacks its actual GPU/runtime identity")
    expected_python = identity.get("python_version")
    if list(sys.version_info[:3]) != expected_python:
        raise D0CLIError("Python version differs from the actual Fit120 runtime")
    if str(torch.__version__) != identity.get("torch_version") or str(np.__version__) != identity.get("numpy_version"):
        raise D0CLIError("Torch/NumPy versions differ from the actual Fit120 runtime")
    if str(torch.version.cuda) != identity.get("torch_cuda_version"):
        raise D0CLIError("CUDA runtime version differs from the actual Fit120 runtime")
    required_policy = {
        "torch_use_deterministic_algorithms": True,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    if any(identity.get(key) is not expected for key, expected in required_policy.items()):
        raise D0CLIError("actual Fit120 completion does not record the required strict deterministic policy")
    device_literal = str(identity.get("runtime_device", ""))
    if not re.fullmatch(r"cuda(?::[0-9]+)?", device_literal):
        raise D0CLIError("actual Fit120 runtime device is not an explicit CUDA device")
    if requested_device is not None and requested_device != device_literal:
        raise D0CLIError("requested CUDA device differs from the actual Fit120 device literal")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != identity.get("cublas_workspace_config"):
        raise D0CLIError("CUBLAS_WORKSPACE_CONFIG differs from actual Fit120 runtime")
    for env_name, key in (("OMP_NUM_THREADS", "omp_num_threads"), ("MKL_NUM_THREADS", "mkl_num_threads"), ("OPENBLAS_NUM_THREADS", "openblas_num_threads")):
        if os.environ.get(env_name) != identity.get(key):
            raise D0CLIError(f"{env_name} differs from actual Fit120 runtime")
    # Apply the exact recorded thread and deterministic policy only when a
    # setting differs, then require an exact postcondition. This runs directly
    # after import, before any model/tensor work in this entrypoint.
    intra = int(identity.get("torch_num_threads", -1))
    inter = int(identity.get("torch_num_interop_threads", -1))
    if intra < 1 or inter < 1:
        raise D0CLIError("Fit120 thread settings are invalid")
    if torch.get_num_threads() != intra:
        torch.set_num_threads(intra)
    if torch.get_num_interop_threads() != inter:
        try:
            torch.set_num_interop_threads(inter)
        except RuntimeError as exc:
            raise D0CLIError(f"cannot establish Fit120 inter-op threads: {exc}") from exc
    if identity.get("torch_use_deterministic_algorithms") is True and (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = bool(identity.get("tf32_matmul"))
    torch.backends.cudnn.allow_tf32 = bool(identity.get("tf32_cudnn"))
    torch.backends.cudnn.benchmark = bool(identity.get("cudnn_benchmark"))
    torch.backends.cudnn.deterministic = bool(identity.get("cudnn_deterministic"))
    if not torch.cuda.is_available():
        raise D0CLIError("the exact trained CUDA backend is unavailable; CPU fallback is forbidden")
    device = torch.device(device_literal)
    expected_index = int(identity.get("cuda_device_index", -1))
    if device.index is not None and device.index != expected_index:
        raise D0CLIError("Fit120 CUDA device index differs from its runtime receipt")
    torch.cuda.set_device(device)
    if torch.cuda.current_device() != expected_index:
        raise D0CLIError("current CUDA device index differs from the Fit120 runtime receipt")
    props = torch.cuda.get_device_properties(expected_index)
    uuid = getattr(props, "uuid", None)
    if uuid is None:
        raise D0CLIError("Torch CUDA properties do not expose the UUID required by the Fit120 binding")
    uuid_text = str(uuid)
    if uuid_text.startswith("GPU-"):
        uuid_text = uuid_text[4:]
    try:
        uuid_hex = uuid_text.replace("-", "").lower()
    except Exception as exc:
        raise D0CLIError("cannot normalize current GPU UUID") from exc
    expected_uuid = str(identity.get("uuid", ""))
    if expected_uuid.startswith("GPU-"):
        expected_uuid = expected_uuid[4:]
    if uuid_hex != expected_uuid.replace("-", "").lower():
        raise D0CLIError("current GPU UUID differs from the exact Fit120 GPU")
    if str(props.name) != identity.get("device_name"):
        raise D0CLIError("current GPU device name differs from Fit120")
    for flag_name, actual, key in (
        ("torch_use_deterministic_algorithms", torch.are_deterministic_algorithms_enabled(), "torch_use_deterministic_algorithms"),
        ("tf32_matmul", torch.backends.cuda.matmul.allow_tf32, "tf32_matmul"),
        ("tf32_cudnn", torch.backends.cudnn.allow_tf32, "tf32_cudnn"),
        ("cudnn_benchmark", torch.backends.cudnn.benchmark, "cudnn_benchmark"),
        ("cudnn_deterministic", torch.backends.cudnn.deterministic, "cudnn_deterministic"),
    ):
        if actual != identity.get(key):
            raise D0CLIError(f"runtime determinism flag differs from Fit120: {flag_name}")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise D0CLIError("deterministic algorithms are active in warn-only mode, not the Fit120 strict mode")
    if torch.get_num_threads() != intra or torch.get_num_interop_threads() != inter:
        raise D0CLIError("Torch thread settings do not match Fit120 after idempotent setup")
    if not torch.cuda.is_initialized():
        raise D0CLIError("actual Fit120 CUDA device did not initialize")
    return {
        "runtime_device": str(device),
        "cuda_device_index": int(torch.cuda.current_device()),
        "device_name": str(props.name),
        "uuid": identity["uuid"],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "numpy_version": str(np.__version__),
        "python_version": list(sys.version_info[:3]),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def _validate_completed_binding(
    binding_path: pathlib.Path,
    expected_binding_sha256: str,
    contract: dict[str, Any],
    source_root: pathlib.Path,
    source_members: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, bytes]]:
    if not _HEX256.fullmatch(expected_binding_sha256):
        raise D0CLIError("--expected-binding-sha256 must be a full lowercase SHA-256")
    binding, binding_raw = _read_json(binding_path, maximum=512 * 1024, expected_pin=None, label="actual completion binding")
    if _sha(binding_raw) != expected_binding_sha256:
        raise D0CLIError("completion binding raw SHA differs from its expected pin")
    _verify_self_hash(binding, "self_sha256", "actual completion binding")
    if binding.get("schema") != BINDING_SCHEMA or binding.get("status") != ROOT_QC_STATUS:
        raise D0CLIError("actual completion binding is not the root-reviewed D0 Fit120 binding")
    root_qc_pin = _pin_shape(binding.get("root_independent_qc"), "binding.root_independent_qc")
    root_qc, _ = _read_json(pathlib.Path(root_qc_pin["path"]), maximum=256 * 1024, expected_pin=root_qc_pin, label="root independent completion QC")
    if root_qc.get("status") != ROOT_QC_STATUS or root_qc.get("completion_raw_sha256") != binding.get("completion", {}).get("sha256"):
        raise D0CLIError("root independent QC does not bind the exact D0 completion")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict):
        raise D0CLIError("completion binding artifacts are missing")
    loaded: dict[str, bytes] = {}
    decoded: dict[str, dict[str, Any]] = {}
    for name in ("completion", "checkpoint", "training_record", "runtime_source_pins", "head_config"):
        pin = _pin_shape(artifacts.get(name), f"binding.artifacts.{name}")
        if name == "completion":
            # The root-QC input is not trusted from the binding alone; the expected
            # SHA supplied on argv pins this dynamic terminal before all child pins.
            expected = binding.get("completion", {}).get("sha256")
            if expected != pin["sha256"]:
                raise D0CLIError("completion binding has conflicting raw pins")
        obj, raw = _read_json(pathlib.Path(pin["path"]), maximum=4 * 1024**2, expected_pin=pin, label=f"actual {name}")
        decoded[name] = obj
        loaded[name] = raw
    completion = decoded["completion"]
    checkpoint = decoded["checkpoint"]
    training = decoded["training_record"]
    runtime_pins = decoded["runtime_source_pins"]
    head_config = decoded["head_config"]
    # These two artifacts are part of the completed Fit120 evidence. They are
    # loaded from the producer's actual terminal pins instead of being inferred
    # from summary booleans in the completion object.
    extra_completion_pins = {
        "strict_backward_qualification": _pin_shape(
            completion.get("strict_backward_qualification"),
            "completion.strict_backward_qualification",
        ),
        "fit40_gate_dc_control": _pin_shape(
            completion.get("fit40_gate_dc_control"), "completion.fit40_gate_dc_control"
        ),
    }
    for name, pin in extra_completion_pins.items():
        bound_pin = _pin_shape(artifacts.get(name), f"binding.artifacts.{name}")
        _same_pin(pin, bound_pin, f"completion/binding {name}")
        obj, raw = _read_json(
            pathlib.Path(pin["path"]), maximum=4 * 1024**2,
            expected_pin=pin, label=f"actual {name}",
        ) if name == "strict_backward_qualification" else (None, _verify_pin_file(pin, label=f"actual {name}", maximum=16 * 1024**2))
        loaded[name] = raw
        if obj is not None:
            decoded[name] = obj
    if completion.get("schema") != COMPLETION_SCHEMA or completion.get("status") != COMPLETION_STATUS:
        raise D0CLIError("actual completion is not the full D0 Fit40/Adam120 terminal")
    if (completion.get("completed_optimizer_steps"), completion.get("fit_rows"), completion.get("fit_speakers"), completion.get("zero_gate_rows_passed")) != (120, 40, 20, 40):
        raise D0CLIError("actual completion does not preserve the frozen Fit40/120-step scope")
    if any(completion.get(k) is not False for k in ("selection_record_created", "development_audio_opened", "confirmation_audio_opened")):
        raise D0CLIError("actual completion includes selection, development, or confirmation access")
    if training.get("schema") != TRAINING_SCHEMA or training.get("status") != "COMPLETE":
        raise D0CLIError("training record is not the complete D0/GateDC schema")
    if (training.get("completed_optimizer_steps"), training.get("fit_row_count"), training.get("fit_speaker_count")) != (120, 40, 20):
        raise D0CLIError("training record does not contain the frozen 40-row/120-step denominator")
    if training.get("selection_status") != "NOT_SELECTED" or training.get("selection_record_created") is not False or training.get("development_audio_opened") is not False or training.get("confirmation_audio_opened") is not False:
        raise D0CLIError("D0 checkpoint must remain unselected and Fit-only")
    if training.get("checkpoint_sha256") != _sha(loaded["checkpoint"]) or completion.get("training_record_raw_sha256") != _sha(loaded["training_record"]):
        raise D0CLIError("completion/training raw artifact hashes do not close")
    if completion.get("checkpoint", {}).get("sha256") != _sha(loaded["checkpoint"]):
        raise D0CLIError("completion does not bind exact checkpoint bytes")
    if completion.get("runtime_source_pins", {}).get("sha256") != _sha(loaded["runtime_source_pins"]):
        raise D0CLIError("completion does not bind exact runtime-source pin bytes")
    if completion.get("training_record", {}).get("sha256") != _sha(loaded["training_record"]):
        raise D0CLIError("completion does not bind exact training-record bytes")
    if training.get("runtime_source_pins_sha256") != _sha(loaded["runtime_source_pins"]):
        raise D0CLIError("training record does not bind exact runtime-source pin bytes")
    strict_pin = extra_completion_pins["strict_backward_qualification"]
    strict_runtime = runtime_pins.get("strict_backward_qualification")
    strict_training = training.get("strict_backward_qualification")
    strict_obj = decoded["strict_backward_qualification"]
    if not isinstance(strict_runtime, dict) or not isinstance(strict_training, dict):
        raise D0CLIError("training/runtime records lack strict-backward lineage")
    _same_pin(strict_runtime.get("artifact"), strict_pin, "runtime strict-backward artifact")
    _same_pin(strict_training.get("artifact"), strict_pin, "training strict-backward artifact")
    strict_result = strict_obj.get("full_objective_backward")
    if (
        strict_obj.get("schema") != "cross_tfg_fit40_strict_backward_operator_qualification_v1"
        or strict_obj.get("status") != "PASS_FULL_OBJECTIVE_BACKWARD_BEFORE_OPTIMIZER"
        or not isinstance(strict_result, dict)
        or strict_result.get("gradient_shape") != [3, 3]
        or strict_result.get("gradient_dtype") != "float32; no cast"
        or type(strict_result.get("gradient_nonzero_scalar_count")) is not int
        or strict_result["gradient_nonzero_scalar_count"] <= 0
        or strict_result.get("gradient_positive_signal") is not True
        or not isinstance(strict_result.get("gradient_sha256"), str)
        or not _HEX256.fullmatch(strict_result["gradient_sha256"])
        or not isinstance(strict_result.get("loss_float32"), (int, float))
        or not math.isfinite(float(strict_result["loss_float32"]))
        or strict_result.get("zero_head_unchanged") is not True
        or strict_result.get("v3_head_adam_created") is not False
        or strict_result.get("optimizer_updates") != 0
        or strict_result.get("strict_deterministic_algorithms") is not True
        or strict_result.get("warn_only") is not False
        or strict_result.get("all_three_syncnet_pool_backward_layers_exercised") is not True
        or strict_result.get("native_crop_bilinear_backward_exercised") is not True
        or strict_result.get("v3_reflection_pad2d_backward_exercised") is not True
        or strict_result.get("v3_reflection_pad1d_forward_exercised") is not True
        or strict_result.get("v3_reflection_pad2d_forward_bit_identical") is not True
        or strict_result.get("all_backward_replays_bit_identical") is not True
    ):
        raise D0CLIError("actual strict-backward artifact lacks finite nonzero whole-objective gradient evidence")
    if strict_training.get("status") != strict_obj.get("status") or strict_training.get("result") != strict_result:
        raise D0CLIError("training record's strict-backward summary differs from its pinned artifact")
    if strict_runtime.get("result") != strict_result:
        raise D0CLIError("runtime pins' strict-backward summary differs from its pinned artifact")
    if training.get("head_implementation_sha256") != checkpoint.get("implementation", {}).get("sha256"):
        raise D0CLIError("training record/head implementation source pin differs")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("head_schema") != HEAD_SCHEMA:
        raise D0CLIError("D0 checkpoint/head schema is not supported by this entrypoint")
    if checkpoint.get("selection_status") != "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED":
        raise D0CLIError("D0 checkpoint is not explicitly unselected")
    if checkpoint.get("constructor_sha256") != training.get("constructor_sha256") or checkpoint.get("constructor_sha256") != _sha(_canonical(head_config.get("head_constructor"))):
        raise D0CLIError("checkpoint/training/head-config constructor hashes do not agree")
    if head_config.get("schema") != "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_config_v1":
        raise D0CLIError("D0 runtime head constructor source config schema differs")
    runtime_head_config = _pin_shape(runtime_pins.get("head_config"), "runtime_source_pins.head_config")
    _same_pin(artifacts["head_config"], runtime_head_config, "runtime head configuration")
    if training.get("head_config_sha256") is not None and training.get("head_config_sha256") != runtime_head_config["sha256"]:
        raise D0CLIError("training record head-config hash differs from runtime pins")
    runtime_head = runtime_pins.get("training_modules", {}).get("v3_head")
    implementation = _pin_shape(checkpoint.get("implementation"), "checkpoint.implementation")
    _same_pin(runtime_head, implementation, "D0 head implementation")
    d0_member = source_members.get("scripts/audio_enhancement/shared_recipient_transition_head_v3_d0_gatedc_attempt01.py")
    if d0_member is None:
        raise D0CLIError("source receipt lacks the exact D0 head module")
    if implementation.get("bytes") != d0_member["bytes"] or implementation.get("sha256") != d0_member["sha256"]:
        raise D0CLIError("D0 checkpoint implementation is not the staged source receipt head")
    if pathlib.Path(implementation["path"]).resolve(strict=True) != pathlib.Path(d0_member["path"]).resolve(strict=True):
        raise D0CLIError("D0 checkpoint implementation path differs from staged source receipt")
    for module_name in ("audio_io", "baselines"):
        source_pin = _pin_shape(runtime_pins.get("training_modules", {}).get(module_name), f"runtime_source_pins.training_modules.{module_name}")
        rel = pathlib.Path(source_pin["path"]).resolve(strict=True).relative_to(source_root.resolve(strict=True)).as_posix()
        staged = source_members.get(rel)
        if staged is None or (source_pin["bytes"], source_pin["sha256"]) != (staged["bytes"], staged["sha256"]):
            raise D0CLIError(f"runtime module {module_name} differs from staged source receipt")
    training_bindings = checkpoint.get("training_artifact_bindings")
    if not isinstance(training_bindings, dict) or training_bindings.get("completed_optimizer_steps") != 120:
        raise D0CLIError("checkpoint lacks the exact Fit120 training-artifact bindings")
    for completion_key, training_key in (("training_steps", "training_steps_jsonl_sha256"), ("training_score_arrays", "training_score_arrays_sha256"), ("fit40_gate_dc_control", "fit40_gate_dc_control_sha256"), ("runtime_source_pins", "runtime_source_pins_sha256")):
        completion_art = _pin_shape(completion.get(completion_key), f"completion.{completion_key}")
        if training_bindings.get(training_key) != completion_art["sha256"]:
            raise D0CLIError(f"checkpoint does not bind the actual completion artifact: {completion_key}")
    control = training.get("gate_dc_control")
    gate_pin = _pin_shape(completion.get("fit40_gate_dc_control"), "completion.fit40_gate_dc_control")
    if not isinstance(control, dict) or control.get("status") != "COMPLETE_FIT40_SCORE_ONLY_SAME_THETA" or control.get("branch_rows") != 80 or control.get("optimizer_updates") != 0:
        raise D0CLIError("training record lacks the predeclared no-update Fit-only GateDC control")
    _same_pin(control.get("artifact"), gate_pin, "same-theta GateDC ledger")
    runtime_control = runtime_pins.get("fit40_gate_dc_control")
    if not isinstance(runtime_control, dict):
        raise D0CLIError("runtime source pins lack the GateDC score-only artifact link")
    _same_pin(runtime_control.get("artifact"), gate_pin, "runtime same-theta GateDC ledger")
    gate_raw = loaded["fit40_gate_dc_control"]
    try:
        gate_lines = gate_raw.decode("utf-8").splitlines()
        if any(not line for line in gate_lines):
            raise D0CLIError("GateDC JSONL contains a blank record")
        gate_rows = [json.loads(line) for line in gate_lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D0CLIError("GateDC control artifact is not valid JSONL") from exc
    fit_rows = training.get("fit_rows")
    if not isinstance(fit_rows, list) or len(fit_rows) != 40:
        raise D0CLIError("training record does not preserve the exact 40-row Fit roster")
    expected_row_ids = [row.get("row_id") for row in fit_rows if isinstance(row, dict)]
    if len(expected_row_ids) != 40 or len(set(expected_row_ids)) != 40:
        raise D0CLIError("training Fit40 row identities are missing or duplicated")
    parameter_sha = _sha(_canonical(checkpoint.get("parameter")))
    expected_pairs = [(row_id, mode) for row_id in expected_row_ids for mode in ("dynamic", "gate_dc")]
    actual_pairs = []
    for row in gate_rows:
        if not isinstance(row, dict):
            raise D0CLIError("GateDC JSONL contains a non-object record")
        if (
            row.get("schema") != "cross_tfg_fit40_gate_dc_same_theta_score_v1"
            or row.get("gate_dc_control_is_training_objective") is not False
            or row.get("same_frozen_parameter_sha256") != parameter_sha
            or type(row.get("source_sample_count")) is not int
            or row["source_sample_count"] <= 0
            or not isinstance(row.get("source_pcm_float32_sha256"), str)
            or not _HEX256.fullmatch(row["source_pcm_float32_sha256"])
            or not isinstance(row.get("score_metrics"), dict)
        ):
            raise D0CLIError("GateDC JSONL row has invalid source/theta/scope evidence")
        actual_pairs.append((row.get("row_id"), row.get("gate_mode")))
    if len(gate_rows) != 80 or actual_pairs != expected_pairs:
        raise D0CLIError("GateDC score-only ledger is not the exact ordered 40-row by two-mode result")
    fit_source_by_id = {row["row_id"]: row.get("natural_pcm") for row in fit_rows}
    for row in gate_rows:
        natural = fit_source_by_id[row["row_id"]]
        if not isinstance(natural, dict) or (
            row.get("source_pcm_float32_sha256") != natural.get("pcm_sha256")
            or row.get("source_sample_count") != natural.get("sample_count")
        ):
            raise D0CLIError("GateDC ledger source does not match the actual Fit40 natural PCM identity")
    contract_binding = binding.get("artifact_bindings")
    if not isinstance(contract_binding, dict):
        raise D0CLIError("actual completion binding lacks its artifact cross-links")
    for name in ("completion", "checkpoint", "training_record", "runtime_source_pins", "head_config", "strict_backward_qualification", "fit40_gate_dc_control"):
        _same_pin(contract_binding.get(name), artifacts[name], f"binding artifact {name}")
    if binding.get("source_stage_receipt", {}).get("sha256") != contract.get("source_stage_receipt", {}).get("sha256"):
        raise D0CLIError("actual completion binding is for a different source-stage receipt")
    return binding, completion, checkpoint, training, runtime_pins, loaded


def _write_exclusive(path: pathlib.Path, raw: bytes) -> None:
    _reject_symlink_components(path, allow_missing_leaf=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _readback_float_wav(path: pathlib.Path, expected: Any, audio_io: Any) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, maximum=2**31 - 1, expected_pin=None, label="output generator-drive WAV")
    reread, info = audio_io.read_canonical_wav(path, target_rate=16_000)
    import numpy as np
    if reread.dtype != np.float32 or reread.shape != expected.shape or not np.array_equal(reread, expected):
        raise D0CLIError("written output WAV did not roundtrip to bit-identical canonical PCM")
    if info.source_sample_rate != 16_000 or info.source_channels != 1 or info.source_encoding != "FLOAT_32":
        raise D0CLIError("written output WAV header is not mono 16 kHz float32")
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": _sha(raw), "pcm_float32_sha256": info.pcm_float32_sha256, "sample_count": int(reread.size), "sample_rate_hz": 16_000, "channels": 1, "encoding": info.source_encoding}, reread


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract, contract_raw = _verify_contract(args.contract, args.expected_contract_sha256)
    receipt, receipt_raw, source_root, source_members = _verify_source_stage(contract, args.source_stage_receipt)
    binding, completion, checkpoint, training, runtime_pins, loaded_artifacts = _validate_completed_binding(
        args.binding, args.expected_binding_sha256, contract, source_root, source_members
    )
    for name in ("completion", "checkpoint", "training_record", "runtime_source_pins", "head_config"):
        expected_path = pathlib.Path(_pin_shape(binding["artifacts"][name], f"binding.artifacts.{name}")["path"])
        supplied = getattr(args, name)
        if supplied.resolve(strict=True) != expected_path.resolve(strict=True):
            raise D0CLIError(f"--{name.replace('_', '-')} path differs from root-reviewed binding")
    input_path = args.input.resolve(strict=True)
    for source_path in (args.input, args.completion, args.checkpoint, args.training_record, args.runtime_source_pins, args.head_config, args.source_stage_receipt, args.binding, args.contract):
        _reject_symlink_components(source_path)
    out_path = args.output.absolute()
    metadata_path = args.metadata.absolute()
    _reject_symlink_components(out_path, allow_missing_leaf=True)
    _reject_symlink_components(metadata_path, allow_missing_leaf=True)
    if not out_path.parent.is_dir() or not metadata_path.parent.is_dir():
        raise D0CLIError("output and metadata parent directories must already exist")
    if out_path.suffix.lower() != ".wav" or metadata_path.suffix.lower() != ".json" or out_path == metadata_path:
        raise D0CLIError("output must be .wav and metadata a distinct .json path")
    forbidden = {input_path.resolve(), args.binding.resolve(strict=True), args.completion.resolve(strict=True), args.checkpoint.resolve(strict=True), args.training_record.resolve(strict=True), args.runtime_source_pins.resolve(strict=True), args.head_config.resolve(strict=True), args.source_stage_receipt.resolve(strict=True), args.contract.resolve(strict=True)}
    if out_path.resolve(strict=False) in forbidden or metadata_path.resolve(strict=False) in forbidden:
        raise D0CLIError("output paths may not replace any input, binding, or provenance file")
    if out_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite output audio or metadata")
    input_before = input_path.stat()
    input_raw = _read_regular(input_path, maximum=2**31 - 1, expected_pin=None, label="original input WAV")
    input_file_sha = _sha(input_raw)

    # Runtime modules are not imported until every raw completion, checkpoint,
    # training, receipt, contract, and source-member pin has been checked.
    sys.path.insert(0, str(source_root))
    import numpy as np
    import torch
    runtime_observation = _check_runtime_policy(torch, np, completion, requested_device=args.device)
    audio_io = importlib.import_module("scripts.audio_enhancement.audio_io")
    baselines = importlib.import_module("scripts.audio_enhancement.baselines")
    head_module = importlib.import_module("scripts.audio_enhancement.shared_recipient_transition_head_v3_d0_gatedc_attempt01")
    for module, key in ((audio_io, "audio_io"), (baselines, "baselines"), (head_module, "v3_head")):
        raw_pin = _pin_shape(runtime_pins.get("training_modules", {}).get(key), f"runtime_source_pins.training_modules.{key}")
        module_path = pathlib.Path(module.__file__).resolve(strict=True)
        if module_path != pathlib.Path(raw_pin["path"]).resolve(strict=True):
            raise D0CLIError(f"imported {key} module path differs from Fit120 runtime pin")
        _read_regular(module_path, maximum=16 * 1024**2, expected_pin=raw_pin, label=f"imported {key} module")
    constructor = contract.get("expected_constructor_source")
    if not isinstance(constructor, dict) or constructor.get("schema") != "FROM_ACTUAL_PINNED_HEAD_CONFIG":
        raise D0CLIError("source contract must derive constructor from the actual Fit120 head-config file")
    constructor_obj, head_config_raw = _read_json(
        args.head_config, maximum=256 * 1024,
        expected_pin=_pin_shape(runtime_pins.get("head_config"), "runtime_source_pins.head_config"),
        label="actual Fit120 head config",
    )
    head_constructor = constructor_obj.get("head_constructor")
    if not isinstance(head_constructor, dict) or _sha(_canonical(head_constructor)) != checkpoint.get("constructor_sha256"):
        raise D0CLIError("actual Fit120 constructor is absent or differs from the checkpoint digest")
    parameter = checkpoint.get("parameter")
    required_parameter_keys = {"name", "shape", "dtype", "temporal_components", "non_reference_band_order", "values"}
    if not isinstance(parameter, dict) or set(parameter) != required_parameter_keys:
        raise D0CLIError("actual D0 checkpoint parameter payload is malformed")
    if parameter.get("name") != "band_contrasts_db" or parameter.get("shape") != [3, 3] or parameter.get("dtype") != "float32":
        raise D0CLIError("actual D0 checkpoint is not the frozen nine-scalar head")
    if parameter.get("temporal_components") != ["static", "positive_flux", "negative_flux"]:
        raise D0CLIError("actual D0 temporal component order differs")
    reference_band = head_constructor.get("reference_band_index")
    expected_band_order = [i for i in range(4) if i != reference_band]
    if parameter.get("non_reference_band_order") != expected_band_order:
        raise D0CLIError("checkpoint non-reference band order differs from the actual constructor")
    values = parameter.get("values")
    if not isinstance(values, list) or len(values) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in values):
        raise D0CLIError("D0 checkpoint must contain exactly a 3x3 parameter matrix")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for row in values for v in row):
        raise D0CLIError("D0 checkpoint parameter contains nonfinite or nonnumeric values")
    if args.gate_mode not in {"dynamic", "gate_dc"}:
        raise D0CLIError("gate mode must be dynamic or gate_dc")
    control_record = training.get("gate_dc_control", {})
    if args.gate_mode == "gate_dc" and (
        control_record.get("status") != "COMPLETE_FIT40_SCORE_ONLY_SAME_THETA"
        or control_record.get("optimizer_updates") != 0
        or control_record.get("branch_rows") != 80
        or completion.get("fit40_gate_dc_control") != control_record.get("artifact")
    ):
        raise D0CLIError("gate_dc is permitted only by the actual predeclared same-theta Fit score-control record")

    source, read_info = audio_io.read_canonical_wav(input_path, target_rate=16_000)
    after_read = input_path.stat()
    if (input_before.st_dev, input_before.st_ino, input_before.st_size, input_before.st_mtime_ns) != (after_read.st_dev, after_read.st_ino, after_read.st_size, after_read.st_mtime_ns) or _sha_file(input_path) != input_file_sha:
        raise D0CLIError("original input changed while it was being canonicalized")
    if source.dtype != np.float32 or source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise D0CLIError("canonicalized original input is not finite nonempty mono float32")
    baseline, baseline_info = baselines.apply_global_rms(
        source, target_rms=float(head_constructor["target_rms"]),
        peak_ceiling=head_constructor["peak_ceiling"],
        silence_epsilon=float(head_constructor["rms_epsilon"]),
    )
    if baseline.shape != source.shape or baseline.dtype != np.float32 or not np.isfinite(baseline).all():
        raise D0CLIError("canonical RMS baseline changed sample count or finite float32 type")
    if float(baseline_info.output_rms) < 0 or float(head_constructor["target_rms"]) != TARGET_RMS or head_constructor["peak_ceiling"] is not None:
        raise D0CLIError("actual D0 baseline must preserve the frozen unbounded fixed-RMS policy")
    device = torch.device(str(completion["gpu_identity"]["runtime_device"]))
    head = head_module.SharedRecipientTransitionHeadV3(**head_constructor).to(device=device)
    theta = np.asarray(values, dtype=np.float32)
    if theta.shape != (3, 3) or not np.isfinite(theta).all():
        raise D0CLIError("checkpoint theta did not convert to finite float32[3,3]")
    with torch.no_grad():
        if tuple(head.band_contrasts_db.shape) != (3, 3):
            raise D0CLIError("constructed D0 head has the wrong trainable parameter shape")
        head.band_contrasts_db.copy_(torch.as_tensor(theta, dtype=torch.float32, device=device))
        for parameter_tensor in head.parameters():
            parameter_tensor.requires_grad_(False)
    if head.configuration() != head_constructor:
        raise D0CLIError("constructed D0 head configuration differs from the actual Fit120 constructor")
    head.eval()
    source_tensor = torch.from_numpy(np.ascontiguousarray(source)).to(device=device, dtype=torch.float32)
    baseline_tensor = torch.from_numpy(np.ascontiguousarray(baseline)).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        result = head(source_tensor, baseline_tensor, gate_mode=args.gate_mode)
    if result.get("schema") != HEAD_SCHEMA or result.get("gate_mode") != args.gate_mode:
        raise D0CLIError("D0 head output schema or selected gate mode differs")
    output_tensor = result.get("output_pcm_float32")
    if not isinstance(output_tensor, torch.Tensor) or output_tensor.dtype != torch.float32 or output_tensor.ndim != 1 or output_tensor.numel() != source.size:
        raise D0CLIError("D0 output must be exact-length mono float32 on the original canonical clock")
    if not bool(torch.isfinite(output_tensor).all().item()):
        raise D0CLIError("D0 output contains nonfinite samples")
    output = np.ascontiguousarray(output_tensor.detach().cpu().numpy(), dtype=np.float32)
    if output.size != source.size:
        raise D0CLIError("D0 output changed the exact canonical input sample count")
    output_pcm_sha = audio_io.pcm_float32_sha256(output)
    parameter_sha = _sha(_canonical(parameter))
    # Write through a unique same-directory staging WAV, then link it into its
    # final name with exclusive-create semantics so a race cannot overwrite.
    temp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.{time.time_ns()}.staged.wav")
    if temp_path.exists() or temp_path.is_symlink():
        raise D0CLIError("unique output staging path unexpectedly exists")
    audio_io.write_float32_wav(temp_path, output, sample_rate=16_000, peak_ceiling=None, overwrite=False)
    try:
        os.link(temp_path, out_path)
    finally:
        temp_path.unlink(missing_ok=True)
    output_pin, reread = _readback_float_wav(out_path, output, audio_io)
    if not np.array_equal(reread, output):
        raise D0CLIError("final output WAV readback differs from the exact generator-drive PCM")
    input_after = input_path.stat()
    if (input_before.st_dev, input_before.st_ino, input_before.st_size, input_before.st_mtime_ns) != (input_after.st_dev, input_after.st_ino, input_after.st_size, input_after.st_mtime_ns) or _sha_file(input_path) != input_file_sha:
        raise D0CLIError("original source file changed during generator-drive enhancement")
    meta = {
        "schema": RESULT_SCHEMA,
        "status": "PASS_D0_GENERATOR_DRIVE_ONLY",
        "profile_role": "UNSELECTED_FIT_CHECKPOINT_INFERENCE",
        "head_schema": HEAD_SCHEMA,
        "checkpoint_selection_status": checkpoint["selection_status"],
        "gate_mode": args.gate_mode,
        "gate_dc_scope": "FIT_ONLY_SAME_THETA_SCORE_CONTROL" if args.gate_mode == "gate_dc" else None,
        "checkpoint_parameter_sha256": parameter_sha,
        "constructor_sha256": checkpoint["constructor_sha256"],
        "input_original_file": {"path": str(input_path), "bytes": len(input_raw), "sha256": input_file_sha, "source_sample_rate_hz": read_info.source_sample_rate, "source_channels": read_info.source_channels, "source_frames": read_info.source_frames, "source_encoding": read_info.source_encoding},
        "canonical_generator_input": {"sample_rate_hz": 16_000, "channels": 1, "sample_count": int(source.size), "pcm_float32_sha256": read_info.pcm_float32_sha256, "canonicalization": read_info.resampler},
        "rms_baseline": {"target_rms": float(head_constructor["target_rms"]), "peak_ceiling": None, "sample_count": int(baseline.size), "pcm_float32_sha256": audio_io.pcm_float32_sha256(baseline), "output_rms": float(baseline_info.output_rms), "gain": float(baseline_info.applied_gain)},
        "output_generator_drive_wav": output_pin,
        "output_pcm_float32_sha256": output_pcm_sha,
        "output_sample_count": int(output.size),
        "source_clock_preserved_after_canonicalization": True,
        "final_delivery_and_scoring_audio": "use the complete unchanged original input file; do not use generator-drive WAV as delivery audio",
        "training_selection_created": False,
        "explicit_input_audio_read": True,
        "input_scope": "only_the_explicit_input_path_recorded_as_input_original_file",
        "unrequested_development_or_confirmation_inventory_read": False,
        "optimizer_created_or_stepped": False,
        "render_or_score_performed": False,
        "runtime_identity": runtime_observation,
        "contract": {"path": str(args.contract.resolve()), "bytes": len(contract_raw), "sha256": _sha(contract_raw)},
        "binding": {"path": str(args.binding.resolve()), "bytes": args.binding.stat().st_size, "sha256": args.expected_binding_sha256},
        "source_stage_receipt": {"path": str(args.source_stage_receipt.resolve()), "bytes": len(receipt_raw), "sha256": _sha(receipt_raw), "self_sha256": receipt["self_sha256"], "member_count": len(source_members)},
        "completion_artifacts": {name: _pin_shape(binding["artifacts"][name], f"binding.artifacts.{name}") for name in ("completion", "checkpoint", "training_record", "runtime_source_pins", "head_config")},
        "cli_source_sha256": _sha(pathlib.Path(__file__).resolve().read_bytes()),
    }
    meta["self_sha256"] = _sha(_canonical(meta))
    metadata_raw = _canonical(meta) + b"\n"
    _write_exclusive(metadata_path, metadata_raw)
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the actual unselected D0/GateDC Fit120 head; output is generator-drive audio only."
    )
    parser.add_argument("--contract", required=True, type=pathlib.Path, help="sealed source contract")
    parser.add_argument("--expected-contract-sha256", required=True, help="full raw SHA-256 of the source contract")
    parser.add_argument("--binding", required=True, type=pathlib.Path, help="root-reviewed actual D0 completion binding")
    parser.add_argument("--expected-binding-sha256", required=True, help="full raw SHA-256 of the actual completion binding")
    parser.add_argument("--source-stage-receipt", required=True, type=pathlib.Path, help="actual no-GPU 23-member source-stage receipt")
    parser.add_argument("--completion", required=True, type=pathlib.Path, help="actual Fit40/Adam120 completion.json")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path, help="actual unselected D0 checkpoint JSON")
    parser.add_argument("--training-record", required=True, type=pathlib.Path, help="actual Fit120 training record JSON")
    parser.add_argument("--runtime-source-pins", required=True, type=pathlib.Path, help="actual training runtime source pins")
    parser.add_argument("--head-config", required=True, type=pathlib.Path, help="exact constructor input pinned by Fit120")
    parser.add_argument("--input", required=True, type=pathlib.Path, help="complete original audio file; left unchanged")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="new 16 kHz mono float32 generator-drive WAV")
    parser.add_argument("--metadata", required=True, type=pathlib.Path, help="new provenance JSON sidecar")
    parser.add_argument("--gate-mode", choices=("dynamic", "gate_dc"), default="dynamic", help="dynamic is the default; gate_dc is allowed only by the actual predeclared Fit-only same-theta score-control record")
    parser.add_argument("--device", default=None, help="optional explicit CUDA device literal; must equal the actual Fit120 device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"schema": RESULT_SCHEMA, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"schema": RESULT_SCHEMA, "status": result["status"], "output": result["output_generator_drive_wav"]["path"], "metadata": str(args.metadata.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
