"""Fit-only D0 post-projection score runner; no training or checkpoint selection."""
from __future__ import annotations
import argparse, hashlib, importlib, importlib.util, json, math, os, random, sys, time, traceback, uuid
from pathlib import Path, PurePosixPath
from typing import Any

RUNNER_SCHEMA = "cross_tfg_fit40_post_projection_score_runner_v1"
BINDING_SCHEMA = "cross_tfg_fit40_post_projection_score_input_binding_v1"
SOURCE_RECEIPT_SCHEMA = "fit40_d0_gatedc_source_stage_receipt_attempt02_v1"
SOURCE_RECEIPT_STATUS = "PASS_SOURCE_CLOSURE_EXTRACTED_NO_MODEL_OR_GPU"
SOURCE_MEMBER_COUNT = 23
FIT_SEED = 20261003
FIT_ROWS = 40
SAMPLE_RATE_HZ = 16000
TARGET_RMS = 0.0376838172675746
ALLOWED_OUTPUT_ROOT = Path("/dev/shm/cross_tfg_audio_head_20261003").resolve()
HEAD_RELATIVE_PATH = "scripts/audio_enhancement/shared_recipient_transition_head_v3_d0_gatedc_attempt01.py"
RUNNER_RELATIVE_PATH = "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_d0_direction_gatedc_attempt02.py"
HELPER04_SHA256 = "27a9eae5b52faa218c4ff383e742d3833694e60812a4af79606ad8207545df56"
SCORE_ADAPTER01_SHA256 = "1b6bd4dd292dca2daa9125acb67f6782e477075e6b87a79c9512897747578d54"
RUNTIME_FACTORY_SHA256 = "7903fbd9dfebf95bcb388add26a1967bc39ad855d3d995a4730d316dd0e806c4"
RUNTIME_FACTORY_BYTES = 20085
REQUIRED_ARTIFACTS = ("completion", "checkpoint", "training_record", "runtime_source_pins", "baseline_arrays", "zero_gate_arrays", "strict_backward_qualification", "training_score_arrays", "training_steps", "fit40_gate_dc_control", "fit_source_manifest", "native_geometry_manifest", "head_config", "parent_plan")
REQUIRED_LINEAGE = set(REQUIRED_ARTIFACTS) | {"head_implementation", "runner"}
REQUIRED_EARLY_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": str(FIT_SEED), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}

class ScoreRunError(RuntimeError):
    pass

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def pin_shape(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ScoreRunError(f"{label} must contain exactly path/bytes/sha256")
    if not isinstance(value["path"], str) or not value["path"] or not Path(value["path"]).is_absolute():
        raise ScoreRunError(f"{label}.path must be absolute")
    if type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise ScoreRunError(f"{label}.bytes must be a positive integer")
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ScoreRunError(f"{label}.sha256 must be lowercase full SHA-256")
    return {"path": value["path"], "bytes": value["bytes"], "sha256": digest}

def regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ScoreRunError(f"{label} must be absolute and nonsymlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ScoreRunError(f"{label} is not the exact regular file: {path}")
    return resolved

def read_pin(value: Any, label: str) -> tuple[dict[str, Any], bytes, Path]:
    pin = pin_shape(value, label)
    path = regular(Path(pin["path"]), label)
    raw = path.read_bytes()
    if len(raw) != pin["bytes"] or sha(raw) != pin["sha256"]:
        raise ScoreRunError(f"{label} raw bytes differ from pin")
    return pin, raw, path

def read_json_pin(value: Any, label: str) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    pin, raw, _ = read_pin(value, label)
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreRunError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(obj, dict):
        raise ScoreRunError(f"{label} must be a JSON object")
    return pin, raw, obj

def verify_self_hash(obj: dict[str, Any], label: str) -> None:
    recorded = obj.get("self_sha256")
    payload = {key: value for key, value in obj.items() if key != "self_sha256"}
    if not isinstance(recorded, str) or sha(canonical(payload)) != recorded:
        raise ScoreRunError(f"{label} self-hash mismatch")

def local_pin(path: Path) -> dict[str, Any]:
    path = regular(path, "source file")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)}

def verify_source_stage(binding: dict[str, Any]) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    _, _, receipt = read_json_pin(binding.get("source_stage_receipt"), "source-stage receipt")
    verify_self_hash(receipt, "source-stage receipt")
    if receipt.get("schema") != SOURCE_RECEIPT_SCHEMA or receipt.get("status") != SOURCE_RECEIPT_STATUS:
        raise ScoreRunError("source-stage receipt schema/status differs from the pinned source-only stage")
    scope = receipt.get("scope")
    if not isinstance(scope, dict) or any(scope.get(key) is not False for key in ("confirmation_audio_opened", "cuda_initialized", "development_audio_opened", "model_imported", "optimizer_updated", "render_or_score")):
        raise ScoreRunError("source-stage receipt does not prove its no-model/no-GPU scope")
    members = receipt.get("members")
    if not isinstance(members, list) or len(members) != SOURCE_MEMBER_COUNT:
        raise ScoreRunError("source-stage receipt must contain exactly 23 source members")
    root_text = binding.get("code_root")
    if not isinstance(root_text, str) or not Path(root_text).is_absolute():
        raise ScoreRunError("binding.code_root must be absolute")
    root = Path(root_text)
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root or receipt.get("destination") != str(root):
        raise ScoreRunError("code_root is not the exact source-stage destination")
    seen: set[str] = set()
    for i, member in enumerate(members):
        if not isinstance(member, dict):
            raise ScoreRunError(f"source member {i} is malformed")
        rel_text = member.get("relative_path")
        if not isinstance(rel_text, str) or "\\" in rel_text or not rel_text:
            raise ScoreRunError(f"source member {i} relative path is invalid")
        rel = PurePosixPath(rel_text)
        if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts) or rel_text in seen:
            raise ScoreRunError(f"source member {i} has unsafe/duplicate path")
        seen.add(rel_text)
        expected = root.joinpath(*rel.parts)
        if member.get("path") != str(expected):
            raise ScoreRunError(f"source member path does not match code_root: {rel_text}")
        read_pin({"path": str(expected), "bytes": member.get("bytes"), "sha256": member.get("sha256")}, f"source member {rel_text}")
    if HEAD_RELATIVE_PATH not in seen or RUNNER_RELATIVE_PATH not in seen:
        raise ScoreRunError("source-stage receipt omits the D0 runner or head")
    disk_members = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    if disk_members != seen:
        raise ScoreRunError(f"staged source tree differs from exact receipt closure (extra={sorted(disk_members-seen)}, missing={sorted(seen-disk_members)})")
    return root, {"head_implementation": local_pin(root / HEAD_RELATIVE_PATH), "runner": local_pin(root / RUNNER_RELATIVE_PATH)}, receipt
def verify_binding(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = regular(path, "input binding").read_bytes()
    binding = json.loads(raw)
    if not isinstance(binding, dict) or binding.get("schema") != BINDING_SCHEMA:
        raise ScoreRunError("input binding schema is invalid")
    verify_self_hash(binding, "input binding")
    if binding.get("status") != "ACTUAL_COMPLETION_BOUND_READY_FOR_FIT_ONLY_SCORE_DIAGNOSTIC":
        raise ScoreRunError("binding is not an actual-completion-bound ready record")
    if any(binding.get(key) is not False for key in ("development_audio_opened", "confirmation_audio_opened", "selection_record_created", "api_called")):
        raise ScoreRunError("binding does not preserve Fit-only/no-selection scope")
    artifacts = binding.get("artifacts")
    lineage = binding.get("source_lineage_pins")
    if not isinstance(artifacts, dict) or not set(REQUIRED_ARTIFACTS).issubset(artifacts):
        raise ScoreRunError("binding is missing required actual D0 artifacts")
    if not isinstance(lineage, dict) or not REQUIRED_LINEAGE.issubset(lineage):
        raise ScoreRunError("binding is missing actual D0 source-lineage pins")
    for name in REQUIRED_ARTIFACTS:
        art = pin_shape(artifacts[name], f"artifacts.{name}")
        lin = pin_shape(lineage[name], f"source_lineage_pins.{name}")
        if art != lin:
            raise ScoreRunError(f"artifact and lineage pins disagree: {name}")
        read_pin(art, f"artifact {name}")
    if binding.get("source_lineage_pins_sha256") != sha(canonical(lineage)):
        raise ScoreRunError("source-lineage canonical digest mismatch")
    return binding, raw

def verify_support(value: Any, label: str, expected_sha: str) -> tuple[dict[str, Any], Path]:
    pin = pin_shape(value, label)
    path = regular(Path(pin["path"]), label)
    if path.stat().st_size != pin["bytes"] or sha_file(path) != pin["sha256"] or pin["sha256"] != expected_sha:
        raise ScoreRunError(f"{label} does not match its sealed source bytes")
    return pin, path

def load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScoreRunError(f"cannot load pinned support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def json_bytes(payload: dict[str, Any], with_self_hash: bool = False) -> bytes:
    obj = dict(payload)
    if with_self_hash:
        obj.pop("self_sha256", None)
        obj["self_sha256"] = sha(canonical(obj))
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"

def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical(row) + b"\n" for row in rows)

def write_new(path: Path, raw: bytes) -> dict[str, Any]:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    return {"path": str(path.resolve(strict=True)), "bytes": path.stat().st_size, "sha256": sha_file(path)}

def npz_bytes(np: Any, arrays: dict[str, Any]) -> bytes:
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()

def create_output_dir(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ScoreRunError("output-dir must be absolute and nonsymlink")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(ALLOWED_OUTPUT_ROOT)
    except ValueError as exc:
        raise ScoreRunError("output-dir must remain beneath the dedicated Fit40 /dev/shm namespace") from exc
    if path.exists():
        raise ScoreRunError("output-dir must be absent; partial attempts are never resumed")
    if not path.parent.is_dir() or path.parent.resolve(strict=True) != path.parent:
        raise ScoreRunError("output-dir parent must already exist as a canonical directory")
    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    return path.resolve(strict=True)

def _check_completion(binding: dict[str, Any], artifacts: dict[str, Any]) -> None:
    completion = artifacts["completion"]
    if completion.get("schema") != "cross_tfg_fit40_native_transition_head_v3_d0_direction_gatedc_runner_attempt02" or completion.get("status") != "COMPLETE_FIT40_DIRECT_D0_DIRECTION_GATE_DC_120_STEPS":
        raise ScoreRunError("input completion is not the complete D0/GateDC Fit40 terminal")
    if (completion.get("fit_rows"), completion.get("fit_speakers"), completion.get("completed_optimizer_steps"), completion.get("zero_gate_rows_passed")) != (40, 20, 120, 40):
        raise ScoreRunError("completion does not bind exactly 40 Fit rows/20 speakers/120 updates")
    if any(completion.get(k) is not False for k in ("selection_record_created", "development_audio_opened", "confirmation_audio_opened")):
        raise ScoreRunError("completion is not an unselected Fit-only terminal")
    for name in ("checkpoint", "training_record", "runtime_source_pins", "baseline_arrays", "zero_gate_arrays", "strict_backward_qualification", "training_score_arrays", "training_steps", "fit40_gate_dc_control"):
        if completion.get(name) != binding["artifacts"].get(name):
            raise ScoreRunError(f"completion embeds a different artifact pin: {name}")
    if completion.get("training_record_raw_sha256") != binding["artifacts"]["training_record"].get("sha256"):
        raise ScoreRunError("completion raw training-record digest differs from the pinned training-record artifact")
    training = artifacts["training_record"]
    if (training.get("schema") != "cross_tfg_v3_d0_direction_gatedc_training_record_v1" or training.get("status") != "COMPLETE" or training.get("completed_optimizer_steps") != 120 or training.get("planned_optimizer_steps") != 120 or training.get("fit_row_count") != 40 or training.get("fit_speaker_count") != 20 or training.get("selection_status") != "NOT_SELECTED" or training.get("selection_record_created") is not False or training.get("development_audio_opened") is not False or training.get("confirmation_audio_opened") is not False):
        raise ScoreRunError("training record does not bind the complete unselected Fit40/120 scope")
    gate = training.get("gate_dc_control")
    if not isinstance(gate, dict) or gate.get("status") != "COMPLETE_FIT40_SCORE_ONLY_SAME_THETA" or gate.get("branch_rows") != 80 or gate.get("optimizer_updates") != 0:
        raise ScoreRunError("training record lacks its exact 80-row same-theta reference ledger")

def parse_gate_ledger(raw: bytes) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ScoreRunError(f"GateDC ledger contains blank line {line_no}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ScoreRunError(f"GateDC ledger line {line_no} is not an object")
        rows.append(value)
    if len(rows) != 80:
        raise ScoreRunError("GateDC ledger must contain exactly 80 ordered rows")
    return rows

def check_runtime_environment(core: Any, completion: dict[str, Any], device_literal: str, torch: Any, np: Any) -> dict[str, Any]:
    if list(sys.version_info[:3]) != [3, 10, 12] or str(torch.__version__) != "2.5.1+cu124" or str(np.__version__) != "1.26.4":
        raise ScoreRunError("runtime Python/Torch/NumPy differs from the qualified Fit40 environment")
    if not torch.cuda.is_available():
        raise ScoreRunError("this score diagnostic requires the exact Fit40 CUDA runtime; CPU fallback is forbidden")
    required_env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": str(FIT_SEED), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    for name, expected in required_env.items():
        if os.environ.get(name) != expected:
            raise ScoreRunError(f"runtime environment {name} is not the exact Fit40 value")
    if not os.environ.get("CROSS_TFG_PSF_ROOT"):
        raise ScoreRunError("CROSS_TFG_PSF_ROOT is required by the pinned MFCC runtime")
    if not torch.are_deterministic_algorithms_enabled() or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ScoreRunError("strict deterministic algorithms or 1/1 Torch threads are not active")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic:
        raise ScoreRunError("CUDA TF32/cuDNN determinism flags differ from the Fit40 runtime")
    device = torch.device(device_literal)
    if device.type != "cuda" or (device.index is not None and device.index >= torch.cuda.device_count()):
        raise ScoreRunError(f"requested CUDA device is unavailable: {device_literal}")
    current = torch.cuda.current_device()
    if device.index is not None and current != device.index:
        torch.cuda.set_device(device)
        current = torch.cuda.current_device()
    device = torch.device("cuda", current)
    props = torch.cuda.get_device_properties(current)
    identity = {"uuid": core._required_torch_gpu_uuid(props), "cuda_device_index": int(current), "device_name": str(props.name), "total_memory_bytes": int(props.total_memory), "torch_cuda_version": str(torch.version.cuda), "runtime_device": str(device), "torch_version": str(torch.__version__), "numpy_version": str(np.__version__), "python_version": list(sys.version_info[:3]), "torch_num_threads": int(torch.get_num_threads()), "torch_use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"), "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "torch_num_interop_threads": int(torch.get_num_interop_threads())}
    if canonical(identity) != canonical(completion.get("gpu_identity")):
        raise ScoreRunError("actual CUDA/runtime identity differs from the completed Fit40 training run")
    return identity

def check_early_environment(*, expected_psf_root: Any, expected_cuda_visible_devices: Any) -> None:
    """Require process-start settings before importing Torch or project modules."""
    for name, expected in REQUIRED_EARLY_ENV.items():
        if os.environ.get(name) != expected:
            raise ScoreRunError(f"runtime environment {name} must be {expected!r} before importing Torch")
    if not isinstance(expected_psf_root, str) or not expected_psf_root or os.environ.get("CROSS_TFG_PSF_ROOT") != expected_psf_root:
        raise ScoreRunError("CROSS_TFG_PSF_ROOT differs from the actual qualified MFCC source-root binding")
    if not isinstance(expected_cuda_visible_devices, str) or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_cuda_visible_devices:
        raise ScoreRunError("CUDA_VISIBLE_DEVICES differs from the actual Fit40 geometry runtime profile")
    forbidden = ("PYTHONPATH", "CROSS_TFG_STRICT_STARTUP_POLICY", "CROSS_TFG_STARTUP_POLICY", "CROSS_TFG_STARTUP_POLICY_RECEIPT", "CROSS_TFG_RUNTIME_COMPAT_PREFLIGHT")
    if any(os.environ.get(name) for name in forbidden):
        raise ScoreRunError("runtime must start without inherited Python path or strict-bootstrap bindings")
def check_source_module_pins(code_root: Path, runtime_source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected_paths = {
        "v2_runtime": "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v2_attempt02.py",
        "v4_runtime": "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py",
        "v3_head": HEAD_RELATIVE_PATH,
        "v3_v4_adapter": "scripts/audio_enhancement/v3_v4_native_runtime_adapter_v9_attempt01.py",
        "audio_io": "scripts/audio_enhancement/audio_io.py",
        "baselines": "scripts/audio_enhancement/baselines.py",
        "enhance_v3_cli": "scripts/audio_enhancement/enhance_v3_attempt02.py",
        "fit40_geometry_producer": "scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt05.py",
        "native_encoder_feature_probe": "scripts/experiments/cross_tfg_audio_head_20261003/native_encoder_probe_attempt02.py",
        "native_encoder_feature_contract": "scripts/experiments/cross_tfg_audio_head_20261003/native_encoder_probe_contract_v2.md",
    }
    modules = runtime_source.get("training_modules")
    if not isinstance(modules, dict):
        raise ScoreRunError("actual runtime-source artifact lacks training_modules")
    found = {}
    for name, relative in expected_paths.items():
        pin = pin_shape(modules.get(name), f"training_modules.{name}")
        target = (code_root / relative).resolve(strict=True)
        if target.is_symlink() or pin["path"] != str(target):
            raise ScoreRunError(f"runtime-source module path differs from staged source: {name}")
        actual = local_pin(target)
        if actual["bytes"] != pin["bytes"] or actual["sha256"] != pin["sha256"]:
            raise ScoreRunError(f"runtime-source module bytes differ from staged source: {name}")
        found[name] = actual
    for role, module_name in (("runner", "runner"), ("head_config", "head_config")):
        record = runtime_source.get(role)
        if not isinstance(record, dict):
            raise ScoreRunError(f"runtime-source artifact lacks {role}")
    return found

def check_geometry_profile(core: Any, plan: dict[str, Any], geometry: dict[str, Any], runtime: dict[str, Any], device_literal: str, identity: dict[str, Any], torch: Any, np: Any) -> None:
    producer = geometry.get("runtime_gpu_identity")
    if not isinstance(producer, dict) or producer.get("schema") != "cross_tfg_runtime_gpu_identity_v1":
        raise ScoreRunError("geometry producer GPU/runtime identity is missing")
    if len(geometry.get("rows", [])) != FIT_ROWS or any(canonical(row.get("runtime_gpu_identity")) != canonical(producer) for row in geometry["rows"]):
        raise ScoreRunError("geometry rows do not share the manifest's exact runtime identity")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    expected_device = {"uuid": identity["uuid"], "device_index": int(torch.cuda.current_device()), "device_name": str(props.name), "memory_total_bytes": int(props.total_memory), "memory_total_mib": int(props.total_memory) // (1024 * 1024), "device_literal": device_literal, "torch_device_name": str(props.name), "torch_total_memory_bytes": int(props.total_memory), "torch_compute_capability": [int(props.major), int(props.minor)]}
    for key, expected in expected_device.items():
        actual = producer.get(key)
        if key == "uuid" and actual is not None:
            actual = core._canonical_gpu_uuid(actual)
        if canonical(actual) != canonical(expected):
            raise ScoreRunError(f"geometry GPU identity differs from current qualified CUDA runtime: {key}")
    cfg = plan["runtime"]["full_context"]
    batches = {"video_frame_batch_size": int(cfg["video_frame_batch_size"]), "audio_embedding_batch_size": int(cfg["audio_embedding_batch_size"]), "syncnet_batch_size": int(cfg["syncnet_batch_size"])}
    profile = producer.get("runtime_profile")
    expected_profile = {
        "python_executable_literal": str(sys.executable), "python_version": ".".join(str(x) for x in sys.version_info[:3]),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__), "seed": FIT_SEED,
        "torch_num_threads": int(torch.get_num_threads()), "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "batch_policy": {"wav2lip_native_frame_batch_size": batches["video_frame_batch_size"], "syncnet_audio_embedding_batch_size": batches["audio_embedding_batch_size"], "syncnet_visual_embedding_batch_size": batches["syncnet_batch_size"]},
        "frame_rate": {"numerator": 25, "denominator": 1}, "source_sample_rate_hz": SAMPLE_RATE_HZ,
        "wav2lip_git_head": str(plan["runtime"]["wav2lip"]["git_head"]), "wav2lip_checkpoint_sha256": str(plan["runtime"]["wav2lip"]["checkpoint_sha256"]),
        "syncnet_git_head": str(plan["runtime"]["syncnet"]["git_head"]), "syncnet_checkpoint_sha256": str(plan["runtime"]["syncnet"]["checkpoint_sha256"]),
        "fixed_portrait_sha256": str(geometry.get("source_portrait", {}).get("sha256", "")).lower(),
    }
    if not isinstance(profile, dict):
        raise ScoreRunError("geometry runtime profile is absent")
    for key, expected in expected_profile.items():
        if canonical(profile.get(key)) != canonical(expected):
            raise ScoreRunError(f"geometry runtime profile differs from current runtime: {key}")
    adapter = runtime["adapter"]
    pins = runtime["source_pins"]
    if int(adapter.frame_batch_size) != batches["video_frame_batch_size"] or int(adapter.syncnet_batch_size) != batches["syncnet_batch_size"] or int(pins.get("full_context_audio_embedding_cache", {}).get("batch_size", -1)) != batches["audio_embedding_batch_size"]:
        raise ScoreRunError("loaded renderer/scorer batch settings differ from the qualified plan")

def check_geometry_model_source_pins(code_root: Path, geometry: dict[str, Any], runtime_pins: dict[str, Any]) -> None:
    producer = geometry.get("source_pins", {}).get("runtime_models_and_frontends")
    if not isinstance(producer, dict):
        raise ScoreRunError("native geometry lacks its actual model/frontend source pins")
    required = (
        "wav2lip_git_head", "wav2lip_sources", "wav2lip_checkpoint",
        "syncnet_git_head", "syncnet_sources", "syncnet_checkpoint",
        "official_frontend", "mel_spec", "python_speech_features",
        "torch_version", "numpy_version", "runtime_factory_path", "runtime_factory_sha256",
    )
    for name in required:
        if name not in producer or name not in runtime_pins:
            raise ScoreRunError(f"geometry/current runtime lacks a required model/frontend pin: {name}")
        if name not in ("runtime_factory_path", "runtime_factory_sha256") and canonical(producer[name]) != canonical(runtime_pins[name]):
            raise ScoreRunError(f"native geometry and current runtime model/frontend sources differ: {name}")
    factory_name = "wav2lip_fit_runtime.py"
    producer_path_text = producer.get("runtime_factory_path")
    runtime_path_text = runtime_pins.get("runtime_factory_path")
    if not isinstance(producer_path_text, str) or not isinstance(runtime_path_text, str):
        raise ScoreRunError("runtime factory provenance paths must be strings")
    producer_path, runtime_path = Path(producer_path_text), Path(runtime_path_text)
    staged_factory = code_root / "scripts/audio_enhancement" / factory_name
    if any(not path.is_absolute() or path.name != factory_name or path.is_symlink() for path in (producer_path, runtime_path, staged_factory)):
        raise ScoreRunError("runtime factory provenance paths are not absolute regular source files")
    if producer_path.resolve(strict=True) != producer_path or runtime_path.resolve(strict=True) != staged_factory.resolve(strict=True):
        raise ScoreRunError("current runtime factory is not the exact staged module")
    geometry_factory_sources = geometry.get("source_pins", {}).get("fit_runtime_sources")
    factory_record = geometry_factory_sources.get(factory_name) if isinstance(geometry_factory_sources, dict) else None
    if not isinstance(factory_record, dict) or factory_record.get("path") != producer_path_text:
        raise ScoreRunError("geometry does not bind its exact original runtime-factory source file")
    if factory_record.get("bytes") != RUNTIME_FACTORY_BYTES or producer.get("runtime_factory_sha256") != RUNTIME_FACTORY_SHA256 or runtime_pins.get("runtime_factory_sha256") != RUNTIME_FACTORY_SHA256 or factory_record.get("sha256") != RUNTIME_FACTORY_SHA256:
        raise ScoreRunError("runtime factory bytes/SHA differ from the frozen actual source")
    for path in (producer_path, staged_factory):
        if path.stat().st_size != RUNTIME_FACTORY_BYTES or sha_file(path) != RUNTIME_FACTORY_SHA256:
            raise ScoreRunError(f"runtime factory source bytes differ from the frozen pin: {path}")

def execute(args: argparse.Namespace) -> dict[str, Any]:
    binding_path = Path(args.input_binding)
    binding, binding_raw = verify_binding(binding_path)
    code_root, staged, stage_receipt = verify_source_stage(binding)
    support = binding.get("support_files")
    if not isinstance(support, dict):
        raise ScoreRunError("binding.support_files is missing")
    helper_pin, helper_path = verify_support(support.get("post_projection_controls"), "post_projection_controls", HELPER04_SHA256)
    adapter_pin, score_path = verify_support(support.get("score_adapter"), "score_adapter", SCORE_ADAPTER01_SHA256)
    actual_pins, raws, artifacts = {}, {}, {}
    for name in REQUIRED_ARTIFACTS:
        pin, raw, obj = read_json_pin(binding["artifacts"][name], f"artifact {name}")
        actual_pins[name], raws[name], artifacts[name] = pin, raw, obj
    _check_completion(binding, artifacts)
    input_pin = {"path": str(binding_path.resolve(strict=True)), "bytes": len(binding_raw), "sha256": sha(binding_raw)}
    output = create_output_dir(Path(args.output_dir))
    status_path = output / "status.jsonl"
    write_new(status_path, b"")
    def status(stage: str, **fields: Any) -> None:
        with status_path.open("ab") as handle:
            handle.write(canonical({"schema": RUNNER_SCHEMA, "stage": stage, "timestamp_unix_seconds": time.time(), **fields}) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    status("INPUTS_BOUND", input_binding=input_pin, fit_rows=FIT_ROWS, source_stage_status=stage_receipt["status"])
    plan, geometry, source_manifest = artifacts["parent_plan"], artifacts["native_geometry_manifest"], artifacts["fit_source_manifest"]
    completion = artifacts["completion"]
    checkpoint, training_record = artifacts["checkpoint"], artifacts["training_record"]
    runtime_source = artifacts["runtime_source_pins"]
    runtime_modules = runtime_source.get("training_modules")
    if not isinstance(runtime_modules, dict):
        raise ScoreRunError("actual runtime-source artifact lacks its module pins")
    if binding["source_lineage_pins"].get("runner") != runtime_source.get("runner"):
        raise ScoreRunError("binding source lineage does not match the D0 runner pin")
    if binding["source_lineage_pins"].get("head_implementation") != runtime_modules.get("v3_head"):
        raise ScoreRunError("binding source lineage does not match the D0 head module pin")
    if plan.get("plan_payload_sha256") != binding.get("parent_plan_payload_sha256") or sha(canonical({k:v for k,v in plan.items() if k != "plan_payload_sha256"})) != plan.get("plan_payload_sha256"):
        raise ScoreRunError("parent V2 plan payload digest differs from its actual input binding")
    if source_manifest.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1" or source_manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise ScoreRunError("Fit source manifest is not the frozen source-only roster")
    if any(source_manifest.get("scope", {}).get(k) is not False for k in ("candidate_transform_executed", "confirm_audio_read", "dev_audio_read", "model_forward", "render", "score")):
        raise ScoreRunError("Fit source manifest records treatment, non-Fit audio, rendering, or scoring")
    if geometry.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1" or geometry.get("status") != "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY":
        raise ScoreRunError("native geometry is not the completed Fit-only RMS artifact")
    if sha(canonical({k:v for k,v in geometry.items() if k != "payload_sha256"})) != geometry.get("payload_sha256") or geometry.get("fit_rows") != FIT_ROWS or geometry.get("method_id") != "rms_fixed":
        raise ScoreRunError("geometry canonical payload, row count, or RMS method differs")
    if geometry.get("source_clock_fps_num") != 25 or geometry.get("source_clock_fps_den") != 1 or geometry.get("audio_clock_policy") != "unchanged_original_natural_pcm_mfcc_queries":
        raise ScoreRunError("geometry changed the frozen source clock")
    if geometry.get("source_pins", {}).get("source_manifest", {}).get("sha256") != actual_pins["fit_source_manifest"]["sha256"] or geometry.get("source_pins", {}).get("v2_runtime_plan", {}).get("sha256") != actual_pins["parent_plan"]["sha256"]:
        raise ScoreRunError("geometry does not bind the actual source manifest and parent plan")
    geometry_rows = geometry.get("rows")
    if not isinstance(geometry_rows, list) or len(geometry_rows) != FIT_ROWS:
        raise ScoreRunError("geometry row denominator is not exactly 40")
    geometry_by_id = {str(r.get("row_id")): r for r in geometry_rows if isinstance(r, dict)}
    if len(geometry_by_id) != FIT_ROWS:
        raise ScoreRunError("geometry row IDs are not unique")
    source_root = Path(str(binding.get("fit_source_root", "")))
    if not source_root.is_absolute() or source_root.is_symlink() or not source_root.is_dir() or source_root.resolve(strict=True) != source_root:
        raise ScoreRunError("Fit WAV source root must be an actual canonical directory")
    status("ACTUAL_LINEAGE_PREFLIGHT_PASS", geometry_rows=FIT_ROWS, source_rows=len(source_manifest.get("rows", [])), parent_plan_payload_sha256=plan["plan_payload_sha256"])
    # Import only after the complete actual input and staged-code closure passed hashing.
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    sys.dont_write_bytecode = True
    qualified_v2 = runtime_source.get("qualified_v2_runtime_source_pins")
    psf_record = qualified_v2.get("execution_runtime") if isinstance(qualified_v2, dict) else None
    geometry_profile = geometry.get("runtime_gpu_identity", {}).get("runtime_profile") if isinstance(geometry.get("runtime_gpu_identity"), dict) else None
    if not isinstance(psf_record, dict) or not isinstance(geometry_profile, dict):
        raise ScoreRunError("actual Fit40 runtime/geometry profile is incomplete before Torch import")
    check_early_environment(expected_psf_root=psf_record.get("python_speech_features_root"), expected_cuda_visible_devices=geometry_profile.get("cuda_visible_devices"))
    import numpy as np
    import torch
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(FIT_SEED)
    np.random.seed(FIT_SEED)
    torch.manual_seed(FIT_SEED)
    torch.cuda.manual_seed_all(FIT_SEED)
    import scripts.audio_enhancement as audio_package
    expected_package_init = code_root / "scripts/audio_enhancement/__init__.py"
    if Path(audio_package.__file__).resolve(strict=True) != expected_package_init.resolve(strict=True):
        raise ScoreRunError("audio_enhancement package initializer did not resolve to staged source")
    from scripts.audio_enhancement import baselines
    from scripts.audio_enhancement import shared_recipient_transition_head_v3_d0_gatedc_attempt01 as v3_head
    from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v2_attempt02 as v2
    from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v4_attempt01 as v4
    from scripts.audio_enhancement import v3_v4_native_runtime_adapter_v9_attempt01 as native_adapter
    core_path = code_root / RUNNER_RELATIVE_PATH
    core = load_module_from_file("fit40_d0_core_attempt02", core_path)
    if Path(core.__file__).resolve(strict=True) != core_path.resolve(strict=True):
        raise ScoreRunError("D0 runner import did not resolve to the staged source member")
    if v3_head.SCHEMA != "cross_tfg_shared_recipient_transition_head_v3_d0_gate_dc_attempt01" or v2.RUNTIME_SCHEMA != "cross_tfg_wav2lip_fullcontext_fit_runtime_v2_attempt02" or v4.RUNTIME_SCHEMA != "cross_tfg_wav2lip_fullcontext_fit_runtime_v4_attempt01":
        raise ScoreRunError("staged D0 head/V2/V4 runtime schemas differ from the actual Fit40 run")
    if Path(v3_head.__file__).resolve(strict=True) != code_root / HEAD_RELATIVE_PATH:
        raise ScoreRunError("D0 head import is not the exact pinned staged module")
    imported_source_modules = {
        "baselines": (baselines, "scripts/audio_enhancement/baselines.py"),
        "v2_runtime": (v2, "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v2_attempt02.py"),
        "v4_runtime": (v4, "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py"),
        "v3_v4_adapter": (native_adapter, "scripts/audio_enhancement/v3_v4_native_runtime_adapter_v9_attempt01.py"),
    }
    for module_name, (module, relative) in imported_source_modules.items():
        if Path(module.__file__).resolve(strict=True) != (code_root / relative).resolve(strict=True):
            raise ScoreRunError(f"imported {module_name} did not resolve to its exact staged source file")
    helper = load_module_from_file("fit40_postprojection_controls_attempt04", helper_path)
    scorer = load_module_from_file("fit40_postprojection_score_adapter_attempt01", score_path)
    if helper.SCHEMA != "cross_tfg_fit40_post_projection_control_diagnostic_v2" or scorer.SCHEMA != "cross_tfg_fit40_post_projection_fullquery_score_diagnostic_v1":
        raise ScoreRunError("support helper/scorer schemas differ from their frozen pins")
    source_files = check_source_module_pins(code_root, runtime_source)
    source_files["runner"] = staged["runner"]
    source_files["head_implementation"] = staged["head_implementation"]
    source_files["native_adapter"] = local_pin(Path(native_adapter.__file__).resolve(strict=True))
    source_files["v2_runtime"] = local_pin(Path(v2.__file__).resolve(strict=True))
    source_files["v4_runtime"] = local_pin(Path(v4.__file__).resolve(strict=True))
    source_files["wav2lip_frontend"] = local_pin(code_root / "scripts/audio_enhancement/wav2lip_frontend.py")
    if runtime_source.get("runner") != staged["runner"] or runtime_source.get("head_config") != actual_pins["head_config"] or runtime_source.get("training_modules", {}).get("v3_head") != staged["head_implementation"]:
        raise ScoreRunError("training source-pins do not bind the exact staged runner/head/config")

    if args.device != completion.get("gpu_identity", {}).get("runtime_device"):
        raise ScoreRunError("requested CUDA device differs from the actual completed Fit40 runtime identity")
    identity = check_runtime_environment(core, completion, args.device, torch, np)

    config = artifacts["head_config"]
    if config.get("schema") != "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_config_v1":
        raise ScoreRunError("head-config schema differs from the actual D0 run")
    constructor = config.get("head_constructor")
    if not isinstance(constructor, dict) or constructor.get("target_rms") != TARGET_RMS or constructor.get("peak_ceiling") is not None or constructor.get("sample_rate_hz") != SAMPLE_RATE_HZ or constructor.get("video_fps_num") != 25 or constructor.get("video_fps_den") != 1:
        raise ScoreRunError("head constructor changed the frozen RMS or sample/video clock")
    for key, expected in (("v2_runtime", core.EXPECTED_V2_RUNTIME_SHA256), ("v4_runtime", core.EXPECTED_V4_RUNTIME_SHA256), ("v3_head", core.EXPECTED_V3_HEAD_SHA256), ("v3_v4_adapter", core.EXPECTED_V3_V4_ADAPTER_SHA256)):
        if source_files.get({"v2_runtime":"v2_runtime", "v4_runtime":"v4_runtime", "v3_head":"head_implementation", "v3_v4_adapter":"native_adapter"}[key], {}).get("sha256") != expected:
            raise ScoreRunError(f"staged module differs from D0 runner's fixed source pin: {key}")

    source_manifest = artifacts["fit_source_manifest"]
    geometry = artifacts["native_geometry_manifest"]
    geometry_rows = geometry["rows"]
    geometry_by_id = {str(item["row_id"]): item for item in geometry_rows}
    prepared = core._build_fit_rows(source_manifest, Path(str(binding["fit_source_root"])), geometry_by_id)
    row_ids = [str(item["row"]["row_id"]) for item in prepared]
    if row_ids != [str(item["row_id"]) for item in geometry_rows]:
        raise ScoreRunError("prepared source rows do not preserve the 40-row geometry order")
    if len(prepared) != FIT_ROWS or len({str(item["row"]["speaker_id"]) for item in prepared}) != 20:
        raise ScoreRunError("prepared source rows do not preserve the exact 40-row/20-speaker Fit roster")
    cache_dir = output / "input_cache_clone"
    cache_clone = core._clone_audio_cache(geometry, cache_dir)
    runtime_plan = json.loads(json.dumps(plan))
    runtime_plan["runtime"]["full_context"]["audio_embedding_cache_root"] = str(cache_dir)
    total_samples = sum(int(item["row"]["natural_pcm"]["sample_count"]) for item in prepared)
    runtime_plan["runtime"]["full_context"]["source_sample_count_total"] = total_samples
    runtime_plan["fit_config"]["selected_rows"] = row_ids
    runtime_plan["fit_config"]["source_sample_count_total"] = total_samples
    runtime_cfg = json.loads(json.dumps(runtime_plan["runtime"]))
    runtime_cfg["fit_v4"] = {"schema": "cross_tfg_wav2lip_fullcontext_fit_runtime_v4_attempt01", "native_geometry_manifest_path": actual_pins["native_geometry_manifest"]["path"], "native_geometry_manifest_sha256": actual_pins["native_geometry_manifest"]["sha256"]}
    runtime = v2.create_runtime(plan=runtime_plan, fit_rows=[item["row"] for item in prepared], device=args.device, seed=FIT_SEED)
    if not isinstance(runtime, dict) or not isinstance(runtime.get("source_pins"), dict):
        raise ScoreRunError("V2 runtime did not return its actual source pins")
    adapter = runtime["adapter"]
    for model in (adapter.wav2lip_model, adapter.syncnet_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    if not torch.cuda.is_initialized():
        raise ScoreRunError("model loading did not initialize the exact CUDA runtime")
    check_geometry_model_source_pins(code_root, geometry, runtime["source_pins"])
    check_geometry_profile(core, runtime_plan, geometry, runtime, args.device, identity, torch, np)
    runtime_pins = runtime["source_pins"]
    if runtime_pins.get("v2_runtime", {}).get("sha256") != core.EXPECTED_V2_RUNTIME_SHA256:
        raise ScoreRunError("loaded V2 runtime source pin changed")
    geometry_runtime_cfg = runtime_plan["runtime"].setdefault("fit_v4", runtime_cfg["fit_v4"])
    native_manifest, native_geometries, native_records = v4._load_native_geometry_manifest(runtime_plan, [item["row"] for item in prepared], geometry_runtime_cfg)
    native_record_ids = [str(item["row_id"]) for item in native_records]
    cache_record_ids = [str(item.get("row_id")) for item in native_manifest.get("rows", [])]
    if set(native_geometries) != set(row_ids) or native_record_ids != row_ids or cache_record_ids != row_ids:
        raise ScoreRunError("native geometry/cache loader did not preserve the exact ordered 40-row Fit roster")
    del runtime["optimizer"]
    del runtime["enhancement_stack"]
    if "optimizer" in runtime or "enhancement_stack" in runtime:
        raise ScoreRunError("unused legacy V2 optimizer/enhancement stack were not discarded")
    context_for_row = runtime["context_for_row"]
    geometry_record_by_row = {str(item["row_id"]): item for item in native_records}
    cache_by_id = {str(item["row_id"]): item for item in native_manifest["rows"]}
    if len(cache_by_id) != FIT_ROWS:
        raise ScoreRunError("loaded native cache manifest is not the 40-row Fit roster")
    head = v3_head.SharedRecipientTransitionHeadV3(**constructor).to(device=identity["runtime_device"])
    head.eval()
    checkpoint_parameter = checkpoint.get("parameter")
    if not isinstance(checkpoint_parameter, dict) or checkpoint_parameter.get("shape") != [3, 3] or checkpoint_parameter.get("dtype") != "float32":
        raise ScoreRunError("actual checkpoint parameter is not the frozen [3,3] float32 tensor")
    values = np.asarray(checkpoint_parameter.get("values"), dtype=np.float32)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ScoreRunError("checkpoint parameter values are malformed/nonfinite")
    with torch.no_grad():
        head.band_contrasts_db.copy_(torch.from_numpy(values).to(device=head.band_contrasts_db.device, dtype=torch.float32))
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    helper._assert_pinned_head(head, constructor)
    theta_payload = {"name": "band_contrasts_db", "shape": [3, 3], "dtype": "float32", "temporal_components": ["static", "positive_flux", "negative_flux"], "non_reference_band_order": [0, 1, 2], "values": values.astype("<f4", copy=False).tolist()}
    theta_sha = sha(canonical(theta_payload))
    expected_theta_sha = binding.get("expected_checkpoint_parameter_sha256")
    if not isinstance(expected_theta_sha, str) or len(expected_theta_sha) != 64 or theta_sha != expected_theta_sha:
        raise ScoreRunError("loaded D0 parameter canonical SHA differs from the actual binding")

    gate_raw = raws["fit40_gate_dc_control"]
    gate_rows = parse_gate_ledger(gate_raw)
    gate_map = {}
    for record in gate_rows:
        key = (str(record.get("row_id")), str(record.get("gate_mode")))
        if key in gate_map:
            raise ScoreRunError(f"duplicate GateDC row identity: {key}")
        gate_map[key] = record
    expected_gate_keys = {(row_id, mode) for row_id in row_ids for mode in ("dynamic", "gate_dc")}
    if set(gate_map) != expected_gate_keys:
        raise ScoreRunError("GateDC ledger identities do not equal the 40 rows x 2 branches")

    controls_inputs, sources, baselines_by_row, contexts, geometries_for_score = [], {}, {}, {}, {}
    for item in prepared:
        row = item["row"]
        row_id = str(row["row_id"])
        context = context_for_row(row)
        cache_record = cache_by_id[row_id]["natural_audio_embedding_cache"]
        if context.cache_record.get("npz_sha256") != cache_record.get("npz_sha256") or context.cache_record.get("metadata_sha256") != cache_record.get("metadata_sha256"):
            raise ScoreRunError(f"loaded original-audio cache differs from pinned geometry for {row_id}")
        geom_row = geometry_by_id[row_id]
        source_tensor = torch.from_numpy(np.ascontiguousarray(item["audio"])).to(device=identity["runtime_device"], dtype=torch.float32)
        baseline_np, _ = core._verify_baseline_wav(row, geom_row, item["audio"], baselines)
        baseline_tensor = torch.from_numpy(np.ascontiguousarray(baseline_np)).to(device=identity["runtime_device"], dtype=torch.float32)
        source_path = Path(row["natural_pcm"]["path"])
        baseline_pin_record = geom_row["rms_fixed_baseline"]["rms_input_float32_wav"]
        baseline_path = Path(str(baseline_pin_record["path"]))
        source_file_pin, baseline_file_pin = core._pin(source_path), core._pin(baseline_path)
        source_raw, baseline_raw = source_path.read_bytes(), baseline_path.read_bytes()
        sources[row_id], baselines_by_row[row_id], contexts[row_id] = source_tensor, baseline_tensor, context
        geometries_for_score[row_id] = core._row_geometry_record(native_geometries, row_id)
        controls_inputs.append({
            "row_id": row_id, "sample_id": row["sample_id"], "speaker_id": row["speaker_id"],
            "source_pcm_float32": source_tensor, "canonical_rms_baseline_pcm_float32": baseline_tensor,
            "source_sample_count": int(row["natural_pcm"]["sample_count"]),
            "source_pcm_float32_sha256": row["natural_pcm"]["pcm_sha256"],
            "canonical_rms_baseline_pcm_float32_sha256": baseline_pin_record["output_float32_pcm_sha256"],
            "source_file_pin": source_file_pin, "baseline_file_pin": baseline_file_pin,
            "source_file_raw_bytes": source_raw, "baseline_file_raw_bytes": baseline_raw,
        })
    status("RUNTIME_AND_40_ROWS_READY", runtime_identity=identity, context_count=40, optimizer_updates=0)

    # This input is built only from actual producer files and source pins.
    helper_lineage = dict(binding["source_lineage_pins"])
    helper_lineage["runner"] = staged["runner"]
    helper_lineage["head_implementation"] = staged["head_implementation"]
    if runtime_source.get("training_modules", {}).get("v3_head") != helper_lineage["head_implementation"]:
        raise ScoreRunError("absolute runtime head pin differs from actual staged D0 head")
    status("BUILDING_240_CONTROLS", planned_controls=240, head_update=False)
    controls = helper.run_fit40_post_projection_controls(
        head, controls_inputs,
        actual_completion_pin=actual_pins["completion"], actual_completion_raw=raws["completion"],
        actual_checkpoint=checkpoint, actual_checkpoint_raw=raws["checkpoint"],
        checkpoint_parameter=checkpoint_parameter, expected_checkpoint_parameter_sha256=expected_theta_sha,
        actual_training_record_raw=raws["training_record"], expected_row_ids=row_ids,
        actual_runtime_source_pins_raw=raws["runtime_source_pins"],
        actual_head_config=artifacts["head_config"], actual_head_config_raw=raws["head_config"],
        actual_head_constructor=constructor, actual_runtime_identity=identity,
        expected_original_gate_rows=gate_map, actual_gate_ledger_raw=gate_raw,
        actual_completion=artifacts["completion"], actual_training_record=training_record,
        actual_runtime_source_pins=runtime_source, actual_native_geometry_manifest=geometry,
        actual_native_geometry_manifest_raw=raws["native_geometry_manifest"], source_lineage_pins=helper_lineage,
    )
    expected_control_ids = [(row_id, mode, name) for row_id in row_ids for mode in ("dynamic", "gate_dc") for name in ("post_proj_mean", "post_proj_roll_plus64", "post_proj_roll_minus64")]
    actual_control_ids = [(str(record.get("row_id")), str(record.get("base_gate_mode")), str(record.get("control"))) for record in controls.get("records", [])]
    if len(actual_control_ids) != 240 or actual_control_ids != expected_control_ids:
        raise ScoreRunError("post-projection helper did not preserve the exact ordered 40 x 2 x 3 control denominator")
    control_records_pin = write_new(output / "post_projection_control_records.jsonl", jsonl_bytes(controls["records"]))
    control_wave_arrays, control_wave_index = {}, []
    for index, record in enumerate(controls["records"]):
        control_id = f"{record['row_id']}::{record['base_gate_mode']}::{record['control']}"
        key = f"waveform_{index:03d}"
        array = controls["outputs_float32_tensors"][control_id].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False)
        control_wave_arrays[key] = array
        control_wave_index.append({"key": key, "control_id": control_id, "sample_count": int(array.size), "pcm_sha256": sha(array.tobytes(order="C"))})
    control_wave_pin = write_new(output / "post_projection_control_waveforms.npz", npz_bytes(np, control_wave_arrays))
    control_bundle = {
        "schema": "cross_tfg_fit40_post_projection_control_bundle_v1",
        "status": "BUILT_240_CONTROLS_NO_UPDATE_PENDING_SCORE",
        "helper_receipt": controls["receipt"], "control_records": control_records_pin,
        "waveform_archive": control_wave_pin, "waveform_archive_index": control_wave_index,
        "waveform_archive_key_count": len(control_wave_index), "optimizer_updates": 0,
        "head_parameter_sha256": theta_sha, "full_fit40_rows_preserved": True,
        "scoring_or_rendering_performed_by_helper": False,
    }
    control_bundle_raw = json_bytes(control_bundle, with_self_hash=True)
    control_bundle_pin = write_new(output / "post_projection_control_bundle.json", control_bundle_raw)
    status("CONTROLS_PERSISTED", control_count=240, bundle=control_bundle_pin)

    score_lineage = {key: actual_pins[key] for key in ("completion", "checkpoint", "training_record", "runtime_source_pins", "strict_backward_qualification", "fit40_gate_dc_control", "native_geometry_manifest")}
    score_lineage["post_projection_controls"] = control_bundle_pin
    score_source_pins = {
        "d0_runner": staged["runner"], "d0_head": staged["head_implementation"],
        "native_adapter": source_files["v3_v4_adapter"], "v4_runtime": source_files["v4_runtime"],
        "v2_runtime": source_files["v2_runtime"], "wav2lip_frontend": source_files["wav2lip_frontend"],
        "post_projection_controls": helper_pin, "score_adapter": adapter_pin,
    }
    status("SCORING_FULL_320_ROWS", references=80, controls=240, queries_per_row=27, lags_per_query=31)
    score_result = scorer.score_fit40_post_projection_controls(
        rows=[item["row"] for item in prepared], expected_row_ids=row_ids,
        sources=sources, baseline_waveforms=baselines_by_row, head=head,
        expected_parameter_sha256=expected_theta_sha, gate_ledger_rows=gate_rows,
        post_projection_result=controls, contexts=contexts, native_geometries=geometries_for_score,
        adapter=adapter, v4_runtime=v4, v3_v4_adapter=native_adapter, torch=torch,
        functional=importlib.import_module("torch.nn.functional"),
        lineage_pins=score_lineage, source_pins=score_source_pins, runtime_identity=identity,
    )
    if len(score_result.get("records", [])) != 320:
        raise ScoreRunError("native scorer did not return the full 320-row denominator")
    expected_score_ids = [f"{row_id}::{mode}::original" for row_id in row_ids for mode in ("dynamic", "gate_dc")]
    expected_score_ids += [f"{row_id}::{mode}::{name}" for row_id in row_ids for mode in ("dynamic", "gate_dc") for name in ("post_proj_mean", "post_proj_roll_plus64", "post_proj_roll_minus64")]
    actual_score_ids = [str(record.get("record_id")) for record in score_result["records"]]
    if actual_score_ids != expected_score_ids:
        raise ScoreRunError("native scorer did not preserve the exact ordered 80-reference/240-control denominator")
    arrays = score_result.get("arrays")
    expected_score_shapes = {
        "distance_matrices_float32": (320, 31, 27),
        "curves_float32": (320, 31),
        "zero_lag_unit_cosines_float32": (320,),
        "visual_embedding_norms_float32": (320,),
        "control_minus_same_theta_reference_distance_matrices_float32": (240, 31, 27),
        "seam_query_mask_bool": (240, 27),
    }
    if not isinstance(arrays, dict) or set(arrays) != set(expected_score_shapes):
        raise ScoreRunError("score-array member set differs from the frozen full-query score schema")
    for name, shape in expected_score_shapes.items():
        value = arrays[name]
        if tuple(value.shape) != shape:
            raise ScoreRunError(f"score-array shape differs from its frozen denominator: {name}")
        if name == "seam_query_mask_bool":
            if value.dtype != torch.bool:
                raise ScoreRunError("secondary seam-query array must remain boolean")
        elif value.dtype != torch.float32 or not torch.isfinite(value).all().item():
            raise ScoreRunError(f"score array is nonfinite or not float32: {name}")
    score_records_pin = write_new(output / "score_records.jsonl", jsonl_bytes(score_result["records"]))
    score_arrays = {name: value.detach().to(device="cpu").contiguous().numpy() for name, value in arrays.items()}
    score_arrays_pin = write_new(output / "score_arrays.npz", npz_bytes(np, score_arrays))
    status("SCORE_ARRAYS_PERSISTED", records=320, score_arrays=score_arrays_pin)
    replay = score_result["receipt"]["same_theta_reference_replay"]
    run_status = "COMPLETE_DIAGNOSTIC_ONLY_NOT_QUALIFICATION" if replay.get("failed_row_count") == 0 else "COMPLETE_REFERENCE_REPLAY_MISMATCH_PRESERVED_NOT_QUALIFIED"
    score_receipt = {
        "schema": "cross_tfg_fit40_post_projection_score_run_receipt_v1", "status": run_status,
        "score_adapter_receipt": score_result["receipt"], "score_records": score_records_pin,
        "score_arrays": score_arrays_pin, "post_projection_control_bundle": control_bundle_pin,
        "input_binding": input_pin, "runtime_identity": identity, "optimizer_updates": 0,
        "v2_runtime_legacy_optimizer_created_then_discarded_without_step": True,
        "head_parameter_sha256_after": expected_theta_sha, "candidate_selected": False,
        "development_or_confirmation_opened": False, "api_called": False, "full_video_file_encoded": False,
    }
    score_receipt_pin = write_new(output / "score_run_receipt.json", json_bytes(score_receipt, with_self_hash=True))
    final = {
        "schema": RUNNER_SCHEMA, "status": run_status, "fit_rows": 40,
        "same_theta_reference_rows": 80, "post_projection_control_rows": 240,
        "scored_records": 320, "distance_matrix_shape": [320, 31, 27], "curve_shape": [320, 31],
        "input_binding": input_pin, "source_stage_receipt": binding["source_stage_receipt"],
        "runtime_identity": identity, "cache_clone": cache_clone,
        "control_bundle": control_bundle_pin, "score_records": score_records_pin,
        "score_arrays": score_arrays_pin, "score_receipt": score_receipt_pin,
        "scope": {"fit_only": True, "training": False, "optimizer_updates": 0, "v2_legacy_optimizer_created_then_discarded_without_step": True, "development": False, "confirmation": False, "selection": False, "api": False, "full_video_encoding": False},
    }
    final_pin = write_new(output / "run_manifest.json", json_bytes(final, with_self_hash=True))
    status("COMPLETE", run_manifest=final_pin, run_status=run_status)
    print(json.dumps({"schema": RUNNER_SCHEMA, "status": run_status, "run_manifest": final_pin}, sort_keys=True, ensure_ascii=False))
    return final
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit-only same-theta 320-row post-projection score diagnostic; exact qualified GPU required.")
    parser.add_argument("--input-binding", required=True, help="actual self-hashed Fit40 completion, source, and artifact binding")
    parser.add_argument("--output-dir", required=True, help="fresh create-only directory beneath /dev/shm/cross_tfg_audio_head_20261003")
    parser.add_argument("--device", required=True, help="exact CUDA device recorded by the actual Fit40 completion")
    return parser.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = execute(args)
        return 0 if result["status"] == "COMPLETE_DIAGNOSTIC_ONLY_NOT_QUALIFICATION" else 3
    except Exception as exc:
        failure = {
            "schema": RUNNER_SCHEMA, "status": "FAILED_PARTIAL_OUTPUTS_PRESERVED_NOT_QUALIFIED",
            "exception_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=20), "timestamp_unix_seconds": time.time(),
            "training_or_optimizer_update": False, "development_or_confirmation_opened": False,
            "candidate_selected": False,
        }
        try:
            output_value = getattr(args, "output_dir", "")
            output = Path(output_value)
            if output.is_dir() and not output.is_symlink() and not (output / "failure.json").exists():
                write_new(output / "failure.json", json_bytes(failure, with_self_hash=True))
        except Exception:
            pass
        print(json.dumps(failure, sort_keys=True, ensure_ascii=False))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
