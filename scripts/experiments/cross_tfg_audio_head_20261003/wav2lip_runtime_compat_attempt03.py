"""Wav2Lip-only strict-policy repair and pre-AVI pixel audit.

This is placed before the shared runtime bootstrap on the Wav2Lip child's
PYTHONPATH. It executes that exact pinned bootstrap first, then narrowly
accounts for FaceAlignment.__init__ changing cuDNN benchmarking and observes
the unchanged ndarray frames passed to OpenCV's VideoWriter.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/root/tts-research-system-20261001")
BOOTSTRAP = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt03/sitecustomize.py"
BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
W2L_ENTRY = Path("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/inference.py")
W2L_MODELS_INIT = W2L_ENTRY.parent / "models/__init__.py"
W2L_MODELS_INIT_SHA256 = "d6883f7d5c41623280c9cbce4963ad719347b9d4cdac614cadfff7b544535215"
W2L_MODEL_SOURCE = W2L_ENTRY.parent / "models/wav2lip.py"
W2L_MODEL_SOURCE_SHA256 = "a02b54288f8200332ee4f0ea0c8013bca542b94d8149e514ac25a742a7a70e66"
FACE_API = Path("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/api.py")
FACE_API_SHA256 = "c050676db49d8eb5de19f13c171101892b60b992b2a730cfbc8534967e4faeaa"
FACE_DETECTION_ROOT = W2L_ENTRY.parent / "face_detection"
FACE_DETECTION_SOURCE_PINS = {
    "__init__.py": (183, "7215846a10fcb2162dab619cf1cb45bd6c0faac2b7d1a71ed8cc1cc0811db86a"),
    "api.py": (2266, "c050676db49d8eb5de19f13c171101892b60b992b2a730cfbc8534967e4faeaa"),
    "detection/__init__.py": (30, "1f3233930bfb7ac8b1726e4af8ed9b93925046d16edc640afd218f9d02b6d235"),
    "detection/core.py": (4868, "ba16a80fac378da9b5bbfc9c190975ab4fe153b13b0d18d892ab5bbfcbb9c71e"),
    "detection/sfd/__init__.py": (53, "6fbfb62eccc1a14de7c0a8cb1ca6f3c59af42479955ac08acbf548f28ec1feed"),
    "detection/sfd/bbox.py": (4279, "3ed1487243bbde8ab47c567f46ee4b177a549e31440db8d71376bffc70a7bd86"),
    "detection/sfd/detect.py": (3769, "1848d75a551131c786a5e10d4ffab517dda606af27913e19ab48467c1a779d71"),
    "detection/sfd/net_s3fd.py": (5291, "80960aaefd9fb97a5e9784dc0a48bc38f43bdcaacb268782ede16a959e77a5a9"),
    "detection/sfd/sfd_detector.py": (1809, "155be501e83825f9b0508ee7d596f13047559350adbdbf17fcbf361cfab22e92"),
    "models.py": (8619, "2cde93a32467b9c5cfd009499c917afe8c89f512622237c2d865ba8fbb8a37e9"),
    "utils.py": (11808, "561bfe5ec349a28f086e04efbb2d43cbc45c15fbcac48c311d497cb2bd2efe9a"),
}
SFD_CHECKPOINT = FACE_DETECTION_ROOT / "detection/sfd/s3fd.pth"
SFD_CHECKPOINT_BYTES = 89_843_225
SFD_CHECKPOINT_SHA256 = "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"
W2L_MODEL_ID = "wav2lip_gan_bac9a81e"
POLICY_SHA256 = "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
EXPECTED_FPS = 25.0
EXPECTED_SIZE_WH = (512, 512)


def _sha_bytes(raw: bytes | memoryview) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _bound_row_seed() -> int:
    seed_text = os.environ.get("CROSS_TFG_RENDER_SEED")
    if (not seed_text or not seed_text.isascii() or not seed_text.isdecimal()
            or os.environ.get("TFG_DETERMINISM_SEED") != seed_text):
        raise RuntimeError("Wav2Lip row seed must be bound by matching render and determinism environment values")
    return int(seed_text)


def _bound_expected_frame_count() -> int:
    frame_text = os.environ.get("CROSS_TFG_EXPECTED_FRAME_COUNT")
    if not frame_text or not frame_text.isascii() or not frame_text.isdecimal():
        raise RuntimeError("Wav2Lip expected frame count must be bound from the sealed row")
    frame_count = int(frame_text)
    if frame_count <= 0:
        raise RuntimeError("Wav2Lip expected frame count must be positive")
    return frame_count


def _write_receipt(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["receipt_sha256"] = _sha_bytes(_canonical(record))
    raw = (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return record


def _require_regular(path: Path, expected_sha256: str, description: str) -> dict[str, Any]:
    st = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(st.st_mode):
        raise RuntimeError(f"{description} is not a regular nonsymlink file: {path}")
    digest = _sha_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"{description} SHA-256 mismatch: {path}")
    return {"path": str(path.resolve(strict=True)), "bytes": st.st_size, "sha256": digest}


def _verified_face_detection_sources() -> list[dict[str, Any]]:
    if FACE_DETECTION_ROOT.is_symlink() or not FACE_DETECTION_ROOT.is_dir():
        raise RuntimeError("deployed face_detection source root is missing or symlinked")
    discovered: set[str] = set()
    for current, dirs, files in os.walk(FACE_DETECTION_ROOT, topdown=True, followlinks=False):
        directory = Path(current)
        child_dirs = sorted(dirs)
        if any((directory / name).is_symlink() for name in child_dirs):
            raise RuntimeError("deployed face_detection package contains a symlinked source directory")
        dirs[:] = child_dirs
        for name in files:
            if name.endswith(".py"):
                path = directory / name
                if path.is_symlink():
                    raise RuntimeError(f"deployed face_detection source is symlinked: {path}")
                discovered.add(path.relative_to(FACE_DETECTION_ROOT).as_posix())
    if discovered != set(FACE_DETECTION_SOURCE_PINS):
        raise RuntimeError("deployed face_detection Python source set differs from the sealed 11-file inventory")
    verified = []
    for relative, (expected_bytes, expected_sha256) in sorted(FACE_DETECTION_SOURCE_PINS.items()):
        path = FACE_DETECTION_ROOT / relative
        pin = _require_regular(path, expected_sha256, f"deployed face_detection source {relative}")
        if pin["bytes"] != expected_bytes:
            raise RuntimeError(f"deployed face_detection source byte count changed: {relative}")
        verified.append({"relative_path": relative, **pin})
    return verified


def _module_origin(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    return str(Path(spec.origin).resolve(strict=True))


def _expected_module_origins() -> dict[str, str]:
    return {
        "face_detection": str((FACE_DETECTION_ROOT / "__init__.py").resolve(strict=True)),
        "face_detection.api": str(FACE_API.resolve(strict=True)),
        "face_detection.detection.sfd": str((FACE_DETECTION_ROOT / "detection/sfd/__init__.py").resolve(strict=True)),
        "face_detection.detection.sfd.sfd_detector": str((FACE_DETECTION_ROOT / "detection/sfd/sfd_detector.py").resolve(strict=True)),
        "face_detection.detection.sfd.net_s3fd": str((FACE_DETECTION_ROOT / "detection/sfd/net_s3fd.py").resolve(strict=True)),
        "models": str(W2L_MODELS_INIT.resolve(strict=True)),
        "models.wav2lip": str(W2L_MODEL_SOURCE.resolve(strict=True)),
    }


def _preflight_record() -> dict[str, Any]:
    if (sys.argv[0] != "-c"
            or os.environ.get("CROSS_TFG_ROW_ID")
            or os.environ.get("CROSS_TFG_STRICT_STARTUP_POLICY")
            or os.environ.get("PYTHONNOUSERSITE") != "1"
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or not sys.dont_write_bytecode):
        raise RuntimeError("runtime compatibility preflight must be a clean metadata-only -c interpreter")
    sources = _verified_face_detection_sources()
    model_source = _require_regular(W2L_MODEL_SOURCE, W2L_MODEL_SOURCE_SHA256,
                                    "deployed Wav2Lip model source")
    models_init = _require_regular(W2L_MODELS_INIT, W2L_MODELS_INIT_SHA256,
                                   "deployed Wav2Lip models package initializer")
    sfd_checkpoint = _require_regular(SFD_CHECKPOINT, SFD_CHECKPOINT_SHA256,
                                      "package-local SFD checkpoint")
    if sfd_checkpoint["bytes"] != SFD_CHECKPOINT_BYTES:
        raise RuntimeError("package-local SFD checkpoint byte count differs from the frozen identity")
    site = sys.modules.get("sitecustomize")
    site_origin = None if site is None else getattr(site, "__file__", None)
    expected_site = str(Path(__file__).resolve(strict=True))
    if site_origin is None or str(Path(site_origin).resolve(strict=True)) != expected_site:
        raise RuntimeError("Wav2Lip compatibility sitecustomize was not imported from its pinned runtime directory")
    expected = _expected_module_origins()
    observed = {name: _module_origin(name) for name in expected}
    if observed != expected:
        raise RuntimeError("Wav2Lip preflight module-spec origins differ from the pinned source closure")
    torch = sys.modules.get("torch")
    if torch is None:
        torch = importlib.import_module("torch")
    cuda = getattr(torch, "cuda", None)
    cuda_is_initialized = bool(cuda is not None and callable(getattr(cuda, "is_initialized", None))
                               and cuda.is_initialized())
    if cuda_is_initialized:
        raise RuntimeError("Wav2Lip metadata-only preflight unexpectedly initialized CUDA")
    versions = {}
    for distribution in ("torch", "numpy", "opencv-python", "opencv-python-headless", "librosa"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "schema": "cross_tfg_w2l_runtime_compat_preflight_v1",
        "status": "PASS_STATIC_CHILD_IMPORT_ROUTE",
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "sitecustomize_origin": expected_site,
        "module_spec_origins": observed,
        "runtime_distribution_versions": versions,
        "dotted_module_spec_resolution_may_import_parent_packages": True,
        "torch_imported_for_cuda_initialization_check": True,
        "face_detection_sources": sources,
        "wav2lip_models_package_initializer": models_init,
        "wav2lip_model_source": model_source,
        "sfd_checkpoint": sfd_checkpoint,
        "model_or_scorer_forward_invoked_directly_by_preflight": False,
        "sfd_weight_file_hashed_readonly": True,
        "model_weight_tensors_loaded_by_preflight": False,
        "media_or_audio_read": False,
        "torch_cuda_initialized_at_preflight": cuda_is_initialized,
        "filesystem_mutation": False,
    }


def _observe_policy(torch: Any) -> dict[str, Any]:
    return {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_allow_tf32": bool(torch.backends.cudnn.allow_tf32
                                  or torch.backends.cuda.matmul.allow_tf32),
    }


def _load_pinned_bootstrap() -> Any:
    bootstrap_pin = _require_regular(BOOTSTRAP, BOOTSTRAP_SHA256, "shared strict bootstrap")
    module_name = "_cross_tfg_pinned_runtime_determinism_attempt03"
    spec = importlib.util.spec_from_file_location(module_name, BOOTSTRAP)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned shared strict bootstrap")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not isinstance(getattr(module, "POLICY", None), dict):
        raise RuntimeError("pinned shared bootstrap has no strict policy interface")
    if _sha_bytes(_canonical(module.POLICY)) != POLICY_SHA256:
        raise RuntimeError("shared bootstrap policy does not match the frozen strict policy")
    if _observe_policy(module.torch if hasattr(module, "torch") else importlib.import_module("torch")) != module.POLICY:
        raise RuntimeError("shared bootstrap strict state changed before Wav2Lip compatibility setup")
    row_id = os.environ.get("CROSS_TFG_ROW_ID")
    row_seed = _bound_row_seed()
    receipt_dir = Path(os.environ["CROSS_TFG_DETERMINISM_RECEIPT_DIR"])
    startup_path = receipt_dir / f"{os.getpid()}.json"
    raw = startup_path.read_bytes()
    startup = json.loads(raw.decode("utf-8"))
    if (startup.get("schema") != "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1"
            or startup.get("status") != "PASS"
            or startup.get("receipt_sha256") != _sha_bytes(_canonical({
                key: value for key, value in startup.items() if key != "receipt_sha256"}))
            or startup.get("bootstrap_source_sha256") != BOOTSTRAP_SHA256
            or startup.get("policy") != module.POLICY
            or startup.get("row_id") != row_id
            or startup.get("model_id") != W2L_MODEL_ID
            or startup.get("seed") != row_seed
            or Path(startup.get("entry_path", "")).resolve(strict=True)
            != W2L_ENTRY.resolve(strict=True)):
        raise RuntimeError("pinned shared bootstrap did not write the expected native Wav2Lip receipt")
    module._w2l_runtime_bootstrap_pin = bootstrap_pin
    module._w2l_runtime_startup_receipt = {"path": str(startup_path), "bytes": len(raw),
                                           "sha256": _sha_bytes(raw)}
    return module


def _install_wav2lip_runtime_compat(bootstrap: Any) -> None:
    if (os.environ.get("CROSS_TFG_STRICT_STARTUP_POLICY") != "1"
            or os.environ.get("CROSS_TFG_MODEL_ID") != W2L_MODEL_ID
            or os.environ.get("PYTHONNOUSERSITE") != "1"):
        raise RuntimeError("Wav2Lip-only strict runtime wrapper loaded outside its declared row")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RuntimeError("Wav2Lip strict runtime must preserve the pinned source/model directories")
    row_id = os.environ.get("CROSS_TFG_ROW_ID")
    row_seed = _bound_row_seed()
    expected_frames = _bound_expected_frame_count()
    if not row_id:
        raise RuntimeError("Wav2Lip strict runtime row/seed binding is incomplete")
    if Path(sys.argv[0]).resolve(strict=True) != W2L_ENTRY.resolve(strict=True):
        raise RuntimeError("Wav2Lip strict wrapper did not preserve the official inference entry")

    source_pins = _verified_face_detection_sources()
    api_pin = next(item for item in source_pins if item["relative_path"] == "api.py")
    model_source_pin = _require_regular(W2L_MODEL_SOURCE, W2L_MODEL_SOURCE_SHA256,
                                        "Wav2Lip generator model source")
    models_init_pin = _require_regular(W2L_MODELS_INIT, W2L_MODELS_INIT_SHA256,
                                       "Wav2Lip models package initializer")
    sfd_checkpoint_pin = _require_regular(SFD_CHECKPOINT, SFD_CHECKPOINT_SHA256,
                                          "package-local SFD checkpoint")
    if sfd_checkpoint_pin["bytes"] != SFD_CHECKPOINT_BYTES:
        raise RuntimeError("package-local SFD checkpoint byte count differs from the frozen identity")
    w2l_root = str(W2L_ENTRY.parent.resolve(strict=True))
    if w2l_root not in sys.path:
        sys.path.insert(0, w2l_root)
    face_api = importlib.import_module("face_detection.api")
    if Path(face_api.__file__).resolve(strict=True) != FACE_API.resolve(strict=True):
        raise RuntimeError("imported FaceAlignment module differs from the pinned deployed source")

    face_sfd = importlib.import_module("face_detection.detection.sfd")
    sfd_detector_module = importlib.import_module("face_detection.detection.sfd.sfd_detector")
    s3fd_module = importlib.import_module("face_detection.detection.sfd.net_s3fd")
    models_package = importlib.import_module("models")
    wav2lip_module = importlib.import_module("models.wav2lip")
    expected_origins = _expected_module_origins()
    imported_origins = {
        "face_detection": str(Path(importlib.import_module("face_detection").__file__).resolve(strict=True)),
        "face_detection.api": str(Path(face_api.__file__).resolve(strict=True)),
        "face_detection.detection.sfd": str(Path(face_sfd.__file__).resolve(strict=True)),
        "face_detection.detection.sfd.sfd_detector": str(Path(sfd_detector_module.__file__).resolve(strict=True)),
        "face_detection.detection.sfd.net_s3fd": str(Path(s3fd_module.__file__).resolve(strict=True)),
        "models": str(Path(models_package.__file__).resolve(strict=True)),
        "models.wav2lip": str(Path(wav2lip_module.__file__).resolve(strict=True)),
    }
    if imported_origins != expected_origins:
        raise RuntimeError("imported Wav2Lip module origins differ from the pinned dependency sources")
    if getattr(models_package, "Wav2Lip", None) is not getattr(wav2lip_module, "Wav2Lip", None):
        raise RuntimeError("official `from models import Wav2Lip` class is not the pinned hooked module class")
    if str(Path(sfd_detector_module.__file__).resolve(strict=True)) != str(
            (FACE_DETECTION_ROOT / "detection/sfd/sfd_detector.py").resolve(strict=True)):
        raise RuntimeError("imported SFD detector constructor differs from the pinned deployed source")
    def prohibit_sfd_network_fallback(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("SFD load_url fallback is prohibited; only the pinned package-local checkpoint is permitted")

    sfd_detector_module.load_url = prohibit_sfd_network_fallback
    if (importlib.metadata.version("torch") != str(importlib.import_module("torch").__version__)
            or importlib.metadata.version("numpy") != str(importlib.import_module("numpy").__version__)):
        raise RuntimeError("Wav2Lip runtime package metadata differs from imported Torch/NumPy versions")

    torch = importlib.import_module("torch")
    numpy = importlib.import_module("numpy")
    policy = dict(bootstrap.POLICY)
    if _observe_policy(torch) != policy:
        raise RuntimeError("strict state is not active before the pinned FaceAlignment constructor")

    outfile_index = sys.argv.index("--outfile") if "--outfile" in sys.argv else -1
    if outfile_index < 0 or outfile_index + 1 >= len(sys.argv):
        raise RuntimeError("Wav2Lip child lacks its exact --outfile binding")
    output_path = Path(sys.argv[outfile_index + 1]).resolve()
    if not output_path.is_absolute() or output_path.parent.name != "render":
        raise RuntimeError("Wav2Lip output is outside the isolated row render directory")
    row_root = output_path.parent.parent
    control_dir = Path(os.environ.get("CROSS_TFG_CONTROL_DIR", row_root / "control")).resolve()
    if control_dir != (row_root / "control").resolve():
        raise RuntimeError("Wav2Lip evidence control directory is outside its isolated row root")
    if control_dir.exists():
        if control_dir.is_symlink() or not control_dir.is_dir():
            raise RuntimeError("Wav2Lip evidence control path is not a real directory")
    else:
        control_dir.mkdir(mode=0o700)
    expected_avi = (Path.cwd() / "temp" / "result.avi").resolve()
    if Path.cwd().resolve() != (row_root / "generator_cwd").resolve():
        raise RuntimeError("Wav2Lip renderer cwd differs from the isolated per-row generator workspace")
    wrapper_sha = _sha_file(Path(__file__).resolve(strict=True))
    forward_events: list[dict[str, Any]] = []
    invocation = {"count": 0}
    original_init = face_api.FaceAlignment.__init__

    def strict_facealignment_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        invocation["count"] += 1
        call_index = invocation["count"]
        checkpoint_before = _require_regular(SFD_CHECKPOINT, SFD_CHECKPOINT_SHA256,
                                             "package-local SFD checkpoint before FaceAlignment")
        checkpoint_stat_before = SFD_CHECKPOINT.stat()
        before = _observe_policy(torch)
        if before != policy:
            raise RuntimeError("strict policy changed before FaceAlignment constructor")
        result = None
        constructor_error: BaseException | None = None
        try:
            result = original_init(self, *args, **kwargs)
        except BaseException as exc:
            constructor_error = exc
        after = _observe_policy(torch)
        checkpoint_stat_after = SFD_CHECKPOINT.stat()
        checkpoint_stable = ((checkpoint_stat_before.st_dev, checkpoint_stat_before.st_ino,
                              checkpoint_stat_before.st_size, checkpoint_stat_before.st_mtime_ns)
                             == (checkpoint_stat_after.st_dev, checkpoint_stat_after.st_ino,
                                 checkpoint_stat_after.st_size, checkpoint_stat_after.st_mtime_ns))
        expected_after = dict(policy)
        expected_after["torch_cudnn_benchmark"] = True
        changed_fields = sorted(key for key in policy if policy[key] != after.get(key))
        if torch.backends.cudnn.benchmark is not False:
            torch.backends.cudnn.benchmark = False
        restored = _observe_policy(torch)
        status = ("PASS_CONSTRUCTOR_RETURN_RESTORED" if constructor_error is None
                  and after == expected_after and restored == policy else "FAIL_POLICY_RESTORE")
        receipt = _write_receipt(control_dir / f"w2l_facealignment_policy_{call_index:02d}.json", {
            "schema": "cross_tfg_w2l_facealignment_constructor_policy_receipt_v1",
            "status": status, "row_id": row_id, "model_id": W2L_MODEL_ID,
            "seed": row_seed, "pid": os.getpid(),
            "entry_path": str(W2L_ENTRY.resolve(strict=True)),
            "strict_bootstrap_source_sha256": BOOTSTRAP_SHA256,
            "strict_bootstrap_receipt": bootstrap._w2l_runtime_startup_receipt,
            "runtime_compat_source_path": str(Path(__file__).resolve(strict=True)),
            "runtime_compat_source_sha256": wrapper_sha,
            "facealignment_api": api_pin,
            "face_detection_source_inventory": source_pins,
            "wav2lip_model_source": model_source_pin,
            "sfd_checkpoint": checkpoint_before,
            "sfd_checkpoint_identity_stable_during_constructor": checkpoint_stable,
            "sfd_network_fallback_prohibited": True,
            "policy_sha256": POLICY_SHA256,
            "policy_before_constructor": before,
            "policy_at_constructor_return": after,
            "expected_after_constructor": expected_after,
            "changed_policy_fields": changed_fields,
            "policy_after_restore": restored,
            "benchmark_override_observed": after.get("torch_cudnn_benchmark") is True,
            "policy_restored_after_constructor": restored == policy,
            "constructor_exception": (None if constructor_error is None else
                                       f"{type(constructor_error).__name__}: {constructor_error}"),
        })
        if status != "PASS_CONSTRUCTOR_RETURN_RESTORED" or not checkpoint_stable:
            raise RuntimeError(f"FaceAlignment changed more than the pinned benchmark flag or policy restore failed: {receipt}")
        if constructor_error is not None:
            raise constructor_error
        net = getattr(getattr(self, "face_detector", None), "face_detector", None)
        expected_net_type = getattr(s3fd_module, "s3fd", None)
        if net is None or expected_net_type is None or type(net) is not expected_net_type:
            raise RuntimeError("FaceAlignment did not construct the pinned S3FD module instance")

        def s3fd_forward_pre_hook(module: Any, args: tuple[Any, ...]) -> None:
            observed_before = _observe_policy(torch)
            # The pinned SFD detector sets cudnn.benchmark=True before net(img[s]).
            # Restore immediately at this actual forward pre-hook, then assert all policy fields.
            if torch.backends.cudnn.benchmark is not False:
                torch.backends.cudnn.benchmark = False
            observed_at_forward = _observe_policy(torch)
            shapes = []
            for value in args:
                if hasattr(value, "shape"):
                    shapes.append(list(value.shape))
            event = {
                "kind": "sfd_s3fd_forward_pre_hook",
                "ordinal": 1 + sum(item["kind"] == "sfd_s3fd_forward_pre_hook" for item in forward_events),
                "pid": os.getpid(), "row_id": row_id, "model_id": W2L_MODEL_ID, "seed": row_seed,
                "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
                "module_source_path": str(Path(s3fd_module.__file__).resolve(strict=True)),
                "module_source_sha256": FACE_DETECTION_SOURCE_PINS["detection/sfd/net_s3fd.py"][1],
                "policy_before_guard": observed_before,
                "policy_at_forward_entry": observed_at_forward,
                "benchmark_override_observed_and_restored": (
                    observed_before["torch_cudnn_benchmark"] is True
                    and observed_at_forward == policy),
                "input_shapes": shapes,
            }
            forward_events.append(event)
            if observed_at_forward != policy:
                raise RuntimeError(f"strict policy failed at SFD S3FD forward entry: {event}")

        net.register_forward_pre_hook(s3fd_forward_pre_hook)
        return result

    strict_facealignment_init.__name__ = original_init.__name__
    strict_facealignment_init.__qualname__ = original_init.__qualname__
    strict_facealignment_init.__doc__ = original_init.__doc__
    face_api.FaceAlignment.__init__ = strict_facealignment_init

    wav2lip_class = getattr(wav2lip_module, "Wav2Lip", None)
    if wav2lip_class is None or not callable(getattr(wav2lip_class, "register_forward_pre_hook", None)):
        raise RuntimeError("pinned Wav2Lip model module has no expected Wav2Lip class")
    original_wav2lip_init = wav2lip_class.__init__
    w2l_instance_count = {"count": 0}

    def strict_wav2lip_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_wav2lip_init(self, *args, **kwargs)
        w2l_instance_count["count"] += 1

        def wav2lip_forward_pre_hook(module: Any, inputs: tuple[Any, ...]) -> None:
            observed = _observe_policy(torch)
            shapes = []
            for value in inputs:
                if hasattr(value, "shape"):
                    shapes.append(list(value.shape))
            event = {
                "kind": "wav2lip_generator_forward_pre_hook",
                "ordinal": 1 + sum(item["kind"] == "wav2lip_generator_forward_pre_hook" for item in forward_events),
                "pid": os.getpid(), "row_id": row_id, "model_id": W2L_MODEL_ID, "seed": row_seed,
                "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
                "module_source_path": str(Path(wav2lip_module.__file__).resolve(strict=True)),
                "module_source_sha256": W2L_MODEL_SOURCE_SHA256,
                "policy_at_forward_entry": observed,
                "input_shapes": shapes,
            }
            forward_events.append(event)
            if observed != policy:
                raise RuntimeError(f"strict policy failed at Wav2Lip model forward entry: {event}")

        self.register_forward_pre_hook(wav2lip_forward_pre_hook)
        return result

    strict_wav2lip_init.__name__ = original_wav2lip_init.__name__
    strict_wav2lip_init.__qualname__ = original_wav2lip_init.__qualname__
    strict_wav2lip_init.__doc__ = original_wav2lip_init.__doc__
    wav2lip_class.__init__ = strict_wav2lip_init

    cv2 = face_api.cv2
    original_video_writer = cv2.VideoWriter
    writer_count = {"count": 0}

    class _ObservedVideoWriter:
        def __init__(self, writer: Any, output_path: Path, fps: float,
                     size_wh: tuple[int, int], fourcc: Any) -> None:
            self._writer = writer
            self._path = output_path
            self._fps = fps
            self._size_wh = size_wh
            self._fourcc = fourcc
            self._frames: list[dict[str, Any]] = []
            self._pixels_sha256 = hashlib.sha256()
            self._released = False

        def write(self, frame: Any) -> Any:
            array = numpy.asarray(frame)
            c_contiguous = bool(array.flags.c_contiguous)
            if not c_contiguous:
                array = numpy.ascontiguousarray(array)
            pixel_bytes = memoryview(array).cast("B")
            frame_sha = _sha_bytes(pixel_bytes)
            self._pixels_sha256.update(pixel_bytes)
            self._frames.append({"index": len(self._frames), "shape": list(array.shape),
                                 "dtype": str(array.dtype), "bytes": len(pixel_bytes),
                                 "c_contiguous_before_hash": c_contiguous,
                                 "pixel_array_sha256": frame_sha})
            return self._writer.write(frame)

        def release(self) -> Any:
            if self._released:
                return None
            result = self._writer.release()
            self._released = True
            avi_info = None
            if self._path.is_file() and not self._path.is_symlink():
                avi_info = {"path": str(self._path.resolve(strict=True)),
                            "bytes": self._path.stat().st_size,
                            "sha256": _sha_file(self._path)}
            shapes = sorted({tuple(frame["shape"]) for frame in self._frames})
            dtypes = sorted({frame["dtype"] for frame in self._frames})
            frame_hashes = [frame["pixel_array_sha256"] for frame in self._frames]
            valid = (writer_count["count"] == 1
                     and self._path.resolve() == expected_avi
                     and self._fps == EXPECTED_FPS
                     and self._size_wh == EXPECTED_SIZE_WH
                     and len(self._frames) == expected_frames
                     and shapes == [(512, 512, 3)]
                     and dtypes == ["uint8"]
                     and avi_info is not None and avi_info["bytes"] > 0)
            record = _write_receipt(control_dir / "w2l_prewrite_pixel_arrays.json", {
                "schema": "cross_tfg_w2l_prewrite_pixel_array_receipt_v1",
                "status": "PASS" if valid else "FAIL_WRITER_SCOPE_OR_SHAPE",
                "row_id": row_id, "model_id": W2L_MODEL_ID, "seed": row_seed,
                "pid": os.getpid(), "entry_path": str(W2L_ENTRY.resolve(strict=True)),
                "strict_bootstrap_source_sha256": BOOTSTRAP_SHA256,
                "runtime_compat_source_path": str(Path(__file__).resolve(strict=True)),
                "runtime_compat_source_sha256": wrapper_sha,
                "facealignment_api_sha256": api_pin["sha256"],
                "writer_output": str(self._path), "writer_fps": self._fps,
                "writer_size_wh": list(self._size_wh), "writer_fourcc": self._fourcc,
                "expected_frame_count": expected_frames,
                "frame_count": len(self._frames), "frame_shapes": [list(shape) for shape in shapes],
                "frame_dtypes": dtypes, "pixel_bytes_sha256": self._pixels_sha256.hexdigest(),
                "ordered_frame_hashes_sha256": _sha_bytes(_canonical(frame_hashes)),
                "frame_pixel_array_sha256s": frame_hashes,
                "frame_records": self._frames,
                "raw_intermediate_avi": avi_info,
                "frames_hashed_before_unchanged_writer_write": True,
            })
            if not valid:
                raise RuntimeError(f"pre-write Wav2Lip pixel capture did not match the pinned frame contract: {record}")
            detector_events = [item for item in forward_events if item["kind"] == "sfd_s3fd_forward_pre_hook"]
            generator_events = [item for item in forward_events if item["kind"] == "wav2lip_generator_forward_pre_hook"]
            runtime_valid = (invocation["count"] == 1 and w2l_instance_count["count"] == 1
                             and len(detector_events) > 0 and len(generator_events) > 0
                             and all(item["policy_at_forward_entry"] == policy for item in detector_events)
                             and all(item.get("benchmark_override_observed_and_restored") is True
                                     for item in detector_events)
                             and all(item["policy_at_forward_entry"] == policy for item in generator_events))
            _write_receipt(control_dir / "w2l_actual_forward_policy_receipt.json", {
                "schema": "cross_tfg_w2l_actual_forward_policy_receipt_v1",
                "status": "PASS_ACTUAL_FORWARD_ENTRIES" if runtime_valid else "FAIL_MISSING_OR_NONSTRICT_FORWARD_ENTRY",
                "row_id": row_id, "model_id": W2L_MODEL_ID, "seed": row_seed, "pid": os.getpid(),
                "entry_path": str(W2L_ENTRY.resolve(strict=True)),
                "strict_bootstrap_source_sha256": BOOTSTRAP_SHA256,
                "strict_bootstrap_receipt": bootstrap._w2l_runtime_startup_receipt,
                "runtime_compat_source_path": str(Path(__file__).resolve(strict=True)),
                "runtime_compat_source_sha256": wrapper_sha,
                "policy_sha256": POLICY_SHA256,
                "face_detection_source_inventory": source_pins,
                "wav2lip_models_package_initializer": models_init_pin,
                "wav2lip_model_source": model_source_pin,
                "sfd_checkpoint": sfd_checkpoint_pin,
                "sfd_network_fallback_prohibited": True,
                "imported_module_origins": imported_origins,
                "runtime_versions": {
                    "torch": str(torch.__version__), "numpy": str(numpy.__version__),
                    "opencv": str(getattr(cv2, "__version__", "unknown")),
                },
                "facealignment_constructor_count": invocation["count"],
                "wav2lip_model_instance_count": w2l_instance_count["count"],
                "sfd_forward_count": len(detector_events),
                "wav2lip_generator_forward_count": len(generator_events),
                "forward_entry_observations": forward_events,
                "all_observed_actual_forward_entries_match_strict_policy": runtime_valid,
                "raw_intermediate_avi": avi_info,
            })
            if not runtime_valid:
                raise RuntimeError("Wav2Lip strict runtime lacked exact actual-forward policy receipts")
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(self._writer, name)

    def observed_video_writer(*args: Any, **kwargs: Any) -> Any:
        filename = args[0] if args else kwargs.get("filename")
        fourcc = args[1] if len(args) > 1 else kwargs.get("fourcc")
        fps = args[2] if len(args) > 2 else kwargs.get("fps")
        size = args[3] if len(args) > 3 else kwargs.get("frameSize", kwargs.get("frame_size"))
        if filename is None or size is None or fps is None:
            raise RuntimeError("unexpected Wav2Lip VideoWriter constructor signature")
        output_path = Path(filename)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path = output_path.resolve()
        if output_path != expected_avi:
            raise RuntimeError(f"unexpected Wav2Lip intermediate writer target: {output_path}")
        writer_count["count"] += 1
        writer = original_video_writer(*args, **kwargs)
        return _ObservedVideoWriter(writer, output_path, float(fps),
                                    (int(size[0]), int(size[1])), fourcc)
    cv2.VideoWriter = observed_video_writer


if os.environ.get("CROSS_TFG_RUNTIME_COMPAT_PREFLIGHT") == "1":
    PREFLIGHT_RECORD = _preflight_record()
else:
    _bootstrap = _load_pinned_bootstrap()
    _install_wav2lip_runtime_compat(_bootstrap)
