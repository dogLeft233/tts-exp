"""Wav2Lip-only strict-policy repair and pre-AVI pixel audit.

This is placed before the shared runtime bootstrap on the Wav2Lip child's
PYTHONPATH. It executes that exact pinned bootstrap first, then narrowly
accounts for FaceAlignment.__init__ changing cuDNN benchmarking and observes
the unchanged ndarray frames passed to OpenCV's VideoWriter.
"""
from __future__ import annotations

import hashlib
import importlib
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
FACE_API = Path("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/api.py")
FACE_API_SHA256 = "c050676db49d8eb5de19f13c171101892b60b992b2a730cfbc8534967e4faeaa"
W2L_MODEL_ID = "wav2lip_gan_bac9a81e"
SEED = 20261003
POLICY_SHA256 = "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
EXPECTED_FRAMES = 141
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
    if module.POLICY_SHA256 if hasattr(module, "POLICY_SHA256") else False:
        raise RuntimeError("unexpected shared bootstrap interface")
    if _sha_bytes(_canonical(module.POLICY)) != POLICY_SHA256:
        raise RuntimeError("shared bootstrap policy does not match the frozen strict policy")
    if _observe_policy(module.torch if hasattr(module, "torch") else importlib.import_module("torch")) != module.POLICY:
        raise RuntimeError("shared bootstrap strict state changed before Wav2Lip compatibility setup")
    row_id = os.environ.get("CROSS_TFG_ROW_ID")
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
            or startup.get("seed") != SEED
            or Path(startup.get("entry_path", "")).resolve(strict=True)
            != W2L_ENTRY.resolve(strict=True)):
        raise RuntimeError("pinned shared bootstrap did not write the expected native Wav2Lip receipt")
    module._w2l_runtime_bootstrap_pin = bootstrap_pin
    module._w2l_runtime_startup_receipt = {"path": str(startup_path), "bytes": len(raw),
                                           "sha256": _sha_bytes(raw)}
    return module


def _install_wav2lip_runtime_compat(bootstrap: Any) -> None:
    if (os.environ.get("CROSS_TFG_STRICT_STARTUP_POLICY") != "1"
            or os.environ.get("CROSS_TFG_MODEL_ID") != W2L_MODEL_ID):
        raise RuntimeError("Wav2Lip-only strict runtime wrapper loaded outside its declared row")
    row_id = os.environ.get("CROSS_TFG_ROW_ID")
    if not row_id or os.environ.get("CROSS_TFG_RENDER_SEED") != str(SEED):
        raise RuntimeError("Wav2Lip strict runtime row/seed binding is incomplete")
    if Path(sys.argv[0]).resolve(strict=True) != W2L_ENTRY.resolve(strict=True):
        raise RuntimeError("Wav2Lip strict wrapper did not preserve the official inference entry")

    api_pin = _require_regular(FACE_API, FACE_API_SHA256, "Wav2Lip FaceAlignment API")
    w2l_root = str(W2L_ENTRY.parent.resolve(strict=True))
    if w2l_root not in sys.path:
        sys.path.insert(0, w2l_root)
    face_api = importlib.import_module("face_detection.api")
    if Path(face_api.__file__).resolve(strict=True) != FACE_API.resolve(strict=True):
        raise RuntimeError("imported FaceAlignment module differs from the pinned deployed source")

    torch = importlib.import_module("torch")
    numpy = importlib.import_module("numpy")
    policy = dict(bootstrap.POLICY)
    if _observe_policy(torch) != policy:
        raise RuntimeError("strict state is not active before the pinned FaceAlignment constructor")

    control_dir = Path(os.environ["CROSS_TFG_CONTROL_DIR"]).resolve(strict=True)
    expected_avi = (control_dir.parent / "generator_cwd" / "temp" / "result.avi").resolve()
    compat_path = FACE_API.resolve(strict=True)
    wrapper_sha = _sha_file(Path(__file__).resolve(strict=True))
    invocation = {"count": 0}
    original_init = face_api.FaceAlignment.__init__

    def strict_facealignment_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        invocation["count"] += 1
        call_index = invocation["count"]
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
        expected_after = dict(policy)
        expected_after["torch_cudnn_benchmark"] = True
        changed_fields = sorted(key for key in policy if policy[key] != after.get(key))
        if torch.backends.cudnn.benchmark is not False:
            torch.backends.cudnn.benchmark = False
        restored = _observe_policy(torch)
        status = ("PASS_POLICY_RESTORED" if constructor_error is None
                  and after == expected_after and restored == policy else "FAIL_POLICY_RESTORE")
        receipt = _write_receipt(control_dir / f"w2l_facealignment_policy_{call_index:02d}.json", {
            "schema": "cross_tfg_w2l_facealignment_constructor_policy_receipt_v1",
            "status": status, "row_id": row_id, "model_id": W2L_MODEL_ID,
            "seed": SEED, "pid": os.getpid(),
            "entry_path": str(W2L_ENTRY.resolve(strict=True)),
            "strict_bootstrap_source_sha256": BOOTSTRAP_SHA256,
            "strict_bootstrap_receipt": bootstrap._w2l_runtime_startup_receipt,
            "runtime_compat_source_path": str(Path(__file__).resolve(strict=True)),
            "runtime_compat_source_sha256": wrapper_sha,
            "facealignment_api": api_pin,
            "policy_sha256": POLICY_SHA256,
            "policy_before_constructor": before,
            "policy_after_constructor": after,
            "expected_after_constructor": expected_after,
            "changed_policy_fields": changed_fields,
            "policy_after_restore": restored,
            "benchmark_override_observed": after.get("torch_cudnn_benchmark") is True,
            "benchmark_restored_before_renderer_forwards": restored == policy,
            "constructor_exception": (None if constructor_error is None else
                                       f"{type(constructor_error).__name__}: {constructor_error}"),
        })
        if status != "PASS_POLICY_RESTORED":
            raise RuntimeError(f"FaceAlignment changed more than the pinned benchmark flag or policy restore failed: {receipt}")
        if constructor_error is not None:
            raise constructor_error
        return result

    strict_facealignment_init.__name__ = original_init.__name__
    strict_facealignment_init.__qualname__ = original_init.__qualname__
    strict_facealignment_init.__doc__ = original_init.__doc__
    face_api.FaceAlignment.__init__ = strict_facealignment_init

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
                     and len(self._frames) == EXPECTED_FRAMES
                     and shapes == [(512, 512, 3)]
                     and dtypes == ["uint8"]
                     and avi_info is not None and avi_info["bytes"] > 0)
            record = _write_receipt(control_dir / "w2l_prewrite_pixel_arrays.json", {
                "schema": "cross_tfg_w2l_prewrite_pixel_array_receipt_v1",
                "status": "PASS" if valid else "FAIL_WRITER_SCOPE_OR_SHAPE",
                "row_id": row_id, "model_id": W2L_MODEL_ID, "seed": SEED,
                "pid": os.getpid(), "entry_path": str(W2L_ENTRY.resolve(strict=True)),
                "strict_bootstrap_source_sha256": BOOTSTRAP_SHA256,
                "runtime_compat_source_path": str(Path(__file__).resolve(strict=True)),
                "runtime_compat_source_sha256": wrapper_sha,
                "facealignment_api_sha256": api_pin["sha256"],
                "writer_output": str(self._path), "writer_fps": self._fps,
                "writer_size_wh": list(self._size_wh), "writer_fourcc": self._fourcc,
                "expected_frame_count": EXPECTED_FRAMES,
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


_bootstrap = _load_pinned_bootstrap()
_install_wav2lip_runtime_compat(_bootstrap)
