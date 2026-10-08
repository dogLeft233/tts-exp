#!/usr/bin/env python3
"""Run a frozen Stage-A plan through a receipt-checked silent-video adapter.

This wrapper keeps the pinned Stage-A scorer and its original-PCM mux/scoring
path unchanged.  It adds a strict adapter-side receipt check so model-native
source, weights, render clock, and input waveform are carried into each result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

CORE_DEFAULT = "/root/tts-research-system-20261001/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt06.py"
CORE_SHA256_DEFAULT = "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"
ADAPTER_CONTRACT = "cross_tfg_silent_video_cli_v1"
W2L_MODEL_ID = "wav2lip_gan_bac9a81e"
W2L_ENTRY_SHA256 = "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6"
W2L_WRAPPER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/wav2lip_runtime_compat_attempt04/sitecustomize.py"
W2L_WRAPPER_SHA256 = "462884a2534f73d93eb09cd05c87fefda677e3647366360fb81a9622fa7592ca"
W2L_BOOTSTRAP_REL = "scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt03/sitecustomize.py"
W2L_BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
W2L_POLICY_CONTRACT_SHA256 = "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
W2L_POLICY_PAYLOAD_SHA256 = "54d853a6b6ce09cc8c757297f9617b5ecf4e0d01458a0b2e33dcf0316a4417f5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read_w2l_receipt(path: Path, schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import stat

    if path.is_symlink():
        raise RuntimeError(f"Wav2Lip runtime receipt is symlinked: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size > 8 * 1024 * 1024:
        raise RuntimeError(f"Wav2Lip runtime receipt is not a bounded regular file: {path}")
    raw = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(raw) != before.st_size):
        raise RuntimeError(f"Wav2Lip runtime receipt changed while being read: {path}")
    record = json.loads(raw.decode("utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if (record.get("schema") != schema
            or record.get("receipt_sha256") != hashlib.sha256(_canonical({
                key: value for key, value in record.items() if key != "receipt_sha256"})).hexdigest()):
        raise RuntimeError(f"Wav2Lip runtime receipt schema or self-hash is invalid: {path}")
    return record, {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": digest}


def _wav2lip_native_generator(core: Any, *, row: Mapping[str, Any], model: Mapping[str, Any],
                              audio_path: Path, output_video: Path, work: Path,
                              args: argparse.Namespace) -> dict[str, Any]:
    """Run the official inference entry so pinned sitecustomize hooks cover its forwards."""
    generator = model.get("generator")
    if not isinstance(generator, Mapping) or row.get("model_id") != W2L_MODEL_ID:
        raise RuntimeError("native Wav2Lip entry requires the frozen Wav2Lip model row")
    repository = Path(str(generator.get("repo", ""))).resolve(strict=True)
    entry = repository / "inference.py"
    python = str(generator.get("python", ""))  # keep the environment symlink intact
    if not python or not Path(python).is_file():
        raise RuntimeError("Wav2Lip interpreter is absent")
    if entry.is_symlink() or not entry.is_file() or _sha256(entry) != W2L_ENTRY_SHA256:
        raise RuntimeError("official Wav2Lip inference entry differs from the frozen source pin")
    if (generator.get("native_entry_path") != str(entry)
            or generator.get("native_entry_sha256") != W2L_ENTRY_SHA256):
        raise RuntimeError("Wav2Lip runtime binding does not select the pinned native inference entry")
    project_root = Path(args.project_root).resolve(strict=True)
    wrapper = project_root / W2L_WRAPPER_REL
    bootstrap = project_root / W2L_BOOTSTRAP_REL
    if (_sha256(wrapper) != W2L_WRAPPER_SHA256 or _sha256(bootstrap) != W2L_BOOTSTRAP_SHA256
            or generator.get("runtime_compat_sitecustomize_path") != str(wrapper)
            or generator.get("runtime_compat_sitecustomize_sha256") != W2L_WRAPPER_SHA256
            or generator.get("strict_bootstrap_path") != str(bootstrap)
            or generator.get("strict_bootstrap_sha256") != W2L_BOOTSTRAP_SHA256):
        raise RuntimeError("Wav2Lip compatibility wrapper or shared bootstrap differs from its frozen pin")

    binding = row.get("wav2lip_native_frame_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("Wav2Lip plan row lacks its source-derived native frame-count binding")
    natural = row.get("natural_pcm")
    if (not isinstance(natural, Mapping)
            or binding.get("sample_id") != row.get("sample_id")
            or binding.get("natural_pcm_sha256") != natural.get("pcm_sha256")
            or binding.get("sample_count") != natural.get("sample_count")
            or (binding.get("fps_num"), binding.get("fps_den")) != (25, 1)
            or binding.get("probe_raw_sha256") != generator.get("native_frame_count_probe_raw_sha256")
            or binding.get("probe_receipt_sha256") != generator.get("native_frame_count_probe_receipt_sha256")):
        raise RuntimeError("Wav2Lip frame count is not joined to this exact natural PCM/sample/FPS row")
    expected_frames = binding.get("expected_inference_frame_count")
    if type(expected_frames) is not int or expected_frames <= 0:
        raise RuntimeError("Wav2Lip source-derived expected frame count is invalid")
    from scripts.audio_enhancement.audio_io import read_canonical_wav
    input_samples, input_info = read_canonical_wav(audio_path.resolve(strict=True))
    expected_samples = natural.get("sample_count")
    if (input_info.source_sample_rate != 16000 or input_info.source_channels != 1
            or input_info.source_frames != input_samples.size or input_info.frames != expected_samples
            or input_samples.size != expected_samples):
        raise RuntimeError("actual transformed Wav2Lip WAV does not preserve the bound natural PCM sample count")
    generator_input = {
        "path": str(audio_path.resolve(strict=True)),
        "file_sha256": input_info.file_sha256,
        "pcm_float32_sha256": input_info.pcm_float32_sha256,
        "sample_rate_hz": 16000, "channels": 1, "sample_count": int(input_info.frames),
        "matches_bound_natural_sample_count": True,
    }
    seed = row.get("seed")
    if type(seed) is not int or seed < 0:
        raise RuntimeError("Wav2Lip render seed is not bound by the frozen row")

    child_cwd = work / "generator_cwd"
    child_cwd.mkdir(parents=True, exist_ok=False)
    (child_cwd / "temp").mkdir()
    control_dir = work / "control"
    control_dir.mkdir(mode=0o700, exist_ok=False)
    receipt_dir = Path(os.environ.get("CROSS_TFG_DETERMINISM_RECEIPT_DIR", ""))
    if not receipt_dir.is_absolute() or receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise RuntimeError("Wav2Lip renderer lacks the outer run's strict receipt directory")

    checkpoint = Path(str(generator.get("checkpoint", "")))
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        raise RuntimeError("Wav2Lip checkpoint path is absent from the current-host binding")
    expected_checkpoint = str(generator.get("checkpoint_sha256", "")).lower()
    if len(expected_checkpoint) != 64 or _sha256(checkpoint) != expected_checkpoint:
        raise RuntimeError("Wav2Lip checkpoint SHA differs from the current-host binding")
    face = Path(str(row.get("source_face", {}).get("path", ""))).resolve(strict=True)
    argv = [python, str(entry), "--checkpoint_path", str(checkpoint), "--face", str(face),
            "--audio", str(audio_path.resolve(strict=True)), "--outfile", str(output_video.resolve()),
            "--static", "True", "--fps", "25", "--pads", "0", "10", "0", "0"]
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("CROSS_TFG_"):
            env.pop(key, None)
    wrapper_dir = wrapper.parent
    bootstrap_dir = bootstrap.parent
    env.update({
        "CUDA_VISIBLE_DEVICES": str(generator.get("cuda_visible_devices", "0")),
        "PYTHONHASHSEED": "20261003", "TFG_DETERMINISM_SEED": str(seed),
        "CROSS_TFG_RENDER_SEED": str(seed), "CROSS_TFG_STRICT_STARTUP_POLICY": "1",
        "CROSS_TFG_ROW_ID": str(row["row_id"]), "CROSS_TFG_MODEL_ID": W2L_MODEL_ID,
        "CROSS_TFG_AUDIO_FRONTEND": str(row.get("audio_frontend", "")),
        "CROSS_TFG_BACKEND_ENTRY": str(entry.resolve(strict=True)),
        "CROSS_TFG_EXPECTED_FRAME_COUNT": str(expected_frames),
        "CROSS_TFG_DETERMINISM_RECEIPT_DIR": str(receipt_dir.resolve(strict=True)),
        "CROSS_TFG_CONTROL_DIR": str(control_dir.resolve(strict=True)),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": os.pathsep.join((str(wrapper_dir), str(bootstrap_dir), str(repository),
                                        str(project_root))),
    })
    env.pop("CROSS_TFG_RUNTIME_COMPAT_PREFLIGHT", None)
    execution = core.run_logged(
        argv, cwd=child_cwd, env=env,
        stdout_path=work / "logs/generator.stdout.log",
        stderr_path=work / "logs/generator.stderr.log",
        timeout=float(generator.get("timeout_seconds", args.generator_timeout)),
    )
    if execution.get("returncode") != 0 or not output_video.is_file() or output_video.stat().st_size == 0:
        raise RuntimeError(f"native Wav2Lip inference failed ({execution.get('returncode')}); stderr={execution.get('stderr_path')}")

    forward, forward_ref = _read_w2l_receipt(
        control_dir / "w2l_actual_forward_policy_receipt.json",
        "cross_tfg_w2l_actual_forward_policy_receipt_v1")
    constructor, constructor_ref = _read_w2l_receipt(
        control_dir / "w2l_facealignment_policy_01.json",
        "cross_tfg_w2l_facealignment_constructor_policy_receipt_v1")
    prewrite, prewrite_ref = _read_w2l_receipt(
        control_dir / "w2l_prewrite_pixel_arrays.json",
        "cross_tfg_w2l_prewrite_pixel_array_receipt_v1")
    common = (str(row["row_id"]), W2L_MODEL_ID, seed)
    if (forward.get("status") != "PASS_ACTUAL_FORWARD_ENTRIES"
            or (forward.get("row_id"), forward.get("model_id"), forward.get("seed")) != common
            or forward.get("runtime_compat_source_sha256") != W2L_WRAPPER_SHA256
            or forward.get("strict_bootstrap_source_sha256") != W2L_BOOTSTRAP_SHA256
            or forward.get("policy_sha256") != W2L_POLICY_CONTRACT_SHA256
            or forward.get("policy_payload_sha256") != W2L_POLICY_PAYLOAD_SHA256
            or forward.get("sfd_network_fallback_prohibited") is not True
            or not forward.get("all_observed_actual_forward_entries_match_strict_policy")):
        raise RuntimeError("Wav2Lip actual-forward receipt does not attest the frozen runtime policy")
    if (constructor.get("status") != "PASS_CONSTRUCTOR_RETURN_RESTORED"
            or (constructor.get("row_id"), constructor.get("model_id"), constructor.get("seed")) != common
            or constructor.get("runtime_compat_source_sha256") != W2L_WRAPPER_SHA256
            or constructor.get("strict_bootstrap_source_sha256") != W2L_BOOTSTRAP_SHA256
            or constructor.get("sfd_network_fallback_prohibited") is not True
            or constructor.get("policy_restored_after_constructor") is not True):
        raise RuntimeError("Wav2Lip FaceAlignment constructor receipt does not attest exact policy restoration")
    if (prewrite.get("status") != "PASS" or
            (prewrite.get("row_id"), prewrite.get("model_id"), prewrite.get("seed")) != common
            or prewrite.get("runtime_compat_source_sha256") != W2L_WRAPPER_SHA256
            or prewrite.get("strict_bootstrap_source_sha256") != W2L_BOOTSTRAP_SHA256
            or prewrite.get("expected_frame_count") != expected_frames
            or prewrite.get("frame_count") != expected_frames
            or len(prewrite.get("frame_pixel_array_sha256s", [])) != expected_frames):
        raise RuntimeError("Wav2Lip pre-write pixel receipt differs from the source-derived frame binding")
    execution.update({
        "native_entry_path": str(entry.resolve(strict=True)),
        "native_entry_sha256": W2L_ENTRY_SHA256,
        "runtime_compat_source_sha256": W2L_WRAPPER_SHA256,
        "expected_frame_count": expected_frames,
        "generator_input_audio": generator_input,
        "wav2lip_runtime_compatibility_receipts": {
            "constructor_policy": {**constructor_ref, "status": constructor["status"]},
            "actual_forward_policy": {**forward_ref, "status": forward["status"]},
            "prewrite_pixel_arrays": {**prewrite_ref, "status": prewrite["status"],
                                       "frame_count": prewrite["frame_count"],
                                       "frame_pixel_array_sha256s": prewrite["frame_pixel_array_sha256s"]},
        },
    })
    return execution


def _load_core(path: Path, expected_sha256: str):
    path = path.resolve(strict=True)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"Stage-A core source SHA mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("cross_tfg_audio_head_stagea_core_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned Stage-A scoring core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _adapter_generator(core: Any, *, row: Mapping[str, Any], model: Mapping[str, Any],
                       audio_path: Path, output_video: Path, work: Path,
                       args: argparse.Namespace) -> dict[str, Any]:
    generator = model.get("generator")
    if not isinstance(generator, Mapping) or generator.get("contract") != ADAPTER_CONTRACT:
        raise RuntimeError(f"model generator must declare {ADAPTER_CONTRACT}")
    adapter = Path(str(generator.get("adapter_path", ""))).resolve(strict=True)
    adapter_sha = str(generator.get("adapter_sha256", "")).lower()
    if len(adapter_sha) != 64 or _sha256(adapter) != adapter_sha:
        raise RuntimeError("silent-video adapter is absent or does not match its frozen SHA")
    python = str(generator.get("python", ""))
    if not python or not Path(python).is_file():
        raise RuntimeError("adapter Python interpreter is absent")
    cwd = Path(str(generator.get("cwd", ""))).resolve(strict=True)
    fps_num = int(generator.get("fps_num", 0))
    fps_den = int(generator.get("fps_den", 0))
    if (fps_num, fps_den) != (25, 1):
        raise RuntimeError("the cross-front-end adapter contract is frozen to 25/1 fps")
    face = Path(str(row.get("source_face", {}).get("path", ""))).resolve(strict=True)
    native_entry_text = generator.get("render_entry_path") or generator.get("renderer_path")
    alternate_entry_text = generator.get("renderer_path")
    expected_native_sha = str(generator.get("renderer_sha256", "")).lower()
    if (not isinstance(native_entry_text, str) or not native_entry_text
            or not Path(native_entry_text).is_absolute()
            or (alternate_entry_text is not None and alternate_entry_text != native_entry_text)):
        raise RuntimeError("backend generator must bind one exact absolute native renderer entry")
    native_entry = Path(native_entry_text)
    if (native_entry.is_symlink() or not native_entry.is_file()
            or len(expected_native_sha) != 64 or _sha256(native_entry) != expected_native_sha):
        raise RuntimeError("native renderer entry is absent or differs from its frozen plan SHA")

    strict_receipt_text = os.environ.get("CROSS_TFG_DETERMINISM_RECEIPT_DIR", "")
    strict_receipt_dir = Path(strict_receipt_text)
    if (not strict_receipt_text or not strict_receipt_dir.is_absolute()
            or strict_receipt_dir.is_symlink() or not strict_receipt_dir.is_dir()):
        raise RuntimeError("adapter subprocess lacks the outer run's exact strict receipt directory")
    child_cwd = work / "generator_cwd"
    child_cwd.mkdir(parents=True, exist_ok=False)
    control_dir = work / "control"
    control_dir.mkdir(mode=0o700, exist_ok=False)
    argv = [python, str(adapter), "--audio", str(audio_path), "--face", str(face),
            "--outfile", str(output_video), "--seed", str(int(row["seed"])), "--fps", "25"]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(generator.get("cuda_visible_devices", "0")),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CROSS_TFG_MODEL_ID": str(row.get("model_id", "")),
        "CROSS_TFG_AUDIO_FRONTEND": str(row.get("audio_frontend", "")),
        "CROSS_TFG_ROW_ID": str(row.get("row_id", "")),
        "CROSS_TFG_RENDER_SEED": str(int(row["seed"])),
        "PYTHONPATH": args.project_root + os.pathsep + env.get("PYTHONPATH", ""),
    })
    overrides = generator.get("env", {})
    if not isinstance(overrides, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in overrides.items()):
        raise RuntimeError("generator.env must be a string-to-string object")
    env.update(overrides)
    # Reassert the row-local strict contract after descriptor overlays. Ditto's
    # historic descriptor carries a shared Stage-B control path, which must not
    # receive this row's GPU-gate state or receipts.
    env.update({
        "CROSS_TFG_STRICT_STARTUP_POLICY": "1",
        "CROSS_TFG_ROW_ID": str(row.get("row_id", "")),
        "CROSS_TFG_MODEL_ID": str(row.get("model_id", "")),
        "CROSS_TFG_AUDIO_FRONTEND": str(row.get("audio_frontend", "")),
        "CROSS_TFG_RENDER_SEED": str(int(row["seed"])),
        "CROSS_TFG_BACKEND_ENTRY": str(native_entry),
        "CROSS_TFG_DETERMINISM_RECEIPT_DIR": str(strict_receipt_dir.resolve(strict=True)),
        "CROSS_TFG_CONTROL_DIR": str(control_dir.resolve(strict=True)),
        "PYTHONHASHSEED": "20261003",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    execution = core.run_logged(
        argv, cwd=child_cwd, env=env,
        stdout_path=work / "logs/generator.stdout.log",
        stderr_path=work / "logs/generator.stderr.log",
        timeout=float(generator.get("timeout_seconds", args.generator_timeout)),
    )
    if execution.get("returncode") != 0 or not output_video.is_file() or output_video.stat().st_size == 0:
        raise RuntimeError(f"silent-video adapter failed ({execution.get('returncode')}); stderr={execution.get('stderr_path')}")

    receipt_suffix = generator.get("receipt_suffix")
    receipt_ref: dict[str, Any] | None = None
    if generator.get("receipt_required"):
        if not isinstance(receipt_suffix, str) or not receipt_suffix.startswith("."):
            raise RuntimeError("receipt-required generator has no valid receipt_suffix")
        receipt_path = output_video.with_suffix(receipt_suffix).resolve(strict=True)
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw.decode("utf-8"))
        expected_receipt = str(generator.get("receipt_schema", ""))
        if receipt.get("schema") != expected_receipt:
            raise RuntimeError("model adapter receipt schema mismatch")
        if receipt.get("row_id") != row.get("row_id") or receipt.get("seed") != int(row["seed"]):
            raise RuntimeError("model adapter receipt row/seed does not match this planned render")
        if receipt.get("adapter_sha256") != adapter_sha:
            raise RuntimeError("model adapter receipt source pin differs from the frozen plan")
        output_ref = receipt.get("silent_video")
        if not isinstance(output_ref, Mapping) or output_ref.get("path") != str(output_video.resolve()) or output_ref.get("file_sha256") != _sha256(output_video):
            raise RuntimeError("model adapter receipt does not bind the emitted silent video")
        input_ref = receipt.get("input_audio")
        if not isinstance(input_ref, Mapping) or input_ref.get("file_sha256") != _sha256(audio_path):
            raise RuntimeError("model adapter receipt does not bind the generated audio input")
        receipt_ref = {
            "path": str(receipt_path), "file_sha256": _sha256(receipt_path),
            "schema": receipt.get("schema"), "model_profile": receipt.get("model_profile"),
            "renderer_sha256": receipt.get("renderer_sha256"),
            "native_report_sha256": receipt.get("native_report_sha256"),
            "source_commit": receipt.get("source_commit"),
        }
    execution.update({
        "adapter_contract": ADAPTER_CONTRACT,
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_sha,
        "python": python,
        "cwd": str(cwd),
        "environment_overrides": dict(overrides),
        "output_video_path": str(output_video.resolve()),
    })
    if receipt_ref is not None:
        execution["adapter_receipt"] = receipt_ref
    return execution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stagea-core", default=CORE_DEFAULT)
    parser.add_argument("--expected-stagea-core-sha256", default=CORE_SHA256_DEFAULT)
    own, remaining = parser.parse_known_args(argv)
    core, _ = _load_core(Path(own.stagea_core), own.expected_stagea_core_sha256)
    stagea_args = core.build_parser().parse_args(remaining)
    original_generator = core._run_generator

    def generator(**kwargs: Any) -> dict[str, Any]:
        row = kwargs.get("row", {})
        model = kwargs.get("model", {})
        if isinstance(row, Mapping) and row.get("model_id") == W2L_MODEL_ID:
            return _wav2lip_native_generator(core, **kwargs)
        descriptor = model.get("generator", {}) if isinstance(model, Mapping) else {}
        if isinstance(descriptor, Mapping) and descriptor.get("contract") == ADAPTER_CONTRACT:
            return _adapter_generator(core, **kwargs)
        return original_generator(**kwargs)

    core._run_generator = generator
    return core.run(stagea_args)


if __name__ == "__main__":
    raise SystemExit(main())
