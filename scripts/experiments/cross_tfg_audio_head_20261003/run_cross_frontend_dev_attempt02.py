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

CORE_DEFAULT = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py"
CORE_SHA256_DEFAULT = "30c894ea31541cdde76a5a915bf635a36838a1cdb11f761c334c8986e71a01bc"
ADAPTER_CONTRACT = "cross_tfg_silent_video_cli_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    child_cwd = work / "generator_cwd"
    child_cwd.mkdir(parents=True, exist_ok=False)
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
    core._run_generator = lambda **kw: _adapter_generator(core, **kw)
    return core.run(stagea_args)


if __name__ == "__main__":
    raise SystemExit(main())
