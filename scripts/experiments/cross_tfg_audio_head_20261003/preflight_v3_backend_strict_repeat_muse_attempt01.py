#!/usr/bin/env python3
"""Read-only preflight for the Muse-only strict repeat subset.

It retains the complete six-row denominator and requests two Muse renders at the
frozen QC06 input and seed. It never loads a generator model, renders media, runs
SyncNet, or calls an external API. The root dispatcher owns GPU admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
RUNNER_REL = BASE + "/run_v3_backend_strict_repeat_qualification_attempt04.py"
RUNNER_SHA256 = "d1c58060b415140ffd13aa156d3adc5f9f8b593a50e54bd24c5e96fc83bb502d"
REQUESTED_MODELS = ("musetalk_fixed_portrait_attempt16",)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def write_receipt(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink() or path.exists() or path.parent.is_symlink() or not path.parent.is_dir():
        raise FileExistsError(f"preflight receipt path must be a new file in a real directory: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_runner() -> Any:
    path = ROOT / RUNNER_REL
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"pinned strict repeat runner is missing or unsafe: {path}")
    actual = sha_file(path)
    if actual != RUNNER_SHA256:
        raise ValueError(f"pilot03 source SHA mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("strict_repeat_muse_readonly_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the pinned pilot03 source for preflight")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(output_root: Path) -> dict[str, Any]:
    runner = load_runner()
    resolved_output = output_root.resolve()
    if output_root.is_symlink() or (output_root.exists() and
                                    (not output_root.is_dir() or any(output_root.iterdir()))):
        raise FileExistsError(f"pilot output must be absent or an empty real directory: {output_root}")
    if str(Path(sys.executable).resolve(strict=True)) != str(runner.W2L_PYTHON.resolve(strict=True)):
        raise RuntimeError("preflight must run in the exact pinned Wav2Lip Python environment")

    template, template_raw, template_ref, observed = runner.verify_dependency_template(REQUESTED_MODELS)
    identity = runner.verify_runtime_files(template, observed, REQUESTED_MODELS)
    w2l_python = runner.python_environment_identity(runner.W2L_PYTHON)
    syncnet_code = (
        "import json,sys,torch,numpy; "
        "print(json.dumps({'python':sys.executable,'torch':torch.__version__,'numpy':numpy.__version__},sort_keys=True))"
    )
    syncnet_python = json.loads(runner.run_probe([str(runner.SYNCNET_PYTHON), "-c", syncnet_code]))
    resources = runner.resource_snapshot(resolved_output)
    gpu = next((item for item in resources["gpus"] if item.get("uuid") == runner.GPU_UUID), None)
    gpu_gate = bool(gpu is not None and gpu.get("memory_free_mib", -1) >= runner.MIN_GPU_FREE_MIB)
    self_path = Path(__file__).resolve(strict=True)
    return {
        "schema": "cross_tfg_backend_strict_repeat_subset_preflight_v1",
        "status": "PASS_PRECHECK_DISPATCH_ADMISSION_REQUIRED" if gpu_gate else "BLOCKED_RESOURCE_GATE",
        "performed": False,
        "rendering_performed": False,
        "syncnet_scoring_performed": False,
        "api_calls_performed": False,
        "model_forward_performed": False,
        "root_dispatcher_exclusive_gpu_admission": "REQUIRED_SEPARATE_GATE",
        "runner": {"path": str((ROOT / RUNNER_REL).resolve(strict=True)),
                   "bytes": (ROOT / RUNNER_REL).stat().st_size, "sha256": RUNNER_SHA256},
        "preflight_script": {"path": str(self_path), "bytes": self_path.stat().st_size,
                             "sha256": sha_file(self_path)},
        "run_id": runner.RUN_ID,
        "output_root": str(resolved_output),
        "planned_roster": list(runner.MODELS),
        "requested_models": list(REQUESTED_MODELS),
        "planned_rows": 6,
        "requested_render_rows": 2 * len(REQUESTED_MODELS),
        "strict_policy_sha256": runner.QC06_POLICY_SHA256,
        "qc06_manifest_sha256": runner.QC06_MANIFEST_SHA256,
        "qc06_input_sha256": {"natural_1_wav": runner.QC06_SOURCE_SHA256,
                              "primary_face": runner.QC06_FACE_SHA256},
        "dependency_template": template_ref,
        "dependency_template_raw_bytes": len(template_raw),
        "backend_identity": identity["backend_identity"],
        "source_runtime_identity": identity,
        "wav2lip_python_identity": w2l_python,
        "syncnet_python_identity": syncnet_python,
        "resource_snapshot": resources,
        "gpu_headroom_gate": {"status": "PASS" if gpu_gate else "FAIL",
                              "expected_uuid": runner.GPU_UUID,
                              "minimum_free_mib": runner.MIN_GPU_FREE_MIB,
                              "observed": gpu},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.output_root)
        result["preflight_sha256"] = sha_bytes(canonical(result))
        if args.receipt_out is not None:
            write_receipt(args.receipt_out, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0 if result["status"] == "PASS_PRECHECK_DISPATCH_ADMISSION_REQUIRED" else 2
    except Exception as exc:
        failure = {"schema": "cross_tfg_backend_strict_repeat_subset_preflight_v1",
                   "status": "FAILED_PREFLIGHT", "performed": False,
                   "error_type": type(exc).__name__, "error": str(exc)}
        failure["preflight_sha256"] = sha_bytes(canonical(failure))
        if args.receipt_out is not None and not args.receipt_out.exists():
            try:
                write_receipt(args.receipt_out, failure)
            except OSError:
                pass
        print(json.dumps(failure,
                         ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
