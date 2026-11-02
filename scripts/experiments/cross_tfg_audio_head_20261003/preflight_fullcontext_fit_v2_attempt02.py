"""No-forward preflight for the frozen full-audio/center-32 fit-v2 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

FRAME_SAMPLES = 640
MFCC_FRAME_LENGTH = 400
MFCC_FRAME_STEP = 160
MFCC_WINDOW = 20
MFCC_HOP = 4
VIDEO_QUERIES = 27
LAG_RADIUS = 15
EXPECTED_PSF_TREE = "a419b997acfd21da8f4c7324e5be22e32357d085e37ac3376165e676c1e67c17"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: dict[str, Any]) -> str:
    return _sha_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _psf_tree_sha(package_root: Path) -> tuple[str, list[dict[str, Any]]]:
    package_root = package_root.resolve(strict=True)
    digest = hashlib.sha256()
    rows = []
    for path in sorted(p for p in package_root.rglob("*.py") if p.is_file() and "__pycache__" not in p.parts):
        relative = path.relative_to(package_root).as_posix()
        size = path.stat().st_size
        file_sha = _sha_file(path)
        digest.update(f"{relative}\0{file_sha}\0{size}\n".encode("utf-8"))
        rows.append({"path": relative, "bytes": size, "sha256": file_sha})
    return digest.hexdigest(), rows


def _counts(sample_count: int) -> tuple[int, int, int]:
    mfcc_frames = 1 if sample_count <= MFCC_FRAME_LENGTH else 1 + math.ceil((sample_count - MFCC_FRAME_LENGTH) / MFCC_FRAME_STEP)
    all_valid = max(0, (mfcc_frames - MFCC_WINDOW) // MFCC_HOP + 1)
    official = max(0, sample_count // FRAME_SAMPLES - 5)
    if all_valid < official:
        raise ValueError(f"PSF window count {all_valid} is below official aligned query count {official}")
    return int(mfcc_frames), int(all_valid), int(official)


def _pcm16(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32 or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("fit source must decode to nonempty finite mono float32")
    return np.ascontiguousarray(
        np.rint(np.clip(audio.astype(np.float64) * 32768.0, -32768.0, 32767.0)).astype("<i2")
    )


def _version_report(execution: dict[str, Any]) -> dict[str, Any]:
    import librosa
    import torch

    if str(sys.executable) != execution.get("python_executable_literal"):
        raise ValueError(f"Python executable literal mismatch: {sys.executable}")
    if str(torch.__version__) != execution.get("torch_version"):
        raise ValueError(f"PyTorch version mismatch: {torch.__version__}")
    if str(np.__version__) != execution.get("numpy_version"):
        raise ValueError(f"NumPy version mismatch: {np.__version__}")
    if str(librosa.__version__) != execution.get("librosa_version"):
        raise ValueError(f"librosa version mismatch: {librosa.__version__}")
    expected_env = execution.get("environment", {})
    for name, value in expected_env.items():
        if os.environ.get(name) != str(value):
            raise ValueError(f"environment variable {name} differs from the frozen fit plan")
    if not torch.cuda.is_available():
        raise ValueError("frozen fit environment requires CUDA")
    if torch.cuda.device_count() != 1:
        raise ValueError(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    actual_controls = {
        "torch_num_threads": int(torch.get_num_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_device_count": int(torch.cuda.device_count()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    if actual_controls != {
        "torch_num_threads": 1,
        "cuda_available": True,
        "visible_cuda_device_count": 1,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }:
        raise ValueError("actual Torch/CUDA controls differ from the pinned fit controls")
    return {
        "python_executable_literal": str(sys.executable),
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "librosa_version": str(librosa.__version__),
        "controls": actual_controls,
    }


def preflight(*, plan_path: Path, payload_root: Path, audio_root: Path, report_path: Path) -> dict[str, Any]:
    from scripts.audio_enhancement.audio_io import read_canonical_wav
    from python_speech_features import mfcc
    import python_speech_features as psf

    plan_path = plan_path.resolve(strict=True)
    payload_root = payload_root.resolve(strict=True)
    audio_root = audio_root.resolve(strict=True)
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw.decode("utf-8"))
    if plan.get("schema") != "cross_tfg_audio_head_historical_fullcontext_fit_v2" or plan.get("status") != "FIT_ONLY_FROZEN":
        raise ValueError("fit-v2 plan schema/status mismatch")
    plan_payload = {key: value for key, value in plan.items() if key != "plan_payload_sha256"}
    if plan.get("plan_payload_sha256") != _canonical_sha(plan_payload):
        raise ValueError("fit-v2 embedded canonical plan payload SHA mismatch")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 8 or any(row.get("split") != "fit" for row in rows):
        raise ValueError("preflight accepts exactly eight fit rows and no development/confirmation rows")
    speaker_ids = {str(row.get("speaker_id")) for row in rows}
    if speaker_ids != {"S0052", "S0053", "S0054", "S0055"}:
        raise ValueError("historical fit speaker roster differs from the frozen four-speaker set")
    if len({row.get("row_id") for row in rows}) != 8 or len({row.get("sample_id") for row in rows}) != 8:
        raise ValueError("fit row/sample identities are missing or duplicated")
    if any(row.get("historical_exposed") is not True for row in rows):
        raise ValueError("historical exposure flag must remain true for each training row")
    if plan.get("development_audio_opened") is True or plan.get("confirmation_audio_opened") is True:
        raise ValueError("fit-only plan cannot claim development or confirmation audio was opened")

    runtime = plan.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("adapter_factory") != "scripts.audio_enhancement.wav2lip_fullcontext_fit_runtime_v2_attempt02:create_runtime":
        raise ValueError("plan does not select the fresh full-context runtime factory")
    execution = runtime.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("plan.runtime.execution is missing")
    if execution.get("python_speech_features_tree_sha256") != EXPECTED_PSF_TREE:
        raise ValueError("plan PSF source-tree pin differs from the independently qualified pure source tree")
    if os.environ.get("CROSS_TFG_PSF_ROOT") != execution.get("python_speech_features_root"):
        raise ValueError("CROSS_TFG_PSF_ROOT differs from the pinned plan path")
    package_root = Path(psf.__file__).resolve(strict=True).parent
    try:
        package_root.relative_to(Path(str(execution["python_speech_features_root"])).resolve(strict=True))
    except ValueError as exc:
        raise ValueError("python_speech_features was imported outside the pinned overlay root") from exc
    tree_sha, psf_files = _psf_tree_sha(package_root)
    if tree_sha != EXPECTED_PSF_TREE or psf_files != [
        {"path": "__init__.py", "bytes": 20, "sha256": "1119a63b1cfff6652422e71c35bcd46b963a8152cb543772738702c5b08251b6"},
        {"path": "base.py", "bytes": 10904, "sha256": "5da9dd3a98ebf697d5ac0dec1cc79e605f6df4874d1aa327096acf3002aaabd1"},
        {"path": "sigproc.py", "bytes": 5371, "sha256": "f4769a1c7b01c1d42d11549e2c5db4a84159e755d61a2f72c59afdbaad0b1352"},
    ]:
        raise ValueError("imported python_speech_features tree/files differ from the frozen overlay")
    if execution.get("torch_controls") != {
        "num_threads": 1,
        "cuda_required": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }:
        raise ValueError("plan Torch/CUDA controls differ from the frozen fit contract")

    source_pins = plan.get("source_revision", {}).get("fit_package_source_files")
    if not isinstance(source_pins, list) or not source_pins:
        raise ValueError("plan does not pin the full fit-package source closure")
    source_rows = []
    for item in source_pins:
        rel = PurePosixPath(str(item.get("path", "")))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"unsafe source path in fit plan: {rel}")
        source_path = payload_root.joinpath("src", *rel.parts)
        digest = _sha_file(source_path)
        if digest != item.get("sha256") or source_path.stat().st_size != int(item.get("bytes", -1)):
            raise ValueError(f"bundled source closure mismatch: {rel}")
        source_rows.append({"path": rel.as_posix(), "bytes": source_path.stat().st_size, "sha256": digest})

    # Validate the pinned external Wav2Lip/SyncNet source trees and checkpoints
    # without constructing either model or running a forward pass.
    external_assets = []
    for family in ("wav2lip", "syncnet"):
        binding = runtime.get(family)
        if not isinstance(binding, dict):
            raise ValueError(f"plan runtime.{family} binding is missing")
        root = Path(str(binding.get("root"))).resolve(strict=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip().lower()
        if head != str(binding.get("git_head", "")).lower():
            raise ValueError(f"{family} Git HEAD differs from the frozen plan")
        files = []
        for relative, expected in binding.get("source_sha256", {}).items():
            target = (root / relative).resolve(strict=True)
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"{family} source path escapes the pinned root: {relative}") from exc
            digest = _sha_file(target)
            if digest != str(expected).lower():
                raise ValueError(f"{family} source SHA mismatch: {relative}")
            files.append({"path": relative, "bytes": target.stat().st_size, "sha256": digest})
        checkpoint = Path(str(binding.get("checkpoint"))).resolve(strict=True)
        checkpoint_sha = _sha_file(checkpoint)
        if checkpoint_sha != str(binding.get("checkpoint_sha256", "")).lower():
            raise ValueError(f"{family} checkpoint SHA mismatch")
        external_assets.append({
            "family": family, "root": str(root), "git_head": head, "sources": files,
            "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
        })

    context = runtime.get("context", {})
    context_npz = Path(str(context.get("npz_path"))).resolve(strict=True)
    context_json = Path(str(context.get("metadata_path"))).resolve(strict=True)
    context_npz_sha, context_json_sha = _sha_file(context_npz), _sha_file(context_json)
    if context_npz_sha != context.get("npz_sha256") or context_json_sha != context.get("metadata_sha256"):
        raise ValueError("fixed portrait context assets differ from the frozen plan")
    context_meta = json.loads(context_json.read_text(encoding="utf-8"))
    if context_meta.get("schema") != "cross_tfg_oldpilot_differentiable_context_v1" or context_meta.get("npz_file_sha256") != context_npz_sha:
        raise ValueError("fixed portrait context metadata/NPZ binding mismatch")

    audio_rows = []
    mfcc_count_records = []
    for row in rows:
        natural = row.get("natural_pcm")
        if not isinstance(natural, dict):
            raise ValueError(f"fit row lacks natural_pcm: {row.get('row_id')}")
        path = Path(str(natural.get("path")))
        if path.parent.resolve(strict=True) != audio_root:
            raise ValueError(f"fit WAV path escapes the frozen input directory: {row['row_id']}")
        raw_sha = _sha_file(path)
        if raw_sha != natural.get("file_sha256"):
            raise ValueError(f"fit WAV file SHA mismatch: {row['row_id']}")
        audio, info = read_canonical_wav(path)
        if (info.sample_rate != 16_000 or info.source_channels != 1 or info.frames != int(natural.get("sample_count", -1))
                or info.pcm_float32_sha256 != natural.get("pcm_sha256")):
            raise ValueError(f"fit WAV canonical float32 identity mismatch: {row['row_id']}")
        if _sha_bytes(_pcm16(audio).tobytes()) != natural.get("direct_source_pcm16_sha256"):
            raise ValueError(f"fit WAV direct PCM16 identity mismatch: {row['row_id']}")
        mfcc_frames, valid_windows, official_queries = _counts(int(audio.size))
        geometry = row.get("full_context_fit", {})
        expected = {
            "source_sample_count": int(audio.size),
            "psf_mfcc_frame_count": mfcc_frames,
            "all_valid_mfcc_window_count": valid_windows,
            "scoring_audio_embedding_count": official_queries,
            "valid_mfcc_windows_excluded_after_official_query_end": valid_windows - official_queries,
        }
        for name, value in expected.items():
            if geometry.get(name) != value:
                raise ValueError(f"frozen count metadata mismatch for {row['row_id']}.{name}")
        pcm16 = _pcm16(audio)
        features = np.asarray(mfcc(pcm16, samplerate=16_000), dtype=np.float32)
        starts_all = np.arange(0, features.shape[0] - MFCC_WINDOW + 1, MFCC_HOP, dtype=np.int32)
        if features.shape != (mfcc_frames, 13) or starts_all.size != valid_windows:
            raise ValueError(f"actual pinned PSF MFCC/window dimensions differ for {row['row_id']}")
        if official_queries < 57:
            raise ValueError(f"fit audio cannot provide a 27-query window with all 31 lags: {row['row_id']}")
        q0 = (official_queries - VIDEO_QUERIES) // 2
        if q0 - LAG_RADIUS < 0 or q0 + VIDEO_QUERIES - 1 + LAG_RADIUS >= official_queries:
            raise ValueError(f"center-32 query mapping leaves the official full-audio lag support: {row['row_id']}")
        audio_rows.append({
            "row_id": row["row_id"], "sample_id": row["sample_id"], "speaker_id": row["speaker_id"],
            "path": str(path), "file_bytes": path.stat().st_size, "file_sha256": raw_sha,
            "sample_count": int(audio.size), "pcm_float32_sha256": info.pcm_float32_sha256,
            "direct_pcm16_sha256": _sha_bytes(pcm16.tobytes()), "psf_mfcc_frames": mfcc_frames,
            "all_valid_windows": valid_windows, "official_syncnet_queries": official_queries,
            "excluded_valid_tail_windows": valid_windows - official_queries, "q0": q0,
            "all_31_lags_in_bounds": True,
        })
        mfcc_count_records.append(expected)

    try:
        gpu_snapshot = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=15,
        )
        gpu_text = gpu_snapshot.stdout.strip()
        gpu_exit = gpu_snapshot.returncode
    except Exception as exc:
        gpu_text, gpu_exit = str(exc), None

    payload = {
        "schema": "cross_tfg_audio_head_fullcontext_fit_no_forward_preflight_v1",
        "status": "FULLCONTEXT_PREFLIGHT_PASS_NO_FORWARD",
        "plan_path": str(plan_path), "plan_bytes": len(plan_raw), "plan_raw_sha256": _sha_bytes(plan_raw),
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "row_count": len(rows), "speaker_count": len(speaker_ids), "sample_count_total": sum(r["sample_count"] for r in audio_rows),
        "audio_rows": audio_rows, "psf_tree_sha256": tree_sha, "psf_source_files": psf_files,
        "package_source_files": source_rows, "external_model_assets": external_assets,
        "context_assets": {
            "npz_path": str(context_npz), "npz_bytes": context_npz.stat().st_size, "npz_sha256": context_npz_sha,
            "metadata_path": str(context_json), "metadata_bytes": context_json.stat().st_size, "metadata_sha256": context_json_sha,
            "frame_count": context_meta.get("frame_count"),
        },
        "execution_runtime": _version_report(execution),
        "gpu_snapshot": {"returncode": gpu_exit, "stdout": gpu_text},
        "forward_pass_performed": False,
        "optimizer_steps_performed": 0,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "effects_or_scores_read": False,
    }
    report_path = report_path.resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite preflight report {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = preflight(plan_path=args.plan, payload_root=args.payload_root, audio_root=args.audio_root, report_path=args.report)
    print(json.dumps({key: payload[key] for key in ("schema", "status", "plan_raw_sha256", "plan_payload_sha256", "row_count", "speaker_count", "forward_pass_performed", "optimizer_steps_performed")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
