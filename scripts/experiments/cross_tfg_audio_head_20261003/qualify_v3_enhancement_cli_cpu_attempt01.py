#!/usr/bin/env python3
"""Bounded CPU-only engineering qualification for the V3 enhancement CLI.

This worker is intentionally an engineering-zero check. It measures three
historical fit WAV duration buckets, verifies exact source preservation and
float-WAV round trips, and exercises the CLI's no-overwrite guard. It does not
read dev/confirmation audio, score generated media, or use trained parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Callable


TASK_ROOT = pathlib.Path(
    "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/"
    "v3_enhancement_cli_cpu_qualification_attempt01"
)
INPUT_ROOT = pathlib.Path(
    "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/"
    "v5_fit40_input_stage_attempt03/fit40_source_inputs"
)
MANIFEST_PATH = INPUT_ROOT / "fit40_source_transfer_manifest.json"
REGISTRY_PATH = INPUT_ROOT / "source_binding_registry.json"
MANIFEST_BYTES = 61469
MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
REGISTRY_BYTES = 55093
REGISTRY_SHA256 = "0a98de21f59c24d82a7a3cfe5ca21bcef82c2bbbae632134b2000ced6f309ec4"
MANIFEST_SCHEMA = "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1"
REGISTRY_SCHEMA = "cross_tfg_v5_fit40_source_binding_registry_v1"
STAGE_STATUS = "PASS_SOURCE_ONLY_FIT40_CANONICAL_PCM16_STAGED"
PYTHON = pathlib.Path("/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python")
CONFIG_REL = "scripts/audio_enhancement/configs/v3_zero_engineering_attempt01.json"
CLI_REL = "scripts/audio_enhancement/enhance_v3_attempt01.py"
WORKER_SCHEMA = "cross_tfg_v3_enhancement_cli_cpu_qualification_v1"
MAX_WALL_SECONDS = 300.0
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024
THREADS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
FIXTURES = {
    "short": "S0065_j00_BAC009S0065W0121",
    "middle": "S0069_j00_BAC009S0069W0121",
    "long": "S0064_j15_BAC009S0064W0495",
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: pathlib.Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "device": int(st.st_dev),
        "inode": int(st.st_ino),
        "bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def read_pinned_json(path: pathlib.Path, expected_bytes: int, expected_sha: str, label: str) -> tuple[dict[str, Any], bytes]:
    before = file_identity(path)
    raw = path.read_bytes()
    after = file_identity(path)
    if before != after:
        raise RuntimeError(f"{label} changed during read")
    if len(raw) != expected_bytes or sha_bytes(raw) != expected_sha:
        raise RuntimeError(f"{label} byte/SHA binding mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value, raw


def verify_fit_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, manifest_raw = read_pinned_json(
        MANIFEST_PATH, MANIFEST_BYTES, MANIFEST_SHA256, "fit40 transfer manifest"
    )
    registry, registry_raw = read_pinned_json(
        REGISTRY_PATH, REGISTRY_BYTES, REGISTRY_SHA256, "fit40 source binding registry"
    )
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise RuntimeError("fit40 source manifest schema/status mismatch")
    if manifest.get("split") != "historical_fit_only" or len(manifest.get("rows", [])) != 40:
        raise RuntimeError("fit40 source manifest is not the sealed 40-row historical-fit input")
    r090_manifest = manifest.get("r090_control")
    if not isinstance(r090_manifest, dict) or r090_manifest.get("status") != "PREPARED_NOT_FROZEN":
        raise RuntimeError("fit40 manifest r090 control must remain prepared and unfrozen")
    r090_config = r090_manifest.get("config")
    if not isinstance(r090_config, dict) or (
        r090_config.get("selection_status") != "PREPARED_NOT_FROZEN"
        or r090_config.get("effect_execution_status") != "BLOCKED_UNTIL_SHARED540_AND_METHOD_FREEZE"
    ):
        raise RuntimeError("fit40 r090 nested selection/effect state differs from the frozen manifest")
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("status") != "SEALED_SOURCE_ONLY_TRANSFER_READY_NOT_EXECUTABLE":
        raise RuntimeError("fit40 source registry schema/status mismatch")
    if len(registry.get("source_rows", [])) != 40:
        raise RuntimeError("fit40 registry does not contain exactly 40 source rows")
    registry_manifest_pin = registry.get("source_manifest")
    if not isinstance(registry_manifest_pin, dict) or (
        registry_manifest_pin.get("bytes") != MANIFEST_BYTES
        or registry_manifest_pin.get("raw_sha256") != MANIFEST_SHA256
    ):
        raise RuntimeError("fit40 registry does not bind the sealed manifest bytes/SHA")
    r090 = registry.get("r090_status")
    if not isinstance(r090, dict) or r090.get("status") != "PREPARED_NOT_FROZEN":
        raise RuntimeError("fit40 r090 control must remain prepared and unfrozen")
    if registry.get("remote_stage", {}).get("plan_is_executable") is not False:
        raise RuntimeError("fit40 source registry unexpectedly marks an executable plan")

    manifest_rows = manifest["rows"]
    registry_rows = registry["source_rows"]
    m_by_id = {row.get("row_id"): row for row in manifest_rows}
    r_by_id = {row.get("row_id"): row for row in registry_rows}
    if None in m_by_id or len(m_by_id) != 40 or set(m_by_id) != set(r_by_id) or len(r_by_id) != 40:
        raise RuntimeError("fit40 row identifiers differ between transfer manifest and source registry")
    for row_id, mrow in m_by_id.items():
        rrow = r_by_id[row_id]
        pairs = (
            (mrow.get("sample_id"), rrow.get("sample_id")),
            (mrow.get("transfer_relative_path"), rrow.get("package_path")),
            (mrow.get("transfer_file_bytes"), rrow.get("file_bytes")),
            (mrow.get("transfer_file_sha256"), rrow.get("file_sha256")),
            (mrow.get("canonical_float32_pcm_sha256"), rrow.get("canonical_float32_pcm_sha256")),
            (mrow.get("sample_count"), rrow.get("sample_count")),
            (mrow.get("sample_rate_hz"), rrow.get("sample_rate_hz")),
            (mrow.get("channels"), rrow.get("channels")),
        )
        if any(left != right for left, right in pairs):
            raise RuntimeError(f"fit40 manifest/registry mismatch for {row_id}")
        if (mrow.get("sample_rate_hz"), mrow.get("channels"), mrow.get("source_encoding")) != (16000, 1, "PCM_16"):
            raise RuntimeError(f"fit40 source contract mismatch for {row_id}")

    sorted_rows = sorted(manifest_rows, key=lambda row: (int(row["sample_count"]), row["row_id"]))
    expected_duration_fixtures = {
        "short": sorted_rows[0]["row_id"],
        "middle": sorted_rows[20]["row_id"],
        "long": sorted_rows[-1]["row_id"],
    }
    if expected_duration_fixtures != FIXTURES:
        raise RuntimeError(f"duration fixture selection changed: {expected_duration_fixtures}")

    selected = []
    for bucket, row_id in FIXTURES.items():
        mrow = m_by_id[row_id]
        rrow = r_by_id[row_id]
        rel = pathlib.PurePosixPath(mrow["transfer_relative_path"])
        if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
            raise RuntimeError(f"unsafe source transfer path for {row_id}")
        path = INPUT_ROOT.joinpath(*rel.parts)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(INPUT_ROOT.resolve(strict=True)):
            raise RuntimeError(f"source WAV escaped staged input root: {row_id}")
        cursor = INPUT_ROOT
        for part in rel.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise RuntimeError(f"symlink component in staged source WAV path: {row_id}")
        st = path.lstat()
        if not path.is_file() or path.is_symlink() or st.st_size != int(mrow["transfer_file_bytes"]):
            raise RuntimeError(f"staged WAV type/size mismatch for {row_id}")
        raw_sha_before = sha_file(path)
        if raw_sha_before != mrow["transfer_file_sha256"]:
            raise RuntimeError(f"staged WAV raw SHA differs from fit40 source binding before CLI: {row_id}")
        selected.append(
            {
                "bucket": bucket,
                "row_id": row_id,
                "sample_id": mrow["sample_id"],
                "path": str(path),
                "relative_path": rel.as_posix(),
                "sample_count": int(mrow["sample_count"]),
                "sample_rate_hz": int(mrow["sample_rate_hz"]),
                "channels": int(mrow["channels"]),
                "source_encoding": mrow["source_encoding"],
                "file_bytes": int(mrow["transfer_file_bytes"]),
                "file_sha256": mrow["transfer_file_sha256"],
                "canonical_float32_pcm_sha256": mrow["canonical_float32_pcm_sha256"],
                "pcm16_payload_sha256": rrow["pcm16_payload_sha256"],
                "before": file_identity(path),
                "raw_file_sha256_before": raw_sha_before,
            }
        )
    bindings = {
        "manifest": {"path": str(MANIFEST_PATH), "bytes": len(manifest_raw), "sha256": sha_bytes(manifest_raw)},
        "registry": {"path": str(REGISTRY_PATH), "bytes": len(registry_raw), "sha256": sha_bytes(registry_raw)},
        "rows_total": len(manifest_rows),
        "duration_buckets": selected,
    }
    return selected, bindings


def load_cli_and_runtime(code_root: pathlib.Path) -> tuple[Any, Any, Any, Any, dict[str, Any], dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name, value in THREADS.items():
        os.environ[name] = value
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = ""
    os.environ.pop("PYTHONHOME", None)
    sys.dont_write_bytecode = True
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU qualification requires CUDA_VISIBLE_DEVICES to be empty")
    if sys.version_info[:3] != (3, 10, 12):
        raise RuntimeError(f"Python version mismatch: {sys.version_info[:3]}")
    if not PYTHON.is_file():
        raise RuntimeError(f"pinned interpreter file is missing: {PYTHON}")

    cli_path = code_root / CLI_REL
    spec = importlib.util.spec_from_file_location("pinned_v3_cli_cpu_qualification", cli_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned V3 CLI module")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config_path = code_root / CONFIG_REL
    config, _config_raw, pins = cli._validate_config(config_path)
    import numpy as np
    import torch
    if np.__version__ != "1.26.4" or str(torch.__version__) != "2.5.1+cu124":
        raise RuntimeError(f"unexpected CPU runtime NumPy={np.__version__} Torch={torch.__version__}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU qualification")
    device = cli._validate_runtime(config["runtime"], torch, np)
    if device != "cpu" or torch.cuda.is_initialized():
        raise RuntimeError("V3 engineering profile did not remain CPU-only")
    imported = cli._verify_imported_sources(pins)
    head_module = __import__("scripts.audio_enhancement.shared_recipient_transition_head_v3", fromlist=["SharedRecipientTransitionHeadV3"])
    head_class = getattr(head_module, "SharedRecipientTransitionHeadV3")
    cli._constructor_fields(head_class, config["constructor"])
    from scripts.audio_enhancement.audio_io import (
        CANONICAL_SAMPLE_RATE,
        pcm_float32_sha256,
        read_canonical_wav,
        write_float32_wav,
    )
    from scripts.audio_enhancement.baselines import apply_global_rms, waveform_rms
    return cli, np, torch, (head_class, read_canonical_wav, write_float32_wav, pcm_float32_sha256, apply_global_rms, waveform_rms, CANONICAL_SAMPLE_RATE), config, imported


def child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(THREADS)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = ""
    env.pop("PYTHONHOME", None)
    return env


def run_cold_cli(
    code_root: pathlib.Path,
    input_row: dict[str, Any],
    sample_output: pathlib.Path,
    config_path: pathlib.Path,
    *,
    deadline: float,
) -> dict[str, Any]:
    cli_path = code_root / CLI_REL
    output_path = sample_output / "enhanced.wav"
    metadata_path = sample_output / "metadata.json"
    sample_output.mkdir(parents=True, exist_ok=False)
    command = [
        str(PYTHON), "-B", str(cli_path),
        "--input", input_row["path"],
        "--output", str(output_path),
        "--config", str(config_path),
        "--metadata", str(metadata_path),
    ]
    remaining = deadline - time.monotonic()
    if remaining <= 1.0:
        raise TimeoutError("qualification wall-time budget exhausted before cold CLI invocation")
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=str(code_root),
        env=child_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=min(90.0, remaining),
        check=False,
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"cold CLI failed for {input_row['row_id']} rc={completed.returncode}: "
            f"stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-4000:]!r}"
        )
    try:
        child_summary = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
        metadata_raw = metadata_path.read_bytes()
        metadata = json.loads(metadata_raw)
    except Exception as exc:
        raise RuntimeError(f"cold CLI output/metadata is invalid for {input_row['row_id']}: {exc}") from exc
    if child_summary.get("schema") != "cross_tfg_v3_enhancement_cli_result_v1" or child_summary.get("status") != "PASS":
        raise RuntimeError(f"cold CLI summary did not pass for {input_row['row_id']}")
    if pathlib.Path(child_summary.get("metadata_path", "")).resolve() != metadata_path.resolve():
        raise RuntimeError(f"cold CLI summary metadata path mismatch for {input_row['row_id']}")
    if metadata.get("schema") != "cross_tfg_v3_enhancement_cli_result_v1" or metadata.get("status") != "PASS":
        raise RuntimeError(f"cold CLI sidecar did not pass for {input_row['row_id']}")
    if metadata.get("profile_role") != "ENGINEERING_ONLY" or metadata.get("head", {}).get("candidate_provenance") is not None:
        raise RuntimeError(f"cold CLI sidecar is not explicitly engineering-only for {input_row['row_id']}")
    if metadata.get("runtime", {}).get("device") != "cpu" or metadata.get("runtime", {}).get("cuda_initialized") is not False:
        raise RuntimeError(f"cold CLI did not prove CPU-only runtime for {input_row['row_id']}")
    theta = metadata.get("head", {}).get("parameter", {}).get("values")
    if theta != [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]:
        raise RuntimeError(f"cold CLI parameter values were not exact engineering zero for {input_row['row_id']}")
    source_audio = metadata.get("source_audio", {})
    canonical_input = metadata.get("canonical_input_pcm", {})
    if source_audio.get("file_sha256") != input_row["file_sha256"]:
        raise RuntimeError(f"CLI did not bind the expected source WAV bytes for {input_row['row_id']}")
    if canonical_input.get("pcm_sha256") != input_row["canonical_float32_pcm_sha256"]:
        raise RuntimeError(f"CLI did not bind the expected canonical input PCM for {input_row['row_id']}")
    if source_audio.get("sample_count") != input_row["sample_count"] or canonical_input.get("sample_count") != input_row["sample_count"]:
        raise RuntimeError(f"CLI input clock/count differs from the fit40 source row for {input_row['row_id']}")
    return {
        "row_id": input_row["row_id"],
        "sample_id": input_row["sample_id"],
        "duration_bucket": input_row["bucket"],
        "sample_count": input_row["sample_count"],
        "duration_seconds": input_row["sample_count"] / 16000.0,
        "input_wav_path": input_row["path"],
        "input_wav_bytes": input_row["file_bytes"],
        "input_wav_sha256_before": input_row["raw_file_sha256_before"],
        "source_file_sha256_as_read_by_cli": source_audio["file_sha256"],
        "canonical_float32_pcm_sha256_expected": input_row["canonical_float32_pcm_sha256"],
        "canonical_float32_pcm_sha256_as_read_by_cli": canonical_input["pcm_sha256"],
        "cli_summary": child_summary,
        "cli_metadata_path": str(metadata_path),
        "cli_metadata_bytes": len(metadata_raw),
        "cli_metadata_sha256": sha_bytes(metadata_raw),
        "cli_metadata": metadata,
        "cold_subprocess_wall_ms": elapsed_ms,
        "cold_subprocess_real_time_factor": elapsed_ms / (1000.0 * input_row["sample_count"] / 16000.0),
        "stdout_bytes": len(completed.stdout),
        "stderr_bytes": len(completed.stderr),
        "output_wav_path": str(output_path),
    }


def measure(fn: Callable[[], Any], *, repeats: int = 3, deadline: float) -> dict[str, Any]:
    fn()  # one explicit warm-up, excluded from timing
    values = []
    for _ in range(repeats):
        if time.monotonic() >= deadline:
            raise TimeoutError("qualification wall-time budget exhausted during component timings")
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    ordered = sorted(values)
    return {
        "warmup_calls_excluded": 1,
        "measured_calls": repeats,
        "samples_ms": values,
        "median_ms": ordered[len(ordered) // 2],
        "p90_ms_nearest_rank": ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
    }


def warm_component_timings(
    *,
    selected: list[dict[str, Any]],
    cold_results: list[dict[str, Any]],
    output_root: pathlib.Path,
    code_root: pathlib.Path,
    cli: Any,
    np: Any,
    torch: Any,
    imports: tuple[Any, ...],
    config: dict[str, Any],
    deadline: float,
) -> list[dict[str, Any]]:
    head_class, read_canonical_wav, write_float32_wav, pcm_float32_sha256, apply_global_rms, waveform_rms, rate = imports
    if rate != 16000:
        raise RuntimeError("canonical sample rate changed unexpectedly")
    timings = []
    warm_root = output_root / "warm_component_io"
    warm_root.mkdir(parents=True, exist_ok=False)
    for index, (row, cold) in enumerate(zip(selected, cold_results)):
        source_path = pathlib.Path(row["path"])
        source, read_info = read_canonical_wav(source_path, target_rate=rate)
        baseline, baseline_info = apply_global_rms(
            source,
            target_rms=float(config["constructor"]["target_rms"]),
            peak_ceiling=config["constructor"]["peak_ceiling"],
            silence_epsilon=float(config["constructor"]["rms_epsilon"]),
        )
        output_path = pathlib.Path(cold["output_wav_path"])
        enhanced, output_info = read_canonical_wav(output_path, target_rate=rate)
        if read_info.file_sha256 != row["file_sha256"] or read_info.pcm_float32_sha256 != row["canonical_float32_pcm_sha256"]:
            raise RuntimeError(f"canonical decoder disagrees with pinned input for {row['row_id']}")
        if enhanced.shape != baseline.shape or not np.array_equal(enhanced, baseline):
            raise RuntimeError(f"engineering-zero CLI output differs from pinned RMS baseline for {row['row_id']}")
        if output_info.source_encoding != "FLOAT_32" or output_info.sample_rate != 16000 or output_info.source_channels != 1:
            raise RuntimeError(f"CLI output encoding/rate/channel mismatch for {row['row_id']}")
        if output_info.pcm_float32_sha256 != pcm_float32_sha256(baseline):
            raise RuntimeError(f"CLI FLOAT PCM readback hash differs from RMS baseline for {row['row_id']}")

        source_tensor = torch.from_numpy(np.ascontiguousarray(source)).to(device="cpu", dtype=torch.float32)
        baseline_tensor = torch.from_numpy(np.ascontiguousarray(baseline)).to(device="cpu", dtype=torch.float32)
        head = head_class(**config["constructor"]).to("cpu")
        if not bool(torch.equal(head.band_contrasts_db.detach(), torch.zeros((3, 3), dtype=torch.float32))):
            raise RuntimeError("new engineering head did not start at exact zero theta")
        head.eval()

        def decode_component() -> None:
            decoded, info = read_canonical_wav(source_path, target_rate=rate)
            if info.file_sha256 != row["file_sha256"] or info.pcm_float32_sha256 != row["canonical_float32_pcm_sha256"]:
                raise RuntimeError("warm decoder source binding changed")
            if decoded.shape != source.shape:
                raise RuntimeError("warm decoder changed exact sample count")

        def rms_component() -> None:
            current, _ = apply_global_rms(
                source,
                target_rms=float(config["constructor"]["target_rms"]),
                peak_ceiling=config["constructor"]["peak_ceiling"],
                silence_epsilon=float(config["constructor"]["rms_epsilon"]),
            )
            if not np.array_equal(current, baseline):
                raise RuntimeError("warm RMS output changed between calls")

        def constructor_component() -> None:
            built = head_class(**config["constructor"]).to("cpu")
            if tuple(built.band_contrasts_db.shape) != (3, 3):
                raise RuntimeError("warm head constructor produced wrong parameter shape")
            del built

        def resident_forward() -> None:
            with torch.inference_mode():
                result = head(source_tensor, baseline_tensor)
            output = result.get("output_pcm_float32")
            if not isinstance(output, torch.Tensor) or output.dtype != torch.float32 or output.shape != source_tensor.shape:
                raise RuntimeError("resident head forward violated finite float32 exact-length shape")
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("resident head forward returned non-finite samples")
            if not torch.equal(output, baseline_tensor):
                raise RuntimeError("resident engineering-zero forward differs from canonical RMS baseline")

        suffix = f"{index:02d}"
        wav_counter = [0]

        def wav_roundtrip() -> None:
            target = warm_root / f"{suffix}_float_{wav_counter[0]:02d}.wav"
            wav_counter[0] += 1
            written = write_float32_wav(target, enhanced, sample_rate=rate, peak_ceiling=None, overwrite=False)
            decoded, info = read_canonical_wav(target, target_rate=rate)
            if not np.array_equal(decoded, enhanced) or written.pcm_float32_sha256 != pcm_float32_sha256(enhanced):
                raise RuntimeError("warm float32 WAV roundtrip was not bit-identical")
            if info.source_encoding != "FLOAT_32" or info.pcm_float32_sha256 != written.pcm_float32_sha256:
                raise RuntimeError("warm float32 WAV metadata/readback mismatch")
            target.unlink()

        metadata_counter = [0]
        metadata_path = pathlib.Path(cold["cli_metadata_path"])
        metadata_obj = json.loads(metadata_path.read_bytes())

        def metadata_write() -> None:
            target = warm_root / f"{suffix}_metadata_{metadata_counter[0]:02d}.json"
            metadata_counter[0] += 1
            raw = json.dumps(metadata_obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
            cli._write_metadata_exclusive(target, raw)
            if target.read_bytes() != raw:
                raise RuntimeError("warm metadata write changed serialized bytes")
            target.unlink()

        with torch.inference_mode():
            warm_result = head(source_tensor, baseline_tensor)
        if not torch.equal(warm_result["output_pcm_float32"], baseline_tensor):
            raise RuntimeError("warm-up head call did not preserve zero-theta baseline")

        functions = {
            "canonical_wav_decode": decode_component,
            "rms_baseline": rms_component,
            "head_constructor": constructor_component,
            "resident_head_forward": resident_forward,
            "float32_wav_write_exact_readback": wav_roundtrip,
            "metadata_serialize_and_exclusive_write": metadata_write,
        }
        stage_timings = {name: measure(fn, repeats=3, deadline=deadline) for name, fn in functions.items()}
        if torch.cuda.is_initialized():
            raise RuntimeError("warm CPU timing initialized CUDA")
        if file_identity(source_path) != row["before"] or sha_file(source_path) != row["file_sha256"]:
            raise RuntimeError(f"original input changed during warm timings for {row['row_id']}")
        duration_ms = row["sample_count"] * 1000.0 / rate
        for item in stage_timings.values():
            item["median_real_time_factor"] = item["median_ms"] / duration_ms
            item["p90_real_time_factor"] = item["p90_ms_nearest_rank"] / duration_ms
        timings.append(
            {
                "row_id": row["row_id"],
                "duration_bucket": row["bucket"],
                "sample_count": row["sample_count"],
                "duration_seconds": row["sample_count"] / rate,
                "rms_baseline": {
                    "input_rms": baseline_info.input_rms,
                    "target_rms": baseline_info.target_rms,
                    "requested_gain": baseline_info.requested_gain,
                    "applied_gain": baseline_info.applied_gain,
                    "output_rms": baseline_info.output_rms,
                    "peak_limited": baseline_info.peak_limited,
                    "output_pcm_float32_sha256": pcm_float32_sha256(baseline),
                    "measured_output_rms": waveform_rms(enhanced),
                },
                "components": stage_timings,
            }
        )
        del head, source_tensor, baseline_tensor
    if any(path.is_file() for path in warm_root.iterdir()):
        raise RuntimeError("temporary warm component files were not fully removed")
    return timings


def output_bytes(root: pathlib.Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"unexpected symlink in qualification output: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def rss_bytes(who: int) -> int:
    # Linux ru_maxrss is KiB; this worker is bound to the Linux 3.10 runtime.
    return int(resource.getrusage(who).ru_maxrss * 1024)


def write_json_exclusive(path: pathlib.Path, value: dict[str, Any]) -> tuple[int, str]:
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return len(raw), sha_bytes(raw)


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS
    code_root = pathlib.Path(args.code_root).resolve(strict=True)
    output_root = pathlib.Path(args.output_root)
    if code_root != TASK_ROOT / "code":
        raise RuntimeError(f"unexpected code root: {code_root}")
    if pathlib.Path(args.input_root).resolve(strict=True) != INPUT_ROOT.resolve(strict=True):
        raise RuntimeError("input root differs from the qualified fit40 stage")
    if pathlib.Path(args.manifest).resolve(strict=True) != MANIFEST_PATH.resolve(strict=True):
        raise RuntimeError("manifest path differs from the qualified fit40 source stage")
    if pathlib.Path(args.registry).resolve(strict=True) != REGISTRY_PATH.resolve(strict=True):
        raise RuntimeError("registry path differs from the qualified fit40 source stage")
    if pathlib.Path(args.config).resolve(strict=True) != (code_root / CONFIG_REL).resolve(strict=True):
        raise RuntimeError("engineering-zero config path mismatch")
    if output_root != TASK_ROOT / "output" / "qualification_attempt01":
        raise RuntimeError("unexpected or reused qualification output path")
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("fresh qualification output path already exists")
    selected, bindings = verify_fit_rows()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    cold_results = []
    try:
        config_path = code_root / CONFIG_REL
        for row in selected:
            if time.monotonic() >= deadline:
                raise TimeoutError("qualification wall-time budget exhausted before cold CLI runs")
            cold = run_cold_cli(
                code_root,
                row,
                output_root / "samples" / row["row_id"],
                config_path,
                deadline=deadline,
            )
            source_path = pathlib.Path(row["path"])
            reread, info = None, None
            # Independent parent-process readback and RMS computation below are
            # performed after the one-time runtime/thread initialization.
            if file_identity(source_path) != row["before"] or sha_file(source_path) != row["file_sha256"]:
                raise RuntimeError(f"original input changed during cold CLI run for {row['row_id']}")
            cold_results.append(cold)

        cli, np, torch, imports, config, imported_sources = load_cli_and_runtime(code_root)
        if torch.cuda.is_initialized():
            raise RuntimeError("worker CPU runtime initialized CUDA before measurements")
        warm_timings = warm_component_timings(
            selected=selected,
            cold_results=cold_results,
            output_root=output_root,
            code_root=code_root,
            cli=cli,
            np=np,
            torch=torch,
            imports=imports,
            config=config,
            deadline=deadline,
        )

        # A second invocation against an existing output must fail closed and
        # leave both sealed files untouched.
        first = cold_results[0]
        row0 = selected[0]
        guard_output = pathlib.Path(first["output_wav_path"])
        guard_metadata = pathlib.Path(first["cli_metadata_path"])
        output_before = {"identity": file_identity(guard_output), "sha256": sha_file(guard_output)}
        metadata_before = {"identity": file_identity(guard_metadata), "sha256": sha_file(guard_metadata)}
        cmd = [
            str(PYTHON), "-B", str(code_root / CLI_REL),
            "--input", row0["path"], "--output", str(guard_output),
            "--config", str(code_root / CONFIG_REL), "--metadata", str(guard_metadata),
        ]
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            raise TimeoutError("qualification wall-time budget exhausted before overwrite guard")
        guard_start = time.perf_counter_ns()
        guard = subprocess.run(
            cmd,
            cwd=str(code_root),
            env=child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(60.0, remaining),
            check=False,
        )
        guard_ms = (time.perf_counter_ns() - guard_start) / 1_000_000.0
        try:
            guard_error = json.loads(guard.stderr.decode("utf-8").strip().splitlines()[-1])
        except Exception as exc:
            raise RuntimeError(f"overwrite rejection was not machine-readable: {exc}") from exc
        if guard.returncode != 2 or guard_error.get("status") != "FAILED" or "refusing to overwrite" not in guard_error.get("error", ""):
            raise RuntimeError(f"CLI overwrite guard did not reject existing outputs: rc={guard.returncode} stderr={guard.stderr[-2000:]!r}")
        output_after = {"identity": file_identity(guard_output), "sha256": sha_file(guard_output)}
        metadata_after = {"identity": file_identity(guard_metadata), "sha256": sha_file(guard_metadata)}
        if output_before != output_after or metadata_before != metadata_after:
            raise RuntimeError("CLI overwrite guard changed an existing output or metadata sidecar")

        audio_module = sys.modules["scripts.audio_enhancement.audio_io"]
        baseline_module = sys.modules["scripts.audio_enhancement.baselines"]
        for row, cold in zip(selected, cold_results):
            original_path = pathlib.Path(row["path"])
            original_samples, input_info = audio_module.read_canonical_wav(original_path, target_rate=16000)
            independent_baseline, baseline_info = baseline_module.apply_global_rms(
                original_samples,
                target_rms=float(config["constructor"]["target_rms"]),
                peak_ceiling=config["constructor"]["peak_ceiling"],
                silence_epsilon=float(config["constructor"]["rms_epsilon"]),
            )
            output_samples, output_info = audio_module.read_canonical_wav(cold["output_wav_path"], target_rate=16000)
            if input_info.file_sha256 != row["file_sha256"] or input_info.pcm_float32_sha256 != row["canonical_float32_pcm_sha256"]:
                raise RuntimeError(f"input WAV/F32 PCM differs from frozen fit40 binding for {row['row_id']}")
            if input_info.source_encoding != "PCM_16" or input_info.source_sample_rate != 16000 or input_info.source_channels != 1:
                raise RuntimeError(f"source WAV encoding/rate/channel mismatch for {row['row_id']}")
            if original_samples.size != row["sample_count"] or not np.array_equal(output_samples, independent_baseline):
                raise RuntimeError(f"independent baseline/output exact comparison failed for {row['row_id']}")
            if output_info.source_encoding != "FLOAT_32" or output_info.source_sample_rate != 16000 or output_info.source_channels != 1:
                raise RuntimeError(f"output WAV is not exact IEEE float32 mono 16k for {row['row_id']}")
            if output_info.frames != row["sample_count"] or output_info.pcm_float32_sha256 != audio_module.pcm_float32_sha256(independent_baseline):
                raise RuntimeError(f"output FLOAT WAV exact count/PCM hash mismatch for {row['row_id']}")
            now = file_identity(original_path)
            if now != row["before"] or sha_file(original_path) != row["file_sha256"]:
                raise RuntimeError(f"original fit PCM16 input changed across qualification for {row['row_id']}")
            cold["independent_check"] = {
                "source_file_sha256_before_after_identical": True,
                "source_file_sha256_before": row["raw_file_sha256_before"],
                "source_file_sha256_after": sha_file(original_path),
                "source_pcm16_file_bytes": row["file_bytes"],
                "source_encoding": input_info.source_encoding,
                "source_sample_rate_hz": input_info.source_sample_rate,
                "source_channels": input_info.source_channels,
                "source_sample_count": input_info.source_frames,
                "source_canonical_pcm_float32_sha256": input_info.pcm_float32_sha256,
                "source_canonical_pcm_float32_sha256_as_read_by_cli": cold["canonical_float32_pcm_sha256_as_read_by_cli"],
                "source_canonical_pcm_float32_sha256_after": input_info.pcm_float32_sha256,
                "output_encoding": output_info.source_encoding,
                "output_sample_rate_hz": output_info.source_sample_rate,
                "output_channels": output_info.source_channels,
                "output_sample_count": output_info.frames,
                "output_file_sha256": output_info.file_sha256,
                "output_pcm_float32_sha256": output_info.pcm_float32_sha256,
                "independent_rms_baseline_bit_identical": True,
                "rms_baseline": {
                    "input_rms": baseline_info.input_rms,
                    "target_rms": baseline_info.target_rms,
                    "requested_gain": baseline_info.requested_gain,
                    "applied_gain": baseline_info.applied_gain,
                    "output_rms": baseline_info.output_rms,
                    "peak_limited": baseline_info.peak_limited,
                },
            }

        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized during CPU engineering qualification")
        self_peak = rss_bytes(resource.RUSAGE_SELF)
        child_peak = rss_bytes(resource.RUSAGE_CHILDREN)
        combined_peak = self_peak + child_peak
        if combined_peak > MAX_RSS_BYTES:
            raise RuntimeError(f"worker+child peak RSS exceeded 2 GiB: {combined_peak} bytes")
        elapsed = time.monotonic() - started
        if elapsed > MAX_WALL_SECONDS:
            raise TimeoutError(f"worker exceeded {MAX_WALL_SECONDS:.0f} second wall-time bound")
        report = {
            "schema": WORKER_SCHEMA,
            "status": "PASS_ENGINEERING_ONLY_V3_CLI_CPU",
            "scope": {
                "historical_fit_only": True,
                "duration_bucket_rows": 3,
                "fit40_rows_opened": 3,
                "dev_audio_read": False,
                "confirm_audio_read": False,
                "candidate_effect_read": False,
                "trained_parameter_checkpoint_used": False,
                "engineering_zero_theta": True,
                "external_tfg_model_api": False,
                "v3_head_cpu_forward": True,
                "gpu_inference": False,
                "cuda_initialized": False,
                "scoring_or_rendering": False,
            },
            "worker_source": {
                "path": str(pathlib.Path(__file__).resolve()),
                "bytes": pathlib.Path(__file__).stat().st_size,
                "sha256": sha_file(pathlib.Path(__file__).resolve()),
            },
            "source_stage": {
                "status": STAGE_STATUS,
                "input_root": str(INPUT_ROOT),
                "manifest": bindings["manifest"],
                "registry": bindings["registry"],
                "fit_rows_in_manifest": bindings["rows_total"],
                "selected_duration_buckets": [
                    {key: row[key] for key in ("bucket", "row_id", "sample_id", "sample_count", "sample_rate_hz", "channels", "file_bytes", "file_sha256", "canonical_float32_pcm_sha256")}
                    for row in selected
                ],
            },
            "profile": {
                "role": config["profile_role"],
                "config_path": str((code_root / CONFIG_REL).resolve()),
                "config_bytes": (code_root / CONFIG_REL).stat().st_size,
                "config_sha256": sha_file(code_root / CONFIG_REL),
                "parameter_name": config["parameter"]["name"],
                "parameter_shape": config["parameter"]["shape"],
                "parameter_values": config["parameter"]["values"],
                "constructor": config["constructor"],
                "runtime": config["runtime"],
                "verified_import_sources": imported_sources,
            },
            "cold_cli_runs": cold_results,
            "overwrite_guard": {
                "status": "PASS_REJECTED_WITHOUT_MUTATION",
                "returncode": guard.returncode,
                "wall_ms": guard_ms,
                "stderr_summary": guard_error,
                "stdout_bytes": len(guard.stdout),
                "output_unchanged": output_before == output_after,
                "metadata_unchanged": metadata_before == metadata_after,
                "output_sha256": output_after["sha256"],
                "metadata_sha256": metadata_after["sha256"],
            },
            "warm_component_timings": warm_timings,
            "resource_bound": {
                "wall_seconds": elapsed,
                "wall_seconds_limit": MAX_WALL_SECONDS,
                "self_peak_rss_bytes": self_peak,
                "max_child_peak_rss_bytes": child_peak,
                "conservative_sum_peak_rss_bytes": combined_peak,
                "peak_rss_limit_bytes": MAX_RSS_BYTES,
                "torch_intraop_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
            },
        }
        report_path = output_root / "qualification_report.json"
        report_raw = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        current = output_bytes(output_root)
        if current + len(report_raw) > MAX_OUTPUT_BYTES:
            raise RuntimeError(f"qualification outputs exceed 16 MiB: current={current} report={len(report_raw)}")
        with report_path.open("xb") as handle:
            handle.write(report_raw)
            handle.flush()
            os.fsync(handle.fileno())
        report["report_file"] = {"path": str(report_path), "bytes": len(report_raw), "sha256": sha_bytes(report_raw)}
        # The returned summary is stdout only; the persisted report remains the
        # exactly sealed object before this convenience field is added.
        report["output_tree_bytes"] = output_bytes(output_root)
        if report["output_tree_bytes"] > MAX_OUTPUT_BYTES:
            raise RuntimeError("final qualification output tree exceeded 16 MiB")
        return report
    except Exception:
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = pathlib.Path(args.output_root)
    try:
        report = run_qualification(args)
        print(json.dumps({
            "schema": WORKER_SCHEMA,
            "status": report["status"],
            "report_file": report["report_file"],
            "output_tree_bytes": report["output_tree_bytes"],
            "wall_seconds": report["resource_bound"]["wall_seconds"],
            "torch_cuda_initialized": report["resource_bound"]["torch_cuda_initialized"],
        }, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as exc:
        failure = {
            "schema": WORKER_SCHEMA,
            "status": "FAILED_ENGINEERING_QUALIFICATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }
        try:
            if output_root.parent.exists() and not output_root.exists():
                output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
            if output_root.is_dir() and not output_root.is_symlink():
                write_json_exclusive(output_root / "failure.json", failure)
        except Exception as write_exc:
            print(f"failure receipt write also failed: {type(write_exc).__name__}: {write_exc}", file=sys.stderr)
        print(json.dumps(failure, sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
