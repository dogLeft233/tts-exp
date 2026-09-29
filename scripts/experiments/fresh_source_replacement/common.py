"""Contracts and small, dependency-light helpers for the fresh-source A route.

The public experiment runner deliberately keeps the orchestration separate from
the numerical endpoint.  This module contains only I/O/contract code and the
frozen arithmetic that is useful to both the producer and the independent
validator.  In particular, no function here decides whether a result is a
replacement signal.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.fresh_source_inputs.protocol import (
    canonical_hash,
    file_sha256,
    load_self_hashed,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FPS = 25
SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
VIDEO_FRAMES = 140
SEEDS = (42, 43)
BOOTSTRAP_SEED = 2_026_0910
BOOTSTRAP_DRAWS = 20_000
EMBEDDING_DIM = 1_024
EPSILON = 1e-6
U_START = 25
U_STOP = 115  # exclusive: range(25, 115)
LAG_MIN = -15
LAG_MAX = 15
LAG_COUNT = LAG_MAX - LAG_MIN + 1
SHIFT_FRAMES = 5
SHIFT_SAMPLES = SHIFT_FRAMES * SAMPLES_PER_FRAME
WAV2LIP_CHECKPOINT = REPO_ROOT / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
WAV2LIP_SCRIPT = REPO_ROOT / "third_party/Wav2Lip/inference.py"
WAV2LIP_PYTHON = Path.home() / ".venvs/wav2lip/bin/python"
SYNCNET_MODEL = REPO_ROOT / "third_party/syncnet_python/data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"


class ReplacementError(RuntimeError):
    """A frozen-input, media, or numerical contract violation."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_hash(value: Any) -> str:
    return canonical_hash(value)


def body_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return canonical_hash(body)


def write_self_hashed(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write one JSON artifact atomically using the shared protocol hash."""

    return write_json(path, dict(payload))


def read_self_hashed(path: Path) -> dict[str, Any]:
    try:
        return load_self_hashed(path)
    except Exception as exc:  # protocol errors should be uniform at the CLI
        raise ReplacementError(str(exc)) from exc


def resolve_run_paths(run_root: Path) -> tuple[Path, Path]:
    """Return ``(run_root, shared_dir)`` for both legacy and new layouts."""

    root = Path(run_root).expanduser().resolve()
    # A few early P runs left compatibility copies at the run root while the
    # frozen ``freeze.json`` lives under ``run/shared``.  Prefer the complete
    # new layout before accepting the legacy root-level fixture.
    for candidate in (root / "run" / "shared", root / "shared"):
        if (candidate / "cohort.json").is_file():
            return root, candidate
    if (root / "cohort.json").is_file():
        return root, root
    # Keep a useful error even when a caller is preparing a new run.
    return root, root / "run" / "shared"


def branch_dir(run_root: Path, branch: str = "A") -> Path:
    root = Path(run_root).expanduser().resolve()
    return root / "run" / branch


def _path_from_record(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplacementError(f"manifest path is not a non-empty string: {value!r}")
    return Path(value).expanduser().resolve()


def _check_file(path: Path, expected_sha: str | None, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"{label}: missing {path}")
        return
    if expected_sha and file_sha256(path) != expected_sha:
        errors.append(f"{label}: sha256 mismatch for {path}")


def _records_from_shared(shared: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = {name: shared / f"{name}.json" for name in ("cohort", "inputs", "freeze", "validation")}
    loaded: dict[str, dict[str, Any]] = {}
    for name, path in required.items():
        loaded[name] = read_self_hashed(path)
    return loaded["cohort"], loaded["inputs"], loaded["freeze"], loaded["validation"]


def validate_frozen_inputs(
    run_root: Path,
    *,
    require_files: bool = True,
    require_ready: bool = True,
) -> dict[str, Any]:
    """Validate the frozen P contract and return a normalized input bundle.

    The checker intentionally does not infer a cohort from media.  A route may
    consume only the already frozen manifests, and all media hashes are checked
    before a generation or score command is launched.
    """

    root, shared = resolve_run_paths(run_root)
    if not shared.is_dir():
        raise ReplacementError(f"shared manifest directory is missing: {shared}")
    cohort, inputs, freeze, validation = _records_from_shared(shared)
    readiness = cohort.get("readiness")
    status = cohort.get("status")
    if require_ready and not (status == "GO" and readiness == "COHORT_READY"):
        raise ReplacementError(
            f"upstream cohort is not ready: status={status!r}, readiness={readiness!r}, blockers={cohort.get('blockers', [])!r}"
        )
    formal = cohort.get("formal")
    smoke = cohort.get("smoke")
    records = inputs.get("records")
    errors: list[str] = []
    if not isinstance(formal, list) or len(formal) != 12:
        errors.append("cohort formal list must contain exactly 12 records")
    if not isinstance(smoke, list) or len(smoke) != 2:
        errors.append("cohort smoke list must contain exactly 2 records")
    if inputs.get("status") != "GO":
        errors.append(f"inputs status is not GO: {inputs.get('status')!r}")
    if inputs.get("formal_count") != 12 or inputs.get("smoke_count") != 2:
        errors.append("inputs formal/smoke counts are not 12/2")
    if freeze.get("status") != "FROZEN":
        errors.append(f"freeze status is not FROZEN: {freeze.get('status')!r}")
    if validation.get("status") != "GO":
        errors.append(f"shared validation status is not GO: {validation.get('status')!r}")

    # The three frozen manifests must carry the exact same P execution
    # contract.  Checking both the contract's own digest and cross-file
    # equality prevents a caller from mixing a re-signed inputs file with an
    # older cohort/validation decision.
    contracts: list[tuple[str, Mapping[str, Any]]] = []
    for label, manifest in (("cohort", cohort), ("inputs", inputs), ("validation", validation)):
        contract = manifest.get("execution_contract")
        if not isinstance(contract, Mapping):
            errors.append(f"{label}.execution_contract is missing or malformed")
            continue
        body = dict(contract)
        actual = body.pop("contract_sha256", None)
        if not isinstance(actual, str) or canonical_hash(body) != actual:
            errors.append(f"{label}.execution_contract contract_sha256 mismatch")
        contracts.append((label, contract))
    if contracts:
        reference = contracts[0][1]
        for label, contract in contracts[1:]:
            if dict(contract) != dict(reference):
                errors.append(f"{label}.execution_contract differs from {contracts[0][0]}")
    if not isinstance(records, list) or len(records) != 14:
        errors.append("inputs records must contain exactly 14 records")
        records = records if isinstance(records, list) else []

    # cohort.json is grouped by source_group (it intentionally does not carry
    # the selected clip's sample_id); inputs.json is the clip-level manifest.
    formal_ids = [str(item.get("source_group")) for item in formal if isinstance(item, Mapping)] if isinstance(formal, list) else []
    smoke_ids = [str(item.get("source_group")) for item in smoke if isinstance(item, Mapping)] if isinstance(smoke, list) else []
    if len(set(formal_ids)) != len(formal_ids) or len(set(smoke_ids)) != len(smoke_ids):
        errors.append("cohort contains duplicate sample IDs")
    if set(formal_ids) & set(smoke_ids):
        errors.append("formal and smoke sample IDs overlap")
    groups = [str(item.get("source_group")) for item in records if isinstance(item, Mapping)]
    if len(set(groups)) != len(groups):
        errors.append("frozen records must have one clip per source group")
    if groups != formal_ids + smoke_ids and set(groups) != set(formal_ids + smoke_ids):
        errors.append("inputs source groups do not bind to cohort formal/smoke groups")

    # Freeze hashes are file hashes, not self-hashes.  This catches a file that
    # was edited and re-signed after P froze it.
    inputs_path = shared / "inputs.json"
    cohort_path = shared / "cohort.json"
    if freeze.get("inputs_sha256") and file_sha256(inputs_path) != freeze.get("inputs_sha256"):
        errors.append("freeze.inputs_sha256 does not match inputs.json")
    if freeze.get("cohort_sha256") and file_sha256(cohort_path) != freeze.get("cohort_sha256"):
        errors.append("freeze.cohort_sha256 does not match cohort.json")
    if freeze.get("record_count") != 14:
        errors.append("freeze record_count is not 14")

    normalized_records: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, Mapping):
            errors.append("inputs record is not an object")
            continue
        item = dict(row)
        sid = str(item.get("sample_id", ""))
        if not sid:
            errors.append("inputs record has no sample_id")
            continue
        paths = {
            "natural_audio": (item.get("natural_audio"), item.get("natural_audio_sha256")),
            "real_video": (item.get("real_video"), item.get("real_video_sha256")),
            "reference_image": (item.get("reference_image"), item.get("reference_image_sha256")),
        }
        direct = item.get("direct_audio")
        if not isinstance(direct, Mapping):
            errors.append(f"{sid}: direct_audio contract missing")
            direct = {}
        paths["direct_audio"] = (direct.get("output_path"), direct.get("output_sha256"))
        for label, (value, expected) in paths.items():
            try:
                path = _path_from_record(value)
            except ReplacementError as exc:
                errors.append(f"{sid}: {label}: {exc}")
                continue
            if require_files:
                _check_file(path, str(expected) if expected else None, f"{sid}/{label}", errors)
            item[f"_{label}_path"] = path
        natural_count = item.get("natural_pcm_sample_count")
        direct_audio = item.get("direct_audio") or {}
        direct_count = (direct_audio.get("audio") or {}).get("sample_count")
        if natural_count is None or direct_count is None or int(natural_count) != int(direct_count):
            errors.append(f"{sid}: N/C PCM sample counts differ")
        if direct_audio.get("length_adjustment", {}).get("action") not in {"right_zero_pad", "none"}:
            errors.append(f"{sid}: direct audio used forbidden tail operation")
        if direct_audio.get("audio", {}).get("exact_natural_sample_count") is not True:
            errors.append(f"{sid}: C is not exact natural length")
        normalized_records.append(item)

    if errors:
        raise ReplacementError("frozen input contract failed: " + "; ".join(errors[:20]))
    return {
        "run_root": root,
        "shared": shared,
        "cohort": cohort,
        "inputs": inputs,
        "freeze": freeze,
        "validation": validation,
        "records": normalized_records,
        "formal_records": normalized_records[:12],
        "smoke_records": normalized_records[12:],
        "formal_ids": formal_ids,
        "smoke_ids": smoke_ids,
        "inputs_sha256": file_sha256(inputs_path),
        "cohort_sha256": file_sha256(cohort_path),
    }


def read_pcm16(path: Path) -> tuple[bytes, np.ndarray, dict[str, int]]:
    """Read a frozen WAV and reject resampling/format ambiguity."""

    try:
        with wave.open(str(path), "rb") as handle:
            params = {
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "frame_count": handle.getnframes(),
            }
            pcm = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise ReplacementError(f"cannot read PCM WAV: {path}") from exc
    expected = {"channels": 1, "sample_width": 2, "sample_rate": SAMPLE_RATE}
    if any(params[key] != value for key, value in expected.items()):
        raise ReplacementError(f"WAV is not mono PCM16/16k: {path}: {params}")
    if len(pcm) != params["frame_count"] * 2 or len(pcm) % 2:
        raise ReplacementError(f"WAV PCM byte/frame mismatch: {path}")
    array = np.frombuffer(pcm, dtype="<i2").copy()
    return pcm, array, params


def decode_pcm(path: Path) -> bytes:
    """Decode the first audio stream as canonical PCM16/16k mono."""

    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", errors="replace")[-1000:]
        raise ReplacementError(f"PCM decode failed for {path}: {detail}")
    if len(result.stdout) % 2:
        raise ReplacementError(f"decoded PCM has odd byte count: {path}")
    return bytes(result.stdout)


def decode_video_raw(path: Path, *, width: int = 512, height: int = 512) -> bytes:
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", errors="replace")[-1000:]
        raise ReplacementError(f"video decode failed for {path}: {detail}")
    return bytes(result.stdout)


def _ffprobe_json(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-print_format", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise ReplacementError(f"ffprobe failed for {path}: {result.stderr[-1000:]}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReplacementError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(value, dict):
        raise ReplacementError(f"ffprobe result is not an object for {path}")
    return value


def media_probe(path: Path) -> dict[str, Any]:
    probe = _ffprobe_json(path)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not isinstance(video, Mapping):
        raise ReplacementError(f"media has no video stream: {path}")
    fps = 0.0
    try:
        num, den = str(video.get("r_frame_rate", "0/1")).split("/", 1)
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        pass
    # ``best_effort_timestamp_time`` is reliable for the first decoded frame,
    # unlike container start_time when a muxer writes an edit list.
    first_pts_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    first_pts = None
    for line in first_pts_result.stdout.splitlines():
        try:
            first_pts = float(line.strip().split(",", 1)[0])
            break
        except ValueError:
            continue
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "video_codec": str(video.get("codec_name") or ""),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": fps,
        "frame_count": frames,
        "first_video_pts": first_pts,
        "video_start_time": video.get("start_time"),
        "has_audio": isinstance(audio, Mapping),
        "audio_codec": str(audio.get("codec_name") or "") if isinstance(audio, Mapping) else None,
        "audio_sample_rate": int(audio.get("sample_rate") or 0) if isinstance(audio, Mapping) else 0,
        "audio_channels": int(audio.get("channels") or 0) if isinstance(audio, Mapping) else 0,
    }


def _strict_run(command: Sequence[str], *, cwd: Path | None = None, log_path: Path | None = None, env: Mapping[str, str] | None = None) -> None:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(list(command), cwd=str(cwd) if cwd else None, env=dict(env) if env else None, capture_output=True, text=True, check=False)
    if log_path:
        log_path.write_text((result.stdout or "") + ("\n[stderr]\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        raise ReplacementError(f"command failed ({result.returncode}): {' '.join(map(str, command))}\n{detail}")


def normalize_video(video_only: Path, output: Path, *, frame_count: int = VIDEO_FRAMES) -> dict[str, Any]:
    """Normalize Wav2Lip output to the fixed 512²/25fps FFV1 video stream."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.video{output.suffix}")
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video_only), "-an",
        "-vf", "scale=512:512:flags=lanczos,fps=25", "-frames:v", str(frame_count),
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p", "-f", "matroska", str(temporary),
    ]
    _strict_run(command)
    os.replace(temporary, output)
    probe = media_probe(output)
    if probe["frame_count"] != frame_count or abs(float(probe["fps"]) - FPS) > 1e-6:
        raise ReplacementError(f"normalized video does not satisfy frame clock: {probe}")
    if probe["first_video_pts"] is not None and abs(float(probe["first_video_pts"])) > 1e-6:
        raise ReplacementError(f"normalized video first PTS is not zero: {probe['first_video_pts']}")
    return {"command": command, "probe": probe, "video_sha256": file_sha256(output), "decoded_sha256": sha256_bytes(decode_video_raw(output))}


def mux_pcm_strict(
    video_only: Path,
    audio_wav: Path,
    output: Path,
    *,
    expected_pcm_sha256: str | None = None,
    expected_frame_count: int = VIDEO_FRAMES,
) -> dict[str, Any]:
    """Mux the complete frozen PCM stream and verify it independently."""

    expected_pcm = decode_pcm(audio_wav)
    expected_sha = sha256_bytes(expected_pcm)
    if expected_pcm_sha256 and expected_sha != expected_pcm_sha256:
        raise ReplacementError(f"audio source PCM hash differs from frozen binding: {audio_wav}")
    video_probe = media_probe(video_only)
    if video_probe["frame_count"] != expected_frame_count:
        raise ReplacementError(f"video has {video_probe['frame_count']} frames; expected {expected_frame_count}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.mux{output.suffix}")
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video_only), "-i", str(audio_wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "matroska", str(temporary),
    ]
    _strict_run(command)
    os.replace(temporary, output)
    output_probe = media_probe(output)
    if output_probe["frame_count"] != expected_frame_count:
        raise ReplacementError(f"muxed video frame count changed: {output_probe}")
    if output_probe["first_video_pts"] is not None and abs(float(output_probe["first_video_pts"])) > 1e-6:
        raise ReplacementError(f"muxed video first PTS is not zero: {output_probe['first_video_pts']}")
    actual_pcm = decode_pcm(output)
    if actual_pcm != expected_pcm:
        raise ReplacementError(f"muxed PCM differs from the frozen audio: {output}")
    return {
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "command": command,
        "video_source": str(video_only.resolve()),
        "video_source_sha256": file_sha256(video_only),
        "video_source_decoded_sha256": sha256_bytes(decode_video_raw(video_only)),
        "video_muxed_decoded_sha256": sha256_bytes(decode_video_raw(output)),
        "video_pixel_identity": sha256_bytes(decode_video_raw(video_only)) == sha256_bytes(decode_video_raw(output)),
        "audio_source": str(audio_wav.resolve()),
        "audio_source_sha256": file_sha256(audio_wav),
        "expected_pcm_sha256": expected_sha,
        "muxed_pcm_sha256": sha256_bytes(actual_pcm),
        "expected_pcm_bytes": len(expected_pcm),
        "muxed_pcm_bytes": len(actual_pcm),
        "frame_count": int(output_probe["frame_count"]),
        "fps": float(output_probe["fps"]),
        "first_video_pts": output_probe["first_video_pts"],
        "video_stream_copy_verified": True,
        "audio_pcm_verified": True,
    }


def seed_environment(seed: int) -> dict[str, str]:
    """Environment knobs recorded for a reproducible independent process."""

    return {
        "PYTHONHASHSEED": str(int(seed)),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_LAUNCH_BLOCKING": "1",
    }


def generated_file_sha(path: Path) -> str:
    if not path.is_file():
        raise ReplacementError(f"generated artifact is missing: {path}")
    return file_sha256(path)


def cell_key(sample_id: str, model: str, arm: str, seed: int, repeat_index: int = 0) -> str:
    return f"{sample_id}/{model}/{arm}/{int(seed)}/{int(repeat_index)}"


def expected_cell_keys(sample_ids: Sequence[str], model: str = "wav2lip", *, include_repeats: bool = True) -> list[str]:
    keys: list[str] = []
    for sample_id in sample_ids:
        for arm in ("N", "C"):
            for seed in SEEDS:
                keys.append(cell_key(sample_id, model, arm, seed, 0))
    if include_repeats:
        for sample_id in list(sample_ids)[:2]:
            keys.append(cell_key(sample_id, model, "N", 42, 1))
    return keys


def copy_without_artifact_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return body


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "EMBEDDING_DIM",
    "EPSILON",
    "FPS",
    "LAG_COUNT",
    "LAG_MAX",
    "LAG_MIN",
    "SAMPLES_PER_FRAME",
    "SAMPLE_RATE",
    "SEEDS",
    "SHIFT_FRAMES",
    "SHIFT_SAMPLES",
    "SYNCNET_MODEL",
    "SYNCNET_MODEL_SHA256",
    "U_START",
    "U_STOP",
    "VIDEO_FRAMES",
    "WAV2LIP_CHECKPOINT",
    "WAV2LIP_PYTHON",
    "WAV2LIP_SCRIPT",
    "ReplacementError",
    "body_hash",
    "branch_dir",
    "cell_key",
    "decode_pcm",
    "decode_video_raw",
    "expected_cell_keys",
    "file_sha256",
    "generated_file_sha",
    "json_hash",
    "media_probe",
    "mux_pcm_strict",
    "normalize_video",
    "read_pcm16",
    "read_self_hashed",
    "resolve_run_paths",
    "seed_environment",
    "sha256_bytes",
    "validate_frozen_inputs",
    "write_self_hashed",
]
