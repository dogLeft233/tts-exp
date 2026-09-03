"""Official-style file verification and fail-closed stage decisions."""
from __future__ import annotations

import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from python_speech_features import mfcc as reference_mfcc
from scipy.io import wavfile
from torch import Tensor, nn

from .config import (
    CURVE_SIZE,
    MEL_TRUST_LIMIT,
    Q,
    SAMPLE_RATE,
    SATURATION_FRACTION_MAX,
    SEGMENT_FRAMES,
    SEGMENT_SAMPLES,
    VSHIFT,
    VIDEO_FPS,
)
from .protocol import ProtocolError, frame_hashes, sha256_bytes, sha256_file
from .syncnet_loss import (
    cached_visual_embeddings,
    official_syncnet_distance_curve,
    pcm16_forward_numpy,
    syncnet_audio_windows,
)

FAILURE_PRECEDENCE = (
    ("input_lock", "INPUT_LOCK_FAILURE"),
    ("input_isolation", "NATURAL_INPUT_LEAKAGE"),
    ("visual_coordinates", "VISUAL_COORDINATE_MISMATCH"),
    ("target_unambiguous", "TARGET_OFFSET_AMBIGUOUS"),
    ("offset_sign", "OFFSET_SIGN_MISMATCH"),
    ("mfcc_parity", "MFCC_PARITY_FAILURE"),
    ("identity", "IDENTITY_FAILURE"),
    ("frozen_state", "FROZEN_STATE_MUTATION"),
    ("candidate_gradient", "NO_CANDIDATE_GRADIENT"),
    ("finite", "NONFINITE_TRAINING"),
    ("trust", "TRUST_REGION_FAILURE"),
    ("proxy_descent", "NO_PROXY_LOSS_DESCENT"),
    ("file_proxy_parity", "FILE_PROXY_PARITY_FAILURE"),
    ("official_d_margin", "OFFICIAL_D_MARGIN_FAILURE"),
    ("official_c_margin", "OFFICIAL_C_MARGIN_FAILURE"),
    ("official_offset", "OFFICIAL_OFFSET_FAILURE"),
)


def curve_metrics(curve: Sequence[float] | np.ndarray | Tensor, *, vshift: int = VSHIFT) -> dict[str, Any]:
    values = np.asarray(torch.as_tensor(curve).detach().cpu(), dtype=np.float64)
    if values.ndim != 1 or values.size != 2 * vshift + 1 or not np.isfinite(values).all():
        raise ValueError("curve must be one finite value per searched offset")
    order = np.argsort(values, kind="stable")
    best_index = int(order[0])
    shift = best_index - vshift
    return {
        "curve": values.tolist(),
        "sync_d": float(values[best_index]),
        "sync_c": float(np.median(values) - values[best_index]),
        "best_index": best_index,
        "best_signed_shift": shift,
        "official_av_offset": -shift,
        "best_second_gap": float(values[order[1]] - values[order[0]]),
    }


def proxy_file_parity(proxy_curve: Sequence[float], official_curve: Sequence[float], *, q: float = Q) -> dict[str, Any]:
    proxy = np.asarray(proxy_curve, dtype=np.float64)
    official = np.asarray(official_curve, dtype=np.float64)
    if proxy.shape != official.shape or proxy.shape != (2 * VSHIFT + 1,):
        raise ValueError("proxy and official curves must both have 31 values")
    if not np.isfinite(proxy).all() or not np.isfinite(official).all():
        raise FloatingPointError("proxy/file curve is non-finite")
    errors = np.abs(proxy - official)
    maximum = float(errors.max())
    return {"pass": maximum <= q, "max_abs_error": maximum, "per_offset_abs_error": errors.tolist(), "q": q}


def p1_predicates(evidence: Mapping[str, Any], *, q: float = Q) -> dict[str, bool]:
    baseline = curve_metrics(evidence["official_step0_curve"])
    final = curve_metrics(evidence["official_final_curve"])
    parity = proxy_file_parity(evidence["proxy_final_curve"], evidence["official_final_curve"], q=q)
    target_offset = int(evidence["target_offset"])
    finite_scalars = [
        evidence.get("proxy_step0_loss"),
        evidence.get("proxy_final_loss"),
        evidence.get("mel_distance"),
        evidence.get("saturation_fraction"),
        baseline["sync_d"],
        baseline["sync_c"],
        final["sync_d"],
        final["sync_c"],
    ]
    final_qc = evidence.get("final_qc", {})
    return {
        "input_lock": bool(evidence.get("input_lock", False)),
        "input_isolation": bool(evidence.get("input_isolation", False)),
        "visual_coordinates": bool(evidence.get("visual_coordinates", False)),
        "target_unambiguous": bool(evidence.get("target_unambiguous", False)),
        "offset_sign": bool(evidence.get("offset_sign", False)),
        "mfcc_parity": bool(evidence.get("mfcc_parity", False)),
        "identity": bool(evidence.get("identity", False)),
        "frozen_state": bool(evidence.get("frozen_state", False)),
        "candidate_gradient": bool(evidence.get("candidate_gradient", False)),
        "finite": all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in finite_scalars)
        and final_qc.get("finite") is True,
        "trust": float(evidence["mel_distance"]) <= MEL_TRUST_LIMIT
        and float(evidence["saturation_fraction"]) <= SATURATION_FRACTION_MAX
        and final_qc.get("exact_shape") is True
        and final_qc.get("residual_bound_pass") is True
        and final_qc.get("pcm_saturation_pass") is True,
        "proxy_descent": float(evidence["proxy_final_loss"]) < float(evidence["proxy_step0_loss"]),
        "file_proxy_parity": bool(parity["pass"]),
        "official_d_margin": baseline["sync_d"] - final["sync_d"] > 3 * q,
        "official_c_margin": final["sync_c"] - baseline["sync_c"] > 3 * q,
        "official_offset": final["best_signed_shift"] == target_offset and final["best_second_gap"] > 2 * q,
    }


def decide_p1(evidence: Mapping[str, Any], *, q: float = Q) -> dict[str, Any]:
    predicates = p1_predicates(evidence, q=q)
    status = "FIXED_DATA_TTS_ONLY_SYNC_CONSTRUCTED"
    for predicate, failure in FAILURE_PRECEDENCE:
        if not predicates[predicate]:
            status = failure
            break
    return {
        "stage": "P1_ONE_RECORD",
        "status": status,
        "pass": status == "FIXED_DATA_TTS_ONLY_SYNC_CONSTRUCTED",
        "predicates": predicates,
        "claim_scope": "fixed-data constructability on the optimized record only",
        "disallowed_claims": [
            "held-out validation",
            "generalization",
            "perceptual improvement",
            "content or speaker preservation",
            "Wav2Lip or TFG gain",
            "replacement safety",
        ],
    }


def decide_p2(record_evidence: Sequence[Mapping[str, Any]], *, q: float = Q) -> dict[str, Any]:
    if len(record_evidence) != 4:
        return {"stage": "P2_SHARED_FOUR", "status": "INPUT_LOCK_FAILURE", "pass": False}
    rows = []
    d_gains = []
    c_gains = []
    joint = 0
    for evidence in record_evidence:
        base = curve_metrics(evidence["official_step0_curve"])
        final = curve_metrics(evidence["official_final_curve"])
        d_gain = base["sync_d"] - final["sync_d"]
        c_gain = final["sync_c"] - base["sync_c"]
        target = int(evidence["target_offset"])
        integrity = all(
            p1_predicates(evidence, q=q)[name]
            for name in (
                "input_lock",
                "input_isolation",
                "visual_coordinates",
                "target_unambiguous",
                "offset_sign",
                "mfcc_parity",
                "identity",
                "frozen_state",
                "candidate_gradient",
                "finite",
                "trust",
                "proxy_descent",
                "file_proxy_parity",
            )
        )
        wins = d_gain > 3 * q and c_gain > 3 * q and final["best_signed_shift"] == target and final["best_second_gap"] > 2 * q
        joint += int(integrity and wins)
        d_gains.append(d_gain)
        c_gains.append(c_gain)
        rows.append({"d_gain": d_gain, "c_gain": c_gain, "integrity": integrity, "joint_win": integrity and wins})
    passed = joint >= 3 and float(np.median(d_gains)) > 3 * q and float(np.median(c_gains)) > 3 * q
    return {
        "stage": "P2_SHARED_FOUR",
        "status": "FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED" if passed else "FIXED_DATA_SHARED_TTS_ONLY_SYNC_NOT_CONSTRUCTED",
        "pass": passed,
        "joint_wins": joint,
        "median_d_gain": float(np.median(d_gains)),
        "median_c_gain": float(np.median(c_gains)),
        "records": rows,
        "claim_scope": "shared fixed-data constructability on four optimized records only",
    }


def write_pcm16_wav_once(path: str | Path, waveform: Tensor) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"create-once PCM already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = pcm16_forward_numpy(waveform).reshape(-1)
    if pcm.size != SEGMENT_SAMPLES:
        raise ValueError("candidate must contain exactly 61,440 samples")
    wavfile.write(target, SAMPLE_RATE, pcm)
    return {"path": str(target.resolve()), "sha256": sha256_file(target), "pcm_sha256": sha256_bytes(pcm.tobytes())}


def mux_condition_once(canonical_video: str | Path, candidate_wav: str | Path, output: str | Path) -> None:
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"create-once condition already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(Path(canonical_video).resolve()),
        "-i",
        str(Path(candidate_wav).resolve()),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        "-shortest",
        str(target),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not target.is_file():
        target.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg condition mux failed: {completed.stderr.strip()}")


def decode_bgr_frames(path: str | Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(Path(path).resolve()))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != SEGMENT_FRAMES:
        raise ProtocolError(f"decoded condition has {len(frames)} frames, expected {SEGMENT_FRAMES}")
    return np.stack(frames)


def reference_mfcc_tensor(pcm: np.ndarray, *, device: torch.device) -> Tensor:
    values = reference_mfcc(
        pcm,
        samplerate=SAMPLE_RATE,
        winlen=0.025,
        winstep=0.01,
        numcep=13,
        nfilt=26,
        nfft=512,
        preemph=0.97,
        appendEnergy=True,
        winfunc=lambda size: np.ones((size,)),
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("official reference MFCC is non-finite")
    return torch.as_tensor(values.T[None], device=device, dtype=torch.float32)


def official_file_curve(
    syncnet: nn.Module,
    condition_video: str | Path,
    *,
    expected_frame_hashes: Sequence[str],
    expected_pcm_sha256: str,
    device: torch.device,
    syncnet_checkpoint: str | Path | None = None,
    expected_scoring_frame_hashes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score through the unmodified SyncNet V2 file scorer, without tracking."""
    del syncnet, device
    condition = Path(condition_video).resolve()
    frames = decode_bgr_frames(condition)
    actual_hashes = frame_hashes(frames)
    if list(expected_frame_hashes) != actual_hashes:
        raise ProtocolError("VISUAL_COORDINATE_MISMATCH: muxed condition frames changed")
    demux = condition.with_suffix(".demux.wav")
    if demux.exists():
        raise FileExistsError(f"create-once demux already exists: {demux}")
    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(condition),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(demux),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        demux.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg demux failed: {completed.stderr.strip()}")
    sample_rate, pcm = wavfile.read(demux)
    if sample_rate != SAMPLE_RATE or pcm.dtype != np.int16 or pcm.ndim != 1 or pcm.size != SEGMENT_SAMPLES:
        raise ProtocolError("demuxed candidate PCM format/length mismatch")
    pcm_hash = sha256_bytes(np.ascontiguousarray(pcm).tobytes())
    if pcm_hash != expected_pcm_sha256:
        raise ProtocolError("demuxed PCM differs from proxy forward PCM")

    package_root = Path(__file__).resolve().parents[3]
    syncnet_root = package_root / "third_party/syncnet_python"
    checkpoint = Path(syncnet_checkpoint or syncnet_root / "data/syncnet_v2.model").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    reference = "fixed_condition"
    data_dir = condition.parent / f".official_syncnet_{condition.stem}"
    if data_dir.exists():
        raise FileExistsError(f"create-once official scorer directory already exists: {data_dir}")
    crop_path = data_dir / "pycrop" / reference / "00000.avi"
    work_path = data_dir / "pywork" / reference
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    work_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(condition, crop_path)
    python_path = Path.home() / ".venvs/syncnet/bin/python"
    if not python_path.is_file():
        python_path = Path(sys.executable)
    command = [
        str(python_path), str(syncnet_root / "run_syncnet.py"),
        "--initial_model", str(checkpoint), "--batch_size", "20", "--vshift", str(VSHIFT),
        "--data_dir", str(data_dir), "--reference", reference, "--videofile", str(crop_path),
    ]
    completed = subprocess.run(command, cwd=str(syncnet_root), check=False, capture_output=True, text=True)
    log_path = condition.with_suffix(".official_syncnet.log")
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"official SyncNet scorer failed: {completed.stderr.strip()}")
    if expected_scoring_frame_hashes is not None:
        scoring_paths = sorted((data_dir / "pytmp" / reference).glob("*.jpg"))
        if len(scoring_paths) != SEGMENT_FRAMES:
            raise ProtocolError("official scorer produced an unexpected number of JPEG frames")
        scoring_frames = []
        for path in scoring_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ProtocolError(f"official scorer produced unreadable JPEG: {path}")
            scoring_frames.append(frame)
        scoring_hashes = frame_hashes(np.stack(scoring_frames))
        if list(expected_scoring_frame_hashes) != scoring_hashes:
            raise ProtocolError("VISUAL_COORDINATE_MISMATCH: official JPEG frames changed")
    active_path = work_path / "activesd.pckl"
    if not active_path.is_file():
        raise ProtocolError("official SyncNet scorer did not emit activesd.pckl")
    with active_path.open("rb") as handle:
        distances = pickle.load(handle)
    matrix = np.asarray(distances)
    if matrix.shape != (1, 91, CURVE_SIZE) or not np.isfinite(matrix).all():
        raise ProtocolError(f"official SyncNet distances have unexpected shape: {matrix.shape}")
    curve = matrix[0].mean(axis=0)
    return {
        **curve_metrics(curve),
        "condition_video": str(condition),
        "condition_video_sha256": sha256_file(condition),
        "demux_wav": str(demux.resolve()),
        "demux_wav_sha256": sha256_file(demux),
        "pcm_sha256": pcm_hash,
        "decoded_bgr_frame_sha256": actual_hashes,
        "window_count": int(matrix.shape[1]),
        "official_scorer": str((syncnet_root / "run_syncnet.py").resolve()),
        "official_scorer_sha256": sha256_file(syncnet_root / "run_syncnet.py"),
        "official_command": command,
        "official_data_dir": str(data_dir),
        "official_checkpoint_sha256": sha256_file(checkpoint),
    }
