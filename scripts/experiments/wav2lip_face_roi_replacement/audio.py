from __future__ import annotations

import math
import subprocess
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config
from .common import (
    ProtocolError,
    bytes_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            count = handle.getnframes()
            payload = handle.readframes(count)
    except (OSError, wave.Error) as exc:
        raise ProtocolError(f"invalid WAV: {path}") from exc
    if (channels, width, rate) != (config.PCM_CHANNELS, config.PCM_SAMPLE_WIDTH, config.SAMPLE_RATE):
        raise ProtocolError(f"audio format is not 16 kHz mono PCM16: {path}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != count or values.size == 0:
        raise ProtocolError(f"audio sample count is invalid: {path}")
    return values, {
        "sample_count": int(values.size),
        "sample_rate": int(rate),
        "channels": int(channels),
        "sample_width": int(width),
        "container_sha256": file_sha256(path),
        "decoded_pcm_sha256": bytes_sha256(payload),
    }


def pcm16_to_float(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    return values.astype(np.float64) / 32768.0


def float_to_pcm16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise FloatingPointError("waveform is empty or non-finite")
    peak = float(np.abs(values).max())
    if peak >= 1.0:
        raise ValueError("waveform is clipped")
    return np.rint(values * 32768.0).clip(-32768, 32767).astype("<i2")


def _periodic_hann() -> torch.Tensor:
    return torch.hann_window(1024, periodic=True, dtype=torch.float64, device="cpu")


def phase_preserving_blend(natural: np.ndarray, mfa_linear: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce the registered BRIDGE_075 construction exactly."""
    natural_f = pcm16_to_float(natural)
    mfa_f = pcm16_to_float(mfa_linear)
    if natural_f.size != mfa_f.size:
        raise ProtocolError("natural and MFA-linear lengths differ")
    if natural_f.size < 1024:
        raise ProtocolError("audio is shorter than the registered minimum")
    window = _periodic_hann()
    natural_tensor = torch.from_numpy(natural_f)
    mfa_tensor = torch.from_numpy(mfa_f)
    natural_spec = torch.stft(natural_tensor, n_fft=1024, hop_length=256, win_length=1024, window=window, center=True, pad_mode="reflect", return_complex=True)
    mfa_spec = torch.stft(mfa_tensor, n_fft=1024, hop_length=256, win_length=1024, window=window, center=True, pad_mode="reflect", return_complex=True)
    natural_mag = torch.abs(natural_spec).clamp_min(1e-7)
    mfa_mag = torch.abs(mfa_spec).clamp_min(1e-7)
    log_mag = 0.25 * torch.log(natural_mag) + 0.75 * torch.log(mfa_mag)
    natural_phase = natural_spec / natural_mag
    blended_spec = torch.exp(log_mag) * natural_phase
    waveform = torch.istft(blended_spec, n_fft=1024, hop_length=256, win_length=1024, window=window, center=True, length=natural_f.size).numpy()
    if not np.isfinite(waveform).all():
        raise ProtocolError("BRIDGE_075 contains non-finite samples")
    natural_rms = float(np.sqrt(np.mean(natural_f * natural_f)))
    reconstructed_rms = float(np.sqrt(np.mean(waveform * waveform)))
    if natural_rms <= 0.0 or reconstructed_rms <= 0.0:
        raise ProtocolError("BRIDGE_075 has zero RMS")
    rms_scale = natural_rms / reconstructed_rms
    waveform = waveform * rms_scale
    peak_before = float(np.abs(waveform).max())
    peak_scale = 1.0
    if peak_before >= 0.999:
        peak_scale = 0.999 / peak_before
        waveform = waveform * peak_scale
    if float(np.abs(waveform).max()) >= 1.0:
        raise ProtocolError("BRIDGE_075 remains clipped after registered scaling")
    return waveform, {
        "alpha": 0.75,
        "stft": {"n_fft": 1024, "win_length": 1024, "hop_length": 256, "window": "periodic_hann", "center": True, "pad_mode": "reflect", "magnitude_floor": 1e-7, "dtype": "float64", "device": "cpu"},
        "phase_policy": "natural_phase",
        "natural_rms": natural_rms,
        "reconstructed_rms_before_scaling": reconstructed_rms,
        "rms_scale": float(rms_scale),
        "peak_before_attenuation": peak_before,
        "peak_scale": float(peak_scale),
        "sample_count": int(natural_f.size),
        "forbidden_repairs": True,
    }


def audio_forward_map(sample_count: int) -> np.ndarray:
    if sample_count < 2 or sample_count - 1 <= 2.0 * math.pi * 1920:
        raise ProtocolError("natural audio is too short for LOCAL_WARP_120")
    coordinates = np.arange(sample_count, dtype=np.float64)
    mapped = coordinates + 1920.0 * np.sin(2.0 * np.pi * coordinates / float(sample_count - 1))
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0):
        raise ProtocolError("LOCAL_WARP_120 map is not strictly increasing")
    return mapped


def reconstruct_warped_pcm(natural_pcm: bytes) -> tuple[bytes, np.ndarray]:
    if len(natural_pcm) % 2:
        raise ProtocolError("natural PCM is not int16 aligned")
    samples = np.frombuffer(natural_pcm, dtype="<i2")
    mapped = audio_forward_map(int(samples.size))
    original = np.arange(samples.size, dtype=np.float64)
    warped = np.interp(mapped, original, samples.astype(np.float64))
    return np.rint(warped).astype("<i2", copy=False).tobytes(), mapped


def waveform_qc(values: np.ndarray) -> dict[str, Any]:
    float_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if float_values.size == 0 or not np.isfinite(float_values).all():
        raise ProtocolError("waveform is empty or non-finite")
    peak = float(np.abs(float_values).max())
    return {"sample_count": int(float_values.size), "rms": float(np.sqrt(np.mean(float_values * float_values))), "peak": peak, "dc_offset": float(np.mean(float_values)), "finite": True, "clipped_sample_count": int(np.sum(np.abs(float_values) >= 1.0)), "clipped": bool(peak >= 1.0)}


def _arm_row(arm: str, path: Path, natural: np.ndarray, mfa: np.ndarray, construction: Mapping[str, Any]) -> dict[str, Any]:
    values, meta = read_pcm16(path)
    if values.size != natural.size:
        raise ProtocolError(f"candidate length differs: {arm}")
    if arm in (config.AUDIO_N, config.AUDIO_N_REPEAT) and values.tobytes() != natural.tobytes():
        raise ProtocolError(f"{arm} PCM differs from natural")
    return {"schema_version": 1, "arm": arm, "path": str(path.resolve()), "output": str(path.resolve()), "output_sha256": meta["container_sha256"], "container_sha256": meta["container_sha256"], "decoded_pcm_sha256": meta["decoded_pcm_sha256"], "sample_count": int(values.size), "format": meta, "construction": dict(construction), "waveform_qc": waveform_qc(pcm16_to_float(values))}


def build_audio_manifest(records: Sequence[Mapping[str, Any]], output: Path, mel_diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    diagnostics_by_id = {str(row.get("sample_id")): row for row in (mel_diagnostics or {}).get("rows", []) if isinstance(row, Mapping)}
    rows: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        natural, natural_meta = read_pcm16(Path(str(record["natural_audio"]["path"])))
        mfa, mfa_meta = read_pcm16(Path(str(record["mfa_linear_audio"]["path"])))
        if natural_meta["container_sha256"] != str(record["natural_audio"]["sha256"]) or mfa_meta["container_sha256"] != str(record["mfa_linear_audio"]["sha256"]):
            raise ProtocolError(f"source audio hash changed: {sample_id}")
        if natural.size != int(record["natural_sample_count"]) or mfa.size != natural.size:
            raise ProtocolError(f"source audio length changed: {sample_id}")
        paths = record["candidate_audio_paths"]
        arms: list[dict[str, Any]] = []
        arms.append(_arm_row(config.AUDIO_N, Path(str(paths[config.AUDIO_N])), natural, mfa, {"construction": "untouched_natural_pcm16"}))
        arms.append(_arm_row(config.AUDIO_N_REPEAT, Path(str(paths[config.AUDIO_N_REPEAT])), natural, mfa, {"construction": "independent_natural_pcm16_identity"}))
        warped, mapped = reconstruct_warped_pcm(natural.tobytes())
        warp_path = Path(str(paths[config.AUDIO_W]))
        warp_values, _ = read_pcm16(warp_path)
        if warp_values.tobytes() != warped:
            raise ProtocolError(f"W LOCAL_WARP_120 reconstruction differs: {sample_id}")
        arms.append(_arm_row(config.AUDIO_W, warp_path, natural, mfa, {"construction": "registered_LOCAL_WARP_120", "formula": "s[n]=n+1920*sin(2*pi*n/(L-1)); endpoints forced", "mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()), "interpolation": "float64 linear", "rounding": "nearest_ties_to_even"}))
        bridge_path = Path(str(paths[config.AUDIO_BRIDGE]))
        bridge_wave, bridge_meta = phase_preserving_blend(natural, mfa)
        bridge_values, _ = read_pcm16(bridge_path)
        expected_bridge = float_to_pcm16(bridge_wave)
        if bridge_values.tobytes() != expected_bridge.tobytes():
            raise ProtocolError(f"BRIDGE_075 reconstruction differs: {sample_id}")
        arms.append(_arm_row(config.AUDIO_BRIDGE, bridge_path, natural, mfa, {"construction": "registered_BRIDGE_075", **bridge_meta}))
        row = {"schema_version": 1, "stage_id": "prepare", "sample_id": sample_id, "source_group": str(record["source_group"]), "natural_audio_sha256": natural_meta["container_sha256"], "mfa_linear_audio_sha256": mfa_meta["container_sha256"], "natural_sample_count": int(natural.size), "arms": arms}
        diagnostic = diagnostics_by_id.get(sample_id)
        if diagnostic is not None:
            row["mel_diagnostics"] = {"arms": diagnostic["arms"], "mfa_linear": diagnostic["mfa_linear"], "bridge_movement": diagnostic["bridge_movement"]}
        rows.append(row)
    payload = {"schema_version": 1, "stage_id": "audio", "protocol_id": config.PROTOCOL_ID, "status": "complete", "record_count": len(rows), "expected_record_count": config.EXPECTED_RECORD_COUNT, "arms": list(config.AUDIO_ARMS), "rows": rows}
    write_self_hashed_json(output, payload)
    return verify_self_hashed_json(output)


def probe_mels(records: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "mel_probe_input.json"
    output_path = output_dir / "mel_diagnostics.json"
    payload = {str(row["sample_id"]): {arm: str(row["candidate_audio_paths"][arm]) for arm in config.AUDIO_ARMS} | {"mfa": str(row["mfa_linear_audio"]["path"])} for row in records}
    input_path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = dict(__import__("os").environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["NUMBA_CACHE_DIR"] = str(output_dir / "numba_cache")
    command = [str(config.WAV2LIP_PYTHON), str(Path(__file__).with_name("mel_probe.py")), "--input", str(input_path), "--output", str(output_path)]
    log_path = output_dir / "mel_probe.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(config.WAV2LIP_ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not output_path.is_file():
        raise ProtocolError(f"Wav2Lip mel probe failed: {log_path}")
    return verify_self_hashed_json(output_path)
