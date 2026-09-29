from __future__ import annotations

import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config
from .common import file_sha256, verify_self_hashed_json, write_self_hashed_json
from .protocol import ProtocolError


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        sample_count = handle.getnframes()
        payload = handle.readframes(sample_count)
    if channels != config.PCM_CHANNELS or sample_width != config.PCM_SAMPLE_WIDTH or sample_rate != config.SAMPLE_RATE:
        raise ProtocolError(f"audio format violates PCM16/16k mono contract: {path}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != sample_count or values.size < config.MIN_AUDIO_SAMPLES:
        raise ProtocolError(f"audio sample count is invalid: {path}")
    return values, {
        "sample_count": int(values.size),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "sample_width": int(sample_width),
        "pcm_sha256": file_sha256(path),
    }


def pcm16_to_float(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    return values.astype(np.float64) / 32768.0


def float_to_pcm16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise FloatingPointError("waveform is empty or contains non-finite values")
    peak = float(np.abs(values).max())
    if peak >= 1.0:
        raise ValueError("waveform is clipped")
    return np.rint(values * 32768.0).clip(-32768, 32767).astype("<i2")


def write_pcm16(path: Path, values: np.ndarray) -> str:
    pcm = np.asarray(values)
    if pcm.dtype != np.int16 or pcm.ndim != 1 or pcm.size == 0:
        raise ValueError("PCM output must be a non-empty one-dimensional int16 array")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(config.PCM_CHANNELS)
        handle.setsampwidth(config.PCM_SAMPLE_WIDTH)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    temporary.replace(path)
    return file_sha256(path)


def invert_pcm16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    if np.any(values == config.INV_MIN_PCM):
        raise ProtocolError("INV cannot represent exact inversion of -32768")
    return (-values).astype("<i2")


def shift_pcm16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    if values.size <= config.SHIFT_SAMPLES:
        raise ProtocolError("SHIFT_200 requires more than 3200 input samples")
    return np.concatenate(
        [np.zeros(config.SHIFT_SAMPLES, dtype="<i2"), values[:-config.SHIFT_SAMPLES].astype("<i2")]
    )


def _periodic_hann() -> torch.Tensor:
    return torch.hann_window(config.WIN_LENGTH, periodic=True, dtype=torch.float64, device="cpu")


def phase_preserving_blend(
    natural: np.ndarray,
    mfa_linear: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if alpha not in config.MAG_STRENGTHS:
        raise ProtocolError(f"unregistered MAG strength: {alpha}")
    natural_f = pcm16_to_float(natural)
    mfa_f = pcm16_to_float(mfa_linear)
    if natural_f.size != mfa_f.size:
        raise ProtocolError("natural and MFA-linear lengths differ")
    if natural_f.size < config.MIN_AUDIO_SAMPLES:
        raise ProtocolError("audio is shorter than the registered minimum")
    window = _periodic_hann()
    natural_tensor = torch.from_numpy(natural_f).to(dtype=torch.float64)
    mfa_tensor = torch.from_numpy(mfa_f).to(dtype=torch.float64)
    natural_spec = torch.stft(
        natural_tensor,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    mfa_spec = torch.stft(
        mfa_tensor,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    natural_mag = torch.abs(natural_spec).clamp_min(config.MAGNITUDE_FLOOR)
    mfa_mag = torch.abs(mfa_spec).clamp_min(config.MAGNITUDE_FLOOR)
    log_mag = (1.0 - alpha) * torch.log(natural_mag) + alpha * torch.log(mfa_mag)
    natural_phase = natural_spec / natural_mag
    blended_spec = torch.exp(log_mag) * natural_phase
    waveform = torch.istft(
        blended_spec,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=window,
        center=True,
        length=natural_f.size,
    ).numpy()
    if not np.isfinite(waveform).all():
        raise ProtocolError("MAG reconstruction contains non-finite samples")
    natural_rms = float(np.sqrt(np.mean(natural_f * natural_f)))
    reconstructed_rms = float(np.sqrt(np.mean(waveform * waveform)))
    if natural_rms <= 0.0 or reconstructed_rms <= 0.0:
        raise ProtocolError("MAG reconstruction has zero RMS")
    rms_scale = natural_rms / reconstructed_rms
    waveform = waveform * rms_scale
    peak_before_attenuation = float(np.abs(waveform).max())
    peak_scale = 1.0
    if peak_before_attenuation >= config.PEAK_LIMIT:
        peak_scale = config.PEAK_LIMIT / peak_before_attenuation
        waveform = waveform * peak_scale
    if float(np.abs(waveform).max()) >= 1.0:
        raise ProtocolError("MAG reconstruction remains clipped after registered scaling")
    return waveform, {
        "alpha": float(alpha),
        "stft": {
            "n_fft": config.N_FFT,
            "win_length": config.WIN_LENGTH,
            "hop_length": config.HOP_LENGTH,
            "window": "periodic_hann",
            "center": True,
            "pad_mode": "reflect",
            "magnitude_floor": config.MAGNITUDE_FLOOR,
            "dtype": "float64",
            "device": "cpu",
        },
        "phase_policy": "natural_phase",
        "natural_rms": natural_rms,
        "reconstructed_rms_before_scaling": reconstructed_rms,
        "rms_scale": float(rms_scale),
        "peak_before_attenuation": peak_before_attenuation,
        "peak_scale": float(peak_scale),
        "sample_count": int(natural_f.size),
        "forbidden_repairs": True,
    }


def waveform_qc(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ProtocolError("candidate waveform is empty or non-finite")
    peak = float(np.abs(values).max())
    return {
        "sample_count": int(values.size),
        "rms": float(np.sqrt(np.mean(values * values))),
        "peak": peak,
        "dc_offset": float(np.mean(values)),
        "finite": True,
        "clipped_sample_count": int(np.sum(np.abs(values) >= 1.0)),
        "clipped": bool(peak >= 1.0),
    }


def construct_arm(natural: np.ndarray, mfa_linear: np.ndarray, arm: str) -> tuple[np.ndarray, dict[str, Any]]:
    if natural.dtype != np.int16 or mfa_linear.dtype != np.int16 or natural.ndim != 1 or mfa_linear.ndim != 1:
        raise ValueError("arm inputs must be one-dimensional int16 arrays")
    if natural.size != mfa_linear.size:
        raise ProtocolError("natural and MFA-linear sample counts differ")
    if arm == "N":
        return natural.astype("<i2", copy=True), {"construction": "untouched_natural_pcm16"}
    if arm == "INV":
        return invert_pcm16(natural), {"construction": "exact_waveform_polarity_inversion"}
    if arm == "SHIFT_200":
        return shift_pcm16(natural), {"construction": "prepend_3200_zero_samples_drop_final_3200_samples", "shift_samples": config.SHIFT_SAMPLES}
    if arm in config.MAG_ARMS:
        waveform, provenance = phase_preserving_blend(natural, mfa_linear, config.ARM_STRENGTHS[arm])
        return float_to_pcm16(waveform), {"construction": "natural_phase_log_stft_magnitude_blend", **provenance}
    raise ProtocolError(f"unknown audio arm: {arm}")


def _candidate_cell(
    record: Mapping[str, Any],
    arm: str,
    output_stage: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    natural_path = Path(str(record["natural_audio"]["path"]))
    mfa_path = Path(str(record["mfa_linear_audio"]["path"]))
    natural, natural_meta = read_pcm16(natural_path)
    mfa_linear, mfa_meta = read_pcm16(mfa_path)
    if natural_meta["pcm_sha256"] != record["natural_audio"]["sha256"] or mfa_meta["pcm_sha256"] != record["mfa_linear_audio"]["sha256"]:
        raise ProtocolError(f"input audio binding changed: {sample_id}")
    if natural.size != int(record["natural_sample_count"]) or mfa_linear.size != natural.size:
        raise ProtocolError(f"input audio lengths violate exact-length contract: {sample_id}")
    output = output_stage / "audio" / arm / f"{sample_id}.wav"
    sidecar = output.with_suffix(".json")
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if (
            prior.get("protocol_id") != config.PROTOCOL_ID
            or prior.get("protocol_sha256") != protocol_sha256
            or prior.get("arm") != arm
            or prior.get("natural_audio_sha256") != natural_meta["pcm_sha256"]
            or prior.get("mfa_linear_audio_sha256") != mfa_meta["pcm_sha256"]
            or prior.get("output_sha256") != file_sha256(output)
        ):
            raise ProtocolError(f"existing candidate sidecar identity changed: {sample_id}/{arm}")
        return prior
    if output.exists() or sidecar.exists():
        raise ProtocolError(f"partial candidate cannot be resumed: {sample_id}/{arm}")
    pcm, construction = construct_arm(natural, mfa_linear, arm)
    if pcm.size != natural.size:
        raise ProtocolError(f"candidate length changed: {sample_id}/{arm}")
    output_sha256 = write_pcm16(output, pcm)
    generated, output_meta = read_pcm16(output)
    if generated.size != natural.size or output_meta["pcm_sha256"] != output_sha256:
        raise ProtocolError(f"candidate output format validation failed: {sample_id}/{arm}")
    float_output = pcm16_to_float(generated)
    row = {
        "schema_version": 1,
        "stage_id": "01_candidates",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "arm": arm,
        "natural_audio": str(natural_path),
        "natural_audio_sha256": natural_meta["pcm_sha256"],
        "mfa_linear_audio": str(mfa_path),
        "mfa_linear_audio_sha256": mfa_meta["pcm_sha256"],
        "output": str(output),
        "output_sha256": output_sha256,
        "sample_count": int(generated.size),
        "format": output_meta,
        "construction": construction,
        "waveform_qc": waveform_qc(float_output),
    }
    write_self_hashed_json(sidecar, row)
    return row


def run_stage01(cohort: Mapping[str, Any], protocol: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    if cohort.get("status") != "complete" or len(cohort.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("Stage 00 cohort is incomplete")
    if protocol.get("protocol_id") != config.PROTOCOL_ID:
        raise ProtocolError("Stage 00 protocol identity changed")
    protocol_sha256 = file_sha256(config.STAGES["00_protocol"] / "protocol.json")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(cohort["records"], 1):
        sample_rows: list[dict[str, Any]] = []
        for arm in config.ARMS:
            try:
                sample_rows.append(_candidate_cell(record, arm, output_stage, protocol_sha256))
            except (OSError, ProtocolError, RuntimeError, TypeError, ValueError, FloatingPointError) as exc:
                failures.append({"sample_id": str(record["sample_id"]), "arm": arm, "error_type": type(exc).__name__, "error": str(exc)})
        if len(sample_rows) == len(config.ARMS):
            rows.append({"sample_id": str(record["sample_id"]), "source_group": str(record["source_group"]), "arms": sample_rows})
        print(f"AUDIO {index}/{config.EXPECTED_RECORD_COUNT} {record['sample_id']}", flush=True)
    cell_count = sum(len(row["arms"]) for row in rows)
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT and cell_count == config.EXPECTED_RECORD_COUNT * len(config.ARMS)
    manifest = {
        "schema_version": 1,
        "stage_id": "01_candidates",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "audio_cell_count": cell_count,
        "expected_audio_cell_count": config.EXPECTED_RECORD_COUNT * len(config.ARMS),
        "arms": list(config.ARMS),
        "rows": rows,
        "failures": failures,
        "protocol_sha256": protocol_sha256,
    }
    write_self_hashed_json(output_stage / "audio_manifest.json", manifest)
    write_self_hashed_json(output_stage / "failures.json", {"schema_version": 1, "stage_id": "01_candidates", "protocol_id": config.PROTOCOL_ID, "status": manifest["status"], "failures": failures})
    if not complete:
        raise ProtocolError(f"candidate audio incomplete: {cell_count}/{config.EXPECTED_RECORD_COUNT * len(config.ARMS)}")
    return manifest
