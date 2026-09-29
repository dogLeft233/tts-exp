from __future__ import annotations

import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config
from .common import (
    ProtocolError,
    copy_verified,
    decoded_pcm_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def read_pcm16(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    target = Path(path)
    try:
        with wave.open(str(target), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_width = int(handle.getsampwidth())
            sample_rate = int(handle.getframerate())
            sample_count = int(handle.getnframes())
            payload = handle.readframes(sample_count)
    except (OSError, wave.Error) as exc:
        raise ProtocolError(f"cannot read PCM16 WAV: {target}") from exc
    if (channels, sample_width, sample_rate) != (
        config.PCM_CHANNELS,
        config.PCM_SAMPLE_WIDTH,
        config.SAMPLE_RATE,
    ):
        raise ProtocolError(f"audio format violates 16 kHz mono PCM16 contract: {target}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != sample_count or values.size < config.MIN_AUDIO_SAMPLES:
        raise ProtocolError(f"audio sample count is invalid: {target}")
    return values, {
        "sample_count": int(values.size),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "decoded_pcm_sha256": decoded_pcm_sha256(values),
        "file_sha256": file_sha256(target),
    }


def pcm16_to_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.int16 or array.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    return array.astype(np.float64) / 32768.0


def quantize_pcm16(values: np.ndarray, *, scale: float = 32768.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ProtocolError("waveform is empty or contains non-finite samples")
    scaled = np.rint(array * scale)
    if float(np.min(scaled)) < -32768.0 or float(np.max(scaled)) > 32767.0:
        raise ProtocolError("PCM quantization would exceed int16 range; clipping is forbidden")
    return scaled.astype("<i2")


def write_pcm16(path: str | Path, values: np.ndarray) -> str:
    target = Path(path)
    array = np.asarray(values)
    if array.dtype != np.int16 or array.ndim != 1 or array.size == 0:
        raise ValueError("PCM output must be a non-empty one-dimensional int16 array")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(config.PCM_CHANNELS)
        handle.setsampwidth(config.PCM_SAMPLE_WIDTH)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(np.asarray(array, dtype="<i2").tobytes())
    temporary.replace(target)
    return file_sha256(target)


def waveform_qc(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ProtocolError("waveform is empty or non-finite")
    peak = float(np.max(np.abs(array)))
    return {
        "samples": int(array.size),
        "rms": float(np.sqrt(np.mean(array * array))),
        "peak": peak,
        "finite": True,
        "clipped_sample_count": int(np.count_nonzero(np.abs(array) >= 1.0)),
        "clipped": bool(peak >= 1.0),
    }


def _periodic_hann() -> torch.Tensor:
    return torch.hann_window(
        config.WIN_LENGTH,
        periodic=True,
        dtype=torch.float64,
        device="cpu",
    )


def _stft(values: np.ndarray) -> torch.Tensor:
    return torch.stft(
        torch.from_numpy(np.asarray(values, dtype=np.float64)),
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=_periodic_hann(),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )


def phase_preserving_bridge(
    natural: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = config.BRIDGE_ALPHA,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move natural log magnitude toward target while retaining natural phase."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("bridge alpha must be between zero and one")
    natural_f = pcm16_to_float(natural)
    target_f = pcm16_to_float(target)
    if natural_f.size != target_f.size:
        raise ProtocolError("natural and target PCM lengths differ")
    if natural_f.size < config.MIN_AUDIO_SAMPLES:
        raise ProtocolError("bridge input is shorter than the registered minimum")
    natural_spec = _stft(natural_f)
    target_spec = _stft(target_f)
    if natural_spec.shape != target_spec.shape:
        raise ProtocolError("natural and target STFT shapes differ")
    natural_mag = torch.abs(natural_spec).clamp_min(config.MAGNITUDE_FLOOR)
    target_mag = torch.abs(target_spec).clamp_min(config.MAGNITUDE_FLOOR)
    blended_mag = torch.exp(
        (1.0 - float(alpha)) * torch.log(natural_mag)
        + float(alpha) * torch.log(target_mag)
    )
    natural_phase = natural_spec / natural_mag
    blended_spec = blended_mag * natural_phase
    waveform = torch.istft(
        blended_spec,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=_periodic_hann(),
        center=True,
        length=int(natural_f.size),
    ).numpy()
    if not np.isfinite(waveform).all():
        raise ProtocolError("bridge reconstruction contains non-finite samples")
    natural_rms = float(np.sqrt(np.mean(natural_f * natural_f)))
    reconstructed_rms = float(np.sqrt(np.mean(waveform * waveform)))
    if natural_rms <= 0.0 or reconstructed_rms <= 0.0:
        raise ProtocolError("bridge reconstruction has zero RMS")
    rms_scale = natural_rms / reconstructed_rms
    waveform = waveform * rms_scale
    peak_before_attenuation = float(np.max(np.abs(waveform)))
    peak_scale = 1.0
    if peak_before_attenuation >= config.PEAK_LIMIT:
        peak_scale = config.PEAK_LIMIT / peak_before_attenuation
        waveform = waveform * peak_scale
    if float(np.max(np.abs(waveform))) >= 1.0:
        raise ProtocolError("bridge reconstruction remains clipped after registered scaling")
    pcm = quantize_pcm16(waveform, scale=32768.0)
    return pcm, {
        "alpha": float(alpha),
        "phase_policy": "natural_phase",
        "stft": {
            "n_fft": config.N_FFT,
            "win_length": config.WIN_LENGTH,
            "hop_length": config.HOP_LENGTH,
            "window": "periodic_hann",
            "center": True,
            "pad_mode": "reflect",
            "dtype": "float64",
            "device": "cpu",
            "magnitude_floor": config.MAGNITUDE_FLOOR,
        },
        "natural_rms": natural_rms,
        "reconstructed_rms_before_scaling": reconstructed_rms,
        "rms_scale": float(rms_scale),
        "peak_before_attenuation": peak_before_attenuation,
        "peak_scale": float(peak_scale),
        "sample_count": int(natural_f.size),
        "quantization": "rint(x*32768).astype(int16)",
    }


def construct_arm(
    natural: np.ndarray,
    local_target: np.ndarray,
    cloud_target: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    natural = np.asarray(natural)
    local_target = np.asarray(local_target)
    cloud_target = np.asarray(cloud_target)
    if any(array.dtype != np.int16 or array.ndim != 1 for array in (natural, local_target, cloud_target)):
        raise ValueError("bridge arms require one-dimensional int16 arrays")
    if not (natural.size == local_target.size == cloud_target.size):
        raise ProtocolError("bridge arm inputs have different sample counts")
    if arm == "B0":
        result, meta = phase_preserving_bridge(natural, natural, alpha=0.0)
        meta["construction"] = "same_bridge_algorithm_alpha_zero_target_N"
        return result, meta
    if arm == "B_LOCAL":
        result, meta = phase_preserving_bridge(natural, local_target, alpha=config.BRIDGE_ALPHA)
        meta["construction"] = "natural_phase_log_stft_bridge_local_target"
        return result, meta
    if arm == "B_CLOUD":
        result, meta = phase_preserving_bridge(natural, cloud_target, alpha=config.BRIDGE_ALPHA)
        meta["construction"] = "natural_phase_log_stft_bridge_cloud_target"
        return result, meta
    raise ProtocolError(f"unknown bridge arm: {arm}")


def directional_metrics(
    natural_mel: np.ndarray,
    target_mel: np.ndarray,
    candidate_mel: np.ndarray,
) -> dict[str, float | bool]:
    natural = np.asarray(natural_mel, dtype=np.float64)
    target = np.asarray(target_mel, dtype=np.float64)
    candidate = np.asarray(candidate_mel, dtype=np.float64)
    if natural.shape != target.shape or natural.shape != candidate.shape:
        raise ProtocolError("Wav2Lip mel shapes differ")
    direction = (target - natural).reshape(-1)
    movement = (candidate - natural).reshape(-1)
    denominator = float(np.dot(direction, direction))
    degenerate = denominator <= 1e-12
    denominator_safe = max(denominator, 1e-12)
    progress = float(np.dot(movement, direction) / denominator_safe)
    residual = movement - progress * direction
    return {
        "progress": progress,
        "relative_distance": float(
            np.linalg.norm(candidate.reshape(-1) - target.reshape(-1))
            / max(float(np.linalg.norm(direction)), 1e-12)
        ),
        "orthogonal_residual": float(np.linalg.norm(residual)),
        "natural_to_target_norm": float(np.linalg.norm(direction)),
        "natural_to_candidate_norm": float(np.linalg.norm(movement)),
        "mel_mae_to_natural": float(np.mean(np.abs(candidate - natural))),
        "mel_mae_to_target": float(np.mean(np.abs(candidate - target))),
        "degenerate_direction": degenerate,
    }


def wav2lip_mels(values: np.ndarray, wav2lip_root: Path = config.WAV2LIP_ROOT) -> np.ndarray:
    import sys

    root = str(Path(wav2lip_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import audio as wav2lip_audio

    waveform = np.asarray(values, dtype=np.float32).reshape(-1)
    if waveform.size < config.MIN_AUDIO_SAMPLES or not np.isfinite(waveform).all():
        raise ProtocolError("official Wav2Lip mel input is empty or non-finite")
    mel = np.asarray(wav2lip_audio.melspectrogram(waveform), dtype=np.float64)
    if mel.ndim != 2 or mel.shape[0] != 80 or not np.isfinite(mel).all():
        raise ProtocolError(f"unexpected official Wav2Lip mel shape: {mel.shape}")
    return mel


def _target_rows(targets: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    if targets.get("status") != "complete":
        raise ProtocolError("target manifest is incomplete")
    result: dict[str, dict[str, Mapping[str, Any]]] = {"LOCAL": {}, "CLOUD": {}}
    for row in targets.get("rows", []):
        provider = str(row.get("provider", ""))
        sample_id = str(row.get("sample_id", ""))
        if provider not in result or sample_id in result[provider]:
            raise ProtocolError(f"duplicate or unknown target row: {provider}/{sample_id}")
        result[provider][sample_id] = row
    if any(len(value) != config.EXPECTED_RECORD_COUNT for value in result.values()):
        raise ProtocolError("target provider coverage is incomplete")
    return result


def _write_or_resume_arm(
    *,
    natural: np.ndarray,
    local_target: np.ndarray,
    cloud_target: np.ndarray,
    record: Mapping[str, Any],
    arm: str,
    output_stage: Path,
    protocol_sha256: str,
    natural_sha256: str,
    local_sha256: str,
    cloud_sha256: str,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    output = output_stage / "audio" / arm / f"{sample_id}.wav"
    sidecar = output.with_suffix(".json")
    expected_inputs = {
        "natural_audio_sha256": natural_sha256,
        "local_target_sha256": local_sha256,
        "cloud_target_sha256": cloud_sha256,
    }
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if (
            prior.get("protocol_id") != config.PROTOCOL_ID
            or prior.get("protocol_sha256") != protocol_sha256
            or prior.get("sample_id") != sample_id
            or prior.get("arm") != arm
            or any(prior.get(key) != value for key, value in expected_inputs.items())
            or prior.get("file_sha256") != file_sha256(output)
        ):
            raise ProtocolError(f"existing bridge arm identity changed: {sample_id}/{arm}")
        return prior
    if output.exists() or sidecar.exists():
        raise ProtocolError(f"partial bridge arm cannot be resumed: {sample_id}/{arm}")
    pcm, construction = construct_arm(natural, local_target, cloud_target, arm)
    write_pcm16(output, pcm)
    generated, generated_meta = read_pcm16(output)
    if generated.size != natural.size:
        raise ProtocolError(f"bridge arm changed natural sample count: {sample_id}/{arm}")
    row = {
        "schema_version": 1,
        "stage_id": "03_bridge",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "arm": arm,
        **expected_inputs,
        "output": str(output.resolve()),
        "file_sha256": generated_meta["file_sha256"],
        "decoded_pcm_sha256": generated_meta["decoded_pcm_sha256"],
        "sample_count": int(generated.size),
        "format": generated_meta,
        "construction": construction,
        "waveform_qc": waveform_qc(pcm16_to_float(generated)),
    }
    write_self_hashed_json(sidecar, row)
    return row


def run_bridge_stage(
    run_root: Path,
    cohort: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    prepare_support: bool = True,
) -> dict[str, Any]:
    if cohort.get("status") != "complete" or len(cohort.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("cohort is incomplete before bridge stage")
    target_by_provider = _target_rows(targets)
    paths = config.RunPaths(run_root)
    paths.bridge.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = file_sha256(paths.protocol / "setup.json")
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        try:
            natural, natural_meta = read_pcm16(Path(str(record["natural_audio"]["path"])))
            local_row = target_by_provider["LOCAL"][sample_id]
            cloud_row = target_by_provider["CLOUD"][sample_id]
            local, local_meta = read_pcm16(Path(str(local_row["target_audio"])))
            cloud, cloud_meta = read_pcm16(Path(str(cloud_row["target_audio"])))
            if not (natural.size == local.size == cloud.size == int(record["natural_audio"]["sample_count"])):
                raise ProtocolError(f"bridge inputs do not share N length: {sample_id}")
            arm_rows: list[dict[str, Any]] = []
            n_output = paths.bridge / "audio" / "N" / f"{sample_id}.wav"
            if n_output.is_file():
                if file_sha256(n_output) != natural_meta["file_sha256"]:
                    raise ProtocolError(f"resumed N bytes changed: {sample_id}")
            else:
                copy_verified(record["natural_audio"]["path"], n_output, natural_meta["file_sha256"])
            n_generated, n_generated_meta = read_pcm16(n_output)
            n_row = {
                "schema_version": 1,
                "stage_id": "03_bridge",
                "protocol_id": config.PROTOCOL_ID,
                "protocol_sha256": protocol_sha256,
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "arm": "N",
                "output": str(n_output.resolve()),
                "file_sha256": n_generated_meta["file_sha256"],
                "decoded_pcm_sha256": n_generated_meta["decoded_pcm_sha256"],
                "natural_audio_sha256": natural_meta["file_sha256"],
                "local_target_sha256": local_meta["file_sha256"],
                "cloud_target_sha256": cloud_meta["file_sha256"],
                "construction": {"construction": "exact_copy_of_natural_file"},
                "sample_count": int(n_generated.size),
                "waveform_qc": waveform_qc(pcm16_to_float(n_generated)),
            }
            n_sidecar = n_output.with_suffix(".json")
            if n_sidecar.is_file():
                prior = verify_self_hashed_json(n_sidecar)
                if (
                    prior.get("protocol_id") != config.PROTOCOL_ID
                    or prior.get("sample_id") != sample_id
                    or prior.get("arm") != "N"
                    or prior.get("file_sha256") != n_generated_meta["file_sha256"]
                    or prior.get("natural_audio_sha256") != natural_meta["file_sha256"]
                    or prior.get("decoded_pcm_sha256") != n_generated_meta["decoded_pcm_sha256"]
                ):
                    raise ProtocolError(f"existing N sidecar identity changed: {sample_id}")
                n_row = prior
            else:
                write_self_hashed_json(n_sidecar, n_row)
            arm_rows.append(n_row)
            for arm in ("B0", "B_LOCAL", "B_CLOUD"):
                arm_rows.append(
                    _write_or_resume_arm(
                        natural=natural,
                        local_target=local,
                        cloud_target=cloud,
                        record=record,
                        arm=arm,
                        output_stage=paths.bridge,
                        protocol_sha256=protocol_sha256,
                        natural_sha256=natural_meta["file_sha256"],
                        local_sha256=local_meta["file_sha256"],
                        cloud_sha256=cloud_meta["file_sha256"],
                    )
                )
            natural_mel = wav2lip_mels(pcm16_to_float(natural))
            local_mel = wav2lip_mels(pcm16_to_float(local))
            cloud_mel = wav2lip_mels(pcm16_to_float(cloud))
            bridge_metrics: dict[str, Any] = {}
            for provider, target_mel, arm in (
                ("LOCAL", local_mel, "B_LOCAL"),
                ("CLOUD", cloud_mel, "B_CLOUD"),
            ):
                candidate_row = next(row for row in arm_rows if row["arm"] == arm)
                candidate, _ = read_pcm16(Path(str(candidate_row["output"])))
                candidate_mel = wav2lip_mels(pcm16_to_float(candidate))
                bridge_metrics[provider] = directional_metrics(natural_mel, target_mel, candidate_mel)
            diagnostics.append(
                {
                    "sample_id": sample_id,
                    "source_group": str(record["source_group"]),
                    "natural_mel_shape": list(natural_mel.shape),
                    "local_target_mel_shape": list(local_mel.shape),
                    "cloud_target_mel_shape": list(cloud_mel.shape),
                    "target_distance_local_cloud": float(np.linalg.norm(local_mel - cloud_mel)),
                    "waveform": {
                        "natural": waveform_qc(pcm16_to_float(natural)),
                        "local_target": waveform_qc(pcm16_to_float(local)),
                        "cloud_target": waveform_qc(pcm16_to_float(cloud)),
                    },
                    "arms": {
                        "LOCAL": bridge_metrics["LOCAL"],
                        "CLOUD": bridge_metrics["CLOUD"],
                    },
                }
            )
            rows.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "arms": arm_rows})
        except Exception as exc:  # noqa: BLE001 - stage failure must be recorded per sample
            failures.append({"sample_id": sample_id, "error_type": type(exc).__name__, "error": str(exc)})
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT
    manifest = {
        "schema_version": 1,
        "stage_id": "03_bridge",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "arm_count": len(config.ARMS),
        "arms": list(config.ARMS),
        "rows": rows,
        "failures": failures,
        "targets_manifest_sha256": file_sha256(paths.targets / "targets_manifest.json"),
        "setup_sha256": protocol_sha256,
    }
    diagnostics_payload = {
        "schema_version": 1,
        "stage_id": "03_bridge",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete and len(diagnostics) == config.EXPECTED_RECORD_COUNT else "blocked",
        "record_count": len(diagnostics),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "frozen_before_scoring": True,
        "score_read_count": 0,
        "movement_threshold": config.MOVEMENT_THRESHOLD,
        "rows": diagnostics,
    }
    write_self_hashed_json(paths.bridge / "audio_manifest.json", manifest)
    write_self_hashed_json(paths.bridge / "diagnostics.json", diagnostics_payload)
    write_self_hashed_json(
        paths.bridge / "failures.json",
        {"schema_version": 1, "stage_id": "03_bridge", "protocol_id": config.PROTOCOL_ID, "status": manifest["status"], "failures": failures},
    )
    if not complete:
        raise ProtocolError(f"bridge audio incomplete: {len(rows)}/{config.EXPECTED_RECORD_COUNT}")
    if prepare_support:
        from .support import freeze_support

        freeze_support(run_root, cohort)
    return manifest
