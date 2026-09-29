from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from . import config
from .audio import pcm16_to_float, read_pcm16
from .common import file_sha256, write_self_hashed_json
from .protocol import ProtocolError


def wav2lip_mels(waveform: np.ndarray, wav2lip_root: Path = config.WAV2LIP_ROOT) -> np.ndarray:
    root = str(wav2lip_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import audio

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size < 1024 or not np.isfinite(values).all():
        raise ValueError("Wav2Lip mel input is empty or non-finite")
    result = np.asarray(audio.melspectrogram(values), dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != 80 or not np.isfinite(result).all():
        raise ValueError(f"unexpected Wav2Lip mel shape: {result.shape}")
    return result


def _same_length(left: np.ndarray, right: np.ndarray, minimum: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size:
        raise ProtocolError(f"fidelity requires exact equal sample counts: {left.size} != {right.size}")
    if left.size < minimum or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ProtocolError("fidelity input is empty or non-finite")
    return left, right


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left, right = _same_length(left, right, minimum=1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = _same_length(reference, estimate)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    denominator = float(np.dot(reference, reference))
    if denominator == 0.0:
        return 0.0
    target = (float(np.dot(estimate, reference)) / denominator) * reference
    noise = estimate - target
    return float(10.0 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def _log_stft_distance(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = _same_length(reference, estimate)
    distances: list[float] = []
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        ref_spec = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop, win_length=n_fft, center=False))
        est_spec = np.abs(librosa.stft(estimate, n_fft=n_fft, hop_length=hop, win_length=n_fft, center=False))
        distances.append(float(np.mean(np.abs(np.log(ref_spec + 1e-7) - np.log(est_spec + 1e-7)))))
    return float(np.mean(distances))


def _f0_and_voicing(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
    f0 = librosa.yin(waveform, fmin=65.0, fmax=1000.0, sr=config.SAMPLE_RATE, frame_length=1024, hop_length=256)
    voiced = np.isfinite(f0) & (f0 > 0.0)
    return np.where(voiced, f0, 0.0), voiced


def _f0_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    ref_f0, ref_voiced = _f0_and_voicing(reference)
    est_f0, est_voiced = _f0_and_voicing(estimate)
    count = min(ref_f0.size, est_f0.size)
    ref_f0, est_f0 = ref_f0[:count], est_f0[:count]
    ref_voiced, est_voiced = ref_voiced[:count], est_voiced[:count]
    both = ref_voiced & est_voiced
    return {
        "voiced_frame_count_reference": int(ref_voiced.sum()),
        "voiced_frame_count_estimate": int(est_voiced.sum()),
        "f0_abs_error_hz": float(np.mean(np.abs(ref_f0[both] - est_f0[both]))) if both.any() else 0.0,
        "voiced_unvoiced_disagreement": float(np.mean(ref_voiced != est_voiced)),
        "compared_frames": int(count),
    }


def _energy_envelope(waveform: np.ndarray) -> np.ndarray:
    return np.asarray(librosa.feature.rms(y=waveform, frame_length=400, hop_length=160, center=False)[0], dtype=np.float64)


def _energy_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = _energy_envelope(reference)
    est = _energy_envelope(estimate)
    count = min(ref.size, est.size)
    return _correlation(ref[:count], est[:count])


def pair_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    reference, estimate = _same_length(reference, estimate)
    ref_mel = wav2lip_mels(reference)
    est_mel = wav2lip_mels(estimate)
    if ref_mel.shape != est_mel.shape:
        raise ProtocolError(f"Wav2Lip mel shapes differ: {ref_mel.shape} != {est_mel.shape}")
    f0 = _f0_metrics(reference, estimate)
    return {
        "sample_count": int(reference.size),
        "wav2lip_mel_shape": list(ref_mel.shape),
        "mel_L1": float(np.mean(np.abs(ref_mel - est_mel))),
        "log_stft_distance": _log_stft_distance(reference, estimate),
        "si_sdr_db": si_sdr(reference, estimate),
        "zero_lag_waveform_correlation": _correlation(reference, estimate),
        "rms_ratio_estimate_over_reference": float(np.sqrt(np.mean(estimate ** 2)) / (np.sqrt(np.mean(reference ** 2)) + 1e-12)),
        "peak_ratio_estimate_over_reference": float(np.max(np.abs(estimate)) / (np.max(np.abs(reference)) + 1e-12)),
        "energy_envelope_correlation": _energy_correlation(reference, estimate),
        "f0": f0,
        "clipped_reference_count": int(np.sum(np.abs(reference) >= 1.0)),
        "clipped_estimate_count": int(np.sum(np.abs(estimate) >= 1.0)),
        "reference_dc_offset": float(np.mean(reference)),
        "estimate_dc_offset": float(np.mean(estimate)),
        "lag_search": False,
        "gain_alignment": False,
        "phase_alignment": False,
        "phone_boundary_diagnostics": {"available": False, "reason": "no bound phone-boundary metadata in frozen parent cohort"},
    }


def run_stage02(cohort: Mapping[str, Any], audio_manifest: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    if audio_manifest.get("status") != "complete" or len(audio_manifest.get("rows", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("Stage 01 is incomplete")
    rows: list[dict[str, Any]] = []
    for record, audio_row in zip(cohort["records"], audio_manifest["rows"]):
        if str(record["sample_id"]) != str(audio_row["sample_id"]):
            raise ProtocolError("cohort and DAC audio order differs")
        natural_int16, natural_meta = read_pcm16(Path(str(record["natural_audio"])))
        wavlm_int16, wavlm_meta = read_pcm16(Path(str(record["wavlm_audio"])))
        dac_info = audio_row["dac"]
        dac_path = Path(str(dac_info["output_path"]))
        if file_sha256(dac_path) != dac_info["output_sha256"]:
            raise ProtocolError(f"DAC output hash changed: {record['sample_id']}")
        dac_int16, dac_meta = read_pcm16(dac_path)
        if natural_int16.size != wavlm_int16.size or natural_int16.size != dac_int16.size:
            raise ProtocolError(f"audio sample counts differ: {record['sample_id']}")
        natural = pcm16_to_float(natural_int16)
        wavlm = pcm16_to_float(wavlm_int16)
        dac = pcm16_to_float(dac_int16)
        wavlm_metrics = pair_metrics(natural, wavlm)
        dac_metrics = pair_metrics(natural, dac)
        rows.append({
            "sample_id": str(record["sample_id"]),
            "source_group": str(record["source_group"]),
            "natural_audio_sha256": natural_meta["pcm_sha256"],
            "wavlm_audio_sha256": wavlm_meta["pcm_sha256"],
            "dac_audio_sha256": dac_meta["pcm_sha256"],
            "natural_sample_count": int(natural.size),
            "wavlm": wavlm_metrics,
            "dac": dac_metrics,
            "mel_improvement": float(wavlm_metrics["mel_L1"] - dac_metrics["mel_L1"]),
            "metric_config_sha256": hashlib.sha256(b"wav2lip80;stft512-128,1024-256,2048-512;yin65-1000-1024-256;rms400-160;zero-lag").hexdigest(),
        })
        print(f"FIDELITY {len(rows)}/{config.EXPECTED_RECORD_COUNT} {record['sample_id']}", flush=True)
    summary = {
        "schema_version": 1,
        "stage_id": "02_fidelity",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if len(rows) == config.EXPECTED_RECORD_COUNT else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "rows": rows,
        "metric_policy": {"sample_grid": "untouched_exact_pcm16_grid", "lag_search": False, "gain_alignment": False, "phase_alignment": False, "waveform_modification": False},
        "mel_improvement": {"definition": "mel_L1(N,W)-mel_L1(N,D)", "mean": float(np.mean([row["mel_improvement"] for row in rows])) if rows else None},
        "audio_manifest_sha256": file_sha256(config.STAGES["01_audio"] / "audio_manifest.json"),
    }
    write_self_hashed_json(output_stage / "fidelity.json", summary)
    if summary["status"] != "complete":
        raise ProtocolError("Stage 02 is incomplete")
    return summary
