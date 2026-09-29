from __future__ import annotations

import hashlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config
from .common import ExperimentError, bytes_sha256, file_sha256, read_pcm16_wav


@contextmanager
def _wav2lip_audio_module():
    root = config.REPO / "third_party/Wav2Lip"
    module_name = "_wav2lip_spectral_audio"
    old_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(module_name, root / "audio.py")
        if spec is None or spec.loader is None:
            raise ExperimentError("cannot load Wav2Lip audio helper")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = old_path
        sys.modules.pop(module_name, None)


def _finite(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ExperimentError(f"{label} is not finite mono audio")
    return array


def _pcm_bytes(values: np.ndarray) -> bytes:
    rounded = np.rint(np.asarray(values, dtype=np.float64) * 32768.0)
    if not np.isfinite(rounded).all():
        raise ExperimentError("PCM rounding produced non-finite values")
    return np.clip(rounded, -32768, 32767).astype("<i2", copy=False).tobytes()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _transform(natural: np.ndarray, mfa: np.ndarray, arm: str) -> tuple[bytes, dict[str, Any]]:
    if natural.shape != mfa.shape:
        raise ExperimentError("natural and MFA-linear PCM lengths differ")
    n = torch.from_numpy(natural.astype(np.float64, copy=False))
    m = torch.from_numpy(mfa.astype(np.float64, copy=False))
    window = torch.hann_window(config.STFT_WIN_LENGTH, periodic=True, dtype=torch.float64)
    sn = torch.stft(
        n,
        n_fft=config.STFT_NFFT,
        hop_length=config.STFT_HOP_LENGTH,
        win_length=config.STFT_WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    sm = torch.stft(
        m,
        n_fft=config.STFT_NFFT,
        hop_length=config.STFT_HOP_LENGTH,
        win_length=config.STFT_WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    mn = sn.abs().clamp_min(config.MAGNITUDE_FLOOR)
    mm = sm.abs().clamp_min(config.MAGNITUDE_FLOOR)
    ln = mn.log()
    lm = mm.log()
    correction = lm - ln
    if arm == config.ARM_RT:
        target_log = ln
    elif arm == config.ARM_MAG:
        target_log = ln + config.ALPHA * correction
    elif arm == config.ARM_ENV:
        target_log = ln + config.ALPHA * correction.mean(dim=-1, keepdim=True)
    else:
        raise ExperimentError(f"unknown transformed arm: {arm}")
    phase = sn / mn
    rebuilt = torch.istft(
        target_log.exp() * phase,
        n_fft=config.STFT_NFFT,
        hop_length=config.STFT_HOP_LENGTH,
        win_length=config.STFT_WIN_LENGTH,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        length=int(natural.size),
    )
    result = rebuilt.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(result).all() or result.shape != natural.shape:
        raise ExperimentError(f"{arm} ISTFT output is invalid")
    natural_rms = _rms(natural)
    before_rms = _rms(result)
    if natural_rms <= config.RMS_EPSILON or before_rms <= config.RMS_EPSILON:
        raise ExperimentError(f"{arm} has zero RMS")
    rms_scale = natural_rms / before_rms
    result = result * rms_scale
    before_peak = float(np.max(np.abs(result)))
    peak_scale = 1.0 if before_peak < 0.999 else 0.999 / before_peak
    result = result * peak_scale
    pcm = _pcm_bytes(result)
    decoded = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    if decoded.size != natural.size:
        raise ExperimentError(f"{arm} PCM length changed")
    metadata = {
        "construction": f"STFT/ISTFT {arm}; natural phase; alpha={config.ALPHA if arm != config.ARM_RT else 0.0}",
        "alpha": 0.0 if arm == config.ARM_RT else config.ALPHA,
        "stft": {
            "n_fft": config.STFT_NFFT,
            "win_length": config.STFT_WIN_LENGTH,
            "hop_length": config.STFT_HOP_LENGTH,
            "window": "periodic_hann",
            "center": True,
            "pad_mode": "reflect",
            "normalized": False,
            "onesided": True,
            "magnitude_floor": config.MAGNITUDE_FLOOR,
            "frames": int(sn.shape[-1]),
        },
        "length": int(decoded.size),
        "natural_rms": natural_rms,
        "reconstructed_rms_before_scaling": before_rms,
        "rms_scale": float(rms_scale),
        "peak_before_attenuation": before_peak,
        "peak_scale": float(peak_scale),
        "rms_after_pcm": _rms(decoded),
        "peak_after_pcm": float(np.max(np.abs(decoded))),
        "dc_after_pcm": float(np.mean(decoded)),
        "pcm_sha256": bytes_sha256(pcm),
        "finite": True,
        "saturated": bool(np.max(np.abs(np.frombuffer(pcm, dtype="<i2"))) >= 32767),
    }
    if metadata["saturated"]:
        raise ExperimentError(f"{arm} PCM unexpectedly saturated")
    return pcm, metadata


def construct_candidates(natural_pcm: bytes, mfa_pcm: bytes) -> dict[str, tuple[bytes, dict[str, Any]]]:
    if len(natural_pcm) % 2 or len(mfa_pcm) % 2 or not natural_pcm or len(natural_pcm) != len(mfa_pcm):
        raise ExperimentError("natural/MFA PCM alignment or length contract failed")
    natural = _finite(np.frombuffer(natural_pcm, dtype="<i2").astype(np.float64) / 32768.0, "natural")
    mfa = _finite(np.frombuffer(mfa_pcm, dtype="<i2").astype(np.float64) / 32768.0, "MFA-linear")
    n_meta = {
        "construction": "byte-identical frozen natural PCM16",
        "length": int(natural.size),
        "pcm_sha256": bytes_sha256(natural_pcm),
        "rms_after_pcm": _rms(natural),
        "peak_after_pcm": float(np.max(np.abs(natural))),
        "dc_after_pcm": float(np.mean(natural)),
        "finite": True,
        "saturated": bool(np.max(np.abs(np.frombuffer(natural_pcm, dtype="<i2"))) >= 32767),
    }
    if n_meta["saturated"]:
        raise ExperimentError("natural PCM is saturated")
    result: dict[str, tuple[bytes, dict[str, Any]]] = {
        config.ARM_N: (bytes(natural_pcm), n_meta),
        config.ARM_N_REPEAT: (bytes(natural_pcm), {**n_meta, "construction": "independent second generation uses the same frozen N PCM"}),
    }
    for arm in (config.ARM_RT, config.ARM_MAG, config.ARM_ENV):
        result[arm] = _transform(natural, mfa, arm)
    return result


def candidate_diagnostics(audio_paths: dict[str, Path], natural_path: Path, mfa_path: Path) -> dict[str, Any]:
    """Compute pre-score Wav2Lip-mel diagnostics without filtering records."""
    try:
        from scipy import signal
        from scipy.io import wavfile

        def mel_basis() -> np.ndarray:
            def hz_to_mel(value: np.ndarray) -> np.ndarray:
                return 2595.0 * np.log10(1.0 + value / 700.0)

            def mel_to_hz(value: np.ndarray) -> np.ndarray:
                return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

            frequencies = np.fft.rfftfreq(800, d=1.0 / config.SAMPLE_RATE)
            points = mel_to_hz(np.linspace(hz_to_mel(np.asarray([55.0]))[0], hz_to_mel(np.asarray([7600.0]))[0], 82))
            basis = np.zeros((80, frequencies.size), dtype=np.float64)
            for index in range(80):
                left, center, right = points[index : index + 3]
                basis[index] = np.maximum(0.0, np.minimum((frequencies - left) / max(center - left, 1e-12), (right - frequencies) / max(right - center, 1e-12)))
            return basis

        basis = mel_basis()

        def mel_for(path: Path) -> np.ndarray:
            sample_rate, raw = wavfile.read(str(path))
            if int(sample_rate) != config.SAMPLE_RATE:
                raise ExperimentError(f"unexpected mel sample rate: {path}")
            waveform = np.asarray(raw, dtype=np.float64) / 32768.0
            if waveform.ndim != 1:
                raise ExperimentError(f"mel source is not mono: {path}")
            emphasized = signal.lfilter([1.0, -0.97], [1.0], waveform)
            frame_count = 1 + emphasized.size // 200
            padded = np.pad(emphasized, (400, 400), mode="constant")
            required = (frame_count - 1) * 200 + 800
            if padded.size < required:
                padded = np.pad(padded, (0, required - padded.size), mode="constant")
            starts = np.arange(frame_count, dtype=np.int64) * 200
            frames = np.stack([padded[start : start + 800] for start in starts], axis=0)
            window = signal.get_window("hann", 800, fftbins=True)
            spectrum = np.fft.rfft(frames * window[None, :], n=800, axis=1).T
            magnitude = np.dot(basis, np.abs(spectrum))
            minimum = np.exp(-100.0 / 20.0 * np.log(10.0))
            db = 20.0 * np.log10(np.maximum(minimum, magnitude)) - 20.0
            normalized = np.clip(8.0 * ((db + 100.0) / 100.0) - 4.0, -4.0, 4.0)
            return np.asarray(normalized, dtype=np.float64)

        mel: dict[str, np.ndarray] = {}
        for arm, path in {config.ARM_N: natural_path, config.ARM_MAG: audio_paths[config.ARM_MAG], config.ARM_ENV: audio_paths[config.ARM_ENV], config.ARM_RT: audio_paths[config.ARM_RT], "M": mfa_path}.items():
            value = mel_for(path)
            if value.ndim != 2 or not np.isfinite(value).all():
                raise ExperimentError(f"invalid Wav2Lip mel for {arm}")
            mel[arm] = value
        shape = {arm: [int(x) for x in value.shape] for arm, value in mel.items()}
        if len({tuple(value) for value in shape.values()}) != 1:
            raise ExperimentError(f"Wav2Lip mel shapes differ: {shape}")
        delta = mel["M"] - mel[config.ARM_N]
        denom = float(np.sum(delta * delta))
        diagnostics: dict[str, Any] = {"mel_shape": shape, "records_not_filtered": True, "arms": {}}
        for arm in (config.ARM_N, config.ARM_RT, config.ARM_MAG, config.ARM_ENV):
            displacement = mel[arm] - mel[config.ARM_N]
            progress = float(np.sum(displacement * delta) / denom) if denom > 0 else 0.0
            residual = displacement - progress * delta
            orthogonal_ratio = float(np.linalg.norm(residual) / max(np.linalg.norm(displacement), 1e-12))
            diagnostics["arms"][arm] = {
                "mae_to_N": float(np.mean(np.abs(displacement))),
                "mae_to_M": float(np.mean(np.abs(mel[arm] - mel["M"]))),
                "progress_to_M_minus_N": progress,
                "orthogonal_ratio": orthogonal_ratio,
            }
        return diagnostics
    except (ImportError, ModuleNotFoundError) as exc:
        raise ExperimentError(f"Wav2Lip mel dependencies unavailable: {exc}") from exc


def write_candidate_wav(path: Path, pcm: bytes) -> dict[str, Any]:
    if path.exists():
        raise ExperimentError(f"refusing to overwrite candidate audio: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    from scripts.experiments.wav2lip_roi_retiming_oracle.common import write_pcm16_wav

    return write_pcm16_wav(path, pcm)


def hash_audio_file(path: Path) -> dict[str, Any]:
    pcm, values, params = read_pcm16_wav(path)
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "decoded_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "sample_count": int(values.size),
        "sample_rate": int(params["sample_rate"]),
        "channels": int(params["channels"]),
        "sample_width": int(params["sample_width"]),
    }
