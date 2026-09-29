from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config
from .common import (
    ProtocolError,
    read_pcm16,
    verify_self_hashed_json,
    write_pcm16,
    write_self_hashed_json,
)


def pcm16_to_float(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ProtocolError("PCM input must be one-dimensional int16")
    return values.astype(np.float64) / 32768.0


def float_to_pcm16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < config.MIN_AUDIO_SAMPLES or not np.isfinite(values).all():
        raise ProtocolError("waveform is empty or non-finite")
    if float(np.abs(values).max()) >= 1.0:
        raise ProtocolError("waveform is clipped before PCM conversion")
    return np.rint(values * 32768.0).clip(-32768, 32767).astype("<i2")


def quarter_swap(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ProtocolError("LOCAL_SWAP input must be one-dimensional int16")
    length = int(values.size)
    if length < config.MIN_AUDIO_SAMPLES:
        raise ProtocolError("LOCAL_SWAP input is too short")
    b1, b2, b3 = length // 4, length // 2, (3 * length) // 4
    result = np.concatenate((values[:b1], values[b2:b3], values[b1:b2], values[b3:])).astype("<i2")
    if result.size != values.size:
        raise ProtocolError("LOCAL_SWAP changed sample count")
    return result, {"length": length, "b1": b1, "b2": b2, "b3": b3, "order": [0, 2, 1, 3]}


def delayed_audio(values: np.ndarray, delay_samples: int = 3200) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1 or values.size <= delay_samples:
        raise ProtocolError("cannot construct fixed-length delayed audio")
    result = np.concatenate((np.zeros(delay_samples, dtype="<i2"), values[:-delay_samples])).astype("<i2")
    if result.size != values.size:
        raise ProtocolError("delayed audio changed sample count")
    return result, {"delay_samples": delay_samples, "delay_frames": delay_samples // config.SAMPLES_PER_FRAME, "front_zero_samples": delay_samples, "tail_truncated_samples": delay_samples}


def _periodic_hann() -> torch.Tensor:
    return torch.hann_window(1024, periodic=True, dtype=torch.float64, device="cpu")


def stft_roundtrip_alpha0(natural: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct natural audio with the historical bridge STFT contract and alpha=0."""
    natural_f = pcm16_to_float(natural)
    window = _periodic_hann()
    tensor = torch.from_numpy(natural_f).to(dtype=torch.float64)
    spec = torch.stft(
        tensor,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    magnitude = torch.abs(spec).clamp_min(1e-7)
    phase = spec / magnitude
    reconstructed = torch.istft(
        magnitude * phase,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        center=True,
        length=natural_f.size,
    ).numpy()
    if not np.isfinite(reconstructed).all():
        raise ProtocolError("RT reconstruction is non-finite")
    natural_rms = float(np.sqrt(np.mean(natural_f * natural_f)))
    reconstructed_rms = float(np.sqrt(np.mean(reconstructed * reconstructed)))
    if natural_rms <= 0.0 or reconstructed_rms <= 0.0:
        raise ProtocolError("RT reconstruction has zero RMS")
    rms_scale = natural_rms / reconstructed_rms
    reconstructed = reconstructed * rms_scale
    peak_before = float(np.abs(reconstructed).max())
    peak_scale = 1.0
    if peak_before >= 0.999:
        peak_scale = 0.999 / peak_before
        reconstructed *= peak_scale
    if float(np.abs(reconstructed).max()) >= 1.0:
        raise ProtocolError("RT reconstruction remains clipped")
    pcm = float_to_pcm16(reconstructed)
    return pcm, {
        "construction": "natural_alpha0_stft_roundtrip",
        "alpha": 0.0,
        "stft": {
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "window": "periodic_hann",
            "center": True,
            "pad_mode": "reflect",
            "magnitude_floor": 1e-7,
            "dtype": "float64",
            "device": "cpu",
        },
        "phase_policy": "natural_phase",
        "natural_rms": natural_rms,
        "reconstructed_rms_before_scaling": reconstructed_rms,
        "rms_scale": float(rms_scale),
        "peak_before_attenuation": peak_before,
        "peak_scale": float(peak_scale),
        "sample_count": int(natural.size),
        "max_abs_pcm_error": int(np.max(np.abs(pcm.astype(np.int32) - natural.astype(np.int32)))),
        "pcm_l2_error": float(np.linalg.norm(pcm.astype(np.float64) - natural.astype(np.float64))),
    }


def _materialize(path: Path, expected: np.ndarray) -> dict[str, Any]:
    if path.exists():
        actual, meta = read_pcm16(path)
        if not np.array_equal(actual, expected):
            raise ProtocolError(f"existing audio differs from frozen construction: {path}")
        return {"path": str(path.resolve()), **meta}
    return write_pcm16(path, expected)


def _arm_row(path: Path, arm: str, source: Mapping[str, Any], construction: Mapping[str, Any]) -> dict[str, Any]:
    values, meta = read_pcm16(path)
    return {
        "arm": arm,
        "path": str(path.resolve()),
        "container_sha256": meta["container_sha256"],
        "pcm_sha256": meta["pcm_sha256"],
        "sample_count": int(values.size),
        "format": {key: meta[key] for key in ("sample_rate", "channels", "sample_width")},
        "source": dict(source),
        "construction": dict(construction),
    }


def _audio_source(row: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    return row["audio"][arm]


def prepare_audio(paths: config.RunPaths, inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(inputs["records"], 1):
        sid = str(item["sample_id"])
        natural, natural_meta = read_pcm16(Path(str(item["audio"]["N"]["path"])))
        bridge, bridge_meta = read_pcm16(Path(str(item["audio"]["B"]["path"])))
        swap, swap_meta = read_pcm16(Path(str(item["audio"]["S"]["path"])))
        if natural.size != bridge.size or natural.size != swap.size:
            raise ProtocolError(f"historical audio lengths differ: {sid}")
        expected_swap, swap_meta_check = quarter_swap(natural)
        if not np.array_equal(expected_swap, swap):
            raise ProtocolError(f"historical LOCAL_SWAP PCM differs from registered quarter swap: {sid}")

        rt_pcm, rt_construction = stft_roundtrip_alpha0(natural)
        nd_pcm, nd_construction = delayed_audio(natural)
        sample_dir = paths.audio_dir / sid
        arms: dict[str, dict[str, Any]] = {}
        for arm, values, source, construction in (
            ("N", natural, {"kind": "historical_N", "path": item["audio"]["N"]["path"], "pcm_sha256": natural_meta["pcm_sha256"]}, {"construction": "byte_copy_of_historical_N"}),
            ("N_REPEAT", natural, {"kind": "historical_N", "path": item["audio"]["N"]["path"], "pcm_sha256": natural_meta["pcm_sha256"]}, {"construction": "independent_byte_copy_for_repeat"}),
            ("RT", rt_pcm, {"kind": "derived_from_N"}, rt_construction),
            ("B", bridge, {"kind": "historical_BRIDGE_075", "path": item["audio"]["B"]["path"], "pcm_sha256": bridge_meta["pcm_sha256"]}, {"construction": "byte_copy_of_historical_BRIDGE_075", "alpha": 0.75}),
            ("S", swap, {"kind": "historical_LOCAL_SWAP", "path": item["audio"]["S"]["path"], "pcm_sha256": swap_meta["pcm_sha256"]}, {"construction": "byte_copy_of_historical_LOCAL_SWAP", **swap_meta_check}),
        ):
            path = sample_dir / f"{arm}.wav"
            _materialize(path, values)
            arms[arm] = _arm_row(path, arm, source, construction)
        nd_path = sample_dir / "ND.wav"
        _materialize(nd_path, nd_pcm)
        control = _arm_row(nd_path, "ND", {"kind": "derived_fixed_delay_from_N"}, nd_construction)
        if arms["N"]["pcm_sha256"] != natural_meta["pcm_sha256"] or arms["N_REPEAT"]["pcm_sha256"] != natural_meta["pcm_sha256"]:
            raise ProtocolError(f"N/N_REPEAT PCM identity changed: {sid}")
        rows.append({
            "sample_id": sid,
            "source_group": str(item["source_group"]),
            "arms": arms,
            "controls": {"ND": control},
            "natural_sample_count": int(natural.size),
            "quarter_boundaries": swap_meta_check,
            "delay": nd_construction,
            "rt": rt_construction,
        })
        print(f"AUDIO {index}/{len(inputs['records'])} {sid}", flush=True)

    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": len(rows),
        "arms": list(config.ARMS),
        "control_audio": ["ND"],
        "rows": rows,
        "source_inputs_sha256": inputs["artifact_sha256"],
        "decoded_pcm_contract": "16kHz mono PCM16 little-endian; hashes are decoded PCM hashes",
    }
    write_self_hashed_json(paths.audio_manifest, payload)
    return verify_self_hashed_json(paths.audio_manifest)


def load_audio_row(manifest: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    rows = [row for row in manifest.get("rows", []) if str(row.get("sample_id")) == sample_id]
    if len(rows) != 1:
        raise ProtocolError(f"expected one audio row: {sample_id}")
    return rows[0]
