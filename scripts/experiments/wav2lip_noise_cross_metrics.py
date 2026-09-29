"""Pure protocol primitives for ``wav2lip_noise_cross_v1``.

The runner and the independent checker both use these small, deterministic
operations.  Nothing in this module loads a model or touches a media file
other than the explicitly requested PCM helpers; keeping the arithmetic here
small makes the manipulation and statistics contracts easy to test.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.tts_native_gain_attribution.audio import (
    active_rms,
    frame_rms_and_activity,
    low_energy_noise_power,
    noise_realization,
    pcm16_to_float,
    quantize_pcm16,
)
from scripts.experiments.tts_native_gain_attribution.common import ProtocolError


SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
LAG_COUNT = 31
VSHIFT = 15
EMBEDDING_DIM = 1024
MIN_INTERIOR_ROWS = 25
BOOTSTRAP_SEED = 20_260_920
BOOTSTRAP_DRAWS = 20_000
PRACTICAL_THRESHOLD = 0.200


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_pcm(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.int16 or array.ndim != 1 or array.size < 2:
        raise ProtocolError(f"{label} must be a non-empty one-dimensional int16 array")
    if not np.isfinite(array.astype(np.float64)).all():
        raise ProtocolError(f"{label} contains non-finite samples")
    return np.asarray(array, dtype="<i2")


def _max_abs(*arrays: np.ndarray) -> float:
    maximum = max(float(np.max(np.abs(np.asarray(item, dtype=np.float64)))) for item in arrays)
    return maximum


def construct_audio_conditions(
    natural_pcm: np.ndarray,
    tts_pcm: np.ndarray,
    sample_id: int,
    *,
    snr_db: float = 20.0,
) -> dict[str, Any]:
    """Construct RAW/A0/NOISE20 for both arms of one source.

    ``g`` is shared by all four manipulated arrays of an ID.  RAW remains the
    exact decoded carrier, while A0 and NOISE20 are quantized once after the
    common headroom calculation.  The returned metadata is deliberately
    JSON-compatible so it can be written directly into the audit manifest.
    """

    if not np.isfinite(float(snr_db)) or float(snr_db) <= 0.0:
        raise ProtocolError("snr_db must be finite and positive")
    raw_n = _require_pcm(natural_pcm, "natural_pcm")
    raw_t = _require_pcm(tts_pcm, "tts_pcm")
    x_n = pcm16_to_float(raw_n)
    x_t = pcm16_to_float(raw_t)
    natural_power, power_meta = low_energy_noise_power(x_n)
    results: dict[str, dict[str, Any]] = {}
    float_records: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for source, waveform in (("N", x_n), ("T", x_t)):
        _, _, mask = frame_rms_and_activity(waveform)
        if int(np.count_nonzero(mask)) < SAMPLE_RATE:
            raise ProtocolError(f"{source} audio has less than one second of active samples")
        noise, noise_meta = noise_realization(
            waveform.size, natural_power, sample_id=int(sample_id), source=source
        )
        signal_rms = active_rms(waveform, mask)
        noise_rms = active_rms(noise, mask)
        if signal_rms <= 0.0 or noise_rms <= 0.0:
            raise ProtocolError(f"{source} active signal/noise RMS is not positive")
        beta = signal_rms / (10.0 ** (float(snr_db) / 20.0) * noise_rms)
        candidate = waveform + beta * noise
        float_records[source] = (waveform, candidate, mask)
        results[source] = {
            "source": source,
            "sample_count": int(waveform.size),
            "active_sample_count": int(np.count_nonzero(mask)),
            "mask": mask,
            "noise": noise,
            "beta": float(beta),
            "signal_rms": float(signal_rms),
            "noise_rms": float(noise_rms),
            "noise_meta": noise_meta,
        }
    maximum = _max_abs(
        x_n, x_t, float_records["N"][1], float_records["T"][1]
    )
    g = min(1.0, 0.98 / maximum) if maximum > 0.0 else 1.0
    for source in ("N", "T"):
        item = results[source]
        waveform, candidate, mask = float_records[source]
        a0 = quantize_pcm16(g * waveform)
        noisy = quantize_pcm16(g * candidate)
        if np.any(noisy == -32768) or np.any(noisy == 32767):
            raise ProtocolError(f"{source} NOISE20 contains int16 saturation")
        # Quantization can move the measured value by a tiny amount.  Keep the
        # actual on-disk estimate in the manifest rather than hiding it.
        active = np.asarray(mask, dtype=bool)
        signal = a0.astype(np.float64) / 32768.0
        perturbation = (noisy.astype(np.float64) - a0.astype(np.float64)) / 32768.0
        signal_power = float(np.mean(signal[active] ** 2))
        noise_power = float(np.mean(perturbation[active] ** 2))
        actual_snr = float(10.0 * np.log10(signal_power / noise_power)) if noise_power > 0.0 else float("inf")
        if not 19.8 <= actual_snr <= 20.2:
            raise ProtocolError(f"{source} measured SNR outside [19.8,20.2] dB: {actual_snr}")
        item.update(
            {
                "raw": raw_n if source == "N" else raw_t,
                "a0": a0,
                "noise20": noisy,
                "actual_snr_db": actual_snr,
            }
        )
        # Masks/noise arrays are useful to the runner but are not serialised in
        # the JSON metadata.  They remain available under private keys.
    metadata = {
        "sample_id": int(sample_id),
        "protocol": "wav2lip_noise_cross_v1",
        "snr_db_target": float(snr_db),
        "g": float(g),
        "natural_noise_power": power_meta,
        "arms": {
            source: {
                key: value
                for key, value in item.items()
                if key not in {"mask", "noise", "raw", "a0", "noise20"}
            }
            for source, item in results.items()
        },
    }
    # Keep the actual spectral template available to the independent checker;
    # only its hash and selection metadata belong in the JSON manifest.
    return {"conditions": results, "metadata": metadata, "natural_noise_power_array": natural_power}


# A short alias is convenient in tests and in downstream notebooks.
build_audio_conditions = construct_audio_conditions


def delayed_audio(values: np.ndarray, delay_ms: int = 200) -> np.ndarray:
    """Prepend zero samples and truncate to preserve the original clock."""

    pcm = _require_pcm(values, "audio")
    samples = int(round(float(delay_ms) * SAMPLE_RATE / 1000.0))
    if samples < 0 or samples >= pcm.size:
        raise ProtocolError("delay must be non-negative and shorter than the audio")
    return np.concatenate((np.zeros(samples, dtype="<i2"), pcm[:-samples])) if samples else pcm.copy()


def common_support(
    matrices: Sequence[np.ndarray], *, trim: int = VSHIFT, min_rows: int = MIN_INTERIOR_ROWS
) -> tuple[list[int], str]:
    values = [np.asarray(item) for item in matrices]
    if not values:
        return [], "NO_MATRICES"
    if any(item.ndim != 2 or item.shape[1] != LAG_COUNT for item in values):
        raise ProtocolError("all matrices must have shape [T,31]")
    length = min(int(item.shape[0]) for item in values)
    rows = list(range(int(trim), length - int(trim)))
    if len(rows) < int(min_rows):
        return rows, "INSUFFICIENT_SUPPORT"
    return rows, "PASS"


def curve_metrics(matrix: np.ndarray, support: Sequence[int]) -> dict[str, Any]:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[1] != LAG_COUNT or value.shape[0] < 1:
        raise ProtocolError(f"matrix must have shape [T,31], got {value.shape}")
    if not np.isfinite(value).all():
        raise ProtocolError("matrix contains non-finite values")
    rows = np.asarray(list(support), dtype=np.int64)
    if rows.ndim != 1 or rows.size < 1 or np.any(rows < 0) or np.any(rows >= value.shape[0]) or np.any(np.diff(rows) <= 0):
        raise ProtocolError("support is invalid")
    curve = value[rows].astype(np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
    minimum_index = int(np.argmin(curve))
    minimum = float(curve[minimum_index])
    background = float(np.median(curve))
    return {
        "support_count": int(rows.size),
        "support_rows": [int(item) for item in rows],
        "curve": [float(item) for item in curve],
        "min_index": minimum_index,
        "official_offset": int(VSHIFT - minimum_index),
        "sync_d": minimum,
        "background_b": background,
        "sync_c": float(background - minimum),
        "d0": float(curve[VSHIFT]),
        "boundary_best": bool(minimum_index in (0, LAG_COUNT - 1)),
        "ties": [int(index) for index, item in enumerate(curve) if item == minimum],
    }


def four_cell(q00: float, q01: float, q10: float, q11: float, *, metric: str = "C") -> dict[str, float]:
    values = np.asarray([q00, q01, q10, q11], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ProtocolError("four-cell values must be finite")
    evaluation = float(q01 - q00)
    generation = float(q10 - q00)
    interaction = float(q11 - q10 - q01 + q00)
    total = float(q11 - q00)
    if abs(evaluation + generation + interaction - total) > 1e-10:
        raise ProtocolError("four-cell identity failed")
    return {
        "q00": float(q00),
        "q01": float(q01),
        "q10": float(q10),
        "q11": float(q11),
        "metric": metric,
        "evaluation": evaluation,
        "generation": generation,
        "interaction": interaction,
        "total": total,
    }


def matrix_from_embeddings(
    visual: np.ndarray, audio: np.ndarray, *, vshift: int = VSHIFT
) -> np.ndarray:
    """Independent NumPy reproduction of SyncNet's float32 distance matrix."""

    visual32 = np.asarray(visual, dtype=np.float32)
    audio32 = np.asarray(audio, dtype=np.float32)
    if visual32.ndim != 2 or audio32.ndim != 2 or visual32.shape[1] != EMBEDDING_DIM or audio32.shape[1] != EMBEDDING_DIM:
        raise ProtocolError(f"embedding dimensions are invalid: {visual32.shape}/{audio32.shape}")
    rows = min(int(visual32.shape[0]), int(audio32.shape[0]))
    if rows < 1:
        raise ProtocolError("no common embedding rows")
    padded = np.pad(audio32[:rows], ((int(vshift), int(vshift)), (0, 0)), mode="constant")
    result = np.empty((rows, 2 * int(vshift) + 1), dtype=np.float32)
    for index in range(rows):
        delta = visual32[index : index + 1] - padded[index : index + 2 * int(vshift) + 1]
        # torch.nn.functional.pairwise_distance adds eps to each coordinate.
        result[index] = np.sqrt(
            np.sum(np.square(delta + np.float32(1e-6), dtype=np.float32), axis=1),
            dtype=np.float32,
        )
    return result.astype(np.float64)


def make_bootstrap_indices(labels: Sequence[str], *, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> tuple[list[str], np.ndarray]:
    unique = sorted({str(label) for label in labels})
    if not unique:
        raise ProtocolError("bootstrap requires at least one source group")
    indices = np.random.Generator(np.random.PCG64(int(seed))).integers(
        0, len(unique), size=(int(draws), len(unique)), dtype=np.int64
    )
    return unique, indices


def bootstrap_group_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    primary: bool = False,
    indices: np.ndarray | None = None,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if not np.isfinite(value):
            raise ProtocolError(f"non-finite bootstrap value: {field}")
        grouped[str(row["source_group"])].append(value)
    labels, generated = make_bootstrap_indices(list(grouped), seed=seed, draws=draws)
    sampled_indices = generated if indices is None else np.asarray(indices, dtype=np.int64)
    if sampled_indices.shape != (int(draws), len(labels)):
        raise ProtocolError(f"bootstrap index shape mismatch: {sampled_indices.shape}")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    estimates = means[sampled_indices].mean(axis=1, dtype=np.float64)
    lower95, upper95 = (0.025, 0.975)
    lower_primary, upper_primary = (0.0125, 0.9875) if primary else (lower95, upper95)
    quantile = lambda p: float(np.quantile(estimates, p, method="linear"))
    return {
        "status": "COMPLETE",
        "field": str(field),
        "mean": float(means.mean(dtype=np.float64)),
        "median": float(np.median(means)),
        "ci95": [quantile(lower95), quantile(upper95)],
        "ci_primary": [quantile(lower_primary), quantile(upper_primary)],
        "primary": bool(primary),
        "group_labels": labels,
        "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)},
        "positive_count": int(np.sum(means > 0.0)),
        "negative_count": int(np.sum(means < 0.0)),
        "zero_count": int(np.sum(means == 0.0)),
        "group_count": len(labels),
        "draws": int(draws),
        "seed": int(seed),
        "indices_sha256": hashlib.sha256(sampled_indices.tobytes()).hexdigest(),
    }


def effect_status(summary: Mapping[str, Any], *, primary: bool = True) -> str:
    interval = summary.get("ci_primary" if primary else "ci95")
    if not isinstance(interval, Sequence) or len(interval) != 2:
        return "INCOMPLETE"
    low, high = float(interval[0]), float(interval[1])
    if low > 0.0:
        return "POSITIVE_RESPONSE"
    if high < 0.0:
        return "NEGATIVE_RESPONSE"
    if low >= -PRACTICAL_THRESHOLD and high <= PRACTICAL_THRESHOLD:
        return "WITHIN_PREDECLARED_SCORE_RANGE"
    return "INCONCLUSIVE"


def audio_manipulation_checks(
    raw: np.ndarray, a0: np.ndarray, noisy: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    raw16, a016, noisy16 = (_require_pcm(item, label) for item, label in ((raw, "RAW"), (a0, "A0"), (noisy, "NOISE20")))
    if not (raw16.size == a016.size == noisy16.size == np.asarray(mask).size):
        raise ProtocolError("audio conditions do not share a clock")
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        raise ProtocolError("audio activity mask is empty")
    signal = a016.astype(np.float64) / 32768.0
    perturbation = (noisy16.astype(np.float64) - a016.astype(np.float64)) / 32768.0
    snr = 10.0 * np.log10(np.mean(signal[active] ** 2) / np.mean(perturbation[active] ** 2))
    return {
        "same_length": True,
        "raw_exact_type": bool(np.array_equal(raw16, np.asarray(raw, dtype=np.int16))),
        "snr_db": float(snr),
        "no_saturation": bool(not np.any(noisy16 == -32768) and not np.any(noisy16 == 32767)),
    }


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "EMBEDDING_DIM",
    "LAG_COUNT",
    "MIN_INTERIOR_ROWS",
    "PRACTICAL_THRESHOLD",
    "VSHIFT",
    "audio_manipulation_checks",
    "bootstrap_group_summary",
    "build_audio_conditions",
    "canonical_hash",
    "common_support",
    "construct_audio_conditions",
    "curve_metrics",
    "delayed_audio",
    "effect_status",
    "file_sha256",
    "four_cell",
    "make_bootstrap_indices",
    "matrix_from_embeddings",
]
