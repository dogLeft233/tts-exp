"""Unfrozen V5 local-transition mask scaffold, version 2.

Preparation only. This module declares no candidate defaults and performs no
audio transform at import time. A caller must prepare a candidate mask, obtain
one explicit common projection across the four signed transition cells, and
then apply that same projection. Diagnostics retain the source-clock arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import numpy as np


SCHEMA = "cross_tfg_v5_local_transition_scaffold_v2"
SAMPLE_RATE_HZ = 16000
N_FFT = 512
HOP_LENGTH = 128
FRAME_SAMPLES_40MS = SAMPLE_RATE_HZ // 25
EPS = 1e-8
COMMON_CELL_KEYS = frozenset(
    {
        ("positive_flux", -1.0),
        ("positive_flux", 1.0),
        ("negative_flux", -1.0),
        ("negative_flux", 1.0),
    }
)

EXPECTED_AUDIO_IO_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
EXPECTED_BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
EXPECTED_SHARED_PROFILE_SHA256 = "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253"

# The earlier checkpoint records b344... as a qualified V1 digest. The only
# V1 source file present in this workspace was independently observed as a7c...;
# V2 is derived from those observed bytes. No b344... source bytes were recovered.
PRIOR_RECORDED_V1_SHA256 = "b34450397c5fa18b8cd4f820ae095b356499ecabb96b3c6a814b676642e63cbd"
OBSERVED_V1_SHA256 = "a7c2286ac3fdab82a75b09ca5b0809073ea33ff19fc88d703d5ea3f11f677020"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_float32_sha(samples: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(samples, dtype="<f4"))
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass(frozen=True)
class AudioBackend:
    """Hash-qualified primitives borrowed from the existing audio helpers."""

    stft: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
    istft: Callable[..., np.ndarray]
    apply_global_rms: Callable[..., tuple[np.ndarray, Any]]
    waveform_rms: Callable[[np.ndarray], float]
    gaussian_kernel: Callable[[float], np.ndarray]
    smooth_axis: Callable[[np.ndarray, np.ndarray, int], np.ndarray]
    source_files: Mapping[str, Mapping[str, Any]]


def load_audio_backend(repo_root: Path) -> AudioBackend:
    """Verify exact helper bytes before importing the pinned implementation."""

    root = Path(repo_root)
    expected = {
        "scripts/audio_enhancement/audio_io.py": EXPECTED_AUDIO_IO_SHA256,
        "scripts/audio_enhancement/baselines.py": EXPECTED_BASELINES_SHA256,
        "scripts/audio_enhancement/shared_frequency_profile_v1.py": EXPECTED_SHARED_PROFILE_SHA256,
    }
    source_files: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in expected.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing or nonregular pinned helper: {relative}")
        observed = sha256_file(path)
        if observed != expected_sha:
            raise RuntimeError(f"pinned helper SHA mismatch: {relative}")
        source_files[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }

    root_string = str(root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    importlib.invalidate_caches()
    audio_io = importlib.import_module("scripts.audio_enhancement.audio_io")
    baseline = importlib.import_module("scripts.audio_enhancement.baselines")
    profile = importlib.import_module("scripts.audio_enhancement.shared_frequency_profile_v1")
    for module, relative in (
        (audio_io, "scripts/audio_enhancement/audio_io.py"),
        (baseline, "scripts/audio_enhancement/baselines.py"),
        (profile, "scripts/audio_enhancement/shared_frequency_profile_v1.py"),
    ):
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise RuntimeError(f"imported helper does not come from the pinned repo: {relative}")
    if int(audio_io.CANONICAL_SAMPLE_RATE) != SAMPLE_RATE_HZ:
        raise RuntimeError("pinned audio helper sample rate differs from the V2 source clock")

    return AudioBackend(
        stft=baseline._stft,
        istft=baseline._istft,
        apply_global_rms=baseline.apply_global_rms,
        waveform_rms=baseline.waveform_rms,
        gaussian_kernel=profile._gaussian_kernel,
        smooth_axis=profile._smooth_axis,
        source_files=source_files,
    )


@dataclass(frozen=True)
class CandidateSpecV2:
    """All transform and postprocess choices are explicit; there are no defaults."""

    candidate_id: str
    gate_kind: str
    contrast_polarity: float
    contrast_strength: float
    flux_low_quantile: float | None
    flux_high_quantile: float | None
    max_mask_db: float
    smooth_frequency_sigma_bins: float
    smooth_time_sigma_frames: float
    target_rms: float | None
    peak_ceiling: float | None
    sample_rate_hz: int
    n_fft: int
    hop_length: int

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if self.gate_kind not in {"ungated", "positive_flux", "negative_flux"}:
            raise ValueError("gate_kind must be ungated, positive_flux, or negative_flux")
        if self.contrast_polarity not in (-1.0, 1.0):
            raise ValueError("contrast_polarity must be exactly -1 or +1")
        numeric = (
            self.contrast_polarity,
            self.contrast_strength,
            self.max_mask_db,
            self.smooth_frequency_sigma_bins,
            self.smooth_time_sigma_frames,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("all numeric candidate parameters must be finite")
        if self.contrast_strength < 0 or self.max_mask_db <= 0:
            raise ValueError("contrast strength must be nonnegative and mask bound positive")
        if self.smooth_frequency_sigma_bins <= 0 or self.smooth_time_sigma_frames <= 0:
            raise ValueError("smoothing scales must be positive")
        if self.gate_kind == "ungated":
            if self.flux_low_quantile is not None or self.flux_high_quantile is not None:
                raise ValueError("ungated candidates must explicitly use null flux quantiles")
        else:
            if self.flux_low_quantile is None or self.flux_high_quantile is None:
                raise ValueError("signed-flux gates require explicit low/high quantiles")
            if not (
                math.isfinite(self.flux_low_quantile)
                and math.isfinite(self.flux_high_quantile)
                and 0 <= self.flux_low_quantile < self.flux_high_quantile <= 1
            ):
                raise ValueError("flux quantiles must be ordered within [0,1]")
        if self.target_rms is not None and (
            not math.isfinite(self.target_rms) or self.target_rms <= 0
        ):
            raise ValueError("target_rms must be null or positive and finite")
        if self.peak_ceiling is not None and (
            not math.isfinite(self.peak_ceiling) or self.peak_ceiling <= 0
        ):
            raise ValueError("peak_ceiling must be null or positive and finite")
        if self.target_rms is None and self.peak_ceiling is not None:
            raise ValueError("peak ceiling is only applied as part of the explicit RMS stage")
        if (self.sample_rate_hz, self.n_fft, self.hop_length) != (
            SAMPLE_RATE_HZ,
            N_FFT,
            HOP_LENGTH,
        ):
            raise ValueError("V2 clock is fixed to 16 kHz, 512-point STFT, and 128-sample hop")


@dataclass(frozen=True)
class PreparedMaskV2:
    """Source STFT and preprojection masks for one candidate and one utterance."""

    spec: CandidateSpecV2
    source_sample_count: int
    source_pcm_float32_sha256: str
    source_stft: np.ndarray
    stft_window: np.ndarray
    frame_centers_samples: np.ndarray
    arrays: Mapping[str, np.ndarray]
    preprojection_summary: Mapping[str, Any]


@dataclass(frozen=True)
class CommonKappaV2:
    """One shared projection computed from exactly the four signed gate cells."""

    kappa: float
    max_mask_db: float
    max_abs_raw_mask_db_across_cells: float
    source_sample_count: int
    source_pcm_float32_sha256: str
    raw_mask_bank_sha256: str
    cells: tuple[tuple[str, float], ...]
    cell_raw_mask_sha256: tuple[tuple[str, float, str], ...]
    additional_candidate_mask_sha256: tuple[tuple[str, str], ...]


def relative_spectral_flux_v2(magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive increases and negative decreases in linear magnitude, not VAD."""

    mag = np.asarray(magnitude, dtype=np.float64)
    if mag.ndim != 2 or mag.shape[1] < 1 or not np.isfinite(mag).all():
        raise ValueError("magnitude must be finite frequency-by-time data")
    positive = np.zeros(mag.shape[1], dtype=np.float64)
    negative = np.zeros(mag.shape[1], dtype=np.float64)
    if mag.shape[1] > 1:
        delta = np.diff(mag, axis=1)
        denominator = np.maximum(np.sum(mag[:, :-1], axis=0, dtype=np.float64), EPS)
        positive[1:] = np.sum(np.maximum(delta, 0.0), axis=0, dtype=np.float64) / denominator
        negative[1:] = np.sum(np.maximum(-delta, 0.0), axis=0, dtype=np.float64) / denominator
    return positive, negative


def _soft_quantile_gate(
    signal: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> tuple[np.ndarray, float, float, bool]:
    values = np.asarray(signal, dtype=np.float64)
    low = float(np.quantile(values, low_quantile))
    high = float(np.quantile(values, high_quantile))
    if high <= low + 1e-12:
        return np.zeros_like(values), low, high, True
    gate = np.clip((values - low) / (high - low), 0.0, 1.0)
    return gate, low, high, False


def _frame_power_arrays(source: np.ndarray, output: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(output, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("physical frame-power comparison requires paired exact-length mono PCM")
    count = x.size // FRAME_SAMPLES_40MS
    used = count * FRAME_SAMPLES_40MS
    if count:
        x_frame = x[:used].reshape(count, FRAME_SAMPLES_40MS)
        y_frame = y[:used].reshape(count, FRAME_SAMPLES_40MS)
        x_power = np.mean(x_frame * x_frame, axis=1, dtype=np.float64)
        y_power = np.mean(y_frame * y_frame, axis=1, dtype=np.float64)
        ratio_db = 10.0 * np.log10(
            np.maximum(y_power, 1e-20) / np.maximum(x_power, 1e-20)
        )
    else:
        x_power = y_power = ratio_db = np.empty(0, dtype=np.float64)
    arrays = {
        "physical_frame_start_samples_40ms": (
            np.arange(count, dtype=np.int64) * FRAME_SAMPLES_40MS
        ),
        "physical_frame_input_power_40ms": x_power.astype(np.float64),
        "physical_frame_output_power_40ms": y_power.astype(np.float64),
        "physical_frame_power_ratio_db_40ms": ratio_db.astype(np.float32),
    }
    summary = {
        "frame_count": int(count),
        "frame_samples": FRAME_SAMPLES_40MS,
        "trailing_samples_excluded": int(x.size - used),
        "mean_log_power_ratio_db": float(np.mean(ratio_db)) if count else None,
        "mean_abs_log_power_ratio_db": float(np.mean(np.abs(ratio_db))) if count else None,
        "minimum_log_power_ratio_db": float(np.min(ratio_db)) if count else None,
        "maximum_log_power_ratio_db": float(np.max(ratio_db)) if count else None,
    }
    return arrays, summary


def prepare_local_transition_mask_v2(
    samples: np.ndarray,
    spec: CandidateSpecV2,
    *,
    backend: AudioBackend,
) -> PreparedMaskV2:
    """Prepare original-source-STFT mask arrays without applying κ or writing audio."""

    spec.validate()
    x = np.asarray(samples)
    if x.dtype != np.float32 or x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("source must be finite, nonempty mono float32 PCM")
    spectrum, frame_centers, window = backend.stft(
        x,
        n_fft=spec.n_fft,
        hop_length=spec.hop_length,
    )
    magnitude = np.abs(spectrum)
    log_magnitude_db = 20.0 * np.log10(np.maximum(magnitude, EPS))
    smoothed = backend.smooth_axis(
        log_magnitude_db,
        backend.gaussian_kernel(spec.smooth_frequency_sigma_bins),
        axis=0,
    )
    smoothed = backend.smooth_axis(
        smoothed,
        backend.gaussian_kernel(spec.smooth_time_sigma_frames),
        axis=1,
    )
    raw_log_contrast_db = log_magnitude_db - smoothed
    centered_log_contrast_db = raw_log_contrast_db - raw_log_contrast_db.mean(
        axis=0, keepdims=True, dtype=np.float64
    )
    positive_flux, negative_flux = relative_spectral_flux_v2(magnitude)

    gate_degenerate = False
    low_cut: float | None = None
    high_cut: float | None = None
    if spec.gate_kind == "ungated":
        gate = np.ones(magnitude.shape[1], dtype=np.float64)
        selected_flux = None
    else:
        selected_flux = positive_flux if spec.gate_kind == "positive_flux" else negative_flux
        if spec.flux_low_quantile is None or spec.flux_high_quantile is None:
            raise ValueError("validated signed-flux candidate is missing quantile bounds")
        gate, low_cut, high_cut, gate_degenerate = _soft_quantile_gate(
            selected_flux,
            spec.flux_low_quantile,
            spec.flux_high_quantile,
        )

    raw_mask_db = (
        spec.contrast_polarity
        * spec.contrast_strength
        * centered_log_contrast_db
        * gate[None, :]
    )
    raw_mask_db -= raw_mask_db.mean(axis=0, keepdims=True, dtype=np.float64)
    if not np.isfinite(raw_mask_db).all():
        raise RuntimeError("raw mask contains non-finite values")

    arrays: dict[str, np.ndarray] = {
        "source_stft_frame_centers_samples": np.asarray(frame_centers, dtype=np.int64),
        "raw_log_magnitude_db": log_magnitude_db.astype(np.float32),
        "smoothed_log_magnitude_db": smoothed.astype(np.float32),
        "raw_log_contrast_db": raw_log_contrast_db.astype(np.float32),
        "centered_log_contrast_db": centered_log_contrast_db.astype(np.float32),
        "positive_relative_spectral_flux": positive_flux.astype(np.float32),
        "negative_relative_spectral_flux": negative_flux.astype(np.float32),
        "gate": gate.astype(np.float32),
        "raw_mask_db_before_common_kappa": raw_mask_db.astype(np.float32),
    }
    if selected_flux is not None:
        arrays["selected_signed_gate_signal"] = selected_flux.astype(np.float32)

    nonzero_fraction = float(np.mean(gate > 0.0))
    preprojection_summary = {
        "schema": SCHEMA,
        "candidate_spec": asdict(spec),
        "gate_kind": spec.gate_kind,
        "gate_signal_semantics": (
            "positive relative spectral flux" if spec.gate_kind == "positive_flux"
            else "negative relative spectral flux" if spec.gate_kind == "negative_flux"
            else "ungated constant-one reference"
        ),
        "gate_is_vad_or_phoneme_label": False,
        "gate_degenerate": bool(gate_degenerate),
        "gate_low_cut": low_cut,
        "gate_high_cut": high_cut,
        "gate_nonzero_frame_fraction": nonzero_fraction,
        "gate_mean": float(np.mean(gate)),
        "gate_sum_frames": float(np.sum(gate, dtype=np.float64)),
        "gating_applies": spec.gate_kind != "ungated",
        "raw_mask_abs_mean_db_per_bin_frame": float(np.mean(np.abs(raw_mask_db))),
        "raw_mask_abs_sum_db_bin_frames": float(np.sum(np.abs(raw_mask_db), dtype=np.float64)),
        "raw_mask_max_abs_db": float(np.max(np.abs(raw_mask_db))),
        "frame_count": int(raw_mask_db.shape[1]),
        "frequency_bin_count": int(raw_mask_db.shape[0]),
        "source_sample_count": int(x.size),
        "source_pcm_float32_sha256": _canonical_float32_sha(x),
        "source_clock": "original 16 kHz sample indices; no resampling or retiming",
        "phase_construction": (
            "mask multiplies source STFT coefficients by positive real magnitude gains; "
            "phase is retained at coefficient construction. ISTFT and subsequent reanalysis "
            "may change phase and are not asserted to be phase-identical."
        ),
    }
    return PreparedMaskV2(
        spec=spec,
        source_sample_count=int(x.size),
        source_pcm_float32_sha256=_canonical_float32_sha(x),
        source_stft=np.asarray(spectrum),
        stft_window=np.asarray(window),
        frame_centers_samples=np.asarray(frame_centers, dtype=np.int64),
        arrays=arrays,
        preprojection_summary=preprojection_summary,
    )


def common_kappa_for_four_transition_cells_v2(
    prepared_cells: Mapping[tuple[str, float], PreparedMaskV2],
    *,
    additional_prepared: Mapping[str, PreparedMaskV2],
) -> CommonKappaV2:
    """Compute one κ over four signed cells plus explicitly named comparators.

    ``additional_prepared`` is required even when empty. An ungated accent can
    share the very same κ by being declared there; callers cannot accidentally
    reuse the four-cell κ for an unregistered mask.
    """

    if frozenset(prepared_cells) != COMMON_CELL_KEYS:
        raise ValueError("one common κ requires exactly the four signed transition cells")
    ordered_keys = tuple(sorted(COMMON_CELL_KEYS))
    ordered = [prepared_cells[key] for key in ordered_keys]
    first = ordered[0]
    first_spec = first.spec
    invariant_spec_fields = (
        "contrast_strength",
        "flux_low_quantile",
        "flux_high_quantile",
        "max_mask_db",
        "smooth_frequency_sigma_bins",
        "smooth_time_sigma_frames",
        "target_rms",
        "peak_ceiling",
        "sample_rate_hz",
        "n_fft",
        "hop_length",
    )
    for key, item in zip(ordered_keys, ordered, strict=True):
        if (item.spec.gate_kind, item.spec.contrast_polarity) != key:
            raise ValueError("prepared cell label does not match its declared gate/polarity key")
        if (
            item.source_sample_count != first.source_sample_count
            or item.source_pcm_float32_sha256 != first.source_pcm_float32_sha256
            or item.frame_centers_samples.shape != first.frame_centers_samples.shape
            or not np.array_equal(item.frame_centers_samples, first.frame_centers_samples)
        ):
            raise ValueError("all four cells must use the same source waveform and source clock")
        if any(getattr(item.spec, field) != getattr(first_spec, field) for field in invariant_spec_fields):
            raise ValueError("all four cells must share strengths, bounds, gates, smoothing, RMS, and clock")

    if set(additional_prepared) != {
        item.spec.candidate_id for item in additional_prepared.values()
    }:
        raise ValueError("additional comparator keys must match their unique candidate IDs")
    additional_items = [additional_prepared[key] for key in sorted(additional_prepared)]
    additional_invariant_fields = tuple(
        field for field in invariant_spec_fields
        if field not in {"flux_low_quantile", "flux_high_quantile"}
    )
    for item in additional_items:
        if item.spec.gate_kind != "ungated":
            raise ValueError("additional comparator bank currently accepts explicitly ungated arms")
        if (
            item.source_sample_count != first.source_sample_count
            or item.source_pcm_float32_sha256 != first.source_pcm_float32_sha256
            or item.frame_centers_samples.shape != first.frame_centers_samples.shape
            or not np.array_equal(item.frame_centers_samples, first.frame_centers_samples)
        ):
            raise ValueError("additional comparators must use the same waveform and source clock")
        if any(
            getattr(item.spec, field) != getattr(first_spec, field)
            for field in additional_invariant_fields
        ):
            raise ValueError("additional comparators must share mask bound, smoothing, RMS, and clock")

    all_items = [*ordered, *additional_items]
    raw_max_abs = max(
        float(np.max(np.abs(item.arrays["raw_mask_db_before_common_kappa"])))
        for item in all_items
    )
    bound = float(first_spec.max_mask_db)
    kappa = min(1.0, bound / max(raw_max_abs, EPS))
    digest = hashlib.sha256()
    digest.update(b"cross_tfg_v5_common_kappa_four_cells_v2\0")
    cell_hashes: list[tuple[str, float, str]] = []
    for key, item in zip(ordered_keys, ordered, strict=True):
        mask_bytes = np.ascontiguousarray(
            item.arrays["raw_mask_db_before_common_kappa"], dtype="<f4"
        ).tobytes()
        cell_hashes.append((key[0], key[1], hashlib.sha256(mask_bytes).hexdigest()))
        digest.update(json.dumps(key, separators=(",", ":")).encode("ascii"))
        digest.update(mask_bytes)
    additional_hashes: list[tuple[str, str]] = []
    for candidate_id, item in sorted(additional_prepared.items()):
        mask_bytes = np.ascontiguousarray(
            item.arrays["raw_mask_db_before_common_kappa"], dtype="<f4"
        ).tobytes()
        additional_hashes.append((candidate_id, hashlib.sha256(mask_bytes).hexdigest()))
        digest.update(b"additional\0")
        digest.update(candidate_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(mask_bytes)
    return CommonKappaV2(
        kappa=float(kappa),
        max_mask_db=bound,
        max_abs_raw_mask_db_across_cells=raw_max_abs,
        source_sample_count=first.source_sample_count,
        source_pcm_float32_sha256=first.source_pcm_float32_sha256,
        raw_mask_bank_sha256=digest.hexdigest(),
        cells=ordered_keys,
        cell_raw_mask_sha256=tuple(cell_hashes),
        additional_candidate_mask_sha256=tuple(additional_hashes),
    )


def apply_prepared_mask_v2(
    samples: np.ndarray,
    prepared: PreparedMaskV2,
    *,
    common_projection: CommonKappaV2,
    backend: AudioBackend,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Apply an explicit shared κ to one prepared mask and return summary/arrays."""

    spec = prepared.spec
    spec.validate()
    x = np.asarray(samples)
    if x.dtype != np.float32 or x.ndim != 1 or x.size != prepared.source_sample_count:
        raise ValueError("apply input must match the prepared float32 source sample count")
    source_sha = _canonical_float32_sha(x)
    if source_sha != prepared.source_pcm_float32_sha256:
        raise ValueError("apply input does not match the source used to prepare the mask")
    if (
        common_projection.source_sample_count != x.size
        or common_projection.source_pcm_float32_sha256 != source_sha
        or common_projection.max_mask_db != spec.max_mask_db
        or common_projection.cells != tuple(sorted(COMMON_CELL_KEYS))
        or len(common_projection.cell_raw_mask_sha256) != len(COMMON_CELL_KEYS)
        or frozenset(
            (gate_kind, polarity)
            for gate_kind, polarity, _ in common_projection.cell_raw_mask_sha256
        ) != COMMON_CELL_KEYS
    ):
        raise ValueError("common κ record is not bound to this utterance/four-cell mask bank")
    current_raw_mask_sha = hashlib.sha256(
        np.ascontiguousarray(
            prepared.arrays["raw_mask_db_before_common_kappa"], dtype="<f4"
        ).tobytes()
    ).hexdigest()
    expected_cell_hashes = {
        (gate_kind, polarity): mask_sha
        for gate_kind, polarity, mask_sha in common_projection.cell_raw_mask_sha256
    }
    expected_additional_hashes = dict(common_projection.additional_candidate_mask_sha256)
    cell_key = (spec.gate_kind, spec.contrast_polarity)
    if cell_key in common_projection.cells:
        registered_mask_sha = expected_cell_hashes.get(cell_key)
    else:
        registered_mask_sha = expected_additional_hashes.get(spec.candidate_id)
    if registered_mask_sha != current_raw_mask_sha:
        raise ValueError("prepared mask bytes are not the matching cell in the common κ bank")
    kappa = float(common_projection.kappa)
    if not math.isfinite(kappa) or not 0.0 < kappa <= 1.0:
        raise ValueError("common κ must be finite and in (0,1]")
    raw_mask = prepared.arrays["raw_mask_db_before_common_kappa"].astype(np.float64)
    projected_mask = raw_mask * kappa
    projected_max_abs = float(np.max(np.abs(projected_mask)))
    if projected_max_abs > spec.max_mask_db + 1e-6:
        raise ValueError("shared κ does not satisfy this mask’s explicit dB bound")

    source_stft = prepared.source_stft
    modified_stft = source_stft * np.power(10.0, projected_mask / 20.0)
    reconstructed = backend.istft(
        modified_stft,
        int(x.size),
        window=prepared.stft_window,
        hop_length=spec.hop_length,
    ).astype(np.float32)
    if reconstructed.shape != x.shape or not np.isfinite(reconstructed).all():
        raise RuntimeError("ISTFT output violated exact source length or finite PCM")

    rms_info = None
    if spec.target_rms is None:
        if spec.peak_ceiling is not None:
            raise ValueError("peak ceiling without a fixed-RMS stage is not supported")
        output = np.ascontiguousarray(reconstructed, dtype=np.float32)
    else:
        output, rms_info = backend.apply_global_rms(
            reconstructed,
            target_rms=spec.target_rms,
            peak_ceiling=spec.peak_ceiling,
        )
    if output.dtype != np.float32 or output.shape != x.shape or not np.isfinite(output).all():
        raise RuntimeError("candidate violated finite float32 exact-length output contract")

    power_arrays, power_summary = _frame_power_arrays(x, output)
    diagnostic_arrays = {
        **prepared.arrays,
        **power_arrays,
        "projected_mask_db_after_common_kappa": projected_mask.astype(np.float32),
        "common_kappa": np.asarray(common_projection.kappa, dtype=np.float64),
        "common_kappa_max_abs_raw_across_cells_db": np.asarray(
            common_projection.max_abs_raw_mask_db_across_cells, dtype=np.float64
        ),
        "common_kappa_bank_sha256_ascii": np.frombuffer(
            common_projection.raw_mask_bank_sha256.encode("ascii"), dtype=np.uint8
        ).copy(),
    }
    summary = {
        "schema": SCHEMA,
        "candidate_spec": asdict(spec),
        "source_sample_count": int(x.size),
        "output_sample_count": int(output.size),
        "sample_rate_hz": int(spec.sample_rate_hz),
        "source_pcm_float32_sha256": source_sha,
        "input_rms": backend.waveform_rms(x),
        "output_rms": backend.waveform_rms(output),
        "input_peak_abs": float(np.max(np.abs(x.astype(np.float64)))),
        "output_peak_abs": float(np.max(np.abs(output.astype(np.float64)))),
        "peak_ceiling": spec.peak_ceiling,
        "peak_cap_applied": bool(rms_info.peak_limited) if rms_info is not None else False,
        "rms_stage": asdict(rms_info) if rms_info is not None else None,
        "phase_construction": prepared.preprojection_summary["phase_construction"],
        "source_clock": "same original sample count and 16 kHz source indices; no retiming",
        "common_projection": {
            "kind": "externally supplied one-per-utterance four-cell shared kappa",
            "kappa": float(common_projection.kappa),
            "max_mask_db": float(common_projection.max_mask_db),
            "raw_mask_bank_sha256": common_projection.raw_mask_bank_sha256,
            "cells": [list(cell) for cell in common_projection.cells],
            "cell_raw_mask_sha256": [
                {
                    "gate_kind": gate_kind,
                    "contrast_polarity": polarity,
                    "raw_mask_sha256": mask_sha,
                }
                for gate_kind, polarity, mask_sha in common_projection.cell_raw_mask_sha256
            ],
            "additional_candidate_mask_sha256": [
                {"candidate_id": candidate_id, "raw_mask_sha256": mask_sha}
                for candidate_id, mask_sha in common_projection.additional_candidate_mask_sha256
            ],
            "max_abs_raw_mask_db_across_cells": float(
                common_projection.max_abs_raw_mask_db_across_cells
            ),
            "projected_mask_max_abs_db": projected_max_abs,
        },
        "gate_dose": {
            "kind": spec.gate_kind,
            "is_vad_or_phoneme_label": False,
            "flux_low_cut": prepared.preprojection_summary["gate_low_cut"],
            "flux_high_cut": prepared.preprojection_summary["gate_high_cut"],
            "gate_degenerate": prepared.preprojection_summary["gate_degenerate"],
            "active_frame_fraction_gate_gt_zero": (
                None if spec.gate_kind == "ungated"
                else float(np.mean(prepared.arrays["gate"] > 0.0))
            ),
            "active_frame_count_gate_gt_zero": (
                None if spec.gate_kind == "ungated"
                else int(np.count_nonzero(prepared.arrays["gate"] > 0.0))
            ),
            "frame_count": int(prepared.arrays["gate"].size),
            "mean_gate_weight": float(np.mean(prepared.arrays["gate"])),
            "raw_mask_abs_mean_db_per_bin_frame": float(np.mean(np.abs(raw_mask))),
            "projected_mask_abs_mean_db_per_bin_frame": float(np.mean(np.abs(projected_mask))),
            "raw_mask_abs_sum_db_bin_frames": float(np.sum(np.abs(raw_mask), dtype=np.float64)),
            "projected_mask_abs_sum_db_bin_frames": float(
                np.sum(np.abs(projected_mask), dtype=np.float64)
            ),
        },
        "physical_frame_power_ratio_40ms": power_summary,
        "interpretation_limit": (
            "Equal gate coverage, equal arithmetic mask mean, or equal RMS does not imply "
            "equal physical frame power. Read the emitted 40 ms frame-power arrays."
        ),
        "backend_source_files": dict(backend.source_files),
    }
    return np.ascontiguousarray(output, dtype=np.float32), summary, diagnostic_arrays


def write_diagnostic_arrays_npz_v2(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Persist numeric diagnostics to a new NPZ; refuse to overwrite an existing file."""

    path = Path(destination)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"diagnostic destination must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not name or not isinstance(name, str):
            raise ValueError("NPZ diagnostic names must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"object arrays are not permitted: {name}")
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise ValueError(f"non-finite diagnostics are not permitted: {name}")
        normalized[name] = array
    with path.open("xb") as stream:
        np.savez_compressed(stream, **normalized)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
