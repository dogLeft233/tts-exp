"""Frozen numerical operations for the portable source-axis experiment.

All acoustic operations consume canonical 16 kHz mono floating-point PCM.
Model latents are kept as float32; cross-clip aggregation and the constrained
projection use float64. This module intentionally has no repository or GPU
path assumptions so the same source can run in the local and remote trees.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TARGET_RMS = 0.0376838172675746
SAMPLE_RATE = 16_000
FPS = 25
HOP_SAMPLES = SAMPLE_RATE // FPS
WINDOW_SAMPLES = SAMPLE_RATE // 5  # 200 ms
SPEECH_RMS_THRESHOLD = 10.0 ** (-45.0 / 20.0)
LEVEL_GAIN = 10.0 ** (3.0 / 20.0)
LATENT_DIM = 512


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("RMS input must be nonempty and finite")
    return float(np.sqrt(np.mean(np.square(array), dtype=np.float64)))


def normalize_rms(value: np.ndarray, target: float = TARGET_RMS) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale a waveform to target RMS without clipping or PCM quantization."""

    source = np.asarray(value)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("waveform must be a nonempty finite mono vector")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target RMS must be positive and finite")
    source64 = source.astype(np.float64, copy=False)
    before = rms(source64)
    if before <= 0:
        raise ValueError("cannot RMS-normalize a silent waveform")
    gain = np.float64(target) / np.float64(before)
    fixed = np.multiply(source64, gain, dtype=np.float64).astype(np.float32)
    peak = float(np.max(np.abs(fixed)))
    meta = {
        "target_rms": float(target),
        "input_rms": before,
        "gain": float(gain),
        "gain_db": float(20.0 * np.log10(gain)),
        "realized_rms_float32": rms(fixed),
        "peak_abs_float32": peak,
        "peak_over_one": bool(peak > 1.0),
        "clipping": False,
        "input_dtype": str(source.dtype),
        "output_dtype": "float32",
        "pcm_sha256": array_sha256(fixed),
    }
    return fixed, meta


def gain_fixed_waveform(value: np.ndarray, db: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a fixed gain in float64, then store float32 without clipping."""

    source = np.asarray(value)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("waveform must be a nonempty finite mono vector")
    if not math.isfinite(db):
        raise ValueError("gain must be finite")
    factor = np.float64(10.0) ** (np.float64(db) / np.float64(20.0))
    output = np.multiply(source.astype(np.float64, copy=False), factor, dtype=np.float64).astype(np.float32)
    peak = float(np.max(np.abs(output)))
    return output, {
        "db": float(db),
        "linear_gain": float(factor),
        "rms_before": rms(source),
        "rms_after": rms(output),
        "peak_abs_after": peak,
        "peak_over_one": bool(peak > 1.0),
        "clipping": False,
        "output_dtype": "float32",
        "pcm_sha256": array_sha256(output),
    }


def mel_chunk_starts(mel_frames: int, *, fps: int = FPS, mel_step: int = 16) -> list[int]:
    """Return starts produced by the official Wav2Lip mel chunk loop."""

    if mel_frames < mel_step or fps <= 0 or mel_step <= 0:
        raise ValueError("mel shape, FPS, or chunk width is invalid")
    multiplier = 80.0 / float(fps)
    starts: list[int] = []
    index = 0
    while True:
        start = int(index * multiplier)
        if start + mel_step > mel_frames:
            starts.append(mel_frames - mel_step)
            break
        starts.append(start)
        index += 1
    return starts


def fixed_speech_mask(
    fixed_pcm: np.ndarray,
    mel_starts: Sequence[int],
    *,
    threshold: float = SPEECH_RMS_THRESHOLD,
    mel_hop_samples: int = 200,
    window_samples: int = WINDOW_SAMPLES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mark each latent window from its fixed 200 ms source-PCM RMS window.

    Window i begins at the sample corresponding to that Wav2Lip mel chunk's
    exact start column. The final window is zero-padded only when it extends
    beyond the waveform. It is not selected or adjusted from latent outputs.
    """

    waveform = np.asarray(fixed_pcm, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("fixed PCM must be a nonempty finite float32 mono vector")
    if threshold <= 0 or window_samples <= 0 or mel_hop_samples <= 0:
        raise ValueError("RMS mask settings must be positive")
    frame_rms = np.empty(len(mel_starts), dtype=np.float64)
    for index, mel_start in enumerate(mel_starts):
        start = int(mel_start) * int(mel_hop_samples)
        stop = start + int(window_samples)
        segment = waveform[max(0, start) : min(waveform.size, stop)].astype(np.float64, copy=False)
        if segment.size < window_samples:
            segment = np.pad(segment, (0, window_samples - segment.size))
        frame_rms[index] = rms(segment)
    mask = frame_rms > float(threshold)
    return mask, {
        "rule": "fixed waveform RMS over [mel_chunk_start_sample, +200ms), zero-pad right edge",
        "threshold_rms": float(threshold),
        "threshold_dbfs": -45.0,
        "window_samples": int(window_samples),
        "window_seconds": float(window_samples / SAMPLE_RATE),
        "mel_hop_samples": int(mel_hop_samples),
        "mel_starts": [int(item) for item in mel_starts],
        "frame_rms": frame_rms.tolist(),
        "active_frames": int(mask.sum()),
        "inactive_frames": int(mask.size - mask.sum()),
        "mask_sha256": array_sha256(mask.astype(np.uint8)),
        "selected_from_audio_only": True,
    }


def _clip_active_mean(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    latent = np.asarray(z)
    active = np.asarray(mask, dtype=bool)
    if latent.dtype != np.float32 or latent.ndim != 2 or latent.shape[1] != LATENT_DIM:
        raise ValueError(f"latent must be float32 [frames,{LATENT_DIM}]")
    if latent.shape[0] != active.size or not active.any() or not np.isfinite(latent).all():
        raise ValueError("latent and nonempty active mask must align")
    return latent[active].astype(np.float64).mean(axis=0, dtype=np.float64)


def _speaker_equal_mean(values: Sequence[np.ndarray], speakers: Sequence[str]) -> np.ndarray:
    if len(values) != len(speakers) or not values:
        raise ValueError("value and speaker rows must be nonempty and aligned")
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for value, speaker in zip(values, speakers, strict=True):
        groups[str(speaker)].append(np.asarray(value, dtype=np.float64))
    speaker_means = [np.stack(group).mean(axis=0, dtype=np.float64) for _, group in sorted(groups.items())]
    return np.stack(speaker_means).mean(axis=0, dtype=np.float64)


def calibration_axes(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute cal-only source d and level u with speaker-equal weighting.

    Each row supplies speaker, fixed natural/TTS latent arrays, their fixed
    waveform masks, and natural +/-3 dB latents. The natural mask is reused
    for the two level perturbations.
    """

    if not rows:
        raise ValueError("calibration rows are empty")
    speakers = [str(row["speaker_id"]) for row in rows]
    expected: dict[str, int] = defaultdict(int)
    for speaker in speakers:
        expected[speaker] += 1
    if any(count != 2 for count in expected.values()):
        raise ValueError(f"expected exactly two calibration clips per speaker: {dict(expected)}")

    natural_means: list[np.ndarray] = []
    tts_means: list[np.ndarray] = []
    level_deltas: list[np.ndarray] = []
    per_clip: list[dict[str, Any]] = []
    for row in rows:
        natural = _clip_active_mean(row["z_natural_fixed"], row["mask_natural_fixed"])
        tts = _clip_active_mean(row["z_tts_fixed"], row["mask_tts_fixed"])
        plus = _clip_active_mean(row["z_natural_plus3db"], row["mask_natural_fixed"])
        minus = _clip_active_mean(row["z_natural_minus3db"], row["mask_natural_fixed"])
        natural_means.append(natural)
        tts_means.append(tts)
        level_deltas.append((plus - minus) / np.float64(6.0))
        per_clip.append({
            "sample_id": str(row["sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "natural_active_frames": int(np.asarray(row["mask_natural_fixed"], dtype=bool).sum()),
            "tts_active_frames": int(np.asarray(row["mask_tts_fixed"], dtype=bool).sum()),
            "natural_active_mean_sha256": array_sha256(natural),
            "tts_active_mean_sha256": array_sha256(tts),
            "level_delta_per_db_sha256": array_sha256(level_deltas[-1]),
        })

    mu_natural = _speaker_equal_mean(natural_means, speakers)
    mu_tts = _speaker_equal_mean(tts_means, speakers)
    d_source = mu_tts - mu_natural
    u_level = _speaker_equal_mean(level_deltas, speakers)
    norm_d = float(np.linalg.norm(d_source))
    norm_u = float(np.linalg.norm(u_level))
    if norm_d <= 1e-12 or norm_u <= 1e-12:
        raise ValueError(f"calibration axis is degenerate: ||d||={norm_d}, ||u||={norm_u}")
    return d_source, u_level, {
        "calibration_clip_count": len(rows),
        "calibration_speaker_count": len(expected),
        "two_clips_per_speaker": True,
        "speaker_weighting": "mean within speaker, then equal weight per speaker",
        "d_definition": "speaker_equal_mean(TTS_fixed_active_z) - speaker_equal_mean(natural_fixed_active_z)",
        "u_definition": "speaker_equal_mean((natural_plus3dB_active_z - natural_minus3dB_active_z)/6)",
        "natural_mean_sha256": array_sha256(mu_natural),
        "tts_mean_sha256": array_sha256(mu_tts),
        "d_sha256": array_sha256(d_source),
        "u_sha256": array_sha256(u_level),
        "d_norm": norm_d,
        "u_norm": norm_u,
        "per_clip": per_clip,
    }


def level_orthogonal_projection(
    z: np.ndarray,
    target: np.ndarray,
    u: np.ndarray,
    *,
    tolerance: float = 2e-12,
    max_iterations: int = 256,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Exact float64 KKT projection onto y>=0 and u·(y-z)=0.

    The scalar dual solution is y=max(0, target-lambda*u). Since z itself is
    feasible, a bracket exists unless u is numerically zero.
    """

    base = np.asarray(z, dtype=np.float64)
    proposal = np.asarray(target, dtype=np.float64)
    axis = np.asarray(u, dtype=np.float64)
    if base.ndim != 1 or proposal.shape != base.shape or axis.shape != base.shape:
        raise ValueError("projection vectors must have equal one-dimensional shape")
    if not np.isfinite(base).all() or not np.isfinite(proposal).all() or not np.isfinite(axis).all() or np.any(base < 0):
        raise ValueError("projection inputs must be finite; baseline must be nonnegative")
    axis_norm_sq = float(axis @ axis)
    if axis_norm_sq <= 1e-24:
        raise ValueError("level axis is numerically zero")
    baseline_projection = float(axis @ base)

    def evaluate(lam: float) -> tuple[float, np.ndarray]:
        y = np.maximum(0.0, proposal - np.float64(lam) * axis)
        return float(axis @ (y - base)), y

    f0, y0 = evaluate(0.0)
    scale = max(1.0, float(np.linalg.norm(axis) * np.linalg.norm(base)))
    tol = float(tolerance * scale)
    if abs(f0) <= tol:
        return y0, {
            "lambda": 0.0,
            "bracket_expansions": 0,
            "bisections": 0,
            "residual64": f0,
            "tolerance64": tol,
            "objective_squared64": float(np.square(y0 - proposal).sum(dtype=np.float64)),
            "projection_change_l2_64": float(np.linalg.norm(y0 - proposal)),
            "baseline_constraint_value": baseline_projection,
            "active_nonnegative_count": int(np.count_nonzero(y0 > 0.0)),
            "converged": True,
        }

    lo = hi = 0.0
    flo = fhi = f0
    expansions = 0
    step = 1.0
    if f0 > 0:
        hi = step
        fhi, _ = evaluate(hi)
        while fhi > 0 and expansions < 256:
            step *= 2.0
            hi = step
            fhi, _ = evaluate(hi)
            expansions += 1
    else:
        lo = -step
        flo, _ = evaluate(lo)
        while flo < 0 and expansions < 256:
            step *= 2.0
            lo = -step
            flo, _ = evaluate(lo)
            expansions += 1
    if not (flo >= -tol and fhi <= tol):
        raise ArithmeticError(f"could not bracket KKT root: f(lo)={flo}, f(hi)={fhi}")

    mid = 0.0
    residual = f0
    y = y0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        mid = (lo + hi) / 2.0
        residual, y = evaluate(mid)
        if abs(residual) <= tol:
            break
        if residual > 0.0:
            lo = mid
        else:
            hi = mid
    converged = abs(residual) <= tol
    if not converged:
        raise ArithmeticError(f"KKT bisection did not converge: residual={residual}, tol={tol}")
    return y, {
        "lambda": float(mid),
        "bracket_expansions": int(expansions),
        "bisections": int(iterations),
        "residual64": float(residual),
        "tolerance64": tol,
        "objective_squared64": float(np.square(y - proposal).sum(dtype=np.float64)),
        "projection_change_l2_64": float(np.linalg.norm(y - proposal)),
        "baseline_constraint_value": baseline_projection,
        "active_nonnegative_count": int(np.count_nonzero(y > 0.0)),
        "converged": True,
    }


def apply_source_direction(
    z: np.ndarray,
    speech_mask: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    *,
    sign: int,
    level_orthogonal: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply ±0.5d to active frames, with optional exact level-orth projection."""

    native = np.asarray(z)
    mask = np.asarray(speech_mask, dtype=bool)
    source_axis = np.asarray(d, dtype=np.float64)
    level_axis = np.asarray(u, dtype=np.float64)
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    if native.dtype != np.float32 or native.ndim != 2 or native.shape[1] != LATENT_DIM:
        raise ValueError("native latent must be float32 [frames,512]")
    if native.shape[0] != mask.size or not mask.any() or source_axis.shape != (LATENT_DIM,) or level_axis.shape != (LATENT_DIM,):
        raise ValueError("latent, mask, source axis, or level axis has invalid shape")
    if not np.isfinite(native).all() or np.any(native < 0):
        raise ValueError("native latent must be finite and nonnegative")
    if not np.isfinite(source_axis).all() or not np.isfinite(level_axis).all():
        raise ValueError("axes must be finite")

    output = native.copy()
    pre_rows: list[np.ndarray] = []
    out_rows: list[np.ndarray] = []
    projections: list[dict[str, float | int | bool]] = []
    projection_rows: list[dict[str, Any]] = []
    rounding_constraint_bounds: list[float] = []
    delta = np.float64(0.5 * sign) * source_axis
    for index in np.flatnonzero(mask):
        baseline = native[index].astype(np.float64)
        proposed = baseline + delta
        if level_orthogonal:
            projected, detail = level_orthogonal_projection(baseline, proposed, level_axis)
            projections.append(detail)
            final = projected.astype(np.float32)
            rounding_bound = float(np.sum(np.abs(level_axis) * np.abs(final.astype(np.float64) - projected), dtype=np.float64))
            rounding_constraint_bounds.append(rounding_bound)
            projection_rows.append({
                "frame_index": int(index),
                **detail,
                "float32_constraint_abs_bound": rounding_bound,
                "float32_constraint_abs": abs(float(level_axis @ (final.astype(np.float64) - baseline))),
            })
        else:
            projected = np.maximum(0.0, proposed)
            final = projected.astype(np.float32)
        if np.any(final < 0) or not np.isfinite(final).all():
            raise ArithmeticError("float32 replacement violated nonnegative finite latent contract")
        output[index] = final
        pre_rows.append(proposed)
        out_rows.append(final.astype(np.float64))

    if not np.array_equal(output[~mask], native[~mask]):
        raise AssertionError("inactive latent frames changed")
    input_active = native[mask].astype(np.float64)
    changed_active = output[mask].astype(np.float64)
    applied = changed_active - input_active
    pre_delta = np.stack(pre_rows) - input_active
    u_norm = float(np.linalg.norm(level_axis))
    d_norm = float(np.linalg.norm(source_axis))
    result: dict[str, Any] = {
        "condition": ("orth" if level_orthogonal else "full") + ("_plus" if sign > 0 else "_minus"),
        "sign": int(sign),
        "level_orthogonal": bool(level_orthogonal),
        "input_z_sha256": array_sha256(native),
        "output_z_sha256": array_sha256(output),
        "inactive_exact": True,
        "active_frames": int(mask.sum()),
        "inactive_frames": int(mask.size - mask.sum()),
        "d_norm": d_norm,
        "u_norm": u_norm,
        "intended_displacement_l2_mean": float(np.linalg.norm(delta)),
        "pre_projection_actual_norm_mean": float(np.linalg.norm(pre_delta, axis=1).mean()),
        "post_projection_actual_norm_mean": float(np.linalg.norm(applied, axis=1).mean()),
        "post_projection_actual_norm_median": float(np.median(np.linalg.norm(applied, axis=1))),
        "post_projection_actual_norm_p10": float(np.quantile(np.linalg.norm(applied, axis=1), 0.10)),
        "post_projection_actual_norm_p90": float(np.quantile(np.linalg.norm(applied, axis=1), 0.90)),
        "pre_projection_level_dot_mean": float(np.mean(pre_delta @ level_axis)),
        "post_projection_level_dot_mean": float(np.mean(applied @ level_axis)),
        "post_projection_level_dot_max_abs": float(np.max(np.abs(applied @ level_axis))),
        "cosine_to_source_axis_mean": float(np.mean((applied @ source_axis) / np.maximum(np.linalg.norm(applied, axis=1) * d_norm, 1e-30))),
        "negative_preprojection_fraction": float(np.mean(np.stack(pre_rows) < 0.0)),
        "float32_output_dtype": str(output.dtype),
        "projection_row_count": len(projections),
    }
    if projections:
        result.update({
            "kkt_bisections_max": max(int(row["bisections"]) for row in projections),
            "kkt_bracket_expansions_max": max(int(row["bracket_expansions"]) for row in projections),
            "kkt_projection_residual64_max_abs": max(abs(float(row["residual64"])) for row in projections),
            "kkt_float32_level_dot_max_abs": result["post_projection_level_dot_max_abs"],
            "kkt_projection_change_l2_mean": float(np.mean([float(row["projection_change_l2_64"]) for row in projections])),
            "kkt_objective_squared64_mean": float(np.mean([float(row["objective_squared64"]) for row in projections])),
            "kkt_all_converged": all(bool(row["converged"]) for row in projections),
            "kkt_float32_rounding_constraint_abs_bound_max": max(rounding_constraint_bounds, default=0.0),
            "kkt_float32_rounding_constraint_abs_max": max((float(row["float32_constraint_abs"]) for row in projection_rows), default=0.0),
            "per_frame_projection_diagnostics": projection_rows,
        })
    return output, result


def orthogonalize_against(source_axis: np.ndarray, level_axis: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Project candidate off span(source_axis, level_axis), in float64."""

    d = np.asarray(source_axis, dtype=np.float64)
    u = np.asarray(level_axis, dtype=np.float64)
    value = np.asarray(candidate, dtype=np.float64)
    if d.ndim != 1 or u.shape != d.shape or value.shape != d.shape:
        raise ValueError("random-direction vectors must have equal one-dimensional shapes")
    uu = float(u @ u)
    if uu <= 1e-24:
        raise ValueError("level axis is degenerate")
    unit_u = u / math.sqrt(uu)
    d_residual = d - unit_u * float(unit_u @ d)
    d_norm_sq = float(d_residual @ d_residual)
    if d_norm_sq <= 1e-24:
        raise ValueError("source axis is collinear with level axis")
    unit_d = d_residual / math.sqrt(d_norm_sq)
    result = value - unit_u * float(unit_u @ value) - unit_d * float(unit_d @ value)
    residual_u = abs(float(result @ unit_u))
    residual_d = abs(float(result @ unit_d))
    if max(residual_u, residual_d) > 1e-10 * max(1.0, float(np.linalg.norm(result))):
        raise ArithmeticError("random direction failed orthogonality projection")
    return result


def random_control_directions(d: np.ndarray, u: np.ndarray, seeds: Sequence[int] = (20260930, 20260931, 20260932)) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Create preseeded random unit directions orthogonal to d and u.

    Each direction has the same intended norm as the projected half source
    dose. ``apply_random_direction`` then matches post-KKT dose per frame to
    orth_plus, recording any target that its frozen scalar search cannot reach.
    """

    source = np.asarray(d, dtype=np.float64)
    level = np.asarray(u, dtype=np.float64)
    if source.shape != (LATENT_DIM,) or level.shape != (LATENT_DIM,):
        raise ValueError("source and level axes must have shape [512]")
    unit_u = level / np.linalg.norm(level)
    d_perp = source - unit_u * float(unit_u @ source)
    target_norm = float(0.5 * np.linalg.norm(d_perp))
    if target_norm <= 1e-12:
        raise ValueError("level-orthogonal source dose is degenerate")
    directions: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for seed in seeds:
        generator = np.random.default_rng(int(seed))
        candidate = generator.standard_normal(LATENT_DIM, dtype=np.float64)
        projected = orthogonalize_against(source, level, candidate)
        norm = float(np.linalg.norm(projected))
        if norm <= 1e-12:
            raise ValueError(f"random direction is degenerate for seed {seed}")
        direction = projected * (target_norm / norm)
        directions.append(direction)
        metadata.append({
            "seed": int(seed),
            "direction_sha256": array_sha256(direction),
            "source_dot": float(direction @ source),
            "level_dot": float(direction @ level),
            "intended_norm": float(np.linalg.norm(direction)),
            "target_half_d_perp_norm": target_norm,
            "postdose_matching": "per-frame orth_plus KKT-projected dose via fixed bounded scalar search",
        })
    return directions, metadata


def apply_random_direction(
    z: np.ndarray,
    speech_mask: np.ndarray,
    direction: np.ndarray,
    u: np.ndarray,
    orth_plus: np.ndarray,
    *,
    max_scale: float = 64.0,
    max_bisections: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strictly level-project a random direction and match orth_plus dose.

    ``direction`` is already dose-scaled to ``||0.5*d_perp||``. For each
    active frame, a scalar multiplier is found so the post-projection L2
    displacement matches that frame's observed orth_plus displacement. If
    that target cannot be reached before the frozen numerical scale cap, the
    best reached point is kept and the frame is explicitly marked infeasible.
    """

    native = np.asarray(z)
    mask = np.asarray(speech_mask, dtype=bool)
    random_delta = np.asarray(direction, dtype=np.float64)
    level_axis = np.asarray(u, dtype=np.float64)
    positive = np.asarray(orth_plus)
    if native.dtype != np.float32 or native.ndim != 2 or native.shape[1] != LATENT_DIM:
        raise ValueError("native latent must be float32 [frames,512]")
    if positive.dtype != np.float32 or positive.shape != native.shape or not np.isfinite(positive).all():
        raise ValueError("orth_plus latent must be finite float32 with the native shape")
    if native.shape[0] != mask.size or not mask.any() or random_delta.shape != (LATENT_DIM,) or level_axis.shape != (LATENT_DIM,):
        raise ValueError("random dose inputs have invalid shapes")
    if not np.isfinite(native).all() or np.any(native < 0) or not np.isfinite(random_delta).all() or not np.isfinite(level_axis).all():
        raise ValueError("random dose inputs must be finite; native z must be nonnegative")
    if not np.array_equal(positive[~mask], native[~mask]):
        raise ValueError("orth_plus must preserve inactive frames exactly")
    if max_scale <= 0 or max_bisections <= 0:
        raise ValueError("random dose search bounds must be positive")

    output = native.copy()
    scales = np.zeros(int(mask.sum()), dtype=np.float64)
    targets = np.linalg.norm(positive[mask].astype(np.float64) - native[mask].astype(np.float64), axis=1)
    actual = np.zeros_like(targets)
    infeasible = np.zeros(targets.shape, dtype=bool)
    iterations = np.zeros(targets.shape, dtype=np.int16)
    expansions = np.zeros(targets.shape, dtype=np.int16)
    residuals = np.zeros(targets.shape, dtype=np.float64)
    per_frame_rows: list[dict[str, Any]] = []
    for row_index, frame_index in enumerate(np.flatnonzero(mask)):
        base = native[frame_index].astype(np.float64)
        target_norm = float(targets[row_index])
        if target_norm <= 1e-15:
            output[frame_index] = native[frame_index]
            scales[row_index] = 0.0
            per_frame_rows.append({
                "frame_index": int(frame_index), "target_norm": target_norm, "actual_norm": 0.0,
                "dose_error": 0.0, "infeasible": False, "scale": 0.0,
                "kkt_residual64": 0.0, "float32_constraint_abs": 0.0, "float32_constraint_abs_bound": 0.0,
            })
            continue

        def dose_at(scale: float) -> tuple[float, np.ndarray, float]:
            projected, detail = level_orthogonal_projection(base, base + np.float64(scale) * random_delta, level_axis)
            dose_norm = float(np.linalg.norm(projected - base))
            return dose_norm, projected, float(detail["residual64"])

        lo = 0.0
        hi = 1.0
        dose_hi, y_hi, residual_hi = dose_at(hi)
        expand_count = 0
        while dose_hi < target_norm and hi < max_scale and expand_count < 64:
            lo = hi
            hi = min(max_scale, hi * 2.0)
            dose_hi, y_hi, residual_hi = dose_at(hi)
            expand_count += 1
        expansions[row_index] = min(expand_count, np.iinfo(np.int16).max)
        if dose_hi < target_norm:
            infeasible[row_index] = True
            scales[row_index] = hi
            actual[row_index] = dose_hi
            residuals[row_index] = residual_hi
            final = y_hi.astype(np.float32)
            output[frame_index] = final
            rounding_bound = float(np.sum(np.abs(level_axis) * np.abs(final.astype(np.float64) - y_hi), dtype=np.float64))
            per_frame_rows.append({
                "frame_index": int(frame_index), "target_norm": target_norm, "actual_norm": dose_hi,
                "dose_error": dose_hi - target_norm, "infeasible": True, "scale": hi,
                "kkt_residual64": residual_hi,
                "float32_constraint_abs": abs(float(level_axis @ (final.astype(np.float64) - base))),
                "float32_constraint_abs_bound": rounding_bound,
            })
            continue

        y_mid = y_hi
        dose_mid = dose_hi
        residual_mid = residual_hi
        for iteration in range(1, max_bisections + 1):
            mid = (lo + hi) / 2.0
            dose_mid, y_mid, residual_mid = dose_at(mid)
            if abs(dose_mid - target_norm) <= max(1e-6, 1e-6 * target_norm):
                lo = hi = mid
                break
            if dose_mid < target_norm:
                lo = mid
            else:
                hi = mid
        iterations[row_index] = min(iteration, np.iinfo(np.int16).max)
        # Select the closer bracket endpoint; ties choose the lower scale.
        low_dose, low_y, low_residual = dose_at(lo)
        high_dose, high_y, high_residual = dose_at(hi)
        if abs(low_dose - target_norm) <= abs(high_dose - target_norm):
            selected_scale, selected_y, selected_dose, selected_residual = lo, low_y, low_dose, low_residual
        else:
            selected_scale, selected_y, selected_dose, selected_residual = hi, high_y, high_dose, high_residual
        scales[row_index] = selected_scale
        actual[row_index] = selected_dose
        residuals[row_index] = selected_residual
        final = selected_y.astype(np.float32)
        output[frame_index] = final
        rounding_bound = float(np.sum(np.abs(level_axis) * np.abs(final.astype(np.float64) - selected_y), dtype=np.float64))
        per_frame_rows.append({
            "frame_index": int(frame_index), "target_norm": target_norm, "actual_norm": selected_dose,
            "dose_error": selected_dose - target_norm, "infeasible": False, "scale": selected_scale,
            "kkt_residual64": selected_residual,
            "float32_constraint_abs": abs(float(level_axis @ (final.astype(np.float64) - base))),
            "float32_constraint_abs_bound": rounding_bound,
        })

    if not np.array_equal(output[~mask], native[~mask]) or np.any(output < 0) or not np.isfinite(output).all():
        raise AssertionError("random intervention violated inactive-copy or nonnegative-latent contract")
    applied = output[mask].astype(np.float64) - native[mask].astype(np.float64)
    dose_error = actual - targets
    metadata = {
        "condition": "random_control_dose_matched_to_orth_plus",
        "random_direction_sha256": array_sha256(random_delta),
        "random_direction_norm": float(np.linalg.norm(random_delta)),
        "random_direction_is_half_d_perp_norm": True,
        "input_z_sha256": array_sha256(native),
        "orth_plus_z_sha256": array_sha256(positive),
        "output_z_sha256": array_sha256(output),
        "inactive_exact": True,
        "active_frames": int(mask.sum()),
        "inactive_frames": int(mask.size - mask.sum()),
        "per_frame_target_dose_mean": float(targets.mean()),
        "per_frame_actual_dose_mean": float(actual.mean()),
        "per_frame_abs_dose_error_max": float(np.max(np.abs(dose_error))),
        "per_frame_abs_dose_error_mean": float(np.mean(np.abs(dose_error))),
        "matched_frame_count": int(np.count_nonzero(~infeasible)),
        "infeasible_frame_count": int(infeasible.sum()),
        "infeasible_mask_sha256": array_sha256(infeasible.astype(np.uint8)),
        "scale_min": float(np.min(scales)),
        "scale_median": float(np.median(scales)),
        "scale_max": float(np.max(scales)),
        "scale_vector_sha256": array_sha256(scales),
        "max_scale_cap": float(max_scale),
        "kkt_float64_residual_max_abs": float(np.max(np.abs(residuals))),
        "float32_level_dot_max_abs": float(np.max(np.abs(applied @ level_axis))),
        "float32_rounding_constraint_abs_bound_max": max((float(row["float32_constraint_abs_bound"]) for row in per_frame_rows), default=0.0),
        "float32_rounding_constraint_abs_max": max((float(row["float32_constraint_abs"]) for row in per_frame_rows), default=0.0),
        "post_projection_norm_mean": float(np.linalg.norm(applied, axis=1).mean()),
        "post_projection_level_orthogonal": True,
        "dose_matching_per_frame": True,
        "per_frame_projection_diagnostics": per_frame_rows,
    }
    return output, metadata


def pcm_to_syncnet_integer_units(waveform: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Preserve floating PCM with the explicit amplitude scale used by int16 MFCC.

    The return type is float64 and values are multiplied by 32768. No rounding,
    int16 conversion, clipping, or saturation occurs, including when samples
    exceed the nominal [-1,1] PCM interval.
    """

    value = np.asarray(waveform)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("SyncNet waveform must be finite nonempty mono PCM")
    result = np.multiply(value.astype(np.float64, copy=False), np.float64(32768.0), dtype=np.float64)
    return result, {
        "input_dtype": str(value.dtype),
        "output_dtype": str(result.dtype),
        "integer_amplitude_scale": 32768,
        "quantization": "none",
        "clipping": "none",
        "peak_abs_input": float(np.max(np.abs(value))),
        "peak_over_one": bool(np.max(np.abs(value)) > 1.0),
        "input_sha256": array_sha256(value),
        "mfcc_input_sha256": array_sha256(result),
    }
