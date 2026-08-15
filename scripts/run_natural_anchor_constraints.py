#!/usr/bin/env python3
"""Generate alpha=1 natural-anchor outputs with residual trajectory constraints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_natural_anchor_alpha_blend import (
    KNN_VC_LOCAL,
    KNN_VC_REVISION,
    SPEAKERS,
    _cosine,
    _load_inputs,
    _read_wave,
    extract_audio_features,
    sha256_file,
    write_json,
)
from knn_vc_retrieval import mfa_linear_target
from pilot_generate_mfa_linear import extend_tokens_for_feature_tail
from wavlm_knn_vc_adapter import SAMPLE_RATE, WavLMKNNVCAdapter

CONDITIONS: dict[str, tuple[float | None, float | None]] = {
    "unconstrained": (None, None),
    "norm_q90": (0.90, None),
    "norm_q75": (0.75, None),
    "velocity_q90": (None, 0.90),
    "norm_q90_velocity_q90": (0.90, 0.90),
    "norm_q75_velocity_q90": (0.75, 0.90),
}


def _quantile(values: torch.Tensor, quantile: float | None) -> torch.Tensor | None:
    if quantile is None:
        return None
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    return torch.quantile(values, quantile)


def constrain_residual(residual: torch.Tensor, norm_quantile: float | None, velocity_quantile: float | None) -> tuple[torch.Tensor, dict[str, Any]]:
    if residual.ndim != 2 or residual.shape[0] < 2:
        raise ValueError("residual must have shape [frames, dimensions]")
    if not torch.isfinite(residual).all():
        raise FloatingPointError("residual contains non-finite values")
    original_norm = torch.linalg.vector_norm(residual, dim=-1)
    original_velocity = torch.linalg.vector_norm(residual[1:] - residual[:-1], dim=-1)
    norm_limit = _quantile(original_norm, norm_quantile)
    velocity_limit = _quantile(original_velocity, velocity_quantile)
    constrained = residual.clone()
    norm_clipped = torch.zeros_like(original_norm, dtype=torch.bool)
    if norm_limit is not None:
        norm_clipped = original_norm > norm_limit
        scale = torch.minimum(torch.ones_like(original_norm), norm_limit / (original_norm + 1e-12))
        constrained = constrained * scale[:, None]
    velocity_clipped = torch.zeros_like(original_velocity, dtype=torch.bool)
    if velocity_limit is not None:
        for index in range(1, constrained.shape[0]):
            step = constrained[index] - constrained[index - 1]
            step_norm = torch.linalg.vector_norm(step)
            if bool(step_norm > velocity_limit):
                constrained[index] = constrained[index - 1] + step * (velocity_limit / (step_norm + 1e-12))
        constrained_velocity = torch.linalg.vector_norm(constrained[1:] - constrained[:-1], dim=-1)
        velocity_clipped = constrained_velocity < original_velocity - 1e-8
    constrained_norm = torch.linalg.vector_norm(constrained, dim=-1)
    constrained_velocity = torch.linalg.vector_norm(constrained[1:] - constrained[:-1], dim=-1)
    stats = {
        "norm_quantile": norm_quantile,
        "velocity_quantile": velocity_quantile,
        "residual_norm_limit": float(norm_limit.item()) if norm_limit is not None else None,
        "residual_velocity_limit": float(velocity_limit.item()) if velocity_limit is not None else None,
        "original_residual_norm_mean": float(original_norm.mean().item()),
        "original_residual_norm_p95": float(torch.quantile(original_norm, 0.95).item()),
        "constrained_residual_norm_mean": float(constrained_norm.mean().item()),
        "constrained_residual_norm_p95": float(torch.quantile(constrained_norm, 0.95).item()),
        "original_residual_velocity_mean": float(original_velocity.mean().item()),
        "original_residual_velocity_p95": float(torch.quantile(original_velocity, 0.95).item()),
        "constrained_residual_velocity_mean": float(constrained_velocity.mean().item()),
        "constrained_residual_velocity_p95": float(torch.quantile(constrained_velocity, 0.95).item()),
        "norm_clipped_fraction": float(norm_clipped.float().mean().item()),
        "velocity_clipped_fraction": float(velocity_clipped.float().mean().item()),
    }
    return constrained, stats


def _write_report(summary: Mapping[str, Any], outdir: Path) -> None:
    rows = list(summary["results"])
    lines = ["# Natural-anchor residual norm and velocity constraints", "", "## Protocol", "", "Alpha is fixed at 1. The unconstrained residual is `d = z_tts_warp - z_natural`; constraints alter only `d` before `z_out = z_natural + d`. Quantile caps are computed separately per utterance from its unconstrained residual trajectory. No Wav2Lip/SyncNet was run.", "", "## Summary", "", "| Speaker | Condition | Mean peak | Mean crest | Mean flux | Mean norm clipped | Mean velocity clipped | Mean output→natural cosine |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for speaker in SPEAKERS:
        for condition in CONDITIONS:
            subset = [row for row in rows if row["speaker_id"] == speaker and row["condition"] == condition]
            if not subset:
                continue
            acoustic = [row["acoustic_features"] for row in subset]
            lines.append(f"| {speaker} | {condition} | {np.mean([row['peak'] for row in subset]):.4f} | {np.mean([item['crest_factor'] for item in acoustic]):.3f} | {np.mean([item['spectral_flux'] for item in acoustic]):.6f} | {np.mean([row['constraint']['norm_clipped_fraction'] for row in subset]):.3f} | {np.mean([row['constraint']['velocity_clipped_fraction'] for row in subset]):.3f} | {np.mean([row['output_to_natural_cosine'] for row in subset]):.4f} |")
    reference = [row for row in rows if row["condition"] == "unconstrained" and "reference_numerically_equal" in row]
    lines.extend(["", "## Reference check", "", f"Unconstrained alpha=1 numerical equality to existing MFA-linear (max abs ≤1e-5): `{sum(bool(row['reference_numerically_equal']) for row in reference)}/{len(reference)}`.", "", "## Interpretation", "", "This is a diagnostic audio-only constraint sweep. A lower peak or smoother feature trajectory is not sufficient evidence of better phone content, speaker similarity, intelligibility, or lip-sync. A candidate must pass audio listening/ASR checks before downstream video evaluation."])
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(inputs: Sequence[Mapping[str, Any]], outdir: Path, device: str, reference_mfa_path: Path) -> dict[str, Any]:
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    adapter = WavLMKNNVCAdapter.load_pretrained(device=device, source=KNN_VC_LOCAL, revision=KNN_VC_REVISION)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(inputs, 1):
        sid = str(row["sample_id"])
        try:
            natural_audio = _read_wave(row["natural_path"])
            tts_audio = _read_wave(row["tts_path"])
            natural_features = adapter.extract(torch.from_numpy(natural_audio).unsqueeze(0))
            tts_features = adapter.extract(torch.from_numpy(tts_audio).unsqueeze(0))
            natural_tokens, natural_tail = extend_tokens_for_feature_tail(list(row["natural_tokens"]), natural_features.shape[0])
            tts_tokens, tts_tail = extend_tokens_for_feature_tail(list(row["tts_tokens"]), tts_features.shape[0])
            tts_target, mfa_meta = mfa_linear_target(natural_features.shape[0], tts_features, natural_tokens, tts_tokens)
            residual = tts_target - natural_features
            for condition, (norm_quantile, velocity_quantile) in CONDITIONS.items():
                constrained_residual, constraint_stats = constrain_residual(residual, norm_quantile, velocity_quantile)
                condition_features = natural_features + constrained_residual
                output = adapter.vocode(condition_features).detach().cpu().numpy().reshape(-1).astype(np.float32)
                if output.size > natural_audio.size:
                    output = output[: natural_audio.size]
                    length_action = "right_crop"
                elif output.size < natural_audio.size:
                    output = np.pad(output, (0, natural_audio.size - output.size)).astype(np.float32)
                    length_action = "right_zero_pad"
                else:
                    length_action = "none"
                if output.size != natural_audio.size or not np.isfinite(output).all():
                    raise FloatingPointError(f"invalid output at {sid}/{condition}")
                label = condition
                path = outdir / "audio" / label / f"{sid}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, output, SAMPLE_RATE, subtype="FLOAT")
                encoded = adapter.extract(torch.from_numpy(output).unsqueeze(0))
                row_result: dict[str, Any] = {
                    "sample_id": sid,
                    "paired_key": row["paired_key"],
                    "speaker_id": row["speaker_id"],
                    "condition": condition,
                    "alpha": 1.0,
                    "audio_path": str(path),
                    "audio_sha256": sha256_file(path),
                    "natural_samples": int(natural_audio.size),
                    "output_samples": int(output.size),
                    "exact_natural_length": True,
                    "length_action": length_action,
                    "natural_feature_shape": list(natural_features.shape),
                    "tts_feature_shape": list(tts_features.shape),
                    "mfa_linear_meta": {**mfa_meta, "natural_feature_tail_extension_s": natural_tail, "tts_feature_tail_extension_s": tts_tail},
                    "constraint": constraint_stats,
                    "output_to_natural_cosine": _cosine(encoded, natural_features),
                    "output_to_condition_cosine": _cosine(encoded, condition_features),
                    "output_to_unconstrained_target_cosine": _cosine(encoded, tts_target),
                    "peak": float(np.max(np.abs(output))),
                    "clipped_sample_count": int(np.sum(np.abs(output) >= 0.999)),
                }
                row_result["acoustic_features"] = extract_audio_features(output)[0]
                if condition == "unconstrained":
                    reference_values = _read_wave(str(row["reference_mfa"]["audio_path"]))
                    difference = np.abs(output - reference_values)
                    row_result["reference_max_abs_diff"] = float(np.max(difference))
                    row_result["reference_rmse"] = float(np.sqrt(np.mean(difference * difference)))
                    row_result["reference_numerically_equal"] = bool(np.max(difference) <= 1e-5)
                results.append(row_result)
            print(f"OK {index}/{len(inputs)} sample={sid}", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sid, "error": str(exc)})
            print(f"FAIL sample={sid}: {exc}", flush=True)
    summary = {"schema_version": 1, "experiment": "natural_anchor_residual_constraints", "status": "complete" if not failures and len(results) == len(inputs) * len(CONDITIONS) else "incomplete", "speakers": list(SPEAKERS), "sample_count": len(inputs), "alpha": 1.0, "conditions": {name: {"norm_quantile": values[0], "velocity_quantile": values[1]} for name, values in CONDITIONS.items()}, "expected_outputs": len(inputs) * len(CONDITIONS), "outputs": len(results), "failures": failures, "device": device, "model": adapter.metadata(), "knn_vc_revision": KNN_VC_REVISION, "reference_mfa_summary": str(reference_mfa_path.resolve()), "results": results}
    write_json(outdir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--reference-mfa", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    inputs, _ = _load_inputs(args.cohort.resolve(), args.tts_meta.resolve(), args.tokens.resolve(), args.reference_mfa.resolve())
    summary = generate(inputs, args.outdir.resolve(), args.device, args.reference_mfa.resolve())
    write_json(args.outdir.resolve() / "manifest.json", {"schema_version": 1, "experiment": "natural_anchor_residual_constraints", "cohort": str(args.cohort.resolve()), "tts_meta": str(args.tts_meta.resolve()), "tokens": str(args.tokens.resolve()), "reference_mfa": str(args.reference_mfa.resolve()), "speakers": list(SPEAKERS), "alpha": 1.0, "conditions": {name: {"norm_quantile": values[0], "velocity_quantile": values[1]} for name, values in CONDITIONS.items()}, "records": inputs})
    _write_report(summary, args.outdir.resolve())
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
