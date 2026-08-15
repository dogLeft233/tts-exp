#!/usr/bin/env python3
"""Generate alpha=1 outputs with silence anchoring and boundary smoothing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from knn_vc_retrieval import mfa_linear_target  # noqa: E402
from pilot_generate_mfa_linear import extend_tokens_for_feature_tail  # noqa: E402
from run_natural_anchor_alpha_blend import (  # noqa: E402
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
from wavlm_knn_vc_adapter import SAMPLE_RATE, WavLMKNNVCAdapter  # noqa: E402

CONDITIONS = (
    "unconstrained",
    "silence_anchor",
    "boundary_smooth_r2",
    "boundary_smooth_r3",
    "silence_anchor_boundary_smooth_r2",
    "silence_anchor_boundary_smooth_r3",
)


def _frame_times(frame_count: int) -> torch.Tensor:
    return (torch.arange(frame_count, dtype=torch.float32) + 0.5) * 320.0 / SAMPLE_RATE


def silence_frame_mask(tokens: Sequence[Mapping[str, Any]], frame_count: int) -> torch.Tensor:
    times = _frame_times(frame_count)
    mask = torch.zeros(frame_count, dtype=torch.bool)
    for token in tokens:
        if not bool(token.get("is_silence", False) or token.get("is_non_speech", False)):
            continue
        mask |= (times >= float(token["start_s"])) & (times < float(token["end_s"]))
    return mask


def boundary_frame_mask(tokens: Sequence[Mapping[str, Any]], frame_count: int, radius_frames: int) -> torch.Tensor:
    if radius_frames <= 0:
        raise ValueError("radius_frames must be positive")
    times = _frame_times(frame_count)
    radius_s = radius_frames * 320.0 / SAMPLE_RATE
    mask = torch.zeros(frame_count, dtype=torch.bool)
    for token in tokens[:-1]:
        boundary = float(token["end_s"])
        mask |= torch.abs(times - boundary) <= radius_s
    return mask


def _smooth_trajectory(residual: torch.Tensor) -> torch.Tensor:
    channels = residual.shape[1]
    kernel = torch.tensor([0.25, 0.5, 0.25], dtype=residual.dtype, device=residual.device).view(1, 1, 3).expand(channels, 1, 3)
    padded = F.pad(residual.T.unsqueeze(0), (1, 1), mode="replicate")
    return F.conv1d(padded, kernel, groups=channels).squeeze(0).T


def apply_condition(residual: torch.Tensor, tokens: Sequence[Mapping[str, Any]], condition: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    output = residual.clone()
    use_silence = condition.startswith("silence_anchor")
    radius = 0
    if "boundary_smooth_r2" in condition:
        radius = 2
    elif "boundary_smooth_r3" in condition:
        radius = 3
    silence_mask = silence_frame_mask(tokens, residual.shape[0])
    boundary_mask = boundary_frame_mask(tokens, residual.shape[0], radius) if radius else torch.zeros(residual.shape[0], dtype=torch.bool)
    if radius:
        smoothed = _smooth_trajectory(output)
        output[boundary_mask.to(output.device)] = smoothed[boundary_mask.to(output.device)]
    if use_silence:
        output[silence_mask.to(output.device)] = 0.0
    stats = {"silence_anchor": use_silence, "boundary_smoothing_radius_frames": radius, "silence_frame_fraction": float(silence_mask.float().mean().item()), "boundary_frame_fraction": float(boundary_mask.float().mean().item()), "residual_l2_change_fraction": float((torch.linalg.vector_norm(output - residual, dim=-1) > 1e-8).float().mean().item())}
    return output, stats


def _write_report(summary: Mapping[str, Any], outdir: Path) -> None:
    rows = list(summary["results"])
    lines = ["# Natural silence and phone-boundary residual experiment", "", "## Protocol", "", "Alpha is fixed at 1. Conditions change only the residual around natural MFA silence regions and phone boundaries. Speech interior remains unchanged except for the local smoothing window. No Wav2Lip/SyncNet was run.", "", "## Summary", "", "| Speaker | Condition | Mean peak | Mean crest | Mean flux | Silence fraction | Boundary fraction | Output→natural cosine |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for speaker in SPEAKERS:
        for condition in CONDITIONS:
            subset = [row for row in rows if row["speaker_id"] == speaker and row["condition"] == condition]
            if not subset:
                continue
            acoustic = [row["acoustic_features"] for row in subset]
            lines.append(f"| {speaker} | {condition} | {np.mean([row['peak'] for row in subset]):.4f} | {np.mean([item['crest_factor'] for item in acoustic]):.3f} | {np.mean([item['spectral_flux'] for item in acoustic]):.6f} | {np.mean([row['condition_stats']['silence_frame_fraction'] for row in subset]):.3f} | {np.mean([row['condition_stats']['boundary_frame_fraction'] for row in subset]):.3f} | {np.mean([row['output_to_natural_cosine'] for row in subset]):.4f} |")
    reference = [row for row in rows if row["condition"] == "unconstrained"]
    lines.extend(["", "## Reference check", "", f"Unconstrained alpha=1 numerical equality to existing MFA-linear (max abs ≤1e-5): `{sum(bool(row.get('reference_numerically_equal')) for row in reference)}/{len(reference)}`.", "", "## Interpretation limits", "", "This is an audio-only diagnostic. Boundary smoothing can trade local artifact reduction for phone sharpness; silence anchoring can change the vocoder's silence realization. Lower peak/crest alone does not establish better speaker similarity, intelligibility, or lip-sync."])
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
            for condition in CONDITIONS:
                constrained_residual, condition_stats = apply_condition(residual, natural_tokens, condition)
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
                path = outdir / "audio" / condition / f"{sid}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, output, SAMPLE_RATE, subtype="FLOAT")
                encoded = adapter.extract(torch.from_numpy(output).unsqueeze(0))
                result: dict[str, Any] = {
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
                    "condition_stats": condition_stats,
                    "output_to_natural_cosine": _cosine(encoded, natural_features),
                    "output_to_condition_cosine": _cosine(encoded, condition_features),
                    "output_to_unconstrained_target_cosine": _cosine(encoded, tts_target),
                    "peak": float(np.max(np.abs(output))),
                    "clipped_sample_count": int(np.sum(np.abs(output) >= 0.999)),
                    "acoustic_features": extract_audio_features(output)[0],
                }
                if condition == "unconstrained":
                    reference_values = _read_wave(str(row["reference_mfa"]["audio_path"]))
                    difference = np.abs(output - reference_values)
                    result["reference_max_abs_diff"] = float(np.max(difference))
                    result["reference_numerically_equal"] = bool(np.max(difference) <= 1e-5)
                results.append(result)
            print(f"OK {index}/{len(inputs)} sample={sid}", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sid, "error": str(exc)})
            print(f"FAIL sample={sid}: {exc}", flush=True)
    summary = {"schema_version": 1, "experiment": "natural_anchor_silence_boundary", "status": "complete" if not failures and len(results) == len(inputs) * len(CONDITIONS) else "incomplete", "speakers": list(SPEAKERS), "sample_count": len(inputs), "alpha": 1.0, "conditions": list(CONDITIONS), "expected_outputs": len(inputs) * len(CONDITIONS), "outputs": len(results), "failures": failures, "device": device, "model": adapter.metadata(), "knn_vc_revision": KNN_VC_REVISION, "reference_mfa_summary": str(reference_mfa_path.resolve()), "results": results}
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
    write_json(args.outdir.resolve() / "manifest.json", {"schema_version": 1, "experiment": "natural_anchor_silence_boundary", "cohort": str(args.cohort.resolve()), "tts_meta": str(args.tts_meta.resolve()), "tokens": str(args.tokens.resolve()), "reference_mfa": str(args.reference_mfa.resolve()), "speakers": list(SPEAKERS), "alpha": 1.0, "conditions": list(CONDITIONS), "records": inputs})
    _write_report(summary, args.outdir.resolve())
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
