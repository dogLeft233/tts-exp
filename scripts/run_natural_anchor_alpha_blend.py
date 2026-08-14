#!/usr/bin/env python3
"""Generate frozen natural-anchor feature blends for a small AISHELL-1 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_aishell1_n25_audio_artifacts import extract_audio_features  # noqa: E402
from knn_vc_retrieval import mfa_linear_target  # noqa: E402
from pilot_generate_mfa_linear import extend_tokens_for_feature_tail  # noqa: E402
from wavlm_knn_vc_adapter import KNN_VC_REVISION, SAMPLE_RATE, WavLMKNNVCAdapter  # noqa: E402

KNN_VC_LOCAL = REPO / "third_party/knn-vc"

SPEAKERS = ("S0765", "S0901", "S0912")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def blend_features(natural_features: torch.Tensor, tts_target: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if natural_features.shape != tts_target.shape:
        raise ValueError(f"feature shape mismatch: {tuple(natural_features.shape)} vs {tuple(tts_target.shape)}")
    if natural_features.ndim != 2:
        raise ValueError("features must have shape [frames, dimensions]")
    output = natural_features + alpha * (tts_target - natural_features)
    if not torch.isfinite(output).all():
        raise FloatingPointError("alpha blend produced non-finite features")
    return output


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["paired_key"]), str(row["speaker_id"])


def _load_inputs(cohort_path: Path, tts_meta_path: Path, tokens_path: Path, reference_mfa_path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    tts_meta = json.loads(tts_meta_path.read_text(encoding="utf-8"))
    tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_mfa_path.read_text(encoding="utf-8")) if reference_mfa_path else None
    records = [row for row in cohort.get("records", []) if str(row.get("speaker_id")) in SPEAKERS]
    tts_by = {str(row["sample_id"]): row for row in tts_meta.get("results", {}).values()}
    token_by = {str(key): row for key, row in tokens.get("records", {}).items()}
    reference_by = {str(row["sample_id"]): row for row in reference.get("results", {}).values()} if reference else {}
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort" or len(records) != 15:
        raise ValueError("expected exactly 15 records from S0765/S0901/S0912")
    if any(sum(str(row.get("speaker_id")) == speaker for row in records) != 5 for speaker in SPEAKERS):
        raise ValueError("expected five utterances per selected speaker")
    if set(str(row["sample_id"]) for row in records) != set(tts_by).intersection(set(str(row["sample_id"]) for row in records)):
        raise ValueError("TTS metadata does not cover selected samples")
    output: list[dict[str, Any]] = []
    for row in records:
        sid = str(row["sample_id"]); tts = tts_by.get(sid); side = token_by.get(sid)
        if tts is None or side is None:
            raise ValueError(f"missing metadata for sample {sid}")
        if _identity(row) != _identity(tts) or str(side.get("paired_key")) != str(row["paired_key"]) or str(side.get("speaker_id")) != str(row["speaker_id"]):
            raise ValueError(f"identity mismatch for sample {sid}")
        natural_path = Path(str(row["audio_path"])).resolve()
        tts_path = Path(str(tts["canonical_16k_audio"])).resolve()
        for path, expected in ((natural_path, row["natural_source_sha256"]), (tts_path, tts["canonical_audio_sha256"])):
            if not path.is_file() or sha256_file(path) != str(expected):
                raise ValueError(f"source hash mismatch for sample {sid}: {path}")
        if reference is not None and sid not in reference_by:
            raise ValueError(f"reference MFA summary misses sample {sid}")
        output.append({
            "sample_id": sid,
            "paired_key": row["paired_key"],
            "speaker_id": row["speaker_id"],
            "split": row["split"],
            "transcript": row["transcript"],
            "natural_path": str(natural_path),
            "natural_sha256": str(row["natural_source_sha256"]),
            "tts_path": str(tts_path),
            "tts_sha256": str(tts["canonical_audio_sha256"]),
            "natural_tokens": side["natural"]["tokens"],
            "tts_tokens": side["tts"]["tokens"],
            "tts_source_sha256": str(tts.get("source_audio_sha256", "")),
            "reference_mfa": reference_by.get(sid, {}),
        })
    return output, {"cohort": cohort, "tts_meta": tts_meta, "tokens": tokens, "reference_mfa": reference}


def _read_wave(path: str) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if values.ndim != 1 or int(sample_rate) != SAMPLE_RATE or not np.isfinite(values).all():
        raise ValueError(f"invalid 16 kHz mono audio: {path}")
    return np.asarray(values, dtype=np.float32)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())


def _alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


def generate(inputs: Sequence[Mapping[str, Any]], outdir: Path, device: str, reference_mfa_path: Path | None) -> dict[str, Any]:
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
            for alpha in ALPHAS:
                condition = blend_features(natural_features, tts_target, alpha)
                output = adapter.vocode(condition).detach().cpu().numpy().reshape(-1).astype(np.float32)
                if output.size > natural_audio.size:
                    output = output[: natural_audio.size]
                    length_action = "right_crop"
                elif output.size < natural_audio.size:
                    output = np.pad(output, (0, natural_audio.size - output.size)).astype(np.float32)
                    length_action = "right_zero_pad"
                else:
                    length_action = "none"
                if output.size != natural_audio.size or not np.isfinite(output).all():
                    raise FloatingPointError(f"invalid output length or finite state at {sid} alpha={alpha}")
                label = _alpha_label(alpha)
                path = outdir / "audio" / label / f"{sid}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, output, SAMPLE_RATE, subtype="FLOAT")
                encoded = adapter.extract(torch.from_numpy(output).unsqueeze(0))
                row_result = {
                    "sample_id": sid,
                    "paired_key": row["paired_key"],
                    "speaker_id": row["speaker_id"],
                    "alpha": alpha,
                    "alpha_label": label,
                    "audio_path": str(path),
                    "audio_sha256": sha256_file(path),
                    "natural_samples": int(natural_audio.size),
                    "output_samples": int(output.size),
                    "exact_natural_length": True,
                    "length_action": length_action,
                    "natural_feature_shape": list(natural_features.shape),
                    "tts_feature_shape": list(tts_features.shape),
                    "mfa_linear_meta": {**mfa_meta, "natural_feature_tail_extension_s": natural_tail, "tts_feature_tail_extension_s": tts_tail},
                    "output_to_blend_cosine": _cosine(encoded, condition),
                    "output_to_natural_cosine": _cosine(encoded, natural_features),
                    "output_to_tts_target_cosine": _cosine(encoded, tts_target),
                    "peak": float(np.max(np.abs(output))),
                    "clipped_sample_count": int(np.sum(np.abs(output) >= 0.999)),
                }
                acoustic, _ = extract_audio_features(output)
                row_result["acoustic_features"] = acoustic
                reference_row = row.get("reference_mfa", {})
                if alpha == 1.0 and reference_row:
                    reference_values = _read_wave(str(reference_row["audio_path"]))
                    if reference_values.shape != output.shape:
                        raise ValueError(f"alpha=1 reference length mismatch at {sid}")
                    difference = np.abs(output - reference_values)
                    row_result["reference_alpha1_audio_sha256"] = reference_row.get("audio_sha256")
                    row_result["alpha1_matches_reference_sha256"] = row_result["audio_sha256"] == reference_row.get("audio_sha256")
                    row_result["alpha1_reference_max_abs_diff"] = float(np.max(difference))
                    row_result["alpha1_reference_rmse"] = float(np.sqrt(np.mean(difference * difference)))
                    row_result["alpha1_matches_reference_numerically"] = bool(np.max(difference) <= 1e-5)
                results.append(row_result)
            print(f"OK {index}/{len(inputs)} sample={sid}", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sid, "error": str(exc)})
            print(f"FAIL sample={sid}: {exc}", flush=True)
    summary = {"schema_version": 1, "experiment": "natural_anchor_mfa_residual_alpha_blend", "status": "complete" if not failures and len(results) == len(inputs) * len(ALPHAS) else "incomplete", "speakers": list(SPEAKERS), "sample_count": len(inputs), "alphas": list(ALPHAS), "expected_outputs": len(inputs) * len(ALPHAS), "outputs": len(results), "failures": failures, "device": device, "model": adapter.metadata(), "knn_vc_revision": KNN_VC_REVISION, "reference_mfa_summary": str(reference_mfa_path.resolve()) if reference_mfa_path else None, "results": results}
    write_json(outdir / "summary.json", summary)
    return summary


def write_report(summary: Mapping[str, Any], outdir: Path) -> None:
    rows = list(summary["results"])
    lines = ["# Natural-anchor MFA residual alpha blend", "", "## Protocol", "", "`z_alpha = z_natural + alpha * (z_tts_warp - z_natural)` on the natural WavLM-L6 frame grid. Alpha 0 is natural-feature direct resynthesis; alpha 1 is the full MFA-linear target. This is audio-only; no Wav2Lip/SyncNet was run.", "", "## Summary", "", "| Speaker | Alpha | Mean peak | Mean crest | Mean spectral flux | Mean output→natural cosine | Mean output→TTS-target cosine | Clipped |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for speaker in SPEAKERS:
        for alpha in ALPHAS:
            subset = [row for row in rows if row["speaker_id"] == speaker and float(row["alpha"]) == alpha]
            if not subset:
                continue
            acoustic = [row["acoustic_features"] for row in subset]
            lines.append(f"| {speaker} | {alpha:.2f} | {np.mean([row['peak'] for row in subset]):.4f} | {np.mean([row['crest_factor'] for row in acoustic]):.3f} | {np.mean([row['spectral_flux'] for row in acoustic]):.6f} | {np.mean([row['output_to_natural_cosine'] for row in subset]):.4f} | {np.mean([row['output_to_tts_target_cosine'] for row in subset]):.4f} | {sum(row['clipped_sample_count'] for row in subset)} |")
    alpha1 = [row for row in rows if float(row["alpha"]) == 1.0 and "alpha1_matches_reference_numerically" in row]
    max_diff = max((float(row["alpha1_reference_max_abs_diff"]) for row in alpha1), default=None)
    max_rmse = max((float(row["alpha1_reference_rmse"]) for row in alpha1), default=None)
    lines.extend(["", "## Reference check", "", f"Alpha=1 exact hash matches: `{sum(bool(row['alpha1_matches_reference_sha256']) for row in alpha1)}/{len(alpha1)}`; numerical max-absolute-difference ≤1e-5: `{sum(bool(row['alpha1_matches_reference_numerically']) for row in alpha1)}/{len(alpha1)}`; worst max difference={max_diff}; worst RMSE={max_rmse}.", "", "## Limitations", "", "The experiment changes only feature blend strength. It does not add F0/energy conditioning, residual norm caps, boundary smoothing, speaker embeddings, or a learned decoder. Lower alpha is therefore a diagnostic interpolation, not a complete natural-speaker-preserving model."])
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    write_json(args.outdir.resolve() / "manifest.json", {"schema_version": 1, "experiment": "natural_anchor_mfa_residual_alpha_blend", "cohort": str(args.cohort.resolve()), "tts_meta": str(args.tts_meta.resolve()), "tokens": str(args.tokens.resolve()), "reference_mfa": str(args.reference_mfa.resolve()), "speakers": list(SPEAKERS), "alphas": list(ALPHAS), "records": inputs})
    write_report(summary, args.outdir.resolve())
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
