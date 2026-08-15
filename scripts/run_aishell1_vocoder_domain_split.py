#!/usr/bin/env python3
"""Split WavLM/HiFi-GAN domain mismatch from MFA-linear timing effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from knn_vc_retrieval import mfa_linear_target  # noqa: E402
from pilot_generate_mfa_linear import extend_tokens_for_feature_tail  # noqa: E402
from run_natural_anchor_alpha_blend import (  # noqa: E402
    _read_wave,
)
from wavlm_knn_vc_adapter import (  # noqa: E402
    FEATURE_DIM,
    KNN_VC_REVISION,
    SAMPLE_RATE,
    WavLMInterface,
    WavLMKNNVCAdapter,
)

KNN_VC_LOCAL = REPO / "third_party/knn-vc"
VOCODER_VARIANTS = ("regular", "prematched")
FEATURE_CONDITIONS = ("natural_direct", "raw_tts_direct", "mfa_linear")
CONDITIONS = tuple(
    f"{feature_condition}_{vocoder}"
    for feature_condition in FEATURE_CONDITIONS
    for vocoder in VOCODER_VARIANTS
)
DEFAULT_SPEAKERS = ("S0765", "S0901", "S0912")
ALL_N25_SPEAKERS = ("S0765", "S0901", "S0906", "S0912", "S0913")


def load_inputs(
    cohort_path: Path,
    tts_meta_path: Path,
    tokens_path: Path,
    reference_mfa_path: Path,
    speakers: Sequence[str],
) -> list[dict[str, Any]]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    tts_meta = json.loads(tts_meta_path.read_text(encoding="utf-8"))
    tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_mfa_path.read_text(encoding="utf-8"))
    wanted = tuple(dict.fromkeys(str(speaker) for speaker in speakers))
    if not wanted:
        raise ValueError("at least one speaker is required")
    records = [row for row in cohort.get("records", []) if str(row.get("speaker_id")) in wanted]
    if len(records) != 5 * len(wanted):
        raise ValueError(f"expected five utterances per selected speaker, found {len(records)}")
    if any(sum(str(row.get("speaker_id")) == speaker for row in records) != 5 for speaker in wanted):
        raise ValueError("selected speaker quota is incomplete")
    tts_by = {str(row["sample_id"]): row for row in tts_meta.get("results", {}).values()}
    token_by = {str(key): row for key, row in tokens.get("records", {}).items()}
    reference_by = {str(row["sample_id"]): row for row in reference.get("results", {}).values()}
    output: list[dict[str, Any]] = []
    for row in records:
        sid = str(row["sample_id"])
        tts = tts_by.get(sid)
        side = token_by.get(sid)
        reference_row = reference_by.get(sid)
        if tts is None or side is None or reference_row is None:
            raise ValueError(f"missing metadata for sample {sid}")
        if (
            str(tts.get("paired_key")) != str(row["paired_key"])
            or str(side.get("paired_key")) != str(row["paired_key"])
            or str(side.get("speaker_id")) != str(row["speaker_id"])
        ):
            raise ValueError(f"identity mismatch for sample {sid}")
        natural_path = Path(str(row["audio_path"])).resolve()
        tts_path = Path(str(tts["canonical_16k_audio"])).resolve()
        for path, expected in (
            (natural_path, row["natural_source_sha256"]),
            (tts_path, tts["canonical_audio_sha256"]),
        ):
            if not path.is_file() or sha256_file(path) != str(expected):
                raise ValueError(f"source hash mismatch for sample {sid}: {path}")
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
            "reference_mfa": reference_row,
        })
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def exact_length(values: np.ndarray, target_samples: int) -> tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if values.size > target_samples:
        return values[:target_samples].copy(), "right_crop"
    if values.size < target_samples:
        return np.pad(values, (0, target_samples - values.size)).astype(np.float32), "right_zero_pad"
    return values.copy(), "none"


def feature_distance(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("feature distance requires [frames, dimensions] with equal dimensions")
    frames = min(left.shape[0], right.shape[0])
    if frames <= 0:
        raise ValueError("feature distance requires non-empty sequences")
    left = left[:frames].float()
    right = right[:frames].float().to(left.device)
    cosine = 1.0 - torch.nn.functional.cosine_similarity(left, right, dim=-1)
    return {
        "cosine_distance": float(cosine.mean().cpu()),
        "mse": float(torch.mean((left - right) ** 2).cpu()),
        "compared_frames": frames,
    }


def make_feature_condition(
    feature_condition: str,
    natural_features: torch.Tensor,
    tts_features: torch.Tensor,
    mfa_features: torch.Tensor,
) -> torch.Tensor:
    if feature_condition == "natural_direct":
        return natural_features
    if feature_condition == "raw_tts_direct":
        return tts_features
    if feature_condition == "mfa_linear":
        return mfa_features
    raise ValueError(f"unsupported feature condition: {feature_condition}")


def load_adapter(device: str, prematched: bool) -> WavLMKNNVCAdapter:
    model = torch.hub.load(
        str(KNN_VC_LOCAL),
        "knn_vc",
        source="local",
        prematched=prematched,
        pretrained=True,
        device=device,
    )
    adapter = WavLMKNNVCAdapter(model, device=device)
    checkpoint = "prematch_g_02500000.pt" if prematched else "g_02500000.pt"
    adapter.interface = WavLMInterface(
        repository="bshall/knn-vc",
        revision=KNN_VC_REVISION,
        sample_rate=SAMPLE_RATE,
        frame_stride_samples=320,
        selected_layer=6,
        feature_dim=FEATURE_DIM,
        prematched_vocoder=prematched,
        frozen=True,
        loudness_normalization=False,
        wavlm_checkpoint_sha256=adapter._checkpoint_sha256("WavLM-Large.pt"),
        vocoder_checkpoint_sha256=adapter._checkpoint_sha256(checkpoint),
    )
    return adapter


def _audio_qc(values: np.ndarray, sample_rate: int, natural_samples: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "sample_rate": int(sample_rate),
        "sample_count": int(values.size),
        "duration_s": float(values.size / sample_rate),
        "natural_duration_s": float(natural_samples / SAMPLE_RATE),
        "finite": bool(np.isfinite(values).all()),
        "peak": float(np.max(np.abs(values))) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "clipped_sample_count": int(np.sum(np.abs(values) >= 0.999)),
        "exact_natural_length": bool(values.size == natural_samples),
    }


def _load_reference_values(row: Mapping[str, Any]) -> np.ndarray | None:
    reference = row.get("reference_mfa")
    if not reference or not reference.get("audio_path"):
        return None
    return _read_wave(str(reference["audio_path"]))


def generate(
    inputs: Sequence[Mapping[str, Any]],
    outdir: Path,
    device: str,
) -> dict[str, Any]:
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    adapters = {variant: load_adapter(device, variant == "prematched") for variant in VOCODER_VARIANTS}
    feature_adapter = adapters["prematched"]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(inputs, 1):
        sid = str(row["sample_id"])
        try:
            natural_audio = _read_wave(str(row["natural_path"]))
            tts_audio = _read_wave(str(row["tts_path"]))
            natural_tensor = torch.from_numpy(natural_audio).unsqueeze(0)
            tts_tensor = torch.from_numpy(tts_audio).unsqueeze(0)
            natural_features = feature_adapter.extract(natural_tensor)
            tts_features = feature_adapter.extract(tts_tensor)
            natural_tokens, natural_tail = extend_tokens_for_feature_tail(
                list(row["natural_tokens"]), natural_features.shape[0]
            )
            tts_tokens, tts_tail = extend_tokens_for_feature_tail(
                list(row["tts_tokens"]), tts_features.shape[0]
            )
            mfa_features, mfa_meta = mfa_linear_target(
                natural_features.shape[0],
                tts_features,
                natural_tokens,
                tts_tokens,
            )
            for condition in CONDITIONS:
                feature_condition, vocoder_variant = condition.rsplit("_", 1)
                adapter = adapters[vocoder_variant]
                conditioning = make_feature_condition(
                    feature_condition,
                    natural_features,
                    tts_features,
                    mfa_features,
                )
                raw_output = adapter.vocode(conditioning).numpy().astype(np.float32)
                output = raw_output
                length_action = "native_feature_length"
                if feature_condition == "mfa_linear":
                    output, length_action = exact_length(output, natural_audio.size)
                if output.size == 0 or not np.isfinite(output).all():
                    raise FloatingPointError(f"invalid output at {sid}/{condition}")
                path = outdir / "audio" / condition / f"{sid}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, output, SAMPLE_RATE, subtype="FLOAT")
                encoded = feature_adapter.extract(torch.from_numpy(output).unsqueeze(0))
                qc = _audio_qc(output, SAMPLE_RATE, natural_audio.size)
                result: dict[str, Any] = {
                    "sample_id": sid,
                    "paired_key": row["paired_key"],
                    "speaker_id": row["speaker_id"],
                    "feature_condition": feature_condition,
                    "vocoder_variant": vocoder_variant,
                    "condition": condition,
                    "audio_path": str(path),
                    "audio_sha256": sha256_file(path),
                    "natural_source_sha256": row["natural_sha256"],
                    "tts_source_sha256": row["tts_sha256"],
                    "natural_samples": int(natural_audio.size),
                    "raw_decoder_samples": int(raw_output.size),
                    "length_action": length_action,
                    "audio_qc": qc,
                    "conditioning_shape": list(conditioning.shape),
                    "conditioning_feature_distance": {
                        "output_to_conditioning": feature_distance(encoded, conditioning),
                        "output_to_natural": feature_distance(encoded, natural_features),
                        "output_to_raw_tts": feature_distance(encoded, tts_features),
                        "output_to_mfa_linear": feature_distance(encoded, mfa_features),
                    },
                    "mfa_linear_meta": {
                        **mfa_meta,
                        "natural_feature_tail_extension_s": natural_tail,
                        "tts_feature_tail_extension_s": tts_tail,
                    },
                }
                if condition == "mfa_linear_prematched":
                    reference_values = _load_reference_values(row)
                    if reference_values is not None:
                        if reference_values.shape != output.shape:
                            raise ValueError(f"reference length mismatch at {sid}")
                        difference = np.abs(reference_values - output)
                        result["reference_max_abs_diff"] = float(np.max(difference))
                        result["reference_rmse"] = float(np.sqrt(np.mean(difference * difference)))
                        result["reference_numerically_equal"] = bool(np.max(difference) <= 1e-5)
                results.append(result)
                print(
                    f"OK {index}/{len(inputs)} {condition} samples={output.size} "
                    f"peak={qc['peak']:.4f}",
                    flush=True,
                )
        except Exception as exc:
            failures.append({"sample_id": sid, "error": str(exc)})
            print(f"FAIL sample={sid}: {exc}", flush=True)

    summary = {
        "schema_version": 1,
        "experiment": "aishell1_wavlm_vocoder_domain_split",
        "status": "complete" if not failures and len(results) == len(inputs) * len(CONDITIONS) else "incomplete",
        "speakers": sorted({str(row["speaker_id"]) for row in inputs}),
        "sample_count": len(inputs),
        "conditions": list(CONDITIONS),
        "expected_outputs": len(inputs) * len(CONDITIONS),
        "outputs": len(results),
        "failures": failures,
        "device": device,
        "model_variants": {variant: adapters[variant].metadata() for variant in VOCODER_VARIANTS},
        "knn_vc_revision": KNN_VC_REVISION,
        "results": results,
    }
    write_json(outdir / "summary.json", summary)
    return summary


def write_report(summary: Mapping[str, Any], outdir: Path) -> None:
    rows = list(summary["results"])
    lines = [
        "# AISHELL-1 WavLM vocoder-domain split",
        "",
        "## Protocol",
        "",
        "Audio-only comparison of natural direct, raw-TTS direct and MFA-linear WavLM features through regular and prematched frozen kNN-VC HiFi-GAN. Raw-TTS direct keeps native decoder length; only MFA-linear is placed on the natural clock.",
        "",
        "## Summary",
        "",
        "| Speaker | Condition | Mean duration s | Mean peak | Mean RMS | Mean output→condition distance | Clipped |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for speaker in summary["speakers"]:
        for condition in CONDITIONS:
            subset = [row for row in rows if row["speaker_id"] == speaker and row["condition"] == condition]
            if not subset:
                continue
            qc = [row["audio_qc"] for row in subset]
            distances = [row["conditioning_feature_distance"]["output_to_conditioning"]["cosine_distance"] for row in subset]
            lines.append(
                f"| {speaker} | {condition} | {np.mean([x['duration_s'] for x in qc]):.3f} | "
                f"{np.mean([x['peak'] for x in qc]):.4f} | {np.mean([x['rms'] for x in qc]):.4f} | "
                f"{np.mean(distances):.4f} | {sum(x['clipped_sample_count'] for x in qc)} |"
            )
    reference = [row for row in rows if row["condition"] == "mfa_linear_prematched" and "reference_numerically_equal" in row]
    if reference:
        lines.extend([
            "",
            "## Reference check",
            "",
            f"MFA-linear prematched versus existing reference: numerical equality <=1e-5 in {sum(bool(x['reference_numerically_equal']) for x in reference)}/{len(reference)} samples; worst max abs diff={max(float(x['reference_max_abs_diff']) for x in reference):.6g}.",
        ])
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "This is an audio-only domain diagnostic. It does not establish speaker preservation, pronunciation, naturalness or lip-sync improvement. SyncNet was not used for selection.",
    ])
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--reference-mfa", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--speaker", action="append", dest="speakers", help="speaker to include; repeat five times for full n25")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    speakers = tuple(args.speakers) if args.speakers else DEFAULT_SPEAKERS
    inputs = load_inputs(
        args.cohort.resolve(),
        args.tts_meta.resolve(),
        args.tokens.resolve(),
        args.reference_mfa.resolve(),
        speakers,
    )
    summary = generate(inputs, args.outdir.resolve(), args.device)
    write_json(
        args.outdir.resolve() / "manifest.json",
        {
            "schema_version": 1,
            "experiment": "aishell1_wavlm_vocoder_domain_split",
            "cohort": str(args.cohort.resolve()),
            "tts_meta": str(args.tts_meta.resolve()),
            "tokens": str(args.tokens.resolve()),
            "reference_mfa": str(args.reference_mfa.resolve()),
            "speakers": sorted({str(row["speaker_id"]) for row in inputs}),
            "conditions": list(CONDITIONS),
            "records": inputs,
        },
    )
    write_report(summary, args.outdir.resolve())
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
