#!/usr/bin/env python3
"""Generate the mfa_linear third arm for pilot utterances.

Frozen kNN-VC chain: WavLM-Large L6 features of natural and canonical TTS,
mapped onto the natural clock with mfa_linear_target (MFA tokens both
sides), vocoded by the frozen prematched HiFi-GAN, right-cropped/padded to
the exact natural sample count. 16 kHz, no loudness normalization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knn_vc_retrieval import mfa_linear_target  # noqa: E402
from wavlm_knn_vc_adapter import KNN_VC_REVISION, SAMPLE_RATE, WavLMKNNVCAdapter  # noqa: E402

REPO = SCRIPT_DIR.parent
KNN_VC_LOCAL = REPO / "third_party/knn-vc"


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_natural_length(values: np.ndarray, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size > count:
        output = values[:count].copy()
        action = "right_crop"
    elif values.size < count:
        output = np.pad(values, (0, count - values.size)).astype(np.float32)
        action = "right_zero_pad"
    else:
        output = values.copy()
        action = "none"
    return output, {"raw_samples": int(values.size), "target_samples": int(count), "action": action}


def extend_tokens_for_feature_tail(tokens: list[Mapping[str, Any]], frame_count: int) -> tuple[list[dict[str, Any]], float]:
    if not tokens or frame_count <= 0:
        raise ValueError("MFA tokens and feature frames must be non-empty")
    required_end = (frame_count - 0.5) * 320 / SAMPLE_RATE
    current_end = float(tokens[-1]["end_s"])
    if required_end <= current_end + 1e-7:
        return [dict(token) for token in tokens], 0.0
    extended = [dict(token) for token in tokens]
    extended[-1]["end_s"] = required_end
    extended[-1]["duration_s"] = required_end - float(extended[-1]["start_s"])
    return extended, required_end - current_end


def generate(
    manifest: Mapping[str, Any],
    tts_meta: Mapping[str, Any],
    tokens_path: Path,
    outdir: Path,
    device: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    tokens_payload = json.loads(tokens_path.read_text(encoding="utf-8"))
    if tokens_payload.get("failures"):
        raise ValueError(f"token manifest has {len(tokens_payload['failures'])} failures")
    if manifest.get("manifest_type") == "aishell1_mfa_linear_predefined_cohort":
        if len(manifest.get("records", [])) != 25 or tokens_payload.get("samples_ok") != 25:
            raise ValueError("n=25 cohort/token manifest is incomplete")
    tts_by_id = {}
    for row in tts_meta.get("results", {}).values():
        sample_id = str(row["sample_id"])
        if sample_id in tts_by_id:
            raise ValueError(f"duplicate TTS sample_id {sample_id}")
        tts_by_id[sample_id] = row
    adapter = WavLMKNNVCAdapter.load_pretrained(device=device, source=KNN_VC_LOCAL, revision=KNN_VC_REVISION)
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for row in manifest["records"]:
        sample_id = str(row["sample_id"])
        side = tokens_payload["records"].get(sample_id)
        if side is None:
            failures.append({"sample_id": sample_id, "error": "missing tokens"})
            continue
        try:
            if str(side.get("paired_key")) != str(row.get("paired_key")) or str(side.get("speaker_id")) != str(row.get("speaker_id")):
                raise ValueError("cohort/token identity mismatch")
            if sample_id not in tts_by_id:
                raise ValueError("missing TTS metadata")
            if str(tts_by_id[sample_id].get("paired_key")) != str(row.get("paired_key")):
                raise ValueError("cohort/TTS identity mismatch")
            tts_source_hash = tts_by_id[sample_id].get("source_audio_sha256")
            if tts_source_hash is not None and str(side["tts"].get("audio_sha256")) != str(tts_source_hash):
                raise ValueError("MFA token TTS source hash does not match TTS metadata")
            natural_audio, natural_sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
            tts_audio, tts_sr = sf.read(tts_by_id[sample_id]["canonical_16k_audio"], dtype="float32", always_2d=False)
            if natural_sr != SAMPLE_RATE or tts_sr != SAMPLE_RATE:
                raise ValueError(f"sample rates {natural_sr}/{tts_sr} != {SAMPLE_RATE}")
            natural_tensor = torch.from_numpy(natural_audio).unsqueeze(0)
            tts_tensor = torch.from_numpy(tts_audio).unsqueeze(0)
            natural_features = adapter.extract(natural_tensor)
            tts_features = adapter.extract(tts_tensor)
            natural_tokens, natural_tail_extension_s = extend_tokens_for_feature_tail(
                list(side["natural"]["tokens"]), natural_features.shape[0],
            )
            tts_tokens, tts_tail_extension_s = extend_tokens_for_feature_tail(
                list(side["tts"]["tokens"]), tts_features.shape[0],
            )
            conditioning, meta = mfa_linear_target(
                natural_features.shape[0], tts_features,
                natural_tokens, tts_tokens,
            )
            raw = adapter.vocode(conditioning).numpy()
            output, length_adjustment = exact_natural_length(raw, natural_audio.size)
            wav_path = outdir / f"{sample_id}.wav"
            sf.write(wav_path, output, SAMPLE_RATE, subtype="FLOAT")
            encoded_output = adapter.extract(torch.from_numpy(output).unsqueeze(0))
            cosine = 1.0 - torch.nn.functional.cosine_similarity(
                encoded_output, conditioning.to(device) if encoded_output.is_cuda else conditioning, dim=-1,
            ).mean().item() if encoded_output.shape[0] > 0 else float("nan")
            results[sample_id] = {
                "sample_id": sample_id,
                "paired_key": row.get("paired_key"),
                "speaker_id": row.get("speaker_id"),
                "split": row.get("split"),
                "audio_path": str(wav_path),
                "audio_sha256": sha256_file(wav_path),
                "natural_samples": int(natural_audio.size),
                "output_samples": int(output.size),
                "exact_natural_length": bool(output.size == natural_audio.size),
                "length_adjustment": length_adjustment,
                "natural_feature_shape": list(natural_features.shape),
                "tts_feature_shape": list(tts_features.shape),
                "conditioning_shape": list(conditioning.shape),
                "mfa_linear_meta": {
                    **meta,
                    "natural_feature_tail_extension_s": natural_tail_extension_s,
                    "tts_feature_tail_extension_s": tts_tail_extension_s,
                },
                "output_to_conditioning_cosine": float(cosine),
                "peak": float(np.abs(output).max()) if output.size else 0.0,
            }
            print(f"OK {sample_id} samples={output.size} peak={results[sample_id]['peak']:.4f}", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
            print(f"FAIL {sample_id}: {exc}", flush=True)
    summary = {
        "schema_version": 1,
        "arm": "mfa_linear",
        "cohort_manifest": str(manifest_path.resolve()) if manifest_path is not None else str(manifest.get("source_manifest", "")),
        "cohort_manifest_sha256": sha256_file(manifest_path.resolve()) if manifest_path is not None else None,
        "tokens_path": str(tokens_path.resolve()),
        "tokens_sha256": sha256_file(tokens_path.resolve()),
        "model": adapter.metadata(),
        "samples_total": len(manifest["records"]),
        "samples_ok": len(results),
        "failures": failures,
        "results": results,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps({"samples_ok": len(results), "failures": len(failures)}))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tts_meta = json.loads(args.tts_meta.read_text(encoding="utf-8"))
    summary = generate(manifest, tts_meta, args.tokens.resolve(), args.outdir.resolve(), args.device, args.manifest.resolve())
    return 0 if not summary["failures"] and summary["samples_ok"] == summary["samples_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
