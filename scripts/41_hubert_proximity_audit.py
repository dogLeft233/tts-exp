#!/usr/bin/env python3
"""41 — Audit two-stage enhanced vs raw-TTS HuBERT layer-6 proximity on the valid split.

Compares, for every valid pair, the masked HuBERT layer-6 proximity of the
natural audio, the two-stage enhanced WAV, and the raw Faster-Qwen3 TTS audio.
Metrics follow the training telemetry contract (unit-normalized feature space,
masked mean per frame, combined = cosine + smooth-L1) so results are directly
comparable with experiments 27/29.

The enhanced arm is the archived seed-29 Stage-2 export:
    runs/two_stage_hubert_aishell1_20260810/stage2_feature_targeted_20260810_v1_seed29/valid_enhanced_wav/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import HubertModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from hubert_feature_alignment import (  # noqa: E402
    align_tts_features_to_natural,
    masked_hubert_proximity,
)

MODEL_ID = "facebook/hubert-base-ls960"
STYLE_LAYER = 6
HIDDEN = 768
FRAME_STRIDE = 320
TARGET_SR = 16_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mono(path: Path) -> np.ndarray:
    """Load mono float32 audio, resampling 24 kHz TTS with the training polyphase policy."""
    audio, sample_rate = sf.read(str(path))
    if sample_rate != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sample_rate)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if not np.isfinite(audio).all():
        raise FloatingPointError(f"non-finite samples in {path}")
    return audio


def encode_layer6(hubert: HubertModel, audio: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        output = hubert(
            torch.from_numpy(audio).unsqueeze(0).to(device=next(hubert.parameters()).device),
            output_hidden_states=True,
        )
    features = output.hidden_states[STYLE_LAYER]
    if features.shape[-1] != HIDDEN:
        raise ValueError(f"unexpected HuBERT dim {features.shape[-1]}")
    return features[0].detach().cpu().float()


def _legacy_to_local(path: str, repo: Path) -> Path:
    local = Path(path)
    if not local.exists() and "/mnt/e/" in path:
        local = repo / path.split("/tts-exp/", 1)[-1]
    return local


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict:
    item_count = len(rows)
    total_matched = sum(int(row["matched_frame_count"]) for row in rows)

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            value: object = row
            for key in path:
                value = value[key]  # type: ignore[index]
            values.append(float(value))  # type: ignore[arg-type]
        result = float(np.mean(values))
        if not np.isfinite(result):
            raise FloatingPointError("aggregate is non-finite")
        return result

    return {
        "schema_version": 1,
        "item_count": item_count,
        "total_matched_frame_count": total_matched,
        "mean_coverage": mean(("coverage",)),
        "natural_to_tts": {
            "cosine": mean(("natural_to_tts", "cosine")),
            "smooth_l1": mean(("natural_to_tts", "smooth_l1")),
            "combined": mean(("natural_to_tts", "combined")),
        },
        "enhanced_to_tts": {
            "cosine": mean(("enhanced_to_tts", "cosine")),
            "smooth_l1": mean(("enhanced_to_tts", "smooth_l1")),
            "combined": mean(("enhanced_to_tts", "combined")),
        },
        "toward_tts": {
            "mean_absolute_delta": mean(("toward_tts", "absolute_delta")),
            "mean_relative_delta": mean(("toward_tts", "relative_delta")),
            "enhanced_closer_fraction": float(
                np.mean([bool(row["toward_tts"]["enhanced_closer_than_natural"]) for row in rows])
            ),
        },
        "natural_to_enhanced": {
            "cosine": mean(("natural_to_enhanced", "cosine")),
            "smooth_l1": mean(("natural_to_enhanced", "smooth_l1")),
            "combined": mean(("natural_to_enhanced", "combined")),
        },
        "finite": True,
    }


def run(manifest: Path, enhanced_dir: Path, output: Path, *, device: str) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    records = [r for r in data["records"] if r["split"] == "valid"]
    repo = manifest.parent
    for _ in range(4):
        repo = repo.parent
    if not records:
        raise ValueError("no valid-split records in eligibility manifest")

    enhanced_paths: dict[str, Path] = {}
    enhanced_manifest = enhanced_dir / "enhanced_manifest.json"
    if enhanced_manifest.exists():
        em = json.loads(enhanced_manifest.read_text(encoding="utf-8"))
        for item in em.get("items", []):
            enhanced_paths[str(item["paired_key"])] = _legacy_to_local(
                item["enhanced_path"], repo
            )
    for wav in sorted(enhanced_dir.glob("*.wav")):
        enhanced_paths.setdefault(wav.stem, wav)

    hubert = HubertModel.from_pretrained(MODEL_ID).to(device=device).eval()
    for parameter in hubert.parameters():
        parameter.requires_grad_(False)

    rows: list[dict] = []
    for record in records:
        paired_key = str(record["paired_key"])
        natural_path = _legacy_to_local(str(record["natural_path"]), repo)
        tts_path = _legacy_to_local(str(record["tts_path"]), repo)
        enhanced_path = enhanced_paths.get(paired_key)
        if enhanced_path is None:
            raise FileNotFoundError(f"no enhanced WAV for {paired_key}")
        if not natural_path.exists() or not tts_path.exists() or not enhanced_path.exists():
            raise FileNotFoundError(
                f"missing audio for {paired_key}: {natural_path}, {tts_path}, {enhanced_path}"
            )

        natural = encode_layer6(hubert, load_mono(natural_path))
        tts = encode_layer6(hubert, load_mono(tts_path))
        enhanced = encode_layer6(hubert, load_mono(enhanced_path))
        if natural.shape[0] != enhanced.shape[0]:
            raise ValueError(
                f"{paired_key}: natural frames {natural.shape[0]} != enhanced {enhanced.shape[0]}"
            )

        aligned_tts, mask, align_stats = align_tts_features_to_natural(
            natural,
            tts,
            record["matched_spans"],
            frame_stride_samples=FRAME_STRIDE,
            sample_rate=TARGET_SR,
            min_confidence=0.8,
            min_frames=1,
        )
        prox = masked_hubert_proximity(
            natural,
            enhanced,
            aligned_tts,
            mask,
            layer=STYLE_LAYER,
            encoder_model=MODEL_ID,
            min_frames=1,
        )
        natural_to_enhanced = _distance_triplet(natural, enhanced, mask)
        row = {
            "paired_key": paired_key,
            "speaker_group": record["speaker_group"],
            "matched_frame_count": prox["matched_frame_count"],
            "coverage": prox["coverage"],
            "align_stats": align_stats,
            "natural_to_tts": prox["natural_to_tts"],
            "enhanced_to_tts": prox["enhanced_to_tts"],
            "toward_tts": prox["toward_tts"],
            "natural_to_enhanced": natural_to_enhanced,
        }
        rows.append(row)
        print(
            f"{paired_key}: natural->TTS {prox['natural_to_tts']['combined']:.6f}  "
            f"enhanced->TTS {prox['enhanced_to_tts']['combined']:.6f}  "
            f"delta {prox['toward_tts']['absolute_delta']:+.6f}  "
            f"closer={prox['toward_tts']['enhanced_closer_than_natural']}",
            flush=True,
        )

    summary = aggregate(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "purpose": "valid_split_two_stage_enhanced_vs_raw_tts_hubert_l6_proximity_audit",
        "encoder_interface": {
            "model_id": MODEL_ID,
            "layer": STYLE_LAYER,
            "hidden_size": HIDDEN,
            "frame_stride_samples": FRAME_STRIDE,
            "sample_rate": TARGET_SR,
        },
        "eligibility_manifest": {
            "path": str(manifest),
            "sha256": file_sha256(manifest),
        },
        "enhanced_dir": str(enhanced_dir),
        "summary": summary,
        "items": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary


def _distance_triplet(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> dict:
    from torch.nn import functional as F

    left_norm = F.normalize(left, dim=-1)
    right_norm = F.normalize(right, dim=-1)
    cosine = 1.0 - F.cosine_similarity(left_norm, right_norm, dim=-1)
    smooth_l1 = F.smooth_l1_loss(left_norm, right_norm, reduction="none").mean(dim=-1)
    selected_cosine = float(cosine[mask].mean().cpu())
    selected_smooth_l1 = float(smooth_l1[mask].mean().cpu())
    return {
        "cosine": selected_cosine,
        "smooth_l1": selected_smooth_l1,
        "combined": selected_cosine + selected_smooth_l1,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--enhanced-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    summary = run(args.manifest, args.enhanced_dir, args.output, device=args.device)
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
