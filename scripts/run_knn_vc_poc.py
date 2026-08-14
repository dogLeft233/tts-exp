#!/usr/bin/env python3
"""Run the frozen valid-only natural-clock / TTS-feature kNN-VC PoC."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knn_vc_poc_schema import (  # noqa: E402
    EXPECTED_CONDITIONS,
    canonical_sha256,
    file_sha256,
    validate_summary,
    write_json,
)
from knn_vc_retrieval import (  # noqa: E402
    RetrievalConfig,
    frame_owners,
    mfa_linear_target,
    retrieve,
)
from prepare_two_stage_hubert_manifest import load_two_stage_manifest  # noqa: E402
from run_two_stage_diagnostics import localize_manifest  # noqa: E402
from train_tts_aligned_feature_puller import _read_stage1_wave  # noqa: E402
from wavlm_knn_vc_adapter import (  # noqa: E402
    FRAME_STRIDE_SAMPLES,
    KNN_VC_REVISION,
    SAMPLE_RATE,
    WavLMKNNVCAdapter,
)

REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_STAGE_MANIFEST = REPO_ROOT / "runs/two_stage_hubert_aishell1_20260810/data_boundary/two_stage_hubert_pair_eligibility.json"
DEFAULT_ALIGNMENT_MANIFEST = REPO_ROOT / "runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json"
VALID_SPEAKER = "S0765"
HELDOUT_SPEAKER = "S0770"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    ).stdout.strip() or "unavailable"


def _legacy_to_local(value: str | Path) -> Path:
    raw = str(value)
    path = Path(raw)
    prefix = "/mnt/e/Documents/tts-audio/tts-exp/"
    if not path.is_file() and raw.startswith(prefix):
        path = REPO_ROOT / raw[len(prefix):]
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _strict_alignment_pairs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("quality_policy", {}).get("fallback_allowed") is not False:
        raise ValueError("alignment manifest is not the strict no-fallback schema")
    if payload.get("split_speakers", {}).get("valid") != [VALID_SPEAKER] or payload.get("split_speakers", {}).get("test") != [HELDOUT_SPEAKER]:
        raise ValueError("alignment split-speaker contract changed")
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in payload.get("records", []):
        if raw.get("split") != "valid" or raw.get("speaker_id") != VALID_SPEAKER:
            continue
        condition = str(raw.get("condition", ""))
        if condition not in {"natural", "tts"}:
            continue
        if raw.get("strict_gate_pass") is not True or raw.get("alignment_source") != "mfa":
            raise ValueError("valid record failed the strict MFA gate")
        grouped.setdefault(str(raw["paired_key"]), {})[condition] = dict(raw)
    pairs: list[dict[str, Any]] = []
    for key in sorted(grouped):
        arms = grouped[key]
        if set(arms) != {"natural", "tts"}:
            raise ValueError(f"incomplete pair: {key}")
        natural, tts = arms["natural"], arms["tts"]
        if natural.get("transcript") != tts.get("transcript"):
            raise ValueError(f"transcript mismatch: {key}")
        pairs.append({"paired_key": key, "natural": natural, "tts": tts})
    if len(pairs) != 50:
        raise ValueError(f"expected exactly 50 valid pairs, found {len(pairs)}")
    return pairs


def select_records(
    stage_manifest: Path,
    alignment_manifest: Path,
    output_dir: Path,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    if count not in {1, 10, 15}:
        raise ValueError("PoC count must be exactly 1, 10, or 15")
    localized = localize_manifest(stage_manifest, repo_root=REPO_ROOT)
    localized_path = output_dir / "localized_stage_manifest.json"
    write_json(localized_path, localized)
    stage_records = load_two_stage_manifest(localized_path)
    valid_stage = {
        str(record["paired_key"]): record for record in stage_records
        if record["split"] == "valid" and record["speaker_group"] == VALID_SPEAKER
    }
    if len(valid_stage) != 50:
        raise ValueError(f"expected 50 valid stage records, found {len(valid_stage)}")
    pairs = _strict_alignment_pairs(alignment_manifest)
    for pair in pairs:
        key = pair["paired_key"]
        stage = valid_stage.get(key)
        if stage is None:
            raise ValueError(f"strict alignment pair missing from stage manifest: {key}")
        natural_path = _legacy_to_local(pair["natural"]["audio_path"])
        tts_path = _legacy_to_local(pair["tts"]["audio_path"])
        if file_sha256(natural_path) != stage["natural_sha256"] or file_sha256(tts_path) != stage["tts_sha256"]:
            raise ValueError(f"source hash mismatch between manifests: {key}")
        pair["stage"] = stage
        pair["natural_path"] = natural_path
        pair["tts_path"] = tts_path
    selected = pairs[:count]
    keys = [pair["paired_key"] for pair in selected]
    selection = {
        "split": "valid",
        "speaker_group": VALID_SPEAKER,
        "available_count": len(pairs),
        "selected_count": len(selected),
        "ordered_paired_keys": keys,
        "ordered_paired_keys_sha256": canonical_sha256(keys),
        "selection_rule": "sorted_paired_key_prefix",
    }
    return selected, selection, localized_path


def _audio_qc(values: np.ndarray, natural_count: int) -> dict[str, Any]:
    return {
        "sample_count": int(values.size),
        "natural_sample_count": int(natural_count),
        "exact_natural_sample_count": bool(values.size == natural_count),
        "finite": bool(np.isfinite(values).all()),
        "peak": float(np.abs(values).max()) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "clipped_sample_count": int((np.abs(values) >= 1.0).sum()),
    }


def exact_natural_length(values: np.ndarray, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    raw_count = int(values.size)
    if raw_count > count:
        output = values[:count].copy()
        action = "right_crop"
    elif raw_count < count:
        output = np.pad(values, (0, count - raw_count)).astype(np.float32)
        action = "right_zero_pad"
    else:
        output = values.copy()
        action = "none"
    return output, {
        "raw_decoder_sample_count": raw_count,
        "target_natural_sample_count": count,
        "action": action,
        "adjustment_sample_count": abs(raw_count - count),
    }


def _distance(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    frames = min(left.shape[0], right.shape[0])
    if frames <= 0 or left.shape[1] != right.shape[1]:
        raise ValueError("feature distance requires compatible non-empty sequences")
    left = left[:frames].float()
    right = right[:frames].float().to(left.device)
    cosine = 1.0 - torch.nn.functional.cosine_similarity(left, right, dim=-1)
    return {
        "cosine": float(cosine.mean().cpu()),
        "mse": float(torch.mean((left - right) ** 2).cpu()),
        "compared_frames": frames,
    }


def _condition_feature(
    name: str,
    natural_features: torch.Tensor,
    tts_features: torch.Tensor,
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if name == "natural_direct_resynthesis":
        return natural_features, {
            "policy": "direct_natural_features", "query_frames": natural_features.shape[0],
            "candidate_frames": natural_features.shape[0], "coverage": 1.0, "fallback_frames": 0,
        }
    if name == "paired_tts_mfa_linear":
        return mfa_linear_target(
            natural_features.shape[0], tts_features, natural_tokens, tts_tokens,
        )
    policy = {
        "paired_tts_unconstrained": "unconstrained",
        "paired_tts_same_phone": "same_phone",
        "paired_tts_same_phone_position": "same_phone_position",
        "paired_tts_wrong_phone": "wrong_phone",
    }[name]
    natural_owners = frame_owners(natural_features.shape[0], natural_tokens)
    tts_owners = frame_owners(tts_features.shape[0], tts_tokens)
    result = retrieve(
        natural_features, tts_features, natural_owners, tts_owners,
        RetrievalConfig(
            policy=policy,
            allow_unconstrained_tts_fallback=policy in {"same_phone", "same_phone_position"},
        ),
    )
    return result.features, result.metadata


def run(
    stage_manifest: Path,
    alignment_manifest: Path,
    output_dir: Path,
    *,
    count: int,
    device: str,
    adapter: WavLMKNNVCAdapter | None = None,
    model_source: str | Path = "bshall/knn-vc",
    conditions: Sequence[str] = EXPECTED_CONDITIONS,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_names = tuple(conditions)
    if not condition_names or len(condition_names) != len(set(condition_names)):
        raise ValueError("conditions must be non-empty and unique")
    if any(name not in EXPECTED_CONDITIONS for name in condition_names):
        raise ValueError("conditions contain an unsupported PoC condition")
    selected, selection, localized_path = select_records(
        stage_manifest.resolve(), alignment_manifest.resolve(), output_dir, count,
    )
    adapter = adapter or WavLMKNNVCAdapter.load_pretrained(
        device=device, source=model_source, revision=KNN_VC_REVISION,
    )
    items: list[dict[str, Any]] = []
    for item_index, pair in enumerate(selected, 1):
        key = pair["paired_key"]
        natural = _read_stage1_wave(pair["natural_path"])[0, 0]
        tts = _read_stage1_wave(pair["tts_path"], allow_24k_resample=True)[0, 0]
        natural_features = adapter.extract(natural.unsqueeze(0))
        tts_features = adapter.extract(tts.unsqueeze(0))
        aligned_tts_features, _ = mfa_linear_target(
            natural_features.shape[0], tts_features,
            pair["natural"]["tokens"], pair["tts"]["tokens"],
        )
        rows: list[dict[str, Any]] = []
        for condition in condition_names:
            conditioning, retrieval_meta = _condition_feature(
                condition, natural_features, tts_features,
                pair["natural"]["tokens"], pair["tts"]["tokens"],
            )
            raw = adapter.vocode(conditioning).numpy()
            output, length_adjustment = exact_natural_length(raw, natural.numel())
            wav_path = output_dir / "audio" / f"{key}__{condition}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav_path, output, SAMPLE_RATE, subtype="FLOAT")
            encoded_output = adapter.extract(torch.from_numpy(output).unsqueeze(0))
            feature_path = output_dir / "features" / f"{key}__{condition}.pt"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"conditioning": conditioning.cpu(), "output": encoded_output.cpu()}, feature_path)
            row = {
                "condition": condition,
                "paired_oracle": True,
                "candidate_scope": "paired_tts" if condition != "natural_direct_resynthesis" else "paired_natural_self",
                "retrieval": retrieval_meta,
                "conditioning_shape": list(conditioning.shape),
                "conditioning_sha256": file_sha256(feature_path),
                "feature_path": str(feature_path),
                "length_adjustment": length_adjustment,
                "audio": _audio_qc(output, natural.numel()),
                "output_path": str(wav_path),
                "output_sha256": file_sha256(wav_path),
                "wavlm_synthesis_1024": {
                    "output_to_conditioning": _distance(encoded_output, conditioning),
                    "output_to_natural_query": _distance(encoded_output, natural_features),
                    "output_to_aligned_tts": _distance(encoded_output, aligned_tts_features),
                    "natural_query_to_aligned_tts": _distance(natural_features, aligned_tts_features),
                },
            }
            sidecar_path = wav_path.with_suffix(".json")
            write_json(sidecar_path, row)
            rows.append(row)
            print(f"OK {item_index}/{count} {condition} samples={output.size} peak={row['audio']['peak']:.4f}", flush=True)
        items.append({
            "paired_key": key,
            "sample_id": int(pair["natural"]["sample_id"]),
            "transcript": str(pair["natural"]["transcript"]),
            "natural_source": str(pair["natural_path"]),
            "natural_source_sha256": file_sha256(pair["natural_path"]),
            "tts_source": str(pair["tts_path"]),
            "tts_source_sha256": file_sha256(pair["tts_path"]),
            "natural_tokens_sha256": canonical_sha256(pair["natural"]["tokens"]),
            "tts_tokens_sha256": canonical_sha256(pair["tts"]["tokens"]),
            "natural_feature_shape": list(natural_features.shape),
            "tts_feature_shape": list(tts_features.shape),
            "conditions": rows,
        })
    summary = {
        "schema_version": 1,
        "experiment_type": "wavlm_knn_vc_valid_poc",
        "status": "completed",
        "manifest": {
            "stage_path": str(stage_manifest.resolve()),
            "stage_sha256": file_sha256(stage_manifest),
            "localized_stage_path": str(localized_path),
            "localized_stage_sha256": file_sha256(localized_path),
            "alignment_path": str(alignment_manifest.resolve()),
            "alignment_sha256": file_sha256(alignment_manifest),
        },
        "selection": selection,
        "model": adapter.metadata(),
        "policies": {
            "topk": 4, "loudness_normalization": False,
            "final_length": "right_crop_or_right_zero_pad_to_natural_samples",
            "frame_center": "(index+0.5)*320/16000",
            "unmatched": "explicit_tts_only_fallback_with_coverage_accounting_no_natural_feature_injection",
        },
        "runtime": {
            "argv": sys.argv,
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": device, "git_commit": _git_commit(),
        },
        "conditions": list(condition_names),
        "items": items,
        "heldout_excluded": True,
    }
    if condition_names == EXPECTED_CONDITIONS:
        validate_summary(summary)
    elif condition_names != ("paired_tts_mfa_linear",):
        raise ValueError("reduced PoC runs support only paired_tts_mfa_linear")
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, default=DEFAULT_STAGE_MANIFEST)
    parser.add_argument("--alignment-manifest", type=Path, default=DEFAULT_ALIGNMENT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(1, 10, 15), default=1)
    parser.add_argument("--condition", action="append", choices=EXPECTED_CONDITIONS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-source", default="bshall/knn-vc")
    args = parser.parse_args(argv)
    result = run(
        args.stage_manifest, args.alignment_manifest, args.output_dir,
        count=args.count, device=args.device, model_source=args.model_source,
        conditions=tuple(args.condition) if args.condition else EXPECTED_CONDITIONS,
    )
    print(json.dumps({"output": str(args.output_dir.resolve()), "count": len(result["items"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
