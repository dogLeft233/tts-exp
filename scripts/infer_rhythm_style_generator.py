#!/usr/bin/env python3
"""Run a trained rhythm/style generator on a paired local item."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_rhythm_style_dataset import (
    TARGET_SR,
    align_content,
    build_rhythm,
    embedding_for_sample,
    extract_style_features,
    load_audio,
    pair_records,
    records_from_manifest,
    resolve_path,
    spans,
)
from rhythm_style_generator import GeneratorConfig, RhythmStyleGenerator


def load_checkpoint(path: str | Path, device: torch.device) -> RhythmStyleGenerator:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    config = GeneratorConfig(**payload.get("model_config", {}))
    model = RhythmStyleGenerator(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _infer_with_model(
    model: RhythmStyleGenerator,
    natural_path: str | Path,
    style_path: str | Path,
    *,
    natural_content: np.ndarray,
    rhythm: np.ndarray,
    device_obj: torch.device,
) -> np.ndarray:
    natural = load_audio(Path(natural_path))
    style = extract_style_features(load_audio(Path(style_path)))
    content = align_content(natural_content, max(1, int(np.ceil(len(natural) / 320))))
    rhythm = rhythm[: content.shape[0]]
    if rhythm.shape[0] < content.shape[0]:
        rhythm = np.pad(rhythm, ((0, content.shape[0] - rhythm.shape[0]), (0, 0)))
    with torch.no_grad():
        output = model(
            torch.from_numpy(natural).to(device_obj)[None, None, :],
            torch.from_numpy(content).to(device_obj)[None],
            torch.from_numpy(rhythm.astype(np.float32)).to(device_obj)[None],
            torch.from_numpy(style).to(device_obj)[None],
        )["waveform"][0, 0].cpu().numpy()
    output = np.asarray(output, dtype=np.float32)
    if len(output) != len(natural) or not np.isfinite(output).all():
        raise ValueError("inference output failed exact-length or finite invariant")
    return output


def infer(
    checkpoint: str | Path,
    natural_path: str | Path,
    style_path: str | Path,
    *,
    natural_content: np.ndarray,
    rhythm: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    device_obj = torch.device(device)
    model = load_checkpoint(checkpoint, device_obj)
    return _infer_with_model(
        model, natural_path, style_path,
        natural_content=natural_content, rhythm=rhythm, device_obj=device_obj,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_record_ids(
    dataset_dir: Path,
    split: str | None,
    sample_ids: Iterable[int] | None,
) -> list[dict[str, Any]]:
    path = dataset_dir / f"{split}.pt" if split else None
    if path is None:
        raise ValueError("batch inference requires --split")
    try:
        records = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        records = torch.load(path, map_location="cpu")
    requested = set(sample_ids or [])
    selected = [dict(record) for record in records if not requested or int(record["sample_id"]) in requested]
    if not selected:
        raise ValueError(f"no samples selected from {path}")
    return selected


def batch_infer(
    checkpoint: str | Path,
    dataset_dir: str | Path,
    alignment_manifest: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    condition: str = "mvp",
    alpha: float = 1.0,
    sample_ids: Iterable[int] | None = None,
    repo_root: str | Path | None = None,
    embedding_dir: str | Path = "results/aishell100_phoneme/embeddings",
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd()).resolve()
    dataset_root = Path(dataset_dir)
    if not dataset_root.is_absolute():
        dataset_root = root / dataset_root
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    condition_dir = output_root / "conditions" / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    records = _batch_record_ids(dataset_root, split, sample_ids)
    alignment_records = records_from_manifest(alignment_manifest)
    groups = pair_records(alignment_records)
    groups_by_sample_id = {
        int(group["natural"]["sample_id"]): group
        for group in groups.values()
        if "natural" in group and group["natural"].get("sample_id") is not None
    }
    device_obj = torch.device(device)
    model = load_checkpoint(checkpoint, device_obj) if condition.startswith("mvp") else None
    checkpoint_path = resolve_path(checkpoint, root)
    embedding_root = Path(embedding_dir)
    if not embedding_root.is_absolute():
        embedding_root = root / embedding_root
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        sample_id = int(record["sample_id"])
        output_path = condition_dir / f"{sample_id}.wav"
        try:
            natural_path = resolve_path(record["natural_audio_path"], root)
            tts_path = resolve_path(record["tts_audio_path"], root)
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"output exists: {output_path}")
            group = groups_by_sample_id.get(sample_id)
            if group is None or "natural" not in group:
                raise ValueError(f"sample {sample_id} has no alignment group")
            natural_audio = load_audio(natural_path)
            rhythm = build_rhythm(
                spans(group["natural"]),
                spans(group.get("faster_qwen3") or group.get("tts", {})),
                len(natural_audio), alpha=alpha,
            )
            if condition == "natural_identity":
                output = natural_audio
            elif condition == "raw_tts":
                output = load_audio(tts_path)
            elif condition == "weak_target":
                output = np.asarray(record["weak_target"].float().numpy(), dtype=np.float32)
            elif condition.startswith("mvp"):
                assert model is not None
                content = embedding_for_sample(embedding_root, sample_id, "natural")
                output = _infer_with_model(
                    model, natural_path, tts_path, natural_content=content,
                    rhythm=rhythm, device_obj=device_obj,
                )
            else:
                raise ValueError(f"unknown batch condition: {condition}")
            if condition.startswith("mvp") and len(output) != len(natural_audio):
                raise ValueError("model output is not exact natural length")
            if output.ndim != 1 or not np.isfinite(output).all():
                raise ValueError("output must be finite mono audio")
            sf.write(output_path, output, TARGET_SR, subtype="PCM_16")
            metadata = {
                "sample_id": sample_id,
                "speaker_id": record.get("speaker_id"),
                "split": split,
                "condition": condition,
                "alpha": alpha,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "natural_path": str(natural_path),
                "tts_path": str(tts_path),
                "natural_sha256": _sha256(natural_path),
                "tts_sha256": _sha256(tts_path),
                "output_path": str(output_path),
                "sample_rate": TARGET_SR,
                "sample_count": int(len(output)),
                "natural_sample_count": int(len(natural_audio)),
                "exact_natural_length": bool(len(output) == len(natural_audio)),
                "finite": True,
                "peak": float(np.max(np.abs(output))),
                "clipped_sample_count": int(np.count_nonzero(np.abs(output) >= 1.0)),
                "coverage": record.get("coverage"),
                "target_type": "tts_phone_local_warp_weak_supervision",
                "weak_target_warning": "phone-local warp is supervision only, not a validated disentangled target",
            }
            output_path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            outputs.append(metadata)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
    manifest = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_dir": str(dataset_root.resolve()),
        "alignment_manifest": str(resolve_path(alignment_manifest, root)),
        "split": split,
        "condition": condition,
        "alpha": alpha,
        "sample_rate": TARGET_SR,
        "outputs": outputs,
        "failures": failures,
        "complete": not failures and bool(outputs),
        "weak_target_warning": "phone-local warp is supervision only, not a validated disentangled target",
    }
    (output_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--condition", default="mvp")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-ids", type=int, nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--natural", type=Path)
    parser.add_argument("--style-reference", type=Path)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--sample-id", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-dir", type=Path, default=Path("results/aishell100_phoneme/embeddings"))
    args = parser.parse_args(argv)
    if args.batch:
        if args.dataset is None or args.output_dir is None:
            parser.error("--batch requires --dataset and --output-dir")
        manifest = batch_infer(
            args.checkpoint, args.dataset, args.alignment_manifest, args.output_dir,
            split=args.split, condition=args.condition, alpha=args.alpha,
            sample_ids=args.sample_ids, repo_root=args.repo_root,
            embedding_dir=args.embedding_dir, device=args.device, overwrite=args.overwrite,
        )
        print(json.dumps({"outputs": len(manifest["outputs"]), "failures": len(manifest["failures"])}, ensure_ascii=False))
        return 0
    if args.natural is None or args.style_reference is None or args.sample_id is None or args.output is None:
        parser.error("single inference requires --natural, --style-reference, --sample-id, and --output")
    records = records_from_manifest(args.alignment_manifest)
    groups = pair_records(records)
    group = next(
        (
            value
            for value in groups.values()
            if value.get("natural", {}).get("sample_id") == args.sample_id
        ),
        None,
    )
    if group is None or "natural" not in group or "faster_qwen3" not in group:
        raise ValueError(f"sample {args.sample_id} is not a complete natural/faster_qwen3 pair")
    natural_path = resolve_path(args.natural, args.repo_root)
    style_path = resolve_path(args.style_reference, args.repo_root)
    natural_audio = load_audio(natural_path)
    content = embedding_for_sample(
        args.embedding_dir if args.embedding_dir.is_absolute() else args.repo_root / args.embedding_dir,
        args.sample_id,
        "natural",
    )
    rhythm = build_rhythm(spans(group["natural"]), spans(group["faster_qwen3"]), len(natural_audio), alpha=args.alpha)
    output = infer(
        args.checkpoint, natural_path, style_path,
        natural_content=content, rhythm=rhythm, device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, TARGET_SR, subtype="PCM_16")
    metadata = {
        "sample_id": args.sample_id,
        "checkpoint": str(args.checkpoint),
        "natural_path": str(natural_path),
        "style_reference_path": str(style_path),
        "output_path": str(args.output),
        "sample_rate": TARGET_SR,
        "sample_count": len(output),
        "alpha": args.alpha,
        "finite": bool(np.isfinite(output).all()),
        "peak": float(np.max(np.abs(output))),
        "clipped_sample_count": int(np.count_nonzero(np.abs(output) >= 1.0)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
