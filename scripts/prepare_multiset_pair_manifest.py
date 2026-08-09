#!/usr/bin/env python3
"""Build an auditable manifest for the existing 250 natural/TTS pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf


_CHINESE = re.compile(r"[㐀-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def classify_language(text: str) -> str:
    chinese = len(_CHINESE.findall(text))
    latin = len(_LATIN.findall(text))
    if chinese and latin:
        return "mixed"
    if chinese:
        return "zh"
    if latin:
        return "en"
    return "other"


def audio_info(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    if info.channels != 1:
        raise ValueError(f"audio is not mono: {path}")
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"audio is empty or invalid: {path}")
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_s": float(info.frames / info.samplerate),
        "subtype": info.subtype,
    }


def load_transcripts(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload)
    if not isinstance(rows, dict):
        raise ValueError("transcript manifest must contain a results mapping")
    result: dict[int, str] = {}
    for key, row in rows.items():
        if not isinstance(row, dict) or not row.get("text"):
            raise ValueError(f"missing transcript for sample {key}")
        result[int(row.get("sample_id", key))] = str(row["text"]).strip()
    return result


def load_alignment_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("records", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    if not isinstance(rows, list):
        return set()
    natural = {
        int(row["sample_id"])
        for row in rows
        if isinstance(row, dict)
        and row.get("condition") == "natural"
        and row.get("variant") == "raw"
        and row.get("tokens")
    }
    tts = {
        int(row["sample_id"])
        for row in rows
        if isinstance(row, dict)
        and row.get("condition") == "faster_qwen3"
        and row.get("variant") == "raw"
        and row.get("tokens")
    }
    return natural & tts


def stable_group_split(groups: list[str], seed: int = 42) -> dict[str, str]:
    import random

    values = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(values)
    n_test = max(1, round(len(values) * 0.15))
    n_valid = max(1, round(len(values) * 0.15))
    result: dict[str, str] = {}
    for group in values[: len(values) - n_valid - n_test]:
        result[group] = "train"
    for group in values[len(values) - n_valid - n_test : len(values) - n_test]:
        result[group] = "valid"
    for group in values[len(values) - n_test :]:
        result[group] = "test"
    return result


def build_manifest(repo_root: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    source_manifest = repo_root / "data/dataset_samples/video_manifest_250.json"
    transcript_manifest = repo_root / "data/tts_audio_250/transcript.json"
    alignment_manifest = repo_root / "data/wav2sem_analysis/manifest/alignment.json"
    records = json.loads(source_manifest.read_text(encoding="utf-8"))["records"]
    transcripts = load_transcripts(transcript_manifest)
    aligned_ids = load_alignment_ids(alignment_manifest)
    if len(records) != 250 or set(range(1, 251)) != set(transcripts):
        raise ValueError("expected exactly 250 samples with complete transcripts")
    groups = [str(row["speaker_key"]) for row in records]
    split_map = stable_group_split(groups, seed=seed)
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_id, row in enumerate(records, 1):
        natural_path = repo_root / "data/data/audio" / f"{sample_id}.wav"
        tts_path = repo_root / "data/tts_audio_250" / f"{sample_id}.wav"
        text = transcripts[sample_id]
        item: dict[str, Any] = {
            "paired_key": f"multiset250/{row['dataset']}/{row['stem']}",
            "sample_id": sample_id,
            "dataset": row["dataset"],
            "speaker_key": row["speaker_key"],
            "split": split_map[row["speaker_key"]],
            "language": classify_language(text),
            "text": text,
            "natural_audio_path": str(natural_path),
            "tts_audio_path": str(tts_path),
            "source_video_path": str(repo_root / row["video_local_path"]),
            "audio_source": row.get("audio_source"),
            "alignment_status": "legacy_partial_candidate" if sample_id in aligned_ids else "missing_token_alignment",
            "alignment_manifest": str(alignment_manifest),
            "target_status": "not_ready_for_rhythm_style_training",
            "recommended_use": "separate_language_or_model_evaluation" if classify_language(text) != "zh" else "candidate_after_strict_alignment",
        }
        try:
            natural_info = audio_info(natural_path)
            tts_info = audio_info(tts_path)
            item["natural"] = {**natural_info, "sha256": sha256_file(natural_path)}
            item["tts"] = {**tts_info, "sha256": sha256_file(tts_path), "provider": "faster_qwen3"}
            if natural_info["sample_rate"] != 16_000:
                item["quality_warning"] = "natural_sample_rate_not_16k"
            if tts_info["duration_s"] < 1.0:
                item["quality_warning"] = "tts_duration_below_1s"
            output_rows.append(item)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append({"sample_id": sample_id, "error": str(exc), "record": item})
    speakers = defaultdict(set)
    for row in output_rows:
        speakers[row["split"]].add(row["speaker_key"])
    manifest = {
        "schema_version": 1,
        "dataset_version": "multiset250_pair_manifest_v1",
        "source_manifest": str(source_manifest),
        "transcript_manifest": str(transcript_manifest),
        "alignment_manifest": str(alignment_manifest),
        "natural_audio_dir": str(repo_root / "data/data/audio"),
        "tts_audio_dir": str(repo_root / "data/tts_audio_250"),
        "tts_provider": "faster_qwen3",
        "seed": seed,
        "counts": {
            "records": len(output_rows),
            "failures": len(failures),
            "alignment_ready_candidates": sum(row["alignment_status"] == "legacy_partial_candidate" for row in output_rows),
            "missing_token_alignment": sum(row["alignment_status"] == "missing_token_alignment" for row in output_rows),
            "by_language": dict(Counter(row["language"] for row in output_rows)),
            "by_dataset": dict(Counter(row["dataset"] for row in output_rows)),
            "by_split": dict(Counter(row["split"] for row in output_rows)),
        },
        "split_speakers": {key: sorted(value) for key, value in speakers.items()},
        "quality_policy": {
            "requires_token_alignment": True,
            "requires_phone_level_mfa": True,
            "min_coverage": 0.90,
            "min_token_confidence": 0.80,
            "fallback_allowed": False,
        },
        "weak_target_warning": "No phone-local weak target has been generated for these rows; missing alignment is not silently replaced with uniform timing.",
        "records": output_rows,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_heldout_manifest(
    manifest: dict[str, Any],
    *,
    training_pair_ids: set[str],
    heldout_pair_ids: set[str],
    required_conditions: tuple[str, ...] = ("ordinary", "identity", "adapted", "random", "raw_tts"),
) -> None:
    """Fail closed before a future remote Ditto/SyncNet evaluation."""
    if training_pair_ids & heldout_pair_ids:
        raise ValueError("held-out pairs overlap adapter-training pairs")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("held-out manifest records are missing")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("held-out records must be objects")
        key = str(record.get("paired_key", "")).strip()
        if not key or key in seen:
            raise ValueError(f"missing or duplicate held-out pair key: {key!r}")
        seen.add(key)
        if key not in heldout_pair_ids:
            raise ValueError(f"record is not in held-out split: {key}")
        if key in training_pair_ids:
            raise ValueError(f"training pair appears in held-out manifest: {key}")
        if str(record.get("split", "")) != "heldout":
            raise ValueError(f"held-out record has wrong split: {key}")
        if not str(record.get("text", "")).strip():
            raise ValueError(f"held-out record has no transcript: {key}")
        for arm in ("natural", "tts"):
            path = record.get(f"{arm}_audio_path")
            info = record.get(arm)
            if not isinstance(path, str) or not isinstance(info, dict) or len(str(info.get("sha256", ""))) != 64:
                raise ValueError(f"held-out {arm} provenance is incomplete: {key}")
        if not record.get("source_video_path"):
            raise ValueError(f"held-out source image/video path is missing: {key}")
    declared = manifest.get("evaluation_protocol", {})
    conditions = tuple(declared.get("conditions", ())) if isinstance(declared, dict) else ()
    if conditions != required_conditions:
        raise ValueError(f"evaluation condition matrix mismatch: {conditions!r}")


def build_heldout_preflight(
    records: list[dict[str, Any]],
    *,
    training_pair_ids: set[str],
    adapter_hash: str,
    ditto_dependency: str,
    syncnet_dependency: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a local-only reproducibility envelope; does not run inference."""
    heldout = []
    for record in records:
        key = str(record.get("paired_key", ""))
        if key in training_pair_ids:
            continue
        heldout.append({**record, "split": "heldout"})
    if not heldout:
        raise ValueError("no non-overlapping held-out records")
    return {
        "schema_version": 1,
        "split_classification": "heldout_not_executed",
        "adapter_training_pair_ids": sorted(training_pair_ids),
        "heldout_pair_ids": sorted(str(row["paired_key"]) for row in heldout),
        "evaluation_protocol": {
            "conditions": ["ordinary", "identity", "adapted", "random", "raw_tts"],
            "seed": int(seed),
            "ditto_dependency": ditto_dependency,
            "syncnet_dependency": syncnet_dependency,
            "adapter_sha256": adapter_hash,
        },
        "records": heldout,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_manifest(args.repo_root.resolve(), args.output, seed=args.seed)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    return 0 if not manifest["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
