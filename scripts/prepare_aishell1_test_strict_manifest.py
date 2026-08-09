#!/usr/bin/env python3
"""Convert strict AISHELL-1 pair summaries into condition-level MFA records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


_PHONE_INTERVAL_RE = re.compile(
    r"xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*\"([^\"]*)\"",
    re.DOTALL,
)
_TONE_MARKS = re.compile(r"[˥˦˧˨˩]")
_SILENCE = frozenset({"", "sp", "sil", "SIL", "SPN", "spn", "<sil>", "<unk>"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_phone_tokens(textgrid_path: Path) -> list[dict[str, Any]]:
    content = textgrid_path.read_text(encoding="utf-8")
    try:
        start = content.index('name = "phones"')
    except ValueError as exc:
        raise ValueError(f"phones tier missing: {textgrid_path}") from exc
    try:
        end = content.index("item [", start + 1)
    except ValueError:
        end = len(content)
    tokens = []
    for match in _PHONE_INTERVAL_RE.finditer(content[start:end]):
        start_s, end_s, label = match.groups()
        if label in _SILENCE:
            continue
        tokens.append(
            {
                "token": _TONE_MARKS.sub("", label),
                "raw_token": label,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "confidence": 1.0,
            }
        )
    if not tokens:
        raise ValueError(f"phones tier contains no speech tokens: {textgrid_path}")
    return tokens


def split_speakers(speakers: list[str], seed: int) -> dict[str, str]:
    values = sorted(set(speakers))
    if len(values) < 3:
        raise ValueError("at least three speakers are required")
    rng = random.Random(seed)
    rng.shuffle(values)
    n_test = max(1, round(len(values) * 0.125))
    n_valid = max(1, round(len(values) * 0.125))
    result = {}
    for speaker in values[: len(values) - n_valid - n_test]:
        result[speaker] = "train"
    for speaker in values[len(values) - n_valid - n_test : len(values) - n_test]:
        result[speaker] = "valid"
    for speaker in values[len(values) - n_test :]:
        result[speaker] = "test"
    return result


def build_manifest(source_path: Path, output_path: Path, seed: int = 42) -> dict[str, Any]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = payload.get("records", [])
    if not isinstance(rows, list):
        raise ValueError("strict manifest records must be a list")
    strict_rows = [row for row in rows if row.get("strict_gate_pass") is True and row.get("target_status") == "strict_alignment_ready"]
    if len(strict_rows) != int(payload.get("counts", {}).get("strict_accepted", len(strict_rows))):
        raise ValueError("strict row count does not match source manifest count")
    split_map = split_speakers([str(row["speaker_id"]) for row in strict_rows], seed)
    source_hash = sha256_file(source_path)
    records: list[dict[str, Any]] = []
    for row in strict_rows:
        expansion_id = int(row["expansion_id"])
        speaker_id = str(row["speaker_id"])
        paired_key = f"aishell1_test_400/{row['source_utterance_id']}"
        for condition, audio_key, grid_key in (
            ("natural", "natural_audio_path", "natural_textgrid_path"),
            ("tts", "tts_audio_path", "tts_textgrid_path"),
        ):
            record = {
                "schema_version": 1,
                "dataset": "AISHELL-1",
                "utterance_id": f"{row['source_utterance_id']}:{condition}:raw",
                "source_utterance_id": str(row["source_utterance_id"]),
                "speaker_id": speaker_id,
                "paired_key": paired_key,
                "sample_id": expansion_id,
                "condition": condition,
                "variant": "raw",
                "tts_provider": "faster_qwen3" if condition == "tts" else None,
                "transcript": str(row["transcript"]),
                "audio_path": str(row[audio_key]),
                "filepath": str(row[audio_key]),
                "split": split_map[speaker_id],
                "license": str(row["license"]),
                "alignment_source": "mfa",
                "alignment_method": str(row["alignment_source"]),
                "alignment_confidence": 1.0,
                "alignment_manifest": str(source_path.resolve()),
                "alignment_manifest_sha256": source_hash,
                "textgrid_path": str(row[grid_key]),
                "source_sha256": str(row["natural_sha256"] if condition == "natural" else row["tts_sha256"]),
                "strict_gate_pass": True,
                "word_coverage": float(row["word_coverage"]),
                "phone_coverage": float(row["phone_coverage"]),
                "phone_label_mismatch_count": int(row["phone_label_mismatch_count"]),
                "tokens": parse_phone_tokens(Path(row[grid_key])),
            }
            records.append(record)
    result = {
        "schema_version": 1,
        "dataset": "AISHELL-1",
        "dataset_version": "aishell1_test_400_condition_mfa_v1",
        "purpose": "strict_condition_level_manifest_for_rhythm_style_dataset",
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": source_hash,
        "seed": seed,
        "quality_policy": payload["quality_policy"],
        "split_speakers": {
            split: sorted(speaker for speaker, value in split_map.items() if value == split)
            for split in ("train", "valid", "test")
        },
        "counts": {
            "strict_pairs": len(strict_rows),
            "condition_records": len(records),
            "natural_records": sum(record["condition"] == "natural" for record in records),
            "tts_records": sum(record["condition"] == "tts" for record in records),
            "quarantine_pairs_excluded": int(payload["counts"]["quarantine"]),
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_manifest(args.source, args.output, seed=args.seed)
    print(json.dumps(result["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
