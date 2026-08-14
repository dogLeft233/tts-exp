#!/usr/bin/env python3
"""Freeze a deterministic multi-speaker AISHELL-1 MFA-linear cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_MANIFEST = REPO / "runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json"
SPEAKERS = ("S0765", "S0901", "S0906", "S0912", "S0913")
HELDOUT_SPEAKERS = ("S0770",)
PER_SPEAKER = 5
COUNT = len(SPEAKERS) * PER_SPEAKER
LEGACY_PREFIX = "/mnt/e/Documents/tts-audio/tts-exp/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def localize_path(value: str | Path, repo: Path = REPO) -> Path:
    raw = str(value)
    path = Path(raw)
    if not path.is_file() and raw.startswith(LEGACY_PREFIX):
        path = repo / raw[len(LEGACY_PREFIX):]
    return path.resolve()


def _record_path(row: Mapping[str, Any], key: str, repo: Path) -> tuple[Path, str]:
    path = localize_path(str(row[key]), repo)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(row.get("source_sha256", ""))
    if not expected:
        raise ValueError(f"missing source_sha256 for {row.get('paired_key')}/{row.get('condition')}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"source hash mismatch for {row.get('paired_key')}/{row.get('condition')}")
    return path, actual


def _validate_source(payload: Mapping[str, Any]) -> None:
    split_speakers = payload.get("split_speakers", {})
    if split_speakers.get("test") != ["S0770"]:
        raise ValueError("source manifest no longer pins S0770 as heldout")
    if payload.get("quality_policy", {}).get("fallback_allowed") is not False:
        raise ValueError("source manifest permits alignment fallback")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported source manifest schema")


def select_cohort(
    source_manifest: Path = SOURCE_MANIFEST,
    *,
    speakers: Sequence[str] = SPEAKERS,
    per_speaker: int = PER_SPEAKER,
    repo: Path = REPO,
) -> dict[str, Any]:
    if tuple(speakers) != SPEAKERS:
        raise ValueError(f"cohort speakers are frozen to {SPEAKERS}")
    if per_speaker != PER_SPEAKER:
        raise ValueError(f"cohort quota is frozen to {PER_SPEAKER}")
    source_manifest = source_manifest.resolve()
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    _validate_source(payload)
    source_hash = sha256_file(source_manifest)
    raw_records = payload.get("records", [])
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for raw in raw_records:
        speaker = str(raw.get("speaker_id", ""))
        condition = str(raw.get("condition", ""))
        if speaker in HELDOUT_SPEAKERS:
            continue
        if speaker not in SPEAKERS or condition not in {"natural", "tts"}:
            continue
        if raw.get("strict_gate_pass") is not True or raw.get("alignment_source") != "mfa":
            raise ValueError(f"non-strict source record for {speaker}/{raw.get('paired_key')}")
        key = str(raw.get("paired_key", ""))
        if not key:
            raise ValueError("source record has empty paired_key")
        if condition in grouped[speaker][key]:
            raise ValueError(f"duplicate {condition} record for {speaker}/{key}")
        grouped[speaker][key][condition] = dict(raw)

    records: list[dict[str, Any]] = []
    ordered_keys: list[str] = []
    speaker_counts: dict[str, int] = {}
    split_by_speaker: dict[str, str] = {}
    for speaker in SPEAKERS:
        candidates = grouped[speaker]
        complete = [key for key, arms in candidates.items() if set(arms) == {"natural", "tts"}]
        complete = [key for key, arms in candidates.items() if set(arms) == {"natural", "tts"}]
        complete.sort()
        if len(complete) < per_speaker:
            raise ValueError(f"speaker {speaker} has only {len(complete)} complete pairs")
        selected_keys = complete[:per_speaker]
        speaker_counts[speaker] = len(selected_keys)
        split_values = {str(candidates[key]["natural"].get("split")) for key in selected_keys}
        split_values.update(str(candidates[key]["tts"].get("split")) for key in selected_keys)
        if len(split_values) != 1:
            raise ValueError(f"speaker {speaker} has inconsistent split values: {split_values}")
        split_by_speaker[speaker] = next(iter(split_values))
        for rank, key in enumerate(selected_keys, 1):
            natural = candidates[key]["natural"]
            tts = candidates[key]["tts"]
            if natural.get("speaker_id") != tts.get("speaker_id") or natural.get("transcript") != tts.get("transcript"):
                raise ValueError(f"natural/tts identity mismatch for {key}")
            natural_path, natural_hash = _record_path(natural, "audio_path", repo)
            tts_path, tts_hash = _record_path(tts, "audio_path", repo)
            sample_id = int(natural["sample_id"])
            records.append({
                "record_index": len(records) + 1,
                "speaker_rank": rank,
                "paired_key": key,
                "sample_id": sample_id,
                "speaker_id": speaker,
                "split": str(natural["split"]),
                "transcript": str(natural["transcript"]),
                "audio_path": str(natural_path),
                "natural_source_sha256": natural_hash,
                "strict_natural": {
                    "textgrid_path": str(localize_path(natural["textgrid_path"], repo)),
                    "textgrid_sha256": sha256_file(localize_path(natural["textgrid_path"], repo)),
                    "source_sha256": natural_hash,
                    "phone_coverage": float(natural["phone_coverage"]),
                    "word_coverage": float(natural["word_coverage"]),
                    "tokens": natural.get("tokens", []),
                },
                "tts_source_path": str(tts_path),
                "tts_source_sha256": tts_hash,
                "strict_tts": {
                    "textgrid_path": str(localize_path(tts["textgrid_path"], repo)),
                    "textgrid_sha256": sha256_file(localize_path(tts["textgrid_path"], repo)),
                    "source_sha256": tts_hash,
                    "phone_coverage": float(tts["phone_coverage"]),
                    "word_coverage": float(tts["word_coverage"]),
                    "tokens": tts.get("tokens", []),
                },
            })
            ordered_keys.append(key)

    if len(records) != COUNT or len({r["paired_key"] for r in records}) != COUNT:
        raise ValueError("cohort does not contain exactly 25 unique pairs")
    if len({r["sample_id"] for r in records}) != COUNT:
        raise ValueError("cohort sample_id values are not unique")
    records.sort(key=lambda r: (SPEAKERS.index(str(r["speaker_id"])), int(r["speaker_rank"])))
    ordered_keys = [str(r["paired_key"]) for r in records]
    all_speakers = sorted({str(raw.get("speaker_id")) for raw in raw_records if raw.get("speaker_id")})
    excluded = [speaker for speaker in all_speakers if speaker not in SPEAKERS]
    return {
        "schema_version": 1,
        "manifest_type": "aishell1_mfa_linear_predefined_cohort",
        "dataset": "AISHELL-1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_hash,
        "cohort": {
            "sample_count": COUNT,
            "speaker_count": len(SPEAKERS),
            "per_speaker": PER_SPEAKER,
            "speakers": list(SPEAKERS),
            "split_by_speaker": split_by_speaker,
            "excluded_speakers": excluded,
            "heldout_speakers": list(HELDOUT_SPEAKERS),
            "heldout_excluded": True,
        },
        "selection": {
            "rule": "explicit_speaker_list_then_sorted_paired_key_prefix",
            "score_screening": False,
            "ordered_paired_keys": ordered_keys,
            "ordered_paired_keys_sha256": canonical_sha256(ordered_keys),
            "speaker_counts": speaker_counts,
        },
        "counts": {"selected_pairs": COUNT, "natural_records": COUNT, "tts_records": COUNT},
        "records": records,
    }


def validate_cohort(cohort: Mapping[str, Any], repo: Path = REPO) -> None:
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort":
        raise ValueError("unexpected cohort manifest type")
    meta = cohort.get("cohort", {})
    if meta.get("heldout_excluded") is not True or meta.get("heldout_speakers") != list(HELDOUT_SPEAKERS):
        raise ValueError("cohort heldout contract failed")
    if meta.get("speakers") != list(SPEAKERS) or meta.get("per_speaker") != PER_SPEAKER:
        raise ValueError("cohort speaker contract failed")
    records = list(cohort.get("records", []))
    if len(records) != COUNT:
        raise ValueError(f"expected {COUNT} records, found {len(records)}")
    counts = defaultdict(int)
    for record in records:
        if record.get("speaker_id") not in SPEAKERS or record.get("speaker_id") in HELDOUT_SPEAKERS:
            raise ValueError("cohort contains forbidden speaker")
        counts[str(record["speaker_id"])] += 1
        audio = Path(record["audio_path"])
        if not audio.is_file() or sha256_file(audio) != record["natural_source_sha256"]:
            raise ValueError(f"natural source hash failed for {record.get('paired_key')}")
        tts = Path(record["tts_source_path"])
        if not tts.is_file() or sha256_file(tts) != record["tts_source_sha256"]:
            raise ValueError(f"tts source hash failed for {record.get('paired_key')}")
    if dict(counts) != {speaker: PER_SPEAKER for speaker in SPEAKERS}:
        raise ValueError(f"speaker counts failed: {dict(counts)}")
    keys = [str(r["paired_key"]) for r in records]
    if canonical_sha256(keys) != cohort["selection"]["ordered_paired_keys_sha256"]:
        raise ValueError("ordered paired key hash failed")


def write_cohort(source_manifest: Path, output: Path) -> dict[str, Any]:
    cohort = select_cohort(source_manifest)
    validate_cohort(cohort)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cohort, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cohort = write_cohort(args.source.resolve(), args.output.resolve())
    print(json.dumps({"selected_pairs": len(cohort["records"]), "speakers": SPEAKERS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
