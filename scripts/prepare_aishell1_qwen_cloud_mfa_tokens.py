#!/usr/bin/env python3
"""Build clean MFA phone tokens for the Qwen cloud n25 MFA-linear arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_SPEAKERS = ("S0765", "S0901", "S0906", "S0912", "S0913")
SILENCE_LABELS = {"", "sil", "sp", "<sil>"}
TONE_MARKS = re.compile(r"[˥˦˧˨˩]")
INTERVAL_RE = re.compile(
    r"intervals\s*\[\d+\]:\s*"
    r"xmin\s*=\s*([^\n]+)\s*"
    r"xmax\s*=\s*([^\n]+)\s*"
    r"text\s*=\s*\"(.*?)\"",
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalized_label(raw: str) -> str:
    return TONE_MARKS.sub("", raw.strip())


def parse_phones(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    marker = None
    for name in ("phones", "phone"):
        candidate = f'name = "{name}"'
        if candidate in raw:
            marker = candidate
            break
    if marker is None:
        raise ValueError(f"phones tier missing: {path}")
    tier = raw.split(marker, 1)[1]
    next_tier = re.search(r"\n\s*name\s*=\s*\"", tier)
    if next_tier:
        tier = tier[: next_tier.start()]

    tokens: list[dict[str, Any]] = []
    previous_end = 0.0
    for match in INTERVAL_RE.finditer(tier):
        start = float(match.group(1).strip())
        end = float(match.group(2).strip())
        raw_token = match.group(3).strip()
        if end <= start:
            continue
        if start < previous_end - 1e-7:
            raise ValueError(f"overlapping phone intervals in {path}")
        label = normalized_label(raw_token)
        silence = raw_token.casefold() in SILENCE_LABELS
        unknown_speech = not silence and label.casefold() in {"spn", "<eps>"}
        tokens.append(
            {
                "token": label,
                "raw_token": raw_token,
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "duration_s": round(end - start, 6),
                "is_silence": silence,
                "is_unknown_speech": unknown_speech,
                "confidence": 1.0,
            }
        )
        previous_end = end
    if not tokens:
        raise ValueError(f"phones tier is empty: {path}")
    if tokens[0]["start_s"] > 1e-6:
        raise ValueError(f"phones tier does not start at zero: {path}")
    if not any(not token["is_silence"] for token in tokens):
        raise ValueError(f"phones tier has no speech phones: {path}")
    return tokens


def load_inputs(
    cohort_path: Path,
    tts_meta_path: Path,
    alignment_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    tts_meta = json.loads(tts_meta_path.read_text(encoding="utf-8"))
    records = list(cohort.get("records", []))
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort":
        raise ValueError("unexpected cohort manifest type")
    if len(records) != 25:
        raise ValueError(f"expected 25 cohort records, found {len(records)}")
    speakers = {str(row.get("speaker_id")) for row in records}
    if speakers != set(EXPECTED_SPEAKERS) or "S0770" in speakers:
        raise ValueError(f"unexpected or forbidden speaker set: {sorted(speakers)}")
    if any(sum(str(row.get("speaker_id")) == speaker for row in records) != 5 for speaker in EXPECTED_SPEAKERS):
        raise ValueError("expected five records per speaker")
    if tts_meta.get("manifest_type") != "aishell1_qwen_cloud_tts_n25":
        raise ValueError("unexpected Qwen TTS metadata type")
    tts_by_id = {str(row["sample_id"]): row for row in tts_meta.get("results", {}).values()}
    if set(tts_by_id) != {str(row["sample_id"]) for row in records}:
        raise ValueError("Qwen TTS metadata does not exactly cover cohort")
    if not alignment_root.is_dir():
        raise FileNotFoundError(alignment_root)
    return cohort, tts_by_id, records


def build_tokens(
    cohort_path: Path,
    tts_meta_path: Path,
    alignment_root: Path,
    out_path: Path,
) -> dict[str, Any]:
    cohort, tts_by_id, records = load_inputs(cohort_path, tts_meta_path, alignment_root)
    input_root = alignment_root / "input"
    mfa_root = alignment_root / "mfa_out"
    token_records: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    for row in records:
        sample_id = str(row["sample_id"])
        speaker = str(row["speaker_id"])
        tts = tts_by_id[sample_id]
        try:
            if str(tts.get("paired_key")) != str(row.get("paired_key")):
                raise ValueError("cohort/TTS paired_key mismatch")
            sides: dict[str, Any] = {}
            for condition, source_path in (
                ("natural", Path(str(row["audio_path"]))),
                ("tts", Path(str(tts["canonical_16k_audio"]))),
            ):
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                stem = f"{speaker}_{sample_id}_{'qwen' if condition == 'tts' else 'natural'}"
                grid_path = mfa_root / f"{stem}.TextGrid"
                input_audio = input_root / f"{stem}.wav"
                if not grid_path.is_file() or not input_audio.is_file():
                    raise FileNotFoundError(f"missing alignment pair for {sample_id}/{condition}")
                tokens = parse_phones(grid_path)
                source_hash = sha256_file(source_path)
                input_hash = sha256_file(input_audio)
                if source_hash != input_hash:
                    raise ValueError(f"MFA input audio hash mismatch for {sample_id}/{condition}")
                expected_hash = (
                    tts.get("source_audio_sha256") if condition == "tts" else row.get("audio_sha256")
                )
                if expected_hash is not None and str(expected_hash) != input_hash:
                    raise ValueError(f"source metadata hash mismatch for {sample_id}/{condition}")
                sides[condition] = {
                    "condition": condition,
                    "tokens": tokens,
                    "textgrid": str(grid_path.resolve()),
                    "textgrid_sha256": sha256_file(grid_path),
                    "audio": str(source_path.resolve()),
                    "audio_sha256": input_hash,
                    "mfa_input_audio": str(input_audio.resolve()),
                    "mfa_input_audio_sha256": input_hash,
                    "phone_count": len(tokens),
                    "speech_phone_count": sum(not token["is_silence"] for token in tokens),
                    "silence_duration_s": round(
                        sum(token["duration_s"] for token in tokens if token["is_silence"]), 6
                    ),
                    "unknown_speech_count": sum(token["is_unknown_speech"] for token in tokens),
                }
            token_records[sample_id] = {
                "sample_id": sample_id,
                "paired_key": row["paired_key"],
                "speaker_id": speaker,
                "split": row["split"],
                "transcript": row["transcript"],
                "natural": sides["natural"],
                "tts": sides["tts"],
                "tts_provider": "qwen3-tts-vc-2026-01-22",
            }
        except Exception as exc:
            failures.append({"sample_id": sample_id, "paired_key": row["paired_key"], "error": str(exc)})

    unknown_count = sum(
        int(side["unknown_speech_count"])
        for record in token_records.values()
        for side in (record["natural"], record["tts"])
    )
    result = {
        "schema_version": 2,
        "manifest_type": "aishell1_qwen_cloud_mfa3_n25_tokens",
        "cohort_manifest": str(cohort_path.resolve()),
        "cohort_manifest_sha256": sha256_file(cohort_path),
        "tts_meta": str(tts_meta_path.resolve()),
        "tts_meta_sha256": sha256_file(tts_meta_path),
        "alignment_root": str(alignment_root.resolve()),
        "alignment_policy": "MFA-3.4.1 clean natural/Qwen paired alignment",
        "label_normalization": "remove Mandarin IPA tone marks U+02E5..U+02E9 for matching; preserve raw_token",
        "silence_policy": "empty, sil, sp, and <sil> are silence; spn is unknown speech, never silence",
        "samples_total": len(records),
        "samples_ok": len(token_records),
        "failures": failures,
        "unknown_speech_token_count": unknown_count,
        "complete": len(token_records) == 25 and not failures and unknown_count == 0,
        "records": token_records,
    }
    write_json(out_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_tokens(
        args.cohort.resolve(), args.tts_meta.resolve(), args.alignment_root.resolve(), args.out.resolve()
    )
    print(
        json.dumps(
            {
                "samples_ok": result["samples_ok"],
                "failures": len(result["failures"]),
                "unknown_speech_token_count": result["unknown_speech_token_count"],
                "complete": result["complete"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
