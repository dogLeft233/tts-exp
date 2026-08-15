#!/usr/bin/env python3
"""Build strict MFA tokens and control manifests for fixed-reference TTS."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_aishell1_n25_mfa import clean_lab_text, parse_phones, run_align, sha256_file  # noqa: E402

SPEAKERS = ("S0765", "S0901", "S0912")


def prepare(
    cohort_path: Path,
    fixed_tts_meta_path: Path,
    frozen_tokens_path: Path,
    outdir: Path,
    mfa_bin: Path,
) -> dict[str, Any]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    fixed_meta = json.loads(fixed_tts_meta_path.read_text(encoding="utf-8"))
    frozen_tokens = json.loads(frozen_tokens_path.read_text(encoding="utf-8"))
    records = [row for row in cohort["records"] if str(row["speaker_id"]) in SPEAKERS]
    records.sort(key=lambda row: int(row["sample_id"]))
    if len(records) != 15:
        raise ValueError("expected 15 source cohort records")
    fixed_by_id = {str(row["sample_id"]): row for row in fixed_meta["results"].values()}
    token_by_id = {str(key): row for key, row in frozen_tokens["records"].items()}
    if set(fixed_by_id) != {str(row["sample_id"]) for row in records}:
        raise ValueError("fixed TTS metadata does not exactly cover selected records")
    if set(token_by_id) < {str(row["sample_id"]) for row in records}:
        raise ValueError("frozen tokens do not cover selected records")
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    input_root = outdir / "mfa_input"
    output_root = outdir / "mfa_out"
    logs_root = outdir / "logs"
    input_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {speaker: [] for speaker in SPEAKERS}
    for row in records:
        fixed = fixed_by_id[str(row["sample_id"])]
        audio = Path(str(fixed["canonical_16k_audio"])).resolve()
        if not audio.is_file() or sha256_file(audio) != str(fixed["canonical_audio_sha256"]):
            raise ValueError(f"fixed TTS hash mismatch for {row['sample_id']}")
        grouped[str(row["speaker_id"])].append((row, fixed))
    for speaker, items in grouped.items():
        speaker_input = input_root / speaker
        speaker_input.mkdir(parents=True, exist_ok=True)
        for row, fixed in items:
            stem = f"{speaker}_{row['sample_id']}_fixed_tts"
            shutil.copyfile(fixed["canonical_16k_audio"], speaker_input / f"{stem}.wav")
            (speaker_input / f"{stem}.lab").write_text(clean_lab_text(str(row["transcript"])) + "\n", encoding="utf-8")
        run_align(mfa_bin.resolve(), speaker_input, output_root / speaker, logs_root / f"{speaker}.log")
    token_records: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for row in records:
        sid = str(row["sample_id"])
        speaker = str(row["speaker_id"])
        fixed = fixed_by_id[sid]
        stem = f"{speaker}_{sid}_fixed_tts"
        grid = output_root / speaker / f"{stem}.TextGrid"
        try:
            tts_tokens = parse_phones(grid)
            frozen = token_by_id[sid]
            token_records[sid] = {
                "sample_id": sid,
                "paired_key": row["paired_key"],
                "speaker_id": speaker,
                "split": row["split"],
                "transcript": row["transcript"],
                "natural": frozen["natural"],
                "tts": {
                    "speaker_id": speaker,
                    "condition": "tts_fixed_reference_voice",
                    "tokens": tts_tokens,
                    "textgrid": str(grid),
                    "textgrid_sha256": sha256_file(grid),
                    "audio": str(Path(fixed["canonical_16k_audio"]).resolve()),
                    "audio_sha256": str(fixed["canonical_audio_sha256"]),
                    "phone_count": len(tts_tokens),
                    "speech_phone_count": sum(not token.get("is_silence", False) for token in tts_tokens),
                    "alignment_source": "mfa",
                    "fixed_reference_audio": fixed["fixed_reference_audio"],
                    "fixed_reference_audio_sha256": fixed["fixed_reference_audio_sha256"],
                },
            }
        except Exception as exc:
            failures.append({"sample_id": sid, "error": str(exc)})
    if failures:
        raise RuntimeError(f"fixed TTS MFA failed: {failures}")
    control_cohort = {
        "schema_version": 1,
        "manifest_type": "aishell1_fixed_reference_voice_control",
        "source_cohort": str(cohort_path.resolve()),
        "source_cohort_sha256": sha256_file(cohort_path),
        "fixed_tts_meta": str(fixed_tts_meta_path.resolve()),
        "fixed_tts_meta_sha256": sha256_file(fixed_tts_meta_path),
        "speakers": list(SPEAKERS),
        "sample_count": len(records),
        "records": records,
    }
    token_summary = {
        "schema_version": 1,
        "manifest_type": "aishell1_fixed_reference_voice_tokens",
        "cohort_manifest": str((outdir / "control_cohort.json").resolve()),
        "cohort_manifest_sha256": "",
        "fixed_tts_meta": str(fixed_tts_meta_path.resolve()),
        "fixed_tts_meta_sha256": sha256_file(fixed_tts_meta_path),
        "samples_total": len(records),
        "samples_ok": len(token_records),
        "failures": failures,
        "complete": len(token_records) == len(records),
        "records": token_records,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    control_path = outdir / "control_cohort.json"
    control_path.write_text(json.dumps(control_cohort, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    token_summary["cohort_manifest_sha256"] = sha256_file(control_path)
    (outdir / "tokens.json").write_text(json.dumps(token_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"complete": True, "samples_ok": len(token_records), "control_cohort": str(control_path), "tokens": str(outdir / "tokens.json")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--fixed-tts-meta", type=Path, required=True)
    parser.add_argument("--frozen-tokens", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mfa-bin", type=Path, default=Path.home() / "miniconda3/envs/mfa/bin/mfa")
    args = parser.parse_args(argv)
    result = prepare(args.cohort.resolve(), args.fixed_tts_meta.resolve(), args.frozen_tokens.resolve(), args.outdir.resolve(), args.mfa_bin.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
