from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .protocol import (
    CANONICAL_STAGE00_ID,
    CANONICAL_STAGE00_PATH,
    MFA_EXECUTABLE,
    MFA_ROOT_DIR,
    PROTOCOL_ID,
    STAGE01_ID,
    canonical_sha256,
    file_sha256,
    load_json,
    parse_alignment_pair,
    prepare_corpus,
    run_mfa3_alignment,
    validate_stage00,
    write_json,
)


def run(stage00_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty exploratory Stage01 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage00 = load_json(stage00_path)
    validate_stage00(
        stage00_path,
        stage00,
        expected_stage_id=CANONICAL_STAGE00_ID,
        expected_path=CANONICAL_STAGE00_PATH,
    )
    records = list(stage00["cohort"]["records"])
    expected_ids = [str(record["sample_id"]) for record in records]
    if canonical_sha256(expected_ids) != stage00["cohort"]["ordered_sample_ids_sha256"]:
        raise ValueError("exploratory Stage00 ordered cohort hash mismatch")

    natural_input = output_dir / "_mfa_input" / "natural"
    tts_input = output_dir / "_mfa_input" / "tts"
    natural_output = output_dir / "natural_textgrids"
    tts_output = output_dir / "tts_textgrids"
    prepare_corpus(records, natural_input, audio_key="natural_audio", hash_key="natural_audio_sha256")
    prepare_corpus(records, tts_input, audio_key="tts_audio", hash_key="tts_audio_sha256")

    natural_run = run_mfa3_alignment(
        natural_input,
        natural_output,
        expected_sample_ids=expected_ids,
        mfa_executable=MFA_EXECUTABLE,
        mfa_root_dir=MFA_ROOT_DIR,
    )
    tts_run = run_mfa3_alignment(
        tts_input,
        tts_output,
        expected_sample_ids=expected_ids,
        mfa_executable=MFA_EXECUTABLE,
        mfa_root_dir=MFA_ROOT_DIR,
    )

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        row, row_failures = parse_alignment_pair(
            record,
            natural_output / f"{sample_id}.TextGrid",
            tts_output / f"{sample_id}.TextGrid",
        )
        if row is None:
            failures.extend(row_failures)
        else:
            successes.append(row)

    clean_ids = [str(row["sample_id"]) for row in successes]
    unknown_rows = [row for row in failures if "spn" in str(row.get("error", "")).lower()]
    alignment_manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_exploratory_alignment_screen",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE01_ID,
        "status": "complete" if not failures else "partial",
        "ordered_input_count": len(records),
        "ordered_input_sample_ids_sha256": canonical_sha256(expected_ids),
        "clean_record_count": len(successes),
        "clean_sample_ids_sha256": canonical_sha256(clean_ids),
        "natural_textgrid_count": natural_run["textgrid_count"],
        "tts_textgrid_count": tts_run["textgrid_count"],
        "natural_mfa": natural_run,
        "tts_mfa": tts_run,
        "records": successes,
        "failures": failures,
        "unknown_phone_failures": unknown_rows,
        "mfa": {
            "executable": str(MFA_EXECUTABLE.resolve()),
            "executable_sha256": file_sha256(MFA_EXECUTABLE),
            "root_dir": str(MFA_ROOT_DIR.resolve()),
            "dictionary": "english_mfa",
            "acoustic_model": "english_mfa",
        },
    }
    write_json(output_dir / "alignment_manifest.json", alignment_manifest)
    write_json(output_dir / "clean_records.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE01_ID,
        "record_count": len(successes),
        "ordered_sample_ids_sha256": canonical_sha256(clean_ids),
        "sample_ids": clean_ids,
    })
    summary = {
        "schema_version": 1,
        "stage_id": STAGE01_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "complete" if not failures else "partial",
        "engineering_decision": "GO" if len(successes) >= 24 else "BLOCKED",
        "scientific_decision": "not_available",
        "input_record_count": len(records),
        "clean_record_count": len(successes),
        "failure_row_count": len(failures),
        "unknown_phone_failure_count": len(unknown_rows),
        "candidate_generation_started": False,
        "gpu_used": False,
        "next_allowed_stage": "02_candidate_audio_exploratory" if len(successes) >= 24 else None,
        "media_access": {
            "fit_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "decision.json", {
        "stage_id": STAGE01_ID,
        "engineering_decision": summary["engineering_decision"],
        "scientific_decision": "not_available",
        "next_allowed_stage": summary["next_allowed_stage"],
        "reason": (
            "at least 24 complete natural/TTS MFA3 phone alignments are available"
            if len(successes) >= 24
            else "fewer than 24 complete natural/TTS MFA3 phone alignments are available"
        ),
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.stage00.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
