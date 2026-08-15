#!/usr/bin/env python3
"""Run the n25 Wav2Lip + SyncNet matrix with Qwen cloud TTS provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_aishell1_n25_wav2lip_syncnet as evaluator

EXPECTED_SPEAKERS = ("S0765", "S0901", "S0906", "S0912", "S0913")
QWEN_PROVIDER = "dashscope_vc"
QWEN_MODEL = "qwen3-tts-vc-2026-01-22"


def build_records(
    cohort_path: Path,
    tts_meta_path: Path,
    mfa_summary_path: Path,
    face: Path,
) -> list[dict[str, Any]]:
    if evaluator.file_sha256(evaluator.WAV2LIP_CHECKPOINT) != evaluator.EXPECTED_WAV2LIP_SHA256:
        raise ValueError("Wav2Lip checkpoint hash changed")
    if evaluator.file_sha256(evaluator.SYNCNET_MODEL) != evaluator.EXPECTED_SYNCNET_SHA256:
        raise ValueError("SyncNet model hash changed")
    face = face.resolve()
    if not face.is_file():
        raise FileNotFoundError(face)

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    tts_meta = json.loads(tts_meta_path.read_text(encoding="utf-8"))
    mfa_summary = json.loads(mfa_summary_path.read_text(encoding="utf-8"))
    cohort_records = list(cohort.get("records", []))
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort" or len(cohort_records) != 25:
        raise ValueError("invalid n=25 cohort")
    if cohort.get("cohort", {}).get("heldout_excluded") is not True:
        raise ValueError("cohort does not exclude heldout speaker")
    if tts_meta.get("manifest_type") != "aishell1_qwen_cloud_tts_n25" or tts_meta.get("complete") is not True:
        raise ValueError("Qwen TTS metadata is incomplete or has the wrong manifest type")
    if tts_meta.get("provider") != QWEN_PROVIDER or tts_meta.get("model") != QWEN_MODEL:
        raise ValueError("unexpected Qwen provider/model")
    if mfa_summary.get("samples_ok") != 25 or mfa_summary.get("failures"):
        raise ValueError("Qwen MFA-linear metadata is incomplete")

    tts_by_id = {str(row["sample_id"]): row for row in tts_meta.get("results", {}).values()}
    mfa_by_id = {str(row["sample_id"]): row for row in mfa_summary.get("results", {}).values()}
    if set(tts_by_id) != {str(row["sample_id"]) for row in cohort_records} or len(mfa_by_id) != 25:
        raise ValueError("Qwen or MFA-linear metadata does not cover the cohort")
    face_hash = evaluator.file_sha256(face)
    records: list[dict[str, Any]] = []

    for row in cohort_records:
        sample_id = str(row["sample_id"])
        speaker_id = str(row["speaker_id"])
        paired_key = str(row["paired_key"])
        tts = tts_by_id[sample_id]
        mfa_row = mfa_by_id[sample_id]
        if str(tts.get("paired_key")) != paired_key or str(tts.get("speaker_id")) != speaker_id:
            raise ValueError(f"Qwen identity mismatch for {paired_key}")
        if str(mfa_row.get("paired_key")) != paired_key or str(mfa_row.get("speaker_id")) != speaker_id:
            raise ValueError(f"MFA-linear identity mismatch for {paired_key}")

        natural = Path(str(row["audio_path"])).resolve()
        raw_tts = Path(str(tts["canonical_16k_audio"])).resolve()
        mfa_linear = Path(str(mfa_row["audio_path"])).resolve()
        paths = {"natural_raw": natural, "raw_tts": raw_tts, "mfa_linear": mfa_linear}
        hashes = {
            "natural_raw": str(row["natural_source_sha256"]),
            "raw_tts": str(tts["canonical_audio_sha256"]),
            "mfa_linear": str(mfa_row["audio_sha256"]),
        }
        if str(tts.get("reference_audio_sha256")) != hashes["natural_raw"]:
            raise ValueError(f"Qwen reference audio mismatch for {paired_key}")
        if str(tts.get("source_audio_sha256")) != hashes["raw_tts"]:
            raise ValueError(f"Qwen canonical source hash mismatch for {paired_key}")

        for arm in evaluator.ARMS:
            audio = paths[arm]
            if not audio.is_file() or evaluator.file_sha256(audio) != hashes[arm]:
                raise ValueError(f"audio hash mismatch for {paired_key}/{arm}")
            records.append(
                {
                    "sample_id": sample_id,
                    "paired_key": paired_key,
                    "speaker_id": speaker_id,
                    "split": row["split"],
                    "transcript": row["transcript"],
                    "arm": arm,
                    "source_condition": arm,
                    "audio": str(audio),
                    "audio_sha256": hashes[arm],
                    "face": str(face),
                    "face_sha256": face_hash,
                    "cohort_manifest": str(cohort_path.resolve()),
                    "cohort_manifest_sha256": evaluator.file_sha256(cohort_path),
                    "qwen_tts_provider": str(tts["provider"]),
                    "qwen_tts_model": str(tts["model"]),
                    "qwen_reference_audio_sha256": str(tts["reference_audio_sha256"]),
                    "cohort_strict_tts_source_sha256": str(row["tts_source_sha256"]),
                }
            )

    if {str(row["speaker_id"]) for row in cohort_records} != set(EXPECTED_SPEAKERS):
        raise ValueError("unexpected speaker set")
    evaluator.validate_records(records)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--mfa-summary", type=Path, required=True)
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    cohort = args.cohort.resolve()
    tts_meta = args.tts_meta.resolve()
    mfa_summary = args.mfa_summary.resolve()
    face = args.face.resolve()
    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")

    records = build_records(cohort, tts_meta, mfa_summary, face)
    evaluator.write_json(
        outdir / "manifest.json",
        {
            "schema_version": 1,
            "manifest_type": "aishell1_qwen_cloud_n25_wav2lip_syncnet",
            "heldout_excluded": True,
            "single_face_protocol": True,
            "face_protocol": "single_fixed_face_from_aishell1_S0765_sample1",
            "sample_count": 25,
            "expected_scores": 75,
            "arms": list(evaluator.ARMS),
            "qwen_tts_provider": QWEN_PROVIDER,
            "qwen_tts_model": QWEN_MODEL,
            "cohort_manifest": str(cohort),
            "cohort_manifest_sha256": evaluator.file_sha256(cohort),
            "tts_meta": str(tts_meta),
            "tts_meta_sha256": evaluator.file_sha256(tts_meta),
            "mfa_summary": str(mfa_summary),
            "mfa_summary_sha256": evaluator.file_sha256(mfa_summary),
            "face": str(face),
            "face_sha256": evaluator.file_sha256(face),
            "records": records,
        },
    )
    summary = evaluator.evaluate(
        records,
        outdir,
        face_protocol="single_fixed_face_from_aishell1_S0765_sample1",
    )
    summary.update(
        {
            "evaluation_input": "qwen_cloud_tts_n25",
            "qwen_tts_provider": QWEN_PROVIDER,
            "qwen_tts_model": QWEN_MODEL,
            "cohort_manifest": str(cohort),
            "tts_meta": str(tts_meta),
            "mfa_summary": str(mfa_summary),
        }
    )
    evaluator.write_json(outdir / "summary.json", summary)
    evaluator.write_json(outdir / "analysis.json", evaluator.analyze(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
