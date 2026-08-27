from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.experiments.lrs3_mfa_linear_replacement import run_stage01 as base_stage01
from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    UNKNOWN_LABELS,
    parsed_token_hash,
    transcript_sha256,
)
from scripts.experiments.lrs3_mfa_linear_replacement.protocol import (
    canonical_sha256,
    file_sha256,
    load_json,
    write_json,
)

from .mfa3_alignment import (
    MFA_ACOUSTIC_MODEL,
    MFA_DEFAULT_ROOT,
    MFA_DICTIONARY,
    NORMALIZATION_POLICY_ID,
    normalization_events,
    normalize_mfa3_transcript,
    run_mfa3_alignment,
)

PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_20260825"
EXPECTED_RECORDS = 24
STAGE_PREFIX = "01_candidate_audio_mfa3_retry"
REPO = Path(__file__).resolve().parents[3]


def stage_id_for_output(output_dir: Path) -> str:
    stage_id = output_dir.name
    if not stage_id.startswith(STAGE_PREFIX):
        raise ValueError(f"unexpected MFA3 Stage01 output directory name: {stage_id}")
    return stage_id


def validate_stage00_binding(stage00_path: Path, stage_id: str, stage00: Mapping[str, Any]) -> None:
    retry_suffix = stage_id.removeprefix(STAGE_PREFIX)
    if not retry_suffix:
        raise ValueError("MFA3 Stage01 retry suffix is missing")
    expected_stage00 = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_20260825" / f"00_protocol_lock_mfa3_retry{retry_suffix}" / "manifest.json"
    if stage00_path.resolve() != expected_stage00.resolve():
        raise ValueError("Stage00 manifest path is not the canonical MFA3 retry lock")
    expected_stage00_id = f"00_protocol_lock_mfa3_retry{retry_suffix}"
    if stage00.get("stage_id") != expected_stage00_id:
        raise ValueError("Stage00 manifest stage ID does not match Stage01 retry")
    digest_path = stage00_path.parent / "manifest.sha256"
    if not digest_path.is_file() or digest_path.read_text(encoding="utf-8").strip() != file_sha256(stage00_path):
        raise ValueError("Stage00 manifest companion SHA-256 lock is missing or mismatched")

def _assert_bound_file(label: str, supplied_path: Path, binding: Mapping[str, Any]) -> None:
    supplied = supplied_path.resolve()
    expected = Path(str(binding.get("path", ""))).resolve()
    expected_hash = str(binding.get("sha256", ""))
    if supplied != expected:
        raise ValueError(f"MFA3 {label} path does not match Stage00")
    if not expected_hash or file_sha256(supplied) != expected_hash:
        raise ValueError(f"MFA3 {label} hash does not match Stage00")


def validate_mfa_contract(
    stage00: Mapping[str, Any],
    *,
    mfa_executable: str,
    mfa_root_dir: Path,
) -> None:
    contract = stage00.get("mfa_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Stage00 MFA3 contract is missing")
    if contract.get("version") != "3.4.1":
        raise ValueError("Stage00 MFA3 version contract is unexpected")
    if contract.get("root_dir") != str(mfa_root_dir.resolve()):
        raise ValueError("MFA3 root directory does not match Stage00")
    if contract.get("dictionary", {}).get("name") != MFA_DICTIONARY:
        raise ValueError("Stage00 MFA3 dictionary contract is unexpected")
    if contract.get("acoustic_model", {}).get("name") != MFA_ACOUSTIC_MODEL:
        raise ValueError("Stage00 MFA3 acoustic model contract is unexpected")
    _assert_bound_file("executable", Path(mfa_executable), contract["executable"])
    _assert_bound_file("root config", mfa_root_dir / "global_config.yaml", contract["root_config"])
    _assert_bound_file("dictionary", Path(str(contract["dictionary"]["path"])), contract["dictionary"])
    _assert_bound_file("acoustic model", Path(str(contract["acoustic_model"]["path"])), contract["acoustic_model"])


def validate_candidate_result_ids(results: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]) -> None:
    actual_ids = [str(result.get("sample_id", "")) for result in results]
    expected = [str(sample_id) for sample_id in expected_ids]
    if actual_ids != expected:
        raise ValueError("candidate results do not match the ordered frozen cohort")


def _normalized_records(stage00: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = base_stage01._cohort_records_with_transcripts(stage00)
    if len(records) != EXPECTED_RECORDS:
        raise ValueError("Stage00 cohort is not n=24")
    for record in records:
        raw_transcript = str(record["transcript"])
        raw_tts_transcript = str(record["tts_transcript"])
        normalized = normalize_mfa3_transcript(raw_transcript)
        normalized_tts = normalize_mfa3_transcript(raw_tts_transcript)
        if normalized != normalized_tts:
            raise ValueError(f"MFA3 normalized natural/TTS transcript mismatch: {record['sample_id']}")
        record["raw_transcript_sha256"] = transcript_sha256(raw_transcript)
        record["raw_tts_transcript_sha256"] = transcript_sha256(raw_tts_transcript)
        record["normalization_events"] = normalization_events(raw_transcript)
        record["transcript"] = normalized
        record["tts_transcript"] = normalized_tts
        record["transcript_sha256"] = transcript_sha256(normalized)
        record["tts_transcript_sha256"] = transcript_sha256(normalized_tts)
    return records


def prepare_and_align(
    stage00: Mapping[str, Any],
    output_dir: Path,
    *,
    stage_id: str,
    mfa_executable: str,
    mfa_root_dir: Path,
) -> list[dict[str, Any]]:
    records = _normalized_records(stage00)
    expected_ids = [str(record["sample_id"]) for record in records]
    natural_input = output_dir / "_mfa_input" / "natural"
    tts_input = output_dir / "_mfa_input" / "tts"
    natural_output = output_dir / "natural_textgrids"
    tts_output = output_dir / "tts_textgrids"
    tts_records = [dict(record) for record in records]
    base_stage01.prepare_mfa_corpus(
        records,
        natural_input,
        audio_key="natural_audio",
        audio_hash_key="natural_audio_sha256",
        expected_records=records,
    )
    base_stage01.prepare_mfa_corpus(
        tts_records,
        tts_input,
        audio_key="tts_audio",
        transcript_key="tts_transcript",
        audio_hash_key="tts_audio_sha256",
        expected_records=records,
    )
    natural_run = run_mfa3_alignment(
        natural_input,
        natural_output,
        mfa_executable=mfa_executable,
        mfa_root_dir=mfa_root_dir,
        expected_sample_ids=expected_ids,
    )
    tts_run = run_mfa3_alignment(
        tts_input,
        tts_output,
        mfa_executable=mfa_executable,
        mfa_root_dir=mfa_root_dir,
        expected_sample_ids=expected_ids,
    )
    rows = base_stage01._alignment_rows(records, natural_output, tts_output)
    by_id = {str(record["sample_id"]): record for record in records}
    for row in rows:
        record = by_id[str(row["sample_id"])]
        row.update({
            "raw_transcript_sha256": record["raw_transcript_sha256"],
            "raw_tts_transcript_sha256": record["raw_tts_transcript_sha256"],
            "normalization_policy_id": NORMALIZATION_POLICY_ID,
            "normalization_events": record["normalization_events"],
            "normalized_transcript_sha256": transcript_sha256(record["transcript"]),
            "normalized_tts_transcript_sha256": transcript_sha256(record["tts_transcript"]),
        })
    write_json(output_dir / "mfa_runtime.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mfa_executable": mfa_executable,
        "mfa_root_dir": str(mfa_root_dir.resolve()),
        "dictionary": MFA_DICTIONARY,
        "acoustic_model": MFA_ACOUSTIC_MODEL,
        "natural": natural_run,
        "tts": tts_run,
    })
    write_json(output_dir / "alignment_manifest.json", {
        "schema_version": 1,
        "stage_id": stage_id,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "record_count": len(rows),
        "ordered_sample_ids_sha256": canonical_sha256(expected_ids),
        "normalization_policy_id": NORMALIZATION_POLICY_ID,
        "mfa": {
            "executable": mfa_executable,
            "root_dir": str(mfa_root_dir.resolve()),
            "dictionary": MFA_DICTIONARY,
            "acoustic_model": MFA_ACOUSTIC_MODEL,
        },
        "records": rows,
    })
    return rows


def _unknown_alignment_rows(output_dir: Path) -> list[dict[str, Any]]:
    return base_stage01._unknown_alignment_rows(output_dir)


def _write_alignment_failure(output_dir: Path, stage_id: str, exc: Exception) -> dict[str, Any]:
    unknown_rows = _unknown_alignment_rows(output_dir)
    failure = {
        "schema_version": 1,
        "stage_id": stage_id,
        "protocol_id": PROTOCOL_ID,
        "status": "blocked",
        "decision": "BLOCKED",
        "error": str(exc),
        "alignment_errors": getattr(exc, "failures", []),
        "unknown_phone_records": unknown_rows,
        "unknown_phone_record_count": len({row["sample_id"] for row in unknown_rows}),
        "candidate_generation_started": False,
    }
    write_json(output_dir / "alignment_failure.json", failure)
    summary = {
        "schema_version": 1,
        "stage_id": stage_id,
        "protocol_id": PROTOCOL_ID,
        "status": "blocked",
        "decision": "BLOCKED",
        "scientific_decision": "not_available",
        "record_count": 0,
        "expected_record_count": EXPECTED_RECORDS,
        "failure": failure,
        "media_access": {
            "fit_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
            "gpu_used": False,
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "decision.json", {
        "stage_id": stage_id,
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "next_allowed_stage": None,
        "reason": str(exc),
    })
    return summary


def run(
    *,
    stage00_path: Path,
    output_dir: Path,
    mfa_executable: str,
    mfa_root_dir: Path,
    align_only: bool,
    device: str,
    local_knn_vc: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty Stage01 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_id = stage_id_for_output(output_dir)
    stage00 = load_json(stage00_path)
    if stage00.get("status") != "complete" or stage00.get("decision") != "GO" or stage00.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("MFA3 Stage00 lock is not a complete GO for this protocol")
    try:
        validate_stage00_binding(stage00_path, stage_id, stage00)
        validate_mfa_contract(
            stage00,
            mfa_executable=mfa_executable,
            mfa_root_dir=mfa_root_dir,
        )
        alignment_rows = prepare_and_align(
            stage00,
            output_dir,
            stage_id=stage_id,
            mfa_executable=mfa_executable,
            mfa_root_dir=mfa_root_dir,
        )
    except Exception as exc:
        return _write_alignment_failure(output_dir, stage_id, exc)
    if align_only:
        summary = {
            "schema_version": 1,
            "stage_id": stage_id,
            "protocol_id": PROTOCOL_ID,
            "status": "alignment_complete_candidate_pending",
            "decision": "BLOCKED",
            "record_count": len(alignment_rows),
            "media_access": {
                "fit_media_opened": True,
                "internal_dev_media_opened": False,
                "validation_media_opened": False,
                "test_media_opened": False,
                "gpu_used": False,
            },
        }
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "decision.json", {
            "stage_id": stage_id,
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "reason": "alignment-only preflight; candidate generation intentionally not run",
        })
        return summary
    if device != "cuda":
        raise ValueError("candidate generation must use the registered CUDA device")
    gpu_gate = None
    try:
        gpu_gate = base_stage01.assert_gpu_ready()
        results = base_stage01.generate_candidates(
            stage00,
            alignment_rows,
            output_dir,
            device=device,
            local_knn_vc=local_knn_vc,
        )
        expected_ids = [str(record["sample_id"]) for record in stage00["cohort"]["records"]]
        validate_candidate_result_ids(results, expected_ids)
    except Exception as exc:
        partial_audio = sorted(str(path) for path in (output_dir / "audio").glob("*.wav"))
        failure = {
            "schema_version": 1,
            "stage_id": stage_id,
            "protocol_id": PROTOCOL_ID,
            "status": "blocked",
            "decision": "BLOCKED",
            "error": str(exc),
            "candidate_generation_started": gpu_gate is not None,
            "partial_audio_paths": partial_audio,
            "partial_audio_count": len(partial_audio),
        }
        write_json(output_dir / "candidate_failure.json", failure)
        summary = {
            "schema_version": 1,
            "stage_id": stage_id,
            "protocol_id": PROTOCOL_ID,
            "status": "blocked",
            "decision": "BLOCKED",
            "scientific_decision": "not_available",
            "record_count": 0,
            "expected_record_count": EXPECTED_RECORDS,
            "failure": failure,
            "gpu_used": gpu_gate is not None,
            "media_access": {
                "fit_media_opened": True,
                "internal_dev_media_opened": False,
                "validation_media_opened": False,
                "test_media_opened": False,
            },
        }
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "decision.json", {
            "stage_id": stage_id,
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "next_allowed_stage": None,
            "reason": str(exc),
        })
        return summary
    failures = [] if len(results) == EXPECTED_RECORDS else [{"error": "not all 24 candidates completed"}]
    status = "complete" if not failures else "incomplete"
    decision = "GO" if status == "complete" else "BLOCKED"
    summary = {
        "schema_version": 1,
        "stage_id": stage_id,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "decision": decision,
        "scientific_decision": "not_available",
        "record_count": len(results),
        "expected_record_count": EXPECTED_RECORDS,
        "failures": failures,
        "gpu_used": True,
        "media_access": {
            "fit_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json(output_dir / "candidate_manifest.json", {
        "schema_version": 1,
        "stage_id": stage_id,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "decision": decision,
        "record_count": len(results),
        "normalization_policy_id": NORMALIZATION_POLICY_ID,
        "results": results,
    })
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "decision.json", {
        "stage_id": stage_id,
        "engineering_decision": decision,
        "scientific_decision": "not_available",
        "next_allowed_stage": stage_id.replace("01_candidate_audio_mfa3", "02_wav2lip_strict_replacement_mfa3", 1) if decision == "GO" else None,
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mfa-executable", default="/home/wjj/miniconda3/envs/mfa3/bin/mfa")
    parser.add_argument("--mfa-root-dir", type=Path, default=MFA_DEFAULT_ROOT)
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-knn-vc", type=Path, default=Path.home() / ".cache/torch/hub/bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358")
    args = parser.parse_args()
    result = run(
        stage00_path=args.stage00.resolve(),
        output_dir=args.output.resolve(),
        mfa_executable=args.mfa_executable,
        mfa_root_dir=args.mfa_root_dir.resolve(),
        align_only=args.align_only,
        device=args.device,
        local_knn_vc=args.local_knn_vc.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("decision") == "GO" or args.align_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
