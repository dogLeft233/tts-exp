#!/usr/bin/env python3
"""Create a new physical-overlay manifest bound to the original EVAL identity.

This narrowly repairs one mistyped source_identity_sha256 field. It refuses to
write over the original overlay and verifies the old/v2 producer records,
original eligibility, probe text, source audio hashes and physical Grid/LAB
hashes before emitting a new immutable candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SAMPLE_ID = "S0064_BAC009S0064W0121"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json({key: item for key, item in value.items() if key != "manifest_sha256"}))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sha256_fields(value: Any, path: str = "$") -> int:
    count = 0
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{path}.{key}"
            if "sha256" in str(key).lower() and item is not None:
                if not isinstance(item, str) or len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item.lower()):
                    raise ValueError(f"malformed SHA-256 at {current}")
                count += 1
            count += validate_sha256_fields(item, current)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            count += validate_sha256_fields(item, f"{path}[{index}]")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--original-producer", type=Path, required=True)
    parser.add_argument("--v2-producer", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    for label, path in (
        ("overlay", args.overlay), ("original producer", args.original_producer),
        ("v2 producer", args.v2_producer), ("protocol", args.protocol),
        ("eligibility", args.eligibility), ("probe", args.probe),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if args.output.exists() or args.audit_output.exists():
        raise FileExistsError("refusing to overwrite a candidate or audit file")

    overlay_raw_sha = sha256_file(args.overlay)
    overlay = read_json(args.overlay)
    producer_raw_sha = sha256_file(args.original_producer)
    producer = read_json(args.original_producer)
    v2_raw_sha = sha256_file(args.v2_producer)
    v2 = read_json(args.v2_producer)
    protocol = read_json(args.protocol)
    eligibility_raw_sha = sha256_file(args.eligibility)
    eligibility = read_json(args.eligibility)
    probe_raw_sha = sha256_file(args.probe)
    probe = read_json(args.probe)

    if overlay.get("manifest_sha256") != fingerprint(overlay):
        raise ValueError("original overlay embedded fingerprint mismatch")
    if producer.get("manifest_sha256") != fingerprint(producer):
        raise ValueError("original EVAL producer embedded fingerprint mismatch")
    if v2.get("manifest_sha256") != fingerprint(v2):
        raise ValueError("v2 producer embedded fingerprint mismatch")
    if protocol.get("manifest_sha256") != fingerprint(protocol):
        raise ValueError("protocol embedded fingerprint mismatch")
    if eligibility.get("manifest_sha256") != fingerprint(eligibility):
        raise ValueError("original EVAL eligibility embedded fingerprint mismatch")
    if probe.get("manifest_sha256") != fingerprint(probe):
        raise ValueError("probe embedded fingerprint mismatch")
    sha_fields_checked = sum(validate_sha256_fields(obj, label) for label, obj in (
        ("producer", producer), ("v2_producer", v2),
        ("protocol", protocol), ("eligibility", eligibility), ("probe", probe),
    ))
    overlay_for_digest_validation = json.loads(json.dumps(overlay, ensure_ascii=False))
    del overlay_for_digest_validation["records"][0]["source_identity_sha256"]
    sha_fields_checked += validate_sha256_fields(overlay_for_digest_validation, "overlay")

    if overlay.get("mfa_producer_manifest_raw_sha256") != producer_raw_sha:
        raise ValueError("overlay original-producer raw SHA binding mismatch")
    if overlay.get("mfa_producer_manifest_sha256") != producer.get("manifest_sha256"):
        raise ValueError("overlay original-producer fingerprint binding mismatch")
    if overlay.get("mfa_protocol_manifest_sha256") != protocol.get("manifest_sha256"):
        raise ValueError("overlay protocol binding mismatch")
    if overlay.get("original_evaluation_eligibility_raw_sha256") != eligibility_raw_sha:
        raise ValueError("overlay eligibility raw SHA binding mismatch")
    if overlay.get("probe_manifest_raw_sha256") != probe_raw_sha or overlay.get("probe_manifest_fingerprint") != probe.get("manifest_sha256"):
        raise ValueError("overlay probe binding mismatch")

    old_records = producer.get("records", {})
    v2_records = v2.get("records", {})
    cal_ids = [key for key, row in old_records.items() if row.get("split") == "calibration"]
    eval_ids = [key for key, row in old_records.items() if row.get("split") == "evaluation"]
    if len(cal_ids) != 20 or len(eval_ids) != 80:
        raise ValueError("expected the frozen 20-CAL/80-EVAL producer records")
    if set(old_records) != set(v2_records) or any(old_records[sid] != v2_records.get(sid) for sid in cal_ids + eval_ids):
        raise ValueError("old/v2 producer row objects differ; refusing to rebind")

    rows = overlay.get("records", [])
    if len(rows) != 1 or rows[0].get("id") != SAMPLE_ID:
        raise ValueError("expected exactly the frozen S0064 physical-only row")
    row = rows[0]
    rec = old_records[SAMPLE_ID]
    producer_identity = rec.get("source_identity_sha256")
    if not isinstance(producer_identity, str) or len(producer_identity) != 64:
        raise ValueError("original EVAL producer identity is missing or malformed")
    if producer_identity == row["source_identity_sha256"]:
        raise ValueError("source identity already matches; no correction is needed")

    eligibility_rows = {str(item["id"]): item for item in eligibility.get("records", [])}
    qrow = eligibility_rows.get(SAMPLE_ID)
    if not qrow or qrow.get("pair_qualified") is not False:
        raise ValueError("original EVAL eligibility row is not the frozen unqualified pair")

    transcript = str(row["publisher_transcript"])
    transcript_sha = sha256_bytes(transcript.encode("utf-8"))
    if transcript != probe.get("publisher_transcript"):
        raise ValueError("publisher transcript differs from the frozen repair probe")
    for condition in ("natural", "tts"):
        old_meta = rec["conditions"][condition]
        overlay_meta = row["conditions"][condition]
        qcondition = qrow.get(condition, {})
        probe_condition = probe.get("conditions", {}).get(condition, {})
        if overlay_meta.get("source_audio_sha256") != old_meta.get("source_audio_sha256"):
            raise ValueError(f"{condition} source-audio SHA differs from original EVAL producer")
        authoritative_audio_sha = rec.get("natural_raw_audio_sha256") if condition == "natural" else rec.get("canonical_tts_16k_audio_sha256")
        if old_meta.get("source_audio_sha256") != authoritative_audio_sha:
            raise ValueError(f"{condition} source-audio SHA differs from the sealed paired input")
        if old_meta.get("transcript_sha256") != transcript_sha or rec.get(f"{condition}_transcript_sha256") != transcript_sha:
            raise ValueError(f"{condition} transcript SHA differs from original paired input")
        if overlay_meta.get("source_audio_sha256") != probe_condition.get("source_audio_sha256"):
            raise ValueError(f"{condition} probe source-audio SHA differs from the original paired input")
        if overlay_meta.get("original_textgrid_sha256") != qcondition.get("textgrid_sha256"):
            raise ValueError(f"{condition} original TextGrid SHA differs from frozen eligibility")
        if overlay_meta.get("physical_textgrid_sha256") != probe.get("alignment_rows", {}).get(condition, {}).get("textgrid_sha256"):
            raise ValueError(f"{condition} physical TextGrid SHA differs from frozen probe")
        if overlay_meta.get("physical_lab_sha256") != probe_condition.get("lab_sha256"):
            raise ValueError(f"{condition} physical LAB SHA differs from frozen probe")
        grid_path = Path(overlay_meta["physical_textgrid_local_path"])
        lab_path = Path(overlay_meta["physical_lab_local_path"])
        if not grid_path.is_file() or sha256_file(grid_path) != overlay_meta["physical_textgrid_sha256"]:
            raise ValueError(f"{condition} physical TextGrid local bytes differ from the frozen hash")
        if not lab_path.is_file() or sha256_file(lab_path) != overlay_meta["physical_lab_sha256"]:
            raise ValueError(f"{condition} physical LAB local bytes differ from the frozen hash")
        if lab_path.read_text(encoding="utf-8").strip() != overlay_meta["physical_lab_text"]:
            raise ValueError(f"{condition} physical LAB text differs from its manifest")
        if overlay_meta["physical_lab_text"] != probe_condition.get("probe_lab_text"):
            raise ValueError(f"{condition} physical LAB text differs from frozen probe")
    if row.get("publisher_transcript") != probe.get("publisher_transcript"):
        raise ValueError("publisher text differs from the frozen repair probe")
    if "".join(str(rec["conditions"]["natural"]["normalized_lab_text"]).split()) != "".join(str(rec["conditions"]["tts"]["normalized_lab_text"]).split()):
        raise ValueError("original natural/TTS normalized publisher texts differ")
    if "".join(str(row["physical_lab_text"]).split()) != "".join(str(rec["conditions"]["natural"]["normalized_lab_text"]).split()):
        raise ValueError("physical LAB token join changes the original publisher text")

    corrected = json.loads(json.dumps(overlay, ensure_ascii=False))
    corrected_row = corrected["records"][0]
    identity_before = corrected_row["source_identity_sha256"]
    corrected_row["source_identity_sha256"] = producer_identity
    corrected_row["record_sha256"] = fingerprint({key: value for key, value in corrected_row.items() if key != "record_sha256"})
    corrected["manifest_sha256"] = fingerprint(corrected)
    corrected_sha_fields = validate_sha256_fields(corrected, "corrected_overlay")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corrected, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    patch = {
        "schema_version": 1,
        "manifest_type": "minimal_physical_overlay_manifest_correction",
        "changes": [
            {"json_pointer": "/records/0/source_identity_sha256", "before": identity_before, "after": producer_identity,
             "source": "original EVAL producer record source_identity_sha256, read programmatically"},
            {"json_pointer": "/records/0/record_sha256", "before": row["record_sha256"], "after": corrected_row["record_sha256"],
             "reason": "recomputed after the single source-identity metadata correction"},
            {"json_pointer": "/manifest_sha256", "before": overlay["manifest_sha256"], "after": corrected["manifest_sha256"],
             "reason": "recomputed after the single source-identity metadata correction"},
        ],
        "all_other_manifest_fields_unchanged": True,
    }
    audit = {
        "schema_version": 1,
        "manifest_type": "world_physical_alignment_source_identity_rebind_audit",
        "status": "CANDIDATE_FOR_ROOT_FREEZE",
        "original_overlay_local_path": str(args.overlay.resolve()),
        "original_overlay_raw_sha256": overlay_raw_sha,
        "original_overlay_fingerprint": overlay["manifest_sha256"],
        "new_overlay_local_path": str(args.output.resolve()),
        "new_overlay_raw_sha256": sha256_file(args.output),
        "new_overlay_fingerprint": corrected["manifest_sha256"],
        "old_eval_producer_raw_sha256": producer_raw_sha,
        "old_eval_producer_fingerprint": producer["manifest_sha256"],
        "v2_producer_raw_sha256": v2_raw_sha,
        "v2_producer_fingerprint": v2["manifest_sha256"],
        "original_eval_eligibility_raw_sha256": eligibility_raw_sha,
        "source_identity_before": row["source_identity_sha256"],
        "source_identity_after": producer_identity,
        "source_identity_before_length": len(identity_before),
        "source_identity_after_length": len(producer_identity),
        "source_identity_copied_programmatically_from_original_eval_record": True,
        "old_and_v2_records_all_identical": True,
        "old_cal_rows": len(cal_ids),
        "old_eval_rows": len(eval_ids),
        "sha256_fields_checked": sha_fields_checked,
        "corrected_overlay_sha256_fields_checked": corrected_sha_fields,
        "row_hash_before": row["record_sha256"],
        "row_hash_after": corrected_row["record_sha256"],
        "overlay_fingerprint_after": corrected["manifest_sha256"],
        "physical_audio_text_alignment_hashes_verified": True,
        "minimal_manifest_diff": patch["changes"],
        "evaluation_waveforms_read": False,
        "evaluation_effects_read": False,
    }
    audit["manifest_sha256"] = fingerprint(audit)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    patch_path = args.audit_output.with_name("minimal_manifest_diff.json")
    if patch_path.exists():
        raise FileExistsError("refusing to overwrite minimal manifest diff")
    patch["manifest_sha256"] = fingerprint(patch)
    patch_path.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "old_overlay_raw_sha256": overlay_raw_sha,
        "new_overlay_raw_sha256": audit["new_overlay_raw_sha256"],
        "new_overlay_fingerprint": corrected["manifest_sha256"],
        "source_identity_before": row["source_identity_sha256"],
        "source_identity_after": producer_identity,
        "old_eval_producer_raw_sha256": producer_raw_sha,
        "v2_producer_raw_sha256": v2_raw_sha,
        "rows": {"CAL": len(cal_ids), "EVAL": len(eval_ids)},
        "evaluation_waveforms_read": False,
        "sha256_fields_checked": sha_fields_checked,
        "audit_path": str(args.audit_output.resolve()),
        "minimal_diff_path": str(patch_path.resolve()),
    }, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
