from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    MFA_ACOUSTIC_MODEL,
    MFA_ACOUSTIC_MODEL_PATH,
    MFA_DICTIONARY,
    MFA_DICTIONARY_PATH,
    MFA_EXECUTABLE,
    MFA_ROOT_CONFIG,
    MFA_ROOT_DIR,
    MFA_VERSION,
    canonical_sha256,
    file_sha256,
    load_json,
    parse_alignment_pair,
    prepare_corpus,
    run_mfa3_alignment,
    write_json,
)

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE_ID = "01_external_fit_mfa3_screen_retry4"
EXPECTED_STAGE00_ID = "00_external_fit_supplement_protocol_retry4"
EXPECTED_INPUT_COUNT = 266
EXPECTED_EXTERNAL_COUNT = 146
EXPECTED_GROUP_COUNT = 23
PREVIOUS_MINIMUM_ELIGIBLE_COUNT = 57
STAGE00_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
    / EXPECTED_STAGE00_ID
    / "manifest.json"
)
EXPECTED_OUTPUT_DIR = STAGE00_PATH.parents[1] / STAGE_ID


def _binding(binding: Mapping[str, Any], path: Path, label: str) -> None:
    resolved = path.resolve()
    if Path(str(binding.get("path", ""))).resolve() != resolved:
        raise ValueError(f"{label} path mismatch")
    if not resolved.is_file() or file_sha256(resolved) != str(binding.get("sha256", "")):
        raise ValueError(f"{label} hash mismatch")


def _validate_stage00(path: Path) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if path.resolve() != STAGE00_PATH.resolve():
        raise ValueError("external-fit MFA3 screen requires the frozen supplement Stage00 path")
    stage00 = load_json(path)
    if (
        stage00.get("protocol_id") != PROTOCOL_ID
        or stage00.get("stage_id") != EXPECTED_STAGE00_ID
        or stage00.get("status") != "complete"
        or stage00.get("engineering_decision") != "GO"
        or stage00.get("scientific_decision") != "not_available"
        or stage00.get("next_allowed_stage") != STAGE_ID
    ):
        raise ValueError("external-fit Stage00 is not a complete engineering GO")
    companion = path.parent / "manifest.sha256"
    if companion.read_text(encoding="utf-8").strip() != file_sha256(path):
        raise ValueError("external-fit Stage00 companion hash mismatch")

    cohort = stage00.get("cohort")
    records = cohort.get("records") if isinstance(cohort, Mapping) else None
    if (
        not isinstance(records, list)
        or len(records) != EXPECTED_INPUT_COUNT
        or cohort.get("record_count") != EXPECTED_INPUT_COUNT
        or cohort.get("source_group_count") != EXPECTED_GROUP_COUNT
    ):
        raise ValueError("external-fit Stage00 cohort count mismatch")
    sample_ids = [str(row.get("sample_id")) for row in records]
    if len(set(sample_ids)) != EXPECTED_INPUT_COUNT:
        raise ValueError("external-fit Stage00 sample IDs are not unique")
    if cohort.get("ordered_sample_ids_sha256") != canonical_sha256(sample_ids):
        raise ValueError("external-fit Stage00 ordered cohort hash mismatch")
    if any(row.get("protocol_split") != "fit" for row in records):
        raise ValueError("external-fit Stage00 contains a non-fit record")

    supplement = stage00.get("supplement")
    if (
        not isinstance(supplement, Mapping)
        or supplement.get("external_record_count") != EXPECTED_EXTERNAL_COUNT
        or supplement.get("current_stage00_normalizable_record_count") != 120
        or supplement.get("current_stage00_normalization_rejected_count") != 2
    ):
        raise ValueError("external-fit Stage00 supplement accounting mismatch")
    external_ids = [str(value) for value in supplement.get("external_sample_ids", [])]
    if len(external_ids) != EXPECTED_EXTERNAL_COUNT or len(set(external_ids)) != EXPECTED_EXTERNAL_COUNT:
        raise ValueError("external-fit supplement IDs are incomplete")
    if supplement.get("external_sample_ids_sha256") != canonical_sha256(external_ids):
        raise ValueError("external-fit supplement ID hash mismatch")
    if not set(external_ids).issubset(set(sample_ids)):
        raise ValueError("external-fit supplement contains a non-cohort sample")

    split = stage00.get("split")
    if not isinstance(split, Mapping) or split.get("parent_policy_id") != "pre_score_common_support_23_v1":
        raise ValueError("external-fit Stage00 split policy mismatch")
    effective_groups = {str(value) for value in split.get("effective_fit_groups", [])}
    sealed_sets = {
        key: {str(value) for value in split.get(key, [])}
        for key in ("internal_dev_groups", "validation_groups", "test_groups")
    }
    sealed_groups = set().union(*sealed_sets.values())
    record_groups = {str(row.get("source_group")) for row in records}
    if (
        len(effective_groups) != EXPECTED_GROUP_COUNT
        or record_groups != effective_groups
        or split.get("excluded_fit_groups") != ["6W2dsnhC18Q"]
        or "6W2dsnhC18Q" in effective_groups
        or [len(sealed_sets[key]) for key in ("internal_dev_groups", "validation_groups", "test_groups")] != [6, 6, 7]
    ):
        raise ValueError("external-fit Stage00 group coverage/partition mismatch")
    partitions = [effective_groups, *sealed_sets.values()]
    if any(partitions[i] & partitions[j] for i in range(len(partitions)) for j in range(i + 1, len(partitions))):
        raise ValueError("external-fit Stage00 split groups overlap")

    test_lock = stage00.get("test_lock")
    if not isinstance(test_lock, Mapping) or test_lock.get("status") != "sealed_unvisited":
        raise ValueError("external-fit Stage00 test lock is not sealed")
    for key, value in test_lock.items():
        if key.endswith(("_media_opened", "_derived_features_created", "_scores_created")) and value is not False:
            raise ValueError(f"external-fit Stage00 records forbidden access: {key}")
    lock_binding = stage00.get("parents", {}).get("current_test_lock", {})
    if not isinstance(lock_binding, Mapping):
        raise ValueError("external-fit Stage00 parent test lock binding is missing")
    lock_path = Path(str(lock_binding.get("path", ""))).resolve()
    _binding(lock_binding, lock_path, "external-fit parent test lock")
    if load_json(lock_path) != dict(test_lock):
        raise ValueError("external-fit Stage00 embedded test lock differs from bound parent")
    media_access = stage00.get("media_access")
    if not isinstance(media_access, Mapping):
        raise ValueError("external-fit Stage00 media access ledger is missing")
    if media_access.get("fit_media_opened_for_hash_verification") is not True:
        raise ValueError("external-fit Stage00 fit-media hash verification is not recorded")
    for key in (
        "fit_media_decoded",
        "internal_dev_media_opened",
        "validation_media_opened",
        "test_media_opened",
        "mfa_run",
        "features_created",
        "scores_created",
    ):
        if media_access.get(key) is not False:
            raise ValueError(f"external-fit Stage00 media access ledger is unsafe: {key}")
    selection = stage00.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("external-fit Stage00 selection ledger is missing")
    expected_selection = {
        "score_based_selection": False,
        "visual_metric_used_for_selection": False,
        "syncnet_used_for_selection": False,
        "audio_quality_used_for_selection": False,
        "fit_groups_only": True,
        "sealed_groups_excluded": True,
        "source_manifest_order": True,
        "parent_selection_ledgers_validated": True,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) is not expected:
            raise ValueError(f"external-fit Stage00 selection ledger mismatch: {key}")

    contract = stage00.get("mfa_contract")
    if not isinstance(contract, Mapping) or contract.get("version") != MFA_VERSION:
        raise ValueError("external-fit Stage00 MFA version mismatch")
    if Path(str(contract.get("root_dir", ""))).resolve() != MFA_ROOT_DIR.resolve():
        raise ValueError("external-fit Stage00 MFA root directory mismatch")
    if contract.get("dictionary", {}).get("name") != MFA_DICTIONARY:
        raise ValueError("external-fit Stage00 dictionary name mismatch")
    if contract.get("acoustic_model", {}).get("name") != MFA_ACOUSTIC_MODEL:
        raise ValueError("external-fit Stage00 acoustic model name mismatch")
    for key, expected in (
        ("executable", MFA_EXECUTABLE),
        ("root_config", MFA_ROOT_CONFIG),
        ("dictionary", MFA_DICTIONARY_PATH),
        ("acoustic_model", MFA_ACOUSTIC_MODEL_PATH),
    ):
        value = contract.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"external-fit Stage00 MFA binding missing: {key}")
        _binding(value, expected, f"MFA {key}")
    return stage00, records


def _input_hashes(stage00_path: Path) -> dict[str, dict[str, str]]:
    paths = {
        "stage00": stage00_path,
        "stage00_companion": stage00_path.parent / "manifest.sha256",
        "mfa_executable": MFA_EXECUTABLE,
        "mfa_root_config": MFA_ROOT_CONFIG,
        "mfa_dictionary": MFA_DICTIONARY_PATH,
        "mfa_acoustic_model": MFA_ACOUSTIC_MODEL_PATH,
    }
    return {
        key: {"path": str(path.resolve()), "sha256": file_sha256(path.resolve())}
        for key, path in paths.items()
    }


def _assert_inputs_unchanged(bindings: Mapping[str, Mapping[str, str]]) -> None:
    for label, binding in bindings.items():
        path = Path(str(binding["path"])).resolve()
        if not path.is_file() or file_sha256(path) != str(binding["sha256"]):
            raise RuntimeError(f"bound input changed during MFA3 execution: {label}")


def _snapshot_corpus(
    records: list[Mapping[str, Any]],
    natural_input: Path,
    tts_input: Path,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        for side, directory, audio_hash_key, transcript_key in (
            ("natural", natural_input, "natural_audio_sha256", "normalized_transcript"),
            ("tts", tts_input, "tts_audio_sha256", "normalized_tts_transcript"),
        ):
            wav_path = (directory / f"{sample_id}.wav").resolve()
            lab_path = (directory / f"{sample_id}.lab").resolve()
            wav_hash = file_sha256(wav_path)
            if wav_hash != str(record[audio_hash_key]):
                raise RuntimeError(f"prepared {side} WAV differs from frozen source: {sample_id}")
            expected_lab = str(record[transcript_key]) + "\n"
            if lab_path.read_text(encoding="utf-8") != expected_lab:
                raise RuntimeError(f"prepared {side} LAB differs from frozen transcript: {sample_id}")
            entries.extend((
                {"sample_id": sample_id, "side": side, "kind": "wav", "path": str(wav_path), "sha256": wav_hash},
                {"sample_id": sample_id, "side": side, "kind": "lab", "path": str(lab_path), "sha256": file_sha256(lab_path)},
            ))
    return entries


def _assert_corpus_unchanged(entries: list[Mapping[str, str]]) -> None:
    for entry in entries:
        path = Path(str(entry["path"])).resolve()
        if not path.is_file() or file_sha256(path) != str(entry["sha256"]):
            raise RuntimeError(
                f"prepared corpus changed during MFA3 execution: {entry['sample_id']} {entry['side']} {entry['kind']}"
            )


def _write_process_failure(
    output: Path,
    stage00_path: Path,
    expected_ids: list[str],
    bindings: Mapping[str, Mapping[str, str]],
    error: Exception,
    *,
    successes: list[Mapping[str, Any]],
    failures: list[Mapping[str, Any]],
    corpus_entries: list[Mapping[str, str]],
) -> dict[str, Any]:
    observed_ids = {
        str(row["sample_id"])
        for row in [*successes, *failures]
        if row.get("sample_id") is not None
    }
    unscreened_ids = [sample_id for sample_id in expected_ids if sample_id not in observed_ids]
    failure = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_external_fit_alignment_process_failure",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "screening_complete": False,
        "ordered_input_count": len(expected_ids),
        "ordered_input_sample_ids_sha256": canonical_sha256(expected_ids),
        "observed_success_count": len(successes),
        "observed_failure_row_count": len(failures),
        "observed_successes": list(successes),
        "observed_failures": list(failures),
        "unscreened_sample_ids": unscreened_ids,
        "prepared_corpus_entry_count": len(corpus_entries),
        "prepared_corpus_entries_sha256": canonical_sha256(corpus_entries),
        "error_type": type(error).__name__,
        "error": str(error),
        "input_bindings_at_start": dict(bindings),
        "parents": {
            "stage00": {"path": str(stage00_path), "sha256": bindings["stage00"]["sha256"]},
        },
        "next_allowed_stage": None,
    }
    write_json(output / "alignment_failure.json", failure)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "input_record_count": len(expected_ids),
        "screening_complete": False,
        "clean_record_count": len(successes),
        "failure_row_count": len(failures) + 1,
        "candidate_generation_started": False,
        "gpu_used": False,
        "next_allowed_stage": None,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "next_allowed_stage": None,
        "reason": f"MFA3 screening did not complete: {type(error).__name__}: {error}",
    })
    return summary


def run(stage00_path: Path, output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if output != EXPECTED_OUTPUT_DIR.resolve():
        raise ValueError(f"external-fit MFA3 output must be canonical: {EXPECTED_OUTPUT_DIR}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty external-fit MFA3 output: {output}")
    stage00_path = stage00_path.resolve()
    input_bindings = _input_hashes(stage00_path)
    stage00, records = _validate_stage00(stage00_path)
    _assert_inputs_unchanged(input_bindings)
    output.mkdir(parents=True, exist_ok=True)
    expected_ids = [str(record["sample_id"]) for record in records]

    natural_input = output / "_mfa_input" / "natural"
    tts_input = output / "_mfa_input" / "tts"
    natural_output = output / "natural_textgrids"
    tts_output = output / "tts_textgrids"
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    corpus_entries: list[dict[str, str]] = []
    corpus_snapshot_sha256 = ""
    try:
        prepare_corpus(records, natural_input, audio_key="natural_audio", hash_key="natural_audio_sha256")
        prepare_corpus(records, tts_input, audio_key="tts_audio", hash_key="tts_audio_sha256")
        corpus_entries = _snapshot_corpus(records, natural_input, tts_input)
        write_json(output / "corpus_snapshot.json", {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "entry_count": len(corpus_entries),
            "entries_sha256": canonical_sha256(corpus_entries),
            "entries": corpus_entries,
        })
        corpus_snapshot_sha256 = file_sha256(output / "corpus_snapshot.json")

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
        _assert_corpus_unchanged(corpus_entries)
        if file_sha256(output / "corpus_snapshot.json") != corpus_snapshot_sha256:
            raise RuntimeError("prepared corpus snapshot changed during MFA3 execution")
        _assert_inputs_unchanged(input_bindings)
    except Exception as exc:
        return _write_process_failure(
            output,
            stage00_path,
            expected_ids,
            input_bindings,
            exc,
            successes=successes,
            failures=failures,
            corpus_entries=corpus_entries,
        )

    clean_ids = [str(row["sample_id"]) for row in successes]
    clean_groups = list(dict.fromkeys(str(row["source_group"]) for row in successes))
    expected_groups = list(dict.fromkeys(str(row["source_group"]) for row in records))
    missing_clean_groups = [group for group in expected_groups if group not in set(clean_groups)]
    failed_record_ids = list(dict.fromkeys(str(row["sample_id"]) for row in failures))
    unknown_rows = [row for row in failures if "spn" in str(row.get("error", "")).lower()]
    missing_grid_rows = [row for row in failures if row.get("error") == "missing TextGrid"]
    minimum_eligible_count = math.ceil(len(successes) * 44 / 48)
    engineering_go = (
        minimum_eligible_count > PREVIOUS_MINIMUM_ELIGIBLE_COUNT
        and len(clean_groups) == EXPECTED_GROUP_COUNT
    )
    reason = (
        "strict MFA3 produced enough clean fit-only records to raise the frozen Stage01 eligibility denominator with complete group coverage"
        if engineering_go
        else "strict MFA3 did not produce enough clean fit-only records and complete group coverage to improve the Stage01 eligibility gate"
    )
    next_stage = "02_external_fit_candidate_audio_retry4" if engineering_go else None

    alignment_manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_external_fit_alignment_screen",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "complete" if not failures else "partial",
        "engineering_decision": "GO" if engineering_go else "BLOCKED",
        "scientific_decision": "not_available",
        "reason": reason,
        "next_allowed_stage": next_stage,
        "parents": {
            "stage00": dict(input_bindings["stage00"]),
        },
        "input_bindings": {key: dict(value) for key, value in input_bindings.items()},
        "prepared_corpus": {
            "snapshot": {
                "path": str((output / "corpus_snapshot.json").resolve()),
                "sha256": corpus_snapshot_sha256,
            },
            "entry_count": len(corpus_entries),
            "entries_sha256": canonical_sha256(corpus_entries),
        },
        "ordered_input_count": len(records),
        "ordered_input_sample_ids_sha256": canonical_sha256(expected_ids),
        "clean_record_count": len(successes),
        "clean_sample_ids_sha256": canonical_sha256(clean_ids),
        "clean_source_group_count": len(clean_groups),
        "clean_source_groups": clean_groups,
        "missing_clean_groups": missing_clean_groups,
        "minimum_eligible_record_count": minimum_eligible_count,
        "previous_minimum_eligible_record_count": PREVIOUS_MINIMUM_ELIGIBLE_COUNT,
        "natural_textgrid_count": natural_run["textgrid_count"],
        "tts_textgrid_count": tts_run["textgrid_count"],
        "natural_mfa": natural_run,
        "tts_mfa": tts_run,
        "records": successes,
        "failures": failures,
        "failed_record_count": len(failed_record_ids),
        "failed_sample_ids": failed_record_ids,
        "unknown_phone_failures": unknown_rows,
        "missing_textgrid_failures": missing_grid_rows,
        "mfa": {
            "version": MFA_VERSION,
            "executable": str(MFA_EXECUTABLE.resolve()),
            "executable_sha256": input_bindings["mfa_executable"]["sha256"],
            "root_dir": str(MFA_ROOT_DIR.resolve()),
            "root_config": str(MFA_ROOT_CONFIG.resolve()),
            "root_config_sha256": input_bindings["mfa_root_config"]["sha256"],
            "dictionary": MFA_DICTIONARY,
            "dictionary_path": str(MFA_DICTIONARY_PATH.resolve()),
            "dictionary_sha256": input_bindings["mfa_dictionary"]["sha256"],
            "acoustic_model": MFA_ACOUSTIC_MODEL,
            "acoustic_model_path": str(MFA_ACOUSTIC_MODEL_PATH.resolve()),
            "acoustic_model_sha256": input_bindings["mfa_acoustic_model"]["sha256"],
        },
        "selection": {
            "all_stage00_records_screened": True,
            "clean_records_defined_by_strict_mfa": True,
            "alignment_result_used_for_stage00_selection": False,
            "score_based_selection": False,
            "visual_metric_used_for_selection": False,
            "syncnet_used_for_selection": False,
            "audio_quality_used_for_selection": False,
        },
        "media_access": {
            "fit_audio_opened": True,
            "fit_video_opened": False,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
            "features_created": False,
            "scores_created": False,
        },
        "test_lock": dict(stage00["test_lock"]),
    }
    try:
        _assert_corpus_unchanged(corpus_entries)
        if file_sha256(output / "corpus_snapshot.json") != corpus_snapshot_sha256:
            raise RuntimeError("prepared corpus snapshot changed before manifest finalization")
        _assert_inputs_unchanged(input_bindings)
    except Exception as exc:
        return _write_process_failure(
            output,
            stage00_path,
            expected_ids,
            input_bindings,
            exc,
            successes=successes,
            failures=failures,
            corpus_entries=corpus_entries,
        )
    write_json(output / "alignment_manifest.json", alignment_manifest)
    write_json(output / "clean_records.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "record_count": len(successes),
        "source_group_count": len(clean_groups),
        "ordered_sample_ids_sha256": canonical_sha256(clean_ids),
        "sample_ids": clean_ids,
    })
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": alignment_manifest["status"],
        "engineering_decision": alignment_manifest["engineering_decision"],
        "scientific_decision": "not_available",
        "input_record_count": len(records),
        "clean_record_count": len(successes),
        "clean_source_group_count": len(clean_groups),
        "missing_clean_groups": missing_clean_groups,
        "minimum_eligible_record_count": minimum_eligible_count,
        "failure_row_count": len(failures),
        "failed_record_count": len(failed_record_ids),
        "unknown_phone_failure_count": len(unknown_rows),
        "missing_textgrid_failure_count": len(missing_grid_rows),
        "candidate_generation_started": False,
        "gpu_used": False,
        "next_allowed_stage": next_stage,
        "media_access": alignment_manifest["media_access"],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "engineering_decision": summary["engineering_decision"],
        "scientific_decision": "not_available",
        "next_allowed_stage": next_stage,
        "reason": reason,
    })
    (output / "alignment_manifest.sha256").write_text(
        file_sha256(output / "alignment_manifest.json") + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, default=STAGE00_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.stage00.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
