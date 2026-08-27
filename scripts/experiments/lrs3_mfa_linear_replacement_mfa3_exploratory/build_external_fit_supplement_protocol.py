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
    SAMPLE_RATE,
    canonical_sha256,
    dictionary_words,
    file_sha256,
    load_json,
    normalization_events,
    normalize_mfa3_transcript,
    transcript_sha256,
    write_json,
)

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE_ID = "00_external_fit_supplement_protocol_retry4"
NEXT_STAGE_ID = "01_external_fit_mfa3_screen_retry4"
EXPECTED_CURRENT_PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
EXPECTED_CURRENT_STAGE_ID = "00_protocol_lock_expanded_retry3"
EXPECTED_VISUAL_PROTOCOL_ID = "lrs3_tts_visual_control_20260825"
EXPECTED_VISUAL_STAGE_ID = "00_protocol_lock_retry2"
EXPECTED_VISUAL_SPLIT_POLICY_ID = "pre_score_common_support_23_v1"
EXPECTED_NORMALIZATION_REJECTED_IDS = {
    "lrs3_73rUjrow5pI_00006",
    "lrs3_6xtmm0MnaS0_00012",
}
CURRENT_STAGE00 = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / "00_protocol_lock_expanded_retry3"
    / "manifest.json"
)
SOURCE_MANIFEST = REPO / "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json"
TTS_META = REPO / "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json"
EXPECTED_OUTPUT_DIR = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
    / STAGE_ID
)


def _asset(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _assert_bindings_unchanged(bindings: Mapping[str, Mapping[str, str]]) -> None:
    for label, binding in bindings.items():
        path = Path(str(binding["path"])).resolve()
        if not path.is_file() or file_sha256(path) != str(binding["sha256"]):
            raise RuntimeError(f"bound input changed while building external-fit Stage00: {label}")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def _row_from_source(source: Mapping[str, Any], tts: Mapping[str, Any], group: str) -> dict[str, Any]:
    sample_id = str(source["sample_id"])
    if str(tts.get("sample_id")) != sample_id:
        raise ValueError(f"TTS sample ID mismatch: {sample_id}")
    if str(source.get("source_group")) != group or str(tts.get("source_group")) != group:
        raise ValueError(f"source/TTS group mismatch: {sample_id}")
    if source.get("status") != "ready" or tts.get("status") != "ok":
        raise ValueError(f"source/TTS status mismatch: {sample_id}")
    transcript = str(source["transcript"])
    tts_transcript = str(tts["tts_transcript"])
    normalized = normalize_mfa3_transcript(transcript)
    normalized_tts = normalize_mfa3_transcript(tts_transcript)
    if normalized != normalized_tts:
        raise ValueError(f"natural/TTS normalized transcript mismatch: {sample_id}")
    natural_audio = _repo_path(str(source["natural_audio_path"]))
    tts_audio = _repo_path(str(tts["canonical_16k_audio"]))
    face_video = _repo_path(str(source["video_local_path"]))
    reference_audio = _repo_path(str(tts["reference_audio"]))
    if reference_audio != natural_audio or str(tts.get("reference_audio_sha256")) != str(source["natural_audio_sha256"]):
        raise ValueError(f"TTS reference audio binding mismatch: {sample_id}")
    if str(tts.get("video_sha256")) != str(source["video_sha256"]):
        raise ValueError(f"TTS/source video binding mismatch: {sample_id}")
    if (
        int(source.get("natural_audio_sample_rate_hz", -1)) != SAMPLE_RATE
        or int(source.get("natural_audio_channels", -1)) != 1
        or int(tts.get("canonical_sample_rate_hz", -1)) != SAMPLE_RATE
    ):
        raise ValueError(f"source/TTS canonical audio format mismatch: {sample_id}")
    if not natural_audio.is_file() or file_sha256(natural_audio) != str(source["natural_audio_sha256"]):
        raise ValueError(f"natural audio missing or hash mismatch: {sample_id}")
    if not tts_audio.is_file() or file_sha256(tts_audio) != str(tts["canonical_audio_sha256"]):
        raise ValueError(f"TTS audio missing or hash mismatch: {sample_id}")
    if not face_video.is_file() or file_sha256(face_video) != str(source["video_sha256"]):
        raise ValueError(f"source video missing or hash mismatch: {sample_id}")
    return {
        "sample_id": sample_id,
        "source_group": group,
        "protocol_split": "fit",
        "source_status": str(source.get("status", "ready")),
        "source_video": str(face_video),
        "source_video_sha256": str(source["video_sha256"]),
        "natural_audio": str(natural_audio),
        "natural_audio_sha256": str(source["natural_audio_sha256"]),
        "natural_audio_samples": int(source["natural_audio_samples"]),
        "natural_duration_s": float(source["natural_audio_duration_s"]),
        "tts_audio": str(tts_audio),
        "tts_audio_sha256": str(tts["canonical_audio_sha256"]),
        "tts_audio_samples": int(tts["canonical_samples"]),
        "tts_duration_s": float(tts["canonical_duration_s"]),
        "duration_ratio": float(tts["duration_ratio"]),
        "tts_status": str(tts.get("status", "ok")),
        "transcript": transcript,
        "tts_transcript": tts_transcript,
        "raw_transcript_sha256": transcript_sha256(transcript),
        "raw_tts_transcript_sha256": transcript_sha256(tts_transcript),
        "transcript_sha256": transcript_sha256(normalized),
        "tts_transcript_sha256": transcript_sha256(normalized_tts),
        "normalized_transcript": normalized,
        "normalized_tts_transcript": normalized_tts,
        "normalization_events": normalization_events(transcript),
        "transcript_confidence": float(source.get("transcript_confidence", 0.0)),
        "word_count": int(source.get("word_count", 0)),
        "video_fps": float(source["video_fps"]),
        "video_width": int(source["video_width"]),
        "video_height": int(source["video_height"]),
    }


def build_manifest() -> dict[str, Any]:
    test_lock_path = CURRENT_STAGE00.parent / "test_lock.json"
    input_bindings: dict[str, dict[str, str]] = {
        "current_stage00": _asset(CURRENT_STAGE00),
        "source_manifest": _asset(SOURCE_MANIFEST),
        "tts_meta": _asset(TTS_META),
        "current_test_lock": _asset(test_lock_path),
        "mfa_executable": _asset(MFA_EXECUTABLE),
        "mfa_root_config": _asset(MFA_ROOT_CONFIG),
        "mfa_dictionary": _asset(MFA_DICTIONARY_PATH),
        "mfa_acoustic_model": _asset(MFA_ACOUSTIC_MODEL_PATH),
    }
    current = load_json(CURRENT_STAGE00)
    source = load_json(SOURCE_MANIFEST)
    tts_meta = load_json(TTS_META)
    current_test_lock_file = load_json(test_lock_path)
    current_companion = CURRENT_STAGE00.parent / "manifest.sha256"
    if current_companion.read_text(encoding="utf-8").strip() != file_sha256(CURRENT_STAGE00):
        raise ValueError("current expanded Stage00 companion hash mismatch")
    if (
        current.get("manifest_type") != "lrs3_mfa3_expanded_candidate_protocol_lock"
        or current.get("protocol_id") != EXPECTED_CURRENT_PROTOCOL_ID
        or current.get("stage_id") != EXPECTED_CURRENT_STAGE_ID
        or current.get("status") != "complete"
        or current.get("engineering_decision") != "GO"
        or current.get("scientific_decision") != "not_available"
    ):
        raise ValueError("current expanded Stage00 identity is not the frozen complete engineering GO")
    current_parents = current.get("parents")
    if not isinstance(current_parents, Mapping):
        raise ValueError("current expanded Stage00 parents are missing")
    parent_test_lock = current_parents.get("parent_test_lock")
    if not isinstance(parent_test_lock, Mapping):
        raise ValueError("current expanded Stage00 parent test lock binding is missing")
    parent_test_lock_path = _repo_path(str(parent_test_lock["path"]))
    if (
        not parent_test_lock_path.is_file()
        or file_sha256(parent_test_lock_path) != str(parent_test_lock["sha256"])
    ):
        raise ValueError("current expanded Stage00 parent test lock hash mismatch")
    parent_test_lock_data = load_json(parent_test_lock_path)
    current_test_groups = {str(value) for value in current_test_lock_file.get("test_groups", [])}
    if (
        parent_test_lock_data.get("status") != "sealed_unvisited"
        or parent_test_lock_data.get("test_group_count") != 7
        or {str(value) for value in parent_test_lock_data.get("test_groups", [])} != current_test_groups
        or parent_test_lock_data.get("media_opened") is not False
        or parent_test_lock_data.get("derived_features_created") is not False
        or parent_test_lock_data.get("scores_created") is not False
    ):
        raise ValueError("current expanded Stage00 parent test lock is not the matching sealed lock")
    input_bindings["current_parent_test_lock"] = {
        "path": str(parent_test_lock_path),
        "sha256": str(parent_test_lock["sha256"]),
    }
    visual_parent = current_parents.get("current_visual_stage00")
    if not isinstance(visual_parent, Mapping):
        raise ValueError("current visual Stage00 parent is missing")
    visual_stage00_path = _repo_path(str(visual_parent["path"]))
    if file_sha256(visual_stage00_path) != str(visual_parent["sha256"]):
        raise ValueError("current visual Stage00 parent hash mismatch")
    input_bindings["current_visual_stage00"] = {
        "path": str(visual_stage00_path),
        "sha256": str(visual_parent["sha256"]),
    }
    visual_stage00 = load_json(visual_stage00_path)
    if (
        visual_stage00.get("manifest_type") != "lrs3_duration_compatible_tts_visual_control_protocol"
        or visual_stage00.get("protocol_id") != EXPECTED_VISUAL_PROTOCOL_ID
        or visual_stage00.get("stage_id") != EXPECTED_VISUAL_STAGE_ID
        or visual_stage00.get("status") != "complete"
        or visual_stage00.get("engineering_decision") != "GO"
        or visual_stage00.get("scientific_decision") != "not_available"
    ):
        raise ValueError("current visual Stage00 identity is not the frozen complete engineering GO")
    split = visual_stage00.get("split")
    if not isinstance(split, Mapping) or split.get("policy_id") != EXPECTED_VISUAL_SPLIT_POLICY_ID:
        raise ValueError("current visual Stage00 split identity is missing")
    effective_groups = [str(value) for value in split.get("effective_fit_groups", [])]
    if (
        len(effective_groups) != 23
        or len(set(effective_groups)) != 23
        or split.get("excluded_fit_groups") != ["6W2dsnhC18Q"]
        or "6W2dsnhC18Q" in set(effective_groups)
        or split.get("source_group_count") != 23
    ):
        raise ValueError("unexpected effective fit-group contract")
    sealed_groups = {
        str(value)
        for key in ("internal_dev_groups", "validation_groups", "test_groups")
        for value in split.get(key, [])
    }
    if set(effective_groups) & sealed_groups:
        raise ValueError("effective fit groups intersect sealed groups")

    current_cohort = current.get("cohort")
    current_records = current_cohort.get("records") if isinstance(current_cohort, Mapping) else None
    if (
        not isinstance(current_records, list)
        or current_cohort.get("record_count") != 122
        or current_cohort.get("source_group_count") != 23
        or len(current_records) != 122
    ):
        raise ValueError("current expanded Stage00 cohort count mismatch")
    current_ids = [str(row["sample_id"]) for row in current_records]
    if len(set(current_ids)) != 122 or current_cohort.get("ordered_sample_ids_sha256") != canonical_sha256(current_ids):
        raise ValueError("current expanded Stage00 cohort order/hash mismatch")
    if any(row.get("protocol_split") != "fit" for row in current_records):
        raise ValueError("current expanded Stage00 contains a non-fit record")
    if {str(row["source_group"]) for row in current_records} != set(effective_groups):
        raise ValueError("current expanded Stage00 group roster differs from frozen visual split")

    source_records = source.get("records")
    tts_results = tts_meta.get("results")
    if (
        source.get("manifest_type") != "lrs3_qwen_cloud_tts_n500"
        or source.get("sample_count") != 500
        or not isinstance(source_records, list)
        or len(source_records) != 500
    ):
        raise ValueError("canonical n500 source inventory is incomplete")
    source_ids = [str(row["sample_id"]) for row in source_records]
    if len(set(source_ids)) != 500:
        raise ValueError("canonical n500 source IDs are not unique")
    if (
        tts_meta.get("manifest_type") != "lrs3_qwen_cloud_tts_n500"
        or tts_meta.get("sample_count") != 500
        or tts_meta.get("samples_ok") != 500
        or tts_meta.get("samples_failed") != 0
        or tts_meta.get("complete") is not True
        or tts_meta.get("failures") not in ([], {})
        or not isinstance(tts_results, Mapping)
        or len(tts_results) != 500
    ):
        raise ValueError("canonical n500 TTS inventory is incomplete")
    tts_ids = [str(key) for key in tts_results]
    if set(tts_ids) != set(source_ids):
        raise ValueError("canonical source/TTS sample IDs do not match")
    source_by_id = {str(row["sample_id"]): row for row in source_records}
    tts_by_id = {str(key): value for key, value in tts_results.items()}
    if any(str(row.get("sample_id")) != sample_id for sample_id, row in tts_by_id.items()):
        raise ValueError("canonical TTS result key/sample ID mismatch")
    if tts_meta.get("source_manifest_sha256") != file_sha256(SOURCE_MANIFEST):
        raise ValueError("TTS metadata is not bound to canonical source manifest")

    words = dictionary_words(MFA_DICTIONARY_PATH)
    records: list[dict[str, Any]] = []
    normalization_rejections: list[dict[str, Any]] = []
    input_source_ids: list[str] = []
    for source_row in source.get("records", []):
        sample_id = str(source_row["sample_id"])
        group = str(source_row["source_group"])
        if group not in effective_groups:
            continue
        input_source_ids.append(sample_id)
        tts_row = tts_by_id.get(sample_id)
        if not isinstance(tts_row, Mapping):
            raise ValueError(f"canonical TTS row missing: {sample_id}")
        if (
            str(tts_row.get("sample_id")) != sample_id
            or str(tts_row.get("source_group")) != group
            or str(tts_row.get("reference_audio_sha256")) != str(source_row["natural_audio_sha256"])
            or str(tts_row.get("video_sha256")) != str(source_row["video_sha256"])
        ):
            raise ValueError(f"canonical source/TTS row binding mismatch: {sample_id}")
        rejection_errors: list[dict[str, str]] = []
        for side, text in (
            ("natural", str(source_row["transcript"])),
            ("tts", str(tts_row["tts_transcript"])),
        ):
            try:
                normalize_mfa3_transcript(text)
            except ValueError as exc:
                rejection_errors.append({"side": side, "error": str(exc)})
        if rejection_errors:
            normalization_rejections.append({
                "sample_id": sample_id,
                "source_group": group,
                "reason": "; ".join(f"{row['side']}: {row['error']}" for row in rejection_errors),
                "errors": rejection_errors,
            })
            continue
        records.append(_row_from_source(source_row, tts_row, group))

    if len(input_source_ids) != 268 or len(set(input_source_ids)) != 268:
        raise ValueError("unexpected 23-group source inventory size")
    if len(records) != 266:
        raise ValueError(f"unexpected normalizable external-fit roster size: {len(records)}")
    if len(normalization_rejections) != 2:
        raise ValueError(f"unexpected normalization rejection count: {len(normalization_rejections)}")
    if {str(row["sample_id"]) for row in normalization_rejections} != EXPECTED_NORMALIZATION_REJECTED_IDS:
        raise ValueError("normalization rejection IDs differ from the frozen {NS} failures")
    for rejection in normalization_rejections:
        errors = rejection.get("errors")
        if (
            not isinstance(errors, list)
            or {str(row.get("side")) for row in errors} != {"natural", "tts"}
            or any("{NS}" not in str(row.get("error", "")) for row in errors)
        ):
            raise ValueError(f"normalization rejection is not the frozen two-sided {{NS}} failure: {rejection['sample_id']}")
    record_ids = [str(row["sample_id"]) for row in records]
    record_id_set = set(record_ids)
    current_id_set = set(current_ids)
    current_normalizable_ids = [sample_id for sample_id in current_ids if sample_id in record_id_set]
    current_rejected_ids = [sample_id for sample_id in current_ids if sample_id not in record_id_set]
    if current_normalizable_ids != [sample_id for sample_id in record_ids if sample_id in current_id_set]:
        raise ValueError("normalizable current Stage01 roster is not an ordered subset of expanded roster")
    if len(current_normalizable_ids) != 120 or len(current_rejected_ids) != 2:
        raise ValueError("unexpected current-roster normalization accounting")
    if set(current_rejected_ids) != {row["sample_id"] for row in normalization_rejections}:
        raise ValueError("normalization rejections do not match current-roster rejected records")
    external_ids = [sample_id for sample_id in record_ids if sample_id not in current_id_set]
    if len(external_ids) != 146 or set(external_ids) & current_id_set:
        raise ValueError("external supplement roster count or overlap mismatch")
    if len({row["source_group"] for row in records}) != 23:
        raise ValueError("external-fit roster lost group coverage")
    if any(row["source_group"] in sealed_groups for row in records):
        raise ValueError("external-fit roster crossed sealed group boundary")
    if any(token not in words for row in records for token in row["normalized_transcript"].split()):
        dictionary_unknown_count = sum(
            1 for row in records for token in row["normalized_transcript"].split() if token not in words
        )
    else:
        dictionary_unknown_count = 0

    split_sets = {
        key: {str(value) for value in split.get(key, [])}
        for key in ("effective_fit_groups", "internal_dev_groups", "validation_groups", "test_groups")
    }
    if [len(split_sets[key]) for key in ("internal_dev_groups", "validation_groups", "test_groups")] != [6, 6, 7]:
        raise ValueError("sealed visual split group counts are not frozen")
    split_values = list(split_sets.values())
    if any(split_values[i] & split_values[j] for i in range(len(split_values)) for j in range(i + 1, len(split_values))):
        raise ValueError("visual split groups overlap")

    current_selection = current.get("selection")
    if not isinstance(current_selection, Mapping):
        raise ValueError("current Stage00 selection ledger is missing")
    expected_current_selection = {
        "score_based_selection": False,
        "visual_metric_used_for_selection": False,
        "syncnet_used_for_selection": False,
        "alignment_result_based_selection": False,
        "current_mfa3_alignment_used_for_selection": False,
        "fit_groups_only": True,
        "sealed_groups_excluded": True,
        "source_manifest_order": True,
        "inventory_roster_frozen": True,
        "record_count_fixed_before_alignment": True,
    }
    for key, expected in expected_current_selection.items():
        if current_selection.get(key) is not expected:
            raise ValueError(f"current Stage00 selection ledger mismatch: {key}")
    visual_selection = visual_stage00.get("selection")
    expected_visual_selection = {
        "candidate_parent_pre_score": True,
        "face_preflight_structural": True,
        "syncnet_preflight_structural_only": True,
        "score_based_selection": False,
        "source_order_preserved": True,
        "sample_substitution_allowed": False,
    }
    if not isinstance(visual_selection, Mapping):
        raise ValueError("visual Stage00 selection ledger is missing")
    for key, expected in expected_visual_selection.items():
        if visual_selection.get(key) is not expected:
            raise ValueError(f"visual Stage00 selection ledger mismatch: {key}")
    for label, ledger in (("current", current_selection), ("visual", visual_selection)):
        for key, value in ledger.items():
            if (
                key in {"score_based_selection", "alignment_result_based_selection"}
                or key.endswith("_used_for_selection")
            ) and value is not False:
                raise ValueError(f"{label} Stage00 records result-based selection: {key}")

    test_lock_file = current_test_lock_file
    embedded_test_lock = current.get("test_lock")
    if not isinstance(embedded_test_lock, Mapping) or dict(embedded_test_lock) != test_lock_file:
        raise ValueError("current embedded test lock differs from test_lock.json")
    test_lock = dict(embedded_test_lock)
    if test_lock.get("status") != "sealed_unvisited" or set(test_lock.get("test_groups", [])) != split_sets["test_groups"]:
        raise ValueError("current test lock identity/group contract mismatch")
    for key, value in test_lock.items():
        if key.endswith(("_media_opened", "_derived_features_created", "_scores_created")) and value is not False:
            raise ValueError(f"current test lock records access: {key}")
    current_media = current.get("media_access")
    visual_media = visual_stage00.get("media_access")
    if not isinstance(current_media, Mapping) or not isinstance(visual_media, Mapping):
        raise ValueError("parent media access ledger is missing")
    for label, ledger in (("current", current_media), ("visual", visual_media)):
        if any(value is not False for value in ledger.values()):
            raise ValueError(f"{label} parent media access ledger is not all false")
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_external_fit_supplement_protocol_lock",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": NEXT_STAGE_ID,
        "parents": {
            "current_stage00": dict(input_bindings["current_stage00"]),
            "current_visual_stage00": dict(input_bindings["current_visual_stage00"]),
            "current_parent_test_lock": dict(input_bindings["current_parent_test_lock"]),
            "source_manifest": dict(input_bindings["source_manifest"]),
            "tts_meta": dict(input_bindings["tts_meta"]),
            "current_test_lock": dict(input_bindings["current_test_lock"]),
        },
        "input_bindings": {key: dict(value) for key, value in input_bindings.items()},
        "split": {
            **{key: list(value) for key, value in split.items() if isinstance(value, list)},
            "effective_fit_groups": effective_groups,
            "parent_policy_id": EXPECTED_VISUAL_SPLIT_POLICY_ID,
            "unit": "source_group_directory_not_verified_speaker_id",
            "selection_basis": "all canonical source-order records in pre-registered effective fit groups",
            "score_based_substitution": False,
        },
        "cohort": {
            "record_count": len(records),
            "source_group_count": len(effective_groups),
            "ordered_sample_ids_sha256": canonical_sha256(record_ids),
            "records": records,
        },
        "supplement": {
            "current_stage00_source_record_count": len(current_ids),
            "current_stage00_normalizable_record_count": len(current_normalizable_ids),
            "current_stage00_normalization_rejected_count": len(current_rejected_ids),
            "current_stage00_normalization_rejected_sample_ids": current_rejected_ids,
            "external_record_count": len(external_ids),
            "external_sample_ids": external_ids,
            "external_sample_ids_sha256": canonical_sha256(external_ids),
            "selection_rule": "all canonical source-order records not present in current Stage01 Stage00 roster",
            "score_based_selection": False,
            "visual_metric_used_for_selection": False,
            "syncnet_used_for_selection": False,
            "audio_quality_used_for_selection": False,
            "alignment_result_used_for_selection": False,
            "visual_parent_audio_quality_selection_audit": {
                "parent_field_present": "audio_quality_used_for_selection" in visual_selection,
                "parent_field_value": visual_selection.get("audio_quality_used_for_selection"),
                "accepted_structural_equivalence": True,
                "basis": "parent pre-score structural ledger with score_based_selection=false, source_order_preserved=true, and sample_substitution_allowed=false",
            },
        },
        "input_screen": {
            "source_inventory_record_count": len(input_source_ids),
            "normalization_rejected_count": len(normalization_rejections),
            "normalization_rejections": normalization_rejections,
            "dictionary_prefilter_used": False,
            "mfa_failures_remain_in_screen_output": True,
        },
        "mfa_contract": {
            "version": MFA_VERSION,
            "root_dir": str(MFA_ROOT_DIR.resolve()),
            "executable": dict(input_bindings["mfa_executable"]),
            "root_config": dict(input_bindings["mfa_root_config"]),
            "dictionary": {"name": MFA_DICTIONARY, **input_bindings["mfa_dictionary"]},
            "acoustic_model": {"name": MFA_ACOUSTIC_MODEL, **input_bindings["mfa_acoustic_model"]},
        },
        "candidate_contract": dict(current.get("candidate_contract", {})),
        "assets": dict(current.get("assets", {})),
        "visual_contract": {
            "valid_fraction_threshold": 0.90,
            "exact_time_coverage_threshold": 0.85,
            "minimum_eligible_ratio": {"numerator": 44, "denominator": 48},
            "primary_clock": "natural_video_exact_time",
            "score_based_selection": False,
        },
        "test_lock": test_lock,
        "media_access": {
            "fit_media_opened_for_hash_verification": True,
            "fit_media_decoded": False,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
            "mfa_run": False,
            "features_created": False,
            "scores_created": False,
        },
        "selection": {
            "source_manifest_order": True,
            "fit_groups_only": True,
            "sealed_groups_excluded": True,
            "current_stage00_source_records_accounted_for": True,
            "all_current_stage00_normalizable_records_retained": True,
            "current_stage00_normalization_failures_explicitly_rejected": True,
            "external_records_added": True,
            "normalization_failures_not_silently_included": True,
            "score_based_selection": False,
            "visual_metric_used_for_selection": False,
            "syncnet_used_for_selection": False,
            "audio_quality_used_for_selection": False,
            "parent_selection_ledgers_validated": True,
        },
        "diagnostics": {
            "dictionary_unknown_token_count_not_used_for_selection": dictionary_unknown_count,
        },
    }
    _assert_bindings_unchanged(input_bindings)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != EXPECTED_OUTPUT_DIR.resolve():
        raise ValueError(f"external-fit Stage00 output must be canonical: {EXPECTED_OUTPUT_DIR}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output}")
    manifest = build_manifest()
    _assert_bindings_unchanged(manifest["input_bindings"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "source_inventory_record_count": manifest["input_screen"]["source_inventory_record_count"],
        "normalization_rejected_count": manifest["input_screen"]["normalization_rejected_count"],
        "record_count": manifest["cohort"]["record_count"],
        "current_stage00_source_record_count": manifest["supplement"]["current_stage00_source_record_count"],
        "current_stage00_normalizable_record_count": manifest["supplement"]["current_stage00_normalizable_record_count"],
        "external_record_count": manifest["supplement"]["external_record_count"],
        "source_group_count": manifest["cohort"]["source_group_count"],
        "mfa_run": False,
        "sealed_media_opened": False,
    })
    write_json(output / "decision.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": manifest["next_allowed_stage"],
        "reason": "all normalizable canonical records from the pre-registered effective fit groups were frozen before MFA, rendering, or visual scoring",
    })
    (output / "manifest.sha256").write_text(file_sha256(output / "manifest.json") + "\n", encoding="utf-8")
    print(json.dumps({
        "stage_id": STAGE_ID,
        "record_count": manifest["cohort"]["record_count"],
        "external_record_count": manifest["supplement"]["external_record_count"],
        "source_group_count": manifest["cohort"]["source_group_count"],
        "normalization_rejected_count": manifest["input_screen"]["normalization_rejected_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
