#!/usr/bin/env python3
"""Lock the duration-compatible LRS3 TTS visual-control cohort."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.experiments.lrs3_tts_visual_advantage.protocol import (
    assert_finite_json,
    canonical_sha256,
    file_sha256,
    git_info,
    load_json,
    validate_test_lock,
    write_json,
)

PROTOCOL_ID = "lrs3_tts_visual_control_20260825"
STAGE_PREFIX = "00_protocol_lock_retry"
SCHEMA_VERSION = 1

VISUAL_STAGE00 = REPO / "runs/lrs3_tts_visual_advantage_20260824/00_protocol_lock_retry2/manifest.json"
VISUAL_TEST_LOCK = VISUAL_STAGE00.with_name("test_lock.json")
NATURAL_CONTROL_STAGE00 = REPO / "runs/lrs3_natural_control_20260823/00_protocol_lock_retry2/manifest.json"
CANDIDATE_MANIFEST = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/02_candidate_audio_retry4/candidate_manifest.json"
REPLACEMENT_MANIFEST = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/03_strict_replacement_face_ready_retry7/replacement_manifest.json"
RENDER_MANIFEST = REPLACEMENT_MANIFEST.with_name("render_manifest.json")
MFA_STATISTICS_DECISION = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/04_statistics_exploratory_retry2/decision.json"
DEFAULT_CODE_REVIEW = REPO / "_reviews/lrs3_tts_visual_control/00_protocol_lock/code_review.json"

CANONICAL_HASHES = {
    "visual_stage00": "89ee465f981419da48d22c119e08bae45b099479523648e3c87fca7bb0255e79",
    "visual_test_lock": "0db2a6da818a51229e106ab9cd15a8bee37c1782822e39b22758e085272d53ba",
    "natural_control_stage00": "ca10e6d98626f578d6c64da9b924980c5a909418ecdc13926ce20a99884de514",
    "candidate_manifest": "2ff6da4923e3a2abb31463014ba69d6587c5bd8bf2da257d9aff32a23fa42d6a",
    "replacement_manifest": "157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272",
    "render_manifest": "d23589e2aab11de3ddfc07b42c5f5338837e1885f018f3c9b38b5ea50b409719",
    "mfa_statistics_decision": "744648bb0a26be310d8173051bef15486e2461f061b1b4fbbf3fd2746edb1518",
}

STRICT_24 = "strict_all_24_fit_groups_v1"
COMMON_23 = "pre_score_common_support_23_v1"
DEFAULT_COHORT_POLICY = COMMON_23
MISSING_COMMON_GROUP = "6W2dsnhC18Q"


def _stage_id(output_dir: Path) -> str:
    value = output_dir.name
    if not value.startswith(STAGE_PREFIX) or not value.removeprefix(STAGE_PREFIX).isdigit():
        raise ValueError(f"unexpected Stage00 output directory name: {value}")
    return value


def _asset(path: Path, expected_sha256: str) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(f"canonical parent hash mismatch: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _unique_strings(values: Any, label: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    result = [str(value) for value in values]
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def minimum_eligible_count(record_count: int) -> int:
    if record_count <= 0:
        raise ValueError("record_count must be positive")
    return math.ceil(record_count * 44 / 48)


def validate_group_support(
    *,
    cohort_groups: Sequence[str],
    candidate_groups: Sequence[str],
    fit_groups: Sequence[str],
    fit_train_groups: Sequence[str],
    fit_selection_groups: Sequence[str],
    policy: str,
) -> dict[str, Any]:
    cohort = set(cohort_groups)
    candidate = set(candidate_groups)
    fit = set(fit_groups)
    train = set(fit_train_groups)
    selection = set(fit_selection_groups)
    if len(fit) != 24 or len(train) != 18 or len(selection) != 6:
        raise ValueError("frozen fit split must be 24=18+6 groups")
    if train | selection != fit or train & selection:
        raise ValueError("frozen fit train/selection groups are inconsistent")
    if not cohort or cohort - fit or candidate - fit:
        raise ValueError("candidate or strict cohort is outside frozen fit groups")
    if cohort != candidate:
        raise ValueError("candidate and strict cohort group support differ")

    missing = fit - cohort
    if policy == STRICT_24:
        if missing:
            raise ValueError(f"strict 24-group policy missing fit groups: {sorted(missing)}")
    elif policy == COMMON_23:
        if missing != {MISSING_COMMON_GROUP}:
            raise ValueError(f"unexpected common-support exclusion: {sorted(missing)}")
        if MISSING_COMMON_GROUP not in train:
            raise ValueError("common-support exclusion must come from fit-train")
    else:
        raise ValueError(f"unknown cohort policy: {policy}")

    effective_train = [group for group in fit_train_groups if group in cohort]
    effective_selection = [group for group in fit_selection_groups if group in cohort]
    if policy == COMMON_23 and (len(effective_train), len(effective_selection)) != (17, 6):
        raise ValueError("common-support split must preserve 17 train and 6 selection groups")
    return {
        "policy_id": policy,
        "policy_basis": (
            "pre_score structural common support between the frozen candidate and strict-render parents"
            if policy == COMMON_23
            else "all frozen fit groups must be represented by the candidate and strict-render parents"
        ),
        "parent_fit_groups": list(fit_groups),
        "parent_fit_train_groups": list(fit_train_groups),
        "parent_fit_selection_groups": list(fit_selection_groups),
        "effective_fit_groups": [group for group in fit_groups if group in cohort],
        "excluded_fit_groups": [group for group in fit_groups if group not in cohort],
        "fit_train_groups": effective_train,
        "fit_selection_groups": effective_selection,
        "source_group_count": len(cohort),
        "selection_basis": "pre_score_structural_candidate_and_render_common_support",
        "score_based_substitution": False,
    }


def _validate_review(path: Path, source_paths: Sequence[Path]) -> dict[str, Any]:
    review = load_json(path)
    if review.get("stage_id") != "00_protocol_lock" or review.get("status") != "PASS":
        raise ValueError("Stage00 code review is not PASS")
    sources = [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in source_paths]
    if review.get("source_manifest_sha256") != canonical_sha256(sources):
        raise ValueError("Stage00 code review source hash mismatch")
    if review.get("test_status") != "PASS":
        raise ValueError("Stage00 code review tests are not PASS")
    return review


def _validate_parent_states(
    visual: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replacement: Mapping[str, Any],
    render: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> None:
    if visual.get("status") != "complete" or visual.get("decision") != "GO":
        raise ValueError("visual split parent is not complete GO")
    if candidate.get("status") != "complete" or candidate.get("engineering_decision") != "GO":
        raise ValueError("candidate parent is not complete engineering GO")
    if candidate.get("record_count") != 146 or not isinstance(candidate.get("results"), list):
        raise ValueError("candidate parent is not the canonical 146-record artifact")
    if replacement.get("status") != "complete" or replacement.get("engineering_decision") != "GO":
        raise ValueError("strict replacement parent is not complete engineering GO")
    if replacement.get("scientific_decision") != "not_available":
        raise ValueError("strict replacement stage must not contain the statistics decision")
    cohort = replacement.get("cohort")
    if not isinstance(cohort, Mapping) or cohort.get("record_count") != 133:
        raise ValueError("strict replacement cohort is not the canonical 133 records")
    if render.get("record_count") != 133 or render.get("render_count") != 266:
        raise ValueError("render parent is not the canonical 266-render matrix")
    if statistics.get("engineering_decision") != "GO" or statistics.get("scientific_decision") != "NO_GO":
        raise ValueError("MFA-linear statistics parent is not the completed NO_GO")
    selection = replacement.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("strict replacement selection contract is missing")
    for key in ("score_based_selection", "syncnet_used_for_selection"):
        if selection.get(key) is not False:
            raise ValueError(f"strict replacement violates pre-score selection contract: {key}")
    if selection.get("source_order_preserved") is not True:
        raise ValueError("strict replacement source order was not preserved")


def _candidate_by_id(candidate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = candidate.get("results")
    if not isinstance(rows, list):
        raise ValueError("candidate results are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("candidate result must be an object")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise ValueError("candidate sample IDs must be unique")
        mapping = row.get("mapping")
        if not isinstance(mapping, Mapping) or mapping.get("speech_fallback_frames") != 0:
            raise ValueError(f"candidate has speech fallback: {sample_id}")
        if row.get("candidate_samples") != row.get("natural_samples"):
            raise ValueError(f"candidate is not exact natural length: {sample_id}")
        result[sample_id] = row
    return result


def _renders_by_id(render: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    rows = render.get("renders")
    if not isinstance(rows, list):
        raise ValueError("render rows are missing")
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("render row must be an object")
        sample_id = str(row.get("sample_id", ""))
        arm = str(row.get("arm", ""))
        if arm not in {"natural", "candidate"} or not sample_id:
            raise ValueError("render row has invalid identity")
        if arm in result.setdefault(sample_id, {}):
            raise ValueError("duplicate render arm")
        result[sample_id][arm] = row
    if any(set(arms) != {"natural", "candidate"} for arms in result.values()):
        raise ValueError("render matrix is incomplete")
    return result


def build_visual_rows(
    replacement: Mapping[str, Any],
    candidate: Mapping[str, Any],
    render: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cohort = replacement.get("cohort")
    records = cohort.get("records") if isinstance(cohort, Mapping) else None
    if not isinstance(records, list) or len(records) != 133:
        raise ValueError("strict cohort records are incomplete")
    candidates = _candidate_by_id(candidate)
    renders = _renders_by_id(render)
    ids = [str(row.get("sample_id", "")) for row in records if isinstance(row, Mapping)]
    if len(ids) != 133 or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("strict cohort sample IDs are invalid")
    if cohort.get("ordered_sample_ids_sha256") != canonical_sha256(ids):
        raise ValueError("strict cohort ordered ID hash mismatch")
    if set(ids) - set(candidates) or set(ids) != set(renders):
        raise ValueError("candidate/render cohort join is incomplete")

    result: list[dict[str, Any]] = []
    for record, sample_id in zip(records, ids, strict=True):
        candidate_row = candidates[sample_id]
        arms = renders[sample_id]
        if candidate_row.get("candidate_audio_sha256") != record.get("candidate_audio_sha256"):
            raise ValueError(f"candidate hash join mismatch: {sample_id}")
        candidate_audio = str(candidate_row.get("candidate_audio", ""))
        replacement_audio = str(record.get("candidate_audio", ""))
        if not candidate_audio or not replacement_audio:
            raise ValueError(f"candidate audio path is missing: {sample_id}")
        if Path(candidate_audio).resolve() != Path(replacement_audio).resolve():
            raise ValueError(f"candidate path join mismatch: {sample_id}")
        if candidate_row.get("candidate_samples") != record.get("natural_samples"):
            raise ValueError(f"candidate/replacement sample-count mismatch: {sample_id}")
        if candidate_row.get("natural_audio_sha256") != record.get("natural_audio_sha256"):
            raise ValueError(f"natural hash join mismatch: {sample_id}")
        group = str(record.get("source_group", ""))
        if not group or candidate_row.get("source_group") != group:
            raise ValueError(f"source group join mismatch: {sample_id}")
        result.append(
            {
                "sample_id": sample_id,
                "source_group": group,
                "protocol_split": "fit",
                "real_video": str(record.get("face_video", "")),
                "real_video_sha256": str(record.get("face_video_sha256", "")),
                "natural_video": str(arms["natural"].get("video", "")),
                "natural_video_sha256": str(arms["natural"].get("video_sha256", "")),
                "candidate_video": str(arms["candidate"].get("video", "")),
                "candidate_video_sha256": str(arms["candidate"].get("video_sha256", "")),
                "natural_audio": str(record.get("natural_audio", "")),
                "natural_audio_sha256": str(record.get("natural_audio_sha256", "")),
                "candidate_audio": str(record.get("candidate_audio", "")),
                "candidate_audio_sha256": str(record.get("candidate_audio_sha256", "")),
                "natural_samples": int(record.get("natural_samples", 0)),
                "speech_fallback_frames": int(record.get("candidate_mapping", {}).get("speech_fallback_frames", -1)),
                "candidate_trace_sha256": str(candidate_row.get("trace_sha256", "")),
            }
        )
    if any(
        not row[key]
        for row in result
        for key in (
            "real_video",
            "real_video_sha256",
            "natural_video",
            "natural_video_sha256",
            "candidate_video",
            "candidate_video_sha256",
            "natural_audio",
            "natural_audio_sha256",
            "candidate_audio",
            "candidate_audio_sha256",
            "candidate_trace_sha256",
        )
    ):
        raise ValueError("visual cohort contains empty path/hash fields")
    if any(row["natural_samples"] <= 0 or row["speech_fallback_frames"] != 0 for row in result):
        raise ValueError("visual cohort violates exact-length or zero-speech-fallback contract")
    return result


def _derive_test_lock(parent: Mapping[str, Any], stage_id: str, parents: Mapping[str, Any]) -> dict[str, Any]:
    validate_test_lock(parent)
    result = dict(parent)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "stage_id": stage_id,
            "parent_test_lock": parents["visual_test_lock"],
            "status": "sealed_unvisited",
        }
    )
    for key, value in result.items():
        if key.endswith(("_media_opened", "_derived_features_created", "_scores_created")) and value is not False:
            raise ValueError(f"test lock records forbidden access: {key}")
    return result


def build_protocol(
    *,
    output_dir: Path,
    cohort_policy: str,
    code_review_path: Path,
    source_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_id = _stage_id(output_dir)
    parent_bindings = {
        "visual_stage00": _asset(VISUAL_STAGE00, CANONICAL_HASHES["visual_stage00"]),
        "visual_test_lock": _asset(VISUAL_TEST_LOCK, CANONICAL_HASHES["visual_test_lock"]),
        "natural_control_stage00": _asset(NATURAL_CONTROL_STAGE00, CANONICAL_HASHES["natural_control_stage00"]),
        "candidate_manifest": _asset(CANDIDATE_MANIFEST, CANONICAL_HASHES["candidate_manifest"]),
        "replacement_manifest": _asset(REPLACEMENT_MANIFEST, CANONICAL_HASHES["replacement_manifest"]),
        "render_manifest": _asset(RENDER_MANIFEST, CANONICAL_HASHES["render_manifest"]),
        "mfa_statistics_decision": _asset(MFA_STATISTICS_DECISION, CANONICAL_HASHES["mfa_statistics_decision"]),
    }
    visual = load_json(VISUAL_STAGE00)
    parent_lock = load_json(VISUAL_TEST_LOCK)
    natural_control = load_json(NATURAL_CONTROL_STAGE00)
    candidate = load_json(CANDIDATE_MANIFEST)
    replacement = load_json(REPLACEMENT_MANIFEST)
    render = load_json(RENDER_MANIFEST)
    statistics = load_json(MFA_STATISTICS_DECISION)
    _validate_parent_states(visual, candidate, replacement, render, statistics)
    review = _validate_review(code_review_path, source_paths)

    visual_split = visual.get("cohorts", {}).get("split")
    natural_split = natural_control.get("protocol", {}).get("split")
    if not isinstance(visual_split, Mapping) or not isinstance(natural_split, Mapping):
        raise ValueError("frozen split parents are missing")
    for key in ("fit_groups", "internal_dev_groups", "validation_groups", "test_groups"):
        if list(visual_split.get(key, [])) != list(natural_split.get(key, [])):
            raise ValueError(f"visual and natural-control splits disagree: {key}")

    rows = build_visual_rows(replacement, candidate, render)
    candidate_groups = sorted({str(row.get("source_group")) for row in candidate["results"]})
    cohort_groups = sorted({row["source_group"] for row in rows})
    support = validate_group_support(
        cohort_groups=cohort_groups,
        candidate_groups=candidate_groups,
        fit_groups=_unique_strings(visual_split.get("fit_groups"), "fit_groups"),
        fit_train_groups=_unique_strings(natural_split.get("fit_train_groups"), "fit_train_groups"),
        fit_selection_groups=_unique_strings(natural_split.get("fit_selection_groups"), "fit_selection_groups"),
        policy=cohort_policy,
    )
    source_group_counts = Counter(row["source_group"] for row in rows)
    ordered_ids = [row["sample_id"] for row in rows]
    test_lock = _derive_test_lock(parent_lock, stage_id, parent_bindings)

    own_sources = [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in source_paths]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "lrs3_duration_compatible_tts_visual_control_protocol",
        "protocol_id": PROTOCOL_ID,
        "stage_id": stage_id,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": "01_visual_teacher_audit_retry1",
        "parents": parent_bindings,
        "cohort": {
            "record_count": len(rows),
            "source_group_count": len(source_group_counts),
            "ordered_sample_ids_sha256": canonical_sha256(ordered_ids),
            "source_group_counts": dict(source_group_counts),
            "records": rows,
        },
        "split": {
            **support,
            "internal_dev_groups": list(visual_split["internal_dev_groups"]),
            "validation_groups": list(visual_split["validation_groups"]),
            "test_groups": list(visual_split["test_groups"]),
            "unit": "source_group_directory_not_verified_speaker_id",
        },
        "visual_contract": {
            "primary": "natural_clock_exact_time_canonical_mouth_landmark_distance",
            "benefit": "D(real,V_N)-D(real,V_M)",
            "benefit_direction": "positive_is_better",
            "secondary": "visual_only_constrained_dtw_canonical_mouth_shape_distance",
            "valid_fraction_min": 0.90,
            "exact_time_coverage_min": 0.85,
            "minimum_eligible_record_count": minimum_eligible_count(len(rows)),
            "all_effective_fit_groups_required": True,
            "native_or_syncnet_scores_used": False,
            "dtw_can_rescue_primary": False,
            "statistics_unit": "source_group_mean",
            "one_sided_exact_sign_flip_alpha": 0.05,
        },
        "future_control_lock": {
            "stage03_action_family": "rank4_fit_train_pca_of_exact_clock_candidate_minus_natural_mel",
            "rank": 4,
            "temporal_knots_per_basis": 4,
            "natural_anchor": True,
            "search_objective": "real_video_exact_time_visual_loss_plus_prelocked_regularizers",
            "official_replacement_score_available_to_search": False,
            "phone_mask": False,
            "fixed_alpha_residual": False,
            "fit_train_group_count": len(support["fit_train_groups"]),
            "fit_selection_group_count": len(support["fit_selection_groups"]),
            "effective_fit_group_count": len(support["effective_fit_groups"]),
        },
        "selection": {
            "candidate_parent_pre_score": True,
            "face_preflight_structural": True,
            "syncnet_preflight_structural_only": True,
            "score_based_selection": False,
            "source_order_preserved": True,
            "sample_substitution_allowed": False,
        },
        "media_access": {
            "stage00_media_decoded": False,
            "stage00_features_created": False,
            "stage00_scores_created": False,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
        "source_manifest": own_sources,
        "source_manifest_sha256": canonical_sha256(own_sources),
        "code_review": {"path": str(code_review_path.resolve()), "sha256": file_sha256(code_review_path)},
        "code_review_payload_sha256": canonical_sha256(review),
        "git": git_info(),
    }
    assert_finite_json(manifest)
    validate_test_lock(test_lock)
    return manifest, test_lock


def _failure_payload(stage_id: str, cohort_policy: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage_id": stage_id,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "cohort_policy": cohort_policy,
        "error_type": type(error).__name__,
        "error": str(error),
        "next_allowed_stage": None,
    }


def run(
    output_dir: Path,
    *,
    cohort_policy: str,
    code_review_path: Path,
    source_paths: Sequence[Path],
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output}")
    stage_id = _stage_id(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        manifest, test_lock = build_protocol(
            output_dir=output,
            cohort_policy=cohort_policy,
            code_review_path=code_review_path.resolve(),
            source_paths=source_paths,
        )
    except Exception as error:
        failure = _failure_payload(stage_id, cohort_policy, error)
        write_json(output / "failure.json", failure)
        write_json(output / "summary.json", failure)
        write_json(output / "decision.json", failure)
        return failure

    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage_id": stage_id,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": manifest["cohort"]["record_count"],
        "source_group_count": manifest["cohort"]["source_group_count"],
        "cohort_policy": cohort_policy,
        "excluded_fit_groups": manifest["split"]["excluded_fit_groups"],
        "minimum_eligible_record_count": manifest["visual_contract"]["minimum_eligible_record_count"],
        "next_allowed_stage": manifest["next_allowed_stage"],
    }
    decision = {
        **summary,
        "reason": "duration-compatible exact-clock visual cohort locked without media decoding or score-based selection",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "test_lock.json", test_lock)
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", decision)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort-policy", choices=(STRICT_24, COMMON_23), default=DEFAULT_COHORT_POLICY)
    parser.add_argument("--code-review", type=Path, default=DEFAULT_CODE_REVIEW)
    parser.add_argument(
        "--test-source",
        type=Path,
        default=REPO / "tests/experiments/lrs3_tts_visual_control/test_protocol.py",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = (Path(__file__).resolve(), args.test_source.resolve())
    result = run(
        args.output_dir,
        cohort_policy=args.cohort_policy,
        code_review_path=args.code_review,
        source_paths=sources,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("engineering_decision") == "GO" else 2


if __name__ == "__main__":
    sys.exit(main())
