from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "scripts/experiments/lrs3_tts_visual_control/protocol.py"
SPEC = importlib.util.spec_from_file_location("lrs3_tts_visual_control_protocol", MODULE_PATH)
assert SPEC and SPEC.loader
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)

VISUAL_STAGE00 = REPO / "runs/lrs3_tts_visual_advantage_20260824/00_protocol_lock_retry2/manifest.json"
VISUAL_TEST_LOCK = VISUAL_STAGE00.with_name("test_lock.json")
NATURAL_CONTROL_STAGE00 = REPO / "runs/lrs3_natural_control_20260823/00_protocol_lock_retry2/manifest.json"
CANDIDATE_MANIFEST = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/02_candidate_audio_retry4/candidate_manifest.json"
REPLACEMENT_MANIFEST = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/03_strict_replacement_face_ready_retry7/replacement_manifest.json"
RENDER_MANIFEST = REPLACEMENT_MANIFEST.with_name("render_manifest.json")


def load_json(path: Path) -> dict:
    return protocol.load_json(path)


def test_common_support_is_pre_score_and_preserves_selection_groups():
    fit = [protocol.MISSING_COMMON_GROUP] + [f"g{i}" for i in range(23)]
    train = fit[:18]
    selection = fit[18:]
    result = protocol.validate_group_support(
        cohort_groups=fit[1:],
        candidate_groups=fit[1:],
        fit_groups=fit,
        fit_train_groups=train,
        fit_selection_groups=selection,
        policy=protocol.COMMON_23,
    )

    assert result["policy_id"] == protocol.COMMON_23
    assert result["policy_basis"].startswith("pre_score structural")
    assert result["excluded_fit_groups"] == [protocol.MISSING_COMMON_GROUP]
    assert len(result["fit_train_groups"]) == 17
    assert len(result["fit_selection_groups"]) == 6
    assert result["score_based_substitution"] is False
    assert result["parent_fit_train_groups"] == train


def test_strict_policy_fails_closed_on_missing_fit_group():
    fit = [protocol.MISSING_COMMON_GROUP] + [f"g{i}" for i in range(23)]
    with pytest.raises(ValueError, match="strict 24-group policy"):
        protocol.validate_group_support(
            cohort_groups=fit[1:],
            candidate_groups=fit[1:],
            fit_groups=fit,
            fit_train_groups=fit[:18],
            fit_selection_groups=fit[18:],
            policy=protocol.STRICT_24,
        )


def test_common_policy_rejects_unregistered_exclusion():
    fit = [protocol.MISSING_COMMON_GROUP] + [f"g{i}" for i in range(23)]
    cohort = [protocol.MISSING_COMMON_GROUP] + [f"g{i}" for i in range(22)]
    with pytest.raises(ValueError, match="unexpected common-support exclusion"):
        protocol.validate_group_support(
            cohort_groups=cohort,
            candidate_groups=cohort,
            fit_groups=fit,
            fit_train_groups=fit[:18],
            fit_selection_groups=fit[18:],
            policy=protocol.COMMON_23,
        )


def test_canonical_parent_join_is_133_records_and_23_groups():
    replacement = load_json(REPLACEMENT_MANIFEST)
    candidate = load_json(CANDIDATE_MANIFEST)
    render = load_json(RENDER_MANIFEST)
    rows = protocol.build_visual_rows(replacement, candidate, render)

    assert len(rows) == 133
    assert len({row["sample_id"] for row in rows}) == 133
    assert len({row["source_group"] for row in rows}) == 23
    assert all(row["protocol_split"] == "fit" for row in rows)
    assert all(row["natural_samples"] > 0 for row in rows)
    assert all(row["speech_fallback_frames"] == 0 for row in rows)


def test_canonical_parent_join_rejects_hash_tampering():
    replacement = load_json(REPLACEMENT_MANIFEST)
    candidate = load_json(CANDIDATE_MANIFEST)
    render = load_json(RENDER_MANIFEST)
    changed = deepcopy(candidate)
    changed["results"][0]["natural_audio_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="natural hash join mismatch"):
        protocol.build_visual_rows(replacement, changed, render)


def test_canonical_parent_join_rejects_candidate_length_tampering():
    replacement = load_json(REPLACEMENT_MANIFEST)
    candidate = load_json(CANDIDATE_MANIFEST)
    render = load_json(RENDER_MANIFEST)
    changed = deepcopy(candidate)
    changed["results"][0]["candidate_samples"] += 1

    with pytest.raises(ValueError, match="exact natural length"):
        protocol.build_visual_rows(replacement, changed, render)


def test_canonical_parent_join_rejects_candidate_path_tampering():
    replacement = load_json(REPLACEMENT_MANIFEST)
    candidate = load_json(CANDIDATE_MANIFEST)
    render = load_json(RENDER_MANIFEST)
    changed = deepcopy(candidate)
    changed["results"][0]["candidate_audio"] = str(REPO / "different.wav")

    with pytest.raises(ValueError, match="candidate path join mismatch"):
        protocol.build_visual_rows(replacement, changed, render)


def test_canonical_parent_join_rejects_replacement_sample_count_tampering():
    replacement = load_json(REPLACEMENT_MANIFEST)
    candidate = load_json(CANDIDATE_MANIFEST)
    render = load_json(RENDER_MANIFEST)
    changed = deepcopy(replacement)
    changed["cohort"]["records"][0]["natural_samples"] += 1

    with pytest.raises(ValueError, match="sample-count mismatch"):
        protocol.build_visual_rows(changed, candidate, render)


def test_derived_test_lock_preserves_sealed_unvisited_state():
    visual_lock = load_json(VISUAL_TEST_LOCK)
    parents = {
        "visual_test_lock": {
            "path": str(VISUAL_TEST_LOCK.resolve()),
            "sha256": protocol.file_sha256(VISUAL_TEST_LOCK),
        }
    }
    derived = protocol._derive_test_lock(visual_lock, "00_protocol_lock_retry9", parents)

    assert derived["status"] == "sealed_unvisited"
    assert derived["parent_test_lock"] == parents["visual_test_lock"]
    assert derived["media_opened"] is False
    assert derived["derived_features_created"] is False
    assert derived["scores_created"] is False
    protocol.validate_test_lock(derived)


def test_parent_split_sources_have_identical_sealed_nonfit_groups():
    visual = load_json(VISUAL_STAGE00)
    natural = load_json(NATURAL_CONTROL_STAGE00)
    visual_split = visual["cohorts"]["split"]
    natural_split = natural["protocol"]["split"]

    for key in ("fit_groups", "internal_dev_groups", "validation_groups", "test_groups"):
        assert visual_split[key] == natural_split[key]
    assert len(visual_split["fit_groups"]) == 24
    assert len(visual_split["internal_dev_groups"]) == 6
    assert len(visual_split["validation_groups"]) == 6
    assert len(visual_split["test_groups"]) == 7


def test_build_protocol_round_trips_canonical_common_support(tmp_path: Path):
    source_paths = (MODULE_PATH, Path(__file__).resolve())
    source_manifest = [
        {"path": str(path.resolve()), "sha256": protocol.file_sha256(path)}
        for path in source_paths
    ]
    review_path = tmp_path / "code_review.json"
    review_path.write_text(
        json.dumps(
            {
                "stage_id": "00_protocol_lock",
                "status": "PASS",
                "test_status": "PASS",
                "source_manifest_sha256": protocol.canonical_sha256(source_manifest),
            }
        ),
        encoding="utf-8",
    )

    manifest, test_lock = protocol.build_protocol(
        output_dir=tmp_path / "00_protocol_lock_retry9",
        cohort_policy=protocol.COMMON_23,
        code_review_path=review_path,
        source_paths=source_paths,
    )

    assert manifest["engineering_decision"] == "GO"
    assert manifest["cohort"]["record_count"] == 133
    assert manifest["split"]["excluded_fit_groups"] == [protocol.MISSING_COMMON_GROUP]
    assert len(manifest["split"]["fit_train_groups"]) == 17
    assert len(manifest["split"]["fit_selection_groups"]) == 6
    assert manifest["future_control_lock"]["fit_train_group_count"] == 17
    assert test_lock["status"] == "sealed_unvisited"
    assert test_lock["test_media_opened"] is False
    assert test_lock["test_derived_features_created"] is False
    assert test_lock["test_scores_created"] is False


def test_nonempty_output_is_never_overwritten(tmp_path: Path):
    output = tmp_path / "00_protocol_lock_retry9"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty output directory"):
        protocol.run(
            output,
            cohort_policy=protocol.COMMON_23,
            code_review_path=tmp_path / "missing-review.json",
            source_paths=(MODULE_PATH,),
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_default_is_registered_common_support_policy():
    args = protocol.parse_args(["--output-dir", "runs/00_protocol_lock_retry9"])
    assert args.cohort_policy == protocol.DEFAULT_COHORT_POLICY == protocol.COMMON_23


def test_source_hashes_are_canonical_parent_bindings():
    expected_paths = {
        "visual_stage00": VISUAL_STAGE00,
        "visual_test_lock": VISUAL_TEST_LOCK,
        "natural_control_stage00": NATURAL_CONTROL_STAGE00,
        "candidate_manifest": CANDIDATE_MANIFEST,
        "replacement_manifest": REPLACEMENT_MANIFEST,
        "render_manifest": RENDER_MANIFEST,
        "mfa_statistics_decision": protocol.MFA_STATISTICS_DECISION,
    }
    for key, path in expected_paths.items():
        assert protocol.file_sha256(path) == protocol.CANONICAL_HASHES[key]
