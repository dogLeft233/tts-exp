#!/usr/bin/env python3
"""Build a metadata-only V5 historical-fit roster proposal.

This script reads only frozen JSON metadata. It does not open or hash any WAV,
archive, video, model weight, or confirmation audio.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "openspec/changes/develop-cross-tfg-audio-head-20261003/data-protocol.json"
REPO_PIN_PATH = ROOT / ".codex/scratch/cross_tfg_aishell_official_hf_repo_20261003.json"
TREE_PIN_PATH = ROOT / ".codex/scratch/cross_tfg_aishell_official_hf_tree_20261003.json"
FIT_PLAN_PATH = ROOT / "runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json"
OUT_PATH = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt01/v5_fit_expansion_roster.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_json(path: Path) -> tuple[dict | list, str, int]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw), len(raw)


def main() -> None:
    protocol, protocol_sha, protocol_bytes = raw_json(PROTOCOL_PATH)
    repo, repo_sha, repo_bytes = raw_json(REPO_PIN_PATH)
    tree, tree_sha, tree_bytes = raw_json(TREE_PIN_PATH)
    fit_plan, fit_plan_sha, fit_plan_bytes = raw_json(FIT_PLAN_PATH)

    assert isinstance(tree, list)
    assert repo["sha"] == protocol["dataset_revision"]
    assert fit_plan["dataset_revision"] == protocol["dataset_revision"]
    assert fit_plan["fit_speaker_ids"] == ["S0052", "S0053", "S0054", "S0055"]

    historical_fit = protocol["fit_speaker_ids"]
    fit_speakers = historical_fit[:20]
    assert len(fit_speakers) == 20 and len(set(fit_speakers)) == 20
    dev = protocol["dev_speaker_ids"]
    confirm = protocol["confirm_speaker_ids"]
    assert set(fit_speakers).isdisjoint(dev)
    assert set(fit_speakers).isdisjoint(confirm)
    assert set(dev).isdisjoint(confirm)

    tree_by_path = {row["path"]: row for row in tree if row.get("type") == "file"}
    archives = []
    for speaker_id in fit_speakers:
        archive_path = f"data_aishell/wav/{speaker_id}.tar.gz"
        pin = tree_by_path.get(archive_path)
        assert pin is not None and pin.get("lfs", {}).get("oid")
        assert pin["size"] == pin["lfs"]["size"]
        archives.append(
            {
                "speaker_id": speaker_id,
                "path": archive_path,
                "git_blob_oid": pin["oid"],
                "lfs_sha256": pin["lfs"]["oid"],
                "bytes": pin["size"],
                "xet_hash": pin.get("xetHash"),
                "local_archive_sha256": None,
                "local_archive_bytes_verified": False,
            }
        )

    fit_manifest = fit_plan.get("source_rows_manifest", {}).get("audio_inputs", [])
    existing_fit_audio_metadata = [
        {
            "sample_id": row["sample_id"],
            "speaker_id": row["speaker_id"],
            "source_archive_member": row["source_archive_member"],
            "source_archive_lfs_sha256": row["source_archive_sha256"],
            "canonical_pcm_file_sha256": row["file_sha256"],
            "canonical_float32_pcm_sha256": row["pcm_sha256"],
            "sample_count": row["sample_count"],
            "metadata_source": "existing frozen fit plan; no audio bytes read by this script",
            "assumed_selected_in_v5": False,
        }
        for row in fit_manifest
    ]

    result = {
        "schema": "cross_tfg_v5_fit_expansion_roster_proposal_v1",
        "status": "METADATA_ONLY_MEMBER_SCAN_PENDING",
        "purpose": "prepare >=20 historical fit speakers and >=40 planned clip slots without reading sealed audio",
        "source_metadata": {
            "dataset": protocol["dataset"],
            "dataset_revision": protocol["dataset_revision"],
            "source_repository": protocol["source_repository"],
            "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "protocol_raw_sha256": protocol_sha,
            "protocol_bytes": protocol_bytes,
            "official_repo_metadata_path": REPO_PIN_PATH.relative_to(ROOT).as_posix(),
            "official_repo_metadata_raw_sha256": repo_sha,
            "official_repo_metadata_bytes": repo_bytes,
            "official_tree_path": TREE_PIN_PATH.relative_to(ROOT).as_posix(),
            "official_tree_raw_sha256": tree_sha,
            "official_tree_bytes": tree_bytes,
            "existing_fit_plan_path": FIT_PLAN_PATH.relative_to(ROOT).as_posix(),
            "existing_fit_plan_raw_sha256": fit_plan_sha,
            "existing_fit_plan_bytes": fit_plan_bytes,
            "generator_script_path": Path(__file__).relative_to(ROOT).as_posix(),
            "generator_script_raw_sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "split_binding": {
            "fit_speaker_ids": fit_speakers,
            "dev_speaker_ids": dev,
            "confirm_speaker_ids": confirm,
            "fit_dev_intersection": sorted(set(fit_speakers) & set(dev)),
            "fit_confirm_intersection": sorted(set(fit_speakers) & set(confirm)),
            "dev_confirm_intersection": sorted(set(dev) & set(confirm)),
            "disjointness_check": "PASS_METADATA_ONLY",
            "historical_fit_pool_size": len(historical_fit),
            "fit_speaker_selection": "first 20 speaker IDs from the frozen fit_speaker_ids order (S0052-S0071)",
        },
        "planned_roster": {
            "planned_speakers": len(fit_speakers),
            "planned_clip_slots": len(fit_speakers) * 16,
            "clip_slots_per_speaker": 16,
            "slot_j_values": list(range(16)),
            "minimum_clip_coverage_target": 40,
            "clip_selection_rule": protocol["clip_selection"]["fit"],
            "order": protocol["clip_selection"]["order"],
            "eligibility": protocol["clip_eligibility"],
            "missing_policy": protocol["clip_selection"]["missing"],
            "interpretation": "320 planned slots follow the already-frozen 16-per-speaker rule. Actual sample IDs and eligible counts remain unknown until archive/member metadata scan; this artifact does not claim 40 eligible clips exist.",
        },
        "upstream_archive_pins": archives,
        "existing_historical_fit_audio_metadata": existing_fit_audio_metadata,
        "bounded_40_clip_alternative": {
            "status": "PROPOSED_ONLY_REQUIRES_ADDITIVE_SELECTION_RULE_FREEZE",
            "speakers": fit_speakers,
            "planned_clip_slots": 40,
            "rule_if_chosen": "two lexical-first eligible utterances per speaker",
            "warning": "This is not the frozen fit clip-selection rule. Do not use unless an explicit additive roster protocol is frozen before member scan or any new effects.",
        },
        "audio_read_scope": {
            "source_audio_bytes_opened": False,
            "archive_bytes_opened": False,
            "archive_members_scanned": False,
            "dev_audio_read": False,
            "confirm_audio_read": False,
            "model_forward": False,
            "effect_or_score_read": False,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw = OUT_PATH.read_bytes()
    print(json.dumps({"path": str(OUT_PATH), "bytes": len(raw), "raw_sha256": sha256_bytes(raw), "speakers": len(fit_speakers), "slots": len(fit_speakers) * 16, "disjointness": result["split_binding"]["disjointness_check"], "audio_scope": result["audio_read_scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
