#!/usr/bin/env python3
"""Build ADR013-bound V5 roster metadata without scanning or opening audio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "openspec/changes/develop-cross-tfg-audio-head-20261003/data-protocol.json"
REPO_PIN_PATH = ROOT / ".codex/scratch/cross_tfg_aishell_official_hf_repo_20261003.json"
TREE_PIN_PATH = ROOT / ".codex/scratch/cross_tfg_aishell_official_hf_tree_20261003.json"
FIT_PLAN_PATH = ROOT / "runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json"
ADR_PATH = ROOT / "basic-memory/docs/adr/013-canonical-endpoint-subset-for-expanded-historical-fit.md"
OUT_PATH = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/v5_fit_expansion_roster.json"

EXPECTED_PINS = {
    "protocol": "cf615529edc8ae0b6ffff0a34246db71b3ec8ce5ac17d9f0daf2618ac624cf87",
    "repo_metadata": "25a209354c4b99ae7603864f406fd0ce74fc80ea238cbfdf2ed67968d46a7b32",
    "tree_metadata": "f3d49f418cb9abb38b546ff4aab95d729205f69a690c49acf10519ca4b1097c0",
    "fit_plan": "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7",
    "adr013": "82a8a397f2317148d026ba4a2e0558b81656df0f1f74dcc4e58015b0323ab04b",
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> tuple[dict | list, str, int]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw), len(raw)


def main() -> None:
    protocol, protocol_sha, protocol_bytes = read_json(PROTOCOL_PATH)
    repo, repo_sha, repo_bytes = read_json(REPO_PIN_PATH)
    tree, tree_sha, tree_bytes = read_json(TREE_PIN_PATH)
    fit_plan, fit_plan_sha, fit_plan_bytes = read_json(FIT_PLAN_PATH)
    adr_raw = ADR_PATH.read_bytes()
    adr_text = adr_raw.decode("utf-8")
    adr_sha = sha256_bytes(adr_raw)
    actual_pins = {
        "protocol": protocol_sha,
        "repo_metadata": repo_sha,
        "tree_metadata": tree_sha,
        "fit_plan": fit_plan_sha,
        "adr013": adr_sha,
    }
    if actual_pins != EXPECTED_PINS:
        raise ValueError(f"frozen input raw SHA mismatch: {actual_pins}")
    if "## Decision" not in adr_text or "j=0" not in adr_text or "j=15" not in adr_text:
        raise ValueError("ADR013 does not bind the canonical endpoint selection")
    if not isinstance(tree, list):
        raise ValueError("official tree pin must be a JSON array")
    if repo["sha"] != protocol["dataset_revision"]:
        raise ValueError("repository dataset revision differs from the frozen protocol")
    if fit_plan["dataset_revision"] != protocol["dataset_revision"]:
        raise ValueError("fit plan dataset revision differs from the frozen protocol")

    historical_fit = protocol["fit_speaker_ids"]
    fit_speakers = historical_fit[:20]
    if fit_speakers != [f"S{n:04d}" for n in range(52, 72)]:
        raise ValueError("first 20 frozen fit speakers are not S0052-S0071")
    dev = protocol["dev_speaker_ids"]
    confirm = protocol["confirm_speaker_ids"]
    if set(fit_speakers) & (set(dev) | set(confirm)):
        raise ValueError("expanded historical-fit speakers overlap dev or confirm")

    tree_by_path = {row["path"]: row for row in tree if row.get("type") == "file"}
    archive_pins = []
    for speaker_id in fit_speakers:
        archive_path = f"data_aishell/wav/{speaker_id}.tar.gz"
        pin = tree_by_path.get(archive_path)
        if pin is None or pin.get("lfs", {}).get("oid") is None:
            raise ValueError(f"official archive pin missing for {speaker_id}")
        if pin["size"] != pin["lfs"]["size"]:
            raise ValueError(f"official archive size/LFS size mismatch for {speaker_id}")
        archive_pins.append({
            "speaker_id": speaker_id,
            "path": archive_path,
            "git_blob_oid": pin["oid"],
            "lfs_sha256": pin["lfs"]["oid"],
            "bytes": pin["size"],
            "xet_hash": pin.get("xetHash"),
            "local_archive_sha256": None,
            "local_archive_bytes_verified": False,
        })

    historical_audio = [
        {
            "sample_id": row["sample_id"],
            "speaker_id": row["speaker_id"],
            "source_archive_member": row["source_archive_member"],
            "source_archive_lfs_sha256": row["source_archive_sha256"],
            "canonical_pcm_file_sha256": row["file_sha256"],
            "canonical_float32_pcm_sha256": row["pcm_sha256"],
            "sample_count": row["sample_count"],
            "metadata_source": "prior frozen fit plan JSON; no audio bytes read by this script",
            "selected_by_adr013": False,
        }
        for row in fit_plan.get("source_rows_manifest", {}).get("audio_inputs", [])
    ]

    fold_order = sorted(
        fit_speakers,
        key=lambda speaker_id: (
            hashlib.sha256(f"cross-tfg-v5-fit-fold-v1|{speaker_id}".encode("utf-8")).hexdigest(),
            speaker_id,
        ),
    )
    fold_by_speaker = {
        speaker_id: {
            "fold_index": rank % 5,
            "fold_hash_sha256": hashlib.sha256(
                f"cross-tfg-v5-fit-fold-v1|{speaker_id}".encode("utf-8")
            ).hexdigest(),
            "hash_rank_ascending": rank,
        }
        for rank, speaker_id in enumerate(fold_order)
    }

    slots = []
    endpoint_slot_keys = []
    for speaker_id in fit_speakers:
        for j in range(16):
            selected = j in (0, 15)
            slots.append([
                speaker_id,
                j,
                selected,
                "NOT_SCANNED",
                None,
                "NOT_ASSESSED",
                "NOT_ASSESSED_PRE_SCAN",
                "NO_REPLACEMENT",
            ])
            if j in (0, 15):
                endpoint_slot_keys.append([speaker_id, j])

    output = {
        "schema": "cross_tfg_v5_fit_expansion_roster_adr013_v1",
        "status": "METADATA_ONLY_ADR013_BOUND_MEMBER_SCAN_PENDING",
        "purpose": "preserve all canonical 16-slot metadata for the first 20 fit speakers and identify the additive j=0/j=15 subset",
        "selection_decision": {
            "adr_path": ADR_PATH.relative_to(ROOT).as_posix(),
            "adr_raw_sha256": adr_sha,
            "adr_git_sha": "0597b3dfbfd6bd5df93c96d7cb70be12ce476e59",
            "canonical_subset_rule": "For S0052-S0071, select the existing 16-slot fit roster endpoints j=0 and j=15; do not substitute lexical-first clips or replace absent/duplicate/ineligible endpoints.",
            "not_authorized_by_this_metadata_artifact": ["archive member scanning", "audio byte reads", "fitting", "rendering", "scoring", "dev/confirm access"],
        },
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
            "fit_dev_intersection": [],
            "fit_confirm_intersection": [],
            "dev_confirm_intersection": sorted(set(dev) & set(confirm)),
            "disjointness_check": "PASS_METADATA_ONLY",
            "historical_fit_pool_size": len(historical_fit),
            "speaker_selection": "first 20 IDs in frozen fit_speaker_ids order (S0052-S0071)",
        },
        "canonical_roster": {
            "planned_speakers": len(fit_speakers),
            "planned_clip_slots": len(slots),
            "clip_slots_per_speaker": 16,
            "slot_j_values": list(range(16)),
            "slot_columns": ["speaker_id", "canonical_j", "selected_by_adr013", "member_scan_status", "selected_sample_id", "eligibility_status", "duplicate_status", "replacement_policy"],
            "selection_rule": protocol["clip_selection"]["fit"],
            "order": protocol["clip_selection"]["order"],
            "eligibility": protocol["clip_eligibility"],
            "missing_policy": protocol["clip_selection"]["missing"],
            "slots": slots,
            "interpretation": "320 canonical slots are retained. Actual rank, member, eligibility, duplicate status, and available independent endpoint count remain unknown until the reviewed archive/member scan.",
        },
        "adr013_endpoint_subset": {
            "status": "PLANNED_NOT_VERIFIED",
            "speakers": fit_speakers,
            "planned_slots": 40,
            "required_canonical_j": [0, 15],
            "slot_key_columns": ["speaker_id", "canonical_j"],
            "slot_keys": endpoint_slot_keys,
            "verified_unique_eligible_clips": None,
            "coverage_claim_allowed": False,
            "duplicate_or_missing_rule": "retained in the 40-slot planned denominator; no replacement",
            "interpretation": "Do not claim 40 eligible clips until a separate reviewed member scan validates all 40 endpoints as unique, finite, mono, 16 kHz, and 51,200-128,000 samples.",
        },
        "five_speaker_folds": {
            "hash_rule": "ascending SHA256(UTF8('cross-tfg-v5-fit-fold-v1|' + speaker_id)); hash ties sort by speaker_id; fold_index = zero_based_hash_rank mod 5",
            "folds": [
                {"fold_index": fold, "held_out_speakers": [s for s in fold_order if fold_by_speaker[s]["fold_index"] == fold]}
                for fold in range(5)
            ],
            "speaker_hash_order": fold_order,
        },
        "upstream_archive_pins": archive_pins,
        "prior_historical_fit_audio_metadata": historical_audio,
        "audio_and_effect_scope": {
            "archive_bytes_opened": False,
            "archive_members_scanned": False,
            "source_audio_bytes_opened": False,
            "dev_audio_read": False,
            "confirm_audio_read": False,
            "model_forward": False,
            "fit_or_effect_read": False,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # This is a 320-slot machine roster, not a prose plan. Compact encoding keeps
    # the full per-slot status matrix below the repository's 50 KiB plan limit.
    raw = (json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    OUT_PATH.write_bytes(raw)
    print(json.dumps({
        "path": str(OUT_PATH),
        "bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "planned_canonical_slots": len(slots),
        "planned_endpoint_slots": len(endpoint_slot_keys),
        "unique_eligible_endpoints_verified": None,
        "scope": output["audio_and_effect_scope"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
