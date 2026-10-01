#!/usr/bin/env python3
"""Verify the downloaded score-free WORLD visual metadata archive and freeze joins."""
from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path("runs/recovered_20260930/remote/world_visual_analysis_attempt09")
BUNDLE_DIR = BASE / "attempt04"
MANIFEST_PATH = BUNDLE_DIR / "bundle_manifest.json"
REPORT_PATH = BUNDLE_DIR / "track_metadata_inventory.json"
PART_PATH = BUNDLE_DIR / "track_metadata_bundle.tar.gz.part000"
FREEZE_PATH = BASE / "world_visual_availability_freeze.json"
FREEZE_SEAL_PATH = BASE / "world_visual_availability_freeze.seal.json"
EXPECTED_MANIFEST_SHA256 = "214ff30bcdd8d05a9821f31d752198b5e196190822f6e5d118202e200f787fba"
EXPECTED_REPORT_SHA256 = "afb6994005369ff9e3141d6c9da625829f8147285235946c3d027ac892ab65ee"
EXPECTED_ARCHIVE_SHA256 = "9ee31fb942c300f5c82217016c48d0d623b58a04d63eb42c7492ac2ead6e7346"
EXPECTED_FREEZE_SHA256 = "67131b9bd381dd2e393ab739ae51408e5edf15594cd3e9b67b844c5857621ccf"
EXPECTED_FREEZE_SEAL_SHA256 = "1e7bab71c2d16689f5f316b543883dec3058f27aa053f7689d9495e95b3b37d2"
EXPECTED_TRACK_ALL_SHA256 = "717c4777ab6907dfdc131476a087d0c14a4f1fc6ea4593559c464d824f9ab75b"
EXPECTED_TRACK_SEAL_SHA256 = "e2f2f212cd95b51801032a0ff717bb3ee4bd60ac8a53270f2e5c38d52637639f"
EXPECTED_GRID_SHA256 = "7b3e879b9a34a6ea20159e770d6530fc77b19fd013381ce4017e87cbbc19203b"
EXPECTED_EXTRACTOR_SHA256 = "6bec294adcfeff320f70a40e6579b9cb67bc69677fcc07f3a9950ad4186dd87b"
EXPECTED_VISUAL_CONSUMER_SHA256 = "8e40c3017ca7c53924302522678c18854c9b0702ce2223821b03d7fd05166a92"
EXPECTED_PLAN_RAW_SHA256 = "bba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d"
EXPECTED_PLAN_FINGERPRINT_SHA256 = "1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc"
VISUAL_IDS = {"V_FIXED", "V_RT", "V_EX", "V_SP", "V_BOTH", "V_F0", "V_AP", "V_EX_EM", "V_SP_EM", "V_E"}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def main() -> None:
    require(sha_file(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256, "bundle_manifest_sha_mismatch")
    require(sha_file(REPORT_PATH) == EXPECTED_REPORT_SHA256, "metadata_report_sha_mismatch")
    require(sha_file(PART_PATH) == EXPECTED_ARCHIVE_SHA256, "archive_part_sha_mismatch")
    require(sha_file(FREEZE_PATH) == EXPECTED_FREEZE_SHA256, "availability_freeze_sha_mismatch")
    require(sha_file(FREEZE_SEAL_PATH) == EXPECTED_FREEZE_SEAL_SHA256, "availability_freeze_seal_sha_mismatch")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_seal = json.loads(FREEZE_SEAL_PATH.read_text(encoding="utf-8"))
    require(manifest.get("status") == "SEALED" and manifest.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256, "archive_seal_binding_mismatch")
    require(manifest.get("archive_bytes") == PART_PATH.stat().st_size == 1_744_412 and manifest.get("member_count") == 1606, "archive_size_or_member_count_mismatch")
    require(manifest.get("track_all_files_sha256") == EXPECTED_TRACK_ALL_SHA256 and manifest.get("track_seal_sha256") == EXPECTED_TRACK_SEAL_SHA256, "track_inventory_or_seal_binding_mismatch")
    require(report.get("status") == "PASS" and report.get("scope", "").startswith("metadata, clocks, identities, sidecar and seal readback only"), "metadata_report_status_or_scope_mismatch")
    require(report.get("producer_bindings", {}).get("execution_plan_raw_sha256") == EXPECTED_PLAN_RAW_SHA256 and report.get("producer_bindings", {}).get("execution_plan_fingerprint_sha256") == EXPECTED_PLAN_FINGERPRINT_SHA256, "producer_plan_binding_mismatch")
    require(report.get("producer_bindings", {}).get("grid_inventory_sha256") == EXPECTED_GRID_SHA256 and report.get("producer_bindings", {}).get("extractor_source_sha256") == EXPECTED_EXTRACTOR_SHA256 and report.get("producer_bindings", {}).get("visual_consumer_source_sha256") == EXPECTED_VISUAL_CONSUMER_SHA256, "producer_code_or_grid_binding_mismatch")
    require(report.get("producer_bindings", {}).get("visual_availability_freeze_sha256") == EXPECTED_FREEZE_SHA256 and report.get("producer_bindings", {}).get("visual_availability_freeze_seal_sha256") == EXPECTED_FREEZE_SEAL_SHA256, "availability_freeze_binding_mismatch")
    require(report.get("visual_availability_freeze", {}).get("status") == "FROZEN_BEFORE_GEOMETRY_READ" and report["visual_availability_freeze"].get("contexts") == 160, "availability_freeze_scope_mismatch")
    require(freeze.get("status") == "FROZEN_BEFORE_GEOMETRY_READ" and freeze_seal.get("freeze_sha256") == EXPECTED_FREEZE_SHA256, "availability_freeze_seal_status_mismatch")
    require(report.get("coverage") == {"all_files_entries": 3202, "complete_progress_rows": 1600, "contexts": 160, "failed_rows": 0, "overlay_samples": ["S0064_BAC009S0064W0121"], "physical_overlay_contexts": 2, "samples": 80, "sidecars_verified": 1600, "sidecars_with_external_hash_match": 1600, "speakers": 40, "videos": 1600}, "track_coverage_mismatch")
    require(report.get("member_hash_verification", {}).get("verified_all_files_entries") == 3202 and report["member_hash_verification"].get("npz_payloads_opened") is False, "opaque_npz_readback_scope_mismatch")
    require(report.get("track_seals", {}).get("all_files_sha256") == EXPECTED_TRACK_ALL_SHA256 and report["track_seals"].get("seal_sha256") == EXPECTED_TRACK_SEAL_SHA256, "track_seal_inventory_mismatch")

    freeze_contexts = {(r["sample_id"], r["direction"]): r for r in freeze["context_rows"]}
    require(len(freeze_contexts) == 160, "freeze_context_coverage_mismatch")
    members = report.get("source_member_manifest")
    require(isinstance(members, dict) and len(members) == 1605, "source_member_manifest_count_mismatch")
    # The report includes the complete member map except its own self-reference;
    # the bundle manifest includes the report's independently measured bytes/hash.
    full_members = dict(members)
    full_members["track_metadata_inventory.json"] = {"bytes": REPORT_PATH.stat().st_size, "sha256": EXPECTED_REPORT_SHA256}
    require(manifest.get("source_member_manifest_sha256") == sha_bytes(canonical(full_members)), "archive_source_member_manifest_sha_mismatch")

    video_rows = report.get("video_metadata_rows")
    require(isinstance(video_rows, list) and len(video_rows) == 1600, "metadata_video_row_count_mismatch")
    seen_video_keys = set()
    context_counts: Counter[tuple[str, str]] = Counter()
    speaker_for_sample: dict[str, str] = {}
    expected_sidecars = set()
    for row in video_rows:
        sample, direction, visual_id = str(row["sample_id"]), str(row["direction"]), str(row["visual_id"])
        key = (sample, direction, visual_id)
        require(key not in seen_video_keys and visual_id in VISUAL_IDS, "duplicate_or_unknown_metadata_video_ref")
        seen_video_keys.add(key)
        context_counts[(sample, direction)] += 1
        prior = speaker_for_sample.setdefault(sample, str(row["speaker_id"]))
        require(prior == row["speaker_id"], "metadata_speaker_identity_mismatch")
        ctx = freeze_contexts[(sample, direction)]
        fr = ctx["feature_refs"][visual_id]
        require(row["row_id"] == fr["row_id"] and row["video_sha256"] == fr["video_sha256"] and row["feature_sha256"] == fr["feature_sha256"] and row["feature_metadata_sha256"] == fr["feature_metadata_sha256"], "metadata_feature_freeze_identity_mismatch")
        require(row["decoded_frame_count"] == fr["frame_count"] and row["valid_frames"] == fr["valid_frames"], "metadata_feature_clock_count_mismatch")
        require(row["recipient_grid_sha256"] == ctx["effective_grid"]["sha256"] and row["recipient_grid_path"] == ctx["effective_grid"]["path"], "metadata_grid_context_mismatch")
        require(row["recipient_grid_overlay_used"] == ctx["effective_grid"]["overlay_used"], "metadata_overlay_context_mismatch")
        intervals = ctx["phone_intervals"]
        require(row["phone_interval_count"] == len(intervals) and row["freeze_phone_intervals_sha256"] == sha_bytes(canonical(intervals)), "metadata_phone_interval_freeze_mismatch")
        require(row["frozen_common_clock"] == ctx["common_clock"] and row["frozen_phone_class_counts"] == ctx["phone_class_counts"], "metadata_frozen_support_mismatch")
        arrays = row["array_schemas"]
        require(set(arrays) == {"landmarks", "canonical_mouth", "valid", "mouth_open", "mouth_width", "mouth_height", "mouth_area", "timestamps_s"}, "metadata_array_schema_coverage_mismatch")
        require(arrays["timestamps_s"]["sha256"] == fr["timestamp_sha256"] and arrays["valid"]["sha256"] == fr["valid_mask_sha256"], "metadata_track_clock_digest_mismatch")
        expected_sidecars.add(f"tracks/{sample}__{direction}__{visual_id}.json")
    require(len(context_counts) == 160 and set(context_counts.values()) == {10} and len(speaker_for_sample) == 80 and len(set(speaker_for_sample.values())) == 40, "metadata_context_or_speaker_coverage_mismatch")
    expected_members = set(members)
    sidecar_paths = {path for path in expected_members if path.startswith("tracks/") and path.endswith(".json")}
    require(sidecar_paths == expected_sidecars and len(sidecar_paths) == 1600, "member_sidecar_coverage_mismatch")

    tar_member_hashes = {}
    with tarfile.open(PART_PATH, mode="r:gz") as archive:
        for item in archive.getmembers():
            require(item.isfile() and not item.issym() and not item.islnk() and item.name not in tar_member_hashes, f"unsafe_or_duplicate_tar_member:{item.name}")
            stream = archive.extractfile(item)
            require(stream is not None, f"tar_member_unreadable:{item.name}")
            data = stream.read()
            tar_member_hashes[item.name] = {"bytes": len(data), "sha256": sha_bytes(data)}
            expected = full_members.get(item.name)
            require(expected == tar_member_hashes[item.name], f"tar_member_hash_mismatch:{item.name}")
            if item.name.endswith(".json") and item.name.startswith("tracks/"):
                sidecar = json.loads(data.decode("utf-8"))
                sample, direction, visual_id = item.name.removeprefix("tracks/").removesuffix(".json").split("__")
                row = next(r for r in video_rows if (r["sample_id"], r["direction"], r["visual_id"]) == (sample, direction, visual_id))
                require(sidecar.get("status") == "complete" and sidecar.get("row_id") == row["row_id"], f"tar_sidecar_identity_mismatch:{item.name}")
                require(sidecar.get("video_sha256") == row["video_sha256"] and sidecar.get("feature_sha256") == row["feature_sha256"], f"tar_sidecar_video_or_npz_binding_mismatch:{item.name}")
                require(sidecar.get("recipient_grid_sha256") == row["recipient_grid_sha256"] and sidecar.get("recipient_phone_intervals_count") == row["phone_interval_count"], f"tar_sidecar_grid_or_phone_count_mismatch:{item.name}")
                desc = sidecar.get("arrays", {})
                require({name: {"shape": v["shape"], "dtype": v["dtype"], "sha256": v["sha256"]} for name, v in desc.items()} == row["array_schemas"], f"tar_sidecar_array_descriptor_mismatch:{item.name}")
    require(set(tar_member_hashes) == set(full_members) and len(tar_member_hashes) == 1606, "tar_member_coverage_mismatch")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "metadata-only archive and availability-freeze join; all NPZ files remain opaque",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "report_sha256": EXPECTED_REPORT_SHA256,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "archive_bytes": manifest["archive_bytes"],
        "archive_members_rehashed": len(tar_member_hashes),
        "track_inventory_entries_bound": 3202,
        "sidecars_rehashed_and_joined": len(expected_sidecars),
        "contexts": len(context_counts),
        "samples": len(speaker_for_sample),
        "speakers": len(set(speaker_for_sample.values())),
        "videos": len(seen_video_keys),
        "overlay_contexts": 2,
        "npz_payloads_opened": False,
    }
    out = BASE / "world_visual_metadata_bundle_independent_readback_attempt01.json"
    out.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
