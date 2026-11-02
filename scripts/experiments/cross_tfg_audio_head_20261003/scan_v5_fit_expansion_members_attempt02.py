#!/usr/bin/env python3
"""Verify the ADR013 j=0/j=15 fit subset against its 20 pinned AISHELL archives.

Reads only the 20 fit-speaker archive files. It hashes each pinned archive,
scans WAV member names and RIFF headers to form the frozen 16-slot grid, and
reads sample payloads only for the 40 planned ADR013 endpoint members. No
replacement selection, transcript, dev/confirm audio, model, render, or score
is accessed. Output contains hashes and metadata, never copied waveforms.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import struct
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

ROSTER_SHA256 = "2fd42f097dbfb046c5bb5a2fba687c1812dc656fb06bf489974384f4eb78667a"
PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
ADR_SHA256 = "82a8a397f2317148d026ba4a2e0558b81656df0f1f74dcc4e58015b0323ab04b"
MIN_SAMPLES = 51_200
MAX_SAMPLES = 128_000
EXPECTED_RATE = 16_000
EXPECTED_CHANNELS = 1


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            total += len(block)
    return h.hexdigest(), total


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ValueError("truncated WAV member")
    return b


def parse_wav_header(f: BinaryIO, file_size: int, check_float_finite: bool) -> dict[str, Any]:
    head = _read_exact(f, 12)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a little-endian RIFF/WAVE file")
    riff_size = struct.unpack_from("<I", head, 4)[0]
    if riff_size + 8 > file_size or riff_size < 4:
        raise ValueError("invalid RIFF declared size")
    fmt = None
    data_chunks: list[tuple[int, int]] = []
    pos = 12
    while pos + 8 <= min(riff_size + 8, file_size):
        f.seek(pos)
        chunk_head = _read_exact(f, 8)
        chunk_id = chunk_head[:4]
        size = struct.unpack_from("<I", chunk_head, 4)[0]
        payload = pos + 8
        next_pos = payload + size + (size & 1)
        if next_pos > riff_size + 8 or next_pos > file_size:
            raise ValueError("WAV chunk exceeds declared RIFF/file bounds")
        if chunk_id == b"fmt ":
            if size < 16 or size > 4096:
                raise ValueError("invalid/oversized WAV fmt chunk")
            f.seek(payload)
            b = _read_exact(f, size)
            code, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", b, 0)
            if code == 0xFFFE and len(b) >= 40:
                code = struct.unpack_from("<H", b, 24)[0]
            fmt = {
                "format_code": code,
                "channels": channels,
                "sample_rate": rate,
                "byte_rate": byte_rate,
                "block_align": block_align,
                "bits_per_sample": bits,
            }
        elif chunk_id == b"data":
            data_chunks.append((payload, size))
        pos = next_pos
    if fmt is None or not data_chunks:
        raise ValueError("WAV missing fmt or data chunk")
    block_align = fmt["block_align"]
    if block_align <= 0 or any(size % block_align for _, size in data_chunks):
        raise ValueError("WAV data byte count is inconsistent with block alignment")
    frames = sum(size // block_align for _, size in data_chunks)
    code = fmt["format_code"]
    finite: bool | None
    if code in (1, 6, 7):
        finite = True
    elif code == 3 and fmt["bits_per_sample"] in (32, 64):
        finite = True
        if check_float_finite:
            scalar_fmt = "<f" if fmt["bits_per_sample"] == 32 else "<d"
            scalar_bytes = 4 if fmt["bits_per_sample"] == 32 else 8
            for data_start, size in data_chunks:
                f.seek(data_start)
                remain = size
                carry = b""
                while remain:
                    block = f.read(min(1 << 16, remain))
                    if not block:
                        raise ValueError("truncated float WAV data")
                    remain -= len(block)
                    block = carry + block
                    usable = len(block) - (len(block) % scalar_bytes)
                    for (value,) in struct.iter_unpack(scalar_fmt, block[:usable]):
                        if not math.isfinite(value):
                            finite = False
                            break
                    if finite is False:
                        break
                    carry = block[usable:]
                if finite is False or carry:
                    if carry:
                        raise ValueError("float WAV data has incomplete scalar")
                    break
    else:
        finite = None
    return {
        **fmt,
        "sample_count": frames,
        "data_chunks": [{"offset": a, "bytes": n} for a, n in data_chunks],
        "finite": finite,
        "finite_check": "all_float_samples_scanned" if code == 3 and check_float_finite else (
            "integer_or_companded_sample_domain_is_finite" if finite is True else "unsupported_encoding_unverified"
        ),
    }


def in_scope_wav(member_name: str, speaker_id: str) -> bool:
    p = PurePosixPath(member_name)
    return (
        p.suffix.lower() == ".wav"
        and len(p.parts) >= 3
        and p.parts[0] == "train"
        and p.parts[1] == speaker_id
        and ".." not in p.parts
        and not p.is_absolute()
    )


def read_archive_headers(archive_path: Path, speaker_id: str) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    wav_count = 0
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for info in tar:
            if not info.isfile() or not in_scope_wav(info.name, speaker_id):
                continue
            wav_count += 1
            fh = tar.extractfile(info)
            if fh is None:
                rows.append({"member_path": info.name, "member_size": info.size, "header_status": "UNREADABLE_MEMBER"})
                continue
            try:
                wav = parse_wav_header(fh, info.size, check_float_finite=True)
                status = "HEADER_OK"
                reason = []
                if wav["sample_rate"] != EXPECTED_RATE:
                    reason.append("SAMPLE_RATE")
                if wav["channels"] != EXPECTED_CHANNELS:
                    reason.append("CHANNELS")
                if not MIN_SAMPLES <= wav["sample_count"] <= MAX_SAMPLES:
                    reason.append("SAMPLE_COUNT")
                if wav["finite"] is not True:
                    reason.append("FINITE_UNVERIFIED_OR_FALSE")
                eligible = not reason
                rows.append({
                    "member_path": info.name,
                    "member_size": info.size,
                    "sample_id": PurePosixPath(info.name).stem,
                    "header_status": status,
                    "eligible": eligible,
                    "eligibility_reasons": reason,
                    "wav": wav,
                })
            except (ValueError, OSError, struct.error) as e:
                rows.append({"member_path": info.name, "member_size": info.size, "header_status": "INVALID_WAV", "eligible": False, "eligibility_reasons": [type(e).__name__]})
            finally:
                fh.close()
    return rows, wav_count


def select_slots(eligible: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    eligible = sorted(eligible, key=lambda x: x["member_path"])
    if not eligible:
        return [None] * 16
    n = len(eligible)
    return [eligible[(j * (n - 1)) // 15] for j in range(16)]


def selected_pcm_hashes(archive_path: Path, wanted_paths: set[str]) -> dict[str, dict[str, Any]]:
    results = {}
    if not wanted_paths:
        return results
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for info in tar:
            if not info.isfile() or info.name not in wanted_paths:
                continue
            fh = tar.extractfile(info)
            if fh is None:
                raise ValueError(f"selected WAV cannot be opened: {info.name}")
            raw = fh.read()
            fh.close()
            wav = parse_wav_header(io.BytesIO(raw), len(raw), check_float_finite=True)
            pcm_hash = hashlib.sha256()
            for chunk in wav["data_chunks"]:
                start, size = chunk["offset"], chunk["bytes"]
                pcm_hash.update(raw[start:start + size])
            results[info.name] = {
                "source_member_sha256": hashlib.sha256(raw).hexdigest(),
                "source_member_bytes": len(raw),
                "pcm_sha256": pcm_hash.hexdigest(),
                "wav": wav,
            }
            if wanted_paths == set(results):
                break
    missing = wanted_paths - set(results)
    if missing:
        raise ValueError(f"selected members disappeared on second pass: {sorted(missing)}")
    return results


def scan(roster_path: Path, archive_root: Path) -> dict[str, Any]:
    roster_raw = roster_path.read_bytes()
    roster_sha = hashlib.sha256(roster_raw).hexdigest()
    if roster_sha != ROSTER_SHA256:
        raise ValueError(f"roster seal mismatch: {roster_sha}")
    roster = json.loads(roster_raw.decode("utf-8"))
    if roster.get("schema") != "cross_tfg_v5_fit_expansion_roster_adr013_v1":
        raise ValueError("unexpected roster schema")
    pins = roster["upstream_archive_pins"]
    slots_in = roster["canonical_roster"]["slots"]
    if len(slots_in) != 320 or len(pins) != 20:
        raise ValueError("frozen expected 320 slots and 20 archive pins")
    if roster["adr013_endpoint_subset"]["slot_keys"] != [[sp, j] for sp in roster["split_binding"]["fit_speaker_ids"][:20] for j in (0, 15)]:
        raise ValueError("ADR013 endpoint roster mismatch")

    archive_receipts = []
    speaker_members: dict[str, list[dict[str, Any]]] = {}
    for pin in pins:
        sp = pin["speaker_id"]
        path = archive_root / f"{sp}.tar.gz"
        st = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(st.st_mode):
            raise ValueError(f"archive must be regular non-symlink: {path}")
        if st.st_size != pin["bytes"]:
            raise ValueError(f"archive size mismatch for {sp}: {st.st_size}")
        digest, size = sha256_file(path)
        if digest != pin["lfs_sha256"] or size != pin["bytes"]:
            raise ValueError(f"archive bytes do not match official LFS pin for {sp}")
        members, wav_count = read_archive_headers(path, sp)
        # Source member basename collisions are explicit and remain in the denominator.
        ids = [m.get("sample_id") for m in members if m.get("sample_id")]
        dup_ids = {sid for sid, count in Counter(ids).items() if count > 1}
        for m in members:
            if m.get("sample_id") in dup_ids:
                m["eligible"] = False
                m.setdefault("eligibility_reasons", []).append("DUPLICATE_SAMPLE_ID_WITHIN_ARCHIVE")
        elig = [m for m in members if m.get("eligible") is True]
        speaker_members[sp] = members
        archive_receipts.append({
            "speaker_id": sp,
            "path": str(path),
            "official_archive_relpath": pin["path"],
            "expected_bytes": pin["bytes"],
            "actual_bytes": size,
            "official_lfs_sha256": pin["lfs_sha256"],
            "actual_sha256": digest,
            "archive_hash_status": "PASS",
            "wav_member_count": wav_count,
            "eligible_member_count": len(elig),
            "invalid_or_ineligible_member_count": len(members) - len(elig),
        })

    # Map all 320 canonical slots, without replacements. Payload SHA checks are limited to the 40 ADR013 endpoints.
    slot_rows: list[dict[str, Any]] = []
    selected_paths_by_speaker: dict[str, set[str]] = defaultdict(set)
    maps: dict[str, list[dict[str, Any] | None]] = {}
    for sp in [p["speaker_id"] for p in pins]:
        elig = [m for m in speaker_members[sp] if m.get("eligible") is True]
        maps[sp] = select_slots(elig)
        for j, chosen in enumerate(maps[sp]):
            is_endpoint = j in (0, 15)
            if is_endpoint and chosen is not None:
                selected_paths_by_speaker[sp].add(chosen["member_path"])
            slot_rows.append({
                "speaker_id": sp,
                "canonical_j": j,
                "selected_by_adr013": is_endpoint,
                "eligible_pool_count": len(elig),
                "selected_member_path": chosen["member_path"] if chosen else None,
                "sample_id": chosen.get("sample_id") if chosen else None,
                "header_eligibility": "ELIGIBLE" if chosen else "NO_ELIGIBLE_UTTERANCE_POOL",
                "member_payload_audit_scope": "ENDPOINT_PCM_HASHED" if is_endpoint and chosen else ("MAPPING_ONLY_PAYLOAD_NOT_READ" if chosen else "NO_MEMBER"),
                "replacement": False,
            })

    payload_by_member: dict[str, dict[str, Any]] = {}
    for pin in pins:
        sp = pin["speaker_id"]
        payload_by_member.update(selected_pcm_hashes(archive_root / f"{sp}.tar.gz", selected_paths_by_speaker[sp]))

    seen_pcm: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for slot in slot_rows:
        if not slot["selected_by_adr013"] or not slot["selected_member_path"]:
            slot["endpoint_payload_status"] = "MISSING_PLANNED_SLOT_NO_REPLACEMENT"
            slot["duplicate_pcm_status"] = "NOT_CHECKABLE"
            continue
        audit = payload_by_member[slot["selected_member_path"]]
        slot.update({
            "source_member_sha256": audit["source_member_sha256"],
            "source_member_bytes": audit["source_member_bytes"],
            "pcm_sha256": audit["pcm_sha256"],
            "endpoint_payload_status": "PASS",
        })
        wav = audit["wav"]
        key = (wav["sample_rate"], wav["channels"], wav["format_code"], wav["bits_per_sample"], wav["sample_count"], audit["pcm_sha256"])
        seen_pcm[key].append(f"{slot['speaker_id']}|j={slot['canonical_j']}|{slot['sample_id']}")
        slot["duplicate_pcm_status"] = "PENDING"
    duplicate_slots = {sid for vals in seen_pcm.values() if len(vals) > 1 for sid in vals}
    endpoint_sample_ids: dict[str, list[str]] = defaultdict(list)
    for slot in slot_rows:
        if slot["selected_by_adr013"] and slot.get("sample_id"):
            endpoint_sample_ids[slot["sample_id"]].append(
                f"{slot['speaker_id']}|j={slot['canonical_j']}|{slot['sample_id']}"
            )
    duplicate_sample_id_slots = {
        label for labels in endpoint_sample_ids.values() if len(labels) > 1 for label in labels
    }
    for slot in slot_rows:
        if not slot["selected_by_adr013"] or slot.get("endpoint_payload_status") != "PASS":
            continue
        label = f"{slot['speaker_id']}|j={slot['canonical_j']}|{slot['sample_id']}"
        slot["duplicate_pcm_status"] = "DUPLICATE" if label in duplicate_slots else "UNIQUE_AMONG_40_PLANNED_ENDPOINTS"
        slot["duplicate_sample_id_status"] = "DUPLICATE" if label in duplicate_sample_id_slots else "UNIQUE_AMONG_40_PLANNED_ENDPOINTS"

    endpoint_rows = [x for x in slot_rows if x["selected_by_adr013"]]
    pass_rows = [x for x in endpoint_rows if x.get("endpoint_payload_status") == "PASS" and x.get("duplicate_pcm_status") == "UNIQUE_AMONG_40_PLANNED_ENDPOINTS" and x.get("duplicate_sample_id_status") == "UNIQUE_AMONG_40_PLANNED_ENDPOINTS"]
    speaker_endpoint_counts = {
        sp: sum(1 for x in pass_rows if x["speaker_id"] == sp)
        for sp in [p["speaker_id"] for p in pins]
    }
    status = "PASS_40_UNIQUE_ENDPOINTS" if len(pass_rows) == 40 and all(v == 2 for v in speaker_endpoint_counts.values()) else "INCOMPLETE_NO_REPLACEMENT"
    return {
        "schema": "cross_tfg_v5_fit_expansion_member_scan_adr013_v1",
        "status": status,
        "roster_raw_sha256": roster_sha,
        "plan_raw_sha256": PLAN_SHA256,
        "adr013_raw_sha256": ADR_SHA256,
        "selection_rule": "20 first frozen fit speakers; preserve all 320 16-slot grid positions; ADR013 evaluates only j=0 and j=15; sorted lexical eligible WAV member paths; index=floor(j*(n_eligible-1)/15); no replacements.",
        "eligibility_rule": {"sample_rate": EXPECTED_RATE, "channels": EXPECTED_CHANNELS, "minimum_samples_inclusive": MIN_SAMPLES, "maximum_samples_inclusive": MAX_SAMPLES, "finite": True},
        "archive_receipts": archive_receipts,
        "slot_rows": slot_rows,
        "summary": {
            "planned_grid_slots": len(slot_rows),
            "planned_adr013_endpoints": len(endpoint_rows),
            "endpoint_payload_pass_count": sum(x.get("endpoint_payload_status") == "PASS" for x in endpoint_rows),
            "endpoint_unique_pcm_count": len(pass_rows),
            "missing_planned_endpoints": sum(x.get("endpoint_payload_status") != "PASS" for x in endpoint_rows),
            "duplicate_endpoint_pcm_groups": [vals for vals in seen_pcm.values() if len(vals) > 1],
            "duplicate_endpoint_sample_id_groups": [labels for labels in endpoint_sample_ids.values() if len(labels) > 1],
            "endpoint_rows_per_speaker": speaker_endpoint_counts,
            "replacement_count": 0,
        },
        "scope": {
            "archives_opened": True,
            "only_frozen_fit_speakers_S0052_S0071": True,
            "archive_member_names_and_wav_headers_scanned": True,
            "full_archive_file_bytes_rehashed_against_official_lfs_sha256": True,
            "non_float_wav_payload_read_during_pool_eligibility_scan": False,
            "float_wav_samples_scanned_for_finiteness_when_present": True,
            "full_selected_member_bytes_hashed_only_for_planned_40_endpoints": True,
            "dev_audio_read": False,
            "confirm_audio_read": False,
            "transcript_read": False,
            "model_or_effect_read": False,
            "render_or_score": False,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--archive-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = scan(args.roster, args.archive_root)
    args.out.parent.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    args.out.write_bytes(raw)
    print(json.dumps({"status": report["status"], "out": str(args.out), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "summary": report["summary"]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
