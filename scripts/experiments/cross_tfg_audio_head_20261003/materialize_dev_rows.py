#!/usr/bin/env python3
"""Materialize the frozen AISHELL development PCM rows.

Only the ten already-acquired development-speaker archives are read.  The
script scans WAV members in lexical basename order and emits the first two
members per speaker that satisfy the frozen source eligibility rule.  It does
not read transcripts, confirmation archives, model outputs, or scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.audio_enhancement.audio_io import (
    _decode_wave_samples,
    _parse_wave,
    canonicalize_samples,
    pcm_float32_sha256,
    write_float32_wav,
)


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_PROTOCOL_RAW_SHA256 = "cf615529edc8ae0b6ffff0a34246db71b3ec8ce5ac17d9f0daf2618ac624cf87"
EXPECTED_PACK_MANIFEST_RAW_SHA256 = "ead6a21a3a2f2ce0d4481eb5df48dd4194dbf23e10eeac7652920da24707b84d"
EXPECTED_PACK_MANIFEST_CANONICAL_SHA256 = "d11eb42e2d0a9f32c4d7d42f0befbacdf222d853de6414db8d7bc6d9b151e9a1"
EXPECTED_REVISION = "bbe295d530192a4cd41644b711c9aecd087df653"
EXPECTED_DEV_SPEAKERS = [
    "S0005", "S0022", "S0024", "S0037", "S0038",
    "S0039", "S0041", "S0044", "S0049", "S0050",
]
SAMPLE_RATE = 16_000
MIN_SAMPLES = 51_200
MAX_SAMPLES = 128_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json_and_sha(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, sha256_bytes(raw)


def _validate_contract(protocol_path: Path, pack_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    protocol, protocol_raw = _read_json_and_sha(protocol_path)
    pack, pack_raw = _read_json_and_sha(pack_manifest_path)
    if protocol_raw != EXPECTED_PROTOCOL_RAW_SHA256:
        raise ValueError(f"frozen data protocol raw SHA mismatch: {protocol_raw}")
    if pack_raw != EXPECTED_PACK_MANIFEST_RAW_SHA256:
        raise ValueError(f"development pack manifest raw SHA mismatch: {pack_raw}")
    if pack.get("manifest_sha256") != EXPECTED_PACK_MANIFEST_CANONICAL_SHA256:
        raise ValueError("development pack manifest embedded canonical SHA mismatch")
    if canonical_sha(pack) != EXPECTED_PACK_MANIFEST_CANONICAL_SHA256:
        raise ValueError("development pack manifest canonical self-hash mismatch")
    if protocol.get("status") != "ROSTER_FROZEN_ROWS_PENDING":
        raise ValueError("data protocol is not the frozen pre-effects roster")
    if protocol.get("selection_before_any_new_effects") is not True:
        raise ValueError("speaker/utterance selection was not frozen before effects")
    if protocol.get("dataset_revision") != EXPECTED_REVISION:
        raise ValueError("protocol dataset revision differs from the acquisition")
    if protocol.get("dev_speaker_ids") != EXPECTED_DEV_SPEAKERS:
        raise ValueError("development speaker roster/order differs from the frozen roster")
    if protocol.get("clip_selection", {}).get("order") != "lexical source utterance basename":
        raise ValueError("unexpected frozen development utterance ordering rule")
    eligibility = protocol.get("clip_eligibility", {})
    expected_eligibility = {
        "minimum_samples": MIN_SAMPLES,
        "maximum_samples": MAX_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "source_extension": "wav",
        "finite": True,
        "speaker_matches_archive": True,
    }
    for key, expected in expected_eligibility.items():
        if eligibility.get(key) != expected:
            raise ValueError(f"frozen eligibility mismatch for {key}")
    if pack.get("schema") != "cross_tfg_dev_archive_acquisition_v1" or pack.get("status") != "COMPLETE":
        raise ValueError("development archive acquisition is not COMPLETE")
    if pack.get("dataset") != "AISHELL/AISHELL-1" or pack.get("revision") != EXPECTED_REVISION:
        raise ValueError("development archive acquisition identity mismatch")
    if pack.get("dev_speaker_ids") != EXPECTED_DEV_SPEAKERS:
        raise ValueError("acquisition manifest roster/order mismatch")
    if pack.get("confirm_speaker_ids_read") is not False or pack.get("audio_samples_read_or_decoded") is not False:
        raise ValueError("acquisition manifest unexpectedly reports audio/confirmation reads")
    records = pack.get("records")
    if not isinstance(records, list) or [row.get("speaker_id") for row in records] != EXPECTED_DEV_SPEAKERS:
        raise ValueError("development archive records are incomplete or reordered")
    return protocol, pack, protocol_raw, pack_raw


def _verify_archive_row(row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    speaker = row["speaker_id"]
    path = Path(row["path"])
    if row.get("split") != "dev" or row.get("status") not in {"REUSED_VERIFIED", "DOWNLOADED_VERIFIED"}:
        raise ValueError(f"unexpected acquisition status for {speaker}")
    expected_sha = str(row.get("sha256", "")).lower()
    if expected_sha != str(row.get("publisher_lfs_sha256", "")).lower():
        raise ValueError(f"archive is not bound to publisher LFS SHA for {speaker}")
    if not path.is_file():
        raise FileNotFoundError(f"development archive missing for {speaker}: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(row.get("bytes", -1)) or digest != expected_sha:
        raise ValueError(f"development archive size/SHA mismatch for {speaker}")
    return path, {"speaker_id": speaker, "path": str(path.resolve()), "bytes": size, "sha256": digest}


def _safe_wav_members(archive: tarfile.TarFile, speaker_id: str) -> list[tarfile.TarInfo]:
    candidates: list[tarfile.TarInfo] = []
    seen_names: set[str] = set()
    for member in archive.getmembers():
        if not member.isfile():
            continue
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"unsafe archive member path in {speaker_id}")
        if member.name in seen_names:
            raise ValueError(f"duplicate archive member path in {speaker_id}")
        seen_names.add(member.name)
        if pure.suffix.lower() != ".wav":
            continue
        if pure.parent.name != speaker_id:
            continue
        candidates.append(member)
    return sorted(candidates, key=lambda member: PurePosixPath(member.name).name)


def _inspect_member(archive: tarfile.TarFile, member: tarfile.TarInfo, speaker_id: str) -> dict[str, Any]:
    source = archive.extractfile(member)
    if source is None:
        return {
            "member_path": member.name,
            "member_bytes": int(member.size),
            "status": "INELIGIBLE",
            "reason": "member_not_readable",
        }
    raw = source.read()
    source.close()
    source_sha = sha256_bytes(raw)
    record: dict[str, Any] = {
        "member_path": member.name,
        "member_bytes": len(raw),
        "member_sha256": source_sha,
    }
    try:
        fmt_code, channels, sample_rate, bits, block_align, payload = _parse_wave(raw)
        decoded = _decode_wave_samples(fmt_code, channels, bits, block_align, payload)
    except Exception as exc:
        record.update({"status": "INELIGIBLE", "reason": f"invalid_wav:{type(exc).__name__}"})
        return record
    source_frames = int(decoded.shape[0])
    record.update(
        {
            "source_sample_rate_hz": int(sample_rate),
            "source_channels": int(channels),
            "source_sample_count": source_frames,
            "source_encoding": f"{'PCM' if fmt_code == 1 else 'FLOAT'}_{bits}",
        }
    )
    if PurePosixPath(member.name).parent.name != speaker_id:
        record.update({"status": "INELIGIBLE", "reason": "speaker_path_mismatch"})
    elif sample_rate != SAMPLE_RATE:
        record.update({"status": "INELIGIBLE", "reason": "sample_rate_mismatch"})
    elif channels != 1:
        record.update({"status": "INELIGIBLE", "reason": "channel_count_mismatch"})
    elif not (MIN_SAMPLES <= source_frames <= MAX_SAMPLES):
        record.update({"status": "INELIGIBLE", "reason": "sample_count_out_of_range"})
    else:
        record["status"] = "ELIGIBLE"
    if record["status"] == "ELIGIBLE":
        record["canonical_samples"] = canonicalize_samples(decoded, int(sample_rate))
    return record


def materialize(
    protocol_path: Path,
    pack_manifest_path: Path,
    output_dir: Path,
    *,
    peak_ceiling: float | None = None,
) -> dict[str, Any]:
    protocol, pack, protocol_raw, pack_raw = _validate_contract(protocol_path, pack_manifest_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse an existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    all_rows: list[dict[str, Any]] = []
    checked_archives: list[dict[str, Any]] = []
    selection_audits: list[dict[str, Any]] = []
    pcm_owner: dict[str, str] = {}
    for roster_index, archive_row in enumerate(pack["records"]):
        speaker_id = archive_row["speaker_id"]
        archive_path, archive_seal = _verify_archive_row(archive_row)
        checked_archives.append(archive_seal)
        selected: list[dict[str, Any]] = []
        scanned: list[dict[str, Any]] = []
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = _safe_wav_members(archive, speaker_id)
            for member in members:
                inspected = _inspect_member(archive, member, speaker_id)
                scanned.append({key: value for key, value in inspected.items() if key != "canonical_samples"})
                if inspected.get("status") == "ELIGIBLE":
                    selected.append(inspected)
                    if len(selected) == 2:
                        break
        selection_audits.append(
            {
                "speaker_id": speaker_id,
                "archive_sha256": archive_seal["sha256"],
                "wav_member_count_in_archive": len(members),
                "members_scanned_in_lexical_order": scanned,
                "selected_member_paths": [row["member_path"] for row in selected],
                "first_two_eligible_proven": len(selected) == 2,
            }
        )
        for speaker_rank, source in enumerate(selected):
            samples = source.pop("canonical_samples")
            pcm_sha = pcm_float32_sha256(samples)
            if pcm_sha in pcm_owner:
                raise ValueError(
                    f"duplicate development PCM detected: {pcm_owner[pcm_sha]} and {speaker_id}/{source['member_path']}"
                )
            pcm_owner[pcm_sha] = f"{speaker_id}/{source['member_path']}"
            utterance = PurePosixPath(source["member_path"]).stem
            relative_pcm_path = Path("pcm") / speaker_id / f"{utterance}.wav"
            output_path = output_dir / relative_pcm_path
            write_info = write_float32_wav(
                output_path,
                samples,
                sample_rate=SAMPLE_RATE,
                peak_ceiling=peak_ceiling,
                overwrite=False,
            )
            natural_pcm = {
                "path": str(output_path.resolve()),
                "file_sha256": write_info.file_sha256,
                "pcm_sha256": write_info.pcm_float32_sha256,
                "sample_rate_hz": write_info.sample_rate,
                "channels": write_info.channels,
                "sample_count": write_info.frames,
                "subtype": write_info.subtype,
            }
            all_rows.append(
                {
                    "ordered_index": len(all_rows),
                    "roster_index": roster_index,
                    "speaker_utterance_order": speaker_rank,
                    "row_id": f"dev:{speaker_id}:{utterance}",
                    "split": "dev",
                    "sample_id": utterance,
                    "speaker_id": speaker_id,
                    "dataset": "AISHELL/AISHELL-1",
                    "dataset_revision": EXPECTED_REVISION,
                    "historical_exposed": False,
                    "source_archive": archive_seal,
                    "source_member": {
                        "path": source["member_path"],
                        "bytes": source["member_bytes"],
                        "file_sha256": source["member_sha256"],
                        "sample_rate_hz": source["source_sample_rate_hz"],
                        "channels": source["source_channels"],
                        "sample_count": source["source_sample_count"],
                        "encoding": source["source_encoding"],
                    },
                    "natural_pcm": natural_pcm,
                    "score_audio": dict(natural_pcm),
                    "selection_rule": "first_two_lexical_basename_ordered_eligible_wav_members",
                    "eligibility": {
                        "status": "PASS",
                        "sample_rate_hz": SAMPLE_RATE,
                        "channels": 1,
                        "minimum_samples_inclusive": MIN_SAMPLES,
                        "maximum_samples_inclusive": MAX_SAMPLES,
                        "finite": True,
                    },
                    "canonicalization": {
                        "downmix": "none_source_is_mono",
                        "resampler": "none_source_is_16000_hz",
                        "gain_or_peak_normalization": "none",
                        "float_conversion": "decoded_source_samples_to_ieee_float32_little_endian",
                        "peak_ceiling": peak_ceiling,
                        "readback_bit_exact": True,
                    },
                }
            )
    status = "COMPLETE" if len(all_rows) == 20 and all(audit["first_two_eligible_proven"] for audit in selection_audits) else "INCOMPLETE"
    manifest: dict[str, Any] = {
        "schema": "cross_tfg_dev_pcm_rows_v1",
        "status": status,
        "split": "dev",
        "dataset": "AISHELL/AISHELL-1",
        "dataset_revision": EXPECTED_REVISION,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_raw_sha256": protocol_raw,
        "acquisition_manifest_path": str(pack_manifest_path.resolve()),
        "acquisition_manifest_raw_sha256": pack_raw,
        "acquisition_manifest_canonical_sha256": pack["manifest_sha256"],
        "frozen_clip_selection": protocol["clip_selection"]["dev"],
        "frozen_order_rule": protocol["clip_selection"]["order"],
        "frozen_eligibility": protocol["clip_eligibility"],
        "dev_speaker_ids": EXPECTED_DEV_SPEAKERS,
        "confirm_speaker_ids_read": False,
        "confirm_audio_read": False,
        "effects_read": False,
        "models_loaded": False,
        "scores_computed": False,
        "archive_seals": checked_archives,
        "selection_audits": selection_audits,
        "row_count": len(all_rows),
        "rows": all_rows,
    }
    manifest["manifest_sha256"] = canonical_sha(manifest)
    manifest_path = output_dir / "dev_pcm_rows_manifest.json"
    atomic_json(manifest_path, manifest)
    manifest_raw_sha = sha256_file(manifest_path)
    if status != "COMPLETE":
        raise RuntimeError(f"dev materialization {status}: {len(all_rows)}/20 rows; manifest={manifest_path}")
    return {
        "status": "PASS",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_raw_sha256": manifest_raw_sha,
        "manifest_canonical_sha256": manifest["manifest_sha256"],
        "row_count": len(all_rows),
        "speaker_count": len({row["speaker_id"] for row in all_rows}),
        "total_pcm_samples": sum(row["natural_pcm"]["sample_count"] for row in all_rows),
        "all_rows_natural_score_pcm_same": all(row["natural_pcm"] == row["score_audio"] for row in all_rows),
        "confirm_audio_read": False,
        "effects_read": False,
        "models_loaded": False,
        "scores_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "openspec/changes/develop-cross-tfg-audio-head-20261003/data-protocol.json")
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peak-ceiling", type=float, default=None)
    args = parser.parse_args()
    try:
        result = materialize(args.protocol, args.acquisition_manifest, args.output_dir, peak_ceiling=args.peak_ceiling)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)}, sort_keys=True, ensure_ascii=False), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
