#!/usr/bin/env python3
"""Verify frozen female Qwen pairs and build portable, hash-bound WAV shards.

This offline producer reads the frozen selection and Qwen output manifests,
re-decodes each natural, provider 24 kHz, and canonical 16 kHz WAV to EOF, and
packages only the natural and canonical 16 kHz PCM needed by the remote runner.
It never reads an API key or model/evaluation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf


MAX_SHARD_BYTES = 7_500_000
SHARD_TARGET_BYTES = 6_800_000
PROVIDER_RATE_HZ = 24_000
CANONICAL_RATE_HZ = 16_000
MODEL = "qwen3-tts-vc-2026-01-22"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def full_decode_audit(path: Path, block_frames: int = 65_536) -> dict[str, Any]:
    """Read PCM to EOF and keep RIFF's frame declaration separate."""
    try:
        with wave.open(str(path), "rb") as header:
            declared_frames: int | None = int(header.getnframes())
            riff_rate: int | None = int(header.getframerate())
            riff_channels: int | None = int(header.getnchannels())
    except (wave.Error, EOFError, OSError):
        declared_frames = None
        riff_rate = None
        riff_channels = None

    pcm_hash = hashlib.sha256()
    decoded_frames = 0
    with sf.SoundFile(str(path), mode="r") as stream:
        header_frames = int(stream.frames)
        sample_rate = int(stream.samplerate)
        channels = int(stream.channels)
        subtype = str(stream.subtype)
        while True:
            chunk = stream.read(block_frames, dtype="int16", always_2d=True)
            if not chunk.size:
                break
            decoded_frames += int(chunk.shape[0])
            pcm_hash.update(chunk.tobytes(order="C"))

    return {
        "declared_frames": declared_frames,
        "soundfile_header_frames": header_frames,
        "decoded_frames": decoded_frames,
        "decoded_pcm16_sha256": pcm_hash.hexdigest(),
        "declared_vs_payload_mismatch": (
            declared_frames is not None and declared_frames != decoded_frames
        ),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "riff_header_sample_rate_hz": riff_rate,
        "riff_header_channels": riff_channels,
        "subtype": subtype,
        "duration_s": decoded_frames / sample_rate if sample_rate else None,
    }


def resolve_file(value: str, *, manifest_dir: Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (manifest_dir, repo_root):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return (repo_root / path).resolve()


def usage_receipts(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    safe: dict[str, list[dict[str, Any]]] = {}
    for phase in ("voice_registration", "synthesis"):
        value = result.get(phase)
        if not isinstance(value, dict):
            safe[phase] = []
            continue
        attempts = value.get("attempts")
        if not isinstance(attempts, list):
            attempts = [value]
        safe[phase] = [
            {
                key: attempt.get(key)
                for key in (
                    "attempt",
                    "phase",
                    "status_code",
                    "request_id",
                    "usage",
                    "error_code",
                )
                if key in attempt
            }
            for attempt in attempts
            if isinstance(attempt, dict)
        ]
    return safe


def add_file(tar: tarfile.TarFile, source: Path, member: str) -> None:
    info = tar.gettarinfo(str(source), arcname=member)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as handle:
        tar.addfile(info, handle)


def group_speakers(records: list[dict[str, Any]]) -> list[list[str]]:
    sizes: dict[str, int] = Counter()
    for row in records:
        speaker = row["speaker"]
        sizes[speaker] += int(row["natural_audio_bytes"]) + int(row["canonical_tts_16k_bytes"])
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for speaker in sorted(sizes):
        speaker_size = int(sizes[speaker])
        if current and current_size + speaker_size > SHARD_TARGET_BYTES:
            groups.append(current)
            current, current_size = [], 0
        current.append(speaker)
        current_size += speaker_size
    if current:
        groups.append(current)
    return groups


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = args.input_manifest.resolve()
    tts_dir = args.tts_output_dir.resolve()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    tts_meta_path = tts_dir / "tts_meta.json"
    frozen_tts_manifest_path = tts_dir / "input_manifest.frozen.json"
    tts_meta = json.loads(tts_meta_path.read_text(encoding="utf-8"))
    frozen_tts_manifest = json.loads(
        frozen_tts_manifest_path.read_text(encoding="utf-8")
    )

    if selection.get("frozen_before_tts") is not True:
        raise ValueError("input selection manifest is not frozen before TTS")
    if selection.get("status") != "complete" or selection.get("records_expected") != 40:
        raise ValueError("female selection manifest is not the frozen 40-row cohort")
    if tts_meta.get("complete") is not True or tts_meta.get("samples_ok") != 40:
        raise ValueError("Qwen generation metadata is not complete for 40 rows")
    if tts_meta.get("samples_failed") != 0 or tts_meta.get("records_expected") != 40:
        raise ValueError("Qwen generation metadata reports failed or missing rows")
    if tts_meta.get("model") != MODEL or selection.get("model") != MODEL:
        raise ValueError("Qwen model does not match the frozen female protocol")
    if tts_meta.get("expected_provider_sample_rate_hz") != PROVIDER_RATE_HZ:
        raise ValueError("provider sample rate differs from the frozen 24 kHz contract")
    if tts_meta.get("canonical_sample_rate_hz") != CANONICAL_RATE_HZ:
        raise ValueError("canonical sample rate differs from the frozen 16 kHz contract")

    selection_records = selection.get("records")
    tts_records = frozen_tts_manifest.get("records")
    results = tts_meta.get("results")
    if not isinstance(selection_records, list) or not isinstance(tts_records, list):
        raise ValueError("manifest rows are missing")
    if not isinstance(results, dict):
        raise ValueError("TTS result records are missing")
    if len(selection_records) != 40 or len(tts_records) != 40 or len(results) != 40:
        raise ValueError("selection, frozen TTS input, and output row counts differ")

    selection_by_id = {row["id"]: row for row in selection_records}
    frozen_by_id = {row["id"]: row for row in tts_records}
    if set(selection_by_id) != set(frozen_by_id) or set(selection_by_id) != set(results):
        raise ValueError("selection, frozen TTS input, and output IDs differ")

    selection_identity = [
        {
            "id": row["id"],
            "speaker": row["speaker"],
            "text": row["text"],
            "split": row["split"],
            "ref_sha256": row["ref_sha256"],
        }
        for row in selection_records
    ]
    identity_sha = canonical_sha256(selection_identity)
    if identity_sha != selection.get("input_identity_sha256"):
        raise ValueError("selection input identity fingerprint does not recompute")
    if frozen_tts_manifest.get("source_manifest_fingerprint") != selection.get("manifest_sha256"):
        raise ValueError("Qwen's frozen snapshot does not point to this selection fingerprint")

    frozen_manifest_raw_sha = sha256_file(frozen_tts_manifest_path)
    frozen_manifest_fingerprint = frozen_tts_manifest.get("manifest_sha256")
    if tts_meta.get("input_manifest_sha256") != frozen_manifest_fingerprint:
        raise ValueError("Qwen metadata does not bind the frozen TTS input manifest fingerprint")

    portable_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    speaker_counts: Counter[str] = Counter()
    natural_mismatch = 0
    provider_mismatch = 0
    canonical_mismatch = 0

    for row in selection_records:
        sample_id = row["id"]
        frozen = frozen_by_id[sample_id]
        result = results[sample_id]
        if row["speaker"] != frozen.get("speaker") or row["text"] != frozen.get("text"):
            raise ValueError(f"frozen TTS input identity changed for {sample_id}")
        publisher_text = row.get("transcript_source_text")
        if not isinstance(publisher_text, str):
            raise ValueError(f"publisher transcript text is missing for {sample_id}")
        if "".join(row["text"].split()) != "".join(publisher_text.split()):
            raise ValueError(f"publisher transcript comparison failed after frozen whitespace removal for {sample_id}")
        if row.get("speaker_metadata_sex") != "F" or row.get("split") != "evaluation":
            raise ValueError(f"female evaluation scope mismatch for {sample_id}")
        if result.get("status") != "ok" or result.get("id") != sample_id:
            raise ValueError(f"Qwen output row is not successful for {sample_id}")
        if result.get("text") != row["text"] or result.get("reference_audio_sha256") != row["ref_sha256"]:
            raise ValueError(f"Qwen output provenance differs from frozen input for {sample_id}")
        speaker_counts[row["speaker"]] += 1

        natural_path = resolve_file(
            row["ref_path"], manifest_dir=selection_path.parent, repo_root=repo_root
        )
        natural_sha = sha256_file(natural_path)
        if natural_sha != row["ref_sha256"]:
            raise ValueError(f"natural source file hash mismatch for {sample_id}")
        natural_decode = full_decode_audit(natural_path)
        if (
            natural_decode["sample_rate_hz"] != CANONICAL_RATE_HZ
            or natural_decode["channels"] != 1
            or natural_decode["decoded_frames"] <= 0
        ):
            raise ValueError(f"natural source is not decodable mono 16 kHz audio for {sample_id}")
        if not 3.0 <= float(natural_decode["duration_s"]) <= 10.0:
            raise ValueError(f"natural source duration falls outside frozen eligibility for {sample_id}")
        if natural_decode["declared_vs_payload_mismatch"]:
            natural_mismatch += 1

        provider_path = Path(result["provider_audio"]).resolve()
        provider_sha = sha256_file(provider_path)
        if provider_sha != result.get("provider_audio_sha256"):
            raise ValueError(f"provider 24 kHz file hash mismatch for {sample_id}")
        provider_decode = full_decode_audit(provider_path)
        if (
            provider_decode["sample_rate_hz"] != PROVIDER_RATE_HZ
            or provider_decode["channels"] != 1
            or provider_decode["decoded_frames"] <= 0
            or provider_decode["subtype"] != "PCM_16"
        ):
            raise ValueError(f"provider audio is not decodable mono 24 kHz PCM16 for {sample_id}")
        if provider_decode["declared_vs_payload_mismatch"]:
            provider_mismatch += 1
        if provider_decode["decoded_frames"] != result.get("provider_full_decode_audit", {}).get("decoded_frames"):
            raise ValueError(f"provider EOF frame count differs from synthesis metadata for {sample_id}")
        if provider_decode["decoded_frames"] != result.get("provider_duration_s", 0) * PROVIDER_RATE_HZ:
            # Durations are rounded in metadata, so accept at most one decoded frame of rounding.
            if abs(provider_decode["duration_s"] - float(result.get("provider_duration_s", 0))) > 1.0 / PROVIDER_RATE_HZ:
                raise ValueError(f"provider duration differs from EOF decode for {sample_id}")

        canonical_path = Path(result["canonical_16k_audio"]).resolve()
        canonical_sha = sha256_file(canonical_path)
        if canonical_sha != result.get("canonical_audio_sha256"):
            raise ValueError(f"canonical 16 kHz file hash mismatch for {sample_id}")
        canonical_decode = full_decode_audit(canonical_path)
        if (
            canonical_decode["sample_rate_hz"] != CANONICAL_RATE_HZ
            or canonical_decode["channels"] != 1
            or canonical_decode["decoded_frames"] <= 0
            or canonical_decode["subtype"] != "PCM_16"
            or canonical_decode["decoded_frames"] != result.get("samples")
        ):
            raise ValueError(f"canonical output is not exact mono 16 kHz PCM16 for {sample_id}")
        if canonical_decode["declared_vs_payload_mismatch"]:
            canonical_mismatch += 1
        if result.get("sample_rate_hz") != CANONICAL_RATE_HZ or result.get("channels") != 1:
            raise ValueError(f"canonical metadata dimensions mismatch for {sample_id}")

        natural_member = f"audio/natural/{row['speaker']}/{sample_id}.wav"
        tts_member = f"audio/tts/{row['speaker']}/{sample_id}.wav"
        safe_receipts = usage_receipts(result)
        portable = {
            "id": sample_id,
            "speaker": row["speaker"],
            "speaker_metadata_sex": row["speaker_metadata_sex"],
            "split": row["split"],
            "text": row["text"],
            "publisher_transcript_text": publisher_text,
            "publisher_text_sha256": hashlib.sha256(publisher_text.encode("utf-8")).hexdigest(),
            "text_normalization": "all whitespace removed, matching the frozen selection rule",
            "transcript_source_sha256": row["transcript_source_sha256"],
            "natural_audio_path": natural_member,
            "natural_audio_sha256": natural_sha,
            "natural_audio_bytes": natural_path.stat().st_size,
            "natural_decoded_pcm16_sha256": natural_decode["decoded_pcm16_sha256"],
            "natural_sample_rate_hz": natural_decode["sample_rate_hz"],
            "natural_samples": natural_decode["decoded_frames"],
            "natural_duration_s": natural_decode["duration_s"],
            "canonical_tts_16k_audio_path": tts_member,
            "canonical_tts_16k_audio_sha256": canonical_sha,
            "canonical_tts_16k_bytes": canonical_path.stat().st_size,
            "canonical_tts_decoded_pcm16_sha256": canonical_decode["decoded_pcm16_sha256"],
            "canonical_tts_sample_rate_hz": canonical_decode["sample_rate_hz"],
            "canonical_tts_samples": canonical_decode["decoded_frames"],
            "canonical_tts_duration_s": canonical_decode["duration_s"],
            "provider": result["provider"],
            "model": result["model"],
            "provider_24k_audio_sha256": provider_sha,
            "provider_24k_bytes": provider_path.stat().st_size,
            "provider_24k_decoded_pcm16_sha256": provider_decode["decoded_pcm16_sha256"],
            "provider_24k_sample_rate_hz": provider_decode["sample_rate_hz"],
            "provider_24k_samples": provider_decode["decoded_frames"],
            "provider_24k_duration_s": provider_decode["duration_s"],
            "provider_24k_declared_vs_payload_mismatch": provider_decode["declared_vs_payload_mismatch"],
            "tts_request_receipts": safe_receipts,
        }
        portable_rows.append(portable)
        audit_rows.append(
            {
                "id": sample_id,
                "speaker": row["speaker"],
                "split": row["split"],
                "natural": {
                    "source_path": str(natural_path),
                    "source_file_sha256": natural_sha,
                    "source_archive_member": row.get("source_archive_member"),
                    "source_archive_member_sha256": row.get("source_archive_member_sha256"),
                    **natural_decode,
                },
                "provider_24k_raw": {
                    "source_path": str(provider_path),
                    "file_sha256": provider_sha,
                    "file_bytes": provider_path.stat().st_size,
                    **provider_decode,
                },
                "canonical_tts_16k": {
                    "source_path": str(canonical_path),
                    "file_sha256": canonical_sha,
                    "file_bytes": canonical_path.stat().st_size,
                    **canonical_decode,
                },
                "provider_vs_canonical_frame_ratio": {
                    "provider_samples": provider_decode["decoded_frames"],
                    "canonical_samples": canonical_decode["decoded_frames"],
                    "expected_ratio_numerator": 2,
                    "expected_ratio_denominator": 3,
                    "canonical_minus_expected_samples": (
                        canonical_decode["decoded_frames"]
                        - provider_decode["decoded_frames"] * 2 / 3
                    ),
                },
                "tts_request_receipts": safe_receipts,
                "natural_source_text_sha256": row["transcript_source_sha256"],
                "publisher_transcript_text": publisher_text,
                "publisher_text_sha256": hashlib.sha256(publisher_text.encode("utf-8")).hexdigest(),
                "publisher_text_matches_qwen_text_after_whitespace_removal": True,
                "qwen_text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
            }
        )

    if len(speaker_counts) != 20 or any(count != 2 for count in speaker_counts.values()):
        raise ValueError("female cohort is not exactly two clips per each of twenty speakers")

    portable_manifest = {
        "schema_version": 1,
        "manifest_type": "portable_female_tts_replication_inputs",
        "status": "FROZEN_BEFORE_MODEL_FORWARD",
        "split": "evaluation",
        "evaluation_outcomes_read": False,
        "evaluation_waveforms_read": False,
        "model_effects_read": False,
        "source_selection_manifest_fingerprint": selection["manifest_sha256"],
        "source_selection_manifest_file_sha256": sha256_file(selection_path),
        "input_identity_sha256": identity_sha,
        "tts_frozen_manifest_file_sha256": frozen_manifest_raw_sha,
        "tts_metadata_model": tts_meta["model"],
        "canonicalization": tts_meta["canonicalization"],
        "records_expected": 40,
        "records": portable_rows,
    }
    portable_manifest["manifest_sha256"] = canonical_sha256(portable_manifest)
    portable_manifest_path = out_dir / "portable_input_manifest.json"
    write_json(portable_manifest_path, portable_manifest)

    path_map = {
        "schema_version": 1,
        "manifest_type": "portable_input_path_conversion_sidecar",
        "source_selection_manifest_file_sha256": sha256_file(selection_path),
        "portable_input_manifest_sha256": portable_manifest["manifest_sha256"],
        "records": [
            {
                "id": row["id"],
                "speaker": row["speaker"],
                "natural_local_path": audit["natural"]["source_path"],
                "natural_portable_path": row["natural_audio_path"],
                "natural_sha256": row["natural_audio_sha256"],
                "provider_24k_local_path": audit["provider_24k_raw"]["source_path"],
                "provider_24k_sha256": row["provider_24k_audio_sha256"],
                "canonical_tts_16k_local_path": audit["canonical_tts_16k"]["source_path"],
                "canonical_tts_16k_portable_path": row["canonical_tts_16k_audio_path"],
                "canonical_tts_16k_sha256": row["canonical_tts_16k_audio_sha256"],
            }
            for row, audit in zip(portable_rows, audit_rows, strict=True)
        ],
    }
    path_map["manifest_sha256"] = canonical_sha256(path_map)
    path_map_path = out_dir / "portable_path_map.json"
    write_json(path_map_path, path_map)

    shards_dir = out_dir / "shards"
    shards_dir.mkdir()
    shard_summaries: list[dict[str, Any]] = []
    natural_source_by_id = {row["id"]: Path(a["natural"]["source_path"]) for row, a in zip(portable_rows, audit_rows, strict=True)}
    canonical_source_by_id = {row["id"]: Path(a["canonical_tts_16k"]["source_path"]) for row, a in zip(portable_rows, audit_rows, strict=True)}
    records_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in portable_rows:
        records_by_speaker.setdefault(row["speaker"], []).append(row)

    portable_json_bytes = portable_manifest_path.read_bytes()
    path_map_json_bytes = path_map_path.read_bytes()
    for index, speaker_group in enumerate(group_speakers(portable_rows), start=1):
        speaker_rows = [row for speaker in speaker_group for row in records_by_speaker[speaker]]
        lo, hi = speaker_group[0], speaker_group[-1]
        shard_path = shards_dir / f"female_eval_{lo}-{hi}_part{index:02d}.tar"
        with tarfile.open(shard_path, mode="w") as tar:
            for name, data in (
                ("manifest/portable_input_manifest.json", portable_json_bytes),
                ("manifest/portable_path_map.json", path_map_json_bytes),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                tar.addfile(info, io.BytesIO(data))
            for row in speaker_rows:
                add_file(tar, natural_source_by_id[row["id"]], row["natural_audio_path"])
                add_file(tar, canonical_source_by_id[row["id"]], row["canonical_tts_16k_audio_path"])
        size = shard_path.stat().st_size
        if size > MAX_SHARD_BYTES:
            raise ValueError(f"portable shard exceeds 7.5 MB safety limit: {shard_path.name}")
        shard_summaries.append(
            {
                "path": str(shard_path.resolve()),
                "name": shard_path.name,
                "sha256": sha256_file(shard_path),
                "bytes": size,
                "speaker_ids": speaker_group,
                "record_ids": [row["id"] for row in speaker_rows],
                "natural_and_tts_audio_members": 2 * len(speaker_rows),
                "portable_manifest_sha256": portable_manifest["manifest_sha256"],
            }
        )

    script_path = Path(__file__).resolve()
    audit = {
        "schema_version": 1,
        "manifest_type": "qwen_female_full_eof_preflight",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_cohort": "independent female AISHELL-1 replication",
        "split": "evaluation",
        "evaluation_outcomes_read": False,
        "evaluation_waveforms_read": False,
        "model_effects_read": False,
        "source_selection_manifest_path": str(selection_path),
        "source_selection_manifest_file_sha256": sha256_file(selection_path),
        "source_selection_manifest_fingerprint": selection["manifest_sha256"],
        "input_identity_sha256": identity_sha,
        "selection_rule_sha256": selection["selection_rule_sha256"],
        "source_dataset_binding": selection["dataset"],
        "tts_frozen_manifest_path": str(frozen_tts_manifest_path),
        "tts_frozen_manifest_file_sha256": frozen_manifest_raw_sha,
        "tts_frozen_manifest_fingerprint": frozen_manifest_fingerprint,
        "tts_meta_path": str(tts_meta_path),
        "tts_meta_file_sha256": sha256_file(tts_meta_path),
        "tts_meta_samples_ok": tts_meta["samples_ok"],
        "tts_meta_samples_failed": tts_meta["samples_failed"],
        "tts_provider": tts_meta["provider"],
        "tts_model": tts_meta["model"],
        "canonicalization": tts_meta["canonicalization"],
        "decoded_counts": {
            "records": len(audit_rows),
            "natural_verified": sum(bool(row["natural"]["source_file_sha256"]) for row in audit_rows),
            "provider_24k_verified": sum(bool(row["provider_24k_raw"]["file_sha256"]) for row in audit_rows),
            "canonical_16k_verified": sum(bool(row["canonical_tts_16k"]["file_sha256"]) for row in audit_rows),
            "provider_header_payload_mismatches": provider_mismatch,
            "canonical_header_payload_mismatches": canonical_mismatch,
            "natural_header_payload_mismatches": natural_mismatch,
        },
        "rows": audit_rows,
        "portable_input_manifest_path": str(portable_manifest_path.resolve()),
        "portable_input_manifest_file_sha256": sha256_file(portable_manifest_path),
        "portable_input_manifest_fingerprint": portable_manifest["manifest_sha256"],
        "portable_path_map_path": str(path_map_path.resolve()),
        "portable_path_map_file_sha256": sha256_file(path_map_path),
        "portable_path_map_fingerprint": path_map["manifest_sha256"],
        "portable_shard_limit_bytes": MAX_SHARD_BYTES,
        "portable_shards": shard_summaries,
        "producer": {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
        },
    }
    audit["manifest_sha256"] = canonical_sha256(audit)
    audit_path = out_dir / "qwen_female_full_eof_preflight.json"
    write_json(audit_path, audit)
    return {
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
        "audit_fingerprint": audit["manifest_sha256"],
        "records": len(audit_rows),
        "provider_header_payload_mismatches": provider_mismatch,
        "canonical_header_payload_mismatches": canonical_mismatch,
        "natural_header_payload_mismatches": natural_mismatch,
        "portable_input_manifest": str(portable_manifest_path.resolve()),
        "portable_shards": shard_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--tts-output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
