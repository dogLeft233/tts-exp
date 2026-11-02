"""Select and freeze AISHELL-1 female replication inputs from verified SLR33.

This is an offline, deterministic input-preparation step. It reads only the
official AISHELL-1 archive and publisher metadata, verifies the archive hashes,
and writes selected natural WAVs plus a Qwen-compatible input manifest. It does
not call a model or read experiment effects.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import wave
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import qwen_tts_pair_prepare_20260930 as qwen


ROOT = Path(__file__).resolve().parents[2]
DATA_ARCHIVE_MD5 = "2f494334227864a8a8fec932999db9d8"
DATA_ARCHIVE_SIZE = 15_582_913_665
DATA_ARCHIVE_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"
RESOURCE_ARCHIVE_MD5 = "957d480a0fcac85fc18e550756f624e5"
RESOURCE_ARCHIVE_URL = "https://www.openslr.org/resources/33/resource_aishell.tgz"
CHECKSUM_URL = "https://www.openslr.org/resources/33/checksum.md5"
TRANSCRIPT_REVISION = qwen.HF_REVISION
INITIAL_SPEAKER_IDS = tuple(f"S{i:04d}" for i in range(124, 144))
KNOWN_PREVIOUS_UTTERANCE_EXCLUSIONS = (
    "S0131_BAC009S0131W0207",
    "S0131_BAC009S0131W0275",
    "S0141_BAC009S0141W0192",
    "S0141_BAC009S0141W0284",
)
KNOWN_PREVIOUS_FEMALE_AUDIO_EVIDENCE = (
    {"id": "S0131_BAC009S0131W0207", "sha256": "61c436d13d0ac6fb95c4e876d1240985474a0ea627d39808e35d90f0042b0c1d"},
    {"id": "S0131_BAC009S0131W0275", "sha256": "5c64a2ea75523ff977d9cd4cca9d9db9c9402ab18d221a73e78bee77602fe46f"},
    {"id": "S0141_BAC009S0141W0192", "sha256": "f2d89fd35d7ed2946155950eb5f932c9dc83c3dd9b97620174207cd3485a2362"},
    {"id": "S0141_BAC009S0141W0284", "sha256": "20201fbb8d5d02289fb7caa8a704bc79442424efb018e91cb47743d2b8b6f619"},
)
READ_BLOCK_FRAMES = 65536


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def write_json_immutable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"immutable output already exists with different contents: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def add_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    clean.pop("manifest_sha256", None)
    fingerprint = hashlib.sha256(qwen.canonical_json_bytes(clean)).hexdigest()
    return {**clean, "manifest_sha256": fingerprint}


def verify_archive(path: Path, verification_path: Path, expected_md5: str, expected_size: int | None, expected_url: str) -> dict[str, Any]:
    metadata = read_json(verification_path)
    recorded_path = Path(str(metadata.get("output_path", metadata.get("archive_path", "")))).resolve()
    if recorded_path != path.resolve():
        raise ValueError(f"archive verification is bound to a different path: {path.name}")
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(f"archive size mismatch for {path.name}")
    actual_md5 = hash_file(path, "md5")
    actual_sha256 = hash_file(path, "sha256")
    expected_metadata_md5 = metadata.get("md5", metadata.get("actual_md5"))
    expected_metadata_sha256 = metadata.get("sha256", metadata.get("actual_sha256"))
    if actual_md5 != expected_md5 or actual_md5 != expected_metadata_md5:
        raise ValueError(f"archive MD5 verification failed for {path.name}")
    if actual_sha256 != expected_metadata_sha256:
        raise ValueError(f"archive SHA256 verification failed for {path.name}")
    if expected_url not in str(metadata.get("source_url", "")):
        raise ValueError(f"archive verification source URL mismatch for {path.name}")
    if expected_url == DATA_ARCHIVE_URL and metadata.get("status") != "VERIFIED_OFFICIAL_OPENSLR_ARCHIVE":
        raise ValueError("data archive lacks the full OpenSLR verification status")
    return {
        **metadata,
        "actual_size_bytes": actual_size,
        "actual_md5": actual_md5,
        "actual_sha256": actual_sha256,
        "verified_manifest_path": str(verification_path.resolve()),
        "verified_manifest_sha256": hash_file(verification_path, "sha256"),
    }


def extract_official_speaker_info(resource_archive: Path) -> tuple[bytes, str]:
    with tarfile.open(resource_archive, mode="r:gz") as archive:
        matches = [
            member for member in archive.getmembers()
            if member.isfile() and PurePosixPath(member.name).name.casefold() == "speaker.info"
        ]
        if len(matches) != 1:
            raise ValueError("resource archive must contain exactly one speaker.info")
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise ValueError("cannot read publisher speaker.info")
        data = stream.read()
    digest = hashlib.sha256(data).hexdigest()
    return data, digest


def parse_speaker_sex(data: bytes) -> dict[str, str]:
    text = data.decode("utf-8-sig")
    result: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[0].isdigit() or fields[1] not in {"M", "F"}:
            continue
        result[f"S{int(fields[0]):04d}"] = fields[1]
    if not result:
        raise ValueError("publisher speaker.info contained no recognized speaker rows")
    return result


def decoded_wav_audit(data: bytes) -> dict[str, Any]:
    import soundfile as sf

    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            declared_frames = int(stream.getnframes())
            riff_rate = int(stream.getframerate())
            riff_channels = int(stream.getnchannels())
    except (wave.Error, EOFError, OSError):
        declared_frames = None
        riff_rate = None
        riff_channels = None
    decoded_frames = 0
    with sf.SoundFile(io.BytesIO(data), mode="r") as stream:
        sample_rate = int(stream.samplerate)
        channels = int(stream.channels)
        subtype = str(stream.subtype)
        while True:
            block = stream.read(READ_BLOCK_FRAMES, dtype="float32", always_2d=True)
            if not block.size:
                break
            decoded_frames += int(block.shape[0])
    return {
        "declared_frames": declared_frames,
        "decoded_frames": decoded_frames,
        "declared_vs_payload_mismatch": declared_frames is not None and declared_frames != decoded_frames,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "subtype": subtype,
        "riff_header_sample_rate_hz": riff_rate,
        "riff_header_channels": riff_channels,
        "duration_s": round(decoded_frames / sample_rate, 9) if sample_rate else None,
    }


def excluded_ids(path: Path | None, explicit: list[str]) -> set[str]:
    values = set(KNOWN_PREVIOUS_UTTERANCE_EXCLUSIONS)
    values.update(explicit)
    if path:
        payload = read_json(path)
        rows = payload.get("records", payload.get("manifest", []))
        for row in rows:
            if isinstance(row, dict):
                sample_id = row.get("id", row.get("sample_id"))
                if sample_id:
                    values.add(str(sample_id))
    return values


def parse_inner_speaker_archive(
    archive_bytes: bytes,
    speaker: str,
    transcript_map: dict[str, str],
    exclusions: set[str],
    minimum_duration_s: float,
    maximum_duration_s: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    counts = {"wav_members": 0, "non_han_or_missing_text": 0, "excluded_utterances": 0, "invalid_wav": 0, "duration_rejected": 0, "format_rejected": 0}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as inner:
        members = sorted(
            (
                member for member in inner.getmembers()
                if member.isfile() and PurePosixPath(member.name).suffix.lower() == ".wav"
            ),
            key=lambda member: (PurePosixPath(member.name).name.casefold(), member.name.casefold()),
        )
        for member in members:
            counts["wav_members"] += 1
            filename = PurePosixPath(member.name).name
            utterance_id = PurePosixPath(filename).stem
            sample_id = f"{speaker}_{utterance_id}"
            source_text = transcript_map.get(utterance_id)
            if not source_text or not qwen._text_is_han_only(source_text):
                counts["non_han_or_missing_text"] += 1
                continue
            if sample_id in exclusions or utterance_id in exclusions:
                counts["excluded_utterances"] += 1
                continue
            stream = inner.extractfile(member)
            if stream is None:
                counts["invalid_wav"] += 1
                continue
            with stream:
                audio_bytes = stream.read()
            try:
                audit = decoded_wav_audit(audio_bytes)
            except Exception:
                counts["invalid_wav"] += 1
                continue
            if not (minimum_duration_s <= float(audit["duration_s"]) <= maximum_duration_s):
                counts["duration_rejected"] += 1
                continue
            if (
                audit["sample_rate_hz"] != 16000
                or audit["channels"] != 1
                or audit["subtype"] != "PCM_16"
            ):
                counts["format_rejected"] += 1
                continue
            candidates.append({
                "filename": filename,
                "member": member.name,
                "audio_bytes": audio_bytes,
                "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
                "source_text": source_text,
                "text": "".join(source_text.split()),
                "wav_audit": audit,
            })
    candidates.sort(key=lambda item: (item["filename"].casefold(), item["member"].casefold()))
    return candidates[:2], counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--data-verification", type=Path, required=True)
    parser.add_argument("--resource-archive", type=Path, required=True)
    parser.add_argument("--resource-verification", type=Path, required=True)
    parser.add_argument("--transcript-file", type=Path, required=True)
    parser.add_argument("--speaker-info-existing", type=Path, required=True)
    parser.add_argument("--reference-out", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--excluded-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-utterance", action="append", default=[])
    parser.add_argument("--initial-start", type=int, default=124)
    parser.add_argument("--initial-end", type=int, default=143)
    parser.add_argument("--required-speakers", type=int, default=20)
    parser.add_argument("--minimum-duration", type=float, default=3.0)
    parser.add_argument("--maximum-duration", type=float, default=10.0)
    args = parser.parse_args()

    data_archive = args.data_archive.resolve()
    resource_archive = args.resource_archive.resolve()
    transcript_path = args.transcript_file.resolve()
    existing_speaker_info = args.speaker_info_existing.resolve()
    reference_root = args.reference_out.resolve()
    manifest_path = args.output_manifest.resolve()
    data_verification = verify_archive(
        data_archive, args.data_verification.resolve(), DATA_ARCHIVE_MD5, DATA_ARCHIVE_SIZE, DATA_ARCHIVE_URL
    )
    resource_verification = verify_archive(
        resource_archive, args.resource_verification.resolve(), RESOURCE_ARCHIVE_MD5, None, RESOURCE_ARCHIVE_URL
    )
    official_info, official_info_sha = extract_official_speaker_info(resource_archive)
    existing_info = existing_speaker_info.read_bytes()
    if official_info != existing_info:
        raise ValueError("resource archive speaker.info differs from pinned publisher metadata")
    transcript_map = qwen._parse_official_transcripts(transcript_path)
    transcript_sha = qwen.sha256_file(transcript_path)
    sex_by_speaker = parse_speaker_sex(official_info)
    initial_speakers = [f"S{i:04d}" for i in range(args.initial_start, args.initial_end + 1)]
    if len(initial_speakers) > args.required_speakers:
        raise ValueError("initial speaker span cannot exceed the required cohort size")
    if any(sex_by_speaker.get(speaker) != "F" for speaker in initial_speakers):
        raise ValueError("one or more predeclared initial speakers are not labeled F by publisher metadata")
    female_fallbacks = sorted(
        speaker for speaker, sex in sex_by_speaker.items()
        if sex == "F" and int(speaker[1:]) > args.initial_end
    )[: args.required_speakers]
    excluded = excluded_ids(None, args.exclude_utterance)
    for path in args.excluded_manifest:
        excluded.update(excluded_ids(path.resolve(), []))

    selected_by_speaker: dict[str, dict[str, Any]] = {}
    audit_by_speaker: dict[str, dict[str, Any]] = {}
    candidates_needed = set(initial_speakers + female_fallbacks)
    outer_inner_hashes: dict[str, dict[str, Any]] = {}
    embedded_transcript: bytes | None = None
    embedded_transcript_member: str | None = None
    print("scanning verified OpenSLR archive for target speaker packages", flush=True)
    with tarfile.open(data_archive, mode="r|gz") as outer:
        for member in outer:
            if not member.isfile():
                continue
            base = PurePosixPath(member.name).name
            if base.casefold() == "aishell_transcript_v0.8.txt" or base.casefold() == "aishell_transcript_v0.8.txt.utf8":
                transcript_stream = outer.extractfile(member)
                if transcript_stream is not None:
                    with transcript_stream:
                        embedded_transcript = transcript_stream.read()
                    embedded_transcript_member = member.name
                continue
            if not base.lower().endswith(".tar.gz"):
                continue
            speaker = base[:-7]
            if speaker not in candidates_needed:
                continue
            if speaker in female_fallbacks and all(target in selected_by_speaker for target in initial_speakers):
                continue
            package_stream = outer.extractfile(member)
            if package_stream is None:
                continue
            with package_stream:
                package_bytes = package_stream.read()
            package_sha = hashlib.sha256(package_bytes).hexdigest()
            outer_inner_hashes[speaker] = {
                "member": member.name,
                "size_bytes": len(package_bytes),
                "sha256": package_sha,
            }
            top_two, counts = parse_inner_speaker_archive(
                package_bytes,
                speaker,
                transcript_map,
                excluded,
                args.minimum_duration,
                args.maximum_duration,
            )
            audit_by_speaker[speaker] = {
                "speaker": speaker,
                "publisher_sex": sex_by_speaker.get(speaker),
                "status": "eligible" if len(top_two) >= 2 else "fewer_than_two_eligible_utterances",
                "selected_ids": [f"{speaker}_{Path(item['filename']).stem}" for item in top_two],
                "selected_members": [item["member"] for item in top_two],
                "selected_count": len(top_two),
                "candidate_counts": counts,
                "source_inner_archive": outer_inner_hashes[speaker],
            }
            if len(top_two) >= 2:
                selected_by_speaker[speaker] = {
                    "items": top_two,
                    "inner_archive": outer_inner_hashes[speaker],
                }
            print(f"speaker={speaker} eligible={len(top_two)}/2 inner_sha256={package_sha}", flush=True)

    if embedded_transcript is not None:
        embedded_map_path = manifest_path.with_name("aishell_transcript_v0.8.openslr.txt")
        if embedded_transcript.decode("utf-8-sig", errors="strict"):
            embedded_map_path.parent.mkdir(parents=True, exist_ok=True)
            if not embedded_map_path.exists():
                embedded_map_path.write_bytes(embedded_transcript)
            elif embedded_map_path.read_bytes() != embedded_transcript:
                raise ValueError("existing official transcript snapshot differs from data archive")
        embedded_map = qwen._parse_official_transcripts(embedded_map_path)
        selected_utterance_ids = {
            Path(item["filename"]).stem
            for selected in selected_by_speaker.values()
            for item in selected["items"]
        }
        text_mismatches = [
            utterance_id for utterance_id in sorted(selected_utterance_ids)
            if utterance_id not in embedded_map or embedded_map[utterance_id] != transcript_map.get(utterance_id)
        ]
        if text_mismatches:
            raise ValueError(f"pinned transcript differs from official archive transcript for {len(text_mismatches)} selected utterances")
        transcript_comparison = {
            "status": "selected_ids_text_equal_to_embedded_publisher_transcript",
            "member": embedded_transcript_member,
            "embedded_sha256": hashlib.sha256(embedded_transcript).hexdigest(),
            "pinned_sha256": transcript_sha,
            "matching_ids_checked": len(selected_utterance_ids),
            "mismatched_ids": [],
            "snapshot_path": str(embedded_map_path.resolve()),
        }
    else:
        transcript_comparison = {
            "status": "not_embedded_in_data_archive; used pinned AISHELL-1 transcript file",
            "member": None,
            "pinned_sha256": transcript_sha,
        }

    ordered_speakers: list[str] = []
    for speaker in initial_speakers:
        if speaker in selected_by_speaker:
            ordered_speakers.append(speaker)
        else:
            audit_by_speaker.setdefault(speaker, {
                "speaker": speaker,
                "publisher_sex": sex_by_speaker.get(speaker),
                "status": "speaker_archive_missing_or_unavailable",
                "selected_ids": [],
                "selected_members": [],
                "selected_count": 0,
                "candidate_counts": None,
                "source_inner_archive": None,
            })
    for fallback in female_fallbacks:
        if len(ordered_speakers) >= args.required_speakers:
            break
        if fallback in selected_by_speaker:
            ordered_speakers.append(fallback)
    if len(ordered_speakers) != args.required_speakers:
        raise ValueError(f"only {len(ordered_speakers)} eligible F speakers; need {args.required_speakers}")

    selected_rows: list[dict[str, Any]] = []
    selected_speaker_set = set(ordered_speakers)
    for speaker in ordered_speakers:
        payload = selected_by_speaker[speaker]
        archive_info = payload["inner_archive"]
        for item in payload["items"]:
            utterance_id = Path(item["filename"]).stem
            sample_id = f"{speaker}_{utterance_id}"
            destination = reference_root / speaker / item["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if qwen.sha256_file(destination) != item["audio_sha256"]:
                    raise ValueError(f"existing reference differs from selected publisher WAV: {sample_id}")
            else:
                destination.write_bytes(item["audio_bytes"])
            audit = item["wav_audit"]
            selected_rows.append({
                "id": sample_id,
                "speaker": speaker,
                "speaker_metadata_sex": sex_by_speaker[speaker],
                "speaker_metadata_source_sha256": official_info_sha,
                "text": item["text"],
                "transcript_source_text": item["source_text"],
                "transcript_source_path": str(transcript_path.resolve()),
                "transcript_source_sha256": transcript_sha,
                "transcript_source_revision": TRANSCRIPT_REVISION,
                "ref_path": str(destination.resolve()),
                "ref_sha256": item["audio_sha256"],
                "ref_sample_rate_hz": audit["sample_rate_hz"],
                "ref_channels": audit["channels"],
                "ref_duration_s": audit["duration_s"],
                "ref_audio_audit": audit,
                "split": "evaluation",
                "study_cohort": "female_replication_20261001",
                "language": qwen.LANGUAGE,
                "reference_role": "natural_self_clone_reference",
                "source_dataset": "AISHELL-1 / official OpenSLR SLR33 data release",
                "source_revision": "OpenSLR SLR33 official archive; transcript pinned to AISHELL/AISHELL-1 revision " + TRANSCRIPT_REVISION,
                "source_archive": {
                    "path": str(data_archive),
                    "url": DATA_ARCHIVE_URL,
                    "size_bytes": data_verification["actual_size_bytes"],
                    "md5": data_verification["actual_md5"],
                    "sha256": data_verification["actual_sha256"],
                    "verified_archive_manifest_sha256": data_verification["verified_manifest_sha256"],
                    "inner_speaker_archive_member": archive_info["member"],
                    "inner_speaker_archive_size_bytes": archive_info["size_bytes"],
                    "inner_speaker_archive_sha256": archive_info["sha256"],
                },
                "source_archive_member": item["member"],
                "source_archive_member_sha256": item["audio_sha256"],
            })

    if len(selected_rows) != args.required_speakers * 2:
        raise ValueError("selected row count does not match required cohort")
    if len({row["id"] for row in selected_rows}) != len(selected_rows):
        raise ValueError("duplicate selected utterance IDs")
    identity_rows = [
        {key: row[key] for key in ("id", "speaker", "text", "split", "ref_sha256")}
        for row in selected_rows
    ]
    identity_sha256 = hashlib.sha256(qwen.canonical_json_bytes(identity_rows)).hexdigest()
    selected_audit = [audit_by_speaker[speaker] for speaker in ordered_speakers]
    payload = add_fingerprint({
        "schema_version": 1,
        "manifest_type": "qwen_tts_pair_input_manifest",
        "status": "complete",
        "frozen_before_tts": True,
        "cohort": "independent female AISHELL-1 replication; no effects consulted during selection",
        "dataset": {
            "name": "AISHELL-1 / OpenSLR SLR33",
            "transcript_dataset": "AISHELL/AISHELL-1",
            "transcript_revision": TRANSCRIPT_REVISION,
            "transcript_path": str(transcript_path.resolve()),
            "transcript_sha256": transcript_sha,
            "speaker_info_source": RESOURCE_ARCHIVE_URL + "#resource_aishell/speaker.info",
            "speaker_info_sha256": official_info_sha,
            "speaker_info_existing_path": str(existing_speaker_info),
            "speaker_info_existing_sha256": hashlib.sha256(existing_info).hexdigest(),
            "data_archive": data_verification,
            "resource_archive": resource_verification,
            "embedded_transcript_comparison": transcript_comparison,
        },
        "selection_rule": {
            "initial_speakers": list(INITIAL_SPEAKER_IDS),
            "fallback": "if any initial F-labeled speaker lacks two qualifying clips after explicit exclusions, advance by lexicographic speaker ID to the next F-labeled speaker; continue until the cohort has 20 speakers",
            "utterances": "within each speaker, sort WAV member filenames lexicographically and take the first two with nonempty publisher text containing only Han characters after Unicode whitespace removal and fully decoded mono 16 kHz PCM16 duration in the closed interval [3.0, 10.0] seconds",
            "text_normalization": "remove Unicode whitespace for TTS text; preserve publisher transcript text verbatim",
            "excluded_scientific_speakers": [f"S{i:04d}" for i in range(52, 102)],
            "excluded_previous_holdout_speakers": [f"S{i:04d}" for i in range(2, 42)],
            "excluded_test_speakers": [f"S{i:04d}" for i in range(764, 917)],
            "excluded_previous_utterance_ids": sorted(excluded),
            "known_previous_female_utterances": list(KNOWN_PREVIOUS_UTTERANCE_EXCLUSIONS),
            "previous_female_audio_evidence": list(KNOWN_PREVIOUS_FEMALE_AUDIO_EVIDENCE),
            "previous_audio_index": "B:\\tts-exp-project\\index\\runs-audio-source-manifest-20260929-complete.jsonl.gz",
            "previous_audio_run": "runs/fresh_aishell_disjoint_ref_target_replication2_20260929_input_prepare/audio",
            "minimum_duration_s": args.minimum_duration,
            "maximum_duration_s": args.maximum_duration,
            "required_speakers": args.required_speakers,
            "clips_per_speaker": 2,
            "publisher_sex_scope": "AISHELL publisher metadata label only; no inference from voice",
        },
        "selection_rule_sha256": hashlib.sha256(qwen.canonical_json_bytes({
            "initial_speakers": INITIAL_SPEAKER_IDS,
            "fallback": "lexicographic next F-labeled speaker when an initial speaker has fewer than two qualifying utterances",
            "eligibility": "publisher Han-only transcript; mono 16 kHz PCM16; full decoded duration [3,10] seconds; first two filename-sorted",
            "exclusions": sorted(excluded),
        })).hexdigest(),
        "speaker_selection_audit": selected_audit,
        "attempted_speaker_audit": [audit_by_speaker[speaker] for speaker in initial_speakers + female_fallbacks if speaker in audit_by_speaker],
        "selected_speakers": ordered_speakers,
        "rejected_speakers": [speaker for speaker in initial_speakers if speaker not in selected_speaker_set],
        "input_identity_sha256": identity_sha256,
        "provider": qwen.PROVIDER,
        "model": qwen.MODEL,
        "language": qwen.LANGUAGE,
        "records_expected": len(selected_rows),
        "records": selected_rows,
        "producer": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": qwen.sha256_file(Path(__file__).resolve()),
            "qwen_prepare_script_path": str(Path(qwen.__file__).resolve()),
            "qwen_prepare_script_sha256": qwen.sha256_file(Path(qwen.__file__).resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    })
    write_json_immutable(manifest_path, payload)
    print(json.dumps({
        "manifest_path": str(manifest_path),
        "manifest_sha256": payload["manifest_sha256"],
        "input_identity_sha256": identity_sha256,
        "selected_speakers": ordered_speakers,
        "records": len(selected_rows),
        "audio_reference_dir": str(reference_root),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
