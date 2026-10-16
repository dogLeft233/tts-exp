from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tarfile
import wave
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "runs/recovered_20260930/local/qwen_source_axis"
INPUT_MANIFEST = SOURCE_ROOT / "female_eval_20261001/input_v2/input_manifest.json"
ARCHIVE = SOURCE_ROOT / "female_eval_20261001/source/data_aishell.tgz.partial"
ARCHIVE_VERIFICATION = SOURCE_ROOT / "female_eval_20261001/source/verified_archive.json"
TRANSCRIPT = SOURCE_ROOT / "hf_source/aishell_transcript_v0.8.txt"
SPEAKER_INFO = SOURCE_ROOT / "hf_source/speaker.info"
RESOURCE_ARCHIVE = SOURCE_ROOT / "female_eval_20261001/source/resource_aishell.tgz"
RESOURCE_VERIFICATION = SOURCE_ROOT / "female_eval_20261001/source/resource_aishell.tgz.verified.json"

RMS_TARGET = 0.0376838172675746
EXPECTED_ARCHIVE_BYTES = 15_582_913_665
EXPECTED_ARCHIVE_MD5 = "2f494334227864a8a8fec932999db9d8"
EXPECTED_ARCHIVE_SHA256 = "a4a0313cde0a933e0e01a451f77de0a23d6c942f4694af5bb7f40b9dc38143fe"
EXPECTED_RESOURCE_MD5 = "957d480a0fcac85fc18e550756f624e5"
TRANSCRIPT_MEMBER = "data_aishell/transcript/aishell_transcript_v0.8.txt"
HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x3134F),
    (0x2F800, 0x2FA1F),
)
PRIMARY_M = tuple(f"S{i:04d}" for i in range(102, 112))
BACKUP_M = tuple(f"S{i:04d}" for i in range(112, 117))
PRIMARY_F = tuple(f"S{i:04d}" for i in range(144, 154))
BACKUP_F = tuple(f"S{i:04d}" for i in range(154, 159))
EXCLUDED_SPEAKER_RANGES = ((2, 101), (124, 143), (764, 916))
DOSE_CANDIDATES_DB = (6.0, 3.0, 1.5, 0.75, 0.375)


class HashingReader:
    """Non-seeking reader that hashes the source stream as tarfile consumes it."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.sha256.update(chunk)
        self.md5.update(chunk)
        self.bytes_read += len(chunk)
        return chunk


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(raw)


def text_is_han_only(text: str) -> bool:
    compact = "".join(text.split())
    return bool(compact) and all(any(lo <= ord(char) <= hi for lo, hi in HAN_RANGES) for char in compact)


def parse_transcripts(raw: bytes) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for line in raw.decode("utf-8-sig").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            transcripts[fields[0]] = fields[1].strip()
    return transcripts


def parse_speaker_info(raw: bytes) -> dict[str, str]:
    speakers: dict[str, str] = {}
    for line in raw.decode("utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1] in {"M", "F"}:
            speakers[f"S{int(fields[0]):04d}"] = fields[1]
    return speakers


def excluded_speakers() -> set[str]:
    return {f"S{i:04d}" for start, end in EXCLUDED_SPEAKER_RANGES for i in range(start, end + 1)}


def wav_quality(raw: bytes) -> dict[str, Any]:
    with wave.open(io.BytesIO(raw), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frames = stream.getnframes()
        compression = stream.getcomptype()
        payload = stream.readframes(frames)
        if stream.readframes(1):
            raise ValueError("WAV contains payload after declared frame count")
    if channels != 1 or width != 2 or sample_rate != 16_000 or compression != "NONE":
        raise ValueError("expected uncompressed mono 16 kHz PCM16")
    if len(payload) != frames * channels * width:
        raise ValueError("decoded payload size differs from WAV frame declaration")
    samples = array.array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ValueError("empty WAV")
    sample_min = min(samples)
    sample_max = max(samples)
    peak_i16 = max(abs(sample_min), abs(sample_max))
    mean_square = sum(int(value) * int(value) for value in samples) / len(samples)
    rms = math.sqrt(mean_square) / 32768.0
    peak = peak_i16 / 32768.0
    fixed_gain = RMS_TARGET / rms if rms > 0 else math.inf
    duration = frames / sample_rate
    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "subtype": "PCM_16",
        "frames": frames,
        "duration_s": duration,
        "finite": True,
        "nonzero": rms > 0.0,
        "peak_abs_normalized": peak,
        "rms_normalized": rms,
        "positive_full_scale_samples": sum(value == 32767 for value in samples),
        "negative_full_scale_samples": sum(value == -32768 for value in samples),
        "raw_unclipped": sample_max < 32767 and sample_min > -32768,
        "fixed_rms_gain": fixed_gain,
        "fixed_rms_gain_db": 20.0 * math.log10(fixed_gain) if math.isfinite(fixed_gain) and fixed_gain > 0 else None,
        "fixed_rms_peak_abs_normalized": peak * fixed_gain,
        "fixed_rms_unclipped": math.isfinite(fixed_gain) and peak * fixed_gain < 1.0,
    }


def safe_relative_member(member_name: str, expected_prefix: str) -> bool:
    path = PurePosixPath(member_name)
    return not path.is_absolute() and ".." not in path.parts and member_name.startswith(expected_prefix)


def extract_candidate_archives(
    archive_path: Path,
    output_dir: Path,
    candidate_speakers: tuple[str, ...],
    expected_sha256: str,
    expected_md5: str,
    expected_size: int,
) -> tuple[dict[str, dict[str, Any]], bytes]:
    output_dir.mkdir(parents=True, exist_ok=False)
    wanted = {f"data_aishell/wav/{speaker}.tar.gz": speaker for speaker in candidate_speakers}
    extracted: dict[str, dict[str, Any]] = {}
    embedded_transcript: bytes | None = None
    with archive_path.open("rb") as raw_stream:
        hashing_stream = HashingReader(raw_stream)
        with tarfile.open(fileobj=hashing_stream, mode="r|gz") as outer:
            for member in outer:
                if not member.isfile():
                    continue
                if member.name == TRANSCRIPT_MEMBER:
                    if embedded_transcript is not None:
                        raise ValueError("publisher transcript appears more than once")
                    source = outer.extractfile(member)
                    if source is None:
                        raise ValueError("cannot read embedded publisher transcript")
                    embedded_transcript = source.read()
                speaker = wanted.get(member.name)
                if speaker is None:
                    continue
                if member.name in extracted or not safe_relative_member(member.name, "data_aishell/wav/"):
                    raise ValueError(f"duplicate or unsafe speaker archive member: {member.name}")
                source = outer.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read speaker archive member: {member.name}")
                target = output_dir / f"{speaker}.tar.gz"
                digest = hashlib.sha256()
                written = 0
                with target.open("xb") as destination:
                    while True:
                        chunk = source.read(1 << 20)
                        if not chunk:
                            break
                        destination.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if written != member.size:
                    raise ValueError(f"speaker archive size mismatch: {speaker}")
                extracted[member.name] = {
                    "speaker": speaker,
                    "outer_member": member.name,
                    "outer_member_bytes": member.size,
                    "inner_archive_path": str(target),
                    "inner_archive_sha256": digest.hexdigest(),
                }
        while hashing_stream.read(1 << 20):
            pass
        actual_size = hashing_stream.bytes_read
        actual_sha256 = hashing_stream.sha256.hexdigest()
        actual_md5 = hashing_stream.md5.hexdigest()
    if actual_size != expected_size or actual_sha256 != expected_sha256 or actual_md5 != expected_md5:
        raise ValueError(f"official outer archive mismatch: bytes={actual_size}, sha256={actual_sha256}, md5={actual_md5}")
    if len(extracted) != len(candidate_speakers):
        missing = sorted(set(candidate_speakers) - {row["speaker"] for row in extracted.values()})
        raise ValueError(f"missing candidate speaker archive members: {missing}")
    if embedded_transcript is None:
        raise ValueError("official archive does not contain the pinned transcript member")
    return {row["speaker"]: row for row in extracted.values()}, embedded_transcript


def inspect_speaker_archive(
    speaker: str,
    archive_path: Path,
    transcripts: dict[str, str],
    audio_dir: Path,
) -> dict[str, Any]:
    rejected: list[dict[str, Any]] = []
    wav_members: list[tarfile.TarInfo] = []
    with tarfile.open(archive_path, mode="r:gz") as inner:
        for member in inner.getmembers():
            path = PurePosixPath(member.name)
            if member.isfile() and path.suffix.lower() == ".wav":
                if not safe_relative_member(member.name, f"train/{speaker}/"):
                    rejected.append({"member": member.name, "reason": "unexpected_archive_path"})
                    continue
                wav_members.append(member)
        wav_members.sort(key=lambda member: (PurePosixPath(member.name).name.casefold(), member.name.casefold()))
        for member in wav_members:
            filename = PurePosixPath(member.name).name
            utterance_id = PurePosixPath(filename).stem
            text = transcripts.get(utterance_id)
            if not text:
                rejected.append({"member": member.name, "reason": "missing_transcript"})
                continue
            if not text_is_han_only(text):
                rejected.append({"member": member.name, "reason": "transcript_not_han_only"})
                continue
            source = inner.extractfile(member)
            if source is None:
                rejected.append({"member": member.name, "reason": "unreadable_wav"})
                continue
            raw = source.read()
            try:
                audit = wav_quality(raw)
            except (EOFError, OSError, ValueError, wave.Error) as error:
                rejected.append({"member": member.name, "reason": "invalid_wav", "detail": str(error)})
                continue
            if not 3.0 <= audit["duration_s"] <= 10.0:
                rejected.append({"member": member.name, "reason": "duration_out_of_range", "duration_s": audit["duration_s"]})
                continue
            if not audit["nonzero"]:
                rejected.append({"member": member.name, "reason": "all_zero_audio"})
                continue
            if not audit["raw_unclipped"]:
                rejected.append({"member": member.name, "reason": "raw_full_scale_sample"})
                continue
            if not audit["fixed_rms_unclipped"]:
                rejected.append({"member": member.name, "reason": "fixed_rms_would_clip", "fixed_rms_peak_abs_normalized": audit["fixed_rms_peak_abs_normalized"]})
                continue
            target = audio_dir / speaker / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as destination:
                destination.write(raw)
            return {
                "speaker": speaker,
                "utterance_id": utterance_id,
                "source_archive_member": member.name,
                "filename": filename,
                "audio_path": str(target),
                "audio_bytes": len(raw),
                "audio_sha256": sha256_bytes(raw),
                "source_text": text,
                "tts_text_normalized": "".join(text.split()),
                "audio_audit": audit,
                "rejected_before_selection": rejected,
            }
    return {"speaker": speaker, "status": "NO_QUALIFYING_UTTERANCE", "rejected_before_selection": rejected, "wav_members_seen": len(wav_members)}


def verify_input_pins() -> tuple[dict[str, Any], dict[str, Any], bytes, bytes, bytes]:
    manifest_raw = INPUT_MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    fingerprint = manifest.get("manifest_sha256")
    if fingerprint != canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ValueError("prior input manifest fingerprint mismatch")
    archive_verification_raw = ARCHIVE_VERIFICATION.read_bytes()
    archive_verification = json.loads(archive_verification_raw.decode("utf-8"))
    if archive_verification.get("status") != "VERIFIED_OFFICIAL_OPENSLR_ARCHIVE":
        raise ValueError("prior archive receipt is not a verified official archive")
    if Path(archive_verification.get("output_path", "")).resolve() != ARCHIVE.resolve():
        raise ValueError("archive receipt is bound to a different source path")
    if archive_verification.get("size") != EXPECTED_ARCHIVE_BYTES or archive_verification.get("sha256") != EXPECTED_ARCHIVE_SHA256 or archive_verification.get("md5") != EXPECTED_ARCHIVE_MD5:
        raise ValueError("archive receipt differs from frozen official source identity")
    if ARCHIVE.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("official archive size differs from frozen source identity")
    dataset_archive = manifest.get("dataset", {}).get("data_archive", {})
    if (dataset_archive.get("sha256"), dataset_archive.get("md5"), dataset_archive.get("size"), dataset_archive.get("source_url")) != (
        EXPECTED_ARCHIVE_SHA256,
        EXPECTED_ARCHIVE_MD5,
        EXPECTED_ARCHIVE_BYTES,
        "https://www.openslr.org/resources/33/data_aishell.tgz",
    ):
        raise ValueError("source input manifest does not bind the official archive")
    resource_raw = RESOURCE_VERIFICATION.read_bytes()
    resource = json.loads(resource_raw.decode("utf-8"))
    if resource.get("actual_md5") != EXPECTED_RESOURCE_MD5 or resource.get("archive_path") != str(RESOURCE_ARCHIVE):
        raise ValueError("speaker-info resource receipt mismatch")
    if hashlib.md5(RESOURCE_ARCHIVE.read_bytes()).hexdigest() != EXPECTED_RESOURCE_MD5:
        raise ValueError("publisher speaker-info resource archive checksum mismatch")
    transcript_raw = TRANSCRIPT.read_bytes()
    speaker_info_raw = SPEAKER_INFO.read_bytes()
    dataset = manifest["dataset"]
    if sha256_bytes(transcript_raw) != dataset["transcript_sha256"]:
        raise ValueError("pinned transcript hash mismatch")
    if sha256_bytes(speaker_info_raw) != dataset["speaker_info_sha256"]:
        raise ValueError("pinned speaker.info hash mismatch")
    return manifest, archive_verification, transcript_raw, speaker_info_raw, archive_verification_raw


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)


def run_prepare(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"official source archive missing: {ARCHIVE}")
    manifest, archive_verification, transcript_raw, speaker_info_raw, archive_verification_raw = verify_input_pins()
    speakers = parse_speaker_info(speaker_info_raw)
    exclusions = excluded_speakers()
    candidate_ids = PRIMARY_M + BACKUP_M + PRIMARY_F + BACKUP_F
    if len(candidate_ids) != 30 or len(set(candidate_ids)) != 30:
        raise ValueError("candidate roster must contain 30 unique speaker IDs")
    if set(candidate_ids) & exclusions:
        raise ValueError("candidate roster overlaps the frozen speaker exclusion union")
    expected_sex = {speaker: "M" for speaker in PRIMARY_M + BACKUP_M} | {speaker: "F" for speaker in PRIMARY_F + BACKUP_F}
    for speaker in candidate_ids:
        if speakers.get(speaker) != expected_sex[speaker]:
            raise ValueError(f"speaker.info sex mismatch or missing speaker: {speaker}")
    if shutil.disk_usage(output_root.parent).free < 2_000_000_000:
        raise OSError("insufficient free space for selected candidate source extraction")
    output_root.mkdir(parents=True, exist_ok=False)
    inner_archive_dir = output_root / "candidate_archives"
    audio_dir = output_root / "candidate_audio"
    archives, embedded_transcript = extract_candidate_archives(
        ARCHIVE,
        inner_archive_dir,
        candidate_ids,
        EXPECTED_ARCHIVE_SHA256,
        EXPECTED_ARCHIVE_MD5,
        EXPECTED_ARCHIVE_BYTES,
    )
    if sha256_bytes(embedded_transcript) != manifest["dataset"]["transcript_sha256"] or embedded_transcript != transcript_raw:
        raise ValueError("embedded official transcript differs from pinned transcript source")
    transcript_map = parse_transcripts(transcript_raw)
    candidate_rows: list[dict[str, Any]] = []
    first_valid_by_sex: dict[str, list[dict[str, Any]]] = {"M": [], "F": []}
    primary = {"M": PRIMARY_M, "F": PRIMARY_F}
    backups = {"M": BACKUP_M, "F": BACKUP_F}
    for sex in ("M", "F"):
        for rank, speaker in enumerate(primary[sex] + backups[sex], start=1):
            row = inspect_speaker_archive(speaker, Path(archives[speaker]["inner_archive_path"]), transcript_map, audio_dir)
            row.update({"sex": sex, "roster_rank": rank, "roster_role": "primary" if speaker in primary[sex] else "backup"})
            row["source_archive"] = archives[speaker]
            candidate_rows.append(row)
            if row.get("audio_sha256"):
                first_valid_by_sex[sex].append(row)
    chosen: dict[str, list[dict[str, Any]]] = {
        sex: first_valid_by_sex[sex][:10] for sex in ("M", "F")
    }
    if any(len(chosen[sex]) != 10 for sex in ("M", "F")):
        raise ValueError("fewer than ten prequalified speakers in a sex; do not substitute outside the frozen backups")
    chosen_ids = {row["speaker"] for sex in ("M", "F") for row in chosen[sex]}
    if len(chosen_ids) != 20:
        raise ValueError("final P1 speaker cohort contains duplicate speakers")
    for row in candidate_rows:
        row["cohort_status"] = "SELECTED" if row["speaker"] in chosen_ids else ("QUALIFIED_BACKUP_NOT_NEEDED" if row.get("audio_sha256") else "PREQA_EXCLUDED")
    peak_max = max(row["audio_audit"]["peak_abs_normalized"] for sex in ("M", "F") for row in chosen[sex])
    safe_dose = next((db for db in DOSE_CANDIDATES_DB if peak_max * 10.0 ** (db / 20.0) < 1.0), None)
    selected_rows = [row for sex in ("M", "F") for row in chosen[sex]]
    source_manifest = {
        "schema": "p1_gain_dose_response_source_manifest_v1",
        "status": "SOURCE_READY_FOR_ROOT_REVIEW",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "AISHELL-1 / OpenSLR SLR33",
            "archive_path": str(ARCHIVE),
            "archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "archive_md5_expected": EXPECTED_ARCHIVE_MD5,
            "archive_sha256_expected": EXPECTED_ARCHIVE_SHA256,
            "archive_actual_bytes": EXPECTED_ARCHIVE_BYTES,
            "archive_actual_md5": EXPECTED_ARCHIVE_MD5,
            "archive_actual_sha256": EXPECTED_ARCHIVE_SHA256,
            "archive_url": archive_verification.get("source_url"),
            "archive_verification_path": str(ARCHIVE_VERIFICATION),
            "archive_verification_sha256": sha256_bytes(archive_verification_raw),
            "input_manifest_path": str(INPUT_MANIFEST),
            "input_manifest_sha256": sha256_bytes(INPUT_MANIFEST.read_bytes()),
            "prior_input_manifest_fingerprint": manifest.get("manifest_sha256"),
            "transcript_revision": manifest["dataset"]["transcript_revision"],
            "transcript_path": str(TRANSCRIPT),
            "transcript_sha256": sha256_bytes(transcript_raw),
            "embedded_transcript_member": TRANSCRIPT_MEMBER,
            "embedded_transcript_sha256": sha256_bytes(embedded_transcript),
            "speaker_info_path": str(SPEAKER_INFO),
            "speaker_info_sha256": sha256_bytes(speaker_info_raw),
            "speaker_info_resource_archive_path": str(RESOURCE_ARCHIVE),
            "speaker_info_resource_archive_md5": EXPECTED_RESOURCE_MD5,
        },
        "selection_rule": {
            "speaker_exclusions": ["S0002-S0101", "S0124-S0143", "S0764-S0916"],
            "selection_order": "10 lowest-ID eligible M and 10 lowest-ID eligible F; if any candidate fails audio pre-QA, advance by ID within the same sex through the five predeclared backups",
            "primary_m": list(PRIMARY_M),
            "backup_m": list(BACKUP_M),
            "primary_f": list(PRIMARY_F),
            "backup_f": list(BACKUP_F),
            "utterance_order": "first member by casefolded WAV filename then member path",
            "text_rule": "official AISHELL transcript exists and contains only Han characters after Unicode whitespace removal",
            "audio_rule": "fully read uncompressed mono 16 kHz PCM16 WAV, 3-10 seconds inclusive, nonzero, finite by PCM16 representation, no full-scale raw samples, and the frozen fixed-RMS gain does not clip",
            "fixed_rms_target": RMS_TARGET,
            "no_score_or_model_output_used": True,
        },
        "uniform_dose_safety": {
            "tested_after_final_roster_selection": True,
            "max_selected_raw_peak": peak_max,
            "candidate_max_symmetric_dose_db": list(DOSE_CANDIDATES_DB),
            "selected_max_symmetric_dose_db": safe_dose,
            "levels_db": None if safe_dose is None else [-safe_dose, -safe_dose / 2.0, 0.0, safe_dose / 2.0, safe_dose],
            "criterion": "all selected RAW peaks times 10^(dB/20) remain strictly below 1; no per-utterance attenuation",
        },
        "cohort": selected_rows,
        "candidate_accounting": candidate_rows,
        "counts": {
            "candidate_speakers": len(candidate_rows),
            "selected_speakers": len(selected_rows),
            "selected_m": len(chosen["M"]),
            "selected_f": len(chosen["F"]),
            "preqa_excluded_speakers": sum(row["cohort_status"] == "PREQA_EXCLUDED" for row in candidate_rows),
            "qualified_backups_not_needed": sum(row["cohort_status"] == "QUALIFIED_BACKUP_NOT_NEEDED" for row in candidate_rows),
        },
        "execution": {
            "candidate_archive_dir": str(inner_archive_dir),
            "candidate_audio_dir": str(audio_dir),
            "prepare_script": str(Path(__file__).resolve()),
            "prepare_script_sha256": sha256_file(Path(__file__).resolve()),
            "renderer_forward_performed": False,
            "scoring_forward_performed": False,
        },
    }
    source_manifest["manifest_sha256"] = canonical_sha256(source_manifest)
    write_json_new(output_root / "source_manifest.json", source_manifest)
    return source_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and pre-QA an independent AISHELL P1 source cohort; performs no model forwards or scoring.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_prepare(args.output_root.resolve())
    print(json.dumps({"status": result["status"], "manifest_sha256": result["manifest_sha256"], "selected_speakers": [row["speaker"] for row in result["cohort"]], "max_dose_db": result["uniform_dose_safety"]["selected_max_symmetric_dose_db"], "source_manifest": str(args.output_root.resolve() / "source_manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
