#!/usr/bin/env python3
"""Materialize the frozen 40 historical fit endpoints for later transfer.

This reads only the ADR013-selected fit audio members. It does not read dev or
confirm audio, execute any enhancement method, render a model, or score output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCAN_PATH = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_member_scan_attempt02/member_scan.json"
ROSTER_PATH = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/v5_fit_expansion_roster.json"
DEFAULT_OUTPUT = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01"
EXPECTED_SCAN_SHA256 = "1b4fd8472842c9c30c4611aa760b3c2b2d9f01193d3aaf7f487f5f8096b0a140"
EXPECTED_ROSTER_SHA256 = "2fd42f097dbfb046c5bb5a2fba687c1812dc656fb06bf489974384f4eb78667a"
EXPECTED_FIT_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
EXPECTED_ADR013_SHA256 = "82a8a397f2317148d026ba4a2e0558b81656df0f1f74dcc4e58015b0323ab04b"
EXPECTED_AUDIO_IO_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
EXPECTED_BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
FIT_SPEAKERS = tuple(f"S{x:04d}" for x in range(52, 72))
MIN_SAMPLES = 51_200
MAX_SAMPLES = 128_000


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_pinned_json(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], str, int]:
    raw = path.read_bytes()
    digest = sha_bytes(raw)
    if digest != expected_sha:
        raise ValueError(f"{label} raw SHA mismatch: {digest}")
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object")
    return obj, digest, len(raw)


def safe_source_path(raw_path: str) -> Path:
    normalized = raw_path.replace("\\", os.sep).replace("/", os.sep)
    path = Path(normalized)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve(strict=True)
    expected_parent = (ROOT / "runs/recovered_20260930/local/qwen_source_axis/hf_source/archives").resolve(strict=True)
    if resolved.parent != expected_parent or resolved.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError(f"archive path is outside the frozen source directory: {resolved}")
    return resolved


def load_pcm16_wav_payload(raw_wav: bytes) -> tuple[int, bytes]:
    # The selected source bytes are parsed with the project’s pinned WAV parser.
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.audio_enhancement.audio_io import _parse_wave

    fmt_code, channels, sample_rate, bits, block_align, payload = _parse_wave(raw_wav)
    if (fmt_code, channels, sample_rate, bits) != (1, 1, 16_000, 16):
        raise ValueError(
            f"selected endpoint must remain mono 16 kHz PCM16 WAV: {(fmt_code, channels, sample_rate, bits)}"
        )
    if len(payload) % block_align:
        raise ValueError("WAV payload length is not a whole number of frames")
    count = len(payload) // block_align
    if not MIN_SAMPLES <= count <= MAX_SAMPLES:
        raise ValueError(f"selected endpoint sample count is outside the frozen range: {count}")
    return count, payload


def folds_from_roster(roster: dict[str, Any]) -> dict[str, int]:
    fold_rows = roster.get("five_speaker_folds", {}).get("folds")
    if not isinstance(fold_rows, list) or len(fold_rows) != 5:
        raise ValueError("frozen roster must contain five speaker folds")
    result: dict[str, int] = {}
    for fold in fold_rows:
        index = int(fold["fold_index"])
        for speaker in fold["held_out_speakers"]:
            if speaker in result:
                raise ValueError(f"speaker appears in multiple folds: {speaker}")
            result[str(speaker)] = index
    if set(result) != set(FIT_SPEAKERS) or any(sum(v == i for v in result.values()) != 4 for i in range(5)):
        raise ValueError("speaker folds do not match the 20-speaker fit roster with four speakers per fold")
    return result


def extract_endpoints(output_root: Path) -> dict[str, Any]:
    scan, scan_sha, scan_bytes = read_pinned_json(SCAN_PATH, EXPECTED_SCAN_SHA256, "member scan")
    roster, roster_sha, roster_bytes = read_pinned_json(ROSTER_PATH, EXPECTED_ROSTER_SHA256, "ADR013 roster")
    if scan.get("schema") != "cross_tfg_v5_fit_expansion_member_scan_adr013_v1" or scan.get("status") != "PASS_40_UNIQUE_ENDPOINTS":
        raise ValueError("member scan is not the reviewed complete 40-endpoint fit roster")
    if scan.get("roster_raw_sha256") != roster_sha or scan.get("plan_raw_sha256") != EXPECTED_FIT_PLAN_SHA256 or scan.get("adr013_raw_sha256") != EXPECTED_ADR013_SHA256:
        raise ValueError("member scan does not bind the pinned roster, fit plan, and ADR013")
    if roster.get("schema") != "cross_tfg_v5_fit_expansion_roster_adr013_v1":
        raise ValueError("unexpected fit roster schema")
    split = roster.get("split_binding", {})
    if split.get("fit_speaker_ids", [])[:20] != list(FIT_SPEAKERS):
        raise ValueError("fit speaker subset differs from the frozen S0052-S0071 cohort")
    if set(FIT_SPEAKERS) & (set(split.get("dev_speaker_ids", [])) | set(split.get("confirm_speaker_ids", []))):
        raise ValueError("fit endpoints overlap dev or confirmation rosters")
    if scan.get("scope", {}).get("dev_audio_read") is not False or scan.get("scope", {}).get("confirm_audio_read") is not False:
        raise ValueError("member scan reports a prohibited split read")
    folds = folds_from_roster(roster)

    selected = [row for row in scan["slot_rows"] if row.get("selected_by_adr013") is True]
    if len(scan["slot_rows"]) != 320 or len(selected) != 40:
        raise ValueError("expected all 320 slots and exactly 40 selected endpoints")
    expected_keys = {(speaker, j) for speaker in FIT_SPEAKERS for j in (0, 15)}
    actual_keys = {(row.get("speaker_id"), int(row.get("canonical_j", -1))) for row in selected}
    if actual_keys != expected_keys:
        raise ValueError("selected endpoints differ from exact ADR013 j=0/j=15 slots")
    if any(row.get("endpoint_payload_status") != "PASS" or row.get("duplicate_pcm_status") != "UNIQUE_AMONG_40_PLANNED_ENDPOINTS" or row.get("duplicate_sample_id_status") != "UNIQUE_AMONG_40_PLANNED_ENDPOINTS" for row in selected):
        raise ValueError("one or more selected source payloads failed the unique endpoint audit")
    if len({row["sample_id"] for row in selected}) != 40 or len({row["pcm_sha256"] for row in selected}) != 40:
        raise ValueError("selected endpoint IDs and PCM payload hashes must both be unique")
    if scan.get("summary", {}).get("endpoint_unique_pcm_count") != 40 or scan.get("summary", {}).get("missing_planned_endpoints") != 0:
        raise ValueError("member scan summary disagrees with endpoint rows")

    archive_by_speaker = {row["speaker_id"]: row for row in scan.get("archive_receipts", [])}
    if set(archive_by_speaker) != set(FIT_SPEAKERS):
        raise ValueError("archive receipt roster differs from fit speaker roster")

    output_root.mkdir(parents=True, exist_ok=False)
    wav_root = output_root / "source_wav"
    wav_root.mkdir()
    try:
        from scripts.audio_enhancement.audio_io import read_canonical_wav
    except ImportError:
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts.audio_enhancement.audio_io import read_canonical_wav

    output_rows: list[dict[str, Any]] = []
    archive_validation: dict[str, dict[str, Any]] = {}
    for speaker in FIT_SPEAKERS:
        receipt = archive_by_speaker[speaker]
        archive_path = safe_source_path(str(receipt["path"]))
        stat = archive_path.lstat()
        if archive_path.is_symlink() or not archive_path.is_file():
            raise ValueError(f"fit archive is not a regular non-symlink file: {archive_path}")
        actual_archive_sha, actual_archive_bytes = sha_file(archive_path)
        if actual_archive_sha != receipt["actual_sha256"] or actual_archive_bytes != int(receipt["actual_bytes"]) or actual_archive_sha != receipt["official_lfs_sha256"]:
            raise ValueError(f"fit archive no longer matches its sealed official bytes: {speaker}")
        archive_validation[speaker] = {
            "path": str(archive_path),
            "bytes": actual_archive_bytes,
            "sha256": actual_archive_sha,
            "source_receipt_sha256": receipt["actual_sha256"],
            "mtime_ns": int(stat.st_mtime_ns),
        }
        speaker_rows = sorted((row for row in selected if row["speaker_id"] == speaker), key=lambda row: int(row["canonical_j"]))
        if [int(row["canonical_j"]) for row in speaker_rows] != [0, 15]:
            raise ValueError(f"expected exactly endpoint j=0 and j=15 for {speaker}")
        with tarfile.open(archive_path, mode="r:gz") as tar:
            for row in speaker_rows:
                member_name = str(row["selected_member_path"])
                safe_name = PurePosixPath(member_name)
                if safe_name.is_absolute() or ".." in safe_name.parts or safe_name.parts[:2] != ("train", speaker):
                    raise ValueError(f"selected archive member escapes the fit speaker tree: {member_name}")
                try:
                    info = tar.getmember(member_name)
                except KeyError as exc:
                    raise ValueError(f"selected endpoint missing from source archive: {member_name}") from exc
                if not info.isfile() or int(info.size) != int(row["source_member_bytes"]):
                    raise ValueError(f"selected endpoint archive member type/size changed: {member_name}")
                member_file = tar.extractfile(info)
                if member_file is None:
                    raise ValueError(f"cannot read selected endpoint: {member_name}")
                raw_wav = member_file.read()
                member_file.close()
                raw_sha = sha_bytes(raw_wav)
                if len(raw_wav) != int(row["source_member_bytes"]) or raw_sha != row["source_member_sha256"]:
                    raise ValueError(f"selected raw WAV member SHA mismatch: {member_name}")
                sample_count, pcm_payload = load_pcm16_wav_payload(raw_wav)
                pcm_sha = sha_bytes(pcm_payload)
                if pcm_sha != row["pcm_sha256"]:
                    raise ValueError(f"selected PCM payload SHA mismatch: {member_name}")
                sample_id = str(row["sample_id"])
                if not re.fullmatch(r"BAC\d{3}S\d{4}W\d{4}", sample_id):
                    raise ValueError(f"unexpected AISHELL sample ID syntax: {sample_id}")
                wav_path = wav_root / f"{sample_id}.wav"
                with wav_path.open("xb") as stream:
                    stream.write(raw_wav)
                canonical_samples, audio_info = read_canonical_wav(wav_path, target_rate=16_000)
                if audio_info.source_sample_rate != 16_000 or audio_info.source_channels != 1 or audio_info.source_frames != sample_count or audio_info.frames != sample_count or audio_info.source_encoding != "PCM_16":
                    raise ValueError(f"source WAV is not exact canonical PCM16 mono: {sample_id}")
                output_sha, output_bytes = sha_file(wav_path)
                if output_sha != raw_sha or output_bytes != len(raw_wav):
                    raise RuntimeError(f"materialized source WAV differs from its archive payload: {sample_id}")
                output_rows.append({
                    "row_id": f"{speaker}_j{int(row['canonical_j']):02d}_{sample_id}",
                    "speaker_id": speaker,
                    "fold_index": folds[speaker],
                    "canonical_j": int(row["canonical_j"]),
                    "sample_id": sample_id,
                    "source_archive_path": str(archive_path),
                    "source_archive_sha256": actual_archive_sha,
                    "source_archive_member": member_name,
                    "source_member_sha256": raw_sha,
                    "source_member_bytes": len(raw_wav),
                    "source_pcm_payload_sha256": pcm_sha,
                    "canonical_float32_pcm_sha256": audio_info.pcm_float32_sha256,
                    "sample_count": sample_count,
                    "sample_rate_hz": 16_000,
                    "channels": 1,
                    "source_encoding": "PCM_16",
                    "transfer_local_path": str(wav_path.resolve()),
                    "transfer_relative_path": f"source_wav/{sample_id}.wav",
                    "transfer_file_sha256": output_sha,
                    "transfer_file_bytes": output_bytes,
                    "transport_status": "NOT_TRANSFERRED",
                })

    output_rows.sort(key=lambda row: (FIT_SPEAKERS.index(row["speaker_id"]), row["canonical_j"]))
    if len(output_rows) != 40:
        raise RuntimeError(f"materialized unexpected number of endpoint WAVs: {len(output_rows)}")
    config_module_path = ROOT / "scripts/audio_enhancement/localcompress_r090_control_attempt01.py"
    from scripts.audio_enhancement.localcompress_r090_control_attempt01 import production_config

    config = production_config()
    config_bytes = canonical_json(config) + b"\n"
    config_path = output_root / "localcompress_r090_production_config.json"
    with config_path.open("xb") as stream:
        stream.write(config_bytes)
    config_sha = sha_bytes(config_bytes)
    scanner_path = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/scan_v5_fit_expansion_members_attempt02.py"
    audio_io_path = ROOT / "scripts/audio_enhancement/audio_io.py"
    baseline_path = ROOT / "scripts/audio_enhancement/baselines.py"
    profile_path = ROOT / "scripts/audio_enhancement/shared_frequency_profile_v1.py"
    diagnostic_helper = ROOT / "scripts/audio_enhancement_eval/decompose_v4_delta_common_mode_attempt04.py"
    manifest = {
        "schema": "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1",
        "status": "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED",
        "created_by": Path(__file__).relative_to(ROOT).as_posix(),
        "created_by_raw_sha256": sha_file(Path(__file__))[0],
        "split": "historical_fit_only",
        "fit_speakers": list(FIT_SPEAKERS),
        "selection_rule": "ADR013 exact canonical endpoints j=0 and j=15 for S0052-S0071; no replacement",
        "selection_scan": {
            "path": SCAN_PATH.relative_to(ROOT).as_posix(),
            "raw_sha256": scan_sha,
            "bytes": scan_bytes,
            "roster_raw_sha256": roster_sha,
            "roster_bytes": roster_bytes,
            "fit_plan_raw_sha256": EXPECTED_FIT_PLAN_SHA256,
            "adr013_raw_sha256": EXPECTED_ADR013_SHA256,
            "archive_receipts": archive_validation,
        },
        "input_contract": {
            "container": "RIFF/WAVE integer PCM16, mono, 16000 Hz",
            "canonical_decode": "project audio_io.read_canonical_wav; exact identity rate/channel mapping for these 40 members; PCM16 samples divide by 32768 into float32",
            "sample_count_inclusive": [MIN_SAMPLES, MAX_SAMPLES],
            "source_clock": "original 16 kHz sample clock; no resampling or time shift",
        },
        "r090_control": {
            "status": "PREPARED_NOT_FROZEN",
            "config_path": config_path.relative_to(ROOT).as_posix(),
            "config_raw_sha256": config_sha,
            "config_bytes": len(config_bytes),
            "implementation_path": config_module_path.relative_to(ROOT).as_posix(),
            "implementation_raw_sha256": sha_file(config_module_path)[0],
            "config": config,
        },
        "runtime_transfer_metadata": {
            "planned_input_rows": 40,
            "planned_speakers": 20,
            "planned_endpoints_per_speaker": 2,
            "frontends": ["Wav2Lip", "MuseTalk", "Ditto"],
            "candidate_render_rows": 0,
            "transfer_status": "LOCAL_SOURCE_WAVS_READY; REMOTE_DESTINATION_NOT_ASSIGNED",
            "remote_namespace": None,
            "remote_interpreter": None,
            "model_effect_run": False,
            "candidate_effect_read": False,
            "dev_audio_read": False,
            "confirm_audio_read": False,
            "fit_method_roster": ["identity", "rms_fixed", "localcompress_r090"],
            "fit_method_roster_status": "PREPARATION_ONLY_REQUIRES_POST_SHARED540_METHOD_FREEZE",
            "max_v5_candidate_or_ablation_methods": 6,
            "localcompress_r090_uses_v5_candidate_or_ablation_slot": True,
            "remaining_transition_candidate_slots_if_r090_is_retained": 5,
            "max_methods_per_clip_per_frontend_if_full_slate": 8,
            "max_future_rows_one_seed_three_frontends": 960,
            "budget_note": "The r090 envelope-only method occupies one of the existing six V5 candidate/ablation slots; identity and fixed RMS are the two separate controls. Five transition-mask slots remain if r090 is retained. The slate and coefficients remain unfrozen.",
            "next_remote_requirements": [
                "copy only the 40 sealed fit source WAVs and manifest to a fresh non-confirm namespace",
                "bind the frozen frontend adapters, exact source-clock maps, and direct-original PCM16 scorer",
                "freeze candidate slate and method order only after shared540 gate review",
                "preserve all 40 endpoint denominators and five hash-assigned speaker folds",
            ],
        },
        "source_implementation_pins": {
            "member_scan_script": {"path": scanner_path.relative_to(ROOT).as_posix(), "raw_sha256": sha_file(scanner_path)[0]},
            "audio_io": {"path": audio_io_path.relative_to(ROOT).as_posix(), "raw_sha256": sha_file(audio_io_path)[0]},
            "baselines": {"path": baseline_path.relative_to(ROOT).as_posix(), "raw_sha256": sha_file(baseline_path)[0]},
            "shared_profile_reference": {"path": profile_path.relative_to(ROOT).as_posix(), "raw_sha256": sha_file(profile_path)[0]},
            "v4_envelope_flux_helper": {"path": diagnostic_helper.relative_to(ROOT).as_posix(), "raw_sha256": sha_file(diagnostic_helper)[0]},
        },
        "rows": output_rows,
        "scope": {
            "fit_audio_members_read": 40,
            "fit_archives_rehashed": 20,
            "dev_audio_read": False,
            "confirm_audio_read": False,
            "candidate_transform_executed": False,
            "model_forward": False,
            "render": False,
            "score": False,
        },
    }
    manifest_raw = canonical_json(manifest) + b"\n"
    manifest_path = output_root / "fit40_source_transfer_manifest.json"
    with manifest_path.open("xb") as stream:
        stream.write(manifest_raw)
    return {
        "status": manifest["status"],
        "output_root": str(output_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_bytes": len(manifest_raw),
        "manifest_raw_sha256": sha_bytes(manifest_raw),
        "source_wav_count": len(output_rows),
        "source_wav_total_bytes": sum(row["transfer_file_bytes"] for row in output_rows),
        "unique_speakers": len({row["speaker_id"] for row in output_rows}),
        "archive_rehashed_count": len(archive_validation),
        "candidate_transform_executed": False,
        "dev_audio_read": False,
        "confirm_audio_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = extract_endpoints(args.output_dir.resolve())
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
