from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_ZIP_BYTES = 3_548_546
EXPECTED_ZIP_SHA256 = "6f9de4953da6fd52a4e916604c21b0fd187ab145fefc92ea47cd8e274a460d9a"
EXPECTED_MANIFEST_BYTES = 24_072
EXPECTED_MANIFEST_SHA256 = "fff18619b7be8f992002f99c33121ef3eb25acce6aa3caa3da55f398eb3faffa"
EXPECTED_MANIFEST_SELF_SHA256 = "ad151b990fd8aee6f8aca7b63458231adeaf88713ecbbde5fa36549ef856ae88"
READER_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_sha(obj: dict[str, Any]) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw)


def fail(message: str) -> None:
    raise RuntimeError(message)


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        fail(f"unsafe ZIP member path: {name!r}")
    return path


def parse_pcm16_mono_16k(raw: bytes, row: dict[str, Any]) -> bytes:
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        fail(f"invalid RIFF/WAVE source: {row['sample_id']}")
    declared_size = struct.unpack_from("<I", raw, 4)[0] + 8
    if declared_size != len(raw):
        fail(f"RIFF size mismatch: {row['sample_id']}")
    pos = 12
    fmt: tuple[int, int, int, int, int] | None = None
    data_chunks: list[bytes] = []
    while pos + 8 <= declared_size:
        chunk_id = raw[pos : pos + 4]
        size = struct.unpack_from("<I", raw, pos + 4)[0]
        start = pos + 8
        end = start + size
        if end > declared_size:
            fail(f"truncated WAV chunk: {row['sample_id']}")
        chunk = raw[start:end]
        if chunk_id == b"fmt ":
            if len(chunk) < 16:
                fail(f"short fmt chunk: {row['sample_id']}")
            code, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", chunk, 0)
            fmt = (code, channels, rate, block_align, bits)
        elif chunk_id == b"data":
            data_chunks.append(chunk)
        pos = end + (size & 1)
    if pos != declared_size or fmt is None or not data_chunks:
        fail(f"incomplete WAV structure: {row['sample_id']}")
    code, channels, rate, block_align, bits = fmt
    pcm = b"".join(data_chunks)
    expected = row["riff"]
    count = len(pcm) // 2
    if (
        (code, channels, rate, bits) != (1, 1, 16_000, 16)
        or block_align != 2
        or len(pcm) % 2
        or count != expected["sample_count"]
        or expected["format_code"] != code
        or expected["channels"] != channels
        or expected["sample_rate_hz"] != rate
        or expected["bits_per_sample"] != bits
    ):
        fail(f"PCM16 header contract mismatch: {row['sample_id']}")
    return pcm


def canonical_f32_bytes(pcm: bytes, sample_id: str) -> tuple[bytes, int]:
    if len(pcm) % 2:
        fail(f"PCM16 payload alignment mismatch: {sample_id}")
    out = bytearray(len(pcm) * 2)
    cursor = 0
    for (sample,) in struct.iter_unpack("<h", pcm):
        value = sample / 32768.0
        if not math.isfinite(value):
            fail(f"non-finite canonical sample: {sample_id}")
        struct.pack_into("<f", out, cursor, value)
        cursor += 4
    return bytes(out), len(pcm) // 2


def exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Never remove a partial output; the caller must preserve and inspect it.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    bundle_path = Path(args.bundle)
    manifest_path = Path(args.manifest)
    destination = Path(args.destination)
    if destination.exists() or destination.is_symlink():
        fail("create-only destination already exists")

    bundle = bundle_path.read_bytes()
    sidecar = manifest_path.read_bytes()
    if len(bundle) != EXPECTED_ZIP_BYTES or sha256(bundle) != EXPECTED_ZIP_SHA256:
        fail("uploaded bundle size/SHA mismatch")
    if len(sidecar) != EXPECTED_MANIFEST_BYTES or sha256(sidecar) != EXPECTED_MANIFEST_SHA256:
        fail("uploaded manifest size/SHA mismatch")
    manifest = json.loads(sidecar)
    expected_self = manifest.pop("self_sha256", None)
    if expected_self != EXPECTED_MANIFEST_SELF_SHA256 or compact_sha(manifest) != expected_self:
        fail("portable manifest self-hash mismatch")
    manifest["self_sha256"] = expected_self
    if manifest["status"] != "SEALED_SOURCE_ONLY_FUTURE_DEV20":
        fail("manifest cohort status mismatch")
    if manifest["canonicalization"]["reader_source_sha256"] != READER_SHA256:
        fail("pinned canonical reader identity mismatch")
    rows = manifest["rows"]
    if len(rows) != 20 or len({row["sample_id"] for row in rows}) != 20:
        fail("manifest does not contain exactly 20 unique source rows")
    speaker_counts: dict[str, int] = {}
    for row in rows:
        speaker_counts[row["speaker_id"]] = speaker_counts.get(row["speaker_id"], 0) + 1
    if sorted(speaker_counts) != manifest["selection"]["selected_speakers"] or set(speaker_counts.values()) != {2}:
        fail("manifest is not the frozen ten-speaker, two-row cohort")

    evidence = manifest["evidence"]
    expected_names = {"source_manifest.json"}
    expected_names.update(safe_member(item["archive_path"]).as_posix() for item in evidence)
    expected_names.update(safe_member(row["bundle_path"]).as_posix() for row in rows)
    outputs: list[tuple[str, bytes, str, str]] = []

    destination_parent = destination.parent
    if not destination_parent.is_dir():
        fail("destination parent must already exist")
    with zipfile.ZipFile(bundle_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != expected_names:
            fail("ZIP member set differs from the pinned manifest")
        if archive.testzip() is not None:
            fail("ZIP CRC failure")
        if archive.read("source_manifest.json") != sidecar:
            fail("manifest sidecar differs from ZIP manifest member")
        outputs.append(("source_manifest.json", sidecar, EXPECTED_MANIFEST_SHA256, "source_manifest"))
        for evidence_row in evidence:
            name = evidence_row["archive_path"]
            data = archive.read(name)
            if len(data) != evidence_row["bytes"] or sha256(data) != evidence_row["sha256"]:
                fail(f"evidence member size/SHA mismatch: {name}")

        # Reconstruct all WAV and raw little-endian float32 products in memory,
        # verify the full roster, then begin create-only writes.
        qc_rows: list[dict[str, Any]] = []
        for row in rows:
            wav_name = row["bundle_path"]
            wav_raw = archive.read(wav_name)
            if len(wav_raw) != row["bytes"] or sha256(wav_raw) != row["sha256"]:
                fail(f"WAV source size/SHA mismatch: {row['sample_id']}")
            pcm = parse_pcm16_mono_16k(wav_raw, row)
            f32, count = canonical_f32_bytes(pcm, row["sample_id"])
            f32sha = sha256(f32)
            expected_canonical = row["canonical_float32"]
            if (
                count != expected_canonical["sample_count"]
                or count != row["riff"]["sample_count"]
                or len(f32) != count * 4
                or f32sha != expected_canonical["pcm_sha256"]
                or expected_canonical["reader_source_sha256"] != READER_SHA256
                or expected_canonical["resampler"] != "none"
            ):
                fail(f"independent raw-F32 reconstruction mismatch: {row['sample_id']}")
            f32_name = f"canonical_float32/train/{row['speaker_id']}/{row['basename'][:-4]}.f32"
            outputs.append((wav_name, wav_raw, row["sha256"], row["sample_id"]))
            outputs.append((f32_name, f32, f32sha, row["sample_id"]))
            qc_rows.append(
                {
                    "speaker_id": row["speaker_id"],
                    "sample_id": row["sample_id"],
                    "source_member": wav_name,
                    "source_bytes": len(wav_raw),
                    "source_sha256": sha256(wav_raw),
                    "canonical_member": f32_name,
                    "canonical_bytes": len(f32),
                    "canonical_sha256": f32sha,
                    "sample_rate_hz": 16_000,
                    "sample_count": count,
                    "dtype": "<f4",
                    "decoder": "standard-library RIFF PCM16 parse + signed-int16 / 32768.0 IEEE-F32 little-endian reconstruction",
                }
            )
        for evidence_row in evidence:
            outputs.append((evidence_row["archive_path"], archive.read(evidence_row["archive_path"]), evidence_row["sha256"], evidence_row["archive_path"]))

    # Create the stage only after complete archive/input validation.
    destination.mkdir(mode=0o755, parents=False, exist_ok=False)
    verified_outputs: list[dict[str, Any]] = []
    for member_name, data, expected_sha, identity in outputs:
        rel = safe_member(member_name)
        out_path = destination.joinpath(*rel.parts)
        exclusive_write(out_path, data)
        readback = out_path.read_bytes()
        got_sha = sha256(readback)
        if len(readback) != len(data) or got_sha != expected_sha:
            fail(f"create-only output readback mismatch: {member_name}")
        verified_outputs.append(
            {"member": member_name, "bytes": len(readback), "sha256": got_sha, "identity": identity}
        )

    qc: dict[str, Any] = {
        "schema": "aishell_future_dev20_remote_source_stage_qc_attempt01_v1",
        "status": "ALL_20_SOURCE_AND_RAW_F32_VERIFIED_CREATE_ONLY",
        "source_manifest": {
            "bytes": len(sidecar),
            "sha256": sha256(sidecar),
            "self_sha256": expected_self,
        },
        "bundle": {
            "bytes": len(bundle),
            "sha256": sha256(bundle),
            "member_count": len(expected_names),
            "wav_count": len(rows),
            "evidence_count": len(evidence),
        },
        "stage_root": str(destination),
        "source_file_count": 20,
        "source_file_total_bytes": sum(row["bytes"] for row in rows),
        "canonical_f32_file_count": 20,
        "canonical_f32_total_bytes": sum(row["canonical_bytes"] for row in qc_rows),
        "speaker_counts": speaker_counts,
        "sample_clock_preserved": True,
        "excluded_or_confirm_audio_accessed": False,
        "asr_or_scores_run": False,
        "model_or_gpu_run": False,
        "independent_f32_method": "This worker reconstructs each complete raw little-endian float32 stream from the exact verified PCM16 WAV bytes using Python standard-library RIFF parsing and IEEE-754 packing; each full F32 byte stream is written and read back, then checked against the locally pinned canonical F32 SHA-256.",
        "rows": qc_rows,
        "written_outputs": verified_outputs,
    }
    qc["self_sha256"] = compact_sha(qc)
    qc_raw = json.dumps(qc, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    qc_path = destination / "source_stage_qc_attempt01.json"
    exclusive_write(qc_path, qc_raw)
    qc_readback = qc_path.read_bytes()
    if qc_readback != qc_raw:
        fail("QC receipt readback mismatch")
    # Emit one parseable JSON object only; the bridge caller validates stdout.
    print(
        json.dumps(
            {
                "status": qc["status"],
                "receipt_path": str(qc_path),
                "receipt_bytes": len(qc_readback),
                "receipt_sha256": sha256(qc_readback),
                "source_file_count": 20,
                "canonical_f32_file_count": 20,
                "sample_clock_preserved": True,
                "model_or_gpu_run": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
