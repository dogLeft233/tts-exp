from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

PACKAGE_DIR = Path(r"E:\Documents\tts-exp\runs\recovered_20260930\remote\world_v6_consumer04_attempt02\report_chunks_attempt02")
OUTPUT_DIR = Path(r"E:\Documents\tts-exp\runs\recovered_20260930\independent_validation\world_v6_consumer05_attempt02_readback")
EXPECTED_MANIFEST_BYTES = 2061
EXPECTED_MANIFEST_SHA256 = "227531a1df2cf98e82745cc2d7813801963483960c102266dc802c93d9f90dbb"
EXPECTED_PART = {
    "name": "report.part000",
    "size_bytes": 640288,
    "sha256": "85de8605925d5d3692ca06e6865dde7f6144e5d24a978ddf71814d9e1ddc58c2",
}
EXPECTED_CONSUMER_SOURCE_SHA256 = "8e40c3017ca7c53924302522678c18854c9b0702ce2223821b03d7fd05166a92"
EXPECTED_QA04_SHA256 = "e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d"
EXPECTED_MEMBERS = {
    "world_consumer_statistics.json": (13183106, "b11f5530820ac82d654d77240bbb05b93ef2b6ab554a8e2c05e1b34fb9313b88"),
    "world_consumer_audit.json": (144139, "fa06babb3a106bf1ed3acc337a6dc3bddc2c7df35dbd176f6c545c80cf599e30"),
    "world_consumer_protocol.json": (3682, "e2e61f18c221123ccd43433333a405de685e3b0d6e77d3e082b21b11aca7aae5"),
    "consumer_all_files_sha256.json": (325, "097736a5a6cfb74882ca2f005001f34223370f6f86d84ff4ad9536de6ce003ce"),
    "consumer_output_seal.json": (723, "5f4f6677b5740a010e89120670f751a16b5e929899545b9be2e7856340290350"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(name: str) -> str:
    p = PurePosixPath(name)
    if not name or p.is_absolute() or "\\" in name or any(part in {"", ".", ".."} for part in p.parts):
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    normalized = p.as_posix()
    if normalized != name or "/" in normalized:
        raise RuntimeError(f"unexpected nested report member: {name!r}")
    return normalized


def verify_and_extract() -> dict[str, object]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    part_path = PACKAGE_DIR / EXPECTED_PART["name"]
    if OUTPUT_DIR.exists() or OUTPUT_DIR.is_symlink():
        raise RuntimeError(f"refusing existing readback output: {OUTPUT_DIR}")
    if not manifest_path.is_file() or not part_path.is_file():
        raise RuntimeError("downloaded package manifest or part is missing")

    manifest_bytes = manifest_path.read_bytes()
    if len(manifest_bytes) != EXPECTED_MANIFEST_BYTES or sha256_bytes(manifest_bytes) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("package manifest size/hash differs from sealed transfer receipt")
    part_bytes = part_path.read_bytes()
    if len(part_bytes) != EXPECTED_PART["size_bytes"] or sha256_bytes(part_bytes) != EXPECTED_PART["sha256"]:
        raise RuntimeError("archive part size/hash differs from sealed transfer receipt")

    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise RuntimeError("package manifest schema/status mismatch")
    if manifest.get("consumer_source_sha256") != EXPECTED_CONSUMER_SOURCE_SHA256:
        raise RuntimeError("consumer source seal differs")
    if manifest.get("producer_qa04_sha256") != EXPECTED_QA04_SHA256:
        raise RuntimeError("physical QA04 seal differs")
    if manifest.get("producer_run_root") != "/dev/shm/tts-research-20261001/world_phone_transfer_gpu_attempt07_v6/run":
        raise RuntimeError("producer run root differs")
    if manifest.get("parts") != [EXPECTED_PART]:
        raise RuntimeError("package part inventory differs")
    if manifest.get("archive_bytes") != len(part_bytes) or manifest.get("archive_sha256") != sha256_bytes(part_bytes):
        raise RuntimeError("whole archive size/hash differs from package manifest")
    listed = manifest.get("members")
    if not isinstance(listed, list) or len(listed) != len(EXPECTED_MEMBERS):
        raise RuntimeError("package member count differs")
    member_map: dict[str, dict[str, object]] = {}
    for row in listed:
        name = safe_member(str(row.get("path", "")))
        if name in member_map:
            raise RuntimeError(f"duplicate package member: {name}")
        member_map[name] = row
    if set(member_map) != set(EXPECTED_MEMBERS):
        raise RuntimeError("package member names differ")
    for name, (size, digest) in EXPECTED_MEMBERS.items():
        row = member_map[name]
        if row.get("size_bytes") != size or row.get("sha256") != digest:
            raise RuntimeError(f"sealed consumer member reference differs: {name}")

    OUTPUT_DIR.mkdir(parents=True, mode=0o700, exist_ok=False)
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(mode=0o700)
    with (archive_dir / EXPECTED_PART["name"]).open("xb") as out:
        out.write(part_bytes)
        out.flush()
        os.fsync(out.fileno())
    (OUTPUT_DIR / "package_manifest.json").write_bytes(manifest_bytes)

    seen: set[str] = set()
    observed: dict[str, dict[str, object]] = {}
    with tarfile.open(fileobj=io.BytesIO(part_bytes), mode="r:gz") as archive:
        for member in archive:
            name = safe_member(member.name)
            if name in seen:
                raise RuntimeError(f"duplicate tar member: {name}")
            seen.add(name)
            if name not in EXPECTED_MEMBERS or not member.isfile():
                raise RuntimeError(f"unlisted or non-regular tar member: {name}")
            expected_size, expected_sha = EXPECTED_MEMBERS[name]
            if member.size != expected_size:
                raise RuntimeError(f"tar member size differs: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"tar member data missing: {name}")
            target = OUTPUT_DIR / name
            digest = hashlib.sha256()
            count = 0
            with source, target.open("xb") as out:
                for block in iter(lambda: source.read(1 << 20), b""):
                    out.write(block)
                    digest.update(block)
                    count += len(block)
                out.flush()
                os.fsync(out.fileno())
            observed_sha = digest.hexdigest()
            if count != expected_size or observed_sha != expected_sha:
                raise RuntimeError(f"extracted member differs from expected seal: {name}")
            observed[name] = {"size_bytes": count, "sha256": observed_sha}
    if seen != set(EXPECTED_MEMBERS):
        raise RuntimeError("tar member coverage differs from sealed five-member inventory")
    if sha256_file(OUTPUT_DIR / "world_consumer_statistics.json") != EXPECTED_MEMBERS["world_consumer_statistics.json"][1]:
        raise RuntimeError("statistics file readback SHA mismatch")

    seal = json.loads((OUTPUT_DIR / "consumer_output_seal.json").read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED" or seal.get("consumer_source_sha256") != EXPECTED_CONSUMER_SOURCE_SHA256:
        raise RuntimeError("consumer output seal content does not bind expected source")
    if seal.get("producer_qa04_sha256") != EXPECTED_QA04_SHA256:
        raise RuntimeError("consumer output seal does not bind QA04")
    result = {
        "status": "PASS",
        "package_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "package_part_sha256": EXPECTED_PART["sha256"],
        "whole_archive_sha256": manifest["archive_sha256"],
        "consumer_source_sha256": seal["consumer_source_sha256"],
        "producer_run_root": manifest["producer_run_root"],
        "producer_plan_raw_file_sha256": manifest.get("producer_plan_raw_file_sha256"),
        "producer_plan_embedded_fingerprint_sha256": manifest.get("producer_plan_embedded_fingerprint_sha256"),
        "producer_qa04_sha256": seal["producer_qa04_sha256"],
        "consumer_inventory_sha256": seal["inventory_sha256"],
        "member_count": len(observed),
        "members": observed,
        "statistics_content_loaded_as_json": False,
        "output_dir": str(OUTPUT_DIR),
    }
    result_path = OUTPUT_DIR / "readback_verification.json"
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify_and_extract(), sort_keys=True, separators=(",", ":")))