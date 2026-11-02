#!/usr/bin/env python3
"""Build a QA-gated inventory for the frozen WORLD EVAL NPZ inputs.

The output is a small member manifest; this script does not package or remove
scientific files. It refuses to run unless the independent EVAL waveform QA is
sealed PASS and every referenced NPZ matches the physical manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PHYSICAL_MANIFEST_SHA256 = "27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3"
PHYSICAL_FINGERPRINT_SHA256 = "eb77432936c3454a241eb2505a966ffa0d05aa5be8514451cc2da66aa7ed9de8"
PHYSICAL_PLAN_SHA256 = "f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e"
CAL_FREEZE_SHA256 = "0a5b694975efeb07b9d1e9c0beb10f0142bb4203ba4a026d7ff965477991ab9d"
INPUT_MANIFEST_SHA256 = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
TTS_PAIR_MANIFEST_SHA256 = "2aea4750a02125629d82f85b4842932465a202dd03eec96864d096d8856070d2"
SPEC_SHA256 = "f085e2915349523891a478e3fc4dd1dada8493d68d3d73608bcc042d4a4e62ca"
PRODUCER_SOURCE_SHA256 = "8143f5d51831693d03a4b9ce1f8f3a42134b5d903ed9111f749e61cc878b3e95"
QA_SOURCE_SHA256 = "cf42e67bbb98f3e72656d6f0d46a6f3ded6ae4bea31b721b3d7551eebbb99d4f"
EXPECTED_NPZ_COUNT = 1920


class ArchiveManifestError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def embedded_fingerprint(value: dict[str, Any], label: str) -> str:
    expected = value.get("manifest_sha256")
    observed = sha_bytes(canonical({key: item for key, item in value.items() if key != "manifest_sha256"}))
    if expected != observed:
        raise ArchiveManifestError(f"{label} embedded fingerprint mismatch")
    return observed


def iter_npz_refs(value: Any):
    if isinstance(value, dict):
        candidate_path = value.get("path")
        candidate_sha = value.get("file_sha256")
        if isinstance(candidate_path, str) and candidate_path.lower().endswith(".npz"):
            if not isinstance(candidate_sha, str) or len(candidate_sha) != 64:
                raise ArchiveManifestError(f"NPZ path lacks a 64-character file SHA: {candidate_path}")
            yield candidate_path, candidate_sha
        for child in value.values():
            yield from iter_npz_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_npz_refs(child)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-manifest", type=Path, required=True)
    parser.add_argument("--physical-plan", type=Path, required=True)
    parser.add_argument("--cal-freeze", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--tts-pair-manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--producer-source", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    parser.add_argument("--qa-execution-record", type=Path, required=True)
    parser.add_argument("--qa-source", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fixed_files = {
        "physical_manifest": (args.physical_manifest, PHYSICAL_MANIFEST_SHA256),
        "physical_plan": (args.physical_plan, PHYSICAL_PLAN_SHA256),
        "cal_freeze": (args.cal_freeze, CAL_FREEZE_SHA256),
        "input_manifest": (args.input_manifest, INPUT_MANIFEST_SHA256),
        "tts_pair_manifest": (args.tts_pair_manifest, TTS_PAIR_MANIFEST_SHA256),
        "spec": (args.spec, SPEC_SHA256),
        "producer_source": (args.producer_source, PRODUCER_SOURCE_SHA256),
        "qa_source": (args.qa_source, QA_SOURCE_SHA256),
    }
    source_seals = {}
    for label, (raw_path, expected_sha) in fixed_files.items():
        path = raw_path.resolve(strict=True)
        observed_sha = sha_file(path)
        if observed_sha != expected_sha:
            raise ArchiveManifestError(f"{label} SHA mismatch")
        source_seals[label] = {"path": str(path), "sha256": observed_sha, "size_bytes": path.stat().st_size}

    physical = load_json(args.physical_manifest.resolve(strict=True))
    plan = load_json(args.physical_plan.resolve(strict=True))
    freeze = load_json(args.cal_freeze.resolve(strict=True))
    qa = load_json(args.qa_report.resolve(strict=True))
    qa_record = load_json(args.qa_execution_record.resolve(strict=True))
    physical_fingerprint = embedded_fingerprint(physical, "physical EVAL manifest")
    plan_fingerprint = embedded_fingerprint(plan, "physical EVAL plan")
    if physical_fingerprint != PHYSICAL_FINGERPRINT_SHA256:
        raise ArchiveManifestError("physical EVAL fingerprint differs from the frozen value")
    if freeze.get("status") != "FROZEN":
        raise ArchiveManifestError("independent CAL freeze is not frozen")
    if qa.get("status") != "PASS":
        raise ArchiveManifestError("independent EVAL waveform QA is not PASS")
    if qa_record.get("qa_source_sha256") != QA_SOURCE_SHA256:
        raise ArchiveManifestError("QA execution record is not bound to the sealed attempt02 source")
    if qa_record.get("output_sha256") != sha_file(args.qa_report.resolve(strict=True)):
        raise ArchiveManifestError("QA execution record does not bind the exact report")

    expected_qa_inputs = {
        "input_manifest": INPUT_MANIFEST_SHA256,
        "tts_pair_manifest": TTS_PAIR_MANIFEST_SHA256,
        "physical_manifest": PHYSICAL_MANIFEST_SHA256,
        "physical_plan": PHYSICAL_PLAN_SHA256,
        "cal_freeze": CAL_FREEZE_SHA256,
        "spec": SPEC_SHA256,
        "producer_source": PRODUCER_SOURCE_SHA256,
    }
    for label, expected_sha in expected_qa_inputs.items():
        row = qa.get("inputs", {}).get(label, {})
        record_row = qa_record.get("input_seals", {}).get(label, {})
        if row.get("sha256") != expected_sha or record_row.get("sha256") != expected_sha:
            raise ArchiveManifestError(f"QA input seal mismatch: {label}")

    coverage = qa.get("coverage", {})
    required_coverage = {
        "sample_pairs": 80,
        "recipient_contexts": 160,
        "completed_recipient_contexts": 160,
        "unique_source_raw_wavs": 160,
        "unique_physical_wavs": 1920,
        "unique_npz_artifacts": EXPECTED_NPZ_COUNT,
    }
    for key, expected in required_coverage.items():
        if coverage.get(key) != expected:
            raise ArchiveManifestError(f"QA coverage mismatch: {key}")

    artifact_root = args.artifact_root.resolve(strict=True)
    referenced: dict[str, dict[str, Any]] = {}
    for raw_path, expected_sha in iter_npz_refs(physical):
        source_path = Path(raw_path)
        if source_path.is_symlink():
            raise ArchiveManifestError(f"referenced NPZ is a symlink: {raw_path}")
        path = source_path.resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".npz":
            raise ArchiveManifestError(f"referenced NPZ is not a regular NPZ file: {raw_path}")
        try:
            relative = path.relative_to(artifact_root).as_posix()
        except ValueError as exc:
            raise ArchiveManifestError(f"NPZ reference escapes the fixed artifact root: {raw_path}") from exc
        observed_sha = sha_file(path)
        if observed_sha != expected_sha:
            raise ArchiveManifestError(f"referenced NPZ SHA differs from physical manifest: {relative}")
        record = {"path": str(path), "archive_path": relative, "size_bytes": path.stat().st_size, "sha256": observed_sha}
        previous = referenced.get(relative)
        if previous is not None and previous != record:
            raise ArchiveManifestError(f"duplicate NPZ references disagree: {relative}")
        referenced[relative] = record

    if len(referenced) != EXPECTED_NPZ_COUNT:
        raise ArchiveManifestError(f"unique NPZ reference count {len(referenced)} != {EXPECTED_NPZ_COUNT}")
    on_disk = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*.npz")
        if path.is_file() and not path.is_symlink()
    }
    if on_disk != set(referenced):
        missing = sorted(set(referenced) - on_disk)[:10]
        extra = sorted(on_disk - set(referenced))[:10]
        raise ArchiveManifestError(f"physical NPZ directory differs from manifest refs: missing={missing}, extra={extra}")

    members = [referenced[key] for key in sorted(referenced)]
    payload = {
        "schema_version": 1,
        "manifest_type": "world_physical_eval_npz_archive_member_manifest",
        "status": "SEALED_READ_ONLY_INVENTORY",
        "classification": "1920 hash-verified physical EVAL NPZ inputs; source data retained until archive verification and restore checks pass",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "source_seals": source_seals,
        "physical_manifest_fingerprint_sha256": physical_fingerprint,
        "physical_plan_fingerprint_sha256": plan_fingerprint,
        "qa_report": {"path": str(args.qa_report.resolve()), "sha256": sha_file(args.qa_report.resolve()), "status": qa.get("status")},
        "qa_execution_record": {"path": str(args.qa_execution_record.resolve()), "sha256": sha_file(args.qa_execution_record.resolve())},
        "coverage": {"npz_member_count": len(members), "total_member_bytes": sum(item["size_bytes"] for item in members)},
        "members": members,
    }
    payload["manifest_sha256"] = sha_bytes(canonical(payload))
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite archive member manifest: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "status": payload["status"], "member_count": len(members),
        "total_member_bytes": payload["coverage"]["total_member_bytes"],
        "manifest_path": str(out), "manifest_sha256": sha_file(out),
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
