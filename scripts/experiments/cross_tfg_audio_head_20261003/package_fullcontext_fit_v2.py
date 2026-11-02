"""Build a small, hash-checked transfer bundle for the v2 fit-only inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v2.py"
BUILDER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/build_fullcontext_fit_plan_v2.py"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload, sha256_bytes(raw)


def _entry(path: Path, archive_path: str, role: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"package input is not a file: {resolved}")
    return {
        "archive_path": archive_path,
        "source_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "role": role,
    }


def build_bundle(*, plan_path: Path, plan_receipt_path: Path, manifest_path: Path,
                 base_plan_path: Path, output_path: Path) -> dict[str, Any]:
    plan, plan_raw_sha = read_json(plan_path.resolve(strict=True))
    receipt, receipt_raw_sha = read_json(plan_receipt_path.resolve(strict=True))
    manifest, manifest_raw_sha = read_json(manifest_path.resolve(strict=True))
    base_plan, base_plan_raw_sha = read_json(base_plan_path.resolve(strict=True))

    embedded_plan = str(plan.get("plan_payload_sha256", "")).lower()
    if embedded_plan != canonical_sha({key: value for key, value in plan.items() if key != "plan_payload_sha256"}):
        raise ValueError("v2 plan embedded payload fingerprint mismatch")
    embedded_manifest = str(manifest.get("manifest_sha256", "")).lower()
    if embedded_manifest != canonical_sha({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ValueError("full-audio manifest embedded payload fingerprint mismatch")
    if plan.get("full_context_fit_v2", {}).get("source_plan_raw_sha256") != base_plan_raw_sha:
        raise ValueError("v2 plan does not bind the supplied historical base plan")
    if plan.get("full_context_fit_v2", {}).get("full_audio_manifest_raw_sha256") != manifest_raw_sha:
        raise ValueError("v2 plan does not bind the supplied full-audio transfer manifest")
    if receipt.get("plan_raw_sha256") != plan_raw_sha or receipt.get("plan_payload_sha256") != embedded_plan:
        raise ValueError("plan build receipt does not bind the supplied plan bytes")
    if receipt.get("full_audio_manifest_raw_sha256") != manifest_raw_sha:
        raise ValueError("plan build receipt does not bind the supplied input manifest bytes")
    if len(plan.get("rows", [])) != 8 or len(manifest.get("rows", [])) != 8:
        raise ValueError("fit-v2 transfer bundle must contain exactly eight historical fit rows")

    manifest_rows = {str(row.get("row_id", "")): row for row in manifest["rows"]}
    if len(manifest_rows) != 8:
        raise ValueError("full-audio manifest has missing or duplicate row identities")

    items: list[dict[str, Any]] = []
    plan_rows = {str(row.get("legacy_row_id", "")): row for row in plan["rows"]}
    if set(plan_rows) != set(manifest_rows):
        raise ValueError("full-audio row identities differ from the v2 fit plan")

    for legacy_id, row in plan_rows.items():
        transfer = manifest_rows[legacy_id]
        source = Path(str(transfer.get("transfer_local_path", ""))).resolve(strict=True)
        natural = row["natural_pcm"]
        expected_sha = str(natural.get("file_sha256", "")).lower()
        actual_sha = sha256_file(source)
        if actual_sha != expected_sha or actual_sha != str(transfer.get("file_sha256", "")).lower():
            raise ValueError(f"full-audio original WAV bytes changed: {legacy_id}")
        if source.stat().st_size <= 0:
            raise ValueError(f"full-audio original WAV is empty: {legacy_id}")
        archived_name = source.name
        if PurePosixPath(archived_name).name != archived_name:
            raise ValueError("unsafe full-audio basename")
        items.append(_entry(source, f"full_audio/{archived_name}", "original_full_audio_wav"))

    support_files = [
        (plan_path.resolve(strict=True), "fit/plan.json", "frozen_fit_plan"),
        (plan_receipt_path.resolve(strict=True), "fit/plan_build_receipt.json", "plan_build_receipt"),
        (manifest_path.resolve(strict=True), "fit/full_audio_transfer_manifest.json", "full_audio_transfer_manifest"),
        (base_plan_path.resolve(strict=True), "fit/historical_base_plan.json", "historical_fit_base_plan"),
        (BUILDER.resolve(strict=True), "source/build_fullcontext_fit_plan_v2.py", "plan_builder_source"),
        (RUNTIME.resolve(strict=True), "source/wav2lip_fullcontext_fit_runtime_v2.py", "fit_runtime_source"),
    ]
    for path, archive_path, role in support_files:
        items.append(_entry(path, archive_path, role))
    archive_names = [item["archive_path"] for item in items]
    if len(archive_names) != len(set(archive_names)):
        raise ValueError("duplicate package archive path")

    bundle_manifest = {
        "schema": "cross_tfg_audio_head_fullcontext_fit_bundle_v1",
        "status": "READY_FOR_REVIEW",
        "plan_path": str(plan_path.resolve()),
        "plan_raw_sha256": plan_raw_sha,
        "plan_payload_sha256": embedded_plan,
        "plan_build_receipt_raw_sha256": receipt_raw_sha,
        "full_audio_manifest_raw_sha256": manifest_raw_sha,
        "full_audio_manifest_payload_sha256": embedded_manifest,
        "historical_base_plan_raw_sha256": base_plan_raw_sha,
        "row_count": 8,
        "speaker_count": len({str(row.get("speaker_id")) for row in plan["rows"]}),
        "fit_seed": plan["fit_config"]["fit_seed"],
        "epochs": plan["fit_config"]["epochs"],
        "steps": plan["fit_config"]["steps_requested"],
        "confirmation_audio_opened": False,
        "development_audio_opened": False,
        "effects_or_scores_read": False,
        "items": items,
    }
    bundle_manifest["bundle_manifest_payload_sha256"] = canonical_sha(bundle_manifest)
    manifest_raw = json.dumps(bundle_manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"

    target = output_path.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite fit-v2 bundle: {target}")
    receipt_path = target.with_name("bundle_build_receipt.json")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite fit-v2 bundle receipt: {receipt_path}")
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(temp_fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("wb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w|") as tar:
                    for item in items:
                        source_path = Path(item["source_path"])
                        info = tarfile.TarInfo(item["archive_path"])
                        info.size = int(item["bytes"])
                        info.mode = 0o644
                        info.mtime = 0
                        with source_path.open("rb") as source_handle:
                            tar.addfile(info, source_handle)
                    info = tarfile.TarInfo("bundle_manifest.json")
                    info.size = len(manifest_raw)
                    info.mode = 0o644
                    info.mtime = 0
                    import io
                    tar.addfile(info, io.BytesIO(manifest_raw))
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    actual_bundle_sha = sha256_file(target)
    actual_bundle_bytes = target.stat().st_size
    with tarfile.open(target, mode="r:gz") as tar:
        names = tar.getnames()
        expected_names = archive_names + ["bundle_manifest.json"]
        if names != expected_names:
            raise RuntimeError("fit-v2 bundle archive member set/order differs from its manifest")
        for item in items:
            extracted = tar.extractfile(item["archive_path"])
            if extracted is None:
                raise RuntimeError(f"fit-v2 bundle member cannot be read: {item['archive_path']}")
            digest = hashlib.sha256()
            byte_count = 0
            for block in iter(lambda: extracted.read(1 << 20), b""):
                digest.update(block)
                byte_count += len(block)
            if byte_count != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"fit-v2 bundle member readback mismatch: {item['archive_path']}")
        manifest_member = tar.extractfile("bundle_manifest.json")
        if manifest_member is None or manifest_member.read() != manifest_raw:
            raise RuntimeError("fit-v2 bundle manifest readback mismatch")

    receipt = {
        "schema": "cross_tfg_audio_head_fullcontext_fit_bundle_receipt_v1",
        "status": "READY_FOR_REVIEW",
        "bundle_path": str(target),
        "bundle_bytes": actual_bundle_bytes,
        "bundle_sha256": actual_bundle_sha,
        "bundle_manifest_payload_sha256": bundle_manifest["bundle_manifest_payload_sha256"],
        "member_count": len(items) + 1,
        "member_bytes_uncompressed": sum(int(item["bytes"]) for item in items) + len(manifest_raw),
        "row_count": 8,
        "speaker_count": bundle_manifest["speaker_count"],
        "plan_raw_sha256": plan_raw_sha,
        "full_audio_manifest_raw_sha256": manifest_raw_sha,
        "confirmation_audio_opened": False,
        "development_audio_opened": False,
        "effects_or_scores_read": False,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-receipt", type=Path, required=True)
    parser.add_argument("--full-audio-manifest", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_bundle(
        plan_path=args.plan,
        plan_receipt_path=args.plan_receipt,
        manifest_path=args.full_audio_manifest,
        base_plan_path=args.base_plan,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
