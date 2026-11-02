"""Create the immutable attempt02 full-audio fit bundle and verify every member."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT = "fit_v2_fullcontext_attempt02"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, sha256_bytes(raw)


def _entry(path: Path, archive_path: str, role: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"bundle input is not a file: {path}")
    member = PurePosixPath(archive_path)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"unsafe archive path: {archive_path}")
    return {
        "archive_path": member.as_posix(),
        "source_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
    }


def build_bundle(*, plan_path: Path, plan_receipt_path: Path, transfer_manifest_path: Path,
                 base_plan_path: Path, parent_plan_path: Path, parent_transfer_manifest_path: Path,
                 output_path: Path) -> dict[str, Any]:
    plan, plan_raw_sha = read_json(plan_path.resolve(strict=True))
    plan_receipt, plan_receipt_sha = read_json(plan_receipt_path.resolve(strict=True))
    transfer, transfer_raw_sha = read_json(transfer_manifest_path.resolve(strict=True))
    base_plan, base_raw_sha = read_json(base_plan_path.resolve(strict=True))
    parent_plan, parent_plan_sha = read_json(parent_plan_path.resolve(strict=True))
    parent_transfer, parent_transfer_sha = read_json(parent_transfer_manifest_path.resolve(strict=True))

    if plan.get("schema") != "cross_tfg_audio_head_historical_fullcontext_fit_v2" or plan.get("status") != "FIT_ONLY_FROZEN":
        raise ValueError("wrong attempt02 fit-plan schema/status")
    if plan.get("plan_payload_sha256") != canonical_sha({k: v for k, v in plan.items() if k != "plan_payload_sha256"}):
        raise ValueError("attempt02 plan canonical payload SHA mismatch")
    if transfer.get("manifest_sha256") != canonical_sha({k: v for k, v in transfer.items() if k != "manifest_sha256"}):
        raise ValueError("attempt02 transfer manifest canonical payload SHA mismatch")
    if plan.get("full_context_fit_v2", {}).get("source_plan_raw_sha256") != base_raw_sha:
        raise ValueError("plan does not bind the exact historical fit plan")
    if plan.get("full_context_fit_v2", {}).get("full_audio_manifest_raw_sha256") != transfer_raw_sha:
        raise ValueError("plan does not bind the exact attempt02 full-audio manifest")
    if plan_receipt.get("plan_raw_sha256") != plan_raw_sha or plan_receipt.get("plan_payload_sha256") != plan["plan_payload_sha256"]:
        raise ValueError("plan build receipt does not bind the plan bytes")
    if plan_receipt.get("full_audio_manifest_raw_sha256") != transfer_raw_sha:
        raise ValueError("plan build receipt does not bind the attempt02 transfer manifest")
    if plan.get("source_revision", {}).get("current_git_head") != plan.get("git_commit"):
        raise ValueError("plan current Git commit/source revision mismatch")
    if len(plan.get("rows", [])) != 8 or len(transfer.get("rows", [])) != 8:
        raise ValueError("attempt02 package must contain exactly eight fit audio rows")
    if plan.get("runtime", {}).get("adapter_factory") != "scripts.audio_enhancement.wav2lip_fullcontext_fit_runtime_v2_attempt02:create_runtime":
        raise ValueError("plan does not bind the attempt02 runtime factory")
    if plan.get("runtime", {}).get("execution", {}).get("python_executable_literal") != "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python":
        raise ValueError("plan fit interpreter path differs from the actual qualified Wav2Lip environment")

    transfer_rows = {str(row.get("row_id", "")): row for row in transfer.get("rows", [])}
    plan_rows = {str(row.get("legacy_row_id", "")): row for row in plan["rows"]}
    if len(transfer_rows) != 8 or len(plan_rows) != 8 or set(transfer_rows) != set(plan_rows):
        raise ValueError("plan and transfer manifest fit-row identities disagree")

    items: list[dict[str, Any]] = []
    for row_id, transfer_row in transfer_rows.items():
        local_wav = Path(str(transfer_row.get("transfer_local_path", ""))).resolve(strict=True)
        natural = plan_rows[row_id]["natural_pcm"]
        if sha256_file(local_wav) != natural.get("file_sha256") or natural.get("file_sha256") != transfer_row.get("file_sha256"):
            raise ValueError(f"fit audio raw SHA mismatch: {row_id}")
        if local_wav.stat().st_size <= 0:
            raise ValueError(f"fit audio is empty: {row_id}")
        items.append(_entry(local_wav, f"input/full_audio/{local_wav.name}", "original_full_audio_wav"))

    fixed_sources = {str(item["path"]): item for item in plan["source_revision"]["fit_package_source_files"]}
    if len(fixed_sources) != len(plan["source_revision"]["fit_package_source_files"]):
        raise ValueError("plan contains duplicate source-closure paths")
    for relative, binding in fixed_sources.items():
        source = (ROOT / relative).resolve(strict=True)
        if source.stat().st_size != int(binding["bytes"]) or sha256_file(source) != binding["sha256"]:
            raise ValueError(f"fit runtime source changed since plan freeze: {relative}")
        items.append(_entry(source, f"src/{relative}", "fit_runtime_source"))

    support = [
        (plan_path, "fit/plan.json", "frozen_fit_plan"),
        (plan_receipt_path, "fit/plan_build_receipt.json", "plan_build_receipt"),
        (transfer_manifest_path, "fit/full_audio_transfer_manifest.json", "full_audio_transfer_manifest"),
        (base_plan_path, "fit/historical_base_plan.json", "historical_fit_base_plan"),
        (parent_plan_path, "lineage/attempt01_plan.json", "superseded_attempt01_plan_provenance"),
        (parent_transfer_manifest_path, "lineage/attempt01_full_audio_manifest.json", "attempt01_transfer_provenance"),
    ]
    for path, archive_path, role in support:
        items.append(_entry(Path(path), archive_path, role))

    archive_paths = [item["archive_path"] for item in items]
    if len(archive_paths) != len(set(archive_paths)):
        raise ValueError("attempt02 bundle member paths are duplicated")
    manifest: dict[str, Any] = {
        "schema": "cross_tfg_audio_head_fullcontext_fit_bundle_v2",
        "attempt": ATTEMPT,
        "status": "READY_FOR_ROOT_REVIEW",
        "plan_path": str(plan_path.resolve()),
        "plan_bytes": plan_path.stat().st_size,
        "plan_raw_sha256": plan_raw_sha,
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "plan_receipt_raw_sha256": plan_receipt_sha,
        "transfer_manifest_raw_sha256": transfer_raw_sha,
        "transfer_manifest_payload_sha256": transfer["manifest_sha256"],
        "historical_base_plan_raw_sha256": base_raw_sha,
        "parent_attempt01_plan_raw_sha256": parent_plan_sha,
        "parent_attempt01_transfer_manifest_raw_sha256": parent_transfer_sha,
        "fit_row_count": 8,
        "fit_speaker_ids": sorted({str(row["speaker_id"]) for row in plan["rows"]}),
        "optimizer_steps": int(plan["fit_config"]["steps_requested"]),
        "epochs": int(plan["fit_config"]["epochs"]),
        "forward_performed_during_packaging": False,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "effects_or_scores_read": False,
        "items": items,
    }
    manifest["manifest_payload_sha256"] = canonical_sha(manifest)
    manifest_raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"

    output_path = output_path.resolve()
    receipt_path = output_path.with_name("bundle_build_receipt.json")
    manifest_path = output_path.with_name("bundle_manifest.json")
    for target in (output_path, receipt_path, manifest_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite attempt02 bundle output: {target}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with temp.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w|") as tar:
                    for item in items:
                        member = tarfile.TarInfo(item["archive_path"])
                        member.size = int(item["bytes"])
                        member.mode = 0o644
                        member.mtime = 0
                        with Path(item["source_path"]).open("rb") as fh:
                            tar.addfile(member, fh)
                    member = tarfile.TarInfo("bundle_manifest.json")
                    member.size = len(manifest_raw)
                    member.mode = 0o644
                    member.mtime = 0
                    tar.addfile(member, io.BytesIO(manifest_raw))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temp, output_path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    with tarfile.open(output_path, mode="r:gz") as tar:
        expected_names = archive_paths + ["bundle_manifest.json"]
        if tar.getnames() != expected_names:
            raise RuntimeError("attempt02 bundle archive inventory/order mismatch")
        for item in items:
            stream = tar.extractfile(item["archive_path"])
            if stream is None:
                raise RuntimeError(f"bundle member could not be read: {item['archive_path']}")
            digest, count = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
                count += len(block)
            if count != int(item["bytes"]) or digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"bundle member readback mismatch: {item['archive_path']}")
        stream = tar.extractfile("bundle_manifest.json")
        if stream is None or stream.read() != manifest_raw:
            raise RuntimeError("bundle manifest readback mismatch")

    manifest_path.write_bytes(manifest_raw)
    receipt = {
        "schema": "cross_tfg_audio_head_fullcontext_fit_bundle_receipt_v2",
        "status": "READY_FOR_ROOT_REVIEW",
        "bundle_path": str(output_path),
        "bundle_bytes": output_path.stat().st_size,
        "bundle_sha256": sha256_file(output_path),
        "bundle_manifest_path": str(manifest_path),
        "bundle_manifest_bytes": len(manifest_raw),
        "bundle_manifest_raw_sha256": sha256_bytes(manifest_raw),
        "bundle_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "member_count": len(items) + 1,
        "uncompressed_member_bytes": sum(int(item["bytes"]) for item in items) + len(manifest_raw),
        "plan_raw_sha256": plan_raw_sha,
        "transfer_manifest_raw_sha256": transfer_raw_sha,
        "fit_row_count": 8,
        "forward_performed_during_packaging": False,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "effects_or_scores_read": False,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    root = ROOT / "runs/cross_tfg_audio_head_20261003"
    attempt02 = root / ATTEMPT
    result = build_bundle(
        plan_path=attempt02 / "plan.json",
        plan_receipt_path=attempt02 / "fullcontext_fit_plan_build_receipt.json",
        transfer_manifest_path=attempt02 / "full_audio_transfer_manifest.json",
        base_plan_path=root / "fit_subset_attempt02/historical_fit_subset_plan.json",
        parent_plan_path=root / "fit_v2_fullcontext_attempt01/plan.json",
        parent_transfer_manifest_path=root / "fit_v2_full_audio_attempt01/full_audio_manifest.json",
        output_path=attempt02 / "staging_attempt02/fullcontext_fit_v2_attempt02.tar.gz",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
