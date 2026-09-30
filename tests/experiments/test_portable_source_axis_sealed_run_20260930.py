from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from scripts.experiments.portable_source_axis_20260930.independent_sealed_run import verify_core_transfer


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_core_transfer_verifies_archives_members_parent_and_split_results(tmp_path: Path) -> None:
    run = tmp_path / "run"
    package = tmp_path / "package"
    run.mkdir()
    package.mkdir()
    artifacts = {
        "evaluation_protocol.json": b'{"protocol":1}\n',
        "evaluation_rows.jsonl": b'{"row":1}\n',
        "continuation_parent.json": b'{"parent":"sealed"}\n',
        "recovery_artifact_hashes.jsonl": b'{"artifact":"sealed"}\n',
        "recovery_detached_metadata_manifest.jsonl": b'{"detached":true}\n',
        "recovery_code_seal.json": b'{"code":"sealed"}\n',
        "evaluation_clip_status.jsonl": b'{"sample":"s1","status":"complete"}\n',
        "evaluation_attempts.jsonl": b'{"event":"attempt"}\n',
        "payload_parts/evaluation_results.jsonl.part0001": b'{"row":',
        "payload_parts/evaluation_results.jsonl.part0002": b'1}\n',
    }
    for name in (
        "evaluation_protocol.json", "evaluation_rows.jsonl", "continuation_parent.json",
        "recovery_artifact_hashes.jsonl", "recovery_detached_metadata_manifest.jsonl",
        "recovery_code_seal.json",
        "evaluation_clip_status.jsonl", "evaluation_attempts.jsonl",
    ):
        (run / name).write_bytes(artifacts[name])
    complete_results = artifacts["payload_parts/evaluation_results.jsonl.part0001"] + artifacts[
        "payload_parts/evaluation_results.jsonl.part0002"
    ]
    (run / "evaluation_results.jsonl").write_bytes(complete_results)

    archive_path = package / "core_0001.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for name, payload in artifacts.items():
            source = tmp_path / name.replace("/", "_")
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)
            tar.add(source, arcname=name)

    item_rows = []
    for name, payload in artifacts.items():
        category = "recovery_code" if name == "recovery_code_seal.json" else "sealed_run_metadata"
        item_rows.append({
            "archive_name": archive_path.name,
            "archive_path": name,
            "category": category,
            "size_bytes": len(payload),
            "sha256": _sha(payload),
        })
    protocol_sha = _sha(artifacts["evaluation_protocol.json"])
    row_sha = _sha(artifacts["evaluation_rows.jsonl"])
    result_sha = _sha(complete_results)
    manifest = {
        "package_type": "analysis_core",
        "protocol_sha256": protocol_sha,
        "row_manifest_sha256": row_sha,
        "planned_row_count": 2,
        "unique_context_count": 1,
        "unique_visual_feature_count": 1,
        "unique_score_file_count": 2,
        "contains_effect_summary": False,
        "face_row_counts": {"primary": 1, "secondary": 1},
        "items": item_rows,
        "archives": [{
            "name": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "sha256": _sha(archive_path.read_bytes()),
        }],
        "split_files": {"evaluation_results.jsonl": {
            "original_size_bytes": len(complete_results),
            "original_sha256": result_sha,
            "parts": [
                {
                    "archive_path": name,
                    "size_bytes": len(artifacts[name]),
                    "sha256": _sha(artifacts[name]),
                }
                for name in (
                    "payload_parts/evaluation_results.jsonl.part0001",
                    "payload_parts/evaluation_results.jsonl.part0002",
                )
            ],
        }},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inherited_summary = tmp_path / "inherited.json"
    inherited_summary.write_text('{"verified":true}\n', encoding="utf-8")
    expected_inventory = {
        "items": 10, "archives": 1, "planned_row_count": 2,
        "unique_context_count": 1, "unique_visual_feature_count": 1,
        "unique_score_file_count": 2, "total_archive_bytes": archive_path.stat().st_size,
        "recovery_code_items": 1, "recovery_metadata_items": 9,
    }

    verified = verify_core_transfer(
        run_dir=run,
        manifest_path=manifest_path,
        transfer_dir=package,
        inherited_summary_path=inherited_summary,
        expected_manifest_sha256=_sha(manifest_path.read_bytes()),
        expected_inherited_summary_sha256=_sha(inherited_summary.read_bytes()),
        expected_protocol_sha256=protocol_sha,
        expected_row_manifest_sha256=row_sha,
        expected_recovery_code_seal_sha256=_sha(artifacts["recovery_code_seal.json"]),
        expected_face_row_counts={"primary": 1, "secondary": 1},
        expected_inventory=expected_inventory,
    )

    assert verified["status"] == "verified"
    assert verified["archive_count"] == 1
    assert verified["member_count"] == 10
    assert verified["input_result_log_sha256"] == result_sha
