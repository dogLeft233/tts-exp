#!/usr/bin/env python3
"""Audit nullable word_index values in the sealed WORLD shape run.

This is a metadata-only audit. It does not open score NPZs, compute score
statistics, or read evaluation_results.jsonl.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_RUN_STATUS_SHA256 = "3908c4e417edb8ac68d879b0ceef4c82d88a44ea3c7d55c2f4466b2516510deb"
EXPECTED_ALL_FILES_SHA256 = "8076463991b66b1d7fbeddc322b3bb5e38fd072a00b121178663b63eb8ebc0ce"
EXPECTED_PROTOCOL_SHA256 = "905d8839d789a36e82466f293d823bbdf058a8c1bcdfa150a535eeac47e53b16"
EXPECTED_ROWS_SHA256 = "23a2a0f50109f936ca2d7559fc0aa6f17a388312e46bba109267e3b30ef45796"
EXPECTED_READBACK_RECEIPT_SHA256 = "4f829985e70c276bbdedf88fa0bb1922b8d2c87a95efcffcbee156a402781963"
EXPECTED_READBACK_SCHEMA = "WORLD_SHAPE_GPU_ATTEMPT14_FULL_READBACK_V1"
EXPECTED_CONSUMER_SHA256 = "0aea5d17558c3985453119525890529f5a24addac41539b07d6b49a52d355470"
EXPECTED_CONTEXTS = 160
EXPECTED_RECIPIENT_TOKENS = 5803


class AuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def frozen_phone_labels(consumer_source: Path) -> tuple[str, ...]:
    tree = ast.parse(consumer_source.read_text(encoding="utf-8"), filename=str(consumer_source))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PHONE_LABELS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            require(isinstance(value, tuple) and all(isinstance(x, str) for x in value),
                    "consumer PHONE_LABELS is not a literal tuple of strings")
            return value
    raise AuditError("consumer source has no literal PHONE_LABELS assignment")


def matched_indices(phone_record: dict[str, Any], role: str) -> set[int]:
    key = "recipient_phone_index" if role == "recipient" else "donor_phone_index"
    rows = phone_record.get("matched_phones")
    require(isinstance(rows, list), "phone_map_record.matched_phones is not a list")
    indices: set[int] = set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get(key), int),
                f"matched phone row lacks integer {key}")
        index = int(row[key])
        require(index not in indices, f"duplicate matched {role} phone index {index}")
        indices.add(index)
    return indices


def audit_tier(
    *,
    context: dict[str, Any],
    phone_record: dict[str, Any],
    role: str,
    labels: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_key = f"{role}_phones"
    rows = phone_record.get(rows_key)
    require(isinstance(rows, list), f"phone_map_record.{rows_key} is not a list")
    matched = matched_indices(phone_record, role)
    seen: set[int] = set()
    counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict), f"{rows_key} contains a non-object row")
        index = row.get("index")
        require(type(index) is int, f"{rows_key} has a non-integer phone index")
        require(index not in seen, f"{rows_key} repeats phone index {index}")
        seen.add(index)
        word_index = row.get("word_index")
        if word_index is None:
            counts["word_index_null"] += 1
        elif type(word_index) is int:
            counts["word_index_integer"] += 1
        else:
            counts["word_index_other"] += 1
        is_matched = index in matched
        label = row.get("normalized_label", "")
        raw_label = row.get("raw_label", "")
        require(isinstance(label, str) and isinstance(raw_label, str),
                f"{rows_key}[{index}] labels are not strings")
        if is_matched:
            counts["matched"] += 1
        else:
            counts["unmatched"] += 1
        if label in labels:
            counts["selected_label"] += 1
            counts["selected_label_matched" if is_matched else "selected_label_unmatched"] += 1
        if word_index is None:
            counts["null_blank_label"] += int(label == "" and raw_label == "")
            counts["null_unmatched"] += int(not is_matched)
            counts["null_selected_label"] += int(label in labels)
            counts["null_selected_matched"] += int(label in labels and is_matched)
        details.append({
            "index": index,
            "normalized_label": label,
            "raw_label": raw_label,
            "interval_s": row.get("interval_s"),
            "word_index": word_index,
            "matched": is_matched,
            "in_frozen_inventory": row.get("in_frozen_inventory"),
        })
    require(matched.issubset(seen), f"matched {role} indices are absent from the full tier")
    return dict(counts), details


def verify(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    readback_path = args.readback_receipt.resolve(strict=True)
    consumer_path = args.consumer_source.resolve(strict=True)
    readback_sha = sha256_file(readback_path)
    require(readback_sha == EXPECTED_READBACK_RECEIPT_SHA256,
            "full run readback receipt SHA differs from the pinned receipt")
    readback = read_json(readback_path)
    require(readback.get("schema") == EXPECTED_READBACK_SCHEMA, "readback receipt schema mismatch")
    require(readback.get("status") == "FULL_SEAL_AND_ROW_COVERAGE_PASS", "readback receipt is not a full PASS")
    require(readback.get("verified_file_count") == 4327, "readback file count mismatch")
    require(readback.get("verified_payload_bytes") == 1376583771, "readback payload byte count mismatch")

    all_files_path = run_root / "all_files_sha256.json"
    protocol_path = run_root / "evaluation_protocol.json"
    rows_path = run_root / "evaluation_rows.jsonl"
    status_path = run_root / "run_status.json"
    expected_raw = {
        "all_files_sha256.json": EXPECTED_ALL_FILES_SHA256,
        "evaluation_protocol.json": EXPECTED_PROTOCOL_SHA256,
        "evaluation_rows.jsonl": EXPECTED_ROWS_SHA256,
        "run_status.json": EXPECTED_RUN_STATUS_SHA256,
    }
    paths = {
        "all_files_sha256.json": all_files_path,
        "evaluation_protocol.json": protocol_path,
        "evaluation_rows.jsonl": rows_path,
        "run_status.json": status_path,
    }
    file_bindings: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        observed = sha256_file(path)
        require(observed == expected_raw[name], f"run input seal mismatch: {name}")
        file_bindings[name] = {"path": str(path), "raw_sha256": observed, "size_bytes": path.stat().st_size}
    require(readback.get("all_files_sha256_json_sha256") == file_bindings["all_files_sha256.json"]["raw_sha256"],
            "readback receipt does not bind the local all-files manifest")
    require(readback.get("execution_plan_sha256") == "b7c6da80a380a387ef6c1ef3b7d62d1181d419ddf58d83f214891ac9827b3c85",
            "readback receipt execution plan binding mismatch")

    status = read_json(status_path)
    require(status.get("status") == "complete", "run status is not complete")
    require(status.get("planned_contexts") == EXPECTED_CONTEXTS and status.get("completed_contexts") == EXPECTED_CONTEXTS,
            "run context coverage mismatch")
    require(status.get("planned_rows") == 1600 and status.get("scored_rows") == 1600 and status.get("failed_rows") == 0,
            "run row coverage mismatch")
    require(status.get("failed_contexts") == 0, "run has failed contexts")

    inventory = read_json(all_files_path)
    require(isinstance(inventory, dict), "all-files manifest is not an object")
    context_paths = sorted(
        rel for rel in inventory
        if rel.startswith("contexts/") and rel.endswith(".json") and not rel.endswith(".npz.json")
    )
    require(len(context_paths) == EXPECTED_CONTEXTS, f"expected {EXPECTED_CONTEXTS} context JSONs, found {len(context_paths)}")
    labels_tuple = frozen_phone_labels(consumer_path)
    labels = set(labels_tuple)
    consumer_sha = sha256_file(consumer_path)
    require(consumer_sha == EXPECTED_CONSUMER_SHA256, "consumer17 source SHA differs from the audited source")

    total = {"recipient": Counter(), "donor": Counter()}
    per_context: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    observed_contexts: set[tuple[str, str]] = set()
    map_paths: set[str] = set()
    map_shas: set[str] = set()
    context_hash_bindings: list[dict[str, Any]] = []

    for rel in context_paths:
        context_path = run_root / Path(rel.replace("/", "\\"))
        context_sha = sha256_file(context_path)
        require(context_sha == inventory[rel], f"context sidecar SHA differs from all-files manifest: {rel}")
        context = read_json(context_path)
        sample_id = context.get("sample_id")
        direction = context.get("direction")
        key = context.get("key")
        phone_ref = context.get("phone_map")
        phone_record = context.get("phone_map_record")
        require(isinstance(sample_id, str) and direction in {"natural_to_tts", "tts_to_natural"},
                f"malformed context identity in {rel}")
        require(key == f"{sample_id}__{direction}", f"context key differs from its identity in {rel}")
        context_identity = (sample_id, direction)
        require(context_identity not in observed_contexts,
                f"duplicate context identity {sample_id}/{direction}")
        observed_contexts.add(context_identity)
        require(isinstance(phone_ref, dict) and isinstance(phone_ref.get("path"), str),
                f"phone-map source reference missing in {rel}")
        map_sha = phone_ref.get("file_sha256")
        require(isinstance(map_sha, str) and len(map_sha) == 64 and all(c in "0123456789abcdef" for c in map_sha),
                f"invalid phone-map raw SHA reference in {rel}")
        map_path = phone_ref["path"]
        map_paths.add(map_path)
        map_shas.add(map_sha)
        require(isinstance(phone_record, dict), f"embedded phone_map_record missing in {rel}")
        require(phone_record.get("id") == sample_id, f"embedded phone map id differs in {rel}")
        require(phone_record.get("speaker") == context.get("speaker_id"), f"embedded phone map speaker differs in {rel}")
        require(phone_record.get("split") == "evaluation", f"phone map is not an evaluation record in {rel}")

        context_counts: dict[str, dict[str, Any]] = {}
        tiers: dict[str, list[dict[str, Any]]] = {}
        for role in ("recipient", "donor"):
            counts, details = audit_tier(context=context, phone_record=phone_record, role=role, labels=labels)
            context_counts[role] = counts
            tiers[role] = details
            total[role].update(counts)
        recipient_nulls = [row for row in tiers["recipient"] if row["word_index"] is None]
        donor_nulls = [row for row in tiers["donor"] if row["word_index"] is None]
        for row in recipient_nulls:
            require(row["normalized_label"] == "" and row["raw_label"] == "" and not row["matched"],
                    f"null recipient word_index is not a blank unmatched row: {sample_id}/{direction}/{row['index']}")
        primary = context_counts["recipient"]
        per_context.append({
            "sample_id": sample_id,
            "speaker_id": context.get("speaker_id"),
            "direction": direction,
            "context_sidecar": {"path": rel, "raw_sha256": context_sha, "size_bytes": context_path.stat().st_size},
            "phone_map_source_ref": {"path": map_path, "raw_sha256": map_sha,
                                     "source_bytes_independently_present": False},
            "recipient": {"counts": context_counts["recipient"], "tokens": tiers["recipient"]},
            "donor": {"counts": context_counts["donor"], "tokens": tiers["donor"]},
        })
        for role, rows in (("recipient", recipient_nulls), ("donor", donor_nulls)):
            for row in rows:
                null_rows.append({
                    "sample_id": sample_id, "direction": direction, "role": role,
                    "phone_map_raw_sha256": map_sha, "context_sidecar_sha256": context_sha,
                    **row,
                })
        context_hash_bindings.append({"path": rel, "sha256": context_sha})

    require(len(observed_contexts) == EXPECTED_CONTEXTS, "duplicate or missing context identities")
    require(len(map_paths) <= EXPECTED_CONTEXTS and len(map_shas) <= EXPECTED_CONTEXTS,
            "phone-map reference/hash cardinality exceeds context count")
    require(total["recipient"]["word_index_integer"] + total["recipient"]["word_index_null"] == EXPECTED_RECIPIENT_TOKENS,
            "recipient token total differs from audited expected roster")
    require(total["recipient"]["word_index_null"] == 432, "recipient null word_index count mismatch")
    require(total["recipient"]["word_index_other"] == 0, "recipient word_index has non-integer/non-null values")
    require(total["recipient"]["null_blank_label"] == 432 and total["recipient"]["null_unmatched"] == 432,
            "recipient null rows are not all blank and unmatched")
    require(total["recipient"]["null_selected_label"] == 0 and total["recipient"]["null_selected_matched"] == 0,
            "recipient null rows include a frozen selected label")
    require(total["recipient"]["matched"] == 5344 and total["recipient"]["unmatched"] == 459,
            "recipient match classification differs from the sealed audit totals")
    require(total["donor"]["word_index_other"] == 0, "donor word_index has non-integer/non-null values")

    payload = {
        "schema": "WORLD_SHAPE_WORD_INDEX_NULL_AUDIT_V1",
        "status": "PASS",
        "audit_scope": "160 sealed evaluation context sidecars; embedded recipient/donor phone tiers; word_index type and null-row classification only",
        "effects_or_scores_read": False,
        "npz_arrays_read": False,
        "evaluation_results_jsonl_read": False,
        "source": {"path": str(Path(__file__).resolve()), "raw_sha256": sha256_file(Path(__file__).resolve())},
        "consumer_source": {"path": str(consumer_path), "raw_sha256": consumer_sha},
        "readback_receipt": {"path": str(readback_path), "raw_sha256": readback_sha,
                             "schema": readback["schema"], "status": readback["status"],
                             "verified_file_count": readback["verified_file_count"],
                             "verified_payload_bytes": readback["verified_payload_bytes"]},
        "run_bindings": file_bindings,
        "run_status": {key: status.get(key) for key in (
            "status", "planned_contexts", "completed_contexts", "planned_rows", "scored_rows",
            "failed_contexts", "failed_rows", "row_manifest_sha256", "protocol_sha256"
        )},
        "all_files_manifest": {"path": str(all_files_path), "raw_sha256": file_bindings["all_files_sha256.json"]["raw_sha256"],
                               "member_count": len(inventory), "context_json_count": len(context_paths)},
        "frozen_phone_labels": list(labels_tuple),
        "summary": {
            "contexts": len(per_context),
            "unique_phone_map_paths": len(map_paths),
            "unique_phone_map_raw_sha256_references": len(map_shas),
            "map_raw_bytes_available_in_restored_run": False,
            "recipient": dict(total["recipient"]),
            "donor": dict(total["donor"]),
            "recipient_null_word_index_rows_all_blank_unmatched": total["recipient"]["word_index_null"] == 432
                and total["recipient"]["null_blank_label"] == 432
                and total["recipient"]["null_unmatched"] == 432,
            "recipient_null_rows_matching_frozen_selected_label": total["recipient"]["null_selected_matched"],
            "direction_null_row_counts": dict(Counter(
                row["direction"] for row in null_rows if row["role"] == "recipient"
            )),
        },
        "contexts": per_context,
        "null_word_index_rows": null_rows,
        "limitations": [
            "The original phone_map.json source bytes are not members of the restored model run; their raw SHA values are preserved from the verified context sidecars, but those external source files were not independently rehashed locally.",
            "This metadata audit does not rerun the score or local-response consumer.",
        ],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--readback-receipt", type=Path, required=True)
    parser.add_argument("--consumer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "receipt_sha256": report["receipt_sha256"],
                          "contexts": report["summary"]["contexts"],
                          "recipient_null_rows": report["summary"]["recipient"]["word_index_null"]},
                         sort_keys=True))
        return 0
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
