from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.asr_sync_error_correlation.io import canonical_hash, file_sha256
from scripts.experiments.asr_sync_error_correlation.word_errors import align_words, normalized_words

from .config import (
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    ASR_SNAPSHOT_HASH,
    ASR_TOKENIZER_VOCABULARY_HASH,
    CONDITIONS,
    PARENT_ASR_ASSETS,
    PARENT_ASR_CONFIG,
    PARENT_ASR_DECISION,
    PARENT_ASR_MANIFEST,
    PARENT_ASR_MANIFEST_DECISION,
    PARENT_ASR_MODEL_LOCK,
    PARENT_ASR_RECORDS,
    PARENT_MANIFEST,
    PARENT_TEST_LOCK,
    SEED,
    SYNCNET_CHECKPOINT_SHA256,
)


FORBIDDEN_SELECTION_TERMS = frozenset({"syncnet", "sync_c", "sync_d", "visual", "plot", "local_c", "confidence"})


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _repo_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _file_hash(root: Path, value: str | Path, expected: str, label: str) -> str:
    path = _resolve(root, value)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if actual != str(expected):
        raise ValueError(f"{label} hash mismatch: {path}")
    return actual


def _require(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} missing {key}")
    return mapping[key]


def assert_score_blind_fields(fields: Sequence[str]) -> None:
    lowered = {str(field).lower() for field in fields}
    leaked = sorted(field for field in lowered if any(term in field for term in FORBIDDEN_SELECTION_TERMS))
    if leaked:
        raise ValueError(f"score-bearing selection fields are forbidden: {leaked}")


def _validate_parent_lock(root: Path, parent: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    if parent.get("status") != "complete" or parent.get("decision") != "GO":
        raise ValueError("parent cohort lock is not complete GO")
    cohort = parent.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("parent cohort section is missing")
    records = cohort.get("records")
    if not isinstance(records, list) or len(records) != 24:
        raise ValueError("parent cohort is not exactly 24 records")
    if cohort.get("record_count") != 24 or cohort.get("source_group_count") != 24:
        raise ValueError("parent cohort counts are not 24")
    if cohort.get("selection") != "existing ordered parent fresh_confirmation records":
        raise ValueError("parent cohort selection is not frozen")
    media_access = parent.get("media_access", {})
    if not isinstance(media_access, Mapping) or any(media_access.get(key) is not False for key in ("fit_media_opened", "internal_dev_media_opened", "validation_media_opened", "test_media_opened", "mfa_run", "features_created", "scores_created")):
        raise ValueError("parent lock does not prove sealed splits were unvisited")
    if lock.get("status") != "sealed_unvisited":
        raise ValueError("parent test lock is not sealed_unvisited")
    if lock.get("media_opened") is not False or lock.get("derived_features_created") is not False or lock.get("scores_created") is not False:
        raise ValueError("parent test lock records sealed access")
    if lock.get("test_group_count") != 7 or len(lock.get("test_groups", [])) != 7:
        raise ValueError("parent test lock has unexpected sealed groups")
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("parent cohort record is not an object")
        sample_id = str(_require(row, "sample_id", "parent record"))
        group = str(_require(row, "source_group", "parent record"))
        if sample_id in seen_ids or group in seen_groups:
            raise ValueError("parent cohort sample_id/source_group is not unique")
        seen_ids.add(sample_id)
        seen_groups.add(group)
        if row.get("protocol_split") != "fit":
            raise ValueError(f"cohort record is not fit-only: {sample_id}")
        for key in ("face", "natural_audio", "tts_audio", "face_sha256", "natural_audio_sha256", "tts_audio_sha256"):
            if not row.get(key):
                raise ValueError(f"parent cohort missing {key}: {sample_id}")
        if group in set(map(str, lock.get("test_groups", []))) or sample_id in set(map(str, lock.get("test_sample_ids", []))):
            raise ValueError(f"fit cohort intersects sealed split: {sample_id}")


def _validate_parent_files(root: Path, parent: Mapping[str, Any]) -> dict[str, str]:
    parent_files = parent.get("parents", {}).get("parent_files", {})
    paths = {
        "source_manifest": parent_files.get("source_manifest", {}),
        "tts_meta": parent_files.get("tts_meta", {}),
    }
    hashes: dict[str, str] = {}
    for name, row in paths.items():
        if not isinstance(row, Mapping) or not row.get("path") or not row.get("sha256"):
            raise ValueError(f"parent lock has no {name} hash")
        hashes[name] = _file_hash(root, str(row["path"]), str(row["sha256"]), name)
    return hashes


def _validate_asr_record(root: Path, record: Mapping[str, Any], expected: Mapping[str, Any], arm: str) -> dict[str, Any]:
    sample_id = str(expected["sample_id"])
    if record.get("schema_version") != 1 or record.get("sample_id") != sample_id:
        raise ValueError(f"ASR record identity mismatch: {sample_id}/{arm}")
    if record.get("arm") != arm or record.get("source_group") != expected["source_group"]:
        raise ValueError(f"ASR record arm/source mismatch: {sample_id}/{arm}")
    if record.get("audio_path") != expected["audio_path"] or record.get("audio_sha256") != expected["audio_sha256"]:
        raise ValueError(f"ASR record audio binding mismatch: {sample_id}/{arm}")
    model = record.get("model")
    if not isinstance(model, Mapping) or model.get("model_id") != ASR_MODEL_ID or model.get("revision") != ASR_MODEL_REVISION or model.get("snapshot_hash") != ASR_SNAPSHOT_HASH or model.get("tokenizer_vocabulary_hash") != ASR_TOKENIZER_VOCABULARY_HASH:
        raise ValueError(f"ASR model lock mismatch: {sample_id}/{arm}")
    if record.get("greedy", {}).get("transcript") is None or not isinstance(record.get("greedy", {}).get("words"), list):
        raise ValueError(f"missing greedy CTC result: {sample_id}/{arm}")
    reference = record.get("reference_alignment")
    if not isinstance(reference, Mapping) or not isinstance(reference.get("words"), list):
        raise ValueError(f"missing forced reference alignment: {sample_id}/{arm}")
    emission = record.get("emissions")
    if not isinstance(emission, Mapping) or not emission.get("path") or not emission.get("sha256"):
        raise ValueError(f"missing emission binding: {sample_id}/{arm}")
    _file_hash(root, str(emission["path"]), str(emission["sha256"]), f"ASR emission {sample_id}/{arm}")
    return dict(record)


def load_frozen_inputs(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    asr_manifest_path = root / PARENT_ASR_MANIFEST
    asr_manifest = _load(asr_manifest_path)
    if asr_manifest.get("manifest_type") != "lrs3_asr_sync_error_correlation" or asr_manifest.get("sealed_splits_accessed") is not False:
        raise ValueError("ASR parent manifest is not the sealed-safe fit manifest")
    if asr_manifest.get("sample_count") != 24 or asr_manifest.get("source_group_count") != 24 or asr_manifest.get("arm_count") != 48:
        raise ValueError("ASR parent manifest counts are not 24 samples/48 arms")
    parent_path = root / PARENT_MANIFEST
    parent_lock_path = root / PARENT_TEST_LOCK
    parent = _load(parent_path)
    parent_lock = _load(parent_lock_path)
    _validate_parent_lock(root, parent, parent_lock)
    parent_file_hashes = _validate_parent_files(root, parent)

    config_path = root / PARENT_ASR_CONFIG
    model_lock_path = root / PARENT_ASR_MODEL_LOCK
    assets_path = root / PARENT_ASR_ASSETS
    manifest_decision_path = root / PARENT_ASR_MANIFEST_DECISION
    asr_decision_path = root / PARENT_ASR_DECISION
    for path in (config_path, model_lock_path, assets_path, manifest_decision_path, asr_decision_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen ASR artifact: {path}")
    asr_config = _load(config_path)
    model_lock = _load(model_lock_path)
    assets = _load(assets_path)
    if asr_config.get("asr", {}).get("model_id") != ASR_MODEL_ID or asr_config.get("asr", {}).get("decoder") != "ctc_argmax_greedy":
        raise ValueError("frozen ASR config is not uncorrected CTC greedy")
    if asr_config.get("seed") != 20260831:
        raise ValueError("frozen ASR config seed differs")
    if model_lock.get("model_id") != ASR_MODEL_ID or model_lock.get("revision") != ASR_MODEL_REVISION or model_lock.get("snapshot_hash") != ASR_SNAPSHOT_HASH or model_lock.get("tokenizer_vocabulary_hash") != ASR_TOKENIZER_VOCABULARY_HASH:
        raise ValueError("frozen ASR model lock differs")
    if assets.get("syncnet_checkpoint_sha256") != SYNCNET_CHECKPOINT_SHA256:
        raise ValueError("frozen SyncNet checkpoint lock differs")
    if _load(manifest_decision_path).get("status") != "GO" or _load(asr_decision_path).get("status") != "GO":
        raise ValueError("frozen ASR parent stage is incomplete")

    parent_rows = {str(row["sample_id"]): dict(row) for row in parent["cohort"]["records"]}
    manifest_rows = list(asr_manifest["records"])
    by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in manifest_rows:
        sample_id, arm = str(row["sample_id"]), str(row["arm"])
        if row.get("split") != "fit" or arm not in ("natural", "tts"):
            raise ValueError(f"ASR manifest contains non-fit or unsupported record: {sample_id}/{arm}")
        if sample_id not in parent_rows:
            raise ValueError(f"ASR manifest sample is absent from frozen parent: {sample_id}")
        if sample_id in by_sample and arm in by_sample[sample_id]:
            raise ValueError(f"duplicate ASR manifest cell: {sample_id}/{arm}")
        by_sample.setdefault(sample_id, {})[arm] = row
    if list(parent_rows) != list(by_sample):
        raise ValueError("ASR manifest order differs from frozen parent order")
    samples: list[dict[str, Any]] = []
    asr_records: dict[str, dict[str, dict[str, Any]]] = {}
    ledger: list[dict[str, Any]] = []
    for sample_id, parent_row in parent_rows.items():
        if set(by_sample[sample_id]) != {"natural", "tts"}:
            raise ValueError(f"incomplete ASR pair: {sample_id}")
        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "source_group": str(parent_row["source_group"]),
            "split": "fit",
            "transcript": str(by_sample[sample_id]["natural"]["transcript"]),
            "normalized_transcript": str(by_sample[sample_id]["natural"]["normalized_transcript"]),
            "video_path": str(by_sample[sample_id]["natural"]["video_path"]),
            "video_sha256": str(by_sample[sample_id]["natural"]["video_sha256"]),
            "video_duration_s": float(by_sample[sample_id]["natural"]["video_duration_s"]),
            "arms": {},
        }
        for arm in ("natural", "tts"):
            row = by_sample[sample_id][arm]
            if row["source_group"] != parent_row["source_group"] or row["video_sha256"] != parent_row["face_sha256"]:
                raise ValueError(f"frozen ASR manifest binding mismatch: {sample_id}/{arm}")
            audio_path = _resolve(root, str(row["audio_path"]))
            _file_hash(root, audio_path, str(row["audio_sha256"]), f"{arm} audio {sample_id}")
            _file_hash(root, str(row["video_path"]), str(row["video_sha256"]), f"face video {sample_id}")
            record_path = root / PARENT_ASR_RECORDS / arm / f"{sample_id}.json"
            record = _load(record_path)
            checked = _validate_asr_record(root, record, row, arm)
            asr_records.setdefault(sample_id, {})[arm] = checked
            sample["arms"][arm] = {
                "audio_path": str(row["audio_path"]),
                "audio_sha256": str(row["audio_sha256"]),
                "audio_duration_s": float(row["audio_duration_s"]),
                "record_path": _repo_path(root, record_path),
                "record_sha256": file_sha256(record_path),
            }
            ledger.append({
                "path": _repo_path(root, record_path),
                "fields": ["sample_id", "source_group", "arm", "audio_path", "audio_sha256", "greedy.transcript", "greedy.words", "reference_alignment.transcript", "reference_alignment.words", "model.model_id", "model.revision", "model.snapshot_hash", "emissions.sha256"],
                "purpose": "score-blind ASR/reference target and control selection",
            })
        samples.append(sample)
    assert_score_blind_fields([field for row in ledger for field in row["fields"] if "confidence" in field or "sync" in field or "score" in field])
    ledger.append({
        "path": _repo_path(root, asr_manifest_path),
        "fields": ["records.sample_id", "records.source_group", "records.split", "records.transcript", "records.audio_path", "records.audio_sha256", "records.video_path", "records.video_sha256"],
        "purpose": "frozen fit-only cohort and media provenance",
    })
    for row in ledger:
        assert_score_blind_fields(row["fields"])
    return {
        "schema_version": 1,
        "selection": "score_blind_03_asr_greedy_and_reference_timing_only",
        "sealed_splits_accessed": False,
        "parent_manifest_sha256": file_sha256(parent_path),
        "parent_test_lock_sha256": file_sha256(parent_lock_path),
        "asr_manifest_sha256": file_sha256(asr_manifest_path),
        "asr_config_sha256": file_sha256(config_path),
        "asr_model_lock_sha256": file_sha256(model_lock_path),
        "asr_assets_sha256": file_sha256(assets_path),
        "parent_file_hashes": parent_file_hashes,
        "model_lock": {"model_id": ASR_MODEL_ID, "revision": ASR_MODEL_REVISION, "snapshot_hash": ASR_SNAPSHOT_HASH, "tokenizer_vocabulary_hash": ASR_TOKENIZER_VOCABULARY_HASH},
        "samples": samples,
        "asr_records": asr_records,
        "consumed_field_ledger": ledger,
    }


def _span(word: Mapping[str, Any]) -> tuple[float, float]:
    start, end = float(word["start_s"]), float(word["end_s"])
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise ValueError("forced-reference word span is invalid")
    return start, end


def _quantize_span(span: tuple[float, float], sample_count: int, sample_rate: int = 16000) -> tuple[int, int]:
    start = max(0, int(math.floor(span[0] * sample_rate)))
    end = min(int(sample_count), int(math.ceil(span[1] * sample_rate)))
    if not start < end:
        raise ValueError("span is empty after sample-bound quantization")
    return start, end


def _contiguous(values: Sequence[int]) -> bool:
    return bool(values) and list(values) == list(range(min(values), max(values) + 1))


def _ops_by_reference(operations: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(op["reference_index"]): op for op in operations if op.get("reference_index") is not None}


def _edit_blocks(operations: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    blocks: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for operation in operations:
        if operation["operation"] == "equal":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(operation)
    if current:
        blocks.append(current)
    return blocks


def _stable_id(prefix: str, *parts: Any) -> str:
    text = "|".join([prefix, *(str(part) for part in parts)])
    return f"{prefix}_{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def _make_patch(sample: Mapping[str, Any], natural_words: Sequence[Mapping[str, Any]], tts_words: Sequence[Mapping[str, Any]], indices: Sequence[int], *, block_id: str, kind: str, sample_rate: int = 16000) -> dict[str, Any]:
    if not _contiguous(indices):
        raise ValueError("patch reference indices are not contiguous")
    first, last = min(indices), max(indices)
    natural_span = (_span(natural_words[first])[0], _span(natural_words[last])[1])
    tts_span = (_span(tts_words[first])[0], _span(tts_words[last])[1])
    natural_count = int(round(float(sample["arms"]["natural"]["audio_duration_s"]) * sample_rate))
    tts_count = int(round(float(sample["arms"]["tts"]["audio_duration_s"]) * sample_rate))
    destination = _quantize_span(natural_span, natural_count, sample_rate)
    donor = _quantize_span(tts_span, tts_count, sample_rate)
    return {
        "block_id": block_id,
        "kind": kind,
        "reference_indices": [int(value) for value in indices],
        "reference_word_count": len(indices),
        "destination_start_s": natural_span[0],
        "destination_end_s": natural_span[1],
        "donor_start_s": tts_span[0],
        "donor_end_s": tts_span[1],
        "destination_start_sample": destination[0],
        "destination_end_sample": destination[1],
        "donor_start_sample": donor[0],
        "donor_end_sample": donor[1],
        "destination_samples": destination[1] - destination[0],
        "donor_samples": donor[1] - donor[0],
        "sample_rate_hz": sample_rate,
        "words": [str(natural_words[index]["word"]) for index in indices],
    }


def _alignment(record: Mapping[str, Any]) -> tuple[list[str], list[Mapping[str, Any]], list[str], list[Mapping[str, Any]], list[dict[str, Any]]]:
    reference_words = list(record["reference_alignment"]["words"])
    reference = [str(row["word"]) for row in reference_words]
    prediction_words = list(record["greedy"]["words"])
    prediction = [str(row["word"]) for row in prediction_words]
    if str(record["reference_alignment"].get("transcript")) != " ".join(reference) or str(record["greedy"].get("transcript")) != " ".join(prediction):
        raise ValueError(f"ASR record transcript/word mismatch: {record.get('sample_id')}/{record.get('arm')}")
    operations = align_words(reference, prediction)
    return reference, reference_words, prediction, prediction_words, operations


def _tts_interval_clean(operations: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> bool:
    if not indices:
        return False
    first, last = min(indices), max(indices)
    positions = [position for position, operation in enumerate(operations) if operation.get("reference_index") is not None and first <= int(operation["reference_index"]) <= last]
    if not positions:
        return False
    interval = operations[min(positions):max(positions) + 1]
    return all(operation.get("operation") == "equal" for operation in interval)


def _target_blocks(sample: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    natural_ref, natural_words, _, _, natural_ops = _alignment(records["natural"])
    tts_ref, tts_words, _, _, tts_ops = _alignment(records["tts"])
    if natural_ref != tts_ref or natural_ref != normalized_words(str(sample["transcript"])):
        raise ValueError(f"reference transcript mismatch: {sample['sample_id']}")
    tts_by_ref = _ops_by_reference(tts_ops)
    targets: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    counts = {"natural_error_blocks": 0, "reference_mappable_blocks": 0, "tts_clean_donor_blocks": 0}
    for block_index, block in enumerate(_edit_blocks(natural_ops)):
        counts["natural_error_blocks"] += 1
        block_id = _stable_id("natural_error", sample["sample_id"], block_index, *(op["operation_id"] for op in block))
        ref_indices = [int(op["reference_index"]) for op in block if op.get("reference_index") is not None]
        has_sub_or_del = any(op["operation"] in ("substitution", "deletion") for op in block)
        if not ref_indices:
            excluded.append({"block_id": block_id, "reason": "no_reference_side_donor_mapping", "operations": [dict(op) for op in block]})
            continue
        counts["reference_mappable_blocks"] += 1
        if not _contiguous(ref_indices):
            excluded.append({"block_id": block_id, "reason": "non_contiguous_reference_range", "reference_indices": ref_indices})
            continue
        if not has_sub_or_del:
            excluded.append({"block_id": block_id, "reason": "insertion_only", "reference_indices": ref_indices})
            continue
        if any(op["operation"] == "insertion" for op in block):
            excluded.append({"block_id": block_id, "reason": "mixed_insertion_block", "reference_indices": ref_indices})
            continue
        if not _tts_interval_clean(tts_ops, ref_indices) or any(index not in tts_by_ref or tts_by_ref[index].get("operation") != "equal" for index in ref_indices):
            excluded.append({"block_id": block_id, "reason": "tts_donor_not_asr_clean", "reference_indices": ref_indices})
            continue
        try:
            patch = _make_patch(sample, natural_words, tts_words, ref_indices, block_id=block_id, kind="target")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            excluded.append({"block_id": block_id, "reason": "unsupported_reference_span", "message": str(exc), "reference_indices": ref_indices})
            continue
        counts["tts_clean_donor_blocks"] += 1
        targets.append({**patch, "operation_ids": [str(op["operation_id"]) for op in block], "operations": [dict(op) for op in block]})
    return targets, excluded, counts


def _overlap(left: Sequence[int], right: Sequence[int]) -> bool:
    return bool(set(left).intersection(right))


def _control_pool(sample: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    natural_ref, natural_words, _, _, natural_ops = _alignment(records["natural"])
    tts_ref, tts_words, _, _, tts_ops = _alignment(records["tts"])
    if natural_ref != tts_ref:
        raise ValueError("natural/TTS reference words differ")
    natural_by_ref = _ops_by_reference(natural_ops)
    tts_by_ref = _ops_by_reference(tts_ops)
    target_indices = [index for target in targets for index in target["reference_indices"]]
    max_len = len(natural_ref)
    all_controls: list[dict[str, Any]] = []
    for start in range(max_len):
        for length in range(1, max_len - start + 1):
            indices = list(range(start, start + length))
            if any(natural_by_ref.get(index, {}).get("operation") != "equal" or tts_by_ref.get(index, {}).get("operation") != "equal" for index in indices):
                continue
            if not _tts_interval_clean(tts_ops, indices):
                continue
            if _overlap(indices, target_indices):
                continue
            block_id = _stable_id("control", sample["sample_id"], start, length)
            try:
                patch = _make_patch(sample, natural_words, tts_words, indices, block_id=block_id, kind="control")
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            all_controls.append(patch)
    pools: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        target_duration = int(target["destination_samples"])
        target_count = int(target["reference_word_count"])
        ranked: list[dict[str, Any]] = []
        for candidate in all_controls:
            count_difference = abs(int(candidate["reference_word_count"]) - target_count)
            duration_ratio = float(candidate["destination_samples"]) / float(target_duration)
            tts_stretch_ratio = float(candidate["destination_samples"]) / float(candidate["donor_samples"])
            if count_difference > 1 or not 0.5 <= duration_ratio <= 2.0:
                continue
            ranked.append({**candidate, "target_block_id": str(target["block_id"]), "count_difference": count_difference, "duration_ratio": duration_ratio, "tts_stretch_ratio": tts_stretch_ratio})
        pools[str(target["block_id"])] = ranked
    return pools


def _ranked(pool: Sequence[Mapping[str, Any]], sample_id: str, target_id: str, replicate: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in pool:
        key_text = f"{SEED}|{sample_id}|{target_id}|{replicate}|{row['block_id']}"
        hash_key = hashlib.sha256(key_text.encode()).hexdigest()
        result.append({**dict(row), "ranking_key": [int(row["count_difference"]), abs(math.log(float(row["duration_ratio"]))), abs(math.log(float(row["tts_stretch_ratio"]))), hash_key]})
    return sorted(result, key=lambda row: tuple(row["ranking_key"]))


def _assignment(targets: Sequence[Mapping[str, Any]], pools: Mapping[str, Sequence[Mapping[str, Any]]], sample_id: str, replicate: int, forbidden_by_target: Mapping[str, set[str]] | None = None) -> dict[str, Any] | None:
    forbidden_by_target = forbidden_by_target or {}
    target_order = sorted(targets, key=lambda row: str(row["block_id"]))
    choices: dict[str, dict[str, Any]] = {}
    used: set[int] = set()

    def visit(position: int) -> bool:
        if position == len(target_order):
            total_target = sum(int(row["destination_samples"]) for row in targets)
            total_control = sum(int(row["destination_samples"]) for row in choices.values())
            return total_target > 0 and 0.8 <= total_control / total_target <= 1.25
        target = target_order[position]
        target_id = str(target["block_id"])
        ranked = _ranked(pools.get(target_id, ()), sample_id, target_id, replicate)
        forbidden = forbidden_by_target.get(target_id, set())
        for candidate in ranked:
            candidate_id = str(candidate["block_id"])
            indices = set(map(int, candidate["reference_indices"]))
            if candidate_id in forbidden or used.intersection(indices):
                continue
            used.update(indices)
            choices[target_id] = candidate
            if visit(position + 1):
                return True
            choices.pop(target_id, None)
            used.difference_update(indices)
        return False

    if not visit(0):
        return None
    selected = [choices[str(target["block_id"])] for target in targets]
    return {
        "replicate": int(replicate),
        "blocks": selected,
        "total_edited_duration_s": float(sum(row["destination_samples"] for row in selected) / 16000.0),
        "target_total_edited_duration_s": float(sum(row["destination_samples"] for row in targets) / 16000.0),
        "duration_ratio": float(sum(row["destination_samples"] for row in selected) / sum(row["destination_samples"] for row in targets)),
    }


def build_target_control_manifest(frozen: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(frozen["samples"])
    by_id = frozen["asr_records"]
    output_samples: list[dict[str, Any]] = []
    sample_reasons: list[dict[str, Any]] = []
    donor_sample_ids: set[str] = set()
    totals = {"natural_error_blocks": 0, "reference_mappable_blocks": 0, "tts_clean_donor_blocks": 0}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        targets, excluded, counts = _target_blocks(sample, by_id[sample_id])
        for key in totals:
            totals[key] += counts[key]
        if targets:
            donor_sample_ids.add(sample_id)
        if not targets:
            sample_reasons.append({"sample_id": sample_id, "reason": "no_eligible_target_block", "excluded_blocks": excluded})
            continue
        pools = _control_pool(sample, by_id[sample_id], targets)
        first = _assignment(targets, pools, sample_id, 0)
        if first is None:
            sample_reasons.append({"sample_id": sample_id, "reason": "incomplete_matched_controls", "target_blocks": targets, "candidate_pools": pools})
            continue
        selected0 = {str(row["target_block_id"]): str(row["block_id"]) for row in first["blocks"]}
        second = _assignment(targets, pools, sample_id, 1, {key: {value} for key, value in selected0.items()})
        if second is None:
            sample_reasons.append({"sample_id": sample_id, "reason": "incomplete_matched_controls", "target_blocks": targets, "candidate_pools": pools})
            continue
        output_samples.append({
            "sample_id": sample_id,
            "source_group": str(sample["source_group"]),
            "split": "fit",
            "target_blocks": targets,
            "control_candidate_pools": pools,
            "control_replicates": [first, second],
            "conditions": [{"condition": condition, "patch_blocks": [] if condition == "natural" else ([dict(row) for row in targets] if condition == "asr_targeted" else [dict(row) for row in ([first, second][int(condition[-1])])["blocks"]])} for condition in CONDITIONS],
        })
    output_samples.sort(key=lambda row: [str(sample["sample_id"]) for sample in samples].index(str(row["sample_id"])))
    retained_groups = [str(row["source_group"]) for row in output_samples]
    if len(retained_groups) != len(set(retained_groups)):
        raise ValueError("source_group_unit_mismatch")
    if len(output_samples) < 12:
        science_status = "INSUFFICIENT"
    else:
        science_status = "PENDING"
    return {
        "schema_version": 1,
        "selection": "natural_substitution_deletion_with_tts_asr_clean_reference_donor",
        "seed": SEED,
        "source_manifest_sha256": frozen["asr_manifest_sha256"],
        "frozen_lock_sha256": canonical_hash({key: value for key, value in frozen.items() if key not in ("asr_records",)}),
        "sealed_splits_accessed": False,
        "readiness_counts": {**totals, "samples_containing_tts_clean_donor_blocks": len(donor_sample_ids)},
        "eligible_sample_count": len(output_samples),
        "eligible_source_group_count": len({str(row["source_group"]) for row in output_samples}),
        "excluded_samples": sample_reasons,
        "samples": output_samples,
        "required_conditions": list(CONDITIONS),
        "pre_render_science_status": science_status,
    }
