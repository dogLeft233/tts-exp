#!/usr/bin/env python3
"""Verify the frozen WORLD recipient-phone pipeline with NULL word indices.

This audit reads the sealed shape-run context sidecars and only the mel-clock,
support, and start arrays from context NPZs. It binds each embedded phone map
record to the exact physical-manifest entry and phone_map.json bytes in the
completed v6 archive. It never opens score NPZs or computes/report score effects.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import tarfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


RUN_ROOT = Path(r"C:\Users\dogle\.codex\tts-exp-artifacts\20261002\world_shape_gpu14_bundle05\restored_verified_attempt02\run")
ARCHIVE_PATH = Path(r"F:\tts-research-archives\20261001\world_v6_attempt07_v6_attempt01\world_v6_attempt07_v6.tar")
CONSUMER17 = Path("scripts/experiments/independent_world_phone_shape_mean_dynamic_consumer_20261001_attempt17.py")
CONSUMER18 = Path("scripts/experiments/independent_world_phone_shape_mean_dynamic_consumer_20261001_attempt18.py")
PHYSICAL_MANIFEST_MEMBER = "world_phone_transfer_20261001_overlay/mfa_world_physical_eval_20261001_v2_view/world_phone_transfer_evaluation_physical_overlay.json"
PHYSICAL_ARCHIVE_PREFIX = "/root/tts-research-system-20261001/"
PHYSICAL_MANIFEST_RAW_SHA256 = "27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3"
PHYSICAL_MANIFEST_PAYLOAD_SHA256 = "eb77432936c3454a241eb2505a966ffa0d05aa5be8514451cc2da66aa7ed9de8"
EXPECTED_RUN_ROWS = 160
EXPECTED_CLIPS = 80
EXPECTED_SPEAKERS = 40


class AuditError(RuntimeError):
    pass


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_selected_symbols(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected_functions = {
        "_phone_intervals", "_owner_arrays", "_window_unions", "_phone_weights",
        "_boundary_diagnostics", "_token_counts_by_label", "_support_indices",
        "_official_mel_starts",
    }
    selected_assignments = {"PHONE_LABELS", "PHONE_OUTCOMES"}
    namespace: dict[str, Any] = {"np": np, "math": math, "Any": Any}
    count = 0
    for node in tree.body:
        keep = False
        if isinstance(node, ast.ClassDef) and node.name == "ConsumerError":
            keep = True
        elif isinstance(node, ast.FunctionDef) and node.name in selected_functions:
            keep = True
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in selected_assignments for target in node.targets
        ):
            keep = True
        if keep:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(path), "exec"), namespace)
            count += 1
    if count != 11 or not selected_functions.issubset(namespace) or not selected_assignments.issubset(namespace):
        raise AuditError(f"selected consumer symbol extraction incomplete: nodes={count}")
    return namespace


def independent_center_reference(
    unions: list[np.ndarray], mel_n: int, intervals: list[dict[str, Any]], labels: tuple[str, ...], phases: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.arange(mel_n, dtype=np.float64) / 80.0
    label_slot = {label: index for index, label in enumerate(labels)}
    owner_count = np.zeros(mel_n, dtype=np.int16)
    owner_index = np.full(mel_n, -1, dtype=np.int32)
    for item in intervals:
        active = (centers >= item["start"]) & (centers < item["end"])
        owner_count += active.astype(np.int16)
        owner_index[active] = item["phone_index"]
    center_weights = np.zeros((len(labels), len(phases), len(unions)), dtype=np.float64)
    for q, columns in enumerate(unions):
        columns = np.asarray(columns, dtype=np.int64)
        for token in intervals:
            slot = label_slot.get(token["label"])
            if slot is None or not token["matched"]:
                continue
            phase = (centers[columns] - token["start"]) / (token["end"] - token["start"])
            assigned = (
                (centers[columns] >= token["start"])
                & (centers[columns] < token["end"])
                & (owner_count[columns] == 1)
                & (owner_index[columns] == token["phone_index"])
            )
            phase_masks = (
                assigned,
                assigned & (phase < 1.0 / 3.0),
                assigned & (phase >= 1.0 / 3.0) & (phase < 2.0 / 3.0),
                assigned & (phase >= 2.0 / 3.0) & (phase < 1.0),
            )
            for phase_index, mask in enumerate(phase_masks):
                center_weights[slot, phase_index, q] += np.count_nonzero(mask) / float(len(columns))

    # Independent exact 50 ms integration: cut each column at all phone and
    # selected-token phase boundaries, then assign each subinterval by midpoint.
    stft_weights = np.zeros_like(center_weights)
    global_cuts = {float(v) for item in intervals for v in (item["start"], item["end"])}
    for item in intervals:
        if item["matched"] and item["label"] in label_slot:
            duration = item["end"] - item["start"]
            global_cuts.add(item["start"] + duration / 3.0)
            global_cuts.add(item["start"] + 2.0 * duration / 3.0)
    global_cuts = sorted(global_cuts)
    for q, columns in enumerate(unions):
        columns = np.asarray(columns, dtype=np.int64)
        for col in columns:
            left = float(centers[col] - 0.025)
            right = float(centers[col] + 0.025)
            cuts = [left] + [v for v in global_cuts if left < v < right] + [right]
            for start, end in zip(cuts, cuts[1:]):
                if end <= start:
                    continue
                midpoint = 0.5 * (start + end)
                active = [item for item in intervals if item["start"] <= midpoint < item["end"]]
                if len(active) != 1 or not active[0]["matched"]:
                    continue
                token = active[0]
                slot = label_slot.get(token["label"])
                if slot is None:
                    continue
                fraction = (midpoint - token["start"]) / (token["end"] - token["start"])
                phase_index = 1 if fraction < 1.0 / 3.0 else 2 if fraction < 2.0 / 3.0 else 3
                mass = (end - start) / 0.05 / float(len(columns))
                stft_weights[slot, 0, q] += mass
                stft_weights[slot, phase_index, q] += mass
    return owner_count, owner_index, center_weights, stft_weights


def independent_boundary_reference(
    unions: list[np.ndarray], centers: np.ndarray, intervals: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    boundary_times = [float(value) for item in intervals for value in (item["start"], item["end"])]
    near40 = np.zeros(len(unions), dtype=np.float32)
    crossing25 = np.zeros(len(unions), dtype=np.float32)
    mixed = np.zeros(len(unions), dtype=np.uint8)
    identities = np.zeros(len(unions), dtype=np.int16)
    for query, columns in enumerate(unions):
        times = [float(centers[int(col)]) for col in columns]
        owners = set()
        for time_value in times:
            for item in intervals:
                if item["start"] <= time_value < item["end"]:
                    owners.add(item["phone_index"])
        identities[query] = len(owners)
        mixed[query] = int(len(owners) > 1)
        if boundary_times:
            distances = [min(abs(time_value - boundary) for boundary in boundary_times) for time_value in times]
            near40[query] = sum(distance <= 0.040 + 1e-12 for distance in distances) / len(distances)
            crossing25[query] = sum(distance <= 0.025 + 1e-12 for distance in distances) / len(distances)
    return {
        "boundary_within_40ms_column_fraction": near40,
        "stft50_boundary_crossing_column_fraction": crossing25,
        "mixed_phone_rf": mixed,
        "recipient_phone_identity_count": identities,
    }


def compare_exact(left: Any, right: Any, label: str) -> None:
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AuditError(f"{label}: key mismatch")
        for key in left:
            compare_exact(left[key], right[key], f"{label}/{key}")
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            raise AuditError(f"{label}: exact array parity failed")
    elif left != right:
        raise AuditError(f"{label}: {left!r} != {right!r}")


def main() -> None:
    run_root = RUN_ROOT.resolve(strict=True)
    context_dir = run_root / "contexts"
    if not context_dir.is_dir():
        raise AuditError("restored contexts directory is missing")
    run_inventory = json.loads((run_root / "all_files_sha256.json").read_text(encoding="utf-8"))
    if not isinstance(run_inventory, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in run_inventory.items()):
        raise AuditError("shape run all-files inventory is not a flat relative-path/hash map")
    consumer18 = load_selected_symbols(CONSUMER18)
    consumer17 = load_selected_symbols(CONSUMER17)

    with tarfile.open(ARCHIVE_PATH, mode="r:") as tar:
        members = {member.name: member for member in tar.getmembers()}
        if PHYSICAL_MANIFEST_MEMBER not in members:
            raise AuditError("frozen physical parent manifest is absent from the verified v6 archive")
        manifest_member = members[PHYSICAL_MANIFEST_MEMBER]
        if not manifest_member.isfile():
            raise AuditError("physical parent manifest member is not a regular file")
        manifest_raw = tar.extractfile(manifest_member).read()
        manifest_raw_sha = sha_bytes(manifest_raw)
        if manifest_raw_sha != PHYSICAL_MANIFEST_RAW_SHA256:
            raise AuditError(f"physical manifest raw SHA mismatch: {manifest_raw_sha}")
        physical_manifest = json.loads(manifest_raw.decode("utf-8"))
        physical_payload_sha = sha_bytes(canonical({k: v for k, v in physical_manifest.items() if k != "manifest_sha256"}))
        if physical_manifest.get("manifest_sha256") != PHYSICAL_MANIFEST_PAYLOAD_SHA256 or physical_payload_sha != PHYSICAL_MANIFEST_PAYLOAD_SHA256:
            raise AuditError("physical parent manifest embedded payload fingerprint mismatch")
        physical_refs: dict[tuple[str, str], dict[str, Any]] = {}
        for record in physical_manifest.get("records", []):
            sample_id = record.get("id")
            for direction, direction_record in record.get("directions", {}).items():
                if direction not in {"natural_to_tts", "tts_to_natural"}:
                    raise AuditError(f"unexpected physical manifest direction: {direction}")
                physical_refs[(sample_id, direction)] = direction_record["phone_map"]
        if len(physical_refs) != EXPECTED_RUN_ROWS:
            raise AuditError(f"physical parent map references are not 160: {len(physical_refs)}")

        contexts = sorted(context_dir.glob("*.json"))
        if len(contexts) != EXPECTED_RUN_ROWS:
            raise AuditError(f"shape context sidecar count is not 160: {len(contexts)}")
        observed_context_keys: set[tuple[str, str]] = set()
        clip_speakers: dict[str, str] = {}
        total: Counter[str] = Counter()
        word_type_counts: Counter[str] = Counter()
        total_query_rows = 0
        total_union_columns = 0
        selected_weight_mass = Counter()
        exact_legacy_parity_rows = 0
        exact_reference_center_rows = 0
        exact_reference_boundary_rows = 0
        max_center_reference_error = 0.0
        max_stft_reference_error = 0.0
        per_context: list[dict[str, Any]] = []
        run_inventory_checks = 0
        map_tar_hash_checks = 0
        map_record_parity_checks = 0
        context_npz_hash_checks = 0
        context_json_hash_checks = 0

        for sidecar_path in contexts:
            sidecar_rel = sidecar_path.relative_to(run_root).as_posix()
            expected_sidecar_sha = run_inventory.get(sidecar_rel)
            if not expected_sidecar_sha or sha_file(sidecar_path) != expected_sidecar_sha:
                raise AuditError(f"shape run inventory does not bind context JSON: {sidecar_rel}")
            context_json_hash_checks += 1
            meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sample_id = str(meta.get("sample_id", ""))
            direction = str(meta.get("direction", ""))
            key = (sample_id, direction)
            if key in observed_context_keys or key not in physical_refs:
                raise AuditError(f"duplicate or absent physical manifest context: {key}")
            observed_context_keys.add(key)
            if direction not in {"natural_to_tts", "tts_to_natural"}:
                raise AuditError(f"unexpected shape context direction: {direction}")
            if meta.get("key") != f"{sample_id}__{direction}":
                raise AuditError(f"context key mismatch: {key}")
            clip_speakers[sample_id] = str(meta.get("speaker_id", ""))

            manifest_ref = physical_refs[key]
            sidecar_ref = meta.get("phone_map")
            if (
                not isinstance(sidecar_ref, dict)
                or set(sidecar_ref) != {"path", "file_sha256"}
                or sidecar_ref.get("path") != manifest_ref.get("path")
                or sidecar_ref.get("file_sha256") != manifest_ref.get("file_sha256")
            ):
                raise AuditError(f"physical-manifest to shape-sidecar phone-map reference mismatch: {key}")
            remote_path = str(manifest_ref.get("path", ""))
            if not remote_path.startswith(PHYSICAL_ARCHIVE_PREFIX):
                raise AuditError(f"phone-map source path is outside the archived physical root: {key}")
            map_member_name = remote_path[len(PHYSICAL_ARCHIVE_PREFIX):]
            member = members.get(map_member_name)
            if member is None or not member.isfile():
                raise AuditError(f"phone-map regular member is absent from archive: {map_member_name}")
            phone_raw = tar.extractfile(member).read()
            phone_sha = sha_bytes(phone_raw)
            if phone_sha != manifest_ref.get("file_sha256"):
                raise AuditError(f"actual phone-map bytes differ from physical manifest: {key}")
            map_tar_hash_checks += 1
            phone_map = json.loads(phone_raw.decode("utf-8"))
            if phone_map != meta.get("phone_map_record"):
                raise AuditError(f"shape sidecar embedded phone-map record differs from exact archived bytes: {key}")
            if phone_map.get("matched_phone_count") != manifest_ref.get("matched_phone_count"):
                raise AuditError(f"physical-manifest matched_phone_count differs from exact map payload: {key}")
            map_record_parity_checks += 1
            if (
                phone_map.get("id") != sample_id
                or phone_map.get("recipient_condition") != meta.get("recipient_condition")
                or phone_map.get("donor_condition") != meta.get("donor_condition")
            ):
                raise AuditError(f"phone-map condition/id identity mismatch: {key}")

            recipient_rows = phone_map.get("recipient_phones")
            matched_rows = phone_map.get("matched_phones")
            if not isinstance(recipient_rows, list) or not isinstance(matched_rows, list):
                raise AuditError(f"phone map does not expose the complete recipient/matched tiers: {key}")
            matched_indexes = [int(row["recipient_phone_index"]) for row in matched_rows]
            if len(set(matched_indexes)) != len(matched_indexes):
                raise AuditError(f"matched phone map repeats recipient index: {key}")
            matched_set = set(matched_indexes)
            total["recipient_rows"] += len(recipient_rows)
            total["matched_rows"] += len(matched_rows)
            total["unmatched_rows"] += len(recipient_rows) - len(matched_set)
            null_count = 0
            missing_count = 0
            null_blank = 0
            null_matched = 0
            null_selected = 0
            selected_unmatched = 0
            local_types: Counter[str] = Counter()
            for row in recipient_rows:
                raw_index = row.get("word_index", -1)
                if "word_index" not in row:
                    missing_count += 1
                    word_type_counts["missing"] += 1
                    local_types["missing"] += 1
                elif raw_index is None:
                    null_count += 1
                    word_type_counts["null"] += 1
                    local_types["null"] += 1
                    label = str(row.get("normalized_label", ""))
                    index_value = int(row["index"])
                    is_matched = index_value in matched_set
                    null_blank += int(label == "")
                    null_matched += int(is_matched)
                    null_selected += int(label in consumer18["PHONE_LABELS"])
                elif type(raw_index) is int:
                    word_type_counts["integer"] += 1
                    local_types["integer"] += 1
                else:
                    word_type_counts[type(raw_index).__name__] += 1
                    local_types[type(raw_index).__name__] += 1
                if int(row["index"]) in matched_set and str(row.get("normalized_label", "")) not in consumer18["PHONE_LABELS"]:
                    pass
                if int(row["index"]) not in matched_set and str(row.get("normalized_label", "")) in consumer18["PHONE_LABELS"]:
                    selected_unmatched += 1
            total["null_word_index"] += null_count
            total["missing_word_index"] += missing_count
            total["null_blank_label"] += null_blank
            total["null_matched"] += null_matched
            total["null_selected_label"] += null_selected
            total["selected_label_unmatched"] += selected_unmatched
            if not set(matched_set).issubset({int(row["index"]) for row in recipient_rows}):
                raise AuditError(f"matched phone tier contains index outside full recipient tier: {key}")

            # Exercise the exact pre-patch failure on the first sealed row so the
            # null cast is independently localized, while keeping its traceback
            # as audit evidence rather than swallowing the exception.
            original_error = None
            if len(observed_context_keys) == 1:
                try:
                    consumer17["_phone_intervals"](phone_map)
                except Exception as exc:  # intentional diagnostic capture
                    original_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                else:
                    raise AuditError("frozen consumer17 unexpectedly accepted first actual phone map")

            intervals = consumer18["_phone_intervals"](phone_map)
            if len(intervals) != len(recipient_rows):
                raise AuditError(f"phone interval conversion dropped recipient-tier rows: {key}")
            output_null_count = sum(item["word_index"] is None for item in intervals)
            if output_null_count != null_count:
                raise AuditError(f"nullable word_index values were not preserved 1:1: {key}")
            if any(item["word_index"] is None and (item["matched"] or item["label"] in consumer18["PHONE_LABELS"]) for item in intervals):
                raise AuditError(f"nullable word_index row is matched/selected: {key}")

            context_npz_ref = meta.get("context_npz")
            if not isinstance(context_npz_ref, dict) or not context_npz_ref.get("path"):
                raise AuditError(f"context sidecar lacks NPZ reference: {key}")
            npz_rel = str(context_npz_ref["path"])
            expected_npz_sha = run_inventory.get(npz_rel)
            if not expected_npz_sha or expected_npz_sha != context_npz_ref.get("sha256"):
                raise AuditError(f"context NPZ sidecar/inventory hash ref mismatch: {key}")
            npz_path = (run_root / npz_rel).resolve(strict=True)
            if npz_path.parent != context_dir.resolve(strict=True) or sha_file(npz_path) != expected_npz_sha:
                raise AuditError(f"context NPZ readback hash mismatch: {key}")
            context_npz_hash_checks += 1
            with np.load(npz_path, allow_pickle=False) as z:
                mel_keys = [f"{cell}_mel" for cell in ("q00", "q10", "q01", "q11", "SELF_GRID")]
                start_keys = [f"{cell}_starts" for cell in ("q00", "q10", "q01", "q11", "SELF_GRID")]
                support_keys = [f"support_{support}" for support in ("full", "valid", "guard20", "guard15")]
                needed = set(mel_keys + start_keys + support_keys)
                if not needed.issubset(z.files):
                    raise AuditError(f"required mel/start/support arrays are absent: {key}")
                mel_arrays = {name: z[name].copy() for name in mel_keys}
                start_arrays = {name: z[name].copy() for name in start_keys}
                saved_supports = {name.removeprefix("support_"): z[name].copy() for name in support_keys}

            query_n = int(meta["common_query_count"])
            mel_shape = tuple(mel_arrays["q00_mel"].shape)
            if len(mel_shape) != 2 or mel_shape[0] != 80 or not np.issubdtype(mel_arrays["q00_mel"].dtype, np.floating):
                raise AuditError(f"q00 mel clock has unexpected shape/dtype: {key}/{mel_shape}")
            mel_n = int(mel_shape[-1])
            starts = start_arrays["q00_starts"]
            if starts.dtype != np.int32 or starts.ndim != 1 or len(starts) != query_n + 5:
                raise AuditError(f"q00 starts do not bind query_count+five-frame clock: {key}")
            if not np.array_equal(starts, consumer18["_official_mel_starts"](mel_n)):
                raise AuditError(f"q00 mel starts differ from official Wav2Lip chunk loop: {key}")
            if any(not np.array_equal(start_arrays[name], starts) for name in start_keys[1:]):
                raise AuditError(f"five-cell mel starts are not bit-identical: {key}")
            if any(tuple(mel_arrays[name].shape) != mel_shape or not np.isfinite(mel_arrays[name]).all() for name in mel_keys):
                raise AuditError(f"five-cell mel clocks/dtypes are inconsistent or nonfinite: {key}")
            supports = consumer18["_support_indices"](query_n)
            if set(saved_supports) != set(supports):
                raise AuditError(f"support names differ from frozen four-support contract: {key}")
            for name in supports:
                if saved_supports[name].dtype != np.int32 or not np.array_equal(saved_supports[name], supports[name]):
                    raise AuditError(f"stored {name} support differs from exact frozen support rule: {key}")
            unions, indptr, flat = consumer18["_window_unions"](starts, query_n, mel_n)
            if len(unions) != query_n or indptr.dtype != np.int32 or flat.dtype != np.int32:
                raise AuditError(f"receptive-field union structure has unexpected dimensions: {key}")
            if np.any(flat < 0) or np.any(flat >= mel_n) or np.any(np.diff(indptr) <= 0):
                raise AuditError(f"receptive-field union has invalid/empty mel support: {key}")
            weights, stft_weights, owner_count, owner_index, centers = consumer18["_phone_weights"](unions, mel_n, intervals)
            boundary = consumer18["_boundary_diagnostics"](unions, centers, intervals)
            token_counts = consumer18["_token_counts_by_label"](intervals, centers, owner_count, owner_index, unions, supports)
            if weights.shape != (16, 4, query_n) or stft_weights.shape != weights.shape:
                raise AuditError(f"phone coverage weight shape differs from frozen 16x4xquery design: {key}")
            if not np.isfinite(weights).all() or not np.isfinite(stft_weights).all() or np.any(weights < 0) or np.any(stft_weights < 0):
                raise AuditError(f"phone weights are nonfinite or negative: {key}")
            if not np.allclose(weights[:, 1:, :].sum(axis=1), weights[:, 0, :], rtol=0, atol=1e-7):
                raise AuditError(f"center ownership phase weights do not partition whole-phone weights: {key}")
            if not np.allclose(stft_weights[:, 1:, :].sum(axis=1), stft_weights[:, 0, :], rtol=0, atol=1e-7):
                raise AuditError(f"STFT50 sensitivity phase weights do not partition whole-phone weights: {key}")
            if np.any(weights.sum(axis=0)[0] > 1.0 + 1e-7) or np.any(stft_weights.sum(axis=0)[0] > 1.0 + 1e-7):
                raise AuditError(f"unique selected-phone coverage exceeds fixed receptive-field denominator: {key}")
            if any(not np.isfinite(value).all() for value in boundary.values()):
                raise AuditError(f"boundary diagnostics contain nonfinite values: {key}")
            if set(token_counts) != set(supports) or any(set(token_counts[s]) != set(consumer18["PHONE_LABELS"]) for s in supports):
                raise AuditError(f"matched token count output is incomplete: {key}")
            if any(len(token_counts[s][label]) != 4 or any(int(v) < 0 for v in token_counts[s][label]) for s in supports for label in consumer18["PHONE_LABELS"]):
                raise AuditError(f"matched token counts have invalid phase vector: {key}")

            # Independent reconstruction of center ownership, center weights,
            # exact STFT50 overlap weights, and boundary summaries.
            ref_owner_count, ref_owner_index, ref_weights, ref_stft = independent_center_reference(
                unions, mel_n, intervals, tuple(consumer18["PHONE_LABELS"]), tuple(consumer18["PHONE_OUTCOMES"]),
            )
            compare_exact(owner_count, ref_owner_count, f"{key}/owner_count")
            compare_exact(owner_index, ref_owner_index, f"{key}/owner_index")
            if not np.array_equal(weights, ref_weights):
                raise AuditError(f"independent center-weight reconstruction differs: {key}")
            max_center_reference_error = max(max_center_reference_error, float(np.max(np.abs(weights - ref_weights), initial=0.0)))
            max_stft_reference_error = max(max_stft_reference_error, float(np.max(np.abs(stft_weights - ref_stft), initial=0.0)))
            if not np.allclose(stft_weights, ref_stft, rtol=0, atol=1e-12):
                raise AuditError(f"independent exact STFT50-weight reconstruction differs: {key}")
            exact_reference_center_rows += 1
            exact_reference_boundary_rows += 1
            ref_boundary = independent_boundary_reference(unions, centers, intervals)
            compare_exact(boundary, ref_boundary, f"{key}/boundary")

            # The null->legacy sentinel mapping is deliberately used only for a
            # diagnostic parity check: word_index is not read by ownership,
            # boundaries, receptive-field coverage, or token-count logic.
            legacy_intervals = [
                {**item, "word_index": -1 if item["word_index"] is None else item["word_index"]}
                for item in intervals
            ]
            legacy_w, legacy_stft_w, legacy_owner_count, legacy_owner_index, legacy_centers = consumer18["_phone_weights"](unions, mel_n, legacy_intervals)
            legacy_boundary = consumer18["_boundary_diagnostics"](unions, legacy_centers, legacy_intervals)
            legacy_tokens = consumer18["_token_counts_by_label"](legacy_intervals, legacy_centers, legacy_owner_count, legacy_owner_index, unions, supports)
            compare_exact(weights, legacy_w, f"{key}/null_vs_sentinel_center_weights")
            compare_exact(stft_weights, legacy_stft_w, f"{key}/null_vs_sentinel_stft_weights")
            compare_exact(owner_count, legacy_owner_count, f"{key}/null_vs_sentinel_owner_count")
            compare_exact(owner_index, legacy_owner_index, f"{key}/null_vs_sentinel_owner_index")
            compare_exact(boundary, legacy_boundary, f"{key}/null_vs_sentinel_boundary")
            compare_exact(token_counts, legacy_tokens, f"{key}/null_vs_sentinel_token_counts")
            exact_legacy_parity_rows += 1

            for support in token_counts:
                for label in consumer18["PHONE_LABELS"]:
                    selected_weight_mass[f"{support}:whole"] += float(np.sum(weights[consumer18["PHONE_LABELS"].index(label), 0, supports[support]], dtype=np.float64))
            total_query_rows += query_n
            total_union_columns += len(flat)
            per_context.append({
                "sample_id": sample_id,
                "direction": direction,
                "sidecar_sha256": expected_sidecar_sha,
                "context_npz_sha256": expected_npz_sha,
                "phone_map_sha256": phone_sha,
                "recipient_phone_count": len(recipient_rows),
                "matched_phone_count": len(matched_rows),
                "null_word_index_count": null_count,
                "missing_word_index_count": missing_count,
                "null_blank_label_unmatched_count": null_blank,
                "null_selected_or_matched_count": null_selected + null_matched,
                "selected_unmatched_count": selected_unmatched,
                "query_count": query_n,
                "mel_frame_count": mel_n,
                "union_column_count": len(flat),
                "null_word_index_error": original_error,
                "center_weight_shape": list(weights.shape),
                "boundary_diagnostic_names": sorted(boundary),
                "support_counts": {name: len(value) for name, value in supports.items()},
            })

    if observed_context_keys != set(physical_refs):
        raise AuditError("shape contexts and physical phone-map manifest do not have exact 160-row identity parity")
    if len(clip_speakers) != EXPECTED_CLIPS or len(set(clip_speakers.values())) != EXPECTED_SPEAKERS:
        raise AuditError("shape context rows do not retain the frozen 80-clips/40-speakers denominator")
    if total["recipient_rows"] != 5803 or total["matched_rows"] != 5344 or total["unmatched_rows"] != 459:
        raise AuditError(f"full recipient/matched phone tier counts drifted: {dict(total)}")
    if total["null_word_index"] != 432 or total["missing_word_index"] != 0:
        raise AuditError(f"nullable/missing word-index counts drifted: {dict(total)}")
    if total["null_blank_label"] != 432 or total["null_matched"] != 0 or total["null_selected_label"] != 0:
        raise AuditError(f"null word-index rows are not exactly blank/unmatched/unselected: {dict(total)}")
    if total["selected_label_unmatched"] != 11:
        raise AuditError(f"selected-label unmatched token count drifted: {dict(total)}")
    if map_tar_hash_checks != EXPECTED_RUN_ROWS or map_record_parity_checks != EXPECTED_RUN_ROWS:
        raise AuditError("not all 160 physical phone maps were hash- and record-verified")
    if (context_json_hash_checks, context_npz_hash_checks) != (EXPECTED_RUN_ROWS, EXPECTED_RUN_ROWS):
        raise AuditError("not all 160 context sidecars/NPZs were verified against run inventory")
    if (exact_legacy_parity_rows, exact_reference_center_rows, exact_reference_boundary_rows) != (EXPECTED_RUN_ROWS,) * 3:
        raise AuditError("not all 160 phone contexts completed full pipeline parity")

    report = {
        "schema": "world_shape_phone_interval_pipeline_audit_v1",
        "status": "PASS",
        "scope": "metadata and receptive-field phone ownership only; no score NPZ, score matrix, effect, or contrast values opened",
        "runtime": {"python": __import__("sys").version, "numpy": np.__version__},
        "inputs": {
            "shape_run_root": str(run_root),
            "shape_all_files_sha256": sha_file(run_root / "all_files_sha256.json"),
            "shape_all_files_entry_count": len(run_inventory),
            "physical_v6_archive": str(ARCHIVE_PATH),
            "physical_parent_manifest_member": PHYSICAL_MANIFEST_MEMBER,
            "physical_parent_manifest_raw_sha256": manifest_raw_sha,
            "physical_parent_manifest_payload_sha256": physical_payload_sha,
            "consumer17_path": str(CONSUMER17.resolve(strict=True)),
            "consumer17_sha256": sha_file(CONSUMER17),
            "consumer18_path": str(CONSUMER18.resolve(strict=True)),
            "consumer18_sha256": sha_file(CONSUMER18),
        },
        "denominators": {
            "context_rows": len(per_context),
            "clips": len(clip_speakers),
            "speakers": len(set(clip_speakers.values())),
            "directions": {name: sum(row["direction"] == name for row in per_context) for name in ("tts_to_natural", "natural_to_tts")},
            "phone_maps_tar_sha_checked": map_tar_hash_checks,
            "phone_map_record_sidecar_parity": map_record_parity_checks,
            "context_json_inventory_hash_checked": context_json_hash_checks,
            "context_npz_inventory_hash_checked": context_npz_hash_checks,
        },
        "phone_tier_counts": {
            **dict(total),
            "recipient_word_index_value_types": dict(word_type_counts),
            "recipient_rows_by_context": {"min": min(row["recipient_phone_count"] for row in per_context), "max": max(row["recipient_phone_count"] for row in per_context)},
            "matched_rows_by_context": {"min": min(row["matched_phone_count"] for row in per_context), "max": max(row["matched_phone_count"] for row in per_context)},
        },
        "pipeline_checks": {
            "official_mel_starts_and_five_cell_identity": "160/160 PASS",
            "full_query_plus_five_frame_support_and_exact_saved_four_supports": "160/160 PASS",
            "complete_recipient_intervals_retained": "5803/5803 PASS",
            "null_word_index_preserved_as_json_null": "432/432 PASS",
            "owner_counts_and_indices_independent_reconstruction": "160/160 exact PASS",
            "center_phone_weights_independent_reconstruction": "160/160 exact PASS",
            "stft50_phase_weight_partition_and_finite": "160/160 PASS",
            "boundary_diagnostics_independent_reconstruction": "160/160 exact PASS",
            "support_matched_token_counts_and_phase_vectors": "160/160 PASS",
            "null_to_legacy_sentinel_downstream_parity": "160/160 exact PASS",
            "max_center_weight_independent_abs_error": max_center_reference_error,
            "max_stft50_weight_independent_abs_error": max_stft_reference_error,
            "summed_query_windows": total_query_rows,
            "summed_receptive_union_columns": total_union_columns,
            "selected_phone_weight_mass_by_support": dict(selected_weight_mass),
        },
        "first_frozen_consumer17_error": next((row["null_word_index_error"] for row in per_context if row["null_word_index_error"] is not None), None),
        "contexts": per_context,
    }
    out = Path("runs/recovered_20260930/remote/world_shape_phone_interval_audit_attempt01")
    out.mkdir(parents=True, exist_ok=False)
    output_path = out / "phone_interval_pipeline_audit.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "report": str(output_path.resolve()), "report_sha256": sha_file(output_path), "context_rows": len(per_context), "recipient_rows": total["recipient_rows"], "null_word_index": total["null_word_index"], "legacy_parity_rows": exact_legacy_parity_rows}, sort_keys=True))


if __name__ == "__main__":
    main()
