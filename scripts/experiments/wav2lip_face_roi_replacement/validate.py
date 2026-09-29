from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing.media import (
    decode_pcm16,
    decode_video_frames,
    source_pcm16,
)

from . import config
from .audio import read_pcm16
from .common import (
    ProtocolError,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)

ALLOWED_SCIENTIFIC = {"CONTROL_FAILED", "REPLACEMENT_NOT_ESTABLISHED", "WAV2LIP_REPLACEMENT_PILOT_PASS"}


def bootstrap_ci(values: Sequence[float], groups: Sequence[str], *, seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> list[float]:
    if len(values) != len(groups) or not values or draws <= 0:
        raise ProtocolError("bootstrap inputs are empty or mismatched")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        number = float(value)
        if not math.isfinite(number):
            raise ProtocolError("bootstrap values must be finite")
        by_group[str(group)].append(number)
    labels = sorted(by_group)
    means = np.asarray([np.mean(by_group[label]) for label in labels], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(labels), size=(int(draws), len(labels)))
    estimates = means[sampled].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _same_float_list(left: Sequence[Any], right: Sequence[Any], tolerance: float = 1e-12) -> bool:
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right, strict=True))


def _required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"required artifact is missing: {path}")
    return verify_self_hashed_json(path)


def _validate_preflight_block(root: Path, final: Mapping[str, Any]) -> dict[str, Any]:
    if final.get("protocol_id") != config.PROTOCOL_ID or final.get("status") != "blocked" or final.get("engineering_decision") != "BLOCKED" or final.get("scientific_decision") is not None:
        raise ProtocolError("preflight blocked final is malformed")
    for key in ("training_authorized", "reference_conditioned_audio_head_spec_eligible", "generalization_established"):
        if final.get(key) is not False:
            raise ProtocolError(f"blocked final has unsafe flag: {key}")
    result = root / "result.md"
    if not result.is_file():
        raise ProtocolError("blocked run result.md is missing")
    payload = {"status": "valid", "mode": "engineering_blocked", "final_sha256": file_sha256(root / "final.json"), "result_sha256": file_sha256(result), "scientific_decision": None}
    return payload


def _validate_protocol(root: Path) -> dict[str, Any]:
    protocol = _required_json(root / "protocol.json")
    if protocol.get("protocol_id") != config.PROTOCOL_ID or protocol.get("protocol_revision") != config.PROTOCOL_REVISION or protocol.get("status") != "locked" or protocol.get("classification") != "seen_fit_pilot":
        raise ProtocolError("protocol identity/status is invalid")
    if protocol.get("cohort", {}).get("sha256") != config.COHORT_SHA256 or protocol.get("cohort", {}).get("ordered_sample_id_sha256") != config.ORDERED_SAMPLE_ID_SHA256:
        raise ProtocolError("frozen cohort binding changed")
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("protocol record count is invalid")
    ids = [str(row.get("sample_id")) for row in records if isinstance(row, Mapping)]
    groups = [str(row.get("source_group")) for row in records if isinstance(row, Mapping)]
    if len(ids) != config.EXPECTED_RECORD_COUNT or len(set(ids)) != config.EXPECTED_RECORD_COUNT or len(set(groups)) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ProtocolError("protocol sample/source-group identity is invalid")
    if str(protocol.get("input_audit", {}).get("sha256")) != file_sha256(root / "input_audit.json"):
        raise ProtocolError("input audit hash changed")
    audit = _required_json(root / "input_audit.json")
    if audit.get("status") != "complete" or audit.get("passed_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("input audit is not complete")
    audio_manifest = _required_json(root / "audio_manifest.json")
    roi_manifest = _required_json(root / "roi/manifest.json")
    review = _required_json(root / "roi/review.json")
    if str(protocol.get("audio_manifest", {}).get("sha256")) != file_sha256(root / "audio_manifest.json") or str(protocol.get("roi_manifest", {}).get("sha256")) != file_sha256(root / "roi/manifest.json") or str(protocol.get("roi_review", {}).get("sha256")) != file_sha256(root / "roi/review.json"):
        raise ProtocolError("prepare manifest hash changed")
    _validate_audio(root, records, audio_manifest)
    _validate_roi(records, roi_manifest, review)
    for record in records:
        audio_row = record.get("audio", {}).get("manifest_row")
        if not isinstance(audio_row, Mapping) or audio_row.get("sample_id") != record.get("sample_id"):
            raise ProtocolError(f"protocol audio row identity changed: {record.get('sample_id')}")
        if str(record.get("audio", {}).get("manifest_sha256")) != file_sha256(root / "audio_manifest.json"):
            raise ProtocolError(f"protocol audio manifest binding changed: {record.get('sample_id')}")
        masks = record.get("masks")
        if not isinstance(masks, Mapping) or not masks.get("common_window_rows") or len(masks.get("plus_rows", [])) < config.MIN_LOCAL_ROWS or len(masks.get("minus_rows", [])) < config.MIN_LOCAL_ROWS:
            raise ProtocolError(f"timing masks are incomplete: {record.get('sample_id')}")
        counts = record.get("predicted_frame_counts")
        if not isinstance(counts, Mapping) or any(int(counts.get(arm, 0)) <= 0 for arm in (config.VIDEO_R, config.VIDEO_GN, config.VIDEO_GNR, config.VIDEO_GW, config.VIDEO_GB)):
            raise ProtocolError(f"predicted frame counts are incomplete: {record.get('sample_id')}")
        if len({int(counts[arm]) for arm in (config.VIDEO_GN, config.VIDEO_GNR, config.VIDEO_GW, config.VIDEO_GB)}) != 1:
            raise ProtocolError(f"generated frame counts differ: {record.get('sample_id')}")
    return protocol


def _validate_audio(root: Path, records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "complete" or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("audio manifest is incomplete")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("audio manifest row count is invalid")
    by_id = {str(row.get("sample_id")): row for row in rows if isinstance(row, Mapping)}
    if len(by_id) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("audio manifest sample IDs are not unique")
    for record in records:
        sample_id = str(record["sample_id"])
        row = by_id.get(sample_id)
        if not isinstance(row, Mapping) or str(row.get("natural_audio_sha256")) != str(record["natural_audio"]["sha256"]):
            raise ProtocolError(f"audio manifest source identity changed: {sample_id}")
        arms = row.get("arms")
        if not isinstance(arms, list) or {str(item.get("arm")) for item in arms if isinstance(item, Mapping)} != set(config.AUDIO_ARMS):
            raise ProtocolError(f"audio arm set is invalid: {sample_id}")
        decoded: dict[str, bytes] = {}
        for item in arms:
            if not isinstance(item, Mapping):
                raise ProtocolError(f"audio arm is malformed: {sample_id}")
            arm = str(item["arm"])
            path = Path(str(item.get("output", item.get("path", ""))))
            if not path.is_file() or str(item.get("output_sha256")) != file_sha256(path):
                raise ProtocolError(f"audio arm hash changed: {sample_id}/{arm}")
            values, meta = read_pcm16(path)
            if int(item.get("sample_count", -1)) != values.size or int(record["natural_sample_count"]) != values.size or meta["decoded_pcm_sha256"] != item.get("decoded_pcm_sha256"):
                raise ProtocolError(f"audio arm length/PCM identity changed: {sample_id}/{arm}")
            decoded[arm] = values.tobytes()
        if decoded[config.AUDIO_N] != decoded[config.AUDIO_N_REPEAT]:
            raise ProtocolError(f"N_REPEAT is not PCM-identical to N: {sample_id}")
        # Independent audit of the registered LOCAL_WARP_120 formula.
        natural = np.frombuffer(decoded[config.AUDIO_N], dtype="<i2")
        coordinates = np.arange(natural.size, dtype=np.float64)
        mapped = coordinates + 1920.0 * np.sin(2.0 * np.pi * coordinates / float(natural.size - 1))
        mapped[0] = 0.0
        mapped[-1] = float(natural.size - 1)
        expected_warp = np.rint(np.interp(mapped, coordinates, natural.astype(np.float64))).astype("<i2").tobytes()
        if decoded[config.AUDIO_W] != expected_warp:
            raise ProtocolError(f"W does not match registered LOCAL_WARP_120: {sample_id}")
        bridge_item = next(item for item in arms if str(item.get("arm")) == config.AUDIO_BRIDGE)
        construction = bridge_item.get("construction", {})
        if not isinstance(construction, Mapping) or construction.get("construction") != "registered_BRIDGE_075" or float(construction.get("alpha", -1)) != 0.75 or construction.get("phase_policy") != "natural_phase":
            raise ProtocolError(f"bridge construction identity changed: {sample_id}")


def _validate_roi(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], review: Mapping[str, Any]) -> None:
    if manifest.get("status") != "complete" or review.get("status") != "complete" or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT or review.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("ROI manifest/review is incomplete")
    rows = manifest.get("rows")
    reviews = review.get("rows")
    if not isinstance(rows, list) or len(rows) != config.EXPECTED_RECORD_COUNT or not isinstance(reviews, list) or len(reviews) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("ROI rows are incomplete")
    review_by_id = {str(row.get("sample_id")): row for row in reviews if isinstance(row, Mapping)}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("ROI row is malformed")
        sample_id = str(row["sample_id"])
        box_path = Path(str(row["boxes_path"]))
        box_payload = _required_json(box_path)
        if str(row.get("boxes_sha256")) != file_sha256(box_path) or box_payload.get("source_video_sha256") != row.get("source_video_sha256") or box_payload.get("box_order") != ["top", "bottom", "left", "right"]:
            raise ProtocolError(f"ROI source/order binding changed: {sample_id}")
        boxes = box_payload.get("boxes")
        raw = box_payload.get("raw_boxes")
        width, height, frame_count = int(row.get("width", 0)), int(row.get("height", 0)), int(row.get("frame_count", 0))
        if not isinstance(boxes, list) or len(boxes) != frame_count or not isinstance(raw, list) or len(raw) != frame_count:
            raise ProtocolError(f"ROI frame count changed: {sample_id}")
        for index, box in enumerate(boxes):
            if not isinstance(box, list) or len(box) != 4:
                raise ProtocolError(f"ROI box malformed: {sample_id}/{index}")
            top, bottom, left, right = (int(value) for value in box)
            if not (0 <= top < bottom <= height and 0 <= left < right <= width) or (top == 0 and bottom == height and left == 0 and right == width) or (bottom - top) * (right - left) >= 0.95 * width * height:
                raise ProtocolError(f"ROI box invalid/full-frame: {sample_id}/{index}")
        if str(row.get("box_track_sha256")) != canonical_json_sha256(boxes) or box_payload.get("box_track_sha256") != canonical_json_sha256(boxes):
            raise ProtocolError(f"ROI track hash changed: {sample_id}")
        review_row = review_by_id.get(sample_id)
        if not isinstance(review_row, Mapping) or review_row.get("decision") != "PASS" or review_row.get("reviewer") != "codex-automated-geometry-audit":
            raise ProtocolError(f"ROI geometry review failed: {sample_id}")
        evidence = review_row.get("evidence")
        if not isinstance(evidence, list) or {str(item.get("label")) for item in evidence if isinstance(item, Mapping)} != {"head", "middle", "tail"}:
            raise ProtocolError(f"ROI review evidence is incomplete: {sample_id}")
        for item in evidence:
            path = Path(str(item.get("path", "")))
            if not path.is_file() or str(item.get("sha256")) != file_sha256(path):
                raise ProtocolError(f"ROI review image changed: {sample_id}")


def _validate_videos(root: Path, protocol: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    path = root / "videos" / stage / "manifest.json"
    manifest = _required_json(path)
    expected_arms = (config.VIDEO_GN, config.VIDEO_GNR, config.VIDEO_GW) if stage == "control" else (config.VIDEO_GB,)
    expected_count = config.EXPECTED_CONTROL_VIDEO_COUNT if stage == "control" else config.EXPECTED_BRIDGE_VIDEO_COUNT
    if manifest.get("status") != "complete" or manifest.get("stage") != stage or manifest.get("video_count") != expected_count:
        raise ProtocolError(f"{stage} video manifest is incomplete")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError(f"{stage} video rows are incomplete")
    for record, row in zip(protocol["records"], rows, strict=True):
        if str(row.get("sample_id")) != str(record["sample_id"]):
            raise ProtocolError(f"{stage} video order changed")
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(expected_arms):
            raise ProtocolError(f"{stage} video arm set changed: {record['sample_id']}")
        for arm in expected_arms:
            item = arms[arm]
            sidecar = Path(str(item.get("output", "")).replace(".mkv", ".json"))
            if not sidecar.is_file() or str(item.get("output_sha256")) != file_sha256(Path(str(item["output"]))):
                raise ProtocolError(f"{stage} video hash changed: {record['sample_id']}/{arm}")
            side = _required_json(sidecar)
            if side.get("sample_id") != record["sample_id"] or side.get("arm") != arm or side.get("boxes_sha256") != record["roi"]["boxes_sha256"] or side.get("frame_count") != record["predicted_frame_counts"][arm]:
                raise ProtocolError(f"{stage} video identity/count changed: {record['sample_id']}/{arm}")
            if str(side.get("checkpoint_sha256")) != config.WAV2LIP_CHECKPOINT_SHA256:
                raise ProtocolError(f"{stage} checkpoint binding changed: {record['sample_id']}/{arm}")
    return manifest


def _media_cell_map(media_manifest: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in media_manifest.get("rows", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("cells"), Mapping):
            raise ProtocolError("media row is malformed")
        for cell in row["cells"].values():
            if not isinstance(cell, Mapping):
                raise ProtocolError("media cell is malformed")
            video, audio = str(cell.get("video_arm")), str(cell.get("audio_arm"))
            cell_key = (str(row.get("sample_id")), video, audio)
            if cell_key in result:
                raise ProtocolError(f"duplicate media cell: {cell_key}")
            result[cell_key] = cell
    return result


def _validate_media(root: Path, protocol: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    path = root / "media" / stage / "manifest.json"
    manifest = _required_json(path)
    expected_cells = config.EXPECTED_CONTROL_MAIN_CELL_COUNT if stage == "control" else config.EXPECTED_BRIDGE_CELL_COUNT
    if manifest.get("status") != "complete" or manifest.get("stage") != stage or manifest.get("cell_count") != expected_cells or manifest.get("audio_modified") is not False:
        raise ProtocolError(f"{stage} media manifest is incomplete")
    cells = _media_cell_map(manifest)
    if len(cells) != expected_cells:
        raise ProtocolError(f"{stage} media cell count differs: {len(cells)}")
    record_by_id = {str(row["sample_id"]): row for row in protocol["records"]}
    for (sample_id, video_arm, audio_arm), cell in cells.items():
        record = record_by_id.get(sample_id)
        if record is None:
            raise ProtocolError(f"media cell identity changed: {sample_id}/{video_arm}/{audio_arm}")
        output = Path(str(cell.get("output", "")))
        sidecar = Path(str(cell.get("sidecar", "")))
        if not output.is_file() or not sidecar.is_file() or str(cell.get("output_sha256")) != file_sha256(output):
            raise ProtocolError(f"media cell artifact is missing/changed: {sample_id}/{video_arm}/{audio_arm}")
        side = _required_json(sidecar)
        if side.get("protocol_id") != config.PROTOCOL_ID or side.get("stage") != stage or side.get("sample_id") != sample_id or side.get("video_arm") != video_arm or side.get("audio_arm") != audio_arm or side.get("audio_modified") is not False or "-shortest" in [str(item) for item in side.get("command", [])]:
            raise ProtocolError(f"media cell sidecar identity/policy changed: {sample_id}/{video_arm}/{audio_arm}")
        audio_manifest_row = record["audio"]["manifest_row"]
        audio_item = next(item for item in audio_manifest_row["arms"] if str(item.get("arm")) == audio_arm)
        expected_pcm = source_pcm16(Path(str(audio_item["output"])))
        if decode_pcm16(output) != expected_pcm or str(cell.get("audio_pcm_sha256")) != bytes_sha256(expected_pcm):
            raise ProtocolError(f"media cell PCM was swapped or modified: {sample_id}/{video_arm}/{audio_arm}")
        frames = decode_video_frames(output)
        expected_count = int(record["predicted_frame_counts"][video_arm])
        if len(frames) != expected_count or int(cell.get("frame_count", -1)) != expected_count:
            raise ProtocolError(f"media cell frame count changed: {sample_id}/{video_arm}/{audio_arm}")
    return manifest


def _validate_scores(root: Path, protocol: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    path = root / "scores" / stage / "manifest.json"
    manifest = _required_json(path)
    expected = config.EXPECTED_CONTROL_SCORE_COUNT if stage == "control" else config.EXPECTED_BRIDGE_CELL_COUNT
    if manifest.get("status") != "complete" or manifest.get("stage") != stage or manifest.get("score_count") != expected:
        raise ProtocolError(f"{stage} score manifest is incomplete")
    rows = manifest.get("scores")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ProtocolError(f"{stage} score rows are incomplete")
    media = _required_json(root / "media" / stage / "manifest.json")
    media_by_key = _media_cell_map(media)
    seen: set[tuple[str, str, str, bool]] = set()
    records = {str(row["sample_id"]): row for row in protocol["records"]}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("score row is malformed")
        key = (str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm")), bool(row.get("repeat", False)))
        if key in seen:
            raise ProtocolError(f"duplicate score row: {key}")
        seen.add(key)
        if key[3]:
            if stage != "control" or (key[1], key[2]) not in ((config.VIDEO_R, config.AUDIO_N), (config.VIDEO_GN, config.AUDIO_N)):
                raise ProtocolError(f"invalid repeat score: {key}")
        else:
            if (key[0], key[1], key[2]) not in media_by_key:
                raise ProtocolError(f"score/media pairing is missing: {key}")
        record = records.get(key[0])
        if record is None:
            raise ProtocolError(f"unknown score sample: {key}")
        score_path = root / "scores" / stage / ("repeat_cells" if key[3] else "cells") / f"{key[0]}__{config.cell_key(key[1], key[2], key[3])}" / "score.json"
        side = _required_json(score_path)
        if side.get("sample_id") != key[0] or side.get("video_arm") != key[1] or side.get("audio_arm") != key[2] or bool(side.get("repeat", False)) is not key[3] or str(side.get("matrix_sha256")) != file_sha256(Path(str(side["matrix"]))) or str(side.get("media_sha256")) != file_sha256(Path(str(side["media"]))):
            raise ProtocolError(f"score sidecar identity changed: {key}")
        matrix_path = Path(str(side["matrix"]))
        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
        record_audio = int(record["natural_sample_count"])
        expected_rows = min(int(record["predicted_frame_counts"][key[1]]), record_audio // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
        if matrix.shape != (expected_rows, 31) or not np.isfinite(matrix).all():
            raise ProtocolError(f"score matrix shape/finite contract changed: {key}")
        common_rows = [int(item) for item in record["masks"]["common_window_rows"]]
        if not common_rows or min(common_rows) < 0 or max(common_rows) >= matrix.shape[0]:
            raise ProtocolError(f"score common support changed: {key}")
        curve = matrix[common_rows, :].mean(axis=0)
        minimum_index = int(np.argmin(curve))
        reconstructed = side.get("common_global")
        if not isinstance(reconstructed, Mapping) or abs(float(reconstructed.get("sync_d", float("nan"))) - float(curve[minimum_index])) > 0.001 or int(reconstructed.get("offset", 10000)) != config.VSHIFT - minimum_index or abs(float(reconstructed.get("sync_c", float("nan"))) - float(np.median(curve) - curve[minimum_index])) > 0.001:
            raise ProtocolError(f"score common-window reconstruction changed: {key}")
    if len(seen) != expected:
        raise ProtocolError(f"score denominator shrank: {len(seen)} != {expected}")
    return manifest


def _validate_control_analysis(root: Path, protocol: Mapping[str, Any], final: Mapping[str, Any]) -> Mapping[str, Any]:
    control = _required_json(root / "control.json")
    if control.get("engineering_decision") != "GO" or control.get("record_count") != config.EXPECTED_RECORD_COUNT or control.get("score_count") != config.EXPECTED_CONTROL_SCORE_COUNT:
        raise ProtocolError("control analysis is incomplete")
    per = control.get("per_record")
    if not isinstance(per, list) or len(per) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("control analysis denominator shrank")
    groups = [str(row["source_group"]) for row in protocol["records"]]
    def check_bootstrap(values: Sequence[float], stored: Mapping[str, Any]) -> None:
        ci = bootstrap_ci(values, groups)
        if not _same_float_list(ci, stored.get("ci95", []), 1e-10):
            raise ProtocolError("control bootstrap changed")
    gates = control.get("gates")
    if not isinstance(gates, Mapping):
        raise ProtocolError("control gates are missing")
    generated = gates.get("generated_repeat")
    own = gates.get("own_audio")
    damage = gates.get("replacement_damage")
    if not isinstance(generated, Mapping) or not isinstance(own, Mapping) or not isinstance(damage, Mapping):
        raise ProtocolError("control gate payload is incomplete")
    check_bootstrap([float(row["generated_repeat"]["c_difference"]) for row in per], generated["c_difference"])
    check_bootstrap([float(row["generated_repeat"]["d_difference"]) for row in per], generated["d_difference_N_minus_NR"])
    check_bootstrap([float(row["own_audio"]["c"]) for row in per], own["c"])
    check_bootstrap([float(row["own_audio"]["d"]) for row in per], own["d"])
    check_bootstrap([float(row["replacement_damage"]["c"]) for row in per], damage["c"])
    check_bootstrap([float(row["replacement_damage"]["d"]) for row in per], damage["d"])
    counts = {
        "repeatability": sum(bool(row.get("repeatability", {}).get("passes")) for row in per),
        "r_baseline": sum(bool(row.get("baseline", {}).get("R_N", {}).get("passes")) for row in per),
        "gn_baseline": sum(bool(row.get("baseline", {}).get("G_N_N", {}).get("passes")) for row in per),
        "gnr_offset": sum(abs(int(row.get("generated_repeat", {}).get("offset_difference", 10000))) <= config.OFFSET_TOLERANCE_FRAMES for row in per),
        "own_offset": sum(bool(row.get("own_audio", {}).get("offset_noninferior")) for row in per),
        "damage_positive": sum(bool(row.get("replacement_damage", {}).get("both_positive")) for row in per),
    }
    if int(gates["repeatability"]["count"]) != counts["repeatability"] or int(gates["baseline"]["R_N_count"]) != counts["r_baseline"] or int(gates["baseline"]["G_N_N_count"]) != counts["gn_baseline"] or int(generated["offset_count"]) != counts["gnr_offset"] or int(own["offset_count"]) != counts["own_offset"] or int(damage["both_positive_count"]) != counts["damage_positive"]:
        raise ProtocolError("control gate count was altered")
    inherited = gates.get("inherited")
    if not isinstance(inherited, Mapping):
        raise ProtocolError("inherited gate is missing")
    for name in ("A", "B", "C", "O"):
        count = sum(bool(row.get("checks", {}).get(name, {}).get("passes")) for row in per)
        if int(inherited[name]["count"]) != count or int(inherited[name]["denominator"]) != config.EXPECTED_RECORD_COUNT:
            raise ProtocolError(f"control inherited count was altered: {name}")
    control_pass = all(bool(gate.get("passes")) for gate in gates.values())
    if bool(control.get("control_pass")) != control_pass or control.get("scientific_decision") != ("CONTROL_PASS" if control_pass else "CONTROL_FAILED"):
        raise ProtocolError("control decision does not match independently checked gates")
    return control


def _validate_bridge_analysis(root: Path, protocol: Mapping[str, Any], control: Mapping[str, Any]) -> Mapping[str, Any]:
    bridge = _required_json(root / "bridge.json")
    per = bridge.get("per_record")
    if bridge.get("engineering_decision") != "GO" or bridge.get("record_count") != config.EXPECTED_RECORD_COUNT or not isinstance(per, list) or len(per) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("bridge analysis is incomplete")
    groups = [str(row["source_group"]) for row in protocol["records"]]
    gates = bridge.get("gates")
    if not isinstance(gates, Mapping):
        raise ProtocolError("bridge gates are missing")
    for field, gate, stored_key in (("movement_progress", gates["movement"], "mean_ci"), ("gain_C", gates["natural_endpoint_noninferiority"], "gain_C"), ("gain_D", gates["natural_endpoint_noninferiority"], "gain_D"), ("gain_C", gates["positive_primary_gain"], "gain_C")):
        values = [float(row[field]) for row in per]
        stored = gate[stored_key]
        if not _same_float_list(bootstrap_ci(values, groups), stored.get("ci95", []), 1e-10):
            raise ProtocolError(f"bridge bootstrap changed: {field}")
    movement_count = sum(bool(row.get("movement_pass")) for row in per)
    offset_count = sum(bool(row.get("offset_noninferior")) for row in per)
    if int(gates["movement"]["count"]) != movement_count or int(gates["offset"]["count"]) != offset_count:
        raise ProtocolError("bridge count was altered")
    bridge_pass = all(bool(gate.get("passes")) for gate in gates.values())
    if bool(bridge.get("replacement_pass")) != bridge_pass or bridge.get("scientific_decision") != ("WAV2LIP_REPLACEMENT_PILOT_PASS" if bridge_pass else "REPLACEMENT_NOT_ESTABLISHED"):
        raise ProtocolError("bridge decision does not match independently checked gates")
    return bridge


def validate_run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    final_path = root / "final.json"
    final = _required_json(final_path)
    if final.get("status") == "blocked":
        result = _validate_preflight_block(root, final)
    else:
        if final.get("status") != "complete" or final.get("engineering_decision") != "GO" or str(final.get("protocol_id")) != config.PROTOCOL_ID:
            raise ProtocolError("complete final marker is malformed")
        protocol = _validate_protocol(root)
        if str(final.get("protocol_sha256")) != file_sha256(root / "protocol.json"):
            raise ProtocolError("final protocol hash changed")
        _validate_videos(root, protocol, "control")
        _validate_media(root, protocol, "control")
        _validate_scores(root, protocol, "control")
        control = _validate_control_analysis(root, protocol, final)
        if str(final.get("control_analysis_sha256")) != file_sha256(root / "control.json"):
            raise ProtocolError("final control analysis hash changed")
        scientific = str(final.get("scientific_decision"))
        if scientific not in ALLOWED_SCIENTIFIC:
            raise ProtocolError("final scientific terminal is invalid")
        counts = final.get("counts")
        expected_counts = final.get("expected_counts")
        if not isinstance(counts, Mapping) or not isinstance(expected_counts, Mapping) or counts != expected_counts:
            raise ProtocolError("final expected/actual count binding is invalid")
        expected_control_counts = {"control_generated_videos": config.EXPECTED_CONTROL_VIDEO_COUNT, "control_scores": config.EXPECTED_CONTROL_SCORE_COUNT}
        if any(int(counts.get(key, -1)) != value for key, value in expected_control_counts.items()):
            raise ProtocolError("final control counts are invalid")
        if bool(final.get("control_pass")) != bool(control.get("control_pass")) or (scientific == "CONTROL_FAILED") != (not bool(control.get("control_pass"))):
            raise ProtocolError("final control decision binding is invalid")
        if scientific == "CONTROL_FAILED":
            if final.get("bridge_status") != "NOT_RUN_CONTROL_FAILED":
                raise ProtocolError("control-failed final does not prove bridge was skipped")
            if int(counts.get("bridge_generated_videos", -1)) != 0 or int(counts.get("bridge_scores", -1)) != 0:
                raise ProtocolError("control-failed final has bridge counts")
            if any(((root / "videos/bridge/manifest.json").exists(), (root / "scores/bridge/manifest.json").exists())):
                raise ProtocolError("bridge artifacts exist after control failure")
        else:
            if not bool(control.get("control_pass")):
                raise ProtocolError("bridge terminal exists without control pass")
            _validate_videos(root, protocol, "bridge")
            _validate_media(root, protocol, "bridge")
            _validate_scores(root, protocol, "bridge")
            bridge = _validate_bridge_analysis(root, protocol, control)
            if scientific != bridge.get("scientific_decision"):
                raise ProtocolError("final scientific terminal differs from bridge analysis")
            if int(counts.get("bridge_generated_videos", -1)) != config.EXPECTED_BRIDGE_VIDEO_COUNT or int(counts.get("bridge_scores", -1)) != config.EXPECTED_BRIDGE_CELL_COUNT:
                raise ProtocolError("final bridge counts are invalid")
            if str(final.get("bridge_analysis_sha256")) != file_sha256(root / "bridge.json"):
                raise ProtocolError("final bridge analysis hash changed")
        if str(final.get("result_sha256")) != file_sha256(root / "result.md"):
            raise ProtocolError("final result hash changed")
        for key in ("training_authorized", "reference_conditioned_audio_head_spec_eligible", "generalization_established"):
            if final.get(key) is not False:
                raise ProtocolError(f"final unsafe flag is not false: {key}")
        result = {"status": "valid", "mode": "complete", "final_sha256": file_sha256(final_path), "scientific_decision": scientific}
    validation_path = root / "validation.json"
    payload = {"schema_version": 1, "stage_id": "validation", "protocol_id": config.PROTOCOL_ID, **result}
    write_self_hashed_json(validation_path, payload)
    return verify_self_hashed_json(validation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent validator for the Wav2Lip face-ROI replacement pilot")
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_run(args.run_root)
    except (ProtocolError, OSError, ValueError, KeyError, IndexError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
