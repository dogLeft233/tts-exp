from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config
from .common import ProtocolError, bytes_sha256, file_sha256, read_json, source_pcm16, write_json_atomic, write_self_hashed_json


def _hash_parent_assets() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, expected in config.PARENT_HASHES.items():
        path = {
            "cohort": config.COHORT,
            "natural_mels": config.NATURAL_MELS,
            "mask_manifest": config.MASK_MANIFEST,
            "reconstruction": config.RECONSTRUCTION,
            "drivers": config.DRIVERS,
            "box_manifest": config.BOX_MANIFEST,
        }[name]
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"parent asset hash changed: {name} {path}")
        result[name] = expected
    if not config.PARITY.is_file() or file_sha256(config.PARITY) != config.PARITY_HASH:
        raise ProtocolError("historical parity manifest hash changed")
    result["parity"] = config.PARITY_HASH
    return result


def _save_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(value), allow_pickle=False)
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "shape": list(value.shape), "dtype": str(value.dtype)}


def _read_boxes(path: Path) -> list[list[int]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid parent box file: {path}") from exc
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"parent box file is empty: {path}")
    result: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ProtocolError(f"parent box row is not xyxy: {path}")
        result.append([int(item) for item in row])
    return result


def _load_first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open parent face video: {path}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ProtocolError(f"parent face video has no valid first frame: {path}")
    return np.ascontiguousarray(frame)


def _mask_identity(mask: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(mask.get("mask_sha256")),
        int(mask.get("global_start_frame")),
        int(mask.get("global_end_frame")),
        int(mask.get("core_start")),
        int(mask.get("core_end")),
    )


def _build_shuffle(correct: np.ndarray, sample_id: str, columns: list[int]) -> tuple[np.ndarray, list[int]]:
    if len(columns) < 2:
        raise ProtocolError(f"K is too small for deterministic shuffle: {sample_id}")
    digest = hashlib.sha256(config.SHUFFLE_SALT + sample_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(columns)).astype(np.int64)
    if np.array_equal(permutation, np.arange(len(columns), dtype=np.int64)):
        permutation[[0, 1]] = permutation[[1, 0]]
    shuffled = np.zeros_like(correct, dtype=np.float64)
    index = np.asarray(columns, dtype=np.int64)
    shuffled[:, index] = correct[:, index][:, permutation]
    return shuffled, [int(item) for item in permutation]


def construct_arms(sample_id: str, natural: np.ndarray, parent_arrays: Mapping[str, np.ndarray], columns: list[int]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paired = np.asarray(parent_arrays["PAIRED_TTS"], dtype=np.float64)
    wrong = np.asarray(parent_arrays["SAME_PHONE_WRONG_INSTANCE"], dtype=np.float64)
    zero = np.asarray(parent_arrays["NAT_ONLY"], dtype=np.float64)
    correct_residual = np.mean(np.stack([paired - zero], axis=0), axis=0)
    wrong_residual = np.mean(np.stack([wrong - zero], axis=0), axis=0)
    correct_norm = float(np.linalg.norm(correct_residual[:, columns]))
    wrong_norm = float(np.linalg.norm(wrong_residual[:, columns]))
    if correct_norm <= 1e-12 or wrong_norm <= 1e-12:
        raise ProtocolError(f"INPUT_DEGENERATE: zero residual norm for {sample_id}")
    wrong_residual = wrong_residual * (correct_norm / wrong_norm)
    shuffled, permutation = _build_shuffle(correct_residual, sample_id, columns)
    maximum = float(max(np.max(np.abs(correct_residual)), np.max(np.abs(wrong_residual)), np.max(np.abs(shuffled))))
    if not np.isfinite(maximum) or maximum <= 1e-12:
        raise ProtocolError(f"INPUT_DEGENERATE: zero residual amplitude for {sample_id}")
    amplitude = min(0.25, 0.5 / maximum)
    natural64 = np.asarray(natural, dtype=np.float64)
    arms: dict[str, np.ndarray] = {"N": np.asarray(natural, dtype=np.float32).copy()}
    for name, residual in (("CORRECT", correct_residual), ("WRONG", wrong_residual), ("SHUFFLE", shuffled)):
        value = natural64.copy()
        index = np.asarray(columns, dtype=np.int64)
        value[:, index] = np.clip(natural64[:, index] + amplitude * residual[:, index], -4.0, 4.0)
        arms[name] = np.asarray(value, dtype=np.float32)
    norm_rows: dict[str, Any] = {}
    c_norm_post = float(np.linalg.norm((arms["CORRECT"].astype(np.float64) - natural64)[:, columns]))
    for name in ("CORRECT", "WRONG", "SHUFFLE"):
        post = float(np.linalg.norm((arms[name].astype(np.float64) - natural64)[:, columns]))
        norm_rows[name] = {"post_l2": post, "ratio_to_correct": post / c_norm_post if c_norm_post > 1e-12 else None}
    norm_valid = bool(
        c_norm_post > 1e-12
        and all(0.95 <= float(norm_rows[name]["ratio_to_correct"]) <= 1.05 for name in ("WRONG", "SHUFFLE"))
    )
    changed = {name: float(np.mean(np.abs(arms[name].astype(np.float64) - natural64) > 0.0)) for name in ("CORRECT", "WRONG", "SHUFFLE")}
    metadata = {
        "sample_id": sample_id,
        "K": columns,
        "K_count": len(columns),
        "shuffle_permutation": permutation,
        "correct_residual_l2": correct_norm,
        "wrong_residual_l2_before_match": wrong_norm,
        "wrong_residual_l2_after_match": float(np.linalg.norm(wrong_residual[:, columns])),
        "amplitude": float(amplitude),
        "preclip_max_abs": maximum,
        "norm_control": norm_rows,
        "norm_control_valid": norm_valid,
        "changed_fraction": changed,
        "clip_fraction": {name: float(np.mean(np.abs(natural64[:, columns] + amplitude * residual[:, columns]) > 4.0)) for name, residual in (("CORRECT", correct_residual), ("WRONG", wrong_residual), ("SHUFFLE", shuffled))},
    }
    return arms, metadata


def prepare(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_hashes = _hash_parent_assets()
    cohort = read_json(config.COHORT)
    drivers_manifest = read_json(config.DRIVERS)
    reconstruction = read_json(config.RECONSTRUCTION)
    box_manifest = read_json(config.BOX_MANIFEST)
    records = cohort.get("records")
    driver_rows = drivers_manifest.get("drivers")
    recon_rows = reconstruction.get("records")
    box_rows = box_manifest.get("renders")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("parent cohort record count is not 16")
    if not isinstance(driver_rows, list) or len(driver_rows) != 192:
        raise ProtocolError("parent driver count is not 192")
    if not isinstance(recon_rows, list) or not isinstance(box_rows, list):
        raise ProtocolError("parent reconstruction or box rows are missing")
    records = sorted(records, key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    if len({str(row["source_group"]) for row in records}) != config.EXPECTED_GROUP_COUNT:
        raise ProtocolError("parent source-group count is not 8")
    driver_index: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in driver_rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("malformed parent driver row")
        key = (str(row["sample_id"]), int(row["seed"]), str(row["condition"]))
        if key in driver_index:
            raise ProtocolError(f"duplicate parent driver key: {key}")
        driver_index[key] = row
    box_index = {str(row["sample_id"]): row for row in box_rows if isinstance(row, Mapping)}
    reconstruction_index = {(str(row["sample_id"]), int(row["seed"]), str(row["condition"])): row for row in recon_rows if isinstance(row, Mapping)}
    run_drivers = root / "drivers"
    static_dir = root / "static_faces"
    protocol_records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    with np.load(config.NATURAL_MELS, allow_pickle=False) as natural_archive:
        for record in records:
            sample_id = str(record["sample_id"])
            source_group = str(record["source_group"])
            if sample_id not in natural_archive.files:
                raise ProtocolError(f"natural mel is missing: {sample_id}")
            natural = np.asarray(natural_archive[sample_id], dtype=np.float32)
            if natural.shape != (config.MEL_BINS, config.MEL_FRAMES) or not np.isfinite(natural).all():
                raise ProtocolError(f"natural mel shape/finite check failed: {sample_id}")
            face_video = Path(str(record["face"]))
            natural_audio = Path(str(record["natural_audio"]))
            if not face_video.is_file() or file_sha256(face_video) != str(record["face_sha256"]):
                raise ProtocolError(f"face binding changed: {sample_id}")
            if not natural_audio.is_file() or file_sha256(natural_audio) != str(record["natural_audio_sha256"]):
                raise ProtocolError(f"natural audio binding changed: {sample_id}")
            pcm = source_pcm16(natural_audio)
            if len(pcm) // 2 < config.SUPPORT_SAMPLES:
                raise ProtocolError(f"natural audio is shorter than support: {sample_id}")
            box_row = box_index.get(sample_id)
            if not isinstance(box_row, Mapping):
                raise ProtocolError(f"box manifest row is missing: {sample_id}")
            box_path = Path(str(box_row["boxes"]))
            boxes = _read_boxes(box_path)
            frame = _load_first_frame(face_video)
            x1, y1, x2, y2 = boxes[0]
            if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                raise ProtocolError(f"parent xyxy box is outside first frame: {sample_id}")
            crop = frame[y1:y2, x1:x2]
            static_face = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            static_path = static_dir / f"{sample_id}.npy"
            static_meta = _save_npy(static_path, np.asarray(static_face, dtype=np.uint8))
            mask_identities: list[tuple[Any, ...]] | None = None
            union: set[int] = set()
            parent_arrays: dict[str, dict[int, np.ndarray]] = {condition: {} for condition in config.DRIVER_CONDITIONS}
            driver_bindings: dict[str, dict[str, Any]] = {}
            for seed in config.SEEDS:
                for condition in config.DRIVER_CONDITIONS:
                    key = (sample_id, seed, condition)
                    row = driver_index.get(key)
                    if not isinstance(row, Mapping):
                        raise ProtocolError(f"parent driver key is missing: {key}")
                    path = Path(str(row["path"]))
                    if not path.is_file() or file_sha256(path) != str(row["sha256"]):
                        raise ProtocolError(f"parent driver hash changed: {key}")
                    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
                    if array.shape != (config.MEL_BINS, config.MEL_FRAMES) or not np.isfinite(array).all():
                        raise ProtocolError(f"parent driver shape/finite check failed: {key}")
                    identities = [_mask_identity(mask) for mask in row.get("used_masks", [])]
                    if mask_identities is None:
                        mask_identities = identities
                    elif identities != mask_identities:
                        raise ProtocolError(f"mask identity/location differs across seeds or conditions: {sample_id}")
                    for mask in row.get("used_masks", []):
                        union.update(range(int(mask["global_start_frame"]), int(mask["global_end_frame"])))
                    parent_arrays[condition][seed] = array
                    recon = reconstruction_index.get(key)
                    if not isinstance(recon, Mapping) or str(recon.get("checkpoint_sha256")) not in {"42f8b8ac7595219c9336e427311747dd71bf589f14e42cfc7a0cfb0d1ebb7def", "1aac2221d5b586a9e66896c3e4e4f38f08275a08f34f001723d826137c824d96", "fc9dd8199494dd8dabbc08172ace7d1849a1849ec523747242470a208ddce920"}:
                        raise ProtocolError(f"checkpoint provenance is missing: {key}")
                    driver_bindings[f"{condition}__{seed}"] = {"path": str(path.resolve()), "sha256": str(row["sha256"]), "checkpoint_sha256": str(recon["checkpoint_sha256"])}
            columns = sorted(union)
            if not columns or min(columns) < 0 or max(columns) >= config.MEL_FRAMES:
                raise ProtocolError(f"mask union is outside mel support: {sample_id}")
            averaged = {condition: np.mean(np.stack([parent_arrays[condition][seed] for seed in config.SEEDS], axis=0), axis=0, dtype=np.float64) for condition in config.DRIVER_CONDITIONS}
            for condition, value in averaged.items():
                if not np.array_equal(value[:, [index for index in range(config.MEL_FRAMES) if index not in set(columns)]], natural[:, [index for index in range(config.MEL_FRAMES) if index not in set(columns)] ]):
                    raise ProtocolError(f"parent driver leaks outside K: {sample_id}/{condition}")
            parent_for_construct = {condition: averaged[condition] for condition in config.DRIVER_CONDITIONS}
            arms, construct_meta = construct_arms(sample_id, natural, parent_for_construct, columns)
            arm_rows: dict[str, Any] = {}
            for arm, value in arms.items():
                arm_rows[arm] = _save_npy(run_drivers / f"{sample_id}__{arm}.npy", value)
            manifest_rows.append({"sample_id": sample_id, "source_group": source_group, "natural_audio": {"path": str(natural_audio.resolve()), "sha256": str(record["natural_audio_sha256"]), "pcm_sha256": bytes_sha256(pcm), "sample_count": len(pcm) // 2}, "face_video": {"path": str(face_video.resolve()), "sha256": str(record["face_sha256"])}, "box": {"path": str(box_path.resolve()), "sha256": file_sha256(box_path), "xyxy": [x1, y1, x2, y2], "frame_shape": list(frame.shape)}, "static_face": static_meta, "arms": arm_rows, "construct": construct_meta, "parent_drivers": driver_bindings})
            protocol_records.append({"sample_id": sample_id, "source_group": source_group, "natural_audio": manifest_rows[-1]["natural_audio"], "face_video": manifest_rows[-1]["face_video"], "box": manifest_rows[-1]["box"], "static_face": static_meta, "K": columns, "predicted_frame_count": config.WAV2LIP_FRAMES, "matrix_rows": config.MATRIX_ROWS, "U": list(range(config.U_START, config.U_STOP))})
            audit_rows.append({"sample_id": sample_id, "source_group": source_group, "passed": True, "audio_support_samples": len(pcm) // 2, "mel_shape": list(natural.shape), "K_count": len(columns), "static_face_sha256": static_meta["sha256"]})
    audit = {"schema_version": 1, "status": "complete", "record_count": len(audit_rows), "passed_count": len(audit_rows), "parent_hashes": parent_hashes, "records": audit_rows}
    protocol = {"schema_version": 1, "protocol_id": "wav2lip_natural_content_residual", "protocol_revision": "v1", "status": "locked", "scope": "seen-record short-support exploratory probe", "parent": {"root": str(config.PARENT.resolve()), "hashes": parent_hashes}, "runtime": {"wav2lip_python": str(config.WAV2LIP_PYTHON), "syncnet_python": str(config.SYNCNET_PYTHON), "wav2lip_checkpoint": str(config.WAV2LIP_CHECKPOINT.resolve()), "wav2lip_checkpoint_sha256": config.WAV2LIP_CHECKPOINT_SHA256, "syncnet_model": str(config.SYNCNET_MODEL.resolve()), "syncnet_model_sha256": config.SYNCNET_MODEL_SHA256}, "records": protocol_records, "record_count": len(protocol_records), "source_group_count": len({str(row["source_group"]) for row in protocol_records}), "spec_bindings": {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in config.spec_paths().items()}, "limits": {"support_samples": config.SUPPORT_SAMPLES, "support_seconds": config.SUPPORT_SAMPLES / config.SAMPLE_RATE, "video_count_max": config.MAX_VIDEO_COUNT, "score_count_max": config.MAX_SCORE_COUNT, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}}
    driver_manifest = {"schema_version": 1, "status": "complete", "protocol_id": protocol["protocol_id"], "record_count": len(manifest_rows), "source_group_count": len({str(row["source_group"]) for row in manifest_rows}), "rows": manifest_rows}
    return audit, protocol, driver_manifest


def write_prepared(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit, protocol, driver_manifest = prepare(root)
    write_self_hashed_json(root / "input_audit.json", audit)
    write_self_hashed_json(root / "protocol.json", protocol)
    write_self_hashed_json(root / "drivers/manifest.json", driver_manifest)
    static_manifest = {"schema_version": 1, "status": "complete", "protocol_id": protocol["protocol_id"], "rows": [{"sample_id": row["sample_id"], "path": row["static_face"]["path"], "sha256": row["static_face"]["sha256"], "source_face": row["face_video"], "box": row["box"]} for row in driver_manifest["rows"]]}
    write_self_hashed_json(root / "static_faces/manifest.json", static_manifest)
    return audit, protocol, driver_manifest
