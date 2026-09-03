"""Hashing, parent validation, and score-independent cohort locking."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.io import wavfile

from ..mfa_linear_real_video_sync.config import P1_MFA_SUMMARY_SHA256
from ..mfa_linear_real_video_sync.model import build_waveform_model, module_state_sha256
from ..mfa_linear_real_video_sync.protocol import (
    calibrate_target_offset,
    extract_official_bgr_frames,
    frame_hashes,
    load_mfa_linear_waveform,
    load_pcm16_waveform,
    materialize_ffv1_once,
    read_fixed_video_frames,
    validate_target_offset_artifact,
    write_json_once,
)
from .config import (
    BLOCKED_COHORT,
    BLOCKED_P2,
    COHORT_SIZE,
    P2_RECORD_ORDER,
    P2_SOURCE_GROUPS,
    P2_STAGE,
    P2_STATUS,
    P2_STEPS,
    SCHEMA_VERSION,
    SELECTION_SALT,
    SYNCNET_CHECKPOINT_SHA256,
    TARGET_GAP,
    TransferConfig,
)
from ..mfa_linear_real_video_sync.syncnet_loss import (
    audio_embeddings,
    cached_visual_embeddings,
    official_syncnet_distance_curve,
)


class TransferProtocolError(ValueError):
    pass


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once_checked(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    write_json_once(path, payload)
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def read_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransferProtocolError(f"missing or corrupt artifact: {target}") from error
    if not isinstance(payload, dict):
        raise TransferProtocolError(f"artifact must be an object: {target}")
    return payload


def _hash_model_checkpoint(path: Path) -> str:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise TransferProtocolError(f"cannot load model checkpoint: {path}") from error
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    recorded = payload.get("model_sha256") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or not isinstance(recorded, str):
        raise TransferProtocolError(f"checkpoint lacks state_dict/model_sha256: {path}")
    model = build_waveform_model(seed=20_260_903, device="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise TransferProtocolError(f"checkpoint state shape mismatch: {path}") from error
    actual = module_state_sha256(model)
    if actual != recorded:
        raise TransferProtocolError(f"checkpoint internal hash mismatch: {path}")
    return actual


def validate_p2_parent(parent_root: str | Path, *, repo: str | Path | None = None) -> dict[str, Any]:
    root = Path(parent_root).resolve()
    if repo is not None and root == Path(repo).resolve() / "runs/lrs3_mfa_linear_real_video_sync_20260903_v4":
        raise TransferProtocolError(f"{BLOCKED_P2}: P1 v4 is not a P2 parent")
    try:
        decision = read_object(root / "decision.json")
        validation = read_object(root / "validation.json")
        p2_decision = read_object(root / "03_p2_shared_four/decision.json")
        history = read_object(root / "03_p2_shared_four/history.json")
        manifest = read_object(root / "00_lock/p2_manifest.json")
        lock = read_object(root / "00_lock/input_lock.json")
    except TransferProtocolError as error:
        raise TransferProtocolError(f"{BLOCKED_P2}: {error}") from error
    if decision.get("status") != P2_STATUS or decision.get("pass") is not True:
        raise TransferProtocolError(f"{BLOCKED_P2}: terminal P2 decision is not passing")
    if p2_decision.get("status") != P2_STATUS or p2_decision.get("pass") is not True:
        raise TransferProtocolError(f"{BLOCKED_P2}: stage P2 decision is not passing")
    if validation.get("artifact_graph_valid") is not True or validation.get("status") != "valid":
        raise TransferProtocolError(f"{BLOCKED_P2}: parent validation is not valid")
    if history.get("status") != "complete" or history.get("stage") != P2_STAGE or history.get("steps") != P2_STEPS:
        raise TransferProtocolError(f"{BLOCKED_P2}: history is not exactly fresh 100-step P2")
    if history.get("record_order") != list(P2_RECORD_ORDER):
        raise TransferProtocolError(f"{BLOCKED_P2}: P2 record order changed")
    if manifest.get("record_order") != list(P2_RECORD_ORDER) or manifest.get("source_groups") != list(P2_SOURCE_GROUPS):
        raise TransferProtocolError(f"{BLOCKED_P2}: P2 manifest bindings changed")
    if manifest.get("protocol_split") != "train" or manifest.get("natural_in_training_records") is not False:
        raise TransferProtocolError(f"{BLOCKED_P2}: P2 manifest policy changed")
    expected_config = TransferConfig().to_dict()["predecessor_config"]
    if history.get("config") != expected_config:
        raise TransferProtocolError(f"{BLOCKED_P2}: predecessor configuration changed")
    step0 = root / "03_p2_shared_four/step0/model.pt"
    step100 = root / "03_p2_shared_four/step100/model.pt"
    initial_hash = _hash_model_checkpoint(step0)
    final_hash = _hash_model_checkpoint(step100)
    fresh = build_waveform_model(seed=20_260_903, device="cpu")
    fresh_hash = module_state_sha256(fresh)
    if initial_hash != fresh_hash or history.get("initial_model_sha256") != initial_hash:
        raise TransferProtocolError(f"{BLOCKED_P2}: P2 was not fresh initialized")
    if history.get("final_model_sha256") != final_hash or p2_decision.get("step100_model_sha256") != final_hash:
        raise TransferProtocolError(f"{BLOCKED_P2}: step-100 checkpoint hash mismatch")
    before = history.get("syncnet_sha256_before")
    after = history.get("syncnet_sha256_after")
    if not isinstance(before, str) or before != after:
        raise TransferProtocolError(f"{BLOCKED_P2}: frozen SyncNet state changed")
    if lock.get("syncnet_checkpoint_sha256") != SYNCNET_CHECKPOINT_SHA256:
        raise TransferProtocolError(f"{BLOCKED_P2}: SyncNet checkpoint binding changed")
    if validation.get("decision_sha256") != sha256_file(root / "decision.json"):
        raise TransferProtocolError(f"{BLOCKED_P2}: terminal decision hash is stale")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "parent_root": str(root),
        "parent_root_sha256": sha256_bytes(str(root).encode()),
        "terminal_status": decision["status"],
        "validation_sha256": sha256_file(root / "validation.json"),
        "decision_sha256": sha256_file(root / "decision.json"),
        "p2_decision_sha256": sha256_file(root / "03_p2_shared_four/decision.json"),
        "history_sha256": sha256_file(root / "03_p2_shared_four/history.json"),
        "step0_checkpoint": str(step0),
        "step0_model_sha256": initial_hash,
        "step100_checkpoint": str(step100),
        "step100_model_sha256": final_hash,
        "record_order": list(P2_RECORD_ORDER),
        "source_groups": list(P2_SOURCE_GROUPS),
        "syncnet_state_sha256": before,
        "syncnet_checkpoint_sha256": SYNCNET_CHECKPOINT_SHA256,
        "config": expected_config,
    }


class FrozenTTSOnlyAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor) or tuple(waveform.shape) != (1, 1, 61_440):
            raise TransferProtocolError("NATURAL_OR_SIDE_CHANNEL_LEAKAGE: adapter accepts only [1,1,61440] waveform")
        if not torch.is_floating_point(waveform):
            raise TransferProtocolError("adapter waveform must be floating point")
        return self._model(waveform)

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, torch.Tensor]:
        return self._model.state_dict(*args, **kwargs)


def load_adapter(parent_lock: Mapping[str, Any], *, device: str | torch.device) -> torch.nn.Module:
    checkpoint = Path(str(parent_lock["step100_checkpoint"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_waveform_model(seed=20_260_903, device=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = FrozenTTSOnlyAdapter(model)
    if module_state_sha256(adapter) != parent_lock["step100_model_sha256"]:
        raise TransferProtocolError(f"{BLOCKED_P2}: loaded adapter hash mismatch")
    return adapter


def selection_key(source_group: str, sample_id: str) -> str:
    return sha256_bytes(f"{SELECTION_SALT}\0{source_group}\0{sample_id}".encode("utf-8"))


def _policy_record(path: Path) -> dict[str, Any]:
    outer = read_object(path)
    policy = outer.get("policy")
    if not isinstance(policy, dict):
        raise TransferProtocolError("policy object missing")
    policy = dict(policy)
    crop = outer.get("crop", {})
    if isinstance(crop, Mapping):
        policy["_track_path"] = crop.get("track_path")
    return policy


def _structural_row(policy_path: Path, policy: Mapping[str, Any], mfa_summary: Mapping[str, Any], fit_groups: set[str]) -> dict[str, Any]:
    sid = str(policy.get("sample_id"))
    group = str(policy.get("source_group"))
    if policy.get("protocol_split") != "train" or group in fit_groups:
        raise TransferProtocolError("non-train or P2 source group")
    source = policy.get("source", {})
    legacy = policy.get("legacy_crop", {})
    geometry = policy.get("syncnet", {}).get("reference_geometry", {})
    required = (source, legacy, geometry, policy.get("natural_audio", {}))
    if not all(isinstance(value, Mapping) for value in required):
        raise TransferProtocolError("policy geometry/audio metadata missing")
    result = mfa_summary.get("results", {}).get(sid)
    if not isinstance(result, Mapping) or result.get("split") != "train":
        raise TransferProtocolError("MFA row missing or not train")
    mfa_path = Path(str(result.get("audio_path", ""))).resolve()
    mfa_hash = str(result.get("audio_sha256", ""))
    waveform, mfa_lock = load_mfa_linear_waveform(mfa_path, expected_sha256=mfa_hash)
    del waveform
    source_path = Path(str(source.get("path", ""))).resolve()
    crop_path = Path(str(legacy.get("path", ""))).resolve()
    natural_path = Path(str(policy["natural_audio"].get("path", ""))).resolve()
    track_path = Path(str(policy.get("_track_path", ""))).resolve()
    crop_audio_path = Path(str(legacy.get("audio_path", ""))).resolve()
    for path, expected in (
        (source_path, source.get("sha256")), (crop_path, legacy.get("sha256")),
        (crop_audio_path, legacy.get("audio_sha256")),
        (natural_path, policy["natural_audio"].get("sha256")), (track_path, geometry.get("track_sha256")),
    ):
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise TransferProtocolError(f"hash verification failed: {path}")
    if source.get("fps") != 25.0 or source.get("width") != 224 or source.get("height") != 224:
        raise TransferProtocolError("source geometry mismatch")
    crop_meta = legacy.get("crop_metadata", {})
    if any(crop_meta.get(key) != value for key, value in (("fps", 25.0), ("width", 224), ("height", 224))):
        raise TransferProtocolError("crop geometry mismatch")
    if int(crop_meta.get("frame_count", 0)) < 96 or geometry.get("frame_count", 0) < 96:
        raise TransferProtocolError("insufficient frame support")
    if geometry.get("crop_scale") != 0.4 or geometry.get("padding_value") != 110 or geometry.get("output_size") != 224:
        raise TransferProtocolError("tracked geometry parameters changed")
    read_fixed_video_frames(crop_path)
    natural_audio, natural_lock = load_pcm16_waveform(
        natural_path, expected_sha256=str(policy["natural_audio"]["sha256"]), sample_count=61_440
    )
    return {
        "sample_id": sid,
        "source_group": group,
        "protocol_split": "train",
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "source_video": str(source_path),
        "source_video_sha256": str(source["sha256"]),
        "tracked_crop": str(crop_path),
        "tracked_crop_sha256": str(legacy["sha256"]),
        "crop_audio": str(Path(str(legacy["audio_path"])).resolve()),
        "crop_audio_sha256": str(legacy["audio_sha256"]),
        "natural_audio": str(natural_path),
        "natural_audio_sha256": str(policy["natural_audio"]["sha256"]),
        "track_path": str(track_path),
        "track_sha256": str(geometry["track_sha256"]),
        "mfa_audio": str(mfa_path),
        "mfa_audio_sha256": mfa_hash,
        "mfa_segment_float32_sha256": mfa_lock["segment_float32_sha256"],
        "mfa_step0_pcm16_sha256": mfa_lock["step0_rounded_pcm16_sha256"],
        "natural_segment_pcm16_sha256": natural_lock["segment_pcm_sha256"],
        "selection_key": selection_key(group, sid),
        "natural_audio_values": natural_audio,
    }


def scan_structural_eligibility(
    policy_records: str | Path,
    mfa_summary_path: str | Path,
    *,
    fit_groups: Iterable[str] = P2_SOURCE_GROUPS,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = Path(policy_records).resolve()
    summary_path = Path(mfa_summary_path).resolve()
    if sha256_file(summary_path) != P1_MFA_SUMMARY_SHA256:
        raise TransferProtocolError("MFA summary hash changed")
    summary = read_object(summary_path)
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    seen_samples: set[str] = set()
    for path in sorted(root.glob("*.json")):
        try:
            policy = _policy_record(path)
            sid = str(policy.get("sample_id"))
            group = str(policy.get("source_group"))
            if sid in seen_samples:
                raise TransferProtocolError("duplicate sample identity")
            row = _structural_row(path, policy, summary, set(fit_groups))
            seen_samples.add(sid)
            eligible.append(row)
        except (TransferProtocolError, FileNotFoundError, ValueError, OSError) as error:
            exclusions.append({"path": str(path.resolve()), "reason": str(error)})
    if not eligible:
        raise TransferProtocolError(f"{BLOCKED_COHORT}: no structurally eligible records")
    return eligible, exclusions


def calibrate_and_materialize_cohort(
    rows: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    syncnet: torch.nn.Module,
    output_root: str | Path,
    *,
    device: str | torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = Path(output_root).resolve()
    calibrated: list[dict[str, Any]] = []
    failures = list(exclusions)
    for source in rows:
        row = dict(source)
        try:
            frames = read_fixed_video_frames(row["tracked_crop"])
            visual = materialize_ffv1_once(output / "01_cohort_lock/visual" / f"{row['sample_id']}.avi", frames)
            scoring = extract_official_bgr_frames(
                visual["path"], output / "01_cohort_lock/scoring_frames" / row["sample_id"]
            )
            visual["scoring_bgr_frame_sha256"] = frame_hashes(scoring)
            visual_embedding = cached_visual_embeddings(syncnet, torch.from_numpy(scoring).to(device))
            natural = torch.from_numpy(np.asarray(row.pop("natural_audio_values"), dtype=np.float32)).to(device)[None, None]
            with torch.no_grad():
                curve = official_syncnet_distance_curve(audio_embeddings(syncnet, natural), visual_embedding).cpu()
            target = calibrate_target_offset(curve)
            validate_target_offset_artifact(target)
            row["visual"] = visual
            row["natural_target"] = target
            row["visual_embedding_shape"] = list(visual_embedding.shape)
            row["mfa_summary_sha256"] = P1_MFA_SUMMARY_SHA256
            calibrated.append(row)
            del natural, visual_embedding
        except (TransferProtocolError, FileNotFoundError, ValueError, OSError, RuntimeError) as error:
            failures.append({"sample_id": row.get("sample_id"), "source_group": row.get("source_group"), "reason": str(error)})
    return calibrated, {"exclusions": failures, "calibration_count": len(calibrated)}


def select_cohort(
    calibrated_rows: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    *,
    fit_groups: Iterable[str] = P2_SOURCE_GROUPS,
) -> dict[str, Any]:
    fit = set(fit_groups)
    if any(str(row.get("source_group")) in fit for row in calibrated_rows):
        raise TransferProtocolError(f"{BLOCKED_COHORT}: P2 group leaked into cohort")
    reps: dict[str, Mapping[str, Any]] = {}
    for row in calibrated_rows:
        group = str(row["source_group"])
        key = (str(row["selection_key"]), group.encode(), str(row["sample_id"]).encode())
        old = reps.get(group)
        old_key = None if old is None else (str(old["selection_key"]), group.encode(), str(old["sample_id"]).encode())
        if old is None or key < old_key:
            reps[group] = row
    ordered = sorted(
        reps.values(),
        key=lambda row: (bytes.fromhex(str(row["selection_key"])), str(row["source_group"]).encode(), str(row["sample_id"]).encode()),
    )
    if len(ordered) < COHORT_SIZE:
        raise TransferProtocolError(f"{BLOCKED_COHORT}: only {len(ordered)} eligible source groups")
    selected = [dict(row) for row in ordered[:COHORT_SIZE]]
    for row in selected:
        row.pop("natural_audio_values", None)

    def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result.pop("natural_audio_values", None)
        return result

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "manifest_type": "adapter_heldout_source_group_cohort",
        "selection_salt": SELECTION_SALT,
        "selection_uses_outcomes": False,
        "heldout_definition": "adapter-heldout within fit-only LRS3 universe",
        "p2_source_groups": sorted(fit),
        "eligible_universe": [public_row(row) for row in calibrated_rows],
        "exclusions": [dict(item) for item in exclusions],
        "group_representatives": [public_row(row) for row in ordered],
        "selected": selected,
        "selected_sample_ids": [str(row["sample_id"]) for row in selected],
        "selected_source_groups": [str(row["source_group"]) for row in selected],
        "denominator": COHORT_SIZE,
    }
    write_json_once(output_root / "01_cohort_lock/manifest.json", manifest)
    return manifest
