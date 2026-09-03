"""Hash-locked P1 data construction; natural audio is discarded on return."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import (
    AssetRoots,
    P1_CROP_96_CONCAT_SHA256,
    P1_CROP_AUDIO_SHA256,
    P1_CROP_FRAME0_SHA256,
    P1_CROP_FRAME95_SHA256,
    P1_LEGACY_CROP_SHA256,
    P1_MFA_PARENT_SHA256,
    P1_MFA_SUMMARY_SHA256,
    P1_NATURAL_AUDIO_SHA256,
    P1_POLICY_RECORD_SHA256,
    P1_SAMPLE_ID,
    P1_SOURCE_VIDEO_SHA256,
    P1_STEP0_PCM16_SHA256,
    P1_TRACK_SHA256,
    P2_ASSET_LOCKS,
    P2_RECORDS,
    Q,
    SEGMENT_FRAMES,
    SEGMENT_SAMPLES,
)
from .model import load_frozen_syncnet, module_state_sha256
from .protocol import (
    ProtocolError,
    calibrate_target_offset,
    frame_hashes,
    extract_official_bgr_frames,
    load_mfa_linear_waveform,
    load_pcm16_waveform,
    materialize_ffv1_once,
    read_fixed_video_frames,
    read_json,
    save_torch_once,
    sha256_bytes,
    sha256_file,
    validate_target_offset_artifact,
    write_json_once,
)
from .syncnet_loss import audio_embeddings, cached_visual_embeddings, official_syncnet_distance_curve
from .train import PreparedRecord


def _expect(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ProtocolError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _lock_policy(policy_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(policy_path) != P1_POLICY_RECORD_SHA256:
        raise ProtocolError("P1 policy record hash mismatch")
    outer = read_json(policy_path)
    policy = outer.get("policy")
    if not isinstance(policy, dict):
        raise ProtocolError("policy record has no policy object")
    _expect(policy.get("sample_id"), P1_SAMPLE_ID, "sample_id")
    _expect(policy.get("source_group"), "6ORDQFh0Byw", "source_group")
    _expect(policy.get("protocol_split"), "train", "protocol_split")
    _expect(policy.get("failures"), [], "policy failures")
    source = policy["source"]
    _expect(source.get("sha256"), P1_SOURCE_VIDEO_SHA256, "source video hash in policy")
    _expect(float(source.get("fps")), 25.0, "source fps")
    source_path = Path(source["path"]).resolve()
    if sha256_file(source_path) != P1_SOURCE_VIDEO_SHA256:
        raise ProtocolError("source video content hash mismatch")
    legacy = policy["legacy_crop"]
    _expect(legacy.get("sha256"), P1_LEGACY_CROP_SHA256, "tracked crop hash in policy")
    crop_path = Path(legacy["path"]).resolve()
    if sha256_file(crop_path) != P1_LEGACY_CROP_SHA256:
        raise ProtocolError("tracked crop content hash mismatch")
    crop_meta = legacy["crop_metadata"]
    for key, value in {"fps": 25.0, "frame_count": 395, "width": 224, "height": 224}.items():
        _expect(crop_meta.get(key), value, f"tracked crop {key}")
    alignment = policy["frame_alignment"]
    required_alignment = {
        "verified": True,
        "local_start_frame": 0,
        "local_end_frame": 95,
        "original_start_frame": 0,
        "original_end_frame": 95,
        "frame_count": SEGMENT_FRAMES,
        "fps": 25.0,
    }
    for key, value in required_alignment.items():
        _expect(alignment.get(key), value, f"frame alignment {key}")
    _expect(alignment["evidence"].get("audio_sample_span"), {"start": 0, "end": SEGMENT_SAMPLES}, "audio span")
    _expect(alignment["evidence"].get("crop_audio_sample_span"), {"start": 0, "end": SEGMENT_SAMPLES}, "crop audio span")
    geometry = policy["syncnet"]["reference_geometry"]
    _expect(geometry.get("track_sha256"), P1_TRACK_SHA256, "track geometry hash")
    _expect(geometry.get("frame_count"), 395, "track geometry frame count")
    _expect(geometry.get("crop_scale"), 0.4, "track crop scale")
    _expect(geometry.get("padding_value"), 110, "track padding")
    _expect(geometry.get("output_size"), 224, "track output size")
    if any(len(geometry["processed"][key]) < SEGMENT_FRAMES for key in ("x", "y", "s")):
        raise ProtocolError("track processed geometry does not cover the segment")
    if len(geometry.get("boxes", [])) < SEGMENT_FRAMES:
        raise ProtocolError("track boxes do not cover the segment")
    natural = policy["natural_audio"]
    _expect(natural.get("sha256"), P1_NATURAL_AUDIO_SHA256, "natural audio hash in policy")
    natural_path = Path(natural["path"]).resolve()
    if sha256_file(natural_path) != P1_NATURAL_AUDIO_SHA256:
        raise ProtocolError("natural audio content hash mismatch")
    crop_audio_path = Path(legacy["audio_path"]).resolve()
    _expect(legacy.get("audio_sha256"), P1_CROP_AUDIO_SHA256, "crop audio hash in policy")
    if sha256_file(crop_audio_path) != P1_CROP_AUDIO_SHA256:
        raise ProtocolError("crop audio content hash mismatch")
    return policy, {
        "policy_record": str(policy_path),
        "policy_record_sha256": P1_POLICY_RECORD_SHA256,
        "source_video": str(source_path),
        "source_video_sha256": P1_SOURCE_VIDEO_SHA256,
        "legacy_tracked_crop": str(crop_path),
        "legacy_tracked_crop_sha256": P1_LEGACY_CROP_SHA256,
        "track_sha256": P1_TRACK_SHA256,
        "geometry": {
            "frame_start": 0,
            "frame_end": 95,
            "crop_scale": 0.4,
            "padding_value": 110,
            "output_size": 224,
            "boxes": geometry["boxes"][:SEGMENT_FRAMES],
            "processed": {key: geometry["processed"][key][:SEGMENT_FRAMES] for key in ("x", "y", "s")},
        },
        "natural_audio": str(natural_path),
        "natural_audio_sha256": P1_NATURAL_AUDIO_SHA256,
        "crop_audio": str(crop_audio_path),
        "crop_audio_sha256": P1_CROP_AUDIO_SHA256,
    }


def _lock_mfa(roots: AssetRoots) -> tuple[np.ndarray, dict[str, Any]]:
    summary_path = roots.mfa_linear / "summary.json"
    if sha256_file(summary_path) != P1_MFA_SUMMARY_SHA256:
        raise ProtocolError("MFA-linear summary hash mismatch")
    summary = read_json(summary_path)
    result = summary.get("results", {}).get(P1_SAMPLE_ID)
    if not isinstance(result, dict):
        raise ProtocolError("P1 record is absent from MFA-linear summary")
    _expect(result.get("sample_id"), P1_SAMPLE_ID, "MFA sample ID")
    _expect(result.get("split"), "train", "MFA split")
    _expect(result.get("audio_sha256"), P1_MFA_PARENT_SHA256, "MFA parent hash in summary")
    _expect(result.get("exact_natural_length"), True, "MFA exact natural length")
    _expect(result.get("output_samples"), 251904, "MFA parent sample count")
    audio_path = Path(result["audio_path"]).resolve()
    waveform, audio_lock = load_mfa_linear_waveform(audio_path, expected_sha256=P1_MFA_PARENT_SHA256)
    _expect(audio_lock["step0_rounded_pcm16_sha256"], P1_STEP0_PCM16_SHA256, "MFA step-0 PCM16 hash")
    return waveform, {
        "summary": str(summary_path.resolve()),
        "summary_sha256": P1_MFA_SUMMARY_SHA256,
        "summary_row": result,
        "audio": audio_lock,
    }


def prepare_p1(
    roots: AssetRoots,
    output_root: str | Path,
    *,
    device: torch.device,
) -> tuple[PreparedRecord, torch.nn.Module, dict[str, Any]]:
    """Construct locks once, then return a natural-free training record."""
    output = Path(output_root).resolve()
    policy_path = roots.policy_records / f"{P1_SAMPLE_ID}.json"
    policy, policy_lock = _lock_policy(policy_path)
    mfa_waveform, mfa_lock = _lock_mfa(roots)

    crop_path = Path(policy_lock["legacy_tracked_crop"])
    frames = read_fixed_video_frames(crop_path)
    hashes = frame_hashes(frames)
    _expect(hashes[0], P1_CROP_FRAME0_SHA256, "tracked crop frame 0")
    _expect(hashes[-1], P1_CROP_FRAME95_SHA256, "tracked crop frame 95")
    _expect(sha256_bytes(np.ascontiguousarray(frames).tobytes()), P1_CROP_96_CONCAT_SHA256, "tracked crop 96-frame concatenation")
    visual_lock = materialize_ffv1_once(output / "00_lock/visual" / f"{P1_SAMPLE_ID}.avi", frames)

    syncnet = load_frozen_syncnet(roots.syncnet_model, device=device)
    syncnet_hash = module_state_sha256(syncnet)
    scoring_frames = extract_official_bgr_frames(
        visual_lock["path"], output / "00_lock/official_visual_frames" / P1_SAMPLE_ID
    )
    visual_lock["scoring_bgr_frame_sha256"] = frame_hashes(scoring_frames)
    frames_tensor = torch.from_numpy(scoring_frames).to(device)
    visual_embedding = cached_visual_embeddings(syncnet, frames_tensor)
    if visual_embedding.shape != (91, 1024):
        raise ProtocolError(f"unexpected visual embedding shape: {tuple(visual_embedding.shape)}")

    crop_natural, crop_natural_lock = load_pcm16_waveform(
        policy_lock["crop_audio"], expected_sha256=P1_CROP_AUDIO_SHA256
    )
    parent_natural, parent_natural_lock = load_pcm16_waveform(
        policy_lock["natural_audio"], expected_sha256=P1_NATURAL_AUDIO_SHA256
    )
    # The policy crop audio contains tracker-pipeline padding and is provenance only.
    # Calibrate from the hash-locked original natural segment on the verified
    # frame_alignment audio span; never require the two containers to be equal.
    with torch.no_grad():
        natural_tensor = torch.from_numpy(parent_natural).to(device)[None, None]
        natural_audio_embedding = audio_embeddings(syncnet, natural_tensor)
        natural_curve = official_syncnet_distance_curve(natural_audio_embedding, visual_embedding).cpu()
    target = calibrate_target_offset(natural_curve)
    validate_target_offset_artifact(target)
    _expect(target["target_offset"], int(policy["target_curve"]["internal_offset"]), "exact-segment versus stored target offset")
    target.update(
        {
            "status": "complete",
            "source_role": "one_time_detached_coordinate_calibration_only",
            "natural_parent": parent_natural_lock,
            "tracked_crop_audio": crop_natural_lock,
            "visual_ffv1_sha256": visual_lock["sha256"],
            "syncnet_state_sha256": syncnet_hash,
        }
    )

    mfa_tensor = torch.from_numpy(mfa_waveform).to(device)[None, None]
    with torch.no_grad():
        pristine_embedding = audio_embeddings(syncnet, mfa_tensor)
        pristine_curve = official_syncnet_distance_curve(pristine_embedding, visual_embedding).detach()
    record = PreparedRecord(
        sample_id=P1_SAMPLE_ID,
        mfa_linear_tts_waveform=mfa_tensor,
        visual_embedding=visual_embedding.detach(),
        target_offset=int(target["target_offset"]),
        pristine_curve=pristine_curve.detach(),
    )
    record.validate()
    lock = {
        "status": "complete",
        "schema_version": 1,
        "sample_id": P1_SAMPLE_ID,
        "source_group": "6ORDQFh0Byw",
        "protocol_split": "train",
        "frame_interval": [0, SEGMENT_FRAMES],
        "sample_interval": [0, SEGMENT_SAMPLES],
        "trainable_input_modalities": ["mfa_linear_tts_waveform"],
        "natural_in_training_record": False,
        "policy": policy_lock,
        "mfa_linear": mfa_lock,
        "visual": visual_lock,
        "target_offset_artifact": target,
        "pristine_curve": [float(value) for value in pristine_curve.cpu()],
        "syncnet_checkpoint": str(roots.syncnet_model.resolve()),
        "syncnet_checkpoint_sha256": sha256_file(roots.syncnet_model),
        "syncnet_state_sha256": syncnet_hash,
        "q": Q,
    }
    write_json_once(output / "00_lock/input_lock.json", lock)
    save_torch_once(
        output / "00_lock/training_tensors.pt",
        {
            "mfa_linear_tts_waveform": record.mfa_linear_tts_waveform.detach().cpu(),
            "visual_embedding": record.visual_embedding.detach().cpu(),
            "target_offset": record.target_offset,
            "pristine_curve": record.pristine_curve.detach().cpu(),
        },
    )
    # Natural tensors leave scope here and are not members of PreparedRecord.
    del natural_tensor, natural_audio_embedding, crop_natural, parent_natural
    return record, syncnet, lock


def _prepare_additional_record(
    roots: AssetRoots,
    output: Path,
    *,
    sample_id: str,
    source_group: str,
    syncnet: torch.nn.Module,
    device: torch.device,
) -> tuple[PreparedRecord, dict[str, Any]]:
    summary_path = roots.mfa_linear / "summary.json"
    if sha256_file(summary_path) != P1_MFA_SUMMARY_SHA256:
        raise ProtocolError("shared MFA-linear summary hash changed")
    expected = P2_ASSET_LOCKS.get(sample_id)
    if expected is None:
        raise ProtocolError(f"no immutable P2 asset lock exists for {sample_id}")
    result = read_json(summary_path).get("results", {}).get(sample_id)
    if not isinstance(result, dict) or result.get("split") != "train" or result.get("sample_id") != sample_id:
        raise ProtocolError(f"invalid fit-only MFA row for {sample_id}")
    _expect(result.get("audio_sha256"), expected["mfa"], f"P2 MFA hash {sample_id}")
    audio_path = Path(result["audio_path"]).resolve()
    waveform, mfa_audio_lock = load_mfa_linear_waveform(
        audio_path, expected_sha256=expected["mfa"]
    )

    policy_path = roots.policy_records / f"{sample_id}.json"
    policy_sha = sha256_file(policy_path)
    _expect(policy_sha, expected["policy"], f"P2 policy hash {sample_id}")
    outer = read_json(policy_path)
    policy = outer.get("policy")
    if not isinstance(policy, dict):
        raise ProtocolError(f"missing policy object for {sample_id}")
    _expect(policy.get("sample_id"), sample_id, "P2 sample ID")
    _expect(policy.get("source_group"), source_group, "P2 source group")
    _expect(policy.get("protocol_split"), "train", "P2 protocol split")
    _expect(policy.get("failures"), [], "P2 policy failures")
    alignment = policy["frame_alignment"]
    for key, value in {
        "verified": True,
        "local_start_frame": 0,
        "local_end_frame": 95,
        "original_start_frame": 0,
        "original_end_frame": 95,
        "frame_count": SEGMENT_FRAMES,
        "fps": 25.0,
    }.items():
        _expect(alignment.get(key), value, f"P2 frame alignment {key}")
    _expect(alignment["evidence"].get("audio_sample_span"), {"start": 0, "end": SEGMENT_SAMPLES}, "P2 audio span")
    _expect(alignment["evidence"].get("crop_audio_sample_span"), {"start": 0, "end": SEGMENT_SAMPLES}, "P2 crop audio span")

    source = policy["source"]
    _expect(source.get("sha256"), expected["source"], f"P2 source hash {sample_id}")
    for key, value in {"fps": 25.0, "width": 224, "height": 224}.items():
        _expect(source.get(key), value, f"P2 source {key} {sample_id}")
    source_path = Path(source["path"]).resolve()
    if sha256_file(source_path) != expected["source"]:
        raise ProtocolError(f"source hash mismatch for {sample_id}")
    legacy = policy["legacy_crop"]
    _expect(legacy.get("sha256"), expected["crop"], f"P2 crop hash {sample_id}")
    _expect(legacy.get("audio_sha256"), expected["crop_audio"], f"P2 crop audio hash {sample_id}")
    crop_meta = legacy.get("crop_metadata", {})
    for key, value in {"fps": 25.0, "width": 224, "height": 224}.items():
        _expect(crop_meta.get(key), value, f"P2 crop {key} {sample_id}")
    if int(crop_meta.get("frame_count", 0)) < SEGMENT_FRAMES:
        raise ProtocolError(f"P2 crop frame count is too short for {sample_id}")
    crop_path = Path(legacy["path"]).resolve()
    if sha256_file(crop_path) != expected["crop"]:
        raise ProtocolError(f"tracked crop hash mismatch for {sample_id}")
    crop_audio_path = Path(legacy["audio_path"]).resolve()
    if sha256_file(crop_audio_path) != expected["crop_audio"]:
        raise ProtocolError(f"tracked crop audio hash mismatch for {sample_id}")
    natural_meta = policy["natural_audio"]
    _expect(natural_meta.get("sha256"), expected["natural"], f"P2 natural hash {sample_id}")
    natural_path = Path(natural_meta["path"]).resolve()
    if sha256_file(natural_path) != expected["natural"]:
        raise ProtocolError(f"natural parent hash mismatch for {sample_id}")
    geometry = policy["syncnet"]["reference_geometry"]
    _expect(geometry.get("track_sha256"), expected["track"], f"P2 track hash {sample_id}")
    track_path = Path(outer["crop"]["track_path"]).resolve()
    if sha256_file(track_path) != expected["track"]:
        raise ProtocolError(f"track geometry hash mismatch for {sample_id}")
    for key, value in {"crop_scale": 0.4, "padding_value": 110, "output_size": 224}.items():
        _expect(geometry.get(key), value, f"P2 geometry {key} {sample_id}")
    if geometry.get("frame_count", 0) < SEGMENT_FRAMES or len(geometry.get("boxes", [])) < SEGMENT_FRAMES:
        raise ProtocolError(f"track geometry support is insufficient for {sample_id}")

    frames = read_fixed_video_frames(crop_path)
    visual_lock = materialize_ffv1_once(output / "00_lock/visual" / f"{sample_id}.avi", frames)
    scoring_frames = extract_official_bgr_frames(
        visual_lock["path"], output / "00_lock/official_visual_frames" / sample_id
    )
    visual_lock["scoring_bgr_frame_sha256"] = frame_hashes(scoring_frames)
    visual = cached_visual_embeddings(syncnet, torch.from_numpy(scoring_frames).to(device))
    crop_natural, crop_natural_lock = load_pcm16_waveform(
        crop_audio_path, expected_sha256=str(legacy["audio_sha256"])
    )
    parent_natural, parent_natural_lock = load_pcm16_waveform(
        natural_path, expected_sha256=str(natural_meta["sha256"])
    )
    with torch.no_grad():
        natural_tensor = torch.from_numpy(parent_natural).to(device)[None, None]
        natural_curve = official_syncnet_distance_curve(
            audio_embeddings(syncnet, natural_tensor), visual
        ).cpu()
    target = calibrate_target_offset(natural_curve)
    validate_target_offset_artifact(target)
    _expect(target["target_offset"], int(policy["target_curve"]["internal_offset"]), f"P2 target offset {sample_id}")
    target.update(
        {
            "status": "complete",
            "source_role": "one_time_detached_coordinate_calibration_only",
            "natural_parent": parent_natural_lock,
            "tracked_crop_audio": crop_natural_lock,
            "visual_ffv1_sha256": visual_lock["sha256"],
            "syncnet_state_sha256": module_state_sha256(syncnet),
        }
    )
    waveform_tensor = torch.from_numpy(waveform).to(device)[None, None]
    with torch.no_grad():
        pristine = official_syncnet_distance_curve(
            audio_embeddings(syncnet, waveform_tensor), visual
        ).detach()
    record = PreparedRecord(sample_id, waveform_tensor, visual.detach(), int(target["target_offset"]), pristine)
    record.validate()
    lock = {
        "status": "complete",
        "schema_version": 1,
        "sample_id": sample_id,
        "source_group": source_group,
        "protocol_split": "train",
        "policy_record": str(policy_path.resolve()),
        "policy_record_sha256": policy_sha,
        "source_video": {"path": str(source_path), "sha256": source["sha256"]},
        "legacy_tracked_crop": {"path": str(crop_path), "sha256": legacy["sha256"]},
        "track_sha256": geometry["track_sha256"],
        "mfa_linear": {"summary_sha256": P1_MFA_SUMMARY_SHA256, "summary_row": result, "audio": mfa_audio_lock},
        "visual": visual_lock,
        "target_offset_artifact": target,
        "pristine_curve": [float(value) for value in pristine.cpu()],
        "trainable_input_modalities": ["mfa_linear_tts_waveform"],
        "natural_in_training_record": False,
    }
    write_json_once(output / "00_lock/p2" / f"{sample_id}.json", lock)
    del natural_tensor, crop_natural, parent_natural
    return record, lock


def prepare_p2(
    roots: AssetRoots,
    output_root: str | Path,
    *,
    p1_record: PreparedRecord,
    p1_lock: dict[str, Any],
    syncnet: torch.nn.Module,
    device: torch.device,
) -> tuple[list[PreparedRecord], list[dict[str, Any]]]:
    output = Path(output_root).resolve()
    records = [p1_record]
    locks = [p1_lock]
    for sample_id, source_group in P2_RECORDS[1:]:
        record, lock = _prepare_additional_record(
            roots,
            output,
            sample_id=sample_id,
            source_group=source_group,
            syncnet=syncnet,
            device=device,
        )
        records.append(record)
        locks.append(lock)
    if [record.sample_id for record in records] != [sample_id for sample_id, _ in P2_RECORDS]:
        raise ProtocolError("P2 record order changed")
    if len({lock["source_group"] for lock in locks}) != 4 or any(lock["protocol_split"] != "train" for lock in locks):
        raise ProtocolError("P2 requires four distinct fit-only source groups")
    save_torch_once(
        output / "00_lock/p2_training_tensors.pt",
        {
            record.sample_id: {
                "mfa_linear_tts_waveform": record.mfa_linear_tts_waveform.detach().cpu(),
                "visual_embedding": record.visual_embedding.detach().cpu(),
                "target_offset": record.target_offset,
                "pristine_curve": record.pristine_curve.detach().cpu(),
            }
            for record in records
        },
    )
    write_json_once(
        output / "00_lock/p2_manifest.json",
        {
            "status": "complete",
            "record_order": [record.sample_id for record in records],
            "source_groups": [lock["source_group"] for lock in locks],
            "protocol_split": "train",
            "selection_rule": "frozen_spec_order",
            "natural_in_training_records": False,
        },
    )
    return records, locks
