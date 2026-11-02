#!/usr/bin/env python3
"""CAL-only AP diagnostics conditioned on activity, voicing and mapped phones.

This read-only sidecar consumes the frozen WORLD CAL output and its feature,
phone-map and transfer-audit files. It never opens EVAL audio or endpoint data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16_000
ACTIVE_THRESHOLD_DBFS = -45.0
EXPECTED_EXECUTION_BINDINGS_SHA256 = "cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d"
BANDS = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8001))
DIRECTIONS = {
    "tts_to_natural": {"recipient": "natural", "donor": "tts", "contrast": "T_minus_N"},
    "natural_to_tts": {"recipient": "tts", "donor": "natural", "contrast": "N_minus_T"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} hash mismatch: expected={expected}, actual={actual}, path={path}")


def load_arrays(path: Path, np: Any) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def quantiles(values: Any, np: Any) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return [float(x) for x in np.percentile(array, [10, 50, 90])]


def ap_band_summary(ap: Any, mask: Any, frequencies: Any, np: Any) -> dict[str, Any]:
    result = {}
    for low, high in BANDS:
        freq_mask = (frequencies >= low) & (frequencies < high)
        frame_band = np.mean(ap[:, freq_mask], axis=1) if np.any(freq_mask) else np.full(len(mask), np.nan)
        values = frame_band[mask]
        result[f"{low}_{high}Hz"] = {
            "frequency_bin_count": int(np.count_nonzero(freq_mask)),
            "mean": float(np.mean(values)) if values.size else None,
            "p10_p50_p90": quantiles(values, np),
        }
    return result


def temporal_transition_counts(indices: Any, recipient_voiced: Any, donor_voiced: Any, np: Any) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size < 2:
        return {
            "consecutive_exposed_frame_pairs": 0,
            "recipient_0_to_1": 0, "recipient_1_to_0": 0,
            "donor_0_to_1": 0, "donor_1_to_0": 0,
            "joint_state_transitions": {},
        }
    keep = np.diff(idx) == 1
    left, right = idx[:-1][keep], idx[1:][keep]
    r0, r1 = recipient_voiced[left], recipient_voiced[right]
    d0, d1 = donor_voiced[left], donor_voiced[right]
    transitions: dict[str, int] = defaultdict(int)
    for ra, da, rb, db in zip(r0, d0, r1, d1):
        transitions[f"r{int(ra)}d{int(da)}_to_r{int(rb)}d{int(db)}"] += 1
    return {
        "consecutive_exposed_frame_pairs": int(len(left)),
        "recipient_0_to_1": int(np.count_nonzero((~r0) & r1)),
        "recipient_1_to_0": int(np.count_nonzero(r0 & (~r1))),
        "donor_0_to_1": int(np.count_nonzero((~d0) & d1)),
        "donor_1_to_0": int(np.count_nonzero(d0 & (~d1))),
        "joint_state_transitions": dict(sorted(transitions.items())),
    }


def summarize_phone_exposure(phone_map: dict[str, Any], times: Any, active: Any, mapped: Any, joint: Any, recipient_voiced: Any, donor_voiced: Any, duration_s: float, np: Any) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for phone in phone_map["matched_phones"]:
        label = str(phone["recipient_normalized_label"])
        start, end = map(float, phone["recipient_interval_s"])
        in_phone = (times >= start) & (times < end)
        if abs(end - duration_s) <= 1e-9:
            in_phone |= np.abs(times - end) <= 1e-9
        phone_indices = np.flatnonzero(in_phone)
        exposed_indices = np.flatnonzero(in_phone & mapped)
        active_exposed = in_phone & mapped & active
        active_joint = in_phone & active & joint
        row = grouped.setdefault(label, {
            "phone_category": label,
            "definition": "exact normalized MFA phone label; no stress/tone/aspiration folding",
            "matched_phone_token_count": 0,
            "recipient_phone_world_frames": 0,
            "mapped_exposure_frames": 0,
            "active_mapped_exposure_frames": 0,
            "active_jointly_voiced_frames": 0,
            "recipient_voiced_exposure_frames": 0,
            "donor_voiced_exposure_frames": 0,
            "voicing_pair_counts": {"recipient_unvoiced_donor_unvoiced": 0, "recipient_unvoiced_donor_voiced": 0, "recipient_voiced_donor_unvoiced": 0, "recipient_voiced_donor_voiced": 0},
            "transition_counts": {"consecutive_exposed_frame_pairs": 0, "recipient_0_to_1": 0, "recipient_1_to_0": 0, "donor_0_to_1": 0, "donor_1_to_0": 0, "joint_state_transitions": {}},
        })
        row["matched_phone_token_count"] += 1
        row["recipient_phone_world_frames"] += int(len(phone_indices))
        row["mapped_exposure_frames"] += int(len(exposed_indices))
        row["active_mapped_exposure_frames"] += int(np.count_nonzero(active_exposed))
        row["active_jointly_voiced_frames"] += int(np.count_nonzero(active_joint))
        row["recipient_voiced_exposure_frames"] += int(np.count_nonzero(recipient_voiced[exposed_indices]))
        row["donor_voiced_exposure_frames"] += int(np.count_nonzero(donor_voiced[exposed_indices]))
        for index in exposed_indices:
            r_state, d_state = bool(recipient_voiced[index]), bool(donor_voiced[index])
            cell = f"recipient_{'voiced' if r_state else 'unvoiced'}_donor_{'voiced' if d_state else 'unvoiced'}"
            row["voicing_pair_counts"][cell] += 1
        transitions = temporal_transition_counts(exposed_indices, recipient_voiced, donor_voiced, np)
        for key in ("consecutive_exposed_frame_pairs", "recipient_0_to_1", "recipient_1_to_0", "donor_0_to_1", "donor_1_to_0"):
            row["transition_counts"][key] += transitions[key]
        for key, count in transitions["joint_state_transitions"].items():
            row["transition_counts"]["joint_state_transitions"][key] = row["transition_counts"]["joint_state_transitions"].get(key, 0) + count
    result = []
    for label in sorted(grouped):
        row = grouped[label]
        row["exposure_fraction_of_phone_frames"] = row["mapped_exposure_frames"] / max(1, row["recipient_phone_world_frames"])
        row["active_exposure_fraction_of_phone_frames"] = row["active_mapped_exposure_frames"] / max(1, row["recipient_phone_world_frames"])
        result.append(row)
    return result


def residual_summary(values: Any, np: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"frame_count": 0, "median_db": None, "p95_absolute_db": None, "max_absolute_db": None}
    return {
        "frame_count": int(array.size),
        "median_db": float(np.median(array)),
        "p95_absolute_db": float(np.percentile(np.abs(array), 95)),
        "max_absolute_db": float(np.max(np.abs(array))),
    }


def energy_qc_for_direction(direction_name: str, direction: dict[str, Any], phone_map: dict[str, Any], recipient_audio: Any, module: Any, sf: Any, np: Any, pooled: dict[Any, Any]) -> dict[str, Any]:
    """Read saved controls and explain their envelope dose without refitting them."""
    rt_meta = direction["arms"]["RT"]
    rt_path = Path(rt_meta["path"])
    require_hash(rt_path, rt_meta["file_sha256"], "RT arm audio")
    rt_audio, rt_rate = sf.read(str(rt_path), dtype="float32", always_2d=True)
    if int(rt_rate) != SAMPLE_RATE or rt_audio.shape[1] != 1:
        raise ValueError(f"RT arm audio clock mismatch: {rt_path}")
    rt_times, rt_envelope = module.local_rms_envelope(rt_audio[:, 0], np)
    recipient_times, recipient_envelope = module.local_rms_envelope(recipient_audio, np)
    result: dict[str, Any] = {}
    radius = int(getattr(module, "ENERGY_WINDOW_RADIUS", 400))
    floor = float(getattr(module, "LOCAL_RMS_FLOOR", 1e-5))
    bound = float(getattr(module, "LOCAL_BOUND_DB", 12.0))

    for arm in ("EX_EM", "SP_EM", "E"):
        meta = direction["arms"].get(arm)
        if not meta or meta.get("status") != "ok":
            result[arm] = {"status": "missing_or_failed"}
            continue
        gain_meta = meta["gain_audit"]
        gain_path = Path(gain_meta["path"])
        require_hash(gain_path, gain_meta["file_sha256"], f"{arm} gain audit")
        audit = load_arrays(gain_path, np)
        times = audit["envelope_times_s"].astype(np.float64)
        target = audit["target_rms"].astype(np.float64)
        achieved = audit["achieved_rms"].astype(np.float64)
        sample_gain_db = audit["local_gain_db"].astype(np.float64)
        if times.shape != target.shape or times.shape != achieved.shape:
            raise ValueError(f"energy envelope dimensions differ: {arm}/{gain_path}")
        if sample_gain_db.size != len(recipient_audio):
            raise ValueError(f"sample gain length differs from recipient PCM: {arm}/{gain_path}")
        recipient_env_at_times = np.interp(times, recipient_times, recipient_envelope, left=float(recipient_envelope[0]), right=float(recipient_envelope[-1]))
        fixed_activity = recipient_env_at_times >= 10.0 ** (ACTIVE_THRESHOLD_DBFS / 20.0)
        residual_db = 20.0 * np.log10(np.maximum(achieved, floor) / np.maximum(target, floor))
        global_median = float(np.median(residual_db)) if residual_db.size else 0.0
        shape_residual = residual_db - global_median

        saturated_samples = np.abs(sample_gain_db) >= bound - 1e-6
        padded_saturation = np.pad(saturated_samples, (radius, radius), mode="reflect")
        centers = np.rint(times * SAMPLE_RATE).astype(np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(padded_saturation, 2 * radius + 1)[centers]
        saturated_frames = np.any(windows, axis=1)

        rt_at_times = np.interp(times, rt_times, rt_envelope, left=float(rt_envelope[0]), right=float(rt_envelope[-1]))
        target_power = target * target
        achieved_power = achieved * achieved
        rt_power = rt_at_times * rt_at_times
        matched_phone_rows = []
        category_counts: dict[str, dict[str, int]] = {}
        for phone in phone_map.get("matched_phones", []):
            start, end = map(float, phone["recipient_interval_s"])
            in_phone = (times >= start) & (times < end)
            if abs(end - len(recipient_audio) / SAMPLE_RATE) <= 1e-9:
                in_phone |= np.abs(times - end) <= 1e-9
            frame_count = int(np.count_nonzero(in_phone))
            saturated_count = int(np.count_nonzero(in_phone & saturated_frames))
            label = str(phone["recipient_normalized_label"])
            counts = category_counts.setdefault(label, {"matched_phone_tokens": 0, "tokens_with_any_saturated_frame": 0, "saturated_frames": 0, "total_phone_frames": 0})
            counts["matched_phone_tokens"] += 1
            counts["tokens_with_any_saturated_frame"] += int(saturated_count > 0)
            counts["saturated_frames"] += saturated_count
            counts["total_phone_frames"] += frame_count
            matched_phone_rows.append({
                "phone_category": label,
                "recipient_interval_s": phone["recipient_interval_s"],
                "world_envelope_frame_count": frame_count,
                "saturated_frame_count": saturated_count,
                "has_any_saturated_frame": bool(saturated_count > 0),
            })
        saturated_phone_count = sum(x["has_any_saturated_frame"] for x in matched_phone_rows)
        status_audit = meta.get("energy_match_audit", {})
        pool_key = (direction_name, arm)
        pool = pooled.setdefault(pool_key, {
            "direction": direction_name, "arm": arm, "sample_count": 0,
            "active_residuals_db": [], "active_shape_residuals_db": [],
            "target_power": 0.0, "achieved_power": 0.0, "rt_power": 0.0,
            "saturated_samples": 0, "samples": 0, "saturated_frames": 0, "frames": 0,
            "saturated_active_frames": 0, "active_frames": 0,
            "saturated_phones": 0, "phones": 0, "phone_categories": {},
        })
        pool["sample_count"] += 1
        pool["active_residuals_db"].extend(residual_db[fixed_activity].tolist())
        pool["active_shape_residuals_db"].extend(shape_residual[fixed_activity].tolist())
        pool["target_power"] += float(np.sum(target_power))
        pool["achieved_power"] += float(np.sum(achieved_power))
        pool["rt_power"] += float(np.sum(rt_power))
        pool["saturated_samples"] += int(np.count_nonzero(saturated_samples))
        pool["samples"] += int(sample_gain_db.size)
        pool["saturated_frames"] += int(np.count_nonzero(saturated_frames))
        pool["frames"] += int(saturated_frames.size)
        pool["saturated_active_frames"] += int(np.count_nonzero(saturated_frames & fixed_activity))
        pool["active_frames"] += int(np.count_nonzero(fixed_activity))
        pool["saturated_phones"] += int(saturated_phone_count)
        pool["phones"] += len(matched_phone_rows)
        for label, counts in category_counts.items():
            target_category = pool["phone_categories"].setdefault(label, {key: 0 for key in counts})
            for key, value in counts.items():
                target_category[key] += int(value)
        result[arm] = {
            "status": "ok",
            "gain_audit_path": str(gain_path),
            "gain_audit_sha256": gain_meta["file_sha256"],
            "energy_audit_active_definition": "target envelope RMS >= 1e-5 (producer floor); this is not speech activity",
            "producer_target_active_frames_at_floor": status_audit.get("target_rms_envelope_active_frames"),
            "recipient_fixed_activity_definition": "recipient FIXED local RMS envelope >= -45 dBFS",
            "recipient_fixed_active_frame_count": int(np.count_nonzero(fixed_activity)),
            "recipient_fixed_active_frame_fraction": float(np.mean(fixed_activity)) if fixed_activity.size else 0.0,
            "achieved_minus_target_db_on_recipient_fixed_activity": residual_summary(residual_db[fixed_activity], np),
            "global_median_residual_db_all_envelope_frames": global_median,
            "global_median_centered_shape_residual_db_on_recipient_fixed_activity": residual_summary(shape_residual[fixed_activity], np),
            "global_median_centered_shape_residual_db_all_envelope_frames": residual_summary(shape_residual, np),
            "total_envelope_power": {
                "target_sum_rms_squared": float(np.sum(target_power)),
                "target_mean_rms_squared": float(np.mean(target_power)) if target_power.size else None,
                "achieved_sum_rms_squared": float(np.sum(achieved_power)),
                "achieved_mean_rms_squared": float(np.mean(achieved_power)) if achieved_power.size else None,
                "rt_sum_rms_squared": float(np.sum(rt_power)),
                "rt_mean_rms_squared": float(np.mean(rt_power)) if rt_power.size else None,
                "rt_to_target_power_ratio": float(np.sum(rt_power) / max(np.sum(target_power), 1e-30)),
                "rt_to_achieved_power_ratio": float(np.sum(rt_power) / max(np.sum(achieved_power), 1e-30)),
                "target_to_achieved_power_ratio": float(np.sum(target_power) / max(np.sum(achieved_power), 1e-30)),
            },
            "local_gain_saturation": {
                "bound_db": bound,
                "saturated_sample_count": int(np.count_nonzero(saturated_samples)),
                "sample_count": int(sample_gain_db.size),
                "saturated_sample_fraction": float(np.mean(saturated_samples)) if saturated_samples.size else 0.0,
                "saturated_frame_count_any_within_801_sample_window": int(np.count_nonzero(saturated_frames)),
                "frame_count": int(saturated_frames.size),
                "saturated_frame_fraction": float(np.mean(saturated_frames)) if saturated_frames.size else 0.0,
                "saturated_active_frame_count": int(np.count_nonzero(saturated_frames & fixed_activity)),
                "active_frame_count": int(np.count_nonzero(fixed_activity)),
                "saturated_phone_token_count_any_frame": int(saturated_phone_count),
                "matched_phone_token_count": len(matched_phone_rows),
                "saturated_phone_token_fraction": float(saturated_phone_count / max(1, len(matched_phone_rows))),
                "by_phone_category": category_counts,
                "per_matched_phone": matched_phone_rows,
            },
        }
    return result


def load_source_module(path: Path, expected_sha: str) -> Any:
    require_hash(path, expected_sha, "WORLD producer source")
    spec = importlib.util.spec_from_file_location("world_producer_for_cal_ap_summary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import WORLD producer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    manifest_path = args.world_manifest.resolve()
    source_path = args.producer_source.resolve()
    execution_bindings_path = args.execution_bindings.resolve()
    execution_bindings_sha = sha256_file(execution_bindings_path)
    if execution_bindings_sha != args.expected_execution_bindings_sha256:
        raise ValueError("execution-bindings annex hash mismatch")
    world = read_json(manifest_path)
    if world.get("split") != "calibration" or world.get("manifest_type") != "world_phone_transfer_calibration_producer":
        raise ValueError("input must be the full frozen calibration WORLD producer manifest")
    if world.get("evaluation_outcomes_read") or world.get("evaluation_waveforms_read"):
        raise ValueError("CAL-only producer manifest contains an EVAL-read marker")
    if args.expected_records and len(world.get("records", [])) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} CAL records, got {len(world.get('records', []))}")
    source_sha = str(world.get("phone_normalizer_source_sha256", ""))
    module = load_source_module(source_path, source_sha)
    plan_path = Path(str(world.get("world_transfer_plan_path", ""))).resolve()
    if not plan_path.is_file():
        raise ValueError("sealed WORLD plan file is missing")
    plan = read_json(plan_path)
    plan_raw_sha = sha256_file(plan_path)
    plan_embedded_sha = str(plan.get("manifest_sha256", ""))
    if not plan_embedded_sha or fingerprint(plan) != plan_embedded_sha:
        raise ValueError("sealed WORLD plan embedded fingerprint mismatch")
    if plan_embedded_sha != str(world.get("world_transfer_plan_sha256", "")):
        raise ValueError("WORLD manifest is bound to a different canonical plan fingerprint")
    if plan.get("evaluation_outcomes_read"):
        raise ValueError("sealed WORLD plan indicates evaluation outcomes were read")
    expected_arms = set(world.get("arms", []))
    if expected_arms != {"RT", "EX", "SP", "BOTH", "F0", "AP", "EX_EM", "SP_EM", "E"}:
        raise ValueError("WORLD arm set does not match the sealed nine-arm calibration plan")
    per_record = []
    pooled: dict[str, dict[str, Any]] = {}
    phone_pooled: dict[str, dict[str, Any]] = {}
    energy_pooled: dict[Any, Any] = {}

    for record in world["records"]:
        sample_id = str(record["id"])
        if record.get("split") != "calibration" or not record.get("pair_qualified"):
            raise ValueError(f"CAL row is not qualified: {sample_id}")
        sides = {}
        side_domains = {}
        for condition in ("natural", "tts"):
            info = record[condition]
            feature_path = Path(info["world_features"]["path"])
            fixed_path = Path(info["fixed_audio"]["path"])
            require_hash(feature_path, info["world_features"]["file_sha256"], f"{sample_id}/{condition} WORLD feature")
            require_hash(fixed_path, info["fixed_audio"]["file_sha256"], f"{sample_id}/{condition} fixed audio")
            feature = load_arrays(feature_path, np)
            audio, sample_rate = sf.read(str(fixed_path), dtype="float32", always_2d=True)
            if int(sample_rate) != SAMPLE_RATE or audio.shape[1] != 1 or audio.shape[0] != int(info["raw_decoded_samples"]):
                raise ValueError(f"fixed PCM clock mismatch: {sample_id}/{condition}")
            audio = audio[:, 0]
            times, f0, ap, frequencies = feature["time"], feature["f0"], feature["ap"], feature["frequency_hz"]
            env_times, env = module.local_rms_envelope(audio, np)
            env_at_world = np.interp(times, env_times, env, left=float(env[0]), right=float(env[-1]))
            active = env_at_world >= 10.0 ** (ACTIVE_THRESHOLD_DBFS / 20.0)
            voiced = f0 > 0.0
            domains = {}
            for name, mask in (("world_voiced", voiced), ("fixed_activity_and_world_voiced", active & voiced)):
                domains[name] = {
                    "frame_count": int(np.count_nonzero(mask)),
                    "weight_fraction_of_world_frames": float(np.mean(mask)) if mask.size else 0.0,
                    "active_mask_threshold_dbfs": ACTIVE_THRESHOLD_DBFS if name == "fixed_activity_and_world_voiced" else None,
                    "aperiodicity_by_band": ap_band_summary(ap, mask, frequencies, np),
                }
            sides[condition] = {"feature": feature, "audio": audio, "active": active, "voiced": voiced, "domains": domains}
            side_domains[condition] = domains

        direction_rows = {}
        for direction_name, rule in DIRECTIONS.items():
            direction = record["directions"].get(direction_name)
            if not direction:
                raise ValueError(f"missing direction {direction_name}: {sample_id}")
            recipient_condition, donor_condition = rule["recipient"], rule["donor"]
            if direction.get("recipient_condition") != recipient_condition or direction.get("donor_condition") != donor_condition:
                raise ValueError(f"direction label/donor identity mismatch: {sample_id}/{direction_name}")
            phone_meta = direction["phone_map"]
            audit_meta = direction["parameter_transfer_audit"]
            phone_path, audit_path = Path(phone_meta["path"]), Path(audit_meta["path"])
            require_hash(phone_path, phone_meta["file_sha256"], f"{sample_id}/{direction_name} phone map")
            require_hash(audit_path, audit_meta["file_sha256"], f"{sample_id}/{direction_name} parameter audit")
            phone_map = read_json(phone_path)
            audit = load_arrays(audit_path, np)
            recipient, donor = sides[recipient_condition], sides[donor_condition]
            rt = recipient["feature"]["time"]
            mapped = audit["mapped_frames"].astype(bool)
            joint = audit["joint_voiced"].astype(bool)
            donor_voiced = audit["donor_voiced"].astype(bool)
            own_voiced = recipient["voiced"]
            if mapped.shape != rt.shape or joint.shape != rt.shape or donor_voiced.shape != rt.shape:
                raise ValueError(f"transfer-audit frame dimensions mismatch: {sample_id}/{direction_name}")
            active = recipient["active"]
            domain = active & joint & mapped
            mapped_ap = audit["mapped_ap"]
            own_ap = recipient["feature"]["ap"]
            frequencies = recipient["feature"]["frequency_hz"]
            per_band = {}
            for low, high in BANDS:
                freq_mask = (frequencies >= low) & (frequencies < high)
                own_frame = np.mean(own_ap[:, freq_mask], axis=1)
                donor_frame = np.mean(mapped_ap[:, freq_mask], axis=1)
                delta = donor_frame - own_frame
                per_band[f"{low}_{high}Hz"] = {
                    "frequency_bin_count": int(np.count_nonzero(freq_mask)),
                    "recipient_own_mean": float(np.mean(own_frame[domain])) if np.any(domain) else None,
                    "donor_mapped_mean": float(np.mean(donor_frame[domain])) if np.any(domain) else None,
                    "donor_minus_recipient_mean_p10_p50_p90": quantiles(delta[domain], np),
                }
            phone_exposure = summarize_phone_exposure(
                phone_map, rt, active, mapped, joint, own_voiced, donor_voiced,
                len(recipient["audio"]) / SAMPLE_RATE, np,
            )
            local_energy_qc = energy_qc_for_direction(
                direction_name, direction, phone_map, recipient["audio"], module, sf, np, energy_pooled,
            )
            direction_rows[direction_name] = {
                "recipient_condition": recipient_condition,
                "donor_condition": donor_condition,
                "contrast": rule["contrast"],
                "matched_phone_count": int(phone_meta["matched_phone_count"]),
                "mapped_frame_count": int(np.count_nonzero(mapped)),
                "mapped_frame_weight_of_recipient_clock": float(np.mean(mapped)) if mapped.size else 0.0,
                "recipient_active_frame_count": int(np.count_nonzero(active)),
                "jointly_voiced_active_mapped_frame_count": int(np.count_nonzero(domain)),
                "jointly_voiced_active_mapped_weight_of_recipient_clock": float(np.mean(domain)) if domain.size else 0.0,
                "active_threshold_dbfs": ACTIVE_THRESHOLD_DBFS,
                "aperiodicity_by_band": per_band,
                "matched_phone_categories": phone_exposure,
                "local_energy_qc": local_energy_qc,
            }
            aggregate = pooled.setdefault(direction_name, {
                "direction": direction_name, "contrast": rule["contrast"], "recipient_condition": recipient_condition,
                "donor_condition": donor_condition, "sample_count": 0, "recipient_world_frame_count": 0,
                "mapped_frame_count": 0, "recipient_active_frame_count": 0,
                "jointly_voiced_active_mapped_frame_count": 0, "band_deltas": defaultdict(list),
            })
            aggregate["sample_count"] += 1
            aggregate["recipient_world_frame_count"] += int(len(rt))
            aggregate["mapped_frame_count"] += int(np.count_nonzero(mapped))
            aggregate["recipient_active_frame_count"] += int(np.count_nonzero(active))
            aggregate["jointly_voiced_active_mapped_frame_count"] += int(np.count_nonzero(domain))
            for band, stats in per_band.items():
                if stats["donor_minus_recipient_mean_p10_p50_p90"] is not None:
                    # The p50 is the median clip-frame value, so pool band values
                    # by each included frame count below, rather than by clip count.
                    low, high = (int(x) for x in band.removesuffix("Hz").split("_"))
                    freq_mask = (frequencies >= low) & (frequencies < high)
                    own_frame = np.mean(own_ap[:, freq_mask], axis=1)
                    donor_frame = np.mean(mapped_ap[:, freq_mask], axis=1)
                    aggregate["band_deltas"][band].extend((donor_frame[domain] - own_frame[domain]).tolist())
            for row in phone_exposure:
                key = (direction_name, row["phone_category"])
                summary = phone_pooled.setdefault(key, {k: 0 for k in (
                    "matched_phone_token_count", "recipient_phone_world_frames", "mapped_exposure_frames",
                    "active_mapped_exposure_frames", "active_jointly_voiced_frames",
                    "recipient_voiced_exposure_frames", "donor_voiced_exposure_frames",
                )})
                for field in (
                    "matched_phone_token_count", "recipient_phone_world_frames", "mapped_exposure_frames",
                    "active_mapped_exposure_frames", "active_jointly_voiced_frames",
                    "recipient_voiced_exposure_frames", "donor_voiced_exposure_frames",
                ):
                    summary[field] += int(row[field])
                for cell, count in row["voicing_pair_counts"].items():
                    summary.setdefault("voicing_pair_counts", {}).setdefault(cell, 0)
                    summary["voicing_pair_counts"][cell] += int(count)
                for field, count in row["transition_counts"].items():
                    if field == "joint_state_transitions":
                        target = summary.setdefault(field, {})
                        for transition, n in count.items():
                            target[transition] = target.get(transition, 0) + int(n)
                    else:
                        summary.setdefault(field, 0)
                        summary[field] += int(count)

        per_record.append({
            "id": sample_id, "speaker": record["speaker"],
            "source_identity_sha256": record["source_identity_sha256"],
            "condition_domains": side_domains,
            "directions": direction_rows,
        })

    aggregate_rows = []
    for direction_name, item in sorted(pooled.items()):
        frames = int(item["jointly_voiced_active_mapped_frame_count"])
        aggregate_rows.append({
            "direction": direction_name, "contrast": item["contrast"],
            "recipient_condition": item["recipient_condition"], "donor_condition": item["donor_condition"],
            "sample_count": item["sample_count"],
            "recipient_world_frame_count": item["recipient_world_frame_count"],
            "mapped_frame_count": item["mapped_frame_count"],
            "recipient_active_frame_count": item["recipient_active_frame_count"],
            "jointly_voiced_active_mapped_frame_count": frames,
            "jointly_voiced_active_mapped_weight_of_recipient_clock": item["jointly_voiced_active_mapped_frame_count"] / max(1, item["recipient_world_frame_count"]),
            "aperiodicity_delta_by_band": {
                band: {"frame_count": len(values), "donor_minus_recipient_mean_p10_p50_p90": quantiles(values, np)}
                for band, values in sorted(item["band_deltas"].items())
            },
        })
    phone_rows = []
    for (direction, label), item in sorted(phone_pooled.items()):
        item["direction"] = direction
        item["contrast"] = DIRECTIONS[direction]["contrast"]
        item["phone_category"] = label
        item["definition"] = "exact normalized MFA phone label; no stress/tone/aspiration folding"
        item["exposure_fraction_of_phone_frames"] = item["mapped_exposure_frames"] / max(1, item["recipient_phone_world_frames"])
        item["active_exposure_fraction_of_phone_frames"] = item["active_mapped_exposure_frames"] / max(1, item["recipient_phone_world_frames"])
        phone_rows.append(item)

    energy_rows = []
    for (direction, arm), item in sorted(energy_pooled.items()):
        energy_rows.append({
            "direction": direction,
            "contrast": DIRECTIONS[direction]["contrast"],
            "arm": arm,
            "sample_count": item["sample_count"],
            "recipient_fixed_active_frame_count": item["active_frames"],
            "achieved_minus_target_db_on_recipient_fixed_activity": residual_summary(item["active_residuals_db"], np),
            "global_median_centered_shape_residual_db_on_recipient_fixed_activity": residual_summary(item["active_shape_residuals_db"], np),
            "total_envelope_power": {
                "target_sum_rms_squared": item["target_power"],
                "achieved_sum_rms_squared": item["achieved_power"],
                "rt_sum_rms_squared": item["rt_power"],
                "rt_to_target_power_ratio": item["rt_power"] / max(item["target_power"], 1e-30),
                "rt_to_achieved_power_ratio": item["rt_power"] / max(item["achieved_power"], 1e-30),
                "target_to_achieved_power_ratio": item["target_power"] / max(item["achieved_power"], 1e-30),
            },
            "local_gain_saturation": {
                "saturated_sample_count": item["saturated_samples"], "sample_count": item["samples"],
                "saturated_sample_fraction": item["saturated_samples"] / max(1, item["samples"]),
                "saturated_frame_count": item["saturated_frames"], "frame_count": item["frames"],
                "saturated_frame_fraction": item["saturated_frames"] / max(1, item["frames"]),
                "saturated_active_frame_count": item["saturated_active_frames"], "active_frame_count": item["active_frames"],
                "saturated_phone_token_count": item["saturated_phones"], "matched_phone_token_count": item["phones"],
                "saturated_phone_token_fraction": item["saturated_phones"] / max(1, item["phones"]),
                "by_phone_category": item["phone_categories"],
            },
        })

    payload = {
        "schema_version": 1,
        "manifest_type": "world_calibration_conditional_acoustic_qc",
        "status": "CAL_ONLY_READ_ONLY_DIAGNOSTIC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "calibration",
        "evaluation_waveforms_read": False,
        "evaluation_outcomes_read": False,
        "world_manifest_path": str(manifest_path),
        "world_manifest_file_sha256": sha256_file(manifest_path),
        "world_manifest_sha256": world.get("manifest_sha256"),
        "world_plan_path": str(plan_path),
        "world_plan_raw_file_sha256": plan_raw_sha,
        "world_plan_embedded_canonical_sha256": plan_embedded_sha,
        "world_transfer_plan_sha256": world.get("world_transfer_plan_sha256"),
        "mfa_producer_manifest_sha256": world.get("mfa_producer_manifest_sha256"),
        "mfa_protocol_manifest_sha256": world.get("mfa_protocol_manifest_sha256"),
        "mfa_qualification_manifest_sha256": world.get("mfa_qualification_manifest_sha256"),
        "phone_normalization_table_sha256": world.get("phone_normalization_table_sha256"),
        "world_producer_source_path": str(source_path),
        "world_producer_source_sha256": source_sha,
        "sidecar_script_path": str(Path(__file__).resolve()),
        "sidecar_script_sha256": sha256_file(Path(__file__).resolve()),
        "execution_bindings_path": str(execution_bindings_path),
        "execution_bindings_sha256": execution_bindings_sha,
        "activity_rule": {"source": "recipient FIXED PCM local RMS envelope", "threshold_dbfs": ACTIVE_THRESHOLD_DBFS, "window_samples": 801, "hop_samples": 80, "window": "same frozen WORLD producer local_rms_envelope implementation"},
        "aperiodicity_bands_hz": [[low, high] for low, high in BANDS],
        "per_condition_domains": "own AP summarized over WORLD voiced frames and fixed-activity AND WORLD-voiced frames separately for N and T",
        "paired_direction_domains": "recipient clock, recipient fixed-activity AND recipient F0>0 AND mapped donor F0>0; comparisons are donor-mapped AP minus recipient-own AP",
        "phone_category_definition": "exact normalized MFA label from the frozen WORLD phone map; no label folding",
        "record_count": len(per_record),
        "records": per_record,
        "direction_pooled_frame_weighted": aggregate_rows,
        "phone_category_pooled_exposure_and_voicing": phone_rows,
        "local_energy_pooled_frame_weighted": energy_rows,
    }
    payload["manifest_sha256"] = fingerprint(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-manifest", required=True, type=Path, help="full CAL WORLD producer manifest")
    parser.add_argument("--producer-source", required=True, type=Path, help="exact WORLD source file bound by the CAL manifest")
    parser.add_argument("--execution-bindings", required=True, type=Path, help="execution annex bound to the frozen physical protocol")
    parser.add_argument("--expected-execution-bindings-sha256", default=EXPECTED_EXECUTION_BINDINGS_SHA256)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-records", type=int, default=20)
    args = parser.parse_args()
    payload = build(args)
    write_json(args.out.resolve(), payload)
    print(json.dumps({"status": payload["status"], "records": payload["record_count"], "manifest_sha256": payload["manifest_sha256"], "out": str(args.out.resolve())}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
