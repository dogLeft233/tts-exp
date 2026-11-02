#!/usr/bin/env python3
"""CAL-only per-phone AP and spectral-shape dynamics audit.

Reads only the frozen 20-record WORLD calibration manifest, its natural/TTS
fixed audio and feature archives, and the already sealed phone maps/transfer
audits. It does not open evaluation records or score outcomes.
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
WORLD_MANIFEST_RAW_SHA256 = "297d3c138fc2de75e86f32db43f7d42f94476e3275c9ef9b72235c1880af07cc"
WORLD_MANIFEST_PAYLOAD_SHA256 = "75b47f39a7a462753877db6f03e41758335011764b279061d546c3e812cec29c"
WORLD_PLAN_RAW_SHA256 = "2c5d26b09c0bd3fea4812c16612387ec5c96308082c9da1e2a823ee1f6c9d8df"
WORLD_PLAN_PAYLOAD_SHA256 = "e444a0fc6b322b8e73589ce23969a0142642df76527703cab69f6f8cf9f71760"
WORLD_SOURCE_SHA256 = "7a6661afced892037d5f32fa3760e8d0906b77f40bb06f5f3ce53a66a04729b2"
EXECUTION_BINDINGS_SHA256 = "cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d"
PHONE_TABLE_SHA256 = "586d86fedd159ba1c654586bf6892d87b36b1ed8d88c666217d648d230fb2259"
PRIOR_AP_SIDECAR_SHA256 = "5ff39cfd82817f56eb9cc348e2d8481432973a09db7fd9a41ed9a280d91301c5"
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


def require_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} hash mismatch: expected={expected}, actual={actual}, path={path}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path, np: Any) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def quantiles(values: Any, np: Any) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return [float(x) for x in np.percentile(array, [10, 50, 90])]


def value_summary(values: Any, np: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    return {
        "frame_or_pair_count": int(array.size),
        "mean": float(np.mean(array)) if array.size else None,
        "p10_p50_p90": quantiles(array, np),
    }


def normalized_shape(sp: Any, np: Any) -> Any:
    power = np.sum(sp, axis=1, dtype=np.float64)
    if np.any(~np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError("WORLD SP contains a nonfinite/nonpositive frame power")
    return np.asarray(sp, dtype=np.float64) / power[:, None]


def adjacent_delta(values: Any, indices: Any, np: Any) -> Any:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < 2:
        return np.asarray([], dtype=np.float64)
    left = indices[:-1][np.diff(indices) == 1]
    right = left + 1
    if not len(left):
        return np.asarray([], dtype=np.float64)
    log_values = np.log(np.maximum(np.asarray(values, dtype=np.float64), 1e-12))
    return np.mean(np.abs(log_values[right] - log_values[left]), axis=1, dtype=np.float64)


def band_masks(frequencies: Any, np: Any) -> dict[str, Any]:
    result = {}
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            raise ValueError(f"empty WORLD frequency band {low}-{high}Hz")
        result[f"{low}_{high}Hz"] = mask
    return result


def summarize_condition(feature: dict[str, Any], active: Any, condition: str, np: Any) -> dict[str, Any]:
    f0 = np.asarray(feature["f0"], dtype=np.float64)
    ap = np.asarray(feature["ap"], dtype=np.float64)
    shape = normalized_shape(np.asarray(feature["sp"], dtype=np.float64), np)
    frequencies = np.asarray(feature["frequency_hz"], dtype=np.float64)
    voiced = f0 > 0.0
    domains = {
        "world_voiced": voiced,
        "fixed_activity_and_world_voiced": active & voiced,
    }
    masks = band_masks(frequencies, np)
    result: dict[str, Any] = {"condition": condition, "world_frame_count": int(len(f0)), "domains": {}}
    for domain_name, domain in domains.items():
        indices = np.flatnonzero(domain)
        domain_record: dict[str, Any] = {"frame_count": int(len(indices)), "weight_fraction_of_world_frames": float(np.mean(domain)) if domain.size else 0.0, "aperiodicity_by_band": {}, "spectral_shape_by_band": {}, "spectral_shape_temporal_dynamics": {}}
        for band, mask in masks.items():
            ap_frame = np.mean(ap[:, mask], axis=1, dtype=np.float64)
            sp_fraction = np.sum(shape[:, mask], axis=1, dtype=np.float64)
            domain_record["aperiodicity_by_band"][band] = value_summary(ap_frame[domain], np)
            domain_record["spectral_shape_by_band"][band] = value_summary(sp_fraction[domain], np)
            domain_record["spectral_shape_temporal_dynamics"][band] = value_summary(adjacent_delta(shape[:, mask], indices, np), np)
        domain_record["spectral_shape_temporal_dynamics"]["all_bins"] = value_summary(adjacent_delta(shape, indices, np), np)
        result["domains"][domain_name] = domain_record
    return result


def new_phone_accumulator() -> dict[str, Any]:
    return {
        "matched_phone_token_count": 0,
        "recipient_phone_world_frames": 0,
        "mapped_exposure_frames": 0,
        "active_mapped_exposure_frames": 0,
        "active_jointly_voiced_mapped_frames": 0,
        "recipient_voiced_mapped_frames": 0,
        "donor_voiced_mapped_frames": 0,
        "voicing_pair_counts": defaultdict(int),
        "ap_delta": {f"{low}_{high}Hz": [] for low, high in BANDS},
        "ap_own": {f"{low}_{high}Hz": [] for low, high in BANDS},
        "ap_donor_mapped": {f"{low}_{high}Hz": [] for low, high in BANDS},
        "sp_active_mapped": {f"{low}_{high}Hz": {"own_fraction": [], "donor_fraction": [], "donor_to_own_db": []} for low, high in BANDS},
        "sp_joint_voiced": {f"{low}_{high}Hz": {"own_fraction": [], "donor_fraction": [], "donor_to_own_db": []} for low, high in BANDS},
        "dynamics_active_mapped": {f"{low}_{high}Hz": {"own": [], "donor_mapped": []} for low, high in BANDS} | {"all_bins": {"own": [], "donor_mapped": []}},
        "dynamics_joint_voiced": {f"{low}_{high}Hz": {"own": [], "donor_mapped": []} for low, high in BANDS} | {"all_bins": {"own": [], "donor_mapped": []}},
        "dynamic_pair_counts_active_mapped": 0,
        "dynamic_pair_counts_joint_voiced": 0,
    }


def summarize_phone_accumulator(item: dict[str, Any], np: Any) -> dict[str, Any]:
    result = {key: int(item[key]) for key in (
        "matched_phone_token_count", "recipient_phone_world_frames", "mapped_exposure_frames",
        "active_mapped_exposure_frames", "active_jointly_voiced_mapped_frames",
        "recipient_voiced_mapped_frames", "donor_voiced_mapped_frames",
        "dynamic_pair_counts_active_mapped", "dynamic_pair_counts_joint_voiced",
    )}
    result["voicing_pair_counts"] = dict(sorted((str(k), int(v)) for k, v in item["voicing_pair_counts"].items()))
    result["active_mapped_phone_frame_fraction"] = result["active_mapped_exposure_frames"] / max(1, result["recipient_phone_world_frames"])
    result["active_jointly_voiced_phone_frame_fraction"] = result["active_jointly_voiced_mapped_frames"] / max(1, result["recipient_phone_world_frames"])
    result["aperiodicity_active_jointly_voiced_mapped"] = {}
    for band in item["ap_delta"]:
        own = item["ap_own"][band]
        donor = item["ap_donor_mapped"][band]
        delta = item["ap_delta"][band]
        result["aperiodicity_active_jointly_voiced_mapped"][band] = {
            "recipient_own_mean": float(np.mean(own)) if own else None,
            "donor_mapped_mean": float(np.mean(donor)) if donor else None,
            "donor_minus_recipient_frame_delta_p10_p50_p90": quantiles(delta, np),
            "frame_count": int(len(delta)),
        }
    result["spectral_shape_active_mapped"] = {}
    result["spectral_shape_active_jointly_voiced_mapped"] = {}
    for domain_name, domain_key in (("active_mapped", "sp_active_mapped"), ("active_jointly_voiced_mapped", "sp_joint_voiced")):
        target = result["spectral_shape_active_mapped"] if domain_name == "active_mapped" else result["spectral_shape_active_jointly_voiced_mapped"]
        for band, values in item[domain_key].items():
            own, donor, ratio = values["own_fraction"], values["donor_fraction"], values["donor_to_own_db"]
            target[band] = {
                "recipient_band_power_fraction_mean": float(np.mean(own)) if own else None,
                "donor_mapped_band_power_fraction_mean": float(np.mean(donor)) if donor else None,
                "donor_to_recipient_band_power_ratio_db_mean": float(np.mean(ratio)) if ratio else None,
                "donor_to_recipient_band_power_ratio_db_p10_p50_p90": quantiles(ratio, np),
                "frame_count": int(len(ratio)),
            }
    result["spectral_shape_temporal_dynamics"] = {}
    for domain_name, domain_key in (("active_mapped", "dynamics_active_mapped"), ("active_jointly_voiced_mapped", "dynamics_joint_voiced")):
        for band, values in item[domain_key].items():
            result["spectral_shape_temporal_dynamics"].setdefault(domain_name, {})[band] = {
                "recipient_mean_abs_consecutive_delta_log_shape": float(np.mean(values["own"])) if values["own"] else None,
                "donor_mapped_mean_abs_consecutive_delta_log_shape": float(np.mean(values["donor_mapped"])) if values["donor_mapped"] else None,
                "recipient_p10_p50_p90": quantiles(values["own"], np),
                "donor_mapped_p10_p50_p90": quantiles(values["donor_mapped"], np),
                "consecutive_pair_count": int(len(values["own"])),
            }
    return result


def validate_inputs(args: argparse.Namespace, np: Any) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    manifest_path = args.world_manifest.resolve()
    plan_path = args.world_plan.resolve()
    source_path = args.producer_source.resolve()
    execution_path = args.execution_bindings.resolve()
    table_path = args.phone_table.resolve()
    require_hash(manifest_path, WORLD_MANIFEST_RAW_SHA256, "frozen CAL WORLD manifest raw bytes")
    require_hash(plan_path, WORLD_PLAN_RAW_SHA256, "frozen CAL WORLD plan raw bytes")
    require_hash(source_path, WORLD_SOURCE_SHA256, "executed WORLD producer source")
    require_hash(execution_path, EXECUTION_BINDINGS_SHA256, "execution bindings annex")
    require_hash(table_path, PHONE_TABLE_SHA256, "CAL phone normalization table")
    manifest = load_json(manifest_path)
    plan = load_json(plan_path)
    if manifest.get("manifest_sha256") != WORLD_MANIFEST_PAYLOAD_SHA256 or fingerprint(manifest) != WORLD_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("frozen CAL WORLD manifest embedded fingerprint mismatch")
    if plan.get("manifest_sha256") != WORLD_PLAN_PAYLOAD_SHA256 or fingerprint(plan) != WORLD_PLAN_PAYLOAD_SHA256:
        raise ValueError("frozen CAL WORLD plan embedded fingerprint mismatch")
    if manifest.get("world_transfer_plan_path") != str(plan_path) or manifest.get("world_transfer_plan_sha256") != WORLD_PLAN_PAYLOAD_SHA256:
        raise ValueError("CAL manifest does not bind the frozen plan path/fingerprint")
    if manifest.get("split") != "calibration" or manifest.get("manifest_type") != "world_phone_transfer_calibration_producer":
        raise ValueError("input manifest is not the frozen CAL-only WORLD producer manifest")
    if manifest.get("evaluation_outcomes_read") is not False or manifest.get("evaluation_waveforms_read") is not False:
        raise ValueError("CAL manifest has an EVAL access marker")
    if plan.get("evaluation_outcomes_read") is not False or plan.get("status") != "SEALED_BEFORE_ENDPOINT_FORWARD":
        raise ValueError("CAL plan status/outcome marker differs from the frozen protocol")
    if manifest.get("phone_normalizer_source_sha256") != WORLD_SOURCE_SHA256 or manifest.get("phone_normalization_table_sha256") != PHONE_TABLE_SHA256:
        raise ValueError("CAL phone normalization source/table binding mismatch")
    if Path(str(manifest.get("phone_normalization_table_path", ""))).resolve() != table_path:
        raise ValueError("CAL manifest does not bind the supplied phone normalization table path")
    if Path(str(plan.get("world_producer_source", ""))).resolve() != source_path or plan.get("world_producer_sha256") != WORLD_SOURCE_SHA256:
        raise ValueError("CAL plan does not bind the supplied WORLD producer source path/hash")
    if set(manifest.get("arms", [])) != {"RT", "EX", "SP", "BOTH", "F0", "AP", "EX_EM", "SP_EM", "E"}:
        raise ValueError("CAL manifest arm inventory differs from the frozen nine-arm plan")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 20 or len({str(r.get("id")) for r in records}) != 20:
        raise ValueError("expected exactly 20 unique frozen CAL records")
    for record in records:
        if record.get("split") != "calibration" or record.get("pair_qualified") is not True:
            raise ValueError(f"unqualified/non-CAL record in sealed input manifest: {record.get('id')}")
        for condition in ("natural", "tts"):
            info = record[condition]
            feature_meta, audio_meta = info["world_features"], info["fixed_audio"]
            require_hash(Path(feature_meta["path"]), feature_meta["file_sha256"], f"{record['id']}/{condition} WORLD features")
            require_hash(Path(audio_meta["path"]), audio_meta["file_sha256"], f"{record['id']}/{condition} fixed PCM")
        for direction_name in DIRECTIONS:
            direction = record["directions"][direction_name]
            require_hash(Path(direction["phone_map"]["path"]), direction["phone_map"]["file_sha256"], f"{record['id']}/{direction_name} phone map")
            require_hash(Path(direction["parameter_transfer_audit"]["path"]), direction["parameter_transfer_audit"]["file_sha256"], f"{record['id']}/{direction_name} transfer audit")
    return manifest, plan, source_path, execution_path


def load_producer_module(source_path: Path, np: Any) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_world_source_for_phone_spectral_detail", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import bound WORLD producer source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "local_rms_envelope", None)):
        raise RuntimeError("bound WORLD producer does not expose local_rms_envelope")
    return module


def build(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    manifest, plan, source_path, execution_path = validate_inputs(args, np)
    if args.preflight_only:
        return {
            "status": "PREFLIGHT_PASS",
            "split": "calibration",
            "record_count": len(manifest["records"]),
            "evaluation_waveforms_read": False,
            "evaluation_outcomes_read": False,
            "world_manifest_raw_sha256": sha256_file(args.world_manifest.resolve()),
            "world_manifest_payload_sha256": manifest["manifest_sha256"],
            "world_plan_raw_sha256": sha256_file(args.world_plan.resolve()),
            "world_plan_payload_sha256": plan["manifest_sha256"],
            "producer_source_sha256": sha256_file(source_path),
            "execution_bindings_sha256": sha256_file(execution_path),
        }

    producer = load_producer_module(source_path, np)
    pooled_conditions: dict[str, Any] = {}
    category_pooled: dict[tuple[str, str], dict[str, Any]] = {}
    records_out = []
    input_file_count = 0
    for record in manifest["records"]:
        sample_id = str(record["id"])
        side_data = {}
        condition_out = {}
        for condition in ("natural", "tts"):
            info = record[condition]
            feature_meta, audio_meta = info["world_features"], info["fixed_audio"]
            feature = load_npz(Path(feature_meta["path"]), np)
            audio, sample_rate = sf.read(str(audio_meta["path"]), dtype="float32", always_2d=True)
            input_file_count += 2
            if int(sample_rate) != SAMPLE_RATE or audio.shape[1] != 1 or audio.shape[0] != int(info["raw_decoded_samples"]):
                raise ValueError(f"fixed PCM clock mismatch: {sample_id}/{condition}")
            if hashlib.sha256(np.asarray(audio[:, 0], dtype="<f4").tobytes(order="C")).hexdigest() != audio_meta["float32_pcm_sha256"]:
                raise ValueError(f"fixed PCM decoded array hash mismatch: {sample_id}/{condition}")
            required = {"time", "f0", "sp", "ap", "frequency_hz"}
            if not required.issubset(feature):
                raise ValueError(f"WORLD feature key inventory incomplete: {sample_id}/{condition}")
            times = np.asarray(feature["time"], dtype=np.float64)
            f0 = np.asarray(feature["f0"], dtype=np.float64)
            sp = np.asarray(feature["sp"], dtype=np.float64)
            ap = np.asarray(feature["ap"], dtype=np.float64)
            frequencies = np.asarray(feature["frequency_hz"], dtype=np.float64)
            if times.ndim != 1 or f0.shape != times.shape or sp.ndim != 2 or ap.shape != sp.shape or sp.shape[0] != len(times) or frequencies.shape != (sp.shape[1],):
                raise ValueError(f"WORLD feature dimensions differ: {sample_id}/{condition}")
            if not all(np.isfinite(x).all() for x in (times, f0, sp, ap, frequencies)) or np.any(sp < 0.0):
                raise ValueError(f"WORLD feature contains nonfinite or negative SP values: {sample_id}/{condition}")
            if ap.size and (float(np.min(ap)) < -1e-6 or float(np.max(ap)) > 1.0 + 1e-6):
                raise ValueError(f"WORLD AP outside [0,1] tolerance: {sample_id}/{condition}")
            audio_1d = np.asarray(audio[:, 0], dtype=np.float32)
            env_times, env = producer.local_rms_envelope(audio_1d, np)
            env_at_world = np.interp(times, env_times, env, left=float(env[0]), right=float(env[-1]))
            active = env_at_world >= 10.0 ** (ACTIVE_THRESHOLD_DBFS / 20.0)
            side_data[condition] = {"feature": feature, "audio": audio_1d, "active": active}
            condition_out[condition] = summarize_condition(feature, active, condition, np)

        if not np.array_equal(np.asarray(side_data["natural"]["feature"]["frequency_hz"]), np.asarray(side_data["tts"]["feature"]["frequency_hz"])):
            raise ValueError(f"N/T WORLD frequency grids differ: {sample_id}")

        direction_out = {}
        for direction_name, rule in DIRECTIONS.items():
            direction = record["directions"][direction_name]
            if direction.get("recipient_condition") != rule["recipient"] or direction.get("donor_condition") != rule["donor"]:
                raise ValueError(f"direction identity mismatch: {sample_id}/{direction_name}")
            recipient = side_data[rule["recipient"]]
            donor = side_data[rule["donor"]]
            rec_feature, donor_feature = recipient["feature"], donor["feature"]
            rec_times = np.asarray(rec_feature["time"], dtype=np.float64)
            rec_f0 = np.asarray(rec_feature["f0"], dtype=np.float64)
            rec_ap = np.asarray(rec_feature["ap"], dtype=np.float64)
            rec_sp = np.asarray(rec_feature["sp"], dtype=np.float64)
            frequencies = np.asarray(rec_feature["frequency_hz"], dtype=np.float64)
            own_shape = normalized_shape(rec_sp, np)
            masks = band_masks(frequencies, np)
            phone_meta, audit_meta = direction["phone_map"], direction["parameter_transfer_audit"]
            phone_map = load_json(Path(phone_meta["path"]))
            audit = load_npz(Path(audit_meta["path"]), np)
            input_file_count += 2
            phone_duration = float(phone_map["recipient_audio_duration_s"])
            decoded_duration = len(recipient["audio"]) / SAMPLE_RATE
            if abs(phone_duration - decoded_duration) > 1e-9:
                raise ValueError(f"phone-map duration differs from decoded recipient PCM: {sample_id}/{direction_name}")
            if len(phone_map.get("matched_phones", [])) != int(phone_meta["matched_phone_count"]):
                raise ValueError(f"phone map count differs from its manifest record: {sample_id}/{direction_name}")
            mapped = np.asarray(audit["mapped_frames"], dtype=bool)
            mapped_ap = np.asarray(audit["mapped_ap"], dtype=np.float64)
            mapped_shape = np.asarray(audit["mapped_shape"], dtype=np.float64)
            donor_voiced = np.asarray(audit["donor_voiced"], dtype=bool)
            joint = np.asarray(audit["joint_voiced"], dtype=bool)
            if mapped.shape != rec_times.shape or donor_voiced.shape != rec_times.shape or joint.shape != rec_times.shape:
                raise ValueError(f"transfer audit frame clock mismatch: {sample_id}/{direction_name}")
            if mapped_ap.shape != rec_ap.shape or mapped_shape.shape != rec_sp.shape:
                raise ValueError(f"transfer audit spectral shape mismatch: {sample_id}/{direction_name}")
            if not np.isfinite(mapped_ap).all() or not np.isfinite(mapped_shape).all():
                raise ValueError(f"transfer audit contains nonfinite arrays: {sample_id}/{direction_name}")
            expected_joint = mapped & (rec_f0 > 0.0) & donor_voiced
            if not np.array_equal(joint, expected_joint):
                raise ValueError(f"joint-voicing audit differs from reconstructed CAL domain: {sample_id}/{direction_name}")
            if mapped.any():
                row_sums = np.sum(mapped_shape[mapped], axis=1, dtype=np.float64)
                if float(np.max(np.abs(row_sums - 1.0), initial=0.0)) > 1e-6:
                    raise ValueError(f"mapped spectral shapes are not normalized: {sample_id}/{direction_name}")
            active = recipient["active"]
            active_mapped = active & mapped
            active_joint = active & mapped & joint
            pooled_row = {"matched_phone_count": int(len(phone_map["matched_phones"])), "recipient_world_frame_count": int(len(rec_times)),
                          "mapped_frame_count": int(np.count_nonzero(mapped)), "active_mapped_frame_count": int(np.count_nonzero(active_mapped)),
                          "active_jointly_voiced_mapped_frame_count": int(np.count_nonzero(active_joint)), "phone_categories": int(len({str(p["recipient_normalized_label"]) for p in phone_map["matched_phones"]}))}
            category_names = set()
            for phone in phone_map["matched_phones"]:
                label = str(phone["recipient_normalized_label"])
                category_names.add(label)
                category = category_pooled.setdefault((direction_name, label), new_phone_accumulator())
                start, end = map(float, phone["recipient_interval_s"])
                in_phone = (rec_times >= start) & (rec_times < end)
                if abs(end - phone_duration) <= 1e-9:
                    in_phone |= np.abs(rec_times - end) <= 1e-9
                phone_frames = int(np.count_nonzero(in_phone))
                map_mask = in_phone & mapped
                active_map_mask = in_phone & active_mapped
                ap_mask = in_phone & active_joint
                category["matched_phone_token_count"] += 1
                category["recipient_phone_world_frames"] += phone_frames
                category["mapped_exposure_frames"] += int(np.count_nonzero(map_mask))
                category["active_mapped_exposure_frames"] += int(np.count_nonzero(active_map_mask))
                category["active_jointly_voiced_mapped_frames"] += int(np.count_nonzero(ap_mask))
                mapped_indices = np.flatnonzero(map_mask)
                for ix in mapped_indices:
                    rs, ds = bool(rec_f0[ix] > 0.0), bool(donor_voiced[ix])
                    category["voicing_pair_counts"][f"recipient_{'voiced' if rs else 'unvoiced'}_donor_{'voiced' if ds else 'unvoiced'}"] += 1
                category["recipient_voiced_mapped_frames"] += int(np.count_nonzero(map_mask & (rec_f0 > 0.0)))
                category["donor_voiced_mapped_frames"] += int(np.count_nonzero(map_mask & donor_voiced))

                for band, freq_mask in masks.items():
                    own_ap_band = np.mean(rec_ap[:, freq_mask], axis=1, dtype=np.float64)
                    donor_ap_band = np.mean(mapped_ap[:, freq_mask], axis=1, dtype=np.float64)
                    category["ap_own"][band].extend(own_ap_band[ap_mask].tolist())
                    category["ap_donor_mapped"][band].extend(donor_ap_band[ap_mask].tolist())
                    category["ap_delta"][band].extend((donor_ap_band[ap_mask] - own_ap_band[ap_mask]).tolist())

                    own_fraction = np.sum(own_shape[:, freq_mask], axis=1, dtype=np.float64)
                    donor_fraction = np.sum(mapped_shape[:, freq_mask], axis=1, dtype=np.float64)
                    ratio_db = 10.0 * np.log10(np.maximum(donor_fraction, 1e-12) / np.maximum(own_fraction, 1e-12))
                    for key, mask in (("sp_active_mapped", active_map_mask), ("sp_joint_voiced", ap_mask)):
                        category[key][band]["own_fraction"].extend(own_fraction[mask].tolist())
                        category[key][band]["donor_fraction"].extend(donor_fraction[mask].tolist())
                        category[key][band]["donor_to_own_db"].extend(ratio_db[mask].tolist())

                    for key, mask in (("dynamics_active_mapped", active_map_mask), ("dynamics_joint_voiced", ap_mask)):
                        indices = np.flatnonzero(mask)
                        own_delta = adjacent_delta(own_shape[:, freq_mask], indices, np)
                        donor_delta = adjacent_delta(mapped_shape[:, freq_mask], indices, np)
                        category[key][band]["own"].extend(own_delta.tolist())
                        category[key][band]["donor_mapped"].extend(donor_delta.tolist())

                for key, mask in (("dynamics_active_mapped", active_map_mask), ("dynamics_joint_voiced", ap_mask)):
                    indices = np.flatnonzero(mask)
                    own_delta = adjacent_delta(own_shape, indices, np)
                    donor_delta = adjacent_delta(mapped_shape, indices, np)
                    category[key]["all_bins"]["own"].extend(own_delta.tolist())
                    category[key]["all_bins"]["donor_mapped"].extend(donor_delta.tolist())
                    if key == "dynamics_active_mapped":
                        category["dynamic_pair_counts_active_mapped"] += len(own_delta)
                    else:
                        category["dynamic_pair_counts_joint_voiced"] += len(own_delta)

            direction_out[direction_name] = {
                "recipient_condition": rule["recipient"], "donor_condition": rule["donor"],
                "contrast": rule["contrast"], **pooled_row,
                "active_mapped_fraction_of_recipient_clock": float(np.mean(active_mapped)) if active_mapped.size else 0.0,
                "active_jointly_voiced_mapped_fraction_of_recipient_clock": float(np.mean(active_joint)) if active_joint.size else 0.0,
            }
        records_out.append({"id": sample_id, "speaker": str(record["speaker"]), "source_identity_sha256": record["source_identity_sha256"],
                            "condition_domains": condition_out, "directions": direction_out})

    phone_rows = []
    for (direction_name, label), accumulator in sorted(category_pooled.items()):
        phone_rows.append({"direction": direction_name, "contrast": DIRECTIONS[direction_name]["contrast"],
                           "recipient_condition": DIRECTIONS[direction_name]["recipient"], "donor_condition": DIRECTIONS[direction_name]["donor"],
                           "phone_category": label, "definition": "exact normalized recipient MFA phone label; no folding",
                           **summarize_phone_accumulator(accumulator, np)})

    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": "world_calibration_per_phone_ap_and_spectral_shape_dynamics",
        "status": "CAL_ONLY_READ_ONLY_DIAGNOSTIC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "calibration",
        "evaluation_waveforms_read": False,
        "evaluation_outcomes_read": False,
        "effects_read": False,
        "world_manifest_path": str(args.world_manifest.resolve()),
        "world_manifest_raw_sha256": sha256_file(args.world_manifest.resolve()),
        "world_manifest_payload_sha256": manifest.get("manifest_sha256"),
        "world_plan_path": str(args.world_plan.resolve()),
        "world_plan_raw_sha256": sha256_file(args.world_plan.resolve()),
        "world_plan_payload_sha256": plan.get("manifest_sha256"),
        "world_producer_source_path": str(source_path),
        "world_producer_source_sha256": sha256_file(source_path),
        "execution_bindings_path": str(execution_path),
        "execution_bindings_sha256": sha256_file(execution_path),
        "phone_normalization_table_path": str(args.phone_table.resolve()),
        "phone_normalization_table_sha256": sha256_file(args.phone_table.resolve()),
        "prior_calibration_ap_sidecar_sha256": PRIOR_AP_SIDECAR_SHA256,
        "sample_count": len(records_out),
        "input_file_count_verified": input_file_count,
        "activity_rule": {"source": "recipient fixed PCM; frozen producer local_rms_envelope", "threshold_dbfs": ACTIVE_THRESHOLD_DBFS, "window_samples": 801, "hop_samples": 80},
        "aperiodicity_bands_hz": [[lo, hi] for lo, hi in BANDS],
        "spectral_shape_rule": "SP divided by each frame's total SP power; donor mapped shape is the sealed audit mapped_shape; band share is summed normalized power; ratio dB is 10*log10(donor/recipient)",
        "spectral_dynamics_rule": "mean absolute adjacent-frame delta of log normalized SP over frequency bins; pairs must be consecutive WORLD frame indices inside the same matched phone and selected domain",
        "phone_ap_domain": "matched recipient phone interval AND recipient fixed activity >= -45 dBFS AND mapped donor frame AND recipient/donor jointly voiced",
        "phone_sp_domains": ["matched recipient phone interval AND recipient fixed activity AND mapped donor frame", "same, additionally recipient/donor jointly voiced"],
        "per_condition_domains": "natural and TTS own AP/SP summarized separately for WORLD voiced and fixed-activity AND WORLD-voiced frames",
        "records": records_out,
        "pooled_exact_phone_category_results": phone_rows,
    }
    payload["manifest_sha256"] = fingerprint(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-manifest", required=True, type=Path)
    parser.add_argument("--world-plan", required=True, type=Path)
    parser.add_argument("--producer-source", required=True, type=Path)
    parser.add_argument("--execution-bindings", required=True, type=Path)
    parser.add_argument("--phone-table", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.out is None:
        parser.error("--out is required unless --preflight-only is used")
    payload = build(args)
    if args.preflight_only:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": payload["status"], "records": payload["sample_count"], "phone_category_rows": len(payload["pooled_exact_phone_category_results"]), "manifest_sha256": payload["manifest_sha256"], "out": str(out)}, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
