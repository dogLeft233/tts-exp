"""Protocol runner for the repaired static-TFG/MFA experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.experiments.phone_separability_enhancement.audio import BandGainRenderer, export_pcm, make_protected_mask
from scripts.experiments.phone_separability_enhancement.teacher import load_frozen_teacher

from .assets import import_parent_assets, load_registry, read_pcm16, rows_by_split, write_pcm16
from .conditioning import PhoneVocabulary, fit_duration_stats
from .config import REPO_ROOT, canonical_hash, ensure_protocol_unchanged, file_sha256, load_yaml, protocol_snapshot, proxy_environment, read_json, read_jsonl, resource_decision, resource_snapshot, state_dict_hash, validate_run_id, write_json, write_jsonl
from .phone_eval import score_candidate_set, score_waveform, summarize_pairwise
from .report import write_report
from .support import entries_for, freeze_phone_support, load_support
from .train import _normalize_features, load_checkpoint, make_training_sample, train_arm

STAGES = ("audit", "calibrate", "train", "lock", "infer", "quality", "phone", "render", "score", "official", "analyze", "report")
ARM_MODES = {"A": "AUDIO_FEATURES", "B": "BOUNDARY_TIME", "C": "MFA_PHONE_TIME"}


def _run_dir(run_id: str) -> Path:
    return REPO_ROOT / "runs" / f"phone_gain_static_tfg_mfa_{validate_run_id(run_id)}"


def _status_path(root: Path) -> Path:
    return root / "status.json"


def _load_status(root: Path) -> dict[str, Any]:
    path = _status_path(root)
    return read_json(path) if path.is_file() else {"schema_version": 1, "execution": "RUNNING", "states": {}}


def _set_state(root: Path, stage: str, state: str, *, reasons: list[str] | None = None, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    status = _load_status(root)
    status.setdefault("states", {})[stage] = {"state": state, "reasons": list(reasons or []), "updated_at": time.time(), **dict(details or {})}
    states = status["states"]
    complete_states = {"COMPLETE", "NOT_APPLICABLE"}
    status["execution"] = "COMPLETE" if stage == "report" and state == "COMPLETE" and all(str(item.get("state")) in complete_states for item in states.values()) else ("PARTIAL" if stage == "report" else "RUNNING")
    protocol_path = root / "protocol.json"
    status["protocol_id"] = read_json(protocol_path).get("protocol_id", "phone_gain_static_tfg_mfa_v1") if protocol_path.is_file() else "phone_gain_static_tfg_mfa_v1"
    write_json(_status_path(root), status)
    return status


def _load_protocol(config: Mapping[str, Any], config_path: Path, root: Path, *, resume: bool) -> dict[str, Any]:
    current = protocol_snapshot(config, config_path)
    path = root / "protocol.json"
    if path.is_file():
        previous = read_json(path)
        if resume:
            ensure_protocol_unchanged(previous, current)
        else:
            raise ValueError(f"run already exists; pass --resume: {root}")
    else:
        write_json(path, current)
    return current


def _assert_protocol(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Require every standalone stage to use the same immutable protocol lock."""
    path = root / "protocol.json"
    if not path.is_file():
        raise RuntimeError("PROTOCOL_LOCK_MISSING; run audit first")
    previous = read_json(path)
    config_path = Path(str(previous.get("config_path", "")))
    if not config_path.is_file():
        raise RuntimeError("PROTOCOL_CONFIG_MISSING")
    current = protocol_snapshot(config, config_path)
    ensure_protocol_unchanged(previous, current)
    return previous


def _load_conditioning_config(root: Path) -> tuple[PhoneVocabulary, dict[str, float]]:
    payload = read_json(root / "01_calibration/conditioning.json")
    return PhoneVocabulary.from_labels(payload["labels"]), dict(payload["duration_stats"])


def _load_feature_stats(root: Path) -> dict[str, Any]:
    payload = read_json(root / "01_calibration/conditioning.json")
    stats = payload.get("feature_stats")
    if not isinstance(stats, Mapping) or not stats.get("mean") or not stats.get("std"):
        raise RuntimeError("FIT_FEATURE_STATS_MISSING")
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    if mean.size != std.size or mean.size == 0 or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-6):
        raise RuntimeError("FIT_FEATURE_STATS_INVALID")
    return {"mean": mean.tolist(), "std": std.tolist(), "n_frames": int(stats.get("n_frames", 0)), "hash": canonical_hash({"mean": mean.tolist(), "std": std.tolist(), "n_frames": int(stats.get("n_frames", 0))})}


def _calibration_payload(root: Path) -> dict[str, Any]:
    path = root / "01_calibration/calibration.json"
    return read_json(path) if path.is_file() else {"status": "NOT_RUN"}


def _require_calibration(root: Path, stage: str) -> bool:
    payload = _calibration_payload(root)
    if payload.get("status") == "PASS":
        return True
    reason = f"CALIBRATION_NOT_PASS:{payload.get('status', 'NOT_RUN')}"
    _set_state(root, stage, "DEPENDENCY_BLOCKED", reasons=[reason])
    return False


def stage_audit(config: Mapping[str, Any], config_path: Path, root: Path, *, resume: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    protocol = _load_protocol(config, config_path, root, resume=resume)
    resource = {"cpu": resource_snapshot(), "gpu_for_detection": resource_decision(config, stage="audit", gpu_required=bool(config.get("static_geometry", False)))}
    if resource["gpu_for_detection"]["decision"] != "PASS":
        _set_state(root, "audit", "RESOURCE_WAIT", reasons=resource["gpu_for_detection"]["reasons"], details=resource)
        write_json(root / "00_protocol/resources.json", resource)
        return resource
    write_json(root / "00_protocol/resources.json", resource)
    try:
        registry = import_parent_assets(config, root)
        support = freeze_phone_support(config, root, registry)
        parent_lock = read_json(REPO_ROOT / config["source"]["parent_lock"])
        write_json(root / "00_protocol/exposure_ledger.json", {"parent_lock": parent_lock, "e_seen_status": "EXPLORATORY_HISTORIC_EXPOSURE", "selection_before_tfg": True, "video_paths_in_provenance_only": True})
        old_run = REPO_ROOT / str(config.get("source", {}).get("old_run", "")) if config.get("source", {}).get("old_run") else None
        audit_findings = [
            {"id": "F01", "priority": "P0", "status": "REPAIRED_IN_NEW_SCOPE", "evidence": "assets.py/_portrait_meta", "target": "detected_face_geometry"},
            {"id": "F02", "priority": "P0", "status": "PENDING_INSTRUMENT_CALIBRATION", "evidence": "01_calibration/calibration.json", "target": "sync_delay_repeat_mfa_processor"},
            {"id": "F03", "priority": "P0", "status": "REPAIRED", "evidence": "phone_eval.py", "target": "natural_primary_excludes_T"},
            {"id": "F04", "priority": "P0", "status": "REPAIRED_PARTIAL", "evidence": "support.py", "target": "eligibility_and_runtime_frame_indices"},
            {"id": "F05", "priority": "P0", "status": "REPAIRED", "evidence": "protocol_snapshot_and_cache_bindings", "target": "dependency_bound_cache"},
            {"id": "F06", "priority": "P0", "status": "REPAIRED", "evidence": "analyze.py/check.py", "target": "expected_scope_and_inconclusive_propagation"},
            {"id": "F07", "priority": "P1", "status": "REPAIRED", "evidence": "run.py/_dev_samples", "target": "parent_pilot24_per_seed"},
            {"id": "F08", "priority": "P1", "status": "REPAIRED", "evidence": "run.py/_select_dev_candidate", "target": "zero_residual_and_step_tie"},
            {"id": "F09", "priority": "P1", "status": "REPAIRED_PARTIAL", "evidence": "conditioning.py/train.py/infer.py", "target": "fit_mask_and_pcm_export"},
            {"id": "F10", "priority": "P1", "status": "REPAIRED_PARTIAL", "evidence": "train.py", "target": "epoch_rng_cursor_budget"},
            {"id": "F11", "priority": "P1", "status": "REPAIRED_PARTIAL", "evidence": "run.py/asr.py", "target": "all_arm_quality"},
            {"id": "F12", "priority": "P1", "status": "REPAIRED_PARTIAL", "evidence": "quality.py/mfa.py", "target": "levenshtein_and_nonempty_phone_tier"},
            {"id": "F13", "priority": "P1", "status": "REPAIRED_FOR_NEW_EXPORT_ONLY", "evidence": "audio_contract.py", "target": "pcm_domain_contract"},
            {"id": "F14", "priority": "P1", "status": "REPAIRED_PARTIAL", "evidence": "sync_support.py/analyze.py", "target": "d_anchor_and_layered_status"},
            {"id": "F15", "priority": "P1", "status": "REPAIRED_PARTIAL", "evidence": "run.py/render_portraits", "target": "portrait_scope"},
            {"id": "F16", "priority": "P1", "status": "REPAIRED_PARTIAL", "evidence": "render.py/official_score.py", "target": "semantic_cell_key_and_input_hash"},
            {"id": "F17", "priority": "P2", "status": "REPAIRED_PARTIAL", "evidence": "official_score.py/static_render_worker", "target": "strict_parser_gpu_gate_streaming"},
        ]
        spec_path = REPO_ROOT / "openspec/changes/repair-phone-gain-static-tfg-mfa/spec.md"
        write_json(root / "00_protocol/audit_findings.json", {"schema_version": 1, "spec": {"path": str(spec_path.resolve()), "sha256": file_sha256(spec_path) if spec_path.is_file() else None}, "findings": audit_findings, "old_run": {"path": str(old_run.resolve()) if old_run else None, "status": "READ_ONLY" if old_run and old_run.is_dir() else "NOT_CONFIGURED"}})
        expected_pairs = [str(row["pair_id"]) for row in rows_by_split(registry, "e_seen")]
        portraits = [str(value) for value in config.get("tfg", {}).get("render_portraits", ["3"])]
        expected_cells = [{"pair_id": pair_id, "portrait_id": portrait, "seed": int(config.get("tfg", {}).get("seed", 42)), "video_arm": arm, "audio_arm": arm} for pair_id in expected_pairs for portrait in portraits for arm in ("N", "T", "D", "A", "B", "C")]
        write_json(root / "00_protocol/expected_artifacts.json", {"schema_version": 1, "scope": config.get("experiment_mode", "repair"), "pairs": expected_pairs, "portraits": portraits, "expected_render_cells": expected_cells, "expected_sync_cells_per_pair": [f"V_{v}/A_{a}" for v, a in (("N", "N"), ("D", "N"), ("A", "N"), ("B", "N"), ("C", "N"), ("N", "D"), ("N", "A"), ("N", "B"), ("N", "C"), ("D", "D"), ("A", "A"), ("B", "B"), ("C", "C"), ("T", "T"))], "expected_sync_pairs": len(expected_pairs)})
        write_json(root / "00_protocol/protocol_lock.json", {"protocol_sha256": __import__("hashlib").sha256((root / "protocol.json").read_bytes()).hexdigest(), "protocol": protocol, "support_hash": support["support_hash"], "cohort": registry["split_counts"], "resource_at_audit": resource})
    except Exception as exc:
        _set_state(root, "audit", "ENGINEERING_FAILURE", reasons=[f"{type(exc).__name__}:{exc}"])
        raise
    _set_state(root, "audit", "COMPLETE", details={"pairs": len(registry["rows"]), "support_hash": support["support_hash"], "resource_snapshot": resource})
    return registry


def stage_calibrate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _assert_protocol(config, root)
    registry = load_registry(root)
    support = load_support(root)
    fit_rows = rows_by_split(registry, "fit")
    vocab = PhoneVocabulary.from_labels(support["labels"])
    duration_stats = fit_duration_stats([row["natural"] for row in fit_rows])
    write_json(root / "01_calibration/conditioning.json", {"labels": list(vocab.labels), "vocabulary_hash": vocab.hash, "duration_stats": duration_stats})
    from .calibration import fit_feature_stats, run_pcm_calibration, run_teacher_calibration, run_timing_calibration

    result: dict[str, Any] = {"schema_version": 2, "status": "PARTIAL_CALIBRATION", "vocabulary_hash": vocab.hash, "duration_stats": duration_stats, "phone_label_count": len(vocab.labels)}
    try:
        feature_stats = fit_feature_stats(config, registry)
        result["feature_stats"] = {"status": "PASS", **feature_stats}
    except Exception as exc:
        feature_stats = None
        result["feature_stats"] = {"status": "ENGINEERING_FAILURE", "reason": f"{type(exc).__name__}:{exc}"}
    write_json(root / "01_calibration/conditioning.json", {"labels": list(vocab.labels), "vocabulary_hash": vocab.hash, "duration_stats": duration_stats, "feature_stats": feature_stats})
    try:
        result["teacher"] = run_teacher_calibration(config, root, registry, support)
    except Exception as exc:
        result["teacher"] = {"status": "ENGINEERING_FAILURE", "reason": f"{type(exc).__name__}:{exc}"}
    try:
        result["stft_identity"] = run_pcm_calibration(config, root, registry)
    except Exception as exc:
        result["stft_identity"] = {"status": "ENGINEERING_FAILURE", "reason": f"{type(exc).__name__}:{exc}"}
    try:
        result["timing_controls"] = run_timing_calibration(config, root, registry)
    except Exception as exc:
        result["timing_controls"] = {"status": "ENGINEERING_FAILURE", "reason": f"{type(exc).__name__}:{exc}"}
    result["processor_parity"] = result["teacher"].get("processor_parity", {"status": "NOT_RUN", "reason": result["teacher"].get("reason", "TEACHER_NOT_RUN")})
    required = (result["teacher"], result["stft_identity"], result["timing_controls"], result["feature_stats"])
    result["status"] = "PASS" if all(item.get("status") == "PASS" for item in required) else ("RESOURCE_WAIT" if any(item.get("status") == "RESOURCE_WAIT" for item in required) else "PARTIAL_CALIBRATION")
    write_json(root / "01_calibration/calibration.json", result)
    state = "COMPLETE" if result["status"] == "PASS" else ("RESOURCE_WAIT" if result["status"] == "RESOURCE_WAIT" else "PARTIAL")
    reasons = [] if state == "COMPLETE" else [f"{key}:{value.get('status', 'UNKNOWN')}" for key, value in (("teacher", result["teacher"]), ("stft_identity", result["stft_identity"]), ("timing_controls", result["timing_controls"]), ("feature_stats", result["feature_stats"])) if value.get("status") != "PASS"]
    _set_state(root, "calibrate", state, reasons=reasons)
    return result


def _make_fit_samples(registry: Mapping[str, Any], support: Mapping[str, Any], encoder: str = "hubert") -> list[dict[str, Any]]:
    result = []
    for row in rows_by_split(registry, "fit"):
        result.append(make_training_sample(row, [item for item in entries_for(support, encoder, "natural_primary", "fit") if str(item["pair_id"]) == str(row["pair_id"]) and bool(item.get("pair_eligible", True))]))
    return result


def stage_train(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _assert_protocol(config, root)
    if str(config.get("experiment_mode", "repair")) == "remeasure":
        result = {"status": "NOT_APPLICABLE", "reason": "FROZEN_AUDIO_REMEASUREMENT"}
        write_json(root / "02_train/summary.json", result)
        _set_state(root, "train", "NOT_APPLICABLE", details=result)
        return result
    if not _require_calibration(root, "train"):
        return {"status": "DEPENDENCY_BLOCKED", "reason": "CALIBRATION_NOT_PASS"}
    decision = resource_decision(config, stage="train", gpu_required=True)
    write_json(root / "02_train/resource_preflight.json", decision)
    if decision["decision"] != "PASS":
        _set_state(root, "train", "RESOURCE_WAIT", reasons=decision["reasons"], details={"resource_snapshot": decision["snapshot"]})
        return decision
    registry = load_registry(root)
    support = load_support(root)
    vocab, duration_stats = _load_conditioning_config(root)
    feature_stats = _load_feature_stats(root)
    device = str(config["runtime"].get("device", config["models"].get("device", "cuda:0")))
    try:
        with proxy_environment(config["runtime"].get("proxy")):
            teacher = load_frozen_teacher(config["models"]["primary"], device=device, proxy=config["runtime"].get("proxy"), allow_download=bool(config["runtime"].get("allow_model_download", True)))
        samples = _make_fit_samples(registry, support)
        training_contract = {
            "protocol_id": config["protocol_id"],
            "support_hash": support["support_hash"],
            "teacher_state_hash": state_dict_hash(teacher.model.state_dict()),
            "teacher_processor": teacher.processor_info,
            "feature_stats_hash": feature_stats["hash"],
            "fit_samples": [{"pair_id": sample["pair_id"], "pcm_sha256": sample["audio_meta"]["pcm_sha256"], "support_keys": sorted(str(item.get("support_key")) for item in sample.get("support_entries", []))} for sample in samples],
        }
        results = []
        for mode_name, mode in ARM_MODES.items():
            for seed in config["training"]["seeds"]:
                result = train_arm(samples, teacher, support["mixed_centroids"]["hubert"], mode=mode, vocabulary=vocab, duration_stats=duration_stats, seed=int(seed), output_dir=root / "02_train" / mode_name / f"seed_{int(seed)}", network=config["network"], training=config["training"], audio=config["audio"], device=device, resume=True, contract=training_contract, feature_stats=feature_stats)
                results.append(result)
        status = "COMPLETE" if results and all(str(item.get("status")) == "COMPLETE" and int(item.get("optimizer_updates", 0)) >= int(config["training"].get("max_optimizer_updates", 400)) for item in results) else "PARTIAL"
        write_json(root / "02_train/summary.json", {"status": status, "results": results})
        _set_state(root, "train", status)
        return {"status": status, "results": results}
    except Exception as exc:
        write_json(root / "02_train/summary.json", {"status": "ENGINEERING_FAILURE", "error": f"{type(exc).__name__}:{exc}"})
        _set_state(root, "train", "ENGINEERING_FAILURE", reasons=[f"{type(exc).__name__}:{exc}"])
        raise


def _checkpoint_candidates(root: Path, arm: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for seed_dir in sorted((root / "02_train" / arm).glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        for path in sorted(seed_dir.glob("step_*.pt")):
            try:
                step = int(path.stem.removeprefix("step_"))
            except ValueError:
                continue
            candidates.append({"seed": seed, "step": step, "path": path})
        last = seed_dir / "last.pt"
        if last.is_file():
            try:
                import torch

                payload = torch.load(str(last), map_location="cpu", weights_only=False)
                step = int(payload.get("optimizer_updates", -1))
                if step >= 0 and not any(item["path"] == last for item in candidates):
                    candidates.append({"seed": seed, "step": step, "path": last, "budget_terminal": True})
            except (OSError, RuntimeError, ValueError, KeyError):
                continue
    return sorted(candidates, key=lambda item: (int(item["seed"]), int(item["step"])))


def _dev_samples(config: Mapping[str, Any], registry: Mapping[str, Any], support: Mapping[str, Any]) -> list[dict[str, Any]]:
    pilot_path = REPO_ROOT / config["source"].get("parent_pilot", "runs/phone_separability_enhancement_audit_v2_20260921/00_audit/pilot.json")
    pilot = read_json(pilot_path)
    pilot_ids = [str(value) for value in pilot.get("dev", [])]
    if len(pilot_ids) != 24:
        raise ValueError(f"parent pilot must contain exactly 24 DEV ids, got {len(pilot_ids)}")
    rows = {str(row["pair_id"]): row for row in rows_by_split(registry, "dev")}
    if not set(pilot_ids).issubset(rows):
        raise ValueError("parent pilot contains a DEV id absent from the imported registry")
    result = []
    for pair_id in pilot_ids:
        row = rows[pair_id]
        entries = [
            item
            for item in entries_for(support, "hubert", "natural_primary", "dev")
            if str(item["pair_id"]) == str(row["pair_id"]) and bool(item.get("pair_eligible", True))
        ]
        result.append(make_training_sample(row, entries))
    return result


def _aggregate_dev_scores(scores: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"accuracy": [], "margin": []})
    support_count = 0
    valid_count = 0
    for score in scores:
        support_count += int(score.get("support_count", 0))
        valid_count += int(score.get("valid_count", 0))
        for row in score.get("group_metrics", []):
            group = str(row.get("source_group", ""))
            grouped[group]["accuracy"].append(float(row.get("accuracy", 0.0)))
            grouped[group]["margin"].append(float(row.get("margin", -2.0)))
    group_metrics = [
        {
            "source_group": group,
            "accuracy": float(np.mean(values["accuracy"])),
            "margin": float(np.mean(values["margin"])),
            "n_samples": len(values["accuracy"]),
        }
        for group, values in sorted(grouped.items())
        if values["accuracy"]
    ]
    return {
        "accuracy": float(np.mean([row["accuracy"] for row in group_metrics])) if group_metrics else None,
        "margin": float(np.mean([row["margin"] for row in group_metrics])) if group_metrics else None,
        "group_metrics": group_metrics,
        "support_count": support_count,
        "valid_count": valid_count,
        "coverage": valid_count / max(support_count, 1),
    }


def _evaluate_dev_checkpoint(config: Mapping[str, Any], root: Path, teacher: Any, samples: list[Mapping[str, Any]], vocab: Any, duration_stats: Mapping[str, float], feature_stats: Mapping[str, Any], support: Mapping[str, Any], checkpoint: Mapping[str, Any], *, mode: str, device: str) -> dict[str, Any]:
    import torch

    from scripts.experiments.phone_separability_enhancement.audio import BandGainRenderer

    from .conditioning import build_conditioning
    from .infer import infer_pcm
    from .model import build_model
    from .quality import pcm_contract

    audio = config["audio"]
    network = config["network"]
    model = build_model(
        vocab_size=len(vocab.labels),
        embedding_dim=int(network.get("embedding_dim", 16)),
        audio_channels=int(audio.get("n_bands", 24)),
        hidden_channels=int(network.get("hidden_channels", 64)),
        max_gain_db=float(audio.get("max_gain_db", 6.0)),
        dilations=tuple(int(value) for value in network.get("residual_dilations", [1, 2, 4])),
    ).to(device)
    payload = load_checkpoint(model, checkpoint["path"], device=device)
    if int(payload.get("schema_version", 0)) < 3 or str(payload.get("mode")) != str(mode) or int(payload.get("seed", -1)) != int(checkpoint["seed"]):
        raise ValueError(f"checkpoint contract metadata mismatch: {checkpoint['path']}")
    if payload.get("model_hash") != state_dict_hash(payload.get("model", {})):
        raise ValueError(f"checkpoint state hash mismatch: {checkpoint['path']}")
    model.eval()
    renderer = BandGainRenderer(
        sample_rate=int(audio.get("sample_rate", 16000)),
        n_fft=int(audio.get("n_fft", 512)),
        win_length=int(audio.get("win_length", 512)),
        hop_length=int(audio.get("hop_length", 128)),
        n_bands=int(audio.get("n_bands", 24)),
        max_gain_db=float(audio.get("max_gain_db", 6.0)),
    )
    scores: list[dict[str, Any]] = []
    qualities: list[dict[str, Any]] = []
    identity = True
    with torch.inference_mode():
        for sample in samples:
            source = np.asarray(sample["audio_pcm"], dtype=np.int16)
            x = torch.as_tensor(source.astype(np.float32) / 32768.0, device=device)
            features, _ = renderer.band_features(x)
            features = _normalize_features(features, feature_stats)
            condition = build_conditioning(
                sample["tokens"],
                sample_count=source.size,
                frame_count=int(features.shape[-1]),
                vocabulary=vocab,
                duration_stats=duration_stats,
                protected_mask=sample["protected_mask"],
                edit_mask=sample.get("edit_mask"),
                mode=mode,
                sample_rate=int(audio.get("sample_rate", 16000)),
                hop_length=int(audio.get("hop_length", 128)),
            )
            candidate, _ = infer_pcm(
                source,
                sample["protected_mask"],
                condition,
                checkpoint["path"],
                vocabulary=vocab,
                device=device,
                audio=audio,
                network=network,
                feature_stats=feature_stats,
                model=model,
            )
            identity = identity and bool(np.array_equal(source, candidate))
            quality = pcm_contract(
                source,
                candidate,
                sample["protected_mask"],
                max_residual_ratio=float(audio.get("max_residual_energy_ratio", 0.01)),
                min_snr_db=float(audio.get("min_snr_db", 20.0)),
                max_rms_change_db=float(audio.get("max_rms_change_db", 1.0)),
            )
            score = score_waveform(
                teacher,
                candidate,
                sample["tokens"],
                sample.get("support_entries", []),
                support["mixed_centroids"]["hubert"],
                layer=int(config["models"]["primary"].get("layer", 6)),
                condition="natural",
            )
            qualities.append(quality)
            scores.append(score)
    aggregate = _aggregate_dev_scores(scores)
    quality_pass = bool(qualities) and all(bool(item.get("pass", False)) for item in qualities)
    return {
        "seed": int(checkpoint["seed"]),
        "step": int(checkpoint["step"]),
        "checkpoint": str(Path(checkpoint["path"]).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint["path"]),
        "mode": mode,
        "dev": aggregate,
        "quality_pass": quality_pass,
        "quality": {
            "pass_count": int(sum(bool(item.get("pass", False)) for item in qualities)),
            "count": len(qualities),
            "max_residual_energy_ratio": max((float(item["residual_energy_ratio"]) for item in qualities), default=None),
            "min_snr_db": min((float(item["snr_db"]) for item in qualities if item.get("snr_db") is not None), default=None),
            "max_abs_rms_change_db": max((abs(float(item["rms_change_db"])) for item in qualities), default=None),
        },
        "zero_gain_identity": identity if int(checkpoint["step"]) == 0 else None,
    }


def _select_dev_candidate(candidates: list[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    eligible = [
        dict(item)
        for item in candidates
        if bool(item.get("quality_pass"))
        and float(item.get("dev", {}).get("coverage", 0.0)) >= 0.90
        and item.get("dev", {}).get("margin") is not None
    ]
    eligible.sort(
        key=lambda item: (
            -float(item["dev"]["margin"]),
            -float(item["dev"].get("accuracy", -1.0)),
            float(item.get("quality", {}).get("max_residual_energy_ratio") if item.get("quality", {}).get("max_residual_energy_ratio") is not None else float("inf")),
            int(item.get("step", 0)),
        )
    )
    return (eligible[0] if eligible else None), "quality_pass_and_dev_coverage_then_group_equal_signed_margin_accuracy_residual_step"


def stage_lock(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _assert_protocol(config, root)
    if str(config.get("experiment_mode", "repair")) == "remeasure":
        result = {"status": "NOT_APPLICABLE", "reason": "FROZEN_AUDIO_REMEASUREMENT"}
        write_json(root / "03_lock/model_lock.json", result)
        _set_state(root, "lock", "NOT_APPLICABLE", details=result)
        return result
    if not _require_calibration(root, "lock"):
        return {"status": "DEPENDENCY_BLOCKED", "reason": "CALIBRATION_NOT_PASS"}
    decision = resource_decision(config, stage="lock", gpu_required=True)
    write_json(root / "03_lock/resource_preflight.json", decision)
    if decision["decision"] != "PASS":
        _set_state(root, "lock", "RESOURCE_WAIT", reasons=decision["reasons"], details={"resource_snapshot": decision["snapshot"]})
        return decision
    train_summary = read_json(root / "02_train/summary.json") if (root / "02_train/summary.json").is_file() else {"status": "NOT_RUN"}
    if train_summary.get("status") != "COMPLETE":
        result = {"status": "PARTIAL_DIRECT_LOCK_TRAINING_PENDING", "training_status": train_summary.get("status", "NOT_RUN")}
        write_json(root / "03_lock/model_lock.json", result)
        _set_state(root, "lock", "PARTIAL", reasons=["TRAINING_NOT_COMPLETE"])
        return result
    support = load_support(root)
    registry = load_registry(root)
    vocab, duration_stats = _load_conditioning_config(root)
    feature_stats = _load_feature_stats(root)
    samples = _dev_samples(config, registry, support)
    device = str(config["runtime"].get("device", config["models"].get("device", "cuda:0")))
    lock: dict[str, Any] = {
        "schema_version": 2,
        "status": "LOCKED_BEFORE_XLSR_E",
        "protocol_id": config["protocol_id"],
        "support_hash": support["support_hash"],
        "direct_arm": "OPT_DYNAMIC_6",
        "direct_parent_lock": read_json(REPO_ROOT / config["source"]["parent_lock"]),
        "dev_selection": {"split": "parent_pilot24", "encoder": "hubert", "layer": int(config["models"]["primary"].get("layer", 6)), "sample_count": len(samples), "candidate_rule": "all_step_checkpoints_per_seed", "eligibility": "quality_pass_and_dev_coverage_at_least_0.90", "selection_rule": None, "release_seed": int(config["training"].get("release_seed", config["training"]["seeds"][0]))},
        "arms": {},
    }
    missing: list[str] = []
    with proxy_environment(config["runtime"].get("proxy")):
        teacher = load_frozen_teacher(config["models"]["primary"], device=device, proxy=config["runtime"].get("proxy"), allow_download=bool(config["runtime"].get("allow_model_download", True)))
        try:
            for arm, mode in ARM_MODES.items():
                candidates = _checkpoint_candidates(root, arm)
                if not candidates:
                    missing.append(arm)
                    continue
                by_seed: dict[str, Any] = {}
                selected_release = None
                rule = None
                for seed in [int(value) for value in config["training"]["seeds"]]:
                    evaluated = [_evaluate_dev_checkpoint(config, root, teacher, samples, vocab, duration_stats, feature_stats, support, candidate, mode=mode, device=device) for candidate in candidates if int(candidate["seed"]) == seed]
                    selected, rule = _select_dev_candidate(evaluated)
                    by_seed[str(seed)] = {"candidates": evaluated, "selected": selected}
                    if seed == int(config["training"].get("release_seed", config["training"]["seeds"][0])):
                        selected_release = selected
                selected = selected_release
                lock["dev_selection"]["selection_rule"] = rule
                lock["arms"][arm] = {"mode": mode, "per_seed": by_seed, "selected": selected, "release_seed": int(config["training"].get("release_seed", config["training"]["seeds"][0]))}
                if selected is None:
                    missing.append(f"{arm}:NO_DEV_ELIGIBLE_CHECKPOINT")
        finally:
            del teacher
    if missing:
        lock["status"] = "PARTIAL_DIRECT_LOCK_DEV_SELECTION_FAILED"
        lock["missing_arms"] = missing
        _set_state(root, "lock", "PARTIAL", reasons=[f"DEV_SELECTION:{','.join(missing)}"])
    else:
        _set_state(root, "lock", "COMPLETE", details={"dev_samples": len(samples), "selection": "DEV_ONLY"})
    write_json(root / "03_lock/model_lock.json", lock)
    return lock


def stage_infer(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _assert_protocol(config, root)
    if str(config.get("experiment_mode", "repair")) == "remeasure":
        old_root = REPO_ROOT / config["source"]["old_run"]
        old_manifest = old_root / "04_audio/manifest.json"
        if not old_manifest.is_file():
            _set_state(root, "infer", "DEPENDENCY_BLOCKED", reasons=["OLD_AUDIO_MANIFEST_MISSING"])
            return {"status": "DEPENDENCY_BLOCKED", "reason": "OLD_AUDIO_MANIFEST_MISSING"}
        payload = read_json(old_manifest)
        expected_e_seen = int(config.get("cohort", {}).get("split_counts", {}).get("e_seen", 40))
        if payload.get("status") != "COMPLETE" or len(payload.get("rows", [])) != expected_e_seen:
            raise ValueError(f"old frozen audio manifest is not the expected {expected_e_seen} E_SEEN rows")
        for row in payload["rows"]:
            for arm, path in row.get("paths", {}).items():
                if arm in {"N", "T", "D", "A", "B", "C"} and path and not Path(str(path)).is_file():
                    raise FileNotFoundError(path)
        write_json(root / "04_audio/manifest.json", {**payload, "schema_version": 2, "status": "COMPLETE", "source": "OLD_READ_ONLY", "old_manifest": str(old_manifest.resolve()), "old_manifest_sha256": file_sha256(old_manifest)})
        _set_state(root, "infer", "COMPLETE", details={"frozen_audio": True, "rows": len(payload["rows"])})
        return {"status": "COMPLETE", "rows": len(payload["rows"]), "frozen_audio": True}
    if not _require_calibration(root, "infer"):
        return {"status": "DEPENDENCY_BLOCKED", "reason": "CALIBRATION_NOT_PASS"}
    lock_path = root / "03_lock/model_lock.json"
    if not lock_path.is_file():
        _set_state(root, "infer", "DEPENDENCY_BLOCKED", reasons=["MODEL_LOCK_MISSING"])
        return {"status": "DEPENDENCY_BLOCKED"}
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED_BEFORE_XLSR_E" or any(arm not in lock.get("arms", {}) for arm in ARM_MODES):
        _set_state(root, "infer", "DEPENDENCY_BLOCKED", reasons=["CONDITIONAL_MODEL_LOCK_INCOMPLETE"])
        return {"status": "DEPENDENCY_BLOCKED", "reason": "CONDITIONAL_MODEL_LOCK_INCOMPLETE"}
    checkpoints: dict[str, str] = {}
    for arm in ARM_MODES:
        info = lock["arms"][arm]
        selected = info.get("selected") if isinstance(info, Mapping) else None
        checkpoint = info.get("checkpoint") if isinstance(info, Mapping) else None
        if not checkpoint and isinstance(selected, Mapping):
            checkpoint = selected.get("checkpoint")
        if not checkpoint:
            _set_state(root, "infer", "DEPENDENCY_BLOCKED", reasons=[f"SELECTED_CHECKPOINT_MISSING:{arm}"])
            return {"status": "DEPENDENCY_BLOCKED", "reason": f"SELECTED_CHECKPOINT_MISSING:{arm}"}
        checkpoints[arm] = str(checkpoint)
    available = resource_decision(config, stage="infer", gpu_required=True)
    write_json(root / "04_audio/resource_preflight.json", available)
    if available["decision"] != "PASS":
        _set_state(root, "infer", "RESOURCE_WAIT", reasons=available["reasons"], details={"resource_snapshot": available["snapshot"]})
        return available
    from .infer import infer_pcm

    registry = load_registry(root)
    support = load_support(root)
    vocab, duration_stats = _load_conditioning_config(root)
    feature_stats = _load_feature_stats(root)
    device = str(config["runtime"].get("device", "cuda:0"))
    rows_out: list[dict[str, Any]] = []
    for row in rows_by_split(registry, "e_seen"):
        source, _ = read_pcm16(row["natural"]["audio_path"])
        protection = make_protected_mask(row["natural"]["tokens"], source.size)
        protected = protection["protected"]
        arms = {"N": row["natural"]["audio_path"], "T": row["tts"]["audio_path"], "D": row["direct"]["audio_path"] if row["direct"]["exists"] else None}
        for arm, mode in ARM_MODES.items():
            import torch
            renderer = BandGainRenderer(sample_rate=16000, n_fft=512, win_length=512, hop_length=128, n_bands=24, max_gain_db=6.0)
            x = torch.as_tensor(source.astype(np.float32) / 32768.0, device=device)
            features, _ = renderer.band_features(x)
            from .conditioning import build_conditioning
            condition = build_conditioning(row["natural"]["tokens"], sample_count=source.size, frame_count=int(features.shape[-1]), vocabulary=vocab, duration_stats=duration_stats, protected_mask=protected, edit_mask=protection["mask"], mode=mode)
            candidate, metadata = infer_pcm(source, protected, condition, checkpoints[arm], vocabulary=vocab, device=device, audio=config["audio"], network=config["network"], feature_stats=feature_stats)
            path = root / "04_audio" / arm / f"{row['pair_id']}.wav"
            pcm_meta = write_pcm16(path, candidate)
            arms[arm] = str(path.resolve())
            write_json(root / "04_audio" / arm / f"{row['pair_id']}.json", {"pair_id": row["pair_id"], "arm": arm, "metadata": metadata, "pcm": pcm_meta})
        rows_out.append({"pair_id": row["pair_id"], "source_group": row["source_group"], "paths": arms})
    write_json(root / "04_audio/manifest.json", {"status": "COMPLETE", "rows": rows_out})
    _set_state(root, "infer", "COMPLETE")
    return {"status": "COMPLETE", "rows": len(rows_out)}


def stage_quality(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    from .mfa import mfa_batch_attempt
    from .quality import audit_timing, pcm_contract
    from .asr import batch_content_audit

    manifest = root / "04_audio/manifest.json"
    if not manifest.is_file():
        _set_state(root, "quality", "DEPENDENCY_BLOCKED", reasons=["INFERENCE_MANIFEST_MISSING"])
        return {"status": "DEPENDENCY_BLOCKED"}
    registry = load_registry(root)
    rows = read_json(manifest)["rows"]
    output = []
    for item in rows:
        source_row = next(row for row in registry["rows"] if row["pair_id"] == item["pair_id"])
        source, _ = read_pcm16(source_row["natural"]["audio_path"])
        protected = make_protected_mask(source_row["natural"]["tokens"], source.size)["protected"]
        for arm in ("N", "D", "A", "B", "C"):
            path = item["paths"].get(arm)
            if not path:
                continue
            candidate, _ = read_pcm16(path)
            try:
                t0 = pcm_contract(source, candidate, protected, max_residual_ratio=float(config["audio"]["max_residual_energy_ratio"]), min_snr_db=float(config["audio"]["min_snr_db"]), max_rms_change_db=float(config["audio"]["max_rms_change_db"]))
            except ValueError as exc:
                t0 = {"pass": False, "status": "DIRECT_LENGTH_MISMATCH" if arm == "D" else "LENGTH_MISMATCH", "reason": str(exc), "same_length": False}
            output.append({"pair_id": item["pair_id"], "source_group": item["source_group"], "arm": arm, "t0": t0})
    write_jsonl(root / "06_quality/t0.jsonl", output)
    mfa_payloads: dict[str, Any] = {}
    mfa_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in ("N", "T", "D", *ARM_MODES):
        batch_rows = []
        for item in rows:
            source_row = next(row for row in registry["rows"] if row["pair_id"] == item["pair_id"])
            path = source_row["natural"]["audio_path"] if arm == "N" else item["paths"].get(arm)
            if not path:
                continue
            pcm, _ = read_pcm16(path)
            batch_rows.append({"pair_id": item["pair_id"], "pcm": pcm, "transcript": source_row["transcript"]})
        payload = mfa_batch_attempt(config, root, rows=batch_rows, condition=f"e_seen_{arm}")
        mfa_payloads[arm] = {key: value for key, value in payload.items() if key != "rows"}
        for aligned in payload.get("rows", []):
            mfa_rows[(str(aligned["pair_id"]), arm)] = aligned
    write_json(root / "06_quality/mfa_summary.json", {"status": "COMPLETE" if mfa_payloads and all(value.get("status") == "COMPLETE" for value in mfa_payloads.values()) else "PARTIAL", "conditions": mfa_payloads})
    timing_rows: list[dict[str, Any]] = []
    for item in rows:
        pair_id = str(item["pair_id"])
        reference = mfa_rows.get((pair_id, "N"))
        for arm in ("T", "D", *ARM_MODES):
            candidate = mfa_rows.get((pair_id, arm))
            row = {"pair_id": pair_id, "source_group": item["source_group"], "arm": arm, "reference_condition": "N", "candidate_condition": arm}
            if not reference or not candidate or reference.get("status") != "COMPLETE" or candidate.get("status") != "COMPLETE":
                row.update({"status": "MFA_INCOMPLETE", "timing_pass": False})
            else:
                metrics = audit_timing(
                    reference.get("tokens", []),
                    candidate.get("tokens", []),
                    pause_min_ms=float(config["quality"].get("pause_min_ms", 50.0)),
                    pause_iou_min=float(config["quality"].get("pause_iou_min", 0.5)),
                    edge_error_ms=float(config["quality"].get("pause_edge_ms", 20.0)),
                    max_edit_rate=float(config["quality"].get("timing_max_edit_rate", 0.05)),
                    min_coverage=float(config["quality"].get("timing_min_coverage", 0.90)),
                    median_error_ms=float(config["quality"].get("timing_median_ms", 20.0)),
                    p95_error_ms=float(config["quality"].get("timing_p95_ms", 40.0)),
                )
                row.update({"status": "COMPLETE", **metrics})
            timing_rows.append(row)
    write_jsonl(root / "06_quality/timing.jsonl", timing_rows)
    asr_rows = []
    for item in rows:
        source_row = next(row for row in registry["rows"] if row["pair_id"] == item["pair_id"])
        for arm in ("N", "T", "D", *ARM_MODES):
            path = item["paths"].get(arm)
            if path:
                asr_rows.append({"pair_id": item["pair_id"], "source_group": item["source_group"], "arm": arm, "path": path, "reference": source_row["transcript"]})
    asr = batch_content_audit(asr_rows, config)
    write_json(root / "06_quality/asr.json", asr)
    timing_complete = bool(timing_rows) and all(row.get("status") == "COMPLETE" for row in timing_rows)
    timing_pass = bool(timing_rows) and all(bool(row.get("timing_pass", False)) for row in timing_rows)
    mfa_complete = bool(mfa_payloads) and all(value.get("status") == "COMPLETE" for value in mfa_payloads.values())
    t0_complete = len(output) == len(rows) * 5
    result = {"status": "COMPLETE" if t0_complete and mfa_complete and timing_complete and asr.get("status") == "COMPLETE" else "PARTIAL", "t0_rows": len(output), "t0_expected_rows": len(rows) * 5, "t0_pass_rows": int(sum(bool(row["t0"].get("pass", False)) for row in output)), "t0_contract_arms": ["N", "D", "A", "B", "C"], "mfa": {"status": "COMPLETE" if mfa_complete else "PARTIAL", "conditions": list(mfa_payloads)}, "timing": {"status": "COMPLETE" if timing_complete else "PARTIAL", "pass_rows": int(sum(bool(row.get("timing_pass", False)) for row in timing_rows)), "rows": len(timing_rows), "arms": ["T", "D", "A", "B", "C"]}, "asr": asr}
    write_json(root / "06_quality/summary.json", result)
    _set_state(root, "quality", result["status"], reasons=[] if result["status"] == "COMPLETE" else ["ASR_OR_QUALITY_CONTRACT_INCOMPLETE"])
    return result


def stage_phone(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    from .phone_eval import classify_noninferiority, score_candidate_set, summarize_pairwise

    manifest_path = root / "04_audio/manifest.json"
    if not manifest_path.is_file():
        _set_state(root, "phone", "DEPENDENCY_BLOCKED", reasons=["INFERENCE_MANIFEST_MISSING"])
        return {"status": "DEPENDENCY_BLOCKED"}
    if not _require_calibration(root, "phone"):
        return {"status": "DEPENDENCY_BLOCKED", "reason": "CALIBRATION_NOT_PASS"}
    gate = resource_decision(config, stage="phone", gpu_required=True)
    write_json(root / "05_phone/resource_preflight.json", gate)
    if gate["decision"] != "PASS":
        _set_state(root, "phone", "RESOURCE_WAIT", reasons=gate["reasons"], details={"resource_snapshot": gate["snapshot"]})
        return gate
    registry = load_registry(root)
    support = load_support(root)
    manifest = read_json(manifest_path)
    device = str(config["runtime"].get("device", "cuda:0"))
    with proxy_environment(config["runtime"].get("proxy")):
        teachers = {"hubert": load_frozen_teacher(config["models"]["primary"], device=device, proxy=config["runtime"].get("proxy"), allow_download=bool(config["runtime"].get("allow_model_download", True))), "xlsr": load_frozen_teacher(config["models"]["cross_encoder"], device=device, proxy=config["runtime"].get("proxy"), allow_download=bool(config["runtime"].get("allow_model_download", True)))}
    results = []
    natural_arms = ("N", "D", "A", "B", "C")
    matched_arms = ("N", "T", "D", "A", "B", "C")
    for item in manifest["rows"]:
        row = next(value for value in registry["rows"] if value["pair_id"] == item["pair_id"])
        arms = {arm: read_pcm16(path)[0] for arm, path in item["paths"].items() if path}
        for encoder, teacher in teachers.items():
            results.append(score_candidate_set(teacher, row, support, {arm: arms[arm] for arm in natural_arms if arm in arms}, encoder=encoder, view="natural_primary", split="e_seen"))
            results.append(score_candidate_set(teacher, row, support, {arm: arms[arm] for arm in matched_arms if arm in arms}, encoder=encoder, view="matched_nt", split="e_seen"))
    write_jsonl(root / "05_phone/results.jsonl", results)
    comparisons = {
        "natural_primary": [("D", "N"), ("A", "N"), ("B", "N"), ("C", "N"), ("C", "A"), ("C", "B")],
        "matched_nt": [("D", "T"), ("A", "T"), ("B", "T"), ("C", "T")],
    }
    pairwise: dict[str, Any] = {}
    for encoder in ("hubert", "xlsr"):
        for view in ("natural_primary", "matched_nt"):
            subset = [row for row in results if row["view"] == view and row["encoder"] == encoder]
            accuracy = summarize_pairwise(subset, comparisons[view], metric="accuracy", confidence=0.95 if view == "natural_primary" else 0.9875)
            margin = summarize_pairwise(subset, comparisons[view], metric="margin", confidence=0.95 if view == "natural_primary" else 0.9875)
            combined = {key: {**value, "margin": margin.get(key)} for key, value in accuracy.items()}
            if view == "matched_nt":
                combined["noninferiority_98_75"] = {key: {"accuracy": {**value, "classification": classify_noninferiority(value, epsilon=float(config.get("statistics", {}).get("tts_noninferiority_epsilon", 0.02)))}, "margin": {**margin.get(key, {}), "classification": classify_noninferiority(margin.get(key, {}), epsilon=float(config.get("statistics", {}).get("tts_noninferiority_epsilon", 0.02)))}} for key, value in accuracy.items()}
            pairwise[f"{encoder}_{view}"] = combined
    expected_results = len(manifest.get("rows", [])) * 2 * 2
    summary = {"status": "COMPLETE" if len(results) == expected_results else "PARTIAL", "rows": len(results), "expected_rows": expected_results, "pairwise": pairwise, "comparison_contract": "natural_primary excludes T; matched_nt uses each arm's own timing and TTS T slots; T/T is descriptive only"}
    write_json(root / "05_phone/summary.json", summary)
    _set_state(root, "phone", summary["status"], reasons=[] if summary["status"] == "COMPLETE" else ["PHONE_SCOPE_INCOMPLETE"])
    return summary


def stage_render(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if not _require_calibration(root, "render"):
        return {"status": "DEPENDENCY_BLOCKED", "reason": "CALIBRATION_NOT_PASS"}
    gate = resource_decision(config, stage="render", gpu_required=True)
    write_json(root / "07_tfg/resource_preflight.json", gate)
    if gate["decision"] != "PASS":
        _set_state(root, "render", "RESOURCE_WAIT", reasons=gate["reasons"], details={"resource_snapshot": gate["snapshot"]})
        return gate
    from .render import render_cell
    manifest = read_json(root / "00_protocol/render_manifest.json")
    audio_manifest = read_json(root / "04_audio/manifest.json")
    wav2lip = REPO_ROOT / config["tfg"]["wav2lip_checkpoint"]
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    python = config["tfg"].get("wav2lip_python", "/home/wjj/.venvs/wav2lip/bin/python")
    if not wav2lip.is_file() or not Path(str(python)).is_file():
        reason = "WAV2LIP_DEPENDENCY_MISSING"
        _set_state(root, "render", "DEPENDENCY_BLOCKED", reasons=[reason])
        return {"status": "DEPENDENCY_BLOCKED", "reason": reason}
    audio_by_pair = {str(row["pair_id"]): row for row in audio_manifest["rows"]}
    render_rows = [item for item in manifest["rows"] if str(item["pair_id"]) in audio_by_pair]
    if not render_rows:
        _set_state(root, "render", "DEPENDENCY_BLOCKED", reasons=["NO_RENDER_AUDIO_INTERSECTION"])
        return {"status": "DEPENDENCY_BLOCKED", "reason": "NO_RENDER_AUDIO_INTERSECTION"}
    results = []
    portrait_ids = tuple(str(value) for value in config.get("tfg", {}).get("render_portraits", [config.get("tfg", {}).get("primary_portrait", "3")]))
    unknown_portraits = sorted(set(portrait_ids) - set(str(key) for key in manifest["rows"][0].get("portraits", {})))
    if unknown_portraits:
        _set_state(root, "render", "DEPENDENCY_BLOCKED", reasons=[f"PORTRAIT_METADATA_MISSING:{','.join(unknown_portraits)}"])
        return {"status": "DEPENDENCY_BLOCKED", "reason": "PORTRAIT_METADATA_MISSING"}
    for item in render_rows:
        audio_row = audio_by_pair[str(item["pair_id"])]
        for portrait_id in portrait_ids:
            for arm, audio_path in audio_row["paths"].items():
                if not audio_path:
                    continue
                results.append(render_cell(item, portrait_id=portrait_id, video_arm=arm, audio_path=audio_path, output_dir=root / "07_tfg" / f"portrait_{portrait_id}", checkpoint=wav2lip, ffmpeg=ffmpeg, device="cuda", python=python, seed=int(config["tfg"].get("seed", 42))))
    expected = len(audio_by_pair) * len(config.get("tfg", {}).get("render_portraits", ["3"])) * 6
    status = "COMPLETE" if len(results) == expected else "PARTIAL"
    write_json(root / "07_tfg/manifest.json", {"status": status, "scope": "E_SEEN_AUDIO_INTERSECTION", "audio_rows": len(audio_by_pair), "render_rows": len(render_rows), "portrait_ids": list(portrait_ids), "expected_cells": expected, "cells": len(results), "results": results})
    _set_state(root, "render", status, reasons=[] if status == "COMPLETE" else ["RENDER_SCOPE_INCOMPLETE"])
    return {"status": status, "cells": len(results), "expected_cells": expected}


def stage_score(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    from .score_worker import validate_batch_score_request
    from .sync_support import frozen_support_windows, score_distance_matrix

    if not _require_calibration(root, "score"):
        return {"status": "DEPENDENCY_BLOCKED", "reason": "CALIBRATION_NOT_PASS"}
    gate = resource_decision(config, stage="score", gpu_required=True)
    write_json(root / "08_sync/resource_preflight.json", gate)
    if gate["decision"] != "PASS":
        _set_state(root, "score", "RESOURCE_WAIT", reasons=gate["reasons"], details={"resource_snapshot": gate["snapshot"]})
        return gate
    render_manifest = root / "07_tfg/manifest.json"
    if not render_manifest.is_file():
        _set_state(root, "score", "DEPENDENCY_BLOCKED", reasons=["RENDER_MANIFEST_MISSING"])
        return {"status": "DEPENDENCY_BLOCKED"}
    syncnet = REPO_ROOT / config["tfg"]["syncnet_checkpoint"]
    python = Path(str(config["tfg"].get("syncnet_python", "/home/wjj/.venvs/syncnet/bin/python")))
    if not syncnet.is_file() or not python.is_file():
        reason = "SYNCNET_DEPENDENCY_MISSING"
        _set_state(root, "score", "DEPENDENCY_BLOCKED", reasons=[reason])
        return {"status": "DEPENDENCY_BLOCKED", "reason": reason}
    render_payload = read_json(render_manifest)
    if render_payload.get("status") != "COMPLETE":
        _set_state(root, "score", "DEPENDENCY_BLOCKED", reasons=["RENDER_INCOMPLETE"])
        return {"status": "DEPENDENCY_BLOCKED", "reason": "RENDER_INCOMPLETE"}
    audio_payload = read_json(root / "04_audio/manifest.json")
    audio_by_pair = {str(row["pair_id"]): row for row in audio_payload.get("rows", [])}
    primary_portrait = str(config.get("tfg", {}).get("primary_portrait", "3"))
    render_by_pair_arm = {
        (str(row["pair_id"]), str(row["video_arm"])): row
        for row in render_payload.get("results", [])
        if str(row.get("portrait_id")) == primary_portrait
    }
    audio_arms = ("N", "T", "D", "A", "B", "C")
    natural_arms = ("N", "D", "A", "B", "C")
    selected_cells = [("N", "N")]
    selected_cells.extend((arm, "N") for arm in ("D", "A", "B", "C"))
    selected_cells.extend(("N", arm) for arm in ("D", "A", "B", "C"))
    selected_cells.extend((arm, arm) for arm in ("D", "A", "B", "C"))
    selected_cells.append(("T", "T"))
    expected_videos = len(audio_by_pair) * len(audio_arms)
    if len(render_by_pair_arm) != expected_videos:
        reason = f"RENDER_CELL_COUNT_MISMATCH:{len(render_by_pair_arm)}:{expected_videos}"
        _set_state(root, "score", "DEPENDENCY_BLOCKED", reasons=[reason])
        return {"status": "DEPENDENCY_BLOCKED", "reason": reason}

    score_root = root / "08_sync"
    worker_root = score_root / "worker"
    boxes_root = score_root / "score_boxes"
    audio_json_root = score_root / "audio_maps"
    score_root.mkdir(parents=True, exist_ok=True)
    worker_receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    worker_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    vshift = int(config["tfg"].get("vshift", 15))
    batch_size = int(config["tfg"].get("syncnet_batch_size", 20))
    worker_script = Path(__file__).with_name("batch_score_worker.py")

    for pair_id in sorted(audio_by_pair):
        audio_row = audio_by_pair[pair_id]
        render_row = next((row for row in read_json(root / "00_protocol/render_manifest.json")["rows"] if str(row["pair_id"]) == pair_id), None)
        if render_row is None:
            failures.append({"pair_id": pair_id, "stage": "score", "reason": "RENDER_ROW_MISSING"})
            continue
        portrait = render_row["portraits"][primary_portrait]
        box_path = boxes_root / f"{pair_id}.json"
        write_json(box_path, portrait["score_box"])
        audio_paths = {arm: str(audio_row["paths"][arm]) for arm in audio_arms}
        audio_json = audio_json_root / f"{pair_id}.json"
        write_json(audio_json, audio_paths)
        video_paths = {arm: str(Path(str(render_by_pair_arm[(pair_id, arm)]["output"])).resolve()) for arm in audio_arms}
        output_root = worker_root / pair_id
        request = {
            "video_paths": video_paths,
            "video_sha256": {arm: file_sha256(Path(path)) for arm, path in video_paths.items()},
            "sample_id": pair_id,
            "score_box": portrait["score_box"],
            "score_box_sha256": canonical_hash(portrait["score_box"]),
            "audio_paths": audio_paths,
            "audio_sha256": {arm: file_sha256(Path(path)) for arm, path in audio_paths.items()},
            "model": str(syncnet.resolve()),
            "model_sha256": file_sha256(syncnet),
            "output_root": str(output_root.resolve()),
            "vshift": vshift,
            "batch_size": batch_size,
            "python": str(python.resolve()),
        }
        validate_batch_score_request(request)
        output_root.mkdir(parents=True, exist_ok=True)
        request_path = output_root / "request.json"
        write_json(request_path, request)
        sidecars = [output_root / video_arm / f"{video_arm}__{audio_arm}__worker.json" for video_arm in audio_arms for audio_arm in audio_arms]
        request_hash = canonical_hash(request)
        batch_manifest_path = output_root / "batch_manifest.json"
        cached = False
        if all(path.is_file() for path in sidecars) and batch_manifest_path.is_file():
            try:
                batch_manifest = read_json(batch_manifest_path)
                cached = bool(batch_manifest.get("status") == "PASS" and batch_manifest.get("request_sha256") == request_hash and batch_manifest.get("model_sha256") == file_sha256(syncnet))
            except (OSError, ValueError, KeyError):
                cached = False
        receipt = {"pair_id": pair_id, "request": str(request_path.resolve()), "request_sha256": request_hash, "status": "CACHED" if cached else "NOT_RUN"}
        if not cached:
            videos_json = output_root / "videos.json"
            write_json(videos_json, video_paths)
            command = [str(python), str(worker_script), "--videos-json", str(videos_json), "--audio-json", str(audio_json), "--score-box", str(box_path), "--model", str(syncnet), "--output-root", str(output_root), "--vshift", str(vshift), "--batch-size", str(batch_size), "--request-sha256", request_hash]
            try:
                completed = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False, timeout=1800)
                receipt.update({"status": "COMPLETE" if completed.returncode == 0 else "FAILED", "returncode": int(completed.returncode), "command": command, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-4000:]})
            except (OSError, subprocess.TimeoutExpired) as exc:
                receipt.update({"status": "FAILED", "error": f"{type(exc).__name__}:{exc}", "command": command})
        worker_receipts.append(receipt)
        write_json(output_root / "receipt.json", receipt)
        if receipt["status"] == "FAILED":
            failures.append({"pair_id": pair_id, "stage": "score_worker", "reason": receipt.get("error", receipt.get("stderr", "WORKER_FAILED"))})
            continue
        for video_arm in audio_arms:
            for audio_arm in audio_arms:
                sidecar_path = output_root / video_arm / f"{video_arm}__{audio_arm}__worker.json"
                if not sidecar_path.is_file():
                    failures.append({"pair_id": pair_id, "video_arm": video_arm, "audio_arm": audio_arm, "stage": "score_worker", "reason": "WORKER_SIDECAR_MISSING"})
                    continue
                sidecar = read_json(sidecar_path)
                matrix_path = Path(str(sidecar.get("matrix", "")))
                if sidecar.get("status") != "complete" or not matrix_path.is_file() or file_sha256(matrix_path) != sidecar.get("matrix_sha256"):
                    failures.append({"pair_id": pair_id, "video_arm": video_arm, "audio_arm": audio_arm, "stage": "score_worker", "reason": "WORKER_MATRIX_INVALID"})
                    continue
                worker_rows[(pair_id, video_arm, audio_arm)] = sidecar

    score_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for pair_id in sorted(audio_by_pair):
        audio_row = audio_by_pair[pair_id]
        natural_reference = worker_rows.get((pair_id, "N", "N"))
        tts_reference = worker_rows.get((pair_id, "T", "T"))
        if natural_reference is None:
            failures.append({"pair_id": pair_id, "stage": "score", "reason": "NATURAL_REFERENCE_MISSING"})
            continue
        natural_samples = int(natural_reference["audio_sample_count"])
        natural_frames = int(natural_reference["video_meta"]["frame_count"])
        natural_support = frozen_support_windows(natural_samples, natural_frames, vshift=vshift, fps=float(config["tfg"].get("fps", 25)), mel_hz=80.0)
        tts_support = frozen_support_windows(int(tts_reference["audio_sample_count"]), int(tts_reference["video_meta"]["frame_count"]), vshift=vshift, fps=float(config["tfg"].get("fps", 25)), mel_hz=80.0) if tts_reference else {"windows": [], "support_count": 0, "reason": "TTS_REFERENCE_MISSING"}
        natural_shapes = {(int(worker_rows[(pair_id, arm, "N")]["video_meta"]["frame_count"]), int(worker_rows[(pair_id, arm, "N")] ["audio_sample_count"])) for arm in natural_arms if (pair_id, arm, "N") in worker_rows}
        support_status = "COMPLETE" if len(natural_shapes) == 1 and natural_support["support_count"] >= 20 else "BASELINE_SUPPORT_INELIGIBLE"
        support_record = {"pair_id": pair_id, "natural": natural_support, "tts": tts_support, "natural_clock_shapes": sorted([list(value) for value in natural_shapes]), "status": support_status, "rule": "natural N/D/A/B/C share W_N; T/T uses independent W_T"}
        support_rows.append(support_record)
        for video_arm, audio_arm in selected_cells:
            sidecar = worker_rows.get((pair_id, video_arm, audio_arm))
            if sidecar is None:
                score_rows.append({"pair_id": pair_id, "source_group": next(row["source_group"] for row in audio_payload["rows"] if str(row["pair_id"]) == pair_id), "video_arm": video_arm, "audio_arm": audio_arm, "status": "CELL_FAILURE", "reason": "WORKER_OUTPUT_MISSING"})
                continue
            support = tts_support["windows"] if (video_arm, audio_arm) == ("T", "T") else natural_support["windows"]
            values = np.load(Path(str(sidecar["matrix"])), allow_pickle=False)
            metric = score_distance_matrix(values, support, lags=list(range(-vshift, vshift + 1)), anchor_lag=int(natural_reference.get("lag", 0))) if support_status == "COMPLETE" or (video_arm, audio_arm) == ("T", "T") else {"status": "BASELINE_SUPPORT_INELIGIBLE", "support_count": len(support)}
            score_rows.append({"pair_id": pair_id, "source_group": next(row["source_group"] for row in audio_payload["rows"] if str(row["pair_id"]) == pair_id), "portrait_id": primary_portrait, "seed": render_row.get("seed"), "video_arm": video_arm, "audio_arm": audio_arm, "video": sidecar.get("video"), "audio": sidecar.get("audio"), "video_sha256": sidecar.get("video_sha256"), "audio_sha256": sidecar.get("audio_sha256"), "matrix": sidecar.get("matrix"), "matrix_sha256": sidecar.get("matrix_sha256"), "support_rule": "W_T" if (video_arm, audio_arm) == ("T", "T") else "W_N", **metric})
    write_jsonl(score_root / "scores.jsonl", score_rows)
    write_jsonl(score_root / "support.jsonl", support_rows)
    write_jsonl(score_root / "worker_receipts.jsonl", worker_receipts)
    complete_cells = sum(row.get("status") == "COMPLETE" for row in score_rows)
    expected_cells = len(audio_by_pair) * len(selected_cells)
    result = {"status": "COMPLETE" if not failures and complete_cells == expected_cells else "PARTIAL", "pairs": len(audio_by_pair), "selected_cells_per_pair": len(selected_cells), "expected_cells": expected_cells, "complete_cells": int(complete_cells), "worker_jobs": len(worker_receipts), "worker_failures": failures, "support": {"natural_min": int(min((row["natural"]["support_count"] for row in support_rows), default=0)), "tts_min": int(min((row["tts"]["support_count"] for row in support_rows), default=0)), "rule": "W_N frozen from natural N/N and shared by N/D/A/B/C; W_T independently frozen for T/T"}, "scoring_contract": "official SyncNet V2 frontend; generated video only; 13 natural-clock cells plus native T/T descriptive cell"}
    write_json(score_root / "summary.json", result)
    _set_state(root, "score", result["status"], reasons=[] if result["status"] == "COMPLETE" else ["SCORE_WORKER_OR_CELL_INCOMPLETE"], details={"complete_cells": int(complete_cells), "expected_cells": expected_cells})
    return result


def stage_analyze(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    from .analyze import analyze_tfg_scores, classify_phone, interpret

    phone = read_json(root / "05_phone/summary.json") if (root / "05_phone/summary.json").is_file() else {"status": "NOT_RUN"}
    sync = read_json(root / "08_sync/summary.json") if (root / "08_sync/summary.json").is_file() else {"status": "NOT_RUN"}
    official = read_json(root / "10_official_syncnet/summary.json") if (root / "10_official_syncnet/summary.json").is_file() else {"status": "NOT_RUN"}
    calibration = _calibration_payload(root)
    phone_gate = classify_phone(phone) if phone.get("status") == "COMPLETE" else {"classification": "INCONCLUSIVE_MISSING", "reason": "PHONE_SUMMARY_MISSING"}
    score_path = root / "08_sync/scores.jsonl"
    if sync.get("status") == "COMPLETE" and score_path.is_file():
        expected_pairs = len(read_json(root / "04_audio/manifest.json").get("rows", [])) if (root / "04_audio/manifest.json").is_file() else None
        required_cells = [("N", "N"), ("D", "N"), ("A", "N"), ("B", "N"), ("C", "N"), ("N", "D"), ("N", "A"), ("N", "B"), ("N", "C"), ("D", "D"), ("A", "A"), ("B", "B"), ("C", "C"), ("T", "T")]
        tfg = analyze_tfg_scores(read_jsonl(score_path), expected_pairs=expected_pairs, required_cells=required_cells)
        if calibration.get("status") != "PASS":
            tfg = {**tfg, "raw_status": tfg.get("status"), "status": "INCONCLUSIVE_CALIBRATION", "classification": "INCONCLUSIVE_CALIBRATION", "measurement_valid": False, "calibration_status": calibration.get("status", "NOT_RUN")}
    else:
        tfg = {"status": "INCONCLUSIVE_MISSING", "classification": "INCONCLUSIVE_MISSING", "reason": "SYNCNET_SCORE_SUMMARY_INCOMPLETE"}
    result = {"phone": phone, "phone_gate": phone_gate, "calibration": calibration, "tfg": tfg, "official": official, "tts_comparison": tfg.get("tts_native", "NOT_RUN"), "interpretation": interpret(phone_gate, tfg)}
    write_json(root / "08_sync/summary.json", {**sync, "analysis_status": tfg.get("status", "NOT_RUN")})
    write_json(root / "08_analysis/summary.json", result)
    complete = calibration.get("status") == "PASS" and phone.get("status") == "COMPLETE" and sync.get("status") == "COMPLETE" and official.get("status") == "COMPLETE" and tfg.get("status") == "COMPLETE"
    _set_state(root, "analyze", "COMPLETE" if complete else "PARTIAL", reasons=[] if complete else ["SCIENTIFIC_BRANCH_INCOMPLETE"])
    return result


def stage_official(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if not _require_calibration(root, "official"):
        return {"status": "DEPENDENCY_BLOCKED", "reason": "CALIBRATION_NOT_PASS"}
    gate = resource_decision(config, stage="score", gpu_required=True)
    write_json(root / "10_official_syncnet/resource_preflight.json", gate)
    if gate["decision"] != "PASS":
        _set_state(root, "official", "RESOURCE_WAIT", reasons=gate["reasons"], details={"resource_snapshot": gate["snapshot"]})
        return gate
    from .official_score import evaluate

    tfg = config["tfg"]
    args = argparse.Namespace(
        run_root=root,
        output_root=root / "10_official_syncnet",
        syncnet_root=REPO_ROOT / "third_party/syncnet_python",
        syncnet_python=Path(str(tfg.get("syncnet_python", "/home/wjj/.venvs/syncnet/bin/python"))),
        model=REPO_ROOT / str(tfg["syncnet_checkpoint"]),
        ffmpeg=Path(str(shutil.which("ffmpeg") or "/usr/bin/ffmpeg")),
        ffprobe=Path(str(shutil.which("ffprobe") or "/usr/bin/ffprobe")),
        min_track=25,
        pipeline_timeout=900,
        syncnet_timeout=600,
        limit=None,
        resume=True,
        portrait_id=str(tfg.get("primary_portrait", "3")),
        seed=int(tfg.get("seed", 42)),
        gpu_index=int(config.get("runtime", {}).get("gpu_index", 0)),
        min_free_gpu_mib=float(config.get("runtime", {}).get("min_free_gpu_gib_render", 5.0)) * 1024.0,
        allow_process_name=list(config.get("runtime", {}).get("gpu_process_allowlist", [])),
        skip_gpu_check=False,
    )
    try:
        result = evaluate(args)
    except Exception as exc:
        result = {"status": "ENGINEERING_FAILURE", "error": f"{type(exc).__name__}:{exc}"}
    state = "COMPLETE" if result.get("status") == "COMPLETE" else "PARTIAL"
    _set_state(root, "official", state, reasons=[] if state == "COMPLETE" else ["OFFICIAL_SCOPE_INCOMPLETE"])
    return result


def stage_report(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    status = _load_status(root)
    analysis = read_json(root / "08_analysis/summary.json") if (root / "08_analysis/summary.json").is_file() else {}
    result = write_report(root, config=config, statuses=status.get("states", {}), analysis=analysis)
    _set_state(root, "report", "COMPLETE" if result.get("status") == "COMPLETE" else "PARTIAL", details={"decision": result})
    return result


STAGE_FUNCTIONS = {"audit": stage_audit, "calibrate": stage_calibrate, "train": stage_train, "lock": stage_lock, "infer": stage_infer, "quality": stage_quality, "phone": stage_phone, "render": stage_render, "score": stage_score, "official": stage_official, "analyze": stage_analyze, "report": stage_report}


def run(config: Mapping[str, Any], config_path: Path, root: Path, stage: str, *, resume: bool) -> dict[str, Any]:
    if stage == "all":
        result = {}
        for name in STAGES:
            try:
                if name == "audit": value = stage_audit(config, config_path, root, resume=resume)
                elif name == "calibrate": value = stage_calibrate(config, root)
                else:
                    _assert_protocol(config, root)
                    value = STAGE_FUNCTIONS[name](config, root)
                result[name] = value
                state = _load_status(root).get("states", {}).get(name, {}).get("state")
                if name in {"audit", "calibrate"} and state != "COMPLETE":
                    # Do not launch dependent GPU stages after an input or
                    # instrument gate failed.  The partial artifact remains
                    # resumable after the external dependency is repaired.
                    break
            except Exception as exc:
                result[name] = {"status": "ENGINEERING_FAILURE", "error": f"{type(exc).__name__}:{exc}"}
                _set_state(root, name, "ENGINEERING_FAILURE", reasons=[f"{type(exc).__name__}:{exc}"])
        return result
    if stage == "audit":
        return stage_audit(config, config_path, root, resume=resume)
    _assert_protocol(config, root)
    return STAGE_FUNCTIONS[stage](config, root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/configs/phone_gain_static_tfg_mfa_v1.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="audit")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_yaml(config_path)
    root = _run_dir(args.run_id)
    result = run(config, config_path, root, args.stage, resume=bool(args.resume))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run"]
