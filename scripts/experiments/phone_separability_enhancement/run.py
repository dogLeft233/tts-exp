"""Runner for the constrained natural-clock phone-enhancement protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_phone_rules_worker import read_pcm16, write_pcm16

from . import MEASUREMENT_VERSION, PROTOCOL_ID
from .audio import BandGainRenderer, build_gain_control, export_pcm, make_protected_mask
from .config import ensure_protocol_unchanged, file_sha256, load_yaml, protocol_snapshot, read_json, read_jsonl, resolve_path, validate_run_id, write_json, write_jsonl
from .data import FeatureStore, freeze_support, get_side, inherit_registry, load_feature_store, pair_index, select_pilot, write_support
from .mechanisms import codec_status
from .metrics import build_abx_triplets, score_fixed_support
from .optimize import optimize_utterance
from .quality import audit_timing, blind_pack, pcm_contract
from .teacher import load_frozen_teacher, phone_margins_from_hidden, pool_hidden, processor_parity
from .train import BoundedGainEnhancer, evaluate_model, load_checkpoint, select_dev_candidate, train_enhancer


REPO_ROOT = Path(__file__).resolve().parents[3]


class ResourceBusy(RuntimeError):
    pass


class DependencyBlocked(RuntimeError):
    pass


class InputInvalid(RuntimeError):
    pass


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "status.json"
    current = dict(read_json(path)) if path.is_file() else {"schema_version": 2, "protocol": PROTOCOL_ID, "measurement_version": MEASUREMENT_VERSION, "execution": "NOT_STARTED", "stage_states": {}, "reason_codes": []}
    for key, value in updates.items():
        if key == "stage_states" and isinstance(value, Mapping):
            current[key] = {**current.get(key, {}), **dict(value)}
        elif key == "reason_codes":
            current[key] = sorted(set(current.get(key, [])) | set(value or []))
        else:
            current[key] = value
    write_json(path, current)
    return current


def _resource_snapshot(repo_root: Path) -> dict[str, Any]:
    usage = __import__("shutil").disk_usage(repo_root)
    snapshot: dict[str, Any] = {"free_gib": usage.free / 1024**3, "total_gib": usage.total / 1024**3, "pid": os.getpid(), "gpu": {"devices": [], "compute_apps": []}}
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        available = next(float(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemAvailable:"))
        snapshot["ram_available_gib"] = available / 1024**2
    except (OSError, StopIteration, ValueError):
        snapshot["ram_available_gib"] = None
    if __import__("shutil").which("nvidia-smi"):
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        devices = []
        for line in query.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) >= 6:
                devices.append({"index": fields[0], "uuid": fields[1], "memory_total_mib": fields[2], "memory_used_mib": fields[3], "memory_free_mib": fields[4], "utilization_gpu": fields[5]})
        app_rows = []
        for line in apps.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if fields:
                try:
                    pid = int(fields[0])
                except ValueError:
                    pid = None
                app_rows.append({"pid": pid, "raw": line.strip()})
        snapshot["gpu"] = {"devices": devices, "compute_apps": app_rows}
    return snapshot


def _require_resources(config: Mapping[str, Any], run_dir: Path, *, device: str, estimate_bytes: int = 0) -> dict[str, Any]:
    runtime = config.get("runtime", {})
    snapshot = _resource_snapshot(REPO_ROOT)
    reasons: list[str] = []
    min_disk = max(float(runtime.get("min_free_disk_gib", 4.0)), estimate_bytes * 1.2 / 1024**3)
    if float(snapshot["free_gib"]) < min_disk:
        reasons.append("DISK_BELOW_MINIMUM")
    if snapshot.get("ram_available_gib") is not None and float(snapshot["ram_available_gib"]) < float(runtime.get("min_available_ram_gib", 8.0)):
        reasons.append("RAM_BELOW_MINIMUM")
    if str(device).startswith("cuda"):
        index = str(device).split(":")[-1]
        devices = [row for row in snapshot.get("gpu", {}).get("devices", []) if str(row.get("index")) == index]
        external_apps = [row for row in snapshot.get("gpu", {}).get("compute_apps", []) if row.get("pid") not in {os.getpid(), os.getppid()}]
        if external_apps:
            reasons.append("GPU_COMPUTE_BUSY")
        if not devices:
            reasons.append("GPU_DEVICE_UNAVAILABLE")
        else:
            try:
                if float(devices[0]["memory_free_mib"]) < float(runtime.get("min_free_gpu_gib", 12.0)) * 1024:
                    reasons.append("GPU_FREE_MEMORY_BELOW_MINIMUM")
            except (KeyError, TypeError, ValueError):
                reasons.append("GPU_MEMORY_UNREADABLE")
    snapshot["required_free_disk_gib"] = min_disk
    snapshot["decision"] = "RESOURCE_BUSY" if reasons else "ALLOW"
    snapshot["reason_codes"] = reasons
    write_json(run_dir / "00_audit" / "resources.json", snapshot)
    if reasons:
        raise ResourceBusy(";".join(reasons))
    return snapshot


def _load_context(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = dict(read_json(run_dir / "00_audit" / "registry.json"))
    support = dict(read_json(run_dir / "00_audit" / "support_plan.json"))
    protocol = dict(read_json(run_dir / "protocol.json"))
    return registry, support, protocol


def _selected_pairs(registry: Mapping[str, Any], pilot: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    ids = set(str(value) for value in pilot.get(split, []))
    return [dict(row) for row in registry.get("pairs", []) if str(row.get("pair_id")) in ids]


def _cached_rows(parent_run: Path, encoder: str, support: Mapping[str, Any], *, condition: str, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    store = load_feature_store(parent_run, encoder, "core")
    row_map = {(str(row.get("sample_id")), str(row.get("condition")), int(row.get("token_index", -1))): row for row in store.records}
    expected = [dict(row) for row in support.get("entries", {}).get(encoder, []) if str(row.get("analysis_split")) == split]
    actual: list[dict[str, Any]] = []
    for entry in expected:
        token_index = int(entry["natural_token_index"] if condition == "natural" else entry["tts_token_index"])
        row = row_map.get((str(entry["pair_id"]), condition, token_index))
        if row is None:
            continue
        value = store.vector(row)
        if value is None:
            continue
        item = dict(row)
        item.update({"support_key": entry["support_key"], "occurrence_key": f"{entry['pair_id']}:{token_index}", "label": entry["label"], "source_group": entry["source_group"], "embedding": value.tolist()})
        actual.append(item)
    return actual, expected


def _score_cached_baseline(parent_run: Path, encoder: str, support: Mapping[str, Any], *, condition: str, split: str, center_kind: str = "mixed_centroids") -> dict[str, Any]:
    rows, expected = _cached_rows(parent_run, encoder, support, condition=condition, split=split)
    centers = support.get(center_kind, {}).get(encoder, {})
    return score_fixed_support(rows, centers, expected=expected, condition=condition)


def _score_waveform(teacher: Any, audio_pcm: np.ndarray, tokens: Sequence[Mapping[str, Any]], *, pair_id: str, support_entries: Sequence[Mapping[str, Any]], condition: str, centroids: Mapping[str, Sequence[float]], layer: int, view: str = "core") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = __import__("torch")
    waveform = torch.as_tensor(np.asarray(audio_pcm, dtype=np.float32) / 32768.0, device=teacher.device)
    hidden, frame_times, _ = teacher.encode(waveform, layer=layer)
    pooled = pool_hidden(hidden, frame_times, tokens, view=view)
    rows: list[dict[str, Any]] = []
    for entry in support_entries:
        token_index = int(entry["natural_token_index"] if condition == "natural" else entry["tts_token_index"])
        token = next((row for row in pooled.values() if int(row["token_index"]) == token_index), None)
        if token is None:
            continue
        rows.append({"support_key": entry["support_key"], "occurrence_key": f"{pair_id}:{token_index}", "label": entry["label"], "source_group": entry["source_group"], "embedding": token["embedding"].detach().cpu().numpy().tolist(), "valid": True, "speech": True})
    return score_fixed_support(rows, centroids, expected=list(support_entries)), rows


def stage_audit(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    _require_resources(config, run_dir, device="cpu", estimate_bytes=256 * 1024**2)
    registry = inherit_registry(config, REPO_ROOT, run_dir, smoke=smoke)
    pilot = select_pilot(registry, config, smoke=smoke)
    parent_run = resolve_path(str(config.get("parent_run", "")), REPO_ROOT)
    if not parent_run.is_dir():
        raise InputInvalid(f"parent run is missing: {parent_run}")
    support = freeze_support(registry, config, parent_run=parent_run, pilot=pilot)
    audit = run_dir / "00_audit"
    write_json(audit / "pilot.json", pilot)
    write_support(run_dir, support)
    parent_protocol = parent_run / "protocol.json"
    write_json(audit / "parent_binding.json", {"parent_run": str(parent_run), "protocol_sha256": file_sha256(parent_protocol) if parent_protocol.is_file() else None, "registry_sha256": file_sha256(parent_run / "00_inventory/registry.json") if (parent_run / "00_inventory/registry.json").is_file() else None, "support_hash": support["support_hash"]})
    dependencies = {"python": sys.version, "packages": {}, "parent_features": {}, "historical_exposure": {"xlsr_e_seen_seen_before": True, "new_e_scores_locked_until": "04_lock/selection_lock.json"}}
    for package in ("numpy", "torch", "transformers", "PyYAML"):
        try:
            dependencies["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies["packages"][package] = None
    for encoder in ("hubert", "xlsr"):
        feature_dir = parent_run / "02_features" / encoder
        dependencies["parent_features"][encoder] = {"exists": feature_dir.is_dir(), "meta_sha256": file_sha256(feature_dir / "feature_meta.json") if (feature_dir / "feature_meta.json").is_file() else None}
    write_json(audit / "dependencies.json", dependencies)
    write_json(audit / "exposure_ledger.json", {"status": "EXPLORATORY_HISTORIC_EXPOSURE", "historical_parent_run": str(parent_run), "xlsr_e_seen": "previously inspected in v1", "new_candidate_xlsr_before_lock": False})
    write_json(audit / "budget.json", {"max_new_artifacts_gib": float(config.get("runtime", {}).get("max_new_artifacts_gib", 3.0)), "gpu_hours_total": 16, "stages": {"calibration_mechanisms": 4, "optimization": 8, "training": 4}})
    _write_status(run_dir, execution="RUNNING", engineering="AUDIT_COMPLETE", stage="audit", stage_states={"audit": "COMPLETE"}, measurement="NOT_RUN", optimization="NOT_RUN", phone_gain="NOT_RUN", timing="NOT_RUN", content="NOT_RUN", human_quality="NOT_ASSESSED", generalization="NOT_RUN", mechanism="NOT_RUN", downstream="NOT_RUN", reason_codes=[])
    return {"registry": registry, "pilot": pilot, "support": support}


def stage_calibrate(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    registry, support, _ = _load_context(run_dir)
    parent_run = resolve_path(str(config.get("parent_run", "")), REPO_ROOT)
    calibration_dir = run_dir / "01_calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    baseline: dict[str, Any] = {}
    for encoder in ("hubert", "xlsr"):
        baseline[encoder] = {}
        for split in ("fit", "dev", "e_seen"):
            if smoke and split != "fit":
                continue
            baseline[encoder][split] = {condition: _score_cached_baseline(parent_run, encoder, support, condition=condition, split=split) for condition in ("natural", "tts")}
            write_json(calibration_dir / "probe" / encoder / f"{split}.json", baseline[encoder][split])
    triplets = {}
    for encoder in ("hubert", "xlsr"):
        rows = [{"source_group": row["source_group"], "label": row["label"], "occurrence_key": f"{row['pair_id']}:{row['natural_token_index']}", "token_id": row["natural_token_id"], "valid": True, "speech": True} for row in support.get("entries", {}).get(encoder, []) if not smoke or row.get("analysis_split") == "fit"]
        triplets[encoder] = build_abx_triplets(rows, triplets_per_group=int(config.get("probe", {}).get("abx_triplets_per_group", 20)), seed=int(config.get("probe", {}).get("bootstrap_seed", 20260921)))
    write_json(calibration_dir / "abx_triplets.json", triplets)
    legacy = {"parent_run": str(parent_run), "parent_strict_summary": read_json(parent_run / "03_atlas/strict_summary.json") if (parent_run / "03_atlas/strict_summary.json").is_file() else None, "legacy_margin_semantics": "top1_minus_top2", "v2_margin_semantics": "correct_minus_strongest_wrong", "abx_semantics": "A_and_X_must_be_distinct_occurrences"}
    write_json(calibration_dir / "legacy_metric_audit.json", legacy)
    teacher_result: dict[str, Any] = {"status": "NOT_RUN"}
    gradient_result: dict[str, Any] = {"status": "NOT_RUN"}
    identity_result: dict[str, Any] = {"status": "NOT_RUN"}
    model_cfg = config.get("models", {}).get("primary", {})
    device = str(config.get("runtime", {}).get("device", config.get("models", {}).get("device", "cuda:0")))
    try:
        _require_resources(config, run_dir, device=device, estimate_bytes=512 * 1024**2)
        teacher = load_frozen_teacher(model_cfg, device=device, proxy=str(config.get("runtime", {}).get("proxy", "")) or None, allow_download=bool(config.get("runtime", {}).get("allow_model_download", True)))
        first_id = str((support.get("candidate_scope", {}).get("fit") or [])[0])
        pair = pair_index(registry).get(first_id)
        if pair is None:
            raise DependencyBlocked("no FIT calibration pair")
        audio, _ = read_pcm16(get_side(pair, "natural")["audio_path"])
        teacher_result = {"status": "PASS", "processor_parity": processor_parity(teacher, audio.astype(np.float32) / 32768.0)}
        import torch
        waveform = torch.as_tensor(audio.astype(np.float32) / 32768.0, device=device).requires_grad_(True)
        hidden, frame_times, _ = teacher.encode(waveform, layer=int(model_cfg.get("layer", 6)))
        loss, rows = phone_margins_from_hidden(hidden, frame_times, get_side(pair, "natural")["tokens"], support.get("mixed_centroids", {}).get("hubert", {}), view="core")
        if loss.requires_grad:
            loss.backward()
        teacher_grads = [parameter.grad for parameter in teacher.model.parameters() if parameter.grad is not None]
        gradient_result = {"status": "PASS" if waveform.grad is not None and torch.isfinite(waveform.grad).all() and not teacher_grads else "FAIL", "waveform_grad_norm": float(torch.linalg.vector_norm(waveform.grad).detach().cpu()) if waveform.grad is not None else None, "teacher_parameter_grads": len(teacher_grads), "phone_tokens": len(rows)}
        renderer = BandGainRenderer(max_gain_db=6.0)
        info = make_protected_mask(get_side(pair, "natural")["tokens"], audio.size)
        x = torch.as_tensor(audio.astype(np.float32) / 32768.0, device=device)
        _, spectrum = renderer.band_features(x)
        zero = torch.zeros((1, 24, spectrum.shape[-1]), device=device)
        y, render_meta = renderer.render(x, zero, torch.as_tensor(info["mask"], device=device))
        pcm, pcm_meta = export_pcm(y, audio, protected=info["protected"])
        identity_result = {"status": "PASS" if int(np.max(np.abs(pcm.astype(np.int32) - audio.astype(np.int32)))) <= 1 else "FAIL", "max_abs_pcm_error": int(np.max(np.abs(pcm.astype(np.int32) - audio.astype(np.int32)))), "render": render_meta, "pcm": pcm_meta}
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (OSError, RuntimeError, ValueError, KeyError, DependencyBlocked) as exc:
        teacher_result = {"status": "DEPENDENCY_BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}
        gradient_result = {"status": "NOT_RUN", "reason": "TEACHER_UNAVAILABLE"}
        identity_result = {"status": "NOT_RUN", "reason": "TEACHER_UNAVAILABLE"}
    timing = {"status": "NOT_RUN", "reason": "run_mfa=false"} if not bool(config.get("quality", {}).get("run_mfa", False)) else {"status": "DEPENDENCY_BLOCKED", "reason": "MFA runner not wired in this revision"}
    calibration = {"schema_version": 2, "protocol": PROTOCOL_ID, "measurement_version": MEASUREMENT_VERSION, "baseline": baseline, "teacher_parity": teacher_result, "gradient_check": gradient_result, "stft_identity": identity_result, "timing_controls": timing, "status": "PASS" if teacher_result.get("status") == "PASS" and gradient_result.get("status") == "PASS" and identity_result.get("status") == "PASS" else "BLOCKED"}
    write_json(calibration_dir / "calibration.json", calibration)
    _write_status(run_dir, execution="RUNNING", stage="calibrate", stage_states={"calibrate": "COMPLETE" if calibration["status"] == "PASS" else "DEPENDENCY_BLOCKED"}, measurement=calibration["status"], timing=timing["status"], reason_codes=[] if calibration["status"] == "PASS" else ["CALIBRATION_BLOCKED"])
    return calibration


def _save_optimization_result(run_dir: Path, pair: Mapping[str, Any], result: Mapping[str, Any], *, split: str, arm: str, source_pcm: np.ndarray, tokens: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = run_dir / "02_optimize" / split / arm
    base.mkdir(parents=True, exist_ok=True)
    pair_id = str(pair["pair_id"])
    pcm = np.asarray(result.get("selected_pcm", source_pcm), dtype=np.int16)
    pcm_path = base / f"{pair_id}.wav"
    pcm_meta = write_pcm16(pcm_path, pcm)
    mask_info = make_protected_mask(tokens, source_pcm.size)
    np.savez_compressed(base / f"{pair_id}.npz", parameters=np.asarray(result.get("parameters", []), dtype=np.float32), protected=mask_info["protected"].astype(np.uint8), edit_mask=mask_info["mask"].astype(np.float32))
    control_path = None
    control_meta: dict[str, Any] | None = None
    if arm not in {"N_ID", "STFT_RT"}:
        try:
            control, control_meta = build_gain_control(source_pcm, pcm, mask_info["mask"])
            control_path = base / f"{pair_id}_GAIN.wav"
            write_pcm16(control_path, control)
            control_meta["status"] = "GAIN_CONTROL_AVAILABLE" if float(control_meta.get("relative_energy_error", 1.0)) <= 1e-3 else "GAIN_CONTROL_UNAVAILABLE"
        except (ValueError, OSError) as exc:
            control_meta = {"status": "GAIN_CONTROL_UNAVAILABLE", "reason": f"{type(exc).__name__}:{exc}"}
    row = {"protocol": PROTOCOL_ID, "measurement_version": MEASUREMENT_VERSION, "pair_id": pair_id, "source_group": pair["source_group"], "split": split, "arm": arm, "method": "DIRECT", "input_mode": "natural_plus_phone_labels", "status": result.get("status"), "steps": result.get("steps"), "selected_step": result.get("selected_step"), "selected_score": result.get("selected_score"), "selected_distortion": result.get("selected_distortion"), "output_path": str(pcm_path), "output_pcm_sha256": pcm_meta["pcm_sha256"], "output_container_sha256": pcm_meta["container_sha256"], "gain_field_path": str(base / f"{pair_id}.npz"), "gain_control_path": str(control_path) if control_path else None, "gain_control": control_meta, "effective_edit": bool(result.get("selected_render", {}).get("effective_edit", False)), "support_hash": None, "reason": result.get("status")}
    write_json(base / f"{pair_id}.json", {**row, "history": result.get("history", [])})
    return row


def stage_optimize(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    registry, support, _ = _load_context(run_dir)
    calibration = read_json(run_dir / "01_calibration/calibration.json")
    if calibration.get("status") != "PASS":
        decision = {"status": "DEPENDENCY_BLOCKED", "reason_codes": ["CALIBRATION_BLOCKED"], "rows": []}
        write_json(run_dir / "02_optimize/summary.json", decision)
        _write_status(run_dir, execution="RUNNING", stage="optimize", stage_states={"optimize": "DEPENDENCY_BLOCKED"}, optimization="DEPENDENCY_BLOCKED", reason_codes=decision["reason_codes"])
        return decision
    device = str(config.get("runtime", {}).get("device", "cuda:0"))
    _require_resources(config, run_dir, device=device, estimate_bytes=1024 * 1024**2)
    teacher = load_frozen_teacher(config["models"]["primary"], device=device, proxy=str(config.get("runtime", {}).get("proxy", "")) or None, allow_download=bool(config.get("runtime", {}).get("allow_model_download", True)))
    lookup = pair_index(registry)
    pilot = dict(read_json(run_dir / "00_audit/pilot.json"))
    arms = list(config.get("optimization", {}).get("arms", []))
    rows: list[dict[str, Any]] = []
    for split in ("fit", "dev"):
        for pair_id in pilot.get(split, []):
            pair = lookup.get(str(pair_id))
            if pair is None:
                continue
            side = get_side(pair, "natural")
            source_pcm, _ = read_pcm16(side["audio_path"])
            for arm in arms:
                max_gain = 3.0 if arm.endswith("_3") else 6.0
                result = optimize_utterance(source_pcm, side["tokens"], teacher, support.get("mixed_centroids", {}).get("hubert", {}), arm=str(arm), max_gain_db=max_gain, layer=int(config["models"]["primary"].get("layer", 6)), view="core", sample_rate=int(config.get("audio", {}).get("sample_rate", 16000)), node_ms=float(config.get("optimization", {}).get("dynamic_node_ms", 80.0)), lr=float(config.get("optimization", {}).get("lr", 0.05)), target_margin=float(config.get("optimization", {}).get("target_margin", 0.05)), temperature=float(config.get("optimization", {}).get("temperature", 0.1)), keep_weight=float(config.get("optimization", {}).get("keep_weight", 1.0)), tv_weight=float(config.get("optimization", {}).get("tv_weight", 0.01)), steps=10 if smoke else int(config.get("runtime", {}).get("optimize_steps", 120)), checkpoints=(0, 5, 10) if smoke else tuple(config.get("optimization", {}).get("checkpoints", [0, 25, 50, 75, 100, 120])), timeout_s=float(config.get("runtime", {}).get("optimize_timeout_s", 120)))
                rows.append(_save_optimization_result(run_dir, pair, result, split=split, arm=str(arm), source_pcm=source_pcm, tokens=side["tokens"]))
    write_jsonl(run_dir / "02_optimize/results.jsonl", rows)
    summary = {"status": "COMPLETE", "row_count": len(rows), "rows": rows, "arms": arms, "scope": "engineering_smoke" if smoke else "pilot_fit_dev", "label_adaptive": True}
    write_json(run_dir / "02_optimize/summary.json", summary)
    del teacher
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    _write_status(run_dir, execution="RUNNING", stage="optimize", stage_states={"optimize": "COMPLETE"}, optimization="COMPLETE", reason_codes=[])
    return summary


def _training_samples(registry: Mapping[str, Any], pilot: Mapping[str, Any], *, split: str) -> list[dict[str, Any]]:
    result = []
    lookup = pair_index(registry)
    for pair_id in pilot.get(split, []):
        pair = lookup.get(str(pair_id))
        if pair is None:
            continue
        side = get_side(pair, "natural")
        pcm, _ = read_pcm16(side["audio_path"])
        result.append({"pair_id": pair_id, "source_group": pair["source_group"], "audio_pcm": pcm, "tokens": side["tokens"], "view": "core", "layer": 6, "mask_info": make_protected_mask(side["tokens"], pcm.size)})
    return result


def stage_train(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    registry, support, _ = _load_context(run_dir)
    calibration = read_json(run_dir / "01_calibration/calibration.json")
    if calibration.get("status") != "PASS":
        decision = {"status": "DEPENDENCY_BLOCKED", "reason_codes": ["CALIBRATION_BLOCKED"], "seeds": config.get("runtime", {}).get("seeds", [])}
        write_json(run_dir / "03_train/decision.json", decision)
        _write_status(run_dir, execution="RUNNING", stage="train", stage_states={"train": "DEPENDENCY_BLOCKED"}, reason_codes=decision["reason_codes"])
        return decision
    device = str(config.get("runtime", {}).get("device", "cuda:0"))
    _require_resources(config, run_dir, device=device, estimate_bytes=512 * 1024**2)
    teacher = load_frozen_teacher(config["models"]["primary"], device=device, proxy=str(config.get("runtime", {}).get("proxy", "")) or None, allow_download=bool(config.get("runtime", {}).get("allow_model_download", True)))
    pilot = dict(read_json(run_dir / "00_audit/pilot.json"))
    fit_samples = _training_samples(registry, pilot, split="fit")
    dev_samples = _training_samples(registry, pilot, split="dev")
    seeds = [int(config.get("runtime", {}).get("seeds", [20260921, 20260922])[0])] if smoke else [int(value) for value in config.get("runtime", {}).get("seeds", [20260921, 20260922])]
    results = []
    for seed in seeds:
        output_dir = run_dir / "03_train" / f"seed_{seed}"
        result = train_enhancer(fit_samples, teacher, support.get("mixed_centroids", {}).get("hubert", {}), seed=seed, output_dir=output_dir, max_steps=101 if smoke else int(config.get("runtime", {}).get("train_max_steps", 1500)), timeout_s=900 if smoke else float(config.get("runtime", {}).get("train_timeout_s", 7200)), device=device)
        json_result = {key: value for key, value in result.items() if key not in {"model"}}
        dev_candidates: list[dict[str, Any]] = []
        if dev_samples:
            for checkpoint_path in result.get("checkpoint_paths", []):
                model = BoundedGainEnhancer(mel_bands=24, channels=64, max_gain_db=6.0).to(device)
                load_checkpoint(model, checkpoint_path, device=device)
                evaluated = evaluate_model(model, dev_samples, teacher, support.get("mixed_centroids", {}).get("hubert", {}), device=device)
                valid = [row for row in evaluated if row.get("signed_margin") is not None]
                candidate = {"split": "dev", "step": int(Path(checkpoint_path).stem.split("_")[-1]), "checkpoint": checkpoint_path, "accuracy": float(np.mean([row["accuracy"] for row in valid])) if valid else None, "signed_margin": float(np.mean([row["signed_margin"] for row in valid])) if valid else None, "t0_pass": bool(valid) and all(bool(row.get("t0_pass")) for row in valid), "distortion_pass": bool(valid) and all(bool(row.get("distortion_pass")) for row in valid), "residual_energy_ratio": float(np.mean([row.get("residual_energy_ratio", 1.0) for row in valid])) if valid else None, "coverage": len(valid) / max(len(dev_samples), 1)}
                dev_candidates.append(candidate)
                del model
            json_result["dev_candidates"] = dev_candidates
            json_result["dev_selected"] = select_dev_candidate(dev_candidates)
        else:
            json_result["dev_candidates"] = []
            json_result["dev_selected"] = None
            json_result["dev_selection_status"] = "NO_DEV_SMOKE"
        write_json(output_dir / "result.json", json_result)
        results.append(json_result)
    no_phone = None
    if not smoke and fit_samples:
        result = train_enhancer(fit_samples, teacher, support.get("mixed_centroids", {}).get("hubert", {}), seed=seeds[0], output_dir=run_dir / "03_train/no_phone", max_steps=100, timeout_s=900, no_phone=True, device=device)
        no_phone = {key: value for key, value in result.items() if key not in {"model"}}
        write_json(run_dir / "03_train/no_phone/result.json", no_phone)
    decision = {"status": "COMPLETE", "seeds": results, "no_phone": no_phone, "fit_sample_count": len(fit_samples), "dev_sample_count": len(dev_samples), "training_does_not_depend_on_rule_gate": True}
    write_json(run_dir / "03_train/decision.json", decision)
    del teacher
    gc.collect()
    _write_status(run_dir, execution="RUNNING", stage="train", stage_states={"train": "COMPLETE"}, reason_codes=[])
    return decision


def stage_lock(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    optimize = read_json(run_dir / "02_optimize/summary.json") if (run_dir / "02_optimize/summary.json").is_file() else {"rows": []}
    train = read_json(run_dir / "03_train/decision.json") if (run_dir / "03_train/decision.json").is_file() else {"status": "NOT_RUN"}
    dev_rows = [row for row in optimize.get("rows", []) if row.get("split") == "dev" and str(row.get("arm", "")).startswith("OPT_") and str(row.get("arm")) != "LABEL_PERM_6"]
    eligible = [row for row in dev_rows if row.get("selected_distortion", {}).get("pass", False) and row.get("selected_score", {}).get("signed_margin") is not None]
    eligible.sort(key=lambda row: (-float(row["selected_score"]["signed_margin"]), -float(row["selected_score"].get("accuracy", -1.0)), float(row.get("selected_distortion", {}).get("residual_energy_ratio", 1.0)), str(row["arm"]), str(row["pair_id"])))
    arm_scores: dict[str, list[float]] = {}
    for row in eligible:
        arm_scores.setdefault(str(row["arm"]), []).append(float(row["selected_score"]["signed_margin"]))
    arm_rank = sorted(((float(np.mean(values)), arm) for arm, values in arm_scores.items()), reverse=True)
    selected_direct = arm_rank[0][1] if arm_rank else None
    learned_candidates = []
    for seed_result in train.get("seeds", []):
        selected = seed_result.get("dev_selected")
        if selected is not None:
            learned_candidates.append({"seed": seed_result.get("seed"), **dict(selected)})
    selected_learned = learned_candidates[0] if learned_candidates else None
    lock = {"schema_version": 2, "protocol": PROTOCOL_ID, "measurement_version": MEASUREMENT_VERSION, "status": "LOCKED_BEFORE_XLSR_E", "selected_direct_arm": selected_direct, "direct_arm_scores_dev": {arm: {"mean_signed_margin": float(np.mean(values)), "n": len(values)} for arm, values in arm_scores.items()}, "selected_learned": selected_learned, "learned": {"status": "DEV_CHECKPOINT_SELECTED" if selected_learned else ("NO_DEV_CHECKPOINT" if train.get("status") == "COMPLETE" else train.get("status")), "candidates": learned_candidates}, "xlsr_and_e_seen_opened": False, "selection_rule": "HuBERT DEV T0/distortion then signed_margin, accuracy, residual_energy, arm, pair_id", "learned_selection_rule": "first seed only; step0/every100; T0/distortion then signed_margin, accuracy, residual_energy, step", "support_hash": read_json(run_dir / "00_audit/support_plan.json")["support_hash"], "protocol_hash": file_sha256(run_dir / "protocol.json"), "smoke": bool(smoke)}
    write_json(run_dir / "04_lock/selection_lock.json", lock)
    write_json(run_dir / "04_lock/eval_plan.json", {"status": "LOCKED", "e_seen": not smoke, "encoders": ["hubert", "xlsr"], "new_xlsr_scores_after_lock": True, "direct_label_adaptation": True, "learned_fixed_forward": True})
    _write_status(run_dir, execution="RUNNING", stage="lock", stage_states={"lock": "COMPLETE"}, generalization="LOCKED_EXPLORATORY", reason_codes=[])
    return lock


def _candidate_row_from_pcm(teacher: Any, pair: Mapping[str, Any], support: Mapping[str, Any], pcm: np.ndarray, *, encoder: str, condition: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries = [row for row in support.get("entries", {}).get(encoder, []) if str(row.get("pair_id")) == str(pair["pair_id"])]
    model_cfg = {"hubert": {"layer": 6}, "xlsr": {"layer": 10}}[encoder]
    return _score_waveform(teacher, pcm, get_side(pair, "natural")["tokens"], pair_id=str(pair["pair_id"]), support_entries=entries, condition=condition, centroids=support.get("mixed_centroids", {}).get(encoder, {}), layer=model_cfg["layer"])


def stage_evaluate(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    registry, support, _ = _load_context(run_dir)
    lock = read_json(run_dir / "04_lock/selection_lock.json")
    if lock.get("status") != "LOCKED_BEFORE_XLSR_E":
        raise InputInvalid("selection lock missing")
    if smoke:
        result = {"status": "ENGINEERING_SMOKE_NO_E", "reason_codes": ["SMOKE_NO_E_SCIENTIFIC_EVALUATION"]}
        write_json(run_dir / "05_evaluation/summary.json", result)
        _write_status(run_dir, execution="RUNNING", stage="evaluate", stage_states={"evaluate": "COMPLETE"}, generalization="SMOKE_ONLY", reason_codes=result["reason_codes"])
        return result
    device = str(config.get("runtime", {}).get("device", "cuda:0"))
    _require_resources(config, run_dir, device=device, estimate_bytes=1024 * 1024**2)
    teacher = load_frozen_teacher(config["models"]["primary"], device=device, proxy=str(config.get("runtime", {}).get("proxy", "")) or None, allow_download=bool(config.get("runtime", {}).get("allow_model_download", True)))
    lookup = pair_index(registry)
    rows = []
    selected_arm = lock.get("selected_direct_arm")
    if selected_arm:
        for pair_id in read_json(run_dir / "00_audit/pilot.json").get("e_seen", []):
            pair = lookup.get(str(pair_id))
            if pair is None:
                continue
            side = get_side(pair, "natural")
            source_pcm, _ = read_pcm16(side["audio_path"])
            result = optimize_utterance(source_pcm, side["tokens"], teacher, support.get("mixed_centroids", {}).get("hubert", {}), arm=str(selected_arm), max_gain_db=3.0 if str(selected_arm).endswith("_3") else 6.0, layer=6, view="core", steps=int(config.get("runtime", {}).get("optimize_steps", 120)), timeout_s=float(config.get("runtime", {}).get("optimize_timeout_s", 120)))
            candidate = np.asarray(result.get("selected_pcm", source_pcm), dtype=np.int16)
            output_dir = run_dir / "05_evaluation" / "direct" / str(selected_arm)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{pair_id}.wav"
            pcm_meta = write_pcm16(path, candidate)
            mask = make_protected_mask(side["tokens"], source_pcm.size)
            quality = pcm_contract(source_pcm, candidate, mask["protected"])
            hubert_candidate, candidate_rows = _candidate_row_from_pcm(teacher, pair, support, candidate, encoder="hubert", condition="natural")
            hubert_natural, _ = _candidate_row_from_pcm(teacher, pair, support, source_pcm, encoder="hubert", condition="natural")
            rows.append({"pair_id": pair_id, "source_group": pair["source_group"], "arm": selected_arm, "method": "DIRECT", "baseline": hubert_natural, "candidate": hubert_candidate, "quality": quality, "output_path": str(path), "output_pcm_sha256": pcm_meta["pcm_sha256"], "candidate_rows": candidate_rows})
    selected_learned = lock.get("selected_learned")
    if selected_learned:
        import torch
        model = BoundedGainEnhancer(mel_bands=24, channels=64, max_gain_db=6.0).to(device)
        load_checkpoint(model, str(selected_learned["checkpoint"]), device=device)
        model.eval()
        learned_dir = run_dir / "05_evaluation" / "learned" / f"seed_{selected_learned.get('seed')}_step_{selected_learned.get('step')}"
        for pair_id in read_json(run_dir / "00_audit/pilot.json").get("e_seen", []):
            pair = lookup.get(str(pair_id))
            if pair is None:
                continue
            side = get_side(pair, "natural")
            source_pcm, _ = read_pcm16(side["audio_path"])
            sample = {"pair_id": pair_id, "source_group": pair["source_group"], "audio_pcm": source_pcm, "tokens": side["tokens"], "view": "core", "layer": 6, "mask_info": make_protected_mask(side["tokens"], source_pcm.size)}
            evaluated = evaluate_model(model, [sample], teacher, support.get("mixed_centroids", {}).get("hubert", {}), device=device)
            if not evaluated:
                continue
            item = evaluated[0]
            candidate = np.asarray(item.pop("pcm"), dtype=np.int16)
            learned_dir.mkdir(parents=True, exist_ok=True)
            path = learned_dir / f"{pair_id}.wav"
            pcm_meta = write_pcm16(path, candidate)
            hubert_candidate, candidate_rows = _candidate_row_from_pcm(teacher, pair, support, candidate, encoder="hubert", condition="natural")
            hubert_natural, _ = _candidate_row_from_pcm(teacher, pair, support, source_pcm, encoder="hubert", condition="natural")
            rows.append({"pair_id": pair_id, "source_group": pair["source_group"], "arm": f"LEARNED_SEED_{selected_learned.get('seed')}", "method": "LEARNED", "baseline": hubert_natural, "candidate": hubert_candidate, "quality": pcm_contract(source_pcm, candidate, make_protected_mask(side["tokens"], source_pcm.size)["protected"]), "output_path": str(path), "output_pcm_sha256": pcm_meta["pcm_sha256"], "candidate_rows": candidate_rows, "dev_selected": selected_learned})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # The cross encoder is opened only after the lock. It is intentionally a
    # separate load and score pass, never consulted by the preceding selection.
    xlsr = load_frozen_teacher(config["models"]["cross_encoder"], device=device, proxy=str(config.get("runtime", {}).get("proxy", "")) or None, allow_download=bool(config.get("runtime", {}).get("allow_model_download", True)))
    for row in rows:
        pair = lookup[str(row["pair_id"])]
        source_pcm, _ = read_pcm16(get_side(pair, "natural")["audio_path"])
        candidate_pcm, _ = read_pcm16(row["output_path"])
        row["xlsr_baseline"], _ = _candidate_row_from_pcm(xlsr, pair, support, source_pcm, encoder="xlsr", condition="natural")
        row["xlsr_candidate"], _ = _candidate_row_from_pcm(xlsr, pair, support, candidate_pcm, encoder="xlsr", condition="natural")
    summary = {"status": "COMPLETE", "selected_direct_arm": selected_arm, "rows": rows, "new_xlsr_after_lock": True, "coverage": len(rows) / max(len(read_json(run_dir / "00_audit/pilot.json").get("e_seen", [])), 1), "classification": "EXPLORATORY_ONLY_HISTORIC_E_EXPOSURE"}
    write_json(run_dir / "05_evaluation/summary.json", summary)
    del teacher, xlsr
    gc.collect()
    _write_status(run_dir, execution="RUNNING", stage="evaluate", stage_states={"evaluate": "COMPLETE"}, generalization="EXPLORATORY_CROSS_ENCODER_PENDING", reason_codes=[])
    return summary


def stage_mechanisms(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    lock = read_json(run_dir / "04_lock/selection_lock.json")
    codec = codec_status(REPO_ROOT, config)
    evidence = {"schema_version": 2, "protocol": PROTOCOL_ID, "static_dynamic": {"status": "DIAGNOSTIC_ONLY", "selected_arm": lock.get("selected_direct_arm"), "dose_matching": "implemented_after_selected_gain_field_is_available"}, "codec": codec, "interpretation": "static/dynamic and codec contrasts are mechanism candidates, not proof of the TTS cause"}
    write_json(run_dir / "06_mechanisms/mechanism_evidence.json", evidence)
    _write_status(run_dir, execution="RUNNING", stage="mechanisms", stage_states={"mechanisms": "COMPLETE"}, mechanism="COMPLETE_WITH_OPTIONAL_CODEC_STATUS", reason_codes=[])
    return evidence


def stage_report(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    status = dict(read_json(run_dir / "status.json")) if (run_dir / "status.json").is_file() else {}
    calibration = read_json(run_dir / "01_calibration/calibration.json") if (run_dir / "01_calibration/calibration.json").is_file() else {}
    optimize = read_json(run_dir / "02_optimize/summary.json") if (run_dir / "02_optimize/summary.json").is_file() else {}
    train = read_json(run_dir / "03_train/decision.json") if (run_dir / "03_train/decision.json").is_file() else {}
    lock = read_json(run_dir / "04_lock/selection_lock.json") if (run_dir / "04_lock/selection_lock.json").is_file() else {}
    evaluate = read_json(run_dir / "05_evaluation/summary.json") if (run_dir / "05_evaluation/summary.json").is_file() else {}
    mechanisms = read_json(run_dir / "06_mechanisms/mechanism_evidence.json") if (run_dir / "06_mechanisms/mechanism_evidence.json").is_file() else {}
    report_dir = run_dir / "08_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# 自然时钟受约束音素增强 v2", "", f"- protocol: `{PROTOCOL_ID}`", f"- measurement: `{MEASUREMENT_VERSION}`", f"- scope: `{'engineering_smoke' if smoke else 'exploratory'}`", "", "## 工程状态", "", f"- calibration: `{calibration.get('status', 'NOT_RUN')}`", f"- optimize rows: `{optimize.get('row_count', 0)}`", f"- train: `{train.get('status', 'NOT_RUN')}`", f"- lock: `{lock.get('status', 'NOT_RUN')}`", f"- evaluation: `{evaluate.get('status', 'NOT_RUN')}`", "", "## 科学边界", "", "逐句 DIRECT 使用已知音素标签，只能回答教师目标下的局部可行性；LEARNED 才是固定增强器前向问题。XLSR/E_SEEN 结果在锁定后才打开，但父实验曾查看过相同 E_SEEN，因此本轮跨编码器结果仍标记为探索性。", "", "## 当前判定", "", f"- direct arm: `{lock.get('selected_direct_arm')}`", f"- generalization: `{status.get('generalization', 'NOT_RUN')}`", f"- mechanism: `{mechanisms.get('codec', {}).get('status', 'NOT_RUN')}`", "- TFG/SyncNet: `NOT_RUN`", ""]
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    decision = {"schema_version": 2, "protocol": PROTOCOL_ID, "status": "EXPLORATORY_REPORT_READY", "engineering": status.get("engineering"), "measurement": status.get("measurement"), "optimization": status.get("optimization"), "phone_gain": "NOT_DETERMINED", "timing": status.get("timing", "NOT_RUN"), "content": "NOT_ASSESSED", "human_quality": "HUMAN_NOT_ASSESSED", "generalization": status.get("generalization", "NOT_RUN"), "mechanism": status.get("mechanism", "NOT_RUN"), "downstream": "NOT_RUN", "reason_codes": ["HISTORIC_E_EXPOSURE", "NO_TFG_RUN"]}
    write_json(report_dir / "decision.json", decision)
    write_json(report_dir / "confirmation_plan.json", {"status": "NOT_STARTED", "reason": "new confirmation source groups required"})
    _write_status(run_dir, execution="COMPLETE", engineering="COMPLETE", stage="report", stage_states={"report": "COMPLETE"}, phone_gain=decision["phone_gain"], content=decision["content"], human_quality=decision["human_quality"], downstream=decision["downstream"], reason_codes=decision["reason_codes"])
    return decision


def _run_dir(run_id: str | None) -> Path:
    value = run_id or ("phone_separability_enhancement_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    validate_run_id(Path(value).name)
    if Path(value).is_absolute() or ".." in Path(value).parts or Path(value).name != value:
        raise InputInvalid("run-id must be a simple name below runs/")
    return REPO_ROOT / "runs" / value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/configs/phone_separability_enhancement_v2.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage", choices=("audit", "calibrate", "optimize", "train", "lock", "evaluate", "mechanisms", "report", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = _run_dir(args.run_id)
    config_path = resolve_path(args.config, REPO_ROOT)
    config = load_yaml(config_path)
    try:
        snapshot = protocol_snapshot(config, config_path, repo_root=REPO_ROOT)
        if run_dir.exists() and not args.resume:
            raise InputInvalid(f"run directory exists; choose a new run-id or use --resume: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            previous = read_json(run_dir / "protocol.json")
            ensure_protocol_unchanged(previous, snapshot)
        else:
            write_json(run_dir / "protocol.json", snapshot)
        if args.stage == "audit":
            stage_audit(config, run_dir, smoke=bool(args.smoke))
        else:
            if not (run_dir / "00_audit/registry.json").is_file():
                stage_audit(config, run_dir, smoke=bool(args.smoke))
            calibration_path = run_dir / "01_calibration/calibration.json"
            calibration_ok = calibration_path.is_file() and read_json(calibration_path).get("status") == "PASS"
            if args.stage in {"calibrate", "optimize", "train", "lock", "evaluate", "mechanisms", "report", "all"} and not calibration_ok:
                stage_calibrate(config, run_dir, smoke=bool(args.smoke))
            optimize_path = run_dir / "02_optimize/summary.json"
            optimize_ok = optimize_path.is_file() and read_json(optimize_path).get("status") == "COMPLETE"
            if args.stage in {"optimize", "train", "lock", "evaluate", "mechanisms", "report", "all"} and not optimize_ok:
                stage_optimize(config, run_dir, smoke=bool(args.smoke))
            train_path = run_dir / "03_train/decision.json"
            train_ok = train_path.is_file() and read_json(train_path).get("status") == "COMPLETE"
            if args.stage in {"train", "lock", "evaluate", "mechanisms", "report", "all"} and not train_ok:
                stage_train(config, run_dir, smoke=bool(args.smoke))
            lock_path = run_dir / "04_lock/selection_lock.json"
            lock_ok = lock_path.is_file() and read_json(lock_path).get("status") == "COMPLETE"
            if args.stage in {"lock", "evaluate", "mechanisms", "report", "all"} and not lock_ok:
                stage_lock(config, run_dir, smoke=bool(args.smoke))
            evaluation_path = run_dir / "05_evaluation/summary.json"
            evaluation_ok = evaluation_path.is_file() and read_json(evaluation_path).get("status") == "COMPLETE"
            if args.stage in {"evaluate", "mechanisms", "report", "all"} and not evaluation_ok:
                stage_evaluate(config, run_dir, smoke=bool(args.smoke))
            mechanism_path = run_dir / "06_mechanisms/mechanism_evidence.json"
            mechanism_ok = mechanism_path.is_file() and read_json(mechanism_path).get("schema_version") == 2
            if args.stage in {"mechanisms", "report", "all"} and not mechanism_ok:
                stage_mechanisms(config, run_dir, smoke=bool(args.smoke))
            if args.stage in {"report", "all"}:
                stage_report(config, run_dir, smoke=bool(args.smoke))
        print(f"run complete: {run_dir}")
        return 0
    except ResourceBusy as exc:
        _write_status(run_dir, execution="RESOURCE_BUSY", stage=args.stage, engineering="RESOURCE_BUSY", reason_codes=[str(exc)])
        print(f"RESOURCE_BUSY: {exc}", file=sys.stderr)
        return 2
    except DependencyBlocked as exc:
        _write_status(run_dir, execution="DEPENDENCY_BLOCKED", stage=args.stage, engineering="DEPENDENCY_BLOCKED", reason_codes=[str(exc)])
        print(f"DEPENDENCY_BLOCKED: {exc}", file=sys.stderr)
        return 2
    except (InputInvalid, OSError, ValueError, KeyError, RuntimeError) as exc:
        _write_status(run_dir, execution="IMPLEMENTATION_INVALID", stage=args.stage, engineering="IMPLEMENTATION_INVALID", reason_codes=[f"{type(exc).__name__}:{exc}"])
        print(f"IMPLEMENTATION_INVALID: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "stage_audit", "stage_calibrate", "stage_evaluate", "stage_lock", "stage_mechanisms", "stage_optimize", "stage_report", "stage_train"]
