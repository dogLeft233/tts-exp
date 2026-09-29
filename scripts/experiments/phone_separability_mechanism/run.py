"""Runner for ``phone_separability_mechanism_v1``.

Stages are deliberately resumable and scientifically conservative:
``audit`` never loads a model, ``atlas`` freezes references before scoring,
``construct`` never selects from E_SEEN, and ``train`` is conditional.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .audio import (
    build_edit_mask,
    build_speech_mask,
    fit_phone_shape_templates,
    make_gain_control,
    render_paired_shape,
    render_phone_shape,
    render_roundtrip,
    render_rule_arm,
    validate_waveform,
)
from .candidate import run_candidates
from .diagnostics import duration_analysis, historical_contrasts, mask_dose_audit
from .features import extract_atlas, load_token_records
from .inventory import InventoryError, check_resources, load_registry
from .metrics import (
    abx_score,
    contrast_scores,
    domain_auc,
    fit_probe_bundle,
    paired_group_bootstrap,
    score_frozen_support,
)
from .timing import waveform_contract
from .rescore import score_atlas_rows

from scripts.experiments.lrs3_phone_rules_worker import read_json, read_pcm16, sha256_file, write_pcm16


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "phone_separability_mechanism_v1"
PACKAGE_FILES = (
    "scripts/experiments/phone_separability_mechanism/__init__.py",
    "scripts/experiments/phone_separability_mechanism/inventory.py",
    "scripts/experiments/phone_separability_mechanism/features.py",
    "scripts/experiments/phone_separability_mechanism/metrics.py",
    "scripts/experiments/phone_separability_mechanism/diagnostics.py",
    "scripts/experiments/phone_separability_mechanism/audio.py",
    "scripts/experiments/phone_separability_mechanism/timing.py",
    "scripts/experiments/phone_separability_mechanism/train.py",
    "scripts/experiments/phone_separability_mechanism/check.py",
    "scripts/experiments/phone_separability_mechanism/run.py",
    "scripts/experiments/phone_separability_mechanism/rescore.py",
    "scripts/experiments/phone_separability_mechanism/continue_run.py",
    "scripts/experiments/phone_separability_mechanism/candidate.py",
)


class ExperimentError(RuntimeError):
    pass


class DependencyBlocked(ExperimentError):
    pass


class ResourceBusy(ExperimentError):
    pass


class InputInvalid(ExperimentError):
    pass


def _resolve(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = _resolve(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputInvalid(f"config unreadable: {config_path}") from exc
    if not isinstance(payload, dict):
        raise InputInvalid("config must be a mapping")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise InputInvalid(f"JSONL row is not an object: {path}")
            rows.append(value)
    return rows


def _git_snapshot() -> dict[str, Any]:
    import subprocess

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {"head": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain")), "status_porcelain": run("status", "--porcelain")}


def _code_hashes(config_path: Path) -> dict[str, str]:
    paths = [REPO_ROOT / relative for relative in PACKAGE_FILES] + [config_path]
    spec = REPO_ROOT / "basic-memory" / "Research" / "TTS 音素可分性机制与自然时钟增强探索 Implementation Spec.md"
    if spec.is_file():
        paths.append(spec)
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths if path.is_file()}


def _protocol(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "torch", "transformers", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_version": int(config.get("protocol_version", 1)),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "code_sha256": _code_hashes(config_path),
        "git": _git_snapshot(),
        "dependencies": versions,
        "models": config.get("models", {}),
        "probe": config.get("probe", {}),
        "audio": config.get("audio", {}),
        "source": config.get("source", {}),
        "frame_time_convention": "frontend_receptive_field_centers",
        "pcm_convention": "mono_16khz_pcm16_little_endian",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _verify_protocol(existing: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for key in ("protocol_id", "protocol_version", "config_sha256", "code_sha256", "models", "probe", "audio", "source", "frame_time_convention", "pcm_convention"):
        if existing.get(key) != current.get(key):
            raise InputInvalid(f"resume fingerprint mismatch: protocol.{key}; use a new run-id")


def _status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "status.json"
    current = dict(read_json(path)) if path.is_file() else {}
    current.update(updates)
    current.setdefault("schema_version", 1)
    current.setdefault("protocol_id", PROTOCOL_ID)
    current.setdefault("stage_states", {})
    current.setdefault("measurement", "NOT_RUN")
    current.setdefault("manipulation", "NOT_RUN")
    current.setdefault("phone_gain", "NOT_RUN")
    current.setdefault("timing", "NOT_RUN")
    current.setdefault("quality", "NOT_ASSESSED")
    current.setdefault("generalization", "NOT_RUN")
    current.setdefault("downstream", "OUT_OF_SCOPE")
    current["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(path, current)
    return current


def _load_registry(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "00_inventory" / "registry.json"
    if not path.is_file():
        raise InputInvalid("inventory missing; run audit first")
    return dict(read_json(path))


def _ensure_resources(config: Mapping[str, Any], run_dir: Path, *, device: str, estimate_bytes: int = 0) -> dict[str, Any]:
    runtime = config.get("runtime", {})
    if not bool(runtime.get("resource_check", True)):
        snapshot = {"decision": "CHECK_DISABLED_BY_CONFIG"}
    else:
        snapshot = check_resources(REPO_ROOT, device=device, min_free_disk_gib=float(runtime.get("min_free_disk_gib", 4.0)), min_free_gpu_gib=float(runtime.get("min_free_gpu_gib", 8.0)), min_available_ram_gib=float(runtime.get("min_available_ram_gib", 8.0)), stage_estimated_bytes=estimate_bytes)
        if snapshot.get("decision") != "ALLOW":
            _write_json(run_dir / "00_inventory" / "resources.json", snapshot)
            raise ResourceBusy(";".join(snapshot.get("reason_codes", ["RESOURCE_BUSY"])))
    _write_json(run_dir / "00_inventory" / "resources.json", snapshot)
    return snapshot


def stage_audit(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    resources = _ensure_resources(config, run_dir, device="cpu", estimate_bytes=32 * 1024 * 1024)
    try:
        registry = load_registry(config, run_dir, repo_root=REPO_ROOT, smoke=smoke)
    except InventoryError as exc:
        _status(run_dir, execution="INPUT_INVALID", stage="audit", reason_codes=[str(exc)])
        raise InputInvalid(str(exc)) from exc
    if registry.get("overlap_violations"):
        raise InputInvalid(f"overlap components cross analysis splits: {registry['overlap_violations']}")
    _write_json(run_dir / "00_inventory" / "path_resolution.json", {"status": "NO_AUTOMATIC_RECOVERY", "recovered": [], "missing": [row["asset_id"] for row in registry["assets"] if row["audit_status"] != "OK"]})
    _status(run_dir, execution="RUNNING", stage="audit", stage_states={"audit": "COMPLETE"}, engineering_pass=True, reason_codes=[], scope=registry.get("scope"), resource_decision=resources.get("decision"))
    return registry


def _model_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = config.get("models", {})
    result = []
    for key in ("primary", "cross_encoder"):
        value = models.get(key)
        if isinstance(value, Mapping) and value.get("key"):
            row = dict(value)
            row["diagnostic_layers"] = list(models.get("diagnostic_layers", {}).get(str(value["key"]), []))
            result.append(row)
    return result


def _atlas_rows(run_dir: Path, model_key: str) -> list[dict[str, Any]]:
    return load_token_records(run_dir / "02_features" / model_key / "token_records.jsonl")


def stage_atlas(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    registry = _load_registry(run_dir)
    device = str(config.get("models", {}).get("device", "cpu"))
    _ensure_resources(config, run_dir, device=device, estimate_bytes=1024 * 1024 * 1024)
    runtime = config.get("runtime", {})
    selected = registry.get("selected_pair_ids") if smoke else [str(row["pair_id"]) for row in registry.get("pairs", [])]
    model_summaries: dict[str, Any] = {}
    for model_cfg in _model_configs(config):
        try:
            extracted = extract_atlas(registry, model_cfg, run_dir / "02_features", device=device, allow_download=bool(runtime.get("allow_model_download", True)), proxy=str(runtime.get("proxy", "")) or None, selected_pair_ids=selected, include_diagnostics=False)
        except Exception as exc:
            raise DependencyBlocked(f"atlas model {model_cfg['key']} failed: {exc}") from exc
        rows = extracted["rows"]
        model_key = str(model_cfg["key"])
        scored = score_atlas_rows(rows, model_key=model_key, config=config, output_dir=run_dir / "03_atlas")
        model_summaries[model_key] = {
            "meta": extracted["meta"],
            **scored,
        }
        del extracted
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    summary = {"schema_version": 1, "protocol": PROTOCOL_ID, "scope": "engineering_smoke" if smoke else "registered_exploratory", "models": model_summaries, "evaluation_is_seen": True, "selection_uses_e_seen": False}
    _write_json(run_dir / "03_atlas" / "summary.json", summary)
    _status(run_dir, execution="RUNNING", stage="atlas", stage_states={"audit": "COMPLETE", "atlas": "COMPLETE"}, measurement="ATLAS_COMPLETE", engineering_pass=True, reason_codes=[])
    return summary


def _pair_by_id(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["pair_id"]): dict(row) for row in registry.get("pairs", [])}


def _construction_arms(config: Mapping[str, Any]) -> list[str]:
    arms = ["N_ID", "N_RT", "SPECTRAL_DRC", "PHONE_SHAPE_BETA05_CORE", "PHONE_SHAPE_BETA1_CORE", "PHONE_SHAPE_BETA05_SPEECH", "PHONE_SHAPE_BETA1_SPEECH", "EQ_MATCH", "SHUFFLED_LABEL"]
    if bool(config.get("construction", {}).get("run_paired_oracle", True)):
        arms.append("PAIRED_SHAPE_ORACLE")
    return arms


def _templates_for_arm(templates: Mapping[str, Any], arm: str) -> tuple[Mapping[str, Any], float, str]:
    if arm.endswith("BETA05_CORE"):
        return templates, 0.5, "core"
    if arm.endswith("BETA1_CORE"):
        return templates, 1.0, "core"
    if arm.endswith("BETA05_SPEECH"):
        return templates, 0.5, "speech"
    if arm.endswith("BETA1_SPEECH"):
        return templates, 1.0, "speech"
    return templates, 1.0, "speech"


def _shuffle_templates(templates: Mapping[str, Any]) -> dict[str, list[float]]:
    labels = sorted(str(label) for label in templates)
    values = [list(templates[label]) for label in labels]
    rng = np.random.Generator(np.random.PCG64(20260921))
    order = rng.permutation(len(labels)) if labels else np.asarray([], dtype=np.int64)
    return {label: values[int(order[index])] for index, label in enumerate(labels)}


def stage_construct(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    registry = _load_registry(run_dir)
    _ensure_resources(config, run_dir, device="cpu", estimate_bytes=512 * 1024 * 1024)
    pairs_by_id = _pair_by_id(registry)
    selected = [pairs_by_id[str(pair_id)] for pair_id in registry.get("selected_pair_ids", []) if str(pair_id) in pairs_by_id]
    candidate_pairs = [row for row in selected if row.get("analysis_split") in {"dev", "e_seen"}]
    if not candidate_pairs:
        candidate_pairs = selected[:2] if smoke else [row for row in registry.get("pairs", []) if row.get("analysis_split") == "dev"]
    audio_cfg = config.get("audio", {})
    templates = fit_phone_shape_templates(registry.get("pairs", []), sample_rate=int(audio_cfg.get("sample_rate", 16000)), n_fft=int(audio_cfg.get("n_fft", 512)), win_length=int(audio_cfg.get("win_length", 512)), hop_length=int(audio_cfg.get("hop_length", 128)), min_tokens=int(config.get("construction", {}).get("min_template_tokens", 20)), min_groups=int(config.get("construction", {}).get("min_template_groups", 3)))
    _write_json(run_dir / "04_mechanisms" / "phone_shape_templates.json", templates)
    shuffled = _shuffle_templates(templates.get("templates", {}))
    rows: list[dict[str, Any]] = []
    for pair in candidate_pairs:
        natural = pair["sides"]["natural"]
        tts = pair["sides"]["tts"]
        if natural.get("audit_status") != "OK":
            continue
        source, source_meta = read_pcm16(Path(str(natural["audio_path"])), sample_rate=int(audio_cfg.get("sample_rate", 16000)))
        donor = None
        if tts.get("audit_status") == "OK":
            donor, _ = read_pcm16(Path(str(tts["audio_path"])), sample_rate=int(audio_cfg.get("sample_rate", 16000)))
        tokens = natural.get("tokens", [])
        speech_mask = build_speech_mask(source.size, tokens, sample_rate=int(audio_cfg.get("sample_rate", 16000)), edge_guard_s=float(audio_cfg.get("edge_guard_s", 0.010)), taper_s=float(audio_cfg.get("taper_s", 0.005)))
        core_mask = build_edit_mask(source.size, tokens, sample_rate=int(audio_cfg.get("sample_rate", 16000)), edge_guard_s=float(audio_cfg.get("edge_guard_s", 0.010)), taper_s=float(audio_cfg.get("taper_s", 0.005)))
        rendered: dict[str, tuple[np.ndarray, dict[str, Any], np.ndarray]] = {}
        rendered["N_ID"] = (source.copy(), {"arm": "N_ID", "alpha": 0.0}, np.zeros(source.size, dtype=np.float64))
        rt, rt_meta = render_roundtrip(source, mask=speech_mask, sample_rate=int(audio_cfg.get("sample_rate", 16000)), n_fft=int(audio_cfg.get("n_fft", 512)), win_length=int(audio_cfg.get("win_length", 512)), hop_length=int(audio_cfg.get("hop_length", 128)))
        rendered["N_RT"] = (rt, rt_meta, speech_mask)
        old, old_meta = render_rule_arm(source, tokens, "spectral_drc", sample_rate=int(audio_cfg.get("sample_rate", 16000)), edge_guard_s=float(audio_cfg.get("edge_guard_s", 0.010)), taper_s=float(audio_cfg.get("taper_s", 0.005)), n_fft=int(audio_cfg.get("n_fft", 512)), win_length=int(audio_cfg.get("win_length", 512)), hop_length=int(audio_cfg.get("hop_length", 128)))
        old_meta = {**old_meta, "mask_mode": "core"}
        rendered["SPECTRAL_DRC"] = (old, old_meta, core_mask)
        for arm in ("PHONE_SHAPE_BETA05_CORE", "PHONE_SHAPE_BETA1_CORE", "PHONE_SHAPE_BETA05_SPEECH", "PHONE_SHAPE_BETA1_SPEECH"):
            arm_templates, beta, mask_mode = _templates_for_arm(templates, arm)
            candidate, metadata = render_phone_shape(source, tokens, arm_templates, beta=beta, mask_mode=mask_mode, sample_rate=int(audio_cfg.get("sample_rate", 16000)), n_fft=int(audio_cfg.get("n_fft", 512)), win_length=int(audio_cfg.get("win_length", 512)), hop_length=int(audio_cfg.get("hop_length", 128)), edge_guard_s=float(audio_cfg.get("edge_guard_s", 0.010)), taper_s=float(audio_cfg.get("taper_s", 0.005)), max_gain_db=float(audio_cfg.get("max_template_db", 6.0)))
            rendered[arm] = (candidate, metadata, core_mask if mask_mode == "core" else speech_mask)
        eq = {"__global__": list(np.mean(np.stack([np.asarray(value) for value in templates.get("templates", {}).values()]), axis=0))} if templates.get("templates") else {}
        eq_candidate, eq_meta = render_phone_shape(source, tokens, eq, beta=1.0, mask_mode="speech", sample_rate=int(audio_cfg.get("sample_rate", 16000)), n_fft=int(audio_cfg.get("n_fft", 512)), win_length=int(audio_cfg.get("win_length", 512)), hop_length=int(audio_cfg.get("hop_length", 128)), edge_guard_s=float(audio_cfg.get("edge_guard_s", 0.010)), taper_s=float(audio_cfg.get("taper_s", 0.005)), max_gain_db=float(audio_cfg.get("max_template_db", 6.0))) if eq else (source.copy(), {"arm": "EQ_MATCH", "effective_edit": False}, speech_mask)
        rendered["EQ_MATCH"] = (eq_candidate, eq_meta, speech_mask)
        shuffled_candidate, shuffled_meta = render_phone_shape(source, tokens, shuffled, beta=1.0, mask_mode="speech", sample_rate=int(audio_cfg.get("sample_rate", 16000)), n_fft=int(audio_cfg.get("n_fft", 512)), win_length=int(audio_cfg.get("win_length", 512)), hop_length=int(audio_cfg.get("hop_length", 128)), edge_guard_s=float(audio_cfg.get("edge_guard_s", 0.010)), taper_s=float(audio_cfg.get("taper_s", 0.005)), max_gain_db=float(audio_cfg.get("max_template_db", 6.0)))
        rendered["SHUFFLED_LABEL"] = (shuffled_candidate, shuffled_meta, speech_mask)
        if donor is not None and tts.get("tokens"):
            oracle, oracle_meta = render_paired_shape(source, donor, tokens, tts["tokens"], beta=1.0, mask_mode="speech", sample_rate=int(audio_cfg.get("sample_rate", 16000)), n_fft=int(audio_cfg.get("n_fft", 512)), win_length=int(audio_cfg.get("win_length", 512)), hop_length=int(audio_cfg.get("hop_length", 128)), edge_guard_s=float(audio_cfg.get("edge_guard_s", 0.010)), taper_s=float(audio_cfg.get("taper_s", 0.005)), max_gain_db=float(audio_cfg.get("max_template_db", 6.0)))
            rendered["PAIRED_SHAPE_ORACLE"] = (oracle, oracle_meta, speech_mask)
        for arm in _construction_arms(config):
            if arm not in rendered:
                continue
            candidate, metadata, mask = rendered[arm]
            gain_control = None
            if arm not in {"N_ID", "N_RT"}:
                target_energy = float(np.sum((candidate[mask > 0].astype(np.float64) / 32768.0) ** 2)) if np.any(mask > 0) else 0.0
                gain_pcm, gain_meta = make_gain_control(source, mask, target_energy)
                gain_control = {"valid": bool(gain_meta.get("valid")), "metadata": gain_meta, "candidate_path": None}
            output_path = run_dir / "05_construction" / "wav" / arm / f"{pair['pair_id']}.wav"
            output_meta = write_pcm16(output_path, candidate, sample_rate=int(audio_cfg.get("sample_rate", 16000)))
            qc = validate_waveform(source, candidate, mask)
            row = {"pair_id": pair["pair_id"], "sample_id": pair["sample_id"], "source_group": pair["source_group"], "analysis_split": pair["analysis_split"], "arm": arm, "input_mode": "paired_tts_oracle" if arm == "PAIRED_SHAPE_ORACLE" else "natural_only", "clock_owner": "natural", "parent_natural_sha256": source_meta["container_sha256"], "output": output_meta, "construction": metadata, "waveform_qc": qc, "dose": mask_dose_audit(source, candidate, mask), "gain_control_available": gain_control is not None and bool(gain_control.get("valid"))}
            rows.append(row)
    _write_jsonl(run_dir / "05_construction" / "construction.jsonl", rows)
    selection = {"schema_version": 1, "status": "PENDING_ATLAS_REEXTRACTION", "candidate_arms": _construction_arms(config), "selection_split": "dev", "e_seen_locked": True, "selected_arm": None, "reason_codes": ["CONSTRUCTION_COMPLETED_BEFORE_CANDIDATE_REEXTRACTION"]}
    _write_json(run_dir / "05_construction" / "selection.json", selection)
    _status(run_dir, execution="RUNNING", stage="construct", stage_states={"audit": "COMPLETE", "atlas": "COMPLETE", "construct": "COMPLETE"}, manipulation="CONSTRUCTION_COMPLETE", engineering_pass=all(bool(row["waveform_qc"].get("pass")) for row in rows), reason_codes=[])
    return {"rows": rows, "templates": templates, "selection": selection}


def stage_mechanisms(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    registry = _load_registry(run_dir)
    _ensure_resources(config, run_dir, device="cpu", estimate_bytes=256 * 1024 * 1024)
    rows: list[dict[str, Any]] = []
    for model_cfg in _model_configs(config):
        feature_rows = _atlas_rows(run_dir, str(model_cfg["key"]))
        if not feature_rows:
            continue
        rows.append({"model_key": model_cfg["key"], "duration": {split: duration_analysis(feature_rows, split=split) for split in ("dev", "e_seen")}, "domain_auc": {split: domain_auc(feature_rows, split=split) for split in ("dev", "e_seen")}, "abx": {"natural": {split: abx_score([row for row in feature_rows if row.get("condition") == "natural"], split=split) for split in ("dev", "e_seen")}, "tts": {split: abx_score([row for row in feature_rows if row.get("condition") == "tts"], split=split) for split in ("dev", "e_seen")}}})
    historical = {"registered_inputs": [], "notes": "Only explicitly registered manifests are eligible; absent paths remain missing."}
    for key, path in {"trajectory": "runs/mfa_linear_trajectory_ablation_v2_support30_20260916/audio_manifest.json", "f0": "runs/tts_f0_swap_v3_audit2/audio_manifest.json", "learned_old": "runs/two_stage_hubert_aishell1_20260810/stage2_scale05_full_20260812/valid_enhanced_wav/enhanced_manifest.json"}.items():
        resolved = _resolve(path)
        historical["registered_inputs"].append({"key": key, "path": str(resolved), "exists": resolved.is_file(), "sha256": sha256_file(resolved) if resolved.is_file() else None})
    evidence = {"schema_version": 1, "protocol": PROTOCOL_ID, "scope": "smoke" if smoke else "registered_historical_assets", "lrs3": rows, "historical": historical, "interpretation": "phone identity, domain AUC, audio timing, and SyncNet are separate outcomes"}
    _write_json(run_dir / "04_mechanisms" / "mechanism_evidence.json", evidence)
    _status(run_dir, execution="RUNNING", stage="mechanisms", stage_states={"audit": "COMPLETE", "atlas": "COMPLETE", "mechanisms": "COMPLETE"}, measurement="MECHANISM_ATLAS_COMPLETE", reason_codes=[])
    return evidence


def stage_train(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    selection_path = run_dir / "05_construction" / "selection.json"
    selection = dict(read_json(selection_path)) if selection_path.is_file() else {}
    candidate_path = run_dir / "07_candidates" / "selection.json"
    candidate = dict(read_json(candidate_path)) if candidate_path.is_file() else {}
    if candidate:
        selected_arm = candidate.get("selected_arm")
        decision = {
            "status": "SKIPPED_NO_RULE_PASSES_DEV_GATE" if not selected_arm else "SKIPPED_CONDITIONAL_TRAINER_DATASET_PENDING",
            "reason_codes": ["NO_RULE_PASSES_DEV_GATE"] if not selected_arm else ["RULE_GATE_PASSED_BUT_TRAINER_DATASET_INTEGRATION_PENDING"],
            "selected_arm": selected_arm,
            "smoke": bool(smoke),
            "seeds": config.get("runtime", {}).get("seeds", [20260921, 20260922, 20260923]),
        }
    else:
        decision = {"status": "SKIPPED_CONDITIONAL_GATE", "reason_codes": ["CONSTRUCTED_CANDIDATE_NOT_YET_REEXTRACTED_AND_SELECTED"], "smoke": bool(smoke), "seeds": config.get("runtime", {}).get("seeds", [20260921, 20260922, 20260923])}
    _write_json(run_dir / "06_training" / "decision.json", decision)
    if candidate:
        # Keep the construction provenance while replacing its pre-extraction
        # placeholder with the actual frozen-encoder gate result.  Otherwise
        # selection_input.json falsely says that candidate scoring is pending
        # after the candidate stage has completed.
        selection_input = {
            **selection,
            "construction_status": selection.get("status"),
            "construction_reason_codes": selection.get("reason_codes", []),
            "status": candidate.get("status", selection.get("status")),
            "selection_split": candidate.get("selection_split", "dev"),
            "selected_arm": selected_arm,
            "e_seen_locked": bool(candidate.get("e_seen_locked", True)),
            "reason_codes": decision["reason_codes"],
            "candidate_gate": candidate,
        }
    else:
        selection_input = selection
    _write_json(run_dir / "06_training" / "selection_input.json", selection_input)
    train_state = str(decision["status"])
    _status(run_dir, execution="RUNNING", stage="train", stage_states={"train": train_state}, phone_gain="NOT_RUN", reason_codes=decision["reason_codes"])
    return decision


def stage_report(config: Mapping[str, Any], run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    status = dict(read_json(run_dir / "status.json")) if (run_dir / "status.json").is_file() else {}
    inventory = dict(read_json(run_dir / "00_inventory" / "registry.json")) if (run_dir / "00_inventory" / "registry.json").is_file() else {}
    atlas = dict(read_json(run_dir / "03_atlas" / "summary.json")) if (run_dir / "03_atlas" / "summary.json").is_file() else {}
    mechanisms = dict(read_json(run_dir / "04_mechanisms" / "mechanism_evidence.json")) if (run_dir / "04_mechanisms" / "mechanism_evidence.json").is_file() else {}
    candidates = dict(read_json(run_dir / "07_candidates" / "summary.json")) if (run_dir / "07_candidates" / "summary.json").is_file() else {}
    construction = _read_jsonl(run_dir / "05_construction" / "construction.jsonl")
    report_execution = "COMPLETE"
    lines = ["# TTS 音素可分性机制与自然时钟增强", "", f"- protocol: `{PROTOCOL_ID}`", f"- scope: `{'engineering_smoke' if smoke else 'registered_exploratory'}`", f"- assets: {inventory.get('counts', {})}", f"- status: `{report_execution}`", "", "## 当前边界", "", "音素身份 probe、来源域 AUC、音频时序/停顿、音质和 SyncNet 下游分别报告；本报告不把任一单项提升写成 TFG 因果解释。", "", "## Atlas", ""]
    for model_key, model in atlas.get("models", {}).items():
        lines.append(f"- `{model_key}`: rows={model.get('row_count')}, views={list(model.get('reference_labels', {}))}")
        for row in model.get("contrast_rows", []):
            effect = row.get("accuracy", {})
            lines.append(f"  - {row.get('split')}/{row.get('view')}: T-N={effect.get('estimate')}, CI=[{effect.get('ci_low')}, {effect.get('ci_high')}], groups={effect.get('n_groups')}")
    lines.extend(["", "## Auxiliary diagnostics", ""])
    for model_key, model in atlas.get("models", {}).items():
        dev_domain = model.get("domain", {}).get("dev", {})
        seen_domain = model.get("domain", {}).get("e_seen", {})
        dev_abx_n = model.get("abx_natural", {}).get("dev", {})
        dev_abx_t = model.get("abx_tts", {}).get("dev", {})
        seen_abx_n = model.get("abx_natural", {}).get("e_seen", {})
        seen_abx_t = model.get("abx_tts", {}).get("e_seen", {})
        lines.append(f"- `{model_key}` domain AUC N-vs-T: DEV={dev_domain.get('auc')}, E_SEEN={seen_domain.get('auc')}; ABX error DEV N/T={dev_abx_n.get('error')}/{dev_abx_t.get('error')}, E_SEEN N/T={seen_abx_n.get('error')}/{seen_abx_t.get('error')}")
    candidate_status = candidates.get("status", "NOT_RUN")
    candidate_selected = candidates.get("selected_arm")
    training_note = "conditional trainer dataset integration remains pending" if candidate_selected else "conditional branch skipped because no rule passed the DEV gate"
    lines.extend(["", "## Candidate gate", "", f"- status: `{candidate_status}`", f"- selected arm: `{candidate_selected}`", "- selection split: DEV only; E_SEEN remains locked", "", "## Construction", "", f"- generated rows: {len(construction)}", f"- effective edits: {sum(1 for row in construction if row.get('construction', {}).get('effective_edit'))}", "- E_SEEN selection: locked and not used for rule selection", "", "## Mechanism evidence", "", f"- registered historical inputs: {len(mechanisms.get('historical', {}).get('registered_inputs', []))}", f"- training: {training_note}", ""])
    report_path = run_dir / "08_report" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    phone_gain = "NOT_DETERMINED" if not atlas else ("NO_RULE_PASSES_DEV_GATE" if candidates and not candidates.get("selected_arm") else ("CANDIDATE_DEV_GATE_PASS" if candidates else "ATLAS_AVAILABLE"))
    decision = {"schema_version": 1, "protocol": PROTOCOL_ID, "status": "EXPLORATORY_REPORT_READY", "phone_gain": phone_gain, "natural_clock_gain": "NOT_DETERMINED", "timing": "NOT_ASSESSED" if not construction else "T0_ONLY", "downstream_tfg": "OUT_OF_SCOPE", "reason_codes": ["E_SEEN_LOCKED_AND_NO_DOWNSTREAM_TFG_CLAIM"]}
    _write_json(run_dir / "08_report" / "decision.json", decision)
    _write_json(run_dir / "08_report" / "confirmation_plan.json", {"status": "NOT_STARTED", "reason": "registered 40 evaluation groups are E_SEEN; new confirmation groups must be collected separately", "target_source_groups": 60})
    _status(run_dir, execution="COMPLETE", stage="report", stage_states={**status.get("stage_states", {}), "report": "COMPLETE"}, measurement=status.get("measurement", "NOT_RUN"), manipulation=status.get("manipulation", "NOT_RUN"), phone_gain=decision["phone_gain"], timing=decision["timing"], quality="NOT_ASSESSED", generalization="E_SEEN_LOCKED", downstream="OUT_OF_SCOPE", reason_codes=decision["reason_codes"])
    return decision


def _run_dir(run_id: str | None) -> Path:
    name = run_id or ("phone_separability_mechanism_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise InputInvalid("run-id must be a relative path below runs/")
    return (REPO_ROOT / "runs" / path).resolve()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/configs/phone_separability_mechanism_v1.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage", choices=("audit", "atlas", "mechanisms", "construct", "candidate", "train", "report", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_dir = _run_dir(args.run_id)
        config_path = _resolve(args.config)
        config = _load_config(config_path)
        if run_dir.exists() and not args.resume:
            raise InputInvalid(f"run directory exists; use a new run-id or --resume: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        current_protocol = _protocol(config, config_path)
        protocol_path = run_dir / "protocol.json"
        if args.resume:
            if not protocol_path.is_file():
                raise InputInvalid("--resume requires protocol.json")
            _verify_protocol(dict(read_json(protocol_path)), current_protocol)
        else:
            _write_json(protocol_path, current_protocol)
        if not (run_dir / "00_inventory" / "registry.json").is_file() or not args.resume:
            stage_audit(config, run_dir, smoke=bool(args.smoke))
        if args.stage == "audit":
            stage_report(config, run_dir, smoke=bool(args.smoke))
            print(f"audit complete: {run_dir}")
            return 0
        if args.stage in {"atlas", "mechanisms", "construct", "train", "report", "all"} and not (run_dir / "03_atlas" / "summary.json").is_file():
            stage_atlas(config, run_dir, smoke=bool(args.smoke))
        if args.stage in {"mechanisms", "construct", "candidate", "train", "report", "all"} and not (run_dir / "04_mechanisms" / "mechanism_evidence.json").is_file():
            stage_mechanisms(config, run_dir, smoke=bool(args.smoke))
        if args.stage in {"construct", "candidate", "train", "report", "all"} and not (run_dir / "05_construction" / "construction.jsonl").is_file():
            stage_construct(config, run_dir, smoke=bool(args.smoke))
        if args.stage in {"candidate", "train", "report", "all"} and not (run_dir / "07_candidates" / "summary.json").is_file():
            run_candidates(run_dir)
        if args.stage in {"train", "report", "all"} and not (run_dir / "06_training" / "decision.json").is_file():
            stage_train(config, run_dir, smoke=bool(args.smoke))
        if args.stage in {"report", "all"}:
            stage_report(config, run_dir, smoke=bool(args.smoke))
        print(f"run complete: {run_dir}")
        return 0
    except ResourceBusy as exc:
        try:
            _status(_run_dir(args.run_id), execution="RESOURCE_BUSY", stage=args.stage, engineering_pass=False, reason_codes=[str(exc)])
        except Exception:
            pass
        print(f"RESOURCE_BUSY: {exc}", file=sys.stderr)
        return 2
    except DependencyBlocked as exc:
        try:
            _status(_run_dir(args.run_id), execution="DEPENDENCY_BLOCKED", stage=args.stage, engineering_pass=False, reason_codes=[str(exc)])
        except Exception:
            pass
        print(f"DEPENDENCY_BLOCKED: {exc}", file=sys.stderr)
        return 2
    except (InputInvalid, InventoryError, OSError, ValueError, KeyError) as exc:
        try:
            _status(_run_dir(args.run_id), execution="IMPLEMENTATION_INVALID", stage=args.stage, engineering_pass=False, reason_codes=[str(exc)])
        except Exception:
            pass
        print(f"IMPLEMENTATION_INVALID: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "stage_atlas", "stage_audit", "stage_construct", "stage_mechanisms", "stage_report", "stage_train"]
