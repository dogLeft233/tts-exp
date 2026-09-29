"""Run the frozen LRS3 TTS phone-advantage and natural-rule experiment.

The command is intentionally staged:

* ``audit`` is CPU-only and never loads a model or starts MFA.
* ``a`` extracts frozen HuBERT/XLSR features and applies the preregistered
  TTS-versus-natural gate.
* ``b`` is authorized only after a passing Stage A and renders natural-input
  rule arms with a hard waveform contract.
* ``all`` runs the same sequence with no gate bypass.

Scientific success is never inferred from the process exit code alone; the
run artifacts contain engineering status, stage decisions, and reason codes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:  # direct ``python scripts/experiments/...py`` entry
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.experiments.lrs3_phone_rules_audio import (  # type: ignore[import-not-found]
        build_edit_mask,
        make_gain_control,
        render_rule_arm,
        validate_waveform,
    )
    from scripts.experiments.lrs3_phone_rules_metrics import (  # type: ignore[import-not-found]
        decide_stage_a,
        decide_stage_b,
        fit_reference_centroids,
        paired_bootstrap,
        pool_phone_tokens,
        score_pair,
    )
    from scripts.experiments.lrs3_phone_rules_worker import (  # type: ignore[import-not-found]
        WorkerError,
        extract_features,
        load_ssl_bundle,
        parse_textgrid,
        read_json,
        read_pcm16,
        realign_audio,
        sha256_file,
        token_signature,
        write_json_atomic,
        write_pcm16,
    )
else:  # pragma: no cover - imported by tests/package callers
    from .lrs3_phone_rules_audio import (
        build_edit_mask,
        make_gain_control,
        render_rule_arm,
        validate_waveform,
    )
    from .lrs3_phone_rules_metrics import (
        decide_stage_a,
        decide_stage_b,
        fit_reference_centroids,
        paired_bootstrap,
        pool_phone_tokens,
        score_pair,
    )
    from .lrs3_phone_rules_worker import (
        WorkerError,
        extract_features,
        load_ssl_bundle,
        parse_textgrid,
        read_json,
        read_pcm16,
        realign_audio,
        sha256_file,
        token_signature,
        write_json_atomic,
        write_pcm16,
    )


class ExperimentError(RuntimeError):
    """Base class for a contract-level runner failure."""


class InputInvalid(ExperimentError):
    pass


class DependencyBlocked(ExperimentError):
    pass


class ResourceBusy(ExperimentError):
    pass


SAMPLE_RATE = 16_000
MANIFEST_SHA256 = "dd109c8dfdbde9419f6cc8eaa11e7b9f78dc5d41ac3d437036c9122d14f97304"
TOKENS_SHA256 = "ad2a54b0aa6f01323acc7fa7650b38d3508ba7edf4282f87f98d55731ac38509"
CODE_FILES = (
    "scripts/experiments/lrs3_phone_rules.py",
    "scripts/experiments/lrs3_phone_rules_metrics.py",
    "scripts/experiments/lrs3_phone_rules_audio.py",
    "scripts/experiments/lrs3_phone_rules_worker.py",
    "scripts/experiments/check_lrs3_phone_rules.py",
    "scripts/configs/lrs3_phone_rules_v1.yaml",
)


def _resolve(value: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = _resolve(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputInvalid(f"config missing/unreadable: {config_path}") from exc
    if not isinstance(data, dict):
        raise InputInvalid("config must be a YAML mapping")
    return data


def _write_status(run_dir: Path, payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body.setdefault("schema_version", 1)
    body["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(run_dir / "status.json", body)


def _read_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "status.json"
    return dict(read_json(path)) if path.is_file() else {}


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "status_porcelain": run("status", "--porcelain"),
    }


def resource_snapshot() -> dict[str, Any]:
    """Collect read-only resource state before a model/MFA stage."""

    load = None
    try:
        load = [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        pass
    usage = shutil.disk_usage(REPO_ROOT)
    snapshot: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cpu_count": os.cpu_count(),
        "load_average": load,
        "disk": {
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "free_gib": float(usage.free / (1024**3)),
        },
        "gpu": {"available": False, "devices": [], "compute_apps": []},
    }
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
        )
        devices: list[dict[str, Any]] = []
        if query.returncode == 0:
            for line in query.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) >= 5:
                    devices.append({"index": fields[0], "name": fields[1], "memory_total_mib": fields[2], "memory_used_mib": fields[3], "utilization_gpu": fields[4]})
        compute_apps = []
        if apps.returncode == 0:
            for line in apps.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if fields and any(fields):
                    compute_apps.append(fields)
        snapshot["gpu"] = {"available": query.returncode == 0, "devices": devices, "compute_apps": compute_apps}
    return snapshot


def ensure_resources(config: Mapping[str, Any], *, device: str, run_dir: Path) -> dict[str, Any]:
    snapshot = resource_snapshot()
    write_json_atomic(run_dir / "resources.json", snapshot)
    runtime = config.get("runtime", {})
    if not bool(runtime.get("resource_check", True)):
        snapshot["decision"] = "CHECK_DISABLED_BY_CONFIG"
        return snapshot
    minimum = float(runtime.get("min_free_disk_gb", 4.0))
    if float(snapshot["disk"]["free_gib"]) < minimum:
        raise ResourceBusy(f"free disk {snapshot['disk']['free_gib']:.2f} GiB is below {minimum:.2f} GiB")
    if str(device).startswith("cuda") and snapshot["gpu"].get("compute_apps"):
        raise ResourceBusy("requested CUDA device has active compute processes")
    snapshot["decision"] = "ALLOW"
    write_json_atomic(run_dir / "resources.json", snapshot)
    return snapshot


def _compare_token_lists(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    for first, second in zip(left, right):
        if str(first.get("label", first.get("token", ""))).strip() != str(second.get("label", second.get("token", ""))).strip():
            return False
        if abs(float(first["start_s"]) - float(second["start_s"])) > 1e-4:
            return False
        if abs(float(first["end_s"]) - float(second["end_s"])) > 1e-4:
            return False
    return True


def _audit_side(
    record: Mapping[str, Any],
    side: str,
    side_tokens: Mapping[str, Any],
    *,
    sample_rate: int,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    audio_key = f"{side}_audio_path"
    hash_key = f"{side}_audio_sha256"
    audio_path = Path(str(record.get(audio_key, "")))
    if not audio_path.is_file():
        return {"path": str(audio_path), "exists": False}, [f"MISSING_{side.upper()}_AUDIO"]
    try:
        pcm, audio_meta = read_pcm16(audio_path, sample_rate=sample_rate)
    except WorkerError as exc:
        return {"path": str(audio_path), "exists": True, "error": str(exc)}, [f"INVALID_{side.upper()}_AUDIO"]
    expected_audio_hash = str(record.get(hash_key, ""))
    if expected_audio_hash and audio_meta["container_sha256"] != expected_audio_hash:
        reasons.append(f"{side.upper()}_AUDIO_HASH_MISMATCH")
    side_data = side_tokens.get(side) if isinstance(side_tokens, Mapping) else None
    if not isinstance(side_data, Mapping):
        reasons.append(f"MISSING_{side.upper()}_TOKENS")
        return {"audio": audio_meta, "path": str(audio_path), "sample_count": int(pcm.size)}, reasons
    textgrid_path = Path(str(side_data.get("textgrid", "")))
    if not textgrid_path.is_file():
        reasons.append(f"MISSING_{side.upper()}_TEXTGRID")
        return {"audio": audio_meta, "path": str(audio_path), "sample_count": int(pcm.size), "textgrid": str(textgrid_path)}, reasons
    expected_grid_hash = str(side_data.get("textgrid_sha256", ""))
    actual_grid_hash = sha256_file(textgrid_path)
    if expected_grid_hash and actual_grid_hash != expected_grid_hash:
        reasons.append(f"{side.upper()}_TEXTGRID_HASH_MISMATCH")
    try:
        parsed = parse_textgrid(textgrid_path)
    except WorkerError as exc:
        reasons.append(f"INVALID_{side.upper()}_TEXTGRID")
        return {"audio": audio_meta, "path": str(audio_path), "sample_count": int(pcm.size), "textgrid": str(textgrid_path), "error": str(exc)}, reasons
    expected_tokens = side_data.get("tokens", [])
    if not isinstance(expected_tokens, list) or not _compare_token_lists(expected_tokens, parsed):
        reasons.append(f"{side.upper()}_TEXTGRID_TOKEN_MISMATCH")
    previous_end = 0.0
    for token in parsed:
        begin = float(token["start_s"])
        if begin - previous_end > 1.0 / sample_rate + 1e-9:
            reasons.append(f"{side.upper()}_INTERNAL_ALIGNMENT_GAP")
        previous_end = float(token["end_s"])
    tail = float(pcm.size / sample_rate) - previous_end
    if tail < -1e-4 or tail > 0.020 + 1e-6:
        reasons.append(f"{side.upper()}_AUDIO_TEXTGRID_DURATION_MISMATCH")
    return {
        "audio": audio_meta,
        "path": str(audio_path),
        "sample_count": int(pcm.size),
        "textgrid": str(textgrid_path),
        "textgrid_sha256": actual_grid_hash,
        "textgrid_tokens": parsed,
        "textgrid_token_signature": token_signature(parsed),
        "tokens": expected_tokens,
        "token_signature": token_signature(expected_tokens),
        "tail_s": tail,
        "speech_token_count": sum(1 for token in expected_tokens if not bool(token.get("silence", False))),
    }, reasons


def audit_inputs(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Perform all CPU-only source, audio, and alignment checks."""

    source = config.get("source", {})
    manifest_path = _resolve(str(source.get("manifest", "")))
    tokens_path = _resolve(str(source.get("tokens", "")))
    expected_manifest_hash = str(source.get("manifest_sha256", MANIFEST_SHA256))
    expected_tokens_hash = str(source.get("tokens_sha256", TOKENS_SHA256))
    if not manifest_path.is_file() or not tokens_path.is_file():
        raise InputInvalid(f"fixed cohort inputs missing: {manifest_path}, {tokens_path}")
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise InputInvalid("cohort manifest SHA256 mismatch")
    if sha256_file(tokens_path) != expected_tokens_hash:
        raise InputInvalid("cohort tokens SHA256 mismatch")
    manifest = read_json(manifest_path)
    tokens_payload = read_json(tokens_path)
    records = manifest.get("records") if isinstance(manifest, Mapping) else None
    token_records = tokens_payload.get("records") if isinstance(tokens_payload, Mapping) else None
    if not isinstance(records, list) or not isinstance(token_records, Mapping):
        raise InputInvalid("fixed cohort schemas are not manifest.records list and tokens.records mapping")
    if str(tokens_payload.get("source_manifest_sha256", "")) != expected_manifest_hash:
        raise InputInvalid("tokens source manifest hash does not match fixed manifest")
    expected_count = int(source.get("expected_records", 240))
    if len(records) != expected_count:
        raise InputInvalid(f"expected {expected_count} records, got {len(records)}")
    seen: set[str] = set()
    train_groups: set[str] = set()
    eval_groups: set[str] = set()
    audited: list[dict[str, Any]] = []
    failures: list[str] = []
    for raw in records:
        sample_id = str(raw.get("sample_id", ""))
        source_group = str(raw.get("source_group", ""))
        split = str(raw.get("protocol_split", ""))
        if not sample_id or sample_id in seen or not source_group or split not in {"train", "evaluation"}:
            failures.append(f"INVALID_RECORD_ID_OR_SPLIT:{sample_id}")
            continue
        seen.add(sample_id)
        (train_groups if split == "train" else eval_groups).add(source_group)
        side_token_payload = token_records.get(sample_id)
        if not isinstance(side_token_payload, Mapping):
            failures.append(f"MISSING_TOKEN_RECORD:{sample_id}")
            continue
        side_result: dict[str, Any] = {}
        record_reasons: list[str] = []
        for side in ("natural", "tts"):
            result, reasons = _audit_side(raw, side, side_token_payload, sample_rate=int(source.get("sample_rate", SAMPLE_RATE)))
            side_result[side] = result
            record_reasons.extend(f"{sample_id}:{reason}" for reason in reasons)
        audited.append({
            "sample_id": sample_id,
            "source_group": source_group,
            "protocol_split": split,
            "transcript": raw.get("transcript"),
            "mfa_transcript": raw.get("mfa_transcript", raw.get("transcript")),
            "transcript_sha256": raw.get("transcript_sha256"),
            "tts_audio_origin": raw.get("tts_audio_origin"),
            "natural": side_result.get("natural"),
            "tts": side_result.get("tts"),
            "reasons": record_reasons,
        })
        failures.extend(record_reasons)
    expected_train = int(source.get("expected_train_records", 200))
    expected_eval = int(source.get("expected_eval_records", 40))
    expected_train_groups = int(source.get("expected_train_source_groups", 18))
    expected_eval_groups = int(source.get("expected_eval_source_groups", 40))
    if sum(1 for row in audited if row["protocol_split"] == "train") != expected_train:
        failures.append("TRAIN_RECORD_COUNT_MISMATCH")
    if sum(1 for row in audited if row["protocol_split"] == "evaluation") != expected_eval:
        failures.append("EVAL_RECORD_COUNT_MISMATCH")
    if len(train_groups) != expected_train_groups or len(eval_groups) != expected_eval_groups:
        failures.append("SOURCE_GROUP_COUNT_MISMATCH")
    if train_groups & eval_groups:
        failures.append("TRAIN_EVAL_SOURCE_GROUP_OVERLAP")
    artifact = {
        "schema_version": 1,
        "status": "INPUT_INVALID" if failures else "COMPLETE",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tokens_path": str(tokens_path),
        "tokens_sha256": sha256_file(tokens_path),
        "record_count": len(audited),
        "train_record_count": sum(1 for row in audited if row["protocol_split"] == "train"),
        "evaluation_record_count": sum(1 for row in audited if row["protocol_split"] == "evaluation"),
        "train_source_groups": sorted(train_groups),
        "evaluation_source_groups": sorted(eval_groups),
        "tts_audio_origins": sorted({str(row.get("tts_audio_origin")) for row in audited}),
        "failures": sorted(set(failures)),
        "records": audited,
    }
    run_dir.joinpath("00_audit").mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "00_audit" / "cohort.json", artifact)
    if failures:
        _write_status(run_dir, {"stage": "audit", "status": "INPUT_INVALID", "reason_codes": sorted(set(failures)), "allow_stage_a": False})
        raise InputInvalid(f"input audit failed with {len(set(failures))} reason codes")
    return artifact


def freeze_protocol(
    config: Mapping[str, Any],
    config_path: Path,
    run_dir: Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec_candidates = [
        REPO_ROOT / "basic-memory" / "Research" / "LRS3 音素优势验证与自然音频规则增强 Implementation Spec.md",
    ]
    spec_path = next((path for path in spec_candidates if path.is_file()), None)
    code_hashes = {path: sha256_file(REPO_ROOT / path) for path in CODE_FILES if (REPO_ROOT / path).is_file()}
    versions: dict[str, str | None] = {}
    for package in ("numpy", "torch", "transformers", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = {
        "schema_version": 1,
        "protocol_version": int(config.get("protocol_version", 1)),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "spec_path": str(spec_path.resolve()) if spec_path else None,
        "spec_sha256": sha256_file(spec_path) if spec_path else None,
        "code_sha256": code_hashes,
        "git": _git_snapshot(),
        "dependencies": versions,
        "random_seed": int(config.get("probe", {}).get("bootstrap_seed", 20260920)),
        "models": config.get("models", {}),
        "thresholds": config.get("probe", {}),
        "source_inputs": config.get("source", {}),
        "model_assets": "pending-local-resolution",
        "frame_time_convention": "frontend_receptive_field_centers",
        "pcm_convention": "mono_16khz_pcm16_little_endian",
    }
    if expected is not None:
        for key in ("protocol_version", "config_sha256", "spec_sha256", "code_sha256", "models", "thresholds", "source_inputs", "frame_time_convention", "pcm_convention"):
            if expected.get(key) != protocol.get(key):
                raise InputInvalid(f"--resume fingerprint mismatch at protocol.{key}; use a new run-id")
    write_json_atomic(run_dir / "protocol.json", protocol)
    return protocol


def _selected_records(cohort: Mapping[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    records = [dict(row) for row in cohort.get("records", [])]
    records.sort(key=lambda row: str(row["sample_id"]))
    if not smoke:
        return records
    selected: list[dict[str, Any]] = []
    for split in ("train", "evaluation"):
        selected.extend([row for row in records if row.get("protocol_split") == split][:1])
    return selected


def _tokens_for(row: Mapping[str, Any], condition: str) -> list[dict[str, Any]]:
    side = row.get(condition)
    if not isinstance(side, Mapping) or not isinstance(side.get("tokens"), list):
        raise InputInvalid(f"missing audited {condition} tokens for {row.get('sample_id')}")
    return [dict(token) for token in side["tokens"]]


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return sha256_file(path)


def _extract_model_records(
    config: Mapping[str, Any],
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    model_cfg: Mapping[str, Any],
    *,
    smoke: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    model_key = str(model_cfg["key"])
    target_layer = int(model_cfg["layer"])
    diagnostic_layers = [int(layer) for layer in config.get("models", {}).get("diagnostic_layers", {}).get(model_key, [])]
    layers = sorted({target_layer, *diagnostic_layers})
    device = str(config.get("models", {}).get("device", "cpu"))
    try:
        bundle = load_ssl_bundle(
            str(model_cfg["model_name"]),
            processor_name=str(model_cfg.get("processor_name") or model_cfg["model_name"]),
            revision=model_cfg.get("revision"),
            device=device,
        )
    except Exception as exc:
        raise DependencyBlocked(f"cannot load {model_key} local SSL assets: {exc}") from exc
    feature_dir = run_dir / "01_features" / model_key
    feature_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    token_lines: list[str] = []
    bundle_frontend = dict(bundle.get("frontend", {}))
    bundle_processor = dict(bundle.get("processor_fingerprint", {}))
    try:
        for row in rows:
            sample_id = str(row["sample_id"])
            results[sample_id] = {
                "sample_id": sample_id,
                "source_group": str(row["source_group"]),
                "protocol_split": str(row["protocol_split"]),
                "natural": [],
                "tts": [],
            }
            for condition in ("natural", "tts"):
                side = row[condition]
                audio_path = Path(str(side["path"]))
                pcm, audio_meta = read_pcm16(audio_path, sample_rate=int(config.get("source", {}).get("sample_rate", SAMPLE_RATE)))
                selected, frame_times, feature_meta = extract_features(
                    bundle["model"],
                    bundle["processor"],
                    pcm.astype(np.float32) / 32768.0,
                    int(config.get("source", {}).get("sample_rate", SAMPLE_RATE)),
                    layers,
                    device=device,
                    frontend=bundle["frontend"],
                )
                arrays = {f"layer_{layer}": selected[layer] for layer in layers}
                arrays["frame_times"] = frame_times
                feature_path = feature_dir / f"{sample_id}__{condition}.npz"
                feature_sha = _write_npz_atomic(feature_path, arrays)
                pooled = pool_phone_tokens(
                    selected[target_layer],
                    frame_times,
                    _tokens_for(row, condition),
                    sample_id=sample_id,
                    source_group=str(row["source_group"]),
                    condition=condition,
                )
                for token in pooled:
                    token["layer"] = target_layer
                results[sample_id][condition] = pooled
                token_lines.append(json.dumps({
                    "sample_id": sample_id,
                    "source_group": str(row["source_group"]),
                    "protocol_split": str(row["protocol_split"]),
                    "condition": condition,
                    "layer": target_layer,
                    "audio": audio_meta,
                    "feature": feature_meta,
                    "feature_path": str(feature_path),
                    "feature_sha256": feature_sha,
                    "tokens": pooled,
                }, ensure_ascii=False, allow_nan=False))
    except WorkerError as exc:
        raise DependencyBlocked(f"feature extraction blocked for {model_key}: {exc}") from exc
    finally:
        # Release the one-model-at-a-time worker before the next encoder.
        del bundle
    token_path = feature_dir / "token_records.jsonl"
    token_path.write_text("\n".join(token_lines) + ("\n" if token_lines else ""), encoding="utf-8")
    fingerprint = {
        "model_key": model_key,
        "model_name": model_cfg["model_name"],
        "processor_name": model_cfg.get("processor_name", model_cfg["model_name"]),
        "revision": model_cfg.get("revision"),
        "device": device,
        "target_layer": target_layer,
        "diagnostic_layers": diagnostic_layers,
        "bundle_frontend": bundle_frontend,
        "processor": bundle_processor,
        "record_count": len(rows),
        "smoke": smoke,
    }
    # ``bundle`` is deleted above; retain the fields needed in the protocol by
    # reading the per-run token metadata rather than holding model memory.
    fingerprint["token_records_sha256"] = sha256_file(token_path)
    write_json_atomic(feature_dir / "model.json", fingerprint)
    return results, fingerprint


def _score_common_pair(
    row: Mapping[str, Any],
    feature_by_model: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    model_keys: Sequence[str],
    probe_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    pair: dict[str, Any] = {
        "sample_id": sample_id,
        "source_group": str(row["source_group"]),
        "protocol_split": row.get("protocol_split"),
        "models": {},
        "eligible": True,
        "reason_codes": [],
    }
    for model_key in model_keys:
        model_records = feature_by_model[model_key][sample_id]
        reference = references[model_key]
        reference_labels = {str(label) for label in reference.get("labels", [])}
        natural = [token for token in model_records["natural"] if token.get("speech", True)]
        tts = [token for token in model_records["tts"] if token.get("speech", True)]
        natural_labels = {str(token.get("label", "")) for token in natural if token.get("valid") and str(token.get("label", "")) in reference_labels}
        tts_labels = {str(token.get("label", "")) for token in tts if token.get("valid") and str(token.get("label", "")) in reference_labels}
        common = sorted(natural_labels & tts_labels)
        restricted = dict(reference)
        restricted["labels"] = common
        restricted["centroids"] = {label: reference["centroids"][label] for label in common}
        min_labels = int(probe_cfg.get("min_labels_per_pair", 5))
        min_tokens = int(probe_cfg.get("min_tokens_per_pair_side", 10))
        min_coverage = float(probe_cfg.get("min_speech_coverage", 0.70))
        natural_score = score_pair(natural, restricted, condition="natural", pair_id=sample_id, min_labels=min_labels, min_tokens=min_tokens, min_coverage=min_coverage)
        tts_score = score_pair(tts, restricted, condition="tts", pair_id=sample_id, min_labels=min_labels, min_tokens=min_tokens, min_coverage=min_coverage)
        model_reasons = list(natural_score["reason_codes"]) + list(tts_score["reason_codes"])
        model_ok = bool(common) and natural_score["eligible"] and tts_score["eligible"]
        pair["models"][model_key] = {
            "common_labels": common,
            "natural": natural_score,
            "tts": tts_score,
            "eligible": model_ok,
            "reason_codes": sorted(set(model_reasons)),
        }
        if not model_ok:
            pair["eligible"] = False
            pair["reason_codes"].append(f"{model_key.upper()}_PAIR_NOT_ELIGIBLE")
    pair["reason_codes"] = sorted(set(pair["reason_codes"]))
    return pair


def _group_bootstrap_from_pairs(pairs: Sequence[Mapping[str, Any]], value_fn: Any, *, seed: int, draws: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        value = float(value_fn(pair))
        if math.isfinite(value):
            grouped[str(pair["source_group"])].append(value)
    group_means = {group: float(np.mean(values)) for group, values in grouped.items() if values}
    return paired_bootstrap(group_means, seed=seed, draws=draws)


def _stage_a_stats(
    pairs: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    eligible = [pair for pair in pairs if pair.get("eligible")]
    natural_values = [pair["models"][model_key]["natural"]["accuracy"] for pair in eligible]
    tts_values = [pair["models"][model_key]["tts"]["accuracy"] for pair in eligible]
    effects = _group_bootstrap_from_pairs(
        eligible,
        lambda pair: float(pair["models"][model_key]["tts"]["accuracy"] - pair["models"][model_key]["natural"]["accuracy"]),
        seed=seed,
        draws=draws,
    ) if eligible else {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_groups": 0, "seed": seed, "draws": draws}
    empty_effect = {
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "n_groups": 0,
        "seed": seed,
        "draws": draws,
    }
    return {
        "model_key": model_key,
        "n_pairs": len(eligible),
        "n_source_groups": len({str(pair["source_group"]) for pair in eligible}),
        "mean_natural": float(np.mean(natural_values)) if natural_values else None,
        "mean_tts": float(np.mean(tts_values)) if tts_values else None,
        "effect_tts_minus_natural": effects if eligible else empty_effect,
        "group_effects": [
            {"sample_id": pair["sample_id"], "source_group": pair["source_group"], "effect": float(pair["models"][model_key]["tts"]["accuracy"] - pair["models"][model_key]["natural"]["accuracy"])}
            for pair in eligible
        ],
    }


def run_stage_a(config: Mapping[str, Any], run_dir: Path, cohort: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    probe_cfg = config.get("probe", {})
    model_cfgs = [config.get("models", {}).get("primary", {}), config.get("models", {}).get("cross_encoder", {})]
    model_cfgs = [cfg for cfg in model_cfgs if cfg]
    rows = _selected_records(cohort, smoke=smoke)
    feature_by_model: dict[str, dict[str, Any]] = {}
    references: dict[str, dict[str, Any]] = {}
    model_fingerprints: dict[str, Any] = {}
    for model_cfg in model_cfgs:
        model_key = str(model_cfg["key"])
        extracted, fingerprint = _extract_model_records(config, run_dir, rows, model_cfg, smoke=smoke)
        feature_by_model[model_key] = extracted
        model_fingerprints[model_key] = fingerprint
        train_records = [
            token
            for sample_id, sample in extracted.items()
            for condition in ("natural", "tts")
            for token in sample[condition]
            if next(row for row in rows if str(row["sample_id"]) == sample_id)["protocol_split"] == "train"
        ]
        reference = fit_reference_centroids(
            train_records,
            min_tokens=int(probe_cfg.get("min_tokens_per_label", 20)),
            min_groups=int(probe_cfg.get("min_groups_per_label", 3)),
        )
        references[model_key] = reference
        write_json_atomic(run_dir / "02_reference" / f"{model_key}.json", reference)
    model_keys = [str(cfg["key"]) for cfg in model_cfgs]
    eval_rows = [row for row in rows if row.get("protocol_split") == "evaluation"]
    pairs = [
        _score_common_pair(row, feature_by_model, references, model_keys, probe_cfg)
        for row in eval_rows
    ]
    eligible_pairs = [pair for pair in pairs if pair.get("eligible")]
    min_groups = int(probe_cfg.get("min_eval_source_groups", 30))
    support_ok = (
        len({str(pair["source_group"]) for pair in eligible_pairs}) >= min_groups
        and all(len(references[key].get("labels", [])) >= 10 for key in model_keys)
        and not smoke
    )
    seed = int(probe_cfg.get("bootstrap_seed", 20260920))
    draws = int(probe_cfg.get("bootstrap_draws", 10_000))
    stats = {key: _stage_a_stats(eligible_pairs, model_key=key, seed=seed, draws=draws) for key in model_keys}
    primary_key = str(config.get("models", {}).get("primary", {}).get("key", "hubert"))
    cross_key = str(config.get("models", {}).get("cross_encoder", {}).get("key", "xlsr"))
    if smoke:
        decision = {"science_decision": "SMOKE_NO_SCIENTIFIC_DECISION", "reason_codes": ["SMOKE_RUN"], "engineering_pass": True}
    else:
        decision = decide_stage_a(
            stats.get(primary_key, {}).get("effect_tts_minus_natural"),
            stats.get(cross_key, {}).get("effect_tts_minus_natural"),
            support_ok=support_ok,
            threshold=float(probe_cfg.get("stage_a_min_effect", 0.020)),
        )
        decision["engineering_pass"] = True
    stage_dir = run_dir / "03_stage_a"
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / "per_pair.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n")
    write_json_atomic(stage_dir / "statistics.json", {
        "schema_version": 1,
        "models": stats,
        "eligible_pair_count": len(eligible_pairs),
        "eligible_source_group_count": len({str(pair["source_group"]) for pair in eligible_pairs}),
        "support_ok": support_ok,
        "smoke": smoke,
    })
    write_json_atomic(stage_dir / "decision.json", {
        "schema_version": 1,
        **decision,
        "model_fingerprints": model_fingerprints,
        "reference_labels": {key: references[key].get("labels", []) for key in model_keys},
        "reference_sha256": {key: sha256_file(run_dir / "02_reference" / f"{key}.json") for key in model_keys},
    })
    if decision["science_decision"] not in {"ADVANTAGE_SUPPORTED"}:
        skip = {
            "schema_version": 1,
            "status": "SKIPPED_GATE_NOT_PASSED",
            "parent_decision": decision["science_decision"],
            "reason_codes": ["STAGE_A_GATE_NOT_PASSED"],
        }
        write_json_atomic(run_dir / "05_stage_b" / "decision.json", skip)
    _write_status(run_dir, {
        "stage": "a",
        "status": "COMPLETE",
        "engineering_pass": True,
        "science_decision": decision["science_decision"],
        "reason_codes": decision.get("reason_codes", []),
        "allow_stage_b": decision["science_decision"] == "ADVANTAGE_SUPPORTED",
    })
    return {"decision": decision, "statistics": stats, "pairs": pairs, "feature_by_model": feature_by_model, "references": references}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise InputInvalid(f"artifact missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputInvalid(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise InputInvalid(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _load_token_feature_records(run_dir: Path, model_key: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    lines = _read_jsonl(run_dir / "01_features" / model_key / "token_records.jsonl")
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for line in lines:
        sample_id = str(line["sample_id"])
        condition = str(line["condition"])
        tokens = [dict(token) for token in line.get("tokens", [])]
        result[sample_id][condition] = tokens
    return result


def _copy_tokens_for_arm(tokens: Sequence[Mapping[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [{**dict(token), "condition": arm} for token in tokens]


def _edit_rate(original: Sequence[str], candidate: Sequence[str]) -> float:
    n = len(original)
    m = len(candidate)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + (original[i - 1] != candidate[j - 1]),
            )
    return float(table[n][m] / max(n, 1))


def _timing_qc_from_waveforms(
    config: Mapping[str, Any],
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    eligible_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the optional same-environment MFA timing check.

    A missing MFA executable or failed realignment is explicitly unverified;
    equal WAV lengths alone never make this pass.
    """

    timing_cfg = config.get("rules", {}).get("timing", {})
    if not bool(timing_cfg.get("run_mfa", True)):
        return {"status": "TIMING_UNVERIFIED", "timing_pass": False, "reason_codes": ["MFA_DISABLED_BY_CONFIG"]}
    executable = str(config.get("mfa", {}).get("executable", "mfa"))
    if shutil.which(executable) is None:
        return {"status": "TIMING_UNVERIFIED", "timing_pass": False, "reason_codes": ["MFA_EXECUTABLE_MISSING"]}
    row_by_id = {str(row["sample_id"]): row for row in rows}
    natural_records = []
    for pair in eligible_pairs:
        sample_id = str(pair["sample_id"])
        if sample_id not in row_by_id:
            continue
        row = row_by_id[sample_id]
        natural_records.append({
            "sample_id": sample_id,
            "mfa_transcript": row.get("mfa_transcript", row.get("transcript", "")),
            "audio_path": str(row["natural"]["path"]),
        })
    if not natural_records:
        return {"status": "TIMING_UNVERIFIED", "timing_pass": False, "reason_codes": ["NO_ELIGIBLE_TIMING_PAIRS"]}
    mfa_cfg = dict(config.get("mfa", {}))
    try:
        natural_run = realign_audio(
            natural_records,
            audio_key="audio_path",
            output_dir=run_dir / "06_timing" / "natural_mfa",
            mfa_config=mfa_cfg,
            repo_root=REPO_ROOT,
            execute=True,
        )
    except (KeyError, OSError, ValueError, WorkerError) as exc:
        return {"status": "TIMING_UNVERIFIED", "timing_pass": False, "reason_codes": ["NATURAL_MFA_FAILED"], "error": str(exc)}
    # The generated arm is laid out as a flat side-specific record map for
    # realign_audio; retain the original transcript and source group.
    enhanced_records: list[dict[str, Any]] = []
    for row in natural_records:
        sample_id = str(row["sample_id"])
        enhanced_records.append({
            "sample_id": sample_id,
            "mfa_transcript": row.get("mfa_transcript", row.get("transcript", "")),
            "audio_path": str(run_dir / "04_rules" / "wav" / "spectral_drc" / f"{sample_id}.wav"),
        })
    try:
        enhanced_run = realign_audio(
            enhanced_records,
            audio_key="audio_path",
            output_dir=run_dir / "06_timing" / "enhanced_mfa",
            mfa_config=mfa_cfg,
            repo_root=REPO_ROOT,
            execute=True,
        )
    except (KeyError, OSError, ValueError, WorkerError) as exc:
        return {"status": "TIMING_UNVERIFIED", "timing_pass": False, "reason_codes": ["ENHANCED_MFA_FAILED"], "error": str(exc), "natural_run": natural_run}
    boundary_errors: list[float] = []
    edit_rates: list[float] = []
    for row in natural_records:
        sample_id = str(row["sample_id"])
        natural_grid = Path(natural_run["aligned_dir"]) / f"{sample_id}.TextGrid"
        enhanced_grid = Path(enhanced_run["aligned_dir"]) / f"{sample_id}.TextGrid"
        if not natural_grid.is_file() or not enhanced_grid.is_file():
            return {"status": "TIMING_UNVERIFIED", "timing_pass": False, "reason_codes": ["MFA_TEXTGRID_MISSING"], "natural_run": natural_run, "enhanced_run": enhanced_run}
        try:
            natural_tokens = parse_textgrid(natural_grid)
            enhanced_tokens = parse_textgrid(enhanced_grid)
        except WorkerError as exc:
            return {"status": "TIMING_UNVERIFIED", "timing_pass": False, "reason_codes": ["MFA_TEXTGRID_INVALID"], "error": str(exc)}
        natural_labels = [str(token["label"]) for token in natural_tokens if bool(token["speech"])]
        enhanced_labels = [str(token["label"]) for token in enhanced_tokens if bool(token["speech"])]
        edit_rates.append(_edit_rate(natural_labels, enhanced_labels))
        # Same-label ordered matching is conservative; if MFA inserts/deletes a
        # phone, the mismatch remains visible through edit rate.
        for first, second in zip(natural_tokens, enhanced_tokens):
            if first["label"] == second["label"] and bool(first["speech"]) and bool(second["speech"]):
                boundary_errors.extend([
                    abs(float(first["start_s"]) - float(second["start_s"])) * 1000.0,
                    abs(float(first["end_s"]) - float(second["end_s"])) * 1000.0,
                ])
    median = float(np.median(boundary_errors)) if boundary_errors else float("inf")
    p95 = float(np.quantile(boundary_errors, 0.95)) if boundary_errors else float("inf")
    max_edit = max(edit_rates, default=float("inf"))
    passed = (
        max_edit <= float(timing_cfg.get("max_speech_edit_rate", 0.05))
        and median <= float(timing_cfg.get("median_boundary_abs_ms", 20.0))
        and p95 <= float(timing_cfg.get("p95_boundary_abs_ms", 40.0))
    )
    return {
        "status": "TIMING_PASS" if passed else "TIMING_NOT_ESTABLISHED",
        "timing_pass": passed,
        "reason_codes": [] if passed else ["TIMING_THRESHOLD_FAILED"],
        "max_speech_edit_rate": max_edit,
        "median_boundary_abs_ms": median,
        "p95_boundary_abs_ms": p95,
        "natural_run": natural_run,
        "enhanced_run": enhanced_run,
    }


def _stage_b_arm_stats(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    arms = ("natural", "tts", "spectral_drc", "spectral_only", "drc_only", "gain_control", "identity")
    scores: dict[str, list[float]] = defaultdict(list)
    group_rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair in pair_rows:
        model = pair.get("models", {}).get(model_key, {})
        values = model.get("arms", {})
        for arm in arms:
            if arm in values and math.isfinite(float(values[arm].get("accuracy", float("nan")))):
                value = float(values[arm]["accuracy"])
                scores[arm].append(value)
                group_rows[str(pair["source_group"])][arm].append(value)
    means = {arm: (float(np.mean(values)) if values else None) for arm, values in scores.items()}
    def effect(arm_a: str, arm_b: str) -> dict[str, Any]:
        grouped: dict[str, float] = {}
        for group, values in group_rows.items():
            if values.get(arm_a) and values.get(arm_b):
                grouped[group] = float(np.mean(values[arm_a]) - np.mean(values[arm_b]))
        if not grouped:
            return {"estimate": None, "ci_low": None, "ci_high": None, "n_groups": 0, "seed": seed, "draws": draws}
        return paired_bootstrap(grouped, seed=seed, draws=draws)
    half_grouped: dict[str, float] = {}
    for group, values in group_rows.items():
        if values.get("spectral_drc") and values.get("natural") and values.get("tts"):
            enhanced = float(np.mean(values["spectral_drc"]))
            natural = float(np.mean(values["natural"]))
            tts = float(np.mean(values["tts"]))
            half_grouped[group] = (enhanced - natural) - 0.5 * (tts - natural)
    half_effect = paired_bootstrap(half_grouped, seed=seed, draws=draws) if half_grouped else {
        "estimate": None, "ci_low": None, "ci_high": None, "n_groups": 0, "seed": seed, "draws": draws,
    }
    return {
        "model_key": model_key,
        "means": means,
        "effects": {
            "D_E": effect("spectral_drc", "natural"),
            "D_C": effect("spectral_drc", "gain_control"),
            "D_T": effect("spectral_drc", "tts"),
            "D_half": half_effect,
        },
        "n_pairs": len(pair_rows),
        "n_source_groups": len({str(pair["source_group"]) for pair in pair_rows}),
    }


def _finite_greater(value: Any, threshold: float) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > threshold
    except (TypeError, ValueError):
        return False


def _finite_at_least(value: Any, threshold: float) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= threshold
    except (TypeError, ValueError):
        return False


def _secondary_stage_b_decisions(stats: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    effects = stats.get(model_key, {}).get("effects", {})
    return {
        "half_tts_gap": bool(_finite_at_least(effects.get("D_half", {}).get("ci_low"), 0.0)),
        "not_worse_than_tts_1pp": bool(_finite_greater(effects.get("D_T", {}).get("ci_low"), -0.010)),
        "surpasses_tts": bool(_finite_greater(effects.get("D_T", {}).get("ci_low"), 0.0)),
    }


def run_stage_b(config: Mapping[str, Any], run_dir: Path, cohort: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    decision_path = run_dir / "03_stage_a" / "decision.json"
    if not decision_path.is_file():
        raise InputInvalid("Stage A decision is missing")
    stage_a_decision = dict(read_json(decision_path))
    if stage_a_decision.get("science_decision") != "ADVANTAGE_SUPPORTED":
        skip = {
            "schema_version": 1,
            "status": "SKIPPED_GATE_NOT_PASSED",
            "parent_decision": stage_a_decision.get("science_decision"),
            "reason_codes": ["STAGE_A_GATE_NOT_PASSED"],
        }
        write_json_atomic(run_dir / "05_stage_b" / "decision.json", skip)
        return {"decision": skip, "skipped": True}
    if smoke:
        raise InputInvalid("smoke runs cannot authorize Stage B")
    probe_cfg = config.get("probe", {})
    rules_cfg = config.get("rules", {})
    rows = _selected_records(cohort, smoke=False)
    row_by_id = {str(row["sample_id"]): row for row in rows}
    stage_a_pairs = _read_jsonl(run_dir / "03_stage_a" / "per_pair.jsonl")
    eligible_pairs = [pair for pair in stage_a_pairs if pair.get("eligible")]
    if not eligible_pairs:
        raise InputInvalid("Stage A passed without eligible evaluation pairs")
    model_cfgs = [config.get("models", {}).get("primary", {}), config.get("models", {}).get("cross_encoder", {})]
    model_cfgs = [cfg for cfg in model_cfgs if cfg]
    references = {str(cfg["key"]): dict(read_json(run_dir / "02_reference" / f"{cfg['key']}.json")) for cfg in model_cfgs}
    natural_feature_records = {str(cfg["key"]): _load_token_feature_records(run_dir, str(cfg["key"])) for cfg in model_cfgs}
    arms = ("spectral_drc", "spectral_only", "drc_only", "gain_control", "identity")
    rule_dir = run_dir / "04_rules" / "wav"
    rule_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_dir / "04_rules" / "artifacts.jsonl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pair_arm_audio: dict[str, dict[str, dict[str, Any]]] = {}
    with artifact_path.open("w", encoding="utf-8") as artifact_handle:
        for pair in eligible_pairs:
            sample_id = str(pair["sample_id"])
            row = row_by_id[sample_id]
            natural_path = Path(str(row["natural"]["path"]))
            source_pcm, source_meta = read_pcm16(natural_path, sample_rate=int(rules_cfg.get("sample_rate", SAMPLE_RATE)))
            natural_tokens = _tokens_for(row, "natural")
            mask = build_edit_mask(
                source_pcm.size,
                natural_tokens,
                sample_rate=int(rules_cfg.get("sample_rate", SAMPLE_RATE)),
                edge_guard_s=float(rules_cfg.get("edge_guard_s", 0.010)),
                taper_s=float(rules_cfg.get("taper_s", 0.005)),
            )
            pair_arm_audio[sample_id] = {}
            rendered: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
            for arm in ("spectral_drc", "spectral_only", "drc_only", "identity"):
                rendered[arm] = render_rule_arm(
                    source_pcm,
                    natural_tokens,
                    arm,
                    sample_rate=int(rules_cfg.get("sample_rate", SAMPLE_RATE)),
                    edge_guard_s=float(rules_cfg.get("edge_guard_s", 0.010)),
                    taper_s=float(rules_cfg.get("taper_s", 0.005)),
                    n_fft=int(rules_cfg.get("stft", {}).get("n_fft", 512)),
                    win_length=int(rules_cfg.get("stft", {}).get("win_length", 512)),
                    hop_length=int(rules_cfg.get("stft", {}).get("hop_length", 128)),
                )
            main_pcm, _main_meta = rendered["spectral_drc"]
            target_energy = float(np.sum((main_pcm.astype(np.float64)[mask > 0.0] / 32768.0) ** 2))
            gain_pcm, gain_meta = make_gain_control(source_pcm, mask, target_energy)
            rendered["gain_control"] = (gain_pcm, gain_meta)
            for arm in arms:
                pcm, meta = rendered[arm]
                path = rule_dir / arm / f"{sample_id}.wav"
                write_meta = write_pcm16(path, pcm, sample_rate=int(rules_cfg.get("sample_rate", SAMPLE_RATE)))
                qc = validate_waveform(source_pcm, pcm, mask)
                if arm == "identity" and not np.array_equal(source_pcm, pcm):
                    raise ExperimentError(f"identity arm changed PCM: {sample_id}")
                record = {
                    "sample_id": sample_id,
                    "source_group": row["source_group"],
                    "arm": arm,
                    "parent_natural_sha256": source_meta["container_sha256"],
                    "output": write_meta,
                    "rule": meta,
                    "waveform_qc": qc,
                }
                artifact_handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                pair_arm_audio[sample_id][arm] = record
    # Re-extract the target layer for generated arms one encoder at a time.
    generated_features: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(dict))
    for model_cfg in model_cfgs:
        model_key = str(model_cfg["key"])
        target_layer = int(model_cfg["layer"])
        device = str(config.get("models", {}).get("device", "cpu"))
        try:
            bundle = load_ssl_bundle(
                str(model_cfg["model_name"]),
                processor_name=str(model_cfg.get("processor_name") or model_cfg["model_name"]),
                revision=model_cfg.get("revision"),
                device=device,
            )
        except Exception as exc:
            raise DependencyBlocked(f"cannot reload {model_key} for Stage B: {exc}") from exc
        try:
            for pair in eligible_pairs:
                sample_id = str(pair["sample_id"])
                row = row_by_id[sample_id]
                natural_tokens = _tokens_for(row, "natural")
                for arm in arms:
                    if arm == "identity":
                        generated_features[model_key][sample_id][arm] = _copy_tokens_for_arm(natural_feature_records[model_key][sample_id]["natural"], arm)
                        continue
                    audio_path = Path(str(pair_arm_audio[sample_id][arm]["output"]["path"]))
                    pcm, _meta = read_pcm16(audio_path, sample_rate=int(rules_cfg.get("sample_rate", SAMPLE_RATE)))
                    selected, frame_times, _feature_meta = extract_features(
                        bundle["model"], bundle["processor"], pcm.astype(np.float32) / 32768.0,
                        int(rules_cfg.get("sample_rate", SAMPLE_RATE)), [target_layer],
                        device=device, frontend=bundle["frontend"],
                    )
                    generated_features[model_key][sample_id][arm] = pool_phone_tokens(
                        selected[target_layer], frame_times, natural_tokens,
                        sample_id=sample_id, source_group=str(row["source_group"]), condition=arm,
                    )
        finally:
            del bundle
    b_pairs: list[dict[str, Any]] = []
    for pair in eligible_pairs:
        sample_id = str(pair["sample_id"])
        output_pair = dict(pair)
        output_pair["models"] = {}
        for model_cfg in model_cfgs:
            model_key = str(model_cfg["key"])
            ref = references[model_key]
            model_a = pair["models"][model_key]
            common = list(model_a["common_labels"])
            restricted = dict(ref)
            restricted["labels"] = common
            restricted["centroids"] = {label: ref["centroids"][label] for label in common}
            scores: dict[str, Any] = {
                "natural": model_a["natural"],
                "tts": model_a["tts"],
            }
            for arm in arms:
                scores[arm] = score_pair(
                    generated_features[model_key][sample_id][arm],
                    restricted,
                    condition=arm,
                    pair_id=sample_id,
                    min_labels=int(probe_cfg.get("min_labels_per_pair", 5)),
                    min_tokens=int(probe_cfg.get("min_tokens_per_pair_side", 10)),
                    min_coverage=float(probe_cfg.get("min_speech_coverage", 0.70)),
                )
            output_pair["models"][model_key] = {"common_labels": common, "arms": scores}
        b_pairs.append(output_pair)
    timing = _timing_qc_from_waveforms(config, run_dir, rows, eligible_pairs)
    seed = int(probe_cfg.get("bootstrap_seed", 20260920))
    draws = int(probe_cfg.get("bootstrap_draws", 10_000))
    b_stats = {str(cfg["key"]): _stage_b_arm_stats(b_pairs, model_key=str(cfg["key"]), seed=seed, draws=draws) for cfg in model_cfgs}
    primary_key = str(config.get("models", {}).get("primary", {}).get("key", "hubert"))
    cross_key = str(config.get("models", {}).get("cross_encoder", {}).get("key", "xlsr"))
    decision = decide_stage_b(
        b_stats.get(primary_key, {}).get("effects", {}).get("D_E"),
        b_stats.get(primary_key, {}).get("effects", {}).get("D_C"),
        b_stats.get(cross_key, {}).get("effects", {}).get("D_E"),
        timing,
        min_effect=float(probe_cfg.get("stage_b_min_effect", 0.010)),
    )
    decision["engineering_pass"] = all(
        bool(record["waveform_qc"].get("pass"))
        for records in pair_arm_audio.values()
        for record in records.values()
    )
    decision["secondary"] = _secondary_stage_b_decisions(b_stats, primary_key)
    stage_dir = run_dir / "05_stage_b"
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / "per_pair.jsonl").open("w", encoding="utf-8") as handle:
        for pair in b_pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n")
    write_json_atomic(run_dir / "04_rules" / "summary.json", {"artifact_sha256": sha256_file(artifact_path), "pair_count": len(b_pairs), "arms": list(arms)})
    write_json_atomic(run_dir / "06_timing" / "timing.json", timing)
    write_json_atomic(stage_dir / "statistics.json", {"models": b_stats, "timing": timing, "pair_count": len(b_pairs)})
    write_json_atomic(stage_dir / "decision.json", decision)
    _write_status(run_dir, {
        "stage": "b",
        "status": "COMPLETE",
        "engineering_pass": decision["engineering_pass"],
        "science_decision": decision["science_decision"],
        "reason_codes": decision.get("reason_codes", []),
        "allow_stage_b": True,
    })
    return {"decision": decision, "statistics": b_stats, "pairs": b_pairs, "timing": timing}


def write_report(run_dir: Path) -> Path:
    status = _read_status(run_dir)
    stage_a = dict(read_json(run_dir / "03_stage_a" / "decision.json")) if (run_dir / "03_stage_a" / "decision.json").is_file() else {}
    stage_b = dict(read_json(run_dir / "05_stage_b" / "decision.json")) if (run_dir / "05_stage_b" / "decision.json").is_file() else {}
    audit = dict(read_json(run_dir / "00_audit" / "cohort.json")) if (run_dir / "00_audit" / "cohort.json").is_file() else {}
    lines = [
        "# LRS3 音素优势与自然音频规则增强",
        "",
        "## 工程验收状态／A结论／B是否运行及结论",
        "",
        f"- 工程状态：`{status.get('status', 'UNKNOWN')}`；engineering_pass=`{status.get('engineering_pass')}`",
        f"- Stage A：`{stage_a.get('science_decision', 'NOT_RUN')}`；reason_codes=`{stage_a.get('reason_codes', [])}`",
        f"- Stage B：`{stage_b.get('science_decision', stage_b.get('status', 'NOT_RUN'))}`；reason_codes=`{stage_b.get('reason_codes', [])}`",
        "",
        "## 固定输入",
        "",
        f"- audit records={audit.get('record_count')}，train={audit.get('train_record_count')}，evaluation={audit.get('evaluation_record_count')}",
        f"- evaluation source groups={len(audit.get('evaluation_source_groups', []))}",
        f"- TTS origins={audit.get('tts_audio_origins', [])}",
    ]
    if (run_dir / "03_stage_a" / "statistics.json").is_file():
        stats = dict(read_json(run_dir / "03_stage_a" / "statistics.json"))
        lines.extend(["", "## Stage A统计", ""])
        for model_key, model_stats in stats.get("models", {}).items():
            effect = model_stats.get("effect_tts_minus_natural", {})
            lines.append(
                f"- {model_key}: pairs={model_stats.get('n_pairs')}, groups={model_stats.get('n_source_groups')}, "
                f"A(N)={model_stats.get('mean_natural')}, A(T)={model_stats.get('mean_tts')}, "
                f"Δ={effect.get('estimate')}, CI=[{effect.get('ci_low')}, {effect.get('ci_high')}]"
            )
    if (run_dir / "05_stage_b" / "statistics.json").is_file():
        stats = dict(read_json(run_dir / "05_stage_b" / "statistics.json"))
        lines.extend(["", "## Stage B统计", ""])
        for model_key, model_stats in stats.get("models", {}).items():
            lines.append(f"- {model_key}: {json.dumps(model_stats.get('effects', {}), ensure_ascii=False)}")
        lines.append(f"- timing: {json.dumps(stats.get('timing', {}), ensure_ascii=False)}")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "本实验的音素可分性是冻结 SSL 表征、MFA 标签、共享最近中心探针下的操作性指标；不等同于人耳可懂度、ASR 或 Sync-C。",
        "",
    ])
    report = run_dir / "07_report" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _run_dir_from_id(run_id: str | None) -> Path:
    if not run_id:
        run_id = "lrs3_phone_rules_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(run_id)
    if not path.is_absolute():
        path = REPO_ROOT / "runs" / path
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/configs/lrs3_phone_rules_v1.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage", choices=("audit", "a", "b", "all"), default="all")
    parser.add_argument("--smoke", action="store_true", help="two-record engineering smoke; never a scientific pass")
    parser.add_argument("--resume", action="store_true", help="resume an existing run with the same frozen inputs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = _run_dir_from_id(args.run_id)
    config_path = _resolve(args.config)
    try:
        config = load_config(config_path)
        if run_dir.exists() and not args.resume:
            raise InputInvalid(f"run directory already exists; use a new run-id or --resume: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        existing_protocol = None
        if args.resume:
            protocol_path = run_dir / "protocol.json"
            if not protocol_path.is_file():
                raise InputInvalid("--resume requires an existing protocol.json")
            existing_protocol = dict(read_json(protocol_path))
        freeze_protocol(config, config_path, run_dir, expected=existing_protocol)
        if not (run_dir / "00_audit" / "cohort.json").is_file() or not args.resume:
            cohort = audit_inputs(config, run_dir)
        else:
            cohort = dict(read_json(run_dir / "00_audit" / "cohort.json"))
            if cohort.get("status") != "COMPLETE":
                raise InputInvalid("cannot resume from an incomplete audit")
        if args.stage == "audit":
            _write_status(run_dir, {"stage": "audit", "status": "COMPLETE", "engineering_pass": True, "allow_stage_a": True})
            write_report(run_dir)
            print(f"audit complete: {run_dir}")
            return 0
        ensure_resources(config, device=str(config.get("models", {}).get("device", "cpu")), run_dir=run_dir)
        if args.stage in {"a", "all"}:
            run_stage_a(config, run_dir, cohort, smoke=bool(args.smoke))
        if args.stage in {"b", "all"}:
            stage_a_decision = dict(read_json(run_dir / "03_stage_a" / "decision.json")) if (run_dir / "03_stage_a" / "decision.json").is_file() else {}
            if stage_a_decision.get("science_decision") == "ADVANTAGE_SUPPORTED":
                run_stage_b(config, run_dir, cohort, smoke=bool(args.smoke))
            else:
                skip = {
                    "schema_version": 1,
                    "status": "SKIPPED_GATE_NOT_PASSED",
                    "parent_decision": stage_a_decision.get("science_decision", "MISSING_STAGE_A"),
                    "reason_codes": ["STAGE_A_GATE_NOT_PASSED"],
                }
                write_json_atomic(run_dir / "05_stage_b" / "decision.json", skip)
        report = write_report(run_dir)
        final_status = _read_status(run_dir)
        print(f"run complete: {run_dir}")
        print(f"report: {report}")
        print(f"science_decision: {final_status.get('science_decision', 'UNKNOWN')}")
        return 0
    except InputInvalid as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_status(run_dir, {"stage": args.stage, "status": "INPUT_INVALID", "engineering_pass": False, "reason_codes": [str(exc)]})
        print(f"INPUT_INVALID: {exc}", file=sys.stderr)
        return 2
    except (DependencyBlocked, ResourceBusy) as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_status(run_dir, {"stage": args.stage, "status": "DEPENDENCY_BLOCKED", "engineering_pass": False, "reason_codes": [str(exc)]})
        print(f"DEPENDENCY_BLOCKED: {exc}", file=sys.stderr)
        return 2
    except (WorkerError, ExperimentError, OSError, ValueError) as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_status(run_dir, {"stage": args.stage, "status": "IMPLEMENTATION_INVALID", "engineering_pass": False, "reason_codes": [str(exc)]})
        print(f"IMPLEMENTATION_INVALID: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit_inputs",
    "ensure_resources",
    "freeze_protocol",
    "main",
    "resource_snapshot",
    "run_stage_a",
    "run_stage_b",
    "write_report",
]
