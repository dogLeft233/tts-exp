"""Explicit, hash-bound asset inventory for phone separability experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from scripts.experiments.lrs3_phone_rules_worker import (
    WorkerError,
    parse_textgrid,
    read_json,
    read_pcm16,
    sha256_file,
)


class InventoryError(RuntimeError):
    """Raised when a frozen asset binding is malformed."""


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(value: str | Path, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _asset_id(dataset: str, pair_id: str, arm: str, pcm_sha256: str | None) -> str:
    # The readable prefix helps audits; the digest prevents short sample IDs
    # from becoming accidental identities when a run contains multiple arms.
    digest = _json_sha256([dataset, pair_id, arm, pcm_sha256])[:16]
    return f"{dataset}:{pair_id}:{arm}:{digest}"


def freeze_splits(
    source_groups: Sequence[str],
    *,
    fit_groups: int = 12,
    dev_groups: int = 6,
    salt: str = "psm-v1-split",
) -> dict[str, Any]:
    """Freeze group splits by hash, independent of manifest ordering.

    The registered LRS3 cohort has exactly 18 training groups, for which the
    protocol is 12 FIT and 6 DEV.  Smaller synthetic fixtures receive a
    deterministic proportional fallback while never creating an empty side.
    """

    unique = sorted({str(group) for group in source_groups if str(group)}, key=lambda group: hashlib.sha256(f"{salt}|{group}".encode()).hexdigest())
    if not unique:
        return {"salt": salt, "fit": [], "dev": [], "groups": {}}
    if len(unique) >= fit_groups + dev_groups:
        n_fit, n_dev = int(fit_groups), int(dev_groups)
    else:
        n_fit = min(max(1, int(round(len(unique) * 2 / 3))), max(1, len(unique) - 1))
        n_dev = len(unique) - n_fit
    assignments = {group: ("fit" if index < n_fit else "dev") for index, group in enumerate(unique)}
    return {
        "salt": salt,
        "fit": [group for group in unique if assignments[group] == "fit"],
        "dev": [group for group in unique if assignments[group] == "dev"],
        "groups": assignments,
        "algorithm": "sha256(salt|source_group), lexical digest order",
    }


def _union_find(values: Iterable[str]) -> tuple[dict[str, str], Any]:
    parent = {value: value for value in values}

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(left: str, right: str) -> None:
        lroot, rroot = find(left), find(right)
        if lroot != rroot:
            parent[max(lroot, rroot)] = min(lroot, rroot)

    return parent, (find, union)


def build_overlap_graph(
    assets: Sequence[Mapping[str, Any]],
    *,
    source_group_key: str = "source_group",
) -> dict[str, Any]:
    """Build connected components over explicit provenance aliases.

    A source group always connects its paired natural/TTS assets.  Exact
    video/audio hashes additionally connect aliases across historical runs.
    Transcript text is deliberately not an edge: same text is a generality
    risk, not proof of speaker identity.
    """

    node_ids = [str(row["asset_id"]) for row in assets]
    parent, (find, union) = _union_find(node_ids)
    by_alias: dict[str, list[str]] = defaultdict(list)
    for row in assets:
        asset_id = str(row["asset_id"])
        group = str(row.get(source_group_key, ""))
        if group:
            by_alias[f"group:{group}"].append(asset_id)
        for key in ("pcm_sha256", "container_sha256", "video_sha256"):
            value = row.get(key)
            if value:
                by_alias[f"{key}:{value}"].append(asset_id)
    for members in by_alias.values():
        for member in members[1:]:
            union(members[0], member)
    components: dict[str, list[str]] = defaultdict(list)
    for node in node_ids:
        components[find(node)].append(node)
    component_rows = {
        root: sorted(members)
        for root, members in sorted(components.items())
    }
    return {
        "nodes": sorted(node_ids),
        "aliases": {key: sorted(set(values)) for key, values in sorted(by_alias.items())},
        "components": component_rows,
        "component_count": len(component_rows),
    }


def _resource_snapshot(repo_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(repo_root)
    snapshot: dict[str, Any] = {
        "free_gib": usage.free / (1024**3),
        "total_gib": usage.total / (1024**3),
        "cpu_count": os.cpu_count(),
        "gpu": {"available": False, "devices": [], "compute_apps": []},
    }
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
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
        devices = []
        for line in query.stdout.splitlines() if query.returncode == 0 else []:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 6:
                devices.append({"index": fields[0], "name": fields[1], "memory_total_mib": fields[2], "memory_used_mib": fields[3], "memory_free_mib": fields[4], "utilization_gpu": fields[5]})
        compute_apps = [line.strip() for line in apps.stdout.splitlines() if line.strip()] if apps.returncode == 0 else []
        snapshot["gpu"] = {"available": query.returncode == 0, "devices": devices, "compute_apps": compute_apps}
    return snapshot


def check_resources(
    repo_root: Path,
    *,
    device: str = "cpu",
    min_free_disk_gib: float = 4.0,
    min_free_gpu_gib: float = 8.0,
    min_available_ram_gib: float = 8.0,
    stage_estimated_bytes: int = 0,
) -> dict[str, Any]:
    """Read resources and fail safely when another compute task is active."""

    snapshot = _resource_snapshot(repo_root)
    reasons: list[str] = []
    required_disk = max(float(min_free_disk_gib), float(stage_estimated_bytes) * 1.2 / (1024**3))
    if float(snapshot["free_gib"]) < required_disk:
        reasons.append("DISK_BELOW_MINIMUM")
    if str(device).startswith("cuda"):
        devices = snapshot.get("gpu", {}).get("devices", [])
        apps = snapshot.get("gpu", {}).get("compute_apps", [])
        if apps:
            reasons.append("GPU_COMPUTE_BUSY")
        matching = [row for row in devices if str(row.get("index", "")) == str(device).split(":")[-1]]
        if matching:
            try:
                free_mib = float(matching[0]["memory_free_mib"])
                if free_mib < min_free_gpu_gib * 1024:
                    reasons.append("GPU_FREE_MEMORY_BELOW_MINIMUM")
            except (KeyError, TypeError, ValueError):
                reasons.append("GPU_MEMORY_UNREADABLE")
        else:
            reasons.append("GPU_DEVICE_UNAVAILABLE")
    snapshot["decision"] = "RESOURCE_BUSY" if reasons else "ALLOW"
    snapshot["reason_codes"] = reasons
    snapshot["required_free_disk_gib"] = required_disk
    # RAM is recorded even when psutil is unavailable; the runner can use the
    # explicit gate only when the host exposes /proc/meminfo.
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        available_kib = next(float(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemAvailable:"))
        snapshot["ram_available_gib"] = available_kib / (1024**2)
        if snapshot["ram_available_gib"] < min_available_ram_gib:
            snapshot["reason_codes"].append("RAM_BELOW_MINIMUM")
            snapshot["decision"] = "RESOURCE_BUSY"
    except (OSError, StopIteration, ValueError):
        snapshot["ram_available_gib"] = None
    return snapshot


def _side_asset(
    row: Mapping[str, Any],
    side: str,
    side_payload: Mapping[str, Any],
    *,
    repo_root: Path,
    analysis_split: str,
) -> dict[str, Any]:
    arm = "N_RAW" if side == "natural" else "T_RAW"
    path_key = f"{side}_audio_path"
    expected_hash_key = f"{side}_audio_sha256"
    audio_path = _resolve(str(row.get(path_key, "")), repo_root)
    reasons: list[str] = []
    container_hash = None
    pcm_hash = None
    sample_rate = None
    sample_count = None
    if not audio_path.is_file():
        reasons.append("ASSET_MISSING")
    else:
        try:
            pcm, meta = read_pcm16(audio_path)
            container_hash = str(meta["container_sha256"])
            pcm_hash = str(meta["pcm_sha256"])
            sample_rate = int(meta["sample_rate"])
            sample_count = int(meta["sample_count"])
            if str(row.get(expected_hash_key, "")) and container_hash != str(row[expected_hash_key]):
                reasons.append("CONTAINER_HASH_MISMATCH")
        except (WorkerError, OSError, ValueError) as exc:
            reasons.append(f"AUDIO_INVALID:{type(exc).__name__}")
    grid_path = _resolve(str(side_payload.get("textgrid", "")), repo_root)
    grid_hash = sha256_file(grid_path) if grid_path.is_file() else None
    if not grid_path.is_file():
        reasons.append("TEXTGRID_MISSING")
    elif side_payload.get("textgrid_sha256") and grid_hash != str(side_payload["textgrid_sha256"]):
        reasons.append("TEXTGRID_HASH_MISMATCH")
    parsed = None
    if grid_path.is_file():
        try:
            parsed = parse_textgrid(grid_path)
            expected_tokens = side_payload.get("tokens", [])
            if isinstance(expected_tokens, list) and len(parsed) != len(expected_tokens):
                reasons.append("TEXTGRID_TOKEN_COUNT_MISMATCH")
        except (WorkerError, ValueError):
            reasons.append("TEXTGRID_INVALID")
    expected_tokens = side_payload.get("tokens", [])
    if not isinstance(expected_tokens, list):
        expected_tokens = []
        reasons.append("TOKENS_MISSING")
    status = "OK" if not reasons else ("ASSET_MISSING" if "ASSET_MISSING" in reasons else "AUDIT_FAILED")
    return {
        "asset_id": _asset_id(str(row.get("dataset", "lrs3")), str(row["sample_id"]), arm, pcm_hash or str(row.get(expected_hash_key, "")) or None),
        "dataset": str(row.get("dataset", "lrs3")),
        "language": str(row.get("language", "")),
        "corpus_key": "LRS3",
        "sample_id": str(row["sample_id"]),
        "paired_key": str(row["sample_id"]),
        "occurrence_namespace": f"{row['sample_id']}:{side}",
        "source_group": str(row.get("source_group", "")),
        "speaker_id": None,
        "speaker_verified": False,
        "parent_asset_ids": [],
        "arm": arm,
        "condition": side,
        "input_mode": "natural_only" if side == "natural" else "paired_tts_oracle",
        "clock_owner": side,
        "speech_identity": side,
        "audio_path": str(audio_path),
        "container_sha256": container_hash,
        "expected_container_sha256": row.get(expected_hash_key),
        "pcm_sha256": pcm_hash,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "transcript": str(row.get("mfa_transcript", row.get("transcript", ""))),
        "transcript_sha256": row.get("mfa_transcript_sha256", row.get("transcript_sha256")),
        "textgrid_path": str(grid_path),
        "textgrid_sha256": grid_hash,
        "tokens": [dict(token) for token in expected_tokens],
        "parsed_tokens": parsed,
        "alignment_method": "MFA",
        "model_origin": str(row.get("tts_audio_origin", "natural_recording")) if side == "tts" else "natural_recording",
        "historical_split": str(row.get("protocol_split", "unknown")),
        "analysis_split": analysis_split,
        "audit_status": status,
        "missing_reason": reasons or None,
        "video_sha256": row.get("video_sha256"),
    }


def adapt_lrs3_manifest(
    manifest_path: str | Path,
    tokens_path: str | Path,
    *,
    repo_root: str | Path,
    expected_manifest_sha256: str | None = None,
    expected_tokens_sha256: str | None = None,
    fit_groups: int = 12,
    dev_groups: int = 6,
) -> dict[str, Any]:
    """Adapt the sealed LRS3 manifest into the protocol's common schema."""

    root = Path(repo_root).resolve()
    manifest = _resolve(manifest_path, root)
    tokens_file = _resolve(tokens_path, root)
    if not manifest.is_file() or not tokens_file.is_file():
        raise InventoryError(f"registered LRS3 inputs are missing: {manifest}, {tokens_file}")
    actual_manifest_hash = sha256_file(manifest)
    actual_tokens_hash = sha256_file(tokens_file)
    if expected_manifest_sha256 and actual_manifest_hash != expected_manifest_sha256:
        raise InventoryError("LRS3 manifest SHA256 mismatch")
    if expected_tokens_sha256 and actual_tokens_hash != expected_tokens_sha256:
        raise InventoryError("LRS3 tokens SHA256 mismatch")
    manifest_payload = read_json(manifest)
    token_payload = read_json(tokens_file)
    records = manifest_payload.get("records") if isinstance(manifest_payload, Mapping) else None
    token_records = token_payload.get("records") if isinstance(token_payload, Mapping) else None
    if not isinstance(records, list) or not isinstance(token_records, Mapping):
        raise InventoryError("LRS3 manifest/tokens schema is invalid")
    train_groups = [str(row.get("source_group", "")) for row in records if str(row.get("protocol_split", "")) == "train"]
    splits = freeze_splits(train_groups, fit_groups=fit_groups, dev_groups=dev_groups)
    pairs: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for raw in records:
        pair_id = str(raw.get("sample_id", ""))
        if not pair_id or pair_id not in token_records:
            raise InventoryError(f"manifest record has no token record: {pair_id}")
        original_split = str(raw.get("protocol_split", "unknown"))
        source_group = str(raw.get("source_group", ""))
        analysis_split = "e_seen" if original_split == "evaluation" else str(splits["groups"].get(source_group, "unassigned"))
        if analysis_split == "unassigned":
            raise InventoryError(f"training group was not assigned: {source_group}")
        sides: dict[str, dict[str, Any]] = {}
        side_payload = token_records[pair_id]
        for side in ("natural", "tts"):
            payload = side_payload.get(side) if isinstance(side_payload, Mapping) else None
            if not isinstance(payload, Mapping):
                raise InventoryError(f"missing {side} token payload: {pair_id}")
            asset = _side_asset(raw, side, payload, repo_root=root, analysis_split=analysis_split)
            sides[side] = asset
            assets.append(asset)
        pairs.append({
            "pair_id": pair_id,
            "sample_id": pair_id,
            "dataset": str(raw.get("dataset", "lrs3")),
            "language": str(raw.get("language", "")),
            "source_group": source_group,
            "historical_split": original_split,
            "analysis_split": analysis_split,
            "transcript": str(raw.get("transcript", "")),
            "mfa_transcript": str(raw.get("mfa_transcript", raw.get("transcript", ""))),
            "transcript_sha256": raw.get("transcript_sha256"),
            "video_path": raw.get("video_path"),
            "video_sha256": raw.get("video_sha256"),
            "tts_audio_origin": raw.get("tts_audio_origin"),
            "sides": sides,
        })
    overlap = build_overlap_graph(assets)
    components_by_node = {node: root for root, members in overlap["components"].items() for node in members}
    split_components: dict[str, set[str]] = defaultdict(set)
    for asset in assets:
        split_components[components_by_node[str(asset["asset_id"])]].add(str(asset["analysis_split"]))
    overlap_violations = {
        root: sorted(values) for root, values in split_components.items() if len(values - {"e_seen"}) > 1
    }
    return {
        "schema_version": 1,
        "protocol": "phone_separability_mechanism_v1",
        "status": "COMPLETE",
        "source": {
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": actual_manifest_hash,
            "tokens_path": str(tokens_file.resolve()),
            "tokens_sha256": actual_tokens_hash,
        },
        "splits": splits,
        "pairs": pairs,
        "assets": assets,
        "overlap": overlap,
        "overlap_violations": overlap_violations,
        "counts": {
            "pairs": len(pairs),
            "assets": len(assets),
            "available_assets": sum(1 for row in assets if row["audit_status"] == "OK"),
            "missing_assets": sum(1 for row in assets if row["audit_status"] != "OK"),
        },
    }


def load_registry(config: Mapping[str, Any], run_dir: str | Path, *, repo_root: str | Path, smoke: bool = False) -> dict[str, Any]:
    source = config.get("source", {})
    registry = adapt_lrs3_manifest(
        str(source.get("manifest", "")),
        str(source.get("tokens", "")),
        repo_root=repo_root,
        expected_manifest_sha256=str(source.get("manifest_sha256", "")) or None,
        expected_tokens_sha256=str(source.get("tokens_sha256", "")) or None,
        fit_groups=int(config.get("splits", {}).get("fit_groups", 12)),
        dev_groups=int(config.get("splits", {}).get("dev_groups", 6)),
    )
    if smoke:
        selected = sorted(registry["pairs"], key=lambda row: hashlib.sha256(f"{row['source_group']}|{row['pair_id']}".encode()).hexdigest())[: int(config.get("runtime", {}).get("smoke_pairs", 2))]
        registry["selected_pair_ids"] = [row["pair_id"] for row in selected]
        registry["scope"] = "engineering_smoke"
    else:
        registry["selected_pair_ids"] = [row["pair_id"] for row in registry["pairs"]]
        registry["scope"] = "full_registered_cohort"
    root = Path(run_dir)
    inventory_dir = root / "00_inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(inventory_dir / "assets.jsonl", registry["assets"])
    _write_json(inventory_dir / "overlap.json", registry["overlap"])
    _write_json(inventory_dir / "splits.json", registry["splits"])
    missing_path = inventory_dir / "missing.csv"
    with missing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "pair_id", "audit_status", "missing_reason"])
        writer.writeheader()
        for asset in registry["assets"]:
            if asset["audit_status"] != "OK":
                writer.writerow({"asset_id": asset["asset_id"], "pair_id": asset["paired_key"], "audit_status": asset["audit_status"], "missing_reason": ";".join(asset.get("missing_reason") or [])})
    snapshot = dict(registry)
    snapshot.pop("assets", None)
    snapshot["asset_registry_sha256"] = _json_sha256(registry["assets"])
    with (inventory_dir / "registry_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
    _write_json(inventory_dir / "registry.json", registry)
    return registry


__all__ = [
    "InventoryError",
    "adapt_lrs3_manifest",
    "build_overlap_graph",
    "check_resources",
    "freeze_splits",
    "load_registry",
]
