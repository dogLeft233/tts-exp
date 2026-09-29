"""Asset inheritance, deterministic pilot selection, and frozen support plans."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from scripts.experiments.phone_separability_mechanism.features import match_occurrences
from scripts.experiments.phone_separability_mechanism.inventory import adapt_lrs3_manifest
from scripts.experiments.lrs3_phone_rules_metrics import normalize_phone

from .config import canonical_hash, read_json, read_jsonl, resolve_path, write_json, write_jsonl


VIEWS = ("core", "full", "matched_1frame")


@dataclass(frozen=True)
class FeatureStore:
    encoder: str
    view: str
    records: tuple[dict[str, Any], ...]
    vectors: np.ndarray
    source_dir: Path

    def vector(self, row: Mapping[str, Any]) -> np.ndarray | None:
        index = row.get("embedding_index")
        if index is None or not bool(row.get("valid", False)):
            return None
        value = np.asarray(self.vectors[int(index)], dtype=np.float64)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            return None
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 0.0 else None


def _sha_pair(pair_id: str, prefix: str) -> str:
    return hashlib.sha256(f"{prefix}|{pair_id}".encode("utf-8")).hexdigest()


def inherit_registry(config: Mapping[str, Any], repo_root: str | Path, run_dir: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    """Re-validate the registered parent inputs without mutating the parent run."""

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
    selected = list(registry["pairs"])
    if smoke:
        count = int(config.get("runtime", {}).get("smoke_pairs", 2))
        selected = sorted((row for row in selected if str(row.get("analysis_split")) == "fit"), key=lambda row: _sha_pair(str(row["pair_id"]), "pse-v2-smoke"))[:count]
    registry["selected_pair_ids"] = [str(row["pair_id"]) for row in selected]
    registry["scope"] = "engineering_smoke" if smoke else "full_registered_cohort"
    audit_dir = Path(run_dir) / "00_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / "registry.json", registry)
    write_jsonl(audit_dir / "assets.jsonl", [dict(row) for row in registry["assets"]])
    write_json(audit_dir / "splits.json", registry["splits"])
    write_json(audit_dir / "overlap.json", registry["overlap"])
    return registry


def select_pilot(registry: Mapping[str, Any], config: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Select by source group and hash before any candidate score is available."""

    pilot_cfg = config.get("pilot", {})
    result: dict[str, Any] = {"algorithm": "sha256(pse-v2-pilot|pair_id)", "fit": [], "dev": [], "e_seen": [], "excluded": []}
    allowed = set(str(value) for value in registry.get("selected_pair_ids", [])) if smoke else None
    by_split_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in registry.get("pairs", []):
        if allowed is not None and str(pair.get("pair_id")) not in allowed:
            continue
        split = str(pair.get("analysis_split", ""))
        group = str(pair.get("source_group", ""))
        by_split_group[(split, group)].append(dict(pair))
    for (split, group), rows in sorted(by_split_group.items()):
        ordered = sorted(rows, key=lambda row: _sha_pair(str(row["pair_id"]), "pse-v2-pilot"))
        if split == "fit":
            limit = 2 if smoke else int(pilot_cfg.get("fit_per_group", 2))
        elif split == "dev":
            limit = 2 if smoke else int(pilot_cfg.get("dev_per_group", 4))
        else:
            limit = 0 if smoke else len(ordered)
        chosen = ordered[:limit]
        result.setdefault(split, []).extend(str(row["pair_id"]) for row in chosen)
        result["excluded"].extend({"pair_id": str(row["pair_id"]), "split": split, "reason": "PILOT_CAP", "source_group": group} for row in ordered[limit:])
    result["fit"].sort()
    result["dev"].sort()
    result["e_seen"].sort()
    result["pilot_hash"] = canonical_hash({key: result[key] for key in ("fit", "dev", "e_seen")})
    return result


def pair_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["pair_id"]): dict(row) for row in registry.get("pairs", [])}


def get_side(pair: Mapping[str, Any], side: str) -> dict[str, Any]:
    if side not in {"natural", "tts"}:
        raise ValueError(f"unknown side: {side}")
    value = pair.get("sides", {}).get(side)
    if not isinstance(value, Mapping):
        raise KeyError(f"pair has no {side} side: {pair.get('pair_id')}")
    return dict(value)


def load_feature_store(parent_run: str | Path, encoder: str, view: str) -> FeatureStore:
    source_dir = Path(parent_run) / "02_features" / encoder
    records = read_jsonl(source_dir / "token_records.jsonl")
    with np.load(source_dir / "embeddings.npz", allow_pickle=False) as payload:
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    filtered = tuple(row for row in records if str(row.get("view")) == str(view))
    return FeatureStore(encoder=str(encoder), view=str(view), records=filtered, vectors=vectors, source_dir=source_dir)


def _row_map(store: FeatureStore) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (str(row.get("sample_id")), str(row.get("condition")), int(row.get("token_index", -1))): row
        for row in store.records
    }


def _matched_token_indices(pair: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    natural = get_side(pair, "natural").get("tokens", [])
    tts = get_side(pair, "tts").get("tokens", [])
    matched = match_occurrences(natural, tts, speech_only=True)
    return [(int(row["source_index"]), int(row["target_index"]), normalize_phone(row["label"])) for row in matched.get("matches", [])]


def _group_label_stats(store: FeatureStore, registry: Mapping[str, Any], *, condition: str, split: str, min_tokens: int, min_groups: int) -> dict[str, dict[str, Any]]:
    pairs = pair_index(registry)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in store.records:
        if str(row.get("condition")) != condition or str(row.get("analysis_split")) != split or not bool(row.get("valid", False)) or not bool(row.get("speech", True)):
            continue
        label = normalize_phone(row.get("label", ""))
        group = str(row.get("source_group", ""))
        if label and group and str(row.get("sample_id")) in pairs:
            counts[label][str(row.get("sample_id"))] += 1
            groups[label].add(group)
    result: dict[str, dict[str, Any]] = {}
    for label in sorted(counts):
        token_count = sum(counts[label].values())
        group_count = len(groups[label])
        result[label] = {"token_count": token_count, "group_count": group_count, "eligible": token_count >= min_tokens and group_count >= min_groups, "groups": sorted(groups[label])}
    return result


def _center(values: Iterable[np.ndarray]) -> list[float] | None:
    vectors = [value for value in values if value is not None]
    if not vectors:
        return None
    result = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(result))
    return (result / norm).astype(np.float32).tolist() if norm > 0.0 and np.all(np.isfinite(result)) else None


def _fit_centroids(store: FeatureStore, registry: Mapping[str, Any], labels: set[str], *, conditions: tuple[str, ...]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Return natural-only and condition-balanced frozen centroids."""

    by_label_condition_group: dict[str, dict[str, dict[str, list[np.ndarray]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in store.records:
        if str(row.get("analysis_split")) != "fit" or str(row.get("condition")) not in conditions or not bool(row.get("valid", False)) or not bool(row.get("speech", True)):
            continue
        label = normalize_phone(row.get("label", ""))
        group = str(row.get("source_group", ""))
        if label not in labels or not group:
            continue
        vector = store.vector(row)
        if vector is not None:
            by_label_condition_group[label][str(row["condition"])][group].append(vector)
    natural: dict[str, list[float]] = {}
    mixed: dict[str, list[float]] = {}
    for label in sorted(labels):
        group_centers_n = [_center(values) for values in by_label_condition_group[label].get("natural", {}).values()]
        n = _center(np.asarray(value, dtype=np.float64) for value in group_centers_n if value is not None)
        if n is not None:
            natural[label] = n
        condition_centers: list[np.ndarray] = []
        for condition in conditions:
            condition_group_centers = [_center(values) for values in by_label_condition_group[label].get(condition, {}).values()]
            condition_center = _center(np.asarray(value, dtype=np.float64) for value in condition_group_centers if value is not None)
            if condition_center is not None:
                condition_centers.append(np.asarray(condition_center, dtype=np.float64))
        mixed_center = _center(condition_centers)
        if mixed_center is not None:
            mixed[label] = mixed_center
    return natural, mixed


def freeze_support(registry: Mapping[str, Any], config: Mapping[str, Any], *, parent_run: str | Path, pilot: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze labels and occurrence slots before new candidate generation."""

    probe = config.get("probe", {})
    stores: dict[str, FeatureStore] = {}
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    label_sets: list[set[str]] = []
    for encoder in ("hubert", "xlsr"):
        store = load_feature_store(parent_run, encoder, "core")
        stores[encoder] = store
        natural = _group_label_stats(store, registry, condition="natural", split="fit", min_tokens=int(probe.get("min_tokens_per_label", 20)), min_groups=int(probe.get("min_groups_per_label", 3)))
        tts = _group_label_stats(store, registry, condition="tts", split="fit", min_tokens=int(probe.get("min_tokens_per_label", 20)), min_groups=int(probe.get("min_groups_per_label", 3)))
        stats[encoder] = {"natural": natural, "tts": tts}
        label_sets.append({label for label in natural if natural[label]["eligible"] and label in tts and tts[label]["eligible"]})
    labels = sorted(set.intersection(*label_sets) if label_sets else set())
    pair_lookup = pair_index(registry)
    # Build the identity index once per encoder.  Rebuilding it inside the
    # occurrence loop makes support freezing quadratic in the full feature
    # store size (the formal cohort has hundreds of thousands of rows).
    row_maps = {encoder: _row_map(store) for encoder, store in stores.items()}
    entries: dict[str, list[dict[str, Any]]] = {encoder: [] for encoder in stores}
    for pair_id in sorted(set(pilot.get("fit", [])) | set(pilot.get("dev", [])) | set(pilot.get("e_seen", []))):
        pair = pair_lookup.get(str(pair_id))
        if pair is None:
            continue
        matches = _matched_token_indices(pair)
        for natural_index, tts_index, label in matches:
            if label not in labels:
                continue
            for encoder in stores:
                rows = row_maps[encoder]
                natural_row = rows.get((str(pair_id), "natural", natural_index))
                tts_row = rows.get((str(pair_id), "tts", tts_index))
                if natural_row is None or tts_row is None or not bool(natural_row.get("valid")):
                    continue
                entries[encoder].append({
                    "pair_id": str(pair_id),
                    "source_group": str(pair["source_group"]),
                    "analysis_split": str(pair["analysis_split"]),
                    "label": label,
                    "natural_token_index": natural_index,
                    "tts_token_index": tts_index,
                    "natural_token_id": natural_row.get("token_id"),
                    "tts_token_id": tts_row.get("token_id"),
                    "support_key": f"{pair_id}:{natural_index}:{tts_index}:{label}",
                })
    natural_centroids: dict[str, dict[str, list[float]]] = {}
    mixed_centroids: dict[str, dict[str, list[float]]] = {}
    for encoder, store in stores.items():
        n, m = _fit_centroids(store, registry, set(labels), conditions=("natural", "tts"))
        natural_centroids[encoder] = n
        mixed_centroids[encoder] = m
    support = {
        "schema_version": 2,
        "measurement_version": "signed_margin_fixed_support_v2",
        "fit_scope": "all_fit_groups",
        "candidate_scope": {"fit": list(pilot.get("fit", [])), "dev": list(pilot.get("dev", [])), "e_seen": list(pilot.get("e_seen", []))},
        "labels": labels,
        "stats": stats,
        "entries": entries,
        "natural_centroids": natural_centroids,
        "mixed_centroids": mixed_centroids,
        "support_hash": canonical_hash({"labels": labels, "entries": entries, "natural_centroids": natural_centroids, "mixed_centroids": mixed_centroids}),
        "source": {"parent_run": str(Path(parent_run).resolve()), "encoders": ["hubert", "xlsr"], "view": "core"},
    }
    return support


def write_support(run_dir: str | Path, support: Mapping[str, Any]) -> None:
    audit_dir = Path(run_dir) / "00_audit"
    write_json(audit_dir / "support_plan.json", dict(support))
    rows = []
    for encoder, values in support.get("entries", {}).items():
        rows.extend({"encoder": encoder, **dict(row)} for row in values)
    write_jsonl(Path(run_dir) / "01_calibration" / "support.jsonl", rows)


__all__ = [
    "FeatureStore",
    "VIEWS",
    "freeze_support",
    "get_side",
    "inherit_registry",
    "load_feature_store",
    "pair_index",
    "select_pilot",
    "write_support",
]
