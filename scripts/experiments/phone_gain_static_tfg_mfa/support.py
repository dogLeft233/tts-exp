"""Frozen natural and matched-N/T support contracts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from scripts.experiments.lrs3_phone_rules_metrics import normalize_phone

from .config import REPO_ROOT, canonical_hash, read_json, read_jsonl, write_json


def _speech_tokens(row: Mapping[str, Any], side: str) -> list[dict[str, Any]]:
    return [dict(token) for token in row[side].get("tokens", []) if bool(token.get("speech", not token.get("silence", False))) and normalize_phone(token.get("label", ""))]


def freeze_phone_support(config: Mapping[str, Any], run_dir: str | Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    parent_support = read_json(REPO_ROOT / source["parent_support"])
    labels = sorted(str(label) for label in parent_support["labels"])
    label_set = set(labels)
    natural_entries: dict[str, list[dict[str, Any]]] = {"hubert": [], "xlsr": []}
    natural_seen: set[tuple[str, str]] = set()
    for row in registry["rows"]:
        natural = _speech_tokens(row, "natural")
        eligible = [token for token in natural if normalize_phone(token["label"]) in label_set]
        coverage = len(eligible) / max(len(natural), 1)
        labels_in_row = {normalize_phone(token["label"]) for token in eligible}
        row_ok = len(labels_in_row) >= int(config["support"]["min_labels_per_pair"]) and len(eligible) >= int(config["support"]["min_tokens_per_pair"]) and coverage >= float(config["support"]["min_speech_coverage"])
        for token in eligible:
            item = {"pair_id": row["pair_id"], "source_group": row["source_group"], "analysis_split": row["analysis_split"], "label": normalize_phone(token["label"]), "natural_token_index": int(token["token_index"]), "natural_token_id": str(token["token_id"]), "natural_start_s": float(token["start_s"]), "natural_end_s": float(token["end_s"]), "support_key": f"{row['pair_id']}:{token['token_index']}:{normalize_phone(token['label'])}", "pair_eligible": bool(row_ok), "frame_indices": None}
            for encoder in natural_entries:
                key = (encoder, str(item["support_key"]))
                if key in natural_seen:
                    raise ValueError(f"duplicate natural support key: {key}")
                natural_seen.add(key)
                natural_entries[encoder].append(dict(item))
    matched_rows = read_jsonl(REPO_ROOT / source["parent_support_jsonl"])
    matched: dict[str, list[dict[str, Any]]] = {"hubert": [], "xlsr": []}
    matched_seen: set[tuple[str, str]] = set()
    row_by_id = {str(row["pair_id"]): row for row in registry["rows"]}
    for item in matched_rows:
        encoder = str(item.get("encoder"))
        if encoder not in matched:
            continue
        copy = {key: value for key, value in item.items() if key != "encoder"}
        row = row_by_id.get(str(copy.get("pair_id")))
        if row is not None:
            natural = row["natural"]["tokens"]
            tts = row["tts"]["tokens"]
            natural_speech = [token for token in natural if bool(token.get("speech", not token.get("silence", False))) and normalize_phone(token.get("label", ""))]
            natural_eligible = [token for token in natural_speech if normalize_phone(token.get("label", "")) in label_set]
            row_labels = {normalize_phone(token.get("label", "")) for token in natural_eligible}
            row_ok = len(row_labels) >= int(config["support"]["min_labels_per_pair"]) and len(natural_eligible) >= int(config["support"]["min_tokens_per_pair"]) and len(natural_eligible) / max(len(natural_speech), 1) >= float(config["support"]["min_speech_coverage"])
            natural_index = int(copy.get("natural_token_index", -1))
            tts_index = int(copy.get("tts_token_index", -1))
            if 0 <= natural_index < len(natural) and 0 <= tts_index < len(tts):
                natural_label = normalize_phone(natural[natural_index].get("label", ""))
                tts_label = normalize_phone(tts[tts_index].get("label", ""))
                copy.update({"natural_start_s": float(natural[natural_index]["start_s"]), "natural_end_s": float(natural[natural_index]["end_s"]), "tts_start_s": float(tts[tts_index]["start_s"]), "tts_end_s": float(tts[tts_index]["end_s"]), "pair_eligible": bool(row_ok) and bool(natural[natural_index].get("speech", True)) and bool(tts[tts_index].get("speech", True)) and natural_label == tts_label, "frame_indices": None})
            else:
                copy["pair_eligible"] = False
        support_key = str(copy.get("support_key", f"{copy.get('pair_id')}:{copy.get('natural_token_index')}:{copy.get('label')}"))
        copy["support_key"] = support_key
        key = (encoder, support_key)
        if key in matched_seen:
            raise ValueError(f"duplicate matched support key: {key}")
        matched_seen.add(key)
        matched[encoder].append(copy)
    payload = {
        "schema_version": 1,
        "measurement_version": "signed_margin_fixed_support_v2",
        "labels": labels,
        "natural_centroids": parent_support["natural_centroids"],
        "mixed_centroids": parent_support["mixed_centroids"],
        "natural_primary": natural_entries,
        "matched_nt": matched,
        "legacy_matched": {"entries": parent_support["entries"], "natural_centroids": parent_support["natural_centroids"], "mixed_centroids": parent_support["mixed_centroids"], "support_hash": parent_support.get("support_hash")},
        "source": {"parent_support": str((REPO_ROOT / source["parent_support"]).resolve()), "parent_support_sha256": __import__("hashlib").sha256((REPO_ROOT / source["parent_support"]).read_bytes()).hexdigest()},
    }
    payload["support_hash"] = canonical_hash({"labels": labels, "natural_primary": natural_entries, "matched_nt": matched, "mixed_centroids": payload["mixed_centroids"]})
    write_json(Path(run_dir) / "00_protocol/support.json", payload)
    return payload


def load_support(run_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(run_dir) / "00_protocol/support.json")


def entries_for(support: Mapping[str, Any], encoder: str, view: str, split: str | None = None) -> list[dict[str, Any]]:
    key = "natural_primary" if view == "natural_primary" else "matched_nt"
    rows = list(support.get(key, {}).get(encoder, []))
    return [row for row in rows if split is None or str(row.get("analysis_split")) == str(split)]


__all__ = ["entries_for", "freeze_phone_support", "load_support"]
