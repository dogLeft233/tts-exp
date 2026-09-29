"""Strict split scoring for an already extracted phone-separability atlas.

The extractor is the expensive part of the atlas.  This module lets a run be
rescored after a bookkeeping correction without downloading or re-running the
encoders. References are still fit on FIT only; DEV and E_SEEN are scored
separately and are never pooled into a selection statistic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .diagnostics import duration_analysis
from .features import load_token_records
from .inventory import check_resources
from .metrics import abx_score, contrast_scores, domain_auc, fit_probe_bundle, score_frozen_support


REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWS = ("full", "core", "boundary_start", "boundary_end", "matched_1frame")
EVALUATION_SPLITS = ("dev", "e_seen")


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_config(protocol_path: Path) -> tuple[dict[str, Any], Path]:
    protocol = _read_json(protocol_path)
    config_path = Path(str(protocol["config_path"])).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return config, config_path


def _model_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = config.get("models", {})
    result: list[dict[str, Any]] = []
    for key in ("primary", "cross_encoder"):
        value = models.get(key)
        if isinstance(value, Mapping) and value.get("key"):
            result.append(dict(value))
    return result


def _compact_json(value: Any) -> Any:
    """Drop verbose ABX triplet listings from summary-level artifacts."""

    if isinstance(value, Mapping):
        return {str(key): _compact_json(item) for key, item in value.items() if key != "triplets"}
    if isinstance(value, list):
        return [_compact_json(item) for item in value]
    return value


def score_atlas_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Score one model with frozen FIT references and split-specific outputs."""

    probe = config.get("probe", {})
    min_tokens = int(probe.get("min_tokens_per_label", 20))
    min_groups = int(probe.get("min_groups_per_label", 3))
    min_labels = int(probe.get("min_labels_per_pair", 5))
    min_pair_tokens = int(probe.get("min_tokens_per_pair_side", 10))
    min_coverage = float(probe.get("min_speech_coverage", 0.70))
    seed = int(probe.get("bootstrap_seed", 20260921))
    draws = int(probe.get("bootstrap_draws", 10_000))

    references: dict[str, Any] = {}
    contrast_rows: list[dict[str, Any]] = []
    for view in VIEWS:
        reference = fit_probe_bundle(rows, view=view, reference="mixed", min_tokens=min_tokens, min_groups=min_groups)
        references[view] = reference
        _write_json(output_dir / "reference" / model_key / f"{view}.json", reference)
        for split in EVALUATION_SPLITS:
            scores: dict[str, Any] = {}
            for condition in ("natural", "tts"):
                scores[condition] = score_frozen_support(
                    rows,
                    reference,
                    view=view,
                    condition=condition,
                    split=split,
                    min_labels=min_labels,
                    min_tokens=min_pair_tokens,
                    min_coverage=min_coverage,
                )
            contrast = contrast_scores(scores, "tts", "natural", seed=seed, draws=draws)
            contrast.update({"model_key": model_key, "view": view, "split": split, "reference_fit_splits": ["fit"], "evaluation_scope": "frozen_fit_reference"})
            contrast_rows.append(contrast)

    _write_jsonl(output_dir / "contrasts" / f"{model_key}.jsonl", contrast_rows)
    natural = [row for row in rows if row.get("condition") == "natural"]
    tts = [row for row in rows if row.get("condition") == "tts"]
    duration = {split: duration_analysis(rows, view="full", split=split) for split in EVALUATION_SPLITS}
    domain = {split: domain_auc(rows, view="full", split=split) for split in EVALUATION_SPLITS}
    abx_natural = {split: abx_score(natural, view="full", split=split) for split in EVALUATION_SPLITS}
    abx_tts = {split: abx_score(tts, view="full", split=split) for split in EVALUATION_SPLITS}
    return {
        "row_count": len(rows),
        "reference_labels": {view: references[view].get("labels", []) for view in VIEWS},
        "contrast_rows": contrast_rows,
        "duration": duration,
        "domain": domain,
        "abx_natural": {split: _compact_json(value) for split, value in abx_natural.items()},
        "abx_tts": {split: _compact_json(value) for split, value in abx_tts.items()},
        "evaluation_splits": list(EVALUATION_SPLITS),
        "selection_uses_e_seen": False,
    }


def rescore_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    config, config_path = _load_config(root / "protocol.json")
    runtime = config.get("runtime", {})
    resources = check_resources(
        REPO_ROOT,
        device="cpu",
        min_free_disk_gib=float(runtime.get("min_free_disk_gib", 4.0)),
        min_free_gpu_gib=float(runtime.get("min_free_gpu_gib", 8.0)),
        min_available_ram_gib=float(runtime.get("min_available_ram_gib", 8.0)),
        stage_estimated_bytes=256 * 1024 * 1024,
    )
    if resources.get("decision") != "ALLOW":
        raise RuntimeError(f"RESOURCE_BUSY: {resources.get('reason_codes', [])}")
    models: dict[str, Any] = {}
    for model_cfg in _model_configs(config):
        model_key = str(model_cfg["key"])
        feature_path = root / "02_features" / model_key / "token_records.jsonl"
        if not feature_path.is_file():
            raise FileNotFoundError(f"feature store missing: {feature_path}")
        rows = load_token_records(feature_path)
        models[model_key] = {"meta": _read_json(root / "02_features" / model_key / "feature_meta.json"), **score_atlas_rows(rows, model_key=model_key, config=config, output_dir=root / "03_atlas")}
    summary = {
        "schema_version": 2,
        "protocol": "phone_separability_mechanism_v1",
        "scope": "registered_exploratory_strict_rescore",
        "models": models,
        "evaluation_is_seen": True,
        "selection_uses_e_seen": False,
        "scoring": "FIT references frozen before separate DEV/E_SEEN scoring",
        "rescore_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_path": str(config_path),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resource_snapshot": resources,
    }
    _write_json(root / "03_atlas" / "strict_summary.json", summary)
    # Keep the canonical summary/report path aligned with the corrected scoring
    # so downstream tools cannot silently read the old all-split result.
    _write_json(root / "03_atlas" / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    summary = rescore_run(args.run_dir)
    print(json.dumps({"status": "COMPLETE", "models": list(summary["models"]), "run_dir": str(Path(args.run_dir).resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "rescore_run", "score_atlas_rows"]
