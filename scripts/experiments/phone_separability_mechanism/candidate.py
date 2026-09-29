"""Re-extract constructed natural-clock candidates and apply DEV-only gates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .features import extract_asset_views, load_bundle, load_token_records
from .inventory import check_resources
from .metrics import contrast_scores, score_frozen_support
from scripts.experiments.lrs3_phone_rules_worker import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWS = ("full", "core", "boundary_start", "boundary_end", "matched_1frame")
DEFAULT_SELECTABLE_ARMS = (
    "SPECTRAL_DRC",
    "PHONE_SHAPE_BETA05_CORE",
    "PHONE_SHAPE_BETA1_CORE",
    "PHONE_SHAPE_BETA05_SPEECH",
    "PHONE_SHAPE_BETA1_SPEECH",
    "EQ_MATCH",
)


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_config(run_dir: Path) -> tuple[dict[str, Any], Path]:
    protocol = dict(read_json(run_dir / "protocol.json"))
    config_path = Path(str(protocol["config_path"])).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config, config_path


def _model_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = config.get("models", {})
    result: list[dict[str, Any]] = []
    for key in ("primary", "cross_encoder"):
        value = models.get(key)
        if isinstance(value, Mapping) and value.get("key"):
            result.append(dict(value))
    return result


def _candidate_asset(row: Mapping[str, Any], pair: Mapping[str, Any]) -> dict[str, Any]:
    natural = pair["sides"]["natural"]
    output = row["output"]
    return {
        "asset_id": f"candidate:{row['pair_id']}:{row['arm']}",
        "paired_key": row["pair_id"],
        "sample_id": row["sample_id"],
        "source_group": row["source_group"],
        "condition": "candidate",
        "analysis_split": row["analysis_split"],
        "input_mode": "natural_only",
        "clock_owner": "natural",
        "audio_path": output["path"],
        "textgrid_sha256": natural.get("textgrid_sha256"),
        "tokens": natural.get("tokens", []),
    }


def _extract_arm(
    bundle: Mapping[str, Any],
    construction_rows: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Any]],
    *,
    arm: str,
    split: str,
    sample_rate: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in construction_rows:
        if str(row.get("arm")) != arm or str(row.get("analysis_split")) != split:
            continue
        pair = pairs.get(str(row["pair_id"]))
        if pair is None:
            continue
        asset = _candidate_asset(row, pair)
        extracted = extract_asset_views(bundle, asset, layer=int(bundle["_candidate_layer"]), sample_rate=sample_rate)
        for item in extracted:
            item["arm"] = arm
            item["candidate_output_sha256"] = row["output"].get("container_sha256")
        result.extend(extracted)
    return result


def _reference(run_dir: Path, model_key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for view in VIEWS:
        result[view] = dict(read_json(run_dir / "03_atlas" / "reference" / model_key / f"{view}.json"))
    return result


def _score_candidate(
    natural_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    model_key: str,
    arm: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    probe = config.get("probe", {})
    result: dict[str, Any] = {"model_key": model_key, "arm": arm, "split": split, "views": {}}
    for view in VIEWS:
        reference = references[view]
        rows = list(natural_rows) + list(candidate_rows)
        scores = {
            "natural": score_frozen_support(rows, reference, view=view, condition="natural", split=split, min_labels=int(probe.get("min_labels_per_pair", 5)), min_tokens=int(probe.get("min_tokens_per_pair_side", 10)), min_coverage=float(probe.get("min_speech_coverage", 0.70))),
            "candidate": score_frozen_support(rows, reference, view=view, condition="candidate", split=split, min_labels=int(probe.get("min_labels_per_pair", 5)), min_tokens=int(probe.get("min_tokens_per_pair_side", 10)), min_coverage=float(probe.get("min_speech_coverage", 0.70))),
        }
        result["views"][view] = contrast_scores(scores, "candidate", "natural", seed=int(probe.get("bootstrap_seed", 20260921)), draws=int(probe.get("bootstrap_draws", 10000)))
        result["views"][view]["evaluation_scope"] = "frozen_fit_reference"
    return result


def _select_dev(scores: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    candidate_cfg = config.get("candidate", {})
    view = str(candidate_cfg.get("selection_view", "core"))
    min_groups = int(candidate_cfg.get("min_gate_groups", 5))
    min_accuracy_ci_low = float(candidate_cfg.get("min_accuracy_ci_low", 0.0))
    min_margin_ci_low = float(candidate_cfg.get("min_margin_ci_low", 0.0))
    min_tts_fraction = float(candidate_cfg.get("min_tts_fraction", 0.50))
    selectable = set(str(value) for value in candidate_cfg.get("selectable_arms", DEFAULT_SELECTABLE_ARMS))
    ranked: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in scores:
        arm = str(row["arm"])
        if arm not in selectable:
            continue
        metrics = row["views"].get(view, {})
        accuracy = metrics.get("accuracy", {})
        margin = metrics.get("margin", {})
        effect = accuracy.get("estimate")
        tts_effect = row.get("tts_baseline", {}).get("accuracy", {}).get("estimate")
        reasons: list[str] = []
        if int(accuracy.get("n_groups", 0)) < min_groups:
            reasons.append("DEV_GROUPS_BELOW_MIN")
        if effect is None or accuracy.get("ci_low") is None or float(accuracy["ci_low"]) <= min_accuracy_ci_low:
            reasons.append("CANDIDATE_ACCURACY_CI_NOT_POSITIVE")
        if margin.get("ci_low") is None or float(margin["ci_low"]) <= min_margin_ci_low:
            reasons.append("CANDIDATE_MARGIN_CI_NOT_POSITIVE")
        if tts_effect is None or effect is None or float(effect) < min_tts_fraction * float(tts_effect):
            reasons.append("BELOW_TTS_EFFECT_FRACTION")
        candidate = {"arm": arm, "view": view, "effect": effect, "accuracy_ci_low": accuracy.get("ci_low"), "margin": margin.get("estimate"), "margin_ci_low": margin.get("ci_low"), "tts_effect": tts_effect, "gate_pass": not reasons, "reason_codes": reasons}
        (ranked if not reasons else rejected).append(candidate)
    ranked.sort(key=lambda row: (-float(row["margin"] if row["margin"] is not None else -1e9), -float(row["effect"] if row["effect"] is not None else -1e9), str(row["arm"])))
    selected = ranked[0] if ranked else None
    return {"status": "SELECTED" if selected else "NO_RULE_PASSES_DEV_GATE", "selection_split": "dev", "selection_view": view, "selected_arm": selected["arm"] if selected else None, "selected": selected, "passed": ranked, "rejected": rejected, "e_seen_locked": True, "gate": {"min_groups": min_groups, "min_accuracy_ci_low": min_accuracy_ci_low, "min_margin_ci_low": min_margin_ci_low, "min_tts_fraction": min_tts_fraction}}


def run_candidates(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    config, config_path = _load_config(root)
    runtime = config.get("runtime", {})
    device = str(config.get("models", {}).get("device", "cuda:0"))
    resources = check_resources(REPO_ROOT, device=device, min_free_disk_gib=float(runtime.get("min_free_disk_gib", 4.0)), min_free_gpu_gib=float(runtime.get("min_free_gpu_gib", 8.0)), min_available_ram_gib=float(runtime.get("min_available_ram_gib", 8.0)), stage_estimated_bytes=256 * 1024 * 1024)
    if resources.get("decision") != "ALLOW":
        raise RuntimeError(f"RESOURCE_BUSY: {resources.get('reason_codes', [])}")
    registry = dict(read_json(root / "00_inventory" / "registry.json"))
    pairs = {str(row["pair_id"]): row for row in registry.get("pairs", [])}
    construction = _read_jsonl(root / "05_construction" / "construction.jsonl")
    output_dir = root / "07_candidates"
    all_model_scores: dict[str, Any] = {}
    primary_key = str(config.get("models", {}).get("primary", {}).get("key", "hubert"))
    selected_arm: str | None = None
    for model_cfg in _model_configs(config):
        model_key = str(model_cfg["key"])
        atlas_rows = load_token_records(root / "02_features" / model_key / "token_records.jsonl")
        natural_dev = [row for row in atlas_rows if row.get("condition") == "natural" and row.get("analysis_split") == "dev"]
        references = _reference(root, model_key)
        bundle = load_bundle(model_cfg, device=device, allow_download=bool(runtime.get("allow_model_download", True)), proxy=str(runtime.get("proxy", "")) or None)
        bundle = {**bundle, "_candidate_layer": int(model_cfg["layer"])}
        arm_scores: list[dict[str, Any]] = []
        arms = list(config.get("candidate", {}).get("candidate_arms", _construction_arms_fallback(construction)))
        for arm in arms:
            candidate_rows = _extract_arm(bundle, construction, pairs, arm=str(arm), split="dev", sample_rate=int(config.get("audio", {}).get("sample_rate", 16000)))
            scored = _score_candidate(natural_dev, candidate_rows, references, split="dev", model_key=model_key, arm=str(arm), config=config)
            baseline_rows = [row for row in read_json(root / "03_atlas" / "summary.json").get("models", {}).get(model_key, {}).get("contrast_rows", []) if row.get("split") == "dev"]
            baseline_by_view = {str(row.get("view")): row for row in baseline_rows}
            scored["tts_baseline"] = {view: baseline_by_view.get(view, {}) for view in VIEWS}
            # The gate uses the same view, with TTS's frozen contrast as target.
            scored["tts_baseline"] = scored["tts_baseline"].get(str(config.get("candidate", {}).get("selection_view", "core")), {})
            scored["candidate_row_count"] = len(candidate_rows)
            arm_scores.append(scored)
        if model_key == primary_key:
            selection = _select_dev(arm_scores, config)
            selected_arm = selection.get("selected_arm")
        else:
            selection = {"status": "HELD_OUT", "selection_split": "dev", "selected_arm": selected_arm, "e_seen_locked": True}
        e_seen_scores: list[dict[str, Any]] = []
        if selected_arm:
            natural_seen = [row for row in atlas_rows if row.get("condition") == "natural" and row.get("analysis_split") == "e_seen"]
            candidate_seen = _extract_arm(bundle, construction, pairs, arm=selected_arm, split="e_seen", sample_rate=int(config.get("audio", {}).get("sample_rate", 16000)))
            e_seen_scores.append(_score_candidate(natural_seen, candidate_seen, references, split="e_seen", model_key=model_key, arm=selected_arm, config=config))
            e_seen_scores[-1]["candidate_row_count"] = len(candidate_seen)
        all_model_scores[model_key] = {"dev": arm_scores, "e_seen_selected": e_seen_scores, "reference_fit_splits": ["fit"], "selection_uses_e_seen": False}
        del bundle, atlas_rows
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    selection = _select_dev(all_model_scores.get(primary_key, {}).get("dev", []), config)
    selection["selected_arm"] = selected_arm
    selection["config_path"] = str(config_path)
    selection["resource_snapshot"] = resources
    _write_json(output_dir / "selection.json", selection)
    for model_key, model_scores in all_model_scores.items():
        _write_jsonl(output_dir / f"scores_{model_key}.jsonl", model_scores.get("dev", []))
        _write_json(output_dir / f"e_seen_{model_key}.json", {"scores": model_scores.get("e_seen_selected", []), "selection_uses_e_seen": False})
    summary = {"schema_version": 1, "status": selection["status"], "selected_arm": selected_arm, "models": {key: {"dev_arms": len(value["dev"]), "e_seen_selected": len(value["e_seen_selected"])} for key, value in all_model_scores.items()}, "selection_uses_e_seen": False, "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _write_json(output_dir / "summary.json", summary)
    return {"summary": summary, "selection": selection, "models": all_model_scores}


def _construction_arms_fallback(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row.get("arm")) for row in rows if row.get("arm")})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    result = run_candidates(args.run_dir)
    print(json.dumps({"status": result["summary"]["status"], "selected_arm": result["summary"]["selected_arm"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_candidates"]
