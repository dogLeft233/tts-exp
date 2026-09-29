"""Independent artifact checker for ``lrs3_phone_rules`` runs.

This checker deliberately does not import the runner's stage-decision
functions.  It verifies hashes, rebuilds the group statistics from JSONL, and
reconstructs the mask from the frozen natural tokens before checking PCM
preservation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.experiments.lrs3_phone_rules_audio import (
        build_edit_mask,  # type: ignore[import-not-found]
    )
    from scripts.experiments.lrs3_phone_rules_metrics import (
        paired_bootstrap,  # type: ignore[import-not-found]
    )
    from scripts.experiments.lrs3_phone_rules_worker import (  # type: ignore[import-not-found]
        read_json,
        read_pcm16,
    )
else:  # pragma: no cover
    from .lrs3_phone_rules_audio import build_edit_mask
    from .lrs3_phone_rules_metrics import paired_bootstrap
    from .lrs3_phone_rules_worker import read_json, read_pcm16


class CheckFailure(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CheckFailure(f"missing artifact: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CheckFailure(f"invalid JSONL {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise CheckFailure(f"JSONL row is not an object {path}:{line_number}")
        rows.append(value)
    return rows


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        left_f = float(left)
        right_f = float(right)
    except (TypeError, ValueError):
        return left == right
    if not math.isfinite(left_f) or not math.isfinite(right_f):
        return False
    return abs(left_f - right_f) <= tolerance


def verify_artifact_graph(run_dir: Path) -> dict[str, Any]:
    required = [run_dir / "00_audit" / "cohort.json", run_dir / "protocol.json", run_dir / "status.json", run_dir / "03_stage_a" / "decision.json", run_dir / "03_stage_a" / "statistics.json", run_dir / "03_stage_a" / "per_pair.jsonl"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CheckFailure(f"required artifacts missing: {missing}")
    cohort = dict(read_json(required[0]))
    if cohort.get("status") != "COMPLETE" or cohort.get("failures"):
        raise CheckFailure("input audit is not complete")
    stage_a_decision = dict(read_json(run_dir / "03_stage_a" / "decision.json"))
    stage_a_passed = stage_a_decision.get("science_decision") == "ADVANTAGE_SUPPORTED"
    stage_b_decision_path = run_dir / "05_stage_b" / "decision.json"
    if not stage_b_decision_path.is_file():
        raise CheckFailure("Stage B decision is missing")
    stage_b_decision = dict(read_json(stage_b_decision_path))
    if not stage_a_passed:
        rule_dir = run_dir / "04_rules"
        wavs = list(rule_dir.glob("**/*.wav")) if rule_dir.is_dir() else []
        if wavs:
            raise CheckFailure("B WAVs exist although Stage A did not pass")
    return {
        "cohort": cohort,
        "stage_a_decision": stage_a_decision,
        "stage_b_decision": stage_b_decision,
        "stage_a_passed": stage_a_passed,
    }


def recompute_statistics(run_dir: Path, *, stage: str = "a") -> dict[str, Any]:
    if stage == "a":
        rows = _read_jsonl(run_dir / "03_stage_a" / "per_pair.jsonl")
        recorded = dict(read_json(run_dir / "03_stage_a" / "statistics.json"))
        eligible = [row for row in rows if row.get("eligible")]
        expected_models = recorded.get("models", {})
        seed = int(next(iter(expected_models.values())).get("effect_tts_minus_natural", {}).get("seed", 20260920)) if expected_models else 20260920
        draws = int(next(iter(expected_models.values())).get("effect_tts_minus_natural", {}).get("draws", 10_000)) if expected_models else 10_000
        recomputed: dict[str, Any] = {}
        for model_key in expected_models:
            grouped: dict[str, list[float]] = {}
            natural: list[float] = []
            tts: list[float] = []
            for row in eligible:
                model = row.get("models", {}).get(model_key, {})
                if not model:
                    continue
                n = float(model["natural"]["accuracy"])
                t = float(model["tts"]["accuracy"])
                natural.append(n)
                tts.append(t)
                grouped.setdefault(str(row["source_group"]), []).append(t - n)
            effects = {group: float(np.mean(values)) for group, values in grouped.items()}
            bootstrap = (
                paired_bootstrap(effects, seed=seed, draws=draws)
                if effects
                else {
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "n_groups": 0,
                    "seed": seed,
                    "draws": draws,
                }
            )
            recomputed[model_key] = {
                "n_pairs": len(eligible),
                "n_source_groups": len(effects),
                "mean_natural": float(np.mean(natural)) if natural else None,
                "mean_tts": float(np.mean(tts)) if tts else None,
                "effect_tts_minus_natural": bootstrap,
            }
        for model_key, expected in expected_models.items():
            actual = recomputed.get(model_key, {})
            for field in ("n_pairs", "n_source_groups", "mean_natural", "mean_tts"):
                if expected.get(field) != actual.get(field) and not _close(expected.get(field), actual.get(field)):
                    raise CheckFailure(f"Stage A statistic mismatch {model_key}.{field}")
            for field in ("estimate", "ci_low", "ci_high"):
                if not _close(expected.get("effect_tts_minus_natural", {}).get(field), actual.get("effect_tts_minus_natural", {}).get(field)):
                    raise CheckFailure(f"Stage A bootstrap mismatch {model_key}.{field}")
        return {"stage": "a", "models": recomputed, "eligible_pairs": len(eligible)}
    raise CheckFailure(f"unsupported independent statistics stage: {stage}")


def verify_waveform_contract(run_dir: Path) -> dict[str, Any]:
    stage_a = dict(read_json(run_dir / "03_stage_a" / "decision.json"))
    if stage_a.get("science_decision") != "ADVANTAGE_SUPPORTED":
        return {"checked": 0, "status": "NOT_APPLICABLE"}
    cohort = dict(read_json(run_dir / "00_audit" / "cohort.json"))
    row_by_id = {str(row["sample_id"]): row for row in cohort.get("records", [])}
    artifact_path = run_dir / "04_rules" / "artifacts.jsonl"
    artifacts = _read_jsonl(artifact_path)
    failures: list[str] = []
    checked = 0
    rules_cfg = {
        "sample_rate": 16_000,
        "edge_guard_s": 0.010,
        "taper_s": 0.005,
    }
    for artifact in artifacts:
        sample_id = str(artifact["sample_id"])
        arm = str(artifact["arm"])
        row = row_by_id.get(sample_id)
        if row is None:
            failures.append(f"UNKNOWN_SAMPLE:{sample_id}")
            continue
        source, source_meta = read_pcm16(Path(str(row["natural"]["path"])))
        output_path = Path(str(artifact["output"]["path"]))
        if not output_path.is_file():
            failures.append(f"MISSING_OUTPUT:{sample_id}:{arm}")
            continue
        output, output_meta = read_pcm16(output_path)
        if output_meta["container_sha256"] != artifact["output"].get("container_sha256"):
            failures.append(f"OUTPUT_HASH_MISMATCH:{sample_id}:{arm}")
        if source_meta["container_sha256"] != artifact.get("parent_natural_sha256"):
            failures.append(f"PARENT_HASH_MISMATCH:{sample_id}:{arm}")
        tokens = row["natural"].get("tokens", [])
        mask = build_edit_mask(source.size, tokens, **rules_cfg)
        if not np.array_equal(source[mask <= 0.0], output[mask <= 0.0]):
            failures.append(f"PAUSE_PCM_CHANGED:{sample_id}:{arm}")
        if arm == "identity" and not np.array_equal(source, output):
            failures.append(f"IDENTITY_CHANGED:{sample_id}")
        checked += 1
    if failures:
        raise CheckFailure("waveform contract failures: " + ", ".join(failures[:10]))
    return {"checked": checked, "status": "PASS"}


def check_run(run_dir: Path) -> dict[str, Any]:
    graph = verify_artifact_graph(run_dir)
    stage_a = recompute_statistics(run_dir, stage="a")
    waveform = verify_waveform_contract(run_dir)
    result = {
        "schema_version": 1,
        "engineering_pass": True,
        "science_decision": graph["stage_b_decision"].get("science_decision", graph["stage_b_decision"].get("status")),
        "artifact_graph": {"stage_a_passed": graph["stage_a_passed"]},
        "recomputed": stage_a,
        "waveform": waveform,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = check_run(Path(args.run_dir).resolve())
    except (CheckFailure, OSError, ValueError, KeyError) as exc:
        print(f"INDEPENDENT_CHECK_FAILED: {exc}", file=sys.stderr)
        return 3
    output = Path(args.run_dir).resolve() / "07_report" / "checker.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["check_run", "main", "recompute_statistics", "verify_artifact_graph", "verify_waveform_contract"]
