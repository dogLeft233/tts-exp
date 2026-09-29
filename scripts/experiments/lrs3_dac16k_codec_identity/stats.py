from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import config
from .common import file_sha256, write_self_hashed_json
from .protocol import ProtocolError


def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if len(values) != len(groups) or not values:
        raise ValueError("bootstrap inputs are empty or have different lengths")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        by_group[str(group)].append(value)
    labels = sorted(by_group)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        means[index] = np.concatenate([np.asarray(by_group[label], dtype=np.float64) for label in sampled]).mean()
    ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "field": None,
        "draws": int(draws),
        "seed": int(seed),
        "source_group_count": len(labels),
        "record_count": len(values),
        "mean": float(np.mean(values)),
        "ci95": ci,
        "lower_bound_strictly_positive": bool(ci[0] > 0.0),
        "group_means": {label: float(np.mean(by_group[label])) for label in labels},
    }


def endpoint(values: Sequence[float], groups: Sequence[str], name: str) -> dict[str, Any]:
    result = cluster_bootstrap(values, groups, seed=config.BOOTSTRAP_SEED)
    result["field"] = name
    result["lower_bound_strictly_positive"] = bool(result["ci95"][0] > 0.0)
    return result


def _wins(values: Sequence[float]) -> dict[str, int]:
    positive = [float(value) > 0.0 for value in values]
    return {"positive_count": int(sum(positive)), "record_count": len(values)}


def _rows_by_key(matrix: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = matrix.get("scores", [])
    expected = config.EXPECTED_RECORD_COUNT * len(config.MATRIX_CELLS)
    if matrix.get("status") != "complete" or len(rows) != expected:
        raise ProtocolError(f"matrix is incomplete: {len(rows)}/{expected}")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("cell")))
        if key in result:
            raise ProtocolError(f"duplicate matrix cell: {key}")
        result[key] = row
    if len(result) != expected:
        raise ProtocolError("matrix cell identity is incomplete")
    return result


def run_stage05(cohort: Mapping[str, Any], fidelity: Mapping[str, Any], matrix: Mapping[str, Any], output_stage: Any) -> dict[str, Any]:
    by_key = _rows_by_key(matrix)
    fidelity_rows = fidelity.get("rows", [])
    if fidelity.get("status") != "complete" or len(fidelity_rows) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("fidelity stage is incomplete")
    fidelity_by_id = {str(row["sample_id"]): row for row in fidelity_rows}
    if len(fidelity_by_id) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("fidelity row identity is incomplete")

    endpoints: dict[str, list[float]] = {name: [] for name in ("DAC_over_W_C", "DAC_over_W_D", "DAC_identity_C", "DAC_identity_D", "mel_improvement")}
    groups: dict[str, list[str]] = {name: [] for name in endpoints}
    per_record: list[dict[str, Any]] = []
    descriptive: list[dict[str, Any]] = []
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        natural = by_key[(sample_id, "V_N/A_N")]
        w_replacement = by_key[(sample_id, "V_W/A_N")]
        d_replacement = by_key[(sample_id, "V_D/A_N")]
        frow = fidelity_by_id[sample_id]
        values = {
            "DAC_over_W_C": float(d_replacement["sync_c"]) - float(w_replacement["sync_c"]),
            "DAC_over_W_D": float(w_replacement["sync_d"]) - float(d_replacement["sync_d"]),
            "DAC_identity_C": float(d_replacement["sync_c"]) - float(natural["sync_c"]),
            "DAC_identity_D": float(natural["sync_d"]) - float(d_replacement["sync_d"]),
            "mel_improvement": float(frow["mel_improvement"]),
        }
        for name, value in values.items():
            endpoints[name].append(value)
            groups[name].append(group)
        per_record.append({"sample_id": sample_id, "source_group": group, **values})
        descriptive.append({
            "sample_id": sample_id,
            "source_group": group,
            "diagonal_N": {"sync_c": float(natural["sync_c"]), "sync_d": float(natural["sync_d"])},
            "diagonal_W": {"sync_c": float(by_key[(sample_id, "V_W/A_W")]["sync_c"]), "sync_d": float(by_key[(sample_id, "V_W/A_W")]["sync_d"])},
            "diagonal_D": {"sync_c": float(by_key[(sample_id, "V_D/A_D")]["sync_c"]), "sync_d": float(by_key[(sample_id, "V_D/A_D")]["sync_d"])},
            "W_own_preference_C": float(by_key[(sample_id, "V_W/A_W")]["sync_c"]) - float(by_key[(sample_id, "V_W/A_N")]["sync_c"]),
            "W_own_preference_D": float(by_key[(sample_id, "V_W/A_N")]["sync_d"]) - float(by_key[(sample_id, "V_W/A_W")]["sync_d"]),
            "D_own_preference_C": float(by_key[(sample_id, "V_D/A_D")]["sync_c"]) - float(by_key[(sample_id, "V_D/A_N")]["sync_c"]),
            "D_own_preference_D": float(by_key[(sample_id, "V_D/A_N")]["sync_d"]) - float(by_key[(sample_id, "V_D/A_D")]["sync_d"]),
        })
    stats = {name: endpoint(endpoints[name], groups[name], name) for name in endpoints}
    improvement_pass = all(stats[name]["ci95"][0] > 0.0 for name in ("DAC_over_W_C", "DAC_over_W_D", "mel_improvement"))
    identity_pass = improvement_pass and all(stats[name]["ci95"][0] > config.IDENTITY_MARGIN for name in ("DAC_identity_C", "DAC_identity_D"))
    if identity_pass:
        decision = "DAC_CODEC_IDENTITY_COMPATIBLE"
    elif improvement_pass:
        decision = "DAC_CODEC_PARTIAL_IMPROVEMENT"
    else:
        decision = "DAC_CODEC_HYPOTHESIS_NOT_SUPPORTED"
    analysis = {
        "schema_version": 1,
        "stage_id": "05_analysis",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": len(per_record),
        "source_group_count": len({row["source_group"] for row in per_record}),
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "unit": "source_group_cluster"},
        "endpoints": stats,
        "wins": {name: _wins(endpoints[name]) for name in endpoints},
        "decisions": {
            "DAC_REDUCES_CODEC_REPLACEMENT_PENALTY": improvement_pass,
            "DAC_IDENTITY_COMPATIBLE": identity_pass,
            "scientific_decision": decision,
            "identity_non_inferiority_margin": config.IDENTITY_MARGIN,
            "future_tts_alignment_experiment_eligible": identity_pass,
        },
        "per_record": per_record,
        "descriptive_diagnostics": descriptive,
        "matrix_manifest_sha256": file_sha256(config.STAGES["04_matrix"] / "matrix_manifest.json"),
        "fidelity_sha256": file_sha256(config.STAGES["02_fidelity"] / "fidelity.json"),
    }
    write_self_hashed_json(output_stage / "analysis.json", analysis)
    return analysis
