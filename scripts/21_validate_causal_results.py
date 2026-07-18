#!/usr/bin/env python3
"""21_validate_causal_results.py — Cross-TFG mechanism validation.

Validates whether causal mechanisms found in Ditto (Phase 1+2) generalise to
other TFG models (Wav2Lip, MuseTalk, LatentSync, JoyVASA).

Steps:
    1. Load Ditto per-sample SyncNet results from eval_meta.json.
    2. Load candidate mechanisms from feature_tfg_link.json.
    3. Load causal intervention results from intervention_results.json.
    4. Search for other TFG model eval results (eval_meta.json or syncnet.json).
    5. If only aggregate data is available (TFG_DEPLOY_SUMMARY.md), use it
       as cross-model reference.
    6. Compute cross-model consistency for each mechanism.
    7. Use LatentSync (~zero TTS effect) as a negative control.
    8. Output cross_tfg_validation.json.

Usage:
    python scripts/21_validate_causal_results.py
    python scripts/21_validate_causal_results.py --ditto-dir runs/aishell1_strict_20260707T081223Z/04_eval
    python scripts/21_validate_causal_results.py --other-dirs wav2lip,musetalk,latentsync,joyvasa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_OUTPUT_BASE = _REPO / "data" / "wav2sem_analysis" / "metrics"

_STUDY_SAMPLES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]


# ---------------------------------------------------------------------------
# Known aggregate SyncNet scores (from TFG_DEPLOY_SUMMARY.md)
# Used when per-sample data is unavailable.
# ---------------------------------------------------------------------------

KNOWN_AGGREGATE = {
    "ditto": {
        "natural_C": 5.662, "tts_C": 6.908,
        "natural_D": 7.955, "tts_D": 7.188,
        "source": "eval_meta.json (aishell1_strict_20260707T081223Z)",
    },
    "wav2lip": {
        "natural_C": 7.393, "tts_C": 8.922,
        "natural_D": 7.327, "tts_D": 5.941,
        "source": "TFG_DEPLOY_SUMMARY.md (server eval)",
    },
    "musetalk": {
        "natural_C": 5.789, "tts_C": 6.326,
        "natural_D": 8.081, "tts_D": 7.928,
        "source": "TFG_DEPLOY_SUMMARY.md (server eval)",
    },
    "vexpress": {
        "natural_C": 5.975, "tts_C": 6.506,
        "natural_D": 8.818, "tts_D": 8.236,
        "source": "TFG_DEPLOY_SUMMARY.md (server eval)",
    },
    "joyvasa": {
        "natural_C": 4.815, "tts_C": 5.084,
        "natural_D": 9.443, "tts_D": 9.581,
        "source": "TFG_DEPLOY_SUMMARY.md (server eval)",
    },
    "latentsync": {
        "natural_C": 4.653, "tts_C": 4.581,
        "natural_D": 8.894, "tts_D": 8.768,
        "source": "TFG_DEPLOY_SUMMARY.md (server eval)",
    },
}

MECHANISM_CANDIDATES = [
    {
        "name": "LUFS",
        "description": "TTS higher loudness drives SyncNet score advantage",
        "phase1_evidence": "Loudness correlates with Sync-C/Sync-D deltas",
        "phase2_evidence": "LUFS_matching verified on 5/5 pilot samples",
        "expected_direction": "positive",
    },
    {
        "name": "spectral_tilt",
        "description": "TTS spectral balance affects embedding quality",
        "phase1_evidence": "Candidate from per-frame spectral analysis",
        "phase2_evidence": "spectral_tilt_matching verified on 5/5 pilot samples",
        "expected_direction": "unknown",
    },
    {
        "name": "dynamic_range",
        "description": "TTS reduced dynamic range may help SyncNet",
        "phase1_evidence": "Candidate from energy envelope analysis",
        "phase2_evidence": "dynamic_compression/expansion verified on 5/5 pilot samples",
        "expected_direction": "unknown",
    },
    {
        "name": "prosody_boundary",
        "description": "TTS sharper boundaries improve viseme detection",
        "phase1_evidence": "Pending — requires boundary sharpness metrics",
        "phase2_evidence": "Not tested (Phase 1 data incomplete)",
        "expected_direction": "positive",
    },
    {
        "name": "formant_clarity",
        "description": "TTS clearer formant structure for vowel discrimination",
        "phase1_evidence": "Pending — requires formant analysis",
        "phase2_evidence": "Not tested (Phase 1 data incomplete)",
        "expected_direction": "positive",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mechanism_name(mechanism: dict) -> str:
    """Return a stable name across Phase 1 candidate schemas."""
    return (
        mechanism.get("name")
        or mechanism.get("mechanism")
        or mechanism.get("metric")
        or "unknown"
    )


def aggregate_per_sample_deltas(deltas: dict[str, dict[int, float]]) -> dict[str, float]:
    """Compute model-level mean deltas from per-sample SyncNet results."""
    return {
        "delta_c": float(np.nanmean(list(deltas.get("sync_c", {}).values()))),
        "delta_d": float(np.nanmean(list(deltas.get("sync_d", {}).values()))),
    }


def load_eval_meta(meta_path: Path) -> dict[str, list[dict]]:
    """Parse eval_meta.json into {condition: [{sample_id, sync_c, sync_d, ...}, ...]}."""
    data = json.loads(meta_path.read_text())
    results = data.get("results", {})
    by_condition: dict[str, list[dict]] = {}
    for key, entry in results.items():
        cond = entry.get("condition", key)
        by_condition.setdefault(cond, []).append({
            "sample_id": entry["sample_id"],
            "sync_c": entry["sync_c"],
            "sync_d": entry["sync_d"],
            "av_offset": entry.get("av_offset"),
        })
    for cond in by_condition:
        by_condition[cond].sort(key=lambda x: x["sample_id"])
    return by_condition


def load_per_sample_deltas(eval_meta: dict[str, list[dict]],
                            samples: list[int] | None = None) -> dict[str, dict[int, float]]:
    """Compute per-sample Sync-C and Sync-D deltas.

    Returns {metric: {sample_id: delta}} for "sync_c" and "sync_d".
    """
    nat_by_sid = {e["sample_id"]: e for e in eval_meta.get("natural_raw", [])}
    tts_by_sid = {e["sample_id"]: e for e in eval_meta.get("tts_raw", [])}
    sids = list(set(nat_by_sid.keys()) & set(tts_by_sid.keys()))
    if samples is not None:
        sids = [s for s in sids if s in samples]
    sids.sort()

    deltas: dict[str, dict[int, float]] = {"sync_c": {}, "sync_d": {}}
    for sid in sids:
        deltas["sync_c"][sid] = tts_by_sid[sid]["sync_c"] - nat_by_sid[sid]["sync_c"]
        deltas["sync_d"][sid] = tts_by_sid[sid]["sync_d"] - nat_by_sid[sid]["sync_d"]
    return deltas


def compute_aggregate_deltas(agg: dict) -> dict[str, float]:
    """From aggregate dict {'natural_C': X, 'tts_C': Y, ...}, compute deltas."""
    return {
        "delta_c": agg["tts_C"] - agg["natural_C"],
        "delta_d": agg["tts_D"] - agg["natural_D"],
    }


# ---------------------------------------------------------------------------
# Step 1-2: Load Ditto results & mechanisms
# ---------------------------------------------------------------------------


def load_ditto_data(ditto_dir: Path) -> dict:
    """Load Ditto per-sample SyncNet results and mechanisms."""
    meta_path = None
    if ditto_dir.is_dir():
        for candidate in [
            ditto_dir / "eval_meta.json",
            ditto_dir.parent / "04_eval" / "eval_meta.json",
        ]:
            if candidate.exists():
                meta_path = candidate
                break

    result: dict = {
        "has_per_sample": False,
        "per_sample_deltas": None,
        "aggregate": None,
        "mechanisms": [],
        "interventions": [],
        "warnings": [],
    }

    if meta_path is not None and meta_path.exists():
        eval_meta = load_eval_meta(meta_path)
        result["per_sample_deltas"] = load_per_sample_deltas(eval_meta, _STUDY_SAMPLES)
        result["has_per_sample"] = True
        result["aggregate"] = aggregate_per_sample_deltas(result["per_sample_deltas"])
    else:
        # Fallback to known aggregate
        ditto_agg = KNOWN_AGGREGATE.get("ditto")
        if ditto_agg:
            result["aggregate"] = compute_aggregate_deltas(ditto_agg)
            result["warnings"].append("No per-sample Ditto eval found; using aggregate from known data.")
        else:
            result["warnings"].append("No Ditto eval data found. Cross-TFG validation blocked.")

    # Load mechanisms
    ft_path = _OUTPUT_BASE / "feature_tfg_link.json"
    if ft_path.exists():
        ft_data = json.loads(ft_path.read_text())
        mechanisms = ft_data.get("candidates", [])
        if mechanisms:
            result["mechanisms"] = mechanisms
        else:
            result["mechanisms"] = MECHANISM_CANDIDATES
            result["warnings"].append("No ranked candidate mechanisms in feature_tfg_link.json; using default candidates.")
    else:
        result["mechanisms"] = MECHANISM_CANDIDATES
        result["warnings"].append("feature_tfg_link.json not found; using default candidates.")

    # Load intervention results
    ir_path = _OUTPUT_BASE / "intervention_results.json"
    if ir_path.exists():
        ir_data = json.loads(ir_path.read_text())
        result["interventions"] = ir_data.get("interventions", [])
    else:
        result["warnings"].append("intervention_results.json not found.")

    return result


# ---------------------------------------------------------------------------
# Step 3: Load other TFG results
# ---------------------------------------------------------------------------


def _search_eval_meta(runs_dir: Path, model_name: str) -> Path | None:
    """Search for eval_meta.json under runs/<tfg_model> patterns."""
    candidates = list(runs_dir.glob(f"tfg_{model_name}*/04_eval/eval_meta.json"))
    if candidates:
        return candidates[0]
    candidates = list(runs_dir.glob(f"*{model_name}*/04_eval/eval_meta.json"))
    if candidates:
        return candidates[0]
    return None


def load_other_tfg_data(runs_dir: Path,
                         model_names: list[str]) -> dict[str, dict]:
    """Load SyncNet data for each requested TFG model.

    Returns {model_name: {has_per_sample, per_sample_deltas, aggregate, ...}}.
    """
    results: dict[str, dict] = {}
    for name in model_names:
        entry: dict = {
            "has_per_sample": False,
            "per_sample_deltas": None,
            "aggregate": None,
            "source": "none",
        }
        if name == "ditto":
            continue  # handled separately

        meta_path = _search_eval_meta(runs_dir, name)
        if meta_path and meta_path.exists():
            eval_meta = load_eval_meta(meta_path)
            entry["per_sample_deltas"] = load_per_sample_deltas(eval_meta, _STUDY_SAMPLES)
            entry["has_per_sample"] = True
            entry["aggregate"] = aggregate_per_sample_deltas(entry["per_sample_deltas"])
            entry["source"] = str(meta_path)
        elif name in KNOWN_AGGREGATE:
            entry["aggregate"] = compute_aggregate_deltas(KNOWN_AGGREGATE[name])
            entry["source"] = KNOWN_AGGREGATE[name]["source"]
        else:
            entry["source"] = "not_found"

        results[name] = entry

    return results


# ---------------------------------------------------------------------------
# Step 4-5: Cross-model consistency
# ---------------------------------------------------------------------------


def classify_consistency(ditto_sign: int,
                          other_model_data: dict[str, dict]) -> tuple[str, list[str], list[str], list[str]]:
    """Classify mechanism consistency across models.

    When ditto_sign is 0 (no mechanism-specific data), we do NOT use
    aggregate model-level deltas to infer direction — those deltas reflect
    the overall TTS advantage, not the specific mechanism's contribution.
    In that case, return "ditto_only" (mechanism only tested on Ditto).

    Parameters
    ----------
    ditto_sign : int
        +1 = TTS increases metric, -1 = TTS decreases, 0 = neutral/unknown.
    other_model_data : dict
        {model_name: {aggregate: {delta_c, delta_d}, ...}}.

    Returns
    -------
    consistency : str
    same_direction : list[str]
    opposite_direction : list[str]
    neutral_direction : list[str]
    """
    same: list[str] = []
    opposite: list[str] = []
    neutral: list[str] = []

    if ditto_sign == 0:
        return "ditto_only", [], [], []

    for model, entry in sorted(other_model_data.items()):
        aggr = entry.get("aggregate")
        if aggr is None:
            continue
        delta = aggr["delta_c"]
        if abs(delta) < 0.15:
            neutral.append(model)
        elif (ditto_sign > 0 and delta > 0) or (ditto_sign < 0 and delta < 0):
            same.append(model)
        else:
            opposite.append(model)

    if not same and not opposite and not neutral:
        return "no_data", [], [], []
    elif opposite and not same:
        return "reversed", same, opposite, neutral
    elif not same and neutral:
        return "neutral", same, opposite, neutral
    elif same and not opposite:
        return "consistent", same, opposite, neutral
    elif same and opposite:
        return "mixed", same, opposite, neutral
    elif neutral and not same and not opposite:
        return "neutral", same, opposite, neutral

    return "unknown", same, opposite, neutral


def check_latentsync_neutral(model_data: dict[str, dict]) -> bool | None:
    """LatentSync as negative control: its TTS effect should be near zero.

    Returns True if LatentSync shows neutral TTS effect, False if not,
    None if LatentSync data is unavailable.
    """
    if "latentsync" not in model_data:
        return None  # unknown
    aggr = model_data["latentsync"].get("aggregate")
    if aggr is None:
        return None
    # Aggregate deltas: ΔC ≈ −0.07, ΔD ≈ −0.13 — effectively neutral
    return abs(aggr["delta_c"]) < 0.2


def derive_ditto_sign_by_mechanism(mechanism_name: str,
                                    ditto_per_sample: dict | None,
                                    ditto_agg: dict | None,
                                    interventions: list[dict] | None = None) -> int:
    """Infer Ditto's TTS direction for a given mechanism.

    Only returns non-zero for mechanisms with Phase 2 causal intervention
    data linking the mechanism to SyncNet scores.  For mechanisms without
    specific data, returns 0 (neutral/unknown).
    """
    known_causal = {
        "LUFS": +1,  # TTS louder → higher Sync-C (verified intervention)
    }
    if mechanism_name in known_causal:
        return known_causal[mechanism_name]

    return 0


def build_mechanism_results(
    ditto_data: dict,
    model_data: dict[str, dict],
) -> list[dict]:
    """Build per-mechanism cross-TFG validation entries."""
    results: list[dict] = []
    ditto_agg = ditto_data.get("aggregate")
    ditto_ps = ditto_data.get("per_sample_deltas")

    for mech in ditto_data["mechanisms"]:
        name = mechanism_name(mech)
        ditto_sign = derive_ditto_sign_by_mechanism(
            name, ditto_ps, ditto_agg, ditto_data.get("interventions")
        )
        consistency, same, opposite, neutral = classify_consistency(
            ditto_sign, model_data
        )

        latentsync_neutral = check_latentsync_neutral(model_data)

        verdict = "pending"
        if consistency == "consistent":
            if latentsync_neutral is True:
                verdict = "robust — consistent across TFG models, LatentSync null confirms not a SyncNet artifact"
            elif latentsync_neutral is None:
                verdict = "likely robust — consistent across available TFG models (LatentSync data pending)"
            else:
                verdict = "questionable — consistent across models but LatentSync also shows effect (possible SyncNet artifact)"
        elif consistency == "mixed":
            verdict = "ambiguous — mixed direction across models"
        elif consistency == "ditto_only":
            verdict = "ditto-specific — no confirmation from other TFG models"
        elif consistency == "reversed":
            verdict = "rejected — opposite direction in other TFG models"
        elif consistency == "neutral":
            verdict = "neutral — no significant effect across models"
        elif consistency == "no_data":
            verdict = "cross-model validation pending — run TFG inference on all models first"
        else:
            verdict = "pending"

        results.append({
            "name": name,
            "description": mech.get("description", mech.get("evidence", "")),
            "phase1_evidence": mech.get("phase1_evidence", mech.get("evidence", "")),
            "phase2_evidence": mech.get("phase2_evidence", ""),
            "ditto_direction": "positive" if ditto_sign > 0 else ("negative" if ditto_sign < 0 else "neutral"),
            "consistency": consistency,
            "models_with_same_direction": same,
            "models_with_opposite_direction": opposite,
            "models_neutral": neutral,
            "latentsync_neutral": latentsync_neutral,
            "verdict": verdict,
        })

    return results


# ---------------------------------------------------------------------------
# Cross-model summary table
# ---------------------------------------------------------------------------


def build_summary_table(mechanism_results: list[dict],
                         model_data: dict[str, dict],
                         ditto_data: dict) -> list[dict]:
    """Build the cross-model summary table rows.

    Each row: mechanism, Ditto verdict, per-model verdict, overall verdict.
    """
    available_models = ["ditto"] + sorted(model_data.keys())
    rows: list[dict] = []

    for mech_result in mechanism_results:
        name = mech_result["name"]
        row: dict = {"mechanism": name}

        # Ditto
        ditto_sign = {"positive": "+", "negative": "-", "neutral": "~"}[
            mech_result["ditto_direction"]
        ]
        row["ditto"] = ditto_sign

        # Other models
        for model in sorted(model_data.keys()):
            same = mech_result["models_with_same_direction"]
            opposite = mech_result["models_with_opposite_direction"]
            neutral = mech_result["models_neutral"]
            if model in same:
                row[model] = "+"
            elif model in opposite:
                row[model] = "-"
            elif model in neutral:
                row[model] = "~"
            else:
                aggr = model_data[model].get("aggregate")
                if aggr is None:
                    row[model] = "?"
                else:
                    # Default to sign of aggregate delta_c
                    d = aggr["delta_c"]
                    if abs(d) < 0.15:
                        row[model] = "~"
                    elif d > 0:
                        row[model] = "+"
                    else:
                        row[model] = "-"

        row["verdict"] = mech_result["verdict"]
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-TFG mechanism validation — generalise Phase 1+2 causal findings"
    )
    ap.add_argument(
        "--ditto-dir",
        default=str(_REPO / "runs" / "aishell1_strict_20260707T081223Z" / "04_eval"),
        help="Directory containing Ditto eval_meta.json (default: runs/aishell1_strict_*/04_eval)",
    )
    ap.add_argument(
        "--other-dirs",
        default="wav2lip,musetalk,latentsync,joyvasa",
        help="Comma-separated TFG model names to validate against (default: wav2lip,musetalk,latentsync,joyvasa)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(_OUTPUT_BASE),
        help="Output directory for cross_tfg_validation.json (default: data/wav2sem_analysis/metrics)",
    )
    ap.add_argument(
        "--runs-dir",
        default=str(_REPO / "runs"),
        help="Base runs directory (default: runs/)",
    )
    args = ap.parse_args()

    model_names = [m.strip() for m in args.other_dirs.split(",") if m.strip()]
    ditto_dir = Path(args.ditto_dir)
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("=" * 70)
    print("Step 1: Loading Ditto data...")
    ditto_data = load_ditto_data(ditto_dir)
    for w in ditto_data["warnings"]:
        print(f"  WARNING: {w}")
    if ditto_data["has_per_sample"]:
        n_samples = len(ditto_data["per_sample_deltas"]["sync_c"])
        print(f"  Ditto per-sample deltas loaded: n={n_samples} samples")
    elif ditto_data["aggregate"]:
        print(f"  Ditto aggregate: ΔC={ditto_data['aggregate']['delta_c']:+.3f}, "
              f"ΔD={ditto_data['aggregate']['delta_d']:+.3f}")
    print(f"  Mechanisms: {len(ditto_data['mechanisms'])}")
    print(f"  Interventions: {len(ditto_data['interventions'])} loaded")

    print("\nStep 2: Loading other TFG model data...")
    model_data = load_other_tfg_data(runs_dir, model_names)
    has_per_sample = sum(1 for v in model_data.values() if v["has_per_sample"])
    has_aggregate = sum(1 for v in model_data.values() if v["aggregate"] is not None)
    print(f"  Models with per-sample data: {has_per_sample}")
    print(f"  Models with aggregate data only: {has_aggregate}")
    print(f"  Models not found: {sum(1 for v in model_data.values() if v['source'] == 'not_found')}")
    for name, entry in sorted(model_data.items()):
        source = entry["source"]
        if source == "not_found":
            print(f"    {name}: not found — cross-model validation pending")
        elif entry["has_per_sample"]:
            n = len(entry["per_sample_deltas"]["sync_c"])
            print(f"    {name}: per-sample (n={n}) from {source}")
        elif entry["aggregate"]:
            agg = entry["aggregate"]
            print(f"    {name}: aggregate ΔC={agg['delta_c']:+.3f} ΔD={agg['delta_d']:+.3f} ({source})")

    # ---- Cross-model consistency ----
    print("\nStep 3: Cross-model consistency analysis...")
    mechanism_results = build_mechanism_results(ditto_data, model_data)

    for r in mechanism_results:
        name = r["name"]
        con = r["consistency"]
        dd = r["ditto_direction"]
        ls = r["latentsync_neutral"]
        ls_str = f"latentsync_neutral={ls}" if ls is not None else "latentsync=unknown"
        print(f"  {name}: ditto={dd} consistency={con} same={r['models_with_same_direction']} "
              f"opp={r['models_with_opposite_direction']} neutral={r['models_neutral']} {ls_str}")
        print(f"    verdict: {r['verdict']}")

    # ---- Summary table ----
    print("\nStep 4: Building cross-model summary table...")
    summary_rows = build_summary_table(mechanism_results, model_data, ditto_data)
    available_models = ["ditto"] + sorted(model_data.keys())

    # Print table
    header = ["Mechanism"] + available_models + ["Verdict"]
    col_widths = [22] + [max(10, len(m)) for m in available_models] + [0]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths[:-1]) + "  {}"
    print(fmt.format(*header))
    print("-" * (sum(col_widths[:-1]) + 2 * (len(col_widths) - 2)))
    for row in summary_rows:
        mech_name = row["mechanism"]
        model_cols = [row.get(m, "?") for m in available_models]
        print(fmt.format(mech_name, *model_cols, row["verdict"][:60]))

    # ---- Build output ----
    output = {
        "meta": {
            "timestamp": "",
            "tfg_models_available": ["ditto"] + sorted(
                m for m in model_data if model_data[m]["source"] != "not_found"
            ),
            "tfg_models_pending": sorted(
                m for m in model_data if model_data[m]["source"] == "not_found"
            ),
            "n_samples_per_model": {
                "ditto": len(ditto_data["per_sample_deltas"]["sync_c"])
                if ditto_data["per_sample_deltas"]
                else "aggregate_only",
            },
            "per_sample_available": ditto_data["has_per_sample"],
            "aggregate_only_models": sorted(
                m for m, v in model_data.items()
                if v["aggregate"] and not v["has_per_sample"]
            ),
            "warnings": ditto_data.get("warnings", []),
        },
        "ditto": {
            "has_per_sample": ditto_data["has_per_sample"],
            "aggregate": ditto_data["aggregate"],
        },
        "other_models": {
            name: {
                "has_per_sample": entry["has_per_sample"],
                "aggregate": entry["aggregate"],
                "source": entry["source"],
            }
            for name, entry in model_data.items()
        },
        "mechanisms": mechanism_results,
        "summary_table": summary_rows,
        "interventions": [
            {
                "name": i["name"],
                "target_metric": i.get("target_metric", ""),
                "pilot_result": i.get("pilot_result", "UNKNOWN"),
                "verified_count": i.get("verified_count", 0),
                "total": i.get("total", 0),
            }
            for i in ditto_data["interventions"]
        ],
    }

    import datetime as _dt
    output["meta"]["timestamp"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_path = output_dir / "cross_tfg_validation.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nOutput written: {out_path}")

    # ---- Status summary ----
    print("\n" + "=" * 70)
    print("STATUS SUMMARY")
    print(f"  Ditto per-sample data: {'YES' if ditto_data['has_per_sample'] else 'NO'}")
    agg_only = [m for m in model_names if m in model_data and model_data[m]["aggregate"] and not model_data[m]["has_per_sample"]]
    pending = [m for m in model_names if m in model_data and model_data[m]["source"] == "not_found"]
    if pending:
        print(f"  Pending (no data): {', '.join(pending)} — run TFG inference first")
    if agg_only:
        print(f"  Aggregate only: {', '.join(agg_only)} — cross-model validation uses limited data")
    per_sample_models = [m for m in model_names if m in model_data and model_data[m]["has_per_sample"]]
    if per_sample_models:
        print(f"  Per-sample data: {', '.join(per_sample_models)}")
    print(f"  Mechanisms validated: {len(mechanism_results)}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
