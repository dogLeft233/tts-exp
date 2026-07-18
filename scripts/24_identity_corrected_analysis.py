#!/usr/bin/env python3
"""24_identity_corrected_analysis.py — Reanalyse dose-response with the identity baseline.

The Phase 2 dose-response (script 22) found that all 4 TTS-source acoustic
interventions drop Sync-C by ≈ −1.0 to −1.2. Script 23 (identity control)
subsequently established that re-running the same TTS audio through the
write/Ditto/SyncNet pipeline *without any modification* also costs ≈ −1.16
Sync-C. The dose-response drops must therefore be reinterpreted as a
convolution of (a) Ditto run-to-run non-determinism and (b) any true
feature-specific effect.

This script implements the per-sample paired analysis:

    residual[intervention][sample] = Δ_sync_c[intervention][sample]
                                     − Δ_sync_c[identity][sample]

and tests whether the residuals for each TTS-source intervention are
significantly different from zero (paired t-test).  A non-significant
residual means the intervention's observed Sync-C drop is fully explained
by pipeline noise.

Natural-source interventions are reported with their raw deltas (no
identity subtraction is applicable because the identity control is TTS-side
only — a parallel natural identity would need to be run to support that
subtraction).

Inputs (read from data/wav2sem_analysis/metrics/):
    dose_response.json
    identity_control.json

Output (written to the same dir):
    identity_corrected_summary.json

Usage:
    python scripts/24_identity_corrected_analysis.py
    python scripts/24_identity_corrected_analysis.py --dose /path/to/dose_response.json --identity /path/to/identity_control.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DOSE = _REPO / "data" / "wav2sem_analysis" / "metrics" / "dose_response.json"
_DEFAULT_IDENT = _REPO / "data" / "wav2sem_analysis" / "metrics" / "identity_control.json"
_DEFAULT_OUT = _REPO / "data" / "wav2sem_analysis" / "metrics" / "identity_corrected_summary.json"

TTS_SOURCE = (
    "LUFS_match_tts_to_nat",
    "spectral_tilt_match_tts_to_nat",
    "dynamic_compression_tts_compress",
    "dynamic_expansion_tts_expand",
)
NAT_SOURCE = (
    "LUFS_match_nat_to_tts",
    "spectral_tilt_match_nat_to_tts",
    "dynamic_compression_nat_compress",
    "dynamic_expansion_nat_expand",
)


def load_inputs(dose_path: Path, ident_path: Path) -> tuple[dict, dict]:
    if not dose_path.exists():
        raise FileNotFoundError(f"dose_response.json not found: {dose_path}")
    if not ident_path.exists():
        raise FileNotFoundError(f"identity_control.json not found: {ident_path}")
    return json.loads(dose_path.read_text()), json.loads(ident_path.read_text())


def get_identity_per_sample(identity_data: dict) -> dict[int, dict]:
    """Return {sample_id: per-sample record} for the identity intervention."""
    iv = identity_data.get("interventions", {}).get("identity_tts")
    if not iv:
        raise ValueError("identity_control.json missing identity_tts intervention")
    out: dict[int, dict] = {}
    for sid_str, rec in iv.get("per_sample", {}).items():
        out[int(sid_str)] = rec
    return out


def compute_residuals(
    intervention_per_sample: dict[str, dict],
    identity_per_sample: dict[int, dict],
) -> dict:
    """For one intervention, compute per-sample residuals & summary stats.

    The residual for sample s is:
        residual[s] = Δ_intervention[s] − Δ_identity[s]

    where Δ is the Sync-C delta vs the original tts_raw diagonal baseline.

    Returns a dict containing per-sample residuals, mean, std, t, p, and
    Cohen's d (using the residual SD as the denominator).
    """
    residuals: list[float] = []
    per_sample: dict[int, dict] = {}
    for sid_str, rec in intervention_per_sample.items():
        sid = int(sid_str)
        base = identity_per_sample.get(sid)
        if base is None:
            continue  # sample missing from identity run (e.g., sample 9)
        if "delta_c" not in rec or "delta_c" not in base:
            continue
        r = float(rec["delta_c"]) - float(base["delta_c"])
        residuals.append(r)
        per_sample[sid] = {
            "intervention_delta_c": rec["delta_c"],
            "identity_delta_c": base["delta_c"],
            "residual_c": r,
        }
    n = len(residuals)
    if n == 0:
        return {
            "n_samples": 0,
            "mean_residual_c": None,
            "std_residual_c": None,
            "t_stat": None,
            "p_value": None,
            "cohens_d": None,
            "per_sample": {},
        }
    arr = np.asarray(residuals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    t_stat: float | None = None
    p_val: float | None = None
    if n > 1 and std > 0:
        # Pair = n samples, mean residues t-distributed with df = n-1
        se = std / float(np.sqrt(n))
        t_stat = mean / se if se > 0 else float("nan")
        p_val = 2.0 * float(stats.t.sf(abs(t_stat), df=n - 1))  # type: ignore[arg-type]
    else:
        t_stat, p_val = float("nan"), float("nan")
    cohens_d = mean / std if std > 0 else None
    return {
        "n_samples": n,
        "mean_residual_c": mean,
        "std_residual_c": std,
        "t_stat": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
        "per_sample": per_sample,
    }


def classify(p_value: float | None, alpha: float = 0.05) -> str:
    if p_value is None:
        return "no_data"
    return "significant" if p_value < alpha else "not_significant"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Identity-corrected dose-response residual analysis"
    )
    ap.add_argument("--dose", type=Path, default=_DEFAULT_DOSE)
    ap.add_argument("--identity", type=Path, default=_DEFAULT_IDENT)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    dose, ident = load_inputs(args.dose, args.identity)
    id_per_sample = get_identity_per_sample(ident)

    # Identity baseline summary
    id_deltas = np.array([v["delta_c"] for v in id_per_sample.values()])
    id_summary = {
        "n_samples": int(len(id_deltas)),
        "mean_delta_c": float(id_deltas.mean()) if len(id_deltas) else None,
        "std_delta_c": float(id_deltas.std(ddof=1)) if len(id_deltas) > 1 else None,
        "verdict": ident.get("verdict"),
        "verdict_message": ident.get("verdict_message"),
        "samples": sorted(id_per_sample.keys()),
    }

    # Per-intervention residual analysis (TTS-source)
    tts_residuals: dict[str, dict] = {}
    for name in TTS_SOURCE:
        iv = dose.get("interventions", {}).get(name)
        if not iv:
            continue
        res = compute_residuals(iv["per_sample"], id_per_sample)
        res["source"] = "tts"
        res["description"] = iv.get("description")
        res["expected_sync_direction"] = iv.get("expected_sync_direction")
        res["significance_class"] = classify(res["p_value"], args.alpha)
        tts_residuals[name] = res

    # Natural-source interventions: report raw deltas (no identity subtraction)
    nat_residuals: dict[str, dict] = {}
    for name in NAT_SOURCE:
        iv = dose.get("interventions", {}).get(name)
        if not iv:
            continue
        nat_residuals[name] = {
            "source": "natural",
            "description": iv.get("description"),
            "expected_sync_direction": iv.get("expected_sync_direction"),
            "n_samples": iv.get("n_samples"),
            "raw_mean_delta_c": iv.get("mean_delta_c"),
            "raw_std_delta_c": iv.get("std_delta_c"),
            "note": (
                "No identity subtraction applied: identity control is TTS-side only."
            ),
        }

    out = {
        "identity_baseline": id_summary,
        "tts_source_residuals": tts_residuals,
        "natural_source_raw": nat_residuals,
        "alpha": args.alpha,
        "interpretation": (
            "Each TTS-source intervention's residual = Δ_intervention − Δ_identity, "
            "per sample. A statistically non-significant residual (p ≥ alpha) means "
            "the intervention's observed Sync-C drop is fully explained by Ditto "
            "run-to-run non-determinism, not by the acoustic feature manipulation."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"[identity-corrected] written: {args.output}")

    # Console summary
    print("\n=== Identity-corrected dose-response ===")
    id_s = id_summary
    print(
        f"Identity baseline (no-op): n={id_s['n_samples']}, "
        f"ΔSync-C={id_s['mean_delta_c']:+.3f} ± {id_s['std_delta_c']:.3f}"
    )
    print(f"\nTTS-source interventions (paired residuals ≈ feature-specific effect):")
    print(f"  {'intervention':<36} n   {'net ΔC':>8}  {'±SD':>6}  {'t':>7}  {'p':>7}  {'d':>6}  verdict")
    for name in TTS_SOURCE:
        r = tts_residuals.get(name)
        if not r or r["n_samples"] == 0:
            continue
        d_str = f"{r['cohens_d']:+6.2f}" if r["cohens_d"] is not None else f"{'n/a':>6}"
        print(
            f"  {name:<36} {r['n_samples']}  "
            f"{r['mean_residual_c']:+8.3f}  {r['std_residual_c']:6.3f}  "
            f"{r['t_stat']:+7.3f}  {r['p_value']:7.4f}  "
            f"{d_str}  {r['significance_class']}"
        )
    print(f"\nNatural-source interventions (raw Δ vs natural baseline; no identity subtraction):")
    for name in NAT_SOURCE:
        r = nat_residuals.get(name)
        if not r:
            continue
        print(
            f"  {name:<36} {r['n_samples'] or 0}  "
            f"raw ΔC={r['raw_mean_delta_c']:+.3f}  ± {r['raw_std_delta_c'] or 0:.3f}"
        )


if __name__ == "__main__":
    main()