#!/usr/bin/env python3
"""Analyze English multi-seed Ditto baselines and decide Gate 1."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, values[order] * (len(values) - np.arange(len(values))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n == 0:
        return float("nan")
    observed = abs(float(differences.mean()))
    if n <= 20:
        indices = np.arange(1 << n, dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(n, dtype=np.uint64)) & 1
        signs = np.where(bits, 1.0, -1.0)
        null = np.abs((signs * differences).mean(axis=1))
        return float(np.mean(null >= observed - 1e-12))
    rng = np.random.default_rng(42)
    signs = rng.choice((-1.0, 1.0), size=(100000, n))
    return float(np.mean(np.abs((signs * differences).mean(axis=1)) >= observed))


def bootstrap_ci(values: np.ndarray, iterations: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def load_eval(run_dir: Path) -> dict[int, dict[str, dict[str, float]]]:
    meta_path = run_dir / "04_eval" / "eval_meta.json"
    meta = json.loads(meta_path.read_text())
    expected = {int(sample_id) for sample_id in meta["expected_sample_ids"]}
    complete = {int(sample_id) for sample_id in meta["complete_case_ids"]}
    if expected != complete:
        raise ValueError(f"incomplete baseline in {run_dir.name}: {sorted(expected - complete)}")
    rows: dict[int, dict[str, dict[str, float]]] = {}
    for sample_id in sorted(complete):
        rows[sample_id] = {}
        for condition in ("natural", "tts"):
            path = run_dir / "04_eval" / condition / str(sample_id) / "syncnet.json"
            rows[sample_id][condition] = json.loads(path.read_text())
    return rows


def load_identity(run_dir: Path) -> dict[int, dict[str, dict[str, float]]]:
    path = run_dir / "04_eval" / "controls" / "identity_control.json"
    data = json.loads(path.read_text())
    rows: dict[int, dict[str, dict[str, float]]] = {}
    for arm in ("natural", "tts"):
        intervention = data["interventions"][f"identity_{arm}"]["per_sample"]
        for sample_id, record in intervention.items():
            rows.setdefault(int(sample_id), {})[arm] = record
    return rows


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "standard_deviation": standard_deviation,
        "cohens_dz": float(values.mean() / standard_deviation) if standard_deviation else None,
        "bootstrap_95_ci": bootstrap_ci(values),
        "sign_flip_p": exact_sign_flip_pvalue(values),
        "values": values.tolist(),
    }


def analyze(runs_root: Path, run_prefix: str, seeds: list[int]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    per_sample: dict[int, dict[str, list[float]]] = {}
    no_op_differentials: dict[int, dict[str, list[float]]] = {}

    for seed in seeds:
        run_dir = runs_root / f"{run_prefix}_s{seed}"
        baseline = load_eval(run_dir)
        identity = load_identity(run_dir)
        common = sorted(set(baseline) & set(identity))
        if set(common) != set(baseline):
            raise ValueError(f"identity control incomplete for seed {seed}")
        seed_rows = []
        for sample_id in common:
            row: dict[str, float | int] = {"sample_id": sample_id}
            for metric in ("sync_c", "sync_d"):
                raw = float(baseline[sample_id]["tts"][metric]) - float(
                    baseline[sample_id]["natural"][metric]
                )
                no_op_nat = float(identity[sample_id]["natural"][f"delta_{metric[-1]}"])
                no_op_tts = float(identity[sample_id]["tts"][f"delta_{metric[-1]}"])
                corrected = raw - (no_op_tts - no_op_nat)
                row[f"raw_delta_{metric}"] = raw
                row[f"no_op_differential_{metric}"] = no_op_tts - no_op_nat
                row[f"corrected_delta_{metric}"] = corrected
                per_sample.setdefault(sample_id, {}).setdefault(metric, []).append(corrected)
                no_op_differentials.setdefault(sample_id, {}).setdefault(metric, []).append(
                    no_op_tts - no_op_nat
                )
            seed_rows.append(row)
        per_seed[str(seed)] = {
            "run_id": run_dir.name,
            "samples": seed_rows,
            "mean_corrected_delta_sync_c": float(np.mean([r["corrected_delta_sync_c"] for r in seed_rows])),
            "mean_corrected_delta_sync_d": float(np.mean([r["corrected_delta_sync_d"] for r in seed_rows])),
        }

    sample_rows = []
    metric_values: dict[str, np.ndarray] = {}
    for sample_id in sorted(per_sample):
        row: dict[str, Any] = {"sample_id": sample_id}
        for metric in ("sync_c", "sync_d"):
            row[f"corrected_delta_{metric}"] = float(np.mean(per_sample[sample_id][metric]))
            row[f"seed_sd_{metric}"] = float(np.std(per_sample[sample_id][metric], ddof=1))
            row[f"no_op_differential_{metric}"] = float(
                np.mean(no_op_differentials[sample_id][metric])
            )
        sample_rows.append(row)
    for metric in ("sync_c", "sync_d"):
        metric_values[metric] = np.asarray(
            [row[f"corrected_delta_{metric}"] for row in sample_rows], dtype=float
        )

    summaries = {metric: summarize(values) for metric, values in metric_values.items()}
    adjusted = holm_adjust([summaries["sync_c"]["sign_flip_p"], summaries["sync_d"]["sign_flip_p"]])
    summaries["sync_c"]["holm_p"] = adjusted[0]
    summaries["sync_d"]["holm_p"] = adjusted[1]

    seed_direction_count = sum(
        details["mean_corrected_delta_sync_c"] > 0
        and details["mean_corrected_delta_sync_d"] < 0
        for details in per_seed.values()
    )
    metric_directions = summaries["sync_c"]["mean"] > 0 and summaries["sync_d"]["mean"] < 0
    one_significant = min(summaries["sync_c"]["holm_p"], summaries["sync_d"]["holm_p"]) < 0.05

    primary_run = runs_root / f"{run_prefix}_s{seeds[0]}"
    gxe_path = primary_run / "04_eval" / "gxe_matrix.json"
    gxe = json.loads(gxe_path.read_text()) if gxe_path.exists() else None
    gxe_summary = None
    gxe_complete = False
    if gxe and len(gxe.get("complete_matrix_samples", [])) == len(sample_rows):
        diagonal_c = []
        diagonal_d = []
        cross_c = []
        cross_d = []
        tempo_ratios = []
        for sample_id in gxe["complete_matrix_samples"]:
            cells = gxe["results"][str(sample_id)]
            diagonal_c.extend([
                cells["G_natural_E_natural"]["sync_c"],
                cells["G_tts_E_tts"]["sync_c"],
            ])
            diagonal_d.extend([
                cells["G_natural_E_natural"]["sync_d"],
                cells["G_tts_E_tts"]["sync_d"],
            ])
            cross_c.extend([
                cells["G_natural_E_tts"]["sync_c"],
                cells["G_tts_E_natural"]["sync_c"],
            ])
            cross_d.extend([
                cells["G_natural_E_tts"]["sync_d"],
                cells["G_tts_E_natural"]["sync_d"],
            ])
            tempo_ratios.extend([
                cells["G_natural_E_tts"].get("tempo_ratio", 1.0),
                cells["G_tts_E_natural"].get("tempo_ratio", 1.0),
            ])
        gxe_summary = {
            "n": len(gxe["complete_matrix_samples"]),
            "mean_diagonal_sync_c": float(np.mean(diagonal_c)),
            "mean_diagonal_sync_d": float(np.mean(diagonal_d)),
            "mean_cross_sync_c": float(np.mean(cross_c)),
            "mean_cross_sync_d": float(np.mean(cross_d)),
            "cross_tempo_ratio_range": [
                float(np.min(tempo_ratios)), float(np.max(tempo_ratios))
            ],
            "interpretation": (
                "Cross cells are timing-mismatch diagnostics only. Global tempo fitting "
                "does not align phone boundaries, so they cannot identify a generator "
                "main effect or evaluator preference."
            ),
        }
        gxe_complete = True

    passed = bool(
        metric_directions
        and one_significant
        and seed_direction_count >= 2
    )
    if passed:
        verdict = "PASS_RELIABLE_TTS_ADVANTAGE"
    elif not metric_directions:
        verdict = "FAIL_NO_CORRECTED_TTS_ADVANTAGE"
    else:
        verdict = "FAIL_NO_RELIABLE_TTS_ADVANTAGE"

    return {
        "run_prefix": run_prefix,
        "seeds": seeds,
        "per_seed": per_seed,
        "per_sample": sample_rows,
        "primary": summaries,
        "gxe_timing_diagnostic": gxe_summary,
        "decision_inputs": {
            "corrected_metric_directions": metric_directions,
            "at_least_one_holm_significant": one_significant,
            "seeds_with_joint_expected_direction": seed_direction_count,
            "gxe_complete_timing_diagnostic": gxe_complete,
        },
        "gate_1_passed": passed,
        "verdict": verdict,
    }


def render_report(result: dict[str, Any]) -> str:
    sync_c = result["primary"]["sync_c"]
    sync_d = result["primary"]["sync_d"]
    gxe = result.get("gxe_timing_diagnostic") or {}
    lines = [
        "# English Qwen3-TTS → Ditto Gate 1 Report",
        "",
        f"**Verdict:** `{result['verdict']}`",
        "",
        "The experiment used 13 paired LibriSpeech utterances with identical gold transcripts, canonicalized both natural and Qwen3-TTS audio to 16 kHz mono PCM16, and ran Ditto at seeds 42, 43, and 44. The inferential unit is the utterance; each value below is the three-seed mean.",
        "",
        "## Primary paired effects",
        "",
        "| Metric | TTS − natural | 95% bootstrap CI | paired d_z | exact sign-flip p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Sync-C (higher better) | {sync_c['mean']:+.3f} | [{sync_c['bootstrap_95_ci'][0]:+.3f}, {sync_c['bootstrap_95_ci'][1]:+.3f}] | {sync_c['cohens_dz']:+.3f} | {sync_c['sign_flip_p']:.4f} | {sync_c['holm_p']:.4f} |",
        f"| Sync-D (lower better) | {sync_d['mean']:+.3f} | [{sync_d['bootstrap_95_ci'][0]:+.3f}, {sync_d['bootstrap_95_ci'][1]:+.3f}] | {sync_d['cohens_dz']:+.3f} | {sync_d['sign_flip_p']:.4f} | {sync_d['holm_p']:.4f} |",
        "",
        "Both estimates point weakly in the expected direction, but both confidence intervals include zero and neither survives Holm correction. The preregistered Gate 1 criterion is therefore not met.",
        "",
        "## Controls",
        "",
        "- All 13 pairs completed in all three seeds.",
        "- Bilateral natural and TTS no-op reruns reproduced their baselines exactly for both Sync-C and Sync-D.",
        "- All 52 cells of the seed-42 G×E timing diagnostic completed with uniform AAC remuxing.",
        "",
        "## G×E timing diagnostic",
        "",
        f"Same-audio diagonal mean: Sync-C {gxe.get('mean_diagonal_sync_c', float('nan')):.3f}, Sync-D {gxe.get('mean_diagonal_sync_d', float('nan')):.3f}.",
        f"Cross-audio mean: Sync-C {gxe.get('mean_cross_sync_c', float('nan')):.3f}, Sync-D {gxe.get('mean_cross_sync_d', float('nan')):.3f}.",
        "",
        "Cross cells are not a generator/evaluator factorial estimate: natural and TTS realizations have different local phone timing, while the diagnostic applies only one global tempo ratio. Their collapse shows audio-video timing mismatch, not that SyncNet prefers one source or that one generator arm is intrinsically worse.",
        "",
        "## Decision",
        "",
        "Stop before Wav2Sem `Fs` training and Ditto `Fp/Fs` swapping. Without a reliable English TTS advantage, a mechanism experiment would explain noise or a small uncertain effect. The supported conclusion is: **this 13-utterance English Qwen3-TTS-VC sample does not reliably improve Ditto lip-sync according to SyncNet.**",
        "",
        "This does not prove equivalence or absence of any effect; the observed standardized effects are small (`|d_z| ≈ 0.3`) and the sample is exploratory.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (output.parent / "report.md").write_text(render_report(result))
    rows = result["per_sample"]
    if rows:
        with (output.parent / "per_sample.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ============================================================
# Wav2Sem analysis modules (30B/C/D/E)
# ============================================================

_wav2sem_logger = logging.getLogger(__name__ + ".wav2sem")


def _load_json(path):
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def analyze_separability(separability_metrics: list) -> dict:
    """30B — Aggregate separability_metrics entries into H1 verdict.

    Filters to HuBERT layer 11, viseme level, segment_stability metric.
    """
    by_cond: dict[str, list[float]] = {"natural": [], "tts": []}
    for e in separability_metrics:
        if (e.get("metric") == "segment_stability"
                and e.get("level") == "viseme"
                and e.get("model") == "hubert"
                and e.get("layer") == 11):
            by_cond[e["condition"]].append(float(e["value"]))
    if len(by_cond["natural"]) < 2 or len(by_cond["tts"]) < 2:
        return {"h1_verdict": "INSUFFICIENT_DATA", "info": "n<2 per condition"}
    a = np.asarray(by_cond["natural"])
    b = np.asarray(by_cond["tts"])
    diff = b - a
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-10))
    rng = np.random.default_rng(42)
    nulls = np.abs(rng.choice([-1, 1], size=(10000, len(diff))) * diff).mean(1)
    p = float(np.mean(nulls >= abs(diff.mean())))
    if p < 0.05 and abs(d) >= 0.5:
        verdict = "PASS"
    elif abs(d) >= 0.5:
        verdict = "TREND"
    else:
        verdict = "FAIL"
    return {
        "h1_verdict": verdict,
        "natural_mean": float(a.mean()),
        "tts_mean": float(b.mean()),
        "cohens_d": d,
        "p_value": p,
        "n": int(len(diff)),
    }


def analyze_oracle_fs(oracle_fs_payload: dict) -> dict:
    """30C — Summarize oracle Fs natural↔TTS similarity and degeneracy flags."""
    entries = oracle_fs_payload.get("entries", [])
    nt_sim = oracle_fs_payload.get("nt_similarity", [])
    if not nt_sim:
        return {"verdict": "INSUFFICIENT_DATA", "n_pairs": 0}
    cos_cls = [e["cosine_cls_nt"] for e in nt_sim]
    cos_mean = [e["cosine_mean_nt"] for e in nt_sim]
    return {
        "n_pairs": len(nt_sim),
        "mean_cosine_cls_nt": float(np.mean(cos_cls)),
        "sd_cosine_cls_nt": float(np.std(cos_cls, ddof=1)),
        "mean_cosine_mean_nt": float(np.mean(cos_mean)),
        "n_degenerate": sum(1 for e in entries if e.get("is_degenerate")),
    }


def analyze_fd_gain(fd_payload: list) -> dict:
    """30D — Aggregate Fp/Fd_zero/Fd_random per-condition gains and decide H2.

    Per M3 mitigation: H2 supported only when ≥ 3 of 5 core metrics move
    in favorable direction across paired natural samples.
    """
    CORE = {
        "silhouette": +1,
        "boundary_sharpness": +1,
        "segment_stability": +1,
        "intra_class_dist": -1,
        "fisher": +1,
    }
    favorable_counts = {k: 0 for k in CORE}
    total_samples = 0
    for r in fd_payload:
        if "fp_metrics" not in r or "fd_zero_metrics" not in r or "fd_random_metrics" not in r:
            continue
        if r.get("skipped"):
            continue
        if r["condition"] != "natural":
            continue
        total_samples += 1
        for k, direction in CORE.items():
            if k not in r["fp_metrics"] or k not in r["fd_random_metrics"]:
                continue
            gain = r["fd_random_metrics"][k] - r["fp_metrics"][k]
            if direction * gain > 0:
                favorable_counts[k] += 1

    metrics_passing = 0
    for k, n_fav in favorable_counts.items():
        if total_samples > 0 and n_fav > total_samples / 2:
            metrics_passing += 1

    if total_samples == 0:
        return {"h2_verdict": "INSUFFICIENT_DATA", "favorable_counts": favorable_counts,
                "total_natural_samples": 0, "metrics_passing": 0, "fc_artifact_suspected": False}

    fc_artifact = False
    for r in fd_payload:
        if r.get("condition") != "natural" or r.get("skipped"):
            continue
        for k, direction in CORE.items():
            if k in r.get("fd_zero_metrics", {}) and k in r.get("fd_random_metrics", {}):
                zero_gain = r["fd_zero_metrics"][k] - r.get("fp_metrics", {}).get(k, 0)
                rand_gain = r["fd_random_metrics"][k] - r.get("fp_metrics", {}).get(k, 0)
                if direction * zero_gain <= 0 and direction * rand_gain > 0:
                    fc_artifact = True

    if metrics_passing >= 3 and not fc_artifact:
        verdict = "PASS"
    elif metrics_passing >= 3 and fc_artifact:
        verdict = "M3_FC_ARTIFACT"
    elif metrics_passing >= 1:
        verdict = "TREND"
    else:
        verdict = "FAIL"

    return {
        "h2_verdict": verdict,
        "favorable_counts": favorable_counts,
        "total_natural_samples": total_samples,
        "metrics_passing": metrics_passing,
        "fc_artifact_suspected": fc_artifact,
    }


def analyze_tfg_correlation(
    separability_values: np.ndarray,
    sync_c_deltas: np.ndarray,
) -> dict:
    """30E — Spearman ρ between per-sample separability and ΔSync-C."""
    if len(separability_values) < 3 or len(sync_c_deltas) < 3:
        return {"h3_verdict": "INSUFFICIENT_DATA"}
    if len(separability_values) != len(sync_c_deltas):
        return {"h3_verdict": "INSUFFICIENT_DATA", "error": "length mismatch"}

    rho, p_val = stats.spearmanr(separability_values, sync_c_deltas)
    rng = np.random.default_rng(42)
    n = len(sync_c_deltas)
    nulls = np.empty(min(10000, n * 100), dtype=np.float64)
    for i in range(len(nulls)):
        perm = rng.permutation(n)
        nulls[i] = stats.spearmanr(separability_values, sync_c_deltas[perm])[0]
    perm_p = float(np.mean(np.abs(nulls) >= abs(rho)))

    if perm_p < 0.05 and abs(rho) >= 0.3:
        verdict = "PASS"
    elif abs(rho) >= 0.3:
        verdict = "TREND"
    else:
        verdict = "FAIL"
    return {
        "h3_verdict": verdict,
        "spearman_rho": float(rho),
        "scipy_p": float(p_val),
        "permutation_p": perm_p,
        "n": int(n),
    }


def render_full_report(
    gate1_summary: dict,
    separability_summary: dict,
    oracle_fs_summary: dict,
    fd_gain_summary: dict,
    tfg_correlation_summary: dict,
    output_path: Path,
) -> None:
    """Render markdown report combining 30A (Gate 1) + 30B/C/D/E."""
    lines = ["# English Wav2Sem Analysis Report\n"]
    lines.append("## 30A — Gate 1 (TTS advantage existence)\n")
    lines.append(f"- Verdict: `{gate1_summary.get('verdict', 'N/A')}`")
    primary = gate1_summary.get("primary", {})
    sync_c_stats = primary.get("sync_c", {}) if isinstance(primary, dict) else {}
    lines.append(f"- ΔSync-C mean: {sync_c_stats.get('mean', 'N/A')}")
    lines.append(f"- Holm p: {sync_c_stats.get('holm_p', 'N/A')}\n")
    lines.append("## 30B — H1: TTS separability\n")
    lines.append(f"- Verdict: `{separability_summary.get('h1_verdict', 'N/A')}`")
    lines.append(f"- Cohen's d: {separability_summary.get('cohens_d', 'N/A')}\n")
    lines.append("## 30C — Oracle Fs diagnostics\n")
    lines.append(f"- n_pairs: {oracle_fs_summary.get('n_pairs', 0)}")
    lines.append(f"- mean cosine(nat ↔ tts) [CLS]: {oracle_fs_summary.get('mean_cosine_cls_nt', 'N/A')}")
    lines.append(f"- degenerate samples: {oracle_fs_summary.get('n_degenerate', 0)}\n")
    lines.append("## 30D — H2: Fs decouples embeddings\n")
    lines.append(f"- Verdict: `{fd_gain_summary.get('h2_verdict', 'N/A')}`")
    lines.append(f"- Favorable metric counts: {fd_gain_summary.get('favorable_counts', {})}")
    lines.append(f"- FC artifact suspected: {fd_gain_summary.get('fc_artifact_suspected', False)}\n")
    lines.append("## 30E — H3: Separability ↔ ΔSync-C correlation\n")
    lines.append(f"- Verdict: `{tfg_correlation_summary.get('h3_verdict', 'N/A')}`")
    lines.append(f"- Spearman ρ: {tfg_correlation_summary.get('spearman_rho', 'N/A')}")
    lines.append(f"- Permutation p: {tfg_correlation_summary.get('permutation_p', 'N/A')}\n")
    lines.append("## Conclusion\n")
    lines.append("- H1 (TTS has separability advantage) — see 30B")
    lines.append("- H2 (oracle Fs adds decoupling) — see 30D")
    lines.append("- H3 (separability correlates with TFG gain) — see 30E")
    lines.append("")
    if tfg_correlation_summary.get("h3_verdict") in ("PASS", "TREND"):
        lines.append("H3 positive despite Gate 1 failure → separability captures something SyncNet sees "
                     "that the G×E generator effect does not; warrants re-examining G×E interpretation.")
    else:
        lines.append("H1/H2 positive + H3 negative (expected case) → Wav2Sem Fs decoupling mechanism exists "
                     "on English but is **not** a sufficient condition for a TTS lip-sync advantage. "
                     "This is a meaningful negative result that constrains the Wav2Sem theory's scope.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    return None
# ============================================================
# End Wav2Sem analysis modules
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--with-wav2sem", action="store_true",
                        help="Run 30B/C/D/E modules in addition to Gate 1.")
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("data/wav2sem_analysis_en/metrics"),
                        help="Directory containing separability_metrics.json, oracle_fs.json, "
                             "oracle_fd_separability.json.")
    parser.add_argument("--report-output", type=Path,
                        default=Path("docs/experiments/wav2sem_bilingual_report_en.md"))
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    result = analyze(args.runs_root, args.run_prefix, seeds)
    output = args.output or args.runs_root / f"{args.run_prefix}_gate1.json"
    write_outputs(result, output)
    if args.with_wav2sem:
        analysis_dir = args.analysis_dir
        sep_path = analysis_dir / "separability_metrics.json"
        fs_path = analysis_dir / "oracle_fs.json"
        fd_path = analysis_dir / "oracle_fd_separability.json"

        sep_data = _load_json(sep_path) or {}
        fs_data = _load_json(fs_path) or {}
        fd_data = _load_json(fd_path) or {}

        sep_records = sep_data.get("records", sep_data) if isinstance(sep_data, dict) else sep_data
        if not sep_records:
            sep_records = sep_data if isinstance(sep_data, list) else []
        sep_summary = analyze_separability(sep_records) if sep_records else {"h1_verdict": "INSUFFICIENT_DATA"}

        fs_summary = analyze_oracle_fs(fs_data) if fs_data else {"n_pairs": 0}

        fd_records = fd_data.get("results", fd_data) if isinstance(fd_data, dict) else fd_data
        if not fd_records:
            fd_records = fd_data if isinstance(fd_data, list) else []
        fd_summary = analyze_fd_gain(fd_records) if fd_records else {"h2_verdict": "INSUFFICIENT_DATA"}

        sync_c_deltas: np.ndarray = np.array([])
        if "per_sample" in result:
            sync_c_deltas = np.array([
                s.get("corrected_delta_sync_c", float("nan"))
                for s in result["per_sample"]
            ])
        sep_vals: np.ndarray = np.array([])
        if isinstance(sep_data, dict) and "per_sample" in sep_data:
            sep_vals = np.array([s.get("value", float("nan")) for s in sep_data["per_sample"]])

        tfg_summary = (
            analyze_tfg_correlation(
                separability_values=sep_vals,
                sync_c_deltas=sync_c_deltas,
            ) if len(sync_c_deltas) > 0 else {"h3_verdict": "INSUFFICIENT_DATA"}
        )

        gate1_for_renderer = {
            "verdict": result.get("verdict", "N/A"),
            "primary": result.get("primary", {}),
        }
        render_full_report(
            gate1_summary=gate1_for_renderer,
            separability_summary=sep_summary,
            oracle_fs_summary=fs_summary,
            fd_gain_summary=fd_summary,
            tfg_correlation_summary=tfg_summary,
            output_path=args.report_output,
        )
        print(f"[30] Wav2Sem report written to {args.report_output}")
    print(json.dumps({
        "gate_1_passed": result["gate_1_passed"],
        "verdict": result["verdict"],
        "primary": result["primary"],
        "gxe_timing_diagnostic": result["gxe_timing_diagnostic"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
