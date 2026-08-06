#!/usr/bin/env python3
"""B1 — Duration structure JSD between natural and TTS speech (MFA aligned).

Tests the "duration flattening / mean collapse" hypothesis for TTS:

  1. Per-phoneme duration distributions: for each phone symbol with enough
     samples, compare natural vs TTS duration histograms via the Jensen-
     Shannon divergence (JSD), with a permutation test for significance.
     Durations are RATE-NORMALIZED per utterance (divided by that utterance's
     mean phone duration) so the JSD isolates distribution *shape* from the
     global speech-rate shift.  A null-reference median JSD is reported
     alongside the observed value so readers can calibrate power.
  2. Pause structure: internal pause (sil) duration distribution JSD + pause
     count per utterance.
  3. Per-utterance paired metrics (natural vs TTS, paired by sample id):
     total duration, phone count, mean phone duration, intra-utterance
     duration std and coefficient of variation (std/mean — flatness
     independent of speech rate), pause count — paired permutation test +
     bootstrap CI + Cohen's d, with BH-FDR within the family and an
     outlier-exclusion sensitivity pass.

Statistical hardening (from adversarial review):
  - 6 histogram bins (16 made the null JSD ≈ observed at n=20-40).
  - Permutation p = (1 + #{null >= obs}) / (1 + n_perm); no p == 0.0 floor.
  - BH-FDR applied within each test family; raw p always reported.
  - NaN pairs dropped explicitly before paired tests.
  - MFA "spn" (noise) intervals excluded from phone metrics.
  - Outlier utterances (TTS/natural duration ratio far from 1) flagged and
    the paired metrics reported with and without them.

Literature anchor: Abbas et al. (Interspeech 2022, arXiv:2206.14165) use JSD
between predicted and target duration distributions; deterministic L1/L2
duration models collapse toward mean durations (flat prosody, insufficient
pauses).

Usage
-----
    python scripts/36_duration_jsd.py [--out-dir results/aishell100_phoneme/duration_analysis]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import paired_permutation_test, bootstrap_paired_ci, fdr_bh_correction

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEXTGRID_DIRS = {
    "natural": PROJECT_ROOT / "results/aishell100_phoneme/mfa_full/out/natural",
    "tts": PROJECT_ROOT / "results/aishell100_phoneme/mfa_full/out/tts",
}
DEFAULT_MANIFEST = PROJECT_ROOT / "results/aishell100_phoneme/alignment/manifest.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/aishell100_phoneme/duration_analysis"

MIN_PHONE_SAMPLES = 20   # per condition, for a phone to enter the per-phone JSD test
N_HIST_BINS = 6          # low bin count: keeps the null JSD small at n=20-40
N_PERM = 2000
RNG_SEED = 42

# MFA "noise" label — acoustically filled/noise intervals, not phonemes.
NOISE_SYMBOLS = {"spn"}

# Utterances with |log(tts_dur / natural_dur)| > this are flagged as outliers
# (likely TTS generation failures) and reported as a sensitivity analysis.
OUTLIER_LOG_RATIO = 0.5   # ratio < 0.607 or > 1.649

# Tone marks + combining diacritics stripped to recover the base phone.
_TONE_RE = re.compile(r"[˥-˩ˊˋ̀-ͯ]+$")


def strip_tone(symbol: str) -> str:
    """Remove trailing tone / combining diacritics, keep the base phone.

    Modifier letters (ʰ aspiration U+02B0, ʲ palatalization U+02B2, ʷ
    labialization U+02B7) are NOT combining diacritics and are preserved.
    """
    return _TONE_RE.sub("", symbol)


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------
def parse_textgrid_phones(path: Path) -> list[dict]:
    """Parse the ``phones`` IntervalTier of an MFA TextGrid.

    Returns
    -------
    list[dict] : [{symbol, start_s, end_s}, ...] in time order.
    """
    text = path.read_text(encoding="utf-8")
    tier_m = re.search(r'item\s*\[\d+\]:\s*class\s*=\s*"IntervalTier"\s*name\s*=\s*"phones"', text)
    if not tier_m:
        tier_m = re.search(r'name\s*=\s*"phones"\s*xmin', text)
    if not tier_m:
        tier_m = None
        for m in re.finditer(r"item\s*\[\d+\]:", text):
            tier_m = m
    if tier_m is None:
        logger.warning("No interval tier found in %s", path.name)
        return []

    body = text[tier_m.start():]
    next_item = re.search(r"\n\s*item\s*\[\d+\]:", body[20:])
    if next_item:
        body = body[: 20 + next_item.start()]

    intervals: list[dict] = []
    for m in re.finditer(
        r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*\"([^\"]*)\"",
        body,
    ):
        symbol = m.group(3)
        if not symbol or symbol == "<eps>":
            continue
        intervals.append({
            "symbol": symbol,
            "start_s": float(m.group(1)),
            "end_s": float(m.group(2)),
        })
    return intervals


def split_pauses(intervals: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split intervals into (phones, internal pauses).

    Leading/trailing sil are dropped; only sil between two non-sil intervals
    counts as an internal pause.  Adjacent internal sil intervals merge.
    """
    phones: list[dict] = []
    pauses: list[dict] = []
    seen_phone = False
    pending_pause: dict | None = None
    for iv in intervals:
        if iv["symbol"] == "sil":
            if seen_phone and pending_pause is None:
                pending_pause = dict(iv)
            continue
        if pending_pause is not None:
            pauses.append(pending_pause)
            pending_pause = None
        phones.append(iv)
        seen_phone = True
    return phones, pauses


# ---------------------------------------------------------------------------
# Histogram / divergence helpers
# ---------------------------------------------------------------------------
def shared_hist_bins(a: np.ndarray, b: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-spaced bin edges over the combined sample (robust to skew)."""
    combined = np.concatenate([a, b])
    if len(combined) < n_bins:
        n_bins = max(2, len(combined))
    if np.ptp(combined) == 0:
        eps = 0.001
        return np.array([combined[0] - eps, combined[0] + eps])
    edges = np.unique(np.quantile(combined, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        eps = 0.001
        return np.array([combined[0] - eps, combined[0] + eps])
    return edges


def histogram(edges: np.ndarray, x: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x, bins=edges)
    h = h.astype(np.float64)
    return h / h.sum() if h.sum() > 0 else h


def smooth(hist: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    h = hist + eps
    return h / h.sum()


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = smooth(p)
    q = smooth(q)
    return float(np.sum(p * np.log(p / q)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (smooth(p) + smooth(q))
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def jsd_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    edges: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Observed JSD, permutation p-value, and null-median JSD.

    p = (1 + #{null >= observed}) / (1 + n_perm): never exactly 0.
    The null-median JSD is returned so readers can calibrate the test's
    resolving power (if observed ≈ null median, the test is underpowered).
    """
    rng = np.random.default_rng(seed)
    observed = jsd(histogram(edges, a), histogram(edges, b))
    pooled = np.concatenate([a, b])
    na = len(a)
    null_jsds = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null_jsds[i] = jsd(histogram(edges, perm[:na]), histogram(edges, perm[na:]))
    p = float((1 + np.sum(null_jsds >= observed)) / (1 + n_perm))
    return observed, p, float(np.median(null_jsds))


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d (d_z): mean(diff) / std(diff)."""
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(diff) / sd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_samples(textgrid_dir: Path) -> dict[str, dict]:
    """Return ``{sample_id: {"phones": [...], "pauses": [...], "file": str}}``."""
    samples: dict[str, dict] = {}
    for tg in sorted(textgrid_dir.glob("*.TextGrid")):
        sid = tg.stem
        try:
            intervals = parse_textgrid_phones(tg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse %s: %s", tg.name, exc)
            continue
        phones, pauses = split_pauses(intervals)
        # Exclude MFA noise labels from phone-level metrics.
        phones = [p for p in phones if p["symbol"] not in NOISE_SYMBOLS]
        samples[sid] = {"phones": phones, "pauses": pauses, "file": tg.name}
    return samples


def _rate_normalize(phones: list[dict]) -> list[float]:
    """Return per-utterance durations divided by the utterance mean duration."""
    durs = np.array([p["end_s"] - p["start_s"] for p in phones], dtype=np.float64)
    if durs.size == 0 or np.mean(durs) == 0:
        return []
    return (durs / np.mean(durs)).tolist()


def _compute_per_phone_jsd(
    phone_dur: dict[str, dict[str, list[float]]],
    min_phone_samples: int,
    n_bins: int,
    n_perm: int,
    label: str,
) -> tuple[dict[str, dict], dict]:
    """Per-phone duration-distribution JSD with within-family BH-FDR.

    Returns ``(per_phone, summary)``.  ``per_phone[symbol]`` carries the
    observed JSD, the null-reference median JSD (power calibration), raw p,
    and the FDR-corrected q.
    """
    per_phone: dict[str, dict] = {}
    phones_with_p: list[tuple[str, float]] = []
    for symbol, conds in sorted(phone_dur.items()):
        a = np.array(conds["natural"], dtype=np.float64)
        b = np.array(conds["tts"], dtype=np.float64)
        if len(a) < min_phone_samples or len(b) < min_phone_samples:
            continue
        edges = shared_hist_bins(a, b, n_bins)
        jsd_val, p, null_med = jsd_permutation_test(a, b, edges, n_perm=n_perm)
        per_phone[symbol] = {
            "natural_n": int(len(a)),
            "tts_n": int(len(b)),
            "natural_mean_dur": float(np.mean(a)),
            "tts_mean_dur": float(np.mean(b)),
            "jsd": float(jsd_val),
            "null_median_jsd": float(null_med),
            "p_perm": float(p),
            "q_fdr": None,
            "significant": False,
        }
        phones_with_p.append((symbol, p))

    if phones_with_p:
        pvals = np.array([p for _, p in phones_with_p])
        qvals = fdr_bh_correction(pvals)
        for (symbol, _), q in zip(phones_with_p, qvals):
            per_phone[symbol]["q_fdr"] = float(q)
            per_phone[symbol]["significant"] = bool(q < 0.05)

    tested_jsds = [v["jsd"] for v in per_phone.values()]
    summary = {
        "label": label,
        "n_phones_tested": len(per_phone),
        "n_significant_fdr": sum(1 for v in per_phone.values() if v["significant"]),
        "mean_jsd": float(np.mean(tested_jsds)) if tested_jsds else None,
        "median_jsd": float(np.median(tested_jsds)) if tested_jsds else None,
        "mean_null_median_jsd": float(np.mean([v["null_median_jsd"] for v in per_phone.values()])) if per_phone else None,
        "max_jsd_phone": max(per_phone, key=lambda k: per_phone[k]["jsd"]) if per_phone else None,
    }
    return per_phone, summary


def run(
    textgrid_dirs: dict[str, Path],
    manifest_path: Path,
    out_dir: Path,
    min_phone_samples: int = MIN_PHONE_SAMPLES,
    n_bins: int = N_HIST_BINS,
    n_perm: int = N_PERM,
) -> int:
    natural = _load_samples(textgrid_dirs["natural"])
    tts = _load_samples(textgrid_dirs["tts"])
    logger.info("Loaded %d natural / %d tts TextGrids", len(natural), len(tts))

    common_ids = sorted(set(natural) & set(tts))
    if not common_ids:
        logger.error("No paired samples found between conditions.")
        return 1
    logger.info("%d paired samples", len(common_ids))

    # ---- Cross-check pairing against the alignment manifest --------------
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            transcripts = {}
            for rec in manifest.get("records", []):
                transcripts.setdefault(rec["paired_key"], {})[rec["condition"]] = rec.get("transcript", "")
            mismatch = [sid for sid in common_ids
                        if sid in transcripts and transcripts[sid].get("natural") != transcripts[sid].get("tts")]
            if mismatch:
                logger.warning("Transcript mismatch for paired samples: %s", mismatch[:10])
            else:
                logger.info("Manifest pairing verified for %d samples", len(common_ids))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read manifest %s: %s", manifest_path, exc)
    else:
        logger.warning("Manifest not found at %s; pairing is by filename stem", manifest_path)

    # ---- Outlier flagging (TTS/natural duration ratio far from 1) --------
    dur_ratio = {}
    for sid in common_ids:
        nat_dur = natural[sid]["phones"][-1]["end_s"] - natural[sid]["phones"][0]["start_s"] if natural[sid]["phones"] else 0.0
        tts_dur = tts[sid]["phones"][-1]["end_s"] - tts[sid]["phones"][0]["start_s"] if tts[sid]["phones"] else 0.0
        dur_ratio[sid] = float(tts_dur / nat_dur) if nat_dur > 0 else float("nan")
    outlier_ids = sorted(sid for sid, r in dur_ratio.items() if r == r and abs(np.log(r)) > OUTLIER_LOG_RATIO)
    if outlier_ids:
        logger.warning("Flagged %d outlier utterances (|log tts/natural dur| > %.2f): %s",
                       len(outlier_ids), OUTLIER_LOG_RATIO, outlier_ids[:15])

    # ---- 1. Per-phone duration distribution JSD --------------------------
    # Two views: raw durations and rate-normalized (isolates distribution
    # shape from the global speech-rate shift).  Two groupings: IPA-with-tone
    # and tone-stripped base phone.
    phone_dur: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"natural": [], "tts": []})
    phone_dur_rn: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"natural": [], "tts": []})
    phone_base_dur_rn: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"natural": [], "tts": []})
    for sid in common_ids:
        for cond, pool in (("natural", natural), ("tts", tts)):
            phones = pool[sid]["phones"]
            rn = _rate_normalize(phones)
            for i, ph in enumerate(phones):
                dur = ph["end_s"] - ph["start_s"]
                if dur <= 0:
                    continue
                phone_dur[ph["symbol"]][cond].append(dur)
                if i < len(rn):
                    phone_dur_rn[ph["symbol"]][cond].append(rn[i])
                    phone_base_dur_rn[strip_tone(ph["symbol"])][cond].append(rn[i])

    per_phone_raw, summary_raw = _compute_per_phone_jsd(
        phone_dur, min_phone_samples, n_bins, n_perm, label="raw_duration"
    )
    per_phone_rn, summary_rn = _compute_per_phone_jsd(
        phone_dur_rn, min_phone_samples, n_bins, n_perm, label="rate_normalized_duration"
    )
    per_phone_base_rn, summary_base_rn = _compute_per_phone_jsd(
        phone_base_dur_rn, min_phone_samples, n_bins, n_perm, label="rate_normalized_base_phone"
    )

    # ---- 2. Pause structure ----------------------------------------------
    pause_dur: dict[str, list[float]] = {"natural": [], "tts": []}
    pause_counts: dict[str, dict[str, int]] = {"natural": {}, "tts": {}}
    for sid in common_ids:
        for cond, pool in (("natural", natural), ("tts", tts)):
            pauses = pool[sid]["pauses"]
            pause_counts[cond][sid] = len(pauses)
            for p in pauses:
                d = p["end_s"] - p["start_s"]
                if d > 0:
                    pause_dur[cond].append(d)

    pause_summary: dict = {"natural": {}, "tts": {}}
    if len(pause_dur["natural"]) >= 5 and len(pause_dur["tts"]) >= 5:
        a = np.array(pause_dur["natural"], dtype=np.float64)
        b = np.array(pause_dur["tts"], dtype=np.float64)
        edges = shared_hist_bins(a, b, n_bins)
        jsd_val, p, null_med = jsd_permutation_test(a, b, edges, n_perm=n_perm)
        pause_summary = {
            "natural": {"n": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a))},
            "tts": {"n": int(len(b)), "mean": float(np.mean(b)), "median": float(np.median(b))},
            "jsd": float(jsd_val),
            "null_median_jsd": float(null_med),
            "p_perm": float(p),
            "q_fdr": None,
            "significant": False,
            "note": "Unpaired (pooled durations); underpowered for duration-shape claims.",
        }

    # ---- 3. Per-utterance paired metrics --------------------------------
    def _durs(p: list[dict]) -> np.ndarray:
        return np.array([x["end_s"] - x["start_s"] for x in p], dtype=np.float64)

    pair_metrics = {
        "total_duration_s": (lambda p: (p[-1]["end_s"] - p[0]["start_s"]) if p else float("nan")),
        "phone_count": (lambda p: float(len(p))),
        "mean_phone_dur": (lambda p: float(np.mean(_durs(p))) if p else float("nan")),
        "dur_std_within_utterance": (lambda p: float(np.std(_durs(p))) if len(p) > 1 else float("nan")),
        "dur_cv_within_utterance": (lambda p: (float(np.std(_durs(p)) / np.mean(_durs(p)))
                                               if len(p) > 1 and np.mean(_durs(p)) > 0 else float("nan"))),
    }
    # ``pause_count`` reads from the pauses list, not the phones list.
    pair_metric_sources = {"phones": list(pair_metrics.keys()), "pauses": ["pause_count"]}

    def _paired_rows(ids: list[str]) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        for source, names in pair_metric_sources.items():
            for name in names:
                fn = pair_metrics[name] if source == "phones" else (lambda p: float(len(p)))
                a = np.array([fn(natural[sid][source]) for sid in ids], dtype=np.float64)
                b = np.array([fn(tts[sid][source]) for sid in ids], dtype=np.float64)
                # Drop NaN pairs explicitly (empty/singleton utterances).
                ok = ~(np.isnan(a) | np.isnan(b))
                a, b = a[ok], b[ok]
                if len(a) < 2:
                    continue
                _, observed, null_dist = paired_permutation_test(
                    a, b, n_permutations=n_perm, random_seed=RNG_SEED
                )
                # +1 correction: never report p == 0.0 (finite permutation budget).
                p_perm = float((1 + np.sum(np.abs(null_dist) >= np.abs(observed))) / (1 + n_perm))
                ci_low, ci_high, mean_diff = bootstrap_paired_ci(a, b, n_bootstrap=1000, random_seed=RNG_SEED)
                rows[name] = {
                    "natural_mean": float(np.mean(a)),
                    "tts_mean": float(np.mean(b)),
                    "delta_natural_minus_tts": float(mean_diff),
                    "ci_95": [float(ci_low), float(ci_high)],
                    "cohens_d": cohens_d_paired(a, b),
                    "p_perm": float(p_perm),
                    "q_fdr": None,
                    "significant": False,
                    "n_pairs": int(len(a)),
                }
        return rows

    paired_results = _paired_rows(common_ids)

    # Within-family FDR over the per-utterance metrics.
    utt_p = [(k, v["p_perm"]) for k, v in paired_results.items()]
    if utt_p:
        qvals = fdr_bh_correction(np.array([p for _, p in utt_p]))
        for (k, _), q in zip(utt_p, qvals):
            paired_results[k]["q_fdr"] = float(q)
            paired_results[k]["significant"] = bool(q < 0.05)

    # Outlier-exclusion sensitivity: re-run the per-utterance family without
    # the flagged utterances.
    kept_ids = [sid for sid in common_ids if sid not in outlier_ids]
    paired_results_no_outliers = _paired_rows(kept_ids) if kept_ids else {}
    utt_p_no = [(k, v["p_perm"]) for k, v in paired_results_no_outliers.items()]
    if utt_p_no:
        qvals = fdr_bh_correction(np.array([p for _, p in utt_p_no]))
        for (k, _), q in zip(utt_p_no, qvals):
            paired_results_no_outliers[k]["q_fdr"] = float(q)
            paired_results_no_outliers[k]["significant"] = bool(q < 0.05)

    # Leave-one-out sensitivity for the fragile dur_cv metric.
    cv_sensitivity: dict | None = None
    if "dur_cv_within_utterance" in pair_metrics:
        cv_ps = []
        for sid in common_ids:
            ids_loo = [s for s in common_ids if s != sid]
            rows = _paired_rows(ids_loo)
            if "dur_cv_within_utterance" in rows:
                cv_ps.append((sid, rows["dur_cv_within_utterance"]["p_perm"]))
        if cv_ps:
            ps = np.array([p for _, p in cv_ps])
            full_p = paired_results["dur_cv_within_utterance"]["p_perm"]
            most_influential = max(cv_ps, key=lambda kv: abs(kv[1] - full_p))
            cv_sensitivity = {
                "n_loo_runs": len(cv_ps),
                "p_range": [float(ps.min()), float(ps.max())],
                "full_data_p": float(full_p),
                "most_influential_sample": most_influential[0],
                "p_without_most_influential": float(most_influential[1]),
            }

    # ---- Per-sample deltas for downstream reuse --------------------------
    per_sample: dict[str, dict] = {}
    for name, fn in pair_metrics.items():
        per_sample[name] = {}
        for sid in common_ids:
            per_sample[name][sid] = {
                "natural": float(fn(natural[sid]["phones"])),
                "tts": float(fn(tts[sid]["phones"])),
            }
    # pause_count in per_sample uses the actual pause lists:
    for sid in common_ids:
        per_sample.setdefault("pause_count", {})[sid] = {
            "natural": float(len(natural[sid]["pauses"])),
            "tts": float(len(tts[sid]["pauses"])),
        }

    # ---- Output ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "textgrid_dirs": {k: str(v) for k, v in textgrid_dirs.items()},
            "n_paired_samples": len(common_ids),
            "min_phone_samples": min_phone_samples,
            "n_hist_bins": n_bins,
            "n_perm": n_perm,
            "seed": RNG_SEED,
            "outlier_ids": outlier_ids,
            "outlier_criterion": "|log(tts_dur/natural_dur)| > %.2f" % OUTLIER_LOG_RATIO,
            "fdr_scope": "BH-FDR applied within each test family (per-phone, per-utterance); raw p always reported",
            "noise_excluded": sorted(NOISE_SYMBOLS),
        },
        "summary": {
            "per_phone_raw": summary_raw,
            "per_phone_rate_normalized": summary_rn,
            "per_phone_base_rate_normalized": summary_base_rn,
            "pause": pause_summary,
            "per_utterance": paired_results,
            "per_utterance_no_outliers": paired_results_no_outliers,
            "dur_cv_leave_one_out": cv_sensitivity,
        },
        "per_phone_raw": per_phone_raw,
        "per_phone_rate_normalized": per_phone_rn,
        "per_phone_base_rate_normalized": per_phone_base_rn,
        "per_utterance": paired_results,
        "per_utterance_no_outliers": paired_results_no_outliers,
        "per_sample": per_sample,
    }
    out_json = out_dir / "duration_jsd.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    # ---- Console summary --------------------------------------------------
    print("\n=== B1 Duration JSD summary (rate-normalized headline) ===")
    print("Per-phone (rate-norm): mean JSD=%.4f (null med ~%.4f), n=%d, sig=%d"
          % (summary_rn["mean_jsd"], summary_rn["mean_null_median_jsd"],
             summary_rn["n_phones_tested"], summary_rn["n_significant_fdr"]))
    print("Per-phone (base, rate-norm): mean JSD=%.4f (null med ~%.4f), n=%d, sig=%d"
          % (summary_base_rn["mean_jsd"], summary_base_rn["mean_null_median_jsd"],
             summary_base_rn["n_phones_tested"], summary_base_rn["n_significant_fdr"]))
    if pause_summary.get("jsd") is not None:
        print("Pause JSD=%.4f (null med ~%.4f) p=%.4f | nat mean=%.3fs (n=%d) tts mean=%.3fs (n=%d)"
              % (pause_summary["jsd"], pause_summary["null_median_jsd"], pause_summary["p_perm"],
                 pause_summary["natural"]["mean"], pause_summary["natural"]["n"],
                 pause_summary["tts"]["mean"], pause_summary["tts"]["n"]))
    print("\nOutliers flagged (n=%d): %s" % (len(outlier_ids), outlier_ids[:12]))
    print("\nPer-utterance paired (natural - tts):")
    for name, r in paired_results.items():
        print("  %-26s nat=%.4f tts=%.4f d=%+.2f p=%.4f q=%.3f sig=%s"
              % (name, r["natural_mean"], r["tts_mean"], r["cohens_d"],
                 r["p_perm"], r["q_fdr"], r["significant"]))
    if paired_results_no_outliers:
        print("\n  -- without outliers (%d samples) --" % len(kept_ids))
        for name, r in paired_results_no_outliers.items():
            print("  %-26s nat=%.4f tts=%.4f d=%+.2f p=%.4f q=%.3f sig=%s"
                  % (name, r["natural_mean"], r["tts_mean"], r["cohens_d"],
                     r["p_perm"], r["q_fdr"], r["significant"]))
    if cv_sensitivity:
        print("\ndur_cv leave-one-out p range: [%.4f, %.4f] (full-data p=%.4f)"
              % (cv_sensitivity["p_range"][0], cv_sensitivity["p_range"][1],
                 paired_results["dur_cv_within_utterance"]["p_perm"]))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--natural-dir", type=Path, default=DEFAULT_TEXTGRID_DIRS["natural"])
    parser.add_argument("--tts-dir", type=Path, default=DEFAULT_TEXTGRID_DIRS["tts"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-phone-samples", type=int, default=MIN_PHONE_SAMPLES)
    parser.add_argument("--n-bins", type=int, default=N_HIST_BINS)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    textgrid_dirs = {"natural": args.natural_dir, "tts": args.tts_dir}
    return run(
        textgrid_dirs,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        min_phone_samples=args.min_phone_samples,
        n_bins=args.n_bins,
        n_perm=args.n_perm,
    )


if __name__ == "__main__":
    sys.exit(main())
