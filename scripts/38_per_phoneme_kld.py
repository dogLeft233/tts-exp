#!/usr/bin/env python3
"""B3 — Per-phoneme KLD + per-phoneme natural-vs-TTS probe (HuBERT).

Compares natural vs TTS HuBERT embedding distributions per PHONEME, using
MFA phone-level boundaries (not whole syllables):

  1. **Per-phone pooling** — for each sample, pool HuBERT frames within each
     MFA phone interval (TextGrid phones tier).  Phone symbols are tone-
     stripped and grouped into base-phone classes.
  2. **Per-class Gaussian KLD** — PCA-reduced (default 30-d), Ledoit-Wolf
     shrunk covariance, symmetric KL divergence between natural and TTS.
  3. **Per-class grouped classifier** — natural vs TTS logistic regression on
     PCA components, folds grouped by UTTERANCE (GroupShuffleSplit) so no
     utterance's tokens straddle train/test; reports accuracy AND AUC (mean
     ± std across repeated splits).
  4. **Cross-class correlation** — Spearman(KLD, AUC) per consonant/vowel
     grouping (Temmar et al. IEEE SMC 2025 template).

Mandarin-phonology checks:
  - aspiration pairs (送气对立): b/p, d/t, g/k, zh/ch, j/q, z/c
  - vowel (final nuclei) vs consonant groups

Methodology notes (from adversarial review of v1):
  - v1 pooled the whole syllable and re-labeled the same vector across
    initial/final/tone views; this version pools per phone region.
  - v1's token-level folds collapsed out-of-sample (100 speakers vs 1 voice);
    this version groups folds by utterance.

Usage
-----
    python scripts/38_per_phoneme_kld.py --layers 1,4,7,10
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from tfg_feature_common import half_open_span_mask

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "runs/mdc_en_phoneme_20260807_f5_full/04_embeddings"
DEFAULT_TEXTGRID_DIRS = {
    "natural": PROJECT_ROOT / "runs/mdc_en_phoneme_20260807_f5_full/03_alignment/mfa_out/natural",
    "tts": PROJECT_ROOT / "runs/mdc_en_phoneme_20260807_f5_full/03_alignment/mfa_out/tts",
}
DEFAULT_MANIFEST = PROJECT_ROOT / "runs/mdc_en_phoneme_20260807_f5_full/03_alignment/alignment.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs/mdc_en_phoneme_20260807_f5_full/09_per_phoneme_kld"

FRAME_STRIDE = 320
TARGET_SR = 16000
RNG_SEED = 42
_PCA_DIM = 30
_MIN_SAMPLES = 20
CV_SPLITS = 5
CV_REPEATS = 3

ASPIRATION_PAIRS = [("b", "p"), ("d", "t"), ("g", "k"), ("zh", "ch"), ("j", "q"), ("z", "c")]

_TONE_RE = re.compile(r"[˥-˩ˊˋ̀-ͯ]+$")
_VOWEL_RE = re.compile(r"[aæɐɑɒeəɛɜɘɵɤoɔøɞʊuʉɯyʏiɪɨ]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def strip_tone(symbol: str) -> str:
    return _TONE_RE.sub("", symbol)


def is_vowel(phone: str) -> bool:
    return bool(_VOWEL_RE.search(phone))


def _load_f16() -> Any:
    spec = importlib.util.spec_from_file_location(
        "feature_separability_16", str(PROJECT_ROOT / "scripts/16_feature_separability.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_textgrid_phones(path: Path) -> list[dict]:
    """Parse the ``phones`` IntervalTier of an MFA TextGrid (sil excluded)."""
    text = path.read_text(encoding="utf-8")
    tier_m = re.search(r'item\s*\[\d+\]:\s*class\s*=\s*"IntervalTier"\s*name\s*=\s*"phones"', text)
    if not tier_m:
        tier_m = re.search(r'name\s*=\s*"phones"\s*xmin', text)
    if not tier_m:
        tier_m = None
        for m in re.finditer(r"item\s*\[\d+\]:", text):
            tier_m = m
    if tier_m is None:
        return []
    body = text[tier_m.start():]
    nxt = re.search(r"\n\s*item\s*\[\d+\]:", body[20:])
    if nxt:
        body = body[: 20 + nxt.start()]
    out: list[dict] = []
    for m in re.finditer(
        r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*\"([^\"]*)\"",
        body,
    ):
        sym = m.group(3)
        if not sym or sym in ("sil", "<eps>", "spn"):
            continue
        out.append({"symbol": sym, "start_s": float(m.group(1)), "end_s": float(m.group(2))})
    return out


def pool_frames_in_spans(layer_emb: np.ndarray, spans: list[tuple[float, float]]) -> list[np.ndarray]:
    """Mean-pool frames whose centre time falls in each span."""
    n_frames = layer_emb.shape[0]
    frame_times = np.arange(n_frames, dtype=np.float32) * (FRAME_STRIDE / TARGET_SR)
    out: list[np.ndarray] = []
    for start, end in spans:
        mask = half_open_span_mask(frame_times, start, end)
        if not np.any(mask):
            continue
        out.append(layer_emb[mask].mean(axis=0))
    return out


def _gaussian_kld_sym(x: np.ndarray, y: np.ndarray, n_components: int) -> float | None:
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    if len(x) < 2 or len(y) < 2:
        return None
    n_comp = min(n_components, len(x) + len(y) - 2, x.shape[1])
    if n_comp < 2:
        return None
    combined = np.vstack([x, y])
    proj = PCA(n_components=n_comp, random_state=RNG_SEED).fit_transform(combined)
    xp, yp = proj[: len(x)], proj[len(x):]
    mu_x, mu_y = xp.mean(axis=0), yp.mean(axis=0)
    cov_x = LedoitWolf().fit(xp).covariance_
    cov_y = LedoitWolf().fit(yp).covariance_

    def _kl(mu_a, cov_a, mu_b, cov_b) -> float | None:
        d = cov_a.shape[0]
        reg = max(1e-8, 1e-6 * (np.trace(cov_a) + np.trace(cov_b)) / (2.0 * d))
        cov_a_reg = cov_a + reg * np.eye(d)
        cov_b_reg = cov_b + reg * np.eye(d)
        sign_a, logdet_a = np.linalg.slogdet(cov_a_reg)
        sign_b, logdet_b = np.linalg.slogdet(cov_b_reg)
        if sign_a <= 0 or sign_b <= 0:
            return None
        try:
            trace_term = np.trace(np.linalg.solve(cov_b_reg, cov_a_reg))
            diff = mu_b - mu_a
            mean_term = diff @ np.linalg.solve(cov_b_reg, diff)
        except np.linalg.LinAlgError:
            return None
        value = 0.5 * (logdet_b - logdet_a - d + trace_term + mean_term)
        return float(value) if np.isfinite(value) else None

    kld_xy = _kl(mu_x, cov_x, mu_y, cov_y)
    kld_yx = _kl(mu_y, cov_y, mu_x, cov_x)
    if kld_xy is None or kld_yx is None:
        return None
    return float(0.5 * (kld_xy + kld_yx))


def _match_occurrences(
    natural: dict[str, Any],
    tts: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Match same-base-phone occurrences within each paired utterance."""
    nat_vecs = np.asarray(natural["vecs"])
    tts_vecs = np.asarray(tts["vecs"])
    nat_by_utt: dict[str, list[np.ndarray]] = defaultdict(list)
    tts_by_utt: dict[str, list[np.ndarray]] = defaultdict(list)
    for vec, utt in zip(nat_vecs, natural["utterances"]):
        nat_by_utt[str(utt)].append(vec)
    for vec, utt in zip(tts_vecs, tts["utterances"]):
        tts_by_utt[str(utt)].append(vec)
    x_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []
    g_out: list[str] = []
    for utt in sorted(set(nat_by_utt) & set(tts_by_utt)):
        n = min(len(nat_by_utt[utt]), len(tts_by_utt[utt]))
        x_out.extend(nat_by_utt[utt][:n])
        y_out.extend(tts_by_utt[utt][:n])
        g_out.extend([utt] * n)
    x = np.stack(x_out) if x_out else np.empty((0, nat_vecs.shape[1]))
    y = np.stack(y_out) if y_out else np.empty((0, tts_vecs.shape[1]))
    groups = np.asarray(g_out, dtype=object)
    return x, y, groups, len(x_out)


def _grouped_binary_metrics(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_components: int) -> dict:
    """Grouped natural-vs-TTS logistic probe with PCA fit inside each fold."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_comp = min(n_components, 15, max(2, len(x) + len(y) - 2))
    X = np.vstack([x, y])
    labels = np.array([0] * len(x) + [1] * len(y))
    if len(np.unique(labels)) < 2 or len(labels) < CV_SPLITS:
        return {"accuracy": None, "auc": None, "n_utterances": int(len(np.unique(groups)))}

    pipe = Pipeline([
        ("std", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=RNG_SEED)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    accs: list[float] = []
    aucs: list[float] = []
    for seed in range(CV_REPEATS):
        cv = GroupShuffleSplit(
            n_splits=CV_SPLITS,
            test_size=1.0 / CV_SPLITS,
            random_state=seed,
        )
        try:
            accs.extend(cross_val_score(pipe, X, labels, groups=groups, cv=cv, scoring="accuracy").tolist())
            aucs.extend(cross_val_score(pipe, X, labels, groups=groups, cv=cv, scoring="roc_auc").tolist())
        except ValueError:
            continue
    if not accs:
        return {"accuracy": None, "auc": None, "n_utterances": int(len(np.unique(groups)))}
    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "n_utterances": int(len(np.unique(groups))),
    }


def _spearman(a: list[float], b: list[float]) -> float | None:
    from scipy.stats import rankdata
    if len(a) < 3:
        return None
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    if np.isnan(rho):
        return None
    return rho


def _partial_spearman(r_ab: float, r_ac: float, r_bc: float) -> float:
    """Partial Spearman of a vs b controlling for c."""
    denom = np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))
    if denom == 0:
        return float("nan")
    return float((r_ab - r_ac * r_bc) / denom)


def gather_phone_vectors(
    entries: list[dict],
    model: str,
    layer: int,
    textgrid_dirs: dict[str, Path],
    f16: Any,
) -> dict[str, dict[str, dict]]:
    """Per base-phone vectors for each condition, pooled from MFA phone spans.

    Returns ``{condition: {base_phone: {"vecs": ndarray, "utterances": ndarray}}}``.
    """
    raw: dict[str, dict[str, dict]] = {
        cond: defaultdict(lambda: {"vecs": [], "utts": []}) for cond in ("natural", "tts")
    }
    for meta in entries:
        if meta.get("model") != model:
            continue
        if meta.get("variant") != "raw":
            continue
        analysis_cond = f16._analysis_condition(meta)
        cond = "natural" if analysis_cond == "natural" else "tts"
        sid = str(meta.get("paired_key", "?"))
        tg_cond = "tts" if cond == "tts" else "natural"
        tg = textgrid_dirs[tg_cond] / f"{sid}.TextGrid"
        if not tg.exists():
            continue
        phones = parse_textgrid_phones(tg)
        if not phones:
            continue
        saved = meta.get("layers", [])
        if layer not in saved:
            continue
        li = saved.index(layer)
        npy = Path(meta.get("_npy_path", ""))
        if not npy.exists():
            continue
        emb_all = np.load(npy)
        if li >= emb_all.shape[0]:
            continue
        spans = [(p["start_s"], p["end_s"]) for p in phones]
        vecs = pool_frames_in_spans(emb_all[li], spans)
        if len(vecs) != len(phones):
            continue  # frame/phone mismatch for this sample — skip entirely
        for p, v in zip(phones, vecs):
            base = strip_tone(p["symbol"])
            raw[cond][base]["vecs"].append(v)
            raw[cond][base]["utts"].append(sid)

    out: dict[str, dict[str, dict]] = {}
    for cond in raw:
        out[cond] = {}
        for base, acc in raw[cond].items():
            if not acc["vecs"]:
                continue
            out[cond][base] = {
                "vecs": np.stack(acc["vecs"]),
                "utterances": np.array(acc["utts"]),
            }
    return out


def _input_inventory(
    entries: list[dict],
    paired_keys: set[str],
    model: str,
) -> tuple[int, int, str]:
    rows: list[str] = []
    n_json = 0
    n_npy = 0
    for entry in entries:
        if entry.get("model") != model or entry.get("variant") != "raw":
            continue
        sid = str(entry.get("paired_key", entry.get("sample_id", "unknown")))
        if sid not in paired_keys:
            continue
        for key, label in (("_json_path", "json"), ("_npy_path", "npy")):
            path = Path(entry.get(key, ""))
            if not path.exists():
                raise FileNotFoundError(f"Missing embedding input: {path}")
            rows.append(f"{label}\t{path.name}\t{_sha256(path)}")
            if label == "json":
                n_json += 1
            else:
                n_npy += 1
    digest = hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()
    return n_json, n_npy, digest


def run(
    embeddings_dir: Path,
    model: str,
    layers: list[int],
    textgrid_dirs: dict[str, Path],
    min_class_samples: int,
    pca_dim: int,
    out_dir: Path,
    manifest_path: Path,
) -> int:
    f16 = _load_f16()
    entries = f16._discover_embedding_files(embeddings_dir)
    if not entries:
        logger.error("No embedding files in %s", embeddings_dir)
        return 1

    tts_conditions = sorted({f16._analysis_condition(e) for e in entries if f16._analysis_condition(e) != "natural"})
    if tts_conditions != ["f5_tts"]:
        raise ValueError(f"Expected exactly one MDC TTS provider f5_tts, found {tts_conditions}")
    natural_ids = {
        str(e.get("paired_key", e.get("sample_id", "unknown")))
        for e in entries
        if e.get("model") == model and e.get("variant") == "raw"
        and f16._analysis_condition(e) == "natural"
    }
    tts_ids = {
        str(e.get("paired_key", e.get("sample_id", "unknown")))
        for e in entries
        if e.get("model") == model and e.get("variant") == "raw"
        and f16._analysis_condition(e) == "f5_tts"
    }
    expected_ids = {f"en_{i:03d}" for i in range(1, 51)}
    if natural_ids != expected_ids or tts_ids != expected_ids:
        raise ValueError(
            "MDC B3 requires exact natural and f5_tts en_001..en_050 coverage; "
            f"natural_missing={sorted(expected_ids - natural_ids)} "
            f"natural_extra={sorted(natural_ids - expected_ids)} "
            f"tts_missing={sorted(expected_ids - tts_ids)} "
            f"tts_extra={sorted(tts_ids - expected_ids)}"
        )
    paired_keys = natural_ids & tts_ids
    n_json, n_npy, inventory_hash = _input_inventory(entries, paired_keys, model)

    per_layer: dict[int, dict] = {}
    exclusions_by_layer: dict[str, dict[str, str]] = {}
    support_inventory_by_layer: dict[str, dict[str, dict[str, int]]] = {}
    for layer in layers:
        data = gather_phone_vectors(entries, model, layer, textgrid_dirs, f16)
        classes = sorted(set(data["natural"]) | set(data["tts"]))
        class_out: dict[str, dict] = {}
        excluded: dict[str, str] = {}
        support_inventory: dict[str, dict[str, int]] = {}
        kld_list: list[float] = []
        auc_list: list[float] = []
        n_list: list[float] = []
        for base in classes:
            natural_entry = data["natural"].get(base)
            tts_entry = data["tts"].get(base)
            natural_raw = len(natural_entry["vecs"]) if natural_entry else 0
            tts_raw = len(tts_entry["vecs"]) if tts_entry else 0
            support_inventory[base] = {
                "natural_raw": int(natural_raw),
                "tts_raw": int(tts_raw),
                "matched": 0,
            }
            if natural_entry is None or tts_entry is None:
                excluded[base] = "missing_condition_support"
                continue
            x_raw, y_raw, groups, n = _match_occurrences(natural_entry, tts_entry)
            support_inventory[base]["matched"] = int(n)
            if natural_raw < min_class_samples or tts_raw < min_class_samples:
                excluded[base] = "raw_support_below_threshold"
                continue
            if n < min_class_samples:
                excluded[base] = "matched_support_below_threshold"
                continue
            kld = _gaussian_kld_sym(x_raw, y_raw, pca_dim)
            metrics = _grouped_binary_metrics(x_raw, y_raw, np.concatenate([groups, groups]), pca_dim)
            if kld is None or metrics["auc"] is None:
                excluded[base] = "nonfinite_kld_or_unavailable_auc"
                continue
            class_out[base] = {
                "natural_raw": int(natural_raw),
                "tts_raw": int(tts_raw),
                "matched": int(n),
                "n": int(n),
                "vowel": is_vowel(base),
                "kld_sym": kld,
                "accuracy": metrics["accuracy"],
                "accuracy_std": metrics["accuracy_std"],
                "auc": metrics["auc"],
                "auc_std": metrics["auc_std"],
                "n_utterances": metrics["n_utterances"],
            }
            kld_list.append(kld)
            auc_list.append(metrics["auc"])
            n_list.append(float(n))

        # Overall + consonant/vowel Spearman (KLD vs AUC), with sample-size
        # control (partial Spearman controlling for n).
        rho_all = _spearman(kld_list, auc_list)
        rho_n_kld = _spearman(n_list, kld_list)
        rho_n_auc = _spearman(n_list, auc_list)
        rho_partial = None
        if rho_all is not None and rho_n_kld is not None and rho_n_auc is not None:
            rho_partial = _partial_spearman(rho_all, rho_n_kld, rho_n_auc)

        def _group_rho(view_dict: dict[str, dict]) -> float | None:
            kl = [c["kld_sym"] for c in view_dict.values()]
            au = [c["auc"] for c in view_dict.values()]
            return _spearman(kl, au)

        rho_vowel = _group_rho({b: c for b, c in class_out.items() if c["vowel"]})
        rho_cons = _group_rho({b: c for b, c in class_out.items() if not c["vowel"]})

        aspiration = {}
        for a, b in ASPIRATION_PAIRS:
            row = {}
            for ph in (a, b):
                if ph in class_out:
                    c = class_out[ph]
                    row[ph] = {"kld": c["kld_sym"], "auc": c["auc"], "n": c["n"]}
            if row:
                aspiration[f"{a}/{b}"] = row

        per_layer[layer] = {
            "n_classes": len(class_out),
            "n_classes_tested": len(class_out),
            "n_classes_excluded": len(excluded),
            "spearman_kld_auc": rho_all,
            "spearman_kld_auc_vowel": rho_vowel,
            "spearman_kld_auc_consonant": rho_cons,
            "spearman_n_kld": rho_n_kld,
            "spearman_n_auc": rho_n_auc,
            "spearman_kld_auc_partial_n": rho_partial,
            "classes": class_out,
            "exclusions": excluded,
            "aspiration_pairs": aspiration,
        }
        exclusions_by_layer[str(layer)] = excluded
        support_inventory_by_layer[str(layer)] = support_inventory
        logger.info(
            "[%s] %s: %d tested, %d excluded, Spearman(KLD,AUC)=%s partial(n)=%s (V %s / C %s)",
            model, layer, len(class_out), len(excluded),
            "%.3f" % rho_all if rho_all is not None else "n/a",
            "%.3f" % rho_partial if rho_partial is not None else "n/a",
            "%.3f" % rho_vowel if rho_vowel is not None else "n/a",
            "%.3f" % rho_cons if rho_cons is not None else "n/a",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "dataset": "mdc_tts",
            "run_id": "mdc_en_phoneme_20260807_f5_full",
            "model": model,
            "layers": layers,
            "min_class_samples": min_class_samples,
            "pca_dim": pca_dim,
            "seed": RNG_SEED,
            "mfa_textgrid_dirs": {key: str(path) for key, path in textgrid_dirs.items()},
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "helper_script_path": str(PROJECT_ROOT / "scripts/16_feature_separability.py"),
            "helper_script_sha256": _sha256(PROJECT_ROOT / "scripts/16_feature_separability.py"),
            "embedding_dir": str(embeddings_dir),
            "embedding_input_json_count": n_json,
            "embedding_input_npy_count": n_npy,
            "embedding_input_inventory_sha256": inventory_hash,
            "conditions": ["natural", "f5_tts"],
            "tts_provider": "f5_tts",
            "paired_keys": sorted(paired_keys),
            "n_paired_keys": len(paired_keys),
            "n_records": len(entries),
            "unit": "MFA phone intervals (TextGrid phones tier), tone-stripped base phone",
            "classifier": "LR on PCA, GroupShuffleSplit by paired utterance, accuracy+AUC",
            "matching": "same-base-phone occurrence order within paired utterance; KLD and probe use matched support",
            "support": "raw natural/TTS support and matched support are recorded; classes below threshold are explicit exclusions",
            "exclusions_by_layer": exclusions_by_layer,
            "support_inventory_by_layer": support_inventory_by_layer,
        },
        "results": {str(l): v for l, v in per_layer.items()},
    }
    out_json = out_dir / "per_phoneme_kld.json"
    out_json.write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out_json)

    # ---- Console summary --------------------------------------------------
    print("\n=== B3 Per-phoneme KLD (MFA phone-level) ===")
    for layer, lres in per_layer.items():
        print("\n[l%d]  %d classes" % (layer, lres["n_classes"]))
        r = lres["spearman_kld_auc"]
        rp = lres["spearman_kld_auc_partial_n"]
        rv = lres["spearman_kld_auc_vowel"]
        rc = lres["spearman_kld_auc_consonant"]
        print("  Spearman(KLD,AUC)=%s  partial(n)=%s  V=%s C=%s"
              % ("%.3f" % r if r is not None else "n/a",
                 "%.3f" % rp if rp is not None else "n/a",
                 "%.3f" % rv if rv is not None else "n/a",
                 "%.3f" % rc if rc is not None else "n/a"))
        top = sorted(lres["classes"].items(), key=lambda kv: -kv[1]["kld_sym"])[:8]
        for cls, c in top:
            print("    %-5s %-4s KLD=%.2f AUC=%.2f acc=%.2f (n=%d)"
                  % (cls, "V" if c["vowel"] else "C", c["kld_sym"], c["auc"],
                     c["accuracy"], c["n"]))
        if lres.get("aspiration_pairs"):
            print("  aspiration pairs:")
            for pair, row in lres["aspiration_pairs"].items():
                parts = "  ".join("%s:K%.2f/AUC%.2f/n%d" % (ph, r_["kld"], r_["auc"], r_["n"])
                                  for ph, r_ in row.items())
                print("    %-8s %s" % (pair, parts))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--model", type=str, default="hubert")
    parser.add_argument("--layers", type=str, default="0,6,11,12")
    parser.add_argument("--min-class-samples", type=int, default=_MIN_SAMPLES)
    parser.add_argument("--pca-dim", type=int, default=_PCA_DIM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--natural-tg", type=Path, default=DEFAULT_TEXTGRID_DIRS["natural"])
    parser.add_argument("--tts-tg", type=Path, default=DEFAULT_TEXTGRID_DIRS["tts"])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    return parser.parse_args(argv)
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    tg_dirs = {"natural": args.natural_tg, "tts": args.tts_tg}
    return run(
        args.embeddings_dir, args.model, layers, tg_dirs,
        args.min_class_samples, args.pca_dim, args.out_dir, args.manifest,
    )


if __name__ == "__main__":
    sys.exit(main())
