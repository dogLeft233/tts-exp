#!/usr/bin/env python3
"""B5 — Speaker/phoneme disentanglement control for natural-vs-TTS separability.

The natural pool mixes 100 speakers (1 utterance each) while the TTS pool is a
single synthetic voice.  This script quantifies how much of the phoneme
separability difference between conditions could be an artefact of that
speaker mismatch:

  1. **Speaker probe** (natural only): grouped logistic probe predicting which
     of the 100 speakers a token belongs to — how dominant speaker identity is
     at each layer.
  2. **Phoneme probe** (per condition): grouped logistic probe predicting the
     phoneme/viseme class (the metric B2 used for layer selection).
  3. **Speaker-orthogonalised phoneme probe** (natural): regress each token on
     the speaker one-hot (ridge), take residuals, re-run the phoneme probe —
     how much phoneme structure survives removing speaker identity.
  4. **Paired per-sample robustness**: the 09 experiment's natural-vs-TTS
     conclusions rest on within-utterance paired permutation tests, which are
     already speaker-controlled (each pair shares utterance content).  Report
     the pooled probe difference both raw and after speaker control.

Usage
-----
    python scripts/39_ph_spk_disentangle.py [--layers 7,10] [--level viseme]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "results/aishell100_phoneme/embeddings"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/aishell100_phoneme/ph_spk"

FRAME_STRIDE = 320
TARGET_SR = 16000
RNG_SEED = 42
CV_REPEATS = 3


def _load_f16() -> Any:
    spec = importlib.util.spec_from_file_location(
        "feature_separability_16", str(PROJECT_ROOT / "scripts/16_feature_separability.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grouped_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_components: int = 20,
    n_repeats: int = CV_REPEATS,
) -> dict:
    """Grouped (by utterance) linear probe: accuracy + AUC with spread.

    PCA + ridge inside a Pipeline; GroupShuffleSplit by utterance.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2 or len(x) < 6:
        return {"accuracy": None, "auc": None}
    n_comp = min(n_components, 15, max(2, len(x) - 2))
    pipe = Pipeline([
        ("std", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=RNG_SEED)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    accs: list[float] = []
    aucs: list[float] = []
    n_classes = len(np.unique(y))
    binary = n_classes == 2
    for seed in range(n_repeats):
        cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        try:
            accs.extend(cross_val_score(pipe, x, y, groups=groups, cv=cv, scoring="accuracy").tolist())
            if binary:
                aucs.extend(cross_val_score(pipe, x, y, groups=groups, cv=cv, scoring="roc_auc").tolist())
        except ValueError:
            continue
    if not accs:
        return {"accuracy": None, "auc": None}
    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "auc": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
    }


def _orthogonalize_means(x: np.ndarray, group_idx: np.ndarray) -> np.ndarray:
    """Subtract each group's mean (exact OLS projection on group one-hot).

    Equivalent to ``(I - S(S'S)^{-1}S')X``: removes the per-group baseline
    (per-utterance / per-speaker) from every token.
    """
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    S = enc.fit_transform(group_idx.reshape(-1, 1)).astype(np.float64)
    # S(S'S)^{-1}S' is the group-means projector (S'S is diagonal for one-hot).
    proj = S @ np.linalg.pinv(S.T @ S) @ S.T
    resid = x.astype(np.float64) - proj @ x
    return resid.astype(np.float32)


def _rv_coefficient(x1: np.ndarray, x2: np.ndarray) -> float:
    """RV coefficient between two centred matrices (0..1)."""
    x1c = x1 - x1.mean(axis=0)
    x2c = x2 - x2.mean(axis=0)
    sxx = float(np.trace(x1c @ x1c.T @ x1c @ x1c.T))
    syy = float(np.trace(x2c @ x2c.T @ x2c @ x2c.T))
    sxy = float(np.trace(x1c @ x1c.T @ x2c @ x2c.T))
    if sxx == 0 or syy == 0:
        return float("nan")
    return float(sxy / np.sqrt(sxx * syy))


def _fitted_ols(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-group-mean matrix: OLS projection of embeddings on group one-hot."""
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    S = enc.fit_transform(labels.reshape(-1, 1)).astype(np.float64)
    proj = S @ np.linalg.pinv(S.T @ S) @ S.T
    return proj @ emb


def _rv_with_permutation(
    nat_x: np.ndarray,
    nat_lab: np.ndarray,
    nat_utt: np.ndarray,
    n_perm: int = 200,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """RV(Ph, Spk) plus a permutation null (shuffle utterance labels)."""
    f_ph = _fitted_ols(nat_x, nat_lab)
    f_spk = _fitted_ols(nat_x, nat_utt)
    obs = _rv_coefficient(f_ph, f_spk)
    rng = np.random.default_rng(seed)
    utt_arr = nat_utt.copy()
    nulls: list[float] = []
    for _ in range(n_perm):
        rng.shuffle(utt_arr)
        nulls.append(_rv_coefficient(f_ph, _fitted_ols(nat_x, utt_arr)))
    p = float((1 + sum(1 for v in nulls if v >= obs)) / (1 + n_perm))
    return obs, p, float(np.median(nulls))


def _cramers_v(labels: np.ndarray, groups: np.ndarray) -> float:
    """Cramér's V between phoneme labels and utterance/speaker labels (0..1)."""
    from sklearn.metrics.cluster import contingency_matrix

    C = np.asarray(contingency_matrix(labels, groups), dtype=np.float64)
    if C.size == 0:
        return float("nan")
    n = float(C.sum())
    if n == 0:
        return float("nan")
    # chi2 from the contingency table (no continuity correction).
    row = C.sum(axis=1, keepdims=True)
    col = C.sum(axis=0, keepdims=True)
    expected = (row @ col) / n
    chi2 = float(np.sum((C - expected) ** 2 / np.where(expected == 0, 1.0, expected)))
    phi2 = chi2 / n
    r, k = C.shape
    denom = min(k - 1, r - 1)
    v = np.sqrt(phi2 / denom) if denom > 0 else float("nan")
    return float(v)


def gather_tokens(entries: list[dict], model: str, layer: int, level: str, cond: str, f16: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool per-token vectors + phoneme labels + speaker/utterance labels."""
    vecs: list[np.ndarray] = []
    labs: list[str] = []
    utts: list[str] = []
    for meta in entries:
        if meta.get("model") != model:
            continue
        if f16._analysis_condition(meta) != cond:
            continue
        tokens = meta.get("tokens", [])
        if not tokens:
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
        pooled, labels = f16._pool_frames_for_layer(emb_all[li], FRAME_STRIDE, TARGET_SR, tokens)
        if pooled.shape[0] == 0 or level not in labels:
            continue
        vecs.append(pooled)
        labs.append(labels[level])
        utts.extend([str(meta.get("paired_key", "?"))] * pooled.shape[0])
    if not vecs:
        return np.empty((0, 0)), np.array([], dtype=str), np.array([], dtype=str)
    return (
        np.concatenate(vecs, axis=0),
        np.concatenate(labs, axis=0),
        np.array(utts),
    )


def run(
    embeddings_dir: Path,
    models: list[str],
    layers: list[int],
    level: str,
    out_dir: Path,
) -> int:
    f16 = _load_f16()
    entries = f16._discover_embedding_files(embeddings_dir)
    if not entries:
        logger.error("No embedding files in %s", embeddings_dir)
        return 1

    results: dict[str, dict] = {}
    for model in models:
        for layer in layers:
            nat_x, nat_lab, nat_utt = gather_tokens(entries, model, layer, level, "natural", f16)
            tts_x, tts_lab, tts_utt = gather_tokens(entries, model, layer, level, "faster_qwen3", f16)
            if nat_x.shape[0] == 0 or tts_x.shape[0] == 0:
                logger.warning("No tokens for %s l%s", model, layer)
                continue

            # Phoneme probe per condition (raw).
            nat_ph_probe = _grouped_probe(nat_x, nat_lab, nat_utt)
            tts_ph_probe = _grouped_probe(tts_x, tts_lab, tts_utt)

            # Symmetric speaker/utterance control: residualise BOTH conditions
            # by their own per-utterance baseline.
            nat_resid = _orthogonalize_means(nat_x, nat_utt)
            tts_resid = _orthogonalize_means(tts_x, tts_utt)
            nat_ph_probe_ctrl = _grouped_probe(nat_resid, nat_lab, nat_utt)
            tts_ph_probe_ctrl = _grouped_probe(tts_resid, tts_lab, tts_utt)

            # Phoneme × speaker label independence (Cramér's V) on natural:
            # near 0 means the AISHELL design gives no phoneme-speaker confound.
            cramers_v = _cramers_v(nat_lab, nat_utt)

            # RV(Ph, Spk) with permutation null: is the phoneme-speaker subspace
            # overlap distinguishable from chance?
            rv, rv_p, rv_null_med = _rv_with_permutation(nat_x, nat_lab, nat_utt)

            results[f"{model}_l{layer}"] = {
                "model": model,
                "layer": layer,
                "n_natural_tokens": int(nat_x.shape[0]),
                "n_tts_tokens": int(tts_x.shape[0]),
                "n_natural_utterances": int(len(np.unique(nat_utt))),
                "n_natural_phoneme_classes": int(len(np.unique(nat_lab))),
                "natural_phoneme_probe": nat_ph_probe,
                "tts_phoneme_probe": tts_ph_probe,
                "natural_phoneme_probe_speaker_controlled": nat_ph_probe_ctrl,
                "tts_phoneme_probe_utterance_controlled": tts_ph_probe_ctrl,
                "cramers_v_ph_spk": cramers_v,
                "ph_spk_rv": rv,
                "ph_spk_rv_p_perm": rv_p,
                "ph_spk_rv_null_median": rv_null_med,
            }
            d = results[f"{model}_l{layer}"]
            logger.info(
                "[%s l%s] ph-nat=%.3f ph-tts=%.3f | ctrl-nat=%.3f ctrl-tts=%.3f | CramV=%.3f | RV=%.3f (p=%.2f)",
                model, layer,
                d["natural_phoneme_probe"]["accuracy"] or float("nan"),
                d["tts_phoneme_probe"]["accuracy"] or float("nan"),
                d["natural_phoneme_probe_speaker_controlled"]["accuracy"] or float("nan"),
                d["tts_phoneme_probe_utterance_controlled"]["accuracy"] or float("nan"),
                cramers_v, rv, rv_p,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "ph_spk_disentangle.json"
    out_json.write_text(json.dumps({
        "meta": {"models": models, "layers": layers, "level": level},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    # ---- Console summary --------------------------------------------------
    print("\n=== B5 Ph/Spk disentanglement ===")
    for key, d in sorted(results.items()):
        np_ = d["natural_phoneme_probe"]
        tp = d["tts_phoneme_probe"]
        nc = d["natural_phoneme_probe_speaker_controlled"]
        tc = d["tts_phoneme_probe_utterance_controlled"]
        print("%-12s ph-nat=%.3f ph-tts=%.3f | ctrl-nat=%.3f ctrl-tts=%.3f | CramV=%.3f | RV=%.3f (p=%.2f)" % (
            key, np_["accuracy"] or 0, tp["accuracy"] or 0,
            nc["accuracy"] or 0, tc["accuracy"] or 0,
            d["cramers_v_ph_spk"], d["ph_spk_rv"], d["ph_spk_rv_p_perm"]))
        print("   nat-tts diff raw=%.3f  symmetric-ctrl=%.3f" % (
            (np_["accuracy"] or 0) - (tp["accuracy"] or 0),
            (nc["accuracy"] or 0) - (tc["accuracy"] or 0)))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--models", type=str, default="hubert,xlsr")
    parser.add_argument("--layers", type=str, default="6,10")
    parser.add_argument("--level", type=str, default="viseme")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    return run(args.embeddings_dir, models, layers, args.level, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
