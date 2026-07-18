#!/usr/bin/env python3
"""Extract oracle Fs = BERT CLS / mean-pooled sentence embeddings.

Per the design spec Section 6.3, this script encodes each sample's gold
transcript (lowercased) with ``bert-base-uncased`` and stores the CLS and
mean-pooled token vectors as two Fs candidates. It also computes:
  - per-sample ||Fs|| and L1 distance to BERT("") baseline (M2 detector)
  - paired natural↔TTS cosine similarity (and L1)

Usage
-----
    python scripts/29_oracle_fs_bert.py [--samples IDS] [--smoke] \
        [--bert-model bert-base-uncased] [--device cuda] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import ENGLISH_STUDY_SAMPLES, OUTPUT_BASE_EN

ALIGNMENT_MANIFEST_REL = OUTPUT_BASE_EN / "manifest" / "alignment.json"
OUTPUT_PATH = OUTPUT_BASE_EN / "metrics" / "oracle_fs.json"

DEFAULT_BERT_MODEL = "bert-base-uncased"
DEFAULT_CACHE_DIR = Path("/root/autodl-tmp/checkpoints")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-10
    nb = np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / (na * nb))


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=1))


# ---------------------------------------------------------------------------
# BERT loading
# ---------------------------------------------------------------------------


def load_bert_model(
    model_name: str = DEFAULT_BERT_MODEL,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> tuple:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(f"transformers/torch required: {e}")

    kwargs = {"local_files_only": False}
    if cache_dir is not None:
        # transformers resolves from cache_dir first; if HF_ENDPOINT is set,
        # downloads via mirror on first call.
        kwargs["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Fs extraction
# ---------------------------------------------------------------------------


def encode_text(text: str, model, tokenizer, device: str = "cpu", max_length: int = 128):
    """Encode a text into a BERT forward pass; returns (outputs, inputs)."""
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs, inputs


def extract_fs_from_outputs(outputs) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pull CLS and mean-pooled Fs from a BERT ModelOutput.

    last_hidden_state shape: (1, T, 768). We take index 0 (CLS) and the
    mean of indices 1..T-2 (token positions, excluding special tokens).
    """
    lhs = outputs.last_hidden_state  # (1, T, 768)
    Fs_cls = lhs[0, 0].cpu().numpy().astype(np.float32)
    T = lhs.shape[1]
    if T > 2:
        Fs_mean = lhs[0, 1:-1].mean(0).cpu().numpy().astype(np.float32)
        token_count = T - 2
    else:
        Fs_mean = Fs_cls.copy()
        token_count = 0
    return Fs_cls, Fs_mean, {"token_count": int(token_count)}


def compute_fs_diagnostics(fs: np.ndarray, fs_empty_baseline: np.ndarray) -> dict:
    return {
        "fs_norm": float(np.linalg.norm(fs)),
        "fs_empty_l1": l1_distance(fs, fs_empty_baseline),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def process_sample(
    sample_id: int,
    condition: str,
    text: str,
    model,
    tokenizer,
    device: str,
    fs_empty: np.ndarray,
) -> dict:
    text_lower = text.lower()
    outputs, _ = encode_text(text_lower, model, tokenizer, device)
    Fs_cls, Fs_mean, meta = extract_fs_from_outputs(outputs)
    diag = compute_fs_diagnostics(Fs_cls, fs_empty)
    return {
        "sample_id": sample_id,
        "condition": condition,
        "text": text,
        "text_lower": text_lower,
        "token_count": meta["token_count"],
        "Fs_cls": Fs_cls.tolist(),
        "Fs_mean": Fs_mean.tolist(),
        "fs_cls_norm": diag["fs_norm"],
        "fs_empty_l1": diag["fs_empty_l1"],
        "is_degenerate": False,  # set later in batch stats pass
    }


def compute_nt_similarity(entries: list[dict]) -> list[dict]:
    """For each sample_id, compute natural↔TTS Fs similarity."""
    by_sid: dict[int, dict[str, dict]] = {}
    for e in entries:
        by_sid.setdefault(e["sample_id"], {})[e["condition"]] = e

    out: list[dict] = []
    for sid in sorted(by_sid):
        arms = by_sid[sid]
        if "natural" not in arms or "tts" not in arms:
            continue
        Fs_nat_cls = np.asarray(arms["natural"]["Fs_cls"], dtype=np.float32)
        Fs_tts_cls = np.asarray(arms["tts"]["Fs_cls"], dtype=np.float32)
        Fs_nat_mean = np.asarray(arms["natural"]["Fs_mean"], dtype=np.float32)
        Fs_tts_mean = np.asarray(arms["tts"]["Fs_mean"], dtype=np.float32)
        out.append({
            "sample_id": sid,
            "cosine_cls_nt": cosine_similarity(Fs_nat_cls, Fs_tts_cls),
            "cosine_mean_nt": cosine_similarity(Fs_nat_mean, Fs_tts_mean),
            "l1_cls_nt": l1_distance(Fs_nat_cls, Fs_tts_cls),
        })
    return out


def flag_degenerate(entries: list[dict]) -> list[dict]:
    """Mark is_degenerate=True for entries with ||Fs|| outside ±5σ of distribution."""
    norms = np.array([e["fs_cls_norm"] for e in entries], dtype=np.float64)
    mu = float(norms.mean()) if len(norms) > 0 else 0.0
    sigma = float(norms.std(ddof=0)) if len(norms) > 1 else 0.0
    threshold_hi = mu + 5 * sigma if sigma > 0 else float("inf")
    threshold_lo = mu - 5 * sigma if sigma > 0 else float("-inf")
    for e in entries:
        bad_norm = not (threshold_lo < e["fs_cls_norm"] < threshold_hi)
        bad_empty = e["fs_empty_l1"] < 0.5
        e["is_degenerate"] = bool(bad_norm or bad_empty)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=str,
        default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--alignment-manifest", type=str,
                        default=str(ALIGNMENT_MANIFEST_REL))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [int(x) for x in args.samples.split(",") if x.strip()]
    repo_root = Path(__file__).resolve().parent.parent

    alignment_path = Path(args.alignment_manifest)
    if not alignment_path.is_absolute():
        alignment_path = repo_root / alignment_path
    if not alignment_path.exists():
        print(f"[29] alignment manifest not found: {alignment_path}", file=sys.stderr)
        return 1
    alignment = json.loads(alignment_path.read_text())
    entries_all = alignment.get("manifest", alignment)  # support both shapes

    # Filter to requested sample_ids and only carry forward entries with text
    target_entries = [
        e for e in entries_all
        if int(e["sample_id"]) in sample_ids and e.get("text")
    ]

    output_path = Path(args.output)
    if output_path.exists() and not args.no_cache:
        # Check mtime vs alignment manifest
        if output_path.stat().st_mtime >= alignment_path.stat().st_mtime:
            print(f"[29] cached: {output_path}")
            return 0

    print(f"[29] loading {args.bert_model} on {args.device} ...")
    model, tokenizer = load_bert_model(
        args.bert_model, device=args.device, cache_dir=Path(args.cache_dir),
    )

    # Empty-text baseline Fs for M2 degeneracy detector
    empty_outputs, _ = encode_text("", model, tokenizer, args.device)
    Fs_empty_cls, _, _ = extract_fs_from_outputs(empty_outputs)

    entries_out: list[dict] = []
    for e in target_entries:
        out = process_sample(
            e["sample_id"], e["condition"], e["text"],
            model, tokenizer, args.device, Fs_empty_cls,
        )
        entries_out.append(out)

    entries_out = flag_degenerate(entries_out)
    nt_sim = compute_nt_similarity(entries_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "bert_model": args.bert_model,
        "device": args.device,
        "n_entries": len(entries_out),
        "n_degenerate": sum(1 for e in entries_out if e["is_degenerate"]),
        "entries": entries_out,
        "nt_similarity": nt_sim,
    }, ensure_ascii=False, indent=2))
    print(f"[29] wrote {len(entries_out)} entries → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
