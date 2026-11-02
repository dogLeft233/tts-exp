#!/usr/bin/env python
"""Build metric-suite manifests for the three finding-lines.

Families
--------
A1  LeapTalk cross-dataset TFG, 250 paired samples, natural vs TTS audio
    (runs/leaptalk_multiset).  Real source videos exist in data/dataset_samples.
A2  AISHELL-1 13-sample core set, 12 TFG models, natural vs TTS audio
    (results/<model>/{natural_raw,tts_raw}).
B   SyncTalk May encoder-gain factorial: trained norm {off,on} x train gain
    {-6,+6} x inference norm {off,on} x test gain {-6,0,+6}.
C   SyncTalk May post-norm scale: trained scale {0.50,1.00} x inference scale
    {0.50,1.00,2.00}.

Usage:  python scripts/42_build_video_metric_manifests.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
F_DATA = Path("F:/Datasets/tts-exp")

A2_MODELS = [
    "avatarforcing", "avtr1", "echomimic_v2", "hallo2", "hallo2_no_separator",
    "imtalker", "joyvasa", "latentsync", "leaptalk", "soulx_flashhead",
    "stableavatar", "vexpress",
]


def family_a2() -> list[dict]:
    entries = []
    for model in A2_MODELS:
        for cond in ("natural_raw", "tts_raw"):
            d = REPO / "results" / model / cond
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.mp4")):
                entries.append({"group": f"A2/{model}/{cond}", "item": p.stem,
                                "video": p.relative_to(REPO).as_posix()})
    return entries


def family_a1() -> list[dict]:
    entries = []
    base = REPO / "runs" / "leaptalk_multiset"
    for cond in ("natural_raw", "tts_raw"):
        d = base / cond
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.mp4"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
            entries.append({"group": f"A1/{cond}", "item": p.stem,
                            "video": p.relative_to(REPO).as_posix()})
    return entries


def family_b() -> list[dict]:
    entries = []
    root = F_DATA / "may_encoder_gain_norm_factorial_20261019" / "remote" / "renders"
    for model_dir in sorted(root.glob("wav2vec2_base_*")):
        m = re.match(r"wav2vec2_base_(off|on)_train(m6|p6|0)$", model_dir.name)
        if not m:
            continue
        trained_norm, train_gain = m.group(1), m.group(2)
        for cell in sorted(model_dir.glob("*_g*")):
            cm = re.match(r"(off|on)_g([+-]\d+|\+0)$", cell.name)
            if not cm:
                continue
            infer_norm, test_gain = cm.group(1), cm.group(2).replace("+", "p").replace("-", "m")
            for clip_dir in sorted(cell.glob("test_*")):
                for p in sorted(clip_dir.glob("*.mkv")):
                    entries.append({
                        "group": f"B/{trained_norm}norm_train{train_gain}/{infer_norm}norm_test{test_gain}",
                        "item": clip_dir.name,
                        "video": str(p),
                    })
    return entries


def family_c() -> list[dict]:
    entries = []
    root = F_DATA / "may_postnorm_scale_20261020" / "remote" / "renders"
    for model_dir in sorted(root.glob("*_base_scale*")):
        m = re.match(r"(hubert|wav2vec2)_base_scale(\d+)$", model_dir.name)
        if not m:
            continue
        enc, train_scale = m.group(1), m.group(2)
        for cell in sorted(model_dir.glob("s*")):
            for clip_dir in sorted(cell.glob("test_*")):
                for p in sorted(clip_dir.glob("*.mkv")):
                    entries.append({
                        "group": f"C/{enc}_train{train_scale}/{cell.name}",
                        "item": clip_dir.name,
                        "video": str(p),
                    })
    return entries


FAMILIES = {"a1": family_a1, "a2": family_a2, "b": family_b, "c": family_c}


def main() -> int:
    global REPO, F_DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="a2,b,c,a1")
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--data-root", type=Path, default=F_DATA)
    args = ap.parse_args()
    REPO, F_DATA = args.repo_root.resolve(), args.data_root.resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fam in [f.strip() for f in args.families.split(",") if f.strip()]:
        entries = FAMILIES[fam]()
        groups = sorted({e["group"] for e in entries})
        manifest = {"family": fam, "n_entries": len(entries), "groups": groups,
                    "entries": entries}
        path = out / f"manifest_{fam}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        print(f"{fam}: {len(entries)} videos, {len(groups)} groups -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
