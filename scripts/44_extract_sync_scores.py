#!/usr/bin/env python
"""Extract per-sample Sync-C into the {group: {item: c}} form used by
43_aggregate_video_metrics.py --sync-json.

Sources
-------
A1  runs/leaptalk_eval/04_eval/scores_leaptalk.json      (list of i/cond/c/d)
A2  results/<model>/04_eval/eval_meta.json               (results["cond:id"])

Usage: python scripts/44_extract_sync_scores.py --family a1 --out sync_a1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
F_DATA = Path("F:/Datasets/tts-exp")
A2_MODELS = [
    "avatarforcing", "avtr1", "echomimic_v2", "hallo2", "hallo2_no_separator",
    "imtalker", "joyvasa", "latentsync", "leaptalk", "soulx_flashhead",
    "stableavatar", "vexpress",
]


def a1() -> dict:
    p = REPO / "runs" / "leaptalk_eval" / "04_eval" / "scores_leaptalk.json"
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    out: dict[str, dict[str, float]] = {"A1/natural_raw": {}, "A1/tts_raw": {}}
    for row in data:
        cond = row.get("cond")
        if cond not in ("natural_raw", "tts_raw"):
            continue
        c = row.get("c")
        if c is None:
            continue
        out[f"A1/{cond}"][str(row["i"])] = float(c)
    return out


def a2() -> dict:
    out: dict[str, dict[str, float]] = {}
    for model in A2_MODELS:
        meta = REPO / "results" / model / "04_eval" / "eval_meta.json"
        if not meta.exists():
            continue
        data = json.loads(meta.read_text(encoding="utf-8-sig"))
        for key, rec in (data.get("results") or {}).items():
            cond = rec.get("condition") or key.split(":")[0]
            sid = rec.get("sample_id")
            c = rec.get("sync_c")
            if sid is None or c is None:
                continue
            out.setdefault(f"A2/{model}/{cond}", {})[str(sid)] = float(c)
    return out


def _guard20_cells(path: Path) -> dict:
    """Map group name -> {clip_id: sync_c} from a scores results.json."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data.get("cells", [])


def b() -> dict:
    p = (F_DATA / "may_encoder_gain_norm_factorial_20261019" / "remote" / "scores"
         / "results.json")
    out: dict[str, dict[str, float]] = {}
    for cell in _guard20_cells(p):
        arm = cell["arm"]
        tnorm = "on" if arm.get("norm") else "off"
        tgain = "m6" if arm.get("train_gain_db") == -6 else "p6"
        inorm = "on" if cell.get("test_normalize") else "off"
        g = cell.get("gain_db")
        sgain = {(-6): "m6", 0: "p0", 6: "p6"}.get(g)
        if sgain is None:
            continue
        group = f"B/{tnorm}norm_train{tgain}/{inorm}norm_test{sgain}"
        for clip in cell.get("clips", []):
            c = (clip.get("guard20") or {}).get("sync_c")
            if c is not None:
                out.setdefault(group, {})[clip["clip_id"]] = float(c)
    return out


def c() -> dict:
    p = F_DATA / "may_postnorm_scale_20261020" / "remote" / "scores" / "results.json"
    out: dict[str, dict[str, float]] = {}
    for cell in _guard20_cells(p):
        arm = cell["arm"]
        enc = "hubert" if "hubert" in arm.get("encoder", "") else "wav2vec2"
        train_scale = int(round(float(arm.get("input_scale", 1.0)) * 100))
        test_scale = cell.get("input_scale")
        if test_scale is None:
            continue
        group = f"C/{enc}_train{train_scale:03d}/s{float(test_scale):.2f}"
        for clip in cell.get("clips", []):
            cc = (clip.get("guard20") or {}).get("sync_c")
            if cc is not None:
                out.setdefault(group, {})[clip["clip_id"]] = float(cc)
    return out


def main() -> int:
    global REPO, F_DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["a1", "a2", "b", "c"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--data-root", type=Path, default=F_DATA)
    args = ap.parse_args()
    REPO, F_DATA = args.repo_root.resolve(), args.data_root.resolve()
    data = {"a1": a1, "a2": a2, "b": b, "c": c}[args.family]()
    Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    n = {k: len(v) for k, v in data.items()}
    print(f"{args.family}: {len(data)} groups, samples per group: "
          f"{sorted(set(n.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
