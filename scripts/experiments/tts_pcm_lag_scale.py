"""No-render diagnostic of the time scale used by SyncNet's background term."""

from __future__ import annotations

import json

import numpy as np

from scripts.experiments.tts_clock_cross import check_hash, read, sha, write
from scripts.experiments.tts_pcm_residual import OUT, cluster


def main():
    base = OUT / "lag_scale"
    protocol = {
        "parent_sha256": sha(OUT / "protocol.json"),
        "reason": "All TTS shorter; native delta C mostly background B. Is B sensitive to using absolute instead of relative utterance lag?",
        "definition": "Reuse exact source features, no audio/video modification. Original distance curve at integer lags -ceil(K)..ceil(K), K=max(15,15*duration_arm/geomean_pair_duration). Interpolate mean distance curve at 31 equally spaced lags +/-15*duration_arm/geomean_pair_duration for normalized B. Hold D at original min over integer -15..15. Cstar=Bnormalized-Doriginal is explicitly a diagnostic, NOT official Sync-C.",
        "support": "both original and normalized term same rows, per-arm guard=max(20,ceil(K)); sensitivity max(15,ceil(K)); no zero padding",
        "selection": "all74; no score selection; fixed before reading speed-intervention outcomes",
        "limits": "changes metric definition; measures background-scale dependence, not perceptual or generation improvement; cannot infer explained causal percentage",
    }
    path = base / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("lag protocol differs")
    write(path, protocol)
    rows = []
    for row in read(OUT / "protocol.json")["rows"]:
        sid = row["id"]
        fp = OUT / "features" / sid / "features.npz"
        check_hash(fp, read(OUT / "scores" / f"{sid}.json")["features_sha256"])
        f = np.load(fp)
        d = row["source_lengths"]
        gm = np.sqrt(d["N"] * d["T"])
        cells = {}
        for a in ("N", "T"):
            v = f["v_" + a]
            audio = f["a_" + a + "_source"]
            n = min(len(v), len(audio))
            limit = 15 * d[a] / gm
            k = int(np.ceil(max(15, limit)))
            for guard in (15, 20):
                g = max(guard, k)
                indices = np.arange(g, n - g)
                assert len(indices) > 0
                curve = []
                for lag in range(-k, k + 1):
                    diff = (
                        v[indices].astype(np.float32)
                        - audio[indices + lag].astype(np.float32)
                        + np.float32(1e-6)
                    )
                    curve.append(
                        float(
                            np.sqrt(np.sum(diff * diff, axis=1))
                            .astype(np.float64)
                            .mean()
                        )
                    )
                original = np.array(curve)[k - 15 : k + 16]
                best = float(original.min())
                b = float(np.median(original))
                bn = float(
                    np.median(
                        np.interp(
                            np.linspace(-limit, limit, 31), np.arange(-k, k + 1), curve
                        )
                    )
                )
                cells[f"{a}_{guard}"] = {
                    "D": best,
                    "B": b,
                    "B_relative": bn,
                    "C": b - best,
                    "C_star": bn - best,
                    "lag_limit_frames": float(limit),
                    "guard": g,
                    "rows": len(indices),
                    "curve": curve,
                    "integer_lags": list(range(-k, k + 1)),
                }
        rows.append({"id": sid, "speaker": row["speaker"], "cells": cells})
    result = {"guards": {}}
    for guard in (15, 20):
        values = {
            metric: np.array(
                [
                    r["cells"][f"T_{guard}"][metric] - r["cells"][f"N_{guard}"][metric]
                    for r in rows
                ]
            )
            for metric in ("C", "C_star", "B", "B_relative", "D")
        }
        values["change"] = values["C_star"] - values["C"]
        result["guards"][str(guard)] = {
            m: cluster(x.tolist(), [r["speaker"] for r in rows])
            for m, x in values.items()
        }
    write(base / "scores.json", rows)
    write(base / "analysis.json", result)
    print(json.dumps(result["guards"]["20"], ensure_ascii=False))


if __name__ == "__main__":
    main()
