"""Test fixed generator-lookahead compensation before nonlinear video retiming."""

from __future__ import annotations

import argparse

import numpy as np

from scripts.experiments import tts_raw_video_transfer as transfer
from scripts.experiments.tts_independent_visual import (
    OUT,
    PARENT,
    ROOT,
    read,
    sha,
    write,
)
from scripts.experiments.tts_pcm_residual import cluster
from scripts.experiments.tts_pcm_timing import decode

BASE = ROOT / "runs/tts_window_clock_20260926"
PHONE = OUT / "phone_transfer"


def prepare():
    protocol = {
        "selection": "same38 previous evaluation phone-compatible rows;13 audio speakers; historical one-face cohort",
        "parent_manifest_sha256": sha(PHONE / "manifest.json"),
        "parent_pcm_sha256": sha(PARENT / "protocol.json"),
        "mapping": "source=phi(target+L)-L with audio phone-boundary phi; primary L=.100; sensitivity .080; negative -.100; no parameter search",
        "hypothesis": "200ms Wav2Lip input mel window may induce lookahead; actual event at midpoint is a hypothesis, not established by architecture",
        "primary": "+100ms vs old boundary and vsN, guard20; same exact naturalPCM; C/B/D and N-best-lag distance; guard0/15 sensitivity",
        "statistics": "speaker equal,20000 bootstrap,seed20260926,95/99%; exploratory",
        "inference_sha256": sha(ROOT / "third_party/Wav2Lip/inference.py"),
        "hparams_sha256": sha(ROOT / "third_party/Wav2Lip/hparams.py"),
    }
    dest = BASE / "protocol.json"
    if dest.exists():
        assert read(dest) == protocol
    write(dest, protocol)
    old = {r["id"]: r for r in read(PARENT / "protocol.json")["rows"]}
    rows = []
    for row in read(PHONE / "manifest.json")["rows"]:
        sid = row["id"]
        c = old[sid]["cells"]["T"]
        assert sha(c["crop"]) == c["crop_sha256"]
        frames = decode(c["crop"])
        t = np.arange(row["videos"]["N_identity"]["frames"]) / 25
        x, y = [np.array(row["maps"]["boundaries"][k]) for k in ("N", "T")]
        result = {k: row[k] for k in ("id", "speaker", "split")}
        result.update(
            videos={
                "N_identity": row["videos"]["N_identity"],
                "T_zero": row["videos"]["T_phone_boundaries"],
            },
            maps={},
        )
        for name, lag in [("T_plus100", 0.1), ("T_plus80", 0.08), ("T_minus100", -0.1)]:
            source = np.interp(t + lag, x, y) - lag
            assert np.all(np.diff(source) >= -1e-12)
            # Analytic identity and affine sanity checks on strictly internal times.
            interior = t[(t > 0.2) & (t < x[-1] - 0.2)]
            assert (
                np.max(np.abs(np.interp(interior + lag, x, x) - lag - interior)) < 1e-12
            )
            rate = y[-1] / x[-1]
            assert (
                np.max(
                    np.abs(
                        (rate * (interior + lag) - lag)
                        - (rate * interior + (rate - 1) * lag)
                    )
                )
                < 1e-12
            )
            file = BASE / "videos" / sid / f"{name}.avi"
            transfer.render(frames, source, file)
            result["videos"][name] = {
                "path": str(file),
                "sha256": sha(file),
                "frames": len(t),
            }
            result["maps"][name] = {"L": lag, "source": source.tolist()}
        result["anchors"] = {"N": x.tolist(), "T": y.tolist()}
        result["source_crop_sha256"] = c["crop_sha256"]
        rows.append(result)
        print("prepared clock", sid, flush=True)
    write(BASE / "manifest.json", {"rows": rows})


def score():
    transfer.BASE = BASE
    transfer.score()


def calibration_extension():
    cal = [
        r for r in read(OUT / "protocol.json")["rows"] if r["split"] == "calibration"
    ]
    offsets = {
        arm: [
            read(PARENT / "scores" / f"{r['id']}.json")["cells"][arm + "_source"]["20"][
                "offset"
            ]
            for r in cal
        ]
        for arm in ("N", "T")
    }
    lags = {arm: -float(np.median(vals)) / 25 for arm, vals in offsets.items()}
    write(
        BASE / "calibration_extension.json",
        {
            "ids": [r["id"] for r in cal],
            "offsets": offsets,
            "lags": lags,
            "scope": "before any current treatment scores; no evaluation-fitted parameters; secondary endpoint only",
        },
    )
    old = {r["id"]: r for r in read(PARENT / "protocol.json")["rows"]}
    manifest = read(BASE / "manifest.json")
    for row in manifest["rows"]:
        sid = row["id"]
        frames = decode(old[sid]["cells"]["T"]["crop"])
        t = np.arange(row["videos"]["N_identity"]["frames"]) / 25
        x, y = [np.array(row["anchors"][arm]) for arm in ("N", "T")]
        source = np.interp(t + lags["N"], x, y) - lags["T"]
        file = BASE / "videos" / sid / "T_calibrated.avi"
        transfer.render(frames, source, file)
        row["videos"]["T_calibrated"] = {
            "path": str(file),
            "sha256": sha(file),
            "frames": len(t),
        }
        row["maps"]["T_calibrated"] = {
            "L": lags["N"],
            "L_T": lags["T"],
            "source": source.tolist(),
        }
    write(BASE / "manifest.json", manifest)
    print("calibration-derived clock", lags, flush=True)


def analyze():
    rows = [
        read(BASE / "scores" / f"{r['id']}.json")
        for r in read(BASE / "manifest.json")["rows"]
    ]
    result = {}
    for guard in ("0", "15", "20"):
        g = int(guard)
        out = {}
        for left, right in [
            ("T_zero", "N_identity"),
            ("T_plus100", "T_zero"),
            ("T_plus100", "N_identity"),
            ("T_plus80", "T_zero"),
            ("T_plus80", "N_identity"),
            ("T_minus100", "T_zero"),
            ("T_plus100", "T_minus100"),
            ("T_calibrated", "T_zero"),
            ("T_calibrated", "N_identity"),
        ]:
            values = {m: [] for m in ("C", "D", "B", "N_anchor_distance")}
            for row in rows:
                for m in ("C", "D", "B"):
                    values[m].append(
                        row["cells"][left][guard][m] - row["cells"][right][guard][m]
                    )
                matrices = np.load(BASE / "scores" / f"{row['id']}.npz")
                curves = {k: (v[g:-g] if g else v).mean(0) for k, v in matrices.items()}
                j = int(np.argmin(curves["N_identity"]))
                values["N_anchor_distance"].append(
                    float(curves[left][j] - curves[right][j])
                )
            out[left + "_minus_" + right] = {
                m: cluster(v, [r["speaker"] for r in rows]) for m, v in values.items()
            }
        result[guard] = out
    write(BASE / "analysis.json", result)
    for name, v in result["20"].items():
        c = v["C"]
        print(name, c["speaker_mean"], c["speaker_ci95"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["prepare", "calibration_extension", "score", "analyze"]
    )
    globals()[p.parse_args().stage]()
