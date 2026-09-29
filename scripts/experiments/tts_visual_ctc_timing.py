"""Calibrate visual CTC character timing using known pixel-time interventions."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from scripts.experiments.tts_independent_visual import OUT, ROOT, read, sha, write

BASE = OUT / "timing"


def posterior_times(logp, target):
    """Forward/backward CTC occupancy centers; independent of audio timestamps."""
    logp = np.asarray(logp, dtype=np.float64)
    states = np.zeros(2 * len(target) + 1, dtype=int)
    states[1::2] = target
    emissions = logp[:, states]
    tmax, nstates = emissions.shape
    skip = np.zeros(nstates, dtype=bool)
    skip[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    forward = np.full((tmax, nstates), -np.inf)
    forward[0, :2] = emissions[0, :2]
    for t in range(1, tmax):
        prev = forward[t - 1]
        one = np.r_[-np.inf, prev[:-1]]
        two = np.r_[-np.inf, -np.inf, prev[:-2]]
        two[~skip] = -np.inf
        forward[t] = np.logaddexp(np.logaddexp(prev, one), two) + emissions[t]
    backward = np.full_like(forward, -np.inf)
    backward[-1, -2:] = 0
    for t in range(tmax - 2, -1, -1):
        nxt = backward[t + 1] + emissions[t + 1]
        one = np.r_[nxt[1:], -np.inf]
        two = np.r_[nxt[2:], -np.inf, -np.inf]
        valid = np.r_[skip[2:], False, False]
        two[~valid] = -np.inf
        backward[t] = np.logaddexp(np.logaddexp(nxt, one), two)
    logz = np.logaddexp(forward[-1, -1], forward[-1, -2])
    gamma = np.exp(forward + backward - logz)[:, 1::2]
    assert np.all(gamma.sum(0) > 0) and np.isfinite(logz)
    times = ((np.arange(tmax) + 0.5) / 25) @ gamma / gamma.sum(0)
    return times, float(-logz / len(target))


def freeze():
    p = {
        "parent_sha256": sha(OUT / "protocol.json"),
        "selection": "26 predetermined calibration N videos only for measurement gate; 42 evaluation rows for M-N endpoint; no score selection",
        "perturbations": "prepend8 copies first frame (+320ms, no frame loss); local pixel resampling source t=t_target +/- 0.240*sin(2*pi*t_target/duration), fixed frame count; endpoints unchanged, monotone; no audio",
        "estimator": "full CTC forward-backward token occupancy mean at 25fps with half-frame origin; character positions not character classes; no audio input",
        "gate": "global median measured/known shift in [0.75,1.25]; local perturbation median MAE/zero-response-MAE <=0.5 and >=80% clips ratio<1, separately positive/negative; parent natural content gate must also pass",
        "local_expected": "inverse known time map applied to each native visual character center; compare displaced observed centers; frozen input is invariant under these fixed-length time maps",
        "endpoint": "visual character center vs natural MFA character center, remove per-clip median global lag then mean absolute residual; evaluation M-N lower better; strict full lexical sequence match; no tuning using evaluation outcomes",
        "limits": "known-intervention response calibration, not ground-truth physical mouth events; CTC timing may reflect contextual emission; real alignment accuracy is not established merely by equivariance",
    }
    dest = BASE / "protocol.json"
    if dest.exists() and read(dest) != p:
        raise ValueError("timing protocol differs")
    write(dest, p)
    print("CTC timing frozen", flush=True)


def extract():
    from scripts.experiments.tts_native_gain_attribution.common import gpu_lease

    with gpu_lease(
        gpu_peak_bytes=6 << 30, disk_temp_bytes=100 << 20, disk_persistent_bytes=1 << 30
    ) as gate:
        write(BASE / "compute_gate.json", gate)
        env = dict(os.environ)
        env.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "2",
                "PYTHONNOUSERSITE": "1",
            }
        )
        subprocess.run(
            [
                "/home/wjj/miniconda3/envs/autoavsr/bin/python",
                "-m",
                "scripts.experiments.tts_visual_ctc_timing",
                "worker",
            ],
            cwd=str(ROOT),
            env=env,
            check=True,
        )


def worker():
    import torch

    from scripts.experiments.vsr_tts_pilot import (
        _forward_view,
        _set_determinism,
        load_vsr,
    )

    torch.set_num_threads(2)
    _set_determinism(20260926)
    protocol = read(OUT / "protocol.json")
    assert sha(OUT / "protocol.json") == read(BASE / "protocol.json")["parent_sha256"]
    for key in ("model", "model_json", "config"):
        assert sha(protocol[key]) == protocol[key + "_sha256"]
    pipeline = load_vsr(OUT, "cuda:0")
    for row in read(OUT / "protocol.json")["rows"]:
        if row["split"] != "calibration":
            continue
        sid = row["id"]
        done = BASE / "features" / sid / "receipt.json"
        if done.exists():
            continue
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        assert not (
            {int(x) for x in pids.splitlines() if x.strip().isdigit()} - {os.getpid()}
        )
        file = OUT / "mouths" / sid / "N.npz"
        assert sha(file) == read(file.with_suffix(".json"))["cache_sha256"]
        x = np.load(file)["x"]
        n = x.shape[1]
        duration = (n - 1) / 25
        t = np.arange(n) / 25
        views = {"delay8": np.concatenate([np.repeat(x[:, :1], 8, axis=1), x], axis=1)}
        maps = {}
        for name, amplitude in [("warp_pos", 0.240), ("warp_neg", -0.240)]:
            source = t + amplitude * np.sin(2 * np.pi * t / duration)
            assert np.all(np.diff(source) > 0)
            pos = np.clip(source * 25, 0, n - 1)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, n - 1)
            weight = (pos - lo)[None, :, None, None]
            views[name] = (x[:, lo] * (1 - weight) + x[:, hi] * weight).astype(
                np.float32
            )
            maps[name] = {"target_time": t.tolist(), "source_time": source.tolist()}
        hashes = {}
        for name, view in views.items():
            _, logp = _forward_view(pipeline, view, "cuda:0")
            assert len(logp) == view.shape[1], "CTC/frame time grid mismatch"
            path = done.parent / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, logp=logp)
            hashes[name] = sha(path)
        write(
            done,
            {
                "id": sid,
                "speaker": row["speaker"],
                "hashes": hashes,
                "maps": maps,
                "source_cache_sha256": sha(file),
            },
        )
        print("CTC timing calibration extracted", sid, flush=True)


def words(path):
    content = Path(path).read_text()
    tier = content.split('name = "words"', 1)[1].split("item [", 1)[0]
    return [
        (word, (float(a) + float(b)) / 2)
        for a, b, word in re.findall(
            r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',
            tier,
        )
        if word.strip()
    ]


def analyze():
    from scripts.experiments.tts_pcm_residual import cluster

    p = read(OUT / "protocol.json")
    rows = []
    cal = []
    missing = []
    parent_scores = {r["id"]: r for r in read(OUT / "scores.json")}
    for row in p["rows"]:
        sid = row["id"]
        if sid not in parent_scores:
            continue
        receipt = read(OUT / "receipts" / f"{sid}.json")
        centers = {}
        for arm in parent_scores[sid]["arms"]:
            file = OUT / "features" / sid / f"{arm}_native.npz"
            assert sha(file) == receipt["feature_hashes"][arm + "_native"]
            f = np.load(file)
            centers[arm], loss = posterior_times(f["logp"], row["target"])
            assert (
                abs(loss - parent_scores[sid]["cells"][arm + "_native"]["target_nll"])
                < 1e-5
            )
        if row["split"] == "calibration":
            receipt = read(BASE / "features" / sid / "receipt.json")
            control = {}
            for mode, digest in receipt["hashes"].items():
                file = BASE / "features" / sid / f"{mode}.npz"
                assert sha(file) == digest
                ct, _ = posterior_times(np.load(file)["logp"], row["target"])
                observed = ct - centers["N"]
                if mode == "delay8":
                    expected = np.full_like(observed, 0.32)
                else:
                    mapping = receipt["maps"][mode]
                    expected = (
                        np.interp(
                            centers["N"], mapping["source_time"], mapping["target_time"]
                        )
                        - centers["N"]
                    )
                err = float(np.mean(np.abs(observed - expected)))
                null = float(np.mean(np.abs(expected)))
                control[mode] = {
                    "observed_delta": observed.tolist(),
                    "expected_delta": expected.tolist(),
                    "MAE": err,
                    "null_MAE": null,
                    "error_ratio": err / null,
                    "median_shift_ratio": float(np.median(observed) / 0.32)
                    if mode == "delay8"
                    else None,
                }
            cal.append({"id": sid, "speaker": row["speaker"], "controls": control})
        path = (
            ROOT
            / "runs/tts_pcm_residual_20260926/timing/textgrids"
            / f"{row['speaker']}_N"
            / f"{sid}_N.TextGrid"
        )
        spans = words(path)
        if [w[0] for w in spans] != list(row["text"]):
            missing.append(
                {"id": sid, "reason": "MFA lexical sequence not exact characters"}
            )
            continue
        target = np.array([w[1] for w in spans])
        cell = {}
        for arm, ct in centers.items():
            if arm == "T":
                continue
            delta = ct - target
            global_lag = float(np.median(delta))
            cell[arm] = {
                "absolute_MAE_ms": float(np.mean(np.abs(delta)) * 1000),
                "residual_MAE_ms": float(np.mean(np.abs(delta - global_lag)) * 1000),
                "global_lag_ms": global_lag * 1000,
                "visual_centers": ct.tolist(),
            }
        rows.append(
            {
                "id": sid,
                "speaker": row["speaker"],
                "split": row["split"],
                "mfa": str(path),
                "mfa_sha256": sha(path),
                "target_centers": target.tolist(),
                "cells": cell,
            }
        )
    gate = {}
    for mode in ("delay8", "warp_pos", "warp_neg"):
        ratios = [r["controls"][mode]["error_ratio"] for r in cal]
        if mode == "delay8":
            median = float(
                np.median([r["controls"][mode]["median_shift_ratio"] for r in cal])
            )
            passed = 0.75 <= median <= 1.25
        else:
            median = None
            passed = np.median(ratios) <= 0.5 and np.mean(np.array(ratios) < 1) >= 0.8
        gate[mode] = {
            "n": len(cal),
            "median_error_ratio": float(np.median(ratios)),
            "fraction_better_than_zero": float(np.mean(np.array(ratios) < 1)),
            "median_shift_ratio": median,
            "pass": bool(passed),
        }
    result = {
        "calibration": gate,
        "content_gate": read(OUT / "analysis.json")["primary_gate"],
        "splits": {},
        "missing": missing,
    }
    result["timing_gate"] = result["content_gate"] and all(
        x["pass"] for x in gate.values()
    )
    for split in ("evaluation", "calibration", "all"):
        group = [r for r in rows if split == "all" or r["split"] == split]
        out = {}
        for left, right in [("M", "N"), ("D", "N"), ("M", "D")]:
            rr = [r for r in group if left in r["cells"] and right in r["cells"]]
            if not rr:
                continue
            for metric in ("residual_MAE_ms", "absolute_MAE_ms"):
                out[f"{left}_{right}_{metric}"] = cluster(
                    [r["cells"][left][metric] - r["cells"][right][metric] for r in rr],
                    [r["speaker"] for r in rr],
                )
        result["splits"][split] = out
    write(BASE / "calibration_scores.json", cal)
    write(BASE / "scores.json", rows)
    write(BASE / "analysis.json", result)
    print("CTC timing gate:", result["timing_gate"], json_string(gate), flush=True)


def json_string(value):
    import json

    return json.dumps(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "extract", "worker", "analyze"])
    globals()[parser.parse_args().stage]()
