"""Post-hoc measurement robustness: known time perturbations on T/M videos."""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

from scripts.experiments.tts_independent_visual import OUT, ROOT, read, sha, write
from scripts.experiments.tts_visual_ctc_timing import posterior_times

BASE = OUT / "timing_candidates"


def extract():
    from scripts.experiments.tts_native_gain_attribution.common import gpu_lease

    write(
        BASE / "protocol.json",
        {
            "parent_sha256": sha(OUT / "protocol.json"),
            "scope": "Post-hoc measurement robustness after N timing/content results, before T/M perturbation outputs. Same fixed26 calibration IDs; no primary gate or cohort changes.",
            "perturbations": "same +8 frame prefix and +/-240ms sinusoidal monotone pixel time maps as original N calibration",
            "thresholds": "same as N: delay median response ratio [.75,1.25]; each local median error ratio <=.5 and >=.8 clips better than zero",
        },
    )
    with gpu_lease(
        gpu_peak_bytes=6 << 30, disk_temp_bytes=100 << 20, disk_persistent_bytes=1 << 30
    ) as gate:
        write(BASE / "compute_gate.json", gate)
        env = dict(
            os.environ,
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            PYTHONNOUSERSITE="1",
        )
        subprocess.run(
            [
                "/home/wjj/miniconda3/envs/autoavsr/bin/python",
                "-m",
                "scripts.experiments.tts_visual_timing_robustness",
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
    p = read(OUT / "protocol.json")
    assert sha(OUT / "protocol.json") == read(BASE / "protocol.json")["parent_sha256"]
    for key in ("model", "model_json", "config"):
        assert sha(p[key]) == p[key + "_sha256"]
    pipeline = load_vsr(OUT, "cuda:0")
    rows = []
    for row in p["rows"]:
        if row["split"] != "calibration":
            continue
        sid = row["id"]
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        assert not (
            {int(v) for v in pids.splitlines() if v.strip().isdigit()} - {os.getpid()}
        )
        for arm in ("T", "M"):
            cache = OUT / "mouths" / sid / f"{arm}.npz"
            assert sha(cache) == read(cache.with_suffix(".json"))["cache_sha256"]
            x = np.load(cache)["x"]
            nativefile = OUT / "features" / sid / f"{arm}_native.npz"
            assert (
                sha(nativefile)
                == read(OUT / "receipts" / f"{sid}.json")["feature_hashes"][
                    arm + "_native"
                ]
            )
            centers, _ = posterior_times(np.load(nativefile)["logp"], row["target"])
            n = x.shape[1]
            t = np.arange(n) / 25
            control = {}
            for mode, amplitude in [
                ("delay8", None),
                ("warp_pos", 0.240),
                ("warp_neg", -0.240),
            ]:
                if amplitude is None:
                    view = np.concatenate([np.repeat(x[:, :1], 8, axis=1), x], axis=1)
                    expected = np.full_like(centers, 0.32)
                else:
                    source = t + amplitude * np.sin(2 * np.pi * t / t[-1])
                    assert np.all(np.diff(source) > 0)
                    pos = np.clip(source * 25, 0, n - 1)
                    lo = np.floor(pos).astype(int)
                    hi = np.minimum(lo + 1, n - 1)
                    w = (pos - lo)[None, :, None, None]
                    view = (x[:, lo] * (1 - w) + x[:, hi] * w).astype(np.float32)
                    expected = np.interp(centers, source, t) - centers
                _, logp = _forward_view(pipeline, view, "cuda:0")
                assert len(logp) == view.shape[1]
                file = BASE / "features" / sid / f"{arm}_{mode}.npz"
                file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(file, logp=logp)
                actual, _ = posterior_times(logp, row["target"])
                observed = actual - centers
                control[mode] = {
                    "path": str(file),
                    "sha256": sha(file),
                    "observed_delta": observed.tolist(),
                    "expected_delta": expected.tolist(),
                    "error_ratio": float(
                        np.mean(np.abs(observed - expected)) / np.mean(np.abs(expected))
                    ),
                    "shift_ratio": float(np.median(observed) / 0.32)
                    if amplitude is None
                    else None,
                }
            rows.append({"id": sid, "arm": arm, "controls": control})
            print("candidate timing", sid, arm, flush=True)
    write(BASE / "scores.json", rows)
    result = {}
    for arm in ("T", "M"):
        result[arm] = {}
        for mode in ("delay8", "warp_pos", "warp_neg"):
            cells = [r["controls"][mode] for r in rows if r["arm"] == arm]
            ratio = float(np.median([c["error_ratio"] for c in cells]))
            fraction = float(np.mean([c["error_ratio"] < 1 for c in cells]))
            shift = (
                float(np.median([c["shift_ratio"] for c in cells]))
                if mode == "delay8"
                else None
            )
            result[arm][mode] = {
                "median_error_ratio": ratio,
                "fraction_better_than_zero": fraction,
                "median_shift_ratio": shift,
                "pass": 0.75 <= shift <= 1.25
                if shift is not None
                else ratio <= 0.5 and fraction >= 0.8,
            }
    write(BASE / "analysis.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "worker"])
    globals()[parser.parse_args().stage]()
