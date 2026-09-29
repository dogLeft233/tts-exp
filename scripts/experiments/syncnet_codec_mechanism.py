#!/usr/bin/env python3
"""Frozen codec factorial, spectral interventions, and embedding diagnostics."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

from scripts.experiments.tts_clock_cross import (
    ROOT,
    check_hash,
    convert,
    estimate_delay,
    interval,
    read,
    sha,
    write,
)

PARENT = ROOT / "runs/tts_clock_cross_20260926"
OUT = ROOT / "runs/syncnet_codec_mechanism_20260926"
RATES = (16000, 24000)
BITRATES = (24, 32, 64)
CUTOFFS = (3500, 4500, 5500, 6500, 7500)


def lowpass(x, cutoff):
    """Zero-phase FFT filter: unity to cutoff-150, cosine taper to cutoff+150."""
    pad = 16000
    y = np.pad(np.asarray(x, dtype=float), (pad, pad), mode="reflect")
    freq = np.fft.rfftfreq(len(y), 1 / 16000)
    ramp = np.clip((freq - (cutoff - 150)) / 300, 0, 1)
    response = 0.5 * (1 + np.cos(np.pi * ramp))
    return np.fft.irfft(np.fft.rfft(y) * response, n=len(y))[pad:-pad]


def pcm_write(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.max(np.abs(x)) > 32767:
        raise ValueError(f"intervention clipping: {path}")
    sf.write(path, np.rint(x).astype(np.int16), 16000, subtype="PCM_16")
    return {"path": str(path), "sha256": sha(path)}


def spectral(x, y):
    n = min(len(x), len(y))
    x, y = np.asarray(x[:n], dtype=float), np.asarray(y[:n], dtype=float)
    f, px = welch(x, 16000, nperseg=1024)
    _, py = welch(y, 16000, nperseg=1024)
    bands = {}
    for lo, hi in [
        (0, 2000),
        (2000, 3500),
        (3500, 4500),
        (4500, 5500),
        (5500, 6500),
        (6500, 7500),
        (7500, 8001),
    ]:
        mask = (f >= lo) & (f < hi)
        bands[f"{lo}_{hi}"] = float(
            10 * np.log10((py[mask].sum() + 1e-20) / (px[mask].sum() + 1e-20))
        )
    return {
        "band_power_db_vs_pcm": bands,
        "rms_ratio": float(np.linalg.norm(y) / np.linalg.norm(x)),
    }


def freeze():
    rows = []
    for cohort, base in [("ditto", PARENT), ("real", PARENT / "real12")]:
        for r in read(base / "protocol.json")["rows"]:
            rows.append(
                {**r, "cohort": cohort, "key": f"{cohort}_{r['id']}_{r['arm']}"}
            )
    protocol = {
        "date": "2026-09-26",
        "parent": str(PARENT),
        "design": "all frozen 47 Ditto pairs and 12 real sources; rate x bitrate x cutoff; fixed videos",
        "rates": RATES,
        "bitrates_kbps": BITRATES,
        "codec_cutoff_hz": [None, 7500],
        "pcm_lowpass_hz": CUTOFFS,
        "lowpass_transition_hz": 300,
        "guard": 20,
        "support": "minimum across all 18 audio conditions and visual; guard20",
        "delay": "waveform xcorr only; no video/score selection",
        "diagnostics": "C=B-D; PCM fixed lag; offset; unit-vector score; source/codec direction x norm swap",
        "primary": "fixed-bitrate sample-rate effect; bitrate effect; cutoff override; pure lowpass vs PCM",
        "bootstrap": "20000 draws; seed20260926; paired utterances within each arm; real source groups separately",
        "limits": "exploratory extension; Ditto single speaker; no perceptual claims; feature surgery diagnostic only",
        "model_sha256": read(PARENT / "protocol.json")["model_sha256"],
        "rows": rows,
    }
    path = OUT / "protocol.json"
    normalized = __import__("json").loads(__import__("json").dumps(protocol))
    if path.exists() and read(path) != normalized:
        raise ValueError("frozen protocol differs")
    write(path, protocol)
    print("frozen", len(rows), "x 18", flush=True)


def prepare_row(row):
    root = OUT / "audio" / row["key"]
    receipt = root / "receipt.json"
    if receipt.exists():
        return row["key"]
    check_hash(row["source_pcm"], row["source_pcm_sha256"])
    check_hash(row["visual"], row["visual_sha256"])
    source, sr = sf.read(row["source_pcm"], dtype="int16")
    assert sr == 16000
    modes = {"pcm": {"path": row["source_pcm"], "sha256": row["source_pcm_sha256"]}}
    root.mkdir(parents=True, exist_ok=True)
    for cutoff in CUTOFFS:
        key = f"lp{cutoff}"
        modes[key] = pcm_write(root / f"{key}.wav", lowpass(source, cutoff))
    for rate in RATES:
        for bitrate in BITRATES:
            for cutoff in (None, 7500):
                key = f"mp3_{rate}_{bitrate}_{cutoff or 'auto'}"
                avi = root / f"{key}.avi"
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    row["source_pcm"],
                    "-ac",
                    "1",
                    "-ar",
                    str(rate),
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    f"{bitrate}k",
                ]
                if cutoff:
                    cmd += ["-cutoff", str(cutoff)]
                cmd += [str(avi)]
                subprocess.run(cmd, check=True, capture_output=True)
                raw = root / f"{key}_raw.wav"
                convert(avi, raw)
                pcm, _ = sf.read(raw, dtype="int16")
                delay = estimate_delay(source, pcm)
                lag = delay["lag_samples"]
                if not 0 <= lag <= 1600:
                    raise ValueError(f"invalid lag {row['key']}/{key}: {delay}")
                aligned = pcm[lag : lag + len(source)]
                modes[key] = {
                    **pcm_write(root / f"{key}.wav", aligned),
                    "delay": delay,
                    "command": cmd,
                    "encoded_sha256": sha(avi),
                }
    for info in modes.values():
        pcm, _ = sf.read(info["path"], dtype="int16")
        info["samples"] = len(pcm)
        info["spectrum"] = spectral(source, pcm)
    write(receipt, {"row": row, "modes": modes})
    return row["key"]


def prepare():
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key in pool.map(prepare_row, read(OUT / "protocol.json")["rows"]):
            print("prepared", key, flush=True)


def metric(matrix, fixed_index=None, guard=20):
    if len(matrix) <= 2 * guard:
        raise ValueError("insufficient common support")
    curve = matrix[guard:-guard].mean(axis=0)
    j = int(curve.argmin())
    b, d = float(np.median(curve)), float(curve[j])
    return {
        "C": b - d,
        "B": b,
        "D": d,
        "offset": 15 - j,
        "fixed_D": float(curve[j if fixed_index is None else fixed_index]),
        "boundary": j in (0, 30),
        "rows": len(matrix) - 2 * guard,
    }


def score(device):
    import torch

    from scripts.experiments.tts_native_gain_attribution.common import (
        gpu_compute_pids,
        gpu_lease,
    )
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    protocol = read(OUT / "protocol.json")
    lease = (
        gpu_lease(
            gpu_peak_bytes=2 << 30,
            disk_temp_bytes=1 << 30,
            disk_persistent_bytes=3 << 30,
        )
        if device == "cuda"
        else nullcontext({"device": "cpu"})
    )
    with lease as gate:
        write(OUT / "compute_gate.json", gate)
        engine = SyncNetEngine(batch_size=64, device=device)
        assert engine.model_hash == protocol["model_sha256"]
        try:
            for row in protocol["rows"]:
                dest = OUT / "scores" / f"{row['key']}.json"
                if dest.exists():
                    continue
                if device == "cuda" and set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError("foreign GPU process appeared; checkpoint saved")
                receipt = read(OUT / "audio" / row["key"] / "receipt.json")
                check_hash(row["visual"], row["visual_sha256"])
                v = np.load(row["visual"])
                features = {}
                for key, info in receipt["modes"].items():
                    check_hash(info["path"], info["sha256"])
                    features[key], _ = engine.extract_audio(info["path"])
                n = min(len(v), *(len(x) for x in features.values()))
                v = v[:n]
                features = {k: x[:n] for k, x in features.items()}
                matrices = {
                    k: engine.distance_matrix(v, x) for k, x in features.items()
                }
                pcm_index = int(matrices["pcm"][20:-20].mean(axis=0).argmin())
                cells = {k: metric(m, pcm_index) for k, m in matrices.items()}
                # All representation counterfactuals keep each window at its original time.
                a0 = features["pcm"]
                r0 = np.linalg.norm(a0, axis=1, keepdims=True)
                u0 = a0 / r0
                vu = engine.unit_features(v)
                diagnostic = {}
                for key, a in features.items():
                    r = np.linalg.norm(a, axis=1, keepdims=True)
                    u = a / r
                    swapped = {
                        "unit": (vu, u),
                        "source_norm": (v, u * r0),
                        "source_direction": (v, u0 * r),
                    }
                    diagnostic[key] = {
                        "audio_norm": float(r[20:-20].mean()),
                        "cos_to_pcm": float((u * u0).sum(axis=1)[20:-20].mean()),
                    }
                    for label, (vv, aa) in swapped.items():
                        mk = f"{key}__{label}"
                        matrices[mk] = engine.distance_matrix(vv, aa)
                        diagnostic[key][label] = metric(matrices[mk], pcm_index)
                dest.parent.mkdir(parents=True, exist_ok=True)
                fp = OUT / "features" / f"{row['key']}.npz"
                fp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(fp, **features)
                mp = dest.with_suffix(".npz")
                np.savez_compressed(mp, **matrices)
                write(
                    dest,
                    {
                        "key": row["key"],
                        "cohort": row["cohort"],
                        "id": row["id"],
                        "arm": row["arm"],
                        "device": device,
                        "common_rows": n,
                        "pcm_lag_index": pcm_index,
                        "cells": cells,
                        "diagnostic": diagnostic,
                        "features_sha256": sha(fp),
                        "matrices_sha256": sha(mp),
                        "receipt_sha256": sha(
                            OUT / "audio" / row["key"] / "receipt.json"
                        ),
                    },
                )
                print("scored", row["key"], flush=True)
        finally:
            engine.close()


def analyze():
    protocol = read(OUT / "protocol.json")
    rows = [read(OUT / "scores" / f"{r['key']}.json") for r in protocol["rows"]]
    contrasts = []
    for rate in RATES:
        for br in BITRATES:
            for cut in ("auto", "7500"):
                contrasts.append((f"mp3_{rate}_{br}_{cut}", "pcm"))
            contrasts.append((f"mp3_{rate}_{br}_7500", f"mp3_{rate}_{br}_auto"))
        for br in (32, 64):
            contrasts.append((f"mp3_{rate}_{br}_auto", f"mp3_{rate}_24_auto"))
    for br in BITRATES:
        for cut in ("auto", "7500"):
            contrasts.append((f"mp3_24000_{br}_{cut}", f"mp3_16000_{br}_{cut}"))
    contrasts += [(f"lp{c}", "pcm") for c in CUTOFFS]
    contrasts += [("mp3_24000_32_auto", "mp3_16000_24_auto")]
    result = {"groups": {}}
    for group, cohort, arm in [
        ("natural", "ditto", "N"),
        ("tts", "ditto", "T"),
        ("real", "real", "R"),
    ]:
        rr = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm]
        stats = {}
        for left, right in contrasts:
            label = left + "__minus__" + right
            stats[label] = {
                metric_name: interval(
                    [
                        r["cells"][left][metric_name] - r["cells"][right][metric_name]
                        for r in rr
                    ]
                )
                for metric_name in ["C", "B", "D", "fixed_D"]
            }
            stats[label]["same_offset"] = sum(
                r["cells"][left]["offset"] == r["cells"][right]["offset"] for r in rr
            )
        diag = {}
        for mode in rr[0]["cells"]:
            diag[mode] = {
                label: interval(
                    [
                        r["diagnostic"][mode][label]["C"]
                        - r["diagnostic"]["pcm"][label]["C"]
                        for r in rr
                    ]
                )
                for label in ["unit", "source_norm", "source_direction"]
            }
            diag[mode]["norm_delta"] = interval(
                [
                    r["diagnostic"][mode]["audio_norm"]
                    - r["diagnostic"]["pcm"]["audio_norm"]
                    for r in rr
                ]
            )
        result["groups"][group] = {
            "n": len(rr),
            "contrasts": stats,
            "diagnostic": diag,
            "means": {
                m: {
                    k: float(np.mean([r["cells"][m][k] for r in rr]))
                    for k in ["C", "B", "D"]
                }
                for m in rr[0]["cells"]
            },
        }
    lookup = {(r["id"], r["arm"]): r for r in rows if r["cohort"] == "ditto"}
    ids = sorted({i for i, a in lookup})
    result["ditto_tts_minus_natural"] = {
        m: interval(
            [
                lookup[i, "T"]["cells"][m]["C"] - lookup[i, "N"]["cells"][m]["C"]
                for i in ids
            ]
        )
        for m in rows[0]["cells"]
    }
    result["boundary_peaks"] = sum(
        c["boundary"] for r in rows for c in r["cells"].values()
    )
    write(OUT / "analysis.json", result)
    print("analysis complete", len(rows), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["freeze", "prepare", "score", "analyze"])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.stage == "score":
        score(a.device)
    else:
        globals()[a.stage]()


if __name__ == "__main__":
    main()
