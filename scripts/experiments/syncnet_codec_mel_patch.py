"""Diagnostic interventions at SyncNet's actual log-mel to MFCC input path."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext

import numpy as np
import python_speech_features as psf
import soundfile as sf
from scipy.fftpack import dct

from scripts.experiments.syncnet_codec_mechanism import OUT, metric
from scripts.experiments.tts_clock_cross import check_hash, interval, read, sha, write

PATCH = OUT / "mel_patch"


def freeze():
    centers = psf.base.mel2hz(
        np.linspace(psf.base.hz2mel(0), psf.base.hz2mel(8000), 28)
    )[1:-1]
    high = np.flatnonzero(centers >= 5500).tolist()
    payload = {
        "parent_protocol_sha256": sha(OUT / "protocol.json"),
        "adaptive_reason": "Interim factorial outcomes suggest both bandwidth and residual distortion; next stage tests the actual evaluator input pathway, not generator quality.",
        "cohort": "all 106 frozen fixed videos; no outcome selection",
        "codec": "mp3_16000_24_auto",
        "mel_centers_hz": centers.tolist(),
        "high_indices": high,
        "split_rule": "mel center >=5500Hz; fixed before patch outcomes; 0-based indices",
        "modes": [
            "source",
            "codec",
            "restore_high",
            "restore_low",
            "restore_energy",
            "restore_high_energy",
            "damage_high",
        ],
        "definition": "restore_high: codec logmel with source high bands, codec log energy; restore_low complementary bands; energy swaps only MFCC c0; damage_high reverse source with codec high bands",
        "support": "parent common row count, guard20; source/codec rebuilt feature replay tolerance1e-4",
        "limits": "representation counterfactuals may be off-manifold; pair with physical codec/lowpass interventions; no new audio synthesis or claims about TFG",
    }
    path = PATCH / "protocol.json"
    if path.exists() and read(path) != payload:
        raise ValueError("patch protocol differs")
    write(path, payload)
    print("frozen mel intervention", high, centers[high], flush=True)


def cepstra(logmel, energy):
    cc = psf.base.lifter(dct(logmel, type=2, axis=1, norm="ortho")[:, :13], 22)
    cc[:, 0] = np.log(energy)
    return cc


def run(device):
    import torch

    from scripts.experiments.tts_native_gain_attribution.common import (
        gpu_compute_pids,
        gpu_lease,
    )
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    protocol = read(PATCH / "protocol.json")
    check_hash(OUT / "protocol.json", protocol["parent_protocol_sha256"])
    lease = (
        gpu_lease(
            gpu_peak_bytes=2 << 30,
            disk_temp_bytes=200 << 20,
            disk_persistent_bytes=600 << 20,
        )
        if device == "cuda"
        else nullcontext({"device": "cpu"})
    )
    with lease as gate:
        write(PATCH / "compute_gate.json", gate)
        engine = SyncNetEngine(batch_size=64, device=device)
        try:
            for row in read(OUT / "protocol.json")["rows"]:
                dest = PATCH / "scores" / f"{row['key']}.json"
                if dest.exists():
                    continue
                if device == "cuda" and set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError("foreign GPU process appeared")
                parent_score = read(OUT / "scores" / f"{row['key']}.json")
                check_hash(
                    OUT / "features" / f"{row['key']}.npz",
                    parent_score["features_sha256"],
                )
                cache = np.load(OUT / "features" / f"{row['key']}.npz")
                receipt = read(OUT / "audio" / row["key"] / "receipt.json")
                mel, energy = {}, {}
                for label, mode in [("source", "pcm"), ("codec", protocol["codec"])]:
                    info = receipt["modes"][mode]
                    check_hash(info["path"], info["sha256"])
                    wave, sr = sf.read(info["path"], dtype="int16")
                    assert sr == 16000
                    fb, e = psf.fbank(wave, 16000)
                    mel[label], energy[label] = np.log(fb), e
                length = min(len(v) for v in mel.values())
                mel = {k: v[:length] for k, v in mel.items()}
                energy = {k: v[:length] for k, v in energy.items()}
                high = protocol["high_indices"]
                low = [i for i in range(26) if i not in high]
                restored_high, restored_low = mel["codec"].copy(), mel["codec"].copy()
                restored_high[:, high] = mel["source"][:, high]
                restored_low[:, low] = mel["source"][:, low]
                damaged = mel["source"].copy()
                damaged[:, high] = mel["codec"][:, high]
                specs = {
                    "source": (mel["source"], energy["source"]),
                    "codec": (mel["codec"], energy["codec"]),
                    "restore_high": (restored_high, energy["codec"]),
                    "restore_low": (restored_low, energy["codec"]),
                    "restore_energy": (mel["codec"], energy["source"]),
                    "restore_high_energy": (restored_high, energy["source"]),
                    "damage_high": (damaged, energy["source"]),
                }
                n = parent_score["common_rows"]
                features = {}
                with torch.inference_mode():
                    for mode, (logmel, e) in specs.items():
                        cc = cepstra(logmel, e)
                        windows = np.stack(
                            [cc[i * 4 : i * 4 + 20].T for i in range(n)]
                        ).astype(np.float32)
                        outputs = []
                        for start in range(0, n, 64):
                            batch = torch.from_numpy(windows[start : start + 64])[
                                :, None
                            ].to(device)
                            outputs.append(
                                engine.network.forward_aud(batch).cpu().numpy()
                            )
                        features[mode] = np.concatenate(outputs)
                errors = {
                    label: float(np.abs(features[label] - cache[mode][:n]).max())
                    for label, mode in [("source", "pcm"), ("codec", protocol["codec"])]
                }
                if max(errors.values()) > 1e-4:
                    raise ValueError(f"MFCC path failed replay {errors}")
                check_hash(row["visual"], row["visual_sha256"])
                v = np.load(row["visual"])[:n]
                matrices = {
                    k: engine.distance_matrix(v, a) for k, a in features.items()
                }
                cells = {
                    k: metric(m, parent_score["pcm_lag_index"])
                    for k, m in matrices.items()
                }
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                write(
                    dest,
                    {
                        "key": row["key"],
                        "cohort": row["cohort"],
                        "arm": row["arm"],
                        "cells": cells,
                        "replay_errors": errors,
                        "matrix_sha256": sha(dest.with_suffix(".npz")),
                    },
                )
                print("patched", row["key"], flush=True)
        finally:
            engine.close()


def analyze():
    rows = [
        read(PATCH / "scores" / f"{r['key']}.json")
        for r in read(OUT / "protocol.json")["rows"]
    ]
    groups = {}
    for group, arm in [("natural", "N"), ("tts", "T"), ("real", "R")]:
        rr = [r for r in rows if r["arm"] == arm]
        stats = {}
        for mode in rr[0]["cells"]:
            for base in ("source", "codec"):
                stats[mode + "_minus_" + base] = interval(
                    [r["cells"][mode]["C"] - r["cells"][base]["C"] for r in rr]
                )
        groups[group] = stats
    write(
        PATCH / "analysis.json",
        {
            "groups": groups,
            "max_replay_error": max(
                e for r in rows for e in r["replay_errors"].values()
            ),
        },
    )
    print("patch analysis complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["freeze", "run", "analyze"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    if a.stage == "run":
        run(a.device)
    else:
        globals()[a.stage]()
