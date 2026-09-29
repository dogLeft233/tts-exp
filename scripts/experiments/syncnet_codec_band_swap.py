"""Physical waveform frequency-band swaps, following the frozen MFCC diagnosis."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext

import numpy as np
import soundfile as sf

from scripts.experiments.syncnet_codec_mechanism import OUT, lowpass, metric, pcm_write
from scripts.experiments.tts_clock_cross import check_hash, interval, read, sha, write

DEST = OUT / "band_swap"


def freeze():
    payload = {
        "parent_sha256": sha(OUT / "protocol.json"),
        "reason": "Completed log-mel patch suggests top bands explain most compression loss; test with physical waveforms to address feature-surgery limitations.",
        "cohort": "all106 prior videos, fixed denominator; no score-based selection",
        "cutoff_hz": 5500,
        "transition_hz": 300,
        "codec": "mp3_16000_24_auto",
        "conditions": {
            "pcm": "original PCM",
            "codec": "aligned16k24kbps MP3",
            "restore_high": "codec + highpass(source-codec)",
            "damage_high": "source - highpass(source-codec)",
            "restore_rms": "codec rescaled to source RMS",
        },
        "support": "parent common rows guard20; source/codec scores replay within1e-4",
        "filter": "same fixed zero-phase complementary FFT filter as factorial; reflect pad1s; no time warp",
        "primary": "restore_high-codec and damage_high-pcm; RMS-only control",
        "limits": "Uses original audio as oracle donor; diagnoses evaluation corruption, not a deployable enhancement or TTS transfer method.",
    }
    path = DEST / "protocol.json"
    if path.exists() and read(path) != payload:
        raise ValueError("frozen protocol differs")
    write(path, payload)
    print("band swap frozen", flush=True)


def run(device):
    import torch

    from scripts.experiments.tts_native_gain_attribution.common import (
        gpu_compute_pids,
        gpu_lease,
    )
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    p = read(DEST / "protocol.json")
    check_hash(OUT / "protocol.json", p["parent_sha256"])
    lease = (
        gpu_lease(
            gpu_peak_bytes=2 << 30,
            disk_temp_bytes=200 << 20,
            disk_persistent_bytes=500 << 20,
        )
        if device == "cuda"
        else nullcontext({"device": "cpu"})
    )
    with lease as gate:
        write(DEST / "compute_gate.json", gate)
        engine = SyncNetEngine(batch_size=64, device=device)
        assert engine.model_hash == read(OUT / "protocol.json")["model_sha256"]
        try:
            for row in read(OUT / "protocol.json")["rows"]:
                key = row["key"]
                dest = DEST / "scores" / f"{key}.json"
                if dest.exists():
                    continue
                if device == "cuda" and set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError("foreign GPU process")
                receipt = read(OUT / "audio" / key / "receipt.json")
                waves = {}
                for name, mode in [("pcm", "pcm"), ("codec", p["codec"])]:
                    info = receipt["modes"][mode]
                    check_hash(info["path"], info["sha256"])
                    waves[name], sr = sf.read(info["path"], dtype="int16")
                    assert sr == 16000
                length = min(map(len, waves.values()))
                s, c = (waves[k][:length].astype(float) for k in ["pcm", "codec"])
                difference = s - c
                high = difference - lowpass(difference, p["cutoff_hz"])
                rms_gain = np.linalg.norm(s) / np.linalg.norm(c)
                conditions = {
                    "pcm": s,
                    "codec": c,
                    "restore_high": c + high,
                    "damage_high": s - high,
                    "restore_rms": c * rms_gain,
                }
                n = read(OUT / "scores" / f"{key}.json")["common_rows"]
                paths, features = {}, {}
                for mode, wave in conditions.items():
                    paths[mode] = pcm_write(DEST / "audio" / key / f"{mode}.wav", wave)
                    features[mode], _ = engine.extract_audio(paths[mode]["path"])
                    features[mode] = features[mode][:n]
                check_hash(row["visual"], row["visual_sha256"])
                v = np.load(row["visual"])[:n]
                matrices = {
                    m: engine.distance_matrix(v, a) for m, a in features.items()
                }
                idx = int(matrices["pcm"][20:-20].mean(0).argmin())
                cells = {m: metric(x, idx) for m, x in matrices.items()}
                previous = read(OUT / "scores" / f"{key}.json")
                replay = {
                    m: abs(cells[m]["C"] - previous["cells"][mode]["C"])
                    for m, mode in [("pcm", "pcm"), ("codec", p["codec"])]
                }
                if max(replay.values()) > 1e-4:
                    raise ValueError(f"score replay failed {replay}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                write(
                    dest,
                    {
                        "key": key,
                        "arm": row["arm"],
                        "cells": cells,
                        "paths": paths,
                        "replay_errors": replay,
                        "rms_gain": float(rms_gain),
                        "matrix_sha256": sha(dest.with_suffix(".npz")),
                    },
                )
                print("band-swapped", key, flush=True)
        finally:
            engine.close()


def analyze():
    groups = {}
    rows = [
        read(DEST / "scores" / f"{r['key']}.json")
        for r in read(OUT / "protocol.json")["rows"]
    ]
    for group, arm in [("natural", "N"), ("tts", "T"), ("real", "R")]:
        rr = [r for r in rows if r["arm"] == arm]
        stats = {}
        for mode in rr[0]["cells"]:
            for base in ("pcm", "codec"):
                stats[mode + "_minus_" + base] = interval(
                    [r["cells"][mode]["C"] - r["cells"][base]["C"] for r in rr]
                )
        groups[group] = stats
    write(
        DEST / "analysis.json",
        {
            "groups": groups,
            "max_replay_error": max(
                e for r in rows for e in r["replay_errors"].values()
            ),
        },
    )
    print("band analysis complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["freeze", "run", "analyze"])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.stage == "run":
        run(a.device)
    else:
        globals()[a.stage]()
