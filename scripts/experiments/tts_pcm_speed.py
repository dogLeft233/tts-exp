"""Joint audiovisual speed intervention without changing the generator."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scripts.experiments.tts_clock_cross import check_hash, read, sha, write
from scripts.experiments.tts_pcm_residual import OUT, cluster, metrics
from scripts.experiments.tts_pcm_timing import decode

BASE = OUT / "speed"


def freeze():
    protocol = {
        "parent_sha256": sha(OUT / "protocol.json"),
        "reason": "All 74 TTS sources shorter than natural (median ratio 0.819); residual C mostly wrong-offset background B. Test audiovisual temporal-scale sensitivity.",
        "interventions": "N and T original crops and PCM synchronously rescaled: identity; each to geometric-mean pair duration; each to other arm duration (swap). FFmpeg atempo for pitch-preserving audio, linear pixel interpolation + FFV1 for video.",
        "selection": "all 74 eligible pairs; no outcome filtering; actual ratios within [0.5,2]",
        "support": "each condition keeps whole utterance; native own support with guards0/15/20; equal-duration mode additionally common window count. Speed alters physical lag meaning and receptive-field phonetic span: intended intervention.",
        "primary": "speaker-equal paired delta T-N and difference from identity, guard20; report guard0/15, actual duration and interpolation artifacts",
        "limits": "Existing generator outputs only; not regeneration; atempo/interpolation are possible intervention artifacts; does not isolate speaking rate from pause density or independent perceptual quality",
    }
    path = BASE / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("speed protocol differs")
    write(path, protocol)
    print("speed frozen", flush=True)


def prepare_row(row):
    import cv2

    cv2.setNumThreads(1)
    sid = row["id"]
    folder = BASE / "assets" / sid
    dest = folder / "receipt.json"
    if dest.exists():
        return sid
    folder.mkdir(parents=True, exist_ok=True)
    durations = {a: row["source_lengths"][a] / 16000 for a in ("N", "T")}
    common = np.sqrt(durations["N"] * durations["T"])
    receipt = {}
    for arm in ("N", "T"):
        cell = row["cells"][arm]
        assert cell["track_start"] == 0
        check_hash(cell["crop"], cell["crop_sha256"])
        check_hash(cell["audio"], cell["audio_sha256"])
        frames = decode(cell["crop"])
        for mode, target in [
            ("identity", durations[arm]),
            ("equal", common),
            ("swap", durations["T" if arm == "N" else "N"]),
        ]:
            speed = durations[arm] / target
            if not 0.5 <= speed <= 2:
                raise ValueError("unexpected speed range")
            n = round(target * 16000)
            key = f"{arm}_{mode}"
            audio = folder / f"{key}.wav"
            video = folder / f"{key}.avi"
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                cell["audio"],
                "-af",
                f"atempo={speed:.12f},apad,atrim=end_sample={n}",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(audio),
            ]
            subprocess.run(cmd, check=True)
            # Preserve the original video/audio end discrepancy under the same scale.
            count = int(np.floor(len(frames) / speed))
            pos = np.minimum(np.arange(count) * speed, len(frames) - 1)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, len(frames) - 1)
            w = (pos - lo)[:, None, None, None]
            pixels = np.clip(
                np.rint(frames[lo] * (1 - w) + frames[hi] * w), 0, 255
            ).astype(np.uint8)
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"FFV1"), 25, (224, 224)
            )
            if not writer.isOpened():
                raise RuntimeError("FFV1 unavailable")
            for frame in pixels:
                writer.write(frame)
            writer.release()
            receipt[key] = {
                "audio": str(audio),
                "audio_sha256": sha(audio),
                "video": str(video),
                "video_sha256": sha(video),
                "speed": float(speed),
                "target_duration": float(target),
                "audio_samples": n,
                "video_frames": count,
                "command": cmd,
            }
    write(dest, receipt)
    return sid


def prepare():
    with ThreadPoolExecutor(max_workers=2) as pool:
        for sid in pool.map(prepare_row, read(OUT / "protocol.json")["rows"]):
            print("speed prepared", sid, flush=True)


def score():
    import torch

    from scripts.experiments.tts_native_gain_attribution.common import (
        gpu_compute_pids,
        gpu_lease,
    )
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    with gpu_lease(
        gpu_peak_bytes=3 << 30, disk_temp_bytes=300 << 20, disk_persistent_bytes=2 << 30
    ) as gate:
        write(BASE / "compute_gate.json", gate)
        engine = SyncNetEngine(batch_size=32, device="cuda")
        try:
            for row in read(OUT / "protocol.json")["rows"]:
                sid = row["id"]
                dest = BASE / "scores" / f"{sid}.json"
                if dest.exists():
                    continue
                if set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError("foreign GPU process")
                receipt = read(BASE / "assets" / sid / "receipt.json")
                features = {}
                matrices = {}
                for key, info in receipt.items():
                    for name in ("audio", "video"):
                        check_hash(info[name], info[name + "_sha256"])
                    v, _ = engine.extract_visual(info["video"])
                    a, _ = engine.extract_audio(info["audio"])
                    n = min(len(v), len(a))
                    features["v_" + key] = v
                    features["a_" + key] = a
                    matrices[key] = engine.distance_matrix(v[:n], a[:n])
                n = min(len(matrices["N_equal"]), len(matrices["T_equal"]))
                for arm in ("N", "T"):
                    key = arm + "_equal"
                    matrices[key + "_common"] = engine.distance_matrix(
                        features["v_" + key][:n], features["a_" + key][:n]
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                np.savez_compressed(dest.with_name(sid + "_features.npz"), **features)
                write(
                    dest,
                    {
                        "id": sid,
                        "speaker": row["speaker"],
                        "cells": {k: metrics(v) for k, v in matrices.items()},
                        "matrix_sha256": sha(dest.with_suffix(".npz")),
                        "features_sha256": sha(dest.with_name(sid + "_features.npz")),
                    },
                )
                print("speed scored", sid, flush=True)
        finally:
            engine.close()


def analyze():
    rows = [
        read(BASE / "scores" / f"{r['id']}.json")
        for r in read(OUT / "protocol.json")["rows"]
    ]
    result = {"guards": {}}
    for guard in ("0", "15", "20"):
        result["guards"][guard] = {}
        for metric in ("C", "B", "D"):
            values = {
                mode: np.array(
                    [
                        r["cells"]["T_" + mode][guard][metric]
                        - r["cells"]["N_" + mode][guard][metric]
                        for r in rows
                    ]
                )
                for mode in ("identity", "equal", "equal_common", "swap")
            }
            result["guards"][guard][metric] = {
                key: cluster(x.tolist(), [r["speaker"] for r in rows])
                for key, x in {
                    **values,
                    **{
                        mode + "_minus_identity": x - values["identity"]
                        for mode, x in values.items()
                        if mode != "identity"
                    },
                }.items()
            }
    write(BASE / "analysis.json", result)
    print("speed analysis complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["freeze", "prepare", "score", "analyze"])
    globals()[p.parse_args().stage]()
