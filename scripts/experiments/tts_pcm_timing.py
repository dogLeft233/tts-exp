"""Audio-only forced-alignment intervention on natural-clock video timing."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from scripts.experiments.tts_clock_cross import check_hash, read, sha, write
from scripts.experiments.tts_pcm_residual import OUT, PARENT, cluster, metrics

BASE = OUT / "timing"
MFA = Path("/home/wjj/miniconda3/envs/mfa3/bin/mfa")
MODELS = Path("/home/wjj/Documents/MFA/pretrained_models")


def prepare():
    protocol = {
        "parent_sha256": sha(OUT / "protocol.json"),
        "alignment": "MFA 3.4.1; actual N/M/D audio; common character transcript; separate speaker-arm directories; 4 jobs",
        "mapping": "strictly monotone piecewise-linear word-center N-time to M-time; endpoints 0 and exact duration; all lexical labels must match; no SyncNet tuning",
        "intervention": "N-trajectory common ROI; decoded pixels linearly interpolated at mapped source times, FFV1 lossless; identity, audio-derived forward map, inverse-map negative control; N video mapped with M map as damage control; D subset same procedure",
        "selection": "all 74 parent pairs; alignment failures/label mismatch excluded before video scoring and reported; no score filtering",
        "primary": "guard20, all arms shared absolute support; corrected Mvideo + natural audio vs identity Mvideo and identity Nvideo; speaker bootstrap",
        "limits": "MFA estimated timing is not human ground truth; mapping resamples all facial motion; neither success nor failure proves perceptual quality",
        "dictionary_sha256": sha(MODELS / "dictionary/mandarin_china_mfa.dict"),
        "acoustic_sha256": sha(MODELS / "acoustic/mandarin_mfa.zip"),
    }
    path = BASE / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("timing protocol differs")
    write(path, protocol)
    records = read(PARENT / "02_mfa_mandarin341_ready/mfa_summary.json")["records"]
    for row in read(OUT / "protocol.json")["rows"]:
        for arm in ("N", "M", "D"):
            if arm not in row["cells"]:
                continue
            cell = row["cells"][arm]
            check_hash(cell["audio"], cell["audio_sha256"])
            root = BASE / "input" / f"{row['speaker']}_{arm}"
            root.mkdir(parents=True, exist_ok=True)
            dest = root / f"{row['id']}_{arm}.wav"
            if not dest.exists():
                shutil.copyfile(cell["audio"], dest)
            (root / f"{row['id']}_{arm}.lab").write_text(
                records[row["id"]]["cleaned_lab_text"], encoding="utf-8"
            )
    print("timing design frozen; input prepared", flush=True)


def align():
    env = dict(os.environ)
    env["PATH"] = str(MFA.parent) + os.pathsep + env["PATH"]
    env["MFA_ROOT_DIR"] = str(BASE / "mfa_runtime")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    cmd = [
        str(MFA),
        "align",
        str(BASE / "input"),
        str(MODELS / "dictionary/mandarin_china_mfa.dict"),
        str(MODELS / "acoustic/mandarin_mfa.zip"),
        str(BASE / "textgrids"),
        "--clean",
        "--num_jobs",
        "4",
    ]
    write(
        BASE / "align_command.json",
        {"command": cmd, "MFA_ROOT_DIR": env["MFA_ROOT_DIR"]},
    )
    with (BASE / "align.log").open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("MFA completed", flush=True)


def words(path):
    content = path.read_text(encoding="utf-8")
    tier = content.split('name = "words"', 1)[1].split("item [", 1)[0]
    intervals = re.findall(
        r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',
        tier,
    )
    return [
        (label, (float(start) + float(end)) / 2)
        for start, end, label in intervals
        if label.strip()
    ]


def maps():
    accepted, rejected = [], []
    for row in read(OUT / "protocol.json")["rows"]:
        sid = row["id"]
        anchors = {}
        hashes = {}
        for arm in ("N", "M", "D"):
            if arm not in row["cells"]:
                continue
            path = (
                BASE / "textgrids" / f"{row['speaker']}_{arm}" / f"{sid}_{arm}.TextGrid"
            )
            if path.exists():
                anchors[arm] = words(path)
                hashes[arm] = sha(path)
        if any(a not in anchors for a in ("N", "M")):
            rejected.append({"id": sid, "reason": "missing_alignment"})
            continue
        out = {
            "id": sid,
            "speaker": row["speaker"],
            "textgrid_hashes": hashes,
            "maps": {},
        }
        for arm in ("M", "D"):
            if arm not in anchors:
                continue
            n, m = anchors["N"], anchors[arm]
            if [a[0] for a in n] != [a[0] for a in m] or not n:
                out[arm + "_failure"] = "lexical_label_mismatch"
                continue
            duration = row["source_lengths"]["N"] / 16000
            x = np.array([0] + [a[1] for a in n] + [duration])
            y = np.array([0] + [a[1] for a in m] + [duration])
            if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
                out[arm + "_failure"] = "nonmonotone_anchors"
                continue
            out["maps"][arm] = {
                "target_N": x.tolist(),
                "source": y.tolist(),
                "labels": [a[0] for a in n],
                "median_absolute_shift_ms": float(
                    np.median(np.abs(y[1:-1] - x[1:-1])) * 1000
                ),
                "median_signed_shift_ms": float(np.median(y[1:-1] - x[1:-1]) * 1000),
                "max_absolute_shift_ms": float(np.max(np.abs(y - x)) * 1000),
            }
        if "M" not in out["maps"]:
            rejected.append(out)
        else:
            accepted.append(out)
    write(BASE / "maps.json", {"accepted": accepted, "rejected": rejected})
    print("maps", len(accepted), "accepted", len(rejected), "rejected", flush=True)


def decode(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return np.asarray(frames)


def warp():
    import cv2

    cv2.setNumThreads(1)
    for row in read(BASE / "maps.json")["accepted"]:
        sid = row["id"]
        folder = BASE / "videos" / sid
        receipt = folder / "receipt.json"
        if receipt.exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        roi = read(OUT / "controls/roi" / sid / "receipt.json")
        result = {}
        for arm in ("N", "M", "D"):
            if arm == "D" and "D" not in row["maps"]:
                continue
            source = roi[f"N_{arm}"]
            check_hash(source["path"], source["sha256"])
            frames = decode(source["path"])
            assert len(frames) == source["frames"]
            t = (source["start"] + np.arange(len(frames))) / 25
            mapping = row["maps"]["M" if arm == "N" else arm]
            x, y = mapping["target_N"], mapping["source"]
            modes = (
                ("identity", "forward")
                if arm == "N"
                else ("identity", "forward", "inverse")
            )
            for mode in modes:
                target = (
                    t
                    if mode == "identity"
                    else np.interp(
                        t, x if mode == "forward" else y, y if mode == "forward" else x
                    )
                )
                pos = np.clip(target * 25 - source["start"], 0, len(frames) - 1)
                lo = np.floor(pos).astype(int)
                hi = np.minimum(lo + 1, len(frames) - 1)
                weight = (pos - lo)[:, None, None, None]
                pixels = np.clip(
                    np.rint(frames[lo] * (1 - weight) + frames[hi] * weight), 0, 255
                ).astype(np.uint8)
                key = f"{arm}_{mode}"
                path = folder / f"{key}.avi"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (224, 224)
                )
                if not writer.isOpened():
                    raise RuntimeError("FFV1 unavailable")
                for frame in pixels:
                    writer.write(frame)
                writer.release()
                result[key] = {
                    "path": str(path),
                    "sha256": sha(path),
                    "source_sha256": source["sha256"],
                    "start": source["start"],
                    "frames": len(frames),
                    "max_shift_ms": float(np.max(np.abs(target - t)) * 1000),
                }
        write(receipt, result)
        print("retimed", sid, flush=True)


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
            for row in read(BASE / "maps.json")["accepted"]:
                sid = row["id"]
                dest = BASE / "scores" / f"{sid}.json"
                if dest.exists():
                    continue
                if set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError("foreign GPU process")
                receipt = read(BASE / "videos" / sid / "receipt.json")
                cached = OUT / "features" / sid / "features.npz"
                check_hash(
                    cached, read(OUT / "scores" / f"{sid}.json")["features_sha256"]
                )
                audio = np.load(cached)["a_N_source"]
                visual = {}
                for key, info in receipt.items():
                    check_hash(info["path"], info["sha256"])
                    visual[key], _ = engine.extract_visual(info["path"])
                start = max(v["start"] for v in receipt.values())
                end = min(
                    len(audio),
                    *(receipt[k]["start"] + len(v) for k, v in visual.items()),
                )
                matrices = {
                    k: engine.distance_matrix(
                        v[start - receipt[k]["start"] : end - receipt[k]["start"]],
                        audio[start:end],
                    )
                    for k, v in visual.items()
                }
                old_visual = np.load(OUT / "controls/roi_scores" / f"{sid}_visual.npz")
                replay = {
                    a: float(
                        np.max(np.abs(visual[f"{a}_identity"] - old_visual[f"N_{a}"]))
                    )
                    for a in ("N", "M")
                }
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                np.savez_compressed(dest.with_name(sid + "_visual.npz"), **visual)
                write(
                    dest,
                    {
                        "id": sid,
                        "speaker": row["speaker"],
                        "support": [start, end],
                        "identity_feature_error": replay,
                        "cells": {k: metrics(v) for k, v in matrices.items()},
                        "matrix_sha256": sha(dest.with_suffix(".npz")),
                        "visual_sha256": sha(dest.with_name(sid + "_visual.npz")),
                    },
                )
                print(
                    "timing scored",
                    sid,
                    "identity error",
                    max(replay.values()),
                    flush=True,
                )
        finally:
            engine.close()


def analyze():
    maps_data = read(BASE / "maps.json")
    rows = [read(BASE / "scores" / f"{r['id']}.json") for r in maps_data["accepted"]]
    result = {"accepted": len(rows), "rejected": maps_data["rejected"], "guards": {}}
    for guard in ("0", "15", "20"):
        comparisons = {}
        for name, left, right in [
            ("M_rescue", "M_forward", "M_identity"),
            ("M_after_vs_N", "M_forward", "N_identity"),
            ("M_before_vs_N", "M_identity", "N_identity"),
            ("M_inverse_effect", "M_inverse", "M_identity"),
            ("M_forward_vs_inverse", "M_forward", "M_inverse"),
            ("N_damage", "N_forward", "N_identity"),
            ("D_rescue", "D_forward", "D_identity"),
            ("D_after_vs_N", "D_forward", "N_identity"),
        ]:
            rr = [r for r in rows if left in r["cells"] and right in r["cells"]]
            comparisons[name] = cluster(
                [
                    r["cells"][left][guard]["C"] - r["cells"][right][guard]["C"]
                    for r in rr
                ],
                [r["speaker"] for r in rr],
            )
        result["guards"][guard] = comparisons
    result["max_identity_feature_error"] = max(
        v for r in rows for v in r["identity_feature_error"].values()
    )
    result["M_word_shift"] = {
        key: cluster(
            [r["maps"]["M"][key] for r in maps_data["accepted"]],
            [r["speaker"] for r in maps_data["accepted"]],
        )
        for key in (
            "median_absolute_shift_ms",
            "median_signed_shift_ms",
            "max_absolute_shift_ms",
        )
    }
    write(BASE / "analysis.json", result)
    print("timing analysis complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["prepare", "align", "maps", "warp", "score", "analyze"]
    )
    globals()[parser.parse_args().stage]()
