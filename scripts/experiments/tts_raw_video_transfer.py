"""Natural-audio transfer by retiming original TTS video, bypassing vocoding."""

from __future__ import annotations

import argparse
import os

import numpy as np

from scripts.experiments.tts_independent_visual import (
    OUT,
    PARENT,
    ROOT,
    read,
    sha,
    write,
)
from scripts.experiments.tts_pcm_residual import cluster, metrics
from scripts.experiments.tts_pcm_timing import decode
from scripts.experiments.tts_visual_ctc_timing import posterior_times, words

BASE = OUT / "raw_transfer"


def freeze():
    p = {
        "parent_sha256": sha(OUT / "protocol.json"),
        "source_pcm_audit_sha256": sha(PARENT / "protocol.json"),
        "question": "Does original TTS video contain transferable benefit that MFA-linear audio feature mapping/vocoding lost?",
        "selection": "all68 eligible visual cohort; fixed26/42 split, no outcome filtering; strict matching source N/T MFA word sequences",
        "primary": "42 evaluation clips: original Tvideo independently retimed to natural clock, scored against exact original N PCM; compare with processed N-video baseline, guard20; guard0/15 sensitivity",
        "arms": "N identity; original T uniform-duration resampling; original T word-center MFA map N-audio->T-audio; optional T visual-character Nvideo->Tvideo map ONLY IF prior independent visual timing gate passes",
        "mapping": "piecewise linear centers with zero/duration endpoints; 25fps linear pixel interpolation and lossless FFV1; same original official face crop per source; no new TFG/audio generation, no SyncNet optimization",
        "limits": "individual historical face crops for N/T; full-frame retiming changes head motion too; interpolation artifacts; current visual-content measurement is not an absolute human quality truth",
        "bootstrap": "speaker equal, 20000, seed20260926,95/99%; primary evaluation and exploratory all",
    }
    path = BASE / "protocol.json"
    if path.exists() and read(path) != p:
        raise ValueError("raw transfer protocol differs")
    write(path, p)
    print("raw video transfer frozen", flush=True)


def prepare():
    import cv2

    cv2.setNumThreads(1)
    old = {r["id"]: r for r in read(PARENT / "protocol.json")["rows"]}
    aligned = read(
        ROOT
        / "runs/aishell1_qwen_mfa_linear_n100_20260816/02_mfa_mandarin341_ready/mfa_summary.json"
    )["records"]
    rows = []
    missing = []
    for row in read(OUT / "protocol.json")["rows"]:
        sid = row["id"]
        r = old[sid]
        folder = BASE / "videos" / sid
        receipt = folder / "receipt.json"
        if receipt.exists():
            rows.append(read(receipt))
            continue
        anchors = {}
        for arm, label in [("N", "natural"), ("T", "tts")]:
            a = aligned[sid][label]
            assert a["audio_sha256"] == r["cells"][arm]["audio_sha256"]
            assert sha(a["textgrid"]) == a["textgrid_sha256"]
            anchors[arm] = words(a["textgrid"])
        if [v[0] for v in anchors["N"]] != [v[0] for v in anchors["T"]]:
            missing.append({"id": sid, "reason": "MFA word sequence mismatch"})
            continue
        frames = {}
        for arm in ("N", "T"):
            c = r["cells"][arm]
            assert c["track_start"] == 0 and sha(c["crop"]) == c["crop_sha256"]
            frames[arm] = decode(c["crop"])
        dn = r["source_lengths"]["N"] / 16000
        dt = r["source_lengths"]["T"] / 16000
        target = np.arange(len(frames["N"])) / 25
        x = np.array([0] + [v[1] for v in anchors["N"]] + [dn])
        y = np.array([0] + [v[1] for v in anchors["T"]] + [dt])
        assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
        out = {
            "id": sid,
            "speaker": row["speaker"],
            "split": row["split"],
            "mfa_x": x.tolist(),
            "mfa_y": y.tolist(),
            "videos": {},
        }
        for name, arm, source in [
            ("N_identity", "N", target),
            ("T_uniform", "T", target * dt / dn),
            ("T_mfa", "T", np.interp(target, x, y)),
        ]:
            path = folder / f"{name}.avi"
            render(frames[arm], source, path)
            out["videos"][name] = {
                "path": str(path),
                "sha256": sha(path),
                "source_crop_sha256": r["cells"][arm]["crop_sha256"],
                "frames": len(target),
            }
        write(receipt, out)
        rows.append(out)
        print("raw transfer prepared", sid, flush=True)
    write(BASE / "manifest.json", {"rows": rows, "missing": missing})


def render(frames, source, path):
    import cv2

    pos = np.clip(source * 25, 0, len(frames) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(frames) - 1)
    w = (pos - lo)[:, None, None, None]
    pixels = np.clip(np.rint(frames[lo] * (1 - w) + frames[hi] * w), 0, 255).astype(
        np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (224, 224))
    assert writer.isOpened()
    for frame in pixels:
        writer.write(frame)
    writer.release()


def add_visual():
    gate = read(OUT / "timing/analysis.json")["timing_gate"]
    write(
        BASE / "visual_gate.json",
        {
            "enabled": gate,
            "source_sha256": sha(OUT / "timing/analysis.json"),
            "reason": "known perturbation and content gates required, unchanged thresholds",
        },
    )
    if not gate:
        print("visual retiming not enabled: timing gate failed", flush=True)
        return
    old = {r["id"]: r for r in read(PARENT / "protocol.json")["rows"]}
    p = {r["id"]: r for r in read(OUT / "protocol.json")["rows"]}
    manifest = read(BASE / "manifest.json")
    for row in manifest["rows"]:
        sid = row["id"]
        source = old[sid]
        centers = {}
        receipt = read(OUT / "receipts" / f"{sid}.json")
        for arm in ("N", "T"):
            f = OUT / "features" / sid / f"{arm}_native.npz"
            assert sha(f) == receipt["feature_hashes"][arm + "_native"]
            centers[arm], _ = posterior_times(np.load(f)["logp"], p[sid]["target"])
        dn = source["source_lengths"]["N"] / 16000
        dt = source["source_lengths"]["T"] / 16000
        x = np.r_[0, centers["N"], dn]
        y = np.r_[0, centers["T"], dt]
        assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
        target = np.arange(row["videos"]["N_identity"]["frames"]) / 25
        frames = decode(source["cells"]["T"]["crop"])
        path = BASE / "videos" / sid / "T_visual.avi"
        render(frames, np.interp(target, x, y), path)
        row["videos"]["T_visual"] = {
            "path": str(path),
            "sha256": sha(path),
            "frames": len(target),
            "source_crop_sha256": source["cells"]["T"]["crop_sha256"],
        }
        row["visual_x"] = x.tolist()
        row["visual_y"] = y.tolist()
        write(BASE / "videos" / sid / "receipt.json", row)
    write(BASE / "manifest.json", manifest)


def score():
    import torch

    from scripts.experiments.tts_native_gain_attribution.common import (
        gpu_compute_pids,
        gpu_lease,
    )
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    old = {r["id"]: r for r in read(PARENT / "protocol.json")["rows"]}
    with gpu_lease(
        gpu_peak_bytes=3 << 30, disk_temp_bytes=300 << 20, disk_persistent_bytes=2 << 30
    ) as gate:
        write(BASE / "compute_gate.json", gate)
        engine = SyncNetEngine(batch_size=32, device="cuda")
        try:
            for row in read(BASE / "manifest.json")["rows"]:
                sid = row["id"]
                dest = BASE / "scores" / f"{sid}.json"
                if dest.exists():
                    continue
                if set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError("foreign GPU process")
                audiofile = PARENT / "features" / sid / "features.npz"
                assert (
                    sha(audiofile)
                    == read(PARENT / "scores" / f"{sid}.json")["features_sha256"]
                )
                audio = np.load(audiofile)["a_N_source"]
                visual = {}
                for name, info in row["videos"].items():
                    assert sha(info["path"]) == info["sha256"]
                    visual[name], _ = engine.extract_visual(info["path"])
                n = min(len(audio), *(len(v) for v in visual.values()))
                matrices = {
                    name: engine.distance_matrix(v[:n], audio[:n])
                    for name, v in visual.items()
                }
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                np.savez_compressed(dest.with_name(sid + "_visual.npz"), **visual)
                write(
                    dest,
                    {
                        "id": sid,
                        "speaker": row["speaker"],
                        "split": row["split"],
                        "rows": n,
                        "cells": {k: metrics(v) for k, v in matrices.items()},
                        "matrix_sha256": sha(dest.with_suffix(".npz")),
                        "visual_sha256": sha(dest.with_name(sid + "_visual.npz")),
                        "natural_audio_sha256": old[sid]["cells"]["N"]["audio_sha256"],
                    },
                )
                print("raw transfer scored", sid, flush=True)
        finally:
            engine.close()


def analyze():
    rows = [
        read(BASE / "scores" / f"{r['id']}.json")
        for r in read(BASE / "manifest.json")["rows"]
    ]
    result = {"guards": {}}
    for guard in ("0", "15", "20"):
        result["guards"][guard] = {}
        for split in ("evaluation", "calibration", "all"):
            rr = [r for r in rows if split == "all" or r["split"] == split]
            out = {}
            for left, right in [
                ("T_uniform", "N_identity"),
                ("T_mfa", "N_identity"),
                ("T_mfa", "T_uniform"),
                ("T_visual", "N_identity"),
                ("T_visual", "T_mfa"),
            ]:
                group = [r for r in rr if left in r["cells"] and right in r["cells"]]
                if not group:
                    continue
                out[left + "_minus_" + right] = cluster(
                    [
                        r["cells"][left][guard]["C"] - r["cells"][right][guard]["C"]
                        for r in group
                    ],
                    [r["speaker"] for r in group],
                )
            result["guards"][guard][split] = out
    write(BASE / "analysis.json", result)
    print("raw transfer analysis complete", flush=True)
    for k, v in result["guards"]["20"]["evaluation"].items():
        print(k, v["speaker_mean"], v["speaker_ci95"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["freeze", "prepare", "add_visual", "score", "analyze"]
    )
    globals()[p.parse_args().stage]()
