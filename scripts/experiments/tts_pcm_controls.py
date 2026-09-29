"""Spatial common-crop and evaluation-bandwidth controls for residual audit."""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.experiments.syncnet_codec_mechanism import lowpass, pcm_write
from scripts.experiments.tts_clock_cross import check_hash, read, sha, write
from scripts.experiments.tts_pcm_residual import OUT, cluster, metrics


def freeze():
    p = {
        "parent_sha256": sha(OUT / "protocol.json"),
        "selection": "all74 asset-eligible pairs, no interim-score selection",
        "spatial": "N and M face-crop trajectories each applied to both videos (plus D if present); unchanged parent JPEG frames; own-trajectory replay check",
        "audio": "same raw videos evaluated with sourcePCM and zero-phase LP3500/5500/7500; no generation change",
        "support": "ROI per-plan shared absolute frame support; band controls same source window support per arm",
        "reason": "Residual TTS advantage and MFA self-score may survive PCM; separate codec result from remaining audio/video compatibility and guard against changing face crops.",
        "limits": "exploratory post-parent diagnostic, single common dynamic face; no human truth",
    }
    path = OUT / "controls/protocol.json"
    if path.exists() and read(path) != p:
        raise ValueError("frozen controls differ")
    write(path, p)
    print("controls frozen", flush=True)


def crop_row(row):
    import cv2

    cv2.setNumThreads(1)
    root = OUT / "controls/roi" / row["id"]
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "receipt.json"
    if receipt.exists():
        return row["id"]
    crops = {}
    for plan in ("N", "M"):
        pc = row["cells"][plan]
        check_hash(pc["tracks"], pc["tracks_sha256"])
        with open(pc["tracks"], "rb") as f:
            track = pickle.load(f)[0]
        for arm in (a for a in ("N", "M", "D") if a in row["cells"]):
            c = row["cells"][arm]
            crop_path = Path(c["crop"])
            frames_dir = crop_path.parents[2] / "pyframes" / crop_path.parent.name
            output = root / f"{plan}_{arm}.avi"
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
            )
            if not writer.isOpened():
                raise RuntimeError("video writer unavailable")
            digest = hashlib.sha256()
            used = []
            for idx, frame_no in enumerate(track["track"]["frame"]):
                file = frames_dir / f"{int(frame_no) + 1:06d}.jpg"
                if not file.exists():
                    break
                raw = file.read_bytes()
                digest.update(raw)
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                d = track["proc_track"]
                bs = d["s"][idx]
                pad = int(bs * 1.8)
                image = np.pad(
                    image,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode="constant",
                    constant_values=110,
                )
                y = d["y"][idx] + pad
                x = d["x"][idx] + pad
                face = image[
                    int(y - bs) : int(y + bs * 1.8),
                    int(x - bs * 1.4) : int(x + bs * 1.4),
                ]
                if not face.size:
                    raise ValueError("empty crop")
                writer.write(cv2.resize(face, (224, 224)))
                used.append(int(frame_no))
            writer.release()
            if len(used) < 50 or used != list(range(used[0], used[-1] + 1)):
                raise ValueError("bad crop support")
            crops[f"{plan}_{arm}"] = {
                "path": str(output),
                "sha256": sha(output),
                "start": used[0],
                "frames": len(used),
                "input_jpeg_sequence_sha256": digest.hexdigest(),
                "frames_dir": str(frames_dir),
                "plan_track_sha256": pc["tracks_sha256"],
            }
    write(receipt, crops)
    return row["id"]


def prepare_roi():
    with ThreadPoolExecutor(max_workers=3) as pool:
        for sid in pool.map(crop_row, read(OUT / "protocol.json")["rows"]):
            print("cropped", sid, flush=True)


def engine_context(device):
    from scripts.experiments.tts_native_gain_attribution.common import gpu_lease

    return (
        gpu_lease(
            gpu_peak_bytes=3 << 30,
            disk_temp_bytes=300 << 20,
            disk_persistent_bytes=2 << 30,
        )
        if device == "cuda"
        else nullcontext({"device": "cpu"})
    )


def score_roi(device):
    import torch

    from scripts.experiments.tts_native_gain_attribution.common import gpu_compute_pids
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    with engine_context(device) as gate:
        write(OUT / "controls/roi_compute.json", gate)
        engine = SyncNetEngine(batch_size=32, device=device)
        try:
            for row in read(OUT / "protocol.json")["rows"]:
                sid = row["id"]
                dest = OUT / "controls/roi_scores" / f"{sid}.json"
                if dest.exists():
                    continue
                if device == "cuda" and set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError("foreign GPU process")
                receipt = read(OUT / "controls/roi" / sid / "receipt.json")
                old = read(OUT / "scores" / f"{sid}.json")
                fpath = OUT / "features" / sid / "features.npz"
                check_hash(fpath, old["features_sha256"])
                cache = np.load(fpath)
                visual = {}
                for key, info in receipt.items():
                    check_hash(info["path"], info["sha256"])
                    visual[key], _ = engine.extract_visual(info["path"])
                replay = {}
                for arm in ("N", "M"):
                    a, b = visual[f"{arm}_{arm}"], cache[f"v_{arm}"]
                    if a.shape != b.shape:
                        raise ValueError("own-ROI replay shape")
                    replay[arm] = float(np.max(np.abs(a - b)))
                matrices = {}
                supports = {}
                for plan in ("N", "M"):
                    arms = [a for a in ("N", "M", "D") if a in row["cells"]]
                    start = max(receipt[f"{plan}_{a}"]["start"] for a in arms)
                    end = min(
                        *(
                            receipt[f"{plan}_{a}"]["start"] + len(visual[f"{plan}_{a}"])
                            for a in arms
                        ),
                        *(len(cache[f"a_{a}_source"]) for a in arms),
                    )
                    supports[plan] = [start, end]
                    for va in arms:
                        st = receipt[f"{plan}_{va}"]["start"]
                        v = visual[f"{plan}_{va}"][start - st : end - st]
                        for aa in arms:
                            matrices[f"{plan}_{va}_{aa}"] = engine.distance_matrix(
                                v, cache[f"a_{aa}_source"][start:end]
                            )
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                np.savez_compressed(dest.with_name(sid + "_visual.npz"), **visual)
                write(
                    dest,
                    {
                        "id": sid,
                        "speaker": row["speaker"],
                        "replay_feature_error": replay,
                        "support": supports,
                        "cells": {k: metrics(m) for k, m in matrices.items()},
                        "matrix_sha256": sha(dest.with_suffix(".npz")),
                        "visual_sha256": sha(dest.with_name(sid + "_visual.npz")),
                    },
                )
                print("ROI scored", sid, "replay", max(replay.values()), flush=True)
        finally:
            engine.close()


def score_bands(device):
    import torch

    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    with engine_context(device) as gate:
        write(OUT / "controls/band_compute.json", gate)
        engine = SyncNetEngine(batch_size=64, device=device)
        try:
            for row in read(OUT / "protocol.json")["rows"]:
                sid = row["id"]
                dest = OUT / "controls/band_scores" / f"{sid}.json"
                if dest.exists():
                    continue
                old = read(OUT / "scores" / f"{sid}.json")
                fpath = OUT / "features" / sid / "features.npz"
                check_hash(fpath, old["features_sha256"])
                cache = np.load(fpath)
                paths = {}
                matrices = {}
                for arm, c in row["cells"].items():
                    source = read(OUT / "pcm" / sid / "receipt.json")[f"{arm}_source"]
                    check_hash(source["path"], source["sha256"])
                    wave, sr = sf.read(source["path"], dtype="int16")
                    assert sr == 16000
                    for cut in (3500, 5500, 7500):
                        mode = f"{arm}_{cut}"
                        info = pcm_write(
                            OUT / "controls/band_audio" / sid / f"{mode}.wav",
                            lowpass(wave, cut),
                        )
                        paths[mode] = info
                        a, _ = engine.extract_audio(info["path"])
                        start = c["track_start"]
                        v = cache[f"v_{arm}"]
                        n = old["cells"][f"{arm}_source"]["0"]["rows"]
                        matrices[mode] = engine.distance_matrix(
                            v[:n], a[start : start + n]
                        )
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                write(
                    dest,
                    {
                        "id": sid,
                        "speaker": row["speaker"],
                        "cells": {k: metrics(m) for k, m in matrices.items()},
                        "paths": paths,
                        "matrix_sha256": sha(dest.with_suffix(".npz")),
                    },
                )
                print("bands scored", sid, flush=True)
        finally:
            engine.close()


def analyze():
    protocol = read(OUT / "protocol.json")
    result = {"roi": {}, "bands": {}}
    roi = [
        read(OUT / "controls/roi_scores" / f"{r['id']}.json") for r in protocol["rows"]
    ]
    bands = [
        read(OUT / "controls/band_scores" / f"{r['id']}.json") for r in protocol["rows"]
    ]
    for guard in ("0", "15", "20"):
        result["roi"][guard] = {}
        for plan in ("N", "M"):
            for name, left, right in [
                ("replacement", "M_N", "N_N"),
                ("self", "M_M", "N_N"),
                ("evaluator_on_N", "N_M", "N_N"),
                ("generator_on_M", "M_M", "N_M"),
                ("direct_replacement", "D_N", "N_N"),
                ("mfa_vs_direct", "M_N", "D_N"),
            ]:
                rr = [
                    r
                    for r in roi
                    if f"{plan}_{left}" in r["cells"]
                    and f"{plan}_{right}" in r["cells"]
                ]
                result["roi"][guard][plan + "_" + name] = cluster(
                    [
                        r["cells"][f"{plan}_{left}"][guard]["C"]
                        - r["cells"][f"{plan}_{right}"][guard]["C"]
                        for r in rr
                    ],
                    [r["speaker"] for r in rr],
                )
        result["bands"][guard] = {}
        for cut in (3500, 5500, 7500):
            result["bands"][guard][str(cut)] = cluster(
                [
                    r["cells"][f"T_{cut}"][guard]["C"]
                    - r["cells"][f"N_{cut}"][guard]["C"]
                    for r in bands
                ],
                [r["speaker"] for r in bands],
            )
    result["max_own_roi_feature_error"] = max(
        v for r in roi for v in r["replay_feature_error"].values()
    )
    write(OUT / "controls/analysis.json", result)
    print("controls analysis complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=["freeze", "prepare_roi", "score_roi", "score_bands", "analyze"],
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.stage.startswith("score_"):
        globals()[a.stage](a.device)
    else:
        globals()[a.stage]()
