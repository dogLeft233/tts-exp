"""Fixed static-source experiment: temporal warp before versus after Wav2Lip."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

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
from scripts.experiments.tts_raw_video_transfer import render

BASE = ROOT / "runs/tts_prepost_mel_20260926"
PHONE = OUT / "phone_transfer"
MODEL = ROOT / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"


def freeze():
    import cv2

    old = {r["id"]: r for r in read(PARENT / "protocol.json")["rows"]}
    image_protocol = ROOT / "runs/static_image_bridge_frontal_20260913/protocol.json"
    images = read(image_protocol)["images"]
    for image in images:
        pixels = cv2.imread(image["path"])
        assert (
            hashlib.sha256(
                cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB).tobytes()
            ).hexdigest()
            == image["rgb_sha256"]
        )
        image["sha256"] = sha(image["path"])
    rows = []
    for r in read(PHONE / "manifest.json")["rows"]:
        sid = r["id"]
        rows.append(
            {
                "id": sid,
                "speaker": r["speaker"],
                "map": r["maps"]["boundaries"],
                "audio": {
                    a: {
                        "path": old[sid]["cells"][a]["audio"],
                        "sha256": old[sid]["cells"][a]["audio_sha256"],
                    }
                    for a in ("N", "T")
                },
            }
        )
    p = {
        "rows": rows,
        "images": images,
        "parent_sha256": sha(PHONE / "manifest.json"),
        "image_protocol_sha256": sha(image_protocol),
        "model_sha256": sha(MODEL),
        "mel_frontend_sha256": sha(ROOT / "third_party/Wav2Lip/audio.py"),
        "arms": "N,T native mel; PRE Tmel->N clock before G; NRT Nmel->T->N damage; POST warp G(T); POST40 fixed40ms clock; no vocoder",
        "primary": "PRE-POST and PRE-N, guard20; mean3images within utterance then speaker equal;20000 bootstrap seed20260926;95/99%",
        "limits": "historical38 audio/13speakers,3 historical static images; new matched geometry protocol; no human grading; not unseen-source confirmation",
    }
    if (BASE / "protocol.json").exists():
        assert read(BASE / "protocol.json") == p
    write(BASE / "protocol.json", p)
    print("frozen", len(rows), "rows", len(images), "images", flush=True)


def mel_worker():
    sys.path.insert(0, str(ROOT / "third_party/Wav2Lip"))
    import audio

    p = read(BASE / "protocol.json")
    assert sha(ROOT / "third_party/Wav2Lip/audio.py") == p["mel_frontend_sha256"]
    for row in p["rows"]:
        m = {}
        for arm, info in row["audio"].items():
            assert sha(info["path"]) == info["sha256"]
            m[arm] = audio.melspectrogram(audio.load_wav(info["path"], 16000)).astype(
                np.float32
            )
        x, y = np.array(row["map"]["N"]), np.array(row["map"]["T"])
        tn, tt = np.arange(m["N"].shape[1]) / 80, np.arange(m["T"].shape[1]) / 80
        source = np.interp(tn, x, y)
        inverse = np.interp(tt, y, x)
        m["PRE"] = np.stack([np.interp(source, tt, band) for band in m["T"]]).astype(
            np.float32
        )
        intermediate = np.stack(
            [np.interp(inverse, tn, band) for band in m["N"]]
        ).astype(np.float32)
        m["NRT"] = np.stack(
            [np.interp(source, tt, band) for band in intermediate]
        ).astype(np.float32)
        file = BASE / "mels" / f"{row['id']}.npz"
        file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(file, **m)
        write(
            file.with_suffix(".json"),
            {
                "sha256": sha(file),
                "shapes": {k: list(v.shape) for k, v in m.items()},
                "source": source.tolist(),
                "inverse": inverse.tolist(),
            },
        )
        print("mel prepared", row["id"], flush=True)
    write(
        BASE / "mel_environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "frontend": audio.__file__,
        },
    )


def prepare():
    env = dict(
        os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1"
    )
    subprocess.run(
        [
            "/home/wjj/.venvs/wav2lip/bin/python",
            "-m",
            "scripts.experiments.tts_prepost_mel",
            "mel_worker",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def run():
    import cv2
    import torch

    from scripts.experiments.masked_tts_tfg_probe.direct_mel import (
        _load_model,
        chunk_mels,
    )
    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
    from scripts.experiments.tts_native_gain_attribution.common import (
        gpu_compute_pids,
        gpu_lease,
    )
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(20260926)
    np.random.seed(20260926)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    p = read(BASE / "protocol.json")
    assert sha(MODEL) == p["model_sha256"]
    with gpu_lease(
        gpu_peak_bytes=8 << 30, disk_temp_bytes=1 << 30, disk_persistent_bytes=5 << 30
    ) as gate:
        write(BASE / "compute_gate.json", gate)
        model = _load_model(MODEL, "cuda")
        engine = SyncNetEngine(batch_size=32, device="cuda")
        write(
            BASE / "environment.json",
            {
                "python": sys.version,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "batch_size": 32,
            },
        )
        try:
            for image in p["images"]:
                assert sha(image["path"]) == image["sha256"]
                frame = cv2.imread(image["path"])
                x1, y1, x2, y2 = image["generation_box"]
                face = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
                masked = face.copy()
                masked[48:] = 0
                channels = (
                    np.concatenate([masked, face], axis=2)
                    .transpose(2, 0, 1)
                    .astype(np.float32)
                    / 255
                )
                image_tensor = torch.from_numpy(channels).unsqueeze(0).to("cuda")
                outside = np.ones(frame.shape[:2], dtype=bool)
                outside[y1:y2, x1:x2] = False

                def generate(
                    chunks,
                    image_tensor=image_tensor,
                    frame=frame,
                    box=(x1, y1, x2, y2),
                    outside=outside,
                    score_box=image["score_box"]["box"],
                ):
                    x1, y1, x2, y2 = box
                    frames = []
                    for start in range(0, len(chunks), 32):
                        block = np.asarray(chunks[start : start + 32], dtype=np.float32)
                        mt = torch.from_numpy(block[:, None]).to("cuda")
                        with torch.inference_mode():
                            pred = (
                                model(mt, image_tensor.expand(len(block), -1, -1, -1))
                                .cpu()
                                .numpy()
                                .transpose(0, 2, 3, 1)
                                * 255
                            )
                        for faceout in pred:
                            full = frame.copy()
                            full[y1:y2, x1:x2] = cv2.resize(
                                faceout.astype(np.uint8), (x2 - x1, y2 - y1)
                            )
                            assert np.array_equal(full[outside], frame[outside])
                            frames.append(crop_zero_padded(full, score_box))
                    return np.stack(frames)

                for row in p["rows"]:
                    sid = row["id"]
                    dest = BASE / "scores" / image["id"] / f"{sid}.json"
                    if dest.exists():
                        continue
                    assert not set(gpu_compute_pids()) - {os.getpid()}
                    mf = BASE / "mels" / f"{sid}.npz"
                    assert sha(mf) == read(mf.with_suffix(".json"))["sha256"]
                    mels = np.load(mf)
                    frames = {
                        k: generate(chunk_mels(mels[k], 25))
                        for k in ("N", "T", "PRE", "NRT")
                    }
                    repeat = None
                    if sid == p["rows"][0]["id"]:
                        repeat = int(
                            np.max(
                                np.abs(
                                    generate(chunk_mels(mels["N"], 25)).astype(int)
                                    - frames["N"].astype(int)
                                )
                            )
                        )
                        assert repeat == 0
                    assert len(frames["N"]) == len(frames["PRE"]) == len(frames["NRT"])
                    t = np.arange(len(frames["N"])) / 25
                    x, y = np.array(row["map"]["N"]), np.array(row["map"]["T"])
                    videos = {}
                    for k, arr in frames.items():
                        path = BASE / "videos" / image["id"] / sid / f"{k}.avi"
                        render(arr, np.arange(len(arr)) / 25, path)
                        videos[k] = {
                            "path": str(path),
                            "sha256": sha(path),
                            "frames": len(arr),
                        }
                    for k, lag in [("POST", 0), ("POST40", 0.04)]:
                        path = BASE / "videos" / image["id"] / sid / f"{k}.avi"
                        render(frames["T"], np.interp(t + lag, x, y) - lag, path)
                        videos[k] = {
                            "path": str(path),
                            "sha256": sha(path),
                            "frames": len(t),
                        }
                    visuals = {
                        k: engine.extract_visual(info["path"])[0]
                        for k, info in videos.items()
                    }
                    af = PARENT / "features" / sid / "features.npz"
                    assert (
                        sha(af)
                        == read(PARENT / "scores" / f"{sid}.json")["features_sha256"]
                    )
                    audio = np.load(af)
                    n = min(
                        len(audio["a_N_source"]),
                        *(len(v) for k, v in visuals.items() if k != "T"),
                    )
                    matrices = {
                        k: engine.distance_matrix(v[:n], audio["a_N_source"][:n])
                        for k, v in visuals.items()
                        if k != "T"
                    }
                    nt = min(len(visuals["T"]), len(audio["a_T_source"]))
                    matrices["T"] = engine.distance_matrix(
                        visuals["T"][:nt], audio["a_T_source"][:nt]
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                    np.savez_compressed(dest.with_name(sid + "_visual.npz"), **visuals)
                    write(
                        dest,
                        {
                            "id": sid,
                            "speaker": row["speaker"],
                            "image": image["id"],
                            "videos": videos,
                            "repeat_pixel_error": repeat,
                            "outside_generation_box_identity": True,
                            "cells": {k: metrics(v) for k, v in matrices.items()},
                            "matrix_sha256": sha(dest.with_suffix(".npz")),
                            "visual_sha256": sha(dest.with_name(sid + "_visual.npz")),
                            "audio_features_sha256": sha(af),
                            "mel_sha256": sha(mf),
                            "model_sha256": p["model_sha256"],
                        },
                    )
                    print("static generated/scored", image["id"], sid, flush=True)
        finally:
            engine.close()


def analyze():
    p = read(BASE / "protocol.json")
    scored = {
        (im["id"], r["id"]): read(BASE / "scores" / im["id"] / f"{r['id']}.json")
        for im in p["images"]
        for r in p["rows"]
    }
    matrices = {
        (im["id"], r["id"]): dict(
            np.load(BASE / "scores" / im["id"] / f"{r['id']}.npz")
        )
        for im in p["images"]
        for r in p["rows"]
    }
    result = {}
    for guard in ("0", "15", "20"):
        result[guard] = {}
        for images in (
            [im["id"] for im in p["images"]],
            *[[im["id"]] for im in p["images"]],
        ):
            group = "mean3" if len(images) == 3 else images[0]
            out = {}
            for left, right in [
                ("T", "N"),
                ("PRE", "N"),
                ("POST", "N"),
                ("PRE", "POST"),
                ("POST40", "POST"),
                ("POST40", "N"),
                ("NRT", "N"),
                ("PRE", "NRT"),
            ]:
                out[left + "_minus_" + right] = {
                    metric: cluster(
                        [
                            np.mean(
                                [
                                    scored[(im, r["id"])]["cells"][left][guard][metric]
                                    - scored[(im, r["id"])]["cells"][right][guard][
                                        metric
                                    ]
                                    for im in images
                                ]
                            )
                            for r in p["rows"]
                        ],
                        [r["speaker"] for r in p["rows"]],
                    )
                    for metric in ("C", "D", "B")
                }
                if left != "T":
                    values = []
                    g = int(guard)
                    for r in p["rows"]:
                        local = []
                        for im in images:
                            curves = {
                                k: (m[g:-g] if g else m).mean(0)
                                for k, m in matrices[(im, r["id"])].items()
                            }
                            j = int(np.argmin(curves["N"]))
                            local.append(float(curves[left][j] - curves[right][j]))
                        values.append(float(np.mean(local)))
                    out[left + "_minus_" + right]["N_anchor_distance"] = cluster(
                        values, [r["speaker"] for r in p["rows"]]
                    )
            result[guard][group] = out
    write(BASE / "analysis.json", result)
    for key, value in result["20"]["mean3"].items():
        print(key, value["C"]["speaker_mean"], value["C"]["speaker_ci95"], flush=True)


def speech():
    p = read(BASE / "protocol.json")
    sourcefile = PARENT / "speech_windows/scores.json"
    masks = {r["id"]: r["cells"]["N_20"]["speech_rows"] for r in read(sourcefile)}
    write(
        BASE / "speech_protocol.json",
        {
            "source_sha256": sha(sourcefile),
            "definition": "existing N MFA speech-center windows intersect common current n and guard20; all candidates same N audio and same indices; secondary sensitivity, not pure-phone truth",
        },
    )
    records = {}
    for image in p["images"]:
        for row in p["rows"]:
            sid = row["id"]
            matrices = np.load(BASE / "scores" / image["id"] / f"{sid}.npz")
            n = len(matrices["N"])
            inds = np.array([i for i in masks[sid] if 20 <= i < n - 20])
            assert len(inds) >= 5
            cells = {}
            for k, m in matrices.items():
                if k == "T":
                    continue
                curve = m[inds].mean(0)
                cells[k] = {
                    "C": float(np.median(curve) - curve.min()),
                    "D": float(curve.min()),
                    "B": float(np.median(curve)),
                }
            records[image["id"] + "/" + sid] = {"rows": inds.tolist(), "cells": cells}
    result = {}
    for images in (
        [im["id"] for im in p["images"]],
        *[[im["id"]] for im in p["images"]],
    ):
        group = "mean3" if len(images) == 3 else images[0]
        out = {}
        for left, right in [
            ("PRE", "N"),
            ("POST", "N"),
            ("PRE", "POST"),
            ("POST40", "N"),
            ("NRT", "N"),
            ("PRE", "NRT"),
        ]:
            out[left + "_minus_" + right] = {
                metric: cluster(
                    [
                        np.mean(
                            [
                                records[im + "/" + r["id"]]["cells"][left][metric]
                                - records[im + "/" + r["id"]]["cells"][right][metric]
                                for im in images
                            ]
                        )
                        for r in p["rows"]
                    ],
                    [r["speaker"] for r in p["rows"]],
                )
                for metric in ("C", "D", "B")
            }
        result[group] = out
    write(BASE / "speech_scores.json", records)
    write(BASE / "speech_analysis.json", result)
    print("speech sensitivity complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["freeze", "prepare", "mel_worker", "run", "analyze", "speech"]
    )
    globals()[p.parse_args().stage]()
