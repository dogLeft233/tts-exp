#!/usr/bin/env python3
"""WORLD-controlled F0 and direct speech-energy interventions on frozen M audio.

Run all stages in the project venv with pyworld installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from mfa_linear_content_pause_mapping import (
    OUT, PRIOR, REPO, WAV2LIP, WAV2LIP_PY, FFMPEG, SYNCNET, SYNCNET_PY,
    canonicalize_tail, read, run, score_matrix, sha, write,
)

ARMS = ("W", "F", "E")  # WORLD identity, natural F0 on doubly voiced frames, natural energy


def exact_length(y: np.ndarray, size: int) -> np.ndarray:
    y = y.astype(np.float32)
    if len(y) < size:
        return np.pad(y, (0, size - len(y)))
    return y[:size].copy()


def audio() -> None:
    import pyworld

    rows = read(PRIOR / "manifest.json")["rows"]
    prior_pauses = read(OUT / "manifest.json")["rows"]
    by_id = {r["id"]: r for r in prior_pauses}
    result = []
    for row in rows:
        sid = row["id"]
        m, sr = sf.read(row["mfa"], dtype="float64")
        n, nr = sf.read(row["natural"], dtype="float64")
        if sr != 16000 or nr != 16000 or len(m) != len(n):
            raise ValueError(f"clock mismatch {sid}")
        f0m, tm = pyworld.dio(m, sr, f0_floor=70, f0_ceil=450, frame_period=20)
        f0m = pyworld.stonemask(m, f0m, tm, sr)
        f0n, tn = pyworld.dio(n, sr, f0_floor=70, f0_ceil=450, frame_period=20)
        f0n = pyworld.stonemask(n, f0n, tn, sr)
        f0n_at_m = np.interp(tm, tn, f0n, left=0, right=0)
        sp = pyworld.cheaptrick(m, f0m, tm, sr)
        ap = pyworld.d4c(m, f0m, tm, sr)
        voiced = (f0m > 0) & (f0n_at_m > 0)
        pauses = by_id[sid]["missing_pauses"]
        for a, b in pauses:
            voiced &= ~((tm >= a - .04) & (tm <= b + .04))
        f0_target = f0m.copy()
        f0_target[voiced] = np.clip(f0n_at_m[voiced], f0m[voiced] * .7, f0m[voiced] * 1.4)
        w = exact_length(pyworld.synthesize(f0m, sp, ap, sr, frame_period=20), len(m))
        f = exact_length(pyworld.synthesize(f0_target, sp, ap, sr, frame_period=20), len(m))
        # Energy gain acts directly on M, so F0 and pauses stay untouched.
        hop = 320
        gain = np.ones((len(m) + hop - 1) // hop)
        changed = 0
        for i in range(len(gain)):
            a, b = i * hop, min((i + 1) * hop, len(m))
            center = (a + b) / (2 * sr)
            if any(lo - .04 <= center <= hi + .04 for lo, hi in pauses):
                continue
            rn = float(np.sqrt(np.mean(n[a:b] ** 2)))
            rm = float(np.sqrt(np.mean(m[a:b] ** 2)))
            if rn > .003 and rm > .003:
                gain[i] = np.clip(rn / rm, .5, 2.)
                changed += 1
        # Smooth over 5 frames and only use speech frames as selected above.
        smooth = np.convolve(gain, np.ones(5) / 5, mode="same")
        smooth[0] = gain[0]
        smooth[-1] = gain[-1]
        e = (m * np.interp(np.arange(len(m)), np.arange(len(gain)) * hop + hop / 2,
                            smooth, left=smooth[0], right=smooth[-1])).astype(np.float32)
        info = {"id": sid, "speaker": row["speaker"], "face": row["face"],
                "natural": row["natural"], "mfa": row["mfa"],
                "target_frames": row["target_frames"], "samples": len(m),
                "voiced_f0_changed_frames": int(voiced.sum()), "energy_target_frames": changed,
                "median_f0_shift_semitones": float(np.median(12 * np.log2(f0_target[voiced] / f0m[voiced]))) if voiced.any() else 0.,
                "median_energy_gain_db": float(np.median(20 * np.log10(gain[gain != 1]))) if np.any(gain != 1) else 0.,
                "arms": {}}
        for arm, values in (("W", w), ("F", f), ("E", e)):
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {sid}/{arm}")
            path = OUT / "prosody" / "audio" / f"{sid}_{arm}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, values, 16000, subtype="FLOAT")
            info["arms"][arm] = {"audio": str(path), "sha256": sha(path),
                                 "max_abs_from_M": float(np.max(np.abs(values - m)))}
        result.append(info)
        write(OUT / "prosody" / "manifest.json", {"protocol": "W=pyworld DIO/StoneMask/cheaptrick/d4c/synthesize M identity control; F=same WORLD spectral envelope/aperiodicity with natural F0 on both-voiced 20ms frames clipped to 0.7-1.4x M; E=direct M waveform 20ms RMS gain to natural on both-active frames, clipped to 0.5-2x, smoothed 5 frames; flagged pauses excluded", "rows": result})
        print("audio", sid, "F0 frames", int(voiced.sum()), "energy", changed, flush=True)


def render() -> None:
    for row in read(OUT / "prosody" / "manifest.json")["rows"]:
        sid = row["id"]
        for arm in ARMS:
            root = OUT / "prosody" / "video" / sid / arm
            raw, normalized, video = root / "raw.mp4", root / "normalized.mkv", root / "video.mp4"
            if not raw.is_file():
                work = OUT / "prosody" / "wav2lip_work" / sid / arm
                (work / "temp").mkdir(parents=True, exist_ok=True)
                root.mkdir(parents=True, exist_ok=True)
                run([str(WAV2LIP_PY), str(WAV2LIP / "inference.py"), "--checkpoint_path",
                     str(WAV2LIP / "checkpoints/wav2lip_gan.pth"), "--face", row["face"],
                     "--audio", row["arms"][arm]["audio"], "--outfile", str(raw),
                     "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth"],
                    OUT / "prosody" / "logs" / f"{sid}_{arm}_wav2lip.log", work)
            if not normalized.is_file():
                receipt = canonicalize_tail(raw, normalized, target_frame_count=row["target_frames"], ffmpeg=FFMPEG)
                write(root / "normalized.json", receipt)
            if not video.is_file():
                run([str(FFMPEG), "-y", "-v", "error", "-i", str(normalized),
                     "-i", row["natural"], "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
                     "-r", "25", "-frames:v", str(row["target_frames"]),
                     "-c:a", "aac", "-b:a", "192k", str(video)],
                    OUT / "prosody" / "logs" / f"{sid}_{arm}_mux.log")
            print("render", sid, arm, flush=True)


def score() -> None:
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    rows = read(OUT / "prosody" / "manifest.json")["rows"]
    reference = {r["id"]: r["cells"]["VM_AN"] for r in read(PRIOR / "scores.json")["rows"]}
    engine = SyncNetEngine(batch_size=32, device="cuda")
    results = []
    try:
        for row in rows:
            sid = row["id"]
            audio_features, _ = engine.extract_audio(PRIOR / "clock" / sid / "N/audio.wav")
            baseline_crop = PRIOR / "fresh_pipeline" / sid / "M" / "pycrop" / f"split_{sid}_M" / "00000.avi"
            _, baseline_meta = engine.extract_visual(baseline_crop)
            cells = {"M": reference[sid]}
            for arm in ARMS:
                root = OUT / "prosody" / "pipeline" / sid / arm
                ref = f"pro_{sid}_{arm}"
                crop = root / "pycrop" / ref / "00000.avi"
                if not crop.is_file():
                    run([str(SYNCNET_PY), "run_pipeline.py", "--videofile",
                         str(OUT / "prosody" / "video" / sid / arm / "video.mp4"),
                         "--reference", ref, "--data_dir", str(root),
                         "--min_track", "100", "--overwrite"],
                        OUT / "prosody" / "logs" / f"{sid}_{arm}_pipeline.log", SYNCNET)
                visual, meta = engine.extract_visual(crop)
                if meta["frame_count"] != baseline_meta["frame_count"]:
                    raise ValueError(f"frame count mismatch {sid}/{arm}")
                cells[arm] = score_matrix(engine.distance_matrix(visual, audio_features))
            results.append({"id": sid, "speaker": row["speaker"], "cells": cells})
            write(OUT / "prosody" / "scores" / f"{sid}.json", results[-1])
            print("score", sid, {k: v["sync_c"] for k,v in cells.items()}, flush=True)
    finally:
        engine.close()
    write(OUT / "prosody" / "scores.json", {"protocol": "same official natural PCM and crop/SyncNet V2 as frozen M; WORLD F0 must be compared to WORLD W control", "rows": results})


def analyze() -> None:
    rows = read(OUT / "prosody" / "scores.json")["rows"]
    summary = {}
    for arm, reference in (("W", "M"), ("F", "W"), ("E", "M")):
        scores = np.array([r["cells"][arm]["sync_c"] for r in rows])
        delta = np.array([r["cells"][arm]["sync_c"] - r["cells"][reference]["sync_c"] for r in rows])
        summary[arm] = {"reference": reference, "mean_sync_c": float(scores.mean()),
                        "mean_delta": float(delta.mean()), "improved": int(np.sum(delta > 0)),
                        "mean_fixed_lag_distance": float(np.mean([r["cells"][arm]["fixed_lag_distance"] for r in rows]))}
    write(OUT / "prosody" / "analysis.json", {"rows": rows, "summaries": summary})
    print(json.dumps(summary, indent=2), flush=True)


def verify() -> None:
    import pyworld

    rows = read(OUT / "prosody" / "manifest.json")["rows"]
    results = []
    for row in rows:
        waves = {}
        for arm, path in (("N", row["natural"]), ("M", row["mfa"]),
                          *((arm, row["arms"][arm]["audio"]) for arm in ARMS)):
            waves[arm] = sf.read(path, dtype="float64")[0]
        if len({len(x) for x in waves.values()}) != 1:
            raise ValueError(f"length mismatch {row['id']}")
        tracks = {}
        energies = {}
        for arm, wave in waves.items():
            f0, times = pyworld.dio(wave, 16000, f0_floor=70, f0_ceil=450, frame_period=20)
            tracks[arm] = pyworld.stonemask(wave, f0, times, 16000)
            energies[arm] = np.asarray([float(np.sqrt(np.mean(wave[max(0, round(t * 16000) - 320):
                                                                      min(len(wave), round(t * 16000) + 320)] ** 2)))
                                          for t in times])
        count = min(len(x) for x in tracks.values())
        mask = np.ones(count, dtype=bool)
        for arm in ("N", "W", "F"):
            mask &= tracks[arm][:count] > 0
        f0_mae = {arm: float(np.median(np.abs(12 * np.log2(tracks[arm][:count][mask] / tracks["N"][:count][mask]))))
                  if mask.any() else None for arm in ("W", "F")}
        active = (energies["N"][:count] > .003) & (energies["M"][:count] > .003)
        energy_mae = {arm: float(np.median(np.abs(20 * np.log10(energies[arm][:count][active] /
                                                              energies["N"][:count][active]))))
                      if active.any() else None for arm in ("M", "E")}
        results.append({"id": row["id"], "common_voiced_frames": int(mask.sum()),
                        "common_active_frames": int(active.sum()),
                        "median_abs_f0_semitones": f0_mae,
                        "median_abs_energy_db": energy_mae})
    write(OUT / "prosody" / "verification.json", {"protocol": "20ms WORLD DIO+StoneMask output remeasurement; F0 compared only where N/W/F voiced; 40ms RMS energy compared on N/M active frames", "rows": results})
    for arm, baseline in (("F", "W"), ("E", "M")):
        metric = "median_abs_f0_semitones" if arm == "F" else "median_abs_energy_db"
        print(arm, "improved", sum(r[metric][arm] < r[metric][baseline] for r in results),
              "of", len(results), flush=True)


def acoustic() -> None:
    if str(WAV2LIP) not in sys.path:
        sys.path.insert(0, str(WAV2LIP))
    import audio as wav2lip_audio

    results = []
    for row in read(OUT / "prosody" / "manifest.json")["rows"]:
        mels = {}
        for arm, path in (("N", row["natural"]), ("M", row["mfa"]),
                          *((arm, row["arms"][arm]["audio"]) for arm in ARMS)):
            mels[arm] = wav2lip_audio.melspectrogram(wav2lip_audio.load_wav(path, 16000))
        if len({x.shape for x in mels.values()}) != 1:
            raise ValueError(f"mel grid mismatch {row['id']}")
        chunks = {arm: np.stack([x[:, min(int(frame * 80 / 25), x.shape[1] - 16):][:, :16]
                                 for frame in range(row["target_frames"])]) for arm, x in mels.items()}
        results.append({"id": row["id"], "mel_mae_to_N": {
            arm: float(np.mean(np.abs(chunks[arm] - chunks["N"]))) for arm in ("M", *ARMS)}})
    write(OUT / "prosody" / "acoustic.json", {"protocol": "exact Wav2Lip mel and 16-frame chunks, all video frames, MAE to N", "rows": results})
    print("acoustic", len(results), "samples", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audio", "render", "score", "verify", "acoustic", "analyze"))
    globals()[parser.parse_args().stage]()
