#!/usr/bin/env python3
"""Three-arm N/R/M audio-video crossover on the 10-speaker MFA-linear cohort.

N: natural speech; R: direct WavLM-L6 + prematched HiFi-GAN resynthesis of N;
M: existing TTS MFA-linear WavLM + the same vocoder. R and M have the exact
natural sample count. All videos use the same face and Wav2Lip checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.pilot_generate_mfa_linear import exact_natural_length
from scripts.wavlm_knn_vc_adapter import KNN_VC_REVISION, WavLMKNNVCAdapter
from scripts.experiments.mfa_linear_video_retiming.generation import canonicalize_tail
from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

COHORT = REPO / "runs/aishell1_qwen_mfa_linear_n100_20260816"
PRIOR = REPO / "runs/mfa_linear_nas_short_expanded_20260924"
OUT = REPO / "runs/mfa_linear_vocoder_wav2lip_split_20260925"
WAV2LIP = REPO / "third_party/Wav2Lip"
SYNCNET = REPO / "third_party/syncnet_python"
WAV2LIP_PY = Path("/home/wjj/.venvs/wav2lip/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
SYNCNET_PY = Path("/home/wjj/.venvs/syncnet/bin/python")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(command: list[str], log: Path, *, cwd: Path = REPO) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        result = subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[:3]}; log={log}")


def records() -> list[dict]:
    ids = read_json(PRIOR / "summary.json")["protocol"]["sample_ids"]
    manifest = read_json(COHORT / "05_wav2lip_syncnet_n100/manifest.json")
    by_id = {str(r["sample_id"]): r for r in manifest["records"] if r["arm"] == "natural_raw"}
    rows = []
    for id in ids:
        r = by_id[id]
        natural = Path(r["audio"])
        mfa = COHORT / "04_mfa_linear" / f"{id}.wav"
        face = Path(r["face"])
        if sha(natural) != r["audio_sha256"] or sha(face) != r["face_sha256"]:
            raise ValueError(f"source hash mismatch for {id}")
        n, sr = sf.read(natural)
        m, mr = sf.read(mfa)
        if sr != 16000 or mr != 16000 or len(n) != len(m):
            raise ValueError(f"source sample-clock mismatch for {id}")
        rows.append({"id": id, "speaker": r["speaker_id"], "paired_key": r["paired_key"],
                     "natural": str(natural), "mfa": str(mfa), "face": str(face),
                     "natural_sha256": sha(natural), "mfa_sha256": sha(mfa),
                     "face_sha256": sha(face), "samples": len(n),
                     "target_frames": math.ceil(len(n) / 640)})
    return rows


def generate() -> None:
    rows = records()
    adapter = WavLMKNNVCAdapter.load_pretrained(
        device="cuda", source=REPO / "third_party/knn-vc", revision=KNN_VC_REVISION
    )
    results = []
    for row in rows:
        id = row["id"]
        source = Path(row["natural"])
        output = OUT / "audio" / f"{id}_R.wav"
        if not output.is_file():
            x, sr = sf.read(source, dtype="float32")
            if sr != 16000 or x.ndim != 1:
                raise ValueError(f"bad natural waveform {id}")
            features = adapter.extract(torch.from_numpy(x).unsqueeze(0))
            raw = adapter.vocode(features).numpy()
            y, adjustment = exact_natural_length(raw, len(x))
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, y, 16000, subtype="FLOAT")
        else:
            adjustment = {"action": "cached"}
        y, sr = sf.read(output, dtype="float32")
        if sr != 16000 or len(y) != row["samples"] or not np.isfinite(y).all():
            raise ValueError(f"invalid resynthesis output {id}")
        results.append({**row, "reconstruction": str(output), "reconstruction_sha256": sha(output),
                        "length_adjustment": adjustment,
                        "rms_N": float(np.sqrt(np.mean(sf.read(source, dtype="float32")[0] ** 2))),
                        "rms_R": float(np.sqrt(np.mean(y**2)))})
        print("R", id, len(y), flush=True)
    write_json(OUT / "manifest.json", {"protocol": "N/R/M audio, same 10 speakers, same frozen WavLM-L6/prematched HiFi-GAN as M", "model": adapter.metadata(), "rows": results})


def render() -> None:
    rows = read_json(OUT / "manifest.json")["rows"]
    for row in rows:
        id = row["id"]
        raw = OUT / "video" / id / "R_raw.mp4"
        work = OUT / "wav2lip_work" / id
        if not raw.is_file():
            (work / "temp").mkdir(parents=True, exist_ok=True)
            raw.parent.mkdir(parents=True, exist_ok=True)
            run([str(WAV2LIP_PY), str(WAV2LIP / "inference.py"),
                 "--checkpoint_path", str(WAV2LIP / "checkpoints/wav2lip_gan.pth"),
                 "--face", row["face"], "--audio", row["reconstruction"],
                 "--outfile", str(raw), "--face_det_batch_size", "4",
                 "--wav2lip_batch_size", "4", "--nosmooth"],
                OUT / "logs" / f"{id}_wav2lip.log", cwd=work)
        normalized = OUT / "video" / id / "R_normalized.mkv"
        receipt = OUT / "video" / id / "R_normalized.json"
        if not normalized.is_file():
            detail = canonicalize_tail(raw, normalized, target_frame_count=row["target_frames"], ffmpeg=FFMPEG)
            write_json(receipt, detail)
        video = OUT / "video" / id / "R.mp4"
        if not video.is_file():
            run([str(FFMPEG), "-y", "-v", "error", "-i", str(normalized),
                 "-i", row["natural"], "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                 "-pix_fmt", "yuv420p", "-r", "25", "-frames:v", str(row["target_frames"]),
                 "-c:a", "aac", "-b:a", "192k", str(video)],
                OUT / "logs" / f"{id}_mux.log")
        print("video", id, video, flush=True)


def r_crop(row: dict) -> tuple[Path, Path]:
    id = row["id"]
    root = OUT / "pipeline" / id
    reference = f"split_R_{id}"
    crop = root / "pycrop" / reference / "00000.avi"
    pcm = root / "pyavi" / reference / "audio.wav"
    if not crop.is_file():
        run([str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(OUT / "video" / id / "R.mp4"),
             "--reference", reference, "--data_dir", str(root), "--min_track", "100", "--overwrite"],
            OUT / "logs" / f"{id}_pipeline.log", cwd=SYNCNET)
    if not crop.is_file() or not pcm.is_file():
        raise FileNotFoundError(f"R crop missing for {id}")
    return crop, pcm


def fresh_crop(row: dict, arm: str) -> tuple[Path, Path]:
    id = row["id"]
    if arm == "R":
        return r_crop(row)
    root = OUT / "fresh_pipeline" / id / arm
    reference = f"split_{id}_{arm}"
    crop = root / "pycrop" / reference / "00000.avi"
    pcm = root / "pyavi" / reference / "audio.wav"
    if not crop.is_file():
        run([str(SYNCNET_PY), "run_pipeline.py", "--videofile",
             str(PRIOR / id / "input" / f"{arm}.mp4"), "--reference", reference,
             "--data_dir", str(root), "--min_track", "100", "--overwrite"],
            OUT / "logs" / f"{id}_{arm}_fresh_pipeline.log", cwd=SYNCNET)
    if not crop.is_file() or not pcm.is_file():
        raise FileNotFoundError(f"fresh crop missing {id}/{arm}")
    return crop, pcm


def prepare_clock() -> None:
    rows = read_json(OUT / "manifest.json")["rows"]
    for row in rows:
        id = row["id"]
        _, fresh_n = fresh_crop(row, "N")
        fresh_crop(row, "M")
        r_crop(row)
        for arm, source in (("N", row["natural"]), ("R", row["reconstruction"]), ("M", row["mfa"])):
            root = OUT / "clock" / id / arm
            pcm = root / "audio.wav"
            if not pcm.is_file():
                root.mkdir(parents=True, exist_ok=True)
                mux = root / "mux.mp4"
                avi = root / "video.avi"
                run([str(FFMPEG), "-y", "-v", "error", "-i", str(PRIOR / id / "normalized/N.mkv"),
                     "-i", source, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                     "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "25",
                     "-frames:v", str(row["target_frames"]), "-c:a", "aac", "-b:a", "192k", str(mux)],
                    OUT / "logs" / f"{id}_{arm}_clock_mux.log")
                run([str(FFMPEG), "-y", "-v", "error", "-i", str(mux), "-qscale:v", "2",
                     "-async", "1", "-r", "25", str(avi)],
                    OUT / "logs" / f"{id}_{arm}_clock_avi.log")
                run([str(FFMPEG), "-y", "-v", "error", "-i", str(avi), "-ac", "1", "-vn",
                     "-acodec", "pcm_s16le", "-ar", "16000", str(pcm)],
                    OUT / "logs" / f"{id}_{arm}_clock_pcm.log")
            if arm == "N" and sha(pcm) != sha(fresh_n):
                raise ValueError(f"fresh N pipeline PCM differs from exact protocol remux: {id}")
            print("clock", id, arm, sf.info(pcm).frames, flush=True)
        counts = [sf.info(OUT / "clock" / id / arm / "audio.wav").frames for arm in ("N", "R", "M")]
        if len(set(counts)) != 1:
            raise ValueError(f"evaluation PCM sample counts differ {id}: {counts}")


def score_matrix(d: np.ndarray) -> dict:
    mean = d.mean(axis=0)
    index = int(np.argmin(mean))
    if index in (0, 30):
        boundary = True
    else:
        boundary = False
    return {"sync_c": round(float(np.median(mean) - mean[index]), 3),
            "sync_d": round(float(mean[index]), 3),
            "offset": 15 - index, "offset_at_boundary": boundary,
            "fixed_lag_distance": float(mean[15]), "window_count": int(d.shape[0])}


def score() -> None:
    rows = read_json(OUT / "manifest.json")["rows"]
    engine = SyncNetEngine(batch_size=32, device="cuda")
    results = []
    try:
        for row in rows:
            id = row["id"]
            audio_features = {}
            for arm in ("N", "R", "M"):
                wav = OUT / "clock" / id / arm / "audio.wav"
                if not wav.is_file():
                    raise FileNotFoundError(f"run prepare_clock first: {wav}")
                audio_features[arm], _ = engine.extract_audio(wav)
            video_features = {}
            for arm in ("N", "R", "M"):
                path, _ = fresh_crop(row, arm)
                video_features[arm], meta = engine.extract_visual(path)
                print("features", id, arm, meta["frame_count"], flush=True)
            cells = {}
            for video_arm, visual in video_features.items():
                for audio_arm, audio in audio_features.items():
                    matrix = engine.distance_matrix(visual, audio)
                    cells[f"V{video_arm}_A{audio_arm}"] = score_matrix(matrix)
            results.append({"id": id, "speaker": row["speaker"], "cells": cells,
                            "reconstruction_sha256": row["reconstruction_sha256"],
                            "reconstruction_video_sha256": sha(OUT / "video" / id / "R.mp4")})
            write_json(OUT / "scores" / f"{id}.json", results[-1])
            print("scores", id, {k: v["sync_c"] for k,v in cells.items()}, flush=True)
    finally:
        engine.close()
    write_json(OUT / "scores.json", {"protocol": "N/R/M sources muxed through identical libx264/AAC 25fps path, decoded by the current official SyncNet FFmpeg video.avi/PCM chain; fresh N/M/R face crops; official SyncNet V2 embeddings and confidence formula", "rows": results})


def mel() -> None:
    if str(WAV2LIP) not in sys.path:
        sys.path.insert(0, str(WAV2LIP))
    import audio as wav2lip_audio

    rows = read_json(OUT / "manifest.json")["rows"]
    natural_tokens = read_json(COHORT / "03_tokens_paired/tokens.json")["records"]
    silence_audit = read_json(REPO / "runs/mfa_linear_phoneme_video_audit_20260925/analysis.json")
    bad_pauses = {}
    for pause in silence_audit["silences_ge_100ms"]:
        if not pause["matched_to_tts_silence"]:
            bad_pauses.setdefault(pause["sample_id"], []).append((pause["start_s"], pause["end_s"]))
    results = []
    for row in rows:
        id = row["id"]
        mels = {}
        for arm, source in (("N", row["natural"]), ("R", row["reconstruction"]), ("M", row["mfa"])):
            y = wav2lip_audio.load_wav(source, 16000)
            mels[arm] = wav2lip_audio.melspectrogram(y)
        if len({m.shape for m in mels.values()}) != 1:
            raise ValueError(f"Wav2Lip mel grid mismatch {id}")
        count = row["target_frames"]
        chunks = {}
        for arm, m in mels.items():
            parts = []
            for frame in range(count):
                start = int(frame * 80.0 / 25.0)
                if start + 16 > m.shape[1]:
                    parts.append(m[:, -16:])
                else:
                    parts.append(m[:, start:start+16])
            chunks[arm] = np.stack(parts)
        time = (np.arange(count) + .5) / 25.0
        speech = np.zeros(count, dtype=bool)
        for token in natural_tokens[id]["natural"]["tokens"]:
            if token["is_silence"]:
                continue
            speech |= (time >= float(token["start_s"])) & (time < float(token["end_s"]))
        clean = speech.copy()
        for a, b in bad_pauses.get(id, []):
            clean &= ~((time >= a - .2) & (time <= b + .2))
        if clean.sum() < 30:
            raise ValueError(f"too little clean speech for mel comparison {id}")
        diff_r = np.mean(np.abs(chunks["R"] - chunks["N"]), axis=(1, 2))
        diff_m = np.mean(np.abs(chunks["M"] - chunks["N"]), axis=(1, 2))
        result = {"id": id, "mel_shape": list(mels["N"].shape),
                  "clean_speech_windows": int(clean.sum()),
                  "mel_mae_R_N_clean": float(diff_r[clean].mean()),
                  "mel_mae_M_N_clean": float(diff_m[clean].mean()),
                  "mel_mae_R_N_all": float(diff_r.mean()),
                  "mel_mae_M_N_all": float(diff_m.mean()),
                  "fraction_clean_windows_M_farther_than_R": float(np.mean(diff_m[clean] > diff_r[clean]))}
        results.append(result)
        print("mel", id, round(result["mel_mae_R_N_clean"], 3), round(result["mel_mae_M_N_clean"], 3), flush=True)
    write_json(OUT / "mel.json", {"protocol": "Wav2Lip audio.load_wav and melspectrogram; inference.py 80 Hz/16-frame chunks; clean natural-speech windows exclude unmatched pauses +/-0.2s", "rows": results})


def mouth() -> None:
    import cv2
    import mediapipe as mp

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(REPO / "checkpoints/mediapipe/face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1,
    )
    baseline = REPO / "runs/mfa_linear_phoneme_video_audit_20260925/mouth"
    results = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as model:
        for row in read_json(OUT / "manifest.json")["rows"]:
            id = row["id"]
            dest = OUT / "mouth" / f"{id}_R.npz"
            if not dest.is_file():
                capture = cv2.VideoCapture(str(OUT / "video" / id / "R.mp4"))
                values = []
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                        if not result.face_landmarks:
                            values.append(float("nan"))
                            continue
                        face = result.face_landmarks[0]
                        eye = np.hypot(face[33].x - face[263].x, face[33].y - face[263].y)
                        values.append(float(np.hypot(face[13].x - face[14].x, face[13].y - face[14].y) / eye))
                finally:
                    capture.release()
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, opening=np.asarray(values))
            n = np.load(baseline / f"{id}_N.npz")["opening"]
            m = np.load(baseline / f"{id}_M.npz")["opening"]
            r = np.load(dest)["opening"]
            if min(len(n),len(m),len(r)) != row["target_frames"] or max(len(n),len(m),len(r)) != row["target_frames"]:
                raise ValueError(f"video length mismatch {id}")
            if np.isnan(n).any() or np.isnan(m).any() or np.isnan(r).any():
                raise ValueError(f"landmark missing {id}")
            scale = float(np.percentile(n, 90) - np.percentile(n, 10))
            results.append({"id": id, "corr_R_N": float(np.corrcoef(r,n)[0,1]),
                            "corr_M_N": float(np.corrcoef(m,n)[0,1]),
                            "gap_R_N_norm": float(np.mean(np.abs(r-n))/scale),
                            "gap_M_N_norm": float(np.mean(np.abs(m-n))/scale)})
            print("mouth", id, round(results[-1]["corr_R_N"],3), round(results[-1]["corr_M_N"],3), flush=True)
    write_json(OUT / "mouth.json", {"protocol": "same 25fps frames; MediaPipe lip points 13/14 normalized by eye points 33/263, prior N/M curves reused", "rows": results})


def align() -> None:
    rows = read_json(OUT / "manifest.json")["rows"]
    source_input = COHORT / "02_mfa_mandarin341_ready/input"
    for row in rows:
        id, speaker = row["id"], row["speaker"]
        input_dir = OUT / "mfa_input" / speaker
        input_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{speaker}_{id}_R"
        wav = input_dir / f"{stem}.wav"
        lab = input_dir / f"{stem}.lab"
        source_lab = source_input / speaker / "natural" / f"{speaker}_{id}_natural.lab"
        if not wav.exists():
            wav.symlink_to(Path(row["reconstruction"]).resolve())
        if not lab.exists():
            shutil.copyfile(source_lab, lab)
        if sha(wav) != row["reconstruction_sha256"] or sha(lab) != sha(source_lab):
            raise ValueError(f"MFA input differs from source for {id}")
    output = OUT / "mfa_alignment"
    expected = [output / row["speaker"] / f"{row['speaker']}_{row['id']}_R.json" for row in rows]
    if not all(path.is_file() for path in expected):
        mfa = shutil.which("mfa")
        if mfa is None:
            raise FileNotFoundError("Montreal Forced Aligner executable 'mfa' not found")
        run([mfa, "align", str(OUT / "mfa_input"), "mandarin_china_mfa", "mandarin_mfa",
             str(output), "--output_format", "json", "--num_jobs", "3",
             "--temporary_directory", str(OUT / "mfa_tmp")],
            OUT / "logs" / "mfa_align.log")
    for path in expected:
        if not path.is_file() or not read_json(path)["tiers"]["phones"]["entries"]:
            raise ValueError(f"MFA output missing or empty: {path}")
    print("MFA aligned", len(expected), "reconstructed clips", flush=True)


def analyze() -> None:
    rows = read_json(OUT / "manifest.json")["rows"]
    scores = {r["id"]: r["cells"] for r in read_json(OUT / "scores.json")["rows"]}
    mels = {r["id"]: r for r in read_json(OUT / "mel.json")["rows"]}
    mouths = {r["id"]: r for r in read_json(OUT / "mouth.json")["rows"]}
    tokens = read_json(COHORT / "03_tokens_paired/tokens.json")["records"]
    prior = read_json(REPO / "runs/mfa_linear_phoneme_video_audit_20260925/analysis.json")
    unmatched_pause_ids = {p["sample_id"] for p in prior["silences_ge_100ms"]
                           if not p["matched_to_tts_silence"]}
    tone = re.compile("[\u02e5-\u02e9]")
    data = []
    for row in rows:
        id = row["id"]
        natural = [p for p in tokens[id]["natural"]["tokens"] if not p["is_silence"]]
        path = OUT / "mfa_alignment" / row["speaker"] / f"{row['speaker']}_{id}_R.json"
        reconstructed = [{"token": tone.sub("", label), "start_s": float(a), "end_s": float(b)}
                         for a, b, label in read_json(path)["tiers"]["phones"]["entries"]
                         if label not in ("", "sil", "sp", "spn")]
        matcher = SequenceMatcher(a=[p["token"] for p in natural],
                                  b=[p["token"] for p in reconstructed], autojunk=False)
        errors = [abs((natural[block.a + k]["start_s"] + natural[block.a + k]["end_s"]
                       - reconstructed[block.b + k]["start_s"] - reconstructed[block.b + k]["end_s"]) * 500)
                  for block in matcher.get_matching_blocks() for k in range(block.size)]
        if not errors:
            raise ValueError(f"no matching reconstructed phonemes for {id}")
        cells = scores[id]
        data.append({"id": id, "speaker": row["speaker"],
                     "clean_no_unmatched_pause": id not in unmatched_pause_ids,
                     "sync_c": {k: v["sync_c"] for k, v in cells.items()},
                     "sync_d": {k: v["sync_d"] for k, v in cells.items()},
                     "offset": {k: v["offset"] for k, v in cells.items()},
                     "fixed_lag_distance": {k: v["fixed_lag_distance"] for k, v in cells.items()},
                     "window_count": {k: v["window_count"] for k, v in cells.items()},
                     "mel_mae_R_N_clean": mels[id]["mel_mae_R_N_clean"],
                     "mel_mae_M_N_clean": mels[id]["mel_mae_M_N_clean"],
                     "mouth_corr_R_N": mouths[id]["corr_R_N"],
                     "mouth_corr_M_N": mouths[id]["corr_M_N"],
                     "mouth_gap_R_N_norm": mouths[id]["gap_R_N_norm"],
                     "mouth_gap_M_N_norm": mouths[id]["gap_M_N_norm"],
                     "r_mfa_matched_phones": len(errors),
                     "r_mfa_natural_phones": len(natural),
                     "r_mfa_median_abs_center_error_ms": float(np.median(errors)),
                     "r_mfa_p90_abs_center_error_ms": float(np.percentile(errors, 90))})
    keys = [f"V{v}_A{a}" for v in ("N", "R", "M") for a in ("N", "R", "M")]
    mean = lambda values: float(np.mean(values))
    summary = {"n": len(data), "clean_no_unmatched_pause_n": sum(r["clean_no_unmatched_pause"] for r in data),
               "mean_sync_c": {k: mean([r["sync_c"][k] for r in data]) for k in keys},
               "mean_sync_c_clean": {k: mean([r["sync_c"][k] for r in data if r["clean_no_unmatched_pause"]]) for k in keys},
               "mean_offset": {k: mean([r["offset"][k] for r in data]) for k in keys},
               "mean_fixed_lag_distance": {k: mean([r["fixed_lag_distance"][k] for r in data]) for k in keys},
               "mean_mel_mae_R_N_clean": mean([r["mel_mae_R_N_clean"] for r in data]),
               "mean_mel_mae_M_N_clean": mean([r["mel_mae_M_N_clean"] for r in data]),
               "mean_mouth_corr_R_N": mean([r["mouth_corr_R_N"] for r in data]),
               "mean_mouth_corr_M_N": mean([r["mouth_corr_M_N"] for r in data]),
               "mean_mouth_gap_R_N_norm": mean([r["mouth_gap_R_N_norm"] for r in data]),
               "mean_mouth_gap_M_N_norm": mean([r["mouth_gap_M_N_norm"] for r in data]),
               "r_mfa_matched_phones": sum(r["r_mfa_matched_phones"] for r in data),
               "r_mfa_natural_phones": sum(r["r_mfa_natural_phones"] for r in data),
               "r_mfa_median_of_sample_medians_ms": float(np.median([r["r_mfa_median_abs_center_error_ms"] for r in data]))}
    write_json(OUT / "analysis.json", {"protocol": "paired within sample; no significance claim from exploratory n=10; same exact decoded audio clock for all 3 arms", "rows": data, "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generate", "render", "prepare_clock", "score", "mel", "mouth", "align", "analyze"))
    stage = parser.parse_args().stage
    {"generate": generate, "render": render, "prepare_clock": prepare_clock,
     "score": score, "mel": mel, "mouth": mouth, "align": align, "analyze": analyze}[stage]()


if __name__ == "__main__":
    main()
