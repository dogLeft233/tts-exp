#!/usr/bin/env python3
"""08_audio_features.py — Extract acoustic features from all TTS audio files
and merge with existing SyncNet scores.

Output: data/r4_results/audio_features.csv
"""
import csv
import json
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "r4_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000  # resample all to 16kHz for consistent feature comparison


def extract_features(filepath, target_sr=TARGET_SR):
    """Extract acoustic features from a WAV file."""
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        dur = len(y) / sr
        rms = float(np.sqrt(np.mean(y ** 2)))
        peak = float(np.max(np.abs(y)))

        # LUFS (EBU R128 integrated)
        try:
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(y)
        except Exception:
            loudness = float("nan")

        # F0 via pyin
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=50, fmax=600, sr=sr,
                frame_length=2048, hop_length=512
            )
            f0_voiced = f0[voiced_flag]
            f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else float("nan")
            f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 1 else 0.0
        except Exception:
            f0_mean = float("nan")
            f0_std = float("nan")

        # Spectral centroid
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        spec_centroid = float(np.mean(centroid))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
        spec_rolloff = float(np.mean(rolloff))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))

        return {
            "duration": round(dur, 4),
            "rms": round(rms, 6),
            "peak": round(peak, 4),
            "lufs": round(loudness, 2) if not np.isnan(loudness) else None,
            "f0_mean": round(f0_mean, 1) if not np.isnan(f0_mean) else None,
            "f0_std": round(f0_std, 1),
            "spec_centroid": round(spec_centroid, 0),
            "spec_rolloff": round(spec_rolloff, 0),
            "zcr": round(zcr_mean, 6),
            "sample_rate": sr,
        }
    except Exception as e:
        print(f"  ERROR {filepath}: {e}", file=sys.stderr)
        return None


def main():
    # Collect all TTS audio files by source
    sources = {}

    # r2 providers
    r2_sources = {
        "fish_audio":   "tmp/r2_tts/fish_audio",
        "faster_qwen3": "tmp/r2_tts/faster_qwen3",
        "siliconflow":  "tmp/r2_tts/siliconflow",
        "dashscope_vc": "tmp/r2_tts/dashscope_vc",
    }
    # Also fish from original location
    r2_orig = {
        "fish_audio": "runs/r2_fish_audio_20260707T150723Z/02_tts",
    }

    for name, rel_path in {**r2_sources, **r2_orig}.items():
        path = REPO / rel_path
        for f in path.glob("*.wav"):
            sid = int(os.path.splitext(f.stem)[0])
            sources[str(f)] = {"provider": name, "sample": sid, "source": "r2"}

    # r3: IndexTTS2 baseline
    for f in (REPO / "tmp" / "indextts2_output").glob("*.wav"):
        sid = int(os.path.splitext(f.stem)[0])
        sources[str(f)] = {"provider": "indextts2_baseline", "sample": sid, "source": "r3"}

    # r3: Emotion sweep (P2)
    for f in (REPO / "tmp" / "indextts2_emo_sweep").glob("*.wav"):
        # Format: {sample}_{emotion}.wav
        parts = f.stem.split("_", 1)
        if len(parts) == 2:
            sid, emotion = int(parts[0]), parts[1]
            sources[str(f)] = {"provider": f"indextts2_emo_{emotion}", "sample": sid,
                               "source": "r3", "emotion": emotion}

    # r3: Intensity sweep (P3)
    for f in (REPO / "tmp" / "indextts2_emo_intensity").glob("*.wav"):
        parts = f.stem.split("_", 2)
        if len(parts) == 3:
            sid, emotion, alpha = int(parts[0]), parts[1], parts[2]
            sources[str(f)] = {"provider": f"indextts2_int_{emotion}_a{alpha}",
                               "sample": sid, "source": "r3",
                               "emotion": emotion, "alpha": alpha}

    # r3: GSV-TTS
    for f in (REPO / "data" / "gsv-tts").glob("*.wav"):
        sid = int(f.stem.replace("gsv", ""))
        sources[str(f)] = {"provider": "gsv_tts", "sample": sid, "source": "r3"}

    print(f"Total TTS files: {len(sources)}")

    # Extract features
    rows = []
    for i, (filepath, meta) in enumerate(sorted(sources.items())):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(sources)}...", file=sys.stderr)
        feats = extract_features(filepath)
        if feats:
            row = {**meta, "filepath": os.path.relpath(filepath, REPO), **feats}
            rows.append(row)

    print(f"Successfully extracted: {len(rows)} files", file=sys.stderr)

    # Load existing SyncNet scores and merge
    score_map = {}

    # r2 providers: load from syncnet.json on server
    r2_run_map = {
        "fish_audio":   "r2_fish_audio_20260707T150723Z",
        "dashscope_vc": "r2_dashscope_vc_20260707T143930Z",
        "faster_qwen3": "r2_faster_qwen3_20260707T145233Z",
        "siliconflow":  "r2_siliconflow_20260707T142139Z",
    }
    r2_base = REPO / "runs"
    for provider_name, run_dir in r2_run_map.items():
        tts_dir = r2_base / run_dir / "04_eval" / "tts_raw"
        nat_dir = r2_base / run_dir / "04_eval" / "natural_raw"
        for jf in tts_dir.glob("*/syncnet.json"):
            try:
                d = json.loads(jf.read_text())
                key = (provider_name, int(d["sample_id"]))
                score_map[(key, "tts")] = {"sync_c": float(d["sync_c"]), "sync_d": float(d["sync_d"])}
            except Exception:
                pass
        for jf in nat_dir.glob("*/syncnet.json"):
            try:
                d = json.loads(jf.read_text())
                key = (provider_name, int(d["sample_id"]))
                score_map[(key, "nat")] = {"sync_c": float(d["sync_c"]), "sync_d": float(d["sync_d"])}
            except Exception:
                pass

    # r3: Load from r3_results CSVs
    def _load_csv(path):
        rows_out = []
        with open(path) as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if r.get("nat_sync_c"):
                    rows_out.append(r)
        return rows_out

    # P2 emotion sweep
    p2_path = REPO / "data" / "r3_results" / "p2_emotion_sweep.csv"
    if p2_path.exists():
        for r in _load_csv(p2_path):
            sid = int(r["sample"])
            emo = r["emotion"]
            key_base = (f"indextts2_emo_{emo}", sid)
            score_map[(key_base, "tts")] = {"sync_c": float(r["tts_sync_c"]), "sync_d": float(r["tts_sync_d"])}
            score_map[(key_base, "nat")] = {"sync_c": float(r["nat_sync_c"]), "sync_d": float(r["nat_sync_d"])}

    # P3 intensity sweep
    p3_path = REPO / "data" / "r3_results" / "p3_intensity_sweep.csv"
    if p3_path.exists():
        for r in _load_csv(p3_path):
            sid = int(r["sample"])
            emo = r["emotion"]
            alpha = r["alpha"]
            key_base = (f"indextts2_int_{emo}_a{alpha}", sid)
            score_map[(key_base, "tts")] = {"sync_c": float(r["tts_sync_c"]), "sync_d": float(r["tts_sync_d"])}
            score_map[(key_base, "nat")] = {"sync_c": float(r["nat_sync_c"]), "sync_d": float(r["nat_sync_d"])}

    # Baseline + GSV
    bg_path = REPO / "data" / "r3_results" / "baseline_gsv.csv"
    if bg_path.exists():
        for r in _load_csv(bg_path):
            sid = int(r["sample"])
            cond = r["condition"]
            if cond == "r3_indextts2_baseline":
                key_base = ("indextts2_baseline", sid)
            else:
                key_base = ("gsv_tts", sid)
            score_map[(key_base, "nat")] = {"sync_c": float(r["nat_sync_c"]), "sync_d": float(r["nat_sync_d"])}
            score_map[(key_base, "tts")] = {"sync_c": float(r["tts_sync_c"]), "sync_d": float(r["tts_sync_d"])}

    print(f"Score map entries: {len(score_map)}", file=sys.stderr)

    # Merge
    for row in rows:
        key_base = (row["provider"], row["sample"])
        tts_key = (key_base, "tts")
        nat_key = (key_base, "nat")
        row["sync_c"] = score_map[tts_key]["sync_c"] if tts_key in score_map else None
        row["sync_d"] = score_map[tts_key]["sync_d"] if tts_key in score_map else None
        row["nat_sync_c"] = score_map[nat_key]["sync_c"] if nat_key in score_map else None
        row["nat_sync_d"] = score_map[nat_key]["sync_d"] if nat_key in score_map else None

    # Write CSV
    fieldnames = [
        "provider", "sample", "source",
        "emotion", "alpha",
        "duration", "rms", "peak", "lufs",
        "f0_mean", "f0_std",
        "spec_centroid", "spec_rolloff", "zcr",
        "sync_c", "sync_d", "nat_sync_c", "nat_sync_d",
        "filepath"
    ]
    out_path = OUT_DIR / "audio_features.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path} ({len(rows)} rows)", file=sys.stderr)

    # Quick summary
    scored_rows = [r for r in rows if r["sync_c"] is not None]
    print(f"Rows with SyncC: {len(scored_rows)}", file=sys.stderr)

    return rows


if __name__ == "__main__":
    main()
