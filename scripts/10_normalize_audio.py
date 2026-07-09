#!/usr/bin/env python3
"""10_normalize_audio.py — Apply 3 acoustic normalization conditions to
selected TTS audio files for the interventional experiment.

Output: tmp/r4_normalized/{loudnorm,durnorm,bothnorm}/

Selected file set (79 files):
  - 40 emotion sweep files (8 emotions × 5 samples)
  - 13 fish_audio files
  - 13 indextts2_baseline files
  - 13 gsv_tts files

Normalization:
  loudnorm: LUFS normalize to -23 dB (EBU R128)
  durnorm:  Time-stretch to per-sample IndexTTS2 baseline duration
  bothnorm: Apply loudnorm first, then durnorm
"""
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln

REPO = Path(__file__).resolve().parent.parent
OUT_BASE = REPO / "tmp" / "r4_normalized"
TARGET_SR = 16000  # resample to 16kHz for Ditto pipeline
TARGET_LUFS = -23.0  # EBU R128 target

# === Selected files ===
P2_IDS = [1, 4, 7, 10, 13]
EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
ALL_IDS = list(range(1, 14))

EMO_SRC = REPO / "tmp" / "indextts2_emo_sweep"
FISH_SRC = REPO / "tmp" / "r2_tts" / "fish_audio"
BASELINE_SRC = REPO / "tmp" / "indextts2_output"
GSV_SRC = REPO / "data" / "gsv-tts"

NORM_TYPES = ["loudnorm", "durnorm", "bothnorm"]


def get_baseline_duration(sample_id, sr=TARGET_SR):
    """Load baseline IndexTTS2 file and return its duration in seconds."""
    f = BASELINE_SRC / f"{sample_id}.wav"
    if not f.exists():
        print(f"  WARNING: baseline {f} not found", file=sys.stderr)
        return None
    y, _ = librosa.load(str(f), sr=sr, mono=True)
    return len(y) / sr


def load_audio(path, sr=TARGET_SR):
    y, orig_sr = sf.read(str(path))
    if len(y.shape) > 1:
        y = y[:, 0]  # take first channel
    # Resample if needed
    if orig_sr != sr:
        if orig_sr != sr:
            y = librosa.resample(y.astype(np.float64), orig_sr=orig_sr, target_sr=sr)
    y = y.astype(np.float32)
    return y, sr


def apply_loudnorm(y, sr):
    """LUFS normalize to TARGET_LUFS."""
    meter = pyln.Meter(sr)
    current = meter.integrated_loudness(y)
    y_norm = pyln.normalize.loudness(y, current, TARGET_LUFS)
    # Clamp to prevent clipping
    peak = np.max(np.abs(y_norm))
    if peak > 0.99:
        y_norm = y_norm / peak * 0.99
    return y_norm.astype(np.float32)


def apply_durnorm(y, sr, target_dur):
    """Time-stretch to match target duration while preserving pitch."""
    current_dur = len(y) / sr
    if abs(current_dur - target_dur) < 0.01:
        return y  # already close enough
    rate = target_dur / current_dur
    try:
        y_stretched = librosa.phase_vocoder(y.astype(np.float64),
                                            rate=rate, n_fft=1024, hop_length=256)
        # Trim/pad to exact target length
        target_len = int(target_dur * sr)
        if len(y_stretched) > target_len:
            y_stretched = y_stretched[:target_len]
        elif len(y_stretched) < target_len:
            pad = target_len - len(y_stretched)
            y_stretched = np.pad(y_stretched, (0, pad), mode='constant')
        return y_stretched.astype(np.float32)
    except Exception as e:
        print(f"    durnorm error: {e}, skipping", file=sys.stderr)
        return y


def process_file(src_path, norm_type, sample_id, baseline_dur_map):
    """Apply normalization and return output path."""
    y, sr = load_audio(src_path, TARGET_SR)
    current_dur = len(y) / sr

    if norm_type == "loudnorm":
        y = apply_loudnorm(y, sr)
    elif norm_type == "durnorm":
        if sample_id in baseline_dur_map and baseline_dur_map[sample_id] is not None:
            y = apply_durnorm(y, sr, baseline_dur_map[sample_id])
        else:
            print(f"    no baseline dur for sample {sample_id}, skipping durnorm", file=sys.stderr)
    elif norm_type == "bothnorm":
        y = apply_loudnorm(y, sr)
        if sample_id in baseline_dur_map and baseline_dur_map[sample_id] is not None:
            y = apply_durnorm(y, sr, baseline_dur_map[sample_id])

    return y, sr


def main():
    print("=" * 60)
    print("Audio Normalization — Phase B Preparation")
    print("=" * 60)

    # Pre-compute baseline durations
    print("\nComputing baseline durations...")
    baseline_dur_map = {}
    for sid in ALL_IDS:
        dur = get_baseline_duration(sid, TARGET_SR)
        if dur is not None:
            baseline_dur_map[sid] = dur
            print(f"  sample {sid:2d}: {dur:.2f}s")
        else:
            baseline_dur_map[sid] = None

    # Create output directories
    for nt in NORM_TYPES:
        (OUT_BASE / nt).mkdir(parents=True, exist_ok=True)

    total_written = 0

    # === Process emotion sweep files ===
    print(f"\n{'='*60}")
    print("Processing emotion sweep (40 files)...")
    for sid in P2_IDS:
        for emo in EMOTIONS:
            src = EMO_SRC / f"{sid}_{emo}.wav"
            if not src.exists():
                print(f"  MISSING: {src}", file=sys.stderr)
                continue
            for nt in NORM_TYPES:
                y, sr = process_file(src, nt, sid, baseline_dur_map)
                out = OUT_BASE / nt / f"{sid}_{emo}.wav"
                sf.write(str(out), y, sr)
                total_written += 1

    # === Process fish_audio files ===
    print("Processing fish_audio (13 files)...")
    for sid in ALL_IDS:
        src = FISH_SRC / f"{sid}.wav"
        if not src.exists():
            print(f"  MISSING: {src}", file=sys.stderr)
            continue
        for nt in NORM_TYPES:
            y, sr = process_file(src, nt, sid, baseline_dur_map)
            out = OUT_BASE / nt / f"{sid}_fish.wav"
            sf.write(str(out), y, sr)
            total_written += 1

    # === Process indextts2_baseline files ===
    print("Processing indextts2_baseline (13 files)...")
    for sid in ALL_IDS:
        src = BASELINE_SRC / f"{sid}.wav"
        if not src.exists():
            print(f"  MISSING: {src}", file=sys.stderr)
            continue
        for nt in NORM_TYPES:
            y, sr = process_file(src, nt, sid, baseline_dur_map)
            out = OUT_BASE / nt / f"{sid}_indextts2.wav"
            sf.write(str(out), y, sr)
            total_written += 1

    # === Process GSV-TTS files ===
    print("Processing GSV-TTS (13 files)...")
    for sid in ALL_IDS:
        src = GSV_SRC / f"{sid}gsv.wav"
        if not src.exists():
            print(f"  MISSING: {src}", file=sys.stderr)
            continue
        for nt in NORM_TYPES:
            y, sr = process_file(src, nt, sid, baseline_dur_map)
            out = OUT_BASE / nt / f"{sid}_gsv.wav"
            sf.write(str(out), y, sr)
            total_written += 1

    print(f"\n{'='*60}")
    print(f"Done. {total_written} files written.")
    for nt in NORM_TYPES:
        count = len(list((OUT_BASE / nt).glob("*.wav")))
        print(f"  {nt}: {count} files")

    # Quick verification
    print(f"\n=== Verification: post-normalization RMS/duration ===")
    for nt in NORM_TYPES:
        rms_list = []
        dur_list = []
        for f in sorted((OUT_BASE / nt).glob("*.wav")):
            y, sr = librosa.load(str(f), sr=TARGET_SR, mono=True)
            rms = float(np.sqrt(np.mean(y ** 2)))
            dur = len(y) / sr
            rms_list.append(rms)
            dur_list.append(dur)
        if rms_list:
            print(f"  {nt:12s}: RMS={np.mean(rms_list):.4f}±{np.std(rms_list):.4f}  "
                  f"dur={np.mean(dur_list):.2f}±{np.std(dur_list):.2f}s")


if __name__ == "__main__":
    main()
