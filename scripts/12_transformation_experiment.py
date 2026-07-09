#!/usr/bin/env python3
"""12_transformation_experiment.py — Controlled audio transformations on
F5-TTS baseline, measuring SyncD/SyncC change via Ditto+SyncNet.

Baseline: F5-TTS, samples [3, 6, 8, 10, 12]
Transformations:
  1. Gain: -10, -5, 0, +5, +10 dB
  2. Low-pass: 3k, 6k, 8k Hz
  3. EQ (1-4k): +6, -6 dB

Each transformation variant creates a run directory with 5 audio files,
then runs Ditto + SyncNet.

Output: data/r4_results/transformation_experiment.csv
"""
import sys, os, json, csv, time, subprocess, argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal

REPO = Path(__file__).resolve().parent.parent
RES_DIR = REPO / "data" / "r4_results"
RES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [3, 6, 8, 10, 12]
BASELINE_RUN = "r5_f5_tts"
SRC_AUDIO_DIR = REPO / "runs" / BASELINE_RUN / "02_tts"
SRC_TRANSCRIPT_DIR = REPO / "runs" / BASELINE_RUN / "01_transcript"
SRC_IMAGE = REPO / "data" / "data" / "image"
DITTO_BIN = REPO.parent / "repos" / "ditto-talkinghead" if (REPO.parent / "repos" / "ditto-talkinghead").exists() else None


def apply_gain(y, db):
    """Apply gain in dB. 0 = unity, +6 = double amplitude."""
    factor = 10 ** (db / 20.0)
    return y * factor


def apply_lowpass(y, sr, cutoff_hz):
    """Apply low-pass FIR filter."""
    nyq = sr / 2
    if cutoff_hz >= nyq:
        return y
    b = signal.firwin(65, cutoff_hz / nyq, window="hamming")
    return signal.lfilter(b, 1.0, y)


def apply_eq_band(y, sr, low_hz, high_hz, gain_db):
    """Boost/cut a frequency band using IIR bandpass."""
    nyq = sr / 2
    low_n = low_hz / nyq
    high_n = high_hz / nyq
    if high_n >= 1.0:
        high_n = 0.99
    if low_n <= 0:
        low_n = 0.01
    # 4th-order butterworth bandpass
    b, a = signal.butter(4, [low_n, high_n], btype="band")
    filtered = signal.lfilter(b, a, y)
    factor = 10 ** (gain_db / 20.0) - 1.0
    return y + factor * filtered


def normalize_lufs(y, sr, target_lufs=-23.0):
    """Normalize audio to target LUFS using pyloudnorm."""
    try:
        meter = pyln.Meter(sr)
        current = meter.integrated_loudness(y)
        if np.isnan(current) or current == -np.inf:
            return y
        gain = target_lufs - current
        return apply_gain(y, gain)
    except Exception:
        return y


def get_ditto_bin_path():
    """Find ditto-talkinghead directory."""
    candidates = [
        REPO / "third_party" / "ditto-talkinghead",
        REPO.parent / "repos" / "ditto-talkinghead",
        Path("/root/autodl-tmp/repos/ditto-talkinghead"),
    ]
    for c in candidates:
        if (c / "inference.py").exists():
            return str(c)
    return None


def list_ditto_bin(ditto_bin):
    """Find ffmpeg in the ditto environment."""
    if ditto_bin:
        ffmpeg = Path(ditto_bin) / "ffmpeg"
        if ffmpeg.exists():
            return ffmpeg
    env_path = Path("/root/autodl-tmp/envs/ditto/bin/ffmpeg")
    if env_path.exists():
        return env_path
    return "ffmpeg"


def find_python():
    """Find the correct Python interpreter."""
    candidates = [
        "/root/autodl-tmp/envs/ditto/bin/python",
        "python3",
        "python",
    ]
    for c in candidates:
        try:
            subprocess.run([c, "--version"], capture_output=True, timeout=5)
            return c
        except Exception:
            continue
    return "python3"


PYTHON_BIN = find_python()


def run_ditto_on_run(run_id):
    """Run 03_ditto.py for a specific run ID."""
    script = str(REPO / "scripts" / "03_ditto.py")
    try:
        result = subprocess.run(
            [PYTHON_BIN, script, "--run_id", run_id],
            cwd=str(REPO), capture_output=True, text=True, timeout=600
        )
        ok = "failed" not in result.stdout.lower() and "error" not in result.stderr.lower()
        for line in result.stdout.strip().split("\n"):
            if "ok" in line or "failed" in line or "error" in line.lower():
                print(f"    {line.strip()}")
        return ok
    except Exception as e:
        print(f"    ditto ERROR: {e}")
        return False


def run_eval_on_run(run_id):
    """Run 04_eval.py for a specific run ID."""
    script = str(REPO / "scripts" / "04_eval.py")
    try:
        result = subprocess.run(
            [PYTHON_BIN, script, "--run_id", run_id],
            cwd=str(REPO), capture_output=True, text=True, timeout=300
        )
        return True
    except Exception as e:
        print(f"    eval ERROR: {e}")
        return False



def collect_syncnet(run_dir):
    """Collect SyncC/SyncD from a run directory's 04_eval."""
    results = {}
    tts_raw = run_dir / "04_eval" / "tts_raw"
    for jf in tts_raw.glob("*/syncnet.json"):
        try:
            d = json.loads(jf.read_text())
            sid = int(d["sample_id"])
            results[sid] = {
                "sync_c": float(d["sync_c"]),
                "sync_d": float(d["sync_d"])
            }
        except Exception:
            pass
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, help="Comma-separated transform types (gain,lowpass,eq)")
    parser.add_argument("--skip_ditto", action="store_true", help="Skip Ditto, only generate audio")
    parser.add_argument("--skip_eval", action="store_true", help="Skip eval")
    args = parser.parse_args()

    # Load baseline audio
    print("Loading baseline audio...")
    baseline_audio = {}
    for i in SAMPLES:
        fpath = SRC_AUDIO_DIR / f"{i}.wav"
        if fpath.exists():
            y, sr = librosa.load(str(fpath), sr=None, mono=True)
            baseline_audio[i] = (y, sr)
            print(f"  sample {i}: {len(y)/sr:.1f}s @ {sr}Hz")
        else:
            print(f"  sample {i}: MISSING {fpath}")

    if not baseline_audio:
        print("ERROR: No baseline audio found. Run on server.")
        sys.exit(1)

    ditto_bin = get_ditto_bin_path()
    ffmpeg_bin = list_ditto_bin(ditto_bin)
    print(f"Ditto path: {ditto_bin}")
    print(f"FFmpeg: {ffmpeg_bin}")

    # =========================================================================
    # Define transformations
    # =========================================================================
    transforms = []

    if not args.only or "gain" in args.only:
        for db in [-10, -5, 0, +5, +10]:
            transforms.append({
                "name": f"gain_{db:+d}dB",
                "type": "gain",
                "param": db,
                "apply": lambda y, sr, db=db: apply_gain(y, db)
            })

    if not args.only or "lowpass" in args.only:
        for cutoff in [3000, 6000, 8000]:
            transforms.append({
                "name": f"lowpass_{cutoff}hz",
                "type": "lowpass",
                "param": cutoff,
                "apply": lambda y, sr, cutoff=cutoff: apply_lowpass(y, sr, cutoff)
            })

    if not args.only or "eq" in args.only:
        for gain in [+6, -6]:
            transforms.append({
                "name": f"eq_1k4k_{gain:+d}dB",
                "type": "eq",
                "param": gain,
                "apply": lambda y, sr, gain=gain: apply_eq_band(y, sr, 1000, 4000, gain)
            })

    print(f"\nTransformation variants: {len(transforms)}")
    for t in transforms:
        print(f"  {t['name']}")

    # =========================================================================
    # Generate transformed audio and run Ditto
    # =========================================================================
    all_rows = []

    for tform in transforms:
        tname = tform["name"]
        run_id = f"r5_f5_{tname.replace('+','p').replace('-','m')}"
        run_dir = REPO / "runs" / run_id

        print(f"\n{'='*60}")
        print(f"[{tname}] run_id={run_id}")
        print(f"{'='*60}")

        # Create directory structure
        out_dir = run_dir / "02_tts"
        out_dir.mkdir(parents=True, exist_ok=True)
        trans_dir = run_dir / "01_transcript"
        trans_dir.mkdir(parents=True, exist_ok=True)

        # Copy transcripts
        for i in SAMPLES:
            src = SRC_TRANSCRIPT_DIR / f"{i}.txt"
            dst = trans_dir / f"{i}.txt"
            if src.exists():
                dst.write_text(src.read_text())

        # Apply transformation and save
        for i, (y, sr) in baseline_audio.items():
            y_t = tform["apply"](y, sr)
            # Normalize after transformation to prevent clipping
            peak = np.max(np.abs(y_t))
            if peak > 0.95:
                y_t = y_t / peak * 0.95
            out_path = out_dir / f"{i}.wav"
            sf.write(str(out_path), y_t, sr, subtype="PCM_16")
            print(f"  sample {i}: peak={np.max(np.abs(y_t)):.3f} rms={np.sqrt(np.mean(y_t**2)):.4f}")

        # Run Ditto
        if not args.skip_ditto and ditto_bin:
            print(f"  Running Ditto [{run_id}]...")
            t0 = time.monotonic()
            ok = run_ditto_on_run(run_id)
            print(f"  Ditto: {time.monotonic()-t0:.0f}s")
            if not ok:
                print(f"  WARNING: Ditto may have failed, checking output...")
                # Check output anyway
                mp4s = list((run_dir / "03_ditto" / "tts_raw").glob("*.mp4"))
                print(f"  Generated MP4s: {len(mp4s)}")
        else:
            print(f"  Skipping Ditto (--skip_ditto or no ditto_bin)")

        # Run SyncNet eval
        if not args.skip_eval and ditto_bin:
            print(f"  Running Eval [{run_id}]...")
            ok = run_eval_on_run(run_id)
        elif not ditto_bin:
            print(f"  Skipping eval (no ditto_bin)")

        # Collect results
        scores = collect_syncnet(run_dir)
        print(f"  SyncNet scores: {len(scores)} samples")
        for sid, s in scores.items():
            row = {
                "transform": tname,
                "transform_type": tform["type"],
                "transform_param": tform["param"],
                "sample": sid,
                "sync_c": s["sync_c"],
                "sync_d": s["sync_d"],
            }
            all_rows.append(row)
            print(f"    [{sid}] SyncC={s['sync_c']:.3f} SyncD={s['sync_d']:.3f}")

    # =========================================================================
    # Collect baseline scores
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"[BASELINE] {BASELINE_RUN}")
    baseline_scores = collect_syncnet(REPO / "runs" / BASELINE_RUN)
    for sid, s in baseline_scores.items():
        if sid in SAMPLES:
            row = {
                "transform": "baseline",
                "transform_type": "none",
                "transform_param": 0,
                "sample": sid,
                "sync_c": s["sync_c"],
                "sync_d": s["sync_d"],
            }
            all_rows.append(row)
            print(f"  [{sid}] SyncC={s['sync_c']:.3f} SyncD={s['sync_d']:.3f}")

    # =========================================================================
    # Write CSV
    # =========================================================================
    fieldnames = ["transform", "transform_type", "transform_param", "sample", "sync_c", "sync_d"]
    out_path = RES_DIR / "transformation_experiment.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {out_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
