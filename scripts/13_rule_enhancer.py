#!/usr/bin/env python3
"""13_rule_enhancer.py — Rule-based audio enhancer for TFG lip-sync.
Based on Phase 2+3 findings: linear gain + high-freq emphasis, no compression.

3 conditions:
  1. linear_gain: target -25 LUFS via linear gain (no compression)
  2. hf_boost: +2dB shelf at 3-8kHz (no gain)
  3. combined: linear_gain + hf_boost

Baseline: F5-TTS samples [3,6,8,10,12]
Run Ditto + SyncNet for each condition.
"""
import sys, os, json, csv, time, subprocess, argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln

REPO = Path(__file__).resolve().parent.parent
SAMPLES = [3, 6, 8, 10, 12]
SRC_AUDIO = REPO / "runs" / "r5_f5_tts" / "02_tts"
SRC_TRANS = REPO / "runs" / "r5_f5_tts" / "01_transcript"
TARGET_LUFS = -25.0


def apply_linear_gain(y, sr, current_lufs, target_lufs):
    gain_db = target_lufs - current_lufs
    factor = 10 ** (gain_db / 20.0)
    return y * factor


def apply_hf_shelf(y, sr, low_hz=3000, high_hz=8000, gain_db=2.0):
    """Gentle high-frequency shelf boost using IIR filter."""
    from scipy import signal
    nyq = sr / 2
    # Bandpass filter centered on 3-8kHz
    b, a = signal.butter(2, [low_hz/nyq, high_hz/nyq], btype="band")
    filtered = signal.lfilter(b, a, y)
    factor = 10 ** (gain_db / 20.0) - 1.0
    result = y + factor * filtered
    # Prevent clipping
    peak = np.max(np.abs(result))
    if peak > 0.95:
        result = result / peak * 0.95
    return result


def get_meter(sr):
    return pyln.Meter(sr)


def find_python():
    for c in ["/root/autodl-tmp/envs/ditto/bin/python", "python3", "python"]:
        try:
            subprocess.run([c, "--version"], capture_output=True, timeout=5)
            return c
        except:
            continue
    return "python3"


PYTHON = find_python()


def run_ditto(run_id):
    script = str(REPO / "scripts" / "03_ditto.py")
    result = subprocess.run([PYTHON, script, "--run_id", run_id],
                            cwd=str(REPO), capture_output=True, text=True, timeout=600)
    for line in result.stdout.strip().split("\n"):
        if "ok" in line or "failed" in line:
            print(f"    {line.strip()}")


def run_eval(run_id):
    script = str(REPO / "scripts" / "04_eval.py")
    subprocess.run([PYTHON, script, "--run_id", run_id],
                   cwd=str(REPO), capture_output=True, text=True, timeout=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_ditto", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    conditions = {
        "linear_gain": ("r5_f5_enh_gain", lambda y, sr, lufs: apply_linear_gain(y, sr, lufs, TARGET_LUFS)),
        "hf_boost": ("r5_f5_enh_hf", lambda y, sr, lufs: apply_hf_shelf(y, sr)),
        "combined": ("r5_f5_enh_both", lambda y, sr, lufs: apply_hf_shelf(
            apply_linear_gain(y, sr, lufs, TARGET_LUFS), sr)),
    }

    # Load baseline audio and compute LUFS
    print("Loading F5-TTS baseline...")
    baseline = {}
    for i in SAMPLES:
        fpath = SRC_AUDIO / f"{i}.wav"
        if fpath.exists():
            y, sr = librosa.load(str(fpath), sr=None, mono=True)
            meter = get_meter(sr)
            try:
                lufs = meter.integrated_loudness(y)
            except:
                lufs = float("nan")
            baseline[i] = {"y": y, "sr": sr, "lufs": lufs, "peak": np.max(np.abs(y)),
                           "rms": np.sqrt(np.mean(y**2))}
            print(f"  sample {i}: LUFS={lufs:.1f} RMS={baseline[i]['rms']:.4f} "
                  f"peak={baseline[i]['peak']:.3f} dur={len(y)/sr:.1f}s")
        else:
            print(f"  sample {i}: MISSING {fpath}")

    all_rows = []

    for cond_name, (run_id, transform_fn) in conditions.items():
        print(f"\n{'='*60}")
        print(f"[{cond_name}] → {run_id}")
        run_dir = REPO / "runs" / run_id
        out_dir = run_dir / "02_tts"
        out_dir.mkdir(parents=True, exist_ok=True)
        trans_dir = run_dir / "01_transcript"
        trans_dir.mkdir(parents=True, exist_ok=True)

        # Copy transcripts
        for i in SAMPLES:
            src = SRC_TRANS / f"{i}.txt"
            if src.exists():
                (trans_dir / f"{i}.txt").write_text(src.read_text())

        # Apply enhancement
        for i in SAMPLES:
            data = baseline[i]
            y_enh = transform_fn(data["y"], data["sr"], data["lufs"])
            new_peak = np.max(np.abs(y_enh))
            new_rms = np.sqrt(np.mean(y_enh**2))
            try:
                new_lufs = get_meter(data["sr"]).integrated_loudness(y_enh)
            except:
                new_lufs = float("nan")
            gain_actual = new_lufs - data["lufs"]
            out_path = out_dir / f"{i}.wav"
            sf.write(str(out_path), y_enh, data["sr"], subtype="PCM_16")
            print(f"  [{i}] LUFS {data['lufs']:.1f}→{new_lufs:.1f} "
                  f"(Δ={gain_actual:+.1f}dB) RMS={new_rms:.4f} peak={new_peak:.3f}")

        # Ditto
        if not args.skip_ditto:
            print(f"  Ditto [{run_id}]...")
            t0 = time.monotonic()
            run_ditto(run_id)
            print(f"  Ditto: {time.monotonic()-t0:.0f}s")
        else:
            print(f"  Skipping Ditto")

        # Eval
        if not args.skip_eval:
            print(f"  Eval [{run_id}]...")
            run_eval(run_id)

        # Collect results
        tts_raw = run_dir / "04_eval" / "tts_raw"
        for i in SAMPLES:
            jf = tts_raw / str(i) / "syncnet.json"
            if jf.exists():
                d = json.loads(jf.read_text())
                all_rows.append({
                    "condition": cond_name, "sample": i,
                    "sync_c": float(d["sync_c"]), "sync_d": float(d["sync_d"]),
                })
                print(f"  [{i}] SyncC={d['sync_c']} SyncD={d['sync_d']}")

    # Baseline
    base_run = REPO / "runs" / "r5_f5_tts"
    for i in SAMPLES:
        jf = base_run / "04_eval" / "tts_raw" / str(i) / "syncnet.json"
        if jf.exists():
            d = json.loads(jf.read_text())
            all_rows.append({
                "condition": "baseline", "sample": i,
                "sync_c": float(d["sync_c"]), "sync_d": float(d["sync_d"]),
            })

    # Save
    import statistics
    outf = REPO / "data" / "r4_results" / "rule_enhancer.csv"
    with open(outf, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition","sample","sync_c","sync_d"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {outf} ({len(all_rows)} rows)")

    # Summary
    print(f"\n{'Condition':<20s} {'n':>3}  {'SyncC':>7} {'SyncD':>7}  {'ΔC':>7} {'ΔD':>7}")
    print("-" * 65)
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in all_rows:
        by_cond[r["condition"]].append(r)
    base = {r["sample"]: r for r in all_rows if r["condition"] == "baseline"}
    for cond in ["baseline", "linear_gain", "hf_boost", "combined"]:
        items = by_cond[cond]
        mc = statistics.mean([x["sync_c"] for x in items])
        md = statistics.mean([x["sync_d"] for x in items])
        dc = mc - (statistics.mean([x["sync_c"] for x in by_cond["baseline"]]) if cond != "baseline" else mc)
        dd = md - (statistics.mean([x["sync_d"] for x in by_cond["baseline"]]) if cond != "baseline" else md)
        print(f"{cond:<20s} {len(items):>3}  {mc:7.3f} {md:7.3f}  {dc:+7.3f} {dd:+7.3f}")


if __name__ == "__main__":
    main()
