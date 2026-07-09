#!/usr/bin/env python3
"""08_audio_features_v2.py — Extended acoustic feature extraction from all
natural + TTS audio files, merged with SyncNet (SyncC/SyncD) scores.

Adds: r4/r5 runs, natural audio, 35+ features (MFCC, formants, spectral
flatness/flux/bandwidth, onset strength, energy envelope, silence ratio).

Output: data/r4_results/audio_features_v2.csv
"""
import sys, os, json, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import pyloudnorm as pyln
from scipy.stats import percentileofscore

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "r4_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SR = 16000

EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
NORM_TYPES = {"loudnorm": "loudnorm", "durnorm": "durnorm", "bothnorm": "bothnorm"}


def extract_features_v2(filepath, target_sr=TARGET_SR):
    """Extract 35+ acoustic features from a WAV file."""
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        dur = len(y) / sr
        n_fft = 2048
        hop_length = 512

        # --- Energy / Loudness ---
        rms_energy = np.sqrt(np.mean(y ** 2))
        peak = float(np.max(np.abs(y)))

        # LUFS + Meter (create meter once, used for LUFS and LRA)
        meter = pyln.Meter(sr)
        try:
            lufs = meter.integrated_loudness(y)
        except Exception:
            lufs = float("nan")

        # LRA (Loudness Range) via short-term loudness in 3s windows
        try:
            block_size = int(3 * sr)
            hop_size = block_size // 2
            st_loudness = []
            for start in range(0, max(1, len(y) - block_size), hop_size):
                block = y[start:start + block_size]
                if len(block) < sr * 0.5:
                    continue
                try:
                    st_loudness.append(meter.integrated_loudness(block))
                except Exception:
                    pass
            if len(st_loudness) >= 3:
                st_loudness.sort()
                lo_idx = max(0, int(len(st_loudness) * 0.10))
                hi_idx = min(len(st_loudness) - 1, int(len(st_loudness) * 0.95))
                lra = float(st_loudness[hi_idx] - st_loudness[lo_idx])
            else:
                lra = float("nan")
        except Exception:
            lra = float("nan")

        # Frame-level energy
        frame_rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        energy_env_std = float(np.std(frame_rms))
        energy_env_smoothness = float(np.std(np.diff(frame_rms))) if len(frame_rms) > 1 else 0.0
        silence_ratio = float(np.mean(frame_rms < 1e-4))

        # --- F0 (pyin) ---
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=50, fmax=600, sr=sr,
                frame_length=n_fft, hop_length=hop_length
            )
            f0_voiced = f0[voiced_flag]
            f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else float("nan")
            f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 1 else 0.0
            f0_range = (max(f0_voiced) - min(f0_voiced)) if len(f0_voiced) > 1 else 0.0
        except Exception:
            f0_mean = float("nan")
            f0_std = float("nan")
            f0_range = float("nan")

        # --- Spectral ---
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        spec_centroid = float(np.mean(centroid))
        spec_centroid_std = float(np.std(centroid))

        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
        spec_rolloff = float(np.mean(rolloff))

        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        spec_bandwidth = float(np.mean(bandwidth))
        spec_bandwidth_std = float(np.std(bandwidth))

        flatness = librosa.feature.spectral_flatness(S=S)
        spec_flatness = float(np.mean(flatness))
        spec_flatness_std = float(np.std(flatness))

        # Spectral flux (frame-to-frame change)
        flux = np.diff(S, axis=1)
        spec_flux = float(np.mean(np.abs(flux)))

        # --- Zero Crossing Rate ---
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # --- Onset Strength ---
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            onset_mean = float(np.mean(onset_env))
            onset_std = float(np.std(onset_env))
        except Exception:
            onset_mean = float("nan")
            onset_std = float("nan")

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        mfcc_means = {f"mfcc_{i+1}_mean": float(np.mean(mfcc[i])) for i in range(13)}
        mfcc_stds = {f"mfcc_{i+1}_std": float(np.std(mfcc[i])) for i in range(13)}

        # --- F1, F2, F3 formants via LPC ---
        lpc_order = int(sr // 1000 + 4)
        try:
            F = librosa.lpc(y, order=lpc_order)
            roots = np.roots(F)
            roots = roots[np.imag(roots) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))
            formants_hz = angles * sr / (2 * np.pi)
            formants_hz = sorted([f for f in formants_hz if 50 < f < sr / 2 - 50])
            f1 = formants_hz[0] if len(formants_hz) >= 1 else float("nan")
            f2 = formants_hz[1] if len(formants_hz) >= 2 else float("nan")
            f3 = formants_hz[2] if len(formants_hz) >= 3 else float("nan")
        except Exception:
            f1 = f2 = f3 = float("nan")

        # --- RMS ratio (high-freq / low-freq energy) ---
        spec_sum = np.sum(S, axis=1)
        low_mask = freqs < 1000
        high_mask = (freqs >= 3000) & (freqs < sr / 2)
        low_energy = np.sum(spec_sum[low_mask])
        high_energy = np.sum(spec_sum[high_mask])
        hf_ratio = float(high_energy / (low_energy + 1e-10))

        return {
            "duration": round(dur, 4),
            "rms": round(float(rms_energy), 6),
            "peak": round(peak, 4),
            "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
            "lra": round(lra, 2) if not np.isnan(lra) else None,
            "energy_env_std": round(energy_env_std, 6),
            "energy_env_smoothness": round(energy_env_smoothness, 6),
            "silence_ratio": round(silence_ratio, 6),
            "f0_mean": round(f0_mean, 1) if not np.isnan(f0_mean) else None,
            "f0_std": round(f0_std, 1),
            "f0_range": round(f0_range, 1) if not np.isnan(f0_range) else None,
            "spec_centroid": round(spec_centroid, 0),
            "spec_centroid_std": round(spec_centroid_std, 0),
            "spec_rolloff": round(spec_rolloff, 0),
            "spec_bandwidth": round(spec_bandwidth, 0),
            "spec_bandwidth_std": round(spec_bandwidth_std, 0),
            "spec_flatness": round(spec_flatness, 6),
            "spec_flatness_std": round(spec_flatness_std, 6),
            "spec_flux": round(spec_flux, 4),
            "zcr_mean": round(zcr_mean, 6),
            "zcr_std": round(zcr_std, 6),
            "onset_mean": round(onset_mean, 4),
            "onset_std": round(onset_std, 4),
            "f1": round(f1, 1) if not np.isnan(f1) else None,
            "f2": round(f2, 1) if not np.isnan(f2) else None,
            "f3": round(f3, 1) if not np.isnan(f3) else None,
            "hf_ratio": round(hf_ratio, 4),
            "sample_rate": sr,
            **mfcc_means,
            **mfcc_stds,
        }
    except Exception as e:
        print(f"  ERROR {filepath}: {e}", file=sys.stderr)
        return None


def _parse_audio_file(fpath, run_dir, provider, run_id):
    """Parse metadata from an audio filename in a run directory."""
    sid = int(os.path.splitext(fpath.stem)[0])
    meta = {"provider": provider, "sample": sid, "run_id": run_id, "src_round": None}

    if "r5" in run_id:
        meta["src_round"] = "r5"
    elif "r4" in run_id:
        meta["src_round"] = "r4"
    elif "r3" in run_id:
        meta["src_round"] = "r3"
    elif "r2" in run_id:
        meta["src_round"] = "r2"

    return meta


def _load_syncnet_from_run(run_dir):
    """Load all SyncNet scores from a run directory."""
    run_path = REPO / "runs" / run_dir
    scores = {}  # (provider, sample_id) -> {"sync_c": ..., "sync_d": ...}

    provider_name = run_dir

    tts_raw = run_path / "04_eval" / "tts_raw"
    nat_raw = run_path / "04_eval" / "natural_raw"

    for jf in tts_raw.glob("*/syncnet.json"):
        try:
            d = json.loads(jf.read_text())
            sid = int(d["sample_id"])
            scores[(provider_name, sid, "tts")] = {
                "sync_c": float(d["sync_c"]), "sync_d": float(d["sync_d"])
            }
        except Exception:
            pass

    for jf in nat_raw.glob("*/syncnet.json"):
        try:
            d = json.loads(jf.read_text())
            sid = int(d["sample_id"])
            scores[(provider_name, sid, "nat")] = {
                "sync_c": float(d["sync_c"]), "sync_d": float(d["sync_d"])
            }
        except Exception:
            pass

    return scores


def main():
    sources = {}  # filepath (abs) -> meta dict

    # =========================================================================
    # Natural audio (as reference)
    # =========================================================================
    natural_dir = REPO / "data" / "data" / "audio"
    for f in natural_dir.glob("*.wav"):
        sid = int(os.path.splitext(f.stem)[0])
        sources[str(f)] = {
            "provider": "natural", "sample": sid, "run_id": "natural",
            "src_round": "natural"
        }
    print(f"Natural audio: {len([s for s in sources.values() if s['provider']=='natural'])}")

    # =========================================================================
    # r2 runs: scan tmp/r2_tts/ and runs/r2_*/
    # =========================================================================
    r2_providers = {
        "fish_audio": "r2_fish_audio_20260707T150723Z",
        "faster_qwen3": "r2_faster_qwen3_20260707T145233Z",
        "dashscope_vc": "r2_dashscope_vc_20260707T143930Z",
        "siliconflow": "r2_siliconflow_20260707T142139Z",
    }

    for provider_name, run_dir in r2_providers.items():
        # Check tmp dir first
        tmp_dir = REPO / "tmp" / "r2_tts" / provider_name
        for f in tmp_dir.glob("*.wav"):
            if str(f) not in sources:
                sid = int(os.path.splitext(f.stem)[0])
                sources[str(f)] = {
                    "provider": provider_name, "sample": sid,
                    "run_id": run_dir, "src_round": "r2"
                }
        # Also check run dir
        run_path = REPO / "runs" / run_dir / "02_tts"
        for f in run_path.glob("*.wav"):
            if str(f) not in sources:
                sid = int(os.path.splitext(f.stem)[0])
                sources[str(f)] = {
                    "provider": provider_name, "sample": sid,
                    "run_id": run_dir, "src_round": "r2"
                }

    print(f"After r2: {len(sources)} total files")

    # =========================================================================
    # r3 runs: indextts2 baseline + emotion + intensity + gsv
    # =========================================================================
    r3_providers = [
        ("indextts2_baseline", "r3_indextts2_baseline"),
        ("gsv_tts", "r3_gsv_tts"),
    ]
    for provider_name, run_dir in r3_providers:
        run_path = REPO / "runs" / run_dir / "02_tts"
        for f in run_path.glob("*.wav"):
            if str(f) not in sources:
                sid = int(os.path.splitext(f.stem)[0])
                sources[str(f)] = {
                    "provider": provider_name, "sample": sid,
                    "run_id": run_dir, "src_round": "r3"
                }
        # Also check tmp
        alt = REPO / "tmp" / f"{provider_name}_output"
        for f in alt.glob("*.wav"):
            if str(f) not in sources:
                sid = int(os.path.splitext(f.stem)[0])
                sources[str(f)] = {
                    "provider": provider_name, "sample": sid,
                    "run_id": run_dir, "src_round": "r3"
                }

    # r3 emotion sweep
    for emo in EMOTIONS:
        run_dir = f"r3_indextts2_emo_{emo}"
        # Check tmp
        tmp_dir = REPO / "tmp" / "indextts2_emo_sweep"
        for f in tmp_dir.glob(f"*_{emo}.wav"):
            if str(f) not in sources:
                sid_str = f.stem.split("_", 1)[0]
                try:
                    sid = int(sid_str)
                    sources[str(f)] = {
                        "provider": f"indextts2_emo_{emo}", "sample": sid,
                        "run_id": run_dir, "src_round": "r3", "emotion": emo
                    }
                except ValueError:
                    pass
        # Check runs dir
        run_path = REPO / "runs" / run_dir / "02_tts"
        for f in run_path.glob("*.wav"):
            if str(f) not in sources:
                sid = int(os.path.splitext(f.stem)[0])
                sources[str(f)] = {
                    "provider": f"indextts2_emo_{emo}", "sample": sid,
                    "run_id": run_dir, "src_round": "r3", "emotion": emo
                }

    # r3 intensity sweep
    for emo in ["calm", "happy"]:
        for alpha in ["0.3", "0.6", "1.0"]:
            run_dir = f"r3_indextts2_int_{emo}_a{alpha}"
            tmp_dir = REPO / "tmp" / "indextts2_emo_intensity"
            for f in tmp_dir.glob(f"*_{emo}_{alpha}.wav"):
                if str(f) not in sources:
                    parts = f.stem.split("_")
                    try:
                        sid = int(parts[0])
                        sources[str(f)] = {
                            "provider": f"indextts2_int_{emo}_a{alpha}",
                            "sample": sid, "run_id": run_dir, "src_round": "r3",
                            "emotion": emo, "alpha": alpha
                        }
                    except (ValueError, IndexError):
                        pass
            run_path = REPO / "runs" / run_dir / "02_tts"
            for f in run_path.glob("*.wav"):
                if str(f) not in sources:
                    sid = int(os.path.splitext(f.stem)[0])
                    sources[str(f)] = {
                        "provider": f"indextts2_int_{emo}_a{alpha}",
                        "sample": sid, "run_id": run_dir, "src_round": "r3",
                        "emotion": emo, "alpha": alpha
                    }

    print(f"After r3: {len(sources)} total files")

    # =========================================================================
    # r4 runs: normalization variants
    # =========================================================================
    for norm_type in NORM_TYPES.values():
        for provider_label, run_suffix in [
            ("fish", "fish"), ("indextts2", "indextts2"), ("gsv", "gsv")
        ]:
            run_dir = f"r4_{norm_type}_{run_suffix}"
            run_path = REPO / "runs" / run_dir / "02_tts"
            for f in run_path.glob("*.wav"):
                if str(f) not in sources:
                    sid = int(os.path.splitext(f.stem)[0])
                    provider_full = f"{provider_label}_{norm_type}"
                    sources[str(f)] = {
                        "provider": provider_full, "sample": sid,
                        "run_id": run_dir, "src_round": "r4",
                        "norm_type": norm_type, "orig_provider": provider_label
                    }
        # Emotion
        for emo in EMOTIONS:
            run_dir = f"r4_{norm_type}_emo_{emo}"
            run_path = REPO / "runs" / run_dir / "02_tts"
            for f in run_path.glob("*.wav"):
                if str(f) not in sources:
                    sid = int(os.path.splitext(f.stem)[0])
                    provider_full = f"emo_{emo}_{norm_type}"
                    sources[str(f)] = {
                        "provider": provider_full, "sample": sid,
                        "run_id": run_dir, "src_round": "r4",
                        "norm_type": norm_type, "emotion": emo
                    }

    print(f"After r4: {len(sources)} total files")

    # =========================================================================
    # r5 runs: cross-model comparison (F5, ChatTTS, Qwen3, Higgs)
    # =========================================================================
    r5_run_map = {
        "r5_f5_tts": "f5_tts",
        "r5_f5_tts_loudnorm": "f5_tts_loudnorm",
        "r5_chattts_original": "chattts_original",
        "r5_chattts_loudnorm": "chattts_loudnorm",
        "r5_qwen_loudnorm": "qwen3_loudnorm",
        "r5_higgs_tts": "higgs_tts",
        "r5_higgs_loudnorm": "higgs_loudnorm",
    }
    for run_dir, provider_name in r5_run_map.items():
        run_path = REPO / "runs" / run_dir / "02_tts"
        for f in run_path.glob("*.wav"):
            if str(f) not in sources:
                sid = int(os.path.splitext(f.stem)[0])
                sources[str(f)] = {
                    "provider": provider_name, "sample": sid,
                    "run_id": run_dir, "src_round": "r5"
                }

    print(f"Final total TTS+Natural files: {len(sources)}")

    # =========================================================================
    # Extract features
    # =========================================================================
    rows = []
    total = len(sources)
    for i, (filepath, meta) in enumerate(sorted(sources.items())):
        if (i + 1) % 50 == 0:
            print(f"  Extracting {i+1}/{total}...", file=sys.stderr)
        feats = extract_features_v2(filepath)
        if feats:
            row = {**meta, "filepath": os.path.relpath(filepath, REPO), **feats}
            rows.append(row)

    print(f"Features extracted: {len(rows)} / {total} files", file=sys.stderr)

    # =========================================================================
    # Load SyncNet scores and merge
    # =========================================================================
    print("Loading SyncNet scores...", file=sys.stderr)
    all_scores = {}  # (run_id, sample_id, "tts"/"nat") -> {sync_c, sync_d}

    for run_dir in set(row["run_id"] for row in rows if row["run_id"] != "natural"):
        scores = _load_syncnet_from_run(run_dir)
        for (prov, sid, cond), d in scores.items():
            # Store with run_id as key for matching
            all_scores[(run_dir, sid, cond)] = d
        if len(scores) > 0:
            sample_ids = set(s for _, s, _ in scores)
            tts_count = sum(1 for _, _, c in scores if c == "tts")
            nat_count = sum(1 for _, _, c in scores if c == "nat")
            print(f"  {run_dir}: TTS={tts_count} NAT={nat_count} (samples: {sorted(sample_ids)})",
                  file=sys.stderr)

    # Merge scores into rows
    for row in rows:
        run_id = row["run_id"]
        sid = row["sample"]
        if run_id == "natural":
            continue
        tts_key = (run_id, sid, "tts")
        nat_key = (run_id, sid, "nat")
        if tts_key in all_scores:
            row["sync_c"] = all_scores[tts_key]["sync_c"]
            row["sync_d"] = all_scores[tts_key]["sync_d"]
        else:
            row["sync_c"] = None
            row["sync_d"] = None
        if nat_key in all_scores:
            row["nat_sync_c"] = all_scores[nat_key]["sync_c"]
            row["nat_sync_d"] = all_scores[nat_key]["sync_d"]
        else:
            row["nat_sync_c"] = None
            row["nat_sync_d"] = None

    # =========================================================================
    # Write CSV
    # =========================================================================
    # Collect all fieldnames (features first, then metafields)
    meta_fields = ["provider", "sample", "run_id", "src_round",
                   "emotion", "alpha", "norm_type", "orig_provider",
                   "sync_c", "sync_d", "nat_sync_c", "nat_sync_d", "filepath"]
    feature_order = [
        "duration", "rms", "peak", "lufs", "lra",
        "energy_env_std", "energy_env_smoothness", "silence_ratio",
        "f0_mean", "f0_std", "f0_range",
        "spec_centroid", "spec_centroid_std", "spec_rolloff",
        "spec_bandwidth", "spec_bandwidth_std",
        "spec_flatness", "spec_flatness_std", "spec_flux",
        "zcr_mean", "zcr_std",
        "onset_mean", "onset_std",
        "f1", "f2", "f3", "hf_ratio",
    ] + [f"mfcc_{i+1}_{s}" for i in range(13) for s in ["mean", "std"]]

    fieldnames = meta_fields + feature_order

    out_path = OUT_DIR / "audio_features_v2.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    scored = [r for r in rows if r.get("sync_c") is not None]
    providers = sorted(set(r["provider"] for r in rows))
    print(f"\nSaved: {out_path} ({len(rows)} rows, {len(scored)} with SyncC)", file=sys.stderr)
    print(f"Providers ({len(providers)}): {', '.join(providers)}", file=sys.stderr)


if __name__ == "__main__":
    main()
