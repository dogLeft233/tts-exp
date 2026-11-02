#!/usr/bin/env python3
"""Read-only decomposition of the frozen V4 log mask into a(t) and r(f,t).

This is descriptive analysis only. It consumes the sealed log-mask NPZ and
hash-bound original-audio references; it does not call a model, score video,
train, tune, select, or access confirmation audio.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import struct
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

NPZ_SHA256 = "69925e7659e8a3e2defd5acb55a7aab5e6b92660fcbd8f9fc377aa9c1e1982be"
SUMMARY_SHA256 = "6ccdecbc270e4a577aa19c71b16cae26493d26943ca688478510e7f6f4b5e639"
STAGE_MANIFEST_SHA256 = "85be3063e9d324e6d2b519a99ec37eb781f0ba2314db47f0e014a4d7e83b9cf8"
FIT_AUDIO_MANIFEST_SHA256 = "279baeca4b05c21fbc57bd9fbc8a625572f4b99799fd9548e904f17cf4416cc0"
ASR_INPUT_MANIFEST_SHA256 = "63c463d1147f52f36e775dbfc5ab4e0b6abd880224771f3d7c869ac644ce284e"
PAIRED_SCORE_SUMMARY_SHA256 = "a5858f087ef5b83f04665d30b7f61c24cff2d494d0d58050b4e15348727ca9bf"
NPZ_BYTES = 46_753_644
SAMPLE_RATE = 16_000
N_FFT = 512
HOP = 128
LN_TO_DB = 20.0 / math.log(10.0)
BANDS = (("0_300", 0.0, 300.0), ("300_1000", 300.0, 1000.0),
         ("1000_3000", 1000.0, 3000.0), ("3000_6000", 3000.0, 6000.0),
         ("6000_8000", 6000.0, 8000.0001))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pinned(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular nonsymlink file: {path}")
    raw = path.read_bytes()
    actual = sha_bytes(raw)
    if actual != expected_sha:
        raise RuntimeError(f"{label} raw SHA mismatch: {actual}")
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return obj, actual


def read_pcm16_wav(path: Path, expected_frames: int) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"natural waveform must be a regular nonsymlink file: {path}")
    file_sha = sha_file(path)
    with wave.open(str(path), "rb") as w:
        params = (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype(), w.getnframes())
        if params != (SAMPLE_RATE, 1, 2, "NONE", expected_frames):
            raise RuntimeError(f"PCM16 WAV header/frame mismatch: {path}: {params}")
        payload = w.readframes(w.getnframes())
    expected_payload_bytes = expected_frames * 2
    if len(payload) != expected_payload_bytes:
        raise RuntimeError(f"PCM16 payload length mismatch: {path}")
    pcm_payload_sha = sha_bytes(payload)
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    return samples, {"path": str(path.resolve()), "file_sha256": file_sha,
                     "pcm16_payload_sha256": pcm_payload_sha, "frames": expected_frames,
                     "sample_rate_hz": SAMPLE_RATE, "channels": 1, "encoding": "PCM_S16LE",
                     "waveform_use": "same-clock 16-kHz source derivative for descriptive envelope/flux only"}


def source_waveform(row: dict[str, Any], stage_by_id: dict[str, dict[str, Any]],
                    fit_by_id: dict[str, dict[str, Any]], asr_manifest: dict[str, Any],
                    natural_pcm16_dir: Path, fit_audio_dir: Path) -> tuple[Any, dict[str, Any]]:
    split = row["split"]
    sid = row["sample_id"]
    if split == "dev":
        stage_row = stage_by_id.get(sid)
        if not stage_row or stage_row.get("status") != "PASS":
            raise RuntimeError(f"missing PASS stage row for {sid}")
        nref = stage_row["natural_pcm"]
        float_pcm_sha = nref["pcm_float32_sha256"]
        path = natural_pcm16_dir / f"natural_{sid}_{float_pcm_sha}.pcm16.wav"
        x, info = read_pcm16_wav(path, int(nref["sample_count"]))
        expected_payload_sha = asr_manifest["natural_baseline_asr_reuse"]["direct_pcm16_sha256_by_sample"].get(sid)
        if expected_payload_sha is None or info["pcm16_payload_sha256"] != expected_payload_sha:
            raise RuntimeError(f"natural PCM16 payload lineage mismatch: {sid}")
        info.update({"sample_id": sid, "split": split,
                     "natural_float32_pcm_sha256": float_pcm_sha,
                     "natural_float32_wav_sha256": nref["file_sha256"],
                     "original_source_path": nref["source_path"],
                     "original_source_file_sha256": nref["source_file_sha256"],
                     "pcm16_lineage": "file stem binds natural float32 PCM digest; payload SHA matches sealed direct_pcm16_sha256_by_sample; FFmpeg conversion is pinned in ASR input manifest"})
        return x, info

    fit_row = fit_by_id.get(sid)
    if not fit_row:
        raise RuntimeError(f"missing frozen fit manifest row for {sid}")
    path = fit_audio_dir / f"{sid}.wav"
    x, info = read_pcm16_wav(path, int(fit_row["sample_count"]))
    if info["file_sha256"] != fit_row["file_sha256"]:
        raise RuntimeError(f"fit source WAV SHA mismatch: {sid}")
    if info["pcm16_payload_sha256"] != fit_row["direct_source_pcm16_sha256"]:
        raise RuntimeError(f"fit source PCM16 payload SHA mismatch: {sid}")
    info.update({"sample_id": sid, "split": split,
                 "natural_float32_pcm_sha256": fit_row["pcm_float32_sha256"],
                 "natural_wav_sha256": fit_row["file_sha256"],
                 "source_archive_sha256": fit_row["source_archive_sha256"],
                 "source_archive_member": fit_row["source_archive_member"],
                 "pcm16_lineage": "byte-exact frozen original fit WAV; file and PCM16 payload hashes match the fit manifest"})
    return x, info


def _corr(a: Any, b: Any) -> float | None:
    import numpy as np
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def envelope_and_flux(x: Any, frame_centers: Any) -> tuple[Any, Any]:
    """Centered periodic-Hann 512/128 features on the original sample clock."""
    import numpy as np
    n_frames = len(frame_centers)
    if not np.array_equal(np.asarray(frame_centers, dtype=np.int64), np.arange(n_frames, dtype=np.int64) * HOP):
        raise RuntimeError("mask frame centers are not the frozen 128-sample zero-origin clock")
    padded = np.pad(np.asarray(x, dtype=np.float32), (N_FFT // 2, N_FFT // 2))
    windows = np.stack([padded[int(c):int(c) + N_FFT] for c in frame_centers], axis=0)
    k = np.arange(N_FFT, dtype=np.float64)
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * k / N_FFT)).astype(np.float32)
    w = windows * hann[None, :]
    env = np.sqrt(np.mean(w.astype(np.float64) ** 2, axis=1))
    env_db = 20.0 * np.log10(np.maximum(env, 1e-12))
    mag = np.abs(np.fft.rfft(w, n=N_FFT, axis=1))
    logmag = 20.0 * np.log10(np.maximum(mag, 1e-12))
    flux = np.full(n_frames, np.nan, dtype=np.float64)
    if n_frames > 1:
        flux[1:] = np.maximum(logmag[1:, 1:] - logmag[:-1, 1:], 0.0).mean(axis=1)
    return env_db, flux


def mean_by_speaker(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    import numpy as np
    usable = [r for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in usable:
        grouped[row["speaker_id"]].append(float(row[field]))
    speaker_means = {sid: float(np.mean(vals)) for sid, vals in grouped.items()}
    return {"equal_clip_mean": float(np.mean([r[field] for r in usable])) if usable else None,
            "equal_speaker_mean": float(np.mean(list(speaker_means.values()))) if speaker_means else None,
            "speakers_with_value": len(speaker_means), "speaker_means": speaker_means}


def analyze_row(npz: Any, row: dict[str, Any], x: Any, audio_info: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    prefix = row["mask_arrays_prefix"]
    mask = np.asarray(npz[prefix + "mask_ln"])
    mu_saved = np.asarray(npz[prefix + "mu_ln"], dtype=np.float64)
    delta_saved = np.asarray(npz[prefix + "delta_ln"], dtype=np.float64)
    freqs = np.asarray(npz[prefix + "frequencies_hz"], dtype=np.float64)
    activity = np.asarray(npz[prefix + "energy_activity"], dtype=np.int8)
    centers = np.asarray(npz[prefix + "frame_center_sample"], dtype=np.int64)
    slope = np.asarray(npz[prefix + "natural_slope_db_octave"], dtype=np.float64)
    if mask.dtype != np.float32 or mask.shape != (257, len(centers)) or delta_saved.shape != mask.shape:
        raise RuntimeError(f"mask/clock shape or dtype mismatch: {row['sample_id']}")
    recomputed_mu = mask.astype(np.float64).mean(axis=1, dtype=np.float64)
    recomputed_delta = mask.astype(np.float64) - recomputed_mu[:, None]
    if not np.array_equal(mu_saved, recomputed_mu) or not np.array_equal(delta_saved, recomputed_delta):
        raise RuntimeError(f"stored mu/delta do not exactly reconstruct frozen float32 mask: {row['sample_id']}")
    mask_db = mask.astype(np.float64) * LN_TO_DB
    delta_db = delta_saved * LN_TO_DB
    a = delta_db.mean(axis=0, dtype=np.float64)
    r = delta_db - a[None, :]
    total_e = float(np.mean(delta_db * delta_db))
    common_e = float(np.mean(a * a))
    residual_e = float(np.mean(r * r))
    closure = abs(total_e - common_e - residual_e)
    if closure > max(1e-10, total_e * 1e-10):
        raise RuntimeError(f"a/r orthogonal energy identity failed: {row['sample_id']}: {closure}")
    env_db, flux = envelope_and_flux(x, centers)
    mu_db = mask_db.mean(axis=1)
    mu_centered = mu_db - mu_db.mean()
    bands: dict[str, dict[str, Any]] = {}
    band_rows: list[dict[str, Any]] = []
    r_band_series: dict[str, Any] = {}
    d_band_series: dict[str, Any] = {}
    for name, lo, hi in BANDS:
        keep = (freqs >= lo) & (freqs < hi)
        if not keep.any():
            raise RuntimeError(f"frequency band has no bins: {name}")
        rb = r[keep, :].mean(axis=0)
        db = delta_db[keep, :].mean(axis=0)
        r_band_series[name] = rb
        d_band_series[name] = db
        band_entry = {"band": name, "low_hz": lo, "high_hz_exclusive": hi,
                      "frequency_bins": int(keep.sum()),
                      "centered_mu_mean_db": float(mu_centered[keep].mean()),
                      "delta_band_rms_db": float(np.sqrt(np.mean(delta_db[keep, :] ** 2))),
                      "common_gain_rms_db": float(np.sqrt(common_e)),
                      "residual_band_rms_db": float(np.sqrt(np.mean(r[keep, :] ** 2))),
                      "residual_band_mean_series_r": _corr(rb, a),
                      "delta_band_mean_series_r": _corr(db, a)}
        bands[name] = band_entry
        band_rows.append({"split": row["split"], "speaker_id": row["speaker_id"], "sample_id": row["sample_id"], **band_entry})
    corr_r, corr_delta = {}, {}
    for i, (left, _, _) in enumerate(BANDS):
        for right, _, _ in BANDS[i + 1:]:
            key = left + "__" + right
            corr_r[key] = _corr(r_band_series[left], r_band_series[right])
            corr_delta[key] = _corr(d_band_series[left], d_band_series[right])
    rec = {
        "split": row["split"], "speaker_id": row["speaker_id"], "sample_id": row["sample_id"],
        "status": row["status"], "mask_frames": int(mask.shape[1]), "sample_count": len(x),
        "natural_audio": audio_info,
        "decomposition": {
            "definition": "In dB log-amplitude mask units, delta(f,t)=a(t)+r(f,t), a(t)=mean_f delta(f,t), r=delta-a.",
            "delta_rms_db": float(np.sqrt(total_e)), "common_gain_a_rms_db": float(np.sqrt(common_e)),
            "spectral_residual_r_rms_db": float(np.sqrt(residual_e)),
            "common_gain_share_of_unweighted_delta_square_percent": 100.0 * common_e / total_e if total_e else 0.0,
            "spectral_residual_share_of_unweighted_delta_square_percent": 100.0 * residual_e / total_e if total_e else 0.0,
            "orthogonal_energy_closure_abs_db2": closure,
            "max_abs_per_frame_frequency_mean_r_db": float(np.max(np.abs(r.mean(axis=0)))),
            "max_abs_time_mean_a_db": float(abs(a.mean())),
            "max_abs_per_frequency_time_mean_r_db": float(np.max(np.abs(r.mean(axis=1)))),
            "units_note": "unweighted squared log-mask dB norm; not waveform power or speech-signal energy",
        },
        "source_audio_association": {
            "envelope_definition": "centered periodic-Hann 512-sample frame RMS at the frozen 128-sample (8 ms) frame centers; logged in dB",
            "positive_spectral_flux_definition": "mean positive frame-to-frame dB-magnitude difference over non-DC rFFT bins; same centered periodic-Hann 512/128 clock; first frame unavailable",
            "common_gain_vs_natural_frame_log_rms_r": _corr(a, env_db),
            "common_gain_vs_natural_positive_log_spectral_flux_r": _corr(a, flux),
            "flux_correlated_frames": int(np.isfinite(flux).sum()),
            "activity_proxy_counts": {"active": int(np.sum(activity == 1)), "quiet": int(np.sum(activity == 0)),
                                      "unclassified": int(np.sum(activity < 0)),
                                      "rule": "40ms natural-RMS proxy from frozen mask summary; not phoneme/VAD labels"},
        },
        "centered_mu_band_profile": bands,
        "cross_band_pearson_r_of_mean_time_series": {"r_spectral_residual": corr_r, "delta_total": corr_delta},
        "mask_arrays_prefix": row.get("mask_arrays_prefix"),
    }
    return rec, band_rows


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{float(v):.{digits}f}"


def write_figure(out_path: Path, cohort: dict[str, Any], paired: dict[str, Any]) -> None:
    """Render a compact scientific PNG with Matplotlib."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), constrained_layout=True)
    fig.suptitle("V4 natural-audio diagnostics: full-support SyncNet and frozen log-mask structure", fontsize=16)
    ax = axes[0]
    models = ["wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18"]
    labels = ["Wav2Lip", "MuseTalk", "Ditto"]
    keys = ("full|v4_minus_identity", "full|v4_minus_rms_fixed")
    colors = ("#4472c4", "#ed7d31")
    vals: list[list[float]] = []
    for key in keys:
        vals.append([paired["aggregates"]["paired_contrasts"][key]["by_model"][m]["speaker_equal_mean_delta"]["Cbest"] for m in models])
    xx = np.arange(len(models)); width = 0.34
    ax.axhline(0.0, color="#555555", lw=0.9)
    ax.bar(xx - width / 2, vals[0], width, color=colors[0], label="V4 − identity")
    ax.bar(xx + width / 2, vals[1], width, color=colors[1], label="V4 − RMS")
    ax.set_xticks(xx, labels)
    ax.set_ylabel("Δ Cbest (SyncNet points)")
    ax.set_title("A. FULL-support Cbest deltas")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    eq_id = paired["aggregates"]["paired_contrasts"][keys[0]]["model_equal_mean_delta"]["Cbest"]
    eq_rms = paired["aggregates"]["paired_contrasts"][keys[1]]["model_equal_mean_delta"]["Cbest"]
    ax.text(0.02, -0.20, f"Model-equal: {eq_id:+.3f} vs identity; {eq_rms:+.3f} vs RMS\nPositive favors V4.",
            transform=ax.transAxes, fontsize=9, va="top")

    # Panel B: frequency-band mean of centered static profile, dev versus fit.
    ax = axes[1]
    band_names = [b[0] for b in BANDS]
    band_means = {split: cohort[split]["equal_speaker_centered_mu_band_means_db"]
                  for split in ("dev", "historical_fit")}
    colors2 = {"dev": "#4472c4", "historical_fit": "#70ad47"}
    for split, lab in (("dev", "Dev · n=20/10 speakers"), ("historical_fit", "Fit · n=8/4 speakers")):
        vals2 = [band_means[split][bn] for bn in band_names]
        ax.plot(range(len(vals2)), vals2, marker="o", lw=2, color=colors2[split], label=lab)
        for j, value in enumerate(vals2):
            ax.annotate(f"{value:.2f}", (j, value), xytext=(0, 7 if split == "dev" else -14),
                        textcoords="offset points", ha="center", fontsize=8, color=colors2[split])
    ax.set_xticks(range(5), ["0–300", "300–1k", "1–3k", "3–6k", "6–8k"])
    ax.set_ylabel("Centered μ profile mean (dB)")
    ax.set_title("B. Static frequency profile")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Panel C: a(t) common gain versus r(f,t) spectral residual.
    ax = axes[2]
    shares = {split: cohort[split]["equal_speaker_dynamic_square_shares_percent"]
              for split in ("dev", "historical_fit")}
    common_vals, residual_vals = [], []
    for i, split in enumerate(("dev", "historical_fit")):
        common_vals.append(shares[split]["common_gain_a"]["equal_speaker_mean"])
        residual_vals.append(shares[split]["spectral_residual_r"]["equal_speaker_mean"])
    xx = np.arange(2)
    ax.bar(xx, residual_vals, color="#4472c4", label="r(f,t): spectral residual")
    ax.bar(xx, common_vals, bottom=residual_vals, color="#ed7d31", label="a(t): common gain")
    ax.set_xticks(xx, ["Dev · 20 clips", "Fit · 8 clips"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of unweighted δ log-mask dB² (%)")
    ax.set_title("C. Dynamic δ energy split")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    for i, (rv, av) in enumerate(zip(residual_vals, common_vals)):
        ax.text(i, rv / 2, f"r {rv:.1f}%", ha="center", va="center", color="white", fontsize=10)
        ax.text(i, rv + av / 2, f"a {av:.1f}%", ha="center", va="center", color="white", fontsize=10)
    fig.text(0.02, 0.005, "Descriptive only; no CI. Squared log-mask norm is not waveform or speech-signal energy. Global DC is removed by post-RMS matching without an active peak cap.", fontsize=9)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--fit-audio-manifest", type=Path, required=True)
    parser.add_argument("--asr-input-manifest", type=Path, required=True)
    parser.add_argument("--natural-pcm16-dir", type=Path, required=True)
    parser.add_argument("--fit-audio-dir", type=Path, required=True)
    parser.add_argument("--paired-score-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    npz_path = args.npz.resolve(strict=True)
    if npz_path.is_symlink() or npz_path.stat().st_size != NPZ_BYTES or sha_file(npz_path) != NPZ_SHA256:
        raise RuntimeError("full log-mask NPZ size/SHA pin mismatch")
    summary, summary_sha = load_pinned(args.summary, SUMMARY_SHA256, "decomposition summary")
    stage, stage_sha = load_pinned(args.stage_manifest, STAGE_MANIFEST_SHA256, "V4 audio-stage manifest")
    fit, fit_sha = load_pinned(args.fit_audio_manifest, FIT_AUDIO_MANIFEST_SHA256, "fit audio manifest")
    asr, asr_sha = load_pinned(args.asr_input_manifest, ASR_INPUT_MANIFEST_SHA256, "natural PCM16 lineage manifest")
    paired, paired_sha = load_pinned(args.paired_score_summary, PAIRED_SCORE_SUMMARY_SHA256, "V4 paired score summary")
    if summary.get("status") != "PASS" or summary.get("scope", {}).get("confirmation_audio_read") is not False:
        raise RuntimeError("decomposition summary is incomplete or read confirmation")
    if summary.get("arrays", {}).get("raw_sha256") != NPZ_SHA256:
        raise RuntimeError("summary does not bind the pinned NPZ")
    if stage.get("status") != "PASS" or stage.get("confirmation_audio_read") is not False:
        raise RuntimeError("audio-stage manifest scope/status mismatch")
    if asr.get("scope", {}).get("confirmation_audio_read") is not False:
        raise RuntimeError("ASR input manifest is not closed to confirmation")
    stage_rows = {r["sample_id"]: r for r in stage["rows"] if r.get("split") == "dev"}
    fit_rows: dict[str, dict[str, Any]] = {}
    for source_row in fit["rows"]:
        source_manifest_row_id = str(source_row.get("source_manifest_row_id", ""))
        speaker_id, separator, _audio_basename = source_manifest_row_id.partition("_")
        if not separator or not speaker_id or source_manifest_row_id in fit_rows:
            raise RuntimeError(f"invalid or duplicate frozen fit source row id: {source_manifest_row_id}")
        fit_rows[source_manifest_row_id] = {**source_row, "sample_id": source_manifest_row_id,
                                            "speaker_id": speaker_id}
    rows = summary.get("rows")
    if len(rows) != 28 or sum(r.get("split") == "dev" for r in rows) != 20 or sum(r.get("split") == "historical_fit" for r in rows) != 8:
        raise RuntimeError("frozen 20-dev / 8-fit denominator mismatch")
    dev_ids = {r["sample_id"] for r in rows if r.get("split") == "dev"}
    fit_ids = {r["sample_id"] for r in rows if r.get("split") == "historical_fit"}
    if len(dev_ids) != 20 or len(fit_ids) != 8 or dev_ids & fit_ids:
        raise RuntimeError("development and historical fit identities are duplicated or incomplete")
    if set(stage_rows) != dev_ids or set(fit_rows) != fit_ids:
        raise RuntimeError("audio manifest sample IDs do not exactly match the frozen mask rows")
    summary_speakers = {r["sample_id"]: r["speaker_id"] for r in rows}
    if any(fit_rows[sid]["speaker_id"] != summary_speakers[sid] for sid in fit_ids):
        raise RuntimeError("frozen fit audio speaker binding differs from the mask summary")
    expected_keys: set[str] = set()
    for row in rows:
        pref = row["mask_arrays_prefix"]
        expected_keys.update(pref + s for s in ("mask_ln", "mu_ln", "delta_ln", "natural_slope_db_octave", "energy_activity", "frame_center_sample", "frequencies_hz"))
    if len(expected_keys) != 28 * 7:
        raise RuntimeError("expected NPZ array namespace changed")
    records: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    npz = np.load(npz_path, allow_pickle=False)
    try:
        if set(npz.files) != expected_keys:
            raise RuntimeError("NPZ keys differ from frozen 28-row × 7-array contract")
        for row in rows:
            x, audio_info = source_waveform(row, stage_rows, fit_rows, asr, args.natural_pcm16_dir.resolve(), args.fit_audio_dir.resolve())
            rec, br = analyze_row(npz, row, x, audio_info)
            records.append(rec)
            band_rows.extend(br)
    finally:
        npz.close()

    cohort: dict[str, Any] = {}
    for split in ("dev", "historical_fit"):
        subset = [r for r in records if r["split"] == split]
        common_stat = "decomposition"
        vals = {"common_gain_a": [], "spectral_residual_r": []}
        for row in subset:
            vals["common_gain_a"].append({"speaker_id": row["speaker_id"], "value": row[common_stat]["common_gain_share_of_unweighted_delta_square_percent"]})
            vals["spectral_residual_r"].append({"speaker_id": row["speaker_id"], "value": row[common_stat]["spectral_residual_share_of_unweighted_delta_square_percent"]})
        def aggregate(group: list[dict[str, Any]], planned_speakers: int) -> dict[str, Any]:
            speakers: dict[str, list[float]] = defaultdict(list)
            for entry in group:
                value = entry["value"]
                if value is not None and math.isfinite(float(value)):
                    speakers[entry["speaker_id"]].append(float(value))
            sm = {sid: float(np.mean(v)) for sid, v in speakers.items()}
            usable = [float(e["value"]) for e in group if e["value"] is not None and math.isfinite(float(e["value"]))]
            return {"equal_clip_mean": float(np.mean(usable)) if usable else None,
                    "equal_speaker_mean": float(np.mean(list(sm.values()))) if sm else None,
                    "speaker_means": sm, "planned_speakers": planned_speakers,
                    "speakers_with_value": len(sm), "planned_clips": len(group), "clips_with_value": len(usable)}
        share_summary = {name: aggregate(group, 10 if split == "dev" else 4) for name, group in vals.items()}
        profile: dict[str, float | None] = {}
        for bname, _, _ in BANDS:
            by_sp: dict[str, list[float]] = defaultdict(list)
            for row in subset:
                by_sp[row["speaker_id"]].append(float(row["centered_mu_band_profile"][bname]["centered_mu_mean_db"]))
            profile[bname] = float(np.mean([np.mean(x) for x in by_sp.values()])) if by_sp else None
        corr_env = [{"speaker_id": r["speaker_id"], "value": r["source_audio_association"]["common_gain_vs_natural_frame_log_rms_r"]} for r in subset]
        corr_flux = [{"speaker_id": r["speaker_id"], "value": r["source_audio_association"]["common_gain_vs_natural_positive_log_spectral_flux_r"]} for r in subset]
        coh = {}
        pairs = list(subset[0]["cross_band_pearson_r_of_mean_time_series"]["r_spectral_residual"]) if subset else []
        for pair in pairs:
            entries = [{"speaker_id": r["speaker_id"], "value": r["cross_band_pearson_r_of_mean_time_series"]["r_spectral_residual"][pair]} for r in subset]
            by_sp: dict[str, list[float]] = defaultdict(list)
            for e in entries:
                if e["value"] is not None:
                    by_sp[e["speaker_id"]].append(float(e["value"]))
            coh[pair] = float(np.mean([np.mean(v) for v in by_sp.values()])) if by_sp else None
        cohort[split] = {
            "planned_rows": 20 if split == "dev" else 8, "completed_rows": len(subset),
            "planned_speakers": 10 if split == "dev" else 4,
            "completed_speakers": len({r["speaker_id"] for r in subset}),
            "equal_speaker_dynamic_square_shares_percent": share_summary,
            "equal_speaker_centered_mu_band_means_db": profile,
            "equal_speaker_common_gain_vs_natural_log_rms_r": aggregate(corr_env, 10 if split == "dev" else 4),
            "equal_speaker_common_gain_vs_natural_positive_log_spectral_flux_r": aggregate(corr_flux, 10 if split == "dev" else 4),
            "equal_speaker_cross_band_residual_mean_series_pearson_r": coh,
            "descriptive_only_no_inference": True,
        }
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise RuntimeError(f"output directory must be new: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    rows_path = args.out_dir / "decomposition_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    bands_path = args.out_dir / "band_rows.csv"
    with bands_path.open("x", encoding="utf-8-sig", newline="") as f:
        fields = list(band_rows[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(band_rows)
    paired_metrics = paired.get("aggregates", {}).get("paired_contrasts", {})
    for key in ("full|v4_minus_identity", "full|v4_minus_rms_fixed"):
        if key not in paired_metrics:
            raise RuntimeError(f"missing FULL paired endpoint in pinned score summary: {key}")
    plot_cohort = {
        split: {
            "equal_speaker_centered_mu_band_means_db": cohort[split]["equal_speaker_centered_mu_band_means_db"],
            "equal_speaker_dynamic_square_shares_percent": cohort[split]["equal_speaker_dynamic_square_shares_percent"],
        }
        for split in ("dev", "historical_fit")
    }
    write_figure(args.out_dir / "v4_mechanism_descriptives.png", plot_cohort, paired)
    source_manifest = {
        "schema": "v4_logmask_delta_common_mode_source_manifest_v1",
        "scope": {"development_rows": 20, "development_speakers": 10, "historical_fit_rows": 8,
                  "historical_fit_speakers": 4, "confirmation_audio_read": False,
                  "video_rendered": False, "syncnet_called": False, "asr_called": False,
                  "training_performed": False, "selection_performed": False,
                  "inferential_tests_or_ci": False,
                  "description": "Audio-only descriptive decomposition; no model/scorer/API execution."},
        "analysis_source": {"path": str(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size,
                            "sha256": sha_file(Path(__file__).resolve())},
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                    "matplotlib": __import__("importlib.metadata", fromlist=["version"]).version("matplotlib")},
        "pinned_inputs": {
            "logmask_npz": {"path": str(npz_path), "bytes": npz_path.stat().st_size, "sha256": NPZ_SHA256},
            "decomposition_summary": {"path": str(args.summary.resolve()), "bytes": args.summary.stat().st_size, "sha256": summary_sha},
            "stage_manifest": {"path": str(args.stage_manifest.resolve()), "bytes": args.stage_manifest.stat().st_size, "sha256": stage_sha},
            "fit_audio_manifest": {"path": str(args.fit_audio_manifest.resolve()), "bytes": args.fit_audio_manifest.stat().st_size, "sha256": fit_sha},
            "asr_input_manifest": {"path": str(args.asr_input_manifest.resolve()), "bytes": args.asr_input_manifest.stat().st_size, "sha256": asr_sha},
            "paired_score_summary": {"path": str(args.paired_score_summary.resolve()), "bytes": args.paired_score_summary.stat().st_size, "sha256": paired_sha},
        },
        "row_audio_inputs": [{"split": r["split"], "speaker_id": r["speaker_id"], "sample_id": r["sample_id"],
                              **r["natural_audio"]} for r in records],
        "outputs_planned": ["summary.json", "decomposition_rows.jsonl", "band_rows.csv",
                            "v4_mechanism_descriptives.png", "receipt.json"],
    }
    source_manifest_path = args.out_dir / "source_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                    encoding="utf-8", newline="\n")
    source_manifest_sha = sha_file(source_manifest_path)
    output = {
        "schema": "v4_logmask_delta_common_mode_descriptive_v1", "status": "PASS",
        "scope": {"development_rows": 20, "development_speakers": 10, "historical_fit_rows": 8,
                  "historical_fit_speakers": 4, "confirmation_audio_read": False,
                  "video_rendered": False, "syncnet_called": False, "asr_called": False,
                  "training_performed": False, "selection_performed": False,
                  "inferential_tests_or_ci": False},
        "methodology": {
            "delta_split": "dB log-amplitude delta(f,t)=a(t)+r(f,t), with a(t)=frequency-bin mean of delta at each original 8-ms mask frame; r is the frequency-specific remainder.",
            "energy": "unweighted squared dB log-mask norm. a and r are orthogonal under equal-bin/equal-frame summation because r has zero per-frame frequency mean.",
            "cross_band": "Pearson correlations between each broad band's mean temporal r(f,t) series; descriptive only.",
            "source_waveform": "Original clock and sample count. Dev uses the exact direct PCM16 derivative associated with the canonical natural float32 PCM digest; historical fit uses byte-exact frozen source WAVs. Envelope/flux are descriptive inputs only.",
            "envelope_flux": "Centered periodic-Hann STFT at 512 samples / 128-sample hop, matching frozen mask frame centers; no lag search or time shifting.",
            "limitation": "This does not claim that mask dB-squared norm is waveform power or causal speech energy. Audio association is descriptive and no CI is computed."},
        "cohort_summary": cohort,
        "inputs": {"npz_path": str(npz_path), "npz_bytes": NPZ_BYTES, "npz_sha256": NPZ_SHA256,
                   "decomposition_summary_sha256": summary_sha, "stage_manifest_sha256": stage_sha,
                   "fit_audio_manifest_sha256": fit_sha, "asr_input_manifest_sha256": asr_sha,
                   "paired_score_summary_sha256": paired_sha,
                   "source_manifest_sha256": source_manifest_sha,
                   "analysis_source_sha256": source_manifest["analysis_source"]["sha256"]},
        "outputs": {},
    }
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    for p in (rows_path, bands_path, args.out_dir / "v4_mechanism_descriptives.png", source_manifest_path):
        output["outputs"][p.name] = {"bytes": p.stat().st_size, "sha256": sha_file(p)}
    out_json.write_text(json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    receipt = {"schema": "v4_logmask_delta_common_mode_receipt_v1", "status": "PASS",
               "summary_path": str(out_json), "summary_bytes": out_json.stat().st_size,
               "summary_sha256": sha_file(out_json), "figure_path": str(args.out_dir / "v4_mechanism_descriptives.png"),
               "figure_sha256": sha_file(args.out_dir / "v4_mechanism_descriptives.png"),
               "scope": output["scope"]}
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "output_dir": str(args.out_dir), "rows": len(records),
                      "dev_rows": cohort["dev"]["completed_rows"], "fit_rows": cohort["historical_fit"]["completed_rows"],
                      "confirmation_audio_read": False, "video_rendered": False, "syncnet_called": False,
                      "asr_called": False, "training_performed": False,
                      "summary_sha256": sha_file(out_json), "figure_sha256": sha_file(args.out_dir / "v4_mechanism_descriptives.png"),
                      "receipt_sha256": sha_file(args.out_dir / "receipt.json")},
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
