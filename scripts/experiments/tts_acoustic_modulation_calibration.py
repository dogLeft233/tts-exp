"""Frozen CPU-only waveform manipulation calibration on old 26cal N/cloud-T pairs.

No scorer, model, GPU, ASR, or evaluation-cohort input is used here. The public
transform_pair function implements the exact candidate for later G x E work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tts_acoustic_modulation_calibration_20260926"
SR, NFFT, HOP = 16000, 512, 128
WINDOW = np.hanning(NFFT + 1)[:-1]
CONDITIONS = {"raw": None, "alpha08": 0.8, "identity": 1.0, "alpha12": 1.2}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def centered_frames(x):
    """Whole-hop tail coverage; reflect padding does not change time origin."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite mono waveform")
    tail = (-len(x)) % HOP
    padded = np.pad(x, (NFFT // 2, NFFT // 2 + tail), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(padded, NFFT)[::HOP]


def stft(x):
    return np.fft.rfft(centered_frames(x) * WINDOW, axis=1).T


def istft(z, length):
    frames = np.fft.irfft(z.T, n=NFFT, axis=1) * WINDOW
    total = (len(frames) - 1) * HOP + NFFT
    y, weight = np.zeros(total), np.zeros(total)
    for i, frame in enumerate(frames):
        start = i * HOP
        y[start:start + NFFT] += frame
        weight[start:start + NFFT] += WINDOW**2
    sl = slice(NFFT // 2, NFFT // 2 + length)
    return y[sl] / weight[sl]


def decompose(z):
    a = np.abs(z)
    maximum = float(a.max())
    if maximum == 0:
        log = np.zeros_like(a)
        floor_fraction = 1.0
    else:
        floor = maximum * 1e-4
        log = 20 * np.log10(np.maximum(a, floor))
        floor_fraction = float(np.mean(a < floor))
    mu = float(log.mean())
    f = log.mean(axis=1, keepdims=True) - mu
    e = log.mean(axis=0, keepdims=True) - mu
    residual = log - mu - f - e
    return residual, {"log_mu_db": mu, "floor_fraction": floor_fraction,
                      "residual_freq_mean_max": float(np.max(np.abs(residual.mean(axis=1)))),
                      "residual_time_mean_max": float(np.max(np.abs(residual.mean(axis=0))))}


def transform(x, alpha):
    """Apply a frozen positive-real gain to original complex STFT and restore RMS."""
    x = np.asarray(x, dtype=np.float64)
    z = stft(x)
    residual, info = decompose(z)
    requested = (alpha - 1.0) * residual
    gain_db = np.clip(requested, -6.0, 6.0)
    gain = 10 ** (gain_db / 20)
    modified = z * gain
    y = istft(modified, len(x))
    input_rms, output_rms = rms(x), rms(y)
    rms_gain = input_rms / output_rms if output_rms else 1.0
    y *= rms_gain
    nonzero = np.abs(z) > 0
    ratio = modified[nonzero] / z[nonzero]
    info.update({"gain_clipping_fraction": float(np.mean(np.abs(requested) > 6)),
                 "gain_db_min": float(gain_db.min()), "gain_db_max": float(gain_db.max()),
                 "rms_restore_gain": rms_gain,
                 "rms_restored_relative_error": abs(rms(y) - input_rms) / input_rms if input_rms else 0.0,
                 "design_phase_error_radians": float(np.max(np.abs(np.angle(ratio)))) if ratio.size else 0.0,
                 "design_zero_bins_preserved": bool(np.all(modified[~nonzero] == 0)),
                 "identity_float64_max_abs": float(np.max(np.abs(y - x))) if alpha == 1 else None})
    return y, info


def transform_pair(natural, tts):
    """Return unscaled/ready-to-render waveforms and one gain shared by all 8 cells."""
    before, metadata = {}, {}
    for arm, x in (("N", natural), ("T", tts)):
        before[arm], metadata[arm] = {}, {}
        for condition, alpha in CONDITIONS.items():
            if alpha is None:
                before[arm][condition] = np.asarray(x, dtype=np.float64).copy()
                metadata[arm][condition] = {"gain_clipping_fraction": 0.0, "gain_db_min": 0.0,
                    "gain_db_max": 0.0, "rms_restore_gain": 1.0, "rms_restored_relative_error": 0.0,
                    "design_phase_error_radians": 0.0, "design_zero_bins_preserved": True,
                    "identity_float64_max_abs": None}
            else:
                before[arm][condition], metadata[arm][condition] = transform(x, alpha)
    peak = max(float(np.max(np.abs(y))) for waveforms in before.values() for y in waveforms.values())
    g = min(1.0, 0.98 / peak) if peak else 1.0
    after = {arm: {condition: y * g for condition, y in waveforms.items()} for arm, waveforms in before.items()}
    return before, after, g, metadata


def f0_proxy(x):
    try:
        import pyworld
    except ImportError:
        return np.empty(0), {"f0_status": "not_measured_pyworld_unavailable",
                             "f0_median_hz": None, "voiced_fraction": None}
    f0, time = pyworld.dio(np.ascontiguousarray(x, dtype=np.float64), SR,
                         f0_floor=50, f0_ceil=500, frame_period=8.0)
    f0 = pyworld.stonemask(np.ascontiguousarray(x, dtype=np.float64), f0, time, SR)
    voiced = f0 > 0
    return f0, {"f0_status": "pyworld_dio_stonemask_proxy", "f0_median_hz": float(np.median(f0[voiced])) if voiced.any() else None,
                "voiced_fraction": float(voiced.mean())}


def analyze(x):
    z = stft(x)
    residual, info = decompose(z)
    envelope = np.sqrt(np.mean(centered_frames(x)**2, axis=1))
    spectrum = np.abs(z).mean(axis=1)
    floor = max(float(spectrum.max()) * 1e-4, np.finfo(float).tiny)
    spectrum_db = 20 * np.log10(np.maximum(spectrum, floor))
    freq = np.fft.rfftfreq(NFFT, 1 / SR)
    band = (freq >= 125) & (freq <= 7500)
    tilt = float(np.polyfit(np.log2(freq[band]), spectrum_db[band], 1)[0])
    f0, pitch_info = f0_proxy(x)
    info.update({"residual_rms_db": rms(residual), "global_rms": rms(x), "peak": float(np.max(np.abs(x))),
                 "sample_count": len(x), "all_finite": bool(np.isfinite(x).all()),
                 "sample_clipping_fraction": float(np.mean(np.abs(x) >= 1)), "spectrum_tilt_db_octave": tilt,
                 "envelope_log_sd_db": float(np.std(20 * np.log10(np.maximum(envelope, max(float(envelope.max()) * 1e-4, np.finfo(float).tiny))))),
                 **pitch_info})
    return info, {"envelope_rms": envelope, "spectrum_amplitude": spectrum, "spectrum_db": spectrum_db, "f0_hz": f0,
                  "residual_rms_by_frame_db": np.sqrt(np.mean(residual**2, axis=0))}


def couplings(values, raw):
    env, ref = values["envelope_rms"], raw["envelope_rms"]
    floor = max(float(ref.max()) * 1e-4, np.finfo(float).tiny)
    diff = 20 * np.log10(np.maximum(env, floor) / np.maximum(ref, floor))
    denom = rms(ref)
    a, b = values["f0_hz"], raw["f0_hz"]
    voiced = (a > 0) & (b > 0) if len(a) == len(b) else np.zeros(0, dtype=bool)
    cents = np.abs(1200 * np.log2(a[voiced] / b[voiced])) if voiced.any() else np.empty(0)
    return {"envelope_db_rmse_vs_raw": rms(diff), "envelope_db_abs_p95_vs_raw": float(np.percentile(np.abs(diff), 95)),
            "envelope_linear_nrmse_vs_raw": rms(env-ref) / denom if denom else 0.0,
            "envelope_correlation_vs_raw": float(np.corrcoef(env, ref)[0, 1]) if np.std(env) and np.std(ref) else None,
            "long_term_spectrum_db_rmse_vs_raw": rms((values["spectrum_db"] - raw["spectrum_db"])[1:]),
            "f0_common_voiced_frames": int(voiced.sum()),
            "f0_common_voiced_median_abs_cents": float(np.median(cents)) if len(cents) else None,
            "f0_common_voiced_p95_abs_cents": float(np.percentile(cents, 95)) if len(cents) else None,
            "f0_voicing_disagreement_fraction": float(np.mean((a > 0) != (b > 0))) if len(a) and len(a) == len(b) else None}


def synthetic_checks():
    n = 16003
    t = np.arange(n) / SR
    examples = {"zero": np.zeros(n), "tone220": 0.4*np.sin(2*np.pi*220*t),
                "am_multitone": 0.12*(1+.7*np.sin(2*np.pi*3*t))*np.sin(2*np.pi*193*t) +
                                0.1*(1+.6*np.cos(2*np.pi*7*t))*np.sin(2*np.pi*733*t)}
    for position in [0, 1, n//2, n-2, n-1]:
        wave = np.zeros(n); wave[position] = 0.5; examples[f"impulse_{position}"] = wave
    results = {}
    for name, x in examples.items():
        results[name] = {}
        for condition, alpha in CONDITIONS.items():
            if alpha is None:
                continue
            y, detail = transform(x, alpha)
            lag = int(correlation_lags(len(x), len(y))[np.argmax(correlate(y, x, method="fft"))]) if np.any(x) else 0
            detail.update({"finite": bool(np.isfinite(y).all()), "exact_length": len(y) == len(x),
                           "correlation_lag_samples": lag, "zero_stays_zero": bool(np.all(y == 0)) if name == "zero" else None})
            assert detail["finite"] and detail["exact_length"] and detail["design_zero_bins_preserved"]
            assert detail["design_phase_error_radians"] < 1e-12
            assert detail["residual_freq_mean_max"] < 1e-10 and detail["residual_time_mean_max"] < 1e-10
            if alpha == 1:
                assert detail["identity_float64_max_abs"] <= 1e-8 and lag == 0
            if name == "zero":
                assert detail["zero_stays_zero"]
            results[name][condition] = detail
    return results


def summary_stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(values), "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}


def run(out=OUT):
    out = Path(out)
    p = json.loads((out / "protocol.json").read_text())
    assert sha(out / "protocol.json") == (out / "protocol.sha256").read_text().strip()
    assert len(p["cal_ids"]) == 26 and len(p["sources"]) == 52
    assert p["conditions"] == CONDITIONS
    if (out / "summary.json").exists():
        raise RuntimeError("Calibration already exists; preserve first outcome")
    (out / "code_snapshot").mkdir(exist_ok=True)
    code = Path(__file__)
    (out / "code_snapshot" / code.name).write_bytes(code.read_bytes())
    controls = synthetic_checks()
    write_json(out / "synthetic_controls.json", controls)
    rows, manifest, gains = [], [], {}
    for sample_id in p["cal_ids"]:
        sources = {r["arm"]: r for r in p["sources"] if r["id"] == sample_id}
        assert set(sources) == {"N", "T"}
        inputs = {}
        for arm, source in sources.items():
            assert sha(source["path"]) == source["sha256"]
            x, sr = sf.read(source["path"], dtype="float64")
            assert sr == SR and x.ndim == 1
            inputs[arm] = x
        before, after, g, metadata = transform_pair(inputs["N"], inputs["T"])
        gains[sample_id] = g
        for arm in ["N", "T"]:
            raw_values = None
            for condition in CONDITIONS:
                destinations = {}
                for stage, waveforms in (("pre_headroom", before), ("waveforms", after)):
                    path = out / stage / sample_id / arm / f"{condition}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, waveforms[arm][condition], SR, subtype="FLOAT")
                    destinations[stage] = {"path": str(path), "sha256": sha(path)}
                y, sr = sf.read(destinations["waveforms"]["path"], dtype="float64")
                info, values = analyze(y)
                if condition == "raw":
                    raw_values = values
                coupling = couplings(values, raw_values)
                detail = metadata[arm][condition]
                row = {"id": sample_id, "arm": arm, "condition": condition, "speaker": sources[arm]["speaker"],
                       "source_rms": rms(inputs[arm]), "non_silent": rms(inputs[arm]) > 1e-8,
                       "shared_headroom_gain": g, **detail, **info, **coupling,
                       "saved_rms_relative_error": abs(rms(y)-g*rms(inputs[arm]))/(g*rms(inputs[arm])) if rms(inputs[arm]) else 0.0,
                       "saved_identity_float32_max_abs": float(np.max(np.abs(y - after[arm]["raw"]))) if condition == "identity" else None,
                       "exact_length": len(y) == len(inputs[arm])}
                rows.append(row)
                array_path = out / "acoustics" / sample_id / arm / f"{condition}.npz"
                array_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(array_path, **values)
                manifest.append({"id": sample_id, "arm": arm, "condition": condition, **destinations,
                                 "acoustics": {"path": str(array_path), "sha256": sha(array_path)}})
    lookup = {(r["id"], r["arm"], r["condition"]): r for r in rows}
    changes, raw_differences = [], []
    for sample_id in p["cal_ids"]:
        for arm in ["N", "T"]:
            a, b, c = [lookup[sample_id, arm, cond] for cond in ["alpha08", "identity", "alpha12"]]
            base = b["residual_rms_db"]
            changes.append({"id": sample_id, "arm": arm, "non_silent": b["non_silent"],
                            "ordered": a["residual_rms_db"] < base < c["residual_rms_db"],
                            "decrease_relative": (base-a["residual_rms_db"])/base if base else 0.0,
                            "increase_relative": (c["residual_rms_db"]-base)/base if base else 0.0})
        n, t = [lookup[sample_id, arm, "raw"] for arm in ["N", "T"]]
        fields = ["residual_rms_db", "source_rms", "envelope_log_sd_db", "spectrum_tilt_db_octave", "f0_median_hz", "voiced_fraction", "floor_fraction"]
        raw_differences.append({"id": sample_id, "speaker": n["speaker"],
                                **{k: t[k]-n[k] if t[k] is not None and n[k] is not None else None for k in fields}})
    active = [r for r in changes if r["non_silent"]]
    ordering = float(np.mean([r["ordered"] for r in active])) if active else 0.0
    decrease = float(np.median([r["decrease_relative"] for r in active])) if active else 0.0
    increase = float(np.median([r["increase_relative"] for r in active])) if active else 0.0
    gates = {"ordered_at_least_90_percent": ordering >= .9, "decrease_median_at_least_5_percent": decrease >= .05,
             "increase_median_at_least_5_percent": increase >= .05,
             "identity_float64": all(r["identity_float64_max_abs"] <= 1e-8 for r in rows if r["condition"] == "identity"),
             "identity_stored_float32": all(r["saved_identity_float32_max_abs"] <= 1e-7 for r in rows if r["condition"] == "identity"),
             "restored_rms": all(r["rms_restored_relative_error"] <= 1e-5 for r in rows),
             "saved_rms": all(r["saved_rms_relative_error"] <= 1e-5 for r in rows),
             "exact_length": all(r["exact_length"] for r in rows), "finite": all(r["all_finite"] for r in rows),
             "no_clipping": all(r["sample_clipping_fraction"] == 0 and r["peak"] <= .9800001 for r in rows)}
    coupling_fields = ["envelope_db_rmse_vs_raw", "envelope_linear_nrmse_vs_raw", "long_term_spectrum_db_rmse_vs_raw",
                       "f0_common_voiced_median_abs_cents", "f0_common_voiced_p95_abs_cents", "f0_voicing_disagreement_fraction", "gain_clipping_fraction"]
    summary = {"status": "calibration_passed" if all(gates.values()) else "calibration_failed", "gates": gates,
               "non_silent_arms": len(active), "ordered_fraction": ordering, "median_decrease_relative": decrease,
               "median_increase_relative": increase, "headroom_gains": gains,
               "couplings": {condition: {k: summary_stats([r[k] for r in rows if r["condition"] == condition]) for k in coupling_fields} for condition in CONDITIONS},
               "raw_T_minus_N_descriptive": {k: {**summary_stats([r[k] for r in raw_differences]),
                    "mean": float(np.mean([r[k] for r in raw_differences if r[k] is not None])) if any(r[k] is not None for r in raw_differences) else None} for k in fields},
               "code_sha256": sha(code), "protocol_sha256": sha(out / "protocol.json")}
    write_json(out / "metrics.json", rows)
    write_json(out / "manipulation_checks.json", changes)
    write_json(out / "raw_paired_differences.json", raw_differences)
    write_json(out / "waveform_manifest.json", manifest)
    write_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["status", "gates", "ordered_fraction", "median_decrease_relative", "median_increase_relative"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out)
