#!/usr/bin/env python3
"""Frozen-video audio audit preceding the bidirectional clock experiment.

New outputs only. The original visual embeddings are reused after hash checks;
audio forwards use the same frozen official SyncNet model. All within-video
comparisons use identical window counts and exclude padding in INTERIOR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "runs/ditto_timing_rate_v1"
MANIFEST = ROOT / "runs/vsr_ditto50_linkage_v1/manifest.json"
OUT = ROOT / "runs/tts_clock_cross_20260926"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"input hash mismatch: {path}")


def convert(source, dest, *, samples=None, start_s=0, audio_filter=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    filters = []
    if start_s:
        filters.append(f"atrim=start={start_s:.9f},asetpts=PTS-STARTPTS")
    if audio_filter:
        filters.append(audio_filter)
    if samples is not None:
        filters.append(f"atrim=end_sample={samples}")
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "pcm_s16le", str(dest)]
    subprocess.run(cmd, check=True, capture_output=True)
    return cmd


def summarize_matrix(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 31 or a.shape[0] < 40:
        raise ValueError(f"bad matrix: {a.shape}")
    result = {}
    for name, m in [("full", a), ("interior", a[15:-15])]:
        curve = m.mean(axis=0)
        j = int(np.argmin(curve))
        d, b = float(curve[j]), float(np.median(curve))
        result[name] = {
            "C": b - d,
            "D": d,
            "B": b,
            "offset": 15 - j,
            "rows": len(m),
            "boundary_peak": j in (0, 30),
        }
    return result


def interval(values, seed=20260926):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(20000, len(x)))].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "ci99": np.quantile(means, [0.005, 0.995]).tolist(),
        "positive": int((x > 0).sum()),
        "n": len(x),
    }


def prepare():
    inputs = read(PARENT / "inputs.json")
    manifest = {r["id"]: r for r in read(MANIFEST)["records"]}
    rows = []
    for record in inputs["records"]:
        i = record["id"]
        for arm, condition in [("N", "natural"), ("T", "tts")]:
            r = manifest[i]
            source = r[f"{condition}_audio"]
            check_hash(source, r["sha256"][f"{condition}_audio"])
            receipt = read(PARENT / "receipts" / str(i) / f"{arm}_O.json")
            for key in ("visual", "audio", "official"):
                check_hash(receipt[f"{key}_path"], receipt[f"{key}_sha256"])
            crop = record["conditions"][condition]["crop"]
            check_hash(crop, receipt["source_crop_sha256"])
            track = read(record["conditions"][condition]["track"])
            if track["frame"] != list(range(track["frame"][0], track["frame"][-1] + 1)):
                raise ValueError("noncontinuous track")
            dest = OUT / "pcm" / str(i) / arm
            commands = {
                "source_pcm": convert(
                    source, dest / "source.wav", start_s=track["frame"][0] / 25
                ),
                "legacy_replay": convert(
                    crop, dest / "legacy.wav", samples=receipt["source_L"] * 640
                ),
            }
            rows.append(
                {
                    "id": i,
                    "arm": arm,
                    "speaker": r["speaker"],
                    "source": source,
                    "source_sha256": sha(source),
                    "crop": crop,
                    "crop_sha256": sha(crop),
                    "visual": receipt["visual_path"],
                    "visual_sha256": receipt["visual_sha256"],
                    "legacy_audio": receipt["audio_path"],
                    "legacy_audio_sha256": receipt["audio_sha256"],
                    "legacy_matrix": receipt["official_path"],
                    "legacy_matrix_sha256": receipt["official_sha256"],
                    "source_pcm": str(dest / "source.wav"),
                    "legacy_pcm": str(dest / "legacy.wav"),
                    "source_pcm_sha256": sha(dest / "source.wav"),
                    "legacy_pcm_sha256": sha(dest / "legacy.wav"),
                    "track_start": track["frame"][0],
                    "commands": commands,
                }
            )
    protocol = {
        "date": "2026-09-26",
        "stage": "fixed-video audio-chain prerequisite",
        "cohort": "all 47 previously eligible Ditto-50 pairs; single speaker S0765",
        "primary": "paired T-N INTERIOR C with original WAV uniformly decoded to PCM16/16k",
        "secondary": "legacy-vs-source difference of paired gains, B/D and FULL; same rows per video",
        "limits": "historical exploratory cohort; video generation unchanged; no perceptual claims",
        "model_sha256": read(PARENT / "protocol.json")["model_sha256"],
        "replay_tolerance": 1e-4,
        "bootstrap_draws": 20000,
        "parent_inputs_sha256": sha(PARENT / "inputs.json"),
        "rows": rows,
    }
    path = OUT / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("refusing to change frozen protocol")
    write(path, protocol)
    print(f"Prepared {len(rows)} cells", flush=True)


def extract(limit=None, device="cuda"):
    import torch

    from scripts.experiments.tts_native_gain_attribution.common import (
        gpu_compute_pids,
        gpu_lease,
    )
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    protocol = read(OUT / "protocol.json")
    lease = (
        gpu_lease(
            gpu_peak_bytes=2 * 1024**3,
            disk_temp_bytes=200 * 1024**2,
            disk_persistent_bytes=500 * 1024**2,
        )
        if device == "cuda"
        else nullcontext({"device": "cpu", "threads": 2})
    )
    with lease as gate:
        write(OUT / "compute_gate.json", gate)
        engine = SyncNetEngine(batch_size=32, device=device)
        if engine.model_hash != protocol["model_sha256"]:
            raise ValueError("model changed")
        try:
            for row in protocol["rows"][:limit]:
                result = OUT / "scores" / f"{row['id']}_{row['arm']}.json"
                if result.exists():
                    continue
                import os

                if device == "cuda" and set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError(
                        "GPU acquired by another process; checkpoint and retry later"
                    )
                for key in (
                    "source_pcm",
                    "legacy_pcm",
                    "visual",
                    "legacy_audio",
                    "legacy_matrix",
                ):
                    check_hash(row[key], row[f"{key}_sha256"])
                source, meta = engine.extract_audio(row["source_pcm"])
                replay, replay_meta = engine.extract_audio(row["legacy_pcm"])
                legacy = np.load(row["legacy_audio"])
                if replay.shape != legacy.shape:
                    raise ValueError(
                        f"replay shape mismatch: {row['id']}/{row['arm']}: {replay.shape} vs {legacy.shape}"
                    )
                replay_error = float(np.max(np.abs(replay - legacy)))
                if replay_error > protocol["replay_tolerance"]:
                    raise ValueError(f"audio replay mismatch {replay_error}")
                visual = np.load(row["visual"])
                count = min(len(visual), len(legacy), len(source))
                matrices = {
                    "legacy": engine.distance_matrix(visual[:count], legacy[:count]),
                    "source": engine.distance_matrix(visual[:count], source[:count]),
                }
                old_matrix = np.load(row["legacy_matrix"])
                # Whole legacy reconstruction checks cache provenance before support trimming.
                matrix_error = float(
                    np.max(np.abs(engine.distance_matrix(visual, legacy) - old_matrix))
                )
                if matrix_error > protocol["replay_tolerance"]:
                    raise ValueError(f"legacy matrix replay mismatch: {matrix_error}")
                dest = OUT / "features" / str(row["id"]) / row["arm"]
                dest.mkdir(parents=True, exist_ok=True)
                np.save(dest / "source.npy", source)
                np.savez_compressed(dest / "matrices.npz", **matrices)
                detail = {
                    "id": row["id"],
                    "arm": row["arm"],
                    "speaker": row["speaker"],
                    "common_rows": count,
                    "source_audio": meta,
                    "replay_audio": replay_meta,
                    "device": device,
                    "audio_replay_error": replay_error,
                    "matrix_replay_error": matrix_error,
                    "cells": {k: summarize_matrix(v) for k, v in matrices.items()},
                    "matrix_path": str(dest / "matrices.npz"),
                    "matrix_sha256": sha(dest / "matrices.npz"),
                }
                write(result, detail)
                print(
                    row["id"],
                    row["arm"],
                    {
                        k: round(v["interior"]["C"], 3)
                        for k, v in detail["cells"].items()
                    },
                    flush=True,
                )
        finally:
            engine.close()


def analyze():
    protocol = read(OUT / "protocol.json")
    scores = {
        (r["id"], r["arm"]): read(OUT / "scores" / f"{r['id']}_{r['arm']}.json")
        for r in protocol["rows"]
    }
    ids = sorted({i for i, _ in scores})
    stats = {}
    for support in ("full", "interior"):
        for metric in ("C", "D", "B"):
            gains = {}
            for mode in ("legacy", "source"):
                gains[mode] = [
                    scores[i, "T"]["cells"][mode][support][metric]
                    - scores[i, "N"]["cells"][mode][support][metric]
                    for i in ids
                ]
                stats[f"{support}_{metric}_{mode}_T_minus_N"] = interval(gains[mode])
            stats[f"{support}_{metric}_source_minus_legacy_gain"] = interval(
                np.array(gains["source"]) - gains["legacy"]
            )
    result = {
        "pairs": len(ids),
        "speakers": sorted({r["speaker"] for r in protocol["rows"]}),
        "stats": stats,
        "max_replay_error": max(r["audio_replay_error"] for r in scores.values()),
        "boundary_peaks": [
            f"{i}/{a}/{m}"
            for (i, a), r in scores.items()
            for m in ("legacy", "source")
            if r["cells"][m]["interior"]["boundary_peak"]
        ],
    }
    write(OUT / "analysis.json", result)
    print(json.dumps(result, indent=2), flush=True)


def estimate_delay(source, encoded, max_delay=3200):
    """Estimate the audio-chain delay without consulting video or SyncNet."""
    from scipy.signal import correlate, correlation_lags

    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(encoded, dtype=np.float64)
    c = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a))
    valid = np.abs(lags) <= max_delay
    lag = int(lags[valid][np.argmax(c[valid])])
    start_a, start_b = max(0, -lag), max(0, lag)
    n = min(len(a) - start_a, len(b) - start_b)
    x, y = a[start_a : start_a + n], b[start_b : start_b + n]
    scale = float(x @ y / (x @ x))
    corr = float(np.corrcoef(x, y)[0, 1])
    return {
        "lag_samples": lag,
        "lag_ms": lag / 16,
        "aligned_correlation": corr,
        "least_squares_gain": scale,
        "aligned_snr_db": float(
            10 * np.log10(np.sum((scale * x) ** 2) / np.sum((y - scale * x) ** 2))
        ),
    }


def shift_pcm(pcm, delay):
    """Positive delay pads the beginning, preserving sample count."""
    x = np.asarray(pcm)
    out = np.zeros_like(x)
    if delay == 0:
        return x.copy()
    if abs(delay) >= len(x):
        raise ValueError("delay exceeds audio")
    if delay > 0:
        out[delay:] = x[:-delay]
    else:
        out[:delay] = x[-delay:]
    return out


def prepare_delay():
    import soundfile as sf

    rows = []
    for row in read(OUT / "protocol.json")["rows"]:
        source, sr = sf.read(row["source_pcm"], dtype="int16")
        legacy, sr2 = sf.read(row["legacy_pcm"], dtype="int16")
        assert sr == sr2 == 16000
        d = estimate_delay(source, legacy)
        lag = d["lag_samples"]
        if lag < 0 or lag > 1600 or d["aligned_correlation"] < 0.90:
            raise ValueError(
                f"audio chain not a well-matched positive delay: {row['id']}/{row['arm']}: {d}"
            )
        n = min(len(source), len(legacy) - lag)
        paths = {}
        waves = {
            "source": source[:n],
            "source_delayed": shift_pcm(source[:n], lag),
            "legacy": legacy[:n],
            "legacy_advanced": legacy[lag : lag + n],
        }
        for mode, wave in waves.items():
            path = OUT / "delay" / "pcm" / str(row["id"]) / row["arm"] / f"{mode}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, wave, 16000, subtype="PCM_16")
            paths[mode] = {"path": str(path), "sha256": sha(path)}
        rows.append({**row, "delay": d, "sample_count": n, "modes": paths})
    protocol = {
        "question": "Can audio-chain delay alone recreate the legacy TTS advantage, and can its removal rescue natural scores?",
        "selection": "all 47 previously frozen pairs; no outcome filtering",
        "primary": "INTERIOR gain(source_delayed)-gain(source); reverse removal on legacy PCM",
        "support": "identical rows across four cells per video, symmetric 20-frame boundary exclusion",
        "timing_estimation": "waveform cross-correlation only, no SyncNet/video optimization",
        "inference": "exploratory causal chain check; source speaker remains S0765",
        "rows": rows,
    }
    path = OUT / "delay/protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("frozen delay protocol differs")
    write(path, protocol)
    print("Delay protocol frozen", len(rows), flush=True)


def run_delay():
    import torch

    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    engine = SyncNetEngine(batch_size=64, device="cpu")
    protocol = read(OUT / "delay/protocol.json")
    try:
        for row in protocol["rows"]:
            dest = OUT / "delay/scores" / f"{row['id']}_{row['arm']}.json"
            if dest.exists():
                continue
            features = {}
            for mode, info in row["modes"].items():
                check_hash(info["path"], info["sha256"])
                features[mode], _ = engine.extract_audio(info["path"])
            visual = np.load(row["visual"])
            n = min(len(visual), *(len(a) for a in features.values()))
            matrices = {
                mode: engine.distance_matrix(visual[:n], a[:n])
                for mode, a in features.items()
            }
            scores = {}
            for mode, matrix in matrices.items():
                # Additional edge guard covers the injected/removed sub-100ms shift.
                curve = matrix[20:-20].mean(axis=0)
                j = int(curve.argmin())
                d = float(curve[j])
                b = float(np.median(curve))
                scores[mode] = {
                    "C": b - d,
                    "D": d,
                    "B": b,
                    "offset": 15 - j,
                    "rows": n - 40,
                    "boundary_peak": j in (0, 30),
                }
            artifact = dest.with_suffix(".npz")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact, **matrices)
            write(
                dest,
                {
                    "id": row["id"],
                    "arm": row["arm"],
                    "delay": row["delay"],
                    "cells": scores,
                    "matrix_sha256": sha(artifact),
                },
            )
            print(
                "delay",
                row["id"],
                row["arm"],
                {k: round(v["C"], 3) for k, v in scores.items()},
                flush=True,
            )
    finally:
        engine.close()


def analyze_delay():
    protocol = read(OUT / "delay/protocol.json")
    rows = {
        (r["id"], r["arm"]): read(OUT / "delay/scores" / f"{r['id']}_{r['arm']}.json")
        for r in protocol["rows"]
    }
    ids = sorted({i for i, a in rows})
    modes = ["source", "source_delayed", "legacy", "legacy_advanced"]
    gains = {
        m: np.array(
            [rows[i, "T"]["cells"][m]["C"] - rows[i, "N"]["cells"][m]["C"] for i in ids]
        )
        for m in modes
    }
    stats = {m: interval(v) for m, v in gains.items()}
    for label, left, right in [
        ("inject_delay", "source_delayed", "source"),
        ("remove_delay", "legacy_advanced", "legacy"),
        ("codec_residual_delayed", "legacy", "source_delayed"),
        ("codec_residual_aligned", "legacy_advanced", "source"),
    ]:
        stats[label] = interval(gains[left] - gains[right])
    write(
        OUT / "delay/analysis.json",
        {
            "pairs": len(ids),
            "stats": stats,
            "delay_by_arm": {
                a: sorted(
                    {
                        r["delay"]["lag_samples"]
                        for (i, arm), r in rows.items()
                        if arm == a
                    }
                )
                for a in ["N", "T"]
            },
            "min_waveform_correlation": min(
                r["delay"]["aligned_correlation"] for r in rows.values()
            ),
        },
    )
    print(json.dumps(read(OUT / "delay/analysis.json"), indent=2), flush=True)


def prepare_codec():
    """Cross codec sample rate within each fixed original audio/video pair."""
    import soundfile as sf

    rows = []
    for row in read(OUT / "protocol.json")["rows"]:
        root = OUT / "codec/pcm" / str(row["id"]) / row["arm"]
        root.mkdir(parents=True, exist_ok=True)
        source, _ = sf.read(row["source_pcm"], dtype="int16")
        modes = {
            "pcm16": {"path": row["source_pcm"], "sha256": row["source_pcm_sha256"]}
        }
        for sr in [16000, 24000]:
            wav = root / f"input_{sr}.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    row["source_pcm"],
                    "-ar",
                    str(sr),
                    "-c:a",
                    "pcm_s16le",
                    str(wav),
                ],
                check=True,
            )
            if sr == 24000:
                dest = root / "pcm24_roundtrip.wav"
                convert(wav, dest)
                modes["pcm24_roundtrip"] = {"path": str(dest), "sha256": sha(dest)}
            for bitrate in ["default", "64k"]:
                avi = root / f"avi{sr}_{bitrate}.avi"
                command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(wav)]
                if bitrate != "default":
                    command += ["-b:a", bitrate]
                command += [str(avi)]
                subprocess.run(command, check=True)
                probe = json.loads(
                    subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_streams",
                            "-of",
                            "json",
                            str(avi),
                        ]
                    )
                )["streams"][0]
                dest = avi.with_suffix(".wav")
                convert(avi, dest)
                pcm, _ = sf.read(dest, dtype="int16")
                delay = estimate_delay(source, pcm)
                lag = delay["lag_samples"]
                if lag < 0 or lag > 1600:
                    raise ValueError("unexpected codec delay")
                key = f"avi{sr}_{bitrate}"
                modes[key] = {
                    "path": str(dest),
                    "sha256": sha(dest),
                    "codec": probe,
                    "delay": delay,
                    "command": command,
                }
                aligned = root / f"{key}_aligned.wav"
                sf.write(aligned, pcm[lag : lag + len(source)], 16000, subtype="PCM_16")
                modes[key + "_aligned"] = {
                    "path": str(aligned),
                    "sha256": sha(aligned),
                    "delay_removed_samples": lag,
                }
        rows.append({**row, "modes": modes})
    protocol = {
        "question": "Does the same sound score differently when passed through default AVI/MP3 at 16k versus 24k?",
        "selection": "all 12 fixed LRS3 real videos, no score-based selection"
        if OUT.name == "real12"
        else "all 47 fixed pairs; no score-based selection",
        "primary": "within-audio score AVI24-default-aligned minus AVI16-default-aligned; independently for N and T",
        "secondary": "cross source N/T with codec rate; PCM roundtrip and fixed64k controls; raw delay versus aligned",
        "support": "same rows per video across all modes, guard20; visual embeddings unchanged",
        "causal_scope": "evaluation audio codec/sample-rate path, not generator performance or perception",
        "rows": rows,
    }
    path = OUT / "codec/protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("frozen codec protocol differs")
    write(path, protocol)
    print("Codec protocol frozen", len(rows), flush=True)


def run_codec():
    import torch

    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    engine = SyncNetEngine(batch_size=64, device="cpu")
    try:
        for row in read(OUT / "codec/protocol.json")["rows"]:
            dest = OUT / "codec/scores" / f"{row['id']}_{row['arm']}.json"
            if dest.exists():
                continue
            features = {}
            for mode, info in row["modes"].items():
                check_hash(info["path"], info["sha256"])
                features[mode], _ = engine.extract_audio(info["path"])
            visual = np.load(row["visual"])
            n = min(len(visual), *(len(x) for x in features.values()))
            matrices = {
                mode: engine.distance_matrix(visual[:n], a[:n])
                for mode, a in features.items()
            }
            scores = {}
            for mode, matrix in matrices.items():
                curve = matrix[20:-20].mean(axis=0)
                j = int(curve.argmin())
                d = float(curve[j])
                b = float(np.median(curve))
                scores[mode] = {
                    "C": b - d,
                    "D": d,
                    "B": b,
                    "offset": 15 - j,
                    "rows": n - 40,
                    "boundary_peak": j in (0, 30),
                }
            artifact = dest.with_suffix(".npz")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact, **matrices)
            write(
                dest,
                {
                    "id": row["id"],
                    "arm": row["arm"],
                    "cells": scores,
                    "matrix_sha256": sha(artifact),
                },
            )
            print(
                "codec",
                row["id"],
                row["arm"],
                {
                    k: round(v["C"], 3)
                    for k, v in scores.items()
                    if k.endswith("aligned") or k == "pcm16"
                },
                flush=True,
            )
    finally:
        engine.close()


def analyze_codec():
    protocol = read(OUT / "codec/protocol.json")
    rows = {
        (r["id"], r["arm"]): read(OUT / "codec/scores" / f"{r['id']}_{r['arm']}.json")
        for r in protocol["rows"]
    }
    ids = sorted({i for i, a in rows})
    modes = list(next(iter(rows.values()))["cells"])
    stats = {}
    for arm in ["N", "T"]:
        for left, right in [
            ("avi24000_default_aligned", "avi16000_default_aligned"),
            ("avi24000_64k_aligned", "avi16000_64k_aligned"),
            ("pcm24_roundtrip", "pcm16"),
            ("avi16000_default_aligned", "pcm16"),
            ("avi24000_default_aligned", "pcm16"),
        ]:
            stats[f"{arm}_{left}_minus_{right}"] = interval(
                [
                    rows[i, arm]["cells"][left]["C"] - rows[i, arm]["cells"][right]["C"]
                    for i in ids
                ]
            )
    gains = {
        m: interval(
            [rows[i, "T"]["cells"][m]["C"] - rows[i, "N"]["cells"][m]["C"] for i in ids]
        )
        for m in modes
    }
    cross = {}
    for nr, tr in [(16000, 24000), (24000, 16000)]:
        for suffix in ["default", "default_aligned", "64k_aligned"]:
            cross[f"N{nr}_T{tr}_{suffix}"] = interval(
                [
                    rows[i, "T"]["cells"][f"avi{tr}_{suffix}"]["C"]
                    - rows[i, "N"]["cells"][f"avi{nr}_{suffix}"]["C"]
                    for i in ids
                ]
            )
    result = {
        "pairs": len(ids),
        "within_audio_effects": stats,
        "same_codec_T_minus_N": gains,
        "crossed_codec_T_minus_N": cross,
    }
    write(OUT / "codec/analysis.json", result)
    print(json.dumps(result, indent=2), flush=True)


def prepare_real():
    parent = ROOT / "runs/tts_native_gain_attribution_implementation_20260915_v1"
    inputs = read(parent / "inputs.json")["records"]
    videos = {
        r["id"]: r
        for r in read(parent / "02_fixed_video/manifest.json")["videos"]
        if r["source"] == "R"
    }
    rows = []
    for r in inputs:
        i = r["sample_id"]
        v = videos[i]
        meta = read(parent / f"02_fixed_video/features/visual/R/{i}.json")
        check_hash(meta["path"], meta["sha256"])
        check_hash(v["path"], v["file_sha256"])
        source = r["sources"]["R"]["media"]["path"]
        check_hash(source, r["sources"]["R"]["media"]["sha256"])
        selection = read(v["selection"])
        track = selection["selected"]
        first = track["start_frame"]
        pcm = OUT / f"pcm/{i}/R/source.wav"
        convert(source, pcm, start_s=first / 25)
        rows.append(
            {
                "id": i,
                "arm": "R",
                "speaker": r["source_group"],
                "source": source,
                "source_sha256": sha(source),
                "source_pcm": str(pcm),
                "source_pcm_sha256": sha(pcm),
                "visual": meta["path"],
                "visual_sha256": meta["sha256"],
                "crop": v["path"],
                "crop_sha256": v["file_sha256"],
                "track_start": first,
            }
        )
    protocol = {
        "stage": "real-video cross-source replication",
        "cohort": "same frozen 12 LRS3 sources, real videos; distinct from Chinese S0765 discovery",
        "question": "can changing only the evaluation encoding produce the same preference with natural real videos?",
        "rows": rows,
    }
    path = OUT / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("frozen real protocol differs")
    write(path, protocol)
    print("Real-video replication frozen", len(rows), flush=True)


def analyze_real():
    rows = [
        read(OUT / "codec/scores" / f"{r['id']}_R.json")
        for r in read(OUT / "protocol.json")["rows"]
    ]
    stats = {}
    for left, right in [
        ("avi24000_default_aligned", "avi16000_default_aligned"),
        ("avi24000_64k_aligned", "avi16000_64k_aligned"),
        ("pcm24_roundtrip", "pcm16"),
        ("avi16000_default_aligned", "pcm16"),
        ("avi24000_default_aligned", "pcm16"),
    ]:
        stats[f"{left}_minus_{right}"] = interval(
            [r["cells"][left]["C"] - r["cells"][right]["C"] for r in rows]
        )
    result = {
        "sources": len(rows),
        "within_audio_effects": stats,
        "means": {
            m: float(np.mean([r["cells"][m]["C"] for r in rows]))
            for m in rows[0]["cells"]
        },
    }
    write(OUT / "codec/analysis.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "extract",
            "analyze",
            "prepare-delay",
            "run-delay",
            "analyze-delay",
            "prepare-codec",
            "run-codec",
            "analyze-codec",
            "prepare-real",
            "analyze-real",
        ],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--real12", action="store_true")
    args = parser.parse_args()
    if args.real12:
        OUT = OUT / "real12"
    if args.stage == "prepare":
        prepare()
    elif args.stage == "extract":
        extract(args.limit, args.device)
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "prepare-delay":
        prepare_delay()
    elif args.stage == "run-delay":
        run_delay()
    elif args.stage == "analyze-delay":
        analyze_delay()
    elif args.stage == "prepare-codec":
        prepare_codec()
    elif args.stage == "run-codec":
        run_codec()
    elif args.stage == "analyze-codec":
        analyze_codec()
    elif args.stage == "prepare-real":
        prepare_real()
    else:
        analyze_real()


if __name__ == "__main__":
    main()
