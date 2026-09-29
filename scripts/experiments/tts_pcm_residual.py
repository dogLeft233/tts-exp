"""Cross-model source-PCM audit and natural-clock audio/video crossed scores."""

from __future__ import annotations

import argparse
import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.experiments.tts_clock_cross import (
    ROOT,
    check_hash,
    convert,
    read,
    sha,
    write,
)

PARENT = ROOT / "runs/aishell1_qwen_mfa_linear_n100_20260816"
OUT = ROOT / "runs/tts_pcm_residual_20260926"
STAGES = ["05_wav2lip_syncnet_n100", "12_conditional_adapter_heldout_wav2lip_syncnet"]
ARMS = {
    "N": "natural_raw",
    "T": "raw_tts",
    "M": "mfa_linear",
    "D": "direct_resynthesis",
}


def candidate(base, speaker, sid, arm):
    score_path = base / "scores" / ARMS[arm] / speaker / f"{sid}.json"
    if not score_path.exists():
        return None, "missing_score_receipt"
    s = read(score_path)
    for name in ("audio", "video", "face"):
        if not Path(s[name]).exists():
            return None, f"missing_{name}"
        check_hash(s[name], s[f"{name}_sha256"])
    cell = base / "syncnet" / ARMS[arm] / speaker / sid
    crops = sorted(cell.glob("pycrop/*/*.avi"))
    tracks = sorted(cell.glob("pywork/*/tracks.pckl"))
    if len(crops) != 1 or len(tracks) != 1:
        return None, f"nonunique_crop_track:{len(crops)}/{len(tracks)}"
    with tracks[0].open("rb") as f:
        tr = pickle.load(f)
    if len(tr) != 1:
        return None, "nonunique_track"
    frames = np.asarray(tr[0]["track"]["frame"]).astype(int)
    if not np.array_equal(frames, np.arange(frames[0], frames[-1] + 1)):
        return None, "noncontiguous_track"
    if len(frames) < 50:
        return None, "short_track"
    return {
        "arm": arm,
        "base": str(base),
        "audio": s["audio"],
        "audio_sha256": s["audio_sha256"],
        "video": s["video"],
        "video_sha256": s["video_sha256"],
        "face_sha256": s["face_sha256"],
        "crop": str(crops[0]),
        "crop_sha256": sha(crops[0]),
        "tracks": str(tracks[0]),
        "tracks_sha256": sha(tracks[0]),
        "track_start": int(frames[0]),
        "track_end": int(frames[-1]),
        "track_frames": len(frames),
        "score_receipt": str(score_path),
        "score_receipt_sha256": sha(score_path),
        "old_C": s["sync_c"],
        "old_D": s["sync_d"],
        "old_offset": s["av_offset"],
        "model_sha256": s["syncnet_model_sha256"],
        "generator_sha256": s["wav2lip_checkpoint_sha256"],
    }, None


def freeze():
    cohort = read(PARENT / "00_pairs/cohort.json")["records"]
    eligible, missing = [], []
    for r in cohort:
        sid, speaker = r["sample_id"], r["speaker_id"]
        attempts, selected = [], None
        for name in STAGES:
            cells, reasons = {}, {}
            for arm in ("N", "T", "M"):
                cell, reason = candidate(PARENT / name, speaker, sid, arm)
                if cell is not None:
                    cells[arm] = cell
                else:
                    reasons[arm] = reason
            attempts.append({"stage": name, "reasons": reasons})
            if not reasons:
                selected = cells
                break
        if selected is None:
            missing.append({"id": sid, "speaker": speaker, "attempts": attempts})
            continue
        d, dreason = candidate(
            PARENT / "14_direct_wavlm_resynthesis_wav2lip_syncnet", speaker, sid, "D"
        )
        if d:
            selected["D"] = d
        assert len({v["face_sha256"] for v in selected.values()}) == 1
        assert len({v["generator_sha256"] for v in selected.values()}) == 1
        source_lengths = {a: sf.info(v["audio"]).frames for a, v in selected.items()}
        source_rates = {a: sf.info(v["audio"]).samplerate for a, v in selected.items()}
        assert set(source_rates.values()) == {16000}
        assert source_lengths["N"] == source_lengths["M"]
        if d:
            assert source_lengths["N"] == source_lengths["D"]
        eligible.append(
            {
                "id": sid,
                "speaker": speaker,
                "cells": selected,
                "source_lengths": source_lengths,
                "direct_missing": dreason,
            }
        )
    p = {
        "date": "2026-09-26",
        "parent": str(PARENT),
        "universe": len(cohort),
        "eligible_pairs": len(eligible),
        "eligibility": "source hash, original video/face hash, unique continuous official crop >=50 frames; N/T/M all in same prioritized stage; no score selection",
        "primary": "source PCM native T-N and common-natural-audio Mvideo-Nvideo, guard20",
        "cross": "N/M share exact natural sample count; four cells on common absolute windows; D subset additional control",
        "sensitivity": "guard0/15/20; original full legacy replay; same-video legacy vs PCM common support",
        "bootstrap": "20000; seed20260926; speaker-equal means primary; utterance mean clustered secondary",
        "replay_tolerance_rounded": 0.006,
        "limits": "historical asset-available subset, cloudQwen/Wav2Lip, one common dynamic source face, exploratory",
        "rows": eligible,
        "missing": missing,
    }
    path = OUT / "protocol.json"
    if path.exists() and read(path) != p:
        raise ValueError("frozen protocol differs")
    write(path, p)
    print(
        "frozen",
        len(eligible),
        "pairs",
        len({r["speaker"] for r in eligible}),
        "speakers",
        flush=True,
    )


def prepare():
    for row in read(OUT / "protocol.json")["rows"]:
        paths = {}
        for arm, c in row["cells"].items():
            folder = OUT / "pcm" / row["id"] / arm
            for name, source in [("source", c["audio"]), ("legacy", c["crop"])]:
                path = folder / f"{name}.wav"
                if not path.exists():
                    convert(source, path)
                paths[f"{arm}_{name}"] = {"path": str(path), "sha256": sha(path)}
        write(OUT / "pcm" / row["id"] / "receipt.json", paths)
    print("audio prepared", flush=True)


def metrics(matrix):
    if len(matrix) <= 40:
        raise ValueError("insufficient common rows")
    result = {}
    for guard in (0, 15, 20):
        x = matrix[guard:-guard] if guard else matrix
        curve = x.mean(0)
        j = int(curve.argmin())
        d = float(curve[j])
        b = float(np.median(curve))
        result[str(guard)] = {
            "C": b - d,
            "B": b,
            "D": d,
            "offset": 15 - j,
            "rows": len(x),
            "boundary": j in (0, 30),
        }
    return result


def score(device):
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
            gpu_peak_bytes=3 << 30,
            disk_temp_bytes=500 << 20,
            disk_persistent_bytes=2 << 30,
        )
        if device == "cuda"
        else nullcontext({"device": "cpu"})
    )
    with lease as gate:
        write(OUT / "compute_gate.json", gate)
        engine = SyncNetEngine(batch_size=32, device=device)
        try:
            for row in protocol["rows"]:
                sid = row["id"]
                dest = OUT / "scores" / f"{sid}.json"
                if dest.exists():
                    continue
                if device == "cuda" and set(gpu_compute_pids()) - {os.getpid()}:
                    raise RuntimeError("foreign GPU process appeared")
                paths = read(OUT / "pcm" / sid / "receipt.json")
                audio, visual, features, matrices, replay = {}, {}, {}, {}, {}
                for arm, c in row["cells"].items():
                    assert engine.model_hash == c["model_sha256"]
                    for label in ("audio", "video", "crop", "tracks", "score_receipt"):
                        check_hash(c[label], c[label + "_sha256"])
                    v, vm = engine.extract_visual(c["crop"])
                    visual[arm] = v
                    features[f"v_{arm}"] = v
                    for name in ("source", "legacy"):
                        info = paths[f"{arm}_{name}"]
                        check_hash(info["path"], info["sha256"])
                        a, _ = engine.extract_audio(info["path"])
                        audio[f"{arm}_{name}"] = a
                        features[f"a_{arm}_{name}"] = a
                    old = engine.distance_matrix(v, audio[f"{arm}_legacy"])
                    oldscore = metrics(old)["0"]
                    err = max(
                        abs(oldscore["C"] - c["old_C"]), abs(oldscore["D"] - c["old_D"])
                    )
                    if err > protocol["replay_tolerance_rounded"]:
                        raise ValueError(
                            f"legacy replay {sid}/{arm}: {err}, {oldscore}, oldC {c['old_C']}"
                        )
                    replay[arm] = {"error": err, "score": oldscore, "video_meta": vm}
                    start = c["track_start"]
                    src = audio[f"{arm}_source"][start:]
                    n = min(len(v), len(src), len(audio[f"{arm}_legacy"]))
                    matrices[f"{arm}_source"] = engine.distance_matrix(v[:n], src[:n])
                    matrices[f"{arm}_legacy"] = engine.distance_matrix(
                        v[:n], audio[f"{arm}_legacy"][:n]
                    )
                # Natural-clock counterfactuals all see exactly the same absolute frame support.
                natural_arms = [a for a in ("N", "M", "D") if a in visual]
                start = max(row["cells"][a]["track_start"] for a in natural_arms)
                end = min(
                    *(
                        row["cells"][a]["track_start"] + len(visual[a])
                        for a in natural_arms
                    ),
                    *(len(audio[f"{a}_source"]) for a in natural_arms),
                )
                for va in natural_arms:
                    v = visual[va][
                        start - row["cells"][va]["track_start"] : end
                        - row["cells"][va]["track_start"]
                    ]
                    for aa in natural_arms:
                        matrices[f"cross_{va}_{aa}"] = engine.distance_matrix(
                            v, audio[f"{aa}_source"][start:end]
                        )
                cells = {k: metrics(m) for k, m in matrices.items()}
                folder = OUT / "features" / sid
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "features.npz", **features)
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                write(
                    dest,
                    {
                        "id": sid,
                        "speaker": row["speaker"],
                        "cells": cells,
                        "replay": replay,
                        "cross_support": [start, end],
                        "features_sha256": sha(folder / "features.npz"),
                        "matrices_sha256": sha(dest.with_suffix(".npz")),
                    },
                )
                print("scored", sid, row["speaker"], flush=True)
        finally:
            engine.close()


def cluster(values, speakers):
    x = np.asarray(values, dtype=float)
    names = sorted(set(speakers))
    speaker_values = np.array([x[np.array(speakers) == s].mean() for s in names])
    counts = np.array([speakers.count(s) for s in names])
    rng = np.random.default_rng(20260926)
    ix = rng.integers(len(names), size=(20000, len(names)))
    equal = speaker_values[ix].mean(1)
    weighted = (speaker_values[ix] * counts[ix]).sum(1) / counts[ix].sum(1)
    return {
        "n": len(x),
        "speakers": len(names),
        "speaker_mean": float(speaker_values.mean()),
        "speaker_ci95": np.quantile(equal, [0.025, 0.975]).tolist(),
        "speaker_ci99": np.quantile(equal, [0.005, 0.995]).tolist(),
        "utterance_mean": float(x.mean()),
        "utterance_cluster_ci95": np.quantile(weighted, [0.025, 0.975]).tolist(),
        "positive": int((x > 0).sum()),
        "speaker_positive": int((speaker_values > 0).sum()),
        "per_speaker": dict(zip(names, speaker_values.tolist(), strict=True)),
    }


def analyze():
    rows = [
        read(OUT / "scores" / f"{r['id']}.json")
        for r in read(OUT / "protocol.json")["rows"]
    ]
    result = {
        "guards": {},
        "max_replay_error": max(v["error"] for r in rows for v in r["replay"].values()),
    }
    for guard in ("0", "15", "20"):
        comparisons = {}
        for name, left, right in [
            ("native_pcm_T_N", "T_source", "N_source"),
            ("native_legacy_T_N", "T_legacy", "N_legacy"),
            ("mfa_native_M_N", "cross_M_M", "cross_N_N"),
            ("mfa_replacement_M_N", "cross_M_N", "cross_N_N"),
            ("mfa_evaluator_on_N", "cross_N_M", "cross_N_N"),
            ("mfa_generator_on_M", "cross_M_M", "cross_N_M"),
            ("direct_replacement_D_N", "cross_D_N", "cross_N_N"),
            ("mfa_vs_direct_replacement", "cross_M_N", "cross_D_N"),
        ]:
            rr = [r for r in rows if left in r["cells"] and right in r["cells"]]
            comparisons[name] = {
                m: cluster(
                    [
                        r["cells"][left][guard][m] - r["cells"][right][guard][m]
                        for r in rr
                    ],
                    [r["speaker"] for r in rr],
                )
                for m in ("C", "B", "D")
            }
        result["guards"][guard] = comparisons
    write(OUT / "analysis.json", result)
    print("analysis complete", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["freeze", "prepare", "score", "analyze"])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.stage == "score":
        score(a.device)
    else:
        globals()[a.stage]()
