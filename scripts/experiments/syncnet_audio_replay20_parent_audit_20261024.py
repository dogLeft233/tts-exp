"""Independent CPU audit of frozen SyncNet replay; imports no experiment runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(root: Path):
    start = time.time()
    torch.set_num_threads(2)
    protocol = read(root / "protocol.json")
    # Verified freeze() binds this legacy field to the source manifest, not the
    # enriched replay manifest. Preserve the failed initial interpretation receipt.
    source_manifest = root.parent / "syncnet_preference_probe_20261024_attempt01/input_manifest.json"
    assert digest(source_manifest) == protocol["input_manifest_sha256"]
    rows = read(root / "input_manifest.json")["rows"]
    original_rows = read(source_manifest)["rows"]
    assert len(original_rows) == len(rows)
    for source_row, enriched_row in zip(original_rows, rows):
        assert all(enriched_row[k] == v for k, v in source_row.items()), source_row["id"]
        assert digest(enriched_row["score_path"]) == enriched_row["score_sha256"]
        official_score = read(enriched_row["score_path"])
        original_track = official_score["tracks"][0]["metadata"]
        assert enriched_row["track"] == original_track
        assert enriched_row["crop_path"] == original_track["crop_file"]
        assert enriched_row["crop_sha256"] == original_track["crop_sha256"]
    assert len(rows) == 140 and len({x["speaker_id"] for x in rows}) == 20
    assert len({x["id"] for x in rows}) == 140
    for path, expected in protocol["source_pins"].items():
        assert digest(path) == expected, path
    model_path = next(p for p in protocol["source_pins"] if p.endswith("syncnet_v2.model"))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    weight = state["netfcaud.3.weight"].numpy()
    bias = state["netfcaud.3.bias"].numpy()
    checks = []
    for row in rows:
        cell = root / "items" / row["id"]
        rec = read(cell / "replay.json")
        assert digest(cell / "sequence.npz") == rec["sequence_sha256"]
        assert digest(cell / "scoring_audio.wav") == rec["scoring_audio_sha256"]
        for pathkey, hashkey in [("activesd_path", "activesd_sha256"),
                                  ("audio_path", "audio_sha256"),
                                  ("crop_path", "crop_sha256"),
                                  ("pretrack_audio_path", "pretrack_audio_sha256")]:
            assert digest(row[pathkey]) == row[hashkey], (row["id"], pathkey)
        with np.load(cell / "sequence.npz", allow_pickle=False) as z:
            a = z["audio1024"]
            v = z["video1024"]
            a512 = z["audio512"]
            curve = z["distance"]
            saved_original = z["original_distance"]
            mfcc = z["mfcc"]
            indices = z["row_indices"]
            n = len(a)
            assert a.shape == v.shape == (n, 1024)
            assert a512.shape == (n, 512)
            assert curve.shape == (n, 31)
            assert np.array_equal(indices, np.arange(n))
            assert all(np.isfinite(arr).all() for arr in (a, v, a512, curve, mfcc))
            original = np.asarray(pickle.loads(Path(row["activesd_path"]).read_bytes())[0])
            assert np.array_equal(original, saved_original)
            assert np.array_equal(curve, original), row["id"]
            mu = original.mean(axis=0)
            c = float(np.median(mu) - mu.min())
            assert c == row["score_float32"], (row["id"], c, row["score_float32"])
            assert n == row["score_windows"] == rec["windows"]
            assert int(15 - mu.argmin()) == rec["original"]["offset"]
            # Independent NumPy implementation of PyTorch PairwiseDistance(p=2, eps=1e-6).
            padded = np.pad(a, ((15, 15), (0, 0)), mode="constant")
            rebuilt = np.empty_like(curve)
            for lag_col in range(31):
                diff = v - padded[lag_col:lag_col+n] + np.float32(1e-6)
                rebuilt[:, lag_col] = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32))
            err = rebuilt.astype(np.float64) - curve
            distance_max = float(np.abs(err).max())
            distance_rms = float(np.sqrt(np.mean(err * err)))
            assert distance_max <= protocol["tolerances"]["curve_max_abs"]
            assert distance_rms <= protocol["tolerances"]["curve_rms"]
            projection_max = float(np.abs(a512 @ weight.T + bias - a).max())
            assert projection_max <= 1e-4, (row["id"], projection_max)
            sr, pcm = wavfile.read(cell / "scoring_audio.wav")
            assert sr == 16000 and pcm.ndim == 1 and pcm.dtype == np.int16
            mfcc_rebuilt = python_speech_features.mfcc(pcm, sr).T.astype(np.float32)
            assert np.array_equal(mfcc_rebuilt, mfcc), row["id"]
            assert n == min(rec["frame_count"], len(pcm) // 640) - 5
            assert rec["audio_samples"] == len(pcm)
            for k, idx in [("first_support", 0), ("last_support", n-1)]:
                support = rec[k]
                lo, hi = max(0, 640 * idx - 1), 640 * idx + 3439
                assert support["raw_samples_inclusive"] == [lo, hi]
                assert support["mfcc_columns"] == [4 * idx, 4 * idx + 20]
                assert abs(support["center_s"] - (lo+hi)/32000) < 1e-12
                assert hi < len(pcm)
            checks.append({"id": row["id"], "windows": n,
                           "numpy_distance_max_abs": distance_max,
                           "numpy_distance_rms": distance_rms,
                           "audio512_projection_max_abs": projection_max,
                           "canonical_score_exact": True,
                           "official_torch_vs_canonical_c_abs": abs(rec["original"]["C"] - c),
                           "saved_original_and_replay_curves_exact": True,
                           "mfcc_exact": True})
    return {"schema": "independent_syncnet_audio_replay20_parent_audit_v1",
            "status": "PASS", "training": False, "gpu_forward": False,
            "scope": "All140 original curve hashes, cached512->final projection, NumPy distances, PCM->MFCC, canonical labels and endpoint window arithmetic; no independent convolution or face detection rerun; FA/patch separately audited",
            "protocol_sha256": digest(root / "protocol.json"),
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": digest(source_manifest),
            "enriched_manifest_sha256": digest(root / "input_manifest.json"),
            "all_source_rows_preserved_exact": True,
            "code_sha256": digest(__file__), "items": checks,
            "count": len(checks), "windows": sum(x["windows"] for x in checks),
            "max_numpy_distance_error": max(x["numpy_distance_max_abs"] for x in checks),
            "max_projection_error": max(x["audio512_projection_max_abs"] for x in checks),
            "max_reduction_c_difference": max(x["official_torch_vs_canonical_c_abs"] for x in checks),
            "elapsed_s": time.time() - start}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists(), "Audit output is immutable; choose a fresh attempt."
    result = audit(args.root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False))
