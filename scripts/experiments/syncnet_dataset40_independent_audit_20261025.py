"""Parent CPU audit. Does not import the producer, cache consumer or scorer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


_hashes = {}


def sha(path):
    path = Path(path)
    if path not in _hashes:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _hashes[path] = h.hexdigest()
    return _hashes[path]


def checked(path, expected):
    assert sha(path) == expected, (str(path), "SHA mismatch")


def distance_numpy(audio, video):
    import numpy as np
    n = len(audio)
    padded = np.pad(audio, ((15, 15), (0, 0)))
    result = np.empty((n, 31), dtype=np.float32)
    for k in range(31):
        difference = video - padded[k:k+n] + np.float32(1e-6)
        result[:, k] = np.sqrt(np.sum(difference*difference, axis=1, dtype=np.float32))
    return result


def verify_window(w, i, samples):
    lo, hi = max(0, 640*i-1), 640*i+3439
    assert w["row"] == i
    assert w["raw_samples_inclusive"] == [lo, hi]
    assert w["mfcc_columns_halfopen"] == [4*i, 4*i+20]
    assert w["video_frames_halfopen"] == [i, i+5]
    center = (lo+hi)/32000
    assert abs(w["score_audio_center_s"] - center) < 1e-12
    assert hi < samples
    return center


def audit(root):
    import numpy as np
    import python_speech_features
    import torch
    from scipy.io import wavfile
    from scipy.signal import correlate, correlation_lags

    start = time.time()
    torch.set_num_threads(2)
    registry_path = root / "dataset280_consumer_manifest.json"
    registry = load(registry_path)
    rows = registry["rows"]
    assert len(rows) == len({x["id"] for x in rows}) == 280
    assert len({x["speaker_id"] for x in rows}) == 40
    assert len({x["sample_id"] for x in rows}) == 40
    assert Counter(x["arm"] for x in rows) == {"N": 40, "L06": 120, "L17": 120}
    assert sum(x["original_cache_reused"] for x in rows) == 140
    groups = defaultdict(list)
    for r in rows:
        assert r["id"] == f'{r["sample_id"]}_{r["arm"]}_{r["draw"]}'
        assert r["sample_id"][6:11] == r["speaker_id"]
        groups[r["sample_id"]].append(r)
    expected = {("N", 0)} | {(a, d) for a in ("L06", "L17") for d in (1, 2, 3)}
    for group in groups.values():
        assert {(r["arm"], r["draw"]) for r in group} == expected
        assert len({r["text"] for r in group}) == 1

    checked(registry["dataset_input_manifest_path"], registry["dataset_input_manifest_sha256"])
    inputs = load(registry["dataset_input_manifest_path"])
    units = {r["id"]: r for r in inputs["units"]}
    assert set(units) == {r["id"] for r in rows}
    for p, h in inputs["source_pins"].items():
        checked(p, h)
    replay_root = Path(inputs["reuse_root"])
    protocol = load(replay_root / "protocol.json")
    for p, h in protocol["source_pins"].items():
        checked(p, h)
    weights_path = next(p for p in protocol["source_pins"] if p.endswith("syncnet_v2.model"))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    weight, bias = (state[k].numpy() for k in ("netfcaud.3.weight", "netfcaud.3.bias"))
    old = {r["id"]: r for r in load(replay_root / "input_manifest.json")["rows"]}
    baseline_path = root.parent / "syncnet_audio_dataset40_20261024_attempt02/original220_identity_baseline.json"
    checked(baseline_path, "860b9c868e785b4378c5b67c9ad9c748535834ac3060d14743d82a95de5737d2")
    baseline = {r["id"]: r for r in load(baseline_path)["rows"]}
    assert len(baseline) == 220
    clocks = {r["id"]: r for r in load(root / "dataset280_window_clock_audit.json")["rows"]}
    assert set(clocks) == set(units)
    fa_files, align, centers, replays, results = {}, {}, {}, {}, []
    for row in rows:
        key = row["id"]
        for field in ("sample_id", "speaker_id", "arm", "draw", "text", "score_path", "score_sha256"):
            assert row[field] == units[key][field], (key, field)
        for field in ("sequence", "scoring_audio", "replay", "canonical_audio", "video", "score", "activesd"):
            checked(row[field+"_path"], row[field+"_sha256"])
        if key in baseline:
            for field, value in baseline[key].items():
                if field in row:
                    assert row[field] == value, (key, field)
            assert row["canonical_audio_path"] == baseline[key]["audio_path"]
            assert row["canonical_audio_sha256"] == baseline[key]["audio_sha256"]
        if row["original_cache_reused"]:
            assert key in old
            assert row["sequence_path"] == str(replay_root / "items" / key / "sequence.npz")
            assert row["canonical_audio_sha256"] == old[key]["audio_sha256"]
        else:
            assert key not in old
        rec = load(row["replay_path"])
        replays[key] = rec
        assert (rec["id"], rec["source"], rec["arm"], rec["draw"]) == (key, row["sample_id"], row["arm"], row["draw"])
        score = load(row["score_path"])
        assert len(score["tracks"]) == 1
        track = score["tracks"][0]["metadata"]
        checked(track["crop_file"], track["crop_sha256"])
        assert rec["input_pins"]["crop_sha256"] == track["crop_sha256"]
        pretrack = Path(track["crop_file"]).parents[2] / "pyavi" / Path(track["crop_file"]).parent.name / "audio.wav"
        checked(pretrack, rec["input_pins"]["pretrack_audio_sha256"])
        assert rec["input_pins"]["audio_sha256"] == row["canonical_audio_sha256"]
        assert rec["input_pins"]["activesd_sha256"] == row["activesd_sha256"]
        original = np.asarray(pickle.loads(Path(row["activesd_path"]).read_bytes())[0])
        with np.load(row["sequence_path"], allow_pickle=False) as z:
            a, v, a512, d = (z[k] for k in ("audio1024", "video1024", "audio512", "distance"))
            n = len(a)
            assert a.shape == v.shape == (n, 1024) and a512.shape == (n, 512)
            assert np.array_equal(z["row_indices"], np.arange(n))
            assert d.shape == (n, 31) and all(np.isfinite(z[k]).all() for k in z.files)
            assert np.array_equal(original, d) and np.array_equal(z["original_distance"], original)
            curve_error = distance_numpy(a, v).astype(np.float64)-d
            maximum = float(np.abs(curve_error).max())
            rms = float(np.sqrt(np.mean(curve_error**2)))
            assert maximum <= protocol["tolerances"]["curve_max_abs"]
            assert rms <= protocol["tolerances"]["curve_rms"]
            projection = float(np.abs(a512 @ weight.T+bias-a).max())
            assert projection <= 1e-4
            mu = original.mean(axis=0)
            assert float(np.median(mu)-mu.min()) == row["score_float32"]
            assert int(15-mu.argmin()) == rec["original"]["offset"]
            sr, pcm = wavfile.read(row["scoring_audio_path"])
            assert sr == 16000 and pcm.ndim == 1 and pcm.dtype == np.int16
            assert np.array_equal(python_speech_features.mfcc(pcm, sr).T.astype(np.float32), z["mfcc"])
        assert n == row["rows"] == rec["windows"] == min(rec["frame_count"], len(pcm)//640)-5
        assert rec["audio_samples"] == len(pcm)
        windows = clocks[key]["rows"]
        assert len(windows) == n
        centers[key] = [verify_window(w, i, len(pcm)) for i, w in enumerate(windows)]
        sr0, canonical = wavfile.read(row["canonical_audio_path"])
        assert sr0 == 16000 and canonical.ndim == 1
        x, y = pcm.astype(np.float64), canonical.astype(np.float64)
        x -= x.mean(); y -= y.mean()
        corr = correlate(x, y, mode="full", method="fft")
        lags = correlation_lags(len(x), len(y))
        region = np.abs(lags) <= 4000
        lag = int(lags[region][corr[region].argmax()])
        assert lag == rec["clock"]["lag_samples_scoring_minus_canonical"], (key, lag, rec["clock"])
        assert abs(clocks[key]["delay_s"]-lag/16000) < 1e-12
        fa = row["FA"]
        checked(fa["path"], fa["sha256"])
        if fa["path"] not in fa_files:
            obj = load(fa["path"])
            fa_files[fa["path"]] = {r["id"]: r for r in obj.get("audio", obj.get("rows", []))}
        aligned = fa_files[fa["path"]][key]
        assert aligned["audio_sha256"] == row["canonical_audio_sha256"]
        assert aligned["text"] == row["text"]
        assert len(aligned["tokens"]) == len(aligned["timestamps_ms"])
        previous = 0
        for lo, hi in aligned["timestamps_ms"]:
            assert previous <= lo < hi <= len(canonical)/16
            previous = hi
        align[key] = aligned
        results.append({"id": key, "rows": n, "saved_curve_exact": True,
                        "numpy_distance_max_abs": maximum, "numpy_distance_rms": rms,
                        "projection_max_abs": projection, "mfcc_exact": True,
                        "canonical_C_exact": True, "clock_lag_independent_samples": lag})

    patchdata = load(root / "dataset280_patch_index.json")
    nominal, robust, seen = defaultdict(set), defaultdict(set), set()
    variants = 0
    for p in patchdata["patches"]:
        key, index = p["id"], p["character_index"]
        assert (key, index) not in seen
        seen.add((key, index))
        a = align[key]
        assert p["character"] == a["tokens"][index]
        interval = a["timestamps_ms"][index]
        assert p["canonical_interval_ms"] == interval
        assert p["FA_truth"] == "UNVERIFIED"
        assert len(p["variants"]) == 3
        all_good = True
        for shift, variant in zip((-.04, 0., .04), p["variants"]):
            anchor = sum(interval)/2000 + replays[key]["clock"]["lag_samples_scoring_minus_canonical"]/16000 + shift
            index0 = min(range(len(centers[key])), key=lambda i: abs(centers[key][i]-anchor))
            good = 5 <= index0 < len(centers[key])-5
            assert variant["shift_s"] == shift
            assert variant["nearest_row"] == index0
            assert abs(variant["mapped_anchor_s"]-anchor) < 1e-12
            assert abs(variant["nearest_center_s"]-centers[key][index0]) < 1e-12
            assert abs(variant["center_error_ms"]-(centers[key][index0]-anchor)*1000) < 1e-8
            assert variant["support_ok"] == good
            assert variant["patch_rows"] == (list(range(index0-5, index0+6)) if good else None)
            assert variant["patch_raw_samples_inclusive"] == ([max(0, (index0-5)*640-1), (index0+5)*640+3439] if good else None)
            if good and shift == 0:
                nominal[key].add(index)
            all_good &= good
            variants += 1
        if all_good:
            robust[key].add(index)
    assert len(seen) == sum(len(a["tokens"]) for a in align.values())
    bad = []
    for cover in patchdata["coverage"]:
        key = cover["id"]
        coverage = len(nominal[key])/len(align[key]["tokens"])
        good = len(nominal[key]) >= 5 and coverage >= .8
        assert cover["valid_nominal_anchors"] == len(nominal[key])
        assert abs(cover["nominal_coverage"]-coverage) < 1e-12
        assert cover["temporary_measurability_gate"] == good
        if not good:
            bad.append({"id": key, "coverage": coverage})
    pair_ok = 0
    assert len(patchdata["pair_coverage"]) == 600
    for pair in patchdata["pair_coverage"]:
        a, b = pair["a_id"], pair["b_id"]
        assert align[a]["tokens"] == align[b]["tokens"]
        common = sorted(nominal[a] & nominal[b])
        assert common == pair["common_character_indices"]
        good = len(common) >= 5 and len(common)/len(align[a]["tokens"]) >= .8
        assert pair["temporary_measurability_gate"] == good
        pair_ok += good
    source_ok = sum(all(r["id"] not in {x["id"] for x in bad} for r in g) for g in groups.values())
    assert len(results) == 280 and sum(r["rows"] for r in results) == 29485
    return {"schema": "parent_independent_dataset40_audit_v1", "status": "PASS_TECHNICAL_REPLAY_AND_INDEX_ARITHMETIC",
            "training": False, "GPU_forward": False, "registry_sha256": sha(registry_path),
            "code_sha256": sha(__file__), "clips": 280, "sources": 40, "official_rows": 29485,
            "old220_identity_pass": True, "original140_reuse_pass": True,
            "saved_curves_exact": True, "canonical_C_exact": True,
            "max_numpy_distance_error": max(x["numpy_distance_max_abs"] for x in results),
            "max_projection_error": max(x["projection_max_abs"] for x in results),
            "all_MFCC_exact": True, "all_clock_lags_independently_exact": True,
            "anchors": len(seen), "variants": variants, "local_clip_gate_pass": 280-len(bad),
            "all7_clip_source_gate_pass": source_ok, "pair_gate_pass": pair_ok, "local_clip_failures": bad,
            "FA_truth": "UNVERIFIED", "local615ms_sufficiency": "UNVERIFIED",
            "independent_convolution_or_face_detection_rerun": False,
            "hashes_read": len(_hashes), "elapsed_s": time.time()-start, "items": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists(), "Choose a new immutable audit output"
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False))
