"""Independent CTC, media provenance and grouped-statistic verification."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

BASE = Path(__file__).resolve().parents[2] / "runs/tts_independent_visual_20260926"
COUNTS = {"hashes": 0, "ctc_values": 0, "logp_rows": 0, "summaries": 0}
MAXIMUM = {"ctc": 0.0, "logp_normalization": 0.0, "bootstrap": 0.0}


def read(path):
    return json.loads(Path(path).read_text())


def check(path, digest):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == digest, str(path)
    COUNTS["hashes"] += 1


def independent_ctc(logp, target):
    # Scalar log-space dynamic program, independent of torch.ctc_loss and timing posterior code.
    x = np.asarray(logp, dtype=np.float64)
    states = [0]
    for token in target:
        states.extend([token, 0])
    prev = np.full(len(states), -np.inf)
    prev[0] = x[0, 0]
    prev[1] = x[0, states[1]]
    for frame_index in range(1, len(x)):
        nxt = np.full_like(prev, -np.inf)
        for j, token in enumerate(states):
            total = prev[j]
            if j > 0:
                total = np.logaddexp(total, prev[j - 1])
            if j > 1 and token != 0 and token != states[j - 2]:
                total = np.logaddexp(total, prev[j - 2])
            nxt[j] = total + x[frame_index, token]
        prev = nxt
    return float(-np.logaddexp(prev[-1], prev[-2]) / len(target))


def bootstrap_walk(obj):
    if isinstance(obj, dict):
        if "per_speaker" in obj:
            vals = np.array([obj["per_speaker"][k] for k in sorted(obj["per_speaker"])])
            ix = np.random.default_rng(20260926).integers(
                len(vals), size=(20000, len(vals))
            )
            dist = vals[ix].mean(1)
            for key, expect in [
                ("speaker_mean", vals.mean()),
                ("speaker_ci95", np.quantile(dist, [0.025, 0.975])),
                ("speaker_ci99", np.quantile(dist, [0.005, 0.995])),
            ]:
                error = float(np.max(np.abs(np.asarray(obj[key]) - expect)))
                assert error < 1e-12
                MAXIMUM["bootstrap"] = max(MAXIMUM["bootstrap"], error)
            COUNTS["summaries"] += 1
        else:
            for value in obj.values():
                bootstrap_walk(value)
    elif isinstance(obj, list):
        for value in obj:
            bootstrap_walk(value)


def collect():
    """Compute independent per-view likelihoods as finished GPU artifacts arrive."""
    protocol = read(BASE / "protocol.json")
    folder = BASE / "independent_ctc"
    folder.mkdir(parents=True, exist_ok=True)
    for row in protocol["rows"]:
        sid = row["id"]
        destination = folder / f"{sid}.json"
        if destination.exists():
            continue
        receipt_path = BASE / "receipts" / f"{sid}.json"
        while not receipt_path.exists():
            time.sleep(1)
        receipt = read(receipt_path)
        targets = [row["target"]] + [r["target"] for r in row["decoys"]]
        cells = {}
        for key, digest in receipt["feature_hashes"].items():
            file = BASE / "features" / sid / f"{key}.npz"
            check(file, digest)
            logp = np.load(file)["logp"]
            cells[key] = {
                "feature_sha256": digest,
                "losses": [independent_ctc(logp, target) for target in targets],
            }
        destination.write_text(
            json.dumps({"id": sid, "targets": targets, "cells": cells}, indent=2) + "\n"
        )
        print("independent CTC collected", sid, flush=True)


def main():
    p = read(BASE / "protocol.json")
    scores = {r["id"]: r for r in read(BASE / "scores.json")}
    assert len(p["rows"]) == 68 and len(p["missing"]) == 6
    assert sum(r["split"] == "calibration" for r in p["rows"]) == 26
    assert sum(r["split"] == "evaluation" for r in p["rows"]) == 42
    for name in ("model", "model_json", "config", "text_source"):
        check(p[name], p[name + "_sha256"])
    original = {
        r["id"]: r
        for r in read(BASE.parent / "tts_pcm_residual_20260926/protocol.json")["rows"]
    }
    text_records = read(p["text_source"])["records"]
    for row in p["rows"]:
        sid = row["id"]
        for arm, label in [("N", "natural"), ("T", "tts")]:
            assert (
                text_records[sid][label]["audio_sha256"]
                == original[sid]["cells"][arm]["audio_sha256"]
            )
        receipt = read(BASE / "receipts" / f"{sid}.json")
        for arm, cell in row["cells"].items():
            check(cell["video"], cell["sha256"])
            qc = receipt["qc"][arm]
            assert (
                qc["frame_digest_equal"]
                and qc["pts_equal"]
                and qc["video_only_stream"]["audio_stream_count"] == 0
            )
            cache = BASE / "mouths" / sid / f"{arm}.npz"
            check(cache, qc["cache_sha256"])
            x = np.load(cache)["x"]
            assert list(x.shape) == qc["tensor_shape"]
            assert (
                hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                == qc["tensor_sha256"]
            )
        for errors in receipt["repeat"].values():
            assert max(errors.values()) <= 1e-5
        if sid not in scores:
            assert not receipt["eligible"]
            continue
        oracle_path = BASE / "independent_ctc" / f"{sid}.json"
        oracle = read(oracle_path) if oracle_path.exists() else None
        for key, digest in receipt["feature_hashes"].items():
            file = BASE / "features" / sid / f"{key}.npz"
            check(file, digest)
            f = np.load(file)
            lp = f["logp"]
            error = float(np.max(np.abs(logsumexp(lp.astype(float), axis=1))))
            MAXIMUM["logp_normalization"] = max(MAXIMUM["logp_normalization"], error)
            assert error < 1e-5
            COUNTS["logp_rows"] += len(lp)
            targets = [row["target"]] + [r["target"] for r in row["decoys"]]
            expected = scores[sid]["cells"][key]
            if oracle is None:
                losses = [independent_ctc(lp, target) for target in targets]
            else:
                assert oracle["targets"] == targets
                assert oracle["cells"][key]["feature_sha256"] == digest
                losses = oracle["cells"][key]["losses"]
            err = float(np.max(np.abs(np.array(losses) - expected["losses"])))
            MAXIMUM["ctc"] = max(MAXIMUM["ctc"], err)
            assert err < 1e-5, (sid, key, err)
            assert abs(np.mean(losses[1:]) - losses[0] - expected["M"]) < 1e-5
            COUNTS["ctc_values"] += len(targets)
        print("verified visual", sid, flush=True)
    # Recreate paired speaker means from scored videos rather than trusting aggregate records.
    analysis = read(BASE / "analysis.json")
    for split, comparisons in analysis["splits"].items():
        group = [r for r in scores.values() if split == "all" or r["split"] == split]
        for name, result in comparisons.items():
            left, right, metric = name.split("_", 2)
            by = {}
            for r in group:
                if left not in r["arms"] or right not in r["arms"]:
                    continue
                by.setdefault(r["speaker"], []).append(
                    r["arms"][left][metric] - r["arms"][right][metric]
                )
            for speaker, values in by.items():
                assert abs(np.mean(values) - result["per_speaker"][speaker]) < 1e-12
    bootstrap_walk(analysis)
    timing = BASE / "timing/analysis.json"
    if timing.exists():
        bootstrap_walk(read(timing))
        for row in read(BASE / "timing/calibration_scores.json"):
            sid = row["id"]
            receipt = read(BASE / "timing/features" / sid / "receipt.json")
            for mode, digest in receipt["hashes"].items():
                check(BASE / "timing/features" / sid / f"{mode}.npz", digest)
                c = row["controls"][mode]
                observed = np.array(c["observed_delta"])
                expected = np.array(c["expected_delta"])
                assert abs(np.mean(np.abs(observed - expected)) - c["MAE"]) < 1e-12
                assert abs(np.mean(np.abs(expected)) - c["null_MAE"]) < 1e-12
        for row in read(BASE / "timing/scores.json"):
            check(row["mfa"], row["mfa_sha256"])
            target = np.array(row["target_centers"])
            for cell in row["cells"].values():
                diff = np.array(cell["visual_centers"]) - target
                assert (
                    abs(
                        np.mean(np.abs(diff - np.median(diff))) * 1000
                        - cell["residual_MAE_ms"]
                    )
                    < 1e-9
                )
    raw = BASE / "raw_transfer"
    if (raw / "analysis.json").exists():
        from scripts.experiments import check_tts_pcm_residual as pcm
        from scripts.experiments.tts_pcm_timing import decode

        bootstrap_walk(read(raw / "analysis.json"))
        for guard, splits in read(raw / "analysis.json")["guards"].items():
            for split, comparisons in splits.items():
                for name, summary in comparisons.items():
                    left, right = name.split("_minus_")
                    by = {}
                    for r in read(raw / "manifest.json")["rows"]:
                        if split != "all" and r["split"] != split:
                            continue
                        s = read(raw / "scores" / f"{r['id']}.json")
                        by.setdefault(r["speaker"], []).append(
                            s["cells"][left][guard]["C"] - s["cells"][right][guard]["C"]
                        )
                    for speaker, values in by.items():
                        assert (
                            abs(np.mean(values) - summary["per_speaker"][speaker])
                            < 1e-12
                        )
        for row in read(raw / "manifest.json")["rows"]:
            sid = row["id"]
            for info in row["videos"].values():
                check(info["path"], info["sha256"])
            assert np.array_equal(
                decode(row["videos"]["N_identity"]["path"]),
                decode(original[sid]["cells"]["N"]["crop"]),
            ), (sid, "identity pixels changed")
            for axis in ("mfa_x", "mfa_y"):
                assert np.all(np.diff(row[axis]) > 0)
            scored, matrices = pcm.scorefile(raw / "scores" / f"{sid}.json")
            vf = raw / "scores" / f"{sid}_visual.npz"
            check(vf, scored["visual_sha256"])
            visual = np.load(vf)
            source = BASE.parent / "tts_pcm_residual_20260926"
            af = source / "features" / sid / "features.npz"
            check(af, read(source / "scores" / f"{sid}.json")["features_sha256"])
            audio = np.load(af)["a_N_source"]
            assert (
                scored["natural_audio_sha256"]
                == original[sid]["cells"]["N"]["audio_sha256"]
            )
            for key, matrix in matrices.items():
                pcm.distance(matrix, visual[key], audio)
        COUNTS["raw_transfer"] = pcm.COUNTS
        MAXIMUM["raw_transfer"] = pcm.ERRORS
    for name in ("roundtrip", "phone_transfer"):
        folder = BASE / name
        if not (folder / "analysis.json").exists():
            continue
        bootstrap_walk(read(folder / "analysis.json"))
        for row in read(folder / "manifest.json")["rows"]:
            sid = row["id"]
            for info in (
                list(row["videos"].values())
                + list(row.get("intermediate", {}).values())
                + list(row.get("textgrids", {}).values())
            ):
                check(info["path"], info["sha256"])
            for mapping in row.get("maps", {}).values():
                assert np.all(np.diff(mapping["N"]) > 0) and np.all(
                    np.diff(mapping["T"]) >= 0
                )
            s, matrices = pcm.scorefile(folder / "scores" / f"{sid}.json")
            vf = folder / "scores" / f"{sid}_visual.npz"
            check(vf, s["visual_sha256"])
            v = np.load(vf)
            audio = np.load(
                BASE.parent
                / "tts_pcm_residual_20260926/features"
                / sid
                / "features.npz"
            )["a_N_source"]
            original_score = read(raw / "scores" / f"{sid}.json")
            for guard in ("0", "15", "20"):
                assert (
                    abs(
                        s["cells"]["N_identity"][guard]["C"]
                        - original_score["cells"]["N_identity"][guard]["C"]
                    )
                    < 1e-5
                )
            for key, matrix in matrices.items():
                pcm.distance(matrix, v[key], audio)
    if (BASE / "timing_candidates/scores.json").exists():
        for row in read(BASE / "timing_candidates/scores.json"):
            for c in row["controls"].values():
                check(c["path"], c["sha256"])
                observed, expected = (
                    np.array(c["observed_delta"]),
                    np.array(c["expected_delta"]),
                )
                ratio = np.mean(np.abs(observed - expected)) / np.mean(np.abs(expected))
                assert abs(ratio - c["error_ratio"]) < 1e-12
    if (BASE / "supplement.json").exists():
        bootstrap_walk(read(BASE / "supplement.json"))
    result = {
        "status": "PASS",
        "counts": COUNTS,
        "maximum_errors": MAXIMUM,
        "limits": "CTC probability arithmetic independently recomputed for every target/decoy/view; timing center estimator uses separately checked forward/backward code, no independent human timing truth.",
    }
    (BASE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    if parser.parse_args().collect:
        collect()
    else:
        main()
