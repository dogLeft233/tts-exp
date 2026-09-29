"""Independent artifact/numerical audit; does not call the runner's scoring helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/syncnet_codec_mechanism_20260926"


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bootstrap(x):
    rng = np.random.default_rng(20260926)
    x = np.array(x, dtype=float)
    sample = x[rng.integers(0, len(x), (20000, len(x)))].mean(1)
    return float(x.mean()), np.quantile(sample, [0.025, 0.975])


def main():
    protocol = read(OUT / "protocol.json")
    assert len(protocol["rows"]) == 106
    checks, hashes, distances = 0, 0, 0
    max_score_error, max_distance_error = 0.0, 0.0
    grouped = {"natural": [], "tts": [], "real": []}
    for row in protocol["rows"]:
        key = row["key"]
        score = read(OUT / "scores" / f"{key}.json")
        receipt_path = OUT / "audio" / key / "receipt.json"
        receipt = read(receipt_path)
        assert digest(receipt_path) == score["receipt_sha256"]
        hashes += 1
        for field in ("source_pcm", "visual"):
            assert digest(row[field]) == row[field + "_sha256"]
            hashes += 1
        for mode, info in receipt["modes"].items():
            assert digest(info["path"]) == info["sha256"]
            hashes += 1
            if "encoded_sha256" in info:
                assert (
                    digest(OUT / "audio" / key / f"{mode}.avi")
                    == info["encoded_sha256"]
                )
                hashes += 1
        features_path = OUT / "features" / f"{key}.npz"
        assert digest(features_path) == score["features_sha256"]
        hashes += 1
        af = np.load(features_path)
        vf = np.load(row["visual"])[: score["common_rows"]]
        assert len(af.files) == 18
        for folder, expected_modes in [
            (OUT, 72),
            (OUT / "mel_patch", 7),
            (OUT / "band_swap", 5),
        ]:
            current = read(folder / "scores" / f"{key}.json")
            if folder.name == "band_swap":
                for info in current["paths"].values():
                    assert digest(info["path"]) == info["sha256"]
                    hashes += 1
                assert max(current["replay_errors"].values()) < 1e-4
            matrix_path = folder / "scores" / f"{key}.npz"
            assert digest(matrix_path) == current.get(
                "matrices_sha256", current.get("matrix_sha256")
            )
            hashes += 1
            mm = np.load(matrix_path)
            assert len(mm.files) == expected_modes
            for mode in mm.files:
                mat = mm[mode]
                assert np.isfinite(mat).all() and mat.shape == (
                    score["common_rows"],
                    31,
                )
                curve = np.mean(mat[20:-20], axis=0)
                minimum, med = np.min(curve), np.median(curve)
                if "__" in mode:
                    stem, label = mode.split("__")
                    reported = current["diagnostic"][stem][label]
                else:
                    reported = current["cells"][mode]
                for name, value in [
                    ("C", med - minimum),
                    ("D", minimum),
                    ("B", med),
                    ("fixed_D", curve[score["pcm_lag_index"]]),
                ]:
                    err = abs(float(value) - reported[name])
                    max_score_error = max(max_score_error, err)
                    assert err < 1e-10
                assert reported["offset"] == 15 - int(curve.argmin())
                assert abs(reported["offset"]) < 15
                checks += 1
                # Independent float32 Euclidean reconstruction of five interior rows per raw condition.
                if folder == OUT and "__" not in mode:
                    a = af[mode]
                    for t in sorted({20, len(mat) // 2, len(mat) - 21}):
                        d = np.sqrt(
                            np.sum(
                                (vf[t][None, :] - a[t - 15 : t + 16] + np.float32(1e-6))
                                ** 2,
                                axis=1,
                            )
                        )
                        err = float(np.max(np.abs(d - mat[t])))
                        max_distance_error = max(max_distance_error, err)
                        assert err < 1e-4
                        distances += 31
        patch = read(OUT / "mel_patch/scores" / f"{key}.json")
        assert max(patch["replay_errors"].values()) < 1e-4
        group = {"N": "natural", "T": "tts", "R": "real"}[row["arm"]]
        grouped[group].append(score)
    analysis = read(OUT / "analysis.json")
    bootstrap_checks = 0
    for group, rr in grouped.items():
        for label, r in analysis["groups"][group]["contrasts"].items():
            left, right = label.split("__minus__")
            mean, ci = bootstrap(
                [s["cells"][left]["C"] - s["cells"][right]["C"] for s in rr]
            )
            assert abs(mean - r["C"]["mean"]) < 1e-10
            assert np.max(np.abs(ci - r["C"]["ci95"])) < 1e-10
            bootstrap_checks += 1
    for stage in ("mel_patch", "band_swap"):
        analysis = read(OUT / stage / "analysis.json")
        for group, arm in [("natural", "N"), ("tts", "T"), ("real", "R")]:
            rr = [
                read(OUT / stage / "scores" / f"{r['key']}.json")
                for r in protocol["rows"]
                if r["arm"] == arm
            ]
            for label, r in analysis["groups"][group].items():
                left, right = label.split("_minus_")
                mean, ci = bootstrap(
                    [s["cells"][left]["C"] - s["cells"][right]["C"] for s in rr]
                )
                assert abs(mean - r["mean"]) < 1e-10
                assert np.max(np.abs(ci - r["ci95"])) < 1e-10
                bootstrap_checks += 1
    result = {
        "status": "PASS",
        "score_recomputations": checks,
        "sha256_checks": hashes,
        "independent_distance_elements": distances,
        "independent_bootstraps": bootstrap_checks,
        "max_score_error": max_score_error,
        "max_distance_error": max_distance_error,
        "scope": "all factorial, mel-patch and waveform-swap scores/hashes; all raw C-contrast bootstrap95; sampled interior distances for every factorial physical audio condition",
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
