"""Independent arithmetic, map and source audit for the window-clock experiment."""

from __future__ import annotations

import numpy as np

from scripts.experiments import check_tts_pcm_residual as check
from scripts.experiments.tts_window_clock import (
    BASE,
    PARENT,
    PHONE,
    ROOT,
    read,
    write,
)


def main():
    p = read(BASE / "protocol.json")
    check.check_hash(PHONE / "manifest.json", p["parent_manifest_sha256"])
    check.check_hash(PARENT / "protocol.json", p["parent_pcm_sha256"])
    check.check_hash(ROOT / "third_party/Wav2Lip/inference.py", p["inference_sha256"])
    check.check_hash(ROOT / "third_party/Wav2Lip/hparams.py", p["hparams_sha256"])
    rows = read(BASE / "manifest.json")["rows"]
    assert len(rows) == 38 and len({r["speaker"] for r in rows}) == 13
    calibration = read(BASE / "calibration_extension.json")
    assert not set(calibration["ids"]) & {r["id"] for r in rows}
    for arm, vals in calibration["offsets"].items():
        assert vals == [
            read(PARENT / "scores" / f"{sid}.json")["cells"][arm + "_source"]["20"][
                "offset"
            ]
            for sid in calibration["ids"]
        ]
        assert calibration["lags"][arm] == -np.median(vals) / 25
    cache = {}
    max_map_error = 0.0
    for row in rows:
        sid = row["id"]
        for info in row["videos"].values():
            check.check_hash(info["path"], info["sha256"])
        x, y = np.array(row["anchors"]["N"]), np.array(row["anchors"]["T"])
        for name, mapping in row["maps"].items():
            targets = np.arange(len(mapping["source"])) / 25 + mapping["L"]
            expected = []
            for t in targets:
                if t <= x[0]:
                    value = y[0]
                elif t >= x[-1]:
                    value = y[-1]
                else:
                    j = int(np.searchsorted(x, t, side="right") - 1)
                    value = y[j] + (y[j + 1] - y[j]) * (t - x[j]) / (x[j + 1] - x[j])
                expected.append(value - mapping.get("L_T", mapping["L"]))
            error = float(np.max(np.abs(np.array(expected) - mapping["source"])))
            assert error < 1e-12
            max_map_error = max(max_map_error, error)
        s, matrices = check.scorefile(BASE / "scores" / f"{sid}.json")
        vf = BASE / "scores" / f"{sid}_visual.npz"
        check.check_hash(vf, s["visual_sha256"])
        visuals = np.load(vf)
        af = PARENT / "features" / sid / "features.npz"
        check.check_hash(af, read(PARENT / "scores" / f"{sid}.json")["features_sha256"])
        audio = np.load(af)["a_N_source"]
        for name, matrix in matrices.items():
            check.distance(matrix, visuals[name], audio)
        prior = read(PHONE / "scores" / f"{sid}.json")
        for guard in ("0", "15", "20"):
            for current, old in [
                ("N_identity", "N_identity"),
                ("T_zero", "T_phone_boundaries"),
            ]:
                for metric in ("C", "B", "D"):
                    assert (
                        abs(
                            s["cells"][current][guard][metric]
                            - prior["cells"][old][guard][metric]
                        )
                        < 1e-5
                    )
        cache[sid] = (s, dict(matrices))
    analysis = read(BASE / "analysis.json")
    for guard, comparisons in analysis.items():
        g = int(guard)
        for name, summaries in comparisons.items():
            left, right = name.split("_minus_")
            for metric, summary in summaries.items():
                by = {}
                for r in rows:
                    s, matrices = cache[r["id"]]
                    if metric == "N_anchor_distance":
                        curves = {
                            k: (v[g:-g] if g else v).mean(0)
                            for k, v in matrices.items()
                        }
                        j = np.argmin(curves["N_identity"])
                        value = curves[left][j] - curves[right][j]
                    else:
                        value = (
                            s["cells"][left][guard][metric]
                            - s["cells"][right][guard][metric]
                        )
                    by.setdefault(r["speaker"], []).append(value)
                for speaker, values in by.items():
                    assert (
                        abs(np.mean(values) - summary["per_speaker"][speaker]) < 1e-12
                    )
    check.bootstrap_walk(analysis)
    write(
        BASE / "validation.json",
        {
            "status": "PASS",
            "counts": check.COUNTS,
            "errors": check.ERRORS,
            "map_max_error": max_map_error,
            "limits": "distance rows deterministically sampled; no independent second network or human truth; historical cohort",
        },
    )
    print(read(BASE / "validation.json"), flush=True)


if __name__ == "__main__":
    main()
