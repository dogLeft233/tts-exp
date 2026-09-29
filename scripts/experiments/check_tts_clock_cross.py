#!/usr/bin/env python3
"""Independently recompute saved scores and key paired effects from matrices."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/tts_clock_cross_20260926"


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    errors = []
    cells = 0
    worst = 0.0
    files = 0
    for root, sub, guard in [
        (RUN, "scores", 15),
        (RUN, "delay/scores", 20),
        (RUN, "codec/scores", 20),
        (RUN / "real12", "codec/scores", 20),
    ]:
        protocol = load(
            root
            / (
                "protocol.json"
                if sub == "scores"
                else sub.replace("scores", "protocol.json")
            )
        )
        for row in protocol["rows"]:
            path = root / sub / f"{row['id']}_{row['arm']}.json"
            item = load(path)
            archive = (
                Path(item["matrix_path"])
                if "matrix_path" in item
                else path.with_suffix(".npz")
            )
            assert digest(archive) == item["matrix_sha256"], archive
            matrices = np.load(archive)
            counts = {m.shape[0] for m in matrices.values()}
            assert len(counts) == 1, (path, counts)
            for mode, m in matrices.items():
                assert m.ndim == 2 and m.shape[1] == 31 and np.isfinite(m).all()
                supports = (
                    [("full", m), ("interior", m[15:-15])]
                    if sub == "scores"
                    else [("interior", m[guard:-guard])]
                )
                for support, view in supports:
                    curve = np.sum(view, axis=0) / len(view)
                    best = int(np.argmin(curve))
                    d = float(curve[best])
                    b = float(sorted(curve)[15])
                    c = b - d
                    ref = (
                        item["cells"][mode][support]
                        if sub == "scores"
                        else item["cells"][mode]
                    )
                    err = max(abs(ref["C"] - c), abs(ref["D"] - d), abs(ref["B"] - b))
                    worst = max(worst, err)
                    if (
                        err > 1e-10
                        or ref["offset"] != 15 - best
                        or ref["rows"] != len(view)
                    ):
                        errors.append(str(path))
                    if best in [0, 30]:
                        errors.append(f"boundary peak:{path}:{mode}:{support}")
                    cells += 1
            for key in ["source_pcm", "visual", "crop"]:
                if key in row:
                    assert digest(row[key]) == row[key + "_sha256"], row[key]
                    files += 1
            for info in row.get("modes", {}).values():
                assert digest(info["path"]) == info["sha256"], info["path"]
                files += 1
    # Recompute primary effects and intervals using raw matrices, not score JSON.
    ids = sorted({r["id"] for r in load(RUN / "protocol.json")["rows"]})

    def score(archive, mode, guard):
        m = np.load(archive)[mode][guard:-guard]
        curve = np.mean(m, axis=0)
        return float(np.median(curve) - np.min(curve))

    def ci(x):
        rng = np.random.default_rng(20260926)
        boot = np.asarray(x)[rng.integers(len(x), size=(20000, len(x)))].mean(axis=1)
        return np.quantile(boot, [0.025, 0.975])

    for arm in ["N", "T"]:
        values = []
        for i in ids:
            p = RUN / "codec/scores" / f"{i}_{arm}.npz"
            values.append(
                score(p, "avi24000_default_aligned", 20)
                - score(p, "avi16000_default_aligned", 20)
            )
        ref = load(RUN / "codec/analysis.json")["within_audio_effects"][
            f"{arm}_avi24000_default_aligned_minus_avi16000_default_aligned"
        ]
        assert np.allclose(
            [np.mean(values), *ci(values)], [ref["mean"], *ref["ci95"]], atol=1e-10
        )
    result = {
        "status": "PASS" if not errors else "FAIL",
        "score_checks": cells,
        "input_hash_checks": files,
        "max_score_error": worst,
        "errors": errors,
        "independent_primary_bootstrap": True,
        "checker_sha256": digest(__file__),
    }
    (RUN / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    check()
