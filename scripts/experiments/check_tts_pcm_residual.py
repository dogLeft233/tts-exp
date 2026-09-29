"""Independent arithmetic, provenance and speaker-bootstrap audit of PCM experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/tts_pcm_residual_20260926"
COUNTS = {"hashes": 0, "scores": 0, "distance_elements": 0, "bootstrap_summaries": 0}
ERRORS = {"score": 0.0, "distance": 0.0, "bootstrap": 0.0}


def read(path):
    return json.loads(Path(path).read_text())


def check_hash(path, expected):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == expected, str(path)
    COUNTS["hashes"] += 1


def distance(matrix, video, audio):
    n = len(matrix)
    video = video[:n].astype(np.float32)
    audio = audio[:n].astype(np.float32)
    padded = np.pad(audio, ((15, 15), (0, 0)))
    # Five deterministic interior/boundary rows, all 31 shifts, every stored cell.
    rows = sorted({0, min(15, n - 1), n // 2, max(0, n - 16), n - 1})
    for i in rows:
        difference = video[i] - padded[i : i + 31] + np.float32(1e-6)
        values = np.sqrt(np.sum(difference * difference, axis=1, dtype=np.float32))
        error = float(np.max(np.abs(values - matrix[i])))
        assert error < 1e-5, (i, error)
        ERRORS["distance"] = max(ERRORS["distance"], error)
        COUNTS["distance_elements"] += 31


def scorefile(path):
    s = read(path)
    matpath = path.with_suffix(".npz")
    check_hash(matpath, s.get("matrix_sha256", s.get("matrices_sha256")))
    matrices = np.load(matpath)
    for key, guards in s["cells"].items():
        matrix = matrices[key]
        assert matrix.ndim == 2 and matrix.shape[1] == 31 and np.isfinite(matrix).all()
        for guard, expected in guards.items():
            g = int(guard)
            m = matrix[g : len(matrix) - g] if g else matrix
            curve = np.sum(m, axis=0) / len(m)
            j = np.argmin(curve)
            d = float(curve[j])
            b = float(sorted(curve)[15])
            actual = {
                "C": b - d,
                "D": d,
                "B": b,
                "offset": int(15 - j),
                "rows": len(m),
                "boundary": bool(j == 0 or j == 30),
            }
            for field, value in actual.items():
                err = abs(value - expected[field])
                assert err < 1e-10, (path, key, guard, field, err)
                ERRORS["score"] = max(ERRORS["score"], err)
            COUNTS["scores"] += 1
    return s, matrices


def bootstrap_walk(obj):
    if isinstance(obj, dict):
        if "per_speaker" in obj:
            a = np.array([obj["per_speaker"][k] for k in sorted(obj["per_speaker"])])
            rng = np.random.default_rng(20260926)
            indices = rng.integers(len(a), size=(20000, len(a)))
            distribution = np.sum(a[indices], axis=1) / len(a)
            for field, expected in [
                ("speaker_mean", a.mean()),
                ("speaker_ci95", np.percentile(distribution, [2.5, 97.5])),
                ("speaker_ci99", np.percentile(distribution, [0.5, 99.5])),
            ]:
                err = float(np.max(np.abs(np.asarray(obj[field]) - expected)))
                assert err < 1e-10
                ERRORS["bootstrap"] = max(ERRORS["bootstrap"], err)
            assert obj["speaker_positive"] == int((a > 0).sum())
            COUNTS["bootstrap_summaries"] += 1
        else:
            for v in obj.values():
                bootstrap_walk(v)
    elif isinstance(obj, list):
        for v in obj:
            bootstrap_walk(v)


def main():
    protocol = read(BASE / "protocol.json")
    assert len(protocol["rows"]) == 74 and len(protocol["missing"]) == 26
    assert len({r["speaker"] for r in protocol["rows"]}) == 13
    timing_maps = read(BASE / "timing/maps.json")
    assert len(timing_maps["accepted"]) == 74 and not timing_maps["rejected"]
    for row in timing_maps["accepted"]:
        for arm, digest in row["textgrid_hashes"].items():
            check_hash(
                BASE
                / "timing/textgrids"
                / f"{row['speaker']}_{arm}"
                / f"{row['id']}_{arm}.TextGrid",
                digest,
            )
        for mapping in row["maps"].values():
            assert np.all(np.diff(mapping["target_N"]) > 0)
            assert np.all(np.diff(mapping["source"]) > 0)
            assert mapping["target_N"][0] == mapping["source"][0] == 0
            assert mapping["target_N"][-1] == mapping["source"][-1]
    for row in read(BASE / "lag_scale/scores.json"):
        for cell in row["cells"].values():
            curve = np.array(cell["curve"])
            lags = np.array(cell["integer_lags"])
            original = curve[np.abs(lags) <= 15]
            assert abs(np.min(original) - cell["D"]) < 1e-12
            assert abs(np.median(original) - cell["B"]) < 1e-12
            interp = np.interp(
                np.linspace(-cell["lag_limit_frames"], cell["lag_limit_frames"], 31),
                lags,
                curve,
            )
            assert abs(np.median(interp) - cell["B_relative"]) < 1e-12
            assert abs(cell["C_star"] - (cell["B_relative"] - cell["D"])) < 1e-12
    for row in read(BASE / "speech_windows/scores.json"):
        matrices = np.load(BASE / "scores" / f"{row['id']}.npz")
        for info in row["source"].values():
            check_hash(info["textgrid"], info["sha256"])
        for key, cell in row["cells"].items():
            arm, _ = key.split("_")
            for mode in ("all", "speech"):
                curve = matrices[arm + "_source"][cell[mode + "_rows"]].mean(0)
                assert abs(np.median(curve) - curve.min() - cell[mode]["C"]) < 1e-12
    for row in read(BASE / "speech_windows/joint_scores.json"):
        matrices = np.load(BASE / "speed/scores" / f"{row['id']}.npz")
        for key, cell in row["cells"].items():
            curve = matrices[key.rsplit("_", 1)[0]][cell["rows"]].mean(0)
            assert abs(np.median(curve) - curve.min() - cell["C"]) < 1e-12
    for relative in [
        "controls/protocol.json",
        "timing/protocol.json",
        "speed/protocol.json",
    ]:
        check_hash(BASE / "protocol.json", read(BASE / relative)["parent_sha256"])
    all_rows = {}
    for row in protocol["rows"]:
        sid = row["id"]
        assert row["source_lengths"]["N"] == row["source_lengths"]["M"]
        for arm, cell in row["cells"].items():
            for name in ("audio", "video", "crop", "tracks", "score_receipt"):
                check_hash(cell[name], cell[name + "_sha256"])
        parent, mats = scorefile(BASE / "scores" / f"{sid}.json")
        all_rows[sid] = parent
        features_path = BASE / "features" / sid / "features.npz"
        check_hash(features_path, parent["features_sha256"])
        features = np.load(features_path)
        for key, m in mats.items():
            if key.startswith("cross_"):
                _, v, a = key.split("_")
                start, end = parent["cross_support"]
                vs = row["cells"][v]["track_start"]
                distance(
                    m,
                    features["v_" + v][start - vs : end - vs],
                    features[f"a_{a}_source"][start:end],
                )
            else:
                arm, kind = key.split("_")
                start = row["cells"][arm]["track_start"] if kind == "source" else 0
                distance(m, features["v_" + arm], features[f"a_{arm}_{kind}"][start:])
        for arm, replay in parent["replay"].items():
            c = row["cells"][arm]
            assert (
                max(
                    abs(replay["score"]["C"] - c["old_C"]),
                    abs(replay["score"]["D"] - c["old_D"]),
                )
                < 0.000501
            )
        roi, mat = scorefile(BASE / "controls/roi_scores" / f"{sid}.json")
        vp = BASE / "controls/roi_scores" / f"{sid}_visual.npz"
        check_hash(vp, roi["visual_sha256"])
        vf = np.load(vp)
        receipt = read(BASE / "controls/roi" / sid / "receipt.json")
        for info in receipt.values():
            check_hash(info["path"], info["sha256"])
        assert max(roi["replay_feature_error"].values()) == 0
        for key, m in mat.items():
            plan, v, a = key.split("_")
            start, end = roi["support"][plan]
            vs = receipt[plan + "_" + v]["start"]
            distance(
                m,
                vf[plan + "_" + v][start - vs : end - vs],
                features[f"a_{a}_source"][start:end],
            )
        band, _ = scorefile(BASE / "controls/band_scores" / f"{sid}.json")
        for info in band["paths"].values():
            check_hash(info["path"], info["sha256"])
        for branch in ("timing", "speed"):
            p = BASE / branch / "scores" / f"{sid}.json"
            if not p.exists():
                raise AssertionError("missing " + str(p))
            scores, mat = scorefile(p)
            if branch == "timing":
                vp = p.with_name(sid + "_visual.npz")
                check_hash(vp, scores["visual_sha256"])
                vf = np.load(vp)
                receipt = read(BASE / "timing/videos" / sid / "receipt.json")
                for info in receipt.values():
                    check_hash(info["path"], info["sha256"])
                start, end = scores["support"]
                for key, m in mat.items():
                    vs = receipt[key]["start"]
                    distance(
                        m,
                        vf[key][start - vs : end - vs],
                        features["a_N_source"][start:end],
                    )
            else:
                fp = p.with_name(sid + "_features.npz")
                check_hash(fp, scores["features_sha256"])
                f = np.load(fp)
                receipt = read(BASE / "speed/assets" / sid / "receipt.json")
                for info in receipt.values():
                    for kind in ("audio", "video"):
                        check_hash(info[kind], info[kind + "_sha256"])
                for key, m in mat.items():
                    k = key.removesuffix("_common")
                    distance(m, f["v_" + k], f["a_" + k])
        for info in read(BASE / "pcm" / sid / "receipt.json").values():
            check_hash(info["path"], info["sha256"])
    # Reconstruct primary paired speaker means directly from raw saved cell scores.
    analysis = read(BASE / "analysis.json")
    for name, left, right in [
        ("native_pcm_T_N", "T_source", "N_source"),
        ("mfa_replacement_M_N", "cross_M_N", "cross_N_N"),
    ]:
        for guard in ("0", "15", "20"):
            for metric in ("C", "B", "D"):
                byspeaker = {}
                for r in all_rows.values():
                    byspeaker.setdefault(r["speaker"], []).append(
                        r["cells"][left][guard][metric]
                        - r["cells"][right][guard][metric]
                    )
                stat = analysis["guards"][guard][name][metric]
                for speaker, values in byspeaker.items():
                    assert abs(np.mean(values) - stat["per_speaker"][speaker]) < 1e-12
    for relative in [
        "analysis.json",
        "controls/analysis.json",
        "timing/analysis.json",
        "speed/analysis.json",
        "lag_scale/analysis.json",
        "direct_supplement.json",
        "speech_windows/analysis.json",
        "speech_windows/joint_analysis.json",
    ]:
        bootstrap_walk(read(BASE / relative))
    result = {
        "status": "PASS",
        "counts": COUNTS,
        "maximum_errors": ERRORS,
        "limitations": "Sampled five rows per raw embedding matrix; band matrices checked by saved-distance arithmetic and audio hashes, no new inference; bootstrap intervals are exploratory, not multiplicity-adjusted.",
    }
    (BASE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
