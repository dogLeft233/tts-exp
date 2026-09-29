"""Independent source, pixel, mel interpolation and grouped-statistic checks."""

from __future__ import annotations

import argparse
import time

import numpy as np

from scripts.experiments import check_tts_pcm_residual as audit
from scripts.experiments.tts_prepost_mel import BASE, MODEL, PARENT, read, write


def interpolate(x, y, target):
    """Linear interpolation via explicit bracketing, independent of np.interp."""
    q = np.clip(target, x[0], x[-1])
    j = np.clip(np.searchsorted(x, q, side="right") - 1, 0, len(x) - 2)
    width = x[j + 1] - x[j]
    w = np.divide(q - x[j], width, out=np.zeros_like(q), where=width > 0)
    result = y[..., j] * (1 - w) + y[..., j + 1] * w
    # At a repeated first anchor, np.interp uses the rightmost equal abscissa;
    # only strictly out-of-domain queries receive the first ordinate.
    result[..., target < x[0]] = y[..., 0, None]
    result[..., target >= x[-1]] = y[..., -1, None]
    return result


def collect():
    import cv2

    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded

    cv2.setNumThreads(1)
    p = read(BASE / "protocol.json")
    assert np.allclose(
        interpolate(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.1, 1.0]),
            np.array([-0.1, 0.0, 1.0, 1.1]),
        ),
        [0.0, 0.1, 1.0, 1.0],
    )
    audit.check_hash(MODEL, p["model_sha256"])
    max_mel_error = 0.0
    for row in p["rows"]:
        sid = row["id"]
        for info in row["audio"].values():
            audit.check_hash(info["path"], info["sha256"])
        file = BASE / "mels" / f"{sid}.npz"
        receipt = read(file.with_suffix(".json"))
        audit.check_hash(file, receipt["sha256"])
        m = np.load(file)
        x, y = np.array(row["map"]["N"]), np.array(row["map"]["T"])
        tn, tt = np.arange(m["N"].shape[1]) / 80, np.arange(m["T"].shape[1]) / 80
        source = interpolate(x, y, tn)
        inverse = interpolate(y, x, tt)
        assert np.max(np.abs(source - receipt["source"])) < 1e-12
        assert np.max(np.abs(inverse - receipt["inverse"])) < 1e-12
        expected_pre = interpolate(tt, m["T"], source).astype(np.float32)
        intermediate = interpolate(tn, m["N"], inverse).astype(np.float32)
        expected_rt = interpolate(tt, intermediate, source).astype(np.float32)
        error = float(
            max(
                np.max(np.abs(expected_pre - m["PRE"])),
                np.max(np.abs(expected_rt - m["NRT"])),
            )
        )
        assert error < 1e-5
        max_mel_error = max(max_mel_error, error)
    pixels_checked = 0
    receipts = {}
    for image in p["images"]:
        audit.check_hash(image["path"], image["sha256"])
        frame = cv2.imread(image["path"])
        ref = crop_zero_padded(frame, image["score_box"]["box"])
        mask = np.zeros_like(frame)
        x1, y1, x2, y2 = image["generation_box"]
        mask[y1:y2, x1:x2] = 255
        affected = (
            crop_zero_padded(mask, image["score_box"]["box"]).max(2) > 0
        ).astype(np.uint8)
        outside = cv2.dilate(affected, np.ones((3, 3), np.uint8)) == 0
        for row in p["rows"]:
            sid = row["id"]
            file = BASE / "scores" / image["id"] / f"{sid}.json"
            while not file.exists():
                time.sleep(1)
            scored, matrices = audit.scorefile(file)
            assert scored["repeat_pixel_error"] in (None, 0)
            assert scored["outside_generation_box_identity"]
            audit.check_hash(BASE / "mels" / f"{sid}.npz", scored["mel_sha256"])
            for info in scored["videos"].values():
                audit.check_hash(info["path"], info["sha256"])
                cap = cv2.VideoCapture(info["path"])
                count = 0
                while True:
                    ok, decoded = cap.read()
                    if not ok:
                        break
                    assert np.array_equal(decoded[outside], ref[outside]), (
                        image["id"],
                        sid,
                        "outside pixels",
                    )
                    count += 1
                cap.release()
                assert count == info["frames"]
                pixels_checked += count
            vf = file.with_name(sid + "_visual.npz")
            audit.check_hash(vf, scored["visual_sha256"])
            visuals = np.load(vf)
            af = PARENT / "features" / sid / "features.npz"
            audit.check_hash(af, scored["audio_features_sha256"])
            audio = np.load(af)
            for key, matrix in matrices.items():
                audit.distance(
                    matrix,
                    visuals[key],
                    audio["a_T_source" if key == "T" else "a_N_source"],
                )
            receipts[image["id"] + "/" + sid] = {
                "sha256": __import__("hashlib").sha256(file.read_bytes()).hexdigest()
            }
            print("independently verified", image["id"], sid, flush=True)
    write(
        BASE / "collection_validation.json",
        {
            "status": "PASS",
            "counts": audit.COUNTS,
            "errors": audit.ERRORS,
            "mel_max_error": max_mel_error,
            "outside_frames_checked": pixels_checked,
            "receipts": receipts,
        },
    )


def finish():
    p = read(BASE / "protocol.json")
    collection = read(BASE / "collection_validation.json")
    rows = {}
    matrices = {}
    for image in p["images"]:
        for row in p["rows"]:
            sid = row["id"]
            file = BASE / "scores" / image["id"] / f"{sid}.json"
            audit.check_hash(
                file, collection["receipts"][image["id"] + "/" + sid]["sha256"]
            )
            rows[(image["id"], sid)] = read(file)
            matrices[(image["id"], sid)] = dict(np.load(file.with_suffix(".npz")))
    analysis = read(BASE / "analysis.json")
    for guard, groups in analysis.items():
        g = int(guard)
        for group, comparisons in groups.items():
            images = [i["id"] for i in p["images"]] if group == "mean3" else [group]
            for name, values in comparisons.items():
                left, right = name.split("_minus_")
                for metric, summary in values.items():
                    by = {}
                    for row in p["rows"]:
                        local = []
                        for im in images:
                            key = (im, row["id"])
                            if metric == "N_anchor_distance":
                                curves = {
                                    k: (v[g:-g] if g else v).mean(0)
                                    for k, v in matrices[key].items()
                                }
                                j = np.argmin(curves["N"])
                                delta = curves[left][j] - curves[right][j]
                            else:
                                s = rows[key]["cells"]
                                delta = s[left][guard][metric] - s[right][guard][metric]
                            local.append(delta)
                        by.setdefault(row["speaker"], []).append(np.mean(local))
                    for speaker, vals in by.items():
                        assert (
                            abs(np.mean(vals) - summary["per_speaker"][speaker]) < 1e-12
                        )
    audit.bootstrap_walk(analysis)
    controls = read(BASE / "controls/analysis.json")
    for row in controls["rows"]:
        file = BASE / "controls" / row["image"] / row["id"] / "scores.json"
        s, _ = audit.scorefile(file)
        for info in s["videos"].values():
            audit.check_hash(info["path"], info["sha256"])
        delta = s["N"]["20"]["C"] - s["cells"]["frozen"]["20"]["C"]
        shift = s["cells"]["delay5"]["20"]["offset"] - s["N"]["20"]["offset"]
        assert abs(delta - s["N_minus_frozen"]) < 1e-12 and shift == s["offset_delta"]
        assert s["pass"] == (delta > 0.5 and abs(shift - 5) <= 1)
    assert controls["pass"] == all(r["pass"] for r in controls["rows"])
    speech = read(BASE / "speech_scores.json")
    sourcefile = PARENT / "speech_windows/scores.json"
    audit.check_hash(sourcefile, read(BASE / "speech_protocol.json")["source_sha256"])
    masks = {r["id"]: r["cells"]["N_20"]["speech_rows"] for r in read(sourcefile)}
    for key, record in speech.items():
        image, sid = key.split("/")
        n = len(matrices[(image, sid)]["N"])
        assert record["rows"] == [i for i in masks[sid] if 20 <= i < n - 20]
        for arm, cell in record["cells"].items():
            curve = matrices[(image, sid)][arm][record["rows"]].mean(0)
            assert abs(float(np.median(curve) - curve.min()) - cell["C"]) < 1e-12
            assert abs(float(curve.min()) - cell["D"]) < 1e-12
            assert abs(float(np.median(curve)) - cell["B"]) < 1e-12
    speech_analysis = read(BASE / "speech_analysis.json")
    for group, comparisons in speech_analysis.items():
        images = [i["id"] for i in p["images"]] if group == "mean3" else [group]
        for name, values in comparisons.items():
            left, right = name.split("_minus_")
            for metric, summary in values.items():
                by = {}
                for row in p["rows"]:
                    vals = [
                        speech[im + "/" + row["id"]]["cells"][left][metric]
                        - speech[im + "/" + row["id"]]["cells"][right][metric]
                        for im in images
                    ]
                    by.setdefault(row["speaker"], []).append(np.mean(vals))
                for speaker, vals in by.items():
                    assert abs(np.mean(vals) - summary["per_speaker"][speaker]) < 1e-12
    audit.bootstrap_walk(speech_analysis)
    write(
        BASE / "validation.json",
        {
            "status": "PASS",
            "collection": {k: v for k, v in collection.items() if k != "receipts"},
            "final_counts": audit.COUNTS,
            "final_errors": audit.ERRORS,
            "limits": "Independent interpolation/pixel invariance/arithmetic/bootstrap. Five sampled embedding rows; no second neural implementation or human truth. Three images averaged within utterance before speaker bootstrap.",
        },
    )
    print(read(BASE / "validation.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["collect", "finish"])
    globals()[p.parse_args().stage]()
