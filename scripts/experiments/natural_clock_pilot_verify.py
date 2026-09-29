"""Independent CPU artifact/matrix audit; never modifies experimental scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def support(length):
    return [
        t
        for t in range(length // 640)
        if (t - 15) * 640 - 1 >= 0
        and (t + 15) * 640 + 3440 <= length
        and int(t * 80 / 25) * 200 - 401 >= 0
        and (int((t + 4) * 80 / 25) + 15) * 200 + 400 <= length
    ]


def deltas(n, a):
    return {
        "delta_C": float(np.median(a) - a.min() - np.median(n) + n.min()),
        "D_improvement": float(n.min() - a.min()),
        "anchor_improvement": float(n.min() - a[n.argmin()]),
        "lag_change": int(a.argmin() - n.argmin()),
    }


def verify(root):
    p = read(root / "protocol.json")
    a = read(root / "audio_manifest.json")
    result = read(root / "analysis.json")
    assert (
        a["protocol_sha256"] == sha(root / "protocol.json") == result["protocol_sha256"]
    )
    for path, digest in p["bindings"].items():
        assert sha(path) == digest, path
    expected = {
        (r["sample_id"], arm)
        for r in a["records"]
        for arm in ("EQ", "DAC", "FASTPITCH")
    }
    assert (
        len(result["rows"]) == 9
        and {(r["sample_id"], r["arm"]) for r in result["rows"]} == expected
    )
    im = p["images"][0]
    image = cv2.imread(im["path"])
    mask = np.ones(image.shape[:2], bool)
    x1, y1, x2, y2 = im["generation_box"]
    mask[y1:y2, x1:x2] = False
    count_v = count_m = 0
    errors = []
    for i, r in enumerate(a["records"]):
        for arm, meta in r["audio"].items():
            assert sha(meta["path"]) == meta["container_sha256"]
            rate, x = wavfile.read(meta["path"])
            assert (
                rate == 16000
                and x.ndim == 1
                and x.dtype == np.int16
                and len(x) == r["length"]
            )
        w = support(r["length"])
        assert w == r["support"]["primary"]
        sid = r["sample_id"]
        matrix_root = root / "matrices" / im["id"] / sid
        for video_arm in (
            ("N", "EQ", "DAC", "FASTPITCH", "N_REPEAT", "S")
            if i == 0
            else ("N", "EQ", "DAC", "FASTPITCH")
        ):
            video = root / "videos" / im["id"] / sid / f"{video_arm}.mkv"
            receipt = read(video.with_suffix(".json"))
            assert sha(video) == receipt["compact_sha256"]
            worker = receipt["worker"]
            assert worker["audio_sha256"] == sha(r["audio"][video_arm]["path"])
            cap = cv2.VideoCapture(str(video))
            frames = 0
            digest = hashlib.sha256()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                assert np.array_equal(frame[mask], image[mask])
                digest.update(frame.tobytes())
                frames += 1
            cap.release()
            assert digest.hexdigest() == receipt["parity"]["raw_bgr_sha256"]
            assert frames == receipt["parity"]["frame_count"]
            count_v += 1
        for path in matrix_root.glob("*__worker.json"):
            m = read(path)
            assert m["sample_id"] == sid and m["video_arm"] == Path(m["video"]).stem
            assert m["audio"] == r["audio"][m["audio_arm"]]["path"]
            for key in ("audio", "video", "matrix", "model"):
                if key == "model":
                    assert m["model_sha256"] == sha(
                        root.parents[1]
                        / "third_party/syncnet_python/data/syncnet_v2.model"
                    )
                else:
                    assert sha(m[key]) == m[key + "_sha256"]
            count_m += 1
        for arm in ("EQ", "DAC", "FASTPITCH"):
            row = next(
                v for v in result["rows"] if v["sample_id"] == sid and v["arm"] == arm
            )
            n = np.load(matrix_root / "N__N__matrix.npy")[w].mean(0)
            own = np.load(matrix_root / f"{arm}__{arm}__matrix.npy")[w].mean(0)
            cross = np.load(matrix_root / f"{arm}__N__matrix.npy")[w].mean(0)
            real = read(root / "real" / sid / "result.json")
            for k in ("N", arm):
                m = real["scores"][k]
                assert sha(m["matrix"]) == m["matrix_sha256"]
            rn = np.load(root / "real" / sid / "N.npy")[w].mean(0)
            ra = np.load(root / "real" / sid / f"{arm}.npy")[w].mean(0)
            for endpoint, computed in [
                ("native", deltas(n, own)),
                ("replacement", deltas(n, cross)),
                ("real", deltas(rn, ra)),
            ]:
                errors.extend(
                    abs(value - row[endpoint][key]) for key, value in computed.items()
                )
    assert count_v == 14 and count_m == 26, (count_v, count_m)
    assert max(errors) < 1e-12, max(errors)
    check = {
        "status": "PASS",
        "videos_verified": count_v,
        "static_matrices_verified": count_m,
        "comparisons_recomputed": 27,
        "maximum_metric_error": max(errors),
        "support_independently_rederived": True,
        "outside_roi_static_all_frames": True,
        "limitations": "CPU artifact audit, not independent neural-network rerun or human listening; n=3 only",
    }
    (root / "verification.json").write_text(json.dumps(check, indent=2) + "\n")
    print(check)


def forward(root):
    """First source: independent crop + official calc_pdist, seven cells."""
    import python_speech_features
    import torch

    from scripts.experiments.static_image_bridge.frontal_verify import independent_crop

    repo = root.parents[1]
    sys.path.insert(0, str(repo / "third_party/syncnet_python"))
    from SyncNetInstance import SyncNetInstance, calc_pdist

    torch.set_num_threads(2)
    scorer = SyncNetInstance(device="cuda")
    scorer.loadParameters(
        str(repo / "third_party/syncnet_python/data/syncnet_v2.model")
    )
    scorer.eval()
    p = read(root / "protocol.json")
    r = read(root / "audio_manifest.json")["records"][0]
    im = p["images"][0]
    base = root / "matrices" / im["id"] / r["sample_id"]
    features = {}
    for arm in ("N", "EQ", "DAC", "FASTPITCH"):
        rate, wav = wavfile.read(r["audio"][arm]["path"])
        mfcc = np.stack(list(zip(*python_speech_features.mfcc(wav, rate))))
        full = torch.from_numpy(mfcc[None, None].astype(float)).float()
        count = (mfcc.shape[1] - 20) // 4 + 1
        batches = []
        with torch.no_grad():
            for i in range(0, count, 20):
                batch = torch.cat(
                    [
                        full[:, :, :, 4 * t : 4 * t + 20]
                        for t in range(i, min(count, i + 20))
                    ]
                )
                batches.append(scorer.__S__.forward_aud(batch.cuda()).cpu())
        features[arm] = torch.cat(batches)
    checks = []
    for arm in ("N", "EQ", "DAC", "FASTPITCH"):
        cap = cv2.VideoCapture(
            str(root / "videos" / im["id"] / r["sample_id"] / f"{arm}.mkv")
        )
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(independent_crop(frame, im["score_box"]["box"]))
        cap.release()
        count = len(frames) - 4
        batches = []
        with torch.no_grad():
            for i in range(0, count, 20):
                windows = np.stack(
                    [np.stack(frames[t : t + 5]) for t in range(i, min(count, i + 20))]
                )
                # Match published worker tensor strides, avoiding CUDA-layout drift.
                tensor = (
                    torch.from_numpy(np.transpose(windows, (0, 4, 1, 2, 3)))
                    .float()
                    .cuda()
                )
                batches.append(scorer.__S__.forward_lip(tensor).cpu())
        vf = torch.cat(batches)
        for audio_arm in ("N",) if arm == "N" else ("N", arm):
            computed = torch.stack(
                calc_pdist(vf, features[audio_arm], vshift=15)
            ).numpy()
            stored = np.load(base / f"{arm}__{audio_arm}__matrix.npy")
            mask = np.isfinite(stored)
            err = float(np.max(np.abs(computed[: len(stored)][mask] - stored[mask])))
            checks.append({"video": arm, "audio": audio_arm, "max_error": err})
    value = {
        "status": "PASS" if all(c["max_error"] < 1e-5 for c in checks) else "FAIL",
        "scope": "first source x one portrait, seven cells; not all matrices",
        "checks": checks,
    }
    (root / "forward_verification.json").write_text(json.dumps(value, indent=2) + "\n")
    print(value)
    assert value["status"] == "PASS"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--forward", action="store_true")
    args = p.parse_args()
    (forward if args.forward else verify)(args.root)
