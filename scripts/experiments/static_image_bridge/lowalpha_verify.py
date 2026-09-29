"""Independent matrix reanalysis and official-forward checks; never rewrites scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .frontal_verify import independent_crop


def support(length: int) -> list[int]:
    return [t for t in range(length // 640)
            if (t - 15) * 640 - 1 >= 0 and (t + 15) * 640 + 3440 <= length
            and int(t * 80 / 25) * 200 - 401 >= 0
            and (int((t + 4) * 80 / 25) + 15) * 200 + 400 <= length]


def reanalyze(root: Path, p: dict, result: dict) -> dict:
    errors = []
    for row in result["pairs"]:
        r = next(r for r in p["records"] if r["sample_id"] == row["sample_id"])
        w = support(r["length"])
        assert w == r["support"]["primary"]
        base = root / "matrices" / row["image_id"] / row["sample_id"]
        curves = [np.load(base / f"{v}__N__matrix.npy")[w].mean(0) for v in ("N", row["arm"])]
        assert all(np.isfinite(x).all() for x in curves)
        n, b = curves
        own = np.load(base / f"{row['arm']}__{row['arm']}__matrix.npy")[w].mean(0)
        calculated = {"delta_C": float(np.median(b) - b.min() - np.median(n) + n.min()),
                      "D_improvement": float(n.min() - b.min()),
                      "anchor_improvement": float(n.min() - b[n.argmin()]),
                      "own_vs_N_diagonal_C": float(np.median(own) - own.min() - np.median(n) + n.min())}
        errors.extend(abs(calculated[k] - row[k]) for k in calculated)
    expected = {(r["sample_id"], im["id"], arm) for r in p["records"] for im in p["images"] for arm in ("B025", "B050")}
    actual = {(r["sample_id"], r["image_id"], r["arm"]) for r in result["pairs"]}
    assert actual == expected and len(result["pairs"]) == len(expected) and max(errors) < 1e-12
    return {"pairs": len(result["pairs"]), "max_error": max(errors), "support_independently_rederived": True}


def forward(root: Path, p: dict, layout: str = "matched") -> list[dict]:
    import python_speech_features
    import torch
    from scipy.io import wavfile

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "third_party/syncnet_python"))
    from SyncNetInstance import SyncNetInstance, calc_pdist

    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 5 * 1024**3
    scorer = SyncNetInstance(device="cuda")
    scorer.loadParameters(str(repo / "third_party/syncnet_python/data/syncnet_v2.model"))
    scorer.eval()
    record = p["records"][0]
    features = {}
    for arm in ("N", "B025", "B050"):
        rate, wav = wavfile.read(record["audio"][arm]["path"])
        mfcc = np.stack(list(zip(*python_speech_features.mfcc(wav, rate))))
        full = torch.from_numpy(mfcc[None, None].astype(float)).float()
        count = (mfcc.shape[1] - 20) // 4 + 1
        batches = []
        with torch.no_grad():
            for i in range(0, count, 20):
                batch = torch.cat([full[:, :, :, 4*t:4*t+20] for t in range(i, min(count, i+20))])
                batches.append(scorer.__S__.forward_aud(batch.cuda()).cpu())
        features[arm] = torch.cat(batches)
    rows = []
    for im in p["images"]:
        for v in ("N", "B025", "B050"):
            video = root / "videos" / im["id"] / record["sample_id"] / f"{v}.mkv"
            cap = cv2.VideoCapture(str(video))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(independent_crop(frame, im["score_box"]["box"]))
            cap.release()
            full = torch.from_numpy(np.transpose(np.stack(frames), (3, 0, 1, 2))[None]).float()
            batches = []
            with torch.no_grad():
                for i in range(0, len(frames)-4, 20):
                    if layout == "legacy":
                        batch = torch.cat([full[:, :, t:t+5] for t in range(i, min(len(frames)-4, i+20))])
                    else:
                        # Match the worker's channel-interleaved tensor strides;
                        # equal values with different strides can select different CUDA kernels.
                        windows = np.stack([np.stack(frames[t:t+5]) for t in range(i, min(len(frames)-4, i+20))])
                        batch = torch.from_numpy(np.transpose(windows, (0, 4, 1, 2, 3))).float()
                    batches.append(scorer.__S__.forward_lip(batch.cuda()).cpu())
            vf = torch.cat(batches)
            saved_visual = np.load(root / "matrices" / im["id"] / record["sample_id"] / f"{v}__visual.npy")
            visual_error = float(np.max(np.abs(vf.numpy() - saved_visual)))
            for a in (("N",) if v == "N" else ("N", v)):
                recomputed = torch.stack(calc_pdist(vf, features[a], vshift=15)).numpy()
                stored = np.load(root / "matrices" / im["id"] / record["sample_id"] / f"{v}__{a}__matrix.npy")
                mask = np.isfinite(stored)
                err = float(np.max(np.abs(recomputed[:len(stored)][mask] - stored[mask])))
                if err >= 1e-5:
                    (root / f"forward_failure_{layout}.json").write_text(json.dumps({"image": im["id"], "video": v, "audio": a, "max_error": err, "visual_error": visual_error, "layout": layout}, indent=2) + "\n")
                assert err < 1e-5, (im["id"], v, a, err)
                rows.append({"image": im["id"], "video": v, "audio": a, "max_error": err, "visual_error": visual_error, "layout": layout})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--layout", choices=("legacy", "matched"), default="matched")
    parser.add_argument("--image-id")
    args = parser.parse_args()
    root = args.run_root.resolve()
    p = json.loads((root / "protocol.json").read_text())
    if (root / "analysis_scope.json").exists():
        scope = json.loads((root / "analysis_scope.json").read_text())
        p["records"] = [r for r in p["records"] if r["sample_id"] in scope["sample_ids"]]
    if args.image_id:
        assert args.forward_only
        p["images"] = [im for im in p["images"] if im["id"] == args.image_id]
    result = {"status": "PASS", "official_forward_scope": f"first audio x {len(p['images'])} portraits, {5 * len(p['images'])} cells only", "forward": forward(root, p, args.layout)}
    if not args.forward_only:
        result["reanalysis"] = reanalyze(root, p, json.loads((root / "analysis.json").read_text()))
    target = root / ("forward_verification.json" if args.forward_only else "independent_verification.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
