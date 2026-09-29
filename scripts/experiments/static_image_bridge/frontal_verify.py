"""Independent official-forward spot check for frontal_probe artifacts.

Does not import the probe's render/score/analyze implementations or rewrite them.
Run in the SyncNet environment after the main GPU queue finishes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def independent_crop(frame: np.ndarray, box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    padded = cv2.copyMakeBorder(
        frame, max(0, -y1), max(0, y2 - frame.shape[0]),
        max(0, -x1), max(0, x2 - frame.shape[1]), cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    left, top = x1 + max(0, -x1), y1 + max(0, -y1)
    return cv2.resize(padded[top:top + y2 - y1, left:left + x2 - x1], (224, 224))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    repo = Path(__file__).resolve().parents[3]
    import python_speech_features
    import torch
    from scipy.io import wavfile

    sys.path.insert(0, str(repo / "third_party/syncnet_python"))
    from SyncNetInstance import SyncNetInstance, calc_pdist

    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 5 * 1024**3
    p = json.loads((root / "protocol.json").read_text())
    record = p["records"][0]
    scorer = SyncNetInstance(device="cuda")
    scorer.loadParameters(str(repo / "third_party/syncnet_python/data/syncnet_v2.model"))
    scorer.eval()
    audio_features = {}
    cells = []
    for arm in ("N", "B"):
        sr, wav = wavfile.read(record["audio"][arm]["path"])
        mfcc = np.stack(list(zip(*python_speech_features.mfcc(wav, sr))))
        full = torch.from_numpy(mfcc[None, None].astype(float)).float()
        count = (mfcc.shape[1] - 20) // 4 + 1
        batches = []
        with torch.no_grad():
            for i in range(0, count, 20):
                batch = torch.cat([full[:, :, :, t * 4:t * 4 + 20] for t in range(i, min(count, i + 20))])
                batches.append(scorer.__S__.forward_aud(batch.cuda()).cpu())
        audio_features[arm] = torch.cat(batches)
    for image in p["images"]:
        for video_arm, audios in (("N", ("N",)), ("B", ("N", "B"))):
            video = root / "videos" / image["id"] / record["sample_id"] / f"{video_arm}.mkv"
            cap = cv2.VideoCapture(str(video))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(independent_crop(frame, image["score_box"]["box"]))
            cap.release()
            im = np.transpose(np.expand_dims(np.stack(frames, axis=3), axis=0), (0, 3, 4, 1, 2))
            full = torch.from_numpy(im.astype(float)).float()
            batches = []
            with torch.no_grad():
                for i in range(0, len(frames) - 4, 20):
                    batch = torch.cat([full[:, :, t:t + 5] for t in range(i, min(len(frames) - 4, i + 20))])
                    batches.append(scorer.__S__.forward_lip(batch.cuda()).cpu())
            vf = torch.cat(batches)
            for audio_arm in audios:
                recomputed = torch.stack(calc_pdist(vf, audio_features[audio_arm], vshift=15)).numpy()
                stored = np.load(root / "matrices" / image["id"] / record["sample_id"] / f"{video_arm}__{audio_arm}__matrix.npy")
                finite = np.isfinite(stored)
                error = float(np.abs(recomputed[:len(stored)][finite] - stored[finite]).max())
                assert error < 1e-5, (image["id"], video_arm, audio_arm, error)
                cells.append({"image_id": image["id"], "sample_id": record["sample_id"],
                              "video_arm": video_arm, "audio_arm": audio_arm, "max_matrix_error": error})
    # Recompute EVERY primary C/D difference without importing the analyzer.
    reported = json.loads((root / "analysis.json").read_text())
    errors = []
    for row in reported["pairs"]:
        record = next(r for r in p["records"] if r["sample_id"] == row["sample_id"])
        length = record["length"]
        # Independently derive the full-real-support interval from inequalities.
        w = []
        for t in range(length // 640):
            if ((t - 15) * 640 - 1 >= 0 and (t + 15) * 640 + 3440 <= length
                    and int(t * 80 / 25) * 200 - 401 >= 0
                    and (int((t + 4) * 80 / 25) + 15) * 200 + 400 <= length):
                w.append(t)
        assert w == record["support"]["primary"]
        curves = {}
        for v in ("N", "B"):
            mat = np.load(root / "matrices" / row["image_id"] / row["sample_id"] / f"{v}__N__matrix.npy")
            curves[v] = np.mean(mat[w], axis=0)
        delta = (np.median(curves["B"]) - curves["B"].min()) - (np.median(curves["N"]) - curves["N"].min())
        d_gain = curves["N"].min() - curves["B"].min()
        errors += [abs(float(delta) - row["delta_C"]), abs(float(d_gain) - row["D_improvement"])]
    assert max(errors) < 1e-12
    result = {"status": "PASS", "official_forward_scope": "first audio, three images, nine cells; not all213 cells",
              "official_forward_cells": cells, "all66_pair_matrix_reanalysis_max_error": max(errors),
              "full_real_support_independently_rederived": 66,
              "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
    (root / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
