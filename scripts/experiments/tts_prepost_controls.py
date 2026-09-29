"""Known delay and frozen-frame measurement controls for the static protocol."""

from __future__ import annotations

import os

import numpy as np

from scripts.experiments.tts_independent_visual import sha
from scripts.experiments.tts_native_gain_attribution.common import (
    gpu_compute_pids,
    gpu_lease,
)
from scripts.experiments.tts_pcm_residual import metrics
from scripts.experiments.tts_pcm_timing import decode
from scripts.experiments.tts_prepost_mel import BASE, PARENT, read, write
from scripts.experiments.tts_raw_video_transfer import render


def main():
    import torch

    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    p = read(BASE / "protocol.json")
    write(
        BASE / "controls/protocol.json",
        {
            "selection": "first2 fixed evaluation IDs on each3image, no score selection",
            "arms": "N replay; middle frame frozen; prepend5 first-frame copies",
            "support": "same N embedding rows,guard20",
            "gate": "each N-frozenC>.5; abs((delay_offset-N_offset)-5)<=1; no exclusion if gate fails",
        },
    )
    rows = []
    with gpu_lease(
        gpu_peak_bytes=3 << 30, disk_temp_bytes=100 << 20, disk_persistent_bytes=1 << 30
    ) as gate:
        write(BASE / "controls/compute_gate.json", gate)
        engine = SyncNetEngine(batch_size=32, device="cuda")
        try:
            for image in p["images"]:
                for r in p["rows"][:2]:
                    assert not set(gpu_compute_pids()) - {os.getpid()}
                    sid = r["id"]
                    old = read(BASE / "scores" / image["id"] / f"{sid}.json")
                    info = old["videos"]["N"]
                    assert sha(info["path"]) == info["sha256"]
                    frames = decode(info["path"])
                    arrays = {
                        "frozen": np.repeat(
                            frames[len(frames) // 2 : len(frames) // 2 + 1],
                            len(frames),
                            axis=0,
                        ),
                        "delay5": np.concatenate(
                            [np.repeat(frames[:1], 5, axis=0), frames], axis=0
                        ),
                    }
                    audio = np.load(PARENT / "features" / sid / "features.npz")[
                        "a_N_source"
                    ]
                    n = old["cells"]["N"]["0"]["rows"]
                    matrices = {}
                    cells = {}
                    folder = BASE / "controls" / image["id"] / sid
                    receipts = {}
                    for key, arr in arrays.items():
                        file = folder / f"{key}.avi"
                        render(arr, np.arange(len(arr)) / 25, file)
                        v, _ = engine.extract_visual(file)
                        matrix = engine.distance_matrix(v[:n], audio[:n])
                        matrices[key] = matrix
                        cells[key] = metrics(matrix)
                        receipts[key] = {"path": str(file), "sha256": sha(file)}
                    np.savez_compressed(folder / "scores.npz", **matrices)
                    delta = old["cells"]["N"]["20"]["C"] - cells["frozen"]["20"]["C"]
                    shift = (
                        cells["delay5"]["20"]["offset"]
                        - old["cells"]["N"]["20"]["offset"]
                    )
                    row = {
                        "id": sid,
                        "image": image["id"],
                        "cells": cells,
                        "N": old["cells"]["N"],
                        "N_minus_frozen": delta,
                        "offset_delta": shift,
                        "pass": delta > 0.5 and abs(shift - 5) <= 1,
                        "videos": receipts,
                        "matrix_sha256": sha(folder / "scores.npz"),
                    }
                    rows.append(row)
                    write(folder / "scores.json", row)
                    print(
                        "static controls",
                        image["id"],
                        sid,
                        delta,
                        shift,
                        row["pass"],
                        flush=True,
                    )
        finally:
            engine.close()
    write(
        BASE / "controls/analysis.json",
        {"pass": all(r["pass"] for r in rows), "rows": rows},
    )


if __name__ == "__main__":
    main()
