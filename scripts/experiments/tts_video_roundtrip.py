"""Post-hoc pixel-resampling damage control on the fixed evaluation cohort."""

from __future__ import annotations

import argparse

import numpy as np

from scripts.experiments import tts_raw_video_transfer as transfer
from scripts.experiments.tts_independent_visual import OUT, PARENT, read, sha, write
from scripts.experiments.tts_pcm_residual import cluster
from scripts.experiments.tts_pcm_timing import decode

BASE = OUT / "roundtrip"
RAW = OUT / "raw_transfer"


def prepare():
    write(
        BASE / "protocol.json",
        {
            "scope": "Post-hoc after primary transfer outputs; fixed42 evaluation only; no score selection or adjustment of primary endpoint",
            "raw_manifest_sha256": sha(RAW / "manifest.json"),
            "control": "N original pixels inverse-warped to T frame grid, then forward-warped back to N grid, same fixed MFA/visual maps; lossless intermediate and output; compare exact natural audio and processed N identity",
            "limit": "two interpolation passes vs one for T; damage diagnostic, not equivalent counterfactual or causal correction",
        },
    )
    old = {r["id"]: r for r in read(PARENT / "protocol.json")["rows"]}
    rows = []
    for r in read(RAW / "manifest.json")["rows"]:
        if r["split"] != "evaluation":
            continue
        sid = r["id"]
        assert (
            sha(old[sid]["cells"]["N"]["crop"]) == old[sid]["cells"]["N"]["crop_sha256"]
        )
        frames = decode(old[sid]["cells"]["N"]["crop"])
        tn = np.arange(len(frames)) / 25
        tt = np.arange(old[sid]["cells"]["T"]["track_frames"]) / 25
        result = {k: r[k] for k in ("id", "speaker", "split")}
        result["videos"] = {"N_identity": r["videos"]["N_identity"]}
        result["intermediate"] = {}
        for kind in ("mfa", "visual"):
            x, y = np.array(r[kind + "_x"]), np.array(r[kind + "_y"])
            intermediate = BASE / "videos" / sid / f"N_to_T_{kind}.avi"
            transfer.render(frames, np.interp(tt, y, x), intermediate)
            assert len(decode(intermediate)) == len(tt)
            final = intermediate.with_name(f"N_roundtrip_{kind}.avi")
            transfer.render(decode(intermediate), np.interp(tn, x, y), final)
            result["videos"]["N_roundtrip_" + kind] = {
                "path": str(final),
                "sha256": sha(final),
                "frames": len(tn),
            }
            result["intermediate"][kind] = {
                "path": str(intermediate),
                "sha256": sha(intermediate),
            }
        rows.append(result)
        print("roundtrip prepared", sid, flush=True)
    write(BASE / "manifest.json", {"rows": rows})


def score():
    transfer.BASE = BASE
    transfer.score()


def analyze():
    rows = [
        read(BASE / "scores" / f"{r['id']}.json")
        for r in read(BASE / "manifest.json")["rows"]
    ]
    result = {}
    for guard in ("0", "15", "20"):
        result[guard] = {}
        for kind in ("mfa", "visual"):
            result[guard][kind] = cluster(
                [
                    r["cells"]["N_roundtrip_" + kind][guard]["C"]
                    - r["cells"]["N_identity"][guard]["C"]
                    for r in rows
                ],
                [r["speaker"] for r in rows],
            )
    write(BASE / "analysis.json", result)
    for kind, value in result["20"].items():
        print(kind, value["speaker_mean"], value["speaker_ci95"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "score", "analyze"])
    globals()[p.parse_args().stage]()
