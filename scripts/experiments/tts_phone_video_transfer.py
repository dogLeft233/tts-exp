"""Post-hoc finer MFA clocks: phone centers and boundaries, fixed evaluation split."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from scripts.experiments import tts_raw_video_transfer as transfer
from scripts.experiments.tts_independent_visual import (
    OUT,
    PARENT,
    ROOT,
    read,
    sha,
    write,
)
from scripts.experiments.tts_pcm_residual import cluster
from scripts.experiments.tts_pcm_timing import decode

BASE = OUT / "phone_transfer"
RAW = OUT / "raw_transfer"


def phones(path):
    content = (
        Path(path).read_text().split('name = "phones"', 1)[1].split("item [", 1)[0]
    )
    return [
        (p, float(a), float(b))
        for a, b, p in re.findall(
            r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',
            content,
        )
        if p.strip() and p not in ("sil", "sp", "spn")
    ]


def prepare():
    write(
        BASE / "protocol.json",
        {
            "scope": "Post-hoc finer-clock sensitivity after word/visual transfer and roundtrip diagnostics; fixed42 evaluation, strict source phone sequence eligibility, no outcome selection",
            "mapping": "phone centers or all phone start/end boundaries; duplicate N boundary times average T anchors; fixed 0/duration endpoints; monotone source map; linear pixels FFV1",
            "primary_parent_unchanged": True,
        },
    )
    old = {r["id"]: r for r in read(PARENT / "protocol.json")["rows"]}
    aligned = read(
        ROOT
        / "runs/aishell1_qwen_mfa_linear_n100_20260816/02_mfa_mandarin341_ready/mfa_summary.json"
    )["records"]
    rows, missing = [], []
    for r in read(RAW / "manifest.json")["rows"]:
        if r["split"] != "evaluation":
            continue
        sid = r["id"]
        anchors, grids = {}, {}
        for arm, label in [("N", "natural"), ("T", "tts")]:
            a = aligned[sid][label]
            assert a["audio_sha256"] == old[sid]["cells"][arm]["audio_sha256"]
            assert sha(a["textgrid"]) == a["textgrid_sha256"]
            anchors[arm] = phones(a["textgrid"])
            grids[arm] = {"path": a["textgrid"], "sha256": a["textgrid_sha256"]}
        if [v[0] for v in anchors["N"]] != [v[0] for v in anchors["T"]]:
            missing.append({"id": sid, "reason": "phone sequence differs"})
            continue
        source = old[sid]
        assert sha(source["cells"]["T"]["crop"]) == source["cells"]["T"]["crop_sha256"]
        frames = decode(source["cells"]["T"]["crop"])
        tn = np.arange(source["cells"]["N"]["track_frames"]) / 25
        dn, dt = (
            source["source_lengths"]["N"] / 16000,
            source["source_lengths"]["T"] / 16000,
        )
        row = {k: r[k] for k in ("id", "speaker", "split")}
        row.update(
            videos={"N_identity": r["videos"]["N_identity"]}, textgrids=grids, maps={}
        )
        for mode in ("centers", "boundaries"):
            na = np.array([[a, b] for _, a, b in anchors["N"]])
            ta = np.array([[a, b] for _, a, b in anchors["T"]])
            if mode == "centers":
                x, y = na.mean(1), ta.mean(1)
            else:
                nx, ty = na.ravel(), ta.ravel()
                x = np.unique(nx)
                y = np.array([ty[nx == v].mean() for v in x])
            keep = (x > 0) & (x < dn)
            x, y = np.r_[0, x[keep], dn], np.r_[0, y[keep], dt]
            assert np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)
            file = BASE / "videos" / sid / f"T_phone_{mode}.avi"
            transfer.render(frames, np.interp(tn, x, y), file)
            row["videos"]["T_phone_" + mode] = {
                "path": str(file),
                "sha256": sha(file),
                "frames": len(tn),
            }
            row["maps"][mode] = {"N": x.tolist(), "T": y.tolist()}
        rows.append(row)
        print("phone transfer prepared", sid, flush=True)
    write(BASE / "manifest.json", {"rows": rows, "missing": missing})


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
        for mode in ("centers", "boundaries"):
            for ref in ("N_identity", "T_mfa"):
                values = []
                for r in rows:
                    baseline = read(RAW / "scores" / f"{r['id']}.json")["cells"][ref][
                        guard
                    ]["C"]
                    values.append(r["cells"]["T_phone_" + mode][guard]["C"] - baseline)
                result[guard][mode + "_minus_" + ref] = cluster(
                    values, [r["speaker"] for r in rows]
                )
    write(BASE / "analysis.json", result)
    for k, v in result["20"].items():
        print(
            k, v["speaker_mean"], v["speaker_ci95"], v["n"], v["speakers"], flush=True
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "score", "analyze"])
    globals()[p.parse_args().stage]()
