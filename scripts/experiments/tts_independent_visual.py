"""Frozen visual-only content measurement on the source-PCM audit cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "runs/tts_pcm_residual_20260926"
OUT = ROOT / "runs/tts_independent_visual_20260926"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def freeze():
    from scripts.experiments.vsr_tts_metrics import normalize_text, token_ids
    from scripts.experiments.vsr_tts_pilot import (
        AVSR_ROOT,
        MODEL_JSON_REL,
        MODEL_REL,
        _load_model_char_list,
        _make_config,
    )

    source = (
        ROOT
        / "runs/aishell1_qwen_mfa_linear_n100_20260816/02_mfa_mandarin341_ready/mfa_summary.json"
    )
    texts = read(source)["records"]
    chars = _load_model_char_list(AVSR_ROOT / MODEL_JSON_REL)
    eligible, missing = [], []
    for row in read(PARENT / "protocol.json")["rows"]:
        sid = row["id"]
        text = normalize_text(texts[sid]["cleaned_lab_text"])
        try:
            ids = token_ids(text, chars)
        except ValueError as exc:
            missing.append(
                {"id": sid, "speaker": row["speaker"], "text": text, "reason": str(exc)}
            )
            continue
        cells = {}
        for arm, cell in row["cells"].items():
            assert sha(cell["video"]) == cell["video_sha256"]
            cells[arm] = {"video": cell["video"], "sha256": cell["video_sha256"]}
        eligible.append(
            {
                "id": sid,
                "speaker": row["speaker"],
                "text": text,
                "original_transcript": texts[sid]["transcript"],
                "target": ids,
                "cells": cells,
            }
        )
    for speaker in sorted({r["speaker"] for r in eligible}):
        group = sorted(
            [r for r in eligible if r["speaker"] == speaker], key=lambda r: r["id"]
        )
        for i, row in enumerate(group):
            row["split"] = "calibration" if i < 2 else "evaluation"
    for row in eligible:
        decoys = sorted(
            [r for r in eligible if r["text"] != row["text"]],
            key=lambda r: (abs(len(r["target"]) - len(row["target"])), r["id"]),
        )[:5]
        row["decoys"] = [
            {"id": r["id"], "text": r["text"], "target": r["target"]} for r in decoys
        ]
    config = _make_config(OUT)
    p = {
        "parent_sha256": sha(PARENT / "protocol.json"),
        "text_source": str(source),
        "text_source_sha256": sha(source),
        "model": str(AVSR_ROOT / MODEL_REL),
        "model_sha256": sha(AVSR_ROOT / MODEL_REL),
        "model_json": str(AVSR_ROOT / MODEL_JSON_REL),
        "model_json_sha256": sha(AVSR_ROOT / MODEL_JSON_REL),
        "config": str(config),
        "config_sha256": sha(config),
        "chars": chars,
        "selection": "all parent74 with source-bound text in frozen CMLR vocabulary, no output-based filtering; per speaker first2 calibration, remaining evaluation",
        "views": "native/frozen/reversed for N,T,M,D; N/T matched and matched_frozen at rounded geometric mean frame count",
        "metric": "M=mean 5 fixed length-matched decoy per-char CTC NLL - target per-char CTC NLL; Q=Mnative-Mfrozen; target NLL and greedy CER secondary",
        "calibration_gate": {
            "strict_top1_fraction": 0.60,
            "q_positive_fraction": 0.70,
            "q_speaker_ci95_low": 0.0,
            "reverse_speaker_ci95_low": 0.0,
        },
        "primary": "evaluation split M-N Q and raw M; secondary T-N native and matched, D subset. No human ratings; content only, not precise timing/perception truth.",
        "bootstrap": "speaker-equal, 20000 draws, seed20260926, 95/99%; all-cohort results exploratory only",
        "rows": eligible,
        "missing": missing,
    }
    dest = OUT / "protocol.json"
    if dest.exists() and read(dest) != p:
        raise ValueError("frozen protocol differs")
    write(dest, p)
    print(
        "frozen",
        len(eligible),
        "eligible",
        len(missing),
        "OOV;",
        sum(r["split"] == "calibration" for r in eligible),
        "calibration",
        flush=True,
    )


def extract():
    from scripts.experiments.tts_native_gain_attribution.common import gpu_lease

    with gpu_lease(
        gpu_peak_bytes=6 << 30, disk_temp_bytes=500 << 20, disk_persistent_bytes=4 << 30
    ) as gate:
        write(OUT / "compute_gate.json", gate)
        env = dict(os.environ)
        env.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "2",
                "PYTHONNOUSERSITE": "1",
            }
        )
        cmd = [
            "/home/wjj/miniconda3/envs/autoavsr/bin/python",
            "-m",
            "scripts.experiments.tts_independent_visual",
            "worker",
        ]
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def worker():
    import torch

    from scripts.experiments.vsr_tts_metrics import make_views
    from scripts.experiments.vsr_tts_pilot import (
        _forward_view,
        _set_determinism,
        extract_views,
        load_vsr,
    )

    torch.set_num_threads(2)
    _set_determinism(20260926)
    p = read(OUT / "protocol.json")
    for key in ("model", "model_json", "config"):
        assert sha(p[key]) == p[key + "_sha256"]
    pipeline = load_vsr(OUT, device="cuda:0")
    assert pipeline.modality == "video"
    write(
        OUT / "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "protocol_sha256": sha(OUT / "protocol.json"),
            "worker_pid": os.getpid(),
            "modality": pipeline.modality,
            "decoder": "CTC head only; no audio and no language-model beam decoding",
        },
    )
    for row in p["rows"]:
        sid = row["id"]
        done = OUT / "receipts" / f"{sid}.json"
        if done.exists():
            continue
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        if {int(x.strip()) for x in pids.splitlines() if x.strip().isdigit()} - {
            os.getpid()
        }:
            raise RuntimeError("foreign GPU process")
        cropped = {}
        qc = {}
        hashes = {}
        for arm, cell in row["cells"].items():
            assert sha(cell["video"]) == cell["sha256"]
            cache = OUT / "mouths" / sid / f"{arm}.npz"
            qpath = cache.with_suffix(".json")
            if cache.exists() and qpath.exists():
                qc[arm] = read(qpath)
                assert sha(cache) == qc[arm]["cache_sha256"]
                cropped[arm] = np.load(cache)["x"]
            else:
                result = extract_views(
                    pipeline, {"id": sid, arm + "_video": cell["video"]}, arm, OUT
                )
                cropped[arm] = result["x"]
                qc[arm] = result["qc"]
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, x=cropped[arm])
                qc[arm]["cache_sha256"] = sha(cache)
                write(qpath, qc[arm])
        matched_length = int(
            np.floor(np.sqrt(cropped["N"].shape[1] * cropped["T"].shape[1]) + 0.5)
        )
        repeats = {}
        for arm, x in cropped.items():
            views = make_views(
                x, matched_length=matched_length if arm in ("N", "T") else x.shape[1]
            )
            for name in (
                ("native", "frozen", "reversed", "matched", "matched_frozen")
                if arm in ("N", "T")
                else ("native", "frozen", "reversed")
            ):
                path = OUT / "features" / sid / f"{arm}_{name}.npz"
                if not path.exists():
                    encoder, logp = _forward_view(pipeline, views[name], "cuda:0")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, encoder=encoder, logp=logp)
                hashes[f"{arm}_{name}"] = sha(path)
                if sid == p["rows"][0]["id"] and name == "native":
                    e, l = _forward_view(pipeline, views[name], "cuda:0")
                    old = np.load(path)
                    repeats[arm] = {
                        "encoder_max_error": float(np.max(np.abs(e - old["encoder"]))),
                        "logp_max_error": float(np.max(np.abs(l - old["logp"]))),
                    }
        write(
            done,
            {
                "id": sid,
                "speaker": row["speaker"],
                "split": row["split"],
                "qc": qc,
                "feature_hashes": hashes,
                "repeat": repeats,
                "matched_length": matched_length,
                "eligible": all(qc[a]["eligible_for_pair"] for a in ("N", "T", "M")),
            },
        )
        print(
            "visual extracted",
            sid,
            row["split"],
            "QC",
            all(qc[a]["eligible_for_pair"] for a in ("N", "T", "M")),
            flush=True,
        )


def analyze():
    from scripts.experiments.tts_pcm_residual import cluster
    from scripts.experiments.vsr_tts_metrics import ctc_nll, greedy_cer

    rows = []
    missing = []
    for row in read(OUT / "protocol.json")["rows"]:
        sid = row["id"]
        receipt = read(OUT / "receipts" / f"{sid}.json")
        if not receipt["eligible"]:
            missing.append({"id": sid, "reason": "media_crop_QC"})
            continue
        cells = {}
        for key, digest in receipt["feature_hashes"].items():
            path = OUT / "features" / sid / f"{key}.npz"
            assert sha(path) == digest
            logp = np.load(path)["logp"]
            targets = [row["target"]] + [r["target"] for r in row["decoys"]]
            losses = [ctc_nll(logp, ids) / len(ids) for ids in targets]
            cells[key] = {
                "losses": losses,
                "M": float(np.mean(losses[1:]) - losses[0]),
                "target_nll": losses[0],
                "top1": all(losses[0] < v for v in losses[1:]),
                "CER": greedy_cer(logp, row["target"]),
            }
        arms = {}
        for arm in row["cells"]:
            if not receipt["qc"][arm]["eligible_for_pair"]:
                missing.append(
                    {"id": sid, "arm": arm, "reason": "auxiliary_media_crop_QC"}
                )
                continue
            native = cells[arm + "_native"]
            frozen = cells[arm + "_frozen"]
            reverse = cells[arm + "_reversed"]
            arms[arm] = {
                "M": native["M"],
                "Q": native["M"] - frozen["M"],
                "reverse": native["M"] - reverse["M"],
                "CER": native["CER"],
                "target_support": -native["target_nll"],
                "top1": native["top1"],
            }
            if arm in ("N", "T"):
                arms[arm]["Mmatched"] = cells[arm + "_matched"]["M"]
                arms[arm]["Qmatched"] = (
                    cells[arm + "_matched"]["M"] - cells[arm + "_matched_frozen"]["M"]
                )
        rows.append(
            {
                "id": sid,
                "speaker": row["speaker"],
                "split": row["split"],
                "cells": cells,
                "arms": arms,
            }
        )
    cal = [r for r in rows if r["split"] == "calibration"]
    calibration = {}
    for arm in ("N", "T", "M"):
        q = cluster([r["arms"][arm]["Q"] for r in cal], [r["speaker"] for r in cal])
        rev = cluster(
            [r["arms"][arm]["reverse"] for r in cal], [r["speaker"] for r in cal]
        )
        top = sum(r["arms"][arm]["top1"] for r in cal) / len(cal)
        pos = sum(r["arms"][arm]["Q"] > 0 for r in cal) / len(cal)
        calibration[arm] = {
            "n": len(cal),
            "top1_fraction": top,
            "q_positive_fraction": pos,
            "Q": q,
            "reverse": rev,
            "pass": top >= 0.60
            and pos >= 0.70
            and q["speaker_ci95"][0] > 0
            and rev["speaker_ci95"][0] > 0,
        }
    result = {
        "calibration": calibration,
        "primary_gate": calibration["N"]["pass"],
        "splits": {},
        "missing": missing,
    }
    for split in ("evaluation", "calibration", "all"):
        group = [r for r in rows if split == "all" or r["split"] == split]
        results = {}
        for left, right in [("M", "N"), ("T", "N"), ("D", "N"), ("M", "D")]:
            rr = [r for r in group if left in r["arms"] and right in r["arms"]]
            if not rr:
                continue
            metrics = ("M", "Q", "CER", "target_support") + (
                ("Mmatched", "Qmatched") if left == "T" and right == "N" else ()
            )
            for metric in metrics:
                results[f"{left}_{right}_{metric}"] = cluster(
                    [r["arms"][left][metric] - r["arms"][right][metric] for r in rr],
                    [r["speaker"] for r in rr],
                )
        result["splits"][split] = results
    write(OUT / "scores.json", rows)
    write(OUT / "analysis.json", result)
    print(
        "visual analysis complete, calibration N:", result["primary_gate"], flush=True
    )
    for key, v in result["splits"]["evaluation"].items():
        print(key, v["speaker_mean"], v["speaker_ci95"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "extract", "worker", "analyze"])
    globals()[parser.parse_args().stage]()
