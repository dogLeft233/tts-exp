"""Audio-alignment-defined speech windows, without selecting on SyncNet scores."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from scripts.experiments.tts_clock_cross import check_hash, read, sha, write
from scripts.experiments.tts_pcm_residual import OUT, PARENT, cluster


def main():
    base = OUT / "speech_windows"
    protocol = {
        "parent_sha256": sha(OUT / "protocol.json"),
        "question": "Is residual native TTS advantage simply caused by more silence windows in natural speech?",
        "selection": "all74; old MFA N/T TextGrid and input waveform hashes must match frozen source; no score-based selection",
        "mask": "nonempty word-tier intervals; window audio center = row/25 + 0.1s; retain centers inside speech, independently per arm; guards0/15/20; compare all vs speech-only on original distance matrices",
        "limits": "subset diagnostic of evaluation windows, not removal of pauses at generation; MFA can misalign; retained counts differ",
    }
    path = base / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("speech protocol differs")
    write(path, protocol)
    source = read(PARENT / "02_mfa_mandarin341_ready/mfa_summary.json")["records"]
    rows = []
    for row in read(OUT / "protocol.json")["rows"]:
        sid = row["id"]
        score = read(OUT / "scores" / f"{sid}.json")
        fp = OUT / "scores" / f"{sid}.npz"
        check_hash(fp, score["matrices_sha256"])
        matrices = np.load(fp)
        out = {"id": sid, "speaker": row["speaker"], "cells": {}, "source": {}}
        for arm, name in [("N", "natural"), ("T", "tts")]:
            info = source[sid][name]
            assert info["audio_sha256"] == row["cells"][arm]["audio_sha256"]
            check_hash(info["textgrid"], info["textgrid_sha256"])
            text = Path(info["textgrid"]).read_text()
            tier = text.split('name = "words"', 1)[1].split("item [", 1)[0]
            intervals = [
                (float(a), float(b))
                for a, b, word in re.findall(
                    r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',
                    tier,
                )
                if word.strip()
            ]
            matrix = matrices[arm + "_source"]
            n = len(matrix)
            center = np.arange(n) / 25 + 0.1
            mask = np.zeros(n, dtype=bool)
            for a, b in intervals:
                mask |= (center >= a) & (center < b)
            out["source"][arm] = {
                "textgrid": info["textgrid"],
                "sha256": info["textgrid_sha256"],
                "speech_seconds": sum(b - a for a, b in intervals),
                "duration_seconds": row["source_lengths"][arm] / 16000,
            }
            for guard in (0, 15, 20):
                inds = np.arange(guard, n - guard)
                speech = inds[mask[inds]]
                assert len(speech) > 0
                cell = {
                    "speech_rows": speech.tolist(),
                    "all_rows": inds.tolist(),
                    "speech_fraction": len(speech) / len(inds),
                }
                for mode, indices in [("all", inds), ("speech", speech)]:
                    curve = matrix[indices].mean(0)
                    d = float(curve.min())
                    b = float(np.median(curve))
                    cell[mode] = {"C": b - d, "B": b, "D": d}
                out["cells"][f"{arm}_{guard}"] = cell
        rows.append(out)
    result = {"guards": {}}
    for guard in (0, 15, 20):
        values = {
            mode: np.array(
                [
                    r["cells"][f"T_{guard}"][mode]["C"]
                    - r["cells"][f"N_{guard}"][mode]["C"]
                    for r in rows
                ]
            )
            for mode in ("all", "speech")
        }
        values["speech_minus_all"] = values["speech"] - values["all"]
        values["speech_fraction_T_minus_N"] = np.array(
            [
                r["cells"][f"T_{guard}"]["speech_fraction"]
                - r["cells"][f"N_{guard}"]["speech_fraction"]
                for r in rows
            ]
        )
        result["guards"][str(guard)] = {
            key: cluster(x.tolist(), [r["speaker"] for r in rows])
            for key, x in values.items()
        }
    write(base / "scores.json", rows)
    write(base / "analysis.json", result)
    for key, v in result["guards"]["20"].items():
        print(key, v["speaker_mean"], v["speaker_ci95"])


def joint():
    base = OUT / "speech_windows"
    protocol = {
        "parent_sha256": sha(base / "protocol.json"),
        "speed_protocol_sha256": sha(OUT / "speed/protocol.json"),
        "reason": "Adaptive combination after separate speed and speech-window controls each reduced only part of the gap; freeze before inspecting combined result.",
        "definition": "Existing identity/equal-common/swap matrices; speech centers mapped back to original audio time using the frozen atempo factor; old source-audio-bound MFA word intervals; guards0/15/20; all74, no outcome exclusion.",
        "limits": "Exploratory combination, processed-media artifacts and MFA errors remain; not a causal partition of total native gain.",
    }
    path = base / "joint_protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("joint protocol differs")
    write(path, protocol)
    rows = []
    for old in read(base / "scores.json"):
        sid = old["id"]
        receipt = read(OUT / "speed/assets" / sid / "receipt.json")
        score = read(OUT / "speed/scores" / f"{sid}.json")
        fp = OUT / "speed/scores" / f"{sid}.npz"
        check_hash(fp, score["matrix_sha256"])
        matrices = np.load(fp)
        out = {"id": sid, "speaker": old["speaker"], "cells": {}}
        for arm in ("N", "T"):
            info = old["source"][arm]
            check_hash(info["textgrid"], info["sha256"])
            tier = (
                Path(info["textgrid"])
                .read_text()
                .split('name = "words"', 1)[1]
                .split("item [", 1)[0]
            )
            intervals = [
                (float(a), float(b))
                for a, b, word in re.findall(
                    r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',
                    tier,
                )
                if word.strip()
            ]
            for mode in ("identity", "equal_common", "swap"):
                key = arm + "_" + mode
                speed = receipt[key.removesuffix("_common")]["speed"]
                matrix = matrices[key]
                centers = (np.arange(len(matrix)) / 25 + 0.1) * speed
                mask = np.zeros(len(matrix), dtype=bool)
                for a, b in intervals:
                    mask |= (centers >= a) & (centers < b)
                for guard in (0, 15, 20):
                    inds = np.arange(guard, len(matrix) - guard)
                    inds = inds[mask[inds]]
                    assert len(inds) > 0
                    curve = matrix[inds].mean(0)
                    out["cells"][f"{key}_{guard}"] = {
                        "rows": inds.tolist(),
                        "C": float(np.median(curve) - curve.min()),
                    }
        rows.append(out)
    result = {"guards": {}}
    for guard in (0, 15, 20):
        values = {
            mode: np.array(
                [
                    r["cells"][f"T_{mode}_{guard}"]["C"]
                    - r["cells"][f"N_{mode}_{guard}"]["C"]
                    for r in rows
                ]
            )
            for mode in ("identity", "equal_common", "swap")
        }
        result["guards"][str(guard)] = {
            m: cluster(x.tolist(), [r["speaker"] for r in rows])
            for m, x in values.items()
        }
    write(base / "joint_scores.json", rows)
    write(base / "joint_analysis.json", result)
    for k, v in result["guards"]["20"].items():
        print("joint", k, v["speaker_mean"], v["speaker_ci95"])


if __name__ == "__main__":
    main()
    joint()
