#!/usr/bin/env python3
"""Fresh three-arm n15 evaluation with one fixed face for every sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scipy import stats

REPO = Path(__file__).resolve().parent.parent
WAV2LIP = REPO / "third_party/Wav2Lip"
SYNCNET = REPO / "third_party/syncnet_python"
WAV2LIP_PY = Path.home() / ".venvs/wav2lip/bin/python"
SYNCNET_PY = Path.home() / ".venvs/syncnet/bin/python"
WAV2LIP_CHECKPOINT = WAV2LIP / "checkpoints/wav2lip_gan.pth"
SYNCNET_MODEL = SYNCNET / "data/syncnet_v2.model"
MIN_TRACK = 50
ARMS = ("natural_raw", "raw_tts", "mfa_linear")
EXPECTED_WAV2LIP_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
EXPECTED_SYNCNET_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parse_syncnet(path: Path) -> dict[str, float | int] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s+([0-9.]+)", text)
    distance = re.search(r"Min dist:\s+([0-9.]+)", text)
    offset = re.search(r"AV offset:\s+(-?\d+)", text)
    if confidence is None or distance is None:
        return None
    return {
        "sync_c": float(confidence.group(1)),
        "sync_d": float(distance.group(1)),
        "av_offset": int(offset.group(1)) if offset else 0,
    }


def run_command(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    return int(result.returncode)


def finite_score(row: Mapping[str, Any]) -> bool:
    return all(isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key])) for key in ("sync_c", "sync_d", "av_offset"))


def build_records(counterfactual_path: Path, mfa_summary_path: Path, face: Path) -> list[dict[str, Any]]:
    if file_sha256(WAV2LIP_CHECKPOINT) != EXPECTED_WAV2LIP_SHA256 or file_sha256(SYNCNET_MODEL) != EXPECTED_SYNCNET_SHA256:
        raise ValueError("evaluator model hashes changed")
    face = face.resolve()
    if not face.is_file():
        raise FileNotFoundError(face)
    counter = json.loads(counterfactual_path.read_text(encoding="utf-8"))
    mfa = json.loads(mfa_summary_path.read_text(encoding="utf-8"))
    if counter.get("heldout_excluded") is not True or counter.get("valid_speaker") != "S0765":
        raise ValueError("counterfactual source is not the original valid-only n15 cohort")
    counter_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in counter.get("records", []):
        key = (int(row["sample_id"]), str(row["condition"]))
        if str(row["condition"]) in ("natural_noop", "tts_noop") and int(row["sample_id"]) <= 15:
            if key in counter_by_key:
                raise ValueError(f"duplicate counterfactual record {key}")
            counter_by_key[key] = row
    if len(counter_by_key) != 30:
        raise ValueError("counterfactual source does not contain 15 natural/TTS pairs")
    if mfa.get("status") != "completed" or mfa.get("selection", {}).get("selected_count") != 15 or mfa.get("conditions") != ["paired_tts_mfa_linear"]:
        raise ValueError("MFA-linear source is incomplete")
    mfa_by_id: dict[int, Mapping[str, Any]] = {}
    for item in mfa.get("items", []):
        conditions = [row for row in item.get("conditions", []) if row.get("condition") == "paired_tts_mfa_linear"]
        if len(conditions) != 1:
            raise ValueError(f"missing unique MFA-linear condition at sample {item.get('sample_id')}")
        mfa_by_id[int(item["sample_id"])] = {
            **conditions[0],
            "paired_key": item["paired_key"],
            "sample_id": item["sample_id"],
        }
    if len(mfa_by_id) != 15 or set(mfa_by_id) != set(range(1, 16)):
        raise ValueError("MFA-linear source does not contain samples 1..15")
    face_hash = file_sha256(face)
    records: list[dict[str, Any]] = []
    for sample_id in range(1, 16):
        natural = counter_by_key[(sample_id, "natural_noop")]
        raw_tts = counter_by_key[(sample_id, "tts_noop")]
        mfa_row = mfa_by_id[sample_id]
        if natural["paired_key"] != raw_tts["paired_key"] or natural["paired_key"] != mfa_row["paired_key"]:
            raise ValueError(f"paired key mismatch at sample {sample_id}")
        if natural["speaker_id"] != raw_tts["speaker_id"] or natural["speaker_id"] != "S0765":
            raise ValueError(f"speaker mismatch at sample {sample_id}")
        for arm, row, condition in (("natural_raw", natural, "natural_noop"), ("raw_tts", raw_tts, "tts_noop")):
            audio = Path(str(row["output_path"])).resolve()
            if not audio.is_file() or file_sha256(audio) != str(row["output_sha256"]):
                raise ValueError(f"{arm} audio hash mismatch at sample {sample_id}")
            records.append({
                "sample_id": sample_id,
                "paired_key": str(natural["paired_key"]),
                "speaker_id": str(natural["speaker_id"]),
                "split": str(natural["split"]),
                "transcript": str(natural.get("transcript", "")),
                "arm": arm,
                "source_condition": condition,
                "audio": str(audio),
                "audio_sha256": str(row["output_sha256"]),
                "face": str(face),
                "face_sha256": face_hash,
                "cache_reused": False,
                "counterfactual_summary": str(counterfactual_path.resolve()),
                "counterfactual_summary_sha256": file_sha256(counterfactual_path),
            })
        mfa_audio = Path(str(mfa_row["output_path"])).resolve()
        if not mfa_audio.is_file() or file_sha256(mfa_audio) != str(mfa_row["output_sha256"]):
            raise ValueError(f"mfa_linear audio hash mismatch at sample {sample_id}")
        records.append({
            "sample_id": sample_id,
            "paired_key": str(natural["paired_key"]),
            "speaker_id": str(natural["speaker_id"]),
            "split": str(natural["split"]),
            "transcript": str(natural.get("transcript", "")),
            "arm": "mfa_linear",
            "source_condition": "paired_tts_mfa_linear",
            "audio": str(mfa_audio),
            "audio_sha256": str(mfa_row["output_sha256"]),
            "face": str(face),
            "face_sha256": face_hash,
            "cache_reused": False,
            "mfa_summary": str(mfa_summary_path.resolve()),
            "mfa_summary_sha256": file_sha256(mfa_summary_path),
        })
    if len(records) != 45:
        raise ValueError(f"expected 45 records, found {len(records)}")
    return records


def evaluate(records: Sequence[Mapping[str, Any]], output_dir: Path, face_label: str) -> dict[str, Any]:
    if len(records) != 45:
        raise ValueError("fixed-face n15 evaluation requires 45 records")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        sample_id = int(record["sample_id"])
        arm = str(record["arm"])
        audio = Path(str(record["audio"]))
        face = Path(str(record["face"]))
        video = output_dir / "wav2lip" / arm / f"{sample_id}.mp4"
        cell_cwd = output_dir / "wav2lip_work" / arm / str(sample_id)
        (cell_cwd / "temp").mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        wav_log = output_dir / "logs" / arm / f"{sample_id}.wav2lip.log"
        rc = run_command([
            str(WAV2LIP_PY), str(WAV2LIP / "inference.py"), "--checkpoint_path", str(WAV2LIP_CHECKPOINT),
            "--face", str(face), "--audio", str(audio), "--outfile", str(video),
            "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth",
        ], cwd=cell_cwd, log=wav_log)
        if rc != 0 or not video.is_file():
            failures.append({"sample_id": sample_id, "arm": arm, "stage": "wav2lip", "returncode": rc})
            continue
        sync_dir = output_dir / "syncnet" / arm / str(sample_id)
        sync_dir.mkdir(parents=True, exist_ok=True)
        reference = f"n15_same_{face_label}_{arm}_{sample_id}"
        pipeline_log = output_dir / "logs" / arm / f"{sample_id}.pipeline.log"
        rc_pipeline = run_command([
            str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(video), "--reference", reference,
            "--data_dir", str(sync_dir), "--min_track", str(MIN_TRACK), "--overwrite",
        ], cwd=SYNCNET, log=pipeline_log)
        if rc_pipeline != 0:
            failures.append({"sample_id": sample_id, "arm": arm, "stage": "syncnet_pipeline", "returncode": rc_pipeline})
            continue
        sync_log = output_dir / "logs" / arm / f"{sample_id}.syncnet.log"
        rc_sync = run_command([
            str(SYNCNET_PY), "run_syncnet.py", "--videofile", str(video), "--reference", reference,
            "--data_dir", str(sync_dir), "--initial_model", str(SYNCNET_MODEL),
        ], cwd=SYNCNET, log=sync_log)
        parsed = parse_syncnet(sync_log)
        if rc_sync != 0 or parsed is None:
            failures.append({"sample_id": sample_id, "arm": arm, "stage": "syncnet_score", "returncode": rc_sync})
            continue
        scores.append({
            **record,
            "face_protocol": face_label,
            "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
            "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
            "min_track": MIN_TRACK,
            "video": str(video),
            "video_sha256": file_sha256(video),
            **parsed,
        })
        print(f"OK {index}/{len(records)} {arm} sample={sample_id} face={face_label} C={parsed['sync_c']:.3f} D={parsed['sync_d']:.3f}", flush=True)
    complete = not failures and len(scores) == 45
    summary = {
        "schema_version": 1,
        "evaluation": "valid15_same_face_three_arm_wav2lip_syncnet",
        "status": "complete" if complete else "incomplete",
        "face_protocol": face_label,
        "same_face_for_all_samples": True,
        "paired_same_utterance": True,
        "sample_count": 15,
        "expected_scores": 45,
        "arms": list(ARMS),
        "min_track": MIN_TRACK,
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
        "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
        "scores": scores,
        "failures": failures,
    }
    write_json(output_dir / "summary.json", summary)
    if not complete:
        raise RuntimeError(f"fixed-face evaluation incomplete: {len(scores)}/45; failures={len(failures)}")
    return summary


def bootstrap(values: Sequence[float], seed: int, draws: int = 10000) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(draws))
    return [estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]]


def analyze(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "complete" or len(summary.get("scores", [])) != 45 or summary.get("failures"):
        raise ValueError("analysis requires a complete 45-score matrix")
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = {arm: {} for arm in ARMS}
    for row in summary["scores"]:
        sample_id = int(row["sample_id"])
        arm = str(row["arm"])
        if sample_id in by_arm[arm] or not finite_score(row):
            raise ValueError("duplicate or non-finite score")
        by_arm[arm][sample_id] = row
    shared = sorted(set.intersection(*(set(by_arm[arm]) for arm in ARMS)))
    if shared != list(range(1, 16)):
        raise ValueError("analysis does not have 15 complete paired samples")
    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        rows = [by_arm[arm][sample_id] for sample_id in shared]
        arm_summary[arm] = {
            "n": 15,
            "mean_sync_c": statistics.fmean(float(row["sync_c"]) for row in rows),
            "sd_sync_c": statistics.stdev(float(row["sync_c"]) for row in rows),
            "mean_sync_d": statistics.fmean(float(row["sync_d"]) for row in rows),
            "sd_sync_d": statistics.stdev(float(row["sync_d"]) for row in rows),
            "mean_av_offset": statistics.fmean(float(row["av_offset"]) for row in rows),
        }
    comparisons: dict[str, Any] = {}
    for offset, baseline in enumerate(("natural_raw", "raw_tts")):
        dc = [float(by_arm["mfa_linear"][sample_id]["sync_c"]) - float(by_arm[baseline][sample_id]["sync_c"]) for sample_id in shared]
        dd = [float(by_arm["mfa_linear"][sample_id]["sync_d"]) - float(by_arm[baseline][sample_id]["sync_d"]) for sample_id in shared]
        comparisons[f"mfa_linear_vs_{baseline}"] = {
            "n": 15,
            "mean_delta_sync_c": statistics.fmean(dc),
            "bootstrap_95_ci_delta_sync_c": bootstrap(dc, 42 + offset * 10),
            "mean_delta_sync_d": statistics.fmean(dd),
            "bootstrap_95_ci_delta_sync_d": bootstrap(dd, 43 + offset * 10),
            "sync_c_better_count": sum(value > 0 for value in dc),
            "sync_d_better_count": sum(value < 0 for value in dd),
            "joint_better_count": sum(c > 0 and d < 0 for c, d in zip(dc, dd, strict=True)),
            "sync_c_paired_t_p": float(stats.ttest_1samp(dc, 0.0).pvalue),
            "sync_d_paired_t_p": float(stats.ttest_1samp(dd, 0.0).pvalue),
            "per_sample": {str(sample_id): {"delta_sync_c": dc[index], "delta_sync_d": dd[index]} for index, sample_id in enumerate(shared)},
        }
    return {
        "schema_version": 1,
        "analysis": "valid15_same_face_paired_exploratory",
        "scope": "n=15 S0765 paired utterances; one fixed face for all samples; no heldout/generalization claim",
        "face_protocol": summary["face_protocol"],
        "directions": {"sync_c": "higher_is_better", "sync_d": "lower_is_better"},
        "arm_summary": arm_summary,
        "comparisons": comparisons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-summary", type=Path, required=True)
    parser.add_argument("--mfa-summary", type=Path, required=True)
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--face-label", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {outdir}")
    records = build_records(args.counterfactual_summary.resolve(), args.mfa_summary.resolve(), args.face.resolve())
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "manifest.json", {
        "schema_version": 1,
        "manifest_type": "valid15_same_face_three_arm_wav2lip_syncnet",
        "face_protocol": args.face_label,
        "same_face_for_all_samples": True,
        "paired_same_utterance": True,
        "face": str(args.face.resolve()),
        "face_sha256": file_sha256(args.face.resolve()),
        "counterfactual_summary": str(args.counterfactual_summary.resolve()),
        "counterfactual_summary_sha256": file_sha256(args.counterfactual_summary.resolve()),
        "mfa_summary": str(args.mfa_summary.resolve()),
        "mfa_summary_sha256": file_sha256(args.mfa_summary.resolve()),
        "expected_scores": 45,
        "arms": list(ARMS),
        "min_track": MIN_TRACK,
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
        "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
        "records": records,
    })
    summary = evaluate(records, outdir, args.face_label)
    write_json(outdir / "analysis.json", analyze(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
