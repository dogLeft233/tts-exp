#!/usr/bin/env python3
"""Wav2Lip + SyncNet evaluation for the RAMC/AliMeeting pilot.

One fixed reference face video for all samples and arms (single-face
protocol, recorded in the manifest). Each sample/arm cell runs Wav2Lip in
its own isolated cwd/temp (third_party/Wav2Lip hardcodes relative temp/).

Arms are read from the pilot manifest:
  natural_raw  -> manifest record audio_path
  raw_tts      -> tts_meta.json canonical_16k_audio
  mfa_linear   -> <outdir>/../mfa_linear/{sample_id}.wav (optional arm)

Requires exactly sample_count x len(arms) finite scores and zero failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


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


def _finite_score(row: Mapping[str, Any]) -> bool:
    return all(isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key])) for key in ("sync_c", "sync_d", "av_offset"))


def build_records(manifest: Mapping[str, Any], tts_meta: Mapping[str, Any], face: Path, mfa_dir: Path | None, arms: Sequence[str]) -> list[dict[str, Any]]:
    if file_sha256(WAV2LIP_CHECKPOINT) != EXPECTED_WAV2LIP_SHA256 or file_sha256(SYNCNET_MODEL) != EXPECTED_SYNCNET_SHA256:
        raise ValueError("evaluator model hashes changed")
    face = face.resolve()
    if not face.is_file():
        raise FileNotFoundError(face)
    tts_by_id = {str(row["sample_id"]): row for row in tts_meta.get("results", {}).values()}
    records: list[dict[str, Any]] = []
    for row in manifest["records"]:
        sample_id = str(row["sample_id"])
        for arm in arms:
            if arm == "natural_raw":
                audio = Path(row["audio_path"]).resolve()
                source_condition = "natural_raw"
            elif arm == "raw_tts":
                tts_row = tts_by_id[sample_id]
                audio = Path(tts_row["canonical_16k_audio"]).resolve()
                source_condition = "raw_tts"
            elif arm == "mfa_linear":
                if mfa_dir is None:
                    raise ValueError("mfa_linear arm requires --mfa-dir")
                audio = (mfa_dir / f"{sample_id}.wav").resolve()
                source_condition = "mfa_linear"
            else:
                raise ValueError(f"unsupported arm: {arm}")
            if not audio.is_file():
                raise FileNotFoundError(audio)
            records.append({
                "sample_id": sample_id, "arm": arm, "source_condition": source_condition,
                "speaker_id": row["speaker_id"], "session": row.get("session") or row.get("conversation"),
                "transcript": row["transcript"], "dataset": manifest["dataset"],
                "audio": str(audio), "audio_sha256": file_sha256(audio),
                "face": str(face), "face_sha256": file_sha256(face),
            })
    return records


def evaluate(records: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        arm = str(record["arm"])
        sample_id = str(record["sample_id"])
        audio = Path(record["audio"])
        face = Path(record["face"])
        video = output_dir / "wav2lip" / arm / f"{sample_id}.mp4"
        cell_cwd = output_dir / "wav2lip_work" / arm / sample_id
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
        sync_dir = output_dir / "syncnet" / arm / sample_id
        sync_dir.mkdir(parents=True, exist_ok=True)
        reference = f"pilot_{arm}_{sample_id}"
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
        score = {
            **{key: record[key] for key in ("sample_id", "arm", "source_condition", "speaker_id", "session", "dataset", "transcript", "audio", "audio_sha256", "face", "face_sha256")},
            "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
            "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256, "min_track": MIN_TRACK,
            "video": str(video), "video_sha256": file_sha256(video), **parsed,
        }
        score_path = output_dir / "scores" / arm / f"{sample_id}.json"
        write_json(score_path, score)
        scores.append(score)
        print(f"OK {index}/{len(records)} {arm} {sample_id} C={score['sync_c']:.3f} D={score['sync_d']:.3f}", flush=True)
    complete = not failures and len(scores) == len(records)
    summary = {
        "schema_version": 1, "evaluation": "pilot_wav2lip_syncnet",
        "status": "complete" if complete else "incomplete",
        "single_face_protocol": True, "min_track": MIN_TRACK,
        "expected_scores": len(records), "scores": scores, "failures": failures,
    }
    write_json(output_dir / "summary.json", summary)
    if not complete:
        raise RuntimeError(f"pilot matrix incomplete: {len(scores)}/{len(records)}, failures={failures}")
    return summary


def _bootstrap(values: list[float], seed: int, draws: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(draws))
    return [estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]]


def analyze(summary: Mapping[str, Any], arms: Sequence[str]) -> dict[str, Any]:
    if summary.get("status") != "complete" or summary.get("failures"):
        raise ValueError("analysis requires a complete matrix")
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in summary["scores"]:
        arm = str(row["arm"]); sample_id = str(row["sample_id"])
        if sample_id in by_arm.setdefault(arm, {}) or not _finite_score(row):
            raise ValueError("duplicate or non-finite score")
        by_arm[arm][sample_id] = row
    shared = sorted(set.intersection(*(set(by_arm[arm]) for arm in arms)))
    if not shared:
        raise ValueError("no complete paired samples")
    arm_summary = {}
    for arm in arms:
        rows = [by_arm[arm][sid] for sid in shared]
        arm_summary[arm] = {
            "n": len(shared),
            "mean_sync_c": statistics.fmean(float(r["sync_c"]) for r in rows),
            "sd_sync_c": statistics.stdev(float(r["sync_c"]) for r in rows),
            "mean_sync_d": statistics.fmean(float(r["sync_d"]) for r in rows),
            "sd_sync_d": statistics.stdev(float(r["sync_d"]) for r in rows),
            "mean_av_offset": statistics.fmean(float(r["av_offset"]) for r in rows),
        }
    comparisons = {}
    if "natural_raw" in arms:
        for target in [a for a in arms if a != "natural_raw"]:
            delta_c = [float(by_arm[target][sid]["sync_c"]) - float(by_arm["natural_raw"][sid]["sync_c"]) for sid in shared]
            delta_d = [float(by_arm[target][sid]["sync_d"]) - float(by_arm["natural_raw"][sid]["sync_d"]) for sid in shared]
            comparisons[f"{target}_vs_natural_raw"] = {
                "n": len(shared),
                "mean_delta_sync_c": statistics.fmean(delta_c),
                "bootstrap_95_ci_delta_sync_c": _bootstrap(delta_c, 42),
                "mean_delta_sync_d": statistics.fmean(delta_d),
                "bootstrap_95_ci_delta_sync_d": _bootstrap(delta_d, 43),
                "sync_c_better_count": sum(v > 0 for v in delta_c),
                "sync_d_better_count": sum(v < 0 for v in delta_d),
                "joint_better_count": sum(c > 0 and d < 0 for c, d in zip(delta_c, delta_d, strict=True)),
                "sync_c_paired_t_p": float(stats.ttest_1samp(delta_c, 0.0).pvalue),
                "sync_d_paired_t_p": float(stats.ttest_1samp(delta_d, 0.0).pvalue),
                "per_sample": {sid: {"delta_sync_c": delta_c[i], "delta_sync_d": delta_d[i]} for i, sid in enumerate(shared)},
            }
    payload = {
        "schema_version": 1, "analysis": "pilot_paired_exploratory",
        "scope": "pilot exploratory; no heldout/generalization claim",
        "directions": {"sync_c": "higher_is_better", "sync_d": "lower_is_better"},
        "arm_summary": arm_summary, "comparisons": comparisons,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mfa-dir", type=Path, default=None)
    parser.add_argument("--arms", default="natural_raw,raw_tts")
    args = parser.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tts_meta = json.loads(args.tts_meta.read_text(encoding="utf-8"))
    records = build_records(manifest, tts_meta, args.face, args.mfa_dir, arms)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_json(args.outdir / "records.json", {"records": records, "arms": arms})
    summary = evaluate(records, args.outdir)
    analysis = analyze(summary, arms)
    write_json(args.outdir / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
