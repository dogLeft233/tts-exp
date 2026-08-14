#!/usr/bin/env python3
"""Evaluate valid-only rhythm counterfactuals with local Wav2Lip and SyncNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs/two_stage_hubert_aishell1_20260810"
DEFAULT_INPUT = REPO / "runs/rhythm_timing/20260812_counterfactual_valid50/summary.json"
DEFAULT_OUTPUT = REPO / "runs/rhythm_timing/20260812_syncnet_smoke10"
FACE_DIR = RUN / "ditto_videos/natural_raw"
MAPPING = RUN / "ditto_videos/sample_mapping.json"
WAV2LIP = REPO / "third_party/Wav2Lip"
SYNCNET = REPO / "third_party/syncnet_python"
WAV2LIP_PY = Path.home() / ".venvs/wav2lip/bin/python"
SYNCNET_PY = Path.home() / ".venvs/syncnet/bin/python"
WAV2LIP_CHECKPOINT = WAV2LIP / "checkpoints/wav2lip_gan.pth"
SYNCNET_MODEL = SYNCNET / "data/syncnet_v2.model"
MIN_TRACK = 50
VALID_SPEAKER = "S0765"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    return int(result.returncode)


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


def finite_score(result: dict[str, Any]) -> bool:
    return all(
        isinstance(result.get(key), (int, float)) and math.isfinite(float(result[key]))
        for key in ("sync_c", "sync_d")
    )


def cached_score(path: Path, audio_sha256: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("audio_sha256") != audio_sha256 or not finite_score(result):
        return None
    return result


def wav2lip_work_dir(output_dir: Path, condition: str, sample_id: int) -> Path:
    """Create a per-cell cwd so Wav2Lip's hard-coded temp path is isolated."""
    work_dir = output_dir / "wav2lip_work" / condition / str(sample_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    for name in ("audio.py", "hparams.py", "inference.py", "models", "face_detection"):
        source = WAV2LIP / name
        target = work_dir / name
        if not target.exists():
            target.symlink_to(source)
    (work_dir / "temp").mkdir(exist_ok=True)
    return work_dir


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for score in scores:
        by_condition.setdefault(str(score["condition"]), []).append(score)
    summary: dict[str, Any] = {}
    for condition, rows in sorted(by_condition.items()):
        summary[condition] = {
            "count": len(rows),
            "mean_sync_c": sum(float(row["sync_c"]) for row in rows) / len(rows),
            "mean_sync_d": sum(float(row["sync_d"]) for row in rows) / len(rows),
            "mean_av_offset": sum(float(row["av_offset"]) for row in rows) / len(rows),
        }
    baseline = by_condition.get("tts_noop", [])
    baseline_by_id = {int(row["sample_id"]): row for row in baseline}
    for condition, rows in by_condition.items():
        deltas_c: list[float] = []
        deltas_d: list[float] = []
        for row in rows:
            base = baseline_by_id.get(int(row["sample_id"]))
            if base is None:
                continue
            deltas_c.append(float(row["sync_c"]) - float(base["sync_c"]))
            deltas_d.append(float(row["sync_d"]) - float(base["sync_d"]))
        summary[condition]["paired_vs_tts_noop"] = {
            "count": len(deltas_c),
            "mean_delta_sync_c": sum(deltas_c) / len(deltas_c) if deltas_c else None,
            "mean_delta_sync_d": sum(deltas_d) / len(deltas_d) if deltas_d else None,
        }
    return summary


def evaluate(input_summary: Path, output_dir: Path, count: int) -> dict[str, Any]:
    source = json.loads(input_summary.read_text(encoding="utf-8"))
    mapping = {int(row["id"]): row for row in json.loads(MAPPING.read_text(encoding="utf-8"))}
    all_records = [record for record in source["records"] if record.get("speaker_id") == VALID_SPEAKER]
    sample_ids = sorted({int(record["sample_id"]) for record in all_records})[:count]
    records = [record for record in all_records if int(record["sample_id"]) in sample_ids]
    conditions = sorted({str(record["condition"]) for record in records})
    expected = count * len(conditions)
    if len(records) != expected:
        raise ValueError(f"expected {expected} source records, found {len(records)}")
    required = [WAV2LIP_PY, SYNCNET_PY, WAV2LIP_CHECKPOINT, SYNCNET_MODEL]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing evaluator assets: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(sorted(records, key=lambda row: (int(row["sample_id"]), str(row["condition"]))), 1):
        sample_id = int(record["sample_id"])
        condition = str(record["condition"])
        audio = Path(record["output_path"])
        if not audio.is_absolute():
            audio = (REPO / audio).resolve()
        face = FACE_DIR / f"{sample_id}.mp4"
        if sample_id not in mapping or mapping[sample_id].get("paired_key") != record.get("paired_key"):
            failures.append({"sample_id": sample_id, "condition": condition, "stage": "mapping"})
            continue
        if not audio.exists() or not face.exists():
            failures.append({"sample_id": sample_id, "condition": condition, "stage": "input"})
            continue
        audio_sha256 = sha256_file(audio)
        score_path = output_dir / "scores" / condition / f"{sample_id}.json"
        cached = cached_score(score_path, audio_sha256)
        if cached is not None:
            scores.append(cached)
            print(f"CACHED {index}/{expected} {condition} {sample_id}", flush=True)
            continue

        video = output_dir / "wav2lip" / condition / f"{sample_id}.mp4"
        wav_log = output_dir / "logs" / condition / f"{sample_id}.wav2lip.log"
        video.parent.mkdir(parents=True, exist_ok=True)
        cell_work_dir = wav2lip_work_dir(output_dir, condition, sample_id)
        rc_wav = run_command(
            [
                str(WAV2LIP_PY),
                str(WAV2LIP / "inference.py"),
                "--checkpoint_path",
                str(WAV2LIP_CHECKPOINT),
                "--face",
                str(face),
                "--audio",
                str(audio),
                "--outfile",
                str(video),
                "--face_det_batch_size",
                "4",
                "--wav2lip_batch_size",
                "4",
                "--nosmooth",
            ],
            cwd=cell_work_dir,
            log=wav_log,
        )
        if rc_wav != 0 or not video.exists():
            failures.append({"sample_id": sample_id, "condition": condition, "stage": "wav2lip", "returncode": rc_wav})
            print(f"FAIL {index}/{expected} {condition} {sample_id} wav2lip", flush=True)
            continue

        sync_dir = output_dir / "syncnet" / condition / str(sample_id)
        sync_dir.mkdir(parents=True, exist_ok=True)
        sync_log = output_dir / "logs" / condition / f"{sample_id}.syncnet.log"
        reference = f"rhythm_{condition}_{sample_id}"
        rc_pipeline = run_command(
            [
                str(SYNCNET_PY),
                "run_pipeline.py",
                "--videofile",
                str(video),
                "--reference",
                reference,
                "--data_dir",
                str(sync_dir),
                "--min_track",
                str(MIN_TRACK),
                "--overwrite",
            ],
            cwd=SYNCNET,
            log=sync_log.with_suffix(".pipeline.log"),
        )
        if rc_pipeline != 0:
            failures.append({"sample_id": sample_id, "condition": condition, "stage": "syncnet_pipeline", "returncode": rc_pipeline})
            print(f"FAIL {index}/{expected} {condition} {sample_id} pipeline", flush=True)
            continue
        rc_sync = run_command(
            [
                str(SYNCNET_PY),
                "run_syncnet.py",
                "--videofile",
                str(video),
                "--reference",
                reference,
                "--data_dir",
                str(sync_dir),
                "--initial_model",
                str(SYNCNET_MODEL),
            ],
            cwd=SYNCNET,
            log=sync_log,
        )
        parsed = parse_syncnet(sync_log)
        if rc_sync != 0 or parsed is None:
            failures.append({"sample_id": sample_id, "condition": condition, "stage": "syncnet_score", "returncode": rc_sync})
            print(f"FAIL {index}/{expected} {condition} {sample_id} score", flush=True)
            continue
        result = {
            "schema_version": 1,
            "sample_id": sample_id,
            "paired_key": record["paired_key"],
            "speaker_id": VALID_SPEAKER,
            "split": "valid",
            "condition": condition,
            "audio": str(audio),
            "audio_sha256": audio_sha256,
            "face": str(face),
            "face_sha256": sha256_file(face),
            "video": str(video),
            "video_sha256": sha256_file(video),
            "min_track": MIN_TRACK,
            **parsed,
        }
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        scores.append(result)
        print(f"OK {index}/{expected} {condition} {sample_id} C={result['sync_c']:.3f} D={result['sync_d']:.3f}", flush=True)

    payload = {
        "schema_version": 1,
        "evaluation": "valid_only_rhythm_counterfactual_wav2lip_syncnet",
        "input_summary": str(input_summary.resolve()),
        "input_summary_sha256": sha256_file(input_summary),
        "valid_speaker": VALID_SPEAKER,
        "heldout_excluded": True,
        "sample_ids": sample_ids,
        "conditions": conditions,
        "expected_scores": expected,
        "min_track": MIN_TRACK,
        "wav2lip_checkpoint": str(WAV2LIP_CHECKPOINT),
        "wav2lip_checkpoint_sha256": sha256_file(WAV2LIP_CHECKPOINT),
        "syncnet_model": str(SYNCNET_MODEL),
        "syncnet_model_sha256": sha256_file(SYNCNET_MODEL),
        "scores": scores,
        "failures": failures,
        "aggregate": aggregate(scores),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE scores={len(scores)} failures={len(failures)} output={output_dir / 'summary.json'}", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.count <= 50:
        raise SystemExit("--count must be in [1, 50]")
    input_summary = args.input_summary if args.input_summary.is_absolute() else (REPO / args.input_summary).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (REPO / args.output_dir).resolve()
    result = evaluate(input_summary, output_dir, args.count)
    return 0 if not result["failures"] and len(result["scores"]) == result["expected_scores"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
