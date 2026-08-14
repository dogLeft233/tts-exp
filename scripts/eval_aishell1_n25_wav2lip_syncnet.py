#!/usr/bin/env python3
"""Fresh three-arm Wav2Lip + SyncNet evaluation for AISHELL-1 n=25."""

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
    return {"sync_c": float(confidence.group(1)), "sync_d": float(distance.group(1)), "av_offset": int(offset.group(1)) if offset else 0}


def run_command(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    return int(result.returncode)


def finite_score(row: Mapping[str, Any]) -> bool:
    return all(isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key])) for key in ("sync_c", "sync_d", "av_offset"))


def build_records(cohort_path: Path, tts_meta_path: Path, mfa_summary_path: Path, face: Path) -> list[dict[str, Any]]:
    if file_sha256(WAV2LIP_CHECKPOINT) != EXPECTED_WAV2LIP_SHA256 or file_sha256(SYNCNET_MODEL) != EXPECTED_SYNCNET_SHA256:
        raise ValueError("evaluator model hashes changed")
    face = face.resolve()
    if not face.is_file():
        raise FileNotFoundError(face)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    tts_meta = json.loads(tts_meta_path.read_text(encoding="utf-8"))
    mfa = json.loads(mfa_summary_path.read_text(encoding="utf-8"))
    cohort_records = list(cohort.get("records", []))
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort" or len(cohort_records) != 25:
        raise ValueError("invalid n=25 cohort")
    if cohort.get("cohort", {}).get("heldout_excluded") is not True:
        raise ValueError("cohort does not exclude heldout")
    tts_by_id: dict[str, Mapping[str, Any]] = {}
    for row in tts_meta.get("results", {}).values():
        sample_id = str(row["sample_id"])
        if sample_id in tts_by_id:
            raise ValueError(f"duplicate TTS metadata sample_id {sample_id}")
        tts_by_id[sample_id] = row
    mfa_by_id = {str(row["sample_id"]): row for row in mfa.get("results", {}).values()}
    if len(tts_by_id) != 25 or len(mfa_by_id) != 25 or mfa.get("failures"):
        raise ValueError("TTS or MFA-linear metadata is incomplete")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    face_hash = file_sha256(face)
    for row in cohort_records:
        sample_id = str(row["sample_id"])
        speaker_id = str(row["speaker_id"])
        paired_key = str(row["paired_key"])
        tts = tts_by_id.get(sample_id)
        mfa_row = mfa_by_id.get(sample_id)
        if tts is None or mfa_row is None:
            raise ValueError(f"missing generated metadata for {paired_key}")
        if str(tts.get("paired_key")) != paired_key or str(tts.get("speaker_id")) != speaker_id:
            raise ValueError(f"TTS identity mismatch for {paired_key}")
        if str(tts.get("source_audio_sha256", "")) != str(row["tts_source_sha256"]):
            raise ValueError(f"TTS source is not the strict cohort source for {paired_key}")
        if str(mfa_row.get("paired_key", paired_key)) != paired_key or str(mfa_row.get("speaker_id", speaker_id)) != speaker_id:
            raise ValueError(f"MFA-linear identity mismatch for {paired_key}")
        paths = {
            "natural_raw": Path(row["audio_path"]).resolve(),
            "raw_tts": Path(str(tts["canonical_16k_audio"])).resolve(),
            "mfa_linear": Path(str(mfa_row["audio_path"])).resolve(),
        }
        hashes = {
            "natural_raw": str(row["natural_source_sha256"]),
            "raw_tts": str(tts["canonical_audio_sha256"]),
            "mfa_linear": str(mfa_row["audio_sha256"]),
        }
        for arm in ARMS:
            audio = paths[arm]
            if not audio.is_file() or file_sha256(audio) != hashes[arm]:
                raise ValueError(f"audio hash mismatch for {paired_key}/{arm}")
            identity = (speaker_id, paired_key, arm)
            if identity in seen:
                raise ValueError(f"duplicate evaluator identity {identity}")
            seen.add(identity)
            records.append({
                "sample_id": sample_id,
                "paired_key": paired_key,
                "speaker_id": speaker_id,
                "split": row["split"],
                "transcript": row["transcript"],
                "arm": arm,
                "source_condition": arm,
                "audio": str(audio),
                "audio_sha256": hashes[arm],
                "face": str(face),
                "face_sha256": face_hash,
                "cohort_manifest": str(cohort_path.resolve()),
                "cohort_manifest_sha256": file_sha256(cohort_path),
            })
    if len(records) != 75:
        raise ValueError(f"expected 75 evaluator records, found {len(records)}")
    return records


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != 75:
        raise ValueError("n=25 evaluator requires 75 records")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("arm") not in ARMS or row.get("speaker_id") == "S0770":
            raise ValueError("invalid arm or heldout speaker")
        grouped[str(row["paired_key"])].append(row)
    if len(grouped) != 25:
        raise ValueError("expected 25 unique paired keys")
    for key, rows in grouped.items():
        if {str(row["arm"]) for row in rows} != set(ARMS):
            raise ValueError(f"incomplete arm set for {key}")
        if len({str(row["speaker_id"]) for row in rows}) != 1 or len({str(row["face_sha256"]) for row in rows}) != 1:
            raise ValueError(f"identity/face mismatch for {key}")
    by_speaker = defaultdict(int)
    for row in records:
        if row["arm"] == "natural_raw":
            by_speaker[str(row["speaker_id"])] += 1
    if dict(by_speaker) != {speaker: 5 for speaker in sorted(by_speaker)} or len(by_speaker) != 5:
        raise ValueError(f"speaker cohort is not 5x5: {dict(by_speaker)}")


def evaluate(records: Sequence[Mapping[str, Any]], output_dir: Path, *, face_protocol: str) -> dict[str, Any]:
    validate_records(records)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        arm = str(record["arm"])
        sample_id = str(record["sample_id"])
        speaker_id = str(record["speaker_id"])
        key_slug = f"{speaker_id}_{sample_id}"
        audio = Path(record["audio"])
        face = Path(record["face"])
        video = output_dir / "wav2lip" / arm / speaker_id / f"{sample_id}.mp4"
        cell_cwd = output_dir / "wav2lip_work" / arm / speaker_id / sample_id
        (cell_cwd / "temp").mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        wav_log = output_dir / "logs" / arm / speaker_id / f"{sample_id}.wav2lip.log"
        rc = run_command([
            str(WAV2LIP_PY), str(WAV2LIP / "inference.py"), "--checkpoint_path", str(WAV2LIP_CHECKPOINT),
            "--face", str(face), "--audio", str(audio), "--outfile", str(video),
            "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth",
        ], cwd=cell_cwd, log=wav_log)
        if rc != 0 or not video.is_file():
            failures.append({"sample_id": sample_id, "paired_key": record["paired_key"], "speaker_id": speaker_id, "arm": arm, "stage": "wav2lip", "returncode": rc})
            continue
        sync_dir = output_dir / "syncnet" / arm / speaker_id / sample_id
        sync_dir.mkdir(parents=True, exist_ok=True)
        reference = f"aishell1_n25_{key_slug}_{arm}"
        pipeline_log = output_dir / "logs" / arm / speaker_id / f"{sample_id}.pipeline.log"
        rc_pipeline = run_command([
            str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(video), "--reference", reference,
            "--data_dir", str(sync_dir), "--min_track", str(MIN_TRACK), "--overwrite",
        ], cwd=SYNCNET, log=pipeline_log)
        if rc_pipeline != 0:
            failures.append({"sample_id": sample_id, "paired_key": record["paired_key"], "speaker_id": speaker_id, "arm": arm, "stage": "syncnet_pipeline", "returncode": rc_pipeline})
            continue
        sync_log = output_dir / "logs" / arm / speaker_id / f"{sample_id}.syncnet.log"
        rc_sync = run_command([
            str(SYNCNET_PY), "run_syncnet.py", "--videofile", str(video), "--reference", reference,
            "--data_dir", str(sync_dir), "--initial_model", str(SYNCNET_MODEL),
        ], cwd=SYNCNET, log=sync_log)
        parsed = parse_syncnet(sync_log)
        if rc_sync != 0 or parsed is None:
            failures.append({"sample_id": sample_id, "paired_key": record["paired_key"], "speaker_id": speaker_id, "arm": arm, "stage": "syncnet_score", "returncode": rc_sync})
            continue
        score = {
            **{key: record[key] for key in ("sample_id", "paired_key", "speaker_id", "split", "transcript", "arm", "source_condition", "audio", "audio_sha256", "face", "face_sha256", "cohort_manifest", "cohort_manifest_sha256")},
            "cache_reused": False,
            "face_protocol": face_protocol,
            "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
            "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
            "min_track": MIN_TRACK,
            "video": str(video),
            "video_sha256": file_sha256(video),
            **parsed,
        }
        write_json(output_dir / "scores" / arm / speaker_id / f"{sample_id}.json", score)
        scores.append(score)
        print(f"OK {index}/{len(records)} {speaker_id} {arm} {sample_id} C={score['sync_c']:.3f} D={score['sync_d']:.3f}", flush=True)
    complete = not failures and len(scores) == 75
    summary = {
        "schema_version": 1,
        "evaluation": "aishell1_mfa_linear_n25_wav2lip_syncnet",
        "status": "complete" if complete else "incomplete",
        "single_face_protocol": True,
        "face_protocol": face_protocol,
        "heldout_excluded": True,
        "sample_count": 25,
        "speaker_count": 5,
        "per_speaker_count": 5,
        "arms": list(ARMS),
        "expected_scores": 75,
        "min_track": MIN_TRACK,
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
        "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
        "scores": scores,
        "failures": failures,
    }
    write_json(output_dir / "summary.json", summary)
    if not complete:
        raise RuntimeError(f"evaluation incomplete: {len(scores)}/75; failures={len(failures)}")
    return summary


def cluster_bootstrap(values_by_speaker: Mapping[str, Sequence[float]], seed: int, draws: int = 10_000) -> list[float]:
    speakers = sorted(values_by_speaker)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        selected = [speakers[rng.randrange(len(speakers))] for _ in speakers]
        values = [value for speaker in selected for value in values_by_speaker[speaker]]
        estimates.append(statistics.fmean(values))
    estimates.sort()
    return [estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]]


def analyze(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "complete" or summary.get("failures") or len(summary.get("scores", [])) != 75:
        raise ValueError("analysis requires a complete 75-score matrix")
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = {arm: {} for arm in ARMS}
    for row in summary["scores"]:
        if not finite_score(row):
            raise ValueError("non-finite score")
        identity = str(row["paired_key"])
        arm = str(row["arm"])
        if identity in by_arm[arm]:
            raise ValueError("duplicate paired score")
        by_arm[arm][identity] = row
    shared = sorted(set.intersection(*(set(by_arm[arm]) for arm in ARMS)))
    if len(shared) != 25:
        raise ValueError("matrix is not paired across 25 utterances")
    arm_summary = {}
    for arm in ARMS:
        rows = [by_arm[arm][key] for key in shared]
        arm_summary[arm] = {
            "n": len(rows),
            "mean_sync_c": statistics.fmean(float(row["sync_c"]) for row in rows),
            "sd_sync_c": statistics.stdev(float(row["sync_c"]) for row in rows),
            "mean_sync_d": statistics.fmean(float(row["sync_d"]) for row in rows),
            "sd_sync_d": statistics.stdev(float(row["sync_d"]) for row in rows),
            "mean_av_offset": statistics.fmean(float(row["av_offset"]) for row in rows),
        }
    comparisons: dict[str, Any] = {}
    for baseline in ("natural_raw", "raw_tts"):
        dc = [float(by_arm["mfa_linear"][key]["sync_c"]) - float(by_arm[baseline][key]["sync_c"]) for key in shared]
        dd = [float(by_arm["mfa_linear"][key]["sync_d"]) - float(by_arm[baseline][key]["sync_d"]) for key in shared]
        by_speaker_c: dict[str, list[float]] = defaultdict(list)
        by_speaker_d: dict[str, list[float]] = defaultdict(list)
        for key, c, d in zip(shared, dc, dd, strict=True):
            speaker = str(by_arm["mfa_linear"][key]["speaker_id"])
            by_speaker_c[speaker].append(c)
            by_speaker_d[speaker].append(d)
        speaker_summary = {}
        for speaker in sorted(by_speaker_c):
            speaker_summary[speaker] = {
                "n": len(by_speaker_c[speaker]),
                "mean_delta_sync_c": statistics.fmean(by_speaker_c[speaker]),
                "mean_delta_sync_d": statistics.fmean(by_speaker_d[speaker]),
                "sync_c_better_count": sum(v > 0 for v in by_speaker_c[speaker]),
                "sync_d_better_count": sum(v < 0 for v in by_speaker_d[speaker]),
                "joint_better_count": sum(c > 0 and d < 0 for c, d in zip(by_speaker_c[speaker], by_speaker_d[speaker], strict=True)),
            }
        comparisons[f"mfa_linear_vs_{baseline}"] = {
            "n": 25,
            "speaker_count": 5,
            "mean_delta_sync_c": statistics.fmean(dc),
            "mean_delta_sync_d": statistics.fmean(dd),
            "cluster_bootstrap_95_ci_delta_sync_c": cluster_bootstrap(by_speaker_c, 42),
            "cluster_bootstrap_95_ci_delta_sync_d": cluster_bootstrap(by_speaker_d, 43),
            "sync_c_better_count": sum(value > 0 for value in dc),
            "sync_d_better_count": sum(value < 0 for value in dd),
            "joint_better_count": sum(c > 0 and d < 0 for c, d in zip(dc, dd, strict=True)),
            "sync_c_paired_t_p": float(stats.ttest_1samp(dc, 0.0).pvalue),
            "sync_d_paired_t_p": float(stats.ttest_1samp(dd, 0.0).pvalue),
            "speaker_summary": speaker_summary,
            "per_sample": {key: {"speaker_id": by_arm["mfa_linear"][key]["speaker_id"], "delta_sync_c": dc[i], "delta_sync_d": dd[i]} for i, key in enumerate(shared)},
        }
    return {
        "schema_version": 1,
        "analysis": "aishell1_mfa_linear_n25_clustered_exploratory",
        "scope": "n=25 paired utterances; 5 speakers x 5; 1 valid + 4 train; no S0770; fixed-face exploratory; no heldout/generalization claim",
        "directions": {"sync_c": "higher_is_better", "sync_d": "lower_is_better"},
        "arm_summary": arm_summary,
        "comparisons": comparisons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--mfa-summary", type=Path, required=True)
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    records = build_records(args.cohort.resolve(), args.tts_meta.resolve(), args.mfa_summary.resolve(), args.face.resolve())
    outdir = args.outdir.resolve()
    write_json(outdir / "manifest.json", {"schema_version": 1, "manifest_type": "aishell1_mfa_linear_n25_eval", "records": records, "arms": list(ARMS), "expected_scores": 75, "heldout_excluded": True})
    summary = evaluate(records, outdir, face_protocol="single_fixed_face_from_aishell1_S0765_sample1")
    write_json(outdir / "analysis.json", analyze(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
