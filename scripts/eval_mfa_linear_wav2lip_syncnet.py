#!/usr/bin/env python3
"""Evaluate 15 valid natural/TTS/MFA-linear arms with local Wav2Lip + SyncNet."""

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

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs/two_stage_hubert_aishell1_20260810"
FACE_DIR = RUN / "ditto_videos/natural_raw"
MAPPING = RUN / "ditto_videos/sample_mapping.json"
WAV2LIP = REPO / "third_party/Wav2Lip"
SYNCNET = REPO / "third_party/syncnet_python"
WAV2LIP_PY = Path.home() / ".venvs/wav2lip/bin/python"
SYNCNET_PY = Path.home() / ".venvs/syncnet/bin/python"
WAV2LIP_CHECKPOINT = WAV2LIP / "checkpoints/wav2lip_gan.pth"
SYNCNET_MODEL = SYNCNET / "data/syncnet_v2.model"
MIN_TRACK = 50
COUNT = 15
VALID_SPEAKER = "S0765"
EXPECTED_WAV2LIP_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
EXPECTED_SYNCNET_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
ARMS = ("natural_raw", "raw_tts", "mfa_linear")
SOURCE_CONDITION = {"natural_raw": "natural_noop", "raw_tts": "tts_noop"}


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


def build_manifest(
    mfa_summary_path: Path,
    counterfactual_summary_path: Path,
    cached_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mfa = json.loads(mfa_summary_path.read_text(encoding="utf-8"))
    counter = json.loads(counterfactual_summary_path.read_text(encoding="utf-8"))
    cached = json.loads(cached_summary_path.read_text(encoding="utf-8"))
    if mfa.get("heldout_excluded") is not True or counter.get("heldout_excluded") is not True or cached.get("heldout_excluded") is not True:
        raise ValueError("all sources must exclude heldout")
    if mfa.get("conditions") != ["paired_tts_mfa_linear"] or len(mfa.get("items", [])) != COUNT:
        raise ValueError("MFA summary must contain exactly 15 MFA-linear items")
    if cached.get("failures") or len(cached.get("scores", [])) != int(cached.get("expected_scores", -1)):
        raise ValueError("cached baseline matrix is incomplete")
    if cached.get("wav2lip_checkpoint_sha256") != EXPECTED_WAV2LIP_SHA256 or cached.get("syncnet_model_sha256") != EXPECTED_SYNCNET_SHA256:
        raise ValueError("cached evaluator model hashes changed")
    if file_sha256(WAV2LIP_CHECKPOINT) != EXPECTED_WAV2LIP_SHA256 or file_sha256(SYNCNET_MODEL) != EXPECTED_SYNCNET_SHA256:
        raise ValueError("current evaluator model hashes changed")

    mapping = {int(row["id"]): row for row in json.loads(MAPPING.read_text(encoding="utf-8"))}
    counter_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in counter.get("records", []):
        condition = str(row.get("condition"))
        if condition in SOURCE_CONDITION.values():
            counter_by_key[(str(row["paired_key"]), condition)] = row
    cache_by_key = {
        (str(row["paired_key"]), str(row["condition"])): row for row in cached["scores"]
        if str(row.get("condition")) in SOURCE_CONDITION.values()
    }

    records: list[dict[str, Any]] = []
    selected_keys = list(mfa["selection"]["ordered_paired_keys"])
    for item in mfa["items"]:
        paired_key = str(item["paired_key"])
        sample_id = int(item["sample_id"])
        if paired_key != selected_keys[len({record["paired_key"] for record in records})]:
            raise ValueError("MFA item order differs from frozen selection")
        map_row = mapping.get(sample_id)
        if map_row is None or map_row.get("paired_key") != paired_key:
            raise ValueError(f"visual mapping mismatch for {paired_key}")
        face = FACE_DIR / f"{sample_id}.mp4"
        if not face.is_file():
            raise FileNotFoundError(face)
        face_hash = file_sha256(face)
        for arm in ("natural_raw", "raw_tts"):
            source_condition = SOURCE_CONDITION[arm]
            source = counter_by_key.get((paired_key, source_condition))
            cache = cache_by_key.get((paired_key, source_condition))
            if source is None or cache is None:
                raise ValueError(f"missing {arm} source/cache for {paired_key}")
            audio = Path(source["output_path"]).resolve()
            if not audio.is_file() or file_sha256(audio) != source["output_sha256"]:
                raise ValueError(f"{arm} audio hash mismatch for {paired_key}")
            if int(cache["sample_id"]) != sample_id or cache["audio_sha256"] != source["output_sha256"]:
                raise ValueError(f"{arm} cached identity/audio mismatch for {paired_key}")
            if Path(cache["audio"]).resolve() != audio or Path(cache["face"]).resolve() != face.resolve():
                raise ValueError(f"{arm} cached input path mismatch for {paired_key}")
            if cache["face_sha256"] != face_hash or int(cache["min_track"]) != MIN_TRACK or not _finite_score(cache):
                raise ValueError(f"{arm} cached provenance/score mismatch for {paired_key}")
            records.append({
                "sample_id": sample_id, "paired_key": paired_key, "transcript": item["transcript"],
                "speaker_id": VALID_SPEAKER, "split": "valid", "arm": arm,
                "source_condition": source_condition, "audio": str(audio),
                "audio_sha256": source["output_sha256"], "face": str(face.resolve()),
                "face_sha256": face_hash, "cache_source": str(cached_summary_path.resolve()),
            })
        mfa_row = item["conditions"][0]
        if mfa_row.get("condition") != "paired_tts_mfa_linear":
            raise ValueError("MFA item contains a non-linear condition")
        audio = Path(mfa_row["output_path"]).resolve()
        if not audio.is_file() or file_sha256(audio) != mfa_row["output_sha256"]:
            raise ValueError(f"MFA audio hash mismatch for {paired_key}")
        records.append({
            "sample_id": sample_id, "paired_key": paired_key, "transcript": item["transcript"],
            "speaker_id": VALID_SPEAKER, "split": "valid", "arm": "mfa_linear",
            "source_condition": "paired_tts_mfa_linear", "audio": str(audio),
            "audio_sha256": mfa_row["output_sha256"], "face": str(face.resolve()),
            "face_sha256": face_hash, "cache_source": None,
        })
    _validate_records(records)
    manifest = {
        "schema_version": 1, "manifest_type": "valid15_mfa_linear_wav2lip_syncnet",
        "valid_speaker": VALID_SPEAKER, "heldout_excluded": True, "sample_count": COUNT,
        "arms": list(ARMS), "expected_scores": COUNT * len(ARMS), "min_track": MIN_TRACK,
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
        "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
        "sources": {
            "mfa_summary": str(mfa_summary_path.resolve()), "mfa_summary_sha256": file_sha256(mfa_summary_path),
            "counterfactual_summary": str(counterfactual_summary_path.resolve()), "counterfactual_summary_sha256": file_sha256(counterfactual_summary_path),
            "cached_summary": str(cached_summary_path.resolve()), "cached_summary_sha256": file_sha256(cached_summary_path),
        },
        "records": records,
    }
    return manifest, cached


def _validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != COUNT * len(ARMS):
        raise ValueError(f"expected 45 manifest records, found {len(records)}")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("speaker_id") != VALID_SPEAKER or row.get("split") != "valid":
            raise ValueError("manifest crossed valid split/speaker boundary")
        grouped[int(row["sample_id"])].append(row)
    if len(grouped) != COUNT:
        raise ValueError(f"expected 15 unique samples, found {len(grouped)}")
    for sample_id, rows in grouped.items():
        if {str(row["arm"]) for row in rows} != set(ARMS):
            raise ValueError(f"incomplete arm set for sample {sample_id}")
        if len({str(row["paired_key"]) for row in rows}) != 1 or len({str(row["face_sha256"]) for row in rows}) != 1:
            raise ValueError(f"paired-key/visual mismatch for sample {sample_id}")


def evaluate(manifest: Mapping[str, Any], cached: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _validate_records(manifest["records"])
    cache_by_key = {
        (str(row["paired_key"]), str(row["condition"])): row for row in cached["scores"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(manifest["records"], 1):
        arm = str(record["arm"])
        sample_id = int(record["sample_id"])
        paired_key = str(record["paired_key"])
        if arm != "mfa_linear":
            source = cache_by_key[(paired_key, str(record["source_condition"]))]
            scores.append({
                **{key: record[key] for key in ("sample_id", "paired_key", "speaker_id", "split", "arm", "source_condition", "audio", "audio_sha256", "face", "face_sha256")},
                "cache_reused": True, "cache_summary_sha256": manifest["sources"]["cached_summary_sha256"],
                "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
                "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256, "min_track": MIN_TRACK,
                "video": source["video"], "video_sha256": source["video_sha256"],
                "sync_c": float(source["sync_c"]), "sync_d": float(source["sync_d"]),
                "av_offset": int(source["av_offset"]),
            })
            continue
        audio = Path(record["audio"])
        face = Path(record["face"])
        video = output_dir / "wav2lip" / arm / f"{sample_id}.mp4"
        wav_log = output_dir / "logs" / arm / f"{sample_id}.wav2lip.log"
        cell_cwd = output_dir / "wav2lip_work" / arm / str(sample_id)
        (cell_cwd / "temp").mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
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
        reference = f"mfa_linear_{sample_id}"
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
            **{key: record[key] for key in ("sample_id", "paired_key", "speaker_id", "split", "arm", "source_condition", "audio", "audio_sha256", "face", "face_sha256")},
            "cache_reused": False, "cache_summary_sha256": None,
            "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
            "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256, "min_track": MIN_TRACK,
            "video": str(video), "video_sha256": file_sha256(video), **parsed,
        }
        score_path = output_dir / "scores" / arm / f"{sample_id}.json"
        write_json(score_path, score)
        scores.append(score)
        print(f"OK {index}/45 {arm} {sample_id} C={score['sync_c']:.3f} D={score['sync_d']:.3f}", flush=True)
    complete = not failures and len(scores) == COUNT * len(ARMS)
    result = {
        "schema_version": 1, "evaluation": "valid15_mfa_linear_wav2lip_syncnet",
        "status": "complete" if complete else "incomplete", "valid_speaker": VALID_SPEAKER,
        "heldout_excluded": True, "arms": list(ARMS), "sample_count": COUNT,
        "expected_scores": COUNT * len(ARMS), "min_track": MIN_TRACK,
        "scores": scores, "failures": failures,
    }
    write_json(output_dir / "summary.json", result)
    if not complete:
        raise RuntimeError(f"downstream matrix incomplete: {len(scores)}/45, failures={failures}")
    return result


def _bootstrap(values: list[float], seed: int, draws: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(draws))
    return [estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]]


def _test_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("paired test requires values")
    t_result = stats.ttest_1samp(values, 0.0)
    try:
        wilcoxon = stats.wilcoxon(values)
        wilcoxon_result: dict[str, Any] | None = {"statistic": float(wilcoxon.statistic), "pvalue": float(wilcoxon.pvalue)}
    except ValueError:
        wilcoxon_result = None
    return {
        "paired_t": {"statistic": float(t_result.statistic), "pvalue": float(t_result.pvalue)},
        "wilcoxon": wilcoxon_result,
    }


def analyze(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "complete" or summary.get("failures") or len(summary.get("scores", [])) != 45:
        raise ValueError("analysis requires a complete 45-score matrix")
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in summary["scores"]:
        arm = str(row["arm"]); sample_id = int(row["sample_id"])
        if sample_id in by_arm[arm] or not _finite_score(row):
            raise ValueError("duplicate or non-finite score")
        by_arm[arm][sample_id] = row
    shared = sorted(set.intersection(*(set(by_arm[arm]) for arm in ARMS)))
    if len(shared) != COUNT:
        raise ValueError("analysis does not have 15 complete paired samples")
    arm_summary = {}
    for arm in ARMS:
        rows = [by_arm[arm][sample_id] for sample_id in shared]
        arm_summary[arm] = {
            "n": COUNT,
            "mean_sync_c": statistics.fmean(float(row["sync_c"]) for row in rows),
            "sd_sync_c": statistics.stdev(float(row["sync_c"]) for row in rows),
            "mean_sync_d": statistics.fmean(float(row["sync_d"]) for row in rows),
            "sd_sync_d": statistics.stdev(float(row["sync_d"]) for row in rows),
            "mean_av_offset": statistics.fmean(float(row["av_offset"]) for row in rows),
        }
    comparisons = {}
    for offset, baseline in enumerate(("natural_raw", "raw_tts")):
        delta_c = [float(by_arm["mfa_linear"][sid]["sync_c"]) - float(by_arm[baseline][sid]["sync_c"]) for sid in shared]
        delta_d = [float(by_arm["mfa_linear"][sid]["sync_d"]) - float(by_arm[baseline][sid]["sync_d"]) for sid in shared]
        comparisons[f"mfa_linear_vs_{baseline}"] = {
            "n": COUNT, "mean_delta_sync_c": statistics.fmean(delta_c),
            "bootstrap_95_ci_delta_sync_c": _bootstrap(delta_c, 42 + offset * 10),
            "mean_delta_sync_d": statistics.fmean(delta_d),
            "bootstrap_95_ci_delta_sync_d": _bootstrap(delta_d, 43 + offset * 10),
            "sync_c_better_count": sum(value > 0 for value in delta_c),
            "sync_d_better_count": sum(value < 0 for value in delta_d),
            "joint_better_count": sum(c > 0 and d < 0 for c, d in zip(delta_c, delta_d, strict=True)),
            "sync_c_tests": _test_summary(delta_c), "sync_d_tests": _test_summary(delta_d),
            "per_sample": {str(sid): {"delta_sync_c": delta_c[i], "delta_sync_d": delta_d[i]} for i, sid in enumerate(shared)},
        }
    return {
        "schema_version": 1, "analysis": "valid15_mfa_linear_paired_exploratory",
        "scope": "n=15 exploratory valid-only; no heldout/generalization claim",
        "directions": {"sync_c": "higher_is_better", "sync_d": "lower_is_better"},
        "arm_summary": arm_summary, "comparisons": comparisons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mfa-summary", type=Path, required=True)
    parser.add_argument("--counterfactual-summary", type=Path, required=True)
    parser.add_argument("--cached-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest, cached = build_manifest(args.mfa_summary, args.counterfactual_summary, args.cached_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", manifest)
    summary = evaluate(manifest, cached, args.output_dir)
    analysis = analyze(summary)
    write_json(args.output_dir / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
