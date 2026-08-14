#!/usr/bin/env python3
"""Evaluate n=25 MFA-linear audio in the original n=15 face-slot protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
    import re

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


def _old_score_index(summary: Mapping[str, Any]) -> dict[int, dict[str, Mapping[str, Any]]]:
    if summary.get("status") != "complete" or len(summary.get("scores", [])) != 45 or summary.get("failures"):
        raise ValueError("n15 baseline summary is incomplete")
    indexed: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in summary["scores"]:
        slot = int(row["sample_id"])
        arm = str(row["arm"])
        if arm in indexed[slot]:
            raise ValueError(f"duplicate n15 baseline score for slot {slot}/{arm}")
        if arm not in ("natural_raw", "raw_tts"):
            continue
        if row.get("cache_reused") is not True:
            raise ValueError(f"n15 baseline is not the original cache for slot {slot}/{arm}")
        if row.get("wav2lip_checkpoint_sha256") != EXPECTED_WAV2LIP_SHA256 or row.get("syncnet_model_sha256") != EXPECTED_SYNCNET_SHA256 or int(row.get("min_track", -1)) != MIN_TRACK:
            raise ValueError(f"n15 evaluator provenance mismatch for slot {slot}/{arm}")
        if not finite_score(row):
            raise ValueError(f"non-finite n15 baseline score for slot {slot}/{arm}")
        video = Path(str(row["video"]))
        if not video.is_file() or file_sha256(video) != row["video_sha256"]:
            raise ValueError(f"n15 cached video hash mismatch for slot {slot}/{arm}")
        audio = Path(str(row["audio"]))
        if not audio.is_file() or file_sha256(audio) != row["audio_sha256"]:
            raise ValueError(f"n15 cached audio hash mismatch for slot {slot}/{arm}")
        face = Path(str(row["face"]))
        if not face.is_file() or file_sha256(face) != row["face_sha256"]:
            raise ValueError(f"n15 face hash mismatch for slot {slot}/{arm}")
        indexed[slot][arm] = row
    if set(indexed) != set(range(1, 16)) or any(set(rows) != {"natural_raw", "raw_tts"} for rows in indexed.values()):
        raise ValueError("n15 cache does not contain exactly 15 natural/raw-TTS face slots")
    return indexed


def build_records(cohort_path: Path, n25_mfa_summary_path: Path, n15_summary_path: Path) -> list[dict[str, Any]]:
    if file_sha256(WAV2LIP_CHECKPOINT) != EXPECTED_WAV2LIP_SHA256 or file_sha256(SYNCNET_MODEL) != EXPECTED_SYNCNET_SHA256:
        raise ValueError("current evaluator model hashes changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    n25 = json.loads(n25_mfa_summary_path.read_text(encoding="utf-8"))
    n15 = json.loads(n15_summary_path.read_text(encoding="utf-8"))
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort" or len(cohort.get("records", [])) != 25:
        raise ValueError("invalid n25 cohort")
    if n25.get("arm") != "mfa_linear" or n25.get("samples_ok") != 25 or n25.get("failures"):
        raise ValueError("n25 MFA-linear summary is incomplete")
    baseline = _old_score_index(n15)
    n25_by_id = {str(row["sample_id"]): row for row in n25.get("results", {}).values()}
    if len(n25_by_id) != 25:
        raise ValueError("n25 MFA-linear summary does not contain 25 unique samples")
    selected = list(cohort["records"][:15])
    if len(selected) != 15:
        raise ValueError("n25 cohort has fewer than 15 records")
    records: list[dict[str, Any]] = []
    for slot, source_row in enumerate(selected, 1):
        source_id = str(source_row["sample_id"])
        mfa = n25_by_id.get(source_id)
        if mfa is None or str(mfa.get("paired_key")) != str(source_row["paired_key"]) or str(mfa.get("speaker_id")) != str(source_row["speaker_id"]):
            raise ValueError(f"n25 MFA identity mismatch at source sample {source_id}")
        mfa_audio = Path(str(mfa["audio_path"])).resolve()
        if not mfa_audio.is_file() or file_sha256(mfa_audio) != str(mfa["audio_sha256"]):
            raise ValueError(f"n25 MFA audio hash mismatch at source sample {source_id}")
        face = Path(str(baseline[slot]["natural_raw"]["face"])).resolve()
        face_hash = str(baseline[slot]["natural_raw"]["face_sha256"])
        for arm in ("natural_raw", "raw_tts"):
            cached = baseline[slot][arm]
            records.append({
                "sample_id": slot,
                "face_slot": slot,
                "arm": arm,
                "source_condition": str(cached["source_condition"]),
                "audio_source_sample_id": int(cached["sample_id"]),
                "audio_source_paired_key": str(cached["paired_key"]),
                "audio_source_speaker_id": str(cached["speaker_id"]),
                "paired_key": str(cached["paired_key"]),
                "speaker_id": str(cached["speaker_id"]),
                "split": str(cached["split"]),
                "transcript": str(cached.get("transcript", "")),
                "audio": str(Path(str(cached["audio"])).resolve()),
                "audio_sha256": str(cached["audio_sha256"]),
                "face": str(face),
                "face_sha256": face_hash,
                "cache_reused": True,
                "cache_summary_sha256": file_sha256(n15_summary_path),
            })
        records.append({
            "sample_id": slot,
            "face_slot": slot,
            "arm": "mfa_linear",
            "source_condition": "n25_mfa_linear",
            "audio_source_sample_id": int(source_id),
            "audio_source_paired_key": str(source_row["paired_key"]),
            "audio_source_speaker_id": str(source_row["speaker_id"]),
            "paired_key": str(source_row["paired_key"]),
            "speaker_id": str(source_row["speaker_id"]),
            "split": str(source_row["split"]),
            "transcript": str(source_row["transcript"]),
            "audio": str(mfa_audio),
            "audio_sha256": str(mfa["audio_sha256"]),
            "face": str(face),
            "face_sha256": face_hash,
            "cache_reused": False,
            "cache_summary_sha256": None,
        })
    if len(records) != 45:
        raise ValueError(f"expected 45 records, found {len(records)}")
    for slot in range(1, 16):
        rows = [row for row in records if row["face_slot"] == slot]
        if {row["arm"] for row in rows} != set(ARMS):
            raise ValueError(f"incomplete arm set at face slot {slot}")
    return records


def evaluate(records: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        if record["arm"] != "mfa_linear":
            scores.append({**record, **{key: record[key] for key in ()}})
            continue
        slot = int(record["face_slot"])
        audio = Path(str(record["audio"]))
        face = Path(str(record["face"]))
        video = output_dir / "wav2lip" / "mfa_linear" / f"face_slot_{slot}.mp4"
        cell_cwd = output_dir / "wav2lip_work" / "mfa_linear" / f"face_slot_{slot}"
        (cell_cwd / "temp").mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        wav_log = output_dir / "logs" / "mfa_linear" / f"face_slot_{slot}.wav2lip.log"
        rc = run_command([
            str(WAV2LIP_PY), str(WAV2LIP / "inference.py"), "--checkpoint_path", str(WAV2LIP_CHECKPOINT),
            "--face", str(face), "--audio", str(audio), "--outfile", str(video),
            "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth",
        ], cwd=cell_cwd, log=wav_log)
        if rc != 0 or not video.is_file():
            failures.append({"face_slot": slot, "audio_source_sample_id": record["audio_source_sample_id"], "stage": "wav2lip", "returncode": rc})
            continue
        sync_dir = output_dir / "syncnet" / "mfa_linear" / f"face_slot_{slot}"
        sync_dir.mkdir(parents=True, exist_ok=True)
        reference = f"n25_audio_n15_face_slot_{slot}"
        pipeline_log = output_dir / "logs" / "mfa_linear" / f"face_slot_{slot}.pipeline.log"
        rc_pipeline = run_command([
            str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(video), "--reference", reference,
            "--data_dir", str(sync_dir), "--min_track", str(MIN_TRACK), "--overwrite",
        ], cwd=SYNCNET, log=pipeline_log)
        if rc_pipeline != 0:
            failures.append({"face_slot": slot, "audio_source_sample_id": record["audio_source_sample_id"], "stage": "syncnet_pipeline", "returncode": rc_pipeline})
            continue
        sync_log = output_dir / "logs" / "mfa_linear" / f"face_slot_{slot}.syncnet.log"
        rc_sync = run_command([
            str(SYNCNET_PY), "run_syncnet.py", "--videofile", str(video), "--reference", reference,
            "--data_dir", str(sync_dir), "--initial_model", str(SYNCNET_MODEL),
        ], cwd=SYNCNET, log=sync_log)
        parsed = parse_syncnet(sync_log)
        if rc_sync != 0 or parsed is None:
            failures.append({"face_slot": slot, "audio_source_sample_id": record["audio_source_sample_id"], "stage": "syncnet_score", "returncode": rc_sync})
            continue
        scores.append({
            **record,
            "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
            "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
            "min_track": MIN_TRACK,
            "video": str(video),
            "video_sha256": file_sha256(video),
            **parsed,
        })
        print(f"OK {index}/{len(records)} face_slot={slot} n25_sample={record['audio_source_sample_id']} C={parsed['sync_c']:.3f} D={parsed['sync_d']:.3f}", flush=True)
    complete = not failures and len(scores) == 45
    summary = {
        "schema_version": 1,
        "evaluation": "n25_first15_audio_n15_face_slots_wav2lip_syncnet",
        "status": "complete" if complete else "incomplete",
        "audio_protocol": "n25_first15_in_cohort_order",
        "face_protocol": "n15_face_slots_1_to_15",
        "paired_comparison": "by_face_slot_not_same_utterance",
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
        raise RuntimeError(f"evaluation incomplete: {len(scores)}/45; failures={len(failures)}")
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
        slot = int(row["face_slot"])
        arm = str(row["arm"])
        if slot in by_arm[arm] or not finite_score(row):
            raise ValueError("duplicate or non-finite face-slot score")
        by_arm[arm][slot] = row
    shared = sorted(set.intersection(*(set(by_arm[arm]) for arm in ARMS)))
    if shared != list(range(1, 16)):
        raise ValueError("analysis does not have all 15 face slots")
    arm_summary = {}
    for arm in ARMS:
        rows = [by_arm[arm][slot] for slot in shared]
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
        dc = [float(by_arm["mfa_linear"][slot]["sync_c"]) - float(by_arm[baseline][slot]["sync_c"]) for slot in shared]
        dd = [float(by_arm["mfa_linear"][slot]["sync_d"]) - float(by_arm[baseline][slot]["sync_d"]) for slot in shared]
        comparisons[f"mfa_linear_vs_{baseline}"] = {
            "n": 15,
            "paired_unit": "face_slot",
            "same_utterance_pairing": False,
            "mean_delta_sync_c": statistics.fmean(dc),
            "bootstrap_95_ci_delta_sync_c": bootstrap(dc, 42 + offset * 10),
            "mean_delta_sync_d": statistics.fmean(dd),
            "bootstrap_95_ci_delta_sync_d": bootstrap(dd, 43 + offset * 10),
            "sync_c_better_count": sum(value > 0 for value in dc),
            "sync_d_better_count": sum(value < 0 for value in dd),
            "joint_better_count": sum(c > 0 and d < 0 for c, d in zip(dc, dd, strict=True)),
            "sync_c_paired_t_p": float(stats.ttest_1samp(dc, 0.0).pvalue),
            "sync_d_paired_t_p": float(stats.ttest_1samp(dd, 0.0).pvalue),
            "per_face_slot": {
                str(slot): {
                    "audio_source_sample_id": by_arm["mfa_linear"][slot]["audio_source_sample_id"],
                    "audio_source_paired_key": by_arm["mfa_linear"][slot]["audio_source_paired_key"],
                    "delta_sync_c": dc[index],
                    "delta_sync_d": dd[index],
                }
                for index, slot in enumerate(shared)
            },
        }
    return {
        "schema_version": 1,
        "analysis": "n25_first15_audio_n15_face_slots_exploratory",
        "scope": "15 n25 MFA-linear audio samples evaluated on n15 face slots; cross-utterance slot comparison; no same-text pairing or generalization claim",
        "directions": {"sync_c": "higher_is_better", "sync_d": "lower_is_better"},
        "arm_summary": arm_summary,
        "comparisons": comparisons,
    }


def attach_cached_scores(records: list[dict[str, Any]], n15_summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(n15_summary_path.read_text(encoding="utf-8"))
    indexed = _old_score_index(summary)
    result = []
    for row in records:
        if row["arm"] in ("natural_raw", "raw_tts"):
            cached = indexed[int(row["face_slot"])][str(row["arm"])]
            result.append({**row, **{key: cached[key] for key in ("video", "video_sha256", "sync_c", "sync_d", "av_offset", "wav2lip_checkpoint_sha256", "syncnet_model_sha256", "min_track")}})
        else:
            result.append(row)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n25-cohort", type=Path, required=True)
    parser.add_argument("--n25-mfa-summary", type=Path, required=True)
    parser.add_argument("--n15-summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    records = build_records(args.n25_cohort.resolve(), args.n25_mfa_summary.resolve(), args.n15_summary.resolve())
    records = attach_cached_scores(records, args.n15_summary.resolve())
    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "manifest.json", {
        "schema_version": 1,
        "manifest_type": "n25_first15_audio_n15_face_slots_wav2lip_syncnet",
        "audio_cohort": "n25_first15_in_cohort_order",
        "face_protocol": "n15_face_slots_1_to_15",
        "same_utterance_pairing": False,
        "expected_scores": 45,
        "arms": list(ARMS),
        "n25_cohort": str(args.n25_cohort.resolve()),
        "n25_mfa_summary": str(args.n25_mfa_summary.resolve()),
        "n15_summary": str(args.n15_summary.resolve()),
        "n15_summary_sha256": file_sha256(args.n15_summary.resolve()),
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
        "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
        "min_track": MIN_TRACK,
        "records": records,
    })
    summary = evaluate(records, outdir)
    write_json(outdir / "analysis.json", analyze(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
