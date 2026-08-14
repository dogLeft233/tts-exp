#!/usr/bin/env python3
"""Fresh four-arm n15 sampling-rate isolation evaluation."""

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
ARMS = ("natural_raw", "raw_tts", "mfa_linear_old_resample_poly", "mfa_linear_n25_ffmpeg_16k")
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
        return int(subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT).returncode)


def finite_score(row: Mapping[str, Any]) -> bool:
    return all(isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key])) for key in ("sync_c", "sync_d", "av_offset"))


def build_records(counterfactual: Path, old_mfa: Path, new_mfa: Path) -> list[dict[str, Any]]:
    if file_sha256(WAV2LIP_CHECKPOINT) != EXPECTED_WAV2LIP_SHA256 or file_sha256(SYNCNET_MODEL) != EXPECTED_SYNCNET_SHA256:
        raise ValueError("evaluator model hashes changed")
    counter = json.loads(counterfactual.read_text(encoding="utf-8"))
    old = json.loads(old_mfa.read_text(encoding="utf-8"))
    new = json.loads(new_mfa.read_text(encoding="utf-8"))
    if counter.get("heldout_excluded") is not True or counter.get("valid_speaker") != "S0765":
        raise ValueError("counterfactual is not original valid-only n15")
    pairs: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in counter.get("records", []):
        sid, condition = int(row["sample_id"]), str(row["condition"])
        if sid <= 15 and condition in ("natural_noop", "tts_noop"):
            if (sid, condition) in pairs:
                raise ValueError(f"duplicate counterfactual {sid}/{condition}")
            pairs[(sid, condition)] = row
    old_by_id = {}
    for item in old.get("items", []):
        cond = [c for c in item.get("conditions", []) if c.get("condition") == "paired_tts_mfa_linear"]
        if len(cond) != 1:
            raise ValueError(f"missing old MFA condition for {item.get('sample_id')}")
        old_by_id[int(item["sample_id"])] = {**cond[0], "paired_key": item["paired_key"], "sample_id": item["sample_id"]}
    new_by_id = {int(row["sample_id"]): row for row in new.get("results", {}).values()}
    if set(pairs) != {(i, c) for i in range(1, 16) for c in ("natural_noop", "tts_noop")} or set(old_by_id) != set(range(1, 16)) or set(new_by_id) != set(range(1, 16)):
        raise ValueError("input matrices are not exactly n15")
    records = []
    for sid in range(1, 16):
        natural = pairs[(sid, "natural_noop")]
        raw_tts = pairs[(sid, "tts_noop")]
        if natural["paired_key"] != raw_tts["paired_key"]:
            raise ValueError(f"natural/TTS pair mismatch at {sid}")
        face = (REPO / "runs/two_stage_hubert_aishell1_20260810/ditto_videos/natural_raw" / f"{sid}.mp4").resolve()
        if not face.is_file():
            raise FileNotFoundError(face)
        for arm, row, condition in (("natural_raw", natural, "natural_noop"), ("raw_tts", raw_tts, "tts_noop")):
            audio = Path(str(row["output_path"])).resolve()
            if not audio.is_file() or file_sha256(audio) != row["output_sha256"]:
                raise ValueError(f"audio hash mismatch {sid}/{arm}")
            records.append({"sample_id": sid, "paired_key": natural["paired_key"], "speaker_id": "S0765", "split": "valid", "transcript": natural.get("transcript", ""), "arm": arm, "sampling_condition": condition, "audio": str(audio), "audio_sha256": row["output_sha256"], "face": str(face), "face_sha256": file_sha256(face), "cache_reused": False})
        for arm, source, sampling in (("mfa_linear_old_resample_poly", old_by_id[sid], "24k_source_to_16k_in_memory_scipy_resample_poly"), ("mfa_linear_n25_ffmpeg_16k", new_by_id[sid], "24k_source_to_16k_ffmpeg_pcm_s16le_canonical")):
            audio_key, hash_key = ("output_path", "output_sha256") if "output_path" in source else ("audio_path", "audio_sha256")
            audio = Path(str(source[audio_key])).resolve()
            if not audio.is_file() or file_sha256(audio) != source[hash_key]:
                raise ValueError(f"MFA audio hash mismatch {sid}/{arm}")
            if str(source.get("paired_key")) != str(natural["paired_key"]):
                raise ValueError(f"MFA paired key mismatch {sid}/{arm}")
            records.append({"sample_id": sid, "paired_key": natural["paired_key"], "speaker_id": "S0765", "split": "valid", "transcript": natural.get("transcript", ""), "arm": arm, "sampling_condition": sampling, "audio": str(audio), "audio_sha256": source[hash_key], "face": str(face), "face_sha256": file_sha256(face), "cache_reused": False})
    return records


def evaluate(records: Sequence[Mapping[str, Any]], outdir: Path) -> dict[str, Any]:
    if len(records) != 60:
        raise ValueError("sampling isolation requires 60 records")
    outdir = outdir.resolve(); outdir.mkdir(parents=True, exist_ok=True)
    scores = []; failures = []
    for index, record in enumerate(records, 1):
        sid, arm = int(record["sample_id"]), str(record["arm"])
        audio, face = Path(record["audio"]), Path(record["face"])
        video = outdir / "wav2lip" / arm / f"{sid}.mp4"
        cell = outdir / "wav2lip_work" / arm / str(sid); (cell / "temp").mkdir(parents=True, exist_ok=True); video.parent.mkdir(parents=True, exist_ok=True)
        rc = run_command([str(WAV2LIP_PY), str(WAV2LIP / "inference.py"), "--checkpoint_path", str(WAV2LIP_CHECKPOINT), "--face", str(face), "--audio", str(audio), "--outfile", str(video), "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth"], cwd=cell, log=outdir / "logs" / arm / f"{sid}.wav2lip.log")
        if rc or not video.is_file(): failures.append({"sample_id": sid, "arm": arm, "stage": "wav2lip", "returncode": rc}); continue
        syncdir = outdir / "syncnet" / arm / str(sid); syncdir.mkdir(parents=True, exist_ok=True); reference=f"sampling_rate_{arm}_{sid}"
        rp = run_command([str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(video), "--reference", reference, "--data_dir", str(syncdir), "--min_track", str(MIN_TRACK), "--overwrite"], cwd=SYNCNET, log=outdir / "logs" / arm / f"{sid}.pipeline.log")
        if rp: failures.append({"sample_id": sid, "arm": arm, "stage": "syncnet_pipeline", "returncode": rp}); continue
        log=outdir / "logs" / arm / f"{sid}.syncnet.log"
        rs=run_command([str(SYNCNET_PY), "run_syncnet.py", "--videofile", str(video), "--reference", reference, "--data_dir", str(syncdir), "--initial_model", str(SYNCNET_MODEL)], cwd=SYNCNET, log=log)
        parsed=parse_syncnet(log)
        if rs or parsed is None: failures.append({"sample_id": sid, "arm": arm, "stage": "syncnet_score", "returncode": rs}); continue
        scores.append({**record, "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256, "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256, "min_track": MIN_TRACK, "video": str(video), "video_sha256": file_sha256(video), **parsed})
        print(f"OK {index}/{len(records)} {arm} {sid} C={parsed['sync_c']:.3f} D={parsed['sync_d']:.3f}", flush=True)
    complete = not failures and len(scores) == 60
    summary={"schema_version":1,"evaluation":"valid15_sampling_rate_isolation","status":"complete" if complete else "incomplete","sample_count":15,"expected_scores":60,"arms":list(ARMS),"same_utterance_pairing":True,"cache_reused":False,"min_track":MIN_TRACK,"wav2lip_checkpoint_sha256":EXPECTED_WAV2LIP_SHA256,"syncnet_model_sha256":EXPECTED_SYNCNET_SHA256,"scores":scores,"failures":failures}
    write_json(outdir/"summary.json",summary)
    if not complete: raise RuntimeError(f"incomplete {len(scores)}/60 failures={len(failures)}")
    return summary


def bootstrap(values: Sequence[float], seed: int, draws: int=10000) -> list[float]:
    rng=random.Random(seed); estimates=sorted(statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(draws)); return [estimates[int(.025*(draws-1))],estimates[int(.975*(draws-1))]]


def analyze(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "complete": raise ValueError("analysis requires complete summary")
    by={arm:{} for arm in ARMS}
    for row in summary["scores"]:
        sid=int(row["sample_id"]); arm=str(row["arm"])
        if sid in by[arm] or not finite_score(row): raise ValueError("duplicate/nonfinite")
        by[arm][sid]=row
    def cmp(target, baseline, offset):
        dc=[float(by[target][i]["sync_c"])-float(by[baseline][i]["sync_c"]) for i in range(1,16)]; dd=[float(by[target][i]["sync_d"])-float(by[baseline][i]["sync_d"]) for i in range(1,16)]
        return {"n":15,"mean_delta_sync_c":statistics.fmean(dc),"bootstrap_95_ci_delta_sync_c":bootstrap(dc,42+offset*10),"mean_delta_sync_d":statistics.fmean(dd),"bootstrap_95_ci_delta_sync_d":bootstrap(dd,43+offset*10),"sync_c_better_count":sum(x>0 for x in dc),"sync_d_better_count":sum(x<0 for x in dd),"joint_better_count":sum(c>0 and d<0 for c,d in zip(dc,dd,strict=True)),"paired_t_sync_c_p":float(stats.ttest_1samp(dc,0).pvalue),"paired_t_sync_d_p":float(stats.ttest_1samp(dd,0).pvalue),"per_sample":{str(i):{"delta_sync_c":dc[i-1],"delta_sync_d":dd[i-1]} for i in range(1,16)}}
    comparisons={}
    for target in ("mfa_linear_old_resample_poly","mfa_linear_n25_ffmpeg_16k"):
        comparisons[f"{target}_vs_natural_raw"]=cmp(target,"natural_raw",len(comparisons))
        comparisons[f"{target}_vs_raw_tts"]=cmp(target,"raw_tts",len(comparisons))
    comparisons["new_ffmpeg_16k_vs_old_resample_poly"]=cmp("mfa_linear_n25_ffmpeg_16k","mfa_linear_old_resample_poly",len(comparisons))
    arm_summary={}
    for arm in ARMS:
        vals=[by[arm][i] for i in range(1,16)]
        arm_summary[arm]={"n":15,"mean_sync_c":statistics.fmean(float(x["sync_c"]) for x in vals),"sd_sync_c":statistics.stdev(float(x["sync_c"]) for x in vals),"mean_sync_d":statistics.fmean(float(x["sync_d"]) for x in vals),"sd_sync_d":statistics.stdev(float(x["sync_d"]) for x in vals),"mean_av_offset":statistics.fmean(float(x["av_offset"]) for x in vals)}
    return {"schema_version":1,"analysis":"valid15_sampling_rate_isolation_paired","scope":"S0765 valid n15; same utterance/per-sample face; old scipy resample_poly vs n25 ffmpeg 16k canonical TTS","directions":{"sync_c":"higher_is_better","sync_d":"lower_is_better"},"arm_summary":arm_summary,"comparisons":comparisons}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--counterfactual",type=Path,required=True); parser.add_argument("--old-mfa",type=Path,required=True); parser.add_argument("--new-mfa",type=Path,required=True); parser.add_argument("--outdir",type=Path,required=True); args=parser.parse_args(); out=args.outdir.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError(f"refusing non-empty output {out}")
    records=build_records(args.counterfactual.resolve(),args.old_mfa.resolve(),args.new_mfa.resolve()); out.mkdir(parents=True,exist_ok=True)
    write_json(out/"manifest.json",{"schema_version":1,"manifest_type":"valid15_sampling_rate_isolation","same_utterance_pairing":True,"arms":list(ARMS),"expected_scores":60,"counterfactual":str(args.counterfactual.resolve()),"old_mfa":str(args.old_mfa.resolve()),"new_mfa":str(args.new_mfa.resolve()),"wav2lip_checkpoint_sha256":EXPECTED_WAV2LIP_SHA256,"syncnet_model_sha256":EXPECTED_SYNCNET_SHA256,"min_track":MIN_TRACK,"records":records})
    summary=evaluate(records,out); write_json(out/"analysis.json",analyze(summary))
