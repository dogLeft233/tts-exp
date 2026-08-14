#!/usr/bin/env python3
"""Fresh n25 three-arm evaluation with n15 resample-poly MFA-linear."""

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
ARMS = ("natural_raw", "raw_tts", "mfa_linear_resample_poly")
EXPECTED_WAV2LIP_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
EXPECTED_SYNCNET_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"


def sha(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def parse_score(path: Path) -> dict[str,float|int]|None:
    text=path.read_text(encoding="utf-8",errors="replace")
    c=re.search(r"Confidence:\s+([0-9.]+)",text); d=re.search(r"Min dist:\s+([0-9.]+)",text); o=re.search(r"AV offset:\s+(-?\d+)",text)
    if c is None or d is None:return None
    return {"sync_c":float(c.group(1)),"sync_d":float(d.group(1)),"av_offset":int(o.group(1)) if o else 0}


def run(cmd:list[str],cwd:Path,log:Path)->int:
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w",encoding="utf-8") as f:return int(subprocess.run(cmd,cwd=str(cwd),stdout=f,stderr=subprocess.STDOUT).returncode)


def finite(row:Mapping[str,Any])->bool:return all(isinstance(row.get(k),(int,float)) and math.isfinite(float(row[k])) for k in ("sync_c","sync_d","av_offset"))


def build_records(cohort_path:Path,strict_tts_path:Path,resample_tts_path:Path,mfa_path:Path,face:Path)->list[dict[str,Any]]:
    if sha(WAV2LIP_CHECKPOINT)!=EXPECTED_WAV2LIP_SHA256 or sha(SYNCNET_MODEL)!=EXPECTED_SYNCNET_SHA256:raise ValueError("model hash changed")
    cohort=json.loads(cohort_path.read_text()); strict=json.loads(strict_tts_path.read_text()); resample=json.loads(resample_tts_path.read_text()); mfa=json.loads(mfa_path.read_text()); face=face.resolve()
    records=list(cohort.get("records",[])); strict_by={str(r["sample_id"]):r for r in strict.get("results",{}).values()}; resample_by={str(r["sample_id"]):r for r in resample.get("results",{}).values()}; mfa_by={str(r["sample_id"]):r for r in mfa.get("results",{}).values()}
    if cohort.get("manifest_type")!="aishell1_mfa_linear_predefined_cohort" or len(records)!=25 or len(strict_by)!=25 or len(resample_by)!=25 or len(mfa_by)!=25 or mfa.get("failures"):raise ValueError("incomplete n25 inputs")
    if not face.is_file():raise FileNotFoundError(face)
    fh=sha(face); out=[]
    for row in records:
        sid=str(row["sample_id"]); strict_row=strict_by[sid]; resample_row=resample_by[sid]; mfa_row=mfa_by[sid]
        if any(str(x.get("paired_key"))!=str(row["paired_key"]) or str(x.get("speaker_id"))!=str(row["speaker_id"]) for x in (strict_row,resample_row,mfa_row)):raise ValueError(f"identity mismatch {sid}")
        natural=Path(str(row["audio_path"])).resolve(); raw=Path(str(strict_row["canonical_16k_audio"])).resolve(); mfa_audio=Path(str(mfa_row["audio_path"])).resolve()
        for p,h in ((natural,row["natural_source_sha256"]),(raw,strict_row["canonical_audio_sha256"]),(mfa_audio,mfa_row["audio_sha256"])):
            if not p.is_file() or sha(p)!=h:raise ValueError(f"audio hash mismatch {sid}: {p}")
        common={"sample_id":sid,"paired_key":row["paired_key"],"speaker_id":row["speaker_id"],"split":row["split"],"transcript":row["transcript"],"face":str(face),"face_sha256":fh,"cache_reused":False}
        out += [{**common,"arm":"natural_raw","source_condition":"natural_raw","audio":str(natural),"audio_sha256":row["natural_source_sha256"]},{**common,"arm":"raw_tts","source_condition":"raw_tts_16k_ffmpeg_canonical","audio":str(raw),"audio_sha256":strict_row["canonical_audio_sha256"]},{**common,"arm":"mfa_linear_resample_poly","source_condition":"24k_source_to_16k_in_memory_scipy_resample_poly","audio":str(mfa_audio),"audio_sha256":mfa_row["audio_sha256"]}]
    return out


def evaluate(records:Sequence[Mapping[str,Any]],outdir:Path)->dict[str,Any]:
    if len(records)!=75:raise ValueError("expected 75 records")
    outdir=outdir.resolve();outdir.mkdir(parents=True,exist_ok=True);scores=[];failures=[]
    for i,r in enumerate(records,1):
        sid=str(r["sample_id"]);arm=str(r["arm"]);audio=Path(r["audio"]);face=Path(r["face"]);video=outdir/"wav2lip"/arm/sid/f"{sid}.mp4";cell=outdir/"wav2lip_work"/arm/sid;(cell/"temp").mkdir(parents=True,exist_ok=True);video.parent.mkdir(parents=True,exist_ok=True)
        rc=run([str(WAV2LIP_PY),str(WAV2LIP/"inference.py"),"--checkpoint_path",str(WAV2LIP_CHECKPOINT),"--face",str(face),"--audio",str(audio),"--outfile",str(video),"--face_det_batch_size","4","--wav2lip_batch_size","4","--nosmooth"],cell,outdir/"logs"/arm/f"{sid}.wav2lip.log")
        if rc or not video.is_file():failures.append({"sample_id":sid,"arm":arm,"stage":"wav2lip","returncode":rc});continue
        syncdir=outdir/"syncnet"/arm/sid;syncdir.mkdir(parents=True,exist_ok=True);ref=f"n25_resample_poly_face1_{arm}_{sid}"
        rp=run([str(SYNCNET_PY),"run_pipeline.py","--videofile",str(video),"--reference",ref,"--data_dir",str(syncdir),"--min_track",str(MIN_TRACK),"--overwrite"],SYNCNET,outdir/"logs"/arm/f"{sid}.pipeline.log")
        if rp:failures.append({"sample_id":sid,"arm":arm,"stage":"syncnet_pipeline","returncode":rp});continue
        slog=outdir/"logs"/arm/f"{sid}.syncnet.log";rs=run([str(SYNCNET_PY),"run_syncnet.py","--videofile",str(video),"--reference",ref,"--data_dir",str(syncdir),"--initial_model",str(SYNCNET_MODEL)],SYNCNET,slog);parsed=parse_score(slog)
        if rs or parsed is None:failures.append({"sample_id":sid,"arm":arm,"stage":"syncnet_score","returncode":rs});continue
        scores.append({**r,"wav2lip_checkpoint_sha256":EXPECTED_WAV2LIP_SHA256,"syncnet_model_sha256":EXPECTED_SYNCNET_SHA256,"min_track":MIN_TRACK,"video":str(video),"video_sha256":sha(video),**parsed});print(f"OK {i}/{len(records)} {arm} {sid} C={parsed['sync_c']:.3f} D={parsed['sync_d']:.3f}",flush=True)
    complete=not failures and len(scores)==75;summary={"schema_version":1,"evaluation":"aishell1_n25_resample_poly_face1","status":"complete" if complete else "incomplete","face_protocol":"single_fixed_face1","same_face_for_all_samples":True,"sample_count":25,"expected_scores":75,"arms":list(ARMS),"cache_reused":False,"min_track":MIN_TRACK,"wav2lip_checkpoint_sha256":EXPECTED_WAV2LIP_SHA256,"syncnet_model_sha256":EXPECTED_SYNCNET_SHA256,"scores":scores,"failures":failures};write_json(outdir/"summary.json",summary)
    if not complete:raise RuntimeError(f"incomplete {len(scores)}/75 failures={len(failures)}")
    return summary


def boot(v:Sequence[float],seed:int,draws:int=10000)->list[float]:
    rng=random.Random(seed);x=sorted(statistics.fmean(v[rng.randrange(len(v))] for _ in v) for _ in range(draws));return [x[int(.025*(draws-1))],x[int(.975*(draws-1))]]


def analyze(s:Mapping[str,Any])->dict[str,Any]:
    by={arm:{} for arm in ARMS}
    for r in s["scores"]:by[r["arm"]][str(r["sample_id"])] = r
    shared=sorted(set.intersection(*(set(by[arm]) for arm in ARMS)), key=lambda value: int(value))
    if len(shared) != 25: raise ValueError(f"expected 25 paired sample ids, found {len(shared)}")
    def cmp(target,base,seed):
        dc=[float(by[target][sid]["sync_c"])-float(by[base][sid]["sync_c"]) for sid in shared];dd=[float(by[target][sid]["sync_d"])-float(by[base][sid]["sync_d"]) for sid in shared]
        return {"n":25,"paired_sample_ids":[int(sid) for sid in shared],"mean_delta_sync_c":statistics.fmean(dc),"bootstrap_95_ci_delta_sync_c":boot(dc,seed),"mean_delta_sync_d":statistics.fmean(dd),"bootstrap_95_ci_delta_sync_d":boot(dd,seed+1),"sync_c_better_count":sum(x>0 for x in dc),"sync_d_better_count":sum(x<0 for x in dd),"joint_better_count":sum(c>0 and d<0 for c,d in zip(dc,dd,strict=True)),"paired_t_sync_c_p":float(stats.ttest_1samp(dc,0).pvalue),"paired_t_sync_d_p":float(stats.ttest_1samp(dd,0).pvalue)}
    comps={"mfa_resample_poly_vs_natural":cmp("mfa_linear_resample_poly","natural_raw",42),"mfa_resample_poly_vs_raw_tts":cmp("mfa_linear_resample_poly","raw_tts",52)}
    arms={}
    for arm in ARMS:
        rows=[by[arm][sid] for sid in shared];arms[arm]={"n":25,"mean_sync_c":statistics.fmean(float(r["sync_c"]) for r in rows),"sd_sync_c":statistics.stdev(float(r["sync_c"]) for r in rows),"mean_sync_d":statistics.fmean(float(r["sync_d"]) for r in rows),"sd_sync_d":statistics.stdev(float(r["sync_d"]) for r in rows)}
    return {"schema_version":1,"analysis":"aishell1_n25_resample_poly_face1_exploratory","scope":"n25 5-speaker cohort; single fixed face1; n15 scipy resample_poly TTS condition; no speaker-matched video generalization","directions":{"sync_c":"higher_is_better","sync_d":"lower_is_better"},"arm_summary":arms,"comparisons":comps}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cohort",type=Path,required=True);p.add_argument("--strict-tts",type=Path,required=True);p.add_argument("--resample-tts",type=Path,required=True);p.add_argument("--mfa",type=Path,required=True);p.add_argument("--face",type=Path,required=True);p.add_argument("--outdir",type=Path,required=True);a=p.parse_args();out=a.outdir.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError(f"non-empty output {out}")
    rec=build_records(a.cohort.resolve(),a.strict_tts.resolve(),a.resample_tts.resolve(),a.mfa.resolve(),a.face.resolve());out.mkdir(parents=True,exist_ok=True);write_json(out/"manifest.json",{"schema_version":1,"manifest_type":"aishell1_n25_resample_poly_face1","expected_scores":75,"arms":list(ARMS),"same_face_for_all_samples":True,"face":str(a.face.resolve()),"face_sha256":sha(a.face.resolve()),"records":rec});summary=evaluate(rec,out);write_json(out/"analysis.json",analyze(summary))
