#!/usr/bin/env python3
"""Freeze a learned-head-only 60-row dev render and its 240-row merge plan.

Reads only the already frozen Stage-A development plan plus fit checkpoint
metadata. No development score values or confirmation audio are opened.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

BASE_PLAN_SHA="bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585"
FIT_PLAN_SHA="34b474fbc740103142b170d8d07e271b47b0bf0fc76fc5056666048fd9f0d956"
FIT_REPORT_SHA="c5ea4c3877945719a1c680646629f65fb73b8e59883ad9a0c755adde1ffc58d0"
CHECKPOINT_SHA="bc5aeec15f9197ca0c1d0d606eebe41ff2dbf6620767eee5bc37ca4b9395aaed"
BASE_METHODS=("identity","rms_fixed","rms_localcompress")

def sha(raw:bytes)->str: return hashlib.sha256(raw).hexdigest()
def canon(value:Any)->bytes: return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def seal(value:dict[str,Any],field:str)->str:
    body=dict(value); body.pop(field,None); return sha(canon(body))
def write(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f: f.write(data); f.flush()
def freeze(plan:dict[str,Any])->dict[str,Any]:
    plan["plan_sha256"]=seal(plan,"plan_sha256")
    return plan
def write_triplet(path:Path,plan:dict[str,Any],rows:list[dict[str,Any]],manifest_extra:dict[str,Any])->dict[str,Any]:
    if path.exists() and any(path.iterdir()): raise FileExistsError(f"refusing nonempty output: {path}")
    path.mkdir(parents=True,exist_ok=True)
    plan_raw=(json.dumps(plan,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    rows_raw=b"".join(canon(r)+b"\n" for r in rows)
    write(path/"plan.json",plan_raw); write(path/"planned_rows.jsonl",rows_raw)
    manifest={"schema":"cross_tfg_stageb_plan_v1","run_id":plan["run_id"],"plan_sha256":plan["plan_sha256"],"plan_raw_sha256":sha(plan_raw),"planned_rows_sha256":sha(rows_raw),"planned_rows":len(rows),"method_counts":{m["method_id"]:sum(r["method_id"]==m["method_id"] for r in rows) for m in plan["methods"]},**manifest_extra}
    manifest["manifest_sha256"]=seal(manifest,"manifest_sha256")
    write(path/"plan_manifest.json",(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode())
    return manifest

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-plan",type=Path,required=True)
    p.add_argument("--runner",type=Path,required=True)
    p.add_argument("--pipeline",type=Path,required=True)
    p.add_argument("--heads",type=Path,required=True)
    p.add_argument("--checkpoint-path",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--fit-report-sha256",required=True)
    args=p.parse_args()
    base_raw=args.base_plan.read_bytes()
    base_sha=sha(base_raw)
    if base_sha!=BASE_PLAN_SHA: raise ValueError(f"Stage-A base plan SHA mismatch: {base_sha}")
    if args.checkpoint_sha256.lower()!=CHECKPOINT_SHA or args.fit_report_sha256.lower()!=FIT_REPORT_SHA:
        raise ValueError("fit checkpoint/report pins differ from the completed 24-step fit07 receipt")
    base=json.loads(base_raw)
    if base.get("status")!="FROZEN" or base.get("plan_sha256")!=seal(base,"plan_sha256"):
        raise ValueError("base Stage-A plan seal invalid")
    if len(base["rows"])!=180 or len(base["methods"])!=3 or [m["method_id"] for m in base["methods"]]!=list(BASE_METHODS):
        raise ValueError("unexpected Stage-A method/row inventory")
    runner_sha=sha(args.runner.read_bytes()); pipeline_sha=sha(args.pipeline.read_bytes()); heads_sha=sha(args.heads.read_bytes())
    architecture={"class":"scripts.audio_enhancement.heads.SpectralResidualHead","n_fft":512,"hop_length":128,"hidden_channels":24,"max_residual_db":3.0,"level_normalize":True,"frequency_center_mask":False,"frame_energy_normalize":False,"sample_rate_hz":16000,"identity_initialization":"final_conv_zero_weight_and_bias"}
    config={
        "method":"spectral_head","sample_rate_hz":16000,
        "use_global_rms":True,"use_local_envelope":False,"use_spectral_head":True,
        "post_rms_match":"target","target_rms":0.0376838172675746,
        "rms_stage_peak_ceiling":0.99,"peak_ceiling":0.99,
        "local_window_ms":32.0,"local_ratio":0.8,"local_min_gain_db":-6.0,"local_max_gain_db":6.0,
        "head_architecture":architecture,
        "checkpoint_path":args.checkpoint_path,
        "checkpoint_sha256":CHECKPOINT_SHA,
        "fit_report_sha256":FIT_REPORT_SHA,
        "fit_plan_raw_sha256":FIT_PLAN_SHA,
        "fit_optimizer_steps":24,
        "training_objective":"31-lag median-minus-normalized-softmin tau=0.1; historical fit only",
        "transform_source_sha256":{"run_stagea_dev_attempt02.py":runner_sha,"audio_enhancement_pipeline.py":pipeline_sha,"audio_enhancement_heads.py":heads_sha},
    }
    method={"method_id":"spectral_head_fit24","kind":"learned_audio_head_candidate","config":config,"config_sha256":sha(canon(config)),"checkpoint_sha256":CHECKPOINT_SHA,"fit_report_sha256":FIT_REPORT_SHA,"fit_plan_raw_sha256":FIT_PLAN_SHA,"training_steps":24,"selection_scope":"development_only; no confirmation audio opened"}
    by_context:dict[tuple[Any,...],list[dict[str,Any]]]={}; context_order=[]
    for row in base["rows"]:
        key=(row["split"],row["sample_id"],row["speaker_id"],row["model_id"],row["face_id"],row["seed"])
        if key not in by_context: by_context[key]=[]; context_order.append(key)
        by_context[key].append(row)
    head_rows=[]; combined_rows=[]
    for key in context_order:
        rows=by_context[key]
        if [r["method_id"] for r in rows]!=list(BASE_METHODS): raise ValueError(f"unexpected context methods: {key}")
        combined_rows.extend(copy.deepcopy(rows))
        source=next(r for r in rows if r["method_id"]=="identity")
        candidate=copy.deepcopy(source)
        candidate["row_id"]=source["row_id"].rsplit("_identity",1)[0]+"_spectral_head_fit24"
        candidate["method_id"]=method["method_id"]
        candidate["method_config_sha256"]=method["config_sha256"]
        candidate["checkpoint_sha256"]=CHECKPOINT_SHA
        candidate["generation_settings"]={**candidate["generation_settings"],"audio_transform":"fit07_spectral_head_full_audio","fit_report_sha256":FIT_REPORT_SHA,"head_checkpoint_sha256":CHECKPOINT_SHA}
        head_rows.append(candidate); combined_rows.append(candidate)
    if len(head_rows)!=60 or len(combined_rows)!=240: raise ValueError("Stage-B must add exactly sixty rows to the frozen Stage-A plan")

    combined=copy.deepcopy(base)
    combined["run_id"]="cross_tfg_audio_head_20261003_stageb_wav2lip_dev_attempt01"
    combined["methods"].append(method)
    combined["rows"]=combined_rows
    combined["status"]="FROZEN"
    combined["execution"]={**combined["execution"],"stage":"B","scope":"dev-only; Stage-A baseline rows reused byte-for-byte; new fit07 candidate rows; no confirmation audio/effects read","planned_rows":240,"stagea_base_plan_sha256":BASE_PLAN_SHA,"stagea_reused_rows":180,"stageb_candidate_rows":60,"candidate_checkpoint_sha256":CHECKPOINT_SHA,"candidate_fit_report_sha256":FIT_REPORT_SHA,"candidate_transform_runner_sha256":runner_sha,"candidate_selection":"development_only_after_combined_support_rebuild_and_independent_consumer_audit"}
    new_contrasts=[
        {"contrast_id":"spectral_head_fit24_minus_identity","method_a":"spectral_head_fit24","method_b":"identity"},
        {"contrast_id":"spectral_head_fit24_minus_rms_fixed","method_a":"spectral_head_fit24","method_b":"rms_fixed"},
        {"contrast_id":"spectral_head_fit24_minus_rms_localcompress","method_a":"spectral_head_fit24","method_b":"rms_localcompress"},
    ]
    contrasts=list(base["contrasts"])+new_contrasts
    combined["contrasts"]=contrasts; combined["analysis"]["contrasts"]=contrasts
    endpoints=[]
    for c in contrasts:
        for support in ("full","guard20"):
            endpoints.append({"endpoint_id":f"dev_{c['contrast_id']}_{support}_Cbest","split":"dev","contrast_id":c["contrast_id"],"support_id":support,"metric":"Cbest","model_scope":"model","model_id":base["models"][0]["model_id"],"roi_estimand":"native_official","audio_estimand":"native_roi_original_pcm16","exploratory":True})
    combined["analysis"]["summary_endpoints"]=endpoints
    freeze(combined)

    candidate=copy.deepcopy(combined)
    candidate["run_id"]="cross_tfg_audio_head_20261003_stageb_head_only_dev_attempt01"
    candidate["methods"]=[method]
    candidate["rows"]=head_rows
    candidate["generation_repeat_qc"]=[]
    candidate["contrasts"]=[]
    candidate["analysis"]["contrasts"]=[]
    candidate["analysis"]["summary_endpoints"]=[]
    candidate["execution"]={**combined["execution"],"scope":"candidate-only development render subrun; no independent efficacy interpretation; merged into the sealed 240-row combined plan","stage":"B_candidate_subrun","planned_rows":60,"combined_plan_sha256":combined["plan_sha256"],"candidate_selection":"not performed in this subrun"}
    freeze(candidate)

    combined_dir=args.output_dir/"combined240"
    candidate_dir=args.output_dir/"candidate60"
    m1=write_triplet(combined_dir,combined,combined_rows,{"base_stagea_plan_raw_sha256":BASE_PLAN_SHA,"fit_plan_raw_sha256":FIT_PLAN_SHA,"fit_report_sha256":FIT_REPORT_SHA,"candidate_checkpoint_sha256":CHECKPOINT_SHA,"confirm_speaker_ids_in_plan":0,"development_audio_only":True})
    m2=write_triplet(candidate_dir,candidate,head_rows,{"combined_plan_sha256":combined["plan_sha256"],"candidate_checkpoint_sha256":CHECKPOINT_SHA,"planned_candidate_rows":60,"development_audio_only":True,"confirm_audio_read":False})
    print(json.dumps({"status":"STAGEB_HEAD_PLAN_FROZEN","combined_plan_sha256":combined["plan_sha256"],"combined_plan_raw_sha256":m1["plan_raw_sha256"],"combined_rows":240,"candidate_plan_sha256":candidate["plan_sha256"],"candidate_plan_raw_sha256":m2["plan_raw_sha256"],"candidate_rows":60,"candidate_method_config_sha256":method["config_sha256"],"checkpoint_sha256":CHECKPOINT_SHA,"runner_sha256":runner_sha},sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
