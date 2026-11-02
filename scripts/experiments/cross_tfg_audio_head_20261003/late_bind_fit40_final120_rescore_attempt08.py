#!/usr/bin/env python3
"""Late-bind actual complete fit40/V3/120-step and geometry artifacts for a no-grad rescore (attempt07)."""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
from pathlib import Path

SCHEMA = "cross_tfg_fit40_final120_rescore_late_binding_v1"
CONFIG_SCHEMA = "cross_tfg_fit40_final120_rescore_config_v1"
SOURCE_BUNDLE_SCHEMA = "cross_tfg_fit40_final120_rescore_source_bundle_v1"
EXPECTED_PARENT_RAW = "e491d0bb210fe78647f4845543a88c629d37cecc09a0823c287560fbb1e71a06"
EXPECTED_PARENT_PAYLOAD = "7200d93801ca0749b472937645d1e955e5b39c730f6a2d1a5e24d3f380f6dce9"
EXPECTED_SOURCE_MANIFEST = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
EXPECTED_SOURCE_BYTES = 61469
EXPECTED_HEAD_CONFIG = "0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5"
EXPECTED_HEAD_CONFIG_BYTES = 10721
EXPECTED_PRODUCER = "0ecfd1f3be25f6c56d76fa6fecd1ab5dd9b1868e03b5394f355939e74dc14df7"
EXPECTED_PRODUCER_BYTES = 62375
EXPECTED_RUNNER = "66e3a59e088a46a9681ccb07ce18111b9153c12eb59134edd7c8dd38e57dc1aa"
EXPECTED_RESCORE = "0ff8d38d5d6343e4efd7edb633ee8e16a9354e8ef60a814bb7940571d7526e0f"
EXPECTED_STAGE_MEMBERS = 26

class NotReady(RuntimeError):
    pass

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def regular(path: Path) -> Path:
    if path.is_symlink(): raise NotReady("symlink rejected: "+str(path))
    q=path.resolve(strict=True)
    if not q.is_file(): raise NotReady("regular file required: "+str(q))
    return q

def pin(path: Path) -> dict:
    q=regular(path); st=q.stat()
    return {"path":str(q),"bytes":st.st_size,"sha256":sha_file(q)}

def load(path: Path, expected_bytes=None, expected_sha=None):
    q=regular(path); raw=q.read_bytes()
    if expected_bytes is not None and len(raw)!=expected_bytes: raise NotReady("byte count mismatch: "+str(q))
    if expected_sha is not None and sha(raw)!=expected_sha: raise NotReady("raw SHA mismatch: "+str(q))
    value=json.loads(raw.decode("utf-8"))
    if not isinstance(value,dict): raise NotReady("JSON object required: "+str(q))
    return value,raw

def verify_ref(ref: dict, label: str) -> Path:
    if not isinstance(ref,dict): raise NotReady(label+" pin record absent")
    p=regular(Path(str(ref.get("path",""))))
    st=p.stat()
    digest=sha_file(p)
    if st.st_size!=int(ref.get("bytes",-1)) or digest!=str(ref.get("sha256","")).lower():
        raise NotReady(label+" artifact pin mismatch")
    return p

def require_hex(value: str, label: str):
    if not re.fullmatch(r"[0-9a-f]{64}",str(value).lower()): raise NotReady(label+" is not a 64-hex SHA-256")
    return str(value).lower()

def write_exclusive(path: Path, value: dict):
    if path.exists() or path.is_symlink(): raise NotReady("refusing to overwrite binding: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")
    with path.open("xb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return {"path":str(path),"bytes":len(raw),"sha256":sha(raw)}

def make_binding(cfg: dict, cfg_raw: bytes, config_path: Path):
    if cfg.get("schema")!=CONFIG_SCHEMA or cfg.get("scope")!="fit-only post-update rescore; no selection; development and confirmation closed":
        raise NotReady("rescore config schema/scope mismatch")
    paths=cfg.get("remote_paths")
    if not isinstance(paths,dict): raise NotReady("remote paths absent")
    code_root=Path(paths["code_root"]).resolve(strict=True)
    repo=code_root/"src"
    control=Path(paths["control_root"]).resolve(strict=True)
    if not code_root.is_dir() or not control.is_dir(): raise NotReady("staged code/control root missing")
    bundle_manifest,manifest_raw=load(code_root/"payload_manifest.json")
    bundle_pin=cfg.get("source_bundle",{})
    expected_bundle_sha=require_hex(bundle_pin.get("archive_sha256",""),"source archive digest")
    expected_manifest_sha=require_hex(bundle_pin.get("manifest_sha256",""),"source manifest digest")
    if bundle_manifest.get("schema")!=SOURCE_BUNDLE_SCHEMA or len(bundle_manifest.get("members",[]))!=EXPECTED_STAGE_MEMBERS:
        raise NotReady("staged rescore bundle manifest mismatch")
    if len(manifest_raw)!=int(bundle_pin.get("manifest_bytes",-1)) or sha(manifest_raw)!=expected_manifest_sha:
        raise NotReady("staged bundle manifest raw digest mismatch")
    stage_path=Path(paths["source_stage_receipt_path"])
    stage,stage_raw=load(stage_path)
    if stage.get("schema")!="cross_tfg_fit40_final120_rescore_source_stage_receipt_v1" or stage.get("status")!="PASS_SOURCE_ONLY_FIT40_FINAL120_RESCORE_PAYLOAD_EXTRACTED":
        raise NotReady("source-stage receipt is not the expected complete source-only stage")
    scope=stage.get("scope",{})
    if any(scope.get(k) is not False for k in ("model_imported","cuda_initialized","optimizer_updated","render_or_score","development_audio_opened","confirmation_audio_opened")):
        raise NotReady("source stage exceeded code-only scope")
    archive=stage.get("archive",{}); bundle=stage.get("bundle_manifest",{})
    if archive.get("sha256")!=expected_bundle_sha or bundle.get("sha256")!=expected_manifest_sha or int(archive.get("bytes",-1))!=int(bundle_pin.get("archive_bytes",-2)):
        raise NotReady("stage receipt archive/manifest pins differ from this packet")
    stage_members=stage.get("members",[])
    expected={m["archive_path"]:(int(m["bytes"]),str(m["sha256"]).lower()) for m in bundle_manifest["members"]}
    if len(stage_members)!=EXPECTED_STAGE_MEMBERS or {m.get("archive_path") for m in stage_members}!=set(expected):
        raise NotReady("staged member set differs from bundle manifest")
    for m in stage_members:
        n=m["archive_path"]; p=regular(Path(m["path"])); b,h=expected[n]
        if p.stat().st_size!=b or sha_file(p)!=h or int(m.get("bytes",-1))!=b or str(m.get("sha256","")).lower()!=h:
            raise NotReady("staged source member changed: "+n)

    producer_cfg_path=repo/".codex/scratch/fit40_native_rms_geometry_packet_attempt04_revision02/sealed_packet_attempt04_revision02/payload/fit40_native_rms_geometry_config_attempt04.json"
    train_cfg_path=repo/".codex/scratch/fit40_transition_head_training_packet_attempt05_revision02/fit40_transition_head_v3_late_binding_config_attempt05.json"
    producer_cfg,producer_cfg_raw=load(producer_cfg_path,15576,"eb32966df34bcf29750805463aa4a9bc8e1084c8700bc6e8571cde31527057c2")
    train_cfg,train_cfg_raw=load(train_cfg_path,2228,"799564da00f9b7a71a818d838967fe356bef37e35ce9305938ac6eeec8ceee8e")
    producer_remote=producer_cfg.get("remote_paths",{})
    producer_outputs=producer_cfg.get("outputs",{})
    expected_geometry_path=str(Path(producer_remote.get("output_root", ""))/str(producer_outputs.get("geometry_manifest_relative_path", "")))
    expected_cache_path=str(Path(producer_remote.get("output_root", ""))/str(producer_outputs.get("natural_audio_cache_manifest_relative_path", "")))
    if expected_geometry_path!=paths.get("geometry_manifest_path") or expected_cache_path!=paths.get("cache_manifest_path"):
        raise NotReady("geometry/cache path differs from sealed producer configuration")
    if producer_remote.get("run_state_path")!=paths.get("producer_state_path") or producer_remote.get("execution_receipt_path")!=paths.get("producer_execution_receipt_path"):
        raise NotReady("producer terminal-state paths differ from sealed producer configuration")
    if train_cfg.get("remote_paths",{}).get("training_output_dir")!=paths.get("training_output_dir"):
        raise NotReady("training output path differs from sealed training config")
    if train_cfg.get("remote_paths",{}).get("source_root")!=paths.get("source_root"):
        raise NotReady("fit input root differs from sealed training config")
    repo_files={
      "parent_plan":(repo/"runs/cross_tfg_audio_head_20261003/fit_v2_fullcontext_attempt02/plan.json",63205,EXPECTED_PARENT_RAW),
      "fit_source_manifest":(repo/"runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01/fit40_source_transfer_manifest.json",EXPECTED_SOURCE_BYTES,EXPECTED_SOURCE_MANIFEST),
      "head_config":(repo/"runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02/qualification_config.json",EXPECTED_HEAD_CONFIG_BYTES,EXPECTED_HEAD_CONFIG),
      "fit_runner":(repo/"scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_attempt04.py",72032,EXPECTED_RUNNER),
      "geometry_producer":(repo/"scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt04.py",EXPECTED_PRODUCER_BYTES,EXPECTED_PRODUCER),
      "rescorer":(repo/"scripts/experiments/cross_tfg_audio_head_20261003/rescore_fit40_native_transition_head_v3_final120_attempt04.py",None,EXPECTED_RESCORE),
    }
    actual={}
    for name,(path,nbytes,digest) in repo_files.items():
        q=regular(path); raw=q.read_bytes()
        if (nbytes is not None and len(raw)!=nbytes) or sha(raw)!=digest: raise NotReady("source pin mismatch: "+name)
        actual[name]={"path":str(q),"bytes":len(raw),"sha256":sha(raw)}
    parent, parent_raw=load(repo_files["parent_plan"][0])
    body={k:v for k,v in parent.items() if k!="plan_payload_sha256"}
    from_payload=sha(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8"))
    if parent.get("plan_payload_sha256")!=EXPECTED_PARENT_PAYLOAD or from_payload!=EXPECTED_PARENT_PAYLOAD:
        raise NotReady("canonical parent-plan payload pin mismatch")
    src,source_raw=load(repo_files["fit_source_manifest"][0])
    if len(src.get("rows",[]))!=40 or src.get("status")!="PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise NotReady("fit source manifest does not certify the full forty-row fit roster")
    source_ids=[str(r.get("row_id")) for r in src["rows"]]
    if len(set(source_ids))!=40: raise NotReady("source row IDs are not unique")
    source_root=Path(paths["source_root"]).resolve(strict=True)
    source_files=[]
    for row in src["rows"]:
        rel=Path(str(row["transfer_relative_path"]))
        file_path=source_root/rel
        p=regular(file_path)
        if not p.resolve().is_relative_to(source_root): raise NotReady("source WAV escaped the pinned fit input root")
        raw=p.read_bytes()
        if len(raw)!=int(row["transfer_file_bytes"]) or sha(raw)!=str(row["transfer_file_sha256"]).lower():
            raise NotReady("fit PCM16 source member pin mismatch: "+str(row["row_id"]))
        source_files.append({"row_id":row["row_id"],"path":str(p),"bytes":len(raw),"sha256":sha(raw)})

    producer_paths={"execution_receipt_path": producer_remote["execution_receipt_path"], "worker_state_path": producer_remote["run_state_path"]}
    pstate,pstate_raw=load(Path(paths["producer_state_path"]))
    pexec,pexec_raw=load(Path(paths["producer_execution_receipt_path"]))
    geom_path=Path(paths["geometry_manifest_path"])
    cache_path=Path(paths["cache_manifest_path"])
    geometry,geometry_raw=load(geom_path)
    cache,cache_raw=load(cache_path)
    gpin=pstate.get("geometry_manifest",{})
    cpin=pstate.get("audio_cache_manifest",{})
    if pstate.get("status")!="COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY" or int(pstate.get("completed_rows",-1))!=40 or int(pstate.get("expected_rows",-1))!=40:
        raise NotReady("fit40 producer is not complete and terminal")
    if pexec.get("schema")!="cross_tfg_fit40_native_rms_geometry_execution_receipt_v1" or pexec.get("status")!="COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY" or int(pexec.get("completed_rows",-1))!=40 or int(pexec.get("expected_rows",-1))!=40:
        raise NotReady("fit40 producer execution receipt is not complete for all forty rows")
    if pstate.get("model_score_computed") is not False or pstate.get("training_performed") is not False or pstate.get("development_audio_opened") is not False or pstate.get("confirmation_audio_opened") is not False:
        raise NotReady("producer scope flags are not fit-only engineering")
    if gpin.get("path")!=str(geom_path) or gpin.get("bytes")!=len(geometry_raw) or gpin.get("sha256")!=sha(geometry_raw):
        raise NotReady("producer state geometry pin mismatch")
    if cpin.get("path")!=str(cache_path) or cpin.get("bytes")!=len(cache_raw) or cpin.get("sha256")!=sha(cache_raw):
        raise NotReady("producer state audio-cache pin mismatch")
    if geometry.get("status")!="COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY" or cache.get("row_count")!=40:
        raise NotReady("producer geometry/cache terminal status mismatch")
    geometry_payload={k:v for k,v in geometry.items() if k!="payload_sha256"}
    geometry_payload_sha=sha(json.dumps(geometry_payload,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8"))
    if geometry.get("payload_sha256")!=geometry_payload_sha:
        raise NotReady("geometry canonical payload digest mismatch")
    cache_payload={k:v for k,v in cache.items() if k!="manifest_sha256"}
    cache_payload_sha=sha(json.dumps(cache_payload,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8"))
    if cache.get("manifest_sha256")!=cache_payload_sha:
        raise NotReady("natural audio cache canonical payload digest mismatch")
    geometry_ids=[str(r.get("row_id")) for r in geometry.get("rows",[])]
    cache_ids=[str(r.get("row_id")) for r in cache.get("rows",[])]
    if geometry_ids!=source_ids or cache_ids!=source_ids: raise NotReady("geometry/cache rows do not equal exact fit manifest order")
    if geometry.get("source_pins",{}).get("source_manifest",{}).get("sha256")!=EXPECTED_SOURCE_MANIFEST:
        raise NotReady("geometry source manifest pin mismatch")
    geometry_pin={"path":str(geom_path.resolve(strict=True)),"bytes":len(geometry_raw),"sha256":sha(geometry_raw)}
    cache_pin={"path":str(cache_path.resolve(strict=True)),"bytes":len(cache_raw),"sha256":sha(cache_raw)}

    training_root=Path(paths["training_output_dir"]).resolve(strict=True)
    completion,completion_raw=load(training_root/"completion.json")
    status,_=load(training_root/"status.json")
    if completion.get("schema")!="cross_tfg_fit40_native_transition_head_v3_training_runner_attempt02" or completion.get("status")!="COMPLETE_FIT40_V3_120_STEPS" or int(completion.get("completed_optimizer_steps",-1))!=120 or int(completion.get("fit_rows",-1))!=40 or int(completion.get("fit_speakers",-1))!=20 or int(completion.get("zero_gate_rows_passed",-1))!=40:
        raise NotReady("training completion is not the exact fit40/120-step terminal")
    if completion.get("selection_record_created") is not False or completion.get("development_audio_opened") is not False or completion.get("confirmation_audio_opened") is not False:
        raise NotReady("training terminal is selected or accessed held-out audio")
    if status.get("stage")!="complete" or status.get("status")!=completion.get("status"):
        raise NotReady("training status file is not terminal/consistent")
    artifacts={}
    for key in ("training_record","checkpoint","runtime_source_pins","baseline_arrays","zero_gate_arrays","training_score_arrays","training_steps"):
        q=verify_ref(completion.get(key),key)
        artifacts[key]=pin(q)
    runtime,runtime_raw=load(Path(artifacts["runtime_source_pins"]["path"]))
    record,record_raw=load(Path(artifacts["training_record"]["path"]))
    ckpt,ckpt_raw=load(Path(artifacts["checkpoint"]["path"]))
    if record.get("completed_optimizer_steps")!=120 or record.get("planned_optimizer_steps")!=120 or record.get("selection_record_created") is not False:
        raise NotReady("training record does not certify the full unselected 120 updates")
    if ckpt.get("schema")!="cross_tfg_v3_parameter_checkpoint_v1" or ckpt.get("selection_status")!="UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED" or ckpt.get("training_artifact_bindings",{}).get("completed_optimizer_steps")!=120:
        raise NotReady("final checkpoint is not the unselected post-update step-120 checkpoint")
    theta=ckpt.get("parameter",{}).get("values")
    step_rows=[json.loads(line) for line in Path(artifacts["training_steps"]["path"]).read_text(encoding="utf-8").splitlines()]
    if len(step_rows)!=120 or [int(r.get("step",-1)) for r in step_rows]!=list(range(1,121)) or step_rows[-1].get("theta_after_update")!=theta:
        raise NotReady("step ledger does not bind checkpoint theta after update 120")
    runtime_geom=runtime.get("native_geometry_manifest",{})
    if runtime_geom.get("sha256")!=geometry_pin["sha256"] or record.get("native_geometry_manifest_sha256")!=geometry_pin["sha256"]:
        raise NotReady("training record/runtime geometry hash differs from actual producer output")
    gidentity=geometry.get("runtime_gpu_identity",{})
    tidentity=completion.get("gpu_identity",{})
    uuid=str(gidentity.get("uuid","")).lower()
    if not uuid or uuid!=str(tidentity.get("uuid","")).lower():
        raise NotReady("producer and training GPU UUIDs differ")
    device=str(tidentity.get("runtime_device",""))
    if device not in ("cuda","cuda:0",f"cuda:{int(tidentity.get('cuda_device_index',-1))}"):
        raise NotReady("training runtime device literal is invalid")
    if gidentity.get("device_literal")!=device:
        raise NotReady("training and geometry producer device literals differ")
    train_cfg_paths=train_cfg.get("remote_paths",{})
    train_binding_path=Path(train_cfg_paths["binding_output_path"])
    train_binding,train_binding_raw=load(train_binding_path)
    if train_binding.get("execution_started") is not False or train_binding.get("optimizer_created_or_updated") is not False:
        raise NotReady("training launch binding is malformed")
    if train_binding.get("training_output_dir")!=str(training_root):
        raise NotReady("training launch binding output path mismatch")
    env=train_binding.get("environment")
    argv0=train_binding.get("argv",[])
    if not isinstance(env,dict) or not isinstance(argv0,list) or not argv0 or not Path(str(argv0[0])).is_file():
        raise NotReady("actual training Python/environment binding is absent")
    rescore_path=repo_files["rescorer"][0].resolve(strict=True)
    output_dir=Path(paths["rescore_output_dir"])
    if output_dir.exists() or output_dir.is_symlink(): raise NotReady("fresh final-rescore output path is already occupied")
    argv=[
      str(argv0[0]),"-B",str(rescore_path),
      "--training-output-dir",str(training_root),
      "--parent-plan",str(repo_files["parent_plan"][0].resolve(strict=True)),
      "--source-manifest",str(repo_files["fit_source_manifest"][0].resolve(strict=True)),
      "--source-root",str(source_root),
      "--geometry-manifest",str(geom_path.resolve(strict=True)),
      "--expected-geometry-manifest-sha256",geometry_pin["sha256"],
      "--head-config",str(repo_files["head_config"][0].resolve(strict=True)),
      "--output-dir",str(output_dir),
      "--device",device,
    ]
    return {
      "schema":SCHEMA,"status":"READY_FOR_FRESH_FINAL120_FIT_ONLY_RESCORE",
      "scope":{"fit_rows":40,"speakers":20,"completed_optimizer_steps":120,"development_audio_opened":False,"confirmation_audio_opened":False,"checkpoint_selected":False,"parameters_updated":False},
      "source_stage_receipt":pin(stage_path),"staged_bundle_manifest":pin(code_root/"payload_manifest.json"),
      "static_config":{"path":str(config_path.resolve(strict=True)),"bytes":len(cfg_raw),"sha256":sha(cfg_raw)},
      "frozen_sources":actual,"source_manifest":pin(repo_files["fit_source_manifest"][0]),"source_audio_members":source_files,
      "parent_plan":pin(repo_files["parent_plan"][0]),"head_config":pin(repo_files["head_config"][0]),
      "producer_execution_receipt":pin(Path(producer_paths["execution_receipt_path"])),
      "producer_worker_state":pin(Path(producer_paths["worker_state_path"])),"native_geometry_manifest":geometry_pin,"natural_audio_cache_manifest":cache_pin,
      "training_completion":{"path":str((training_root/"completion.json").resolve(strict=True)),"bytes":len(completion_raw),"sha256":sha(completion_raw)},
      "training_status":pin(training_root/"status.json"),"training_artifacts":artifacts,
      "training_record":pin(Path(artifacts["training_record"]["path"])),"checkpoint":pin(Path(artifacts["checkpoint"]["path"])),
      "training_runtime_source_pins":pin(Path(artifacts["runtime_source_pins"]["path"])),"training_launch_binding":pin(train_binding_path),
      "gpu_identity":{"uuid":uuid,"device":device,"geometry_identity":gidentity,"training_identity":tidentity},
      "working_directory":str(repo),"source_root":str(source_root),"environment":env,"argv":argv,"rescore_output_dir":str(output_dir),
      "execution_started":False,"optimizer_created_or_updated":False,
      "admission":{"stable_samples":2,"interval_seconds":5,"minimum_free_gpu_mib":18432,"compute_pids_allowed":0,"minimum_shm_free_bytes":4294967296,"same_uuid":uuid},
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--expected-config-sha256",required=True)
    ap.add_argument("--write-binding",action="store_true")
    a=ap.parse_args()
    try:
        expected=require_hex(a.expected_config_sha256,"config digest")
        config,raw=load(a.config,expected_sha=expected)
        binding=make_binding(config,raw,a.config)
        if not a.write_binding:
            print(json.dumps(binding,sort_keys=True,ensure_ascii=False,allow_nan=False)); return 0
        path=Path(config["remote_paths"]["binding_output_path"])
        result=write_exclusive(path,binding)
        print(json.dumps({"schema":SCHEMA,"status":binding["status"],"binding":result,"execution_started":False},sort_keys=True));return 0
    except Exception as e:
        print(json.dumps({"schema":SCHEMA,"status":"NOT_READY","error_type":type(e).__name__,"error":str(e)},sort_keys=True),file=sys.stderr)
        return 2

if __name__=="__main__": raise SystemExit(main())
