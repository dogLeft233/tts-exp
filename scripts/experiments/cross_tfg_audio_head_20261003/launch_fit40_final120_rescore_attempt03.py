#!/usr/bin/env python3
"""Fail-closed same-GPU admission and detached launch for the final fit40 rescore."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

SCHEMA="cross_tfg_fit40_final120_rescore_launch_receipt_v1"
def sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def write_once(path:Path,value:dict):
    if path.exists() or path.is_symlink(): raise RuntimeError("refuse overwrite: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def verify_all_pins(obj):
    seen=set()
    def walk(x):
        if isinstance(x,dict):
            if {"path","bytes","sha256"}<=set(x) and isinstance(x["path"],str):
                p=Path(x["path"])
                k=str(p)
                if k not in seen:
                    seen.add(k)
                    if p.is_symlink() or not p.is_file() or p.stat().st_size!=int(x["bytes"]) or sha_file(p)!=str(x["sha256"]).lower():
                        raise RuntimeError("late-bound input changed: "+k)
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(obj)
    return len(seen)
def gpu_snapshot(expected_uuid:str,min_free_mib:int):
    q=subprocess.run(["nvidia-smi","--query-gpu=uuid,index,memory.free,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)
    selected=None
    for line in q.stdout.splitlines():
        parts=[v.strip() for v in line.split(",")]
        if len(parts)!=4: raise RuntimeError("unparseable nvidia-smi GPU row")
        if parts[0].lower()==expected_uuid.lower():
            selected={"uuid":parts[0],"index":int(parts[1]),"memory_free_mib":int(parts[2]),"memory_total_mib":int(parts[3])}
    if selected is None: raise RuntimeError("training/geometry GPU UUID is absent")
    apps=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)
    pids=[]
    for line in apps.stdout.splitlines():
        value=line.strip()
        if value and value.lower() not in ("no running processes found","n/a"): pids.append(value)
    return {"gpu":selected,"compute_pids":pids,"free_ok":selected["memory_free_mib"]>=min_free_mib}
def proc_start_ticks(pid:int):
    raw=Path(f"/proc/{pid}/stat").read_text()
    end=raw.rfind(")")
    fields=raw[end+2:].split()
    return int(fields[19])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binding",type=Path,required=True)
    ap.add_argument("--expected-binding-sha256",required=True)
    ap.add_argument("--launch-receipt",type=Path,required=True)
    ap.add_argument("--stdout-log",type=Path,required=True)
    ap.add_argument("--stderr-log",type=Path,required=True)
    ap.add_argument("--wait-timeout-seconds",type=int,required=True)
    a=ap.parse_args()
    if len(a.expected_binding_sha256)!=64 or any(c not in "0123456789abcdef" for c in a.expected_binding_sha256.lower()):
        raise RuntimeError("binding SHA must be 64 lowercase hex")
    if a.binding.is_symlink() or not a.binding.is_file() or sha_file(a.binding)!=a.expected_binding_sha256.lower():
        raise RuntimeError("late-binding raw SHA mismatch")
    b=json.loads(a.binding.read_text(encoding="utf-8"))
    if b.get("schema")!="cross_tfg_fit40_final120_rescore_late_binding_v1" or b.get("status")!="READY_FOR_FRESH_FINAL120_FIT_ONLY_RESCORE":
        raise RuntimeError("late binding is not ready")
    if b.get("execution_started") is not False or b.get("optimizer_created_or_updated") is not False:
        raise RuntimeError("late binding already claims execution or update")
    if b.get("scope")!={"fit_rows":40,"speakers":20,"completed_optimizer_steps":120,"development_audio_opened":False,"confirmation_audio_opened":False,"checkpoint_selected":False,"parameters_updated":False}:
        raise RuntimeError("scope differs from frozen fit40/final120 scope")
    pinned_count=verify_all_pins(b)
    # The command/environment are reconstructed from the actual fit launch binding.
    training_binding_pin=b.get("training_launch_binding",{})
    if not isinstance(training_binding_pin,dict) or not isinstance(training_binding_pin.get("path"),str):
        raise RuntimeError("actual training launch binding pin is missing")
    training_binding=json.loads(Path(training_binding_pin["path"]).read_text(encoding="utf-8"))
    if b.get("environment")!=training_binding.get("environment"):
        raise RuntimeError("rescore environment differs from the executed training environment")
    if b.get("argv",[])[:1]!=training_binding.get("argv",[])[:1]:
        raise RuntimeError("rescore interpreter differs from the executed training interpreter")
    gi=b.get("gpu_identity",{})
    geometry_gpu=gi.get("geometry_identity",{})
    training_gpu=gi.get("training_identity",{})
    bound_uuid=str(gi.get("uuid","")).lower()
    if not bound_uuid or bound_uuid!=str(geometry_gpu.get("uuid","")).lower() or bound_uuid!=str(training_gpu.get("uuid","")).lower():
        raise RuntimeError("producer/training UUIDs do not agree in the late binding")
    if gi.get("device")!=geometry_gpu.get("device_literal") or gi.get("device")!=training_gpu.get("runtime_device"):
        raise RuntimeError("producer/training device literals do not agree")
    sources=b.get("frozen_sources",{})
    completion_path=Path(b.get("training_completion",{}).get("path",""))
    geometry_pin=b.get("native_geometry_manifest",{})
    expected_argv=[
      str(training_binding.get("argv",[""])[0]),"-B",str(sources.get("rescorer",{}).get("path","")),
      "--training-output-dir",str(completion_path.parent),
      "--parent-plan",str(b.get("parent_plan",{}).get("path","")),
      "--source-manifest",str(b.get("source_manifest",{}).get("path","")),
      "--source-root",str(b.get("source_root","")),
      "--geometry-manifest",str(geometry_pin.get("path","")),
      "--expected-geometry-manifest-sha256",str(geometry_pin.get("sha256","")),
      "--head-config",str(b.get("head_config",{}).get("path","")),
      "--output-dir",str(b.get("rescore_output_dir","")),
      "--device",str(gi.get("device","")),
    ]
    if b.get("argv")!=expected_argv:
        raise RuntimeError("late-bound rescorer argv differs from pinned fit-only inputs")
    if b.get("working_directory")!=str(Path(sources.get("fit_runner",{}).get("path","")).parents[3]):
        raise RuntimeError("late-bound working directory differs from staged source tree")
    gate=b.get("admission",{})
    expected_uuid=str(gate.get("same_uuid",""))
    min_free=int(gate.get("minimum_free_gpu_mib",-1))
    min_shm=int(gate.get("minimum_shm_free_bytes",-1))
    if gate.get("stable_samples")!=2 or gate.get("interval_seconds")!=5 or gate.get("compute_pids_allowed")!=0 or min_free!=18432 or min_shm!=4294967296 or not expected_uuid:
        raise RuntimeError("admission gate differs from the reviewed strict policy")
    output=Path(b["rescore_output_dir"])
    if output.exists() or output.is_symlink(): raise RuntimeError("rescore output path is already occupied")
    stable=[]; start=time.monotonic(); deadline=start+a.wait_timeout_seconds
    while time.monotonic()<deadline:
        snap=gpu_snapshot(expected_uuid,min_free)
        shm_free=shutil.disk_usage("/dev/shm").free
        good=snap["free_ok"] and not snap["compute_pids"] and shm_free>=min_shm
        current={"observed_unix_seconds":time.time(),**snap,"shm_free_bytes":shm_free,"pass":good}
        if good:
            if stable and current["observed_unix_seconds"]-stable[-1]["observed_unix_seconds"]>=5.0:
                stable.append(current)
            elif not stable:
                stable.append(current)
            else:
                stable=[current]
            if len(stable)>=2: break
        else:
            stable=[]
        time.sleep(5)
    if len(stable)<2: raise RuntimeError("GPU/SHM admission did not reach two stable samples before timeout")
    verify_all_pins(b)
    argv=b.get("argv")
    env=b.get("environment")
    cwd=Path(b.get("working_directory",""))
    if not isinstance(argv,list) or not argv or not isinstance(env,dict) or not cwd.is_dir():
        raise RuntimeError("late-bound argv/environment/working directory is malformed")
    child_env=os.environ.copy()
    child_env.update({str(k):str(v) for k,v in env.items()})
    for path in (a.launch_receipt,a.stdout_log,a.stderr_log):
        if path.exists() or path.is_symlink(): raise RuntimeError("launch output already exists: "+str(path))
    a.stdout_log.parent.mkdir(parents=True,exist_ok=True)
    with a.stdout_log.open("xb") as out, a.stderr_log.open("xb") as err:
        proc=subprocess.Popen([str(x) for x in argv],cwd=str(cwd),env=child_env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
    ticks=proc_start_ticks(proc.pid)
    receipt={
      "schema":SCHEMA,"status":"STARTED","binding":{"path":str(a.binding.resolve()),"bytes":a.binding.stat().st_size,"sha256":sha_file(a.binding)},
      "child":{"pid":proc.pid,"start_ticks":ticks,"argv":argv,"working_directory":str(cwd)},
      "verified_unique_pinned_files":pinned_count,"gpu_admission_samples":stable,
      "stdout_log":str(a.stdout_log.resolve()),"stderr_log":str(a.stderr_log.resolve()),
      "optimizer_or_checkpoint_update_by_launcher":False,"development_audio_opened_by_launcher":False,"confirmation_audio_opened_by_launcher":False,
      "launched_unix_seconds":time.time()
    }
    print(json.dumps({"status":"STARTED","pid":proc.pid,"start_ticks":ticks,"pinned_files":pinned_count},sort_keys=True))
    write_once(a.launch_receipt,receipt)
    return 0
if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"schema":SCHEMA,"status":"LAUNCH_FAILED","error_type":type(exc).__name__,"error":str(exc)},sort_keys=True),file=sys.stderr)
        raise SystemExit(2)
