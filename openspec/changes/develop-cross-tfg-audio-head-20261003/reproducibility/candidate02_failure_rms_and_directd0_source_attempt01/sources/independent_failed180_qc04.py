from __future__ import annotations
import argparse, ast, collections, hashlib, json, os, stat
from pathlib import Path

MAX_LEDGER=8*1024*1024
MAX_META=256*1024
MAX_EXEC=2*1024*1024
EXPECTED_RUN_ID="cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt02"
EXPECTED_MODELS=("wav2lip_gan_bac9a81e","musetalk_fixed_portrait_attempt16","ditto_hubert_fixed_portrait_attempt18")
EXPECTED_METHOD="shared_recipient_transition_head_v3_fit40_step120"
EXPECTED_SEEDS=(20261003,20261004,20261005)
SEED_ROWS_PER_SEED=60
MODEL_SEED_ROWS=20
FAILURE_TEXT="V3CLIError: cannot establish configured PyTorch interop threads: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called"
EMPTY_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def payload_sha(value,field): return sha(canonical({k:v for k,v in value.items() if k!=field}))
def fail(message): raise ValueError(message)
def read_bounded(path,cap):
    path=Path(path)
    before=path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode): fail("not a regular nonsymlink file: "+str(path))
    if before.st_size>cap: fail("bounded read cap exceeded: "+str(path))
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0)
    fd=os.open(str(path),flags)
    try:
        opened=os.fstat(fd)
        if (opened.st_dev,opened.st_ino,opened.st_size)!=(before.st_dev,before.st_ino,before.st_size): fail("file changed before read: "+str(path))
        parts=[]; total=0
        while True:
            chunk=os.read(fd,min(1024*1024,cap+1-total))
            if not chunk: break
            parts.append(chunk); total+=len(chunk)
            if total>cap: fail("file exceeded read cap: "+str(path))
        after=os.fstat(fd)
        if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)!=(opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns) or total!=opened.st_size: fail("file changed during read: "+str(path))
        return b"".join(parts)
    finally: os.close(fd)
def read_json(path,cap):
    raw=read_bounded(path,cap)
    obj=json.loads(raw.decode("utf-8-sig"))
    if not isinstance(obj,dict): fail("expected JSON object: "+str(path))
    return obj,raw
def check_pin(pin):
    if not isinstance(pin,dict) or set(pin)!={"path","bytes","sha256"}: fail("malformed flat source_evidence pin")
    raw=read_bounded(pin["path"],max(int(pin["bytes"]),1))
    if len(raw)!=int(pin["bytes"]) or sha(raw)!=str(pin["sha256"]).lower(): fail("source evidence pin mismatch: "+str(pin["path"]))
    return raw
def parse_jsonl(raw,label):
    if raw and not raw.endswith(b"\n"): fail(label+" lacks canonical final newline")
    output=[]
    for i,line in enumerate(raw.splitlines(),1):
        if not line: fail(label+" contains a blank row at line "+str(i))
        try: obj=json.loads(line.decode("utf-8"))
        except Exception as exc: fail(label+" malformed JSON line "+str(i)+": "+str(exc))
        if not isinstance(obj,dict): fail(label+" line is not a JSON object: "+str(i))
        if canonical(obj)+b"\n"!=line+b"\n": fail(label+" line is not producer-canonical JSON: "+str(i))
        output.append(obj)
    return output
def main():
    parser=argparse.ArgumentParser(description="Read-only local QC for the failed candidate180 terminal packet; this does not admit efficacy results.")
    parser.add_argument("--index",required=True,help="QC-only request_index_attempt04.json; reuses attempt03 download responses")
    parser.add_argument("--report",required=True,help="new output path; created exclusively after all QC gates pass")
    args=parser.parse_args()
    index_path=Path(args.index).resolve(strict=True)
    index,index_raw=read_json(index_path,2*1024*1024)
    saved_index_sha=index.get("index_sha256")
    if saved_index_sha!=payload_sha(index,"index_sha256"): fail("download index self hash mismatch")
    if index.get("schema")!="cross_tfg_candidate180_failure_terminal_download_packet_v1" or index.get("run_id")!=EXPECTED_RUN_ID: fail("packet identity/schema mismatch")
    if index.get("qc_attempt")!="attempt04" or index.get("packet_role")!="post_download_qc_only_not_for_dispatch": fail("QC04 index purpose/attempt mismatch")
    if index.get("scope",{}).get("remote_dispatch_performed") is not False or index.get("scope",{}).get("scientific_admission") is not False: fail("packet scope is not read-only failure QC")
    pins=index.get("source_evidence")
    if not isinstance(pins,dict) or not pins: fail("flat source_evidence map is missing")
    source_raw={key:check_pin(value) for key,value in pins.items()}
    refs=index["source_refs"]
    needed_refs=("parent_packet_index_key","superseded_qc_script_key","local_qc_script_key","qc_manifest_key")
    if any(not refs.get(key) for key in needed_refs): fail("QC04 source references are incomplete")
    parent_raw=source_raw[refs["parent_packet_index_key"]]
    parent=json.loads(parent_raw.decode("utf-8-sig"))
    if sha(parent_raw)!=index.get("parent_packet_index_raw_sha256") or parent.get("index_sha256")!=payload_sha(parent,"index_sha256"):
        fail("immutable attempt03 packet-index raw/self pin mismatch")
    if parent.get("schema")!="cross_tfg_candidate180_failure_terminal_download_packet_v1" or parent.get("run_id")!=EXPECTED_RUN_ID:
        fail("parent attempt03 packet identity mismatch")
    if parent.get("requests")!=index.get("requests") or parent.get("frozen_input")!=index.get("frozen_input") or parent.get("strict_attestation")!=index.get("strict_attestation"):
        fail("QC04 changed sealed attempt03 download requests or frozen failure evidence")
    if parent.get("scope")!=index.get("scope"):
        fail("QC04 changed parent packet scope")
    parent_script_pin=pins[refs["superseded_qc_script_key"]]
    current_script_pin=pins[refs["local_qc_script_key"]]
    qc_manifest_pin=pins[refs["qc_manifest_key"]]
    qc_manifest=json.loads(source_raw[refs["qc_manifest_key"]].decode("utf-8-sig"))
    if qc_manifest.get("schema")!="cross_tfg_candidate180_failure_qc_manifest_v1" or qc_manifest.get("attempt")!="attempt04":
        fail("QC04 manifest identity mismatch")
    if qc_manifest.get("manifest_sha256")!=payload_sha(qc_manifest,"manifest_sha256"):
        fail("QC04 manifest self hash mismatch")
    if qc_manifest.get("parent_packet_index")!={"path":pins[refs["parent_packet_index_key"]]["path"],"bytes":pins[refs["parent_packet_index_key"]]["bytes"],"sha256":pins[refs["parent_packet_index_key"]]["sha256"],"index_sha256":parent.get("index_sha256")}:
        fail("QC04 manifest does not bind immutable parent packet index")
    if qc_manifest.get("superseded_qc03_script")!={"path":parent_script_pin["path"],"bytes":parent_script_pin["bytes"],"sha256":parent_script_pin["sha256"]}:
        fail("QC04 manifest does not bind the unchanged QC03 source")
    if qc_manifest.get("qc04_script")!={"path":current_script_pin["path"],"bytes":current_script_pin["bytes"],"sha256":current_script_pin["sha256"]}:
        fail("QC04 manifest does not bind the QC04 source")
    request_pin_rows=[{"name":item["name"],"path":item["path"],"bytes":item["bytes"],"sha256":item["sha256"],"remote":item["remote"],"local":item["local"],"expected_remote_bytes":item["expected_remote_bytes"],"expected_remote_sha256":item["expected_remote_sha256"]} for item in index["requests"]]
    if qc_manifest.get("download_request_pins")!=request_pin_rows or qc_manifest.get("download_request_count")!=6:
        fail("QC04 manifest request pins differ from the immutable six-request packet")
    if qc_manifest.get("expected_seed_roster")!=list(EXPECTED_SEEDS) or qc_manifest.get("rows_per_seed")!={str(seed):SEED_ROWS_PER_SEED for seed in EXPECTED_SEEDS}:
        fail("QC04 manifest seed roster contract mismatch")
    if qc_manifest.get("model_rows_per_seed")!={model:{str(seed):MODEL_SEED_ROWS for seed in EXPECTED_SEEDS} for model in EXPECTED_MODELS}:
        fail("QC04 manifest model-by-seed contract mismatch")
    if qc_manifest.get("expected_identity_fields")!=["row_id","split","sample_id","speaker_id","model_id","audio_frontend","model_role","face_id","seed","method_id","method_config_sha256","checkpoint_sha256"]:
        fail("QC04 manifest identity-field contract mismatch")
    if qc_manifest.get("strict_attestation")!="ABSENT_PER_FAILURE_DIAGNOSTIC_NO_DOWNLOAD_REQUEST":
        fail("QC04 manifest changes strict-attestation absence")
    plan_path=Path(pins[refs["candidate_plan_key"]]["path"])
    rows_path=Path(pins[refs["planned_rows_key"]]["path"])
    core_path=Path(pins[refs["stagea_core_key"]]["path"])
    guard_path=Path(pins[refs["candidate_guard_key"]]["path"])
    manifest14_path=Path(pins[refs["plan_manifest_key"]]["path"])
    runtime_path=Path(pins[refs["runtime_binding_key"]]["path"])
    diag_response_path=Path(pins[refs["failure_diagnostic_response_key"]]["path"])
    diag_response,_=read_json(diag_response_path,MAX_META*2)
    if not diag_response.get("ok") or diag_response.get("code")!=0 or diag_response.get("timed_out") or diag_response.get("incomplete_stream") or diag_response.get("stream_error") or diag_response.get("truncated",{}).get("stdout") or diag_response.get("truncated",{}).get("stderr"):
        fail("pinned failure-diagnostic response is incomplete")
    diag=json.loads(diag_response["stdout"])
    if diag.get("schema")!="cross_tfg_candidate180_failure_diagnostic_v1" or diag.get("status")!="READ_ONLY_FAILURE_DIAGNOSTIC_COMPLETE": fail("failure diagnostic identity/status mismatch")
    launch_receipt=json.loads(source_raw["actual_launch_receipt"].decode("utf-8-sig"))
    if launch_receipt.get("schema")!="cross_tfg_candidate180_detached_launch_receipt_v1" or launch_receipt.get("status")!="STARTED_DETACHED" or launch_receipt.get("stage_id")!="candidate180" or launch_receipt.get("run_id")!=EXPECTED_RUN_ID: fail("actual launch receipt identity/status mismatch")
    if launch_receipt.get("receipt_sha256")!=payload_sha(launch_receipt,"receipt_sha256"): fail("actual launch receipt self hash mismatch")
    if launch_receipt.get("output_root")!=diag.get("identity",{}).get("candidate_root") or launch_receipt.get("pid")!=diag.get("identity",{}).get("process",{}).get("pid") or launch_receipt.get("start_ticks")!=diag.get("identity",{}).get("process",{}).get("expected_start_ticks"): fail("diagnostic process identity differs from actual launch receipt")
    plan_raw=check_pin(pins[refs["candidate_plan_key"]])
    rows_raw=check_pin(pins[refs["planned_rows_key"]])
    manifest14_raw=check_pin(pins[refs["plan_manifest_key"]])
    runtime_raw=check_pin(pins[refs["runtime_binding_key"]])
    core_raw=check_pin(pins[refs["stagea_core_key"]])
    guard_raw=check_pin(pins[refs["candidate_guard_key"]])
    plan=json.loads(plan_raw.decode("utf-8-sig"))
    planned_rows=parse_jsonl(rows_raw,"frozen planned rows")
    if plan.get("status")!="FROZEN" or plan.get("schema_version")!="cross-tfg-audio-eval-v1" or plan.get("run_id")!=EXPECTED_RUN_ID: fail("frozen plan schema/status/run binding mismatch")
    if plan.get("plan_sha256")!=payload_sha(plan,"plan_sha256"): fail("candidate plan payload hash mismatch")
    if plan.get("plan_sha256")!="fee2718c0c4b18e5acabf3ca7d7103344bfb3859f3d8a3b93862c1ad4ce7d612": fail("candidate plan canonical payload pin differs")
    plan_manifest=json.loads(manifest14_raw.decode("utf-8-sig"))
    if plan_manifest.get("schema")!="cross_tfg_v3_candidate_only_dev_plan_v1" or plan_manifest.get("status")!="FROZEN" or plan_manifest.get("run_id")!=EXPECTED_RUN_ID: fail("candidate plan-manifest schema/status/run mismatch")
    if plan_manifest.get("manifest_sha256")!=payload_sha(plan_manifest,"manifest_sha256"): fail("candidate plan-manifest payload self hash mismatch")
    if plan_manifest.get("plan_raw_sha256")!=sha(plan_raw) or plan_manifest.get("plan_sha256")!=plan.get("plan_sha256") or plan_manifest.get("planned_rows_sha256")!=sha(rows_raw) or plan_manifest.get("planned_rows")!=180: fail("candidate plan-manifest plan/rows lineage mismatch")
    if plan_manifest.get("candidate_runtime_qualification_performed") is not False or plan_manifest.get("confirm_audio_read") is not False or plan_manifest.get("confirm_effects_read") is not False: fail("candidate plan-manifest scope changed")
    plan_execution=plan.get("execution",{})
    if plan_execution.get("samehost_runtime_binding",{}).get("raw_sha256")!=sha(runtime_raw) or plan_execution.get("samehost_candidate_guarded_launcher",{}).get("raw_sha256")!=sha(guard_raw): fail("candidate plan does not bind exact runtime/guard sources")
    if plan.get("rows")!=planned_rows or len(planned_rows)!=180 or len({r.get("row_id") for r in planned_rows})!=180: fail("plan and canonical rows are not exact ordered 180 rows")
    if launch_receipt.get("candidate_plan",{}).get("raw_sha256")!=sha(plan_raw) or launch_receipt.get("candidate_plan_manifest",{}).get("raw_sha256")!=sha(manifest14_raw) or launch_receipt.get("candidate_rows",{}).get("raw_sha256")!=sha(rows_raw) or launch_receipt.get("runtime_binding",{}).get("raw_sha256")!=sha(runtime_raw): fail("actual launch receipt does not bind the frozen plan/rows/runtime")
    if launch_receipt.get("source_pins",{}).get("candidate13_guard",{}).get("raw_sha256")!=sha(guard_raw): fail("actual launch receipt candidate-guard source pin mismatch")
    if sha(plan_raw)!=index["frozen_input"]["plan_raw_sha256"] or sha(rows_raw)!=index["frozen_input"]["planned_rows_raw_sha256"]: fail("frozen input raw pins mismatch")
    stagea_ast=ast.parse(core_raw.decode("utf-8"),filename=str(core_path))
    identity_fields=None
    for node in stagea_ast.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="IDENTITY_FIELDS" for t in node.targets):
            identity_fields=ast.literal_eval(node.value); break
    expected_identity=("row_id","split","sample_id","speaker_id","model_id","audio_frontend","model_role","face_id","seed","method_id","method_config_sha256","checkpoint_sha256")
    if tuple(identity_fields or ())!=expected_identity or len(expected_identity)!=12: fail("pinned producer 12-field identity contract mismatch")
    models=tuple(item.get("model_id") for item in plan.get("models",[]))
    if models!=EXPECTED_MODELS: fail("three-model roster/order differs from candidate plan contract")
    methods=plan.get("methods",[])
    if len(methods)!=1 or methods[0].get("method_id")!=EXPECTED_METHOD: fail("candidate method contract mismatch")
    seed_counts=collections.Counter(row.get("seed") for row in planned_rows)
    expected_seed_counts={seed:SEED_ROWS_PER_SEED for seed in EXPECTED_SEEDS}
    if dict(seed_counts)!=expected_seed_counts: fail("candidate 180 plan seed roster/counts differ from the frozen three-seed 3x60 contract")
    model_seed_counts={model:{seed:sum(row.get("model_id")==model and row.get("seed")==seed for row in planned_rows) for seed in EXPECTED_SEEDS} for model in EXPECTED_MODELS}
    if any(model_seed_counts[model]!={seed:MODEL_SEED_ROWS for seed in EXPECTED_SEEDS} for model in EXPECTED_MODELS): fail("candidate plan is not exactly 20 rows per model and seed")
    if any(row.get("seed") not in EXPECTED_SEEDS or row.get("split")!="dev" or row.get("method_id")!=EXPECTED_METHOD for row in planned_rows): fail("candidate 180 plan row scope differs")
    if qc_manifest.get("candidate_plan_raw_sha256")!=sha(plan_raw) or qc_manifest.get("planned_rows_raw_sha256")!=sha(rows_raw) or qc_manifest.get("candidate_plan_sha256")!=plan.get("plan_sha256"):
        fail("QC04 manifest does not bind the exact frozen candidate plan/rows")
    expected_artifacts=index.get("requests")
    if not isinstance(expected_artifacts,list) or len(expected_artifacts)!=6: fail("expected exactly six direct downloads")
    if len({item.get("local") for item in expected_artifacts})!=6 or len({item.get("remote") for item in expected_artifacts})!=6: fail("download destinations or sources are not unique")
    downloads={}
    for item in expected_artifacts:
        if item.get("op")!="download": fail("non-download request in failure packet")
        request_raw=read_bounded(item["path"],65536)
        if len(request_raw)!=item.get("bytes") or sha(request_raw)!=item.get("sha256") or item.get("raw_sha256")!=item.get("sha256"): fail("download request pin mismatch")
        request=json.loads(request_raw.decode("utf-8"))
        if request.get("op")!="download" or request.get("remote")!=item.get("remote") or request.get("local")!=item.get("local"): fail("request content differs from pinned transfer envelope")
        response_path=Path(item["path"]+".response.json")
        response,response_raw=read_json(response_path,MAX_META)
        if not response.get("ok") or response.get("local")!=item["local"] or response.get("remote")!=item["remote"] or response.get("size")!=item["expected_remote_bytes"] or response.get("sha256")!=item["expected_remote_sha256"]: fail("download response envelope does not match actual terminal pin: "+item["name"])
        payload=read_bounded(item["local"],max(int(item["expected_remote_bytes"]),1))
        if len(payload)!=item["expected_remote_bytes"] or sha(payload)!=item["expected_remote_sha256"]: fail("downloaded artifact bytes/SHA mismatch: "+item["name"])
        downloads[item["name"]]=payload
    if len(downloads["results_ledger"])>MAX_LEDGER: fail("candidate ledger exceeds bounded direct-read size")
    ledger=parse_jsonl(downloads["results_ledger"],"candidate failure ledger")
    if len(ledger)!=180: fail("candidate failure ledger is not exactly 180 rows")
    if b"".join(canonical(row)+b"\n" for row in ledger)!=downloads["results_ledger"]: fail("candidate failure ledger is not canonical JSONL")
    for planned,result in zip(planned_rows,ledger):
        if any(result.get(field)!=planned.get(field) for field in expected_identity): fail("ordered 12-field row identity mismatch at "+str(planned.get("row_id")))
    if [row.get("row_id") for row in ledger]!=[row.get("row_id") for row in planned_rows]: fail("ordered row IDs differ from frozen plan")
    if any(row.get("schema_version")!="cross_tfg_audio_head_stagea_v1" for row in ledger): fail("ledger row schema mismatch")
    counts={state:sum(row.get("status")==state for row in ledger) for state in ("FAILED","INCOMPLETE","PASS")}
    if counts!={"FAILED":180,"INCOMPLETE":0,"PASS":0}: fail("failure ledger status counts are not 180/0/0")
    model_counts={model:{state:sum(row.get("model_id")==model and row.get("status")==state for row in ledger) for state in ("FAILED","INCOMPLETE","PASS")} for model in EXPECTED_MODELS}
    if any(model_counts[m]!={"FAILED":60,"INCOMPLETE":0,"PASS":0} for m in EXPECTED_MODELS): fail("each backend is not exactly 60 failures")
    failure_hist=collections.Counter((row.get("model_id"),row.get("failure_reason")) for row in ledger)
    diagnostic_hist=diag["ledger"].get("failure_reason_histogram",[])
    expected_hist={(row["model_id"],row["failure_reason"]):row["count"] for row in diagnostic_hist}
    if dict(failure_hist)!=expected_hist or len(expected_hist)!=3: fail("full failure histogram differs from actual diagnostic response")
    if set(reason for _,reason in failure_hist)!= {FAILURE_TEXT}: fail("not every failed row has the same configured interop-thread error")
    if any(row.get("failure_reason")!=FAILURE_TEXT for row in ledger): fail("ledger contains a non-interop failure")
    manifest_raw=downloads["result_manifest"]
    manifest=json.loads(manifest_raw.decode("utf-8-sig"))
    if sha(manifest_raw)!=diag["result_manifest"]["raw_sha256"] or len(manifest_raw)!=69049: fail("manifest downloaded bytes do not match diagnostic actual pin")
    if manifest.get("manifest_sha256")!=payload_sha(manifest,"manifest_sha256"): fail("result manifest payload self hash mismatch")
    if manifest.get("schema_version")!="cross-tfg-audio-eval-v1" or manifest.get("run_id")!=EXPECTED_RUN_ID or manifest.get("status")!="INCOMPLETE": fail("failure result manifest identity/status mismatch")
    if manifest.get("plan_raw_sha256")!=sha(plan_raw) or manifest.get("plan_sha256")!=plan.get("plan_sha256") or manifest.get("results_jsonl_sha256")!=sha(downloads["results_ledger"]): fail("manifest plan/ledger lineage mismatch")
    if manifest.get("planned_rows")!=180 or manifest.get("result_rows")!=180 or manifest.get("status_counts")!={"PASS":0,"FAILED":180,"INCOMPLETE":0}: fail("manifest 180-row status counts mismatch")
    if manifest.get("generation_repeat_qc_jsonl_sha256")!=EMPTY_SHA256 or manifest.get("generation_repeat_qc_rows")!=0 or manifest.get("repeat_qc_status")!="NOT_RUN_SUBRUN": fail("empty repeat ledger contract changed")
    if downloads["generation_repeat_qc_ledger"]!=b"" or sha(downloads["generation_repeat_qc_ledger"])!=EMPTY_SHA256: fail("generation repeat artifact is not the pinned zero-byte file")
    by_method=[]
    for model in EXPECTED_MODELS:
        subset=[row for row in ledger if row.get("model_id")==model and row.get("method_id")==EXPECTED_METHOD]
        by_method.append({"model_id":model,"method_id":EXPECTED_METHOD,"planned_rows":60,"result_rows":len(subset),"status_counts":{"FAILED":sum(r.get("status")=="FAILED" for r in subset),"INCOMPLETE":sum(r.get("status")=="INCOMPLETE" for r in subset),"PASS":sum(r.get("status")=="PASS" for r in subset)}})
    if manifest.get("counts_by_model_method")!=by_method: fail("manifest model/method count table differs from ledger")
    expected_manifest_failures=[{"row_id":r["row_id"],"reason":r["failure_reason"]} for r in ledger]
    if manifest.get("failure_reasons")!=expected_manifest_failures: fail("manifest failure list differs from complete ledger")
    run_raw=downloads["run_status"]; run=json.loads(run_raw.decode("utf-8-sig"))
    if run.get("schema_version")!="cross_tfg_audio_head_stagea_v1" or run.get("status")!="COMPLETE_WITH_FAILURES" or run.get("run_id")!=EXPECTED_RUN_ID: fail("run-status schema/status/run identity mismatch")
    if run.get("completed_rows")!=180 or run.get("planned_rows")!=180 or run.get("status_counts")!=manifest.get("status_counts"): fail("run-status counts mismatch")
    if run.get("plan_path")!=index["frozen_input"]["plan_path_remote"] or run.get("planned_rows_path")!=index["frozen_input"]["planned_rows_path_remote"]: fail("run-status frozen plan/rows path lineage mismatch")
    if run.get("result_manifest_sha256")!=sha(manifest_raw) or run.get("result_manifest_path")!=diag["result_manifest"]["path"]: fail("run status does not bind exact downloaded manifest")
    if run.get("repeat_qc_status")!="NOT_RUN_SUBRUN" or run.get("plan_raw_sha256")!=sha(plan_raw) or run.get("planned_rows_sha256")!=sha(rows_raw): fail("run-status frozen input/repeat lineage mismatch")
    execution_raw=downloads["execution_record"]; execution=json.loads(execution_raw.decode("utf-8-sig"))
    if execution.get("schema")!="cross_tfg_v3_step120_candidate_execution_v1" or execution.get("status")!="INCOMPLETE_STRICT_POLICY_RECEIPTS" or execution.get("child_returncode")!=2: fail("execution terminal failure status/child code mismatch")
    if "no strict bootstrap receipts" not in str(execution.get("strict_policy_error","")): fail("execution record lacks strict-receipt failure reason")
    frozen=index["frozen_input"]
    if execution.get("plan_raw_sha256")!=sha(plan_raw) or execution.get("plan_manifest_raw_sha256")!=sha(manifest14_raw) or execution.get("planned_rows_raw_sha256")!=sha(rows_raw) or execution.get("runtime_binding_raw_sha256")!=sha(runtime_raw): fail("execution input lineage hashes mismatch")
    runtime_binding=json.loads(runtime_raw.decode("utf-8-sig"))
    command=execution.get("command")
    bindings=runtime_binding.get("execution_bindings",{})
    if not isinstance(command,list) or len(command)<4 or command[0]!=bindings.get("stagea_python") or command[1]!="-c": fail("execution command does not use the pinned Stage-A Python strict-bootstrap route")
    guard_tree=ast.parse(guard_raw.decode("utf-8-sig"),filename=str(guard_path))
    expected_gate_code=None
    for node in guard_tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=="strict_startup_gate_code":
            for child in node.body:
                if isinstance(child,ast.Return): expected_gate_code=ast.literal_eval(child.value); break
    if not isinstance(expected_gate_code,str) or command[2]!=expected_gate_code: fail("execution embedded strict-startup gate differs from pinned candidate guard")
    source_refs=plan.get("source_reference_plans",{})
    project_root=frozen["plan_path_remote"].split("/runs/",1)[0]
    expected_dispatcher=project_root+"/"+source_refs["v3_candidate_dispatcher"]["path"].lstrip("/")
    if command[3]!=expected_dispatcher: fail("execution command does not route through the pinned candidate dispatcher")
    inner_args=command[4:]
    if len(inner_args)%2: fail("execution inner dispatcher argv has an unexpected flag shape")
    inner_flags=dict(zip(inner_args[0::2],inner_args[1::2]))
    expected_inner={
      "--stagea-core":project_root+"/"+source_refs["stagea_core"]["path"].lstrip("/"),
      "--cross-frontend-adapter-runner":project_root+"/"+source_refs["cross_frontend_adapter_runner"]["path"].lstrip("/"),
      "--v3-cli":project_root+"/"+source_refs["v3_cli"]["path"].lstrip("/"),
      "--expected-v3-cli-sha256":source_refs["v3_cli"]["raw_sha256"],
      "--plan":frozen["plan_path_remote"],"--planned-rows":frozen["planned_rows_path_remote"],
      "--expected-plan-raw-sha256":sha(plan_raw),"--output-root":diag["identity"]["candidate_root"],"--project-root":project_root,
    }
    if any(inner_flags.get(key)!=value for key,value in expected_inner.items()): fail("execution dispatcher argv differs from actual frozen source/input references")
    if execution.get("gpu_uuid")!=runtime_binding.get("gpu_uuid") or execution.get("gpu_uuid")!=launch_receipt.get("runtime_binding",{}).get("gpu_uuid"): fail("execution GPU binding differs from pinned runtime/launch receipt")
    launch_command=launch_receipt.get("command")
    if not isinstance(launch_command,list) or len(launch_command)<4 or launch_command[0]!=bindings.get("wav2lip_python") or launch_command[1]!="-B": fail("launch receipt command does not use the pinned candidate Python")
    expected_guard=project_root+"/"+plan_execution["samehost_candidate_guarded_launcher"]["path"].lstrip("/")
    if launch_command[2]!=expected_guard or "--execute-candidate" not in launch_command: fail("launch receipt is not for the pinned guarded candidate execution")
    def launch_arg(flag):
        try: return launch_command[launch_command.index(flag)+1]
        except (ValueError,IndexError): return None
    expected_launch={
      "--project-root":project_root,"--candidate-plan":frozen["plan_path_remote"],
      "--candidate-plan-raw-sha256":sha(plan_raw),"--planned-rows":frozen["planned_rows_path_remote"],
      "--planned-rows-raw-sha256":sha(rows_raw),
      "--plan-manifest":plan_execution.get("plan_manifest_path"),
      "--plan-manifest-raw-sha256":sha(manifest14_raw),
      "--runtime-binding":plan_execution["samehost_runtime_binding"]["path"],
      "--runtime-binding-raw-sha256":sha(runtime_raw),
    }
    # plan_manifest_path is bound by the actual launch receipt if the frozen plan omits a top-level path.
    expected_launch["--plan-manifest"]=launch_receipt.get("candidate_plan_manifest",{}).get("path")
    if any(launch_arg(flag)!=value for flag,value in expected_launch.items()): fail("launch command argv differs from actual plan/manifest/rows/runtime pins")
    startup_raw=downloads["startup_policy_receipt"]; startup=json.loads(startup_raw.decode("utf-8-sig"))
    startup_ref=execution.get("startup_policy_gate_receipt")
    if not isinstance(startup_ref,dict) or startup_ref.get("bytes")!=len(startup_raw) or startup_ref.get("raw_sha256")!=sha(startup_raw) or startup_ref.get("path")!=diag["startup_policy"]["path"]: fail("execution record does not pin exact startup receipt")
    if startup.get("schema")!="cross_tfg_step120_candidate_startup_policy_gate_v1" or startup.get("status")!="PASS" or startup.get("pid")!=startup_ref.get("pid"): fail("startup receipt schema/status/process identity mismatch")
    if startup!=diag.get("startup_policy",{}).get("body"): fail("startup receipt differs from the complete actual diagnostic body")
    strict_bootstrap_pin=pins[refs["strict_bootstrap_key"]]
    if startup.get("bootstrap_source_sha256")!=strict_bootstrap_pin["sha256"] or startup.get("receipt_sha256")!=payload_sha(startup,"receipt_sha256"): fail("startup bootstrap or self-hash mismatch")
    if startup_ref.get("receipt_sha256")!=startup.get("receipt_sha256"): fail("execution startup receipt hash mismatch")
    if execution.get("confirm_audio_read") is not False: fail("execution does not explicitly preserve closed confirmation scope")
    if "strict_determinism_attestation" in execution: fail("unexpected strict attestation reference; preserve actual absence")
    attestation=index.get("strict_attestation",{})
    if attestation.get("status")!="ABSENT_PER_FAILURE_DIAGNOSTIC_NO_DOWNLOAD_REQUEST" or attestation.get("download_request_created") is not False: fail("strict attestation absence contract was not preserved")
    if len(downloads["results_ledger"])!=817337 or sha(downloads["results_ledger"])!=diag["ledger"]["raw_sha256"]: fail("ledger actual diagnostic pin mismatch")
    if diag["ledger"].get("parsed_rows")!=180 or diag["ledger"].get("ordered_identity_fields_match") is not True or diag["ledger"].get("ordered_row_ids_match") is not True: fail("actual failure diagnostic ledger summary differs")
    report={
      "schema":"cross_tfg_candidate180_failure_readback_qc_attempt04_v1",
      "status":"FAILURE_TERMINAL_READBACK_QC04_PASS_NO_SCIENTIFIC_ADMISSION",
      "run_id":EXPECTED_RUN_ID,
      "scope":{"local_read_only":True,"remote_dispatch_performed":False,"gpu_or_cuda_queried":False,"models_or_scorers_imported":False,"api_called":False,"media_read_or_written":False,"scientific_admission":False},
      "terminal":{"planned_rows":180,"ledger_rows":len(ledger),"ordered_identity_field_count":len(expected_identity),"ordered_identity_fields":list(expected_identity),"all_identity_rows_match_plan":True,"seed_roster":list(EXPECTED_SEEDS),"seed_counts":{str(seed):seed_counts[seed] for seed in EXPECTED_SEEDS},"model_seed_counts":{model:{str(seed):model_seed_counts[model][seed] for seed in EXPECTED_SEEDS} for model in EXPECTED_MODELS},"model_status_counts":model_counts,"full_failure_reason_histogram":[{"model_id":m,"failure_reason":reason,"count":count} for (m,reason),count in sorted(failure_hist.items())],"manifest_status":manifest["status"],"run_status":run["status"],"execution_status":execution["status"],"child_returncode":execution["child_returncode"],"strict_policy_error":execution["strict_policy_error"],"repeat_qc_status":manifest["repeat_qc_status"],"repeat_qc_rows":manifest["generation_repeat_qc_rows"],"repeat_qc_bytes":0,"strict_attestation":{"status":"ABSENT_AS_RECORDED","downloaded":False}},
      "actual_artifact_pins":[{"name":item["name"],"bytes":len(downloads[item["name"]]),"sha256":sha(downloads[item["name"]]),"remote":item["remote"],"local":item["local"]} for item in expected_artifacts],
      "source_pins_verified":sorted(pins),
      "qc_manifest":{"path":pins[refs["qc_manifest_key"]]["path"],"bytes":pins[refs["qc_manifest_key"]]["bytes"],"sha256":pins[refs["qc_manifest_key"]]["sha256"],"manifest_sha256":qc_manifest["manifest_sha256"]},
      "parent_packet_index":{"path":pins[refs["parent_packet_index_key"]]["path"],"bytes":len(parent_raw),"sha256":sha(parent_raw),"index_sha256":parent["index_sha256"]},
      "packet_index":{"path":str(index_path),"bytes":len(index_raw),"sha256":sha(index_raw),"index_sha256":saved_index_sha},
    }
    out=Path(args.report)
    if out.exists() or out.is_symlink(): fail("QC report destination already exists; use a fresh path")
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as stream: stream.write((json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode("utf-8"))
    print(json.dumps({"status":report["status"],"report_path":str(out.resolve()),"ledger_rows":len(ledger),"scientific_admission":False},sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
