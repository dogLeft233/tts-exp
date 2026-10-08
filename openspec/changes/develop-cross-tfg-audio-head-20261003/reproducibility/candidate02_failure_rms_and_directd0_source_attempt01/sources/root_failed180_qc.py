import collections, hashlib, json
from pathlib import Path

ROOT=Path('E:/Documents/tts-exp')
BASE=ROOT/'.codex/scratch/dev360_candidate180_failure_readback_attempt03'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def canon(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
index=json.loads((BASE/'request_index_attempt03.json').read_bytes())
assert index['index_sha256']==sha(canon({k:v for k,v in index.items() if k!='index_sha256'}))
for pin in index['source_evidence'].values():
    raw=Path(pin['path']).read_bytes()
    assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256']
files={}
for pin in index['requests']:
    request_raw=Path(pin['path']).read_bytes(); req=json.loads(request_raw)
    assert len(request_raw)==pin['bytes'] and sha(request_raw)==pin['sha256'] and req['op']=='download'
    response=json.loads(Path(pin['path']+'.response.json').read_bytes())
    raw=Path(pin['local']).read_bytes()
    assert response['ok'] and response['remote']==pin['remote'] and response['local']==pin['local']
    assert len(raw)==response['size']==pin['expected_remote_bytes'] and sha(raw)==response['sha256']==pin['expected_remote_sha256']
    files[pin['name']]=raw
rows=[json.loads(line) for line in files['results_ledger'].splitlines()]
assert len(rows)==180 and b''.join(canon(row)+b'\n' for row in rows)==files['results_ledger']
plan=json.loads(Path(index['source_evidence']['candidate_plan13']['path']).read_bytes())
planned=[json.loads(line) for line in Path(index['source_evidence']['candidate_rows12']['path']).read_bytes().splitlines()]
assert plan['rows']==planned and len(planned)==180
identities=('row_id','split','sample_id','speaker_id','model_id','audio_frontend','model_role','face_id','seed','method_id','method_config_sha256','checkpoint_sha256')
assert all(all(actual[k]==expected[k] for k in identities) for actual,expected in zip(rows,planned))
assert len({row['row_id'] for row in rows})==180
assert len({row['sample_id'] for row in rows})==20 and len({row['speaker_id'] for row in rows})==10
assert collections.Counter(row['seed'] for row in rows)=={20261003:60,20261004:60,20261005:60}
models=collections.Counter(row['model_id'] for row in rows)
assert models=={'wav2lip_gan_bac9a81e':60,'musetalk_fixed_portrait_attempt16':60,'ditto_hubert_fixed_portrait_attempt18':60}
reason='V3CLIError: cannot establish configured PyTorch interop threads: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called'
assert all(row['status']=='FAILED' and row['failure_reason']==reason for row in rows)
assert all(row['render']['status']=='NOT_STARTED' and row['mux_audio_readback']['status']=='NOT_STARTED' and row['score']['status']=='NOT_RUN' for row in rows)
manifest=json.loads(files['result_manifest']); status=json.loads(files['run_status']); execution=json.loads(files['execution_record']); startup=json.loads(files['startup_policy_receipt'])
for obj,key in ((manifest,'manifest_sha256'),(startup,'receipt_sha256')):
    assert obj[key]==sha(canon({k:v for k,v in obj.items() if k!=key}))
assert manifest['status']=='INCOMPLETE' and manifest['result_rows']==manifest['planned_rows']==180
assert manifest['status_counts']==status['status_counts']=={'FAILED':180,'INCOMPLETE':0,'PASS':0}
assert manifest['results_jsonl_sha256']==sha(files['results_ledger']) and status['result_manifest_sha256']==sha(files['result_manifest'])
assert status['status']=='COMPLETE_WITH_FAILURES' and status['completed_rows']==status['planned_rows']==180
assert manifest['failure_reasons']==[{'row_id':row['row_id'],'reason':reason} for row in rows]
assert execution['status']=='INCOMPLETE_STRICT_POLICY_RECEIPTS' and execution['child_returncode']==2
assert execution['startup_policy_gate_receipt']['raw_sha256']==sha(files['startup_policy_receipt']) and execution['startup_policy_gate_receipt']['receipt_sha256']==startup['receipt_sha256']
assert startup['status']=='PASS' and startup['policy']['torch_num_interop_threads']==startup['policy']['torch_num_threads']==1
assert startup['python_executable']=='/usr/bin/python3.10' and 'strict_determinism_attestation' not in execution
assert files['generation_repeat_qc_ledger']==b'' and manifest['repeat_qc_status']=='NOT_RUN_SUBRUN'
assert manifest['plan_raw_sha256']==execution['plan_raw_sha256']==index['frozen_input']['plan_raw_sha256']
report={'status':'PASS_ROOT_FULL_FAILED180_READBACK','science_admission':False,'all180_failed_before_render':True,'identity_fields_checked':list(identities),'rows':180,'speakers':10,'samples':20,'model_counts':dict(models),'seed_counts':dict(collections.Counter(row['seed'] for row in rows)),'failure_reason':reason,'strict_native_attestation':'ABSENT','producer_child_returncode':2,'duration_seconds':execution['finished_unix']-execution['started_unix'],'artifact_pins':[{k:pin[k] for k in ('name','remote','local','expected_remote_bytes','expected_remote_sha256')} for pin in index['requests']],'remote_gpu_api_or_replay':False}
out=ROOT/'.codex/scratch/root_failed_candidate180_full_qc_attempt01.json'
with out.open('xb') as stream: stream.write(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2).encode()+b'\n')
print(json.dumps({'status':report['status'],'rows':180,'identity_fields':12,'duration_seconds':report['duration_seconds'],'report_sha256':sha(out.read_bytes())}))
