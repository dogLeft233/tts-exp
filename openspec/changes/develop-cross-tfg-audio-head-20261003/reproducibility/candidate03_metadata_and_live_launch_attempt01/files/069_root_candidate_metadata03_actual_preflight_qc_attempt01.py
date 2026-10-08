from pathlib import Path
import json, hashlib, copy
R=Path(__file__).resolve().parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
ip=R/'.codex/scratch/candidate_v3_cli03_metadata_preflight_attempt03/request_index_attempt03.json'
i=json.loads(ip.read_bytes());assert sha(ip.read_bytes())=='b52aa7f269ef0e072852d310bacadbe9240cb3e6627c8e3fa2e9f37253488ba6'
assert i['index_sha256']==sha(canon({k:v for k,v in i.items() if k!='index_sha256'}))
responses={}
for ref in i['source_evidence'].values():
 b=Path(ref['path']).read_bytes();assert len(b)==ref['bytes'] and sha(b)==ref['sha256']
for ref in i['requests']:
 p=Path(ref['path']);b=p.read_bytes();assert len(b)==ref['bytes'] and sha(b)==ref['sha256']
 q=json.loads(b);rp=Path(str(p)+'.response.json');rr=rp.read_bytes();r=json.loads(rr);assert r['ok'] is True
 if q['op']=='exec':
  assert r['code']==0 and r['stderr']=='' and r['timed_out'] is False
  assert r['truncated']=={'stdout':False,'stderr':False} and r['incomplete_stream'] is False and r['stream_error'] is None
  s=json.loads(r['stdout']);assert s['status'].startswith('PASS')
 else:
  assert r['remote']==q['remote'] and r['local']==q['local']
  raw=Path(r['local']).read_bytes();assert len(raw)==r['size'] and sha(raw)==r['sha256']
  if q['op']=='upload': assert r['size']==q['expected_bytes'] and r['sha256']==q['expected_sha256']
 responses[ref['name']]={'bytes':len(rr),'sha256':sha(rr),'op':q['op']}
p=Path(i['metadata_preflight']['local_path']);b=p.read_bytes();v=json.loads(b)
assert len(b)==4776 and sha(b)=='3c51e9c3b7f4e1d4ff7e0d92d44c2e702fe70393210448a5d277c38415bafea5'
assert v['status']=='PASS_METADATA_ONLY' and v['candidate_rows']==180
assert v['execution_performed'] is False and v['confirm_audio_read'] is False
assert v['plan_raw_sha256']==i['candidate_plan']['raw_sha256']
assert v['runtime_binding_raw_sha256']==i['runtime_binding09']['raw_sha256']
assert v['candidate_repair_binding_raw_sha256']==i['candidate_repair_binding']['raw_sha256']
rt=json.loads(Path(i['source_evidence']['samehost_runtime_binding09']['path']).read_bytes())
overlay=copy.deepcopy(rt);overlay['execution_bindings']['output_root']=i['candidate_output']['namespace']
o=v['runtime_validation_overlay'];assert o['changed_fields']==['execution_bindings.output_root']
assert o['overlay_execution_bindings']==overlay['execution_bindings']
assert o['overlay_canonical_sha256']==sha(canon(overlay))
assert o['overlay_is_persisted_as_runtime_binding'] is False
live=v['live_resource_probe'];assert live['capacity_admission']['stage']=='candidate180'
assert live['output_disk']['available_bytes']>=live['capacity_admission']['required_free_bytes']
assert live['gpus'][0]['uuid']==v['gpu_uuid']==rt['gpu_uuid']
assert live['gpus'][0]['memory_free_mib']>=live['thresholds']['min_gpu_free_mib']
assert live['ram']['available_bytes']>=live['thresholds']['min_ram_available_bytes']
result={'status':'PASS_ROOT_ACTUAL_CANDIDATE_METADATA_PREFLIGHT',
 'metadata':{'bytes':len(b),'sha256':sha(b),'path':str(p.relative_to(R))},
 'requests_completed':len(responses),'source_pins_verified':len(i['source_evidence']),
 'candidate180_execution_started':False,'original_runtime_binding_changed':False,
 'overlay_only_output_root':True,'confirm_audio_read':False,
 'live_resource_capacity_pass':True,'responses':responses,
 'source_sha256':sha(Path(__file__).read_bytes())}
out=R/'.codex/scratch/root_candidate_metadata03_actual_preflight_qc_attempt01.json'
if out.exists():raise FileExistsError(out)
out.write_bytes((json.dumps(result,sort_keys=True,indent=2)+'\n').encode());print(json.dumps(result))
