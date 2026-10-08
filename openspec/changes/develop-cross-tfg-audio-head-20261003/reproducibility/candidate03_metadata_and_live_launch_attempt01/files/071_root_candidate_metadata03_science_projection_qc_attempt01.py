from pathlib import Path
import hashlib, json, copy, collections
R = Path(__file__).resolve().parents[2]
P = R / '.codex/scratch/candidate_v3_cli03_runtime_repair_attempt03/metadata_output_attempt03'
sha = lambda b: hashlib.sha256(b).hexdigest()
canon = lambda v: json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def load(p): return json.loads(p.read_bytes())
oldp=R/'.codex/scratch/dev360_candidate_generator_pin_repair_attempt02/candidate_rebound_plan_attempt13.json'
newp=P/'candidate_plan_attempt14.json'
old, new=load(oldp),load(newp)
assert sha(oldp.read_bytes())=='8615754443e653fc79203f108fa689205a77e7f6f234930b352204df08b98280'
assert sha(newp.read_bytes())=='1a6c8ae99c4b8a1141c6f65748d426ba88034c14badf5ecec497ffc11d62bc60'
binding=load(P/'candidate_runtime_repair_binding_attempt10.json')
manifest=load(P/'candidate_plan_manifest_attempt15.json')
expected={
 'candidate_plan_attempt14.json':('1a6c8ae99c4b8a1141c6f65748d426ba88034c14badf5ecec497ffc11d62bc60','plan_sha256'),
 'candidate_plan_manifest_attempt15.json':('cd0aa3e7bca14a7676dab7f82d621b93e1a173e586f04efcdbaf3794b022808c','manifest_sha256'),
 'candidate_runtime_repair_binding_attempt10.json':('4d50fcb6e6fd7a6fcd6a4a59515f7e92c9cbb43250e22ac97994d5558bb84ba8','repair_binding_sha256')}
for name,(h,f) in expected.items():
 raw=(P/name).read_bytes();v=json.loads(raw); assert sha(raw)==h
 assert v[f]==sha(canon({k:x for k,x in v.items() if k!=f}))
oldrows,newrows=copy.deepcopy(old['rows']),copy.deepcopy(new['rows'])
for rows in (oldrows,newrows):
 for row in rows:
  row.pop('method_config_sha256')
  row['generation_settings'].pop('v3_cli_config_sha256')
assert oldrows==newrows
assert sha(canon(newrows))==binding['candidate_rows_context_sha256']
oldmethods,newmethods=copy.deepcopy(old['methods']),copy.deepcopy(new['methods'])
for methods in (oldmethods,newmethods):
 for m in methods:
  m.pop('config_sha256');m['config'].pop('v3_cli_config_path');m['config'].pop('v3_cli_config_sha256')
assert oldmethods==newmethods
assert sha(canon(newmethods))==binding['candidate_method_context_sha256']
rowsraw=(P/'candidate_planned_rows_attempt14.jsonl').read_bytes()
assert sha(rowsraw)=='d781268d02c7495fd35c3bf314848863d57673eeef044a0c5edb19f9e8797f48'
assert [json.loads(x) for x in rowsraw.splitlines()]==new['rows']
assert len(new['rows'])==180 and len({r['row_id'] for r in new['rows']})==180
assert len({r['sample_id'] for r in new['rows']})==20
assert len({r['speaker_id'] for r in new['rows']})==10
assert collections.Counter(r['model_id'] for r in new['rows'])=={
 'wav2lip_gan_bac9a81e':60,'musetalk_fixed_portrait_attempt16':60,'ditto_hubert_fixed_portrait_attempt18':60}
assert collections.Counter(r['seed'] for r in new['rows'])=={20261003:60,20261004:60,20261005:60}
assert all(r['split']=='dev' and r['score_audio']==r['natural_pcm'] for r in new['rows'])
runtimep=R/'.codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/runtime_binding_attempt09.json'
assert sha(runtimep.read_bytes())==binding['runtime_binding_raw_sha256']==manifest['samehost_runtime_binding_raw_sha256']=='dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5'
cpup=R/'.codex/scratch/candidate_v3_cpu_preflight_attempt03/downloads_attempt02_revision02/cpu_preflight_receipt_attempt03.json'
assert sha(cpup.read_bytes())==binding['cpu_preflight_receipt']['raw_sha256']=='8c44b1c80a08e48465f25672ef6b990002653ff7d250ede7369be13dab3a3b09'
assert binding['cpu_preflight_receipt']['path']==binding['cpu_preflight_transport_evidence']['remote_receipt_path']=='/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/candidate_v3_cpu_preflight_attempt03/output/cpu_preflight_receipt_attempt03.json'
for ref in ('execution_request','execution_response','download_request','download_response'):
 r=binding['cpu_preflight_transport_evidence'][ref]; p=R/r['path'];raw=p.read_bytes()
 assert len(raw)==r['bytes'] and sha(raw)==r['raw_sha256']
assert binding['scope']['candidate_gpu_qualification_performed'] is False
assert binding['scope']['confirm_audio_read'] is False
assert manifest['candidate_runtime_qualification_performed'] is False
ns=Path(binding['candidate_output_namespace']);controls=Path(binding['controls_parent_output_leaf'])
assert ns!=controls.parent and ns.parent==controls.parent.parent
q={'status':'PASS_ROOT_FULL_METADATA03_SCIENTIFIC_ROW_AND_METHOD_PROJECTION',
 'candidate_rows':180,'samples':20,'speakers':10,'models':3,'seeds':3,
 'ordered_scientific_rows_exact_predecessor':True,'scientific_method_exact_predecessor':True,
 'original_runtime09_unchanged':True,'actual_CPU_receipt_and_transport_pins_verified':True,
 'namespace_separate_from_controls':True,'candidate_GPU_run_performed':False,
 'source':{'path':str(Path(__file__).relative_to(R)),'sha256':sha(Path(__file__).read_bytes())},
 'inputs':{name:{'bytes':(P/name).stat().st_size,'sha256':sha((P/name).read_bytes())} for name in expected}}
out=R/'.codex/scratch/root_candidate_metadata03_science_projection_qc_attempt01.json'
if out.exists(): raise FileExistsError(out)
out.write_bytes((json.dumps(q,sort_keys=True,indent=2)+'\n').encode());print(json.dumps(q))
