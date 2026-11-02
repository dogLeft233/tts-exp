"""Wait for real final coarse states, freeze score pins, qualify, then new 27-grid."""
import hashlib,json,os,shlex,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import other_identities_trial_v1 as trial
from execution_journal import atomic_json,exclusive_lock
BASE=trial.BASE;ROOT=trial.REMOTE;VERSION='patrick_ave_rms_coarse_v1'
OUT=BASE/'trials'/VERSION;RUN=ROOT+'/trials/'+VERSION
SSH=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2']
trial.OUT=OUT;trial.RUN=RUN;trial.SSH=SSH
def remote(code):return subprocess.check_output(SSH+['python3 -'],input=code.encode())
def upload(path,raw):
 code='from pathlib import Path;import sys,hashlib;p=Path(%r);p.parent.mkdir(parents=True,exist_ok=True);b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==%r;assert not p.exists() or p.read_bytes()==b;\np.write_bytes(b) if not p.exists() else None'%(path,hashlib.sha256(raw).hexdigest())
 subprocess.run(SSH+['python3 -c '+shlex.quote(code)],input=raw,check=True)
def run():
 with exclusive_lock(OUT/'score_handoff.lock'):
  assert not (OUT/'score_handoff_receipt.json').exists()
  state=dict(pid=os.getpid(),status='waiting_for_actual_coarse10',score_jobs_started=0)
  def save():atomic_json(OUT/'score_handoff_receipt.json',state)
  save()
  while not (OUT/'coarse_ready_for_27_grid.json').exists():
   assert not (OUT/'controller_failure.json').exists(),'training controller failed; no scoring'
   if (OUT/'receipt.json').exists():
    s=json.loads((OUT/'receipt.json').read_text());assert s['status'] not in ['terminal_failure_preserved_no_new_jobs','paused_at_verified_boundary'],'training stopped; no scoring'
   time.sleep(30)
  ready_raw=(OUT/'coarse_ready_for_27_grid.json').read_bytes();ready=json.loads(ready_raw)
  assert ready['passed'] and ready['identity']=='PatrickLeahy' and ready['version']==VERSION and ready['no_training_job_active']
  assert ready['contract_sha256']==trial.sha(OUT/'contract.json')
  old=BASE/'trials/fp16_other_identities_trial_v1/early_score_preview_v2'
  original=json.loads((old/'contract.json').read_text());host=json.loads((old/'host_source_manifest.json').read_text())
  score=OUT/'score_27_v1';score.mkdir(exist_ok=False);sources=score/'sources';sources.mkdir()
  names=set(host)-{'coarse_preview_controller_v1.py'}
  names.update(['coarse_preview_rms_controller_v1.py','coarse_preview_rms_v1.py','patrick_rms_score_receipts_v1.py','patrick_rms_runtime_v1.py'])
  pins={}
  for n in sorted(names):
   raw=(HERE/n).read_bytes();(sources/n).write_bytes(raw);pins[n]=hashlib.sha256(raw).hexdigest();upload(RUN+'/score_27_v1/sources/'+n,raw)
  training=json.loads((OUT/'contract.json').read_text())
  assert pins['patrick_rms_runtime_v1.py']==training['sources']['patrick_rms_runtime_v1.py']
  deps=original['dependencies']
  for n,s in deps.items():
   if not n.startswith('official/'):
    raw=(HERE/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==s
    upload(ROOT+'/'+n,raw)
  for n in ['cloud_archive_readback_stream.py']:
   upload(ROOT+'/'+n,(HERE/n).read_bytes())
  result=json.loads(remote('''from pathlib import Path
import json,hashlib
r=Path(%r);deps=%r
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==s for n,s in deps.items())
print(json.dumps(dict(passed=True)))
'''%(ROOT,deps)));assert result['passed']
  cps={}
  for gain in [-6,0,6]:
   r=ready['final_receipts'][str(gain)];assert r['phase']=='coarse' and r['epoch']==10 and r['global_step']==60000
   assert r['normalization_contract_sha256']==ready['contract_sha256']
   rel=Path(r['checkpoint_path']).relative_to(Path(RUN));assert trial.sha(OUT/rel)==r['checkpoint_sha256']
   cps[str(gain)]=dict(sha256=r['checkpoint_sha256'],successful_updates=r['cumulative_successful_updates'])
  contract=dict(original,version='patrick_rms_score_27_v1',sources=pins,dependencies=deps,checkpoints=cps,
   training_control_status=ready['phase']['status'],training_qualification_passed=ready['phase']['passed'],
   request_sha256=hashlib.sha256(ready_raw).hexdigest(),normalization=training['normalization'],training_contract_sha256=ready['contract_sha256'],original_versions_unchanged=True)
  contract_raw=json.dumps(contract,indent=2).encode();(score/'contract.json').write_bytes(contract_raw);upload(RUN+'/score_27_v1/contract.json',contract_raw)
  (score/'controller_preview_ready.json').write_bytes(ready_raw);upload(RUN+'/score_27_v1/controller_preview_ready.json',ready_raw)
  atomic_json(score/'host_source_manifest.json',pins)
  state.update(status='actual_final_checkpoints_bound_render_qualification',score_contract_sha256=trial.sha(score/'contract.json'));save()
  path=RUN+'/score_27_v1/jobs/qualify_+0_+0/attempt01'
  cmd=['flock','-n',ROOT+'/preparation_GPU.lock','bash',ROOT+'/runtime_exec.sh',RUN+'/score_27_v1/sources/coarse_preview_rms_v1.py','qualify','--gain','0','--driver','0']
  code='import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import start_job;print(json.dumps(start_job(Path(%r),%r)))'%(ROOT,path,cmd)
  result=json.loads(remote(code));state['score_jobs_started']=1;save()
  while result['state'] in ['starting','running']:
   time.sleep(20);result=json.loads(remote('import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import status;print(json.dumps(status(Path(%r))))'%(ROOT,path)))
  trial.persist('score_27_v1/jobs/qualify_+0_+0/attempt01');assert result['state']=='completed' and result['passed'],result
  log=(score/'controller_stdout.log').open('xb');err=(score/'controller_stderr.log').open('xb')
  process=subprocess.Popen([sys.executable,str(sources/'coarse_preview_rms_controller_v1.py')],stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,cwd=Path.cwd())
  state.update(status='new27_controller_started',controller_pid=process.pid);save()
  assert process.wait()==0,'27 grid failed; keep source/failure artifacts, no blind restart'
  assert json.loads((score/'all27_preview.json').read_text())['passed']
  state.update(status='all27_ready_for_final_independent_delivery_review',passed=True);save()
if __name__=='__main__':
 try:run()
 except BaseException:atomic_json(OUT/'score_handoff_failure.json',dict(pid=os.getpid(),traceback=traceback.format_exc(),no_automatic_replay=True));raise
