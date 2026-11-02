"""Host-only handoff: remote execution identity/completion is authoritative, never SSH lifetime."""
from pathlib import Path
import json,subprocess,time,os,traceback
from protocol import OUT,REMOTE,sha,write
from controller import SSH,LOCK,copy_tree,remote_json
REV=OUT/'controller_revision02_remote_completion_handoff_v1'
def plan_handoff(arms,closed,current):
 ids=[a['id'] for a in arms];assert [r['arm'] for r in closed]==ids[:len(closed)]
 assert all(r['all_f_members_and_actual_fullstate_passed'] for r in closed)
 assert len(closed)<len(ids) and current==ids[len(closed)]
 return len(closed)
def remote_is_complete(execution):
 if execution['status']=='running':return False
 assert execution['status']=='completed' and execution['rc']==0,('real_remote_job_failure',execution)
 return True
def probe(job):
 code=f'''from pathlib import Path
import json,time
r=Path({REMOTE!r});p=r/'jobs'/{job!r}/'execution.json'
if not p.exists():print('null');raise SystemExit
for i in range(5):
 try:e=json.loads(p.read_text());break
 except json.JSONDecodeError:time.sleep(.05)
else:raise RuntimeError('execution marker persistently invalid')
q=Path('/proc')/str(e['pid']);same=q.exists() and (q/'stat').read_text().split()[21]==e['proc_start_ticks']
child=Path('/proc')/str(e.get('child_pid',0));child_live=child.exists()
print(json.dumps(dict(execution=e,wrapper_identity_live=same,child_live=child_live)))
'''
 return remote_json(code)
def persist_checkpoint(arm,seen):
 item=remote_json(f"from pathlib import Path\nimport json\np=Path({(REMOTE+'/training/'+arm+'/latest_checkpoint.json')!r});print(p.read_text() if p.exists() else 'null')")
 if not item or item['step'] in seen:return
 dest=OUT/'training'/arm/Path(item['path']).name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert sha(dest)==item['sha256']
 else:
  temp=dest.with_suffix('.revision02.partial');assert not temp.exists()
  with temp.open('xb') as f:r=subprocess.run(SSH+['cat '+item['path']],stdout=f,stderr=subprocess.PIPE)
  assert r.returncode==0 and sha(temp)==item['sha256'];temp.replace(dest)
 seen.add(item['step']);write(dest.parent/'latest_f_checkpoint_revision02.json',dict(**item,f_path=str(dest),fresh_sha256_verified=True))
def observe_job(job,arm=None,proc=None,expected_identity=None,job_dest=None):
 seen=set();identity=expected_identity;lost=0;started=time.monotonic();transport_recorded=False
 pins=json.loads((OUT/'execution_source_pins.json').read_text())
 while True:
  data=probe(job)
  if data is None:
   assert time.monotonic()-started<60,'job launch marker absent; no automatic retry'
   if proc is not None:assert proc.poll() is None,('SSH exited before actual launch marker',proc.returncode)
   time.sleep(2);continue
  e=data['execution'];now=(e['pid'],e['proc_start_ticks'],e.get('child_pid'))
  if identity is None:identity=now
  else:assert tuple(identity)==now,'remote job identity changed'
  for name in ['train.py','runtime.py']:assert e['source_sha256'][name]==pins['source_sha256'][name]
  assert e['source_sha256']['protocol.json']==sha(OUT/'protocol.json') and e['source_sha256']['run_env.sh']==pins['run_env_sha256']
  if job.startswith('evaluate_'):assert e['source_sha256']['evaluate.py']==pins['source_sha256']['evaluate.py']
  if proc is not None and proc.poll() is not None and not transport_recorded:
   stdout,stderr=proc.communicate();folder=REV/'transport_observations';folder.mkdir(exist_ok=True)
   (folder/(job+'.stdout')).write_bytes(stdout);(folder/(job+'.stderr')).write_bytes(stderr)
   write(folder/(job+'.json'),dict(ssh_returncode=proc.returncode,remote_status=e['status'],remote_identity=now,ssh_exit_is_not_scientific_completion=True));transport_recorded=True
  write(REV/'receipt.json',dict(status='observing_remote_execution',host_pid=os.getpid(),job=job,arm=arm,remote_execution=e,remote_identity_live=data['wrapper_identity_live'],ssh_client_exited=proc.poll() is not None if proc else None))
  if e['status']!='running':break
  lost=0 if data['wrapper_identity_live'] and data['child_live'] else lost+1
  assert lost<3,'actual remote job lost across three observations; stop new jobs'
  if arm:persist_checkpoint(arm,seen)
  time.sleep(10)
  # Only observations are repeated. This function never launches or restarts a job.
 complete=remote_is_complete(e)
 dest=job_dest or OUT/'jobs'/job;copy_tree('jobs/'+job,dest)
 assert sha(dest/'stdout.log')==e['stdout_sha256']
 if arm:copy_tree('training/'+arm,OUT/'training'/arm)
 if proc is not None and proc.poll() is None:
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:
   # Scientific completion and immutable readback are already proven; only retire this host transport.
   proc.terminate();proc.wait(timeout=10)
 return dict(job=job,rc=0,elapsed_seconds=e['elapsed_seconds'],job_f_path=str(dest),remote_execution_source='immutable execution.json',actual_identity=identity)
def run_new_job(job,script,args,arm=None,evaluation=False):
 assert probe(job) is None,'never launch an existing command identity'
 wrapper='evaluation_wrapper_v1.py' if evaluation else 'job_wrapper.py'
 cmd=f'flock -n {LOCK} python3 {REMOTE}/{wrapper} {job} {REMOTE}/{script} '+' '.join(args)
 proc=subprocess.Popen(SSH+[cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 write(REV/(job+'_dispatch.json'),dict(command=cmd,ssh_pid=proc.pid,host_pid=os.getpid(),new_job_dispatches=1))
 return observe_job(job,arm=arm,proc=proc)
def main():
 audit=json.loads((REV/'audit.json').read_text());assert audit['passed'];p=json.loads((OUT/'protocol.json').read_text());arms=p['arms']
 current=audit['current_existing_job']['execution'];key=arms[5]['id'];assert plan_handoff(arms,audit['completed_prefix'],key)==5
 assert not (REV/'launcher.json').exists() and not (OUT/'training_complete.json').exists();write(REV/'launcher.json',dict(host_pid=os.getpid(),source_sha256=sha(__file__),audit_sha256=sha(REV/'audit.json'),existing_job_attached_not_launched='train_'+key))
 write(OUT/'active_controller.json',dict(authority='controller_revision02_remote_completion_handoff_v1/receipt.json',failure='controller_revision02_remote_completion_handoff_v1/controller_failure.json',host_pid=os.getpid(),historical_root_failure='training_failure.json'))
 done=json.loads((REV/'original_training_failure.json').read_text())['completed_jobs'];locations={}
 try:
  dest=REV/'completed_job_readback'/('train_'+key)
  result=observe_job('train_'+key,arm=key,expected_identity=(current['pid'],current['proc_start_ticks'],current['child_pid']),job_dest=dest);done.append(result);locations['train_'+key]=str(dest)
  write(OUT/'completed_job_locations.json',locations)
  for arm in arms[6:]:done.append(run_new_job('train_'+arm['id'],'train.py',['train',arm['id']],arm=arm['id']))
  write(OUT/'training_complete.json',dict(passed=True,mode='training',jobs=done,controller_revision=2,coordinator_handoff_without_scientific_replay=True,completed_unix=time.time()))
  write(REV/'receipt.json',dict(status='training_complete',host_pid=os.getpid(),jobs=done))
 except BaseException as e:
  write(REV/'controller_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_jobs=done,new_jobs_stopped=True));raise
if __name__=='__main__':main()
