"""Single Patrick RMS-AVE trial: frozen sources, no replay, exact epoch persistence."""
import json,os,shlex,sys,time,traceback,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import other_identities_trial_v1 as t
from execution_journal import atomic_json,exclusive_lock
VERSION='patrick_ave_rms_coarse_repeat_v1'
t.VERSION=VERSION;t.OUT=t.BASE/'trials'/VERSION;t.RUN=t.REMOTE+'/trials/'+VERSION
t.SSH=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2']
def launch(gain,epoch,prior,reference):
 rel=f'training/PatrickLeahy/seed42/gain_{gain:+d}/coarse/ep{epoch:04d}/attempt01';job=rel.replace('training/','jobs/',1)
 command=['flock','-n',t.REMOTE+'/preparation_GPU.lock','bash',t.REMOTE+'/runtime_exec.sh',t.RUN+'/sources/train_patrick_rms_repeat_epoch_v1.py','--identity','PatrickLeahy','--seed','42','--gain',str(gain),'--phase','coarse','--epoch',str(epoch),'--attempt','1']
 if prior:command+=['--checkpoint',prior['checkpoint_path'],'--resume',prior['_receipt']]
 if reference:command+=['--reference-trace',reference]
 code='import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import start_job;print(json.dumps(start_job(Path(%r),%r)))'%(t.REMOTE,t.RUN+'/'+job,command)
 state=json.loads(t.remote_python(code));return rel,job,state
def status(job):
 return json.loads(t.remote_python('import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import status;print(json.dumps(status(Path(%r))))'%(t.REMOTE,t.RUN+'/'+job)))
def run():
 contract=json.loads((t.OUT/'contract.json').read_text());digest=t.sha(t.OUT/'contract.json')
 for n,s in contract['sources'].items():assert t.sha(t.OUT/'sources'/n)==s
 q=json.loads((t.OUT/'qualification/result.json').read_text());assert q['passed'] and q['contract_sha256']==digest
 assert json.loads((t.OUT/'qualification/verification.json').read_text())['passed']
 with exclusive_lock(t.OUT/'controller.lock'):
  assert not (t.OUT/'receipt.json').exists(),'no automatic restart/replay; new audited handoff needed'
  state=dict(pid=os.getpid(),version=VERSION,contract_sha256=digest,status='starting',completed=[],current=None,failures=[],old_fine_resumed=False)
  def save():atomic_json(t.OUT/'receipt.json',state)
  save();previous={g:None for g in [0,-6,6]};epochs=[]
  for epoch in range(1,11):
   receipts={};traces={};reference=None
   for gain in [0,-6,6]:
    for n,s in contract['sources'].items():assert t.sha(t.OUT/'sources'/n)==s
    assert not (t.OUT/'pause_request.json').exists(),'user pause before any new job'
    rel,job,result=launch(gain,epoch,previous[gain],reference)
    state.update(status='training',current=dict(epoch=epoch,gain_db=gain,relative=rel,job_relative=job),remote_job=result);save()
    while result['state'] in ['starting','running']:
     time.sleep(30);result=status(job);state['remote_job']=result;save()
    assert result['state']=='completed','lost job: stop and preserve; never replay'
    t.persist(job)
    exists=json.loads(t.remote_python('import json;from pathlib import Path;print(json.dumps(Path(%r).exists()))'%(t.RUN+'/'+rel)))
    if exists:t.persist(rel)
    if result['returncode']!=0:
     state['failures'].append(dict(epoch=epoch,gain_db=gain,execution=result));state['status']='terminal_failure_preserved_no_new_jobs';save();return
    r=json.loads((t.OUT/rel/'epoch_receipt.json').read_text());assert r['passed'] and r['normalization_contract_sha256']==digest
    r['_receipt']=t.RUN+'/'+rel+'/epoch_receipt.json';receipts[gain]=r;traces[gain]=t.OUT/rel/'attempts.jsonl'
    previous[gain]=r  # retain all repeat checkpoints; no cache retirement
    if gain==0:reference=t.RUN+'/'+rel+'/attempts.jsonl'
    state['completed'].append(dict(epoch=epoch,gain_db=gain,receipt=rel+'/epoch_receipt.json'));save()
   block=t.qualify_epoch(receipts,traces)
   # Original LR schedule is compared in addition to the inherited sampling fields.
   lr=[[json.loads(x)['optimizer_lr'] for x in Path(traces[g]).read_text().splitlines()] for g in [0,-6,6]]
   assert lr[0]==lr[1]==lr[2];block['actual_lr_equal']=True;epochs.append(block)
   p=t.OUT/'blocks'/f'coarse_ep{epoch:04d}.json';p.parent.mkdir(exist_ok=True);atomic_json(p,block)
  phase=t.qualify_phase(epochs,previous);atomic_json(t.OUT/'blocks/coarse_qualification.json',phase)
  ready=dict(passed=True,identity='PatrickLeahy',version=VERSION,phase=phase,final_receipts={str(g):r for g,r in previous.items()},contract_sha256=digest,no_training_job_active=True,normalization=contract['normalization'],new_27_grid_required=True)
  atomic_json(t.OUT/'coarse_ready_for_27_grid.json',ready)
  t.publish('coarse_ready_for_27_grid.json',ready)
  state.update(status='coarse10_complete_ready_for_new27',current=None,phase=phase);save()
if __name__=='__main__':
 try:run()
 except BaseException:
  atomic_json(t.OUT/'controller_failure.json',dict(pid=os.getpid(),status='stopped_no_automatic_replay',traceback=traceback.format_exc()));raise
