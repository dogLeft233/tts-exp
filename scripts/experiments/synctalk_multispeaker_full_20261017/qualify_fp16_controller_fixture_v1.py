"""CUDA fixture for actual original accounting plus separately versioned guard."""
import copy,hashlib,json,os,sys
from pathlib import Path
import torch
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017');sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'fp16_revision_v1'))
import runtime_common as rc
if os.environ.get('AUDIT_OFFICIAL_SOURCE'):rc.OFFICIAL=Path(os.environ['AUDIT_OFFICIAL_SOURCE'])
rc.configure()
import controlled_training as ct
from execution_state import tree_sha
# Old Torch has no public post-step hook; same counting-only bridge as old audit.
if not hasattr(torch.optim.Optimizer,'register_step_post_hook'):
 class OldCounter:
  def __init__(self,optimizer,initial=0):
   self.successful=initial;self.optimizer=optimizer;self.original=optimizer.step
   def step(*a,**k):
    result=self.original(*a,**k);self.successful+=1;return result
   optimizer.step=step
  def close(self):self.optimizer.step=self.original
 ct.UpdateCounter=OldCounter
from controlled_training_fp16_v1 import GuardedControlledTrainer
from fp16_preupdate_guard_v1 import NumericalGuardError
out=Path(sys.argv[1]);assert not out.exists();out.mkdir()
torch.manual_seed(42);template=torch.nn.Linear(4,2).cuda().state_dict();results=[]
for guarded in [False,True]:
 cls=GuardedControlledTrainer if guarded else ct.ControlledTrainer
 trainer=cls.__new__(cls);trainer.model=torch.nn.Linear(4,2).cuda();trainer.model.load_state_dict(template);trainer.model.update_extra_state=lambda:None
 trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=.001,betas=(0,.99),eps=1e-8)
 trainer.scaler=torch.cuda.amp.GradScaler(init_scale=128);trainer.epoch=1;trainer.global_step=0
 def fixture_forward(data):
  with torch.cuda.amp.autocast(enabled=True):loss=(trainer.model(data)**2).mean()
  trainer.last_loss=float(loss.detach());return None,None,loss
 trainer.train_step=fixture_forward
 trace=out/('guarded.jsonl' if guarded else 'original.jsonl');trainer.attach_controls(42,trace)
 for index in range(3):
  trainer.global_step=index+1;trainer.optimizer.zero_grad()
  trainer.current_sampling=dict(frame_index=index,source_audio_id=index,lip_mode=False,pixel_index_sha256='fixture',actual_rays=4,rect=None)
  value=trainer.train_step(torch.ones(4,4,device='cuda'))
  trainer.scaler.scale(value[2]).backward()
  if index==1:next(trainer.model.parameters()).grad.fill_(float('inf'))
  trainer.scaler.step(trainer.optimizer);trainer.scaler.update()
 trainer.trace.flush();rows=[json.loads(x) for x in trace.read_text().splitlines()]
 assert [r['successful_update'] for r in rows]==[True,False,True]
 assert trainer.update_counter.successful==2
 results.append(dict(guarded=guarded,rows=rows,state_sha256=tree_sha(dict(model=trainer.model.state_dict(),optimizer=trainer.optimizer.state_dict(),scaler=trainer.scaler.state_dict())),optimizer_updates=2))
 if guarded:
  trainer.global_step=4;trainer.optimizer.zero_grad();state=trainer.scaler.state_dict();state['scale']=2.0**-128;trainer.scaler.load_state_dict(state)
  before=tree_sha(dict(model=trainer.model.state_dict(),optimizer=trainer.optimizer.state_dict(),scaler=trainer.scaler.state_dict()))
  value=trainer.train_step(torch.ones(4,4,device='cuda'));trainer.scaler.scale(value[2]).backward()
  try:trainer.scaler.step(trainer.optimizer);raise AssertionError('guard did not reject')
  except NumericalGuardError as e:assert e.event['reason']=='nonfinite_inverse_scale_before_unscale'
  assert before==tree_sha(dict(model=trainer.model.state_dict(),optimizer=trainer.optimizer.state_dict(),scaler=trainer.scaler.state_dict()))
  trainer.trace.flush();assert len(trace.read_text().splitlines())==3
  terminal=json.loads(Path(str(trace)+'.numerical_failure.json').read_text());assert terminal['global_step']==4 and trainer.update_counter.successful==2
 trainer.close_controls()
assert results[0]['state_sha256']==results[1]['state_sha256']
assert results[0]['rows']==results[1]['rows']
report=dict(torch=torch.__version__,passed=True,original_and_guarded_trace_exact=True,state_exact=True,fixture_optimizer_updates=4,formal_optimizer_updates=0,terminal_receipt_separate=True,no_false_completed_row=True,results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
