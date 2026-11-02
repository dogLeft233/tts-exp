"""Run the exact baseline50 trainer; observe and guard without changing math."""
from pathlib import Path
import json,hashlib,random,sys,traceback
import numpy as np
import torch
from common import *
def finite(x):
 if torch.is_tensor(x):return bool(torch.isfinite(x).all()) if x.is_floating_point() else True
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
 return True
def main():
 configure();p=json.loads((ROOT/'protocol.json').read_text())
 admission=json.loads((ROOT/'formal_admission.json').read_text());assert admission['passed']
 assert admission['protocol_sha256']==sha(ROOT/'protocol.json')
 for name,digest in admission['source_sha256'].items():assert sha(ROOT/name)==digest
 assert json.loads((ROOT/'qualification/receipt.json').read_text())['passed']
 assert sha(ROOT/'initialization.pt')==p['baseline_initialization_file_sha256']
 import baseline_train_original as baseline
 import nerf_triplane.provider as provider
 from nerf_triplane.utils import Trainer
 from nerf_triplane.renderer import NeRFRenderer
 original_collate=provider.NeRFDataset.collate;original_train_step=Trainer.train_step;original_step=torch.optim.AdamW.step
 original_update=NeRFRenderer.update_extra_state;original_save=torch.save
 context={};counts={'success':0};out=ROOT/'training/gain_0'
 # The exact baseline train function creates this output directory itself.
 assert not out.exists();trace=ROOT/'formal_trace.jsonl';log=trace.open('x')
 def collate(self,index):
  data=original_collate(self,index)
  if self.type=='train':
   i=int(index[0]);f=self.source_frames[i];context['sample']=dict(index=i,source_img_id=f['img_id'],source_aud_id=f['aud_id'])
  return data
 def train_step(owner,data):
  context['owner']=owner
  result=original_train_step(owner,data);assert torch.isfinite(result[2]).all(),f'nonfinite_loss_step_{owner.global_step}'
  context['event']=dict(step=owner.global_step,loss=float(result[2].detach()),**context['sample']);return result
 def step(optimizer,*args,**kwargs):
  params=[x for g in optimizer.param_groups for x in g['params'] if x.grad is not None]
  assert bool(torch.stack([torch.isfinite(x.grad).all() for x in params]).all()),'nonfinite_gradient_before_optimizer'
  lr=float(optimizer.param_groups[0]['lr']);result=original_step(optimizer,*args,**kwargs);counts['success']+=1
  row=dict(context['event'],success=True,successful_updates=counts['success'],lr_before_step=lr,skip=0)
  log.write(json.dumps(row)+'\n')
  if counts['success']%100==0 or counts['success']==1:log.flush()
  return result
 def update(model,*args,**kwargs):
  result=original_update(model,*args,**kwargs)
  assert torch.isfinite(model.density_grid).all() and np.isfinite(float(model.mean_density)),'nonfinite_density_refresh'
  return result
 def save(state,path,*args,**kwargs):
  if isinstance(state,dict) and 'optimizer' in state and 'ema' in state and 'step' in state:
   assert finite(state),'nonfinite_checkpoint'
   state=dict(state,rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
    counters={'attempts':state['step'],'success':counts['success'],'skip':0},data_contract_sha256=p['prepared_manifest_sha256'])
   log.flush()
  return original_save(state,path,*args,**kwargs)
 provider.NeRFDataset.collate=collate;Trainer.train_step=train_step;torch.optim.AdamW.step=step;NeRFRenderer.update_extra_state=update;torch.save=save
 try:
  baseline.train(p,0)
  assert counts['success']==60000
  save_json(ROOT/'formal_complete.json',dict(passed=True,success=60000,attempts=60000,skip=0,trace_sha256=sha(trace),
   final_checkpoint_sha256=sha(out/'checkpoints/step060000.pt'),original_trainer_source_sha256=sha(ROOT/'baseline_train_original.py'),new_training_data_only=True))
 except Exception as exc:
  owner=context.get('owner');model=owner.model if owner is not None else None;state={}
  tb=exc.__traceback__
  while tb:
   scope=tb.tb_frame.f_locals
   if 'optimizer' in scope and 'model' in scope:
    for k in ['model','optimizer','scheduler','ema']:
     if k in scope:state[k]=scope[k].state_dict()
    state['step']=scope.get('step');state['metadata']=model_metadata(scope['model'])
   tb=tb.tb_next
  state['counters']=counts;state['rng']={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
  original_save(state,ROOT/'terminal_state.pt');log.flush()
  save_json(ROOT/'formal_failure.json',dict(type=type(exc).__name__,message=str(exc),successful_updates=counts['success'],last_event=context.get('event'),
   terminal_state_sha256=sha(ROOT/'terminal_state.pt'),traceback=traceback.format_exc(),automatic_restart_allowed=False));raise
 finally:log.close()
if __name__=='__main__':main()
