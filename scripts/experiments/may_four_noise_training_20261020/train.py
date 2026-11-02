"""Four independent arms, paired state/frames/rays/LR; stop on first failure."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,time,hashlib,traceback,os
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
from runtime import *
def initialize():
 assert not (ROOT/'initialization.pt').exists()
 d=dataset('hubert_base','train',True);seed(0,'initialization');m=network(d.opt);bind(m,d)
 m.mark_untrained_grid(d.poses,d.intrinsics);finite(m);density_finite(m)
 state=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},metadata=metadata(m),rng=rng_state(),learned_generator_loaded=False)
 torch.save(state,ROOT/'initialization.pt')
 write(ROOT/'initialization.json',dict(file_sha256=sha(ROOT/'initialization.pt'),tensor_sha256=tensor_sha(state['model']),metadata=state['metadata'],seed=42,learned_generator_loaded=False))
 old=np.load(ROOT/'old_frame_order.npy');assert len(old)==60000
 np.save(ROOT/'frame_order.npy',old[:12000]);write(ROOT/'frame_order_receipt.json',dict(sha256=sha(ROOT/'frame_order.npy'),same_report50_prefix=True))
def save_state(path,m,opt,sched,ema,step,arm):
 state=dict(step=step,successful_optimizer_updates=step,attempts=step,skips=0,arm=arm,protocol_sha256=sha(ROOT/'protocol.json'),initial_sha256=sha(ROOT/'initialization.pt'),model={k:v.detach().cpu() for k,v in m.state_dict().items()},optimizer=opt.state_dict(),scheduler=sched.state_dict(),ema=ema.state_dict(),metadata=metadata(m),rng=rng_state(),gradient={k:v.grad.detach().cpu() for k,v in m.named_parameters() if v.grad is not None},precision='FP32',scaler=None)
 torch.save(state,path);return state
def arm_run(arm,qualification=False):
 from nerf_triplane.utils import Trainer
 import nerf_triplane.provider as provider
 key=arm['id'];out=ROOT/('qualification' if qualification else 'training')/key;out.mkdir(parents=True,exist_ok=False)
 ds=dataset(arm['id'],'train',arm['norm'],arm['input_scale']);vs={scale:dataset('clean','val',True,scale) for scale in [0.5,1.0]}
 initial=torch.load(ROOT/'initialization.pt',map_location='cpu',weights_only=False);m=network(ds.opt);m.load_state_dict(initial['model'],strict=True);restore_metadata(m,initial['metadata']);bind(m,ds);set_rng(initial['rng'])
 assert tensor_sha(m.state_dict())==json.loads((ROOT/'initialization.json').read_text())['tensor_sha256']
 opt=torch.optim.AdamW(m.get_params(ds.opt.lr,ds.opt.lr_net),betas=(0,.99),eps=1e-8)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:.5**(s/60000));ema=ExponentialMovingAverage(m.parameters(),decay=.95)
 owner=SimpleNamespace(opt=ds.opt,model=m,criterion=torch.nn.L1Loss(reduction='none'),global_step=0,flip_finetune_lips=False,flip_init_lips=False)
 budget=2 if qualification else 12000;order=np.load(ROOT/'frame_order.npy');rows=np.rint(np.linspace(0,549,16)).astype(int).tolist()
 original=provider.get_rays;sample=None
 def observed(*a,**kw):
  nonlocal sample
  r=original(*a,**kw);sample=hashlib.sha256(r['inds'].detach().cpu().contiguous().numpy().tobytes()).hexdigest();return r
 reference=ROOT/('qualification' if qualification else 'training')/'clean_s100'/'trace.jsonl'
 ref=reference.open() if key!='clean_s100' else None
 log=(out/'trace.jsonl').open('x');curve=[];step=0;successful=0;phase='before_training';started=time.monotonic()
 try:
  seed(0,'occupancy')
  with torch.no_grad():m.update_extra_state()
  density_finite(m)
  if not qualification:
   curve.append(dict(step=0,validation=validation(m,ema,vs,rows)));write(out/'validation_curve.json',curve)
  for step in range(1,budget+1):
   if step>1 and (step-1)%ds.opt.update_extra_interval==0:
    seed(step-1,'occupancy')
    with torch.no_grad():m.update_extra_state()
    density_finite(m)
   seed(step,'ray_sampling');sample=None;provider.get_rays=observed
   data=ds.collate([int(order[step-1])]);provider.get_rays=original;assert sample
   lr=[float(g['lr']) for g in opt.param_groups]
   if ref:
    baseline=json.loads(ref.readline());assert baseline['step']==step and baseline['index']==int(order[step-1]) and baseline['ray_sha256']==sample and baseline['lr']==lr
   seed(step,'forward_perturbation');owner.global_step=step;opt.zero_grad(set_to_none=True)
   phase='forward';y=Trainer.train_step(owner,data);loss=y[2];assert torch.isfinite(loss),'nonfinite loss'
   phase='backward';loss.backward();finite(m,grad=True)
   phase='optimizer';opt.step();successful+=1;finite(m);sched.step();phase='after_optimizer'
   if step%1000==0:ema.update()
   row=dict(step=step,index=int(order[step-1]),source_id=ds.source_frames[int(order[step-1])]['img_id'],ray_sha256=sample,lr=lr,loss=float(loss.detach()),successful_optimizer_updates=step,skips=0)
   log.write(json.dumps(row)+'\n')
   if step%100==0 or step==1:
    log.flush();write(ROOT/'status.json',dict(phase='qualification' if qualification else 'training',arm=key,step=step,budget=budget,loss=row['loss'],elapsed_seconds=time.monotonic()-started));print(key,step,row['loss'],flush=True)
   if not qualification and step%2000==0:
    checkpoint=out/f'full_step{step:06d}.pt';save_state(checkpoint,m,opt,sched,ema,step,arm)
    curve.append(dict(step=step,validation=validation(m,ema,vs,rows)));write(out/'validation_curve.json',curve)
    write(out/'latest_checkpoint.json',dict(path=str(checkpoint),sha256=sha(checkpoint),step=step))
  log.close()
  if qualification:
   checkpoint=out/'roundtrip.pt';state=save_state(checkpoint,m,opt,sched,ema,step,arm)
   reloaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
   assert tensor_sha(state['model'])==tensor_sha(reloaded['model']);assert reloaded['successful_optimizer_updates']==2
   n=network(ds.opt);n.load_state_dict(reloaded['model'],strict=True);restore_metadata(n,reloaded['metadata'])
   oo=torch.optim.AdamW(n.get_params(ds.opt.lr,ds.opt.lr_net),betas=(0,.99),eps=1e-8)
   ss=torch.optim.lr_scheduler.LambdaLR(oo,lambda s:.5**(s/60000));oo.load_state_dict(reloaded['optimizer']);ss.load_state_dict(reloaded['scheduler'])
   ee=ExponentialMovingAverage(n.parameters(),decay=.95);ee.load_state_dict(reloaded['ema'])
   assert tensor_sha(n.state_dict())==tensor_sha(m.state_dict());assert len(oo.state)==len(opt.state) and ss.last_epoch==sched.last_epoch
   assert oo.state_dict()['param_groups']==opt.state_dict()['param_groups'] and ss.state_dict()==sched.state_dict()
   assert all(torch.equal(v,oo.state[next(q for q,p in zip(n.parameters(),m.parameters()) if p is param)][k]) for param,s in opt.state.items() for k,v in s.items() if torch.is_tensor(v))
  write(out/'receipt.json',dict(passed=True,arm=arm,steps=step,attempts=step,success=step,skip=0,checkpoint=checkpoint.name,checkpoint_sha256=sha(checkpoint),initial_sha256=sha(ROOT/'initialization.pt'),trace_sha256=sha(out/'trace.jsonl'),same_actual_frame_ray_lr=True,elapsed_seconds=time.monotonic()-started,fullstate_roundtrip=qualification,qualification_only=qualification))
 except BaseException as e:
  log.flush();save_state(out/'terminal_state.pt',m,opt,sched,ema,successful,arm)
  write(out/'failure.json',dict(error=repr(e),entered_step=step,actual_successful_optimizer_calls=successful,phase=phase,traceback=traceback.format_exc()))
  raise
 finally:
  provider.get_rays=original
  if ref:ref.close()
  if not log.closed:log.close()
def main():
 configure()
 if sys.argv[1]=='initialize':initialize();return
 arm=next(a for a in protocol()['arms'] if a['id']==sys.argv[2]);arm_run(arm,sys.argv[1]=='qualify')
if __name__=='__main__':main()
