"""Isolated, bounded FP16 diagnostic; never writes into formal training attempts."""
import argparse,copy,hashlib,json,os,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
sys.path.insert(0,str(BASE))
from runtime_common import configure,dataset,seed_domain,tensor_state_sha
from execution_state import tree_sha,restore_model_counters,model_counters,verify_full_checkpoint,rng_state
configure()
from controlled_training import ControlledTrainer,ControlledLoader
from nerf_triplane.utils import Trainer
from nerf_triplane.network import NeRFNetwork

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def stats(x):
 if not torch.is_tensor(x):return str(type(x).__name__)
 y=x.detach()
 result=dict(shape=list(y.shape),dtype=str(y.dtype),numel=y.numel())
 if y.numel():
  result.update(finite=bool(torch.isfinite(y).all()),nan=int(torch.isnan(y).sum()),inf=int(torch.isinf(y).sum()),min=float(y.min()),max=float(y.max()))
 return result

def run(out,steps,mode):
 out=Path(out);assert 'fp16_adapter_environment_audit' in str(out) and not out.exists()
 out.mkdir(parents=True)
 ep=BASE/'training/SherrodBrown/seed42/gain_+6/coarse/ep0004/attempt01'
 info=json.loads((ep/'epoch_receipt.json').read_text());cp=Path(info['checkpoint_path'])
 assert sha(cp)==info['checkpoint_sha256']
 seed_domain(42,0,'generator_initialization')
 data=dataset('SherrodBrown','train',6,42,'coarse');opt=data.opt;opt.workspace=str(out/'workspace')
 seed_domain(42,0,'generator_initialization')
 model=NeRFNetwork(opt).cuda();previous=torch.load(cp,map_location='cpu',weights_only=False)
 model.load_state_dict(previous['model'],strict=True)
 optimizer=lambda m:torch.optim.AdamW(m.get_params(opt.lr,opt.lr_net),betas=(0,.99),eps=1e-8)
 scheduler=lambda o:torch.optim.lr_scheduler.LambdaLR(o,lambda step:.5**(step/opt.iters))
 trainer=ControlledTrainer('ngp',opt,model,device=torch.device('cuda'),workspace=opt.workspace,
   criterion=torch.nn.MSELoss(reduction='none'),optimizer=optimizer,lr_scheduler=scheduler,
   ema_decay=.95,fp16=True,metrics=[],scheduler_update_every_step=True,use_tensorboardX=False,use_checkpoint=str(cp))
 verify_full_checkpoint(trainer,previous);restore_model_counters(model,info['model_counters'])
 trainer.stats=copy.deepcopy(info['post_validation_stats']);trainer.ema.store()
 assert tree_sha(trainer.ema.collected_params)==info['post_validation_ema_collected_sha256']
 opt.finetune_lips=info['next_finetune_lips'];model.aud_features=data.auds;model.eye_area=data.eye_area;model.poses=data.poses
 trainer.epoch=5
 trainer.attach_controls(42,out/'controlled_trace.jsonl',info['cumulative_successful_updates'])
 receipt=dict(mode=mode,scope='diagnostic_only_not_formal_resume',fp16=True,max_attempts=steps,
  python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),os=platform.platform(),
  checkpoint_sha256=sha(cp),state_before=tensor_state_sha(model.state_dict()),source_sha256=sha(__file__),
  original_failure_unchanged=True,completed_attempts=0,events=[],started=time.time(),state='running')
 def persist(): (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=True))
 persist(); latest={};seen=set();current={};diag=(out/'diagnostic_trace.jsonl').open('x')
 def event(kind,extra):
  if kind in seen:return
  seen.add(kind)
  item=dict(kind=kind,global_step=trainer.global_step,sampling=trainer.current_sampling,scale=float(trainer.scaler.get_scale()),**extra)
  name=kind+'.pth'
  torch.save(dict(model=model.state_dict(),optimizer=trainer.optimizer.state_dict(),scaler=trainer.scaler.state_dict(),
   scheduler=trainer.lr_scheduler.state_dict(),ema=trainer.ema.state_dict(),model_counters=model_counters(model),
   rng=rng_state(),batch=current.get('batch'),global_step=trainer.global_step,note='diagnostic replay state, never the lost original failure state'),out/name)
  item['state_file']=name;item['state_sha256']=sha(out/name);receipt['events'].append(item);persist()
  print(json.dumps(item),flush=True)
 density_watch={'occupancy':False,'captured':False}
 original_density=model.density
 def density(*a,**k):
  result=original_density(*a,**k)
  if density_watch['occupancy'] and not density_watch['captured']:
   bad=~torch.isfinite(result['sigma'])
   if bool(bad.any()):
    density_watch['captured']=True
    torch.save(dict(x=a[0][bad].detach().cpu(),enc_a=a[1].detach().cpu(),eye=a[2].detach().cpu() if len(a)>2 and a[2] is not None else None,sigma=result['sigma'][bad].detach().cpu(),global_step=trainer.global_step,note='exact first nonfinite density inputs in new official-environment replay'),out/'first_nonfinite_density_inputs.pth')
    receipt['first_nonfinite_density_inputs_sha256']=sha(out/'first_nonfinite_density_inputs.pth');persist()
  return result
 model.density=density
 original_extra=model.update_extra_state
 def extra(*a,**k):
  density_watch['occupancy']=True
  try:result=original_extra(*a,**k)
  finally:density_watch['occupancy']=False
  if not bool(torch.isfinite(model.density_grid).all()):
   event('first_nonfinite_occupancy_grid',dict(stage='after_original_occupancy_before_next_train_step',batch_is_previous_step_not_event_input=True,grid=stats(model.density_grid),mean_density=float(model.mean_density)))
  return result
 model.update_extra_state=extra
 original_render=model.render
 def render(*a,**k):
  v=original_render(*a,**k);latest.clear();latest.update(v);return v
 model.render=render
 base_train=trainer.train_step
 def train_step(batch):
  current.clear();current['batch']=batch
  if mode=='direct_original_step':
   seed_domain(42,trainer.global_step,'render_perturbation_and_regularization')
   result=Trainer.train_step(trainer,batch);trainer.last_loss=float(result[2].detach())
  else:result=base_train(batch)
  current['loss']=float(result[2].detach())
  if not np.isfinite(current['loss']):
   event('nonfinite_forward_loss',dict(loss=current['loss'],render={k:stats(v) for k,v in latest.items() if torch.is_tensor(v)},inputs={k:stats(v) for k,v in batch.items() if torch.is_tensor(v)}))
   raise FloatingPointError('nonfinite forward loss; captured before backward/optimizer')
  return result
 trainer.train_step=train_step
 original_step=trainer.scaler.step
 def step(o,*a,**k):
  bad=[name for name,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
  current['nonfinite_scaled_gradients']=bad
  if bad:event('first_nonfinite_scaled_gradient',dict(loss=current['loss'],bad=bad,gradients={name:stats(p.grad) for name,p in model.named_parameters() if name in bad},render={k:stats(v) for k,v in latest.items() if torch.is_tensor(v)}))
  if float(trainer.scaler.get_scale())<1:event('first_scale_below_one',dict(loss=current['loss'],bad=bad))
  return original_step(o,*a,**k)
 trainer.scaler.step=step
 original_update=trainer.scaler.update
 def update(*a,**k):
  value=original_update(*a,**k)
  row=dict(global_step=trainer.global_step,loss=current['loss'],scale_before=trainer._scale_before,scale_after=float(trainer.scaler.get_scale()),
   successful_updates=trainer.update_counter.successful,bad_gradients=current['nonfinite_scaled_gradients'],sampling=trainer.current_sampling)
  diag.write(json.dumps(row)+'\n');diag.flush();receipt['completed_attempts']+=1
  if receipt['completed_attempts']%100==0:persist()
  return value
 trainer.scaler.update=update
 class Bounded:
  batch_size=1;_data=data
  def __len__(self):return steps
  def __iter__(self):
   for i,batch in enumerate(ControlledLoader(data,trainer,42)):
    if i>=steps:break
    yield batch
 try:
  trainer.train_one_epoch(Bounded());receipt['state']='bounded_replay_completed'
 except Exception as e:
  receipt['state']='diagnostic_stopped';receipt['error']=repr(e);receipt['traceback']=traceback.format_exc()
 finally:
  trainer.trace.flush();trainer.trace.close();trainer.update_counter.close();diag.close()
  receipt['ended']=time.time();receipt['elapsed']=receipt['ended']-receipt['started'];receipt['checkpoint_sha256_after']=sha(cp)
  assert receipt['checkpoint_sha256_after']==receipt['checkpoint_sha256']
  persist();print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--mode',choices=['controlled','direct_original_step'],default='controlled')
 a=p.parse_args();assert 1<=a.steps<=6000;run(a.out,a.steps,a.mode)
