"""Six isolated real FP16 attempts; exact full-state roundtrip before formal training."""
import copy,itertools,json,sys,gc
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017');sys.path.insert(0,str(ROOT))
import torch
from patrick_rms_repeat_runtime_v1 import install,RUN
from flow_stream import sha
from execution_state import restore_model_counters,model_counters,verify_full_checkpoint,tree_sha
from runtime_common import configure,dataset,seed_domain,tensor_state_sha
def run():
 contract=install();configure()
 from nerf_triplane.network import NeRFNetwork
 from controlled_training import ControlledLoader
 from controlled_training_trial_v1 import TrialControlledTrainer
 out=RUN/'qualification';out.mkdir(exist_ok=False)
 initial_path=ROOT/'capacity_qualification/PatrickLeahy/initial_seed42.pt'
 assert sha(initial_path)==contract['initial_checkpoint_sha256']
 initial=torch.load(initial_path,map_location='cpu',weights_only=False)
 rows=[];sampling=[]
 for gain in [0,-6,6]:
  folder=out/f'gain_{gain:+d}/coarse/ep0001/attempt01';folder.mkdir(parents=True)
  data=dataset('PatrickLeahy','train',gain,42,'coarse');opt=data.opt;opt.workspace=str(folder/'workspace')
  seed_domain(42,0,'generator_initialization');model=NeRFNetwork(opt).cuda()
  model.load_state_dict(initial['model'],strict=True);restore_model_counters(model,initial['model_counters'])
  assert tensor_state_sha(model.state_dict())==contract['initial_model_tensor_sha256']
  optimizer=lambda m:torch.optim.AdamW(m.get_params(opt.lr,opt.lr_net),betas=(0,.99),eps=1e-8)
  scheduler=lambda o:torch.optim.lr_scheduler.LambdaLR(o,lambda step:.5**(step/opt.iters))
  trainer=TrialControlledTrainer('ngp',opt,model,device=torch.device('cuda'),workspace=opt.workspace,criterion=torch.nn.MSELoss(reduction='none'),optimizer=optimizer,lr_scheduler=scheduler,ema_decay=.95,fp16=True,metrics=[],scheduler_update_every_step=True,use_tensorboardX=False,use_checkpoint='scratch')
  model.aud_features=data.auds;model.eye_area=data.eye_area;model.poses=data.poses;model.mark_untrained_grid(data.poses,data.intrinsics)
  trainer.epoch=1;trace=folder/'attempts.jsonl';trainer.attach_controls(42,trace)
  full=ControlledLoader(data,trainer,42)
  class ShortLoader:
   batch_size=1;has_gt=True
   def __len__(self):return 2
   def __iter__(self):return itertools.islice(iter(full),2)
  trainer.train_one_epoch(ShortLoader());assert trainer.global_step==2
  successes=trainer.update_counter.successful;trainer.close_controls()
  counters=model_counters(model);trainer.save_checkpoint(name='entry_roundtrip',full=True,best=False,remove_old=False)
  cp=Path(trainer.ckpt_path)/'entry_roundtrip.pth';saved=torch.load(cp,map_location='cpu',weights_only=False)
  other=NeRFNetwork(opt).cuda()
  restored=TrialControlledTrainer('ngp',opt,other,device=torch.device('cuda'),workspace=str(folder/'restore'),criterion=torch.nn.MSELoss(reduction='none'),optimizer=optimizer,lr_scheduler=scheduler,ema_decay=.95,fp16=True,metrics=[],scheduler_update_every_step=True,use_tensorboardX=False,use_checkpoint=str(cp))
  verify_full_checkpoint(restored,saved);restore_model_counters(other,counters)
  assert tree_sha(model_counters(other))==tree_sha(counters)
  records=[json.loads(line) for line in trace.read_text().splitlines()];assert len(records)==2
  assert sum(r['successful_update'] for r in records)==successes
  fields=['frame_index','source_audio_id','lip_mode','pixel_index_sha256','actual_rays','rect','optimizer_lr']
  sampling.append([{k:r[k] for k in fields} for r in records])
  row=dict(gain_db=gain,attempts=2,actual_successful_updates=successes,amp_skips=2-successes,checkpoint_sha256=sha(cp),model_counters=counters,full_state_roundtrip_exact=True,trace_sha256=sha(trace),passed=True)
  (folder/'entry_receipt.json').write_text(json.dumps(row,indent=2));rows.append(row)
  del restored,other,trainer,model,data,full,saved;gc.collect();torch.cuda.empty_cache()
 assert sampling[0]==sampling[1]==sampling[2]
 result=dict(passed=True,contract_sha256=sha(RUN/'contract.json'),cases=rows,total_attempts=6,actual_successful_updates=sum(r['actual_successful_updates'] for r in rows),ordinary_amp_skips=sum(r['amp_skips'] for r in rows),actual_sampling_equal=True,formal_attempts=0,initial_unchanged=sha(initial_path)==contract['initial_checkpoint_sha256'],scope='real initial FP16 entry and full checkpoint roundtrip; not full training stability')
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':run()
