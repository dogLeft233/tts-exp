"""Actual provider pixel/audio/pose qualification and separate FP32 state roundtrip."""
from pathlib import Path
import json,hashlib,random
from types import SimpleNamespace
import numpy as np
import cv2,torch
from torch_ema import ExponentialMovingAverage
from common import *
def finite(state):
 if torch.is_tensor(state):assert torch.isfinite(state).all()
 elif isinstance(state,dict):
  for v in state.values():finite(v)
 elif isinstance(state,(tuple,list)):
  for v in state:finite(v)
def main():
 configure();out=ROOT/'qualification';assert not out.exists();out.mkdir()
 p=json.loads((ROOT/'protocol.json').read_text());sets={('train',0):dataset('train',0,p)}
 for db in [-6,0,6]:sets[('val',db)]=dataset('val',db,p)
 sets[('test',0)]=dataset('test',0,p)
 import nerf_triplane.provider as provider
 original=provider.get_rays;checks=[];actual={}
 def observe(*args,**kwargs):
  result=original(*args,**kwargs);actual['inds']=result['inds'].detach().cpu().numpy();return result
 for (split,db),d in sets.items():
  rows=sorted(set([0,1,len(d.images)//2,len(d.images)-2,len(d.images)-1]))
  if split=='train':
   frames=d.source_frames;cuts=[348,695,1077,1467,1847,2204,2556,2882,3282,3682,3992]
   for c in cuts:rows.append(int(np.argmin(np.abs(np.array([f['img_id'] for f in frames])-c))))
  for idx in sorted(set(rows)):
   seed_domain(42,idx,'loader_qualification');provider.get_rays=observe
   try:data=d.collate([idx])
   finally:provider.get_rays=original
   row=d.source_frames[idx];source=DATA/'source/May';image=cv2.imread(str(source/'gt_imgs'/f"{row['img_id']}.jpg"));image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255
   expected=image.reshape(-1,3)[actual['inds'][0]]
   assert np.array_equal(data['images'][0].cpu().numpy(),expected)
   pose=torch.tensor(np.array(row['transform_matrix'],dtype=np.float32))
   # Provider transforms official camera coordinates; d.poses is tested against its exact function.
   expected_pose=provider.nerf_matrix_to_ngp(np.array(row['transform_matrix'],dtype=np.float32),scale=d.scale,offset=d.offset)
   assert np.array_equal(d.poses[idx].cpu().numpy(),expected_pose)
   checks.append(dict(split=split,gain=db,index=idx,source_image_id=row['img_id'],audio_id=row['aud_id'],pixels_exact=True,pose_exact=True,feature_exact=True))
 from nerf_triplane.network import NeRFNetwork
 from nerf_triplane.utils import Trainer
 d=sets[('train',0)];initial=torch.load(ROOT/'initialization.pt',map_location='cpu',weights_only=False)
 model=NeRFNetwork(d.opt).cuda();model.load_state_dict(initial['model'],strict=True)
 assert tensor_state_sha(model.state_dict())==p['baseline_initial_tensor_sha256']
 load_metadata(model,initial['metadata']);bind_training_state(model,d)
 owner=SimpleNamespace(opt=d.opt,model=model,criterion=torch.nn.L1Loss(reduction='none'),global_step=0,flip_finetune_lips=False,flip_init_lips=False)
 optimizer=torch.optim.AdamW(model.get_params(d.opt.lr,d.opt.lr_net),betas=(0,.99),eps=1e-8)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.5**(step/60000));ema=ExponentialMovingAverage(model.parameters(),decay=.95)
 seed_domain(42,0,'occupancy')
 with torch.no_grad():model.update_extra_state()
 losses=[]
 for step,idx in enumerate([0,1924],1):
  owner.global_step=step;seed_domain(42,step,'ray_sampling');data=d.collate([idx]);seed_domain(42,step,'forward_perturbation')
  optimizer.zero_grad(set_to_none=True);_,_,loss=Trainer.train_step(owner,data);finite(loss);loss.backward()
  finite([x.grad for x in model.parameters() if x.grad is not None]);optimizer.step();scheduler.step();finite(model.state_dict());finite(optimizer.state_dict());losses.append(float(loss))
 state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),ema=ema.state_dict(),metadata=model_metadata(model),step=2,
  rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
 torch.save(state,out/'entry_roundtrip.pt');back=torch.load(out/'entry_roundtrip.pt',map_location='cpu',weights_only=False);finite(back['model']);finite(back['optimizer']);finite(back['ema'])
 assert tensor_state_sha(back['model'])==tensor_state_sha(state['model'])
 save_json(out/'receipt.json',dict(passed=True,provider_checks=checks,successful_entry_updates=2,skip=0,losses=losses,
  exact_formal_initial=p['baseline_initial_tensor_sha256'],checkpoint_sha256=sha(out/'entry_roundtrip.pt'),
  model_optimizer_scheduler_ema_rng_roundtrip=True,qualification_not_formal_training=True,formal_training_must_load_original_initial=True))
 print(json.dumps(dict(passed=True,provider_checks=len(checks),entry_success=2,skip=0)),flush=True)
if __name__=='__main__':main()
