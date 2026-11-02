"""CPU only complete-state and trace audit, usable after qualification or training."""
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
from protocol import OUT,sha,write
def main():
 mode=sys.argv[1];assert mode in ['qualification','training'];expected=2 if mode=='qualification' else 12000
 p=json.loads((OUT/'protocol.json').read_text());initial_sha=sha(OUT/'initialization.pt');rows=[];reference=None
 for arm in p['arms']:
  root=OUT/mode/arm['id'];r=json.loads((root/'receipt.json').read_text());assert r['passed'] and r['success']==r['attempts']==expected and r['skip']==0
  cp=root/r['checkpoint'];assert sha(cp)==r['checkpoint_sha256'] and r['initial_sha256']==initial_sha
  state=torch.load(cp,map_location='cpu',weights_only=False);assert state['step']==state['successful_optimizer_updates']==expected and state['precision']=='FP32' and state['scaler'] is None
  for group in ['model','gradient']:
   assert all(torch.isfinite(x).all() for x in state[group].values())
  for opt in state['optimizer']['state'].values():
   assert all(torch.isfinite(x).all() for x in opt.values() if torch.is_tensor(x))
   assert float(opt['step'])==expected
  assert all(torch.isfinite(x).all() for x in state['ema']['shadow_params']) and int(state['ema']['num_updates'])==expected//1000
  assert np.isfinite(state['metadata']['mean_density'])
  assert state['scheduler']['last_epoch']==expected
  for g in state['optimizer']['param_groups']:assert abs(g['lr']-g['initial_lr']*.5**(expected/60000))<1e-14
  t=[json.loads(x) for x in (root/'trace.jsonl').read_text().splitlines()];assert len(t)==expected and all(x['step']==i+1 and np.isfinite(x['loss']) and x['skips']==0 for i,x in enumerate(t))
  pair=[(x['step'],x['index'],x['source_id'],x['ray_sha256'],x['lr']) for x in t]
  if reference is None:reference=pair
  else:assert reference==pair
  rows.append(dict(arm=arm['id'],steps=expected,checkpoint_sha256=sha(cp),trace_sha256=sha(root/'trace.jsonl'),all_model_optimizer_gradient_ema_finite=True,source_execution_bound=True))
 write(OUT/(mode+'_independent_review.json'),dict(passed=True,mode=mode,arms=rows,initial_sha256=initial_sha,actual_paired_frame_ray_lr=True,new_gpu_forward=0,new_optimizer_calls=0,scope='entry qualification is not full training stability' if mode=='qualification' else 'small fixed budget only, not full convergence'))
 print(json.dumps(dict(passed=True,mode=mode,arms=len(rows),steps_each=expected)))
if __name__=='__main__':main()
