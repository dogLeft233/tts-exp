"""Zero-update repeated gradient/adapter parity at one common saved state."""
import copy,json,sys
from pathlib import Path
import torch,numpy as np
import diagnose_fp16_failure as d
from execution_state import rng_state,restore_rng,model_counters,restore_model_counters
from runtime_common import seed_domain,tensor_state_sha
from nerf_triplane.utils import Trainer

out=Path(sys.argv[1])
def measure(self,loader):
 data=loader._data;model=self.model
 start_model=copy.deepcopy(model.state_dict());start_counters=model_counters(model)
 start_optimizer=d.tree_sha(self.optimizer.state_dict());start_scaler=d.tree_sha(self.scaler.state_dict())
 results=[]
 for frame,step in [(4105,24001),(2931,28637),(3407,24001)]:
  pairs=[]
  for kind in ['controlled','direct']:
   for repeat in range(2):
    model.load_state_dict(start_model);restore_model_counters(model,start_counters);model.train();self.global_step=step
    seed_domain(42,step,'pixel_ray_patch_rectangle');batch=data.collate([frame])
    self.optimizer.zero_grad();seed_domain(42,step,'render_perturbation_and_regularization')
    with torch.cuda.amp.autocast(enabled=True):
     value=self.train_step(batch) if kind=='controlled' else Trainer.train_step(self,batch)
    self.scaler.scale(value[2]).backward()
    grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    row=dict(frame=frame,global_step=step,kind=kind,repeat=repeat,loss=float(value[2]),grad_sha256=d.tree_sha(grads),
      input_sha256=d.tree_sha(batch),nonfinite=[n for n,g in grads.items() if not bool(torch.isfinite(g).all())])
    if pairs:
     old=pairs[0][1];ss=0.;den=0.;ma=0.
     for n,g in grads.items():
      delta=g.double()-old[n].double();ss+=float((delta**2).sum());den+=float((old[n].double()**2).sum());ma=max(ma,float(delta.abs().max()))
     row.update(grad_rel_l2=(ss/max(den,1e-300))**.5,grad_max_abs=ma,loss_delta=row['loss']-pairs[0][0]['loss'])
    pairs.append((row,grads));results.append(row)
  print(json.dumps(results[-4:]),flush=True)
 model.load_state_dict(start_model);restore_model_counters(model,start_counters)
 assert d.tree_sha(self.optimizer.state_dict())==start_optimizer and d.tree_sha(self.scaler.state_dict())==start_scaler
 assert self.update_counter.successful==23996
 result=dict(optimizer_updates=0,scaler_updates=0,model_restored=True,results=results,
  explanation='FP16 scaled backward only. FP64 CPU reductions summarize differences; no FP32 model forward/backward.')
 (out/'zero_update_parity.json').write_text(json.dumps(result,indent=2))
d.ControlledTrainer.train_one_epoch=measure
d.run(out,1,'controlled')
