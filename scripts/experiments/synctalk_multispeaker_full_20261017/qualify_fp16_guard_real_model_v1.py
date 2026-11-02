"""Four zero-update cases: same-graph causal probe and real-model rejection."""
import copy, hashlib, json, os, sys
from pathlib import Path
import torch
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017');sys.path.insert(0,str(BASE))
import runtime_common as rc
if os.environ.get('AUDIT_OFFICIAL_SOURCE'):rc.OFFICIAL=Path(os.environ['AUDIT_OFFICIAL_SOURCE'])
rc.configure()
from execution_state import restore_model_counters,model_counters,tree_sha
from nerf_triplane.network import NeRFNetwork
from nerf_triplane.utils import Trainer
from fp16_preupdate_guard_v1 import attach_trainer_guard,NumericalGuardError

def load(p):
 return torch.load(p,map_location='cpu') if torch.__version__.startswith('1.12.') else torch.load(p,map_location='cpu',weights_only=False)
def cuda(x):
 if torch.is_tensor(x):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 if isinstance(x,list):return [cuda(v) for v in x]
 if isinstance(x,tuple):return tuple(cuda(v) for v in x)
 return x
def brief(x):
 x=x.detach();return dict(nan=int(torch.isnan(x).sum()),inf=int(torch.isinf(x).sum()),dtype=str(x.dtype),shape=list(x.shape))
def badgrad(model):return [n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
out=Path(sys.argv[1]);assert not out.exists();out.mkdir()
audit=BASE/'controller_qualification/fp16_adapter_environment_audit_attempt01/official_environment_replay'
state=load(audit/'first_scale_below_one.pth')
opt=rc.options('SherrodBrown','train',6,42,'coarse');model=NeRFNetwork(opt).cuda();model.train()
model.load_state_dict(state['model']);restore_model_counters(model,state['model_counters'])
trainer=Trainer.__new__(Trainer);trainer.model=model;trainer.opt=opt;trainer.global_step=state['global_step'];trainer.flip_finetune_lips=False;trainer.criterion=torch.nn.MSELoss(reduction='none')
mode='baseline';records=[]
def hook(module,args,h):
 def backward(g):
  record=dict(mode=mode,logit_gradient=brief(g[...,0]),other_gradient=brief(g[...,1:]))
  if mode=='zero_density_logit_gradient':
   g=g.clone();g[...,0]=0
  record['returned_gradient']=brief(g);records.append(record)
  return g
 h.register_hook(backward)
handle=model.sigma_net.register_forward_hook(hook)
rc.seed_domain(42,trainer.global_step,'render_perturbation_and_regularization')
before=tree_sha(model.state_dict())
with torch.cuda.amp.autocast(enabled=True):value=Trainer.train_step(trainer,cuda(state['batch']))
loss=value[2];assert bool(torch.isfinite(loss));loss_sha=tree_sha(loss)
loss.backward(retain_graph=True);baseline=badgrad(model)
model.zero_grad();mode='zero_density_logit_gradient';loss.backward();intervened=badgrad(model)
assert tree_sha(loss)==loss_sha
handle.remove();model.load_state_dict(state['model']);restore_model_counters(model,state['model_counters']);assert tree_sha(model.state_dict())==before
rows=[dict(case='same_graph_baseline',loss=float(loss),loss_sha256=loss_sha,bad_gradients=baseline),dict(case='same_graph_zero_density_logit_gradient',loss=float(loss),loss_sha256=loss_sha,bad_gradients=intervened)]
(out/'partial.json').write_text(json.dumps(dict(results=rows,gradient_hooks=records),indent=2))
del value,loss;model.zero_grad();torch.cuda.empty_cache()

for name,filename in [('captured_nonfinite_grid','first_nonfinite_occupancy_grid.pth'),('synthetic_inverse_scale_on_finite_state','first_nonfinite_scaled_gradient.pth')]:
 s=load(audit/filename);model.load_state_dict(s['model']);restore_model_counters(model,s['model_counters']);model.zero_grad()
 trainer=Trainer.__new__(Trainer);trainer.model=model;trainer.opt=opt;trainer.global_step=s['global_step'];trainer.flip_finetune_lips=False;trainer.criterion=torch.nn.MSELoss(reduction='none')
 trainer.optimizer=torch.optim.AdamW(model.get_params(opt.lr,opt.lr_net),betas=(0,.99),eps=1e-8)
 trainer.optimizer.load_state_dict(s['optimizer'])
 trainer.scaler=torch.cuda.amp.GradScaler(enabled=True)
 scaler_state=copy.deepcopy(s['scaler'])
 if name.startswith('synthetic'):scaler_state['scale']=2.0**-128
 trainer.scaler.load_state_dict(scaler_state)
 # This counter wraps the real bound method; it does not fake optimizer math.
 calls={'optimizer':0,'forward':0}
 original_optimizer=trainer.optimizer.step;original_forward=trainer.train_step
 def counted_optimizer(*a,**k):calls['optimizer']+=1;return original_optimizer(*a,**k)
 def counted_forward(*a,**k):calls['forward']+=1;return original_forward(*a,**k)
 trainer.optimizer.step=counted_optimizer;trainer.train_step=counted_forward
 guard=attach_trainer_guard(trainer)
 def snapshot():return tree_sha(dict(model=model.state_dict(),counters=model_counters(model),optimizer=trainer.optimizer.state_dict(),scaler=trainer.scaler.state_dict()))
 digest=snapshot();event=None
 try:
  if name=='captured_nonfinite_grid':trainer.train_step(cuda(s['batch']))
  else:
   # Deliberate copied-state fixture; no NeRF forward or historical claim.
   finite_loss=next(model.parameters()).reshape(-1)[0]*0+1
   guard.check_loss(finite_loss);trainer.scaler.scale(finite_loss)
   for p in model.parameters():p.grad=torch.zeros_like(p)
   trainer.scaler.step(trainer.optimizer)
 except NumericalGuardError as exc:event=exc.event
 assert event is not None and calls==dict(optimizer=0,forward=0)
 expected='nonfinite_state_before_optimizer' if name=='captured_nonfinite_grid' else 'nonfinite_inverse_scale_before_unscale'
 assert event['reason']==expected,event
 assert snapshot()==digest
 rows.append(dict(case=name,event=event,calls=calls,state_before_sha256=digest,state_after_sha256=snapshot(),state_unchanged=True))
 guard.close();(out/'partial.json').write_text(json.dumps(dict(results=rows,gradient_hooks=records),indent=2))
report=dict(torch=torch.__version__,fp16=True,model_cases=4,optimizer_updates=0,scaler_updates=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=rows,gradient_hooks=records,scope='single forward graph and original Trainer attachment; not full supervisor integration',same_graph_density_intervention_eliminates_nonfinite=bool(baseline) and not intervened)
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
