"""Frozen-state, zero-update interventions; never used by formal training."""
import copy,json,hashlib,os,sys
from pathlib import Path
import torch
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017');sys.path.insert(0,str(BASE))
import runtime_common as rc
if os.environ.get('AUDIT_OFFICIAL_SOURCE'):rc.OFFICIAL=Path(os.environ['AUDIT_OFFICIAL_SOURCE'])
rc.configure()
from execution_state import restore_model_counters,tree_sha
from nerf_triplane.network import NeRFNetwork
from nerf_triplane.utils import Trainer

def load(p):
 return torch.load(p,map_location='cpu') if torch.__version__.startswith('1.12.') else torch.load(p,map_location='cpu',weights_only=False)
def brief(t):
 x=t.detach();f=torch.isfinite(x)
 return dict(dtype=str(x.dtype),shape=list(x.shape),nan=int(torch.isnan(x).sum()),inf=int(torch.isinf(x).sum()),max=float(x[f].max()) if bool(f.any()) else None,min=float(x[f].min()) if bool(f.any()) else None)
def cuda(x):
 if torch.is_tensor(x):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 if isinstance(x,list):return [cuda(v) for v in x]
 if isinstance(x,tuple):return tuple(cuda(v) for v in x)
 return x
audit=BASE/'controller_qualification/fp16_adapter_environment_audit_attempt01'
out=Path(sys.argv[1]);assert not out.exists();out.mkdir()
opt=rc.options('SherrodBrown','train',6,42,'coarse');model=NeRFNetwork(opt).cuda();model.train()
original_density=model.density
rows=[];records=[];mode='baseline'

def sigma_hook(module,args,h):
 record={'logit':brief(h[...,0]),'sigma_input_components':{name:brief(args[0][...,lo:hi]) for name,lo,hi in [('spatial',0,36),('audio',36,68),('eye',68,75)]}}
 records.append(record)
 if h.requires_grad:
  def gradient(g):record['logit_gradient']=brief(g[...,0]);return g
  h.register_hook(gradient)

def density(*a,**k):
 v=original_density(*a,**k);record=records[-1];sigma=v['sigma'];record['sigma']=brief(sigma)
 if sigma.requires_grad:
  def gradient(g):
   product=sigma.detach()*g.detach()
   record.update(sigma_gradient=brief(g),exp_backward_product=brief(product),finite_product_exceeds_fp16=int((torch.isfinite(product)&(product.abs()>65504)).sum()),inf_sigma_times_zero=int((torch.isinf(sigma.detach())&(g.detach()==0)).sum()))
   return g
  sigma.register_hook(gradient)
 if mode=='detach_sigma':v['sigma']=v['sigma'].detach()
 if mode=='detach_eye_regularizer':v['ambient_eye']=v['ambient_eye'].detach()
 return v
model.density=density;handle=model.sigma_net.register_forward_hook(sigma_hook)
event=audit/'official_environment_replay/first_scale_below_one.pth';state=load(event)
trainer=Trainer.__new__(Trainer);trainer.model=model;trainer.opt=opt;trainer.global_step=state['global_step'];trainer.flip_finetune_lips=False;trainer.criterion=torch.nn.MSELoss(reduction='none')
for mode,multiplier in [('baseline',.5),('baseline',1.),('detach_sigma',1.),('detach_eye_regularizer',1.)]:
 model.load_state_dict(state['model']);restore_model_counters(model,state['model_counters']);model.zero_grad();records.clear()
 rc.seed_domain(42,trainer.global_step,'render_perturbation_and_regularization')
 before=rc.tensor_state_sha(model.state_dict())
 with torch.cuda.amp.autocast(enabled=True):value=Trainer.train_step(trainer,cuda(state['batch']))
 (value[2]*multiplier).backward()
 rows.append(dict(case='backward',intervention=mode,multiplier=multiplier,loss=float(value[2]),global_step=trainer.global_step,input_sha256=tree_sha(state['batch']),density=copy.deepcopy(records),bad_gradients=[n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]))
 model.load_state_dict(state['model']);restore_model_counters(model,state['model_counters']);assert rc.tensor_state_sha(model.state_dict())==before
 (out/'partial.json').write_text(json.dumps(rows,indent=2))

# Same exact first nonfinite density point, modifying one conditioning path at a time.
event2=audit/'official_environment_replay/first_nonfinite_occupancy_grid.pth';s2=load(event2)
pointfile=audit/'official_environment_replay/first_nonfinite_density_inputs.pth';point=load(pointfile)
ep=BASE/'training/SherrodBrown/seed42/gain_+6/coarse/ep0004/attempt01/epoch_receipt.json'
cp=Path(json.loads(ep.read_text())['checkpoint_path']);ep4=load(cp)['model']
for intervention in ['baseline','zero_audio','zero_eye','zero_both','ep4_weights','ep4_eye_attention','ep4_sigma','ep4_audio_attention']:
 model.load_state_dict(s2['model']);restore_model_counters(model,s2['model_counters']);model.zero_grad();records.clear();mode='baseline'
 source=copy.deepcopy(s2['model'])
 prefixes={'ep4_eye_attention':'eye_att_net.','ep4_sigma':'sigma_net.','ep4_audio_attention':'aud_ch_att_net.'}
 if intervention=='ep4_weights':model.load_state_dict(ep4)
 elif intervention in prefixes:
  for k in source:
   if k.startswith(prefixes[intervention]):source[k]=ep4[k]
  model.load_state_dict(source)
 before=rc.tensor_state_sha(model.state_dict())
 a=point['enc_a'].cuda();e=point['eye'].cuda()
 if intervention in ['zero_audio','zero_both']:a=torch.zeros_like(a)
 if intervention in ['zero_eye','zero_both']:e=torch.zeros_like(e)
 with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):v=model.density(point['x'].cuda(),a,e)
 rows.append(dict(case='density_channel_intervention',intervention=intervention,global_step=point['global_step'],input_file_sha256=hashlib.sha256(pointfile.read_bytes()).hexdigest(),density=copy.deepcopy(records),ambient_audio=brief(v['ambient_aud']),ambient_eye=brief(v['ambient_eye'])))
 assert rc.tensor_state_sha(model.state_dict())==before
 (out/'partial.json').write_text(json.dumps(rows,indent=2))
handle.remove()
report=dict(torch=torch.__version__,cuda=torch.version.cuda,fp16=True,model_cases=len(rows),optimizer_updates=0,scaler_updates=0,scope='frozen diagnostic states and explicit counterfactual interventions; not formal training',event_sha256=hashlib.sha256(event.read_bytes()).hexdigest(),density_state_sha256=hashlib.sha256(event2.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=rows)
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
