"""Zero-update original-mathematics probe of separately captured diagnostic states."""
import copy,json,os,sys,traceback,hashlib
from pathlib import Path
import torch
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
sys.path.insert(0,str(BASE))
import runtime_common as rc
if os.environ.get('AUDIT_OFFICIAL_SOURCE'):rc.OFFICIAL=Path(os.environ['AUDIT_OFFICIAL_SOURCE'])
rc.configure()
from execution_state import restore_model_counters,tree_sha
from nerf_triplane.network import NeRFNetwork
from nerf_triplane.utils import Trainer

def load(p):
    if torch.__version__.startswith('1.12.'):return torch.load(p,map_location='cpu')
    return torch.load(p,map_location='cpu',weights_only=False)
def brief(x):
    y=x.detach();finite=torch.isfinite(y)
    return dict(dtype=str(y.dtype),shape=list(y.shape),finite=bool(finite.all()),nan=int(torch.isnan(y).sum()),inf=int(torch.isinf(y).sum()),max=float(y[finite].max()) if bool(finite.any()) else None,min=float(y[finite].min()) if bool(finite.any()) else None)
def cuda(x):
    if isinstance(x,torch.Tensor):return x.cuda()
    if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
    if isinstance(x,list):return [cuda(v) for v in x]
    if isinstance(x,tuple):return tuple(cuda(v) for v in x)
    return x
event=Path(sys.argv[1]);out=Path(sys.argv[2]);assert not out.exists();out.mkdir(parents=True)
state=load(event);opt=rc.options('SherrodBrown','train',6,42,'coarse');opt.workspace=str(out)
model=NeRFNetwork(opt).cuda();model.load_state_dict(state['model'],strict=True);restore_model_counters(model,state['model_counters']);model.train()
trainer=Trainer.__new__(Trainer);trainer.model=model;trainer.opt=opt;trainer.global_step=state['global_step'];trainer.flip_finetune_lips=False;trainer.criterion=torch.nn.MSELoss(reduction='none')
rows=[];latest={};original_density=model.density
def density(*a,**k):
    v=original_density(*a,**k)
    latest.setdefault('density',[]).append({n:brief(x) for n,x in v.items() if torch.is_tensor(x)})
    return v
model.density=density
def sigma_hook(module,args,value):latest.setdefault('sigma_logits',[]).append(brief(value[...,0]))
handle=model.sigma_net.register_forward_hook(sigma_hook)
before=rc.tensor_state_sha(model.state_dict());batch=state['batch'];input_sha=tree_sha(batch)
for scale in [float(state['scaler']['scale']),1.0]:
    model.load_state_dict(state['model'],strict=True);restore_model_counters(model,state['model_counters']);model.zero_grad();latest.clear()
    rc.seed_domain(42,trainer.global_step,'render_perturbation_and_regularization')
    row=dict(scale=scale,global_step=trainer.global_step,input_sha256=input_sha)
    try:
        with torch.autograd.detect_anomaly():
            with torch.cuda.amp.autocast(enabled=True):value=Trainer.train_step(trainer,cuda(batch))
            row['loss']=float(value[2]);row['forward']=copy.deepcopy(latest)
            (value[2]*scale).backward()
        row['backward']='completed';row['bad_gradients']=[n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    except Exception as e:
        row.update(error=repr(e),traceback=traceback.format_exc(),forward=copy.deepcopy(latest))
    rows.append(row);print(json.dumps(row),flush=True)
handle.remove();model.load_state_dict(state['model']);restore_model_counters(model,state['model_counters'])
assert rc.tensor_state_sha(model.state_dict())==before
result=dict(scope='diagnostic_only_not_original_failure_state',torch=torch.__version__,cuda=torch.version.cuda,fp16=True,optimizer_updates=0,scaler_updates=0,model_restored=True,event_sha256=hashlib.sha256(event.read_bytes()).hexdigest(),official_source=str(rc.OFFICIAL),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=rows)
(out/'result.json').write_text(json.dumps(result,indent=2));print('SAVED',out,flush=True)
