"""Two-environment final zero-update probes; never changes a training policy."""
import sys,os,json,hashlib,inspect
from pathlib import Path
import torch
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017');sys.path.insert(0,str(BASE))
import runtime_common as rc
if os.environ.get('AUDIT_OFFICIAL_SOURCE'):rc.OFFICIAL=Path(os.environ['AUDIT_OFFICIAL_SOURCE'])
rc.configure()
from nerf_triplane.network import NeRFNetwork
from execution_state import restore_model_counters
def load(p):
 return torch.load(p,map_location='cpu') if torch.__version__.startswith('1.12.') else torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=BASE/'controller_qualification/fp16_adapter_environment_audit_attempt01'
event=audit/'official_environment_replay/first_nonfinite_occupancy_grid.pth'
inputs=audit/'official_environment_replay/first_nonfinite_density_inputs.pth'
out=Path(sys.argv[1]);assert not out.exists();out.mkdir()
state=load(event);data=load(inputs);opt=rc.options('SherrodBrown','train',6,42,'coarse')
model=NeRFNetwork(opt).cuda();model.load_state_dict(state['model']);restore_model_counters(model,state['model_counters']);model.train()
before=rc.tensor_state_sha(model.state_dict());record={}
def hook(m,a,v):record['logits']=v[...,0].detach().cpu()
h=model.sigma_net.register_forward_hook(hook)
with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):
 result=model.density(data['x'].cuda(),data['enc_a'].cuda(),data['eye'].cuda() if data['eye'] is not None else None)
h.remove();assert rc.tensor_state_sha(model.state_dict())==before
sigma=result['sigma'].detach().cpu()
# A scalar library-kernel fixture, separate from model parameters and optimizer.
# Mirrors GradScaler's reciprocal calculation and its actual CUDA unscale kernel.
fixtures=[]
for scale in [5.877471754111438e-39,2.938735877055719e-39]:
 st=torch.tensor(scale,device='cuda',dtype=torch.float32)
 inv=st.double().reciprocal().float()
 grads=torch.tensor([0.,scale,-scale],device='cuda',dtype=torch.float32)
 found=torch.zeros((),device='cuda',dtype=torch.float32)
 torch._amp_foreach_non_finite_check_and_unscale_([grads],found,inv)
 fixtures.append(dict(scale=scale,inverse=float(inv),unscaled=grads.cpu().tolist(),found_inf=float(found)))
report=dict(torch=torch.__version__,cuda=torch.version.cuda,fp16_autocast=True,
 scope='exact captured density point subset and isolated scalar unscale fixture; no optimizer/scaler updates',
 optimizer_updates=0,scaler_updates=0,input_sha256=sha(inputs),event_sha256=sha(event),source_sha256=sha(Path(__file__)),
 global_step=data['global_step'],input_points=len(data['x']),logit_dtype=str(record['logits'].dtype),sigma_dtype=str(sigma.dtype),
 logits=record['logits'].tolist(),sigma=sigma.tolist(),captured_sigma=data['sigma'].tolist(),
 input_finite={k:bool(torch.isfinite(v).all()) for k,v in data.items() if torch.is_tensor(v) and k!='sigma'},
 sigma_inf=int(torch.isinf(sigma).sum()),sigma_nan=int(torch.isnan(sigma).sum()),model_unchanged=True,
 unscale_fixtures=fixtures,gradscaler_source_sha256=sha(Path(inspect.getfile(torch.cuda.amp.GradScaler))))
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
