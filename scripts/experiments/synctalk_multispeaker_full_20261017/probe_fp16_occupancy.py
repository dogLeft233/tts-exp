"""Original density function on frozen occupancy inputs, no parameter/scaler updates."""
import sys,os,json,hashlib
from pathlib import Path
import torch
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017');sys.path.insert(0,str(BASE))
import runtime_common as rc
if os.environ.get('AUDIT_OFFICIAL_SOURCE'):rc.OFFICIAL=Path(os.environ['AUDIT_OFFICIAL_SOURCE'])
rc.configure()
from nerf_triplane.network import NeRFNetwork
from execution_state import restore_model_counters,tree_sha
def load(p):
    return torch.load(p,map_location='cpu') if torch.__version__.startswith('1.12.') else torch.load(p,map_location='cpu',weights_only=False)
audit=BASE/'controller_qualification/fp16_adapter_environment_audit_attempt01';event=audit/'current_environment_replay_v2/first_scale_below_one.pth';inputs=audit/'frozen_occupancy_probe_inputs.pth'
out=Path(sys.argv[1]);assert not out.exists();out.mkdir()
state=load(event);data=load(inputs);opt=rc.options('SherrodBrown','train',6,42,'coarse');model=NeRFNetwork(opt).cuda();model.load_state_dict(state['model']);restore_model_counters(model,state['model_counters']);model.train()
before=rc.tensor_state_sha(model.state_dict());record={}
def hook(m,a,v):record['logits']=v[...,0].detach().cpu()
h=model.sigma_net.register_forward_hook(hook)
with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):
    encoded=model.encode_audio(data['auds'].cuda());result=model.density(data['xyzs'].cuda(),encoded,data['eye'].cuda())
h.remove();assert rc.tensor_state_sha(model.state_dict())==before
sigma=result['sigma'].detach().cpu()
report=dict(torch=torch.__version__,cuda=torch.version.cuda,fp16_autocast=True,scope='zero-update density probe, not original failure replay',optimizer_updates=0,scaler_updates=0,input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),event_sha256=hashlib.sha256(event.read_bytes()).hexdigest(),frame_index=data['frame_index'],source_audio_id=data['source_audio_id'],logit_dtype=str(record['logits'].dtype),sigma_dtype=str(sigma.dtype),logits=record['logits'].tolist(),sigma=sigma.tolist(),sigma_inf=int(torch.isinf(sigma).sum()),sigma_nan=int(torch.isnan(sigma).sum()),model_unchanged=True)
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
