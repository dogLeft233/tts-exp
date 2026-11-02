"""CPU-only inspection of existing arrays/checkpoints; no model forward/update."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017');sys.path.insert(0,str(BASE))
import runtime_common as rc
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=Path(sys.argv[1]);assert not out.exists();out.mkdir()
opt=rc.options('SherrodBrown','train',6,42,'coarse');assert opt.bs_area=='upper' and not opt.au45 and not opt.smooth_eye
directory=Path(opt.path);bs=np.load(directory/'bs.npy');upper=np.hstack((bs[:,0:5],bs[:,8:10])).astype(np.float32)
frames=json.loads((directory/'transforms_train.json').read_text())['frames']
aud=BASE/'controller_qualification/fp16_adapter_environment_audit_attempt01/official_environment_replay'
receipt=json.loads((BASE/'training/SherrodBrown/seed42/gain_+6/coarse/ep0004/attempt01/epoch_receipt.json').read_text());ep4=load(receipt['checkpoint_path'])
baseline=ep4['model'];rows=[]
for name in ['ep4','first_nonfinite_scaled_gradient','first_nonfinite_occupancy_grid','first_scale_below_one']:
 path=Path(receipt['checkpoint_path']) if name=='ep4' else aud/(name+'.pth');state=ep4 if name=='ep4' else load(path)
 groups={}
 for prefix in ['eye_att_net.','sigma_net.','aud_ch_att_net.','audio_net.']:
  tensors=[(k,v) for k,v in state['model'].items() if k.startswith(prefix)]
  groups[prefix]=dict(l2=float(sum((v.double()**2).sum() for k,v in tensors).sqrt()),max_abs=max(float(v.abs().max()) for k,v in tensors),l2_change_from_ep4=float(sum(((v.double()-baseline[k].double())**2).sum() for k,v in tensors).sqrt()))
 row=dict(state=name,global_step=state['global_step'],file_sha256=sha(path),parameter_groups=groups)
 if 'batch' in state:
  index=int(state['batch']['index'][0]);img_id=int(frames[index]['img_id']);actual=state['batch']['eye'].numpy();expected=upper[img_id][None,:]
  assert np.array_equal(actual,expected)
  row.update(batch_frame_index=index,img_id=img_id,eye=actual.tolist(),eye_exact_to_original_provider_mapping=True,batch_is_event_input=name!='first_nonfinite_occupancy_grid')
 rows.append(row)
point=load(aud/'first_nonfinite_density_inputs.pth');eye=point['eye'].numpy()
matches=np.flatnonzero(np.all(upper==eye.reshape(1,7),axis=1));assert len(matches)>0
gain_pins=[]
for gain in [-6,0,6]:
 path=Path(rc.options('SherrodBrown','train',gain,42,'coarse').path)/'bs.npy';gain_pins.append(dict(gain=gain,path=str(path),sha256=sha(path)))
assert len(set(x['sha256'] for x in gain_pins))==1
report=dict(scope='CPU existing-state and array inspection only; zero model forward/backward/optimizer/scaler calls',source_sha256=sha(__file__),bs_shape=list(bs.shape),bs_sha256=sha(directory/'bs.npy'),upper_min=upper.min(0).tolist(),upper_max=upper.max(0).tolist(),upper_all_finite=bool(np.isfinite(upper).all()),au45=opt.au45,smooth_eye=opt.smooth_eye,bs_area=opt.bs_area,states=rows,exact_first_density_eye=eye.tolist(),matching_bs_rows=matches.tolist(),cross_gain_bs=gain_pins,limitation='Weight growth observations are not gradient attribution; upstream source bs provenance not requalified here')
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
