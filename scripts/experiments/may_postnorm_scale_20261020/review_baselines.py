"""One CPU complete-state audit of imported standard controls; no model construction."""
import json
from pathlib import Path
import torch,numpy as np
from protocol import OUT,sha,write
def main():
    p=json.loads((OUT/'protocol.json').read_text());rows=[]
    for arm in p['imported_arms']:
        root=OUT/'training'/arm['id'];r=json.loads((root/'receipt.json').read_text());b=json.loads((root/'baseline_binding.json').read_text())
        cp=root/r['checkpoint'];assert sha(cp)==r['checkpoint_sha256']==b['parent_checkpoint_sha256']
        s=torch.load(cp,map_location='cpu',weights_only=False)
        assert s['arm']==b['parent_arm'] and s['initial_sha256']==p['parent_initial_sha256'] and s['step']==s['successful_optimizer_updates']==s['attempts']==12000 and s['skips']==0
        for key in ['model','gradient']:
            assert all(torch.isfinite(v).all() for v in s[key].values())
        for o in s['optimizer']['state'].values():
            assert all(torch.isfinite(v).all() for v in o.values() if torch.is_tensor(v));assert float(o['step'])==12000
        assert s['scheduler']['last_epoch']==12000 and int(s['ema']['num_updates'])==12 and all(torch.isfinite(v).all() for v in s['ema']['shadow_params'])
        assert np.isfinite(s['metadata']['mean_density']) and s['precision']=='FP32' and s['scaler'] is None
        rows.append(dict(arm=arm['id'],checkpoint_sha256=sha(cp),fullstate_finite=True,original_state_identity_preserved=True))
    write(OUT/'baseline_state_review.json',dict(passed=True,arms=rows,new_model_constructions=0,new_gpu_calls=0,new_optimizer_calls=0))
if __name__=='__main__':main()
