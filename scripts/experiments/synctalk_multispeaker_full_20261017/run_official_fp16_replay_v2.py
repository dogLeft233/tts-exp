"""Torch 1.12 API bridge, retaining original numerical code and matched data adapter."""
import argparse,importlib.util,os,sys
from pathlib import Path
import torch
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
sys.path.insert(0,str(BASE))
assert torch.__version__=='1.12.1+cu113'
import runtime_common as rc
rc.OFFICIAL=Path(os.environ['AUDIT_OFFICIAL_SOURCE']);rc.configure()
# Environment is the comparison factor; the verified cache/split/context adapter is held fixed.
import nerf_triplane
spec=importlib.util.spec_from_file_location('nerf_triplane.provider',BASE/'official/nerf_triplane/provider.py')
provider=importlib.util.module_from_spec(spec);sys.modules[spec.name]=provider;spec.loader.exec_module(provider);nerf_triplane.provider=provider
import execution_state
class LegacyUpdateCounter:
    def __init__(self,optimizer,initial=0):
        self.successful=int(initial);self.optimizer=optimizer;self.original=optimizer.step
        def step(*a,**k):
            result=self.original(*a,**k);self.successful+=1;return result
        optimizer.step=step
    def close(self):self.optimizer.step=self.original
execution_state.UpdateCounter=LegacyUpdateCounter
original_load=torch.load
def load(*a,**k):
    assert k.get('weights_only',False) is False
    k.pop('weights_only',None)
    return original_load(*a,**k)
torch.load=load
import diagnose_fp16_failure_v3 as d
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--validate-only',action='store_true');p.add_argument('--steps',type=int,default=6000);a=p.parse_args()
assert 1<=a.steps<=6000
if a.validate_only:
    def validate(trainer,loader):
        assert trainer.fp16 and trainer.global_step==24000 and trainer.update_counter.successful==23996
        assert len(loader._data.poses)==6000
        print('FULL STATE RESTORE VERIFIED; NO FORWARD/BACKWARD/UPDATES',flush=True)
    d.ControlledTrainer.train_one_epoch=validate
d.run(a.out,a.steps,'controlled')
