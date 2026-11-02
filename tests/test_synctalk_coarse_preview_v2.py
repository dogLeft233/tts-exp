import importlib.util,sys,unittest
from pathlib import Path
import torch
from torch_ema import ExponentialMovingAverage
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
sys.path[:0]=[str(BASE),str(BASE/'trials/fp16_other_identities_trial_v1/early_score_preview_v2/sources')]
from coarse_preview_v2 import named_ema_state,finite

class TestEMA(unittest.TestCase):
    def test_frozen_parameters_buffers_and_cloned_shadows(self):
        model=torch.nn.Linear(2,1);model.bias.requires_grad_(False)
        model.register_buffer('density_grid',torch.tensor([8.]))
        ema=ExponentialMovingAverage(model.parameters(),decay=.95)
        with torch.no_grad():model.weight.add_(2)
        ema.update()
        state=dict(model={k:v.clone() for k,v in model.state_dict().items()},ema=ema.state_dict())
        mapped=named_ema_state(model,state);ema.store();ema.copy_to()
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,mapped[k]))
        ema.restore()
        self.assertTrue(torch.equal(mapped['density_grid'],state['model']['density_grid']))
        self.assertTrue(torch.equal(mapped['bias'],state['model']['bias']))
        mapped['weight'].zero_();self.assertFalse(torch.equal(mapped['weight'],state['ema']['shadow_params'][0]))
    def test_missing_shadow_rejected(self):
        m=torch.nn.Linear(2,1)
        with self.assertRaises(AssertionError):named_ema_state(m,dict(model=m.state_dict(),ema=dict(shadow_params=[])))
    def test_nested_nonfinite_checkpoint_rejected(self):
        finite(dict(optimizer=[torch.ones(2)],step=60))
        for value in [float('nan'),float('inf')]:
            with self.assertRaises(AssertionError):finite(dict(optimizer=[torch.tensor([value])]))

if __name__=='__main__':unittest.main()
