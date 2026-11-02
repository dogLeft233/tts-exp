import importlib.util,os
from pathlib import Path
import unittest
import torch
SOURCE=Path(os.environ.get('DENSITY_TRANSACTION_SOURCE',str(Path(__file__).parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/density_refresh_transaction_v1.py')))
spec=importlib.util.spec_from_file_location('transaction',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class TransactionChecks(unittest.TestCase):
    def toy(self,bad=False,mean_bad=False):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(3.));self.register_buffer('density_grid',torch.tensor([2.,4.]));self.mean_density=3.;self.iter_density=1
            def update_extra_state(self):
                self.density_grid.add_(1.);self.iter_density+=1;self.mean_density=float('inf') if mean_bad else 4.
                if bad:self.density_grid[0]=float('inf')
        return Model()
    def test_finite_update_accepted_exactly(self):
        m=self.toy();events=[];module.install(m,['mean_density','iter_density'],events.append);m.update_extra_state()
        self.assertEqual(m.density_grid.tolist(),[3.,5.]);self.assertEqual(m.iter_density,2);self.assertTrue(events[0]['accepted'])
    def test_bad_grid_restores_all_cache_and_counters_without_touching_weights(self):
        m=self.toy(bad=True);events=[];module.install(m,['mean_density','iter_density'],events.append);m.update_extra_state()
        self.assertEqual(m.density_grid.tolist(),[2.,4.]);self.assertEqual((m.mean_density,m.iter_density),(3.,1));self.assertEqual(float(m.weight),3.);self.assertTrue(events[0]['cache_rolled_back'])
    def test_bad_scalar_also_rolls_back_finite_grid_update(self):
        m=self.toy(mean_bad=True);events=[];module.install(m,['mean_density','iter_density'],events.append);m.update_extra_state()
        self.assertEqual(m.density_grid.tolist(),[2.,4.]);self.assertIn('mean_density',events[0]['rejected_buffers_or_counters'])
if __name__=='__main__':unittest.main()
