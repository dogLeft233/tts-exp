import importlib.util,itertools,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/analyze_factorial_gains.py'
spec=importlib.util.spec_from_file_location('factorial',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class FactorialControls(unittest.TestCase):
    def test_scorer_interaction_is_retained_in_exact_decomposition(self):
        cells={}
        for t,d,e in itertools.product([-6,0,6],repeat=3):
            cells[t,d,e]={'guard20':{'sync_c':t+2*d+3*e+d*e}}
        r=mod.decomposition(cells,0,6,'guard20','sync_c')
        self.assertEqual(r['total'],66);self.assertEqual(r['driver_at_original_score'],12)
        self.assertEqual(r['scorer_on_driven_video'],54);self.assertEqual(r['scorer_on_baseline_video'],18);self.assertEqual(r['interaction'],36)
    def test_seeds_average_inside_each_identity(self):
        per_seed={}
        for i,identity in enumerate(mod.IDS):
            for s,seed in enumerate(mod.SEEDS):
                per_seed[identity,seed]={(0,0,0):{'guard20':{m:0. for m in mod.METRICS}},(6,0,0):{'guard20':{m:i+2*s for m in mod.METRICS}}}
        r=mod.paired_contrast(per_seed,(6,0,0),(0,0,0),'guard20')
        self.assertEqual(r['equal_identity']['sync_c']['mean'],2.5);self.assertEqual(r['equal_identity']['sync_c']['identities'],4)
    def test_plateau_screen_handles_zero_and_continued_improvement(self):
        self.assertTrue(mod.plateau([1.,1.,1.],[.99,.99,.99])['plateau_screen_passed'])
        self.assertFalse(mod.plateau([1.,1.,1.],[.9,.9,.9])['plateau_screen_passed'])
        self.assertTrue(mod.plateau([0.,0.,0.],[0.,0.,0.])['plateau_screen_passed'])
        self.assertFalse(mod.plateau([0.,0.,0.],[.01,.01,.01])['plateau_screen_passed'])
if __name__=='__main__':unittest.main()
