import importlib.util,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('four_noise',Path(__file__).resolve().parents[2]/'scripts/experiments/may_four_noise_20261020/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(8);cls.waves=[rng.normal(size=64000) for _ in range(4)];cls.psd=np.linspace(1,.1,513);cls.psd[0]=0
    def test_all_generators_deterministic_finite_and_unit_rms(self):
        for kind in m.KINDS:
            x,a=m.generate_noise(kind,32000,42,self.waves,self.psd,0);y,b=m.generate_noise(kind,32000,42,self.waves,self.psd,0)
            np.testing.assert_array_equal(x,y);self.assertEqual(a,b);self.assertEqual(x.dtype,np.float32);self.assertTrue(np.isfinite(x).all());self.assertAlmostEqual(float(np.sqrt(np.mean(x.astype(float)**2))),1.,places=7)
    def test_babble_uses_four_different_sources(self):
        _,meta=m.generate_noise('babble4',32000,42,self.waves,self.psd,0)
        self.assertEqual(len(set(x['speaker'] for x in meta['components'])),4)
    def test_clicks_remain_sparse_and_peak_reported(self):
        x,_=m.generate_noise('clicks',32000,42,self.waves,self.psd,0);r=m.noise_stats(np.ones_like(x),x)
        self.assertLess(r['active_fraction'],.1);self.assertGreater(r['crest_factor'],5);self.assertGreater(r['active_fraction'],0)
    def test_modulation_independent_and_nonzero_envelope(self):
        _,v=m.generate_noise('modulated_ssn',32000,42,self.waves,self.psd,0)
        self.assertGreaterEqual(v['envelope_min'],.15);self.assertLessEqual(v['envelope_max'],1);self.assertTrue(2<=v['envelope_frequencies_hz'][0]<=5)
    def test_scale_pair_preserves_added_change(self):
        n,_=m.generate_noise('clicks',32000,42,self.waves,self.psd,0);s=self.waves[0][:32000].astype(np.float32)
        r=m.base.waveform_audit(s,n*.01);self.assertEqual(r['input_changed_fraction'],r['half_changed_fraction']);self.assertTrue(r['exact_power_two_difference'])
    def test_streamed_hooks_preserve_values_moments_and_hash_across_chunks(self):
        recovery_path=Path(m.__file__).with_name('recover_streaming.py');sp=importlib.util.spec_from_file_location('streamed_recovery',recovery_path);recovery=importlib.util.module_from_spec(sp);sp.loader.exec_module(recovery)
        for dtype in [np.float32,np.float64]:
            rng=np.random.default_rng(83);x=rng.normal(size=(1025,17)).astype(dtype);y=x+rng.normal(size=x.shape).astype(dtype)*.01
            cap=object.__new__(recovery.StreamCapture);cap.m=m;cap.np=np;cap.refs={'A':{'first_gn':x}};cap.reset('A')
            tensor=m.torch.from_numpy(y.T.copy()).transpose(0,1);cap.consume('first_gn',tensor)
            np.testing.assert_array_equal(cap.frames['first_gn'],m.base.metric_frames(x,y));self.assertEqual(cap.digests['first_gn'],m.base.digest(y))
            np.testing.assert_array_equal(cap.samples['first_gn'],y[cap.samples['first_gn__indices']])

if __name__=='__main__':unittest.main()
