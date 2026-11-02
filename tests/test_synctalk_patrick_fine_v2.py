"""Reject parent mismatches and lost-job replay before fine optimizer work."""
import importlib.util,json,unittest
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]/'scripts/experiments/synctalk_multispeaker_full_20261017/patrick_fine_contract_v2.py'
spec=importlib.util.spec_from_file_location('fine_contract',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FineContractTests(unittest.TestCase):
    def test_lost_job_cannot_be_relaunched(self):
        with self.assertRaisesRegex(AssertionError,'lost'):m.route(False,'lost')
        self.assertEqual(m.route(False,'running'),'observe_existing')
        self.assertEqual(m.route(True,'completed'),'reuse_verified')
    def test_start_preserves_unequal_parent_and_rejects_other_gain(self):
        parent='/tmp/parent';cp=parent+'/training/PatrickLeahy/seed42/gain_+0/coarse/ep0010/cp.pth'
        r=dict(identity='PatrickLeahy',seed=42,gain_db=0,passed=True,phase='coarse',epoch=10,global_step=60000,cumulative_successful_updates=59985)
        c=dict(version=m.VERSION,descriptive_only=True,parent_control_failed_retained=True,parent_run=parent,parent_updates_by_gain={'0':59985,'-6':59985,'6':59984},startpoints={'0':dict(checkpoint_path=cp,receipt_path='/r',checkpoint_sha256='abc',receipt_sha256='def')})
        self.assertTrue(m.validate_start(c,0,cp,'/r',r,'abc','def'))
        with self.assertRaises(AssertionError):m.validate_start(c,0,cp,'/r',dict(r,gain_db=6),'abc','def')
        with self.assertRaises(AssertionError):m.validate_start(c,0,cp,'/r',r,'changed','def')
    def test_ordinary_amp_skips_are_counted_without_faking_updates(self):
        cases={}
        for g in [-6,0,6]:
            rows=[dict(epoch=11,global_step=60001+i,frame_index=i,source_audio_id=50+i,lip_mode=i==0,pixel_index_sha256='same',actual_rays=4096 if i==0 else 65536,rect=None,successful_update=False,amp_skip=True) for i in range(2)]
            cases[g]=dict(passed=True,gain_db=g,contract_sha256='pin',qualification_attempts=2,qualification_optimizer_updates=0,qualification_amp_skips=2,formal_optimizer_updates=0,full_checkpoint_roundtrip_exact=True,counters_roundtrip_exact=True,first_lip_then_nonlip=True,original_parent_unchanged=True,fp16=True,patch_size=64,rows=rows)
        result=m.qualify_cases('pin',cases)
        self.assertTrue(result['passed']);self.assertEqual(result['qualification_optimizer_updates'],0);self.assertEqual(result['qualification_amp_skips'],6)
        cases[0]['qualification_optimizer_updates']=1
        with self.assertRaises(AssertionError):m.qualify_cases('pin',cases)

    def test_wrong_phase_cannot_pass_as_parent(self):
        parent='/tmp/parent';cp=parent+'/training/PatrickLeahy/seed42/gain_+0/cp.pth'
        c=dict(version=m.VERSION,descriptive_only=True,parent_control_failed_retained=True,parent_run=parent,parent_updates_by_gain={'0':59985},startpoints={'0':dict(checkpoint_path=cp,receipt_path='/r',checkpoint_sha256='a',receipt_sha256='b')})
        r=dict(identity='PatrickLeahy',seed=42,gain_db=0,passed=True,phase='fine',epoch=10,global_step=60000,cumulative_successful_updates=59985)
        with self.assertRaises(AssertionError):m.validate_start(c,0,cp,'/r',r,'a','b')

if __name__=='__main__':unittest.main()
