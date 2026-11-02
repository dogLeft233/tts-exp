import importlib.util,itertools,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/qualify_aggregate_delivery.py'
spec=importlib.util.spec_from_file_location('aggregate_qa',p);qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
class AggregateControls(unittest.TestCase):
    def test_clip_reconstruction_and_tampered_mean_rejected(self):
        rows=[]
        for gains in qa.CELLS:
            clips=[]
            for i in range(6):
                v={k:float(i+1) for k in qa.METRICS};v['av_offset']=i-3
                clips.append(dict(clip_id=f'test_{i:02d}',guard20=v,full=v))
            mean={k:3.5 for k in qa.METRICS};mean['offsets']=list(range(-3,3))
            rows.append(dict(zip(qa.AXES,gains),clips=clips,mean_guard20=mean.copy(),mean_full=mean.copy()))
        raw=dict(passed=True,identity='fixture',seed=42,cells=rows,source_owned_ROI=True,
            evaluation_zero_original_PCM16=True,evaluation_plusminus_float64_original_PCM_units=True,
            repeat=dict(visual_exact=True,matrix_exact=True),actual_FP32_MFCC_input_exact=True)
        result=qa.seed_rows(raw,'fixture',42);self.assertEqual(result[0]['guard20']['sync_c'],3.5)
        rows[0]['mean_guard20']['sync_c']=3.6
        with self.assertRaises(AssertionError):qa.seed_rows(raw,'fixture',42)
    def test_duplicate_or_missing_factorial_cell_rejected(self):
        rows=[dict(zip(qa.AXES,g)) for g in qa.CELLS];qa.check_cell_order(rows)
        rows[-1]=rows[0]
        with self.assertRaises(AssertionError):qa.check_cell_order(rows)
    def test_identity_means_equal_weight_preserve_offsets(self):
        rows=[dict({k:float(i*10) for k in qa.METRICS},offsets=[i]*12) for i in range(4)]
        result=qa.reduce_rows(rows);self.assertEqual(result['sync_c'],15.);self.assertEqual(len(result['offsets']),48)
        self.assertEqual(result['offset_mean'],1.5)
if __name__=='__main__':unittest.main()
