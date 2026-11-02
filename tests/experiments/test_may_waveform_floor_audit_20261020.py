"""Controls reject zero-count and AVE-floor conflation, plus time-axis drift."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np

SRC=Path(__file__).resolve().parents[2]/'scripts/experiments/may_waveform_floor_audit_20261020/audit.py'
spec=importlib.util.spec_from_file_location('may_floor_audit',SRC)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditControls(unittest.TestCase):
    def test_existing_zeros_are_not_new_zeros(self):
        x=np.array([0.,.00001,-.00001,.2])
        y=np.array([0.,0.,-.000005,.1])
        row=audit.waveform_stats(x,y,.5)
        self.assertEqual(row['original_zero_points'],1)
        self.assertEqual(row['resulting_zero_points'],2)
        self.assertEqual(row['new_zero_points'],1)

    def test_float_half_double_is_exact_for_small_normal_samples(self):
        x=np.array([0,1e-8,-1e-8,.3],dtype=np.float32)
        row=audit.waveform_stats(x,x*np.float32(.5),.5)
        self.assertEqual(row['new_zero_points'],0)
        self.assertTrue(row['restored_exact'])

    def test_log_floor_and_final_clip_are_different(self):
        x=np.array([[3e-5,1.5e-4,1e-3]])
        y=x*.5
        mapped=lambda z:np.clip(.08*(20*np.log10(np.maximum(z,1e-5))-20)+4,-4,4)
        masks=audit.clip_masks(x,y,mapped(x),mapped(y),1e-5)
        self.assertEqual(int(masks['newly_log_floored'].sum()),0)
        self.assertEqual(int(masks['newly_lower_clipped'].sum()),1)

    def test_changed_time_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'time axis'):
            audit.waveform_stats(np.ones(3),np.ones(2),.5)


if __name__=='__main__':unittest.main()
