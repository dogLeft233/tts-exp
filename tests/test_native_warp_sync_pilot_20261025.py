import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('nw',Path(__file__).resolve().parents[1]/'scripts/experiments/native_warp_sync_pilot_20261025.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_quantization_identity_and_clipping_guard():
 x=np.array([-32767,-3,0,1,32767],dtype=np.int16)
 assert np.array_equal(m.quantize(x.astype(float)/32768),x)
 try:m.quantize(np.array([1.0]));assert False
 except ValueError:pass
def test_source_equal_pairs_not_windows():
 rows=[dict(sample_id=s,role=r,C=i+j) for i,s in enumerate(['a','b']) for j,r in enumerate(m.ROLES)]
 q=m.paired_summary(rows)
 assert q['deltas']['T']['mean']==4 and q['deltas']['T']['positive']==2
 assert q['means']['N']==.5
