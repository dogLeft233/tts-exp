import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'scripts/experiments/tts_level_reference_evaluator_bridge_20260927.py'
s=importlib.util.spec_from_file_location('levelbridge',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_delay_and_full_condition_support():
    assert m.CONDS==['D0','D66','S0','S66']
    x=np.random.default_rng(47).normal(size=(100,16)).astype(np.float32);shift=np.zeros_like(x);shift[5:]=x[:-5]
    assert m.base.summarize(m.base.matrix(x,shift),'guard20')['best_lag']==5
    assert m.base.bootstrap([1,1,5],['A','A','B'])['mean']==3
