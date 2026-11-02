import importlib.util
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

folder=Path(__file__).parents[1]/'scripts/experiments/cogenav_decoder_pilot_20261025'
sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location('decoder_cli',folder/'decode_features.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_raw_features_npy_and_pt_require_correct_finite_dimensions():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);good=np.ones((3,768),dtype=np.float32)
        np.save(root/'good.npy',good)
        torch.save(dict(cogen=torch.from_numpy(good)),root/'good.pt')
        assert torch.equal(module.load_features(root/'good.npy'),module.load_features(root/'good.pt'))
        for data in (np.ones((3,767),dtype=np.float32),np.ones((0,768),dtype=np.float32),np.full((3,768),np.nan,dtype=np.float32),np.full((3,768),1e300,dtype=np.float64)):
            np.save(root/'bad.npy',data)
            try:module.load_features(root/'bad.npy')
            except ValueError:pass
            else:raise AssertionError('Invalid features admitted')


def test_duration_never_stretches_features_beyond_native_support():
    assert module.validate_duration(3,None)==1920
    assert module.validate_duration(3,1701)==1701
    for samples in (0,-1,1921):
        try:module.validate_duration(3,samples)
        except ValueError:pass
        else:raise AssertionError('Unsupported duration admitted')
