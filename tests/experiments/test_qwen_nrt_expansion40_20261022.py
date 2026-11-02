import importlib.util
from pathlib import Path
import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[2]/'scripts/experiments/qwen_nrt_expansion40_20261022/generation_stage.py'
spec=importlib.util.spec_from_file_location('generation40',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def manifest():
    return {'samples':[{'sample_id':f'utt{i}','speaker_id':f'S{i:04d}','transcript':' 测 试\n'} for i in range(2,42)],
            'seed_bases':module.BASES.copy(),'generation_kwargs':module.KWARGS.copy(),'non_streaming_mode':False}


def test_fixed_90_new_draws_and_no_old_generation():
    jobs=module.validate_manifest(manifest())
    assert len(jobs)==90 and jobs[0]['speaker_id']=='S0012' and jobs[-1]['speaker_id']=='S0041'
    assert jobs[0]['seed']==2026103401 and jobs[-1]['seed']==2026106303
    assert jobs[0]['target_text']==jobs[0]['reference_text']=='测试'
    assert [r['seed_index'] for r in jobs[:3]]==[1,2,3]


def test_changed_cohort_rejected():
    m=manifest();m['samples'][-1]['speaker_id']='S0012'
    with pytest.raises(ValueError):module.validate_manifest(m)


def test_changed_settings_rejected():
    m=manifest();m['generation_kwargs']['max_new_tokens']=1024
    with pytest.raises(ValueError):module.validate_manifest(m)


def test_unsorted_cohort_rejected():
    m=manifest();m['samples'].reverse()
    with pytest.raises(ValueError):module.validate_manifest(m)


def test_exact_bridge_does_not_allow_approximation():
    module.require_exact(np.zeros(3,dtype=np.float32),np.zeros(3,dtype=np.float32),'exact')
    with pytest.raises(RuntimeError):module.require_exact(np.zeros(3),np.array([0,0,1e-15]),'different')


def test_disk_budget_preserves_existing_and_prevents_overshoot(tmp_path):
    b=module.Budget(tmp_path,cap=20);b.json(tmp_path/'a.json',{'a':1})
    before=(tmp_path/'a.json').read_bytes()
    with pytest.raises(RuntimeError):b.json(tmp_path/'large.json',{'b':'x'*30})
    assert not (tmp_path/'large.json').exists() and (tmp_path/'a.json').read_bytes()==before


def test_helper_is_sha_bound_and_official_cut_keeps_rounding():
    h,path=module.load_helpers()
    assert module.sha(path)==module.PILOT_SHA
    assert h.official_cut(1,3,10)==3
    with pytest.raises(ValueError):h.waveform(np.array([np.nan],dtype=np.float32),24000)
    with pytest.raises(ValueError):h.waveform(np.ones(3,dtype=np.float32),16000)
