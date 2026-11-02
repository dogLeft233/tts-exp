import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

BASE=Path(__file__).resolve().parents[2]/'scripts/experiments/nrt_two_arm_current_pipeline_20261023'
def load(name):
    spec=importlib.util.spec_from_file_location('nrt_two_'+name,BASE/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
G=load('generation_stage');D=load('downstream')

def manifest():
    ids=[1,8,15,22,29,35,41,47,53,59,65,71,77,83,89,95]
    speakers=['S0764','S0765','S0766','S0768','S0770','S0901','S0902','S0905','S0906','S0907','S0908','S0912','S0913','S0914','S0915','S0916']
    return dict(samples=[dict(sample_id='historical_'+str(i),original_sample_id=i,speaker_id=s,transcript=' 文 本 ') for i,s in zip(ids,speakers)],seed_bases=G.BASES,generation_kwargs=G.KWARGS,non_streaming_mode=False)

def test_frozen_speaker_seed_draws():
    jobs=G.validate_manifest(manifest());assert len(jobs)==48
    assert jobs[0]['seed']==2026102201+76400
    assert jobs[0]['target_text']=='文本'
    assert [r['seed_index'] for r in jobs[:3]]==[1,2,3]

def test_wrong_original_id_or_duplicate_speaker_rejected():
    m=manifest();m['samples'][0]['original_sample_id']=2
    with pytest.raises(ValueError):G.validate_manifest(m)
    m=manifest();m['samples'][1]['speaker_id']=m['samples'][0]['speaker_id']
    with pytest.raises(ValueError):G.validate_manifest(m)

def test_changed_scientific_settings_rejected():
    m=copy.deepcopy(manifest());m['generation_kwargs']['temperature']=1.
    with pytest.raises(ValueError):G.validate_manifest(m)

def test_float_normalization_preserves_clock_and_gain():
    x=np.tile(np.array([-.1,.1],np.float32),8000)
    y,rms,gain=D.normalize_audio(x,.0376838172675746)
    assert len(y)==len(x) and y.dtype==np.float32
    np.testing.assert_array_equal(y,x*np.float32(gain))
    assert abs(np.sqrt(np.mean(y.astype(float)**2))-.0376838172675746)<1e-8

@pytest.mark.parametrize('x',[np.zeros(100),np.array([np.nan]),np.array([np.inf]),np.ones((2,2)),np.array([])])
def test_invalid_audio_rejected(x):
    with pytest.raises(ValueError):D.normalize_audio(x,.03)

def test_normalization_no_clipping_policy():
    x=np.zeros(16000,np.float32);x[0]=1.
    with pytest.raises(ValueError):D.normalize_audio(x,.0376838172675746)

def test_tracks_preserve_first_even_when_second_is_better():
    tracks=D.parse_tracks('AV offset: 1\nMin dist: 8.000\nConfidence: 1.000\nAV offset: -2\nMin dist: 4.000\nConfidence: 9.000\n')
    assert tracks[0]['sync_c']==1. and tracks[1]['sync_c']==9.
    with pytest.raises(ValueError):D.parse_tracks('AV offset: 1\nMin dist: 8.0\n')

def test_raw_exact_coverage_and_sha(tmp_path):
    p=tmp_path/'audio';p.write_bytes(b'123')
    config=dict(samples=[dict(sample_id='a',speaker_id='S0001')],roles=['C1'],expected_cells=1)
    r=dict(sample_id='a',speaker_id='S0001',role='C1',path=str(p),sha256=D.sha(p))
    assert D.validate_raw([r],config)==[r]
    with pytest.raises(ValueError):D.validate_raw([r,r],config)
    p.write_bytes(b'456')
    with pytest.raises(ValueError):D.validate_raw([r],config)

def test_frame_pts_difference_detectable():
    a=D.frame_clock({'frames':[{'best_effort_timestamp_time':'0.000'},{'best_effort_timestamp_time':'0.040'}]})
    b=D.frame_clock({'frames':[{'best_effort_timestamp_time':'0.001'},{'best_effort_timestamp_time':'0.041'}]})
    assert a!=b
