import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('phrase_local',Path(__file__).parents[2]/'scripts/experiments/qwen_phrase_local_20261024.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
    return {'array_shape':[100,31],'frames':list(range(25,130)),'crop_source_frames_valid':[True]*105,'crop_audio_samples':67200,'audio_binding':{'crop_zero_canonical_s':.9309375},'fixed_best_column':17},{'canonical_audio_samples':100000}

def test_frame_origin_lag_and_full_RF():
    t,r=fixture();w=m.support(20,t,r)
    assert w['matched_audio_feature_row']==22
    assert w['video_RF']==[1.8,2.0]
    assert abs(w['matched_audio_RF'][0]-(.9309375+.88-1/16000))<1e-10
    assert abs(w['matched_audio_RF'][1]-(.9309375+.88+.215))<1e-10
    assert w['all_31lag_actual_support']

def test_padding_canonical_and_duplicate_tail_rejected():
    t,r=fixture();assert 'lag_feature_padding' in m.support(3,t,r)['invalid_reasons']
    t['crop_source_frames_valid'][22]=False
    assert 'video_duplicate_tail' in m.support(20,t,r)['invalid_reasons']
    t,r=fixture();t['crop_audio_samples']=23000
    assert 'crop_audio_or_mfcc_padding' in m.support(20,t,r)['invalid_reasons']
    t,r=fixture();t['audio_binding']['crop_zero_canonical_s']=-.3
    assert 'outside_canonical_audio' in m.support(20,t,r)['invalid_reasons']

def test_both_RF_and_five_variants_no_direction_rescue():
    w={'video_RF':[1,1.2],'matched_audio_RF':[1.01,1.225]}
    assert not m.inside(w,1,1.22)
    assert m.inside(w,1,1.225)
    assert set(m.VARIANTS.values())=={(0,0),(-.04,-.04),(.04,.04),(.04,-.04),(-.04,.04)}
    assert m.direction([1,1,None])=='INCOMPLETE'
    assert m.direction([1,1,0])=='mixed_or_near_zero'

if __name__=='__main__':
    import json
    tests=[value for name,value in list(globals().items()) if name.startswith('test_')]
    for test in tests:test()
    print(json.dumps({'tests':len(tests),'passed':True}))
