"""CoGenAV adapter control semantics and frozen legacy-source compatibility."""
import importlib.util
from pathlib import Path
import sys
import pytest


def load():
    folder=Path(__file__).parents[1]/'scripts/experiments'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('ears_cogenav',folder/'ears40_cogenav_scorer_20261029.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_controls_use_natural_pairs_and_strict_score_decrease():
    m=load()
    cs=[dict(shift_samples=s,eq1=.5,pass_within_one_frame=True) for s in [-6400,6400]]
    scores=[dict(id='p001_N_0',arm='N',eq1=.6,raw_shift_controls=cs),
            dict(id='p001_T_0',arm='T',eq1=.7,raw_shift_controls=[])]
    result=m.controls_from_scores(scores)
    assert result['total']==result['lower']==result['peak_within_one_frame']==2
    cs[0]['eq1']=.6
    assert m.controls_from_scores(scores)['status']=='CONTROL_FAILED'
    scores[1]['raw_shift_controls']=cs
    with pytest.raises(ValueError,match='no TTS'):m.controls_from_scores(scores)


def test_current_source_is_compatible_and_fails_closed_if_changed():
    m=load();original=Path(m.__file__).with_name('cogenav_sync_audit_20261024.py').read_text()
    transformed,receipt=m.transformed_source(original)
    compile(transformed,'<frozen CoGenAV>','exec')
    assert len(receipt)==7
    with pytest.raises(ValueError,match='source anchor'):
        m.transformed_source(original.replace('for shift in [-4, 4]:','for shift in [0]:'))
