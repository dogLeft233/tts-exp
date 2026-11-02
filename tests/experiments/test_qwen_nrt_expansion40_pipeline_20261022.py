import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('expanded_pipeline',Path(__file__).parents[2]/'scripts/experiments/qwen_nrt_expansion40_20261022/pipeline.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def test_reliability_is_metadata_only_and_bounds_are_inclusive():
    samples=[dict(sample_id='a',duration_s=4)]
    rows=[dict(sample_id='a',seed_index=i,seed=i,samples=int(duration*24000),sample_rate_hz=24000,eos_in_talker_sequences=eos)
          for i,(duration,eos) in enumerate([(2,True),(8,True),(8,False)],1)]
    got=M.reliability(rows,samples)
    assert [got['a',f'T{i}']['qualified'] for i in range(1,4)]==[True,True,False]


def synthetic(sid,values,qualified):
    return dict(sample_id=sid,policies={policy:{role:{metric:value for metric in M.METRICS} for role,value in zip(M.ROLES,values)} for policy in M.POLICIES},
                draw_reliability={role:dict(qualified=ok) for role,ok in zip(M.ROLES[2:],qualified)})


def test_all_draw_primary_and_conditional_coverage_are_separate():
    rows=[synthetic('a',[1,2,3,5,10],[True,False,False]),synthetic('b',[2,3,7,8,9],[False,False,False])]
    result=M.grouped_summary(rows)
    main=result['summaries']['common']['C']
    assert main['contrasts']['T-N']['mean']==pytest.approx(5.5)
    assert main['contrasts']['T-N']['n_speakers']==2
    assert main['per_seed']['T1']['T-N']['n_speakers']==2
    assert result['conditional_coverage']==1 and result['conditional_missing']==['b']
    conditional=result['conditional_generation']['common']['C']['contrasts']
    assert conditional['T-N']['mean']==2 and conditional['T-R']['mean']==1


def test_no_qualified_draw_is_not_imputed_as_zero_effect():
    result=M.grouped_summary([synthetic('a',[1,2,3,5,10],[False,False,False])])
    assert result['conditional_coverage']==0
    assert result['conditional_generation']['common']['C']['n_speakers']==0
    assert result['summaries']['common']['C']['contrasts']['T-N']['mean']==5
