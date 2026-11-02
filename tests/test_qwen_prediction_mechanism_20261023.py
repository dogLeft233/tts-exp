"""Capture boundary tests: recorded distribution precedes forcing; hooks restore."""
from types import SimpleNamespace
import numpy as np
import unittest
import tempfile,json
from pathlib import Path
import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import trace_full,save,persist_trace
from scripts.experiments.qwen_prediction_mechanism_20261023.analyze_bank import average,entropy,interval,match_temperature
from scripts.experiments.qwen_prediction_mechanism_20261023.cpu_audit import softmax,effective
from scripts.experiments.qwen_prediction_mechanism_20261023.analyze_confirmation import main as confirmation_analysis
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import sha
from scripts.experiments.qwen_prediction_mechanism_20261023.describe_scores import centered_correlation

def wrapper():
    predictor=SimpleNamespace(generate=lambda: None)
    return SimpleNamespace(model=SimpleNamespace(talker=SimpleNamespace(code_predictor=predictor),config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=3))))

def test_probability_is_before_force_and_native_topk():
    w=wrapper();processors=LogitsProcessorList([TemperatureLogitsWarper(.9),TopKLogitsWarper(2)])
    raw=torch.tensor([[1.,2.,3.,4.]])
    with trace_full(w,np.array([[0]*16])) as cap:
        forced=processors(torch.zeros((1,0),dtype=torch.long),raw.clone())
    row=cap['rows'][0]
    assert np.array_equal(row['raw'],raw[0].numpy())
    assert row['probability'][0]==0 and row['probability'][3]>0
    assert forced.argmax().item()==0
    expected=torch.softmax(torch.tensor([3.,4.])/.9,0).numpy()
    assert np.allclose(row['probability'][2:],expected)

def test_hooks_restore_after_exception():
    w=wrapper();original=LogitsProcessorList.__call__;generate=w.model.talker.code_predictor.generate
    with unittest.TestCase().assertRaises(RuntimeError):
        with trace_full(w): raise RuntimeError('deliberate')
    assert LogitsProcessorList.__call__ is original
    assert w.model.talker.code_predictor.generate is generate

def test_nonfinite_processor_metadata_is_strict_json():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'metadata.json'
        save(p,{'processors':[{'filter_value':-float('inf')}], 'nested':{'positive':float('inf')}})
        got=json.loads(p.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
        assert got['processors'][0]['filter_value']=='-inf'
        assert got['nested']['positive']=='inf'

def test_source_history_draw_weighting_and_preexisting_qualification():
    rows=[];values={}
    for source in ['A','B']:
        for src,nums in [('N',[0]),('L06',[3,6,999]),('L17',[9,9,9])]:
            for i,num in enumerate(nums):
                key=f'{source}_{src}_{i}'
                rows.append({'id':key,'sample_id':source,'source':src,'qualified':num!=999})
                values[key]=np.array([float(num)])
    assert np.array_equal(average(rows,values),np.array([[4.5],[4.5]]))

def test_entropy_calibration_monotone_and_source_bootstrap_fixed():
    logits=np.array([[0.,1.,2.]])
    assert entropy(softmax(logits/.4))[0]<entropy(softmax(logits/1.6))[0]
    assert interval(np.arange(10))==interval(np.arange(10))

def test_CPU_special_EOS_mask_and_repetition():
    raw=np.zeros(3072,dtype=np.float32);raw[2150]=20;raw[7]=10
    first=effective(raw,[],0);later=effective(raw,[7],0)
    assert not np.isfinite(first[2150]) and not np.isfinite(later[2150])
    after_min=effective(raw,[7,8],0)
    assert np.isfinite(after_min[2150])
    assert np.isclose(after_min[7],np.float32(10)/np.float32(1.05)/np.float32(.9))

def test_confirmation_recovers_known_rescue_and_gap_shrink_without_pseudoreplication():
    with tempfile.TemporaryDirectory() as tmp:
        run=Path(tmp);rows=[];reuse=[];samples=[]
        for speaker in range(10):
            sid=f'S{speaker:04d}';samples.append({'sample_id':sid})
            for model in ['L06','L17']:
                for policy in ['baseline','coarse','fine','both']:
                    for draw in range(1,4):
                        key=f'{sid}_{model}_{policy}_{draw}'
                        value=speaker+draw*.1+(.6 if model=='L17' else 0)
                        if policy=='both':value+=.3 if model=='L06' else -.05
                        f=run/(key+'.json');save(f,{'status':'COMPLETE','canonical_audio_sha256':'known','sync_c':value})
                        rows.append({'id':key,'duplicate_of':None,'model':model,'policy':policy,'sample_id':sid,'draw':draw,'reuse_L06_baseline':False})
                        reuse.append({'id':key,'score_path':str(f),'score_sha':sha(f),'canonical_audio_sha':'known'})
        save(run/'C_protocol.json',{'samples':samples,'rows':rows});save(run/'C_downstream_plan.json',{'reuse':reuse,'jobs':[]})
        confirmation_analysis(run);result=json.loads((run/'C_analysis.json').read_text());c=result['primary_and_gap_contrasts']
        assert c['P1_L06_both_minus_baseline']['n_sources']==10
        assert np.isclose(c['P1_L06_both_minus_baseline']['mean'],.3)
        assert np.isclose(c['P2_gap_shrink']['mean'],.35)
        assert result['full10_support'] and all(result['numeric_support_gates'].values())
        assert result['status']=='NUMERIC_ANALYSIS_REQUIRES_CONTENT_AND_MANIPULATION_REVIEW'

def test_constant_centered_entropy_is_unmeasurable_not_zero_correlation():
    result=centered_correlation(np.ones((10,3)),np.tile([0.,1.,2.],(10,1)))
    assert result['Pearson_r'] is None and result['source_cluster_CI95'] is None
    assert result['undefined_bootstrap_replicates']==20000

def test_bounded_temperature_uses_frozen_error_tolerance_at_endpoints():
    near=match_temperature(lambda t:t,.4-.019)
    far=match_temperature(lambda t:t,.4-.021)
    interior=match_temperature(lambda t:t,1.1)
    assert near['status']=='MATCHED' and near['temperature']==.4
    assert far['status']=='UNMATCHABLE'
    assert interior['status']=='MATCHED' and abs(interior['temperature']-1.1)<1e-8

def test_partial_trace_survives_failure_before_first_fine_code():
    row={'frame':0,'cb':0,'raw':np.array([1.,2.],dtype='float32'),'effective':np.array([1.,2.],dtype='float32'),'probability':np.array([.25,.75],dtype='float32'),'history':np.array([],dtype='int64')}
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'partial.npz';persist_trace(f,{'rows':[row],'processors':{}})
        z=np.load(f);assert z['raw_0'].shape==(1,2) and z['raw_1'].shape==(0,0)
        assert len(json.loads(f.with_suffix('.json').read_text())['rows'])==1

if __name__=='__main__':
    for f in [test_probability_is_before_force_and_native_topk,test_hooks_restore_after_exception,test_nonfinite_processor_metadata_is_strict_json,test_source_history_draw_weighting_and_preexisting_qualification,test_entropy_calibration_monotone_and_source_bootstrap_fixed,test_CPU_special_EOS_mask_and_repetition,test_confirmation_recovers_known_rescue_and_gap_shrink_without_pseudoreplication,test_constant_centered_entropy_is_unmeasurable_not_zero_correlation,test_bounded_temperature_uses_frozen_error_tolerance_at_endpoints,test_partial_trace_survives_failure_before_first_fine_code]:f();print(f.__name__,'PASS')
