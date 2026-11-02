"""Check cluster estimands, missing-pair rules and strict cache protocol compatibility."""
import sys,importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]/'scripts/experiments/pctalk_lips_full104_20261031'
def load(name):
    spec=importlib.util.spec_from_file_location('full_'+name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(ROOT));spec.loader.exec_module(m);return m

def fixture():
    rows=[];pairs=[]
    for speaker,count,delta in [('A',1,0),('B',5,1)]:
        for i in range(count):
            source=speaker+str(i)
            for arm in ['N','T']:rows.append(dict(source=source,speaker=speaker,arm=arm,paired_available=True,strict_quality=True,pilot_seen=i==0))
            pairs.append(dict(source=source,speaker=speaker,N=0.,T=float(delta),delta=float(delta)))
    return rows,pairs

def test_repeated_sentences_do_not_increase_speaker_weight():
    a=load('analyze');rows,pairs=fixture();result,sp=a.summarize(rows,pairs,'all_pairs')
    assert result['n_speakers']==2 and result['n_pairs']==6
    assert result['delta']==.5
    assert result['delta']!=pytest.approx(5/6)

def test_one_missing_utterance_excludes_the_whole_speaker():
    a=load('analyze');rows,pairs=fixture();pairs=pairs[:-1]
    sp,excluded,intended=a.speaker_pairs(rows,pairs,'all_pairs')
    assert [r['speaker'] for r in sp]==['A']
    assert excluded==[dict(speaker='B',missing_sources=['B4'])]
    assert len(intended['B'])==5

def test_new_only_uses_metadata_not_scores_and_omits_unpaired():
    a=load('analyze');rows,pairs=fixture()
    rows.append(dict(source='missing',speaker='C',arm='N',paired_available=False,strict_quality=False,pilot_seen=False))
    sp,excluded,intended=a.speaker_pairs(rows,pairs,'new_only_all')
    assert not excluded and list(intended)==['B'] and sp[0]['n_pairs']==4

def test_cogenav_reuse_allows_cohort_only_changes():
    r=load('reuse_protocol');old=dict(sources=21,input_manifest_sha256='old',weight='pinned',fps=25,audio='native',video='native')
    assert r.compatible(old,dict(old,sources=105,input_manifest_sha256='new'))
    for key,value in [('weight','changed'),('fps',30),('audio','normalized-differently'),('video','different-crop')]:
        assert not r.compatible(old,dict(old,**{key:value}))
