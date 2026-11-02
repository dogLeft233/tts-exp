import importlib.util
import io
from pathlib import Path
import wave

p=Path(__file__).parents[1]/'scripts/experiments/audio_data_inventory200_20261024.py'
s=importlib.util.spec_from_file_location('inventory200',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_normalized_and_near_duplicate_texts_bind():
    assert m.near_duplicate('Ａ 股 市场今天上涨很多','A股市场今天上涨很多')
    assert m.near_duplicate('房地产市场最近持续快速上涨','房地产市场最近持续迅速上涨')
    assert not m.near_duplicate('楼市上涨','今天天气晴朗')

def test_header_qualification_preserves_rejected_denominator():
    b=io.BytesIO()
    with wave.open(b,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0\0'*32000)
    row=m.source_row(b.getvalue(),'BAC009S0002W0122',{'path':'fixture'},'测试句子','fit')
    assert not row['eligible_header_text']
    assert row['exclusions']==['duration_outside_3_12']
    assert row['duration_s']==2

def test_missing_asset_fails_closed(tmp_path):
    assert not m.asset(tmp_path/'missing.wav','a'*64)['sha_match']
    p=tmp_path/'p';p.write_bytes(b'abc')
    assert not m.asset(p,'a'*64)['sha_match']
    assert m.asset(p,m.digest(p))['sha_match']

def test_provisional_split_excludes_legacy_even_when_physically_readable():
    old=[f'S{i:04d}' for i in range(2,42)]
    packs=[{'speaker_id':f'S{i:04d}','gender':'M' if i<202 else 'F','dataset_split':'train'} for i in range(2,402)]
    excluded={'S0044','S0102','S0300'}
    rows=m.plan_speakers(packs,old,excluded)
    assert len(rows)==200 and len({x['speaker_id'] for x in rows})==200
    assert not {x['speaker_id'] for x in rows}&excluded
    assert {x['speaker_id'] for x in rows if x['selection']=='explicit_required_old40'}==set(old)
    assert [sum(x['split']==s for x in rows) for s in ['fit','development','confirmation']]==[100,40,60]
    assert rows==m.plan_speakers(packs,old,excluded)

def test_gender_goal_does_not_override_admission_exclusion():
    old=[f'S{i:04d}' for i in range(2,42)]
    packs=[{'speaker_id':f'S{i:04d}','gender':'M' if i<100 else 'F','dataset_split':'train'} for i in range(2,352)]
    rows=m.plan_speakers(packs,old,set())
    confirm=[x for x in rows if x['split']=='confirmation']
    assert sum(x['gender']=='M' for x in confirm)==28
    assert sum(x['gender']=='F' for x in confirm)==32
