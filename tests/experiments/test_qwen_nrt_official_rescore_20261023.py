import importlib.util
from pathlib import Path
from decimal import Decimal
import pytest

path=Path(__file__).resolve().parents[2]/'scripts/experiments/qwen_nrt_official_rescore_20261023.py'
spec=importlib.util.spec_from_file_location('official200',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_first_printed_track_is_preserved_even_if_worse():
    text='AV offset: -2\nMin dist: 8.01\nConfidence: 1.25\nAV offset: 0\nMin dist: 6\nConfidence: 9.9\n'
    result=module.parse_tracks(text)
    assert len(result)==2 and result[0]['sync_c']==1.25 and result[1]['sync_c']==9.9


def test_incomplete_or_interleaved_tracks_fail():
    for text in ['AV offset:0\nConfidence:1', 'Min dist:2\nAV offset:0\nConfidence:1']:
        with pytest.raises(ValueError):module.parse_tracks(text)


def test_signed_and_scientific_numbers_parse():
    assert module.parse_tracks('AV offset: -3\nMin dist: 2e0\nConfidence: -.5\n')[0]['sync_c']==-.5


def test_hash_and_clock_checks_are_independent():
    text='#tb 0: 1/25\n0, 0, 0, 1, 20, abc\n0, 1, 1, 1, 20, def\n'
    assert module.frame_hashes(text)==['abc','def']
    assert module.frame_clock({'frames':[{'best_effort_timestamp_time':'0.040000'}]})==[Decimal('.04')]
    with pytest.raises(ValueError):module.frame_clock({'frames':[{}]})


def test_cohort_and_full200_binding():
    audio=[];video=[]
    for i in range(2,42):
        for role in module.ROLES:
            row=dict(sample_id=f'utt{i:04d}',speaker_id=f'S{i:04d}',cohort='pilot10' if i<=11 else 'new30',role=role,path='x',sha256='s',pcm_sha256='p',samples=10)
            audio.append(row);video.append(dict(row,audio_sha256='s'))
    rows=module.validate_inputs(audio,video)
    assert len(rows)==200 and rows[0]['role']=='N' and rows[-1]['role']=='T3'
    video[-1]['audio_sha256']='changed'
    with pytest.raises(ValueError):module.validate_inputs(audio,video)
