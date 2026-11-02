"""CPU contract checks; no model imports, GPU calls or generated media."""
import importlib.util
from pathlib import Path
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('supplement',ROOT/'scripts/experiments/audio_data_supplement_20261024.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def cohorts():
    samples=[dict(sample_id=f'source{i}',speaker_id=f'S{i:04}',transcript=' 同 一 文 本 ') for i in range(40)]
    rows=[dict(sample_id=f'source{i}',arm=arm,draw=draw,qualified=False,score_float32=-100)
          for i in range(20) for arm,draw in [('N',0),*[(a,d) for a in ['L06','L17'] for d in (1,2,3)]]]
    return dict(samples=samples),dict(rows=rows)


def test_missing_roster_uses_identity_and_keeps_old_exposure_fit():
    p,b=cohorts(); missing=m.frozen_rows(p,b)
    assert len(missing)==20
    assert {r['sample_id'] for r in missing}=={f'source{i}' for i in range(20,40)}
    assert all(r['split']=='fit' for r in missing)
    b['rows'][0]['qualified']=True; b['rows'][0]['score_float32']=100
    assert m.frozen_rows(p,b)==missing


def test_partial_or_duplicate_old_three_arm_roster_rejected():
    p,b=cohorts(); b['rows'].pop()
    with pytest.raises(ValueError,match='incomplete'): m.frozen_rows(p,b)
    p,b=cohorts(); b['rows'][-1]=dict(b['rows'][-2])
    with pytest.raises(ValueError,match='incomplete'): m.frozen_rows(p,b)


def test_sampling_and_target_cannot_drift():
    p=dict(seed_bases=m.BASES,generation_kwargs=m.KWARGS,non_streaming_mode=False,target_rms=m.TARGET)
    m.validate_settings(p)
    p['target_rms']=.04
    with pytest.raises(ValueError,match='RMS'):m.validate_settings(p)
    p['target_rms']=m.TARGET; p['generation_kwargs']={**m.KWARGS,'max_new_tokens':1024}
    with pytest.raises(ValueError,match='sampling'):m.validate_settings(p)


def test_release_requires_both_bound_pass_gates(tmp_path):
    protocol=tmp_path/'protocol.json';m.write(protocol,{'scope':'missing60_L17_only'})
    gates={}
    for name in ['replay20','inventory200']:
        path=tmp_path/(name+'.json');m.write(path,dict(status='PASS'))
        gates[name]=dict(path=str(path),sha256=m.sha(path))
    release=tmp_path/'release.json'
    data=dict(status='RELEASED',scope='missing60_L17_only',protocol_sha256=m.sha(protocol),gates=gates)
    m.write(release,data);m.require_release(release,protocol)
    (tmp_path/'replay20.json').write_text(json.dumps(dict(status='FAILED')))
    with pytest.raises(ValueError,match='gate'):m.require_release(release,protocol)


def test_outputs_and_receipts_never_overwrite(tmp_path):
    target=tmp_path/'retained.json';m.write(target,{'status':'FAILED'})
    with pytest.raises(FileExistsError):m.write(target,{'status':'COMPLETE'})
    assert m.read(target)['status']=='FAILED'
    summary=tmp_path/'summary.json';m.write(summary,{'status':'INCOMPLETE'})
    with pytest.raises(ValueError,match='incomplete'):m.artifacts(summary)


def test_gpu_busy_fails_without_process_termination(monkeypatch):
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'1234, other-research, 200 MiB\n')
    monkeypatch.setattr(m.os,'readlink',lambda *a:'/research/other-research')
    with pytest.raises(RuntimeError,match='GPU occupied'):m.gpu_idle()


def test_only_verified_exact_display_executable_is_allowed(monkeypatch):
    name='/usr/libexec/gnome-remote-desktop-daemon'
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:f'2488, {name}, 313 MiB\n')
    monkeypatch.setattr(m.os,'readlink',lambda *a:name)
    assert m.gpu_idle()['allowed_display_processes']==[
        dict(pid=2488,executable=name,reported_name=name,used_gpu_memory='313 MiB')]
    monkeypatch.setattr(m.os,'readlink',lambda *a:'/tmp/gnome-remote-desktop-daemon')
    with pytest.raises(RuntimeError,match='GPU occupied'):m.gpu_idle()


def test_display_substring_cannot_bypass_compute_veto(monkeypatch):
    name='/tmp/gnome-remote-desktop-daemon-research'
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:f'9999, {name}, 12000 MiB\n')
    monkeypatch.setattr(m.os,'readlink',lambda *a:name)
    with pytest.raises(RuntimeError,match='GPU occupied'):m.gpu_idle()


def test_prepare_does_not_import_models_or_invoke_subprocess(tmp_path,monkeypatch):
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('prepare invoked a process'))
    run=tmp_path/m.NAME;m.prepare(run,m.REMOTE)
    p=m.read(run/'protocol.json');d=m.read(run/'dryrun.json')
    assert d['media_generated']==d['models_loaded']==0
    assert p['expected_new_audio']==p['expected_new_video']==60
    assert len(p['samples'])==20 and len(p['tasks'])==100
    assert all(t['command'] is None for t in p['tasks'] if t['stage']=='features')
    assert sum(t['stage']=='synthesize' for t in p['tasks'])==20
    assert all(s['split']=='fit' for s in p['samples'])
    assert p['larger_cohort_status'].startswith('HOLD')
    assert len(list(run.rglob('*.wav')))==0
