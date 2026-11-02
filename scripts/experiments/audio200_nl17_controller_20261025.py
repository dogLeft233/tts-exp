"""Separate bounded N/L17 phase coordinator. No L06 work and no training."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PHASE='N_L17_source600_v1'
UTILITY_SHA='86d64aca90a9aaa1e0b088084eeb6fbe6a17b39e3d203a73b57fd37b61bf9474'
AUDIT_PASS='PASS_PARENT_NL17_BATCH_REPLAY_CLOCK_PATCH'
FINAL_PASS='COMPLETE_NL17_DATASET2400_TECHNICAL_PREPARATION_ONLY'
LOCK=ROOT/'runs/.audio200_nl17_controller.lock'


def utility():
    import hashlib
    source=ROOT/'scripts/experiments/audio200_batch_controller_20261025_attempt02.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==UTILITY_SHA
    spec=importlib.util.spec_from_file_location('frozen_serial_controller_utilities',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def validate(a,u):
    assert a['phase']==PHASE and a['allow_training'] is False
    assert a['allow_media_generation'] is True and a['allow_frozen_syncnet_forward'] is True and a['allow_cpu_fa'] is True
    assert a['controller_source_sha256']==u.sha(__file__)
    for path,sha in [('producer_protocol_path','producer_protocol_sha256'),('producer_source_path','producer_source_sha256'),
                     ('consumer_source_path','consumer_source_sha256'),('consumer_contract_path','consumer_contract_sha256'),
                     ('parent_audit_source_path','parent_audit_source_sha256'),('pilot_parent_audit_path','pilot_parent_audit_sha256'),
                     ('finalizer_source_path','finalizer_source_sha256')]:
        assert Path(a[path]).resolve().is_relative_to(ROOT.resolve())
        u.verify({a[path]:a[sha]})
    p=u.read(a['producer_protocol_path'])
    assert p['phase']==PHASE and p['expected_units']==2400 and p['no_training'] is True
    assert p['cache_reused_units']==248 and p['ready_media_units']==10 and len(p['work_batches'])==55
    assert all(g['arm'] in ('N','L17') for b in p['work_batches'] for g in b['groups'])
    pilot=p['work_batches'][0]
    assert pilot['id']=='bridge003a' and len(pilot['source_ids'])==1
    audit=u.read(a['pilot_parent_audit_path'])
    assert audit['status']==AUDIT_PASS and audit['batch_id']==pilot['id'] and audit['source_ids']==pilot['source_ids']
    remaining={s for b in p['work_batches'][1:] for s in b['source_ids']}
    assert len(remaining)==537 and set(a['allowed_source_ids'])==remaining and len(a['allowed_source_ids'])==537
    assert a['material_roots']==p['material_roots'] and len(a['material_roots'])==3
    assert a['aggregate_output_cap_bytes']==p['aggregate_output_cap_bytes']==60*1024**3
    assert 0<a.get('hard_timeout_seconds',604800)<=604800
    for name in ('controller_run','parent_audit_root','final_output_root'):
        assert Path(a[name]).resolve().is_relative_to((ROOT/'runs').resolve())
    u.MATERIAL_ROOTS=p['material_roots'][:];u.material_guard()
    u.verify(p['source_pins'])
    u.verify({p['reused_cache_manifest_path']:p['reused_cache_manifest_sha256'],p['ready_media_manifest_path']:p['ready_media_manifest_sha256']})
    return p


def final_output(output,u):
    output=Path(output);terminal=output/'final_status.json';s=u.read(terminal)
    assert s['status']==FINAL_PASS and s['no_training'] is True
    assert (s['units'],s['source_count'],s['speaker_count'])==(2400,600,200)
    assert s['pairs']==3600
    manifest=output/'final_manifest.json';u.verify({str(manifest):s['final_manifest_sha256']})
    for f in u.read(manifest)['files']:
        p=output/f['path'];assert p.resolve().is_relative_to(output.resolve())
        u.verify({str(p):f['sha256']});assert p.stat().st_size==f['bytes']
    return terminal


def execute(auth_path):
    u=utility();a=u.read(auth_path);p=validate(a,u)
    run=Path(a['controller_run']);producer_run=Path(p['run']);audit_root=Path(a['parent_audit_root'])
    run.mkdir(parents=True,exist_ok=True);audit_root.mkdir(parents=True,exist_ok=True)
    for lock in (LOCK,ROOT/'runs/.audio200_batch_controller.lock'):
        if lock.exists():raise RuntimeError('other controller lock; no forced takeover '+str(lock))
    LOCK.mkdir();u.put(LOCK/'owner.json',dict(pid=os.getpid(),run=str(run),phase=PHASE,authorization_sha256=u.sha(auth_path)))
    attempts=run/'attempts';attempts.mkdir(exist_ok=True);attempt=attempts/f'{len(list(attempts.iterdir()))+1:03d}';attempt.mkdir()
    u.put(attempt/'launch.json',dict(pid=os.getpid(),command=sys.argv,utc=u.utc(),authorization_sha256=u.sha(auth_path),phase=PHASE))
    error=None;deadline=time.monotonic()+a.get('hard_timeout_seconds',604800)
    try:
        snapshot=run/'authorization_snapshot.json'
        if snapshot.exists():assert u.read(snapshot)==a
        else:u.put(snapshot,a)
        for batch in p['work_batches'][1:]:
            validate(a,u);folder=run/'batches'/batch['id'];folder.mkdir(parents=True,exist_ok=True)
            complete=folder/'completion.json'
            if complete.exists():
                old=u.read(complete);assert old['status']=='PASS_NL17_BATCH_AND_PARENT_AUDIT';u.verify(old['artifacts'])
                u.event(run,'REUSED_NL17_BATCH',batch=batch['id']);continue
            release=folder/'child_release.json'
            narrow=dict(phase=PHASE,allow_training=False,allow_media_generation=True,allow_frozen_syncnet_forward=True,allow_cpu_fa=True,
                producer_protocol_sha256=a['producer_protocol_sha256'],producer_source_sha256=a['producer_source_sha256'],
                consumer_source_path=a['consumer_source_path'],consumer_source_sha256=a['consumer_source_sha256'],
                consumer_contract_sha256=a['consumer_contract_sha256'],allowed_source_ids=batch['source_ids'],hard_timeout_seconds=10800,
                parent_authorization_path=str(auth_path),parent_authorization_sha256=u.sha(auth_path))
            if release.exists():assert u.read(release)==narrow
            else:u.put(release,narrow)
            command=[u.SYNC,a['producer_source_path'],'execute','--run',str(producer_run),'--release',str(release),'--consumer-contract',a['consumer_contract_path']]
            u.event(run,'NL17_BATCH_START',batch=batch['id'],sources=batch['source_ids'])
            u.checked(run,folder,'producer',command,deadline,10830)
            media=Path(batch['media_manifest_path']);feature=media.parent/'features'/('confirmation_sealed' if batch['split']=='confirmation' else batch['split'])
            audit=audit_root/(batch['id']+'.json')
            command=[u.SYNC,a['parent_audit_source_path'],'--manifest',str(feature/'manifest.json'),'--clock',str(feature/'window_clock.json'),
                '--patch',str(feature/'patch_index.json'),'--output',str(audit)]
            u.checked(run,folder,'parent_audit',command,deadline,1200)
            report=u.read(audit);assert report['status']==AUDIT_PASS and report['batch_id']==batch['id']
            bindings=u.artifact_bindings(producer_run,batch,audit)
            if batch['ready_media_ids']:
                for origin in p['ready_media_origin_receipts']:
                    u.verify({origin['path']:origin['sha256']});bindings[origin['path']]=origin['sha256']
                    bindings.update(u.read(origin['path'])['artifacts'])
            u.verify(bindings)
            u.put(complete,dict(status='PASS_NL17_BATCH_AND_PARENT_AUDIT',phase=PHASE,batch=batch['id'],source_ids=batch['source_ids'],
                artifacts=bindings,utc=u.utc(),no_training=True))
            u.event(run,'NL17_BATCH_COMPLETE',batch=batch['id'],units=len(batch['source_ids'])*4)
        folder=run/'finalization';folder.mkdir(exist_ok=True)
        command=[u.SYNC,a['finalizer_source_path'],'--producer-run',str(producer_run),'--parent-audit-root',str(audit_root),'--output-root',a['final_output_root']]
        u.checked(run,folder,'finalize',command,deadline,1800)
        final=final_output(a['final_output_root'],u)
        u.put(attempt/'completion.json',dict(status=FINAL_PASS,phase=PHASE,completion_path=str(final),completion_sha256=u.sha(final),no_training=True,utc=u.utc()))
        u.event(run,'NL17_DATA_READY',completion_path=str(final),units=2400)
    except BaseException as e:
        error=repr(e);u.put(attempt/'failure.json',dict(status='FAILED_RETAINED_NO_RETRY',error=error,phase=PHASE,no_training=True,utc=u.utc()));raise
    finally:
        (LOCK/'owner.json').unlink();LOCK.rmdir();u.event(run,'NL17_CONTROLLER_EXIT',error=error,no_training=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True)
    a=parser.parse_args();execute(a.authorization)
