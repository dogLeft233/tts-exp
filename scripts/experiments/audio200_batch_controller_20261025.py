"""Durable exact-batch coordinator, gated by a completed independent pilot audit.

No training and no scientific analysis. Never retries a launched failed task.
The original producer source/protocol are immutable dependencies.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SYNC='/home/wjj/.venvs/syncnet/bin/python'
LOCK=ROOT/'runs/.audio200_batch_controller.lock'
FREE=12*1024**3
BATCH_BUDGET=2*1024**3  # Up to70 media plus temporary forwards; verify against pilot actuals.
AUDIT_PASS='PASS_PARENT_BATCH_REPLAY_CLOCK_PATCH'
FINAL_PASS='COMPLETE_DATASET4200_TECHNICAL_PREPARATION_ONLY'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def put(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def verify(pins):
    for path,value in pins.items():
        if sha(path)!=value:
            raise ValueError('bound artifact drift: '+str(path))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_authorization(a):
    assert a['allow_media_generation'] is True
    assert a['allow_training'] is False
    assert a['allow_frozen_syncnet_forward'] is True and a['allow_cpu_fa'] is True
    assert a['controller_source_sha256']==sha(__file__)
    assert 0<int(a.get('hard_timeout_seconds',7*86400))<=7*86400
    pairs=[('producer_protocol_path','producer_protocol_sha256'),
           ('producer_source_path','producer_source_sha256'),
           ('consumer_contract_path','consumer_contract_sha256'),
           ('consumer_source_path','consumer_source_sha256'),
           ('parent_audit_source_path','parent_audit_source_sha256'),
           ('pilot_parent_audit_path','pilot_parent_audit_sha256'),
           ('finalizer_source_path','finalizer_source_sha256')]
    verify({a[p]:a[h] for p,h in pairs})
    for p,_ in pairs:
        assert Path(a[p]).resolve().is_relative_to(ROOT.resolve())
    for p in ('controller_run','parent_audit_root','final_output_root'):
        assert Path(a[p]).resolve().is_relative_to((ROOT/'runs').resolve())
    producer=read(a['producer_protocol_path'])
    assert producer['no_training'] is True and producer['expected_units']==4200
    assert producer['reused_units']==280 and producer['new_units']==3920
    batches=producer['batches']
    assert len(batches)==57
    assert batches[0]['source_ids']==['BAC009S0002W0322','BAC009S0002W0363']
    assert read(a['pilot_parent_audit_path'])['status']==AUDIT_PASS
    assert Path(a['pilot_parent_audit_path']).name==batches[0]['id']+'.json'
    rest={s for b in batches[1:] for s in b['source_ids']}
    assert len(rest)==558 and len(a['allowed_source_ids'])==558 and set(a['allowed_source_ids'])==rest
    assert len({b['id'] for b in batches})==57
    assert all(len(b['source_ids'])<=10 for b in batches)
    assert len({s for b in batches for s in b['source_ids']})==560
    return producer


def event(run,name,**kw):
    row=dict(event=name,pid=os.getpid(),utc=utc(),**kw)
    with (run/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def kill_own(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait()


def checked(run,directory,name,command,deadline,timeout_s):
    launch=directory/(name+'_launch.json')
    exit_file=directory/(name+'_exit.json')
    if exit_file.exists():
        old=read(exit_file)
        if old['returncode']!=0 or old.get('error'):
            raise RuntimeError('retained previous failure: '+str(exit_file))
        verify({old['log']:old['log_sha256']})
        assert old['command']==command
        return old
    if launch.exists():
        raise RuntimeError('unresolved previous task; do not silently retry: '+str(launch))
    if shutil.disk_usage(run).free<FREE+BATCH_BUDGET:
        raise RuntimeError('disk budget floor reached')
    put(launch,dict(command=command,utc=utc(),parent_pid=os.getpid(),no_training=True))
    log=directory/(name+'.log')
    started=time.monotonic();p=None;error=None;returncode=None
    try:
        with log.open('x',encoding='utf-8') as f:
            p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic()>deadline or time.monotonic()-started>timeout_s:
                    kill_own(p)
                    raise TimeoutError('own task hard timeout')
                if shutil.disk_usage(run).free<FREE:
                    kill_own(p)
                    raise RuntimeError('disk floor breached; own task stopped')
            returncode=p.wait()
            if returncode:
                raise RuntimeError('child failed '+name)
    except BaseException as exc:
        error=repr(exc)
        if p is not None:
            kill_own(p);returncode=p.returncode
    receipt=dict(command=command,returncode=returncode,error=error,elapsed_s=time.monotonic()-started,
        pid=p.pid if p else None,log=str(log),log_sha256=sha(log),utc=utc(),no_training=True)
    put(exit_file,receipt)
    if error:
        raise RuntimeError(error)
    return receipt


def artifact_bindings(producer_run,batch,audit_file):
    pins={str(audit_file):sha(audit_file),batch['media_manifest_path']:sha(batch['media_manifest_path'])}
    for group in batch['groups']:
        for stage in ('synthesize','canonical','render','official'):
            receipt=producer_run/'stage_receipts'/f"{batch['id']}_{group['arm']}_{stage}.json"
            data=read(receipt)
            assert data['status']=='COMPLETE'
            pins[str(receipt)]=sha(receipt)
            pins.update(data['artifacts'])
    consumer_receipt=producer_run/'consumer_receipts'/f"{batch['id']}.json"
    data=read(consumer_receipt)
    assert data['status']=='COMPLETE'
    pins[str(consumer_receipt)]=sha(consumer_receipt)
    pins.update(data['artifacts'])
    verify(pins)
    return pins


def accept_final_output(output):
    """Consume the finalizer's real terminal file and recheck its content seal."""
    output=Path(output)
    terminal=output/'final_status.json'
    status=read(terminal)
    assert status['status']==FINAL_PASS and status['no_training'] is True
    assert (status['units'],status['source_count'],status['speaker_count'])==(4200,600,200)
    assert (status['reused'],status['new'],status['pairs'])==(280,3920,9000)
    manifest=output/'final_manifest.json'
    verify({str(manifest):status['final_manifest_sha256']})
    entries=read(manifest)['files']
    for entry in entries:
        path=output/entry['path']
        assert path.resolve().is_relative_to(output.resolve())
        verify({str(path):entry['sha256']})
        assert path.stat().st_size==entry['bytes']
    return terminal


def execute(auth_path):
    a=read(auth_path);producer=validate_authorization(a)
    root=Path(a['controller_run']);producer_run=Path(producer['run'])
    audit_root=Path(a['parent_audit_root'])
    root.mkdir(parents=True,exist_ok=True)
    audit_root.mkdir(parents=True,exist_ok=True)
    if LOCK.exists():
        raise RuntimeError('controller lock exists; parent review required')
    LOCK.mkdir()
    put(LOCK/'owner.json',dict(pid=os.getpid(),run=str(root),authorization_path=str(auth_path),authorization_sha256=sha(auth_path),utc=utc()))
    attempts=root/'attempts';attempts.mkdir(exist_ok=True)
    attempt=attempts/f'{len(list(attempts.iterdir()))+1:03d}';attempt.mkdir()
    put(attempt/'launch.json',dict(pid=os.getpid(),authorization_sha256=sha(auth_path),command=sys.argv,utc=utc(),no_training=True))
    deadline=time.monotonic()+int(a.get('hard_timeout_seconds',7*86400))
    error=None
    try:
        if not (root/'authorization_snapshot.json').exists():
            put(root/'authorization_snapshot.json',a)
        else:
            assert read(root/'authorization_snapshot.json')==a,'authorization changed; use reviewed new controller attempt'
        for batch in producer['batches'][1:]:
            # Recheck executable and audit bindings before admitting each batch.
            validate_authorization(a)
            directory=root/'batches'/batch['id'];directory.mkdir(parents=True,exist_ok=True)
            complete=directory/'completion.json'
            if complete.exists():
                old=read(complete)
                assert old['status']=='PASS_BATCH_AND_PARENT_AUDIT'
                verify(old['artifacts'])
                event(root,'REUSED_VERIFIED_BATCH',batch=batch['id'])
                continue
            event(root,'BATCH_START',batch=batch['id'],sources=batch['source_ids'],split=batch['split'])
            release=directory/'child_release.json'
            narrow=dict(allow_media_generation=True,allow_training=False,allow_frozen_syncnet_forward=True,allow_cpu_fa=True,
                producer_protocol_sha256=a['producer_protocol_sha256'],producer_source_sha256=a['producer_source_sha256'],
                consumer_contract_sha256=a['consumer_contract_sha256'],consumer_source_sha256=a['consumer_source_sha256'],
                allowed_source_ids=batch['source_ids'],hard_timeout_seconds=10800,additional_disk_budget_bytes=BATCH_BUDGET,
                parent_authorization_path=str(auth_path),parent_authorization_sha256=sha(auth_path),
                pilot_parent_audit_sha256=a['pilot_parent_audit_sha256'])
            if release.exists():
                assert read(release)==narrow
            else:
                put(release,narrow)
            command=[SYNC,a['producer_source_path'],'execute','--run',str(producer_run),'--release',str(release),'--consumer-contract',a['consumer_contract_path']]
            checked(root,directory,'producer',command,deadline,10830)
            media=Path(batch['media_manifest_path']);feature=media.parent/'features'/('confirmation_sealed' if batch['split']=='confirmation' else batch['split'])
            audit_file=audit_root/(batch['id']+'.json')
            command=[SYNC,a['parent_audit_source_path'],'--manifest',str(feature/'manifest.json'),
                '--clock',str(feature/'window_clock.json'),'--patch',str(feature/'patch_index.json'),'--output',str(audit_file)]
            checked(root,directory,'parent_audit',command,deadline,1200)
            assert read(audit_file)['status']==AUDIT_PASS
            bindings=artifact_bindings(producer_run,batch,audit_file)
            put(complete,dict(status='PASS_BATCH_AND_PARENT_AUDIT',batch=batch['id'],source_ids=batch['source_ids'],
                split=batch['split'],artifacts=bindings,utc=utc(),no_training=True))
            event(root,'BATCH_COMPLETE',batch=batch['id'],units=len(batch['source_ids'])*7)
        directory=root/'finalization';directory.mkdir(exist_ok=True)
        command=[SYNC,a['finalizer_source_path'],'--producer-run',str(producer_run),
            '--parent-audit-root',str(audit_root),'--output-root',a['final_output_root']]
        checked(root,directory,'finalize',command,deadline,1800)
        final=accept_final_output(a['final_output_root'])
        put(attempt/'completion.json',dict(status=FINAL_PASS,completion_path=str(final),completion_sha256=sha(final),
            batches56_pass=True,no_training=True,utc=utc()))
        event(root,'ALL_DATA_READY',completion_path=str(final),completion_sha256=sha(final))
    except BaseException as exc:
        error=repr(exc)
        put(attempt/'failure.json',dict(status='FAILED_RETAINED_NO_RETRY',error=error,utc=utc(),no_training=True))
        raise
    finally:
        (LOCK/'owner.json').unlink();LOCK.rmdir()
        event(root,'CONTROLLER_EXIT',error=error,no_training=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--authorization',type=Path,required=True)
    args=parser.parse_args()
    execute(args.authorization)
