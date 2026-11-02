"""Boundary-gated Patrick FP16 fine training; immutable parents and explicit qualification."""
import hashlib,importlib.util,io,json,os,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
HERE=Path('E:/Documents/tts-exp/scripts/experiments/synctalk_multispeaker_full_20261017')
BASE=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
OLD=BASE/'trials/fp16_other_identities_trial_v1'
OUT=BASE/'trials/patrick_fp16_fine_v3'
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017'
RUN=REMOTE+'/trials/patrick_fp16_fine_v3'
sys.path.insert(0,str(HERE))
from patrick_fine_contract_v3 import route,qualify_cases
from execution_journal import atomic_json,exclusive_lock
from paired_budget import qualify_epoch,qualify_phase

def load_parent(name):
    spec=importlib.util.spec_from_file_location(name,OLD/'sources/other_identities_trial_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
parent=load_parent('fine_parent_lifecycle')
store=load_parent('fine_new_persistence');store.OUT=OUT;store.RUN=RUN

def verify_closed(relative):
    d=OUT/relative;v=json.loads((d/'verification.json').read_text(encoding='utf-8'))
    assert v['passed'] and v['all_archived_member_bytes_verified']
    assert store.sha(d/'manifest.json')==v['manifest_sha256'] and store.sha(d/'artifacts.tar.gz')==v['archive_sha256']
    rows={r['path']:r for r in json.loads((d/'manifest.json').read_text(encoding='utf-8'))};seen=set()
    with tarfile.open(d/'artifacts.tar.gz','r|gz') as archive:
        for item in archive:
            assert item.isfile() and item.name in rows and item.name not in seen
            raw=archive.extractfile(item).read();r=rows[item.name]
            assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'];seen.add(item.name)
    for name,r in rows.items():
        if name in seen:continue
        assert 'workspace/checkpoints/' in name
        assert (d/name).stat().st_size==r['bytes'] and store.sha(d/name)==r['sha256'];seen.add(name)
    assert seen==set(rows)
    return v

def remote_status(job):
    code='import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import status;print(json.dumps(status(Path(%r))))'%(REMOTE,job)
    return json.loads(parent.remote_python(code))

def run_job(job,command):
    exists=json.loads(parent.remote_python('import json;from pathlib import Path;print(json.dumps(Path(%r).exists()))'%(job+'/command.json')))
    state=remote_status(job) if exists else None
    decision=route(False,state['state'] if state else None)
    if decision=='launch_once':
        code='import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import start_job;print(json.dumps(start_job(Path(%r),%r)))'%(REMOTE,job,command)
        state=json.loads(parent.remote_python(code))
    while state['state'] in ['starting','running']:
        atomic_json(OUT/'active_job.json',dict(job=job,execution=state));time.sleep(20);state=remote_status(job)
    assert state['state']=='completed','lost job must never replay'
    return state

def restore():
    root,archive,rows,proof=store.verify_archive('PatrickLeahy')
    inventory=parent.remote_action('inventory','PatrickLeahy')
    if inventory['bytes_required']==0 and inventory['existing']==16500:
        code='import sys,json;sys.path.insert(0,%r);from other_identities_trial_remote_v1 import selected_images,sha;root,p,rows=selected_images("PatrickLeahy");assert all((root/r["path"]).stat().st_size==r["bytes"] and sha(root/r["path"])==r["sha256"] for r in rows);print(json.dumps(dict(passed=True,verified_files=len(rows),archive_sha256=p["archive_sha256"],manifest_sha256=p["manifest_sha256"],reused_exact_files=len(rows),restored_files=0,all_restored_bytes_verified=True,local_supervision_complete=True,explicit_fresh_reverification=True)))'%(parent.RUN+'/sources')
        value=json.loads(parent.remote_python(code));assert value['archive_sha256']==proof['archive_sha256'] and value['manifest_sha256']==proof['manifest_sha256']
        atomic_json(OUT/'storage/PatrickLeahy_full_restore.json',value);store.publish('storage/PatrickLeahy_full_restore.json',value);return
    assert inventory['free']>=inventory['bytes_required']+180*1024**2,inventory
    selected={k:r for k,r in rows.items() if len(Path(k).parts)==5 and Path(k).parts[:2]==('prepared','PatrickLeahy') and Path(k).parts[2] in ['train','val','test'] and Path(k).parts[3] in ['gt_imgs','torso_imgs']}
    assert len(selected)==16500
    code='import sys,importlib.util;sys.path.insert(0,%r);s=importlib.util.spec_from_file_location("fine_restore",%r);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.run("PatrickLeahy","full")'%(REMOTE,RUN+'/sources/receive_patrick_fine_supervision_v3.py')
    log=OUT/'storage/PatrickLeahy_restore.log'
    with log.open('xb') as errors:
        p=subprocess.Popen(parent.SSH+['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
        try:
            with tarfile.open(archive,'r|gz') as source,tarfile.open(fileobj=p.stdin,mode='w|') as target:
                seen=set()
                for item in source:
                    if item.name not in selected:continue
                    b=source.extractfile(item).read();r=selected[item.name]
                    assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
                    info=tarfile.TarInfo(item.name);info.size=len(b);target.addfile(info,io.BytesIO(b));seen.add(item.name)
                assert seen==selected.keys()
            p.stdin.close();raw=p.stdout.read();assert p.wait()==0,log
        except BaseException:
            p.stdin.close();p.wait();raise
    r=json.loads(raw);assert r['passed'] and r['verified_files']==16500
    atomic_json(OUT/'storage/PatrickLeahy_full_restore.json',r)

def command(gain,epoch,checkpoint,resume,reference=None,qualify=False):
    cmd=['flock','-n',REMOTE+'/preparation_GPU.lock','bash',REMOTE+'/runtime_exec.sh',RUN+'/sources/train_patrick_fine_epoch_v3.py','--identity','PatrickLeahy','--seed','42','--gain',str(gain),'--phase','fine','--epoch',str(epoch),'--attempt','1','--checkpoint',checkpoint,'--resume',resume]
    if reference:cmd+=['--reference-trace',reference]
    if qualify:cmd+=['--qualify-only']
    return cmd

def rotate(prior,current):
    if prior['epoch']==10:return
    old=PurePosixPath(prior['checkpoint_path']);new=PurePosixPath(current['checkpoint_path'])
    local_old=OUT/old.relative_to(PurePosixPath(RUN)).as_posix();local_new=OUT/new.relative_to(PurePosixPath(RUN)).as_posix()
    assert prior['epoch']<current['epoch'] and old!=new
    assert store.sha(local_old)==prior['checkpoint_sha256'] and store.sha(local_new)==current['checkpoint_sha256']
    relative=old.parent.parent.parent.relative_to(PurePosixPath(RUN)).as_posix()
    atomic_json(OUT/relative/'checkpoint_retirement_intent.json',dict(old_sha256=prior['checkpoint_sha256'],superseded_by=current['checkpoint_sha256'],parent_coarse_preserved=True))
    code='''import fcntl,os,hashlib
from pathlib import Path
p=Path(%r);q=Path(%r);root=Path(%r)
assert p.resolve().is_relative_to(root/'training') and q.resolve().is_relative_to(root/'training')
assert hashlib.sha256(p.read_bytes()).hexdigest()==%r and hashlib.sha256(q.read_bytes()).hexdigest()==%r
fd=os.open(p,os.O_RDWR)
try:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK);p.unlink()
finally:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
'''%(str(old),str(new),RUN,prior['checkpoint_sha256'],current['checkpoint_sha256'])
    parent.remote_python(code);local_old.unlink()
    atomic_json(OUT/relative/'checkpoint_cache_retirement.json',dict(passed=True,old_sha256=prior['checkpoint_sha256'],superseded_by=current['checkpoint_sha256']))

def run():
    contract=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
    for name,digest in contract['sources'].items():assert store.sha(OUT/'sources'/name)==digest
    assert store.sha(OLD/'contract.json')==contract['parent_contract_sha256']
    with exclusive_lock(OUT/'controller.lock'):
        assert not (OUT/'receipt.json').exists(),'existing fine run requires a new audited handoff, no blind restart'
        state=dict(pid=os.getpid(),version=contract['version'],status='waiting_chris_boundary',current=None,completed=[],failures=[],descriptive_only=True,parent_control_failed_retained=True)
        def save():atomic_json(OUT/'receipt.json',state)
        save()
        pause=OLD/'controller_revision04_pause_for_patrick_fine_v1/paused.json'
        failure=OLD/'controller_revision04_pause_for_patrick_fine_v1/failure_v3.json'
        while not pause.exists():
            assert not failure.exists(),'boundary observation failed; do not start GPU';time.sleep(10)
        proof=json.loads(pause.read_text(encoding='utf-8'));assert proof['passed'] and proof['status']=='paused_at_verified_epoch_boundary'
        while not (OUT/'capacity_handoff_ready.json').exists():time.sleep(10)
        assert json.loads((OUT/'capacity_handoff_ready.json').read_text(encoding='utf-8'))['passed']
        assert not parent.remote_python("import subprocess;print(subprocess.check_output(['pgrep','-af','[t]rain_other_identity_epoch_v1.py']).decode() if subprocess.run(['pgrep','-f','[t]rain_other_identity_epoch_v1.py'],stdout=subprocess.DEVNULL).returncode==0 else '',end='')").strip(),'training reader still active'
        state['status']='preserving_chris_supervision_before_cache_handoff';save()
        retired=OLD/'storage/ChrisVanHollen_cache_retirement.json'
        if retired.exists():assert json.loads(retired.read_text(encoding='utf-8'))['passed']
        else:parent.retire('ChrisVanHollen')
        state['status']='restoring_patrick_full_supervision';save();restore()
        state['status']='qualifying_fine_entry';save();cases={}
        for gain in [0,-6,6]:
            pin=contract['startpoints'][str(gain)]
            job=RUN+f'/qualification/jobs/gain_{gain:+d}/attempt01'
            result=run_job(job,command(gain,11,pin['checkpoint_path'],pin['receipt_path'],qualify=True))
            relative=f'qualification/cases/PatrickLeahy/seed42/gain_{gain:+d}/fine/ep0011/attempt01'
            store.persist(f'qualification/jobs/gain_{gain:+d}/attempt01');store.persist(relative)
            assert result['returncode']==0,'fine entry qualification failed; preserved, no formal updates'
            raw=parent.remote_python('from pathlib import Path;print(Path(%r).read_text())'%(RUN+'/'+relative+'/qualification_receipt.json'))
            r=json.loads(raw);atomic_json(OUT/relative/'qualification_receipt.json',r);cases[gain]=r
        qualified=qualify_cases(store.sha(OUT/'contract.json'),cases)
        atomic_json(OUT/'qualification/result.json',qualified);store.publish('qualification/result.json',qualified)
        state['qualification']=qualified;state['status']='fine_training';save()
        previous={g:dict(json.loads((OLD/'training/PatrickLeahy/seed42'/f'gain_{g:+d}'/'coarse/ep0010/attempt01/epoch_receipt.json').read_text(encoding='utf-8')),_receipt=contract['startpoints'][str(g)]['receipt_path']) for g in [0,-6,6]}
        epochs=[]
        for epoch in range(11,18):
            receipts={};traces={};reference=None
            for gain in [0,-6,6]:
                relative=f'training/PatrickLeahy/seed42/gain_{gain:+d}/fine/ep{epoch:04d}/attempt01';prior=previous[gain]
                state.update(current=dict(gain_db=gain,epoch=epoch,relative=relative),status='fine_training');save()
                result=run_job(RUN+'/'+relative.replace('training/','jobs/',1),command(gain,epoch,prior['checkpoint_path'],prior['_receipt'],reference))
                store.persist(relative.replace('training/','jobs/',1));store.persist(relative)
                verify_closed(relative)
                if result['returncode']!=0:
                    state['failures'].append(dict(gain_db=gain,epoch=epoch,execution=result));state['status']='terminal_failure_preserved';save();return
                r=json.loads((OUT/relative/'epoch_receipt.json').read_text(encoding='utf-8'));assert r['passed']
                r['_receipt']=RUN+'/'+relative+'/epoch_receipt.json';receipts[gain]=r;traces[gain]=OUT/relative/'attempts.jsonl'
                if gain==0:reference=RUN+'/'+relative+'/attempts.jsonl'
                rotate(prior,r);previous[gain]=r
            block=qualify_epoch(receipts,traces);epochs.append(block);atomic_json(OUT/'blocks'/f'fine_ep{epoch:04d}.json',block)
        phase=qualify_phase(epochs,previous)
        phase.update(parent_control_failed_retained=True,descriptive_only=True,cumulative_actual_updates_by_gain={g:r['cumulative_successful_updates'] for g,r in previous.items()},all_phases_control_passed=False)
        atomic_json(OUT/'blocks/fine_qualification.json',phase)
        state.update(status='fine_complete_pending_27_grid',current=None,completed=[phase],passed=True,scientific_control_passed=False);save()
        atomic_json(OUT/'fine_ready_for_27_grid.json',dict(passed=True,phase=phase,contract_sha256=store.sha(OUT/'contract.json'),required_cells=27,checkpoint_epoch=17,global_step=102000,do_not_reuse_coarse_outputs=True))

if __name__=='__main__':
    try:run()
    except BaseException:
        atomic_json(OUT/'controller_failure.json',dict(pid=os.getpid(),status='failed_no_automatic_replay',traceback=traceback.format_exc()));raise
