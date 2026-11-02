"""Windows controller: three independent full coarse FP16 trials, immutable failures."""
import hashlib,io,json,os,shlex,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017'
VERSION='fp16_other_identities_trial_v1'
OUT=BASE/'trials'/VERSION
RUN=REMOTE+'/trials/'+VERSION
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2']
from cohort_binding import identity_root,expected_root,sha
from execution_journal import atomic_json,exclusive_lock
from paired_budget import qualify_epoch,qualify_phase

def remote_python(code):
    return subprocess.check_output(SSH+['python3 -'],input=code.encode())

def remote_action(action,identity,**values):
    args=['python3',RUN+'/sources/other_identities_trial_remote_v1.py',action,'--identity',identity]
    for name,value in values.items():
        if value is not None:args+=['--'+name.replace('_','-'),str(value)]
    return json.loads(subprocess.check_output(SSH+[shlex.join(args)]))

def publish(relative,value):
    code='from pathlib import Path;import sys;p=Path(%r);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(sys.stdin.buffer.read())'%(RUN+'/'+relative)
    subprocess.run(SSH+['python3 -c '+shlex.quote(code)],input=json.dumps(value,indent=2).encode(),check=True)

def verify_archive(identity):
    root=identity_root(identity,BASE,require_admission=True);folder=root/'derived_packages'/identity
    proof=json.loads((folder/'verification.json').read_text());manifest=folder/'manifest.json';archive=Path(proof['archive'])
    assert proof['passed'] and proof['all_archived_member_bytes_verified'] and sha(archive)==proof['archive_sha256'] and sha(manifest)==proof['manifest_sha256']
    rows={r['path']:r for r in json.loads(manifest.read_text())['files']};seen=set()
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            assert member.isfile() and member.name in rows and member.name not in seen
            payload=stream.extractfile(member).read();row=rows[member.name]
            assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256'];seen.add(member.name)
    assert seen==rows.keys()
    result=dict(passed=True,identity=identity,archive=str(archive),archive_sha256=proof['archive_sha256'],manifest_sha256=proof['manifest_sha256'],all_archived_member_bytes_verified=True,files=len(seen))
    path=OUT/'storage'/f'{identity}_fresh_archive_verification.json';path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,result)
    return root,archive,rows,result

def retire(identity):
    _,_,_,proof=verify_archive(identity)
    relative=f'storage/{identity}_fresh_archive_verification.json'
    publish(relative,proof)
    receipt=remote_action('retire',identity,authorization=RUN+'/'+relative)
    atomic_json(OUT/'storage'/f'{identity}_cache_retirement.json',receipt)

def restore(identity):
    root,archive,rows,proof=verify_archive(identity)
    inventory=remote_action('inventory',identity)
    # Full input plus a bounded checkpoint/trace/workspace headroom; never start under known shortage.
    assert inventory['free']>=inventory['bytes_required']+180*1024**2,inventory
    remote_root=expected_root(Path(REMOTE),identity).as_posix()
    expected={k:r for k,r in rows.items() if len(Path(k).parts)==5 and Path(k).parts[:2]==('prepared',identity) and Path(k).parts[2] in ['train','val','test'] and Path(k).parts[3] in ['gt_imgs','torso_imgs']}
    assert len(expected)==16500
    # Call the already qualified receiver, preserving its original namespace and byte checks.
    # This is the first full restore for these three identities; existing receipts are not overwritten.
    code='import sys;sys.path.insert(0,%r);import receive_supervision_restore as r;r.run(%r,"full")'%(remote_root,identity)
    log=OUT/'storage'/f'{identity}_restore.log'
    with log.open('xb') as errors:
        process=subprocess.Popen(SSH+['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
        try:
            with tarfile.open(archive,'r|gz') as source,tarfile.open(fileobj=process.stdin,mode='w|') as target:
                seen=set()
                for item in source:
                    if item.name not in expected:continue
                    payload=source.extractfile(item).read();row=expected[item.name]
                    assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
                    info=tarfile.TarInfo(item.name);info.size=len(payload);target.addfile(info,io.BytesIO(payload));seen.add(item.name)
                assert seen==expected.keys()
            process.stdin.close();output=process.stdout.read();assert process.wait()==0,log
        except BaseException:
            process.stdin.close();process.wait();raise
    receipt=json.loads(output);assert receipt['passed'] and receipt['verified_files']==16500
    atomic_json(OUT/'storage'/f'{identity}_full_restore.json',receipt)

def persist(relative):
    """Store all closed epoch/job bytes; independently verify each tar member before cache rotation."""
    destination=OUT/relative;destination.mkdir(parents=True,exist_ok=False)
    remote=RUN+'/'+relative
    code='''import json,hashlib
from pathlib import Path
p=Path(%r);rows=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and '__pycache__' not in f.parts:
  assert not f.is_symlink()
  rows.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
print(json.dumps(rows))
'''%remote
    manifest_raw=remote_python(code);rows=json.loads(manifest_raw)
    assert rows
    (destination/'manifest.json').write_bytes(manifest_raw)
    # Separate full checkpoints so verified superseded intermediate state caches can rotate.
    checkpoints=[r for r in rows if 'workspace/checkpoints/' in r['path']]
    artifacts=[r for r in rows if r not in checkpoints]
    for row in checkpoints:
        target=destination/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:subprocess.run(SSH+['cat '+shlex.quote(remote+'/'+row['path'])],stdout=f,check=True)
        assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
    archive=destination/'artifacts.tar.gz'
    with archive.open('xb') as f:
        subprocess.run(SSH+['tar --hard-dereference -czf - -C '+shlex.quote(remote)+' --null -T -'],input=('\0'.join(r['path'] for r in artifacts)+'\0').encode(),stdout=f,check=True)
    seen=set();expected={r['path']:r for r in artifacts}
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            assert member.isfile() and member.name in expected and member.name not in seen
            row=expected[member.name];payload=stream.extractfile(member).read()
            assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256'];seen.add(member.name)
            # Exact small working copies for QA. All source/state/log/curve bytes remain in verified archive.
            if member.name in ['attempts.jsonl','epoch_receipt.json','execution.json','command.json']:
                (destination/member.name).write_bytes(payload)
    assert seen==expected.keys() and remote_python(code)==manifest_raw,'source changed during persistence'
    receipt=dict(passed=True,relative=relative,files=len(rows),bytes=sum(r['bytes'] for r in rows),archive_sha256=sha(archive),manifest_sha256=sha(destination/'manifest.json'),all_archived_member_bytes_verified=True,checkpoints=checkpoints)
    atomic_json(destination/'verification.json',receipt);return receipt

def retire_previous(previous,current):
    if previous is None:return
    old=Path(previous['checkpoint_path']);new=Path(current['checkpoint_path'])
    assert old!=new and previous['epoch']<10
    old_relative=old.relative_to(Path(RUN)).as_posix();new_relative=new.relative_to(Path(RUN)).as_posix()
    f_old=OUT/old_relative;f_new=OUT/new_relative
    assert f_old.resolve().is_relative_to(OUT.resolve()) and sha(f_old)==previous['checkpoint_sha256'] and sha(f_new)==current['checkpoint_sha256']
    old_epoch=old.parent.parent.parent.relative_to(Path(RUN)).as_posix()
    assert json.loads((OUT/old_epoch/'verification.json').read_text())['passed']
    atomic_json(OUT/old_epoch/'checkpoint_retirement_intent.json',dict(old_sha256=previous['checkpoint_sha256'],superseded_by=current['checkpoint_sha256']))
    code='''import hashlib,fcntl,os,json
from pathlib import Path
old=Path(%r);new=Path(%r);root=Path(%r)
assert old.resolve().is_relative_to(root/'training') and new.resolve().is_relative_to(root/'training')
assert hashlib.sha256(new.read_bytes()).hexdigest()==%r and hashlib.sha256(old.read_bytes()).hexdigest()==%r
fd=os.open(old,os.O_RDWR)
try:
 fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK);old.unlink()
finally:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
'''%(str(old),str(new),RUN,current['checkpoint_sha256'],previous['checkpoint_sha256'])
    remote_python(code);f_old.unlink()
    atomic_json(OUT/old_epoch/'checkpoint_cache_retirement.json',dict(passed=True,old_sha256=previous['checkpoint_sha256'],superseded_by=current['checkpoint_sha256'],reason='verified newer own-arm full checkpoint; final and failure states retained'))

def run():
    contract=json.loads((OUT/'contract.json').read_text())
    for name,digest in contract['sources'].items():assert sha(OUT/'sources'/name)==digest
    qualification=json.loads((OUT/'qualification/result.json').read_text());assert qualification['passed']
    state=dict(pid=os.getpid(),version=VERSION,status='starting',identities=contract['identities'],completed=[],failures=[],current=None,formal_original_models_completed=0)
    def save():atomic_json(OUT/'receipt.json',state)
    with exclusive_lock(OUT/'controller.lock'):
        assert not (OUT/'receipt.json').exists(),'manual review required before any coordinator recovery'
        save();retire('SherrodBrown')
        for identity in contract['identities']:
            state.update(status='restoring_full_supervision',current=dict(identity=identity));save()
            restore(identity);previous={g:None for g in [0,-6,6]};epochs=[];failed=False
            for epoch in range(1,11):
                receipts={};traces={};reference=None
                for gain in [0,-6,6]:
                    relative=f'training/{identity}/seed42/gain_{gain:+d}/coarse/ep{epoch:04d}/attempt01'
                    job=relative.replace('training/','jobs/',1)
                    state.update(status='training',current=dict(identity=identity,seed=42,epoch=epoch,gain_db=gain,relative=relative));save()
                    prior=previous[gain]
                    args=dict(gain=gain,epoch=epoch,checkpoint=prior['checkpoint_path'] if prior else None,resume=prior['_receipt'] if prior else None,reference_trace=reference)
                    result=remote_action('launch',identity,**args)
                    while result['state'] in ['running','starting']:
                        state['remote_job']=result;save();time.sleep(30)
                        result=remote_action('status',identity,gain=gain,epoch=epoch)
                    state['remote_job']=result;save()
                    assert result['state']=='completed','lost job requires manual review, never replay'
                    jobproof=persist(job)
                    if result['returncode']!=0:
                        # May fail before output creation: preserve job and conditionally any produced epoch.
                        exists=json.loads(remote_python('import json;from pathlib import Path;print(json.dumps(Path(%r).exists()))'%(RUN+'/'+relative)))
                        proof=persist(relative) if exists else None
                        state['failures'].append(dict(identity=identity,epoch=epoch,gain_db=gain,execution=result,job_proof=jobproof,epoch_proof=proof))
                        failed=True;save();break
                    proof=persist(relative)
                    receipt=json.loads((OUT/relative/'epoch_receipt.json').read_text());assert receipt['passed']
                    receipt['_receipt']=RUN+'/'+relative+'/epoch_receipt.json'
                    receipts[gain]=receipt;traces[gain]=OUT/relative/'attempts.jsonl'
                    if gain==0:reference=RUN+'/'+relative+'/attempts.jsonl'
                    retire_previous(prior,receipt);previous[gain]=receipt
                if failed:break
                accepted=qualify_epoch(receipts,traces);epochs.append(accepted)
                target=OUT/'blocks'/identity/f'coarse_ep{epoch:04d}.json';target.parent.mkdir(parents=True,exist_ok=True);atomic_json(target,accepted)
            if not failed:
                phase=qualify_phase(epochs,previous);atomic_json(OUT/'blocks'/identity/'coarse_qualification.json',phase)
                state['completed'].append(phase)
            state.update(status='identity_trial_preserved');save()
            # A terminal diagnostic trial may hand off verified caches without calling it successful training.
            retire(identity)
        state.update(status='all_three_trials_finished',current=None,passed=not state['failures'] and all(x['passed'] for x in state['completed']));save()

if __name__=='__main__':
    try:run()
    except BaseException:
        atomic_json(OUT/'controller_failure.json',dict(pid=os.getpid(),traceback=traceback.format_exc(),state='stopped_no_automatic_replay'));raise
