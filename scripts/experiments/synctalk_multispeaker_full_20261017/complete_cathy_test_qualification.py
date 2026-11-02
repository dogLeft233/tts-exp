"""Preparation-only, independent F delivery and all remaining qualification for one revision."""
import argparse,hashlib,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
from execution_journal import atomic_json,exclusive_lock
import persist_prepared as prepared
import persist_runtime_qualification as runtime
import restore_supervision as restoration
import qualify_all_data as independent
BASE=prepared.ROOT;REMOTE_BASE=prepared.REMOTE;REVISION='CathyMcMorrisRodgers_recording2_v2';IDENTITY='CathyMcMorrisRodgers'
ROOT=BASE/'dataset_revisions'/REVISION;REMOTE=REMOTE_BASE+'/dataset_revisions/'+REVISION
SSH=prepared.SSH;PY='/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python'
SYNC_PY='/home/wjj/.venvs/syncnet/bin/python'
CAPSULE=None
WORKER_RECEIPT='preparation_worker_receipt.json'
def check_sources():
    code="import hashlib,json;from pathlib import Path;r=Path(%r);names=%r;print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}))"%(REMOTE,list(CAPSULE['remote_sources']))
    actual=json.loads(subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)]))
    assert actual==CAPSULE['remote_sources'],'versioned qualification source drift'
    for name,digest in CAPSULE['local_sources'].items():assert prepared.sha(Path(__file__).with_name(name))==digest
def read(relative,optional=False):
    result=subprocess.run(SSH+['cat '+shlex.quote(REMOTE+'/'+relative)],capture_output=True)
    if result.returncode and optional:return None
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout)
def copy_to_remote(source,relative):
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',str(source),prepared.HOST+':'+REMOTE+'/'+relative],check=True)
def wait(predicate):
    deadline=time.monotonic()+43200
    while not predicate():
        worker=read(WORKER_RECEIPT,True);assert not worker or worker.get('passed') is not False,worker
        assert time.monotonic()<deadline;time.sleep(30)
def job(name,arguments,target,interpreter=PY):
    check_sources()
    assert read(target,True) is None,'never overwrite or replay this qualification attempt'
    with (ROOT/(name+'_transport.log')).open('xb') as log:
        subprocess.run(SSH+[shlex.join(['flock','-x',REMOTE_BASE+'/preparation_GPU.lock',interpreter,*arguments])],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(target);assert result['passed'],result
    return result
def persist_inputs(kind):
    import tarfile
    subprocess.run(SSH+[shlex.join([PY,REMOTE+'/seal_trial_inputs.py','--root',REMOTE,'--identity',IDENTITY,'--kind',kind])],check=True)
    source=REMOTE+'/input_snapshots/'+kind
    out=ROOT/(kind+'_packages')/IDENTITY;out.mkdir(parents=True,exist_ok=False)
    raw=subprocess.check_output(SSH+['cat '+source+'/manifest.json']);manifest=json.loads(raw);(out/'manifest.json').write_bytes(raw)
    archive=out/(kind+'.tar.gz')
    with archive.open('xb') as stream:
        subprocess.run(SSH+[f'tar --hard-dereference -czf - -C {REMOTE} --null -T {source}/files.nul'],stdout=stream,check=True)
        stream.flush();os.fsync(stream.fileno())
    files={r['path']:r for r in manifest['files']};seen=set()
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            assert member.isfile() and member.name in files and member.name not in seen
            row=files[member.name];assert row['bytes']==member.size
            data=tf.extractfile(member);h=hashlib.sha256()
            # Retain material PCM/metadata and stage1 receipt locally; full videos/images remain in verified archive.
            keep=(kind=='material' and not member.name.endswith('.mkv')) or (kind=='stage1' and member.name.endswith('/stage1_receipt.json'))
            target=ROOT/member.name;assert target.resolve().is_relative_to(ROOT.resolve())
            if keep:target.parent.mkdir(parents=True,exist_ok=True);output=target.open('xb')
            else:output=None
            try:
                for block in iter(lambda:data.read(4*1024**2),b''):
                    h.update(block)
                    if output:output.write(block)
            finally:
                if output:output.close()
            assert h.hexdigest()==row['sha256'];seen.add(member.name)
    assert seen==files.keys()
    proof=dict(passed=True,identity=IDENTITY,files=len(seen),all_archived_member_bytes_verified=True,
        archive=str(archive),archive_sha256=prepared.sha(archive),manifest_sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=sum(r['bytes'] for r in files.values()),training_admitted=False,dataset_revision=REVISION)
    atomic_json(out/'verification.json',proof);return proof
def run(attempt=1,worker_receipt='preparation_worker_receipt.json'):
    global CAPSULE,WORKER_RECEIPT
    assert attempt>0 and Path(worker_receipt).name==worker_receipt;WORKER_RECEIPT=worker_receipt
    with exclusive_lock(ROOT/'post_geometry_qualification.lock'):
        suffix='' if attempt==1 else f'_attempt{attempt:02d}'
        CAPSULE=json.loads((ROOT/('post_geometry_source_capsule'+suffix+'.json')).read_text());check_sources()
        path=ROOT/('post_geometry_worker'+suffix+'_receipt.json');assert not path.exists()
        status=dict(pid=os.getpid(),state='waiting_geometry_qualified',passed=None,training_admitted=False,worker_can_start_training=False,checks=[],source_capsule=CAPSULE)
        def save():atomic_json(path,status)
        save()
        try:
            wait(lambda:(read(WORKER_RECEIPT,True) or {}).get('passed') is True)
            independent.qualify_geometry(read('prepared/'+IDENTITY+'/geometry_v2/geometry_qualification.json'))
            for kind in ['material','stage1']:
                status['state']='persist_'+kind;save();status['checks'].append(persist_inputs(kind));save()
            status['state']='persist_and_retire_training_dense_flow';save()
            flow_path=ROOT/'flow_cache_persistence_receipt.json'
            if flow_path.exists():
                deadline=time.monotonic()+43200
                while not json.loads(flow_path.read_text())['passed']:
                    status_flow=json.loads(flow_path.read_text());assert status_flow['state']!='failed',status_flow
                    assert time.monotonic()<deadline;time.sleep(30)
            else:
                with (ROOT/'train_flow_persistence_transport.log').open('xb') as log:
                    subprocess.run([sys.executable,str(ROOT/'persist_flow_trial.py'),'--identities',IDENTITY,'--splits','train','--worker-receipt',WORKER_RECEIPT],stdout=log,stderr=subprocess.STDOUT,check=True)
            flow=json.loads((ROOT/'flow_cache_persistence_receipt.json').read_text());assert flow['passed']
            for row in flow['verified']:assert row['decompressed_source_verified'] and row['reclamation']['passed'] and prepared.sha(Path(row['path']))==row['gzip_sha256']
            status['state']='supervision_and_AVE';save()
            wait(lambda:float(subprocess.check_output(SSH+["python3 -c "+shlex.quote("import shutil;print(shutil.disk_usage(%r).free)"%REMOTE)]))>2*1024**3)
            for name,target in [('supervision.py','prepared/'+IDENTITY+'/supervision_receipt.json'),('ave_cache.py','features/'+IDENTITY+'/ave_qualification.json')]:
                job(name,[REMOTE+'/'+name,'--root',REMOTE,'--identity',IDENTITY],target)
            status['state']='new_test_ROI_and_scoring_reference';save()
            job('score_reference',[REMOTE+'/prepare_score_reference.py','--root',REMOTE,'--identity',IDENTITY],'score_reference/'+IDENTITY+'/score_reference_receipt.json',SYNC_PY)
            job('independent_reference',[REMOTE+'/qualify_score_references_trial.py'],'score_reference/'+IDENTITY+'/independent_reference_qualification.json',SYNC_PY)
            prepared.ROOT=ROOT;prepared.REMOTE=REMOTE;runtime.ROOT=ROOT;runtime.REMOTE=REMOTE
            restoration.ROOT=ROOT;restoration.REMOTE=REMOTE;independent.ROOT=ROOT
            for kind in ['derived','qualification']:
                status['state']='persist_'+kind;save()
                subprocess.run(SSH+[shlex.join([PY,REMOTE+'/prepare_snapshot_manifest.py','--root',REMOTE,'--identity',IDENTITY,'--kind',kind])],check=True)
                prepared.run(IDENTITY,kind)
            proof=ROOT/'derived_packages'/IDENTITY/'verification.json';copy_to_remote(proof,'snapshots/'+IDENTITY+'/F_archive_verification.json')
            job('retire_supervision',[REMOTE+'/reclaim_derived_supervision.py','--root',REMOTE,'--identity',IDENTITY,'--proof',REMOTE+'/snapshots/'+IDENTITY+'/F_archive_verification.json'],'snapshots/'+IDENTITY+'/supervision_reclamation.json')
            restoration.run(IDENTITY,'control')
            job('dataset_binding',[REMOTE+'/build_datasets.py','--root',REMOTE,'--identity',IDENTITY,'--persisted-supervision'],'datasets/'+IDENTITY+'/binding_receipt.json')
            for stage in ['loading','capacity']:
                status['state']=stage+'_qualification';save()
                job(stage,[REMOTE+'/runtime_probe_trial.py','--root',REMOTE,'--stage',stage],stage+'_qualification/'+IDENTITY+'/'+stage+'_qualification.json')
            runtime_proof=runtime.run(IDENTITY,1)
            qdir=ROOT/'qualification_state';qdir.mkdir(exist_ok=True)
            raw=subprocess.check_output(SSH+['cat '+REMOTE+'/score_reference/'+IDENTITY+'/independent_reference_qualification.json'])
            qpath=qdir/(IDENTITY+'_independent_reference_qualification.json');qpath.write_bytes(raw)
            assert prepared.sha(qpath)==hashlib.sha256(raw).hexdigest()
            atomic_json(qdir/'independent_reference_persistence.json',dict(passed=True,files=[dict(identity=IDENTITY,path=str(qpath),sha256=prepared.sha(qpath))]))
            result=independent.qualify_identity(IDENTITY,runtime_proof)
            out=ROOT/'data_identity_preflight'/IDENTITY;out.mkdir(parents=True,exist_ok=False);atomic_json(out/'verification.json',result)
            status.update(state='revision_independently_qualified_pending_cohort_version_admission',passed=True,runtime_proof=runtime_proof);save()
        except BaseException:
            status.update(state='failed',passed=False,error=traceback.format_exc());save();raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=1);p.add_argument('--worker-receipt',default='preparation_worker_receipt.json');a=p.parse_args();run(a.attempt,a.worker_receipt)
