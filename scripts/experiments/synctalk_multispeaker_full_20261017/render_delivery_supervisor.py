"""After all24 accepted models, render/score all27 cells and independently cold-deliver each seed."""
import json,os,shlex,subprocess,time,traceback
from pathlib import Path
from execution_journal import atomic_json,exclusive_lock
from persist_prepared import ROOT,REMOTE,SSH,sha
from restore_cohort_supervision import run as restore
from cohort_binding import identity_root,expected_root,REMOTE as REMOTE_PATH,VERSION
from persist_media_delivery import run as persist
from cloud_archive_bounded import run as cloud_backup
IDS=['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']
STATUS_PATH=ROOT/f'render_delivery_supervisor_{VERSION}_receipt.json'

def retire_completed_model_copies(identity,seed,delivery,cloud):
    assert cloud['passed'] and cloud['independent_linux_cloud_readback'] and cloud['sha256']==delivery['archive_sha256']
    manifest_path=ROOT/'result_delivery'/identity/f'seed{seed}'/'manifest.json'
    assert sha(manifest_path)==delivery['manifest_sha256']
    archived={row['path']:row for row in json.loads(manifest_path.read_text())['files']}
    targets=[]
    for gain in [-6,0,6]:
        for phase,epoch in [('coarse',10),('fine',17)]:
            directory=ROOT/'training'/identity/f'seed{seed}'/f'gain_{gain:+d}'/phase/f'ep{epoch:04d}/attempt01'
            proof=json.loads((directory/'verification.json').read_text());assert proof['passed']
            # Remote paths need POSIX parsing on Windows.
            from pathlib import PurePosixPath
            relative=str(PurePosixPath(proof['remote_checkpoint']).relative_to(PurePosixPath(REMOTE)))
            assert archived[relative]['sha256']==proof['checkpoint_sha256']
            file=Path(proof['checkpoint']);assert file.resolve().is_relative_to((ROOT/'training'/identity/f'seed{seed}').resolve())
            marker=directory/'checkpoint_cloud_retirement.json'
            if marker.exists():
                old=json.loads(marker.read_text());assert old['passed'] and old['archive_sha256']==cloud['sha256'];continue
            intent=directory/'checkpoint_cloud_retirement_intent.json'
            if file.exists():assert sha(file)==proof['checkpoint_sha256']
            else:
                prior=json.loads(intent.read_text());assert prior['checkpoint_sha256']==proof['checkpoint_sha256'] and prior['archive_sha256']==cloud['sha256']
            targets.append((file,marker,intent,proof['checkpoint_sha256']))
    for file,marker,intent,digest in targets:
        atomic_json(intent,dict(checkpoint_sha256=digest,archive_sha256=cloud['sha256']))
        if file.exists():file.unlink()
        atomic_json(marker,dict(passed=True,checkpoint_sha256=digest,archive_sha256=cloud['sha256'],
            independent_cloud_archive_contains_complete_original_state=True,final_EMA_local_copy_and_all_training_receipts_retained=True))

def remote_json(path,optional=False):
    result=subprocess.run(SSH+['cat '+shlex.quote(REMOTE+'/'+path)],capture_output=True)
    if optional and result.returncode:return None
    assert result.returncode==0,result.stderr;return json.loads(result.stdout)

def publish(path,raw):
    destination=REMOTE+'/'+path
    code="from pathlib import Path;import sys;p=Path(%r);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(sys.stdin.buffer.read())"%destination
    subprocess.run(SSH+['python3 -c '+shlex.quote(code)],input=raw,check=True)

def wait_job(identity,seed,stage,gain=0):
    arguments=['--identity',identity,'--seed',str(seed),'--stage',stage,'--gain',str(gain)]
    command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',f'{REMOTE}/remote_result_job.py']
    state=json.loads(subprocess.check_output(SSH+[shlex.join(command+['start']+arguments)]))
    while state['state'] in ['starting','running']:
        time.sleep(30);state=json.loads(subprocess.check_output(SSH+[shlex.join(command+['status']+arguments)]))
    assert state['state']=='completed' and state['passed'],state

def run():
    with exclusive_lock(ROOT/'render_delivery_supervisor.lock'):
        sources=['cohort_binding.py','restore_cohort_supervision.py','render_delivery_supervisor.py','remote_result_job.py','render_model.py','score_identity_seed.py','validate_scores.py',
            'measure_video_changes.py','seal_media_delivery.py','persist_media_delivery.py','retire_result_cache.py','retire_test_supervision.py','cloud_archive.py','cloud_archive_readback.py','cloud_archive_stream.py','cloud_archive_bounded.py','cloud_archive_readback_stream.py','assemble_results.py','qualify_aggregate_delivery.py','analyze_factorial_gains.py','analysis_contract_v1.json']
        captured={name:sha(Path(__file__).with_name(name)) for name in sources}
        path=STATUS_PATH
        status=dict(pid=os.getpid(),state='waiting_all_24_qualified_models',passed=False,delivered=[],loaded_source_capsule=captured)
        atomic_json(path,status)
        while True:
            training=ROOT/'training_coordinator_receipt.json'
            if training.exists() and json.loads(training.read_text())['passed']:break
            supervisor=ROOT/'versioned_experiment_supervisor_receipt.json'
            if supervisor.exists():assert json.loads(supervisor.read_text())['state']!='failed','training supervisor failed; no result generation'
            time.sleep(30)
        for identity in IDS:
            inputs=identity_root(identity,ROOT,require_admission=True);remote_inputs=expected_root(REMOTE_PATH,identity).as_posix()
            remote_sources=[name for name in sources if name not in ['restore_cohort_supervision.py','render_delivery_supervisor.py','persist_media_delivery.py','cloud_archive.py','cloud_archive_stream.py','cloud_archive_bounded.py']]
            code="import hashlib,json;from pathlib import Path;r=Path(%r);names=%r;print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}))"%(REMOTE,remote_sources)
            actual=json.loads(subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)]))
            assert all(actual[name]==captured[name] for name in remote_sources),'executing result source differs from captured local source'
            test_restore=inputs/'derived_packages'/identity/'test_restoration_verification.json'
            if test_restore.exists():assert json.loads(test_restore.read_text())['passed']
            else:restore(identity,'test')
            for seed in [42,2026]:
                assert all(sha(Path(__file__).with_name(name))==digest for name,digest in captured.items()),'loaded result source changed'
                status.update(state='rendering_and_scoring',identity=identity,seed=seed);atomic_json(path,status)
                for gain in [-6,0,6]:wait_job(identity,seed,'render',gain)
                for stage in ['score','qa','pixels']:wait_job(identity,seed,stage)
                manifest=remote_json(f'media_manifests/{identity}/seed{seed}/manifest.json',optional=True)
                if manifest is None:
                    command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',f'{REMOTE}/seal_media_delivery.py','--identity',identity,'--seed',str(seed)]
                    subprocess.run(SSH+[shlex.join(command)],check=True)
                status['state']='F_full_member_readback';atomic_json(path,status)
                proof=persist(identity,seed);out=ROOT/'result_delivery'/identity/f'seed{seed}'
                cloud_name=f'{identity}-seed{seed}';cloud_dir=ROOT/'cloud_delivery'/cloud_name/'transport_attempt01'
                cloud_receipt=cloud_dir/'cloud_readback_receipt.json'
                if cloud_receipt.exists():
                    receipt=json.loads(cloud_receipt.read_text());assert receipt['passed'] and receipt['sha256']==proof['archive_sha256']
                else:
                    status['state']='independent_Baidu_readback';atomic_json(path,status)
                    receipt=cloud_backup(Path(proof['archive']),proof['archive_sha256'],cloud_name,1)
                for filename,raw in [('F_delivery_verification.json',json.dumps(proof,indent=2).encode()),('cloud_readback_receipt.json',cloud_receipt.read_bytes())]:
                    relative=f'media_manifests/{identity}/seed{seed}/{filename}'
                    if remote_json(relative,optional=True) is None:publish(relative,raw)
                command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',f'{REMOTE}/retire_result_cache.py','--identity',identity,'--seed',str(seed)]
                subprocess.run(SSH+[shlex.join(command)],check=True)
                archive=Path(proof['archive'])
                if archive.exists():
                    assert archive.resolve().is_relative_to((ROOT/'result_delivery').resolve()) and sha(archive)==receipt['sha256']
                    for row in proof['local_files']:assert sha(out/'local'/row['path'])==row['sha256']
                    archive.unlink()
                    atomic_json(out/'archive_cache_retirement.json',dict(passed=True,sha256=receipt['sha256'],cloud_receipt_path=str(cloud_receipt),
                        local_scores_arrays_and_previews_retained=True,independent_linux_cloud_readback=True))
                retire_completed_model_copies(identity,seed,proof,receipt)
                status['delivered'].append(dict(identity=identity,seed=seed,archive_sha256=proof['archive_sha256'],passed=True));atomic_json(path,status)
            package=inputs/'derived_packages'/identity;original=json.loads((package/'verification.json').read_text())
            assert original['passed'] and sha(Path(original['archive']))==original['archive_sha256']
            if identity!=IDS[-1]:
                command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',f'{REMOTE}/retire_test_supervision.py','--identity',identity]
            else:
                command=['bash',f'{REMOTE}/runtime_exec.sh',f'{REMOTE}/reclaim_derived_supervision.py','--root',remote_inputs,'--identity',identity,
                    '--proof',f'{remote_inputs}/snapshots/{identity}/F_archive_verification.json','--report-name','post_training_supervision_reclamation.json']
            relative_inputs=str(expected_root(REMOTE_PATH,identity).relative_to(REMOTE_PATH))
            prefix='' if relative_inputs=='.' else relative_inputs+'/'
            existing=remote_json(prefix+f'snapshots/{identity}/post_training_supervision_reclamation.json',optional=True) if identity==IDS[-1] else None
            if existing is not None:assert existing['passed']
            else:subprocess.run(SSH+[shlex.join(command)],check=True)
        status['state']='equal_identity_tables_and_learning_curves';atomic_json(path,status)
        analysis=remote_json('delivery/analysis_receipt.json',optional=True)
        if analysis is None:
            command=['bash',f'{REMOTE}/runtime_exec.sh',f'{REMOTE}/assemble_results.py'];subprocess.run(SSH+[shlex.join(command)],check=True)
        else:assert analysis['passed']
        overall=remote_json('delivery/independent_aggregate_qualification.json',optional=True)
        if overall is None:
            command=['bash',f'{REMOTE}/runtime_exec.sh',f'{REMOTE}/qualify_aggregate_delivery.py'];subprocess.run(SSH+[shlex.join(command)],check=True)
            overall=remote_json('delivery/independent_aggregate_qualification.json')
        assert overall['passed'] and overall['combined_cells']==27 and overall['learning_curve_rows']==408
        interpretation=remote_json('delivery/interpretation_receipt.json',optional=True)
        if interpretation is None:
            command=['bash',f'{REMOTE}/runtime_exec.sh',f'{REMOTE}/analyze_factorial_gains.py'];subprocess.run(SSH+[shlex.join(command)],check=True)
            interpretation=remote_json('delivery/interpretation_receipt.json')
        assert interpretation['passed']
        status.update(state='all27_delivered_aggregate_QA_and_descriptive_analysis_passed_final_scientific_review_pending',passed=False)
        atomic_json(path,status)

if __name__=='__main__':
    try:run()
    except BaseException:
        from datetime import datetime,timezone
        directory=ROOT/'render_delivery_supervisor_failures';directory.mkdir(exist_ok=True)
        target=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.txt');target.write_text(traceback.format_exc())
        path=STATUS_PATH
        if path.exists():
            receipt=json.loads(path.read_text());receipt.update(state='failed',passed=False,failure_path=str(target));atomic_json(path,receipt)
        raise
