"""Resume immutable jobs by receipts; reserve first completed identity for preview."""
import importlib.util,json,os,sys,time,traceback
from pathlib import Path
HERE=Path('E:/Documents/tts-exp/scripts/experiments/synctalk_multispeaker_full_20261017')
OUT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1')
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('preview_handoff_helpers',OUT/'controller_revision02/other_identities_controller_handoff_v2.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);t=h.trial

def verified_copy(relative):
    folder=OUT/relative
    proof=json.loads((folder/'verification.json').read_text())
    assert proof['passed'] and proof['all_archived_member_bytes_verified']
    assert t.sha(folder/'manifest.json')==proof['manifest_sha256']
    assert t.sha(folder/'artifacts.tar.gz')==proof['archive_sha256']
    rows={x['path']:x for x in json.loads((folder/'manifest.json').read_text())}
    for name in ['execution.json','epoch_receipt.json','attempts.jsonl']:
        if name in rows:assert t.sha(folder/name)==rows[name]['sha256']
    return proof

def job_route(has_persisted_epoch,remote_state):
    if has_persisted_epoch:return 'reuse_verified'
    if remote_state is None:return 'launch_new'
    assert remote_state in ['starting','running','completed'],'lost job must never replay'
    return 'observe_existing'

def preview_ready(phase,receipts):
    # A complete but unequal-update phase remains CONTROL_FAILED, eligible only for descriptive preview.
    return phase['identity']=='PatrickLeahy' and set(receipts)=={0,-6,6} and all(r['epoch']==10 and r['passed'] for r in receipts.values())

def run():
    rev=OUT/'controller_revision03';pin=json.loads((rev/'handoff.json').read_text())
    assert t.sha(Path(__file__))==pin['source_sha256']
    contract=json.loads((OUT/'contract.json').read_text())
    assert t.sha(OUT/'contract.json')==pin['contract_sha256']
    for name,digest in contract['sources'].items():assert t.sha(OUT/'sources'/name)==digest
    assert json.loads((OUT/'qualification/result.json').read_text())['passed']
    with t.exclusive_lock(OUT/'controller.lock'):
        assert not (OUT/'receipt.json').exists()
        state=dict(pid=os.getpid(),version=t.VERSION,controller_revision=3,status='recovering_verified_receipts',completed=[],failures=[],current=None,formal_original_models_completed=0)
        def save():t.atomic_json(OUT/'receipt.json',state)
        save()
        for identity in contract['identities']:
            restored=OUT/'storage'/f'{identity}_full_restore.json'
            retired=OUT/'storage'/f'{identity}_cache_retirement.json'
            assert not retired.exists(),'this handoff does not resume already retired identities'
            if restored.exists():
                proof=json.loads(restored.read_text());assert proof['passed'] and proof['verified_files']==16500
            else:
                state.update(status='restoring_full_supervision',current={'identity':identity});save();t.restore(identity)
            previous={g:None for g in [0,-6,6]};epochs=[];failed=False
            for epoch in range(1,11):
                receipts={};traces={};reference=None
                for gain in [0,-6,6]:
                    relative=f'training/{identity}/seed42/gain_{gain:+d}/coarse/ep{epoch:04d}/attempt01'
                    job=relative.replace('training/','jobs/',1);prior=previous[gain]
                    state.update(status='training',current=dict(identity=identity,seed=42,epoch=epoch,gain_db=gain,relative=relative));save()
                    persisted=(OUT/relative/'verification.json').exists()
                    if persisted:
                        verified_copy(relative);verified_copy(job)
                        result=json.loads((OUT/job/'execution.json').read_text());assert result['state']=='completed'
                    else:
                        code='import json;from pathlib import Path;print(json.dumps(Path(%r).exists()))'%(t.RUN+'/'+job+'/command.json')
                        exists=json.loads(t.remote_python(code))
                        result=t.remote_action('status',identity,gain=gain,epoch=epoch) if exists else None
                        route=job_route(False,result['state'] if result else None)
                        if route=='launch_new':
                            result=t.remote_action('launch',identity,gain=gain,epoch=epoch,checkpoint=prior['checkpoint_path'] if prior else None,resume=prior['_receipt'] if prior else None,reference_trace=reference)
                        while result['state'] in ['running','starting']:
                            state['remote_job']=result;save();time.sleep(30)
                            result=t.remote_action('status',identity,gain=gain,epoch=epoch)
                        assert result['state']=='completed','lost job requires manual audit'
                        if (OUT/job/'verification.json').exists():verified_copy(job)
                        else:t.persist(job)
                        exists=json.loads(t.remote_python('import json;from pathlib import Path;print(json.dumps(Path(%r).exists()))'%(t.RUN+'/'+relative)))
                        if exists:t.persist(relative)
                    state['remote_job']=result;save()
                    if result['returncode']!=0:
                        state['failures'].append(dict(identity=identity,epoch=epoch,gain_db=gain,execution=result));failed=True;save();break
                    receipt=json.loads((OUT/relative/'epoch_receipt.json').read_text());assert receipt['passed']
                    receipt['_receipt']=t.RUN+'/'+relative+'/epoch_receipt.json'
                    receipts[gain]=receipt;traces[gain]=OUT/relative/'attempts.jsonl'
                    if gain==0:reference=t.RUN+'/'+relative+'/attempts.jsonl'
                    if not persisted:h.retire_previous(prior,receipt)
                    previous[gain]=receipt
                if failed:break
                accepted=t.qualify_epoch(receipts,traces);epochs.append(accepted)
                target=OUT/'blocks'/identity/f'coarse_ep{epoch:04d}.json';target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert json.loads(target.read_text())==accepted
                else:t.atomic_json(target,accepted)
            if not failed:
                phase=t.qualify_phase(epochs,previous);target=OUT/'blocks'/identity/'coarse_qualification.json'
                if target.exists():assert json.loads(target.read_text())==phase
                else:t.atomic_json(target,phase)
                state['completed'].append(phase)
                if preview_ready(phase,previous):
                    state.update(status='waiting_first_identity_preview',current=dict(identity=identity,phase='coarse',epoch=10));save()
                    t.atomic_json(rev/'preview_ready.json',dict(identity=identity,phase=phase,receipt_paths=[previous[g]['_receipt'] for g in [0,-6,6]],no_training_job_active=True))
                    release=rev/'preview_delivery_release.json'
                    while not release.exists():time.sleep(30)
                    delivery=json.loads(release.read_text())
                    assert delivery['identity']==identity and delivery['passed'] and delivery['independent_score_qa_passed'] and delivery['all_persistent_bytes_verified']
                    assert t.sha(Path(delivery['receipt_path']))==delivery['receipt_sha256']
            state.update(status='identity_trial_preserved');save();t.retire(identity)
        state.update(status='all_three_trials_finished',current=None,passed=not state['failures'] and all(x['passed'] for x in state['completed']));save()

if __name__=='__main__':
    try:run()
    except BaseException:
        t.atomic_json(OUT/'controller_revision03/controller_failure.json',dict(pid=os.getpid(),state='stopped_no_automatic_replay',traceback=traceback.format_exc()));raise
