"""Persist one already launched fine epoch and pause; never launch or replay jobs."""
import hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path

OUT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_fp16_fine_v3')
REV=OUT/'user_pause_v1'
HERE=Path('E:/Documents/tts-exp/scripts/experiments/synctalk_multispeaker_full_20261017')
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('frozen_fine_pause_lifecycle',OUT/'sources/patrick_fine_controller_v3.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


def run():
    request=json.loads((REV/'request.json').read_text(encoding='utf-8'))
    assert request['source_sha256']==controller.store.sha(Path(__file__))
    assert request['gain_db']==6 and request['epoch']==14
    contract=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
    assert request['contract_sha256']==controller.store.sha(OUT/'contract.json')
    for name,digest in contract['sources'].items():assert controller.store.sha(OUT/'sources'/name)==digest
    with controller.exclusive_lock(REV/'observer.lock'):
        assert not (REV/'paused.json').exists()
        state=dict(pid=os.getpid(),status='waiting_pinned_epoch_boundary',request=request,
                   no_new_training_jobs=True,gpu_signals_sent=0,optimizer_replay=False)
        def save():controller.atomic_json(REV/'receipt.json',state)
        save()
        while True:
            execution=controller.remote_status(request['remote_job'])
            assert execution['pid']==request['remote_job_pid']
            assert execution['command_sha256']==request['command_sha256']
            state['execution']=execution;save()
            if execution['state'] not in ('starting','running'):break
            time.sleep(15)
        assert execution['state']=='completed','lost job must be audited, never replayed'
        for relative in [request['relative'].replace('training/','jobs/',1),request['relative']]:
            if not (OUT/relative/'verification.json').exists():controller.store.persist(relative)
            controller.verify_closed(relative)
        assert execution['returncode']==0,'failed epoch retained; no successful pause claim'
        receipts={};traces={};latest={}
        for gain in (0,-6,6):
            relative=f'training/PatrickLeahy/seed42/gain_{gain:+d}/fine/ep0014/attempt01'
            controller.verify_closed(relative)
            receipt=json.loads((OUT/relative/'epoch_receipt.json').read_text(encoding='utf-8'))
            assert receipt['passed'] and receipt['epoch']==14 and receipt['global_step']==84000
            receipts[gain]=receipt;traces[gain]=OUT/relative/'attempts.jsonl'
            latest[str(gain)]=dict(epoch=14,global_step=84000,
                checkpoint_path=receipt['checkpoint_path'],checkpoint_sha256=receipt['checkpoint_sha256'],
                F_checkpoint_path=str(OUT/relative/'workspace/checkpoints/ngp_ep0014.pth'),
                F_receipt_path=str(OUT/relative/'epoch_receipt.json'),
                cumulative_successful_updates=receipt['cumulative_successful_updates'])
        block=controller.qualify_epoch(receipts,traces)
        controller.atomic_json(REV/'fine_ep0014_boundary_qualification.json',block)
        code="from pathlib import Path;import json,subprocess;root=Path(%r);print(json.dumps(dict(future_jobs=[str(x) for x in (root/'jobs').glob('PatrickLeahy/seed42/gain_*/fine/ep00[1][5-7]/attempt01/command.json')],active_trainers=subprocess.run(['pgrep','-af','[t]rain_patrick_fine_epoch_v3.py'],capture_output=True,text=True).stdout)))"%controller.RUN
        remote=json.loads(controller.parent.remote_python(code))
        assert not remote['future_jobs'] and not remote['active_trainers'],remote
        state.update(status='paused_at_verified_epoch_boundary',passed=True,latest_by_gain=latest,
                     remote_no_future_jobs_or_active_trainers=remote,
                     original_training_and_failure_records_preserved=True,
                     parent_control_failed_retained=True,fine17_and_27_grid_not_complete=True,
                     resume_requires_new_audited_handoff=True)
        save();controller.atomic_json(REV/'paused.json',state)


if __name__=='__main__':
    try:run()
    except BaseException:
        controller.atomic_json(REV/'failure.json',dict(status='pause_observation_failed_no_automatic_replay',traceback=traceback.format_exc()));raise
