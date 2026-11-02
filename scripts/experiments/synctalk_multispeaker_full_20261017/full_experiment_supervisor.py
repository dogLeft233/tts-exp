"""Continue authorized work only after all independently verified preparation gates."""
import json,os,subprocess,time,traceback
from pathlib import Path
from execution_journal import atomic_json,exclusive_lock
from persist_prepared import ROOT,REMOTE,SSH,sha
import qualify_all_data,training_coordinator

def run():
    with exclusive_lock(ROOT/'full_experiment_supervisor.lock'):
        source_names=['full_experiment_supervisor.py','qualify_all_data.py','training_coordinator.py','persist_training_epoch.py',
            'execution_journal.py','paired_budget.py','restore_supervision.py','persist_prepared.py']
        captured={name:sha(Path(__file__).with_name(name)) for name in source_names}
        path=ROOT/'full_experiment_supervisor_receipt.json'
        status=dict(pid=os.getpid(),state='waiting_all_four_preparation_gates',training_started=False,passed=False,
            supervisor_source_sha256=sha(Path(__file__)),loaded_source_capsule=captured)
        atomic_json(path,status)
        while True:
            assert not (ROOT/'runtime_qualification_worker_failure.txt').exists(),'runtime qualification worker failed; inspect preserved failure'
            upstream=json.loads(subprocess.check_output(SSH+[f'cat {REMOTE}/stage2_v2_worker_receipt.json']))
            assert upstream.get('passed') is not False,upstream
            file=ROOT/'runtime_qualification_worker_receipt.json'
            if file.exists():
                runtime=json.loads(file.read_text())
                assert not (runtime.get('passed') is False and runtime.get('current_phase')=='failed'),runtime
                if runtime.get('passed') and [x['identity'] for x in runtime['identities']]==qualify_all_data.IDS:break
            time.sleep(30)
        status['state']='independent_all_identity_admission';atomic_json(path,status)
        assert all(sha(Path(__file__).with_name(name))==digest for name,digest in captured.items()),'loaded source changed while waiting; restart with the verified version'
        admission=ROOT/'data_qualification.json'
        if admission.exists():
            qualified=json.loads(admission.read_text());assert qualified['passed'] and qualified['training_ready']
        else:qualified=qualify_all_data.run()
        assert qualified['passed'] and qualified['training_ready'] and qualified['identities']==qualify_all_data.IDS
        assert all(sha(Path(__file__).with_name(name))==digest for name,digest in captured.items())
        status.update(state='official_full_training',training_started=True,admission_sha256=sha(admission));atomic_json(path,status)
        training_coordinator.run()
        trained=json.loads((ROOT/'training_coordinator_receipt.json').read_text());assert trained['passed']
        # Rendering requires its own verified media-capacity handoff, supplied by the live task.
        status.update(state='all_24_models_trained_render_delivery_pending',training_completed=True,passed=False)
        atomic_json(path,status)

if __name__=='__main__':
    try:run()
    except BaseException:
        from datetime import datetime,timezone
        directory=ROOT/'full_experiment_supervisor_failures';directory.mkdir(exist_ok=True)
        target=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.txt')
        target.write_text(traceback.format_exc())
        path=ROOT/'full_experiment_supervisor_receipt.json'
        if path.exists():
            receipt=json.loads(path.read_text());receipt.update(state='failed',passed=False,failure_path=str(target));atomic_json(path,receipt)
        raise
