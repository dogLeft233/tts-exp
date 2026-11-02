"""Wait for the two unique candidates, independently admit, then execute the spec."""
import json
import os
import time
import traceback
from pathlib import Path
from cohort_binding import FROOT, VERSION, sha
from execution_journal import atomic_json, exclusive_lock
import qualify_cohort_version
import training_coordinator
import render_delivery_supervisor


def run():
    with exclusive_lock(FROOT/'versioned_experiment_supervisor.lock'):
        names=sorted(set(qualify_cohort_version.EXECUTING+qualify_cohort_version.CONTROLLERS+[
            'render_delivery_supervisor.py','render_model.py','score_identity_seed.py','validate_scores.py',
            'measure_video_changes.py','seal_media_delivery.py','retire_test_supervision.py',
            'persist_media_delivery.py','cloud_archive_stream.py','cloud_archive_bounded.py','cloud_archive_readback_stream.py',
            'assemble_results.py','qualify_aggregate_delivery.py','analyze_factorial_gains.py','analysis_contract_v1.json']))
        captured={name:sha(Path(__file__).with_name(name)) for name in names}
        path=FROOT/'versioned_experiment_supervisor_receipt.json'
        assert not path.exists(), 'inspect original supervisor before any restart; never replay updates'
        status=dict(pid=os.getpid(),state='waiting_unique_candidate_qualification',cohort_version=VERSION,
                    training_started=False,passed=False,loaded_source_capsule=captured)
        atomic_json(path,status)
        try:
            while not qualify_cohort_version.candidate_ready():
                assert all(sha(Path(__file__).with_name(n))==h for n,h in captured.items()), 'loaded source drift'
                time.sleep(30)
            status['state']='fresh_versioned_four_identity_admission';atomic_json(path,status)
            assert all(sha(Path(__file__).with_name(n))==h for n,h in captured.items())
            admission=qualify_cohort_version.run()
            assert admission['passed'] and admission['training_ready'] and admission['cohort_version']==VERSION
            assert all(sha(Path(__file__).with_name(n))==h for n,h in captured.items())
            status.update(state='official_24_model_training',training_started=True,
                          admission_sha256=sha(FROOT/'data_qualification.json'));atomic_json(path,status)
            training_coordinator.run()
            assert json.loads((FROOT/'training_coordinator_receipt.json').read_text())['passed']
            status['state']='three_driver_three_scorer_delivery';atomic_json(path,status)
            render_delivery_supervisor.run()
            # The human-readable scientific report and final persistent QA are
            # deliberately explicit remaining work, even when all media passes.
            status['state']='actual_results_delivered_final_scientific_review_pending';atomic_json(path,status)
        except BaseException:
            status.update(state='failed',passed=False,error=traceback.format_exc());atomic_json(path,status)
            raise


if __name__=='__main__':run()
