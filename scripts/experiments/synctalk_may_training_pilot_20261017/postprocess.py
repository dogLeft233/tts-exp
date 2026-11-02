"""Sealed sequential final evaluation, invoked by the training supervisor."""
import json
import os
import subprocess
import sys
import time

from common import ROOT,DATA,sha,save_json


def main():
    admission=json.loads((ROOT/'postprocess_admission.json').read_text())
    assert admission['protocol_sha256']==sha(ROOT/'protocol.json')
    for name,digest in admission['source_sha256'].items():assert sha(ROOT/name)==digest
    assert json.loads((ROOT/'training_complete.json').read_text())['passed']
    assert not (ROOT/'experiment_complete.json').exists()
    for gain in [0,-6,6]:
        save_json(ROOT/'status.json',{'phase':'rendering_test_and_full_validation','training_gain_db':gain})
        subprocess.run([sys.executable,'-u',str(ROOT/'render_test.py'),'--train-gain',str(gain)],check=True)
    save_json(ROOT/'status.json',{'phase':'scoring_original_pcm'})
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT)
    subprocess.run(['/home/wjj/.venvs/syncnet/bin/python','-u',str(ROOT/'score_test.py')],env=env,check=True)
    save_json(ROOT/'status.json',{'phase':'independent_results_qa'})
    subprocess.run([sys.executable,'-u',str(ROOT/'validate_results.py'),'--root',str(ROOT),'--data',str(DATA)],check=True)
    subprocess.run([sys.executable,'-u',str(ROOT/'analyze_plot.py')],check=True)
    qa=json.loads((ROOT/'independent_results_qa.json').read_text());assert qa['passed']
    save_json(ROOT/'experiment_complete.json',{'passed':True,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'protocol_sha256':sha(ROOT/'protocol.json'),'qa_sha256':sha(ROOT/'independent_results_qa.json'),
        'analysis_sha256':sha(ROOT/'analysis/analysis.json'),'training_models':3,'steps_each':60000,'test_cells':9,
        'persistent_backup_verified':False,'note':'F-disk and cloud delivery are separate receipts; SHM is volatile'})
    save_json(ROOT/'status.json',{'phase':'evaluation_complete_awaiting_persistent_delivery','passed':True})


if __name__=='__main__':main()
