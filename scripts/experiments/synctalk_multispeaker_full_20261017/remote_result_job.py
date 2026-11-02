"""Detached rendering/scoring jobs reuse the qualified transport, without training updates."""
import argparse,json,sys
from pathlib import Path
from remote_epoch_job import ROOT,IDS,start_job,status

def run():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status']);p.add_argument('--stage',choices=['render','score','qa','pixels'],required=True)
    p.add_argument('--identity',choices=IDS,required=True);p.add_argument('--seed',type=int,choices=[42,2026],required=True);p.add_argument('--gain',type=int,choices=[-6,0,6],default=0);a=p.parse_args()
    directory=ROOT/'result_jobs'/a.identity/f'seed{a.seed}'/a.stage/(f'gain{a.gain:+d}' if a.stage=='render' else 'all27')
    if a.action=='status':print(json.dumps(status(directory)));return
    for seed in [42,2026]:assert json.loads((ROOT/'blocks'/a.identity/f'seed{seed}'/'fine_qualification.json').read_text())['passed']
    if a.stage=='render':
        command=['bash',str(ROOT/'runtime_exec.sh'),str(ROOT/'render_model.py'),'--identity',a.identity,'--seed',str(a.seed),'--train-gain',str(a.gain)]
    elif a.stage=='score':
        command=['env',f'PYTHONPATH={ROOT}','/home/wjj/.venvs/syncnet/bin/python',str(ROOT/'score_identity_seed.py'),'--identity',a.identity,'--seed',str(a.seed)]
    elif a.stage=='qa':
        command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',str(ROOT/'validate_scores.py'),'--identity',a.identity,'--seed',str(a.seed)]
    else:
        command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',str(ROOT/'measure_video_changes.py'),'--identity',a.identity,'--seed',str(a.seed)]
    print(json.dumps(start_job(directory,command)))
if __name__=='__main__':run()
