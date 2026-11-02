"""Persistent bounded phase wrapper, preserving real process exit codes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run_phase(stage,run,command=None,timeout=None):
    run=Path(run)
    receipt=run/(stage+'_exit.json')
    if receipt.exists():raise FileExistsError(receipt)
    if command is None:
        snapshot=run/'source_snapshots'/stage
        snapshot.mkdir(parents=True,exist_ok=False)
        pins={}
        for name in ('common.py','pipeline.py','run_job.py'):
            source=Path(__file__).with_name(name)
            shutil.copy2(source,snapshot/name)
            pins[name]=hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
        (snapshot/'source_manifest.json').write_text(json.dumps(pins,indent=2))
        command=[sys.executable,'-u',str(snapshot/'pipeline.py'),stage,'--run',str(run)]
    started=time.time()
    with (run/(stage+'.log')).open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        launch=dict(stage=stage,pid=process.pid,wrapper_pid=__import__('os').getpid(),
                    command=command,started_unix=started,hard_timeout_s=timeout or (21600 if stage=='train' else 7200))
        (run/(stage+'_launch.json')).write_text(json.dumps(launch,indent=2))
        try:
            code=process.wait(timeout=launch['hard_timeout_s'])
            result=dict(**launch,exit_code=code,ended_unix=time.time(),status='PASS' if code==0 else 'CRASHED')
        except subprocess.TimeoutExpired:
            log.write('\nHARD_TIMEOUT: terminating this task phase only\n');log.flush()
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            result=dict(**launch,exit_code=process.returncode,ended_unix=time.time(),status='HARD_TIMEOUT')
    receipt.write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['cache','smoke','smoke_revision02','train','infer'])
    p.add_argument('--run',required=True,type=Path)
    args=p.parse_args()
    result=run_phase(args.stage,args.run)
    raise SystemExit(0 if result['status']=='PASS' else 1)
