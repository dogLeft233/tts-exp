"""Scheduling-only priority override; preserve frozen generation/scoring code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import cloud_av_preference_20261027 as experiment


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(command=(proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                state=stat[0], start_ticks=stat[19])


def same_paused_runner(identity, expected):
    return (identity['state'] in ['T', 't'] and identity['start_ticks'] == expected['start_ticks']
            and identity['command'] == expected['command'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    authorization = experiment.read(out / 'cloud_priority_authorization_20261027.json')
    pid = authorization['older_runner_pid']
    expected = process_identity(pid)
    assert expected['command'] == authorization['older_command']
    assert same_paused_runner(expected, expected)
    receipt = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), older_pid=pid,
        older_identity=expected, scheduling_only=True, scoring_functions_unchanged=True,
        poll_seconds=5, idle_checks_required=3, cloud_pid=os.getpid())
    experiment.put(out / 'priority_execution_receipt.json', receipt)

    def wait_gpu(target, stage):
        checks = 0
        while True:
            assert same_paused_runner(process_identity(pid), expected), 'Older runner unexpectedly resumed or replaced'
            memory, util = map(int, experiment.capture(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits']).decode().strip().split(','))
            processes = experiment.capture(['nvidia-smi', '--query-compute-apps=pid,process_name',
                '--format=csv,noheader']).decode().strip()
            foreign = [p for p in processes.splitlines() if p and 'gnome-remote-desktop' not in p]
            production = experiment.active_production_commands(experiment.capture(['ps', '-eo', 'stat=,args=']).decode().splitlines())
            checks = checks + 1 if memory < 1024 and util < 15 and not foreign and not production else 0
            state = 'RUNNING_GPU' if checks >= 3 else 'WAITING_GPU'
            experiment.put(target / 'status.json', dict(stage=state, next_stage=stage, memory_mib=memory,
                utilization=util, older_cross_runner_paused=True, priority_cloud=True, pid=os.getpid(), time=time.time()))
            if checks >= 3:
                return
            time.sleep(5)

    experiment.wait_gpu = wait_gpu
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        experiment.execute(out)
    finally:
        identity = process_identity(pid)
        assert same_paused_runner(identity, expected), 'Cannot safely resume changed older runner'
        os.kill(pid, signal.SIGCONT)
        experiment.put(out / 'older_cross_runner_resume.json', dict(pid=pid, start_ticks=expected['start_ticks'],
            resumed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
