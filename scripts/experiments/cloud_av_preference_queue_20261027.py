"""Launch a detached cloud->GPU pipeline without persisting credentials."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    key = sys.stdin.readline().strip()
    if not key:
        raise RuntimeError('Missing runtime credential')
    if not args.worker:
        with (out / 'pipeline.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--out', str(out), '--worker'],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, text=True)
        child.stdin.write(key + '\n')
        child.stdin.close()
        del key
        (out / 'pipeline.pid').write_text(str(child.pid))
        print(json.dumps(dict(pid=child.pid, run=str(out), secret_in_files=False)), flush=True)
        return
    with (out / 'cloud.log').open('x') as log:
        cloud = subprocess.Popen(['/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python',
            str(SCRIPTS / 'cloud_av_preference_synthesis_20261027.py'), '--run', str(out), '--key-stdin'],
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
    cloud.stdin.write(key + '\n')
    cloud.stdin.close()
    del key
    returncode = cloud.wait()
    completion = json.loads((out / 'cloud_completion.json').read_text())
    if returncode or completion['status'] != 'COMPLETE':
        (out / 'status.json').write_text(json.dumps(dict(stage='CLOUD_FAILED', completion=completion), indent=2))
        raise SystemExit(1)
    process = subprocess.run(['/home/wjj/.venvs/syncnet/bin/python',
        str(SCRIPTS / 'cloud_av_preference_20261027.py'), 'execute', '--out', str(out)])
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
