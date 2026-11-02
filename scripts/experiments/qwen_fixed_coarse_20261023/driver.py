"""Bounded serial probe; stops at the first failed control, retains actual exits."""
from pathlib import Path
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_fixed_coarse_20261023_attempt01'
PY='/home/wjj/.venvs/qwen3-tts-1p7b/bin/python'
PROBE=Path(__file__).with_name('probe.py')


def main():
    for stage in ['control','cross']:
        for model in ['L06','L17']:
            name=f'{model}_{stage}'
            start=time.monotonic()
            with (RUN/f'{name}.log').open('x') as log:
                r=subprocess.run([PY,str(PROBE),stage,'--model',model],cwd=ROOT,
                                 stdout=log,stderr=subprocess.STDOUT,timeout=960)
            with (RUN/f'{name}_exit.json').open('x') as f:
                json.dump(dict(returncode=r.returncode,elapsed_s=time.monotonic()-start),f)
            print(name,r.returncode,flush=True)
            if r.returncode: return r.returncode
    return 0


if __name__=='__main__': raise SystemExit(main())
