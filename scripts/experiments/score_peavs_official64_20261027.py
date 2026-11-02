"""Run upstream PEAVS in a fresh directory; retain features and full precision scores."""
import argparse
import csv
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/wjj/av-eval-20261026')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('videos', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--metric_batch_size', type=int, default=64, help='Official metric.py default; record exact sorted batch composition')
    args = p.parse_args()
    videos = [Path(x).resolve(strict=True) for x in args.videos]
    if len({x.stem for x in videos}) != len(videos):
        raise ValueError('Input video stems must be unique')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / (output.stem + '-work-' + str(time.time_ns()))
    source = ROOT / 'peavs-source' / 'PEAVS'
    shutil.copytree(source, work, copy_function=lambda src, dst: os.symlink(src, dst))
    inputs = work / 'inputs'
    inputs.mkdir()
    for video in videos:
        if video.suffix != '.mp4':
            raise ValueError('PEAVS upstream accepts .mp4 inputs')
        (inputs / video.name).symlink_to(video)
    extract = work / 'av_features_extraction'
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    torch_site = Path(importlib.util.find_spec('torch').origin).parent.parent
    cuda_libs = sorted((torch_site / 'nvidia').glob('*/lib'))
    env['LD_LIBRARY_PATH'] = ':'.join([*(str(x) for x in cuda_libs), env.get('LD_LIBRARY_PATH', '')])
    subprocess.run([sys.executable, str(ROOT / 'peavs_extract_bootstrap.py'), str(inputs)], cwd=extract, env=env, check=True)
    # Remove upstream's automatic feature deletion only in the in-memory source.
    metric_code = (work / 'metric.py').read_text().replace("    os.system('rm -rf ./av_features_extraction/output_feats/')", '    pass # retain deployment evidence')
    subprocess.run([sys.executable, '-c', metric_code, '--batch_size', str(args.metric_batch_size)], cwd=work, env=env, check=True)
    rows = list(csv.DictReader((work / 'results/results.csv').open()))
    if len(rows) != len(videos):
        raise RuntimeError(f'Expected {len(videos)} scores; got {len(rows)}')
    import math
    for row in rows:
        row['score'] = float(row['score'])
        if not math.isfinite(row['score']) or not 1 <= row['score'] <= 5:
            raise RuntimeError(f'Invalid PEAVS output: {row}')
    output.write_text(json.dumps({'metric': 'PEAVS', 'scores': rows, 'work_dir': str(work), 'metric_batch_size': args.metric_batch_size, 'padding_warning': 'Upstream can depend on batch composition when audio/video feature lengths differ'}, indent=2))
    print(output.read_text())

if __name__ == '__main__':
    main()
