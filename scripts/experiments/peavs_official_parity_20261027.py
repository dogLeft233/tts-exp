"""Compare unmodified upstream metric.py against deployed PEAVS scores.

Only intercept upstream feature deletion. Also test the CuPy compatibility
kernel against an independent PyTorch definition and re-extract probe media.
"""
import argparse
import csv
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = Path('/home/wjj/av-eval-20261026')
SOURCE = DEPLOY / 'peavs-source/PEAVS'
OLD = ROOT / 'runs/av_metric_preference_20261026_attempt02'


def read(path):
    return json.loads(Path(path).read_text())


def score_original(out, batch_size, mapping):
    work = out / ('peavs_original_batch' + str(batch_size))
    if (work / 'results/results.csv').exists():
        return {r['filename']: float(r['score']) for r in csv.DictReader((work / 'results/results.csv').open())}
    shutil.copytree(SOURCE, work, copy_function=lambda src, dest: os.symlink(src, dest))
    feat = work / 'av_features_extraction/output_feats'
    for kind in ['vggish', 'i3d_feats']:
        (feat / kind).mkdir(parents=True, exist_ok=True)
    for name, row in mapping.items():
        (feat / 'vggish' / (name + '_vggish.npy')).symlink_to(row['audio'])
        (feat / 'i3d_feats' / (name + '_i3d.npy')).symlink_to(row['video'])
    previous = Path.cwd()
    actual_system = os.system
    cleanup_calls = []

    def prevent_cleanup(command):
        assert command == 'rm -rf ./av_features_extraction/output_feats/', command
        cleanup_calls.append(command)
        return 0

    try:
        os.chdir(work)
        sys.path.insert(0, str(work))
        os.system = prevent_cleanup
        sys.argv = [str(work / 'metric.py'), '--batch_size', str(batch_size)]
        runpy.run_path(str(work / 'metric.py'), run_name='__main__')
    finally:
        os.chdir(previous)
        os.system = actual_system
    assert cleanup_calls == ['rm -rf ./av_features_extraction/output_feats/']
    return {r['filename']: float(r['score']) for r in csv.DictReader((work / 'results/results.csv').open())}


def test_correlation_adapter():
    import cupy
    feature_source = SOURCE / 'av_features_extraction'
    # Remove the PEAVS head utility namespace before importing extractor models.
    for name in list(sys.modules):
        if name == 'utils' or name.startswith('utils.'):
            del sys.modules[name]
    sys.path.insert(0, str(feature_source))
    from models.pwc.pwc_src import correlation

    @cupy.memoize(for_each_device=True)
    def launch(name, code):
        kernel = cupy.RawModule(code=code).get_function(name)
        def invoke(**kwargs):
            kwargs['stream'] = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
            return kernel(**kwargs)
        return invoke

    correlation.cupy_launch = launch
    torch.manual_seed(20261027)
    a = torch.randn(2, 8, 9, 11, device='cuda').contiguous()
    b = torch.randn_like(a).contiguous()
    actual = correlation.FunctionCorrelation(a, b, torch.device('cuda:0'))
    padded = F.pad(b, (4, 4, 4, 4))
    expected = torch.stack([(a * padded[:, :, dy:dy+9, dx:dx+11]).mean(1)
                            for dy in range(9) for dx in range(9)], dim=1)
    torch.cuda.synchronize()
    error = float((actual - expected).abs().max())
    assert error < 1e-6
    return error


def main(out):
    torch.set_num_threads(4)
    reproduction = read(out / 'cached_control_reproduction.json')['peavs']['rows']
    expected = {r['id']: r['shifted'] for r in reproduction}
    expected.update({'native__' + r['source'] + '_N_0': r['native'] for r in reproduction})
    mapping = {}
    for raw in (OLD / 'peavs').glob('batch*.raw.json'):
        data = read(raw)
        feat = Path(data['work_dir']) / 'av_features_extraction/output_feats'
        for row in data['scores']:
            name = Path(row['filename']).stem
            if name in expected:
                mapping[name] = dict(audio=feat / 'vggish' / (name + '_vggish.npy'),
                                     video=feat / 'i3d_feats' / (name + '_i3d.npy'))
    assert len(mapping) == 60
    scores1 = score_original(out, 1, mapping)
    scores64 = score_original(out, 64, mapping)
    assert set(scores1) == set(scores64) == set(expected)
    error1 = max(abs(scores1[k] - expected[k]) for k in expected)
    batch_error = max(abs(scores1[k] - scores64[k]) for k in expected)
    assert error1 < 1e-5, error1
    controls = {}
    for name, values in [('original_batch1', scores1), ('original_default64', scores64)]:
        controls[name] = dict(lower=sum(values[r['id']] < values['native__' + r['source'] + '_N_0']
                                       for r in reproduction), total=40)
    # A different default batch size is observable even when it only changes
    # round-off; record it instead of assuming sample independence.
    # Match upstream bootstrap import order: CuPy before Torch. Running this
    # after a head-only Torch process can load the MFA CUDA libraries first.
    kernel_code = ('import cupy\nimport torch\nimport torch.nn.functional as F\nimport sys\n'
                   'from pathlib import Path\nSOURCE=Path(' + repr(str(SOURCE)) + ')\n'
                   + inspect.getsource(test_correlation_adapter) + '\nprint(test_correlation_adapter())\n')
    checked = subprocess.run([sys.executable, '-c', kernel_code], capture_output=True, text=True)
    (out / 'peavs_kernel_reference.log').write_text(checked.stdout + checked.stderr)
    assert checked.returncode == 0, 'Kernel reference failed; see dedicated log'
    kernel_error = float(checked.stdout.splitlines()[-1])
    videos = [OLD / 'inputs' / (name + '.mp4') for name in
              ['native__BAC009S0003W0121_N_0', 'shift__BAC009S0003W0121__-6400']]
    videos += sorted((out / 'probe_inputs').glob('*.mp4'))
    output = out / 'peavs_fresh_probe.json'
    with (out / 'peavs_fresh_probe.log').open('w') as log:
        done = subprocess.run([sys.executable, str(DEPLOY / 'score_peavs.py'), *map(str, videos),
                               '--output', str(output)], stdout=log, stderr=subprocess.STDOUT)
    assert done.returncode == 0, 'Fresh PEAVS extraction failed; see log'
    fresh = read(output)
    fresh_parity = []
    for row in fresh['scores']:
        name = Path(row['filename']).stem
        if name in expected:
            error = abs(row['score'] - expected[name])
            assert error < 1e-5
            fresh_parity.append(dict(id=name, previous=expected[name], fresh=row['score'], absolute_error=error))
    report = dict(status='PASS' if batch_error < 1e-4 else 'OFFICIAL_DEFAULT_BATCH_DIFFERS', original_metric_source_unmodified=True, cleanup_intercepted_only=True,
        cases=60, max_cached_head_parity_error=error1, max_batch1_vs_default64_error=batch_error,
        control_counts=controls, correlation_cuda_vs_torch_max_error=kernel_error,
        fresh_extraction_parity=fresh_parity, probes=fresh['scores'])
    (out / 'peavs_official_parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
