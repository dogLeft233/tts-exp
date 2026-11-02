"""Independent, read-only checks of saved raw arrays and reported statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify_stage(row, path):
    x = np.load(path, allow_pickle=False)
    assert list(x.shape) == row['shape'], f'shape mismatch: {path}'
    assert str(x.dtype) == row['dtype'], f'dtype mismatch: {path}'
    assert hashlib.sha256(x.tobytes()).hexdigest() == row['sha256'], f'hash mismatch: {path}'
    v = x.astype(np.float64)
    assert np.isfinite(v).all(), f'nonfinite: {path}'
    assert np.isclose(np.sqrt(np.sum(v*v) / v.size), row['rms'], rtol=1e-12)
    assert np.max(np.abs(v)) == row['peak']
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reports = []
    for folder, source_prefix in [
        ('frontend_attempt01', '/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/results/frontend_attempt01'),
        ('hubert_attempt03', '/dev/shm/synctalk-amplitude-audit-20261016/hubert_attempt03'),
        ('conditioning_attempt02', '/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/results/conditioning_attempt02')]:
        root = a.bundle / folder
        s = json.loads((root / 'statistics.json').read_text())
        arrays = {}
        for row in s['stages']:
            path = root / Path(row['path']).relative_to(source_prefix)
            x = verify_stage(row, path)
            arrays[row['input_id'], row['gain_db'], row.get('stage', 'conditioning')] = x
        checked = 0
        for row in s['comparisons']:
            stage = row.get('stage', 'conditioning')
            key = row['input_id'], row['gain_db'], stage
            if key not in arrays:
                continue
            x = arrays[key].astype(np.float64)
            b = arrays[row['input_id'], 0, stage].astype(np.float64)
            delta = x-b
            assert row['same_shape'] and x.shape == b.shape
            assert np.max(np.abs(delta)) == row['max_abs_diff']
            relative = np.sqrt(np.sum(delta*delta) / max(np.sum(b*b), 1e-60))
            assert np.isclose(relative, row['relative_l2'], rtol=1e-11, atol=1e-15)
            checked += 1
        assert all(row['max_abs_diff'] == 0 for row in s['repeats'])
        if folder == 'frontend_attempt01':
            for row in s['inputs']:
                base = arrays[row['id'], 0, 'double_waveform']
                for c in row['conditions']:
                    if c['status'] == 'RUN':
                        x = arrays[row['id'], c['gain_db'], 'double_waveform']
                        assert np.array_equal(x, base * 10**(c['gain_db']/20))
                        assert np.max(np.abs(x)) < 1
                        assert c['sample_rate'] == 16000 and len(x) == c['samples']
            formula = [r for r in s['comparisons'] if r['stage'] == 'unclipped_mel_shift_formula']
            assert max(r['max_abs_error'] for r in formula) < 2e-5
        if folder == 'hubert_attempt03':
            assert not s['strict_load_missing'] and not s['strict_load_unexpected']
            assert s['effective_posconv_weight_max_error'] == 0 and not s['model_training']
        extra_arrays = 0
        if folder == 'conditioning_attempt02':
            verify_stage(s['may']['embedding'], root / 'may_first32_ave.npy')
            verify_stage(s['may']['encoded_conditioning'], root / 'may_middle24_generator_conditioning.npy')
            assert s['may']['repeat']['max_abs_diff'] == 0
            extra_arrays = 2
        reports.append({'folder': folder, 'raw_arrays_checked': len(arrays),
                        'additional_may_arrays_checked': extra_arrays,
                        'comparisons_recomputed': checked, 'repeat_errors_zero': True,
                        'statistics_sha256': hashlib.sha256((root/'statistics.json').read_bytes()).hexdigest()})
    a.output.write_text(json.dumps({'status': 'PASS', 'reports': reports}, indent=2))
    print(a.output.read_text())


if __name__ == '__main__':
    main()
