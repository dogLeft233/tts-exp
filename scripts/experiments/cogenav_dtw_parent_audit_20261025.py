"""Independent saved-path and clock audit; does not import the pilot executor."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(error, mask):
    values = error[mask] * 1000
    return dict(frames=len(values), mae_ms=float(values.mean()),
                median_ms=float(np.median(values)), p95_ms=float(np.quantile(values, .95)),
                over80_fraction=float(np.mean(values > 80)),
                within40_fraction=float(np.mean(values <= 40)))


def clock(times, segments, reverse=False):
    truth = np.full(len(times), np.nan)
    mask = np.zeros(len(times), bool)
    for segment in segments:
        a, b, c, d = np.asarray(segment, float)
        if reverse:
            a, b, c, d = c, d, a, b
        inside = (times >= a/16000) & (times < b/16000)
        truth[inside] = c/16000 + (times[inside] - a/16000) * (d-c) / (b-a)
        mask |= inside
    return truth, mask


def projected(path, times, length):
    groups = [[] for _ in range(length)]
    for i, j in path:
        groups[i].append(times[j])
    assert all(groups)
    return np.array([np.median(group) for group in groups])


def validate_path(path, n, m):
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (n-1, m-1)
    steps = np.diff(path, axis=0)
    assert all(tuple(step) in {(1, 1), (1, 0), (0, 1)} for step in steps)
    assert np.all(abs(path[:, 0]/max(n-1, 1)-path[:, 1]/max(m-1, 1)) <= .25+1e-12)


def solve_cost(cost):
    """Independent recurrence with one rolling row; no executor traceback reuse."""
    n, m = cost.shape
    previous = np.full(m, np.inf)
    for i in range(n):
        current = np.full(m, np.inf)
        for j in range(m):
            if abs(i/max(n-1, 1)-j/max(m-1, 1)) > .25:
                continue
            if i == j == 0:
                current[j] = cost[i, j]
            else:
                candidates = []
                if i and j:
                    candidates.append(previous[j-1])
                if i:
                    candidates.append(previous[j]+.05)
                if j:
                    candidates.append(current[j-1]+.05)
                current[j] = cost[i, j] + min(candidates)
        previous = current
    return previous[-1]


def audit(root):
    inputs = json.loads((root/'selected_inputs.json').read_text())
    manifest = json.loads((root/'signals.json').read_text())
    reported = json.loads((root/'metrics.json').read_text())
    lookup = {(r['id'], r['condition'], r['method']): r for r in reported}
    max_metric = max_prediction = max_clock = max_cost = 0.
    reverse_rows = []
    verified_paths = 0
    for row in inputs:
        original = root/'signals'/row['id']/'identity'
        x, sr = sf.read(original/'audio.wav', dtype='float32')
        assert sr == 16000
        original_features = np.load(original/'features.npz')
        times = original_features['times']
        energy = np.array([np.sqrt(np.mean(x[max(0, int(t*sr)-320):min(len(x), int(t*sr)+320)]**2)) for t in times])
        speech = energy > max(energy.max()*.01, 1e-8)
        for condition in ['identity', 'stretch', 'pause', 'combined']:
            directory = root/'signals'/row['id']/condition
            entry = next(r for r in manifest if r['id'] == row['id'] and r['condition'] == condition)
            assert digest(directory/'audio.wav') == entry['audio_sha256']
            y, ysr = sf.read(directory/'audio.wav', dtype='float32')
            assert ysr == sr and len(y) == entry['samples']
            if condition == 'identity':
                assert np.array_equal(x, y)
            data = np.load(directory/'alignment.npz')
            target = np.load(directory/'features.npz')
            segments = np.load(directory/'oracle.npz')['segments']
            assert np.array_equal(data['segments'], segments)
            truth, valid = clock(times, segments)
            max_clock = max(max_clock, float(np.nanmax(abs(truth-data['truth']))))
            valid &= truth <= target['times'][-1]
            if condition != 'identity':
                for cut in segments[:-1, 1]/sr:
                    valid &= abs(times-cut) > .08
            active = valid & speech
            assert np.array_equal(valid, data['valid']) and np.array_equal(active, data['active']), (row['id'], condition, np.where(valid != data['valid'])[0].tolist(), np.where(active != data['active'])[0].tolist())
            reverse_truth, reverse_valid = clock(target['times'], segments, reverse=True)
            reverse_valid &= reverse_truth <= times[-1]
            if condition != 'identity':
                for boundary in np.unique(segments[:, 2:].ravel())[1:-1]/sr:
                    reverse_valid &= abs(target['times']-boundary) > .08
            reverse_energy = np.array([np.sqrt(np.mean(y[max(0, int(t*sr)-320):min(len(y), int(t*sr)+320)]**2)) for t in target['times']])
            reverse_active = reverse_valid & (reverse_energy > max(reverse_energy.max()*.01, 1e-8))
            for method in ['cogenav', 'mfcc', 'mel', 'linear', 'oracle_grid']:
                if method in ['cogenav', 'mfcc', 'mel']:
                    path = data[method+'_path']
                    validate_path(path, len(times), len(target['times']))
                    pred = projected(path, target['times'], len(times))
                    max_prediction = max(max_prediction, float(np.max(abs(pred-data[method+'_prediction']))))
                    left, right = original_features[method], target[method]
                    left = left/np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
                    right = right/np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
                    cost = np.clip(1-left@right.T, 0, 2)
                    optimum = solve_cost(cost)
                    steps = np.diff(path, axis=0)
                    observed = float(cost[path[:, 0], path[:, 1]].sum(dtype=np.float64)) + .05*np.count_nonzero(steps.sum(axis=1) == 1)
                    max_cost = max(max_cost, abs(observed-optimum))
                    reverse_pred = projected(path[:, ::-1], times, len(target['times']))
                    reverse_rows.append(dict(id=row['id'], sample_id=row['sample_id'], condition=condition, method=method,
                                             speech=metric(abs(reverse_pred-reverse_truth), reverse_active)))
                    verified_paths += 1
                elif method == 'linear':
                    pred = times*(len(y)/len(x))
                else:
                    pred = np.round(truth/.04)*.04
                expected = lookup[row['id'], condition, method]
                for support, mask in [('speech', active), ('allvalid', valid)]:
                    computed = metric(abs(pred-truth), mask)
                    for name, value in computed.items():
                        max_metric = max(max_metric, abs(value-expected[support][name]))
                if condition == 'identity':
                    assert np.max(abs(pred-truth)) < 1e-10
    assert max_clock < 1e-10 and max_prediction < 1e-10 and max_metric < 1e-8 and max_cost < 1e-4, (max_clock, max_prediction, max_metric, max_cost)
    summary = json.loads((root/'summary.json').read_text())
    for group in summary:
        cells = [r for r in reported if r['method'] == group['method'] and
                 (r['condition'] == group['condition'] or (group['condition'] == 'nonidentity_mean' and r['condition'] != 'identity'))]
        source_values = [np.mean([r['speech']['mae_ms'] for r in cells if r['sample_id'] == sid]) for sid in sorted({r['sample_id'] for r in cells})]
        assert np.allclose(source_values, group['source_mae_ms'], atol=1e-10, rtol=0)
        assert abs(np.mean(source_values)-group['source_equal_mae_ms']) < 1e-10
    actual_paths = 0
    for natural, tts in zip(inputs[::2], inputs[1::2]):
        assert natural['sample_id'] == tts['sample_id'] and tts['score_float32'] > natural['score_float32']
        left = np.load(root/'signals'/natural['id']/'identity/features.npz')
        right = np.load(root/'signals'/tts['id']/'identity/features.npz')
        pair = np.load(root/'actual_pairs'/natural['sample_id']/'alignment.npz')
        for method in ['cogenav', 'mfcc', 'mel']:
            path = pair[method+'_path']
            validate_path(path, len(left['times']), len(right['times']))
            assert np.array_equal(projected(path, right['times'], len(left['times'])), pair[method+'_prediction'])
            actual_paths += 1
    output = dict(status='PASS', verified_paths=verified_paths, actual_exploratory_paths=actual_paths, max_clock_error_s=max_clock,
                  max_prediction_error_s=max_prediction, max_metric_difference=max_metric,
                  max_saved_path_cost_vs_independent_optimum=max_cost,
                  reverse_diagnostic=reverse_rows,
                  scope='Saved features, signal hashes, forward metrics, independent path optimality, reverse path diagnostic; no new GPU inference')
    (root/'parent_independent_audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k != 'reverse_diagnostic'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    audit(Path(parser.parse_args().run))
