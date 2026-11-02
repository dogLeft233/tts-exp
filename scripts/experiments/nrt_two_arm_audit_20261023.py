"""Independent CPU audit from raw distances, logs, waveform hashes and plan.

Does not import either runner or analysis implementation; never runs a model.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import re
import wave
from decimal import Decimal
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_score(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 31 or not np.isfinite(values).all() or not len(values):
        raise ValueError('Invalid official distance support')
    curve = values.mean(axis=0)
    best = int(np.argmin(curve))
    return [float(np.median(curve) - curve[best]), float(curve[best]), 15 - best]


def audit(run):
    plan = read(run / 'analysis_protocol.json')
    result = read(run / 'analysis.json')
    errors, cells, lookup = [], [], {}
    for binding in plan['score_bindings']:
        sid, role = binding['sample_id'], binding['role']
        path = Path(binding['path'])
        if not path.is_file():
            errors.append(f'Missing {sid}:{role}')
            continue
        receipt = read(path)
        if receipt.get('status') != 'COMPLETE':
            errors.append(f'Incomplete {sid}:{role}')
            continue
        lookup[sid, role] = receipt
        checks = []
        for field, hashfield in [('video_path', 'source_video_sha256'),
                ('canonical_audio_path', 'canonical_audio_sha256'), ('input_mkv', 'input_mkv_sha256'),
                ('activesd_path', 'activesd_sha256'), ('tracks_pickle_path', 'tracks_pickle_sha256')]:
            checks.append(digest(receipt[field]) == receipt[hashfield])
        if binding.get('sha256'):
            checks.append(digest(path) == binding['sha256'])
        with wave.open(receipt['canonical_audio_path'], 'rb') as f:
            checks += [(f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000),
                       f.getnframes() == receipt['audio_samples'],
                       hashlib.sha256(f.readframes(f.getnframes())).hexdigest() == receipt['PCM_sha256']]
        checks += [receipt['decoded_frames_exact'], receipt['frame_PTS_exact'], receipt['mux_PCM_exact']]
        outputs = receipt['checks']
        for output, expected in outputs.items():
            checks.append(digest(output) == expected)
        by_name = {Path(output).name: Path(output) for output in outputs}
        def frame_hashes(path):
            return [line.split(',')[-1].strip() for line in path.read_text().splitlines()
                    if line.strip() and not line.startswith('#')]
        original = frame_hashes(by_name['source.framemd5'])
        copied = frame_hashes(by_name['mux.framemd5'])
        clocks = [[Decimal(f['best_effort_timestamp_time']) for f in read(by_name[name])['frames']]
                  for name in ['source_frames.json', 'mux_frames.json']]
        checks += [bool(original), original == copied, clocks[0] == clocks[1],
                   len(original) == len(clocks[0]) == receipt['frame_count'],
                   digest(by_name['mux_PCM.s16le']) == receipt['PCM_sha256'],
                   by_name['mux_PCM.s16le'].stat().st_size == 2 * receipt['audio_samples']]
        for stage in ['pipeline_receipt', 'syncnet_receipt']:
            r = receipt[stage]
            checks += [r['returncode'] == 0, not r['timed_out'], digest(r['log']) == r['log_sha256']]
        log = Path(receipt['syncnet_log']).read_text()
        c = [float(x) for x in re.findall(r'\bConfidence:\s*([-+\d.eE]+)', log)]
        d = [float(x) for x in re.findall(r'\bMin dist:\s*([-+\d.eE]+)', log)]
        o = [int(x) for x in re.findall(r'\bAV offset:\s*(-?\d+)', log)]
        dist = pickle.loads(Path(receipt['activesd_path']).read_bytes())
        tracks = pickle.loads(Path(receipt['tracks_pickle_path']).read_bytes())
        checks += [len(c) == len(d) == len(o) == len(dist) == len(tracks) == len(receipt['tracks']), bool(c)]
        max_error = 0.0
        if c:
            checks.append([receipt['sync_c'], receipt['sync_d'], receipt['av_offset']] == [c[0], d[0], o[0]])
        for i, matrix in enumerate(dist):
            recomputed = raw_score(matrix)
            error = max(abs(recomputed[0] - c[i]), abs(recomputed[1] - d[i]))
            max_error = max(max_error, error)
            checks += [error <= .00051, recomputed[2] == o[i],
                       0 < len(matrix) <= len(tracks[i]['track']['frame']) - 5]
        passed = all(checks)
        cells.append({'sample_id': sid, 'role': role, 'PASS': passed,
                      'tracks': len(dist), 'max_raw_distance_rounding_error': max_error})
        if not passed:
            errors.append(f'Cell check failed {sid}:{role}')
    checks, max_stats_error = [], 0.0
    records = {r['sample_id']: r for r in result['records']}
    for metric, field in [('C', 'sync_c'), ('D', 'sync_d'), ('offset', 'av_offset')]:
        values = {name: [] for name in plan['contrasts']}
        for sample in plan['samples']:
            sid = sample['sample_id']
            if not records[sid]['complete']:
                continue
            means = {name: sum(lookup[sid, role][field] for role in roles) / len(roles)
                     for name, roles in plan['groups'].items()}
            for name, (left, right) in plan['contrasts'].items():
                value = means[left] - means[right]
                values[name].append(value)
                error = abs(value - records[sid]['metrics'][metric]['contrasts'][name])
                max_stats_error = max(max_stats_error, error)
        for name, x in values.items():
            n = len(x)
            if not n:
                continue
            summary = result['metrics'][metric]['contrasts'][name]
            indices = np.random.Generator(np.random.PCG64(plan['bootstrap']['seed'])).integers(
                0, n, size=(plan['bootstrap']['replicates'], n))
            boot = np.asarray(x)[indices].mean(axis=1)
            expected = [sum(x) / n, *np.quantile(boot, [.025, .975]), *np.quantile(boot, [.0125, .9875])]
            observed = [summary['mean'], *summary['ci95'], *summary['ci97_5']]
            error = max(abs(a - b) for a, b in zip(expected, observed))
            max_stats_error = max(max_stats_error, error)
            passed = error < 1e-12 and summary['n'] == n and summary['positive'] == sum(v > 0 for v in x)
            checks.append({'metric': metric, 'contrast': name, 'PASS': passed, 'max_error': error})
            if not passed:
                errors.append(f'Statistics check failed {metric}:{name}')
    if max_stats_error >= 1e-12:
        errors.append('Per-person arithmetic mismatch')
    if digest(run / 'analysis_protocol.json') != result['analysis_protocol_sha256']:
        errors.append('Analysis protocol SHA mismatch')
    strict_checks = []
    strict_path = run / 'strict_subset_analysis.json'
    if strict_path.is_file():
        strict = read(strict_path)
        selected = set(plan['strict_subset']['retained_speaker_ids'])
        if strict['coverage']['complete_speakers'] != 15 or strict['coverage']['finite_cells'] != 75:
            errors.append('Strict15 coverage mismatch')
        for metric, field in [('C', 'sync_c'), ('D', 'sync_d'), ('offset', 'av_offset')]:
            for name, (left, right) in plan['contrasts'].items():
                values = []
                for sample in plan['samples']:
                    if sample['speaker_id'] not in selected:
                        continue
                    sid = sample['sample_id']
                    mean = lambda group: sum(lookup[sid, role][field] for role in plan['groups'][group]) / len(plan['groups'][group])
                    values.append(mean(left) - mean(right))
                n = len(values)
                indices = np.random.Generator(np.random.PCG64(plan['bootstrap']['seed'])).integers(
                    0, n, size=(plan['bootstrap']['replicates'], n))
                boot = np.asarray(values)[indices].mean(axis=1)
                expected = [sum(values) / n, *np.quantile(boot, [.025, .975]), *np.quantile(boot, [.0125, .9875])]
                observed = strict['metrics'][metric]['contrasts'][name]
                error = max(abs(a - b) for a, b in zip(expected, [observed['mean'], *observed['ci95'], *observed['ci97_5']]))
                passed = error < 1e-12 and observed['n'] == n == 15
                strict_checks.append({'metric': metric, 'contrast': name, 'PASS': passed, 'max_error': error})
                if not passed:
                    errors.append(f'Strict15 arithmetic {metric}:{name}')
    return {'status': 'PASS' if not errors else 'FAIL', 'cell_checks': cells, 'statistics_checks': checks,
            'strict_subset_statistics_checks': strict_checks,
            'max_statistics_error': max_stats_error, 'errors': errors,
            'scope': 'CPU raw-artifact and arithmetic audit; no stochastic TTS reproducibility rerun',
            'source_sha256': digest(__file__)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    args = p.parse_args()
    result = audit(args.run)
    with (args.run / 'independent_audit.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'cells': len(result['cell_checks']),
                      'statistics_checks': len(result['statistics_checks']), 'errors': result['errors']}))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
