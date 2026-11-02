"""CPU paired analysis of frozen historical16 and cloud10 pilot protocols.

Score files are inputs only. No models, retries, filtering, or media edits.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = {'C': 'sync_c', 'D': 'sync_d', 'offset': 'av_offset'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values, seed=20261023, replicates=10000):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Nonfinite speaker values')
    if not len(x):
        return {'n': 0, 'mean': None, 'ci95': None, 'ci97_5': None,
                'positive': 0, 'negative': 0, 'zero': 0, 'values': []}
    index = np.random.Generator(np.random.PCG64(seed)).integers(0, len(x), (replicates, len(x)))
    draws = x[index].mean(axis=1)
    return {'n': len(x), 'mean': float(x.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'ci97_5': np.quantile(draws, [.0125, .9875]).tolist(),
            'positive': int((x > 0).sum()), 'negative': int((x < 0).sum()),
            'zero': int((x == 0).sum()), 'values': x.tolist()}


def paired_values(scores, groups, contrasts):
    """Average repeated draws within source speaker before any bootstrap."""
    group_values = {name: float(np.mean([scores[role] for role in roles]))
                    for name, roles in groups.items()}
    return group_values, {name: group_values[left] - group_values[right]
                          for name, (left, right) in contrasts.items()}


def analyze(plan):
    samples, groups = plan['samples'], plan['groups']
    if len({r['sample_id'] for r in samples}) != len(samples):
        raise ValueError('Duplicate planned sample')
    if len({r['speaker_id'] for r in samples}) != len(samples):
        raise ValueError('Expected one source recording per independent speaker')
    roles = sorted({r for rs in groups.values() for r in rs})
    if not all(groups.values()):
        raise ValueError('Empty role group')
    planned = {(s['sample_id'], r) for s in samples for r in roles}
    bindings = {(b['sample_id'], b['role']): b for b in plan['score_bindings']}
    if len(bindings) != len(plan['score_bindings']) or set(bindings) != planned:
        raise ValueError('Score bindings must exactly cover planned layout')
    records, failures, receipts = [], [], []
    for sample in samples:
        sid = sample['sample_id']
        scores, missing = {}, []
        for role in roles:
            binding = bindings[sid, role]
            path = Path(binding['path'])
            if not path.is_file():
                missing.append(role)
                failures.append({'sample_id': sid, 'role': role, 'status': 'MISSING'})
                continue
            digest = sha(path)
            if binding.get('sha256') and digest != binding['sha256']:
                raise ValueError('Frozen score receipt changed: ' + str(path))
            row = read(path)
            if (row.get('sample_id'), row.get('role')) != (sid, role):
                raise ValueError('Mislabeled score receipt: ' + str(path))
            if row.get('status') != 'COMPLETE' or not all(isinstance(row.get(f), (int, float))
                    and np.isfinite(row[f]) for f in FIELDS.values()):
                missing.append(role)
                failures.append({'sample_id': sid, 'role': role, 'status': row.get('status')})
                continue
            scores[role] = row
            receipts.append({'sample_id': sid, 'role': role, 'path': str(path), 'sha256': digest})
        if missing:
            records.append({**sample, 'complete': False, 'missing_roles': missing})
            continue
        output = {**sample, 'complete': True, 'metrics': {}}
        for metric, field in FIELDS.items():
            values = {role: float(scores[role][field]) for role in roles}
            means, differences = paired_values(values, groups, plan['contrasts'])
            output['metrics'][metric] = {'roles': values, 'groups': means, 'contrasts': differences}
        records.append(output)
    complete = [r for r in records if r['complete']]
    metrics = {}
    for metric in FIELDS:
        metrics[metric] = {
            'group_means': {g: float(np.mean([r['metrics'][metric]['groups'][g] for r in complete]))
                            if complete else None for g in groups},
            'contrasts': {c: summarize([r['metrics'][metric]['contrasts'][c] for r in complete],
                    plan['bootstrap']['seed'], plan['bootstrap']['replicates']) for c in plan['contrasts']}}
    return {'status': 'ANALYZED_COMPLETE_COVERAGE' if not failures else 'INCOMPLETE_DESCRIPTIVE_ONLY',
            'coverage': {'planned_speakers': len(samples), 'complete_speakers': len(complete),
                         'planned_cells': len(planned), 'finite_cells': len(receipts), 'failures': failures},
            'metrics': metrics, 'records': records, 'score_receipts': receipts,
            'primary_contrasts_C': plan['primary_contrasts_C'],
            'interpretation': plan['interpretation'], 'bootstrap_unit': 'source speaker, draw-averaged',
            'analysis_source_sha256': sha(__file__)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    args = p.parse_args()
    path = args.run / 'analysis_protocol.json'
    if sha(path) != args.plan_sha256:
        raise ValueError('Analysis plan changed')
    result = analyze(read(path))
    result['analysis_protocol_sha256'] = sha(path)
    with (args.run / 'analysis.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    with (args.run / 'speaker_contrasts.csv').open('x', encoding='utf8', newline='') as f:
        names = list(read(path)['contrasts'])
        w = csv.writer(f)
        w.writerow(['sample_id', 'speaker_id', 'complete', *names])
        for row in result['records']:
            w.writerow([row['sample_id'], row['speaker_id'], row['complete'],
                        *[row['metrics']['C']['contrasts'][c] if row['complete'] else '' for c in names]])
    print(json.dumps({'status': result['status'], 'coverage': result['coverage']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
