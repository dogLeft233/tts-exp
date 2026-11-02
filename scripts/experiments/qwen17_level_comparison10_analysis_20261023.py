"""CPU-only source-speaker analysis of the frozen four-model, two-level pilot.

Consumes receipts without importing the generator or launching model forwards.
All planned cells remain in the coverage table; incomplete contrasts are explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ('N', 'L06', 'L17', 'CLOUD')
LEVELS = ('norm', 'raw')
METRICS = {'C': 'sync_c', 'D': 'sync_d', 'offset': 'av_offset'}
PAIRS = {'L17-N': ('L17', 'N'), 'L17-L06': ('L17', 'L06'),
         'L17-CLOUD': ('L17', 'CLOUD'), 'L06-N': ('L06', 'N'),
         'CLOUD-N': ('CLOUD', 'N'), 'CLOUD-L06': ('CLOUD', 'L06')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite speaker values')
    if not len(x):
        return {'n': 0, 'mean': None, 'ci95': None, 'ci98_333': None,
                'positive': 0, 'negative': 0, 'values': []}
    indices = np.random.Generator(np.random.PCG64(20261023)).integers(
        0, len(x), size=(10000, len(x)))
    bootstrap = x[indices].mean(axis=1)
    return {'n': len(x), 'mean': float(x.mean()),
            'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'ci98_333': np.quantile(bootstrap, [.05 / 6, 1 - .05 / 6]).tolist(),
            'positive': int((x > 0).sum()), 'negative': int((x < 0).sum()),
            'zero': int((x == 0).sum()), 'values': x.tolist()}


def analyze(index, sample_plan):
    samples = [(r['sample_id'], r['speaker_id']) for r in sample_plan]
    if len(samples) != 10 or len(set(s for s, _ in samples)) != 10 or len(set(p for _, p in samples)) != 10:
        raise ValueError('Exactly ten distinct utterances and speakers required')
    expected = {(sid, model, level, draw) for sid, _ in samples
                for level in LEVELS for model in MODELS
                for draw in ([0] if model == 'N' else [1, 2, 3])}
    cells, failed = {}, []
    for row in index['records']:
        key = (row['sample_id'], row['model'], row['level'], row['draw_index'])
        if key not in expected or key in cells:
            raise ValueError(f'Unplanned or duplicate cell: {key}')
        if row['speaker_id'] != dict(samples)[row['sample_id']]:
            raise ValueError(f'Speaker binding mismatch: {key}')
        path = Path(row['score_path']) if row.get('score_path') else None
        score = read(path) if path is not None and path.is_file() else None
        valid = score is not None and score.get('status') == 'COMPLETE'
        if valid:
            if row.get('score_sha256') != sha(path):
                raise ValueError(f'Score SHA mismatch: {key}')
            if score['sample_id'] != row['sample_id']:
                raise ValueError(f'Score sample mismatch: {key}')
            valid = all(isinstance(score.get(f), (int, float)) and np.isfinite(score[f])
                        for f in METRICS.values())
        cells[key] = score if valid else None
        if not valid:
            failed.append({'key': key, 'status': score.get('status') if score else 'MISSING'})
    for key in sorted(expected - set(cells)):
        cells[key] = None
        failed.append({'key': key, 'status': 'MISSING_INDEX_RECORD'})
    people = []
    for sid, speaker in samples:
        person = {'sample_id': sid, 'speaker_id': speaker, 'values': {}, 'missing': []}
        for level in LEVELS:
            person['values'][level] = {}
            for model in MODELS:
                keys = [(sid, model, level, draw) for draw in ([0] if model == 'N' else [1, 2, 3])]
                missing = [key for key in keys if cells[key] is None]
                person['missing'].extend(missing)
                person['values'][level][model] = {
                    metric: None if missing else float(np.mean([cells[k][field] for k in keys]))
                    for metric, field in METRICS.items()}
        people.append(person)
    by_level, level_effects, interactions = {}, {}, {}
    for metric in METRICS:
        by_level[metric] = {}
        for level in LEVELS:
            groups, contrasts = {}, {}
            for model in MODELS:
                groups[model] = summarize([p['values'][level][model][metric] for p in people
                                           if p['values'][level][model][metric] is not None])
            for label, (a, b) in PAIRS.items():
                contrasts[label] = summarize([p['values'][level][a][metric] - p['values'][level][b][metric]
                                              for p in people if all(p['values'][level][m][metric] is not None
                                                                     for m in (a, b))])
            by_level[metric][level] = {'groups': groups, 'contrasts': contrasts}
        level_effects[metric] = {}
        for model in MODELS:
            level_effects[metric][model] = summarize([
                p['values']['norm'][model][metric] - p['values']['raw'][model][metric]
                for p in people if all(p['values'][v][model][metric] is not None for v in LEVELS)])
        interactions[metric] = {}
        for label, (a, b) in PAIRS.items():
            interactions[metric][label] = summarize([
                (p['values']['norm'][a][metric] - p['values']['norm'][b][metric]) -
                (p['values']['raw'][a][metric] - p['values']['raw'][b][metric])
                for p in people if all(p['values'][v][m][metric] is not None
                                       for v in LEVELS for m in (a, b))])
    return {'status': 'ANALYZED_COMPLETE_COVERAGE' if not failed else 'ANALYZED_INCOMPLETE',
            'coverage': {'expected_cells': len(expected), 'valid_cells': sum(v is not None for v in cells.values()),
                         'failures': failed, 'complete_speakers': sum(not p['missing'] for p in people)},
            'records': people, 'by_level': by_level, 'norm_minus_raw': level_effects,
            'contrast_norm_minus_raw': interactions,
            'primary_family': {'metric': 'C', 'level': 'norm',
                               'contrasts': ['L17-N', 'L17-L06', 'L17-CLOUD'],
                               'CI': '98.333333% percentile; Bonferroni three comparisons'},
            'bootstrap': {'unit': 'source speaker; three draws averaged within person',
                          'seed': 20261023, 'replicates': 10000},
            'interpretation': 'Exploratory selected ten-person cohort; raw and normalized change both '
                              'generator and evaluator audio. No pure evaluator, model-size, human '
                              'quality, or equivalence inference. All draws retained, no training.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    result = analyze(read(args.run / 'comparison_index.json'), read(args.run / 'protocol.json')['samples'])
    result['index_sha256'] = sha(args.run / 'comparison_index.json')
    result['protocol_sha256'] = sha(args.run / 'protocol.json')
    result['analyzer_sha256'] = sha(__file__)
    with (args.run / 'analysis.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    with (args.run / 'person_means.csv').open('x', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'speaker_id', 'level', 'model', *METRICS])
        for p in result['records']:
            for level in LEVELS:
                for model in MODELS:
                    writer.writerow([p['sample_id'], p['speaker_id'], level, model,
                                     *[p['values'][level][model][m] for m in METRICS]])
    print(json.dumps(result['coverage']))


if __name__ == '__main__':
    main()
