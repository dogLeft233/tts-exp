"""Frozen strict15 sensitivity for the single authorized clipping exception."""
import argparse
import importlib.util
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    plan = read(args.run / 'analysis_protocol.json')
    subset = plan['strict_subset']
    selected = set(subset['retained_speaker_ids'])
    excluded = subset['excluded_speaker_id']
    if len(selected) != 15 or excluded in selected:
        raise ValueError('Frozen15 must exclude only the preprocessing-exception speaker')
    source = Path(__file__).with_name('nrt_two_arm_analysis_20261023.py')
    spec = importlib.util.spec_from_file_location('pilot_analysis', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    original_count = len(plan['samples'])
    plan['samples'] = [r for r in plan['samples'] if r['speaker_id'] in selected]
    ids = {r['sample_id'] for r in plan['samples']}
    plan['score_bindings'] = [r for r in plan['score_bindings'] if r['sample_id'] in ids]
    if original_count != 16 or len(plan['samples']) != 15 or len(plan['score_bindings']) != 75:
        raise ValueError('Strict subset coverage')
    result = m.analyze(plan)
    result['scope'] = 'Predefined strict15 sensitivity, not a new independent experiment; main16 unchanged'
    result['excluded_speaker_id'] = excluded
    result['analysis_protocol_sha256'] = m.sha(args.run / 'analysis_protocol.json')
    with (args.run / 'strict_subset_analysis.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'coverage': result['coverage']}))


if __name__ == '__main__':
    main()
