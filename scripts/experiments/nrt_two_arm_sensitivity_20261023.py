"""Descriptive reliability sensitivity; never replaces frozen all-draw analysis."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--kind', choices=['cloud', 'history'], required=True)
    args = p.parse_args()
    run = args.run
    result = read(run / 'analysis.json')
    source = Path(__file__).with_name('nrt_two_arm_analysis_20261023.py')
    spec = importlib.util.spec_from_file_location('pilot_analysis', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    qualification = {}
    if args.kind == 'cloud':
        old = read(run.parent / 'qwen_nrt_expansion40_20261022_attempt01/analysis.json')
        qualification = {r['sample_id']: r['draw_reliability'] for r in old['records']}
    else:
        protocol = read(run / 'protocol.json')
        references = {r['sample_id']: r for r in protocol['samples']}
        generation = run / 'generation/tts_manifest.json'
        if not generation.is_file():
            generation = run / 'source_attempt01/generation/tts_manifest.json'
        for row in read(generation)['records']:
            sid = row['sample_id']
            # Stored samples are raw24k outputs; source duration is the full original recording.
            ratio = row['samples'] / 24000 / references[sid]['duration_s']
            qualification.setdefault(sid, {})[row['role']] = {
                'qualified': row['eos_in_talker_sequences'] is True and .5 <= ratio <= 2,
                'eos_present': row['eos_in_talker_sequences'], 'duration_ratio': ratio}
    rows = []
    for record in result['records']:
        sid = record['sample_id']
        available = [role for role in ['T1', 'T2', 'T3'] if qualification[sid][role]['qualified']]
        row = {'sample_id': sid, 'speaker_id': record['speaker_id'], 'qualified_local_roles': available,
               'local_reliability': qualification[sid], 'metrics': {}}
        if available:
            for metric in ['C', 'D', 'offset']:
                roles = record['metrics'][metric]['roles']
                local = float(np.mean([roles[r] for r in available]))
                differences = {'localT_qualified-N': local - roles['N'],
                               'localT_qualified-R': local - roles['R']}
                if args.kind == 'cloud':
                    differences['cloudT-localT_qualified'] = record['metrics'][metric]['groups']['cloudT'] - local
                row['metrics'][metric] = differences
        rows.append(row)
    metrics = {}
    for metric in ['C', 'D', 'offset']:
        names = ['localT_qualified-N', 'localT_qualified-R']
        if args.kind == 'cloud':
            names.append('cloudT-localT_qualified')
        metrics[metric] = {name: m.summarize([r['metrics'][metric][name] for r in rows
                                             if r['qualified_local_roles']]) for name in names}
    output = {'status': 'DESCRIPTIVE_SUPPLEMENT_ONLY', 'records': rows, 'metrics': metrics,
              'source_speakers': len(rows), 'speakers_with_no_qualified_local_draw':
                  [r['speaker_id'] for r in rows if not r['qualified_local_roles']],
              'qualified_local_draws': sum(len(r['qualified_local_roles']) for r in rows),
              'planned_local_draws': 3 * len(rows),
              'interpretation': 'Post-treatment reliability conditioning can bias comparisons; 95% intervals descriptive. No main draw deleted or replaced. Cloud EOS is unavailable, so all cloud calls remain.'}
    with (run / 'generation_reliability_sensitivity.json').open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': output['status'], 'qualified_local_draws': output['qualified_local_draws'],
                      'planned_local_draws': output['planned_local_draws']}))


if __name__ == '__main__':
    main()
