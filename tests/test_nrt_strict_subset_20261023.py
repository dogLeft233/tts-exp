import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('strict_subset', Path(__file__).parents[1] /
    'scripts/experiments/nrt_strict_subset_20261023.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_strict_subset_excludes_entire_exception_speaker_not_just_bad_draw(tmp_path, monkeypatch):
    samples, bindings = [], []
    for i in range(16):
        sid, speaker = f'a{i}', ('S0908' if i == 0 else f's{i}')
        samples.append({'sample_id': sid, 'speaker_id': speaker})
        for role in ['N', 'R', 'T1', 'T2', 'T3']:
            c = 4 if role in ['N', 'R'] else (100 if i == 0 else 5)
            path = tmp_path / f'{sid}_{role}.json'
            path.write_text(json.dumps({'sample_id': sid, 'role': role, 'status': 'COMPLETE',
                                        'sync_c': c, 'sync_d': 8, 'av_offset': 0}))
            bindings.append({'sample_id': sid, 'role': role, 'path': str(path)})
    plan = {'samples': samples, 'score_bindings': bindings,
            'groups': {'N': ['N'], 'R': ['R'], 'T': ['T1', 'T2', 'T3']},
            'contrasts': {'T-N': ['T', 'N']}, 'primary_contrasts_C': ['T-N'],
            'bootstrap': {'seed': 1, 'replicates': 100}, 'interpretation': 'pilot',
            'strict_subset': {'excluded_speaker_id': 'S0908',
                              'retained_speaker_ids': [s['speaker_id'] for s in samples[1:]]}}
    (tmp_path / 'analysis_protocol.json').write_text(json.dumps(plan))
    monkeypatch.setattr(sys, 'argv', ['strict', '--run', str(tmp_path)])
    m.main()
    result = json.loads((tmp_path / 'strict_subset_analysis.json').read_text())
    assert result['coverage']['finite_cells'] == 75
    assert result['metrics']['C']['contrasts']['T-N']['mean'] == 1
    assert len(result['records']) == 15
