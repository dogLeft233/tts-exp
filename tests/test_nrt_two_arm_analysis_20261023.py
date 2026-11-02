import importlib.util
from pathlib import Path
import json
import pytest

spec = importlib.util.spec_from_file_location('paired_pilot', Path(__file__).parents[1] /
    'scripts/experiments/nrt_two_arm_analysis_20261023.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_repeated_draws_are_averaged_before_person_contrast():
    means, diffs = m.paired_values({'N': 4, 'T1': 1, 'T2': 4, 'T3': 10},
                                 {'N': ['N'], 'T': ['T1', 'T2', 'T3']}, {'T-N': ['T', 'N']})
    assert means['T'] == 5
    assert diffs['T-N'] == 1
    assert m.summarize([1, -1])['n'] == 2


def test_failed_role_retains_incomplete_person_and_never_imputes(tmp_path):
    bindings = []
    for role in ['N', 'T1']:
        p = tmp_path / (role + '.json')
        row = {'sample_id': 'a', 'role': role, 'status': 'COMPLETE',
               'sync_c': 3, 'sync_d': 7, 'av_offset': 0}
        if role == 'T1':
            row.update(status='FAILED', sync_c=None)
        p.write_text(json.dumps(row))
        bindings.append({'sample_id': 'a', 'role': role, 'path': str(p)})
    plan = {'samples': [{'sample_id': 'a', 'speaker_id': 's'}],
            'groups': {'N': ['N'], 'T': ['T1']}, 'contrasts': {'T-N': ['T', 'N']},
            'primary_contrasts_C': ['T-N'], 'score_bindings': bindings,
            'bootstrap': {'seed': 1, 'replicates': 100}, 'interpretation': 'pilot'}
    result = m.analyze(plan)
    assert result['status'] == 'INCOMPLETE_DESCRIPTIVE_ONLY'
    assert len(result['records']) == 1
    assert result['records'][0]['missing_roles'] == ['T1']
    assert result['metrics']['C']['contrasts']['T-N']['mean'] is None
    plan['samples'].append({'sample_id': 'b', 'speaker_id': 's'})
    with pytest.raises(ValueError, match='independent speaker'):
        m.analyze(plan)
