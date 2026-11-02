"""Rebuild actual common-window artifacts independently of their producer."""
import importlib.util
import json
from pathlib import Path

import pytest


def test_saved_features_reconstruct_curves_pairs_and_cluster_intervals():
    repo=Path(__file__).resolve().parents[1]
    run=repo/'runs/common_window_curves_cloud_20261028_attempt01'
    if not (run/'items.json').exists():
        pytest.skip('Actual experiment artifacts are not present')
    spec=importlib.util.spec_from_file_location('independent_curve_audit',repo/'scripts/experiments/common_window_curves_independent_audit_20261028.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _,pairs=module.audit(run)
    result=json.loads((run/'parent_independent_audit.json').read_text())
    assert result['status']=='PASS' and result['clips']==80 and result['pairs']==len(pairs)==60
    assert result['sources']==20 and result['metrics']==5
    assert max(result['max_errors'].values())<1e-9
    assert result['controls']=={'sync':{'total':6,'passed':6},'avs':{'total':40,'passed':40},'co':{'total':160,'passed':160}}
