"""Independent reconstruction of the actual boundary intervention dataset."""
import importlib.util
import json
from pathlib import Path

import pytest


def test_native_parity_target_invariance_and_independent_statistics():
    repo=Path(__file__).resolve().parents[1]
    root=repo/'runs/avs_boundary_content_cloud_20261028_attempt01'
    if not (root/'items.json').exists():
        pytest.skip('Real boundary intervention artifacts unavailable')
    spec=importlib.util.spec_from_file_location('boundary_independent_audit',repo/'scripts/experiments/avs_boundary_content_independent_audit_20261028.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reconstructed=module.audit(root)
    result=json.loads((root/'parent_independent_audit.json').read_text())
    assert result['status']=='PASS' and result['clips']==len(reconstructed)==80
    assert result['sources']==20
    assert result['max_native_parity_independently_verified']<=1e-5
    assert max(result['max_endpoint_error'],result['max_stat_error'])<1e-10
    assert result['valid_landmark_frames']>0
