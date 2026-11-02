"""Integration check against the actual full original-model result artifacts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/experiments'))
from vocalist_official_results_audit_20261028 import audit


def test_actual_original_vocalist_all80_results():
    root = Path('/home/wjj/tts-audio/tts-exp/runs/vocalist_official_cloud_rescore_20261028_attempt01')
    result = audit(root)
    assert result['status'] == 'PASS'
    assert result['independently_recomputed_endpoints'] == 480
    assert result['checked_pair_endpoints'] == 360
