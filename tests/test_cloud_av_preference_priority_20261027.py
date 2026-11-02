"""Only the exact paused runner may be resumed after priority scoring."""
import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/experiments'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('priority', SCRIPTS / 'cloud_av_preference_priority_20261027.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resume_gate_rejects_running_replaced_or_different_process():
    expected = dict(command='python older_evaluation.py', state='T', start_ticks='12345')
    assert module.same_paused_runner(expected, expected)
    for changed in [dict(state='S'), dict(start_ticks='12346'), dict(command='python unrelated.py')]:
        assert not module.same_paused_runner(dict(expected, **changed), expected)
