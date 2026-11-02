import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('parent_audit',Path(__file__).parents[1]/'scripts/experiments/cogenav_decoder_parent_audit_20261025.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_log_schedule_keeps_both_stage_boundaries():
    values=module.expected_logged_steps()
    assert len(values)==302
    assert values[0]==1 and values[-1]==6000
    assert 2000 in values and 2001 in values and 2020 in values
    assert len(values)==len(set(values))
