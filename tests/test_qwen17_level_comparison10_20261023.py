import importlib.util
from pathlib import Path

def load():
    p=Path(__file__).resolve().parents[1]/'scripts/experiments/qwen17_level_comparison10_20261023/experiment.py'
    s=importlib.util.spec_from_file_location('q17_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_planned_factorial_new_cells():
    m=load();roles=m.planned_roles()
    assert len(roles)==13 and len(set(roles))==13
    assert 'N_raw' in roles and 'N_norm' not in roles
    assert all(m.role('L17',level,d) in roles for level in ['norm','raw'] for d in [1,2,3])

def test_save_refuses_overwrite(tmp_path):
    import pytest
    m=load();p=tmp_path/'frozen.json';m.save(p,{'x':1})
    with pytest.raises(FileExistsError):m.save(p,{'x':2})
    assert m.read(p)=={'x':1}

def test_continuation_rejects_unbound_exception_before_writing(tmp_path,monkeypatch):
    import pytest
    p=Path(__file__).resolve().parents[1]/'scripts/experiments/qwen17_level_comparison10_20261023/continuation_attempt02.py'
    s=importlib.util.spec_from_file_location('q17_cont_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    monkeypatch.setattr(m,'RUN',tmp_path/'new_attempt')
    fake={'flagged':[{'sample_id':'BAC009S0016W0121','model':'L06','draw_index':3,
          'source_sha256':'wrong','modes':{'raw':{'abs_gt1':8,'affected_indices':[]}}}]}
    monkeypatch.setattr(m,'read',lambda p:fake)
    with pytest.raises(ValueError,match='source/index'):m.prepare()
    assert not m.RUN.exists()
