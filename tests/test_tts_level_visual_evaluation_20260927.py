"""The total storage ceiling includes earlier calibration artifacts."""
import pytest
from scripts.experiments import tts_level_visual_evaluation_20260927 as evaluation


def test_budget_includes_calibration_and_eval_files(tmp_path, monkeypatch):
    cal = tmp_path / 'full_firstcal.npz'
    with cal.open('wb') as handle:
        handle.truncate(40 << 20)
    folder = tmp_path / 'evaluation'
    folder.mkdir()
    with (folder / 'selected.npz').open('wb') as handle:
        handle.truncate(90 << 20)
    monkeypatch.setattr(evaluation, 'CAL', tmp_path)
    monkeypatch.setattr(evaluation.cal, 'disk', lambda extra: 6 << 30)
    assert evaluation.budget(10 << 20) == 6 << 30
    with pytest.raises(AssertionError, match='shared visual150MiB'):
        evaluation.budget(21 << 20)
