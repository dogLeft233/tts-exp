"""Contracts for the new cloud arm; no API or GPU work."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/experiments'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('cloud_av', SCRIPTS / 'cloud_av_preference_20261027.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sync_score_matches_official_lag_major_layout():
    import torch
    values = np.random.default_rng(42).normal(size=(113, 31)).astype('float32')
    lag_major = torch.stack([torch.from_numpy(values[:, i]) for i in range(31)])
    mean = torch.mean(lag_major, 1)
    result = module.sync_scalars(values)
    assert result['C'] == float(torch.median(mean) - mean.min())
    assert result['D'] == float(mean.min())
    assert result['offset'] == 15 - int(mean.argmin())


def test_metric_loading_ignores_raw_intermediates_and_keeps_precision():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'peavs').mkdir()
        value = 1.123456789
        (root / 'peavs/batch000.json').write_text(json.dumps([dict(filename='/tmp/source_C1.mp4', score=value)]))
        (root / 'peavs/batch000.raw.json').write_text('{bad intermediate')
        assert module.load_metric(root, 'peavs') == {'source_C1': value}


def test_stopped_production_does_not_block_but_running_does():
    stopped = 'Ts python /repo/scripts/experiments/audio200_nl17_controller_20261025.py'
    running = 'Sl python /repo/scripts/experiments/audio200_nl17_controller_20261025.py'
    assert module.active_production_commands([stopped]) == []
    assert module.active_production_commands([running]) == [running.split(None, 1)[1]]
