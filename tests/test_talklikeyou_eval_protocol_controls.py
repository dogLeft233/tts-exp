"""Diagnostic delay sign must not silently reverse observed codec latency."""
import importlib.util
from pathlib import Path

import numpy as np


def load_runner(filename='run.py'):
    path = Path(__file__).resolve().parents[1]/'scripts/experiments/talklikeyou_eval_protocol_controls_20261031'/filename
    spec = importlib.util.spec_from_file_location('protocol_controls', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delayed_and_scaled_signal():
    rng = np.random.default_rng(42)
    x = rng.normal(size=4000)
    result = load_runner().compare_audio(x, np.pad(x*.8, (321, 0)))
    assert result['lag_samples'] == 321
    assert abs(result['aligned_gain']-.8) < 1e-12
    assert result['aligned_cosine'] > .999999


def test_advanced_signal_and_identity():
    rng = np.random.default_rng(4)
    x = rng.normal(size=4000)
    compare = load_runner().compare_audio
    assert compare(x, x[117:])['lag_samples'] == -117
    exact = compare(x, x)
    assert exact['exact_equal'] and exact['lag_samples'] == 0
    assert exact['zero_lag_rmse_pcm_units'] == 0


def test_complete_source_support_and_draw_average():
    analyze = load_runner('analyze.py')
    rows = [dict(id=str(i), source='s', speaker='s', strict_reference=True,
                 arm='N' if i == 0 else 'T') for i in range(4)]
    lookup = {str(i): dict(cells={c: dict(status='PASS', **{k: float(i) for k in
              ['C', 'D', 'D0', 'offset', 'B']}) for c in analyze.CELLS}) for i in range(4)}
    pairs, missing = analyze.make_pairs(rows, lookup, analyze.CELLS)
    assert len(pairs) == 1 and not missing
    assert pairs[0]['cells']['original_mp4_25']['C']['delta'] == 2.
    lookup['3']['cells']['original_mp4_100']['status'] = 'UNMEASURABLE'
    pairs, missing = analyze.make_pairs(rows, lookup, analyze.CELLS)
    assert not pairs and len(missing) == 1


def test_exact_signflip_and_zero_effect():
    stats = load_runner('analyze.py').stats
    assert stats([1., 1.])['exact_signflip_p'] == .5
    assert stats([0., 0., 0.])['exact_signflip_p'] == 1.


def test_busy_gpu_waits_then_continues(monkeypatch, tmp_path):
    runner = load_runner()
    observations = iter([b'47000,99', b'47000,0'])
    sleeps = []
    monkeypatch.setattr(runner, 'RUN', tmp_path)
    monkeypatch.setattr(runner, 'capture', lambda command: next(observations))
    monkeypatch.setattr(runner.time, 'sleep', sleeps.append)
    runner.resource_check()
    assert sleeps == [30]
    assert runner.read(tmp_path/'resource_wait.json')['utilization'] == 99
