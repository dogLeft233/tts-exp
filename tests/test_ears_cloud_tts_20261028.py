import importlib.util
from pathlib import Path
import numpy as np

def module():
    p = Path(__file__).resolve().parents[1] / 'scripts/experiments/ears_cloud_tts_20261028.py'
    spec = importlib.util.spec_from_file_location('ears_cloud', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def test_english_text_preserves_word_boundaries():
    assert module().english_text('My head\n aches  dreadfully now.') == 'My head aches dreadfully now.'

def test_mono_resample_preserves_duration_and_amplitude_without_rms_scaling():
    m = module(); x = np.full((48000, 2), 0.1)
    y = m.mono_resample(x, 48000, 16000)
    assert y.shape == (16000,)
    assert np.max(np.abs(y[20:-20] - .1)) < 1e-8
