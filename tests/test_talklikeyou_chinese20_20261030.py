"""Protect native length eligibility and source-level paired inference."""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]/'scripts/experiments/talklikeyou_chinese20_20261030'
if not SOURCE.exists():
    SOURCE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('tly_'+name, SOURCE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_five_second_audio_does_not_imply_five_second_video():
    frames = load('prepare').native_frames
    assert frames(80000) == 120
    assert frames(320512) == 495  # Previously measured official20.032s smoke.
    assert frames(83200) == 125
    for samples in [39680,76000,80000,80640,95984,126704]:
        assert 0 < samples-frames(samples)*640 < 4000


def test_only_complete_source_pairs_are_analyzed():
    analyze = load('analyze')
    rows = [dict(id='n', source='s', speaker='p', arm='N', strict_reference=True),
            *[dict(id=f't{i}', source='s', speaker='p', arm='T', strict_reference=True) for i in range(3)]]
    pairs, missing = analyze.make_pairs(rows, dict(n=2., t0=3., t1=3., t2=6.))
    assert not missing and len(pairs) == 1 and pairs[0]['T'] == 4. and pairs[0]['delta'] == 2.
    pairs, missing = analyze.make_pairs(rows, dict(n=2., t0=3., t1=3.))
    assert not pairs and missing[0]['unmeasurable_ids'] == ['t2']


def test_holm_retains_family_size_when_endpoints_unmeasurable():
    values = load('analyze').holm([.001, .02, 1., 1., 1., 1., 1., 1., 1.])
    assert abs(values[0]-.009) < 1e-12 and abs(values[1]-.16) < 1e-12
    assert values[2:] == [1.]*7


def test_classifier_audit_uses_saved_fp32_precision_and_rejects_changes():
    import pytest
    torch = pytest.importorskip('torch')
    pytest.importorskip('soundfile')
    import numpy as np
    # Actual saved SparseSync logits exposed a 1.266e-7 FP64/FP32 difference.
    logits = [-.951678454875946,-1.2415474653244019,.5668298006057739,-2.38632869720459,
              -3.429565191268921,1.9692002534866333,-1.4209755659103394,2.3077569007873535,
              4.183653831481934,-3.80523681640625,14.509099960327148,10.091157913208008,
              -4.355128765106201,3.964902400970459,-2.1808111667633057,-1.480488657951355,
              -1.189263939857483,-3.8314788341522217,.6334486603736877,-1.8454959392547607,
              -4.642661094665527]
    saved = torch.tensor(logits, dtype=torch.float32).softmax(0).tolist()
    verify = load('audit_precision_replay').verify_classifier_probabilities
    error = verify(logits, saved, saved[10])
    assert 1e-7 < error < 2e-7
    corrupt = saved.copy()
    corrupt[10] = float(np.nextafter(np.float32(saved[10]), np.float32(0)))
    with pytest.raises(AssertionError, match='replay mismatch'):
        verify(logits, corrupt, corrupt[10])
