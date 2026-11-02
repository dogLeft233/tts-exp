"""Guard native temporal support and the frozen source-pair interpretation."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'scripts/experiments/echomimic_v3_chinese20_20261030'


def load(name):
    spec = importlib.util.spec_from_file_location('echo_'+name, ROOT/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_frames_never_extend_or_default_cap_long_audio():
    frames = load('prepare').native_frames
    for samples in [39680,76000,80000,80640,95984,126704]:
        actual = frames(samples)
        assert actual % 4 == 1
        assert actual*640 <= samples
        assert 0 <= samples-actual*640 < 2560
    assert frames(126704) == 197  # Longest cohort clip must exceed demo's81.
    assert frames(80000) == 125  # Five-second support remains measurable.


def test_source_pairing_does_not_treat_three_draws_as_three_people():
    analyze = load('analyze')
    rows = [dict(id='n',source='s',speaker='p',arm='N',strict_reference=True),
            *[dict(id=f't{i}',source='s',speaker='p',arm='TTS',strict_reference=True) for i in range(3)]]
    pairs, missing = analyze.make_pairs(rows,dict(n=2.,t0=3.,t1=3.,t2=6.))
    assert not missing and len(pairs) == 1
    assert pairs[0]['T'] == 4. and pairs[0]['delta'] == 2.
    pairs, missing = analyze.make_pairs(rows,dict(n=2.,t0=3.,t1=3.))
    assert not pairs and missing[0]['unmeasurable_ids'] == ['t2']


def test_attention_excludes_padding_against_independent_sliced_softmax():
    import torch
    attention = load('attention').attention
    generator = torch.Generator().manual_seed(52)
    q = torch.randn(2,4,3,8,generator=generator,dtype=torch.float64)
    k = torch.randn(2,7,3,8,generator=generator,dtype=torch.float64)
    v = torch.randn(2,7,3,8,generator=generator,dtype=torch.float64)
    actual = attention(q,k,v,k_lens=[3,5],softmax_scale=.2,q_scale=1.5)
    for i, length in enumerate([3,5]):
        qs,ks,vs = q[i].transpose(0,1),k[i,:length].transpose(0,1),v[i,:length].transpose(0,1)
        expected = (torch.softmax((qs*1.5)@ks.transpose(-1,-2)*.2,dim=-1)@vs).transpose(0,1)
        torch.testing.assert_close(actual[i],expected,atol=1e-12,rtol=1e-12)
    # Arbitrarily changing excluded keys/values must leave the output unchanged.
    k[0,3:] = 1000
    v[0,3:] = -1000
    changed = attention(q,k,v,k_lens=[3,5],softmax_scale=.2,q_scale=1.5)
    torch.testing.assert_close(actual,changed,atol=1e-12,rtol=1e-12)
