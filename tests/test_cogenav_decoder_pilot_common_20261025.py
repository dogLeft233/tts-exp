import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('decoder_common',Path(__file__).parents[1]/'scripts/experiments/cogenav_decoder_pilot_20261025/common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)


def test_crop_preserves_absolute_grid_and_context():
    a=torch.arange(40.).view(40,1).expand(40,768)
    teacher=torch.arange(80.).view(80,1).expand(80,1024)
    wave=torch.arange(40*640.)
    x,t,m,w=common.context_crop(a,teacher,torch.ones(80,dtype=torch.bool),wave,7)
    assert x.shape==(43,768) and t.shape==(22,1024) and w.shape==(7040,)
    assert x[16,0]==7 and t[0,0]==14 and w[0]==7*640 and w[-1]==18*640-1
    assert x[0,0]==0  # edge context repeats; never changes center time


def test_teacher_tail_is_repeated_but_never_scored():
    native=torch.arange(9.).view(9,1).expand(9,1024)
    t,m=common.teacher_grid(native,5)
    assert torch.equal(t[:9],native) and t[-1,0]==8
    assert m.sum()==9 and not m[-1]
    pred=t.clone();pred[-1]+=10000
    assert common.feature_loss(pred[None],t[None],m[None])==0


def test_raw_affine_adapter_length_and_frozen_generator_gradient():
    torch.manual_seed(1)
    adapter=common.Adapter(mean=torch.ones(768),std=torch.ones(768)*2,channels=16)
    generator=torch.nn.Linear(1024,1)
    a=torch.randn(2,11,768)
    common.freeze_generator(generator,True)
    prediction=adapter(a)
    assert prediction.shape==(2,22,1024)
    generator(prediction).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in adapter.parameters())
    assert all(p.grad is None for p in generator.parameters())
    common.freeze_generator(generator,False)
    generator(adapter(a)).square().mean().backward()
    assert all(p.grad is not None for p in generator.parameters())


def test_spectral_identity_and_silence_are_finite_with_finite_gradients():
    criterion=common.SpectralLoss()
    for target in (torch.zeros(1,7040),torch.randn(1,7040)*.1):
        predicted=target.clone().requires_grad_(True)
        mel,stft=criterion(predicted,target)
        assert torch.isfinite(mel+stft)
        (mel+stft).backward()
        assert torch.isfinite(predicted.grad).all()
