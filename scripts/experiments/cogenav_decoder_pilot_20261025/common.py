"""Frozen clock contracts and trainable CoGenAV-to-HiFiGAN adapter."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEED = 20261025
SR = 16000
HOP25 = 640
HOP50 = 320
SEGMENT25 = 11
CONTEXT25 = 16


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def configure(gpu=True):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if gpu:
        torch.cuda.set_per_process_memory_fraction(7.5/16)


def resource_gate(run, label, initial=False):
    """Never control other jobs; wait while external footprint exceeds 6 GiB."""
    path = Path(run)/'resource_events.jsonl'
    waited = 0
    while True:
        text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True)
        external = 0
        for line in text.splitlines():
            pid, mem = line.split(',')
            if int(pid) != os.getpid():
                external += int(mem)
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'],text=True).strip())
        if external <= 6144 and (not initial or free >= 10240):
            if waited:
                with path.open('a') as f:
                    f.write(json.dumps(dict(event='RESUME',label=label,external_mib=external,waited_s=waited))+'\n')
            return
        with path.open('a') as f:
            f.write(json.dumps(dict(event='WAIT_EXTERNAL_GPU',label=label,external_mib=external,
                                    free_mib=free,unix=time.time()))+'\n')
        time.sleep(30)
        waited += 30


def teacher_grid(native, frames25):
    """Preserve official WavLM native 400/320 grid, mask replicated tail positions."""
    target = 2*frames25
    valid = min(target, native.shape[0])
    if valid == 0:
        raise ValueError('Empty teacher')
    cropped = native[:target]
    if len(cropped) < target:
        cropped = torch.cat([cropped, cropped[-1:].expand(target-len(cropped), -1)])
    mask = torch.arange(target) < valid
    return cropped, mask


def context_crop(a, teacher, teacher_mask, wave, start25, segment25=SEGMENT25, context25=CONTEXT25):
    """All crops share absolute 40ms start; loss center has exactly segment25*640 samples."""
    stop25 = start25+segment25
    if start25 < 0 or stop25 > a.shape[0]:
        raise ValueError('Crop outside feature support')
    indices = torch.arange(start25-context25, stop25+context25).clamp(0, len(a)-1)
    a_crop = a[indices]
    t_crop = teacher[2*start25:2*stop25]
    mask = teacher_mask[2*start25:2*stop25]
    wav = wave[start25*HOP25:stop25*HOP25]
    if len(wav) < segment25*HOP25:
        wav = F.pad(wav, (0,segment25*HOP25-len(wav)))
    return a_crop,t_crop,mask,wav


class Residual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.GELU(), nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),
                                 nn.GELU(), nn.Conv1d(channels,channels,1))

    def forward(self,x):
        return x+self.net(x)


class Adapter(nn.Module):
    def __init__(self, mean=None, std=None, channels=512):
        super().__init__()
        self.register_buffer('mean', torch.zeros(768) if mean is None else mean.float())
        self.register_buffer('std', torch.ones(768) if std is None else std.float().clamp_min(1e-4))
        self.stem = nn.Conv1d(768,channels,3,padding=1)
        self.blocks = nn.Sequential(*(Residual(channels,d) for d in (1,2,4)))
        self.head = nn.Conv1d(channels,1024,1)

    def forward(self,a):
        x=((a-self.mean)/self.std).transpose(1,2)
        x=self.blocks(self.stem(x))
        x=F.interpolate(x,scale_factor=2,mode='linear',align_corners=False)
        return self.head(x).transpose(1,2)


def freeze_generator(generator, frozen):
    for p in generator.parameters():
        p.requires_grad_(not frozen)
    generator.eval() if frozen else generator.train()


def feature_loss(pred, target, mask):
    difference=(pred-target).square().mean(-1)
    return (difference*mask).sum()/mask.sum().clamp_min(1)


class SpectralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=1024,win_length=1024,
            hop_length=320,n_mels=80,power=1.,norm='slaney',mel_scale='slaney',f_min=0,f_max=8000)
        for n in (512,1024,2048):
            self.register_buffer('window'+str(n),torch.hann_window(n))

    def forward(self,pred,target):
        logmel=(self.mel(pred).clamp_min(1e-5).log()-self.mel(target).clamp_min(1e-5).log()).abs().mean()
        terms=[]
        for n in (512,1024,2048):
            kw=dict(n_fft=n,hop_length=n//4,win_length=n,window=getattr(self,'window'+str(n)),return_complex=True)
            p=torch.stft(pred,**kw).abs().clamp_min(1e-5)
            t=torch.stft(target,**kw).abs().clamp_min(1e-5)
            convergence=torch.linalg.vector_norm(p-t,dim=(-2,-1))/torch.linalg.vector_norm(t,dim=(-2,-1)).clamp_min(1e-5)
            terms.append(convergence.mean()+(p.log()-t.log()).abs().mean())
        mrstft=torch.stack(terms).mean()
        return logmel,mrstft
