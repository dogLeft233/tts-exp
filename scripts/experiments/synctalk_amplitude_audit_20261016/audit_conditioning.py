"""Strictly load May's audio MLP/attention and execute official encode_audio.

This is a generator-conditioning forward, not a rendered image or video.
"""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

from audit import difference, extract, sha, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    ns = dict(np=np, torch=torch, nn=nn, F=F, librosa=librosa, signal=signal,
              basename=lambda x: Path(x).name)
    provenance = []
    src = a.root / 'official/nerf_triplane'
    extract(src / 'utils.py', ['load_wav', 'preemphasis', 'melspectrogram', '_stft', '_linear_to_mel',
                              '_build_mel_basis', '_amp_to_db', '_normalize', 'AudDataset',
                              'get_audio_features'], ns, provenance)
    extract(src / 'network.py', ['Conv2d', 'AudioEncoder', 'AudioNet_ave', 'AudioAttNet'], ns, provenance)
    tree = ast.parse((src / 'network.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NeRFNetwork')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encode_audio')
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(src / 'network.py'), 'exec'), ns)
    weight = a.root / 'assets/trial_may/checkpoints/ngp.pth'
    state = torch.load(weight, map_location='cpu', weights_only=False)['model']
    net = ns['AudioNet_ave'](dim_aud=32).eval()
    att = ns['AudioAttNet'](dim_aud=32).eval()
    net.load_state_dict({k[len('audio_net.'):]: v for k, v in state.items() if k.startswith('audio_net.')}, strict=True)
    att.load_state_dict({k[len('audio_att_net.'):]: v for k, v in state.items() if k.startswith('audio_att_net.')}, strict=True)
    owner = SimpleNamespace(emb=False, att=2, audio_net=net, audio_att_net=att)
    def forward(x):
        with torch.no_grad():
            return torch.cat([ns['encode_audio'](owner, t) for t in torch.from_numpy(x).float()]).numpy()
    frontend = json.loads((a.root / 'results/frontend_attempt01/statistics.json').read_text())
    result = {'checkpoint_sha256': sha(weight), 'strict_audio_modules_loaded': True,
              'renderer_run': False, 'provenance': provenance,
              'stages': [], 'comparisons': [], 'repeats': []}
    variants = {}
    for row in frontend['stages']:
        if row['stage'] != 'ave_conditioning_att2':
            continue
        x = np.load(row['path'])
        y = forward(x)
        dest = a.output / f"input{row['input_id']}_gain{row['gain_db']:+d}_encoded.npy"
        np.save(dest, y)
        result['stages'].append({'input_id': row['input_id'], 'gain_db': row['gain_db'],
                                  'path': str(dest), **summary(y)})
        result['repeats'].append({'input_id': row['input_id'], 'gain_db': row['gain_db'],
                                  **difference(forward(x), y)})
        variants[row['input_id'], row['gain_db']] = y
    for (idx, db), y in variants.items():
        result['comparisons'].append({'input_id': idx, 'gain_db': db, **difference(y, variants[idx, 0])})
    maywav = a.root / 'assets/May/aud.wav'
    dataset = ns['AudDataset'](str(maywav))
    windows = torch.stack([dataset[j] for j in range(32)])
    ave = ns['AudioEncoder']().eval()
    ave.load_state_dict({'audio_encoder.' + k: v for k, v in torch.load(
        src / 'checkpoints/audio_visual_encoder.pth', map_location='cpu', weights_only=True).items()}, strict=True)
    with torch.no_grad():
        emb = ave(windows).numpy()
    cached = np.load(a.root / 'assets/May/aud_ave.npy')
    np.save(a.output / 'may_first32_ave.npy', emb)
    np.save(a.output / 'may_first32_windows.npy', windows.numpy())
    padded = np.concatenate([np.repeat(emb[:1], 2, 0), emb, np.repeat(emb[-1:], 2, 0)])
    provider = torch.from_numpy(padded).unsqueeze(0).permute(1,0,2)
    cond = torch.stack([ns['get_audio_features'](provider, 2, i) for i in range(4, 28)]).numpy()
    out = forward(cond)
    np.save(a.output / 'may_middle24_generator_conditioning.npy', out)
    result['may'] = {'source_wav_sha256': sha(maywav), 'source_info': str(sf.info(maywav)),
                     'dataset_frames': len(dataset), 'orig_mel_shape': list(dataset.orig_mel.shape),
                     'embedding': summary(emb), 'cache_shape': list(cached.shape),
                     'fresh_versus_supplied_cache_frames2_to33': difference(emb, cached[2:34]),
                     'encoded_conditioning': summary(out), 'repeat': difference(forward(cond), out),
                     'scope': 'First 32 official AVE windows, 24 interior conditioning frames; no rendered output'}
    (a.output / 'statistics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'output': str(a.output), 'statistics_sha256': sha(a.output / 'statistics.json'),
                      'may': result['may']}))


if __name__ == '__main__':
    main()
