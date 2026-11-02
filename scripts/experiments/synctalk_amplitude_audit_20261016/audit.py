"""Frozen-gain audit of verbatim official SyncTalk audio functions/classes.

AST extraction excludes renderer imports only; selected definitions are unmodified.
All intermediate arrays and comparisons are persisted. No training is performed.
"""
import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from transformers import Wav2Vec2FeatureExtractor

COMMIT = '43079fb3f349675f6cb33031bb07262f7566cdd5'
GAINS = (-6, 0, 6)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract(path, names, scope, provenance):
    text = path.read_text()
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for n in nodes:
        src = ast.get_source_segment(text, n)
        provenance.append({'file': str(path), 'symbol': n.name, 'start': n.lineno,
                           'end': n.end_lineno, 'source_sha256': hashlib.sha256(src.encode()).hexdigest()})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def summary(x):
    x = np.asarray(x)
    v = x.astype(np.float64)
    return {'shape': list(x.shape), 'dtype': str(x.dtype), 'finite': bool(np.isfinite(v).all()),
            'rms': float(np.sqrt(np.mean(v * v))), 'peak': float(np.max(np.abs(v))),
            'mean': float(np.mean(v)), 'std': float(np.std(v)), 'min': float(np.min(v)),
            'max': float(np.max(v)), 'sha256': hashlib.sha256(x.tobytes()).hexdigest()}


def difference(x, ref):
    x, ref = np.asarray(x, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if x.shape != ref.shape:
        return {'same_shape': False, 'shape': list(x.shape), 'reference_shape': list(ref.shape)}
    d = x - ref
    norm = np.linalg.norm(ref.ravel())
    denom = np.linalg.norm(x.ravel()) * norm
    return {'same_shape': True, 'max_abs_diff': float(np.max(np.abs(d))),
            'relative_l2': float(np.linalg.norm(d.ravel()) / max(norm, 1e-30)),
            'cosine': float(np.dot(x.ravel(), ref.ravel()) / denom) if denom else None,
            'length_change': int(x.shape[0] - ref.shape[0])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Use a fresh output directory'
    a.output.mkdir(parents=True)
    actual = subprocess.check_output(['git', '-C', str(a.root / 'official'), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == COMMIT
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    provenance = []
    ns = dict(np=np, torch=torch, nn=nn, F=F, librosa=librosa, signal=signal, basename=lambda x: Path(x).name)
    src = a.root / 'official' / 'nerf_triplane'
    extract(src / 'utils.py', ['load_wav', 'preemphasis', 'melspectrogram', '_stft', '_linear_to_mel',
                              '_build_mel_basis', '_amp_to_db', '_normalize', 'AudDataset',
                              'get_audio_features'], ns, provenance)
    extract(src / 'network.py', ['Conv2d', 'AudioEncoder', 'AudioNet_ave', 'AudioAttNet'], ns, provenance)
    model = ns['AudioEncoder']().cuda().eval()
    weight = src / 'checkpoints/audio_visual_encoder.pth'
    ckpt = torch.load(weight, map_location='cpu', weights_only=True)
    loaded = model.load_state_dict({'audio_encoder.' + k: v for k, v in ckpt.items()}, strict=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(a.root)
    # DeepSpeech frontend alone; the TF graph is deliberately not loaded.
    from python_speech_features import mfcc
    ds = dict(np=np, mfcc=mfcc)
    extract(a.root / 'official/data_utils/deepspeech_features/deepspeech_features.py',
            ['conv_audio_to_deepspeech_input_vector'], ds, provenance)
    results = {'commit': COMMIT, 'gains_db': list(GAINS), 'inputs': [], 'stages': [], 'comparisons': [],
               'repeats': [], 'provenance': provenance, 'ave_weight_sha256': sha(weight),
               'ave_load_missing': list(loaded.missing_keys), 'ave_load_unexpected': list(loaded.unexpected_keys),
               'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                               'librosa': librosa.__version__, 'soundfile': sf.__version__,
                               'numpy': np.__version__, 'device': torch.cuda.get_device_name(0)},
               'hubert_model_run': False, 'deepspeech_model_run': False}
    for ix, source in enumerate(a.inputs):
        raw, sr = sf.read(source, dtype='float64', always_2d=False)
        assert sr == 16000 and raw.ndim == 1, 'Frozen primary inputs are mono 16k; no resampling confound'
        row = {'id': ix, 'path': str(source), 'source_sha256': sha(source), 'sample_rate': sr,
               'source': summary(raw), 'conditions': []}
        results['inputs'].append(row)
        all_stages = {}
        for db in GAINS:
            gain = 10 ** (db / 20)
            intended = raw * gain
            if np.max(np.abs(intended)) >= 1:
                row['conditions'].append({'gain_db': db, 'status': 'SKIPPED_UNSAFE_PEAK',
                                          'intended_peak': float(np.max(np.abs(intended)))})
                continue
            out = a.output / f'input{ix}_gain{db:+d}'
            out.mkdir()
            wav = out / 'audio_double.wav'
            sf.write(wav, intended, sr, subtype='DOUBLE')
            back, back_sr = sf.read(wav, dtype='float64')
            assert back_sr == sr and np.array_equal(back, intended)
            y = ns['load_wav'](str(wav), 16000)
            pre = ns['preemphasis'](y, .97)
            mag = np.abs(ns['_stft'](pre))
            melamp = ns['_linear_to_mel'](mag)
            absdb = ns['_amp_to_db'](melamp) - 20
            mel = ns['_normalize'](absdb)
            dataset = ns['AudDataset'](str(wav))
            assert np.array_equal(dataset.orig_mel, mel.T)
            windows = torch.stack([dataset[j] for j in range(len(dataset))])
            def encode(w):
                with torch.no_grad():
                    return torch.cat([model(chunk.cuda()).cpu() for chunk in w.split(64)]).numpy()
            embedding = encode(windows)
            repeat = encode(windows)
            padded = np.concatenate([np.repeat(embedding[:1], 2, axis=0), embedding,
                                     np.repeat(embedding[-1:], 2, axis=0)])
            provider = torch.from_numpy(padded).unsqueeze(0).permute(1, 0, 2).float()
            conditioning = torch.stack([ns['get_audio_features'](provider, 2, j)
                                        for j in range(len(provider))]).numpy()
            huinput = extractor(back, return_tensors='pt', sampling_rate=16000).input_values
            stages = {'double_waveform': back, 'decoded_float32': y, 'preemphasis': pre,
                      'stft_magnitude': mag, 'mel_amplitude': melamp, 'absolute_db_minus20': absdb,
                      'mapped_mel': mel, 'ave_windows': windows.numpy(), 'ave_embedding': embedding,
                      'ave_provider_padded': padded, 'ave_conditioning_att2': conditioning,
                      'hubert_processor': huinput.numpy()}
            # Float WAV is unsafe for the unmodified DS int16 cast. Record it explicitly.
            stages['ds_official_cast_from_float'] = back.astype(np.int16)
            # Separate PCM16 interface diagnostic. Quantization is explicit, not a pure-gain arm.
            pcm = np.rint(back * 32768).astype(np.int16)
            stages['ds_pcm16_quantized'] = pcm
            stages['ds_mfcc'] = mfcc(pcm, samplerate=sr, numcep=26)
            stages['ds_standardized_context'] = ds['conv_audio_to_deepspeech_input_vector'](pcm, sr, 26, 9)
            results['repeats'].append({'input_id': ix, 'gain_db': db, 'stage': 'ave_embedding',
                                       **difference(repeat, embedding)})
            results['repeats'].append({'input_id': ix, 'gain_db': db, 'stage': 'mapped_mel',
                                       **difference(ns['melspectrogram'](y), mel)})
            all_stages[db] = stages
            row['conditions'].append({'gain_db': db, 'status': 'RUN', 'gain': gain,
                                      'samples': len(back), 'sample_rate': back_sr,
                                      'pure_scalar_max_error_double': float(np.max(np.abs(back - raw * gain))),
                                      'lower_clip_fraction': float(np.mean(absdb <= -100)),
                                      'upper_clip_fraction': float(np.mean(absdb >= 0)),
                                      'amplitude_floor_fraction': float(np.mean(melamp <= 1e-5)),
                                      'path': str(wav)})
            for name, values in stages.items():
                dest = out / (name + '.npy')
                np.save(dest, values)
                results['stages'].append({'input_id': ix, 'gain_db': db, 'stage': name,
                                           'path': str(dest), **summary(values)})
        base = all_stages[0]
        for db, stages in all_stages.items():
            for name, values in stages.items():
                results['comparisons'].append({'input_id': ix, 'gain_db': db, 'stage': name,
                                                **difference(values, base[name])})
            if db:
                unclipped = (base['absolute_db_minus20'] > -100) & (base['absolute_db_minus20'] < 0)
                unclipped &= (stages['absolute_db_minus20'] > -100) & (stages['absolute_db_minus20'] < 0)
                error = (stages['mapped_mel'] - base['mapped_mel'])[unclipped] - db * .08
                results['comparisons'].append({'input_id': ix, 'gain_db': db,
                                               'stage': 'unclipped_mel_shift_formula',
                                               'expected_shift': db * .08, 'max_abs_error': float(np.max(np.abs(error))),
                                               'cells': int(unclipped.sum())})
        (a.output / 'statistics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print('Completed input', ix, source, flush=True)
    print(json.dumps({'output': str(a.output), 'statistics_sha256': sha(a.output / 'statistics.json'),
                      'ave_weight_sha256': sha(weight), 'inputs': len(results['inputs'])}))


if __name__ == '__main__':
    main()
