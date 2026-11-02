"""Execute official AVE and NeRFDataset with two explicit split fixes only.

AST extraction avoids renderer/CUDA-extension imports. Audio, pose, landmark,
and collate definitions stay verbatim. No generator is instantiated or trained.
"""
import argparse
import ast
import gc
import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from packaging import version as pver
from scipy import signal
from scipy.spatial.transform import Rotation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def definitions(path, names, scope, provenance, transformer=None):
    source = path.read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert set(n.name for n in nodes) == set(names)
    for n in nodes:
        provenance.append({'file': str(path), 'symbol': n.name, 'line': n.lineno,
                           'source_sha256': hashlib.sha256(ast.get_source_segment(source, n).encode()).hexdigest()})
    module = ast.Module(body=nodes, type_ignores=[])
    if transformer:
        module = transformer.visit(module)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)


class SplitFixes(ast.NodeTransformer):
    def __init__(self):
        self.test_fixes = 0
        self.val_fixes = 0

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == '_split' and isinstance(node.value, ast.IfExp):
                assert ast.unparse(node.value) == "'val' if type == 'test' else type"
                node.value = ast.Name(id='type', ctx=ast.Load())
                self.test_fixes += 1
            elif name == 'frames' and ast.unparse(node.value) == 'frames[:100]':
                node.value = ast.Name(id='frames', ctx=ast.Load())
                self.val_fixes += 1
        return self.generic_visit(node)


def parser_from_official(path):
    tree = ast.parse(path.read_text())
    block = next(n for n in tree.body if isinstance(n, ast.If) and any(
        isinstance(x, ast.Name) and x.id == '__name__' for x in ast.walk(n.test)))
    nodes = []
    for n in block.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'opt' for t in n.targets):
            break
        nodes.append(n)
    ns = {'argparse': argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['parser']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--official', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'use a fresh output'
    a.output.mkdir(parents=True)
    dataset_root, official = a.dataset.resolve(), a.official.resolve()
    protocol = json.loads((dataset_root / 'protocol.json').read_text())
    manifest = json.loads((dataset_root / 'manifest.json').read_text())
    assert subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() == protocol['official_commit']
    weight = official / 'nerf_triplane/checkpoints/audio_visual_encoder.pth'
    assert sha(weight) == protocol['ave_weight_sha256']
    os.chdir(official)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    provenance = []
    ns = dict(np=np, torch=torch, nn=nn, F=F, librosa=librosa, signal=signal,
              basename=lambda x: Path(x).name, cv2=cv2, os=os, json=json, glob=glob,
              tqdm=tqdm, Rotation=Rotation, DataLoader=DataLoader, pver=pver)
    src = official / 'nerf_triplane'
    definitions(src / 'utils.py', ['load_wav', 'preemphasis', 'melspectrogram', '_stft',
                '_linear_to_mel', '_build_mel_basis', '_amp_to_db', '_normalize',
                'AudDataset', 'get_audio_features', 'get_bg_coords', 'custom_meshgrid', 'get_rays'], ns, provenance)
    definitions(src / 'network.py', ['Conv2d', 'AudioEncoder'], ns, provenance)
    fixes = SplitFixes()
    definitions(src / 'provider.py', ['nerf_matrix_to_ngp', 'smooth_camera_path', 'NeRFDataset'], ns, provenance, fixes)
    assert fixes.test_fixes == fixes.val_fixes == 1
    capture = []
    original_init = ns['AudioEncoder'].__init__

    def observe_init(self):
        original_init(self)
        self.register_forward_hook(lambda owner, args, out: capture.append(out.detach().cpu().numpy().copy()))

    ns['AudioEncoder'].__init__ = observe_init
    parser = parser_from_official(official / 'main.py')
    report = {'schema': 'synctalk_provider_qualification_v1', 'generator_training_runs': 0,
              'generator_forwards': 0, 'official_commit': protocol['official_commit'],
              'ave_weight_sha256': sha(weight), 'provenance': provenance,
              'adapter_fixes': {'test_reads_test_json': fixes.test_fixes, 'val_not_truncated_100': fixes.val_fixes},
              'environment': {'torch': torch.__version__, 'numpy': np.__version__,
                              'librosa': librosa.__version__, 'device': torch.cuda.get_device_name(0)},
              'arms': [], 'repeats': []}
    for arm in manifest['arms']:
        name, label = arm['split'], arm['id']
        data = dataset_root / arm['path']
        metadata = json.loads((data / f'transforms_{name}.json').read_text())
        ids = np.array([f['aud_id'] for f in metadata['frames']], dtype=np.int64)

        def build():
            capture.clear()
            opt = parser.parse_args([str(data), '-O', '--asr_model', 'ave', '--preload', '0'])
            opt.fp16 = False
            opt.exp_eye = True
            opt.cuda_ray = False
            torch.manual_seed(42)
            d = ns['NeRFDataset'](opt, device=torch.device('cuda'), type=name)
            embedding = np.concatenate(capture)
            padded = np.concatenate([np.repeat(embedding[:1], 2, 0), embedding, np.repeat(embedding[-1:], 2, 0)])
            assert np.max(ids) < len(padded), 'forbidden provider min/clamp fallback'
            aligned = padded[ids, None, :]
            assert np.array_equal(d.auds.cpu().numpy(), aligned)
            return d, padded, aligned

        d, padded, aligned = build()
        out = a.output / name / label
        out.mkdir(parents=True)
        np.save(out / 'ave_padded.npy', padded)
        np.save(out / 'ave_aligned.npy', aligned)
        n = len(metadata['frames'])
        assert len(d.poses) == len(d.images) == len(d.auds) == n
        collates = []
        for i in [0, n // 2, n - 1]:
            item = d.collate([i])
            assert torch.isfinite(item['auds']).all() and torch.isfinite(item['images']).all()
            expected = ns['get_audio_features'](torch.from_numpy(aligned), 2, i)
            assert torch.equal(item['auds'].cpu(), expected)
            collates.append({'index': i, 'auds': list(item['auds'].shape),
                             'images': list(item['images'].shape), 'rays': list(item['rays_o'].shape)})
        row = {'split': name, 'id': label, 'gain_db': arm['gain_db'], 'frames': n,
               'padded_shape': list(padded.shape), 'aligned_shape': list(aligned.shape),
               'padded_sha256': sha(out / 'ave_padded.npy'), 'aligned_sha256': sha(out / 'ave_aligned.npy'),
               'collate_checks': collates, 'no_audio_clamp': True,
               'audio_sha256': sha(data / 'aud.wav')}
        report['arms'].append(row)
        if arm['gain_db'] == 0:
            del d
            repeated, p2, a2 = build()
            err = float(np.max(np.abs(a2 - aligned)))
            assert np.array_equal(p2, padded) and err == 0
            np.save(out / 'ave_padded_repeat.npy', p2)
            report['repeats'].append({'split': name, 'max_abs_error': err, 'padded_bitwise_equal': True})
            del repeated
        else:
            del d
        del padded, aligned
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({'completed': name + '/' + label, 'frames': n}), flush=True)
    (a.output / 'provider_qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'arms': len(report['arms']), 'repeats': len(report['repeats'])}), flush=True)


if __name__ == '__main__':
    main()
