"""Frozen May pilot contracts and official data/model entry points."""
import argparse
import ast
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/dev/shm/synctalk-may-training-pilot-20261017')
RUNTIME = Path('/dev/shm/synctalk-may-gain-inference-20261016')
OFFICIAL = RUNTIME / 'official'
DATA = Path('/dev/shm/synctalk-gain-training-data-20261017/dataset')
FEATURES = Path('/dev/shm/synctalk-gain-training-data-20261017/features_attempt03')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def tensor_state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode() + b'\0' + str(x.dtype).encode() + str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def seed_domain(seed, step, domain):
    value = int.from_bytes(hashlib.sha256(f'{seed}/{step}/{domain}'.encode()).digest()[:4], 'little')
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    return value


def label(db):
    return {0:'gain_0', -6:'gain_m6', 6:'gain_p6'}[db]


def parser():
    tree = ast.parse((OFFICIAL / 'main.py').read_text())
    block = next(n for n in tree.body if isinstance(n, ast.If) and any(
        isinstance(x, ast.Name) and x.id == '__name__' for x in ast.walk(n.test)))
    nodes = []
    for n in block.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'opt' for t in n.targets):
            break
        nodes.append(n)
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OFFICIAL/'main.py'), 'exec'), scope)
    return scope['parser']


def options(split, db, protocol):
    directory = DATA / 'arms' / split / label(db)
    p = FEATURES / split / label(db)
    manifest = json.loads((DATA/'manifest.json').read_text())
    q = json.loads((FEATURES/'provider_qualification.json').read_text())
    assert q['official_commit'] == protocol['official_commit'] and q['generator_training_runs'] == 0
    arm = next(r for r in manifest['arms'] if r['split'] == split and r['gain_db'] == db)
    feature = next(r for r in q['arms'] if r['split'] == split and r['gain_db'] == db)
    assert sha(DATA/'protocol.json') == protocol['data_protocol_sha256']
    assert sha(directory/'aud.wav') == arm['audio_sha256'] == feature['audio_sha256']
    assert sha(p/'ave_padded.npy') == feature['padded_sha256']
    assert sha(p/'ave_aligned.npy') == feature['aligned_sha256']
    padded, aligned = np.load(p/'ave_padded.npy'), np.load(p/'ave_aligned.npy')
    transform = json.loads((directory/f'transforms_{split}.json').read_text())
    ids = [f['aud_id'] for f in transform['frames']]
    assert min(ids) >= 0 and max(ids) < len(padded)
    assert np.array_equal(aligned, padded[ids, None, :])
    opt = parser().parse_args([str(directory), '-O', '--asr_model', 'ave', '--preload', '0',
                              '--seed', str(protocol['seed']), '--ckpt', 'scratch',
                              '--iters', str(protocol['steps_per_model']),
                              '--num_rays', str(protocol['num_rays'])])
    opt.fp16 = protocol['fp16_training']
    opt.exp_eye = True
    opt.cuda_ray = True
    opt.verified_ave_cache = str(p/'ave_padded.npy')
    assert not opt.finetune_lips and opt.patch_size == 1 and not opt.torso
    assert not opt.test_train and opt.aud == '' and not opt.train_camera
    return opt


def dataset(split, db, protocol):
    from nerf_triplane.provider import NeRFDataset
    d = NeRFDataset(options(split, db, protocol), device=torch.device('cuda'), type=split)
    expected = {'train':4150,'val':550,'test':1100}[split]
    assert len(d.poses) == len(d.images) == len(d.auds) == expected
    reference = np.load(FEATURES/split/label(db)/'ave_aligned.npy')
    assert np.array_equal(d.auds.cpu().numpy(), reference)
    return d


def configure():
    os.chdir(OFFICIAL)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def bind_training_state(model, train):
    model.aud_features = train.auds
    model.eye_area = train.eye_area
    model.poses = train.poses


def model_metadata(model):
    return {name:getattr(model,name) for name in ['mean_count','mean_density','iter_density','local_step']
            if hasattr(model,name)}


def load_metadata(model, metadata):
    for k,v in metadata.items():
        setattr(model,k,v)
