"""Prepare exact paired gain data from the sealed official May archive.

No model training, resampling, video re-encoding, or audio normalization.
Shared directories are junctions on Windows and relative symlinks on Linux.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import wave
import zipfile
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def read_pcm16(path):
    with wave.open(str(path), 'rb') as w:
        assert w.getsampwidth() == 2 and w.getcomptype() == 'NONE'
        sr, channels = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').reshape(-1, channels)
    return x, sr


def write_pcm(path, x, sr):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(x, dtype='<i2').tobytes())


def write_double(path, x, sr):
    data = np.asarray(x, dtype='<f8').tobytes()
    fmt = struct.pack('<HHIIHH', 3, 1, sr, sr * 8, 8, 64)
    body = b'WAVEfmt ' + struct.pack('<I', 16) + fmt
    body += b'fact' + struct.pack('<II', 4, len(x))
    body += b'data' + struct.pack('<I', len(data)) + data
    Path(path).write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)


def shared_link(target, link, directory=False):
    target, link = Path(target).resolve(), Path(link).absolute()
    if os.name == 'nt' and directory:
        quote = lambda x: "'" + str(x).replace("'", "''") + "'"
        subprocess.run(['powershell', '-NoProfile', '-Command',
                        f'New-Item -ItemType Junction -Path {quote(link)} -Target {quote(target)} | Out-Null'],
                       check=True, capture_output=True)
    elif directory:
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
    else:
        os.link(target, link)


def validate_split_ranges(splits, nframes, guard):
    previous = -1
    for s in splits:
        lo, hi = s['source_start_frame'], s['source_end_frame_exclusive']
        if not (0 <= lo < hi <= nframes and hi - lo > 2 * guard):
            raise ValueError('invalid split support')
        if lo < previous:
            raise ValueError('overlapping audio split support')
        previous = hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding='utf-8'))
    if (a.output / 'manifest.json').exists() or (a.output / 'source').exists():
        raise FileExistsError('Refusing to replace a prepared dataset; use a fresh output')
    a.output.mkdir(parents=True, exist_ok=True)
    assert sha(a.archive) == protocol['source_zip_sha256']
    assert shutil.disk_usage(a.output).free > 3 * 1024**3, 'Need 3 GiB free before extraction'
    source_root = a.output / 'source'
    source_root.mkdir()
    archive_entries = []
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None, 'archive CRC failure'
        for item in z.infolist():
            if item.is_dir():
                continue
            target = (source_root / item.filename).resolve()
            if not target.is_relative_to(source_root.resolve()):
                raise ValueError('unsafe archive member')
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            archive_entries.append({'path': 'source/' + item.filename,
                                    'bytes': item.file_size, 'sha256': sha(target)})
    source = source_root / 'May'
    original = [json.loads((source / f'transforms_{name}.json').read_text()) for name in ['train', 'val']]
    frames = sorted(original[0]['frames'] + original[1]['frames'], key=lambda f: f['img_id'])
    assert [f['img_id'] for f in frames] == list(range(6073))
    assert all(f['aud_id'] == f['img_id'] for f in frames)
    guard = protocol['context_guard_frames_each_side']
    validate_split_ranges(protocol['splits'], len(frames), guard)
    pcm, sr = read_pcm16(source / 'aud.wav')
    assert sr == protocol['sample_rate'] and pcm.shape[1] == 2
    assert np.array_equal(pcm[:, 0], pcm[:, 1]), 'cannot silently average unequal source channels'
    pcm = pcm[:, 0]
    wave64 = pcm.astype(np.float64) / 32768.0
    clock = protocol['samples_per_frame']
    assert len(pcm) >= len(frames) * clock
    manifest = {'schema': protocol['schema'], 'protocol_sha256': sha(a.protocol),
                'source_archive_sha256': sha(a.archive), 'source_archive_path': str(a.archive.resolve()),
                'source_pcm_sha256': hashlib.sha256(pcm.tobytes()).hexdigest(),
                'source_audio_samples': len(pcm), 'source_transform_frames': len(frames),
                'qualification': 'ENGINEERING_PILOT_ONLY', 'training_runs': 0,
                'official_generator_used': False, 'source_entries': archive_entries,
                'splits': [], 'arms': [], 'eval_clips': [],
                'rejected_source_candidates': [
                    {'source': 'AISHELL', 'reason': 'audio only: no real lip video supervision'},
                    {'source': 'Ubuntu data/hdtf_videos_test', 'reason': 'manifest present but 3 referenced videos missing'},
                    {'source': 'May for Mandarin confirmation', 'reason': 'English, short, previously exposed, official full-sequence preprocessing'}]}
    for split in protocol['splits']:
        name = split['name']
        lo, hi = split['source_start_frame'], split['source_end_frame_exclusive']
        selected = frames[lo + guard:hi - guard]
        transform = {k: v for k, v in original[0].items() if k != 'frames'}
        transform.update(h=512, w=512)
        transform['frames'] = [dict(f, aud_id=f['img_id'] - lo) for f in selected]
        ids = [f['img_id'] for f in selected]
        split_dir = a.output / 'splits' / name
        split_dir.mkdir(parents=True)
        unmodified = pcm[lo * clock:hi * clock]
        baseline = wave64[lo * clock:hi * clock]
        write_pcm(split_dir / 'original_score_audio.wav', unmodified, sr)
        save_json(split_dir / f'transforms_{name}.json', transform)
        info = dict(split, supervision_start_frame=ids[0], supervision_end_frame_exclusive=ids[-1] + 1,
                    supervision_frames=len(ids), audio_samples=len(baseline),
                    source_start_sample=lo * clock, source_end_sample_exclusive=hi * clock,
                    audio_seconds=len(baseline) / sr, supervision_seconds=len(ids) / protocol['fps'],
                    transform_path=f'splits/{name}/transforms_{name}.json',
                    score_audio_path=f'splits/{name}/original_score_audio.wav',
                    score_audio_sha256=sha(split_dir / 'original_score_audio.wav'),
                    score_pcm_sha256=hashlib.sha256(unmodified.tobytes()).hexdigest(),
                    attention_edge_source_ids=ids[:4] + ids[-4:])
        manifest['splits'].append(info)
        for db in protocol['gains_db']:
            label = 'gain_m6' if db == -6 else 'gain_p6' if db == 6 else 'gain_0'
            arm_dir = a.output / 'arms' / name / label
            arm_dir.mkdir(parents=True)
            intended = baseline * 10**(db / 20)
            if not np.isfinite(intended).all() or np.abs(intended).max() >= 1:
                raise ValueError('unsafe driver: no limiter or selective clipping permitted')
            write_double(arm_dir / 'aud.wav', intended, sr)
            shared_link(split_dir / f'transforms_{name}.json', arm_dir / f'transforms_{name}.json')
            for directory in ['gt_imgs', 'ori_imgs', 'torso_imgs']:
                shared_link(source / directory, arm_dir / directory, directory=True)
            for filename in ['bc.jpg', 'bs.npy']:
                shared_link(source / filename, arm_dir / filename)
            save_json(arm_dir / 'binding.json', {
                'split': name, 'gain_db': db, 'shared_source_relative': '../../../source/May',
                'source_start_frame': lo, 'source_end_frame_exclusive': hi,
                'supervised_source_ids': ids, 'aud_id_rule': 'source_img_id minus source_start_frame',
                'official_test_val_fallback_permitted': False,
                'official_val_100_frame_limit_permitted': False,
                'cached_official_audio_features_permitted': False})
            manifest['arms'].append({'split': name, 'id': label, 'gain_db': db,
                                     'path': f'arms/{name}/{label}', 'samples': len(intended),
                                     'peak': float(np.abs(intended).max()),
                                     'rms': float(np.sqrt(np.mean(intended**2))),
                                     'audio_sha256': sha(arm_dir / 'aud.wav')})
        # Explicitly omit provider attention boundary rows from primary clips.
        core_lo, core_hi = ids[4], ids[-4]
        step = protocol['eval_clip_frames']
        for ix, start in enumerate(range(core_lo, core_hi - step + 1, step)):
            if name == 'train':
                continue
            clip_id = f'{name}_{ix:02d}'
            path = split_dir / f'{clip_id}_original_pcm.wav'
            write_pcm(path, pcm[start * clock:(start + step) * clock], sr)
            manifest['eval_clips'].append({'id': clip_id, 'split': name,
                                         'source_start_frame': start, 'frames': step,
                                         'split_audio_start_sample': (start - lo) * clock,
                                         'source_start_sample': start * clock,
                                         'score_audio_path': str(path.relative_to(a.output)).replace('\\', '/'),
                                         'score_audio_sha256': sha(path)})
    shutil.copyfile(a.protocol, a.output / 'protocol.json')
    save_json(a.output / 'manifest.json', manifest)
    print(json.dumps({'output': str(a.output.resolve()), 'source_files': len(archive_entries),
                      'splits': [(s['name'], s['supervision_frames']) for s in manifest['splits']],
                      'arms': len(manifest['arms']), 'eval_clips': len(manifest['eval_clips']),
                      'manifest_sha256': sha(a.output / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
