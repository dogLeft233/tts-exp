"""Independent readback of sealed source, exact clocks, gain WAVs and AVE arrays."""
import argparse
import hashlib
import json
import struct
import wave
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SOURCE_SHA = '008fe3fb46430a92beff334e26f37eab229c717ff5b577dfbba90f46c3848d4b'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def pcm16(path):
    with wave.open(str(path), 'rb') as f:
        assert f.getsampwidth() == 2 and f.getframerate() == 16000
        return np.frombuffer(f.readframes(f.getnframes()), '<i2').reshape(-1, f.getnchannels())


def float_wav(path):
    data = Path(path).read_bytes()
    assert data[:4] == b'RIFF' and data[8:12] == b'WAVE'
    assert struct.unpack_from('<I', data, 4)[0] == len(data) - 8
    pos, fmt, samples = 12, None, None
    while pos < len(data):
        key = data[pos:pos + 4]
        n = struct.unpack_from('<I', data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + n]
        if key == b'fmt ':
            fmt = struct.unpack('<HHIIHH', body)
        elif key == b'data':
            samples = np.frombuffer(body, '<f8')
        pos += 8 + n + n % 2
    assert fmt == (3, 1, 16000, 128000, 8, 64) and samples is not None
    try:
        import soundfile as sf
    except ImportError:
        pass
    else:
        decoded, sr = sf.read(path, dtype='float64')
        assert sr == 16000 and np.array_equal(decoded, samples)
    return samples


def verify_gain(actual, original, db):
    if actual.shape != original.shape:
        raise ValueError('driver changed duration')
    if not np.array_equal(actual, original.astype(np.float64) / 32768 * 10**(db / 20)):
        raise ValueError('driver is not exact pure gain of source-clock PCM')
    if not np.isfinite(actual).all() or np.abs(actual).max() >= 1:
        raise ValueError('unsafe or non-finite driver')


def verify_clock(frames, source_frames, split):
    lo = split['source_start_frame']
    for frame in frames:
        source = source_frames[frame['img_id']]
        if frame['aud_id'] != frame['img_id'] - lo:
            raise ValueError('wrong source audio clock')
        if frame['transform_matrix'] != source['transform_matrix']:
            raise ValueError('pose changed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--features', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'use a new result file'
    root = a.dataset
    m = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    assert file_sha(a.archive) == SOURCE_SHA == m['source_archive_sha256']
    assert file_sha(root / 'protocol.json') == m['protocol_sha256']
    report = {'schema': 'synctalk_independent_data_qa_v1', 'training_runs': 0,
              'source_archive_sha256': SOURCE_SHA, 'source_files_verified': 0,
              'source_bytes_verified': 0, 'image_files_decoded': 0, 'landmark_files_verified': 0,
              'arms_verified': 0, 'features_verified': 0, 'splits': [], 'clips': [],
              'qualification': 'ENGINEERING_PILOT_ONLY', 'visual_previews': []}
    with zipfile.ZipFile(a.archive) as z:
        expected = {i.filename: i for i in z.infolist() if not i.is_dir()}
        assert len(expected) == len(m['source_entries'])
        source_metadata = [json.loads(z.read(f'May/transforms_{s}.json')) for s in ['train', 'val']]
        source_frames = {f['img_id']: f for d in source_metadata for f in d['frames']}
        assert sorted(source_frames) == list(range(6073))
        for item in m['source_entries']:
            name = item['path'].removeprefix('source/')
            actual = root / item['path']
            raw = z.read(name)
            assert len(raw) == item['bytes'] == expected[name].file_size
            assert digest(raw) == file_sha(actual) == item['sha256'], name
            report['source_files_verified'] += 1
            report['source_bytes_verified'] += len(raw)
    source_pcm = pcm16(root / 'source/May/aud.wav')
    assert source_pcm.shape[1] == 2 and np.array_equal(source_pcm[:, 0], source_pcm[:, 1])
    source_pcm = source_pcm[:, 0]
    assert digest(source_pcm.tobytes()) == m['source_pcm_sha256']
    bs = np.load(root / 'source/May/bs.npy', allow_pickle=False)
    assert len(bs) >= 6073 and np.isfinite(bs).all()
    supports, supervised_sets = [], []
    preview_dir = root / 'verification_previews'
    preview_dir.mkdir(exist_ok=True)
    for s in m['splits']:
        name, lo, hi = s['name'], s['source_start_frame'], s['source_end_frame_exclusive']
        supports.append(set(range(lo, hi)))
        t = json.loads((root / s['transform_path']).read_text())
        frames = t['frames']
        ids = [f['img_id'] for f in frames]
        assert ids == list(range(lo + 25, hi - 25))
        assert len(ids) == s['supervision_frames']
        supervised_sets.append(set(ids))
        verify_clock(frames, source_frames, s)
        score = pcm16(root / s['score_audio_path'])[:, 0]
        wanted = source_pcm[lo * 640:hi * 640]
        assert np.array_equal(score, wanted) and digest(score.tobytes()) == s['score_pcm_sha256']
        assert file_sha(root / s['score_audio_path']) == s['score_audio_sha256']
        mouth_values = []
        for frame in frames:
            i = frame['img_id']
            matrix = np.array(frame['transform_matrix'])
            assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
            assert np.max(np.abs(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3))) < 0.001
            assert np.array_equal(matrix[3], [0, 0, 0, 1])
            lms = np.loadtxt(root / f'source/May/ori_imgs/{i}.lms')
            assert lms.shape == (68, 2) and np.isfinite(lms).all()
            assert np.all((lms[48:68] >= 0) & (lms[48:68] < 512)), i
            mouth_values.append(float(np.linalg.norm(lms[62] - lms[66]) /
                                      max(np.linalg.norm(lms[48] - lms[54]), 1e-8)))
            report['landmark_files_verified'] += 1
            for directory, suffix, mode in [('gt_imgs', '.jpg', 'RGB'), ('ori_imgs', '.jpg', 'RGB'),
                                             ('torso_imgs', '.png', 'RGBA')]:
                path = root / f'source/May/{directory}/{i}{suffix}'
                with Image.open(path) as image:
                    image.load()
                    assert image.size == (512, 512) and image.mode == mode, (path, image.size, image.mode)
                report['image_files_decoded'] += 1
        canvas = Image.new('RGB', (4 * 256, 3 * 282), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, idx in enumerate(np.linspace(0, len(ids) - 1, 12, dtype=int)):
            i = ids[idx]
            with Image.open(root / f'source/May/gt_imgs/{i}.jpg') as im:
                canvas.paste(im.resize((256, 256)), ((j % 4) * 256, (j // 4) * 282))
            draw.text(((j % 4) * 256 + 4, (j // 4) * 282 + 260), f'{name}: source {i}', fill='black')
        preview = preview_dir / f'{name}_contact.png'
        canvas.save(preview)
        report['visual_previews'].append(str(preview.relative_to(root)))
        report['splits'].append({'name': name, 'support_frames': hi - lo, 'supervised_frames': len(ids),
                                 'support_seconds': (hi - lo) / 25, 'mouth_aperture_min': min(mouth_values),
                                 'mouth_aperture_max': max(mouth_values), 'mouth_aperture_std': float(np.std(mouth_values)),
                                 'score_pcm_exact': True})
    for i in range(3):
        for j in range(i):
            assert not supports[i] & supports[j] and not supervised_sets[i] & supervised_sets[j]
    for arm in m['arms']:
        s = next(s for s in m['splits'] if s['name'] == arm['split'])
        data = root / arm['path']
        x = float_wav(data / 'aud.wav')
        expected = source_pcm[s['source_start_sample']:s['source_end_sample_exclusive']]
        verify_gain(x, expected, arm['gain_db'])
        assert file_sha(data / 'aud.wav') == arm['audio_sha256']
        assert file_sha(data / f"transforms_{s['name']}.json") == file_sha(root / s['transform_path'])
        if s['name'] == 'test':
            assert not (data / 'transforms_val.json').exists()
        for d in ['gt_imgs', 'ori_imgs', 'torso_imgs']:
            assert (data / d).resolve() == (root / 'source/May' / d).resolve()
        for d in ['bc.jpg', 'bs.npy']:
            assert file_sha(data / d) == file_sha(root / 'source/May' / d)
        report['arms_verified'] += 1
    for clip in m['eval_clips']:
        start, n = clip['source_start_frame'], clip['frames']
        s = next(s for s in m['splits'] if s['name'] == clip['split'])
        assert start >= s['supervision_start_frame'] + 4
        assert start + n <= s['supervision_end_frame_exclusive'] - 4
        x = pcm16(root / clip['score_audio_path'])[:, 0]
        assert np.array_equal(x, source_pcm[start * 640:(start + n) * 640])
        assert file_sha(root / clip['score_audio_path']) == clip['score_audio_sha256']
        report['clips'].append({'id': clip['id'], 'frames': n, 'pcm_exact': True})
    if a.features:
        q = json.loads((a.features / 'provider_qualification.json').read_text())
        assert q['generator_training_runs'] == q['generator_forwards'] == 0
        assert q['adapter_fixes'] == {'test_reads_test_json': 1, 'val_not_truncated_100': 1}
        assert len(q['arms']) == 9 and len(q['repeats']) == 3
        for row in q['arms']:
            folder = a.features / row['split'] / row['id']
            padded = np.load(folder / 'ave_padded.npy', allow_pickle=False)
            aligned = np.load(folder / 'ave_aligned.npy', allow_pickle=False)
            s = next(s for s in m['splits'] if s['name'] == row['split'])
            indices = np.arange(25, s['source_end_frame_exclusive'] - s['source_start_frame'] - 25)
            assert padded.dtype == aligned.dtype == np.float32
            assert padded.ndim == 2 and padded.shape[1] == 512 and np.isfinite(padded).all()
            assert np.array_equal(aligned, padded[indices, None, :])
            assert file_sha(folder / 'ave_padded.npy') == row['padded_sha256']
            assert file_sha(folder / 'ave_aligned.npy') == row['aligned_sha256']
            if row['gain_db'] == 0:
                assert np.array_equal(np.load(folder / 'ave_padded_repeat.npy', allow_pickle=False), padded)
            report['features_verified'] += 1
        report['provider_live_verified'] = True
    report['passed'] = True
    report['ready_for_engineering_pilot'] = bool(a.features)
    report['formal_mandarin_cohort_ready'] = False
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['splits', 'visual_previews', 'clips']}), flush=True)


if __name__ == '__main__':
    main()
