"""Independent CPU verification of shifted controls and PEAVS cached features."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'runs/av_metric_preference_20261026_attempt02'


def read(path):
    return json.loads(Path(path).read_text())


def capture(args):
    return subprocess.check_output(list(map(str, args)))


def audio(path):
    return np.frombuffer(capture(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:a:0',
        '-ac', '1', '-ar', '16000', '-f', 's16le', '-']), dtype='<i2')


def frames(path):
    text = capture(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v:0', '-an', '-f', 'framemd5', '-']).decode()
    return [line.split(',')[-1].strip() for line in text.splitlines() if not line.startswith('#')]


def metadata(path):
    return json.loads(capture(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path]))


def feature_index():
    result = {}
    for raw in sorted((OLD / 'peavs').glob('batch*.raw.json')):
        data = read(raw)
        work = Path(data['work_dir']) / 'av_features_extraction/output_feats'
        for row in data['scores']:
            name = Path(row['filename']).stem
            result[name] = dict(audio=work / 'vggish' / (name + '_vggish.npy'),
                                video=work / 'i3d_feats' / (name + '_i3d.npy'))
    return result


def main(out):
    cells = read(OLD / 'cells.json')
    natives = {c['video_id']: c for c in cells if c['kind'] == 'native'}
    features = feature_index()
    cache = {}
    results = []
    for c in cells:
        if c['kind'] != 'shift':
            continue
        n = natives[c['video_id']]
        a = OLD / 'inputs' / (n['id'] + '.mp4')
        b = OLD / 'inputs' / (c['id'] + '.mp4')
        if n['id'] not in cache:
            cache[n['id']] = dict(audio=audio(a), frames=frames(a), metadata=metadata(a))
        original = cache[n['id']]['audio']
        actual = audio(b)
        shift = c['shift_samples']
        expected = np.zeros_like(original)
        if shift > 0:
            expected[shift:] = original[:-shift]
        else:
            expected[:shift] = original[-shift:]
        assert np.array_equal(actual, expected), (c['id'], 'WRONG_WAVEFORM_SHIFT')
        assert frames(b) == cache[n['id']]['frames'], (c['id'], 'VIDEO_CHANGED')
        probes = [cache[n['id']]['metadata'], metadata(b)]
        streams = [{s['codec_type']: s for s in p['streams']} for p in probes]
        for kind in ['audio', 'video']:
            for key in ['start_time', 'time_base', 'sample_rate', 'r_frame_rate']:
                assert streams[0][kind].get(key) == streams[1][kind].get(key), (c['id'], kind, key)
        av = np.load(features[n['id']]['video'])
        bv = np.load(features[c['id']]['video'])
        assert np.array_equal(av, bv), (c['id'], 'PEAVS_VIDEO_FEATURES_CHANGED')
        aa = np.load(features[n['id']]['audio'])
        ba = np.load(features[c['id']]['audio'])
        assert not np.array_equal(aa, ba), (c['id'], 'PEAVS_AUDIO_SHIFT_DISAPPEARED')
        results.append(dict(id=c['id'], shift_samples=shift, samples=len(actual),
            waveform_exact=True, video_frames_exact=True, stream_start_and_rates_equal=True,
            peavs_video_features_exact=True, peavs_audio_features_changed=True,
            peavs_video_shape=list(av.shape), peavs_audio_shape=list(aa.shape)))
    assert len(results) == 40
    report = dict(status='PASS', controls=40, rows=results)
    (out / 'media_and_cached_features_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status='PASS', controls=len(results), checks='waveform,zero-padding,frames,PTS origin,rates,PEAVS video invariance/audio change')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
