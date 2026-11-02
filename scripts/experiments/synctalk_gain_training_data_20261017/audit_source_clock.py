"""Check extracted AV assets against the untouched source MP4 clock."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    source = a.dataset / 'source/May'
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                              '-show_format', '-of', 'json', str(source / 'May.mp4')]))
    v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    assert v['avg_frame_rate'] == '25/1' and int(v['nb_frames']) == 6073
    assert (v['width'], v['height']) == (512, 512) and float(v['start_time']) == 0
    manifest = json.loads((a.dataset / 'manifest.json').read_text())
    cap = cv2.VideoCapture(str(source / 'May.mp4'))
    images = []
    for split in manifest['splits']:
        lo, hi = split['supervision_start_frame'], split['supervision_end_frame_exclusive']
        for idx in [lo, (lo + hi) // 2, hi - 1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            assert ok
            errors = {}
            for lag in range(-2, 3):
                ref = cv2.imread(str(source / f'ori_imgs/{idx + lag}.jpg'))
                errors[lag] = float(np.mean(np.abs(frame.astype(np.float64) - ref.astype(np.float64))))
            best = min(errors, key=errors.get)
            assert best == 0, (idx, errors)
            images.append({'split': split['name'], 'source_frame': idx,
                           'best_extracted_frame_lag': best, 'mae_by_lag': errors})
    cap.release()
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(source / 'May.mp4'),
                                   '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', '-'])
    decoded = np.frombuffer(raw, '<i2').astype(np.float64) / 32768
    wav, sr = sf.read(source / 'aud.wav', dtype='float64', always_2d=True)
    assert sr == 16000 and np.array_equal(wav[:, 0], wav[:, 1])
    actual = wav[:, 0]
    length = min(len(decoded), len(actual))
    audio = []
    for split in manifest['splits']:
        lo = split['source_start_sample'] + 16000
        hi = min(lo + 12 * 16000, length)
        x, y = actual[lo:hi], decoded[lo:hi]
        x, y = x - x.mean(), y - y.mean()
        corr = correlate(x, y, mode='full', method='fft')
        lags = correlation_lags(len(x), len(y), mode='full')
        support = np.abs(lags) <= 1600
        best = int(lags[support][np.argmax(corr[support])])
        c0 = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        assert best == 0 and c0 > 0.99, (split['name'], best, c0)
        audio.append({'split': split['name'], 'start_sample': lo, 'samples': len(x),
                      'best_sample_lag_in_100ms': best, 'zero_lag_correlation': c0})
    report = {'passed': True, 'source_video_fps': 25, 'source_video_frames': 6073,
              'source_video_start_time': 0, 'source_audio_decode_samples': len(decoded),
              'extracted_wav_samples': len(actual), 'image_clock_samples': images,
              'audio_clock_samples': audio, 'new_audio_shifts': 0,
              'scope': 'Asset extraction clock identity; does not prove human-perceived source AV synchronization',
              'training_runs': 0}
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'image_samples': len(images), 'audio_samples': audio}), flush=True)


if __name__ == '__main__':
    main()
