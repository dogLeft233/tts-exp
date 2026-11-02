"""Decompose the unchanged STOI calculation, without masking its result."""
import importlib
import numpy as np
import soundfile as sf
from calibrate import ROOT, read, write

M = importlib.import_module('pystoi.stoi')
OLD = ROOT / 'runs/phoneme_direction_20261026_repair01'
RUN = ROOT / 'runs/phoneme_direction_20261026_repair02'


def parts(a, b, sr):
    x = M.utils.resample_oct(a, M.FS, sr)
    y = M.utils.resample_oct(b, M.FS, sr)
    x, y = M.utils.remove_silent_frames(x, y, M.DYN_RANGE, M.N_FRAME, M.N_FRAME // 2)
    xs = M.utils.stft(x, M.N_FRAME, M.NFFT, overlap=2).T
    ys = M.utils.stft(y, M.N_FRAME, M.NFFT, overlap=2).T
    xt = np.sqrt(M.OBM @ abs(xs) ** 2); yt = np.sqrt(M.OBM @ abs(ys) ** 2)
    xx = np.array([xt[:, j - M.N:j] for j in range(M.N, xt.shape[1] + 1)])
    yy = np.array([yt[:, j - M.N:j] for j in range(M.N, yt.shape[1] + 1)])
    energy = np.sum(xx ** 2, axis=(1, 2))
    weak = 10 * np.log10(np.maximum(energy / energy.max(), 1e-12)) < -20
    yn = yy * np.linalg.norm(xx, axis=2, keepdims=True) / (np.linalg.norm(yy, axis=2, keepdims=True) + M.utils.EPS)
    yp = np.minimum(yn, xx * (1 + 10 ** (-M.BETA / 20)))
    yp -= yp.mean(axis=2, keepdims=True); xx -= xx.mean(axis=2, keepdims=True)
    yp /= np.linalg.norm(yp, axis=2, keepdims=True) + M.utils.EPS
    xx /= np.linalg.norm(xx, axis=2, keepdims=True) + M.utils.EPS
    correlations = np.sum(yp * xx, axis=2).mean(axis=1)
    exact = float(M.stoi(a, b, sr))
    assert abs(correlations.mean() - exact) < 1e-12
    return {'official_stoi': exact, 'windows': len(weak), 'weak_window_fraction': float(weak.mean()),
            'weak_window_mean_correlation': float(correlations[weak].mean()) if weak.any() else None,
            'other_window_mean_correlation': float(correlations[~weak].mean()),
            'weak_fraction_of_total_one_minus_correlation': float(np.sum(1 - correlations[weak]) / np.sum(1 - correlations)),
            'definition': 'diagnostic only: original third-octave segment energy more than 20dB below maximal segment; not speech labels'}


def main():
    failed = {x['id'] for x in read(OLD / 'regular_full_fit_quality.json')['failed']}
    rows = []
    for shard in range(3):
        for x in read(OLD / f'regular_fit_shard{shard}_complete.json')['rows']:
            if x['id'] not in failed: continue
            a, sr = sf.read(x['source']); b, _ = sf.read(x['output'])
            rows.append({'id': x['id'], 'baseline': parts(a, b, sr)})
    write(RUN / 'stoi_loss_decomposition.json', {'rows': rows, 'gate_changed': False, 'samples_removed': 0})
    print(rows, flush=True)


if __name__ == '__main__': main()
