"""Candidate-only L6 spectral head, speaker-held-out; no waveform bypass."""
import sys
import numpy as np
import soundfile as sf
from scipy.linalg import solve
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft, istft
from calibrate import ROOT, read, write, sha
from separate_encoder_decoder import RUN, OLD, bands


def fit_head(features, targets):
    h = np.concatenate(features).astype('float64'); y = np.concatenate(targets)
    hm = h.mean(0); hs = np.maximum(h.std(0), 1e-6); ym = y.mean(0)
    h = (h - hm) / hs
    gram = h.T @ h; gram.flat[::len(gram) + 1] += 100.
    w = solve(gram, h.T @ (y - ym), assume_a='pos')
    return dict(hm=hm, hs=hs, ym=ym, w=w)


def predict_head(h, model):
    return (h - model['hm']) / model['hs'] @ model['w'] + model['ym']


def postfilter(decoded, predicted, peak_gain):
    """Reads only decoder audio, predicted L6 bands, and the existing scalar gain.

    Correction fades out between -20 and -10dB of decoded frame band energy.
    This avoids using an original-audio silence mask to bypass reconstruction.
    """
    actual = bands(decoded * peak_gain)
    if predicted.shape != actual.shape: raise ValueError('Frame clock mismatch')
    power = (10 ** (actual / 10)).sum(1)
    relative = 10 * np.log10(np.maximum(power / max(power.max(), 1e-12), 1e-12))
    weight = np.clip((-relative - 10) / 10, 0, 1)
    delta = .5 * np.clip(predicted - actual, -6, 6) * weight[:, None]
    delta = uniform_filter1d(delta, 5, axis=0, mode='nearest')
    frequency, times, z = stft(decoded, 16000, nperseg=1024, noverlap=704,
                             boundary='zeros', padded=True)
    centers = 150 * 2 ** (np.arange(15) / 3)
    grid = (np.arange(len(delta)) * 320 + 199.5) / 16000
    temporal = np.stack([np.interp(times, grid, delta[:, j]) for j in range(15)], 1)
    gains = np.stack([np.interp(frequency, centers, d, left=0, right=0) for d in temporal], 1)
    _, restored = istft(z * 10 ** (gains / 20), 16000, nperseg=1024, noverlap=704,
                       boundary=True)
    return restored[:len(decoded)]


def main():
    rows = read(RUN / 'separation_features.json')['rows']
    speakers = sorted({x['speaker_id'] for x in rows})
    data = [dict(x, arrays=dict(np.load(x['path'])), fold=speakers.index(x['speaker_id']) % 5) for x in rows]
    prior = {x['id']: x for x in read(OLD / 'full_peak95_quality.json')['rows']}
    write(RUN / 'postfilter_probe_contract.json', {
        'rows': [x['id'] for x in rows], 'ridge_alpha': 100.,
        'target': '15 original log bandpowers', 'input': 'L6 only',
        'correction_db': '0.5 * clip(predicted-decoded,-6,6) * decoded-only weak-frame weight',
        'temporal_smoothing_frames': 5, 'stft_nfft': 1024, 'stft_hop': 320,
        'speaker_held_out': True, 'script_sha256': sha(__file__),
        'no_original_waveform_bypass': True, 'candidate_only': True,
        'all_gates_unchanged': True, 'development_allowed': False, 'confirmation_allowed': False})
    outputs = []
    for fold in range(5):
        train = [x for x in data if x['fold'] != fold]; test = [x for x in data if x['fold'] == fold]
        assert not {x['speaker_id'] for x in train} & {x['speaker_id'] for x in test}
        model = fit_head([x['arrays']['l6'] for x in train], [x['arrays']['target'] for x in train])
        p = RUN / f'postfilter_head_fold{fold}.npz'
        if p.exists(): raise FileExistsError(p)
        np.savez_compressed(p, **model)
        for x in test:
            old = prior[x['id']]; assert sha(old['output']) == old['output_sha256']
            a, sr = sf.read(old['source']); b, sb = sf.read(old['output']); assert sr == sb == 16000
            gain = .95 / max(float(abs(a).max()), 1e-8)
            c = postfilter(b, predict_head(x['arrays']['l6'], model), gain)
            c *= np.sqrt(np.mean(a * a) / np.mean(c * c))
            dest = RUN / 'postfilter_probe_audio' / (x['id'] + '.wav'); dest.parent.mkdir(exist_ok=True)
            if dest.exists(): raise FileExistsError(dest)
            sf.write(dest, c, sr, subtype='FLOAT')
            outputs.append(dict(id=x['id'], speaker_id=x['speaker_id'], source=old['source'],
                source_sha256=old['source_sha256'], output=str(dest), output_sha256=sha(dest),
                head=str(p), head_sha256=sha(p), fold=fold))
        print('POSTFILTER_FOLD', fold, flush=True)
    write(RUN / 'postfilter_probe_outputs.json', {'rows': outputs})


if __name__ == '__main__': main()
