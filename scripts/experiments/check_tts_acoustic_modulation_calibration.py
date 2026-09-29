"""Independent CPU validation using scipy STFT/ISTFT, no producer import."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

OUT = Path(__file__).resolve().parents[2] / 'runs/tts_acoustic_modulation_calibration_20260926'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_spectrum(x):
    padded = np.pad(x, (256, 256 + (-len(x)) % 128), mode='reflect')
    _, _, z = signal.stft(padded, fs=16000, window='hann', nperseg=512,
                          noverlap=384, nfft=512, boundary=None, padded=False)
    return z * 256, padded


def independent_residual(z):
    amplitude = abs(z)
    if amplitude.max() == 0:
        return np.zeros_like(amplitude)
    log = 20*np.log10(np.maximum(amplitude, amplitude.max()*1e-4))
    # Matrix centering performed sequentially instead of producer's mu/f/e expression.
    r = log - np.mean(log, axis=0, keepdims=True)
    r -= np.mean(r, axis=1, keepdims=True)
    return r


def check(out):
    p = json.loads((out/'protocol.json').read_text())
    metrics = json.loads((out/'metrics.json').read_text())
    manifest = json.loads((out/'waveform_manifest.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip() == summary['protocol_sha256']
    lookup = {(r['id'], r['arm'], r['condition']):r for r in metrics}
    records = {(r['id'], r['arm']):r for r in p['sources']}
    assert len(manifest) == len(metrics) == 208
    assert {r['id'] for r in manifest} == set(p['cal_ids'])
    reconstructed, residuals = {}, {}
    errors = {'waveform_float32_vs_independent64':0.0, 'target_residual_rms_db':0.0, 'envelope_rms':0.0,
              'spectrum_amplitude':0.0, 'rms_relative':0.0, 'identity_float32_max_abs':0.0}
    source_audio = {}
    for key, rec in records.items():
        assert sha(rec['path']) == rec['sha256']
        x, sr = sf.read(rec['path'], dtype='float64'); assert sr == 16000 and x.ndim == 1
        source_audio[key] = x
        z, _ = independent_spectrum(x)
        r = independent_residual(z)
        for condition, alpha in p['conditions'].items():
            if alpha is None:
                y = x.copy()
            else:
                changed = z * np.power(10, np.minimum(6, np.maximum(-6, (alpha-1)*r))/20)
                # SciPy overlap-add implementation and normalization independently reconstruct.
                _, padded_y = signal.istft(changed/256, fs=16000, window='hann', nperseg=512,
                                           noverlap=384, nfft=512, boundary=False)
                y = padded_y[256:256+len(x)]
                yrms = np.sqrt(np.mean(y*y)); xrms = np.sqrt(np.mean(x*x))
                if yrms:
                    y *= xrms / yrms
            reconstructed[key+(condition,)] = y
    for rec in manifest:
        key = rec['id'], rec['arm'], rec['condition']
        m = lookup[key]
        source = source_audio[key[:2]]
        g = summary['headroom_gains'][key[0]]
        for stage in ['pre_headroom', 'waveforms', 'acoustics']:
            assert sha(rec[stage]['path']) == rec[stage]['sha256']
        before, sr = sf.read(rec['pre_headroom']['path'], dtype='float64')
        y, sr = sf.read(rec['waveforms']['path'], dtype='float64')
        assert sr == 16000 and len(y) == len(source) and len(before) == len(source)
        assert np.isfinite(y).all() and max(abs(y)) < 1
        reference = reconstructed[key]
        err = max(np.max(abs(before-reference)), np.max(abs(y-reference*g)))
        errors['waveform_float32_vs_independent64'] = max(errors['waveform_float32_vs_independent64'], float(err))
        assert np.allclose(before, reference, rtol=6e-8, atol=1e-12)
        assert np.allclose(y, reference*g, rtol=6e-8, atol=1e-12)
        z, padded = independent_spectrum(y)
        r = independent_residual(z); target = float(np.sqrt(np.mean(r*r)))
        residuals[key] = target
        errors['target_residual_rms_db'] = max(errors['target_residual_rms_db'], abs(target-m['residual_rms_db']))
        expected_rms = g*np.sqrt(np.mean(source*source))
        rel = abs(np.sqrt(np.mean(y*y))-expected_rms)/expected_rms if expected_rms else 0
        errors['rms_relative'] = max(errors['rms_relative'], float(rel))
        arr = np.load(rec['acoustics']['path'])
        env = np.sqrt(np.maximum(np.convolve(padded*padded, np.ones(512)/512, mode='valid')[::128], 0))
        errors['envelope_rms'] = max(errors['envelope_rms'], float(np.max(abs(env-arr['envelope_rms']))))
        errors['spectrum_amplitude'] = max(errors['spectrum_amplitude'], float(np.max(abs(abs(z).mean(1)-arr['spectrum_amplitude']))))
        if rec['condition'] == 'identity':
            errors['identity_float32_max_abs'] = max(errors['identity_float32_max_abs'], float(np.max(abs(y-source*g))))
    for sid in p['cal_ids']:
        peak = max(float(max(abs(reconstructed[sid, arm, condition]))) for arm in ['N','T'] for condition in p['conditions'])
        expected = min(1, .98/peak) if peak else 1
        assert abs(expected-summary['headroom_gains'][sid]) < 1e-12
    active = [(sid,arm) for sid in p['cal_ids'] for arm in ['N','T'] if np.sqrt(np.mean(source_audio[sid,arm]**2)) > 1e-8]
    ordered = np.mean([residuals[sid,arm,'alpha08'] < residuals[sid,arm,'identity'] < residuals[sid,arm,'alpha12'] for sid,arm in active])
    decrease = np.median([1-residuals[sid,arm,'alpha08']/residuals[sid,arm,'identity'] for sid,arm in active])
    increase = np.median([residuals[sid,arm,'alpha12']/residuals[sid,arm,'identity']-1 for sid,arm in active])
    assert abs(ordered-summary['ordered_fraction']) < 1e-12
    assert abs(decrease-summary['median_decrease_relative']) < 1e-12
    assert abs(increase-summary['median_increase_relative']) < 1e-12
    assert errors['target_residual_rms_db'] < 1e-10 and errors['rms_relative'] <= 1e-5
    assert errors['identity_float32_max_abs'] <= 1e-7
    result = {'status':'independently_verified', 'implementation':'scipy.signal.stft/istft + sequential centering + convolution RMS',
              'checked_waveforms':208,'checked_sources':52,'maximum_errors':errors,
              'ordered_fraction':float(ordered),'median_decrease_relative':float(decrease),'median_increase_relative':float(increase),
              'checker_sha256':sha(__file__), 'producer_snapshot_sha256':sha(out/'code_snapshot/tts_acoustic_modulation_calibration.py')}
    (out/'independent_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=OUT)
    check(parser.parse_args().out)
