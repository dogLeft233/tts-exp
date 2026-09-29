"""Independent scalar/FFT audit; imports neither primary nor parent transforms."""
from pathlib import Path
import hashlib
import json
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tts_native_level_match_calibration_20260927'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def energy(a):
    return float(np.linalg.norm(a)/np.sqrt(a.size))


def fft(a):
    padded = np.pad(a, (256, 256+(-len(a)) % 128), mode='reflect')
    # Explicit frame indexing, independent of parent's sliding-window helper.
    idx = np.arange(512)[None, :] + 128*np.arange(1+(len(padded)-512)//128)[:, None]
    window = .5-.5*np.cos(2*np.pi*np.arange(512)/512)
    return np.fft.rfft(padded[idx]*window, axis=1).T


def features(a):
    z = fft(a)
    amp = abs(z)
    maximum = amp.max()
    logs = 20*np.log10(np.maximum(amp, maximum/10000)) if maximum else np.zeros_like(amp)
    mean = np.mean(logs, axis=1)
    kernel = np.hanning(9)
    padded = np.pad(mean, 4, mode='reflect')
    smooth = np.array([np.dot(padded[i:i+9], kernel)/kernel.sum() for i in range(len(mean))])
    shape = smooth-np.mean(smooth[1:])
    # Remove row means then column means instead of parent mu/f/e expression.
    residual = logs-np.mean(logs, axis=1, keepdims=True)
    residual -= np.mean(residual, axis=0, keepdims=True)
    return shape, energy(residual), mean, z


def reconstruct_eq(a, gain):
    z = fft(a)*np.exp(np.log(10)*gain[:, None]/20)
    window = .5-.5*np.cos(2*np.pi*np.arange(512)/512)
    blocks = np.fft.irfft(z.T, n=512, axis=1)*window
    total = (len(blocks)-1)*128+512
    output, weights = np.zeros(total), np.zeros(total)
    for j in range(len(blocks)):
        output[j*128:j*128+512] += blocks[j]
        weights[j*128:j*128+512] += window*window
    output = output[256:256+len(a)]/weights[256:256+len(a)]
    return output*(energy(a)/energy(output))


def main():
    p, rows, pairs = [read(OUT/(name+'.json')) for name in ('protocol', 'waveform_manifest', 'pairs')]
    assert sha(OUT/'protocol.json') == (OUT/'protocol.sha256').read_text().strip()
    for path, digest in {**p['parent_hashes'], **p['code_hashes'], **read(OUT/'artifact_hashes.json')}.items():
        assert sha(path) == digest, path
    lookup = {(v['id'], v['arm'], v['condition']): v for v in rows}
    maxima = {}
    gates = {}
    def bound(name, value, limit):
        maxima[name] = max(maxima.get(name, 0.), float(value))
        gates[name] = gates.get(name, True) and bool(value <= limit)
    for pair in pairs:
        sid = pair['id']
        originals, eqs = {}, {}
        for arm in ('N', 'T'):
            s = next(s for s in p['sources'] if s['id'] == sid and s['arm'] == arm)
            assert sha(s['path']) == s['sha256']
            pcm, sr = sf.read(s['path'], dtype='int16')
            assert sr == 16000 and sf.info(s['path']).subtype == 'PCM_16'
            originals[arm] = pcm.astype('float64')/32768
        shapes = {a: features(x)[0] for a, x in originals.items()}
        midpoint = (shapes['N']+shapes['T'])/2
        for arm in originals:
            eqs[arm] = reconstruct_eq(originals[arm], np.clip(midpoint-shapes[arm], -6, 6))
        rmses = {a: energy(x) for a, x in originals.items()}
        assert all(r > 1e-8 for r in rmses.values())
        m0 = np.sqrt(rmses['N']*rmses['T'])
        mcap = min(.98*rmses[a]/np.max(abs(x)) for a in originals for x in (originals[a], eqs[a]))
        target = min(m0, mcap)
        for k, v in (('m0', m0), ('mcap', mcap), ('m', target)):
            bound('scalar_target', abs(pair[k]-v), 1e-12)
        saved_rms = {}
        for arm, x in originals.items():
            gain = target/rmses[arm]
            bound('gain', abs(gain-pair['gain'][arm]), 1e-12)
            assert pair['cap_active'] == (mcap < m0)
            for cond in p['conditions']:
                row = lookup[sid, arm, cond]
                assert sha(row['path']) == row['sha256'] and sha(row['float64_path']) == row['float64_sha256']
                y = np.load(row['float64_path'])
                saved, sr = sf.read(row['path'], dtype='float64')
                assert sr == 16000 and len(y) == len(saved) == len(x)
                gates['finite'] = gates.get('finite', True) and bool(np.isfinite(y).all() and np.isfinite(saved).all())
                gates['no_clip'] = gates.get('no_clip', True) and bool(np.max(abs(y)) < 1 and np.max(abs(saved)) < 1)
                if cond == 'RAW':
                    assert sha(row['path']) == sha(next(s['path'] for s in p['sources'] if s['id'] == sid and s['arm'] == arm))
                if cond in ('RAW', 'identity'):
                    assert np.array_equal(saved, x) and np.array_equal(y, x)
                expected = {'RAW': x, 'identity': x, 'EQ': eqs[arm], 'LEVEL': x*gain, 'EQ_LEVEL': eqs[arm]*gain}[cond]
                bound('independent_waveform_reconstruction', np.max(abs(y-expected)), 1e-12)
                saved_rms[arm, cond] = energy(saved)
                if cond in ('LEVEL', 'EQ_LEVEL'):
                    base = x if cond == 'LEVEL' else eqs[arm]
                    fbase = features(base)
                    for label, arr in (('float64', y), ('float32', saved)):
                        f = features(arr)
                        bound(label+'_rms_target_relative', abs(energy(arr)/target-1), 1e-5)
                        bound(label+'_divide_gain', np.max(abs(arr/gain-base)), 1e-7)
                        normalized = arr/np.linalg.norm(arr)-base/np.linalg.norm(base)
                        bound(label+'_cosine_error', .5*np.sum(normalized**2), 1e-10)
                        bound(label+'_shape_db', np.max(abs(f[0]-fbase[0])), 1e-8 if label == 'float64' else 1e-4)
                        bound(label+'_R_RMS_db', abs(f[1]-fbase[1]), 1e-8 if label == 'float64' else 1e-5)
                        # EQ has possible tiny zero reconstruction differences. Check
                        # zeros against saved primary float64 EQ, already verified.
                        source = x if cond == 'LEVEL' else np.load(lookup[sid, arm, 'EQ']['float64_path'])
                        assert np.all(arr[source == 0] == 0)
            for key in ('mean_log_db', 'shape_db'):
                pass
        for cond in ('LEVEL', 'EQ_LEVEL'):
            bound('saved_pair_rms_relative', abs(saved_rms['N', cond]-saved_rms['T', cond])/target, 1e-5)
        for cond in p['conditions']:
            fs = [features(np.load(lookup[sid, arm, cond]['float64_path'])) for arm in ('N', 'T')]
            for key, index in (('shape_db', 0), ('mean_log_db', 2)):
                bound('pair_distance', abs(energy((fs[1][index]-fs[0][index])[1:])-pair['distances'][cond][key]), 1e-9)
        print('independent', sid, flush=True)
    result = {'status': 'PASS' if all(gates.values()) else 'FAIL', 'gates': gates, 'max_errors': maxima,
              'source_waveforms': 52, 'pairs': len(pairs), 'waveforms_verified': len(rows), 'GPU': 0,
              'method': 'No primary or parent imports; explicit frame index FFT, cosine Hann, independent overlap-add EQ, norm-based RMS and scalar reconstruction; all artifact/input hashes checked'}
    (OUT/'independent_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
