"""Independent saved-WAV readback; keeps the original full-signal gates."""
import numpy as np
import soundfile as sf
from pystoi import stoi
from scipy.signal import correlate, correlation_lags
from calibrate import ROOT, RUN as PARENT, read, write, sha

RUN = ROOT / 'runs/phoneme_direction_20261026_repair02'


def measure(x):
    assert sha(x['source']) == x['source_sha256']
    assert sha(x['output']) == x['output_sha256']
    a, sr = sf.read(x['source']); b, sb = sf.read(x['output'])
    assert sr == sb == 16000 and a.shape == b.shape and np.isfinite(b).all()
    n = len(a) // 160
    ea = np.sqrt(np.mean(a[:n * 160].reshape(n, 160) ** 2, axis=1))
    eb = np.sqrt(np.mean(b[:n * 160].reshape(n, 160) ** 2, axis=1))
    c = correlate(eb - eb.mean(), ea - ea.mean(), method='fft')
    l = correlation_lags(n, n); mask = abs(l) <= 20
    lag = int(l[mask][np.argmax(c[mask])]) * 10
    score = float(stoi(a, b, sr)); clipped = int(np.sum(abs(b) >= 1))
    rms = float(abs(np.sqrt(np.mean(b * b) / np.mean(a * a)) - 1))
    failures = []
    if score < .75: failures.append('stoi')
    if abs(lag) > 20: failures.append('lag')
    if clipped: failures.append('clipping')
    if rms > .001: failures.append('rms')
    return dict(x, stoi=score, clipped=clipped, rms_relative_error=rms,
                global_envelope_lag_ms=lag, failures=failures)


def main():
    probes = [measure(x) for x in read(RUN / 'probe_outputs.json')['rows']]
    summary = {}
    for kind in sorted({x['kind'] for x in probes}):
        group = [x for x in probes if x['kind'] == kind]
        summary[kind] = {'sources': len(group), 'stoi_median': float(np.median([x['stoi'] for x in group])),
                         'failed': [x for x in group if x['failures']]}
    write(RUN / 'probe_quality.json', {'rows': probes, 'summary': summary, 'candidate_only': True})
    rows = []
    for shard in range(3):
        contract = read(RUN / f'full_peak95_shard{shard}_contract.json')
        previous = read(ROOT / f'runs/phoneme_direction_20261026_repair01/regular_fit_shard{shard}_contract.json')
        assert contract['checkpoint_sha256'] == previous['checkpoint_sha256']
        assert contract['parent_contract_sha256'] == sha(PARENT / 'calibration_contract.json')
        rows.extend(read(RUN / f'full_peak95_shard{shard}_complete.json')['rows'])
    expected = {x['id'] for x in read(PARENT / 'input_manifest.json')['fit'] if x['arm'] == 'N'}
    assert len(rows) == 300 and {x['id'] for x in rows} == expected
    measured = [measure(x) for x in rows]
    median = float(np.median([x['stoi'] for x in measured]))
    failed = [x for x in measured if x['failures']]
    result = {'status': 'PASS_BASIC_RECONSTRUCTION' if not failed and median >= .85 else 'HOLD_RECONSTRUCTION',
              'sources': 300, 'speakers': len({x['speaker_id'] for x in measured}),
              'stoi_median': median, 'stoi_min': min(x['stoi'] for x in measured),
              'failed': failed, 'rows': measured, 'floors_unchanged': True,
              'parent_contract_sha256': sha(PARENT / 'calibration_contract.json'),
              'script_sha256': sha(__file__), 'candidate_only': True,
              'development_started': False, 'confirmation_started': False}
    write(RUN / 'full_peak95_quality.json', result)
    print({k: result[k] for k in ['status', 'sources', 'speakers', 'stoi_median', 'stoi_min']},
          'failed', [(x['id'], x['stoi'], x['failures']) for x in failed], flush=True)


if __name__ == '__main__':
    main()
