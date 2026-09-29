"""CPU old26cal scalar level matching; freeze before waveform computation."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import types
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tts_native_level_match_calibration_20260927'
PARENT = ROOT / 'runs/tts_native_spectrum_match_calibration_20260926'
CONDITIONS = ('RAW', 'identity', 'EQ', 'LEVEL', 'EQ_LEVEL')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def parent_module():
    path = PARENT / 'code_snapshot/tts_native_spectrum_match_calibration.py'
    m = types.ModuleType('frozen_spectrum')
    m.__file__ = str(ROOT / 'scripts/experiments/tts_native_spectrum_match_calibration.py')
    exec(compile(path.read_text(), str(path), 'exec'), m.__dict__)
    return m


def freeze():
    assert not (OUT / 'protocol.json').exists()
    parent = read(PARENT / 'protocol.json')
    assert sha(PARENT / 'protocol.json') == read(PARENT / 'final.json')['protocol_sha256']
    assert read(PARENT / 'final.json')['validation_passed']
    deps = [PARENT / k for k in ('protocol.json', 'final.json', 'independent_validation.json', 'summary.json', 'waveform_manifest.json')]
    for name, digest in parent['code_hashes'].items():
        path = PARENT / 'code_snapshot' / Path(name).name
        assert sha(path) == digest
        deps.append(path)
        if '/runs/' in name:
            assert sha(name) == digest
            deps.append(Path(name))
    scripts = [Path(__file__), ROOT / 'scripts/experiments/check_tts_native_level_match_calibration.py']
    protocol = {
        'status': 'frozen_before_calculation', 'created_epoch': time.time(),
        'scope': 'CPU only; old26cal original52 PCM16; GPU=0; no evaluation audio, scores, ASR, TFG or SyncNet',
        'sources': parent['sources'], 'cal_ids': parent['cal_ids'], 'conditions': CONDITIONS,
        'algorithm': {'r': 'sqrt(mean(x*x)) float64; x=original int16/32768', 'silence': 'r<=1e-8 undefined; preserve sample and stop scoring',
            'm0': 'sqrt(rN*rT)', 'mcap': 'min(.98*rN/pN,.98*rT/pT,.98*rN/pEQ_N,.98*rT/pEQ_T)', 'm': 'min(m0,mcap)',
            'gain': 'm/r_arm; same gain for LEVEL and EQ_LEVEL', 'identity': 'original PCM/32768 directly float32; no STFT',
            'EQ': 'parent frozen spectral_vectors; midpoint shape; clip gain +/-6 dB; eq(original,gain), float64 own-RMS-restored; before parent shared headroom',
            'LEVEL': 'x*gain', 'EQ_LEVEL': 'EQ*gain', 'headroom': 'none for baseline or EQ; no additional shared scaling',
            'zero': 'scalar preserves every exact zero of its own input; EQ may fill original time-domain zeros; all-zero log=0 but silent pair undefined'},
        'gates': {'finite_length_no_clip': True, 'identity_saved_exact': True,
            'rms_pair_and_target_relative_max': 1e-5, 'saved_divide_gain_max_abs': 1e-7,
            'float64_cosine_error_max': 1e-10, 'float64_shape_and_R_RMS_max_abs_db': 1e-8,
            'saved_float32_shape_max_abs_db': 1e-4, 'saved_float32_R_RMS_abs_db': 1e-5,
            'saved_float32_cosine_error_max': 1e-10},
        'spectral_definition': parent['stft'], 'smooth_shape': parent['spectrum']['smooth'],
        'R': 'parent v1 double-centered relative-floor log-STFT residual; report RMS',
        'statistics': 'descriptive only; no sample deletion, thresholds or algorithm tuned; parent acoustic gates inherited without refitting',
        'parent_hashes': {str(f): sha(f) for f in deps}, 'code_hashes': {str(f): sha(f) for f in scripts}}
    write(OUT / 'protocol.json', protocol)
    (OUT / 'protocol.sha256').write_text(sha(OUT / 'protocol.json') + '\n')
    for path in scripts:
        dest = OUT / 'code_snapshot' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print('FROZEN', sha(OUT / 'protocol.json'), flush=True)


def rms(x):
    return float(np.sqrt(np.mean(x*x)))


def stats(xs):
    a = np.asarray(xs, dtype=float)
    return {'n': len(a), 'min': float(a.min()), 'median': float(np.median(a)), 'mean': float(a.mean()), 'max': float(a.max())}


def features(m, x):
    vectors, z = m.spectral_vectors(x)
    residual, _ = m.v1.decompose(z)
    return {'mean_log_db': vectors['mean_log_db'], 'shape_db': vectors['smooth_mean_log_shape_db'], 'R_rms_db': rms(residual)}


def invariant(a, b, fa, fb, g):
    cosine = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)))
    return {'divide_gain_max_abs': float(np.max(abs(b/g-a))), 'cosine_error': abs(1-cosine),
            'shape_max_abs_db': float(np.max(abs(fb['shape_db']-fa['shape_db']))),
            'R_rms_abs_db': abs(fb['R_rms_db']-fa['R_rms_db']), 'zero_preserved': bool(np.all(b[a == 0] == 0))}


def run():
    p = read(OUT / 'protocol.json')
    assert sha(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    assert not (OUT / 'summary.json').exists()
    for path, digest in {**p['code_hashes'], **p['parent_hashes']}.items():
        assert sha(path) == digest, path
    m = parent_module()
    metrics, pairs, manifest, checks = [], [], [], []
    parent_waves = {(v['id'], v['arm'], v['condition']): v for v in read(PARENT / 'waveform_manifest.json')}
    for sid in p['cal_ids']:
        sources = {v['arm']: v for v in p['sources'] if v['id'] == sid}
        xs, eqs = {}, {}
        for arm, source in sources.items():
            assert sha(source['path']) == source['sha256']
            assert sf.info(source['path']).subtype == 'PCM_16'
            x, sr = sf.read(source['path'], dtype='int16')
            assert sr == 16000 and x.ndim == 1
            xs[arm] = x.astype(np.float64)/32768
        if any(rms(x) <= 1e-8 for x in xs.values()):
            write(OUT / 'undefined_silence.json', {'id': sid, 'status': 'BLOCKED_UNDEFINED', 'no_samples_deleted': True})
            raise RuntimeError('Undefined silent pair; stop without scoring')
        shapes = {a: m.spectral_vectors(x)[0]['smooth_mean_log_shape_db'] for a, x in xs.items()}
        target = (shapes['N']+shapes['T'])/2
        parent_exact = {}
        for arm, x in xs.items():
            eqs[arm], _ = m.eq(x, np.clip(target-shapes[arm], -6, 6))
            old = parent_waves[sid, arm, 'EQ']['pre_headroom']
            assert sha(old['path']) == old['sha256']
            saved, _ = sf.read(old['path'], dtype='float32')
            parent_exact[arm] = bool(np.array_equal(saved, eqs[arm].astype(np.float32)))
            assert parent_exact[arm], 'parent EQ reconstruction mismatch'
        rs = {a: rms(x) for a, x in xs.items()}
        m0 = float(np.sqrt(rs['N']*rs['T']))
        caps = {f'{a}/{label}': .98*rs[a]/float(np.max(abs(y))) for a in xs for label, y in [('RAW', xs[a]), ('EQ', eqs[a])]}
        mcap = min(caps.values())
        level = min(m0, mcap)
        gs = {a: level/rs[a] for a in xs}
        pair = {'id': sid, 'speaker': sources['N']['speaker'], 'original_T_N_rms_db': float(20*np.log10(rs['T']/rs['N'])),
                'm0': m0, 'mcap': mcap, 'm': level, 'cap_active': mcap < m0, 'cap_candidates': caps,
                'limiting_candidate': min(caps, key=caps.get), 'gain': gs,
                'gain_db': {a: float(20*np.log10(g)) for a, g in gs.items()}, 'parent_EQ_float32_exact': parent_exact, 'distances': {}, 'rms': {}}
        f_by, stored_by = {}, {}
        for arm, x in xs.items():
            ys = {'RAW': x, 'identity': x.copy(), 'EQ': eqs[arm], 'LEVEL': x*gs[arm], 'EQ_LEVEL': eqs[arm]*gs[arm]}
            f_by[arm], stored_by[arm] = {}, {}
            for cond, y in ys.items():
                dest = OUT / 'waveforms' / sid / arm / (cond + '.wav')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if cond == 'RAW':
                    shutil.copyfile(sources[arm]['path'], dest)
                else:
                    sf.write(dest, y, 16000, subtype='FLOAT')
                npy = OUT / 'float64' / sid / arm / (cond + '.npy')
                npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(npy, y)
                stored, rate = sf.read(dest, dtype='float64')
                ff = features(m, y)
                f_by[arm][cond] = ff
                stored_by[arm][cond] = stored
                saved_ff = features(m, stored)
                metric = {'id': sid, 'arm': arm, 'condition': cond, 'n': len(y), 'global_rms_float64': rms(y),
                          'global_rms_saved': rms(stored), 'peak_float64': float(np.max(abs(y))),
                          'peak_saved': float(np.max(abs(stored))), 'finite': bool(np.isfinite(y).all() and np.isfinite(stored).all()),
                          'same_length': len(y) == len(x) == len(stored), 'rate': rate,
                          'identity_exact': bool(np.array_equal(stored, x)) if cond == 'identity' else None,
                          'zero_sample_count': int(np.sum(y == 0)), 'R_rms_db': ff['R_rms_db'],
                          'target_relative_error_saved': abs(rms(stored)/level-1) if cond in ('LEVEL', 'EQ_LEVEL') else None,
                          'mean_log_db_mean': float(ff['mean_log_db'].mean())}
                if cond in ('LEVEL', 'EQ_LEVEL'):
                    base = 'identity' if cond == 'LEVEL' else 'EQ'
                    inv64 = invariant(ys[base], y, f_by[arm][base], ff, gs[arm])
                    inv32 = invariant(ys[base], stored, f_by[arm][base], saved_ff, gs[arm])
                    checks.append({'id': sid, 'arm': arm, 'condition': cond, 'float64': inv64, 'saved_float32': inv32})
                metrics.append(metric)
                manifest.append({'id': sid, 'arm': arm, 'condition': cond, 'path': str(dest), 'sha256': sha(dest), 'float64_path': str(npy), 'float64_sha256': sha(npy)})
        for cond in CONDITIONS:
            fn, ft = f_by['N'][cond], f_by['T'][cond]
            pair['distances'][cond] = {key: rms((ft[key]-fn[key])[1:]) for key in ('mean_log_db', 'shape_db')}
            pair['rms'][cond] = {a: rms(stored_by[a][cond]) for a in xs}
        pair['rms_pair_relative_error'] = {c: abs(pair['rms'][c]['N']-pair['rms'][c]['T'])/level for c in ('LEVEL', 'EQ_LEVEL')}
        pairs.append(pair)
        print('calibrated', sid, flush=True)
    maxima = {stage: {k: max(c[stage][k] for c in checks) for k in ('divide_gain_max_abs', 'cosine_error', 'shape_max_abs_db', 'R_rms_abs_db')} for stage in ('float64', 'saved_float32')}
    gates = {'finite': all(v['finite'] for v in metrics), 'same_length': all(v['same_length'] for v in metrics),
             'no_clip': all(v['peak_float64'] < 1 and v['peak_saved'] < 1 for v in metrics),
             'identity_exact': all(v['identity_exact'] for v in metrics if v['condition'] == 'identity'),
             'saved_target_rms': all(v['target_relative_error_saved'] <= 1e-5 for v in metrics if v['condition'] in ('LEVEL', 'EQ_LEVEL')),
             'saved_pair_rms': all(v <= 1e-5 for pp in pairs for v in pp['rms_pair_relative_error'].values()),
             'saved_divide_gain': maxima['saved_float32']['divide_gain_max_abs'] <= 1e-7,
             'float64_cosine': maxima['float64']['cosine_error'] <= 1e-10,
             'float64_shape': maxima['float64']['shape_max_abs_db'] <= 1e-8,
             'float64_R_rms': maxima['float64']['R_rms_abs_db'] <= 1e-8,
             'zero_preserved': all(c[s]['zero_preserved'] for c in checks for s in ('float64', 'saved_float32')),
             'saved_shape': maxima['saved_float32']['shape_max_abs_db'] <= 1e-4,
             'saved_R_rms': maxima['saved_float32']['R_rms_abs_db'] <= 1e-5,
             'saved_cosine': maxima['saved_float32']['cosine_error'] <= 1e-10}
    summary = {'status': 'PASS' if all(gates.values()) else 'FAIL', 'gates': gates, 'pairs': len(pairs), 'source_waveforms': len(p['sources']),
               'GPU': 0, 'evaluation_waveforms': 0, 'new_scores': 0, 'max_errors': maxima,
               'original_T_N_rms_db': stats([v['original_T_N_rms_db'] for v in pairs]),
               'T_louder_pairs': sum(v['original_T_N_rms_db'] > 0 for v in pairs),
               'cap_active_count': sum(v['cap_active'] for v in pairs), 'cap_active_fraction': float(np.mean([v['cap_active'] for v in pairs])),
               'gain_db': {a: stats([v['gain_db'][a] for v in pairs]) for a in ('N', 'T')},
               'gain_direction': {a: {d: sum((v['gain'][a] > 1 if d == 'increase' else v['gain'][a] < 1 if d == 'decrease' else v['gain'][a] == 1) for v in pairs) for d in ('increase', 'decrease', 'unchanged')} for a in ('N', 'T')},
               'pair_distances': {c: {k: stats([v['distances'][c][k] for v in pairs]) for k in ('mean_log_db', 'shape_db')} for c in CONDITIONS},
               'global_rms_by_arm': {c: {a: stats([v['rms'][c][a] for v in pairs]) for a in ('N', 'T')} for c in CONDITIONS}}
    for name, value in [('metrics', metrics), ('pairs', pairs), ('invariants', checks), ('waveform_manifest', manifest), ('summary', summary)]:
        write(OUT / (name + '.json'), value)
    write(OUT / 'artifact_hashes.json', {str(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ('.log',) and f.name != 'artifact_hashes.json'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('freeze', 'run'))
    globals()[ap.parse_args().stage]()
