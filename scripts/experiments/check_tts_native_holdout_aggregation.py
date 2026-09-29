"""Additional independent aggregation audit; no scoring or endpoint changes."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'runs/tts_native_holdout_confirmation_20260926'


def read(path):
    return json.loads(path.read_text())


def main():
    summary = read(OUT / 'mechanism_summary.json')
    speaker = {r['id']: r['speaker'] for r in read(OUT / 'protocol.json')['rows']}
    errors = {'permutation_metrics': 0., 'speaker_mean': 0., 'ci99': 0.}
    checked = 0
    for key, result in summary.items():
        geom, kk = key.split('/')
        k = int(kk[1:])
        vectors = {}
        for sid in result['ids']:
            pair_values = []
            for arm in ('Q1', 'Q2'):
                dest = OUT / 'scores/pairs' / geom / sid / (arm + '.json')
                rec = read(dest)['k_results'][str(k)]
                s = rec['scores']
                if k == 3:
                    curves = np.load(dest.with_suffix('.npz'))
                    for name in ('N_base', 'T_base', 'N_M', 'T_M'):
                        x = curves[name]
                        b, d, anchor = np.median(x, axis=1), np.min(x, axis=1), x[:, 18]
                        independent = dict(C=(b-d).mean(), B=b.mean(), D=d.mean(),
                                           D_anchor=anchor.mean(), C_anchor=(b-anchor).mean(),
                                           best_lag=(np.argmin(x, axis=1)-15).mean())
                        for field, value in independent.items():
                            errors['permutation_metrics'] = max(errors['permutation_metrics'], abs(value-rec['permuted'][name][field]))
                values = {}
                for field in ('C', 'B', 'D', 'D_anchor', 'C_anchor', 'best_lag'):
                    gap = s['T_base'][field]-s['N_base'][field]
                    forward = s['T_M'][field]-s['N_base'][field]
                    reverse = s['T_base'][field]-s['N_M'][field]
                    block = dict(original=gap, T2N_effect=forward-gap, T2N_residual=forward,
                                 N2T_effect=gap-reverse, N2T_residual=reverse)
                    if k == 3:
                        p = rec['permuted']
                        pg = p['T_base'][field]-p['N_base'][field]
                        jf = p['T_M'][field]-p['N_base'][field]
                        jr = p['T_base'][field]-p['N_M'][field]
                        block.update(whole_gap=pg, whole_effect=pg-gap, T2N_joint=jf,
                                     T2N_interaction=(jf-pg)-(forward-gap), N2T_joint=jr,
                                     N2T_interaction=(jr-pg)-(reverse-gap),
                                     N_whole_effect=p['N_base'][field]-s['N_base'][field],
                                     T_whole_effect=p['T_base'][field]-s['T_base'][field])
                    values.update({(name, field): value for name, value in block.items()})
                values.update({('coefficient', f): rec[f] for f in result['coefficients']})
                pair_values.append(values)
            for name in pair_values[0]:
                vectors.setdefault(name, {}).setdefault(speaker[sid], []).append(
                    (pair_values[0][name]+pair_values[1][name])/2)
        for (name, field), grouped in vectors.items():
            y = np.array([np.mean(grouped[s]) for s in sorted(grouped)])
            exp = result['coefficients'][field] if name == 'coefficient' else result['contrasts'][name][field]
            draws = np.random.Generator(np.random.PCG64(20260926)).integers(len(y), size=(20000, len(y)))
            ci = np.quantile(y[draws].mean(axis=1), [.005, .995])
            errors['speaker_mean'] = max(errors['speaker_mean'], abs(y.mean()-exp['mean']))
            errors['ci99'] = max(errors['ci99'], float(abs(ci-exp['ci99']).max()))
            checked += 1
    assert max(errors.values()) < 1e-10, errors
    result = {'passed': True, 'errors': errors, 'summaries_checked': checked,
              'all256_repeat_metrics_recomputed': True,
              'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Added independent audit only, after score lock; no endpoint, data, score, or analysis changes'}
    (OUT / 'independent_aggregation_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
