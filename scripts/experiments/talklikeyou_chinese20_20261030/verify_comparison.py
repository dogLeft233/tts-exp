"""Local CPU recheck of the reported source-level generator effect differences."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    current = json.loads((args.output/'evidence/paired_scores.json').read_text())
    comparison = json.loads((args.output/'evidence/cross_model_comparison.json').read_text())
    baselines = {
        'FlashHead': args.repo/'results/soulx_flashhead_pro/chinese20_rms24_20261030_attempt01/paired_scores.json',
        'EchoMimicV3-Flash': args.repo/'results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/paired_scores.json',
    }
    checked = 0
    maximum = 0.
    for name, info in comparison['contrasts'].items():
        assert hashlib.sha256(baselines[name].read_bytes()).hexdigest() == info['baseline_pairs_sha256']
        baseline = json.loads(baselines[name].read_text())
        for subset, metrics in info['subsets'].items():
            for metric, saved in metrics.items():
                a = {r['source']:r for r in current[metric]}
                b = {r['source']:r for r in baseline[metric]}
                sources = sorted(s for s in a.keys() & b.keys() if subset == 'all20' or a[s]['strict_reference'] == (subset == 'same_reference12'))
                assert len(sources) == saved['n_sources']
                if not sources:
                    continue
                assert sources == saved['sources']
                n = np.array([b[s]['delta'] for s in sources])
                t = np.array([a[s]['delta'] for s in sources])
                difference = t-n
                rng = np.random.default_rng(20261030)
                indices = rng.integers(0, len(sources), size=(10000, len(sources)))
                ci = np.quantile(np.array([difference[i].mean() for i in indices]), [.025, .975])
                extreme = 0
                for _ in range(50):
                    signs = 2*rng.integers(0, 2, size=(2000, len(sources)))-1
                    extreme += np.count_nonzero(abs(signs@difference/len(sources)) >= abs(difference.mean())-1e-15)
                pvalue = (int(extreme)+1)/100001
                errors = [abs(n.mean()-saved['baseline_T_minus_N']), abs(t.mean()-saved['TalkLikeYou_T_minus_N']),
                          abs(difference.mean()-saved['effect_difference']), abs(ci-np.array(saved['ci95'])).max(),
                          abs(pvalue-saved['signflip_p'])]
                assert max(errors) < 1e-9, (name, subset, metric, errors)
                assert saved['same_strict_direction'] == sum(a[s]['delta']*b[s]['delta'] > 0 for s in sources)
                maximum = max(maximum, *errors)
                checked += 1
    for subset in ['same_reference12', 'extended_reference8', 'all20']:
        values = [v['subsets'][subset][m] for v in comparison['contrasts'].values() for m in current]
        ordered = sorted(values, key=lambda x:x.get('signflip_p', 1.))
        running = 0.
        for rank, cell in enumerate(ordered):
            running = min(1., max(running, (18-rank)*cell.get('signflip_p', 1.)))
            assert abs(cell['holm18_p']-running) < 1e-12
    result = dict(status='PASS', checked_nonempty_contrasts=checked, max_abs_error=float(maximum),
                  method='Independent current/baseline source joins,10000 bootstrap,100000 signflip and18-cell Holm families')
    (args.output/'local_comparison_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
