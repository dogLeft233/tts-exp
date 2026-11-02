"""Recompute original VocaLiST endpoints and paired statistics from saved matrices."""
import json
from collections import Counter
from pathlib import Path

import numpy as np


def audit(root):
    run = Path(root)
    read = lambda name: json.loads((run / name).read_text())
    scores = read('scores.json')
    pairs = read('sync_pairs.json')
    summary = read('summary.json')
    assert len(scores) == 80 and len(pairs) == 60
    derived = {}
    offsets = {'N': Counter(), 'C': Counter()}
    boundary = []
    for key, score in scores.items():
        z = np.load(run / 'matrices' / (key + '.npy'))
        assert z.shape[0] == score['windows'] and z.shape[1] == 31
        assert np.isfinite(z).all()
        assert len(z) > 40
        # Diagnostic only: remove boundary rows before averaging candidates.
        # Full author matrices contain -1000 masks that otherwise force a zero peak.
        offset = int(15-z[20:-20].mean(axis=0).argmax())
        offsets[score['arm']][offset] += 1
        correct = np.abs(15-z.argmax(axis=1)) <= 1
        boundary.append(dict(id=key, arm=score['arm'], full_correct=int(correct.sum()),
                             interior_correct=int(correct[20:-20].sum()), full_windows=len(z),
                             interior_windows=len(z)-40, guard20_k5=float(correct[20:-20].mean())))
        derived[key] = {}
        for k in range(5, 16, 2):
            radius = (k - 5) // 2
            count = 0
            for i in range(len(z)):
                curve = z[max(0, i-radius):min(len(z), i+radius+1)].mean(axis=0)
                count += abs(15 - int(curve.argmax())) <= 1
            value = count / len(z)
            assert value == score['metrics']['K' + str(k)]
            derived[key]['K' + str(k)] = value
    natural = {v['source']: key for key, v in scores.items() if v['arm'] == 'N'}
    assert len(natural) == 20
    detailed = {(p['metric'], p['id']): p for p in read('paired_scores.json')}
    checked = 0
    for k in range(5, 16, 2):
        metric = 'K' + str(k)
        for label, cohort in [('all20', pairs), ('strict12', [p for p in pairs if p['strict_reference']])]:
            n = np.array([derived[natural[p['source']]][metric] for p in cohort])
            t = np.array([derived[p['id']][metric] for p in cohort])
            delta = t - n
            sync = np.array([p['delta']['SyncC'] for p in cohort])
            target = summary['views'][metric][label]
            expected = dict(tts_wins=int((delta > 0).sum()), natural_wins=int((delta < 0).sum()),
                            ties=int((delta == 0).sum()), conflict_count=int((delta * sync < 0).sum()),
                            sync_tts_metric_natural=int(((sync > 0) & (delta < 0)).sum()),
                            sync_natural_metric_tts=int(((sync < 0) & (delta > 0)).sum()))
            for name, value in expected.items():
                assert target[name] == value
            for name, value in [('mean_delta', delta.mean()), ('natural_mean', n.mean()), ('tts_mean', t.mean())]:
                assert abs(target[name] - float(value)) < 1e-12
            if label == 'all20':
                for p, nv, tv, dv, sv in zip(cohort, n, t, delta, sync):
                    row = detailed[(metric, p['id'])]
                    assert (row['N'], row['T'], row['delta'], row['delta_sync_c']) == (nv, tv, dv, sv)
                    checked += 1
    result = dict(status='PASS', matrices=80, independently_recomputed_endpoints=480,
                  checked_pair_endpoints=checked, independent_sources=20, gpu_inference_repeated=False,
                  diagnostic_guard20_offset_counts={arm: dict(counts) for arm, counts in offsets.items()},
                  diagnostic_only_no_native_score_correction=True)
    (run / 'independent_audit.json').write_text(json.dumps(result, indent=2))
    boundary_summary = {}
    for arm in ['N', 'C']:
        rows = [d for d in boundary if d['arm'] == arm]
        full = sum(d['full_correct'] for d in rows)
        interior = sum(d['interior_correct'] for d in rows)
        boundary_summary[arm] = dict(media=len(rows), mean_guard20_k5=float(np.mean([d['guard20_k5'] for d in rows])),
                                    full_correct_windows=full, guard20_correct_windows=interior,
                                    boundary_share_of_all_correct=(full-interior)/full if full else None,
                                    zero_interior_media=sum(d['interior_correct'] == 0 for d in rows))
    (run / 'boundary_floor_diagnostic.json').write_text(json.dumps(
        dict(summary=boundary_summary, per_media=boundary, diagnostic_only_native_scores_unchanged=True), indent=2))
    return result


if __name__ == '__main__':
    import sys
    print(json.dumps(audit(sys.argv[1])))
