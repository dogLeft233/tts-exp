"""Independent reconstruction from saved features; never imports the producer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def matrix(audio, video, q, mode):
    a, v = audio.astype('float64'), video.astype('float64')
    assert np.isfinite(a).all() and np.isfinite(v).all()
    if mode != 'distance':
        a = a / np.sqrt(np.sum(a*a, axis=1))[:, None]
        v = v / np.sqrt(np.sum(v*v, axis=1))[:, None]
    result = np.empty((len(q), 31), dtype='float64')
    for column, lag in enumerate(range(-15, 16)):
        if mode in ('distance', 'unit_distance'):
            difference = v[q] - a[q+lag] + 1e-6
            result[:, column] = -np.sqrt(np.sum(difference*difference, axis=1))
        else:
            scores = np.zeros(len(q))
            for t in range(5):
                frame = np.sum(v[q+t] * a[q+t+lag], axis=1)
                scores += np.clip(frame, 0, None) if mode == 'relu_cosine' else frame
            result[:, column] = scores / 5
    return result


def endpoint(x):
    curve = np.sum(x, axis=0) / len(x)
    peak = int(np.argmax(curve))
    best, background, zero = float(curve[peak]), float(np.sort(curve)[15]), float(curve[15])
    return dict(best=best, background=background, zero=zero, contrast=best-background,
                zero_contrast=zero-background, search_uplift=best-zero,
                peak_lag_frames=peak-15, curve=curve.tolist())


def audit(root):
    rows = read(root/'inputs.json')
    items = read(root/'items.json')
    inputs = {r['id']: r for r in rows}
    assert len(items) == len(rows) == 80 and len(inputs) == 80
    assert digest(root/'source_frozen.py') == read(root/'protocol.json')['source_sha256']
    assert Counter(r['arm'] for r in rows) == {'N': 20, 'C': 60}
    metrics = dict(sync=('sync', 'distance'), sync_unit=('sync', 'unit_distance'),
                   avs=('avs', 'cosine'), co=('co', 'relu_cosine'), co_raw=('co', 'cosine'))
    reconstructed, controls = {}, []
    errors = dict(matrix=0., endpoint=0., bootstrap=0., pair=0., closure=0.)
    for item in items:
        key = item['id']
        row = inputs[key]
        p = root/'items'/f'{key}.npz'
        assert digest(p) == item['npz_sha256']
        with np.load(p, allow_pickle=False) as z:
            q = z['queries']
            assert np.array_equal(z['lags'], np.arange(-15, 16))
            assert np.array_equal(q, np.arange(row['query_start'], row['query_end_exclusive']))
            assert len(q) == row['query_count'] and len(q) > 0
            assert q[0] >= 15 and q[-1]+15 < row['sync_windows']
            assert q[-1]+15+4 < row['common_native_frames']
            assert row['common_native_frames'] <= min(row['source_frames'], row['samples']//640)
            assert z['sync_mfcc'].shape[1] >= (row['sync_windows']-1)*4+20
            reconstructed[key] = {}
            for metric, (prefix, kind) in metrics.items():
                rebuilt = matrix(z[prefix+'_audio'], z[prefix+'_video'], q, kind)
                error = float(np.abs(rebuilt-z[metric+'_matrix']).max())
                errors['matrix'] = max(errors['matrix'], error)
                assert error < 1e-9, (key, metric, error)
                ep = endpoint(rebuilt)
                old = item['endpoints'][metric]
                assert ep['peak_lag_frames'] == old['peak_lag_frames']
                for field in ('best', 'background', 'zero', 'contrast', 'zero_contrast', 'search_uplift'):
                    errors['endpoint'] = max(errors['endpoint'], abs(ep[field]-old[field]))
                assert np.allclose(ep['curve'], old['curve'], rtol=0, atol=1e-10)
                reconstructed[key][metric] = ep
            for c in item['controls']:
                prefix = c['metric']
                label = f"{prefix}_control_{c['shift_frames']}"
                kind = metrics[prefix][1]
                rebuilt = matrix(z[label+'_audio'], z[prefix+'_video'], q, kind)
                error = float(np.abs(rebuilt-z[label+'_matrix']).max())
                errors['matrix'] = max(errors['matrix'], error)
                assert error < 1e-9
                peak = endpoint(rebuilt)['peak_lag_frames']
                relative = peak-reconstructed[key][prefix]['peak_lag_frames']-c['shift_frames']
                assert relative == c['relative_error_frames']
                assert c['passed'] == (abs(relative) <= 1)
                controls.append(dict(id=key, metric=prefix, passed=abs(relative) <= 1, relative=relative))
    fields = ['best', 'background', 'zero', 'contrast', 'zero_contrast', 'search_uplift',
              'best_contribution', 'background_contribution']
    naturals = {r['sample_id']: r['id'] for r in rows if r['arm']=='N'}
    pairs = []
    old_pairs = {p['id']: p for p in read(root/'pairs.json')}
    for r in rows:
        if r['arm'] == 'N':
            continue
        nid = naturals[r['sample_id']]
        assert inputs[nid]['query_count'] == r['query_count']
        assert inputs[nid]['strict_reference'] == r['strict_reference']
        delta = {}
        for metric in metrics:
            te, ne = reconstructed[r['id']][metric], reconstructed[nid][metric]
            d = {f: te[f]-ne[f] for f in fields[:6]}
            d['best_contribution'], d['background_contribution'] = d['best'], -d['background']
            closure = abs(d['contrast']-d['best_contribution']-d['background_contribution'])
            errors['closure'] = max(errors['closure'], closure)
            for f in fields:
                errors['pair'] = max(errors['pair'], abs(d[f]-old_pairs[r['id']]['delta'][metric][f]))
            delta[metric] = d
        pairs.append(dict(id=r['id'], source=r['sample_id'], strict_reference=r['strict_reference'], delta=delta))
    assert len(pairs) == 60 and len(naturals) == 20
    summary = read(root/'summary.json')
    for view, ps in [('all20', pairs), ('strict12', [p for p in pairs if p['strict_reference']]),
                     ('extended8', [p for p in pairs if not p['strict_reference']])]:
        sources = sorted(set(p['source'] for p in ps))
        assert len(sources) == summary[view]['sources'] and len(ps) == summary[view]['pairs']
        resampling = np.random.default_rng(20261028).integers(0, len(sources), size=(10000, len(sources)))
        for metric in metrics:
            for field in fields:
                values = np.array([sum(p['delta'][metric][field] for p in ps if p['source']==s)/
                                   sum(p['source']==s for p in ps) for s in sources])
                ci = np.quantile(np.take(values, resampling).mean(1), [.025, .975])
                saved = summary[view]['metrics'][metric][field]
                err = max(abs(float(values.mean())-saved['mean']), float(np.max(abs(ci-saved['ci95']))))
                errors['bootstrap'] = max(errors['bootstrap'], err)
                assert err < 1e-10
                assert saved['tts_higher'] == sum(p['delta'][metric][field]>0 for p in ps)
                assert saved['natural_higher'] == sum(p['delta'][metric][field]<0 for p in ps)
                assert saved['ties'] == sum(p['delta'][metric][field]==0 for p in ps)
    assert max(errors.values()) < 1e-9
    cs = {m: dict(total=sum(c['metric']==m for c in controls),
                  passed=sum(c['passed'] for c in controls if c['metric']==m)) for m in ['sync','avs','co']}
    saved_controls = read(root/'controls.json')['summary']
    for m in cs:
        assert cs[m]['total']==saved_controls[m]['total'] and cs[m]['passed']==saved_controls[m]['passed']
    result = dict(status='PASS', clips=80, pairs=60, sources=20, metrics=5, controls=cs,
                  max_errors=errors, source_snapshot_sha256=digest(root/'source_frozen.py'),
                  audit_source_sha256=digest(__file__), claims='Numerical reconstruction only; not perceptual validation')
    save(root/'parent_independent_audit.json', result)
    print(json.dumps(result, indent=2))
    return reconstructed, pairs


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read(root/'items.json')
    stats = read(root/'summary.json')['all20']['metrics']
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), constrained_layout=True)
    labels = [('sync','SyncNet: negative distance'), ('avs','AVS_u: cosine'), ('co','CoGenAV: ReLU cosine')]
    for col, (metric, label) in enumerate(labels):
        ax = axes[0, col]
        for arm, color, name in [('N','#2977b5','Natural'), ('C','#d66324','Cloud TTS')]:
            curves = [r['endpoints'][metric]['curve'] for r in rows if r['arm']==arm]
            ax.plot(np.arange(-15,16)*40, np.mean(curves, axis=0), label=name, color=color, lw=2)
        ax.axvline(0, color='gray', ls=':', lw=1)
        ax.set(title=label, xlabel='Audio feature index shift (ms)', ylabel='Mean score (higher = closer)')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        ax = axes[1,col]
        for i,(field,color) in enumerate([('best_contribution','#3987ba'),('background_contribution','#df8f39'),('contrast','#568f58')]):
            s = stats[metric][field]
            ax.plot([i,i], s['ci95'], color=color, lw=3)
            ax.scatter([i], [s['mean']], color=color, s=45, zorder=3)
        ax.axhline(0, color='gray', lw=1)
        ax.set_xticks([0,1,2], ['Best match\nchange','Minus background\nchange','Contrast\nchange'])
        ax.set(ylabel='Cloud TTS minus Natural', title='Source bootstrap 95% descriptive intervals')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same query support, 20 sources / 60 pairs; model-specific units\nFull-context features; derived lag contrasts are not official benchmarks', fontsize=13)
    fig.savefig(root/'common_curve_decomposition.png', dpi=170)
    fig.savefig(root/'common_curve_decomposition.pdf')
    plt.close(fig)


def support_diagnostic(root):
    """Posthoc support checks prompted by the AVS zero-offset sign change."""
    inputs=read(root/'inputs.json')
    media={}
    modes=dict(sync=('sync','distance'),avs=('avs','cosine'),co=('co','relu_cosine'))
    for row in inputs:
        with np.load(root/'items'/f"{row['id']}.npz",allow_pickle=False) as z:
            cell={}
            for metric,(prefix,kind) in modes.items():
                q=np.arange(15,row['sync_windows']-15)
                assert len(q)==row['valid_query_count']
                ep=endpoint(matrix(z[prefix+'_audio'],z[prefix+'_video'],q,kind))
                cell.setdefault('all_valid_queries_unequal_counts',{})[metric]=ep
                curve=z[metric+'_matrix'].mean(0)[5:26]
                cell.setdefault('pm400_same_primary_queries',{})[metric]=dict(
                    best=float(curve.max()),background=float(np.median(curve)),zero=float(curve[10]),
                    contrast=float(curve.max()-np.median(curve)))
                if metric!='sync':
                    a,v=z[prefix+'_audio'].astype('float64'),z[prefix+'_video'].astype('float64')
                    frame=np.sum(a*v,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(v,axis=1))
                    if metric=='co':frame=np.maximum(frame,0)
                    cell.setdefault('original_cached_full_frame_zero',{})[metric]={'zero':float(frame.mean())}
                    cell.setdefault('primary_queries_center_frame_zero',{})[metric]={'zero':float(frame[z['queries']+2].mean())}
            media[row['id']]=cell
    n={r['sample_id']:r['id'] for r in inputs if r['arm']=='N'}
    output={}
    for variant in next(iter(media.values())):
        output[variant]={}
        for metric in next(iter(media.values()))[variant]:
            fields=['zero'] if variant.endswith('_zero') else ['best','background','contrast','zero']
            output[variant][metric]={}
            for field in fields:
                deltas=[dict(source=r['sample_id'],value=media[r['id']][variant][metric][field]-
                             media[n[r['sample_id']]][variant][metric][field]) for r in inputs if r['arm']=='C']
                sources=sorted(n)
                values=np.array([np.mean([d['value'] for d in deltas if d['source']==s]) for s in sources])
                index=np.random.default_rng(20261028).integers(0,len(values),(10000,len(values)))
                output[variant][metric][field]=dict(mean=float(values.mean()),
                    ci95=np.quantile(values[index].mean(1),[.025,.975]).tolist(),
                    tts_higher=sum(d['value']>0 for d in deltas),natural_higher=sum(d['value']<0 for d in deltas))
    main=read(root/'summary.json')['all20']['metrics']
    component_comparison={}
    for metric in ['sync','sync_unit','avs','co']:
        values=np.array([-d['background']-d['best'] for d in main[metric]['source_means'].values()])
        index=np.random.default_rng(20261028).integers(0,len(values),(10000,len(values)))
        component_comparison[metric]=dict(background_contribution_minus_best_contribution=float(values.mean()),
            ci95=np.quantile(values[index].mean(1),[.025,.975]).tolist())
    items=read(root/'items.json')
    peak_counts={m:{a:dict(Counter(str(x['endpoints'][m]['peak_lag_frames']) for x in items if x['arm']==a))
                      for a in ['N','C']} for m in ['sync','sync_unit','avs','co']}
    result=dict(status='COMPLETE_POSTHOC_SUPPORT_DIAGNOSTIC',posthoc=True,
                reason='AVS primary zero-offset direction differs from historical full-frame score; isolate time-support/aggregation and lag-grid sensitivity',
                formal_endpoints_unchanged=True,new_gpu_forwards=0,results=output,
                component_comparison=component_comparison,peak_lag_counts=peak_counts,
                source_sha256=digest(__file__))
    save(root/'posthoc_support_diagnostic.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--support-only', action='store_true')
    args=p.parse_args()
    if args.support_only:
        support_diagnostic(args.run)
    else:
        audit(args.run)
        if args.plot:
            plot(args.run)
