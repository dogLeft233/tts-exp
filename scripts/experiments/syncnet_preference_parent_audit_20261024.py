"""Independent numerical audit; does not import the experiment runner."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

MODELS = ('M0', 'M1', 'M2', 'M3')
PRIMARY = ('N-L06', 'N-L17', 'L17-L06')
FEATURES = ('duration_s', 'quiet_fraction', 'envelope_cv', 'spectral_flux', 'centroid_hz', 'high_band_fraction')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def aggregate(oof, model, threshold=0., families=PRIMARY):
    cells = defaultdict(list)
    for p in oof:
        t, v = p['target'], p['predictions'][model]
        if p['family'] not in families or abs(t) < threshold:
            continue
        a = None if t == 0 else .5 if abs(v) <= 1e-9 else float(t*v > 0)
        cells[p['source'], p['family']].append((abs(t-v), a))
    by_source = defaultdict(list)
    for (source, family), values in cells.items():
        accs = [v[1] for v in values if v[1] is not None]
        by_source[source].append((np.mean([v[0] for v in values]), np.mean(accs) if accs else None))
    result = {s: {'mae': float(np.mean([v[0] for v in vs])), 'accuracy': float(np.mean([v[1] for v in vs if v[1] is not None]))} for s,vs in by_source.items()}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    run = args.run
    manifest = read(run/'input_manifest.json')
    rows = manifest['rows']
    assert Counter(r['arm'] for r in rows) == {'N': 20, 'L06': 60, 'L17': 60}
    assert len({r['sample_id'] for r in rows}) == 20
    assert len({r['speaker_id'] for r in rows}) == 20
    assert digest(run/'protocol.json') == manifest['protocol_sha256']
    hz = np.load(run/'hubert_pooled.npz', allow_pickle=False)
    assert list(hz['ids']) == [r['id'] for r in rows]
    hidden = hz['pooled']
    assert hidden.shape == (140, 1536) and np.isfinite(hidden).all()
    frame_meta = read(run/'encoder_metadata.json')['frames']
    assert all(x['valid_hidden_frames'] == (x['input_samples']-400)//320+1 for x in frame_meta)
    bridge = 0.
    if args.check_inputs:
        for r in rows:
            for field in ('audio', 'video', 'score', 'activesd'):
                assert digest(r[field+'_path']) == r[field+'_sha256']
            with open(r['activesd_path'], 'rb') as f:
                matrix = np.asarray(pickle.load(f)[0])
            assert matrix.dtype == np.float32 and matrix.shape == (r['score_windows'],31)
            curve = np.mean(matrix, axis=0, dtype=np.float32)
            c = float(np.median(curve)-np.min(curve))
            assert c == r['score_float32']
            bridge = max(bridge, abs(c-r['score_printed']))
        assert bridge <= .000501
    arm = np.array([[float(r['arm']==a) for a in ('N','L06','L17')] for r in rows])
    ac = np.array([[r['features'][k] for k in FEATURES] for r in rows])
    support = np.log1p([r['score_windows'] for r in rows])[:,None]
    max_pred_error = max_fit_error = max_metric_error = 0.
    summaries = {}
    for scope, expected in (('full',300), ('qualified',286)):
        oof = read(run/f'{scope}_oof.json')
        result = read(run/f'{scope}_results.json')
        assert len(oof) == expected
        assert len({(p['a_id'],p['b_id'],p['family']) for p in oof}) == expected
        for p in oof:
            a,b = rows[p['a']], rows[p['b']]
            assert a['id']==p['a_id'] and b['id']==p['b_id']
            assert a['sample_id']==b['sample_id']==p['source']
            assert a['group_id']==b['group_id']==p['group']
            assert p['target'] == a['score_float32']-b['score_float32']
            if scope=='qualified' and a['speaker_id'] in ('S0011','S0016'):
                assert a['draw'] != 3 and b['draw'] != 3
        for fold in result['folds']:
            group = fold['group']
            train = [p for p in oof if p['group'] != group]
            test = [p for p in oof if p['group'] == group]
            params = np.load(run/'folds'/f'{scope}_{group}.npz', allow_pickle=False)
            tr = params['train_indices']
            expected_train = sorted({p[k] for p in train for k in ('a','b')})
            assert tr.tolist()==expected_train
            assert all(rows[i]['group_id'] != group for i in tr)
            assert set(fold['train_sources']).isdisjoint(fold['test_sources'])
            # Match float32 arithmetic boundaries of the exported transformations.
            h = hidden.copy()
            h -= params['hidden_scaler_mean']
            h /= params['hidden_scaler_scale']
            z = h @ params['pca_components'].T
            z -= params['pca_mean'] @ params['pca_components'].T
            raw = {'M0':arm, 'M1':np.c_[arm,ac,support], 'M2':np.c_[arm,z,np.log(ac[:,0]),support], 'M3':np.c_[arm,ac,support,z]}
            counts = Counter((p['source'],p['family']) for p in train)
            w = np.array([1/counts[p['source'],p['family']] for p in train])
            y = np.array([p['target'] for p in train])
            for model in MODELS:
                features = (raw[model]-params[model+'_scaler_mean'])/params[model+'_scaler_scale']
                x = np.array([features[p['a']]-features[p['b']] for p in train])
                alpha = float(params[model+'_alpha'])
                coef = np.linalg.solve(x.T@(w[:,None]*x)+alpha*np.eye(x.shape[1]), x.T@(w*y))
                max_fit_error = max(max_fit_error,float(np.max(abs(coef-params[model+'_coef']))))
                for p in test:
                    pred = float((features[p['a']]-features[p['b']])@coef)
                    max_pred_error = max(max_pred_error,abs(pred-p['predictions'][model]))
        summaries[scope] = {}
        for model in MODELS:
            for label, threshold in (('primary',0.),('threshold_.1',.1),('threshold_.3',.3)):
                by = aggregate(oof,model,threshold)
                expected_summary = result['summaries'][model][label]
                for key in ('mae','accuracy'):
                    val = float(np.mean([v[key] for v in by.values()]))
                    max_metric_error = max(max_metric_error,abs(val-expected_summary[key]))
                for s, v in by.items():
                    for key in ('mae','accuracy'):
                        max_metric_error=max(max_metric_error,abs(v[key]-expected_summary['sources'][s][key]))
            summaries[scope][model] = {k:result['summaries'][model]['primary'][k] for k in ('mae','accuracy')}
        a,b = aggregate(oof,'M1'), aggregate(oof,'M3')
        d=np.array([a[s]['mae']-b[s]['mae'] for s in sorted(a)])
        rng=np.random.default_rng(20261024)
        boot=np.mean(d[rng.integers(len(d),size=(10000,len(d)))],axis=1)
        assert np.max(abs(np.quantile(boot,[.025,.975])-result['descriptive_CI95']))<1e-12
    assert max_pred_error<1e-4 and max_fit_error<1e-4 and max_metric_error<1e-12
    receipt={'status':'PASS','independent_runner_import':False,'input_hashes_and_distance_arrays_checked':args.check_inputs,'rows':140,'outer_saved_refits':160,'bridge_max_error':bridge,'max_closed_form_coef_error':max_fit_error,'max_oof_replay_error':max_pred_error,'max_metric_error':max_metric_error,'summaries':summaries,'script_sha256':digest(__file__),'limitations':['No independent new-cohort confirmation','No human audiovisual quality assessment','Source bootstrap is descriptive with shared training folds','No independent full encoder replay in this script']}
    target=run/'parent_independent_audit.json'
    assert not target.exists()
    target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
