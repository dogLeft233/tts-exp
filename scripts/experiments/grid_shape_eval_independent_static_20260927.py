"""CPU synthetic audit of the sealed evaluator; never imports its I/O or reads media."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/grid_shape_eval_audit_20260927'
WORKER = ROOT / 'scripts/experiments/grid_shape_correlation_evaluation_20260927.py'
PRODUCER = ROOT / 'runs/grid_shape_correlation_evaluation_20260927'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
source = WORKER.read_text()
tree = ast.parse(source)
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
views = ['BASE', 'TX_-8', 'TX_8', 'TY_-8', 'TY_8', 'ROT_-5', 'ROT_5', 'SCALE_0.9', 'SCALE_1.1']


def metric(y, x):
    # Separate centered-vector definition, not the producer helper.
    yc, xc = y - np.mean(y), x - np.mean(x)
    sy, sx = np.linalg.norm(yc) / np.sqrt(len(y)), np.linalg.norm(xc) / np.sqrt(len(x))
    return {'r': float(np.dot(yc, xc) / np.linalg.norm(yc) / np.linalg.norm(xc))
            if min(sy, sx) > 1e-6 else None,
            'centered_rmse': float(np.linalg.norm(xc-yc) / np.sqrt(len(y)))}


ns = {'np': np, 'math': math, 'T': np.arange(14, 55), 'LAGS': [0, -3],
      'VIEWS': views, 'DOMAINS': ['REAL', 'RAW', 'FIXED'],
      'geom': SimpleNamespace(metric=metric, qc=lambda z, real=True: {'passed': z[2]})}
ast_stats = ast.Module(body=[copy.deepcopy(funcs['stats'])], type_ignores=[])
exec(compile(ast.fix_missing_locations(ast_stats), str(WORKER), 'exec'), ns)
analysis = funcs['analyze']
rowloop = next(n for n in analysis.body if isinstance(n, ast.For)
               and isinstance(n.target, ast.Tuple) and ast.unparse(n.target) == '(j, r)')
start = next(i for i, n in enumerate(rowloop.body) if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == 't')
body = copy.deepcopy(rowloop.body[start:])
body = [n for n in body if not (isinstance(n, ast.Expr) and ast.unparse(n.value) == 'ref.close()')]
fn = ast.parse('def evaluate_row(data, j, r, records, deltas):\n pass').body[0]
fn.body = ast.parse("sid=r['id']").body + body
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(WORKER), 'exec'), ns)
decision_start = next(i for i, n in enumerate(analysis.body) if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == 'measurable')
decision_stop = next(i for i in range(decision_start, len(analysis.body))
                     if isinstance(analysis.body[i], ast.Expr))
dec = ast.parse('def decision(records, n, summary):\n pass').body[0]
dec.body = copy.deepcopy(analysis.body[decision_start:decision_stop]) + ast.parse('return conclusion, measurable, complete, gate').body
exec(compile(ast.fix_missing_locations(ast.Module(body=[dec], type_ignores=[])), str(WORKER), 'exec'), ns)


def fixture():
    t = np.arange(75)
    y = .06 + .03*np.sin(t*.41) + .01*np.cos(t*.13)
    return {(d, v): (y.copy(), np.ones(75, bool), True)
            for d in ns['DOMAINS'] for v in views}


def row(data):
    records, deltas = [], np.full((1, 2, 9, 9), np.nan)
    ns['evaluate_row'](data, 0, {'id': 'synthetic'}, records, deltas)
    return records[0], deltas


checks = {}
r, delta = row(fixture())
assert r['indices'] == list(range(14, 55)) and r['measurement_pass'] and r['all_required_computable']
assert np.max(abs(delta)) == 0
checks['all_41_frames_and_all_324_correlations'] = True
data = fixture(); data['FIXED', 'SCALE_1.1'][1][20] = False
r, _ = row(data)
assert r['indices'] == [i for i in range(14, 55) if i not in (20, 23)]
checks['single_view_invalid_removes_both_lag_indices_globally'] = True
data = fixture(); data['RAW', 'BASE'][1][14:35] = False
r, delta = row(data)
assert len(r['indices']) < 33 and not r['all_required_computable'] and not r['measurement_pass']
assert np.isnan(delta).all()
checks['insufficient_support_retained_as_uncomputable'] = True
data = fixture(); data['FIXED', 'BASE'][0][:] = .04
r, delta = row(data)
assert not r['all_required_computable'] and not r['measurement_pass']
assert np.isnan(delta[0, :, :, 0]).all() and np.isfinite(delta[0, :, :, 1:]).all()
checks['constant_output_blocks_required_correlations_without_source_drop'] = True
data = fixture(); data['FIXED', 'BASE'] = (*data['FIXED', 'BASE'][:2], False)
r, delta = row(data)
assert not r['measurement_pass'] and r['all_required_computable'] and np.isfinite(delta).all()
checks['failed_generated_QC_remains_in_effects'] = True

draws = np.random.default_rng(20260927).integers(0, 12, size=(100000, 12))
x = np.linspace(-.02, .2, 12)
s = ns['stats'](x, draws)
manual = sum(x[draws[:, k]] for k in range(12))/12
assert abs(s['mean'] - sum(x)/12) < 1e-15
assert np.max(abs(np.array(s['ci95'])-np.quantile(manual,[.025,.975]))) < 1e-15
bad = x.copy(); bad[-1] = np.nan
assert ns['stats'](bad, draws) == {'n':12, 'mean':None, 'ci95':None, 'ci99':None, 'computable':False}
checks['paired_bootstrap_and_NA_keep_full_denominator'] = True
for sign, expected in [(1,'ROBUST_IMPROVEMENT'),(-1,'ROBUST_DECLINE')]:
    summary={'0':{str(k):{'ci95':[sign*.1,sign*.1]} for k in range(81)}}
    records=[{'measurement_pass':i<9,'all_required_computable':True} for i in range(12)]
    assert ns['decision'](records,12,summary)[0] == expected
    records[8]['measurement_pass']=False
    assert ns['decision'](records,12,summary)[0] == 'NOT_CONFIRMED'
    records[8]['measurement_pass']=True; records[-1]['all_required_computable']=False
    assert ns['decision'](records,12,summary)[0] == 'NOT_CONFIRMED'
    records[-1]['all_required_computable']=True; summary['0']['80']['ci95']=[-.1,.1]
    assert ns['decision'](records,12,summary)[0] == 'NOT_CONFIRMED'
checks['81_conjunction_and_measurement_NA_blocks_both_directions'] = True
assert 47648//640-5 == 69 and 75-4 == 71
assert min(np.arange(20,49)-15) >= 0 and max(np.arange(20,49)+15) < 69
checks['SyncNet_69_joint_rows_29_queries_no_lag_padding'] = True

p=json.loads((PRODUCER/'protocol.json').read_text())
dependencies={f:sha(f)==h for f,h in p['dependencies'].items()}
assert all(dependencies.values())
assert len(p['rows']) == len({r['id'] for r in p['rows']}) == 16
assert all(r['split']=='eval' and r['pcm_samples']==47648 for r in p['rows'])
# Never hash/read paths in rows: only frozen metadata and implementation dependencies.
out={'status':'PASS','scope':'synthetic implementation only; no eval media, features, scores, model forward or GPU',
     'protocol_sha256':sha(PRODUCER/'protocol.json'),'worker_sha256':sha(WORKER),
     'auditor_sha256':sha(__file__),'checks':checks,'dependency_hashes_checked':len(dependencies),
     'synthetic_bootstrap_draws_sha256':hashlib.sha256(draws.tobytes()).hexdigest()}
OUT.mkdir(exist_ok=True)
target=OUT/('synthetic_'+out['protocol_sha256'][:8]+'.json')
with target.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
