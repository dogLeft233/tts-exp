"""Independent CPU analysis of the frozen EARS40 nine-scorer experiment.

Does not import producers. Missing models cannot yield a complete report.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SEED = 20261029
METRICS = {'SyncNet': ('syncnet', 'C'), 'CoGenAV': ('cogenav', 'eq1'),
           'PEAVS': ('peavs', 'score'), 'SyncReward': ('syncreward', 'score'),
           'AVS_u': ('avs_u', 'frame_mean'), 'MTDVocaLiST': ('mtd', 'metrics.K5'),
           'VocaLiST': ('vocalist', 'metrics.K5'),
           'Synchformer': ('synchformer', 'output.p_zero'),
           'SparseSync': ('sparsesync', 'output.p_zero')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    if not np.isfinite(ps).all() or np.any((ps < 0) | (ps > 1)):
        raise ValueError('Invalid p values')
    order = np.argsort(ps, kind='stable')
    adjusted = np.minimum(1, np.maximum.accumulate(ps[order] * np.arange(len(ps), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def statistics(n, t, seed=SEED, bootstraps=10000, permutations=100000):
    n, t = np.asarray(n, dtype=float), np.asarray(t, dtype=float)
    if n.shape != t.shape or n.ndim != 1 or not len(n):
        raise ValueError('Matched nonempty vectors required')
    if not np.isfinite(n).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite observations')
    d = t - n
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(bootstraps, len(d)))].mean(1)
    ci = np.quantile(means, [.025, .975])
    # A conditional sign-flip sensitivity assumes exchangeability/symmetry of
    # paired differences; this is not randomized allocation of TTS treatment.
    observed = abs(d.mean())
    exceed = 0
    for start in range(0, permutations, 2000):
        signs = rng.integers(0, 2, size=(min(2000, permutations-start), len(d))) * 2 - 1
        simulated = np.abs((signs*d).mean(1))
        exceed += int(np.sum(simulated >= observed - 1e-15))
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.
    return dict(speakers=len(d), N_mean=float(n.mean()), T_mean=float(t.mean()),
                mean_difference=float(d.mean()), ci95=ci.tolist(),
                paired_dz=float(d.mean()/sd) if sd > 0 else None,
                T_wins=int(np.sum(d > 0)), N_wins=int(np.sum(d < 0)), ties=int(np.sum(d == 0)),
                signflip_p=(exceed+1)/(permutations+1), bootstrap_draws=bootstraps,
                signflip_draws=permutations)


def field(record, key):
    for part in key.split('.'):
        record = record[part]
    return float(record)


def scorer_rows(run, folder):
    obj = read(run/folder/'scores.json')
    return list(obj.values()) if isinstance(obj, dict) else obj


def metric_pairs(rows, key, speakers):
    lookup = {}
    for r in rows:
        source = r.get('source', r.get('sample_id'))
        if 'id' in r and r['id'] != f"{source}_{r['arm']}_0":
            raise ValueError('Scorer identity does not match frozen source/arm')
        pairkey = (source, r['arm'])
        if pairkey in lookup:
            raise ValueError('Repeated speaker/arm: '+str(pairkey))
        lookup[pairkey] = r
    expected = {(s, a) for s in speakers for a in ('N', 'T')}
    if set(lookup) != expected:
        raise ValueError('Incomplete frozen cohort: '+str(set(lookup) ^ expected))
    return [dict(speaker=s, N=field(lookup[(s,'N')],key), T=field(lookup[(s,'T')],key)) for s in speakers]


def k_accuracy(z, k=5):
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 31 or not len(z):
        raise ValueError('Expected query by 31 retrieval scores')
    half = (k-5)//2
    means = np.stack([z[max(0, i-half):min(len(z),i+half+1)].mean(0) for i in range(len(z))])
    return float(np.mean(np.abs(means.argmax(1)-15) <= 1))


def cosine_frames(a, v, rectify=False):
    a, v = np.asarray(a, dtype=float), np.asarray(v, dtype=float)
    if a.shape != v.shape or a.ndim != 2:
        raise ValueError('Matching frame embeddings required')
    den = np.linalg.norm(a,axis=1)*np.linalg.norm(v,axis=1)
    if np.any(den <= 0):
        raise ValueError('Zero norm')
    c = (a*v).sum(1)/den
    return np.maximum(0,c) if rectify else c


def raw_audit(run, models=None):
    results = {}
    inputs={r['id']:r for r in read(run/'inputs.json')}
    for metric, (folder, key) in METRICS.items():
        if models is not None and metric not in models:continue
        maximum, tested, secondary_error = 0., 0, 0.
        for r in scorer_rows(run, folder):
            identity = r['id']
            if folder == 'syncnet':
                z=np.load(run/'syncnet/matrices'/(identity+'.npy'))
                curve=np.asarray(z,dtype=np.float32).T.mean(1)
                recomputed=float(np.partition(curve,15)[15]-curve.min())
            elif folder in ('mtd','vocalist'):
                z=np.load(run/folder/'matrices'/(identity+'_0.npy'))
                recomputed=k_accuracy(z)
                secondary_error=max(secondary_error,*(abs(k_accuracy(z,k)-r['metrics']['K'+str(k)]) for k in (7,9,11,13,15)))
            elif folder in ('cogenav','avs_u'):
                p=run/folder/'embeddings'/(identity+'.npz')
                if sha(p) != r['embedding_sha256']:
                    raise ValueError('Embedding SHA changed: '+str(p))
                with np.load(p) as data:
                    for stream in ('audio','video'):
                        if str(data['input_'+stream+'_sha256']) != inputs[identity][stream+'_sha256']:
                            raise ValueError('Embedding input binding mismatch: '+identity)
                    recomputed=float(cosine_frames(data['audio'],data['video'],folder=='cogenav').mean())
            elif folder in ('synchformer','sparsesync'):
                z=np.asarray(r['output']['logits'],dtype=float)
                probabilities=np.exp(z-z.max());probabilities/=probabilities.sum()
                if z.shape != (21,) or not np.isfinite(z).all():
                    raise ValueError('Invalid class logits')
                recomputed=float(probabilities[10])
                grid=np.linspace(-2,2,21)
                secondary_error=max(secondary_error,
                    abs(float(-np.sum(probabilities*np.abs(grid)))-r['output']['negative_expected_abs_offset']),
                    abs(float(grid[z.argmax()])-r['output']['predicted_offset']))
            elif folder=='peavs':
                raw=read(run/'peavs/native.raw.json')
                source={Path(a['filename']).stem:float(a['score']) for a in raw['scores']}
                recomputed=source[identity]
            elif folder=='syncreward':
                source={Path(a['video_path']).stem:float(a['reward']) for a in
                        [json.loads(line) for line in (run/'syncreward/all_scores.raw.jsonl').read_text().splitlines()]}
                recomputed=source[identity]
            else:
                continue  # scalar regressors are audited against retained raw outputs separately
            error=abs(recomputed-field(r,key));maximum=max(maximum,error);tested+=1
        tolerance = 2e-5 if folder=='syncnet' else 2e-6
        if maximum > tolerance or secondary_error > tolerance:
            raise ValueError(f'{metric} raw error {maximum} exceeds {tolerance}')
        results[metric]=dict(checked_media=tested,max_error=maximum,tolerance=tolerance,
                             secondary_max_error=secondary_error,
                             status='PASS' if tested==80 else 'INCOMPLETE',
                             scope='raw retained model output binding, not model reexecution' if folder in ('peavs','syncreward') else 'raw endpoint numerical reconstruction')
    return results


def diagnostics(run, speakers, models=None):
    results = {}
    for metric,(folder,_) in METRICS.items():
        if models is not None and metric not in models:continue
        filename='known_shift_controls.json' if folder in ('cogenav','avs_u','synchformer','sparsesync') else 'control_summary.json'
        control=read(run/folder/filename)
        count=control.get('total',control.get('tested'))
        lower=control.get('lower',control.get('decreased',control.get('k5_decreased')))
        if count != 80:
            raise ValueError(f'{metric}: expected 80 controls, got {count}')
        item=dict(primary_lower=int(lower),tested=count,primary_response_pass=lower>=64,
                  source_receipt=filename,source_sha256=sha(run/folder/filename))
        if folder=='syncnet':
            data=read(run/folder/'controls.json')
            cs=[c for r in data.values() for c in r['outputs'].values()]
            item.update(relative_offset_recovered=sum(c['relative_peak_error_frames']<=1 for c in cs),
                        relative_offset_max_error_frames=max(c['relative_peak_error_frames'] for c in cs),
                        zero_replay_max_error=max(r['zero_replay_max_error'] for r in data.values()),
                        interpretation='Sync-C searches best lag; confidence-decrease and recovery of imposed lag are separate tests.')
        if folder in ('mtd','vocalist'):
            item.update(relative_offset_recovered=control['relative_offset_recovered'],
                        relative_offset_max_error_frames=control['relative_offset_max_error_frames'])
        if folder=='cogenav':
            item.update(relative_offset_recovered=control['peak_within_one_frame'],
                        relative_offset_max_error_frames=max(abs(r['peak_movement_error_frames']) for r in control['rows']))
        rows=scorer_rows(run,folder)
        if folder in ('synchformer','sparsesync'):
            item.update(relative_offset_recovered=sum(r['relative_offset_error']<=.04 for r in control['rows']),
                        relative_offset_max_error_seconds=max(r['relative_offset_error'] for r in control['rows']))
            item['ceiling']=dict(p0_over_9995=sum(r['output']['p_zero']>.9995 for r in rows),
                p0_exact1=sum(r['output']['p_zero']==1 for r in rows),
                zero_class_argmax=sum(int(np.argmax(r['output']['logits']))==10 for r in rows),media=80)
            item['precision_sensitivity']=classifier_precision(rows)
        if folder in ('mtd','vocalist'):
            item['floor']=dict(k5_exact_zero=sum(r['metrics']['K5']==0 for r in rows),media=80)
            item['native_peak_histogram']={str(v):sum(r['global_offset_frames']==v for r in rows)
                 for v in sorted({r['global_offset_frames'] for r in rows})}
        results[metric]=item
    return results


def classifier_precision(rows):
    lookup={(r['source'],r['arm']):r['output'] for r in rows}
    sources=sorted({s for s,_ in lookup})
    deltas32=[];deltas64=[];changed=[]
    for source in sources:
        probabilities=[]
        for arm in ('N','T'):
            z=np.asarray(lookup[(source,arm)]['logits'],dtype=np.float64)
            p=np.exp(z-z.max());p/=p.sum();probabilities.append(float(p[10]))
        d32=lookup[(source,'T')]['p_zero']-lookup[(source,'N')]['p_zero']
        d64=probabilities[1]-probabilities[0]
        deltas32.append(d32);deltas64.append(d64)
        if np.sign(d32)!=np.sign(d64):changed.append(source)
    return dict(label='post_forward_saved_logits_FP64_softmax_only_no_endpoint_replacement',
        pairs=len(sources),sign_changes=changed,
        fp32_T_wins=sum(d>0 for d in deltas32),fp64_T_wins=sum(d>0 for d in deltas64),
        min_abs_fp32_pair_difference=min(abs(d) for d in deltas32),
        min_abs_fp64_pair_difference=min(abs(d) for d in deltas64))


def secondary_metrics(run, speakers):
    metrics={}
    for model,folder in [('MTDVocaLiST','mtd'),('VocaLiST','vocalist')]:
        for k in (7,9,11,13,15):
            pairs=metric_pairs(scorer_rows(run,folder),'metrics.K'+str(k),speakers)
            metrics[model+'_K'+str(k)]=statistics([p['N'] for p in pairs],[p['T'] for p in pairs])
    for model,folder in [('Synchformer','synchformer'),('SparseSync','sparsesync')]:
        pairs=metric_pairs(scorer_rows(run,folder),'output.negative_expected_abs_offset',speakers)
        metrics[model+'_negative_expected_class_offset']=statistics([p['N'] for p in pairs],[p['T'] for p in pairs])
    batch1=read(run/'peavs/native_batch1.json')['scores']
    rows=[dict(id=r['id'],source=r['source'],arm=r['arm'],score=batch1[r['id']]) for r in scorer_rows(run,'peavs')]
    pairs=metric_pairs(rows,'score',speakers)
    metrics['PEAVS_batch1_sensitivity']=statistics([p['N'] for p in pairs],[p['T'] for p in pairs])
    return dict(label='SECONDARY_DESCRIPTIVE_NOT_HOLM9_PRIMARY_FAMILY',metrics=metrics)


def audit_controls(run, models=None):
    audits={}
    for metric,(folder,_) in METRICS.items():
        if models is not None and metric not in models:continue
        diffs=[];declines=0;seen=set()
        if folder in ('cogenav','avs_u'):
            rows=scorer_rows(run,folder)
            for r in rows:
                if r['arm']!='N':continue
                with np.load(run/folder/'embeddings'/(r['id']+'.npz')) as z:
                    v=z['video'];baseline=float(cosine_frames(z['audio'],v,folder=='cogenav').mean())
                suffix='_rawshifts.npz' if folder=='cogenav' else '_controls.npz'
                with np.load(run/folder/'embeddings'/(r['id']+suffix)) as shifts:
                    for c in r['raw_shift_controls' if folder=='cogenav' else 'controls']:
                        amount=c['shift_samples']
                        a=shifts[str(c['shift_frames']) if folder=='cogenav' else str(amount)]
                        score=float(cosine_frames(a,v,folder=='cogenav').mean())
                        expected=c['eq1'] if folder=='cogenav' else c['frame_mean']
                        diffs.append(abs(score-expected));declines+=score<baseline;seen.add((r['id'],amount))
        elif folder in ('syncnet','mtd','vocalist'):
            data=read(run/folder/'controls.json')
            for identity,r in data.items():
                for amount,c in r['outputs' if folder=='syncnet' else 'shifts'].items():
                    z=np.load(run/folder/'matrices'/(identity+'_'+amount+'.npy'))
                    if folder=='syncnet':
                        curve=np.asarray(z,dtype=np.float32).T.mean(1)
                        value=float(np.partition(curve,15)[15]-curve.min())
                        baseline=r['baseline'];expected=c['C']
                    else:
                        value=k_accuracy(z);baseline=c['check']['native_k5'];expected=c['metrics']['K5']
                    diffs.append(abs(value-expected));declines+=value<baseline;seen.add((identity,int(amount)))
        elif folder in ('synchformer','sparsesync'):
            data=read(run/folder/'known_shift_controls.json')['rows']
            for r in data:
                values=[]
                for record in (r['native'],r['shifted']):
                    z=np.asarray(record['logits'],dtype=float);p=np.exp(z-z.max());p/=p.sum()
                    values.append(float(p[10]));diffs.append(abs(float(p[10])-record['p_zero']))
                # Official FP32 probabilities own the strict-sign control endpoint;
                # double precision is a reconstruction diagnostic, not replacement.
                declines+=r['shifted']['p_zero']<r['native']['p_zero']
                seen.add((r['id'],r['shift_samples']))
        else:
            data=read(run/folder/'controls.json')
            if folder=='syncreward':
                raw={Path(a['video_path']).stem:float(a['reward']) for a in
                    [json.loads(line) for line in (run/folder/'all_scores.raw.jsonl').read_text().splitlines()]}
            for identity,r in data.items():
                if folder=='peavs':
                    batch=read(run/folder/(r['batch_label']+'.raw.json'))
                    raw={Path(a['filename']).stem:float(a['score']) for a in batch['scores']}
                diffs.append(abs(raw[identity]-r['native_baseline']))
                for amount,c in r['shifts'].items():
                    label='minus400ms' if int(amount)<0 else 'plus400ms'
                    value=raw[identity+'_'+label]
                    diffs.append(abs(value-c['score']));declines+=value<raw[identity]
                    seen.add((identity,int(amount)))
        if len(seen)!=80 or any(a not in (-6400,6400) for _,a in seen):
            raise ValueError(f'{metric}: incomplete unique physical controls')
        maximum=max(diffs)
        if maximum>2e-5:raise ValueError(f'{metric}: control reconstruction {maximum}')
        audits[metric]=dict(status='PASS',unique_controls=len(seen),max_scalar_error=maximum,
                            reconstructed_primary_decreases=int(declines))
    return audits


def central_sensitivity(run, speakers, models=None):
    """Nominal central 125 queries; original encoder context is retained."""
    answer = {}
    for metric, folder in [('CoGenAV','cogenav'),('AVS_u','avs_u'),('SyncNet','syncnet'),
                           ('MTDVocaLiST','mtd'),('VocaLiST','vocalist')]:
        if models is not None and metric not in models:continue
        values = []
        for r in scorer_rows(run,folder):
            sid=r.get('source',r.get('sample_id'));identity=r['id']
            if folder in ('cogenav','avs_u'):
                with np.load(run/folder/'embeddings'/(identity+'.npz')) as z:
                    c=cosine_frames(z['audio'],z['video'],folder=='cogenav')
                start=(len(c)-125)//2
                if start < 0: raise ValueError('Insufficient central query support')
                value=float(c[start:start+125].mean())
            elif folder=='syncnet':
                z=np.load(run/folder/'matrices'/(identity+'.npy'))
                start=(len(z)-125)//2
                if start < 0: raise ValueError('Insufficient central query support')
                c=np.asarray(z[start:start+125],dtype=np.float32).T.mean(1)
                value=float(np.partition(c,15)[15]-c.min())
            else:
                z=np.load(run/folder/'matrices'/(identity+'_0.npy'))
                start=(len(z)-125)//2
                if start < 0: raise ValueError('Insufficient central query support')
                value=k_accuracy(z[start:start+125])
            values.append(dict(source=sid,arm=r['arm'],value=value))
        pairs=metric_pairs(values,'value',speakers)
        answer[metric]=dict(statistics=statistics([p['N'] for p in pairs],[p['T'] for p in pairs]),
                           pairs=pairs)
    return dict(label='POST_ENCODING_CENTRAL_125_QUERIES_DESCRIPTIVE',
        limitation='125 nominal 25Hz query centers; receptive fields and lag candidate context can extend beyond these centers. Native full-context embeddings/official scoring crops retained, no new five-second media benchmark or matched phonetic events. No Holm9 claims from this secondary view.',metrics=answer)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--raw-audit',action='store_true')
    parser.add_argument('--central',action='store_true')
    args=parser.parse_args();run=args.run.resolve();out=args.out or run/'parent_analysis'
    protocol=read(run/'protocol.json')
    inputs=read(run/'audio_inputs.json')
    speakers=sorted({r['sample_id'] for r in inputs})
    assert len(speakers)==40 and len(inputs)==80
    sex={r['sample_id']:r['gender'] for r in inputs}
    summaries,pairsets={},{}
    for metric,(folder,key) in METRICS.items():
        pairs=metric_pairs(scorer_rows(run,folder),key,speakers)
        stats=statistics([p['N'] for p in pairs],[p['T'] for p in pairs])
        stats['unit']='proportion' if metric in ('MTDVocaLiST','VocaLiST','Synchformer','SparseSync') else 'native scorer units'
        summaries[metric]=stats;pairsets[metric]=pairs
    adjusted=holm([summaries[m]['signflip_p'] for m in METRICS])
    reference=np.array([p['T']-p['N'] for p in pairsets['SyncNet']])
    for i,metric in enumerate(METRICS):
        summaries[metric]['holm9_p']=float(adjusted[i])
        summaries[metric]['conflicts_vs_SyncC']=int(np.sum(reference*np.array([p['T']-p['N'] for p in pairsets[metric]]) < 0))
    groups={}
    for group in sorted(set(sex.values())):
        groups[group]={}
        for metric,pairs in pairsets.items():
            selected=[p for p in pairs if sex[p['speaker']]==group]
            groups[group][metric]=statistics([p['N'] for p in selected],[p['T'] for p in selected])
    summary=dict(status='NINE_PRIMARY_PAIRED_STATS_COMPLETE',speakers=40,media=80,
        protocol_sha256=sha(run/'protocol.json'),analysis_source_sha256=sha(__file__),
        analysis_environment=dict(python=sys.version,numpy=np.__version__,seed=SEED),
        metrics=summaries,gender_descriptive=groups,diagnostics=diagnostics(run,speakers),
        inference='Paired bootstrap conditions on speakers from the acquired English-native EARS pool and one shared text. Sign-flip p values assume paired exchangeability/symmetry, not randomized treatment. Marginal CI95 and Holm9 p use different uncertainty procedures; an unadjusted CI is not simultaneous evidence.',
        scientific_scope=protocol['interpretation'])
    write(out/'summary.json',summary);write(out/'paired_scores.json',pairsets)
    with (out/'paired_scores.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','speaker','gender','unit','N','T','delta','delta_sync_c'])
        writer.writeheader()
        for metric,pairs in pairsets.items():
            for i,pair in enumerate(pairs):
                writer.writerow(dict(metric=metric,speaker=pair['speaker'],gender=sex[pair['speaker']],
                    unit=summaries[metric]['unit'],N=pair['N'],T=pair['T'],delta=pair['T']-pair['N'],delta_sync_c=float(reference[i])))
    write(out/'secondary_metrics.json',secondary_metrics(run,speakers))
    if args.raw_audit:
        write(out/'raw_endpoint_audit.json',raw_audit(run))
        write(out/'raw_control_audit.json',audit_controls(run))
    if args.central:write(out/'central_support_sensitivity.json',central_sensitivity(run,speakers))
    write(out/'analysis_status.json',dict(status='COMPLETE',stats=True,
        raw_reconstruction_requested_and_passed=args.raw_audit,
        central_sensitivity_requested_and_complete=args.central,
        metrics=9,speakers=40,analysis_source_sha256=sha(__file__)))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
