"""Fixed 20-step ENVmatched controls; old 26cal only, frozen v1 remains immutable."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/'runs/tts_acoustic_modulation_calibration_20260926'
OUT = ROOT/'runs/tts_acoustic_modulation_calibration_20260926_v2'
# Reuse the immutable v1 acoustic definitions, not a mutable working-copy import.
spec = importlib.util.spec_from_file_location('frozen_acoustic_v1', V1/'code_snapshot/tts_acoustic_modulation_calibration.py')
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
WINDOW = np.hanning(513)[:-1]; WINDOW /= WINDOW.sum()
EPS = 1e-8
LOW, HIGH = 10**(-12/20), 10**(12/20)
CONDITIONS = ['raw','alpha08','identity','alpha12','ENVmatched_alpha08','ENVmatched_alpha12']


def envelope(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.convolve(np.pad(x*x, (256,255), mode='reflect'), WINDOW, mode='valid') + EPS**2)


def envelope_match(source, target):
    """Fixed damped amplitude updates; no division by source or early stopping."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.ndim != 1 or source.shape != target.shape or not len(source):
        raise ValueError('Expected equal-length nonempty mono source and target')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite input')
    gain = np.ones_like(source); u = source.copy()
    target_e = envelope(target); source_rms = v1.rms(source)
    trace = []
    for iteration in range(20):
        proposed = gain*np.sqrt(target_e/np.maximum(envelope(u), EPS))
        clipped = np.clip(proposed, LOW, HIGH)
        u = source*clipped
        norm = source_rms/v1.rms(u) if v1.rms(u) else 1.0
        gain = clipped*norm
        u = source*gain
        trace.append({'iteration':iteration+1, 'proposed_below_fraction':float(np.mean(proposed<LOW)),
                      'proposed_above_fraction':float(np.mean(proposed>HIGH)),
                      'clipped_boundary_fraction':float(np.mean((clipped==LOW)|(clipped==HIGH))),
                      'rms_normalization_gain':norm,
                      'post_normalization_outside_fraction':float(np.mean((gain<LOW)|(gain>HIGH)))})
    return u, gain, {'iterations':trace, 'final_gain_min':float(gain.min()), 'final_gain_max':float(gain.max()),
                     'final_gain_outside_bounds_fraction':float(np.mean((gain<LOW)|(gain>HIGH))),
                     'zero_samples_preserved':bool(np.all(u[source==0]==0)),
                     'restored_rms_relative_error':abs(v1.rms(u)-source_rms)/source_rms if source_rms else 0.0}


def shared_headroom(waveforms):
    peak = max(float(np.max(abs(x))) for conditions in waveforms.values() for x in conditions.values())
    return min(1.,.98/peak) if peak else 1.


def run(out=OUT):
    out = Path(out); p=json.loads((out/'protocol.json').read_text())
    assert v1.sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    assert v1.sha(V1/'artifact_hashes.json') == p['parent_artifact_manifest_sha256']
    assert p['conditions'] == CONDITIONS
    assert not (out/'summary.json').exists(), 'Preserve first outcome'
    archived=json.loads((V1/'artifact_hashes.json').read_text())
    for rel, hash_value in archived.items():
        assert v1.sha(V1/rel)==hash_value
    (out/'code_snapshot').mkdir(exist_ok=True)
    (out/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (out/'code_snapshot'/'frozen_v1_acoustics.py').write_bytes((V1/'code_snapshot/tts_acoustic_modulation_calibration.py').read_bytes())
    rows, manifest, traces, common_g = [], [], {}, {}
    for sid in p['cal_ids']:
        sources={r['arm']:r for r in p['sources'] if r['id']==sid}
        inputs, before, final_gains={}, {}, {}
        for arm in ['N','T']:
            source=sources[arm]; assert v1.sha(source['path'])==source['sha256']
            inputs[arm], sr=sf.read(source['path'],dtype='float64'); assert sr==16000 and inputs[arm].ndim==1
            before[arm]={}; final_gains[arm]={}
            for condition in CONDITIONS[:4]:
                path=V1/'pre_headroom'/sid/arm/f'{condition}.wav'
                before[arm][condition],sr=sf.read(path,dtype='float64'); assert sr==16000
            for condition in CONDITIONS[4:]:
                target_condition=condition.removeprefix('ENVmatched_')
                u,gain,trace=envelope_match(inputs[arm],before[arm][target_condition])
                before[arm][condition]=u; final_gains[arm][condition]=gain
                traces[f'{sid}/{arm}/{condition}']=trace
        g=shared_headroom(before); common_g[sid]=g
        for arm in ['N','T']:
            raw_values=None; condition_metrics={}
            for condition in CONDITIONS:
                item={'id':sid,'arm':arm,'condition':condition}
                for stage,factor in [('pre_headroom',1.0),('waveforms',g)]:
                    path=out/stage/sid/arm/f'{condition}.wav'; path.parent.mkdir(parents=True,exist_ok=True)
                    sf.write(path,before[arm][condition]*factor,16000,subtype='FLOAT')
                    item[stage]={'path':str(path),'sha256':v1.sha(path)}
                y,sr=sf.read(item['waveforms']['path'],dtype='float64')
                metrics,values=v1.analyze(y)
                values['sample_hann_envelope']=envelope(y)
                if condition=='raw': raw_values=values
                metrics.update(v1.couplings(values,raw_values))
                xrms=v1.rms(inputs[arm]); expected=xrms*g
                row={'id':sid,'arm':arm,'condition':condition,'speaker':sources[arm]['speaker'],
                     'source_rms':xrms,'non_silent':xrms>1e-8,'shared_headroom_gain':g,**metrics,
                     'saved_rms_relative_error':abs(v1.rms(y)-expected)/expected if expected else 0.,
                     'exact_length':len(y)==len(inputs[arm]),
                     'restored_rms_relative_error':abs(v1.rms(before[arm][condition])-xrms)/xrms if xrms else 0.}
                if condition.startswith('ENVmatched_'):
                    target_condition=condition.removeprefix('ENVmatched_')
                    target_path=out/'waveforms'/sid/arm/f'{target_condition}.wav'
                    target,sr=sf.read(target_path,dtype='float64')
                    row['target_envelope_db_rmse']=v1.rms(20*np.log10(envelope(y)/envelope(target)))
                    row['R_RMS_relative_change_from_identity']=row['residual_rms_db']/condition_metrics['identity']['residual_rms_db']-1
                    trace=traces[f'{sid}/{arm}/{condition}']
                    row.update({k:val for k,val in trace.items() if k!='iterations'})
                    values['final_gain_before_common_headroom']=final_gains[arm][condition]
                    values['final_gain_after_common_headroom']=final_gains[arm][condition]*g
                    values['target_sample_hann_envelope']=envelope(target)
                condition_metrics[condition]=row; rows.append(row)
                path=out/'acoustics'/sid/arm/f'{condition}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**values)
                item['acoustics']={'path':str(path),'sha256':v1.sha(path)}; manifest.append(item)
    directions={}
    for condition in CONDITIONS[4:]:
        selected=[r for r in rows if r['condition']==condition and r['non_silent']]
        error=np.asarray([r['target_envelope_db_rmse'] for r in selected])
        change=np.abs([r['R_RMS_relative_change_from_identity'] for r in selected])
        directions[condition]={'n':len(selected),'envelope_db_rmse':v1.summary_stats(error),
                               'fraction_envelope_at_most_075':float(np.mean(error<=.75)),
                               'absolute_R_relative_change':v1.summary_stats(change),
                               'gates':{'envelope_median':bool(np.median(error)<=.35),'envelope_90_percent':bool(np.mean(error<=.75)>=.9),
                                        'R_absolute_median_at_most_2_percent':bool(np.median(change)<=.02)}}
    engineering={'restored_rms':all(r['restored_rms_relative_error']<=1e-5 for r in rows),
                 'saved_rms':all(r['saved_rms_relative_error']<=1e-5 for r in rows),
                 'finite':all(r['all_finite'] for r in rows),'exact_length':all(r['exact_length'] for r in rows),
                 'no_clipping':all(r['sample_clipping_fraction']==0 and r['peak']<=.9800001 for r in rows),
                 'zero_samples_preserved':all(r['zero_samples_preserved'] for r in rows if r['condition'].startswith('ENVmatched_'))}
    passed=all(engineering.values()) and all(all(d['gates'].values()) for d in directions.values())
    summary={'status':'calibration_passed' if passed else 'calibration_failed','directions':directions,'engineering_gates':engineering,
             'headroom_gains':common_g,'protocol_sha256':v1.sha(out/'protocol.json'),'code_sha256':v1.sha(__file__),
             'parent_artifact_manifest_sha256':v1.sha(V1/'artifact_hashes.json')}
    for name,value in [('metrics',rows),('waveform_manifest',manifest),('gain_iterations',traces),('summary',summary)]:
        v1.write_json(out/f'{name}.json',value)
    for rel,hash_value in archived.items(): assert v1.sha(V1/rel)==hash_value
    print(json.dumps({k:summary[k] for k in ['status','directions','engineering_gates']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=OUT)
    run(parser.parse_args().out)
