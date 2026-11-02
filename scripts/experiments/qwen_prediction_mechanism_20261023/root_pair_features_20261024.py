"""Bounded post-hoc CPU description of existing native pairs; no inference."""
from root_review_20261024 import R, C, O, D, read, sha
import json
import numpy as np
import soundfile as sf
from scipy.stats import spearmanr

FEATURES = ['duration_s', 'quiet_fraction', 'envelope_cv', 'spectral_flux',
            'centroid_hz', 'high_band_fraction']

def features(path):
    y, sr = sf.read(path, dtype='float64')
    assert sr == 16000 and y.ndim == 1 and np.isfinite(y).all()
    frames = np.lib.stride_tricks.sliding_window_view(y, 400)[::160]
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    rms = np.sqrt(np.mean(y ** 2))
    active = energy > rms * 10 ** (-30 / 20)
    power = abs(np.fft.rfft(frames * np.hanning(400), axis=1)) ** 2
    prob = power / np.maximum(power.sum(1, keepdims=True), 1e-30)
    freq = np.fft.rfftfreq(400, 1 / sr)
    transitions = active[1:] & active[:-1]
    flux = abs(prob[1:] - prob[:-1]).sum(1)
    return dict(zip(FEATURES, [len(y)/sr, float(1-active.mean()),
        float(energy.std()/energy.mean()), float(flux[transitions].mean()),
        float((prob @ freq)[active].mean()),
        float(prob[:, freq >= 4000].sum(1)[active].mean())]))

def main():
    target = R/'runs/qwen_native_pair_features_20261024_attempt03'
    target.mkdir(exist_ok=False)
    review = read(D/'review.json'); op=read(O/'protocol.json'); cp=read(C/'C_protocol.json')
    paths = {}
    for row in read(O/'comparison_index.json')['records']:
        if row['level']=='norm' and row['model'] in ['L06','L17']:
            paths[('old',row['sample_id'],row['model'],row['draw_index'])]=row['audio_path']
    for row in cp['rows']:
        if row['policy']=='baseline':
            paths[('new',row['sample_id'],row['model'],row['draw'])]=read(C/'C_audio'/row['id']/'result.json')['path']
    assert len(paths)==120
    raw={k:features(p) for k,p in paths.items()}
    qualified={(r['sample_id'],r['source'],r['draw']):r['qualified'] for r in read(C/'B_protocol.json')['rows']}
    rows=[]
    for cohort, protocol in [('old',op),('new',cp)]:
        for src in review['cohorts'][cohort]['sources']:
            sid=src['id']; sample=next(s for s in protocol['samples'] if s['sample_id']==sid)
            row=dict(src, cohort=cohort, reference_duration_s=sample['duration_s'],
                characters=len(''.join(sample['transcript'].split())))
            row['positive_draws']=sum(g>0 for g in src['draw_gaps'])
            row['negative_draws']=sum(g<0 for g in src['draw_gaps'])
            row['features']={}
            use=[d for d in [1,2,3] if cohort=='new' or all(qualified[(sid,m,d)] for m in ['L06','L17'])]
            row['common_qualified_draws']=use
            row['qualified_gap']=float(np.mean([src['draw_gaps'][d-1] for d in use]))
            for f in FEATURES:
                a=np.array([raw[(cohort,sid,'L06',d)][f] for d in [1,2,3]])
                b=np.array([raw[(cohort,sid,'L17',d)][f] for d in [1,2,3]])
                row['features'][f]={'L06':float(a.mean()),'L17':float(b.mean()),'difference':float((b-a).mean()),
                    'qualified_difference':float(np.mean([(b-a)[d-1] for d in use]))}
            rows.append(row)
    correlations={}
    candidates=['reference_duration_s','characters']+FEATURES
    for f in candidates:
        correlations[f]={}
        for cohort in ['old','new']:
            use=[r for r in rows if r['cohort']==cohort]
            x=[r[f] if f in ['reference_duration_s','characters'] else r['features'][f]['difference'] for r in use]
            correlations[f][cohort]=float(spearmanr(x,[r['gap'] for r in use]).statistic)
    sensitivity={}
    for f in FEATURES:
        sensitivity[f]={}
        for cohort in ['old','new']:
            use=[r for r in rows if r['cohort']==cohort]
            x=np.array([r['features'][f]['qualified_difference'] for r in use]); y=np.array([r['qualified_gap'] for r in use])
            loo=[float(spearmanr(np.delete(x,i),np.delete(y,i)).statistic) for i in range(10)]
            xx=[];yy=[]
            for r in use:
                sid=r['id'];ds=r['common_qualified_draws']
                a=np.array([raw[(cohort,sid,'L17',d)][f]-raw[(cohort,sid,'L06',d)][f] for d in ds])
                b=np.array([r['draw_gaps'][d-1] for d in ds]);xx.extend(a-a.mean());yy.extend(b-b.mean())
            sensitivity[f][cohort]={'qualified_source_spearman':float(spearmanr(x,y).statistic),
                'delete_one_source_spearman_range':[min(loo),max(loo)],
                'qualified_source_centered_Pearson':float(np.corrcoef(xx,yy)[0,1])}
    result={'scope':'Post-hoc descriptive; 20 sources, 3 draws each; no multiplicity-adjusted inference or causal claims.',
        'feature_definitions':{'quiet_fraction':'25ms frames/10ms hop RMS below whole-wave RMS -30dB; energy proxy, not VAD',
        'envelope_cv':'25ms frame RMS std/mean', 'spectral_flux':'mean L1 change of normalized power spectra on adjacent active frames',
        'centroid_hz':'mean power centroid on active frames', 'high_band_fraction':'power fraction >=4kHz on active frames'},
        'input_pins':{p:sha(p) for p in paths.values()},'rows':rows,'source_spearman':correlations,
        'sensitivity':sensitivity,'raw_features':[dict(cohort=k[0],sample_id=k[1],model=k[2],draw=k[3],**v) for k,v in raw.items()]}
    (target/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'rows':len(rows),'audio_files':len(raw),'correlations':correlations,
        'sensitivity':sensitivity,'stability':{c:{'all3positive':[r['id'] for r in rows if r['cohort']==c and r['positive_draws']==3],
        'all3negative':[r['id'] for r in rows if r['cohort']==c and r['negative_draws']==3]} for c in ['old','new']}},indent=2))

if __name__=='__main__': main()
