"""Independent known-map, clip weighting, count and bootstrap verification."""
import numpy as np
from scripts.experiments.tts_instance_audio_dtw import ROOT,read,sha,write

def main():
    out=ROOT/'runs/tts_dtw_calibration_audit_20260926';p=read(out/'protocol.json');summary=read(out/'analysis.json');maxerr=0.;pointcount=0
    assert len(p['cal_ids'])==26 and all(r['id'] in p['cal_ids'] and (r['arm']=='N' or r['stop']=='EOS') for r in p['rows'])
    allclips={}
    for r in p['rows']:
        for key in ('audio','grid'):assert sha(r[key]['path'])==r[key]['sha256']
        z=read(out/'clips'/r['id']/r['arm']/'controls.json');allclips[r['id'],r['arm']]=z
        for phone in z['identity_paths']['phones']:
            assert (phone['cost']==0 and all(a==b for a,b in phone['path'])) if phone['path'] else phone['cost'] is None
        for q in z['records']:
            ph=z['warp_paths']['phones'][q['event']];lo,hi=[round(t*16000)/16000 for t in ph['spans'][0]];forward=q['direction']=='Q1Q2';xs=[lo,(lo+hi)/2,hi];ys=[lo,lo+.75*(hi-lo),hi]
            expected=float(np.interp(q['source_time'],xs,ys)) if forward else float(np.interp(q['source_time'],ys,xs));assert abs(expected-q['truth_time'])<1e-12
            path=ph['path'];a,b=(0,1) if forward else (1,0);knots=sorted({v[a] for v in path});mapped=[sum(v[b] for v in path if v[a]==i)/sum(v[a]==i for v in path)/80 for i in knots];t=float(np.interp(q['source_time'],np.asarray(knots)/80,mapped));assert abs(t-q['dtw_time'])<1e-12
            assert q['identity_error_ms']==0
            for method,time in [('mfa',q['source_time']),('dtw',t)]:
                err=abs(time-expected)*1000;maxerr=max(maxerr,abs(err-q[method+'_error_ms']))
                ix=lambda x:int(np.floor((x-.1075)*25+.5));assert abs(ix(time)-ix(q['truth_time']))==q[method+'_index_error_frames']
            pointcount+=1
    for arm in ('N','Q1'):
        for name,res in summary[arm].items():
            cutoff=.16 if name.startswith('phone_ge160ms') else .08;direction='both' if name.endswith('both') else name.rsplit('_',1)[1];clipvals=[];nphones=0;npoints=0;zeros=[]
            for sid in p['cal_ids']:
                if (sid,arm) not in allclips:continue
                rr=[q for q in allclips[sid,arm]['records'] if q['duration']>=cutoff-1e-9 and (direction=='both' or q['direction']==direction)]
                if not rr:zeros.append(sid);continue
                vals={m:sum(q[m] for q in rr)/len(rr) for m in ('mfa_error_ms','dtw_error_ms','identity_error_ms','mfa_index_error_frames','dtw_index_error_frames')}
                vals['delta_error_ms']=sum(q['dtw_error_ms']-q['mfa_error_ms'] for q in rr)/len(rr)
                for method in ('mfa','dtw'):vals[method+'_index_hit']=sum(q[method+'_index_error_frames']==0 for q in rr)/len(rr)
                clipvals.append(vals);nphones+=len({q['event'] for q in rr});npoints+=len(rr)
            assert (len(clipvals),nphones,npoints,zeros)==(res['clips'],res['phones'],res['points'],res['zero_phone_clip_ids'])
            for metric,z in res['analysis'].items():
                vals=np.array([c[metric] for c in clipvals]);rng=np.random.default_rng(20260926);idx=rng.choice(len(vals),size=(20000,len(vals)),replace=True);means=vals[idx].sum(axis=1)/len(vals)
                maxerr=max(maxerr,abs(vals.sum()/len(vals)-z['mean']),float(np.max(abs(np.quantile(means,[.005,.995])-z['ci99']))))
            # Independently check each stored weighted quantile brackets its target CDF.
            for metric,qq in res['time_quantiles'].items():
                for key,target in zip(('min','q25','median','q75','q90','q95','q99','max'),(0,.25,.5,.75,.9,.95,.99,1)):
                    cut=qq[key];below=0.;at=0.
                    for sid in p['cal_ids']:
                        if (sid,arm) not in allclips:continue
                        vv=[q[metric] for q in allclips[sid,arm]['records'] if q['duration']>=cutoff-1e-9 and (direction=='both' or q['direction']==direction)]
                        if not vv:continue
                        below+=sum(v<cut for v in vv)/len(vv)/len(clipvals);at+=sum(v<=cut for v in vv)/len(vv)/len(clipvals)
                    assert below<=target+1e-12 and at>=target-1e-12
    assert read(out/'original_replay.json')['PASS']
    result={'PASS':maxerr<1e-10,'clips':len(p['rows']),'phase_direction_points':pointcount,'known_warp_truth_and_mapping_recomputed':True,'counts_clip_weights_bootstrap_quantiles_recomputed':True,'max_error':maxerr,'original_control_exact_replay':True}
    write(out/'validation.json',result);print(result);assert result['PASS']

if __name__=='__main__':main()
