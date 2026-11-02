"""Independent CPU curve/scalar and source-bootstrap audit; no model inference."""
from pathlib import Path
import argparse,hashlib,json,math,pickle,statistics
import numpy as np

def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main(run):
    p=load(run/'C_protocol.json');plan=load(run/'C_downstream_plan.json');result=load(run/'C_analysis.json');rows={r['id']:r for r in p['rows']};scored={};curve_checks=[];ambiguities=[]
    for r in plan['reuse']:
        assert sha(r['score_path'])==r['score_sha'];scored[r['id']]=load(r['score_path'])
    for job in plan['jobs']:
        d=Path(job['directory']);summary=load(d/'official/summary.json');assert summary['status']=='COMPLETE' and summary['completed']==job['count']
        for r in load(d/'unit_mapping.json'):
            scored[r['id']]=load(d/'official/scores'/f'{r["sample_id"]}_{r["role"]}.json')
    assert len(scored)==len(plan['reuse'])+sum(j['count'] for j in plan['jobs'])
    for key,s in scored.items():
        assert s['status']=='COMPLETE'
        for receipt in ['pipeline_receipt','syncnet_receipt']:
            assert s[receipt]['returncode']==0 and not s[receipt]['timed_out']
        assert s['decoded_frames_exact'] and s['frame_PTS_exact'] and s['mux_PCM_exact']
        assert sha(s['canonical_audio_path'])==s['canonical_audio_sha256']
        assert sha(s['activesd_path'])==s['activesd_sha256']
        curves=pickle.loads(Path(s['activesd_path']).read_bytes());assert len(curves)==len(s['tracks'])
        for index,(curve,t) in enumerate(zip(curves,s['tracks'])):
            x=np.asarray(curve,dtype=np.float64);assert x.ndim==2 and x.shape[1]==31 and x.shape[0]>0 and np.isfinite(x).all()
            # Verbatim original formula: mean(torch.stack(dists,1),1),
            # offset=15-argmin, C=median(mdist)-min; save is[N,31].
            means=x.mean(axis=0);i=int(means.argmin());d=float(means[i]);c=float(np.median(means)-d);offset=15-i
            assert abs(c-t['sync_c'])<=.00051 and abs(d-t['sync_d'])<=.00051
            if offset!=t['av_offset']:
                original_index=15-t['av_offset'];assert 0<=original_index<31 and means[original_index]-means[i]<=2e-5
                ambiguities.append({'id':key,'track':index,'native_offset':t['av_offset'],'float64_min_offset':offset,'mean_distance_gap':float(means[original_index]-means[i])})
            curve_checks.append({'id':key,'track':index,'windows':x.shape[0],'C_float64':c,'Dmin_float64':d,'native_print_C':t['sync_c'],'native_print_Dmin':t['sync_d'],'C_rounding_difference':c-t['sync_c'],'Dmin_rounding_difference':d-t['sync_d'],'offset_float64':offset})
        assert s['sync_c']==s['tracks'][0]['sync_c'] and s['sync_d']==s['tracks'][0]['sync_d']
    def root(key):
        while rows[key]['duplicate_of']:key=rows[key]['duplicate_of']
        return key
    all_scores={r['id']:scored[root(r['id'])] for r in p['rows'] if root(r['id']) in scored};sources=sorted(s['sample_id'] for s in p['samples']);support=[];vectors={'P1_L06_both_minus_baseline':[],'P2_gap_shrink':[]}
    for sid in sources:
        ids=[f'{sid}_{m}_{pol}_{d}' for m,pol in [('L06','baseline'),('L17','baseline'),('L06','both'),('L17','both')] for d in range(1,4)]
        if not all(key in all_scores for key in ids):continue
        support.append(sid);p1=[];p2=[]
        for d in range(1,4):
            get=lambda m,pol:all_scores[f'{sid}_{m}_{pol}_{d}']['sync_c']
            p1.append(get('L06','both')-get('L06','baseline'));p2.append((get('L17','baseline')-get('L06','baseline'))-(get('L17','both')-get('L06','both')))
        vectors['P1_L06_both_minus_baseline'].append(statistics.fmean(p1));vectors['P2_gap_shrink'].append(statistics.fmean(p2))
    assert support==result['complete_primary_source_ids'];statistics_checks=[]
    if support:
        indices=np.random.default_rng(2026102308).integers(0,len(support),(20000,len(support)))
        for name,values in vectors.items():
            point=statistics.fmean(values);boots=[statistics.fmean(values[int(i)] for i in draw) for draw in indices];ci=np.quantile(boots,[.0125,.9875]);reported=result['primary_and_gap_contrasts'][name]
            assert abs(point-reported['mean'])<1e-12 and np.max(abs(ci-np.asarray(reported['CI97_5'])))<1e-12
            statistics_checks.append({'comparison':name,'mean':point,'CI97_5':ci.tolist(),'source_count':len(support),'source_unit_not_draws':True})
    else:statistics_checks.append({'status':'PRIMARY_UNMEASURABLE','reason':'no common full3draw4arm source support; not a zero effect'})
    gen=[]
    for r in p['rows']:
        q=load(run/('C_generation_'+r['model'])/r['id']/'result.json')
        if q['status']=='COMPLETE':
            kwargs=q['talker_kwargs'];assert kwargs['temperature']==r['temperatures'][0] and kwargs['subtalker_temperature']==r['temperatures'][1]
            assert kwargs['top_k']==kwargs['subtalker_top_k']==50 and kwargs['top_p']==kwargs['subtalker_top_p']==1. and kwargs['repetition_penalty']==1.05
            assert kwargs['do_sample'] and kwargs['subtalker_dosample'] and kwargs['max_new_tokens']==512 and kwargs['min_new_tokens']==2 and kwargs['eos_token_id']==2150
            assert set(kwargs['suppress_tokens'])==set(range(2048,3072))-{2150}
            assert sha(q['raw']['path'])==q['raw']['sha256'] and sha(q['codes']['path'])==q['codes']['sha256']
        gen.append(q['status'])
    out={'status':'PASS','score_units':len(scored),'logical_scored_references':len(all_scores),'curve_tracks':len(curve_checks),'curve_checks':curve_checks,'float32_near_tied_offset_ambiguities':ambiguities,'scalar_rounding_tolerance':.00051,'scalar_tolerance_basis':'original stdout3decimal rounding0.0005 plus1e-5 float32 reduction allowance; no score is changed','source_bootstrap_checks':statistics_checks,'C_generation_status_counts':{s:gen.count(s) for s in sorted(set(gen))},'frozen_generation_policy_verified_for_all_successful_new_units':True,'no_new_model_inference':True,'C_protocol_sha':sha(run/'C_protocol.json')}
    with (run/'C_independent_CPU_audit.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
