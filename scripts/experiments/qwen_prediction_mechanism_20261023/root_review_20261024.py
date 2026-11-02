"""Parent-authored CPU readback and cohort diagnosis; no inference imports."""
from pathlib import Path
import json, hashlib, pickle, csv, statistics, itertools
import numpy as np
import soundfile as sf

R=Path('/home/wjj/tts-audio/tts-exp')
C=R/'runs/qwen_prediction_mechanism_20261023_attempt02'
O=R/'runs/qwen17_level_comparison10_20261023_attempt02'
D=R/'runs/qwen_prediction_root_review_20261024_attempt01'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(v, message):
    if not v: raise AssertionError(message)
def boot(v,seed=2026102308):
    a=np.asarray(v);z=a[np.random.default_rng(seed).integers(len(a),size=(20000,len(a)))].mean(1)
    return {'mean':float(a.mean()),'CI95':np.quantile(z,[.025,.975]).tolist(),'CI97_5':np.quantile(z,[.0125,.9875]).tolist()}
def summary(v):
    v=np.asarray(v);return dict(boot(v),median=float(np.median(v)),positive=int(sum(v>0)),negative=int(sum(v<0)),leave_one_out=[float(np.delete(v,i).mean()) for i in range(len(v))],trim_one_each_end=float(np.sort(v)[1:-1].mean()))
def main():
    D.mkdir(exist_ok=False)
    out={'no_new_inference':True,'author':'root independent implementation'}
    # A: actual arrays and historical media, not producer pass flags.
    ac=[]
    for m in ['L06','L17']:
        a=np.load(C/f'A_{m}/native.npz');b=np.load(C/f'A_{m}/teacher.npz')
        dif=max(float(np.max(abs(a[k]-b[k]))) for k in a.files if k.startswith(('raw_','probability_')))
        check(dif==0,'A native/teacher trace mismatch')
        check(np.array_equal(np.load(C/f'A_{m}/native_codes.npy'),np.load(C/f'A_{m}/teacher_codes.npy')),'A code mismatch')
        check(np.array_equal(sf.read(C/f'A_{m}/native.wav')[0],sf.read(C/f'A_{m}/teacher.wav')[0]),'A wave mismatch')
        ac.append({'model':m,'raw_probability_difference':dif,'wave_codes_exact':True})
    out['A']=ac
    # B: recompute from every stored effective logit, average by source/origin.
    bp=read(C/'B_protocol.json');bg=read(C/'B_gate_and_calibration.json');bd=read(C/'B_history_diagnostics.json')['statistics']
    lookup={(x['reader'],x['id']):x for x in bd if 'reader' in x};vals={};worst=0.;softdiff=0.
    for m in ['L06','L17']:
        for row in bp['rows']:
            with np.load(C/f'B_{m}'/row['id']/'trace.npz') as z:
                books=[]
                for cb in range(16):
                    e=z[f'effective_{cb}'][:row['frames'] if 'frames' in row else lookup[(m,row['id'])]['frames'],:2048].astype(float)
                    ex=np.exp(e-e.max(1,keepdims=True));p=ex/ex.sum(1,keepdims=True)
                    h=-(p*np.log(np.maximum(p,1e-300))).sum(1).mean();books.append(float(h))
                    worst=max(worst,abs(h-lookup[(m,row['id'])]['codebooks'][cb]['effective_content_entropy']))
                    full=z[f'effective_{cb}'].astype(float);ex=np.exp(full-full.max(1,keepdims=True));q=ex/ex.sum(1,keepdims=True)
                    softdiff=max(softdiff,float(np.max(abs(q-z[f'probability_{cb}']))))
                vals[(m,row['id'])]=books
    check(worst<1e-10 and softdiff<1e-5,'B entropy/probability mismatch')
    source={}
    for m in ['L06','L17']:
        source[m]=np.array([np.mean([np.mean([vals[(m,r['id'])] for r in bp['rows'] if r['sample_id']==sid and r['source']==origin and r['qualified']],axis=0) for origin in ['N','L06','L17']],axis=0) for sid in bg['source_ids']])
        check(np.max(abs(source[m]-np.array(bg['source_codebook_means'][m])))<1e-10,'B aggregation mismatch')
    bstats={}
    for name,ix in [('coarse',[0]),('fine',list(range(1,16)))]:
        s=boot((source['L17']-source['L06'])[:,ix].mean(1),2026102307);check(np.max(abs(np.array(s['CI97_5'])-bg['gates'][name]['CI97_5']))<1e-10,'B bootstrap mismatch');bstats[name]=s
    out['B']={'passes':len(vals),'entropy_max_error':worst,'softmax_max_error':softdiff,'recomputed':bstats}
    print('A/B independent raw-array checks passed',flush=True)
    # Read C unique official scores and old normalized L06/L17 scores.
    cp=read(C/'C_protocol.json');ca=read(C/'C_analysis.json');plan=read(C/'C_downstream_plan.json');scores={}
    for x in plan['reuse']: check(sha(x['score_path'])==x['score_sha'],'reuse SHA');scores[x['id']]=read(x['score_path'])
    for job in plan['jobs']:
        j=Path(job['directory'])
        for x in read(j/'unit_mapping.json'):scores[x['id']]=read(j/'official/scores'/f'{x["sample_id"]}_{x["role"]}.json')
    oldix=[x for x in read(O/'comparison_index.json')['records'] if x['level']=='norm' and x['model'] in ['L06','L17']]
    oldscores={}
    for x in oldix:check(sha(x['score_path'])==x['score_sha256'],'old score SHA');oldscores[(x['sample_id'],x['model'],x['draw_index'])]=read(x['score_path'])
    curves=[];commands=[]
    for s in list(scores.values())+list(oldscores.values()):
        check(sha(s['activesd_path'])==s['activesd_sha256'],'curve SHA')
        for ar,t in zip(pickle.loads(Path(s['activesd_path']).read_bytes()),s['tracks']):
            x=np.asarray(ar,float);mu=x.mean(0);err=max(abs(float(np.median(mu)-mu.min())-t['sync_c']),abs(float(mu.min())-t['sync_d']))
            check(err<=.00051,'official curve/scalar mismatch');curves.append(err)
        commands.append(s['pipeline_receipt']['command'])
        for k in ['pipeline_receipt','syncnet_receipt']:check(s[k]['returncode']==0 and not s[k]['timed_out'],'score exit')
        check(s['mux_PCM_exact'] and s['decoded_frames_exact'] and s['frame_PTS_exact'],'media binding')
        check(sha(s['canonical_audio_path'])==s['canonical_audio_sha256'],'audio SHA')
    out['score_audit']={'unique_scores':len(scores),'old_norm_scores':len(oldscores),'track_count':len(curves),'max_scalar_rounding_error':max(curves)}
    rows={x['id']:x for x in cp['rows']}
    def get(sid,m,pol,d):
        key=f'{sid}_{m}_{pol}_{d}'
        while rows[key]['duplicate_of']:key=rows[key]['duplicate_of']
        return scores[key]
    for name,fun in [('P1_L06_both_minus_baseline',lambda sid,d:get(sid,'L06','both',d)['sync_c']-get(sid,'L06','baseline',d)['sync_c']),('P2_gap_shrink',lambda sid,d:get(sid,'L17','baseline',d)['sync_c']-get(sid,'L06','baseline',d)['sync_c']-get(sid,'L17','both',d)['sync_c']+get(sid,'L06','both',d)['sync_c'])]:
        s=boot([statistics.mean(fun(sid,d) for d in [1,2,3]) for sid in ca['source_ids']]);check(abs(s['mean']-ca['primary_and_gap_contrasts'][name]['mean'])<1e-12,'C primary point');check(np.max(abs(np.array(s['CI97_5'])-ca['primary_and_gap_contrasts'][name]['CI97_5']))<1e-12,'C primary CI');out[name]=s
    # Cross-cohort native baseline comparisons, all 3 seeds retained.
    records={};audio={x['id']:x for x in read(C/'C_audio_manifest.json')['records'] if x['status']=='COMPLETE'}
    for cohort,sids in [('old',sorted({k[0] for k in oldscores})),('new',ca['source_ids'])]:
        rr=[]
        for sid in sids:
            per={m:[oldscores[(sid,m,d)] if cohort=='old' else get(sid,m,'baseline',d) for d in [1,2,3]] for m in ['L06','L17']}
            gap=[per['L17'][i]['sync_c']-per['L06'][i]['sync_c'] for i in range(3)]
            rr.append({'id':sid,'L06':statistics.mean(x['sync_c'] for x in per['L06']),'L17':statistics.mean(x['sync_c'] for x in per['L17']),'gap':statistics.mean(gap),'draw_gaps':gap,'D_change':statistics.mean(x['sync_d'] for x in per['L17'])-statistics.mean(x['sync_d'] for x in per['L06']),'median_distance_change':statistics.mean(x['sync_d']+x['sync_c'] for x in per['L17'])-statistics.mean(x['sync_d']+x['sync_c'] for x in per['L06'])})
        records[cohort]=rr
    out['cohorts']={c:dict(summary([x['gap'] for x in rr]),L06=statistics.mean(x['L06'] for x in rr),L17=statistics.mean(x['L17'] for x in rr),by_draw=[summary([x['draw_gaps'][i] for x in rr]) for i in range(3)],sources=rr) for c,rr in records.items()}
    # Frozen downstream and sampling metadata comparisons; verify declared hashes.
    op=read(O/'protocol.json');oldp=read(R/'runs/qwen17_level_comparison10_20261023_attempt01/protocol.json')
    check(cp['downstream']==op['downstream'],'downstream differs');check(cp['generation_kwargs']==op['generation_kwargs'],'generation differs')
    common=set(cp['pins'])&set(op['source_pins']);diff={k:[cp['pins'][k],op['source_pins'][k]] for k in common if cp['pins'][k]!=op['source_pins'][k]};check(not diff,'shared pins differ')
    for f,h in cp['pins'].items():check(sha(f)==h,'C frozen pin '+f)
    out['protocol']={'downstream_exact':True,'generation_kwargs_exact':True,'common_pins':len(common),'shared_pin_differences':diff,'verified_C_pins':len(cp['pins']),'target_rms_old':op['target_rms'],'target_rms_new':cp['target_rms']}
    gen=[]
    for row in cp['rows']:
        if row['duplicate_of'] or row['policy']!='baseline':continue
        a=audio[row['id']];gen.append({'id':row['id'],'EOS':a['EOS'],'qualified':a['qualified_EOS_duration'],'ratio':a['duration_ratio'],'rms':a['rms_after']})
        if row['model']=='L17':
            q=read(C/'C_generation_L17'/row['id']/'result.json');check(q['talker_kwargs']['temperature']==.9 and q['talker_kwargs']['subtalker_temperature']==.9,'baseline changed')
            oldrec=read(row['old_record']);check(sha(q['reference_codes']['path'])==oldrec['reference_codes']['sha256'],'ref codes differ')
    out['new_native_generation']=gen
    out['status']='PASS_PARENT_CPU_READBACK'
    (D/'review.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in out.items() if k not in ['cohorts','new_native_generation']},indent=2),flush=True)
    for c,v in out['cohorts'].items():print(c,json.dumps(v),flush=True)
if __name__=='__main__':main()
