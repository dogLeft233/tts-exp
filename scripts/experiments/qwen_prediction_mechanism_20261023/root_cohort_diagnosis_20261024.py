"""CPU-only extension of parent review: observed anomaly and protocol checks."""
from root_review_20261024 import *
import io

def main():
    review=read(D/'review.json');cp=read(C/'C_protocol.json');op=read(O/'protocol.json');ix=read(O/'comparison_index.json')['records'];bp=read(C/'B_protocol.json')
    out={'no_new_inference':True};out['native_gap_by_source']=review['cohorts']
    # Population gap is exploratory; the cohorts were not randomly assigned.
    rng=np.random.default_rng(2026102401)
    old=np.array([x['gap'] for x in review['cohorts']['old']['sources']]);new=np.array([x['gap'] for x in review['cohorts']['new']['sources']])
    diff=new[rng.integers(10,size=(20000,10))].mean(1)-old[rng.integers(10,size=(20000,10))].mean(1)
    out['new_minus_old_gap']={'mean':float(new.mean()-old.mean()),'CI95_exploratory':np.quantile(diff,[.025,.975]).tolist()}
    out['pooled20_exploratory']=summary(np.r_[old,new])
    # Every old sample/draw qualification is fixed before this diagnosis.
    qual={(r['sample_id'],r['source'],r['draw']):r['qualified'] for r in bp['rows']}
    oq=read(O/'comparison_index.json')['records'];olds={}
    for a in oq:
        if a['level']=='norm' and a['model'] in ['L06','L17']:olds[(a['sample_id'],a['model'],a['draw_index'])]=read(a['score_path'])
    clean=[];excluded=[]
    for sid in sorted({k[0] for k in olds}):
        dr=[d for d in [1,2,3] if all(qual[(sid,m,d)] for m in ['L06','L17'])]
        clean.append(statistics.mean(olds[(sid,'L17',d)]['sync_c']-olds[(sid,'L06',d)]['sync_c'] for d in dr))
        excluded.extend((sid,d) for d in [1,2,3] if d not in dr)
    out['old_qualified_draws']=dict(summary(clean),excluded=excluded)
    out['old_exclude_both_anomalous_sources']=summary([x['gap'] for x in review['cohorts']['old']['sources'] if x['id'] not in {s for s,d in excluded}])
    # Parse actual native generation arguments across 30 old and 30 new draws.
    oldgen=[];newgen=[];normalization=[];newdetails=[]
    for row in cp['rows']:
        if row['policy']!='baseline':continue
        if row['model']=='L17':
            q=read(C/'C_generation_L17'/row['id']/'result.json');newgen.append(q['talker_kwargs'])
        a=read(C/'C_audio'/row['id']/'result.json');z,sr=sf.read(C/'C_audio'/row['id']/'float16.wav',dtype='float32');pcm,s=sf.read(a['path'],dtype='int16')
        gain=cp['target_rms']/float(np.sqrt(np.mean(z.astype(float)**2)));y=z*np.float32(gain)
        # libsndfile float->PCM conversion has boundary rounding behavior not
        # exactly reproduced by a float64 floor formula. Replay the fixed writer
        # in memory rather than misclassify one-LSB conversion boundaries.
        mem=io.BytesIO();sf.write(mem,y,16000,subtype='PCM_16',format='WAV');mem.seek(0);expected,_=sf.read(mem,dtype='int16')
        check(sr==s==16000 and np.array_equal(expected,pcm),'canonical PCM independent reconstruction')
        normalization.append(abs(a['rms_after']-cp['target_rms']))
        newdetails.append({'source':row['sample_id'],'model':row['model'],'draw':row['draw'],'duration_ratio':a['duration_ratio'],'EOS':a['EOS'],'qualified':a['qualified_EOS_duration']})
    olddetails=[]
    for a in ix:
        if a['level']!='norm' or a['model'] not in ['L06','L17']:continue
        if a['model']=='L17':
            q=read(Path(a['source_path']).parent/f'draw{a["draw_index"]}.json');oldgen.append(q['talker_effective_kwargs'])
        else:q=read(Path(a['source_path']).parent/'record.json')
        olddetails.append({'source':a['sample_id'],'model':a['model'],'draw':a['draw_index'],'duration_ratio':q['duration_s']/next(s['duration_s'] for s in op['samples'] if s['sample_id']==a['sample_id']),'EOS':q['eos_in_talker_sequences']})
    check(all(x==oldgen[0] for x in oldgen+newgen),'native old/new actual sampler differs')
    out['actual_sampling']={'old_draws':len(oldgen),'new_draws':len(newgen),'all_kwargs_exact':True,'kwargs':{k:v for k,v in oldgen[0].items() if k!='suppress_tokens'}}
    out['new_native_audio']={'independent_PCM_reconstructions':len(normalization),'max_RMS_error':max(normalization),'qualification_failures':[x for x in newdetails if not x['qualified']]}
    out['native_duration_diagnostics']={cohort:{m:{'median_ratio':float(np.median([x['duration_ratio'] for x in rows if x['model']==m])),'min_ratio':min(x['duration_ratio'] for x in rows if x['model']==m),'max_ratio':max(x['duration_ratio'] for x in rows if x['model']==m),'noEOS':sum(not x['EOS'] for x in rows if x['model']==m)} for m in ['L06','L17']} for cohort,rows in [('old',olddetails),('new',newdetails)]}
    # Compare actual pipeline and syncnet command options, removing only sample paths.
    def signature(cmd):
        dynamic={'--videofile','--reference','--data_dir','--audio','--outfile'};out=[];i=0
        while i<len(cmd):
            v=cmd[i];out.append(v)
            if v in dynamic and i+1<len(cmd):out.append('<sample-specific>');i+=2
            else:i+=1
        return out
    groups={};scorefiles=list(read(C/'C_analysis.json')['score_pins'])+[x['score_path'] for x in ix if x['level']=='norm' and x['model'] in ['L06','L17']]
    for k in ['pipeline_receipt','syncnet_receipt']:
        sigs={json.dumps(signature(read(p)[k]['command'])) for p in scorefiles};check(len(sigs)==1,'actual score commands differ '+k);groups[k]=json.loads(next(iter(sigs)))
    out['actual_official_command_signatures']=groups
    out['cohort_material']={c:{'mean_duration':statistics.mean(s['duration_s'] for s in p['samples']),'mean_characters':statistics.mean(len(''.join(s['transcript'].split())) for s in p['samples']),'selection':p.get('selection','old cloud10 fixed subset'),'ids':[s['sample_id'] for s in p['samples']]} for c,p in [('old',op),('new',cp)]}
    # Recompute the previously reported native-entropy/score association with/without anomalies.
    desc=read(C/'B_entropy_score_description.json');assoc=[]
    for cand in desc['candidates']:
        x=np.array(cand['entropy_gaps']);y=np.array(cand['score_gaps']);xc=x-x.mean(1,keepdims=True);yc=y-y.mean(1,keepdims=True)
        full=float(np.corrcoef(xc.ravel(),yc.ravel())[0,1]);aa=[];bb=[]
        for i,sid in enumerate(cand['source_ids']):
            use=[j for j in range(3) if all(qual[(sid,m,j+1)] for m in ['L06','L17'])];a=x[i,use];b=y[i,use];aa.extend(a-a.mean());bb.extend(b-b.mean())
        assoc.append({'category':cand['candidate'],'all_draw_source_centered_r':full,'qualified_draw_source_centered_r':float(np.corrcoef(aa,bb)[0,1]),'interpretation':'posthoc sensitivity, not causal'})
    out['old_entropy_score_anomaly_sensitivity']=assoc
    (D/'cohort_diagnosis.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in out.items() if k not in ['native_gap_by_source','actual_official_command_signatures','actual_sampling']},indent=2))
if __name__=='__main__':main()
