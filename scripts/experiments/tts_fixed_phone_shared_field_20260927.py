"""Cal-only shared phone field. Freeze/review/bridge precede fit and scoring."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse,hashlib,json,shutil,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_fixed_phone_shared_field_20260927'
PARENT=ROOT/'runs/tts_level_event_cross_20260927'
FIX=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
CAL=ROOT/'runs/tts_shared_translation_20260926/calibration_support.json'
FEAS=ROOT/'runs/tts_fixed_phone_shift_feasibility_20260927'
import tts_level_event_cross_20260927 as old
GEOMS=['raw','unit']; METRICS=['positive','negative','margin','rank']
FAMILIES=['loso_shrink','loso_raw','loso_global','pooled_shrink']
SPECS=[{'name':'native','family':None,'sign':0}]+[{'name':f'{fam}_{s:+d}','family':fam,'sign':s} for fam in FAMILIES for s in [-1,1]]+[{'name':f'Q{i:02d}','family':'loso_shrink','Q':i,'sign':'main'} for i in range(16)]
CAP=48*2**20;FLOOR=int(4.5*2**30)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return old.sha(p)
def resource(extra=0):
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert used+extra<=CAP,('persistent cap',used,extra)
    assert shutil.disk_usage(ROOT).free>=FLOOR+extra,'disk floor'
def write(p,x):
    data=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n';resource(len(data.encode()));assert not Path(p).exists();Path(p).write_text(data)
def save(p,**arrays):
    resource(sum(a.nbytes for a in arrays.values())+4096);assert not Path(p).exists();np.savez_compressed(p,**arrays)
def load_clip(sid,phase=None):
    arrays={}
    for a in 'NT':
        with np.load(FIX/'features'/sid/a/'FIXED.npz') as f:
            arrays[a,'A']=f['audio'];arrays[a,'V']=f['visual']
        if phase is not None:
            entry=phase[sid][a]
            if entry.get('format')=='npy':arrays[a,'A']=np.load(entry['path'])
            else:
                with np.load(entry['path']) as f:arrays[a,'A']=f[entry['key']]
    return arrays
def geom(arrays,gi):
    return {k:(x if gi==0 else x/np.linalg.norm(x,axis=1)[:,None]).astype(np.float64) for k,x in arrays.items()}
def freeze():
    assert not (OUT/'protocol.json').exists();resource()
    c=read(CAL);ev=read(PARENT/'indices.json');cal=[r for r in c['rows'] if r['eligible']]
    assert len(cal)==20 and sum(len(r['queries']) for r in cal)==890
    assert len(ev['rows'])==32 and sum(len(r['queries']) for r in ev['rows'])==1411
    labels=sorted({r['nodes'][q['node']]['phone'] for r in cal for q in r['queries']});assert len(labels)==63
    folds=sorted({r['speaker'] for r in ev['rows']})+['pooled']
    inputs={}
    def bind(p,h=None):
        p=Path(p);v=sha(p);assert h is None or v==h;inputs[str(p)]=v
    for p in [CAL,CAL.parent/'fit.json',CAL.parent/'provenance.json',PARENT/'indices.json',PARENT/'query_metrics.npz',PARENT/'summary.json',PARENT/'protocol.json',FIX/'protocol.json',FIX/'feature_seal_calibration.json',FIX/'feature_seal_evaluation.json',FEAS/'inventory.json',FEAS/'loso_counts.json']:
        bind(p)
    for rows,split in [(cal,'calibration'),(ev['rows'],'evaluation')]:
        seal=read(FIX/f'feature_seal_{split}.json')
        for r in rows:
            for a in 'NT':
                fp=FIX/'features'/r['id']/a/'FIXED.npz';bind(fp,seal[str(fp)]);bind(fp.with_suffix('.json'),seal[str(fp.with_suffix('.json'))])
                with np.load(fp) as nf:
                    f={key:nf[key] for key in ['audio','visual']}
                    for key in ['audio','visual']:
                        x=f[key];assert x.ndim==2 and x.shape[1]==1024 and x.dtype==np.float32 and np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>0).all()
                    for q in r['queries']:
                        j=r['nodes'][q['node']]['j'][a];assert 0<=j<len(f['audio']) and 0<=j-3<len(f['visual'])
                        assert all(0<=r['nodes'][d]['j'][a]<len(f['audio']) for d in q['donors'])
    write(OUT/'indices.json',{'cal_rows':cal,'eval_rows':ev['rows'],'spans':ev['spans'],'labels':labels,'folds':folds})
    rng=np.random.Generator(np.random.PCG64(20260927));perms=[];signs=[]
    for _ in range(16):
        perms.append(rng.permutation(1024));signs.append(rng.choice(np.array([-1,1],np.int8),1024))
    save(OUT/'Q.npz',permutation=np.array(perms,dtype=np.int16),sign=np.array(signs,dtype=np.int8))
    write(OUT/'inputs.json',inputs)
    p={'status':'frozen_before_fit_and_score','created_epoch':time.time(),'design_sha256':sha(OUT/'protocol.md'),'specs':SPECS,'geometry':GEOMS,'metrics':METRICS,'labels':labels,'folds':folds,'code':{str(Path(__file__).resolve()):sha(__file__),str(Path(old.__file__).resolve()):sha(old.__file__)},'indices_sha256':sha(OUT/'indices.json'),'inputs_sha256':sha(OUT/'inputs.json'),'Q_sha256':sha(OUT/'Q.npz'),'joint_phase_frozen':True,'resources':{'cap_bytes':CAP,'floor_bytes':FLOOR,'cpu_threads_max':2,'RAM_target_bytes':256*2**20},'phase_binding_schema':'phase_binding.json: protocol_sha256=current; status=PASS; inputs={absolutePath:sha256}; features={sid:{N/T:{path,key}}}; phase_protocol_path; phase_execution_lock_path; producer_binding_path. Must bind before reading new A; no phase scores used.'}
    write(OUT/'protocol.json',p);write(OUT/'seal.json',{n:sha(OUT/n) for n in ['protocol.md','protocol.json','inputs.json','indices.json','Q.npz']})
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)
def locked(review=True):
    p=read(OUT/'protocol.json')
    for n,h in read(OUT/'seal.json').items():assert sha(OUT/n)==h,n
    for f,h in {**read(OUT/'inputs.json'),**p['code']}.items():assert sha(f)==h,f
    if review:
        r=read(OUT/'review.json');assert r['status']=='PASS' and r['protocol_sha256']==sha(OUT/'protocol.json')
    return read(OUT/'indices.json')
def score_row(row,f,arm,delta=None,checks=None):
    z=[]
    for q in row['queries']:
        node=row['nodes'][q['node']];p=node['phone'];j=node['j'][arm];ds=[row['nodes'][d] for d in q['donors']]
        v=f[arm,'V'][j-3];a=f[arm,'A'][j];neg=f[arm,'A'][[n['j'][arm] for n in ds]]
        d=v-neg+1e-6;dp=v-a+1e-6
        if delta is not None:
            sp=delta(p);sn=np.stack([delta(n['phone']) for n in ds]);vs=v+sp;aps=a+sp;ns=neg+sn
            dn=vs-ns+1e-6;dpp=vs-aps+1e-6;w=sp-sn
            if checks is not None:
                checks['positive_vector']=max(checks['positive_vector'],float(abs(dpp-dp).max()))
                same=np.array([n['phone']==p for n in ds]);
                if same.any():
                    checks['same_phone_vector']=max(checks['same_phone_vector'],float(abs(dn[same]-d[same]).max()))
                    checks['same_phone_distance']=max(checks['same_phone_distance'],float(abs(np.sqrt((dn[same]**2).sum(1))-np.sqrt((d[same]**2).sum(1))).max()))
                expected=np.sum(d*d,1)+2*np.sum(d*w,1)+np.sum(w*w,1)
                checks['squared_expansion']=max(checks['squared_expansion'],float(abs(np.sum(dn*dn,1)-expected).max()))
                opposite=d-w
                checks['odd_even']=max(checks['odd_even'],float(abs((np.sum(dn*dn,1)+np.sum(opposite*opposite,1))/2-np.sum(d*d,1)-np.sum(w*w,1)).max()))
            d,dp=dn,dpp
        positive=float(np.sqrt(np.sum(dp*dp)));negative=np.sqrt(np.sum(d*d,1));z.append([positive,float(negative.mean()),float(negative.mean()-positive),float(np.mean((negative>positive)+.5*(negative==positive)))])
    return np.array(z)
def bridge():
    idx=locked();out=np.zeros((1411,2,2,4));sp=idx['spans'];
    with np.load(PARENT/'query_metrics.npz') as f:expected=f['metrics'][:,:,[12,15],:]
    for r,s in zip(idx['eval_rows'],sp):
        arrays=load_clip(r['id'])
        for gi in range(2):
            f=geom(arrays,gi)
            for ai,a in enumerate('NT'):out[s['start']:s['stop'],gi,ai]=score_row(r,f,a)
    err=float(abs(out-expected).max());assert err<=1e-9
    cm=old.clipmeans(out,sp);parent=read(PARENT/'summary.json');se=0.;speakers=[s['speaker'] for s in sp]
    for gi,g in enumerate(GEOMS):
        for mi,m in enumerate(METRICS):
            for ai,a in enumerate('NT'):
                actual=old.stat(cm[:,gi,ai,mi],speakers);target=parent[f'{g}/{m}/cell/FF_{a}{a}']
                for k in ['speaker_mean','speaker_ci99','utterance_mean','per_speaker']:
                    av=actual[k];bv=target[k]
                    if isinstance(av,dict):av=list(av.values());bv=list(bv.values())
                    se=max(se,float(abs(np.array(av)-np.array(bv)).max()))
    assert se<=1e-9;save(OUT/'baseline.npz',metrics=out)
    write(OUT/'bridge.json',{'status':'PASS','created_epoch':time.time(),'query_max':err,'stat_max':se})
    print('BRIDGE',err,se,flush=True)
def fit():
    idx=locked();assert read(OUT/'bridge.json')['status']=='PASS';resource(15*2**20)
    labels=idx['labels'];folds=idx['folds'];perclip=[]
    for r in idx['cal_rows']:
        arrays=load_clip(r['id']);means=np.full((2,len(labels)+1,1024),np.nan)
        for gi in range(2):
            f=geom(arrays,gi);vals=[];phones=[]
            for q in r['queries']:
                n=r['nodes'][q['node']];p=n['phone'];m={a:(f[a,'A'][n['j'][a]]+f[a,'V'][n['j'][a]-3])/2 for a in 'NT'};vals.append(m['T']-m['N']);phones.append(p)
            vals=np.array(vals);means[gi,-1]=vals.mean(0)
            for li,p in enumerate(labels):
                take=np.array(phones)==p
                if take.any():means[gi,li]=vals[take].mean(0)
        perclip.append(means)
    perclip=np.array(perclip);raw=np.zeros((len(folds),2,len(labels)+1,1024));counts=np.zeros((len(folds),len(labels)),np.int16);train={}
    for fi,fold in enumerate(folds):
        use=[i for i,r in enumerate(idx['cal_rows']) if r['speaker']!=fold];speakers=sorted({idx['cal_rows'][i]['speaker'] for i in use});train[fold]=[idx['cal_rows'][i]['id'] for i in use]
        for li in range(len(labels)+1):
            sm=[]
            for s in speakers:
                take=[i for i in use if idx['cal_rows'][i]['speaker']==s and np.isfinite(perclip[i,0,li]).all()]
                if take:sm.append(perclip[take,:,li,:].mean(0))
            if sm:raw[fi,:,li]=np.array(sm).mean(0)
            if li<len(labels):counts[fi,li]=len(sm)
        for li in range(len(labels)):
            if counts[fi,li]==0:raw[fi,:,li]=raw[fi,:,-1]
    assert np.isfinite(raw).all();save(OUT/'fit.npz',raw_delta=raw,speaker_counts=counts)
    write(OUT/'fit.json',{'status':'CAL_ONLY_FIT_SEALED','created_epoch':time.time(),'fit_sha256':sha(OUT/'fit.npz'),'training_ids':train,'eval_features_used':False,'weights':'query within phone/clip -> clip within speaker -> speaker equal; global all query -> clip -> speaker; k is independent cal speakers','shape':list(raw.shape),'counts':counts.tolist()})
    print('FIT_SEALED',flush=True)
def field_model(idx,models,counts,fold,gi,spec,arm,Q):
    if spec['family'] is None:return None
    fam=spec['family'];fi=idx['folds'].index('pooled' if fam=='pooled_shrink' else fold);values=models[fi,gi];global_=values[-1];eta=values[:-1]-global_
    if fam=='loso_global':eta=np.zeros_like(eta)
    elif fam!='loso_raw':eta=eta*(counts[fi]/(counts[fi]+4))[:,None]
    if 'Q' in spec:
        i=spec['Q'];eta=eta[:,Q['permutation'][i]]*Q['sign'][i]
    sign=(-1 if arm=='T' else 1) if spec['sign']=='main' else spec['sign'];mapping={p:sign*(global_+eta[i]) for i,p in enumerate(idx['labels'])};fallback=sign*global_
    return lambda p:mapping.get(p,fallback)
def analyze(metrics,idx,stem):
    cm=old.clipmeans(metrics,idx['spans']);save(OUT/(stem+'_clips.npz'),metrics=cm);speakers=[s['speaker'] for s in idx['spans']];summary={}
    for gi,g in enumerate(GEOMS):
        for mi,m in enumerate(METRICS):
            for ci,c in enumerate(SPECS):
                v=cm[:,gi,ci,:,mi];base=cm[:,gi,0,:,mi]
                for ai,a in enumerate('NT'):
                    summary[f'{g}/{m}/{c["name"]}/{a}']=old.stat(v[:,ai],speakers)
                    response=v[:,ai]-base[:,ai]
                    summary[f'{g}/{m}/{c["name"]}/response_{a}']=old.stat(response,speakers)
                    summary[f'{g}/{m}/{c["name"]}/gap_change_only_{a}']=old.stat(response*(1 if a=='T' else -1),speakers)
                summary[f'{g}/{m}/{c["name"]}/both_shifted_T_minus_N']=old.stat(v[:,1]-v[:,0],speakers)
            summary[f'{g}/{m}/sign_specificity/Tminus_minus_Tplus']=old.stat(cm[:,gi,1,1,mi]-cm[:,gi,2,1,mi],speakers)
            summary[f'{g}/{m}/sign_specificity/Nplus_minus_Nminus']=old.stat(cm[:,gi,2,0,mi]-cm[:,gi,1,0,mi],speakers)
            qmean=cm[:,gi,9:,:,mi].mean(1)
            for ai,a in enumerate('NT'):
                ci=1 if a=='T' else 2
                summary[f'{g}/{m}/main_minus_meanQ/{a}']=old.stat(cm[:,gi,ci,ai,mi]-qmean[:,ai],speakers)
    write(OUT/(stem+'_summary.json'),summary)
def execute():
    idx=locked();assert read(OUT/'bridge.json')['status']=='PASS';fitinfo=read(OUT/'fit.json');assert sha(OUT/'fit.npz')==fitinfo['fit_sha256']
    with np.load(OUT/'fit.npz') as f:models=f['raw_delta'];counts=f['speaker_counts']
    with np.load(OUT/'Q.npz') as f:Q={k:f[k] for k in f.files}
    checks=dict(positive_vector=0.,same_phone_vector=0.,same_phone_distance=0.,squared_expansion=0.,odd_even=0.,Q_pairwise_squared=0.,positive_distance=0.)
    for fi in range(len(idx['folds'])):
        for gi in range(2):
            eta=(models[fi,gi,:-1]-models[fi,gi,-1])*(counts[fi]/(counts[fi]+4))[:,None];eta=np.vstack([eta,np.zeros((1,1024))]);d=eta[:,None]-eta[None,:];norm=(d*d).sum(-1)
            for i in range(16):
                e=eta[:,Q['permutation'][i]]*Q['sign'][i];dd=e[:,None]-e[None,:];checks['Q_pairwise_squared']=max(checks['Q_pairwise_squared'],float(abs((dd*dd).sum(-1)-norm).max()))
    assert checks['Q_pairwise_squared']<=1e-10
    result=np.zeros((1411,2,len(SPECS),2,4))
    for r,s in zip(idx['eval_rows'],idx['spans']):
        resource(10*2**20);arrays=load_clip(r['id'])
        for gi in range(2):
            f=geom(arrays,gi)
            for ci,spec in enumerate(SPECS):
                for ai,a in enumerate('NT'):
                    field=field_model(idx,models,counts,r['speaker'],gi,spec,a,Q)
                    result[s['start']:s['stop'],gi,ci,ai]=score_row(r,f,a,field,checks)
        print('scored',r['id'],flush=True)
    with np.load(OUT/'baseline.npz') as f:assert float(abs(result[:,:,0]-f['metrics']).max())<=1e-9
    checks['positive_distance']=float(abs(result[:,:,:,:,0]-result[:,:,0:1,:,0]).max())
    assert checks['positive_vector']<=1e-12 and checks['same_phone_vector']<=1e-12 and checks['positive_distance']<=1e-10 and checks['same_phone_distance']<=1e-10
    assert checks['squared_expansion']<=1e-9 and checks['odd_even']<=1e-9
    save(OUT/'query_metrics.npz',metrics=result);analyze(result,idx,'field')
    write(OUT/'execution.json',{'status':'SCORED_PENDING_INDEPENDENT','created_epoch':time.time(),'checks':checks,'protocol_sha256':sha(OUT/'protocol.json'),'fit_sha256':fitinfo['fit_sha256'],'shape':list(result.shape)})
    print('SCORED',checks,flush=True)
def phase():
    idx=locked();assert read(OUT/'execution.json')['status']=='SCORED_PENDING_INDEPENDENT'
    binding=read(OUT/'phase_binding.json');review=read(OUT/'phase_review.json')
    assert binding['status']=='PASS' and binding['protocol_sha256']==sha(OUT/'protocol.json')
    assert review['status']=='PASS' and review['phase_binding_sha256']==sha(OUT/'phase_binding.json')
    for p,h in binding['inputs'].items():assert sha(p)==h,p
    producer=read(binding['producer_binding_path'])
    assert producer['field_protocol_sha256']==sha(OUT/'protocol.json')
    assert producer['field_design_sha256']==sha(OUT/'protocol.md')
    assert producer['created_epoch']<read(binding['phase_execution_lock_path'])['created_epoch']
    with np.load(OUT/'fit.npz') as f:models=f['raw_delta'];counts=f['speaker_counts']
    assert sha(OUT/'fit.npz')==read(OUT/'fit.json')['fit_sha256']
    result=np.zeros((1411,2,2,4,2,4));positive_error=0.
    for r,s in zip(idx['eval_rows'],idx['spans']):
        original=load_clip(r['id']);altered=load_clip(r['id'],binding['features'])
        for a in 'NT':
            entry=binding['features'][r['id']][a]
            assert entry['path'] in binding['inputs']
            assert original[a,'A'].shape==altered[a,'A'].shape
            assert np.isfinite(altered[a,'A']).all() and (np.linalg.norm(altered[a,'A'],axis=1)>0).all()
        for gi in range(2):
            backgrounds=[geom(original,gi),geom(altered,gi)]
            for di,target in enumerate(['T','N']):
                spec=SPECS[1 if target=='T' else 2]
                for pi,f in enumerate(backgrounds):
                    for hasfield in [0,1]:
                        ci=2*pi+hasfield
                        for ai,a in enumerate('NT'):
                            field=field_model(idx,models,counts,r['speaker'],gi,spec,a,None) if hasfield and a==target else None
                            result[s['start']:s['stop'],gi,di,ci,ai]=score_row(r,f,a,field)
    for pi in [0,2]:positive_error=max(positive_error,float(abs(result[:,:,:,pi+1,:,0]-result[:,:,:,pi,:,0]).max()))
    assert positive_error<=1e-10
    with np.load(OUT/'baseline.npz') as f:assert float(abs(result[:,:,:,0]-f['metrics'][:,:,None]).max())<=1e-9
    save(OUT/'phase_query_metrics.npz',metrics=result);cm=old.clipmeans(result,idx['spans']);save(OUT/'phase_clip_metrics.npz',metrics=cm)
    speakers=[s['speaker'] for s in idx['spans']];summary={};closure=0.
    for gi,g in enumerate(GEOMS):
        for di,target in enumerate(['T','N']):
            for mi,m in enumerate(METRICS):
                val=cm[:,gi,di,:,:,mi]
                series={name:val[:,ci] for ci,name in enumerate(['identity','field','phase','both'])}
                series.update(field_effect=val[:,1]-val[:,0],phase_effect=val[:,2]-val[:,0],interaction=val[:,3]-val[:,1]-val[:,2]+val[:,0],total=val[:,3]-val[:,0])
                closure=max(closure,float(abs(series['field_effect']+series['phase_effect']+series['interaction']-series['total']).max()))
                for name,x in series.items():
                    for ai,a in enumerate('NT'):summary[f'{g}/{m}/only_field_{target}/{name}/{a}']=old.stat(x[:,ai],speakers)
                    summary[f'{g}/{m}/only_field_{target}/{name}/T_minus_N']=old.stat(x[:,1]-x[:,0],speakers)
    assert closure<=1e-10;write(OUT/'phase_summary.json',summary)
    write(OUT/'phase_execution.json',{'status':'SCORED_PENDING_INDEPENDENT','created_epoch':time.time(),'positive_field_difference':positive_error,'closure_max':closure,'phase_binding_sha256':sha(OUT/'phase_binding.json'),'fit_sha256':sha(OUT/'fit.npz')})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','bridge','fit','execute','phase']);globals()[parser.parse_args().stage]()
