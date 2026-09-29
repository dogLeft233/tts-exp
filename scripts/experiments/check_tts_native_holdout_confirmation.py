"""Independent numerical and artifact audit for the frozen holdout; CPU only."""
from pathlib import Path
import hashlib,json
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_native_holdout_confirmation_20260926'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(curve,k):
    j=np.argmin(curve);b=float(np.median(curve));d=float(curve[j]);a=float(curve[k+15]);return {'C':b-d,'B':b,'D':d,'D_anchor':a,'C_anchor':b-a,'best_lag':float(j-15)}
def direct(v,a,rows,lags):
    # Independent float64 vector differences with eps; score worker uses Torch float32.
    return np.stack([np.sqrt(np.sum((v[rows].astype(float)-a[rows+s].astype(float)+1e-6)**2,axis=1)) for s in lags],axis=1)
def altered(v,a,k,alpha):
    n=len(v);J=np.arange(max(0,-k),min(n,n-k));I=np.arange(20,n-20)
    v,a=v.astype(float),a.astype(float);M=(v[J]+a[J+k])/2;R=(a[J+k]-v[J])/2
    mu=M[I-J[0]].mean(0);sigma=np.sqrt(np.mean(np.sum((M[I-J[0]]-mu)**2,axis=1)))
    mp=mu+alpha*(M-mu);vp=v.copy();ap=a.copy();vp[J]=mp-R;ap[J+k]=mp+R
    sigma_after=np.sqrt(np.mean(np.sum((mp[I-J[0]]-mu)**2,axis=1)))
    return vp,ap,{'sigma':float(sigma),'sigma_after':float(sigma_after),'anchor64':float(np.max(abs((ap[J+k]-vp[J])-(a[J+k]-v[J])))),'mu_error':float(abs(mp[I-J[0]].mean(0)-mu).max())}

def main():
    p=read(OUT/'protocol.json');lock=read(OUT/'score_lock.json')
    assert sha(OUT/'protocol.json')==lock['protocol_sha256']
    assert sha(ROOT/'scripts/experiments/tts_native_holdout_confirmation.py')==lock['analysis_code_sha256']
    for path,h in read(OUT/'permutation_seal.json').items():assert sha(path)==h,path
    records=read(OUT/'feature_records.json');rr={(r['id'],r['arm']):r for r in records};errors={'native_policy':0.,'M_score':0.,'anchor64':0.,'mu':0.,'sigma_matching':0.,'inverse':0.,'constant_probe':0.,'permutation_curves':0.,'bootstrap':0.};compared=0
    for r in records:
        dat=np.load(r['path']);L=r['joint_L'];v,a=dat['visual'][:L],dat['audio'][:L];wav=sf.info(r['audio']['path']);assert L==min(r['visual_metadata']['frame_count'],wav.frames//640)-5
        stored=read(OUT/'scores/native'/r['id']/(r['arm']+'.json'))
        for policy,metrics in stored['policies'].items():
            curves=[]
            for s in range(-15,16):
                if policy=='valid':I=np.arange(max(0,-s),min(L,L-s));aa=a[I+s]
                else:
                    g=int(policy[5:]);I=np.arange(g,L-g) if g else np.arange(L);j=I+s;aa=np.zeros_like(v[I]);valid=(j>=0)&(j<L);aa[valid]=a[j[valid]]
                curves.append(np.sqrt(np.sum((v[I].astype(float)-aa.astype(float)+1e-6)**2,axis=1)).mean())
            sm=score(np.array(curves),3)
            for f in sm:
                if f=='best_lag':assert sm[f]==metrics[f]
                else:errors['native_policy']=max(errors['native_policy'],abs(sm[f]-metrics[f]))
    for dest in sorted((OUT/'scores/pairs').glob('*/*/Q[12].json')):
        rec=read(dest);sid,ta,geom=rec['id'],rec['T_arm'],rec['geometry'];original={}
        for arm in ('N',ta):
            z=rr[sid,arm];dat=np.load(z['path']);v,a=dat['visual'][:z['joint_L']],dat['audio'][:z['joint_L']]
            if geom=='unit':v=(v.astype(float)/np.linalg.norm(v.astype(float),axis=1)[:,None]).astype(np.float32);a=(a.astype(float)/np.linalg.norm(a.astype(float),axis=1)[:,None]).astype(np.float32)
            original[arm]=(v,a)
        for k in (2,3,4):
            result=rec['k_results'][str(k)];alpha=result['alpha_T2N']
            for label,arm,scale in [('N_base','N',1.),('T_base',ta,1.),('N_M','N',1/alpha),('T_M',ta,alpha)]:
                v,a=original[arm];vp,ap,control=altered(v,a,k,scale);errors['anchor64']=max(errors['anchor64'],control['anchor64']);errors['mu']=max(errors['mu'],control['mu_error']);errors['sigma_matching']=max(errors['sigma_matching'],abs(control['sigma_after']-scale*control['sigma']))
                inverse=altered(vp,ap,k,1/scale);errors['inverse']=max(errors['inverse'],float(abs(inverse[0]-v).max()),float(abs(inverse[1]-a).max()))
                L=len(v);I=np.arange(20,L-20);vp=vp.astype(np.float32);ap=ap.astype(np.float32);dm=direct(vp,ap,I,np.arange(-15,16));sm=score(dm.mean(0),k);compared+=dm.size
                for f in sm:
                    if f=='best_lag':assert sm[f]==result['scores'][label][f]
                    else:errors['M_score']=max(errors['M_score'],abs(sm[f]-result['scores'][label][f]))
                # Deterministic constant-shift identity on every transformed cell.
                cc=np.linspace(-.25,.25,vp.shape[1]);errs=np.sqrt(np.sum((vp[I[:3]].astype(float)+cc-(ap[I[:3]+k].astype(float)+cc)+1e-6)**2,axis=1))-dm[:3,k+15];errors['constant_probe']=max(errors['constant_probe'],float(abs(errs).max()))
                if k!=3:continue
                pp=np.load(OUT/'permutations'/sid/(arm+'.npz'));pi=pp['whole'];J=pp['J'];region=pp['region'];assert pi.shape==(256,L);assert np.all(region[pi[:,J]]==region[J]);assert np.all(np.sort(pi[:,I],axis=1)==I)
                stored=np.load(dest.with_suffix('.npz'))[label]
                # Complete first and last predetermined repeat, directly from vectors.
                for b in (0,255):
                    curves=[]
                    for s in range(-15,16):
                        left=vp[pi[b,I]].astype(float);right=ap[pi[b,I+s-k]+k].astype(float)
                        curves.append(np.sqrt(np.sum((left-right+1e-6)**2,axis=1)).mean())
                    errors['permutation_curves']=max(errors['permutation_curves'],float(abs(np.array(curves)-stored[b]).max()))
    # Independently aggregate each recorded native effect and bootstrap speaker clusters.
    native=read(OUT/'native_summary.json');mapping={r['id']:r['speaker'] for r in p['rows']}
    for support,policies in native.items():
        for policy,result in policies.items():
            ids=result['ids'];speakers=sorted({mapping[s] for s in ids})
            if not ids:continue
            for field in ('C','B','D','D_anchor','C_anchor'):
                grouped={s:[] for s in speakers}
                for sid in ids:
                    x={a:read(OUT/'scores/native'/sid/(a+'.json'))['policies'][policy][field] for a in ('N','Q1','Q2')};grouped[mapping[sid]].append((x['Q1']+x['Q2'])/2-x['N'])
                y=np.array([np.mean(grouped[s]) for s in speakers]);expected=result['contrasts'][field];errors['bootstrap']=max(errors['bootstrap'],abs(y.mean()-expected['mean']))
                if len(y)>1:
                    draws=np.random.Generator(np.random.PCG64(20260926)).integers(len(y),size=(20000,len(y)));ci=np.quantile(y[draws].mean(1),[.005,.995]);errors['bootstrap']=max(errors['bootstrap'],float(abs(ci-expected['ci99']).max()))
    assert errors['native_policy']<2e-5 and errors['M_score']<2e-5 and errors['permutation_curves']<2e-5
    assert max(errors[k] for k in ('anchor64','mu','sigma_matching','inverse','constant_probe','bootstrap'))<1e-8
    receipt={'passed':True,'errors':errors,'independent_direct_distance_count':compared,'native_cells':len(records),'pairs':len(list((OUT/'scores/pairs').glob('*/*/Q[12].json'))),'permutation_repeats_recomputed':[0,255],'all256_permutation_regions_checked':True,'score_lock_sha256':sha(OUT/'score_lock.json'),'validator_sha256':sha(__file__)}
    (OUT/'independent_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
