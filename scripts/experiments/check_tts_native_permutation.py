"""Independent vector probes, permutation/scalar replay and paired bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_native_permutation_20260926'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(d):
    with np.load(d['path']) as z:a=z[d['key']]
    return a[d.get('start',0):]

errors={'vectors':0.,'scalar_curves':0.,'statistics':0.,'coefficients':0.}
def close(a,b,key,tol=1e-10):
    error=float(np.max(abs(np.array(a)-np.array(b))));errors[key]=max(errors[key],error)
    assert error<tol,(key,error)


def metrics(z,k):
    rows=[]
    for x in z:
        b=sorted(x)[15];j=int(np.argmin(x));d=x[j];a=x[k+15]
        rows.append([b-d,b,d,a,b-a,j-15])
    return np.array(rows)


def bootstrap(values,speakers,expected):
    names=sorted(set(speakers));v=np.array([np.mean([x for x,s in zip(values,speakers) if s==n]) for n in names])
    close(v.mean(),expected['mean'],'statistics')
    assert len(values)==expected['n'] and len(names)==expected['speakers']
    ix=np.random.Generator(np.random.PCG64(20260926)).integers(len(names),size=(20000,len(names)))
    counts=np.stack([(ix==j).sum(1) for j in range(len(names))],axis=1)/len(names)
    b=counts@v
    close(np.percentile(b,[.5,99.5]),expected['ci99'],'statistics')
    close(np.percentile(b,[2.5,97.5]),expected['ci95'],'statistics')


def main(repeats):
    p=read(OUT/'protocol.json');seals=read(OUT/'permutation_seal.json');modal_multisets=0
    for file,h in seals.items():assert sha(file)==h,file
    for receipt in p['records']:
        with np.load(receipt['path']) as z:
            domain=z['domain'];region=z['region'];phone=z['phone_labels'] if 'phone_labels' in z else np.load(OUT/'label_vectors'/(receipt['key']+'.npy'));n=receipt['n'];k=receipt['k0']
            i=np.arange(20,n-20);q=i[:,None]+np.arange(-15,16)-k
            for mode in receipt['modes']:
                pi=z[mode]
                assert np.all(np.sort(pi[:,domain],axis=1)==domain)
                assert np.all(np.sort(pi[:,i],axis=1)==i)
                assert np.all(region[pi[:,domain]]==region[domain])
                if mode=='phone':
                    assert np.all(phone[pi[:,domain]]==phone[domain])
                    for b in (0,31,63,255):assert np.array_equal(phone[pi[b,q]],phone[q])
                modal_multisets+=len(pi)
    vector_probes=scalar_points=0;pair_scores={}
    files=sorted((OUT/'matrices').glob('*/*/*/*/receipt.json'))
    for fi,rp in enumerate(files):
        info=read(rp);group,geom,sid,ta=info['group'],info['geometry'],info['id'],info['T_arm'];k=info['k0']
        g=p['parent']['groups'][group];rows={(r['id'],r['arm']):r for r in g['records']};coords={}
        for arm in ('N',ta):
            r=rows[sid,arm];n=r['joint_L'];v,a=load(r['visual'])[:n],load(r['audio'])[:n]
            if geom=='unit':
                v=(v.astype(float)/np.sqrt((v.astype(float)**2).sum(1))[:,None]).astype(np.float32)
                a=(a.astype(float)/np.sqrt((a.astype(float)**2).sum(1))[:,None]).astype(np.float32)
            v,a=v.astype(float),a.astype(float);t=np.arange(max(0,-k),min(n,n-k));i=np.arange(20,n-20)
            m=(v[i]+a[i+k])/2;mu=m.mean(0);sigma=np.sqrt(np.mean(((m-mu)**2).sum(1)))
            coords[arm]=(v,a,t,i,mu,sigma)
        ratio=coords['N'][5]/coords[ta][5]
        score_path=OUT/'scores'/str(repeats)/group/geom/sid/f'{ta}.npz'
        with np.load(score_path) as saved:
            pair_scores[group,geom,sid,ta]={key:metrics(saved[key],k) for key in saved.files}
            for label,meta in info['labels'].items():
                arm=meta['arm'];v,a,t,i,mu,sigma=coords[arm];alpha=1. if label.endswith('base') else (ratio if arm==ta else 1/ratio)
                close(alpha,meta['alpha'],'coefficients')
                vv=.5*(alpha+1)*v[t]+.5*(alpha-1)*a[t+k]+(1-alpha)*mu
                aa=.5*(alpha-1)*v[t]+.5*(alpha+1)*a[t+k]+(1-alpha)*mu
                vf,af=vv.astype(np.float32),aa.astype(np.float32)
                dp=rp.parent/f'{label}.npy';assert sha(dp)==meta['matrix_sha256']
                d=np.load(dp,mmap_mode='r')
                with np.load(rp.parent/f'{label}_probes.npz') as probes:
                    ix=probes['indices'];diff=vf[ix[:,0]]-af[ix[:,1]]+np.float32(1e-6)
                    direct=np.sqrt((diff*diff).sum(1,dtype=np.float32))
                    close(direct,probes['values'],'vectors',tol=2e-5);vector_probes+=len(ix)
                pp=np.load(OUT/'permutations'/group/sid/f'{arm}.npz');lo=meta['lo'];diag=d[i-lo,i-lo]
                q=i[:,None]+np.arange(-15,16)-k
                for mode in ('whole','phone'):
                    if label+'_'+mode not in saved:continue
                    pi=pp[mode]
                    for b in (0,repeats-1):
                        curve=np.array([sum(float(d[int(pi[b,ii])-lo,int(pi[b,ii+s-k])-lo]) for ii in i)/len(i) for s in range(-15,16)])
                        close(curve,saved[label+'_'+mode][b],'scalar_curves')
                        scalar_points+=len(i)*31
                        assert np.array_equal(np.sort(d[pi[b,i]-lo,pi[b,i]-lo]),np.sort(diag))
                del d
        if (fi+1)%100==0:print('independently checked',fi+1,'pairs',flush=True)
    summary=read(OUT/f'summary_{repeats}.json');mc={}
    for key,result in summary.items():
        group,geom,support,mode=key.split('/');g=p['parent']['groups'][group];ids=result['ids']
        speakers=[next(r['speaker'] for r in g['records'] if r['id']==sid) for sid in ids]
        contrasts={sid:[] for sid in ids}
        for sid in ids:
            for arm in g['arms'][1:]:
                data=pair_scores[group,geom,sid,arm];v={k:x.mean(0) for k,x in data.items() if k.endswith('_identity') or k.endswith('_'+mode)}
                a=v['T_base_identity']-v['N_base_identity'];b=v['T_base_'+mode]-v['N_base_'+mode]
                c=v['T_M_identity']-v['N_base_identity'];d=v['T_M_'+mode]-v['N_base_'+mode]
                e=v['T_base_identity']-v['N_M_identity'];f=v['T_base_'+mode]-v['N_M_'+mode]
                contrasts[sid].append({'original':a,'permuted':b,'permutation_change':b-a,'M_only':c,'M_permuted':d,'interaction':d-c-b+a,
                    'M_effect_after_permutation':d-b,'reverse_M_only':e,'reverse_M_permuted':f,'reverse_interaction':f-e-b+a,
                    'N_permutation_change':v['N_base_'+mode]-v['N_base_identity'],'T_permutation_change':v['T_base_'+mode]-v['T_base_identity']})
        for contrast,fields in result['contrasts'].items():
            for j,field in enumerate(('C','B','D','D_anchor','C_anchor','best_lag')):
                values=[np.mean([x[contrast][j] for x in contrasts[sid]]) for sid in ids]
                bootstrap(values,speakers,fields[field])
    for file,h in read(OUT/'annotation_hashes.json').items():assert sha(file)==h
    result={'status':'PASS','pair_geometries':len(files),'vector_probes':vector_probes,'scalar_distance_cells':scalar_points,
            'permutation_multiset_checks':modal_multisets,'max_errors':errors,'repeats':repeats,
            'method':'Independent expanded vectors/float32 norm probes; integer bijections establish modal and M/R multiset preservation; scalar first/last-repeat complete lag curves; independent median, native contrasts and speaker bootstrap multiplicities'}
    (OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,required=True);main(parser.parse_args().repeats)
