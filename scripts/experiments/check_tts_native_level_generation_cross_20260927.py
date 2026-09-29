"""Independent NumPy distance, four-cell contrast, and bootstrap audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[2]/'runs/tts_native_level_generation_cross_20260927'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_matrix(v,a):
    # Explicit per-lag valid and zero-padded audio, float64 norm of float32 eps offset.
    v=np.asarray(v,dtype=np.float32);a=np.asarray(a,dtype=np.float32);n=len(v);out=np.empty((n,31))
    for column,lag in enumerate(range(-15,16)):
        partner=np.zeros_like(a);i=np.arange(n);valid=(i+lag>=0)&(i+lag<n);partner[valid]=a[i[valid]+lag]
        difference=(v-partner+np.float32(1e-6)).astype(np.float64)
        out[:,column]=np.sqrt(np.einsum('ij,ij->i',difference,difference))
    return out

def independent_stats(values,groups):
    names=sorted(set(groups));means=[]
    for name in names:means.append(sum(float(v) for v,g in zip(values,groups) if g==name)/sum(g==name for g in groups))
    means=np.array(means);rng=np.random.Generator(np.random.PCG64(20260926));draws=rng.integers(0,len(names),(20000,len(names)));boot=np.sum(means[draws],axis=1)/len(names)
    return float(sum(means)/len(means)),np.percentile(boot,[.5,99.5])

def check(split='all'):
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip();support=read(OUT/'support.json');rows=[r for r in support if split=='all' or r['split']==split]
    checked=0;maximum_distance=0.;maximum_metric=0.;closure=0.;metric_count=0
    for row in rows:
        sid=row['id'];scores=read(OUT/'scores'/(sid+'.json'));mp=OUT/'scores'/(sid+'.npz');assert sha(mp)==scores['matrix_sha256'];saved=np.load(mp)
        for arm in ['N','T']:
            L=row['arms'][arm]['joint_L'];features={c:np.load(OUT/'features'/sid/arm/(c+'.npz')) for c in p['conditions']}
            for geom in ['raw','unit']:
                for key,cell in scores['cells'][arm][geom].items():
                    if key=='historical':continue
                    vc,ac=key.split('__');v=features[vc]['visual'][:L];a=features[ac]['audio'][:L]
                    if geom=='unit':v=(v.astype(float)/np.sqrt(np.sum(v.astype(float)**2,axis=1))[:,None]).astype(np.float32);a=(a.astype(float)/np.sqrt(np.sum(a.astype(float)**2,axis=1))[:,None]).astype(np.float32)
                    expected=independent_matrix(v,a);actual=saved[arm+'__'+geom+'__'+key];err=float(abs(expected-actual).max());maximum_distance=max(maximum_distance,err);assert err<1e-5;checked+=expected.size
                    for policy,metrics in cell['policies'].items():
                        if metrics is None:assert L<=40 and policy=='guard20';continue
                        if policy=='guard20':curve=expected[20:L-20].mean(0)
                        elif policy=='guard0':curve=expected.mean(0)
                        else:curve=np.array([expected[max(0,-lag):min(L,L-lag),lag+15].mean() for lag in range(-15,16)])
                        B=float(np.median(curve));D=float(curve.min());values={'C':B-D,'B':B,'D':D,'C_anchor':B-float(curve[18]),'D_anchor':float(curve[18])}
                        maximum_metric=max(maximum_metric,max(abs(value-metrics[k]) for k,value in values.items()));metric_count+=len(values)
        print('checked',sid,flush=True)
    result={'status':'passed','split':split,'distance_entries':checked,'metric_entries':metric_count,'max_distance_error':maximum_distance,'max_metric_error':maximum_metric}
    if split=='all':
        summary=read(OUT/'summary.json');effects=read(OUT/'effects.json');buckets={};err=0.;closure=0.
        for r in effects:
            closure=max(closure,abs(r['G']+r['E']+r['I']-r['total']))
            assert abs(r['G']-(r['q10']-r['q00']))<1e-12
            assert abs(r['E']-(r['q01']-r['q00']))<1e-12
            assert abs(r['I']-(r['q11']-r['q10']-r['q01']+r['q00']))<1e-12
            key=r['geometry']+'/'+r['policy']+'/'+r['support'],r['metric'],r['condition']
            buckets.setdefault(key,{}).setdefault(r['id'],{})[r['arm']]=r
        def compare(values,speakers,stored):
            mean,ci=independent_stats(values,speakers)
            return max(abs(mean-stored['mean']),float(abs(ci-stored['ci99']).max()))
        for view,block in summary['results'].items():
            for metric,detail in block.items():
                for h in ['EQ','LEVEL','EQ_LEVEL']:
                    data=buckets[view,metric,h];pairs=list(data.values());speakers=[d['N']['speaker'] for d in pairs]
                    baseline=[d['T']['q00']-d['N']['q00'] for d in pairs]
                    err=max(err,compare(baseline,speakers,detail['baseline_T_minus_N']))
                    err=max(err,compare([d['T']['q11']-d['N']['q11'] for d in pairs],speakers,detail['conditions'][h]['processed_T_minus_N']))
                    for component in ['G','E','I','total']:
                        for arm in ['N','T','T_minus_N']:
                            values=[d[arm][component] if arm!='T_minus_N' else d['T'][component]-d['N'][component] for d in pairs]
                            err=max(err,compare(values,speakers,detail['conditions'][h][component][arm]))
                for label, weights in {'EQxLEVEL':{'EQ_LEVEL':1,'EQ':-1,'LEVEL':-1}, 'EQ_LEVEL_minus_EQ':{'EQ_LEVEL':1,'EQ':-1}, 'LEVEL_minus_identity':{'LEVEL':1}}.items():
                    data={h:buckets[view,metric,h] for h in weights};ids=list(next(iter(data.values())));speakers=[data[next(iter(data))][s]['N']['speaker'] for s in ids]
                    for component in ['G','E','I','total']:
                        n=np.array([sum(w*data[h][s]['N'][component] for h,w in weights.items()) for s in ids]);t=np.array([sum(w*data[h][s]['T'][component] for h,w in weights.items()) for s in ids])
                        for arm, values in [('N',n),('T',t),('T_minus_N',t-n)]:err=max(err,compare(values,speakers,detail['contrasts'][label][component][arm]))
        assert err<1e-10 and closure<1e-12
        result.update(max_statistics_error=err,max_fourcell_closure=closure)
    assert maximum_metric < 1e-5
    result['checker_sha256']=sha(__file__)
    (OUT/('independent_validation_'+split+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def acoustic():
    import types
    import soundfile as sf
    path=OUT.parents[1]/'runs/tts_native_level_match_calibration_20260927/code_snapshot/check_tts_native_level_match_calibration.py'
    m=types.ModuleType('independent_scalar_fft');m.__file__=str(OUT.parents[1]/'scripts/experiments/check_tts_native_level_match_calibration.py')
    exec(compile(path.read_text(),str(path),'exec'),m.__dict__)
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for f,h in {**p['dependencies'],**p['new_code_hashes']}.items():assert sha(f)==h
    pairs={r['id']:r for r in read(OUT/'pairs.json')};supports=read(OUT/'support.json');maximum={};gates={}
    def bound(key,x,limit):
        maximum[key]=max(maximum.get(key,0.),float(x));gates[key]=gates.get(key,True) and bool(x<=limit)
    for row in supports:
        sid=row['id'];src=next(r for r in p['rows'] if r['id']==sid);xs={}
        for arm,s in src['audio'].items():
            assert sha(s['path'])==s['sha256'];pcm,sr=sf.read(s['path'],dtype='int16');assert sr==16000 and sf.info(s['path']).subtype=='PCM_16'
            xs[arm]=pcm.astype(float)/32768
        shapes={a:m.features(x)[0] for a,x in xs.items()};mid=(shapes['N']+shapes['T'])/2
        eqs={a:m.reconstruct_eq(x,np.clip(mid-shapes[a],-6,6)) for a,x in xs.items()};rs={a:m.energy(x) for a,x in xs.items()};assert min(rs.values())>1e-8
        target=min(np.sqrt(rs['N']*rs['T']),min(.98*rs[a]/abs(y).max() for a in xs for y in [xs[a],eqs[a]]))
        bound('target',abs(target-pairs[sid]['m']),1e-12);rmses={}
        for arm,x in xs.items():
            gain=target/rs[arm];bound('gain',abs(gain-pairs[sid]['gain'][arm]),1e-12)
            for cond in p['conditions']:
                info=row['arms'][arm][cond];assert sha(info['waveform'])==info['sha256'] and sha(info['frontend'])==info['frontend_sha256']
                saved,sr=sf.read(info['waveform'],dtype='float64');assert sr==16000 and len(saved)==len(x) and np.isfinite(saved).all() and abs(saved).max()<1
                expected={'raw':x,'identity':x,'EQ':eqs[arm],'LEVEL':x*gain,'EQ_LEVEL':eqs[arm]*gain}[cond]
                bound('waveform',abs(saved-expected).max(),1e-7);rmses[arm,cond]=m.energy(saved)
                if cond in ['raw','identity']:assert np.array_equal(saved,x)
                if cond in ['LEVEL','EQ_LEVEL']:
                    original=x if cond=='LEVEL' else eqs[arm];f0=m.features(original);fs=m.features(saved);fi=m.features(expected)
                    bound('saved_target_relative',abs(m.energy(saved)/target-1),1e-5)
                    bound('saved_recover',abs(saved/gain-original).max(),1e-7)
                    bound('saved_shape',abs(fs[0]-f0[0]).max(),1e-4);bound('saved_R',abs(fs[1]-f0[1]),1e-5)
                    bound('float64_shape',abs(fi[0]-f0[0]).max(),1e-8);bound('float64_R',abs(fi[1]-f0[1]),1e-8)
                    normalized=saved/np.linalg.norm(saved)-original/np.linalg.norm(original);bound('cosine_error',.5*np.sum(normalized**2),1e-10)
                    if cond=='LEVEL':assert np.all(saved[x==0]==0)
        for cond in ['LEVEL','EQ_LEVEL']:bound('saved_pair_relative',abs(rmses['N',cond]-rmses['T',cond])/target,1e-5)
        print('acoustic checked',sid,flush=True)
    result={'status':'PASS' if all(gates.values()) else 'FAIL','max_errors':maximum,'gates':gates,'pairs':len(supports),'waveforms':len(supports)*10,'checker_sha256':sha(__file__),'method':'Independent explicit FFT/OLA scalar reconstruction from original PCM; no primary transform imported'}
    (OUT/'independent_acoustic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def hashes():
    p=read(OUT/'protocol.json');seal=read(OUT/'cpu_seal.json');count=0
    for f,h in {**p['dependencies'],**seal['code']}.items():assert sha(f)==h;count+=1
    assert sha(OUT/'support.json')==seal['support_sha256']
    for row in read(OUT/'support.json'):
        for a in ['N','T']:
            for c in p['conditions']:
                info=row['arms'][a][c];assert sha(info['waveform'])==info['sha256'] and sha(info['frontend'])==info['frontend_sha256'];count+=2
                meta=read(OUT/'features'/row['id']/a/(c+'.json'));assert sha(meta['features'])==meta['sha256'] and sha(meta['video']['path'])==meta['video']['sha256'];count+=2
    reuse=read(OUT/'reuse_manifest.json')
    for row in reuse['rows']:
        if row['reused']:assert sha(row['source_metadata'])==row['source_metadata_sha256'];count+=1
    result={'status':'PASS','checked':count,'reuse_count':reuse['reuse_count'],'repeat_exact':read(OUT/'reuse_repeat_validation.json')['passed']}
    assert result['repeat_exact'];(OUT/'independent_hashes.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['acoustic','scores','hashes'],default='scores');parser.add_argument('--split',choices=['all','calibration','evaluation'],default='all');args=parser.parse_args()
    if args.stage=='scores':check(args.split)
    else:globals()[args.stage]()
