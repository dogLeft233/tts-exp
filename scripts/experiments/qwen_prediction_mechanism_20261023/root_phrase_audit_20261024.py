"""Parent independent reconstruction from official arrays and saved FA boundaries."""
from pathlib import Path
import json, hashlib, pickle
import numpy as np

R=Path('/home/wjj/tts-audio/tts-exp')
P=R/'runs/qwen_phrase_local_20261024_attempt02'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, why):
    if not ok: raise AssertionError(why)

def main():
    manifest=read(P/'input_manifest.json')['audio']
    aligned={a['id']:a for a in read(P/'alignment.json')['audio']}
    measured=read(P/'local_metrics.json'); rebuilt={}; max_error=0.; nvalues=0
    require(len(manifest)==28 and len(aligned)==28,'28 bindings')
    variants={'nominal':(0,0),'left':(-.04,-.04),'right':(.04,.04),'contract':(.04,-.04),'expand':(-.04,.04)}
    for item in manifest:
        ident=item['id']; require(sha(item['source'])==item['sha256'],'audio hash')
        require(sha(item['activesd_path'])==item['activesd_sha256'],'matrix hash')
        require(sha(item['score_path'])==item['score_sha256'],'score hash')
        t=item['tracks'][0]; x=np.asarray(pickle.loads(Path(item['activesd_path']).read_bytes())[0],dtype=float)
        require(x.shape==tuple(t['array_shape']),'array shape')
        mu=x.mean(0); best=int(mu.argmin()); require(best==t['fixed_best_column'],'global best lag')
        require(abs(np.median(mu)-mu[best]-t['official_C'])<=.00051,'global C bridge')
        require(abs(mu[best]-t['official_D'])<=.00051,'global D bridge')
        frames=np.array(t['frames']); base=t['audio_binding']['crop_zero_canonical_s']; n=len(x)
        good=[]; vr={}; ar={}
        for i in range(n):
            j=np.arange(i-15,i+16)
            starts=j*.04-np.where(j>0,1/16000,0); ends=j*.04+.215
            v0=frames[i]/25; v1=(frames[i+4]+1)/25
            valid=(j.min()>=0 and j.max()<n and starts.min()>=0 and ends.max()<=t['crop_audio_samples']/16000
                and base+starts.min()>=0 and base+ends.max()<=item['canonical_audio_samples']/16000
                and all(t['crop_source_frames_valid'][i:i+5]))
            jj=i+best-15
            vr[i]=(v0,v1); ar[i]=(base+jj*.04-(1/16000 if jj>0 else 0),base+jj*.04+.215)
            if valid: good.append(i)
        for m in [m for m in measured if m['id']==ident]:
            c0,c1=m['character_span']; times=aligned[ident]['alignment'][0]['timestamp']
            bounds=np.array([times[c0][0],times[c1-1][1]])/1000
            require(np.max(abs(bounds-np.array(m['nominal_bounds'])))<1e-12,'FA character boundary')
            bounds+=np.array(variants[m['variant']]); bounds[0]=max(0,bounds[0]);bounds[1]=min(item['canonical_audio_samples']/16000,bounds[1])
            require(np.max(abs(bounds-np.array(m['bounds'])))<1e-12,'boundary sensitivity')
            use=[i for i in good if min(vr[i][0],ar[i][0])>=bounds[0] and max(vr[i][1],ar[i][1])<=bounds[1]]
            require(use==m['valid_rows'],'independent support rows '+ident)
            require(len(use)==m['window_count'],'support count')
            key=(ident,m['phrase_index'],m['variant'])
            if len(use)<5:
                require(m['contrast'] is None,'under-supported must be NA');rebuilt[key]=None;continue
            z=x[use].mean(0);a=float(z[best]);b=float(np.median(z));v={'contrast':b-a,'matched_distance':a,'median_background':b}
            for k,y in v.items():
                err=abs(y-m[k]);max_error=max(max_error,err);nvalues+=1;require(err<1e-10,'metric reconstruction')
            rebuilt[key]=v
    checks=0
    for row in read(P/'comparisons.json'):
        aa,bb=row['comparison'].split('-');sid=row['sample_id'];pi=row['phrase_index']
        for variant,v in row['variants'].items():
            signs=[]
            for d in v['draws']:
                a=rebuilt[(f'{sid}_{aa}_{0 if aa=="N" else d["draw"]}',pi,variant)]
                b=rebuilt[(f'{sid}_{bb}_{0 if bb=="N" else d["draw"]}',pi,variant)]
                if a is None or b is None:
                    require(d['contrast'] is None,'missing contrast'); signs.append(None);continue
                for k in a:
                    err=abs(a[k]-b[k]-d[k]);require(err<1e-10,'pair reconstruction');max_error=max(max_error,err);checks+=1
                signs.append(a['contrast']-b['contrast'])
            direction='INCOMPLETE' if None in signs else ('positive' if min(signs)>1e-6 else ('negative' if max(signs)<-1e-6 else 'mixed_or_near_zero'))
            require(direction==v['direction'],'stability sign')
    out={'status':'PASS_PARENT_RAW_DISTANCE_AND_BOUNDARY_RECONSTRUCTION','audio':28,'audio_phrase_variants':len(measured),
        'local_scalar_checks':nvalues,'pair_scalar_checks':checks,'maximum_error':max_error,
        'scope':'Original official matrices, SHA, FA character boundaries, real RF/lag/padding support, all local metrics and pair directions independently recomputed; not new ASR, human verification or visual inference.'}
    target=P/'parent_raw_array_audit.json'; require(not target.exists(),'immutable output exists');target.write_text(json.dumps(out,indent=2));print(json.dumps(out))

if __name__=='__main__': main()
