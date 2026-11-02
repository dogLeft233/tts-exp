"""Independent NumPy endpoint reconstruction and frozen paired pilot statistics."""
import os,json,hashlib
from pathlib import Path
import numpy as np

OUT=Path(os.environ.get('GAIN_RUN_DIR','C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/run_attempt02'))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def summarize(x,draws):
    x=np.asarray(x,dtype=float); samples=x[draws].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'positive':int((x>0).sum()),'negative':int((x<0).sum()),'n':len(x),'values':x.tolist()}

def endpoint(matrix,anchor):
    curve=np.mean(matrix,axis=0,dtype=np.float32).astype(float)
    background=float(np.median(curve)); best=int(curve.argmin())
    return {'C':background-float(curve[best]),'Dmin':float(curve[best]),'D0':float(curve[15]),
            'B':background,'Cfixed':background-float(curve[anchor]),'Dfixed':float(curve[anchor]),'offset':15-best}

def main():
    r=json.loads((OUT/'run_receipt.json').read_text(encoding='utf8'))
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    assert r['status']=='COMPLETE' and len(r['clips'])==10
    assert sha(OUT/'protocol.json')==r['protocol_sha256']
    assert {x['speaker'] for x in p['fit_speakers']}.isdisjoint(x['speaker'] for x in r['clips'])
    assert [x['speaker'] for x in r['clips']]==[x['speaker'] for x in p['test_speakers']]
    conditions=p['conditions']; keys=['C','Dmin','D0','B','Cfixed','Dfixed','offset']
    values={support:{c:{k:[] for k in keys} for c in conditions} for support in ['guard20','full']}
    errors=[]; metric_errors=[]; norm_errors=[]; recon_errors=[]; qa=[]
    g=np.load(OUT/'input/common.npy',allow_pickle=False)
    for rec in r['clips']:
        sid=rec['speaker']; folder=OUT/sid
        assert rec['status']=='COMPLETE' and set(rec['conditions'])==set(conditions)
        A=np.load(folder/'audio.npy',allow_pickle=False)
        assert hashlib.sha256(A.tobytes()).hexdigest()==rec['audio']['audio_embedding_sha256']
        az=np.pad(A,((15,15),(0,0)))
        base=None; base_visual=None
        with np.load(folder/'latents.npz',allow_pickle=False) as z:
            z0,z1=z['BASE'].astype(float),z['GAIN_P3'].astype(float)
            norm=np.sqrt(np.einsum('ij,ij->i',z0,z0))[:,None]
            u0=z0/norm; u1=z1/np.linalg.norm(z1,axis=1,keepdims=True)
            components={'COMMON':u0+g,'RESIDUAL':u1-g,'FULL_DIRECTION':u1,
                        'COMMON_NONNEG':np.maximum(u0+g,0),'RESIDUAL_NONNEG':np.maximum(u1-g,0)}
            for c,t in components.items():
                independent=(norm*t/np.linalg.norm(t,axis=1,keepdims=True)).astype(np.float32)
                recon_errors.append(float(np.max(abs(independent-z[c]))))
                norm_errors.append(float(np.max(abs(np.linalg.norm(z[c].astype(float),axis=1,keepdims=True)/norm-1))))
            assert np.array_equal(z['BASE'],z['BASE_REPEAT'])
        for c in conditions:
            meta=rec['conditions'][c]; file=folder/(c+'_score.npz')
            assert sha(file)==meta['score_npz_sha256']
            assert sha(folder/(c+'.mp4'))==meta['video']['sha256']
            with np.load(file,allow_pickle=False) as a:
                V,D,guard=a['visual'],a['distance'],a['guard20']
                # F.pairwise_distance uses epsilon per coordinate, even without exact equality.
                rebuilt=np.stack([np.sqrt(np.sum((v[None,:]-az[i:i+31]+np.float32(1e-6))**2,axis=1,dtype=np.float32)) for i,v in enumerate(V)])
                errors.append(float(np.max(abs(rebuilt-D))))
                n=min(len(A),len(V))
                assert np.array_equal(guard,np.arange(20,n-20))
                if c=='BASE': base=D.copy(); base_visual=V.copy()
                if c=='BASE_REPEAT': assert np.array_equal(base,D) and np.array_equal(base_visual,V)
                for support,idx in [('guard20',guard),('full',np.arange(n))]:
                    # Validate saved metric values exactly using saved float32 matrix.
                    derived=endpoint(D[idx],meta[support]['anchor'])
                    for k,val in derived.items():
                        metric_errors.append(abs(val-meta[support][k]))
                        values[support][c][k].append(val)
            qa.append({'speaker':sid,'condition':c,'distance_max_abs':errors[-1]})
    assert max(errors)<1e-5 and max(metric_errors)<1e-10 and max(norm_errors)<1e-6 and max(recon_errors)<2e-7
    rng=np.random.default_rng(20261018); draws=rng.integers(0,10,size=(20000,10))
    result={'status':'CONCLUDED_DESCRIPTIVE_PILOT','speakers':10,'conditions':conditions,
            'scientific_scope':'fixed natural audio SyncNet response, not perceptual improvement; calibration speakers reused but disjoint from axis fit',
            'receipt_sha256':sha(OUT/'run_receipt.json'),'protocol_sha256':sha(OUT/'protocol.json'),
            'absolute':values,'contrasts':{}}
    pairs=[(c,'BASE') for c in conditions if c not in ['BASE','BASE_REPEAT']]
    pairs += [('COMMON','GAIN_P3'),('COMMON','FULL_DIRECTION'),('COMMON','RESIDUAL'),
              ('COMMON_NONNEG','GAIN_P3'),('COMMON_NONNEG','FULL_DIRECTION'),('COMMON_NONNEG','RESIDUAL_NONNEG')]
    for support in values:
        result['contrasts'][support]={}
        for a,b in pairs:
            result['contrasts'][support][a+'-'+b]={k:summarize(np.array(values[support][a][k])-np.array(values[support][b][k]),draws) for k in keys if k!='offset'}
    primary=result['contrasts']['guard20']
    lo=lambda name:primary[name]['C']['ci95'][0]
    equiv=lambda name: primary[name]['C']['ci95'][0]>=-.05 and primary[name]['C']['ci95'][1]<=.05
    result['frozen_replication_gates']={'actual_gain_positive':lo('GAIN_P3-BASE')>0,
         'common_positive':lo('COMMON-BASE')>0,'common_equivalent_to_actual_gain':equiv('COMMON-GAIN_P3'),
         'common_equivalent_to_full_direction':equiv('COMMON-FULL_DIRECTION'),
         'nonnegative_common_positive':lo('COMMON_NONNEG-BASE')>0,
         'nonnegative_equivalent_to_actual_gain':equiv('COMMON_NONNEG-GAIN_P3'),
         'nonnegative_equivalent_to_full_direction':equiv('COMMON_NONNEG-FULL_DIRECTION')}
    result['all_replication_gates_pass']=all(result['frozen_replication_gates'].values())
    quality={'status':'PASS','videos':len(qa),'repeated_baseline_exact_speakers':10,
             'max_distance_reconstruction_error':max(errors),'max_scalar_metric_error':max(metric_errors),
             'max_norm_relative_error':max(norm_errors),'max_condition_reconstruction_error':max(recon_errors),
             'per_video':qa,'analyzer_sha256':sha(Path(__file__))}
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'independent_qa.json').write_text(json.dumps(quality,indent=2),encoding='utf8')
    print(json.dumps({'gates':result['frozen_replication_gates'],'effects':{k:v['C'] for k,v in primary.items()},'qa':{k:v for k,v in quality.items() if k!='per_video'}},indent=2))

if __name__=='__main__': main()
