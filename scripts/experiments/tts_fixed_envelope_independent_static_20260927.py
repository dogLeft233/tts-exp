"""Pre-forward synthetic scalar/frontend-index/resource audit, no real scores."""
import ast
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.signal import convolve

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/tts_fixed_envelope_generation_cross_20260927'
OUT=ROOT/'runs/tts_fixed_envelope_independent_audit_20260927'
WORKER=ROOT/'scripts/experiments/tts_fixed_envelope_generation_cross_20260927.py'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    p=read(RUN/'protocol.json'); seal=read(RUN/'seal.json')
    assert sha(RUN/'protocol.json')==seal['protocol_sha256']
    assert sha(WORKER)==seal['worker_sha256']
    for path,digest in p['dependencies'].items():assert sha(path)==digest,path
    tree=ast.parse(WORKER.read_text()); funcs=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name in ('envelope','transform')]
    W=np.hanning(513);W/=W.sum();ns={'np':np,'convolve':convolve,'W':W}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<sealed pure scalar transform>','exec'),ns)
    R=.0376838172675746;assert p['target_RMS']==R
    n=1024;t=np.arange(n);x=(np.sin(.077*t)+.4*np.cos(.173*t))*np.where(t<500,.02,.05);x[150:200]=0
    x=(x*(R/np.sqrt(np.mean(x*x)))).astype('float32').astype('float64')
    # Explicit reflect mapping and cosine Hann weights; independent of scipy convolution.
    weights=np.array([.5-.5*math.cos(2*math.pi*j/512) for j in range(513)]);weights/=sum(weights)
    e=[]
    for i in range(n):
        ix=[i+j-256 for j in range(513)]
        ix=[-j if j<0 else 2*(n-1)-j if j>=n else j for j in ix]
        e.append(math.sqrt(sum(float(x[j])**2*float(w) for j,w in zip(ix,weights))))
    e=np.array(e);g=np.array([max(z/R,.1)**(-.5) for z in e]);z=x*g
    restore=R/math.sqrt(sum(float(q)**2 for q in z)/n)
    independent=(z*restore).astype('float32')
    y,actualg,actuale,a=ns['transform'](x,R)
    error=float(abs(actuale-e).max());assert error<1e-15
    assert np.array_equal(y,independent)
    assert np.all(y[x==0]==0) and np.all(actualg>0) and len(y)==n
    assert abs(np.sqrt(np.mean(y.astype(float)**2))/R-1)<1e-6
    assert np.max(abs(y.astype(float)/actualg-x))<1e-7
    unfloored=actuale/R>=.1
    assert np.max(abs(np.log(actualg[unfloored]/a)+.5*np.log(actuale[unfloored]/R)))<1e-14
    rows=p['rows'];assert len(rows)==100
    assert sum(r['split']=='calibration' for r in rows)==26
    assert len(p['main_ids'])==71 and len({r['speaker'] for r in rows if r['id'] in p['main_ids']})==15
    payload=0;allocated=0
    for r in rows:
        for arm in 'NT':
            z=r['arms'][arm];info=z['FIXED']
            assert z['joint_L']==min(info['frames'],info['samples']//640)-5
            for nwin in [info['frames']-4,info['audio_windows']]:
                payload+=nwin*1024*4;allocated+=math.ceil((nwin*1024*4+128)/4096)*4096
    assert allocated==p['budget']['feature_allocated_exact_bound']==185016320
    assert allocated+17*2**20 <194*2**20
    receipt={'status':'SYNTHETIC_AND_BINDING_PASS_PENDING_STATIC_VERSION_REVIEW','time':time.time(),
       'protocol_sha256':sha(RUN/'protocol.json'),'worker_sha256':sha(WORKER),'audit_code_sha256':sha(__file__),
       'dependency_hashes_verified':len(p['dependencies']),'synthetic_reflect_hann_max_error':error,
       'independent_float32_output_exact':True,'gain_log_relation_pass':True,'zeros_length_RMS_inverse_pass':True,
       'count100_cal26_eval74_main71_speakers15':True,'all_arm_joint_L_verified':True,
       'feature_payload':payload,'feature_allocated_bound':allocated,'new_real_waveforms_scores_or_forward':False}
    dest=OUT/('synthetic_'+sha(RUN/'protocol.json')[:12]+'.json');assert not dest.exists()
    dest.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(dest),'sha256':sha(dest)}))

if __name__=='__main__':main()
