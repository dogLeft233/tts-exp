"""Independent CPU hash/formula/scale audit, no encoder forward."""
import json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
from protocol import OUT,DATA,sha,write
def main():
    p=json.loads((OUT/'protocol.json').read_text());rows=[]
    for spec in p['models']:
        mid=spec['id'];root=OUT/'features'/mid;r=json.loads((root/'receipt.json').read_text());assert r['status']=='COMPLETED' and len(r['conditions'])==7
        for c in r['conditions']:
            raw,sr=sf.read(DATA/'arms'/c['split']/'gain_0/aud.wav',dtype='float64');assert sr==16000
            x=raw.astype(np.float32);native=(x-x.mean())/np.sqrt(x.var()+1e-7);actual=(native*np.float32(c['input_scale'])).astype(np.float32)
            assert hashlib.sha256(actual.tobytes()).hexdigest()==c['processor_sha256']
            assert hashlib.sha256(native.tobytes()).hexdigest()==c['native_processor_sha256']
            assert sha(DATA/'arms'/c['split']/'gain_0/aud.wav')==c['source_audio_sha256']
            rms=np.sqrt(np.mean(actual.astype(float)**2));assert abs(rms/c['native_stats']['rms']-c['input_scale'])<1e-6
            path=root/c['file'];z=np.load(path,allow_pickle=False);assert sha(path)==c['sha256'] and z.dtype==np.float32 and np.isfinite(z).all() and list(z.shape)==c['shape']
            if c['input_scale']==1:
                old=json.loads((Path(p['parent'])/'features'/mid/'receipt.json').read_text());b=next(t for t in old['conditions'] if t['split']==c['split'] and t['normalize'] and t['gain_db']==0)
                assert c['sha256']==b['sha256']
            rows.append(dict(encoder=mid,split=c['split'],scale=c['input_scale'],rms=float(rms),formula_and_sha_exact=True))
    write(OUT/'feature_independent_review.json',dict(passed=True,conditions=rows,new_gpu_calls=0,order='zscore then multiply; no second normalization'))
    print(json.dumps(dict(passed=True,conditions=len(rows))))
if __name__=='__main__':main()
