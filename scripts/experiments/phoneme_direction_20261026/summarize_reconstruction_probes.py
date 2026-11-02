"""Measure controlled diagnostics without changing acceptance floors."""
from collections import defaultdict
import numpy as np
import soundfile as sf
from pystoi import stoi
from scipy.signal import correlate,correlation_lags
from calibrate import ROOT,RUN as PARENT,read,write,sha

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'


def main():
    groups=defaultdict(list)
    outputs=read(RUN/'diagnostic_outputs.json')['rows']
    outputs += [dict(kind='prematch_raw',id=x['id'],source=x['source'],output=x['output'],sha256=x['output_sha256'])
                for x in read(PARENT/'reconstruction_complete.json')['rows']]
    for x in outputs:
        a,sr=sf.read(x['source']);b,sb=sf.read(x['output'])
        if sr!=16000 or sb!=sr or len(a)!=len(b) or sha(x['output'])!=x['sha256']:raise ValueError('Changed input')
        n=len(a)//160;ea=np.sqrt(np.mean(a[:n*160].reshape(n,160)**2,axis=1));eb=np.sqrt(np.mean(b[:n*160].reshape(n,160)**2,axis=1))
        c=correlate(eb-eb.mean(),ea-ea.mean(),method='fft');lags=correlation_lags(n,n);mask=abs(lags)<=20
        groups[x['kind']].append({'id':x['id'],'stoi':float(stoi(a,b,16000)),
               'stoi_inner':float(stoi(a[8000:-8000],b[8000:-8000],16000)),
               'identity_stoi':float(stoi(a,a,16000)),
               'global_envelope_lag_ms':int(lags[mask][np.argmax(c[mask])])*10,
               'clipped':int(np.sum(abs(b)>=1)),
               'relative_rms_error':float(abs(np.sqrt(np.mean(b*b))/np.sqrt(np.mean(a*a))-1))})
    summary={k:{'n':len(v),'stoi_median':float(np.median([r['stoi'] for r in v])),
                'stoi_min':min(r['stoi'] for r in v),'inner_median':float(np.median([r['stoi_inner'] for r in v])),
                'all_unchanged_floors_pass':bool(len(v)==20 and np.median([r['stoi'] for r in v])>=0.85 and all(r['stoi']>=0.75 and not r['clipped'] and abs(r['global_envelope_lag_ms'])<=20 and r['relative_rms_error']<=0.001 for r in v))} for k,v in groups.items()}
    write(RUN/'diagnostic_quality_revision02.json',{'summary':summary,'groups':dict(groups),'diagnostic_only':True,
           'development_allowed':False,'confirmation_allowed':False,'code_sha256':sha(__file__),
           'first_measurement_failure':'NumPy bool JSON serialization, incomplete diagnostic_quality.json preserved; waveforms unchanged'})
    print(summary,flush=True)


if __name__=='__main__':main()
