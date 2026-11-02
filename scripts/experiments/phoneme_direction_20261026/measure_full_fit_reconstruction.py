"""Read back all 300 candidate reconstructions against unchanged quality floors."""
import numpy as np
import soundfile as sf
from pystoi import stoi
from scipy.signal import correlate,correlation_lags
from calibrate import ROOT,RUN as PARENT,read,write,sha

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'


def main():
    rows=[]
    for k in range(3):rows.extend(read(RUN/f'regular_fit_shard{k}_complete.json')['rows'])
    expected={x['id'] for x in read(PARENT/'input_manifest.json')['fit'] if x['arm']=='N'}
    if len(rows)!=300 or {x['id'] for x in rows}!=expected:raise ValueError('Cohort mismatch')
    scores=[]
    for x in rows:
        if sha(x['source'])!=x['source_sha256'] or sha(x['output'])!=x['output_sha256'] or sha(x['features'])!=x['features_sha256']:
            raise ValueError('Changed artifact')
        a,sr=sf.read(x['source']);b,sb=sf.read(x['output'])
        if sr!=16000 or sb!=sr or len(a)!=len(b) or not np.isfinite(b).all():raise ValueError('Waveform contract')
        h=np.load(x['features'],mmap_mode='r')
        if h.shape!=((len(a)+319)//320,1024):raise ValueError('Feature contract')
        score=float(stoi(a,b,16000));n=len(a)//160
        ea=np.sqrt(np.mean(a[:n*160].reshape(n,160)**2,axis=1));eb=np.sqrt(np.mean(b[:n*160].reshape(n,160)**2,axis=1))
        c=correlate(eb-eb.mean(),ea-ea.mean(),method='fft');l=correlation_lags(n,n);mask=abs(l)<=20
        lag=int(l[mask][np.argmax(c[mask])])*10;clipped=int(np.sum(abs(b)>=1));rms=float(abs(np.sqrt(np.mean(b*b))/np.sqrt(np.mean(a*a))-1))
        failed=[]
        if score<.75:failed.append('stoi')
        if abs(lag)>20:failed.append('lag')
        if clipped:failed.append('clipping')
        if rms>.001:failed.append('rms')
        scores.append({'id':x['id'],'speaker_id':x['speaker_id'],'stoi':score,'global_envelope_lag_ms':lag,
                       'clipped':clipped,'rms_relative_error':rms,'failures':failed})
    median=float(np.median([x['stoi'] for x in scores]));failed=[x for x in scores if x['failures']]
    payload={'status':'PASS_BASIC_RECONSTRUCTION' if not failed and median>=.85 else 'HOLD_RECONSTRUCTION',
       'sources':len(scores),'speakers':len({x['speaker_id'] for x in scores}),'stoi_median':median,
       'stoi_min':min(x['stoi'] for x in scores),'failed':failed,'rows':scores,
       'unchanged_parent_contract_sha256':sha(PARENT/'calibration_contract.json'),
       'checkpoint_candidate_only':True,'development_allowed':False,'confirmation_allowed':False,'script_sha256':sha(__file__)}
    write(RUN/'regular_full_fit_quality.json',payload)
    print({k:payload[k] for k in ['status','sources','speakers','stoi_median','stoi_min']},'failed',len(failed),flush=True)


if __name__=='__main__':main()
