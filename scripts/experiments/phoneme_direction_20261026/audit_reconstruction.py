"""Read back calibration waveforms and independently recompute quality aggregates."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from pystoi import stoi
from scipy.signal import correlate, correlation_lags, convolve
from calibrate import RUN, read, sha, write


def main():
    manifest=read(RUN/'input_manifest.json')
    expected={x['id']:x for x in manifest['calibration'] if x['arm']=='N'}
    recorded=read(RUN/'reconstruction_complete.json')['rows']
    if len(recorded)!=20 or {x['id'] for x in recorded}!=set(expected):
        raise ValueError('Unexpected reconstruction cohort')
    checks=[]
    for x in recorded:
        ref=expected[x['id']]
        if sha(x['source'])!=ref['canonical_audio_sha256'] or sha(x['output'])!=x['output_sha256']:
            raise ValueError('Waveform byte mismatch')
        a,sa=sf.read(x['source']);b,sb=sf.read(x['output'])
        if sa!=16000 or sb!=sa or a.shape!=b.shape or not np.isfinite(b).all():
            raise ValueError('Invalid waveform shape/rate/values')
        # Resolve common encoder/decoder latency at finer resolution than initial 10ms check.
        def envelope(z):
            return np.sqrt(np.maximum(convolve(z*z,np.ones(160)/160,mode='valid',method='fft')[::16],0))
        ea=envelope(a);eb=envelope(b);ea-=ea.mean();eb-=eb.mean()
        c=correlate(eb,ea,method='fft');lags=correlation_lags(len(eb),len(ea));mask=abs(lags)<=80
        lag=int(lags[mask][np.argmax(c[mask])])
        checks.append({'id':x['id'],'stoi':float(stoi(a,b,16000)),
                       'lag_1ms_grid_ms':lag,'samples':len(a),'clipped':int(np.sum(abs(b)>=1)),
                       'rms_relative_error':float(abs(np.sqrt(np.mean(b*b))/np.sqrt(np.mean(a*a))-1))})
    initial=read(RUN/'reconstruction_quality.json')
    by_id={x['id']:x for x in initial['rows']}
    discrepancy=max(abs(x['stoi']-by_id[x['id']]['stoi']) for x in checks)
    if discrepancy>1e-12:raise ValueError('Quality readback differs')
    median=float(np.median([x['stoi'] for x in checks]))
    lag=float(np.median([x['lag_1ms_grid_ms'] for x in checks]))
    write(RUN/'reconstruction_readback.json',{
        'status':'VERIFIED_TECHNICAL_HOLD' if initial['status']=='HOLD_RECONSTRUCTION' else 'VERIFIED_BASIC_ONLY',
        'rows':checks,'stoi_median':median,'stoi_min':min(x['stoi'] for x in checks),
        'max_stoi_discrepancy':discrepancy,'median_common_latency_ms':lag,
        'common_latency_correction_samples':round(lag*16),
        'latency_method':'10ms RMS energy window, 1ms hop, max cross-correlation within +/-80ms; median across fixed fit cohort',
        'numeric_protocol_frozen':False,'development_allowed':False,'confirmation_allowed':False,
        'quality_script_sha256':sha(__file__),'contract_sha256':sha(RUN/'calibration_contract.json')})
    print(json.dumps({'stoi_median':median,'common_latency_ms':lag,'verified':True}),flush=True)


if __name__=='__main__':main()
