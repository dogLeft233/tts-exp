"""Independent CPU replay of saved PCM/EQ/SyncNet effects, no runner import."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(k,'2')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, correlate, correlation_lags


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def logshape(x):
    power=welch(x.astype(float),16000,nperseg=400,noverlap=240,nfft=512)[1]
    z=10*np.log10(np.maximum(power/power.sum(),1e-12))
    return z-z[1:].mean()


def distance(v,a):
    pad=np.zeros((len(a)+30,a.shape[1]),np.float32)
    pad[15:15+len(a)]=a
    d=np.empty((len(v),31),np.float32)
    for k in range(31):
        diff=v-pad[k:k+len(v)]+np.float32(1e-6)
        d[:,k]=np.sqrt(np.sum(diff**2,axis=1,dtype=np.float32))
    return d


def scalar(d,anchor):
    curve=d.mean(0)
    b=float(np.median(curve)); dd=float(curve.min()); da=float(curve[anchor])
    return dict(C=b-dd,B=b,D=dd,D_anchor=da,C_anchor=b-da,search_uplift=da-dd,
                offset=int(15-curve.argmin()))


def lag(x,y,maximum):
    co=correlate(y,x,method='fft'); ls=correlation_lags(len(y),len(x))
    keep=abs(ls)<=maximum
    return int(ls[keep][co[keep].argmax()])


def audit(root):
    protocol=read(root/'protocol.json')
    inputs=read(root/'input_manifest.json')['rows']
    assert len(inputs)==60 and len({r['sample_id'] for r in inputs})==20
    assert sha(root/'source_frozen.py')==protocol['source_sha256']
    for p,h in protocol['pins'].items():
        assert sha(p)==h
    shapes={}
    for row in inputs:
        assert sha(row['scoring_audio_path'])==row['scoring_audio_sha256']
        assert sha(row['sequence_path'])==row['sequence_sha256']
        sr,x=wavfile.read(row['scoring_audio_path'])
        assert sr==16000 and x.dtype==np.int16
        shapes.setdefault(row['sample_id'],{})[row['arm']]=logshape(x)
    profile_error=0.
    for source in sorted(shapes):
        p=read(root/'profiles'/f'{source}.json')
        donors=sorted(set(shapes)-{source})
        assert p['donors']==donors and source not in donors and len(donors)==19
        n=np.mean([shapes[s]['N'] for s in donors],0)
        t=np.mean([(shapes[s]['L06']+shapes[s]['L17'])/2 for s in donors],0)
        diff=gaussian_filter1d(t-n,3,mode='reflect')
        diff-=diff[1:].mean(); diff=np.clip(diff,-3,3); diff[0]=0
        rev=np.concatenate(([0.],diff[:0:-1]))
        for actual,expected in [(p['natural_logshape'],n),(p['tts_logshape'],t),(p['direction_db'],diff),(p['reversed_db'],rev)]:
            profile_error=max(profile_error,float(np.max(np.abs(np.array(actual)-expected))))
    assert profile_error<1e-10
    result_paths=sorted((root/'results').glob('*/*.json'))
    assert len(result_paths)==300
    index={r['id']:r for r in inputs}
    max_distance_error=0.;max_scalar_error=0.;pcm_checks=0;metric_rows=[];diagnostic_checks=[]
    timing_count=0
    for identity in sorted(index):
        row=index[identity]
        sr,x=wavfile.read(row['scoring_audio_path'])
        sequence=np.load(row['sequence_path'])
        video=sequence['video1024']
        old=sequence['original_distance']
        original_anchor=int(old.mean(0).argmin())
        profile=read(root/'profiles'/f"{row['sample_id']}.json")
        basefft=np.fft.rfft(np.pad(x.astype(float),(2048,2048),mode='reflect'))
        padded_length=len(x)+4096
        frequencies=np.fft.rfftfreq(padded_length,1/16000)
        for condition in protocol['conditions']:
            r=read(root/'results'/identity/f'{condition}.json')
            assert r['video_array_sha256']==hashlib.sha256(video.tobytes()).hexdigest()
            assert r['sequence_sha256']==row['sequence_sha256']
            assert sha(r['waveform_path'])==r['waveform_sha256']
            assert sha(r['arrays_path'])==r['arrays_sha256']
            assert sha(r['profile_path'])==r['profile_sha256']
            sr,y=wavfile.read(r['waveform_path'])
            assert sr==16000 and len(y)==len(x) and y.dtype==np.int16
            db={'identity':np.zeros(257),'plus':np.array(profile['direction_db']),
                'minus':-np.array(profile['direction_db']),'reverse_plus':np.array(profile['reversed_db']),
                'reverse_minus':-np.array(profile['reversed_db'])}[condition]
            response=10**(np.interp(frequencies,np.arange(257)*31.25,db)/20)
            transformed=np.fft.irfft(basefft*response,n=padded_length)[2048:-2048]
            transformed*=np.linalg.norm(x.astype(float))/np.linalg.norm(transformed)
            # Algebraically equivalent RMS normalization may differ by a few ulps;
            # compare actual quantized outputs, not a floating approximation.
            rebuilt=np.rint(transformed)
            assert np.max(np.abs(rebuilt))<=32768
            pcm_error=int(np.max(np.abs(rebuilt-y.astype(float))))
            assert pcm_error<=1
            if condition=='identity':
                assert np.array_equal(x,y)
            a=np.load(r['arrays_path'])
            d=distance(video,a['audio1024'])
            max_distance_error=max(max_distance_error,float(np.max(np.abs(d-a['distance']))))
            score=scalar(a['distance'],original_anchor)
            for k in ['C','B','D','D_anchor','C_anchor','search_uplift']:
                max_scalar_error=max(max_scalar_error,abs(score[k]-r['score'][k]))
            assert score['offset']==r['score']['offset']
            assert abs(score['C']-score['C_anchor']-score['search_uplift'])<1e-10
            if condition=='identity':
                assert float(np.max(np.abs(a['distance']-old)))<=1e-4
            ex=np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(x.astype(float),400)[::160]**2,1))
            ey=np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(y.astype(float),400)[::160]**2,1))
            correlation=float(np.corrcoef(ex,ey)[0,1])
            wave_lag=lag(x.astype(float),y.astype(float),2560)
            envelope_lag=lag(ex-ex.mean(),ey-ey.mean(),16)
            rms=abs(np.linalg.norm(y.astype(float))/np.linalg.norm(x.astype(float))-1)
            diag=r['diagnostics']
            wx=np.lib.stride_tricks.sliding_window_view(x.astype(float),640)[::160]**2
            wy=np.lib.stride_tricks.sliding_window_view(y.astype(float),640)[::160]**2
            active=wx.sum(1)>wx.sum(1).max()*1e-4
            moment=np.arange(640)/16
            cx=(wx*moment).sum(1)/np.maximum(wx.sum(1),1e-30)
            cy=(wy*moment).sum(1)/np.maximum(wy.sum(1),1e-30)
            q95=float(np.quantile(abs(cy-cx)[active],.95))
            assert abs(q95-diag['centroid_q95_abs_ms'])<1e-8
            timing_pass=(envelope_lag==0 and abs(wave_lag)<=1 and correlation>=.98 and q95<=10 and rms<=1e-4)
            assert bool(timing_pass)==diag['timing_pass']
            before,after=logshape(x),logshape(y)
            for target,key in [('natural_logshape','natural'),('tts_logshape','tts')]:
                reference=np.array(profile[target])
                for feature,when in [(before,'before'),(after,'after')]:
                    expected=float(np.mean((feature[1:]-reference[1:])**2))
                    assert abs(expected-diag[f'distance_to_{key}_{when}'])<1e-8
            assert wave_lag==diag['waveform_lag_samples'] and envelope_lag==diag['envelope_lag_10ms_bins']
            assert abs(correlation-diag['envelope_correlation'])<1e-10
            assert abs(rms-diag['rms_relative_error'])<1e-10
            assert rms<=1e-4
            timing_count+=diag['timing_pass'];pcm_checks+=1
            metric_rows.append(dict(id=identity,source=r['source'],arm=r['arm'],condition=condition,score=score))
            diagnostic_checks.append(dict(id=identity,condition=condition,pcm_max_lsb_error=pcm_error,
                                          timing_pass=diag['timing_pass'],wave_lag=wave_lag,envelope_lag=envelope_lag))
    assert max_distance_error<1e-4 and max_scalar_error<1e-4
    effects=[]
    for identity in sorted(index):
        rs={r['condition']:r for r in metric_rows if r['id']==identity}
        g={k:r['score'] for k,r in rs.items()}
        effects.append(dict(id=identity,arm=rs['identity']['arm'],source=rs['identity']['source'],
                       plus_identity=g['plus']['C']-g['identity']['C'],
                       minus_identity=g['minus']['C']-g['identity']['C'],
                       direction_specific=g['plus']['C']-g['minus']['C']-g['reverse_plus']['C']+g['reverse_minus']['C']))
    production=read(root/'summary.json')['statistics']
    stats={};max_stats_error=0.
    for arm in ['N','L06','L17']:
        group=[r for r in effects if r['arm']==arm]
        stats[arm]={}
        for field in ['plus_identity','minus_identity','direction_specific']:
            x=np.array([r[field] for r in group])
            rng=np.random.default_rng(20261025)
            distribution=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
            ci=np.quantile(distribution,[.005,.995])
            stats[arm][field]=dict(n=len(x),mean=float(x.mean()),ci99=ci.tolist())
            oldstat=production[arm]['all']['C'][field]
            max_stats_error=max(max_stats_error,abs(x.mean()-oldstat['mean']),float(np.max(np.abs(ci-oldstat['ci99']))))
    assert max_stats_error<1e-4
    return dict(status='PASS_INDEPENDENT_PCM_PROFILE_DISTANCE_SCALAR_STATISTICS_REPLAY',
                protocol_sha256=sha(root/'protocol.json'),profile_max_error=profile_error,pcm_cells=pcm_checks,
                numpy_distance_max_error=max_distance_error,scalar_max_error=max_scalar_error,
                summary_max_error=max_stats_error,timing_pass_cells=timing_count,statistics=stats,
                diagnostics=diagnostic_checks,
                boundary='No independent GPU convolution forward or human perception verification; source bootstrap conditional on frozen shared LOO profiles')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    root=Path(args.run); result=audit(root)
    output=root/'parent_independent_audit.json'
    with output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['diagnostics']},indent=2))
