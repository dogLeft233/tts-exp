"""Fixed official-video-cache, source-LOO PCM spectral EQ diagnostic. No training."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='2'
import argparse, datetime, hashlib, inspect, json, shutil, sys, time
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, correlate, correlation_lags

R=Path('/home/wjj/tts-audio/tts-exp')
OUT=R/'runs/syncnet_fixed_pcm_eq_20261025_attempt01'
BASE=R/'runs/syncnet_audio_replay20_20261024_attempt01'
CONDITIONS=['identity','plus','minus','reverse_plus','reverse_minus']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),str(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def logshape(x):
    _,ps=welch(np.asarray(x,dtype=float),fs=16000,nperseg=400,noverlap=240,nfft=512)
    ps=ps/max(ps.sum(),1e-30)
    z=10*np.log10(np.maximum(ps,1e-12))
    return z-z[1:].mean()

def profile_for(source,shapes):
    donors=sorted(set(shapes)-{source})
    assert len(donors)==19 and source not in donors
    n=np.mean([shapes[s]['N'] for s in donors],axis=0)
    t=np.mean([(shapes[s]['L06']+shapes[s]['L17'])/2 for s in donors],axis=0)
    delta=gaussian_filter1d(t-n,3,mode='reflect')
    delta-=delta[1:].mean()
    delta=np.clip(delta,-3,3); delta[0]=0
    rev=np.r_[0.,delta[1:][::-1]]
    return dict(source=source,donors=donors,natural_logshape=n.tolist(),tts_logshape=t.tolist(),
                direction_db=delta.tolist(),reversed_db=rev.tolist())

def transform(x,db):
    x=np.asarray(x,dtype=np.int16)
    a=x.astype(np.float64)
    padded=np.pad(a,(2048,2048),mode='reflect')
    ft=np.fft.rfft(padded)
    response=np.interp(np.fft.rfftfreq(len(padded),1/16000),np.linspace(0,8000,257),db)
    h=10**(response/20)
    changed=ft*h
    used=abs(ft)>max(abs(ft).max()*1e-8,1e-10)
    phase_error=float(np.max(abs(np.angle(changed[used]*ft[used].conj()))))
    raw=np.fft.irfft(changed,n=len(padded))[2048:2048+len(a)]
    raw*=np.sqrt(np.mean(a*a))/max(np.sqrt(np.mean(raw*raw)),1e-30)
    rounded=np.rint(raw)
    clipping=int(np.sum((rounded<-32768)|(rounded>32767)))
    if clipping: raise ValueError(f'EQ requires clipping ({clipping} samples)')
    y=rounded.astype(np.int16)
    return y,dict(phase_error_radians=phase_error,clipping_samples=clipping,
                  rms_relative_error=float(abs(np.sqrt(np.mean(y.astype(float)**2))/np.sqrt(np.mean(a*a))-1)),
                  same_samples=len(y)==len(x),real_positive_response=bool(np.all(h>0)))

def rms_envelope(x):
    n=(len(x)-400)//160+1
    return np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(np.asarray(x,dtype=float),400)[::160][:n]**2,axis=1))

def xlag(a,b,limit):
    c=correlate(b,a,method='fft'); l=correlation_lags(len(b),len(a)); k=abs(l)<=limit
    return int(l[k][np.argmax(c[k])])

def diagnostics(x,y,physical,profile):
    ex,ey=rms_envelope(x),rms_envelope(y)
    rho=float(np.corrcoef(ex,ey)[0,1])
    # Local energy centroid inside unchanged 40ms windows, stepped10ms.
    xx=np.lib.stride_tricks.sliding_window_view(x.astype(float),640)[::160]**2
    yy=np.lib.stride_tricks.sliding_window_view(y.astype(float),640)[::160]**2
    active=xx.sum(1)>xx.sum(1).max()*1e-4
    axis=np.arange(640)/16 # milliseconds
    shift=(yy@axis/np.maximum(yy.sum(1),1e-30)-xx@axis/np.maximum(xx.sum(1),1e-30))[active]
    q95=float(np.quantile(abs(shift),.95))
    wave_lag=xlag(x.astype(float),y.astype(float),2560)
    env_lag=xlag(ex-ex.mean(),ey-ey.mean(),16)
    before,after=logshape(x),logshape(y)
    n=np.array(profile['natural_logshape']); t=np.array(profile['tts_logshape'])
    mse=lambda a,b:float(np.mean((a[1:]-b[1:])**2))
    d=dict(physical,waveform_lag_samples=wave_lag,envelope_lag_10ms_bins=env_lag,
           envelope_correlation=rho,centroid_q95_abs_ms=q95,
           distance_to_natural_before=mse(before,n),distance_to_natural_after=mse(after,n),
           distance_to_tts_before=mse(before,t),distance_to_tts_after=mse(after,t))
    d['timing_pass']=bool(env_lag==0 and abs(wave_lag)<=1 and rho>=.98 and q95<=10 and physical['same_samples'] and physical['rms_relative_error']<=1e-4 and physical['clipping_samples']==0)
    return d

def prepare():
    assert not OUT.exists(); OUT.mkdir(parents=True)
    rows=[r for r in read(BASE/'input_manifest.json')['rows'] if (r['arm']=='N' and r['draw']==0) or (r['arm']!='N' and r['draw']==1)]
    assert len(rows)==60 and len({r['sample_id'] for r in rows})==20
    bound=[]; shapes={}
    for r in rows:
        item=BASE/'items'/r['id']; replay=read(item/'replay.json')
        assert sha(item/'scoring_audio.wav')==replay['scoring_audio_sha256']
        assert sha(item/'sequence.npz')==replay['sequence_sha256']
        sr,x=wavfile.read(item/'scoring_audio.wav'); assert sr==16000 and x.dtype==np.int16
        shapes.setdefault(r['sample_id'],{})[r['arm']]=logshape(x)
        bound.append(dict(r,scoring_audio_path=str(item/'scoring_audio.wav'),scoring_audio_sha256=sha(item/'scoring_audio.wav'),
                          sequence_path=str(item/'sequence.npz'),sequence_sha256=sha(item/'sequence.npz'),replay_path=str(item/'replay.json')))
    import torch,python_speech_features as psf
    off=R/'third_party/syncnet_python'
    pins={str(p):sha(p) for p in [off/'SyncNetInstance.py',off/'SyncNetModel.py',off/'data/syncnet_v2.model',Path(psf.__file__),Path(inspect.getfile(psf.base)),Path(inspect.getfile(psf.sigproc))]}
    protocol=dict(schema='fixed_pcm_loo_eq_v1',frozen_utc=utc(),source_sha256=sha(__file__),
       scope='20 existing sources x N0,L06d1,L17d1 x5conditions; fixed actual official scoring PCM/video cache; no training/new video/CoGenAV',
       conditions=CONDITIONS,profile='19source LOO; Welch400/160hop/512FFT normalizedPSD; average source-logPSD; two TTS equal per source; Gaussian sigma3; mean-center nonDC,clip+-3dB,DC0; reverse nonDC bins matched norm',
       transform='reflect2048; full padded rFFT positive real static gain; crop original exact length; match global RMS; round PCM16; fail on clipping',
       official='int16 default python_speech_features.mfcc; 20columns/4hop; cached row count/V1024; original1024 audio encoderFP32/eval; calc_pdist15; batch20',
       gates=dict(baseline_curve_abs=1e-4,baseline_curve_rms=1e-5,baseline_scalar_abs=1e-4,offset_exact=True,identity_pcm_exact=True),
       timing_qualification=dict(wave_lag_samples_max=1,envelope_lag_bins=0,envelope_corr_min=.98,centroid_q95_abs_ms_max=10,rms_relative_max=1e-4),
       primary='N source paired C contrast (plus-minus)-(reverse_plus-reverse_minus); plus-identity and minus-identity companion',
       interpretation_gate='Toward-TTS spectral response requires BOTH N plus-identity C and direction-specific contrast 99% marginal intervals above0, actual donor-TTS spectral distance improvement and timing checks; never perception equivalence or general TTS bias',
       secondary='L06/L17 same contrasts; C/B/D,D_anchor,C_anchor,search_uplift; all source primary, common per-recipient timing-pass subset sensitivity',
       statistics='source bootstrap10000 seed20261025 99% marginal descriptive intervals; no multiple-testing confirmatory claim',
       boundary='clocks/samplegrid preserved does not prove perceptual/event equivalence; candidate spectral factor not general TTS preference; historical EQ/band studies acknowledged',
       pins=pins,input_manifest_sha256=sha(BASE/'input_manifest.json'),resource=dict(min_free_bytes=5*1024**3,timeout_s=1800))
    put(OUT/'protocol.json',protocol);put(OUT/'input_manifest.json',dict(rows=bound));shutil.copy2(__file__,OUT/'source_frozen.py')
    put(OUT/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,executable=sys.executable,cuda=torch.version.cuda))
    for source in sorted(shapes): put(OUT/'profiles'/f'{source}.json',profile_for(source,shapes))
    cells=[]
    for r in bound:
        sr,x=wavfile.read(r['scoring_audio_path']); p=read(OUT/'profiles'/f'{r["sample_id"]}.json')
        direction=np.array(p['direction_db']);rev=np.array(p['reversed_db'])
        for condition,db in zip(CONDITIONS,[np.zeros(257),direction,-direction,rev,-rev]):
            y,phys=transform(x,db)
            if condition=='identity': assert np.array_equal(x,y)
            diag=diagnostics(x,y,phys,p)
            assert phys['rms_relative_error']<=1e-4 and phys['phase_error_radians']<1e-12
            path=OUT/'audio'/r['id']/f'{condition}.wav'; path.parent.mkdir(parents=True,exist_ok=True);wavfile.write(path,16000,y)
            cells.append(dict(id=r['id'],source=r['sample_id'],arm=r['arm'],condition=condition,
                         waveform_path=str(path),waveform_sha256=sha(path),profile_path=str(OUT/'profiles'/f'{r["sample_id"]}.json'),
                         profile_sha256=sha(OUT/'profiles'/f'{r["sample_id"]}.json'),sequence_path=r['sequence_path'],sequence_sha256=r['sequence_sha256'],diagnostics=diag))
    put(OUT/'cells.json',cells);put(OUT/'prepared.json',dict(cells=len(cells),source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),utc=utc()))
    print('PREPARED',len(cells),sha(OUT/'protocol.json'),sha(__file__),flush=True)

def scores(distance,anchor):
    import torch
    curve=torch.from_numpy(distance).mean(0); d,k=curve.min(0);b=torch.median(curve)
    da=curve[anchor]
    return dict(curve=curve.tolist(),C=float(b-d),B=float(b),D=float(d),offset=int(15-k),
                D_anchor=float(da),C_anchor=float(b-da),search_uplift=float(da-d),anchor_index=int(anchor))

def score(stage):
    import torch,python_speech_features as psf
    off=R/'third_party/syncnet_python';sys.path.insert(0,str(off))
    from SyncNetInstance import SyncNetInstance,calc_pdist
    torch.set_num_threads(2);torch.manual_seed(20261025)
    p=read(OUT/'protocol.json')
    for path,h in p['pins'].items(): assert sha(path)==h
    assert sha(__file__)==p['source_sha256']
    if stage=='effects': assert read(OUT/'baseline_complete.json')['pass_baseline']
    s=SyncNetInstance(device='cuda');s.loadParameters(str(off/'data/syncnet_v2.model'));s.__S__.eval()
    cells=read(OUT/'cells.json');selected=[c for c in cells if (c['condition']=='identity')==(stage=='baseline')]
    start=time.time();done=[]
    for cell in selected:
        assert time.time()-start<1800 and shutil.disk_usage(OUT).free>=5*1024**3
        assert sha(cell['waveform_path'])==cell['waveform_sha256'] and sha(cell['sequence_path'])==cell['sequence_sha256']
        data=np.load(cell['sequence_path']);v=data['video1024'];old=data['original_distance'];n=len(v)
        sr,x=wavfile.read(cell['waveform_path']); assert sr==16000 and x.dtype==np.int16
        mfcc=psf.mfcc(x,sr).T
        cin=np.stack([mfcc[:,i*4:i*4+20] for i in range(n)])[:,None].astype(np.float32)
        aa=[]
        with torch.inference_mode():
            for i in range(0,n,20): aa.append(s.__S__.forward_aud(torch.from_numpy(cin[i:i+20]).cuda()).cpu())
        a=torch.cat(aa);distance=np.stack([q.numpy() for q in calc_pdist(torch.from_numpy(v),a,15)])
        anchor=int(torch.from_numpy(old).mean(0).argmin())
        result=dict(cell,windows=n,score=scores(distance,anchor),video_array_sha256=hashlib.sha256(v.tobytes()).hexdigest())
        if stage=='baseline':
            error=distance-old; oldscore=scores(old,anchor)
            result['baseline_gate']=dict(curve_abs=float(abs(error).max()),curve_rms=float(np.sqrt(np.mean(error.astype(float)**2))),scalar_abs=max(abs(result['score'][k]-oldscore[k]) for k in ['C','B','D']),offset_exact=result['score']['offset']==oldscore['offset'])
            g=result['baseline_gate'];assert g['curve_abs']<=1e-4 and g['curve_rms']<=1e-5 and g['scalar_abs']<=1e-4 and g['offset_exact']
        path=OUT/'arrays'/cell['id']/f'{cell["condition"]}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,audio1024=a.numpy(),distance=distance)
        result['arrays_path']=str(path);result['arrays_sha256']=sha(path)
        put(OUT/'results'/cell['id']/f'{cell["condition"]}.json',result);done.append(result)
        print(stage,len(done),len(selected),cell['id'],cell['condition'],f'C={result["score"]["C"]:.6f}',flush=True)
    put(OUT/f'{stage}_complete.json',dict(pass_baseline=True if stage=='baseline' else None,cells=len(done),elapsed_s=time.time()-start,utc=utc(),protocol_sha256=sha(OUT/'protocol.json')))

def bootstrap(x):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(20261025)
    if not len(x): return dict(n=0,mean=None,ci99=None)
    z=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),ci99=np.quantile(z,[.005,.995]).tolist())

def analyze():
    assert read(OUT/'effects_complete.json')['cells']==240
    results=[read(p) for p in sorted((OUT/'results').glob('*/*.json'))];assert len(results)==300
    effects=[]
    for id in sorted({r['id'] for r in results}):
        g={r['condition']:r for r in results if r['id']==id};assert set(g)==set(CONDITIONS)
        common=all(r['diagnostics']['timing_pass'] for r in g.values())
        effects.append(dict(id=id,source=g['identity']['source'],arm=g['identity']['arm'],common_timing_pass=common,
           endpoints={k:dict(plus_identity=g['plus']['score'][k]-g['identity']['score'][k],minus_identity=g['minus']['score'][k]-g['identity']['score'][k],
                            direction_specific=(g['plus']['score'][k]-g['minus']['score'][k])-(g['reverse_plus']['score'][k]-g['reverse_minus']['score'][k]))
                      for k in ['C','B','D','D_anchor','C_anchor','search_uplift']}))
    summary={}
    for arm in ['N','L06','L17']:
        summary[arm]={}
        for subset in ['all','common_timing_pass']:
            group=[x for x in effects if x['arm']==arm and (subset=='all' or x['common_timing_pass'])]
            summary[arm][subset]={k:{contrast:bootstrap([x['endpoints'][k][contrast] for x in group]) for contrast in ['plus_identity','minus_identity','direction_specific']}
                                 for k in ['C','B','D','D_anchor','C_anchor','search_uplift']}
    put(OUT/'effects.json',effects);put(OUT/'summary.json',dict(statistics=summary,results=len(results),timing_pass_cells=sum(r['diagnostics']['timing_pass'] for r in results),
           timing_pass_recipients=sum(x['common_timing_pass'] for x in effects),baseline_max_curve_error=max(r['baseline_gate']['curve_abs'] for r in results if r['condition']=='identity'),
           protocol_sha256=sha(OUT/'protocol.json')))
    lines=['# Fixed PCM spectral EQ / SyncNet diagnostic','','Fixed actual official video cache and scoring PCM; source-LOO ±3dB spectral profile plus reversed-frequency controls. No training, video generation or CoGenAV. Historical midpoint-EQ/band interventions exist; this is a current N/0.6B/1.7B cohort/control extension.','','|Arm|Subset|N|C plus−identity|C minus−identity|C direction-specific 99% descriptive CI|','|---|---|---:|---:|---:|---|']
    for arm in summary:
        for subset in summary[arm]:
            s=summary[arm][subset]['C'];c=s['direction_specific'];lines.append(f"|{arm}|{subset}|{c['n']}|{s['plus_identity']['mean']}|{s['minus_identity']['mean']}|{c['mean']} {c['ci99']}|")
    lines+=['','Clocks and sample grids are unchanged, but static spectral filtering can change acoustic envelope landmarks. Timing diagnostics are gates/sensitivity, not perceptual or phoneme truth. Effects only establish response to a candidate spectral factor. They do not identify general TTS preference or human sync quality. All recipients stay in primary denominator; timing-passing sensitivity is separately reported.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'source_completed.py')
    records=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_manifest.json']
    put(OUT/'final_manifest.json',records)
    print('COMPLETE',len(results),sum(x['common_timing_pass'] for x in effects),len(records),sum(r['bytes'] for r in records),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','baseline','effects','analyze']);a=p.parse_args()
    if a.stage=='prepare': prepare()
    elif a.stage in ['baseline','effects']: score(a.stage)
    else: analyze()
if __name__=='__main__':main()
