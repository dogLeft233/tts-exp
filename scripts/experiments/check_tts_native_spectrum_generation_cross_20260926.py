"""Independent CPU reconstruction; no producer or frozen implementation import."""
from pathlib import Path
import hashlib,json
import numpy as np
import soundfile as sf
from scipy.signal import convolve
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_native_spectrum_match_calibration_20260926'
W=.5-.5*np.cos(2*np.pi*np.arange(512)/512);H=np.hanning(9);H/=sum(H)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rms(x):return float(np.sqrt(np.mean(x*x)))
def fft(x):
    x=np.pad(x,(256,256+(-len(x))%128),mode='reflect');return np.fft.rfft(np.stack([x[t:t+512]*W for t in range(0,len(x)-511,128)]),axis=1).T

def inverse(z,n):
    frames=np.fft.irfft(z.T,n=512,axis=1);length=(len(frames)-1)*128+512;y=np.zeros(length);d=np.zeros(length)
    for i in range(len(frames)):y[i*128:i*128+512]+=frames[i]*W;d[i*128:i*128+512]+=W**2
    return (y/d.clip(1e-300))[256:256+n]
def vectors(x):
    z=fft(x);a=abs(z);log=20*np.log10(np.maximum(a,a.max()*1e-4)) if a.max() else np.zeros_like(a);mu=log.mean(1);s=convolve(np.pad(mu,4,mode='reflect'),H,mode='valid',method='direct');s-=s[1:].mean();res=log-log.mean(0,keepdims=True)-log.mean(1,keepdims=True)+log.mean();return s,rms(res)
def envelope(x):return np.sqrt(convolve(np.pad(x*x,(256,255),mode='reflect'),W/W.sum(),mode='valid',method='direct')+1e-16)
def reconstruct(x,gdb):
    y=inverse(fft(x)*np.power(10.,gdb[:,None]/20),len(x));return y*(rms(x)/rms(y) if rms(y) else 1.)
def envmatch(x,target):
    y=x.copy();gain=np.ones(len(x));e=envelope(target)
    for _ in range(20):
        gain=np.minimum(np.maximum(gain*np.sqrt(e/np.maximum(envelope(y),1e-8)),10**(-12/20)),10**(12/20));y=x*gain;gain*=rms(x)/rms(y) if rms(y) else 1.;y=x*gain
    return y
def main():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for path,h in {**p['parent_hashes'],**p['code_hashes']}.items():assert sha(path)==h,path
    manifest=read(OUT/'waveform_manifest.json');assert len(manifest)==8*len(p['cal_ids'])
    for item in manifest:
        for stage in ('waveforms','pre_headroom','acoustics'):assert sha(item[stage]['path'])==item[stage]['sha256']
    metrics={(r['id'],r['arm'],r['condition']):r for r in read(OUT/'metrics.json')};storedpairs={r['id']:r for r in read(OUT/'pairs.json')};summary=read(OUT/'summary.json');errs={'waveforms':0.,'R_rms_db':0.,'spectral_distance':0.,'headroom':0.,'envelope_db':0.,'identity_float64':0.};reductions=[];envs=[];changes={'EQ':[],'ENVeq':[]};lengths=[];finite=[];peaks=[];rms_errors=[];restored_errors=[]
    for sid in p['cal_ids']:
        src={s['arm']:s for s in p['sources'] if s['id']==sid};xs={}
        for arm,s in src.items():assert sha(s['path'])==s['sha256'];xs[arm]=sf.read(s['path'],dtype='float64')[0]
        shapes={a:vectors(x)[0] for a,x in xs.items()};target=(shapes['N']+shapes['T'])/2;before={}
        for a,x in xs.items():
            eq=reconstruct(x,np.clip(target-shapes[a],-6,6));identity=reconstruct(x,np.zeros(257));before[a]={'raw':x,'identity':identity,'EQ':eq,'ENVeq':envmatch(x,eq)};errs['identity_float64']=max(errs['identity_float64'],float(abs(identity-x).max()))
        peak=max(abs(y).max() for dd in before.values() for y in dd.values());g=min(1,.98/peak) if peak else 1;errs['headroom']=max(errs['headroom'],abs(g-summary['headroom_gains'][sid]));actual={}
        for a,x in xs.items():
            actual[a]={};rs={};ss={}
            for c,y in before[a].items():
                restored_errors.append(abs(rms(y)-rms(x))/rms(x) if rms(x) else 0.)
                for stage,factor in [('pre_headroom',1.),('waveforms',g)]:
                    saved,sr=sf.read(OUT/stage/sid/a/(c+'.wav'),dtype='float64');assert sr==16000;errs['waveforms']=max(errs['waveforms'],float(abs(saved-y*factor).max()))
                actual[a][c]=saved;shape,r=vectors(saved);ss[c]=shape;rs[c]=r;m=metrics[sid,a,c];errs['R_rms_db']=max(errs['R_rms_db'],abs(r-m['residual_rms_db']));lengths.append(len(saved)==len(x));finite.append(np.isfinite(saved).all());peaks.append(abs(saved).max());expected=rms(x)*g;rms_errors.append(abs(rms(saved)-expected)/expected if expected else 0)
            for c in changes:changes[c].append(abs(rs[c]/rs['identity']-1) if rs['identity'] else 0)
            er=rms(20*np.log10(envelope(actual[a]['ENVeq'])/envelope(actual[a]['EQ'])));envs.append(er);errs['envelope_db']=max(errs['envelope_db'],abs(er-metrics[sid,a,'ENVeq']['target_envelope_db_rmse']))
        distances={c:rms((vectors(actual['N'][c])[0]-vectors(actual['T'][c])[0])[1:]) for c in before['N']}
        for c,d in distances.items():errs['spectral_distance']=max(errs['spectral_distance'],abs(d-storedpairs[sid]['distances'][c]['smooth_mean_log_shape_db']))
        if distances['raw']>=1e-8:reductions.append(1-distances['EQ']/distances['raw'])
        print('independent',sid,flush=True)
    gates={'finite':bool(all(finite)),'exact_length':all(lengths),'rms_restored':max(restored_errors)<=1e-5,'rms_saved':max(rms_errors)<=1e-5,'no_clip':max(peaks)<=.9800001,'identity_float64':errs['identity_float64']<=1e-8,'spectrum_reduction_median':bool(len(reductions) and np.median(reductions)>=.5),'spectrum_reduction_90pct':bool(len(reductions) and np.mean(np.array(reductions)>=.25)>=.9),'ENVeq_envelope_median':bool(np.median(envs)<=.35),'ENVeq_envelope_90pct':bool(np.mean(np.array(envs)<=.75)>=.9),**{c+'_R_median':bool(np.median(v)<=.02) for c,v in changes.items()}}
    assert gates==summary['gates'],(gates,summary['gates']);assert errs['waveforms']<1e-7 and errs['R_rms_db']<1e-9 and errs['spectral_distance']<1e-9 and errs['headroom']<1e-12 and errs['envelope_db']<1e-9
    result={'passed':True,'errors':errs,'gates_independently_reproduced':gates,'pairs':len(p['cal_ids']),'waveforms':len(manifest),'validator_sha256':sha(__file__),'note':'Independent FFT/window/OLA/spectral smoothing/EQ/envelope iteration/headroom and observed metrics; no producer import; calibration PASS or FAIL is preserved'};(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n');print(json.dumps(result,indent=2,default=lambda x:x.item()))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['calibration','evaluation']);args=ap.parse_args()
    OUT=ROOT/'runs/tts_native_spectrum_generation_cross_20260926/acoustic'/args.split
    main()
