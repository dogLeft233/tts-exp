"""Independent ENVmatched replay with log gain state and SciPy convolutions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

OUT=Path(__file__).resolve().parents[2]/'runs/tts_acoustic_modulation_calibration_20260926_v2'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def energy(x):
    window=signal.windows.hann(512,sym=False); window/=sum(window)
    return np.sqrt(signal.convolve(np.pad(x**2,(256,255),mode='reflect'),window,mode='valid',method='direct')+1e-16)


def replay(x,y):
    # Independent log-state form of damped multiplicative amplitude update.
    loggain=np.zeros_like(x); u=x.copy(); ey=energy(y)
    for _ in range(20):
        loggain += .5*(np.log(ey)-np.log(np.maximum(energy(u),1e-8)))
        loggain=np.clip(loggain,-12*np.log(10)/20,12*np.log(10)/20)
        u=x*np.exp(loggain)
        if np.dot(u,u):
            norm=np.sqrt(np.dot(x,x)/np.dot(u,u)); loggain += np.log(norm)
        u=x*np.exp(loggain)
    return u,np.exp(loggain)


def residual_rms(x):
    xpad=np.pad(x,(256,256+(-len(x))%128),mode='reflect')
    _,_,z=signal.stft(xpad,window='hann',nperseg=512,noverlap=384,nfft=512,boundary=None,padded=False)
    a=np.abs(z)
    if not a.max(): return 0.
    log=20*np.log10(np.maximum(a,a.max()*1e-4))
    centered=log-log.mean(0,keepdims=True)
    centered-=centered.mean(1,keepdims=True)
    return float(np.sqrt(np.mean(centered**2)))


def check(out):
    p=json.loads((out/'protocol.json').read_text()); s=json.loads((out/'summary.json').read_text())
    rows=json.loads((out/'metrics.json').read_text()); lookup={(r['id'],r['arm'],r['condition']):r for r in rows}
    manifest=json.loads((out/'waveform_manifest.json').read_text()); assert len(manifest)==312
    base=Path(p['parent_v1']); assert sha(base/'artifact_hashes.json')==p['parent_artifact_manifest_sha256']
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip() == s['protocol_sha256']
    archive=json.loads((base/'artifact_hashes.json').read_text())
    for rel,h in archive.items(): assert sha(base/rel)==h
    source={(r['id'],r['arm']):r for r in p['sources']}
    for rec in source.values(): assert sha(rec['path'])==rec['sha256']
    errors={'float64_control_vs_saved32_max_abs':0.,'gain_max_abs':0.,'target_envelope_db_rmse_max_abs':0.,
            'R_relative_change_max_abs':0.,'saved_rms_relative_max':0.,'all_conditions_vs_expected_float32_max_abs':0.}
    expected={}; controls=[]
    for sid in p['cal_ids']:
        waves={}
        for arm in ['N','T']:
            x,sr=sf.read(source[sid,arm]['path'],dtype='float64'); assert sr==16000
            for cond in p['conditions'][:4]: waves[arm,cond]=sf.read(base/'pre_headroom'/sid/arm/f'{cond}.wav',dtype='float64')[0]
            for cond in p['conditions'][4:]:
                target_cond=cond.removeprefix('ENVmatched_')
                u,gain=replay(x,waves[arm,target_cond]); waves[arm,cond]=u
                arr=np.load(out/'acoustics'/sid/arm/f'{cond}.npz')
                errors['gain_max_abs']=max(errors['gain_max_abs'],float(max(abs(gain-arr['final_gain_before_common_headroom']))))
                y=sf.read(out/'waveforms'/sid/arm/f'{cond}.wav',dtype='float64')[0]
                target=sf.read(out/'waveforms'/sid/arm/f'{target_cond}.wav',dtype='float64')[0]
                matched=float(np.sqrt(np.mean((20*np.log10(energy(y)/energy(target)))**2)))
                identity=sf.read(out/'waveforms'/sid/arm/'identity.wav',dtype='float64')[0]
                change=residual_rms(y)/residual_rms(identity)-1
                row=lookup[sid,arm,cond]
                errors['target_envelope_db_rmse_max_abs']=max(errors['target_envelope_db_rmse_max_abs'],abs(matched-row['target_envelope_db_rmse']))
                errors['R_relative_change_max_abs']=max(errors['R_relative_change_max_abs'],abs(change-row['R_RMS_relative_change_from_identity']))
                controls.append({'id':sid,'arm':arm,'condition':cond,'envelope_db_rmse':matched,'absolute_R_relative_change':abs(change)})
                assert np.all(y[x==0]==0)
        peak=max(float(max(abs(y))) for y in waves.values()); g=min(1,.98/peak) if peak else 1.
        assert abs(g-s['headroom_gains'][sid])<1e-12
        for key,y in waves.items(): expected[sid,*key]=y
    for item in manifest:
        key=item['id'],item['arm'],item['condition']; g=s['headroom_gains'][item['id']]
        for stage in ['pre_headroom','waveforms','acoustics']: assert sha(item[stage]['path'])==item[stage]['sha256']
        y,sr=sf.read(item['waveforms']['path'],dtype='float64'); u=expected[key]
        before=sf.read(item['pre_headroom']['path'],dtype='float64')[0]
        assert sr==16000 and len(y)==len(u) and np.isfinite(y).all() and max(abs(y))<1
        assert np.allclose(y,u*g,rtol=6e-8,atol=1e-12) and np.allclose(before,u,rtol=6e-8,atol=1e-12)
        err=float(max(abs(y-u*g))); errors['all_conditions_vs_expected_float32_max_abs']=max(errors['all_conditions_vs_expected_float32_max_abs'],err)
        if key[2].startswith('ENVmatched_'): errors['float64_control_vs_saved32_max_abs']=max(errors['float64_control_vs_saved32_max_abs'],err)
        x=sf.read(source[key[:2]]['path'],dtype='float64')[0]
        expected_rms=np.sqrt(np.mean(x*x))*g
        err=abs(np.sqrt(np.mean(y*y))-expected_rms)/expected_rms if expected_rms else 0.
        errors['saved_rms_relative_max']=max(errors['saved_rms_relative_max'],float(err))
    for cond in p['conditions'][4:]:
        a=[r for r in controls if r['condition']==cond]
        d=s['directions'][cond]
        em=float(np.median([r['envelope_db_rmse'] for r in a])); frac=float(np.mean([r['envelope_db_rmse']<=.75 for r in a])); rm=float(np.median([r['absolute_R_relative_change'] for r in a]))
        assert abs(em-d['envelope_db_rmse']['median'])<1e-10 and abs(frac-d['fraction_envelope_at_most_075'])<1e-12
        assert abs(rm-d['absolute_R_relative_change']['median'])<1e-10
    assert errors['gain_max_abs']<1e-10 and errors['saved_rms_relative_max']<=1e-5
    result={'status':'independently_verified','checker':'log-gain replay; scipy convolve and STFT; sequential spectral centering',
            'waveforms':312,'controls_reconstructed':104,'original_v1_artifacts_unchanged':len(archive),'maximum_errors':errors,'checker_sha256':sha(__file__)}
    (out/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=OUT); check(parser.parse_args().out)
