"""Frozen historical static image3 waveform G x E experiment; stages are separated."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_acoustic_generation_cross_20260926'
W2L=ROOT/'third_party/Wav2Lip'
PARENT=ROOT/'runs/tts_chinese_instance_primed_20260926'
V1=ROOT/'runs/tts_acoustic_modulation_calibration_20260926'
V2=ROOT/'runs/tts_acoustic_modulation_calibration_20260926_v2'
COND=['raw','identity','Rlo','Rhi','ENVlo','ENVhi']
H=['Rlo','Rhi','ENVlo','ENVhi']
LAGS=np.arange(-15,16)

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def acoustic_modules():
    return (load_file('cal1',V1/'code_snapshot/tts_acoustic_modulation_calibration.py'),
            load_file('cal2',ROOT/'scripts/experiments/tts_acoustic_modulation_env_control.py'))

def proc_identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':int(pid),'exe':str((p/'exe').resolve()),'starttime_ticks':fields[19]}

def compute_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return [{'pid':int(a),'name':b.strip(),'memory_mib':int(c)} for a,b,c in (line.split(',',2) for line in text.splitlines() if line.strip())]

def resource_check(p,startup=False):
    allowed=p['resources']['allowed_system_process'];rows=compute_processes()
    for r in rows:
        if r['pid']==os.getpid():continue
        assert proc_identity(r['pid'])==allowed,('foreign compute',r)
        assert r['memory_mib']<=p['resources']['daemon_memory_limit_mib']
    assert proc_identity(allowed['pid'])==allowed
    used,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if startup:assert util<=10 and used<=p['resources']['daemon_memory_limit_mib']+128,(used,util)
    free=shutil.disk_usage(OUT).free;assert free>=5<<30,('disk reserve',free)
    return {'time':time.time(),'compute':rows,'memory_mib':used,'utilization':util,'free_bytes':free}

@contextmanager
def lease(stage,p):
    with open('/tmp/tts-exp-gpu.lock','a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checks=[]
        for i in range(3):
            checks.append(resource_check(p,True))
            if i<2:time.sleep(5)
        write(OUT/(stage+'_resource_gate.json'),{'checks':checks,'pid':os.getpid(),'lease':'/tmp/tts-exp-gpu.lock'})
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def freeze():
    assert not (OUT/'protocol.json').exists()
    parent=read(PARENT/'protocol.json');mid=read(ROOT/'runs/tts_native_midpoint_20260926/protocol.json')
    old={(r['id'],r['arm']):r for r in mid['groups']['static_cloud']['records']}
    rows=[]
    for r in parent['rows']:
        row={'id':r['id'],'speaker':r['speaker'],'split':r['split'],'audio':{},'historical':{}}
        for a,legacy in [('N','N'),('T','C')]:
            row['audio'][a]=r['audio'][legacy];assert sha(r['audio'][legacy]['path'])==r['audio'][legacy]['sha256']
            record=old[r['id'],legacy]
            f=record['visual']['path'];af=record['audio']['path']
            row['historical'][a]={'visual_path':f,'visual_key':record['visual']['key'],'visual_sha256':sha(f),
                                  'audio_path':af,'audio_key':record['audio']['key'],'audio_sha256':sha(af),
                                  'frames':record['frames'],'joint_L':record['joint_L'],'sample_count':record['sample_count']}
        rows.append(row)
    system=next(r for r in compute_processes() if r['name']=='/usr/libexec/gnome-remote-desktop-daemon')
    files=[V1/'protocol.json',V1/'code_snapshot/tts_acoustic_modulation_calibration.py',V2/'protocol.json',
           ROOT/'scripts/experiments/tts_acoustic_modulation_env_control.py',W2L/'audio.py',W2L/'hparams.py',W2L/'models/wav2lip.py',
           W2L/'checkpoints/wav2lip_gan.pth',ROOT/'third_party/syncnet_python/data/syncnet_v2.model',
           ROOT/'third_party/syncnet_python/SyncNetModel.py',ROOT/'scripts/experiments/tts_native_gain_attribution/syncnet.py',
           ROOT/'scripts/experiments/static_image_bridge/score_worker.py',ROOT/'scripts/experiments/static_image_bridge/render_worker.py']
    im=next(i for i in parent['images'] if i['id']=='3')
    p={'status':'frozen_before_new_waveforms_features_scores','created_epoch':time.time(),'rows':rows,'image':im,'conditions':COND,
       'science':'Historical100 N/cloudT; old26cal engineering; original74eval statistical, joint guard20 support frozen before scoring; old/new80 not included; new exploration, not unseen confirmation. Frozen independently of source-condition results.',
       'acoustics':'Exact v1 float64 residual alpha .8/1/1.2, +/-6dB, Hann512/hop128, RMS restore. R targets cast FLOAT32 as v2, ENV20 fixed iterations. All N/T x6 shared final g. No resynthesis or N/T time alignment.',
       'waveform_format':'FLOAT32 WAV; Wav2Lip official audio.load_wav/melspectrogram; SyncNet MFCC waveform float64*32768 in historical PCM integer amplitude units, no quantization. Original integer-valued RAW must match old PCM16 MFCC/features on cal.',
       'acoustic_gates':'v1 ordering>=90%,median relative target each direction>=5%; v2 envelope RMSE median<=.35dB and>=90%<=.75dB,abs R change median<=2%, each direction and each split; float64 identity<=1e-8,FLOAT32<=1e-7,RMSrel<=1e-5,length finite no clip. Eval failure reported/no deletion/no retune; pause dependent attribution.',
       'generation':{'image':'3','box':im['generation_box'],'score_box':im['score_box']['box'],'fps':25,'batch':32,'seed':20260926,'codec':'FFV1 cropped224 AVI BGR, no audio','timing':'official Wav2Lip chunk_mels exact own audio count','identity':'fresh render main; raw alias only if actual mel arrays identical, with firstcal independent raw render; historical features only bridge'},
       'scoring':{'L':'min(frame_count,PCM_samples//640)-5; same each source arm all6 conditions, no cross-N/T clocks','lags':list(range(-15,16)),'distance':'torch.float32 pairwise_distance eps1e-6; float64 lag mean','k':3,'policies':['guard20','valid','guard0'],'geometries':['raw','unit'],'endpoint':'C=median31-min31; B=median31; D=min31; C_anchor=B-curve[k3]; bestlag=argmin-15; offset=-bestlag','frontend':'SyncNetEngine.extract_visual FFmpeg JPEG image stream; float PCM integer-unit MFCC default python_speech_features,13x20 stride4; same forward_aud'},
       'cross':'Per arm,h: q00=V_identity/A_identity,q10=V_h/A_identity,q01=V_identity/A_h,q11=V_h/A_h. G=q10-q00,E=q01-q00,I=q11-q10-q01+q00,total=q11-q00; exact closure. T-N contrasts and same-dose R-ENV for G/E/I/total.',
       'statistics':{'speaker_equal':True,'bootstrap':20000,'seed':20260926,'ci':.99,'family':'individual comparison CI, no FWER claim'},
       'bridge_features':'sigmaA,V,M and covAV at fixed k3 on guard20; M=(V_i+A_i+3)/2; temporal centering; descriptive only, no regression/mediation fractions',
       'controls':{'repeat':'lexicographically first cal ID N/T identity repeated; pixel equality and feature max<=1e-5','identity':'RAW/identity waveform<=1e-7 and mel<=1e-5; audio feature<=1e-4; score<=1e-4; original PCM16 vs float integer-unit frontend parity','delay':'+5frames=3200samples zero head same length firstcal N/T; common queries25:L-25; overlap lags -10..15 compare base -15..10; expected bestlag+5 if interior, curve max delta<=.15 diagnostic and bestlag shift tolerance1; no k/support adjustment','anchor':'cal all6 diagonal medianlag eacharm within1 of3 and arm medians diff<=1; failure flags insufficient anchor migration, does not change standard C/support'},
       'resources':{'allowed_system_process':proc_identity(system['pid']),'daemon_memory_limit_mib':system['memory_mib']+256,'startup':'3 low-occupancy snapshots5s apart,util<=10%; only bound daemon exception; all other foreign compute blocks','lease':'/tmp/tts-exp-gpu.lock','reserve_bytes':5<<30,'estimated_core_bytes':9<<30,'historical_5_condition_video_estimate_GiB':5.345},
       'limits':'Scores are evaluator responses; G-only changes generated video at same audio, E-only changes evaluator audio at fixed video. R-ENV still differs static spectral/F0 properties. No claim real mouth improved or pure temporal-contrast causality.',
       'parent_hashes':{str(f):sha(f) for f in [PARENT/'protocol.json',ROOT/'runs/tts_native_midpoint_20260926/protocol.json']},'dependencies':{str(f):sha(f) for f in files}}
    write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print('frozen',len(rows),flush=True)


def protocol():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for f,h in p['dependencies'].items():assert sha(f)==h,f
    return p


def frontend_windows(x):
    import python_speech_features
    mfcc=np.asarray(python_speech_features.mfcc(np.asarray(x,dtype=np.float64)*32768,16000),dtype=np.float64)
    n=min(len(x)//640-5,(len(mfcc)-20)//4+1)
    windows=np.stack([mfcc[i*4:i*4+20].T for i in range(n)]).astype(np.float32)
    return windows,mfcc


def prepare():
    p=protocol();a1,a2=acoustic_modules();sys.path.insert(0,str(W2L));import audio
    from scripts.experiments.static_image_bridge.render_worker import chunk_mels
    manifest=[];acoustic=[];spectral=[];all_gates={}
    for row in p['rows']:
        sid=row['id'];dest=OUT/'prepared'/f'{sid}.json'
        if dest.exists():
            saved=read(dest);manifest.append(saved['manifest']);acoustic.extend(saved['acoustic']);spectral.extend(saved['spectral']);continue
        source={}
        for arm,r in row['audio'].items():
            assert sha(r['path'])==r['sha256'];source[arm],sr=sf.read(r['path'],dtype='float64');assert sr==16000
        before,_,_,detail=a1.transform_pair(source['N'],source['T']);waves={};gains={};traces={}
        for arm in ['N','T']:
            waves[arm]={k:before[arm][v].astype(np.float32).astype(np.float64) for k,v in [('raw','raw'),('identity','identity'),('Rlo','alpha08'),('Rhi','alpha12')]}
            gains[arm]={}
            for env,target in [('ENVlo','Rlo'),('ENVhi','Rhi')]:
                y,gain,trace=a2.envelope_match(source[arm],waves[arm][target]);waves[arm][env]=y;gains[arm][env]=gain;traces[arm+'/'+env]=trace
        g=a2.shared_headroom(waves);record={'id':sid,'speaker':row['speaker'],'split':row['split'],'headroom_gain':g,'arms':{}};local=[];local_spec=[]
        for arm in ['N','T']:
            record['arms'][arm]={};raw_values=None;raw_spectral=None;metrics={}
            for c in COND:
                prefix=OUT/'audio'/sid/arm/c;prefix.parent.mkdir(parents=True,exist_ok=True)
                sf.write(prefix.with_suffix('.wav'),waves[arm][c]*g,16000,subtype='FLOAT');y=sf.read(prefix.with_suffix('.wav'),dtype='float64')[0]
                info,values=a1.analyze(y)
                if c=='raw':raw_values=values
                info.update(a1.couplings(values,raw_values));info.update({'id':sid,'arm':arm,'condition':c,'split':row['split'],'non_silent':a1.rms(source[arm])>1e-8,
                    'rms_relative_error':abs(a1.rms(y)-g*a1.rms(source[arm]))/(g*a1.rms(source[arm])),
                    'exact_length':len(y)==len(source[arm]),'headroom_gain':g})
                if c=='identity':info['identity_max_abs']=float(max(abs(y-source[arm]*g)));info['identity_float64_max_abs']=detail[arm]['identity']['identity_float64_max_abs']
                if c.startswith('ENV'):
                    target='Rlo' if c=='ENVlo' else 'Rhi';target_y=sf.read(OUT/'audio'/sid/arm/(target+'.wav'),dtype='float64')[0]
                    info['target_envelope_db_rmse']=a1.rms(20*np.log10(a2.envelope(y)/a2.envelope(target_y)))
                    info['R_relative_change']=info['residual_rms_db']/metrics['identity']['residual_rms_db']-1
                    values['final_gain']=gains[arm][c];info['gain_outside_fraction']=traces[arm+'/'+c]['final_gain_outside_bounds_fraction']
                else:info['gain_clipping_fraction']=detail[arm][{'raw':'raw','identity':'identity','Rlo':'alpha08','Rhi':'alpha12'}[c]]['gain_clipping_fraction']
                z=a1.stft(y);mag=abs(z);ls=20*np.log10(np.maximum(mag,mag.max()*1e-4));meanlog=ls.mean(1);meanamp=mag.mean(1);arith=20*np.log10(np.maximum(meanamp,meanamp.max()*1e-4))
                spec={'mean_log_db':meanlog,'mean_amplitude_db':arith}
                if c=='raw':raw_spectral=spec
                specinfo={'id':sid,'arm':arm,'condition':c}
                for name,vector in spec.items():
                    delta=(vector-raw_spectral[name])[1:];specinfo[name+'_rmse']=a1.rms(delta);specinfo[name+'_shift']=float(delta.mean());specinfo[name+'_centered_rmse']=a1.rms(delta-delta.mean());values[name]=vector
                np.savez_compressed(prefix.with_suffix('.npz'),**values)
                wav=audio.load_wav(str(prefix.with_suffix('.wav')),16000);mel=audio.melspectrogram(wav).astype(np.float32);chunks=chunk_mels(mel,25);windows,mfcc=frontend_windows(y)
                mp=OUT/'frontend'/sid/arm/(c+'.npz');mp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(mp,mel=mel,windows=windows)
                record['arms'][arm][c]={'waveform':str(prefix.with_suffix('.wav')),'sha256':sha(prefix.with_suffix('.wav')),'samples':len(y),'frames':len(chunks),'audio_windows':len(windows),'L':min(len(chunks),len(y)//640)-5,'frontend':str(mp),'frontend_sha256':sha(mp),'acoustics':str(prefix.with_suffix('.npz')),'acoustics_sha256':sha(prefix.with_suffix('.npz'))}
                metrics[c]=info;local.append(info);local_spec.append(specinfo)
            record['arms'][arm]['joint_L']=min(record['arms'][arm][c]['L'] for c in COND)
            record['arms'][arm]['raw_identity_mel_max']=float(np.max(abs(np.load(record['arms'][arm]['raw']['frontend'])['mel']-np.load(record['arms'][arm]['identity']['frontend'])['mel'])))
        record['guard20_eligible']=all(record['arms'][a]['joint_L']>40 for a in ['N','T']);write(dest,{'manifest':record,'acoustic':local,'spectral':local_spec,'gain_traces':traces})
        manifest.append(record);acoustic.extend(local);spectral.extend(local_spec);print('prepared',row['split'],sid,flush=True)
    lookup={(r['id'],r['arm'],r['condition']):r for r in acoustic}
    for split in ['calibration','evaluation']:
        selected=[r for r in manifest if r['split']==split];keys=[(r['id'],a) for r in selected for a in ['N','T'] if lookup[r['id'],a,'identity']['non_silent']]
        base=np.array([lookup[*k,'identity']['residual_rms_db'] for k in keys]);lo=np.array([lookup[*k,'Rlo']['residual_rms_db'] for k in keys]);hi=np.array([lookup[*k,'Rhi']['residual_rms_db'] for k in keys]);d={'arms':len(keys),'ordered_fraction':float(np.mean((lo<base)&(base<hi))),'median_decrease':float(np.median((base-lo)/base)),'median_increase':float(np.median((hi-base)/base)),'ENV':{}}
        for c in ['ENVlo','ENVhi']:
            env=[lookup[*k,c] for k in keys];err=np.array([r['target_envelope_db_rmse'] for r in env]);change=np.abs([r['R_relative_change'] for r in env]);d['ENV'][c]={'median_db_rmse':float(np.median(err)),'fraction_at_most_075':float(np.mean(err<=.75)),'median_abs_R_relative':float(np.median(change))}
        d['passed']=d['ordered_fraction']>=.9 and d['median_decrease']>=.05 and d['median_increase']>=.05 and all(v['median_db_rmse']<=.35 and v['fraction_at_most_075']>=.9 and v['median_abs_R_relative']<=.02 for v in d['ENV'].values());all_gates[split]=d
    engineering=all(r['rms_relative_error']<=1e-5 and r['all_finite'] and r['exact_length'] and r['peak']<.9800001 and r.get('identity_max_abs',0)<=1e-7 and r.get('identity_float64_max_abs',0)<=1e-8 for r in acoustic)
    old=read(ROOT/'runs/tts_native_midpoint_20260926/protocol.json')['groups']['static_cloud']['eval_ids'];actual=[r['id'] for r in manifest if r['split']=='evaluation' and r['guard20_eligible']]
    write(OUT/'support.json',manifest);write(OUT/'acoustic_metrics.json',acoustic);write(OUT/'spectral_descriptive.json',spectral)
    write(OUT/'acoustic_gate.json',{'splits':all_gates,'engineering_passed':engineering,'passed':engineering and all(d['passed'] for d in all_gates.values()),'old_main_ids':old,'new_main_ids':actual,'same_old_support':old==actual,'excluded_short':[r['id'] for r in manifest if not r['guard20_eligible']]})
    print('CPU prepare complete',all_gates,engineering,flush=True)


def seal():
    p=protocol();assert read(OUT/'acoustic_gate.json')['passed'];assert not (OUT/'features').exists()
    assert all(r['arms'][a]['raw_identity_mel_max']<=1e-5 for r in read(OUT/'support.json') for a in ['N','T'])
    files=[Path(__file__),ROOT/'scripts/experiments/check_tts_acoustic_generation_cross_20260926.py',ROOT/'tests/test_tts_acoustic_generation_cross_20260926.py',ROOT/'scripts/experiments/report_tts_acoustic_generation_cross_20260926.py']
    code=OUT/'code_snapshot';code.mkdir(exist_ok=True)
    for f in files:(code/f.name).write_bytes(f.read_bytes())
    write(OUT/'cpu_seal.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'support_sha256':sha(OUT/'support.json'),'acoustic_gate_sha256':sha(OUT/'acoustic_gate.json'),'code':{str(f):sha(f) for f in files},'status':'before_any_new_feature_or_score'})


def render_cell(model,chunks,frame,image_tensor,box,roi,path,capture_z=False):
    import cv2,torch
    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
    path.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),25,(224,224));assert writer.isOpened();digest=hashlib.sha256();zs=[]
    handle=model.audio_encoder.register_forward_hook(lambda module,args,out:zs.append(out.detach().cpu().numpy().reshape(len(out),-1))) if capture_z else None
    x1,y1,x2,y2=box
    try:
        for start in range(0,len(chunks),32):
            mt=torch.from_numpy(np.asarray(chunks[start:start+32],dtype=np.float32)[:,None]).cuda()
            with torch.inference_mode():pred=model(mt,image_tensor.expand(len(mt),-1,-1,-1)).cpu().numpy().transpose(0,2,3,1)*255
            for fo in pred:
                full=frame.copy();full[y1:y2,x1:x2]=cv2.resize(fo.astype(np.uint8),(x2-x1,y2-y1));cropped=crop_zero_padded(full,roi);writer.write(cropped);digest.update(cropped.tobytes())
    finally:
        writer.release()
        if handle is not None:handle.remove()
    return {'path':str(path),'sha256':sha(path),'pixel_sha256':digest.hexdigest(),'frames':len(chunks)},np.concatenate(zs) if zs else None


def audio_forward(engine,windows):
    torch=engine._torch;outputs=[]
    with torch.inference_mode():
        for start in range(0,len(windows),32):outputs.append(engine.network.forward_aud(torch.from_numpy(windows[start:start+32])[:,None].cuda()).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)


def produce(split):
    import cv2,torch
    from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model
    from scripts.experiments.static_image_bridge.render_worker import chunk_mels
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine,scoring_environment
    p=protocol();lock=read(OUT/'cpu_seal.json')
    for f,h in lock['code'].items():assert sha(f)==h,f
    assert sha(OUT/'support.json')==lock['support_sha256'];assert read(OUT/'acoustic_gate.json')['passed']
    if split=='evaluation':assert (OUT/'cal_controls.json').exists() and read(OUT/'cal_controls.json')['engineering_passed']
    rows=[r for r in read(OUT/'support.json') if r['split']==split];first=min(r['id'] for r in read(OUT/'support.json') if r['split']=='calibration')
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    with lease('produce_'+split,p):
        write(OUT/'runtime.json',scoring_environment());model=_load_model(W2L/'checkpoints/wav2lip_gan.pth','cuda');engine=SyncNetEngine(batch_size=32,device='cuda')
        im=p['image'];assert sha(im['path'])==im['sha256'];frame=cv2.imread(im['path']);x1,y1,x2,y2=im['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));mask=face.copy();mask[48:]=0
        it=torch.from_numpy(np.concatenate([mask,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
        try:
            for row in rows:
                sid=row['id']
                for arm in ['N','T']:
                    for c in ['identity','raw',*H]:
                        resource_check(p);info=row['arms'][arm][c];dest=OUT/'features'/sid/arm/c
                        if dest.with_suffix('.json').exists():continue
                        assert sha(info['waveform'])==info['sha256'];assert sha(info['frontend'])==info['frontend_sha256'];front=np.load(info['frontend']);ae=audio_forward(engine,front['windows']);chunks=chunk_mels(front['mel'],25)
                        alias=c=='raw' and sid!=first and row['arms'][arm]['raw_identity_mel_max']==0
                        if alias:
                            orig=read(OUT/'features'/sid/arm/'identity.json');v=np.load(orig['features'])['visual'];vm=orig['visual_metadata'];video=orig['video'];z=np.load(orig['features'])['z']
                        else:
                            video,z=render_cell(model,chunks,frame,it,im['generation_box'],im['score_box']['box'],OUT/'videos'/sid/arm/(c+'.avi'),True);v,vm=engine.extract_visual(video['path'])
                        assert video['frames']==info['frames'];assert min(len(v),len(ae))>=row['arms'][arm]['joint_L']
                        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),visual=v,audio=ae,z=z)
                        write(dest.with_suffix('.json'),{'id':sid,'arm':arm,'condition':c,'features':str(dest.with_suffix('.npz')),'sha256':sha(dest.with_suffix('.npz')),'video':video,'visual_metadata':vm,'raw_alias_new_identity':alias,'audio_frontend':info})
                    if sid==first:
                        cp=OUT/'controls'/arm;cp.mkdir(parents=True,exist_ok=True);info=row['arms'][arm]['identity'];front=np.load(info['frontend']);chunks=chunk_mels(front['mel'],25)
                        video,z=render_cell(model,chunks,frame,it,im['generation_box'],im['score_box']['box'],cp/'repeat.avi');v,vm=engine.extract_visual(video['path'])
                        identity=np.load(OUT/'features'/sid/arm/'identity.npz');original=read(OUT/'features'/sid/arm/'identity.json')
                        pcm=sf.read(info['waveform'],dtype='float64')[0];shift=np.zeros_like(pcm);shift[3200:]=pcm[:-3200];sf.write(cp/'delay5.wav',shift,16000,subtype='FLOAT');w,m=frontend_windows(shift);delay=audio_forward(engine,w)
                        parent=next(r for r in p['rows'] if r['id']==sid);old_audio,old_meta=engine.extract_audio(parent['audio'][arm]['path']);raw=np.load(OUT/'features'/sid/arm/'raw.npz')
                        old_pcm=sf.read(parent['audio'][arm]['path'],dtype='int16')[0];old_w,_=engine._audio_windows(old_pcm);new_w,_=frontend_windows(old_pcm.astype(np.float64)/32768)
                        float_original_audio=audio_forward(engine,new_w)
                        np.savez_compressed(cp/'controls.npz',repeat_visual=v,delay_audio=delay,old_audio=old_audio,float_original_audio=float_original_audio)
                        write(cp/'controls.json',{'video':video,'features_sha256':sha(cp/'controls.npz'),'repeat_pixel_equal':video['pixel_sha256']==original['video']['pixel_sha256'],'repeat_feature_max':float(np.max(abs(v-identity['visual']))),'old_float_mfcc_max':float(np.max(abs(old_w-new_w))),'old_float_audio_max':float(np.max(abs(old_audio-float_original_audio))),'original_float_waveform_max':float(np.max(abs(old_pcm.astype(np.float32)/32768-old_pcm.astype(np.float64)/32768))),'delay_waveform_sha256':sha(cp/'delay5.wav')})
                print('produced',split,sid,flush=True)
        finally:engine.close()
    write(OUT/('gpu_release_'+split+'.json'),{'time':time.time(),'pid':os.getpid(),'lease_released':True})


def matrix(v,a):
    import torch
    v=torch.from_numpy(np.asarray(v,dtype=np.float32));a=np.pad(np.asarray(a,dtype=np.float32),((15,15),(0,0)));n=len(v)
    with torch.inference_mode():return np.stack([torch.nn.functional.pairwise_distance(v,torch.from_numpy(a[15+s:15+s+n]),eps=1e-6).numpy() for s in LAGS],1).astype(np.float64)

def summarize(m,policy,k=3):
    n=len(m)
    if policy=='guard20':curve=m[20:n-20].mean(0) if n>40 else None
    elif policy=='guard0':curve=m.mean(0)
    else:curve=np.array([m[max(0,-lag):min(n,n-lag),lag+15].mean() for lag in LAGS])
    if curve is None:return None
    j=int(np.argmin(curve));b=float(np.median(curve));d=float(curve[j]);anchor=float(curve[k+15])
    return {'C':b-d,'B':b,'D':d,'C_anchor':b-anchor,'D_anchor':anchor,'best_lag':j-15,'offset':15-j,'curve':curve.tolist()}

def unit(a):return (a.astype(float)/np.linalg.norm(a.astype(float),axis=1)[:,None]).astype(np.float32)

def covariance_features(v,a,L):
    if L<=40:return None
    v=v[20:L-20].astype(float);a=a[23:L-17].astype(float);vc=v-v.mean(0);ac=a-a.mean(0);mc=(vc+ac)/2
    return {'sigmaA':float(np.sqrt(np.mean(np.sum(ac*ac,axis=1)))),'sigmaV':float(np.sqrt(np.mean(np.sum(vc*vc,axis=1)))),
            'sigmaM':float(np.sqrt(np.mean(np.sum(mc*mc,axis=1)))),'covAV':float(np.mean(np.sum(ac*vc,axis=1)))}


def score(split):
    import torch
    torch.set_num_threads(2);p=protocol();rows=[r for r in read(OUT/'support.json') if r['split']==split]
    if split=='evaluation':assert read(OUT/'cal_controls.json')['engineering_passed']
    seal={str(f):sha(f) for r in rows for f in (OUT/'features'/r['id']).rglob('*') if f.suffix in ['.json','.npz']}
    write(OUT/('feature_seal_'+split+'.json'),seal)
    write(OUT/('score_lock_'+split+'.json'),{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'feature_seal_sha256':sha(OUT/('feature_seal_'+split+'.json')),'support_sha256':sha(OUT/'support.json'),'status':'before_'+split+'_distance_scores'})
    for row in rows:
        sid=row['id'];dest=OUT/'scores'/sid;results={};matrices={}
        for arm in ['N','T']:
            L=row['arms'][arm]['joint_L'];features={c:np.load(OUT/'features'/sid/arm/(c+'.npz')) for c in COND};results[arm]={}
            for geom in ['raw','unit']:
                va={c:((unit(f['visual'][:L]),unit(f['audio'][:L])) if geom=='unit' else (f['visual'][:L],f['audio'][:L])) for c,f in features.items()};cells={}
                pairs=[('identity','identity'),('raw','raw')]+[(h,'identity') for h in H]+[('identity',h) for h in H]+[(h,h) for h in H]
                for vc,ac in pairs:
                    key=vc+'__'+ac;m=matrix(va[vc][0],va[ac][1]);matrices[arm+'__'+geom+'__'+key]=m
                    cells[key]={'policies':{policy:summarize(m,policy) for policy in p['scoring']['policies']},'bridge':covariance_features(va[vc][0],va[ac][1],L)}
                # Historical raw scores are bridge only, recomputed on common joint support.
                hist=next(r for r in p['rows'] if r['id']==sid)['historical'][arm];assert sha(hist['visual_path'])==hist['visual_sha256'];assert sha(hist['audio_path'])==hist['audio_sha256']
                vv=np.load(hist['visual_path'])[hist['visual_key']];aa=np.load(hist['audio_path'])[hist['audio_key']];ll=min(L,hist['joint_L']);vv=vv[:ll];aa=aa[:ll]
                if geom=='unit':vv,aa=unit(vv),unit(aa)
                old=matrix(vv,aa);matrices[arm+'__'+geom+'__historical']=old
                cells['historical']={'policies':{policy:summarize(old,policy) for policy in p['scoring']['policies']}}
                results[arm][geom]=cells
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),**matrices);write(dest.with_suffix('.json'),{'id':sid,'speaker':row['speaker'],'split':split,'eligible':row['guard20_eligible'],'L':{a:row['arms'][a]['joint_L'] for a in ['N','T']},'cells':results,'matrix_sha256':sha(dest.with_suffix('.npz'))})
    if split=='calibration':cal_controls()
    print('scored',split,len(rows),flush=True)


def cal_controls():
    p=protocol();rows=[r for r in read(OUT/'support.json') if r['split']=='calibration'];first=min(r['id'] for r in rows);lags={};parity={};delay={}
    for c in COND:
        lags[c]={a:float(np.median([read(OUT/'scores'/(r['id']+'.json'))['cells'][a]['raw'][c+'__'+c if c in ['raw','identity'] else c+'__'+c]['policies']['guard20']['best_lag'] for r in rows])) for a in ['N','T']}
    for arm in ['N','T']:
        r=next(r for r in rows if r['id']==first);L=r['arms'][arm]['joint_L'];control=read(OUT/'controls'/arm/'controls.json');f=np.load(OUT/'controls'/arm/'controls.npz');base=np.load(OUT/'features'/first/arm/'identity.npz');raw=np.load(OUT/'features'/first/arm/'raw.npz')
        m=matrix(base['visual'][:L],base['audio'][:L]);d=matrix(base['visual'][:L],f['delay_audio'][:L]);q=slice(25,L-25);base_curve=m[q].mean(0);delay_curve=d[q].mean(0)
        be=int(np.argmin(base_curve))-15;de=int(np.argmin(delay_curve))-15
        delay[arm]={'base_bestlag':be,'delay_bestlag':de,'expected':be+5,'curve_overlap_max':float(np.max(abs(delay_curve[5:]-base_curve[:-5]))),'lag_shift_pass':abs(de-be-5)<=1,'common_query_start':25,'common_query_stop':L-25}
        old_new_score=abs(summarize(matrix(base['visual'][:L],f['old_audio'][:L]),'guard20')['C']-summarize(matrix(base['visual'][:L],f['float_original_audio'][:L]),'guard20')['C'])
        parity[arm]={**control,'old_float_score_C_difference':old_new_score,'raw_identity_visual_max':float(np.max(abs(raw['visual']-base['visual']))),'raw_identity_audio_max':float(np.max(abs(raw['audio']-base['audio']))),'raw_identity_score_C_difference':abs(summarize(matrix(raw['visual'][:L],raw['audio'][:L]),'guard20')['C']-summarize(m,'guard20')['C'])}
    anchor=all(abs(value-3)<=1 for v in lags.values() for value in v.values()) and all(abs(v['N']-v['T'])<=1 for v in lags.values())
    passed=all(r['repeat_pixel_equal'] and r['repeat_feature_max']<=1e-5 and r['old_float_mfcc_max']<=1e-5 and r['old_float_audio_max']<=1e-4 and r['old_float_score_C_difference']<=1e-4 and r['original_float_waveform_max']<=1e-7 and r['raw_identity_audio_max']<=1e-4 and r['raw_identity_score_C_difference']<=1e-4 for r in parity.values()) and all(r['lag_shift_pass'] and r['curve_overlap_max']<=.15 for r in delay.values())
    write(OUT/'cal_controls.json',{'engineering_passed':passed,'anchor_transfer_passed':anchor,'fixed_anchor':3,'condition_arm_medianlags':lags,'parity':parity,'delay':delay,'no_k_support_adjustment':True})


def bootstrap(values,speakers):
    groups=sorted(set(speakers));means=np.array([np.mean([v for v,g in zip(values,speakers) if g==name]) for name in groups]);inds=np.random.Generator(np.random.PCG64(20260926)).integers(0,len(groups),(20000,len(groups)));draw=means[inds].mean(1)
    return {'mean':float(means.mean()),'ci99':np.quantile(draw,[.005,.995]).tolist(),'n':len(values),'speakers':len(groups),'group_means':dict(zip(groups,means.tolist()))}

def effects(q):return {'G':q[1]-q[0],'E':q[2]-q[0],'I':q[3]-q[1]-q[2]+q[0],'total':q[3]-q[0]}


def analyze():
    p=protocol();rows=[read(f) for f in sorted((OUT/'scores').glob('*.json')) if read(f)['split']=='evaluation'];assert len(rows)==74
    output={};per=[];closure=0.
    for geom in ['raw','unit']:
        for policy in ['guard20','valid','guard0']:
            use=[r for r in rows if r['eligible']] if policy=='guard20' else rows;groups=[r['speaker'] for r in use];view={}
            for metric in ['C','B','D','C_anchor','D_anchor','best_lag']:
                per_arm={};native=[]
                for r in use:native.append(r['cells']['T'][geom]['identity__identity']['policies'][policy][metric]-r['cells']['N'][geom]['identity__identity']['policies'][policy][metric])
                view[metric]={'baseline_T_minus_N':bootstrap(native,groups),'conditions':{},'R_minus_ENV':{}}
                for h in H:
                    per_arm[h]={a:[] for a in ['N','T']}
                    for r in use:
                        for a in ['N','T']:
                            cells=r['cells'][a][geom];q=[cells[key]['policies'][policy][metric] for key in ['identity__identity',h+'__identity','identity__'+h,h+'__'+h]];e=effects(q);closure=max(closure,abs(e['G']+e['E']+e['I']-e['total']));per_arm[h][a].append(e)
                            per.append({'id':r['id'],'speaker':r['speaker'],'geometry':geom,'policy':policy,'metric':metric,'arm':a,'condition':h,'q00':q[0],'q10':q[1],'q01':q[2],'q11':q[3],**e})
                    view[metric]['conditions'][h]={}
                    for effect in ['G','E','I','total']:
                        n=np.array([r[effect] for r in per_arm[h]['N']]);t=np.array([r[effect] for r in per_arm[h]['T']]);view[metric]['conditions'][h][effect]={'N':bootstrap(n,groups),'T':bootstrap(t,groups),'T_minus_N':bootstrap(t-n,groups)}
                for dose in ['lo','hi']:
                    view[metric]['R_minus_ENV'][dose]={}
                    for effect in ['G','E','I','total']:
                        n=np.array([r[effect] for r in per_arm['R'+dose]['N']])-np.array([r[effect] for r in per_arm['ENV'+dose]['N']]);t=np.array([r[effect] for r in per_arm['R'+dose]['T']])-np.array([r[effect] for r in per_arm['ENV'+dose]['T']]);view[metric]['R_minus_ENV'][dose][effect]={'N':bootstrap(n,groups),'T':bootstrap(t,groups),'T_minus_N':bootstrap(t-n,groups)}
            output[geom+'/'+policy]=view
    write(OUT/'summary.json',{'results':output,'maximum_algebra_closure_error':closure,'cal_controls':read(OUT/'cal_controls.json'),'acoustic_gate':read(OUT/'acoustic_gate.json'),'status':'complete' if read(OUT/'cal_controls.json')['engineering_passed'] else 'control_failed'});write(OUT/'effects.json',per)
    print('analysis complete',closure,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','prepare','seal','produce','score','analyze']);parser.add_argument('--split',choices=['calibration','evaluation']);args=parser.parse_args()
    globals()[args.stage](args.split) if args.stage in ['produce','score'] else globals()[args.stage]()
