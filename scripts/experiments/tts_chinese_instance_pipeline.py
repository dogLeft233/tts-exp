"""Alignment and frozen native rendering for the Chinese instance experiment."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from scripts.experiments.tts_chinese_instance import BASE,SOURCE,CAL,PRE,ARMS,ROOT,read,write,sha
from scripts.experiments.tts_prepost_mel import MODEL
from scripts.experiments.tts_pcm_residual import metrics
from scripts.experiments.tts_raw_video_transfer import render

def manifest():
    p=read(BASE/'protocol.json');old=read(SOURCE/'02_mfa_mandarin341_ready/mfa_summary.json')['records']
    rows=[]
    for r in p['rows']:
        r=dict(r);r['audio']=dict(r['audio']);r['grids']={}
        for a,label in [('N','natural'),('C','tts')]:
            g=old[r['id']][label]
            if g.get('textgrid') and Path(g['textgrid']).exists():
                assert g['audio_sha256']==r['audio'][a]['sha256'] and sha(g['textgrid'])==g['textgrid_sha256']
                r['grids'][a]={'path':g['textgrid'],'sha256':g['textgrid_sha256'],'reused':True}
        for a in ('Q1','Q2'):
            f=BASE/'tts'/r['id']/(a+'.json')
            if f.exists() and read(f)['status']=='COMPLETE':
                z=read(f); assert sha(z['path'])==z['sha256'];r['audio'][a]={'path':z['path'],'sha256':z['sha256']}
                g=BASE/'mfa_output'/r['speaker']/(r['id']+'_'+a+'.TextGrid')
                if g.exists():r['grids'][a]={'path':str(g),'sha256':sha(g),'reused':False}
        rows.append(r)
    write(BASE/'manifest.json',{'rows':rows,'protocol_sha256':sha(BASE/'protocol.json')})
    return rows

def align():
    rows=manifest()
    for r in rows:
        lab=SOURCE/'02_mfa_mandarin341_ready/input'/r['speaker']/'natural'/f"{r['speaker']}_{r['id']}_natural.lab"
        assert lab.exists()
        assert ''.join(lab.read_text().split())==''.join(r['text'].split())
        for a in ('Q1','Q2'):
            if a not in r['audio']:continue
            dest=BASE/'mfa_input'/r['speaker']/(r['id']+'_'+a)
            dest.parent.mkdir(exist_ok=True,parents=True)
            if not dest.with_suffix('.wav').exists():dest.with_suffix('.wav').symlink_to(r['audio'][a]['path'])
            shutil.copyfile(lab,dest.with_suffix('.lab'))
    exe='/home/wjj/miniconda3/envs/mfa3/bin/mfa'
    dictionary='/home/wjj/Documents/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict'
    acoustic='/home/wjj/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip'
    cmd=[exe,'align',str(BASE/'mfa_input'),dictionary,acoustic,str(BASE/'mfa_output'),'--num_jobs','4','--temporary_directory',str(BASE/'mfa_tmp'),'--clean']
    write(BASE/'mfa_protocol.json',{'command':cmd,'dictionary_sha256':sha(dictionary),'acoustic_sha256':sha(acoustic),'labs':{str(f):sha(f) for f in (BASE/'mfa_input').rglob('*.lab')},'note':'same original character-tokenized transcript, Q1/Q2 pooled per speaker; forced alignment not proof of correct reading'})
    env=dict(os.environ);env['PATH']='/home/wjj/miniconda3/envs/mfa3/bin:'+env['PATH'];env['OMP_NUM_THREADS']='2';env['MFA_ROOT_DIR']=str(BASE/'mfa_runtime')
    with (BASE/'mfa.log').open('w') as log: result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(BASE/'mfa_exit.json',{'returncode':result.returncode})
    manifest()

def cache(r,im,a):
    if a not in ('N','C'):return None
    olda='N' if a=='N' else 'T'
    for root in (CAL,PRE):
        p=read(root/'protocol.json');old=next((x for x in p['rows'] if x['id']==r['id']),None)
        if old is None or old['audio'][olda]['sha256']!=r['audio'][a]['sha256']:continue
        assert p['images']==read(BASE/'protocol.json')['images'] and p['model_sha256']==sha(MODEL)
        f=root/'scores'/im['id']/(r['id']+'.json')
        z=read(f);v=f.with_name(r['id']+'_visual.npz');assert sha(v)==z['visual_sha256']
        return np.load(v)[olda],{'reused_visual':str(v),'reused_sha256':sha(v),'reused_arm':olda,'video':z.get('videos',{}).get(olda)}
    return None

def score(control=False):
    import cv2,torch
    from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model,chunk_mels
    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
    from scripts.experiments.tts_native_gain_attribution.common import gpu_compute_pids,gpu_lease
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    from scripts.experiments.tts_native_gain_attribution import config as sync_config
    sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
    p=read(BASE/'protocol.json');rows=manifest()
    if control: rows=[next(r for r in rows if r['split']=='evaluation')]
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert sha(MODEL)==read(CAL/'protocol.json')['model_sha256']
    write(BASE/'render_protocol.json',{'model_sha256':sha(MODEL),'frontend_sha256':sha(audio.__file__),'syncnet_weight_sha256':sha(sync_config.SYNCNET_MODEL),'syncnet_definition_sha256':sha(sync_config.SYNCNET_DEFINITION),'syncnet_instance_sha256':sha(sync_config.SYNCNET_INSTANCE),'pipeline_sha256':sha(__file__),'images':p['images']})
    with gpu_lease(gpu_peak_bytes=8<<30,disk_temp_bytes=1<<30,disk_persistent_bytes=8<<30) as gate:
        write(BASE/('control_compute_gate.json' if control else 'score_compute_gate.json'),gate)
        assert not set(gpu_compute_pids())-{os.getpid()}
        model=_load_model(MODEL,'cuda');engine=SyncNetEngine(batch_size=32,device='cuda')
        try:
            for r in rows:
                sid=r['id'];af=BASE/'audio_features'/(sid+'.npz')
                if not af.exists():
                    arrays={};meta={}
                    for a,info in r['audio'].items():
                        assert sha(info['path'])==info['sha256'];arrays[a],meta[a]=engine.extract_audio(info['path'])
                    af.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(af,**arrays)
                    write(af.with_suffix('.json'),{'sha256':sha(af),'metadata':meta,'audio':r['audio']})
                assert sha(af)==read(af.with_suffix('.json'))['sha256']
                aud=np.load(af)
                for im in p['images']:
                    if im['id']!='3' and (control or sid not in p['sensitivity_ids']):continue
                    assert sha(im['path'])==im['sha256']
                    frame=cv2.imread(im['path']);x1,y1,x2,y2=im['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));mask=face.copy();mask[48:]=0
                    it=torch.from_numpy(np.concatenate([mask,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
                    for a,info in r['audio'].items():
                        dest=BASE/('control_scores' if control else 'scores')/im['id']/sid/a
                        if dest.with_suffix('.json').exists():continue
                        assert not set(gpu_compute_pids())-{os.getpid()}
                        c=None if control else cache(r,im,a)
                        if c:v,meta=c
                        else:
                            mel=audio.melspectrogram(audio.load_wav(info['path'],16000)).astype(np.float32);chunks=chunk_mels(mel,25);frames=[]
                            for start in range(0,len(chunks),32):
                                mt=torch.from_numpy(np.asarray(chunks[start:start+32],dtype=np.float32)[:,None]).cuda()
                                with torch.inference_mode():pred=model(mt,it.expand(len(mt),-1,-1,-1)).cpu().numpy().transpose(0,2,3,1)*255
                                for fo in pred:
                                    full=frame.copy();full[y1:y2,x1:x2]=cv2.resize(fo.astype(np.uint8),(x2-x1,y2-y1));frames.append(crop_zero_padded(full,im['score_box']['box']))
                            vp=BASE/('control_videos' if control else 'videos')/im['id']/sid/(a+'.avi');render(np.stack(frames),np.arange(len(frames))/25,vp);v,vm=engine.extract_visual(vp)
                            meta={'video':str(vp),'video_sha256':sha(vp),'visual_metadata':vm,'mel_shape':list(mel.shape)}
                        m=engine.distance_matrix(v,aud[a]);dest.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(dest.with_suffix('.npz'),visual=v,matrix=m)
                        write(dest.with_suffix('.json'),{'id':sid,'speaker':r['speaker'],'image':im['id'],'arm':a,'status':'COMPLETE' if len(m)>40 else 'INSUFFICIENT_ROWS','metrics':metrics(m) if len(m)>40 else {},'sha256':sha(dest.with_suffix('.npz')),'audio_features_sha256':sha(af),**meta})
                        if control:
                            import soundfile as sf
                            pcm,sr=sf.read(info['path'],dtype='int16');assert sr==16000
                            base=np.load(BASE/'scores/3'/sid/(a+'.npz'))
                            controls={'visual_repeat_max':float(np.max(abs(base['visual']-v))),'matrix_repeat_max':float(np.max(abs(base['matrix']-m))),'delays':{}}
                            for shift in (-5,5):
                                moved=np.zeros_like(pcm);count=abs(shift)*640
                                if shift>0:moved[count:]=pcm[:-count]
                                else:moved[:-count]=pcm[count:]
                                wp=dest.with_name(a+f'_delay{shift:+d}.wav');sf.write(wp,moved,16000,subtype='PCM_16');ae,am=engine.extract_audio(wp);dm=engine.distance_matrix(v,ae)
                                dp=dest.with_name(a+f'_delay{shift:+d}.npz');np.savez_compressed(dp,matrix=dm,audio=ae)
                                controls['delays'][str(shift)]={'metrics':metrics(dm),'expected_offset':metrics(m)['20']['offset']-shift,'sha256':sha(dp),'pcm_sha256':sha(wp),'metadata':am}
                            write(dest.with_name(a+'_controls.json'),controls)
                print('scored',sid,flush=True)
        finally:engine.close()

def controls():score(control=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('manifest','align','score','controls'));globals()[p.parse_args().stage]()
