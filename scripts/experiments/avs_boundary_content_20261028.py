"""AVS_u boundary/context interventions, preserving target content exactly."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

NAME='avs_boundary_content_cloud_20261028_attempt01'
CONDITIONS=['native','reflect','repeat','crop_head','crop_mid','crop_tail']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def cosine(a,v):
    a,v=np.asarray(a,float),np.asarray(v,float)
    return (a*v).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(v,axis=1))


def freeze(repo,out):
    source=repo/'runs/common_window_curves_cloud_20261028_attempt01'
    rows=read(source/'inputs.json')
    assert len(rows)==80
    for r in rows:
        f=r['source_frames'];t=(f-5)//2
        assert t>=20 and t+25<=f
        for kind in ['audio','video']:assert sha(r[kind+'_path'])==r[kind+'_sha256']
        assert sha(r['avs_cache'])==r['avs_cache_sha256']
    out.mkdir(parents=True,exist_ok=False)
    p=dict(status='FROZEN_BEFORE_NEW_FORWARD_OR_LANDMARK_ANALYSIS',created_unix=time.time(),run=NAME,
       source_manifest_sha256=sha(source/'inputs.json'),script_sha256=sha(__file__),
       cohort='20N/60 cloudTTS Wav2Lip GAN native clips, 20 source clusters, no cross-track or new generation',
       encoder='same AV-HuBERT VSR Base LRS3 433h official backbone / project AVS_u replication',
       frontend='exact prior fullclip logfbank stack4 per-frame normalize and AutoAVSR4point crop -> grayscale88 transform',
       parity='all80 native A/V embeddings reproduce previous frozen cache maxabs<=1e-5 BEFORE per-item intervention',
       short_crop=dict(input_frames=25,target_frames=5,target='(F-5)//2 : +5',crop_starts='t,t-10,t-20',target_local_starts=[0,10,20],
           invariant='same original target preprocessed A/V tensors bit-exact in all three conditions; input length fixed',
           caveat='1-second stress-test changes real neighboring context and boundary position; not pure absolute-position or official benchmark'),
       extension=dict(pad_frames_each_side=75,types=['reflect','edge-repeat'],
           reason='saved config conv_pos128 temporal frames;75 moves original edge beyond halfwidth64',
           invariant='original F preprocessed frames copied exactly at75:75+F, both modalities same mapping',
           caveat='synthetic surrounding context and sequence length change; global attention can affect center too'),
       regions='head first15, tail last15, center15 at (F-15)//2, same original indices after extension',
       state=dict(low_energy='40ms RMS <0.1 wholeclipRMS; not VAD',energy_thresholds=[.05,.1,.2],
           closed='MediaPipeFaceMesh Euclidean innerlip13-14 / width78-308 <.05; proxy only',closed_thresholds=[.03,.05,.08],
           four_states=['quiet_closed','quiet_open','active_closed','active_open'],missing='keep invalid missing, never impute'),
       stats='all20/strict12; N and C source-equal means;10000 source bootstrap seed20261028 marginal95CI; missing cells report coverage',
       constraints='no model training, cross-track, DTW, original source mutation, or endpoint tuning; exploratory follow-up from known result')
    write(out/'protocol.json',p);write(out/'inputs.json',rows);shutil.copy2(__file__,out/'source_frozen.py')
    write(out/'status.json',dict(status='FROZEN',clips=80,frames_min=min(r['source_frames'] for r in rows)))
    print('FROZEN',min(r['source_frames'] for r in rows),max(r['source_frames'] for r in rows),flush=True)


def run(repo,out):
    import cv2
    import mediapipe as mp
    import torch
    from scipy.io import wavfile
    sys.path.insert(0,str(repo/'scripts/experiments'))
    import avs_u_zh_pilot_20261027 as prior
    torch.set_num_threads(4);torch.manual_seed(20261027)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    model,task,_=prior.load_encoder(Path('/home/wjj/avs-u-20261027'),out,'cuda')
    # Prior loader exposes its vendored frontend before this import.
    from python_speech_features import logfbank
    avutils=importlib.import_module('avhubert.utils')
    deploy=repo/'runs/cogenav_sync_audit_20261024_attempt01'
    detector=prior.module('boundary_detector',deploy/'autoavsr_detector.py').LandmarksDetector()
    cropper=prior.module('boundary_crop',deploy/'autoavsr_video_process.py').VideoProcess(convert_gray=False)
    transform=avutils.Compose([avutils.Normalize(0,255),avutils.CenterCrop((int(task.image_crop_size),int(task.image_crop_size))),avutils.Normalize(float(task.image_mean),float(task.image_std))])
    rows=read(out/'inputs.json');results=[];started_all=time.time()
    (out/'items').mkdir(exist_ok=True);(out/'qa').mkdir(exist_ok=True)

    def encode(a,v):
        with torch.inference_mode():
            ae=model.extract_finetune({'audio':torch.from_numpy(np.ascontiguousarray(a.T))[None].cuda(),'video':None},mask=False)[0][0].cpu().numpy()
            ve=model.extract_finetune({'audio':None,'video':torch.from_numpy(np.ascontiguousarray(v))[None,None].cuda()},mask=False)[0][0].cpu().numpy()
        return ae,ve

    for row in rows:
        key=row['id'];dest=out/'items'/f'{key}.npz';started=time.time()
        if dest.exists():results.append(read(dest.with_suffix('.json')));continue
        assert sha(row['video_path'])==row['video_sha256'] and sha(row['audio_path'])==row['audio_sha256']
        cap=cv2.VideoCapture(row['video_path']);frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release();f=len(frames);assert f==row['source_frames']
        marks=detector(frames);assert sum(m is not None for m in marks)/f>=.95
        patches=cropper(frames,marks)
        gray=np.stack([cv2.cvtColor(p,cv2.COLOR_RGB2GRAY) for p in patches])
        video=np.ascontiguousarray(transform(gray),dtype=np.float32)
        sr,pcm=wavfile.read(row['audio_path']);assert sr==16000 and pcm.dtype==np.int16
        feats=logfbank(pcm,samplerate=16000).astype(np.float32);order=int(task.stack_order_audio)
        if len(feats)%order:feats=np.pad(feats,((0,order-len(feats)%order),(0,0)))
        feats=feats.reshape(-1,feats.shape[1]*order)
        feats=np.pad(feats[:f],((0,max(0,f-len(feats))),(0,0)))
        audio=torch.from_numpy(feats)
        if task.normalize:audio=torch.nn.functional.layer_norm(audio,audio.shape[1:])
        audio=audio.numpy();native_a,native_v=encode(audio,video)
        old=np.load(row['avs_cache']);ae=float(abs(native_a-old['audio']).max());ve=float(abs(native_v-old['video']).max())
        if max(ae,ve)>1e-5:
            write(out/'native_parity_failure.json',dict(id=key,audio_max_error=ae,video_max_error=ve))
            raise AssertionError(('native parity',key,ae,ve))
        t=(f-5)//2;target=np.arange(t,t+5);crop_starts=np.array([t,t-10,t-20]);local_starts=np.array([0,10,20])
        regions=np.stack([np.arange(15),np.arange((f-15)//2,(f-15)//2+15),np.arange(f-15,f)])
        maps=dict(reflect=np.pad(np.arange(f),(75,75),mode='reflect'),repeat=np.pad(np.arange(f),(75,75),mode='edge'))
        arrays=dict(input_audio=audio,input_video=video,source_frame_indices=np.arange(f),target_indices=target,
            segment_indices=regions,crop_starts=crop_starts,target_local_starts=local_starts,
            native_audio=native_a,native_video=native_v,reflect_map=maps['reflect'],repeat_map=maps['repeat'])
        equality=[]
        for condition,mapping in maps.items():
            aa,vv=audio[mapping],video[mapping]
            assert np.array_equal(aa[75:75+f],audio) and np.array_equal(vv[75:75+f],video)
            arrays[condition+'_audio'],arrays[condition+'_video']=encode(aa,vv)
        for condition,start,local in zip(['crop_head','crop_mid','crop_tail'],crop_starts,local_starts):
            aa,vv=audio[start:start+25],video[start:start+25]
            assert np.array_equal(aa[local:local+5],audio[target]) and np.array_equal(vv[local:local+5],video[target])
            arrays[condition+'_audio'],arrays[condition+'_video']=encode(aa,vv)
            equality.append(dict(condition=condition,audio_exact=True,video_exact=True))
        # Independent visual-state proxy. Never use cropper's four face landmarks as a mouth aperture.
        points=np.full((f,4,2),np.nan,dtype=np.float32);valid=np.zeros(f,bool)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=False,min_detection_confidence=.5,min_tracking_confidence=.5) as mesh:
            for i,frame in enumerate(frames):
                result=mesh.process(frame)
                if not result.multi_face_landmarks:continue
                lm=result.multi_face_landmarks[0].landmark
                points[i]=[[lm[j].x*frame.shape[1],lm[j].y*frame.shape[0]] for j in [13,14,78,308]]
                valid[i]=True
        lip=np.linalg.norm(points[:,0]-points[:,1],axis=1)/np.linalg.norm(points[:,2]-points[:,3],axis=1)
        rms=np.sqrt(np.mean(pcm.astype(float)**2));energy=np.array([np.sqrt(np.mean(pcm[i*640:(i+1)*640].astype(float)**2))/rms for i in range(f)])
        assert np.isfinite(energy).all()
        arrays.update(lip_points=points,lip_valid=valid,lip_ratio=lip,frame_rms_ratio=energy)
        np.savez_compressed(dest,**arrays)
        scores={c:cosine(arrays[c+'_audio'],arrays[c+'_video']) for c in CONDITIONS}
        regional={c:{name:float(scores[c][ix+(75 if c in maps else 0)].mean()) for name,ix in zip(['head','center','tail'],regions)} for c in ['native','reflect','repeat']}
        target_scores={c:float(scores[c][local:local+5].mean()) for c,local in zip(['crop_head','crop_mid','crop_tail'],local_starts)}
        item=dict(id=key,source=row['sample_id'],arm=row['arm'],strict_reference=row['strict_reference'],frames=f,
            native_parity=dict(audio_max_error=ae,video_max_error=ve),target_input_equality=equality,
            valid_landmark_frames=int(valid.sum()),regional_scores=regional,target_scores=target_scores,
            npz_sha256=sha(dest),elapsed_s=time.time()-started)
        write(dest.with_suffix('.json'),item);results.append(item)
        if row==rows[0] or key.endswith('_C1') or key.endswith('_N_0'):
            indexes=[0,min(7,f-1),f//2,max(0,f-8),f-1];panels=[]
            for i in indexes:
                image=cv2.cvtColor(frames[i],cv2.COLOR_RGB2BGR)
                if valid[i]:
                    for j in range(4):cv2.circle(image,tuple(np.rint(points[i,j]).astype(int)),2,(0,0,255),-1)
                text=f"{i} lip={lip[i]:.3f} E={energy[i]:.2f}"
                cv2.putText(image,text,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1)
                panels.append(cv2.resize(image,(256,256)))
            cv2.imwrite(str(out/'qa'/f'{key}.jpg'),np.concatenate(panels,axis=1))
        write(out/'status.json',dict(status='RUNNING',scored=len(results),total=80,last=key,pid=os.getpid(),elapsed_s=time.time()-started_all))
        print('ITEM',len(results),key,round(time.time()-started,2),'parity',ae,ve,flush=True)
    write(out/'items.json',results);write(out/'status.json',dict(status='COMPLETE_ENCODINGS',scored=80,elapsed_s=time.time()-started_all))
    analyze(out)


def interval(values):
    x=np.array(values,float);x=x[np.isfinite(x)]
    if not len(x):return dict(mean=None,ci95=None,sources=0)
    ii=np.random.default_rng(20261028).integers(0,len(x),(10000,len(x)))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[ii].mean(1),[.025,.975]).tolist(),sources=len(x))


def analyze(out):
    items=read(out/'items.json');rows=[]
    for item in items:
        z=np.load(out/'items'/f"{item['id']}.npz");s=cosine(z['native_audio'],z['native_video']);regions=z['segment_indices']
        values={}
        for c in ['reflect','repeat']:
            for region in ['head','center','tail']:
                values[c+'_'+region+'_change']=item['regional_scores'][c][region]-item['regional_scores']['native'][region]
            values[c+'_edge_minus_center_change']=(values[c+'_head_change']+values[c+'_tail_change'])/2-values[c+'_center_change']
        ts=item['target_scores'];values.update(crop_head_minus_mid=ts['crop_head']-ts['crop_mid'],crop_tail_minus_mid=ts['crop_tail']-ts['crop_mid'],crop_edges_minus_mid=(ts['crop_head']+ts['crop_tail'])/2-ts['crop_mid'])
        states={}
        for eth in [.05,.1,.2]:
            for lth in [.03,.05,.08]:
                quiet=z['frame_rms_ratio']<eth;closed=z['lip_ratio']<lth;valid=z['lip_valid']
                key=f'E{eth}_L{lth}';states[key]={}
                for name,ix in [('edge',np.r_[regions[0],regions[2]]),('center',regions[1])]:
                    masks=dict(quiet_closed=quiet&closed,quiet_open=quiet&~closed,active_closed=~quiet&closed,active_open=~quiet&~closed)
                    states[key][name]={state:dict(frames=int((mask[ix]&valid[ix]).sum()),score=float(s[ix][mask[ix]&valid[ix]].mean()) if (mask[ix]&valid[ix]).any() else None) for state,mask in masks.items()}
        rows.append(dict(id=item['id'],source=item['source'],arm=item['arm'],strict_reference=item['strict_reference'],values=values,states=states))
    summary={}
    for view,subset in [('all20',rows),('strict12',[r for r in rows if r['strict_reference']])]:
        summary[view]={}
        for arm in ['N','C']:
            rs=[r for r in subset if r['arm']==arm];sources=sorted({r['source'] for r in rs})
            armstats={field:interval([np.mean([r['values'][field] for r in rs if r['source']==source]) for source in sources]) for field in rs[0]['values']}
            state_stats={}
            for setting in rs[0]['states']:
                state_stats[setting]={}
                for state in ['quiet_closed','quiet_open','active_closed','active_open']:
                    deltas=[];coverage={}
                    for source in sources:
                        diffs=[]
                        for r in rs:
                            if r['source']!=source:continue
                            e,c=r['states'][setting]['edge'][state],r['states'][setting]['center'][state]
                            if e['frames'] and c['frames']:diffs.append(e['score']-c['score'])
                        if diffs:deltas.append(float(np.mean(diffs)))
                        coverage[source]=len(diffs)
                    state_stats[setting][state]=dict(**interval(deltas),matched_clip_counts=coverage)
            summary[view][arm]=dict(clips=len(rs),sources=len(sources),effects=armstats,state_edge_minus_center=state_stats)
    write(out/'analysis_rows.json',rows);write(out/'summary.json',summary)
    write(out/'status.json',dict(status='COMPLETE_ENCODINGS_AND_ANALYSIS',scored=len(items)))
    print('ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run','analyze']);p.add_argument('--repo',type=Path,default=Path('/home/wjj/tts-audio/tts-exp'));a=p.parse_args();o=a.repo/'runs'/NAME
    if a.stage=='freeze':freeze(a.repo,o)
    elif a.stage=='run':run(a.repo,o)
    else:analyze(o)
