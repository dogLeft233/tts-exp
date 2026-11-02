"""Paired SyncNet V2 with one baseline-owned crop and identical original PCM."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import python_speech_features
import soundfile as sf
import torch
from scipy import signal

SYNC=Path('/home/wjj/tts-audio/tts-exp/third_party/syncnet_python')
sys.path.insert(0,str(SYNC))
from SyncNetModel import S
from SyncNetInstance import calc_pdist
from detectors import S3FD


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def decode(path, metadata):
    cap=cv2.VideoCapture(str(path));assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FPS)==25
    frames=[]
    while True:
        ok,x=cap.read()
        if not ok:break
        frames.append(x)
    cap.release();x=np.stack(frames)
    assert len(x)==300
    for i,frame in enumerate(x):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==metadata['frames'][i]['rgb_sha256']
    return x


def crops(images, track):
    output=[]
    for image, (s,x,y) in zip(images,track):
        cs=.4;bsi=int(s*(1+2*cs))
        padded=np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)),constant_values=110)
        mx=x+bsi;my=y+bsi
        face=padded[int(my-s):int(my+s*(1+2*cs)),int(mx-s*(1+cs)):int(mx+s*(1+cs))]
        output.append(cv2.resize(face,(224,224)))
    return np.stack(output)


def summarize(matrix,rows):
    curve=matrix[rows].mean(0)
    value,index=torch.min(curve,0)
    return {'sync_c':float(torch.median(curve)-value),'sync_d':float(value),
            'distance_d0':float(curve[15]),'background_median':float(torch.median(curve)),
            'av_offset':int(15-index),'windows':len(rows),'curve':curve.numpy().tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    receipt=json.loads((a.render/'render_receipt.json').read_text());assert len(receipt['arms'])==4
    assert not receipt['smoke']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    os.chdir(SYNC)  # S3FD's official weight path is relative to its repository.
    wav=a.render/'original_score_audio.wav';pcm,sr=sf.read(wav,dtype='int16')
    assert sr==16000 and len(pcm)==192000
    assert hashlib.sha256(pcm.tobytes()).hexdigest()==receipt['score_pcm_sha256']
    meta={x['arm']:x for x in receipt['arms']}
    baseline=decode(a.render/'gain_0/render_lossless.mkv',meta['gain_0'])
    detector=S3FD(device='cuda')
    boxes=[];fallbacks=[];missing=[]
    for i,image in enumerate(baseline):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        detected=detector.detect_faces(rgb,conf_th=.9,scales=[.25])
        if len(detected)==0:
            detected=detector.detect_faces(rgb,conf_th=.9,scales=[1.0]);fallbacks.append(i)
        if len(detected)==0:boxes.append([np.nan]*4);missing.append(i)
        else:
            picked=max(detected,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
            boxes.append(picked[:4].tolist())
        if i%50==0:print('Baseline face detection',i,flush=True)
    del detector;torch.cuda.empty_cache()
    boxes=np.asarray(boxes,dtype=float);valid=np.isfinite(boxes).all(1)
    assert valid.sum()>=275 and valid[0] and valid[-1], 'Face-track coverage failed'
    for k in range(4):boxes[:,k]=np.interp(np.arange(300),np.flatnonzero(valid),boxes[valid,k])
    size=np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2
    x=(boxes[:,0]+boxes[:,2])/2;y=(boxes[:,1]+boxes[:,3])/2
    track=np.stack([signal.medfilt(v,13) for v in [size,x,y]],1)
    np.save(a.output/'shared_track.npy',track);np.save(a.output/'baseline_boxes.npy',boxes)
    weights=SYNC/'data/syncnet_v2.model'
    model=S(num_layers_in_fc_layers=1024).cuda().eval()
    state=torch.load(weights,map_location='cpu',weights_only=True)
    loaded=model.load_state_dict(state,strict=True)
    mfcc=python_speech_features.mfcc(pcm,sr).T
    np.save(a.output/'source_mfcc.npy',mfcc)
    audio=torch.from_numpy(mfcc[None,None].astype(np.float32))
    n=295;batch_size=20
    with torch.no_grad():
        af=[]
        for i in range(0,n,batch_size):
            inp=torch.cat([audio[:,:,:,j*4:j*4+20] for j in range(i,min(i+batch_size,n))])
            af.append(model.forward_aud(inp.cuda()).cpu())
        af=torch.cat(af)
    np.save(a.output/'source_audio_features.npy',af.numpy())
    results={'protocol_sha256':sha(a.protocol),'score_pcm_sha256':receipt['score_pcm_sha256'],
             'syncnet_weight_sha256':sha(weights),'strict_load_missing':list(loaded.missing_keys),
             'strict_load_unexpected':list(loaded.unexpected_keys),'dtype':'float32',
             'network_source_sha256':sha(SYNC/'SyncNetModel.py'),
             'instance_source_sha256':sha(SYNC/'SyncNetInstance.py'),
             'track_sha256':sha(a.output/'shared_track.npy'),'fallback_detection_frames':fallbacks,
             'interpolated_detection_frames':missing,'arms':[]}
    for arm in ['gain_m6','gain_0','gain_0_repeat','gain_p6']:
        container=a.render/arm/'original_pcm.mkv'
        decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(container),'-map','0:a:0',
                                         '-f','s16le','-acodec','pcm_s16le','-ac','1','-ar','16000','pipe:1'])
        assert decoded==pcm.tobytes(), 'Scoring mux PCM changed'
        images=baseline if arm=='gain_0' else decode(a.render/arm/'render_lossless.mkv',meta[arm])
        cropped=crops(images,track)
        np.save(a.output/(arm+'_crops.npy'),cropped)
        visual=torch.from_numpy(cropped).permute(3,0,1,2).unsqueeze(0).float()
        with torch.no_grad():
            vf=[]
            for i in range(0,n,batch_size):
                inp=torch.cat([visual[:,:,j:j+5] for j in range(i,min(i+batch_size,n))])
                vf.append(model.forward_lip(inp.cuda()).cpu())
            vf=torch.cat(vf)
        matrix=torch.stack(calc_pdist(vf,af,vshift=15),0)
        np.save(a.output/(arm+'_visual_features.npy'),vf.numpy())
        np.save(a.output/(arm+'_distances.npy'),matrix.numpy())
        delta=images.astype(np.int16)-baseline.astype(np.int16)
        pixel_rows=[]
        for i,d in enumerate(delta):
            lms=np.loadtxt(Path('/dev/shm/synctalk-may-gain-inference-20261016/data/May/ori_imgs')/f'{5520+i}.lms')
            mouth=lms[48:68]
            xmin=max(0,int(mouth[:,0].min())-12);xmax=min(512,int(mouth[:,0].max())+13)
            ymin=max(0,int(mouth[:,1].min())-12);ymax=min(512,int(mouth[:,1].max())+13)
            m=d[ymin:ymax,xmin:xmax]
            pixel_rows.append({'frame':i,'full_mean_abs':float(np.abs(d).mean()),
                               'full_max_abs':int(np.abs(d).max()),'changed_pixel_fraction':float(np.any(d!=0,axis=2).mean()),
                               'mouth_box':[xmin,xmax,ymin,ymax],'mouth_mean_abs':float(np.abs(m).mean())})
        row={'arm':arm,'full':summarize(matrix,np.arange(n)),
             'guard20':summarize(matrix,np.arange(20,n-20)),
             'pixels':{'full_mean_abs':float(np.abs(delta).mean()),'full_max_abs':int(np.abs(delta).max()),
                       'changed_pixel_fraction':float(np.any(delta!=0,axis=3).mean()),
                       'mouth_mean_abs':float(np.mean([x['mouth_mean_abs'] for x in pixel_rows]))},
             'pixel_frames':pixel_rows,'pcm_readback_exact':True,
             'visual_sha256':sha(a.output/(arm+'_visual_features.npy')),
             'distances_sha256':sha(a.output/(arm+'_distances.npy'))}
        results['arms'].append(row)
        (a.output/'scores.json').write_text(json.dumps(results,indent=2))
        print('Scored',arm,'guard20',row['guard20'],'pixels',row['pixels'],flush=True)
    by={x['arm']:x for x in results['arms']}
    results['zero_repeat_pixels_exact']=meta['gain_0']['rgb_stream_sha256']==meta['gain_0_repeat']['rgb_stream_sha256']
    results['zero_repeat_score_max_abs']=float(np.max(np.abs(np.load(a.output/'gain_0_distances.npy')-
                                                          np.load(a.output/'gain_0_repeat_distances.npy'))))
    for row in results['arms']:
        for scope in ['guard20','full']:
            row[scope]['delta_sync_c_vs_zero']=row[scope]['sync_c']-by['gain_0'][scope]['sync_c']
            row[scope]['delta_sync_d_vs_zero']=row[scope]['sync_d']-by['gain_0'][scope]['sync_d']
    (a.output/'scores.json').write_text(json.dumps(results,indent=2))
    print('SCORE_COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
