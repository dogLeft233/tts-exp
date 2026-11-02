"""Nine-cell SyncNet with source-owned ROI and one original PCM per clip."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
import python_speech_features
from scipy import signal

from common import ROOT,DATA,label,sha,save_json

SYNC=Path('/home/wjj/tts-audio/tts-exp/third_party/syncnet_python')
sys.path.insert(0,str(SYNC))
from SyncNetModel import S
from SyncNetInstance import calc_pdist
from detectors import S3FD


def crops(images,track):
    output=[]
    for image,(s,x,y) in zip(images,track):
        cs=.4;bsi=int(s*(1+2*cs))
        padded=np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)),constant_values=110)
        mx=x+bsi;my=y+bsi
        face=padded[int(my-s):int(my+s*(1+2*cs)),int(mx-s*(1+cs)):int(mx+s*(1+cs))]
        output.append(cv2.resize(face,(224,224)))
    return np.stack(output)


def summarize(matrix,rows):
    curve=matrix[rows].mean(0);value,index=torch.min(curve,0)
    return {'sync_c':float(torch.median(curve)-value),'sync_d':float(value),
            'distance_d0':float(curve[15]),'background_median':float(torch.median(curve)),
            'av_offset':int(15-index),'windows':len(rows),'curve':curve.numpy().tolist()}


def decode(path,metadata):
    cap=cv2.VideoCapture(str(path));assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==25
    images=[]
    while True:
        ok,x=cap.read()
        if not ok:break
        rgb=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==metadata['frames'][len(images)]['rgb_sha256']
        images.append(x)
    cap.release();assert len(images)==1100
    return np.stack(images)


def main():
    output=ROOT/'scores';assert not output.exists();output.mkdir()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.chdir(SYNC)
    manifest=json.loads((DATA/'manifest.json').read_text())
    clips=[c for c in manifest['eval_clips'] if c['split']=='test'];assert len(clips)==4
    info=next(s for s in manifest['splits'] if s['name']=='test')
    lo=info['supervision_start_frame'];count=info['supervision_frames']
    detector=S3FD(device='cuda');boxes=[];fallback=[]
    for i in range(count):
        image=cv2.imread(str(DATA/f'source/May/gt_imgs/{lo+i}.jpg'))
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        found=detector.detect_faces(rgb,conf_th=.9,scales=[.25])
        if len(found)==0:
            found=detector.detect_faces(rgb,conf_th=.9,scales=[1.]);fallback.append(i)
        assert len(found)>0,f'ground-truth face track missing at {i}; retain failure'
        box=max(found,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        boxes.append(box[:4].tolist())
    del detector;torch.cuda.empty_cache()
    boxes=np.array(boxes)
    size=np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2
    x=(boxes[:,0]+boxes[:,2])/2;y=(boxes[:,1]+boxes[:,3])/2
    track=np.stack([signal.medfilt(v,13) for v in [size,x,y]],1)
    np.save(output/'shared_gt_track.npy',track);np.save(output/'gt_boxes.npy',boxes)
    model=S(num_layers_in_fc_layers=1024).cuda().eval()
    weights=SYNC/'data/syncnet_v2.model'
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    audio_features={};audio_pcm_sha={}
    for clip in clips:
        folder=output/clip['id'];folder.mkdir()
        pcm,sr=sf.read(DATA/clip['score_audio_path'],dtype='int16')
        assert sr==16000 and len(pcm)==160000 and sha(DATA/clip['score_audio_path'])==clip['score_audio_sha256']
        audio_pcm_sha[clip['id']]=hashlib.sha256(pcm.tobytes()).hexdigest()
        mfcc=python_speech_features.mfcc(pcm,sr).T
        audio=torch.from_numpy(mfcc[None,None].astype(np.float32));n=clip['frames']-5
        rows=[]
        with torch.no_grad():
            for i in range(0,n,20):
                batch=torch.cat([audio[:,:,:,j*4:j*4+20] for j in range(i,min(i+20,n))])
                rows.append(model.forward_aud(batch.cuda()).cpu())
        af=torch.cat(rows);audio_features[clip['id']]=af
        np.save(folder/'original_mfcc.npy',mfcc);np.save(folder/'original_audio_features.npy',af.numpy())
    result={'protocol_sha256':sha(ROOT/'protocol.json'),'syncnet_weight_sha256':sha(weights),
            'network_source_sha256':sha(SYNC/'SyncNetModel.py'),'instance_source_sha256':sha(SYNC/'SyncNetInstance.py'),
            'crop_owner':'ground truth original test frames; identical across all models/gains',
            'fallback_detection_frames':fallback,'missing_detection_frames':[],
            'strict_weights':True,'precision':'FP32','cells':[],'repeat':None}
    for training_gain in [0,-6,6]:
        for testing_gain in [-6,0,6]:
            cell=ROOT/'renders'/('train_'+label(training_gain))/('test_'+label(testing_gain))
            metadata=json.loads((cell/'render_receipt.json').read_text())
            pcm=sf.read(cell.parent/'original_test_pcm.wav',dtype='int16')[0]
            assert hashlib.sha256(pcm.tobytes()).hexdigest()==metadata['pcm_sha256']
            mux_audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(cell/'original_pcm.mkv'),
                                '-map','0:a:0','-f','s16le','-acodec','pcm_s16le','pipe:1'])
            assert mux_audio==pcm.tobytes()
            images=decode(cell/'original_pcm.mkv',metadata)
            cell_rows=[]
            for clip in clips:
                start=clip['source_start_frame']-lo;n=clip['frames']-5
                folder=output/clip['id']/f'train_{label(training_gain)}_test_{label(testing_gain)}';folder.mkdir()
                cropped=crops(images[start:start+clip['frames']],track[start:start+clip['frames']])
                np.save(folder/'crops.npy',cropped)
                visual=torch.from_numpy(cropped).permute(3,0,1,2).unsqueeze(0).float()
                vf=[]
                with torch.no_grad():
                    for i in range(0,n,20):
                        batch=torch.cat([visual[:,:,j:j+5] for j in range(i,min(i+20,n))])
                        vf.append(model.forward_lip(batch.cuda()).cpu())
                vf=torch.cat(vf);af=audio_features[clip['id']]
                matrix=torch.stack(calc_pdist(vf,af,vshift=15))
                np.save(folder/'visual_features.npy',vf.numpy());np.save(folder/'distances.npy',matrix.numpy())
                row={'clip_id':clip['id'],'training_gain_db':training_gain,'test_gain_db':testing_gain,
                     'source_start_frame':clip['source_start_frame'],'source_pcm_sha256':audio_pcm_sha[clip['id']],
                     'score_audio_sha256':clip['score_audio_sha256'],
                     'guard20':summarize(matrix,np.arange(20,n-20)),'full':summarize(matrix,np.arange(n)),
                     'distances_sha256':sha(folder/'distances.npy')}
                cell_rows.append(row)
                if training_gain==testing_gain==0 and clip==clips[0]:
                    cap=cv2.VideoCapture(str(cell/'repeat_first_clip.mkv'));rep=[]
                    while True:
                        ok,bgr=cap.read()
                        if not ok:break
                        rep.append(bgr)
                    cap.release();assert np.array_equal(np.stack(rep),images[start:start+250])
                    rc=crops(np.stack(rep),track[start:start+250])
                    rv=torch.from_numpy(rc).permute(3,0,1,2).unsqueeze(0).float();rf=[]
                    with torch.no_grad():
                        for i in range(0,n,20):
                            batch=torch.cat([rv[:,:,j:j+5] for j in range(i,min(i+20,n))])
                            rf.append(model.forward_lip(batch.cuda()).cpu())
                    rmatrix=torch.stack(calc_pdist(torch.cat(rf),af,vshift=15))
                    assert torch.equal(rmatrix,matrix)
                    np.save(folder/'repeat_distances.npy',rmatrix.numpy())
                    result['repeat']={'clip_id':clip['id'],'rgb_exact':True,'matrix_max_error':0.0}
            entry={'training_gain_db':training_gain,'test_gain_db':testing_gain,'clips':cell_rows,
                   'mean_guard20':{k:float(np.mean([r['guard20'][k] for r in cell_rows])) for k in
                         ['sync_c','sync_d','distance_d0','background_median']},
                   'mean_full':{k:float(np.mean([r['full'][k] for r in cell_rows])) for k in
                         ['sync_c','sync_d','distance_d0','background_median']}}
            result['cells'].append(entry);save_json(output/'scores.json',result)
            print(json.dumps({'scored_train':training_gain,'scored_test':testing_gain,'mean_guard20':entry['mean_guard20']}),flush=True)
            del images;torch.cuda.empty_cache()
    assert len(result['cells'])==9 and result['repeat']['matrix_max_error']==0
    save_json(output/'scores.json',result)


if __name__=='__main__':main()
