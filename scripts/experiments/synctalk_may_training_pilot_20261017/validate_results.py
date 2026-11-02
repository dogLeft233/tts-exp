"""Independent verification of training controls, final predictions and scores."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from scipy.signal import medfilt


class ContractError(ValueError):pass


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def verify_training(rows,steps,initial):
    if {r['gain_db'] for r in rows}!={-6,0,6} or len(rows)!=3:raise ContractError('missing training arms')
    if any(r['successful_optimizer_updates']!=steps for r in rows):raise ContractError('unequal training budget')
    if any(r['initial_state_sha256']!=initial for r in rows):raise ContractError('unequal initialization')
    if len({r['actual_ray_schedule_sha256'] for r in rows})!=1:raise ContractError('unequal actual rays')
    if len({r['frame_order_sha256'] for r in rows})!=1:raise ContractError('unequal frame schedule')


def verify_pcm(actual,expected):
    if actual!=expected:raise ContractError('scoring PCM differs from original source clock')


def independent_matrix(visual,audio):
    a=np.pad(np.asarray(audio,dtype=np.float64),((15,15),(0,0)))
    result=[]
    for i,x in enumerate(np.asarray(visual,dtype=np.float64)):
        result.append(np.sqrt(np.sum((x[None]-a[i:i+31]+1e-6)**2,axis=1)))
    return np.stack(result)


def metric(matrix,guard):
    x=matrix[20:-20] if guard else matrix
    curve=x.mean(0);idx=int(curve.argmin())
    return {'sync_c':float(np.median(curve)-curve[idx]),'sync_d':float(curve[idx]),
            'distance_d0':float(curve[15]),'background_median':float(np.median(curve)),
            'av_offset':15-idx}


def reconstruct(pred,truth,landmarks):
    delta=pred.astype(np.float64)-truth.astype(np.float64)
    x0,y0=np.maximum(np.floor(landmarks[48:68].min(0)).astype(int)-12,0)
    x1,y1=np.minimum(np.ceil(landmarks[48:68].max(0)).astype(int)+13,512)
    mouth=delta[y0:y1,x0:x1]
    return {'MSE':float(np.mean(delta**2)),'L1':float(np.mean(np.abs(delta))),
            'mouth_bbox_MSE':float(np.mean(mouth**2)),'mouth_bbox_L1':float(np.mean(np.abs(mouth)))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    result_path=a.root/'independent_results_qa.json';assert not result_path.exists()
    protocol=json.loads((a.root/'protocol.json').read_text())
    init=json.loads((a.root/'initialization.json').read_text())
    assert sha(a.root/'initialization.pt')==init['file_sha256']
    initial_state=torch.load(a.root/'initialization.pt',map_location='cpu',weights_only=False)['model']
    initial_digest=hashlib.sha256()
    for name,value in sorted(initial_state.items()):
        x=value.detach().cpu().contiguous()
        initial_digest.update(name.encode()+b'\0'+str(x.dtype).encode()+str(tuple(x.shape)).encode())
        initial_digest.update(x.numpy().tobytes())
    assert initial_digest.hexdigest()==init['tensor_state_sha256']
    supervisor=json.loads((a.root/'supervisor_receipt.json').read_text())
    assert sha(a.root/'protocol.json')==supervisor['protocol_sha256']
    for name,digest in supervisor['source_sha256'].items():assert sha(a.root/name)==digest
    admission=json.loads((a.root/'postprocess_admission.json').read_text())
    assert sha(a.root/'protocol.json')==admission['protocol_sha256']
    for name,digest in admission['source_sha256'].items():assert sha(a.root/name)==digest
    manifest=json.loads((a.data/'manifest.json').read_text())
    score=json.loads((a.root/'scores/scores.json').read_text())
    assert len(score['cells'])==9 and score['precision']=='FP32' and score['strict_weights']
    boxes=np.load(a.root/'scores/gt_boxes.npy')
    track=np.load(a.root/'scores/shared_gt_track.npy')
    sizes=np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2
    centers_x=(boxes[:,0]+boxes[:,2])/2;centers_y=(boxes[:,1]+boxes[:,3])/2
    assert np.array_equal(track,np.stack([medfilt(x,13) for x in [sizes,centers_x,centers_y]],axis=1))
    labels={-6:'gain_m6',0:'gain_0',6:'gain_p6'}
    training=[json.loads((a.root/'training'/labels[g]/'training_receipt.json').read_text()) for g in [-6,0,6]]
    verify_training(training,protocol['steps_per_model'],init['tensor_state_sha256'])
    report={'passed':False,'training_controls':True,'training_steps_each':protocol['steps_per_model'],
            'training_models':3,'test_cells':0,'test_video_frames':0,'validation_float_frames':0,
            'score_matrices':0,'distance_cells':0,'max_distance_error':0.,'max_metric_error':0.,
            'final_curve_max_MSE_error':0.,'source_pcm_exact':True,'convergence_is_not_assumed':True}
    for r in training:
        folder=a.root/'training'/labels[r['gain_db']]
        assert sha(folder/'ray_indices.sha256bin')==r['actual_ray_schedule_sha256']
        assert (folder/'ray_indices.sha256bin').stat().st_size==60000*32
        assert sha(a.root/'frame_order.npy')==r['frame_order_sha256']
        checkpoint=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        assert checkpoint['step']==60000 and checkpoint['gain_db']==r['gain_db']
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        assert checkpoint['ema']['num_updates']==60
        optimizer_steps=[float(s['step']) for s in checkpoint['optimizer']['state'].values() if 'step' in s]
        assert optimizer_steps and min(optimizer_steps)==max(optimizer_steps)==60000
        curve=json.loads((folder/'validation_curve.json').read_text())
        assert {(r['step'],r['test_gain_db']) for r in curve}=={
            (s,g) for s in range(0,60001,2000) for g in [-6,0,6]}
        assert len(curve)==93 and all(np.isfinite(r['MSE']) for r in curve)
    source,sr=sf.read(a.data/'source/May/aud.wav',dtype='int16',always_2d=True)
    assert sr==16000 and np.array_equal(source[:,0],source[:,1]);source=source[:,0]
    test=next(s for s in manifest['splits'] if s['name']=='test')
    val=next(s for s in manifest['splits'] if s['name']=='val')
    lo,hi=test['supervision_start_frame'],test['supervision_end_frame_exclusive']
    original=source[lo*640:hi*640].tobytes()
    metrics_by_cell={}
    for train_gain in [-6,0,6]:
        for test_gain in [-6,0,6]:
            folder=a.root/'renders'/('train_'+labels[train_gain])/('test_'+labels[test_gain])
            receipt=json.loads((folder/'render_receipt.json').read_text())
            assert receipt['training_gain_db']==train_gain and receipt['test_gain_db']==test_gain
            assert receipt['checkpoint_step']==60000 and len(receipt['frames'])==1100
            assert sha(folder/'original_pcm.mkv')==receipt['mux_sha256']
            mux=subprocess.check_output(['ffmpeg','-v','error','-i',str(folder/'original_pcm.mkv'),
                       '-map','0:a:0','-f','s16le','-acodec','pcm_s16le','pipe:1'])
            verify_pcm(mux,original)
            cap=cv2.VideoCapture(str(folder/'original_pcm.mkv'));n=0
            all_bgr=[]
            while True:
                ok,bgr=cap.read()
                if not ok:break
                row=receipt['frames'][n];assert row['index']==n and row['source_id']==lo+n
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                all_bgr.append(bgr)
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['rgb_sha256']
                truth=cv2.cvtColor(cv2.imread(str(a.data/f'source/May/gt_imgs/{lo+n}.jpg')),cv2.COLOR_BGR2RGB)
                truth=truth.astype(np.float32)/255
                lms=np.loadtxt(a.data/f'source/May/ori_imgs/{lo+n}.lms')
                checked=reconstruct(rgb.astype(np.float64)/255,truth,lms)
                epsilon=.5/255+1e-6
                for mse,l1 in [('MSE','L1'),('mouth_bbox_MSE','mouth_bbox_L1')]:
                    assert abs(checked[l1]-row[l1])<=epsilon
                    assert abs(checked[mse]-row[mse])<=2*epsilon*checked[l1]+epsilon**2+1e-7
                n+=1
            cap.release();assert n==1100;report['test_video_frames']+=n
            v=json.loads((folder/'full_validation.json').read_text())
            samples=np.load(folder/'final_validation_selected_float.npy',allow_pickle=False)
            assert samples.shape==(32,512,512,3) and samples.dtype==np.float32 and np.isfinite(samples).all()
            expected_curve=[]
            for i,index in enumerate(v['selected_indices']):
                source_id=val['supervision_start_frame']+index
                truth=cv2.cvtColor(cv2.imread(str(a.data/f'source/May/gt_imgs/{source_id}.jpg')),cv2.COLOR_BGR2RGB)
                truth=truth.astype(np.float32)/255
                lms=np.loadtxt(a.data/f'source/May/ori_imgs/{source_id}.lms')
                computed=reconstruct(samples[i],truth,lms)
                for key,value in computed.items():assert abs(value-v['frames'][index][key])<1e-10
                expected_curve.append(computed['MSE'])
            curve=json.loads((a.root/'training'/labels[train_gain]/'validation_curve.json').read_text())
            final=next(r for r in curve if r['step']==60000 and r['test_gain_db']==test_gain)
            error=abs(float(np.mean(expected_curve))-final['MSE'])
            assert error<1e-6;report['final_curve_max_MSE_error']=max(report['final_curve_max_MSE_error'],error)
            report['validation_float_frames']+=32
            entry=next(r for r in score['cells'] if r['training_gain_db']==train_gain and r['test_gain_db']==test_gain)
            assert len(entry['clips'])==4
            independently=[]
            for row in entry['clips']:
                clip=next(c for c in manifest['eval_clips'] if c['id']==row['clip_id'])
                folder_score=a.root/'scores'/clip['id']/f'train_{labels[train_gain]}_test_{labels[test_gain]}'
                af=np.load(a.root/'scores'/clip['id']/'original_audio_features.npy',allow_pickle=False)
                vf=np.load(folder_score/'visual_features.npy',allow_pickle=False)
                matrix=np.load(folder_score/'distances.npy',allow_pickle=False)
                crop_array=np.load(folder_score/'crops.npy',mmap_mode='r',allow_pickle=False)
                assert crop_array.shape==(250,224,224,3) and crop_array.dtype==np.uint8
                start=clip['source_start_frame']-lo
                for i in range(250):
                    s,x,y=track[start+i];pad=int(s*1.8)
                    image=np.pad(all_bgr[start+i],((pad,pad),(pad,pad),(0,0)),constant_values=110)
                    cx=x+pad;cy=y+pad
                    extracted=image[int(cy-s):int(cy+s*1.8),int(cx-s*1.4):int(cx+s*1.4)]
                    assert np.array_equal(cv2.resize(extracted,(224,224)),crop_array[i])
                assert matrix.shape==(245,31) and sha(folder_score/'distances.npy')==row['distances_sha256']
                original_clip=source[clip['source_start_frame']*640:(clip['source_start_frame']+250)*640]
                assert hashlib.sha256(original_clip.tobytes()).hexdigest()==row['source_pcm_sha256']
                actual_clip=sf.read(a.data/clip['score_audio_path'],dtype='int16')[0]
                verify_pcm(actual_clip.tobytes(),original_clip.tobytes())
                rebuilt=independent_matrix(vf,af);delta=float(np.max(np.abs(matrix-rebuilt)))
                assert delta<1e-5;report['max_distance_error']=max(report['max_distance_error'],delta)
                independently.append(metric(rebuilt,True))
                for guard in [True,False]:
                    scope='guard20' if guard else 'full';values=metric(rebuilt,guard)
                    for key,value in values.items():
                        error=abs(value-row[scope][key]);assert error<1e-5
                        report['max_metric_error']=max(report['max_metric_error'],error)
                repeat=folder_score/'repeat_distances.npy'
                if repeat.exists():assert np.array_equal(np.load(repeat),matrix)
                report['score_matrices']+=1;report['distance_cells']+=matrix.size
            for key in ['sync_c','sync_d','distance_d0','background_median']:
                assert abs(np.mean([x[key] for x in independently])-entry['mean_guard20'][key])<1e-5
            metrics_by_cell[f'{train_gain}/{test_gain}']=entry['mean_guard20']
            report['test_cells']+=1
    assert score['repeat']['rgb_exact'] and score['repeat']['matrix_max_error']==0
    report['repeat_passed']=True;report['passed']=True
    result_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
