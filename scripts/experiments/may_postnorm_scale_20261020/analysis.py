"""Independent float64 matrix and RGB/PCM/GT/crop consumer of the 12 scale cells."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np,cv2,soundfile as sf
from scipy.signal import medfilt
from protocol import OUT,DATA,OLD,sha,write
from consumer_common import matrix,metric,truth,export,FFMPEG
from scaling import validate_cells
def main():
    p=json.loads((OUT/'protocol.json').read_text());parent=Path(p['parent'])
    assert json.loads((OUT/'baseline_state_review.json').read_text())['passed'] and json.loads((OUT/'training_independent_review.json').read_text())['passed']
    seal=json.loads((parent/'final_delivery_receipt.json').read_text());assert seal['passed'] and sha(parent/'final_delivery_receipt.json')==p['parent_delivery_receipt_sha256']
    baseline=json.loads((parent/'analysis/cells.json').read_text());cells=[]
    for mid in ['hubert_base','wav2vec2_base']:
        c=next(r for r in baseline if r['encoder']==mid and r['train_normalize'] and r['test_normalize'] and r['gain_db']==0)
        cells.append(dict(encoder=mid,train_input_scale=1.,inference_input_scale=1.,**{k:c[k] for k in ['sync_c','sync_d','distance_d0','background_median','mouth_bbox_MSE','av_offsets','full_sync_c','full_distance_d0']},origin='exact_report54_standard_cell'))
    reference=[json.loads(x) for x in (parent/'training/hubert_base_on/trace.jsonl').read_text().splitlines()]
    ledger=lambda rows:[(r['step'],r['index'],r['source_id'],r['ray_sha256'],r['lr']) for r in rows]
    for arm in p['arms']:
        t=[json.loads(x) for x in (OUT/'training'/arm['id']/'trace.jsonl').read_text().splitlines()];assert ledger(t)==ledger(reference)
    track=np.load(OLD/'scores/shared_gt_track.npy');boxes=np.load(OLD/'scores/gt_boxes.npy')
    rebuilt_track=np.stack([medfilt(z,13) for z in [np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2,(boxes[:,0]+boxes[:,2])/2,(boxes[:,1]+boxes[:,3])/2]],1);assert np.array_equal(track,rebuilt_track)
    scores=json.loads((OUT/'remote/scores/results.json').read_text());assert len(scores['cells'])==10
    maxerr=0.;frames=0;distances=0
    for number,cell in enumerate(scores['cells']):
        arm=cell['arm'];scale=cell['input_scale'];label=f's{scale:.2f}';per=[]
        for clip in cell['clips']:
            folder=OUT/'remote/scores'/arm['id']/label/clip['clip_id'];v=np.load(folder/'visual.npy');a=np.load(OUT/'remote/scores'/(clip['clip_id']+'_audio.npy'));saved=np.load(folder/'matrix.npy');calc=matrix(v,a)
            assert saved.shape==(245,31) and sha(folder/'matrix.npy')==clip['matrix_sha256'];e=float(abs(saved-calc).max());assert e<1e-5;maxerr=max(e,maxerr);distances+=saved.size
            for scope,guard in [('guard20',True),('full',False)]:
                for k,z in metric(calc,guard).items():assert abs(z-clip[scope][k])<1e-5
            per.append(metric(calc));render=OUT/'remote/renders'/arm['id']/label/clip['clip_id'];r=json.loads((render/'receipt.json').read_text());assert r['arm']==arm and r['input_scale']==scale and r['weights']=='EMA12'
            tr=json.loads((OUT/'training'/arm['id']/'receipt.json').read_text());assert r['checkpoint_sha256']==tr['checkpoint_sha256'] and sha(render/'original_pcm.mkv')==r['mux_sha256']
            info=next(c for c in p['test_clips'] if c['id']==clip['clip_id']);audio,sr=sf.read(DATA/info['score_audio_path'],dtype='int16');assert sr==16000 and hashlib.sha256(audio.tobytes()).hexdigest()==r['pcm_sha256']==clip['source_pcm_sha256']
            decoded=subprocess.check_output([FFMPEG,'-v','error','-i',str(render/'original_pcm.mkv'),'-map','0:a:0','-f','s16le','pipe:1']);assert decoded==audio.tobytes()
            hashes=json.loads((folder/'crop_frame_hashes.json').read_text());cap=cv2.VideoCapture(str(render/'original_pcm.mkv'));selected=np.load(render/'selected_float.npz');n=0
            while True:
                ok,bgr=cap.read()
                if not ok:break
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);rr=r['frames'][n];source=info['source_start_frame']+n
                assert rr['source_id']==source and hashlib.sha256(rgb.tobytes()).hexdigest()==rr['rgb_sha256']
                gt,(x0,y0,x1,y1)=truth(source);delta=rgb.astype(float)/255-gt.astype(float);mouth=delta[y0:y1,x0:x1];eps=.5/255+1e-6
                for km,kl,z in [('MSE','L1',delta),('mouth_bbox_MSE','mouth_bbox_L1',mouth)]:
                    cm=float(np.mean(z*z));cl=float(np.mean(abs(z)));assert abs(cl-rr[kl])<=eps and abs(cm-rr[km])<=2*eps*cl+eps*eps+1e-7
                if str(n) in selected:
                    raw=selected[str(n)];assert hashlib.sha256(raw.tobytes()).hexdigest()==rr['float_sha256'];z=raw.astype(float)-gt.astype(float);m=z[y0:y1,x0:x1]
                    for k,value in dict(MSE=float(np.mean(z*z)),L1=float(np.mean(abs(z))),mouth_bbox_MSE=float(np.mean(m*m)),mouth_bbox_L1=float(np.mean(abs(m)))).items():assert abs(value-rr[k])<1e-7
                s,x,y=track[info['source_start_frame']-4925+n];pad=int(s*1.8);im=np.pad(bgr,((pad,pad),(pad,pad),(0,0)),constant_values=110);cx=x+pad;cy=y+pad
                crop=cv2.resize(im[int(cy-s):int(cy+s*1.8),int(cx-s*1.4):int(cx+s*1.4)],(224,224));assert hashlib.sha256(crop.tobytes()).hexdigest()==hashes[n];n+=1
            cap.release();selected.close();assert n==250;frames+=n
        for k in ['sync_c','sync_d','distance_d0','background_median']:assert abs(np.mean([c[k] for c in per])-cell['mean_guard20'][k])<1e-5
        cells.append(dict(encoder=arm['encoder'],train_input_scale=arm['input_scale'],inference_input_scale=scale,**cell['mean_guard20'],mouth_bbox_MSE=cell['mean_mouth_MSE'],av_offsets=','.join(str(c['guard20']['av_offset']) for c in cell['clips']),full_sync_c=float(np.mean([c['full']['sync_c'] for c in cell['clips']])),full_distance_d0=float(np.mean([c['full']['distance_d0'] for c in cell['clips']])),origin='new_postnorm_scale'))
        print('INDEPENDENTLY_REVIEWED',number+1,'/10',flush=True)
    for clip in p['test_clips']:
        assert np.array_equal(np.load(OUT/'remote/scores'/(clip['id']+'_audio.npy')),np.load(parent/'remote/scores'/(clip['id']+'_audio.npy')))
    validate_cells(cells);effects=[];keys=['sync_c','sync_d','distance_d0','background_median','mouth_bbox_MSE']
    get=lambda mid,t,q:next(c for c in cells if (c['encoder'],c['train_input_scale'],c['inference_input_scale'])==(mid,t,q))
    for mid in ['hubert_base','wav2vec2_base']:
        pairs=[('low_train_high_infer_minus_standard',get(mid,.5,2),get(mid,1,1))]
        pairs += [(f'low_train_minus_standard_given_infer_{q}',get(mid,.5,q),get(mid,1,q)) for q in [.5,1,2]]
        pairs += [(f'high_infer_minus_standard_given_train_{t}',get(mid,t,2),get(mid,t,1)) for t in [.5,1]]
        for label,a,b in pairs:effects.append(dict(encoder=mid,contrast=label,**{k:a[k]-b[k] for k in keys}))
        effects.append(dict(encoder=mid,contrast='training_by_high_inference_interaction',**{k:(get(mid,.5,2)[k]-get(mid,.5,1)[k])-(get(mid,1,2)[k]-get(mid,1,1)[k]) for k in keys}))
    out=OUT/'analysis';out.mkdir(exist_ok=False);export(out,'cells',cells);export(out,'effects',effects)
    write(out/'qa.json',dict(passed=True,total_cells=12,total_matrices=24,new_matrices=20,imported_matrices=4,new_distance_cells=distances,new_video_frames=frames,max_distance_error=maxerr,all_rgb_pcm_crop_float_readback=True,paired_actual_frame_ray_lr=True,scoring_audio_embeddings_exact_parent=True,single_seed=True,no_perceptual_claim=True))
if __name__=='__main__':main()
