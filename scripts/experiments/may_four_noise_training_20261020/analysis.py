"""Independent PCM/RGB/ROI/metric consumer and matched training interactions."""
from pathlib import Path
from functools import lru_cache
import json,hashlib,subprocess,sys,csv
import numpy as np,cv2,soundfile as sf
from scipy.signal import medfilt
from imageio_ffmpeg import get_ffmpeg_exe
from protocol import OUT,DATA,sha,write
from contrast import difference_in_differences
MEDIA=Path('E:/Documents/tts-exp/runs/may_four_noise_training_20261020_v1/renders')
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
@lru_cache(maxsize=16)
def truth(source):
 image=cv2.cvtColor(cv2.imread(str(DATA/f'source/May/gt_imgs/{source}.jpg')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
 lms=np.loadtxt(DATA/f'source/May/ori_imgs/{source}.lms');low=np.maximum(np.floor(lms[48:68].min(0)).astype(int)-12,0);high=np.minimum(np.ceil(lms[48:68].max(0)).astype(int)+13,512)
 return image,(*low,*high)
def metric(m,guard):
 c=(m[20:-20] if guard else m).mean(0);i=int(c.argmin());return dict(sync_c=float(np.median(c)-c[i]),sync_d=float(c[i]),distance_d0=float(c[15]),background_median=float(np.median(c)),av_offset=15-i)
def matrix(v,a):
 a=np.pad(a.astype(float),((15,15),(0,0)))
 return np.stack([np.sqrt(((x.astype(float)[None]-a[i:i+31]+1e-6)**2).sum(1)) for i,x in enumerate(v)])
def export(name,rows):
 write(OUT/'analysis'/(name+'.json'),rows)
 with (OUT/'analysis'/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
 p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'training_independent_review.json').read_text())['passed']
 result=json.loads((OUT/'scores/results.json').read_text());assert len(result['cells'])==36
 track=np.load(OUT/'historical/shared_gt_track.npy');boxes=np.load(OUT/'historical/gt_boxes.npy')
 rebuilt=np.stack([medfilt(z,13) for z in [np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2,(boxes[:,0]+boxes[:,2])/2,(boxes[:,1]+boxes[:,3])/2]],1);assert np.array_equal(rebuilt,track)
 cells=[];clip_rows=[];distances=0;frames=0;max_error=0
 for cell in result['cells']:
  arm=cell['arm'];scale=cell['input_scale'];label=f's{scale:.2f}';per=[]
  for clip in cell['clips']:
   folder=OUT/'scores'/arm['id']/label/clip['clip_id'];v=np.load(folder/'visual.npy');a=np.load(OUT/'scores'/(clip['clip_id']+'_audio.npy'));saved=np.load(folder/'matrix.npy');calc=matrix(v,a)
   assert saved.shape==(245,31) and sha(folder/'matrix.npy')==clip['matrix_sha256'];error=float(abs(calc-saved).max());assert error<1e-5;max_error=max(max_error,error);distances+=saved.size
   for scope,guard in [('guard20',True),('full',False)]:
    for k,value in metric(calc,guard).items():assert abs(value-clip[scope][k])<1e-5
   rendered=MEDIA/arm['id']/label/clip['clip_id'];rec=json.loads((rendered/'receipt.json').read_text());assert rec['arm']==arm and rec['input_scale']==scale and rec['weights']=='EMA12'
   cp=json.loads((OUT/'training'/arm['id']/'receipt.json').read_text());assert rec['checkpoint_sha256']==cp['checkpoint_sha256'] and sha(rendered/'original_pcm.mkv')==rec['mux_sha256']
   info=next(c for c in p['test_clips'] if c['id']==clip['clip_id']);audio,sr=sf.read(DATA/info['score_audio_path'],dtype='int16');assert sr==16000 and digest(audio)==rec['pcm_sha256']==clip['source_pcm_sha256']
   decoded=subprocess.check_output([get_ffmpeg_exe(),'-v','error','-i',str(rendered/'original_pcm.mkv'),'-map','0:a:0','-f','s16le','pipe:1']);assert decoded==audio.tobytes()
   crops=json.loads((folder/'crop_frame_hashes.json').read_text());selected=np.load(rendered/'selected_float.npz');cap=cv2.VideoCapture(str(rendered/'original_pcm.mkv'));n=0
   while True:
    ok,bgr=cap.read()
    if not ok:break
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);row=rec['frames'][n];source=info['source_start_frame']+n;assert row['source_id']==source and digest(rgb)==row['rgb_sha256']
    gt,(x0,y0,x1,y1)=truth(source);delta=rgb.astype(float)/255-gt.astype(float);mouth=delta[y0:y1,x0:x1];eps=.5/255+1e-6
    for km,kl,z in [('MSE','L1',delta),('mouth_bbox_MSE','mouth_bbox_L1',mouth)]:
     cm=float(np.mean(z*z));cl=float(np.mean(abs(z)));assert abs(cl-row[kl])<=eps and abs(cm-row[km])<=2*eps*cl+eps*eps+1e-7
    if str(n) in selected:
     raw=selected[str(n)];assert digest(raw)==row['float_sha256'];z=raw.astype(float)-gt.astype(float);mouth=z[y0:y1,x0:x1]
     for k,value in dict(MSE=float(np.mean(z*z)),L1=float(np.mean(abs(z))),mouth_bbox_MSE=float(np.mean(mouth*mouth)),mouth_bbox_L1=float(np.mean(abs(mouth)))).items():assert abs(value-row[k])<1e-7
    size,x,y=track[info['source_start_frame']-4925+n];pad=int(size*1.8);image=np.pad(bgr,((pad,pad),(pad,pad),(0,0)),constant_values=110);cx=x+pad;cy=y+pad
    cropped=cv2.resize(image[int(cy-size):int(cy+size*1.8),int(cx-size*1.4):int(cx+size*1.4)],(224,224));assert digest(cropped)==crops[n];n+=1
   cap.release();selected.close();assert n==250;frames+=n
   measured=metric(calc,True);per.append(measured);clip_rows.append(dict(arm_id=arm['id'],kind=arm['kind'],snr_db=arm['snr_db'],train_scale=arm['input_scale'],inference_scale=scale,clip_id=clip['clip_id'],**measured,mouth_bbox_MSE=rec['mean']['mouth_bbox_MSE']))
  for k in ['sync_c','sync_d','distance_d0','background_median']:assert abs(np.mean([r[k] for r in per])-cell['mean_guard20'][k])<1e-5
  cells.append(dict(arm_id=arm['id'],kind=arm['kind'],snr_db=arm['snr_db'],train_scale=arm['input_scale'],inference_scale=scale,**cell['mean_guard20'],mouth_bbox_MSE=cell['mean_mouth_MSE'],av_offsets=','.join(str(r['av_offset']) for r in per),full_sync_c=float(np.mean([c['full']['sync_c'] for c in cell['clips']]))))
  print('independent media QA',len(cells),'/36',flush=True)
 assert distances==72*245*31 and frames==18000
 kinds=['single_speech','babble4','modulated_ssn','clicks'];keys=['sync_c','sync_d','distance_d0','background_median','mouth_bbox_MSE'];effects=[]
 get=lambda kind,q,t,i:next(c for c in cells if (c['kind'],c['snr_db'],c['train_scale'],c['inference_scale'])==(kind,q,t,i))
 for infer in [1.,.5]:
  clean_s=get('none',None,1.,infer);clean_l=get('none',None,.5,infer)
  for kind in kinds:
   for q in [30,90]:
    noisy_s=get(kind,q,1.,infer);noisy_l=get(kind,q,.5,infer)
    for contrast,a,b in [('noise_harm_at_standard_train',noisy_s,clean_s),('noise_harm_at_low_train',noisy_l,clean_l),('low_train_effect_with_noise',noisy_l,noisy_s),('low_train_effect_without_added_noise',clean_l,clean_s)]:effects.append(dict(kind=kind,snr_db=q,inference_scale=infer,contrast=contrast,**{k:a[k]-b[k] for k in keys}))
    effects.append(dict(kind=kind,snr_db=q,inference_scale=infer,contrast='noise_by_training_scale_interaction',**{k:difference_in_differences(noisy_l[k],noisy_s[k],clean_l[k],clean_s[k]) for k in keys}))
 (OUT/'analysis').mkdir(exist_ok=False);export('cells',cells);export('clips',clip_rows);export('effects',effects)
 write(OUT/'analysis/qa.json',dict(passed=True,cells=36,matrices=72,frames=frames,distance_cells=distances,max_distance_error=max_error,all_rgb_pcm_crop_float_readback=True,training_review_sha256=sha(OUT/'training_independent_review.json'),single_train_seed=True,single_noise_instance=True,no_perceptual_claim=True))
if __name__=='__main__':main()
