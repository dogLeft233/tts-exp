"""Independent final science consumer. Rebuild distances from saved embeddings."""
from pathlib import Path
import json,hashlib,csv,subprocess
import cv2
from imageio_ffmpeg import get_ffmpeg_exe
FFMPEG=get_ffmpeg_exe()
import numpy as np
from protocol import OUT,OLD,DATA,sha,write,validate_pair
def matrix(v,a):
 a=np.pad(a.astype(float),((15,15),(0,0)))
 return np.stack([np.sqrt(((x.astype(float)[None]-a[i:i+31]+1e-6)**2).sum(1)) for i,x in enumerate(v)])
def metric(m,guard):
 c=(m[20:-20] if guard else m).mean(0);i=int(c.argmin())
 return dict(sync_c=float(np.median(c)-c[i]),sync_d=float(c[i]),distance_d0=float(c[15]),background_median=float(np.median(c)),av_offset=15-i)
def main():
 r=OUT/'remote';p=json.loads((OUT/'protocol.json').read_text());scores=json.loads((r/'scores/results.json').read_text())
 assert len(scores['cells'])==24 and all(c['gain_db'] in [-6,0,6] for c in scores['cells'])
 manifest=[];traces={};maxerr=0.;distance_count=0;result=[];video_frames=0
 track=np.load(OLD/'scores/shared_gt_track.npy');boxes=np.load(OLD/'scores/gt_boxes.npy')
 from scipy.signal import medfilt
 assert np.array_equal(track,np.stack([medfilt(z,13) for z in [np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2,(boxes[:,0]+boxes[:,2])/2,(boxes[:,1]+boxes[:,3])/2]],1))
 for arm in p['arms']:
  execution=json.loads((OUT/'jobs'/('train_'+arm['id'])/'execution.json').read_text());assert execution['status']=='completed' and execution['rc']==0
  for name in ['train.py','runtime.py']:assert execution['source_sha256'][name]==sha(OUT/'execution_sources'/name)
  assert execution['source_sha256']['protocol.json']==sha(OUT/'protocol.json')
  root=OUT/'training'/arm['id'];receipt=json.loads((root/'receipt.json').read_text());assert receipt['passed'] and receipt['success']==receipt['attempts']==12000 and receipt['skip']==0
  assert sha(root/receipt['checkpoint'])==receipt['checkpoint_sha256'] and receipt['initial_sha256']==sha(OUT/'initialization.pt')
  trace=[json.loads(x) for x in (root/'trace.jsonl').read_text().splitlines()];assert len(trace)==12000 and all(x['step']==i+1 and np.isfinite(x['loss']) for i,x in enumerate(trace))
  traces[arm['id']]=trace
  schedule=hashlib.sha256(json.dumps([(x['step'],x['source_id'],x['ray_sha256'],x['lr']) for x in trace],separators=(',',':')).encode()).hexdigest()
  spec=next(m for m in p['models'] if m['id']==arm['encoder']);files=spec['source']['files']
  manifest.append(dict(encoder=arm['encoder'],normalize=arm['norm'],weight_sha256=next(x['sha256'] for x in files if x['file']=='pytorch_model.bin'),config_sha256=next(x['sha256'] for x in files if x['file']=='config.json'),pcm_sha256=next(c['input_pcm_sha256'] for c in json.loads((OUT/'features'/arm['encoder']/'receipt.json').read_text())['conditions'] if c['split']=='train' and c['normalize']==arm['norm']),generator_initial_sha256=sha(OUT/'initialization.pt'),steps=12000,frame_order_sha256=sha(OUT/'frame_order.npy'),ray_schedule_sha256=schedule,source_ids=[4929,5679],precision='FP32',lr_policy=p['learning_rate'],feature_layout=p['feature_layout']))
 for spec in p['models']:
  aa=[m for m in manifest if m['encoder']==spec['id']];assert validate_pair(*aa)
 assert len({x['ray_schedule_sha256'] for x in manifest})==1
 for cell in scores['cells']:
  arm=cell['arm'];norm=cell['test_normalize'];gain=cell['gain_db'];base=r/'scores'/arm['id']/f'{"on" if norm else "off"}_g{gain:+d}';rows=[]
  for clip in cell['clips']:
   folder=base/clip['clip_id'];v=np.load(folder/'visual.npy',allow_pickle=False);a=np.load(r/'scores'/(clip['clip_id']+'_audio.npy'),allow_pickle=False);saved=np.load(folder/'matrix.npy',allow_pickle=False)
   rebuilt=matrix(v,a);assert saved.shape==(245,31) and sha(folder/'matrix.npy')==clip['matrix_sha256']
   e=float(np.max(abs(rebuilt-saved)));maxerr=max(maxerr,e);assert e<1e-5
   for scope,guard in [('guard20',True),('full',False)]:
    comp=metric(rebuilt,guard)
    for k,val in comp.items():assert abs(val-clip[scope][k])<1e-5
   rows.append(metric(rebuilt,True));distance_count+=saved.size
   render=r/'renders'/arm['id']/f'{"on" if norm else "off"}_g{gain:+d}'/clip['clip_id'];receipt=json.loads((render/'receipt.json').read_text())
   assert sha(render/'original_pcm.mkv')==receipt['mux_sha256']
   info=next(c for c in p['test_clips'] if c['id']==clip['clip_id'])
   import soundfile as sf
   original,sr=sf.read(DATA/info['score_audio_path'],dtype='int16');assert sr==16000
   assert hashlib.sha256(original.tobytes()).hexdigest()==receipt['pcm_sha256']==clip['source_pcm_sha256']
   decoded=subprocess.check_output([FFMPEG,'-v','error','-i',str(render/'original_pcm.mkv'),'-map','0:a:0','-f','s16le','pipe:1']);assert decoded==original.tobytes()
   crop_hashes=json.loads((folder/'crop_frame_hashes.json').read_text());cap=cv2.VideoCapture(str(render/'original_pcm.mkv'));n=0
   samples=np.load(render/'selected_float.npz',allow_pickle=False)
   while True:
    ok,image=cap.read()
    if not ok:break
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);row=receipt['frames'][n];assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['rgb_sha256']
    source_id=info['source_start_frame']+n;assert row['source_id']==source_id
    truth=cv2.cvtColor(cv2.imread(str(DATA/f'source/May/gt_imgs/{source_id}.jpg')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    lms=np.loadtxt(DATA/f'source/May/ori_imgs/{source_id}.lms');xmin,ymin=np.maximum(np.floor(lms[48:68].min(0)).astype(int)-12,0);xmax,ymax=np.minimum(np.ceil(lms[48:68].max(0)).astype(int)+13,512)
    delta=rgb.astype(float)/255-truth.astype(float);mouth=delta[ymin:ymax,xmin:xmax]
    for mse,l1,z in [('MSE','L1',delta),('mouth_bbox_MSE','mouth_bbox_L1',mouth)]:
     cm=float(np.mean(z**2));cl=float(np.mean(abs(z)));epsilon=.5/255+1e-6
     assert abs(cl-row[l1])<=epsilon and abs(cm-row[mse])<=2*epsilon*cl+epsilon**2+1e-7
    if str(n) in samples:
     raw=samples[str(n)];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['float_sha256']
     e=raw.astype(float)-truth.astype(float);m=e[ymin:ymax,xmin:xmax]
     for key,value in dict(MSE=float(np.mean(e**2)),L1=float(np.mean(abs(e))),mouth_bbox_MSE=float(np.mean(m**2)),mouth_bbox_L1=float(np.mean(abs(m)))).items():assert abs(value-row[key])<1e-7
    s,x,y=track[info['source_start_frame']-4925+n];pad=int(s*1.8);im=np.pad(image,((pad,pad),(pad,pad),(0,0)),constant_values=110);cx=x+pad;cy=y+pad
    cropped=cv2.resize(im[int(cy-s):int(cy+s*1.8),int(cx-s*1.4):int(cx+s*1.4)],(224,224));assert hashlib.sha256(cropped.tobytes()).hexdigest()==crop_hashes[n]
    n+=1
   cap.release();samples.close();assert n==250;video_frames+=n
  for k in ['sync_c','sync_d','distance_d0','background_median']:assert abs(np.mean([x[k] for x in rows])-cell['mean_guard20'][k])<1e-5
  result.append(dict(encoder=arm['encoder'],train_normalize=arm['norm'],test_normalize=norm,gain_db=gain,**cell['mean_guard20'],mouth_bbox_MSE=cell['mean_mouth_MSE'],av_offsets=','.join(str(c['guard20']['av_offset']) for c in cell['clips']),full_sync_c=float(np.mean([c['full']['sync_c'] for c in cell['clips']])),full_distance_d0=float(np.mean([c['full']['distance_d0'] for c in cell['clips']]))))
 contrast=[]
 def get(mid,tn,qn,g):return next(x for x in result if x['encoder']==mid and x['train_normalize']==tn and x['test_normalize']==qn and x['gain_db']==g)
 for spec in p['models']:
  mid=spec['id']
  for g in [-6,0,6]:
   for label,a,b in [('train_on_minus_off_given_test_on',get(mid,True,True,g),get(mid,False,True,g)),('train_on_minus_off_given_test_off',get(mid,True,False,g),get(mid,False,False,g)),('test_on_minus_off_given_train_on',get(mid,True,True,g),get(mid,True,False,g)),('test_on_minus_off_given_train_off',get(mid,False,True,g),get(mid,False,False,g)),('matched_on_on_minus_off_off_joint',get(mid,True,True,g),get(mid,False,False,g))]:
    contrast.append(dict(encoder=mid,gain_db=g,contrast=label,**{k:a[k]-b[k] for k in ['sync_c','sync_d','distance_d0','background_median','mouth_bbox_MSE']}))
 # Same two fixed clips from the old 60k pilot; never compare unequal clip means.
 old=json.loads((OLD/'scores/scores.json').read_text());historical=[]
 for cell in old['cells']:
  clips=[c for c in cell['clips'] if c['clip_id'] in ['test_00','test_03']];assert len(clips)==2
  historical.append(dict(encoder='AVE',updates=60000,training_gain_db=cell['training_gain_db'],test_gain_db=cell['test_gain_db'],**{k:float(np.mean([c['guard20'][k] for c in clips])) for k in ['sync_c','sync_d','distance_d0','background_median']},causal_norm_contrast=False))
 out=OUT/'analysis';out.mkdir(exist_ok=False)
 for name,rows in [('cells',result),('contrasts',contrast),('historical_same_clips',historical)]:
  write(out/(name+'.json'),rows)
  with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write(out/'qa.json',dict(passed=True,training_arms=4,actual_updates_each=12000,matrices=48,distance_cells=distance_count,video_frames=video_frames,all_actual_crops_rgb_pcm_readback=True,max_distance_error=maxerr,paired_control_manifests=manifest,same_encoder_module_contrast=True,training_randomness_replicated=False,perception_not_measured=True))
 print(json.dumps(dict(passed=True,cells=24,contrasts=len(contrast),max_distance_error=maxerr)))
if __name__=='__main__':main()
