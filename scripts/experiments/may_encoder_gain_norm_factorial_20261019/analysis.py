"""Independent CPU consumer: exact imported zero-gain cells plus all new crossed gain cells."""
from pathlib import Path
from functools import lru_cache
import json,hashlib,csv,subprocess,sys
import numpy as np,cv2,soundfile as sf
from scipy.signal import medfilt
from imageio_ffmpeg import get_ffmpeg_exe
from protocol import OUT,DATA,OLD,sha,write,validate_pair
FFMPEG=get_ffmpeg_exe()
def matrix(v,a):
 a=np.pad(a.astype(float),((15,15),(0,0)))
 return np.stack([np.sqrt(((x.astype(float)[None]-a[i:i+31]+1e-6)**2).sum(1)) for i,x in enumerate(v)])
def metric(m,guard=True):
 c=(m[20:-20] if guard else m).mean(0);i=int(c.argmin())
 return dict(sync_c=float(np.median(c)-c[i]),sync_d=float(c[i]),distance_d0=float(c[15]),background_median=float(np.median(c)),av_offset=15-i)
@lru_cache(maxsize=500)
def truth(source):
 im=cv2.cvtColor(cv2.imread(str(DATA/f'source/May/gt_imgs/{source}.jpg')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
 lms=np.loadtxt(DATA/f'source/May/ori_imgs/{source}.lms');lo=np.maximum(np.floor(lms[48:68].min(0)).astype(int)-12,0);hi=np.minimum(np.ceil(lms[48:68].max(0)).astype(int)+13,512)
 return im,(*lo,*hi)
def export(root,name,rows):
 write(root/(name+'.json'),rows)
 with (root/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 admitted=json.loads((OUT/'evaluation_admission.json').read_text());assert sha(Path(__file__))==admitted['source_sha256']['analysis.py'],'frozen analyzer source changed'
 p=json.loads((OUT/'protocol.json').read_text());parent=Path(p['parent']);oldqa=json.loads((parent/'analysis/qa.json').read_text());assert oldqa['passed']
 parentseal=json.loads((parent/'final_delivery_receipt.json').read_text());assert parentseal['passed'] and sha(parent/'final_delivery_receipt.json')==p['parent_delivery_receipt_sha256']
 assert json.loads((OUT/'training_independent_review.json').read_text())['passed']
 imported=json.loads((parent/'analysis/cells.json').read_text());assert len(imported)==24
 result=[dict(**c,train_gain_db=0,origin='immutable_report54') for c in imported]
 ref=[json.loads(x) for x in (parent/'training/hubert_base_on/trace.jsonl').read_text().splitlines()]
 ledger=lambda t:[(r['step'],r['index'],r['source_id'],r['ray_sha256'],r['lr']) for r in t]
 controls=[]
 for arm in p['arms']:
  key=arm['id'];job_locations=json.loads((OUT/'completed_job_locations.json').read_text()) if (OUT/'completed_job_locations.json').exists() else {};jobroot=Path(job_locations.get('train_'+key,str(OUT/'jobs'/('train_'+key))));ex=json.loads((jobroot/'execution.json').read_text());assert ex['status']=='completed' and ex['rc']==0
  for n in ['train.py','runtime.py']:assert ex['source_sha256'][n]==sha(OUT/'execution_sources'/n)
  root=OUT/'training'/key;rec=json.loads((root/'receipt.json').read_text());assert rec['success']==rec['attempts']==12000 and rec['skip']==0 and rec['initial_sha256']==p['parent_initial_sha256']
  assert sha(root/rec['checkpoint'])==rec['checkpoint_sha256']
  trace=[json.loads(x) for x in (root/'trace.jsonl').read_text().splitlines()];assert ledger(trace)==ledger(ref)
  spec=next(s for s in p['models'] if s['id']==arm['encoder']);fr=json.loads((OUT/'features'/arm['encoder']/'receipt.json').read_text());condition=next(c for c in fr['conditions'] if c['split']=='train' and c['normalize']==arm['norm'] and c['gain_db']==arm['train_gain_db'])
  files=spec['source']['files'];controls.append(dict(encoder=arm['encoder'],normalize=arm['norm'],train_gain_db=arm['train_gain_db'],weight_sha256=next(x['sha256'] for x in files if x['file']=='pytorch_model.bin'),config_sha256=next(x['sha256'] for x in files if x['file']=='config.json'),pcm_sha256=condition['input_pcm_sha256'],generator_initial_sha256=p['parent_initial_sha256'],steps=12000,frame_order_sha256=p['parent_frame_order_sha256'],ray_schedule_sha256=hashlib.sha256(json.dumps(ledger(trace),separators=(',',':')).encode()).hexdigest(),source_ids=[4929,5679],precision='FP32',lr_policy=p['learning_rate'],feature_layout=p['feature_layout']))
 for mid in ['hubert_base','wav2vec2_base']:
  for gain in [-6,6]:assert validate_pair(*[c for c in controls if c['encoder']==mid and c['train_gain_db']==gain])
 assert len({c['ray_schedule_sha256'] for c in controls})==1
 track=np.load(OLD/'scores/shared_gt_track.npy');boxes=np.load(OLD/'scores/gt_boxes.npy');assert np.array_equal(track,np.stack([medfilt(z,13) for z in [np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2,(boxes[:,0]+boxes[:,2])/2,(boxes[:,1]+boxes[:,3])/2]],1))
 scores=json.loads((OUT/'remote/scores/results.json').read_text());assert len(scores['cells'])==48
 maxerr=0.;frames=0;distances=0
 for number,cell in enumerate(scores['cells']):
  arm=cell['arm'];norm=cell['test_normalize'];gain=cell['gain_db'];label=f'{"on" if norm else "off"}_g{gain:+d}';rows=[]
  for clip in cell['clips']:
   folder=OUT/'remote/scores'/arm['id']/label/clip['clip_id'];v=np.load(folder/'visual.npy');a=np.load(OUT/'remote/scores'/(clip['clip_id']+'_audio.npy'));saved=np.load(folder/'matrix.npy');rebuilt=matrix(v,a)
   assert saved.shape==(245,31) and sha(folder/'matrix.npy')==clip['matrix_sha256'];e=float(abs(saved-rebuilt).max());assert e<1e-5;maxerr=max(maxerr,e);distances+=saved.size
   for scope,g in [('guard20',True),('full',False)]:
    calc=metric(rebuilt,g)
    for k,val in calc.items():assert abs(val-clip[scope][k])<1e-5
   rows.append(metric(rebuilt));render=Path(p['new_media_placements'][arm['id']])/label/clip['clip_id'];r=json.loads((render/'receipt.json').read_text());assert sha(render/'original_pcm.mkv')==r['mux_sha256']
   info=next(c for c in p['test_clips'] if c['id']==clip['clip_id']);audio,sr=sf.read(DATA/info['score_audio_path'],dtype='int16');assert sr==16000
   assert hashlib.sha256(audio.tobytes()).hexdigest()==r['pcm_sha256']==clip['source_pcm_sha256']
   decoded=subprocess.check_output([FFMPEG,'-v','error','-i',str(render/'original_pcm.mkv'),'-map','0:a:0','-f','s16le','pipe:1']);assert decoded==audio.tobytes()
   hashes=json.loads((folder/'crop_frame_hashes.json').read_text());cap=cv2.VideoCapture(str(render/'original_pcm.mkv'));selected=np.load(render/'selected_float.npz');n=0
   while True:
    ok,bgr=cap.read()
    if not ok:break
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);rr=r['frames'][n];source=info['source_start_frame']+n;assert rr['source_id']==source and hashlib.sha256(rgb.tobytes()).hexdigest()==rr['rgb_sha256']
    gt,(x0,y0,x1,y1)=truth(source);delta=rgb.astype(float)/255-gt.astype(float);mouth=delta[y0:y1,x0:x1];eps=.5/255+1e-6
    for km,kl,z in [('MSE','L1',delta),('mouth_bbox_MSE','mouth_bbox_L1',mouth)]:
     cm=float(np.mean(z*z));cl=float(np.mean(abs(z)));assert abs(cl-rr[kl])<=eps and abs(cm-rr[km])<=2*eps*cl+eps*eps+1e-7
    if str(n) in selected:
     raw=selected[str(n)];assert hashlib.sha256(raw.tobytes()).hexdigest()==rr['float_sha256'];z=raw.astype(float)-gt.astype(float);m=z[y0:y1,x0:x1]
     for k,value in dict(MSE=float(np.mean(z*z)),L1=float(np.mean(abs(z))),mouth_bbox_MSE=float(np.mean(m*m)),mouth_bbox_L1=float(np.mean(abs(m)))).items():assert abs(value-rr[k])<1e-7
    s,x,y=track[info['source_start_frame']-4925+n];pad=int(s*1.8);im=np.pad(bgr,((pad,pad),(pad,pad),(0,0)),constant_values=110);cx=x+pad;cy=y+pad
    crop=cv2.resize(im[int(cy-s):int(cy+s*1.8),int(cx-s*1.4):int(cx+s*1.4)],(224,224));assert hashlib.sha256(crop.tobytes()).hexdigest()==hashes[n];n+=1
   cap.release();selected.close();assert n==250;frames+=n
  for k in ['sync_c','sync_d','distance_d0','background_median']:assert abs(np.mean([r[k] for r in rows])-cell['mean_guard20'][k])<1e-5
  result.append(dict(encoder=arm['encoder'],train_normalize=arm['norm'],test_normalize=norm,gain_db=gain,**cell['mean_guard20'],mouth_bbox_MSE=cell['mean_mouth_MSE'],av_offsets=','.join(str(c['guard20']['av_offset']) for c in cell['clips']),full_sync_c=float(np.mean([c['full']['sync_c'] for c in cell['clips']])),full_distance_d0=float(np.mean([c['full']['distance_d0'] for c in cell['clips']])),train_gain_db=arm['train_gain_db'],origin='new_gain_training'))
  print('INDEPENDENTLY_REVIEWED',number+1,'/48',arm['id'],norm,gain,flush=True)
 for info in p['test_clips']:
  new_audio=np.load(OUT/'remote/scores'/(info['id']+'_audio.npy'));old_audio=np.load(parent/'remote/scores'/(info['id']+'_audio.npy'));assert np.array_equal(new_audio,old_audio),'scoring audio embedding must match exact report54 baseline'
 assert len(result)==72 and len({(c['encoder'],c['train_normalize'],c['test_normalize'],c['train_gain_db'],c['gain_db']) for c in result})==72
 get=lambda mid,tn,qn,tg,qg:next(c for c in result if (c['encoder'],c['train_normalize'],c['test_normalize'],c['train_gain_db'],c['gain_db'])==(mid,tn,qn,tg,qg))
 keys=['sync_c','sync_d','distance_d0','background_median','mouth_bbox_MSE'];effects=[];norm_effects=[]
 for mid in ['hubert_base','wav2vec2_base']:
  for tn in [True,False]:
   for qn in [True,False]:
    for qg in [-6,0,6]:
     for tg in [-6,6]:
      a=get(mid,tn,qn,tg,qg);b=get(mid,tn,qn,0,qg);effects.append(dict(encoder=mid,train_normalize=tn,test_normalize=qn,test_gain_db=qg,training_gain_db=tg,contrast='training_gain_minus_zero',**{k:a[k]-b[k] for k in keys}))
  for tg in [-6,0,6]:
   for qg in [-6,0,6]:
    for label,a,b in [('matched_norm_on_minus_off',get(mid,True,True,tg,qg),get(mid,False,False,tg,qg)),('train_norm_on_minus_off_given_infer_on',get(mid,True,True,tg,qg),get(mid,False,True,tg,qg)),('train_norm_on_minus_off_given_infer_off',get(mid,True,False,tg,qg),get(mid,False,False,tg,qg)),('infer_norm_on_minus_off_given_train_on',get(mid,True,True,tg,qg),get(mid,True,False,tg,qg)),('infer_norm_on_minus_off_given_train_off',get(mid,False,True,tg,qg),get(mid,False,False,tg,qg))]:norm_effects.append(dict(encoder=mid,training_gain_db=tg,test_gain_db=qg,contrast=label,**{k:a[k]-b[k] for k in keys}))
 out=OUT/'analysis';out.mkdir(exist_ok=False)
 for name,rows in [('cells',result),('training_gain_effects',effects),('normalization_effects',norm_effects)]:export(out,name,rows)
 write(out/'qa.json',dict(passed=True,analyzer_sha256=sha(Path(__file__)),new_training_arms=8,imported_training_arms=4,new_updates=96000,total_updates_including_imports=144000,new_matrices=96,imported_matrices=48,total_cells=72,total_matrices=144,new_distance_cells=distances,new_video_frames=frames,imported_verified_video_frames=oldqa['video_frames'],all_new_rgb_pcm_crop_float_readback=True,max_new_distance_error=maxerr,model_control_manifests=controls,training_randomness_replicated=False,no_perceptual_claim=True,no_zero_gain_replay=True))
 print(json.dumps(dict(passed=True,cells=72,matrices=144,new_video_frames=frames,maxerr=maxerr)))
if __name__=='__main__':main()
