"""Independent retained FFV1 decoding and streamed creation-record audit."""
from pathlib import Path
import json,hashlib,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cv2.setNumThreads(1);rows=[]
for p in sorted((OUT/'features').rglob('FIXED.json')):
 m=read(p)
 if not m['retained_video']:continue
 v=m['video'];assert sha(v['path'])==v['sha256'];cap=cv2.VideoCapture(v['path']);d=hashlib.sha256();n=0
 while True:
  ok,x=cap.read()
  if not ok:break
  d.update(x.tobytes());n+=1
 cap.release();assert n==v['frames'] and d.hexdigest()==v['pixel_sha256'];rows.append({'id':m['id'],'arm':m['arm'],'frames':n,'decoded_pixel_sha256':d.hexdigest(),'metadata_sha256':sha(p)})
assert len(rows)==52
for a in ['N','T']:
 c=read(OUT/'controls'/a/'streaming.json');f=np.load(OUT/'controls'/a/'streamed_control.npz');first=min(r['id'] for r in read(OUT/'support.json') if r['split']=='calibration');g=np.load(OUT/'features'/first/a/'FIXED.npz')
 assert c['passed'] and sha(OUT/'controls'/a/'streamed_control.npz')==c['features_sha256']
 for k in ['visual','audio','z']:assert np.array_equal(f[k],g[k])
assert not list((OUT/'temporary'/'stream_control').rglob('*.avi'))
result={'status':'PASS','retained_cal_videos_decoded':len(rows),'all_decoded_pixel_hashes_match':True,'firstcal_streamed_saved_arrays_exact':True,'rows':rows,'checker_sha256':sha(__file__)}
(OUT/'independent_retained_cal_pixels.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='rows'})
