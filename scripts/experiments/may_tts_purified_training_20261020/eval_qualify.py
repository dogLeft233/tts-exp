"""Fixed final EMA replay checks, without optimizer construction."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,subprocess
import numpy as np
import cv2,soundfile as sf,torch
from torch_ema import ExponentialMovingAverage
from common import ROOT,DATA,configure,dataset,load_metadata,save_json,sha
from render_test import Video
def main():
 configure();torch.set_grad_enabled(False)
 from nerf_triplane.network import NeRFNetwork
 from nerf_triplane.utils import Trainer
 contract=json.loads((ROOT/'evaluation_contract.json').read_text())
 for path,digest in contract['input_sha256'].items():assert sha(ROOT/path)==digest,path
 q=ROOT/'qualification';assert not q.exists();q.mkdir()
 p=json.loads((ROOT/'protocol.json').read_text());d=dataset('test',0,p)
 indices=contract['qualification_indices'];rows=[]
 def load(cp):
  state=torch.load(cp,map_location='cpu',weights_only=False);assert state['step']==60000
  model=NeRFNetwork(d.opt).cuda();model.load_state_dict(state['model'],strict=True);load_metadata(model,state['metadata'])
  ema=ExponentialMovingAverage(model.parameters(),decay=.95);ema.load_state_dict(state['ema']);assert int(state['ema']['num_updates'])==60;ema.copy_to();model.eval()
  return model,SimpleNamespace(opt=d.opt,model=model,criterion=torch.nn.L1Loss(reduction='none'))
 model,owner=load(ROOT/'reference/baseline_final.pt')
 old=json.loads((ROOT/'reference/baseline_render_test0.json').read_text())
 for idx in indices:
  x=Trainer.eval_step(owner,d.collate([idx]))[0][0].cpu().numpy().astype(np.float32)
  rgb=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
  assert hashlib.sha256(x.tobytes()).hexdigest()==old['frames'][idx]['float_sha256'],('baseline float',idx)
  assert hashlib.sha256(rgb.tobytes()).hexdigest()==old['frames'][idx]['rgb_sha256'],('baseline rgb',idx)
 del owner,model;torch.cuda.empty_cache()
 cp=Path(json.loads((ROOT/'training/gain_0/training_receipt.json').read_text())['checkpoint'])
 assert sha(cp)==contract['checkpoint_sha256'];model,owner=load(cp)
 pcm,sr=sf.read(ROOT/'original_source_audio.wav',dtype='int16',always_2d=True);pcm=pcm[4925*640:(4925+len(indices))*640,0]
 sf.write(q/'qualification_audio.wav',pcm,16000,subtype='PCM_16');writer=Video(q/'repeat.mkv',q/'qualification_audio.wav',len(indices))
 for idx in indices:
  x=Trainer.eval_step(owner,d.collate([idx]))[0][0].cpu().numpy().astype(np.float32)
  y=Trainer.eval_step(owner,d.collate([idx]))[0][0].cpu().numpy().astype(np.float32);assert np.array_equal(x,y) and np.isfinite(x).all()
  rgb=np.rint(np.clip(x,0,1)*255).astype(np.uint8);writer.write(rgb)
  rows.append(dict(index=idx,float_sha256=hashlib.sha256(x.tobytes()).hexdigest(),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),repeat_exact=True))
 writer.close();cap=cv2.VideoCapture(str(q/'repeat.mkv'));count=0
 while True:
  ok,bgr=cap.read()
  if not ok:break
  assert hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==rows[count]['rgb_sha256'];count+=1
 cap.release();assert count==len(indices)
 decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(q/'repeat.mkv'),'-map','0:a:0','-f','s16le','pipe:1']);assert decoded==pcm.tobytes()
 save_json(q/'receipt.json',dict(passed=True,baseline_fixed_frames_float_RGB_exact=True,new_model_repeat_exact=True,codec_pixels_PCM_exact=True,frames=rows,checkpoint_sha256=sha(cp),optimizer_updates=0))
 print(json.dumps(dict(passed=True,baseline_frames=len(indices),new_repeat_frames=len(indices))),flush=True)
if __name__=='__main__':main()
