"""Follow-up complete native-frame strips, after initial character-sheet review."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/wjj/tts-audio/tts-exp')
SRC=ROOT/'runs/qwen_phrase_local_20261024_attempt02'
OUT=ROOT/'runs/qwen_phrase_visual_review_20261024_attempt02'
rows=json.loads((OUT/'measurements.json').read_text())
assets={x['id']:x for x in json.loads((SRC/'input_manifest.json').read_text())['audio']}
fa={x['id']:x for x in json.loads((SRC/'alignment.json').read_text())['audio']}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
out=OUT/'fullstrips';out.mkdir(exist_ok=False)
cache={};sampling=[]
for x in rows:
 key=x['id'];asset=assets[key]
 if key not in cache:
  data=subprocess.check_output(['ffmpeg','-v','error','-i',asset['video_path'],'-vsync','0','-vf','format=rgb24,crop=144:90:175:270','-f','rawvideo','-pix_fmt','rgb24','-'])
  cache[key]=np.frombuffer(data,np.uint8).reshape(-1,90,144,3)
 frames=cache[key];lo,hi=x['bounds'];selected=[i for i in range(len(frames)) if lo-.12<=i/25<=hi+.12]
 canvas=Image.new('RGB',(1500,40+125*((len(selected)+9)//10)),'white');draw=ImageDraw.Draw(canvas)
 draw.text((5,5),x['case']+' '+key+' ALL 25fps frames (+/-120ms), unshifted',font=font,fill='black')
 for j,fi in enumerate(selected):
  xx=(j%10)*150;yy=40+(j//10)*125
  canvas.paste(Image.fromarray(frames[fi]),(xx,yy))
  draw.text((xx,yy+92),f'{fi/25:.2f}s f{fi}',font=font,fill='black')
 name=x['case']+'_'+key+'.png';canvas.save(out/name)
 sampling.append({'name':name,'frames':selected,'bounds':x['bounds']})
for case in sorted({x['case'] for x in rows}):
 group=[x for x in rows if x['case']==case]
 canvas=Image.new('RGB',(1100,40+len(group)*105),'white');draw=ImageDraw.Draw(canvas)
 draw.text((5,5),case+' 20ms RMS / 10ms hop; relative to full-utterance RMS; own phrase start',font=font,fill='black')
 for ri,x in enumerate(group):
  key=x['id'];a,sr=sf.read(assets[key]['source']);lo,hi=x['bounds'];base=40+ri*105
  w=a[round(lo*sr):round(hi*sr)];win=np.lib.stride_tricks.sliding_window_view(w,320)[::160]
  v=20*np.log10(np.maximum(np.sqrt(np.mean(win**2,axis=1)),1e-10)/np.sqrt(np.mean(a**2)))
  draw.text((0,base),key.replace('BAC009',''),font=font,fill='black')
  draw.line((215,base+80,1060,base+80),fill='gray')
  pts=[(215+float(i*.01+.01)*275,base+80-float(np.clip(val+40,0,50))*1.4) for i,val in enumerate(v)]
  draw.line(pts,fill='#1764aa',width=2)
  for tick in np.arange(0,3.01,.5):
   xx=215+float(tick)*275;draw.text((xx,base+83),str(tick),font=font,fill='gray')
  draw.text((900,base),f'{hi-lo:.3f}s',font=font,fill='black')
 canvas.save(OUT/(case+'_rms.png'))
(OUT/'fullstrips_receipt.json').write_text(json.dumps({'post_initial_visual_followup':True,'continuous_native_frames':sampling,'no_scoring_or_inference':True},indent=2))
print(json.dumps({'strips':len(sampling),'rms_plots':3,'files_sha256':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}}))
