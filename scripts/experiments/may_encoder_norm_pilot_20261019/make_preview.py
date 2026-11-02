"""A fixed first-clip viewing montage, made only from qualified lossless files."""
from pathlib import Path
import json,hashlib,subprocess
import sys
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps')
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from imageio_ffmpeg import get_ffmpeg_exe
from protocol import OUT,DATA,sha,write
def main():
 assert json.loads((OUT/'analysis/qa.json').read_text())['passed']
 target=OUT/'analysis/normalization_comparison_preview.mp4';assert not target.exists()
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
 conditions=[]
 for mid in ['hubert_base','wav2vec2_base']:
  for train,test in [(True,True),(True,False),(False,True),(False,False)]:
   folder=OUT/'remote/renders'/(mid+('_on' if train else '_off'))/f'{"on" if test else "off"}_g+0/test_00'
   r=json.loads((folder/'receipt.json').read_text());path=folder/'original_pcm.mkv';assert sha(path)==r['mux_sha256']
   cap=cv2.VideoCapture(str(path));assert cap.isOpened();conditions.append(dict(mid=mid,train=train,test=test,cap=cap,receipt=r,path=path))
 audio=DATA/'splits/test/test_00_original_pcm.wav'
 proc=subprocess.Popen([get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1536x888','-r','25','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-frames:v','250','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',str(target)],stdin=subprocess.PIPE)
 for i in range(250):
  canvas=Image.new('RGB',(1536,888),'white');draw=ImageDraw.Draw(canvas);draw.text((10,5),'Viewing copy only | driver 0 dB | original audio | fixed first test clip | no perceptual inference',font=small,fill='black')
  for ix,c in enumerate(conditions):
   ok,bgr=c['cap'].read();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert hashlib.sha256(rgb.tobytes()).hexdigest()==c['receipt']['frames'][i]['rgb_sha256']
   row,col=divmod(ix,4);y=32+row*428;draw.text((col*384+8,y+4),f'{c["mid"]}  T:{"on" if c["train"] else "off"} I:{"on" if c["test"] else "off"}',font=font,fill='black')
   canvas.paste(Image.fromarray(rgb).resize((384,384)),(col*384,y+36))
  proc.stdin.write(np.asarray(canvas).tobytes())
 proc.stdin.close();assert proc.wait()==0
 for c in conditions:c['cap'].release()
 write(OUT/'analysis/preview_receipt.json',dict(passed=True,frames=250,fps=25,clip_id='test_00',driver_gain_db=0,preview_only=True,used_for_scores=False,source_files=[dict(path=str(c['path']),sha256=sha(c['path'])) for c in conditions],preview_sha256=sha(target)))
 print(str(target))
if __name__=='__main__':main()
