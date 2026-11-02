"""Create labeled viewing copies. They are not the scored lossless containers."""
import argparse,subprocess
from pathlib import Path
import cv2
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
arms=[('gain_m6','-6 dB'),('gain_0','0 dB'),('gain_p6','+6 dB')]
cmd=['ffmpeg','-v','error','-nostdin','-y']
for arm,label in arms:cmd+=['-i',str(a.render/arm/'render_lossless.mkv')]
cmd+=['-i',str(a.render/'original_score_audio.wav')]
filters=[f"[{i}:v]drawtext=text='{label}':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.7:x=16:y=12[v{i}]" for i,(_,label) in enumerate(arms)]
filters.append('[v0][v1][v2]hstack=inputs=3[out]')
cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-map','3:a:0','-c:v','libx264','-preset','fast',
      '-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-threads','2',
      str(a.output/'may_gain_comparison.mp4')]
subprocess.run(cmd,check=True)
panels=[]
for arm,label in arms:
    image=cv2.imread(str(a.render/arm/'frame149.png'));assert image is not None
    cv2.rectangle(image,(0,0),(511,42),(0,0,0),-1)
    cv2.putText(image,label,(16,31),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2,cv2.LINE_AA)
    panels.append(image)
cv2.imwrite(str(a.output/'frame149_comparison.png'),np.concatenate(panels,axis=1))
print(a.output,flush=True)
