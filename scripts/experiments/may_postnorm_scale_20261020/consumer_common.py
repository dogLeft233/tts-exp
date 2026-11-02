from pathlib import Path
from functools import lru_cache
import json,csv,subprocess
import numpy as np,cv2
from imageio_ffmpeg import get_ffmpeg_exe
from protocol import OUT,DATA,OLD,sha,write
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

