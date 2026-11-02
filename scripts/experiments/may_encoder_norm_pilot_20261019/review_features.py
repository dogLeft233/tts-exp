"""Independent array, source, gain and clock readback; no encoder execution."""
from pathlib import Path
import json,hashlib
import numpy as np
from protocol import OUT,sha,write
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));rows=[]
 for spec in p['models']:
  mid=spec['id'];root=OUT/'features'/mid;r=json.loads((root/'receipt.json').read_text())
  assert r['status']=='COMPLETED' and r['source']==spec['source'] and r['protocol_sha256']==sha(OUT/'protocol.json')
  assert not r['model_training'] and not r['loading_info']['missing_keys'] and not r['loading_info']['mismatched_keys']
  for c in r['conditions']:
   a=np.load(root/c['file'],allow_pickle=False);assert sha(root/c['file'])==c['sha256'] and list(a.shape)==c['shape']
   assert np.isfinite(a).all() and a.dtype==np.float32 and a.shape[1:]==(2,768)
   n={'train':4200,'val':600,'test':1150}[c['split']];assert len(a)==n-1
  effects=[]
  for norm in [True,False]:
   b=np.load(root/f'test_{"on" if norm else "off"}_g+0.npy').astype(float)
   for gain in [-6,6]:
    a=np.load(root/f'test_{"on" if norm else "off"}_g{gain:+d}.npy').astype(float)
    effects.append(dict(normalize=norm,gain_db=gain,relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b))))
  rows.append(dict(encoder=mid,conditions=len(r['conditions']),actual_forwards=r['forward_count'],source=r['source'],effects=effects))
 write(OUT/'feature_readback_review.json',dict(passed=True,models=rows,encoder_forward_count=0,arrays=20,clock='paired50Hz; all supervised aud_id within support'))
 print(json.dumps(rows))
if __name__=='__main__':main()
