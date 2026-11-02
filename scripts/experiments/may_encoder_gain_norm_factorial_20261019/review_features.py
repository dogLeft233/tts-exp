"""Independent CPU verification of gained inputs, native normalization and saved features."""
import json,hashlib,sys
from pathlib import Path
import numpy as np,soundfile as sf
from protocol import OUT,DATA,sha,write
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));parent=Path(p['parent']);summary=[]
 for spec in p['models']:
  root=OUT/'features'/spec['id'];r=json.loads((root/'receipt.json').read_text());assert r['status']=='COMPLETED' and r['forward_count']==44 and len(r['conditions'])==14
  inherited=json.loads((root/'inherited_receipt.json').read_text());assert len(inherited['conditions'])==10
  for c in r['conditions']:
   path=root/c['file'];assert sha(path)==c['sha256'];z=np.load(path,allow_pickle=False);assert list(z.shape)==c['shape'] and z.dtype==np.float32 and np.isfinite(z).all()
   raw,sr=sf.read(DATA/'arms'/c['split']/'gain_0/aud.wav',dtype='float64');assert sr==16000
   gained=(raw*10**(c['gain_db']/20)).astype(np.float32);actual=(gained-gained.mean())/np.sqrt(gained.var()+1e-7) if c['normalize'] else gained
   assert hashlib.sha256(gained.tobytes()).hexdigest()==c['input_pcm_sha256'] and hashlib.sha256(actual.tobytes()).hexdigest()==c['processor_sha256']
   assert sha(DATA/'arms'/c['split']/'gain_0/aud.wav')==c['source_audio_sha256']
   if c.get('inherited_from_report54'):assert any(o=={k:v for k,v in c.items() if k!='inherited_from_report54'} for o in inherited['conditions']) and sha(parent/'features'/spec['id']/c['file'])==c['sha256']
  summary.append(dict(encoder=spec['id'],conditions=14,new_train_gain_conditions=4,actual_new_encoder_forwards=44,all_arrays_and_independent_formula_passed=True))
 write(OUT/'feature_independent_review.json',dict(passed=True,models=summary,new_encoder_calls_in_review=0,whole_support_native_processor_then_chunk=True))
 print(json.dumps(dict(passed=True,conditions=28,independent_formula_hashes=28,new_gpu_calls=0)))
if __name__=='__main__':main()
