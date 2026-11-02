"""Independent input reconstruction and report65 first-chunk bridge; CPU only."""
from pathlib import Path
import json,hashlib
import numpy as np,soundfile as sf
from protocol import OUT,DATA,sha,write
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text());rec=json.loads((OUT/'features/receipt.json').read_text());assert rec['status']=='COMPLETED'
 old=Path(p['noise_parent']);calls=json.loads((old/'forward_calls.json').read_text());bridges=[];chunks=0
 for row in rec['conditions']:
  x,sr=sf.read(DATA/'arms'/row['split']/'gain_0/aud.wav',dtype='float64');x=x.astype('float32');s=(x-np.mean(x))/np.sqrt(np.var(x)+1e-7)
  assert sr==16000 and digest(s)==row['native_sha256']
  if row['kind']=='none':a=s
  else:
   pin=next(r for r in p['noise_pins'] if r['kind']==row['kind']);assert sha(OUT/pin['path'])==pin['sha256']==sha(old/pin['path'])
   n=np.load(OUT/pin['path']);a=s+n*np.float32(10**((30-row['snr_db'])/20))
  actual=a*np.float32(row['input_scale']);assert digest(actual)==row['input_sha256']
  for c in row['chunks']:assert digest(actual[c['start']:c['start']+c['samples']])==c['input_sha256'];chunks+=1
  z=np.load(OUT/'features'/row['file']);assert z.dtype==np.float32 and np.isfinite(z).all() and list(z.shape)==row['shape'] and sha(OUT/'features'/row['file'])==row['sha256']
  # Old complete call input/output hashes independently bind the exact whole first chunk.
  if row['split']=='train':
   first=row['chunks'][0];matches=[c for c in calls if c['dtype']=='float32' and c['epsilon']==1e-5 and c['input_sha256']==first['input_sha256']]
   assert matches and all(c['last_sha256']==first['last_sha256'] for c in matches),row['arm_id']
   bridges.append(dict(arm=row['arm_id'],input_and_last_exact_report65=True))
  if row['kind']=='none':
   parent=Path(p['feature_parent'])/'features/wav2vec2_base'/f"{row['split']}_s{row['input_scale']:.2f}.npy"
   assert np.array_equal(z,np.load(parent))
 assert len(rec['conditions'])==22 and len(bridges)==18 and chunks==172
 write(OUT/'feature_independent_review.json',dict(passed=True,conditions=22,encoder_input_chunks_reconstructed=chunks,full_train_first_chunks_bridged_report65=bridges,clean_complete_features_exact_report56=6,no_second_normalization=True,training_updates=0))
 print(json.dumps(dict(passed=True,conditions=22,chunks=chunks,first_chunk_bridges=18)))
if __name__=='__main__':main()
