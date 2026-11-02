"""Frozen original AVE on new TTS, with full repeat and natural-cache parity."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch
import soundfile as sf
import librosa
from torch.utils.data import DataLoader
from ave_cache_original import load_original_classes
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def main():
 out=ROOT/'features';assert not out.exists();out.mkdir()
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 Encoder,Dataset=load_original_classes(ROOT/'official')
 weights=ROOT/'official/nerf_triplane/checkpoints/audio_visual_encoder.pth'
 assert sha(weights)=='1f75cdb5556938590c64221f6c47a8d20ab3e368d53c166023eedb12279db8c4'
 model=Encoder().cuda().eval();state=torch.load(weights,map_location='cpu',weights_only=True)
 model.load_state_dict({'audio_encoder.'+k:v for k,v in state.items()},strict=True);model.requires_grad_(False)
 def forward(ds):
  rows=[]
  with torch.no_grad():
   for mel in DataLoader(ds,batch_size=64,shuffle=False):rows.append(model(mel.cuda()).cpu())
  x=torch.cat(rows);return torch.cat([x[:1].repeat(2,1),x,x[-1:].repeat(2,1)]).numpy()
 rows=[]
 for split in ['train','val']:
  wav=ROOT/'data/arms'/split/'gain_0/aud.wav';x,sr=sf.read(wav,dtype='float64');assert sr==16000
  loaded=librosa.load(str(wav),sr=16000)[0];assert np.array_equal(loaded,x.astype(np.float32))
  ds=Dataset(str(wav));values=forward(ds);assert np.isfinite(values).all()
  np.save(out/f'{split}_padded.npy',values)
  repeat=forward(ds);assert np.array_equal(values,repeat)
  np.save(out/f'{split}_repeat.npy',repeat)
  t=json.loads((ROOT/'data/arms'/split/'gain_0'/f'transforms_{split}.json').read_text())
  ids=np.array([r['aud_id'] for r in t['frames']]);assert ids.min()>=0 and ids.max()<len(values)
  aligned=values[ids,None,:];np.save(out/f'{split}_aligned.npy',aligned)
  natural_parity=None
  if split=='val':
   old=np.load(ROOT/'historical_features/val/gain_0/ave_padded.npy');assert np.array_equal(values,old)
   natural_parity=True
  np.save(out/f'{split}_mel_control.npy',ds.orig_mel)
  rows.append(dict(split=split,wav_sha256=sha(wav),padded_sha256=sha(out/f'{split}_padded.npy'),aligned_sha256=sha(out/f'{split}_aligned.npy'),
     padded_shape=list(values.shape),aligned_shape=list(aligned.shape),repeat_exact=True,natural_baseline_bitwise_parity=natural_parity,
     input_float32_cast_exact=True,input_rms=float(np.sqrt(np.mean(x*x))),input_peak=float(np.abs(x).max()),no_waveform_normalization=True))
  print(json.dumps(rows[-1]),flush=True)
 save(out/'receipt.json',dict(passed=True,weights_sha256=sha(weights),rows=rows,training_jobs=0,
  frontend_source_sha256=sha(ROOT/'official/nerf_triplane/utils.py'),network_source_sha256=sha(ROOT/'official/nerf_triplane/network.py')))
if __name__=='__main__':main()
