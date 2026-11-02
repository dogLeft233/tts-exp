"""New AVE forwards for every gain after full-recording RMS; original feature caches retained."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np,torch,soundfile as sf
from torch.utils.data import DataLoader
from ave_cache import load_original_classes,sha
from ave_rms_normalization_v1 import normalize,TARGET_RMS
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
RUN=ROOT/'trials/patrick_ave_rms_coarse_v1'
def run():
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 contract=json.loads((RUN/'contract.json').read_text())
 assert contract['normalization']['target_rms']==TARGET_RMS
 for n,s in contract['sources'].items():assert sha(RUN/'sources'/n)==s
 AudioEncoder,AudDataset=load_original_classes(ROOT/'official_exact')
 model=AudioEncoder().cuda().eval();weight=ROOT/'official_exact/nerf_triplane/checkpoints/audio_visual_encoder.pth'
 assert sha(weight)==contract['ave_weight_sha256']
 state=torch.load(weight,map_location='cpu',weights_only=False)
 model.load_state_dict({'audio_encoder.'+k:v for k,v in state.items()},strict=True);model.requires_grad_(False)
 out=RUN/'features/PatrickLeahy';out.mkdir(parents=True,exist_ok=False)
 rows=[];differences=[]
 def forward(d):
  with torch.no_grad():v=torch.cat([model(m.cuda()).cpu() for m in DataLoader(d,batch_size=64,shuffle=False)])
  assert v.ndim==2 and v.shape[1]==512 and torch.isfinite(v).all()
  return torch.cat([v[:1].repeat(2,1),v,v[-1:].repeat(2,1)]).numpy()
 for role in ['main','test']:
  pcm,sr=sf.read(ROOT/'material/PatrickLeahy'/f'{role}_original.wav',dtype='int16');assert sr==16000 and pcm.ndim==1
  baselines={}
  for gain in [0,-6,6]:
   original=ROOT/'material/PatrickLeahy'/f'{role}_driver_{gain:+d}.wav'
   x,rate=sf.read(original,dtype='float64');assert rate==16000
   expected=pcm.astype(np.float64)/32768*10**(gain/20)
   assert np.array_equal(x.astype(np.float32),expected.astype(np.float32))
   y,r=normalize(x);assert len(y)==len(pcm)
   wave=out/f'{role}_{gain:+d}_rms_float32.wav';sf.write(wave,y,16000,subtype='FLOAT')
   reread,rate=sf.read(wave,dtype='float32');assert rate==16000 and np.array_equal(reread,y)
   d=AudDataset(str(wave));padded=forward(d);repeated=forward(d)
   assert np.array_equal(padded,repeated) and len(padded)>=len(pcm)//640
   feature=out/f'{role}_{gain:+d}_ave_padded.npy';np.save(feature,padded)
   indexes=np.linspace(0,len(d)-1,8).astype(int)
   np.save(out/f'{role}_{gain:+d}_mel_control.npy',np.stack([d.crop_audio_window(d.orig_mel,int(i)) for i in indexes]))
   row=dict(recording=role,gain_db=gain,source_driver_sha256=sha(original),original_pcm_sha256=sha(ROOT/'material/PatrickLeahy'/f'{role}_original.wav'),
    normalized_wave_path=str(wave),normalized_wave_sha256=sha(wave),normalization=r,
    features_path=feature.name,features_sha256=sha(feature),features_shape=list(padded.shape),
    full_repeat_exact=True,mel_shape=list(d.orig_mel.shape),mel_sha256=hashlib.sha256(d.orig_mel.tobytes()).hexdigest())
   rows.append(row)
   if gain==0:baselines=dict(wave=y,feature=padded)
   differences.append(dict(recording=role,gain_db=gain,wave_maxabs_to_0=float(np.max(np.abs(y-baselines['wave']))),feature_relative_l2_to_0=float(np.linalg.norm(padded.astype(np.float64)-baselines['feature'])/np.linalg.norm(baselines['feature'])),feature_exact_to_0=np.array_equal(padded,baselines['feature'])))
   print(json.dumps(row),flush=True)
 result=dict(passed=True,identity='PatrickLeahy',normalization=contract['normalization'],rows=rows,differences=differences,
  ave_weight_sha256=sha(weight),source_sha256=sha(Path(__file__)),contract_sha256=sha(RUN/'contract.json'),
  original_features_unchanged=True,training_started=False)
 (out/'ave_rms_qualification.json').write_text(json.dumps(result,indent=2))
 # Reconstitute the old datasets for original source-pin checks, then distinct normalized bindings.
 manifest=json.loads((ROOT/'runtime_restore_manifest.json').read_text())
 assert manifest['passed']
 for split in ['train','val','test']:
  prepared=ROOT/'prepared/PatrickLeahy'/split
  for gain,label in [(-6,'gain_m6'),(0,'gain_0'),(6,'gain_p6')]:
   old=ROOT/'datasets/PatrickLeahy'/split/label;assert old.is_dir()
   for kind in ['gt_imgs','torso_imgs','ori_imgs','parsing']:
    p=old/kind
    if not p.exists():p.symlink_to(prepared/kind,target_is_directory=True)
   for name in ['bc.jpg','bs.npy']:
    if not (old/name).exists():os.link(prepared/name,old/name)
   role='test' if split=='test' else 'main'
   if not (old/'aud.wav').exists():(old/'aud.wav').symlink_to(ROOT/'material/PatrickLeahy'/f'{role}_driver_{gain:+d}.wav')
   if not (old/'whole_ave.npy').exists():(old/'whole_ave.npy').symlink_to(ROOT/'features/PatrickLeahy'/f'{role}_{gain:+d}_ave_padded.npy')
   new=RUN/'datasets/PatrickLeahy'/split/label;new.mkdir(parents=True,exist_ok=False)
   for kind in ['gt_imgs','torso_imgs','ori_imgs','parsing']:(new/kind).symlink_to(prepared/kind,target_is_directory=True)
   for name in ['bc.jpg','bs.npy',f'transforms_{split}.json']:os.link(old/name,new/name)
   feature=next(r for r in rows if r['recording']==role and r['gain_db']==gain)
   (new/'whole_ave.npy').symlink_to(out/feature['features_path']);(new/'aud.wav').symlink_to(feature['normalized_wave_path'])
   binding=json.loads((old/'binding.json').read_text())
   binding.update(parent_binding_sha256=sha(old/'binding.json'),whole_ave_sha256=feature['features_sha256'],wave_sha256=feature['normalized_wave_sha256'],normalization_contract_sha256=sha(RUN/'contract.json'),normalization=contract['normalization'])
   cache=np.load(new/'whole_ave.npy',allow_pickle=False);aligned=cache[np.array(binding['source_ids']),None,:]
   binding['aligned_audio_tensor_sha256']=hashlib.sha256(aligned.tobytes()).hexdigest()
   (new/'binding.json').write_text(json.dumps(binding,indent=2))
 print(json.dumps(dict(passed=True,features=str(out),new_datasets=True)),flush=True)
if __name__=='__main__':run()
