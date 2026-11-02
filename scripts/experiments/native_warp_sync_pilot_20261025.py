"""Fresh matched Wav2Lip native drivers; official SyncNet full-duration pilot."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, pickle, shutil, subprocess, sys, time
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/native_warp_sync_pilot_20261025_attempt01'
PILOT=ROOT/'runs/cogenav_dtw_pilot_20261025_attempt01'
W2L=ROOT/'third_party/Wav2Lip'; SYNC=ROOT/'third_party/syncnet_python'
SPY='/home/wjj/.venvs/syncnet/bin/python'; WPY='/home/wjj/.venvs/wav2lip/bin/python'
ROLES=('N','stretch','pause','combined','T')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def scalar(z):
 import torch
 x=torch.from_numpy(z);mu=x.mean(0);val,ix=mu.min(0)
 return dict(C=float(torch.median(mu)-val),D=float(val),B=float(torch.median(mu)),offset=int(15-ix),windows=len(z),curve=mu.tolist())
def pcm_sha(x):return hashlib.sha256(np.asarray(x,dtype='<i2').tobytes()).hexdigest()
def quantize(x):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=1 or not np.isfinite(x).all() or np.max(np.abs(x))>=1:raise ValueError('invalid/clip audio')
 # libsndfile PCM16 conversion floor, common for every arm, no RMS renormalization.
 return np.clip(np.floor(x*32768),-32768,32767).astype(np.int16)
def paired_summary(rows):
 by={(r['sample_id'],r['role']):r['C'] for r in rows if r['role'] in ROLES}
 ids=sorted({s for s,r in by});table=[]
 for sid in ids:
  v={role:by[sid,role] for role in ROLES};table.append(dict(sample_id=sid,**v,delta={role:v[role]-v['N'] for role in ROLES[1:]}))
 return dict(per_source=table,means={role:float(np.mean([r[role] for r in table])) for role in ROLES},deltas={role:dict(mean=float(np.mean([r['delta'][role] for r in table])),positive=sum(r['delta'][role]>0 for r in table),values=[r['delta'][role] for r in table]) for role in ROLES[1:]})
class Runner:
 def __init__(self):self.start=time.monotonic();self.i=0
 def check(self):
  if time.monotonic()-self.start>10800:raise TimeoutError('three hour global limit')
  if shutil.disk_usage(OUT).free<5*1024**3:raise RuntimeError('below5GiB free')
 def cmd(self,args,name,cwd=ROOT,timeout=300):
  self.check();log=OUT/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True);t=time.monotonic()
  with log.open('x') as f:
   f.write(json.dumps(list(map(str,args)))+'\n');f.flush()
   p=subprocess.run(list(map(str,args)),cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
  put(OUT/'receipts'/f'{name}.json',dict(command=list(map(str,args)),cwd=str(cwd),returncode=p.returncode,elapsed_s=time.monotonic()-t,log_sha256=sha(log)))
  if p.returncode:raise RuntimeError('command failed '+name)
 def capture(self,args):return subprocess.check_output(list(map(str,args)))
 def prepare(self):
  OUT.mkdir(exist_ok=False);sel=read(PILOT/'selected_inputs.json');signals=read(PILOT/'signals.json')
  gp=read(ROOT/'runs/qwen_codec_roundtrip_20261022_attempt01/input_manifest.json')['generation']
  assert sha(gp['face'])==gp['face_sha256'];assert sha(W2L/'checkpoints/wav2lip_gan.pth')==gp['weights_sha256']
  inputs=[];ids=sorted({r['sample_id'] for r in sel})
  for sid in ids:
   n=next(r for r in sel if r['sample_id']==sid and r['arm']=='N');t=next(r for r in sel if r['sample_id']==sid and r['arm']!='N')
   for role in ROLES:
    old=n if role!='T' else t;condition='identity' if role in ('N','T') else role
    source=PILOT/'signals'/old['id']/condition/'audio.wav' if role!='T' else Path(t['audio_path'])
    expected=next(r['audio_sha256'] for r in signals if r['id']==old['id'] and r['condition']==condition) if role!='T' else t['audio_sha256']
    assert sha(source)==expected;x,sr=sf.read(source,dtype='float64');assert sr==16000;y=quantize(x)
    dest=OUT/'audio'/sid/f'{role}.wav';dest.parent.mkdir(parents=True,exist_ok=True);sf.write(dest,y,16000,subtype='PCM_16')
    identity=None
    if role in ('N','T'):
     orig,osr=sf.read(old['audio_path'],dtype='int16');identity=bool(np.array_equal(orig,y));assert osr==16000 and identity
    inputs.append(dict(sample_id=sid,role=role,source_path=str(source),source_sha256=expected,audio_path=str(dest),audio_sha256=sha(dest),pcm_sha256=pcm_sha(y),samples=len(y),peak=float(np.max(abs(x))),clipped_samples=0,canonical_identity=identity,tts_arm=t['arm'],historical_C=old['score_float32']))
  # First and last source original repeats, no choice based on newly observed scores.
  for sid in (ids[0],ids[-1]):inputs.append(dict(next(r for r in inputs if r['sample_id']==sid and r['role']=='N'),role='N_repeat'))
  pins={str(p):sha(p) for base in (W2L,SYNC) for p in base.rglob('*.py') if '__pycache__' not in p.parts}
  for p in [Path(__file__),Path(gp['face']),W2L/'checkpoints/wav2lip_gan.pth',SYNC/'data/syncnet_v2.model',SYNC/'detectors/s3fd/weights/sfd_face.pth',PILOT/'protocol.json',PILOT/'selected_inputs.json',PILOT/'signals.json']:pins[str(p)]=sha(p)
  put(OUT/'protocol.json',dict(status='FROZEN_BEFORE_ALL_FORWARD',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='same six known-warp pilot sources chosen on historical T>N; L06 preferred elseL17',roles=ROLES,expected_videos=32,generation=gp,settings='fresh32; unmodified Wav2Lip GAN inference, static image,25fps,batch4,nosmooth; native own duration; no waveform/video retiming; default inference seeds unchanged; two preselected N repeats',scoring='unmodified official run_pipeline min_track25 then run_syncnet batch20/vshift15; first sorted printed track; canonical float32 torch mean/median; own PCM16 driver losslessly remuxed to generated video; full native support including inserted pauses',analysis='source-equal paired differences vs fresh N; selected6 descriptive, no population test; report all three transformations; arbitrary rates .8,1.25,1 and320ms silence, not measured TTS rhythm mimic; phase-vocoder artifacts and support changes limit pure timing attribution',pins=pins,inputs=inputs,limits=dict(max_videos=32,min_free_GiB=5,global_seconds=10800,training=False,new_TTS=False)))
  shutil.copyfile(__file__,OUT/'source_frozen.py');put(OUT/'inputs.json',inputs)
  put(OUT/'runtime.json',dict(python=sys.version,numpy=np.__version__,soundfile=sf.__version__,GPU=self.capture(['nvidia-smi']).decode(),ffmpeg=self.capture(['ffmpeg','-version']).decode()))
  print('FROZEN',sha(OUT/'protocol.json'),flush=True)
 def render(self):
  p=read(OUT/'protocol.json');gp=p['generation'];videos=[]
  for row in p['inputs']:
   sid,role=row['sample_id'],row['role'];key=sid+'_'+role
   work=OUT/'render_work'/key;(work/'temp').mkdir(parents=True,exist_ok=False)
   dest=OUT/'video'/sid/f'{role}.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
   self.cmd([WPY,W2L/'inference.py','--checkpoint_path',W2L/'checkpoints/wav2lip_gan.pth','--face',gp['face'],'--audio',row['audio_path'],'--outfile',dest,'--face_det_batch_size','4','--wav2lip_batch_size','4','--nosmooth'],'render_'+key,work,300)
   probe=json.loads(self.capture(['ffprobe','-v','error','-show_streams','-show_format','-of','json',dest]));put(OUT/'probes'/f'{key}.json',probe)
   mux=OUT/'inputs'/sid/f'{role}.mkv';mux.parent.mkdir(parents=True,exist_ok=True)
   self.cmd(['ffmpeg','-nostdin','-v','error','-n','-copyts','-i',dest,'-i',row['audio_path'],'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le','-avoid_negative_ts','disabled',mux],'mux_'+key)
   raw=self.capture(['ffmpeg','-v','error','-i',mux,'-map','0:a:0','-ac','1','-ar','16000','-f','s16le','-']);assert hashlib.sha256(raw).hexdigest()==row['pcm_sha256'] and len(raw)==row['samples']*2
   # Bitstream copy preserves frames; explicitly check decoded frame checksums.
   frame=[]
   for media in [dest,mux]:frame.append(self.capture(['ffmpeg','-v','error','-i',media,'-map','0:v:0','-an','-f','framemd5','-']))
   assert frame[0]==frame[1]
   put(OUT/'checks'/f'{key}.json',dict(mux_pcm_exact=True,frames_exact=True,decoded_framemd5_sha256=hashlib.sha256(frame[0]).hexdigest()))
   videos.append(dict(row,video_path=str(dest),video_sha256=sha(dest),mux_path=str(mux),mux_sha256=sha(mux)))
   print('RENDER',key,flush=True)
  put(OUT/'video_manifest.json',videos)
 def score(self):
  records=[]
  for row in read(OUT/'video_manifest.json'):
   sid,role=row['sample_id'],row['role'];key=sid+'_'+role;ref='nw25_'+key
   self.cmd([SPY,SYNC/'run_pipeline.py','--videofile',row['mux_path'],'--reference',ref,'--data_dir',OUT/'data','--min_track','25','--overwrite'],'pipeline_'+key,SYNC,300)
   self.cmd([SPY,SYNC/'run_syncnet.py','--videofile',row['mux_path'],'--reference',ref,'--data_dir',OUT/'data','--initial_model',SYNC/'data/syncnet_v2.model'],'syncnet_'+key,SYNC,300)
   ap=OUT/'data/pywork'/ref/'activesd.pckl';dists=pickle.loads(ap.read_bytes());assert len(dists)>0
   scalars=[scalar(z) for z in dists];z=np.asarray(dists[0]);assert z.dtype==np.float32 and z.shape[1]==31
   item=OUT/'items'/key;item.mkdir(parents=True);np.savez_compressed(item/'distance.npz',distance=z)
   actual=OUT/'data/pytmp'/ref/'audio.wav';assert actual.exists();shutil.copyfile(actual,item/'scoring_audio.wav')
   crop=sorted((OUT/'data/pycrop'/ref).glob('0*.avi'));tracks=pickle.loads((OUT/'data/pywork'/ref/'tracks.pckl').read_bytes())
   info=[dict(crop_file=str(c),crop_sha256=sha(c),track_start_frame=int(tracks[i]['track']['frame'][0]),track_end_frame=int(tracks[i]['track']['frame'][-1])) for i,c in enumerate(crop)]
   rec=dict(row,**scalars[0],tracks=scalars,track_count=len(dists),crop_tracks=info,reference=ref,activesd_path=str(ap),activesd_sha256=sha(ap),distance_sha256=sha(item/'distance.npz'),scoring_audio_path=str(item/'scoring_audio.wav'),scoring_audio_sha256=sha(item/'scoring_audio.wav'))
   put(OUT/'scores'/f'{key}.json',rec);records.append(rec);print('SCORE',key,rec['C'],flush=True)
  put(OUT/'scores.json',records);summary=paired_summary(records)
  repeats=[]
  for r in records:
   if r['role']=='N_repeat':
    n=next(x for x in records if x['sample_id']==r['sample_id'] and x['role']=='N');repeats.append(dict(sample_id=r['sample_id'],delta_C=r['C']-n['C'],video_bytes_equal=r['video_sha256']==n['video_sha256']))
  summary.update(status='COMPLETE',n_sources=6,videos=32,repeats=repeats,protocol_sha256=sha(OUT/'protocol.json'))
  put(OUT/'summary.json',summary)
  lines=['# Natural timing transformations versus native TTS','', 'Selected six-source exploratory native Wav2Lip GAN/official SyncNet comparison. Full own-duration support; inserted silence included.','', '|Source|N|stretch|pause|combined|T|','|---|---:|---:|---:|---:|---:|']
  for r in summary['per_source']:lines.append('|'+r['sample_id']+'|'+'|'.join(f'{r[k]:.3f}' for k in ROLES)+'|')
  lines.extend(['','Means: '+json.dumps(summary['means']),'','Paired deltas: '+json.dumps(summary['deltas']),'','Controls: '+json.dumps(repeats),'','Arbitrary known-warp controls are not fitted TTS rhythm. Phase-vocoder spectral changes and different numbers of scored windows preclude pure temporal-cause identification. Score changes are evaluator evidence, not perceptual synchronization truth.'])
  (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 def run(self):
  try:self.prepare();self.render();self.score()
  except BaseException as exc:
   import traceback
   if OUT.exists():put(OUT/'terminal_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
   raise
if __name__=='__main__':Runner().run()
