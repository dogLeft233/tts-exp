"""Independent live-media and cached-array engineering audit; never model forward."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['NUMBA_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,subprocess,sys,time,shutil,platform
from fractions import Fraction
from pathlib import Path
import numpy as np
import soundfile as sf
import cv2
import python_speech_features as psf

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'runs/tts_fixed_envelope_generation_cross_20260927'
OUT=ROOT/'runs/tts_fixed_envelope_firstcal_media_audit_20260927'
FF=Path('/home/wjj/miniconda3/bin/ffmpeg')
FP=Path('/home/wjj/miniconda3/bin/ffprobe')
PH='0b0c37864e608e611c132091601d3880a936f3f1ab17e77b83686bff33674f8a'
CAP=4*2**20
inputs={}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p,h=None):
 p=Path(p);got=sha(p);assert h is None or got==h,(str(p),got,h);inputs[str(p.resolve())]=got;return p
def read(p,h=None):return json.loads(bind(p,h).read_text())
def save(name,x):
 s=json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
 assert shutil.disk_usage(ROOT).free>=int(4.5*2**30)+len(s.encode())
 assert sum(f.stat().st_size for f in OUT.iterdir() if f.is_file())+len(s.encode())<CAP
 p=OUT/name;assert not p.exists();p.write_text(s)
def probe(p,video):
 entry=('stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames:frame=pts,best_effort_timestamp_time'
        if video else 'stream=index,codec_type,codec_name,sample_rate,channels,sample_fmt,time_base,start_time,duration,duration_ts:frame=pts,nb_samples')
 return json.loads(subprocess.check_output([str(FP),'-v','error','-show_streams','-show_frames','-show_entries',entry,'-of','json',str(p)]))
def bgr(p):
 proc=subprocess.Popen([str(FF),'-v','error','-threads','1','-i',str(p),'-threads','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 digest=hashlib.sha256();count=0;size=224*224*3
 while True:
  block=proc.stdout.read(size)
  if not block:break
  assert len(block)==size;digest.update(block);count+=1
 err=proc.stderr.read();assert proc.wait()==0,err.decode(errors='replace')
 return digest.hexdigest(),count
def jpeg(p):
 # New independent CPU decoder: arbitrary 4093-byte boundaries versus complete
 # concatenated image stream; no SyncNetEngine or neural network is called.
 cmd=[str(FF),'-y','-loglevel','error','-i',str(p),'-threads','1','-f','image2pipe','-vcodec','mjpeg','pipe:1']
 proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 buf=b'';complete=[];stream=[]
 while True:
  chunk=proc.stdout.read(4093)
  if not chunk:break
  complete.append(chunk);buf+=chunk
  while b'\xff\xd9' in buf:
   end=buf.index(b'\xff\xd9')+2;frame=buf[:end];buf=buf[end:]
   assert frame.startswith(b'\xff\xd8')
   decoded=cv2.imdecode(np.frombuffer(frame,np.uint8),cv2.IMREAD_COLOR)
   assert decoded is not None and decoded.shape==(224,224,3)
   stream.append(ah(decoded))
 err=proc.stderr.read();assert proc.wait()==0,err.decode(errors='replace');assert not buf
 blob=b''.join(complete);parts=blob.split(b'\xff\xd9');assert parts[-1]==b''
 full=[ah(cv2.imdecode(np.frombuffer(v+b'\xff\xd9',np.uint8),cv2.IMREAD_COLOR)) for v in parts[:-1]]
 assert full==stream
 return {'frames':len(stream),'stream_full_frame_hashes_exact':True,'JPEG_stream_sha256':hashlib.sha256(blob).hexdigest(),'decoded_frame_hashes':stream,'read_block_bytes':4093}
def video(p,expect):
 bind(p,expect['sha256']);pr=probe(p,True);assert len(pr['streams'])==1
 st=pr['streams'][0];assert st['codec_type']=='video' and st['codec_name']=='ffv1'
 assert (st['width'],st['height'])==(224,224) and Fraction(st['r_frame_rate'])==25
 frames=pr['frames'];count=len(frames);assert count==expect['frames']
 tb=Fraction(st['time_base']);pts=[int(f['pts']) for f in frames]
 assert all(Fraction(t)*tb==Fraction(i,25) for i,t in enumerate(pts))
 pix,n=bgr(p);assert n==count and pix==expect['pixel_sha256']
 jp=jpeg(p);assert jp['frames']==count
 return {'path':str(p),'present_during_check':True,'file_sha256':sha(p),'streams':pr['streams'],
         'PTS':pts,'time_base':str(tb),'pixel_sha256':pix,'frames':count,'JPEG':jp}
def loadnp(p,h=None):
 return np.load(bind(p,h),allow_pickle=False)
def front(y):
 mf=psf.mfcc(y.astype(np.float64)*32768,samplerate=16000)
 win=np.stack([mf[4*i:4*i+20].T.astype(np.float32) for i in range((len(mf)-20)//4+1)])
 return mf,win

def main():
 assert not OUT.exists();OUT.mkdir()
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);cv2.setNumThreads(2)
 protocol=read(P/'protocol.json',PH);review=read(P/'reviewer_pass.json')
 assert review['status']=='PASS' and review['protocol_sha256']==PH
 assert read(review['receipt'],review['receipt_sha256'])['status']=='PASS'
 live=read(P/'first_cal_live.json');assert live['protocol_sha256']==PH and len(live['files'])==12
 assert live['no_rest_cal_or_eval_before_audit']
 row=next(r for r in protocol['rows'] if r['id']=='a1_001');assert row['split']=='calibration'
 for f,h in live['files'].items():bind(f,h)
 bind(FF,protocol['dependencies'][str(FF)]);bind(FP,protocol['dependencies'][str(FP)])
 sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
 bind(ROOT/'third_party/Wav2Lip/audio.py',protocol['dependencies'][str(ROOT/'third_party/Wav2Lip/audio.py')])
 save('plan.json',{'status':'frozen_before_independent_media_decode','protocol_sha256':PH,
      'live_manifest_sha256':sha(P/'first_cal_live.json'),'code_sha256':sha(__file__),
      'no_model_forward':True,'max_new_bytes':CAP,'cpu_affinity':sorted(os.sched_getaffinity(0)),
      'checks':'live12 hash; FFV1 BGR/PTS; perclip external WAV sampleclock; saved A/V vs RAW/FIXED/ENV; independent envelope/gain/mean-RMS/mel/MFCC/windows; CPU JPEG full/stream; existing +5frame delay arrays only',
      'numerical_gates':{'saved_waveform_max':1e-7,'gain_max':1e-12,'saved_RMS_relative':1e-6,'inverse_saved':1e-7,'mel_MFCC_windows_exact':True,'A_V_exact':True,'delay_overlap':.15},
      'runtime':{'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__,'platform':platform.platform(),'executable':sys.executable},'start_epoch':time.time()})
 media=[];numeric={};all_oldvideo=[]
 for arm in ['N','T']:
  z=row['arms'][arm];ctrl=P/'controls'/arm;ops=read(P/'operations/a1_001'/(arm+'.json'))
  meta=read(P/'features/a1_001'/arm/'metadata.json');assert meta['operation_sha256']==sha(P/'operations/a1_001'/(arm+'.json'))
  bridges=read(ctrl/'bridge.json');assert bridges['passed']
  # Reconstruct operation from exact prior float32 FIXED waveform in float64.
  x,sr=sf.read(bind(z['FIXED']['waveform'],z['FIXED']['sha256']),dtype='float64');assert sr==16000
  fixed_info=sf.info(z['FIXED']['waveform']);assert fixed_info.subtype=='FLOAT'
  R=protocol['target_RMS'];w=np.hanning(513);w/=sum(w)
  env=np.sqrt(np.maximum(np.convolve(np.pad(x*x,256,mode='reflect'),w,mode='valid'),0))
  gain=np.maximum(env/R,.1)**(-.5);gained=x*gain;gain*=R/np.sqrt(np.mean(gained*gained))
  predicted=(x*gain).astype(np.float32)
  wp=next(Path(f) for f in live['files'] if f.endswith(f'a1_001_{arm}.wav'))
  y,sr=sf.read(wp,dtype='float32');assert sr==16000 and y.ndim==1 and len(y)==len(x)==ops['samples']
  wavprobe=probe(wp,False);st=wavprobe['streams'];assert len(st)==1;st=st[0]
  assert st['codec_name']=='pcm_f32le' and st['sample_fmt']=='flt' and st['channels']==1 and st['sample_rate']=='16000' and Fraction(st['time_base'])==Fraction(1,16000)
  counts=np.array([f['nb_samples'] for f in wavprobe['frames']],dtype=np.int64);pts=np.array([f['pts'] for f in wavprobe['frames']],dtype=np.int64)
  assert np.array_equal(pts,np.r_[0,np.cumsum(counts)[:-1]]) and counts.sum()==len(y) and int(st['duration_ts'])==len(y)
  decoded=subprocess.check_output([str(FF),'-v','error','-threads','1','-i',str(wp),'-f','f32le','-acodec','pcm_f32le','pipe:1'])
  assert np.array_equal(np.frombuffer(decoded,'<f4'),y) and ah(y)==ops['waveform_float32_sha256']
  fp=next(Path(f) for f in live['files'] if f.endswith(f'a1_001_{arm}_frontend.npz'))
  with loadnp(fp) as f:fronts={k:f[k] for k in f.files}
  gainerr=float(np.max(abs(gain-fronts['gain'])));waveerr=float(np.max(abs(predicted-y)))
  assert gainerr<=1e-12 and waveerr<=1e-7
  rmsrel=abs(np.sqrt(np.mean(y.astype(float)**2))/R-1);inverse=float(abs(y.astype(float)/fronts['gain']-x).max())
  assert rmsrel<=1e-6 and inverse<=1e-7 and np.max(abs(y))<=.98 and np.isfinite(y).all() and (fronts['gain']>0).all() and np.all(y[x==0]==0)
  assert ah(fronts['gain'])==ops['gain_float64_sha256']
  # No waveform interpolation or inference. Recompute CPU frontend directly.
  loaded=audio.load_wav(str(wp),16000);assert np.array_equal(loaded,y)
  mel=audio.melspectrogram(y).astype(np.float32);mf,win=front(y)
  assert np.array_equal(mel,fronts['mel']) and np.array_equal(mf,fronts['MFCC']) and np.array_equal(win,fronts['windows'])
  assert ah(mel)==ops['mel_sha256'] and ah(mf)==ops['MFCC_float64_sha256'] and ah(win)==ops['windows_sha256']
  chunks=[];starts=[];i=0
  while True:
   start=int(i*80/25)
   if start+16>mel.shape[1]:start=mel.shape[1]-16;chunks.append(mel[:,start:]);starts.append(start);break
   chunks.append(mel[:,start:start+16]);starts.append(start);i+=1
  assert starts==ops['mel_chunk_starts'] and ah(np.asarray(chunks,np.float32))==ops['chunks_float32_sha256']
  assert ops['MFCC_window_starts']==list(range(0,len(win)*4,4))
  assert len(chunks)==z['FIXED']['frames'] and len(win)==z['FIXED']['audio_windows']
  pcm,sr=sf.read(bind(z['source']['path'],z['source']['sha256']),dtype='int16');assert sr==16000
  raw,sr=sf.read(bind(z['RAW']['waveform'],z['RAW']['sha256']),dtype='float64');assert np.array_equal(raw,pcm.astype(float)/32768)
  rawmf,rawwin=front(raw);pcmmf=psf.mfcc(pcm,samplerate=16000)
  assert np.array_equal(rawmf,pcmmf)
  with loadnp(z['RAW']['frontend'],z['RAW']['frontend_sha256']) as f:assert np.array_equal(rawwin,f['windows'])
  avchecks={}
  for cond in ['RAW','FIXED','ENV']:
   c=read(ctrl/(cond+'.json'));vfile=Path(c['video_at_creation']['path']);assert str(vfile) in live['files']
   rec=video(vfile,c['video_at_creation']);assert rec['PTS']==c['decode']['PTS'];media.append(rec)
   aa=loadnp(c['A']['path'],c['A']['sha256']);vv=loadnp(c['V']['path'],c['V']['sha256'])
   assert ah(aa)==c['A']['raw_sha256'] and ah(vv)==c['V']['raw_sha256']
   if cond=='ENV':
    ar=loadnp(meta['audio']['path'],meta['audio']['sha256']);vr=loadnp(meta['visual']['path'],meta['visual']['sha256']);refpixel=meta['video_at_creation']['pixel_sha256']
   else:
    path=z['RAW_metadata'] if cond=='RAW' else z['baseline_metadata'];h=z['RAW_metadata_sha256'] if cond=='RAW' else z['baseline_metadata_sha256'];ref=read(path,h)
    with loadnp(ref['features'],ref['sha256']) as f:ar=f['audio'];vr=f['visual']
    refpixel=ref['video']['pixel_sha256'];oldv=Path(ref['video']['path'])
    # Only assert a video decoded independently when it really exists now.
    if oldv.is_file():
     original=video(oldv,ref['video']);assert original['pixel_sha256']==rec['pixel_sha256'];all_oldvideo.append(original)
   assert np.array_equal(aa,ar) and np.array_equal(vv,vr) and rec['pixel_sha256']==refpixel
   assert aa.shape==(len(win),1024) and vv.shape==(len(chunks)-4,1024)
   assert c['tests']['stream_full_exact'] and c['tests']['passed']
   avchecks[cond]={'audio_max':float(abs(aa-ar).max()),'visual_max':float(abs(vv-vr).max()),'pixel_equal':True,'producer_full_vs_stream_forward_exact_assertion':True}
  primary=video(Path(meta['video_at_creation']['path']),meta['video_at_creation']);media.append(primary)
  duplicate=next(v for v in media if v['path'].endswith(f'{arm}_ENV_repeat.avi'))
  assert primary['pixel_sha256']==duplicate['pixel_sha256'] and primary['JPEG']==duplicate['JPEG']
  assert min(primary['frames'],len(y)//640)-5==z['joint_L']==meta['joint_L']
  delaymeta=bridges['delay'];delay=loadnp(delaymeta['path'],delaymeta['sha256'])
  with loadnp(z['RAW_features'],z['RAW_features_sha256']) as f:ba=f['audio'];bv=f['visual']
  L=z['joint_L'];queries=np.arange(25,L-25);assert len(queries)>0
  def curve(a):
   return np.array([np.sqrt(np.sum((bv[queries].astype(float)-a[queries+lag].astype(float)+1e-6)**2,axis=1)).mean() for lag in range(-15,16)])
  c0,cd=curve(ba),curve(delay);bl=int(c0.argmin())-15;dl=int(cd.argmin())-15;error=float(abs(cd[5:]-c0[:-5]).max())
  assert abs(dl-bl-5)<=1 and error<=.15
  numeric[arm]={'samples':len(y),'sample_rate':16000,'audio_duration_seconds':len(y)/16000,'audio_probe_stream':st,'audio_frame_pts':pts.tolist(),'audio_frame_samples':counts.tolist(),
    'video_frames':primary['frames'],'video_duration_seconds':primary['frames']/25,'joint_L':L,
    'audio_windows':len(win),'visual_windows':primary['frames']-4,'no_muxed_audio_video_has_only_video':True,
    'gain_max_abs_error':gainerr,'waveform_max_abs_error':waveerr,'RMS_relative':float(rmsrel),'inverse_max_abs':inverse,
    'WAV_soundfile_ffmpeg_exact':True,'FLOAT_WAV_load_exact':True,'mel_MFCC_windows_exact':True,'mel_chunk_indices_exact':True,'PCM_FLOAT_MFCC_exact':True,
    'A_V':avchecks,'delay5':{'base_lag':bl,'delay_lag':dl,'overlap_curve_max':error,'query_count':len(queries)},
    'no_model_forward_performed':True}
  print('ARM_PASS',arm,flush=True)
 # Recheck all twelve live files immediately before making release eligible.
 for f,h in live['files'].items():assert sha(f)==h
 save('media.json',{'videos':media,'old_baseline_videos_actually_decoded':all_oldvideo})
 save('numerical.json',numeric);save('inputs.json',inputs)
 save('receipt.json',{'status':'PASS','protocol_sha256':PH,'media_hashes':live['files'],
      'checked_while_files_existed':True,'first_cal_live_sha256':sha(P/'first_cal_live.json'),
      'independent_decoded_new_videos':len(media),'independent_decoded_existing_baseline_videos':len(all_oldvideo),
      'video_PTS_25fps_all':True,'WAV_sampleclock_and_counts_exact':True,'RAW_FIXED_ENV_repeat_A_V_exact':True,
      'CPU_JPEG_full_stream_pixels_exact':True,'producer_full_stream_neural_output_claim_only':'producer exact assertions and sealed code checked; auditor did NOT forward networks or retain independently computed neural output',
      'delay5_controls':{a:numeric[a]['delay5'] for a in numeric},'no_gpu_model_generation':True,
      'not_physical_sync_truth':True,'code_sha256':sha(__file__),
      'evidence_hashes':{n:sha(OUT/n) for n in ['plan.json','media.json','numerical.json','inputs.json']},
      'created_epoch':time.time(),'release_allowed_paths':list(live['files'])})
 print('PASS_RECEIPT',str(OUT/'receipt.json'),sha(OUT/'receipt.json'),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  if OUT.exists() and not (OUT/'failure.json').exists():save('failure.json',{'status':'FAIL','error':repr(e),'time':time.time(),'do_not_release_media':True})
  raise
