"""Independent CPU audit of full-TTS downstream; no experiment pipeline imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import soundfile as sf
import json,hashlib,subprocess,io,datetime,traceback

RUN=Path('/home/wjj/tts-audio/tts-exp/runs/qwen_full_tts_extension_20261022_attempt01')
ROLES=('N','R','T1','T2','T3');POLICIES=('common','native_full','equal_count_spread');METRICS=('C','B','Dmin','D0','offset')
files={};errors=[]

def check(condition,message):
 if not condition:errors.append(message);raise AssertionError(message)

def sha(path):
 path=Path(path);h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def filecheck(path,expected=None):
 path=Path(path);key=str(path)
 if key not in files:files[key]=sha(path)
 if expected is not None:check(files[key]==expected,f'file SHA mismatch: {path}')
 return files[key]

def read(path):return json.loads(Path(path).read_text())
def close(a,b,name,tol=1e-12):
 a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64);check(a.shape==b.shape,f'{name} shape')
 err=float(np.max(abs(a-b))) if a.size else 0.;check(err<=tol,f'{name} error {err}');return err

def bootstrap(values):
 a=np.asarray(values,dtype=np.float64);g=np.random.Generator(np.random.PCG64(20261022));i=g.integers(0,len(a),(10000,len(a)));replicates=np.add.reduce(a[i],axis=1)/len(a)
 return {'n_speakers':len(a),'mean':float(np.mean(a)),'CI95':np.percentile(replicates,[2.5,97.5]).tolist(),
         'CI97_5':np.percentile(replicates,[1.25,98.75]).tolist(),'values':a.tolist(),'positive':int(np.sum(a>0)),'negative':int(np.sum(a<0))}

def compare_summary(a,b,name):
 for key in ('n_speakers','positive','negative'):check(a[key]==b[key],f'{name} {key}')
 return max(close(a[key],b[key],f'{name} {key}') for key in ('mean','CI95','CI97_5','values'))

def summarize_matrix(matrix,selection):
 curve=np.add.reduce(matrix[selection],axis=0,dtype=np.float64)/len(selection);j=int(np.argmin(curve));median=float(np.median(curve));minimum=float(curve[j])
 return {'C':median-minimum,'B':median,'Dmin':minimum,'D0':float(curve[15]),'offset':15-j,
         'curve':curve.tolist(),'query_count':len(selection),'best_lag_at_boundary':j in (0,30)}

def main():
 analysis_path=RUN/'analysis.json';check(analysis_path.is_file(),'analysis.json absent: audit must wait for complete file')
 protocol=read(RUN/'protocol.json');analysis=read(analysis_path);parent=Path(protocol['parent_run'])
 check(filecheck(RUN/'protocol.json')=='63c095d8e74488614f82dfb51cfb709ed4dd40671f9ce6a6adc2c771b9371724','protocol binding')
 filecheck(analysis_path);check(analysis['n_speakers']==10 and analysis['tts_seeds_per_speaker']==3,'sample denominator')
 seal_path=parent/'artifact_manifest.json';filecheck(seal_path,protocol['parent_artifact_manifest_sha256']);seal=read(seal_path)
 check(len(seal['files'])==seal['file_count']==2740,'whole parent seal count')
 parent_bytes=0
 for row in seal['files']:
  p=parent/row['path'];check(p.stat().st_size==row['bytes'],'parent size changed');filecheck(p,row['sha256']);parent_bytes+=row['bytes']
 check(parent_bytes==seal['total_bytes'],'parent seal total bytes')
 for key,row in protocol['parent_assets'].items():filecheck(row['path'],row['sha256'])
 print('PARENT_2740_SEAL_PASS',parent_bytes,flush=True)
 for s in protocol['samples']:
  filecheck(s['source_path'],s['source_sha256']);filecheck(s['input24_path'],s['input24_sha256'])
 tts=read(RUN/'tts/tts_manifest.json');audio=read(RUN/'audio_manifest.json');video=read(RUN/'video_manifest.json');fm=read(RUN/'feature_manifest.json')
 for name in ['tts/tts_manifest.json','audio_manifest.json','video_manifest.json','feature_manifest.json']:filecheck(RUN/name)
 check(len(tts['records'])==len(audio['records'])==len(video['records'])==len(fm['records'])==30,'all30 retained')
 tb={(r['sample_id'],int(r['seed_index'])):r for r in tts['records']};ab={(r['sample_id'],r['role']):r for r in audio['records']};vb={(r['sample_id'],r['role']):r for r in video['records']};fb={(r['sample_id'],r['role']):r for r in fm['records']}
 expected={(s['sample_id'],f'T{i}') for s in protocol['samples'] for i in range(1,4)}
 check(set(ab)==set(vb)==set(fb)==expected and len(tb)==30,'unique full seed layout')
 old_video=read(parent/'video_manifest.json');check(video['checkpoint_sha256']==old_video['checkpoint_sha256']==protocol['downstream']['weights_sha256'],'generation checkpoint consistent')
 check(video['inference_sha256']==old_video['inference_sha256'],'generation source consistent')
 check(fm['SyncNet_weights_sha256']=='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442','SyncNet checkpoint')
 filecheck(protocol['downstream']['face'],protocol['downstream']['face_sha256'])
 audio_checks=[];new_feature_count=0
 for (sid,role),a in ab.items():
  i=int(role[1:]);t=tb[sid,i];v=vb[sid,role];f=fb[sid,role]
  check(a['seed_index']==i and a['seed']==t['seed'],'seed binding')
  filecheck(a['path'],a['sha256']);filecheck(a['source_path'],a['source_sha256']);check(a['source_path']==t['path'] and a['source_sha256']==t['sha256'],'primary official T source')
  raw,srraw=sf.read(t['path'],dtype='float32');x,srx=sf.read(Path(a['path']).with_name(f'{role}_16k_float.wav'),dtype='float32');check(srraw==24000 and srx==16000,'canonical sample rates')
  data=subprocess.check_output(['/usr/bin/ffmpeg','-nostdin','-v','error','-i',t['path'],'-ac','1','-ar','16000','-c:a','pcm_f32le','-f','f32le','pipe:1'])
  check(np.array_equal(np.frombuffer(data,dtype='<f4'),x),'independent ffmpeg resample')
  rms=float(np.sqrt(np.mean(x.astype(float)**2)));gain=protocol['target_rms']/rms;close(rms,a['rms_before'],'rms_before');close(gain,a['gain'],'gain')
  normalized=x*np.float32(gain);check(float(abs(normalized).max())<1,'prequantization clipping')
  # Independent reconstruction through the specified PCM16 writer, no pipeline calls.
  memory=io.BytesIO();sf.write(memory,normalized,16000,format='WAV',subtype='PCM_16');memory.seek(0);expected_pcm,esr=sf.read(memory,dtype='int16')
  pcm,psr=sf.read(a['path'],dtype='int16');check(psr==esr==16000 and np.array_equal(pcm,expected_pcm),'normalized PCM exact')
  check(len(pcm)==len(x)==a['samples'],'untrimmed sample count');check(hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==a['pcm_sha256'],'PCM payload SHA')
  postrms=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2)));close(postrms,a['rms_after'],'rms_after')
  filecheck(v['path'],v['sha256']);check(v['audio_sha256']==a['sha256'],'render audio SHA')
  cmd=v['receipt']['command'];check(cmd[cmd.index('--audio')+1]==a['path'],'render audio path');check(cmd[cmd.index('--face')+1]==protocol['downstream']['face'],'render face');check(v['receipt']['returncode']==0,'render return code')
  check(cmd[cmd.index('--checkpoint_path')+1].endswith('third_party/Wav2Lip/checkpoints/wav2lip_gan.pth'),'render checkpoint path')
  check('--nosmooth' in cmd and cmd[cmd.index('--wav2lip_batch_size')+1]=='4','render frozen options')
  check(f['audio_meta']['pcm_sha256']==a['pcm_sha256'] and f['audio_meta']['sample_count']==len(pcm),'scoring source PCM')
  filecheck(f['audio_path'],f['audio_sha256']);filecheck(f['visual_path'],f['visual_sha256']);new_feature_count+=2
  for value in f['repeat'].values():check(value<=1e-5,'feature repeat control')
  meta=f['frames_meta'];frame_dir=RUN/'decoded_frames'/sid/role;frame_files=sorted(frame_dir.glob('*.jpg'))
  check(len(frame_files)==meta['frame_count']==len(meta['frame_sha256']),'decoded frame count')
  for frame,h in zip(frame_files,meta['frame_sha256']):filecheck(frame,h)
  audio_checks.append({'sample_id':sid,'role':role,'samples16':len(pcm),'samples24':len(raw),'rms_after':postrms,'rms_target_error':abs(postrms-protocol['target_rms']),'all_normalized_PCM_exact':True})
 check(new_feature_count==60,'60 new feature hashes')
 print('AUDIO30_RENDER30_FEATURE60_BINDINGS_PASS',flush=True)
 old_score=read(parent/'matrix_scores.json');ob={r['sample_id']:r for r in old_score['records']};rebuilt=[];matrix_checks=[];max_matrix=0.;max_scalar=0.;feature_arrays=0
 for r in analysis['records']:
  sid=r['sample_id'];condition_matrices={};counts={};arrays={}
  for role in ROLES:
   if role in ('N','R'):
    ap=Path(protocol['parent_assets'][f'features/{sid}/audio_{role}.npy']['path']);vp=Path(protocol['parent_assets'][f'features/{sid}/visual_{role}.npy']['path'])
    meta=ob[sid]['metadata'][role];nframes=meta['frames']['frame_count'];n_mfcc=meta['audio']['mfcc_shape'][0]
   else:
    meta=fb[sid,role];ap=Path(meta['audio_path']);vp=Path(meta['visual_path']);nframes=meta['frames_meta']['frame_count'];n_mfcc=meta['audio_meta']['mfcc_shape'][0]
   a,v=np.load(ap,allow_pickle=False),np.load(vp,allow_pickle=False);feature_arrays+=2
   check(a.ndim==v.ndim==2 and a.shape[1]==v.shape[1]==1024 and np.isfinite(a).all() and np.isfinite(v).all(),'finite 1024D arrays')
   check(len(v)==nframes-4,'visual complete 5-frame support');check((len(a)-1)*4+20<=n_mfcc,'complete MFCC window support')
   n=min(len(a),len(v));check(n>30 and n==r['counts'][role],'native complete count')
   q=np.arange(15,n-15,dtype=np.int64);j=q[:,None]+np.arange(-15,16,dtype=np.int64)[None,:]
   check(j.min()>=0 and j.max()<n and q.max()+4<nframes,'q+-15 and pixel-frame support no padding')
   delta=np.asarray(v[q,None,:]-a[j]+np.float32(1e-6),dtype=np.float32)
   independent=np.sqrt(np.add.reduce(delta*delta,axis=2,dtype=np.float32),dtype=np.float32).astype(np.float64)
   path=RUN/'distances'/sid/f'{role}_native.npy';filecheck(path);saved=np.load(path,allow_pickle=False);check(saved.shape==independent.shape,'native distance matrix shape')
   err=float(np.max(abs(saved-independent)));max_matrix=max(max_matrix,err);check(err<=1e-5,'NumPy vs Torch raw distance discrepancy')
   condition_matrices[role]=saved;counts[role]=n;arrays[role]=(a,v)
   matrix_checks.append({'sample_id':sid,'role':role,'shape':list(saved.shape),'real_query_min':int(q.min()),'real_query_max':int(q.max()),'audio_index_min':int(j.min()),'audio_index_max':int(j.max()),'numpy_matrix_max_error':err})
   if role in ('N','R'):
    old=ob[sid];old_q=np.arange(old['query_start'],old['query_end']);bridge=summarize_matrix(saved,old_q-15)
    for metric in METRICS:close(bridge[metric],old['cells'][f'V{role}_A{role}'][metric],'parent baseline bridge')
  native={role:np.arange(15,counts[role]-15,dtype=np.int64) for role in ROLES};k=min(len(q) for q in native.values());common=np.arange(15,15+k,dtype=np.int64)
  expected_policies={'common':{role:common for role in ROLES},'native_full':native,
                     'equal_count_spread':{role:q[np.round(np.linspace(0,len(q)-1,k)).astype(np.int64)] for role,q in native.items()}}
  policies={}
  for policy in POLICIES:
   policies[policy]={}
   for role in ROLES:
    q=expected_policies[policy][role];check(len(q)==len(np.unique(q)),'unique spread queries');stored=r['policies'][policy][role]
    check(np.array_equal(q,stored['query_indices']),'policy physical query indices')
    a,v=arrays[role];check(q.min()-15>=0 and q.max()+15<min(len(a),len(v)),'policy true support')
    metric=summarize_matrix(condition_matrices[role],q-15)
    check(metric['query_count']==stored['query_count'] and metric['best_lag_at_boundary']==stored['best_lag_at_boundary'],'query denominator/boundary')
    for key in (*METRICS,'curve'):max_scalar=max(max_scalar,close(metric[key],stored[key],f'{sid} {policy} {role} {key}'))
    policies[policy][role]=metric
  check(r['common_query_count']==k and r['low_common_query_warning']==(k<25),'common warning')
  rebuilt.append({'sample_id':sid,'speaker_id':r['speaker_id'],'policies':policies,'common_queries':k})
 check(feature_arrays==100 and len(rebuilt)==10 and len(matrix_checks)==50,'50 groups / 100 feature arrays')
 print('MATRIX50_POLICY150_NUMPY_PASS',max_matrix,flush=True)
 summaries={};summary_count=0;max_summary=0.
 for policy in POLICIES:
  summaries[policy]={}
  for metric in METRICS:
   values=np.asarray([[r['policies'][policy][role][metric] for role in ROLES] for r in rebuilt],dtype=np.float64)
   t=np.add.reduce(values[:,2:],axis=1)/3
   contrasts={'T-N':t-values[:,0],'T-R':t-values[:,1],'R-N':values[:,1]-values[:,0]};wanted=analysis['summary'][policy][metric];new_contrasts={}
   for name,x in contrasts.items():
    z=bootstrap(x);max_summary=max(max_summary,compare_summary(z,wanted['contrasts'][name],f'{policy}.{metric}.{name}'));new_contrasts[name]=z;summary_count+=1
   per_seed={}
   for index in range(3):
    role=f'T{index+1}';per_seed[role]={}
    for name,baseline_index in [('T-N',0),('T-R',1)]:
     z=bootstrap(values[:,index+2]-values[:,baseline_index]);max_summary=max(max_summary,compare_summary(z,wanted['per_seed'][role][name],f'{policy}.{metric}.{role}.{name}'));per_seed[role][name]=z;summary_count+=1
   for i,role in enumerate(ROLES):close(float(np.mean(values[:,i])),wanted['means'][role],f'condition mean {role}')
   close(float(np.mean(values[:,2:])),wanted['T_mean'],'all repeated T mean')
   summaries[policy][metric]={'contrasts':new_contrasts,'per_seed':per_seed}
 for name in ('T-N','T-R'):check(analysis['two_primary_C_bonferroni_positive'][name]==(summaries['common']['C']['contrasts'][name]['CI97_5'][0]>0),'two-main adjusted interval decision')
 check(summary_count==135,'all135 bootstrap endpoints')
 print('ALL135_BOOTSTRAP_95_AND97_5_PASS',max_summary,flush=True)
 filecheck(RUN/'execution_deviation.json');deviation=read(RUN/'execution_deviation.json')
 return {'status':'PASS_WITH_DISCLOSED_EXECUTION_DEVIATION','verification_scope':'Independent CPU NumPy/ffmpeg readback, no model forwards/GPU, no imports of this experiment pipeline support/metrics/statistics.',
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),
         'analysis_sha256':sha(analysis_path),'parent_seal':{'file_count':seal['file_count'],'total_bytes':parent_bytes,'all_listed_hashes_unchanged':True,'artifact_manifest_sha256':sha(seal_path)},
         'checks':{'canonical30_exact':True,'generated_video30_bindings':True,'new_feature60_hashes':True,'native_groups50_and_feature_arrays100':True,'actual_q_plus_minus15_no_padding':True,'all150_policy_condition_metrics':True,'all135_bootstrap_95_and97_5':True,'all30_T_seeds_retained':True,'equal_spread_unique':True,'parent_N_R_score_bridge':True},
         'numeric_errors':{'numpy_vs_torch_matrix_max':max_matrix,'saved_matrix_curve_scalar_max':max_scalar,'bootstrap_max':max_summary},
         'canonical_audio':audio_checks,'matrix_checks':matrix_checks,'common_query_counts':[r['common_queries'] for r in rebuilt],
         'summary':summaries,'execution_deviation_sha256':sha(RUN/'execution_deviation.json'),
         'boundaries':['One TTS draw reached token cap without EOS; all30 preserved per frozen protocol, no rerun or selection.',
                       'The overlapping codec diagnostic process is disclosed separately; numeric readback does not prove no influence on stochastic TTS.',
                       'Official full ICL T includes reference decoder context. T-R is a full generative-path comparison, not isolated language-model learning-objective causality.',
                       'Common query count/positions, native full and equal-count spread are distinct supports; they do not align phonemes across independent utterances.',
                       'SyncNet endpoints are not blinded perceptual or true mouth-accuracy evidence. Ten historically exposed speakers; no new independent confirmation cohort.'],
         'checked_files':files,'errors':errors}

if __name__=='__main__':
 destination=RUN/'independent_downstream_audit.json';check(not destination.exists(),'audit output already exists')
 try:result=main()
 except BaseException as exc:
  result={'status':'FAILED','errors':errors,'exception':repr(exc),'traceback':traceback.format_exc(),'checked_files':files}
  with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
  raise
 with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print('INDEPENDENT_DOWNSTREAM_AUDIT_SHA',sha(destination),destination.stat().st_size,flush=True)
