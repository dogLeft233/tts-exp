"""CPU-only frozen-front-end saturation partition feasibility; no model or scores."""
from pathlib import Path
import hashlib, inspect, json, shutil, sys, time, subprocess
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_fixed_mel_partition_feasibility_20260927'
CAL=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
N80=ROOT/'runs/tts_fixed_natural_holdout80_20260927'
sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'))
import audio
import librosa
from scipy import signal
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):
 Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def stages(x):
 pre=audio.preemphasis(x,audio.hp.preemphasis,audio.hp.preemphasize)
 stft=audio._stft(pre);mag=audio._linear_to_mel(abs(stft));db=audio._amp_to_db(mag)-audio.hp.ref_level_db
 un=8*((db+100)/100)-4;out=np.clip(un,-4,4)
 assert np.array_equal(out,audio.melspectrogram(x))
 return pre,stft,mag,db,un,out
def counts(x,un,mag):
 return dict(cells=x.size,amplitude_floor=int((mag<=1e-5).sum()),lower_clamp_internal=int((un<=-4).sum()),upper_clamp_internal=int((un>=4).sum()),lower_endpoint=int((x==-4).sum()),upper_endpoint=int((x==4).sum()),interior_rounds_to_endpoint=int(((abs(un)<4)&(abs(x)==4)).sum()),one_ULP_interior_boundary=int(((abs(x)<4)&(abs(x)>=np.nextafter(np.float32(4),np.float32(0)))).sum()))
def describe(v):
 v=np.asarray(v,dtype=float);return {'min':float(v.min()),'median':float(np.median(v)),'max':float(v.max()),'mean':float(v.mean())}
def freeze():
 assert not OUT.exists();OUT.mkdir();p=read(CAL/'protocol.json');deps={f:h for f,h in p['dependencies'].items() if any(k in f for k in ['Wav2Lip/audio.py','Wav2Lip/hparams.py','static_image_bridge/render_worker.py'])}
 for f in [CAL/'protocol.json',CAL/'support.json',CAL/'gains.json',N80/'protocol.json',N80/'support.json',N80/'feature_seal.json'] :deps[str(f)]=sha(f)
 for f,h in deps.items():assert sha(f)==h
 protocol={'status':'frozen_before_CPU_quantification_no_new_generation_or_score','time':time.time(),'population':'all26 historical calibration pairs N/T only; natural80 only binding/resource metadata','partition':'on saved float32 M0,M1: U=(-4<M0<4)&(-4<M1<4); S=~U; exact endpoint test, no fitted tolerance. One inward float32 ULP boundary counted separately, never reassign samples.','conditions':{'00':'M0 RAW','10':'where(U,M1,M0)','01':'where(S,M1,M0)','11':'M1 actual FIXED'},'bounds':'all intermediate cells copied from legal endpoints; exact float64 difference closure; no uniform-shift approximation substituted','dependencies':deps,'code_sha256':sha(__file__),'resource':'CPU only, reserve5GiB, audit budget8MiB; no neural forwards or scores, no new waveform/video files'}
 write(OUT/'protocol.json',protocol);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n');shutil.copyfile(__file__,OUT/Path(__file__).name)
 metadata={'hparams':audio.hp.data,'audio_sha256':sha(audio.__file__),'hparams_sha256':sha(ROOT/'third_party/Wav2Lip/hparams.py'),'numpy':np.__version__,'librosa':librosa.__version__,'soundfile':sf.__version__,'stft_signature':str(inspect.signature(librosa.stft)),'mel_signature':str(inspect.signature(librosa.filters.mel)),'source_hashes':{inspect.getsourcefile(f):sha(inspect.getsourcefile(f)) for f in [librosa.stft,librosa.filters.mel,signal.lfilter]},'lfilter_signature':str(inspect.signature(signal.lfilter)),'actual_function_source':{f.__name__:inspect.getsource(f) for f in [audio.preemphasis,audio._stft,audio._build_mel_basis,audio._amp_to_db,audio._normalize,audio.melspectrogram]}}
 write(OUT/'frontend_implementation.json',metadata)
def audit():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip() and sha(__file__)==p['code_sha256']
 for f,h in p['dependencies'].items():assert sha(f)==h
 rows=[];inputs={};gains={(r['id'],r['arm']):r['gain'] for r in read(CAL/'gains.json')};cal=[r for r in read(CAL/'support.json') if r['split']=='calibration'];assert len(cal)==26
 for r in cal:
  for arm in ['N','T']:
   z=r['arms'][arm];st=[];M=[]
   for condition in ['raw','FIXED']:
    a=z[condition]
    for key,hkey in [('waveform','sha256'),('frontend','frontend_sha256')]:assert sha(a[key])==a[hkey];inputs[a[key]]=a[hkey]
    x=sf.read(a['waveform'],dtype='float32')[0];s=stages(x);m=np.load(a['frontend'])['mel'];assert np.array_equal(s[-1].astype(np.float32),m);st.append(s);M.append(m)
   a,b=M;g=gains[r['id'],arm];delta=1.6*np.log10(g);U=(a>-4)&(a<4)&(b>-4)&(b<4);S=~U;diff=b.astype(float)-a.astype(float);du=np.where(U,diff,0);ds=np.where(S,diff,0);mu=np.where(U,b,a);ms=np.where(S,b,a)
   assert np.array_equal(du+ds,diff) and np.array_equal(mu.astype(float)+ms.astype(float)-a.astype(float),b.astype(float)) and np.isfinite(mu).all() and np.isfinite(ms).all() and max(abs(mu).max(),abs(ms).max())<=4
   ideal=audio.melspectrogram(sf.read(z['raw']['waveform'],dtype='float64')[0]*g).astype(np.float32)
   scalarclip=np.clip(a.astype(float)+delta,-4,4).astype(np.float32)
   row={'id':r['id'],'speaker':r['speaker'],'arm':arm,'gain':g,'gain_dB':20*np.log10(g),'predicted_interior_delta':delta,'shape':list(a.shape),'RAW':counts(a,st[0][4],st[0][2]),'FIXED':counts(b,st[1][4],st[1][2]),'U_count':int(U.sum()),'S_count':int(S.sum()),'S_changed_count':int((S&(diff!=0)).sum()),'lower_unlock_count':int(((a==-4)&(b>-4)).sum()),'lower_lock_count':int(((a>-4)&(b==-4)).sum()),'upper_cross_count':int(((a==4)^(b==4)).sum()),'delta_abs_sum_U':float(abs(du).sum()),'delta_abs_sum_S':float(abs(ds).sum()),'delta_sq_sum_U':float((du**2).sum()),'delta_sq_sum_S':float((ds**2).sum()),'U_delta':describe(diff[U]),'S_nonzero_delta':describe(diff[S&(diff!=0)]) if np.any(S&(diff!=0)) else None,'U_scalar_prediction_max_error':float(abs(diff[U]-delta).max()),'float32_waveform_vs_ideal_gain_mel_max':float(abs(ideal.astype(float)-b).max()),'uniform_postmel_shift_vs_actual_max':float(abs(scalarclip.astype(float)-b).max()),'uniform_postmel_shift_error_U_max':float(abs(scalarclip.astype(float)[U]-b[U]).max()),'exact_difference_closure':True,'all_four_legal':True,'array_hashes':{'M0':arrsha(a),'M1':arrsha(b),'U_mask':arrsha(U),'U_only':arrsha(mu),'S_only':arrsha(ms)},'dtypes':{'pre':str(st[0][0].dtype),'STFT':str(st[0][1].dtype),'mag':str(st[0][2].dtype),'normalized_internal':str(st[0][-1].dtype),'saved':str(a.dtype)}}
   rows.append(row);print('audited',r['id'],arm,flush=True)
 write(OUT/'calibration_rows.json',rows);write(OUT/'input_hashes.json',inputs)
 summary={}
 for arm in ['N','T','pooled']:
  q=[r for r in rows if arm=='pooled' or r['arm']==arm];n=sum(r['RAW']['cells'] for r in q);parts={k:sum(r[k] for r in q) for k in ['U_count','S_count','S_changed_count','lower_unlock_count','lower_lock_count','upper_cross_count','delta_abs_sum_U','delta_abs_sum_S','delta_sq_sum_U','delta_sq_sum_S']}
  summary[arm]={'utterances':len(q),'cells':n,'RAW_totals':{k:sum(r['RAW'][k] for r in q) for k in q[0]['RAW']},'FIXED_totals':{k:sum(r['FIXED'][k] for r in q) for k in q[0]['FIXED']},**parts,'U_fraction':parts['U_count']/n,'S_fraction':parts['S_count']/n,'S_changed_fraction':parts['S_changed_count']/n,'S_abs_delta_fraction':parts['delta_abs_sum_S']/(parts['delta_abs_sum_U']+parts['delta_abs_sum_S']),'S_sq_delta_fraction':parts['delta_sq_sum_S']/(parts['delta_sq_sum_U']+parts['delta_sq_sum_S']),'gain_dB':describe([r['gain_dB'] for r in q]),'max_U_scalar_error':max(r['U_scalar_prediction_max_error'] for r in q),'max_waveform_float32_mel_error':max(r['float32_waveform_vs_ideal_gain_mel_max'] for r in q),'max_uniform_shift_error':max(r['uniform_postmel_shift_vs_actual_max'] for r in q),'gain_up':sum(r['gain']>1 for r in q),'gain_down':sum(r['gain']<1 for r in q)}
 write(OUT/'calibration_summary.json',summary)
def budget():
 # Shape/file metadata only. Never load scores or compute encoder norms/outcomes.
 p=read(OUT/'protocol.json');resources={};n80=read(N80/'support.json');cal=[r for r in read(CAL/'support.json') if r['split']=='calibration'];bindings={}
 populations={'cal26_N':[(r['id'],r['arms']['N']['raw'],r['arms']['N']['FIXED'],r['arms']['N']['joint_L'],CAL/'features'/r['id']/'N') for r in cal], 'natural80':[(r['id'],r['source'],r['FIXED'],r['L'],N80/'features'/r['id']) for r in n80]}
 for name,rr in populations.items():
  melbytes=visualbytes=matrixbytes=fullvideo=0;maxrawtemp=0;first2rawtemp=0
  for i,(sid,raw,fixed,L,fp) in enumerate(rr):
   front=Path(fixed['frontend']);assert sha(front)==fixed['frontend_sha256'];bindings[str(front)]=sha(front);z=np.load(front);melbytes+=z['mel'].nbytes*2
   for c in (['raw','FIXED'] if name=='cal26_N' else ['RAW','FIXED']):
    meta=read(fp/(c+'.json'));feature=Path(meta['features']);assert sha(feature)==meta['sha256'];bindings[str(feature)]=meta['sha256'];bindings[str(fp/(c+'.json'))]=sha(fp/(c+'.json'))
    if c=='FIXED':
     v=np.load(feature)['visual'];visualbytes+=2*v.nbytes;frames=meta['video']['frames'];fullvideo+=2*frames*224*224*3;maxrawtemp=max(maxrawtemp,frames*224*224*3)
     if i<2:first2rawtemp+=2*frames*224*224*3
   matrixbytes+=8*4*2*L*31 # 4 generated mel conditions x raw/unit, float64 conservative
  # No permanent videos recommended; first2 replays streamed and sealed, source baselines already retained.
  fixed_allowance=24*2**20;total=melbytes+visualbytes+matrixbytes+fixed_allowance+maxrawtemp
  resources[name]={'utterances':len(rr),'new_conditions':2,'two_mel_arrays_uncompressed_bytes':melbytes,'two_visual_arrays_uncompressed_bytes':visualbytes,'all_four_raw_unit_float64_distances_bytes':matrixbytes,'metadata_curve_NPZ_control_margin_bytes':fixed_allowance,'one_rawvideo_sized_temp_allowance_bytes':maxrawtemp,'estimated_peak_bytes':total,'estimated_peak_GiB':total/2**30,'all_new_video_uncompressed_bytes':fullvideo,'first2_each_newcondition_rawvideo_bytes':first2rawtemp,'limit_statement':'Exact shape-derived components; temporary encoded-file size not guaranteed by raw-pixel allowance. Future runner must enforce total budget before each batch and stop at5GiB, no old deletion.'}
 free=shutil.disk_usage(OUT).free;write(OUT/'natural80_binding_and_budget_hashes.json',bindings);write(OUT/'resources.json',{'time':time.time(),'free_bytes':free,'free_GiB':free/2**30,'headroom_above_5GiB':free-5*2**30,'populations':resources,'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),'GPU_inference':0,'new_scores':0,'cloud_requests':0,'audit_allocated_bytes':sum(f.stat().st_blocks*512 for f in OUT.rglob('*') if f.is_file())});assert free>=5*2**30
if __name__=='__main__':globals()[sys.argv[1]]()
