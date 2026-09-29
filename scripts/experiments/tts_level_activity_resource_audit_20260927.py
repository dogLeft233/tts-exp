"""Frozen CPU activity/level decomposition, cal-only candidate target, resource audit."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_level_activity_resource_audit_20260927'
LEVEL=ROOT/'runs/tts_native_level_generation_cross_20260927'
SOURCE=ROOT/'runs/aishell1_qwen_mfa_linear_n100_20260816'
MFA=SOURCE/'02_mfa_mandarin341_ready/mfa_summary.json'
METHODS=['MFA','MFA_unknown_active','block_-30dB','block_-20dB','block_-40dB']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stats(v,s):
 if any(x is None for x in v):return {'status':'undefined_retained','n':len(v),'undefined_count':sum(x is None for x in v)}
 groups=sorted(set(s));means=np.array([np.mean([x for x,g in zip(v,s) if g==name]) for name in groups]);ix=np.random.Generator(np.random.PCG64(20260926)).integers(0,len(groups),(20000,len(groups)));boot=means[ix].mean(1)
 return {'status':'defined','n':len(v),'speakers':len(groups),'mean':float(means.mean()),'ci99':np.quantile(boot,[.005,.995]).tolist(),'group_means':dict(zip(groups,means.tolist()))}
def freeze():
 assert not (OUT/'protocol.json').exists();parent=read(LEVEL/'protocol.json');mfa=read(MFA);bindings=[]
 for row in parent['rows']:
  for a,label in [('N','natural'),('T','tts')]:
   source=row['audio'][a];assert sha(source['path'])==source['sha256'];grid=mfa['records'].get(row['id'],{}).get(label)
   ok=bool(grid and grid.get('textgrid') and Path(grid['textgrid']).exists() and grid.get('audio_sha256')==source['sha256'])
   if ok:assert sha(grid['textgrid'])==grid['textgrid_sha256']
   bindings.append({'id':row['id'],'arm':a,'available':ok,'textgrid':grid.get('textgrid') if grid else None,'textgrid_sha256':grid.get('textgrid_sha256') if grid else None})
 files=[Path(__file__),ROOT/'scripts/experiments/check_tts_level_activity_resource_audit_20260927.py']
 p={'status':'frozen_before_descriptions','created_epoch':time.time(),'scope':'CPU only original100 pairs; no GPU rendering, new TTS/API requests or scores/fitting; fixed target is candidate requiring root separate approval before intervention',
 'rows':parent['rows'],'methods':METHODS,'MFA':'Frozen token is_silence partitions speech/non-speech; sample centers [start,end); uncovered samples assigned inactive solely for exhaustive identity and coverage disclosed; unknown spn reported; MFA_unknown_active sensitivity marks all unknown as active',
 'blocks':'20ms=320 samples nonoverlapping from sample0, last incomplete block retained; active RMS > max blockRMS *10^(threshold/20), thresholds -30 primary and -20/-40 sensitivity; allzero yields no activity; sample weighting exact',
 'identity':'globalMS=p*activeMS+(1-p)*inactiveMS; log(globalRMS)=log(activeRMS)+.5log(p)+.5log(1+Einactive/Eactive); zero active energy undefined, retain rows; report log and dB=20/ln10*log',
 'gains':'Descriptive active pair midpoint sqrt(activeRMS_N*activeRMS_T)/activeRMS_arm without clip adjustment; report differences vs frozen LEVEL gains and hypothetical peaks. No waveform produced.',
 'target':'exp(median(log(globalRMS)) over all52cal arms), pooled52 cal only. Freeze estimate before eval feasibility. Never choose/adjust on eval. No clipping correction in candidate; report peak threshold .98 and1.',
 'statistics':'cal26/eval74 separately, speaker equal,20000 bootstrap seed20260926,99CI; all methods and all three terms reported,no FWER; undefined labels retained, no effects-based selection',
 'tolerances':{'MS_identity':1e-14,'log_identity':1e-12,'gain_mask_invariance':'exact for gains0.125 and8','independent_numeric':1e-10,'statistics':1e-10},
 'mfa_bindings':bindings,'dependencies':{str(p):sha(p) for p in [LEVEL/'protocol.json',LEVEL/'pairs.json',MFA]},'code':{str(f):sha(f) for f in files}}
 write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
 for f in files:
  dest=OUT/'code_snapshot'/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
 print('FROZEN',sha(OUT/'protocol.json'))
def block_mask(x,threshold):
 lengths=np.minimum(320,len(x)-np.arange(0,len(x),320));energy=np.add.reduceat(x*x,np.arange(0,len(x),320));rms=np.sqrt(energy/lengths)
 active=rms>float(rms.max())*10**(threshold/20) if rms.max()>0 else np.zeros(len(rms),dtype=bool)
 return np.repeat(active,lengths),{'blocks':len(rms),'last_block_samples':int(lengths[-1]),'max_block_rms':float(rms.max()),'active_blocks':int(active.sum())}
def mfa_masks(x,tokens):
 n=len(x);active=np.zeros(n,dtype=bool);unknown=np.zeros(n,dtype=bool);cover=np.zeros(n,dtype=np.int16)
 for t in tokens:
  lo=max(0,min(n,int(np.ceil(t['start_s']*16000-.5))));hi=max(0,min(n,int(np.ceil(t['end_s']*16000-.5))))
  cover[lo:hi]+=1
  if not t['is_silence']:active[lo:hi]=True
  if t.get('is_unknown'):unknown[lo:hi]=True
 return active,unknown,{'covered_samples':int(np.sum(cover>0)),'uncovered_samples':int(np.sum(cover==0)),'overlap_samples':int(np.sum(cover>1)),'unknown_samples':int(unknown.sum()),'unknown_energy_fraction':float(np.sum(x[unknown]**2)/np.sum(x*x)) if np.any(x) else None}
def decompose(x,mask):
 n=len(x);na=int(mask.sum());ni=n-na;ea=float(np.sum(x[mask]**2));ei=float(np.sum(x[~mask]**2));ms=float(np.mean(x*x));pa=na/n;ams=ea/na if na else None;ims=ei/ni if ni else 0.
 reconstructed=pa*(ams or 0)+(1-pa)*ims;defined=bool(na and ea>0 and ms>0)
 logs={'activity_level':float(np.log(np.sqrt(ams))) if defined else None,'occupancy':float(.5*np.log(pa)) if defined else None,'inactive_correction':float(.5*np.log1p(ei/ea)) if defined else None,'global':float(.5*np.log(ms)) if ms>0 else None}
 return {'samples':n,'active_samples':na,'inactive_samples':ni,'p':pa,'active_energy':ea,'inactive_energy':ei,'global_MS':ms,'active_MS':ams,'inactive_MS':ims,'global_RMS':float(np.sqrt(ms)),'active_RMS':float(np.sqrt(ams)) if ams is not None else None,'active_energy_fraction':ea/(ea+ei) if ea+ei else None,'log_terms':logs,'dB_terms':{k:20/np.log(10)*v if v is not None else None for k,v in logs.items()},'defined':defined,'MS_identity_error':abs(ms-reconstructed),'log_identity_error':abs(logs['global']-sum(logs[k] for k in ['activity_level','occupancy','inactive_correction'])) if defined else None}
def run():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in {**p['dependencies'],**p['code']}.items():assert sha(f)==h
 mfa=read(MFA);old={r['id']:r for r in read(LEVEL/'pairs.json')};metrics=[];pairs=[];waves={};coverage=[];gain_invariance=True
 for row in p['rows']:
  for a in ['N','T']:
   source=row['audio'][a];assert sha(source['path'])==source['sha256'];x,sr=sf.read(source['path'],dtype='float64');assert sr==16000 and x.ndim==1;waves[row['id'],a]=x
 # Only calibration values are used to compute the candidate; persist before examining eval feasibility.
 vals=[np.log(np.sqrt(np.mean(waves[r['id'],a]**2))) for r in p['rows'] if r['split']=='calibration' for a in ['N','T']];target=float(np.exp(np.median(vals)))
 write(OUT/'fixed_target.json',{'status':'candidate_not_intervention_authorized','calibration_arms':len(vals),'target_RMS':target,'target_dBFS':float(20*np.log10(target)),'estimator':'exp median log RMS over52cal arms','eval_used_for_estimator':False})
 fixed=[]
 for row in p['rows']:
  sid=row['id'];local={}
  for a,label in [('N','natural'),('T','tts')]:
   x=waves[sid,a];binding=next(r for r in p['mfa_bindings'] if r['id']==sid and r['arm']==a);masks={};meta={}
   if binding['available']:
    assert sha(binding['textgrid'])==binding['textgrid_sha256'];active,unknown,cov=mfa_masks(x,mfa['records'][sid][label]['tokens']);masks['MFA']=active;masks['MFA_unknown_active']=active|unknown;coverage.append({'id':sid,'arm':a,'split':row['split'],'available':True,**cov})
   else:coverage.append({'id':sid,'arm':a,'split':row['split'],'available':False})
   for threshold in [-30,-20,-40]:
    method=f'block_{threshold}dB';masks[method],meta[method]=block_mask(x,threshold)
    for gain in [.125,8.]:gain_invariance &= bool(np.array_equal(masks[method],block_mask(x*gain,threshold)[0]))
   maskpath=OUT/'masks'/sid/(a+'.npz');maskpath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(maskpath,**masks)
   for method in METHODS:
    if method not in masks:continue
    d=decompose(x,masks[method]);local[a,method]=d;metrics.append({'id':sid,'arm':a,'speaker':row['speaker'],'split':row['split'],'method':method,**d,'block_meta':meta.get(method),'mask_path':str(maskpath),'mask_sha256':sha(maskpath)})
   r=float(np.sqrt(np.mean(x*x)));g=target/r;peak=float(abs(x).max());fixed.append({'id':sid,'arm':a,'speaker':row['speaker'],'split':row['split'],'global_RMS':r,'original_peak':peak,'fixed_gain':g,'fixed_gain_dB':float(20*np.log10(g)),'paired_LEVEL_gain':old[sid]['gain'][a],'gain_dB_difference':float(20*np.log10(g/old[sid]['gain'][a])),'hypothetical_peak':peak*g,'exceeds_098':peak*g>.98,'exceeds_1':peak*g>=1})
  for method in METHODS:
   if ('N',method) not in local or ('T',method) not in local:continue
   n,t=local['N',method],local['T',method];defined=n['defined'] and t['defined'];deltas={k:t['dB_terms'][k]-n['dB_terms'][k] if defined else None for k in n['dB_terms']};ga={}
   if defined:
    m=float(np.sqrt(n['active_RMS']*t['active_RMS']))
    for a,values in [('N',n),('T',t)]:
     g=m/values['active_RMS'];ga[a]={'active_match_gain':g,'active_match_gain_dB':float(20*np.log10(g)),'difference_from_LEVEL_dB':float(20*np.log10(g/old[sid]['gain'][a])),'hypothetical_peak':float(abs(waves[sid,a]).max()*g)}
   pairs.append({'id':sid,'speaker':row['speaker'],'split':row['split'],'method':method,'defined':defined,'delta_dB':deltas,'active_gain':ga,'p_N':n['p'],'p_T':t['p'],'active_RMS_N':n['active_RMS'],'active_RMS_T':t['active_RMS']})
 summary={}
 for split in ['calibration','evaluation']:
  summary[split]={}
  for method in METHODS:
   pp=[r for r in pairs if r['split']==split and r['method']==method];speakers=[r['speaker'] for r in pp];mm=[r for r in metrics if r['split']==split and r['method']==method]
   summary[split][method]={'pairs':len(pp),'undefined_ids':[r['id'] for r in pp if not r['defined']],'delta_dB':{k:stats([r['delta_dB'][k] for r in pp],speakers) for k in ['global','activity_level','occupancy','inactive_correction']},'gain_difference_from_LEVEL_dB':{a:stats([r['active_gain'][a]['difference_from_LEVEL_dB'] if r['defined'] else None for r in pp],speakers) for a in ['N','T']},'active_gain_peak_ge1':sum(v['hypothetical_peak']>=1 for r in pp for v in r['active_gain'].values()),'arms':{a:{key:stats([r[key] for r in mm if r['arm']==a],[r['speaker'] for r in mm if r['arm']==a]) for key in ['p','active_RMS','active_energy_fraction']} for a in ['N','T']}}
 fs={split:{'arms':len(rr:= [r for r in fixed if r['split']==split]),'peak_max':max(r['hypothetical_peak'] for r in rr),'exceeds_098':sum(r['exceeds_098'] for r in rr),'exceeds_1':sum(r['exceeds_1'] for r in rr),'max_safe_global_target_098':min(.98*r['global_RMS']/r['original_peak'] for r in rr),'gain_diff':{a:stats([r['gain_dB_difference'] for r in rr if r['arm']==a],[r['speaker'] for r in rr if r['arm']==a]) for a in ['N','T']}} for split in ['calibration','evaluation']}
 errors={'MS_identity':max(r['MS_identity_error'] for r in metrics),'log_identity':max(r['log_identity_error'] for r in metrics if r['defined']),'gain_masks_exact':gain_invariance}
 assert errors['MS_identity']<=1e-14 and errors['log_identity']<=1e-12 and gain_invariance
 for name,value in [('metrics',metrics),('pairs',pairs),('coverage',coverage),('fixed_target_feasibility',fixed),('summary',{'status':'PASS','results':summary,'fixed_target':fs,'errors':errors})]:write(OUT/(name+'.json'),value)
 print(json.dumps({'status':'PASS','target':target,'errors':errors,'fixed_target':fs},indent=2))
def resources():
 # Never print credential values or entire configuration/metadata objects.
 meta=read(SOURCE/'01_tts_retry/tts_meta.json');keynames=['DASHSCOPE_API_KEY','SILICONFLOW_API_KEY','OPENAI_API_KEY'];envfiles=[ROOT/'.env',ROOT/'.env.local',Path.home()/'.env']
 configs={'historical_tts_meta_exists':True,'historical_provider_field_exists':bool(meta.get('provider')),'historical_model_field_exists':bool(meta.get('model')),'current_config_exists':(ROOT/'scripts/config.yaml').exists(),'environment_key_presence':{k:bool(os.environ.get(k)) for k in keynames},'dotenv_presence':{str(p):p.exists() for p in envfiles},'dotenv_key_presence':{}}
 for path in envfiles:
  if path.is_file():
   text=path.read_text();configs['dotenv_key_presence'][str(path)]={k:any(line.strip().removeprefix('export ').startswith(k+'=') and bool(line.split('=',1)[1].strip().strip('\"\'')) for line in text.splitlines()) for k in keynames}
 def sizes(paths):
  logical=allocated=0;seen=set();count=0
  for f in paths:
   if not f.is_file():continue
   st=f.stat();count+=1;logical+=st.st_size
   if (st.st_dev,st.st_ino) not in seen:allocated+=st.st_blocks*512;seen.add((st.st_dev,st.st_ino))
  return {'files':count,'logical_bytes':logical,'unique_allocated_bytes':allocated}
 cost={folder:sizes((LEVEL/folder).rglob('*')) for folder in ['videos','features','audio','frontend','scores','frontend_description']}
 conditions={c:{kind:sizes((LEVEL/kind).rglob(c+ext)) for kind,ext in [('videos','.avi'),('features','.npz'),('audio','.wav'),('frontend','.npz')]} for c in ['LEVEL','EQ_LEVEL']}
 mounts=[]
 for line in Path('/proc/mounts').read_text().splitlines():
  dev,mp,typ=line.split()[:3];mp=mp.replace('\\040',' ')
  if mp=='/' or mp=='/boot/efi' or (typ not in ['tmpfs','proc','sysfs','devtmpfs','devpts','cgroup2','securityfs','pstore','debugfs','tracefs','fusectl','configfs','mqueue','hugetlbfs','bpf','autofs','squashfs','efivarfs'] and not mp.startswith('/snap')):
   try:d=shutil.disk_usage(mp);mounts.append({'device':dev,'mount':mp,'type':typ,'total_bytes':d.total,'free_bytes':d.free,'writable_permission':os.access(mp,os.W_OK),'write_test_performed':False})
   except OSError:pass
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
 percondition=np.mean([sum(v['logical_bytes'] for v in d.values()) for d in conditions.values()]);free=shutil.disk_usage(ROOT).free
 r={'created_epoch':time.time(),'gpu':gpu,'compute':compute,'mounts':mounts,'provider_presence_only':configs,'current_level_run_cost':cost,'new_condition_cost':conditions,'mean_one_new_condition_200_arms_bytes':float(percondition),'planned_one_condition_plus15pct_bytes':float(percondition*1.15),'free_local_bytes':free,'reserve_bytes':5<<30,'can_add_one_condition_keep5GiB':bool(free-percondition*1.15>=5<<30),'network_mount_note':'CloudFS is not a local scratch disk; quota/free stats only, no upload/write/performance/durability validation; boot EFI not experiment storage','api_requests':0,'GPU_computation':0}
 write(OUT/'resources.json',r);print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run','resources']);globals()[ap.parse_args().stage]()
