"""A-only native-phone MFCC-shape reversal, frozen inputs and blinded scoring."""
from pathlib import Path
from decimal import Decimal,ROUND_CEILING
from contextlib import contextmanager
import argparse,fcntl,hashlib,json,math,os,platform,re,resource,shutil,subprocess,sys,time,types
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_fixed_mfcc_phase_reversal_20260927';OLD=ROOT/'runs/tts_fixed_level_generation_cross_20260927';FEAS=ROOT/'runs/tts_phone_context_feasibility_20260927';PY=ROOT/'.venv/bin/python'
MFA=ROOT/'runs/aishell1_qwen_mfa_linear_n100_20260816/02_mfa_mandarin341_ready/mfa_summary.json';EVENT=ROOT/'runs/tts_level_event_cross_20260927/indices.json';CALSUP=ROOT/'runs/tts_shared_translation_20260926/calibration_support.json'
BASE=ROOT/'runs/tts_acoustic_generation_cross_20260926/code_snapshot/tts_acoustic_generation_cross_20260926.py';b=types.ModuleType('sealed_phase_base');b.__file__=str(ROOT/'scripts/experiments/tts_acoustic_generation_cross_20260926.py');exec(compile(BASE.read_text(),str(BASE),'exec'),b.__dict__)
CAP=100*2**20;OTHER=92*2**20;FLOOR=int(4.5*2**30);EXCLUDED={'','sil','sp','spn','<unk>','<eps>','silence'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(2**20),b''):h.update(z)
 return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def allocated(p):return sum(f.stat().st_blocks*512 for f in p.rglob('*') if f.is_file()) if p.exists() else 0
def limits(extra=0,feature=False):
 used=allocated(OUT);free=shutil.disk_usage(OUT).free;meta=used-allocated(OUT/'features')
 assert used+extra<=CAP and free-max(0,CAP-used)-OTHER>=FLOOR,('resource',used,extra,free)
 if not feature:assert meta+extra<=10*2**20,('metadata_cap',meta,extra)
 av=next(int(q.split()[1])*1024 for q in Path('/proc/meminfo').read_text().splitlines() if q.startswith('MemAvailable:'));assert av>=2**30
 return {'time':time.time(),'allocated_bytes':used,'free_bytes':free,'remaining_commitment':CAP-used,'other_reserved':OTHER,'floor':FLOOR,'MemAvailable':av,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def write(p,z):
 s=json.dumps(z,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n';limits(math.ceil(len(s.encode())/4096)*4096);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(s)
 limits()
def save(p,x):
 assert x.dtype==np.float32 and x.ndim==2 and x.shape[1]==1024 and np.isfinite(x).all();n=math.ceil((x.nbytes+128)/4096)*4096;limits(n,True);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:np.save(f,x,allow_pickle=False)
 assert p.stat().st_blocks*512<=n;limits();return {'path':str(p),'sha256':sha(p),'shape':list(x.shape),'dtype':'float32','raw_sha256':ah(x)}
def runtime():
 import scipy,cv2,torch
 return {'python':platform.python_version(),'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'soundfile':sf.__version__,'opencv':cv2.__version__,'torch':torch.__version__,'cuda':torch.version.cuda}
def phones(path):
 text=Path(path).read_text();tier=text.split('name = "phones"',1)[1].split('item [',1)[0];items=re.findall(r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',tier);out=[];previous=Decimal(0)
 for i,(lo,hi,label) in enumerate(items):
  a,c=Decimal(lo),Decimal(hi);assert 0<=a<=c and a>=previous;previous=c
  out.append({'tier_index':i,'label':label,'lo_decimal':lo,'hi_decimal':hi,'lo_sample':int((a*16000).to_integral_value(rounding=ROUND_CEILING)),'hi_sample':int((c*16000).to_integral_value(rounding=ROUND_CEILING)),'usable':bool(c>a and label.strip().casefold() not in EXCLUDED)})
 assert out;return out
def windows(mf,n):return np.stack([mf[4*j:4*j+20].T for j in range(n)]).astype(np.float32)
def operation(mf,tokens,samples):
 start=np.maximum(0,np.arange(len(mf))*160-1);end=np.arange(len(mf))*160+400;groups=[];mask=np.zeros(len(mf),bool);rev=np.arange(len(mf),dtype=np.int64)
 for t in tokens:
  if not t['usable']:continue
  ix=np.flatnonzero((start>=t['lo_sample'])&(end<=t['hi_sample'])&(end<=samples))
  if len(ix)<2:continue
  assert not mask[ix].any();mask[ix]=True;rev[ix]=ix[::-1];groups.append({'tier_index':t['tier_index'],'label':t['label'],'frames':ix.tolist()})
 y=mf.copy();y[:,1:]=mf[rev,1:];back=y.copy();back[:,1:]=y[rev,1:]
 assert np.array_equal(back,mf) and np.array_equal(y[:,0],mf[:,0]) and np.array_equal(y[~mask],mf[~mask]);assert np.array_equal(rev[rev],np.arange(len(mf)))
 return y,back,{'groups':groups,'reverse_source_rows':rev.tolist(),'target_frames':int(mask.sum()),'actual_changed_frames':int(np.any(y!=mf,axis=1).sum()),'total_MFCC_frames':len(mf),'L2_delta_shape':float(np.linalg.norm(y[:,1:]-mf[:,1:])),'RMS_delta_shape':float(np.sqrt(np.mean((y[:,1:]-mf[:,1:])**2))),'C0_exact':True,'unmarked_exact':True,'inverse_exact':True,'original_MFCC_sha256':ah(mf),'reversed_MFCC_sha256':ah(y),'reversal_index_sha256':ah(rev),'unmodified_waveform':True}
def protocol():
 z=read(OUT/'reviewer_pass.json');assert z['status']=='PASS' and sha(z['receipt'])==z['receipt_sha256'] and z['protocol_sha256']==sha(OUT/'protocol.json')
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256'];assert runtime()==p['runtime'];assert Path(sys.executable).resolve()==PY.resolve()
 for f,h in {**p['dependencies'],**p['asset_hashes']}.items():assert sha(f)==h,f
 assert sha(OUT/'event_indices.json')==p['event_indices_sha256']
 for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(name)=='1',name
 return p
def construct(row,arm):
 z=row['arms'][arm];assert sha(z['waveform'])==z['waveform_sha256'];x,sr=sf.read(z['waveform'],dtype='float64');assert sr==16000 and len(x)==z['samples'];import python_speech_features
 mf=np.asarray(python_speech_features.mfcc(x*32768,16000),np.float64);assert np.isfinite(mf).all();orig=windows(mf,z['audio_windows']);assert sha(z['frontend'])==z['frontend_sha256'];cache=np.load(z['frontend']);assert np.array_equal(orig,cache['windows'])
 tokens=z['tokens'];y,back,info=operation(mf,tokens,len(x));new=windows(y,z['audio_windows']);inverse=windows(back,z['audio_windows']);assert np.isfinite(new).all() and np.array_equal(inverse,orig) and np.array_equal(new[:,0],orig[:,0])
 coverage=np.array([sum(s!=i for i,s in enumerate(info['reverse_source_rows'][4*j:4*j+20],start=4*j)) for j in range(z['audio_windows'])]);info.update({'id':row['id'],'arm':arm,'split':row['split'],'audio_window_starts':[4*j for j in range(z['audio_windows'])],'window_changed_column_counts':coverage.tolist(),'original_windows_sha256':ah(orig),'new_windows_sha256':ah(new),'samples':len(x),'baseline_frontend_exact':True,'all_finite':True,'annotation_status':z['annotation_status']});return orig,new,inverse,info
def freeze():
 assert not (OUT/'protocol.json').exists();old=read(OLD/'protocol.json');sp={r['id']:r for r in read(OLD/'support.json')};mfa=read(MFA)['records'];rows=[];bind={};alloc=0
 for r in old['rows']:
  q={'id':r['id'],'speaker':r['speaker'],'split':r['split'],'guard20_eligible':sp[r['id']]['guard20_eligible'],'arms':{}}
  for a,label in [('N','natural'),('T','tts')]:
   z=sp[r['id']]['arms'][a]['FIXED'];mp=OLD/'features'/r['id']/a/'FIXED.json';meta=read(mp);alignment=mfa.get(r['id'],{}).get(label);tokens=[];status='MFA_UNAVAILABLE_IDENTITY';grid=None
   if alignment and Path(alignment.get('textgrid','/missing')).is_file():
    assert alignment['audio_sha256']==r['audio'][a]['sha256'];grid=Path(alignment['textgrid']);assert sha(grid)==alignment['textgrid_sha256'];tokens=phones(grid);status='BOUND';bind[str(grid)]=sha(grid)
   for path,h in [(z['waveform'],z['sha256']),(z['frontend'],z['frontend_sha256']),(meta['features'],meta['sha256'])]:assert sha(path)==h;bind[path]=h
   bind[str(mp)]=sha(mp);n=z['audio_windows'];alloc+=math.ceil((n*4096+128)/4096)*4096
   q['arms'][a]={'waveform':z['waveform'],'waveform_sha256':z['sha256'],'samples':z['samples'],'audio_windows':n,'joint_L':sp[r['id']]['arms'][a]['joint_L'],'frontend':z['frontend'],'frontend_sha256':z['frontend_sha256'],'baseline_features':meta['features'],'baseline_features_sha256':meta['sha256'],'source_PCM':r['audio'][a],'TextGrid':str(grid) if grid else None,'annotation_status':status,'tokens':tokens}
  rows.append(q)
 main=[r['id'] for r in rows if r['split']=='evaluation' and r['guard20_eligible']];assert len(main)==71
 event=read(EVENT);assert len(event['rows'])==32 and sum(len(r['queries']) for r in event['rows'])==1411;write(OUT/'event_indices.json',event)
 import python_speech_features as psf
 deps=[Path(__file__),BASE,OLD/'protocol.json',OLD/'support.json',OLD/'runtime_calibration.json',OLD/'runtime_evaluation.json',MFA,EVENT,CALSUP,FEAS/'cal_support.json',FEAS/'final_addendum.json',ROOT/'scripts/experiments/tts_native_gain_attribution/syncnet.py',ROOT/'scripts/experiments/tts_native_gain_attribution/audio.py',ROOT/'scripts/experiments/tts_native_gain_attribution/config.py',ROOT/'scripts/experiments/tts_native_gain_attribution/common.py',ROOT/'third_party/syncnet_python/SyncNetInstance.py',ROOT/'third_party/syncnet_python/SyncNetModel.py',ROOT/'third_party/syncnet_python/data/syncnet_v2.model',Path(psf.__file__),Path(psf.__file__).parent/'base.py',Path(psf.__file__).parent/'sigproc.py']
 current=runtime();oldrt=read(OLD/'runtime_calibration.json');assert all(current[k if k!='torch_cuda' else 'cuda']==v for k,v in oldrt.items())
 p={'time':time.time(),'version':1,'status':'frozen_before_new_forward_or_scores','rows':rows,'main_ids':main,'operation':'whole native MFCC sequence: per usable original phone reverse C1..C12 vectors, C0 unchanged; cut original windows only afterwards; no cross-arm time/donor','frame_support':'MFCC m raw sample integer support [max(0,160m-1),160m+400); no zero padding allowed. Phone half-open sample ticks [ceil(16000*decimal_start),ceil(16000*decimal_end)); require full containment; groups>=2, no overlap','unavailable':'silence/unknown/boundary/missing MFA or <2 usable frames unchanged. All200 retained including identity/no exposure. Invalid existing hash/overlap is engineering failure, not silently unavailable','excluded_labels':sorted(EXCLUDED),'controls':'all200 original and inverse windows exact; real SyncNet A baseline and double reversal must np.array_equal old FIXED A for every arm; any fail stops, no tolerance. all52 cal complete before eval; no cal score/effect or bestlag gate','generation':'none; V_FIXED exact cached, new A only actual SyncV2 forward_aud full1024. Seed20260926/batch32/torch threads2/cudnn deterministic true benchmarkfalse/project old runtime; no temporary media','scoring':old['scoring'],'statistics':old['statistics'],'primary':'raw guard20 common71/15: response of T-minus-N C to REV A vs FIXED A; report N/T changes and before/after residual,99CI; upper<0 gap shrink,lower>0 gap expansion,else unconfirmed. k3 unchanged; postintervention bestlag descriptive not gate','sensitivity':'raw/unit × guard20common71,valid common71/all74,guard0 common71/all74; C/B/D/C_anchor/D_anchor/bestlag all; no FWER','event':'legacy32/13/1411 query/donor support unchanged; raw/unit, four source cells NN/NT/TN/TT for baseline/reversed audio, V_FIXED; positive/mean negative/margin/rank. Query->clip->speaker, same20k99CI; diagonal responses and source contrasts descriptive, not official71 C decomposition','unblinding':'score/analyze require external joint field×phase protocol binding hash and timestamp predating first score_lock. No phase score delivered to root before joint seal. NewA generation may precede joint seal','boundaries':'original waveform/RMS/envelope/clock unchanged but encoder input edited; no realizable waveform or TFG generation/physical quality claim; preserved phone centroid/multiset does not isolate identity, contextual and exposure/damage effects remain; historical cohort reused','budget':{'feature_NPY_allocated_bound':alloc,'metadata_cap':10*2**20,'proved_total':alloc+10*2**20,'hard_cap':CAP,'other_reserved':OTHER,'floor':FLOOR},'runtime':current,'dependencies':{str(f.resolve()):sha(f) for f in deps},'asset_hashes':bind,'event_indices_sha256':sha(OUT/'event_indices.json')}
 assert alloc==93233152 and p['budget']['proved_total']<CAP;write(OUT/'protocol.json',p);write(OUT/'seal.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'worker_sha256':sha(__file__)});(OUT/'code_snapshot').mkdir();shutil.copyfile(__file__,OUT/'code_snapshot'/Path(__file__).name);print('FROZEN',sha(OUT/'protocol.json'),flush=True)
def prepare():
 p=protocol();out=[]
 for r in p['rows']:
  for a in 'NT':
   _,_,_,info=construct(r,a);write(OUT/'operations'/r['id']/(a+'.json'),info);out.append(info)
  print('prepared',r['id'],flush=True)
 write(OUT/'input_gate.json',{'time':time.time(),'passed':True,'arms':len(out),'MFA_available':sum(r['annotation_status']=='BOUND' for r in out),'no_coverage_selection':True,'no_new_scores':True});write(OUT/'input_seal.json',{str(f):sha(f) for f in sorted((OUT/'operations').rglob('*.json'))});print('INPUT_PASS',flush=True)
def gpucheck(start=False):
 z=limits();jobs=b.compute_processes();assert all(r['pid']==os.getpid() for r in jobs),jobs;used,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
 if start:assert used<=128 and util<=10
 return {**z,'compute':jobs,'gpu_MiB':used,'gpu_util':util}
@contextmanager
def lease(split):
 with open('/tmp/tts-exp-gpu.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   checks=[]
   for i in range(3):
    checks.append(gpucheck(True))
    if i<2:time.sleep(5)
   write(OUT/'runtime'/('resources_'+split+'.json'),checks);yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def forward(engine,w):
 out=[];torch=engine._torch
 with torch.inference_mode():
  for i in range(0,len(w),32):
   gpucheck();x=torch.from_numpy(w[i:i+32])[:,None].cuda();out.append(engine.network.forward_aud(x).cpu().numpy().astype(np.float32));limits()
 return np.concatenate(out)
def produce(split):
 import torch
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 p=protocol();assert read(OUT/'input_gate.json')['passed']
 if split=='evaluation':assert read(OUT/'calibration_controls.json')['passed']
 torch.set_num_threads(2);torch.manual_seed(20260926);torch.cuda.manual_seed_all(20260926);np.random.seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 rows=[r for r in p['rows'] if r['split']==split];controls=[]
 with lease(split):
  engine=SyncNetEngine(batch_size=32,device='cuda');h=hashlib.sha256()
  for k,v in engine.network.state_dict().items():h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  write(OUT/'runtime'/('loaded_'+split+'.json'),{'time':time.time(),'runtime':runtime(),'state_sha256':h.hexdigest(),'seed':20260926,'batch':32,'torch_threads':2})
  try:
   for r in rows:
    for a in 'NT':
     orig,new,inverse,info=construct(r,a);op=OUT/'operations'/r['id']/(a+'.json');assert info==read(op) and sha(op)==read(OUT/'input_seal.json')[str(op)];z=r['arms'][a];assert sha(z['baseline_features'])==z['baseline_features_sha256'];old=np.load(z['baseline_features'])['audio'];aa=forward(engine,orig);back=forward(engine,inverse);ctrl={'id':r['id'],'arm':a,'baseline_exact':bool(np.array_equal(aa,old)),'inverse_exact':bool(np.array_equal(back,old)),'baseline_max':float(abs(aa-old).max()),'inverse_max':float(abs(back-old).max()),'baseline_raw_sha256':ah(aa),'inverse_raw_sha256':ah(back),'baseline_shape':list(old.shape)};write(OUT/'controls'/r['id']/(a+'.json'),ctrl);assert ctrl['baseline_exact'] and ctrl['inverse_exact'];controls.append(ctrl);del aa,back
     result=forward(engine,new);path=OUT/'features'/r['id']/(a+'.npy');meta=save(path,result);write(OUT/'metadata'/r['id']/(a+'.json'),{'id':r['id'],'arm':a,'A':meta,'operation_sha256':sha(op),'control_sha256':sha(OUT/'controls'/r['id']/(a+'.json'))});limits()
    print('produced',split,r['id'],flush=True)
  finally:engine.close()
 write(OUT/(split+'_controls.json'),{'time':time.time(),'passed':True,'arms':len(controls),'baseline_and_inverse_all_exact':True,'no_lag_gate':True,'no_cal_effects':True});write(OUT/'runtime'/('worker_exit_'+split+'.json'),{'time':time.time(),'complete':True,'lease_released':True,'external_compute_empty_check_required':True})
def joint_gate():
 z=read(OUT/'joint_protocol_binding.json');assert z['status']=='FROZEN' and sha(z['protocol'])==z['protocol_sha256']==z['field_protocol_sha256'];assert sha(z['field_design_path'])==z['field_design_sha256'];assert z['created_epoch']<=z['time']<time.time();return z
def features(r,a):
 z=r['arms'][a];assert sha(z['baseline_features'])==z['baseline_features_sha256'];f=np.load(z['baseline_features']);meta=read(OUT/'metadata'/r['id']/(a+'.json'))['A'];assert sha(meta['path'])==meta['sha256'];return f['visual'],f['audio'],np.load(meta['path'])
def score():
 import torch
 torch.set_num_threads(2);p=protocol();joint=joint_gate();assert read(OUT/'evaluation_controls.json')['passed'];rows=[r for r in p['rows'] if r['split']=='evaluation'];write(OUT/'feature_seal.json',{str(f):sha(f) for d in ['features','metadata','controls'] for f in sorted((OUT/d).rglob('*')) if f.is_file()});now=time.time();write(OUT/'score_lock.json',{'time':now,'created_epoch':now,'protocol_sha256':sha(OUT/'protocol.json'),'joint_protocol':joint,'feature_seal_sha256':sha(OUT/'feature_seal.json'),'input_seal_sha256':sha(OUT/'input_seal.json')})
 for r in rows:
  cells={}
  for a in 'NT':
   v,base,rev=features(r,a);L=r['arms'][a]['joint_L'];cells[a]={}
   for geom in ['raw','unit']:
    vv,aa,rr=[b.unit(x[:L]) if geom=='unit' else x[:L] for x in [v,base,rev]];cells[a][geom]={c:{pol:b.summarize(b.matrix(vv,x),pol,3) for pol in p['scoring']['policies']} for c,x in [('FIXED',aa),('REV',rr)]}
  write(OUT/'scores'/(r['id']+'.json'),{'id':r['id'],'speaker':r['speaker'],'cells':cells});print('scored',r['id'],flush=True)
 event=read(OUT/'event_indices.json');table={r['id']:r for r in rows}
 for r in event['rows']:
  arrays={a:features(table[r['id']],a) for a in 'NT'};records=[];bygeom={g:{a:tuple(x/np.linalg.norm(x,axis=1)[:,None] if g=='unit' else x for x in arrays[a]) for a in 'NT'} for g in ['raw','unit']}
  for qi,q in enumerate(r['queries']):
   node=r['nodes'][q['node']];out={'query':qi,'values':{}}
   for geom in ['raw','unit']:
    ff=bygeom[geom];out['values'][geom]={}
    for state,ix in [('FIXED',1),('REV',2)]:
     out['values'][geom][state]={}
     for cell in ['NN','NT','TN','TT']:
      v=ff[cell[0]][0][node['j'][cell[0]]-3].astype(float);af=ff[cell[1]][ix].astype(float);pos=float(np.sqrt(np.sum((v-af[node['j'][cell[1]]]+1e-6)**2)));neg=np.sqrt(np.sum((v-af[[r['nodes'][d]['j'][cell[1]] for d in q['donors']]]+1e-6)**2,axis=1));out['values'][geom][state][cell]={'positive':pos,'negative':float(neg.mean()),'margin':float(neg.mean()-pos),'rank':float(np.mean((neg>pos)+.5*(neg==pos)))}
   records.append(out)
  write(OUT/'event_scores'/(r['id']+'.json'),{'id':r['id'],'speaker':r['speaker'],'queries':records})
def analyze():
 p=protocol();joint_gate();rows=[r for r in p['rows'] if r['split']=='evaluation'];ss={r['id']:read(OUT/'scores'/(r['id']+'.json')) for r in rows};out={};effects=[]
 for geom in ['raw','unit']:
  for pol in ['guard20','valid','guard0']:
   for support in (['common71'] if pol=='guard20' else ['common71','all74']):
    rr=[r for r in rows if support=='all74' or r['id'] in p['main_ids']];groups=[r['speaker'] for r in rr];view={}
    for metric in ['C','B','D','C_anchor','D_anchor','best_lag']:
     vals={c:{a:np.array([ss[r['id']]['cells'][a][geom][c][pol][metric] for r in rr]) for a in 'NT'} for c in ['FIXED','REV']};dn=vals['REV']['N']-vals['FIXED']['N'];dt=vals['REV']['T']-vals['FIXED']['T'];view[metric]={'baseline_T_minus_N':b.bootstrap(vals['FIXED']['T']-vals['FIXED']['N'],groups),'processed_T_minus_N':b.bootstrap(vals['REV']['T']-vals['REV']['N'],groups),'response':{'N':b.bootstrap(dn,groups),'T':b.bootstrap(dt,groups),'T_minus_N':b.bootstrap(dt-dn,groups)}}
     effects.extend({'id':r['id'],'view':geom+'/'+pol+'/'+support,'metric':metric,'N_response':float(dn[i]),'T_response':float(dt[i]),'gap_response':float(dt[i]-dn[i])} for i,r in enumerate(rr))
    out[geom+'/'+pol+'/'+support]=view
 event=read(OUT/'event_indices.json')['rows'];ev={r['id']:read(OUT/'event_scores'/(r['id']+'.json')) for r in event};groups=[r['speaker'] for r in event];eres={}
 contrasts={'visual_at_N_audio':{'TN':1,'NN':-1},'audio_at_N_visual':{'NT':1,'NN':-1},'interaction':{'TT':1,'NN':1,'TN':-1,'NT':-1},'diagonal':{'TT':1,'NN':-1}}
 for g in ['raw','unit']:
  eres[g]={}
  for metric in ['positive','negative','margin','rank']:
   vals={state:{c:np.array([np.mean([q['values'][g][state][c][metric] for q in ev[r['id']]['queries']]) for r in event]) for c in ['NN','NT','TN','TT']} for state in ['FIXED','REV']};res={}
   for name,weights in contrasts.items():
    before=sum(vals['FIXED'][c]*w for c,w in weights.items());after=sum(vals['REV'][c]*w for c,w in weights.items());res[name]={'baseline':b.bootstrap(before,groups),'reversed':b.bootstrap(after,groups),'response':b.bootstrap(after-before,groups)}
   res['native_arm_response']={a:b.bootstrap(vals['REV'][a+a]-vals['FIXED'][a+a],groups) for a in 'NT'};eres[g][metric]=res
 main=out['raw/guard20/common71']['C']['response']['T_minus_N'];status='GAP_SHRINK_CONFIRMED' if main['ci99'][1]<0 else ('GAP_EXPANSION_CONFIRMED' if main['ci99'][0]>0 else 'GAP_CHANGE_NOT_CONFIRMED');write(OUT/'effects.json',effects);write(OUT/'summary.json',{'status':status,'primary_gap_response':main,'views':out,'event':eres,'no_cal_effects':True,'no_coverage_or_score_selection':True});print(status,json.dumps(main),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','produce','score','analyze']);ap.add_argument('--split',choices=['calibration','evaluation']);args=ap.parse_args();globals()[args.stage](args.split) if args.stage=='produce' else globals()[args.stage]()
