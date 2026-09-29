"""Frozen RAW/U/S/FIXED generation-input experiment, evaluation audio always RAW."""
from pathlib import Path
import argparse,copy,hashlib,importlib.metadata,json,os,shutil,sys,time,types
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_fixed_mel_partition_generation_20260927';OLD=ROOT/'runs/tts_fixed_level_generation_cross_20260927';FEAS=ROOT/'runs/tts_fixed_mel_partition_feasibility_20260927'
HELPFILE=ROOT/'runs/tts_fixed_natural_holdout80_20260927/code_snapshot/tts_fixed_natural_holdout80_20260927.py'
def load(path):
 m=types.ModuleType('partition_helpers');m.__file__=str(ROOT/'scripts/experiments'/path.name);exec(compile(path.read_text(),str(path),'exec'),m.__dict__);return m
h=load(HELPFILE);base=h.base;h.OUT=OUT;base.OUT=OUT
read,sha=h.read,h.sha
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
CONDITIONS=['RAW','U','S','FIXED'];FIELDS=['C','B','D','C_anchor','D_anchor','best_lag'];EFFECTS={'U':[-1,1,0,0],'S':[-1,0,1,0],'interaction':[1,-1,-1,1],'FIXED':[-1,0,0,1],'U_after_S':[0,0,-1,1],'S_after_U':[0,-1,0,1],'U_minus_S':[0,1,-1,0]}
def allocation():return sum(f.stat().st_blocks*512 for f in OUT.rglob('*') if f.is_file())
def disk_check(next_bytes=0):
 free=shutil.disk_usage(OUT).free;n=allocation();assert free-next_bytes>=5*2**30,('disk5GiB',free,next_bytes);assert n+next_bytes<=256*2**20,('cap256MiB',n,next_bytes);return {'free_bytes':free,'new_allocated_bytes':n}
def resource_check(p,startup=False):
 r=base.compute_processes();assert all(x['pid']==os.getpid() for x in r),('foreign compute',r)
 used,util=map(int,base.subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
 if startup:
  assert used<=128 and util<=10,(used,util)
  assert shutil.disk_usage(OUT).free>=5*2**30+(256*2**20-allocation())+40*2**20,'reserve ownremaining cap plus visual40MiB'
 return {'time':time.time(),'compute':r,'memory_mib':used,'utilization':util,**disk_check()}
base.resource_check=resource_check;h.disk_check=lambda:disk_check(32*224*224*3+2**20)
def environment():return {'python':sys.executable,'versions':{k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','librosa','opencv-python','python_speech_features','soundfile']}}
def protocol():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip();assert environment()==p['runtime']
 for f,v in {**p['dependencies'],**p['new_code_hashes']}.items():assert sha(f)==v,f
 return p
def freeze():
 assert not OUT.exists();OUT.mkdir();old=read(OLD/'protocol.json');support=read(OLD/'support.json');evaluation=[r for r in support if r['split']=='evaluation'];cal=min((r for r in support if r['split']=='calibration'),key=lambda r:r['id']);main=[r['id'] for r in evaluation if r['guard20_eligible']];assert len(evaluation)==74 and len(main)==71
 deps=copy.deepcopy(old['dependencies'])
 for f in [HELPFILE,OLD/'protocol.json',OLD/'support.json',OLD/'cal_controls.json',FEAS/'report.md',FEAS/'budget74.json',FEAS/'independent_check.json',FEAS/'final.json']:deps[str(f)]=sha(f)
 files=[Path(__file__),ROOT/'scripts/experiments/check_tts_fixed_mel_partition_generation_20260927.py']
 p={'status':'frozen_before_new_mels_V_scores','time':time.time(),'approval':'root explicitly approved fullcloud74 U/S generationfourcell; GPU after visual8cal release; hard256MiB/free5GiB and startup visual40MiB reserve','proposal_sha256':sha(FEAS/'report.md'),'runtime':environment(),'rows':[cal,*evaluation],'main_ids':main,'all_ids':[r['id'] for r in evaluation],'cal_id':cal['id'],'image':old['image'],'conditions':CONDITIONS,'partition':'savedfloat32 endpoints M0 RAW/M1 actualFIXED; U=(-4<M0<4)&(-4<M1<4), S=~U. M_U=where(U,M1,M0), M_S=where(~U,M1,M0). No epsilon, no fitted threshold. Exact boundary and one inwardULP counted; no reassignment.','target_RMS':old['fixed_target']['target_RMS'],'generation':old['generation'],'scoring':old['scoring'],'statistics':old['statistics'],'effects':EFFECTS,'audio':'ALL four generation conditions evaluated on existing RAW A only. New UxS interaction is not old GxE interaction. FullFIXED effect equals prior G-only, not prior native total.','support':'common71 eachraw/unit xguard20/valid/guard0 primary; all74 valid/guard0 separate; N/T/T-N, all6 existingmetrics and all7effects; no deletions','controls':'firstcalN/T RAW/FIXED actual newrender pixel/MFCC/A/V/query exact, failure stop; then U/S newrender frame/query/finite/input chunk hash checks; inherited hashbound +5delay from identical parentfrontend unless exact endpoint gate fails (then stop)','resources':{'cap_bytes':256*2**20,'reserve_bytes':5*2**30,'startup_extra_visual_reserve_bytes':40*2**20,'lease':'/tmp/tts-exp-gpu.lock','checks':'three5s GPUcheck; each32frame batch reserves nextbatch rawpixels+1MiB; disk/budget beforepersist and eachcell'},'retention':'no newcomplete videos retained; persist V.npy/masks/endpoint input and chunk hashes, decodedpixel/file creationhashes, atomiccommit before only newtemporary video deletion; reuseA and endpointV by reference; no newz','limits':'cell subsets differ in size/energy/location; legalmel may be off-manifold or not correspond to waveform. No unitperturbation/causepercentage/equivalence/true mouth claim. No scoreselection.','dependencies':deps,'new_code_hashes':{str(f):sha(f) for f in files}}
 for f,v in deps.items():assert sha(f)==v
 write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
 for f in files:d=OUT/'code_snapshot'/f.name;d.parent.mkdir(exist_ok=True);shutil.copyfile(f,d)
 print('frozen',sha(OUT/'protocol.json'),flush=True)
def mels(row,arm):
 z=row['arms'][arm];a=np.load(z['RAW']['frontend'])['mel'];b=np.load(z['FIXED']['frontend'])['mel'];u=(a>-4)&(a<4)&(b>-4)&(b<4);return {'RAW':a,'U':np.where(u,b,a),'S':np.where(u,a,b),'FIXED':b},u
def prepare():
 from scripts.experiments.static_image_bridge.render_worker import chunk_mels
 p=protocol();sys.path.insert(0,str(h.W2L));import audio
 support=[];bindings={};count=0
 for r in p['rows']:
  row={'id':r['id'],'speaker':r['speaker'],'split':r['split'],'main':r['id'] in p['main_ids'],'arms':{}}
  for arm in ['N','T']:
   z=r['arms'][arm];a={'L':z['joint_L']}
   for c,oldc in [('RAW','raw'),('FIXED','FIXED')]:
    info=z[oldc];meta=OLD/'features'/r['id']/arm/(oldc+'.json');mm=read(meta);fp=Path(mm['features']);a[c]=copy.deepcopy(info);a[c].update(feature=str(fp),feature_sha256=mm['sha256'],metadata=str(meta),metadata_sha256=sha(meta),video=mm['video'])
    for f,v in [(info['waveform'],info['sha256']),(info['frontend'],info['frontend_sha256']),(str(fp),mm['sha256']),(str(meta),sha(meta))]:assert sha(f)==v;bindings[f]=v
    x=sf.read(info['waveform'],dtype='float32')[0];fm=np.load(info['frontend']);assert np.array_equal(audio.melspectrogram(x).astype(np.float32),fm['mel']);w,_=base.frontend_windows(x.astype(float));assert np.array_equal(w,fm['windows']);assert len(x)==info['samples']
   row['arms'][arm]=a;mm,u=mels(row,arm);assert all(np.isfinite(v).all() and v.min()>=-4 and v.max()<=4 for v in mm.values());d=mm['FIXED'].astype(float)-mm['RAW'].astype(float);assert np.array_equal((mm['U'].astype(float)-mm['RAW'])+(mm['S'].astype(float)-mm['RAW']),d)
   chunks={c:chunk_mels(m,25) for c,m in mm.items()};assert len({len(x) for x in chunks.values()})==1 and len(chunks['RAW'])==a['RAW']['frames']==a['FIXED']['frames'];assert min(a['RAW']['frames'],a['RAW']['samples']//640)-5==a['L']
   path=OUT/'masks'/r['id']/(arm+'.npy');path.parent.mkdir(parents=True,exist_ok=True);np.save(path,np.packbits(u.ravel()));a['mask']={'path':str(path),'sha256':sha(path),'shape':list(u.shape),'U_count':int(u.sum()),'S_count':int((~u).sum()),'inward_one_ULP_cells':int(((abs(mm['RAW'])<4)&(abs(mm['RAW'])>=np.nextafter(np.float32(4),np.float32(0)))).sum())};a['inputs']={c:{'mel_sha256':ah(mm[c]),'chunks_sha256':ah(np.asarray(chunks[c],dtype=np.float32)),'frames':len(chunks[c])} for c in CONDITIONS};count+=d.size
  support.append(row);disk_check();print('prepared',r['id'],flush=True)
 write(OUT/'support.json',support);write(OUT/'bindings.json',bindings);write(OUT/'frontend_gate.json',{'passed':True,'rows':75,'arms':150,'partition_cells':count,'all_saved_frontends_rebuilt_exact':True,'range_finite_frames_L_passed':True,'oldmain71_unchanged':True})
def seal():
 p=protocol();assert read(OUT/'frontend_gate.json')['passed'] and read(OUT/'independent_inputs.json')['status']=='PASS';assert not (OUT/'features').exists();write(OUT/'cpu_seal.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'support_sha256':sha(OUT/'support.json'),'bindings_sha256':sha(OUT/'bindings.json'),'independent_inputs_sha256':sha(OUT/'independent_inputs.json'),'status':'beforeGPUnewVandscores'})
def persist(row,arm,c,v,video,extra):
 disk_check(v.nbytes+16384);dest=OUT/'features'/row['id']/arm/c;dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix('.pending.npy');np.save(temp,v);os.replace(temp,dest.with_suffix('.npy'));assert np.array_equal(np.load(dest.with_suffix('.npy')),v) and np.isfinite(v).all()
 meta={'id':row['id'],'arm':arm,'condition':c,'feature':str(dest.with_suffix('.npy')),'feature_sha256':sha(dest.with_suffix('.npy')),'video':video,'retained_video':False,'L':row['arms'][arm]['L'],'input':row['arms'][arm]['inputs'][c.replace('_repeat','')],**extra};tmp=dest.with_suffix('.pending.json');write(tmp,meta);os.replace(tmp,dest.with_suffix('.json'))
 vp=Path(video['path']);assert vp.is_relative_to(OUT/'temporary') and sha(vp)==video['sha256'];write(OUT/'retention_commits'/row['id']/arm/(c+'.json'),{'metadata_sha256':sha(dest.with_suffix('.json')),'feature_sha256':meta['feature_sha256'],'video_sha256_at_creation':video['sha256'],'pixel_sha256_at_creation':video['pixel_sha256'],'time':time.time(),'final_video_retained':False});vp.unlink();disk_check();return meta
def produce():
 import cv2,torch
 from scripts.experiments.static_image_bridge.render_worker import chunk_mels
 from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 p=protocol();ss=read(OUT/'cpu_seal.json');assert sha(OUT/'support.json')==ss['support_sha256'];assert read(OUT/'independent_inputs.json')['status']=='PASS';rows=read(OUT/'support.json');cal=rows[0];assert cal['id']==p['cal_id']
 cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 with base.lease('produce',p):
  write(OUT/'runtime.json',environment());np.savez(OUT/'rng.npz',cpu=torch.get_rng_state().numpy(),cuda=torch.cuda.get_rng_state().cpu().numpy());model=_load_model(h.W2L/'checkpoints/wav2lip_gan.pth','cuda');engine=SyncNetEngine(batch_size=32,device='cuda');im=p['image'];frame=cv2.imread(im['path']);x1,y1,x2,y2=im['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));masked=face.copy();masked[48:]=0;it=torch.from_numpy(np.concatenate([masked,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda();controls=[]
  try:
   for arm in ['N','T']:
    mm,_=mels(cal,arm)
    for c in ['RAW','FIXED']:
     resource_check(p);z=cal['arms'][arm][c];chunks=chunk_mels(mm[c],25);assert ah(np.asarray(chunks,dtype=np.float32))==cal['arms'][arm]['inputs'][c]['chunks_sha256'];video=h.render(model,chunks,frame,it,im,OUT/'temporary'/f'{arm}_{c}_repeat.avi');v,vm=engine.extract_visual(video['path']);front=np.load(z['frontend']);x=sf.read(z['waveform'],dtype='float64')[0];w,_=base.frontend_windows(x);a=base.audio_forward(engine,w);old=np.load(z['feature']);tests={'pixel_exact':video['pixel_sha256']==z['video']['pixel_sha256'],'V_exact':bool(np.array_equal(v,old['visual'])),'MFCC_exact':bool(np.array_equal(w,front['windows'])),'A_exact':bool(np.array_equal(a,old['audio'])),'query_exact':video['frames']==z['frames'] and min(video['frames'],len(x)//640)-5==cal['arms'][arm]['L']};persist(cal,arm,c+'_repeat',v,video,{'tests':tests,'passed':all(tests.values()),'A_sha256':ah(a),'MFCC_sha256':ah(w),'visual_metadata':vm});controls.append({'arm':arm,'condition':c,**tests})
     if not all(tests.values()):write(OUT/'engineering_gate.json',{'passed':False,'controls':controls,'action':'STOP no retune/rerender/subset'});raise AssertionError(('endpoint exact failure',arm,c,tests))
   inherited=read(OLD/'cal_controls.json');assert inherited['engineering_passed'] and all(v['lag_shift_pass'] and v['curve_overlap_max']<=.15 for v in inherited['delay'].values());write(OUT/'engineering_gate.json',{'passed':True,'controls':controls,'before_any_U_S_video':True,'delay_inherited':inherited['delay'],'delay_parent_sha256':sha(OLD/'cal_controls.json'),'reason':'boundidentical waveform/frontend/engine and all4A/V/MFCC exact; no k/query changes'})
   for row in rows:
    for arm in ['N','T']:
     mm,_=mels(row,arm)
     for c in ['U','S']:
      resource_check(p);m=mm[c];chunks=chunk_mels(m,25);spec=row['arms'][arm]['inputs'][c];assert ah(m)==spec['mel_sha256'] and ah(np.asarray(chunks,dtype=np.float32))==spec['chunks_sha256'];video=h.render(model,chunks,frame,it,im,OUT/'temporary'/f'{row["id"]}_{arm}_{c}.avi');v,vm=engine.extract_visual(video['path']);assert video['frames']==spec['frames'] and len(v)>=row['arms'][arm]['L'];persist(row,arm,c,v,video,{'input_chunks_exact':True,'visual_metadata':vm})
    if row['split']=='calibration':write(OUT/'cal_U_S_gate.json',{'passed':True,'id':row['id'],'cells':4,'before_eval':True,'range_query_input_chunk_hash_finite':True})
    else:assert read(OUT/'cal_U_S_gate.json')['passed']
    print('produced',row['id'],flush=True)
  finally:engine.close()
 write(OUT/'gpu_release.json',{'time':time.time(),'pid':os.getpid(),'lease_released':True,'eval_new_videos':296,'cal_new_videos':8,'retained_new_videos':0,**disk_check()})
def features(row,arm):
 d=row['arms'][arm];old={c:np.load(d[c]['feature']) for c in ['RAW','FIXED']};v={c:old[c]['visual'] for c in old}
 for c in ['U','S']:v[c]=np.load(OUT/'features'/row['id']/arm/(c+'.npy'))
 return v,old['RAW']['audio']
def score():
 import torch
 torch.set_num_threads(2);p=protocol();assert read(OUT/'gpu_release.json')['eval_new_videos']==296 and read(OUT/'engineering_gate.json')['passed'];seals={str(f):sha(f) for f in (OUT/'features').rglob('*') if f.is_file()};write(OUT/'feature_seal.json',seals);write(OUT/'score_lock.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'feature_seal_sha256':sha(OUT/'feature_seal.json'),'status':'beforeallnewdistances'})
 for row in read(OUT/'support.json'):
  cells={}
  for arm in ['N','T']:
   v,a=features(row,arm);L=row['arms'][arm]['L'];cells[arm]={}
   for geom in ['raw','unit']:
    aa=base.unit(a[:L]) if geom=='unit' else a[:L];cells[arm][geom]={}
    for c in CONDITIONS:
     vv=base.unit(v[c][:L]) if geom=='unit' else v[c][:L];matrix=base.matrix(vv,aa);dest=OUT/'scores'/row['id']/arm/geom/c;dest.parent.mkdir(parents=True,exist_ok=True);disk_check(matrix.nbytes+32768);np.save(dest.with_suffix('.npy'),matrix);cells[arm][geom][c]={'policies':{pol:h.summarize(matrix,pol) for pol in ['guard20','valid','guard0'] if pol!='guard20' or L>40},'matrix':str(dest.with_suffix('.npy')),'matrix_sha256':sha(dest.with_suffix('.npy'))}
  write(OUT/'scores'/row['id']/'cells.json',{'id':row['id'],'speaker':row['speaker'],'split':row['split'],'cells':cells});print('scored',row['id'],flush=True)
def analyze():
 p=protocol();allrows=read(OUT/'support.json');summary={};effectrows=[];closure=0.
 for geom in ['raw','unit']:
  for pop,policies in [('common71',['guard20','valid','guard0']),('all74',['valid','guard0'])]:
   rows=[r for r in allrows if r['id'] in (p['main_ids'] if pop=='common71' else p['all_ids'])];groups=[r['speaker'] for r in rows];data=[read(OUT/'scores'/r['id']/'cells.json') for r in rows]
   for pol in policies:
    key=f'{geom}/{pol}/{pop}';view={}
    for field in FIELDS:
     qq={a:np.asarray([[d['cells'][a][geom][c]['policies'][pol][field] for c in CONDITIONS] for d in data],dtype=float) for a in ['N','T']};qq['T_minus_N']=qq['T']-qq['N'];view[field]={}
     for arm,q in qq.items():
      ee={k:q@np.asarray(coef,dtype=float) for k,coef in EFFECTS.items()};closure=max(closure,float(abs(ee['U']+ee['S']+ee['interaction']-ee['FIXED']).max()));view[field][arm]={'cells':{c:base.bootstrap(q[:,j],groups) for j,c in enumerate(CONDITIONS)},'effects':{k:base.bootstrap(v,groups) for k,v in ee.items()}}
      for i,r in enumerate(rows):effectrows.append({'id':r['id'],'speaker':r['speaker'],'view':key,'metric':field,'arm':arm,'q':q[i].tolist(),**{k:float(v[i]) for k,v in ee.items()}})
    summary[key]=view
 write(OUT/'summary.json',{'results':summary,'max_closure':closure,'primary':'raw/guard20/common71 T_minus_N C responses','effect_definitions':EFFECTS});write(OUT/'effects.json',effectrows);print('analyzed',len(summary),'views',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','seal','produce','score','analyze']);args=ap.parse_args();globals()[args.stage]()
