"""Frozen CPU metadata, prototype fitting and score analysis. GPU worker is separate."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,ast,gzip,hashlib,json,math,shutil,sys,time
from decimal import Decimal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927'
FIX=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
PHASE=ROOT/'runs/tts_fixed_mfcc_phase_reversal_20260927'
EVENT=ROOT/'runs/tts_level_event_cross_20260927'
OLDJOINT=ROOT/'runs/tts_fixed_phone_shared_field_20260927'
BASE=ROOT/'runs/tts_acoustic_generation_cross_20260926/code_snapshot/tts_acoustic_generation_cross_20260926.py'
CONDS=['baseline','phoneN','phoneT','globalN','globalT'];GEOMS=['raw','unit'];METRICS=['C','B','D','C_anchor','D_anchor','search_uplift','best_lag']
CAP=304*2**20;AUDIT=16*2**20;FLOOR=int(4.25*2**30)
from scripts.experiments import tts_level_event_cross_20260927 as event

def read(p):
 p=Path(p)
 return json.loads(gzip.open(p,'rt').read()) if p.suffix=='.gz' else json.loads(p.read_text())
def sha(p):return event.sha(p)
def allocated(p):return sum(f.stat().st_blocks*512 for f in p.rglob('*') if f.is_file())
def limits(extra=0):
 used=allocated(OUT);free=shutil.disk_usage(OUT).free
 assert used+extra<=CAP and free-max(0,CAP-used)-AUDIT>=FLOOR,('resource',used,extra,free)
 return {'allocated':used,'free':free,'remaining_commitment':max(0,CAP-used)+AUDIT}
def write(p,obj):
 data=json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n';p=Path(p)
 if p.suffix=='.gz':data=gzip.compress(data,mtime=0)
 limits(math.ceil(len(data)/4096)*4096);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
 limits()
def save(p,**arrays):
 limits(sum(a.nbytes for a in arrays.values())+4096);p=Path(p);assert not p.exists();np.savez_compressed(p,**arrays);limits()
def official_functions():
 names={'matrix','summarize','unit','bootstrap'};tree=ast.parse(BASE.read_text());ns={'np':np,'LAGS':np.arange(-15,16)}
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(BASE),'exec'),ns)
 return ns
F=official_functions()
def freeze():
 assert not (OUT/'protocol.json').exists();limits();feas=read(OUT/'feasibility.json');calids={r['id'] for r in feas['rows']};parent=read(FIX/'protocol.json');sp=read(FIX/'support.json');pr={r['id']:r for r in read(PHASE/'protocol.json')['rows']};bindings={}
 def bind(p,expected=None):
  p=Path(p);h=sha(p);assert expected is None or h==expected,str(p);bindings[str(p.resolve())]=h
 for p in [OUT/'feasibility.json',OUT/'prepare_metadata.py',OUT/'design.md',OUT/'budget_start_receipt.json',BASE,FIX/'protocol.json',FIX/'support.json',FIX/'feature_seal_calibration.json',FIX/'feature_seal_evaluation.json',PHASE/'protocol.json',PHASE/'input_seal.json',EVENT/'indices.json',EVENT/'query_metrics.npz',EVENT/'summary.json',OLDJOINT/'final.json']:
  bind(p)
 for path,h in parent['dependencies'].items():
  # Parent dependencies are all immutable and explicitly verified before reference.
  bind(path,h)
 rows=[]
 for r in sp:
  if r['split']=='calibration' and r['id'] not in calids:continue
  row={k:r[k] for k in ['id','speaker','split','guard20_eligible']};row['arms']={};seal=read(FIX/f'feature_seal_{r["split"]}.json')
  for a in 'NT':
   q=r['arms'][a]['FIXED'];fp=FIX/'features'/r['id']/a/'FIXED.npz';mp=fp.with_suffix('.json');bind(fp,seal[str(fp)]);bind(mp,seal[str(mp)]);bind(q['frontend'],q['frontend_sha256']);bind(q['waveform'],q['sha256'])
   pa=pr[r['id']]['arms'][a];assert pa['baseline_features']==str(fp) and pa['baseline_features_sha256']==sha(fp);bind(pa['TextGrid'])
   with np.load(q['frontend']) as f:M=f['mel'].shape[1]
   starts=[];i=0
   while int(i*80/25)+16<=M:starts.append(int(i*80/25));i+=1
   starts.append(M-16);assert len(starts)==q['frames']
   labels=[];mask=[]
   for s in starts:
    t=Decimal(2*s+15)/Decimal(160);matches=[x for x in pa['tokens'] if Decimal(x['lo_decimal'])<=t<Decimal(x['hi_decimal'])];assert len(matches)<=1
    label=matches[0]['label'] if matches and matches[0]['usable'] else None;labels.append(label);mask.append(label is not None)
   with np.load(fp) as f:
    assert f['z'].shape==(q['frames'],512) and f['z'].dtype==np.float32 and np.isfinite(f['z']).all() and (f['z']>=0).all()
    assert f['visual'].shape==(q['frames']-4,1024) and len(f['audio'])==q['audio_windows']
   row['arms'][a]={'features_path':str(fp),'features_sha256':sha(fp),'metadata_path':str(mp),'metadata_sha256':sha(mp),'frontend_path':q['frontend'],'frontend_sha256':q['frontend_sha256'],'waveform_path':q['waveform'],'waveform_sha256':q['sha256'],'frames':q['frames'],'samples':q['samples'],'L':q['L'],'mel_starts':starts,'phone_labels':labels,'speech_mask':mask,'TextGrid':pa['TextGrid'],'TextGrid_sha256':sha(pa['TextGrid'])}
  if r['split']=='evaluation':bind(FIX/'scores'/(r['id']+'.json'))
  rows.append(row)
 assert len(rows)==94 and sum(r['split']=='evaluation' for r in rows)==74
 write(OUT/'rows.json',rows)
 from scripts.experiments import tts_fixed_generator_phone_gpu_20260927 as gpu
 from scripts.experiments.tts_native_gain_attribution import config as sc
 paths=[Path(__file__),ROOT/'scripts/experiments/tts_fixed_generator_phone_gpu_20260927.py',ROOT/'scripts/experiments/tts_fixed_generator_phone_stream_20260927.py',Path(event.__file__),ROOT/'scripts/experiments/masked_tts_tfg_probe/direct_mel.py',ROOT/'scripts/experiments/static_image_bridge/render_worker.py',ROOT/'scripts/experiments/static_image_bridge/score_worker.py']
 paths += [ROOT/'scripts/experiments/tts_native_gain_attribution'/n for n in ['syncnet.py','config.py','common.py']]
 paths += [ROOT/'third_party/syncnet_python'/n for n in ['SyncNetModel.py','SyncNetInstance.py']]
 paths += [ROOT/'third_party/Wav2Lip'/n for n in ['audio.py','hparams.py','models/__init__.py','models/conv.py','models/wav2lip.py','checkpoints/wav2lip_gan.pth']]
 paths += [ROOT/'third_party/syncnet_python/data/syncnet_v2.model',Path(sc.FFMPEG),Path(sc.FFPROBE)]
 for f in paths:bind(f)
 c={'runtime':gpu.runtime(),'image':parent['image'],'checkpoint':str(ROOT/'third_party/Wav2Lip/checkpoints/wav2lip_gan.pth'),'ffmpeg':str(Path(sc.FFMPEG).resolve()),'ffprobe':str(Path(sc.FFPROBE).resolve()),'parent_protocol_path':str(FIX/'protocol.json'),'row_path':str(OUT/'rows.json'),'tmp_cap_bytes':96*2**20,'own_cap_bytes':CAP,'other_reserved_bytes':AUDIT,'floor_bytes':FLOOR,'prerequisite_closure':{'path':str(OLDJOINT/'final.json'),'sha256':sha(OLDJOINT/'final.json')},'first_cal':'a1_001'}
 for key in ['checkpoint','ffmpeg','ffprobe','parent_protocol_path','row_path']:c[key+'_sha256']=sha(c[key])
 bind(parent['image']['path'],parent['image']['sha256'])
 p={'status':'FROZEN_BEFORE_IDENTITY_FIT_NEW_FORWARD_SCORE','created_epoch':time.time(),'design_sha256':sha(OUT/'design.md'),'conditions':CONDS,'geometries':GEOMS,'official_metrics':METRICS,'event_metrics':event.METRICS,'gpu_config':c,'dependencies':bindings,'rows_sha256':sha(OUT/'rows.json'),'cal_support_sha256':sha(OUT/'feasibility.json'),'primary':'raw/guard20/common71 C natural phoneT-phoneN AND phoneT-baseline both99CI lower>0','interpretation':'Only phoneT>phoneN: smaller replacement damage; only phoneT>baseline: mixture benefit not TTS-specific. Anchor/search uplift separately govern timing interpretation.','arithmetic':'float64 frame/occurrence/clip/speaker means; raw_mu source dimensions N,T; shared k; shrink64 then cast32; half_z=np.multiply(z,float32(.5),dtype=float32); half_mu same; np.add(dtype=float32); inactive exact copy','unit_definitions':'official: float64 normalize then cast32, exactly parent; legacy event: original float32 norm/divide, exactly parent. Never mix geometries.','stats':'speaker-equal20000 PCG64 seed20260926 percentile99CI, noFWER; fixedcalfit conditional inference','gates':{'baseline_curve_max':1e-9,'baseline_event_query_max':1e-9,'identity_z_pixels_V':'exact','no_new_scores_before_calibration_gate':True,'triangle_interaction':1e-10},'resources':{'total_including_audit':320*2**20,'main':CAP,'audit':AUDIT,'floor':FLOOR,'temporary_dev_shm_cap':96*2**20,'CPU_max_threads':2,'eval_new_video_cells':592,'expected_GPU_minutes_reference':50,'GPU_budget_minutes':90,'timing_note':'old ENV148 streamed includingA took544.2s;592 linear36.3min; identity/cal/streamaudit overhead; no guarantee'},'primary_support':[r['id'] for r in rows if r['split']=='evaluation' and r['guard20_eligible']],'legacy_event_support_sha256':sha(EVENT/'indices.json')}
 write(OUT/'protocol.json',p);write(OUT/'seal.json',{'protocol.json':sha(OUT/'protocol.json'),'rows.json':sha(OUT/'rows.json'),'design.md':sha(OUT/'design.md'),'feasibility.json':sha(OUT/'feasibility.json')});print('FROZEN',sha(OUT/'protocol.json'),flush=True)
def locked():
 p=read(OUT/'protocol.json')
 for n,h in read(OUT/'seal.json').items():assert sha(OUT/n)==h
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 rv=read(OUT/'reviewer_pass.json');assert rv['status']=='PASS' and rv['protocol_sha256']==sha(OUT/'protocol.json') and sha(rv['receipt'])==rv['receipt_sha256']
 return p,read(OUT/'rows.json')
def fit():
 p,rows=locked();gate=read(OUT/'identity_gate.json');assert gate['passed'] and gate['external_media_audit_passed'];support=read(OUT/'feasibility.json')['rows'];labels=sorted({o['phone'] for r in support for o in r['occurrences']});assert len(labels)==82
 folds=sorted({r['speaker'] for r in rows if r['split']=='evaluation'});means=np.full((20,2,83,512),np.nan)
 for ri,r in enumerate(support):
  z={}
  for a in 'NT':
   with np.load(FIX/'features'/r['id']/a/'FIXED.npz') as f:z[a]=f['z'].astype(np.float64)
  for ai,a in enumerate('NT'):
   v=np.stack([z[a][o['indices'][a]].mean(0) for o in r['occurrences']]);pl=np.array([o['phone'] for o in r['occurrences']]);means[ri,ai,-1]=v.mean(0)
   for li,label in enumerate(labels):
    take=pl==label
    if take.any():means[ri,ai,li]=v[take].mean(0)
 raw=np.zeros((len(folds),2,83,512));counts=np.zeros((len(folds),82),np.int16);training={}
 for fi,fold in enumerate(folds):
  ii=[i for i,r in enumerate(support) if r['speaker']!=fold];speakers=sorted({support[i]['speaker'] for i in ii});training[fold]=[support[i]['id'] for i in ii]
  for li in range(83):
   vals=[]
   for s in speakers:
    use=[i for i in ii if support[i]['speaker']==s and np.isfinite(means[i,0,li]).all()]
    if use:vals.append(means[use,:,li].mean(0))
   if vals:raw[fi,:,li]=np.stack(vals).mean(0)
   if li<82:counts[fi,li]=len(vals)
  for li in range(82):
   if counts[fi,li]==0:raw[fi,:,li]=raw[fi,:,-1]
 assert np.isfinite(raw).all() and (raw>=0).all();save(OUT/'fit.npz',raw_mu=raw,speaker_counts=counts)
 write(OUT/'fit.json',{'status':'CAL_ONLY_LOSO_FIT_SEALED','created_epoch':time.time(),'labels':labels,'folds':folds,'sources':['N','T'],'training_ids':training,'counts':counts.tolist(),'cal_clips':20,'paired_occurrences':723,'excluded_occurrences':27,'fit_sha256':sha(OUT/'fit.npz'),'evaluation_z_used_for_fit':False})
 write(OUT/'fit_seal.json',{'protocol_sha256':sha(OUT/'protocol.json'),'identity_gate_sha256':sha(OUT/'identity_gate.json'),'files':{str(OUT/n):sha(OUT/n) for n in ['fit.npz','fit.json']}});print('FIT_SEALED',flush=True)
def official_scores(v,a,L):
 result={}
 for g in GEOMS:
  vv,aa=(F['unit'](v[:L]),F['unit'](a[:L])) if g=='unit' else (v[:L],a[:L]);m=F['matrix'](vv,aa);result[g]={}
  for policy in ['guard20','valid','guard0']:
   x=F['summarize'](m,policy)
   if x is not None:x['search_uplift']=x['C']-x['C_anchor']
   result[g][policy]=x
 return result

def baseline():
 p,rows=locked();import torch;torch.set_num_threads(2);error=0.;ev=[r for r in rows if r['split']=='evaluation'];erows=read(EVENT/'indices.json')['rows'];esp=read(EVENT/'indices.json')['spans'];evmap={r['id']:r for r in erows};out=np.zeros((1411,2,2,4));spanmap={r['id']:r for r in esp}
 for r in ev:
  arrays={};vals={};old=read(FIX/'scores'/(r['id']+'.json'))
  for a in 'NT':
   with np.load(r['arms'][a]['features_path']) as f:aa=f['audio'];vv=f['visual']
   vals[a]=official_scores(vv,aa,r['arms'][a]['L'])
   for g in GEOMS:
    for policy in ['guard20','valid','guard0']:
     x=vals[a][g][policy];y=old['cells'][a][g]['FIXED__FIXED']['policies'][policy]
     if y is None:assert x is None;continue
     for k in ['C','B','D','C_anchor','D_anchor','best_lag','curve']:error=max(error,float(abs(np.asarray(x[k])-np.asarray(y[k])).max()))
   arrays['A','FIXED',a]=aa;arrays['V','FIXED',a]=vv
  write(OUT/'baseline_scores'/(r['id']+'.json.gz'),{'id':r['id'],'speaker':r['speaker'],'eligible':r['guard20_eligible'],'cells':vals})
  if r['id'] in evmap:
   cells=[{'video_state':'FIXED','audio_state':'FIXED','video_source':a,'audio_source':a} for a in 'NT'];z=event.geometry(evmap[r['id']],arrays,cells);s=spanmap[r['id']];out[s['start']:s['stop']]=z
 with np.load(EVENT/'query_metrics.npz') as f:ee=float(abs(out-f['metrics'][:,:,[12,15],:]).max())
 assert error<=1e-9 and ee<=1e-9;save(OUT/'baseline_event.npz',metrics=out)
 write(OUT/'baseline_validation.json',{'status':'PASS','created_epoch':time.time(),'official_all_curves_metrics_max':error,'event_all_queries_max':ee,'all74':True,'ten_views_same_support':True});print('BASELINE_PASS',error,ee,flush=True)
def score():
 p,rows=locked();assert read(OUT/'calibration_gate.json')['passed'] and read(OUT/'baseline_validation.json')['status']=='PASS';assert read(OUT/'gpu_runtime/worker_exit_evaluation.json')['stage_complete'];import torch;torch.set_num_threads(2)
 seal={};ev=[r for r in rows if r['split']=='evaluation'];idx=read(EVENT/'indices.json');em={r['id']:r for r in idx['rows']};spans={r['id']:r for r in idx['spans']};event_result=np.zeros((1411,2,5,2,4))
 with np.load(OUT/'baseline_event.npz') as f:event_result[:,:,0]=f['metrics']
 for r in ev:
  for a in 'NT':
   for cond in CONDS[1:]:
    mp=OUT/'metadata'/r['id']/a/(cond+'.json');meta=read(mp);vp=Path(meta['V']['path']);assert sha(vp)==meta['V']['sha256'];assert meta['fit_sha256']==sha(OUT/'fit.npz');seal[str(vp)]=sha(vp);seal[str(mp)]=sha(mp)
 write(OUT/'feature_seal_evaluation.json',seal);write(OUT/'score_lock.json',{'created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'feature_seal_sha256':sha(OUT/'feature_seal_evaluation.json'),'fit_seal_sha256':sha(OUT/'fit_seal.json')})
 for r in ev:
  vals={'baseline':read(OUT/'baseline_scores'/(r['id']+'.json.gz'))['cells']};original={}
  for a in 'NT':
   with np.load(r['arms'][a]['features_path']) as f:original[a]=f['audio']
  for ci,cond in enumerate(CONDS[1:],1):
   vals[cond]={};arrays={}
   for a in 'NT':
    v=np.load(OUT/'visual'/r['id']/a/(cond+'.npy'));assert v.shape==(r['arms'][a]['frames']-4,1024)
    vals[cond][a]=official_scores(v,original[a],r['arms'][a]['L']);arrays['A','FIXED',a]=original[a];arrays['V','FIXED',a]=v
   if r['id'] in em:
    cells=[{'video_state':'FIXED','audio_state':'FIXED','video_source':a,'audio_source':a} for a in 'NT'];z=event.geometry(em[r['id']],arrays,cells);s=spans[r['id']];event_result[s['start']:s['stop'],:,ci]=z
  write(OUT/'scores'/(r['id']+'.json.gz'),{'id':r['id'],'speaker':r['speaker'],'eligible':r['guard20_eligible'],'cells':vals});print('scored',r['id'],flush=True)
 save(OUT/'event_query_metrics.npz',metrics=event_result);save(OUT/'event_clip_metrics.npz',metrics=event.clipmeans(event_result,idx['spans']))
 write(OUT/'score_complete.json',{'status':'SCORED_PENDING_ANALYSIS','created_epoch':time.time(),'cells':592,'all74':True,'events1411':True})
def contrasts(val):
 # val: [clip,condition,source] with conditions baseline,phoneN,phoneT,globalN,globalT.
 series={};closure=0.
 for ci,c in enumerate(CONDS):
  for ai,a in enumerate('NT'):
   series[f'cell/{c}/{a}']=val[:,ci,ai];series[f'response/{c}/{a}']=val[:,ci,ai]-val[:,0,ai]
  series[f'gap/{c}']=val[:,ci,1]-val[:,ci,0];series[f'gap_change/{c}']=series[f'gap/{c}']-series['gap/baseline'] if ci else np.zeros(len(val))
 for kind,left,right in [('phone',1,2),('global',3,4)]:
  for ai,a in enumerate('NT'):series[f'{kind}/TminusN_template/{a}']=val[:,right,ai]-val[:,left,ai]
  inter=val[:,right,1]-val[:,left,1]-val[:,right,0]+val[:,left,0];series[f'{kind}/source_template_interaction']=inter
  closure=max(closure,float(abs(inter-(series[f'{kind}/TminusN_template/T']-series[f'{kind}/TminusN_template/N'])).max()))
  for ai,a in enumerate('NT'):
   closure=max(closure,float(abs((val[:,right,ai]-val[:,0,ai])-(val[:,left,ai]-val[:,0,ai])-(val[:,right,ai]-val[:,left,ai])).max()))
 for ai,a in enumerate('NT'):
  for source,pc,gc in [('N',1,3),('T',2,4)]:series[f'phone_minus_global/template{source}/{a}']=val[:,pc,ai]-val[:,gc,ai]
  series[f'phone_global_template_interaction/{a}']=series[f'phone/TminusN_template/{a}']-series[f'global/TminusN_template/{a}']
 return series,closure

def analyze():
 p,rows=locked();assert read(OUT/'score_complete.json')['all74'];rs=[read(OUT/'scores'/(r['id']+'.json.gz')) for r in rows if r['split']=='evaluation'];summary={};maxclosure=0.
 for g in GEOMS:
  for policy in ['guard20','valid','guard0']:
   for support in (['common71'] if policy=='guard20' else ['common71','all74']):
    rr=[r for r in rs if r['eligible']] if support=='common71' else rs;names=[r['speaker'] for r in rr]
    for m in METRICS:
     vals=np.array([[[r['cells'][c][a][g][policy][m] for a in 'NT'] for c in CONDS] for r in rr]);series,err=contrasts(vals);maxclosure=max(maxclosure,err)
     for k,v in series.items():summary[f'{g}/{policy}/{support}/{m}/{k}']=F['bootstrap'](v,names)
 idx=read(EVENT/'indices.json');names=[r['speaker'] for r in idx['spans']];es={}
 with np.load(OUT/'event_clip_metrics.npz') as f:cm=f['metrics']
 for gi,g in enumerate(GEOMS):
  for mi,m in enumerate(event.METRICS):
   series,err=contrasts(cm[:,gi,:,:,mi]);maxclosure=max(maxclosure,err)
   for k,v in series.items():es[f'{g}/{m}/{k}']=F['bootstrap'](v,names)
 assert maxclosure<=1e-10;write(OUT/'summary.json.gz',summary);write(OUT/'event_summary.json.gz',es)
 a=summary['raw/guard20/common71/C/phone/TminusN_template/N'];b=summary['raw/guard20/common71/C/response/phoneT/N'];criteria={'phoneT_beats_phoneN_99CI':a['ci99'][0]>0,'phoneT_beats_baseline_99CI':b['ci99'][0]>0,'natural_generator_gain_both':a['ci99'][0]>0 and b['ci99'][0]>0}
 write(OUT/'analysis.json',{'status':'PENDING_INDEPENDENT','created_epoch':time.time(),'criteria':criteria,'triangle_interaction_max':maxclosure,'official_endpoints':len(summary),'event_endpoints':len(es),'resources':limits()});print(json.dumps(criteria),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','fit','baseline','score','analyze']);globals()[a.parse_args().stage]()
