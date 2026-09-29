"""Input-only preparation and frozen gates for natural80 source-direction transfer."""
from pathlib import Path
from decimal import Decimal
import argparse, gzip, hashlib, importlib.metadata, json, math, os, re, shutil, subprocess, sys, time, zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_fixed_generator_shift_holdout80_20260927'
OLD=ROOT/'runs/tts_fixed_natural_holdout80_20260927'
FIT=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927'
SHIFT=ROOT/'runs/tts_fixed_generator_shift_cross_20260927'
CLOUD=ROOT/'runs/aishell1_qwen_mfa_linear_n100_20260816/02_mfa_mandarin341_ready'
FIX=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
DATA=ROOT/'data/native_mechanism_holdout_20260926'
MFA=Path('/home/wjj/miniconda3/envs/mfa3/bin/mfa')
DICT=Path('/home/wjj/Documents/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict')
ACOUSTIC=Path('/home/wjj/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip')
PY=Path('/home/wjj/.venvs/syncnet/bin/python')
TMP=Path('/dev/shm/tts_fixed_generator_shift_holdout80_20260927')
MFATMP=Path('/dev/shm/tts_fixed_generator_shift_holdout80_mfa_20260927')
CAP=176*2**20;AUDIT=16*2**20;FLOOR=4*2**30
CONDS=['phone_plus','phone_minus','global_plus','global_minus']
EXCLUDED={'','sil','sp','spn','<unk>','<eps>','silence'}
from scripts.experiments.tts_fixed_generator_phone_stream_20260927 import sha,ah
def read(p):
 p=Path(p);return json.loads(gzip.open(p,'rt').read() if p.suffix=='.gz' else p.read_text())
def allocated(p):return sum(x.stat().st_blocks*512 for x in Path(p).rglob('*') if x.is_file() and not x.is_symlink()) if Path(p).exists() else 0
def limits(extra=0):
 used=allocated(OUT);free=shutil.disk_usage(OUT).free
 assert used+extra<=CAP and free-max(0,CAP-used)-AUDIT>=FLOOR,('resource',used,extra,free)
 mt=allocated(MFATMP);gt=allocated(TMP);assert mt<=2*2**30 and gt<=96*2**20
 assert not (mt and gt),'MFA/GPU tmp must be serial'
 av=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'));assert av>=4*2**30
 return {'allocated':used,'free':free,'remaining_commitment':max(0,CAP-used)+AUDIT,'MFA_tmp':mt,'GPU_tmp':gt,'MemAvailable':av}
def write(p,x):
 p=Path(p);data=json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n'
 if p.suffix=='.gz':data=gzip.compress(data,mtime=0)
 limits(math.ceil(len(data)/4096)*4096);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
 limits()
def runtime():
 import torch,cv2
 return {'python':sys.executable,'version':sys.version,'versions':{k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','librosa','python_speech_features','soundfile']},'cv2':cv2.__version__,'cuda':torch.version.cuda}
def mfa_runtime():
 command="import sys,importlib.metadata,json;print(json.dumps({'python':sys.version,'mfa':importlib.metadata.version('Montreal-Forced-Aligner')}))"
 env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
 return json.loads(subprocess.check_output([str(MFA.parent/'python'),'-B','-c',command],env=env,text=True))
def starts(M):
 out=[];i=0
 while int(i*80/25)+16<=M:out.append(int(i*80/25));i+=1
 out.append(M-16);return out
def prepare():
 assert not (OUT/'rows.json').exists();assert str(Path(sys.executable))==str(PY)
 assert read(SHIFT/'final.json')['status'].lower() in ('concluded','complete','pass')
 old=read(OLD/'protocol.json');assert runtime()['versions']==old['runtime']['versions']
 rows=[];bind={}
 def b(p,expected=None):
  p=Path(p);h=sha(p);assert expected is None or h==expected,str(p);bind[str(p.resolve())]=h;return h
 support=read(OLD/'support.json');assert len(support)==80 and len({r['speaker'] for r in support})==40
 for r in support:
  sid=r['id'];q=r['FIXED'];f=OLD/'features'/sid/'FIXED.npz';m=f.with_suffix('.json');pm=read(m);b(f,pm['sha256']);b(m);b(q['waveform'],q['sha256']);b(q['frontend'],q['frontend_sha256']);b(r['source']['path'],r['source']['sha256'])
  lab=DATA/'transcripts'/(sid+'.txt');b(lab);text=''.join(lab.read_text().split());assert text and all('\u3400'<=c<='\u9fff' for c in text),(sid,'nonCJK frozen transcript')
  with np.load(q['frontend']) as z:ix=starts(z['mel'].shape[1]);assert len(ix)==q['frames']
  a={'features_path':str(f),'features_sha256':sha(f),'metadata_path':str(m),'metadata_sha256':sha(m),'frontend_path':q['frontend'],'frontend_sha256':q['frontend_sha256'],'waveform_path':q['waveform'],'waveform_sha256':q['sha256'],'frames':q['frames'],'samples':q['samples'],'L':r['L'],'mel_starts':ix}
  rows.append({'id':sid,'speaker':r['speaker'],'split':'evaluation','guard20_eligible':True,'L':r['L'],'source':r['source'],'transcript_path':str(lab),'transcript_sha256':sha(lab),'lab':' '.join(text),'arms':{'N':a}})
 records=read(CLOUD/'mfa_summary.json')['records'];cal=[]
 for r in read(FIX/'support.json'):
  if r['split']!='calibration':continue
  s=records[r['id']];raw=s['natural'];q=r['arms']['N']['FIXED'];b(raw['audio'],raw['audio_sha256']);b(raw['textgrid'],raw['textgrid_sha256']);b(q['frontend'],q['frontend_sha256'])
  with np.load(q['frontend']) as z:ix=starts(z['mel'].shape[1]);assert len(ix)==q['frames']
  cal.append({'id':r['id'],'speaker':r['speaker'],'source':{'path':raw['audio'],'sha256':raw['audio_sha256']},'lab':s['cleaned_lab_text'],'samples':q['samples'],'mel_starts':ix,'old_TextGrid':raw['textgrid'],'old_TextGrid_sha256':raw['textgrid_sha256']})
 assert len(cal)==26
 fm=read(FIT/'fit.json');ff=np.load(FIT/'fit.npz');fold=fm['folds'].index('S0912');baseids=fm['training_ids']['S0912'];assert len(baseids)==20
 for name in ['S0913','S0915']:
  j=fm['folds'].index(name);assert fm['training_ids'][name]==baseids and np.array_equal(ff['raw_mu'][fold],ff['raw_mu'][j]) and np.array_equal(ff['speaker_counts'][fold],ff['speaker_counts'][j])
 with zipfile.ZipFile(ACOUSTIC) as z:
  meta=json.loads(z.read('mandarin_mfa/meta.json'));assert set(fm['labels'])<=set(meta['phones']);proof={}
  for name in ['final.mdl','lda.mat','tree','phones.txt']:
   oldfile=CLOUD/'mfa_runtime/extracted_models/acoustic/mandarin_mfa_acoustic'/name
   h=hashlib.sha256(z.read('mandarin_mfa/'+name)).hexdigest();assert b(oldfile)==h;proof[name]=h
 for p in [OLD/'protocol.json',OLD/'support.json',OLD/'final.json',OLD/'feature_seal.json',FIT/'fit.npz',FIT/'fit.json',FIT/'fit_seal.json',SHIFT/'protocol.json',SHIFT/'final.json',CLOUD/'mfa_summary.json',DICT,ACOUSTIC,MFA,FIX/'support.json']:b(p)
 for label in fm['labels']:assert int(ff['speaker_counts'][fold,fm['labels'].index(label)])>0
 write(OUT/'rows.json',rows);write(OUT/'cal_rows.json',cal)
 write(OUT/'asset_binding.json',{'created_epoch':time.time(),'bindings':bind,'MFA_inventory':meta['phones'],'template_labels':fm['labels'],'historical_acoustic_members_exact':proof,'dictionary_historical_scope':'current dictionary hash bound; old cloud summary names same dictionary, no invented historical zip/dictionary checksum','pooled_fold':'S0912','runtime':runtime(),'no_new_alignment_forward_or_scores':True,'resource':limits()})
 print('PREPARED',len(rows),len(cal),flush=True)
def freeze():
 assert not (OUT/'protocol.json').exists();b=read(OUT/'asset_binding.json');deps=dict(b['bindings'])
 oldp=read(OLD/'protocol.json')
 for f,h in {**oldp['dependencies'],**oldp['new_code_hashes']}.items():assert sha(f)==h,f;deps[str(Path(f).resolve())]=h
 fitting_rows=read(FIT/'rows.json');assert not ({r['speaker'] for r in read(OUT/'rows.json')}&{r['speaker'] for r in fitting_rows})
 deps[str(FIT/'rows.json')]=sha(FIT/'rows.json')
 import soundfile as sf
 for r in read(OUT/'rows.json')+read(OUT/'cal_rows.json'):
  a=r['arms']['N'] if 'arms' in r else r;info=sf.info(r['source']['path'])
  assert info.samplerate==16000 and info.channels==1 and info.frames==a['samples'],('source_clock',r['id'])
 from scripts.experiments.tts_native_gain_attribution import config as sc
 files=[Path(__file__),ROOT/'scripts/experiments/tts_fixed_generator_shift_holdout80_gpu_20260927.py',ROOT/'scripts/experiments/tts_fixed_generator_shift_holdout80_scores_20260927.py',ROOT/'scripts/experiments/tts_fixed_generator_shift_ops_20260927.py',ROOT/'scripts/experiments/tts_fixed_generator_phone_stream_20260927.py',ROOT/'scripts/experiments/masked_tts_tfg_probe/direct_mel.py',ROOT/'scripts/experiments/static_image_bridge/render_worker.py',ROOT/'scripts/experiments/static_image_bridge/score_worker.py']
 files += [ROOT/'scripts/experiments/tts_native_gain_attribution'/n for n in ['syncnet.py','config.py','audio.py','common.py']]
 files += [ROOT/'third_party/Wav2Lip'/n for n in ['audio.py','hparams.py','models/__init__.py','models/conv.py','models/wav2lip.py','checkpoints/wav2lip_gan.pth']]
 files += [ROOT/'third_party/syncnet_python'/n for n in ['SyncNetModel.py','SyncNetInstance.py','data/syncnet_v2.model']]
 files += [Path(sc.FFMPEG),Path(sc.FFPROBE),OLD/'code_snapshot/tts_fixed_natural_holdout80_20260927.py',ROOT/'runs/tts_acoustic_generation_cross_20260926/code_snapshot/tts_acoustic_generation_cross_20260926.py']
 files += [OLD/'summary.json']+sorted((OLD/'scores').glob('*.json'))
 import python_speech_features.base as fb,python_speech_features.sigproc as fs
 files += [Path(fb.__file__),Path(fs.__file__)]
 for n in ['rows.json','cal_rows.json','asset_binding.json','protocol.md','gpu_supervisor.py']:files.append(OUT/n)
 for p in files:deps[str(p.resolve())]=sha(p)
 c={'runtime':runtime(),'image':read(OLD/'protocol.json')['image'],'checkpoint':str(ROOT/'third_party/Wav2Lip/checkpoints/wav2lip_gan.pth'),'ffmpeg':str(Path(sc.FFMPEG).resolve()),'ffprobe':str(Path(sc.FFPROBE).resolve()),'tmp_cap_bytes':96*2**20,'own_cap_bytes':CAP,'other_reserved_bytes':AUDIT,'floor_bytes':FLOOR,'first_cal':read(OUT/'rows.json')[0]['id']}
 deps[c['image']['path']]=sha(c['image']['path'])
 p={'status':'FROZEN_BEFORE_ALIGNMENT_NATIVE_Z_NEW_FORWARD_SCORE','created_epoch':time.time(),'dependencies':deps,'gpu_config':c,'conditions':CONDS,'primary':'global_plus-baseline and global_plus-global_minus each C,C_anchor rawguard20 allfour99CI lower>0','pooled_fold':'S0912','MFA':{'binary':str(MFA),'dictionary':str(DICT),'acoustic':str(ACOUSTIC),'version':'3.4.1','jobs':2,'seed':0,'group':'speaker independently; cal before80','excluded':sorted(EXCLUDED),'minimum_active_frames_per_clip':1,'minimum_template_coverage':.8,'clock_tolerance_samples':1},'resources':{'main':CAP,'audit':AUDIT,'floor':FLOOR,'GPU_cumulative_seconds_including_live':5400,'MFA_cumulative_seconds':7200,'MFA_tmp':2*2**30,'GPU_tmp':96*2**20,'minimum_MemAvailable':4*2**30,'V_allocated_bound':146423808,'z_allocated_bound':19034112,'calV_allocated_bound':2359296},'statistics':{'draws':20000,'seed':20260926,'CI':[.95,.99],'unit':'40speaker equal means of2clip paired effects','FWER':False},'scoring':read(OLD/'protocol.json')['scoring'],'BM_title':'80句自然语音生成器来源方向正反号迁移 2026-09-27'}
 p['MFA']['runtime']=mfa_runtime();assert p['MFA']['runtime']['mfa']=='3.4.1'
 write(OUT/'protocol.json',p);write(OUT/'seal.json',{'protocol_sha256':sha(OUT/'protocol.json'),'no_new_outputs':True})
 for f in files[:5]:
  dest=OUT/'code_snapshot'/f.name;data=f.read_bytes();limits(math.ceil(len(data)/4096)*4096);dest.parent.mkdir(exist_ok=True)
  with dest.open('xb') as stream:stream.write(data)
 print('FROZEN',sha(OUT/'protocol.json'),flush=True)
def locked():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256']
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 v=read(OUT/'reviewer_pass.json');assert v['status']=='PASS' and v['protocol_sha256']==sha(OUT/'protocol.json') and sha(v['receipt'])==v['receipt_sha256']
 limits();return p,read(OUT/'rows.json')
def parse_grid(path,samples,ix,inventory,labels):
 s=Path(path).read_text();marker='name = "phones"';assert s.count(marker)==1;s=s.split(marker,1)[1].split('item [',1)[0]
 values=re.findall(r'intervals\s*\[\s*\d+\s*\]\s*:?\s*xmin\s*=\s*([0-9.eE+-]+)\s*xmax\s*=\s*([0-9.eE+-]+)\s*text\s*=\s*"([^"]*)"',s,re.S);assert values
 tokens=[];prev=Decimal(0);duration=Decimal(samples)/Decimal(16000)
 for i,(lo,hi,label) in enumerate(values):
  a,b=Decimal(lo),Decimal(hi);assert a.is_finite() and b.is_finite() and a==prev and b>=a and a>=0 and b<=duration+Decimal(1)/16000
  usable=b>a and label.strip().casefold() not in EXCLUDED
  if usable:assert label in inventory,('label_not_in_inventory',label)
  tokens.append({'lo_decimal':lo,'hi_decimal':hi,'label':label,'usable':usable});prev=b
 assert abs(prev-duration)<=Decimal(1)/16000
 phone=[];mask=[]
 for start in ix:
  t=Decimal(2*start+15)/160;hit=[x for x in tokens if Decimal(x['lo_decimal'])<=t<Decimal(x['hi_decimal'])];assert len(hit)<=1
  label=hit[0]['label'] if hit and hit[0]['usable'] else None;phone.append(label);mask.append(label is not None)
 active=sum(mask);matched=sum(x in labels for x in phone if x is not None)
 return {'tokens':tokens,'phone_labels':phone,'speech_mask':mask,'active':active,'matched':matched,'coverage':matched/active if active else None,'unknown_intervals':sum(x['label'].casefold() in {'spn','<unk>'} for x in tokens)}
def align(split):
 p,rows=locked();assert split in ['cal','evaluation'];assert not allocated(TMP)
 assert mfa_runtime()==p['MFA']['runtime']
 if split=='evaluation':assert read(OUT/'mfa_cal_gate.json')['status']=='PASS'
 targets=read(OUT/'cal_rows.json') if split=='cal' else rows
 root=MFATMP/split;assert not root.exists();root.mkdir(parents=True);summaries=[];begin=time.time();logs=0
 prior=read(OUT/'mfa_cal_gate.json').get('elapsed_seconds',0.) if split=='evaluation' else 0.
 config='auto_server: false\nuse_postgres: false\nnum_jobs: 2\nblas_num_threads: 1\nseed: 0\nsingle_speaker: true\nclean: true\nquiet: false\nprofiles: {}\n'
 for speaker in sorted({r['speaker'] for r in targets}):
  limits();group=[r for r in targets if r['speaker']==speaker];work=root/speaker;inp=work/'input';inp.mkdir(parents=True);runtime_root=work/'runtime';runtime_root.mkdir();(runtime_root/'global_config.yaml').write_text(config)
  for r in group:
   src=r['source'];assert sha(src['path'])==src['sha256'];(inp/(r['id']+'.wav')).symlink_to(src['path']);(inp/(r['id']+'.lab')).write_text(r['lab'])
  out=OUT/'textgrids'/split/speaker;env=dict(os.environ);env.update(MFA_ROOT_DIR=str(runtime_root),TMPDIR=str(work),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');env['PATH']=str(MFA.parent)+os.pathsep+env['PATH']
  cmd=[str(MFA),'align',str(inp),str(DICT),str(ACOUSTIC),str(out),'--single_speaker','--clean','--num_jobs','2','--temporary_directory',str(work/'temporary')]
  log=work/'align.log'
  with log.open('xb') as stream:
   child=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while child.poll() is None:
     limits();assert prior+time.time()-begin<7200 and log.stat().st_size<=2**20;time.sleep(1)
    assert child.returncode==0,('MFA_FAILED',speaker,child.returncode)
   except BaseException:
    import signal
    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20)
    raise
  data=log.read_bytes();logs+=len(data);assert logs<=8*2**20
  write(OUT/'mfa_logs'/split/(speaker+'.json.gz'),{'command':cmd,'runtime_config':config,'env':{k:env[k] for k in ['MFA_ROOT_DIR','TMPDIR','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},'exit':0,'log':data.decode(errors='replace')})
  for r in group:
   grid=out/(r['id']+'.TextGrid');a=r if split=='cal' else r['arms']['N'];summary=parse_grid(grid,a['samples'],a['mel_starts'],read(OUT/'asset_binding.json')['MFA_inventory'],read(OUT/'asset_binding.json')['template_labels']);summaries.append({'id':r['id'],'speaker':r['speaker'],'TextGrid':str(grid),'TextGrid_sha256':sha(grid),**summary})
  # Only this new transient directory is removed; durable grids/log/inputs remain bound.
  shutil.rmtree(work);limits();print('MFA',split,speaker,flush=True)
 active=sum(x['active'] for x in summaries);matched=sum(x['matched'] for x in summaries);coverage=matched/active if active else 0
 passed=len(summaries)==len(targets) and all(x['active']>=1 for x in summaries) and coverage>=.8
 write(OUT/('mfa_'+split+'_gate.json'),{'status':'PASS' if passed else 'INPUT_ENGINEERING_FAILED','created_epoch':time.time(),'elapsed_seconds':time.time()-begin,'cumulative_seconds':prior+time.time()-begin,'source_count':len(targets),'rows':summaries,'active':active,'matched':matched,'coverage':coverage,'no_scoring_or_output_selection':True});root.rmdir()
 assert passed,'MFA input gate failed; no parameter/selection changes'
 if split=='evaluation':
  by={r['id']:r for r in summaries};out=[]
  for r in rows:
   rr=json.loads(json.dumps(r));g=by[r['id']];rr['arms']['N'].update({k:g[k] for k in ['phone_labels','speech_mask','TextGrid','TextGrid_sha256']});out.append(rr)
  write(OUT/'input_rows.json',out);write(OUT/'input_seal.json',{'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'files':{str(OUT/n):sha(OUT/n) for n in ['input_rows.json','mfa_cal_gate.json','mfa_evaluation_gate.json']}})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','freeze','align_cal','align_evaluation']);s=a.parse_args().stage
 if s.startswith('align_'):align(s[len('align_'):])
 else:globals()[s]()
