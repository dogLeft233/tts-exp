"""Cal-only fixed absolute RMS GxE; controlled streamed eval retention."""
from pathlib import Path
import argparse, copy, hashlib, json, os, shutil, sys, time, types
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
SOURCE=ROOT/'runs/tts_native_level_generation_cross_20260927'
AUDIT=ROOT/'runs/tts_level_activity_resource_audit_20260927'
OLD=ROOT/'runs/tts_acoustic_generation_cross_20260926'
CAL=ROOT/'runs/tts_native_level_match_calibration_20260927'
COND=['raw','identity','LEVEL','FIXED'];H=['LEVEL','FIXED']
def load(name,path):
    m=types.ModuleType(name);m.__file__=str(ROOT/'scripts/experiments'/Path(path).name)
    exec(compile(Path(path).read_text(),str(path),'exec'),m.__dict__);return m
base=load('fixed_base',OLD/'code_snapshot/tts_acoustic_generation_cross_20260926.py')
base.OUT=OUT;base.COND=COND;base.H=H
cal=load('scalar_cal',CAL/'code_snapshot/tts_native_level_match_calibration.py');spec=cal.parent_module()
read,write,sha=base.read,base.write,base.sha
W2L=base.W2L

def allocated():
    # Reused feature NPZ are hardlinks to sealed older runs, not new blocks.
    seen=set();total=0
    for f in OUT.rglob('*'):
        if not f.is_file():continue
        s=f.stat();key=(s.st_dev,s.st_ino)
        if key in seen:continue
        seen.add(key)
        if s.st_nlink==1:total+=s.st_blocks*512
    return total

def resource_check(p,startup=False):
    rows=base.compute_processes();assert all(r['pid']==os.getpid() for r in rows),('foreign compute',rows)
    used,util=map(int,base.subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if startup:assert used<=128 and util<=10,(used,util)
    free=shutil.disk_usage(OUT).free;new=allocated();assert free>=5<<30,('disk reserve',free)
    assert new<=int(.75*(1<<30)),('new allocation budget',new)
    return {'time':time.time(),'compute':rows,'memory_mib':used,'utilization':util,'free_bytes':free,'new_allocated_bytes':new}
base.resource_check=resource_check

def freeze():
    assert not (OUT/'protocol.json').exists()
    assert read(AUDIT/'final.json')['validation']=='PASS' and read(SOURCE/'final.json')['status']=='concluded'
    p=copy.deepcopy(read(SOURCE/'protocol.json'));p['created_epoch']=time.time();p['conditions']=COND
    target=read(AUDIT/'fixed_target.json');assert target['calibration_arms']==52 and target['eval_used_for_estimator'] is False
    p['fixed_target']={**target,'status':'root_authorized_intervention','source':str(AUDIT/'fixed_target.json'),'source_sha256':sha(AUDIT/'fixed_target.json')}
    p['acoustics']={'operation':'FIXED x*(target/rms(x)), rms float64; each input alone; no clipping/headroom/extra normalization','target_RMS':target['target_RMS'],'silence_gate':'rms>1e-8 else undefined and stop; no deletion','peak_gate':.98,'gates':{'rms_relative':1e-5,'float64_shape_R_db':1e-8,'float64_cosine_error':1e-10,'saved_divide_gain_max':1e-7,'saved_shape_db':1e-4,'saved_R_db':1e-5,'saved_cosine_error':1e-10},'zero_preserved':True}
    p['acoustic_gates']='All100 original inputs, no selection; FIXED finite/same length/peak<=.98/RMS and scalar invariants before GPU or score.'
    p['generation']['identity']='Strict raw/identity/LEVEL cache from concluded LEVEL run, exact original PCM and mel/MFCC model inputs, parameters and model hashes, plus fresh firstcal identity repeat. FIXED always actually new.'
    p['cross']='h=LEVEL,FIXED four q cells; FIXED-minus-LEVEL operation difference all G/E/I/total and N/T/T-N; native gap all views.'
    p['resources']={'lease':'/tmp/tts-exp-gpu.lock','startup':'three snapshots5s apart, no foreigncompute; memory<=128MiB/util<=10','reserve_bytes':5<<30,'new_allocation_max_bytes':int(.75*(1<<30)),'authorized':'root approved controlled streaming 2026-09-27; no old deletion, no CloudFS','estimate_GiB':.70}
    p['retention']={'calibration':'all52FIXED videos plus repeat controls retained','evaluation':'148 new temporary FFV1 videos one at a time; extract original JPEG visual frontend plus audio/z, pixel+file hashes and metadata; atomic feature/metadata seal before remove only own new temp','validation':'firstcal N/T new FIXED replicate through stream route vs retained route pixel/MFCC/features/query exact before evaluation','limitation':'eval videos not retained; creation-time video hashes only, no final video-file hash audit; rebuildable from sealed inputs/model/code/RNG/params'}
    p['limits']='Cal-only single-input RMS operation, same existing100pair cohort not independent data. RMS not LUFS/loudness. FIXED-LEVEL is common operating-level operation difference, no extra explained percentage. Sync-C G/E paths not mouth truth.'
    p.pop('frontend_description',None)
    deps=[AUDIT/'fixed_target.json',AUDIT/'final.json',SOURCE/'protocol.json',SOURCE/'support.json',SOURCE/'cpu_seal.json',SOURCE/'feature_seal_calibration.json',SOURCE/'feature_seal_evaluation.json',CAL/'code_snapshot/tts_native_level_match_calibration.py',CAL/'code_snapshot/check_tts_native_level_match_calibration.py',OLD/'code_snapshot/tts_acoustic_generation_cross_20260926.py']
    p['dependencies'].update({str(f):sha(f) for f in deps})
    files=[Path(__file__),ROOT/'scripts/experiments/check_tts_fixed_level_generation_cross_20260927.py'];p['new_code_hashes']={str(f):sha(f) for f in files}
    for f in files:
        d=OUT/'code_snapshot'/f.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,d)
    write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n');print('FROZEN',sha(OUT/'protocol.json'),flush=True)

def prepare():
    p=base.protocol();assert not (OUT/'acoustic_gate.json').exists()
    sys.path.insert(0,str(W2L));import audio
    from scripts.experiments.static_image_bridge.render_worker import chunk_mels
    target=p['fixed_target']['target_RMS'];source={r['id']:r for r in read(SOURCE/'support.json')};supports=[];metrics=[];invariants=[];gains=[]
    for row in p['rows']:
        sid=row['id'];support=copy.deepcopy(source[sid]);support['arms']={};assert shutil.disk_usage(OUT).free>5<<30
        for a,s in row['audio'].items():
            assert sha(s['path'])==s['sha256'] and sf.info(s['path']).subtype=='PCM_16'
            pcm,sr=sf.read(s['path'],dtype='int16');assert sr==16000 and pcm.ndim==1;x=pcm.astype(float)/32768;r=cal.rms(x);assert r>1e-8
            gain=target/r;y=x*gain;assert abs(y).max()<=.98
            wav=OUT/'audio'/sid/a/'FIXED.wav';wav.parent.mkdir(parents=True,exist_ok=True);sf.write(wav,y,16000,subtype='FLOAT');stored=sf.read(wav,dtype='float64')[0]
            support['arms'][a]={c:copy.deepcopy(source[sid]['arms'][a][c]) for c in ['raw','identity','LEVEL']}
            # Strict original PCM/model frontend cache equality, independently freshly computed.
            for c in ['raw','identity','LEVEL']:
                info=support['arms'][a][c];assert sha(info['waveform'])==info['sha256'] and sha(info['frontend'])==info['frontend_sha256']
                yy=sf.read(info['waveform'],dtype='float64')[0];front=np.load(info['frontend']);mel=audio.melspectrogram(audio.load_wav(info['waveform'],16000)).astype(np.float32);windows,_=base.frontend_windows(yy)
                assert np.array_equal(mel,front['mel']) and np.array_equal(windows,front['windows'])
                if c in ['raw','identity']:assert np.array_equal(x,yy)
            f0=cal.features(spec,x);f1=cal.features(spec,y);fs=cal.features(spec,stored)
            iv={'id':sid,'arm':a,'split':row['split'],'float64':cal.invariant(x,y,f0,f1,gain),'float32':cal.invariant(x,stored,f0,fs,gain)};invariants.append(iv)
            metric={'id':sid,'arm':a,'split':row['split'],'rms':cal.rms(stored),'target_relative':abs(cal.rms(stored)/target-1),'peak':float(max(abs(y).max(),abs(stored).max())),'finite':bool(np.isfinite(stored).all() and np.isfinite(y).all()),'same_length':len(stored)==len(x)};metrics.append(metric)
            mel=audio.melspectrogram(audio.load_wav(str(wav),16000)).astype(np.float32);chunks=chunk_mels(mel,25);windows,_=base.frontend_windows(stored)
            front=OUT/'frontend'/sid/a/'FIXED.npz';front.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(front,mel=mel,windows=windows)
            support['arms'][a]['FIXED']={'waveform':str(wav),'sha256':sha(wav),'samples':len(y),'frames':len(chunks),'audio_windows':len(windows),'L':min(len(chunks),len(y)//640)-5,'frontend':str(front),'frontend_sha256':sha(front)}
            support['arms'][a]['joint_L']=source[sid]['arms'][a]['joint_L'];support['arms'][a]['raw_identity_mel_max']=0.;assert len({support['arms'][a][c]['L'] for c in COND})==1
            gains.append({'id':sid,'arm':a,'split':row['split'],'original_RMS':r,'target_RMS':target,'gain':gain,'gain_dB':float(20*np.log10(gain))})
        supports.append(support);print('prepared',row['split'],sid,flush=True)
    gates={}
    for split in ['calibration','evaluation']:
        mm=[v for v in metrics if v['split']==split];ii=[v for v in invariants if v['split']==split]
        checks={'finite_length_peak':all(r['finite'] and r['same_length'] and r['peak']<=.98 for r in mm),'saved_target':all(r['target_relative']<=1e-5 for r in mm),'float64':all(r['float64']['shape_max_abs_db']<=1e-8 and r['float64']['R_rms_abs_db']<=1e-8 and r['float64']['cosine_error']<=1e-10 and r['float64']['zero_preserved'] for r in ii),'saved':all(r['float32']['divide_gain_max_abs']<=1e-7 and r['float32']['shape_max_abs_db']<=1e-4 and r['float32']['R_rms_abs_db']<=1e-5 and r['float32']['cosine_error']<=1e-10 and r['float32']['zero_preserved'] for r in ii)}
        gates[split]={'passed':all(checks.values()),'checks':checks,'arms':len(mm),'max_peak':max(v['peak'] for v in mm)}
    ids=[r['id'] for r in supports if r['split']=='evaluation' and r['guard20_eligible']];assert ids==read(SOURCE/'acoustic_gate.json')['main_ids'] and len(ids)==71
    for n,v in [('support',supports),('gains',gains),('acoustic_metrics',metrics),('invariants',invariants)]:write(OUT/(n+'.json'),v)
    write(OUT/'acoustic_gate.json',{'passed':all(v['passed'] for v in gates.values()),'splits':gates,'main_ids':ids,'speakers':15});print('ACOUSTIC',gates,flush=True)

def seal():
    p=base.protocol();assert read(OUT/'acoustic_gate.json')['passed'] and read(OUT/'independent_acoustic.json')['status']=='PASS'
    write(OUT/'cpu_seal.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'support_sha256':sha(OUT/'support.json'),'code':p['new_code_hashes'],'acoustic_gate_sha256':sha(OUT/'acoustic_gate.json'),'independent_acoustic_sha256':sha(OUT/'independent_acoustic.json'),'status':'before_features_or_scores'})

def reuse():
    p=base.protocol();sp=read(SOURCE/'protocol.json');assert read(SOURCE/'cal_controls.json')['engineering_passed'];assert p['image']==sp['image']
    for k in ['box','score_box','fps','batch','seed','codec','timing']:assert p['generation'][k]==sp['generation'][k]
    for k in ['lags','distance','k','policies','geometries','endpoint','frontend']:assert p['scoring'][k]==sp['scoring'][k]
    for f,h in sp['dependencies'].items():assert sha(f)==h
    oldrows={r['id']:r for r in sp['rows']};manifest=[]
    for r in read(OUT/'support.json'):
        sid=r['id'];assert next(v for v in p['rows'] if v['id']==sid)['audio']==oldrows[sid]['audio']
        for a in ['N','T']:
            for c in ['raw','identity','LEVEL']:
                jp=SOURCE/'features'/sid/a/(c+'.json');m=read(jp);assert m['audio_frontend']==r['arms'][a][c]
                assert sha(m['features'])==m['sha256'] and sha(m['video']['path'])==m['video']['sha256']
                dest=OUT/'features'/sid/a/(c+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);os.link(m['features'],dest)
                m['features']=str(dest);m['reused_from']=str(jp);m['source_metadata_sha256']=sha(jp);write(dest.with_suffix('.json'),m)
                manifest.append({'id':sid,'arm':a,'condition':c,'reused':True,'source_metadata':str(jp),'source_metadata_sha256':sha(jp),'feature_sha256':sha(dest),'PCM_mel_MFCC_exact':True})
    write(OUT/'reuse_manifest.json',{'rows':manifest,'reuse_count':len(manifest),'source_protocol_sha256':sha(SOURCE/'protocol.json'),'strict_parameters_equal':True});print('REUSED',len(manifest),flush=True)

def produce(split):
    import cv2,torch
    from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model
    from scripts.experiments.static_image_bridge.render_worker import chunk_mels
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine,scoring_environment
    p=base.protocol();lock=read(OUT/'cpu_seal.json')
    for f,h in lock['code'].items():assert sha(f)==h,f
    assert sha(OUT/'support.json')==lock['support_sha256'];assert read(OUT/'acoustic_gate.json')['passed']
    if split=='evaluation':assert (OUT/'cal_controls.json').exists() and read(OUT/'cal_controls.json')['engineering_passed']
    rows=[r for r in read(OUT/'support.json') if r['split']==split];first=min(r['id'] for r in read(OUT/'support.json') if r['split']=='calibration')
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    with base.lease('produce_'+split,p):
        write(OUT/('runtime_'+split+'.json'),scoring_environment());np.savez_compressed(OUT/('rng_before_'+split+'.npz'),torch_cpu=torch.get_rng_state().numpy(),torch_cuda=torch.cuda.get_rng_state().cpu().numpy());model=_load_model(W2L/'checkpoints/wav2lip_gan.pth','cuda');engine=SyncNetEngine(batch_size=32,device='cuda')
        im=p['image'];assert sha(im['path'])==im['sha256'];frame=cv2.imread(im['path']);x1,y1,x2,y2=im['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));mask=face.copy();mask[48:]=0
        it=torch.from_numpy(np.concatenate([mask,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
        try:
            for row in rows:
                sid=row['id']
                for arm in ['N','T']:
                    for c in ['identity','raw',*H]:
                        resource_check(p);info=row['arms'][arm][c];dest=OUT/'features'/sid/arm/c
                        if dest.with_suffix('.json').exists():continue
                        assert sha(info['waveform'])==info['sha256'];assert sha(info['frontend'])==info['frontend_sha256'];front=np.load(info['frontend']);ae=base.audio_forward(engine,front['windows']);chunks=chunk_mels(front['mel'],25)
                        alias=c=='raw' and sid!=first and row['arms'][arm]['raw_identity_mel_max']==0
                        if alias:
                            orig=read(OUT/'features'/sid/arm/'identity.json');v=np.load(orig['features'])['visual'];vm=orig['visual_metadata'];video=orig['video'];z=np.load(orig['features'])['z']
                        else:
                            video,z=base.render_cell(model,chunks,frame,it,im['generation_box'],im['score_box']['box'],(OUT/'videos'/sid/arm/(c+'.avi') if split=='calibration' else OUT/'temporary'/sid/arm/(c+'.avi')),True);v,vm=engine.extract_visual(video['path'])
                        assert video['frames']==info['frames'];assert min(len(v),len(ae))>=row['arms'][arm]['joint_L']
                        dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.pending.npz');np.savez_compressed(tmp,visual=v,audio=ae,z=z);os.replace(tmp,dest.with_suffix('.npz'))
                        metadata={'id':sid,'arm':arm,'condition':c,'features':str(dest.with_suffix('.npz')),'sha256':sha(dest.with_suffix('.npz')),'video':video,'visual_metadata':vm,'raw_alias_new_identity':alias,'audio_frontend':info,'joint_L':row['arms'][arm]['joint_L'],'retained_video':split=='calibration','creation_hash_verified':sha(video['path'])==video['sha256'],'protocol_sha256':sha(OUT/'protocol.json')}
                        saved=np.load(dest.with_suffix('.npz'));assert np.array_equal(saved['visual'],v) and np.array_equal(saved['audio'],ae) and np.array_equal(saved['z'],z)
                        for key in ['visual','audio','z']:assert np.isfinite(saved[key]).all()
                        tmpjson=dest.with_suffix('.pending.json');write(tmpjson,metadata);os.replace(tmpjson,dest.with_suffix('.json'))
                        commit=OUT/'retention_commits'/sid/arm/(c+'.json');write(commit,{'metadata_sha256':sha(dest.with_suffix('.json')),'feature_sha256':sha(dest.with_suffix('.npz')),'video_sha256_at_creation':video['sha256'],'pixel_sha256_at_creation':video['pixel_sha256'],'persisted_and_reopened_equal':True,'video_retained':split=='calibration','time':time.time()})
                        if split=='evaluation':
                            tmpvideo=Path(video['path']);assert tmpvideo.is_relative_to(OUT/'temporary') and tmpvideo.exists();tmpvideo.unlink()
                        resource_check(p)
                    if sid==first:
                        cp=OUT/'controls'/arm;cp.mkdir(parents=True,exist_ok=True);info=row['arms'][arm]['identity'];front=np.load(info['frontend']);chunks=chunk_mels(front['mel'],25)
                        video,z=base.render_cell(model,chunks,frame,it,im['generation_box'],im['score_box']['box'],cp/'repeat.avi');v,vm=engine.extract_visual(video['path'])
                        identity=np.load(OUT/'features'/sid/arm/'identity.npz');original=read(OUT/'features'/sid/arm/'identity.json')
                        pcm=sf.read(info['waveform'],dtype='float64')[0];shift=np.zeros_like(pcm);shift[3200:]=pcm[:-3200];sf.write(cp/'delay5.wav',shift,16000,subtype='FLOAT');w,m=base.frontend_windows(shift);delay=base.audio_forward(engine,w)
                        parent=next(r for r in p['rows'] if r['id']==sid);old_audio,old_meta=engine.extract_audio(parent['audio'][arm]['path']);raw=np.load(OUT/'features'/sid/arm/'raw.npz')
                        old_pcm=sf.read(parent['audio'][arm]['path'],dtype='int16')[0];old_w,_=engine._audio_windows(old_pcm);new_w,_=base.frontend_windows(old_pcm.astype(np.float64)/32768)
                        float_original_audio=base.audio_forward(engine,new_w)
                        np.savez_compressed(cp/'controls.npz',repeat_visual=v,delay_audio=delay,old_audio=old_audio,float_original_audio=float_original_audio)
                        write(cp/'controls.json',{'video':video,'features_sha256':sha(cp/'controls.npz'),'repeat_pixel_equal':video['pixel_sha256']==original['video']['pixel_sha256'],'repeat_feature_max':float(np.max(abs(v-identity['visual']))),'old_float_mfcc_max':float(np.max(abs(old_w-new_w))),'old_float_audio_max':float(np.max(abs(old_audio-float_original_audio))),'original_float_waveform_max':float(np.max(abs(old_pcm.astype(np.float32)/32768-old_pcm.astype(np.float64)/32768))),'delay_waveform_sha256':sha(cp/'delay5.wav')})
                    if sid==first:
                        fi=row['arms'][arm]['FIXED'];ff=np.load(fi['frontend']);chunks=chunk_mels(ff['mel'],25)
                        full=read(OUT/'features'/sid/arm/'FIXED.json');fullf=np.load(full['features'])
                        tv=OUT/'temporary'/'stream_control'/arm/'FIXED.avi';sv,sz=base.render_cell(model,chunks,frame,it,im['generation_box'],im['score_box']['box'],tv,True);vv,vm=engine.extract_visual(sv['path']);aa=base.audio_forward(engine,ff['windows'])
                        wx,_=base.frontend_windows(sf.read(fi['waveform'],dtype='float64')[0]);checks={'pixel_exact':sv['pixel_sha256']==full['video']['pixel_sha256'],'visual_exact':bool(np.array_equal(vv,fullf['visual'])),'audio_exact':bool(np.array_equal(aa,fullf['audio'])),'z_exact':bool(np.array_equal(sz,fullf['z'])),'MFCC_exact':bool(np.array_equal(wx,ff['windows'])),'query_exact':sv['frames']==full['video']['frames'] and fi['L']==row['arms'][arm]['joint_L']}
                        sp=OUT/'controls'/arm/'streamed_control.npz';np.savez_compressed(sp,visual=vv,audio=aa,z=sz);ss=np.load(sp);checks['persistence_exact']=all(np.array_equal(ss[k],fullf[k]) for k in ['visual','audio','z'])
                        write(OUT/'controls'/arm/'streaming.json',{'passed':all(checks.values()),'checks':checks,'video_at_creation':sv,'features_sha256':sha(sp),'full_metadata_sha256':sha(OUT/'features'/sid/arm/'FIXED.json'),'retained_video':False});assert all(checks.values());assert sha(tv)==sv['sha256'];tv.unlink()
                print('produced',split,sid,flush=True)
        finally:engine.close()
    write(OUT/('gpu_release_'+split+'.json'),{'time':time.time(),'pid':os.getpid(),'lease_released':True})

    if split=='calibration':
        checks={a:read(OUT/'controls'/a/'controls.json') for a in ['N','T']};passed=all(v['repeat_pixel_equal'] and v['repeat_feature_max']==0 for v in checks.values())
        write(OUT/'reuse_repeat_validation.json',{'passed':passed});assert passed
        assert all(read(OUT/'controls'/a/'streaming.json')['passed'] for a in ['N','T'])

def analyze():
    rows=[read(f) for f in sorted((OUT/'scores').glob('*.json')) if read(f)['split']=='evaluation'];assert len(rows)==74
    output={};per=[];closure=0.
    for geometry in ['raw','unit']:
      for policy in ['guard20','valid','guard0']:
       for support in (['common71'] if policy=='guard20' else ['common71','all74']):
        use=[r for r in rows if r['eligible']] if support=='common71' else rows;groups=[r['speaker'] for r in use];view={}
        for metric in ['C','B','D','C_anchor','D_anchor','best_lag']:
            native=np.array([r['cells']['T'][geometry]['identity__identity']['policies'][policy][metric]-r['cells']['N'][geometry]['identity__identity']['policies'][policy][metric] for r in use]);detail={'baseline_T_minus_N':base.bootstrap(native,groups),'conditions':{},'contrasts':{}};armvalues={}
            for h in H:
                armvalues[h]={a:[] for a in ['N','T']}
                for r in use:
                    for a in ['N','T']:
                        cells=r['cells'][a][geometry];q=[cells[k]['policies'][policy][metric] for k in ['identity__identity',h+'__identity','identity__'+h,h+'__'+h]];e=base.effects(q);closure=max(closure,abs(e['G']+e['E']+e['I']-e['total']));armvalues[h][a].append(e)
                        per.append({'id':r['id'],'speaker':r['speaker'],'geometry':geometry,'policy':policy,'support':support,'metric':metric,'arm':a,'condition':h,'q00':q[0],'q10':q[1],'q01':q[2],'q11':q[3],**e})
                detail['conditions'][h]={}
                for effect in ['G','E','I','total']:
                    n=np.array([r[effect] for r in armvalues[h]['N']]);t=np.array([r[effect] for r in armvalues[h]['T']]);detail['conditions'][h][effect]={'N':base.bootstrap(n,groups),'T':base.bootstrap(t,groups),'T_minus_N':base.bootstrap(t-n,groups)}
                detail['conditions'][h]['processed_T_minus_N']=base.bootstrap(native+np.array([r['total'] for r in armvalues[h]['T']])-np.array([r['total'] for r in armvalues[h]['N']]),groups)
            for name,weights in {'FIXED_minus_LEVEL':{'FIXED':1,'LEVEL':-1}}.items():
                detail['contrasts'][name]={}
                for effect in ['G','E','I','total']:
                    values={a:sum(w*np.array([r[effect] for r in armvalues[h][a]]) for h,w in weights.items()) for a in ['N','T']};n,t=values['N'],values['T'];detail['contrasts'][name][effect]={'N':base.bootstrap(n,groups),'T':base.bootstrap(t,groups),'T_minus_N':base.bootstrap(t-n,groups)}
            view[metric]=detail
        output[geometry+'/'+policy+'/'+support]=view
    write(OUT/'summary.json',{'results':output,'maximum_algebra_closure_error':closure,'status':'complete','cal_controls':read(OUT/'cal_controls.json')});write(OUT/'effects.json',per)
    print(json.dumps(output['raw/guard20/common71']['C'],indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','seal','reuse','produce','score','analyze']);ap.add_argument('--split',choices=['calibration','evaluation']);args=ap.parse_args()
    if args.stage=='score':base.score(args.split)
    elif args.stage=='produce':produce(args.split)
    else:globals()[args.stage]()
