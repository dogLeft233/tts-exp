"""Frozen LEVEL/EQ_LEVEL GxE experiment, strict reusable frontend cache."""
from pathlib import Path
import argparse, copy, hashlib, importlib.metadata, inspect, json, os, shutil, sys, time, types
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_native_level_generation_cross_20260927'
CAL=ROOT/'runs/tts_native_level_match_calibration_20260927'
OLD=ROOT/'runs/tts_acoustic_generation_cross_20260926'
SPEC=ROOT/'runs/tts_native_spectrum_generation_cross_20260926'
COND=['raw','identity','EQ','LEVEL','EQ_LEVEL']; H=['EQ','LEVEL','EQ_LEVEL']
def load(name,path):
    m=types.ModuleType(name);m.__file__=str(ROOT/'scripts/experiments'/Path(path).name)
    exec(compile(Path(path).read_text(),str(path),'exec'),m.__dict__);return m
base=load('sealed_gxe',OLD/'code_snapshot/tts_acoustic_generation_cross_20260926.py')
base.OUT=OUT;base.COND=COND;base.H=H
cal=load('sealed_level',CAL/'code_snapshot/tts_native_level_match_calibration.py')
spec=cal.parent_module()
read,write,sha=base.read,base.write,base.sha

def resource_check(p,startup=False):
    rows=base.compute_processes();assert all(r['pid']==os.getpid() for r in rows),('foreign compute',rows)
    used,util=map(int,base.subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if startup:assert used<=128 and util<=10,(used,util)
    free=shutil.disk_usage(OUT).free;assert free>=5<<30,('disk reserve',free)
    return {'time':time.time(),'compute':rows,'memory_mib':used,'utilization':util,'free_bytes':free}
base.resource_check=resource_check

def freeze():
    import python_speech_features.base as psf
    import python_speech_features.sigproc as sigproc
    assert not (OUT/'protocol.json').exists() and read(CAL/'final.json')['independent_status']=='PASS'
    p=copy.deepcopy(read(OLD/'protocol.json'));p['created_epoch']=time.time();p['conditions']=COND
    p['acoustics']=read(CAL/'protocol.json');p['acoustic_gates']='Exactly LEVEL calibration numerical/invariance gates for all26cal and74eval before scoring. EQ spectrum/R gates inherited and checked separately; no deletion/retuning.'
    p['generation']['identity']='Strict cached identity/raw/EQ only when original source and all generation/scoring parameters match plus actual mel and MFCC windows bitwise equal. Firstcal N/T fresh repeat verifies pixels/features. LEVEL/EQ_LEVEL always new.'
    p['cross']='h in EQ/LEVEL/EQ_LEVEL; q00 id/id,q10 h/id,q01 id/h,q11 h/h; G,E,I,total; full contrasts EQ_LEVEL-EQ-LEVEL, EQ_LEVEL-EQ and LEVEL-identity, N/T/T-N'
    p['statistics']['support']='raw/unit guard20/valid/guard0 main common71/15speakers; valid/guard0 all74 separately'
    p['resources']={'startup':'three5s-spaced low util snapshots; memory<=128MiB; no foreign compute; only verified graphical services may exist','lease':'/tmp/tts-exp-gpu.lock','reserve_bytes':5<<30,'expected_new_videos':402,'estimated_video_GiB':2.3,'estimated_total_GiB':3.3}
    p['limits']='RMS matching not LUFS/perceived loudness; same-pair test-time operation, not deployable independent single input. G/E are Sync-C paths, not mouth truth. EQxLEVEL algebra interaction not isolated physiological interaction.'
    p['frontend_description']={'MFCC':'actual installed default mfcc appendEnergy=True; c0 gain ideal2ln(g), c1:12 unchanged only nonzero energy/filterbank frames; exact zero/floor exceptions separate. Save ideal float64 and actual saved FLOAT32 coefficients differences, supports, masks.', 'Wav2Lip':'actual hp gain shift on unclipped/unfloored mel; report floor/clip fractions, residuals; no new scoring condition or feature selection','package_version':importlib.metadata.version('python_speech_features'),'mfcc_signature':str(inspect.signature(psf.mfcc))}
    p['dependencies'].update({str(f):sha(f) for f in [CAL/'protocol.json',CAL/'final.json',CAL/'code_snapshot/tts_native_level_match_calibration.py',CAL/'code_snapshot/check_tts_native_level_match_calibration.py',OLD/'code_snapshot/tts_acoustic_generation_cross_20260926.py',SPEC/'protocol.json',SPEC/'cpu_seal.json',SPEC/'support.json',Path(inspect.getfile(psf)),Path(inspect.getfile(sigproc))]})
    files=[Path(__file__),ROOT/'scripts/experiments/check_tts_native_level_generation_cross_20260927.py']
    p['new_code_hashes']={str(f):sha(f) for f in files}
    for f in files:
        dest=OUT/'code_snapshot'/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
    write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)

def mfcc_description(x,y,g,path):
    import python_speech_features as psf
    from python_speech_features import sigproc
    def zeros(a):
        frames=sigproc.framesig(sigproc.preemphasis(a*32768,.97),400,160)
        power=sigproc.powspec(frames,512);fb=power@psf.get_filterbanks(26,512,16000,0,None).T
        return power.sum(1)==0,np.any(fb==0,axis=1)
    old=psf.mfcc(x*32768,16000);new=psf.mfcc(y*32768,16000);delta=new-old
    ez,fz=zeros(x);ez2,fz2=zeros(y);exception=ez|fz|ez2|fz2;good=~exception
    error=delta.copy();error[:,0]-=2*np.log(g)
    np.savez_compressed(path,coefficient_delta=delta,expected_shift=np.r_[2*np.log(g),np.zeros(12)],energy_zero=ez,filter_zero=fz,processed_energy_zero=ez2,processed_filter_zero=fz2,exception=exception)
    return {'frames':len(old),'nonexception_frames':int(good.sum()),'exception_fraction':float(exception.mean()),'zero_energy_fraction':float(ez.mean()),'zero_filter_fraction':float(fz.mean()),'c0_expected':float(2*np.log(g)),'nonexception_max_error_by_coefficient':np.max(abs(error[good]),axis=0).tolist() if good.any() else None,'exception_max_error_by_coefficient':np.max(abs(error[exception]),axis=0).tolist() if exception.any() else None,'artifact':str(path),'sha256':sha(path)}

def mel_description(audio,x,y,g):
    hp=audio.hp
    def f(a):
        amp=audio._linear_to_mel(abs(audio._stft(audio.preemphasis(a,hp.preemphasis,hp.preemphasize))))
        floor=np.exp(hp.min_level_db/20*np.log(10));db=audio._amp_to_db(amp)-hp.ref_level_db
        unnorm=(2*hp.max_abs_value)*(db-hp.min_level_db)/(-hp.min_level_db)-hp.max_abs_value
        return audio._normalize(db),amp<=floor,abs(unnorm)>=hp.max_abs_value
    a,fa,ca=f(x);b,fb,cb=f(y);good=~(fa|fb|ca|cb);shift=20*np.log10(g)*2*hp.max_abs_value/(-hp.min_level_db)
    return {'expected_unclipped_shift':float(shift),'actual_delta_median':float(np.median((b-a)[good])) if good.any() else None,'unclipped_residual_max':float(abs((b-a)[good]-shift).max()) if good.any() else None,'unclipped_fraction':float(good.mean()),'base_floor_fraction':float(fa.mean()),'processed_floor_fraction':float(fb.mean()),'base_clip_fraction':float(ca.mean()),'processed_clip_fraction':float(cb.mean()),'cells':a.size}

def prepare():
    p=base.protocol();assert not (OUT/'acoustic_gate.json').exists()
    for f,h in p['new_code_hashes'].items():assert sha(f)==h
    sys.path.insert(0,str(base.W2L));import audio
    from scripts.experiments.static_image_bridge.render_worker import chunk_mels
    supports=[];metrics=[];pairs=[];invariants=[];diagnostics=[];reuse_candidates=[]
    old_support={r['id']:r for r in read(SPEC/'support.json')}
    for row in p['rows']:
        assert shutil.disk_usage(OUT).free>5<<30
        sid=row['id'];xs={}
        for a,s in row['audio'].items():
            assert sha(s['path'])==s['sha256'] and sf.info(s['path']).subtype=='PCM_16'
            pcm,sr=sf.read(s['path'],dtype='int16');assert sr==16000 and pcm.ndim==1
            xs[a]=pcm.astype(float)/32768
        rs={a:cal.rms(x) for a,x in xs.items()};assert min(rs.values())>1e-8,('undefined silence',sid)
        shapes={a:spec.spectral_vectors(x)[0]['smooth_mean_log_shape_db'] for a,x in xs.items()};target=(shapes['N']+shapes['T'])/2
        eq={a:spec.eq(x,np.clip(target-shapes[a],-6,6))[0] for a,x in xs.items()}
        m0=float(np.sqrt(rs['N']*rs['T']));caps={a+'/'+k:.98*rs[a]/float(abs(y).max()) for a in xs for k,y in [('raw',xs[a]),('EQ',eq[a])]};mcap=min(caps.values());m=min(m0,mcap);g={a:m/rs[a] for a in xs}
        pair={'id':sid,'speaker':row['speaker'],'split':row['split'],'m0':m0,'mcap':mcap,'m':m,'cap_active':mcap<m0,'cap_candidates':caps,'gain':g,'original_T_N_rms_db':float(20*np.log10(rs['T']/rs['N'])),'distances':{},'pair_rms_relative':{},'EQ_R_relative':{}}
        support={'id':sid,'speaker':row['speaker'],'split':row['split'],'headroom_gain':1.,'arms':{}};feats={};saved={}
        for a,x in xs.items():
            ys={'raw':x,'identity':x,'EQ':eq[a],'LEVEL':x*g[a],'EQ_LEVEL':eq[a]*g[a]};feats[a]={};saved[a]={};support['arms'][a]={}
            for c,y in ys.items():
                wav=OUT/'audio'/sid/a/(c+'.wav');wav.parent.mkdir(parents=True,exist_ok=True);sf.write(wav,y,16000,subtype='FLOAT');stored=sf.read(wav,dtype='float64')[0];saved[a][c]=stored
                ff=cal.features(spec,y);feats[a][c]=ff
                metric={'id':sid,'arm':a,'condition':c,'split':row['split'],'finite':bool(np.isfinite(y).all() and np.isfinite(stored).all()),'same_length':len(y)==len(x)==len(stored),'peak':float(max(abs(y).max(),abs(stored).max())),'global_rms':cal.rms(stored),'R_rms':ff['R_rms_db'],'identity_exact':bool(np.array_equal(stored,x)) if c in ['raw','identity'] else None,'target_rms_relative':abs(cal.rms(stored)/m-1) if c in ['LEVEL','EQ_LEVEL'] else None}
                if c in ['LEVEL','EQ_LEVEL']:
                    b='identity' if c=='LEVEL' else 'EQ';inv={'id':sid,'arm':a,'condition':c,'split':row['split'],'float64':cal.invariant(ys[b],y,feats[a][b],ff,g[a]),'float32':cal.invariant(ys[b],stored,feats[a][b],cal.features(spec,stored),g[a])};invariants.append(inv)
                    prefix=OUT/'frontend_description'/sid/a/c;prefix.parent.mkdir(parents=True,exist_ok=True)
                    diagnostic={'id':sid,'arm':a,'condition':c,'split':row['split'],'gain':g[a],
                        'mfcc_float64':mfcc_description(ys[b],y,g[a],prefix.with_suffix('.ideal.npz')),
                        'mfcc_saved_float32':mfcc_description(saved[a][b],stored,g[a],prefix.with_suffix('.saved.npz')),
                        'mel_saved_float32':mel_description(audio,saved[a][b].astype(np.float32),stored.astype(np.float32),g[a])};diagnostics.append(diagnostic)
                metrics.append(metric)
                mel=audio.melspectrogram(audio.load_wav(str(wav),16000)).astype(np.float32);chunks=chunk_mels(mel,25);windows,mfcc=base.frontend_windows(stored)
                front=OUT/'frontend'/sid/a/(c+'.npz');front.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(front,mel=mel,windows=windows)
                info={'waveform':str(wav),'sha256':sha(wav),'samples':len(y),'frames':len(chunks),'audio_windows':len(windows),'L':min(len(chunks),len(y)//640)-5,'frontend':str(front),'frontend_sha256':sha(front)};support['arms'][a][c]=info
                if c in ['raw','identity','EQ']:
                    oi=old_support[sid]['arms'][a][c];assert sha(oi['frontend'])==oi['frontend_sha256'] and sha(oi['waveform'])==oi['sha256']
                    of=np.load(oi['frontend']);oldpcm=sf.read(oi['waveform'],dtype='float64')[0]
                    evidence={'id':sid,'arm':a,'condition':c,'old_info':oi,'source_pcm_equal':bool(np.array_equal(stored,oldpcm)),'mel_equal':bool(np.array_equal(mel,of['mel'])),'MFCC_model_input_equal':bool(np.array_equal(windows,of['windows'])),'frames_L_equal':all(info[k]==oi[k] for k in ['samples','frames','audio_windows','L']),'MFCC_model_input_dtype':'float32 actual forward_aud windows; intermediate float64 MFCC is not model input'}
                    evidence['eligible']=evidence['mel_equal'] and evidence['MFCC_model_input_equal'] and evidence['frames_L_equal'];reuse_candidates.append(evidence)
            support['arms'][a]['joint_L']=support['arms'][a]['identity']['L'];assert len({support['arms'][a][c]['L'] for c in COND})==1
            support['arms'][a]['raw_identity_mel_max']=0.
            pair['EQ_R_relative'][a]=abs(feats[a]['EQ']['R_rms_db']/feats[a]['identity']['R_rms_db']-1)
        for c in COND:pair['distances'][c]={k:cal.rms((feats['N'][c][k]-feats['T'][c][k])[1:]) for k in ['mean_log_db','shape_db']}
        for c in ['LEVEL','EQ_LEVEL']:pair['pair_rms_relative'][c]=abs(cal.rms(saved['N'][c])-cal.rms(saved['T'][c]))/m
        pair['spectrum_reduction']=1-pair['distances']['EQ']['shape_db']/pair['distances']['raw']['shape_db'] if pair['distances']['raw']['shape_db']>=1e-8 else None
        support['guard20_eligible']=all(support['arms'][a]['joint_L']>40 for a in ['N','T']);supports.append(support);pairs.append(pair)
        print('prepared',row['split'],sid,flush=True)
    gates={}
    for split in ['calibration','evaluation']:
        mm=[r for r in metrics if r['split']==split];ii=[r for r in invariants if r['split']==split];pp=[r for r in pairs if r['split']==split];red=[r['spectrum_reduction'] for r in pp if r['spectrum_reduction'] is not None]
        checks={'finite_length_no_clip':all(r['finite'] and r['same_length'] and r['peak']<1 for r in mm),'identity_exact':all(r['identity_exact'] for r in mm if r['condition'] in ['raw','identity']),
            'saved_target':all(r['target_rms_relative']<=1e-5 for r in mm if r['condition'] in ['LEVEL','EQ_LEVEL']), 'saved_pair':all(v<=1e-5 for r in pp for v in r['pair_rms_relative'].values()),
            'float64_invariants':all(r['float64']['shape_max_abs_db']<=1e-8 and r['float64']['R_rms_abs_db']<=1e-8 and r['float64']['cosine_error']<=1e-10 and r['float64']['zero_preserved'] for r in ii),
            'float32_invariants':all(r['float32']['divide_gain_max_abs']<=1e-7 and r['float32']['shape_max_abs_db']<=1e-4 and r['float32']['R_rms_abs_db']<=1e-5 and r['float32']['cosine_error']<=1e-10 and r['float32']['zero_preserved'] for r in ii),
            'EQ_spectrum_reduction_median':float(np.median(red))>=.5,'EQ_spectrum_reduction_fraction':float(np.mean(np.array(red)>=.25))>=.9,'EQ_R_median':float(np.median([v for r in pp for v in r['EQ_R_relative'].values()]))<=.02}
        gates[split]={'passed':all(checks.values()),'checks':checks,'cap_count':sum(r['cap_active'] for r in pp),'pairs':len(pp)}
    oldids=read(OLD/'acoustic_gate.json')['new_main_ids'];ids=[r['id'] for r in supports if r['split']=='evaluation' and r['guard20_eligible']];assert ids==oldids and len(ids)==71
    for name,value in [('support',supports),('pairs',pairs),('acoustic_metrics',metrics),('invariants',invariants),('frontend_description',diagnostics),('reuse_candidates',reuse_candidates)]:write(OUT/(name+'.json'),value)
    write(OUT/'acoustic_gate.json',{'passed':all(d['passed'] for d in gates.values()),'splits':gates,'main_ids':ids,'speakers':len({r['speaker'] for r in supports if r['id'] in ids})})
    print('ACOUSTIC',gates,flush=True)

def seal():
    assert read(OUT/'acoustic_gate.json')['passed'] and read(OUT/'independent_acoustic.json')['status']=='PASS'
    p=base.protocol();assert not (OUT/'features').exists()
    write(OUT/'cpu_seal.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'support_sha256':sha(OUT/'support.json'),'code':p['new_code_hashes'],'acoustic_gate_sha256':sha(OUT/'acoustic_gate.json'),'independent_acoustic_sha256':sha(OUT/'independent_acoustic.json'),'frontend_description_sha256':sha(OUT/'frontend_description.json'),'status':'before_features_or_scores'})

def reuse():
    p=base.protocol();sp=read(SPEC/'protocol.json');assert read(SPEC/'cal_controls.json')['engineering_passed']
    # Only immutable source artifacts, never score-dependent decisions.
    assert p['image']==sp['image']
    for k in ['box','score_box','fps','batch','seed','codec','timing']:assert p['generation'][k]==sp['generation'][k]
    for k in ['lags','distance','k','policies','geometries','endpoint','frontend']:assert p['scoring'][k]==sp['scoring'][k]
    for f,h in sp['dependencies'].items():assert sha(f)==h
    source={r['id']:r for r in sp['rows']}
    for r in p['rows']:assert r['audio']==source[r['id']]['audio']
    manifest=[]
    for r in read(OUT/'reuse_candidates.json'):
        if not r['eligible']:manifest.append({**r,'reused':False});continue
        sid,a,c=r['id'],r['arm'],r['condition'];jp=SPEC/'features'/sid/a/(c+'.json');old=read(jp)
        assert old['audio_frontend']==r['old_info']
        assert sha(old['features'])==old['sha256'] and sha(old['video']['path'])==old['video']['sha256']
        assert old['video']['frames']==r['old_info']['frames']
        dest=OUT/'features'/sid/a/(c+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(old['features'],dest)
        row=copy.deepcopy(old);row['features']=str(dest);row['sha256']=sha(dest);row['reused_from']=str(jp);row['source_metadata_sha256']=sha(jp)
        row['audio_frontend']=next(v for v in read(OUT/'support.json') if v['id']==sid)['arms'][a][c]
        write(dest.with_suffix('.json'),row);manifest.append({**r,'reused':True,'source_metadata':str(jp),'source_metadata_sha256':sha(jp),'feature_sha256':old['sha256'],'video_sha256':old['video']['sha256']})
    write(OUT/'reuse_manifest.json',{'rows':manifest,'reuse_count':sum(r['reused'] for r in manifest),'strict_parameters_equal':True,'source_protocol_sha256':sha(SPEC/'protocol.json'),'firstcal_fresh_repeat_required':True})
    print('REUSED',sum(r['reused'] for r in manifest),flush=True)

def produce(split):
    base.produce(split)
    if split=='calibration':
        first=min(r['id'] for r in read(OUT/'support.json') if r['split']=='calibration');checks={}
        for a in ['N','T']:
            ctrl=read(OUT/'controls'/a/'controls.json');repeat=np.load(OUT/'controls'/a/'controls.npz')['repeat_visual'];cached=np.load(OUT/'features'/first/a/'identity.npz')['visual']
            checks[a]={'pixel_exact':ctrl['repeat_pixel_equal'],'features_exact':bool(np.array_equal(repeat,cached)),'max':float(abs(repeat-cached).max())}
        write(OUT/'reuse_repeat_validation.json',{'passed':all(v['pixel_exact'] and v['features_exact'] for v in checks.values()),'checks':checks})
        assert all(v['pixel_exact'] and v['features_exact'] for v in checks.values()),'strict reuse repeat failure: report root, no scores'

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
            for name,weights in {'EQxLEVEL':{'EQ_LEVEL':1,'EQ':-1,'LEVEL':-1},'EQ_LEVEL_minus_EQ':{'EQ_LEVEL':1,'EQ':-1},'LEVEL_minus_identity':{'LEVEL':1}}.items():
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
