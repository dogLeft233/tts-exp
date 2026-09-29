"""CPU cached E-only level bridge across every frozen source condition."""
from pathlib import Path
import argparse,hashlib,json,shutil,time,types,sys
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_level_reference_evaluator_bridge_20260927'
SOURCE=ROOT/'runs/tts_source_conditioning_20260926';LEVEL=ROOT/'runs/tts_native_level_generation_cross_20260927'
ENGINE=ROOT/'runs/tts_acoustic_generation_cross_20260926/code_snapshot/tts_acoustic_generation_cross_20260926.py'
base=types.ModuleType('frozen_gxe');base.__file__=str(ROOT/'scripts/experiments'/ENGINE.name);exec(compile(ENGINE.read_text(),str(ENGINE),'exec'),base.__dict__)
read,write,sha=base.read,base.write,base.sha
CONDS=['D0','D66','S0','S66'];KINDS=['generated','source_only'];GEOMS=['raw','unit'];POLICIES=['guard20','valid','guard0'];METRICS=['C','B','D','C_anchor','D_anchor']
def frozen():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for f,h in p['bindings'].items():assert sha(f)==h,f
    return p

def freeze():
    assert not (OUT/'protocol.json').exists()
    src=read(SOURCE/'protocol.json');lev=read(LEVEL/'protocol.json');ss=read(SOURCE/'support.json');ls={r['id']:r for r in read(LEVEL/'support.json')};lr={r['id']:r for r in lev['rows']};rows=[]
    assert len(ss)==74 and sum(r['split']=='evaluation' for r in ss)==48 and all(r['eligible'] for r in ss)
    assert read(SOURCE/'calibration.json')['k0']==3
    for row in ss:
        r={'id':row['id'],'speaker':row['speaker'],'split':row['split'],'arms':{}}
        for a,info in row['arms'].items():
            assert info['audio_sha256']==lr[row['id']]['audio'][a]['sha256']
            assert sha(info['audio'])==info['audio_sha256']
            assert all(info['samples']==ls[row['id']]['arms'][a][c]['samples'] and info['L']==ls[row['id']]['arms'][a][c]['L'] for c in ['raw','LEVEL'])
            r['arms'][a]={'source_audio':info['audio'],'source_audio_sha256':info['audio_sha256'],'samples':info['samples'],'L':info['L'],'frames':info['frames'],'source_audio_features':str(SOURCE/'audio_features'/row['id']/(a+'.npz')),'level':{c:ls[row['id']]['arms'][a][c] for c in ['raw','LEVEL']},'source_visual':{c:str(SOURCE/'features'/row['id']/a/(c+'.npz')) for c in CONDS},'level_features':{c:str(LEVEL/'features'/row['id']/a/(c+'.npz')) for c in ['raw','LEVEL']}}
        rows.append(r)
    assert all((SOURCE/'source_only'/(c+'.npz')).exists() for c in CONDS)
    files=[SOURCE/n for n in ['protocol.json','support.json','feature_seal.json','calibration.json','geometry.json','runtime.json','repeat_controls.json','delay_controls.json']]+list((SOURCE/'scores').rglob('*.json'))+[LEVEL/n for n in ['protocol.json','support.json','cpu_seal.json','acoustic_gate.json','independent_acoustic.json','cal_controls.json']]+[ENGINE,Path(__file__),ROOT/'scripts/experiments/check_tts_level_reference_evaluator_bridge_20260927.py',ROOT/'tests/test_tts_level_reference_evaluator_bridge_20260927.py']
    p={'status':'frozen_before_LEVEL_main_results_and_any_new_distance','created_epoch':time.time(),'rows':rows,'conditions':CONDS,'kinds':KINDS,'geometries':GEOMS,'policies':POLICIES,'metrics':METRICS,'k':3,'primary':'original48eval/13speakers, generated raw guard20, all D0 D66 S0 S66; E_a=C(V_original,a,A_LEVEL,a)-C(V_original,a,A_RAW,a), N/T and T-N response; no G or original matched total estimand','statistics':{'speaker_equal':True,'bootstrap':20000,'seed':20260926,'ci':.99,'multiplicity':'individual comparisons, no FWER'},'appendix':'same48 source_only and raw/unit valid/guard0/C_anchor; all74 aggregate explicitly appendix','gates':'exact original source PCM hashes,16k samples and source official L unchanged; new audio covers L; MFCC/model consistency; RAW embeddings prefix max<=1e-5; new RAW native C same support max<=1e-5; all sources retained or stop','anchor':'cal generated condition x arm medians difference >1 or abs(median-3)>1 flags migration only; never refit k/support','controls':'inherit sealed source and LEVEL firstcal N/T repeat/frontend/+5frame controls, no new inference','scope':'CPU cached only; no new audio/video/model inference; same-input exploratory across reference geometry/motion; source-only not mouth truth; all comparisons within source ROI, never subtract cross-ROI scores as G','level_main_scores_read':False,'source_roi':read(SOURCE/'geometry.json')['score_roi'],'bindings':{str(f):sha(f) for f in files},'source_models':src['code_and_models'],'level_dependencies':lev['dependencies'],'source_only':{c:str(SOURCE/'source_only'/(c+'.npz')) for c in CONDS}}
    write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    for f in files[-3:]:
        d=OUT/'code_snapshot'/f.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,d)
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)

def bind():
    import python_speech_features as psf
    p=frozen();assert read(LEVEL/'acoustic_gate.json')['passed'];assert read(LEVEL/'independent_acoustic.json')['status']=='PASS';assert read(LEVEL/'cal_controls.json')['engineering_passed']
    assert read(SOURCE/'repeat_controls.json')['passed'] and all(v['passed'] for v in read(SOURCE/'delay_controls.json')['controls'].values())
    source_seal=read(SOURCE/'feature_seal.json');level_seal={}
    for split in ['calibration','evaluation']:level_seal.update(read(LEVEL/('feature_seal_'+split+'.json')))
    files={};errors={'raw_embedding':0.,'raw_mfcc_window':0.};checks=[]
    for path,h in source_seal.items():assert sha(path)==h,path
    for f,h in p['source_models'].items():
        if f in p['level_dependencies']:assert p['level_dependencies'][f]==h
    model=str(ROOT/'third_party/syncnet_python/data/syncnet_v2.model');assert p['source_models'][model]==p['level_dependencies'][model]==sha(model)
    for row in p['rows']:
        for a,r in row['arms'].items():
            n=r['L'];oldpath=r['source_audio_features'];assert sha(oldpath)==source_seal[oldpath];old=np.load(oldpath)['audio'];assert len(old)>=n;files[oldpath]=sha(oldpath)
            pcm,sr=sf.read(r['source_audio'],dtype='int16');assert sr==16000 and len(pcm)==r['samples'];mfcc=psf.mfcc(pcm,16000);windows=np.stack([mfcc[i*4:i*4+20].T for i in range(n)]).astype(np.float32)
            for c in ['raw','LEVEL']:
                feat=r['level_features'][c];assert sha(feat)==level_seal[feat];meta=Path(feat).with_suffix('.json');assert sha(meta)==level_seal[str(meta)];fm=read(meta);assert fm['audio_frontend']==r['level'][c];files[feat]=sha(feat);files[str(meta)]=sha(meta)
                info=r['level'][c];assert sha(info['waveform'])==info['sha256'] and sha(info['frontend'])==info['frontend_sha256'];wave,rate=sf.read(info['waveform'],dtype='float64');assert rate==16000 and len(wave)==r['samples'];new=np.load(feat)['audio'];assert len(new)>=n
                if c=='raw':
                    err=float(abs(new[:n]-old[:n]).max());front=np.load(info['frontend'])['windows'];werr=float(abs(front[:n]-windows).max());errors['raw_embedding']=max(errors['raw_embedding'],err);errors['raw_mfcc_window']=max(errors['raw_mfcc_window'],werr);checks.append({'id':row['id'],'arm':a,'L':n,'old_count':len(old),'new_count':len(new),'embedding_max':err,'mfcc_window_max':werr})
                files[info['waveform']]=info['sha256'];files[info['frontend']]=info['frontend_sha256']
            for c in CONDS:
                for path in [r['source_visual'][c],p['source_only'][c]]:
                    assert sha(path)==source_seal[path] and len(np.load(path)['visual'])>=n;files[path]=sha(path)
    result={'passed':max(errors.values())<=1e-5,'errors':errors,'checks':checks,'all_inputs':files,'source_runtime':read(SOURCE/'runtime.json'),'level_runtime':read(LEVEL/'runtime.json'),'parent_seals':{str(f):sha(f) for f in [SOURCE/'feature_seal.json',LEVEL/'feature_seal_calibration.json',LEVEL/'feature_seal_evaluation.json']},'time':time.time()}
    write(OUT/'input_validation.json',result)
    if not result['passed']:raise RuntimeError('input numerical consistency gate failed; no distance scoring')
    write(OUT/'score_lock.json',{'status':'before_any_new_distance','time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'input_validation_sha256':sha(OUT/'input_validation.json'),'code':{str(f):sha(f) for f in [Path(__file__),ROOT/'scripts/experiments/check_tts_level_reference_evaluator_bridge_20260927.py']}})
    print('BOUND',errors,flush=True)

def features(p,row,a,c,kind,geometry,audiocond):
    r=row['arms'][a];n=r['L'];path=r['source_visual'][c] if kind=='generated' else p['source_only'][c];v=np.load(path)['visual'][:n];audio=np.load(r['level_features'][audiocond])['audio'][:n]
    assert len(v)==len(audio)==n
    return (base.unit(v),base.unit(audio)) if geometry=='unit' else (v,audio)

def score():
    import torch
    torch.set_num_threads(2);p=frozen();lock=read(OUT/'score_lock.json');assert sha(OUT/'input_validation.json')==lock['input_validation_sha256'];iv=read(OUT/'input_validation.json');assert iv['passed']
    for f,h in {**iv['all_inputs'],**lock['code']}.items():assert sha(f)==h
    # Baseline-only gate is completed across all rows before any LEVEL distance.
    maxerr=0.;baseline={}
    for row in p['rows']:
        old=read(SOURCE/'scores'/row['split']/(row['id']+'.json'));cells={};mats={}
        for a in ['N','T']:
          for c in CONDS:
           for kind in KINDS:
            for geom in GEOMS:
                key='/'.join([kind,geom,a,c]);v,au=features(p,row,a,c,kind,geom,'raw');m=base.matrix(v,au);stats={policy:base.summarize(m,policy,3) for policy in POLICIES};maxerr=max(maxerr,max(abs(stats[policy]['C']-old['cells'][key][policy]['C']) for policy in POLICIES));cells[key+'/raw']=stats;mats[key+'/raw']=m
        baseline[row['id']]={'cells':cells,'matrices':mats}
    write(OUT/'raw_baseline_audit.json',{'passed':maxerr<=1e-5,'max_C_error':maxerr,'all74_all4_all_kinds_geometries_policies':True})
    if maxerr>1e-5:raise RuntimeError('new RAW fails source native C gate; no LEVEL distances')
    for row in p['rows']:
        sid=row['id'];cells=baseline[sid]['cells'];mats=baseline[sid]['matrices']
        for a in ['N','T']:
          for c in CONDS:
           for kind in KINDS:
            for geom in GEOMS:
                key='/'.join([kind,geom,a,c,'LEVEL']);v,au=features(p,row,a,c,kind,geom,'LEVEL');m=base.matrix(v,au);cells[key]={policy:base.summarize(m,policy,3) for policy in POLICIES};mats[key]=m
        dest=OUT/'scores'/sid;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),**mats);write(dest.with_suffix('.json'),{'id':sid,'speaker':row['speaker'],'split':row['split'],'cells':cells,'matrix_sha256':sha(dest.with_suffix('.npz'))});print('scored',sid,flush=True)
    med={c:{a:float(np.median([read(OUT/'scores'/(r['id']+'.json'))['cells'][f'generated/raw/{a}/{c}/LEVEL']['guard20']['best_lag'] for r in p['rows'] if r['split']=='calibration'])) for a in ['N','T']} for c in CONDS};allmed=[v for d in med.values() for v in d.values()]
    write(OUT/'calibration_transfer.json',{'k_fixed':3,'LEVEL_medians':med,'transfer_flag_sufficient':max(allmed)-min(allmed)<=1 and all(abs(v-3)<=1 for v in allmed),'no_refit_or_selection':True})

def analyze():
    p=frozen();rows=[read(OUT/'scores'/(r['id']+'.json')) for r in p['rows']];output={};effects=[]
    for subset in ['evaluation48','all74_appendix']:
     use=[r for r in rows if r['split']=='evaluation'] if subset=='evaluation48' else rows;groups=[r['speaker'] for r in use]
     for kind in KINDS:
      for geom in GEOMS:
       for policy in POLICIES:
        view={}
        for c in CONDS:
         view[c]={}
         for metric in METRICS:
            values={a:[] for a in ['N','T']};raw={a:[] for a in ['N','T']};level={a:[] for a in ['N','T']}
            for r in use:
                for a in ['N','T']:
                    q0=r['cells'][f'{kind}/{geom}/{a}/{c}/raw'][policy][metric];q1=r['cells'][f'{kind}/{geom}/{a}/{c}/LEVEL'][policy][metric];raw[a].append(q0);level[a].append(q1);values[a].append(q1-q0);effects.append({'id':r['id'],'speaker':r['speaker'],'subset':subset,'kind':kind,'geometry':geom,'policy':policy,'condition':c,'metric':metric,'arm':a,'raw':q0,'LEVEL':q1,'E':q1-q0})
            view[c][metric]={a:base.bootstrap(values[a],groups) for a in ['N','T']};view[c][metric].update({'T_minus_N':base.bootstrap(np.array(values['T'])-values['N'],groups),'raw_gap':base.bootstrap(np.array(raw['T'])-raw['N'],groups),'LEVEL_evaluation_gap':base.bootstrap(np.array(level['T'])-level['N'],groups)})
        output['/'.join([subset,kind,geom,policy])]=view
    write(OUT/'summary.json',output);write(OUT/'effects.json',effects)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','bind','score','analyze']);a=ap.parse_args();globals()[a.stage]()
