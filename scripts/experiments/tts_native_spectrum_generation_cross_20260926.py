"""Frozen spectrum midpoint experiment. Reuse sealed numerical and GxE engines."""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, os, shutil, sys, time, types
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_native_spectrum_generation_cross_20260926'
CAL=ROOT/'runs/tts_native_spectrum_match_calibration_20260926'
OLD=ROOT/'runs/tts_acoustic_generation_cross_20260926'
COND=['raw','identity','EQ','ENVeq'];H=['EQ','ENVeq']
def load(name,path):
    m=types.ModuleType(name);m.__file__=str(ROOT/'scripts/experiments'/Path(path).name);exec(compile(Path(path).read_text(),str(path),'exec'),m.__dict__);return m
base=load('sealed_gxe',OLD/'code_snapshot/tts_acoustic_generation_cross_20260926.py')
base.OUT=OUT;base.COND=COND;base.H=H
cal=load('sealed_spectrum',CAL/'code_snapshot/tts_native_spectrum_match_calibration.py')
read=base.read;sha=base.sha;write=base.write

def resource_check(p,startup=False):
    rows=base.compute_processes()
    assert all(r['pid']==os.getpid() for r in rows),('foreign compute',rows)
    used,util=map(int,base.subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if startup:assert util<=10 and used<=128,(used,util)
    free=shutil.disk_usage(OUT).free;assert free>=5<<30,('disk reserve',free)
    return {'time':time.time(),'compute':rows,'memory_mib':used,'utilization':util,'free_bytes':free}
base.resource_check=resource_check

def freeze():
    assert not (OUT/'protocol.json').exists()
    final=read(CAL/'final.json');assert final['validation_passed'] and final['calibration_status']=='calibration_passed'
    for f,h in read(CAL/'artifact_hashes.json').items():assert sha(CAL/f)==h,f
    p=copy.deepcopy(read(OLD/'protocol.json'));p['created_epoch']=time.time();p['conditions']=COND
    p['acoustics']=read(CAL/'protocol.json');p['acoustic_gates']='Exactly spectrum calibration gates, separately for old26cal and all74eval, before any new score; no deletion/retuning.'
    p['science']='Historical100 native N/cloudT, old26cal engineering/74eval, common original71 guard20. Same-pair test-time spectrum midpoint matching; exploratory, no unseen confirmation.'
    p['cross']='h in EQ/ENVeq; q00 id/id, q10 h/id, q01 id/h, q11 h/h; G,E,I,total; predeclared EQ-ENVeq contrasts.'
    p['statistics']['support']='all primary sensitivity raw/unit guard20/valid/guard0 on common71; additional74 valid/guard0 separate'
    p['generation']['identity']='All identity and EQ/ENVeq freshly rendered; raw aliases only numerically identical identity mel, firstcal raw fresh; no old feature reuse.'
    p['resources']={'startup':'3 snapshots5s apart,util<=10%,memory<=128MiB; no foreign compute PID permitted','reserve_bytes':5<<30,'lease':'/tmp/tts-exp-gpu.lock','expected_new_render_count':604,'estimated_core_bytes':6<<30}
    p['limits']='Partial native long-term spectrum difference matching; envelope,F0/voicing coupled. No active time warp, but not an isolated physiological factor. Sync-C is an evaluator response, not physical mouth ground truth.'
    p['dependencies'][str(CAL/'protocol.json')]=sha(CAL/'protocol.json')
    p['dependencies'][str(CAL/'code_snapshot/tts_native_spectrum_match_calibration.py')]=sha(CAL/'code_snapshot/tts_native_spectrum_match_calibration.py')
    p['dependencies'][str(OLD/'code_snapshot/tts_acoustic_generation_cross_20260926.py')]=sha(OLD/'code_snapshot/tts_acoustic_generation_cross_20260926.py')
    write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print('frozen',flush=True)

def waveforms():
    p=base.protocol()
    for split in ['calibration','evaluation']:
        dest=OUT/'acoustic'/split
        if (dest/'summary.json').exists():continue
        cp=copy.deepcopy(read(CAL/'protocol.json'));rows=[r for r in p['rows'] if r['split']==split]
        cp['cal_ids']=sorted(r['id'] for r in rows);cp['sources']=[{'id':r['id'],'speaker':r['speaker'],'arm':a,**r['audio'][a]} for r in rows for a in ['N','T']]
        cp['scope']='Frozen spectrum engine reuse, '+split+'; cal_ids is legacy field for this explicitly specified split; no scores'
        write(dest/'protocol.json',cp);(dest/'protocol.sha256').write_text(sha(dest/'protocol.json')+'\n')
        cal.OUT=dest;cal.run()
    gate={s:read(OUT/'acoustic'/s/'summary.json') for s in ['calibration','evaluation']}
    write(OUT/'acoustic_gate.json',{'passed':all(x['status']=='calibration_passed' for x in gate.values()),'splits':gate})
    print('ALL ACOUSTIC GATES', {s:d['gates'] for s,d in gate.items()},flush=True)

def prepare():
    p=base.protocol();assert read(OUT/'acoustic_gate.json')['passed']
    sys.path.insert(0,str(base.W2L));import audio
    from scripts.experiments.static_image_bridge.render_worker import chunk_mels
    support=[];allmetrics=[]
    for row in p['rows']:
        sid=row['id'];sp=OUT/'acoustic'/row['split'];record={'id':sid,'speaker':row['speaker'],'split':row['split'],'headroom_gain':read(sp/'summary.json')['headroom_gains'][sid],'arms':{}}
        for a in ['N','T']:
            arm={}
            for c in COND:
                wav=sp/'waveforms'/sid/a/(c+'.wav');y=sf.read(wav,dtype='float64')[0];mel=audio.melspectrogram(audio.load_wav(str(wav),16000)).astype(np.float32);chunks=chunk_mels(mel,25);windows,_=base.frontend_windows(y)
                fp=OUT/'frontend'/sid/a/(c+'.npz');fp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(fp,mel=mel,windows=windows)
                ap=sp/'acoustics'/sid/a/(c+'.npz')
                arm[c]={'waveform':str(wav),'sha256':sha(wav),'samples':len(y),'frames':len(chunks),'audio_windows':len(windows),'L':min(len(chunks),len(y)//640)-5,'frontend':str(fp),'frontend_sha256':sha(fp),'acoustics':str(ap),'acoustics_sha256':sha(ap)}
            assert len({arm[c]['L'] for c in COND})==1
            arm['joint_L']=arm['identity']['L'];arm['raw_identity_mel_max']=float(abs(np.load(arm['raw']['frontend'])['mel']-np.load(arm['identity']['frontend'])['mel']).max());assert arm['raw_identity_mel_max']<=1e-5
            record['arms'][a]=arm
        record['guard20_eligible']=all(record['arms'][a]['joint_L']>40 for a in ['N','T']);support.append(record)
        print('frontends',sid,flush=True)
    old=read(OLD/'acoustic_gate.json')['new_main_ids'];actual=[r['id'] for r in support if r['split']=='evaluation' and r['guard20_eligible']];assert old==actual and len(actual)==71
    write(OUT/'support.json',support);write(OUT/'support_audit.json',{'same_old_support':True,'ids':actual,'common71_speakers':len(set(r['speaker'] for r in support if r['id'] in actual))})

def seal():
    assert read(OUT/'acoustic_gate.json')['passed'] and not (OUT/'features').exists()
    for s in ['calibration','evaluation']:assert read(OUT/'acoustic'/s/'independent_validation.json')['passed']
    files=[Path(__file__),ROOT/'scripts/experiments/check_tts_native_spectrum_generation_cross_20260926.py',ROOT/'scripts/experiments/check_tts_native_spectrum_cross_scores_20260926.py',ROOT/'tests/test_tts_native_spectrum_generation_cross_20260926.py']
    for f in files:
        dest=OUT/'code_snapshot'/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
    write(OUT/'cpu_seal.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'support_sha256':sha(OUT/'support.json'),'acoustic_gate_sha256':sha(OUT/'acoustic_gate.json'),'code':{str(f):sha(f) for f in files},'status':'before_any_new_feature_or_score'})

def analyze():
    rows=[read(f) for f in sorted((OUT/'scores').glob('*.json')) if read(f)['split']=='evaluation'];assert len(rows)==74
    output={};per=[];closure=0.
    for geometry in ['raw','unit']:
      for policy in ['guard20','valid','guard0']:
       for support in (['common71'] if policy=='guard20' else ['common71','all74']):
        use=[r for r in rows if r['eligible']] if support=='common71' else rows;groups=[r['speaker'] for r in use];view={}
        for metric in ['C','B','D','C_anchor','D_anchor','best_lag']:
            native=np.array([r['cells']['T'][geometry]['identity__identity']['policies'][policy][metric]-r['cells']['N'][geometry]['identity__identity']['policies'][policy][metric] for r in use]);detail={'baseline_T_minus_N':base.bootstrap(native,groups),'conditions':{},'EQ_minus_ENVeq':{}};armvalues={}
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
            for effect in ['G','E','I','total']:
                n=np.array([r[effect] for r in armvalues['EQ']['N']])-np.array([r[effect] for r in armvalues['ENVeq']['N']]);t=np.array([r[effect] for r in armvalues['EQ']['T']])-np.array([r[effect] for r in armvalues['ENVeq']['T']]);detail['EQ_minus_ENVeq'][effect]={'N':base.bootstrap(n,groups),'T':base.bootstrap(t,groups),'T_minus_N':base.bootstrap(t-n,groups)}
            view[metric]=detail
        output[geometry+'/'+policy+'/'+support]=view
    write(OUT/'summary.json',{'results':output,'maximum_algebra_closure_error':closure,'status':'complete','cal_controls':read(OUT/'cal_controls.json')});write(OUT/'effects.json',per)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','waveforms','prepare','seal','produce','score','analyze']);ap.add_argument('--split',choices=['calibration','evaluation']);args=ap.parse_args()
    if args.stage in ['produce','score']:getattr(base,args.stage)(args.split)
    else:globals()[args.stage]()
