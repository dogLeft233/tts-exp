"""Explicit preprocessing continuation: preserve first attempt and all synthesis."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, signal, subprocess, sys, time
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.experiments.qwen17_level_comparison10_20261023 import experiment as base
ROOT=base.ROOT;PARENT=base.RUN;RUN=ROOT/'runs/qwen17_level_comparison10_20261023_attempt02'
CODE=Path(__file__).parent
sha=base.sha;read=base.read;save=base.save;role=base.role

def prepare():
    audit=read(PARENT/'all100_clipping_audit.json');flag=audit['flagged']
    if len(flag)!=1:raise ValueError('exception coverage changed')
    r=flag[0]
    if (r['sample_id'],r['model'],r['draw_index'])!=('BAC009S0016W0121','L06',3) or r['modes']['raw']['abs_gt1']!=8:
        raise ValueError('exception identity changed')
    if r['source_sha256']!='13a4eee6a54e9abd1ebf59f351c128b5528d395c4d259a4bdcba085df662fc0a' or r['modes']['raw']['affected_indices']!=[363436,363439,363440,363443,363447,471997,472109,487623]:
        raise ValueError('exception source/index changed')
    if read(PARENT/'generate_driver_exit.json')['returncode']!=0 or read(PARENT/'audio_driver_exit.json')['returncode']!=1:
        raise ValueError('predecessor actual exits')
    RUN.mkdir(exist_ok=False);p=read(PARENT/'protocol.json')
    p.update(run_id=RUN.name,predecessor_run=str(PARENT),predecessor_protocol_sha256=sha(PARENT/'protocol.json'),
        generation_manifest_path=str(PARENT/'generation/tts_manifest.json'),generation_manifest_sha256=sha(PARENT/'generation/tts_manifest.json'),
        generation_source_manifest_path=str(PARENT/'generation/tts_manifest.json'),
        generation_reused=True,new_syntheses=0,analysis_protocol_path=str(RUN/'analysis_protocol.json'),
        raw_policy='unit-gain complete waveform; standard PCM16 conversion; exact bound8point exception below; no other clipping',
        raw_pcm16_exception=dict(sample_id=r['sample_id'],model='L06',draw_index=3,source_path=r['source_path'],source_sha256=r['source_sha256'],
            samples=r['samples'],indices=r['modes']['raw']['affected_indices'],values=r['modes']['raw']['affected_values'],
            operation='only these8 samples saturate to [-1,32767/32768]; target/gain/sample count/time axis unchanged',
            basis='User prior persistent preference: retain full cohort with disclosed tiny PCM saturation exception and strict complete-speaker sensitivity',
            reason='Standard PCM16 conversion of8out-of-range original rawsamples; preserveall10 plusstrict9 basedprioruserpreference, disclosedbyroot; no new explicit8pointuserreply',
            strict_sensitivity_exclude_speaker='S0016'))
    p['source_pins'][str(Path(__file__).resolve())]=sha(__file__)
    p['source_pins'][str(PARENT/'existing_sources.json')]=sha(PARENT/'existing_sources.json')
    p['source_pins'][str(PARENT/'reused_scores.json')]=sha(PARENT/'reused_scores.json')
    p['source_pins'][str(PARENT/'all100_clipping_audit.json')]=sha(PARENT/'all100_clipping_audit.json')
    p['source_pins'][p['generation_manifest_path']]=p['generation_manifest_sha256']
    ap=read(PARENT/'analysis_protocol.json');ap['strict_sensitivity']='Repeat all norm/raw contrasts excluding S0016 whole person (strict9), retain full10 main'
    save(RUN/'analysis_protocol.json',ap);save(RUN/'protocol.json',p)
    for name in ['existing_sources.json','reused_scores.json']:(RUN/name).write_bytes((PARENT/name).read_bytes())
    gd=RUN/'generation';gd.mkdir()
    for name in ['tts_manifest.json','runtime.json','completion.json','events.jsonl']:(gd/name).write_bytes((PARENT/'generation'/name).read_bytes())
    save(RUN/'generation_reuse_receipt.json',dict(status='REUSED_NO_GENERATION',source=str(PARENT/'generation'),
        source_completion_sha256=sha(PARENT/'generation/completion.json'),source_manifest_sha256=p['generation_manifest_sha256'],
        original_generation_actual_exit=0,predecessor_preprocessing_actual_exit=1,new_synthesis_count=0,no_training=True))
    snap=RUN/'source_snapshot';snap.mkdir()
    for path in [Path(__file__),CODE/'experiment.py',CODE/'downstream.py']:(snap/path.name).write_bytes(path.read_bytes())
    print('PREPARED02',sha(RUN/'protocol.json'),flush=True)

def audio():
    base.RUN=RUN;p=base.validate();exception=p['raw_pcm16_exception'];records=[];bridge=[];exception_count=0
    sources=read(RUN/'existing_sources.json')['records']
    sources += [dict(sample_id=r['sample_id'],speaker_id=r['speaker_id'],model='L17',draw_index=r['draw_index'],source_path=r['path'],source_sha256=r['sha256']) for r in read(RUN/'generation/tts_manifest.json')['records']]
    for r in sources:
        sid=r['sample_id'];model=r['model'];draw=r['draw_index']
        if sha(r['source_path'])!=r['source_sha256']:raise ValueError('rawsourcechanged')
        f=RUN/'float16'/sid/(role(model,'source',draw)+'.wav');f.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',r['source_path'],'-ac','1','-ar','16000','-c:a','pcm_f32le',str(f)],check=True,timeout=60)
        x,sr=sf.read(f,dtype='float32');rms=float(np.sqrt(np.mean(x.astype(float)**2)))
        if sr!=16000 or x.ndim!=1 or not len(x) or not np.isfinite(x).all() or rms<=0:raise ValueError('invalidwave')
        norm=x*np.float32(p['target_rms']/rms)
        if max(abs(norm))>=1:raise ValueError('normalizationclips; noexception')
        raw=x.copy();clipped=[]
        if max(abs(x))>=1:
            ix=np.where(abs(x)>=1)[0]
            if (sid,model,draw)!=(exception['sample_id'],exception['model'],exception['draw_index']):raise ValueError('unapprovedclipping')
            if r['source_sha256']!=exception['source_sha256'] or len(x)!=exception['samples'] or ix.tolist()!=exception['indices']:
                raise ValueError('boundexceptionchanged')
            if not np.array_equal(x[ix],np.asarray(exception['values'],dtype=np.float32)):raise ValueError('exceptionvalueschanged')
            raw[ix]=np.clip(raw[ix],np.float32(-1),np.float32(32767/32768));clipped=ix.tolist();exception_count+=1
            if not np.array_equal(raw[np.setdiff1d(np.arange(len(x)),ix)],x[np.setdiff1d(np.arange(len(x)),ix)]):raise ValueError('unexpectedmutation')
        if model!='L17':
            import io
            b=io.BytesIO();sf.write(b,norm,16000,format='WAV',subtype='PCM_16');b.seek(0)
            got,_=sf.read(b,dtype='int16');old,_=sf.read(r['old_audio']['path'],dtype='int16')
            if not np.array_equal(got,old):raise ValueError('oldnormalizedPCMbridgefailed')
            bridge.append(dict(sample_id=sid,model=model,draw_index=draw,pcm_exact=True,normalized_sha256=r['old_audio']['sha256']))
        for level,y,gain in [('raw',raw,1.)]+([('norm',norm,p['target_rms']/rms)] if model=='L17' else []):
            name=role(model,level,draw);dest=RUN/'audio'/sid/(name+'_16k_pcm16.wav');dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise FileExistsError('nooverwrite')
            sf.write(dest,y,16000,subtype='PCM_16');pcm,_=sf.read(dest,dtype='int16')
            if len(pcm)!=len(x):raise ValueError('samplecountchanged')
            records.append(dict(sample_id=sid,speaker_id=r['speaker_id'],model=model,level=level,draw_index=draw,role=name,cohort=p['cohort'],
                path=str(dest),sha256=sha(dest),samples=len(pcm),sr=16000,pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                source_path=r['source_path'],source_sha256=r['source_sha256'],float16_path=str(f),float16_sha256=sha(f),
                gain=gain,rms_before=rms,rms_after=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))),no_time_editing=True,
                raw_pcm16_exception_indices=clipped if level=='raw' else []))
    if len(records)!=130 or len(bridge)!=70 or exception_count!=1:raise ValueError('coverage')
    save(RUN/'audio_manifest.json',dict(records=records,count=130,protocol_sha256=sha(RUN/'protocol.json')))
    save(RUN/'normalized_pcm_bridge.json',dict(status='PASS',count=70,records=bridge,exception_count=exception_count))
    print('AUDIO130_BRIDGE70_EXACT_EXCEPTION8_PASS',flush=True)

def driver():
    sync='/home/wjj/.venvs/syncnet/bin/python';protocol=RUN/'protocol.json';h=sha(protocol);start=time.monotonic()
    save(RUN/'driver_launch.json',dict(pid=os.getpid(),command=sys.argv,source_sha256=sha(__file__),protocol_sha256=h,new_TTS=0,no_training=True))
    stages=[('audio',[sync,str(Path(__file__).resolve()),'audio'],1800),
       *[(s,[sync,str(CODE/'downstream.py'),'--stage',s,'--run',str(RUN),'--protocol',str(protocol),'--protocol-sha256',h],10800) for s in ['render','official']],
       ('index',[sync,str(Path(__file__).resolve()),'index'],300)]
    failed=None
    for name,command,timeout in stages:
        print('STAGE',name,flush=True);t=time.monotonic()
        with (RUN/(name+'_console.log')).open('x') as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);timed=False
            try:rc=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                rc=process.returncode
        save(RUN/(name+'_driver_exit.json'),dict(stage=name,pid=process.pid,returncode=rc,timed_out=timed,elapsed_s=time.monotonic()-t,command=command,log_sha256=sha(RUN/(name+'_console.log'))))
        print('EXIT',name,rc,flush=True)
        if rc or timed:failed=name;break
    save(RUN/'driver_exit.json',dict(status='COMPLETE' if failed is None else 'FAILED',failed_stage=failed,elapsed_s=time.monotonic()-start,new_TTS=0,no_training=True,pid=os.getpid()))
    return int(failed is not None)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','audio','driver','index']);args=ap.parse_args()
    if args.stage=='index':base.RUN=RUN;base.index()
    elif args.stage=='driver':raise SystemExit(driver())
    else:globals()[args.stage]()
