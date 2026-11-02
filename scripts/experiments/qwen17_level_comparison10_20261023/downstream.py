"""One-shot official SyncNet rescore of frozen existing N/R/T videos.

CPU remux preserves decoded video/PTS and substitutes bound canonical PCM.
Only after all frozen inputs/protocol/source pins are frozen, one serial official
pipeline->SyncNet worker executes. No model, official script or offset patch.
"""
from __future__ import annotations
import argparse
import csv
import datetime
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import wave

ROOT=Path(__file__).resolve().parents[3]
SOURCE=None
RUN=None
CONFIG=None
EXPECTED=None
STAGE_DEADLINE=None
SYNCROOT=ROOT/'third_party/syncnet_python'
PYTHON='/home/wjj/.venvs/syncnet/bin/python'  # Preserve venv entrypoint, no resolve().
MODEL_SHA='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442'
MANIFEST_SHAS={}
ROLES=();CAP=20*1024**3;FREE=12*1024**3;HARD_SECONDS=10800
NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def pcm_info(path):
    with wave.open(str(path)) as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(1,2,16000):raise ValueError('canonical input must be mono16k PCM16')
        raw=f.readframes(f.getnframes());n=f.getnframes()
    if not n:raise ValueError('empty canonical PCM')
    return {'PCM_sha256':hashlib.sha256(raw).hexdigest(),'samples':n,'sample_rate_hz':16000}


def validate_inputs(audio,video):
    aa={(r['sample_id'],r['role']):r for r in audio};vv={(r['sample_id'],r['role']):r for r in video}
    expected={(r['sample_id'],role) for r in CONFIG['samples'] for role in CONFIG['roles']}
    if len(audio)!=EXPECTED or len(video)!=EXPECTED or len(aa)!=EXPECTED or set(aa)!=set(vv) or set(aa)!=expected:
        raise ValueError('exact frozen audio/video keys required')
    speakers={r['sample_id']:r['speaker_id'] for r in CONFIG['samples']};rows=[]
    for key in sorted(vv,key=lambda k:(k[0],ROLES.index(k[1]))):
        a,v=aa[key],vv[key]
        if a['speaker_id']!=v['speaker_id'] or a['speaker_id']!=speakers[key[0]] or a['sha256']!=v['audio_sha256']:
            raise ValueError('audio/video source binding')
        rows.append(dict(sample_id=key[0],speaker_id=a['speaker_id'],cohort=CONFIG['cohort'],role=key[1],
                    video_path=v['path'],source_video_sha256=v['sha256'],canonical_audio_path=a['path'],
                    canonical_audio_sha256=a['sha256'],PCM_sha256=a['pcm_sha256'],audio_samples=a['samples']))
    return rows


def parse_tracks(text):
    """Preserve printed order, never choose highest confidence."""
    offsets=list(re.finditer(r'AV offset:\s+(-?\d+)',text))
    distances=list(re.finditer(r'Min dist:\s+('+NUMBER+')',text))
    confidences=list(re.finditer(r'Confidence:\s+('+NUMBER+')',text))
    if not offsets or len(offsets)!=len(distances) or len(offsets)!=len(confidences):raise ValueError('missing/incomplete printed track triples')
    result=[]
    for i,(o,d,c) in enumerate(zip(offsets,distances,confidences)):
        if not o.start()<d.start()<c.start() or (i+1<len(offsets) and c.start()>=offsets[i+1].start()):raise ValueError('ambiguous track ordering')
        dc,cc=float(d.group(1)),float(c.group(1))
        if not all(__import__('math').isfinite(x) for x in [dc,cc]):raise ValueError('nonfinite printed score')
        result.append(dict(printed_index=i,sync_c=cc,sync_d=dc,av_offset=int(o.group(1))))
    return result


def frame_hashes(text):
    hashes=[]
    for line in text.splitlines():
        if line.strip() and not line.startswith('#'):
            fields=[f.strip() for f in line.split(',')]
            if len(fields)!=6:raise ValueError('invalid framemd5 line')
            hashes.append(fields[-1])
    if not hashes:raise ValueError('no decoded frames')
    return hashes


def frame_clock(value):
    rows=[]
    for f in value['frames']:
        if 'best_effort_timestamp_time' not in f:raise ValueError('frame PTS absent')
        rows.append(Decimal(f['best_effort_timestamp_time']))
    if not rows:raise ValueError('empty video frame clock')
    return rows


class Runner:
    def __init__(self):
        self.start=time.monotonic();self.deadline=self.start+HARD_SECONDS;self.pins={};self.process=None
        self.rows=[];self.records=[];self.failures=[];self.terminal_error=None
    def disk_bytes(self):return sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    def guard(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('global10800s hard limit')
        if shutil.disk_usage(RUN).free<FREE:raise RuntimeError('free disk below12GiB')
        if self.disk_bytes()>CAP:raise RuntimeError('output disk exceeds20GiB')
    def event(self,name,**fields):
        value=dict(event=name,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-self.start,**fields)
        with (RUN/'events.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
    def kill_own(self,process):
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    def command(self,args,name,*,cwd=ROOT,limit=300):
        self.guard();log=RUN/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        receipt=RUN/'receipts'/f'{name}.json';start=time.monotonic();fatal=None;timed_out=False
        with log.open('x') as f:
            f.write('$ '+' '.join(str(a) for a in args)+'\n');f.flush()
            process=subprocess.Popen([str(a) for a in args],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            self.process=process
            try:
                while process.poll() is None:
                    try:process.wait(timeout=min(2,max(.01,self.deadline-time.monotonic())))
                    except subprocess.TimeoutExpired:pass
                    if time.monotonic()-start>=limit and process.poll() is None:
                        timed_out=True;self.kill_own(process)
                    self.guard()
            except BaseException as exc:fatal=exc;self.kill_own(process)
            finally:self.process=None
            code=process.wait();f.write(f'\n[returncode] {code}\n[elapsed_seconds] {time.monotonic()-start:.6f}\n')
        result=dict(command=[str(a) for a in args],cwd=str(cwd),pid=process.pid,returncode=code,timed_out=timed_out,
                    elapsed_seconds=time.monotonic()-start,log=str(log),log_sha256=sha(log),fatal_error=repr(fatal) if fatal else None)
        write(receipt,result)
        if fatal:raise fatal
        return result
    def checked(self,args,name,*,cwd=ROOT,limit=300):
        r=self.command(args,name,cwd=cwd,limit=limit)
        if r['returncode'] or r['timed_out']:raise RuntimeError('failed preparation command '+name)
        return r
    def capture(self,args,name,*,limit=300):
        # Persist stdout separately from execution log, so framemd5/JSON are exact.
        dest=RUN/'checks'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        wrapped=[PYTHON,'-c','import subprocess,sys;from pathlib import Path;p=Path(sys.argv[1]);f=p.open("xb");r=subprocess.run(sys.argv[2:],stdout=f);f.close();raise SystemExit(r.returncode)',str(dest),*[str(a) for a in args]]
        self.checked(wrapped,'check_'+name.replace('/','_').replace('.','_'),limit=limit)
        return dest
    def assert_gpu_idle(self):
        text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
        allowed=('gnome-remote-desktop-daemon','/Xorg','gnome-shell')
        unexpected=[line for line in text.splitlines() if line.strip() and not any(x in line for x in allowed)]
        if unexpected:raise RuntimeError('other GPU process present: '+repr(unexpected))
        return text
    def check_pins(self):
        for path,expected in self.pins.items():
            if sha(path)!=expected:raise RuntimeError('pinned input/source changed: '+path)
    def prepare(self):
        self.guard();gpu=self.assert_gpu_idle();source_bindings={}
        for name,expected in MANIFEST_SHAS.items():
            path=SOURCE/name
            if sha(path)!=expected:raise ValueError('source manifest differs '+name)
            source_bindings[str(path)]=expected
        self.rows=validate_inputs(read(SOURCE/'audio_manifest.json')['records'],read(SOURCE/'video_manifest.json')['records'])
        self.pins={str(p):sha(p) for p in sorted(SYNCROOT.rglob('*.py')) if '__pycache__' not in p.parts}
        model=SYNCROOT/'data/syncnet_v2.model';detector=SYNCROOT/'detectors/s3fd/weights/sfd_face.pth'
        self.pins.update({str(model):sha(model),str(detector):sha(detector),str(Path(__file__).resolve()):sha(__file__)})
        if self.pins[str(model)]!=MODEL_SHA:raise ValueError('SyncNet weight mismatch')
        analysis_path=Path(CONFIG['analysis_protocol_path'])
        analysis_bytes=analysis_path.read_bytes();(RUN/'analysis_protocol.json').write_bytes(analysis_bytes)
        self.pins[str(RUN/'analysis_protocol.json')]=sha(RUN/'analysis_protocol.json')
        snapshot=RUN/'source_snapshot';snapshot.mkdir(exist_ok=False);(snapshot/'runner.py').write_bytes(Path(__file__).read_bytes())
        for name in ('run_pipeline.py','run_syncnet.py','SyncNetInstance.py','SyncNetModel.py'):
            (snapshot/name).write_bytes((SYNCROOT/name).read_bytes())
        protocol=dict(run_id=RUN.name,app_date='2026-10-23',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='Official pipeline for frozen current-chain videos; no new TTS/TFG within this scoring stage',
            expected=EXPECTED,speakers=len(CONFIG['samples']),roles=ROLES,order='sorted sample_id then N,R,T1,T2,T3; first sorted N is normal scored cell0',
            primary_cohort=CONFIG['cohort'],parent_protocol_sha256=sha(SOURCE/'protocol.json'),source_manifests=source_bindings,
            analysis_protocol_sha256=sha(RUN/'analysis_protocol.json'),source_and_weight_pins=self.pins,
            syncnet_python=PYTHON,syncnet_root=str(SYNCROOT),min_track=25,first_track_policy='first printed track historical primary; all tracks retained, never best-track choice',
            execution=dict(global_timeout_s=HARD_SECONDS,pipeline_timeout_s=300,syncnet_timeout_s=300,output_cap_bytes=CAP,min_free_bytes=FREE,
                           single_serial_GPU_worker=True,no_training=True,other_GPU_processes_not_killed=True),
            input_adapter='Copy video bitstream into MKV with bound canonical16kPCM; no -shortest; decode framemd5+perframePTS+PCM exact controls all frozen before first forward',
            official_policy='Unmodified run_pipeline.py then run_syncnet.py, cwd original SyncNet; preserve intermediate MP3/offset/tracks and every output',
            failure_policy='No retry; any cell failure recorded and retained. First normal cell failure stops for engineering review. Resource/hash/global-time failures stop run; untouched cells NOT_RUN.',
            GPU_preflight=gpu,ffmpeg_path=shutil.which('ffmpeg'),ffprobe_path=shutil.which('ffprobe'))
        write(RUN/'protocol.json',protocol)
        versions={}
        for name in ('torch','numpy','scipy','opencv-python','opencv-python-headless','python-speech-features','scenedetect'):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]=None
        write(RUN/'runtime.json',dict(sys_version=sys.version,sys_executable=sys.executable,versions=versions,
             GPU_driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True),
             ffmpeg_version=subprocess.check_output([shutil.which('ffmpeg'),'-version'],text=True),
             ffprobe_version=subprocess.check_output([shutil.which('ffprobe'),'-version'],text=True),
             no_model_loaded_for_runtime_receipt=True))
        self.pins[str(RUN/'protocol.json')]=sha(RUN/'protocol.json')
        self.pins.update(source_bindings)
        for index,row in enumerate(self.rows):
            sid,role=row['sample_id'],row['role'];key=f'{sid}_{role}'
            if sha(row['video_path'])!=row['source_video_sha256'] or sha(row['canonical_audio_path'])!=row['canonical_audio_sha256']:raise ValueError('source media SHA mismatch '+key)
            pcm=pcm_info(row['canonical_audio_path'])
            if pcm['PCM_sha256']!=row['PCM_sha256'] or pcm['samples']!=row['audio_samples']:raise ValueError('canonical PCM mismatch '+key)
            mux=RUN/'inputs'/sid/f'{role}.mkv';mux.parent.mkdir(parents=True,exist_ok=True)
            remux=self.checked(['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-copyts','-i',row['video_path'],'-i',row['canonical_audio_path'],
                 '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le','-ar','16000','-ac','1','-avoid_negative_ts','disabled',str(mux)],'remux_'+key)
            original=self.capture(['/usr/bin/ffmpeg','-nostdin','-v','error','-i',row['video_path'],'-map','0:v:0','-an','-vsync','0','-f','framemd5','-'],key+'/source.framemd5')
            copied=self.capture(['/usr/bin/ffmpeg','-nostdin','-v','error','-i',mux,'-map','0:v:0','-an','-vsync','0','-f','framemd5','-'],key+'/mux.framemd5')
            source_clock=self.capture(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',row['video_path']],key+'/source_frames.json')
            mux_clock=self.capture(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',mux],key+'/mux_frames.json')
            h1,h2=frame_hashes(original.read_text()),frame_hashes(copied.read_text())
            t1,t2=frame_clock(read(source_clock)),frame_clock(read(mux_clock))
            if h1!=h2 or t1!=t2 or len(h1)!=len(t1):raise ValueError('remux decoded video/PTS parity failure '+key)
            raw=self.capture(['/usr/bin/ffmpeg','-nostdin','-v','error','-i',mux,'-map','0:a:0','-f','s16le','-acodec','pcm_s16le','-ar','16000','-ac','1','pipe:1'],key+'/mux_PCM.s16le')
            if sha(raw)!=row['PCM_sha256'] or raw.stat().st_size!=2*row['audio_samples']:raise ValueError('remux PCM parity failure '+key)
            row.update(input_mkv=str(mux),input_mkv_sha256=sha(mux),reference='nrt23_'+key,frame_count=len(h1),
                       decoded_frames_exact=True,frame_PTS_exact=True,mux_PCM_exact=True,remux_receipt=remux,
                       checks={str(p):sha(p) for p in [original,copied,source_clock,mux_clock,raw]})
            write(RUN/'input_receipts'/f'{key}.json',row)
            if (index+1)%10==0:self.event('prepare_progress',completed=index+1,remaining=EXPECTED-1-index,failed=0,disk_bytes=self.disk_bytes())
        write(RUN/'input_manifest.json',dict(status='FROZEN_BEFORE_FIRST_OFFICIAL_FORWARD',count=EXPECTED,protocol_sha256=sha(RUN/'protocol.json'),records=self.rows))
        self.pins[str(RUN/'input_manifest.json')]=sha(RUN/'input_manifest.json')
        write(RUN/'execution_bindings.json',dict(pins=self.pins,source_snapshot_sha256=sha(snapshot/'runner.py'),pid=os.getpid(),command=sys.argv,
              global_start_monotonic=self.start,hard_limit_seconds=HARD_SECONDS,analysis_protocol_sha256=sha(RUN/'analysis_protocol.json')))
        self.event('all_inputs_frozen',count=EXPECTED,protocol_sha256=sha(RUN/'protocol.json'),input_manifest_sha256=sha(RUN/'input_manifest.json'))
    def probe(self,path,name):
        p=self.capture(['/usr/bin/ffprobe','-v','error','-count_frames','-show_entries','stream=index,codec_type,codec_name,sample_rate,channels,width,height,nb_read_frames,r_frame_rate,duration,start_time','-of','json',path],name)
        return read(p)
    def track_metadata(self,row):
        root=RUN/'data';ref=row['reference'];track_file=root/'pywork'/ref/'tracks.pckl';activesd=root/'pywork'/ref/'activesd.pckl'
        result=dict(tracks_pickle_path=str(track_file),tracks_pickle_sha256=sha(track_file) if track_file.exists() else None,
                    activesd_path=str(activesd),activesd_sha256=sha(activesd) if activesd.exists() else None)
        crop_files=sorted((root/'pycrop'/ref).glob('*.avi'));metadata=[]
        tracks=[]
        if track_file.exists():
            with track_file.open('rb') as f:tracks=pickle.load(f)
        for index,crop in enumerate(crop_files):
            item=dict(crop_index=index,crop_file=str(crop),crop_sha256=sha(crop),probe=self.probe(crop,f'{ref}/crop_{index}_probe.json'))
            if index<len(tracks):
                track=tracks[index].get('track',{});frames=track.get('frame',[])
                if len(frames):item.update(track_frame_count=len(frames),track_start_frame=int(frames[0]),track_end_frame=int(frames[-1]),
                    track_start_s=int(frames[0])/25.,track_end_exclusive_s=(int(frames[-1])+1)/25.)
            metadata.append(item)
        result['crop_tracks']=metadata
        avi=root/'pyavi'/ref/'video.avi'
        result['official_intermediate_AVI_probe']=self.probe(avi,f'{ref}/official_AVI_probe.json') if avi.exists() else None
        return result
    def score(self):
        self.guard();self.check_pins();gpu=self.assert_gpu_idle();self.event('official_scoring_started',GPU_preflight=gpu)
        for index,row in enumerate(self.rows):
            self.guard();self.check_pins();key=f"{row['sample_id']}_{row['role']}"
            base={**row,'status':'FAILED','sync_c':None,'sync_d':None,'av_offset':None,'tracks':[],'first_track_policy':'first printed, no selection'}
            if sha(row['input_mkv'])!=row['input_mkv_sha256']:raise RuntimeError('frozen MKV changed '+key)
            self.event('cell_started',index=index,cell=key)
            pipeline=self.command([PYTHON,SYNCROOT/'run_pipeline.py','--videofile',row['input_mkv'],'--reference',row['reference'],
                 '--data_dir',RUN/'data','--min_track','25','--overwrite'],'official_'+key+'_pipeline',cwd=SYNCROOT,limit=300)
            base.update(pipeline_receipt=pipeline,pipeline_log=pipeline['log'])
            if pipeline['returncode'] or pipeline['timed_out']:base['error']='official pipeline failed'
            else:
                sync=self.command([PYTHON,SYNCROOT/'run_syncnet.py','--videofile',row['input_mkv'],'--reference',row['reference'],
                     '--data_dir',RUN/'data','--initial_model',SYNCROOT/'data/syncnet_v2.model'],'official_'+key+'_syncnet',cwd=SYNCROOT,limit=300)
                base.update(syncnet_receipt=sync,syncnet_log=sync['log'])
                try:base['tracks']=parse_tracks(Path(sync['log']).read_text())
                except ValueError as exc:base['track_parse_error']=str(exc)
                if sync['returncode'] or sync['timed_out']:base['error']='official SyncNet failed'
                else:
                    try:
                        if not base['tracks']:raise ValueError(base.get('track_parse_error','no printed tracks'))
                        tracks=base['tracks'];meta=self.track_metadata(row)
                        for i,track in enumerate(tracks):
                            track['metadata']=meta['crop_tracks'][i] if i<len(meta['crop_tracks']) else None
                        base.update(meta,tracks=tracks,status='COMPLETE',sync_c=tracks[0]['sync_c'],sync_d=tracks[0]['sync_d'],av_offset=tracks[0]['av_offset'],track_count=len(tracks))
                    except Exception as exc:base.update(error='parse/metadata failure: '+repr(exc),traceback=traceback.format_exc())
            write(RUN/'scores'/f'{key}.json',base);self.records.append(base)
            if base['status']!='COMPLETE':self.failures.append(dict(sample_id=row['sample_id'],role=row['role'],error=base['error']))
            self.event('cell_finished',index=index,cell=key,status=base['status'],sync_c=base['sync_c'],track_count=len(base['tracks']))
            if (index+1)%10==0:self.event('score_progress',attempted=index+1,completed=sum(r['status']=='COMPLETE' for r in self.records),
                 failed=len(self.failures),remaining=EXPECTED-1-index,disk_bytes=self.disk_bytes())
            if index==0 and base['status']!='COMPLETE':raise RuntimeError('first normal cell failed: stop for engineering review, no retry')
    def finish(self):
        done={(r['sample_id'],r['role']):r for r in self.records}
        all_records=[done.get((r['sample_id'],r['role']),dict(**r,status='NOT_RUN',sync_c=None,sync_d=None,av_offset=None,tracks=[])) for r in self.rows]
        complete=sum(r['status']=='COMPLETE' for r in all_records);failed=sum(r['status']=='FAILED' for r in all_records)
        result=dict(status='COMPLETE' if complete==EXPECTED and not self.terminal_error else 'INCOMPLETE',expected=EXPECTED,completed=complete,failed=failed,pending=len(all_records)-complete-failed,
                    records=all_records,failures=self.failures,terminal_error=self.terminal_error,pid=os.getpid(),elapsed_seconds=time.monotonic()-self.start,
                    protocol_sha256=sha(RUN/'protocol.json') if (RUN/'protocol.json').exists() else None,input_manifest_sha256=sha(RUN/'input_manifest.json') if (RUN/'input_manifest.json').exists() else None,
                    no_training=True,no_TTS_or_TFG_generation=True,worker_children_exited=self.process is None,disk_bytes=self.disk_bytes())
        write(RUN/'summary.json',result);self.event('finished',completed=complete,failed=failed,pending=result['pending'],status=result['status'])
        return result


def validate_raw(rows,config):
    keys={(r['sample_id'],r['role']) for r in rows}
    expected={(r['sample_id'],role) for r in config['samples'] for role in config['roles']}
    if len(rows)!=config['expected_cells'] or len(keys)!=len(rows) or keys!=expected:
        raise ValueError('raw/canonical cell coverage differs from frozen protocol')
    speakers={r['sample_id']:r['speaker_id'] for r in config['samples']}
    for r in rows:
        if r['speaker_id']!=speakers[r['sample_id']]:raise ValueError('true speaker binding')
        if sha(r['path'])!=r['sha256']:raise ValueError('source file changed')
    return rows


def normalize_audio(x,target):
    import numpy as np
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('invalid finite mono input')
    rms=float(np.sqrt(np.mean(x.astype(float)**2)))
    if rms<=0:raise ValueError('silent audio')
    gain=float(target)/rms
    if float(abs(x).max()*gain)>=1:raise ValueError('target normalization would clip; no limiter')
    return x*np.float32(gain),rms,gain


def canonical(run,config):
    import numpy as np
    import soundfile as sf
    rows=validate_raw(read(config['raw_manifest_path'])['records'],config);out=[]
    for row in rows:
        sid,role=row['sample_id'],row['role'];dest=run/'audio'/sid/f'{role}_16k_pcm16.wav'
        dest.parent.mkdir(parents=True,exist_ok=True);f32=dest.with_name(f'{role}_16k_float.wav')
        command=['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',row['path'],'-ac','1','-ar','16000','-c:a','pcm_f32le',str(f32)]
        checked_stage_command(run,command,'canonical_'+sid+'_'+role,ROOT,60)
        x,sr=sf.read(f32,dtype='float32')
        if sr!=16000:raise ValueError('wrong canonical clock')
        y,rms,gain=normalize_audio(x,config['target_rms']);sf.write(dest,y,16000,subtype='PCM_16')
        pcm,_=sf.read(dest,dtype='int16')
        if len(pcm)!=len(x):raise ValueError('canonical duration changed')
        out.append(dict(sample_id=sid,speaker_id=row['speaker_id'],role=role,cohort=config['cohort'],path=str(dest),sha256=sha(dest),
                    samples=len(pcm),sr=16000,source_path=row['path'],source_sha256=row['sha256'],rms_before=rms,gain=gain,
                    rms_after=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))),
                    pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()))
        print('CANONICAL',len(out),len(rows),sid,role,flush=True)
    write(run/'audio_manifest.json',dict(records=out,count=len(out),protocol_sha256=sha(run/'protocol.json')))


def checked_stage_command(run,command,name,cwd,timeout):
    remaining=STAGE_DEADLINE-time.monotonic() if STAGE_DEADLINE is not None else timeout
    if remaining<=0:raise TimeoutError('global stage deadline reached')
    timeout=min(timeout,remaining)
    if shutil.disk_usage(run).free<FREE:raise RuntimeError('free disk below12GiB')
    log=run/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
    start=time.monotonic()
    with log.open('x') as f:
        process=subprocess.Popen(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=process.returncode
            write(run/'receipts'/f'{name}.json',dict(command=command,pid=process.pid,returncode=code,timed_out=True,log_sha256=sha(log)))
            raise TimeoutError(name)
    receipt=dict(command=command,cwd=str(cwd),pid=process.pid,returncode=code,timed_out=False,elapsed_s=time.monotonic()-start,log=str(log),log_sha256=sha(log))
    write(run/'receipts'/f'{name}.json',receipt)
    if code:raise RuntimeError('stage child failed '+name)
    if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>CAP:raise RuntimeError('run output cap exceeded')
    return receipt


def render(run,config):
    rows=validate_raw(read(run/'audio_manifest.json')['records'],config);d=config['downstream'];w2l=ROOT/'third_party/Wav2Lip'
    ckpt=w2l/'checkpoints/wav2lip_gan.pth';face=Path(d['face']);inference=w2l/'inference.py'
    if sha(ckpt)!=d['weights_sha256'] or sha(face)!=d['face_sha256'] or sha(inference)!=d['inference_sha256']:
        raise ValueError('frozen Wav2Lip/face/source binding mismatch')
    Runner().assert_gpu_idle();out=[]
    for row in rows:
        sid,role=row['sample_id'],row['role'];work=run/'render_work'/sid/role;work.mkdir(parents=True,exist_ok=False);(work/'temp').mkdir()
        dest=run/'video'/sid/f'{role}.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
        command=['/home/wjj/.venvs/wav2lip/bin/python',str(inference),'--checkpoint_path',str(ckpt),'--face',str(face),
                 '--audio',row['path'],'--outfile',str(dest),'--face_det_batch_size','4','--wav2lip_batch_size','4','--nosmooth']
        receipt=checked_stage_command(run,command,'render_'+sid+'_'+role,work,240)
        if not dest.is_file() or not dest.stat().st_size:raise ValueError('missing video')
        out.append(dict(sample_id=sid,speaker_id=row['speaker_id'],role=role,cohort=config['cohort'],path=str(dest),sha256=sha(dest),
                    audio_sha256=row['sha256'],receipt=receipt,weights_sha256=sha(ckpt),face_sha256=sha(face),inference_sha256=sha(inference)))
        print('RENDER',len(out),len(rows),sid,role,flush=True)
    write(run/'video_manifest.json',dict(records=out,count=len(out),protocol_sha256=sha(run/'protocol.json'),weights_sha256=sha(ckpt),inference_sha256=sha(inference)))


def main():
    global SOURCE,RUN,CONFIG,EXPECTED,ROLES,MANIFEST_SHAS,STAGE_DEADLINE
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['canonical','render','official'],required=True)
    p.add_argument('--run',required=True);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);args=p.parse_args()
    run=Path(args.run).resolve();protocol=Path(args.protocol).resolve()
    if sha(protocol)!=args.protocol_sha256 or protocol.parent!=run:raise ValueError('frozen protocol path/SHA')
    CONFIG=read(protocol);SOURCE=run;EXPECTED=CONFIG['expected_cells'];ROLES=tuple(CONFIG['roles'])
    for path,expected in CONFIG['source_pins'].items():
        if sha(path)!=expected:raise ValueError('frozen executable/source/weight changed: '+path)
    if str(Path(__file__).resolve()) not in CONFIG['source_pins']:raise ValueError('runner source must be protocol-pinned')
    stage_limit=CONFIG.get('execution',{}).get('stage_timeout_s',10800)
    if not 0<stage_limit<=10800:raise ValueError('stage timeout must be <=10800s')
    STAGE_DEADLINE=time.monotonic()+stage_limit
    if EXPECTED!=len(CONFIG['samples'])*len(ROLES):raise ValueError('protocol count mismatch')
    write(run/'receipts'/f'{args.stage}_launch.json',dict(command=[sys.executable,*sys.argv],pid=os.getpid(),source_sha256=sha(__file__),protocol_sha256=sha(protocol),no_training=True))
    if args.stage=='canonical':canonical(run,CONFIG)
    elif args.stage=='render':render(run,CONFIG)
    else:
        MANIFEST_SHAS={name:sha(run/name) for name in ['audio_manifest.json','video_manifest.json']}
        RUN=run/'official';RUN.mkdir(exist_ok=False)
        runner=Runner();runner.event('started',command=sys.argv,python=sys.executable)
        try:runner.prepare();runner.score()
        except BaseException as exc:
            runner.terminal_error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            if runner.process is not None:runner.kill_own(runner.process);runner.process=None
            write(RUN/'terminal_failure.json',runner.terminal_error)
        result=runner.finish()
        if result['status']!='COMPLETE':raise SystemExit(1)
    write(run/'receipts'/f'{args.stage}_completion.json',dict(status='COMPLETE',pid=os.getpid(),no_training=True,worker_children_exited=True))


if __name__=='__main__':main()

