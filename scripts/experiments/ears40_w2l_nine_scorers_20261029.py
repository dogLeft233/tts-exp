"""EARS40 paired English Wav2Lip experiment. No API, training or timing edits.

Run-local protocol is immutable; resume skips completed scientific artifacts.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/wjj/tts-audio/tts-exp')
NAME = 'ears40_w2l_nine_scorers_20261029_attempt01'
RUN = ROOT/'runs'/NAME
SHM = Path('/dev/shm')/NAME
SOURCE = ROOT/'runs/ears40_cloud_qwen31_20261028_attempt01'
SCRIPTS = ROOT/'scripts/experiments'
W2L = ROOT/'third_party/Wav2Lip'
SYNC = ROOT/'third_party/syncnet_python'
RMS = .0376838172675746
MODELS = ['SyncNet','CoGenAV','PEAVS','SyncReward','AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def command(args, name, cwd=ROOT, timeout=3600):
    args=list(map(str,args)); dest=RUN/'logs'/(name+'.log')
    dest.parent.mkdir(parents=True,exist_ok=True)
    # Preserve execution failures rather than overwriting their logs on repairs.
    if dest.exists():dest=dest.with_name(dest.stem+'_'+str(time.time_ns())+'.log')
    start=time.time()
    with dest.open('x') as f:
        child=subprocess.Popen(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        write(RUN/'active_process.json',dict(name=name,pid=child.pid,args=args,start_unix=start,timeout=timeout))
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.terminate();rc=child.wait(timeout=30)
            write(RUN/'receipts'/(name+'_timeout.json'),dict(pid=child.pid,timeout=timeout,returncode=rc))
            raise
    write(RUN/'receipts'/(dest.stem+'.json'),dict(args=args,cwd=str(cwd),returncode=rc,elapsed_seconds=time.time()-start,log=str(dest),log_sha256=sha(dest)))
    if rc:raise RuntimeError('stage failed '+name+'; '+str(dest))


def capture(args):return subprocess.check_output(list(map(str,args)))


def video_identity(path):
    raw=capture(['ffmpeg','-v','error','-i',path,'-map','0:v:0','-an','-f','framemd5','-']).decode()
    pixels=[line.split(',')[-1].strip() for line in raw.splitlines() if not line.startswith('#')]
    raw=read_probe=capture(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',path])
    pts=[float(f['best_effort_timestamp_time']) for f in json.loads(raw)['frames']]
    return pixels,pts


def normalize(pcm, target=RMS):
    import numpy as np
    x=np.asarray(pcm,dtype=np.float64)
    assert x.ndim==1 and len(x)>0 and np.isfinite(x).all()
    rms=float(np.sqrt(np.mean(x*x)));assert rms>0
    y=x*(target/rms)
    assert np.max(np.abs(y))<1, 'RMS normalization would clip'
    return y,rms


def shift_pcm(x, samples):
    import numpy as np
    y=np.zeros_like(x)
    if samples>0:y[samples:]=x[:-samples]
    elif samples<0:y[:samples]=x[-samples:]
    else:y[:]=x
    return y


def sync_scalar(matrix):
    import torch
    z=torch.from_numpy(matrix.T.copy());curve=z.mean(1);d,k=curve.min(0)
    return dict(C=float(torch.median(curve)-d),D=float(d),B=float(torch.median(curve)),offset=int(15-k),windows=len(matrix))


def prepare():
    import numpy as np
    import soundfile as sf
    RUN.mkdir(parents=True,exist_ok=True);SHM.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():
        print('Existing immutable protocol; prepare skipped',flush=True);return
    manifest=read(SOURCE/'audio_manifest.json')['records'];assert len(manifest)==40
    gp=read(ROOT/'runs/qwen_codec_roundtrip_20261022_attempt01/input_manifest.json')['generation']
    assert sha(gp['face'])==gp['face_sha256']
    records=[]
    for r in sorted(manifest,key=lambda x:x['speaker_id']):
        sid=r['speaker_id']
        for arm,field in [('N','natural'),('T','tts16')]:
            original=Path(r[field]);pcm,sr=sf.read(original,dtype='float64')
            assert sr==16000
            y,raw_rms=normalize(pcm)
            key=f'{sid}_{arm}_0';path=RUN/'audio'/(key+'.wav');path.parent.mkdir(exist_ok=True)
            assert not path.exists();sf.write(path,y,16000,subtype='PCM_16')
            actual,ar=sf.read(path,dtype='int16');assert ar==16000 and len(actual)==len(pcm)
            records.append(dict(id=key,sample_id=sid,speaker_id=sid,gender=r['gender'],arm=arm,draw=0,
                audio_path=str(path),audio_sha256=sha(path),original_audio_path=str(original),original_audio_sha256=sha(original),
                samples=len(pcm),duration_s=len(pcm)/16000,raw_rms=raw_rms,normalization_gain=RMS/raw_rms,
                achieved_rms=float(np.sqrt(np.mean((actual.astype(np.float64)/32768)**2))),
                peak_abs_pcm16=int(np.abs(actual.astype(np.int32)).max()),text=r['text'],
                video_path=str(RUN/'inputs'/(key+'.mp4')),native_frames=None,center_start_5s=None,eligible_5s=None))
    assert len({r['sample_id'] for r in records})==40 and min(r['duration_s'] for r in records)>5
    pins=[Path(__file__),SCRIPTS/'ears40_extra_scorers_20261029.py',SCRIPTS/'ears40_retrieval_scorers_20261029.py',
        W2L/'inference.py',W2L/'checkpoints/wav2lip_gan.pth',SYNC/'run_pipeline.py',SYNC/'run_syncnet.py',SYNC/'SyncNetInstance.py',SYNC/'data/syncnet_v2.model']
    optional=['ears40_regressor_scorers_20261029.py','ears40_cogenav_scorer_20261029.py']
    for name in optional:
        if (SCRIPTS/name).exists():pins.append(SCRIPTS/name)
    for name in ['mtd_official_audit_20261028.py','mtd_official_cloud_rescore_20261028.py',
            'vocalist_official_cloud_rescore_20261028.py','avs_u_zh_pilot_20261027.py',
            'cloud_mtd_synchformer_20261028.py','sparsesync_official_cloud_rescore_20261028.py',
            'cogenav_sync_audit_20261024.py']:
        pins.append(SCRIPTS/name)
    deploy=Path('/home/wjj/av-sync-eval-20261028')
    for name in ['MTDVocaLiST/test_stu.py','vocalist/test_lrs2.py',
            'Synchformer-source/example.py','SparseSync-source/scripts/example.py']:
        pins.append(deploy/name)
    for name in ['score_peavs_official64_20261027.py','score_syncreward_official_20261027.py']:
        pins.append(Path('/home/wjj/av-eval-20261026')/name)
    protocol=dict(status='FROZEN_BEFORE_GENERATION_AND_SCORING',run_id=NAME,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cohort='40 EARS English speakers,20female20male; one target sentences_16_regular per speaker, one preexisting cloud TTS each',
        reference='independent same-speaker sentences_15_regular; no same-target cloning reference',
        selection='all40 already synthesized before this experiment; no outcome selection or replacement',
        provider='existing qwen-audio-3.1-tts-plus draw, no new API',
        input_format_note='Provider streaming WAV RIFF/data lengths are placeholders; actual decoded PCM/canonical16 headers valid; originals unchanged',
        normalization=dict(target_rms=RMS,format='mono16k PCM16, same sample count; no trimming, stretching, denoising or timing correction'),
        generation=gp,render_options=['--face_det_batch_size','4','--wav2lip_batch_size','4','--nosmooth'],
        video_audio='W2L video stream unchanged; canonical PCM muxed losslessly as ALAC MP4 for metrics and PCM MKV for official SyncNet pipeline',
        primary_endpoints={'SyncNet':'official float32 Sync-C, wholeclip original pipeline',
            'CoGenAV':'paper Eq1 mean rectified cosine, native fullclip context',
            'PEAVS':'official mean score with fixed sorted batch composition, official batch64; padding caveat retained',
            'SyncReward':'official Stage2 BF16 center4.75sec low-memory encode batch1',
            'AVS_u':'paper-definition project reproduction, mean frame cosine native fullclip context',
            'MTDVocaLiST':'verbatim official K5 zero +/-1frame retrieval accuracy, wholeclip31candidates',
            'VocaLiST':'verbatim original author K5 zero +/-1frame retrieval accuracy, wholeclip31candidates',
            'Synchformer':'official LRS3 21-class center5sec P(offset0), project scalar endpoint',
            'SparseSync':'official LRS3 21-class center5sec P(offset0), project scalar endpoint'},
        controls='all40 natural videos, real fixed-length zero-filled PCM +/-400ms; unchanged videos. >=64/80 primary decreases qualifies response; retrieval peak-change recovery separately; failure never removes preference results',
        peavs_batch_policy='native80 sorted default64 gives64+16; controls21/19 source triples gives63/57 with each control-own natural baseline in same batch; cached-feature native batch1 sensitivity separately; compare no cross-batch baseline',
        sync_control_scope='physical +/-400ms on the actual decoded official scoring crop PCM, preserving existing pipeline processing delay; no per-item correction',
        secondary='allK7..15; classifier negative expected absolute offset; score saturation/floor, sex groups descriptive only; conflict vs Sync-C strict sign with ties separate',
        common_support_sensitivity='central5sec CPU aggregation only where saved per-frame embeddings/matrices permit; retains original encoding context and candidate geometry, not a new uniform5sec benchmark; no extra9model crop re-encodes',
        statistics='40 paired speaker differences, higher better; mean T-N,95% paired speaker bootstrap10000 seed20261029; two-sided MonteCarlo paired sign-flip100000(+1 correction),Holm9 primary family; all failures/missing remain denominators',
        interpretation='single shared English prompt, fixed face/Wav2Lip/provider and RMS chain; not all English or human perceptual accuracy; no language interaction identified by cross-cohort comparison',
        source_manifest_sha256=sha(SOURCE/'audio_manifest.json'),pins={str(p):sha(p) for p in pins})
    write(RUN/'audio_inputs.json',records);write(RUN/'protocol.json',protocol)
    shutil.copy2(__file__,RUN/'source_frozen.py')
    write(RUN/'status.json',dict(stage='FROZEN',media=80,sources=40))
    print('FROZEN',len(records),'media',flush=True)


def render_sync():
    import numpy as np
    import soundfile as sf
    rows=read(RUN/'audio_inputs.json');protocol=read(RUN/'protocol.json');scores=[];inputs=[]
    for r in rows:
        key=r['id'];receipt=RUN/'syncnet/items'/(key+'.json')
        if receipt.exists():
            item=read(receipt);scores.append(item);inputs.append(item['input']);continue
        assert sha(r['audio_path'])==r['audio_sha256']
        scratch=SHM/'render'/key;(scratch/'temp').mkdir(parents=True,exist_ok=True)
        video=RUN/'video'/(key+'.mp4');video.parent.mkdir(exist_ok=True)
        if not video.exists():
            command(['/home/wjj/.venvs/wav2lip/bin/python',W2L/'inference.py','--checkpoint_path',W2L/'checkpoints/wav2lip_gan.pth',
                '--face',protocol['generation']['face'],'--audio',r['audio_path'],'--outfile',video,
                '--face_det_batch_size','4','--wav2lip_batch_size','4','--nosmooth'],'render_'+key,scratch)
        mux=Path(r['video_path']);mux.parent.mkdir(exist_ok=True)
        command(['ffmpeg','-v','error','-nostdin','-y','-i',video,'-i',r['audio_path'],'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','alac','-map_metadata','-1',mux],'mux_'+key)
        audio=sf.read(r['audio_path'],dtype='int16')[0]
        assert capture(['ffmpeg','-v','error','-i',mux,'-map','0:a:0','-f','s16le','-'])==audio.astype('<i2').tobytes()
        smux=SHM/'sync_inputs'/(key+'.mkv');smux.parent.mkdir(exist_ok=True)
        command(['ffmpeg','-v','error','-nostdin','-y','-copyts','-i',video,'-i',r['audio_path'],'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le','-ar','16000','-ac','1','-avoid_negative_ts','disabled',smux],'sync_mux_'+key)
        identity=[video_identity(p) for p in [video,mux,smux]]
        assert identity[0][0]==identity[1][0]==identity[2][0]
        assert all(np.allclose(identity[0][1],x[1],atol=1e-6,rtol=0) for x in identity[1:])
        assert capture(['ffmpeg','-v','error','-i',smux,'-map','0:a:0','-f','s16le','-'])==audio.astype('<i2').tobytes()
        write(RUN/'media_identity'/(key+'.json'),dict(decoded_pixels_exact=True,frame_pts_exact=True,pcm_exact=True,
            frames=len(identity[0][0]),pts=identity[0][1],frame_hashes=identity[0][0],video_sha256=sha(video),alac_sha256=sha(mux),sync_mkv_sha256=sha(smux)))
        sd=SHM/'sync_data';ref='ears29_'+key;common=['--videofile',smux,'--reference',ref,'--data_dir',sd]
        command(['/home/wjj/.venvs/syncnet/bin/python',SYNC/'run_pipeline.py',*common,'--min_track','25','--overwrite'],'pipeline_'+key,SYNC)
        command(['/home/wjj/.venvs/syncnet/bin/python',SYNC/'run_syncnet.py',*common,'--initial_model',SYNC/'data/syncnet_v2.model'],'sync_'+key,SYNC)
        work=sd/'pywork'/ref;dist=pickle.loads((work/'activesd.pckl').read_bytes());assert len(dist)==1 and dist[0].shape[1]==31
        target=RUN/'syncnet/crops'/key;target.mkdir(parents=True,exist_ok=True)
        for f in ['activesd.pckl','tracks.pckl','faces.pckl']:
            if (work/f).exists():shutil.copy2(work/f,target/f)
        avi=sd/'pycrop'/ref/'00000.avi';assert avi.exists();shutil.copy2(avi,target/'00000.avi')
        info=json.loads(capture(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',mux]))['streams'][0]
        assert info['r_frame_rate']=='25/1';frames=int(info['nb_read_frames']);native=min(frames,r['samples']//640)
        rr=dict(r,native_frames=native,source_video_frames=frames,video_sha256=sha(mux),eligible_5s=native>=125,center_start_5s=(native-125)//2)
        matrix=RUN/'syncnet/matrices'/(key+'.npy');matrix.parent.mkdir(exist_ok=True);np.save(matrix,dist[0])
        item=dict(id=key,source=r['sample_id'],arm=r['arm'],input=rr,crop=str(target/'00000.avi'),matrix=str(matrix),**sync_scalar(dist[0]))
        write(receipt,item);scores.append(item);inputs.append(rr)
        write(RUN/'syncnet/scores.json',scores);write(RUN/'inputs.json',inputs)
        write(RUN/'status.json',dict(stage='RENDER_SYNC',completed=len(scores),total=80,last=key))
        print('RENDER_SYNC',len(scores),key,item['C'],flush=True)
        # Only our regenerable SHM scratch is released; scientific matrices/crops are persistent.
        for root in [scratch,sd/'pyavi'/ref,sd/'pyframes'/ref,sd/'pytmp'/ref,sd/'pywork'/ref,sd/'pycrop'/ref]:
            if root.exists() and root.resolve().is_relative_to(SHM.resolve()):shutil.rmtree(root)
        smux.unlink(missing_ok=True)
    assert len(scores)==80;write(RUN/'inputs.json',inputs)
    write(RUN/'syncnet/status.json',dict(status='NATIVE_COMPLETE',scored=80))


def sync_controls():
    import numpy as np
    import soundfile as sf
    import torch
    from types import SimpleNamespace
    sys.path.insert(0,str(SYNC));from SyncNetInstance import SyncNetInstance
    torch.set_num_threads(4);model=SyncNetInstance(device='cuda');model.loadParameters(str(SYNC/'data/syncnet_v2.model'))
    rows=read(RUN/'syncnet/scores.json');out=RUN/'syncnet/controls.json';controls=read(out) if out.exists() else {}
    for r in rows:
        if r['arm']!='N' or r['id'] in controls:continue
        key=r['id'];scratch=SHM/'sync_controls'/key;scratch.mkdir(parents=True,exist_ok=True)
        raw=scratch/'original.wav'
        command(['ffmpeg','-v','error','-nostdin','-y','-i',r['crop'],'-map','0:a:0','-ac','1','-ar','16000','-c:a','pcm_s16le',raw],'sync_control_decode_'+key)
        pcm,sr=sf.read(raw,dtype='int16');assert sr==16000
        opt=SimpleNamespace(tmp_dir=str(scratch/'evaltmp'),reference='native_replay',batch_size=20,vshift=15)
        _,replay_conf,replay_ds=model.evaluate(opt,r['crop'])
        original_ds=np.load(r['matrix'])
        replay_error=float(np.max(np.abs(original_ds-replay_ds)))
        assert replay_error<2e-5 and abs(float(replay_conf)-r['C'])<2e-5,'Official crop replay mismatch'
        results={}
        for samples in [-6400,6400]:
            wav=scratch/(str(samples)+'.wav');sf.write(wav,shift_pcm(pcm,samples),16000,subtype='PCM_16')
            mov=scratch/(str(samples)+'.avi')
            command(['ffmpeg','-v','error','-nostdin','-y','-i',r['crop'],'-i',wav,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le',mov],'sync_control_mux_'+key+'_'+str(samples))
            opt=SimpleNamespace(tmp_dir=str(scratch/'evaltmp'),reference=str(samples),batch_size=20,vshift=15)
            offset,conf,ds=model.evaluate(opt,str(mov));path=RUN/'syncnet/matrices'/(key+'_'+str(samples)+'.npy');np.save(path,ds)
            pixel0,pts0=video_identity(r['crop']);pixel1,pts1=video_identity(mov)
            assert pixel0==pixel1 and np.allclose(pts0,pts1,atol=1e-6,rtol=0)
            scalars=sync_scalar(ds)
            results[str(samples)]=dict(**scalars,shift_samples=samples,primary_decreased=float(conf)<r['C'],
                relative_peak_error_frames=abs(scalars['offset']-r['offset']+samples/640),decoded_video_exact=True)
        controls[key]=dict(id=key,baseline=r['C'],outputs=results,pcm_origin='actual decoded official crop PCM',samples=len(pcm),zero_replay_max_error=replay_error)
        write(out,controls);print('SYNC_CONTROL',len(controls),key,flush=True)
        if scratch.resolve().is_relative_to(SHM.resolve()):shutil.rmtree(scratch)
    assert len(controls)==40
    decreased=sum(v['primary_decreased'] for r in controls.values() for v in r['outputs'].values())
    write(RUN/'syncnet/control_summary.json',dict(tested=80,decreased=decreased,required=64,status='PASS' if decreased>=64 else 'CONTROL_FAILED'))


def execute():
    import fcntl
    lock=(RUN/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stages=[('render_sync','syncnet'),('sync_controls','syncnet'),
      ('cogenav','cogenav-sync-20261024'),('avs_u','cogenav-sync-20261024'),
      ('synchformer','synchformer-20261028'),('sparsesync','sparsesync-20261028'),
      ('mtd','mtdvocalist-20261028'),('vocalist','vocalist-20261028'),
      ('peavs','peavs'),('syncreward','syncreward')]
    for stage,env in stages:
        receipt=RUN/'stages'/(stage+'.json')
        if receipt.exists() and read(receipt)['status']=='COMPLETE':continue
        gpu=capture(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip()
        assert int(gpu)>5000,'Insufficient GPU memory; do not disturb foreign processes'
        script=Path(__file__) if stage in ['render_sync','sync_controls'] else SCRIPTS/(
            'ears40_extra_scorers_20261029.py' if stage in ['avs_u','synchformer','sparsesync'] else
            'ears40_retrieval_scorers_20261029.py' if stage in ['mtd','vocalist'] else
            'ears40_regressor_scorers_20261029.py' if stage in ['peavs','syncreward'] else
            'ears40_cogenav_scorer_20261029.py')
        write(RUN/'status.json',dict(stage=stage,status='RUNNING',start_unix=time.time()))
        args=[f'/home/wjj/.venvs/{env}/bin/python',script]
        if stage!='cogenav':args.append(stage)
        command(args,'stage_'+stage,timeout=21600)
        write(receipt,dict(status='COMPLETE',ended_unix=time.time(),source_sha256=sha(script)))
    write(RUN/'status.json',dict(stage='SCORING_COMPLETE',models=9))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','render_sync','sync_controls','execute']);a=p.parse_args()
    globals()[a.stage]()
