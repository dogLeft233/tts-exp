"""Read-only official SyncNet evaluation replay plus audio pre-Linear cache.

All media inputs are inherited; only this new run's scratch may be removed.
No model fitting, no new TTS or video generation.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,sys,time,datetime,argparse,shutil,math,pickle,subprocess,inspect,resource
import numpy as np
import torch,cv2,scipy,python_speech_features as psf
from scipy.io import wavfile
from scipy.signal import correlate,correlation_lags
R=Path('/home/wjj/tts-audio/tts-exp');OUT=R/'runs/syncnet_audio_replay20_20261024_attempt01';OFF=R/'third_party/syncnet_python'
sys.path.insert(0,str(OFF))
from SyncNetInstance import SyncNetInstance,calc_pdist
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(p,v):
    p=Path(p);assert not p.exists(),str(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def scalar(z):
    x=torch.from_numpy(z);mu=x.mean(0);val,idx=mu.min(0)
    return dict(C=float(torch.median(mu)-val),D=float(val),B=float(torch.median(mu)),offset=int(15-idx))
def row_support(i):
    start=640*i;return dict(row=i,mfcc_columns=[4*i,4*i+20],raw_samples_inclusive=[max(0,start-1),start+3439],nominal_halfopen_samples=[start,start+3440],center_s=(max(0,start-1)+start+3439)/32000,nominal_center_s=(start+1720)/16000,video_frames=[i,i+5])
def nearest_row(t,n):
    centers=np.asarray([row_support(i)['center_s'] for i in range(n)]);return int(np.argmin(abs(centers-t)))
def patch_index(i,n):return list(range(i-5,i+6)) if i>=5 and i+5<n else None
def clock_lag(x,y,search=4000):
    """y scoring PCM versus x canonical PCM; never shift evaluation inputs."""
    x=x.astype(np.float64);y=y.astype(np.float64);c=correlate(y,x,method='fft');lags=correlation_lags(len(y),len(x));use=abs(lags)<=search;j=np.argmax(c[use]);lag=int(lags[use][j]);assert abs(lag)<search
    ys=max(0,lag);xs=max(0,-lag);size=min(len(y)-ys,len(x)-xs);a=x[xs:xs+size];b=y[ys:ys+size];rho=float(a@b/np.sqrt((a@a)*(b@b)))
    return dict(lag_samples_scoring_minus_canonical=lag,delay_s=lag/16000,correlation=rho,overlap_samples=size,pass_clock=rho>=.90)
def freeze():
    OUT.mkdir(exist_ok=False)
    rows=read(R/'runs/syncnet_preference_probe_20261024_attempt01/input_manifest.json')['rows'];bound=[]
    for x in rows:
        s=read(x['score_path']);t=s['tracks'][0]['metadata'];assert s['track_count']==1
        data=Path(s['pipeline_receipt']['command'][s['pipeline_receipt']['command'].index('--data_dir')+1]);pre=data/'pyavi'/s['reference']/'audio.wav'
        assert sha(t['crop_file'])==t['crop_sha256'] and sha(s['activesd_path'])==s['activesd_sha256']
        bound.append(dict(x,crop_path=t['crop_file'],crop_sha256=t['crop_sha256'],track=t,pretrack_audio_path=str(pre),pretrack_audio_sha256=sha(pre),source_frames=s['frame_count'],intermediate_probe=s['official_intermediate_AVI_probe']))
    pins={str(p):sha(p) for p in [OFF/'SyncNetInstance.py',OFF/'SyncNetModel.py',OFF/'run_syncnet.py',OFF/'run_pipeline.py',OFF/'data/syncnet_v2.model',Path(psf.__file__),Path(inspect.getfile(psf.sigproc)),Path(inspect.getfile(psf.base))]}
    put(OUT/'protocol.json',dict(schema='syncnet_audio_replay20_v1',frozen_utc=utc(),training=False,scope='140 existing first official tracks; audio512 and final audio/video1024; no new TTS/video or preference fitting',tolerances=dict(curve_max_abs=1e-4,curve_rms=1e-5,scalar_CDB_max_abs=1e-4,offset_exact=True,hook_invariance_max_abs=1e-7,mfcc_input_exact=True,clock_correlation_min=.90),
      small_ids=['BAC009S0002W0122_N_0','BAC009S0002W0122_L06_1','BAC009S0002W0122_L17_1'],official=dict(batch_size=20,vshift=15,device='cuda',eval=True,dtype='float32',decoded_pcm_dtype='int16',mfcc='python_speech_features.mfcc defaults; int16 scale; no waveform normalization',hook='netfcaud[3] forward-pre-hook observes512 input; fc output hook observes final1024; lip final1024 for replay only; hooks returnNone'),
      windows=dict(official_mfcc_columns=20,step_samples=640,mfcc_frame_samples=400,mfcc_step_samples=160,preemphasis=.97,nominal_ms=215,actual_raw_support='row i: max(0,640i-1)..640i+3439 inclusive; center is inclusive sample midpoint; nominal center40i+107.5ms',patch='11 consecutive native rows,615ms nominal; index-only; no sufficiency claim',anchors='canonical FA centers + independently measured PCM delay; nearest actual RF center; +/-40ms sensitivity; unsupported edges retained marked missing'),
      safety=dict(min_free_bytes=5*1024**3,timeout_seconds=7200,scratch='Only run-local scratch removed after per-item manifest; no historical deletion'),source_pins=pins,input_manifest_sha256=sha(R/'runs/syncnet_preference_probe_20261024_attempt01/input_manifest.json')))
    put(OUT/'input_manifest.json',dict(rows=bound));put(OUT/'environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi'],text=True),ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True),source_pins=pins,code_sha256=sha(__file__)))
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)
def replay_one(row,s,small):
    started=time.time();item=OUT/'items'/row['id'];item.mkdir(parents=True,exist_ok=False);assert shutil.disk_usage(OUT).free>=5*1024**3
    for k,h in [('crop_path','crop_sha256'),('audio_path','audio_sha256'),('activesd_path','activesd_sha256'),('pretrack_audio_path','pretrack_audio_sha256')]:assert sha(row[k])==row[h]
    scratch=OUT/'scratch';opt=argparse.Namespace(tmp_dir=str(scratch),reference=row['id'],batch_size=20,vshift=15)
    net=s.__S__;aud512=[];aud1024=[];vid1024=[];audio_inputs=[]
    def prehook(module,args):aud512.append(args[0].detach().cpu().clone())
    def ahook(module,args,out):aud1024.append(out.detach().cpu().clone())
    def vhook(module,args,out):vid1024.append(out.detach().cpu().clone())
    def inputhook(module,args):audio_inputs.append(args[0].detach().cpu().clone())
    hooks=[net.netfcaud[3].register_forward_pre_hook(prehook),net.netfcaud[3].register_forward_hook(ahook),net.netfclip[3].register_forward_hook(vhook),net.netcnnaud.register_forward_pre_hook(inputhook)]
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():offset,conf,dist=s.evaluate(opt,row['crop_path'])
    finally:
        for h in hooks:h.remove()
    a512=torch.cat(aud512).numpy();a1024=torch.cat(aud1024).numpy();v1024=torch.cat(vid1024).numpy();n=len(dist)
    assert a512.shape==(n,512) and a1024.shape==(n,1024) and v1024.shape==(n,1024)
    tmp=scratch/row['id'];sr,audio=wavfile.read(tmp/'audio.wav');assert sr==16000 and audio.dtype==np.int16
    mfcc=psf.mfcc(audio,sr).T;cin=torch.cat(audio_inputs).numpy();expected=np.stack([mfcc[:,i*4:i*4+20] for i in range(n)])[:,None].astype(np.float32);assert np.array_equal(cin,expected)
    # Both inputs return exact same outputs before/after a passive hook, checked on actual first batch.
    with torch.no_grad():
        unhooked=net.forward_aud(torch.from_numpy(cin[:20]).cuda()).cpu().numpy()
    hookerr=float(np.max(abs(unhooked-a1024[:20])));assert hookerr<=1e-7
    original=np.asarray(pickle.loads(Path(row['activesd_path']).read_bytes())[0]);assert original.dtype==np.float32 and original.shape==dist.shape
    delta=dist-original;maxerr=float(np.max(abs(delta)));rmserr=float(np.sqrt(np.mean(delta.astype(float)**2)));old,new=scalar(original),scalar(dist);smax=max(abs(old[k]-new[k]) for k in ['C','D','B'])
    tol=read(OUT/'protocol.json')['tolerances'];passed=maxerr<=tol['curve_max_abs'] and rmserr<=tol['curve_rms'] and smax<=tol['scalar_CDB_max_abs'] and new['offset']==old['offset'] and hookerr<=tol['hook_invariance_max_abs']
    # Replay distances a second way from cached final endpoints, original epsilon/padding.
    replay=np.stack([x.numpy() for x in calc_pdist(torch.from_numpy(v1024),torch.from_numpy(a1024),15)]);assert np.array_equal(replay,dist)
    _,canonical=wavfile.read(row['audio_path']);clock=clock_lag(canonical,audio)
    _,pretrack=wavfile.read(row['pretrack_audio_path']);assert np.array_equal(pretrack[:len(audio)],audio) # all frozen tracks start0
    wavfile.write(item/'scoring_audio.wav',16000,audio)
    np.savez_compressed(item/'sequence.npz',audio512=a512,audio1024=a1024,video1024=v1024,mfcc=mfcc.astype(np.float32),distance=dist,original_distance=original,row_indices=np.arange(n,dtype=np.int32))
    files=sorted(tmp.glob('*.jpg'));frame_sha=[dict(index=i,jpeg_name=f.name,sha256=sha(f),bytes=f.stat().st_size) for i,f in enumerate(files)];assert len(files)-5==n or min(len(files),len(audio)//640)-5==n
    first,last=row_support(0),row_support(n-1);frame_bytes=sum(x['bytes'] for x in frame_sha)
    commands=[['ffmpeg','-y','-loglevel','error','-i',row['crop_path'],'-threads','1','-f','image2',str(tmp/'%06d.jpg')],['ffmpeg','-y','-loglevel','error','-i',row['crop_path'],'-async','1','-ac','1','-vn','-acodec','pcm_s16le','-ar','16000',str(tmp/'audio.wav')]]
    rec=dict(id=row['id'],source=row['sample_id'],arm=row['arm'],draw=row['draw'],pass_replay=bool(passed),original=old,replay=new,curve_max_abs=maxerr,curve_rms=rmserr,scalar_max_abs=smax,hook_invariance_max_abs=hookerr,mfcc_exact=True,distance_replay_exact=True,clock=clock,windows=n,mfcc_shape=list(mfcc.shape),frame_count=len(files),source_frame_count=row['source_frames'],synthesized_frame_indices=[i for i in range(len(files)) if i>=row['source_frames']],audio_samples=len(audio),first_support=first,last_support=last,
      audio_tail_samples=len(audio)-last['raw_samples_inclusive'][1]-1,frame_tail_after_last_window=len(files)-(n-1+5),scratch_frame_bytes=frame_bytes,cache_bytes=sum(p.stat().st_size for p in item.iterdir()),elapsed_s=time.time()-started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,commands=commands,input_pins={k:row[k] for k in ['crop_sha256','audio_sha256','activesd_sha256','pretrack_audio_sha256']},sequence_sha256=sha(item/'sequence.npz'),scoring_audio_sha256=sha(item/'scoring_audio.wav'))
    put(item/'frame_manifest.json',frame_sha);put(item/'replay.json',rec)
    # Verify absolute cleanup scope. It contains only this item's official decoding products.
    resolved=tmp.resolve();assert resolved.is_relative_to((OUT/'scratch').resolve()) and resolved.name==row['id'];shutil.rmtree(resolved)
    print('REPLAY',row['id'],passed,'max',maxerr,'rms',rmserr,'clock',clock['correlation'],'sec',round(time.time()-started,2),flush=True)
    return rec
def replay(stage):
    rows=read(OUT/'input_manifest.json')['rows'];p=read(OUT/'protocol.json');small=p['small_ids']
    if stage=='small':rows=[x for x in rows if x['id'] in small]
    else:
        assert read(OUT/'small_receipt.json')['pass_replay'];rows=[x for x in rows if x['id'] not in small]
    s=SyncNetInstance(device='cuda');s.loadParameters(str(OFF/'data/syncnet_v2.model'));s.__S__.eval();torch.set_num_threads(2)
    for x in rows:replay_one(x,s,stage=='small')
    recs=[read(p) for p in sorted((OUT/'items').glob('*/replay.json'))];put(OUT/(stage+'_receipt.json'),dict(pass_replay=all(x['pass_replay'] for x in recs),items=len(recs),max_curve_error=max(x['curve_max_abs'] for x in recs),max_curve_rms=max(x['curve_rms'] for x in recs),failed_ids=[x['id'] for x in recs if not x['pass_replay']],elapsed_item_sum_s=sum(x['elapsed_s'] for x in recs),protocol_sha256=sha(OUT/'protocol.json')))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','small','full']);args=parser.parse_args();t=time.time()
    if args.stage=='freeze':freeze();return
    put(OUT/(args.stage+'_started.json'),dict(utc=utc(),pid=os.getpid(),command=sys.argv,source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),free_bytes=shutil.disk_usage(OUT).free))
    try:replay(args.stage);put(OUT/(args.stage+'_finished.json'),dict(utc=utc(),elapsed_s=time.time()-t,exitcode=0))
    except Exception as e:put(OUT/(args.stage+'_failed.json'),dict(utc=utc(),elapsed_s=time.time()-t,error=repr(e),exitcode=1));raise
if __name__=='__main__':main()
