"""GRID8cal only. No evaluation or SyncNet implementation; no treatment summaries."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, math, os, resource, shutil, subprocess, sys, time, wave
import numpy as np
import real_av_geometry_calibration_20260927 as old
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/grid_geometry_calibration_20260927';TMP=Path('/dev/shm/tts_exp_grid8cal_20260927')
POINTS=old.POINTS;STABLE=old.STABLE;ASSET=old.ASSET
OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
VIEWS=['BASE','REPEAT','SHIFT_-4','SHIFT_-2','SHIFT_2','SHIFT_4','FROZEN','REVERSE','WARP_0.8','WARP_1.2','TX_-8','TX_8','TY_-8','TY_8','ROT_-5','ROT_5','SCALE_0.9','SCALE_1.1']
INTERIOR=np.arange(14,55);LAGS=np.arange(-5,6);TARGET=.0376838172675746
sha=old.sha;ah=old.ah;read=old.read;write=old.write
CPU_PY='/home/wjj/miniconda3/envs/autoavsr/bin/python'
GPU_PY='/home/wjj/.venvs/syncnet/bin/python'
def environment():
    import cv2
    versions={}
    for name in ['numpy','torch','librosa','soundfile','mediapipe','opencv-python','opencv-python-headless','opencv-contrib-python']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'python':sys.executable,'version':sys.version,'packages':versions,'cv2_version':cv2.__version__,'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]}
def environments():
    command="import sys,json;sys.path.insert(0,sys.argv[1]);import grid_geometry_calibration_20260927 as m;print(json.dumps(m.environment()))"
    return {py:json.loads(subprocess.check_output([py,'-c',command,str(ROOT/'scripts/experiments')],text=True)) for py in [CPU_PY,GPU_PY]}
def child_rss(pid):
    total=0
    try:children=(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()
    except FileNotFoundError:return 0
    for child in children:
        try:
            stat=(Path('/proc')/child/'status').read_text().splitlines();total+=next(int(x.split()[1])*1024 for x in stat if x.startswith('VmRSS:'));total+=child_rss(int(child))
        except (FileNotFoundError,StopIteration):pass
    return total

def limits():
    allocated=sum(f.stat().st_blocks*512 for f in OUT.rglob('*') if f.is_file());free=shutil.disk_usage(OUT).free
    # Keep own entire unspent commitment and acquisition agent's remaining20MiB.
    acquisition=ROOT/'runs/grid_fixed24_acquisition_20260927'
    actual=sum(f.stat().st_blocks*512 for f in acquisition.rglob('*') if f.is_file()) if acquisition.exists() else 0
    other=max(0,20*2**20-actual);remaining=max(0,20*2**20-allocated)
    assert allocated<=20*2**20 and free-remaining-other>=5*2**30,(allocated,free,remaining,other)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    temporary=sum(f.stat().st_blocks*512 for f in TMP.rglob('*') if f.is_file()) if TMP.exists() else 0
    children=child_rss(os.getpid());available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    assert temporary<=64*2**20,('temporary_RAM_video_cap',temporary)
    assert rss+children+temporary<=2**30,('combined_extra_RAM',rss,children,temporary)
    assert available>=2**30,('MemAvailable',available)
    return {'tmpfs_bytes':temporary,'child_rss_bytes':children,'memavailable_bytes':available,'allocated_bytes':allocated,'free_bytes':free,'own_remaining_bytes':remaining,'other_reserved_bytes':other,'peak_rss_bytes':rss}

def protocol():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256']
    for f,h in p['dependencies'].items():assert sha(f)==h,f
    assert environment()==p['runtime'][sys.executable],('runtime_changed',sys.executable)
    assert [r['id'] for r in p['calibration']]==['s'+str(i) for i in range(2,10)]
    return p

def freeze():
    OUT.mkdir(exist_ok=True);assert (OUT/'versions/v1/seal.json').exists() and not (OUT/'landmarks').exists()
    b=read(OUT/'cal_binding.json');assert b['acquisition_status']=='PASS' and len(b['rows'])==8
    deps=[Path(__file__),ROOT/'scripts/experiments/check_grid_geometry_calibration_20260927.py',Path(old.__file__),ASSET,old.MP_API,OUT/'cal_binding.json',ROOT/'runs/grid_geometry_protocol_draft_20260927/proposal.md',ROOT/'third_party/Wav2Lip/audio.py',ROOT/'third_party/Wav2Lip/hparams.py',ROOT/'third_party/Wav2Lip/models/wav2lip.py',ROOT/'third_party/Wav2Lip/checkpoints/wav2lip_gan.pth']
    deps+=list(map(Path,b['audit_artifacts']))
    deps+=sorted((ROOT/'third_party/Wav2Lip/models').glob('*.py'))
    for r in b['rows']:
        for field in ['video','pcm','timeline']:assert sha(r[field])==r[field+'_sha256']
        deps.append(Path(r['timeline']))
    p={'revision':'v3_engineering_before_any_forward','revision_reason':'root approved pure-synthetic quasi-periodic counterexample: allfour shift lags unique correct while abs4 E exceeds abs2; dose monotonicity descriptive only, signs/recovery/support unchanged; v1 preserved','prior_seal':sha(OUT/'versions/v2/seal.json'),'runtime':environments(),'status':'frozen_before_any_new_forward','created':time.time(),'authorization':'root8cal only; acquisition may include24 but16eval model analysis and Sync-C LOCKED','calibration':b['rows'],'canonical_audio':'sealed acquisition16k mono PCM16 WAV -> signed int16 /32768 float32; complete waveform RMS/mel; no alternative MPG decoding','target_RMS':TARGET,'reference':{'frame':0,'oval':OVAL,'expansion_each_side':.2,'no_substitution':True},'frames':75,'interior':INTERIOR.tolist(),'minimum_common_valid':33,'max_source_index_before_mel_clamp':68,'acquisition_override':'root authorizes acquisitionagent24 complete media/engineering timelines; thisworker openscal8 only, eval acquisition is not model-analysis permission','input_gate':read(ROOT/'runs/real_av_geometry_calibration_20260927/protocol.json')['input_gate'],'geometry':{'points':POINTS,'pixel_conversion':[360,288],'eye_pair':[33,263],'similarity':'proper2D stable7; no amplitude normalization','pose':'SVD positive uniform scale then RzRyRx degrees; proxy only'},'views':VIEWS,'timing':{'shift':'y[t]=x[clip(t-d,0,74)]','freeze':37,'reverse':'74-t','warp':'round_even(t+(s-1)*74/(2pi)*sin(2pi*t/74))','space':'cv2.INTER_LINEAR BORDER_CONSTANT0 center(180,144)'},'lag':{'candidates':LAGS.tolist(),'sign':'real[t] versus generated[t+lag]','mask':'REAL+RAW only across all candidate lags; FIXED value and mask inaccessible to selection','criterion':'source-equal mean Pearson r; min6 sources; ties within1e-12 abs then signed ascending','after_lock':'three-domain common valid across all candidate lag and lag0, never reselect','amendment':'rootapproved before all newforward; draft preserved'},'controls':{'repeat_rmse':.001,'repeat_r':.99,'spatial_rmse':.01,'spatial_r':.95,'shift_error_max':1,'shift4_E_no_better_shift2_E':'diagnostic_only_v2_pre_forward','frozen_E_max':.05,'reverse_warp_E_loss_min':.1,'cross_RAW_r_min':.3,'cross_RAW_E':'descriptive only, no quality-selection gate','min_all_gates_same_sources':6,'mask':'eachdomain all18views across selflag candidates; crossdomain RAW fourshifts andBASE across global lag+-5; all>=33'},'metric':'E=1-centeredMSE/realvariance; r=Pearson; zeroSD<=1e-6 nonmeasurable; no cal treatment delta summary','interpretation':'globalcalRAWlag corrects common model/acquisition delay; primary is common-delay-corrected trajectory fidelity; lag0 equally prominent, direction disagreement cannot support original-clock sync improvement; recorded reference not unique physiological truth','resources':{'persistent_cap_bytes':20*2**20,'other_acquisition_cap_bytes':20*2**20,'reserve':5*2**30,'rss':2**30,'lease':'/tmp/tts-exp-gpu.lock','temporary':'newFFV1 files /dev/shm only; cap64MiB; own peakRSS+childRSS+tmpfs<=1GiB; MemAvailable>=1GiB; pixelbridge and scientificreceipts before removal'},'dependencies':{str(x):sha(x) for x in deps}}
    write(OUT/'protocol.json',p);write(OUT/'pose_engineering_contract.json',old.synthetic_pose_contract())
    for f in deps[:3]:d=OUT/'code_snapshot'/f.name;d.parent.mkdir(exist_ok=True);shutil.copyfile(f,d)
    write(OUT/'seal.json',{'protocol_sha256':sha(OUT/'protocol.json'),'pre_forward':True,'resources':limits()});print('FROZEN',sha(OUT/'protocol.json'),flush=True)

def frames(path):
    import cv2
    cap=cv2.VideoCapture(str(path));out=[]
    try:
        while True:
            ok,x=cap.read()
            if not ok:break
            assert x.shape==(288,360,3);out.append(x)
    finally:cap.release()
    assert len(out)==75;return out

def geometry(raw,valid,mats,reference,refmat):
    xy=np.asarray(raw,float)[:,:,:2]*[360,288];ref=np.asarray(reference,float)[:,:2]*[360,288];eye=float(np.linalg.norm(ref[0]-ref[3]));assert eye>1
    n=len(raw);ap=np.full(n,np.nan);width=ap.copy();res=ap.copy();pose=np.full((n,3),np.nan);mouth=np.full((n,6,2),np.nan);v=valid.copy()
    for i in range(n):
        if not v[i]:continue
        try:
            s,r,t=old.similarity(xy[i,:7],ref[:7]);a=s*xy[i]@r+t;pose[i]=old.angles(mats[i]);res[i]=np.sqrt(np.mean(np.sum((a[:7]-ref[:7])**2,1)))/eye;mouth[i]=a[7:]/eye;ap[i]=np.linalg.norm(a[7]-a[8])/eye;width[i]=np.linalg.norm(a[9]-a[10])/eye
        except (ValueError,np.linalg.LinAlgError):v[i]=False
    return {'aperture':ap,'width':width,'residual':res,'pose':pose,'mouth':mouth,'valid':v,'reference_angles':old.angles(refmat),'eye_px':eye}

def detect(fs):
    import cv2
    mp,det=old.detector();raw=[];mats=[];valid=[];meta={'pixel_hashes':[],'full_landmark_hashes':[],'face_counts':[],'invalid_reasons':[]};box=None
    try:
        for i,f in enumerate(fs):
            limits();result=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)));n=len(result.face_landmarks);ok=n==1 and len(result.facial_transformation_matrixes)==1
            if ok:
                a=np.array([[q.x,q.y,q.z] for q in result.face_landmarks[0]],np.float32);raw.append(a[POINTS]);mats.append(result.facial_transformation_matrixes[0]);meta['full_landmark_hashes'].append(ah(a))
                if i==0:
                    xy=a[OVAL,:2].astype(float)*[360,288];lo=xy.min(0);hi=xy.max(0);span=hi-lo;box=np.r_[np.maximum(0,np.floor(lo-.2*span)),np.minimum([360,288],np.ceil(hi+.2*span))].astype(int).tolist()
            else:raw.append(np.full((13,3),np.nan,np.float32));mats.append(np.full((4,4),np.nan));meta['full_landmark_hashes'].append(None)
            valid.append(ok);meta['pixel_hashes'].append(ah(f));meta['face_counts'].append(n);meta['invalid_reasons'].append(None if ok else 'not_exactly_one_face_and_matrix')
    finally:det.close()
    return np.asarray(raw,np.float32),np.asarray(valid,bool),np.asarray(mats,float),meta,box

def save_detection(sid,domain,view,fs,ref=None):
    dest=OUT/'landmarks'/sid/domain/view;assert not Path(str(dest)+'.json').exists()
    raw,v,mat,meta,box=detect(fs);dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(Path(str(dest)+'.npz'),raw=raw,valid=v,matrices=mat,pts=np.arange(75)/25)
    if ref is None:ref=(raw[0],mat[0],bool(v[0]))
    g=geometry(raw,v,mat,ref[0],ref[1]) if ref[2] else None
    qc=old.input_qc(g) if g is not None else {'passed':False,'error':'fixed frame0 invalid; no replacement'}
    # MotionSD .005 is a real-input gate, not a generated amplitude-quality gate.
    if domain!='REAL' and g is not None:qc['checks'].pop('motion');qc['passed']=all(qc['checks'].values())
    meta.update({'id':sid,'domain':domain,'view':view,'qc':qc,'box_from_first_frame':box,'npz_sha256':sha(Path(str(dest)+'.npz')),'protocol_sha256':sha(OUT/'protocol.json'),'resources':limits()});write(Path(str(dest)+'.json'),meta)
    return meta,g

def input_stage():
    import cv2
    cv2.setNumThreads(1);p=protocol();rows=[]
    for r in p['calibration']:
        assert r['split']=='cal' and sha(r['video'])==r['video_sha256'];fs=frames(r['video']);pts=old.read_probe_pts(r['video']);assert len(pts)==75 and max(abs(pts-np.arange(75)/25))<1e-6
        m,g=save_detection(r['id'],'REAL','BASE',fs);rows.append({'id':r['id'],'qc':m['qc'],'box':m['box_from_first_frame']});print('INPUT',r['id'],m['qc'],flush=True)
    n=sum(x['qc']['passed'] for x in rows);write(OUT/'input_gate.json',{'status':'PASS' if n>=6 else 'FAIL','passed_sources':n,'fixed_denominator':8,'rows':rows,'action':'continue_calibration' if n>=6 else 'STOP_before_generation_controls_lag','eval_locked':True})

def canonical_pcm(row):
    assert sha(row['pcm'])==row['pcm_sha256']
    with wave.open(str(row['pcm']),'rb') as w:
        assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2);buf=w.readframes(w.getnframes())
    a=np.frombuffer(buf,dtype='<i2');assert len(a)==row['pcm_samples'];assert hashlib.sha256(buf).hexdigest()==row['pcm_payload_sha256']
    return a.astype(np.float32)/np.float32(32768)

def frontend():
    p=protocol();assert read(OUT/'input_gate.json')['status']=='PASS';assert sys.executable==GPU_PY;sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
    rows=[]
    for r in p['calibration']:
        x=canonical_pcm(r);rms=float(np.sqrt(np.mean(x.astype(float)**2)));gain=TARGET/rms;assert np.max(abs(x*gain))<=.98
        items={}
        for c,y in [('RAW',x),('FIXED',(x*gain).astype(np.float32))]:
            mel=audio.melspectrogram(y).astype(np.float32);starts=[int(i*80/25) for i in range(75)];ix=[min(s,mel.shape[1]-16) for s in starts];assert all(ix[i]==starts[i] for i in range(69));assert np.isfinite(mel).all()
            dest=OUT/'frontends'/r['id']/(c+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,mel=mel,indices=np.array(ix));items[c]={'waveform_hash':ah(y),'mel_hash':ah(mel),'chunks_hash':ah(np.array([mel[:,i:i+16] for i in ix])),'indices':ix,'first_clamp':next((i for i in range(75) if ix[i]!=starts[i]),None),'npz_sha256':sha(dest),'peak':float(max(abs(y))),'rms':float(np.sqrt(np.mean(y.astype(float)**2)))}
        rows.append({'id':r['id'],'canonical_pcm_hash':ah(x),'raw_rms':rms,'gain':gain,'gain_dB':20*math.log10(gain),'gain_direction':'up' if gain>1 else ('down' if gain<1 else 'unchanged'),'conditions':items});print('FRONTEND',r['id'],flush=True)
    write(OUT/'frontend_gate.json',{'status':'PASS','rows':rows,'resources':limits()})

def gpu_state():
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    used,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));return {'processes':processes,'memory_mib':used,'util':util,'time':time.time()}

def decode_hashes_ffmpeg(path):
    cmd=['ffmpeg','-v','error','-threads','1','-i',str(path),'-an','-threads','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1'];proc=subprocess.Popen(cmd,stdout=subprocess.PIPE);hs=[]
    while True:
        data=proc.stdout.read(288*360*3)
        if not data:break
        assert len(data)==288*360*3;hs.append(hashlib.sha256(data).hexdigest())
    assert proc.wait()==0 and len(hs)==75;return hs

def generate():
    import fcntl,cv2,torch
    p=protocol();assert read(OUT/'input_gate.json')['status']=='PASS' and read(OUT/'frontend_gate.json')['status']=='PASS'
    assert read(OUT/'independent_inputs.json')['status']=='PASS' and sys.executable==GPU_PY;limits();TMP.mkdir(exist_ok=False);torch.set_num_threads(1);cv2.setNumThreads(1)
    lock=open('/tmp/tts-exp-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        before=[]
        for i in range(3):
            z=gpu_state();assert not z['processes'] and z['memory_mib']<=128 and z['util']<=10;before.append(z)
            if i<2:time.sleep(5)
        write(OUT/'gpu_before.json',before);sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));from models import Wav2Lip
        torch.manual_seed(20260927);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        with torch.device('cuda'):model=Wav2Lip()
        ck=torch.load(str(ROOT/'third_party/Wav2Lip/checkpoints/wav2lip_gan.pth'),map_location='cuda');model.load_state_dict({k.replace('module.',''):v for k,v in ck['state_dict'].items()});del ck;model.eval();limits()
        state_digest=hashlib.sha256();state_shapes={}
        for name,tensor in sorted(model.state_dict().items()):
            state_shapes[name]=list(tensor.shape);state_digest.update(name.encode());state_digest.update(str(tensor.dtype).encode());state_digest.update(str(tuple(tensor.shape)).encode());state_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        write(OUT/'loaded_model_state.json',{'state_sha256':state_digest.hexdigest(),'shapes':state_shapes,'checkpoint_sha256':sha(ROOT/'third_party/Wav2Lip/checkpoints/wav2lip_gan.pth'),'runtime':environment()})
        for row in p['calibration']:
            sid=row['id'];meta=read(OUT/'landmarks'/sid/'REAL/BASE.json')
            if meta['box_from_first_frame'] is None:continue
            f=frames(row['video'])[0];x1,y1,x2,y2=meta['box_from_first_frame'];face=cv2.resize(f[y1:y2,x1:x2],(96,96));masked=face.copy();masked[48:]=0;im=np.concatenate([masked,face],2).transpose(2,0,1).astype(np.float32)/255;it=torch.from_numpy(im[None]).cuda()
            for c in ['RAW','FIXED']:
                z=np.load(OUT/'frontends'/sid/(c+'.npz'));mel=z['mel'];ix=z['indices'];out=TMP/(sid+'_'+c+'.mkv');cmd=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pixel_format','bgr24','-video_size','360x288','-framerate','25','-i','pipe:0','-an','-c:v','ffv1','-level','3','-threads','1','-pix_fmt','bgr0',str(out)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);hs=[];predhash=[]
                for i in range(75):
                    limits();chunk=mel[:,ix[i]:ix[i]+16];mt=torch.from_numpy(chunk[None,None]).cuda()
                    with torch.inference_mode():pred=model(mt,it).cpu().numpy()[0].transpose(1,2,0)*255
                    predhash.append(ah(pred));frame=f.copy();frame[y1:y2,x1:x2]=cv2.resize(pred.astype(np.uint8),(x2-x1,y2-y1));hs.append(ah(frame));proc.stdin.write(frame.tobytes())
                proc.stdin.close();assert proc.wait()==0;ind=decode_hashes_ffmpeg(out);assert ind==hs
                write(OUT/'generation'/sid/(c+'.json'),{'id':sid,'condition':c,'temporary':str(out),'video_sha256':sha(out),'pixel_hashes':hs,'prediction_hashes':predhash,'decoded_bridge_exact':True,'frames':75,'frame0hash':ah(f),'box':[x1,y1,x2,y2],'mel_npz_sha256':sha(OUT/'frontends'/sid/(c+'.npz')),'resources':limits()});print('GENERATED',sid,c,flush=True)
        del model;torch.cuda.empty_cache()
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    write(OUT/'gpu_worker_end.json',{'worker_pid':os.getpid(),'time':time.time(),'lease_released':True,'resources':limits()})

def transformed(fs,view):
    import cv2
    t=np.arange(75)
    if view in ['BASE','REPEAT']:return iter(fs)
    if view.startswith('SHIFT_'):idx=np.clip(t-int(view.split('_')[1]),0,74)
    elif view=='FROZEN':idx=np.full(75,37)
    elif view=='REVERSE':idx=74-t
    elif view.startswith('WARP_'):s=float(view.split('_')[1]);idx=np.rint(t+(s-1)*74/(2*np.pi)*np.sin(2*np.pi*t/74)).astype(int)
    else:
        kind,val=view.split('_');val=float(val);m=cv2.getRotationMatrix2D((180,144),val if kind=='ROT' else 0,val if kind=='SCALE' else 1)
        if kind=='TX':m[0,2]+=val
        if kind=='TY':m[1,2]+=val
        return (cv2.warpAffine(f,m,(360,288),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0) for f in fs)
    return (fs[i] for i in idx)

def controls():
    import cv2
    cv2.setNumThreads(1);p=protocol();assert read(OUT/'input_gate.json')['status']=='PASS' and (OUT/'gpu_release_confirmed.json').exists()
    reuse={(r['id'],r['domain'],r['view']):r for r in read(OUT/'resume_completed_inventory.json')['records']}
    for row in p['calibration']:
        sid=row['id'];z=np.load(OUT/'landmarks'/sid/'REAL/BASE.npz');ref=(z['raw'][0],z['matrices'][0],bool(z['valid'][0]))
        if not ref[2]:continue
        for domain in ['REAL','RAW','FIXED']:
            path=row['video'] if domain=='REAL' else read(OUT/'generation'/sid/(domain+'.json'))['temporary'];fs=frames(path)
            for view in VIEWS:
                if domain=='REAL' and view=='BASE':continue
                prior=OUT/'landmarks'/sid/domain/(view+'.json');array=OUT/'landmarks'/sid/domain/(view+'.npz')
                if prior.exists():
                    anchor=reuse[(sid,domain,view)];assert sha(prior)==anchor['json_sha256'] and sha(array)==anchor['npz_sha256'];saved=read(prior);assert saved['protocol_sha256']==anchor['original_protocol_sha256'];assert (saved['id'],saved['domain'],saved['view'])==(sid,domain,view);assert array.exists() and sha(array)==saved['npz_sha256'];assert [ah(f) for f in transformed(fs,view)]==saved['pixel_hashes'];print('REUSED_EXACT',sid,domain,view,flush=True);continue
                assert not array.exists(),('incomplete_array_requires_audit',str(array))
                save_detection(sid,domain,view,transformed(fs,view),ref);print('CONTROL',sid,domain,view,flush=True)
            if domain!='REAL':
                rec=read(OUT/'generation'/sid/(domain+'.json'));assert [ah(f) for f in fs]==rec['pixel_hashes'];write(OUT/'retention'/sid/(domain+'.json'),{'video_sha256':sha(path),'pixel_bridge':True,'scientific_views':[sha(OUT/'landmarks'/sid/domain/(v+'.npz')) for v in VIEWS],'deleted_only_new_temporary':str(path)});Path(path).unlink()
    write(OUT/'controls_complete.json',{'status':'COMPLETE','resources':limits(),'eval_locked':True})

def trajectory(sid,domain,view='BASE'):
    ref=np.load(OUT/'landmarks'/sid/'REAL/BASE.npz');z=np.load(OUT/'landmarks'/sid/domain/(view+'.npz'));g=geometry(z['raw'],z['valid'],z['matrices'],ref['raw'][0],ref['matrices'][0]);return g['aperture'],g['valid']

def metric(y,x):
    yc=y-y.mean();xc=x-x.mean();sy=float(np.sqrt(np.mean(yc*yc)));sx=float(np.sqrt(np.mean(xc*xc)));mse=float(np.mean((xc-yc)**2));return {'E':1-mse/sy**2 if sy>1e-6 else None,'r':float(np.mean(xc*yc)/(sx*sy)) if min(sx,sy)>1e-6 else None,'centered_rmse':math.sqrt(mse),'reference_sd':sy,'candidate_sd':sx,'uncentered_rmse':float(np.sqrt(np.mean((x-y)**2)))}

def choose(vals,ls=LAGS):
    good=[(int(l),float(v)) for l,v in zip(ls,vals) if v is not None and np.isfinite(v)]
    if not good:return None
    best=max(v for _,v in good);return min((l for l,v in good if abs(v-best)<=1e-12),key=lambda l:(abs(l),l))

def lag_lock():
    protocol();assert (OUT/'controls_complete.json').exists();rows=[];good=[]
    # This function never loads any FIXED landmarks or metadata.
    for item in read(OUT/'input_gate.json')['rows']:
        sid=item['id']
        if not item['qc']['passed']:rows.append({'id':sid,'eligible':False,'reason':'real_input_QC'});continue
        y,yv=trajectory(sid,'REAL');x,xv=trajectory(sid,'RAW');t=INTERIOR[yv[INTERIOR]&np.all(xv[INTERIOR[:,None]+LAGS],axis=1)]
        rr=[metric(y[t],x[t+l])['r'] for l in LAGS] if len(t)>=33 else []
        eligible=len(rr)==11 and all(r is not None for r in rr);rows.append({'id':sid,'eligible':eligible,'indices':t.tolist(),'r_by_lag':rr})
        if eligible:good.append(rr)
    means=np.mean(good,0).tolist() if len(good)>=6 else [];lag=choose(means) if means else None
    write(OUT/'lag_lock.json',{'status':'PASS' if lag is not None else 'FAIL','lag':lag,'rows':rows,'source_count':len(good),'mean_r_by_lag':means,'FIXED_never_read':True,'protocol_sha256':sha(OUT/'protocol.json'),'created':time.time()})

def diagnostics():
    p=protocol();lock=read(OUT/'lag_lock.json');lag=lock['lag'];rows=[]
    for item in read(OUT/'input_gate.json')['rows']:
        sid=item['id'];row={'id':sid,'input_pass':item['qc']['passed'],'domains':{},'passed':False}
        if not item['qc']['passed']:rows.append(row);continue
        for domain in ['REAL','RAW','FIXED']:
            data={v:trajectory(sid,domain,v) for v in VIEWS};mask=np.ones(41,bool)
            for x,v in data.values():mask&=np.all(v[INTERIOR[:,None]+LAGS],axis=1)
            t=INTERIOR[mask];d={'indices':t.tolist(),'support_pass':len(t)>=33,'controls':{},'baseline_QC_pass':read(OUT/'landmarks'/sid/domain/'BASE.json')['qc']['passed']}
            if len(t)>=33:
                y=data['BASE'][0];scores={v:metric(y[t],x[t]) for v,(x,_) in data.items()};repeat=scores['REPEAT'];flags=[repeat['r'] is not None and repeat['r']>=.99 and repeat['centered_rmse']<=.001]
                for v,s in scores.items():
                    check=True;extra={}
                    if v.startswith('SHIFT_'):
                        shift=int(v.split('_')[1]);rr=[metric(y[t],data[v][0][t+l])['r'] for l in LAGS];best=choose(rr);check=best is not None and best*shift>0 and abs(best-shift)<=1;extra={'r_by_lag':rr,'recovered_lag':best,'expected':shift}
                    elif v=='FROZEN':check=s['E'] is not None and s['E']<=.05
                    elif v=='REVERSE' or v.startswith('WARP_'):check=s['E'] is not None and repeat['E'] is not None and repeat['E']-s['E']>=.1
                    elif v.startswith(('TX_','TY_','ROT_','SCALE_')):check=s['r'] is not None and s['r']>=.95 and s['centered_rmse']<=.01
                    elif v=='REPEAT':check=flags[0]
                    if v!='BASE':flags.append(check)
                    d['controls'][v]={**s,**extra,'passed':bool(check)}
                dose=all(scores[f'SHIFT_{sign*4}']['E']<=scores[f'SHIFT_{sign*2}']['E']+1e-12 for sign in [-1,1]) if all(scores[f'SHIFT_{d}']['E'] is not None for d in [-4,-2,2,4]) else None;d['shift_dose_monotone_diagnostic']=dose;d['shift_dose_is_gate']=False;d['passed']=all(flags) and d['baseline_QC_pass']
            else:d['passed']=False
            row['domains'][domain]=d
        cross={'passed':False,'lag':lag}
        if lag is not None:
            y,yv=trajectory(sid,'REAL');raw={v:trajectory(sid,'RAW',v) for v in ['BASE','SHIFT_-4','SHIFT_-2','SHIFT_2','SHIFT_4']};ls=LAGS+lag;mask=yv[INTERIOR].copy()
            for x,v in raw.values():mask&=np.all(v[INTERIOR[:,None]+ls],axis=1)
            t=INTERIOR[mask];cross['indices']=t.tolist()
            if len(t)>=33:
                base=metric(y[t],raw['BASE'][0][t+lag]);rr=[metric(y[t],raw['BASE'][0][t+l])['r'] for l in ls];b=choose(rr,ls);shiftchecks=[];details={}
                for v in list(raw)[1:]:
                    shift=int(v.split('_')[1]);cur=[metric(y[t],raw[v][0][t+l])['r'] for l in ls];best=choose(cur,ls);ok=best is not None and b is not None and abs((best-b)-shift)<=1 and (best-b)*shift>0;shiftchecks.append(ok);details[v]={'r_by_lag':cur,'best':best,'expected_relative':shift,'observed_relative':best-b if best is not None and b is not None else None,'passed':ok}
                cross.update({'RAW':base,'RAW_lag0_same_support':metric(y[t],raw['BASE'][0][t]),'RAW_E_descriptive_only':True,'base_search_r':rr,'base_best':b,'shift':details,'passed':bool(base['r'] is not None and base['r']>=.3 and all(shiftchecks))})
            # Three-domain validity now only, after immutable RAW-onlylag selection. No cal treatment effects.
            a,av=trajectory(sid,'RAW');f,fv=trajectory(sid,'FIXED');mainmask=yv[INTERIOR]&np.all(av[INTERIOR[:,None]+LAGS],axis=1)&np.all(fv[INTERIOR[:,None]+LAGS],axis=1);mt=INTERIOR[mainmask];cross['three_domain_main_indices']=mt.tolist();cross['three_domain_main_support_pass']=len(mt)>=33;cross['passed']&=len(mt)>=33
        row['cross_domain']=cross;row['passed']=bool(all(v['passed'] for v in row['domains'].values()) and cross['passed']);rows.append(row)
    n=sum(r['passed'] for r in rows);write(OUT/'calibration_diagnostics.json',{'status':'PASS' if n>=6 else 'FAIL','passed_sources':n,'fixed_denominator':8,'rows':rows,'lag_lock_sha256':sha(OUT/'lag_lock.json'),'no_treatment_effect_summary':True,'SyncNet_NOT_RUN':True,'eval_locked':True})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','input_stage','frontend','generate','controls','lag_lock','diagnostics']);a=ap.parse_args();assert sys.executable==(GPU_PY if a.stage in ['frontend','generate'] else CPU_PY),('stage_environment',a.stage,sys.executable);globals()[a.stage]()
