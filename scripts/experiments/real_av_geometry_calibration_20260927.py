"""Frozen, calibration-only continuous geometry experiment; no effect estimates."""
from pathlib import Path
import argparse, hashlib, json, math, os, resource, shutil, subprocess, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/real_av_geometry_calibration_20260927'
FEAS = ROOT / 'runs/real_av_geometry_feasibility_20260927'
ASSET = ROOT / 'checkpoints/mediapipe/face_landmarker.task'
STABLE = [33,133,362,263,168,6,197]
MOUTH = [13,14,61,291,0,17]
POINTS = STABLE + MOUTH
MP_API = Path('/home/wjj/miniconda3/envs/autoavsr/lib/python3.8/site-packages/mediapipe/tasks/python/vision/face_landmarker.py')

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def limits():
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    free=shutil.disk_usage(OUT).free
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert total <= 40*2**20, ('persistent budget',total)
    assert free >= 5*2**30, ('disk reserve',free)
    assert rss <= 2**30, ('host RSS exceeds extra-RAM cap',rss)
    return {'persistent_bytes':total,'free_bytes':free,'peak_process_rss_bytes':rss}

def rotation(m):
    """Extract the nearest proper rotation; never transform normalized xyz."""
    m=np.asarray(m,dtype=np.float64)
    if m.shape!=(4,4) or not np.isfinite(m).all():raise ValueError('invalid pose matrix')
    if not np.allclose(m[3],[0,0,0,1],atol=1e-5):raise ValueError('pose last row')
    u,s,vt=np.linalg.svd(m[:3,:3]);r=u@vt
    if np.linalg.det(r)<0 or min(s)<=0 or max(s)/min(s)>1.02:raise ValueError('non-uniform/reflected pose')
    return r
def angles(m):
    r=rotation(m)
    # R = Rz(roll) Ry(yaw) Rx(pitch), radians -> degrees, right handed metric camera axes.
    yaw=math.asin(float(np.clip(-r[2,0],-1,1)))
    pitch=math.atan2(r[2,1],r[2,2]);roll=math.atan2(r[1,0],r[0,0])
    return np.rad2deg([yaw,pitch,roll])
def synthetic_pose_contract():
    cases=[]
    for yaw,pitch,roll in [(0,0,0),(20,0,0),(-20,0,0),(0,20,0),(0,-20,0),(0,0,20),(12,-8,5)]:
        y,p,r=np.deg2rad([yaw,pitch,roll]);rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
        ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        rz=np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
        m=np.eye(4);m[:3,:3]=1.3*rz@ry@rx;m[:3,3]=[2,3,-40]
        error=float(np.max(np.abs(angles(m)-[yaw,pitch,roll])));assert error<1e-10
        cases.append({'angles':[yaw,pitch,roll],'error':error})
    api=MP_API.read_text()
    assert 'matrix_data.layout' in api and 'matrix_data.rows' in api
    return {'status':'PASS','cases':cases,'api_sha256':sha(MP_API),'definition':'matrix supplied canonical-to-runtime metric camera; SVD uniform scale removed; RzRyRx Euler yaw/pitch in degrees; no inverse on normalized landmarks','limitation':'synthetic algebra+API contract validation, not empirical anatomical angle ground truth'}

def freeze():
    assert not (OUT/'protocol.json').exists()
    proposed=read(FEAS/'proposed_support.json');rows=proposed['records'];assert len(rows)==24
    cal=[r for r in rows if r['proposed_split']=='cal'];assert len(cal)==8
    deps=[Path(__file__),ROOT/'scripts/experiments/check_real_av_geometry_calibration_20260927.py',ROOT/'tests/test_real_av_geometry_calibration_20260927.py',ASSET,MP_API,FEAS/'proposed_support.json',FEAS/'proposed_audio_binding.json',ROOT/'third_party/Wav2Lip/audio.py',ROOT/'third_party/Wav2Lip/hparams.py',ROOT/'third_party/Wav2Lip/checkpoints/wav2lip_gan.pth']
    proto={'status':'frozen_before_new_landmark_or_generation_forward','created':time.time(),'authorization':'root explicit8cal only;16eval locked; no cal RAW-FIXED treatment effects','calibration':cal,'evaluation_locked_ids':[r['sample_id'] for r in rows if r['proposed_split']=='eval'],
      'pixels':{'source':'224square LRS3 face track','reference_frame':0,'core_frames':100,'core_start':'proposal absolute frame','reader':'sequential OpenCV BGR, independent ffprobe PTS; ffmpeg first-cal independent decode bridge','generation_box':[0,0,224,224],'generation_frame':'static exact source frame0 for both arms','audio':'complete source WAV RMS and complete waveform mel, original absolute frame int(frame*80/25) index, no time crop before frontend','target_RMS':.0376838172675746,'gain':'target / complete waveform RMS; float32 waveform into official mel; peak>.98 engineering failure; no limiter'},
      'geometry':{'asset_sha256':sha(ASSET),'running_mode':'IMAGE','delegate':'CPU','num_faces':2,'points':POINTS,'stable':STABLE,'mouth':MOUTH,'coordinates':'raw normalized xyz retained for selected scientific points; convert xy to pixels before2Dsimilarity; no per-clip lip amplitude normalization','pose':'API canonical-to-runtime4x4 metric matrix; SVD remove positive uniform scale; RzRyRx Euler; no inverse of normalized xyz','reference_eye_pair':[33,263]},
      'input_gate':{'unique_valid_fraction_min':.95,'missing_run_max':2,'reference_abs_yaw_pitch_max_deg':20,'relative_abs_yaw_pitch_p95_max_deg':10,'stable_residual_median_max_eye':.02,'stable_residual_p95_max_eye':.04,'real_aperture_SD_min_eye':.005,'minimum_sources':6,'fixed_denominator':8,'width_SD_not_a_gate':True,'no_substitution':True,'early_stop':'if fewer than6/8 real input QC pass, stop scientific pipeline before any GPU generation; controls/generation then explicitly NOT_TESTED'},
      'primary':{'signal':'aperture Euclidean pixel distance13-14 after similarity / reference eye distance','E':'1-centered MSE / reference centered variance; frozen=0','r':'Pearson aperture over paired valid support; zero-variance nonmeasurable','auxiliary':['width61-291','six fixed mouth-point trajectory','uncentered RMSE','motion SD ratio'],'no_cal_effect_statistics':True},
      'lag':{'range':list(range(-5,6)),'criterion':'mean per-source RAW-versus-real aperture correlation on common interior and paired valid support','tie_order':'abs lag ascending, then numeric ascending; equality tolerance1e-12','sign':'metric compares real[t] with generated[t+lag]','frozen_for_both_conditions':True,'lag0_always_reported':True,'no_eval_optimal_lag':True},
      'controls':{'domains':['REAL','RAW','FIXED'],'views':['REPEAT','SHIFT_-4','SHIFT_-2','SHIFT_2','SHIFT_4','FROZEN','REVERSE','WARP_0.8','WARP_1.2','TX_-8','TX_8','TY_-8','TY_8','ROT_-5','ROT_5','SCALE_0.9','SCALE_1.1'],'pixel_first_redetection':True,'mapping':'100-frame core; shift y[t]=x[clip(t-shift,0,99)]; reverse99-t; freeze49; warp f(t)=t+(speed-1)*99/(2pi)*sin(2pi*t/99), round nearest, endpoints fixed; local slope at ends0.8/1.2; no claim constant global speed','spatial':'cv2 warpAffine INTER_LINEAR BORDER_CONSTANT0; fixed224 canvas, no GT coordinate editing','common_interior':'all timing/spatial diagnostics use t=10..89, pair support intersection of all views valid; lag search adds own paired valid; >=60 paired frames required, no imputation','repeat':{'RMSE_max_eye':.001,'r_min':.99},'shift':{'recover_error_max_frames':1,'all_four_signs':True,'abs4_E_no_better_than_abs2_E':True},'frozen_E_max':.05,'reverse_and_warps_E_below_repeat_min':.10,'spatial':{'RMSE_max_eye':.01,'r_min':.95},'cross_domain_RAW':{'E_min_strict':0,'r_min':.30,'relative_shift_recovery_max_frames':1},'aggregation':'at least6/8 fixed sources satisfying input and all mandatory gates; each domain reported; no retuning after diagnostics'},
      'resources':{'persistent_bytes':40*2**20,'rss_limit':2**30,'disk_reserve':5*2**30,'gpu_lease':'/tmp/tts-exp-gpu.lock','gpu_start':'three5s samples no foreigncompute memory<=128MiB util<=10','temporary':'RAM/pipes; only newly created temporaryvideo removable after independent pixel bridge and durable receipts','no_previews':True},
      'dependencies':{str(p):sha(p) for p in deps}}
    write(OUT/'pose_engineering_contract.json',synthetic_pose_contract())
    for p in deps[:3]:
        d=OUT/'code_snapshot'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    write(OUT/'protocol.json',proto);write(OUT/'seal.json',{'protocol_sha256':sha(OUT/'protocol.json'),'before_forward':True});print('FROZEN',sha(OUT/'protocol.json'))

def protocol():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256']
    for f,h in p['dependencies'].items():assert sha(f)==h,(f,'changed')
    return p

def longest_false(v):
    best=run=0
    for x in v:
        run=0 if x else run+1;best=max(best,run)
    return best

def similarity(points,ref):
    a=np.asarray(points,float);b=np.asarray(ref,float);ac=a-a.mean(0);bc=b-b.mean(0)
    u,s,vt=np.linalg.svd(ac.T@bc);rot=u@vt
    if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt;s[-1]*=-1
    scale=s.sum()/np.sum(ac*ac);trans=b.mean(0)-scale*a.mean(0)@rot
    return scale,rot,trans

def geometry(raw,valid,mats,reference,refmat):
    n=len(raw);xy=np.asarray(raw,dtype=float)[:,:,:2]*224;ref=np.asarray(reference,dtype=float)[:,:2]*224
    eye=float(np.linalg.norm(ref[0]-ref[3]));assert eye>1
    aperture=np.full(n,np.nan);width=np.full(n,np.nan);res=np.full(n,np.nan);poses=np.full((n,3),np.nan);mouth=np.full((n,6,2),np.nan)
    refang=angles(refmat);okay=np.asarray(valid,bool).copy()
    for i in range(n):
        if not okay[i]:continue
        try:
            scale,rot,trans=similarity(xy[i,:7],ref[:7]);aligned=scale*xy[i]@rot+trans
            poses[i]=angles(mats[i]);res[i]=np.sqrt(np.mean(np.sum((aligned[:7]-ref[:7])**2,1)))/eye
            mouth[i]=aligned[7:]/eye;aperture[i]=np.linalg.norm(aligned[7]-aligned[8])/eye;width[i]=np.linalg.norm(aligned[9]-aligned[10])/eye
        except (ValueError,np.linalg.LinAlgError):okay[i]=False
    return {'aperture':aperture,'width':width,'residual':res,'pose':poses,'mouth':mouth,'valid':okay,'reference_angles':refang,'eye_px':eye}

def input_qc(g):
    v=g['valid'];n=int(v.sum());ang=g['reference_angles'];dif=np.abs((g['pose'][v,:2]-ang[:2]+180)%360-180)
    q={'valid_fraction':float(v.mean()),'missing_run':longest_false(v),'reference_angles_deg':ang.tolist(),'pose_change_p95_deg':np.percentile(dif,95,axis=0).tolist() if n else None,'stable_residual_median':float(np.median(g['residual'][v])) if n else None,'stable_residual_p95':float(np.percentile(g['residual'][v],95)) if n else None,'aperture_sd':float(np.std(g['aperture'][v])) if n else None}
    flags={'coverage':q['valid_fraction']>=.95,'missing':q['missing_run']<=2,'reference_pose':bool(np.max(np.abs(ang[:2]))<=20),'relative_pose':bool(n and np.max(dif,axis=0).shape==(2,) and max(q['pose_change_p95_deg'])<=10),'similarity':bool(n and q['stable_residual_median']<=.02 and q['stable_residual_p95']<=.04),'motion':bool(n and q['aperture_sd']>=.005)}
    q.update({'checks':flags,'passed':all(flags.values())});return q

def detector():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    options=mp.tasks.vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(ASSET),delegate=BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=2,output_face_blendshapes=False,output_facial_transformation_matrixes=True)
    return mp,mp.tasks.vision.FaceLandmarker.create_from_options(options)

def detect_frames(frames):
    import cv2
    mp,det=detector();raw=[];valid=[];mat=[];hashes=[];faces=[];fullhash=[]
    try:
        for f in frames:
            result=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)))
            faces.append(len(result.face_landmarks));okay=len(result.face_landmarks)==1 and len(result.facial_transformation_matrixes)==1
            if okay:
                allp=np.asarray([[p.x,p.y,p.z] for p in result.face_landmarks[0]],np.float32)
                raw.append(allp[POINTS]);mat.append(result.facial_transformation_matrixes[0]);fullhash.append(ah(allp))
            else:raw.append(np.full((len(POINTS),3),np.nan,np.float32));mat.append(np.full((4,4),np.nan));fullhash.append('invalid')
            valid.append(okay);hashes.append(hashlib.sha256(f.tobytes()).hexdigest())
    finally:det.close()
    return np.asarray(raw,np.float32),np.asarray(valid,bool),np.asarray(mat,np.float64),{'frame_bgr_sha256':hashes,'full478_raw_xyz_sha256':fullhash,'face_counts':faces}

def source_frames(row):
    import cv2
    cap=cv2.VideoCapture(str(ROOT/row['video_local_path']));frames=[];ref=None;i=0;start=row['core_start_frame']
    try:
        while i<start+100:
            ok,f=cap.read();assert ok,(row['sample_id'],i)
            if i==0:ref=f.copy()
            if i>=start:frames.append(f.copy())
            i+=1
    finally:cap.release()
    assert len(frames)==100 and ref.shape==(224,224,3)
    return ref,frames

def input_stage():
    import cv2, importlib.metadata
    p=protocol();cv2.setNumThreads(1);receipts=[]
    write(OUT/'environment_input.json',{'python':sys.executable,'mediapipe':importlib.metadata.version('mediapipe'),'numpy':np.__version__,'cv2':cv2.__version__,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'cpu_only':True})
    for row in p['calibration']:
        limits();sid=row['sample_id'];dest=OUT/'input'/sid;assert not dest.with_suffix('.json').exists()
        assert sha(ROOT/row['video_local_path'])==row['video_sha256']
        ref,frames=source_frames(row);raw,valid,mats,metadata=detect_frames([ref]+frames)
        # Exact absolute source PTS audit, without any inference for held-out16.
        pts=read_probe_pts(ROOT/row['video_local_path']);local=pts[row['core_start_frame']:row['core_start_frame']+100]
        assert len(local)==100 and np.max(np.abs(np.diff(local)-.04))<1e-6
        metadata.update({'source_sha256':row['video_sha256'],'core_start_frame':row['core_start_frame'],'reference_frame':0,'retained_raw_points':POINTS,'raw_coordinate_dtype':'float32','matrix_dtype':'float64','protocol_sha256':sha(OUT/'protocol.json')})
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),raw=raw,valid=valid,matrices=mats,pts=np.r_[pts[0],local])
        if valid[0]:
            try:g=geometry(raw[1:],valid[1:],mats[1:],raw[0],mats[0]);qc=input_qc(g)
            except (ValueError,AssertionError) as e:qc={'passed':False,'error':str(e)}
        else:qc={'passed':False,'error':'fixed reference frame0 not unique/valid; no replacement'}
        metadata.update({'id':sid,'qc':qc,'feature_sha256':sha(dest.with_suffix('.npz')),'resources':limits()});write(dest.with_suffix('.json'),metadata);receipts.append({'id':sid,'qc':qc})
        print('INPUT',sid,qc,flush=True)
    good=sum(r['qc']['passed'] for r in receipts);write(OUT/'input_gate.json',{'status':'PASS' if good>=6 else 'FAIL','passed_sources':good,'fixed_denominator':8,'rows':receipts,'action':'eligible_for_calibration_generation' if good>=6 else 'STOP_BEFORE_GPU_GENERATION_AND_CONTROLS;16eval remains locked','no_effect_scores':True})

def read_probe_pts(path):
    z=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))
    return np.array([float(f['best_effort_timestamp_time']) for f in z['frames']])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','input']);args=ap.parse_args();{'freeze':freeze,'input':input_stage}[args.stage]()
