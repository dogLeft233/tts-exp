"""Low-memory held-out generation; all experimental audio scoring uses original PCM."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,shutil,time,traceback,wave,subprocess
from pathlib import Path
import numpy as np
from core import make_conditions,metrics,check_disjoint

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
BASE=Path('C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018')
OUT=Path(os.environ.get('GAIN_RUN_DIR',str(BASE/'run')))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    path=OUT/name; temp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    for attempt in range(100):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt==99: raise
            time.sleep(.1)

def gpu_snapshot():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    fields=[x.strip() for x in p.stdout.strip().splitlines()[0].split(',')]
    return {'name':fields[0],'used_mib':int(fields[1]),'free_mib':int(fields[2]),'util_percent':int(fields[3]),'time':time.time()}

def main():
    import torch,cv2,imageio_ffmpeg
    from scripts.experiments.portable_source_axis_20260930 import wav2lip as w2l,scoring
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    check_disjoint([x['speaker'] for x in p['fit_speakers']],[x['speaker'] for x in p['test_speakers']])
    if (OUT/'run_receipt.json').exists(): raise FileExistsError('fresh run receipt required')
    start=gpu_snapshot()
    if start['free_mib']<3500 or start['util_percent']>85: raise RuntimeError('GPU busy before launch')
    if shutil.disk_usage(OUT).free<1024**3: raise RuntimeError('insufficient disk reserve')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.manual_seed(20261018); np.random.seed(20261018)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.40)
    device='cuda'; batch=4
    receipt={'status':'RUNNING','protocol_sha256':sha(OUT/'protocol.json'),
             'runtime':{'torch':torch.__version__,'numpy':np.__version__,'cv2':cv2.__version__,'python':sys.version,
                        'device':torch.cuda.get_device_name(),'batch':batch,'allocator_fraction':.40,'tf32':False},
             'resources':[start],'clips':[],'code_sha256':{x.name:sha(x) for x in Path(__file__).parent.glob('*.py')}}
    save('run_receipt.json',receipt)
    asset=BASE/'assets'; wr=asset/'Wav2Lip'; sr=asset/'syncnet'
    assert sha(wr/'wav2lip_gan.pth')==p['weights']['wav2lip']
    assert sha(sr/'syncnet_v2.model')==p['weights']['syncnet']
    assert sha(OUT/'input/common.npy')==p['common_sha256']
    assert sha(OUT/'input/face.png')==p['face']['sha256']
    g=np.load(OUT/'input/common.npy',allow_pickle=False)
    model,modelmeta=w2l.load_model(wr,wr/'wav2lip_gan.pth',device)
    receipt['model']=modelmeta
    bridge=[]
    for b in p['bridge']:
        file=OUT/'input'/b['file']; assert sha(file)==b['sha256']
        with np.load(file,allow_pickle=False) as a:
            for c in ['RAW','DOSE_P3']:
                fresh=w2l.encode_latents(model,a[c+'_chunks'],device=device,batch_size=batch)
                expected=a[c+'_z']; dif=fresh.astype(float)-expected.astype(float)
                rel=float(np.linalg.norm(dif)/np.linalg.norm(expected.astype(float))); mx=float(abs(dif).max())
                bridge.append({'input':b['file'],'condition':c,'relative_l2':rel,'max_abs':mx})
                np.save(OUT/'input'/(b['file']+'_'+c+'_fresh.npy'),fresh)
                if rel>p['bridge_tolerance']['relative_l2'] or mx>p['bridge_tolerance']['max_abs']:
                    raise RuntimeError('fresh encoder is not compatible with frozen axis')
    receipt['bridge']=bridge; save('run_receipt.json',receipt)
    net,netmeta=scoring.load_syncnet(sr,sr/'syncnet_v2.model',device)
    receipt['syncnet']=netmeta
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    receipt['ffmpeg']=ffmpeg
    frame=cv2.imread(str(OUT/'input/face.png')); assert frame.shape==(512,512,3)
    box=p['face']['generation_box_xyxy']; scorebox=p['face']['score_box']['box']
    face=w2l.cache_face_features(model,frame,box,range(1,batch+1),device)
    for row in p['test_speakers']:
        snap=gpu_snapshot(); receipt['resources'].append(snap)
        if snap['free_mib']<2000: raise RuntimeError('GPU reserve below 2GiB; stop before next clip')
        if shutil.disk_usage(OUT).free<1024**3: raise RuntimeError('disk reserve below 1GiB')
        sid=row['speaker']; folder=OUT/sid; folder.mkdir(exist_ok=False)
        wav=OUT/'input'/row['file']; assert sha(wav)==row['sha256']
        with wave.open(str(wav),'rb') as f:
            pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
        x=pcm.astype(np.float32)/32768; y=(x.astype(float)*10**(p['dose_db']/20)).astype(np.float32)
        assert len(x)==len(y) and np.max(abs(y))<1
        mel0,ch0,starts0,_=w2l.mel_chunks(x,wr)
        mel1,ch1,starts1,_=w2l.mel_chunks(y,wr)
        assert starts0==starts1
        z0=w2l.encode_latents(model,ch0,device=device,batch_size=batch)
        z1=w2l.encode_latents(model,ch1,device=device,batch_size=batch)
        condition_z,numeric=make_conditions(z0,z1,g)
        np.savez_compressed(folder/'latents.npz',**condition_z,common=g,mel_starts=np.array(starts0))
        parity=w2l.parity_check(model,ch0,frame,box,face[4],device=device)
        if parity['max_abs_float_prediction']>1e-6 or parity['uint8_differing_values']>20:
            raise RuntimeError('direct decoder differs from native forward')
        audio_count=len(pcm)//640-5
        A,ameta=scoring.audio_embeddings_from_syncnet_units(net,pcm.astype(np.float64),window_count=audio_count,device=device,batch_size=batch)
        np.save(folder/'audio.npy',A)
        rec={'speaker':sid,'source_id':row['id'],'numeric':numeric,'parity':parity,'audio':ameta,'conditions':{}}
        receipt['clips'].append(rec)
        anchor=None; reference_pixels=None; reference_v=None
        for c in p['conditions']:
            body=folder/(c+'.mp4')
            vmeta=w2l.render_video(model,condition_z[c],frame,box,face,body,ffmpeg,device=device,batch_size=batch)
            jpg_dir=folder/(c+'_jpeg')
            assert not jpg_dir.exists()
            images,jmeta=scoring.extract_official_jpegs(body,jpg_dir,ffmpeg)
            x1,y1,x2,y2=scorebox
            cropped=[cv2.resize(f[y1:y2,x1:x2],(224,224),interpolation=cv2.INTER_LINEAR) for f in images]
            V=scoring.lip_embeddings_from_frames(net,cropped,device=device,batch_size=batch)
            D=scoring.pairwise_distance_matrix(V,A,device='cpu')
            n=min(len(V),len(A)); guard=np.arange(20,n-20,dtype=int)
            if len(guard)<10: raise RuntimeError('insufficient guard20 support')
            if anchor is None: anchor=int(np.argmin(D[guard].mean(0,dtype=np.float32)))
            np.savez_compressed(folder/(c+'_score.npz'),visual=V,distance=D,guard20=guard)
            vals={'guard20':metrics(D[guard],anchor),'full':metrics(D[:n],anchor),
                  'video':vmeta,'jpeg':jmeta,'n_visual':len(V),'n_audio':len(A),
                  'score_npz_sha256':sha(folder/(c+'_score.npz'))}
            if c=='BASE': reference_pixels=vmeta['raw_bgr_pixel_sha256']; reference_v=V.copy()
            if c=='BASE_REPEAT':
                assert vmeta['raw_bgr_pixel_sha256']==reference_pixels and np.array_equal(V,reference_v)
                vals['repeat_exact']=True
            rec['conditions'][c]=vals
            receipt['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            receipt['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            save('run_receipt.json',receipt)
            # Only this run's freshly generated JPEG intermediates; video/features remain.
            assert jpg_dir.resolve().parent==folder.resolve() and folder.resolve().parent==OUT.resolve()
            shutil.rmtree(jpg_dir)
            print(sid,c,round(vals['guard20']['C'],6),flush=True)
            del images,cropped,V,D
        rec['status']='COMPLETE'
        torch.cuda.empty_cache(); save('run_receipt.json',receipt)
    receipt['status']='COMPLETE'; receipt['resources'].append(gpu_snapshot())
    save('run_receipt.json',receipt)

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        save('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
