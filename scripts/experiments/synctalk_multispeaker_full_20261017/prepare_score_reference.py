"""Freeze source-owned test ROI and all three evaluation-audio features before training."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import cv2,numpy as np,torch,soundfile as sf,python_speech_features
from scipy import signal

SYNC=Path('/home/wjj/tts-audio/tts-exp/third_party/syncnet_python')
sys.path.insert(0,str(SYNC))
from SyncNetModel import S
from SyncNetInstance import calc_pdist
from detectors import S3FD

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def crop_images(images,track):
    result=[]
    for image,(s,x,y) in zip(images,track):
        cs=.4;bsi=int(s*(1+2*cs));padded=np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)),constant_values=110)
        mx=x+bsi;my=y+bsi
        crop=padded[int(my-s):int(my+s*(1+2*cs)),int(mx-s*(1+cs)):int(mx+s*(1+cs))]
        result.append(cv2.resize(crop,(224,224)))
    return np.stack(result)

def summarize(matrix,rows):
    curve=matrix[rows].mean(0);value,index=torch.min(curve,0)
    return dict(sync_c=float(torch.median(curve)-value),sync_d=float(value),distance_d0=float(curve[15]),
        background_median=float(torch.median(curve)),av_offset=int(15-index),windows=len(rows),curve=curve.numpy().tolist())

def audio_forward(model,mfcc,n=245):
    audio=torch.from_numpy(mfcc[None,None].astype(np.float32));rows=[]
    with torch.no_grad():
        for i in range(0,n,20):
            batch=torch.cat([audio[:,:,:,j*4:j*4+20] for j in range(i,min(i+20,n))])
            rows.append(model.forward_aud(batch.cuda()).cpu())
    return torch.cat(rows)

def visual_forward(model,crops,n=245):
    visual=torch.from_numpy(crops).permute(3,0,1,2).unsqueeze(0).float();rows=[]
    with torch.no_grad():
        for i in range(0,n,20):
            batch=torch.cat([visual[:,:,j:j+5] for j in range(i,min(i+20,n))])
            rows.append(model.forward_lip(batch.cuda()).cpu())
    return torch.cat(rows)

def run(root,identity):
    folder=root/'score_reference'/identity;folder.mkdir(parents=True,exist_ok=False)
    raw=Path(__file__).read_bytes();(folder/Path(__file__).name).write_bytes(raw)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.chdir(SYNC)
    material=root/'material'/identity
    manifest=json.loads((material/'split_manifest.json').read_text())
    info=next(x for x in manifest['splits'] if x['name']=='test')
    assert info['source_start_frame']==50 and info['frames']==1500 and len(manifest['eval_clips'])==6
    frame_hashes=(material/'test_rgb_sha256.bin').read_bytes()
    decoder=root/'decode_canonical_rgb.py'
    process=subprocess.Popen(['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',str(decoder),'--root',str(root),'--identity',identity],stdout=subprocess.PIPE)
    images=[]
    for i in range(50,1550):
        rgb=process.stdout.read(512*512*3);assert len(rgb)==512*512*3
        assert hashlib.sha256(rgb).digest()==frame_hashes[i*32:(i+1)*32]
        images.append(cv2.cvtColor(np.frombuffer(rgb,dtype=np.uint8).reshape(512,512,3),cv2.COLOR_RGB2BGR))
    assert process.stdout.read()==b'' and process.wait()==0
    images=np.stack(images);assert images.shape==(1500,512,512,3)
    detector=S3FD(device='cuda');boxes=[];fallback=[];secondary=[]
    for i,image in enumerate(images):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        found=detector.detect_faces(rgb,conf_th=.9,scales=[.25])
        if len(found)==0:
            found=detector.detect_faces(rgb,conf_th=.9,scales=[1.]);fallback.append(i)
        assert len(found)>0,f'{identity} missing GT face at {i}; do not drop frame'
        box=max(found,key=lambda b:(b[2]-b[0])*(b[3]-b[1]));boxes.append(box[:4].tolist())
        if len(found)>1:secondary.append(dict(frame=i,boxes=np.asarray(found).tolist()))
    del detector;torch.cuda.empty_cache()
    boxes=np.array(boxes);size=np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2
    x=(boxes[:,0]+boxes[:,2])/2;y=(boxes[:,1]+boxes[:,3])/2
    track=np.stack([signal.medfilt(v,13) for v in [size,x,y]],1)
    np.save(folder/'shared_gt_track.npy',track);np.save(folder/'gt_boxes.npy',boxes)
    model=S(num_layers_in_fc_layers=1024).cuda().eval();weights=SYNC/'data/syncnet_v2.model'
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    original,sr=sf.read(material/'test_original.wav',dtype='int16');assert sr==16000
    result=dict(identity=identity,source_sha256=hashlib.sha256(raw).hexdigest(),
        split_manifest_sha256=sha(material/'split_manifest.json'),source_test_sha256=sha(material/'test.mkv'),
        precision='FP32 TF32 off deterministic cuDNN',syncnet_weights_sha256=sha(weights),
        model_source_sha256=sha(SYNC/'SyncNetModel.py'),distance_source_sha256=sha(SYNC/'SyncNetInstance.py'),
        crop_owner='all original canonical test frames; shared across every training/driver/scoring gain',
        fallback_detection_frames=fallback,secondary_detections=secondary,missing_detection_frames=[],
        frame_rgb_sha256_verified=True,decoder_source_sha256=sha(decoder),decoder_runtime='material audit OpenCV4.10.0',clips=[],training_admitted=False)
    for clip in manifest['eval_clips']:
        start=clip['source_start_frame'];length=clip['frames'];assert length==250
        out=folder/clip['id'];out.mkdir()
        pcm=original[start*640:(start+length)*640];assert len(pcm)==160000
        sf.write(out/'score_original_pcm16.wav',pcm,16000,subtype='PCM_16')
        assert np.array_equal(sf.read(out/'score_original_pcm16.wav',dtype='int16')[0],pcm)
        cropped=crop_images(images[start-50:start-50+length],track[start-50:start-50+length])
        vf=visual_forward(model,cropped);np.save(out/'gt_visual_features.npy',vf.numpy())
        rows=[]
        for gain in [-6,0,6]:
            values=pcm if gain==0 else pcm.astype(np.float64)*10**(gain/20)
            if gain!=0:
                path=out/f'score_{gain:+d}dB_float64.wav';sf.write(path,values/32768,16000,subtype='DOUBLE')
                decoded=sf.read(path,dtype='float64')[0]*32768;assert np.array_equal(values,decoded)
            mfcc=python_speech_features.mfcc(values,16000).T
            np.save(out/f'mfcc_{gain:+d}dB.npy',mfcc)
            af=audio_forward(model,mfcc);np.save(out/f'audio_features_{gain:+d}dB.npy',af.numpy())
            if clip['id']=='test_00' and gain==0:
                assert torch.equal(audio_forward(model,mfcc),af)
                assert torch.equal(visual_forward(model,cropped),vf)
            matrix=torch.stack(calc_pdist(vf,af,vshift=15));np.save(out/f'gt_distances_{gain:+d}dB.npy',matrix.numpy())
            rows.append(dict(evaluation_gain_db=gain,guard20=summarize(matrix,np.arange(20,225)),full=summarize(matrix,np.arange(245))))
        result['clips'].append(dict(id=clip['id'],source_start_frame=start,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
            source_crop_sha256=hashlib.sha256(cropped.tobytes()).hexdigest(),baseline=rows))
        (folder/'score_reference_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(identity=identity,clip=clip['id'],baseline_0dB=rows[1]['guard20'])),flush=True)
    result.update(passed=True,repeat_audio_and_visual_exact=True,score_based_exclusions=False)
    (folder/'score_reference_receipt.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
