"""Differential audit against verbatim upstream Dataset and calc_pdist bodies."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/home/wjj/tts-audio/tts-exp')
DEPLOY=Path('/home/wjj/av-sync-eval-20261028')
PREVIOUS=ROOT/'runs/cloud_mtd_synchformer_20261028_attempt01'
RUN=ROOT/'runs/mtd_official_audit_20261028_attempt01'


def read(p):return json.loads(Path(p).read_text())


def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2,allow_nan=False));tmp.replace(p)


def retrieval_metrics(matrix,expected_offset=0,guard=0):
    import numpy as np
    z=np.asarray(matrix)
    z=z[guard:len(z)-guard] if guard else z
    offset=15-z.argmax(1)
    curve=z.mean(0)
    return dict(k5_accuracy=float(np.mean(np.abs(offset-expected_offset)<=1)),
                mode_offset=int(np.bincount(offset+15,minlength=31).argmax()-15),
                global_offset=int(15-curve.argmax()),aligned_logit=float(z[:,15].mean()),
                aligned_minus_background=float(curve[15]-np.median(curve)),windows=len(z))


def shift_tracking(native,shifted,shift_samples,tolerance_frames=1):
    """Known perturbation tests offset CHANGE, without assuming native offset zero."""
    expected_change=-shift_samples/640
    observed_change=shifted['global_offset']-native['global_offset']
    return dict(expected_change_frames=expected_change,observed_change_frames=observed_change,
                absolute_error_frames=abs(observed_change-expected_change),
                passed=abs(observed_change-expected_change)<=tolerance_frames)


def upstream():
    import torch,torchaudio,torchvision,numpy as np
    from natsort import natsorted
    from glob import glob
    from os.path import basename,dirname,isfile,join
    sys.path.insert(0,str(DEPLOY/'MTDVocaLiST'))
    from hparams import hparams
    s=(DEPLOY/'MTDVocaLiST/test_stu.py').read_text();tree=ast.parse(s)
    body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Dataset','calc_pdist']]
    env=dict(torch=torch,torchaudio=torchaudio,torchvision=torchvision,np=np,math=math,os=os,glob=glob,
             natsorted=natsorted,basename=basename,dirname=dirname,isfile=isfile,join=join,
             v_context=5,mel_step_size=16,hparams=hparams,device=torch.device('cuda'),
             MIN_LEVEL=np.exp(-hparams.min_level_db/-20*np.log(10)),TOP_DB=-hparams.min_level_db,
             melscale=torchaudio.transforms.MelScale(n_mels=hparams.num_mels,sample_rate=hparams.sample_rate,
                f_min=hparams.fmin,f_max=hparams.fmax,n_stft=hparams.n_stft,norm='slaney',mel_scale='slaney'),
             cvresize=torchvision.transforms.Resize([hparams.img_size,hparams.img_size]))
    exec(compile(ast.Module(body=body,type_ignores=[]),str(DEPLOY/'MTDVocaLiST/test_stu.py'),'exec'),env)
    return env,hashlib.sha256(s.encode()).hexdigest()


def exported_fixture(row,boxes,label):
    import cv2
    folder=RUN/'fixtures'/row['id']/label
    if (folder/'audio.wav').exists():return folder
    folder.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(row['video_path'])
    for i,box in enumerate(boxes):
        ok,f=cap.read();assert ok
        x1,y1,x2,y2=map(int,box)
        assert cv2.imwrite(str(folder/f'{i}.jpg'),f[y1:y2,x1:x2])
    cap.release();shutil.copyfile(row['audio_path'],folder/'audio.wav')
    return folder


def dataset_get(env,folder):
    d=env['Dataset'].__new__(env['Dataset']);d.all_videos=[str(folder)];d.split='test'
    return d[0]


def windows(vid,mel,count):
    import torch
    frames=vid.reshape(-1,3,48,96)
    vv=torch.stack([frames[i:i+5].reshape(15,48,96) for i in range(count)])
    aa=torch.stack([mel[:,:,int(80*(i/25)):int(80*(i/25))+16] for i in range(count)])
    return vv,aa


def aligned(model,v,a,batch=16):
    import torch
    out=[]
    with torch.inference_mode():
        for i in range(0,len(v),batch):out.append(model(v[i:i+batch].cuda(),a[i:i+batch].cuda())[0].cpu())
    return torch.cat(out)


def main(stage,full512=False):
    import numpy as np
    import torch,torchaudio
    import soundfile as sf
    torch.set_num_threads(4);torch.manual_seed(20261028)
    RUN.mkdir(parents=True,exist_ok=True)
    env,source_sha=upstream()
    from models.student_thin_200_all import SyncTransformer
    from models.student_thin_200 import SyncTransformer as TestModel
    state=torch.load(DEPLOY/'weights/mtdvocalist/pure_MTDVocaLiST.pth',map_location='cpu',weights_only=True)['state_dict']
    model=SyncTransformer(d_model=200);model.load_state_dict(state,strict=True);model.eval().cuda()
    rows=[r for r in read(PREVIOUS/'inputs.json') if r['arm']=='N']
    if stage=='pilot':rows=rows[:3]
    result_file=RUN/('results_full512.json' if full512 else 'results.json')
    results=read(result_file) if result_file.exists() else {}
    detector=None
    for ri,row in enumerate(rows):
        key=row['id']
        if key in results and (stage!='pilot' or 'class_parity' in results[key]):continue
        cache=np.load(PREVIOUS/'mtd_full_crops'/(key+'.npz'))
        if full512:
            import cv2
            sys.path.append(str(ROOT/'third_party/Wav2Lip'))
            import face_detection
            bp=RUN/'boxes512'/(key+'.npy')
            if bp.exists():boxes=np.load(bp)
            else:
                if detector is None:
                    detector=face_detection.FaceAlignment(face_detection.LandmarksType._2D,flip_input=False,device='cuda')
                    torch.backends.cudnn.benchmark=False
                cap=cv2.VideoCapture(row['video_path']);boxes=[]
                for _ in range(row['native_frames']):
                    ok,f=cap.read();assert ok
                    b=detector.get_detections_for_batch(np.asarray([f]))[0]
                    assert b is not None;boxes.append(b)
                cap.release();boxes=np.asarray(boxes);bp.parent.mkdir(exist_ok=True);np.save(bp,boxes)
            folder=exported_fixture(row,boxes,'official512boxes')
        else:folder=exported_fixture(row,cache['boxes'],'original256boxes')
        vid,mel,last=dataset_get(env,folder);vv,aa=windows(vid,mel,last)
        if ri==0:
            write(RUN/'official_source.json',dict(test_stu_sha256=source_sha,upstream_bodies=['Dataset','calc_pdist'],
                  model_class_readme='student_thin_200_all',model_class_test='student_thin_200',weight_strict=True,
                  hypothesis='raw aligned mean vs official K5 retrieval; class mismatch; face detector resizing; audio or shift boundary'))
        own_wave,sr=sf.read(row['audio_path'],dtype='float32');refwave,_=torchaudio.load(folder/'audio.wav')
        assert np.array_equal(own_wave,refwave[0].numpy())
        image_exact=np.array_equal(vid.reshape(-1,3,48,96).byte().numpy(),cache['frames'])
        if not full512:assert image_exact, 'Official image preprocessing differs'
        own_spec=torch.stft(torch.from_numpy(own_wave),n_fft=800,hop_length=200,win_length=800,window=torch.hann_window(800),return_complex=True)
        own_mel=env['melscale'](own_spec.abs());own_mel=torch.clip(8*((20*torch.log10(torch.clamp(own_mel,min=1e-5))-20+100)/100)-4,-4,4)
        mel_error=float((mel[0]-own_mel).abs().max());assert mel_error<1e-5
        record=dict(image_tensor_exact=image_exact,waveform_exact=True,mel_max_abs_error=mel_error,
                    controls={},source=row['sample_id'])
        z=aligned(model,vv,aa)
        saved=read(PREVIOUS/'mtd_full_scores.json')[key]['outputs']['0']['window_logits']
        error=float(np.max(np.abs(z.numpy()-saved)))
        if not full512:assert error<2e-4,'Native scores no longer reproducible'
        record['native_logit_max_abs_error']=error
        if stage=='pilot':
            other=TestModel(d_model=200);other.load_state_dict(state,strict=True);other.eval().cuda()
            q=aligned(other,vv[:31],aa[:31],31);zz=aligned(model,vv[:31],aa[:31],31)
            parity=float((q-zz).abs().max());assert parity<1e-5
            single=aligned(model,vv[:31],aa[:31],1)
            record['class_parity']=dict(max_abs_logit_error=parity,batch1_vs31_max_error=float((single-zz).abs().max()))
            del other;torch.cuda.empty_cache()
        for samples in [0,-6400,6400]:
            if samples:
                shifted=np.zeros_like(own_wave)
                if samples>0:shifted[samples:]=own_wave[:-samples]
                else:shifted[:samples]=own_wave[-samples:]
                sf.write(folder/'audio.wav',shifted,16000,subtype='PCM_16')
                _,ms,n=dataset_get(env,folder);assert n==last
                _,ab=windows(vid,ms,last)
            else:ab=aa
            with torch.inference_mode():matrix=torch.stack(env['calc_pdist'](model,vv,ab,vshift=15)).numpy()
            np.save(RUN/f'{key}_{samples}_{"512" if full512 else "256"}_matrix.npy',matrix)
            record['controls'][str(samples)]=dict(full=retrieval_metrics(matrix,expected_offset=-samples//640),
                                                 interior=retrieval_metrics(matrix,expected_offset=-samples//640,guard=20),
                                                 # Test scoring at zero for shifted PCM, rather than rewarding recovery.
                                                 zero_target_full=retrieval_metrics(matrix),
                                                 zero_target_interior=retrieval_metrics(matrix,guard=20))
        shutil.copyfile(row['audio_path'],folder/'audio.wav')
        results[key]=record;write(result_file,results)
        print(json.dumps(dict(id=key,parity=record.get('class_parity'),controls={s:r['zero_target_interior'] for s,r in record['controls'].items()})),flush=True)
    print('AUDIT_STAGE_COMPLETE',stage,len(results),flush=True)


def finalize():
    import numpy as np
    results=read(RUN/'results.json');full512=read(RUN/'results_full512.json')
    assert len(results)==20 and len(full512)==3
    details=[]
    for key,row in results.items():
        def matrix(sh):
            path=RUN/f'{key}_{sh}_256_matrix.npy'
            return np.load(path if path.exists() else RUN/f'{key}_{sh}_matrix.npy')
        raw=matrix(0);n=raw[20:-20];base=int(15-n.mean(0).argmax())
        fixed=float(n[:,18].mean());checks=[]
        for samples in [-6400,6400]:
            m=matrix(samples)[20:-20];off=int(15-m.mean(0).argmax())
            expected=-samples/640;error=abs((off-base)-expected)
            assert base==row['controls']['0']['interior']['global_offset']
            checks.append(dict(samples=samples,observed_offset=off,expected_change=expected,observed_change=off-base,
                               error_frames=error,tracking_pass=error<=1,
                               zero_lag_raw_lower=bool(m[:,15].mean()<n[:,15].mean()),
                               fixed_pilot_lag3_raw_lower=bool(m[:,18].mean()<fixed)))
        details.append(dict(id=key,native_offset=base,holdout='class_parity' not in row,controls=checks,
                            mel_max_abs_error=row['mel_max_abs_error'],reproduced_logit_max_error=row['native_logit_max_abs_error']))
    all_controls=[c for row in details for c in row['controls']]
    hold=[c for row in details if row['holdout'] for c in row['controls']]
    crop=[]
    for label,data in [('old256', {k:results[k] for k in full512}),('official512',full512)]:
        cc=[]
        for row in data.values():
            n=row['controls']['0']['interior']
            for sh in [-6400,6400]:
                s=row['controls'][str(sh)]['interior']
                cc.append(dict(raw_lower=s['aligned_logit']<n['aligned_logit'],tracking=abs(s['global_offset']-n['global_offset']+sh/640)<=1))
        crop.append(dict(variant=label,controls=len(cc),raw_lower=sum(x['raw_lower'] for x in cc),tracking_pass=sum(x['tracking'] for x in cc)))
    summary=dict(status='AUDIT_COMPLETE',sources=20,real_pcm_shifts=40,
                 frontend=dict(image_tensor_exact_sources=sum(r['image_tensor_exact'] for r in results.values()),
                    waveform_exact_sources=sum(r['waveform_exact'] for r in results.values()),
                    mel_max_abs_error=max(r['mel_max_abs_error'] for r in results.values()),
                    native_logit_max_abs_error=max(r['native_logit_max_abs_error'] for r in results.values()),
                    model_class_max_abs_error=max(r.get('class_parity',{}).get('max_abs_logit_error',0) for r in results.values())),
                 native_offsets=sorted(set(x['native_offset'] for x in details)),
                 relative_offset_recovered=sum(x['tracking_pass'] for x in all_controls),
                 relative_offset_max_error_frames=max(x['error_frames'] for x in all_controls),
                 guarded_zero_lag_raw_lower=sum(x['zero_lag_raw_lower'] for x in all_controls),
                 pilot_fixed_lag3=dict(all_lower=sum(x['fixed_pilot_lag3_raw_lower'] for x in all_controls),total=40,
                                      holdout_lower=sum(x['fixed_pilot_lag3_raw_lower'] for x in hold),holdout_total=len(hold),
                                      diagnostic_only=True,no_preference_scores_replaced=True),
                 official_native_k5_mean=float(np.mean([r['controls']['0']['full']['k5_accuracy'] for r in results.values()])),
                 crop_differential=crop,details=details,
                 conclusion='Network/frontend numerical parity passes; previous raw-mean score is not the official K5 retrieval endpoint. All known shifts are recovered; native peak offset is nonzero. Do not infer no synchronization sensitivity or native TTS preference from raw logit means.')
    write(RUN/'audit_summary.json',summary)
    (RUN/'source_snapshot').mkdir(exist_ok=True)
    shutil.copyfile(__file__,RUN/'source_snapshot'/Path(__file__).name)
    print(json.dumps({k:v for k,v in summary.items() if k!='details'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','naturals','finalize']);p.add_argument('--full512',action='store_true');a=p.parse_args()
    if a.stage=='finalize':finalize()
    else:main(a.stage,a.full512)
