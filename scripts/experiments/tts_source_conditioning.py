"""Frozen source-conditioning experiment. Feature production and scoring are separate."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.tts_native_gain_attribution.common import file_sha256 as sha, read_json as read, write_json as write
from scripts.experiments.tts_native_boundary_audit import stats

OUT = ROOT / "runs/tts_source_conditioning_20260926"
PARENT = ROOT / "runs/tts_pcm_residual_20260926"
MID = ROOT / "runs/tts_native_midpoint_20260926"
W2L = ROOT / "third_party/Wav2Lip"
CONDITIONS = ("D0", "D66", "S0", "S66")
ARMS = ("N", "T")
LAGS = np.arange(-15, 16)


def source_indices(n, condition):
    if condition == "D0": return np.arange(n) % 132
    if condition == "D66": return (np.arange(n) + 66) % 132
    if condition == "S0": return np.zeros(n, dtype=np.int64)
    if condition == "S66": return np.full(n, 66, dtype=np.int64)
    raise ValueError(condition)


def score_roi(boxes):
    b = np.asarray(boxes, dtype=int)
    lo, hi = b[:, :2].min(0), b[:, 2:].max(0)
    side = int(math.ceil(1.5 * max(hi - lo)))
    origin = np.floor((lo + hi) / 2 - side / 2).astype(int)
    return [int(origin[0]), int(origin[1]), int(origin[0] + side), int(origin[1] + side)]


def proc_identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': int(pid), 'exe': str((p / 'exe').resolve()), 'starttime_ticks': fields[19]}


def compute_processes():
    x = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in x.splitlines():
        pid, name, memory = [t.strip() for t in line.split(',', 2)]
        rows.append({'pid': int(pid), 'name': name, 'memory_mib': int(memory)})
    return rows


def resource_check(p, startup=False):
    allowed = p['resources']['allowed_system_process']
    rows = compute_processes()
    for r in rows:
        if r['pid'] == os.getpid(): continue
        assert proc_identity(r['pid']) == allowed, ('foreign process', r)
        assert r['memory_mib'] <= p['resources']['daemon_memory_limit_mib'], ('daemon memory growth', r)
    current = proc_identity(allowed['pid'])
    assert current == allowed, ('system process identity changed', current)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    used, util = map(int, gpu.split(','))
    if startup: assert util <= 10 and used <= p['resources']['daemon_memory_limit_mib'] + 128, (used, util)
    return {'time': time.time(), 'compute': rows, 'memory_mib': used, 'utilization': util}


@contextmanager
def lease(stage):
    p = protocol()
    with open('/tmp/tts-exp-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshots = []
        for i in range(3):
            snapshots.append(resource_check(p, startup=True))
            if i != 2: time.sleep(5)
        write(OUT / f'{stage}_resource_gate.json', {'snapshots': snapshots, 'lease': '/tmp/tts-exp-gpu.lock', 'pid': os.getpid()})
        try: yield p
        finally: fcntl.flock(lock, fcntl.LOCK_UN)


def freeze():
    assert not (OUT / 'protocol.json').exists()
    original = read(PARENT / 'protocol.json')
    group = read(MID / 'protocol.json')['groups']['dynamic_cloud']
    eval_ids = set(group['eval_ids'])
    rows = []
    for r in original['rows']:
        row = {'id': r['id'], 'speaker': r['speaker'], 'split': 'evaluation' if r['id'] in eval_ids else 'calibration', 'cells': {a: r['cells'][a] for a in ARMS}}
        rows.append(row)
    assert len(rows) == 74 and sum(r['split'] == 'calibration' for r in rows) == 26
    source = read(rows[0]['cells']['N']['score_receipt'])['face']
    for r in rows:
        for c in r['cells'].values():
            s = read(c['score_receipt'])
            assert s['face'] == source and c['face_sha256'] == sha(source) and c['track_start'] == 0
    candidates = [r for r in compute_processes() if r['name'] == '/usr/libexec/gnome-remote-desktop-daemon']
    assert len(candidates) == 1 and len(compute_processes()) == 1
    system = candidates[0]
    files = [Path(__file__), ROOT/'scripts/experiments/check_tts_source_conditioning.py', ROOT/'tests/experiments/test_tts_source_conditioning.py',
             W2L/'models/wav2lip.py', W2L/'audio.py', W2L/'inference.py', W2L/'checkpoints/wav2lip_gan.pth',
             ROOT/'third_party/syncnet_python/data/syncnet_v2.model', ROOT/'scripts/experiments/tts_native_gain_attribution/syncnet.py',
             ROOT/'scripts/experiments/static_image_bridge/score_worker.py', ROOT/'scripts/experiments/masked_tts_tfg_probe/direct_mel.py']
    detector = W2L/'face_detection/detection/sfd/s3fd.pth'
    if detector.exists(): files.append(detector)
    p = {'status': 'frozen_before_new_scores', 'created_epoch': time.time(), 'rows': rows,
         'source': {'path': source, 'sha256': sha(source), 'frames': 132, 'fps': 25, 'discard_source_audio': True},
         'conditions': list(CONDITIONS), 'source_mapping': {'D0': 'i%132', 'D66': '(i+66)%132', 'S0': '0', 'S66': '66'},
         'generation': {'detector': 'official Wav2Lip FaceAlignment S3FD all132 frames once', 'batch': 4, 'detector_batch': 4, 'pads': [0,10,0,0], 'nosmooth': True, 'fps': 25, 'mel_step': 16, 'seed': 20260926, 'frames': 'official chunk_mels exact own audio; no looping audio', 'pixel': 'original BGR, resize face96 linear, mask lower48 rows, concatenate masked/full, prediction*255 uint8 then resize into original source box'},
         'evaluation': {'roi': 'union of all132 generation boxes; center union, side=ceil(1.5*max(width,height)); left/top=floor(center-side/2); square, zero pad then linear224 resize; same all outputs/source-only/old-fixed', 'video': 'cropped224 BGR FFV1 AVI 25fps, no audio; exact source PCM extracted separately', 'frontend': 'SyncNetEngine archived ffmpeg JPEG/OpenCV path', 'L': 'min(decoded_frames,PCM_samples//640)-5; truncate both embeddings before lag padding', 'lags': list(range(-15,16)), 'policies': ['guard20','valid','guard0'], 'geometry': ['raw','unit'], 'distance': 'torch float32 pairwise_distance eps1e-6; float64 lag mean; C=median31-min31; bestlag=argmin_index-15, offset=-bestlag'},
         'primary': '48 evaluation/13 speakers, raw guard20 native T-N in D0,D66,S0,S66; (gapD0+gapD66-gapS0-gapS66)/2, gapD66-gapD0, gapS66-gapS0; B,D,C_anchor,D_anchor and source-only same contrasts; all74 exploratory supplemental',
         'calibration': 'raw guard20 generated all26cal x4conditions xN/T bestlag pooled median truncated toward0; seal k before eval scores; max condition-arm median-min>1 flags anchor incomparability, no changes to standard C/support',
         'support': 'all74 kept; each arm common L across4conditions and source-only; pair must have each L>40; no duration matching N/T; no score/ASR exclusions; bridge uses common per-arm Lmin(newD0,oldfixed,oldcrop)',
         'controls': {'repeat': 'first lexical cal ID both arms all4conditions exact rerender; compare decoded pixels/features', 'delay': '+5 frames PCM shift=3200 samples with zero head, same length; all4conditions both arms firstcal; compare lag curves on common rows25:L-25, overlap lags -10..15 maps to base -15..10; bestlag expected+5 if interior', 'bridge': 'new D0 and old original full video with same fixed ROI; old fixedROI vs archived oldcrop, source PCM, common L; report C/B/D on same policies, do not tune based on bridge'},
         'statistics': {'speaker_equal': True, 'bootstrap': 20000, 'seed': 20260926, 'ci': .99},
         'resources': {'allowed_system_process': proc_identity(system['pid']), 'baseline_system_memory_mib': system['memory_mib'], 'daemon_memory_limit_mib': system['memory_mib']+256, 'startup': '3 checks 5s apart, util<=10%, used<=daemon_limit+128; run-local exception only, all other foreign compute PID blocked', 'shared_lease': '/tmp/tts-exp-gpu.lock'},
         'limits': 'historical inputs/new exploratory interventions; source condition changes mouth/pose/reference/crop, not pure mouth factor; source-only not true-motion labels; old n100 generation source/CLI not fully bound, historical bridge required',
         'parent_hashes': {str(f):sha(f) for f in (PARENT/'protocol.json',MID/'protocol.json')}, 'code_and_models': {str(f):sha(f) for f in files}}
    write(OUT/'protocol.json',p)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    snap = OUT/'code';snap.mkdir()
    for f in files[:3]: (snap/f.name).write_bytes(f.read_bytes())
    print('frozen74 cal26 eval48',flush=True)


def protocol():
    p = read(OUT/'protocol.json')
    assert sha(OUT/'protocol.json') == (OUT/'protocol.sha256').read_text().strip()
    for f,h in p['code_and_models'].items(): assert sha(f) == h, f
    return p


def write_video(frames, path):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),25,(224,224))
    assert writer.isOpened()
    digest = hashlib.sha256()
    for frame in frames:
        writer.write(frame); digest.update(frame.tobytes())
    writer.release()
    return digest.hexdigest()


def setup_torch():
    import cv2,torch
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def prepare():
    import cv2,torch
    from scripts.experiments.masked_tts_tfg_probe.direct_mel import load_frames,detect_boxes,chunk_mels
    setup_torch()
    with lease('prepare') as p:
        frames,fps = load_frames(Path(p['source']['path']));assert len(frames)==132 and fps==25
        boxes = detect_boxes(frames,4,[0,10,0,0],True)
        assert len(boxes)==132
        roi = score_roi(boxes)
        assert all(roi[0]<=b[0]<b[2]<=roi[2] and roi[1]<=b[1]<b[3]<=roi[3] for b in boxes)
        np.savez_compressed(OUT/'source_frames.npz',frames=np.stack(frames))
        write(OUT/'geometry.json',{'generation_boxes':boxes,'score_roi':roi,'source_frame_count':132,'source_shape':list(frames[0].shape),'all_boxes_contained':True,'source_frames_sha256':sha(OUT/'source_frames.npz'),'source_pixel_hashes':[hashlib.sha256(f.tobytes()).hexdigest() for f in frames]})
        sys.path.insert(0,str(W2L));import audio
        inventory=[]
        for r in p['rows']:
            row={'id':r['id'],'speaker':r['speaker'],'split':r['split'],'arms':{}}
            for arm in ARMS:
                c=r['cells'][arm];assert sha(c['audio'])==c['audio_sha256']
                pcm=audio.load_wav(c['audio'],16000);mel=audio.melspectrogram(pcm).astype(np.float32);chunks=np.asarray(chunk_mels(mel,25),dtype=np.float32)
                path=OUT/'mel'/r['id']/(arm+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,mel=mel,chunks=chunks)
                row['arms'][arm]={'audio':c['audio'],'audio_sha256':c['audio_sha256'],'samples':len(pcm),'frames':len(chunks),'L':min(len(chunks),len(pcm)//640)-5,'mel_path':str(path),'mel_sha256':sha(path),'source_indices':{cond:source_indices(len(chunks),cond).tolist() for cond in CONDITIONS}}
            row['eligible']=all(a['L']>40 for a in row['arms'].values());inventory.append(row)
        write(OUT/'support.json',inventory)
        write(OUT/'pre_feature_lock.json',{'protocol_sha256':sha(OUT/'protocol.json'),'geometry_sha256':sha(OUT/'geometry.json'),'support_sha256':sha(OUT/'support.json'),'status':'before_any_SyncNet_distance_or_new_score','time':time.time()})
    print('prepared',roi,'eligible',sum(r['eligible'] for r in inventory),flush=True)


def render_cell(model,chunks,source,boxes,roi,condition,path):
    import cv2,torch
    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
    indices=source_indices(len(chunks),condition);frames=[]
    for start in range(0,len(chunks),4):
        inds=indices[start:start+4];faces=[]
        for ix in inds:
            x1,y1,x2,y2=boxes[ix];face=cv2.resize(source[ix][y1:y2,x1:x2],(96,96));mask=face.copy();mask[48:]=0;faces.append(np.concatenate([mask,face],2))
        it=torch.from_numpy(np.asarray(faces,dtype=np.float32).transpose(0,3,1,2)/255).cuda();mt=torch.from_numpy(chunks[start:start+4,None]).cuda()
        with torch.inference_mode(): pred=model(mt,it).cpu().numpy().transpose(0,2,3,1)*255
        for ix,fo in zip(inds,pred):
            full=source[ix].copy();x1,y1,x2,y2=boxes[ix];full[y1:y2,x1:x2]=cv2.resize(fo.astype(np.uint8),(x2-x1,y2-y1));frames.append(crop_zero_padded(full,roi))
    digest=write_video(frames,path)
    return {'path':str(path),'sha256':sha(path),'pixel_sha256':digest,'frames':len(frames),'source_indices':indices.tolist(),'generation_boxes':[boxes[i] for i in indices]}


def produce():
    import cv2,torch,soundfile as sf
    from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model,load_frames
    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine,video_signature,scoring_environment
    setup_torch()
    geom=read(OUT/'geometry.json');source=np.load(OUT/'source_frames.npz')['frames'];rows=read(OUT/'support.json');boxes=geom['generation_boxes'];roi=geom['score_roi']
    with lease('produce') as p:
        write(OUT/'runtime.json',scoring_environment())
        model=_load_model(W2L/'checkpoints/wav2lip_gan.pth','cuda');engine=SyncNetEngine(batch_size=32,device='cuda')
        try:
            maxframes=max(a['frames'] for r in rows for a in r['arms'].values())
            for cond in CONDITIONS:
                dest=OUT/'source_only'/cond
                if dest.with_suffix('.json').exists():continue
                resource_check(p);frames=[crop_zero_padded(source[i],roi) for i in source_indices(maxframes,cond)];vp=dest.with_suffix('.avi');pix=write_video(frames,vp);v,meta=engine.extract_visual(vp);np.savez_compressed(dest.with_suffix('.npz'),visual=v)
                write(dest.with_suffix('.json'),{'video':str(vp),'video_sha256':sha(vp),'pixel_sha256':pix,'feature_sha256':sha(dest.with_suffix('.npz')),'visual_metadata':meta,'condition':cond,'source_only':True})
            lookup={r['id']:r for r in p['rows']}
            for r in rows:
                sid=r['id']
                for arm,info in r['arms'].items():
                    resource_check(p);ad=OUT/'audio_features'/sid/arm
                    if not ad.with_suffix('.json').exists():
                        a,am=engine.extract_audio(info['audio']);ad.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(ad.with_suffix('.npz'),audio=a)
                        write(ad.with_suffix('.json'),{'path':str(ad.with_suffix('.npz')),'sha256':sha(ad.with_suffix('.npz')),'audio':info['audio'],'audio_sha256':info['audio_sha256'],'metadata':am})
                    chunks=np.load(info['mel_path'])['chunks']
                    for cond in CONDITIONS:
                        resource_check(p);dest=OUT/'features'/sid/arm/cond
                        if dest.with_suffix('.json').exists():continue
                        video=render_cell(model,chunks,source,boxes,roi,cond,OUT/'videos'/sid/arm/(cond+'.avi'));v,vm=engine.extract_visual(video['path']);dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),visual=v)
                        write(dest.with_suffix('.json'),{'id':sid,'arm':arm,'condition':cond,'video':video,'path':str(dest.with_suffix('.npz')),'sha256':sha(dest.with_suffix('.npz')),'visual_metadata':vm})
                    bridge=OUT/'bridge'/sid/arm
                    if not bridge.with_suffix('.json').exists():
                        resource_check(p);old=lookup[sid]['cells'][arm];assert sha(old['video'])==old['video_sha256'];full,fps=load_frames(Path(old['video']));assert fps==25
                        vp=OUT/'bridge_videos'/sid/(arm+'.avi');pix=write_video([crop_zero_padded(f,roi) for f in full],vp);v,vm=engine.extract_visual(vp);bridge.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(bridge.with_suffix('.npz'),visual=v)
                        write(bridge.with_suffix('.json'),{'path':str(bridge.with_suffix('.npz')),'sha256':sha(bridge.with_suffix('.npz')),'old_video':old['video'],'old_video_sha256':old['video_sha256'],'fixed_roi_video':str(vp),'video_sha256':sha(vp),'pixel_sha256':pix,'visual_metadata':vm})
                print('produced',r['split'],sid,flush=True)
            first=min((r for r in rows if r['split']=='calibration'),key=lambda r:r['id']);sid=first['id'];repeats={}
            for arm,info in first['arms'].items():
                resource_check(p);pcm,sr=sf.read(info['audio'],dtype='int16');assert sr==16000;delayed=np.zeros_like(pcm);delayed[3200:]=pcm[:-3200];ap=OUT/'controls'/(arm+'_delay5.wav');ap.parent.mkdir(parents=True,exist_ok=True);sf.write(ap,delayed,16000,subtype='PCM_16');a,am=engine.extract_audio(ap);np.savez_compressed(ap.with_suffix('.npz'),audio=a);write(ap.with_suffix('.json'),{'sha256':sha(ap),'features_sha256':sha(ap.with_suffix('.npz')),'metadata':am,'id':sid,'samples_shifted':3200})
                chunks=np.load(info['mel_path'])['chunks']
                for cond in CONDITIONS:
                    resource_check(p);video=render_cell(model,chunks,source,boxes,roi,cond,OUT/'controls'/f'{arm}_{cond}_repeat.avi');v,vm=engine.extract_visual(video['path']);old=read(OUT/'features'/sid/arm/(cond+'.json'));orig=np.load(old['path'])['visual'];err=float(abs(v-orig).max());np.savez_compressed(OUT/'controls'/f'{arm}_{cond}_repeat.npz',visual=v)
                    repeats[f'{arm}/{cond}']={'pixel_equal':video['pixel_sha256']==old['video']['pixel_sha256'],'feature_max_error':err,'video':video}
            write(OUT/'repeat_controls.json',{'id':sid,'repeats':repeats,'passed':all(x['pixel_equal'] and x['feature_max_error']<1e-5 for x in repeats.values())})
        finally:engine.close()
    print('features complete; no scoring performed',flush=True)


def seal():
    p=protocol();assert not (OUT/'scores').exists()
    features={str(f):sha(f) for folder in ('features','audio_features','source_only','bridge','controls') for f in (OUT/folder).rglob('*') if f.suffix in ('.npz','.json')}
    assert len(list((OUT/'features').glob('*/*/*.json')))==592
    assert read(OUT/'repeat_controls.json')['passed']
    write(OUT/'feature_seal.json',features)
    write(OUT/'score_lock.json',{'status':'before_any_new_distance_scoring','protocol_sha256':sha(OUT/'protocol.json'),'support_sha256':sha(OUT/'support.json'),'geometry_sha256':sha(OUT/'geometry.json'),'feature_seal_sha256':sha(OUT/'feature_seal.json'),'repeat_controls_sha256':sha(OUT/'repeat_controls.json'),'time':time.time()})


def matrix(v,a):
    import torch
    n=len(v);assert len(a)==n
    vv=torch.from_numpy(np.asarray(v,dtype=np.float32));aa=np.pad(np.asarray(a,dtype=np.float32),((15,15),(0,0)))
    with torch.inference_mode():return np.stack([torch.nn.functional.pairwise_distance(vv,torch.from_numpy(aa[15+s:15+s+n]),eps=1e-6).numpy().astype(np.float64) for s in LAGS],1)


def summarize_matrix(m,policy,k):
    n=len(m)
    if policy=='valid':c=np.array([m[max(0,-s):min(n,n-s),s+15].mean() for s in LAGS])
    else:
        g=int(policy[5:]);c=(m[g:n-g] if g else m).mean(0)
    j=int(np.argmin(c));b=float(np.median(c));d=float(c[j]);anchor=float(c[k+15])
    return {'C':b-d,'B':b,'D':d,'D_anchor':anchor,'C_anchor':b-anchor,'best_lag':j-15,'offset':15-j,'curve':c.tolist()}


def normalize(v):return (v.astype(float)/np.linalg.norm(v.astype(float),axis=1)[:,None]).astype(np.float32)


def cell_features(sid,arm,condition,kind,L,geom):
    a=np.load(OUT/'audio_features'/sid/(arm+'.npz'))['audio'][:L]
    path=OUT/'features'/sid/arm/(condition+'.npz') if kind=='generated' else OUT/'source_only'/(condition+'.npz')
    v=np.load(path)['visual'][:L]
    return (normalize(v),normalize(a)) if geom=='unit' else (v,a)


def score_split(split):
    import torch
    torch.set_num_threads(2);p=protocol();lock=read(OUT/'score_lock.json');assert sha(OUT/'feature_seal.json')==lock['feature_seal_sha256']
    for f,h in read(OUT/'feature_seal.json').items():assert sha(f)==h,f
    k=0 if split=='calibration' else read(OUT/'calibration.json')['k0'];rows=[r for r in read(OUT/'support.json') if r['split']==split]
    for r in rows:
        if not r['eligible']:continue
        path=OUT/'scores'/split/(r['id']+'.json')
        if path.exists():continue
        cells={};matrices={}
        for arm,info in r['arms'].items():
            for cond in CONDITIONS:
                for kind in ('generated','source_only'):
                    for geom in ('raw','unit'):
                        v,a=cell_features(r['id'],arm,cond,kind,info['L'],geom);m=matrix(v,a);key=f'{kind}/{geom}/{arm}/{cond}';matrices[key]=m
                        cells[key]={pol:summarize_matrix(m,pol,k) for pol in p['evaluation']['policies']}
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**matrices);write(path,{'id':r['id'],'speaker':r['speaker'],'split':split,'k_at_scoring':k,'cells':cells,'matrices_sha256':sha(path.with_suffix('.npz'))})
        print('scored',split,r['id'],flush=True)
    if split=='calibration':
        assert not (OUT/'scores/evaluation').exists();lags=defaultdict(list)
        for r in rows:
            x=read(OUT/'scores/calibration'/(r['id']+'.json'))
            for arm in ARMS:
                for cond in CONDITIONS:lags[f'{cond}/{arm}'].append(x['cells'][f'generated/raw/{arm}/{cond}']['guard20']['best_lag'])
        medians={key:float(np.median(v)) for key,v in lags.items()};k=int(np.median([x for v in lags.values() for x in v]));assert abs(k)<=15
        write(OUT/'calibration.json',{'k0':k,'median_by_condition_arm':medians,'anchor_comparable':max(medians.values())-min(medians.values())<=1,'source_only_not_used_to_fit_k':True,'eval_scores_absent':True,'cal_scores_hashes':{str(f):sha(f) for f in (OUT/'scores/calibration').glob('*')},'created_epoch':time.time()})
        write(OUT/'eval_release.json',{'calibration_sha256':sha(OUT/'calibration.json'),'score_lock_sha256':sha(OUT/'score_lock.json'),'eval_ids':[r['id'] for r in read(OUT/'support.json') if r['split']=='evaluation'],'status':'frozen_before_evaluation_scores'})


def aggregate():
    k=read(OUT/'calibration.json')['k0'];rows=read(OUT/'support.json');results={}
    for subset in ('evaluation','all'):
        selected=[r for r in rows if r['eligible'] and (subset=='all' or r['split']==subset)]
        for kind in ('generated','source_only'):
            for geom in ('raw','unit'):
                for policy in ('guard20','valid','guard0'):
                    values=defaultdict(list)
                    for r in selected:
                        dat=np.load(OUT/'scores'/r['split']/(r['id']+'.npz'));gaps={}
                        for cond in CONDITIONS:
                            both={a:summarize_matrix(dat[f'{kind}/{geom}/{a}/{cond}'],policy,k) for a in ARMS}
                            gaps[cond]={f:both['T'][f]-both['N'][f] for f in ('C','B','D','D_anchor','C_anchor')}
                        for f in gaps['D0']:
                            vals={c:gaps[c][f] for c in CONDITIONS};vals.update(dynamic_minus_static=(gaps['D0'][f]+gaps['D66'][f]-gaps['S0'][f]-gaps['S66'][f])/2,D66_minus_D0=gaps['D66'][f]-gaps['D0'][f],S66_minus_S0=gaps['S66'][f]-gaps['S0'][f])
                            for name,v in vals.items():values[f'{name}/{f}'].append(v)
                    results[f'{subset}/{kind}/{geom}/{policy}']={name:stats(v,[r['speaker'] for r in selected]) for name,v in values.items()}
    write(OUT/'summary.json',results)


def bridge_and_delay():
    p=protocol();k=read(OUT/'calibration.json')['k0'];rows=read(OUT/'support.json');lookup={r['id']:r for r in p['rows']};items=[]
    for r in rows:
        sid=r['id'];cells={}
        for arm,info in r['arms'].items():
            old=lookup[sid]['cells'][arm];z=read(OUT/'bridge'/sid/(arm+'.json'));vold=np.load(PARENT/'features'/sid/'features.npz')[f'v_{arm}'];vfix=np.load(z['path'])['visual'];vnew=np.load(OUT/'features'/sid/arm/'D0.npz')['visual'];a=np.load(OUT/'audio_features'/sid/(arm+'.npz'))['audio']
            oldF=old['track_frames'];fixF=z['visual_metadata']['frame_count'];L=min(info['L'],min(oldF,fixF,info['samples']//640)-5)
            cells[arm]={'L':L,'conditions':{}}
            for name,v in [('new_fixed',vnew),('old_fixed',vfix),('old_crop',vold)]:
                m=matrix(v[:L],a[:L]);cells[arm]['conditions'][name]={pol:summarize_matrix(m,pol,k) for pol in ('guard20','valid','guard0')}
        items.append({'id':sid,'speaker':r['speaker'],'split':r['split'],'cells':cells})
    write(OUT/'bridge_cells.json',items);summary={}
    for subset in ('calibration','evaluation','all'):
        selected=[r for r in items if subset=='all' or r['split']==subset];vals=defaultdict(list)
        for r in selected:
            for pol in ('guard20','valid','guard0'):
                for f in ('C','B','D'):
                    for arm in ARMS:
                        c=r['cells'][arm]['conditions'];v={name:c[name][pol][f] for name in c};vals[f'{pol}/{arm}/render/{f}'].append(v['new_fixed']-v['old_fixed']);vals[f'{pol}/{arm}/geometry/{f}'].append(v['old_fixed']-v['old_crop'])
                    for name in ('new_fixed','old_fixed','old_crop'):
                        vals[f'{pol}/native/{name}/{f}'].append(r['cells']['T']['conditions'][name][pol][f]-r['cells']['N']['conditions'][name][pol][f])
        summary[subset]={key:stats(v,[r['speaker'] for r in selected]) for key,v in vals.items()}
    write(OUT/'bridge_summary.json',summary)
    first=min((r for r in rows if r['split']=='calibration'),key=lambda r:r['id']);sid=first['id'];controls={}
    for arm,info in first['arms'].items():
        L=info['L'];a=np.load(OUT/'audio_features'/sid/(arm+'.npz'))['audio'][:L];ad=np.load(OUT/'controls'/(arm+'_delay5.npz'))['audio'][:L]
        for cond in CONDITIONS:
            v=np.load(OUT/'features'/sid/arm/(cond+'.npz'))['visual'][:L];m=matrix(v,a);md=matrix(v,ad);I=np.arange(25,L-25);assert len(I)>0
            base=m[I].mean(0);delay=md[I].mean(0);err=float(abs(delay[5:]-base[:-5]).max());b=int(np.argmin(base))-15;d=int(np.argmin(delay))-15
            controls[f'{arm}/{cond}']={'rows':I.tolist(),'base_bestlag':b,'delayed_bestlag':d,'expected_delayed_bestlag':b+5,'offset_change':b-d,'overlap_curve_max_error':err,'passed':err<2e-4 and d==b+5}
    write(OUT/'delay_controls.json',{'id':sid,'controls':controls,'passed':all(x['passed'] for x in controls.values()),'interpretation':'positive PCM delay moves distance-curve bestlag +5 and offset -5; same rows, overlapping lag curves independently compared'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare','produce','seal','calibration','evaluation','aggregate','bridge_and_delay'));args=parser.parse_args()
    if args.stage in ('calibration','evaluation'):score_split(args.stage)
    else:globals()[args.stage]()
