"""Original VocaLiST preference using verbatim author test_lrs2 functions.

Full-resolution S3FD, no per-clip lag correction, all official K5..K15 endpoints.
"""
import argparse
import ast
import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time

import mtd_official_audit_20261028 as audit

ROOT=audit.ROOT
DEPLOY=audit.DEPLOY
PREVIOUS=audit.PREVIOUS
RUN=ROOT/'runs/vocalist_official_cloud_rescore_20261028_attempt01'
KS=[5,7,9,11,13,15]
read=audit.read
write=audit.write


def freeze():
    rows=read(PREVIOUS/'inputs.json');pairs=read(PREVIOUS/'sync_pairs.json')
    assert len(rows)==80 and len(pairs)==60
    protocol=dict(status='FROZEN_BEFORE_NEW_OFFICIAL_SCORES',cohort='existing Mandarin20 sources,20 natural+60 cloud TTS native videos',
        primary='author K5 samplewise retrieval accuracy, 31 candidates +/-15frames, prediction within +/-1frame of zero',
        secondary='all author K7,K9,K11,K13,K15 temporal score averaging recipes',
        reduction='direct unmodified test_lrs2.py eval_model body; capture samplewise_acc lists at function return',
        source_commit='c265b59bcf7cd8559a265aaf454a9333ec24cb0f',
        weight='official original vocalist_5f_lrs2.pth into models.model.SyncTransformer, strict loading',
        preprocessing='author Dataset; full512 original resolution S3FD per frame, default OpenCV JPEG, cv2 BGR/255 Resize96 bottom48, author16k Mel',
        support='whole original native video/audio overlap, author min_length-5 starts; no guard rows removed, author invalid candidate masks preserved',
        temporal_policy='NO zero-offset calibration, NO best-lag correction, NO shifting source media, NO time stretching',
        reuse='all80 full512 JPEG face crops from official MTD run reused unchanged; original VocaLiST Dataset and all network forwards fresh',
        statistics='60 T-N comparisons,20 independent source clusters; strict12 subset; source bootstrap10000 seed20261028',
        ties='exact equal accuracy is a tie; strict opposite signs are conflicts. Report denominator60 and non-tie conditional denominator separately.',
        interpretation='zero-frame retrieval scores may have floor effects due nonzero native peaks; do not conflate with known offset-change recovery',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_test_sha256=hashlib.sha256((DEPLOY/'vocalist/test_lrs2.py').read_bytes()).hexdigest())
    for name,obj in [('protocol.json',protocol),('inputs.json',rows),('sync_pairs.json',pairs)]:
        path=RUN/name
        if path.exists():assert read(path)==obj
        else:write(path,obj)
    print('FROZEN 80 native media /60 pairs',flush=True)


class QuietProgress:
    def __init__(self,x):self.x=x
    def __iter__(self):return iter(self.x)
    def set_description(self,*args,**kwargs):pass


def official_env():
    import math, os
    import cv2, numpy as np, torch, soundfile as sf
    from glob import glob
    from os.path import dirname, join, basename, isfile
    from natsort import natsorted
    from torchaudio.transforms import MelScale
    sys.path.insert(0,str(DEPLOY/'vocalist'))
    from hparams import hparams
    path=DEPLOY/'vocalist/test_lrs2.py'
    body=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Dataset','calc_pdist','eval_model']]
    env=dict(cv2=cv2,np=np,torch=torch,sf=sf,os=os,glob=glob,dirname=dirname,join=join,basename=basename,isfile=isfile,natsorted=natsorted,
        math=math,v_context=5,mel_step_size=16,BATCH_SIZE=1,hparams=hparams,tqdm=QuietProgress,device=torch.device('cuda'),
        TOP_DB=-hparams.min_level_db,MIN_LEVEL=np.exp(-hparams.min_level_db/-20*np.log(10)),
        melscale=MelScale(n_mels=hparams.num_mels,sample_rate=hparams.sample_rate,f_min=hparams.fmin,
                         f_max=hparams.fmax,n_stft=hparams.n_stft,norm='slaney',mel_scale='slaney'))
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),env)
    return env


def capture_official_reduction(env,model,vid,mel,last,matrix_path):
    import numpy as np
    import torch
    original=env['calc_pdist']
    def recorded_calc(m,v,a,vshift):
        # Only capture eval_model's return; avoid profiling every network call.
        active_profile=sys.getprofile();sys.setprofile(None)
        try:ds=original(m,v,a,vshift=vshift)
        finally:sys.setprofile(active_profile)
        matrix=torch.stack(ds).numpy();assert np.isfinite(matrix).all()
        np.save(matrix_path,matrix)
        return ds
    env['calc_pdist']=recorded_calc
    captured={}
    def on_return(frame,event,arg):
        if event=='return' and frame.f_code is env['eval_model'].__code__:
            for k in KS:captured['K'+str(k)]=float(frame.f_locals['samplewise_acc_k'+str(k)][0])
    old_profile=sys.getprofile()
    try:
        sys.setprofile(on_return)
        env['eval_model']([(vid[None],mel[None],torch.tensor([last]))],torch.device('cuda'),model)
    finally:
        sys.setprofile(old_profile);env['calc_pdist']=original
    assert set(captured)=={'K'+str(k) for k in KS}
    return captured


def independent_reduction(matrix):
    """Transparent truncated moving means, checked against all author endpoints."""
    import numpy as np
    z=np.asarray(matrix,dtype=np.float32)
    result={}
    for k in KS:
        radius=(k-5)//2
        smoothed=np.stack([z[max(0,i-radius):min(len(z),i+radius+1)].mean(0) for i in range(len(z))])
        offsets=15-smoothed.argmax(1)
        result['K'+str(k)]=float(np.mean(np.abs(offsets)<=1))
    return result


def score_all():
    import numpy as np
    import torch
    import soundfile as sf
    assert read(DEPLOY/'smoke/vocalist_receipt.json')['status']=='PASS'
    torch.set_num_threads(4);torch.manual_seed(20261028)
    torch.backends.cudnn.benchmark=False
    env=official_env()
    from models.model import SyncTransformer
    checkpoint=torch.load(DEPLOY/'weights/vocalist/vocalist_5f_lrs2.pth',map_location='cpu',weights_only=True)
    model=SyncTransformer();model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    model.eval().cuda()
    scores=read(RUN/'scores.json') if (RUN/'scores.json').exists() else {}
    rows=read(RUN/'inputs.json')
    write(RUN/'model_load.json',dict(strict=True,model='Original VocaLiST models.model.SyncTransformer',parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__))
    for row in rows:
        key=row['id']
        if key in scores:continue
        start=time.monotonic()
        fixture=audit.RUN/'fixtures'/key/'official512boxes'
        folder=fixture if (fixture/'audio.wav').exists() else ROOT/'runs/mtd_official_cloud_rescore_20261028_attempt01/preprocessed'/key
        assert (folder/'audio.wav').is_file(), str(folder)
        original,sr=sf.read(row['audio_path']);exported,sr2=sf.read(folder/'audio.wav')
        assert sr==sr2==16000 and np.array_equal(original,exported)
        vid,mel,last=audit.dataset_get(env,folder)
        assert 0<=float(vid.min())<=float(vid.max())<=1
        matrix_path=RUN/'matrices'/(key+'.npy');matrix_path.parent.mkdir(exist_ok=True)
        result=capture_official_reduction(env,model,vid,mel,last,matrix_path)
        matrix=np.load(matrix_path)
        independent=independent_reduction(matrix)
        assert independent==result, 'Independent reduction disagrees with author'
        scores[key]=dict(id=key,source=row['sample_id'],arm=row['arm'],frames=last+5,windows=last,
                         metrics=result,seconds=time.monotonic()-start,pcm_exact=True,author_reduction_exact=True,
                         preprocessing_folder=str(folder),video_min=float(vid.min()),video_max=float(vid.max()),
                         global_offset_frames=int(15-matrix.mean(0).argmax()))
        write(RUN/'scores.json',scores)
        write(RUN/'status.json',dict(status='RUNNING',scored=len(scores),total=80,last=key))
        print(json.dumps(dict(scored=len(scores),total=80,id=key,seconds=round(time.monotonic()-start,2))),flush=True)
    write(RUN/'status.json',dict(status='SCORES_COMPLETE',scored=80))


def analyze():
    import cloud_mtd_synchformer_20261028 as previous
    scores=read(RUN/'scores.json');assert len(scores)==80
    nmap={v['source']:v for v in scores.values() if v['arm']=='N'}
    old=read(RUN/'sync_pairs.json');views={};all_pairs=[]
    for k in KS:
        metric='K'+str(k);ps=[]
        for p in old:
            n=nmap[p['source']]['metrics'][metric];t=scores[p['id']]['metrics'][metric]
            ps.append(dict(id=p['id'],source=p['source'],strict_reference=p['strict_reference'],N=n,T=t,delta=t-n,delta_sync_c=p['delta']['SyncC']))
        view={}
        for label,pp in [('all20',ps),('strict12',[p for p in ps if p['strict_reference']])]:
            s=previous.summarize_pairs(pp,metric)
            comparable=sum(p['delta']!=0 and p['delta_sync_c']!=0 for p in pp)
            s.update(comparable_non_tie_pairs=comparable,conflict_fraction_non_ties=s['conflict_count']/comparable if comparable else None,
                     mean_delta_percentage_points=s['mean_delta']*100)
            view[label]=s
        views[metric]=view
        all_pairs.extend(dict(metric=metric,**p) for p in ps)
    summary=dict(status='COMPLETE_ORIGINAL_VOCALIST_OFFICIAL_NATIVE_RETRIEVAL',sources=20,media=80,pairs=60,primary='K5',views=views,
                 no_lag_correction=True,author_reduction_all80_verified=True,
                 caveat='Official zero-offset retrieval can floor on native nonzero offsets; ties are not evidence of agreement. Not calibrated perceptual quality.')
    write(RUN/'summary.json',summary)
    with (RUN/'paired_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['metric','id','source','strict_reference','N','T','delta','delta_sync_c']);w.writeheader();w.writerows(all_pairs)
    write(RUN/'paired_scores.json',all_pairs)
    (RUN/'source_snapshot').mkdir(exist_ok=True);shutil.copyfile(__file__,RUN/'source_snapshot'/Path(__file__).name)
    print(json.dumps({k:v['all20'] for k,v in views.items()},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','score','analyze']);a=p.parse_args()
    dict(freeze=freeze,score=score_all,analyze=analyze)[a.stage]()
