"""Original 80M VocaLiST official LRS2 checkpoint and Dataset GPU smoke."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

DEPLOY=Path('/home/wjj/av-sync-eval-20261028')
ROOT=Path('/home/wjj/tts-audio/tts-exp')


def main():
    import cv2
    import numpy as np
    import torch
    import soundfile as sf
    from glob import glob
    from os.path import dirname,join,basename,isfile
    from natsort import natsorted
    from torchaudio.transforms import MelScale
    torch.set_num_threads(4);torch.manual_seed(20261028)
    assert torch.cuda.is_available()
    folder=DEPLOY/'weights/vocalist'
    archive=folder/'vocalist_5f_lrs2.zip'
    receipt=json.loads(archive.with_suffix('.zip.download.json').read_text())
    assert receipt['status']=='DOWNLOAD_COMPLETE' and archive.stat().st_size==882753219
    weight=folder/'vocalist_5f_lrs2.pth'
    if not weight.exists():
        tmp=weight.with_suffix('.pth.tmp')
        with zipfile.ZipFile(archive) as z:
            info=z.getinfo('VocaLiST_Weights/vocalist_5f_lrs2.pth')
            assert info.file_size==961635703
            with z.open(info) as src,tmp.open('wb') as dest:shutil.copyfileobj(src,dest,1024*1024)
        assert tmp.stat().st_size==961635703;tmp.replace(weight)
    assert weight.stat().st_size==961635703
    source=DEPLOY/'vocalist';sys.path.insert(0,str(source))
    from models.model import SyncTransformer
    from hparams import hparams
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    keys=list(checkpoint)
    model=SyncTransformer();model.load_state_dict(checkpoint['state_dict'],strict=True)
    del checkpoint
    model.eval().cuda()
    text=(source/'test_lrs2.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Dataset')
    env=dict(cv2=cv2,np=np,torch=torch,sf=sf,os=os,glob=glob,dirname=dirname,join=join,basename=basename,isfile=isfile,
        natsorted=natsorted,math=math,v_context=5,hparams=hparams,TOP_DB=-hparams.min_level_db,
        MIN_LEVEL=np.exp(-hparams.min_level_db/-20*np.log(10)),
        melscale=MelScale(n_mels=hparams.num_mels,sample_rate=hparams.sample_rate,f_min=hparams.fmin,
                         f_max=hparams.fmax,n_stft=hparams.n_stft,norm='slaney',mel_scale='slaney'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source/'test_lrs2.py'),'exec'),env)
    fixture=ROOT/'runs/mtd_official_audit_20261028_attempt01/fixtures/BAC009S0002W0122_N_0/official512boxes'
    data=env['Dataset'].__new__(env['Dataset']);data.all_videos=[str(fixture)];data.split='test'
    video,mel,last=data[0]
    assert 0<=float(video.min())<=float(video.max())<=1
    vv=video[:15][None].cuda();aa=mel[None,:,:,:16].cuda()
    with torch.inference_mode():z=model(vv,aa)
    assert torch.isfinite(z).all()
    out=dict(status='PASS',model='Original VocaLiST LRS2 5f',official_release='https://github.com/vskadandale/vocalist/releases/tag/interspeech2022',
        checkpoint='vocalist_5f_lrs2.pth',checkpoint_bytes=weight.stat().st_size,checkpoint_keys=keys,
        source_commit='c265b59bcf7cd8559a265aaf454a9333ec24cb0f',strict_load=True,
        parameters=sum(p.numel() for p in model.parameters()),author_dataset_verbatim=True,
        preprocessing='author test_lrs2 Dataset, cv2 BGR/255, Resize96 lower48, native16k Mel',
        official_test_sha256=hashlib.sha256(text.encode()).hexdigest(),
        input_video_shape=list(vv.shape),input_audio_shape=list(aa.shape),video_min=float(vv.min()),video_max=float(vv.max()),
        raw_logit=z.cpu().tolist(),torch=torch.__version__,seconds=time.monotonic()-start,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),deployment_smoke_only=True,scientific_validity_test=False)
    target=DEPLOY/'smoke/vocalist_receipt.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
