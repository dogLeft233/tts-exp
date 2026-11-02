"""True frozen official AVE features per whole recording and gain, with original timing."""
import argparse,ast,hashlib,json
from pathlib import Path
from os.path import basename
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import librosa
from scipy import signal
import soundfile as sf

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def load_original_classes(official):
    network=ast.parse((official/'nerf_triplane/network.py').read_text())
    nodes=[x for x in network.body if isinstance(x,ast.ClassDef) and x.name in ['Conv2d','AudioEncoder']]
    assert [x.name for x in nodes]==['Conv2d','AudioEncoder']
    scope=dict(torch=torch,nn=nn,__name__='official_ave_ast')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(official/'nerf_triplane/network.py'),'exec'),scope)
    names=['load_wav','preemphasis','melspectrogram','_stft','_linear_to_mel','_build_mel_basis','_amp_to_db','_normalize','AudDataset']
    utils=ast.parse((official/'nerf_triplane/utils.py').read_text())
    nodes=[x for x in utils.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert set(x.name for x in nodes)==set(names)
    scope.update(np=np,librosa=librosa,signal=signal,basename=basename)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(official/'nerf_triplane/utils.py'),'exec'),scope)
    return scope['AudioEncoder'],scope['AudDataset']

def run(root,identity):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    official=root/'official_exact'
    AudioEncoder,AudDataset=load_original_classes(official)
    model=AudioEncoder().cuda().eval();checkpoint=official/'nerf_triplane/checkpoints/audio_visual_encoder.pth'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict({'audio_encoder.'+k:v for k,v in state.items()},strict=True);model.requires_grad_(False)
    out=root/'features'/identity;assert not out.exists();out.mkdir(parents=True)
    model_hash_before=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.state_dict().values())).hexdigest()
    rows=[]
    def forward(dataset):
        outputs=[]
        with torch.no_grad():
            for mel in DataLoader(dataset,batch_size=64,shuffle=False):outputs.append(model(mel.cuda()).cpu())
        values=torch.cat(outputs,dim=0);assert values.ndim==2 and values.shape[1]==512
        return torch.cat([values[:1].repeat(2,1),values,values[-1:].repeat(2,1)],dim=0).numpy()
    for role in ['main','test']:
        pcm,rate=sf.read(root/'material'/identity/f'{role}_original.wav',dtype='int16');assert rate==16000
        baseline_mel=None
        for gain in [-6,0,6]:
            wav=root/'material'/identity/f'{role}_driver_{gain:+d}.wav'
            signal32=librosa.core.load(str(wav),sr=16000)[0]
            expected=(pcm.astype(np.float64)/32768*10**(gain/20)).astype(np.float32)
            assert np.array_equal(signal32,expected) and signal32.dtype==np.float32
            dataset=AudDataset(str(wav));padded=forward(dataset)
            assert np.isfinite(padded).all() and len(padded)>=len(pcm)//640
            if gain==0:
                original=AudDataset(str(root/'material'/identity/f'{role}_original.wav'))
                assert np.array_equal(dataset.orig_mel,original.orig_mel)
                repeated=forward(dataset);assert np.array_equal(padded,repeated)
                baseline_mel=dataset.orig_mel.copy()
            prefix=f'{role}_{gain:+d}';path=out/(prefix+'_ave_padded.npy');np.save(path,padded)
            indexes=np.linspace(0,len(dataset)-1,8).astype(int)
            np.save(out/(prefix+'_mel_control.npy'),np.stack([dataset.crop_audio_window(dataset.orig_mel,int(i)) for i in indexes]))
            row=dict(recording=role,gain_db=gain,wave_sha256=sha(wav),wave_samples=len(signal32),peak_preserved=float(np.abs(signal32).max()),
                mel_shape=list(dataset.orig_mel.shape),mel_dtype=str(dataset.orig_mel.dtype),mel_sha256=hashlib.sha256(dataset.orig_mel.tobytes()).hexdigest(),
                features_path=str(path.relative_to(out)),features_sha256=sha(path),features_shape=list(padded.shape),full_repeat_exact=(gain==0),
                original_pcm_zero_mel_exact=(gain==0))
            rows.append(row);print(json.dumps(dict(identity=identity,recording=role,gain_db=gain,phase='AVE',frames=len(padded))),flush=True)
    model_hash_after=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.state_dict().values())).hexdigest()
    assert model_hash_before==model_hash_after
    result=dict(identity=identity,passed=True,rows=rows,frozen_model_unchanged=True,weights_sha256=sha(checkpoint),
        encoder_source_sha256=sha(official/'nerf_triplane/network.py'),audio_frontend_source_sha256=sha(official/'nerf_triplane/utils.py'),
        timing='original 80Hz mel/25fps crops; original two-frame edge padding, whole separate recordings, no normalization/limiter/retiming',training_admitted=False)
    (out/'ave_qualification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
