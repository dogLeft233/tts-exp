"""Teacher-free reconstruction from raw CoGenAV [T,768] features and a fitted checkpoint.

Only Adapter+HiFi-GAN run at inference: no WavLM, reference audio, text or training
dataset is opened. Default duration is T*640 samples; --num-samples trims to the
desired duration without resampling the generated waveform.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from common import Adapter,CONTEXT25,HOP25,SR,sha


def load_features(path):
    path=Path(path)
    if path.suffix.lower()=='.npy':
        data=torch.from_numpy(np.load(path,allow_pickle=False))
    elif path.suffix.lower()=='.pt':
        value=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(value,dict) or 'cogen' not in value:
            raise ValueError('PT input must be a dictionary containing raw cogen features')
        data=value['cogen']
    else:
        raise ValueError('Features must be .npy or a cache .pt dictionary')
    if not isinstance(data,torch.Tensor) or data.ndim!=2 or data.shape[1]!=768 or not len(data):
        raise ValueError('Expected nonempty raw [T,768] features')
    if not data.dtype.is_floating_point or not torch.isfinite(data).all():
        raise ValueError('Raw features must be finite floating-point values')
    converted=data.float()
    if not torch.isfinite(converted).all():
        raise ValueError('Raw features cannot be represented as finite FP32 values')
    return converted


def validate_duration(frames,num_samples):
    samples=frames*HOP25 if num_samples is None else num_samples
    if not isinstance(samples,int) or not 0<samples<=frames*HOP25:
        raise ValueError('num_samples must be in (0,T*640]')
    return samples


def decode(features,checkpoint,output,knn_root,num_samples=None,device=None):
    features,checkpoint,output=Path(features),Path(checkpoint),Path(output)
    if output.exists():raise FileExistsError(output)
    raw=load_features(features)
    samples=validate_duration(len(raw),num_samples)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(7.5/16)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    adapter=Adapter().to(device)
    adapter.load_state_dict(state['adapter'],strict=True)
    sys.path.insert(0,str(knn_root))
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    config=AttrDict(json.loads((Path(knn_root)/'hifigan/config_v1_wavlm.json').read_text()))
    generator=Generator(config).to(device)
    generator.load_state_dict(state['generator'],strict=True)
    adapter.eval();generator.eval()
    with torch.inference_mode():
        condition=F.pad(raw.to(device)[None].transpose(1,2),(CONTEXT25,CONTEXT25),mode='replicate').transpose(1,2)
        waveform=generator(adapter(condition))[0,0]
        waveform=waveform[CONTEXT25*HOP25:CONTEXT25*HOP25+samples]
    if len(waveform)!=samples or not torch.isfinite(waveform).all():
        raise ValueError('Nonfinite or wrong-duration reconstruction')
    output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(output,waveform.cpu().numpy(),SR,subtype='FLOAT')
    receipt=dict(status='PASS',features=str(features),features_sha256=sha(features),frames25=len(raw),
                 checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                 trained_protocol_sha256=state['protocol_sha256'],output=str(output),
                 output_sha256=sha(output),num_samples=samples,sample_rate=SR,subtype='FLOAT',
                 device=str(device),teacher_or_reference_audio_used=False,
                 boundary='protocol03 replicate16 eachside, crop16*640 thenexactN')
    with output.with_suffix(output.suffix+'.json').open('x') as f:json.dump(receipt,f,indent=2)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--knn-root',type=Path,default=Path('/home/wjj/tts-audio/tts-exp/third_party/knn-vc'))
    parser.add_argument('--num-samples',type=int)
    parser.add_argument('--device',choices=['cpu','cuda'])
    args=parser.parse_args()
    print(json.dumps(decode(args.features,args.checkpoint,args.output,args.knn_root,args.num_samples,args.device)))
