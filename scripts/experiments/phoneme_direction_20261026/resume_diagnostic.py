"""Continue decoder differential probe after its import-path-only launch failure."""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from calibrate import ROOT, RUN as PARENT, read, write, sha
from diagnose_reconstruction import RUN,REGULAR


def main():
    log=RUN/'diagnostic_infer.log'
    if "No module named 'hubconf'" not in log.read_text():raise ValueError('Unexpected failure')
    selected=[x for x in read(PARENT/'input_manifest.json')['calibration'] if x['arm']=='N']
    for x in selected:
        if not (RUN/'diagnostic_audio/prematch_context16'/f'{x["id"]}.wav').is_file():raise ValueError('Incomplete first probe')
    write(RUN/'diagnostic_import_repair.json',{'failure_log_sha256':sha(log),'repair':'Load original Generator directly from pinned kNN-VC source; state before weight-norm removal',
          'source_sha256':sha(__file__),'regular_sha256':sha(REGULAR),'no_data_or_gate_change':True})
    torch.set_num_threads(4);torch.manual_seed(20261026);torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(ROOT/'third_party/knn-vc'))
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    generator=Generator(AttrDict(read(ROOT/'third_party/knn-vc/hifigan/config_v1_wavlm.json')))
    generator.load_state_dict(torch.load(REGULAR,map_location='cpu',weights_only=False)['generator'])
    generator.eval();generator.remove_weight_norm()
    for p in generator.parameters():p.requires_grad_(False)
    outputs=[]
    for kind in ['prematch_context16','regular_raw','regular_context16']:
        for x in selected:
            path=RUN/'diagnostic_audio'/kind/f'{x["id"]}.wav'
            if kind!='prematch_context16':
                a,sr=sf.read(x['canonical_audio_path'],dtype='float32')
                h=torch.from_numpy(np.load(PARENT/'reconstruction'/f'{x["id"]}_features.npy'))
                context=16 if 'context16' in kind else 0
                h=F.pad(h.T[None],(context,context),mode='replicate').transpose(1,2) if context else h[None]
                with torch.inference_mode():raw=generator(h).flatten().numpy()
                raw=raw[context*320:context*320+len(a)].astype('float64')
                raw*=np.sqrt(np.mean(a.astype('float64')**2))/np.sqrt(np.mean(raw**2))
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise FileExistsError(path)
                sf.write(path,raw,sr,subtype='FLOAT')
            outputs.append({'kind':kind,'id':x['id'],'source':x['canonical_audio_path'],'output':str(path),'sha256':sha(path)})
        print(kind,'complete',flush=True)
    del generator
    sys.path.insert(0,str(ROOT/'scripts'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    adapter=WavLMKNNVCAdapter.load_pretrained(device='cpu',source=ROOT/'third_party/knn-vc')
    for x in [selected[0],next(x for x in selected if x['speaker_id']=='S0200')]:
        a,sr=sf.read(x['canonical_audio_path'],dtype='float32')
        for kind in ['prematch_no_input_pad','prematch_input_rms016']:
            b=a.copy()
            if kind.endswith('rms016'):
                b=b*(10**(-16/20)/np.sqrt(np.mean(b.astype('float64')**2)))
                n=(len(b)+319)//320;b=np.pad(b,(0,n*320+80-len(b)))
            h=adapter.extract(torch.from_numpy(b));raw=adapter.vocode(h).numpy().astype('float64')
            raw=np.pad(raw[:len(a)],(0,max(0,len(a)-len(raw))))
            raw*=np.sqrt(np.mean(a.astype('float64')**2))/np.sqrt(np.mean(raw**2))
            path=RUN/'diagnostic_audio'/kind/f'{x["id"]}.wav';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError(path)
            sf.write(path,raw,sr,subtype='FLOAT')
            outputs.append({'kind':kind,'id':x['id'],'source':x['canonical_audio_path'],'output':str(path),'sha256':sha(path)})
    write(RUN/'diagnostic_outputs.json',{'rows':outputs})


if __name__=='__main__':main()
