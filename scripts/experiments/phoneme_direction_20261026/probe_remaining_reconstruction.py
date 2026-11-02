"""Remaining preprocessing hypotheses on failed fit cases plus a fixed control."""
import sys
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from calibrate import ROOT,RUN as PARENT,read,write,sha
from diagnose_reconstruction import RUN,REGULAR


def main():
    torch.set_num_threads(4);torch.manual_seed(20261026);torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'third_party/knn-vc'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    adapter=WavLMKNNVCAdapter.load_pretrained(device='cpu',source=ROOT/'third_party/knn-vc')
    g=Generator(AttrDict(read(ROOT/'third_party/knn-vc/hifigan/config_v1_wavlm.json')))
    g.load_state_dict(torch.load(REGULAR,map_location='cpu',weights_only=False)['generator']);g.eval();g.remove_weight_norm()
    failures={x['id'] for x in read(RUN/'regular_full_fit_quality.json')['failed']}
    source=[x for x in read(PARENT/'input_manifest.json')['fit'] if x['arm']=='N']
    chosen=[x for x in source if x['id'] in failures or x['id']==source[0]['id']]
    write(RUN/'remaining_probe_contract.json',{'rows':[x['id'] for x in chosen],
          'model_cfg_normalize':bool(adapter.model.wavlm.cfg.normalize),
          'variants':['input_rms_minus16db','input_unit_variance','training_right_pad'],
          'diagnostic_only':True,'floors_unchanged':True,'source_sha256':sha(__file__)})
    results=[]
    for x in chosen:
        a,sr=sf.read(x['canonical_audio_path'],dtype='float32')
        for kind in ['input_rms_minus16db','input_unit_variance','training_right_pad']:
            b=torch.from_numpy(a)
            if kind=='input_rms_minus16db':b=b*(10**(-16/20)/b.square().mean().sqrt())
            if kind=='input_unit_variance':b=F.layer_norm(b,b.shape)
            if kind=='training_right_pad':pad=320-(len(a)%320)
            else:pad=((len(a)+319)//320)*320+80-len(a)
            h=adapter.extract(F.pad(b,(0,pad)))
            with torch.inference_mode():raw=g(h[None]).flatten().numpy().astype('float64')
            raw=np.pad(raw[:len(a)],(0,max(0,len(a)-len(raw))))
            raw*=np.sqrt(np.mean(a.astype('float64')**2))/np.sqrt(np.mean(raw**2))
            p=RUN/'remaining_probe_audio'/kind/f'{x["id"]}.wav';p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():raise FileExistsError(p)
            sf.write(p,raw,sr,subtype='FLOAT')
            results.append({'id':x['id'],'kind':kind,'source':x['canonical_audio_path'],'output':str(p),'output_sha256':sha(p)})
        print('PROBED',x['id'],flush=True)
    write(RUN/'remaining_probe_outputs.json',{'rows':results})


if __name__=='__main__':main()
