"""Full-fit candidate reconstruction calibration, not a formal decoder switch."""
import argparse
import sys
import numpy as np
import soundfile as sf
import torch
from calibrate import ROOT,RUN as PARENT,read,write,sha
from diagnose_reconstruction import RUN,REGULAR


def worker(shard):
    if shard not in range(3):raise ValueError('Expected three fixed shards')
    rows=[x for x in read(PARENT/'input_manifest.json')['fit'] if x['arm']=='N']
    if len(rows)!=300 or any(x['split']!='fit' for x in rows):raise ValueError('Fit only')
    write(RUN/f'regular_fit_shard{shard}_contract.json',{'rows':[x['id'] for x in rows[shard::3]],
          'checkpoint_sha256':sha(REGULAR),'parent_contract_sha256':sha(PARENT/'calibration_contract.json'),
          'script_sha256':sha(__file__),'candidate_only':True,'development_allowed':False,'confirmation_allowed':False})
    torch.set_num_threads(4);torch.manual_seed(20261026);torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'third_party/knn-vc'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    adapter=WavLMKNNVCAdapter.load_pretrained(device='cpu',source=ROOT/'third_party/knn-vc')
    generator=Generator(AttrDict(read(ROOT/'third_party/knn-vc/hifigan/config_v1_wavlm.json')))
    generator.load_state_dict(torch.load(REGULAR,map_location='cpu',weights_only=False)['generator'])
    generator.eval();generator.remove_weight_norm()
    for p in generator.parameters():p.requires_grad_(False)
    outputs=[]
    for x in rows[shard::3]:
        if sha(x['canonical_audio_path'])!=x['canonical_audio_sha256']:raise ValueError('Changed source')
        a,sr=sf.read(x['canonical_audio_path'],dtype='float32');n=(len(a)+319)//320
        h=adapter.extract(torch.from_numpy(np.pad(a,(0,n*320+80-len(a)))))
        if tuple(h.shape)!=(n,1024):raise ValueError('Changed encoder timebase')
        with torch.inference_mode():raw=generator(h[None]).flatten().numpy()[:len(a)].astype('float64')
        if len(raw)!=len(a) or not np.isfinite(raw).all():raise ValueError('Invalid output')
        raw*=np.sqrt(np.mean(a.astype('float64')**2))/np.sqrt(np.mean(raw**2))
        p=RUN/'regular_fit_audio'/f'{x["id"]}.wav';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raise FileExistsError(p)
        sf.write(p,raw,sr,subtype='FLOAT')
        f=RUN/'regular_fit_features'/f'{x["id"]}.npy';f.parent.mkdir(exist_ok=True)
        np.save(f,h.numpy(),allow_pickle=False)
        record={'id':x['id'],'speaker_id':x['speaker_id'],'source':x['canonical_audio_path'],
                 'source_sha256':x['canonical_audio_sha256'],'output':str(p),'output_sha256':sha(p),
                 'features':str(f),'features_sha256':sha(f),'samples':len(a),'frames':len(h)}
        write(RUN/'regular_fit_audio'/f'{x["id"]}.json',record);outputs.append(record)
        print('RECONSTRUCTED',x['id'],flush=True)
    write(RUN/f'regular_fit_shard{shard}_complete.json',{'rows':outputs})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('shard',type=int);a=p.parse_args();worker(a.shard)
