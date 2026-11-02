"""Controlled fit-only decoder probes; never admits a changed model to evaluation."""
import argparse
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
from calibrate import ROOT, RUN as PARENT, read, write, sha

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'
REGULAR=ROOT/'runs/cogenav_decoder_pilot_20261025_attempt01/g_02500000.pt'


def infer():
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(ROOT/'scripts'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    torch.set_num_threads(4);torch.manual_seed(20261026)
    torch.use_deterministic_algorithms(True)
    adapter=WavLMKNNVCAdapter.load_pretrained(device='cpu',source=ROOT/'third_party/knn-vc')
    original=read(PARENT/'input_manifest.json')
    selected=[x for x in original['calibration'] if x['arm']=='N']
    if any(x['split']!='fit' for x in selected):raise ValueError('Fit only')
    write(RUN/'diagnostic_contract.json',{'parent_manifest_sha256':sha(PARENT/'input_manifest.json'),
          'parent_contract_sha256':sha(PARENT/'calibration_contract.json'),
          'prematch_sha256':adapter.metadata()['vocoder_checkpoint_sha256'],
          'regular_sha256':sha(REGULAR),'regular_is_diagnostic_only':True,
          'context_frames':16,'floors_unchanged':True,'development_allowed':False,'confirmation_allowed':False,
          'source_sha256':sha(__file__)})
    outputs=[]
    for kind in ['prematch_context16','regular_raw','regular_context16']:
        if kind=='regular_raw':
            # Use the same architecture. Only the pretrained decoder state changes.
            sys.path.insert(0,str(ROOT/'third_party/knn-vc'))
            from hifigan.models import Generator
            from hifigan.utils import AttrDict
            generator=Generator(AttrDict(read(ROOT/'third_party/knn-vc/hifigan/config_v1_wavlm.json')))
            generator.load_state_dict(torch.load(REGULAR,map_location='cpu',weights_only=False)['generator'])
            generator.eval();generator.remove_weight_norm()
            for p in generator.parameters():p.requires_grad_(False)
        for x in selected:
            a,sr=sf.read(x['canonical_audio_path'],dtype='float32')
            h=torch.from_numpy(np.load(PARENT/'reconstruction'/f'{x["id"]}_features.npy'))
            context=16 if 'context16' in kind else 0
            padded=F.pad(h.T[None],(context,context),mode='replicate').transpose(1,2) if context else h[None]
            with torch.inference_mode():
                if kind.startswith('prematch'):raw=adapter.model.vocode(padded).flatten().numpy()
                else:raw=generator(padded).flatten().numpy()
            raw=raw[context*320:context*320+len(a)].astype('float64')
            if len(raw)!=len(a):raise ValueError('Length changed')
            raw*=np.sqrt(np.mean(a.astype('float64')**2))/np.sqrt(np.mean(raw**2))
            path=RUN/'diagnostic_audio'/kind/f'{x["id"]}.wav';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError(path)
            sf.write(path,raw,sr,subtype='FLOAT')
            outputs.append({'kind':kind,'id':x['id'],'source':x['canonical_audio_path'],
                            'output':str(path),'sha256':sha(path)})
        print(kind,'complete',flush=True)
    # Encoder preprocessing hypothesis: minimize to the first clip plus the failed-floor clip.
    for x in [selected[0],next(x for x in selected if x['speaker_id']=='S0200')]:
        a,sr=sf.read(x['canonical_audio_path'],dtype='float32')
        for kind in ['prematch_no_input_pad','prematch_input_rms016']:
            b=a.copy()
            if kind.endswith('rms016'):
                b=b*(10**(-16/20)/np.sqrt(np.mean(b.astype('float64')**2)))
                n=(len(b)+319)//320;b=np.pad(b,(0,n*320+80-len(b)))
            h=adapter.extract(torch.from_numpy(b))
            raw=adapter.vocode(h).numpy().astype('float64')
            raw=np.pad(raw[:len(a)],(0,max(0,len(a)-len(raw))))
            raw*=np.sqrt(np.mean(a.astype('float64')**2))/np.sqrt(np.mean(raw**2))
            path=RUN/'diagnostic_audio'/kind/f'{x["id"]}.wav';path.parent.mkdir(parents=True,exist_ok=True)
            sf.write(path,raw,sr,subtype='FLOAT')
            outputs.append({'kind':kind,'id':x['id'],'source':x['canonical_audio_path'],'output':str(path),'sha256':sha(path)})
    write(RUN/'diagnostic_outputs.json',{'rows':outputs})


def measure():
    from pystoi import stoi
    from scipy.signal import correlate,correlation_lags
    from collections import defaultdict
    groups=defaultdict(list)
    for x in read(RUN/'diagnostic_outputs.json')['rows']:
        a,sr=sf.read(x['source']);b,sb=sf.read(x['output'])
        if sr!=16000 or sb!=sr or len(a)!=len(b) or sha(x['output'])!=x['sha256']:raise ValueError('Changed input')
        n=len(a)//160;ea=np.sqrt(np.mean(a[:n*160].reshape(n,160)**2,axis=1));eb=np.sqrt(np.mean(b[:n*160].reshape(n,160)**2,axis=1))
        c=correlate(eb-eb.mean(),ea-ea.mean(),method='fft');lags=correlation_lags(n,n);mask=abs(lags)<=20
        groups[x['kind']].append({'id':x['id'],'stoi':float(stoi(a,b,16000)),
               'stoi_inner':float(stoi(a[8000:-8000],b[8000:-8000],16000)),
               'identity_stoi':float(stoi(a,a,16000)),
               'global_envelope_lag_ms':int(lags[mask][np.argmax(c[mask])])*10,
               'clipped':int(np.sum(abs(b)>=1)),
               'relative_rms_error':float(abs(np.sqrt(np.mean(b*b))/np.sqrt(np.mean(a*a))-1))})
    summary={k:{'n':len(v),'stoi_median':float(np.median([r['stoi'] for r in v])),
                'stoi_min':min(r['stoi'] for r in v),'inner_median':float(np.median([r['stoi_inner'] for r in v])),
                'all_unchanged_floors_pass':bool(len(v)==20 and np.median([r['stoi'] for r in v])>=0.85 and all(r['stoi']>=0.75 and not r['clipped'] and abs(r['global_envelope_lag_ms'])<=20 and r['relative_rms_error']<=0.001 for r in v))} for k,v in groups.items()}
    write(RUN/'diagnostic_quality.json',{'summary':summary,'groups':dict(groups),'diagnostic_only':True,'development_allowed':False,'confirmation_allowed':False})
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','measure']);args=p.parse_args();globals()[args.stage]()
