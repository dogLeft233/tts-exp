"""Offline paired ASR check on the fixed twenty fit calibration sources."""
from pathlib import Path
import random
import re
import hashlib
import numpy as np
import torch
from calibrate import ROOT,RUN as PARENT,read,write,sha

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'
MODEL=Path('/home/wjj/tts-audio/model_cache/funasr_20261024/paraformer-zh')


def norm(s):return ''.join(re.findall(r'[\u3400-\u9fffA-Za-z0-9]',s)).lower()


def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=row
    return prev[-1]


def main():
    from funasr import AutoModel
    torch.set_num_threads(2)
    rows=[x for x in read(PARENT/'input_manifest.json')['calibration'] if x['arm']=='N']
    write(RUN/'asr_contract.json',{'model':str(MODEL),'model_files':{p.name:sha(p) for p in MODEL.iterdir() if p.is_file()},
          'cohort':[x['id'] for x in rows],'cpu_threads':2,'per_source_identical_rng_across_arms':True,
          'floors':{'additional_corpus_cer_max':0.02,'additional_edits_per_source_max':2},
          'comparison':'regular minus original natural; prematched is descriptive diagnostic',
          'script_sha256':sha(__file__),'dev_or_confirm_used':False})
    model=AutoModel(model=str(MODEL),device='cpu',disable_update=True,disable_pbar=True)
    results=[]
    for x in rows:
        seed=int(hashlib.sha256(x['id'].encode()).hexdigest()[:8],16)
        for kind,path in [('A',Path(x['canonical_audio_path'])),
                          ('prematch',PARENT/'reconstruction'/f'{x["id"]}.wav'),
                          ('regular',RUN/'diagnostic_audio/regular_raw'/f'{x["id"]}.wav')]:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            pred=model.generate(input=str(path),cache={},batch_size_s=300,disable_pbar=True)[0]['text']
            record={'id':x['id'],'kind':kind,'text':pred,'reference':x['text'],
                    'reference_characters':len(norm(x['text'])),'edit_errors':distance(norm(x['text']),norm(pred)),
                    'audio_sha256':sha(path),'seed':seed}
            results.append(record);write(RUN/'asr_items'/f'{x["id"]}_{kind}.json',record)
        print('ASR_COMPLETE',x['id'],flush=True)
    chars=sum(x['reference_characters'] for x in results if x['kind']=='A')
    edits={kind:sum(x['edit_errors'] for x in results if x['kind']==kind) for kind in ['A','prematch','regular']}
    diffs=[]
    for x in rows:
        group={r['kind']:r for r in results if r['id']==x['id']}
        diffs.append({'id':x['id'],'additional_edits':group['regular']['edit_errors']-group['A']['edit_errors'],
                      'A_to_R_transcript_edits':distance(norm(group['A']['text']),norm(group['regular']['text']))})
    passed=(edits['regular']-edits['A'])/chars<=0.02 and max(x['additional_edits'] for x in diffs)<=2
    write(RUN/'asr_quality.json',{'status':'PASS_PAIRED_ASR' if passed else 'HOLD_ASR',
        'characters':chars,'errors':edits,'cer':{k:v/chars for k,v in edits.items()},'paired_differences':diffs,
        'rows':results,'limits':'Recognizer agreement is not human intelligibility or exact content truth',
        'development_allowed':False,'confirmation_allowed':False})


if __name__=='__main__':main()
