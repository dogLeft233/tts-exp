"""MFA timing check on the same fixed fit natural/reconstruction pairs."""
import argparse
import os
import subprocess
from pathlib import Path
import numpy as np
from calibrate import ROOT,RUN as PARENT,MODELS,MFA,read,write,sha,normalize_text
from audit_alignment_readback import UNKNOWN,SILENCE,matched_words
from canonical_alignment import canonical_words

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'


def align():
    rows=[x for x in read(PARENT/'input_manifest.json')['calibration'] if x['arm']=='N']
    dictionary=Path(read(RUN/'mfa_normalization_repair.json')['dictionary'])
    write(RUN/'roundtrip_timing_contract.json',{'cohort':[x['id'] for x in rows],
        'floors':{'matched_phone_fraction_min':0.95,'median_abs_center_ms_max':20,'p95_abs_center_ms_max':40},
        'dictionary_sha256':sha(dictionary),'source_sha256':sha(__file__),'candidate_only':True,
        'boundary_truth_verified':False,'development_allowed':False,'confirmation_allowed':False})
    corpus=RUN/'mfa_corpus_roundtrip'
    for x in rows:
        for kind,path in [('A',Path(x['canonical_audio_path'])),('R',RUN/'diagnostic_audio/regular_raw'/f'{x["id"]}.wav')]:
            d=corpus/x['speaker_id'];d.mkdir(parents=True,exist_ok=True)
            (d/f'{x["id"]}_{kind}.wav').symlink_to(path)
            (d/f'{x["id"]}_{kind}.lab').write_text(normalize_text(x['text'])+'\n')
    env=dict(os.environ,PATH=str(MFA.parent)+os.pathsep+os.environ.get('PATH',''),CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MFA_ROOT_DIR=str(RUN/'mfa_runtime_roundtrip'),TMPDIR=str(RUN/'tmp'))
    command=[str(MFA),'align',str(corpus),str(dictionary),str(MODELS/'acoustic/mandarin_mfa.zip'),str(RUN/'mfa_grids_roundtrip'),
             '--clean','--num_jobs','2']
    with (RUN/'mfa_roundtrip.log').open('x') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
    write(RUN/'mfa_roundtrip_exit.json',{'returncode':p.returncode,'command':command})
    if p.returncode:raise RuntimeError('Roundtrip MFA failed')


def audit():
    from praatio import textgrid
    rows=[x for x in read(PARENT/'input_manifest.json')['calibration'] if x['arm']=='N']
    differences=[];summary=[];failures=[];total=0
    for x in rows:
        sides={}
        for kind in ['A','R']:
            p=RUN/'mfa_grids_roundtrip'/x['speaker_id']/f'{x["id"]}_{kind}.TextGrid'
            g=textgrid.openTextgrid(str(p),includeEmptyIntervals=True)
            phones=list(g.getTier('phones').entries)
            words=[(a,b,c) for a,b,c in g.getTier('words').entries if c not in SILENCE]
            words,_=canonical_words(words,phones,normalize_text(x['text']).split())
            if any(c.lower() in UNKNOWN for a,b,c in phones):failures.append({'id':x['id'],'kind':kind,'error':'unknown'})
            tokens=[]
            for pos,(a,b,c) in enumerate(words):
                members=[(s,e,label) for s,e,label in phones if label not in SILENCE and a-1e-5<=(s+e)/2<=b+1e-5]
                for j,(s,e,label) in enumerate(members):tokens.append({'text_position':pos,'phone_index':j,'start':s,'end':e,'label':label})
            sides[kind]=tokens
        pairs,bad=matched_words(sides['A'],sides['R']);total+=len(sides['A'])
        ds=[]
        for a,b in pairs:
            d=1000*((b['start']+b['end'])-(a['start']+a['end']))/2;ds.append(d)
            differences.append({'id':x['id'],'label':a['label'],'text_position':a['text_position'],
                                'center_shift_ms':d,'start_shift_ms':1000*(b['start']-a['start']),
                                'end_shift_ms':1000*(b['end']-a['end'])})
        summary.append({'id':x['id'],'phones':len(sides['A']),'matched':len(pairs),'unsupported_positions':bad,
                        'median_abs_center_ms':float(np.median(np.abs(ds))) if ds else None})
    shifts=np.array([d['center_shift_ms'] for d in differences]);fraction=len(shifts)/total
    median=float(np.median(abs(shifts)));p95=float(np.percentile(abs(shifts),95))
    passed=not failures and fraction>=.95 and median<=20 and p95<=40
    write(RUN/'roundtrip_timing_quality.json',{'status':'PASS_AUTOMATIC_TIMING' if passed else 'HOLD_TIMING',
       'matched_phone_fraction':fraction,'median_abs_center_ms':median,'p95_abs_center_ms':p95,
       'rows':summary,'phone_differences':differences,'failures':failures,
       'limits':'Same forced-aligner consistency, not independent phoneme boundary truth; no human validation',
       'development_allowed':False,'confirmation_allowed':False})
    print('TIMING',fraction,median,p95,'passed',passed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['align','audit']);a=p.parse_args();globals()[a.stage]()
