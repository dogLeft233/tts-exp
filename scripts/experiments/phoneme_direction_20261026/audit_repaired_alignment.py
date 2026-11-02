"""Audit every fit-only repaired TextGrid without weakening the original gates."""
import argparse
from collections import Counter,defaultdict
from praatio import textgrid
from calibrate import ROOT,RUN as PARENT,read,write,sha,normalize_text
from audit_alignment_readback import matched_words,UNKNOWN,SILENCE

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'


def audit(which):
    if which not in ['calibration','fit']:raise ValueError('Fit only')
    rows=read(PARENT/'input_manifest.json')[which];failures=[];records=[]
    for x in rows:
        if x['split']!='fit':raise ValueError('Sealed split')
        p=RUN/f'mfa_grids_{which}'/x['speaker_id']/(x['id']+'.TextGrid')
        if not p.is_file():failures.append({'id':x['id'],'error':'missing_grid'});continue
        g=textgrid.openTextgrid(str(p),includeEmptyIntervals=True)
        words=[(a,b,c) for a,b,c in g.getTier('words').entries if c not in SILENCE]
        phones=list(g.getTier('phones').entries)
        if [c for _,_,c in words]!=normalize_text(x['text']).split():
            failures.append({'id':x['id'],'error':'word_sequence'})
        unknown=[(a,b,c) for a,b,c in phones if c.lower() in UNKNOWN]
        if unknown:failures.append({'id':x['id'],'error':'unknown_speech','phones':unknown})
        if any(b<=a or a<0 or b>x['samples']/16000+0.001 for a,b,c in phones):
            failures.append({'id':x['id'],'error':'boundary'})
        if any(p1[1]>p2[0]+1e-6 for p1,p2 in zip(phones,phones[1:])):
            failures.append({'id':x['id'],'error':'nonmonotonic'})
        tokens=[]
        for pos,(a,b,char) in enumerate(words):
            members=[(s,e,c) for s,e,c in phones if c not in SILENCE and a-1e-5<=(s+e)/2<=b+1e-5]
            if not members:failures.append({'id':x['id'],'error':'empty_character','position':pos})
            for j,(s,e,c) in enumerate(members):tokens.append({'text_position':pos,'phone_index':j,'start':s,'end':e,'label':c})
        records.append({'id':x['id'],'speaker_id':x['speaker_id'],'sample_id':x['sample_id'],'arm':x['arm'],'draw':x['draw'],
                        'tokens':tokens,'grid_sha256':sha(p),'grid_path':str(p)})
    groups=defaultdict(list)
    for x in records:groups[x['sample_id']].append(x)
    pairs=[]
    for source,group in groups.items():
        n=next(x for x in group if x['arm']=='N')
        for t in group:
            if t['arm']=='N':continue
            matched,bad=matched_words(n['tokens'],t['tokens'])
            pairs.append({'source':source,'tts_id':t['id'],'matched':len(matched),'unsupported_positions':bad,
                          'natural_phone_count':len(n['tokens'])})
    payload={'status':'PASS_STRUCTURAL_NO_OOV' if not failures and len(records)==len(rows) else 'HOLD_ALIGNMENT',
       'scope':which,'units':len(records),'sources':len(groups),'speakers':len({x['speaker_id'] for x in records}),
       'failures':failures,'failure_types':dict(Counter(x['error'] for x in failures)),'pairs':pairs,'records':records,
       'dictionary_sha256':read(RUN/'dictionary_overlay_receipt.json')['overlay_sha256'],
       'time_truth_verified':False,'development_allowed':False,'confirmation_allowed':False}
    write(RUN/f'alignment_audit_{which}.json',payload)
    print({k:v for k,v in payload.items() if k not in ['records','pairs','failures']},flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=['calibration','fit']);a=p.parse_args();audit(a.scope)
