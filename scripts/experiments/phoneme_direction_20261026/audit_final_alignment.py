"""Full fit alignment audit with evidence-bound orthographic/silence normalization."""
from collections import defaultdict,Counter
import sqlite3
from praatio import textgrid
from calibrate import ROOT,RUN as PARENT,read,write,sha,normalize_text
from audit_alignment_readback import UNKNOWN,SILENCE,matched_words
from canonical_alignment import canonical_words

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'


def main():
    db=RUN/'mfa_runtime_fit_revision02/mfa_corpus_fit/mfa_corpus_fit.db'
    c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
    # Only database-confirmed silence can authorize dropping an orphan word label.
    evidence=defaultdict(list)
    for name,a,b,phone,word in c.execute("select f.name,pi.begin,pi.end,p.phone,w.word from phone_interval pi join phone p on p.id=pi.phone_id join word_interval wi on wi.id=pi.word_interval_id join word w on w.id=wi.word_id join utterance u on u.id=pi.utterance_id join file f on f.id=u.file_id where p.phone='sil' and w.word not in ('<eps>','<unk>')"):
        evidence[name].append((a,b))
    c.close()
    failures=[];records=[];corrections=[]
    rows=read(PARENT/'input_manifest.json')['fit']
    for x in rows:
        if x['split']!='fit':raise ValueError('Fit only')
        p=RUN/'mfa_grids_fit_revision02'/x['speaker_id']/(x['id']+'.TextGrid')
        if not p.is_file():failures.append({'id':x['id'],'error':'missing_grid'});continue
        g=textgrid.openTextgrid(str(p),includeEmptyIntervals=True)
        raw_words=[(a,b,c) for a,b,c in g.getTier('words').entries if c not in SILENCE]
        phones=list(g.getTier('phones').entries)
        try:words,changes=canonical_words(raw_words,phones,normalize_text(x['text']).split(),evidence[x['id']])
        except ValueError:
            failures.append({'id':x['id'],'error':'word_sequence'});words=raw_words;changes=[]
        corrections.extend(dict(change,id=x['id']) for change in changes)
        if any(c.lower() in UNKNOWN for a,b,c in phones):failures.append({'id':x['id'],'error':'unknown_speech'})
        if any(b<=a or a<0 or b>x['samples']/16000+0.001 for a,b,c in phones):failures.append({'id':x['id'],'error':'boundary'})
        if any(p1[1]>p2[0]+1e-6 for p1,p2 in zip(phones,phones[1:])):failures.append({'id':x['id'],'error':'nonmonotonic'})
        tokens=[]
        for pos,(a,b,char) in enumerate(words):
            members=[(s,e,c) for s,e,c in phones if c not in SILENCE and a-1e-5<=(s+e)/2<=b+1e-5]
            if not members:failures.append({'id':x['id'],'error':'empty_character','position':pos})
            for j,(s,e,c) in enumerate(members):tokens.append({'text_position':pos,'phone_index':j,'start':s,'end':e,'label':c})
        records.append({'id':x['id'],'sample_id':x['sample_id'],'speaker_id':x['speaker_id'],'arm':x['arm'],'draw':x['draw'],
                        'tokens':tokens,'grid_sha256':sha(p),'grid_path':str(p)})
    groups=defaultdict(list)
    for x in records:groups[x['sample_id']].append(x)
    pairs=[]
    for source,group in groups.items():
        n=next(x for x in group if x['arm']=='N')
        for t in group:
            if t['arm']=='N':continue
            matched,bad=matched_words(n['tokens'],t['tokens'])
            pairs.append({'source':source,'tts_id':t['id'],'matched':len(matched),'unsupported_positions':bad,'natural_phone_count':len(n['tokens'])})
    payload={'status':'PASS_STRUCTURAL_NO_OOV' if not failures and len(records)==1200 else 'HOLD_ALIGNMENT',
       'units':len(records),'sources':len(groups),'speakers':len({x['speaker_id'] for x in records}),
       'failures':failures,'failure_types':dict(Counter(x['error'] for x in failures)),'records':records,'pairs':pairs,
       'corrections':corrections,'database_silence_evidence':dict(evidence),'database_sha256':sha(db),
       'dictionary_sha256':read(RUN/'mfa_normalization_repair.json')['dictionary_sha256'],
       'phone_boundaries_moved':False,'source_waveforms_changed':False,'time_truth_verified':False,
       'development_allowed':False,'confirmation_allowed':False,'script_sha256':sha(__file__)}
    write(RUN/'alignment_final_fit.json',payload)
    print({k:payload[k] for k in ['status','units','sources','speakers','failure_types']},flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
