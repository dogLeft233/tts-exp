"""Independent TextGrid readback; unknown speech is never a matched phoneme."""
from collections import defaultdict, Counter
from calibrate import RUN, read, sha, write

UNKNOWN={'spn','unk','<unk>'}
SILENCE={'','sil','sp'}


def matched_words(natural, target):
    """Require exact phone sequence within the same transcript position."""
    def group(tokens):
        words=defaultdict(list)
        for t in tokens:words[t['text_position']].append(t)
        return words
    n,t=group(natural),group(target)
    matched=[];unsupported=[]
    for pos in sorted(set(n)|set(t)):
        a=n.get(pos,[]);b=t.get(pos,[])
        al=[x['label'] for x in a];bl=[x['label'] for x in b]
        if not a or al!=bl or any(s.lower() in UNKNOWN|SILENCE for s in al):
            unsupported.append(pos)
        else:matched.extend((x,y) for x,y in zip(a,b))
    return matched,unsupported


def main():
    from praatio import textgrid
    initial=read(RUN/'alignment_audit.json')
    manifest=read(RUN/'input_manifest.json')
    rows=[];unknown=[]
    for x in manifest['calibration']:
        path=RUN/'mfa_grids'/x['speaker_id']/(x['id']+'.TextGrid')
        grid=textgrid.openTextgrid(str(path),includeEmptyIntervals=True)
        words=[(a,b,c) for a,b,c in grid.getTier('words').entries if c and c not in SILENCE]
        phones=list(grid.getTier('phones').entries)
        tokens=[]
        for pos,(a,b,char) in enumerate(words):
            members=[(p,q,s) for p,q,s in phones if s not in SILENCE and a-1e-5 <= (p+q)/2 <= b+1e-5]
            for j,(p,q,s) in enumerate(members):
                token={'text_position':pos,'phone_index':j,'start':p,'end':q,'label':s}
                tokens.append(token)
                if s.lower() in UNKNOWN:
                    unknown.append({'id':x['id'],'text_position':pos,'character':char,'phone':s,'start':p,'end':q})
        before=next(y for y in initial['records'] if y['id']==x['id'])
        if tokens!=before['tokens'] or sha(path)!=before['grid_sha256']:
            raise ValueError('TextGrid readback differs')
        rows.append(dict(x,tokens=tokens))
    groups=defaultdict(list)
    for x in rows:groups[x['sample_id']].append(x)
    pairs=[]
    for source,group in groups.items():
        n=next(x for x in group if x['arm']=='N')
        for t in group:
            if t['arm']=='N':continue
            matched,unsupported=matched_words(n['tokens'],t['tokens'])
            pairs.append({'source':source,'tts_id':t['id'],'valid_matched_phones':len(matched),
                          'unsupported_text_positions':unsupported,'natural_phone_count_including_unknown':len(n['tokens'])})
    write(RUN/'alignment_readback.json',{'status':'VERIFIED_ALIGNMENT_HOLD' if unknown else 'VERIFIED_STRUCTURAL_ONLY',
         'units':len(rows),'unknown':unknown,'unknown_characters':dict(Counter(x['character'] for x in unknown)),
         'pairs':pairs,'valid_matched_phones':sum(x['valid_matched_phones'] for x in pairs),
         'initial_pair_count_included_unknown':True,'pair_policy':'Same text position, exact within-word phone sequence, reject unknown/silence; tone labels untouched',
         'time_truth_verified':False,'numeric_protocol_frozen':False,'development_allowed':False,
         'confirmation_allowed':False,'code_sha256':sha(__file__)})
    print({'units':len(rows),'unknown_characters':dict(Counter(x['character'] for x in unknown)),
           'valid_pairs':sum(x['valid_matched_phones'] for x in pairs)})


if __name__=='__main__':main()
