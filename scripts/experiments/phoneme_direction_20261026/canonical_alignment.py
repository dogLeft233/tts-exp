"""Normalize proven orthographic/export artifacts without moving phone boundaries."""
ALIASES={'綠':'绿','為':'为'}
SILENCE={'','sil','sp'}


def canonical_words(words,phones,expected,authorized_silence_intervals=()):
    words=list(words);corrections=[]
    if len(words)>=2:
        a,b,label=words[-1];p,q,prior=words[-2]
        overlap=[(s,e,c) for s,e,c in phones if e>a+1e-5 and s<b-1e-5]
        evidence=any(abs(a-s)<1e-4 and abs(b-e)<1e-4 for s,e in authorized_silence_intervals)
        if (evidence and abs(q-a)<1e-5 and ALIASES.get(label,label)==ALIASES.get(prior,prior)
            and overlap and all(c.lower() in SILENCE for s,e,c in overlap)
            and overlap[0][0]<=a+1e-5 and overlap[-1][1]>=b-1e-5):
            candidate=words[:-1]
            if [ALIASES.get(c,c) for s,e,c in candidate]==expected:
                corrections.append({'kind':'duplicate_final_word_over_database_silence','start':a,'end':b,'label':label})
                words=candidate
    result=[]
    for a,b,c in words:
        canonical=ALIASES.get(c,c)
        if canonical!=c:corrections.append({'kind':'orthographic_alias','source':c,'canonical':canonical,'start':a,'end':b})
        result.append((a,b,canonical))
    if [c for a,b,c in result]!=expected:
        raise ValueError('Transcript word sequence still differs after proven canonicalization')
    return result,corrections
