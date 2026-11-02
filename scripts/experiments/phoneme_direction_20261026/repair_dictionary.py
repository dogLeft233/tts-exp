"""Create an immutable fit-only lexicon overlay using existing dictionary evidence."""
from collections import Counter
from pathlib import Path
import os
import shutil
import subprocess
from calibrate import ROOT, RUN as PARENT, MODELS, MFA, read, write, sha, normalize_text

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'
# Explicit character syllables, never inferred from acoustic scores. Each sequence must
# occur in the named existing lexical entry. This avoids selecting an erroneous
# high-probability homophone alternative (some entries have multiple readings).
SPECS={
 '涯':('生涯','j a˧˥'), '撰':('撰写','ʈʂ w a˥˩ n'), '膛':('炉膛','tʰ a˧˥ ŋ'),
 '牺':('牺牲','ɕ i˥'), '牲':('牺牲','ʂ o˥ ŋ'), '溺':('溺水','ɲ i˥˩'),
 '惺':('星','ɕ i˥ ŋ'), '迭':('迭代','t j e˧˥'), '矶':('洛杉矶','tɕ i˥'),
 '奢':('奢侈','ʂ o˥'), '侈':('奢侈','ʈʂʰ ʐ̩˨˩˦'), '苹':('苹果','pʲ i˧˥ ŋ'),
 '碚':('北碚','p ej˥˩'), '琉':('琉璃','l j ow˧˥'), '抚':('抚养','f u˨˩˦'),
 '篷':('帐篷','pʰ o˧˥ ŋ'), '壤':('土壤','ʐ a˨˩˦ ŋ'), '瞻':('沾','ʈʂ a˥ n'),
 '玻':('玻璃','p w o˥'), '璃':('玻璃','ʎ i˧˥'), '乓':('乒乓','pʰ a˥ ŋ'),
 '钊':('招','ʈʂ aw˥'), '涤':('笛','tʲ i˧˥'), '惶':('皇','xʷ a˧˥ ŋ'),
 '嘲':('潮','ʈʂʰ aw˧˥'), '簸':('颠簸','p w o˨˩˦'),
}


def parse_dictionary(text):
    entries={}
    for line in text.splitlines():
        fields=line.split()
        if not fields:continue
        # Source dictionary is MFA3 with four numeric pronunciation/silence fields.
        if len(fields)<6:raise ValueError('Unexpected source lexicon format')
        for value in fields[1:5]:float(value)
        entries.setdefault(fields[0],[]).append((fields[1:5],fields[5:],line))
    return entries


def additions(entries, chars):
    result=[];provenance=[]
    for char in sorted(chars):
        if char in entries:raise ValueError('Never override existing character entry')
        source,pron=SPECS[char];phones=pron.split()
        candidates=[]
        for numeric,p,line in entries[source]:
            if any(p[i:i+len(phones)]==phones for i in range(len(p)-len(phones)+1)):
                candidates.append((float(numeric[0]),numeric,line))
        if not candidates:raise ValueError(f'No source evidence for {char}')
        _,numeric,line=max(candidates,key=lambda x:x[0])
        result.append('\t'.join([char,*numeric,' '.join(phones)]))
        provenance.append({'character':char,'reading':pron,'source_word':source,'source_line':line,
                           'method':'existing word syllable or same-tone homophone; phones and tone unchanged'})
    return result,provenance


def prepare():
    original=MODELS/'dictionary/mandarin_china_mfa.dict';text=original.read_text()
    entries=parse_dictionary(text);m=read(PARENT/'input_manifest.json')
    counts=Counter(c for x in m['fit'] if x['arm']=='N' for c in normalize_text(x['text']).split() if c not in entries)
    if set(counts)!=set(SPECS):raise ValueError('Fit OOV inventory changed; explicit review required')
    lines,provenance=additions(entries,counts)
    out=RUN/'mandarin_china_mfa_fit_overlay.dict'
    with out.open('x',encoding='utf8') as f:f.write(text.rstrip()+'\n'+'\n'.join(lines)+'\n')
    patched=parse_dictionary(out.read_text())
    if any(c not in patched for c in counts):raise ValueError('Overlay incomplete')
    write(RUN/'dictionary_overlay_receipt.json',{'source':str(original),'source_sha256':sha(original),
          'overlay':str(out),'overlay_sha256':sha(out),'fit_source_counts':dict(counts),
          'additions':provenance,'original_modified':False,'transcripts_changed':False,
          'input_manifest_sha256':sha(PARENT/'input_manifest.json'),'script_sha256':sha(__file__),
          'dev_or_confirmation_used':False})


def align(which):
    if which not in ['calibration','fit']:raise ValueError('Fit only')
    manifest=read(PARENT/'input_manifest.json');rows=manifest[which]
    if any(x['split']!='fit' for x in rows):raise ValueError('Fit only')
    receipt=read(RUN/'dictionary_overlay_receipt.json');dictionary=Path(receipt['overlay'])
    if sha(dictionary)!=receipt['overlay_sha256']:raise ValueError('Changed dictionary')
    acoustic=MODELS/'acoustic/mandarin_mfa.zip'
    if sha(acoustic)!=manifest['pins'][str(acoustic)]:raise ValueError('Changed acoustic model')
    corpus=RUN/f'mfa_corpus_{which}'
    for x in rows:
        if sha(x['canonical_audio_path'])!=x['canonical_audio_sha256']:raise ValueError('Changed audio')
        d=corpus/x['speaker_id'];d.mkdir(parents=True,exist_ok=True)
        target=d/(x['id']+'.wav')
        if target.exists():raise FileExistsError(target)
        # A symlink avoids copying the complete corpus; original identity is pinned above.
        target.symlink_to(x['canonical_audio_path'])
        (d/(x['id']+'.lab')).write_text(normalize_text(x['text'])+'\n',encoding='utf8')
    env=dict(os.environ,PATH=str(MFA.parent)+os.pathsep+os.environ.get('PATH',''),
             CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             MFA_ROOT_DIR=str(RUN/f'mfa_runtime_{which}'),TMPDIR=str(RUN/'tmp'))
    (RUN/'tmp').mkdir(exist_ok=True)
    command=[str(MFA),'align',str(corpus),str(dictionary),str(acoustic),str(RUN/f'mfa_grids_{which}'),'--clean','--num_jobs','2']
    write(RUN/f'mfa_launch_{which}.json',{'command':command,'dictionary_sha256':sha(dictionary),'acoustic_sha256':sha(acoustic),
          'rows':len(rows),'source_sha256':sha(__file__),'dev_allowed':False,'confirmation_allowed':False})
    with (RUN/f'mfa_{which}.log').open('x') as f:
        result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=14400)
    write(RUN/f'mfa_exit_{which}.json',{'returncode':result.returncode})
    if result.returncode:raise RuntimeError('MFA failed')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','calibration','fit']);a=p.parse_args()
    prepare() if a.stage=='prepare' else align(a.stage)
