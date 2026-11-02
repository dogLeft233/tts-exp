"""Full revised-fit ASR and MFA readback, with the original technical floors."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import sqlite3
import subprocess

from calibrate import ROOT, MFA, MODELS, read, write, sha, normalize_text
from replace_fit_source import RUN, NEW_ID
from canonical_alignment import canonical_words
from audit_alignment_readback import UNKNOWN, SILENCE, matched_words

OLD_MFA = ROOT / 'runs/phoneme_direction_20261026_repair01'


def align():
    rows = read(RUN / 'input_manifest_v2.json')['fit']
    quality = read(RUN / 'full_reconstruction_quality_v2.json')
    assert quality['status'] == 'PASS_BASIC_RECONSTRUCTION'
    decoded = {x['id']: x for x in quality['rows']}
    dictionary = OLD_MFA / 'mandarin_china_mfa_fit_overlay_revision02.dict'
    plan = []
    for x in rows:
        if x['arm'] == 'N' or x['sample_id'] == NEW_ID:
            plan.append(dict(x, path=x['canonical_audio_path'], digest=x['canonical_audio_sha256'], kind='input'))
        if x['arm'] == 'N':
            d = decoded[x['id']]
            plan.append(dict(x, id=x['id'] + '_R', path=d['output'], digest=d['output_sha256'], kind='reconstruction'))
    assert len(plan) == 603
    write(RUN / 'mfa_full_fit_contract.json', dict(
        units=len(plan), plan=plan, dictionary=str(dictionary), dictionary_sha256=sha(dictionary),
        acoustic_sha256=sha(MODELS / 'acoustic/mandarin_mfa.zip'),
        structural_floors=dict(missing_grids_max=0, unknown_phones_max=0, nonmonotonic_max=0),
        timing_floors=dict(matched_phone_fraction_min=.95, median_abs_center_ms_max=20, p95_abs_center_ms_max=40),
        reuse='897 unchanged TTS grids with original SHA and database-bound normalization; newly align all 300 A/R pairs plus 3 replacement TTS',
        boundaries_are_automatic=True, development_allowed=False, confirmation_allowed=False))
    corpus = RUN / 'mfa_corpus_v2'
    for x in plan:
        assert sha(x['path']) == x['digest'] and x['split'] == 'fit'
        d = corpus / x['speaker_id']; d.mkdir(parents=True, exist_ok=True)
        (d / (x['id'] + '.wav')).symlink_to(x['path'])
        (d / (x['id'] + '.lab')).write_text(normalize_text(x['text']) + '\n')
    (RUN / 'tmp').mkdir(exist_ok=True)
    env = dict(os.environ, PATH=str(MFA.parent)+os.pathsep+os.environ.get('PATH',''),
        CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        MFA_ROOT_DIR=str(RUN / 'mfa_runtime_v2'), TMPDIR=str(RUN / 'tmp'))
    cmd = [str(MFA), 'align', str(corpus), str(dictionary), str(MODELS / 'acoustic/mandarin_mfa.zip'),
           str(RUN / 'mfa_grids_v2'), '--clean', '--num_jobs', '2']
    write(RUN / 'mfa_launch.json', dict(command=cmd, script_sha256=sha(__file__)))
    with (RUN / 'mfa.log').open('x') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
    write(RUN / 'mfa_exit.json', dict(returncode=result.returncode))
    if result.returncode: raise RuntimeError('MFA failed; keep log and original attempt')


def silence_evidence(db):
    connection = sqlite3.connect('file:' + str(db) + '?mode=ro', uri=True)
    result = defaultdict(list)
    sql = "select f.name,pi.begin,pi.end from phone_interval pi join phone p on p.id=pi.phone_id join word_interval wi on wi.id=pi.word_interval_id join word w on w.id=wi.word_id join utterance u on u.id=pi.utterance_id join file f on f.id=u.file_id where p.phone='sil' and w.word not in ('<eps>','<unk>')"
    for name, start, end in connection.execute(sql): result[name].append((start, end))
    connection.close()
    return result


def audit():
    import numpy as np
    from praatio import textgrid
    manifest = read(RUN / 'input_manifest_v2.json')['fit']
    old = read(OLD_MFA / 'alignment_final_fit.json')
    old_by = {x['id']: x for x in old['records']}
    db = RUN / 'mfa_runtime_v2/mfa_corpus_v2/mfa_corpus_v2.db'
    old_db = OLD_MFA / 'mfa_runtime_fit_revision02/mfa_corpus_fit/mfa_corpus_fit.db'
    assert sha(old_db) == old['database_sha256']
    fresh_evidence = silence_evidence(db); original_evidence = silence_evidence(old_db)
    failures = []; corrections = []; records = []; roundtrip = {}

    def parse(x, reconstruction=False):
        name = x['id'] + ('_R' if reconstruction else '')
        fresh = reconstruction or x['arm'] == 'N' or x['sample_id'] == NEW_ID
        p = RUN / 'mfa_grids_v2' / x['speaker_id'] / (name + '.TextGrid') if fresh else Path(old_by[name]['grid_path'])
        if not fresh: assert sha(p) == old_by[name]['grid_sha256']
        if not p.exists():
            failures.append(dict(id=name, error='missing_grid')); return []
        g = textgrid.openTextgrid(str(p), includeEmptyIntervals=True)
        phones = list(g.getTier('phones').entries)
        words = [(a,b,c) for a,b,c in g.getTier('words').entries if c not in SILENCE]
        try:
            words, changes = canonical_words(words, phones, normalize_text(x['text']).split(),
                (fresh_evidence if fresh else original_evidence)[name])
            corrections.extend(dict(y, id=name) for y in changes)
        except ValueError:
            failures.append(dict(id=name, error='word_sequence'))
        if any(c.lower() in UNKNOWN for a,b,c in phones): failures.append(dict(id=name,error='unknown_phone'))
        if any(b<=a or a<0 or b>x['samples']/16000+.001 for a,b,c in phones): failures.append(dict(id=name,error='boundary'))
        if any(a[1]>b[0]+1e-6 for a,b in zip(phones,phones[1:])): failures.append(dict(id=name,error='nonmonotonic'))
        tokens = []
        for pos,(a,b,c) in enumerate(words):
            members = [(s,e,label) for s,e,label in phones if label not in SILENCE and a-1e-5 <= (s+e)/2 <= b+1e-5]
            if not members: failures.append(dict(id=name,error='empty_character',position=pos))
            tokens.extend(dict(text_position=pos,phone_index=j,start=s,end=e,label=label) for j,(s,e,label) in enumerate(members))
        if not reconstruction:
            records.append(dict(id=name,sample_id=x['sample_id'],speaker_id=x['speaker_id'],arm=x['arm'],
                draw=x['draw'],tokens=tokens,grid_path=str(p),grid_sha256=sha(p),fresh_alignment=fresh))
        return tokens

    for x in manifest:
        assert sha(x['canonical_audio_path']) == x['canonical_audio_sha256']
        tokens = parse(x)
        if x['arm'] == 'N': roundtrip[x['id']] = (tokens, parse(x, True))
    groups = defaultdict(list)
    for x in records: groups[x['sample_id']].append(x)
    pairs = []
    for sid, group in groups.items():
        natural = next(x for x in group if x['arm']=='N')
        for t in group:
            if t['arm']=='N': continue
            matched, bad = matched_words(natural['tokens'], t['tokens'])
            pairs.append(dict(source=sid,tts_id=t['id'],matched=len(matched),unsupported_positions=bad))
    structural_failures = [x for x in failures if not x['id'].endswith('_R')]
    write(RUN / 'alignment_full_fit_v2.json', dict(status='PASS_STRUCTURAL_NO_OOV' if not structural_failures and len(records)==1200 else 'HOLD_ALIGNMENT',
        units=len(records),sources=len(groups),speakers=len({x['speaker_id'] for x in records}),records=records,
        pairs=pairs,failures=structural_failures,corrections=corrections,database_sha256=sha(db),
        reused_tts_grids=897,fresh_input_grids=303,time_truth_verified=False))
    differences = []; timing_rows = []; total = 0
    for sid,(a,b) in roundtrip.items():
        matched,bad = matched_words(a,b); total += len(a)
        ds = [1000*((y['start']+y['end'])-(x['start']+x['end']))/2 for x,y in matched]
        differences.extend(dict(id=sid,center_shift_ms=d) for d in ds)
        timing_rows.append(dict(id=sid,phones=len(a),matched=len(ds),unsupported_positions=bad,
            median_abs_center_ms=float(np.median(np.abs(ds))) if ds else None))
    shifts = np.abs([x['center_shift_ms'] for x in differences])
    fraction = len(shifts)/total; median=float(np.median(shifts)); p95=float(np.percentile(shifts,95))
    passed = not failures and fraction>=.95 and median<=20 and p95<=40
    write(RUN / 'roundtrip_timing_full_fit_v2.json',dict(status='PASS_AUTOMATIC_TIMING' if passed else 'HOLD_TIMING',
        matched_phone_fraction=fraction,median_abs_center_ms=median,p95_abs_center_ms=p95,
        rows=timing_rows,phone_differences=differences,failures=failures,
        limits='Same forced-aligner consistency, not independent phoneme boundary truth',
        development_allowed=False,confirmation_allowed=False))
    print('ALIGNMENT',len(records),'failures',Counter(x['error'] for x in failures),flush=True)
    print('TIMING',fraction,median,p95,'passed',passed,flush=True)


def asr(shard):
    import random
    import hashlib
    import numpy as np
    import torch
    from funasr import AutoModel
    from check_reconstruction_content import MODEL, norm, distance
    if shard not in range(3): raise ValueError('Three fixed shards')
    rows = [x for x in read(RUN / 'input_manifest_v2.json')['fit'] if x['arm']=='N']
    decoded = {x['id']:x for x in read(RUN / 'full_reconstruction_quality_v2.json')['rows']}
    write(RUN / f'asr_shard{shard}_contract.json', dict(ids=[x['id'] for x in rows[shard::3]],
        model_files_sha256={p.name:sha(p) for p in MODEL.iterdir() if p.is_file()},
        floors=dict(additional_corpus_cer_max=.02,additional_edits_per_source_max=2),
        same_seed_across_A_R=True,development_allowed=False,confirmation_allowed=False))
    torch.set_num_threads(2)
    model = AutoModel(model=str(MODEL),device='cpu',disable_update=True,disable_pbar=True)
    result = []
    subset = rows[shard::3]
    # New TTS content is also checked, outside the paired reconstruction endpoint.
    if shard==0: subset += [x for x in read(RUN / 'input_manifest_v2.json')['replacement'] if x['arm']=='L17']
    for x in subset:
        seed=int(hashlib.sha256(x['id'].encode()).hexdigest()[:8],16)
        paths=[('A',x['canonical_audio_path'],x['canonical_audio_sha256'])]
        if x['arm']=='N': paths.append(('R',decoded[x['id']]['output'],decoded[x['id']]['output_sha256']))
        for kind,path,digest in paths:
            assert sha(path)==digest
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            prediction=model.generate(input=path,cache={},batch_size_s=300,disable_pbar=True)[0]['text']
            record=dict(id=x['id'],arm=x['arm'],kind=kind,text=prediction,reference=x['text'],
                reference_characters=len(norm(x['text'])),edit_errors=distance(norm(x['text']),norm(prediction)),audio_sha256=digest)
            write(RUN / 'asr_items' / (x['id']+'_'+kind+'.json'),record);result.append(record)
        print('ASR_SOURCE',x['id'],flush=True)
    write(RUN / f'asr_shard{shard}_complete.json',dict(rows=result))


def asr_audit():
    rows=[]
    for i in range(3): rows.extend(read(RUN / f'asr_shard{i}_complete.json')['rows'])
    groups=defaultdict(dict); tts=[]
    for x in rows:
        if x['arm']=='N': groups[x['id']][x['kind']]=x
        else: tts.append(x)
    assert len(groups)==300 and all(set(g)=={'A','R'} for g in groups.values()) and len(tts)==3
    chars=sum(g['A']['reference_characters'] for g in groups.values())
    errors={k:sum(g[k]['edit_errors'] for g in groups.values()) for k in ['A','R']}
    diffs=[dict(id=sid,additional_edits=g['R']['edit_errors']-g['A']['edit_errors']) for sid,g in groups.items()]
    passed=(errors['R']-errors['A'])/chars<=.02 and max(x['additional_edits'] for x in diffs)<=2
    write(RUN / 'asr_full_fit_v2.json',dict(status='PASS_PAIRED_ASR' if passed else 'HOLD_ASR',
        sources=300,characters=chars,errors=errors,cer={k:v/chars for k,v in errors.items()},
        paired_differences=diffs,rows=rows,replacement_tts=tts,
        not_manual_ground_truth=True,development_allowed=False,confirmation_allowed=False))
    print('ASR',passed,chars,errors,'max_added',max(x['additional_edits'] for x in diffs),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['align','audit','asr','asr_audit']);parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    if args.stage=='asr': asr(args.shard)
    else: globals()[args.stage]()
