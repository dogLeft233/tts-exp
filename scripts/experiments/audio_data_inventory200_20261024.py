"""CPU-only, score-blind inventory. Reads sources; writes only audit JSON in --output.

No model imports, downloads, extraction to disk, generation, or training. Archive
members are streamed to inspect WAV headers and hash existing source bytes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import datetime
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import unicodedata
import wave

RUN = 'audio_data_inventory200_20261024_attempt01'
OLD = 'qwen_nrt_official_rescore_20261023_attempt01'
PROBE = 'syncnet_preference_probe_20261024_attempt01'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def normalize_text(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if c.isalnum())

def near_duplicate(a, b):
    a, b = normalize_text(a), normalize_text(b)
    return bool(a and b) and (a == b or (min(len(a),len(b)) >= 8 and difflib.SequenceMatcher(None,a,b).ratio() >= .90))

def plan_speakers(packages, old_speakers, excluded_speakers, seed='20261024'):
    """Provisional metadata-only 100/40/60 speaker plan; no outcome input.

    Previously sealed/test/uncertain identities remain excluded even if a
    prior inventory read their headers. Only explicitly required old40 may
    enter fitting. Source IDs/length/content and split leakage still gate the
    final freeze. Public ASR dev/test speakers are conservatively avoided too.
    """
    by={x['speaker_id']:dict(x) for x in packages}
    old=set(old_speakers);excluded=set(excluded_speakers)
    if len(old)!=40 or not old<=by.keys():raise ValueError('exact old40 required')
    picked=[dict(by[sp],split='fit',selection='explicit_required_old40') for sp in sorted(old)]
    used=set(old)
    for split,total in [('fit',100),('development',40),('confirmation',60)]:
        pools={g:[x for sp,x in by.items() if sp not in used|excluded and x.get('dataset_split')=='train' and x.get('gender')==g] for g in ['M','F']}
        needs={g:total//2-sum(x['split']==split and x['gender']==g for x in picked) for g in ['M','F']}
        if min(needs.values())<0:raise ValueError('required old group exceeds gender target')
        # Gender balance is a coverage goal; never admit a legacy/sealed
        # speaker to force 50/50 when the genuinely eligible pool is uneven.
        for g,other in [('M','F'),('F','M')]:
            short=max(0,needs[g]-len(pools[g]));needs[g]-=short;needs[other]+=short
        if any(needs[g]>len(pools[g]) for g in ['M','F']):raise ValueError('insufficient eligible speaker pool')
        for gender in ['M','F']:
            needed=needs[gender];pool=pools[gender]
            pool.sort(key=lambda x:hashlib.sha256((seed+'|'+split+'|'+x['speaker_id']).encode()).hexdigest())
            if len(pool)<needed:raise ValueError(f'insufficient {split} {gender}: {len(pool)} < {needed}')
            for x in pool[:needed]:picked.append(dict(x,split=split,selection='new_metadata_hash_order'));used.add(x['speaker_id'])
    assert len(picked)==len(used)==200
    assert not (used-old)&excluded
    return picked

def wave_meta(raw):
    with wave.open(io.BytesIO(raw)) as w:
        return dict(duration_s=w.getnframes()/w.getframerate(), sample_rate=w.getframerate(), channels=w.getnchannels(), sample_width=w.getsampwidth(), frames=w.getnframes())

def source_row(raw, sid, location, text, exposure):
    row = dict(source_id=sid, speaker_id=re.search(r'S\d{4}',sid).group(), location=location,
               text=text, normalized_text=normalize_text(text), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), exposure=exposure)
    try: row.update(wave_meta(raw))
    except (wave.Error,EOFError) as e: row['header_error']=str(e)
    row['exclusions'] = (["unreadable_wav_header"] if 'header_error' in row else
        (["duration_outside_3_12"] if not 3 <= row['duration_s'] <= 12 else []))
    if not row['normalized_text']: row['exclusions'].append('missing_transcript')
    if row.get('channels') != 1 or row.get('sample_rate') != 16000: row['exclusions'].append('unexpected_audio_format')
    row['eligible_header_text'] = not row['exclusions']
    row['content_listening_and_phonetic_coverage'] = 'NOT_MEASURED'
    row['gender'] = None
    return row

def asset(path, expected=None):
    p=Path(path); row={'path':str(p),'exists':p.is_file(),'expected_sha256':expected}
    if row['exists']:
        row.update(bytes=p.stat().st_size,sha256=digest(p));row['sha_match']=expected is None or expected==row['sha256']
    else: row['sha_match']=False
    return row

def tree_bytes(path):
    p=Path(path)
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0

def command(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'args':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e: return {'args':args,'error':str(e)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cloud-root',type=Path);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();r=a.root.resolve();out=a.output.resolve()
    if out.exists(): raise SystemExit('create-only output already exists')
    if out.parent != (r/'runs').resolve() or out.name != RUN: raise SystemExit('output outside owned audit directory')
    transcript=r/'data/native_mechanism_holdout_20260926/provenance/aishell_transcript_v0.8.txt'
    texts={}
    for line in transcript.read_text().splitlines():
        s=line.split(maxsplit=1)
        if len(s)==2:texts[s[0]]=s[1]
    old=json.loads((r/'runs'/OLD/'input_manifest.json').read_text())['records']
    probe=json.loads((r/'runs'/PROBE/'input_manifest.json').read_text())['rows']
    old_speakers=sorted({x['speaker_id'] for x in old})
    probe_sources={x['sample_id'] for x in probe}
    source_rows={};archives=[];loose=[]
    archive_tree=r/'data/native_mechanism_holdout_20260926/provenance/hf_archive_tree.json'
    archive_pins={Path(x['path']).name:x.get('lfs',{}).get('oid') for x in json.loads(archive_tree.read_text()) if isinstance(x,dict) and x.get('path')}
    for name in ['native_mechanism_holdout_20260926','fresh_aishell_disjoint_ref_target_20260929']:
        base=r/'data'/name
        for p in sorted((base/'archives').glob('S*.tar.gz')):
            rec=asset(p);rec['speaker_id']=p.name.split('.')[0]
            provenance=base/'provenance'/f"{rec['speaker_id']}_archive.json"
            if provenance.exists():
                expected=json.loads(provenance.read_text());rec['expected_sha256']=expected.get('sha256');rec['sha_match']=rec['sha256']==rec['expected_sha256']
            else:
                rec['expected_sha256']=archive_pins.get(p.name)
                rec['sha_match']=bool(rec['expected_sha256']) and rec['expected_sha256']==rec['sha256']
                rec['provenance_pin']=str(archive_tree)
            rec['wave_members']=0;rec['eligible_members']=0
            # Read all members once; getmembers+extractfile repeatedly would
            # decompress gzip many times. No archive paths are written to disk.
            with tarfile.open(p,mode='r|gz') as t:
                for member in t:
                    if not member.isfile() or not member.name.endswith('.wav'):continue
                    sid=Path(member.name).stem
                    if not re.fullmatch(r'BAC009S\d{4}W\d{4}',sid):continue
                    raw=t.extractfile(member).read();rec['wave_members']+=1
                    row=source_row(raw,sid,{'archive':str(p),'member':member.name},texts.get(sid,''),'historical_roster_fit_only' if rec['speaker_id'] in old_speakers else 'historical_fresh_ref_target_exposure_unknown')
                    rec['eligible_members']+=row['eligible_header_text'];source_rows[sid]=row
            archives.append(rec)
        for p in sorted((base/'audio').glob('*.wav')):
            raw=p.read_bytes();sid=p.stem
            row=source_row(raw,sid,{'path':str(p)},texts.get(sid,''),'historically_selected_fit_only')
            loose.append(row)
            if sid not in source_rows:source_rows[sid]=row
            else:source_rows[sid]['loose_path']=str(p)
    if a.cloud_root:
        mirror=a.cloud_root/'tts-exp-project/datasets/aishell_future_dev20_mirror_attempt01'
        for p in sorted(mirror.glob('train/S*/*.wav')):
            sid=p.stem
            row=source_row(p.read_bytes(),sid,{'path':str(p),'storage':'CloudFS'},texts.get(sid,''),'historical_future_dev_mirror_exposure_unknown')
            source_rows.setdefault(sid,row)
    # Historical test corpora: use project-relative current locations rather
    # than executing frozen /mnt/e or retired local absolute paths.
    historical=[]
    for manifest, audio_key, id_key, base, filename_key in [
        ('data/aishell1_100_zh/manifest.json','audio_path','source_utterance_id',None,None),
        ('results/rhythm_style_500/source_manifests/aishell1_test_400_natural.json','natural_audio_path','source_utterance_id','results/rhythm_style_500/aishell1_test_400/natural','expansion_id')]:
        p=r/manifest
        if not p.exists():continue
        for x in json.loads(p.read_text())['records']:
            q=r/x[audio_key] if base is None else r/base/f"{x[filename_key]:04d}.wav"
            rec={'source_id':x[id_key],'speaker_id':x['speaker_id'],'current_path':str(q),'exists':q.is_file()};historical.append(rec)
            if q.is_file():
                row=source_row(q.read_bytes(),x[id_key],{'path':str(q)},x.get('transcript',''),'historical_analysed_fit_only')
                source_rows.setdefault(x[id_key],row)
    reuse={}
    for x in old:
        if x['role']=='R':continue
        arm='N' if x['role']=='N' else 'L06';draw=0 if arm=='N' else int(x['role'][1:]);key=(x['sample_id'],arm,draw)
        reuse[key]={'source_id':key[0],'speaker_id':x['speaker_id'],'arm':arm,'draw':draw,'assets':[asset(x['canonical_audio_path'],x['canonical_audio_sha256']),asset(x['video_path'],x['source_video_sha256'])]}
    for x in probe:
        key=(x['sample_id'],x['arm'],x['draw'])
        if key not in reuse:
            reuse[key]={'source_id':key[0],'speaker_id':x['speaker_id'],'arm':x['arm'],'draw':x['draw'],'assets':[]}
        rec=reuse[key]
        known={y['path'] for y in rec['assets']}
        for kind in ['audio','video','score','activesd']:
            p=x.get(kind+'_path')
            if p and p not in known:rec['assets'].append(asset(p,x.get(kind+'_sha256')))
        rec['prior_qualified']=x.get('qualified')
    for rec in reuse.values(): rec['all_listed_assets_match']=all(x['sha_match'] for x in rec['assets'])
    missing=[{'source_id':x['sample_id'],'speaker_id':x['speaker_id'],'arm':'L17','draw':draw} for x in old if x['role']=='N' and x['sample_id'] not in probe_sources for draw in [1,2,3]]
    by=defaultdict(list)
    for row in source_rows.values():
        if row['eligible_header_text']:by[row['speaker_id']].append(row)
    # Three distinct/near-distinct texts within each speaker, deterministically
    # ordered; cross-speaker split is not frozen while exposure is unresolved.
    candidates=[]
    for sp,rows in sorted(by.items()):
        picked=[]
        old_ids={x['sample_id'] for x in old if x['speaker_id']==sp}
        rows=sorted(rows,key=lambda x:(x['source_id'] not in old_ids, hashlib.sha256(('20261024|'+x['source_id']).encode()).hexdigest()))
        for row in rows:
            if not any(near_duplicate(row['text'],v['text']) for v in picked):picked.append(row)
            if len(picked)==3:break
        candidates.append({'speaker_id':sp,'eligible_utterances':len(rows),'has_three_distinct':len(picked)==3,'source_ids':[x['source_id'] for x in picked], 'proposed_split':'fit' if sp in old_speakers else 'UNASSIGNED_EXPOSURE_AUDIT_REQUIRED'})
    units=[]
    for x in old:
        if x['role']=='R':continue
        data=r/'runs'/OLD/'data';ref=x['reference']; parts={name:tree_bytes(data/name/ref) for name in ['pyavi','pyframes','pycrop','pywork','pytmp']}
        units.append({'source_id':x['sample_id'],'role':x['role'],'seconds':x['audio_samples']/16000,'audio_bytes':Path(x['canonical_audio_path']).stat().st_size,'video_bytes':Path(x['video_path']).stat().st_size,'mkv_bytes':Path(x['input_mkv']).stat().st_size,'official_parts':parts,'official_bytes':sum(parts.values()),'float32_512_audio_bytes_estimate':max(0,int(x['frame_count'])-5)*512*4})
    # Means preserve observed TTS tails; 12s bound only applies to natural
    # selection and cannot cap generated duration without a frozen EOS gate.
    mean=lambda k:sum(x[k] for x in units)/len(units)
    per=mean('audio_bytes')*4+mean('video_bytes')+mean('mkv_bytes')+mean('official_bytes')+mean('float32_512_audio_bytes_estimate')*2
    full=int(per*4200);reserve=5*(1<<30);active=shutil.disk_usage(r)
    capacity={'unit_measurements':units,'mean_retained_bytes_per_unit_conservative':per,'retained4200_bytes_estimate':full,'safety_margin_fraction':.25,'safety_reserve_bytes':reserve,'temporary_peak_one_unit_bytes_estimate':int(max(x['official_bytes']+x['mkv_bytes']+x['video_bytes'] for x in units)*2),'full_with25pct_and5GiB_bytes':int(full*1.25)+reserve,'active_free_bytes':active.free,'active_total_bytes':active.total,'full_retention_active_gate':'GO' if active.free>=int(full*1.25)+reserve else 'HOLD_FULL_RETENTION','batching':'Possible only after verified archive readback and a separate authorized retirement policy; no historical files removed by inventory.'}
    capacity['boundaries']=['Means are empirical old160 N/L06 and not an upper bound for new L17 durations or face crop multiplicity.','512 estimate is float32 full sequence audio+video only; comprehensive cache measurements supplied separately by technical audit.','Archive staging must be streamed with bounded local scratch; never duplicate all4200 on E or active root.']
    storage={'active':capacity,'mounts':[command(['df','-T',str(r)]+([str(a.cloud_root)] if a.cloud_root else [])),command(['findmnt','-T',str(r)]),command(['quota','-s'])]}
    if a.cloud_root:storage['cloud_free_bytes']=shutil.disk_usage(a.cloud_root).free;storage['cloud_boundary']='FUSE/cloud apparent quota; no active throughput or provider durability guarantee'
    inv={'schema':'audio_data_inventory200_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU_ONLY_READ_ONLY_SOURCES_NO_GENERATION_NO_TRAINING_NO_SCORE_SELECTION','root':str(r),'transcript':asset(transcript),'old_speakers':old_speakers,'old_source_count':len({x['sample_id'] for x in old}),'probe_sources_subset_old':probe_sources <= {x['sample_id'] for x in old},'archive_count':len(archives),'archive_speaker_count':len({x['speaker_id'] for x in archives}),'archives':archives,'loose_source_count':len(loose),'historical_manifest_assets':historical,'source_count':len(source_rows),'speaker_count':len({x['speaker_id'] for x in source_rows.values()}),'eligible_source_count':sum(x['eligible_header_text'] for x in source_rows.values()),'eligible_speakers_three_distinct':sum(x['has_three_distinct'] for x in candidates),'exclusions':dict(Counter(z for x in source_rows.values() for z in x['exclusions'])),'reuse_candidate_count':len(reuse),'reuse_asset_sha_pass_count':sum(x['all_listed_assets_match'] for x in reuse.values()),'missing_L17_old40_count':len(missing),'data_gate':'HOLD_200_NOT_AVAILABLE_OR_EXPOSURE_UNRESOLVED','split_freeze':'NOT_FROZEN: insufficient audited 200 speaker pool, gender metadata absent, historic exposure beyond old40 unresolved, cross-speaker near duplicate connected groups not yet allocated','selection_rule':'natural 3..12s mono16k, transcript nonempty, SHA; deterministic hash order seed20261024; preserve old source first; within-speaker normalized exact/SequenceMatcher>=0.90 text exclusion; no score values used','target':{'speakers':200,'sources':600,'units':4200,'fit_speakers':100,'development_speakers':40,'sealed_confirm_speakers':60},'limitations':['header+text eligibility is not listening/content/FA qualification','gender/rate/pause/phonetic strata still require metadata and natural-only measurement','new confirmation means unseen by this development only; upstream training overlap unknown','no scores from future confirmation read','all old40 fit only; other historic corpora conservatively unassigned/fit-only','public400 count is not local availability'], 'storage':storage}
    out.mkdir()
    for name,data in [('inventory.json',inv),('source_manifest.json',{'rows':list(source_rows.values())}),('selection_candidates.json',{'status':inv['split_freeze'],'speakers':candidates}),('reuse_manifest.json',{'rows':list(reuse.values()),'missing_L17':missing})]:
        with (out/name).open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:inv[k] for k in ['source_count','speaker_count','eligible_source_count','eligible_speakers_three_distinct','reuse_candidate_count','reuse_asset_sha_pass_count','missing_L17_old40_count','data_gate']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
