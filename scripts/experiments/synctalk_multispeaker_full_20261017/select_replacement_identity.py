"""Deterministic source eligibility in pinned official metadata order, before scoring."""
import hashlib,json,re
from pathlib import Path
from urllib.parse import urlparse,parse_qs

POLICY=dict(version='identity_replacement_metadata_order_v1',
    official_hdtf_commit='bbde889bbab954c9a9ebcabe9962885c29f2165e',
    mirror_revision='6f9ed465f761b8378c690be4f86f090634281584',
    metadata_file_order=['WDA','WRA','RD'],
    retained_identities=['SherrodBrown','PatrickLeahy','CathyMcMorrisRodgers'],
    rejected_identities=['DebbieDingell'],excluded_pretrained_identity=['May'],
    identity_order='first occurrence in annotation files; no alphabetic resorting',
    recording_order='first occurrence within identity; first eligible main then first independent eligible test',
    member='first mirror chunk _000 only; no later-chunk scan or concatenated annotation intervals',
    main_minimum_seconds=278,main_maximum_seconds=300,test_minimum_seconds=64,
    material_minimum_seconds=360,material_target_seconds=420,material_maximum_seconds=480,
    supervision_frames=dict(train=6000,val=750,test=1500),test_clips=6,
    failure_policy='preserve complete failure then advance one identity in the frozen metadata order; first fully eligible identity retained',
    condition_selection='no SyncNet/TFG/learning-curve data used; no threshold or budget changes')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seconds(clock):
    value=0
    for part in clock.split(':'):value=value*60+int(part)
    return value


def parse_annotation(text,family):
    rows=[]
    for line_number,line in enumerate(text.splitlines(),1):
        if not line.strip():continue
        key=line.split()[0].removesuffix('.mp4')
        intervals=re.findall(r'(\d{1,3}:\d{2}(?::\d{2})?)-(\d{1,3}:\d{2}(?::\d{2})?)',line)
        assert intervals,(family,line_number,'invalid official annotation')
        spans=[dict(start=seconds(a),end=seconds(b)) for a,b in intervals]
        assert all(s['end']>s['start'] for s in spans)
        identity=re.sub(r'\d+$','',key)
        rows.append(dict(identity=identity,family=family,key=key,line_number=line_number,
                         first_contiguous_seconds=spans[0]['end']-spans[0]['start'],intervals=spans,
                         file=f'{family}_{key}_000.mp4'))
    return rows


def video_id(url):
    parsed=urlparse(url)
    if parsed.netloc.removeprefix('www.')=='youtu.be':return parsed.path.strip('/')
    values=parse_qs(parsed.query).get('v');assert values and len(values)==1,url
    return values[0]


def choose(rows,urls,policy=POLICY):
    groups={}
    for row in rows:groups.setdefault(row['identity'],[]).append(row)
    excluded=set(policy['retained_identities']+policy['rejected_identities']+policy['excluded_pretrained_identity'])
    audit=[];eligible=[]
    for identity,recordings in groups.items():
        item=dict(identity=identity,first_metadata_position=dict(family=recordings[0]['family'],line=recordings[0]['line_number']))
        if identity in excluded:item.update(eligible=False,reason='retained/rejected/pretrained identity');audit.append(item);continue
        pair=None
        for main in recordings:
            if main['first_contiguous_seconds']<policy['main_minimum_seconds']:continue
            main_url=urls.get((main['family'],main['key']))
            if not main_url:continue
            main_seconds=min(main['first_contiguous_seconds'],policy['main_maximum_seconds'])
            for test in recordings:
                test_url=urls.get((test['family'],test['key']))
                if test is main or not test_url or video_id(main_url)==video_id(test_url):continue
                test_seconds=min(test['first_contiguous_seconds'],policy['material_target_seconds']-main_seconds)
                if test_seconds<policy['test_minimum_seconds'] or main_seconds+test_seconds<policy['material_minimum_seconds']:continue
                pair=dict(identity=identity,main={**main,'official_source_url':main_url,'original_video_id':video_id(main_url),'seconds':main_seconds},
                          test={**test,'official_source_url':test_url,'original_video_id':video_id(test_url),'seconds':test_seconds})
                break
            if pair:break
        if pair:
            item.update(eligible=True,pair=pair);eligible.append(pair)
        else:item.update(eligible=False,reason='no ordered pair with first-contiguous main>=278, independent test>=64 and total360-480')
        audit.append(item)
    return eligible,audit


def run(root):
    root=Path(root);out=root/'identity_replacement_policy_v1';out.mkdir(exist_ok=False)
    provenance=json.loads((root/'source_provenance.json').read_text())
    assert provenance['official_commit']==POLICY['official_hdtf_commit']
    pins={r['name']:r['sha256'] for r in provenance['metadata_files']}
    rows=[];urls={};files=[]
    for family in POLICY['metadata_file_order']:
        annotation=root/'provenance'/f'{family}_annotion_time.txt';links=root/'provenance'/f'{family}_video_url.txt'
        for p in [annotation,links]:assert sha(p)==pins[p.name];files.append(dict(name=p.name,sha256=sha(p)))
        rows+=parse_annotation(annotation.read_text(encoding='utf-8-sig'),family)
        for line in links.read_text(encoding='utf-8-sig').splitlines():
            parts=line.split()
            if parts:urls[(family,parts[0].removesuffix('.mp4'))]=parts[1]
    eligible,audit=choose(rows,urls)
    assert eligible and eligible[0]['identity']=='BenCardin',eligible[:1]
    policy={**POLICY,'producer_sha256':sha(Path(__file__)),'metadata_files':files}
    (out/'policy.json').write_text(json.dumps(policy,indent=2))
    (out/'ordered_candidates.json').write_text(json.dumps(dict(policy_sha256=sha(out/'policy.json'),candidates=eligible,audit=audit),indent=2))
    (out/'selected_candidate.json').write_text(json.dumps(dict(policy_sha256=sha(out/'policy.json'),candidate=eligible[0],
        selection_stage='metadata_only_not_media_or_geometry_qualified',TFG_updates=0),indent=2))
    print(json.dumps(dict(selected=eligible[0],candidates=len(eligible),policy_sha256=sha(out/'policy.json'))))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
