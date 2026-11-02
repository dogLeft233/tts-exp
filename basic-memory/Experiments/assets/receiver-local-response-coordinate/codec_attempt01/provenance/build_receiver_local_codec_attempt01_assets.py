from __future__ import annotations
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import shutil

PROJECT=Path(r'E:\Documents\tts-exp')
SCRATCH=PROJECT/'.codex'/'scratch'
PROBE=SCRATCH/'native_attribute_response_probe_20261015_attempt01'
CODEC=SCRATCH/'native_attribute_response_probe_20261015_codec_diagnostic_attempt01'
COMPACT=CODEC/'compact_readback_attempt01'
REMOTE_READBACK=COMPACT/'remote_readback_attempt01'
LOCAL_MANIFEST=COMPACT/'local_readback_manifest.json'
QA_PATH=CODEC/'root_codec_matrix_qa.json'
DEST=PROJECT/'basic-memory'/'Experiments'/'assets'/'receiver-local-response-coordinate'/'codec_attempt01'
SELF=Path(__file__).resolve()

if DEST.exists():
    raise SystemExit(f'create-only refusal: destination already exists: {DEST}')

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def load_json(path:Path):
    return json.loads(path.read_text(encoding='utf-8'))

local=load_json(LOCAL_MANIFEST)
if local.get('status')!='PASS' or local.get('selected_artifact_count')!=54 or int(local.get('selected_artifact_bytes',-1))!=260964:
    raise SystemExit('local readback manifest is not the expected PASS/54/260964 artifact set')
qa=load_json(QA_PATH)
if qa.get('status')!='PASS_DIAGNOSTIC_ONLY' or len(qa.get('rows',[]))!=6 or len(qa.get('pairs',[]))!=4:
    raise SystemExit('ROOT codec QA does not match frozen six-row/four-pair scope')
protocol_path=CODEC/'codec_diagnostic_protocol_attempt01.json'
if load_json(protocol_path).get('status')!='FROZEN_SIX_SCORES_ONLY':
    raise SystemExit('codec protocol is not the frozen attempt01 protocol')
waveform_design=PROBE/'waveform_reachability_source_design_attempt01.md'
waveform_pins=PROBE/'waveform_reachability_source_pins_attempt01.json'
if not waveform_design.is_file() or load_json(waveform_pins).get('status')!='SOURCE_ONLY_NOT_EXECUTION_APPROVAL':
    raise SystemExit('waveform reachability material is missing or status changed')
event_result=PROBE/'original_event_alignment_diagnostic_attempt01.json'
event_receipt=PROBE/'original_event_alignment_diagnostic_attempt01.execution_receipt.json'
event_source=PROBE/'original_event_alignment_diagnostic_attempt01.py'
event_json=load_json(event_result)
event_exec=load_json(event_receipt)
if event_json.get('status')!='ORIGINAL_EVENT_ALIGNMENT_DIAGNOSTIC_ONLY' or event_exec.get('status')!='COMPLETED' or event_exec.get('process',{}).get('return_code')!=0:
    raise SystemExit('original event lag0 diagnostic is not a completed read-only extraction')

selected=[]
for item in local['selected_artifact_hashes']:
    rel=Path(*item['path'].split('/'))
    src=REMOTE_READBACK/rel
    if not src.is_file():
        raise SystemExit(f'missing selected compact artifact: {src}')
    if src.stat().st_size!=item['bytes'] or sha(src)!=item['local_sha256'] or item['local_sha256']!=item['expected_sha256'] or item['match'] is not True:
        raise SystemExit(f'selected compact artifact pin mismatch: {src}')
    if src.suffix.lower() in {'.mp4','.mkv','.mov','.wav','.png','.jpg','.jpeg','.webp','.avi'}:
        raise SystemExit(f'media body is outside compact selection: {src}')
    selected.append((src,'readback54/'+rel.as_posix()))
if len(selected)!=54 or sum(src.stat().st_size for src,_ in selected)!=260964:
    raise SystemExit('selected readback 54 file guard mismatch')

copy_items=[
    (CODEC/'codec_diagnostic_protocol_attempt01.json','protocol/codec_diagnostic_protocol_attempt01.json'),
    (CODEC/'codec_diagnostic_attempt01.py','source/codec_diagnostic_attempt01.py'),
    (CODEC/'root_codec_matrix_qa.py','qa/root_codec_matrix_qa.py'),
    (QA_PATH,'qa/root_codec_matrix_qa.json'),
    (PROBE/'waveform_reachability_source_design_attempt01.md','design/waveform_reachability_source_design_attempt01.md'),
    (PROBE/'waveform_reachability_source_pins_attempt01.json','design/waveform_reachability_source_pins_attempt01.json'),
    (PROBE/'original_event_alignment_diagnostic_attempt01.py','event_lag0/original_event_alignment_diagnostic_attempt01.py'),
    (PROBE/'original_event_alignment_diagnostic_attempt01.json','event_lag0/original_event_alignment_diagnostic_attempt01.json'),
    (PROBE/'original_event_alignment_diagnostic_attempt01.execution_receipt.json','event_lag0/original_event_alignment_diagnostic_attempt01.execution_receipt.json'),
    (LOCAL_MANIFEST,'provenance/local_readback_manifest.json'),
    (COMPACT/'remote_readback_attempt01'/'remote_readback_manifest.json','provenance/remote_readback_manifest.json'),
    (COMPACT/'provenance'/'preflight_attempt01.json','provenance/preflight_attempt01.json'),
    (COMPACT/'provenance'/'postrun_snapshot_attempt01.json','provenance/postrun_snapshot_attempt01.json'),
    (SELF,'provenance/build_receiver_local_codec_attempt01_assets.py'),
]
missing=[str(p) for p,_ in copy_items if not p.is_file()]
if missing:
    raise SystemExit('missing required sources:\n'+'\n'.join(missing))

# Prepare the two CSVs in memory from saved QA values only; no arithmetic or score rerun.
score_rows=[]
score_fields=['sample_id','arm','max_f32_mean_error']
for codec in ('ffv1','h264'):
    for scope in ('full','guard20'):
        for metric in ('B','Dmin','D0','Cbest','Canchor','best_lag_frames','official_av_offset_frames'):
            score_fields.append(f'{codec}_{scope}_{metric}')
for row in qa['rows']:
    out={'sample_id':row['sample_id'],'arm':row['arm'],'max_f32_mean_error':row.get('max_f32_mean_error')}
    for codec in ('ffv1','h264'):
        for scope in ('full','guard20'):
            for metric in ('B','Dmin','D0','Cbest','Canchor','best_lag_frames','official_av_offset_frames'):
                out[f'{codec}_{scope}_{metric}']=row[codec][scope].get(metric)
    score_rows.append(out)

pair_rows=[]
pair_fields=['sample_id','arm']
for codec in ('ffv1','h264'):
    for scope in ('full','guard20'):
        for metric in ('B','Dmin','D0','Cbest','Canchor'):
            pair_fields.append(f'{codec}_{scope}_delta_vs_same_codec_zero_{metric}')
for scope in ('full','guard20'):
    for metric in ('B','Dmin','D0','Cbest','Canchor'):
        pair_fields.append(f'codec_difference_of_deltas_{scope}_{metric}')
for pair in qa['pairs']:
    out={'sample_id':pair['sample_id'],'arm':pair['arm']}
    for codec in ('ffv1','h264'):
        for scope in ('full','guard20'):
            for metric in ('B','Dmin','D0','Cbest','Canchor'):
                out[f'{codec}_{scope}_delta_vs_same_codec_zero_{metric}']=pair['deltas_vs_same_codec_zero'][codec][scope].get(metric)
    for scope in ('full','guard20'):
        for metric in ('B','Dmin','D0','Cbest','Canchor'):
            out[f'codec_difference_of_deltas_{scope}_{metric}']=pair['codec_difference_of_deltas'][scope].get(metric)
    pair_rows.append(out)
if len(score_rows)!=6 or len(pair_rows)!=4:
    raise SystemExit('CSV extraction row count guard mismatch')

DEST.mkdir(parents=True,exist_ok=False)
records=[]
def copy_exact(src:Path,rel:str):
    dst=DEST/rel
    if dst.exists(): raise SystemExit(f'create-only collision: {dst}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    if src.stat().st_size!=dst.stat().st_size or sha(src)!=sha(dst):
        raise SystemExit(f'copied body mismatch: {src}')
    records.append({'kind':'exact_copy','source_path':str(src.resolve()),'source_bytes':src.stat().st_size,'source_sha256':sha(src),'package_path':rel,'package_bytes':dst.stat().st_size,'package_sha256':sha(dst)})

def scalar(v):
    if v is None: return 'NA'
    if isinstance(v,bool): return 'true' if v else 'false'
    if isinstance(v,float): return format(v,'.17g')
    return str(v)

def write_table(rel,fields,rows,note):
    dst=DEST/rel
    if dst.exists(): raise SystemExit(f'create-only collision: {dst}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        w.writeheader()
        for row in rows:w.writerow({k:scalar(row.get(k)) for k in fields})
    records.append({'kind':'saved_QA_field_extraction_csv','source_path':str(QA_PATH.resolve()),'source_bytes':QA_PATH.stat().st_size,'source_sha256':sha(QA_PATH),'extraction_note':note,'package_path':rel,'package_bytes':dst.stat().st_size,'package_sha256':sha(dst)})

for src,rel in copy_items: copy_exact(src,rel)
for src,rel in selected: copy_exact(src,rel)
write_table('tables/six_arm_full_guard20_codec_scores.csv',score_fields,score_rows,'Direct extraction from ROOT QA rows; six arms, FFV1 and H264, full and guard20 metrics; numeric values formatted with 17 significant digits.')
write_table('tables/four_paired_codec_delta_did.csv',pair_fields,pair_rows,'Direct extraction from ROOT QA pairs; four plus/minus rows, same-codec deltas and stored codec_difference_of_deltas; no contrasts recomputed.')
manifest={
    'schema':'receiver_local_response_codec_attempt01_assets_v1',
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'package_path':str(DEST.resolve()),
    'scope':'Selected compact codec readback, frozen source/protocol, ROOT matrix QA, fixed-lag0 diagnostic, and faithful CSV extraction. No media bodies copied.',
    'compact_readback':{'selected_file_count':54,'selected_file_bytes':260964,'local_manifest_path':str(LOCAL_MANIFEST.resolve()),'local_manifest_bytes':LOCAL_MANIFEST.stat().st_size,'local_manifest_sha256':sha(LOCAL_MANIFEST),'remote_readback_manifest_sha256':sha(COMPACT/'remote_readback_attempt01'/'remote_readback_manifest.json'),'media_copied':False},
    'root_qa':{'status':qa['status'],'new_matrix_count':qa['new_matrix_count'],'row_count':len(qa['rows']),'pair_count':len(qa['pairs']),'qa_sha256':sha(QA_PATH)},
    'event_lag0':{'included':True,'status':event_json['status'],'execution_status':event_exec['status'],'source_path':str(event_source.resolve()),'result_path':str(event_result.resolve()),'receipt_path':str(event_receipt.resolve())},
    'files':records,
}
manifest_path=DEST/'manifest.json'
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS_CREATE_ONLY','destination':str(DEST.resolve()),'selected_count':54,'selected_bytes':260964,'package_file_count_including_manifest':sum(p.is_file() for p in DEST.rglob('*')),'manifest_path':str(manifest_path.resolve()),'manifest_sha256':sha(manifest_path),'manifest_bytes':manifest_path.stat().st_size,'score_rows':len(score_rows),'paired_rows':len(pair_rows),'media_files':[str(p) for p in DEST.rglob('*') if p.is_file() and p.suffix.lower() in {'.mp4','.mkv','.mov','.wav','.png','.jpg','.jpeg','.webp','.avi'}]},ensure_ascii=False))
