from __future__ import annotations
import csv, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(r'E:\Documents\tts-exp')
SCRATCH = PROJECT / '.codex' / 'scratch'
PROBE = SCRATCH / 'native_attribute_response_probe_20261015_attempt01'
READBACK = PROBE / 'output_owned_musetalk_readout_attempt02_readback'
DEST = PROJECT / 'basic-memory' / 'docs' / 'experiments' / 'assets' / '42-output-owned-native-coordinate'
SELF = Path(__file__).resolve()

if DEST.exists():
    raise SystemExit(f'create-only refusal: destination exists: {DEST}')

copy_items = [
    (PROBE/'concept_protocol.json', 'protocols/concept_protocol.json'),
    (PROBE/'output_owned_musetalk_protocol_attempt01.json', 'protocols/output_owned_musetalk_protocol_attempt01.json'),
    (PROBE/'response_readout_protocol.json', 'protocols/response_readout_protocol.json'),
    (PROBE/'render_control_protocol.json', 'protocols/render_control_protocol.json'),
    (SCRATCH/'functional_lag_response_20261015_attempt01'/'protocol.json', 'protocols/functional_lag_response_protocol.json'),
    (SCRATCH/'functional_lag_response_20261015_attempt01'/'s17_support_attempt01.json', 'protocols/functional_lag_s17_support_attempt01.json'),

    (PROBE/'root_actual_cpu6_qa.json', 'qa/root_actual_cpu6_qa.json'),
    (PROBE/'root_actual_cpu6_qa.py', 'qa/root_actual_cpu6_qa.py'),
    (PROBE/'root_actual_output_owned_qa.json', 'qa/root_actual_output_owned_qa.json'),
    (PROBE/'root_actual_output_owned_qa.py', 'qa/root_actual_output_owned_qa.py'),
    (PROBE/'root_actual_output_owned_response_qa.json', 'qa/root_actual_output_owned_response_qa.json'),
    (PROBE/'root_actual_output_owned_response_qa.py', 'qa/root_actual_output_owned_response_qa.py'),
    (PROBE/'root_actual_output_owned_pcm_clock_qa.json', 'qa/root_actual_output_owned_pcm_clock_qa.json'),
    (PROBE/'root_primitive_projection_qa.json', 'qa/root_primitive_projection_qa.json'),
    (PROBE/'root_primitive_projection_qa.py', 'qa/root_primitive_projection_qa.py'),

    (PROBE/'output_owned_geometry_effect_attempt01.json', 'effects/output_owned_geometry_effect_attempt01.json'),
    (PROBE/'output_owned_geometry_effect_attempt01.execution_receipt_retry01.json', 'effects/output_owned_geometry_effect_attempt01.execution_receipt_retry01.json'),
    (PROBE/'output_owned_musetalk_geometry_consumer_attempt02.py', 'effects/output_owned_musetalk_geometry_consumer_attempt02.py'),
    (PROBE/'root_h264_functional_comparison.json', 'effects/root_h264_functional_comparison.json'),
    (PROBE/'root_h264_functional_comparison.py', 'effects/root_h264_functional_comparison.py'),
    (PROBE/'output_owned_actual_frame_map_attempt01.json', 'effects/output_owned_actual_frame_map_attempt01.json'),
    (PROBE/'output_owned_candidates_attempt01.execution_receipt.json', 'effects/output_owned_candidates_attempt01.execution_receipt.json'),
    (PROBE/'output_owned_musetalk_render_readback_attempt01'/'output'/'controller_receipt.json', 'effects/output_owned_musetalk_render_controller_receipt.json'),

    (PROBE/'h264_rgb_pixel_diagnostic_attempt01'/'h264_rgb_pixel_diagnostic_attempt01.py', 'pixel_diagnostic/h264_rgb_pixel_diagnostic_attempt01.py'),
    (PROBE/'h264_rgb_pixel_diagnostic_attempt01'/'results'/'h264_rgb_pixel_diagnostic_attempt01.json', 'pixel_diagnostic/h264_rgb_pixel_diagnostic_attempt01.json'),
    (PROBE/'h264_rgb_pixel_diagnostic_attempt01'/'results'/'h264_rgb_pixel_execution_receipt_attempt01.json', 'pixel_diagnostic/h264_rgb_pixel_execution_receipt_attempt01.json'),
    (PROBE/'h264_rgb_pixel_diagnostic_attempt01'/'results'/'h264_rgb_pixel_summary_attempt01.json', 'pixel_diagnostic/h264_rgb_pixel_summary_attempt01.json'),

    (PROBE/'whisper_primitive_projection_attempt01.py', 'whisper_primitive_projection/whisper_primitive_projection_attempt01.py'),
    (PROBE/'whisper_primitive_projection_attempt01.json', 'whisper_primitive_projection/whisper_primitive_projection_attempt01.json'),
    (PROBE/'whisper_primitive_projection_attempt01.execution_receipt.json', 'whisper_primitive_projection/whisper_primitive_projection_attempt01.execution_receipt.json'),
    (PROBE/'whisper_primitive_projection_attempt01.execution_started.json', 'whisper_primitive_projection/whisper_primitive_projection_attempt01.execution_started.json'),
]

readout_files = sorted(p for p in READBACK.rglob('*') if p.is_file() and p.relative_to(READBACK).as_posix() != 'local_readback_manifest.json')
readout_bytes = sum(p.stat().st_size for p in readout_files)
if len(readout_files) != 67 or readout_bytes != 887654:
    raise SystemExit(f'readout67 guard mismatch: count={len(readout_files)} bytes={readout_bytes}')
media_ext = {'.mp4','.mkv','.mov','.wav','.png','.jpg','.jpeg','.webp','.avi'}
if any(p.suffix.lower() in media_ext for p in readout_files):
    raise SystemExit('readout67 unexpectedly contains media')

extra_items = [
    (READBACK/'local_readback_manifest.json', 'provenance/output_owned_musetalk_local_readback_manifest.json'),
    (SELF, 'provenance/build_report42_assets_attempt01.py'),
]
missing = [str(p) for p, _ in copy_items + extra_items if not p.is_file()]
if missing:
    raise SystemExit('missing required source files:\n'+'\n'.join(missing))

DEST.mkdir(parents=True, exist_ok=False)
file_records = []

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def copy_record(src: Path, rel: str, kind='exact_copy'):
    dst = DEST / rel
    if dst.exists():
        raise SystemExit(f'create-only collision: {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    record = {
        'kind': kind,
        'source_path': str(src.resolve()),
        'source_bytes': src.stat().st_size,
        'source_sha256': sha256(src),
        'package_path': rel.replace('\\','/'),
        'package_bytes': dst.stat().st_size,
        'package_sha256': sha256(dst),
    }
    file_records.append(record)
    return dst

for src, rel in copy_items:
    copy_record(src, rel)
for src, rel in extra_items:
    copy_record(src, rel)

readout_dest = DEST / 'readout67'
for src in readout_files:
    rel = src.relative_to(READBACK).as_posix()
    copy_record(src, 'readout67/' + rel)

# Tables are faithful field extraction from frozen JSON; values are not recomputed.
response_path = PROBE/'root_actual_output_owned_response_qa.json'
effect_path = PROBE/'output_owned_geometry_effect_attempt01.json'
h264_path = PROBE/'root_h264_functional_comparison.json'
pixel_path = PROBE/'h264_rgb_pixel_diagnostic_attempt01'/'results'/'h264_rgb_pixel_diagnostic_attempt01.json'
precodec_path = READBACK/'readout'/'precodec_nonowned_rgb_control.json'

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def fmt(v):
    if v is None:
        return 'NA'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return format(v, '.17g')
    if isinstance(v, (int, str)):
        return str(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, separators=(',', ':'), sort_keys=isinstance(v, dict))
    return str(v)

def emit_csv(rel: str, fields, rows, source_paths, note: str):
    dst = DEST / rel
    if dst.exists():
        raise SystemExit(f'create-only collision: {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})
    sources = []
    for p in source_paths:
        sources.append({'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    file_records.append({
        'kind': 'json_field_extraction_csv',
        'source_path': [s['path'] for s in sources],
        'source_bytes': [s['bytes'] for s in sources],
        'source_sha256': [s['sha256'] for s in sources],
        'extraction_note': note,
        'package_path': rel.replace('\\','/'),
        'package_bytes': dst.stat().st_size,
        'package_sha256': sha256(dst),
    })

response = read_json(response_path)
effect = read_json(effect_path)
h264 = read_json(h264_path)
pixel = read_json(pixel_path)
precodec = read_json(precodec_path)

# Primary PNG/pre-codec output-owned Cp/Hp endpoints and frozen support/validity.
effect_by_sample = {c['sample_id']: c for c in effect['cells']}
primary_rows = []
for c in response['scientific_cells']:
    sample = c['sample_id']
    ec = effect_by_sample[sample]
    for arm, endpoints in c['endpoints'].items():
        feature = read_json(READBACK/'readout'/sample/'musetalk'/arm/'mouth_geometry_aperture_velocity_precodec.json')
        std = ec['per_arm_aperture_standardization'][arm]
        for scale in ('raw','z'):
            pass
        primary_rows.append({
            'model': c['model'], 'sample_id': sample, 'arm': arm,
            'primary_observer': 'precodec_png', 'p_frames': c['p'],
            'frame_count': c['frame_count'], 'common_valid_frames': c['common_valid_frames'],
            'valid_frame_count': feature.get('valid_frame_count'),
            'valid_fraction': feature.get('valid_fraction'),
            'tracking_valid_fraction': feature.get('metrics',{}).get('tracking_valid_fraction'),
            'supported_high_tokens': c['supported_high'], 'supported_low_tokens': c['supported_low'],
            'support_lost_frame_count': c['support_lost_frame_count'],
            'zero_functional_control': c['zero_functional_control'],
            'bidirectional_ordering': c['bidirectional_ordering'],
            'frozen_response_pass': c['frozen_response_pass'],
            'aperture_mean_common_frames': std['mean_on_three_arm_common_natural_frames'],
            'aperture_population_sd_common_frames': std['population_sd_on_three_arm_common_natural_frames'],
            'raw_Cp': endpoints[arm]['raw']['Cp'], 'raw_Hp': endpoints[arm]['raw']['Hp'],
            'z_Cp': endpoints[arm]['z']['Cp'], 'z_Hp': endpoints[arm]['z']['Hp'],
        })
primary_fields = ['model','sample_id','arm','primary_observer','p_frames','frame_count','common_valid_frames','valid_frame_count','valid_fraction','tracking_valid_fraction','supported_high_tokens','supported_low_tokens','support_lost_frame_count','zero_functional_control','bidirectional_ordering','frozen_response_pass','aperture_mean_common_frames','aperture_population_sd_common_frames','raw_Cp','raw_Hp','z_Cp','z_Hp']
emit_csv('tables/primary_raw_z_Cp_Hp_support.csv', primary_fields, primary_rows, [response_path,effect_path], 'Direct field extraction: root response QA primary PNG endpoints plus existing effect/support and precodec geometry validity fields; no calculations.')

# Canonical full/guard20 score fields retained exactly, including Cbest and Canchor.
score_rows = []
for row in response['score_rows']:
    out = {'model':row['model'],'sample_id':row['sample_id'],'arm':row['arm'],'tracking_fraction':row.get('tracking_fraction'),'aperture_velocity_rms':row.get('aperture_velocity_rms')}
    for scope in ('full','guard20'):
        for key in ('B','Dmin','D0','Cbest','Canchor','best_lag_frames','official_av_offset_frames'):
            out[f'{scope}_{key}'] = row.get(scope,{}).get(key)
    score_rows.append(out)
score_fields=['model','sample_id','arm','tracking_fraction','aperture_velocity_rms']+[f'{scope}_{key}' for scope in ('full','guard20') for key in ('B','Dmin','D0','Cbest','Canchor','best_lag_frames','official_av_offset_frames')]
emit_csv('tables/full_guard20_scores.csv', score_fields, score_rows, [response_path], 'Direct field extraction from six frozen score_rows; Cbest and Canchor both retained as present in schema.')

# Token-level aperture velocity and valid-support description, from existing effect/readout JSON.
velocity_rows=[]
for ec in effect['cells']:
    sample=ec['sample_id']
    for token in ec['tokens']:
        for arm, v in token['velocity_by_arm_on_original_s17_core'].items():
            feature=read_json(READBACK/'readout'/sample/'musetalk'/arm/'mouth_geometry_aperture_velocity_precodec.json')
            velocity_rows.append({
                'model':ec['model'],'sample_id':sample,'arm':arm,
                'phone_index':token['phone_index'],'phone':token['phone'],'basephone':token['basephone'],'class':token['class'],
                'analysis_status':token['analysis_status'],'original_s17_status':token['original_s17_status'],
                'raw_start_s':token['raw_start_s'],'raw_end_s':token['raw_end_s'],
                'original_s17_core_frame_indices':token['original_s17_core_frame_indices'],
                'retained_core_frame_indices':token['retained_core_frame_indices'],'lost_core_frame_indices':token['lost_core_frame_indices'],
                'render_frame_count':feature.get('frame_count'),'valid_frame_count':feature.get('valid_frame_count'),
                'valid_fraction':feature.get('valid_fraction'),'tracking_valid_fraction':feature.get('metrics',{}).get('tracking_valid_fraction'),
                'transition_count':v.get('transition_count'),'mean_ipd_per_s':v.get('mean_ipd_per_s'),'rms_ipd_per_s':v.get('rms_ipd_per_s'),
                'p05_ipd_per_s':v.get('p05_ipd_per_s'),'p95_ipd_per_s':v.get('p95_ipd_per_s'),
            })
velocity_fields=['model','sample_id','arm','phone_index','phone','basephone','class','analysis_status','original_s17_status','raw_start_s','raw_end_s','original_s17_core_frame_indices','retained_core_frame_indices','lost_core_frame_indices','render_frame_count','valid_frame_count','valid_fraction','tracking_valid_fraction','transition_count','mean_ipd_per_s','rms_ipd_per_s','p05_ipd_per_s','p95_ipd_per_s']
emit_csv('tables/token_velocity_and_valid_support.csv',velocity_fields,velocity_rows,[effect_path], 'Direct field extraction from existing token velocity/support and precodec valid-support JSON; no velocity recomputation.')

# Exact precodec PNG nonowned RGB identity control.
precodec_rows=[]
for row in precodec['rows']:
    precodec_rows.append({
        'observer':'precodec_png_rgb8','sample_id':row['sample_id'],'candidate_arms_compared':'plus;minus',
        'candidate_frames_compared':row['candidate_frames_compared'],
        'nonowned_frame_count_checked_per_candidate':row['nonowned_frame_count_checked_per_candidate'],
        'mismatch_count':row['mismatch_count'],'status':row['status'],
        'owned_output_frame_indices':row['owned_output_frame_indices'],
        'comparison':precodec['comparison'],
    })
precodec_fields=['observer','sample_id','candidate_arms_compared','candidate_frames_compared','nonowned_frame_count_checked_per_candidate','mismatch_count','status','owned_output_frame_indices','comparison']
emit_csv('tables/precodec_nonowned_rgb_exact_control.csv',precodec_fields,precodec_rows,[precodec_path], 'Direct field extraction from retained PNG RGB8 control; row counts are the source aggregate across plus/minus.')

# H.264 decoded RGB24 pixel changes with the source-video hashes used by each decode.
decode_by_sample_arm={}
for rec in pixel.get('ffmpeg_decode_processes',[]):
    decode_by_sample_arm[(rec['sample_id'],rec['arm'])]=rec
pixel_rows=[]
for row in pixel['comparisons_vs_zero']:
    sample=row['sample_id']; arm=row['candidate_arm']
    candidate=decode_by_sample_arm.get((sample,arm),{})
    zero=decode_by_sample_arm.get((sample,'zero'),{})
    pixel_rows.append({
        'observer':'silent_h264_decoded_rgb24','sample_id':sample,'candidate_arm':arm,
        'frame_count':row['frame_count'],'owned_frame_count':row['owned_frame_count'],'nonowned_frame_count':row['nonowned_frame_count'],
        'changed_frame_count_owned':row['changed_frame_count_owned'],'changed_frame_count_nonowned':row['changed_frame_count_nonowned'],
        'nonowned_mean_absolute_rgb_byte_difference':row['nonowned_mean_absolute_rgb_byte_difference'],
        'nonowned_max_absolute_rgb_byte_difference':row['nonowned_max_absolute_rgb_byte_difference'],
        'owned_output_frame_indices':row['owned_frame_indices'],
        'candidate_video_sha256':candidate.get('source_video_sha256'),'zero_video_sha256':zero.get('source_video_sha256'),
        'mean_definition':row['mean_definition'],
    })
pixel_fields=['observer','sample_id','candidate_arm','frame_count','owned_frame_count','nonowned_frame_count','changed_frame_count_owned','changed_frame_count_nonowned','nonowned_mean_absolute_rgb_byte_difference','nonowned_max_absolute_rgb_byte_difference','owned_output_frame_indices','candidate_video_sha256','zero_video_sha256','mean_definition']
emit_csv('tables/h264_decoded_rgb24_nonowned_differences.csv',pixel_fields,pixel_rows,[pixel_path], 'Direct field extraction from the one-pass six-video RGB24 diagnostic; difference metrics are source values, not recomputed.')

# H.264 functional endpoints are secondary observer values, separated from primary PNG table.
h264_rows=[]
for cell in h264['cells']:
    for arm, scales in cell['new_owned_operator_h264_endpoints'].items():
        h264_rows.append({
            'observer':'silent_h264_mediapipe_geometry_secondary','model':'musetalk','sample_id':cell['sample_id'],'arm':arm,
            'p_frames':cell['p'],'common_valid_frames':cell['common_valid_frames'],
            'new_owned_operator_h264_bidirectional_pass':cell['new_owned_operator_h264_bidirectional_pass'],
            'raw_Cp':scales['raw']['Cp'],'raw_Hp':scales['raw']['Hp'],'z_Cp':scales['z']['Cp'],'z_Hp':scales['z']['Hp'],
        })
h264_fields=['observer','model','sample_id','arm','p_frames','common_valid_frames','new_owned_operator_h264_bidirectional_pass','raw_Cp','raw_Hp','z_Cp','z_Hp']
emit_csv('tables/h264_functional_endpoints_secondary.csv',h264_fields,h264_rows,[h264_path], 'Direct extraction of H.264 geometry endpoints; retained as a secondary observer and not substituted for primary PNG endpoints.')

# Manifest source metadata is calculated from actual bytes for every copied or extracted package file.
manifest={
    'schema':'report42_output_owned_native_coordinate_assets_v1',
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'package_path':str(DEST.resolve()),
    'scope':'Canonical compact science assets and faithful CSV field extractions for report 42; no MP4/PNG/WAV media bodies copied.',
    'readout67':{'source_root':str(READBACK.resolve()),'copied_file_count':len(readout_files),'copied_bytes':readout_bytes,'excluded_sidecar':'local_readback_manifest.json copied separately to provenance/.'},
    'files':file_records,
}
manifest_path=DEST/'manifest.json'
if manifest_path.exists():
    raise SystemExit('create-only manifest collision')
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({
    'status':'PASS_CREATE_ONLY_ASSET_PACKAGE',
    'destination':str(DEST.resolve()),
    'readout67_file_count':len(readout_files),
    'readout67_bytes':readout_bytes,
    'package_file_count_including_manifest':sum(1 for p in DEST.rglob('*') if p.is_file()),
    'manifest_path':str(manifest_path.resolve()),
    'manifest_sha256':sha256(manifest_path),
    'manifest_bytes':manifest_path.stat().st_size,
    'tables':[p.name for p in (DEST/'tables').glob('*.csv')],
},ensure_ascii=False))
