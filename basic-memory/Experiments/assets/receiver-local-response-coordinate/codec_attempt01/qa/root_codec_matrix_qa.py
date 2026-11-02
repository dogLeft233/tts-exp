"""Check the six new codec-diagnostic raw matrices once; compare verified H264 fields."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
RB = HERE / "compact_readback_attempt01/remote_readback_attempt01/output"
REMOTE = PurePosixPath('/home/wjj/tts-audio/tts-exp/.codex/scratch/native_attribute_response_probe_20261015_codec_diagnostic_attempt01/output_attempt01')
OUT = HERE / 'root_codec_matrix_qa.json'
OLD = HERE.parent / 'native_attribute_response_probe_20261015_attempt01/root_actual_output_owned_response_qa.json'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ash = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def local(p):
    return RB.joinpath(*PurePosixPath(p).relative_to(REMOTE).parts)

def check_matrix(path, meta):
    assert sha(path) == meta['matrix_npz_sha256']
    with np.load(path, allow_pickle=False) as z:
        matrix = z['distance_by_query_lag'].copy()
        assert np.array_equal(z['lag_offsets_frames'], np.arange(-15,16))
    assert matrix.dtype == np.float32 and np.isfinite(matrix).all()
    assert list(matrix.shape) == meta['matrix_shape']
    assert ash(matrix) == meta['matrix_array_sha256_le_f32']
    metrics = meta['metrics']
    curve = np.asarray(metrics['lag_curve'], dtype=np.float32)
    error = float(np.max(np.abs(matrix.astype(np.float64).mean(0)-curve.astype(np.float64))))
    assert error < 5e-6, error
    b, d, d0 = np.median(curve), curve.min(), curve[15]
    index = int(curve.argmin())
    exact = dict(B=float(b),Dmin=float(d),D0=float(d0),Cbest=float(np.float32(b-d)),
                 Canchor=float(np.float32(b-d0)),best_lag_frames=index-15,official_av_offset_frames=15-index)
    assert all(v == metrics[k] for k,v in exact.items())
    return matrix, exact, error

assert not OUT.exists()
manifest = read(RB/'codec_diagnostic_manifest.json')
assert sha(RB/'codec_diagnostic_manifest.json') == 'c24b245de73ab4edd52d2e4281fda9a8c9c0813eb80f7781d569addceaeae51b'
assert manifest['status'] == 'PASS' and manifest['row_count_completed'] == manifest['score_invocations_completed'] == 6
assert manifest['formal_h264_artifacts_modified'] is False
assert sha(OLD) == 'f470d1aa4a926cafb65f73555e6957797bf3b7176b7bbaf8a30348593d3f32d9'
old = read(OLD)['score_rows']
rows = []
for ref in manifest['rows']:
    rp = local(ref['row_receipt_path']); assert sha(rp) == ref['row_receipt_sha256']
    row = read(rp); assert row['status']=='PASS' and row['formal_h264_artifacts_modified'] is False
    ep = local(row['stagea_evaluator_receipt']); assert sha(ep)==row['stagea_evaluator_receipt_sha256']
    ev = read(ep); assert ev['status']=='PASS' and ev['audio_pcm16_sha256']==row['original_h264_pcm16_sha256']
    n = 51376 if row['sample_id']=='BAC009S0022W0123' else 56561
    frames = n*25//16000
    assert row['frame_count']==ev['video_frame_count']==frames
    assert ev['audio_sample_count']==n
    mux=row['mux_original_pcm']
    assert mux['audio_matches_original_decoded_pcm'] is True and mux['decoded_audio_sample_count']==n
    assert mux['decoded_audio_float32_sha256']==row['natural_pcm_float32_sha256']
    for key,size in [('ffv1_512_probe',512),('ffv1_224_probe',224)]:
        p=row[key]; s=p['stream']
        assert p['pts_j_over_25'] and p['frame_count']==frames
        assert s['codec_name']=='ffv1' and s['pix_fmt']=='yuv420p' and s['width']==s['height']==size
    assert all(row[k]['returncode']==0 for k in ('ffmpeg_512_command','ffmpeg_224_command','stagea_score_command'))
    meta=row['full_and_guard20_diagnostic_summary']
    matrix, full, e1=check_matrix(local(meta['matrix_npz_path']),meta)
    gm, guard, e2=check_matrix(local(meta['guard20']['matrix_npz_path']),meta['guard20'])
    assert matrix.shape==(frames-5,31) and np.array_equal(gm,matrix[20:-20])
    with np.load(local(meta['guard20']['matrix_npz_path']),allow_pickle=False) as z:
        assert np.array_equal(z['original_query_row_indices'],np.arange(20,len(matrix)-20))
    assert ash(matrix)==ev['matrix_sha256']
    prior=next(r for r in old if r['sample_id']==row['sample_id'] and r['arm']=='native_height_'+row['arm'])
    rows.append(dict(sample_id=row['sample_id'],arm=row['arm'],ffv1=dict(full=full,guard20=guard),
                     h264={k:prior[k] for k in ('full','guard20')},max_f32_mean_error=max(e1,e2)))
pairs=[]
for row in rows:
    if row['arm']=='zero':continue
    zero=next(x for x in rows if x['sample_id']==row['sample_id'] and x['arm']=='zero')
    deltas={codec:{scope:{k:row[codec][scope][k]-zero[codec][scope][k] for k in ('B','Dmin','D0','Cbest','Canchor')} for scope in ('full','guard20')} for codec in ('ffv1','h264')}
    difference={scope:{k:deltas['ffv1'][scope][k]-deltas['h264'][scope][k] for k in ('B','Dmin','D0','Cbest','Canchor')} for scope in ('full','guard20')}
    pairs.append(dict(sample_id=row['sample_id'],arm=row['arm'],deltas_vs_same_codec_zero=deltas,codec_difference_of_deltas=difference))
result=dict(schema='root_actual_codec_diagnostic_matrices_v1',status='PASS_DIAGNOSTIC_ONLY',
            source_sha256=sha(Path(__file__)),manifest_sha256=sha(RB/'codec_diagnostic_manifest.json'),
            existing_h264_qa_sha256=sha(OLD),new_matrix_count=12,rows=rows,pairs=pairs,
            max_float32_mean_rounding_error=max(r['max_f32_mean_error'] for r in rows),
            scope='Six FFV1 chain scores only; formal H264 retained; no new TFG/encoder/geometry or old-matrix replay.')
with OUT.open('x',encoding='utf-8',newline='\n') as f:
    json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'pairs':pairs,'max_float32_mean_rounding_error':result['max_float32_mean_rounding_error']}))
