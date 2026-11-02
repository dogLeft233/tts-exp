"""Independent readback of actual response matrices and frozen visual endpoints.

Scientific artifact QA only: no model execution, implementation tests, or data edits.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RB = Path(sys.argv[2]).resolve()
RENDER_SOURCE = Path(sys.argv[4]).resolve()
STAGE = RENDER_SOURCE.parent
MEDIA = Path(sys.argv[5]).resolve()
MEDIA_REMOTE = PurePosixPath('/dev/shm/cross_tfg_native_attribute_response_probe_20261015_output_owned_musetalk_render_output_attempt01')
REMOTE = PurePosixPath(sys.argv[3])
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ash = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
arms = ('native_height_zero', 'native_height_plus', 'native_height_minus')
labels = {'musetalk': 'MuseTalk'}
lags = np.arange(-8, 9, dtype=np.int64)
checks = 0
max_geometry_error = 0.0
max_score_mean_rounding_error = 0.0


def compare(actual, expected, tolerance=1e-12):
    global checks, max_geometry_error
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    error = float(np.max(np.abs(a-b))) if a.size else 0.0
    assert np.isfinite(a).all() and np.isfinite(b).all() and error <= tolerance, error
    checks += int(a.size)
    max_geometry_error = max(max_geometry_error, error)


def local(remote_path):
    return RB.joinpath(*PurePosixPath(remote_path).relative_to(REMOTE).parts)


def checked_file(remote_path, expected):
    p = local(remote_path)
    assert sha(p) == expected, str(p)
    return p


def matrix_metrics(matrix, metrics):
    global max_score_mean_rounding_error
    independent = np.mean(matrix.astype(np.float64), axis=0)
    curve = np.asarray(metrics['lag_curve'], dtype=np.float32)
    error = float(np.max(np.abs(independent-curve.astype(np.float64))))
    assert error <= 5e-6, error
    max_score_mean_rounding_error = max(max_score_mean_rounding_error, error)
    b, dmin, d0 = np.median(curve), np.min(curve), curve[15]
    idx = int(np.argmin(curve))
    rebuilt = dict(B=float(b), Dmin=float(dmin), D0=float(d0),
                   Cbest=float(np.float32(b-dmin)), Canchor=float(np.float32(b-d0)),
                   best_lag_frames=idx-15, official_av_offset_frames=15-idx)
    for key, value in rebuilt.items():
        assert value == metrics[key], (key, value, metrics[key])
    return rebuilt


def main():
    effect_path = Path(sys.argv[1]).resolve()
    output = HERE / 'root_actual_output_owned_response_qa.json'
    assert not output.exists()
    protocol = read(HERE/'response_readout_protocol.json')
    manifest = read(RB/'readout_manifest.json')
    contracts = read(RENDER_SOURCE)['audio_contracts']
    effect = read(effect_path)
    assert manifest['row_count'] == 6 and manifest['status'] == 'PASS'
    assert manifest['protocol_sha256'] == sha(HERE/'output_owned_musetalk_protocol_attempt01.json')
    assert effect['input_pins']['readout_manifest']['sha256'] == sha(RB/'readout_manifest.json')
    assert effect['input_pins']['render_source_manifest']['sha256'] == sha(RENDER_SOURCE)
    assert effect['protocol_sha256'] == manifest['protocol_sha256']
    rows, tracks, score_rows = {}, {}, []
    for item in manifest['rows']:
        receipt_path = checked_file(item['row_receipt_path'], item['row_receipt_sha256'])
        r = read(receipt_path)
        assert r['row_id'] == item['row_id'] and r['status'] == 'PASS' and r['seed'] == 20261003
        key = (r['model'], r['sample_id'], r['arm'])
        assert key not in rows
        rows[key] = r
        pcm = contracts[r['sample_id']]['natural_pcm_float32_sha256']
        assert r['natural_pcm_float32_sha256'] == pcm
        crop = read(checked_file(r['crop_mux_receipt_path'], r['crop_mux_receipt_sha256']))
        assert crop['audio_float32_sha256'] == pcm and crop['audio_matches_original_decoded_pcm'] is True
        evaluator = read(checked_file(r['syncnet']['evaluator_receipt'], r['syncnet']['evaluator_receipt_sha256']))
        sync = r['syncnet']
        full_path = checked_file(sync['matrix_npz_path'], sync['matrix_npz_sha256'])
        with np.load(full_path, allow_pickle=False) as z:
            matrix, lag_axis = z['distance_by_query_lag'].copy(), z['lag_offsets_frames'].copy()
        assert matrix.dtype == np.float32 and np.isfinite(matrix).all()
        assert list(matrix.shape) == sync['matrix_shape'] == evaluator['matrix_shape']
        assert np.array_equal(lag_axis, np.arange(-15, 16)) and sync['zero_lag_column'] == 15
        assert ash(matrix) == sync['matrix_array_sha256_le_f32'] == evaluator['matrix_sha256']
        assert evaluator['distance_npz_sha256'] == sha(full_path)
        full = matrix_metrics(matrix, sync['metrics'])
        g = sync['guard20']
        gp = checked_file(g['matrix_npz_path'], g['matrix_npz_sha256'])
        with np.load(gp, allow_pickle=False) as z:
            gm = z['distance_by_query_lag'].copy()
            gi = z['original_query_row_indices'].copy()
            assert np.array_equal(z['lag_offsets_frames'], lag_axis)
        assert np.array_equal(gm, matrix[20:-20])
        assert np.array_equal(gi, np.arange(20, len(matrix)-20))
        assert ash(gm) == g['matrix_array_sha256_le_f32']
        guard = matrix_metrics(gm, g['metrics'])
        motion = r['mouth_motion']
        visual = read(checked_file(motion['receipt'], motion['receipt_sha256']))
        vp = checked_file(motion['arrays_npz'], motion['arrays_npz_sha256'])
        with np.load(vp, allow_pickle=False) as z:
            track = {k: z[k].copy() for k in z.files}
        assert sha(vp) == visual['arrays_npz']['sha256']
        for name, value in track.items():
            meta = visual['arrays_npz']['arrays'][name]
            assert ash(value) == meta['array_sha256']
            assert list(value.shape) == meta['shape'] and value.dtype.str == meta['dtype']
        n = len(track['frame_index_int32'])
        assert np.array_equal(track['frame_index_int32'], np.arange(n))
        assert all(Fraction(int(a), int(b)) == Fraction(j,25) for j,(a,b) in enumerate(zip(track['timestamp_num_int64'],track['timestamp_den_int64'])))
        assert n == crop['source_frame_count'] == visual['frame_count'] == evaluator['video_frame_count']
        ap = track['mouth_aperture_eye_normalized_float64']
        t = track['timestamp_num_int64']/track['timestamp_den_int64']
        valid = track['tracking_valid_bool']
        vv = track['aperture_velocity_valid_bool']
        assert np.array_equal(vv, valid[:-1] & valid[1:])
        compare(track['aperture_velocity_eye_normalized_per_s_float64'][vv], (np.diff(ap)/np.diff(t))[vv], 1e-10)
        tracks[key] = track
        score_rows.append(dict(model=r['model'], sample_id=r['sample_id'], arm=r['arm'], full=full, guard20=guard,
                               tracking_fraction=float(valid.mean()), aperture_velocity_rms=float(np.sqrt(np.mean(track['aperture_velocity_eye_normalized_per_s_float64'][vv]**2)))))
    assert len(rows) == 6
    support_path = ROOT/protocol['support']['reference_s17_path']
    old_path = ROOT/protocol['phase']['source_effect_path']
    assert sha(support_path) == protocol['support']['reference_s17_sha256']
    assert sha(old_path) == protocol['phase']['source_effect_sha256']
    source_tokens = read(support_path)['token_records']
    old_phase = read(old_path)['utterance_rms_offset_reference']
    cells = []
    for c in effect['cells']:
        model, sample = c['model'], c['sample_id']
        refs = [x for x in old_phase if x['attribute_id']=='vowel_height_aperture_proxy' and x['model']==labels[model] and x['sample_id']==sample]
        assert len(refs)==1
        ref = refs[0]
        p = int(np.median([x['full_official_av_offset_frames'] for x in ref['rms_seed_lag_fields']]))
        assert p == ref['full_offset_median_integer'] == c['p_old_rms_absolute_lag']
        arrays = {a:tracks[model,sample,a] for a in arms}
        n = len(arrays[arms[0]]['frame_index_int32'])
        count = contracts[sample]['sample_count']
        common = np.arange(n)*640 < count
        for a in arms:
            assert len(arrays[a]['frame_index_int32']) == n
            common &= arrays[a]['tracking_valid_bool']
        normalized = {}
        for a in arms:
            ap = arrays[a]['mouth_aperture_eye_normalized_float64']
            mu, sd = float(ap[common].mean()), float(ap[common].std())
            moments = c['per_arm_aperture_standardization'][a]
            compare(mu, moments['mean_on_three_arm_common_natural_frames'])
            compare(sd, moments['population_sd_on_three_arm_common_natural_frames'])
            assert moments['frame_count'] == int(common.sum())
            assert sd>1e-8
            normalized[a] = (ap-mu)/sd
        originals = [x for x in source_tokens if x['attribute_id']=='vowel_height_aperture_proxy' and x['model']==labels[model] and x['sample_id']==sample]
        assert len(originals)==len(c['tokens'])
        by_phone = defaultdict(list)
        losses, supported = [], []
        for original, saved in zip(originals,c['tokens']):
            assert original['phone_index']==saved['phone_index'] and original['phone']==saved['phone']
            assert original['status']==saved['original_s17_status']
            if original['status']!='SUPPORTED_S17_TOKEN':
                continue
            indices=original['s17_indices']
            keep=[j for j in indices if 0<=j<n and common[j] and np.all(j+lags>=0) and np.all(j+lags<n) and np.all(common[j+lags])]
            loss=[j for j in indices if j not in keep]
            assert keep==saved['retained_core_frame_indices'] and loss==saved['lost_core_frame_indices']
            losses.extend(loss)
            if len(keep)<2:
                assert saved['analysis_status'].startswith('UNMEASURABLE')
                continue
            supported.append(original)
            for a in arms:
                for scale in ('raw','z'):
                    y=arrays[a]['mouth_aperture_eye_normalized_float64'] if scale=='raw' else normalized[a]
                    curve=np.array([np.mean(y[np.array(keep)+lag]) for lag in lags])
                    compare(curve,saved['curves'][a][scale])
                    by_phone[a,scale,original['class'],original['basephone']].append(curve)
        endpoints={a:{} for a in arms}
        for a in arms:
            for scale in ('raw','z'):
                classes={}
                for cl in ('high','low'):
                    pm={phone:np.mean(np.stack(cs),axis=0) for (aa,ss,cc,phone),cs in by_phone.items() if (aa,ss,cc)==(a,scale,cl)}
                    assert pm
                    classes[cl]=np.mean(np.stack(list(pm.values())),axis=0)
                    saved_class=c['C_curves'][a][scale]
                    compare(classes[cl],saved_class[f'{cl}_class_mean_by_absolute_lag'])
                    for phone,cv in pm.items(): compare(cv,saved_class[f'{cl}_phone_balanced_means'][phone])
                contrast=classes['high']-classes['low']
                compare(contrast,c['C_curves'][a][scale]['C_high_minus_low_by_absolute_lag'])
                cp=float(contrast[p+8]); hp=cp-float((contrast[p+4]+contrast[p+12])/2)
                endpoints[a][scale]=dict(Cp=cp,Hp=hp)
                compare(cp,c['fixed_p_endpoints'][a][scale]['Cp'])
                compare(hp,c['fixed_p_endpoints'][a][scale]['Hp'])
        for a in arms[1:]:
            for scale in ('raw','z'):
                for field in ('Cp','Hp'):
                    compare(endpoints[a][scale][field]-endpoints[arms[0]][scale][field],c['plus_minus_deltas_vs_new_zero'][a][scale]['delta_'+field+'_vs_new_zero'])
        zero_ok=all(endpoints[arms[0]][s][k]>1e-9 for s in ('raw','z') for k in ('Cp','Hp'))
        ordered=all(endpoints[arms[1]][s][k]-endpoints[arms[0]][s][k]>1e-9 and endpoints[arms[0]][s][k]-endpoints[arms[2]][s][k]>1e-9 for s in ('raw','z') for k in ('Cp','Hp'))
        cells.append(dict(model=model,sample_id=sample,p=p,frame_count=n,common_valid_frames=int(common.sum()),
                          supported_high=sum(x['class']=='high' for x in supported),supported_low=sum(x['class']=='low' for x in supported),
                          support_lost_frame_count=len(losses),zero_functional_control=zero_ok,bidirectional_ordering=ordered,
                          frozen_response_pass=zero_ok and ordered,endpoints=endpoints))
    assert len(cells)==2
    pairs=[]
    for r in score_rows:
        if r['arm']==arms[0]:continue
        base=next(x for x in score_rows if (x['model'],x['sample_id'],x['arm'])==(r['model'],r['sample_id'],arms[0]))
        d={e:{k:r[e][k]-base[e][k] for k in ('B','Dmin','D0','Cbest')} for e in ('full','guard20')}
        pairs.append(dict(model=r['model'],sample_id=r['sample_id'],arm=r['arm'],delta_vs_new_zero=d,
                          full_joint=d['full']['Cbest']>0 and d['full']['Dmin']<=0 and d['full']['D0']<=0))
    codec = actual_png_and_h264(rows, tracks)
    payload=dict(schema='root_actual_output_owned_musetalk_response_qa_v1',status='PASS_ACTUAL_ARRAY_QA_NOT_EFFICACY',
                 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 input_pins={str(p.relative_to(ROOT)):sha(p) for p in (effect_path,RB/'readout_manifest.json',RB.parent/'local_readback_manifest.json',HERE/'response_readout_protocol.json',HERE/'output_owned_musetalk_protocol_attempt01.json',RENDER_SOURCE,support_path,old_path)},
                 matrix_count=12,geometry_rows=6,geometry_comparisons=checks,max_geometry_error=max_geometry_error,
                 canonical_curve_check='Independent float64 column mean versus reported official float32 curve <=5e-6; exact B/min/D0/C fields rebuilt from that verified float32 curve.',
                 codec_control=codec,max_float32_mean_rounding_error=max_score_mean_rounding_error,scientific_cells=cells,score_rows=score_rows,score_pairs=pairs,
                 model_response_pass={m:all(x['frozen_response_pass'] for x in cells if x['model']==m) for m in labels},
                 interpretation='Seen-DEV two-person screening only; original PCM/clock; no peak retuning, human accuracy claim, or waveform reachability claim.')
    with output.open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'path':str(output),'sha256':sha(output),'cells':cells,'score_pairs':pairs,'geometry_comparisons':checks,'max_geometry_error':max_geometry_error,'max_score_mean_rounding_error':max_score_mean_rounding_error},ensure_ascii=False))



def actual_png_and_h264(rows, tracks):
    # Verify actual retained RGB bodies independently of the executor control.
    from PIL import Image
    owned_doc = read(HERE/'output_owned_actual_frame_map_attempt01.json')
    owned = owned_doc['owned_output_frame_indices_by_sample']
    body_hashes, h264_tracks, codec_rows = {}, {}, []
    for key, row in rows.items():
        primary = read(checked_file(row['mouth_motion']['receipt'], row['mouth_motion']['receipt_sha256']))
        hashes = []
        for entry in primary['precodec_frames']:
            path = MEDIA.joinpath(*PurePosixPath(entry['path']).relative_to(MEDIA_REMOTE).parts)
            assert sha(path) == entry['png_file_sha256']
            with Image.open(path) as im:
                rgb = np.asarray(im.convert('RGB'), dtype=np.uint8)
            h = ash(rgb)
            assert h == entry['decoded_rgb_body_sha256']
            hashes.append(h)
        assert hashes == primary['precodec_rgb_body_sha256_by_frame']
        body_hashes[key] = hashes
        motion = row['mouth_motion_h264']
        visual = read(checked_file(motion['receipt'], motion['receipt_sha256']))
        npz = checked_file(motion['arrays_npz'], motion['arrays_npz_sha256'])
        assert sha(npz) == visual['arrays_npz']['sha256']
        with np.load(npz, allow_pickle=False) as z:
            h264 = {k:z[k].copy() for k in z.files}
        for name, value in h264.items():
            meta = visual['arrays_npz']['arrays'][name]
            assert ash(value) == meta['array_sha256'] and value.dtype.str == meta['dtype'] and list(value.shape) == meta['shape']
        primary_track = tracks[key]
        n = len(primary_track['frame_index_int32'])
        assert len(hashes) == n == visual['frame_count']
        assert np.array_equal(h264['frame_index_int32'], primary_track['frame_index_int32'])
        assert all(Fraction(int(a),int(b))==Fraction(j,25) for j,(a,b) in enumerate(zip(h264['timestamp_num_int64'],h264['timestamp_den_int64'])))
        valid = h264['tracking_valid_bool']
        vv = h264['aperture_velocity_valid_bool']
        assert np.array_equal(vv, valid[:-1]&valid[1:])
        aperture = h264['mouth_aperture_eye_normalized_float64']
        t = h264['timestamp_num_int64']/h264['timestamp_den_int64']
        compare(h264['aperture_velocity_eye_normalized_per_s_float64'][vv], (np.diff(aperture)/np.diff(t))[vv], 1e-10)
        common = valid & primary_track['tracking_valid_bool']
        delta = aperture[common] - primary_track['mouth_aperture_eye_normalized_float64'][common]
        h264_tracks[key] = h264
        codec_rows.append({'model':key[0], 'sample_id':key[1], 'arm':key[2], 'png_frame_count':n,
                           'h264_tracking_fraction':float(valid.mean()), 'h264_minus_precodec_aperture_rms':float(np.sqrt(np.mean(delta**2))),
                           'h264_minus_precodec_aperture_max_abs':float(np.max(np.abs(delta)))})
    controls = []
    for sample in owned:
        base = ('musetalk',sample,arms[0])
        outside = [j for j in range(len(body_hashes[base])) if j not in owned[sample]]
        for arm in arms[1:]:
            key = ('musetalk',sample,arm)
            assert all(body_hashes[key][j] == body_hashes[base][j] for j in outside)
            primary_delta = tracks[key]['mouth_aperture_eye_normalized_float64'][outside] - tracks[base]['mouth_aperture_eye_normalized_float64'][outside]
            h264_delta = h264_tracks[key]['mouth_aperture_eye_normalized_float64'][outside] - h264_tracks[base]['mouth_aperture_eye_normalized_float64'][outside]
            controls.append({'sample_id':sample, 'arm':arm, 'nonowned_original_rgb_bodies_equal':True,
                             'owned_rgb_changed_frame_indices_descriptive':[j for j in owned[sample] if body_hashes[key][j] != body_hashes[base][j]],
                             'nonowned_frames_checked':len(outside), 'precodec_nonowned_aperture_delta_max_abs':float(np.max(np.abs(primary_delta))),
                             'h264_nonowned_aperture_delta_max_abs':float(np.max(np.abs(h264_delta))),
                             'h264_nonowned_aperture_changed_frame_count_descriptive':int(np.count_nonzero(h264_delta))})
    return {'root_actual_png_body_verification':'PASS_504_ACTUAL_RGB_BODIES_AND_NONOWNED_CONTROL',
            'png_frame_count':sum(len(x) for x in body_hashes.values()), 'h264_secondary_geometry_rows':len(codec_rows),
            'codec_rows':codec_rows, 'nonowned_controls':controls,
            'interpretation':'H264 geometry is a secondary codec contrast; precodec geometry is the frozen primary. No changed support/phase or human accuracy claim.'}

if __name__=='__main__':main()
