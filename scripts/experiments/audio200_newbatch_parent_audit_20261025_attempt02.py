"""Independent CPU readback of each newly prepared media/feature batch."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import pickle
import time


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def expected_pairs(group):
    arms = {(r['arm'], r['draw']): r['id'] for r in group}
    assert len(arms) == len(group) == 7
    assert set(arms) == {('N', 0)} | {(a, d) for a in ('L06', 'L17') for d in (1, 2, 3)}
    pairs = set()
    for d in (1, 2, 3):
        for family, a, b in [('N-L06', ('N', 0), ('L06', d)),
                             ('N-L17', ('N', 0), ('L17', d)),
                             ('L17-L06', ('L17', d), ('L06', d))]:
            pairs.add((family, arms[a], arms[b]))
    for arm in ('L06', 'L17'):
        for a, b in combinations((1, 2, 3), 2):
            pairs.add((arm+'-within', arms[arm, a], arms[arm, b]))
    return pairs


def nearest_window(interval, delay, shift, centers):
    anchor = sum(interval)/2000 + delay + shift
    i = min(range(len(centers)), key=lambda j: abs(centers[j]-anchor))
    return anchor, i, 5 <= i < len(centers)-5


def official_scalar(original):
    import torch
    mu = torch.from_numpy(original).mean(0)
    return float(torch.median(mu)-mu.min())


def audit(manifest, clock_path, patch_path):
    import numpy as np
    import torch
    import python_speech_features
    import soundfile as sf
    from scipy.signal import correlate, correlation_lags
    from scipy.io import wavfile
    started = time.monotonic()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    rows = read(manifest)['rows']
    assert rows and len(rows) == len({r['id'] for r in rows})
    source_file = root/'runs/audio_data_inventory200_20261024_attempt05/selected600_natural_manifest.json'
    assert digest(source_file) == '3ef28c8076aba23e64e2bc1a22963e24dba1e9a89ed9e2aeaf7fdaae6821e608'
    sources = {r['source_id']: r for r in read(source_file)['rows']}
    group, checked_files, centers, aligned, lags, details = defaultdict(list), {}, {}, {}, {}, []
    def check(path, expected):
        if path not in checked_files:
            checked_files[path] = digest(path)
        assert checked_files[path] == expected, (path, 'SHA drift')
    frozen = root/'runs/syncnet_audio_replay20_20261024_attempt01'
    protocol = read(frozen/'protocol.json')
    for p, h in protocol['source_pins'].items():
        check(p, h)
    weights = root/'third_party/syncnet_python/data/syncnet_v2.model'
    state = torch.load(weights, map_location='cpu', weights_only=True)
    weight, bias = (state[k].numpy() for k in ('netfcaud.3.weight', 'netfcaud.3.bias'))
    clocks = {r['id']: r for r in read(clock_path)['rows']}
    assert set(clocks) == {r['id'] for r in rows}
    fa_cache, audio_cache = {}, {}
    for r in rows:
        sid, key = r['sample_id'], r['id']
        s = sources[sid]
        assert r['speaker_id'] == s['speaker_id'] and r['split'] == s['split']
        assert r['text'] == ''.join(s['text'].split())
        assert key == f"{sid}_{r['arm']}_{r['draw']}"
        group[sid].append(r)
        for kind in ('sequence', 'scoring_audio', 'replay', 'canonical_audio', 'video', 'score', 'activesd'):
            check(r[kind+'_path'], r[kind+'_sha256'])
        replay = read(r['replay_path'])
        score = read(r['score_path'])
        assert score['status'] == 'COMPLETE' and len(score['tracks']) == 1
        assert score['sample_id'] == sid and score['speaker_id'] == s['speaker_id']
        assert score['mux_PCM_exact'] and score['decoded_frames_exact'] and score['frame_PTS_exact']
        assert score['canonical_audio_sha256'] == r['canonical_audio_sha256']
        track = score['tracks'][0]['metadata']
        check(track['crop_file'], track['crop_sha256'])
        assert track['track_start_frame'] == 0
        original = np.asarray(pickle.loads(Path(r['activesd_path']).read_bytes())[0])
        with np.load(r['sequence_path'], allow_pickle=False) as z:
            a, v, a512, d = (z[k] for k in ('audio1024', 'video1024', 'audio512', 'distance'))
            n = len(a)
            assert a.shape == v.shape == (n, 1024) and a512.shape == (n, 512)
            assert all(np.isfinite(z[k]).all() for k in z.files)
            assert np.array_equal(z['row_indices'], np.arange(n))
            assert np.array_equal(d, original) and np.array_equal(z['original_distance'], original)
            padded = np.pad(a, ((15, 15), (0, 0)))
            calculated = np.empty((n, 31), dtype=np.float32)
            for k in range(31):
                diff = v-padded[k:k+n]+np.float32(1e-6)
                calculated[:, k] = np.sqrt(np.sum(diff*diff, axis=1, dtype=np.float32))
            delta = calculated.astype(np.float64)-d
            maximum = float(np.abs(delta).max())
            assert maximum <= protocol['tolerances']['curve_max_abs']
            assert float(np.sqrt(np.mean(delta**2))) <= protocol['tolerances']['curve_rms']
            assert float(np.abs(a512 @ weight.T+bias-a).max()) <= 1e-4
            mu = original.mean(0)
            numpy_C_diagnostic = float(np.median(mu)-mu.min())
            assert official_scalar(original) == r['score_float32'] == replay['original']['C']
            sr, pcm = wavfile.read(r['scoring_audio_path'])
            assert sr == 16000 and pcm.ndim == 1 and pcm.dtype == np.int16
            assert np.array_equal(python_speech_features.mfcc(pcm, sr).T.astype(np.float32), z['mfcc'])
        assert n == r['rows'] == replay['windows']
        assert n == min(replay['frame_count'], len(pcm)//640)-5
        sr0, canonical = wavfile.read(r['canonical_audio_path'])
        assert sr0 == 16000 and canonical.ndim == 1 and canonical.dtype == np.int16
        x, y = pcm.astype(np.float64), canonical.astype(np.float64)
        x -= x.mean(); y -= y.mean()
        cor = correlate(x, y, mode='full', method='fft')
        lagv = correlation_lags(len(x), len(y)); region = np.abs(lagv) <= 4000
        lag = int(lagv[region][cor[region].argmax()])
        assert lag == replay['clock']['lag_samples_scoring_minus_canonical']
        lags[key] = lag/16000
        centers[key] = []
        assert len(clocks[key]['rows']) == n and clocks[key]['delay_s'] == lag/16000
        for i, w in enumerate(clocks[key]['rows']):
            lo, hi = max(0, 640*i-1), 640*i+3439
            assert w['row'] == i and w['raw_samples_inclusive'] == [lo, hi] and hi < len(pcm)
            assert w['mfcc_columns_halfopen'] == [4*i, 4*i+20]
            assert w['video_frames_halfopen'] == [i, i+5]
            center = (lo+hi)/32000
            assert abs(w['score_audio_center_s']-center) < 1e-12
            centers[key].append(center)
        fa = r['FA']; check(fa['path'], fa['sha256'])
        if fa['path'] not in fa_cache:
            obj = read(fa['path']); fa_cache[fa['path']] = {f['id']: f for f in obj.get('rows', obj.get('audio', []))}
        f = fa_cache[fa['path']][fa['id']]
        assert f['audio_sha256'] == r['canonical_audio_sha256'] and f['text'] == r['text']
        aligned[key] = f
        # Read back the current canonical normalization, preserving the exact
        # full waveform and PCM quantizer rather than introducing a new rule.
        audio_manifest = Path(r['canonical_audio_path']).parents[2]/'audio_manifest.json'
        if str(audio_manifest) not in audio_cache:
            audio_cache[str(audio_manifest)] = {f['path']: f for f in read(audio_manifest)['records']}
        meta = audio_cache[str(audio_manifest)][r['canonical_audio_path']]
        fp = Path(r['canonical_audio_path']).with_name(meta['role']+'_16k_float.wav')
        raw, rate = sf.read(fp, dtype='float32')
        assert rate == 16000 and raw.shape == canonical.shape and np.isfinite(raw).all()
        rms = float(np.sqrt(np.mean(raw.astype(float)**2)))
        gain = .0376838172675746/rms
        assert float(abs(raw).max()*gain) < 1
        assert gain == meta['gain'] and rms == meta['rms_before']
        import io
        buffer = io.BytesIO()
        sf.write(buffer, raw*np.float32(gain), 16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        reconstructed, _ = sf.read(buffer, dtype='int16')
        assert np.array_equal(reconstructed, canonical)
        details.append(dict(id=key, windows=n, max_numpy_curve_error=maximum, numpy_C_reduction_delta=numpy_C_diagnostic-r['score_float32']))
    patches = read(patch_path)
    valid, seen = defaultdict(set), set()
    for p in patches['patches']:
        key, ci = p['id'], p['character_index']
        assert (key, ci) not in seen; seen.add((key, ci))
        f = aligned[key]
        assert p['character'] == f['tokens'][ci] and p['canonical_interval_ms'] == f['timestamps_ms'][ci]
        structural = not f.get('issues') and ''.join(f['tokens']) == f['text'] and len(f['tokens']) == len(f['timestamps_ms'])
        for shift, v in zip((-.04, 0., .04), p['variants']):
            anchor, i, support = nearest_window(f['timestamps_ms'][ci], lags[key], shift, centers[key])
            support = support and structural
            assert v['shift_s'] == shift and v['nearest_row'] == i
            assert abs(v['mapped_anchor_s']-anchor) < 1e-12
            assert abs(v['nearest_center_s']-centers[key][i]) < 1e-12
            assert abs(v['center_error_ms']-(centers[key][i]-anchor)*1000) < 1e-8
            assert v['support_ok'] == support
            assert v['patch_rows'] == (list(range(i-5, i+6)) if support else None)
            assert v['patch_raw_samples_inclusive'] == ([max(0, (i-5)*640-1), (i+5)*640+3439] if support else None)
            if support and shift == 0:
                valid[key].add(ci)
    assert len(seen) == sum(min(len(f['tokens']), len(f['timestamps_ms'])) for f in aligned.values())
    expected = set.union(*(expected_pairs(g) for g in group.values()))
    actual = [(p['family'], p['a_id'], p['b_id']) for p in patches['pair_coverage']]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    for p in patches['pair_coverage']:
        assert p['common_character_indices'] == sorted(valid[p['a_id']] & valid[p['b_id']])
        den = len(aligned[p['a_id']]['text'])
        assert p['temporary_measurability_gate'] == (len(p['common_character_indices']) >= 5 and len(p['common_character_indices'])/den >= .8)
    for s in patches['source_common_support']:
        common = set.intersection(*(valid[r['id']] for r in group[s['source']]))
        assert s['common_character_indices'] == sorted(common) and s['common_7arm_anchors'] == len(common)
        assert s['common_7arm_gate'] == (len(common) >= 5 and len(common)/s['characters'] >= .8)
    for c in patches['coverage']:
        assert c['valid_anchors'] == len(valid[c['id']])
        assert c['temporary_measurability_gate'] == (len(valid[c['id']]) >= 5 and len(valid[c['id']])/c['characters'] >= .8)
    completion = manifest.parents[1]/'completion.json'
    assert read(completion)['status'] == 'COMPLETE_FEATURE_FA_WINDOW_DATA_ONLY'
    return dict(status='PASS_PARENT_BATCH_REPLAY_CLOCK_PATCH', clips=len(rows), sources=len(group),
                batch_id=manifest.parents[2].name, source_ids=sorted(group),
                manifest_path=str(manifest), clock_path=str(clock_path), patch_path=str(patch_path),
                consumer_completion_path=str(completion), consumer_completion_sha256=digest(completion),
                windows=sum(r['rows'] for r in rows), pairs=len(actual), anchors=len(seen),
                hashes_read=len(checked_files), canonical_PCM_exact=True, all_MFCC_exact=True,
                original_curve_exact=True, FFT_clock_exact=True, no_new_GPU=True, no_training=True,
                FA_truth='UNVERIFIED', manifest_sha256=digest(manifest), clock_sha256=digest(clock_path),
                patch_sha256=digest(patch_path), auditor_source_sha256=digest(__file__), auditor_source_path=str(Path(__file__).resolve()),
                official_scalar_reduction='torch_float32_from_original_saved_field',
                max_numpy_C_reduction_delta=max(abs(d['numpy_C_reduction_delta']) for d in details),
                elapsed_s=time.monotonic()-started, items=details)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--clock', type=Path, required=True)
    p.add_argument('--patch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists(), 'preserve old audit; choose a new output'
    result = audit(args.manifest, args.clock, args.patch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'items'}))
