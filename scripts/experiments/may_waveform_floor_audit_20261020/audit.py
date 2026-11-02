"""Replay sealed May gain inputs and official AVE frontend on CPU only.

Never imports a neural network or invokes a training/render/scoring process.
"""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import sys
import tarfile
from pathlib import Path

if '--deps' in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index('--deps')+1])

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

COMMIT = '43079fb3f349675f6cb33031bb07262f7566cdd5'
NAMES = ['load_wav', 'preemphasis', 'melspectrogram', '_stft', '_linear_to_mel',
         '_build_mel_basis', '_amp_to_db', '_normalize']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if len(starts) else 0


def waveform_stats(reference, altered, gain):
    if reference.shape != altered.shape:
        raise ValueError('time axis changed')
    restored = altered.astype(np.float64) / gain
    difference = restored - reference.astype(np.float64)
    return dict(samples=len(reference), original_zero_points=int((reference == 0).sum()),
                resulting_zero_points=int((altered == 0).sum()),
                new_zero_points=int(((reference != 0) & (altered == 0)).sum()),
                longest_zero_run_samples=longest_run(altered == 0),
                restored_max_abs_error=float(abs(difference).max()),
                restored_relative_l2=float(np.linalg.norm(difference) /
                                          np.linalg.norm(reference.astype(np.float64))),
                restored_exact=bool(np.array_equal(restored, reference.astype(np.float64))),
                minimum_nonzero_abs=float(abs(altered[altered != 0]).min()))


def frontend(namespace, x):
    pre = namespace['preemphasis'](x, .97)
    amp = namespace['_linear_to_mel'](abs(namespace['_stft'](pre)))
    db = namespace['_amp_to_db'](amp) - 20
    mapped = namespace['_normalize'](db)
    assert np.array_equal(mapped, namespace['melspectrogram'](x))
    return amp, db, mapped


def clip_masks(original, low, mapped_original, mapped_low, floor):
    original_floor, low_floor = original <= floor, low <= floor
    original_clip, low_clip = mapped_original == -4, mapped_low == -4
    return dict(original_log_floor=original_floor, low_log_floor=low_floor,
                newly_log_floored=(~original_floor) & low_floor,
                original_lower_clip=original_clip, low_lower_clip=low_clip,
                newly_lower_clipped=(~original_clip) & low_clip)


def load_official(parent, qualification, out):
    archive = parent / 'remote/runtime_sources_snapshot.tar.gz'
    inventory = json.loads((parent / 'remote/runtime_source_inventory_during_training.json').read_text())
    assert inventory['official_commit'] == COMMIT
    manifest = json.loads((parent / 'delivery_manifest.json').read_text())
    # The full delivery manifest also binds the historical source snapshot.
    candidates = manifest.get('files', manifest.get('members', []))
    sealed = next(r for r in candidates if r.get('path', r.get('relative_path', '')) in
                  ['remote/runtime_sources_snapshot.tar.gz','runtime_sources_snapshot.tar.gz'])
    assert sha(archive) == sealed['sha256']
    with tarfile.open(archive) as t:
        source = t.extractfile('nerf_triplane/utils.py').read()
    expected = next(r for r in inventory['files'] if r['path'] == 'nerf_triplane/utils.py')
    assert hashlib.sha256(source).hexdigest() == expected['sha256']
    (out / 'official_utils.py').write_bytes(source)
    text = source.decode('utf-8')
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {n.name for n in nodes} == set(NAMES)
    provenance = []
    for node in nodes:
        h = hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
        old = next(r for r in qualification['provenance'] if r['symbol'] == node.name)
        assert h == old['source_sha256'], node.name
        provenance.append(dict(symbol=node.name, line=node.lineno, sha256=h))
    scope = dict(np=np, librosa=librosa, signal=signal)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(out/'official_utils.py'), 'exec'), scope)
    return scope, dict(archive=str(archive), archive_sha256=sha(archive),
                       utils_sha256=expected['sha256'], symbols=provenance)


def run(args):
    out, data, parent = args.output, args.data, args.parent
    out.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(out).free > 1024**3
    for name in ['arrays', 'replayed_audio', 'sources']:
        (out/name).mkdir()
    manifest = json.loads((data/'manifest.json').read_text())
    protocol = json.loads((data/'protocol.json').read_text())
    training = json.loads((parent/'protocol.json').read_text())
    assert sha(data/'protocol.json') == manifest['protocol_sha256'] == training['data_protocol_sha256']
    qualification = json.loads((data/'features_attempt03/provider_qualification.json').read_text())
    assert qualification['official_commit'] == training['official_commit'] == COMMIT
    official, provenance = load_official(parent, qualification, out/'sources')
    prepare_path = args.repo/'scripts/experiments/synctalk_gain_training_data_20261017/prepare.py'
    spec = importlib.util.spec_from_file_location('historical_prepare', prepare_path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    shutil.copyfile(prepare_path, out/'sources/historical_prepare.py')
    shutil.copyfile(Path(__file__), out/'sources/audit.py')
    for name, path in [('data_manifest',data/'manifest.json'),('data_protocol',data/'protocol.json'),
                       ('qualification',data/'features_attempt03/provider_qualification.json'),
                       ('training_protocol',parent/'protocol.json')]:
        shutil.copyfile(path, out/f'sources/{name}.json')
    original_path = data/'source/May/aud.wav'
    source_expected = next(r for r in manifest['source_entries'] if r['path'] == 'source/May/aud.wav')
    assert sha(original_path) == source_expected['sha256']
    pcm, sr = prepare.read_pcm16(original_path)
    assert sr == 16000 and pcm.shape[1] == 2 and np.array_equal(pcm[:,0],pcm[:,1])
    pcm = pcm[:,0]
    assert digest(pcm) == manifest['source_pcm_sha256']
    results = dict(status='RUNNING', scope='CPU waveform and AVE frontend only; no training',
                   official_commit=COMMIT, source=str(original_path), source_sha256=sha(original_path),
                   provenance=provenance, historical_prepare_sha256=sha(prepare_path),
                   environment=dict(python=sys.version, platform=platform.platform(), versions={
                       k:importlib.metadata.version(k) for k in ['numpy','scipy','soundfile','librosa']}),
                   historical_environment=qualification['environment'],
                   neural_forwards=0, training_updates=0, renders=0, syncnet_calls=0,
                   waveform=[], postnorm=[], spectral=[], input_pins=[])
    gain = 10**(-6/20)
    floor = float(np.exp(-5*np.log(10)))
    results['thresholds'] = dict(log_floor_amplitude=floor,
        lower_clip_equivalent_amplitude=1e-4, newly_lower_clipped_expected_original_band=[1e-4,1e-4/gain],
        gain_db=-6, gain=gain, mapped_shift=-.48,
        energy_strata='within each split, 50ms centered waveform RMS: <=q20, q20-q80, >q80; not VAD/noise labels')
    frequency_rows, time_rows, strata_rows, examples = [], [], [], []
    post_root = args.postnorm
    receipts = {e:json.loads((post_root/f'features/{e}/receipt.json').read_text())
                for e in ['hubert_base','wav2vec2_base']}
    for split in manifest['splits']:
        name = split['name']
        start = split['source_start_frame']*640
        n = split['audio_samples']
        base64 = pcm[start:start+n].astype(np.float64)/32768
        stage = {}
        for db, label in [(0,'gain_0'),(-6,'gain_m6'),(6,'gain_p6')]:
            path = data/f'arms/{name}/{label}/aud.wav'
            arm = next(r for r in manifest['arms'] if r['split']==name and r['gain_db']==db)
            q = next(r for r in qualification['arms'] if r['split']==name and r['gain_db']==db)
            assert sha(path) == arm['audio_sha256'] == q['audio_sha256']
            values, rate = sf.read(path, dtype='float64')
            assert rate == sr and sf.info(path).subtype == 'DOUBLE'
            g = 10**(db/20)
            assert np.array_equal(values, base64*g)
            replay = out/f'replayed_audio/{name}_{label}_double.wav'
            prepare.write_double(replay,base64*g,sr)
            assert sha(replay) == sha(path), 'byte-level historical writer replay mismatch'
            decoded = official['load_wav'](str(path),sr)
            assert np.array_equal(decoded, values.astype(np.float32))
            row = dict(split=name, path='historical_DOUBLE', gain_db=db,
                       **waveform_stats(base64,values,g),
                       decoded_new_zero_points=int(((base64!=0)&(decoded==0)).sum()),
                       decoded_restored_max_abs_error=float(abs(decoded.astype(float)/g-base64).max()),
                       decoded_minimum_nonzero_abs=float(abs(decoded[decoded!=0]).min()))
            results['waveform'].append(row)
            results['input_pins'].append(dict(path=str(path), sha256=sha(path), replay_sha256=sha(replay)))
            amp, logdb, mapped = frontend(official, decoded)
            stage[db] = (decoded, amp, logdb, mapped)
            for key,value in [('decoded',decoded),('mel_amplitude',amp),('db_minus20',logdb),('mapped_mel',mapped)]:
                np.save(out/f'arrays/{name}_{label}_{key}.npy',value)
        base, amp0, db0, f0 = stage[0]
        low, amp_low, db_low, flow = stage[-6]
        masks = clip_masks(amp0,amp_low,f0,flow,floor)
        for key,mask in masks.items():
            np.save(out/f'arrays/{name}_{key}.npy',mask)
        newly = masks['newly_lower_clipped']
        inside = (f0>-4)&(f0<4)&(flow>-4)&(flow<4)
        restored_audio = (low.astype(np.float64)/gain).astype(np.float32)
        _,_,rerun = frontend(official,restored_audio)
        spectral = dict(split=name, mel_shape=list(amp0.shape), cells=amp0.size,
            source_start_seconds=start/sr, source_end_seconds=(start+n)/sr,
            **{k:int(v.sum()) for k,v in masks.items()},
            original_upper_clip=int((f0==4).sum()),low_upper_clip=int((flow==4).sum()),
            newly_clipped_original_amp_min=float(amp0[newly].min()) if newly.any() else None,
            newly_clipped_original_amp_max=float(amp0[newly].max()) if newly.any() else None,
            nonclipped_shift_max_error=float(abs((flow-f0)[inside]+.48).max()),
            waveform_restored_then_frontend_max_error=float(abs(rerun-f0).max()),
            add_gain_back_after_clipping_max_error=float(abs((flow+.48)-f0).max()),
            newly_clipped_restored_feature_error_mean=float(abs((flow+.48-f0)[newly]).mean()) if newly.any() else 0,
            total_mel_amplitude_fraction_in_newly_clipped_cells=float(amp0[newly].sum()/amp0.sum()),
            newly_clipped_frames=int(newly.any(axis=0).sum()))
        results['spectral'].append(spectral)
        # Time-local energy is an unlabeled stratum, never a claim of speech absence.
        padded = np.pad(base.astype(float)**2,(400,400),mode='reflect')
        sums = np.r_[0.,np.cumsum(padded)]
        centers = np.arange(amp0.shape[1])*200
        energy = np.sqrt((sums[centers+800]-sums[centers])/800)
        lo,hi = np.quantile(energy,[.2,.8])
        strata = np.where(energy<=lo,0,np.where(energy>hi,2,1))
        for k,label in enumerate(['low20','middle60','high20']):
            selected = strata==k
            strata_rows.append(dict(split=name,stratum=label,frames=int(selected.sum()),
                cells=int(selected.sum()*80),newly_clipped=int(newly[:,selected].sum()),
                new_clip_fraction=float(newly[:,selected].mean()),rms_min=float(energy[selected].min()),
                rms_max=float(energy[selected].max())))
        melhz=librosa.mel_frequencies(n_mels=82,fmin=55,fmax=7600)
        for band in range(80):
            frequency_rows.append(dict(split=name,mel_band=band,center_hz=float(melhz[band+1]),
                left_hz=float(melhz[band]),right_hz=float(melhz[band+2]),
                frames=amp0.shape[1],original_lower_clip=int(masks['original_lower_clip'][band].sum()),
                low_lower_clip=int(masks['low_lower_clip'][band].sum()),newly_lower_clipped=int(newly[band].sum()),
                newly_log_floored=int(masks['newly_log_floored'][band].sum())))
        for t in range(amp0.shape[1]):
            time_rows.append(dict(split=name,mel_frame=t,split_seconds=t*.0125,
                source_seconds=start/sr+t*.0125,original_50ms_rms=float(energy[t]),
                energy_stratum=['low20','middle60','high20'][strata[t]],newly_lower_clipped=int(newly[:,t].sum()),
                newly_log_floored=int(masks['newly_log_floored'][:,t].sum())))
        # Deterministically select a 4s window with the most changed cells, for localization only.
        counts=newly.sum(axis=0);width=min(320,len(counts))
        begin=int(np.argmax(np.convolve(counts,np.ones(width,dtype=int),mode='valid')))
        examples.append(dict(split=name,start_mel_frame=begin,end_mel_frame=begin+width,
            source_start_seconds=start/sr+begin*.0125,source_end_seconds=start/sr+(begin+width)*.0125,
            selection='maximum newly-clipped count over contiguous 4s; not representative or a speech/noise label'))
        # Explicit PCM16 diagnostics are not historical pure-gain arms. -12dB checks sensitivity.
        for db in [-6,-12]:
            g=10**(db/20)
            p=out/f'replayed_audio/{name}_pcm16_control_{db}.wav'
            sf.write(p,base64*g,sr,subtype='PCM_16')
            y,rate=sf.read(p,dtype='float64');assert rate==sr
            results['waveform'].append(dict(split=name,path='PCM16_control_NOT_historical',gain_db=db,
                                           **waveform_stats(base64,y,g)))
        native=(base-base.mean())/np.sqrt(base.var()+1e-7)
        for encoder,r in receipts.items():
            for scale in [.5,1.] + ([2.] if name=='test' else []):
                c=next(c for c in r['conditions'] if c['split']==name and c['input_scale']==scale)
                actual=(native*np.float32(scale)).astype(np.float32)
                assert digest(actual)==c['processor_sha256'] and digest(native)==c['native_processor_sha256']
                assert c['source_audio_sha256']==next(x['audio_sha256'] for x in manifest['arms']
                                                    if x['split']==name and x['gain_db']==0)
                results['postnorm'].append(dict(split=name,encoder=encoder,scale=scale,
                    historical_input_sha256=c['processor_sha256'],
                    **waveform_stats(native,actual,scale)))
    results.update(status='COMPLETED',examples=examples)
    write(out/'results.json',results)
    csv_write(out/'waveform.csv', [{k:r.get(k) for k in sorted(set().union(*(r.keys() for r in results['waveform'])))}
                                   for r in results['waveform']])
    csv_write(out/'spectral_by_frequency.csv',frequency_rows)
    csv_write(out/'spectral_by_time.csv',time_rows)
    csv_write(out/'spectral_by_energy_stratum.csv',strata_rows)
    print(json.dumps(dict(status=results['status'],spectral=results['spectral'],
                         historical_new_zeros=[r['new_zero_points'] for r in results['waveform']
                                               if r['path']=='historical_DOUBLE']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--postnorm',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deps',type=Path)
    args=parser.parse_args()
    # Dependency path must be applied before imports by the calling launcher.
    run(args)
