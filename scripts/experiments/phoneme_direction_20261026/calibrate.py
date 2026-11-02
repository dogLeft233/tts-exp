"""Fit-only calibration. Never reads scores or opens development/confirm media."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/phoneme_direction_20261026_attempt01'
REVISION = 'c616845c4e309e24d5927f15adbdf277a3d65358'
MFA = Path('/home/wjj/miniconda3/envs/mfa3/bin/mfa')
MODELS = Path('/home/wjj/Documents/MFA/mfa3_chinese_rerun_20260815/pretrained_models')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Receipts are immutable. A failed attempt needs a new run, not overwritten evidence.
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def select_calibration(rows):
    if len(rows) != 1200 or any(x['split'] != 'fit' for x in rows):
        raise ValueError('Expected exactly 1200 fit units')
    ids = [x['id'] for x in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate fit unit')
    speakers = sorted({x['speaker_id'] for x in rows})
    if len(speakers) != 100:
        raise ValueError('Expected 100 fit speakers')
    sources = defaultdict(list)
    for x in rows:
        sources[x['sample_id']].append(x)
    if len(sources) != 300:
        raise ValueError('Expected 300 fit sources')
    for group in sources.values():
        if Counter((x['arm'], int(x['draw'])) for x in group) != Counter({('N', 0): 1, ('L17', 1): 1, ('L17', 2): 1, ('L17', 3): 1}):
            raise ValueError('Incomplete four-arm source')
        if len({x['text'] for x in group}) != 1 or len({x['speaker_id'] for x in group}) != 1:
            raise ValueError('Source identity mismatch')
    chosen = {min(x['sample_id'] for x in rows if x['speaker_id'] == s) for s in speakers[::5]}
    return sorted((x for x in rows if x['sample_id'] in chosen), key=lambda x: x['id'])


def normalize_text(text):
    # All frozen transcripts must already be spoken Chinese; do not silently invent readings.
    chars = re.findall(r'[\u3400-\u9fff]', text)
    rest = re.sub(r'[\u3400-\u9fff\s，。！？、；：“”‘’（）,.!?;:\-]', '', text)
    if rest or not chars:
        raise ValueError(f'Transcript needs explicit normalization: {text!r}')
    return ' '.join(chars)


def prepare():
    import soundfile as sf
    p = ROOT / 'runs/audio200_nl17_production_20261025_attempt01'
    rows = read(p / 'reused248_manifest.json')['rows']
    origins = [p / 'reused248_manifest.json']
    audit = ROOT / 'runs/audio200_nl17_parent_gate_20261025_attempt01/batch_audits'
    # Producer audit manifests provide identity only. Score fields are not retained or used.
    for f in sorted(audit.glob('*.json')):
        d = read(f)
        m = d.get('manifest_path')
        if m and Path(m).exists():
            rows.extend(read(m).get('rows', []))
            origins.extend([f, Path(m)])
    by_id = {}
    keys = ['id', 'sample_id', 'speaker_id', 'arm', 'draw', 'text', 'split', 'canonical_audio_path', 'canonical_audio_sha256']
    for x in rows:
        if x.get('split') != 'fit':
            continue
        y = {k: x[k] for k in keys}
        if y['id'] in by_id and by_id[y['id']] != y:
            raise ValueError('Conflicting fit unit')
        by_id[y['id']] = y
    fit = sorted(by_id.values(), key=lambda x: x['id'])
    cal = select_calibration(fit)
    for x in fit:
        f = Path(x['canonical_audio_path'])
        if sha(f) != x['canonical_audio_sha256']:
            raise ValueError(f'Audio SHA mismatch {x["id"]}')
        info = sf.info(f)
        if info.samplerate != 16000 or info.channels != 1 or info.frames < 1024:
            raise ValueError(f'Invalid canonical PCM {x["id"]}')
        x['samples'] = info.frames
    # Resolve back to annotated fit records.
    cal_ids = {x['id'] for x in cal}
    cal = [x for x in fit if x['id'] in cal_ids]
    knn = ROOT / 'third_party/knn-vc'
    rev = subprocess.check_output(['git', '-C', str(knn), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(knn), 'status', '--porcelain'], text=True).strip()
    if rev != REVISION or dirty:
        raise ValueError('kNN-VC source must match clean pinned revision')
    checkpoint = Path.home() / '.cache/torch/hub/checkpoints'
    pins = [checkpoint / 'WavLM-Large.pt', checkpoint / 'prematch_g_02500000.pt',
            knn / 'matcher.py', knn / 'hubconf.py', knn / 'wavlm/WavLM.py', knn / 'hifigan/models.py',
            MODELS / 'dictionary/mandarin_china_mfa.dict', MODELS / 'acoustic/mandarin_mfa.zip', Path(__file__)]
    write(RUN / 'input_manifest.json', {'fit': fit, 'calibration': cal, 'selection': 'speaker sorted indices 0,5,...95; first source ID',
        'origins': {str(f): sha(f) for f in set(origins)}, 'pins': {str(f): sha(f) for f in pins},
        'knn_revision': rev, 'fit_units': len(fit), 'fit_sources': 300, 'fit_speakers': 100,
        'confirm_media_opened': False, 'dev_effects_read': False})
    # Fixed technical floors precede reconstruction. They cannot be relaxed to pass.
    write(RUN / 'calibration_contract.json', {
        'alpha': 0.25, 'arms': ['A','R','M+','D+','J+','M-','D-','J-','M_r','D_r','J_r'],
        'primary': 'C(A,G(J+))-C(A,G(A))', 'device': 'cpu', 'threads': 4,
        'right_input_zero_pad': 'minimal to ceil(N/320) frames, convolution RF=400; crop decoded right to original N',
        'frame_centers': '(199.5+320*t)/16000; centers outside original PCM are unsupported',
        'reconstruction_floors': {'stoi_median_min': 0.85, 'stoi_each_min': 0.75, 'abs_global_envelope_lag_ms_max': 20,
                                  'clipped_samples_max': 0, 'relative_rms_error_max': 0.001},
        'alignment_floors': {'missing_grids_max': 0, 'unknown_phones_max': 0, 'nonmonotonic_max': 0},
        'numeric_protocol_frozen': False, 'development_allowed': False, 'confirmation_allowed': False})
    for x in cal:
        d = RUN / 'mfa_corpus' / x['speaker_id']
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(x['canonical_audio_path'], d / (x['id'] + '.wav'))
        (d / (x['id'] + '.lab')).write_text(normalize_text(x['text'])+'\n', encoding='utf8')
    print(json.dumps({'fit_units': len(fit), 'calibration_units': len(cal), 'status': 'PREPARED'}, ensure_ascii=False), flush=True)


def reconstruct():
    import numpy as np
    import soundfile as sf
    import torch
    sys.path.insert(0, str(ROOT / 'scripts'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    torch.set_num_threads(4)
    torch.manual_seed(20261026)
    torch.use_deterministic_algorithms(True)
    manifest = read(RUN / 'input_manifest.json')
    for path, expected in manifest['pins'].items():
        if sha(path) != expected:
            raise ValueError(f'Changed pinned input {path}')
    adapter = WavLMKNNVCAdapter.load_pretrained(device='cpu', source=ROOT / 'third_party/knn-vc')
    # Confirm the actual fast path, layer and convolution clock, not just tensor shape.
    import matcher
    if matcher.SPEAKER_INFORMATION_LAYER != 6:
        raise ValueError('Unexpected upstream layer')
    # Config is upstream Python arithmetic; inspect actual conv modules instead.
    conv = []
    for layer in adapter.model.wavlm.feature_extractor.conv_layers:
        c = layer[0]
        conv.append((int(c.kernel_size[0]), int(c.stride[0]), int(c.padding[0])))
    rf, hop = 1, 1
    for kernel, stride, padding in conv:
        if padding:
            raise ValueError('Unexpected frontend padding')
        rf += (kernel-1)*hop
        hop *= stride
    if (rf, hop) != (400, 320):
        raise ValueError(f'Unknown frame clock {rf}/{hop}')
    write(RUN / 'encoder_contract.json', {'adapter': adapter.metadata(), 'conv': conv, 'receptive_field_samples': rf,
          'stride_samples': hop, 'center0_samples': (rf-1)/2, 'torch': torch.__version__,
          'input_normalization': 'upstream get_features tensor path, no RMS normalization, no VAD',
          'pad_policy': read(RUN/'calibration_contract.json')['right_input_zero_pad']})
    selected = [x for x in manifest['calibration'] if x['arm'] == 'N']
    rows = []
    for x in selected:
        started = time.time()
        a, sr = sf.read(x['canonical_audio_path'], dtype='float32')
        frames = (len(a)+319)//320
        padded = np.pad(a, (0, frames*320+80-len(a)))
        h = adapter.extract(torch.from_numpy(padded))
        if tuple(h.shape) != (frames, 1024):
            raise ValueError('Feature timebase mismatch')
        raw = adapter.vocode(h).numpy()
        if len(raw) != frames*320:
            raise ValueError('Unexpected vocoder output length')
        raw = raw[:len(a)].astype('float64')
        target = float(np.sqrt(np.mean(a.astype('float64')**2)))
        rms = float(np.sqrt(np.mean(raw**2)))
        if not rms > 0:
            raise ValueError('Silent reconstruction')
        b = raw*(target/rms)
        out = RUN/'reconstruction'/f'{x["id"]}.wav'
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise FileExistsError(out)
        sf.write(out, b, sr, subtype='FLOAT')
        np.save(RUN/'reconstruction'/f'{x["id"]}_features.npy', h.cpu().numpy(), allow_pickle=False)
        row = {'id': x['id'], 'speaker_id': x['speaker_id'], 'source': x['canonical_audio_path'],
               'source_sha256': x['canonical_audio_sha256'], 'output': str(out), 'output_sha256': sha(out),
               'samples': len(a), 'frames': frames, 'right_pad_samples': len(padded)-len(a),
               'source_rms': target, 'peak': float(np.max(np.abs(b))), 'clipped_samples': int(np.sum(np.abs(b)>=1)),
               'elapsed_s': time.time()-started}
        write(RUN/'reconstruction'/f'{x["id"]}.json', row)
        rows.append(row)
        print(json.dumps(row), flush=True)
    write(RUN/'reconstruction_complete.json', {'rows': rows, 'count': len(rows), 'status': 'COMPLETE_NOT_YET_QUALIFIED'})


def quality():
    import numpy as np
    import soundfile as sf
    from pystoi import stoi
    from scipy.signal import correlate, correlation_lags
    floors = read(RUN/'calibration_contract.json')['reconstruction_floors']
    results = []
    for x in read(RUN/'reconstruction_complete.json')['rows']:
        if sha(x['source']) != x['source_sha256'] or sha(x['output']) != x['output_sha256']:
            raise ValueError('Changed waveform')
        a, sr = sf.read(x['source']); b, sr2 = sf.read(x['output'])
        if sr != 16000 or sr != sr2 or len(a) != len(b):
            raise ValueError('Waveform contract mismatch')
        n = len(a)//160
        ea = np.sqrt(np.mean(a[:n*160].reshape(n,160)**2,axis=1))
        eb = np.sqrt(np.mean(b[:n*160].reshape(n,160)**2,axis=1))
        corr = correlate(eb-eb.mean(),ea-ea.mean(),method='fft')
        lags = correlation_lags(len(eb),len(ea))
        mask = abs(lags)<=20
        lag = int(lags[mask][np.argmax(corr[mask])])*10
        rms_error = abs(np.sqrt(np.mean(b*b))/np.sqrt(np.mean(a*a))-1)
        score = float(stoi(a,b,sr,extended=False))
        failures = []
        if score < floors['stoi_each_min']: failures.append('stoi')
        if abs(lag)>floors['abs_global_envelope_lag_ms_max']: failures.append('lag')
        if np.sum(abs(b)>=1)>floors['clipped_samples_max']: failures.append('clipping')
        if rms_error>floors['relative_rms_error_max']: failures.append('rms')
        results.append({'id':x['id'],'stoi':score,'global_envelope_lag_ms':lag,'rms_relative_error':float(rms_error),'failures':failures})
    median = float(np.median([x['stoi'] for x in results]))
    passed = len(results)==20 and median>=floors['stoi_median_min'] and all(not x['failures'] for x in results)
    payload = {'status':'PASS_BASIC_RECONSTRUCTION' if passed else 'HOLD_RECONSTRUCTION', 'rows':results,
               'stoi_median':median,'development_allowed':False,'confirmation_allowed':False,
               'limits':'STOI and envelope lag do not verify ASR content, phoneme events, or perceptual quality; further gates required'}
    write(RUN/'reconstruction_quality.json',payload)
    print(json.dumps(payload),flush=True)


def align():
    dictionary = MODELS/'dictionary/mandarin_china_mfa.dict'
    acoustic = MODELS/'acoustic/mandarin_mfa.zip'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MFA_ROOT_DIR=str(RUN/'mfa_runtime'), TMPDIR=str(RUN/'tmp'))
    (RUN/'tmp').mkdir(parents=True,exist_ok=True)
    version = subprocess.check_output([str(MFA),'version'],env=env,text=True).strip()
    if version != '3.4.1': raise ValueError(f'Wrong MFA version {version}')
    command = [str(MFA),'align',str(RUN/'mfa_corpus'),str(dictionary),str(acoustic),str(RUN/'mfa_grids'),
               '--clean','--num_jobs','2']
    write(RUN/'mfa_launch.json',{'command':command,'version':version,'model_pins':{str(p):sha(p) for p in [dictionary,acoustic]},
          'isolated_runtime':env['MFA_ROOT_DIR'],'source_manifest_sha256':sha(RUN/'input_manifest.json')})
    with (RUN/'mfa.log').open('x') as log:
        result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
    write(RUN/'mfa_exit.json',{'returncode':result.returncode})
    if result.returncode: raise RuntimeError('MFA failed, retain log and stop')


def audit_alignment():
    from praatio import textgrid
    rows = read(RUN/'input_manifest.json')['calibration']
    parsed = []; failures = []
    for x in rows:
        grid_path = RUN/'mfa_grids'/x['speaker_id']/(x['id']+'.TextGrid')
        if not grid_path.is_file():
            failures.append({'id':x['id'],'error':'missing_grid'}); continue
        grid = textgrid.openTextgrid(str(grid_path),includeEmptyIntervals=True)
        phone_name = next((n for n in grid.tierNames if n in ['phones','phone']),None)
        word_name = next((n for n in grid.tierNames if n in ['words','word']),None)
        if not phone_name or not word_name: raise ValueError('Expected word and phone tiers')
        phones = [dict(start=float(a),end=float(b),label=c) for a,b,c in grid.getTier(phone_name).entries]
        words = [dict(start=float(a),end=float(b),label=c) for a,b,c in grid.getTier(word_name).entries if c and c not in ['sil','sp']]
        unknown = [p for p in phones if p['label'].lower() in ['spn','<unk>','unk']]
        if unknown: failures.append({'id':x['id'],'error':'unknown_speech','count':len(unknown)})
        if any(p['end']<=p['start'] or p['start']<0 or p['end']>x['samples']/16000+0.001 for p in phones):
            failures.append({'id':x['id'],'error':'invalid_boundary'})
        if any(a['end']>b['start']+1e-6 for a,b in zip(phones,phones[1:])):
            failures.append({'id':x['id'],'error':'nonmonotonic'})
        expected = normalize_text(x['text']).split()
        if [w['label'] for w in words] != expected:
            failures.append({'id':x['id'],'error':'word_sequence_mismatch','actual':[w['label'] for w in words],'expected':expected})
        # Key each phone by transcript character position and local phone occurrence.
        tokens = []
        for pos,w in enumerate(words):
            contained = [p for p in phones if p['label'] not in ['','sil','sp'] and w['start']-1e-5 <= (p['start']+p['end'])/2 <= w['end']+1e-5]
            for j,p in enumerate(contained): tokens.append(dict(p,text_position=pos,phone_index=j))
        parsed.append({'id':x['id'],'sample_id':x['sample_id'],'speaker_id':x['speaker_id'],'arm':x['arm'],
                       'draw':x['draw'],'phones':phones,'tokens':tokens,'grid_sha256':sha(grid_path)})
    groups = defaultdict(list)
    for x in parsed: groups[x['sample_id']].append(x)
    pairing = []
    for source,group in groups.items():
        natural = next(x for x in group if x['arm']=='N')
        nk = {(p['text_position'],p['phone_index'],p['label']) for p in natural['tokens']}
        for t in group:
            if t['arm']=='N': continue
            tk = {(p['text_position'],p['phone_index'],p['label']) for p in t['tokens']}
            pairing.append({'source':source,'tts_id':t['id'],'matched':len(nk & tk),'natural_tokens':len(nk),'tts_tokens':len(tk)})
    write(RUN/'alignment_audit.json',{'status':'PASS_STRUCTURAL_ONLY' if not failures and len(parsed)==80 else 'HOLD_ALIGNMENT',
          'units':len(parsed),'failures':failures,'pairing':pairing,'records':parsed,'time_truth_verified':False,
          'development_allowed':False,'confirmation_allowed':False})
    print(json.dumps({'aligned':len(parsed),'failures':len(failures)}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','reconstruct','quality','align','audit_alignment'])
    args=parser.parse_args()
    globals()[args.stage]()
