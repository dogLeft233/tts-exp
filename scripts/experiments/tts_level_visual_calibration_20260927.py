"""Frozen, audio-free LEVEL visual calibration; evaluation requires a separate unlock."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tts_level_visual_calibration_20260927'
AUDIT = ROOT / 'runs/tts_level_visual_audit_20260927'
OLD = ROOT / 'runs/tts_independent_visual_20260926'
CELLS = ('N_raw', 'N_LEVEL', 'T_raw', 'T_LEVEL')
VIEWS = ('native', 'frozen', 'reversed')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def disk(extra=0):
    free = shutil.disk_usage(ROOT).free
    assert free >= (5 << 30) + extra, ('disk reserve', free, extra)
    return free


def write(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    disk(len(payload.encode()) + (1 << 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(payload)
    temp.replace(path)


def freeze():
    assert not (OUT / 'protocol.json').exists()
    a = read(AUDIT / 'asset_audit.json')
    old = read(OLD / 'protocol.json')
    lookup = {(x['id'], x['arm'], x['condition']): x for x in a['videos']}
    rows = []
    for row in a['eligible']:
        rr = dict(row)
        rr['cells'] = {arm + '_' + cond: lookup[(row['id'], arm, cond)]
                       for arm in ('N', 'T') for cond in ('raw', 'LEVEL')}
        rows.append(rr)
    env = OUT / 'environment/CMLR_V_WER8.0.ini'
    env.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(old['config'], env)
    p = {
        'status': 'frozen_before_CMLR_or_MediaPipe_output', 'created_epoch': time.time(),
        'authorization': 'root authorized only full26 calibration; evaluation locked until root reviews gate',
        'audit_sha256': sha(AUDIT / 'asset_audit.json'), 'proposal_sha256': sha(AUDIT / 'proposal.md'),
        'parent_protocol_sha256': a['parent_protocol_sha256'],
        'old_visual_protocol_sha256': a['old_visual_protocol_sha256'],
        'model': old['model'], 'model_sha256': old['model_sha256'],
        'config': str(env), 'config_sha256': sha(env),
        'dependencies': a['dependency_hashes'], 'avsr_commit': a['avsr_commit'],
        'rows': rows, 'oov': a['oov'], 'cells': CELLS, 'views': VIEWS,
        'support': 'fixed26cal and66 lexical-eligible eval; all four cells common technical QC only; no SyncNet support/score filtering',
        'decoy_rule': a['decoy_pool'],
        'metric': 'M=mean5 decoy per-char CTC NLL-target per-char CTC NLL; Q=Mnative-Mfrozen; reverse=Mnative-Mreversed; target NLL and no-LM greedy CER secondary',
        'gate': {'arm': 'N_raw', 'required_complete_calibration': 26,
                 'strict_top1_fraction': .60, 'q_positive_fraction': .70,
                 'q_speaker_ci95_low_strictly_above': 0.,
                 'reverse_speaker_ci95_low_strictly_above': 0.,
                 'failure': 'stop; no evaluation; no retuning, crop/model/decoy change or cal deletion'},
        'qc': {'single_face': True, 'landmark_fraction_min': .95, 'missing_run_max': 5,
               'required_shape': '1,T,88,88', 'fps': 25, 'source': 'already audio-free FFV1 cropped224 AVI; direct official loading, no transcode',
               'frame_pts': 'all frames decoded, same sealed frame count, uniform25fps; within-arm RAW/LEVEL frame count+PTS equal'},
        'stats': {'draws': 20000, 'seed': 20260926, 'weighting': 'speaker-equal mean utterance paired differences', 'CI': [95, 99]},
        'frozen_formal_contrasts': ['LEVEL-RAW N', 'LEVEL-RAW T', '(LEVEL-RAW T)-(LEVEL-RAW N)', 'T-N RAW', 'T-N LEVEL'],
        'formal_primary': 'all ΔN/ΔT/difference-in-differences Q and raw M reported; natural content improvement claim requires ΔN Q and M both99CI>0; other contrasts descriptive',
        'storage': 'full3363 softmax computed; exact original logp columns blank+target+decoys without renormalization; all-vocab greedy IDs, full logp hash/normalization QC; firstcal all4 native full logp and repeated full logp retained',
        'repeat_tolerance': 1e-5, 'logp_normalization_tolerance': 1e-5,
        'limits': 'different source face and224 cropped geometry from old512 dynamic; known reverse/frozen controls calibrate content sensitivity, not absolute visual timing or perception; historical single-face cohort',
        'resource': {'disk_reserve_bytes': 5 << 30, 'persistent_budget_bytes': 512 << 20, 'temporary_budget_bytes': 100 << 20, 'gpu_budget_bytes': 6 << 30, 'lock': '/tmp/tts-exp-gpu.lock'},
        'code_sha256': sha(__file__),
    }
    snapshot = OUT / 'code_snapshot' / Path(__file__).name
    snapshot.parent.mkdir(exist_ok=True)
    shutil.copyfile(__file__, snapshot)
    write(OUT / 'protocol.json', p)
    write(OUT / 'seal.json', {'protocol_sha256': sha(OUT / 'protocol.json'), 'code_sha256': sha(__file__)})
    print('FROZEN', sha(OUT / 'protocol.json'), flush=True)


def resource(startup=False):
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    foreign = {int(x.strip()) for x in pids.splitlines() if x.strip().isdigit()} - {os.getpid()}
    assert not foreign, ('foreign compute', foreign)
    used, util = map(int, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
    if startup:
        assert used <= 128 and util <= 10, (used, util)
    return {'time': time.time(), 'memory_mib': used, 'utilization': util, 'free_bytes': disk(30 << 20), 'pid': os.getpid()}


def crop(pipeline, cell):
    import torch
    from scripts.experiments.vsr_tts_pilot import _face_count_qc, _frame_digest, _max_false_run, _stream_summary
    path = Path(cell['path'])
    assert sha(path) == cell['sha256']
    stream = _stream_summary(path)
    assert stream['audio_stream_count'] == 0 and stream['fps'] == 25
    framedigest, pts = _frame_digest(path)
    frames = pipeline.dataloader.load_video(str(path))
    assert len(frames) == len(pts) == cell['frames']
    assert np.allclose(np.diff(pts), 1 / 25, rtol=0, atol=1e-6)
    landmarks = pipeline.process_landmarks(str(path), None)
    valid = [x is not None for x in landmarks] if landmarks is not None else []
    if not valid:
        return None, {'status': 'no_landmarks', 'eligible': False, 'frame_count': len(frames), 'pts': pts}
    video = pipeline.dataloader.video_process(frames, list(landmarks))
    if video is None:
        return None, {'status': 'no_crop', 'eligible': False, 'frame_count': len(frames), 'pts': pts}
    x = pipeline.dataloader.video_transform(torch.tensor(video)).detach().cpu().numpy().astype(np.float32)
    assert x.shape == (1, len(frames), 88, 88) and np.isfinite(x).all()
    fq = _face_count_qc(path)
    eligible = sum(valid) / len(valid) >= .95 and _max_false_run(valid) <= 5 and fq['multi_face_frame_count'] == 0
    return x, {'status': 'complete', 'eligible': bool(eligible), 'stream': stream,
               'frame_count': len(frames), 'pts': pts, 'frame_md5_digest': framedigest,
               'decoded_RGB_sha256': digest(frames), 'tensor_shape': list(x.shape), 'tensor_sha256': digest(x),
               'landmark_valid_fraction': sum(valid) / len(valid), 'landmark_missing_max_run': _max_false_run(valid), 'face_qc': fq}


def worker():
    import torch
    from scipy.special import logsumexp
    from scripts.experiments.vsr_tts_metrics import make_views
    from scripts.experiments.vsr_tts_pilot import load_vsr, _forward_view, _set_determinism
    p = read(OUT / 'protocol.json')
    assert sha(OUT / 'protocol.json') == read(OUT / 'seal.json')['protocol_sha256']
    assert sha(__file__) == p['code_sha256']
    for path, h in p['dependencies'].items():
        assert sha(path) == h, path
    assert sha(p['model']) == p['model_sha256'] and sha(p['config']) == p['config_sha256']
    with open('/tmp/tts-exp-gpu.lock', 'a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            checks = []
            for i in range(3):
                checks.append(resource(startup=True))
                if i < 2:
                    time.sleep(5)
            write(OUT / 'resource_start.json', checks)
            torch.set_num_threads(2)
            _set_determinism(20260926)
            pipeline = load_vsr(OUT, device='cuda:0')
            assert pipeline.modality == 'video'
            write(OUT / 'environment.json', {'python': sys.version, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'pid': os.getpid(), 'time': time.time(), 'modality': pipeline.modality, 'forward': 'encode then CTC.log_softmax only; no audio or beam/LM decoding'})
            rows = [r for r in p['rows'] if r['split'] == 'calibration']
            assert len(rows) == 26
            for row in rows:
                dest = OUT / 'receipts' / (row['id'] + '.json')
                if dest.exists():
                    continue
                rr = {'id': row['id'], 'speaker': row['speaker'], 'split': row['split'], 'cells': {}}
                for key in CELLS:
                    gate = resource()
                    x, qc = crop(pipeline, row['cells'][key])
                    cell = {'qc': qc, 'features': {}, 'resource': gate}
                    if x is not None:
                        for name in VIEWS:
                            resource()
                            view = make_views(x)[name]
                            enc, lp = _forward_view(pipeline, view, 'cuda:0')
                            assert np.isfinite(lp).all() and lp.shape[0] == x.shape[1]
                            norm_error = float(np.max(np.abs(logsumexp(lp.astype(float), axis=1))))
                            assert norm_error <= 1e-5
                            columns = np.array(row['logp_columns'], dtype=np.int64)
                            path = OUT / 'features' / row['id'] / (key + '_' + name + '.npz')
                            path.parent.mkdir(parents=True, exist_ok=True)
                            disk(lp.nbytes * 3 + x.nbytes + (1 << 20))
                            np.savez_compressed(path, logp=lp[:, columns], columns=columns, greedy=lp.argmax(1).astype(np.int32))
                            feature = {'path': str(path), 'sha256': sha(path), 'full_logp_sha256': digest(lp), 'full_shape': list(lp.shape), 'normalization_max_error': norm_error, 'selected_no_renormalization': True, 'view_sha256': digest(view)}
                            if row['id'] == rows[0]['id'] and name == 'native':
                                _, repeat = _forward_view(pipeline, view, 'cuda:0')
                                err = float(np.max(np.abs(lp - repeat)))
                                assert err <= p['repeat_tolerance']
                                fullpath = path.with_name(key + '_native_full.npz')
                                np.savez_compressed(fullpath, logp=lp, repeat_logp=repeat)
                                feature.update({'full_path': str(fullpath), 'full_sha256': sha(fullpath), 'repeat_max_error': err, 'repeat_logp_sha256': digest(repeat)})
                            cell['features'][name] = feature
                    rr['cells'][key] = cell
                    print('CELL', row['id'], key, qc['eligible'], flush=True)
                for arm in ('N', 'T'):
                    left, right = [rr['cells'][arm + '_' + c]['qc'] for c in ('raw', 'LEVEL')]
                    assert left['frame_count'] == right['frame_count'] and left['pts'] == right['pts']
                rr['eligible'] = all(c['qc']['eligible'] for c in rr['cells'].values())
                write(dest, rr)
                print('COMPLETE', row['id'], rr['eligible'], flush=True)
        finally:
            write(OUT / 'gpu_release.json', {'time': time.time(), 'pid': os.getpid(), 'stage': 'calibration'})
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def cluster(values, speakers):
    x = np.asarray(values, dtype=float)
    names = sorted(set(speakers))
    means = np.array([x[np.array(speakers) == s].mean() for s in names])
    draws = means[np.random.default_rng(20260926).integers(len(names), size=(20000, len(names)))].mean(1)
    return {'n': len(x), 'speakers': len(names), 'speaker_mean': float(means.mean()),
            'speaker_ci95': np.quantile(draws, [.025, .975]).tolist(), 'speaker_ci99': np.quantile(draws, [.005, .995]).tolist(),
            'per_speaker': dict(zip(names, means.tolist()))}


def analyze():
    from scripts.experiments.vsr_tts_metrics import ctc_nll, levenshtein_distance
    from scripts.experiments.check_tts_independent_visual import independent_ctc
    p = read(OUT / 'protocol.json')
    scores, failures = [], []
    max_ctc = max_full = max_repeat = 0.
    count = 0
    for row in [r for r in p['rows'] if r['split'] == 'calibration']:
        receipt = read(OUT / 'receipts' / (row['id'] + '.json'))
        if not receipt['eligible']:
            failures.append({'id': row['id'], 'reason': 'technical_QC', 'cells': {k:v['qc']['eligible'] for k,v in receipt['cells'].items()}})
            continue
        result = {'id': row['id'], 'speaker': row['speaker'], 'cells': {}, 'arms': {}}
        targets = [row['target']] + [d['target'] for d in row['decoys']]
        for key, cell in receipt['cells'].items():
            views = {}
            for view, feature in cell['features'].items():
                assert sha(feature['path']) == feature['sha256']
                z = np.load(feature['path'])
                assert z['columns'].tolist() == row['logp_columns']
                mapping = {int(c): i for i, c in enumerate(z['columns'])}
                mapped = [[mapping[t] for t in target] for target in targets]
                losses = [ctc_nll(z['logp'], target) / len(target) for target in mapped]
                independent = [independent_ctc(z['logp'], target) for target in mapped]
                max_ctc = max(max_ctc, max(abs(x-y) for x,y in zip(losses, independent)))
                assert max_ctc <= 1e-8
                count += len(losses)
                if 'full_path' in feature:
                    assert sha(feature['full_path']) == feature['full_sha256']
                    full = np.load(feature['full_path'])
                    assert digest(full['logp']) == feature['full_logp_sha256']
                    assert digest(full['repeat_logp']) == feature['repeat_logp_sha256']
                    assert np.array_equal(full['logp'][:, z['columns']], z['logp'])
                    assert np.array_equal(full['logp'].argmax(1), z['greedy'])
                    max_repeat = max(max_repeat, float(np.max(np.abs(full['logp']-full['repeat_logp']))))
                    for target, expect in zip(targets, losses):
                        max_full = max(max_full, abs(independent_ctc(full['logp'], target) - expect))
                    assert max_full <= 1e-8 and max_repeat <= 1e-5
                hyp, prev = [], None
                for token in z['greedy'].tolist():
                    if token != prev and token != 0:
                        hyp.append(token)
                    prev = token
                views[view] = {'losses': losses, 'M': float(np.mean(losses[1:]) - losses[0]), 'target_nll': losses[0],
                               'top1': all(losses[0] < n for n in losses[1:]), 'CER': levenshtein_distance(row['target'], hyp) / len(row['target'])}
            result['cells'][key] = views
            result['arms'][key] = {'Q': views['native']['M'] - views['frozen']['M'], 'reverse': views['native']['M'] - views['reversed']['M'], **views['native']}
        scores.append(result)
    gates = {}
    for key in CELLS:
        if not scores:
            gates[key] = {'pass': False, 'n': 0}
            continue
        q = cluster([r['arms'][key]['Q'] for r in scores], [r['speaker'] for r in scores])
        rev = cluster([r['arms'][key]['reverse'] for r in scores], [r['speaker'] for r in scores])
        top = sum(r['arms'][key]['top1'] for r in scores) / 26
        pos = sum(r['arms'][key]['Q'] > 0 for r in scores) / 26
        gates[key] = {'n': len(scores), 'top1_fraction': top, 'q_positive_fraction': pos, 'Q': q, 'reverse': rev,
                      'pass': len(scores) == 26 and top >= .60 and pos >= .70 and q['speaker_ci95'][0] > 0 and rev['speaker_ci95'][0] > 0}
    out = {'stage': 'calibration_only', 'protocol_sha256': sha(OUT / 'protocol.json'), 'gate': gates,
           'primary_gate': gates['N_raw']['pass'], 'failures': failures, 'evaluation': 'LOCKED_PENDING_ROOT_REVIEW',
           'independent': {'ctc_values': count, 'max_ctc_error': max_ctc, 'full_sparse_ctc_error': max_full, 'full_repeat_error': max_repeat, 'status': 'PASS'}}
    # Recompute each bootstrap directly from independently regrouped per-utterance scores.
    for key in CELLS:
        for metric, label in [('Q','Q'), ('reverse','reverse')]:
            if not scores:
                continue
            groups = {}
            for r in scores:
                groups.setdefault(r['speaker'], []).append(r['arms'][key][metric])
            means = np.asarray([sum(groups[s])/len(groups[s]) for s in sorted(groups)])
            samples = np.random.default_rng(20260926).choice(means, size=(20000,len(means))).mean(1)
            for ci, cuts in [('speaker_ci95',[.025,.975]),('speaker_ci99',[.005,.995])]:
                assert np.max(np.abs(np.quantile(samples,cuts)-np.asarray(gates[key][label][ci]))) < 1e-12
    write(OUT / 'scores_calibration.json', scores)
    write(OUT / 'calibration_result.json', out)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'worker', 'analyze'])
    globals()[parser.parse_args().stage]()
