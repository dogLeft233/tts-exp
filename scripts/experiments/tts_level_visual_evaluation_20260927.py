"""Locked 66-utterance LEVEL visual evaluation using the sealed calibration path."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scripts.experiments import tts_level_visual_calibration_20260927 as cal

ROOT, CAL = cal.ROOT, cal.OUT
OUT = CAL / 'evaluation'
read, write, sha, digest = cal.read, cal.write, cal.sha, cal.digest
CELLS, VIEWS = cal.CELLS, cal.VIEWS


def freeze():
    assert not (OUT / 'protocol.json').exists()
    cp = read(CAL / 'protocol.json')
    cr = read(CAL / 'calibration_result.json')
    assert cr['primary_gate'] and cr['independent']['status'] == 'PASS'
    assert read(CAL / 'independent_pixels.json')['status'] == 'PASS'
    assert sha(cal.__file__) == cp['code_sha256']
    rows = [r for r in cp['rows'] if r['split'] == 'evaluation']
    assert len(rows) == 66 and len({r['speaker'] for r in rows}) == 15
    env = OUT / 'environment/CMLR_V_WER8.0.ini'
    env.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cp['config'], env)
    p = {'status': 'frozen_before_evaluation_features_scores', 'created_epoch': time.time(),
         'authorization': 'root approved fixed66 after N_raw calibration and all104 pixel verification PASS; wait FIXED GPU release',
         'parent_protocol_sha256': sha(CAL / 'protocol.json'), 'parent_code_sha256': sha(cal.__file__),
         'calibration_result_sha256': sha(CAL / 'calibration_result.json'), 'calibration_pixel_check_sha256': sha(CAL / 'independent_pixels.json'),
         'calibration_warning': 'N_raw top1 16/26 narrowly passes; N_LEVEL diagnostic15/26 fails; do not alter frozen gate or select samples',
         'rows': rows, 'oov': cp['oov'], 'model': cp['model'], 'model_sha256': cp['model_sha256'],
         'config': str(env), 'config_sha256': sha(env), 'dependencies': cp['dependencies'],
         'code_sha256': sha(__file__), 'metric': cp['metric'], 'qc': cp['qc'], 'stats': cp['stats'],
         'contrasts': cp['frozen_formal_contrasts'], 'formal_primary': cp['formal_primary'],
         'common_support': 'all4 cells technicalQC; exclude whole paired row only for prespecified technical failure, never from Q/M/CER/SyncNet',
         'limits': cp['limits'], 'resource': {**cp['resource'], 'persistent_budget_bytes': 150 << 20, 'scope': 'entire visual calibration+evaluation, no crop or encoder cache'},
         'storage': 'same exact full-softmax selected columns and all-vocabulary greedy IDs/full hash as calibration; no renormalization'}
    snapshot = OUT / 'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    for file in [Path(__file__), Path(cal.__file__)]:
        shutil.copyfile(file, snapshot / file.name)
    write(OUT / 'protocol.json', p)
    write(OUT / 'score_lock.json', {'time': time.time(), 'protocol_sha256': sha(OUT / 'protocol.json'), 'code_sha256': sha(__file__),
                                  'input_index_sha256': digest(np.frombuffer(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode(), dtype=np.uint8)),
                                  'status': 'before_any_eval_visual_features_or_scores', 'eval_ids': [r['id'] for r in rows]})
    print('EVALUATION_FROZEN', sha(OUT / 'protocol.json'), flush=True)


def budget(extra=0):
    total = sum(x.stat().st_size for x in CAL.rglob('*') if x.is_file())
    assert total + extra <= 150 << 20, ('shared visual150MiB budget', total, extra)
    return cal.disk(extra)


def worker():
    import torch
    from scipy.special import logsumexp
    from scripts.experiments.vsr_tts_metrics import make_views
    from scripts.experiments.vsr_tts_pilot import load_vsr, _forward_view, _set_determinism
    p = read(OUT / 'protocol.json')
    assert sha(OUT / 'protocol.json') == read(OUT / 'score_lock.json')['protocol_sha256']
    assert sha(__file__) == p['code_sha256'] and sha(cal.__file__) == p['parent_code_sha256']
    assert sha(p['model']) == p['model_sha256'] and sha(p['config']) == p['config_sha256']
    for path, h in p['dependencies'].items():
        assert sha(path) == h
    with open('/tmp/tts-exp-gpu.lock', 'a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            gates = []
            for n in range(3):
                gates.append(cal.resource(startup=True))
                if n < 2:
                    time.sleep(5)
            write(OUT / 'resource_start.json', gates)
            torch.set_num_threads(2)
            _set_determinism(20260926)
            pipeline = load_vsr(OUT, device='cuda:0')
            assert pipeline.modality == 'video'
            write(OUT / 'environment.json', {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'pid': os.getpid(), 'time': time.time()})
            for row in p['rows']:
                dest = OUT / 'receipts' / (row['id'] + '.json')
                if dest.exists():
                    continue
                rr = {'id': row['id'], 'speaker': row['speaker'], 'split': 'evaluation', 'cells': {}}
                for key in CELLS:
                    gate = cal.resource()
                    budget(3 << 20)
                    x, qc = cal.crop(pipeline, row['cells'][key])
                    cell = {'qc': qc, 'features': {}, 'resource': gate}
                    if x is not None:
                        for name in VIEWS:
                            cal.resource()
                            view = make_views(x)[name]
                            _, lp = _forward_view(pipeline, view, 'cuda:0')
                            assert len(lp) == x.shape[1] and np.isfinite(lp).all()
                            error = float(np.max(np.abs(logsumexp(lp.astype(float), axis=1))))
                            assert error <= 1e-5
                            columns = np.asarray(row['logp_columns'], dtype=np.int64)
                            path = OUT / 'features' / row['id'] / (key + '_' + name + '.npz')
                            path.parent.mkdir(parents=True, exist_ok=True)
                            budget(lp[:, columns].nbytes + (1 << 20))
                            np.savez_compressed(path, logp=lp[:, columns], columns=columns, greedy=lp.argmax(1).astype(np.int32))
                            cell['features'][name] = {'path': str(path), 'sha256': sha(path), 'full_logp_sha256': digest(lp), 'full_shape': list(lp.shape),
                                                      'normalization_max_error': error, 'selected_no_renormalization': True, 'view_sha256': digest(view)}
                    rr['cells'][key] = cell
                for arm in ('N', 'T'):
                    x, y = [rr['cells'][arm + '_' + c]['qc'] for c in ('raw', 'LEVEL')]
                    assert x['frame_count'] == y['frame_count'] and x['pts'] == y['pts']
                rr['eligible'] = all(c['qc']['eligible'] for c in rr['cells'].values())
                write(dest, rr)
                print('COMPLETE', row['id'], rr['eligible'], flush=True)
        finally:
            write(OUT / 'gpu_release.json', {'time': time.time(), 'pid': os.getpid(), 'stage': 'evaluation'})
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def analyze():
    from scripts.experiments.vsr_tts_metrics import ctc_nll, levenshtein_distance
    from scripts.experiments.check_tts_independent_visual import independent_ctc
    p = read(OUT / 'protocol.json')
    scores, missing = [], []
    maximum = 0.
    count = 0
    for row in p['rows']:
        receipt = read(OUT / 'receipts' / (row['id'] + '.json'))
        if not receipt['eligible']:
            missing.append({'id': row['id'], 'speaker': row['speaker'], 'reason': 'technical_QC', 'qc': {k:v['qc'] for k,v in receipt['cells'].items()}})
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
                check = [independent_ctc(z['logp'], target) for target in mapped]
                maximum = max(maximum, max(abs(x-y) for x,y in zip(losses, check)))
                assert maximum <= 1e-8
                count += len(losses)
                hyp, prev = [], None
                for token in z['greedy'].tolist():
                    if token != prev and token != 0:
                        hyp.append(token)
                    prev = token
                views[view] = {'losses': losses, 'M': float(np.mean(losses[1:])-losses[0]), 'target_nll': losses[0],
                               'top1': all(losses[0] < n for n in losses[1:]), 'CER': levenshtein_distance(row['target'], hyp)/len(row['target'])}
            result['cells'][key] = views
            result['arms'][key] = {'Q': views['native']['M']-views['frozen']['M'], 'reverse': views['native']['M']-views['reversed']['M'],
                                   'frozen_M': views['frozen']['M'], **views['native']}
        scores.append(result)
    assert scores, 'no common technical support'
    statistics = {}
    values = {}
    for metric in ['Q', 'M', 'target_nll', 'CER', 'frozen_M', 'reverse']:
        m = lambda row, cell: row['arms'][cell][metric]
        values[metric] = {
            'N_LEVEL_minus_raw': [m(r,'N_LEVEL')-m(r,'N_raw') for r in scores],
            'T_LEVEL_minus_raw': [m(r,'T_LEVEL')-m(r,'T_raw') for r in scores],
            'difference_in_differences': [m(r,'T_LEVEL')-m(r,'T_raw')-m(r,'N_LEVEL')+m(r,'N_raw') for r in scores],
            'T_minus_N_raw': [m(r,'T_raw')-m(r,'N_raw') for r in scores],
            'T_minus_N_LEVEL': [m(r,'T_LEVEL')-m(r,'N_LEVEL') for r in scores],
            **{key: [m(r,key) for r in scores] for key in CELLS}}
        statistics[metric] = {key: cal.cluster(v,[r['speaker'] for r in scores]) for key,v in values[metric].items()}
    max_stats = 0.
    for metric, contrasts in values.items():
        for key, values_ in contrasts.items():
            groups = {}
            for row,value in zip(scores,values_):
                groups.setdefault(row['speaker'],[]).append(value)
            means = np.asarray([sum(groups[s])/len(groups[s]) for s in sorted(groups)])
            sample = np.random.default_rng(20260926).choice(means,size=(20000,len(means))).mean(1)
            expected = {'speaker_mean':means.mean(),'speaker_ci95':np.quantile(sample,[.025,.975]),'speaker_ci99':np.quantile(sample,[.005,.995])}
            for name,value in expected.items():
                max_stats = max(max_stats,float(np.max(np.abs(value-np.asarray(statistics[metric][key][name])))))
    assert max_stats < 1e-12
    summary = {'status':'scored_and_independently_recomputed','protocol_sha256':sha(OUT/'protocol.json'),
               'n':len(scores),'speakers':len({r['speaker'] for r in scores}),'missing':missing,
               'common_ids':[r['id'] for r in scores],'statistics':statistics,
               'natural_content_improvement_99_conjunction':all(statistics[m]['N_LEVEL_minus_raw']['speaker_ci99'][0]>0 for m in ['Q','M']),
               'calibration_warning':p['calibration_warning'],
               'independent':{'status':'PASS','ctc_values':count,'max_ctc_error':maximum,'max_bootstrap_error':max_stats}}
    write(OUT/'scores.json',scores)
    write(OUT/'summary.json',summary)
    print('SCORED',len(scores),'independent',summary['independent'],'natural99conjunction',summary['natural_content_improvement_99_conjunction'],flush=True)


def pixels():
    import hashlib
    p = read(OUT/'protocol.json')
    count = frames_n = 0
    for row in p['rows']:
        r = read(OUT/'receipts'/(row['id']+'.json'))
        for key,v in row['cells'].items():
            assert sha(v['metadata'])==v['metadata_sha256'] and sha(v['path'])==v['sha256']
            raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',v['path'],'-map','0:v:0','-an','-f','rawvideo','-pix_fmt','bgr24','-threads','1','-'])
            assert hashlib.sha256(raw).hexdigest()==v['pixel_sha256']
            frame=np.frombuffer(raw,np.uint8).reshape(-1,224,224,3)
            assert len(frame)==v['frames']
            if r['cells'][key]['qc']['status']=='complete':
                assert digest(frame[...,::-1])==r['cells'][key]['qc']['decoded_RGB_sha256']
            frames_n+=len(frame);count+=1
        print('PIXELS_VERIFIED',row['id'],flush=True)
    assert count==264
    write(OUT/'independent_pixels.json',{'status':'PASS','clips':count,'frames':frames_n,'meaning':'independent FFmpeg BGR equals sealed generated pixel hash; RGB equals actual model input hash','protocol_sha256':sha(OUT/'protocol.json')})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','worker','analyze','pixels'])
    globals()[parser.parse_args().stage]()
