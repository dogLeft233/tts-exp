"""Independent retimed-frame/path/official-score audit; no runner import or forwards."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess

import numpy as np
import soundfile as sf
import torch


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def official_scalar(distance):
    distance = np.asarray(distance)
    assert distance.dtype == np.float32 and distance.ndim == 2 and distance.shape[1] == 31
    assert len(distance) and np.isfinite(distance).all()
    curve = torch.from_numpy(distance.T.copy()).mean(1)
    minimum, index = curve.min(0)
    median = torch.median(curve)
    return dict(C=float(median-minimum), D=float(minimum), B=float(median),
                offset=15-int(index), windows=len(distance), curve=curve.tolist())


def validate_path(path, n, m):
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (n-1, m-1)
    assert path.ndim == 2 and path.shape[1] == 2
    assert np.issubdtype(path.dtype, np.integer)
    assert np.all(path >= 0) and np.all(path[:, 0] < n) and np.all(path[:, 1] < m)
    assert all(tuple(step) in {(1, 0), (0, 1), (1, 1)} for step in np.diff(path, axis=0))
    assert np.all(abs(path[:, 0]/max(n-1, 1)-path[:, 1]/max(m-1, 1)) <= .25+1e-12)


def solve_cost(cost):
    """Rolling-row optimum, separate implementation from executor traceback."""
    n, m = cost.shape
    previous = np.full(m, np.inf)
    for i in range(n):
        current = np.full(m, np.inf)
        for j in range(m):
            if abs(i/max(n-1, 1)-j/max(m-1, 1)) > .25:
                continue
            if i == j == 0:
                current[j] = cost[i, j]
                continue
            choices = []
            if i and j:
                choices.append(previous[j-1])
            if i:
                choices.append(previous[j]+.05)
            if j:
                choices.append(current[j-1]+.05)
            current[j] = min(choices)+cost[i, j]
        previous = current
    return previous[-1]


def project(path, n, m):
    groups = [[] for _ in range(n)]
    for i, j in path:
        groups[i].append(j)
    assert all(groups)
    values = np.array([np.median(g) for g in groups], dtype=float)
    values[0], values[-1] = 0, m-1
    values = np.maximum.accumulate(np.clip(values, 0, m-1))
    frames = np.clip(np.floor(values+.5).astype(np.int64), 0, m-1)
    return values, frames


def capture(args):
    return subprocess.check_output(list(map(str, args)), stderr=subprocess.PIPE)


def frame_hashes(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-25) < 1e-6
    values = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        values.append(hashlib.sha256(frame.tobytes()).hexdigest())
    cap.release()
    return values


def aggregate(records, roles):
    ids = sorted({r['sample_id'] for r in records})
    assert len(ids) == 6
    lookup = {(r['sample_id'], r['role']): r for r in records}
    assert len(lookup) == 6*len(roles)
    per_source = []
    for sid in ids:
        cells = {role: lookup[sid, role] for role in roles}
        native = cells['N_codec']['original_N_C']
        delta = {role: cells[role]['C']-cells['N_codec']['C'] for role in roles}
        original_delta = {role: cells[role]['C']-native for role in roles}
        per_source.append(dict(sample_id=sid, C={k:v['C'] for k,v in cells.items()},
                               delta_vs_codec=delta, delta_vs_original=original_delta))
    effects = {}
    for role in roles:
        values = [r['delta_vs_codec'][role] for r in per_source]
        original = [r['delta_vs_original'][role] for r in per_source]
        effects[role] = dict(C=float(np.mean([r['C'][role] for r in per_source])),
                            delta_C=float(np.mean(values)), positive=sum(v > 0 for v in values),
                            delta_original_C=float(np.mean(original)),
                            positive_original=sum(v > 0 for v in original), values=values,
                            D=float(np.mean([lookup[sid,role]['D'] for sid in ids])),
                            B=float(np.mean([lookup[sid,role]['B'] for sid in ids])))
    return dict(per_source=per_source, effects=effects)


def audit(root):
    protocol = read(root/'protocol.json')
    inputs = read(root/'inputs.json')
    roles = ('N_codec','linear','cogenav_av','cogenav_aa','mfcc_aa')
    assert tuple(protocol['roles']) == roles and inputs == protocol['inputs']
    assert protocol['status'] == 'FROZEN_BEFORE_ALL_FORWARD'
    assert len(inputs) == 6 and [r['sample_id'] for r in inputs] == [
        'BAC009S0003W0121','BAC009S0005W0121','BAC009S0006W0121',
        'BAC009S0007W0121','BAC009S0008W0121','BAC009S0014W0121']
    for path, expected in protocol['pins'].items():
        assert sha(path) == expected, path
    source_pin = next(v for k,v in protocol['pins'].items() if k.endswith('/cogenav_video_retime_20261025.py'))
    assert sha(root/'source_frozen.py') == source_pin
    model = read(root/'model_load.json')
    assert model['strict'] and model['forward_calls'] == 18
    assert model['resolved_config_sha256'] == sha(root/'base_resolved.yaml')
    previous = root.parent/'native_warp_sync_pilot_20261025_attempt01'
    imported_native_audit = read(previous/'parent_independent_audit.json')
    assert imported_native_audit['status'] == 'PASS'
    videos = read(root/'video_manifest.json')
    assert len(videos) == 30
    by_video = {(v['sample_id'],v['role']):v for v in videos}
    assert len(by_video) == 30
    records, maps, source_hashes, frame_counts = [], {}, {}, {}
    max_cost_matrix_error = max_optimal_cost_error = 0.
    path_count = 0
    import librosa
    for row in inputs:
        sid = row['sample_id']
        n,m = row['N']['video_frames'],row['T']['video_frames']
        assert row['output_frames'] == n
        for role in ('N','T'):
            item = row[role]
            for kind in ('audio','video'):
                assert sha(item[kind+'_path']) == item[kind+'_sha256']
            digests = frame_hashes(item['video_path'])
            assert digests == item['decoded_bgr_sha256'] and len(digests) == item['video_frames']
            source_hashes[sid,role] = digests
        features = np.load(root/'features'/sid/'features.npz')
        for key in ('audio_N','audio_T','video_T'):
            assert features[key].dtype == np.float32
            assert features[key].shape == (n if key.endswith('_N') else m,768)
            assert np.isfinite(features[key]).all()
        for role in ('N','T'):
            x,sr = sf.read(row[role]['audio_path'],dtype='float32')
            assert sr == 16000
            count = n if role == 'N' else m
            power = librosa.feature.melspectrogram(y=x,sr=sr,n_fft=400,hop_length=160,
                win_length=400,n_mels=80,fmin=0,fmax=8000,center=True,power=2)
            logmel = librosa.power_to_db(power,ref=1.,top_db=None)
            mfcc = librosa.feature.mfcc(S=logmel,n_mfcc=13)[:,::4][:,:count].T
            mfcc = ((mfcc-mfcc.mean(0,keepdims=True))/np.maximum(mfcc.std(0,keepdims=True),1e-5)).astype(np.float32)
            assert np.array_equal(mfcc,features['mfcc_'+role])
        saved = np.load(root/'features'/sid/'alignment.npz')
        for role,left,right,relu in (
                ('cogenav_av','audio_N','video_T',True),
                ('cogenav_aa','audio_N','audio_T',False),
                ('mfcc_aa','mfcc_N','mfcc_T',False)):
            a,b = features[left],features[right]
            a = a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
            b = b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
            cosine = np.clip(a@b.T,-1,1)
            cost = 1-np.maximum(cosine,0) if relu else 1-cosine
            difference = float(np.max(abs(cost-saved[role+'_cost'])))
            max_cost_matrix_error = max(max_cost_matrix_error,difference)
            assert difference < 1e-5
            path = saved[role+'_path']
            validate_path(path,n,m)
            optimal = solve_cost(saved[role+'_cost'])
            steps = np.diff(path,axis=0)
            observed = saved[role+'_cost'][path[:,0],path[:,1]].sum(dtype=np.float64)+.05*np.count_nonzero(steps.sum(1)==1)
            max_optimal_cost_error = max(max_optimal_cost_error,abs(observed-optimal))
            assert abs(observed-optimal) < 1e-4
            prediction,indices = project(path,n,m)
            assert np.array_equal(prediction,saved[role+'_prediction'])
            assert np.array_equal(indices,saved[role+'_indices'])
            maps[sid,role] = indices
            path_count += 1
        linear = np.linspace(0,m-1,n)
        assert np.array_equal(linear,saved['linear_prediction'])
        maps[sid,'linear'] = np.floor(linear+.5).astype(np.int64)
        assert np.array_equal(maps[sid,'linear'],saved['linear_indices'])
        maps[sid,'N_codec'] = np.arange(n)
        original_score = official_scalar(np.load(previous/'items'/f'{sid}_N'/'distance.npz')['distance'])
        old_audit = next(r for r in imported_native_audit['records'] if r['sample_id']==sid and r['role']=='N')
        assert original_score['C'] == old_audit['C']
        windows,scoring_pcm = [],[]
        for role in roles:
            video = by_video[sid,role]
            assert video['source_role'] == ('N' if role=='N_codec' else 'T')
            assert video['source_video_path'] == row[video['source_role']]['video_path']
            assert sha(video['video_path']) == video['video_sha256']
            assert video['audio_path'] == row['N']['audio_path']
            assert video['audio_sha256'] == row['N']['audio_sha256']
            assert video['output_frames'] == n and video['audio_samples'] == row['N']['samples']
            assert np.array_equal(video['indices'],maps[sid,role])
            expected = [source_hashes[sid,video['source_role']][i] for i in maps[sid,role]]
            actual = frame_hashes(video['video_path'])
            assert actual == expected == video['decoded_frame_sha256'] and len(actual)==n
            probe = json.loads(capture(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=r_frame_rate,start_time,codec_name','-of','json',video['video_path']]))['streams'][0]
            assert probe['codec_name']=='ffv1' and probe['r_frame_rate']=='25/1' and float(probe['start_time'])==0
            pcm = capture(['ffmpeg','-v','error','-i',video['video_path'],'-map','0:a:0',
                           '-ac','1','-ar','16000','-f','s16le','-'])
            natural = sf.read(row['N']['audio_path'],dtype='int16')[0].astype('<i2').tobytes()
            assert pcm == natural and hashlib.sha256(pcm).hexdigest()==video['pcm_sha256']
            key = sid+'_'+role
            record = read(root/'scores'/f'{key}.json')
            assert record['track_count']==1 and record['track_start_frame']==0
            track_data = pickle.loads((root/'data/pywork'/record['reference']/'tracks.pckl').read_bytes())
            assert len(track_data)==1
            track_frames = track_data[0]['track']['frame']
            assert int(track_frames[0])==0 and int(track_frames[-1])==record['track_end_frame']
            assert np.array_equal(track_frames,np.arange(n+2))
            assert sha(record['activesd_path'])==record['activesd_sha256']
            all_distances = pickle.loads(Path(record['activesd_path']).read_bytes())
            assert len(all_distances)==1
            distance_path = root/'items'/key/'distance.npz'
            assert sha(distance_path)==record['distance_sha256']
            distance = np.load(distance_path)['distance']
            assert np.array_equal(distance,all_distances[0])
            canonical = official_scalar(distance)
            assert all(canonical[k]==record[k] for k in ('C','D','B','offset','windows','curve'))
            log = (root/'logs'/f'syncnet_{key}.log').read_text()
            for label,metric in [('Confidence','C'),('Min dist','D')]:
                printed = re.findall(label+r':\s+([-+0-9.eE]+)',log)
                assert len(printed)==1 and abs(float(printed[0])-canonical[metric])<=.00051
            assert int(re.findall(r'AV offset:\s+(-?\d+)',log)[0])==canonical['offset']
            crops = sorted((root/'data/pycrop'/record['reference']).glob('0*.avi'))
            assert len(crops)==1
            exact = capture(['ffmpeg','-v','error','-i',crops[0],'-async','1','-ac','1',
                             '-vn','-acodec','pcm_s16le','-ar','16000','-f','s16le','-'])
            assert sha(record['scoring_audio_path'])==record['scoring_audio_sha256']
            assert exact==sf.read(record['scoring_audio_path'],dtype='int16')[0].astype('<i2').tobytes()
            scoring_pcm.append(hashlib.sha256(exact).hexdigest())
            windows.append(canonical['windows'])
            indices = maps[sid,role]
            records.append(dict(sample_id=sid,role=role,original_N_C=original_score['C'],
                **canonical,scoring_samples=len(exact)//2,output_frames=n,
                official_track_end_frame=int(track_frames[-1]),official_tail_frames=2,
                repeat_fraction=float(np.mean(np.diff(indices)==0)),
                max_source_step=int(np.max(np.diff(indices)))))
        assert len(set(windows))==1 and windows[0]==original_score['windows']
        assert len(set(scoring_pcm))==1, (sid,'different actual scoring PCM')
        frame_counts[sid] = n
    result = aggregate(records,roles)
    summary = read(root/'summary.json')
    for role in roles:
        assert result['effects'][role]['C']==summary['means'][role]
        if role!='N_codec':
            assert result['effects'][role]['delta_C']==summary['effects']['delta_vs_Ncodec'][role]['mean']
            assert result['effects'][role]['delta_original_C']==summary['effects']['delta_vs_Nnative'][role]['mean']
    return dict(status='PASS',protocol_sha256=sha(root/'protocol.json'),
                independently_audited_paths=path_count,independently_audited_videos=len(records),
                max_cost_matrix_abs_error=max_cost_matrix_error,
                max_optimal_cost_abs_error=max_optimal_cost_error,
                output_frame_counts=frame_counts,records=records,**result)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    destination = args.output if args.output else args.run/'parent_independent_audit.json'
    with destination.open('x') as output:
        json.dump(result,output,ensure_ascii=False,indent=2,allow_nan=False)
        output.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','per_source')},ensure_ascii=False))
