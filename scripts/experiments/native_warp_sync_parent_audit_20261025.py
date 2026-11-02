"""Independent saved-media/official-distance audit; no executor import or forwards."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess
import wave

import numpy as np
import soundfile as sf
import torch


ROLES = ('N', 'stretch', 'pause', 'combined', 'T')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def official_scalar(distance):
    # Official stack(dists,1) is contiguous [31,windows], unlike saved [windows,31].
    d = np.asarray(distance)
    assert d.dtype == np.float32 and d.ndim == 2 and d.shape[1] == 31
    assert len(d) > 0 and np.isfinite(d).all()
    curve = torch.from_numpy(d.T.copy()).mean(1)
    minimum, index = curve.min(0)
    median = torch.median(curve)
    return dict(C=float(median-minimum), D=float(minimum), B=float(median),
                offset=15-int(index), windows=len(d), curve=curve.tolist())


def aggregate(records):
    index = {(r['sample_id'], r['role']): r for r in records if r['role'] in ROLES}
    assert len(index) == 30
    ids = sorted({sid for sid, _ in index})
    assert len(ids) == 6
    table = []
    for sid in ids:
        cells = {k: index[sid, k] for k in ROLES}
        delta = {k: {m: cells[k][m]-cells['N'][m] for m in ('C', 'D', 'B')}
                 for k in ROLES[1:]}
        table.append(dict(sample_id=sid, cells=cells, delta=delta))
    means = {k: {m: float(np.mean([r['cells'][k][m] for r in table]))
                 for m in ('C', 'D', 'B')} for k in ROLES}
    effects = {k: {m: dict(mean=float(np.mean([r['delta'][k][m] for r in table])),
                           positive=sum(r['delta'][k][m] > 0 for r in table),
                           values=[r['delta'][k][m] for r in table])
                    for m in ('C', 'D', 'B')} for k in ROLES[1:]}
    return dict(per_source=table, means=means, effects=effects)


def pcm(path):
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        return w.readframes(w.getnframes())


def capture(args):
    return subprocess.check_output(list(map(str, args)), stderr=subprocess.PIPE)


def audit(root):
    protocol = read(root/'protocol.json')
    assert sha(root/'source_frozen.py') == protocol['pins'][str(Path(__file__).parent/'native_warp_sync_pilot_20261025.py')]
    for p, expected in protocol['pins'].items():
        assert sha(p) == expected, p
    inputs = read(root/'inputs.json')
    assert len(inputs) == 32 and inputs == protocol['inputs']
    expected_ids = ['BAC009S0003W0121','BAC009S0005W0121','BAC009S0006W0121',
                    'BAC009S0007W0121','BAC009S0008W0121','BAC009S0014W0121']
    assert sorted({r['sample_id'] for r in inputs}) == expected_ids
    assert {(r['sample_id'],r['role']) for r in inputs if r['role']=='N_repeat'} == {
        (expected_ids[0],'N_repeat'),(expected_ids[-1],'N_repeat')}
    videos = read(root/'video_manifest.json')
    assert len(videos) == 32
    records = []
    runner_error = 0.
    for row, video in zip(inputs, videos):
        assert row['sample_id'] == video['sample_id'] and row['role'] == video['role']
        sid, role = row['sample_id'], row['role']
        key = sid+'_'+role
        assert sha(row['source_path']) == row['source_sha256']
        assert sha(row['audio_path']) == row['audio_sha256']
        source, rate = sf.read(row['source_path'], dtype='float64')
        assert rate == 16000 and source.ndim == 1 and np.isfinite(source).all()
        assert max(abs(source)) < 1 and row['clipped_samples'] == 0
        rebuilt = np.floor(source*32768).astype('<i2').tobytes()
        assert rebuilt == pcm(row['audio_path']) and len(rebuilt) == row['samples']*2
        assert hashlib.sha256(rebuilt).hexdigest() == row['pcm_sha256']
        for name in ('video', 'mux'):
            assert sha(video[name+'_path']) == video[name+'_sha256']
        mux_pcm = capture(['ffmpeg','-v','error','-i',video['mux_path'],
                           '-map','0:a:0','-ac','1','-ar','16000','-f','s16le','-'])
        assert mux_pcm == rebuilt
        frames = [capture(['ffmpeg','-v','error','-i',video[k+'_path'],
                           '-map','0:v:0','-an','-f','framemd5','-']) for k in ('video','mux')]
        assert frames[0] == frames[1]
        probe = json.loads(capture(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                                   '-show_entries','stream=nb_read_frames,r_frame_rate,start_time',
                                   '-of','json',video['mux_path']]))['streams'][0]
        assert probe['r_frame_rate'] == '25/1' and float(probe['start_time']) == 0
        rec = read(root/'scores'/f'{key}.json')
        assert rec['track_count'] == 1 and len(rec['crop_tracks']) == 1
        assert sha(rec['activesd_path']) == rec['activesd_sha256']
        official = pickle.loads(Path(rec['activesd_path']).read_bytes())
        assert len(official) == 1
        array_path = root/'items'/key/'distance.npz'
        assert sha(array_path) == rec['distance_sha256']
        distance = np.load(array_path)['distance']
        assert np.array_equal(distance, official[0])
        canonical = official_scalar(distance)
        for m in ('C','D','B'):
            runner_error = max(runner_error, abs(canonical[m]-rec[m]))
            assert abs(canonical[m]-rec[m]) < 1e-4
        assert canonical['offset'] == rec['offset']
        log = (root/'logs'/f'syncnet_{key}.log').read_text()
        for label, metric in [('Confidence','C'),('Min dist','D')]:
            printed = re.findall(label+r':\s+([-+0-9.eE]+)', log)
            assert len(printed) == 1 and abs(float(printed[0])-canonical[metric]) <= .00051
        assert int(re.findall(r'AV offset:\s+(-?\d+)',log)[0]) == canonical['offset']
        scoring = Path(rec['scoring_audio_path'])
        assert sha(scoring) == rec['scoring_audio_sha256']
        # Reproduce the official waveform extraction from its exact sole crop.
        crop = Path(rec['crop_tracks'][0]['crop_file'])
        assert sha(crop) == rec['crop_tracks'][0]['crop_sha256']
        exact_audio = capture(['ffmpeg','-v','error','-i',crop,'-async','1','-ac','1','-vn',
                               '-acodec','pcm_s16le','-ar','16000','-f','s16le','-'])
        assert exact_audio == pcm(scoring)
        records.append(dict(sample_id=sid,role=role,**canonical,
                            video_frames=int(probe['nb_read_frames']),driver_samples=row['samples'],
                            scoring_samples=len(exact_audio)//2,
                            historical_C=row['historical_C'] if role in ('N','T') else None,
                            fresh_minus_historical_C=canonical['C']-row['historical_C']
                            if role in ('N','T') else None))
    grouped = aggregate(records)
    runner_summary = read(root/'summary.json')
    for role in ROLES:
        assert abs(grouped['means'][role]['C']-runner_summary['means'][role]) < 1e-4
    repeats = []
    for r in records:
        if r['role'] == 'N_repeat':
            n = next(a for a in records if a['sample_id']==r['sample_id'] and a['role']=='N')
            repeats.append(dict(sample_id=r['sample_id'],delta_C=r['C']-n['C'],
                                delta_D=r['D']-n['D'],offset_equal=r['offset']==n['offset']))
    return dict(status='PASS',protocol_sha256=sha(root/'protocol.json'),
                independently_audited_inputs=32,independently_audited_official_distance_arrays=32,
                runner_scalar_max_abs_error=runner_error,records=records,repeats=repeats,**grouped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    with (args.run/'parent_independent_audit.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','per_source')}))
