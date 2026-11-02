"""Cloud Qwen arm for the frozen20-source native AV comparison.

Cloud credentials belong only to the separate synthesis process stdin. This
orchestrator has no API credential. GPU work waits behind the previous run.
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from av_metric_preference_20261026 import active_production_commands, interval

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'runs/cogenav_sync_full20_20261024_attempt01'
PREVIOUS = ROOT / 'runs/av_metric_preference_20261026_attempt02'
DEPLOY = Path('/home/wjj/av-eval-20261026')
W2L = ROOT / 'third_party/Wav2Lip'
SYNC = ROOT / 'third_party/syncnet_python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def capture(command):
    return subprocess.check_output(list(map(str, command)))


def run_command(out, command, name, cwd=ROOT, timeout=1800):
    log = out / 'logs' / (name + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        done = subprocess.run(list(map(str, command)), cwd=cwd, stdout=stream,
                              stderr=subprocess.STDOUT, timeout=timeout)
    put(out / 'receipts' / (name + '.json'), dict(command=list(map(str, command)),
        returncode=done.returncode, cwd=str(cwd), log_sha256=sha(log)))
    if done.returncode:
        raise RuntimeError('Failed stage: ' + name + '; see ' + str(log))


def prepare(out):
    out.mkdir(exist_ok=False)
    samples = []
    selected = read(OLD / 'selected_inputs.json')
    for row in sorted((x for x in selected if x['arm'] == 'N'), key=lambda x: x['sample_id']):
        sid = row['sample_id']
        raw = ROOT / 'data/native_mechanism_holdout_20260926/audio' / (sid + '.wav')
        x, sr = sf.read(raw, dtype='float32')
        assert sr == 16000 and x.ndim == 1
        duration = len(x) / sr
        extras = []
        if duration < 5:
            extra = next(p for p in sorted(raw.parent.glob('BAC009' + row['speaker_id'] + '*.wav')) if p != raw)
            y, ysr = sf.read(extra, dtype='float32')
            assert ysr == sr and y.ndim == 1
            extras = [dict(path=str(extra), sha256=sha(extra), samples=len(y))]
            x = np.concatenate([x, y])
            ref = out / 'references24' / (sid + '.wav')
            ref.parent.mkdir(exist_ok=True)
            temporary = ref.with_name(sid + '_16.wav')
            sf.write(temporary, x, sr, subtype='PCM_16')
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(temporary),
                            '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', str(ref)], check=True)
        else:
            candidates = [ROOT / 'runs' / n / 'input24' / (sid + '.wav') for n in
                          ['qwen_codec_roundtrip_20261022_attempt01', 'qwen_nrt_expansion40_20261022_attempt01']]
            ref = next(p for p in candidates if p.exists())
        samples.append(dict(sample_id=sid, speaker_id=row['speaker_id'], transcript=row['text'],
            duration_s=duration, source_path=str(raw), source_sha256=sha(raw),
            input24_path=str(ref), input24_sha256=sha(ref), additional_reference=extras,
            reference_policy='same_utterance' if not extras else 'target_plus_same_speaker_second_utterance',
            reference_duration_s=sf.info(ref).duration, prior_natural_id=row['id']))
    assert len(samples) == 20 and sum(not x['additional_reference'] for x in samples) == 12
    gp = read(ROOT / 'runs/qwen_codec_roundtrip_20261022_attempt01/input_manifest.json')['generation']
    pins = [Path(__file__), Path(__file__).with_name('cloud_av_preference_synthesis_20261027.py'),
            Path(gp['face']), W2L / 'checkpoints/wav2lip_gan.pth', SYNC / 'data/syncnet_v2.model',
            Path(__file__).with_name('cogenav_sync_audit_20261024.py'),
            DEPLOY / 'score_peavs.py', DEPLOY / 'score_syncreward_v100.py',
            PREVIOUS / 'protocol.json', OLD / 'scores.json', OLD / 'selected_inputs.json']
    protocol = dict(run_id=out.name, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cloud_model='qwen-audio-3.1-tts-plus', draws_per_speaker=3, target_rms=0.0376838172675746,
        samples=samples, generation=gp, selection='Exactly previous20 sources; no outcome selection',
        primary='12 sources with identical single-utterance cloning references',
        secondary='8 short targets: target plus lexical first other same-speaker utterance for cloning only; reference-confounded versus local arms',
        normalization='mono16k PCM16 fixed RMS; generated audio has no time editing',
        metrics=['SyncNet official canonical float32', 'CoGenAV paper Eq1 and separate custom bestlag',
                 'SyncReward official Stage2 V100 FP32 center4.75sec', 'PEAVS official fullclip'],
        stats='Source-equal means of3 draws;10000 source bootstrap seed20261026 descriptive95CI; no perceptual truth',
        previous_run=str(PREVIOUS), planned_syntheses=60, pins={str(p): sha(p) for p in pins},
        control='Reuse original N plus +/-400ms control results; prior PEAVS/SyncReward control failure retained',
        new_api_retries=False, training=False)
    put(out / 'protocol.json', protocol)
    print(json.dumps(dict(run=str(out), sources=20, same_reference=12, planned_syntheses=60,
                          protocol_sha256=sha(out / 'protocol.json'))), flush=True)


def wait_gpu(out, stage):
    good = 0
    while True:
        memory, util = map(int, capture(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits']).decode().strip().split(','))
        processes = capture(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader']).decode().strip()
        foreign = [p for p in processes.splitlines() if p and 'gnome-remote-desktop' not in p]
        production = active_production_commands(capture(['ps', '-eo', 'stat=,args=']).decode().splitlines())
        # The old evaluation owns priority, preventing competing waiters from
        # both starting when ComfyUI eventually releases the device.
        previous_complete = read(PREVIOUS / 'status.json').get('stage') == 'COMPLETE'
        good = good + 1 if memory < 1024 and util < 15 and not foreign and not production and previous_complete else 0
        if good >= 3:
            return
        put(out / 'status.json', dict(stage='WAITING_GPU', next_stage=stage, memory_mib=memory,
                                     utilization=util, previous_complete=previous_complete,
                                     existing_production_active=bool(production), time=time.time()))
        time.sleep(30)


def sync_scalars(distance):
    import torch
    matrix = torch.from_numpy(np.ascontiguousarray(distance.T))
    mean = matrix.mean(1)
    minimum, index = mean.min(0)
    return dict(C=float(torch.median(mean) - minimum), D=float(minimum),
                B=float(torch.median(mean)), offset=int(15 - index), windows=distance.shape[0])


def render_and_sync(out):
    protocol = read(out / 'protocol.json')
    rows = read(out / 'audio_manifest.json')['records']
    results = []
    for row in rows:
        key = row['sample_id'] + '_' + row['role']
        receipt = out / 'sync_scores' / (key + '.json')
        if receipt.exists():
            results.append(read(receipt))
            continue
        wait_gpu(out, 'render_sync_' + key)
        audio = Path(row['path'])
        assert sha(audio) == row['sha256']
        work = out / 'render_work' / key
        (work / 'temp').mkdir(parents=True, exist_ok=True)
        video = out / 'video' / (key + '.mp4')
        video.parent.mkdir(exist_ok=True)
        if not video.exists():
            run_command(out, ['/home/wjj/.venvs/wav2lip/bin/python', W2L / 'inference.py',
                '--checkpoint_path', W2L / 'checkpoints/wav2lip_gan.pth', '--face', protocol['generation']['face'],
                '--audio', audio, '--outfile', video, '--face_det_batch_size', '4',
                '--wav2lip_batch_size', '4', '--nosmooth'], 'render_' + key, work)
        mux = out / 'inputs' / (key + '.mp4')
        mux.parent.mkdir(exist_ok=True)
        run_command(out, ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', video, '-i', audio,
            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'alac', '-map_metadata', '-1', mux], 'mux_' + key)
        actual = capture(['ffmpeg', '-v', 'error', '-i', mux, '-map', '0:a:0', '-f', 's16le', '-'])
        pcm, sr = sf.read(audio, dtype='int16')
        assert sr == 16000 and actual == pcm.astype('<i2').tobytes()
        frame_hashes = [capture(['ffmpeg', '-v', 'error', '-i', p, '-map', '0:v:0', '-an', '-f', 'framemd5', '-']) for p in [video, mux]]
        assert frame_hashes[0] == frame_hashes[1]
        # Match the existing official SyncNet baseline container, PCM codec,
        # and timestamp flags exactly. PEAVS/SyncReward keep their ALAC MP4.
        syncmux = out / 'sync_inputs' / (key + '.mkv')
        syncmux.parent.mkdir(exist_ok=True)
        run_command(out, ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-copyts', '-i', video,
            '-i', audio, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-avoid_negative_ts', 'disabled', syncmux], 'sync_mux_' + key)
        assert capture(['ffmpeg', '-v', 'error', '-i', syncmux, '-map', '0:a:0', '-f', 's16le', '-']) == actual
        def frame_pixels(path):
            result = capture(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v:0', '-an', '-f', 'framemd5', '-']).decode()
            return [line.split(',')[-1].strip() for line in result.splitlines() if not line.startswith('#')]
        def frame_times(path):
            result = json.loads(capture(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', path]))
            return np.array([float(f['best_effort_timestamp_time']) for f in result['frames']])
        assert frame_pixels(syncmux) == frame_pixels(video)
        assert np.allclose(frame_times(syncmux), frame_times(video), atol=1e-6, rtol=0)
        reference = 'cl27_' + key
        common = ['--videofile', syncmux, '--reference', reference, '--data_dir', out / 'sync_data']
        run_command(out, ['/home/wjj/.venvs/syncnet/bin/python', SYNC / 'run_pipeline.py', *common,
                          '--min_track', '25', '--overwrite'], 'pipeline_' + key, SYNC)
        run_command(out, ['/home/wjj/.venvs/syncnet/bin/python', SYNC / 'run_syncnet.py', *common,
                          '--initial_model', SYNC / 'data/syncnet_v2.model'], 'sync_' + key, SYNC)
        ap = out / 'sync_data/pywork' / reference / 'activesd.pckl'
        distances = pickle.loads(ap.read_bytes())
        assert len(distances) == 1 and distances[0].shape[1] == 31
        score = dict(row, id=key, video_path=str(video), video_sha256=sha(video), mux_path=str(mux),
                     mux_sha256=sha(mux), **sync_scalars(distances[0]), activesd_path=str(ap),
                     activesd_sha256=sha(ap), audio_pcm_exact=True, decoded_frames_exact=True,
                     sync_input_mkv=str(syncmux), sync_input_sha256=sha(syncmux), frame_PTS_exact=True)
        put(receipt, score)
        results.append(score)
        print('SYNC', key, score['C'], flush=True)
    put(out / 'sync_scores.json', results)


def new_metrics(out):
    rows = read(out / 'sync_scores.json')
    for metric in ['syncreward', 'peavs']:
        for start in range(0, len(rows), 20):
            batch = rows[start:start + 20]
            dest = out / metric / ('batch%03d.json' % (start // 20))
            if dest.exists():
                assert len(read(dest)) == len(batch)
                continue
            wait_gpu(out, metric + str(start))
            raw = dest.with_suffix('.raw.jsonl' if metric == 'syncreward' else '.raw.json')
            raw.parent.mkdir(exist_ok=True)
            wrapper = 'score_syncreward_v100.py' if metric == 'syncreward' else 'score_peavs.py'
            run_command(out, [f'/home/wjj/.venvs/{metric}/bin/python', DEPLOY / wrapper,
                              *[r['mux_path'] for r in batch], '--output', raw], metric + str(start), timeout=3600)
            scored = [json.loads(line) for line in raw.read_text().splitlines()] if metric == 'syncreward' else read(raw)['scores']
            keys = {Path(x['video_path'] if metric == 'syncreward' else x['filename']).stem for x in scored}
            assert len(scored) == len(batch) and keys == {r['id'] for r in batch}
            put(dest, scored)


def cogenav(out):
    target = out / 'cogenav'
    if (target / 'extraction_complete.json').exists():
        return
    target.mkdir(exist_ok=True)
    for name in ['upstream', 'weights', 'autoavsr_detector.py', 'autoavsr_video_process.py']:
        dest = target / name
        if not dest.exists():
            dest.symlink_to((OLD / name).resolve(), target_is_directory=(OLD / name).is_dir())
    manifest = []
    for row in read(out / 'sync_scores.json'):
        manifest.append(dict(id=row['id'], sample_id=row['sample_id'], arm='CLOUD', draw=row['draw_index'],
            qualified=True, score_float32=row['C'], audio_path=row['path'], audio_sha256=row['sha256'],
            video_path=row['video_path'], video_sha256=row['video_sha256'],
            features=dict(duration_s=row['samples'] / 16000)))
    put(out / 'cogenav_inputs.json', dict(rows=manifest))
    wait_gpu(out, 'cogenav')
    run_command(out, ['/home/wjj/.venvs/cogenav-sync-20261024/bin/python', Path(__file__).with_name('cogenav_sync_audit_20261024.py'),
                      '--run', target, '--manifest', out / 'cogenav_inputs.json', '--sources', '20',
                      '--record-long-clips'], 'cogenav', timeout=7200)


def load_metric(directory, metric):
    values = {}
    for path in sorted((directory / metric).glob('batch*.json')):
        if '.raw.' in path.name:
            continue
        for row in read(path):
            key = Path(row['video_path'] if metric == 'syncreward' else row['filename']).stem
            assert key not in values
            values[key] = row['reward'] if metric == 'syncreward' else row['score']
    return values


def summary(out):
    protocol = read(out / 'protocol.json')
    old = {r['sample_id']: r for r in read(OLD / 'scores.json') if r['arm'] == 'N'}
    co = {r['id']: r for r in read(out / 'cogenav/scores.json')}
    scores = read(out / 'sync_scores.json')
    values = {m: load_metric(out, m) for m in ['syncreward', 'peavs']}
    baselines = {m: load_metric(PREVIOUS, m) for m in values}
    paired = []
    strict = {s['sample_id'] for s in protocol['samples'] if not s['additional_reference']}
    for row in scores:
        key, sid = row['id'], row['sample_id']
        n = old[sid]
        if key not in co:
            continue
        t = dict(SyncC=row['C'], CoGenAV_Eq1=co[key]['eq1'], CoGenAV_bestlag=co[key]['max_score'],
                 **{m: values[m][key] for m in values})
        nv = dict(SyncC=n['score_float32'], CoGenAV_Eq1=n['eq1'], CoGenAV_bestlag=n['max_score'],
                  **{m: baselines[m]['native__' + n['id']] for m in values})
        paired.append(dict(id=key, source=sid, strict_reference=sid in strict, T=t, N=nv,
                           delta={k: t[k] - nv[k] for k in t}))
    report = dict(status='COMPLETE_EXPLORATORY', planned=60, measured=len(paired),
                  controls='Prior +/-400ms tests failed: SyncReward16/40, PEAVS26/40; not validated synchronization truth', subsets={})
    for name, subset in [('same_reference12', [p for p in paired if p['strict_reference']]),
                         ('extended_reference8', [p for p in paired if not p['strict_reference']]), ('all20', paired)]:
        result = {}
        for metric in ['SyncC', 'CoGenAV_Eq1', 'CoGenAV_bestlag', 'syncreward', 'peavs']:
            sources = sorted({p['source'] for p in subset})
            groups = [[p for p in subset if p['source'] == s] for s in sources]
            result[metric] = dict(N_mean=float(np.mean([np.mean([p['N'][metric] for p in g]) for g in groups])),
                T_mean=float(np.mean([np.mean([p['T'][metric] for p in g]) for g in groups])),
                delta=interval(float(np.mean([p['delta'][metric] for p in g])) for g in groups),
                TTS_higher=sum(p['delta'][metric] > 0 for p in subset),
                natural_higher=sum(p['delta'][metric] < 0 for p in subset), pairs=len(subset),
                conflicts_vs_syncC=sum(p['delta'][metric] * p['delta']['SyncC'] < 0 for p in subset))
        report['subsets'][name] = result
    put(out / 'paired_scores.json', paired)
    put(out / 'summary.json', report)
    lines = ['# Cloud Qwen native AV comparison', '', report['controls'], '',
             '12 same-reference sources primary;8 concatenated-reference sources separate. Source-equal descriptive bootstrap; scores are evaluator evidence.', '']
    for name, result in report['subsets'].items():
        lines += [name, '', '|Metric|N|Cloud|Delta|Cloud higher|SyncC conflicts|', '|---|---:|---:|---:|---:|---:|']
        for metric, row in result.items():
            lines.append(f"|{metric}|{row['N_mean']:.6f}|{row['T_mean']:.6f}|{row['delta']['mean']:+.6f}|{row['TTS_higher']}/{row['pairs']}|{row['conflicts_vs_syncC']}|")
        lines.append('')
    (out / 'report.md').write_text('\n'.join(lines))


def execute(out):
    import fcntl
    lease = (out / 'runner.lock').open('w')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(out / 'protocol.json')
    repair = read(out / 'execution_repair.json') if (out / 'execution_repair.json').exists() else {}
    for path, expected in protocol['pins'].items():
        if path == str(Path(__file__)) and repair:
            assert repair['original_code_sha256'] == expected
            expected = repair['repaired_code_sha256']
        assert sha(path) == expected, ('PIN_CHANGED', path)
    assert read(out / 'cloud_completion.json')['status'] == 'COMPLETE'
    try:
        render_and_sync(out)
        new_metrics(out)
        cogenav(out)
        summary(out)
        put(out / 'status.json', dict(stage='COMPLETE', pairs=60))
    except BaseException as exc:
        put(out / 'status.json', dict(stage='FAILED', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'execute', 'summary'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.out.resolve())
