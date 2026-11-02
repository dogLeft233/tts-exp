"""Resumable PEAVS/SyncReward evaluation of the frozen CoGenAV cohort.

Run prepare on CPU, then run waits for the GPU without interrupting other jobs.
Official deployed scoring entry points are invoked without source changes.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path


DEPLOY = Path('/home/wjj/av-eval-20261026')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def call(command):
    return subprocess.check_output([str(x) for x in command], stderr=subprocess.PIPE)


def pcm(path):
    with wave.open(str(path)) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (16000, 1, 2)
        return f.readframes(f.getnframes())


def plan(selected, scored_ids):
    items = {r['id']: r for r in selected if r['id'] in scored_ids}
    naturals = {r['sample_id']: r for r in items.values() if r['arm'] == 'N'}
    cells = []
    for key, row in sorted(items.items()):
        cells.append(dict(id='native__' + key, kind='native', source=row['sample_id'],
                          video_id=key, audio_id=key, arm=row['arm'], draw=row['draw']))
    for key, row in sorted(items.items()):
        if row['arm'] == 'N':
            continue
        natural = naturals[row['sample_id']]
        for vlabel, video in [('N', natural), ('T', row)]:
            for alabel, audio in [('N', natural), ('T', row)]:
                cells.append(dict(id=f'cross__{key}__{vlabel}{alabel}', kind='cross',
                                  source=row['sample_id'], tts_id=key, arm=row['arm'], draw=row['draw'],
                                  video_label=vlabel, audio_label=alabel,
                                  video_id=video['id'], audio_id=audio['id'],
                                  common_ids=[natural['id'], key]))
    for source, row in sorted(naturals.items()):
        for shift in [-6400, 6400]:
            cells.append(dict(id=f'shift__{source}__{shift}', kind='shift', source=source,
                              video_id=row['id'], audio_id=row['id'], shift_samples=shift))
    assert len(cells) == len({c['id'] for c in cells})
    return items, cells


def prepare(repo, out):
    original = repo / 'runs/cogenav_sync_full20_20261024_attempt01'
    scored = read(original / 'scores.json')
    items, cells = plan(read(original / 'selected_inputs.json'), {r['id'] for r in scored})
    protocol = dict(status='EXPLORATORY', sources=20, native_cells=138, cross_cells=472,
                    shift_cells=40, previous_cohort=str(original),
                    previous_scores_sha256=sha(original / 'scores.json'),
                    previous_selection_sha256=sha(original / 'selected_inputs.json'),
                    audio='16kHz mono source PCM, lossless ALAC in MP4; no amplitude adjustment',
                    native='Original video stream copied; original source PCM, no retiming or trimming',
                    cross='Four cells share common prefix floor(min(audio samples/640, video frames))/25; lossless H264 exact decoded prefix, no warp',
                    shifts='Natural native video, audio delayed/advanced 400ms by zero fill and truncation, length fixed',
                    metrics={'PEAVS': 'official deployed wrapper full clip',
                             'SyncReward': 'official Stage2 V100 FP32 wrapper center4.75sec'},
                    stats='Source-equal means; 10000 source bootstrap seed20261026; descriptive intervals',
                    scientific_limits=['No human ground truth', 'Cross audio has different phoneme clocks',
                                       'Native models score different temporal supports',
                                       'Shared preference does not establish perceived superiority'],
                    wrapper_sha256={name: sha(DEPLOY / name) for name in
                                    ['score_peavs.py', 'score_syncreward_v100.py', 'peavs_extract_bootstrap.py']})
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        assert read(out / 'protocol.json') == protocol
    else:
        write(out / 'protocol.json', protocol)
    write(out / 'status.json', dict(stage='CPU_PREPARATION', total=len(cells)))
    pins = []
    lengths = {}
    for key, row in items.items():
        for role in ['audio', 'video']:
            path = Path(row[role + '_path'])
            actual = sha(path)
            assert actual == row[role + '_sha256'], (key, role, 'SOURCE_HASH_MISMATCH')
            pins.append(dict(id=key, role=role, path=str(path), sha256=actual))
        probe = json.loads(call(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                 '-count_frames', '-show_entries', 'stream=nb_read_frames,r_frame_rate',
                                 '-of', 'json', row['video_path']]))['streams'][0]
        assert probe['r_frame_rate'] == '25/1', (key, probe)
        lengths[key] = min(len(pcm(row['audio_path'])) // 1280, int(probe['nb_read_frames']))
    write(out / 'input_pins.json', pins)
    write(out / 'cells.json', cells)
    for index, cell in enumerate(cells):
        dest = out / 'inputs' / (cell['id'] + '.mp4')
        receipt = out / 'preparation' / (cell['id'] + '.json')
        if receipt.exists():
            assert sha(dest) == read(receipt)['sha256']
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        video, audio = items[cell['video_id']], items[cell['audio_id']]
        raw = pcm(audio['audio_path'])
        count = None
        if cell['kind'] == 'cross':
            count = min(lengths[key] for key in cell['common_ids'])
            raw = raw[:count * 1280]
        if cell['kind'] == 'shift':
            shift = cell['shift_samples'] * 2
            raw = (b'\0' * shift + raw)[:len(raw)] if shift > 0 else (raw[-shift:] + b'\0' * -shift)[:len(raw)]
        tempwav = out / 'temporary_audio.wav'
        with wave.open(str(tempwav), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(raw)
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', video['video_path'],
                   '-i', str(tempwav), '-map', '0:v:0', '-map', '1:a:0',
                   '-c:v', 'copy', '-c:a', 'alac', '-map_metadata', '-1']
        if count is not None:
            # Stream-copy trimming leaves future B frames in this source codec.
            # Lossless intra coding yields an exact prefix; decoded hashes qualify it.
            command += ['-c:v', 'libx264', '-preset', 'ultrafast', '-qp', '0', '-bf', '0',
                        '-frames:v', str(count), '-t', str(count / 25)]
        call(command + [str(dest)])
        decoded = call(['ffmpeg', '-v', 'error', '-i', dest, '-map', '0:a:0',
                        '-f', 's16le', '-acodec', 'pcm_s16le', '-'])
        assert decoded == raw, (cell['id'], len(decoded), len(raw), 'LOSSLESS_AUDIO_FAILED')
        # Compare compressed video packet data hashes: no frame pixels can change.
        def video_packets(path):
            data = json.loads(call(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                   '-show_packets', '-show_data_hash', 'sha256',
                                   '-show_entries', 'packet=data_hash', '-of', 'json', path]))
            return [p['data_hash'] for p in data['packets']]
        packets = video_packets(dest)
        source_packets = video_packets(video['video_path'])
        copy_exact = packets == source_packets[:len(packets)]
        pixel_exact = None
        if count is not None:
            assert len(packets) == count, (cell['id'], len(packets), count)
            def frame_hashes(path):
                rawhash = call(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v:0',
                                '-frames:v', str(count), '-f', 'framemd5', '-']).decode()
                return [line.split(',')[-1].strip() for line in rawhash.splitlines() if not line.startswith('#')]
            pixel_exact = frame_hashes(dest) == frame_hashes(video['video_path'])
            assert pixel_exact, (cell['id'], 'DECODED_PIXELS_CHANGED')
        else:
            assert copy_exact
        write(receipt, dict(id=cell['id'], sha256=sha(dest), audio_samples=len(raw) // 2,
                            video_packets=len(packets), video_stream_copy_verified=copy_exact,
                            lossless_decoded_prefix_verified=pixel_exact,
                            audio_pcm_exact=True, cross_frames=count))
        tempwav.unlink()
        if index % 20 == 0:
            print('PREPARED', index + 1, '/', len(cells), flush=True)
    write(out / 'prepared.json', dict(cells=len(cells), protocol_sha256=sha(out / 'protocol.json'),
                                     input_pins_sha256=sha(out / 'input_pins.json')))
    write(out / 'status.json', dict(stage='PREPARED_WAITING_GPU', cells=len(cells)))


def direction(a, b):
    return 'TTS' if a > b else 'NATURAL' if a < b else 'TIE'


def interval(values):
    values = list(values)
    rng = random.Random(20261026)
    boots = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return dict(mean=statistics.mean(values), ci95=[boots[249], boots[9749]], sources=len(values))


def summarize(repo, out):
    cells = read(out / 'cells.json')
    old = {r['id']: r for r in read(repo / 'runs/cogenav_sync_full20_20261024_attempt01/scores.json')}
    result = {}
    for metric in ['syncreward', 'peavs']:
        scores = {}
        for path in sorted((out / metric).glob('batch*.json')):
            if '.raw.' in path.name:
                continue
            rows = read(path)
            for row in rows:
                key = Path(row['video_path'] if metric == 'syncreward' else row['filename']).stem
                value = row['reward'] if metric == 'syncreward' else row['score']
                assert key not in scores and math.isfinite(value)
                scores[key] = value
        if len(scores) != len(cells):
            result[metric] = dict(status='INCOMPLETE', scored=len(scores), planned=len(cells))
            natives = [c for c in cells if c['kind'] == 'native']
            if all(c['id'] in scores for c in natives):
                available = {c['video_id']: scores[c['id']] for c in natives}
                natural_by_source = {r['sample_id']: r['id'] for r in old.values() if r['arm'] == 'N'}
                groups = defaultdict(list)
                for key, r in old.items():
                    if r['arm'] != 'N':
                        groups[r['sample_id']].append(available[key] - available[natural_by_source[r['sample_id']]])
                result[metric]['native_stage'] = dict(
                    delta=interval(statistics.mean(v) for v in groups.values()),
                    TTS_higher=sum(v > 0 for vs in groups.values() for v in vs),
                    pairs=sum(len(vs) for vs in groups.values()),
                    status='NATIVE_COMPLETE_CONTROLS_AND_CROSS_PENDING')
            continue
        native = {c['video_id']: scores[c['id']] for c in cells if c['kind'] == 'native'}
        nat_ids = {r['sample_id']: r['id'] for r in old.values() if r['arm'] == 'N'}
        native_pairs = []
        for key, row in old.items():
            if row['arm'] == 'N':
                continue
            natural = nat_ids[row['sample_id']]
            native_pairs.append(dict(source=row['sample_id'], arm=row['arm'], tts_id=key,
                                     delta=native[key] - native[natural],
                                     delta_sync=row['score_float32'] - old[natural]['score_float32'],
                                     delta_eq1=row['eq1'] - old[natural]['eq1']))
        grouped = defaultdict(list)
        for row in native_pairs:
            grouped[row['source']].append(row['delta'])
        conflicts = {}
        for field in ['delta_sync', 'delta_eq1']:
            conflicts[field] = dict(
                both_TTS=sum(r['delta'] > 0 and r[field] > 0 for r in native_pairs),
                both_N=sum(r['delta'] < 0 and r[field] < 0 for r in native_pairs),
                other_TTS_new_N=sum(r[field] > 0 and r['delta'] < 0 for r in native_pairs),
                other_N_new_TTS=sum(r[field] < 0 and r['delta'] > 0 for r in native_pairs),
                ties=sum(r['delta'] == 0 or r[field] == 0 for r in native_pairs))
        families = {}
        for arm in ['L06', 'L17']:
            rows = [r for r in native_pairs if r['arm'] == arm]
            groups = defaultdict(list)
            for r in rows:
                groups[r['source']].append(r['delta'])
            families[arm] = dict(pairs=len(rows), TTS_higher=sum(r['delta'] > 0 for r in rows),
                                 delta=interval(statistics.mean(v) for v in groups.values()))
        cross = {}
        for vl in ['N', 'T']:
            rows = []
            for c in cells:
                if c['kind'] != 'cross' or c['video_label'] != vl or c['audio_label'] != 'N':
                    continue
                t = c['id'][:-1] + 'T'
                rows.append(dict(source=c['source'], delta=scores[t] - scores[c['id']]))
            groups = defaultdict(list)
            for row in rows:
                groups[row['source']].append(row['delta'])
            cross[vl] = dict(delta_T_minus_N=interval(statistics.mean(v) for v in groups.values()),
                             TTS_higher=sum(r['delta'] > 0 for r in rows), pairs=len(rows))
        shifts = [dict(id=c['id'], source=c['source'], shift_samples=c['shift_samples'],
                       native=native[c['video_id']], shifted=scores[c['id']],
                       pass_lower=scores[c['id']] < native[c['video_id']])
                  for c in cells if c['kind'] == 'shift']
        result[metric] = dict(status='COMPLETE', scored=len(scores), native_pairs=len(native_pairs),
                              native_delta=interval(statistics.mean(v) for v in grouped.values()),
                              native_TTS_higher=sum(r['delta'] > 0 for r in native_pairs),
                              conflicts=conflicts, families=families, cross=cross,
                              scientific_status='EXPLORATORY_SHIFT_CONTROL_PASS' if sum(r['pass_lower'] for r in shifts) >= 32 else 'CONTROL_FAILED_DO_NOT_INTERPRET_AS_SYNC_PREFERENCE',
                              shift_lower=sum(r['pass_lower'] for r in shifts), shift_total=len(shifts))
        write(out / (metric + '_native_pairs.json'), native_pairs)
        write(out / (metric + '_shift_controls.json'), shifts)
    write(out / 'summary.json', result)
    complete = all(r['status'] == 'COMPLETE' for r in result.values())
    lines = ['# PEAVS 与 SyncReward：自然/TTS 偏好复核', '',
             '状态：' + ('评分完成，探索性结果。' if complete else '尚未完成评分，不能给出偏好结论。'), '',
             '20 个来源，138 条原生配对视频，118 个自然–TTS 配对；另有472个固定视频换音轨评分单元和40个±400ms控制。', '',
             '原生音轨用源 PCM 无损 ALAC 封装，视频压缩数据逐包校验保持不变。固定视频四格采用同一共同前缀，无损编码后逐帧像素核验；不拉伸、不搜索最佳时移。', '',
             'SyncReward 使用部署的官方 Stage-2、V100 FP32、中心4.75秒规则；PEAVS使用官方整段提取和评分头。两者时间支持不同，旧SyncNet/CoGenAV分数也不作同窗口复现。', '',
             '所有区间按来源等权bootstrap，仅为描述性；重复TTS不增加独立样本量。跨音轨改变发音时钟，不能单独识别声学偏好；多指标一致也不能替代盲评。', '',
             '|指标|已评分|原生 Δ(TTS−N)|TTS更高配对|固定N视频 Δ(T−N)|固定T视频 Δ(T−N)|时移后降低|',
             '|---|---:|---|---:|---|---|---:|']
    for metric, r in result.items():
        if r['status'] != 'COMPLETE':
            lines.append(f"|{metric}|{r['scored']}/{r['planned']}|待完成|—|—|—|—|")
            continue
        fmt = lambda v: f"{v['mean']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]"
        lines.append(f"|{metric}|{r['scored']}|{fmt(r['native_delta'])}|{r['native_TTS_higher']}/118|{fmt(r['cross']['N']['delta_T_minus_N'])}|{fmt(r['cross']['T']['delta_T_minus_N'])}|{r['shift_lower']}/40|")
    for metric, r in result.items():
        if r['status'] == 'COMPLETE':
            lines += ['', metric + ' 与旧指标冲突方向：', '', '```json', json.dumps(r['conflicts'], indent=2), '```', '', r['scientific_status']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return result


def active_production_commands(process_rows):
    """Stopped controllers retain identity but cannot launch GPU work."""
    active = []
    for row in process_rows:
        parts = row.strip().split(None, 1)
        if len(parts) != 2:
            continue
        state, command = parts
        if any(flag in state for flag in 'TtZ'):
            continue
        if '/scripts/experiments/audio200' in command and (
                'controller_' in command or ' execute ' in command or ' worker-' in command):
            active.append(command)
    return active


def run(repo, out):
    import fcntl
    prepared = read(out / 'prepared.json')
    assert prepared['protocol_sha256'] == sha(out / 'protocol.json')
    protocol = read(out / 'protocol.json')
    original = Path(protocol['previous_cohort'])
    assert sha(original / 'scores.json') == protocol['previous_scores_sha256']
    assert sha(original / 'selected_inputs.json') == protocol['previous_selection_sha256']
    for name, expected in protocol['wrapper_sha256'].items():
        assert sha(DEPLOY / name) == expected, ('WRAPPER_CHANGED', name)
    cells = read(out / 'cells.json')
    _, expected_cells = plan(read(original / 'selected_inputs.json'), {r['id'] for r in read(original / 'scores.json')})
    assert cells == expected_cells
    # Score native pairs with BOTH metrics before the larger cross-soundtrack stage.
    jobs = []
    for kind in ['native', 'shift', 'cross']:
        selected = [c for c in cells if c['kind'] == kind]
        batches = [selected[i:i + 24] for i in range(0, len(selected), 24)]
        for metric in ['syncreward', 'peavs']:
            jobs.extend((metric, f'{kind}{index:03d}', batch) for index, batch in enumerate(batches))
    lease = (out / 'runner.lock').open('w')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for job_index, (metric, index, batch) in enumerate(jobs):
            dest = out / metric / f'batch{index}.json'
            if dest.exists():
                assert len(read(dest)) == len(batch)
                continue
            idle_checks = 0
            while True:
                raw = call(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits']).decode().strip()
                memory, utilization = [int(x.strip()) for x in raw.split(',')]
                processes = call(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader']).decode().strip()
                foreign = [p for p in processes.splitlines() if p and 'gnome-remote-desktop' not in p]
                process_args = call(['ps', '-eo', 'stat=,args=']).decode().splitlines()
                production = active_production_commands(process_args)
                if memory < 1024 and utilization < 15 and not foreign and not production:
                    idle_checks += 1
                    if idle_checks >= 3:
                        break
                else:
                    idle_checks = 0
                write(out / 'status.json', dict(stage='WAITING_GPU', metric=metric, batch=index,
                                               memory_mib=memory, utilization=utilization,
                                               existing_production_active=bool(production),
                                               waiting_since_unix=time.time()))
                time.sleep(30)
            for cell in batch:
                assert sha(out / 'inputs' / (cell['id'] + '.mp4')) == read(out / 'preparation' / (cell['id'] + '.json'))['sha256']
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix('.raw' + ('.jsonl' if metric == 'syncreward' else '.json'))
            wrapper = 'score_syncreward_v100.py' if metric == 'syncreward' else 'score_peavs.py'
            command = [f'/home/wjj/.venvs/{metric}/bin/python', str(DEPLOY / wrapper)]
            command += [str(out / 'inputs' / (c['id'] + '.mp4')) for c in batch]
            command += ['--output', str(temporary)]
            write(out / 'status.json', dict(stage='SCORING', metric=metric, batch=index, job=job_index, total_jobs=len(jobs)))
            with (dest.with_suffix('.log')).open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                write(out / 'status.json', dict(stage='SCORER_FAILED', metric=metric, batch=index, returncode=completed.returncode))
                raise RuntimeError(('SCORER_FAILED', metric, index))
            rows = [json.loads(line) for line in temporary.read_text().splitlines()] if metric == 'syncreward' else read(temporary)['scores']
            assert len(rows) == len(batch)
            keys = {Path(r['video_path'] if metric == 'syncreward' else r['filename']).stem for r in rows}
            assert keys == {c['id'] for c in batch}
            write(dest, rows)
            summarize(repo, out)
            print('SCORED', metric, index, job_index + 1, '/', len(jobs), flush=True)
    summarize(repo, out)
    write(out / 'status.json', dict(stage='COMPLETE', cells=len(cells), metrics=2))


def queue(repo, out):
    while not (out / 'prepared.json').exists():
        time.sleep(30)
    run(repo, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'queue', 'summarize'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.repo.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
