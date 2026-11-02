"""Fixed-video SyncNet threshold/audio factorial; never changes the parent run."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy import signal
from scipy.io import wavfile

B = Path('/home/data2/wangjiajun/av-evaluation')
PARENT = B/'jobs/talklikeyou_chinese20_rms24_20261030_attempt01'
RUN = B/'jobs/talklikeyou_syncnet_protocol_controls_20261031_attempt01'
SYNC = B/'scorers/syncnet'
PYTHON = B/'envs/eval-base/bin/python'
FFMPEG = B/'tools/bin/ffmpeg'
GPU = 'GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(p)


def capture(args):
    return subprocess.check_output(list(map(str, args)))


def command(args, name, cwd=None):
    env = dict(os.environ, GPU_METRICS_PATH=str(RUN/'logs'/(name+'_gpu.json')))
    start = time.time()
    with (RUN/'logs'/(name+'.log')).open('w') as output:
        result = subprocess.run(list(map(str, args)), stdout=output,
                                stderr=subprocess.STDOUT, cwd=cwd, env=env)
    write(RUN/'stages'/(name+'.json'), dict(command=list(map(str, args)),
          returncode=result.returncode, started=start, ended=time.time(),
          log_sha256=sha(RUN/'logs'/(name+'.log'))))
    if result.returncode:
        raise RuntimeError('Failed '+name)


def compare_audio(reference, observed, sr=16000):
    """Positive lag means observed is delayed; alignment is diagnostic only."""
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    assert x.ndim == y.ndim == 1 and len(x) and len(y)
    corr = signal.correlate(y, x, mode='full', method='fft')
    lags = signal.correlation_lags(len(y), len(x), mode='full')
    mask = abs(lags) <= sr
    lag = int(lags[mask][np.argmax(corr[mask])])
    start_x, start_y = max(0, -lag), max(0, lag)
    n = min(len(x)-start_x, len(y)-start_y)
    a, b = x[start_x:start_x+n], y[start_y:start_y+n]
    energy = float(a@a)
    gain = float(a@b/energy) if energy else 0.
    residual = float(np.sum((b-gain*a)**2))
    zero_n = min(len(x), len(y))
    return dict(reference_samples=len(x), observed_samples=len(y), sample_rate=sr,
        exact_equal=bool(np.array_equal(x, y)), length_difference_samples=len(y)-len(x),
        lag_samples=lag, lag_ms=lag*1000/sr, lag_search_limit_samples=sr,
        lag_hit_boundary=abs(lag)==sr, aligned_samples=n,
        aligned_cosine=float(a@b/np.sqrt((a@a)*(b@b))) if energy and b@b else 0.,
        aligned_gain=gain,
        aligned_snr_db=float(10*np.log10(energy/residual)) if energy and residual else None,
        zero_lag_rmse_pcm_units=float(np.sqrt(np.mean((x[:zero_n]-y[:zero_n])**2))),
        aligned_rmse_pcm_units=float(np.sqrt(np.mean((a-b)**2))),
        reference_rms=float(np.sqrt(np.mean(x*x))), observed_rms=float(np.sqrt(np.mean(y*y))))


def probe(p):
    return json.loads(capture(['ffprobe', '-v', 'error', '-show_streams',
                              '-show_format', '-of', 'json', p]))


def pts(p):
    data = json.loads(capture(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', p]))
    return [float(f['best_effort_timestamp_time']) for f in data['frames']]


def decode_audio(p, out):
    subprocess.run([str(FFMPEG), '-nostdin', '-v', 'error', '-y', '-i', str(p),
        '-map', '0:a:0', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(out)], check=True)
    sr, x = wavfile.read(out)
    assert sr == 16000 and x.ndim == 1
    return x


def resource_check():
    deadline = time.time()+12*3600
    while True:
        free, util = map(int, capture(['nvidia-smi', '-i', GPU,
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode().strip().split(','))
        if free >= 8000 and util <= 85:
            return
        write(RUN/'resource_wait.json', dict(status='WAITING_GPU', free_mib=free,
              utilization=util, updated=time.time()))
        if time.time() > deadline:
            raise RuntimeError('GPU resource wait exceeded12h')
        time.sleep(30)


def freeze():
    assert not RUN.exists(), 'Fresh run required; use --resume only for saved stages'
    RUN.mkdir()
    (RUN/'logs').mkdir()
    parent = read(PARENT/'scoring_inputs.json')
    records = []
    for row in parent['unique_rows']:
        raw = PARENT/'generation'/row['id']/'generated_aac.mp4'
        receipt = read(PARENT/'generation'/row['id']/'receipt.json')
        assert sha(raw) == receipt['raw_video_sha256']
        assert sha(row['video_path']) == receipt['video_sha256'] == row['video_sha256']
        assert sha(row['audio_path']) == row['audio_sha256']
        records.append(dict(row, original_mp4=str(raw), original_mp4_sha256=sha(raw)))
    raw_by_id = {r['id']: r['original_mp4_sha256'] for r in records}
    alias_checks = []
    for alias, original in parent['aliases'].items():
        raw = PARENT/'generation'/alias/'generated_aac.mp4'
        receipt = read(PARENT/'generation'/alias/'receipt.json')
        actual = sha(raw)
        assert actual == receipt['raw_video_sha256'] == raw_by_id[original]
        alias_checks.append(dict(alias=alias, original=original, original_mp4_sha256=actual))
    pins = {str(p): sha(p) for p in [SYNC/'run_pipeline.py', SYNC/'run_syncnet.py',
        SYNC/'SyncNetInstance.py', SYNC/'SyncNetModel.py', SYNC/'data/syncnet_v2.model',
        SYNC/'detectors/s3fd/weights/sfd_face.pth', B/'gpu_guard.py', Path(__file__)]}
    write(RUN/'protocol.json', dict(status='FROZEN_BEFORE_FORWARD', parent=str(PARENT),
        unit='source: original three TTS logical draws averaged then paired with natural',
        cells=['original_mp4_25', 'original_mp4_100', 'canonical_pcm_25', 'canonical_pcm_100'],
        policy='Use native original AAC MP4 or copy identical video stream with canonical PCM MKV; no shifts, padding, truncation, gain changes or regeneration. Official upstream intermediate media are retained.',
        threshold_support='Primary: all four cells PASS for all four logical clips of a source; strict_reference subset reported separately. Also audio-only contrast on joint25 support as secondary.',
        endpoints=['C','D','offset','D0'],
        inference='Source paired bootstrap10000 seed20261031; exact two-sided signflip for <=20 sources; report Holm across three protocol-effect contrasts per stratum. No human quality claim.',
        threshold_forward_reuse='Only when the new100 cropped AVI SHA equals the corresponding25 cropped AVI; both official pipelines still run. Missing tracks are UNMEASURABLE, never zero scores.',
        timing='Cross correlation +/-1s at16k, positive means observed delayed. Diagnostic alignment never fed to scorer.',
        source_pins=pins, rows=parent['rows'], aliases=parent['aliases'], alias_original_checks=alias_checks, unique_rows=records,
        gpu=GPU, started=time.time()))
    shutil.copy2(__file__, RUN/'run_frozen.py')
    (RUN/'environment.freeze.txt').write_bytes(capture([PYTHON, '-m', 'pip', 'freeze']))
    (RUN/'ffmpeg_version.txt').write_bytes(capture([FFMPEG, '-version']))


def process(row):
    import torch
    torch.set_num_threads(4)
    key = row['id']
    root = RUN/'items'/key
    root.mkdir(parents=True, exist_ok=True)
    completed = root/'complete.json'
    if completed.exists():
        return read(completed)
    assert sha(row['original_mp4']) == row['original_mp4_sha256']
    assert sha(row['audio_path']) == row['audio_sha256']
    raw = Path(row['original_mp4'])
    canonical = root/'canonical.mkv'
    if not canonical.exists():
        command([FFMPEG, '-nostdin', '-v', 'error', '-y', '-copyts', '-i', raw,
            '-i', row['audio_path'], '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
            '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-avoid_negative_ts',
            'disabled', canonical], key+'_canonical_mux')
    video_hashes = []
    for p in [raw, canonical]:
        # Hash decoded pixels without timestamps; PTS are verified separately.
        h = capture([FFMPEG, '-nostdin', '-v', 'error', '-i', p, '-map', '0:v:0',
            '-pix_fmt', 'bgr24', '-f', 'hash', '-hash', 'sha256', '-']).decode().strip()
        video_hashes.append(h)
    assert video_hashes[0] == video_hashes[1]
    raw_pts, canonical_pts = pts(raw), pts(canonical)
    assert len(raw_pts) == row['actual_video_frames']
    assert np.allclose(raw_pts, canonical_pts, atol=1e-6, rtol=0)
    sr, reference = wavfile.read(row['audio_path'])
    assert sr == 16000 and reference.ndim == 1
    write(root/'input_identity.json', dict(decoded_video_sha256=video_hashes[0],
        frame_pts=raw_pts, canonical_sha256=sha(canonical), original_sha256=sha(raw),
        canonical_probe=probe(canonical), original_probe=probe(raw)))
    results = {}
    for mode, media in [('original_mp4', raw), ('canonical_pcm', canonical)]:
        decoded_input = decode_audio(media, root/(mode+'_decoded_input.wav'))
        if mode == 'canonical_pcm':
            assert np.array_equal(decoded_input, reference)
        for threshold in [25, 100]:
            cell = f'{mode}_{threshold}'
            target = root/cell
            target.mkdir(exist_ok=True)
            resultfile = target/'result.json'
            if resultfile.exists():
                results[cell] = read(resultfile)
                continue
            if threshold == 100 and len(raw_pts) <= 100:
                result = dict(status='UNMEASURABLE', reason='video_frames<=100 implies detection_count cannot exceed100', frames=len(raw_pts))
                write(resultfile, result)
                results[cell] = result
                continue
            work = target/'work'
            common = ['--videofile', media, '--reference', key, '--data_dir', work]
            pipeline_name = key+'_'+cell+'_pipeline'
            prior_stage = RUN/'stages'/(pipeline_name+'.json')
            if prior_stage.exists():
                stage = read(prior_stage)
                assert stage['returncode'] == 0
                assert sha(RUN/'logs'/(pipeline_name+'.log')) == stage['log_sha256']
                assert (work/'pywork'/key/'tracks.pckl').exists()
            else:
                resource_check()
                command([PYTHON, B/'gpu_guard.py', SYNC/'run_pipeline.py', *common,
                    '--min_track', threshold, '--overwrite'], pipeline_name, SYNC)
            pywork = work/'pywork'/key
            tracks = pickle.loads((pywork/'tracks.pckl').read_bytes())
            write(target/'track_summary.json', [dict(first_frame=int(t['track']['frame'][0]),
                last_frame=int(t['track']['frame'][-1]), frames=len(t['track']['frame'])) for t in tracks])
            avi = work/'pyavi'/key/'video.avi'
            intermediate = decode_audio(avi, target/'intermediate_decoded.wav')
            # audio.wav is exactly what official crop_video reads, not our diagnostic decoder.
            rate, official_audio = wavfile.read(work/'pyavi'/key/'audio.wav')
            assert rate == 16000
            write(target/'transport.json', dict(avi_probe=probe(avi),
                canonical_to_input=compare_audio(reference, decoded_input),
                input_to_intermediate=compare_audio(decoded_input, intermediate),
                canonical_to_intermediate=compare_audio(reference, intermediate),
                canonical_to_official_audio=compare_audio(reference, official_audio),
                diagnostic_vs_official_audio=compare_audio(intermediate, official_audio),
                intermediate_video_pts=pts(avi)))
            crops = sorted((work/'pycrop'/key).glob('0*.avi'))
            if len(crops) != 1:
                result = dict(status='UNMEASURABLE', reason='expected exactly one face track', tracks=len(crops))
            else:
                crop = crops[0]
                prior = root/f'{mode}_25'/'work/pycrop'/key/'00000.avi'
                reused = threshold == 100 and prior.exists() and sha(prior) == sha(crop)
                if reused:
                    assert results[f'{mode}_25']['status'] == 'PASS'
                    matrix = np.load(root/f'{mode}_25'/'distances.npy')
                    shutil.copy2(root/f'{mode}_25'/'work/pywork'/key/'activesd.pckl', pywork/'activesd.pckl')
                else:
                    resource_check()
                    command([PYTHON, B/'gpu_guard.py', SYNC/'run_syncnet.py', *common,
                        '--initial_model', SYNC/'data/syncnet_v2.model'], key+'_'+cell+'_score', SYNC)
                    matrices = pickle.loads((pywork/'activesd.pckl').read_bytes())
                    assert len(matrices) == 1
                    matrix = np.asarray(matrices[0], dtype=np.float32)
                assert matrix.ndim == 2 and matrix.shape[1] == 31 and np.isfinite(matrix).all()
                np.save(target/'distances.npy', matrix)
                curve = torch.from_numpy(matrix).T.mean(1)
                d, k = torch.min(curve, 0)
                result = dict(status='PASS', C=float(torch.median(curve)-d), D=float(d),
                    D0=float(curve[15]), B=float(torch.median(curve)), offset=int(15-k),
                    windows=len(matrix), cropped_avi_sha256=sha(crop),
                    forward_reused_from=f'{mode}_25' if reused else None)
                cropped_audio = decode_audio(crop, target/'crop_decoded.wav')
                first = int(tracks[0]['track']['frame'][0])*640
                last = (int(tracks[0]['track']['frame'][-1])+1)*640
                write(target/'crop_audio_diagnostic.json', compare_audio(reference[first:last], cropped_audio))
            write(resultfile, result)
            results[cell] = result
    write(completed, dict(id=key, cells=results, ended=time.time()))
    return read(completed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    os.environ['PATH'] = str(B/'tools/bin')+':'+os.environ['PATH']
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.sched_setaffinity(0, {0, 1, 2, 3})
    if not args.resume:
        freeze()
    lock = (RUN/'experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = read(RUN/'protocol.json')
    for path, expected in protocol['source_pins'].items():
        assert sha(path) == expected, path
    records = []
    for i, row in enumerate(protocol['unique_rows']):
        records.append(process(row))
        write(RUN/'status.json', dict(status='RUNNING', completed=i+1,
            total=len(protocol['unique_rows']), last=row['id'], updated=time.time()))
        print('COMPLETE', i+1, row['id'], flush=True)
    write(RUN/'results.json', records)
    write(RUN/'status.json', dict(status='FORWARDS_COMPLETE', completed=len(records), ended=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if RUN.exists():
            write(RUN/'failure.json', dict(error=repr(error), time=time.time()))
        raise
