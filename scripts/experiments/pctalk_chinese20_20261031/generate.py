"""Sequential, resumable official PC-Talk inferences; identical inputs keep explicit aliases."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from prepare import B, D, GPU, RUN, sha, write


def validate(video, row):
    import cv2
    import numpy as np
    import soundfile as sf
    pcm, sr = sf.read(row['audio_path'], dtype='int16')
    decoded = subprocess.check_output([str(B/'tools/bin/ffmpeg'), '-nostdin', '-v', 'error',
                  '-i', str(video), '-map', '0:a:0', '-f', 's16le', '-'])
    assert sr == 16000 and decoded == pcm.astype('<i2').tobytes()
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                  '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)]))
    pts = [float(f['best_effort_timestamp_time']) for f in probe['frames']]
    assert np.allclose(pts, np.arange(len(pts))/25, atol=1e-6, rtol=0)
    cap = cv2.VideoCapture(str(video))
    assert abs(cap.get(cv2.CAP_PROP_FPS)-25) < 1e-6
    count = 0
    changes = []
    previous = None
    size = None
    previews = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        size = list(image.shape[:2])
        assert size == [512, 512], size
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            changes.append(float(abs(gray-previous).mean()))
        if count in {0, row['expected_frames']//2, row['expected_frames']-1}:
            previews.append(cv2.resize(image, (256, 256)))
        previous = gray
        count += 1
    cap.release()
    assert count == row['expected_frames'] == len(pts)
    assert changes and np.mean(changes) > .01
    cv2.imwrite(str(video.with_suffix('.jpg')), np.concatenate(previews, axis=1))
    return dict(frames=count, samples=len(pcm), decoded_pcm_exact=True, fps=25,
                resolution=size, pts_zero_regular=True, mean_frame_difference=float(np.mean(changes)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    lock = (RUN/'experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows = json.loads((RUN/'inputs.json').read_text())['rows']
    if options.smoke:
        source = rows[0]['source']
        pair = [r for r in rows if r['source'] == source]
        rows = [next(r for r in pair if r['arm'] == 'N'), next(r for r in pair if r['arm'] != 'N')]
    previous = {}
    completed = []
    for row in rows:
        key = row['audio_sha256']
        item = RUN/'generation'/row['id']
        item.mkdir(parents=True, exist_ok=True)
        raw = item/'generated_aac.mp4'
        video = RUN/'videos'/(row['id']+'.mp4')
        video.parent.mkdir(exist_ok=True)
        receipt = item/'receipt.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            assert saved['audio_sha256'] == key and sha(video) == saved['video_sha256']
            previous.setdefault(key, (raw, video, saved))
            completed.append(saved)
            continue
        assert not (item/'worker_result.json').exists(), 'Do not automatically retry a failed scientific inference'
        start = time.monotonic()
        reuse = None
        if key in previous:
            old_raw, old_video, result = previous[key]
            shutil.copyfile(old_raw, raw)
            shutil.copyfile(old_video, video)
            reuse = result['id']
        else:
            free, utilization = map(int, subprocess.check_output(['nvidia-smi', '-i', GPU,
                 '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            assert free >= 16000 and utilization <= 85, 'GPU resource check failed; wait rather than contend'
            command = [str(D/'env/bin/python'), '-u', str(Path(__file__).with_name('worker.py')), row['id']]
            with (item/'worker.log').open('w') as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                write(item/'process.json', dict(pid=process.pid, command=command, started=time.time(), gpu_uuid=GPU))
                code = process.wait()
            write(item/'exit.json', dict(returncode=code, ended=time.time()))
            assert code == 0, 'Inference failed; inspect '+str(item/'worker.log')
            result = json.loads((item/'worker_result.json').read_text())
            assert result['status'] == 'PASS'
            subprocess.run([str(B/'tools/bin/ffmpeg'), '-nostdin', '-v', 'error', '-y', '-i', str(raw),
                 '-i', row['audio_path'], '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'alac', str(video)], check=True)
        validation = validate(video, row)
        record = dict(id=row['id'], source=row['source'], status='PASS', audio_sha256=key,
            video_path=str(video), video_sha256=sha(video), raw_video_sha256=sha(raw),
            **validation, unsupported_tail_samples=row['samples']-validation['frames']*640,
            conditioning_audio_sha256=key, feature_frames=result['feature_frames'], pose_frames=result['pose_frames'],
            reused_from=reuse, seed=42, seconds=time.monotonic()-start,
            peak_allocated_bytes=result['peak_allocated_bytes'],
            peak_reserved_bytes=result['peak_reserved_bytes'],
            peak_nvidia_own_pid_mib=result['peak_nvidia_own_pid_mib'], gpu_uuid=GPU)
        write(receipt, record)
        previous.setdefault(key, (raw, video, record))
        completed.append(record)
        write(RUN/'generation_status.json', dict(status='RUNNING', completed=len(completed),
              planned=len(rows), last_id=row['id'], updated=time.time()))
        print('GENERATED', len(completed), row['id'], 'alias', reuse, flush=True)
    if options.smoke:
        write(RUN/'smoke_complete.json', dict(status='PASS', completed=2, receipts=completed))
    else:
        assert len(completed) == 80
        write(RUN/'generation_complete.json', dict(status='PASS', completed=80,
              actual_unique_forwards=len(previous), receipts=completed))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(RUN/'generation_failure.json', dict(error=repr(error), time=time.time()))
        raise
