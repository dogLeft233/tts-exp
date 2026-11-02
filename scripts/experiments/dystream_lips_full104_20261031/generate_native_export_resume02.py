"""Sequential, resumable official DyStream inferences; identical inputs keep explicit aliases."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from prepare import B, D, GPU, RUN, sha, write
from native_export import exported_frames


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
        assert size == [512,512],size
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            changes.append(float(abs(gray-previous).mean()))
        if count in {0, row['expected_frames']//2, row['expected_frames']-1}:
            previews.append(cv2.resize(image, (256, 256)))
        previous = gray
        count += 1
    cap.release()
    assert count == exported_frames(row['expected_frames']) == len(pts)
    assert changes and np.mean(changes) > .01
    cv2.imwrite(str(video.with_suffix('.jpg')), np.concatenate(previews, axis=1))
    return dict(frames=count, samples=len(pcm), decoded_pcm_exact=True, fps=25,
                resolution=size, pts_zero_regular=True, mean_frame_difference=float(np.mean(changes)),
                rendered_latent_frames=row['expected_frames'], native_export_extra_frames=count-row['expected_frames'],
                motion_unsupported_tail_samples=row['samples']-row['expected_frames']*640,
                native_export_last_frame_difference=changes[-1])


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
        key = row['audio_sha256']+'_'+row['image_sha256']
        item = RUN/'generation'/row['id']
        item.mkdir(parents=True, exist_ok=True)
        raw = item/'generated_aac.mp4'
        video = RUN/'videos'/(row['id']+'.mp4')
        video.parent.mkdir(exist_ok=True)
        receipt = item/'receipt.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            assert saved['input_key'] == key and sha(video) == saved['video_sha256']
            previous.setdefault(key, (raw, video, saved))
            completed.append(saved)
            continue
        existing_worker = item/'worker_result.json'
        start = time.monotonic()
        reuse = None
        if existing_worker.exists():
            result = json.loads(existing_worker.read_text())
            assert result['status'] == 'PASS' and raw.exists() and video.exists()
            assert result['conditioning_audio_sha256'] == row['audio_sha256']
            assert result['conditioning_image_sha256'] == row['image_sha256']
            reuse = 'same previously completed forward; validation repair only'
        elif key in previous:
            old_raw, old_video, result = previous[key]
            shutil.copyfile(old_raw, raw)
            shutil.copyfile(old_video, video)
            reuse = result['id']
        else:
            free, utilization = map(int, subprocess.check_output(['nvidia-smi', '-i', GPU,
                 '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            while free < 16000 or utilization > 85:
                write(RUN/'generation_waiting_gpu.json',dict(status='WAITING_GPU',id=row['id'],free_mib=free,utilization=utilization,updated=time.time()))
                time.sleep(45)
                free, utilization = map(int, subprocess.check_output(['nvidia-smi', '-i', GPU,
                     '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            command = ['bash',str(Path(__file__).with_name('worker_infer.sh')),'--source',row['image_path'],'--audio',row['audio_path'],'--output',str(raw),'--steps','5','--seed','222']
            with (item/'worker.log').open('w') as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                write(item/'process.json', dict(pid=process.pid, command=command, started=time.time(), gpu_uuid=GPU))
                code = process.wait()
            write(item/'exit.json', dict(returncode=code, ended=time.time()))
            assert code == 0, 'Inference failed; inspect '+str(item/'worker.log')
            result = json.loads(raw.with_suffix('.gpu.json').read_text())
            assert result['success'] and result['conditioning_pcm_exact']
            assert result['conditioning_audio_sha256']==row['audio_sha256'] and result['conditioning_image_sha256']==row['image_sha256']
            assert result['feature_frames']==result['pose_frames']==row['expected_frames']
            assert result['arguments']['seed']==result['native_model_seed']==222
            assert result['cuda_forward_module_types'] and result['face_detector_delegate']=='CPU'
            assert all(not r['missing_keys'] and not r['unexpected_keys'] for r in result['state_dict_results'])
            write(item/'worker_result.json',dict(result,status='PASS'))
            subprocess.run([str(B/'tools/bin/ffmpeg'), '-nostdin', '-v', 'error', '-y', '-i', str(raw),
                 '-i', row['audio_path'], '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'alac', str(video)], check=True)
        raw_pts=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(raw)]))
        score_pts=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))
        assert raw_pts==score_pts
        def video_hash(p):return subprocess.check_output([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-i',str(p),'-map','0:v:0','-c:v','copy','-f','hash','-hash','sha256','-']).strip()
        assert video_hash(raw)==video_hash(video)
        validation = validate(video, row)
        record = dict(id=row['id'], source=row['source'], status='PASS', audio_sha256=row['audio_sha256'], input_key=key, image_sha256=row['image_sha256'],
            video_path=str(video), video_sha256=sha(video), raw_video_sha256=sha(raw),
            **validation, unsupported_tail_samples=row['samples']-validation['frames']*640,
            conditioning_audio_sha256=row['audio_sha256'], feature_frames=result['feature_frames'], pose_frames=result['pose_frames'],
            reused_from=reuse, seed=222, seconds=time.monotonic()-start,
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
        assert len(completed) == len(rows)
        write(RUN/'generation_complete.json', dict(status='PASS', completed=len(rows),
              unique_inputs=len(previous), new_tfg_forwards=len(rows), receipts=completed))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(RUN/'generation_failure.json', dict(error=repr(error), time=time.time()))
        raise
