"""One fresh official inference process; wrappers only seed and record diagnostics."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

from prepare import B, D, GPU, RUN, sha, write


def main():
    item_id = sys.argv[1]
    row = next(r for r in json.loads((RUN/'inputs.json').read_text())['rows'] if r['id'] == item_id)
    item = RUN/'generation'/item_id
    os.chdir(D/'TalkLikeYou')
    sys.path.insert(0, str(D/'TalkLikeYou'))
    import numpy as np
    import torch
    import pyximport
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.60)
    assert torch.cuda.device_count() == 1
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    pyximport.install(build_dir=str(D/'cache/pyxbld'))
    from talklikeyou import pipeline as official
    feature_extract = official.MotionGenerationEngine.extract_audio_features
    pose_compute = official.compute_base_motion
    details = dict(conditioning_audio_sha256=sha(row['audio_path']))

    def observed_features(engine, path):
        features = feature_extract(engine, path)
        np.save(item/'audio_features.npy', features)
        details.update(feature_frames=len(features), feature_shape=list(features.shape),
                       feature_sha256=sha(item/'audio_features.npy'))
        return features

    def observed_pose(sdk, path):
        result = pose_compute(sdk, path)
        details['pose_frames'] = result[1]
        return result

    official.MotionGenerationEngine.extract_audio_features = observed_features
    official.compute_base_motion = observed_pose
    run_subprocess = official.subprocess.run

    def safe_ffmpeg(command, *args, **kwargs):
        if isinstance(command, list) and command and command[0] == 'ffmpeg':
            command = [str(B/'tools/bin/ffmpeg'), '-nostdin', *command[1:]]
        return run_subprocess(command, *args, **kwargs)

    official.subprocess.run = safe_ffmpeg
    protocol = json.loads((RUN/'protocol.json').read_text())
    argv = ['--source', row['face_path'], '--audio', row['audio_path'],
            '--person-id', '192', '--stage0', '--neutral', '--motion-scale', '1.0',
            '--guidance-scale', '1.3', '--sampling-steps', '1', '--pose-mode', 'audio',
            '--pose-sampling-steps', '1', '--seed', '42', '--overlap', '5',
            '--crop-scale', '2.3', '--face-index', '0', '--output', str(item/'generated_aac.mp4')]
    write(item/'invocation.json', dict(argv=argv, gpu_uuid=GPU, seed=42,
          official_pipeline_sha256=sha(D/'TalkLikeYou/talklikeyou/pipeline.py'),
          worker_sha256=sha(__file__), cpu_affinity=sorted(os.sched_getaffinity(0)),
          modifications='No upstream file edits. Wrappers set seed, record unchanged feature/pose outputs, and add ffmpeg -nostdin.'))
    peak = 0
    stop = threading.Event()
    start = time.monotonic()

    def monitor():
        nonlocal peak
        with (item/'vram_samples.jsonl').open('w') as output:
            while not stop.is_set():
                try:
                    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                            '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()
                    owned = [int(x.split(',')[2]) for x in lines if x.split(',')[0].strip() == GPU and int(x.split(',')[1]) == os.getpid()]
                    peak = max([peak, *owned])
                    output.write(json.dumps(dict(elapsed=time.monotonic()-start, own_vram_mib=max(owned, default=0)))+'\n')
                    output.flush()
                except Exception as error:
                    output.write(json.dumps(dict(monitor_error=repr(error)))+'\n')
                stop.wait(2)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    status = 'FAILED'
    try:
        sys.argv = ['talklikeyou', *argv]
        official.main()
        torch.cuda.synchronize()
        assert details['feature_frames'] == row['expected_frames']
        assert details['pose_frames'] >= details['feature_frames']
        status = 'PASS'
    finally:
        stop.set()
        monitor_thread.join(7)
        write(item/'worker_result.json', dict(status=status, **details, pid=os.getpid(),
            elapsed_s=time.monotonic()-start, peak_nvidia_own_pid_mib=peak,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), torch=torch.__version__))


if __name__ == '__main__':
    main()
