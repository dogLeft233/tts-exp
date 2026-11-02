"""Unmodified official CLI, with seed/resource bounds and recording hooks."""
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import threading
import time
from prepare import B, SOURCE, GPU, RUN, sha, write

def main():
    item_id = sys.argv[1]
    row = next(r for r in json.loads((RUN/'inputs.json').read_text())['rows'] if r['id'] == item_id)
    item = RUN/'generation'/item_id
    os.chdir(SOURCE)
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    import torch
    import onnxruntime as ort
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.45)
    assert torch.cuda.device_count() == 1
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    sessions=[]
    original_session=ort.InferenceSession
    class CheckedSession(original_session):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            providers=self.get_providers()
            sessions.append(dict(model=str(args[0]),providers=providers))
            assert 'CUDAExecutionProvider' in providers
    ort.InferenceSession=CheckedSession
    from pctalk.audio_features import AudioFeatureExtractor
    from pctalk.pipeline import PCTalkPipeline
    from pctalk.liveportrait.utils.face_analysis_diy import FaceAnalysisDIY
    extract=AudioFeatureExtractor.extract
    generate=PCTalkPipeline.generate_motion
    face_get=FaceAnalysisDIY.get
    details=dict(conditioning_audio_sha256=sha(row['audio_path']),conditioning_image_sha256=sha(row['image_path']),faces=[])
    def observed_features(engine,path):
        features=extract(engine,path)
        np.save(item/'audio_features.npy',features)
        details.update(feature_frames=len(features),feature_shape=list(features.shape),feature_sha256=sha(item/'audio_features.npy'))
        return features
    def observed_motion(engine,*args,**kwargs):
        motion=generate(engine,*args,**kwargs)
        details['pose_frames']=len(motion['motion'])
        return motion
    def observed_faces(engine,image,**kwargs):
        result=face_get(engine,image,**kwargs)
        if result:
            details['faces'].append([dict(bbox=f.bbox.tolist(),score=float(f.det_score)) for f in result])
        return result
    AudioFeatureExtractor.extract=observed_features
    PCTalkPipeline.generate_motion=observed_motion
    FaceAnalysisDIY.get=observed_faces
    protocol=json.loads((RUN/'protocol.json').read_text())
    argv=['--source',row['image_path'],'--audio',row['audio_path'],
          '--person-id','192','--lip-scale','1.0','--face-index',str(row['face_index']),'--crop-scale','2.3',
          '--output-dir',str(item/'official_output')]
    write(item/'invocation.json',dict(argv=argv,seed=42,gpu_uuid=GPU,worker_sha256=sha(__file__),
        cpu_affinity=sorted(os.sched_getaffinity(0)),modifications='Recording hooks and seed/resource bounds only; no model source edits.'))
    peak=0
    stop=threading.Event()
    start=time.monotonic()
    def monitor():
        nonlocal peak
        with (item/'vram_samples.jsonl').open('w') as output:
            while not stop.is_set():
                try:
                    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()
                    owned=[int(x.split(',')[2]) for x in lines if x.split(',')[0].strip()==GPU and int(x.split(',')[1])==os.getpid()]
                    peak=max([peak,*owned])
                    output.write(json.dumps(dict(elapsed=time.monotonic()-start,own_vram_mib=max(owned,default=0)))+'\n')
                    output.flush()
                except Exception as error:
                    output.write(json.dumps(dict(monitor_error=repr(error)))+'\n')
                stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    status='FAILED'
    try:
        sys.argv=['pctalk',*argv]
        runpy.run_module('pctalk',run_name='__main__')
        torch.cuda.synchronize()
        assert details['feature_frames']==details['pose_frames']==row['expected_frames']
        files=list((item/'official_output').glob('*.mp4'))
        assert len(files)==1,files
        shutil.copyfile(files[0],item/'generated_aac.mp4')
        assert sessions
        status='PASS'
    finally:
        stop.set()
        thread.join(7)
        write(item/'worker_result.json',dict(status=status,**details,pid=os.getpid(),
            elapsed_s=time.monotonic()-start,peak_nvidia_own_pid_mib=peak,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            onnx_sessions=sessions,torch=torch.__version__))

if __name__=='__main__':
    main()
