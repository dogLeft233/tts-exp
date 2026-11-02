"""Reuse a loaded official pipeline; prepare_params resets seed and motion per clip."""
from pathlib import Path
import json, os, sys, time, shutil, subprocess, types
import numpy as np, soundfile as sf
from prepare import B, RUN, sha, write

F=Path('/home/data2/wangjiajun/SoulX-FlashHead-Pro')

def main():
    os.chdir(F/'repo');sys.path.insert(0,str(F/'repo'))
    import torch
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60)
    assert torch.cuda.device_count()==1
    import flash_head.src.pipeline.flash_head_pipeline as pp
    pp.COMPILE_MODEL=False;pp.COMPILE_VAE=False
    import generate_video as official
    # Preserve the official generate body and fresh get_base_data on every item.
    pipe=official.get_pipeline(1,str(F/'repo/models/SoulX-FlashHead-1_3B'),'pro',str(F/'repo/models/wav2vec2-base-960h'))
    official.get_pipeline=lambda **kwargs:pipe
    original_run=official.subprocess.run
    def safe_run(cmd,*args,**kwargs):
        if isinstance(cmd,list) and cmd and cmd[0]=='ffmpeg':
            cmd=[cmd[0],'-nostdin',*cmd[1:]];kwargs['check']=True
        return original_run(cmd,*args,**kwargs)
    official.subprocess.run=safe_run
    rows=json.loads((RUN/'inputs.json').read_text())['rows'];protocol=json.loads((RUN/'protocol.json').read_text())
    previous={};completed=[]
    for index,row in enumerate(rows):
        key=row['audio_sha256'];item=RUN/'generation'/row['id'];item.mkdir(parents=True,exist_ok=True)
        raw=item/'generated_aac.mp4';video=RUN/'videos'/(row['id']+'.mp4');video.parent.mkdir(exist_ok=True)
        receipt=item/'receipt.json'
        if receipt.exists():
            r=json.loads(receipt.read_text());assert r['audio_sha256']==key and sha(video)==r['video_sha256']
            previous.setdefault(key,(raw,video,r['id']));completed.append(r);continue
        started=time.monotonic();reuse=None
        if key in previous:
            prevraw,prevvideo,reuse=previous[key];shutil.copyfile(prevraw,raw);shutil.copyfile(prevvideo,video)
        else:
            args=types.SimpleNamespace(ckpt_dir=str(F/'repo/models/SoulX-FlashHead-1_3B'),wav2vec_dir=str(F/'repo/models/wav2vec2-base-960h'),
                model_type='pro',cond_image_dir=None,cond_image=protocol['generation']['face_path'],base_seed=42,use_face_crop=False,
                audio_path=row['audio_path'],audio_encode_mode='stream',save_file=str(raw))
            torch.cuda.reset_peak_memory_stats();official.generate(args)
            subprocess.run([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-y','-i',str(raw),'-i',row['audio_path'],
                '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','alac',str(video)],check=True)
        pcm=np.frombuffer(subprocess.check_output([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-i',str(video),'-map','0:a:0','-f','s16le','-acodec','pcm_s16le','-']),dtype='<i2')
        expected,sr=sf.read(row['audio_path'],dtype='int16');assert np.array_equal(pcm,expected)
        import cv2
        cap=cv2.VideoCapture(str(video));frames=0;motion=[];last=None
        assert abs(cap.get(cv2.CAP_PROP_FPS)-25)<1e-6
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape[:2]==(512,512)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            if last is not None:motion.append(float(abs(gray.astype(float)-last).mean()))
            last=gray;frames+=1
        cap.release();assert frames>=len(pcm)//640 and motion and np.mean(motion)>0.01
        r=dict(id=row['id'],source=row['source'],status='PASS',audio_sha256=key,video_path=str(video),video_sha256=sha(video),
            raw_video_sha256=sha(raw),frames=frames,samples=len(pcm),decoded_pcm_exact=True,fps=25,
            mean_frame_difference=float(np.mean(motion)),reused_from=reuse,seed=42,seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
        write(receipt,r);previous.setdefault(key,(raw,video,row['id']));completed.append(r)
        write(RUN/'generation_status.json',dict(status='RUNNING',completed=len(completed),planned=80,last_id=row['id']))
        print('GENERATED',index+1,row['id'],'reuse',reuse,flush=True)
    write(RUN/'generation_complete.json',dict(status='PASS',completed=80,actual_unique_forwards=len(previous),receipts=completed))

if __name__=='__main__':main()
