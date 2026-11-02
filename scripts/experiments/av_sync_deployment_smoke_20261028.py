"""Real-media, strict-weight deployment smoke; not a synchronization validity test."""
import argparse, json, os, sys, subprocess, time
from pathlib import Path
import numpy as np
import torch

DEPLOY = Path('/home/wjj/av-sync-eval-20261028')
ROOT = Path('/home/wjj/tts-audio/tts-exp')
SOURCE_VIDEO = ROOT/'runs/qwen_codec_roundtrip_20261022_attempt01/video/BAC009S0002W0122/N.mp4'
SOURCE_AUDIO = ROOT/'runs/qwen_codec_roundtrip_20261022_attempt01/audio/BAC009S0002W0122/N_16k_pcm16.wav'

def mtd():
    import cv2, soundfile as sf, torchaudio, torchvision
    from torchaudio.transforms import MelScale
    source = DEPLOY/'MTDVocaLiST'
    sys.path.insert(0,str(source)); os.chdir(source)
    from models.student_thin_200_all import SyncTransformer
    from hparams import hparams
    weights=DEPLOY/'weights/mtdvocalist/pure_MTDVocaLiST.pth'
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    model=SyncTransformer(d_model=200)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    sys.path.append(str(ROOT/'third_party/Wav2Lip'))
    import face_detection
    capture=cv2.VideoCapture(str(SOURCE_VIDEO));frames=[]
    for i in range(5):
        ok,frame=capture.read()
        if not ok:raise RuntimeError('Not enough real video frames')
        frames.append(frame)
    capture.release()
    detector=face_detection.FaceAlignment(face_detection.LandmarksType._2D,flip_input=False,device='cuda')
    boxes=[detector.get_detections_for_batch(np.asarray([frame]))[0] for frame in frames]
    prep=DEPLOY/'smoke/mtd_faces';prep.mkdir(parents=True,exist_ok=True)
    image_tensors=[]
    resize=torchvision.transforms.Resize((96,96))
    for i,(frame,box) in enumerate(zip(frames,boxes)):
        if box is None:raise RuntimeError('Face crop unavailable')
        x1,y1,x2,y2=box
        path=prep/f'{i}.jpg';cv2.imwrite(str(path),frame[y1:y2,x1:x2])
        image_tensors.append(resize(torchvision.io.read_image(str(path))))
    del detector
    video=torch.cat(image_tensors,0)[:,48:].float().unsqueeze(0).cuda()
    audio,sr=sf.read(SOURCE_AUDIO,dtype='float32')
    if sr!=16000 or audio.ndim!=1:raise RuntimeError('Unexpected smoke audio format')
    wave=torch.from_numpy(audio)
    spec=torch.stft(wave,n_fft=hparams.n_fft,hop_length=hparams.hop_size,win_length=hparams.win_size,
                    window=torch.hann_window(hparams.win_size),return_complex=True)
    mel=MelScale(n_mels=80,sample_rate=16000,f_min=hparams.fmin,f_max=hparams.fmax,n_stft=hparams.n_stft,
                 norm='slaney',mel_scale='slaney')(torch.abs(spec))
    db=20*torch.log10(torch.clamp(mel,min=float(np.exp(100/-20*np.log(10)))))-20
    normalized=torch.clip(8*((db+100)/100)-4,-4,4)
    audio_input=normalized[None,None,:,:16].cuda()
    model=model.eval().cuda()
    with torch.inference_mode(): logits,features=model(video,audio_input)
    if not torch.isfinite(logits).all():raise RuntimeError('Non-finite MTD forward')
    return dict(model='MTDVocaLiST',strict_load=True,source_commit='5795416f9277094d3ae6215596a436a0a01f79ad',
                parameters=sum(p.numel() for p in model.parameters()),input_video_shape=list(video.shape),
                input_audio_shape=list(audio_input.shape),raw_logit=logits.cpu().tolist(),
                face_detection_batch_size=1,
                note='Author README model; author test_stu RGB 0-255 and mel recipe, S3FD crop adapter. One real 5-frame window only.')

def synchformer():
    source=DEPLOY/'Synchformer-source';sys.path.insert(0,str(source));os.chdir(source)
    import example
    from omegaconf import OmegaConf
    # The official reader averages audio before casting: integer PCM fails in
    # recent PyTorch. Float PCM preserves the waveform and supplies its expected dtype.
    media=DEPLOY/'smoke/synchformer_input_f32.mov';media.parent.mkdir(parents=True,exist_ok=True)
    if not media.exists():
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(SOURCE_VIDEO),'-i',str(SOURCE_AUDIO),
                        '-map','0:v:0','-map','1:a:0','-t','5','-vf',"fps=25,scale=iw*256/'min(iw,ih)':ih*256/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2",
                        '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','pcm_f32le','-ar','16000','-ac','1',str(media)],check=True)
    cfg=example.patch_config(OmegaConf.load(DEPLOY/'weights/synchformer/cfg-23-12-23T18-33-57.yaml'))
    model,_=example.get_model(cfg,torch.device('cuda'))
    checkpoint=torch.load(DEPLOY/'weights/synchformer/23-12-23T18-33-57.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    rgb,audio,meta=example.get_video_and_audio(str(media),get_meta=True)
    item=dict(video=rgb,audio=audio,meta=meta,path=str(media),split='test',targets=dict(v_start_i_sec=0.,offset_sec=0.))
    item=example.get_transforms(cfg,['test'])['test'](item)
    batch=torch.utils.data.default_collate([item]);aud,vid,targets=example.prepare_inputs(batch,torch.device('cuda'))
    with torch.inference_mode(),torch.autocast('cuda',enabled=cfg.training.use_half_precision):_,logits=model(vid,aud)
    if not torch.isfinite(logits).all():raise RuntimeError('Non-finite Synchformer forward')
    grid=example.make_class_grid(-cfg.data.max_off_sec,cfg.data.max_off_sec,logits.shape[-1])
    return dict(model='Synchformer LRS3',strict_load=True,source_commit='b66668a1521d7567cc760e5544b2b5b53179b687',
                parameters=sum(p.numel() for p in model.parameters()),input_video_shape=list(vid.shape),
                input_audio_shape=list(aud.shape),logits=logits.cpu().tolist(),
                predicted_offset_s=float(grid[logits.argmax(-1).item()]),
                audio_container_codec='pcm_f32le',
                note='Official patch_config/get_model/test transforms/prepare_inputs/forward; float PCM media adapter, PyAV12.3 compatibility runtime, no benchmark validity claim.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['mtd','synchformer'],required=True);a=p.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    result=mtd() if a.model=='mtd' else synchformer()
    result.update(status='PASS',deployment_smoke_only=True,scientific_validity_test=False,seconds=time.monotonic()-start,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__)
    dest=DEPLOY/'smoke'/f'{a.model}_receipt.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
