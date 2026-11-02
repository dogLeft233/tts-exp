"""Native Flash pipeline with fixed RMS24 audio and explicit temporal support."""
from pathlib import Path
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import fcntl
from prepare import B, D, RUN, native_frames, sha, write


def load_pipeline():
    import torch
    import infer_flash as upstream
    from attention import install
    install()
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
    from src.wav2vec2 import Wav2Vec2Model
    # Large LFS files match official SHA256; small files match official Git blobs.
    manifest = json.loads((D/'download_manifest.json').read_text())
    download = json.loads((D/'download_receipt.json').read_text())
    assert download['status'] == 'PASS' and len(download['files']) == len(manifest)
    import hashlib
    verified = []
    for entry in manifest:
        asset = D/'models'/entry['target']
        assert asset.stat().st_size == entry['bytes']
        actual = sha(asset)
        if entry['sha256']:
            assert actual == entry['sha256']
        else:
            data = asset.read_bytes()
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == entry['git_blob_id']
        verified.append(dict(path=entry['target'],bytes=entry['bytes'],sha256=actual))
    write(RUN/'official_assets_validation.json', dict(status='PASS',files=verified))
    cfg = OmegaConf.load(D/'repo/config/config.yaml')
    base = D/'models/Wan2.1-Fun-V1.1-1.3B-InP'
    # Flash is a full checkpoint, so loading and immediately replacing the3GB
    # base DiT would add no parameters to the final model. Use its own config.
    flash_config = json.loads((D/'models/transformer/config.json').read_text())
    transformer = upstream.WanTransformer.from_config(flash_config)
    # The released Flash checkpoint is complete; no random/missing adapter keys.
    weights = load_file(str(D/'models/transformer/diffusion_pytorch_model.safetensors'))
    transformer.load_state_dict(weights, strict=True)
    del weights
    transformer = transformer.to(dtype=torch.bfloat16).eval()
    vae = upstream.AutoencoderKLWan.from_pretrained(str(base/cfg.vae_kwargs.vae_subpath),
        additional_kwargs=OmegaConf.to_container(cfg.vae_kwargs)).to(torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(base/cfg.text_encoder_kwargs.tokenizer_subpath), local_files_only=True)
    text = upstream.WanT5EncoderModel.from_pretrained(str(base/cfg.text_encoder_kwargs.text_encoder_subpath),
        additional_kwargs=OmegaConf.to_container(cfg.text_encoder_kwargs), low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16).to(torch.bfloat16).eval()
    clip = upstream.CLIPModel.from_pretrained(str(base/cfg.image_encoder_kwargs.image_encoder_subpath)).to(torch.bfloat16).eval()
    dtypes = {name:str(module.dtype) for name,module in dict(transformer=transformer,vae=vae,text_encoder=text,clip_image_encoder=clip).items()}
    assert set(dtypes.values()) == {'torch.bfloat16'}, dtypes
    scheduler_kwargs = OmegaConf.to_container(cfg.scheduler_kwargs)
    scheduler_kwargs['shift'] = 1
    scheduler = upstream.FlowUniPCMultistepScheduler(**upstream.filter_kwargs(upstream.FlowUniPCMultistepScheduler, scheduler_kwargs))
    pipe = upstream.WanFunInpaintAudioPipeline(transformer=transformer, vae=vae, tokenizer=tokenizer,
        text_encoder=text, scheduler=scheduler, clip_image_encoder=clip)
    pipe.enable_model_cpu_offload(gpu_id=0)
    wavdir = D/'models/chinese-wav2vec2-base'
    encoder = Wav2Vec2Model.from_pretrained(str(wavdir), local_files_only=True).eval().to('cpu')
    encoder.feature_extractor._freeze_parameters()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(wavdir), local_files_only=True)
    gc.collect()
    write(RUN/'model_load.json', dict(status='PASS', flash_strict_load=True,
        checkpoint_sha256=sha(D/'models/transformer/diffusion_pytorch_model.safetensors'),
        audio_encoder_sha256=sha(wavdir/'pytorch_model.bin'), torch=torch.__version__,
        cuda=torch.version.cuda, memory_cap_gib=22., cpu_offload='model',
        upstream_infer_sha256=sha(D/'repo/infer_flash.py'),
        attention='PyTorch SDPA with upstream key/query length semantics; full noncausal attention only',
        attention_adapter_sha256=sha(Path(__file__).with_name('attention.py')),
        module_dtypes=dtypes, text_loader='CPU load then explicit BF16; avoid incompatible upstream low-memory meta loader',
        worker_sha256=sha(__file__)))
    return pipe, encoder, extractor, cfg, upstream


def validate(video, audio, expected_frames, size):
    import numpy as np
    import soundfile as sf
    import cv2
    pcm, sr = sf.read(audio, dtype='int16')
    actual = subprocess.check_output([str(B/'tools/bin/ffmpeg'), '-nostdin','-v','error','-i',str(video),'-map','0:a:0','-f','s16le','-'])
    assert sr == 16000 and actual == pcm.astype('<i2').tobytes(), 'PCM mismatch'
    cap = cv2.VideoCapture(str(video))
    assert abs(cap.get(cv2.CAP_PROP_FPS)-25) < 1e-6
    frames = 0
    motion = []
    last = None
    while True:
        ok, image = cap.read()
        if not ok:
            break
        assert list(image.shape[:2]) == size
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if last is not None:
            motion.append(float(abs(gray-last).mean()))
        last = gray
        frames += 1
    cap.release()
    assert frames == expected_frames and motion and np.mean(motion) > .01
    return dict(frames=frames, samples=len(pcm), fps=25, resolution=size,
        decoded_pcm_exact=True, mean_frame_difference=float(np.mean(motion)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    lock = (RUN/'experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(D/'repo')
    sys.path.insert(0,str(D/'repo'))
    import torch
    import numpy as np
    import soundfile as sf
    from PIL import Image
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1
    cap_fraction = 22*1024**3/torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap_fraction)
    rows = json.loads((RUN/'inputs.json').read_text())['rows']
    protocol = json.loads((RUN/'protocol.json').read_text())
    g = protocol['generation']
    assert sha(g['face_path']) == g['face_sha256']
    pipe, encoder, extractor, cfg, upstream = load_pipeline()
    face = Image.open(g['face_path']).convert('RGB')
    size = upstream.get_sample_size(face, [512,512])
    assert size == [512,512]
    previous = {}
    complete = []
    for index, row in enumerate(rows):
        if a.limit and len(complete) >= a.limit:
            break
        key = row['audio_sha256']
        assert sha(row['audio_path']) == key
        item = RUN/'generation'/row['id']
        item.mkdir(parents=True, exist_ok=True)
        receipt = item/'receipt.json'
        video = RUN/'videos'/(row['id']+'.mp4')
        video.parent.mkdir(exist_ok=True)
        raw = item/'generated_aac.mp4'
        if receipt.exists():
            rec = json.loads(receipt.read_text())
            assert sha(video) == rec['video_sha256'] and rec['audio_sha256'] == key
            previous.setdefault(key,(raw,video,row['id']))
            complete.append(rec)
            continue
        started = time.monotonic()
        frames = native_frames(row['samples'])
        reuse = None
        torch.cuda.reset_peak_memory_stats()
        if key in previous:
            prev_raw, prev_video, reuse = previous[key]
            shutil.copyfile(prev_raw, raw)
            shutil.copyfile(prev_video, video)
        else:
            pcm, sr = sf.read(row['audio_path'], dtype='int16')
            # librosa's PCM16 float32 conversion, without the extra LUFS gain.
            waveform = pcm.astype(np.float32)/32768
            supported = waveform[:frames*640]
            assert len(supported) == frames*640
            with torch.no_grad():
                audio_embeds = upstream.get_audio_embed(supported, extractor, encoder, frames, sr=16000, fps=25, device='cpu')
                audio_embeds = audio_embeds.to(device='cuda',dtype=torch.bfloat16)
                indices = torch.arange(frames).unsqueeze(1)+(torch.arange(5)-2).unsqueeze(0)
                indices = torch.clamp(indices, min=0, max=audio_embeds.shape[0]-1)
                audio_embeds = audio_embeds[indices].unsqueeze(0)
                input_video, mask, clip_image = upstream.get_image_to_video_latent2(face,None,video_length=frames,sample_size=size)
                # Fresh generator, TeaCache and scheduler timesteps for each item.
                coefficients = upstream.get_teacache_coefficients(str(D/'models/Wan2.1-Fun-V1.1-1.3B-InP'))
                pipe.transformer.enable_teacache(coefficients,8,.1,num_skip_start_steps=5,offload=False)
                generator = torch.Generator(device='cuda').manual_seed(42)
                sample = pipe(g['prompt'],negative_prompt='',num_frames=frames,audio_embeds=audio_embeds,
                    audio_scale=1.,ip_mask=None,use_un_ip_mask=False,height=size[0],width=size[1],generator=generator,
                    neg_scale=1.,neg_steps=0,use_dynamic_cfg=False,use_dynamic_acfg=False,guidance_scale=6.,
                    audio_guidance_scale=3.,num_inference_steps=8,video=input_video,mask_video=mask,
                    clip_image=clip_image,cfg_skip_ratio=0.,shift=5.).videos
                silent = item/'silent.mp4'
                upstream.save_videos_grid(sample[:,:,:frames],str(silent),fps=25)
                del sample, audio_embeds, input_video, mask, clip_image
            for codec, target in [('aac',raw),('alac',video)]:
                subprocess.run([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-y','-i',str(silent),'-i',row['audio_path'],
                    '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a',codec,str(target)],check=True)
            silent.unlink()
            gc.collect()
            torch.cuda.empty_cache()
        media = validate(video,row['audio_path'],frames,size)
        rec = dict(id=row['id'],source=row['source'],status='PASS',audio_sha256=key,
            video_path=str(video),video_sha256=sha(video),raw_video_sha256=sha(raw),**media,
            unsupported_tail_samples=row['samples']-frames*640,
            conditioning_prefix_samples=frames*640, conditioning_prefix_pcm_sha256=__import__('hashlib').sha256(sf.read(row['audio_path'],dtype='int16')[0][:frames*640].astype('<i2').tobytes()).hexdigest(),
            reused_from=reuse,seed=42,seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
        write(receipt,rec)
        previous.setdefault(key,(raw,video,row['id']))
        complete.append(rec)
        write(RUN/'generation_status.json',dict(status='RUNNING',completed=len(complete),planned=80,last_id=row['id']))
        print('GENERATED',index+1,row['id'],'reuse',reuse,'seconds',rec['seconds'],flush=True)
    if len(complete) == 80:
        write(RUN/'generation_complete.json',dict(status='PASS',completed=80,actual_unique_forwards=len(previous),receipts=complete))
    else:
        write(RUN/'smoke_complete.json',dict(status='PASS',completed=len(complete),receipts=complete))


if __name__ == '__main__':
    main()
