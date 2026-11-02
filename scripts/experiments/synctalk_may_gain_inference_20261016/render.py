"""Official SyncTalk renderer, frozen May checkpoint and paired gain inputs."""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch

ROOT=Path('/dev/shm/synctalk-may-gain-inference-20261016')
AUDIT=Path('/home/wjj/tts-audio/synctalk-amplitude-audit-20261016')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def official_parser():
    tree=ast.parse((ROOT/'official/main.py').read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If) and any(
        isinstance(x,ast.Name) and x.id=='__name__' for x in ast.walk(n.test)))
    nodes=[]
    for n in block.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='opt' for t in n.targets):break
        nodes.append(n)
    scope={'argparse':argparse}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official/main.py','exec'),scope)
    return scope['parser']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    os.chdir(ROOT/'official')
    from nerf_triplane.provider import NeRFDataset
    from nerf_triplane.network import NeRFNetwork
    from nerf_triplane.utils import seed_everything
    protocol=json.loads((ROOT/'protocol.json').read_text())
    src=ROOT/'data/May'
    raw,sr=sf.read(src/'aud.wav',dtype='float64')
    integers,_=sf.read(src/'aud.wav',dtype='int16')
    assert sr==16000 and raw.shape[1]==2 and np.array_equal(raw[:,0],raw[:,1])
    wave=raw[:,0]
    start=protocol['score_audio_start_sample'];n=protocol['score_audio_samples']
    pcm=integers[start:start+n,0].copy()
    score_wav=a.output/'original_score_audio.wav'
    sf.write(score_wav,pcm,sr,subtype='PCM_16')
    assert np.array_equal(sf.read(score_wav,dtype='int16')[0],pcm)
    checkpoint=AUDIT/'assets/trial_may/checkpoints/ngp.pth'
    assert sha(checkpoint)==protocol['checkpoint_sha256']
    result={'protocol_sha256':sha(ROOT/'protocol.json'),'checkpoint_sha256':sha(checkpoint),
            'source_wav_sha256':sha(src/'aud.wav'),'score_audio_sha256':sha(score_wav),
            'score_pcm_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),
            'score_audio_samples':len(pcm),'arms':[],'smoke':a.smoke}
    arms=[x for x in protocol['arms'] if x['id']=='gain_0'] if a.smoke else protocol['arms']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    for arm in arms:
        out=a.output/arm['id'];out.mkdir()
        data=out/'data';data.mkdir()
        for name in ['bs.npy','bc.jpg','transforms_val.json','gt_imgs','ori_imgs','torso_imgs']:
            (data/name).symlink_to(src/name,target_is_directory=(src/name).is_dir())
        gain=10**(arm['db']/20)
        intended=wave*gain
        assert np.max(np.abs(intended))<1
        sf.write(data/'aud.wav',intended,sr,subtype='DOUBLE')
        reread,_=sf.read(data/'aud.wav',dtype='float64')
        assert np.array_equal(reread,intended)
        opt=official_parser().parse_args([str(data),'-O','--test','--asr_model','ave',
                                          '--data_range','0','300'])
        opt.fp16=True;opt.exp_eye=True;opt.cuda_ray=True
        seed_everything(opt.seed)
        model=NeRFNetwork(opt).cuda().eval()
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        loaded=model.load_state_dict(state['model'],strict=False)
        assert set(loaded.missing_keys)=={'density_grid'},loaded.missing_keys
        assert not loaded.unexpected_keys,loaded.unexpected_keys
        assert all(name in state['model'] for name,_ in model.named_parameters())
        for field in ['mean_count','mean_density','mean_density_torso']:
            if field in state:setattr(model,field,state[field])
        assert model.density_bitfield.count_nonzero().item()>0
        dataset=NeRFDataset(opt,device=torch.device('cuda'),type='test')
        assert len(dataset.poses)==300 and len(dataset.auds)==300
        model.aud_features=dataset.auds
        model.eye_areas=dataset.eye_area
        model.poses=dataset.poses
        frames=[];rgbhash=hashlib.sha256()
        video=out/'render_lossless.mkv'
        proc=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24',
                               '-s',f'{dataset.W}x{dataset.H}','-r','25','-i','pipe:0','-an',
                               '-c:v','ffv1','-level','3','-g','1','-pix_fmt','bgr0','-threads','2',str(video)],
                              stdin=subprocess.PIPE)
        t=time.monotonic()
        limit=1 if a.smoke else 300
        with torch.no_grad():
            for i in range(limit):
                batch=dataset.collate([i])
                model.testing=True
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=opt.fp16):
                    output=model.render(batch['rays_o'],batch['rays_d'],batch['auds'],
                                        batch['bg_coords'],batch['poses'],eye=batch['eye'],
                                        index=batch['index'],staged=True,bg_color=batch['bg_color'],
                                        perturb=False,**vars(opt))
                model.testing=False
                torch.cuda.synchronize()
                rgb=output['image'].reshape(dataset.H,dataset.W,3).float().cpu().numpy()
                assert np.isfinite(rgb).all()
                pixels=(rgb*255).astype(np.uint8)
                rgbhash.update(pixels.tobytes())
                frames.append({'frame':i,'source_img_id':5520+i,
                               'float_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                               'rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                               'pixel_mean':float(pixels.mean()),'pixel_std':float(pixels.std())})
                proc.stdin.write(pixels.tobytes())
                if i in [0,149,299]:cv2.imwrite(str(out/f'frame{i:03d}.png'),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
                if i%25==0:print(arm['id'],'frame',i,'elapsed',round(time.monotonic()-t,1),flush=True)
        proc.stdin.close();assert proc.wait()==0
        # Lossless formal container carries the same original PCM, never the gained driver.
        scored=out/'original_pcm.mkv'
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(video),'-i',str(score_wav),
                        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le',str(scored)],check=True)
        meta={'arm':arm['id'],'gain_db':arm['db'],'driver_peak':float(np.max(np.abs(intended))),
              'driver_rms':float(np.sqrt(np.mean(intended**2))),
              'scalar_max_error':float(np.max(np.abs(reread-wave*gain))),
              'driver_sha256':sha(data/'aud.wav'),'audio_features_shape':list(dataset.auds.shape),
              'audio_features_sha256':hashlib.sha256(dataset.auds.cpu().numpy().tobytes()).hexdigest(),
              'missing_unneeded_buffer':list(loaded.missing_keys),'frames':frames,
              'rgb_stream_sha256':rgbhash.hexdigest(),'video_sha256':sha(video),
              'original_pcm_container_sha256':sha(scored),'seconds':time.monotonic()-t,
              'H':dataset.H,'W':dataset.W,'fps':25,'fp16':opt.fp16,
              'max_steps':opt.max_steps,'dt_gamma':opt.dt_gamma}
        (out/'metadata.json').write_text(json.dumps(meta,indent=2));result['arms'].append(meta)
        (a.output/'render_receipt.json').write_text(json.dumps(result,indent=2))
        print('Completed',arm['id'],round(meta['seconds'],1),flush=True)
        del model,dataset,state
        torch.cuda.empty_cache()


if __name__=='__main__':main()
