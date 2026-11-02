"""Render fixed-final EMA weights on held-out validation and all test gains."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import soundfile as sf
import torch
from torch_ema import ExponentialMovingAverage

from common import ROOT,DATA,configure,dataset,label,load_metadata,save_json,sha


class Video:
    def __init__(self,path,audio,frames):
        self.path=path
        self.p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24',
            '-s','512x512','-r','25','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0',
            '-frames:v',str(frames),'-c:v','ffv1','-level','3','-pix_fmt','bgr0','-c:a','pcm_s16le',str(path)],
            stdin=subprocess.PIPE)
    def write(self,rgb):self.p.stdin.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR).tobytes())
    def close(self):
        self.p.stdin.close();assert self.p.wait()==0


def metrics(pred,truth,source_id):
    error=pred.astype(np.float64)-truth.astype(np.float64)
    lms=np.loadtxt(DATA/f'source/May/ori_imgs/{source_id}.lms')
    x0,y0=np.maximum(np.floor(lms[48:68].min(0)).astype(int)-12,0)
    x1,y1=np.minimum(np.ceil(lms[48:68].max(0)).astype(int)+13,512)
    mouth=error[y0:y1,x0:x1]
    mse=float(np.mean(error**2))
    return {'MSE':mse,'L1':float(np.mean(np.abs(error))),'PSNR':float(-10*np.log10(max(mse,1e-12))),
            'mouth_bbox_MSE':float(np.mean(mouth**2)),'mouth_bbox_L1':float(np.mean(np.abs(mouth)))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-gain',type=int,required=True);a=p.parse_args()
    configure()
    torch.set_grad_enabled(False)
    from nerf_triplane.network import NeRFNetwork
    from nerf_triplane.utils import Trainer
    protocol=json.loads((ROOT/'protocol.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text())
    receipt=json.loads((ROOT/'training'/label(a.train_gain)/'training_receipt.json').read_text())
    checkpoint=Path(receipt['checkpoint']);assert sha(checkpoint)==receipt['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==protocol['steps_per_model']==60000
    target=ROOT/'renders'/('train_'+label(a.train_gain));assert not target.exists();target.mkdir(parents=True)
    pcm,sr=sf.read(DATA/'source/May/aud.wav',dtype='int16',always_2d=True)
    test_info=next(s for s in manifest['splits'] if s['name']=='test')
    lo,hi=test_info['supervision_start_frame'],test_info['supervision_end_frame_exclusive']
    test_pcm=pcm[lo*640:hi*640,0]
    audio=target/'original_test_pcm.wav';sf.write(audio,test_pcm,sr,subtype='PCM_16')
    assert len(test_pcm)==1100*640 and np.array_equal(sf.read(audio,dtype='int16')[0],test_pcm)
    selected=json.loads((ROOT/'initialization.json').read_text())['validation_indices']
    for test_gain in protocol['test_gains_db']:
        cell=target/('test_'+label(test_gain));cell.mkdir()
        d=dataset('test',test_gain,protocol)
        model=NeRFNetwork(d.opt).cuda();model.load_state_dict(state['model'],strict=True)
        load_metadata(model,state['metadata'])
        ema=ExponentialMovingAverage(model.parameters(),decay=protocol['ema_decay'])
        ema.load_state_dict(state['ema']);ema.copy_to();model.eval()
        owner=SimpleNamespace(opt=d.opt,model=model,criterion=torch.nn.L1Loss(reduction='none'))
        frames=json.loads((Path(d.opt.path)/'transforms_test.json').read_text())['frames']
        rows=[];writer=Video(cell/'original_pcm.mkv',audio,len(frames))
        with torch.no_grad():
            for i,frame in enumerate(frames):
                result=Trainer.eval_step(owner,d.collate([i]))
                pred=result[0][0].cpu().numpy().astype(np.float32)
                truth=result[5][0].cpu().numpy().astype(np.float32)
                assert np.isfinite(pred).all()
                assert pred.min()>=-1e-6 and pred.max()<=1+1e-6
                rgb=np.rint(np.clip(pred,0,1)*255).astype(np.uint8)
                writer.write(rgb)
                rows.append({'index':i,'source_id':frame['img_id'],
                    'float_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),
                    'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                    **metrics(pred,truth,frame['img_id'])})
                if i in [0,len(frames)//2,len(frames)-1]:
                    cv2.imwrite(str(cell/f'frame{i:04d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        writer.close()
        # Pixel-exact FFV1 and original-PCM checks before declaring the cell complete.
        cap=cv2.VideoCapture(str(cell/'original_pcm.mkv'));count=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==rows[count]['rgb_sha256'];count+=1
        cap.release();assert count==1100
        decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(cell/'original_pcm.mkv'),
                        '-map','0:a:0','-f','s16le','-acodec','pcm_s16le','pipe:1'])
        assert decoded==test_pcm.tobytes()
        summary={k:float(np.mean([r[k] for r in rows])) for k in
                 ['MSE','L1','PSNR','mouth_bbox_MSE','mouth_bbox_L1']}
        record={'training_gain_db':a.train_gain,'test_gain_db':test_gain,'checkpoint_sha256':sha(checkpoint),
                'checkpoint_step':state['step'],'weights':'EMA','frames':rows,'summary':summary,
                'pcm_sha256':hashlib.sha256(test_pcm.tobytes()).hexdigest(),
                'mux_sha256':sha(cell/'original_pcm.mkv'),'pcm_readback_exact':True,'rgb_readback_exact':True}
        save_json(cell/'render_receipt.json',record)
        # Repeat the first fixed scored clip, above floating-point and codec precision.
        if a.train_gain==test_gain==0:
            clip=next(c for c in manifest['eval_clips'] if c['split']=='test')
            start=clip['source_start_frame']-lo
            repeat_audio=DATA/clip['score_audio_path']
            repeat=Video(cell/'repeat_first_clip.mkv',repeat_audio,clip['frames'])
            for i in range(start,start+clip['frames']):
                pred=Trainer.eval_step(owner,d.collate([i]))[0][0].cpu().numpy().astype(np.float32)
                rgb=np.rint(np.clip(pred,0,1)*255).astype(np.uint8)
                assert hashlib.sha256(pred.tobytes()).hexdigest()==rows[i]['float_sha256']
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==rows[i]['rgb_sha256']
                repeat.write(rgb)
            repeat.close()
            save_json(cell/'repeat_receipt.json',{'frames':clip['frames'],'float_bitwise_equal':True,
                                                'rgb_bitwise_equal':True,'clip_id':clip['id']})
        # Full independent validation set, with raw final predictions at curve frames.
        v=dataset('val',test_gain,protocol)
        vf=json.loads((Path(v.opt.path)/'transforms_val.json').read_text())['frames']
        owner.opt=v.opt;vrows=[];samples=[]
        with torch.no_grad():
            for i,frame in enumerate(vf):
                result=Trainer.eval_step(owner,v.collate([i]))
                pred=result[0][0].cpu().numpy().astype(np.float32)
                truth=result[5][0].cpu().numpy().astype(np.float32)
                vrows.append({'index':i,'source_id':frame['img_id'],**metrics(pred,truth,frame['img_id'])})
                if i in selected:samples.append(pred.copy())
        np.save(cell/'final_validation_selected_float.npy',np.stack(samples))
        save_json(cell/'full_validation.json',{'frames':vrows,'selected_indices':selected,
                  'sample_path':'final_validation_selected_float.npy',
                  'summary':{k:float(np.mean([r[k] for r in vrows])) for k in summary}})
        print(json.dumps({'rendered':str(cell),'summary':summary}),flush=True)
        del model,owner,d,v,ema
        torch.cuda.empty_cache()


if __name__=='__main__':main()
