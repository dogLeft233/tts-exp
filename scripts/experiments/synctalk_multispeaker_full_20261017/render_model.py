"""Pinned original FP16 test_step and portrait blending on fixed final EMA."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,soundfile as sf,torch
from runtime_common import ROOT,configure,dataset,tensor_state_sha
from cohort_binding import identity_root,verify_member_pins
from execution_state import restore_model_counters
from flow_stream import sha

class Video:
    def __init__(self,path,audio,frames=1500):
        self.process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','512x512','-r','25','-i','pipe:0',
            '-i',str(audio),'-map','0:v:0','-map','1:a:0','-frames:v',str(frames),'-c:v','ffv1','-level','3','-pix_fmt','bgr0','-c:a','pcm_s16le',str(path)],stdin=subprocess.PIPE)
    def write(self,rgb):self.process.stdin.write(rgb.tobytes())
    def close(self):self.process.stdin.close();assert self.process.wait()==0

def run(identity,seed,train_gain):
    inputs=identity_root(identity,ROOT,require_admission=True);verify_member_pins(ROOT,identity)
    configure();torch.set_grad_enabled(False)
    from nerf_triplane.network import NeRFNetwork
    from nerf_triplane.utils import Trainer,blend_with_mask_cuda
    block=json.loads((ROOT/'blocks'/identity/f'seed{seed}'/'fine_qualification.json').read_text());assert block['passed']
    epoch=ROOT/'training'/identity/f'seed{seed}'/f'gain_{train_gain:+d}'/'fine/ep0017/attempt01'
    receipt=json.loads((epoch/'epoch_receipt.json').read_text());checkpoint=Path(receipt['final_ema_path'])
    assert sha(checkpoint)==receipt['final_ema_file_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['ema'] and state['epoch']==17 and state['global_step']==102000
    assert tensor_state_sha(state['model'])==state['model_tensor_sha256']==receipt['final_ema_tensor_sha256']
    folder=ROOT/'renders'/identity/f'seed{seed}'/f'train_{train_gain:+d}';folder.mkdir(parents=True,exist_ok=False)
    # Use the material recording's full canonical original PCM, never the driver-gained WAV.
    manifest=json.loads((inputs/'material'/identity/'split_manifest.json').read_text())
    test=next(x for x in manifest['splits'] if x['name']=='test')
    assert test['source_start_frame']==50 and test['frames']==1500 and test['recording']=='test'
    raw=inputs/'material'/identity/'test_original.wav'
    pcm,sr=sf.read(raw,dtype='int16');assert sr==16000 and pcm.ndim==1
    selected=pcm[50*640:1550*640];assert len(selected)==1500*640
    audio=folder/'original_test_pcm.wav';sf.write(audio,selected,16000,subtype='PCM_16');assert np.array_equal(sf.read(audio,dtype='int16')[0],selected)
    for gain in [-6,0,6]:
        out=folder/f'driver_{gain:+d}';out.mkdir()
        data=dataset(identity,'test',gain,seed,'coarse')
        assert len(data.poses)==1500 and data.opt.portrait and not data.opt.finetune_lips and data.opt.patch_size==1
        model=NeRFNetwork(data.opt).cuda();model.load_state_dict(state['model'],strict=True);restore_model_counters(model,state['model_counters']);model.eval()
        owner=SimpleNamespace(model=model,opt=data.opt,device=torch.device('cuda'))
        writer=Video(out/'original_pcm.mkv',audio);rows=[];mouth=[]
        for i in range(1500):
            item=data.collate([i])
            with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):pred,_=Trainer.test_step(owner,item)
            head=pred[0].detach().float().cpu().numpy();assert np.isfinite(head).all()
            # Exact original portrait floating blend and uint8 truncation, without H264 quantization.
            blended=blend_with_mask_cuda(pred[0],item['bg_gt_images'].squeeze(0),item['bg_face_mask'].squeeze(0))
            rgb=(blended*255).astype(np.uint8);writer.write(rgb)
            truth=item['bg_gt_images'][0].detach().float().cpu().numpy()
            landmarks=np.loadtxt(inputs/'prepared'/identity/'test/ori_imgs'/f'{i}.lms')[48:68]
            lo=np.maximum(np.floor(landmarks.min(0)).astype(int)-12,0);hi=np.minimum(np.ceil(landmarks.max(0)).astype(int)+13,512)
            error=blended.astype(np.float64)-truth.astype(np.float64);patch=error[lo[1]:hi[1],lo[0]:hi[0]]
            mouth.append(float(np.mean(patch**2)))
            rows.append(dict(index=i,source_audio_id=i+50,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                head_float_sha256=hashlib.sha256(head.tobytes()).hexdigest(),mouth_bbox_mse=mouth[-1]))
            if i in [0,750,1499]:
                np.save(out/f'raw_head_frame{i:04d}.npy',head)
                assert cv2.imwrite(str(out/f'portrait_frame{i:04d}.png'),rgb[:,:,::-1])
        writer.close()
        cap=cv2.VideoCapture(str(out/'original_pcm.mkv'));count=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==rows[count]['rgb_sha256'];count+=1
        cap.release();assert count==1500
        decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(out/'original_pcm.mkv'),'-map','0:a:0','-f','s16le','-acodec','pcm_s16le','pipe:1'])
        assert decoded==selected.tobytes()
        if train_gain==gain==0:
            for i in range(250):
                item=data.collate([i])
                with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):pred,_=Trainer.test_step(owner,item)
                head=pred[0].detach().float().cpu().numpy();assert hashlib.sha256(head.tobytes()).hexdigest()==rows[i]['head_float_sha256']
                blended=blend_with_mask_cuda(pred[0],item['bg_gt_images'].squeeze(0),item['bg_face_mask'].squeeze(0))
                assert hashlib.sha256((blended*255).astype(np.uint8).tobytes()).hexdigest()==rows[i]['rgb_sha256']
        result=dict(identity=identity,seed=seed,train_gain_db=train_gain,driver_gain_db=gain,passed=True,frames=rows,
            precision='original -O FP16',portrait='pinned original test_step, blend_with_mask_cuda and uint8 truncation',
            final_ema_file_sha256=sha(checkpoint),original_pcm_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
            movie_sha256=sha(out/'original_pcm.mkv'),rgb_and_pcm_readback_exact=True,mean_mouth_bbox_mse=float(np.mean(mouth)),
            repeated_first_clip_exact=(train_gain==gain==0),source_sha256=sha(Path(__file__)))
        (out/'render_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='frames'}),flush=True)
        del model,data;torch.cuda.empty_cache()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--train-gain',type=int,required=True)
    a=p.parse_args();run(a.identity,a.seed,a.train_gain)
