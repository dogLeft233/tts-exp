"""Isolated Patrick fine17 descriptive score; coarse result and original renderer stay immutable."""
import argparse,hashlib,json,sys,subprocess,shutil
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
RUN=BASE/'trials/patrick_fp16_fine_v2'
OUT=RUN/'fine_score_27_v1'
sys.path.insert(0,str(BASE))
import numpy as np,torch
from runtime_common import configure,dataset,options,tensor_state_sha
from cohort_binding import identity_root,verify_member_pins
from execution_state import restore_model_counters,tree_sha
from flow_stream import sha

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),path
    path.write_text(json.dumps(value,indent=2))

def finite(value):
    if isinstance(value,torch.Tensor):
        assert torch.isfinite(value).all(),'nonfinite checkpoint tensor'
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:finite(v)
    elif isinstance(value,float):assert np.isfinite(value)

def named_ema_state(model,state):
    manual={k:v.detach().cpu().clone() for k,v in state['model'].items()}
    names=[n for n,p in model.named_parameters()]
    assert len(names)==len(state['ema']['shadow_params'])
    for name,shadow in zip(names,state['ema']['shadow_params']):
        assert manual[name].shape==shadow.shape and manual[name].dtype==shadow.dtype
        manual[name]=shadow.detach().cpu().clone()
    return manual

def gate():
    contract=json.loads((OUT/'contract.json').read_text())
    for name,digest in contract['sources'].items():assert sha(OUT/'sources'/name)==digest,name
    for name,digest in contract['dependencies'].items():assert sha(BASE/name)==digest,name
    ready=json.loads((OUT/'controller_preview_ready.json').read_text())
    assert ready['phase']['identity']=='PatrickLeahy' and ready['passed'] and ready['checkpoint_epoch']==17 and ready['global_step']==102000
    assert json.loads((RUN/'fine_ready_for_27_grid.json').read_text())==ready
    assert contract['identity']=='PatrickLeahy' and contract['seed']==42
    for gain in [-6,0,6]:
        epoch=RUN/'training/PatrickLeahy/seed42'/f'gain_{gain:+d}'/'fine/ep0017/attempt01'
        receipt=json.loads((epoch/'epoch_receipt.json').read_text())
        assert receipt['passed'] and receipt['epoch']==17 and receipt['global_step']==102000
        assert sha(Path(receipt['checkpoint_path']))==receipt['checkpoint_sha256']
        assert receipt['checkpoint_sha256']==contract['checkpoints'][str(gain)]['sha256']
    return contract

def make_ema(gain):
    """Cross-check named manual overlay against pinned torch_ema copy_to and source buffers."""
    from nerf_triplane.network import NeRFNetwork
    from torch_ema import ExponentialMovingAverage
    contract=gate();identity=contract['identity']
    epoch=RUN/'training'/identity/'seed42'/f'gain_{gain:+d}'/'fine/ep0017/attempt01'
    receipt=json.loads((epoch/'epoch_receipt.json').read_text());cp=Path(receipt['checkpoint_path'])
    state=torch.load(cp,map_location='cpu',weights_only=False);finite(state)
    assert state['epoch']==17 and state['global_step']==102000
    assert state['ema']['num_updates']==102 and state['ema']['decay']==.95
    model=NeRFNetwork(options(identity,'test',0,42)).cuda()
    model.load_state_dict(state['model'],strict=True)
    restore_model_counters(model,receipt['model_counters'])
    for name in ['mean_count','mean_density','mean_density_torso']:
        assert tree_sha(state[name])==tree_sha(receipt['model_counters'][name])
    ema=ExponentialMovingAverage(model.parameters(),decay=.95);ema.load_state_dict(state['ema'])
    before=tensor_state_sha(model.state_dict());ema.store();ema.copy_to()
    copied={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    manual=named_ema_state(model,state)
    assert tensor_state_sha(manual)==tensor_state_sha(copied)
    ema.restore();assert tensor_state_sha(model.state_dict())==before
    finite(copied)
    metadata=dict(passed=True,gain_db=gain,checkpoint_sha256=sha(cp),epoch=17,global_step=102000,
        successful_updates=receipt['cumulative_successful_updates'],ema_updates=102,ema_decay=.95,
        model_tensor_sha256=tensor_state_sha(copied),full_state_finite=True,
        independent_named_overlay_equals_torch_ema=True,raw_state_restored_exact=True,
        buffers_from_full_checkpoint=True,counters_from_post_validation_receipt=receipt['model_counters'],
        optimizer_updates=0,training_phase='fine',formal_fine17_model=False,parent_control_failed_retained=True,fine17_descriptive_model=True)
    model.load_state_dict(copied,strict=True);model.eval()
    return model,metadata

def frame(model,data,index):
    from nerf_triplane.utils import Trainer,blend_with_mask_cuda
    item=data.collate([index]);owner=SimpleNamespace(model=model,opt=data.opt,device=torch.device('cuda'))
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):pred,_=Trainer.test_step(owner,item)
    head=pred[0].detach().float().cpu().numpy();assert np.isfinite(head).all()
    blended=blend_with_mask_cuda(pred[0],item['bg_gt_images'].squeeze(0),item['bg_face_mask'].squeeze(0))
    assert np.isfinite(blended).all()
    return head,blended,(blended*255).astype(np.uint8),item

def qualify():
    import cv2,soundfile as sf
    from render_model import Video
    contract=gate();identity=contract['identity'];inputs=identity_root(identity,BASE,require_admission=True);verify_member_pins(BASE,identity)
    out=OUT/'qualification';out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for gain in [0,-6,6]:
        model,metadata=make_ema(gain);data=dataset(identity,'test',0,42,'coarse')
        assert len(data.poses)==1500
        frames=[]
        for index in [0,750,1499]:
            a=frame(model,data,index);b=frame(model,data,index)
            assert np.array_equal(a[0],b[0]) and np.array_equal(a[2],b[2])
            frames.append(a[2]);metadata.setdefault('frames',[]).append(dict(index=index,head_sha256=hashlib.sha256(a[0].tobytes()).hexdigest(),rgb_sha256=hashlib.sha256(a[2].tobytes()).hexdigest(),repeat_exact=True))
        pcm,sr=sf.read(inputs/'material'/identity/'test_original.wav',dtype='int16');assert sr==16000
        audio=out/f'qualification_{gain:+d}.wav';selected=pcm[50*640:53*640];sf.write(audio,selected,16000,subtype='PCM_16')
        movie=out/f'qualification_{gain:+d}.mkv';writer=Video(movie,audio,frames=3)
        for rgb in frames:writer.write(rgb)
        writer.close();cap=cv2.VideoCapture(str(movie))
        for rgb in frames:
            ok,bgr=cap.read();assert ok and np.array_equal(bgr[:,:,::-1],rgb)
        assert not cap.read()[0];cap.release()
        decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(movie),'-map','0:a:0','-f','s16le','pipe:1']);assert decoded==selected.tobytes()
        metadata.update(lossless_rgb_pcm_readback=True,qualification_movie_sha256=sha(movie))
        rows.append(metadata);del model,data;torch.cuda.empty_cache()
    save(out/'qualification.json',dict(passed=True,rows=rows,new_training_updates=0,real_model_fixed_frame_repeats=9,formal_test_scores=0))

def render(gain,driver):
    import cv2,soundfile as sf
    from render_model import Video
    contract=gate();identity=contract['identity'];inputs=identity_root(identity,BASE,require_admission=True);verify_member_pins(BASE,identity)
    assert json.loads((OUT/'qualification/qualification.json').read_text())['passed']
    model,ema=make_ema(gain);data=dataset(identity,'test',driver,42,'coarse')
    assert len(data.poses)==1500 and data.opt.portrait and not data.opt.finetune_lips and data.opt.patch_size==1
    assert shutil.disk_usage(OUT).free>1250*1024**2,'require raw RGB video upper-bound plus margin before render'
    out=OUT/'renders'/f'train_{gain:+d}'/f'driver_{driver:+d}';out.mkdir(parents=True,exist_ok=False)
    pcm,sr=sf.read(inputs/'material'/identity/'test_original.wav',dtype='int16');assert sr==16000
    selected=pcm[50*640:1550*640];assert len(selected)==960000
    audio=out/'original_test_pcm.wav';sf.write(audio,selected,16000,subtype='PCM_16')
    movie=out/'original_pcm.mkv';writer=Video(movie,audio);rows=[]
    for i in range(1500):
        head,blended,rgb,item=frame(model,data,i);writer.write(rgb)
        landmarks=np.loadtxt(inputs/'prepared'/identity/'test/ori_imgs'/f'{i}.lms')[48:68]
        lo=np.maximum(np.floor(landmarks.min(0)).astype(int)-12,0);hi=np.minimum(np.ceil(landmarks.max(0)).astype(int)+13,512)
        truth=item['bg_gt_images'][0].detach().float().cpu().numpy();error=blended.astype(np.float64)-truth.astype(np.float64)
        mse=float(np.mean(error[lo[1]:hi[1],lo[0]:hi[0]]**2))
        rows.append(dict(index=i,source_audio_id=i+50,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),head_float_sha256=hashlib.sha256(head.tobytes()).hexdigest(),mouth_bbox_mse=mse))
        if i in [0,750,1499]:
            np.save(out/f'raw_head_frame{i:04d}.npy',head);assert cv2.imwrite(str(out/f'portrait_frame{i:04d}.png'),rgb[:,:,::-1])
        if i%100==0:print(json.dumps(dict(stage='render',train=gain,driver=driver,frame=i)),flush=True)
    writer.close();cap=cv2.VideoCapture(str(movie));count=0
    while True:
        ok,bgr=cap.read()
        if not ok:break
        assert hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==rows[count]['rgb_sha256'];count+=1
    assert count==1500;cap.release()
    decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(movie),'-map','0:a:0','-f','s16le','pipe:1']);assert decoded==selected.tobytes()
    if gain==driver==0:
        for i in range(250):
            head,_,rgb,_=frame(model,data,i)
            assert hashlib.sha256(head.tobytes()).hexdigest()==rows[i]['head_float_sha256']
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==rows[i]['rgb_sha256']
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(movie),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac',str(out/'preview.mp4')],check=True)
    save(out/'render_receipt.json',dict(passed=True,identity=identity,seed=42,train_gain_db=gain,driver_gain_db=driver,
        epoch=10,phase='coarse',formal_fine17_model=False,training_control_status=contract['training_control_status'],
        frames=rows,ema=ema,movie_sha256=sha(movie),original_pcm_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        rgb_and_pcm_readback_exact=True,repeated_first_clip_exact=(gain==driver==0),precision='original FP16 test_step and portrait blend',source_sha256=sha(Path(__file__))))

def score(gain,driver):
    import cv2,soundfile as sf,python_speech_features
    from prepare_score_reference import SYNC,S,calc_pdist,crop_images,visual_forward,audio_forward,summarize
    from validate_scores import distance,endpoint
    contract=gate();identity=contract['identity'];inputs=identity_root(identity,BASE,require_admission=True);verify_member_pins(BASE,identity)
    refdir=inputs/'score_reference'/identity;ref=json.loads((refdir/'score_reference_receipt.json').read_text());assert ref['passed']
    track=np.load(refdir/'shared_gt_track.npy');assert track.shape==(1500,3)
    clips=json.loads((inputs/'material'/identity/'split_manifest.json').read_text())['eval_clips'];assert len(clips)==6
    weights=SYNC/'data/syncnet_v2.model';assert sha(weights)==ref['syncnet_weights_sha256']
    model=S(num_layers_in_fc_layers=1024).cuda().eval();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    render=OUT/'renders'/f'train_{gain:+d}'/f'driver_{driver:+d}';metadata=json.loads((render/'render_receipt.json').read_text());assert metadata['passed'] and metadata['rgb_and_pcm_readback_exact']
    movie=render/'original_pcm.mkv';assert sha(movie)==metadata['movie_sha256']
    pcm,sr=sf.read(inputs/'material'/identity/'test_original.wav',dtype='int16');assert sr==16000
    assert subprocess.check_output(['ffmpeg','-v','error','-i',str(movie),'-map','0:a:0','-f','s16le','pipe:1'])==pcm[50*640:1550*640].tobytes()
    cap=cv2.VideoCapture(str(movie));assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==25
    out=OUT/'scores'/f'train_{gain:+d}_driver_{driver:+d}';out.mkdir(parents=True,exist_ok=False)
    cells={s:[] for s in [-6,0,6]};maxerr=0.;maxend=0.;repeat=None
    for index,clip in enumerate(clips):
        assert clip['source_start_frame']==50+index*250
        images=[]
        for j in range(250):
            ok,bgr=cap.read();assert ok
            assert hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==metadata['frames'][index*250+j]['rgb_sha256'];images.append(bgr)
        cropped=crop_images(np.stack(images),track[index*250:(index+1)*250]);vf=visual_forward(model,cropped)
        folder=out/clip['id'];folder.mkdir();np.save(folder/'visual_features.npy',vf.numpy())
        for s in [-6,0,6]:
            selected=pcm[clip['source_start_frame']*640:(clip['source_start_frame']+250)*640]
            values=selected if s==0 else selected.astype(np.float64)*10**(s/20)
            mfcc=python_speech_features.mfcc(values,16000).T
            assert np.array_equal(mfcc.astype(np.float32),np.load(refdir/clip['id']/f'mfcc_{s:+d}dB.npy').astype(np.float32))
            af=audio_forward(model,mfcc);assert np.array_equal(af.numpy(),np.load(refdir/clip['id']/f'audio_features_{s:+d}dB.npy'))
            matrix=torch.stack(calc_pdist(vf,af,vshift=15));path=folder/f'distances_score_{s:+d}.npy';np.save(path,matrix.numpy())
            rebuilt=distance(vf.numpy(),af.numpy());err=float(np.max(np.abs(matrix.numpy()-rebuilt)));assert err<3e-5;maxerr=max(maxerr,err)
            row=dict(clip_id=clip['id'],distances_sha256=sha(path))
            for support,ids in [('guard20',np.arange(20,225)),('full',np.arange(245))]:
                row[support]=summarize(matrix,ids);independent=endpoint(rebuilt,ids)
                assert independent['av_offset']==row[support]['av_offset']
                for k in ['sync_c','sync_d','distance_d0','background_median']:
                    err=abs(independent[k]-row[support][k]);assert err<3e-5;maxend=max(maxend,err)
            cells[s].append(row)
            if gain==driver==s==0 and index==0:
                vf2=visual_forward(model,cropped);assert torch.equal(vf2,vf) and torch.equal(torch.stack(calc_pdist(vf2,af,vshift=15)),matrix);repeat=True
    assert not cap.read()[0];cap.release()
    result=[]
    for s,rows in cells.items():
        cell=dict(training_gain_db=gain,driver_gain_db=driver,evaluation_gain_db=s,clips=rows)
        for support in ['guard20','full']:
            cell['mean_'+support]={k:float(np.mean([r[support][k] for r in rows])) for k in ['sync_c','sync_d','distance_d0','background_median']}
            cell['mean_'+support]['offsets']=[r[support]['av_offset'] for r in rows]
        result.append(cell)
    save(out/'scores.json',dict(passed=True,phase='fine',epoch=17,seed=42,cells=result,training_control_status=contract['training_control_status'],
        independent_matrix_qa=True,matrix_max_abs_error=maxerr,endpoint_max_abs_error=maxend,checked_matrices=18,repeat=repeat,
        source_owned_ROI=True,precision='FP32 TF32 off',audio_0='original PCM16',audio_plusminus='float64 original PCM units'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qualify','render','score']);p.add_argument('--gain',type=int,choices=[-6,0,6]);p.add_argument('--driver',type=int,choices=[-6,0,6]);a=p.parse_args()
    configure();torch.set_grad_enabled(False)
    if a.action=='qualify':qualify()
    elif a.action=='render':render(a.gain,a.driver)
    else:score(a.gain,a.driver)
