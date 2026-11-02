"""Frozen native cloud preference experiment; official networks, explicit score adapters.

No synthesized audio, generated video, time warping, or best-offset correction.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/wjj/tts-audio/tts-exp')
DEPLOY = Path('/home/wjj/av-sync-eval-20261028')
RUN = ROOT/'runs/cloud_mtd_synchformer_20261028_attempt01'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def center_start(frames, seconds):
    required = round(seconds*25)
    if frames < required:
        raise ValueError('Insufficient native frames; never pad or stretch')
    return (frames-required)//2


def shift_pcm(x, samples):
    import numpy as np
    y = np.zeros_like(x)
    if samples > 0:
        y[samples:] = x[:-samples]
    elif samples < 0:
        y[:samples] = x[-samples:]
    else:
        y[:] = x
    return y


def summarize_pairs(pairs, metric):
    import numpy as np
    ds = np.array([p['delta'] for p in pairs], dtype=float)
    sc = np.array([p['delta_sync_c'] for p in pairs], dtype=float)
    if len(ds) == 0:
        return {'metric': metric, 'pairs': 0}
    # Strict sign, with exact ties reported separately rather than counted as conflict.
    conflict = ds*sc < 0
    grouped = {}
    for p in pairs:
        grouped.setdefault(p['source'], []).append(p['delta'])
    means = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(20261028)
    boots = means[rng.integers(0, len(means), (10000, len(means)))].mean(1)
    return dict(metric=metric, pairs=len(pairs), sources=len(means),
                natural_mean=float(np.mean([p['N'] for p in pairs])),
                tts_mean=float(np.mean([p['T'] for p in pairs])), mean_delta=float(ds.mean()),
                source_mean_delta=float(means.mean()), source_bootstrap_ci95=np.percentile(boots, [2.5,97.5]).tolist(),
                tts_wins=int((ds>0).sum()), natural_wins=int((ds<0).sum()), ties=int((ds==0).sum()),
                tts_win_fraction=float((ds>0).mean()), conflict_count=int(conflict.sum()),
                conflict_fraction=float(conflict.mean()),
                sync_tts_metric_natural=int(((sc>0)&(ds<0)).sum()),
                sync_natural_metric_tts=int(((sc<0)&(ds>0)).sum()),
                sync_ties=int((sc==0).sum()))


def freeze():
    import cv2
    import soundfile as sf
    rows = read(ROOT/'runs/avs_u_zh_pilot_20261027_attempt01/inputs.json')
    paired = read(ROOT/'runs/cloud_av_preference20_20261027_attempt05/paired_scores.json')
    assert len(rows)==80 and len(paired)==60
    for r in rows:
        cap=cv2.VideoCapture(r['video_path'])
        fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        info=sf.info(r['audio_path'])
        assert fps==25 and info.samplerate==16000 and info.channels==1
        r['native_frames']=min(n,info.frames//640)
        r['center_start_2s']=center_start(r['native_frames'],2)
        r['eligible_5s']=r['native_frames']>=125
        r['center_start_5s']=center_start(r['native_frames'],5) if r['eligible_5s'] else None
        assert Path(r['video_path']).is_file() and Path(r['audio_path']).is_file()
    protocol=dict(status='FROZEN_BEFORE_NEW_SCORES', cohort='existing Mandarin 20 sources, 3 cloud TTS takes each; 80 native media',
                  input_manifest='avs_u_zh_pilot_20261027_attempt01/inputs.json',
                  sync_c='cloud_av_preference20_20261027_attempt05/paired_scores.json, reused exact existing per-take Sync-C',
                  primary_support='single center native 2 seconds per clip; frame-grid crop; no padding, stretching, normalization, or best-lag correction',
                  mtd=dict(checkpoint='official pure_MTDVocaLiST.pth, strict README student_thin_200_all',
                           primary='mean aligned 5-frame classifier raw logit, higher better',
                           secondary='mean sigmoid of those logits, higher better',
                           preprocessing='official test_stu RGB0-255 JPEG Resize96 bottom48, original 16k mel hop200; S3FD on256 frames, scale boxes back',
                           aggregation='all 45 starts in 50 frames, author lastframe=min_length-5 convention; project continuous score adapter, not official retrieval accuracy'),
                  synchformer=dict(checkpoint='official LRS3 23-12-23T18-33-57 strict',
                                   primary='softmax probability of zero-offset class, higher better; project scalar adapter, not author benchmark accuracy',
                                   secondary='probability-weighted negative absolute offset (seconds), higher better',
                                   short_input='official transforms configured crop_len_sec2,n_segments5; trained weights unchanged; length-adapted, not exact 5-second benchmark recipe',
                                   official_5s='separate native >=5s paired subset with original crop_len_sec5,n_segments14',
                                   environment='PyAV12.3/Torch2.5.1 compatibility runtime; float PCM adapter'),
                  controls='20 natural fixed videos; actual full PCM shifted +/-6400 samples, zero fill; same crop and support; no offset search',
                  qualification='>=32/40 primary score decreases; report CONTROL_FAILED if not met, never tune score from preference outcomes',
                  stats='60 paired comparisons, 20 independent source clusters; 10000 source bootstrap seed20261028; exact ties separate; strict12 reference subset also',
                  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    RUN.mkdir(parents=True,exist_ok=True)
    for name,obj in [('protocol.json',protocol),('inputs.json',rows),('sync_pairs.json',paired)]:
        p=RUN/name
        if p.exists():
            assert read(p)==obj, 'Frozen artifact differs: '+name
        else:write(p,obj)
    print(json.dumps(dict(prepared=len(rows),eligible_5s=sum(r['eligible_5s'] for r in rows))),flush=True)


def mtd_run(full=False):
    import cv2
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import torchvision
    sys.path.insert(0,str(DEPLOY/'MTDVocaLiST'))
    from models.student_thin_200_all import SyncTransformer
    from hparams import hparams
    sys.path.append(str(ROOT/'third_party/Wav2Lip'))
    import face_detection
    rows=read(RUN/'inputs.json')
    prefix='mtd_full' if full else 'mtd'
    detector=None
    # Cache face tensors first so S3FD and the scoring network never share memory.
    for row in rows:
        dest=RUN/(prefix+'_crops')/(row['id']+'.npz')
        if dest.exists():continue
        if detector is None:
            detector=face_detection.FaceAlignment(face_detection.LandmarksType._2D,flip_input=False,device='cuda')
        start=0 if full else row['center_start_2s']
        count=row['native_frames'] if full else 50
        cap=cv2.VideoCapture(row['video_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        frames=[]
        for _ in range(count):
            ok,f=cap.read()
            if not ok:raise RuntimeError(row['id']+' short decode')
            frames.append(f)
        cap.release();patches=[];boxes=[]
        resize=torchvision.transforms.Resize((96,96))
        old=np.load(RUN/'mtd_crops'/(row['id']+'.npz')) if full else None
        for fi,f in enumerate(frames):
            if old is not None and row['center_start_2s']<=fi<row['center_start_2s']+50:
                oi=fi-row['center_start_2s']
                patches.append(torch.from_numpy(old['frames'][oi]));boxes.append(old['boxes'][oi].tolist())
                continue
            small=cv2.resize(f,(256,256))
            box=detector.get_detections_for_batch(np.asarray([small]))[0]
            if box is None:raise RuntimeError(row['id']+' face absent')
            x1,y1,x2,y2=box
            x1,x2=int(x1*f.shape[1]/256),int(x2*f.shape[1]/256)
            y1,y2=int(y1*f.shape[0]/256),int(y2*f.shape[0]/256)
            ok,jpg=cv2.imencode('.jpg',f[y1:y2,x1:x2])
            assert ok
            tensor=torchvision.io.decode_jpeg(torch.from_numpy(jpg)).contiguous()
            patches.append(resize(tensor)[:,48:]);boxes.append([x1,y1,x2,y2])
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,frames=torch.stack(patches).numpy(),boxes=np.array(boxes))
        print(prefix+' CROP '+row['id'],flush=True)
    del detector;torch.cuda.empty_cache()
    model=SyncTransformer(d_model=200)
    model.load_state_dict(torch.load(DEPLOY/'weights/mtdvocalist/pure_MTDVocaLiST.pth',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    model=model.eval().cuda()
    write(RUN/(prefix+'_load.json'),dict(strict=True,torch=torch.__version__,device='cuda',parameters=sum(p.numel() for p in model.parameters())))
    mel_scale=torchaudio.transforms.MelScale(n_mels=80,sample_rate=16000,f_min=hparams.fmin,f_max=hparams.fmax,n_stft=hparams.n_stft,norm='slaney',mel_scale='slaney')
    scores=read(RUN/(prefix+'_scores.json')) if (RUN/(prefix+'_scores.json')).exists() else {}
    for r in rows:
        if r['id'] in scores:continue
        frames=torch.from_numpy(np.load(RUN/(prefix+'_crops')/(r['id']+'.npz'))['frames']).float()
        wave,sr=sf.read(r['audio_path'],dtype='float32');assert sr==16000
        n_starts=len(frames)-5
        start=0 if full else r['center_start_2s']
        vv=torch.stack([frames[i:i+5].reshape(15,48,96) for i in range(n_starts)])
        outputs={}
        for shift in ([0,-6400,6400] if r['arm']=='N' else [0]):
            pcm=shift_pcm(wave,shift)
            spec=torch.stft(torch.from_numpy(pcm),n_fft=800,hop_length=200,win_length=800,window=torch.hann_window(800),return_complex=True)
            mel=mel_scale(spec.abs())
            db=20*torch.log10(torch.clamp(mel,min=1e-5))-20
            mel=torch.clip(8*((db+100)/100)-4,-4,4)
            starts=[int(80*((start+i)/25)) for i in range(n_starts)]
            aa=torch.stack([mel[:,s:s+16][None] for s in starts])
            logits=[]
            with torch.inference_mode():
                for lo in range(0,len(vv),16):
                    z,_=model(vv[lo:lo+16].cuda(),aa[lo:lo+16].cuda());logits.append(z.cpu())
            z=torch.cat(logits);assert torch.isfinite(z).all()
            outputs[str(shift)]=dict(logit=float(z.mean()),probability=float(z.sigmoid().mean()),window_logits=z.tolist())
        scores[r['id']]=dict(source=r['sample_id'],arm=r['arm'],start_frame=start,frames=len(frames),outputs=outputs)
        write(RUN/(prefix+'_scores.json'),scores)
        write(RUN/(prefix+'_status.json'),dict(status='RUNNING',scored=len(scores),total=80,last=r['id']))
        print(prefix+' '+str(len(scores))+'/80 '+r['id'],flush=True)
    write(RUN/(prefix+'_status.json'),dict(status='COMPLETE',scored=80))


def synchformer_run():
    import subprocess
    import numpy as np
    import torch
    import soundfile as sf
    from omegaconf import OmegaConf
    sys.path.insert(0,str(DEPLOY/'Synchformer-source'))
    # Upstream MotionFormer registers a relative source path at import time.
    os.chdir(DEPLOY/'Synchformer-source')
    import example
    cfg=example.patch_config(OmegaConf.load(DEPLOY/'weights/synchformer/cfg-23-12-23T18-33-57.yaml'))
    model,_=example.get_model(cfg,torch.device('cuda'))
    ckpt=torch.load(DEPLOY/'weights/synchformer/23-12-23T18-33-57.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model'],strict=True);del ckpt;model.eval()
    write(RUN/'synchformer_load.json',dict(strict=True,torch=torch.__version__,device='cuda',parameters=sum(p.numel() for p in model.parameters())))
    transforms={}
    for seconds,segments in [(5,14)]:
        c=copy.deepcopy(cfg)
        for t in c.transform_sequence_test:
            if t.target.endswith('TemporalCropAndOffset'):
                t.params.crop_len_sec=seconds;t.params.n_segments=segments
            if t.target.endswith('GenerateMultipleSegments'):t.params.n_segments=segments
        transforms[seconds]=example.get_transforms(c,['test'])['test']
    scores=read(RUN/'synchformer_scores.json') if (RUN/'synchformer_scores.json').exists() else {}
    grid=torch.linspace(-2,2,21)
    eligible=[r for r in read(RUN/'inputs.json') if r['eligible_5s']]
    for row in eligible:
        if row['id'] in scores:continue
        media=RUN/'synchformer_media'/(row['id']+'.mov');media.parent.mkdir(exist_ok=True)
        if not media.exists():
            tmp=media.with_suffix('.tmp.mov')
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',row['video_path'],'-i',row['audio_path'],'-map','0:v:0','-map','1:a:0',
                            '-vf','fps=25,scale=256:256','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','pcm_f32le','-ar','16000','-ac','1',str(tmp)],check=True)
            tmp.replace(media)
        rgb,audio,meta=example.get_video_and_audio(str(media),get_meta=True)
        ref,sr=sf.read(row['audio_path'],dtype='float32')
        assert len(audio)==len(ref) and np.array_equal(audio.numpy(),ref), 'PCM adapter changed waveform'
        result={}
        for seconds in [5]:
            outs={}
            for shift in ([0,-6400,6400] if row['arm']=='N' else [0]):
                shifted=torch.from_numpy(shift_pcm(audio.numpy(),shift))
                item=dict(video=rgb,audio=shifted,meta=copy.deepcopy(meta),path=str(media),split='test',
                          targets=dict(v_start_i_sec=row['center_start_'+str(seconds)+'s']/25,offset_sec=0.))
                item=transforms[seconds](item)
                batch=torch.utils.data.default_collate([item]);aud,vid,_=example.prepare_inputs(batch,torch.device('cuda'))
                with torch.inference_mode(),torch.autocast('cuda',enabled=cfg.training.use_half_precision):
                    _,z=model(vid,aud)
                z=z[0].float().cpu();assert z.shape==(21,) and torch.isfinite(z).all()
                p=z.softmax(0)
                outs[str(shift)]=dict(p_zero=float(p[10]),negative_expected_abs_offset=float(-(p*grid.abs()).sum()),
                                      predicted_offset=float(grid[z.argmax()]),logits=z.tolist(),probabilities=p.tolist())
            result[str(seconds)]=outs
        scores[row['id']]=dict(source=row['sample_id'],arm=row['arm'],outputs=result,pcm_exact=True)
        write(RUN/'synchformer_scores.json',scores)
        write(RUN/'synchformer_status.json',dict(status='RUNNING',scored=len(scores),total=len(eligible),last=row['id']))
        print('SYNCH '+str(len(scores))+'/'+str(len(eligible))+' '+row['id'],flush=True)
    write(RUN/'synchformer_status.json',dict(status='COMPLETE_ELIGIBLE_NATIVE5S_ONLY',scored=len(eligible),excluded_short=80-len(eligible)))


def analyze():
    rows=read(RUN/'inputs.json'); source_pairs=read(RUN/'sync_pairs.json')
    views={};paired_all={};controls={}
    for model,seconds,field,label in [('mtd_full',0,'logit','MTD_logit_full'),('mtd_full',0,'probability','MTD_probability_full'),
                                      ('mtd',2,'logit','MTD_logit_2s'),('mtd',2,'probability','MTD_probability_2s'),
                                      ('synchformer',2,'p_zero','Synchformer_P0_2s'),('synchformer',2,'negative_expected_abs_offset','Synchformer_negEabs_2s'),
                                      ('synchformer',5,'p_zero','Synchformer_P0_official5s'),
                                      ('synchformer',5,'negative_expected_abs_offset','Synchformer_negEabs_official5s')]:
        scores=read(RUN/(model+'_scores.json'))
        nmap={r['source']:key for key,r in scores.items() if r['arm']=='N'}
        def output(key,shift='0'):
            d=scores[key]['outputs']
            if model=='synchformer':d=d.get(str(seconds),{})
            return d.get(shift,{}).get(field)
        pairs=[]
        for p in source_pairs:
            if p['source'] not in nmap or p['id'] not in scores:continue
            n=output(nmap[p['source']]);t=output(p['id'])
            if n is None or t is None:continue
            pairs.append(dict(id=p['id'],source=p['source'],strict_reference=p['strict_reference'],N=n,T=t,delta=t-n,delta_sync_c=p['delta']['SyncC']))
        views[label]=dict(all20=summarize_pairs(pairs,label),strict12=summarize_pairs([p for p in pairs if p['strict_reference']],label))
        paired_all[label]=pairs
        cc=[]
        for src,key in nmap.items():
            native=output(key)
            for shift in ['-6400','6400']:
                z=output(key,shift)
                if native is not None and z is not None:cc.append(dict(source=src,shift_samples=int(shift),native=native,shifted=z,lower=z<native))
        lower=sum(c['lower'] for c in cc)
        controls[label]=dict(lower=lower,total=len(cc),fraction=lower/len(cc) if cc else None,
                             status=('PASS' if lower>=.8*len(cc) else 'CONTROL_FAILED') if cc else 'UNMEASURABLE',rows=cc)
    common={p['id'] for p in paired_all['Synchformer_P0_official5s']}
    for label in ['MTD_logit_full','MTD_probability_full','MTD_logit_2s']:
        views[label]['common_synchformer5s']=summarize_pairs([p for p in paired_all[label] if p['id'] in common],label)
    out=dict(status='COMPLETE_EXPLORATORY',independent_unit='20 source utterances, repeated 3 TTS takes',views=views,controls=controls,
             note='Scores are explicit project scalar adapters of official classifiers. Synchformer2s UNMEASURABLE: fixed198-token positional embedding; use official5s eligible subset only. MTD full native support primary, original2s retained. Control failure is not evidence of no lip-sync advantage.')
    write(RUN/'summary.json',out);write(RUN/'paired_scores.json',paired_all)
    with (RUN/'paired_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['metric','id','source','strict_reference','N','T','delta','delta_sync_c']);w.writeheader()
        for label,ps in paired_all.items():
            for p in ps:w.writerow(dict(metric=label,**p))
    print(json.dumps({k:v['all20'] for k,v in views.items()},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','mtd','mtd_full','synchformer','analyze']);args=parser.parse_args()
    if args.stage in ['mtd','mtd_full','synchformer']:
        import torch
        torch.set_num_threads(4);torch.manual_seed(20261028)
        assert torch.cuda.is_available()
    dict(freeze=freeze,mtd=mtd_run,mtd_full=lambda:mtd_run(full=True),synchformer=synchformer_run,analyze=analyze)[args.stage]()
