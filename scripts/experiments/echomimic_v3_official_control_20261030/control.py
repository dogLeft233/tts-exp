"""Independent official-default diagnostic; never writes to the frozen parent."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

B = Path('/home/data2/wangjiajun/av-evaluation')
D = Path('/home/data2/wangjiajun/EchoMimicV3-Flash')
PARENT = B/'jobs/echomimic_v3_flash_chinese20_rms24_20261030_attempt01'
RUN = B/'jobs/echomimic_v3_official_control_20261030_attempt01'
OLD = B/'experiments/echomimic_v3_chinese20_20261030'
GPU = 'GPU-de7ff5ff-3510-739f-917c-cb98892de759'
ARMS = {
    'baseline': dict(lufs=False, cap=False, seed=42),
    'lufs_only': dict(lufs=True, cap=False, seed=42),
    'cap_only': dict(lufs=False, cap=True, seed=42),
    'seed_only': dict(lufs=False, cap=False, seed=43),
    'official': dict(lufs=True, cap=True, seed=43),
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while x := f.read(8*1024**2): h.update(x)
    return h.hexdigest()


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)


def read(p):
    return json.loads(Path(p).read_text())


def native_frames(samples):
    return 4*((samples//640-1)//4)+1


def select_rows(rows):
    # Predetermined length coverage within each text stratum; never uses scores.
    selected = []
    for strict in (True, False):
        sources = sorted([r for r in rows if r['arm']=='N' and r['strict_reference']==strict],
                         key=lambda r:(r['samples'], r['source']))
        for i in (0, (len(sources)-1)//2, len(sources)-1):
            sid = sources[i]['source']
            selected.extend([next(r for r in rows if r['source']==sid and r['arm']=='N'),
                             next(r for r in rows if r['source']==sid and r['arm']=='T' and r['draw']==1)])
    assert len(selected)==12 and len({r['source'] for r in selected})==6
    return selected


def common_support(rows):
    return {sid:min(81, *[native_frames(r['samples']) for r in rows if r['source']==sid])
            for sid in sorted({r['source'] for r in rows})}


def prepare():
    import numpy as np
    import soundfile as sf
    import pyloudnorm as pyln
    from PIL import Image
    assert not RUN.exists(), 'New independent run required'
    rows = select_rows(read(PARENT/'inputs.json')['rows'])
    protocol = read(PARENT/'protocol.json')
    RUN.mkdir(parents=True)
    pins = {str(p):sha(p) for p in [PARENT/'inputs.json', PARENT/'protocol.json',
            PARENT/'summary.json', PARENT/'evidence.tar.gz'] if p.exists()}
    for r in rows:
        assert sha(r['audio_path']) == r['audio_sha256']
        assert sha(r['canonical_audio_path']) == r['canonical_audio_sha256']
        old = PARENT/'generation'/r['id']/'receipt.json'
        rec = read(old)
        assert sha(rec['video_path'])==rec['video_sha256']
        pins[str(old)]=sha(old); pins[rec['video_path']]=rec['video_sha256']
        r['parent_video']=rec['video_path']; r['parent_video_sha256']=rec['video_sha256']
        pcm,sr=sf.read(r['audio_path'],dtype='int16'); assert sr==16000
        x=pcm.astype(np.float32)/32768
        loud=pyln.Meter(sr).integrated_loudness(x)
        y=pyln.normalize.loudness(x,loud,-23) if abs(loud)<=100 else x
        canonical,_=sf.read(r['canonical_audio_path'],dtype='float32')
        cy=pyln.normalize.loudness(canonical,pyln.Meter(sr).integrated_loudness(canonical),-23)
        (RUN/'conditioning').mkdir(exist_ok=True)
        np.save(RUN/'conditioning'/f"{r['id']}.npy",y)
        sf.write(RUN/'conditioning'/f"{r['id']}_lufs.wav",y,sr,subtype='PCM_16')
        r['volume_audit']=dict(original_rms_dbfs=float(20*np.log10(np.sqrt(np.mean(x.astype(float)**2)))),
            original_lufs=float(loud),target_lufs=-23.,actual_lufs=float(pyln.Meter(sr).integrated_loudness(y)),
            gain_db=float(-23-loud),peak_float=float(abs(y).max()),dtype=str(y.dtype),
            canonical_vs_rms_then_lufs_max_abs=float(abs(cy-y).max()),
            canonical_vs_rms_then_lufs_rms_error=float(np.sqrt(np.mean((cy-y)**2))),
            conditioning_sha256=sha(RUN/'conditioning'/f"{r['id']}.npy"))
        assert np.isfinite(y).all() and abs(y).max()<1, 'No clipping allowed'
    face=Path(protocol['generation']['face_path'])
    assert sha(face)==protocol['generation']['face_sha256']
    write(RUN/'plan.json',dict(status='FROZEN',parent=str(PARENT),rows=rows,arms=ARMS,
        common_frames=common_support(rows),face_path=str(face),face_sha256=sha(face),
        face_size=list(Image.open(face).size),sample_size_flag=[768,768],
        selection='Six sources: 3 same-reference and3 extended, natural length minimum/middle/maximum within stratum; C1 only. Chosen without score lookup.',
        estimand='Exploratory diagnostic; paired sources, not a replication of Chinese20 three-TTS primary analysis.',
        official='Pinned run_flash.sh settings: LUFS-23 on full input before native prefix, seed43, min81 frame cap, sample_size768 cap. Same512 image is NOT upscaled. Remaining numeric settings unchanged.',
        implementation_adaptations='Verified strict Flash weights, explicit BF16 T5 loader, mask-correct SDPA and model CPU offload retained across arms. Not a literal unmodified upstream CLI or paper benchmark.',
        scorer_audio='All generation arms scored against identical original RMS24 prefix, matching official feature-only LUFS behavior. Additional scorer_gain_only changes audio only on baseline video.',
        source_sha256=sha(__file__),parent_pins=pins))
    print('PREPARED',[(r['id'],r['volume_audit']['gain_db']) for r in rows],flush=True)


def generation():
    import torch
    import numpy as np
    import soundfile as sf
    import gc
    from PIL import Image
    plan=read(RUN/'plan.json')
    sys.path.insert(0,str(OLD)); sys.path.insert(0,str(D/'repo'));os.chdir(D/'repo')
    spec=importlib.util.spec_from_file_location('old_generator',OLD/'generate.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.RUN=RUN # Only private diagnostic receipts; parent module files untouched.
    torch.set_num_threads(4)
    assert torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(0).total_memory)
    pipe,encoder,extractor,cfg,up=old.load_pipeline()
    face=Image.open(plan['face_path']).convert('RGB');assert sha(plan['face_path'])==plan['face_sha256']
    size=up.get_sample_size(face,[768,768]);assert size==[512,512]
    tasks=[('replay',r,ARMS['baseline']) for r in plan['rows'][:2]]
    tasks += [(arm,r,a) for arm,a in ARMS.items() if arm!='baseline' for r in plan['rows']]
    completed=[]
    for arm,r,a in tasks:
        key=arm+'__'+r['id'];folder=RUN/'generation'/key;receipt=folder/'receipt.json'
        if receipt.exists():completed.append(read(receipt));continue
        folder.mkdir(parents=True,exist_ok=True)
        assert sha(r['audio_path'])==r['audio_sha256']
        frames=min(81,native_frames(r['samples'])) if a['cap'] else native_frames(r['samples'])
        pcm,sr=sf.read(r['audio_path'],dtype='int16');x=pcm.astype(np.float32)/32768
        if a['lufs']:
            x=up.loudness_norm(x,sr)
            saved=np.load(RUN/'conditioning'/f"{r['id']}.npy")
            assert np.array_equal(x,saved), 'Official LUFS must match preparation exactly'
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            emb=up.get_audio_embed(x[:frames*640],extractor,encoder,frames,sr=sr,fps=25,device='cpu')
            emb=emb.to(device='cuda',dtype=torch.bfloat16)
            indices=torch.clamp(torch.arange(frames).unsqueeze(1)+(torch.arange(5)-2).unsqueeze(0),0,emb.shape[0]-1)
            emb=emb[indices].unsqueeze(0)
            video,mask,clip=up.get_image_to_video_latent2(face,None,video_length=frames,sample_size=size)
            pipe.transformer.enable_teacache(up.get_teacache_coefficients(str(D/'models/Wan2.1-Fun-V1.1-1.3B-InP')),8,.1,num_skip_start_steps=5,offload=False)
            sample=pipe('A person is speaking.',negative_prompt='',num_frames=frames,audio_embeds=emb,
                audio_scale=1.,ip_mask=None,use_un_ip_mask=False,height=size[0],width=size[1],
                generator=torch.Generator(device='cuda').manual_seed(a['seed']),neg_scale=1.,neg_steps=0,
                use_dynamic_cfg=False,use_dynamic_acfg=False,guidance_scale=6.,audio_guidance_scale=3.,
                num_inference_steps=8,video=video,mask_video=mask,clip_image=clip,cfg_skip_ratio=0.,shift=5.).videos
            silent=folder/'silent.mp4';up.save_videos_grid(sample[:,:,:frames],str(silent),fps=25)
            del sample,emb,video,mask,clip
        target=folder/'video.mp4'
        subprocess.run([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-i',str(silent),'-i',r['audio_path'],
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','alac',str(target)],check=True)
        media=old.validate(target,r['audio_path'],frames,size)
        rec=dict(id=r['id'],arm=arm,settings=a,video_path=str(target),video_sha256=sha(target),
            seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),**media)
        if arm=='replay':
            def video_hash(path):
                return subprocess.check_output([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-i',str(path),'-map','0:v:0','-f','hash','-hash','sha256','-']).decode().strip()
            rec['decoded_replay_sha256']=video_hash(target)
            rec['decoded_parent_sha256']=video_hash(r['parent_video'])
            rec['replay_exact']=rec['decoded_replay_sha256']==rec['decoded_parent_sha256']
            assert rec['replay_exact'], 'Baseline rerender changed: stop before interpreting ablations'
        write(receipt,rec);completed.append(rec)
        gc.collect();torch.cuda.empty_cache()
        print('GENERATED',len(completed),len(tasks),key,round(rec['seconds'],1),flush=True)
    write(RUN/'generation_complete.json',dict(status='PASS',completed=completed))


def score():
    import numpy as np
    import soundfile as sf
    import torch
    plan=read(RUN/'plan.json');assert (RUN/'generation_complete.json').exists()
    source=B/'scorers/syncnet';python=B/'envs/eval-base/bin/python';ff=B/'tools/bin/ffmpeg'
    records=[]
    cells=[(arm,r) for arm in ARMS for r in plan['rows']]
    cells += [('scorer_gain_only',r) for r in plan['rows']]
    cells += [('replay',r) for r in plan['rows'][:2]]
    for arm,r in cells:
        key=arm+'__'+r['id'];root=RUN/'syncnet'/key
        if (root/'score.json').exists():records.append(read(root/'score.json'));continue
        root.mkdir(parents=True,exist_ok=True)
        frames=plan['common_frames'][r['source']]
        video=Path(r['parent_video']) if arm in ('baseline','scorer_gain_only') else RUN/'generation'/key/'video.mp4'
        pcm,sr=sf.read(r['audio_path'],dtype='int16')
        if arm=='scorer_gain_only':pcm=sf.read(RUN/'conditioning'/f"{r['id']}_lufs.wav",dtype='int16')[0]
        prefix=root/'audio.wav';sf.write(prefix,pcm[:frames*640],sr,subtype='PCM_16')
        visual=root/'visual.mp4'
        subprocess.run([str(ff),'-nostdin','-v','error','-i',str(video),'-map','0:v:0','-an','-frames:v',str(frames),'-c:v','copy',str(visual)],check=True)
        media=root/'input.mkv'
        subprocess.run([str(ff),'-nostdin','-v','error','-i',str(visual),'-i',str(prefix),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le',str(media)],check=True)
        decoded=subprocess.check_output([str(ff),'-nostdin','-v','error','-i',str(media),'-map','0:a:0','-f','s16le','-'])
        assert decoded==pcm[:frames*640].astype('<i2').tobytes()
        probe=read_json_command(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(media)])
        pts=[float(f['best_effort_timestamp_time']) for f in probe['frames']]
        assert len(pts)==frames and np.allclose(pts,np.arange(frames)/25,atol=1e-6,rtol=0)
        common=['--videofile',str(media),'--reference',key,'--data_dir',str(root/'work')]
        for script,extra in [('run_pipeline.py',['--min_track','25','--overwrite']),('run_syncnet.py',['--initial_model',str(source/'data/syncnet_v2.model')])]:
            with (root/(script+'.log')).open('x') as log:
                subprocess.run([str(python),str(B/'gpu_guard.py'),str(source/script),*common,*extra],cwd=source,
                    stdout=log,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,GPU_METRICS_PATH=str(root/(script+'_gpu.json'))))
        distances=pickle.loads((root/'work/pywork'/key/'activesd.pckl').read_bytes())
        assert len(distances)==1
        z=np.asarray(distances[0],dtype=np.float32);assert z.ndim==2 and z.shape[1]==31 and np.isfinite(z).all()
        np.save(root/'distances.npy',z)
        curve=torch.from_numpy(z).T.mean(1);d,k=torch.min(curve,0);c=torch.median(curve)-d
        # Independent FP64 reduction; small accumulation differences bounded.
        independent=np.median(z.astype(float).mean(0))-z.astype(float).mean(0).min()
        assert abs(float(c)-independent)<1e-5
        rec=dict(id=r['id'],source=r['source'],audio_arm=r['arm'],strict_reference=r['strict_reference'],
            condition=arm,C=float(c),D=float(d),offset=int(15-k),windows=len(z),frames=frames,
            scored_pcm_sha256=hashlib.sha256(decoded).hexdigest(),video_sha256=sha(video),
            independent_C=independent)
        write(root/'score.json',rec);records.append(rec);print('SCORED',key,float(c),flush=True)
    write(RUN/'scores.json',records)


def read_json_command(command):
    return json.loads(subprocess.check_output(command))


def analyze():
    import numpy as np
    plan=read(RUN/'plan.json');scores=read(RUN/'scores.json')
    lookup={(r['condition'],r['id']):r for r in scores}
    paired=[];conditions={}
    for arm in [*ARMS,'scorer_gain_only']:
        rows=[]
        for sid in sorted(plan['common_frames']):
            pair=[r for r in plan['rows'] if r['source']==sid]
            n=next(r for r in pair if r['arm']=='N');t=next(r for r in pair if r['arm']=='T')
            sn=lookup[arm,n['id']];st=lookup[arm,t['id']]
            bn=lookup['baseline',n['id']];bt=lookup['baseline',t['id']]
            q=dict(condition=arm,source=sid,strict_reference=n['strict_reference'],N=sn['C'],T=st['C'],
                T_minus_N=st['C']-sn['C'],N_change=sn['C']-bn['C'],T_change=st['C']-bt['C'],
                difference_in_differences=(st['C']-sn['C'])-(bt['C']-bn['C']))
            rows.append(q);paired.append(q)
        values=np.array([[r['N'],r['T'],r['T_minus_N'],r['N_change'],r['T_change'],r['difference_in_differences']] for r in rows])
        rng=np.random.default_rng(20261030);boot=values[rng.integers(0,len(rows),(10000,len(rows)))].mean(1)
        conditions[arm]=dict(zip(['N','T','T_minus_N','N_change','T_change','difference_in_differences'],values.mean(0).tolist()))
        conditions[arm]['descriptive_bootstrap_ci95']=np.quantile(boot,[.025,.975],axis=0).tolist()
    for r in scores:
        if r['condition'] not in ('baseline','scorer_gain_only','replay'):
            assert r['scored_pcm_sha256']==lookup['baseline',r['id']]['scored_pcm_sha256']
    replay=[]
    for r in scores:
        if r['condition']=='replay':
            b=lookup['baseline',r['id']];assert abs(r['C']-b['C'])<1e-5
            replay.append(dict(id=r['id'],C_difference=r['C']-b['C']))
    for p,h in plan['parent_pins'].items():assert sha(p)==h, 'Frozen parent changed: '+p
    write(RUN/'paired_comparison.json',dict(status='PASS',sources=6,conditions=conditions,pairs=paired,
        replay=replay,parent_pins_unchanged=True,interpretation='Exploratory six-source C1 diagnostic; no confirmatory p-values, no perceived-sync claim. Same common first min81/native pair frames in all cells.'))
    import csv
    with (RUN/'paired_comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    print(json.dumps(conditions,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generation','score','analyze']);a=parser.parse_args()
    globals()[a.stage]()
