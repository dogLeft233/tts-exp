"""Extend the immutable N/R experiment with official full-ICL TTS only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_codec_roundtrip_20261022.pipeline import sha,read,save,execute
RUN=ROOT/'runs/qwen_full_tts_extension_20261022_attempt01'
CONDITIONS=('N','R','T1','T2','T3')
METRICS=('C','B','Dmin','D0','offset')
POLICIES=('common','native_full','equal_count_spread')


def protocol():
    p=read(RUN/'protocol.json')
    for rel in ['input_manifest.json','audio_manifest.json','video_manifest.json','matrix_scores.json']:
        r=p['parent_assets'][rel]
        if sha(r['path'])!=r['sha256']:raise ValueError(f'parent manifest changed: {rel}')
    return p


def parent_path(p,rel):
    row=p['parent_assets'][rel]
    if sha(row['path'])!=row['sha256']:raise ValueError(f'parent asset changed: {rel}')
    return Path(row['path'])


def valid_rows(count):
    rows=np.arange(15,count-15,dtype=np.int64)
    if not len(rows):raise ValueError('no complete real all-lag support')
    return rows


def support_policies(counts):
    native={k:valid_rows(v) for k,v in counts.items()}
    k=min(map(len,native.values()))
    common=np.arange(15,15+k,dtype=np.int64)
    spread={role:r[np.rint(np.linspace(0,len(r)-1,k)).astype(int)] for role,r in native.items()}
    if any(len(np.unique(v))!=k for v in spread.values()):raise ValueError('duplicate spread queries')
    return {'common':{role:common.copy() for role in native},'native_full':native,'equal_count_spread':spread}


def metrics(matrix,rows):
    curve=np.asarray(matrix[rows],dtype=np.float64).mean(0)
    i=int(curve.argmin());b=float(np.median(curve));d=float(curve[i])
    return dict(C=b-d,B=b,Dmin=d,D0=float(curve[15]),offset=15-i,curve=curve.tolist(),
                query_count=len(rows),query_indices=rows.tolist(),best_lag_at_boundary=i in (0,30))


def paired_contrasts(values):
    # [speakers, N/R/T1/T2/T3]. Average generated repetitions within person.
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or x.shape[1]!=5 or not np.isfinite(x).all():raise ValueError('bad paired layout')
    t=x[:,2:].mean(1)
    return {'T-N':t-x[:,0],'T-R':t-x[:,1],'R-N':x[:,1]-x[:,0]}


def summary(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('invalid speaker values')
    rng=np.random.default_rng(20261022)
    draws=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(n_speakers=len(x),mean=float(x.mean()),CI95=np.quantile(draws,[.025,.975]).tolist(),
                CI97_5=np.quantile(draws,[.0125,.9875]).tolist(),values=x.tolist(),
                positive=int((x>0).sum()),negative=int((x<0).sum()))


def audio():
    import soundfile as sf
    p=protocol();rows=read(RUN/'tts/tts_manifest.json')['records'];out=[]
    if len(rows)!=30:raise ValueError('expected all 30 TTS outputs')
    seen=set()
    for row in rows:
        sid=row['sample_id'];idx=int(row['seed_index']);role=f'T{idx}';key=(sid,role)
        if idx not in (1,2,3) or key in seen:raise ValueError('bad seed index')
        seen.add(key)
        if sha(row['path'])!=row['sha256']:raise ValueError('TTS source changed')
        dest=RUN/'audio'/sid/f'{role}_16k_pcm16.wav';dest.parent.mkdir(parents=True,exist_ok=True)
        f32=dest.with_name(f'{role}_16k_float.wav')
        execute(['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',row['path'],'-ac','1','-ar','16000',
                 '-c:a','pcm_f32le',f32],RUN/'logs'/f'canonical_{sid}_{role}.log',timeout=60)
        x,sr=sf.read(f32,dtype='float32')
        if sr!=16000 or x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('invalid audio')
        rms=float(np.sqrt(np.mean(x.astype(float)**2)))
        if rms<=0:raise ValueError('silent output')
        gain=p['target_rms']/rms
        if float(abs(x).max()*gain)>=1:raise ValueError('normalization clipping')
        sf.write(dest,x*np.float32(gain),sr,subtype='PCM_16')
        y,_=sf.read(dest,dtype='int16')
        if len(x)!=len(y):raise ValueError('sample count changed')
        import hashlib
        out.append(dict(sample_id=sid,speaker_id=row['speaker_id'],role=role,seed_index=idx,seed=row['seed'],
                        path=str(dest),sha256=sha(dest),samples=len(y),sr=16000,
                        source_path=row['path'],source_sha256=row['sha256'],rms_before=rms,gain=gain,
                        rms_after=float(np.sqrt(np.mean((y.astype(float)/32768)**2))),
                        pcm_sha256=hashlib.sha256(y.astype('<i2').tobytes()).hexdigest()))
        print('AUDIO_READY',sid,role,'duration',len(y)/16000,flush=True)
    save(RUN/'audio_manifest.json',dict(records=out,protocol_sha256=sha(RUN/'protocol.json')))


def render():
    p=protocol();results=[];w2l=ROOT/'third_party/Wav2Lip';ckpt=w2l/'checkpoints/wav2lip_gan.pth'
    if sha(ckpt)!=p['downstream']['weights_sha256']:raise ValueError('Wav2Lip weight mismatch')
    if sha(p['downstream']['face'])!=p['downstream']['face_sha256']:raise ValueError('face changed')
    prior=read(Path(p['parent_run'])/'video_manifest.json')
    if sha(w2l/'inference.py')!=prior['inference_sha256']:raise ValueError('generation code differs from reused N/R')
    for row in read(RUN/'audio_manifest.json')['records']:
        sid,role=row['sample_id'],row['role']
        if sha(row['path'])!=row['sha256']:raise ValueError('canonical input changed')
        work=RUN/'render_work'/sid/role;work.mkdir(parents=True,exist_ok=False);(work/'temp').mkdir()
        dest=RUN/'video'/sid/f'{role}.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
        cmd=['/home/wjj/.venvs/wav2lip/bin/python',str(w2l/'inference.py'),'--checkpoint_path',str(ckpt),
             '--face',p['downstream']['face'],'--audio',row['path'],'--outfile',str(dest),
             '--face_det_batch_size','4','--wav2lip_batch_size','4','--nosmooth']
        rec=execute(cmd,RUN/'logs'/f'render_{sid}_{role}.log',cwd=work,timeout=180)
        if not dest.is_file() or not dest.stat().st_size:raise ValueError('missing video')
        results.append(dict(sample_id=sid,speaker_id=row['speaker_id'],role=role,path=str(dest),sha256=sha(dest),
                            audio_sha256=row['sha256'],receipt=rec))
        print('RENDERED',sid,role,flush=True)
    save(RUN/'video_manifest.json',dict(records=results,checkpoint_sha256=sha(ckpt),inference_sha256=sha(w2l/'inference.py')))


def extract():
    import torch
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    from scripts.experiments.repeated_take_crossover_20261015.run_matrix import _extract_fixed_frames,_visual_features,_probe_video
    p=protocol();aa={(r['sample_id'],r['role']):r for r in read(RUN/'audio_manifest.json')['records']}
    vv=read(RUN/'video_manifest.json')['records'];ckpt=ROOT/'third_party/syncnet_python/data/syncnet_v2.model'
    if sha(ckpt)!='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442':raise ValueError('SyncNet mismatch')
    torch.set_num_threads(2);torch.manual_seed(20261022);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    engine=SyncNetEngine(model_path=ckpt,batch_size=32,device='cuda');results=[]
    try:
        for row in vv:
            sid,role=row['sample_id'],row['role'];a=aa[sid,role]
            if sha(row['path'])!=row['sha256'] or sha(a['path'])!=a['sha256']:raise ValueError('source mismatch')
            ae,am=engine.extract_audio(a['path'])
            if am['pcm_sha256']!=a['pcm_sha256']:raise ValueError('PCM mismatch')
            probe=_probe_video(Path(row['path']))
            frames,fm=_extract_fixed_frames(Path(row['path']),RUN/'decoded_frames'/sid/role,p['downstream']['score_box_xyxy'])
            ve=_visual_features(engine,frames)
            dest=RUN/'features'/sid;dest.mkdir(parents=True,exist_ok=True)
            af=dest/f'audio_{role}.npy';vf=dest/f'visual_{role}.npy';np.save(af,ae);np.save(vf,ve)
            repeat={}
            if not results:
                ar,_=engine.extract_audio(a['path']);vr=_visual_features(engine,frames)
                repeat=dict(audio_max_abs=float(abs(ar-ae).max()),visual_max_abs=float(abs(vr-ve).max()))
                if max(repeat.values())>1e-5:raise ValueError('repeat failed')
            results.append(dict(sample_id=sid,speaker_id=row['speaker_id'],role=role,
                                audio_path=str(af),audio_sha256=sha(af),visual_path=str(vf),visual_sha256=sha(vf),
                                audio_meta=am,video_meta=probe,frames_meta=fm,repeat=repeat))
            print('FEATURES',sid,role,'audio',len(ae),'video',len(ve),flush=True)
    finally:engine.close()
    save(RUN/'feature_manifest.json',dict(records=results,SyncNet_weights_sha256=sha(ckpt)))


def analyze():
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    p=protocol();parent=Path(p['parent_run']);fm=read(RUN/'feature_manifest.json')['records']
    feats={(r['sample_id'],r['role']):r for r in fm};old=read(parent/'matrix_scores.json');oldby={r['sample_id']:r for r in old['records']}
    records=[];bridge_error=0.
    for s in p['samples']:
        sid=s['sample_id'];arrays={};matrices={};counts={}
        for role in CONDITIONS:
            if role in ('N','R'):
                ap=parent_path(p,f'features/{sid}/audio_{role}.npy');vp=parent_path(p,f'features/{sid}/visual_{role}.npy')
            else:
                f=feats[sid,role];ap=Path(f['audio_path']);vp=Path(f['visual_path'])
                if sha(ap)!=f['audio_sha256'] or sha(vp)!=f['visual_sha256']:raise ValueError('feature changed')
            a,v=np.load(ap),np.load(vp);counts[role]=min(len(a),len(v));arrays[role]=(a,v)
            matrix=SyncNetEngine.distance_matrix(v,a);matrices[role]=matrix
            dest=RUN/'distances'/sid;dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/f'{role}_native.npy',matrix[valid_rows(counts[role])])
            if role in ('N','R'):
                prev=oldby[sid];q=np.arange(prev['query_start'],prev['query_end'])
                got=metrics(matrix,q);want=prev['cells'][f'V{role}_A{role}']
                for k in METRICS:
                    bridge_error=max(bridge_error,abs(got[k]-want[k]))
                if bridge_error>1e-12:raise ValueError('parent scalar reproduction failed')
        policies=support_policies(counts);result={}
        for policy,roles in policies.items():
            result[policy]={role:metrics(matrices[role],q) for role,q in roles.items()}
        rec=dict(sample_id=sid,speaker_id=s['speaker_id'],counts=counts,policies=result,
                 common_query_count=result['common']['N']['query_count'],
                 low_common_query_warning=result['common']['N']['query_count']<25)
        save(RUN/'scores'/f'{sid}.json',rec);records.append(rec)
        print('SCORED',sid,'queries',rec['common_query_count'],{k:round(v['C'],3) for k,v in result['common'].items()},flush=True)
    all_summaries={}
    for policy in POLICIES:
        all_summaries[policy]={}
        for metric in METRICS:
            x=np.array([[r['policies'][policy][role][metric] for role in CONDITIONS] for r in records])
            contrasts=paired_contrasts(x)
            per_seed={f'T{i+1}':{'T-N':summary(x[:,i+2]-x[:,0]),'T-R':summary(x[:,i+2]-x[:,1])} for i in range(3)}
            all_summaries[policy][metric]=dict(contrasts={k:summary(v) for k,v in contrasts.items()},per_seed=per_seed,
                                             means={role:float(x[:,i].mean()) for i,role in enumerate(CONDITIONS)},T_mean=float(x[:,2:].mean()))
    main=all_summaries['common']['C']['contrasts']
    save(RUN/'analysis.json',dict(status='EXECUTED_PENDING_INDEPENDENT_AUDIT',n_speakers=10,tts_seeds_per_speaker=3,
                                 conditions=CONDITIONS,records=records,summary=all_summaries,parent_score_bridge_max_error=bridge_error,
                                 two_primary_C_bonferroni_positive={k:main[k]['CI97_5'][0]>0 for k in ['T-N','T-R']},
                                 counts_as_independent='speakers only; not 30 TTS draws or video frames',
                                 interpretation='official full ICL generation includes reference decoder context; score effects are not visual accuracy or isolated LM loss effects'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audio','render','extract','analyze'])
    {'audio':audio,'render':render,'extract':extract,'analyze':analyze}[parser.parse_args().stage]()
