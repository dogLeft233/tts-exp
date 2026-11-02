"""Expanded frozen self-score experiment; reuse pilot media/features unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_codec_roundtrip_20261022.pipeline import sha,read,save,execute
from scripts.experiments.qwen_full_tts_extension_20261022.pipeline import metrics,support_policies,summary,paired_contrasts

RUN=ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01'
ROLES=['N','R','T1','T2','T3']
METRICS=['C','B','Dmin','D0','offset']
POLICIES=['common','native_full','equal_count_spread']


def protocol():
    p=read(RUN/'protocol.json')
    for path,row in p['reused_bindings'].items():
        if sha(path)!=row['sha256']:raise ValueError('reused manifest changed: '+path)
    return p


def old_audio(p):
    rows=read(Path(p['parent_run'])/'audio_manifest.json')['records']+read(Path(p['pilot_tts_run'])/'audio_manifest.json')['records']
    return [dict(r,cohort='pilot10',reused=True) for r in rows]


def audio():
    import soundfile as sf
    p=protocol();out=old_audio(p);generation=RUN/'generation'
    rr={r['sample_id']:r for r in read(generation/'codec_manifest.json')['records']}
    tt={(r['sample_id'],int(r['seed_index'])):r for r in read(generation/'tts_manifest.json')['records']}
    if len(rr)!=30 or len(tt)!=90:raise ValueError('incomplete generation')
    for s in [s for s in p['samples'] if s['cohort']=='new30']:
        sid=s['sample_id']
        for role in ROLES:
            if role=='N':src,h=s['input24_path'],s['input24_sha256']
            else:
                r=rr[sid] if role=='R' else tt[sid,int(role[1:])];src,h=r['path'],r['sha256']
            if sha(src)!=h:raise ValueError('raw source changed')
            dest=RUN/'audio'/sid/f'{role}_16k_pcm16.wav';dest.parent.mkdir(parents=True,exist_ok=True)
            f32=dest.with_name(f'{role}_16k_float.wav')
            execute(['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',src,'-ac','1','-ar','16000','-c:a','pcm_f32le',f32],RUN/'logs'/f'canonical_{sid}_{role}.log',timeout=60)
            x,sr=sf.read(f32,dtype='float32')
            if sr!=16000 or x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('invalid canonical float')
            rms=float(np.sqrt(np.mean(x.astype(float)**2)));gain=p['target_rms']/rms
            if rms<=0 or float(abs(x).max()*gain)>=1:raise ValueError('silent or clipping normalization')
            sf.write(dest,x*np.float32(gain),sr,subtype='PCM_16');pcm,_=sf.read(dest,dtype='int16')
            assert len(pcm)==len(x)
            out.append(dict(sample_id=sid,speaker_id=s['speaker_id'],cohort='new30',reused=False,role=role,path=str(dest),sha256=sha(dest),
                            samples=len(pcm),sr=16000,source_path=src,source_sha256=h,rms_before=rms,gain=gain,
                            rms_after=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))),
                            pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()))
            print('AUDIO',sid,role,len(pcm)/16000,flush=True)
    if len(out)!=200 or len({(r['sample_id'],r['role']) for r in out})!=200:raise ValueError('not full40x5 audio')
    save(RUN/'audio_manifest.json',dict(records=out,new_count=150,reused_count=50,protocol_sha256=sha(RUN/'protocol.json')))


def render():
    p=protocol();w2l=ROOT/'third_party/Wav2Lip';ckpt=w2l/'checkpoints/wav2lip_gan.pth'
    if sha(ckpt)!=p['downstream']['weights_sha256'] or sha(p['downstream']['face'])!=p['downstream']['face_sha256']:raise ValueError('model/face changed')
    if sha(w2l/'inference.py')!='42cfc8d3060921e6714c8e08243e5508d2e637d29cd0be78760492c5d26380a9':raise ValueError('W2L source differs from pilot')
    reused=read(Path(p['parent_run'])/'video_manifest.json')['records']+read(Path(p['pilot_tts_run'])/'video_manifest.json')['records']
    out=[dict(r,cohort='pilot10',reused=True) for r in reused]
    for row in read(RUN/'audio_manifest.json')['records']:
        if row['reused']:continue
        sid,role=row['sample_id'],row['role'];assert sha(row['path'])==row['sha256']
        work=RUN/'render_work'/sid/role;work.mkdir(parents=True,exist_ok=False);(work/'temp').mkdir()
        dest=RUN/'video'/sid/f'{role}.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
        cmd=['/home/wjj/.venvs/wav2lip/bin/python',str(w2l/'inference.py'),'--checkpoint_path',str(ckpt),'--face',p['downstream']['face'],
             '--audio',row['path'],'--outfile',str(dest),'--face_det_batch_size','4','--wav2lip_batch_size','4','--nosmooth']
        receipt=execute(cmd,RUN/'logs'/f'render_{sid}_{role}.log',cwd=work,timeout=240)
        if not dest.exists() or not dest.stat().st_size:raise ValueError('missing render')
        out.append(dict(sample_id=sid,speaker_id=row['speaker_id'],role=role,cohort='new30',reused=False,path=str(dest),sha256=sha(dest),audio_sha256=row['sha256'],receipt=receipt))
        print('RENDER',len(out)-50,'/150',sid,role,flush=True)
    save(RUN/'video_manifest.json',dict(records=out,new_count=150,reused_count=50,weights_sha256=sha(ckpt),inference_sha256=sha(w2l/'inference.py')))


def reused_features(p):
    parent=Path(p['parent_run']);pilot=Path(p['pilot_tts_run']);rows=[];sealed={r['path']:r for r in read(parent/'artifact_manifest.json')['files']}
    for old in read(parent/'matrix_scores.json')['records']:
        sid=old['sample_id']
        for role in ['N','R']:
            aa=f'features/{sid}/audio_{role}.npy';vv=f'features/{sid}/visual_{role}.npy';m=old['metadata'][role]
            assert sha(parent/aa)==sealed[aa]['sha256'] and sha(parent/vv)==sealed[vv]['sha256']
            rows.append(dict(sample_id=sid,speaker_id=old['speaker_id'],role=role,cohort='pilot10',reused=True,
                             audio_path=str(parent/aa),audio_sha256=sealed[aa]['sha256'],visual_path=str(parent/vv),visual_sha256=sealed[vv]['sha256'],
                             audio_meta=m['audio'],video_meta=m['video'],frames_meta=m['frames'],repeat=old['repeat_control'] if role=='N' else {}))
    rows.extend(dict(r,cohort='pilot10',reused=True) for r in read(pilot/'feature_manifest.json')['records'])
    assert len(rows)==50
    return rows


def extract():
    import torch
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    from scripts.experiments.repeated_take_crossover_20261015.run_matrix import _extract_fixed_frames,_visual_features,_probe_video
    p=protocol();audio={(r['sample_id'],r['role']):r for r in read(RUN/'audio_manifest.json')['records']}
    model=ROOT/'third_party/syncnet_python/data/syncnet_v2.model'
    assert sha(model)=='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442'
    torch.set_num_threads(2);torch.manual_seed(20261022);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    rows=reused_features(p);engine=SyncNetEngine(model_path=model,batch_size=32,device='cuda')
    try:
        for v in read(RUN/'video_manifest.json')['records']:
            if v['reused']:continue
            sid,role=v['sample_id'],v['role'];a=audio[sid,role]
            assert sha(v['path'])==v['sha256'] and sha(a['path'])==a['sha256']
            ae,am=engine.extract_audio(a['path']);assert am['pcm_sha256']==a['pcm_sha256']
            frames,fm=_extract_fixed_frames(Path(v['path']),RUN/'decoded_frames'/sid/role,p['downstream']['score_box_xyxy'])
            ve=_visual_features(engine,frames);dest=RUN/'features'/sid;dest.mkdir(parents=True,exist_ok=True)
            ap=dest/f'audio_{role}.npy';vp=dest/f'visual_{role}.npy';np.save(ap,ae);np.save(vp,ve);repeat={}
            if len(rows)==50:
                ar,_=engine.extract_audio(a['path']);vr=_visual_features(engine,frames)
                repeat=dict(audio_max_abs=float(abs(ar-ae).max()),visual_max_abs=float(abs(vr-ve).max()));assert max(repeat.values())<=1e-5
            rows.append(dict(sample_id=sid,speaker_id=v['speaker_id'],role=role,cohort='new30',reused=False,
                             audio_path=str(ap),audio_sha256=sha(ap),visual_path=str(vp),visual_sha256=sha(vp),
                             audio_meta=am,video_meta=_probe_video(Path(v['path'])),frames_meta=fm,repeat=repeat))
            print('FEATURES',len(rows)-50,'/150',sid,role,len(ae),len(ve),flush=True)
    finally:engine.close()
    save(RUN/'feature_manifest.json',dict(records=rows,new_count=150,reused_count=50,weights_sha256=sha(model)))


def reliability(rows,samples):
    samples={s['sample_id']:s for s in samples};out={}
    for r in rows:
        ratio=r['samples']/r['sample_rate_hz']/samples[r['sample_id']]['duration_s']
        valid=r['eos_in_talker_sequences'] is True and .5<=ratio<=2.
        out[r['sample_id'],f"T{r['seed_index']}"]=dict(qualified=valid,EOS=r['eos_in_talker_sequences'],duration_ratio=ratio,
                                                     seed=r['seed'],seed_index=r['seed_index'])
    return out


def grouped_summary(records):
    if not records:return dict(measurable_speakers=0,summaries={})
    result={}
    for policy in POLICIES:
        result[policy]={}
        for metric in METRICS:
            x=np.array([[r['policies'][policy][role][metric] for role in ROLES] for r in records]);contrasts=paired_contrasts(x)
            result[policy][metric]=dict(contrasts={k:summary(v) for k,v in contrasts.items()},
                  means={k:float(x[:,i].mean()) for i,k in enumerate(ROLES)},T_mean=float(x[:,2:].mean()),
                  per_seed={f'T{i+1}':{key:summary(x[:,i+2]-x[:,baseline]) for key,baseline in [('T-N',0),('T-R',1)]} for i in range(3)})
            if metric=='C':
                result[policy][metric]['leave_one_out']={key:[dict(omitted=records[i]['sample_id'],mean=float(np.delete(v,i).mean())) for i in range(len(v))] if len(v)>1 else [] for key,v in contrasts.items()}
    conditional={}
    good=[r for r in records if any(v['qualified'] for v in r['draw_reliability'].values())]
    for policy in POLICIES:
        conditional[policy]={}
        for metric in METRICS:
            if not good:conditional[policy][metric]=dict(n_speakers=0);continue
            values=[]
            for r in good:
                c=r['policies'][policy];t=np.mean([c[role][metric] for role,v in r['draw_reliability'].items() if v['qualified']])
                values.append([t-c['N'][metric],t-c['R'][metric]])
            x=np.array(values);conditional[policy][metric]=dict(contrasts={key:summary(x[:,i]) for i,key in enumerate(['T-N','T-R'])})
    return dict(measurable_speakers=len(records),summaries=result,conditional_generation=conditional,
                conditional_coverage=len(good),conditional_missing=[r['sample_id'] for r in records if r not in good])


def analyze():
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    p=protocol();f={(r['sample_id'],r['role']):r for r in read(RUN/'feature_manifest.json')['records']}
    ts=read(Path(p['pilot_tts_run'])/'tts/tts_manifest.json')['records']+read(RUN/'generation/tts_manifest.json')['records']
    rel=reliability(ts,p['samples']);old={r['sample_id']:r for r in read(Path(p['pilot_tts_run'])/'analysis.json')['records']}
    records=[];unmeasurable=[];bridge_error=0.
    for s in p['samples']:
        sid=s['sample_id'];counts={};matrices={}
        for role in ROLES:
            rr=f[sid,role];assert sha(rr['audio_path'])==rr['audio_sha256'] and sha(rr['visual_path'])==rr['visual_sha256']
            a=np.load(rr['audio_path']);v=np.load(rr['visual_path']);n=min(len(a),len(v));counts[role]=n
            matrix=SyncNetEngine.distance_matrix(v,a);matrices[role]=matrix
            dest=RUN/'distances'/sid;dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/f'{role}_native.npy',matrix[15:max(15,n-15)])
        if min(counts.values())<=30:
            unmeasurable.append(dict(sample_id=sid,speaker_id=s['speaker_id'],cohort=s['cohort'],counts=counts,reason='no complete real common all-lag support'));continue
        policies={policy:{role:metrics(matrices[role],queries) for role,queries in roles.items()} for policy,roles in support_policies(counts).items()}
        rec=dict(sample_id=sid,speaker_id=s['speaker_id'],cohort=s['cohort'],counts=counts,policies=policies,
                 common_query_count=policies['common']['N']['query_count'],low_common_query_warning=policies['common']['N']['query_count']<25,
                 draw_reliability={role:rel[sid,role] for role in ROLES[2:]})
        if s['cohort']=='pilot10':
            for policy in POLICIES:
                for role in ROLES:
                    for metric in METRICS:bridge_error=max(bridge_error,abs(policies[policy][role][metric]-old[sid]['policies'][policy][role][metric]))
            if bridge_error>1e-12:raise ValueError('pilot numerical bridge failed')
        save(RUN/'scores'/f'{sid}.json',rec);records.append(rec)
        print('SCORED',sid,s['cohort'],rec['common_query_count'],flush=True)
    cohorts={name:grouped_summary([r for r in records if name=='all40' or r['cohort']==name]) for name in ['new30','all40','pilot10']}
    main=cohorts['new30']['summaries'].get('common',{}).get('C',{}).get('contrasts',{})
    save(RUN/'analysis.json',dict(status='EXECUTED_PENDING_INDEPENDENT_AUDIT' if not unmeasurable else 'INCOMPLETE_MEASURABLE_COHORT',
         protocol_sha256=sha(RUN/'protocol.json'),records=records,unmeasurable=unmeasurable,cohorts=cohorts,pilot_bridge_max_error=bridge_error,
         two_main_new30_bonferroni_positive={k:main[k]['CI97_5'][0]>0 for k in ['T-N','T-R']} if main and not any(r['cohort']=='new30' for r in unmeasurable) else None,
         all120_draws_reliability=[dict(sample_id=sid,role=role,**v) for (sid,role),v in rel.items()],
         interpretation='new30 primary development expansion; all40 secondary; conditional quality subset is diagnostic; no perception or isolated LM objective claim'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audio','render','extract','analyze'])
    globals()[parser.parse_args().stage]()
