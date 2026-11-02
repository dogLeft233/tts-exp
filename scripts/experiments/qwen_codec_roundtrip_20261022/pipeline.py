"""Zero-training, fixed-clock Qwen codec downstream and timing experiment."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/qwen_codec_roundtrip_20261022_attempt01'
ROLES = ('N', 'R')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def execute(cmd, log, cwd=ROOT, timeout=1800, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open('x') as f:
        p = subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env)
    receipt = dict(command=list(map(str, cmd)), cwd=str(cwd), log=str(log), returncode=p.returncode,
                   elapsed_seconds=time.monotonic()-start)
    save(log.with_suffix('.receipt.json'), receipt)
    if p.returncode:
        raise RuntimeError(f'child failed: {log}')
    return receipt


def stats(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict(n=0)
    return dict(n=len(x), mean=float(x.mean()), median_abs=float(np.median(abs(x))),
                mae=float(abs(x).mean()), p95_abs=float(np.quantile(abs(x),.95)), max_abs=float(abs(x).max()))


def bootstrap(values):
    x = np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261022)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(n_speakers=len(x),mean=float(x.mean()),CI95=np.quantile(means,[.025,.975]).tolist(),
                positive=int((x>0).sum()),negative=int((x<0).sum()),values=x.tolist())


def envelope(x, sr=16000):
    # 20ms rectangular energy windows, 10ms advance; same clock for both files.
    win,hop=sr//50,sr//100
    if len(x)<win:
        raise ValueError('too few samples')
    frames=np.lib.stride_tricks.sliding_window_view(np.asarray(x,dtype=float),win)[::hop]
    return np.sqrt(np.mean(frames*frames,axis=1))


def lag_proxy(left,right,radius=20):
    # Positive lag means the right recording has a later corresponding envelope.
    n=min(len(left),len(right))
    radius=min(radius,(n-3)//2)
    if radius<0:
        raise ValueError('short lag proxy')
    q=np.arange(radius,n-radius)
    a=np.asarray(left)[q]
    correlations=[]
    for k in range(-radius,radius+1):
        b=np.asarray(right)[q+k]
        ac,bc=a-a.mean(),b-b.mean()
        den=np.linalg.norm(ac)*np.linalg.norm(bc)
        correlations.append(float(np.dot(ac,bc)/den) if den>1e-12 else None)
    finite=[(c,k) for c,k in zip(correlations,range(-radius,radius+1)) if c is not None]
    if not finite:
        return dict(measurable=False)
    c,k=max(finite,key=lambda v:(v[0],-abs(v[1])))
    return dict(measurable=True,lag_s=k*.01,correlation=c,at_search_boundary=abs(k)==radius,
                radius_s=radius*.01,correlations=correlations)


def prepare_audio():
    import soundfile as sf
    p=read(RUN/'input_manifest.json')
    bindings=read(RUN/'execution_bindings.json')
    cm=read(RUN/bindings['codec_directory']/'codec_manifest.json')
    cr=cm.get('records',cm.get('samples',[]))
    by={r['sample_id']:r for r in cr}
    out=[]
    timing=[]
    for sample in p['samples']:
        sid=sample['sample_id']; rec=by[sid]
        reconstructed=rec.get('reconstruction_path',rec.get('output_path'))
        if not reconstructed:
            raise ValueError(f'no reconstruction path {list(rec)}')
        if sha(reconstructed)!=rec['sha256'] or sha(sample['input24_path'])!=sample['input24_sha256']:
            raise ValueError('codec/raw input hash changed')
        paths={'N':sample['input24_path'],'R':reconstructed}
        waves={}
        for role in ROLES:
            source=Path(paths[role]); target=RUN/'audio'/sid/f'{role}_16k_pcm16.wav'
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_name(f'{role}_16k_float.wav')
            cmd=['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',str(source),'-ac','1','-ar','16000','-c:a','pcm_f32le',str(temp)]
            execute(cmd,RUN/'logs'/f'canonical_{sid}_{role}.log',timeout=60)
            x,sr=sf.read(temp,dtype='float32')
            if sr!=16000 or x.ndim!=1 or not np.isfinite(x).all():
                raise ValueError('invalid canonical input')
            rms=float(np.sqrt(np.mean(x.astype(float)**2)));gain=p['primary_audio']['target_rms']/rms
            peak=float(abs(x).max()*gain)
            if peak>=1 or not np.isfinite(gain):
                raise ValueError('RMS normalization would clip')
            sf.write(target,x*np.float32(gain),sr,subtype='PCM_16')
            y,_=sf.read(target,dtype='int16')
            if len(y)!=len(x):raise ValueError('sample count changed')
            waves[role]=y.astype(float)/32768
            out.append(dict(sample_id=sid,speaker_id=sample['speaker_id'],role=role,path=str(target),sha256=sha(target),
                            source_path=str(source),source_sha256=sha(source),samples=len(y),sr=sr,
                            pcm_sha256=hashlib.sha256(y.astype('<i2').tobytes()).hexdigest(),
                            rms_before=rms,gain=gain,rms_after=float(np.sqrt(np.mean(waves[role]**2))),peak_after=float(abs(waves[role]).max()),
                            original_decoder_length_preserved=True))
        en,er=envelope(waves['N']),envelope(waves['R'])
        glob=lag_proxy(en,er)
        local=[]
        for start in range(0,min(len(en),len(er))-100+1,50):
            local.append(dict(start_s=start*.01,**lag_proxy(en[start:start+100],er[start:start+100])))
        timing.append(dict(sample_id=sid,speaker_id=sample['speaker_id'],duration_N_s=len(waves['N'])/16000,
                           duration_R_s=len(waves['R'])/16000,length_delta_s=(len(waves['R'])-len(waves['N']))/16000,
                           envelope_global=glob,envelope_local=local,
                           warning='energy-envelope correlations are acoustic proxies, not human-annotated event timing'))
        print('AUDIO_READY',sid,'length_delta_s',timing[-1]['length_delta_s'],flush=True)
    save(RUN/'audio_manifest.json',dict(input_manifest_sha256=sha(RUN/'input_manifest.json'),records=out))
    save(RUN/'timing/envelope_timing.json',dict(records=timing,summary=dict(length_delta=stats([r['length_delta_s'] for r in timing]))))


def render():
    p=read(RUN/'input_manifest.json'); rows=read(RUN/'audio_manifest.json')['records']
    w2l=ROOT/'third_party/Wav2Lip'
    checkpoint=w2l/'checkpoints/wav2lip_gan.pth'
    assert sha(checkpoint)==p['generation']['weights_sha256']
    assert sha(p['generation']['face'])==p['generation']['face_sha256']
    results=[]
    for row in rows:
        sid,role=row['sample_id'],row['role']
        assert sha(row['path'])==row['sha256']
        work=RUN/'render_work'/sid/role
        work.mkdir(parents=True,exist_ok=False);(work/'temp').mkdir()
        output=RUN/'video'/sid/f'{role}.mp4';output.parent.mkdir(parents=True,exist_ok=True)
        cmd=['/home/wjj/.venvs/wav2lip/bin/python',str(w2l/'inference.py'),'--checkpoint_path',str(checkpoint),
             '--face',p['generation']['face'],'--audio',row['path'],'--outfile',str(output),
             '--face_det_batch_size','4','--wav2lip_batch_size','4','--nosmooth']
        rec=execute(cmd,RUN/'logs'/f'render_{sid}_{role}.log',cwd=work,timeout=180)
        if not output.is_file() or output.stat().st_size==0:raise ValueError('no video')
        results.append(dict(sample_id=sid,speaker_id=row['speaker_id'],role=role,path=str(output),sha256=sha(output),
                            audio_sha256=row['sha256'],receipt=rec))
        print('RENDERED',sid,role,flush=True)
    save(RUN/'video_manifest.json',dict(records=results,checkpoint_sha256=sha(checkpoint),inference_sha256=sha(w2l/'inference.py')))


def score():
    import torch
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    from scripts.experiments.repeated_take_crossover_20261015.run_matrix import _extract_fixed_frames,_visual_features,_probe_video
    p=read(RUN/'input_manifest.json'); aa=read(RUN/'audio_manifest.json')['records'];vv=read(RUN/'video_manifest.json')['records']
    ab={(r['sample_id'],r['role']):r for r in aa};vb={(r['sample_id'],r['role']):r for r in vv}
    model=ROOT/'third_party/syncnet_python/data/syncnet_v2.model'
    assert sha(model)=='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442'
    torch.set_num_threads(2)
    torch.manual_seed(20261022)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    engine=SyncNetEngine(model_path=model,batch_size=32,device='cuda')
    results=[]
    try:
        for s in p['samples']:
            sid=s['sample_id']; features={};metadata={};cells={};nframes=None
            for role in ROLES:
                a,v=ab[sid,role],vb[sid,role]
                assert sha(a['path'])==a['sha256'] and sha(v['path'])==v['sha256']
                ae,am=engine.extract_audio(a['path'])
                assert am['pcm_sha256']==a['pcm_sha256']
                probe=_probe_video(Path(v['path']))
                frames,fm=_extract_fixed_frames(Path(v['path']),RUN/'decoded_frames'/sid/role,p['generation']['score_box_xyxy'])
                ve=_visual_features(engine,frames)
                if role=='N':nframes=frames
                dest=RUN/'features'/sid;dest.mkdir(parents=True,exist_ok=True)
                np.save(dest/f'audio_{role}.npy',ae);np.save(dest/f'visual_{role}.npy',ve)
                features[role]=(ae,ve);metadata[role]=dict(audio=am,video=probe,frames=fm)
            repeat_audio,_=engine.extract_audio(ab[sid,'N']['path'])
            repeat_visual=_visual_features(engine,nframes)
            repeat=dict(audio_max_abs=float(abs(repeat_audio-features['N'][0]).max()),
                        visual_max_abs=float(abs(repeat_visual-features['N'][1]).max()))
            if max(repeat.values())>1e-5:raise ValueError('repeat feature control failed')
            start=15;end=min(min(a.shape[0],v.shape[0]) for a,v in features.values())-15
            if end<=start:raise ValueError('empty shared all-lag support')
            for vr in ROLES:
                for ar in ROLES:
                    matrix=engine.distance_matrix(features[vr][1],features[ar][0])[start:end]
                    curve=matrix.mean(axis=0);arg=int(curve.argmin());b=float(np.median(curve));d=float(curve[arg])
                    key=f'V{vr}_A{ar}';dest=RUN/'distances'/sid;dest.mkdir(parents=True,exist_ok=True)
                    np.save(dest/f'{key}.npy',matrix)
                    cells[key]=dict(C=b-d,B=b,Dmin=d,D0=float(curve[15]),offset=15-arg,
                                    boundary=arg in (0,30),curve=curve.tolist(),matrix_sha256=sha(dest/f'{key}.npy'))
            r=dict(sample_id=sid,speaker_id=s['speaker_id'],query_start=start,query_end=end,queries=end-start,
                   cells=cells,metadata=metadata,repeat_control=repeat)
            save(RUN/'scores'/f'{sid}.json',r);results.append(r)
            print('SCORED',sid,{k:round(v['C'],3) for k,v in cells.items()},flush=True)
    finally:
        engine.close()
    contrasts={}
    pairs={'R_video_audio_swap':('VR_AR','VR_AN'),'N_video_audio_swap':('VN_AR','VN_AN'),
           'fixed_N_audio_generation':('VR_AN','VN_AN'),'self_paired_total':('VR_AR','VN_AN')}
    for name,(a,b) in pairs.items():
        contrasts[name]={metric:bootstrap([r['cells'][a][metric]-r['cells'][b][metric] for r in results])
                         for metric in ('C','B','Dmin','D0','offset')}
    save(RUN/'matrix_scores.json',dict(status='complete',n_speakers=len(results),records=results,contrasts=contrasts,
                                      SyncNet_weights_sha256=sha(model),definitions='C=B-Dmin; D0 at zero lag; offset=15-argmin',
                                      no_physical_timing_inference_from_scores=True))


def mfa():
    from scripts.experiments.repeated_take_crossover_20261015.timing_diagnostics import _check_mfa_pins
    p=read(RUN/'input_manifest.json');samples={s['sample_id']:s for s in p['samples']};rows=read(RUN/'audio_manifest.json')['records']
    pins=_check_mfa_pins()
    corpus=RUN/'timing/mfa_corpus';corpus.mkdir(parents=True,exist_ok=False)
    for row in rows:
        d=corpus/row['speaker_id'];d.mkdir(exist_ok=True)
        stem=row['sample_id']+'_'+row['role']
        os.symlink(row['path'],d/(stem+'.wav'))
        (d/(stem+'.lab')).write_text(samples[row['sample_id']]['transcript'])
    exe='/home/wjj/miniconda3/envs/mfa/bin/mfa'
    cmd=[exe,'align',str(corpus),'/home/wjj/Documents/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict',
         '/home/wjj/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip',str(RUN/'timing/textgrids'),
         '--num_jobs','2','--temporary_directory',str(RUN/'timing/mfa_temp'),'--clean']
    env=dict(os.environ);env['PATH']=str(Path(exe).parent)+os.pathsep+env.get('PATH','')
    rec=execute(cmd,RUN/'logs/mfa.log',timeout=1800,env=env)
    save(RUN/'timing/mfa_execution.json',dict(pins=pins,receipt=rec))
    summarize_timing()


def summarize_timing():
    from scripts.experiments.repeated_take_crossover_20261015.timing_diagnostics import parse_textgrid,_phone_intervals,_lcs_pairs
    p=read(RUN/'input_manifest.json');results=[];phone_rows=[]
    for s in p['samples']:
        sid=s['sample_id'];parsed={}
        for role in ROLES:
            matches=sorted((RUN/'timing/textgrids').rglob(f'{sid}_{role}.TextGrid'))
            if len(matches)!=1:raise ValueError(f'expected one TextGrid {sid} {role}')
            grid=matches[0];tiers=parse_textgrid(grid)
            phones,onset,coverage,labels,unknown,unknown_dur=_phone_intervals(tiers)
            parsed[role]=dict(phones=phones,intervals=tiers['phones'],onset=onset,coverage=coverage,unknown=unknown,unknown_duration=unknown_dur,
                               grid=str(grid),grid_sha256=sha(grid))
        n,r=parsed['N'],parsed['R'];pairs=_lcs_pairs(n['phones'],r['phones'])
        if not pairs:raise ValueError('no matched phones')
        onset=r['onset']-n['onset'];events=[]
        for i,j in pairs:
            a,b=n['phones'][i],r['phones'][j]
            delta_start=b['start_s']-a['start_s'];delta_end=b['end_s']-a['end_s']
            e=dict(sample_id=sid,speaker_id=s['speaker_id'],label=a['label'],i_N=i,i_R=j,
                   N_start=a['start_s'],R_start=b['start_s'],N_end=a['end_s'],R_end=b['end_s'],
                   start_delta_s=delta_start,end_delta_s=delta_end,center_delta_s=(delta_start+delta_end)/2,
                   duration_delta_s=b['duration_s']-a['duration_s'],center_after_onset_s=(delta_start+delta_end)/2-onset)
            events.append(e);phone_rows.append(e)
        silence=[];unknown_gaps=[]
        for (i,j),(ni,nj) in zip(pairs,pairs[1:]):
            if ni==i+1 and nj==j+1:
                ngap=n['phones'][ni]['start_s']-n['phones'][i]['end_s']
                rgap=r['phones'][nj]['start_s']-r['phones'][j]['end_s']
                if max(ngap,rgap)>.02:
                    labels={}
                    for role,a,b in [('N',n['phones'][i]['end_s'],n['phones'][ni]['start_s']),
                                     ('R',r['phones'][j]['end_s'],r['phones'][nj]['start_s'])]:
                        labels[role]=[x['label'] for x in parsed[role]['intervals']
                                      if min(b,x['end_s'])-max(a,x['start_s'])>1e-6]
                    gap=dict(N_gap_s=ngap,R_gap_s=rgap,delta_s=rgap-ngap,labels=labels)
                    if any(v.casefold() not in {'','sil','sp','noise'} for row in labels.values() for v in row):
                        unknown_gaps.append(gap)
                    else:silence.append(gap)
        results.append(dict(sample_id=sid,speaker_id=s['speaker_id'],N=n,R=r,n_matched=len(pairs),
                            match_fraction_N=len(pairs)/len(n['phones']),match_fraction_R=len(pairs)/len(r['phones']),
                            onset_delta_s=onset,center=stats([e['center_delta_s'] for e in events]),
                            start=stats([e['start_delta_s'] for e in events]),end=stats([e['end_delta_s'] for e in events]),
                            duration=stats([e['duration_delta_s'] for e in events]),
                            center_after_onset=stats([e['center_after_onset_s'] for e in events]),
                            center_abs_over_20ms=sum(abs(e['center_delta_s'])>.0200001 for e in events),
                            center_abs_over_40ms=sum(abs(e['center_delta_s'])>.0400001 for e in events),
                            silence_gaps=silence,unknown_or_non_silence_gaps=unknown_gaps))
        print('TIMING',sid,'matched',len(pairs),'median_ms',results[-1]['center']['median_abs']*1000,flush=True)
    summary={key:bootstrap([r[key]['mae'] for r in results]) for key in ('center','start','end','duration','center_after_onset')}
    summary['onset_delta']=bootstrap([r['onset_delta_s'] for r in results])
    save(RUN/'timing/mfa_timing_v2.json',dict(records=results,speaker_equal_summary=summary,
                caution='MFA estimated boundaries, not manual ground truth; no temporal equivalence claim; medians alone hide tails'))
    dest=RUN/'timing/phone_events_v2.csv'
    with dest.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(phone_rows[0]));w.writeheader();w.writerows(phone_rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audio','render','score','mfa','timing_summary'])
    args=parser.parse_args()
    {'audio':prepare_audio,'render':render,'score':score,'mfa':mfa,'timing_summary':summarize_timing}[args.stage]()


if __name__=='__main__':main()
