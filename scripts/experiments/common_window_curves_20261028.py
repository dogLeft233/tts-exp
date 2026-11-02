"""Frozen common-support native cloud curves; derived endpoints, not official benchmarks.

Reuses verified contextual AVS/CoGenAV encodings. SyncNet audio is recomputed
from canonical PCM because the historical AVI audio has a measured delay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

RUN_NAME = 'common_window_curves_cloud_20261028_attempt01'
LAGS = np.arange(-15, 16)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, p)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def endpoints(matrix):
    c = np.asarray(matrix, dtype=np.float64).mean(0)
    best, bg, zero = float(c.max()), float(np.median(c)), float(c[15])
    return dict(best=best, background=bg, zero=zero, contrast=best-bg,
                zero_contrast=zero-bg, search_uplift=best-zero,
                peak_lag_frames=int(LAGS[c.argmax()]), curve=c.tolist())


def cosine_matrix(a, v, queries, relu=False):
    a, v = np.asarray(a, np.float64), np.asarray(v, np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    ix = np.asarray(queries)[:, None] + np.arange(5)[None, :]
    ans = []
    for k in LAGS:
        x = np.einsum('qfd,qfd->qf', a[ix+k], v[ix])
        ans.append((np.maximum(x, 0) if relu else x).mean(1))
    return np.stack(ans, 1)


def sync_matrix(a, v, queries, unit=False):
    a, v = np.asarray(a, np.float64), np.asarray(v, np.float64)
    if unit:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
    # Official F.pairwise_distance(x1,x2,p=2,eps=1e-6): L2(x1-x2+eps).
    return np.stack([-np.linalg.norm(v[queries]-a[queries+k]+1e-6, axis=1) for k in LAGS], 1)


def freeze(repo, out):
    import cv2
    from scipy.io import wavfile
    avs = repo/'runs/avs_u_zh_pilot_20261027_attempt01'
    cloud = repo/'runs/cloud_av_preference20_20261027_attempt05'
    replay = repo/'runs/syncnet_audio_replay20_20261024_attempt01'
    records = read(avs/'inputs.json')
    oldn = {x['id']: x for x in read(replay/'input_manifest.json')['rows'] if x['arm']=='N'}
    oldc = {x['id']: x for x in read(cloud/'sync_scores.json')}
    strict = {p['source']: p['strict_reference'] for p in read(cloud/'paired_scores.json')}
    selected = []
    for r in records:
        key = r['id']
        co = (repo/'runs/cogenav_sync_full20_20261024_attempt01' if r['arm']=='N' else cloud/'cogenav')
        cap = cv2.VideoCapture(r['video_path'])
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert fps == 25
        sr, pcm = wavfile.read(r['audio_path'])
        assert sr == 16000 and pcm.ndim == 1 and pcm.dtype == np.int16
        if r['arm']=='N':
            prev = oldn[key]
            track = prev['track']
            assert track['track_start_frame'] == 0
            crop = Path(prev['crop_path'])
            sequence = replay/'items'/key/'sequence.npz'
            trace = read(replay/'items'/key/'replay.json')
            assert trace['pass_replay'] and sha(sequence)==trace['sequence_sha256']
            assert sha(crop)==track['crop_sha256']
            sn = len(np.load(sequence)['video1024'])
            oldclock = trace['clock']
        else:
            prev = oldc[key]
            p = Path(prev['activesd_path'])
            tracks = pickle.loads((p.parent/'tracks.pckl').read_bytes())
            assert len(tracks)==1
            fi = tracks[0]['track']['frame']
            assert np.array_equal(fi, np.arange(len(fi)))
            crop = p.parents[2]/'pycrop'/p.parent.name/'00000.avi'
            sequence = None
            sn = prev['windows']
            oldclock = None
        avpath, copath = avs/'embeddings'/f'{key}.npz', co/'embeddings'/f'{key}.npz'
        az, cz = np.load(avpath), np.load(copath)
        assert sha(r['audio_path'])==r['audio_sha256'] and sha(r['video_path'])==r['video_sha256']
        assert str(az['input_audio_sha256'])==r['audio_sha256']
        assert str(az['input_video_sha256'])==r['video_sha256']
        assert az['audio'].shape==az['video'].shape
        assert cz['audio'].shape==cz['video'].shape
        # Never use synthetic AVI tail or padded contextual embedding positions.
        n = min(frames, len(pcm)//640, len(az['audio']), len(cz['audio']))
        n_sync = min(sn, n-5)
        valid = n_sync - 30
        assert valid > 0, (key, n, n_sync)
        selected.append(dict(r, strict_reference=strict[r['sample_id']], source_frames=frames,
            samples=len(pcm), common_native_frames=n, sync_windows=n_sync, valid_query_count=valid,
            avs_cache=str(avpath), avs_cache_sha256=sha(avpath), co_cache=str(copath), co_cache_sha256=sha(copath),
            avs_controls=str(avs/'embeddings'/f'{key}_controls.npz') if r['arm']=='N' else None,
            co_controls=str(co/'embeddings'/f'{key}_rawshifts.npz'),
            sync_crop=str(crop), sync_crop_sha256=sha(crop),
            sync_video_cache=str(sequence) if sequence else None,
            sync_video_cache_sha256=sha(sequence) if sequence else None, historical_clock=oldclock))
    for sid in sorted({x['sample_id'] for x in selected}):
        group = [x for x in selected if x['sample_id']==sid]
        assert len(group)==4
        count = min(x['valid_query_count'] for x in group)
        for row in group:
            start = 15 + (row['valid_query_count']-count)//2
            row['query_start'], row['query_count'] = start, count
            row['query_end_exclusive'] = start+count
    protocol = dict(status='FROZEN_BEFORE_NEW_CURVES', run_id=RUN_NAME,
        cohort='20 Mandarin sources, 20N/60 cloudTTS, Wav2Lip GAN image3, existing RMS-normalized canonical PCM',
        lags_frames=LAGS.tolist(), lag_seconds=(LAGS/25).tolist(), positive_lag='select later audio feature positions, not delay sign correction',
        query='five-video-frame window start q; use q..q+4 for AVS/Co; SyncNet standard video q..q+4 and MFCC 4q:4q+20',
        support='all lags real native input; no padded/synthetic tail query; per-source N+3T min available query count, each clip central selection',
        context='AVS/Co cached full-context encodings; feature-index lag curves, NOT 31 shifted waveform re-encodes. Different receptive fields/ROI remain.',
        sync_audio='recompute official MFCC int16 canonical PCM from sample0; historical N AVI scoring audio has +1105sample delay, not reused',
        sync_visual='official existing track crop JPEG features; N frozen video1024, C re-forward official face crop JPEGs; all track start0 contiguous; truncate synthetic tail',
        primary=dict(sync='negative official Euclidean with eps1e-6', avs='five-frame mean raw cosine', co='five-frame mean ReLU cosine, retain paper rectification'),
        sensitivity=dict(sync_unit='negative eps1e-6 Euclidean after endpoint unit norm', co_raw='unrectified five-frame cosine'),
        endpoints='curve=query mean first; best=max31, background=median31, zero=position15, contrast=best-background; paired contrast=delta_best-delta_background',
        stats='source-equal mean of three T-N draws; all20/strict12/extended8; 10000 source bootstrap seed20261028 marginal descriptive95CI; no percentage mediation',
        controls='AVS20N real +/-400ms cached PCM controls; Co all80 real +/-160ms controls; Sync first three lexical N real +/-160ms canonical PCM reencode. Relative peak movement tolerance1frame; >=80% gate, failed result retained diagnostic.',
        parity='first three lexical N rerun audio net from saved official MFCC and compare saved official final1024 maxerror<=1e-5; canonical/new geometry separately named',
        source_sha256=sha(__file__), created_unix=time.time(), input_manifest_sha256=sha(avs/'inputs.json'))
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', protocol)
    write(out/'inputs.json', selected)
    shutil.copy2(__file__, out/'source_frozen.py')
    write(out/'status.json', dict(status='FROZEN', inputs=len(selected), sources=20))
    print('FROZEN', len(selected), 'queries', min(x['query_count'] for x in selected), max(x['query_count'] for x in selected), flush=True)


def run(repo, out):
    import torch
    import cv2
    import python_speech_features as psf
    from scipy.io import wavfile
    sys.path.insert(0, str(repo/'third_party/syncnet_python'))
    from SyncNetInstance import SyncNetInstance
    torch.set_num_threads(2)
    torch.manual_seed(20261028)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    model=SyncNetInstance(device='cuda')
    model.loadParameters(str(repo/'third_party/syncnet_python/data/syncnet_v2.model'))
    net=model.__S__.eval()
    rows=read(out/'inputs.json')
    (out/'items').mkdir(exist_ok=True)
    def aud(pcm, n):
        mfcc=psf.mfcc(pcm,16000).T
        assert mfcc.shape[1] >= 4*(n-1)+20
        x=np.stack([mfcc[:,4*i:4*i+20] for i in range(n)])[:,None].astype(np.float32)
        with torch.inference_mode():
            a=np.concatenate([net.forward_aud(torch.from_numpy(x[i:i+20]).cuda()).cpu().numpy() for i in range(0,n,20)])
        return a,mfcc.astype(np.float32)
    results=[]
    controls=[]
    parity=[]
    fixtures=sorted(x['id'] for x in rows if x['arm']=='N')[:3]
    for row in rows:
        started=time.time()
        key=row['id']; dest=out/'items'/f'{key}.npz'
        if dest.exists():
            results.append(read(dest.with_suffix('.json')))
            continue
        assert sha(row['audio_path'])==row['audio_sha256']
        sr,pcm=wavfile.read(row['audio_path'])
        n=row['sync_windows']
        a,mfcc=aud(pcm,n)
        if row['sync_video_cache']:
            z=np.load(row['sync_video_cache'])
            v=z['video1024'][:n]
            if key in fixtures:
                x=np.stack([z['mfcc'][:,4*i:4*i+20] for i in range(min(20,n))])[:,None]
                with torch.inference_mode():
                    rec=net.forward_aud(torch.from_numpy(x).cuda()).cpu().numpy()
                err=float(np.max(np.abs(rec-z['audio1024'][:len(rec)])))
                assert err<=1e-5,err
                parity.append(dict(id=key,official_audio_feature_maxerror=err))
        else:
            scratch=out/'scratch'/key
            scratch.mkdir(parents=True,exist_ok=False)
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',row['sync_crop'],'-threads','1','-f','image2',str(scratch/'%06d.jpg')],check=True)
            files=sorted(scratch.glob('*.jpg'))
            assert len(files)>=n+5
            frames=np.stack([cv2.imread(str(p)) for p in files[:n+4]])
            chunks=[]
            with torch.inference_mode():
                for start in range(0,n,20):
                    xs=np.stack([frames[i:i+5].transpose(3,0,1,2) for i in range(start,min(n,start+20))]).astype(np.float32)
                    chunks.append(net.forward_lip(torch.from_numpy(xs).cuda()).cpu().numpy())
            v=np.concatenate(chunks)
            shutil.rmtree(scratch)
        q=np.arange(row['query_start'],row['query_end_exclusive'])
        av=np.load(row['avs_cache']);co=np.load(row['co_cache'])
        arrays=dict(sync_audio=a,sync_video=v,sync_mfcc=mfcc,avs_audio=av['audio'],avs_video=av['video'],co_audio=co['audio'],co_video=co['video'],queries=q,lags=LAGS)
        arrays['sync_matrix']=sync_matrix(a,v,q)
        arrays['sync_unit_matrix']=sync_matrix(a,v,q,True)
        arrays['avs_matrix']=cosine_matrix(av['audio'],av['video'],q)
        arrays['co_matrix']=cosine_matrix(co['audio'],co['video'],q,True)
        arrays['co_raw_matrix']=cosine_matrix(co['audio'],co['video'],q)
        es={m:endpoints(arrays[m+'_matrix']) for m in ['sync','sync_unit','avs','co','co_raw']}
        native_peaks={m:es[m]['peak_lag_frames'] for m in es}
        item_controls=[]
        if key in fixtures:
            for shift in [-4,4]:
                shifted=np.zeros_like(pcm)
                if shift>0: shifted[shift*640:]=pcm[:-shift*640]
                else: shifted[:shift*640]=pcm[-shift*640:]
                sa,_=aud(shifted,n)
                matrix=sync_matrix(sa,v,q)
                label='sync_control_'+str(shift)
                arrays[label+'_audio']=sa;arrays[label+'_matrix']=matrix
                ep=endpoints(matrix)
                error=ep['peak_lag_frames']-native_peaks['sync']-shift
                item_controls.append(dict(metric='sync',shift_frames=shift,peak=ep['peak_lag_frames'],relative_error_frames=error,passed=abs(error)<=1,matrix_key=label+'_matrix'))
        if row['avs_controls']:
            z=np.load(row['avs_controls'])
            for shift in [-10,10]:
                sa=z[str(shift*640)]
                matrix=cosine_matrix(sa,av['video'],q)
                label='avs_control_'+str(shift)
                arrays[label+'_audio']=sa;arrays[label+'_matrix']=matrix
                ep=endpoints(matrix);error=ep['peak_lag_frames']-native_peaks['avs']-shift
                item_controls.append(dict(metric='avs',shift_frames=shift,peak=ep['peak_lag_frames'],relative_error_frames=error,passed=abs(error)<=1,matrix_key=label+'_matrix'))
        z=np.load(row['co_controls'])
        for shift in [-4,4]:
            sa=z[str(shift)]
            matrix=cosine_matrix(sa,co['video'],q,True)
            label='co_control_'+str(shift)
            arrays[label+'_audio']=sa;arrays[label+'_matrix']=matrix
            ep=endpoints(matrix);error=ep['peak_lag_frames']-native_peaks['co']-shift
            item_controls.append(dict(metric='co',shift_frames=shift,peak=ep['peak_lag_frames'],relative_error_frames=error,passed=abs(error)<=1,matrix_key=label+'_matrix'))
        np.savez_compressed(dest,**arrays)
        norm=lambda x: dict(mean=float(np.linalg.norm(x,axis=1).mean()),sd=float(np.linalg.norm(x,axis=1).std()))
        item=dict(id=key,source=row['sample_id'],arm=row['arm'],strict_reference=row['strict_reference'],query_count=len(q),query_start=int(q[0]),query_end_exclusive=int(q[-1]+1),endpoints=es,controls=item_controls,feature_norms=dict(audio=norm(a),video=norm(v)),npz_sha256=sha(dest),elapsed_s=time.time()-started)
        write(dest.with_suffix('.json'),item)
        results.append(item)
        controls += [dict(id=key,**x) for x in item_controls]
        write(out/'status.json',dict(status='RUNNING',scored=len(results),total=80,last=key,pid=os.getpid(),elapsed_s=time.time()-start_all))
        print('ITEM',len(results),key,round(time.time()-started,2),flush=True)
    write(out/'items.json',results)
    # Include resumed item controls, not only newly forwarded rows.
    controls=[dict(id=x['id'],**c) for x in results for c in x['controls']]
    write(out/'controls.json',dict(rows=controls,summary={m:dict(passed=sum(c['passed'] for c in controls if c['metric']==m),total=sum(c['metric']==m for c in controls),status='PASS' if np.mean([c['passed'] for c in controls if c['metric']==m])>=.8 else 'CONTROL_FAILED') for m in ['sync','avs','co']}))
    write(out/'parity.json',dict(status='PASS',fixtures=parity))
    write(out/'status.json',dict(status='COMPLETE_CURVES',scored=80,elapsed_s=time.time()-start_all))
    analyze(out)


def analyze(out):
    rows=read(out/'items.json')
    natural={r['source']:r for r in rows if r['arm']=='N'}
    pairs=[]
    fields=['best','background','zero','contrast','zero_contrast','search_uplift']
    metrics=['sync','sync_unit','avs','co','co_raw']
    for t in rows:
        if t['arm']=='N':continue
        n=natural[t['source']]
        delta={m:{f:t['endpoints'][m][f]-n['endpoints'][m][f] for f in fields} for m in metrics}
        for m in metrics:
            d=delta[m]
            d['best_contribution']=d['best'];d['background_contribution']=-d['background']
            d['closure_error']=d['contrast']-d['best_contribution']-d['background_contribution']
            assert abs(d['closure_error'])<1e-12
        pairs.append(dict(id=t['id'],source=t['source'],strict_reference=t['strict_reference'],N_id=n['id'],delta=delta))
    summary={}
    for view,ps in [('all20',pairs),('strict12',[p for p in pairs if p['strict_reference']]),('extended8',[p for p in pairs if not p['strict_reference']])]:
        sources=sorted({p['source'] for p in ps})
        summary[view]=dict(sources=len(sources),pairs=len(ps),metrics={})
        for m in metrics:
            stats={}
            for f in fields+['best_contribution','background_contribution']:
                d=np.array([np.mean([p['delta'][m][f] for p in ps if p['source']==s]) for s in sources])
                rng=np.random.default_rng(20261028)
                ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
                stats[f]=dict(mean=float(d.mean()),ci95=ci.tolist(),tts_higher=sum(p['delta'][m][f]>0 for p in ps),natural_higher=sum(p['delta'][m][f]<0 for p in ps),ties=sum(p['delta'][m][f]==0 for p in ps))
            stats['source_means']={s:{f:float(np.mean([p['delta'][m][f] for p in ps if p['source']==s])) for f in fields} for s in sources}
            summary[view]['metrics'][m]=stats
    write(out/'pairs.json',pairs)
    write(out/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run','analyze']);p.add_argument('--repo',type=Path,default=Path('/home/wjj/tts-audio/tts-exp'));args=p.parse_args()
    out=args.repo/'runs'/RUN_NAME
    start_all=time.time()
    if args.stage=='freeze':freeze(args.repo,out)
    elif args.stage=='run':run(args.repo,out)
    else:analyze(out)
