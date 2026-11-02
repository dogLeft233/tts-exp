"""Known synthetic clock benchmark; no natural/TTS ground-truth claim."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SR = 16000
HOP = 640
GUARD = .08


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def select(rows):
    selected = []
    for sid in sorted({r['sample_id'] for r in rows}):
        nr = next(r for r in rows if r['sample_id'] == sid and r['arm'] == 'N')
        candidates = [r for arm in ['L06', 'L17'] for r in rows
                      if r['sample_id'] == sid and r['arm'] == arm and r['draw'] == 1
                      and r['qualified'] and r['score_float32'] > nr['score_float32']]
        if candidates:
            selected.extend([nr, candidates[0]])
        if len(selected) == 12:
            break
    return selected


def landmarks(x):
    # Pick low-energy split nearest one third within +/- .15 sec, from originals only.
    central = int(len(x) / 3)
    candidates = np.arange(max(512, central-2400), min(len(x)-512, central+2401), 160)
    rms = [np.mean(x[t-160:t+160] ** 2) for t in candidates]
    first = int(candidates[np.argmin(rms)])
    return [0, first, int(2 * len(x) / 3), len(x)]


def transform(x, cuts, condition):
    import librosa
    rates = [.8, 1.25, 1.] if condition in ['stretch', 'combined'] else [1., 1., 1.]
    pause = int(.32 * SR) if condition in ['pause', 'combined'] else 0
    parts, segments, position = [], [], 0
    for k, (a, b, rate) in enumerate(zip(cuts[:-1], cuts[1:], rates)):
        part = x[a:b].copy()
        if rate != 1:
            part = librosa.effects.time_stretch(part, rate=rate, n_fft=512, hop_length=128)
        segments.append([int(a), int(b), position, position+len(part)])
        parts.append(part)
        position += len(part)
        if k == 0 and pause:
            parts.append(np.zeros(pause, dtype=np.float32))
            position += pause
    return np.concatenate(parts), np.asarray(segments, dtype=np.int64)


def oracle(times, segments):
    out = np.full(len(times), np.nan)
    for a, b, c, d in segments:
        mask = (times >= a / SR) & (times < b / SR)
        out[mask] = c / SR + (times[mask] - a / SR) * (d-c) / (b-a)
    return out


def energy(x, times):
    return np.asarray([np.sqrt(np.mean(x[max(0, int(t*SR)-320):min(len(x), int(t*SR)+320)]**2)) for t in times])


def distance(x, y):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return np.clip(1 - x @ y.T, 0, 2)


def dtw(cost):
    from numba import njit
    return _dp_impl(cost)


def _dp_python(cost):
    n, m = cost.shape
    acc = np.full((n+1, m+1), np.inf)
    back = np.full((n, m), -1, dtype=np.int8)
    acc[0, 0] = 0
    for i in range(n):
        for j in range(m):
            if abs(i / max(n-1, 1) - j / max(m-1, 1)) > .25:
                continue
            options = (acc[i, j], acc[i, j+1]+.05, acc[i+1, j]+.05)
            step = int(np.argmin(np.asarray(options)))
            acc[i+1, j+1] = options[step] + cost[i, j]
            back[i, j] = step
    if not np.isfinite(acc[n, m]):
        raise ValueError('No finite DTW path')
    path = []
    i, j = n-1, m-1
    while i >= 0 and j >= 0:
        path.append((i, j))
        step = back[i, j]
        if step == 0:
            i, j = i-1, j-1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            raise ValueError('Broken traceback')
    return np.asarray(path[::-1], dtype=np.int64)


_dp_impl = _dp_python


def mapping(path, n, target_times):
    return np.asarray([np.median(target_times[path[path[:, 0] == i, 1]]) for i in range(n)])


def statistics(errors, mask):
    e = errors[mask] * 1000
    return {'frames': int(len(e)), 'mae_ms': float(e.mean()), 'median_ms': float(np.median(e)),
            'p95_ms': float(np.quantile(e, .95)), 'over80_fraction': float(np.mean(e > 80)),
            'within40_fraction': float(np.mean(e <= 40))}


def prepare(root, manifest):
    import soundfile as sf
    rows = select(json.loads(Path(manifest).read_text()))
    protocol = {'selection': 'first6 lexical sources with qualified d1 T Sync-C>N; L06 preferred else L17',
                'conditions': ['identity', 'stretch', 'pause', 'combined'],
                'stretch': 'librosa phase vocoder n_fft512 hop128; per-segment rates .8,1.25,1',
                'split': 'low40ms-energy center near original third +-150ms on10ms grid, then two-thirds',
                'pause': 'insert320ms zero samples after first segment',
                'oracle': 'affine segment sample-clock mapping; not exact phonetic-event truth after phase vocoder',
                'guard_s': GUARD, 'evaluation_grid_s': .04,
                'features': 'CoGenAV official audio-only pre-upsampler FP32 raw cosine; librosa80 logMel and13MFCC per-utterance dimension CMVN then cosine',
                'frontend': '16kHz, centered400sample Hann window/400FFT,160sample hop, fmin0 fmax8000 Slaney mel; sample every4frames',
                'dtw': 'same closed endpoints, normalized-index band .25, diagonal/horizontal/vertical; non-diagonal additive .05; tie diagonal first; no score-tuning',
                'primary': 'forward original->transformed MAE on original energy>-40dB relative max; exclude80ms near every interior split; source equally weighted, arm and nonidentityconditions averaged',
                'secondary': 'allvalid, p95, >80ms, within40ms; global-linear baseline; reverse diagnostic; discretized oracle floor',
                'silence': 'inserted pause has no original match; exclude from reverse accuracy, report forward pause match fraction',
                'weight_sha256': '90f8c5165282c190c81c23d8fe92db2aa700862e021b1312fb96779caf9cfe61',
                'seed': 20261025, 'status': 'exploratory synthetic-time benchmark selected on prior Sync-C; not population or naturalTTS accuracy',
                'input_manifest_sha256': sha(manifest)}
    root.mkdir(parents=True, exist_ok=False)
    dump(root/'protocol.json', protocol)
    dump(root/'selected_inputs.json', rows)
    records = []
    for row in rows:
        assert sha(row['audio_path']) == row['audio_sha256']
        x, sr = sf.read(row['audio_path'], dtype='float32')
        assert sr == SR and x.ndim == 1
        cuts = landmarks(x)
        for condition in protocol['conditions']:
            y, segments = transform(x, cuts, condition)
            assert len(y) < 30*SR
            target = root/'signals'/row['id']/condition
            target.mkdir(parents=True)
            sf.write(target/'audio.wav', y, SR, subtype='FLOAT')
            np.savez_compressed(target/'oracle.npz', segments=segments, cuts=np.asarray(cuts))
            records.append({'id':row['id'], 'sample_id':row['sample_id'], 'arm':row['arm'],
                            'condition':condition, 'samples':len(y), 'audio_sha256':sha(target/'audio.wav')})
    dump(root/'signals.json', records)
    dump(root/'prepared.json', {'protocol_sha256':sha(root/'protocol.json'), 'signals':len(records),
                               'sources':len(rows)//2, 'source_sha256':sha(__file__)})
    shutil.copy2(__file__,root/'source_frozen.py')
    print(json.dumps({'prepared':len(records), 'ids':[r['id'] for r in rows], 'protocol_sha256':sha(root/'protocol.json')}),flush=True)


def features(root, modelroot):
    import librosa
    import soundfile as sf
    import torch
    from transformers import WhisperFeatureExtractor
    from omegaconf import OmegaConf
    from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    sys.path.insert(0,str(modelroot/'upstream'))
    from models.cogenav import CoGenAV
    cfg = OmegaConf.merge(OmegaConf.structured(Wav2Vec2Config()),OmegaConf.load(modelroot/'upstream/config/base.yaml'))
    OmegaConf.save(cfg,root/'base_resolved.yaml')
    weight = modelroot/'weights/base_cogen_mirror.pt'
    assert sha(weight) == json.loads((root/'protocol.json').read_text())['weight_sha256']
    model=CoGenAV(str(root/'base_resolved.yaml'),model_tensor=None)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    model.eval().cuda()
    fe=WhisperFeatureExtractor()
    for number, row in enumerate(json.loads((root/'signals.json').read_text())):
        directory=root/'signals'/row['id']/row['condition']
        x,sr=sf.read(directory/'audio.wav',dtype='float32')
        assert sha(directory/'audio.wav')==row['audio_sha256']
        mel=torch.from_numpy(fe(x,sampling_rate=SR).input_features[0])[None].cuda()
        n=len(x)//HOP
        with torch.inference_mode():
            a=model(None,mel,use_upsampler=False)[0,:n].cpu().numpy()
        power=librosa.feature.melspectrogram(y=x,sr=SR,n_fft=400,hop_length=160,win_length=400,n_mels=80,fmin=0,fmax=8000,center=True,power=2)
        lm=librosa.power_to_db(power,ref=1.,top_db=None)
        mf=librosa.feature.mfcc(S=lm,n_mfcc=13)
        raw={'cogenav':a,'mel':lm[:,::4][:,:n].T,'mfcc':mf[:,::4][:,:n].T}
        out={}
        for name,v in raw.items():
            if name!='cogenav':
                v=(v-v.mean(axis=0,keepdims=True))/np.maximum(v.std(axis=0,keepdims=True),1e-5)
            out[name]=v.astype(np.float32)
        np.savez_compressed(directory/'features.npz',times=np.arange(n)*.04,**out)
        print(json.dumps({'features_done':number+1,'id':row['id'],'condition':row['condition'],'frames':n}),flush=True)
    dump(root/'features_complete.json',{'forward_calls':number+1,'torch':torch.__version__,'librosa':librosa.__version__,'strict_load':True})


def score(root):
    global _dp_impl
    import soundfile as sf
    from numba import njit
    _dp_impl=njit(_dp_python)
    result=[]
    rows=json.loads((root/'selected_inputs.json').read_text())
    for row in rows:
        original=root/'signals'/row['id']/'identity'
        baseline=np.load(original/'features.npz')
        x,_=sf.read(original/'audio.wav',dtype='float32')
        times=baseline['times']
        e=energy(x,times)
        speech=e>max(e.max()*.01,1e-8)
        for condition in ['identity','stretch','pause','combined']:
            directory=root/'signals'/row['id']/condition
            target=np.load(directory/'features.npz')
            segments=np.load(directory/'oracle.npz')['segments']
            truth=oracle(times,segments)
            valid=np.isfinite(truth)&(truth<=target['times'][-1])
            if condition!='identity':
                for cut in segments[:-1,1]/SR:
                    valid &= abs(times-cut)>GUARD
            active=valid&speech
            floor=abs(np.round(truth/.04)*.04-truth)
            linear=times*(row_duration:=len(sf.read(directory/'audio.wav')[0])/len(x))
            predictions={'linear':linear,'oracle_grid':np.round(truth/.04)*.04}
            paths={}
            for name in ['cogenav','mfcc','mel']:
                path=dtw(distance(baseline[name],target[name]))
                paths[name]=path
                predictions[name]=mapping(path,len(times),target['times'])
            save={'times':times,'target_times':target['times'],'truth':truth,'valid':valid,'speech':speech,'active':active,'segments':segments,'floor_error':floor}
            for name,pred in predictions.items():
                err=abs(pred-truth)
                save[name+'_prediction']=pred
                if name in paths:
                    save[name+'_path']=paths[name]
                pauses=[(segments[k,3]/SR,segments[k+1,2]/SR) for k in range(len(segments)-1) if segments[k+1,2]>segments[k,3]]
                badpause=np.zeros(len(pred),bool)
                for left,right in pauses:
                    badpause|=(pred>left)&(pred<right)
                result.append({'id':row['id'],'sample_id':row['sample_id'],'arm':row['arm'],'condition':condition,'method':name,
                               'speech':statistics(err,active),'allvalid':statistics(err,valid),
                               'forward_into_pause_fraction':float(badpause[active].mean())})
            np.savez_compressed(directory/'alignment.npz',**save)
        print(json.dumps({'scored':row['id']}),flush=True)
    # Actual natural/TTS paths lack ground truth, so no accuracy ranking.
    actual=[]
    for nr,tr in zip(rows[::2],rows[1::2]):
        n=np.load(root/'signals'/nr['id']/'identity/features.npz')
        t=np.load(root/'signals'/tr['id']/'identity/features.npz')
        d=root/'actual_pairs'/nr['sample_id'];d.mkdir(parents=True)
        out={'natural_times':n['times'],'tts_times':t['times']}
        for name in ['cogenav','mfcc','mel']:
            path=dtw(distance(n[name],t[name]));pred=mapping(path,len(n['times']),t['times'])
            out[name+'_path']=path;out[name+'_prediction']=pred
        np.savez_compressed(d/'alignment.npz',**out)
        disagreement={a+'_'+b:float(np.mean(abs(out[a+'_prediction']-out[b+'_prediction']))*1000)
                      for a,b in [('cogenav','mfcc'),('cogenav','mel'),('mfcc','mel')]}
        actual.append({'sample_id':nr['sample_id'],'tts_arm':tr['arm'],'delta_sync_c':tr['score_float32']-nr['score_float32'],
                       'pairwise_path_disagreement_mae_ms':disagreement,'status':'NO_GROUND_TRUTH_EXPLORATORY_ONLY'})
    dump(root/'metrics.json',result);dump(root/'actual_pairs.json',actual)
    summaries=[]
    for condition in ['identity','stretch','pause','combined','nonidentity_mean']:
        for name in ['cogenav','mfcc','mel','linear','oracle_grid']:
            rr=[r for r in result if r['method']==name and (r['condition']==condition or (condition=='nonidentity_mean' and r['condition']!='identity'))]
            sourcevalues=[]
            for sid in sorted({r['sample_id'] for r in rr}):
                sourcevalues.append(np.mean([r['speech']['mae_ms'] for r in rr if r['sample_id']==sid]))
            summaries.append({'condition':condition,'method':name,'source_equal_mae_ms':float(np.mean(sourcevalues)),
                              'source_mae_ms':sourcevalues,'mean_cell_p95_ms':float(np.mean([r['speech']['p95_ms'] for r in rr])),
                              'mean_cell_over80_fraction':float(np.mean([r['speech']['over80_fraction'] for r in rr]))})
    dump(root/'summary.json',summaries)
    print(json.dumps(summaries),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','score']);p.add_argument('--run',required=True)
    p.add_argument('--manifest');p.add_argument('--model-root');args=p.parse_args();root=Path(args.run)
    if args.stage=='prepare':prepare(root,args.manifest)
    elif args.stage=='features':features(root,Path(args.model_root))
    else:score(root)


if __name__=='__main__':main()
