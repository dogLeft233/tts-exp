"""Frozen cached event geometry: normalization-state x N/T-source factorial.

No models, waveform operations, rendering, lag search, or support selection.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tts_level_event_cross_20260927'
OLD = ROOT / 'runs/tts_event_cross_20260926'
CAL = ROOT / 'runs/tts_static_lag_calibration_20260926/evaluation'
VIS = ROOT / 'runs/tts_prepost_mel_20260926'
AUD = ROOT / 'runs/tts_pcm_residual_20260926'
FIX = ROOT / 'runs/tts_fixed_level_generation_cross_20260927'
GEOMS = ['raw', 'unit']
METRICS = ['positive', 'negative', 'margin', 'rank']
SOURCES = ['NN', 'NT', 'TN', 'TT']
STATES = [('raw', 'raw'), ('FIXED', 'raw'), ('raw', 'FIXED'), ('FIXED', 'FIXED'), ('LEVEL', 'LEVEL')]
STATE_NAMES = ['RR', 'FR', 'RF', 'FF', 'LL']
CELLS = [dict(name=f'{h}_{s}', video_state=v, audio_state=a, video_source=s[0], audio_source=s[1])
         for h, (v, a) in zip(STATE_NAMES, STATES) for s in SOURCES]
CONTRASTS = {'visual_at_N_audio': {'TN': 1, 'NN': -1},
             'audio_at_N_visual': {'NT': 1, 'NN': -1},
             'interaction': {'TT': 1, 'NN': 1, 'TN': -1, 'NT': -1},
             'visual_at_T_audio': {'TT': 1, 'NT': -1},
             'audio_at_T_visual': {'TT': 1, 'TN': -1},
             'diagonal': {'TT': 1, 'NN': -1}}
STATE_CONTRASTS = {'G': {'FR': 1, 'RR': -1}, 'E': {'RF': 1, 'RR': -1},
                   'I': {'FF': 1, 'FR': -1, 'RF': -1, 'RR': 1},
                   'total': {'FF': 1, 'RR': -1}}
FLOOR = int(4.5 * 2**30)
CAP = 16 * 2**20

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def resource(extra=0):
    assert shutil.disk_usage(ROOT).free >= FLOOR + extra, 'disk floor'
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    assert used + extra <= CAP, ('run budget', used, extra)

def write(p, obj):
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    resource(len(data.encode()))
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    assert not Path(p).exists(), f'no overwrite: {p}'
    Path(p).write_text(data)

def save(p, **arrays):
    resource(sum(a.nbytes for a in arrays.values()) + 4096)
    assert not Path(p).exists()
    np.savez_compressed(p, **arrays)

def stat(values, speakers):
    # Same arithmetic and fresh seed as historical cluster(), copied explicitly.
    x = np.asarray(values, dtype=float)
    names = sorted(set(speakers))
    sv = np.array([x[np.array(speakers) == s].mean() for s in names])
    counts = np.array([speakers.count(s) for s in names])
    ix = np.random.default_rng(20260926).integers(len(names), size=(20000, len(names)))
    eq = sv[ix].mean(1)
    weighted = (sv[ix] * counts[ix]).sum(1) / counts[ix].sum(1)
    return {'n': len(x), 'speakers': len(names), 'speaker_mean': float(sv.mean()),
            'speaker_ci95': np.quantile(eq, [.025, .975]).tolist(),
            'speaker_ci99': np.quantile(eq, [.005, .995]).tolist(),
            'utterance_mean': float(x.mean()),
            'utterance_cluster_ci95': np.quantile(weighted, [.025, .975]).tolist(),
            'positive': int((x > 0).sum()), 'speaker_positive': int((sv > 0).sum()),
            'per_speaker': dict(zip(names, sv.tolist()))}

def freeze():
    assert not OUT.exists(), 'new run only'
    resource()
    support = read(CAL / 'support.json')
    rows = [r for r in support['rows'] if r['eligible']]
    assert len(rows) == 32 and sum(len(r['queries']) for r in rows) == 1411
    assert len({r['speaker'] for r in rows}) == 13
    assert read(CAL / 'protocol.json')['fixed_audio_minus_visual_index'] == 3
    oldp = read(VIS / 'protocol.json')
    newp = read(FIX / 'protocol.json')
    assert next(i for i in oldp['images'] if i['id'] == '3') == newp['image']
    assert [i['id'] for i in oldp['images']] == ['3', '6', '9']
    nr = {r['id']: r for r in read(FIX / 'support.json')}
    op = {r['id']: r for r in oldp['rows']}
    np_ = {r['id']: r for r in newp['rows']}
    inputs = {}
    def bind(p, expected=None):
        p = Path(p)
        h = sha(p)
        assert expected is None or h == expected, str(p)
        inputs[str(p)] = h
    for p in [CAL/'support.json', CAL/'protocol.json', CAL/'validation.json',
              OLD/'protocol.json', OLD/'evaluation/scores.json', OLD/'evaluation/analysis.json',
              OLD/'evaluation/validation.json', VIS/'protocol.json', FIX/'protocol.json',
              FIX/'support.json', FIX/'feature_seal_evaluation.json']:
        bind(p)
    seal = read(FIX/'feature_seal_evaluation.json')
    boundary = []
    spans = []
    start = 0
    prefix_max = {'audio': 0., 'visual': 0.}
    videos_exact = 0
    for r in rows:
        sid = r['id']
        assert nr[sid]['split'] == 'evaluation' and nr[sid]['guard20_eligible']
        assert op[sid]['audio'] == np_[sid]['audio']
        for a in 'NT':
            bind(op[sid]['audio'][a]['path'], op[sid]['audio'][a]['sha256'])
        af = AUD/'features'/sid/'features.npz'
        bind(af, support['assets'][str(af)])
        with np.load(af) as f:
            olda = {a: f['a_'+a+'_source'] for a in 'NT'}
        for image in ['3', '6', '9']:
            vf = VIS/'scores'/image/(sid+'_visual.npz')
            bind(vf, support['assets'][str(vf)])
        with np.load(VIS/'scores/3'/(sid+'_visual.npz')) as f:
            oldv = {a: f[a] for a in 'NT'}
        oldvideo = read(VIS/'scores/3'/(sid+'.json'))
        bind(VIS/'scores/3'/(sid+'.json'))
        for a in 'NT':
            ji = np.array([r['nodes'][q['node']]['j'][a] for q in r['queries']])
            dn = np.array([r['nodes'][d]['j'][a] for q in r['queries'] for d in q['donors']])
            for c in ['raw', 'identity', 'LEVEL', 'FIXED']:
                fp = FIX/'features'/sid/a/(c+'.npz')
                bind(fp, seal[str(fp)])
                bind(fp.with_suffix('.json'), seal[str(fp.with_suffix('.json'))])
                with np.load(fp) as f:
                    aa, vv = f['audio'], f['visual']
                assert np.isfinite(aa).all() and np.isfinite(vv).all()
                assert (np.linalg.norm(aa, axis=1) > 0).all() and (np.linalg.norm(vv, axis=1) > 0).all()
                assert 0 <= ji.min()-3 and ji.max()-3 < len(vv)
                assert 0 <= min(ji.min(), dn.min()) and max(ji.max(), dn.max()) < len(aa)
                if c == 'raw':
                    for key, x, y in [('audio', olda[a], aa), ('visual', oldv[a], vv)]:
                        n = min(len(x), len(y))
                        err = float(np.max(abs(x[:n] - y[:n])))
                        prefix_max[key] = max(prefix_max[key], err)
                        assert err == 0
                    assert read(fp.with_suffix('.json'))['video']['sha256'] == oldvideo['videos'][a]['sha256']
                    videos_exact += 1
            L = nr[sid]['arms'][a]['raw']['L']
            for qi, j in enumerate(ji):
                if not 20 <= j-3 < L-20:
                    boundary.append(dict(id=sid, arm=a, query=qi, visual_index=int(j-3), joint_L=L, legacy_length=r['lengths'][a]))
        spans.append(dict(id=sid, speaker=r['speaker'], start=start, stop=start+len(r['queries'])))
        start += len(r['queries'])
    OUT.mkdir()
    write(OUT/'indices.json', {'rows': rows, 'spans': spans, 'legacy_vs_joint_guard20': boundary})
    write(OUT/'inputs.json', inputs)
    write(OUT/'feasibility.json', {'status':'PASS_no_new_scoring', 'raw_prefix_max':prefix_max,
          'raw_video_hashes_exact':videos_exact, 'clips':32, 'speakers':13, 'queries':1411,
          'donor_assignments':20647, 'legacy_queries_outside_new_guard20':len(boundary),
          'all_indices_valid':True, 'input_files_bound':len(inputs), 'free_bytes':shutil.disk_usage(ROOT).free})
    p = {'status':'frozen_before_new_distance', 'created_epoch':time.time(),
         'question':'Residual event margin after level control: state pathways and source pairing dependence',
         'support':'exact legacy32/13/1411 k3; image3 only; no new joint guard20 restriction; no donor reselection',
         'bridge_images':['3','6','9'], 'cells':CELLS, 'geometries':GEOMS, 'metrics':METRICS,
         'source_contrasts':CONTRASTS, 'state_contrasts':STATE_CONTRASTS,
         'arithmetic':'raw float32 features; unit normalize each original float32 row before transform; float64 distance sqrt(sum((V-A+1e-6)^2)); negative donor mean; margin=negative-positive; rank mean(neg>pos)+.5*mean(neg==pos)',
         'indexing':'V at legacy j[video_source]-3; positive A at legacy j[audio_source]; every negative uses same cell audio_state and frozen donor j[audio_source]; no warp/interpolation',
         'aggregation':'query equal->clip; bridge3 images equal within clip; clip equal within speaker; speaker equal. 20000 bootstrap seed20260926 99CI each comparison; no FWER.',
         'primary':['FF native TT-NN margin','FF-RR diagonal gap change and state G/E/I','FF source simple visual_at_N_audio/audio_at_N_visual/interaction'],
         'secondary':'rank; positive/negative components; unit separately; all opposite simple effects; LL same-state source4 cells and LL-RR',
         'bridge_gates':{'query_max':1e-9,'all_historical_statistic_max':1e-9,'new_raw_image3_query_max':1e-9,'raw_feature_prefix_max':0},
         'limits':'Historical exploratory annotation-coordinate representation diagnostic; not physical alignment, natural-track replacement, mouth truth, mediation or fraction of official71 Sync-C residual. LEVEL pairwise and FIXED cal-only are different operations. legacy32 support and single image estimand differ from published three-image and71 official scores.',
         'resources':{'cpu_threads':2,'ram_target_bytes':100*2**20,'persistent_cap_bytes':CAP,'floor_bytes':FLOOR,'no_GPU_forward_media':True},
         'indices_sha256':sha(OUT/'indices.json'),'inputs_sha256':sha(OUT/'inputs.json'),
         'code':{str(Path(__file__).resolve()):sha(__file__)}}
    write(OUT/'protocol.json',p)
    write(OUT/'seal.json', {n:sha(OUT/n) for n in ['protocol.json','indices.json','inputs.json','feasibility.json']})
    print(json.dumps({'status':'FROZEN','protocol_sha256':sha(OUT/'protocol.json')}), flush=True)

def locked():
    p = read(OUT/'protocol.json')
    for n,h in read(OUT/'seal.json').items():
        assert sha(OUT/n)==h,n
    for path,h in {**p['code'], **read(OUT/'inputs.json')}.items():
        assert sha(path)==h,path
    return p,read(OUT/'indices.json')

def geometry(row, arrays, cells):
    result=np.zeros((len(row['queries']),2,len(cells),4),np.float64)
    for gi,g in enumerate(GEOMS):
        f={key:(x if g=='raw' else x/np.linalg.norm(x,axis=1)[:,None]) for key,x in arrays.items()}
        for ci,c in enumerate(cells):
            a=f['A',c['audio_state'],c['audio_source']]
            v=f['V',c['video_state'],c['video_source']]
            for qi,q in enumerate(row['queries']):
                node=row['nodes'][q['node']]
                x=v[node['j'][c['video_source']]-3].astype(np.float64)
                j=node['j'][c['audio_source']]
                pos=float(np.sqrt(np.sum((x-a[j].astype(np.float64)+1e-6)**2)))
                ix=[row['nodes'][d]['j'][c['audio_source']] for d in q['donors']]
                neg=np.sqrt(np.sum((x-a[ix].astype(np.float64)+1e-6)**2,axis=-1))
                result[qi,gi,ci]=[pos,float(neg.mean()),float(neg.mean()-pos),float(np.mean((neg>pos)+.5*(neg==pos)))]
    assert np.isfinite(result).all()
    return result

def old_cells():
    return [dict(video_state='raw',audio_state='raw',video_source=s[0],audio_source=s[1]) for s in SOURCES]

def clipmeans(values,spans):
    return np.stack([values[r['start']:r['stop']].mean(0) for r in spans])

def bridge():
    _,idx=locked()
    old={(r['id'],r['image']):r for r in read(OLD/'evaluation/scores.json')}
    result=np.zeros((3,1411,2,4,4),np.float64)
    errors=0.
    for r,span in zip(idx['rows'],idx['spans']):
        sid=r['id']
        with np.load(AUD/'features'/sid/'features.npz') as f:
            aa={('A','raw',a):f['a_'+a+'_source'] for a in 'NT'}
        for im,image in enumerate(['3','6','9']):
            with np.load(VIS/'scores'/image/(sid+'_visual.npz')) as f:
                arrays={**aa,**{('V','raw',a):f[a] for a in 'NT'}}
            z=geometry(r,arrays,old_cells())
            expected=np.array([[[[q['geometry'][g][c][m] for m in METRICS] for c in SOURCES] for g in GEOMS] for q in old[sid,image]['queries']])
            errors=max(errors,float(np.max(abs(z-expected))))
            result[im,span['start']:span['stop']]=z
    assert errors<=1e-9,('old query bridge',errors)
    speakers=[r['speaker'] for r in idx['spans']]
    # Match historical order: query mean per image; cell contrast per image;
    # then three-image mean per clip; no pooled-query alternate weighting.
    cm=np.stack([clipmeans(z,idx['spans']) for z in result])
    historical=read(OLD/'evaluation/analysis.json')
    replay={};staterr=0.
    image3={}
    for gi,g in enumerate(GEOMS):
        for mi,m in enumerate(METRICS):
            for name,w in {**{'level_'+c:{c:1} for c in SOURCES},**CONTRASTS}.items():
                perim=sum(cm[:,:,gi,SOURCES.index(c),mi]*v for c,v in w.items())
                key=f'{g}_{m}_{name}'
                replay[key]=stat(perim.mean(0),speakers)
                image3[key]=stat(perim[0],speakers)
                for k,v in replay[key].items():
                    if k=='per_speaker':
                        e=max(abs(v[s]-historical[key][k][s]) for s in v)
                    else:e=float(np.max(abs(np.asarray(v)-np.asarray(historical[key][k]))))
                    staterr=max(staterr,e)
    assert staterr<=1e-9,('old statistic bridge',staterr)
    rawerr=0.
    for r,span in zip(idx['rows'],idx['spans']):
        arrays={}
        for a in 'NT':
            with np.load(FIX/'features'/r['id']/a/'raw.npz') as f:
                arrays['A','raw',a]=f['audio'];arrays['V','raw',a]=f['visual']
        z=geometry(r,arrays,old_cells())
        rawerr=max(rawerr,float(np.max(abs(z-result[0,span['start']:span['stop']]))))
    assert rawerr<=1e-9,('new raw bridge',rawerr)
    save(OUT/'bridge_queries.npz',metrics=result)
    write(OUT/'bridge_summary.json',{'historical_three_images':replay,'image3_RAW':image3})
    write(OUT/'bridge_validation.json',{'status':'PASS','old_query_max':errors,'old_statistic_max':staterr,
          'new_raw_image3_query_max':rawerr,'query_geometries':33864,'protocol_sha256':sha(OUT/'protocol.json')})
    print('BRIDGE_PASS',errors,staterr,rawerr,flush=True)

def execute():
    _,idx=locked()
    assert read(OUT/'review.json')['status']=='PASS','independent prior review required'
    assert read(OUT/'review.json')['protocol_sha256']==sha(OUT/'protocol.json')
    if not (OUT/'bridge_validation.json').exists():bridge()
    assert read(OUT/'bridge_validation.json')['status']=='PASS'
    result=np.zeros((1411,2,20,4),np.float64)
    for r,span in zip(idx['rows'],idx['spans']):
        resource(2*2**20)
        arrays={}
        for a in 'NT':
            for state in ['raw','FIXED','LEVEL']:
                with np.load(FIX/'features'/r['id']/a/(state+'.npz')) as f:
                    arrays['A',state,a]=f['audio'];arrays['V',state,a]=f['visual']
        result[span['start']:span['stop']]=geometry(r,arrays,CELLS)
    with np.load(OUT/'bridge_queries.npz') as f:
        e=float(np.max(abs(result[:,:,:4]-f['metrics'][0])))
    assert e<=1e-9
    save(OUT/'query_metrics.npz',metrics=result)
    cm=clipmeans(result,idx['spans'])
    save(OUT/'clip_metrics.npz',metrics=cm)
    speakers=[r['speaker'] for r in idx['spans']]
    summary={};effects={};closure=0.
    for gi,g in enumerate(GEOMS):
        for mi,m in enumerate(METRICS):
            val={c['name']:cm[:,gi,i,mi] for i,c in enumerate(CELLS)}
            series={'cell/'+k:v for k,v in val.items()}
            for h in STATE_NAMES:
                for name,w in CONTRASTS.items():
                    series[f'source/{h}/{name}']=sum(val[f'{h}_{s}']*n for s,n in w.items())
            for s in SOURCES:
                for name,w in STATE_CONTRASTS.items():
                    series[f'state/{s}/{name}']=sum(val[f'{h}_{s}']*n for h,n in w.items())
                series[f'LEVEL_minus_RAW/{s}']=val[f'LL_{s}']-val[f'RR_{s}']
                closure=max(closure,float(np.max(abs(sum(series[f'state/{s}/{n}'] for n in ['G','E','I'])-series[f'state/{s}/total']))))
            for name in STATE_CONTRASTS:
                series[f'state_gap/{name}']=series[f'state/TT/{name}']-series[f'state/NN/{name}']
            series['LEVEL_minus_RAW/diagonal']=series['source/LL/diagonal']-series['source/RR/diagonal']
            for key,v in series.items():
                full=f'{g}/{m}/{key}'
                effects[full]=v
                summary[full]=stat(v,speakers)
    assert closure<1e-12
    # Every endpoint shares an exact per-query negative-positive margin identity.
    margin_error=float(np.max(abs(result[:,:,:,1]-result[:,:,:,0]-result[:,:,:,2])))
    assert margin_error==0
    save(OUT/'clip_effects.npz',**effects)
    write(OUT/'summary.json',summary)
    write(OUT/'execution.json',{'status':'SCORED_PENDING_INDEPENDENT','protocol_sha256':sha(OUT/'protocol.json'),
          'raw_bridge_max':e,'state_closure_max':closure,'margin_closure_max':margin_error,
          'shape':list(result.shape),'summary_endpoints':len(summary),'free_bytes':shutil.disk_usage(ROOT).free})
    print('SCORED',len(summary),closure,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','bridge','execute'])
    globals()[p.parse_args().stage]()
