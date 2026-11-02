"""Fixed-video crossed soundtracks from verified independent encoder caches; CPU only."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cogen(a, v):
    n = min(len(a), len(v))
    a, v = a[:n].astype(float), v[:n].astype(float)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    eq = np.maximum(0, (a*v).sum(1)).mean()
    centers = np.arange(15, n-15)
    assert len(centers) > 0
    curve = np.array([np.maximum(0, (a[centers+k]*v[centers]).sum(1)).mean() for k in range(-15, 16)])
    return dict(eq1=float(eq), zero_common=float(curve[15]), max_score=float(curve.max()), lag=int(curve.argmax()-15), frames=n)


def sync(a, v, calc):
    n = min(len(a), len(v))
    a, v = torch.from_numpy(a[:n]), torch.from_numpy(v[:n])
    d = torch.stack(calc(v, a, 15), 0)
    mean = d.mean(0)
    return dict(C=float(mean.median()-mean.min()), D=float(mean.min()), offset=int(15-mean.argmin()), windows=n)


def summarize(rows):
    out = {}
    for video in ['N', 'T']:
        rr = [r for r in rows if r['video'] == video]
        out[video] = {}
        for metric in ['sync_C', 'co_eq1', 'co_max']:
            # Draws and TTS families share a source: average within each source first.
            sources = sorted({r['source'] for r in rr})
            dif = np.array([np.mean([r[metric+'_T']-r[metric+'_N'] for r in rr if r['source']==s]) for s in sources])
            rng = np.random.default_rng(20261026)
            boot = dif[rng.integers(len(dif), size=(10000, len(dif)))].mean(1)
            means = {arm:float(np.mean([np.mean([r[metric+'_'+arm] for r in rr if r['source']==s]) for s in sources])) for arm in ['N','T']}
            out[video][metric] = dict(**means, delta=float(dif.mean()), descriptive_CI95=np.quantile(boot,[.025,.975]).tolist(), T_higher_pairs=sum(r[metric+'_T']>r[metric+'_N'] for r in rr), pairs=len(rr), sources=len(sources), T_higher_source_means=int((dif>0).sum()))
        out[video]['conflicts_eq1'] = sum((r['sync_C_T']-r['sync_C_N'])*(r['co_eq1_T']-r['co_eq1_N'])<0 for r in rr)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    r, out = args.repo, args.out
    out.mkdir(parents=True, exist_ok=False)
    def put(name, value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    cp = r/'runs/cogenav_sync_full20_20261024_attempt01'
    sp = r/'runs/syncnet_audio_replay20_20261024_attempt01'
    sys.path.insert(0,str(r/'third_party/syncnet_python'))
    from SyncNetInstance import calc_pdist
    torch.set_num_threads(1)
    put('protocol.json',dict(design='All measurable source-matched N/T pairs, crossed fixed N and T video, no alignment or time warping; start0 common prefix.',
        support='Per source/draw pair and per evaluator: all four cells use the same minimum cached length. Full native encoder contexts retained; do not pad shorter media or claim same phonetic events.',
        implementation='SyncNet official calc_pdist + torch mean/median/min; CoGenAV official independent encoders cached, Eq1; max lag is custom diagnostic.',
        statistics='source equal-weight, descriptive source bootstrap10000 seed20261026; repeated draws not independent',
        source_sha256=sha(__file__), upstream_sync_sha256=sha(r/'third_party/syncnet_python/SyncNetInstance.py'),
        cogen_scores_sha256=sha(cp/'scores.json'), sync_manifest_sha256=sha(sp/'input_manifest.json')))
    co_rows=read(cp/'scores.json'); cm={x['id']:x for x in co_rows}
    sm={x['id']:x for x in read(sp/'input_manifest.json')['rows']}
    cache={}; checks=[]; pins=[]
    for ident in sorted(cm):
        cpath=cp/'embeddings'/(ident+'.npz'); sdir=sp/'items'/ident
        rec=read(sdir/'replay.json'); seq=sdir/'sequence.npz'
        assert sha(cpath)==cm[ident]['embedding_sha256']
        assert sha(seq)==rec['sequence_sha256'] and rec['pass_replay']
        cz=np.load(cpath); sz=np.load(seq)
        a,v=cz['audio'],cz['video']; sa,sv=sz['audio1024'],sz['video1024']
        ce=cogen(a,v); ss=sync(sa,sv,calc_pdist)
        assert abs(ce['eq1']-cm[ident]['eq1'])<1e-6
        assert abs(ss['C']-rec['replay']['C'])<1e-5
        checks.append(dict(id=ident,eq1_abs_error=abs(ce['eq1']-cm[ident]['eq1']),sync_C_abs_error=abs(ss['C']-rec['replay']['C'])))
        pins.append(dict(id=ident,cogen_sha256=sha(cpath),sync_sha256=sha(seq)))
        cache[ident]=(a,v,sa,sv)
    rows=[]
    for ident in sorted(cm):
        item=cm[ident]
        if item['arm']=='N': continue
        nid=item['sample_id']+'_N_0'
        na,nv,nsa,nsv=cache[nid]; ta,tv,tsa,tsv=cache[ident]
        nc=min(len(na),len(nv),len(ta),len(tv)); ns=min(len(nsa),len(nsv),len(tsa),len(tsv))
        for label,cv,sv in [('N',nv,nsv),('T',tv,tsv)]:
            row=dict(source=item['sample_id'],tts_id=ident,arm=item['arm'],draw=item['draw'],video=label,video_id=nid if label=='N' else ident,co_frames=nc,sync_windows=ns)
            for arm,ca,sa in [('N',na,nsa),('T',ta,tsa)]:
                co=cogen(ca[:nc],cv[:nc]); sc=sync(sa[:ns],sv[:ns],calc_pdist)
                row.update({f'co_eq1_{arm}':co['eq1'],f'co_max_{arm}':co['max_score'],f'co_lag_{arm}':co['lag'],f'sync_C_{arm}':sc['C'],f'sync_D_{arm}':sc['D'],f'sync_offset_{arm}':sc['offset']})
            rows.append(row)
    put('identity_checks.json',checks);put('cache_pins.json',pins);put('rows.json',rows)
    summary=summarize(rows);put('summary.json',summary)
    lines=['# 固定视频交叉音轨：SyncNet 与 CoGenAV','',f'{len(rows)//2} 个自然–TTS配对，20来源；每配对固定两种视频，各配自然/TTS。CPU缓存重算，无新视频、无训练。','',
        '|固定视频|指标|自然音轨|TTS音轨|TTS−自然|TTS更高配对|','|---|---|---:|---:|---:|---:|']
    for video, vals in summary.items():
        for metric in ['sync_C','co_eq1','co_max']:
            z=vals[metric]; digits=3 if metric=='sync_C' else 6
            lines.append(f"|{video}|{metric}|{z['N']:.{digits}f}|{z['T']:.{digits}f}|{z['delta']:+.{digits}f}|{z['T_higher_pairs']}/{z['pairs']}|")
    lines += ['', 'N视频由自然音频驱动；T视频由对应TTS驱动。两组固定视频分别检验换音轨的影响。每配对四格使用共同起点/有效前缀，保持原生语速，不做DTW。长短音频的尾段排除，故分数不能等同全长原生基线。', '',
        'SyncNet是官方编码器缓存与官方评分函数；CoGenAV是官方编码器缓存加论文Eq1自写计算，最佳时移为自定义诊断。缓存完整上下文不变。未重新封装媒体或重跑全流程；已有缓存的时钟/编码资格以原run为准。两指标尺度不通用，分数不是概率，无公认高同步阈值；交叉音轨包含语速和停顿不匹配，不能单独证明TTS声学偏好或真实口型优劣。','',
        '全部138原生缓存身份重放通过；逐缓存哈希、逐格结果、来源等权描述性区间见JSON。无独立人评。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
