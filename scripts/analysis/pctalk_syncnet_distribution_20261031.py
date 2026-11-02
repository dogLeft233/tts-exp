"""Descriptive source-level Sync-C distribution and existing-video selection."""
from pathlib import Path
import csv
import hashlib
import html
import json
import os
import numpy as np

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'results/pctalk/chinese20_rms24_nine_20261031_attempt01'
PARENT=REPO/'results/pctalk/chinese20_rms24_20261031_attempt01'
OUT=ROOT/'syncnet_distribution'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def distribution(values):
    a=np.asarray(values,dtype=float)
    assert a.ndim==1 and len(a)>1 and np.isfinite(a).all()
    edges=[-np.inf,-1,0,.5,1,2,np.inf]
    counts=np.histogram(a,bins=edges)[0]
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),sample_sd=float(a.std(ddof=1)),
        min=float(a.min()),q10=float(np.quantile(a,.1)),q25=float(np.quantile(a,.25)),
        q75=float(np.quantile(a,.75)),q90=float(np.quantile(a,.9)),max=float(a.max()),
        positive=int((a>0).sum()),negative=int((a<0).sum()),ties=int((a==0).sum()),
        bins=[dict(interval=label,count=int(n)) for label,n in zip(['(-inf,-1)','[-1,0)','[0,.5)','[.5,1)','[1,2)','[2,inf)'],counts)])

def pair_rows(rows):
    sources=[];draws=[]
    for source in sorted({x['source'] for x in rows}):
        group=[x for x in rows if x['source']==source]
        n=[x for x in group if x['arm']=='N'];ts=[x for x in group if x['arm']!='N']
        assert len(n)==1 and len(ts)==3
        natural=float(n[0]['SyncNet']);tts=float(np.mean([t['SyncNet'] for t in ts]))
        sources.append(dict(source=source,speaker=n[0]['speaker'],strict_reference=n[0]['strict_reference'],N=natural,T=tts,delta=tts-natural))
        for t in ts:
            draws.append(dict(source=source,natural_id=n[0]['id'],tts_id=t['id'],N=natural,T=t['SyncNet'],
                delta=t['SyncNet']-natural,audio_sha256=t['audio_sha256'],video_sha256=t['video_sha256']))
    return sources,draws

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read(ROOT/'per_video_scores.json');sources,draws=pair_rows(rows)
    saved=read(ROOT/'paired_scores.json')['SyncNet']
    assert len(sources)==20 and all(abs(a['delta']-b['delta'])<1e-12 for a,b in zip(sources,saved))
    ranked=sorted(sources,key=lambda x:-x['delta']);top=ranked[:5]
    index={x['id']:x for x in read(PARENT/'video_index.json')}
    for pair in top:
        rs=[r for r in rows if r['source']==pair['source']]
        n=next(r for r in rs if r['arm']=='N');t=max((r for r in rs if r['arm']!='N'),key=lambda r:r['SyncNet'])
        pair.update(natural_id=n['id'],tts_id=t['id'],representative_T=t['SyncNet'],
            representative_delta=t['SyncNet']-n['SyncNet'],natural_path=index[n['id']]['path'],tts_path=index[t['id']]['path'],
            unique_tts=len({(r['audio_sha256'],r['video_sha256']) for r in rs if r['arm']!='N'}))
        for row in [n,t]:
            receipt=read(ROOT/'evidence/generation'/row['id']/'receipt.json')
            assert sha(index[row['id']]['path'])==receipt['raw_video_sha256']
    groups=dict(all20=distribution([x['delta'] for x in sources]),
        same_reference12=distribution([x['delta'] for x in sources if x['strict_reference']]),
        extended_reference8=distribution([x['delta'] for x in sources if not x['strict_reference']]))
    OUT.mkdir(exist_ok=True)
    result=dict(status='PASS',unit='source; mean of three TTS Sync-C minus one natural Sync-C',groups=groups,top5=top,
        logical_tts_comparisons=len(draws),unique_tts_media=len({(x['audio_sha256'],x['video_sha256']) for x in draws}),
        selection='Posthoc largest positive differences for viewing; not an unbiased quality sample. No new inference, audio processing or hypothesis tests.',
        source_sha256={p.name:sha(p) for p in [ROOT/'per_video_scores.json',ROOT/'paired_scores.json',PARENT/'video_index.json']})
    (OUT/'distribution.json').write_text(json.dumps(result,indent=2))
    for name,data in [('source_differences.csv',sources),('tts_draw_differences.csv',sorted(draws,key=lambda x:-x['delta']))]:
        if name=='source_differences.csv':
            data=[{k:r[k] for k in ['source','speaker','strict_reference','N','T','delta']} for r in data]
        with (OUT/name).open('w',newline='') as f:
            fields=sorted(set().union(*(r.keys() for r in data)))
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
    page='<!doctype html><meta charset="utf-8"><title>PC-Talk Sync-C正差最大的5对</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto}section{margin:36px 0}.pair{display:flex;gap:20px}.pair div{width:49%}video{width:100%}</style><h1>PC-Talk：TTS−自然 Sync-C正差最大的5个来源</h1><p>按三条TTS来源内均值排序。左自然，右展示该来源最高TTS；下方列出三条draw。原生AAC播放片，评分用完整原PCM。本页为事后选例，不代表所有视频的质量。</p>'
    for rank,pair in enumerate(top,1):
        page+=f'<section><h2>{rank}. {pair["speaker"]}：平均Δ {pair["delta"]:+.3f}</h2><p>{pair["source"]}；'+('同句参考' if pair['strict_reference'] else '扩展参考')+f'；三条TTS共有{pair["unique_tts"]}种不同媒体。</p><div class="pair">'
        for label,path,score in [('自然',pair['natural_path'],pair['N']),('TTS '+pair['tts_id'].split('_')[-1],pair['tts_path'],pair['representative_T'])]:
            link=os.path.relpath(path,OUT)
            page+=f'<div><p>{label}：Sync-C {score:.3f}</p><video controls preload="none" src="{html.escape(link,quote=True)}"></video></div>'
        page+='</div><p>'
        for d in [d for d in draws if d['source']==pair['source']]:
            path=index[d['tts_id']]['path'];link=os.path.relpath(path,OUT)
            page+=f'<a href="{html.escape(link,quote=True)}">{d["tts_id"]}（{d["T"]:.3f}，Δ{d["delta"]:+.3f}）</a>　'
        page+='</p></section>'
    (OUT/'top5.html').write_text(page)
    d=np.array([x['delta'] for x in sources]);fig,(a,b)=plt.subplots(1,2,figsize=(12,7),layout='constrained',gridspec_kw={'width_ratios':[1,1.3]})
    a.hist(d,bins=np.arange(-2,2.501,.5),color='#4b85b4',edgecolor='white')
    a.axvline(0,color='#333',linestyle='--');a.axvline(d.mean(),color='#c45a1c',label=f'Mean {d.mean():+.3f}')
    a.axvline(np.median(d),color='#357145',linestyle=':',label=f'Median {np.median(d):+.3f}')
    a.set(xlabel='TTS mean - natural Sync-C',ylabel='Sources',title='Distribution across 20 sources',yticks=range(0,9,1));a.legend();a.spines[['top','right']].set_visible(False)
    order=sorted(sources,key=lambda x:x['delta'],reverse=True);v=[p['delta'] for p in order]
    b.barh(np.arange(20),v,color=['#3c81b0' if x>0 else '#cc7b48' for x in v]);b.invert_yaxis()
    b.set_yticks(np.arange(20),[p['speaker']+(' *' if p['strict_reference'] else '') for p in order]);b.axvline(0,color='#333',linewidth=1)
    b.set(xlabel='TTS mean - natural Sync-C',title='All sources, sorted (* primary reference)')
    b.spines[['top','right']].set_visible(False)
    fig.suptitle('PC-Talk | fixed face, RMS -24 dBFS | one source = one observation')
    fig.savefig(OUT/'distribution.png',dpi=180);fig.savefig(OUT/'distribution.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
